"""P-G 教程 3DGS 采集:CARLA 静态场景 360° 环绕 → outputs/3dgs/capture/。

工业口径:3D Gaussian Splatting 训练需多视角图像 + 相机位姿。本脚本:
- 在选定观测点做**360° 环绕**(步进角 = 360/n_cams),采集 RGB + 深度
  (深度用于近端验证;3DGS 训练只用 RGB + 位姿)
- **多俯仰**(--pitches "0,-15,-30"):同一环绕圈跑多组相机俯仰,俯角补顶面/近地
  隐面,提升重建覆盖。落盘按 pitch 分目录 images/p{p}/ + poses_{p}.json
- 相机位姿直接取 CARLA 真值(与 pycolmap 建图对照;未知姿态时用真值初始化)
- 静态场景(无 NPC/车流)——Gaussian 重建要求被摄物不动

实现纪律:
- **单相机逐帧移动**(不批量 spawn 多传感器):CARLA 不允许"无父 actor 的悬空
  sensor",批 spawn 裸 camera 会触发 terminate w/o active exception。逐帧
  set_transform 单挂传感器(attach 到静态 spectator)最稳。
- 采集时用 ego 静态锚点(spectator 不可 attach),故相机直接贴 spectator
  逐帧移动位姿。位姿几何抽成纯函数 autodrivedata/collect_rig.ring_cam_pose。

落盘:
  outputs/3dgs/capture/
    images/p{p}/{i:05d}.png      # 每俯仰一圈独立子目录(i 全局序号)
    depth/p{p}/{i:05d}.npy       # 深度米,验证锚点
    sem/p{p}/{i:05d}.png         # 语义 tag(8 位灰度,仅 --sem)
    inst/p{p}/{i:05d}.png        # 实例 id(16 位灰度,仅 --inst)
    poses_{p}.json               # [{i, loc(x,y,z), rot(pitch,yaw,roll)}]
    pitches.json                 # 本批采集的 pitches 列表(供训练侧枚举)

## ★ 帧同步(2026-10-04 修;不修则新采的数据同样带坏帧)

**症状**(详见 [docs/edit-3dgs-plan.md](../../docs/edit-3dgs-plan.md) §1.5):第 0 帧**位姿正常**
(与第 1 帧差 0.42 m / 4°,符合 6 m 半径 + 4° 步进)但**图像不是那个位姿拍的** ——
与第 1 帧的平均绝对差 **75.4**(正常相邻帧 30.1)、中位深度 **24.6 m** vs 其余 9.4–10.5 m、
`>50 m` 占 **22.6%** 而其余 89 帧**全部 0.0%**。后果是**归档里所有 `psnr_val` 都算在这一帧上**。

**根因**(2026-10-04 直读量出来的,与最初猜的"FIFO 攒陈旧帧"**不是一回事**):
CARLA 的传感器投递比 `world.tick()` **晚定额 2 个仿真帧** —— 逐帧打印
`world.get_snapshot().frame` 与 `Image.frame`,差值恒为 **2**、不累积。
于是"每 tick 取一帧(FIFO 取最旧)"这条口径**整段滞后 2 帧**,而**帧号一张张对得上**、
几路相机也彼此同步(`assert_synced` 不红)。预热那 3 次多出的 1 帧只是把第 0 帧变成
"归位 spectator 那一帧"(§1.5 那个 22.6%),**剩下的 89 帧同样错位**,只是环形每帧才转 4°、
错位在统计量上看不出来。

**修法**:`collect_static_gt.shoot_synced` —— 位姿设好 → tick → 记下目标帧号 →
反复"排空 + 取 ≥ 目标帧的最新帧"直到每路都到齐(**旧帧一律不取**)。这是 `settle` 的形态
**按本采集器改**的结果(单相机 attach 到 spectator、逐帧 `set_transform`);
`assert_synced` 仍然每帧跑(它管的是"几路彼此同不同步",与"是不是当次位姿"是两件事)。

判据(离线)`autodrivedata/gs/frame_sync.py`:逐帧 `>50 m` 占比 / 中位深度 / 相邻帧平均绝对差,
**首帧必须与其余帧同分布**。⚠️ 修好的判据对**已归档的** capture 仍然是红的 —— 那批数据没重采。

用法:
  python -m autodrivedata.sim.collect_3dgs [--center-index 77] [--n-cams 90] [--radius 6] [--pitches "0,-15,-30"]
  python -m autodrivedata.sim.collect_3dgs --sem --inst [--out outputs/3dgs_sync]  # 归属用(阶段 B)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import queue
from typing import cast

import carla
import numpy as np

from autodrivedata.calib.depth_codec import decode_depth
from autodrivedata.calib.probe_calib import decode_instance
from autodrivedata.gt.props import PropBox, classify_prop
from autodrivedata.perception.inst_tags import encode_instance_png
from autodrivedata.perception.sem_tags import encode_tag_png
from autodrivedata.sim.carla_common import (
    CAM_ATTRS,
    clear_generated_actors,
    ground_z_at,
    measure_actor_size_yaw0,
    sync_mode,
)
from autodrivedata.sim.collect_rig import ring_cam_pose
from autodrivedata.sim.collect_static_gt import assert_synced, present_prop_models, shoot_synced
from autodrivedata.sim.collect_surround import tag_from_semantic_image
from autodrivedata.utils.paths import project_path

#: 蓝图后缀 → 落盘时的目检名(`assert_synced` 的报错用它定位是哪一路掉队)。
KIND_LABEL = {"semantic_segmentation": "语义", "instance_segmentation": "实例"}


def kind_names(sem: bool, inst: bool) -> list[str]:
    """`--sem` / `--inst` → 要挂的相机蓝图后缀。**纯值**,便于单测钉"开关 → 挂几路"。"""
    return [k for k, want in (("semantic_segmentation", sem), ("instance_segmentation", inst)) if want]


def make_kind_blueprints(bp_lib, kinds: list[str]) -> dict:
    """给每种 GT 相机建一份蓝图,**属性逐字取 `CAM_ATTRS`**(与 RGB 那路同一份常量)。

    ★ 这是"tag 图与 RGB 图**像素对齐**"这条不变量的**唯一落点**:挂点(同一个
    `carla.Transform()`)与这三项(`image_size_x/y` / `fov`)只要有一处不一致,
    两幅画就不再逐像素对应 —— 而"不齐"在下游只表现为「归属错」,看不出是这里错的。
    抽成函数是为了让单测能拿一个假 `bp_lib` 直接问"你到底给它设了哪些属性"。
    """
    out: dict = {}
    for kind in kinds:
        b = bp_lib.find(f"sensor.camera.{kind}")
        for k, v in CAM_ATTRS.items():
            b.set_attribute(k, v)
        out[kind] = b
    return out


def apply_pose(sensors: list, tf) -> None:
    """把**同一个** transform 施加到所有传感器(路径必须逐字相同,见 `make_kind_blueprints`)。"""
    for s in sensors:
        s.set_transform(tf)


def _depth_img_to_meter(dep: carla.Image) -> np.ndarray:
    """CARLA 深度图 → 深度米 (H,W)(**委托** `depth_codec.decode_depth`,同一口径)。"""
    return decode_depth(dep.raw_data, dep.height, dep.width)


#: 摆位自证容差(m)。与 `collect_static_gt.PLACE_TOL_M` 同值同因:摆偏了**不会抛异常**,
#: 只会让后面每一条判据都偏低,而那与"资产形状对不上"长得一样。
PLACE_TOL_M = 0.05


def spawn_prop_at(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    center: carla.Location,
    offset: tuple[float, float],
    model: str,
) -> tuple[carla.Actor, PropBox]:
    """在**环绕中心 + offset** 处摆一个静态道具 → `(actor, PropBox)`,并读回自证。

    ⚠️ **不能复用 `collect_static_gt.spawn_props`** —— 那个是 **ego 相对**的
    (`PROP_SLOTS` = 前向距离 + 侧别),而本采集器是**绕一个静止中心转**,没有 ego。
    照模式写,别照抄。
    ⚠️ **yaw 恒 0**:`Actor.bounding_box` 对**转过**的 actor 给的是**被剪切**的值
    (红线),而 `prop.json` 里的尺寸取自 `measure_actor_size_yaw0` 的蓝图探针。
    yaw 摆成 0,两者才是同一个口径 —— 摆一个角度只为好看,代价是尺寸记录失去意义。
    """
    dims, offs, brot = measure_actor_size_yaw0(world, bp_lib.find(model))
    x, y = center.x + offset[0], center.y + offset[1]
    gz = ground_z_at(world, x, y, center.z)
    want = carla.Location(x=x, y=y, z=gz)
    a = world.try_spawn_actor(bp_lib.find(model), carla.Transform(want, carla.Rotation(yaw=0.0)))
    if a is None:
        raise SystemExit(
            f"{model} 在环心 {offset} 处 spawn 失败 —— 摆位被占。"
            "**不要静默换个位置**:道具与环绕中心的相对几何是这一整套实验的自变量"
        )
    world.tick()  # 快照 tick 后才刷新,不 tick 读到的是全 0 陈旧位姿
    back = a.get_transform().location
    err = max(abs(back.x - want.x), abs(back.y - want.y))
    if err > PLACE_TOL_M:
        raise SystemExit(f"摆位自证失败:请求 ({want.x:.3f},{want.y:.3f}) 读回 ({back.x:.3f},{back.y:.3f})")
    print(f"[prop] {model} id={a.id} @ ({want.x:.2f},{want.y:.2f},{want.z:.2f}) 残差 {err:.4f} m")
    return a, PropBox(
        type_id=model,
        label=classify_prop(model),
        location=(want.x, want.y, want.z),
        yaw_deg=0.0,
        size=dims,
        box_offset=offs,
        box_rotation_deg=brot,
        # ★ **这个 id 是整条编辑链路的钥匙**:归属(`render_gs`/`attribute_instances`
        #   那一侧)靠它认出"哪些高斯是道具"。不落盘就只剩"猜哪个 id 是道具"。
        instance_id=a.id,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/3dgs")
    ap.add_argument("--center-index", type=int, default=77, help="观测中心取第 N 个 spawn point")
    ap.add_argument("--radius", type=float, default=6.0, help="环绕半径(米)")
    ap.add_argument("--pitches", default="0,-15,-30", help="相机俯仰序列(度,逗号分隔,负=朝下)")
    ap.add_argument("--n-cams", type=int, default=90, help="环绕相机数(步进 = 360/n)")
    ap.add_argument("--frames", type=int, default=90, help="每俯仰采集帧数(=相机数)")
    ap.add_argument(
        "--sem",
        action="store_true",
        help="多挂一路语义相机(同挂点同 fov)→ `sem/p{p}/{i:05d}.png`(8 位灰度,tag 即像素值)。"
        "默认关 —— 关着时产物与 2026-09-18 那版**逐字节一致**",
    )
    ap.add_argument(
        "--inst",
        action="store_true",
        help="多挂一路实例相机(同挂点同 fov)→ `inst/p{p}/{i:05d}.png`(16 位灰度,像素值 = "
        "CARLA actor id)。**阶段 B 的 Gaussian→实例归属就靠它**。默认关",
    )
    ap.add_argument(
        "--props",
        action="store_true",
        help="在**环绕中心**摆一个静态道具 → `capture/prop.json`(含 `instance_id`)。"
        "**编辑实验的 A 侧**:配一次不带它的采集 = B 侧,位姿由 `ring_cam_pose` 保证逐位相同。"
        "默认关 —— 关着时产物与既有 capture 逐字节一致",
    )
    ap.add_argument(
        "--prop-model",
        default="static.prop.warningconstruction",
        help="道具资产(默认取**最宽**的那个:2.11 m。物体越小,A/B 的差越接近重建抖动)",
    )
    ap.add_argument(
        "--prop-offset",
        default="0,0",
        help="道具相对环绕中心的 (x, y) 偏移,米。默认就摆在环心",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    pitches = [float(x) for x in args.pitches.split(",")]

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)

    pts = world.get_map().get_spawn_points()
    center = pts[args.center_index % len(pts)].location
    print(f"[3dgs] 中心点 {center} (spawn {args.center_index}) | 半径 {args.radius} | 俯仰 {pitches}°")

    bp_lib = world.get_blueprint_library()

    # ★ **清场(2026-10-04 补 —— 本采集器原先完全没有这一步)**。
    #   不清的代价不是"多几个 actor",是**残留的道具/遮挡物会让下一轮 A 侧带着
    #   上一轮 B 侧的东西采完** —— 数据照出,只是 A/B 的差凭空小一截(红线第三种形态)。
    #   `--props` 让这条从"潜在风险"变成"必然发生":A 摆道具、B 不摆,不请场 B 就带着 A 的道具。
    n_clear = clear_generated_actors(world)
    print(f"[clean] 清掉 {n_clear} 个本仓生成的 actor(车/行人/控制器 + static.prop.*)")

    prop_actor = None
    prop_box = None
    if args.props:
        models = present_prop_models(bp_lib)
        if args.prop_model not in models:
            raise SystemExit(f"--prop-model {args.prop_model} 本图没有;可选 {models}")
        offset = tuple(float(x) for x in args.prop_offset.split(","))
        if len(offset) != 2:
            raise SystemExit(f"--prop-offset 要 `x,y` 两个数,收到 {args.prop_offset!r}")
        prop_actor, prop_box = spawn_prop_at(world, bp_lib, center, offset, args.prop_model)

    cam_bp = bp_lib.find("sensor.camera.rgb")
    for k, v in CAM_ATTRS.items():
        cam_bp.set_attribute(k, v)
    depth_bp = bp_lib.find("sensor.camera.depth")
    for k, v in CAM_ATTRS.items():
        depth_bp.set_attribute(k, v)
    # 语义 / 实例(**同一份挂载代码**,只差蓝图后缀)。⚠️ 挂点与 fov 必须与 RGB 那路
    # **逐字相同** —— 同 `carla.Transform()`(相对 spectator 的同一个偏移)、同 `CAM_ATTRS`。
    # 不齐的话 tag 图与 RGB 图不再像素对齐,而"不齐"在下游只表现为「归属错」,
    # 看不出是这里错的(同 `collect_surround._spawn_kind` 的纪律)。
    # ⚠️ **不能照抄 `_spawn_kind`**:它 attach 到 `ego` 且遍历 `NUS_CAMERA_RIG` 的 6 路,
    # 与本采集器的单相机 spectator 形态不同 —— 照模式写,别照抄。
    kinds = kind_names(args.sem, args.inst)
    kind_bp: dict[str, carla.ActorBlueprint] = make_kind_blueprints(bp_lib, kinds)

    # 单相机逐帧移动:attach 到 spectator(静态锚点),逐帧 set_transform。
    # 关键坑(已实测):attach 子 actor 的 set_transform 是**相对父 actor**的位姿。
    # 若不把 spectator 先挪到环绕中心,传世界坐标 = 相对 spectator 默认位置偏移,
    # 环绕实际绕在空场地上(曾致整段采集低纹理、PSNR ~10)。必须先归位 spectator。
    spec = world.get_spectator()
    spec.set_transform(carla.Transform(center + carla.Location(0, 0, 1.5), carla.Rotation(0.0, 0.0, 0.0)))
    world.tick()
    cam = cast(carla.Sensor, world.spawn_actor(cam_bp, carla.Transform(), attach_to=spec))
    dep = cast(carla.Sensor, world.spawn_actor(depth_bp, carla.Transform(), attach_to=spec))
    q: queue.Queue = queue.Queue()
    dq: queue.Queue = queue.Queue()
    cam.listen(q.put)
    dep.listen(dq.put)
    extra: dict[str, tuple[carla.Sensor, queue.Queue]] = {}
    for kind, b in kind_bp.items():
        s = cast(carla.Sensor, world.spawn_actor(b, carla.Transform(), attach_to=spec))
        qq: queue.Queue = queue.Queue()
        s.listen(qq.put)
        extra[kind] = (s, qq)

    out = project_path(args.out) / "capture"
    for p in pitches:
        (out / "images" / f"p{int(p)}").mkdir(parents=True, exist_ok=True)
        (out / "depth" / f"p{int(p)}").mkdir(parents=True, exist_ok=True)
        if args.sem:
            (out / "sem" / f"p{int(p)}").mkdir(parents=True, exist_ok=True)
        if args.inst:
            (out / "inst" / f"p{int(p)}").mkdir(parents=True, exist_ok=True)
    (out / "pitches.json").write_text(json.dumps(pitches, indent=1), encoding="utf-8")
    if prop_box is not None:
        assert prop_actor is not None
        # ★ 归属的钥匙:编辑链路靠 `instance_id` 认出"哪些高斯属于这个道具"。
        #   只落 `PropBox` 即可 —— 它的字段(世界位姿 / 实测尺寸 / 盒偏移)与
        #   `gt/props.py` 的读者口径一致,不另造一套。
        (out / "prop.json").write_text(
            json.dumps({"prop": dataclasses.asdict(prop_box)}, indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[prop] 记录 → {out / 'prop.json'}(instance_id={prop_box.instance_id})")

    # ★ 帧同步的**做法**见 `collect_static_gt.shoot_synced`:它是 `settle` 的形态**按本采集器
    #   改**的结果 —— 单相机 attach 到 spectator、逐帧 `set_transform`,而且要多补一条
    #   "等当次位姿那一帧真的落地"。原实现(以及只做 `settle`)实测都会得到**整段序列错位**:
    #   投递比 `world.tick()` 晚定额 2 个仿真帧 ⇒ FIFO 取最旧 = 恒定滞后 2 帧。
    qs = [q, dq, *(qq for _, qq in extra.values())]
    extra_names = list(extra)
    print(f"[sync] 四路队列 {len(qs)} 条;每帧按 `shoot_synced` 等位姿帧落地(实测管线延迟 2 帧)")

    for p in pitches:
        p_poses: list[dict] = []
        for i in range(args.frames):
            x, y, z, yaw_cam = ring_cam_pose(
                center.x, center.y, center.z, args.radius, i, args.n_cams, height=1.5
            )
            tf = carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch=p, yaw=yaw_cam, roll=0.0))
            # 四路**同一个 tf** ⇒ 像素级同画幅(挂点与 fov 也逐字相同,见上面的挂载段)
            apply_pose([cam, dep, *(s for s, _qq in extra.values())], tf)
            # ★ 位姿设好之后再推进,直到**当次位姿那一帧**落地(见 `shoot_synced`)
            frames = shoot_synced(world, qs)
            img: carla.Image = frames[0]
            depth: carla.Image = frames[1]
            got_extra = {k: frames[2 + j] for j, k in enumerate(extra_names)}
            # ★ **同 tick 自证**(见 `assert_synced`):几路 frame 号必须相等,不等当场停。
            #   不判的话症状是"图看着正常、tag 整体偏 1 帧",与"采集器时序错了"长得一样。
            sync_note = assert_synced(
                [("RGB", img), ("深度", depth), *((KIND_LABEL[k], v) for k, v in got_extra.items())]
            )
            tmp = out / f".tmp_{int(p)}_{i}.png"
            img.save_to_disk(str(tmp))
            tmp.rename(out / "images" / f"p{int(p)}" / f"{i:05d}.png")
            np.save(out / "depth" / f"p{int(p)}" / f"{i:05d}.npy", _depth_img_to_meter(depth))
            sem_img = got_extra.get("semantic_segmentation")
            if sem_img is not None:
                # 先 `convert(Raw)` 再取 R 通道(见 `collect_surround.tag_from_semantic_image`)
                (out / "sem" / f"p{int(p)}" / f"{i:05d}.png").write_bytes(
                    encode_tag_png(tag_from_semantic_image(sem_img))
                )
            inst_img = got_extra.get("instance_segmentation")
            if inst_img is not None:
                # `id = G + 256·B`(见 `calib.probe_calib.decode_instance`);落 uint16 无损
                ids = decode_instance(inst_img.raw_data, inst_img.height, inst_img.width)
                (out / "inst" / f"p{int(p)}" / f"{i:05d}.png").write_bytes(encode_instance_png(ids))
            p_poses.append(
                {
                    "i": i,
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "z": round(z, 3),
                    "pitch": p,
                    "yaw": round(yaw_cam, 3),
                    "roll": 0.0,
                }
            )
            if (i + 1) % 20 == 0 or i == args.frames - 1:
                print(
                    f"[cam {i + 1}/{args.frames}] pitch {p}° yaw {yaw_cam:.1f}° @ ({x:.1f}, {y:.1f})"
                    + (f"  {sync_note}" if sync_note else "")
                )
        (out / f"poses_{int(p)}.json").write_text(json.dumps(p_poses, indent=1), encoding="utf-8")
        print(f"[pitch {p}°] done, {args.frames} 相机")

    cam.stop()
    dep.stop()
    cam.destroy()
    dep.destroy()
    for s, _qq in extra.values():
        s.stop()
        s.destroy()
    # ★ 收尾走**与清场同一个谓词**(`--props` 摆下的道具也在这里被收掉)。
    #   ⚠️ 不要"先显式 destroy 道具再让清场重扫" —— 那会给已不存在的 actor 各报一条
    #   `failed to destroy`,而"收尾一堆红字"正是"崩溃掩盖成功"同款误导(函数头注)。
    n_left = clear_generated_actors(world)
    if n_left:
        print(f"[clean] 收尾清掉 {n_left} 个 actor(含道具)")
    chans = "RGB + 深度" + (" + 语义" if args.sem else "") + (" + 实例" if args.inst else "")
    print(f"[done] 3dgs capture: {out.resolve()} ({len(pitches)} 俯仰 × {args.frames} 环绕相机 × {chans})")


if __name__ == "__main__":
    main()

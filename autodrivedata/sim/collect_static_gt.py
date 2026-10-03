"""P2 静态目标/道路特征 GT 采集器(地图查询源,2026-09-09)。

数据源(Plan.md §5.6a 实测裁决):Town10HD_Opt 静态定义全在 OpenDRIVE——
signal/landmark(58×Signal_3Light_Post01 红绿灯 + Sign_Stop/Yield)与
waypoint.lane_marking(车道线实体)。非 actor 无 tag,semantic LiDAR 打不到、
无信号 actor 可用 → 地图查询 API 定案;RoadRunner 挂起至 M4。

输出(每帧,与帧对齐、ego 位姿锚定):
  training/static_gt/{fid}.json   StaticFrame(信号 + 车道线段,世界系)
  training/image_2/{fid}.png      原始相机帧
  training/overlay/{fid}.png      目检叠加图:信号锚点(红) + 车道线段
                                  (白=White 黄=Yellow)投影到图像平面

## `--props`:静态**道具** GT(锥桶/路障,2026-10-01 加;默认关)

上一段的地图查询源**够不着道具**:地图里烘死的道具(`Dynamic` tag 那 6999 px)
根本不是 actor,`get_transform()` 无从查起。所以道具走**另一条路** —— 本采集器
自己摆一组受控道具,按 actor API 出世界系 GT。

产物(仅 `--props` 时):
  training/static_prop_gt/{fid}.json  PropFrame(世界系,`gt/props.PropFrame`)
  training/prop_inst/{fid}.png        **实例分割 id 图**(uint16 PNG,id = G + 256·B)
  training/overlay/{fid}.png          道具 3D 盒投影(与信号/车道线同图)

**为什么落 id 图**:它是判据的 oracle。`perception/prop_eval` 拿它当**渲染轮廓**,
判"投影出来的 GT 框有没有把物体框住",**离线**可跑、不需要 CARLA。没有它,判据只能
信采集器自报的数 —— 那就成了自证。

⚠️ **不进 `label_2`**(见 `gt/props.py` 头注):进去会破 P1 A/B 的帧级配对硬门槛。
本采集器本来也不产 label_2,这条是给后来人留的路标,别顺手"统一"过去。

用法(base env):
  python -m autodrivedata.sim.collect_static_gt [--frames 40] [--out outputs/kitti_static_demo]
  python -m autodrivedata.sim.collect_static_gt --props --frames 40 --out outputs/kitti_static_props
"""

from __future__ import annotations

import argparse
import queue
from dataclasses import replace
from itertools import pairwise
from typing import Protocol, cast

import carla
import numpy as np
from PIL import Image, ImageDraw

from autodrivedata.calib.core import CameraIntrinsics, world_to_img
from autodrivedata.calib.probe_calib import decode_instance
from autodrivedata.gt.core import ActorBox, actor_box_from_prop, box_corners_world
from autodrivedata.gt.props import CameraPose, PropBox, PropFrame, classify_prop, is_prop
from autodrivedata.gt.static_gt import (
    LaneSegment,
    StaticFrame,
    StaticSignal,
    landmark_kind,
    merge_lane_marks,
)
from autodrivedata.perception.inst_tags import encode_instance_png
from autodrivedata.perception.sem_tags import encode_tag_png
from autodrivedata.sim.carla_common import (
    CAM_ATTRS,
    SENSOR_OFFSET,
    Vec3,
    ground_z_at,
    loc,
    measure_actor_size_yaw0,
    rad,
    spawn_ego,
    sync_mode,
)
from autodrivedata.sim.collect_surround import tag_from_semantic_image
from autodrivedata.utils import fonts
from autodrivedata.utils import geometry as g
from autodrivedata.utils.paths import project_path

SPEED = 8.0  # m/s 定速直行(A/B 纪律:起点/轨迹可复现)
LANDMARK_HORIZON = 65.0  # 与 GT max_distance 一致的静态锚点视距
LANE_STEP = 5.0  # 车道线采样步长(m)
LANE_STEPS = 13  # 13×5m = 65m 采样长度
MARK_COLOR = {"White": (255, 255, 255), "Yellow": (220, 190, 60)}
PROP_COLOR = (255, 140, 0)  # 道具 3D 盒投影 = 橙

#: 静态道具资产表。**三个都是 2026-10-01 实测过的**(探针量过尺寸与语义 tag)。
PROP_MODELS: tuple[str, ...] = (
    "static.prop.constructioncone",
    "static.prop.streetbarrier",
    "static.prop.warningconstruction",
)
#: 逐道具的 `(前向距离 m, 侧别 ±1, yaw 偏角 deg)`。
#: ⚠️ **横向不写死米数** —— 由 `spawn_props` 按 `ego 半宽 + 该资产半宽 + 余量` 现算:
#: 写死 2.5 m 对锥桶(宽 0.34)够、对 `warningconstruction`(宽 **2.11 m**)就把它
#: 伸进 ego 路径里,而**撞了照样采得完**,只是那一帧道具不在原位(与"spawn 错了"同形)。
PROP_SLOTS: tuple[tuple[float, int, float], ...] = (
    (14.0, -1, 0.0),
    (22.0, +1, 30.0),
    (30.0, -1, 60.0),
    (38.0, +1, 90.0),
    (46.0, -1, 45.0),
)
#: 道具边缘与 ego 车身之间留的余量(m)。
CLEARANCE_M = 0.8
#: 摆位自证容差(m)。摆完 tick 一次读回,与请求位置差超过它就报错停 ——
#: 摆偏了不会抛异常,只会让**每一条 coverage 都偏低**,那与"资产形状对不上"同形。
PLACE_TOL_M = 0.05
#: 道具 z 与它脚下车道 z 的容差(m)。路面有起伏时浮空 0.2 m = 少挡 0.2 m,
#: 而 GT 框会整体偏上 —— 只看 x/y 的自证**抓不到**(§P-M.7「声明 = 渲染」同一条)。
GROUND_TOL_M = 0.30


def collect_static_frame(
    world: carla.World, ego: carla.Vehicle, frame_id: str
) -> tuple[StaticFrame, list[StaticSignal], list[LaneSegment]]:
    """地图查询一帧静态 GT:附近信号 landmark + 沿本车道车道线段。"""
    m = world.get_map()
    ego_pt = loc(ego.get_transform())

    # 信号:landmark(世界系锚点 + yaw),按视距过滤。
    # 同一物理信号杆会挂在多条 lane 上(重复锚点)→ 按 (name, 位置) 去重;
    # OpenDRIVE 方位累积出现 -360/-540 类值 → 规范化到 [0, 360)
    sigs: list[StaticSignal] = []
    seen: set[tuple[str, float, float]] = set()
    for lm in m.get_all_landmarks():
        p = lm.transform.location
        if np.linalg.norm([p.x - ego_pt[0], p.y - ego_pt[1]]) > LANDMARK_HORIZON:
            continue
        kind = landmark_kind(str(lm.name))
        if kind == "unknown":
            continue  # 未知 landmark 不输出(防把非信号类当 GT)
        key = (str(lm.name), round(float(p.x), 1), round(float(p.y), 1))
        if key in seen:
            continue
        seen.add(key)
        yaw = float(lm.transform.rotation.yaw) % 360.0
        sigs.append(
            StaticSignal(
                landmark_id=str(lm.id),
                kind=kind,
                name=str(lm.name),
                location=(p.x, p.y, p.z),
                yaw_deg=yaw,
            )
        )

    # 车道线:沿 ego 车道向前采样,mark 点落在车道边缘(中心 ± lane_width/2)
    wp = m.get_waypoint(carla.Location(x=ego_pt[0], y=ego_pt[1], z=ego_pt[2]))
    samples: list[dict[str, float | str]] = []
    cur = wp
    for _ in range(LANE_STEPS):
        rot = g.carla_rotation_matrix(rad(cur.transform.rotation))
        right = rot[:, 1]  # CARLA R 矩阵: 列0=前向 x,列1=右向 y
        b = cur.transform.location
        half = cur.lane_width / 2.0
        for side, sign in (("right", +1.0), ("left", -1.0)):
            mark = cur.right_lane_marking if side == "right" else cur.left_lane_marking
            if mark is None:
                continue
            if str(mark.type) == "NONE" or str(mark.color) == "NONE" or float(mark.width) <= 0.0:
                continue  # 无实体标线(路口/默认 xodr 占位 w=0),不输出
            px = b.x + right[0] * half * sign
            py = b.y + right[1] * half * sign
            pz = b.z + right[2] * half * sign
            samples.append(
                {
                    "side": side,
                    "mark_type": str(mark.type),
                    "color": str(mark.color),
                    "width": float(mark.width),
                    "x": float(px),
                    "y": float(py),
                    "z": float(pz),
                }
            )
        nxt = cur.next(LANE_STEP)
        if not nxt:
            break
        cur = nxt[0]

    segs: list[LaneSegment] = []
    for s in merge_lane_marks(samples):
        pts = tuple((float(p[0]), float(p[1]), float(p[2])) for p in s["points"])
        segs.append(
            LaneSegment(
                side=str(s["side"]),
                mark_type=str(s["mark_type"]),
                color=str(s["color"]),
                width=float(s["width"]),
                points=pts,
            )
        )

    frame = StaticFrame(
        frame_id=frame_id,
        ego_location=ego_pt,
        ego_yaw_deg=float(ego.get_transform().rotation.yaw),
        signals=tuple(sigs),
        lane_lines=tuple(segs),
    )
    return frame, sigs, segs


def drain(q: queue.Queue) -> int:
    """丢弃队列里**已积压**的帧,返回丢掉几帧。

    同步模式下每 tick 每队列恰一帧,所以"队列里有多少帧"= "相机挂上之后又 tick 了几次"。
    道具通道在开跑前要 tick 好几次(量尺寸 2 次 + 摆位读回 1 次)⇒ **不 drain 的话主循环
    读到的是"道具还没摆好"的那一帧**,而症状是"前几帧 GT 有框、图里没东西" ——
    被眼熟地误读成"模型没检出来"。丢掉的帧数打出来,别静默。
    """
    n = 0
    while True:
        try:
            q.get_nowait()
            n += 1
        except queue.Empty:
            return n


#: `settle` 的收敛尝试上限。
SETTLE_TRIES = 12


class _Ticker(Protocol):
    """`settle` 需要世界做的**唯一**一件事。写成协议是为了让单测能给一个假世界 ——
    "这条到底依赖什么"因此写在签名上,而不是靠读实现推。"""

    def tick(self) -> None: ...


class _Framed(Protocol):
    """`assert_synced` 需要相机帧做的**唯一**一件事:自报帧号。"""

    frame: int


def settle(world: _Ticker, queues: list[queue.Queue]) -> list[int]:
    """排空 + tick,直到**每个队列恰好剩当 tick 那一帧**;返回最后一次的 qsize。

    ## ★ 为什么"排空一次"不够(2026-10-02 实测的真 bug)

    原实现只 `drain()` 一遍就进主循环。**客户端投递是异步的** —— 那一刻判"队列空"只代表
    *已经到的*取完了,**还在途的**帧会在之后补进来。于是主循环每一轮 `get()` 取到的都是
    陈旧帧,而陈旧量**恒定不变**(每 tick 补一帧、取一帧)⇒ **整段序列恒定滞后**,帧号
    却一张张对得上、图也一张张出得来。

    实测(`kitti_static_props`):`image_2/{f}.png` 与**同一个采集器**写出的
    `prop_inst/{f}.png`、`prop_sem/{f}.png`、`static_prop_gt/{f}.json` **差 1 帧**。
    判据 = **拿固定掩膜跨帧扫 argmax**(掩膜来自 `static_sem/f` 的近场车道线,测
    `image_2/*` 各帧的亮线命中率;同帧比是没有判别力的 —— 同帧恰好也高于基线):

    | 掩膜来自 | 扫 `image_2` 各帧 | 峰值 | 基线 |
    |---|---|---|---|
    | `static_sem/5` | **6** | 0.913 | ~0.49 |
    | `static_sem/7` | **8** | 0.985 | ~0.51 |

    ⚠️ **仪器要选对**:同一批数据我先后用过"锥桶橙色占比"和"包围盒偏移"两种读法,
    两个都**给出过干净但错误的答案** —— 橙色阈值同时命中黄色标线、而锥桶在这种光照下
    渲染得偏灰(`prop_inst` 掩膜内橙占比只有 0.03)。**只有"固定掩膜 + 跨帧 argmax +
    与基线分得开"这条立得住**;修好后同一测量给出 offset 0(`kitti_static_sem_v2`,
    帧 5→5 / 7→7),掩膜逐像素套在锥桶上。

    ⚠️ **受害者是"目检图"和一切拿 `image_2` 配这套 GT 的下游**;`prop_eval` / `static_eval`
    **不受影响**(它们只吃 `prop_inst`/`static_sem`,那两路与 GT 同 tick,已实测亚像素吻合)。

    ⚠️ **机制是候选,未证实**:RGB 相机在道具摆位期间积压了 8 帧(实例/语义相机是刚建的、
    积压 0),排空后仍有在途帧补进来 ⇒ 恒定滞后。**没验证的机制不写成因** —— 这里只
    按"排空 + tick 到稳态"处置,并把结果**报出来**。
    """
    for _ in range(SETTLE_TRIES):
        for q in queues:
            drain(q)
        world.tick()
        got = [q.qsize() for q in queues]
        if got == [1] * len(queues):
            for q in queues:
                drain(q)
            return got
    return [q.qsize() for q in queues]


def assert_synced(images: list[tuple[str, _Framed | None]]) -> str:
    """★ **同 tick 自证**:几路相机这一轮的 `frame` 号必须相等。返回一行人读的摘要。

    这是红线那条「**两路本应逐像素相同的东西比一比**」在这里的形态 —— 比的是帧号。
    只在**全都拿到**时判:某一路本轮没有(没开那个开关)就不管它。
    `frame` 属性是 CARLA 的仿真帧号,**帧号对不上**不报错的症状是"图看着正常、框整体偏",
    而那与"标定错了"长得一样。
    """
    got = [(n, int(im.frame)) for n, im in images if im is not None]
    if len(got) < 2:
        return ""
    frames = {f for _, f in got}
    if len(frames) != 1:
        raise SystemExit(
            "相机不同帧 —— " + ", ".join(f"{n}={f}" for n, f in got) + "。"
            "「每 tick 每队列恰一帧」这条前提没成立,继续采会得到**恒定滞后**的序列"
            "(实测过 1 帧),而帧号一张张对得上、图一张张出得来。"
        )
    return f"frame={got[0][1]}(" + "/".join(n for n, _ in got) + ")"


def present_prop_models(bp_lib: carla.BlueprintLibrary) -> list[str]:
    """本图**真的有的**道具资产。

    「蓝图不存在」与「spawn 失败」是两件事,这里先把前者分掉 —— 合并成一个"未测"
    会把"本图根本没这个资产"藏起来,而两者的处置完全不同(换资产 vs 挪位置)。
    一个都没有就直接停:静默出一份**空的道具 GT**是这里最坏的失败态(它看着像
    "这一帧没有道具",而不像"采集器没工作")。
    """
    out: list[str] = []
    for m in PROP_MODELS:
        if bp_lib.find(m) is not None:
            out.append(m)
        else:
            print(f"  [warn] 本图无蓝图 {m} —— 该资产不参与本次采集")
    if not out:
        raise SystemExit(f"本图没有任何一个道具资产 {PROP_MODELS} —— 无法采静态道具 GT")
    return out


def spawn_props(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    ego_t: carla.Transform,
    ego_half_w: float,
    models: list[str],
    sizes: dict[str, tuple[Vec3, Vec3, Vec3]],
) -> list[tuple[carla.Actor, PropBox]]:
    """按 `PROP_SLOTS` 摆道具 → `(actor, 世界系记录)` 表,并**读回自证摆位**。

    自证为什么必要:摆位是一串三角函数,**错了不会报错** —— 只是"GT 框偏了一点",
    而那与"资产形状对不上"在 coverage 上长得一模一样。所以摆完 tick 一次读回,
    逐条比**三个量**:平面位置、离地高度、以及落点确实在车道上。

    横向由 `ego 半宽 + 该资产半宽 + CLEARANCE_M` 现算 —— 见 `PROP_SLOTS` 头注。
    """
    placed: list[tuple[carla.Actor, PropBox]] = []
    for i, (d, side, dyaw) in enumerate(PROP_SLOTS):
        model = models[i % len(models)]
        dims, offs, brot = sizes[model]
        lat = side * (ego_half_w + dims[1] / 2.0 + CLEARANCE_M)
        bp = bp_lib.find(model)
        w = ego_t.transform(carla.Location(x=d, y=lat, z=0.0))
        gz = ground_z_at(world, w.x, w.y, ego_t.location.z)
        want = carla.Location(x=w.x, y=w.y, z=gz)
        yaw = float(ego_t.rotation.yaw) + dyaw
        a = world.try_spawn_actor(bp, carla.Transform(want, carla.Rotation(yaw=yaw)))
        if a is None:
            raise SystemExit(f"{model} spawn 失败 @ ({w.x:.1f}, {w.y:.1f}) —— 摆位被占,停")
        placed.append(
            (
                a,
                PropBox(
                    type_id=model,
                    label=classify_prop(model),
                    location=(want.x, want.y, want.z),
                    yaw_deg=yaw % 360.0,
                    size=dims,
                    box_offset=offs,
                    box_rotation_deg=brot,
                    instance_id=a.id,  # ★ 判据靠它把记录对到 id 图里的渲染轮廓上
                ),
            )
        )
        print(f"  [prop] {model:<36} id={a.id:<6} @ 前 {d:.0f} m / 横 {lat:+.2f} m / yaw {yaw % 360:.0f}°")

    world.tick()  # ★ 读回必须在 tick 之后(快照 tick 后才刷新)
    for a, rec in placed:
        got = a.get_transform().location
        err = float(np.linalg.norm([got.x - rec.location[0], got.y - rec.location[1]]))
        if err > PLACE_TOL_M:
            raise SystemExit(
                f"{rec.type_id} 摆位与请求差 {err:.3f} m > 容差 {PLACE_TOL_M} —— 摆位数学有问题,停"
            )
        # ★ 离地自证:`ground_z_at` 在路肩外会**静默退回 ego 高度**,而那只让框整体偏上
        #   (x/y 判据全绿)。所以另问一次地图:这个点脚下有没有车道、z 差多少。
        try:
            wp = world.get_map().get_waypoint(carla.Location(*rec.location))
        except RuntimeError:
            raise SystemExit(f"{rec.type_id} @ {rec.location} 不在任何车道上 —— 摆到路肩了,停") from None
        dz = abs(float(wp.transform.location.z) - rec.location[2])
        if dz > GROUND_TOL_M:
            raise SystemExit(
                f"{rec.type_id} 离地 {dz:.3f} m > 容差 {GROUND_TOL_M} —— "
                "地面高度取错或道具悬空,GT 框会整体偏上,停"
            )
    return placed


def draw_props(
    img: Image.Image,
    boxes: list[ActorBox],
    cam_loc: tuple[float, float, float],
    cam_rot: tuple[float, float, float],
    k: CameraIntrinsics,
) -> Image.Image:
    """道具 3D 盒 → 目检叠加图(画 12 条棱,不是画外接矩形)。

    画**棱**而不是画 `box_to_gt_line` 那个 2D 外接框:外接框看不出"盒子是不是
    真的套在物体上、朝向对不对",而这正是本通道最怕的错(尺寸被剪切/朝向错 90°
    都会得到一个**看着挺像**的矩形)。
    """
    d = ImageDraw.Draw(img)
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # 底面
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # 顶面
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # 立柱
    )
    for box in boxes:
        uv = [world_to_img(tuple(p), cam_loc, cam_rot, k) for p in box_corners_world(box)]
        for i, j in edges:
            if uv[i] is None or uv[j] is None:
                continue
            d.line([uv[i], uv[j]], fill=PROP_COLOR, width=2)  # type: ignore[list-item]
        base = next((p for p in uv if p is not None), None)
        if base is not None:
            fonts.draw_text(
                d, (base[0] + 8, base[1] - 14), box.type_id.split(".")[-1], size=14, fill=PROP_COLOR
            )
    return img


def draw_overlay(
    img: Image.Image,
    sigs: list[StaticSignal],
    segs: list[LaneSegment],
    cam_loc: tuple[float, float, float],
    cam_rot: tuple[float, float, float],
    k: CameraIntrinsics,
) -> Image.Image:
    """静态 GT → 目检叠加图(红点=信号锚点,彩线=车道线段)。"""
    d = ImageDraw.Draw(img)
    for seg in segs:
        col = MARK_COLOR.get(seg.color, (170, 170, 170))
        pts_2d: list[tuple[float, float]] = []
        for p in seg.points:
            uv = world_to_img((p[0], p[1], p[2]), cam_loc, cam_rot, k)
            if uv is not None:
                pts_2d.append(uv)
        if len(pts_2d) >= 2:
            for a, b2 in pairwise(pts_2d):
                d.line([a, b2], fill=col, width=3)
        for z in pts_2d:
            d.ellipse([z[0] - 2, z[1] - 2, z[0] + 2, z[1] + 2], fill=col)
    for s in sigs:
        uv = world_to_img(s.location, cam_loc, cam_rot, k)
        if uv is None:
            continue
        x, y = uv
        d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(255, 40, 40), width=3)
        d.line([x - 11, y, x + 11, y], fill=(255, 40, 40), width=2)
        d.line([x, y - 11, x, y + 11], fill=(255, 40, 40), width=2)
        fonts.draw_text(d, (x + 10, y - 16), s.kind, size=14, fill=(255, 40, 40))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--out", default="outputs/kitti_static_demo")
    ap.add_argument(
        "--props",
        action="store_true",
        help="摆一组受控静态道具(锥桶/路障)并出 `static_prop_gt/` + 实例分割 id 图。"
        "默认关 —— 关着时产物与 2026-09-09 那版**逐字节一致**(道具通道是新增,不改既有口径)",
    )
    ap.add_argument(
        "--sem",
        action="store_true",
        help="多挂一路语义相机(同挂点同 fov)→ `training/static_sem/` + `static_gt` 里落"
        "相机位姿,给 `perception/static_eval` 当 oracle。默认关 —— 关着时产物与"
        "2026-10-02 那版**逐字节一致**(多一路通道,不改既有口径)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)

    # 清场 + 锚定 pt0(视角固定 + 轨迹可复现,同 collect_ab_route)
    # 2026-09-09 教训:yaw 硬编码 0 在 Town10HD_Opt 恰好成立(pts[0] 固有
    # yaw=0.16°),Town13 pts[0] 固有 yaw=125.9° 时车道线采样沿 lane 方向
    # 走到车后、overlay 全空 → 改用 spawn point 固有 rotation(地图作者设定
    # 的沿车道朝向),各图通用
    # ⚠️ 道具也要清(`static.prop.*` 不在 vehicle/walker/controller 三类里)。
    #    残留的墙/锥会让下一轮带着上一轮的道具开跑 —— 而**数据照出,只是 GT 与画面对不上**。
    #    谓词取 `props.is_prop`(覆盖全部 static.prop.*,不写死我们自己摆的那三个)。
    for a in world.get_actors():
        if a.type_id.startswith(("vehicle", "walker", "controller")) or is_prop(a.type_id):
            a.destroy()
    for _ in range(3):
        world.tick()
    ego = spawn_ego(world)
    ego.set_autopilot(False)
    pts = world.get_map().get_spawn_points()
    ego.set_transform(carla.Transform(pts[0].location, pts[0].rotation))
    world.tick()
    here = ego.get_location()
    if here.distance(pts[0].location) > 1.0:
        raise RuntimeError(f"ego 未能锚定 pts[0]: 落在 {here}")
    yaw_deg = float(ego.get_transform().rotation.yaw)
    print(f"[ego] 锚定 pts[0] @ ({here.x:.1f}, {here.y:.1f}) yaw={yaw_deg:.1f}°(沿车道)")

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    for t, v in CAM_ATTRS.items():
        cam_bp.set_attribute(t, v)
    camera = cast(carla.Sensor, world.spawn_actor(cam_bp, SENSOR_OFFSET, attach_to=ego))
    q: queue.Queue = queue.Queue()
    camera.listen(q.put)
    k = CameraIntrinsics(
        width=int(CAM_ATTRS["image_size_x"]),
        height=int(CAM_ATTRS["image_size_y"]),
        fov_h_deg=float(CAM_ATTRS["fov"]),
    )

    out = project_path(args.out)
    (out / "training/static_gt").mkdir(parents=True, exist_ok=True)
    (out / "training/image_2").mkdir(parents=True, exist_ok=True)
    (out / "training/overlay").mkdir(parents=True, exist_ok=True)

    # ---- 静态道具通道(仅 --props)--------------------------------------------
    prop_records: list[PropBox] = []
    inst_cam: carla.Sensor | None = None
    sem_cam: carla.Sensor | None = None
    inst_q: queue.Queue = queue.Queue()
    sem_q: queue.Queue = queue.Queue()
    if args.sem:
        # ★ 静态 GT 的判据通道:同挂点同 fov 的语义相机 → `static_sem/{fid}.png`。
        #   判据(perception/static_eval)拿它当 oracle,问"xodr 说的信号/车道线位置,
        #   渲染里是不是真有对应的东西" —— **离线可跑,不需要 CARLA**(同 --props 那条)。
        # ⚠️ **拒绝对已有 root 追加**:两轮产物混在一个 root 里,症状是判据命中率莫名偏低
        #   (GT 来自这一轮、图来自那一轮),而它与"xodr 与渲染不符"长得一样。
        d = out / "training" / "static_sem"
        stale = list(d.glob("*.png")) if d.is_dir() else []
        if stale:
            raise SystemExit(
                f"{d} 里已有 {len(stale)} 张图 —— 拒绝覆盖(两轮混在一起会让判据把"
                f"'GT 与 oracle 配错帧'读成'xodr 与渲染不符')。换 --out,或先删掉 {d}"
            )
        for sub in ("static_sem",):
            (out / "training" / sub).mkdir(parents=True, exist_ok=True)

    if args.props:
        # ★ **不许往已经有道具 GT 的 root 里写**。写进去不会报错,只会让上一轮的 json
        #   与这一轮的 id 图**混在同一个 root 里**,而症状是判据报"某几个实例一次都没露面"
        #   —— 那是"GT 与 oracle 配错帧",不是"采集没采到"。判据第一次跑就是这么抓到
        #   本 smoke 目录被覆盖了两轮的(`prop_eval` 的 `never_seen` 那条警告)。
        #   口径同 `gt/refilter.py`:拒绝覆盖,让人自己选新目录或先删。
        for sub in ("static_prop_gt", "prop_inst", "prop_sem"):
            d = out / "training" / sub
            stale = list(d.glob("*.json")) + list(d.glob("*.png")) if d.is_dir() else []
            if stale:
                raise SystemExit(
                    f"{d} 里已有 {len(stale)} 个文件 —— 拒绝覆盖(两轮产物混在一起会让判据"
                    f"误报'实例没露面')。换 --out,或先删掉 {out}/training/{{static_prop_gt,prop_inst}}"
                )
        (out / "training/static_prop_gt").mkdir(parents=True, exist_ok=True)
        (out / "training/prop_inst").mkdir(parents=True, exist_ok=True)
        (out / "training/prop_sem").mkdir(parents=True, exist_ok=True)
        models = present_prop_models(bp_lib)
        # ★ 尺寸只认 **yaw=0 探针**那一读(转过之后读回是被剪切的错值,见 carla_common)
        sizes = {m: measure_actor_size_yaw0(world, bp_lib.find(m)) for m in models}
        for m, (dims, _offs, _rot) in sizes.items():
            print(f"[prop] {m} yaw=0 尺寸 {dims[0]:.4f} × {dims[1]:.4f} × {dims[2]:.4f} m")
        # ego 半宽也走 yaw=0 探针,不读 `ego.bounding_box` —— 转过之后那一读是被剪切的
        # 错值,而 Town13 的 pts[0] 固有 yaw=125.9°(Town10HD_Opt 恰好 0.16° 才不出事)
        ego_half_w = measure_actor_size_yaw0(world, bp_lib.find("vehicle.audi.a2"))[0][1] / 2.0
        print(f"[prop] ego 半宽(yaw=0 探针)= {ego_half_w:.3f} m")
        placed = spawn_props(world, bp_lib, ego.get_transform(), ego_half_w, models, sizes)
        prop_records = [r for _, r in placed]
        print(f"[prop] 摆位自证通过 {len(placed)}/{len(PROP_SLOTS)}(容差 {PLACE_TOL_M} m)")

        # **实例 id 图**是几何判据的 oracle(道具轮廓)。
        inst_bp = bp_lib.find("sensor.camera.instance_segmentation")
        for t, v in CAM_ATTRS.items():
            inst_bp.set_attribute(t, v)
        inst_cam = cast(carla.Sensor, world.spawn_actor(inst_bp, SENSOR_OFFSET, attach_to=ego))
        inst_cam.listen(inst_q.put)

    # ---- 语义 tag 相机(`--props` 与 `--sem` **共用一台**)---------------------------
    # 两个消费者、两个落点:`--props` → `prop_sem/`(道具类别判据),"`--sem`" → `static_sem/`
    # (信号/车道线判据)。**同挂点同 fov** ⇒ 同时开时两份逐字节相同;各留一个落点是为了
    # 让每条判据的输入目录无歧义(`prop_eval` 已经在读 `prop_sem/`,不动它)。
    if args.props or args.sem:
        sem_bp = bp_lib.find("sensor.camera.semantic_segmentation")
        for t, v in CAM_ATTRS.items():
            sem_bp.set_attribute(t, v)
        sem_cam = cast(carla.Sensor, world.spawn_actor(sem_bp, SENSOR_OFFSET, attach_to=ego))
        sem_cam.listen(sem_q.put)
        # ★ 上面量尺寸/摆位自证又 tick 了若干次,各队列都已积压 —— 排空,否则主循环前几帧
        #   读到的是"道具还没摆好"的画面(见 `drain` 头注)。
        # ⚠️ **一个都不能漏**:漏掉的那一路不是"少几帧",是**整体滞后 N 帧**(FIFO 取最旧)。
        # ★ 排空 → tick,直到**每个队列恰好剩当 tick 那一帧**(见 `settle` 头注:只排空一遍
        #   会留下在途帧 ⇒ 整段序列恒定滞后,实测 1 帧)。再排空一次,主循环就从干净态起步。
        qs = [
            qq
            for qq in (q, inst_q, sem_q)
            if not (qq is inst_q and inst_cam is None) and not (qq is sem_q and sem_cam is None)
        ]
        got = settle(world, qs)
        print(f"[sync] 稳态自证:排空后每 tick 各队列 {got}(期望全 1)")
        if got != [1] * len(qs):
            print(f"[sync] ⚠️ 未收敛到 [1]*{len(qs)} —— 继续采可能得到恒定滞后的序列")

    fwd = ego.get_transform().get_forward_vector()
    fwd_v = carla.Vector3D(x=fwd.x * SPEED, y=fwd.y * SPEED, z=0.0)
    try:
        for i in range(args.frames):
            ego.set_target_velocity(fwd_v)
            world.tick()
            image: carla.Image = q.get(timeout=10)
            # ★ 实例相机**每 tick 必须抽干** —— 攒着不取会让队列积压陈旧帧,而
            #   "掩膜恒 0"在下游表现为"这一帧没道具",不是"采集坏了"(§P-M.7 判据⑥同款坑)
            inst_image: carla.Image | None = inst_q.get(timeout=10) if inst_cam is not None else None
            sem_image: carla.Image | None = sem_q.get(timeout=10) if sem_cam is not None else None
            # ★ **同 tick 自证**(见 `assert_synced`):几路 frame 号必须相等,不等当场停。
            #   不判的话症状是"图看着正常、框整体偏 1 帧",与"标定错了"长得一样。
            sync_note = assert_synced([("RGB", image), ("实例", inst_image), ("语义", sem_image)])
            cam_t = camera.get_transform()
            cam_loc, cam_rot = loc(cam_t), rad(cam_t.rotation)
            # ★ 落**实读**的相机位姿与内参:判据住在 `perception/`(层规则禁 carla),
            #   拿不到 CAM_ATTRS/SENSOR_OFFSET;落进文件才让判据离线复现投影链。
            #   两条链(道具 / 静态)**共用这一份** —— 各拼一份迟早漂,而漂了的表现是
            #   "某个通道整体偏",不像口径错。
            cam_pose = CameraPose(
                location=cam_loc,
                rotation_deg=cast(tuple[float, float, float], tuple(np.degrees(v) for v in cam_rot)),
                width=int(CAM_ATTRS["image_size_x"]),
                height=int(CAM_ATTRS["image_size_y"]),
                fov_deg=float(CAM_ATTRS["fov"]),
            )
            frame, sigs, segs = collect_static_frame(world, ego, f"{i:06d}")

            fid = f"{i:06d}"
            if args.sem:
                # 静态 GT 的相机位姿随帧落盘(可选字段,见 `StaticFrame.camera` 头注)
                frame = replace(frame, camera=cam_pose)
            (out / "training/static_gt" / f"{fid}.json").write_text(frame.to_json())

            boxes: list[ActorBox] = []
            if inst_image is not None:
                pf = PropFrame(
                    frame_id=fid,
                    ego_location=loc(ego.get_transform()),
                    ego_yaw_deg=float(ego.get_transform().rotation.yaw),
                    props=tuple(prop_records),
                    # ★ 落**实读**的相机位姿与内参:判据住在 `perception/`(层规则禁 carla),
                    #   拿不到 CAM_ATTRS/SENSOR_OFFSET;落进文件才让判据离线可复现投影链。
                    camera=cam_pose,
                )
                (out / "training/static_prop_gt" / f"{fid}.json").write_text(pf.to_json())
                # id 图落 uint16 PNG:`id = G + 256·B` 上界恰是 65535,uint16 无损承载。
                # 不落 BGRA 原图(每帧 4.5 MB,且判据还得自己解码) —— 落的是**已解码的 id**,
                # 判据直接读,不必再引 `decode_instance`(那会多一处口径)。
                ids = decode_instance(inst_image.raw_data, inst_image.height, inst_image.width)
                if ids.max() > 65535:
                    raise SystemExit(f"实例 id 上界 {ids.max()} > 65535,uint16 装不下")
                # 走共享编解码(越界即抛,不截断)—— 与 `--inst` 那条链同一份实现
                (out / "training/prop_inst" / f"{fid}.png").write_bytes(encode_instance_png(ids))
                boxes = [actor_box_from_prop(p) for p in prop_records]

            if sem_image is not None and sem_cam is not None:
                # 一台相机、两个落点(见上方"语义 tag 相机"那段)。同时开时两份逐字节相同。
                tag_png = encode_tag_png(tag_from_semantic_image(sem_image))
                if args.props:
                    (out / "training/prop_sem" / f"{fid}.png").write_bytes(tag_png)
                if args.sem:
                    (out / "training/static_sem" / f"{fid}.png").write_bytes(tag_png)

            png = out / "training/image_2" / f"{fid}.png"
            image.save_to_disk(str(png))
            img = Image.open(png).convert("RGB")
            draw_overlay(img, sigs, segs, cam_loc, cam_rot, k)
            if boxes:
                draw_props(img, boxes, cam_loc, cam_rot, k)
            img.save(out / "training/overlay" / f"{fid}.png")

            if (i + 1) % 10 == 0 or i == args.frames - 1:
                npts = sum(len(s.points) for s in segs)
                kinds = sorted({s.kind for s in sigs})
                extra = f" 道具 {len(boxes)}" if inst_image is not None else ""
                print(
                    f"[frame {i + 1}/{args.frames}] 信号 {len(sigs)}（{kinds}）"
                    f" 车道线段 {len(segs)} ({npts} 点){extra}{f'  {sync_note}' if sync_note else ''}"
                )
    finally:
        camera.stop()
        camera.destroy()
        for cam in (inst_cam, sem_cam):
            if cam is not None:
                cam.stop()
                cam.destroy()
        # ★ **只扫一遍**:先显式销毁 `props_actors` 再让清场循环重扫一遍,会让客户端对
        #   已经不存在的 actor 各报一条 `failed to destroy ... not found` —— 数据没错,
        #   但"收尾一堆红字"正是 `tools/carla_server.sh` 头注记的"崩溃掩盖成功"同款误导。
        #   单遍扫的代价是依赖 `get_actors()` 快照是新的,所以先 tick 一次(同 `sync_mode`
        #   那条:`spawn 完立刻读` 会拿到不含新 actor 的陈旧快照,清场会静默漏清)。
        world.tick()
        for a in world.get_actors():
            if a.type_id.startswith(("vehicle", "walker", "controller")) or is_prop(a.type_id):
                a.destroy()
    print(f"[done] static GT root: {out.resolve()} ({args.frames} frames)")


if __name__ == "__main__":
    main()

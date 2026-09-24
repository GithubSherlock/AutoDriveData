"""B1:环视 6 相机采集(nuScenes 布局)→ 图像 + 内外参 + 逐帧 ego 位姿。

MapTR 端到端训练的输入侧:6 视角图像 + sensor2ego 外参 + 相机内参 + ego2global
位姿(与 MapTRv2 `nuscenes_converter` 的 cams/ego_pose 字段同构)。B2 组装器
消费本输出拼 MapTR 训练格式,地图 GT 来自 A 阶段矢量库。

采集纪律与 collect_drive 相同(同步模式/预热/清场/场景天气档);**不改动
A/B 采集器**(P1 复现性红线)。NPC 布置复用 collect_drive 的既有函数。

nuScenes 相机布局:**直接取官方 calibrated_sensor**(6DoF 四元数 + 平移),
由 [autodrivedata/camera_rig.py](../autodrivedata/camera_rig.py) 转成 CARLA 口径
(平移 y 翻号、姿态走 `nus_camera_rotation_to_carla`)。采集器不再自己维护角度表。

⚠️ **历史 bug(2026-09-22 修)**:此前 `SURROUND_CAMS` 把官方**方位角**原样抄成正数,
漏了 CARLA↔nuScenes 的 y 符号翻转(`yaw_carla = −az_nus`)⇒ 四个侧/后相机左右镜像
(FRONT_LEFT 差 110.3°、BACK_LEFT 差 217.2°)。前/后相机因近自逆而"看起来对",
所以长期没暴露。同时 pitch/roll 被硬编码 0(官方实测 |pitch| 最大 0.96°)。
镜像表见 camera_rig 模块头注。

落盘:
  outputs/surround_<scene>/cam_front/000000.png ...(6 视角)
  calib.json        — 每相机 sensor2ego + intrinsic(3x3)
  ego_pose.json     — 逐帧 ego2global(CARLA 世界系)

用法:
  python bin/collect_surround.py --frames 100 [--scene day_clear] [--npc-vehicles 15]
  python bin/collect_surround.py --frames 400 --map Town13   # 多图扩数据:运行时切图
  python bin/collect_surround.py --spawn-index 88 --stride 5 --frames 100  # 指定起点 + 0.5s/帧

多图切图(§5.14 Phase 2):`--map` 用 `client.load_world` 运行时切换(默认不动当前图,
零副作用;每次切换 ~2 分钟加载)。**已采集数据的图标记**:default_map 写进 calib.json
顶层 `"map"` 键,供 assemble/merge 溯源(旧产物无此键 = Town10HD_Opt)。

── 画幅与内参口径(2026-09-23,§P-M.11 冻结;改这里前先读 §P-M.7)──────────────
本采集器**历史上是第三处「声明 ≠ 渲染」**:六路共用一个 `cam_bp` 的 `fov=90`,K 也是
`calib_from_fov(w, h, 90)` 一表六用,而官方逐通道水平 FoV 是 **64.31–64.96°×5 + 89.34°**
—— 渲染视野与落盘口径差 25°。现改为**逐相机蓝图 + 逐通道 fov/K**,与 `collect_nus.py`
(§P-M.7)同一条不变量:**渲染与声明由同一份常量导出**。

- **画幅 1600×900**(nuScenes 官方),不再沿用 KITTI 口径的 1242×375;
- **fov** ← `export.nuscenes.NUS_CAMERA_FOV[cam]`(由官方逐通道 fx 反推);
- **K** ← `calib_from_fov(1600, 900, 该通道 fov)`,即 **fx 取官方值、主点取 corner
  `(w−1)/2`** —— 与 `wide` rig 的 `_wide_intrinsics` 同构造。

⚠️ **为什么不直接落官方 K 的 cx(792–829)**:官方主点是**真实相机的装配公差**,而我们的图
是 **CARLA 渲染栅格**,其光栅中心按 §P-M 的 corner 裁决恒为 `(w−1)/2 = 799.5`。若把官方
cx 写进**我方投影链**(GKT / mapviz / eval)消费的 K,world→image 就会**逐通道不一致地偏
7–27 px**(CAM_FRONT_LEFT 最大),等价 0.05–0.2 m 的 BEV 采样偏置 —— 又变成"声明 ≠ 渲染"。
`collect_nus.py` 落官方 cx 是因为**消费方是 devkit / auto3dlabel**(按官方口径解释);本采集器
的消费方是**我们自己的投影代码**,故取 corner。**两处口径不同是角色不同,不是不一致,别来统一**。

⚠️ **绝不改 `carla_common.CAM_ATTRS`**(1242×375/fov90):它被 KITTI 线、P1 A/B 线、静态 GT、
灯态、`probe_calib`、studio 等十余处引用,**动它就是动 P1 复现性红线**。本表是独立常量。
"""

from __future__ import annotations

import argparse
import json
import queue
import time
from typing import Any, cast

import carla
from carla_common import loc, spawn_ego, spawn_ego_at, sync_mode
from collect_drive import spawn_route_walkers, spawn_traffic

from autodrivedata.camera_rig import NUS_CAMERA_RIG, NUS_CAMERA_YAW
from autodrivedata.export.nuscenes import NUS_CAMERA_FOV, NUS_CAMERA_HEIGHT, NUS_CAMERA_WIDTH
from autodrivedata.mapviz import calib_from_fov
from autodrivedata.paths import project_path
from autodrivedata.scenarios import SCENES, merged_weather

# 相机名 → 相对 ego 的 yaw(度)。**别名**,真值在 `autodrivedata/camera_rig.py`
# (`NUS_CAMERA_RIG` 的完整 (平移, (pitch,yaw,roll)));本表只用于**遍历相机名的顺序**
# 与"只关心偏航"的零散打印 —— 挂载/落盘一律走 `NUS_CAMERA_RIG`(含 pitch/roll)。
SURROUND_CAMS: dict[str, float] = dict(NUS_CAMERA_YAW)

# 本采集器专用画幅(nuScenes 官方 1600×900)。**不是** carla_common.CAM_ATTRS(1242×375,
# 那条是 KITTI/P1 线,勿动,见模块头注)。
SURROUND_CAM_ATTRS = {"image_size_x": str(NUS_CAMERA_WIDTH), "image_size_y": str(NUS_CAMERA_HEIGHT)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="输出根目录(默认 outputs/surround_<scene 或 drive>)")
    ap.add_argument("--scene", default=None, choices=sorted(SCENES), help="corner case 场景档(天气覆写)")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--npc-vehicles", type=int, default=15)
    ap.add_argument("--npc-walkers", type=int, default=6)
    ap.add_argument("--route-walkers", type=int, default=6, help="沿 ego 初始朝向布置的行人数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument(
        "--map",
        default=None,
        help="目标地图(如 Town13/Town15;None = 当前服务器默认图)。运行时 load_world 切图,供多图扩数据",
    )
    ap.add_argument(
        "--spawn-index",
        type=int,
        default=None,
        help="固定用第 N 个 spawn point 出生(多段采集用);None = 沿用 spawn_ego 的首个空位(旧行为)",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="每 N 个 tick 存一帧(1 = 旧行为)。5 ⇒ 0.5s/帧 = nuScenes 关键帧率(2Hz)",
    )
    args = ap.parse_args()
    if args.stride < 1:
        ap.error("--stride 必须 ≥ 1")

    scene = SCENES[args.scene] if args.scene else None
    if scene is not None:
        for key, n in scene.traffic.items():
            setattr(args, key, n)
    if args.out is None:
        args.out = f"outputs/surround_{scene.name}" if scene else "outputs/surround_drive"

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    if args.map is not None:
        print(f"[map] load_world {args.map}(运行时切图,~2min)...")
        client.load_world(args.map)
        client.set_timeout(60.0)
    world = client.get_world()
    default_map = args.map or world.get_map().name
    sync_mode(world)

    if scene is not None:
        world.set_weather(carla.WeatherParameters(**merged_weather(scene)))
        print(f"[scene] {scene.name} [{scene.group}] — 覆写 {sorted(scene.weather)}")

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    ego = spawn_ego_at(world, args.spawn_index) if args.spawn_index is not None else spawn_ego(world)
    ego.set_autopilot(True, tm.get_port())
    tm.vehicle_percentage_speed_difference(ego, 30.0)
    # 采集多样性:红绿灯等待在环视数据里是重复帧(250+ 帧原地,占比拉满),
    # 训练多样性被稀释——采集侧按百分比忽略红灯(不影响 TM 其他车与车道保持)
    tm.ignore_lights_percentage(ego, 100.0)
    print("[ego] autopilot on (TM 8000, 70% speed, 忽略红绿灯)")

    ego_t = ego.get_transform()
    spawn_traffic(world, tm, args.npc_vehicles, args.npc_walkers, args.seed)
    spawn_route_walkers(world, ego_t, args.route_walkers)

    bp_lib = world.get_blueprint_library()

    cams: dict[str, carla.Sensor] = {}
    qs: dict[str, queue.Queue] = {}
    for name, (mount, rot) in NUS_CAMERA_RIG.items():
        x, y, z = mount
        tf = carla.Location(x=x, y=y, z=z)
        # 逐相机蓝图:每路设自己的 fov(官方逐通道 64.31–64.96°×5 + 89.34°),
        # **不能**共用一个 cam_bp —— 那就是历史上的"六路共用 90°"(见模块头注)。
        cam_bp = bp_lib.find("sensor.camera.rgb")
        for k, v in SURROUND_CAM_ATTRS.items():
            cam_bp.set_attribute(k, v)
        cam_bp.set_attribute("fov", f"{NUS_CAMERA_FOV[name]:.6f}")
        s = cast(
            carla.Sensor,
            world.spawn_actor(
                cam_bp,
                carla.Transform(tf, carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])),
                attach_to=ego,
            ),
        )
        q: queue.Queue = queue.Queue()
        s.listen(q.put)
        cams[name], qs[name] = s, q

    fovs = " ".join(f"{n.replace('CAM_', '')}={NUS_CAMERA_FOV[n]:.2f}°" for n in SURROUND_CAMS)
    print(f"[cams] {len(cams)} 环视相机挂载(nuScenes 官方 6DoF 挂点,逐通道 fov):{fovs}")

    for _ in range(5):  # 预热
        world.tick()
        for q in qs.values():
            q.get(timeout=10)

    w, h = NUS_CAMERA_WIDTH, NUS_CAMERA_HEIGHT
    # sensor2ego = [x, y, z, yaw, pitch, roll] 度(infos 口径)——由官方标定导出,
    # 与上面 spawn 用的是**同一份** NUS_CAMERA_RIG,不存在"布置与落盘两处维护"。
    # intrinsic 逐通道,且取的是**该通道 spawn 时用的那个 fov**(calib_from_fov 的 fx 与
    # CARLA 蓝图 fov 定义同一 ⇒ 渲染视野 == 落盘 K;主点 corner,理由见模块头注)。
    calib: dict[str, Any] = {
        name: {
            "sensor2ego": [mount[0], mount[1], mount[2], rot[1], rot[0], rot[2]],
            "intrinsic": calib_from_fov(w, h, NUS_CAMERA_FOV[name])["intrinsic"],
        }
        for name, (mount, rot) in NUS_CAMERA_RIG.items()
    }

    out = project_path(args.out)
    for name in SURROUND_CAMS:
        (out / name.lower()).mkdir(parents=True, exist_ok=True)
    # 数据溯源(照 `"map"` 键的既有做法):旧产物无这些键 = 1242×375/六路共用 90°/stride 1
    calib["map"] = default_map  # 该采集来自哪张图(旧产物无此键 = Town10HD_Opt)
    calib["spawn_index"] = args.spawn_index  # None = spawn_ego 首空位
    calib["stride"] = args.stride
    calib["image_size"] = [w, h]
    with open(out / "calib.json", "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=1)

    poses: list[dict] = []
    t0 = time.monotonic()
    try:
        for i in range(args.frames):
            # stride > 1:中间 tick 也必须**逐路抽干队列** —— 只取被测帧会让其余相机积压,
            # 下一帧读到的是更早的陈旧图(§P-M.7 判据 ⑥ 同款坑)。故先循环 tick 并每 tick 收齐。
            drained: dict[str, carla.Image] = {}
            for _ in range(args.stride):
                world.tick()
                drained = {name: qs[name].get(timeout=10) for name in SURROUND_CAMS}
            for name, image in drained.items():
                tmp = out / f".tmp_{i}_{name}.png"
                image.save_to_disk(str(tmp))
                tmp.rename(out / name.lower() / f"{i:06d}.png")
            egot = ego.get_transform()
            poses.append(
                {
                    "frame": i,
                    "tick": i * args.stride,  # 仿真 tick 号(stride>1 时与 frame 不等)
                    "x": round(egot.location.x, 3),
                    "y": round(egot.location.y, 3),
                    "z": round(egot.location.z, 3),
                    "yaw": round(egot.rotation.yaw, 3),
                    "pitch": round(egot.rotation.pitch, 3),
                    "roll": round(egot.rotation.roll, 3),
                }
            )
            if (i + 1) % 10 == 0 or i == args.frames - 1:
                dt = time.monotonic() - t0
                fps = (i + 1) / dt
                print(
                    f"[frame {i + 1}/{args.frames}] ego @ {tuple(round(v, 1) for v in loc(egot))} | {fps:.1f} fps"
                )
    finally:
        with open(out / "ego_pose.json", "w", encoding="utf-8") as f:
            json.dump(poses, f, indent=1)
        for s in cams.values():
            s.stop()
            s.destroy()
        for a in world.get_actors():
            if (
                a.type_id.startswith("vehicle")
                or a.type_id.startswith("walker")
                or a.type_id.startswith("controller")
            ):
                a.destroy()
    print(
        f"[done] surround root: {out.resolve()} "
        f"({args.frames} frames × {len(cams)} cams @ {w}×{h}, stride {args.stride}"
        f" = {args.frames * args.stride * 0.1:.1f}s 仿真时长)"
    )


if __name__ == "__main__":
    main()

"""B1:环视 6 相机采集(nuScenes 布局)→ 图像 + 内外参 + 逐帧 ego 位姿。

MapTR 端到端训练的输入侧:6 视角图像 + sensor2ego 外参 + 相机内参 + ego2global
位姿(与 MapTRv2 `nuscenes_converter` 的 cams/ego_pose 字段同构)。B2 组装器
消费本输出拼 MapTR 训练格式,地图 GT 来自 A 阶段矢量库。

采集纪律与 collect_drive 相同(同步模式/预热/清场/场景天气档);**不改动
A/B 采集器**(P1 复现性红线)。NPC 布置复用 collect_drive 的既有函数。

nuScenes 相机布局(全部挂 SENSOR_OFFSET,仅 yaw 不同,pitch=0):
  CAM_FRONT yaw=0 / FRONT_RIGHT -55 / FRONT_LEFT +55 / BACK 180 / BACK_LEFT 235 / BACK_RIGHT 125

落盘:
  outputs/surround_<scene>/cam_front/000000.png ...(6 视角)
  calib.json        — 每相机 sensor2ego + intrinsic(3x3)
  ego_pose.json     — 逐帧 ego2global(CARLA 世界系)

用法:
  python bin/collect_surround.py --frames 100 [--scene day_clear] [--npc-vehicles 15]
"""

from __future__ import annotations

import argparse
import json
import queue
import time
from typing import cast

import carla
from carla_common import CAM_ATTRS, SENSOR_OFFSET, loc, spawn_ego, sync_mode
from collect_drive import spawn_route_walkers, spawn_traffic

from autodrivedata.paths import project_path
from autodrivedata.scenarios import SCENES, merged_weather

# nuScenes 6 相机布局:名 → 相对 ego 的 yaw(度);pitch/roll 恒 0,挂点共用 SENSOR_OFFSET
SURROUND_CAMS = {
    "CAM_FRONT": 0.0,
    "CAM_FRONT_RIGHT": -55.0,
    "CAM_FRONT_LEFT": 55.0,
    "CAM_BACK": 180.0,
    "CAM_BACK_LEFT": 235.0,
    "CAM_BACK_RIGHT": 125.0,
}


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
    args = ap.parse_args()

    scene = SCENES[args.scene] if args.scene else None
    if scene is not None:
        for key, n in scene.traffic.items():
            setattr(args, key, n)
    if args.out is None:
        args.out = f"outputs/surround_{scene.name}" if scene else "outputs/surround_drive"

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)

    if scene is not None:
        world.set_weather(carla.WeatherParameters(**merged_weather(scene)))
        print(f"[scene] {scene.name} [{scene.group}] — 覆写 {sorted(scene.weather)}")

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    ego = spawn_ego(world)
    ego.set_autopilot(True, tm.get_port())
    tm.vehicle_percentage_speed_difference(ego, 30.0)
    print("[ego] autopilot on (TM 8000, 70% speed)")

    ego_t = ego.get_transform()
    spawn_traffic(world, tm, args.npc_vehicles, args.npc_walkers, args.seed)
    spawn_route_walkers(world, ego_t, args.route_walkers)

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    for k, v in CAM_ATTRS.items():
        cam_bp.set_attribute(k, v)

    cams: dict[str, carla.Sensor] = {}
    qs: dict[str, queue.Queue] = {}
    for name, yaw in SURROUND_CAMS.items():
        tf = carla.Transform(SENSOR_OFFSET.location, carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0))
        s = cast(carla.Sensor, world.spawn_actor(cam_bp, tf, attach_to=ego))
        q: queue.Queue = queue.Queue()
        s.listen(q.put)
        cams[name], qs[name] = s, q
    print(f"[cams] {len(cams)} 环视相机挂载(共用挂点 {SENSOR_OFFSET.location})")

    for _ in range(5):  # 预热
        world.tick()
        for q in qs.values():
            q.get(timeout=10)

    w, h = int(CAM_ATTRS["image_size_x"]), int(CAM_ATTRS["image_size_y"])
    fov = float(CAM_ATTRS["fov"])
    import math

    fx = w / 2 / math.tan(math.radians(fov / 2))
    intrinsic = [[fx, 0.0, w / 2], [0.0, fx, h / 2], [0.0, 0.0, 1.0]]
    # sensor2ego = 挂点 + 相对 yaw(attach 固定,首帧读取最准;此处与布置一致,运行时核验)
    calib = {
        name: {
            "sensor2ego": [
                SENSOR_OFFSET.location.x,
                SENSOR_OFFSET.location.y,
                SENSOR_OFFSET.location.z,
                yaw,
                0.0,
                0.0,
            ],
            "intrinsic": intrinsic,
        }
        for name, yaw in SURROUND_CAMS.items()
    }

    out = project_path(args.out)
    for name in SURROUND_CAMS:
        (out / name.lower()).mkdir(parents=True, exist_ok=True)
    with open(out / "calib.json", "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=1)

    poses: list[dict] = []
    t0 = time.monotonic()
    try:
        for i in range(args.frames):
            world.tick()
            for name in SURROUND_CAMS:
                image: carla.Image = qs[name].get(timeout=10)
                tmp = out / f".tmp_{i}_{name}.png"
                image.save_to_disk(str(tmp))
                tmp.rename(out / name.lower() / f"{i:06d}.png")
            egot = ego.get_transform()
            poses.append(
                {
                    "frame": i,
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
    print(f"[done] surround root: {out.resolve()} ({args.frames} frames × {len(cams)} cams)")


if __name__ == "__main__":
    main()

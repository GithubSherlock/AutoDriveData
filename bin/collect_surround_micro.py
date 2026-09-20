"""B1 微采样:官方布局 tiny 段(10 帧)→ 与已采旧布局做「布局对照」微实验。

用法(CARLA 服务器运行中):
  python bin/collect_surround_micro.py --out outputs/surround_micro --frames 10 \
      --cam-back 官方 --seed <seed>

默认不 spawn NPC(纯道路 + 地图矢量 overlay 对照,不受车流变量污染);
ego 固定起点(spawn point 0,Town10HD_Opt)以贴合周围马茨:
- **同步模式内** spawn 的传感器，位姿在 `world.tick()` 后才真实;首个 tick 前的
  `get_transform()` 是恒等 (0,0,0)(carla 同步快照红线)。这里全程依赖
  tick 后读取。

写盘目录结构同 collect_surround(6 视角 + calib.json + ego_pose.json),数据路径
见 B2 组装器消费面,可直接喂 assemble_maptr。
"""

from __future__ import annotations

import argparse
import json
import queue
import time
from pathlib import Path
from typing import cast

import carla
from carla_common import CAM_ATTRS, SENSOR_MOUNTS, loc, sync_mode

from autodrivedata.paths import project_path

# 布局常量**复制**自 collect_surround(同定义源,勿 import collect_surround 免串 env)
OFFICIAL_CAMS = {
    "CAM_FRONT": 0.0,
    "CAM_FRONT_RIGHT": -55.0,
    "CAM_FRONT_LEFT": 55.0,
    "CAM_BACK": 180.0,
    "CAM_BACK_LEFT": 108.6,
    "CAM_BACK_RIGHT": -110.8,
}
# 旧布局(235/125)——只用于微对照,不改主采集
LEGACY_CAMS = {
    "CAM_FRONT": 0.0,
    "CAM_FRONT_RIGHT": -55.0,
    "CAM_FRONT_LEFT": 55.0,
    "CAM_BACK": 180.0,
    "CAM_BACK_LEFT": 235.0,
    "CAM_BACK_RIGHT": 125.0,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出根目录")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--cam-back", choices=("官方", "legacy"), default="官方", help="后相机布局(对照用)")
    ap.add_argument("--npc", action="store_true", help="spawn NPC(默认不 spawn,纯道路对照)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()
    args.out = str(project_path(args.out))

    cams_spec = OFFICIAL_CAMS if args.cam_back == "官方" else LEGACY_CAMS

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)

    bp = world.get_blueprint_library().find("vehicle.audi.a2")
    ego: carla.Actor | None = None
    for pt in world.get_map().get_spawn_points():
        ego = world.try_spawn_actor(bp, pt)
        if ego is not None:
            break
    if ego is None:
        raise RuntimeError("所有出生点 spawn 失败")
    world.tick()
    eta = loc(ego.get_transform())
    print(f"[ego] {eta[:3]} @ spawn point 0, yaw {ego.get_transform().rotation.yaw:.1f}")

    if args.npc:
        # 轻量 NPC(复用 spawn_npcs 飞船级配置不想引)
        from carla_common import spawn_npcs

        t = ego.get_transform()
        spawn_npcs(world, t)

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    for k, v in CAM_ATTRS.items():
        cam_bp.set_attribute(k, v)

    cams: dict[str, carla.Sensor] = {}
    qs: dict[str, queue.Queue] = {}
    for name, yaw in cams_spec.items():
        x, y, z = SENSOR_MOUNTS[name]
        tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0))
        s = cast(carla.Sensor, world.spawn_actor(cam_bp, tf, attach_to=ego))
        q: queue.Queue = queue.Queue()
        s.listen(q.put)
        cams[name], qs[name] = s, q

    for _ in range(5):
        world.tick()
        for q in qs.values():
            q.get(timeout=10)

    w, h = int(CAM_ATTRS["image_size_x"]), int(CAM_ATTRS["image_size_y"])
    fov = float(CAM_ATTRS["fov"])
    import math

    fx = w / 2 / math.tan(math.radians(fov / 2))
    intrinsic = [[fx, 0.0, w / 2], [0.0, fx, h / 2], [0.0, 0.0, 1.0]]
    calib = {
        name: {
            "sensor2ego": [
                SENSOR_MOUNTS[name][0],
                SENSOR_MOUNTS[name][1],
                SENSOR_MOUNTS[name][2],
                yaw,
                0.0,
                0.0,
            ],
            "intrinsic": intrinsic,
        }
        for name, yaw in cams_spec.items()
    }

    out = Path(args.out)
    for name in cams_spec:
        (out / name.lower()).mkdir(parents=True, exist_ok=True)
    with open(out / "calib.json", "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=1)

    poses: list[dict] = []
    t0 = time.monotonic()
    try:
        for i in range(args.frames):
            world.tick()
            for name in cams_spec:
                image: carla.Image = qs[name].get(timeout=10)
                image.save_to_disk(str(out / name.lower() / f"{i:06d}.png"))
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
            if (i + 1) % 5 == 0 or i == args.frames - 1:
                print(
                    f"[frame {i + 1}/{args.frames}] ego @ {loc(egot)} | {(i + 1) / (time.monotonic() - t0):.1f} fps"
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

"""M2 动态采集:ego autopilot + Traffic Manager 车流 + 行走行人 → KITTI 序列。

用法(base env,CARLA 服务器运行中):
  python scripts/collect_drive.py [--out outputs/kitti_drive] [--frames 200]
    [--npc-vehicles 15] [--npc-walkers 6] [--seed 42]

与 collect_kitti 同格式(KITTI root);区别:场景动态(ego 自动驾驶、NPC 交通流)。
GT 逐帧取 actor 真值(box_to_gt_line 视野过滤),行人由 AI 控制器驱动行走。
"""
from __future__ import annotations

import argparse
import queue
from pathlib import Path
from typing import cast

import carla
import numpy as np

from autodrivedata import geometry as g
from autodrivedata.calib import CameraIntrinsics, KittiCalibOut, tr_velo_to_cam
from autodrivedata.export.kitti import write_frame
from autodrivedata.gt import ActorBox, box_to_gt_line

from carla_common import (
    CAM_ATTRS,
    LIDAR_ATTRS,
    SENSOR_OFFSET,
    _WalkerCtrl,
    loc,
    rad,
    spawn_ego,
    sync_mode,
)

NPC_MODELS = [
    "vehicle.tesla.model3",
    "vehicle.audi.a2",
    "vehicle.ford.mustang",
    "vehicle.toyota.prius",
    "vehicle.mercedes-benz.coupe",
]


def spawn_traffic(
    world: carla.World, tm: carla.TrafficManager, n_vehicles: int, n_walkers: int, seed: int
) -> None:
    """TM 车流 + AI 行走行人(确定性种子)。"""
    rng = np.random.default_rng(seed)
    bp_lib = world.get_blueprint_library()
    spawn_pts = world.get_map().get_spawn_points()

    # 车辆:随机出生点 + autopilot(TM 托管)
    for i in rng.choice(len(spawn_pts), size=min(n_vehicles, len(spawn_pts)), replace=False):
        bp = bp_lib.find(NPC_MODELS[int(i) % len(NPC_MODELS)])
        v = world.try_spawn_actor(bp, spawn_pts[int(i)])
        if v is not None:
            cast(carla.Vehicle, v).set_autopilot(True, tm.get_port())
    # 行人:出生点旁 + AI 控制器行走
    walker_bp = bp_lib.find("walker.pedestrian.0001")
    ctrl_bp = bp_lib.find("controller.ai.walker")
    for i in rng.choice(len(spawn_pts), size=min(n_walkers, len(spawn_pts)), replace=False):
        pt = spawn_pts[int(i)]
        tf = carla.Transform(
            carla.Location(
                x=pt.location.x + float(rng.uniform(-2, 2)),
                y=pt.location.y + float(rng.uniform(-2, 2)),
                z=pt.location.z,
            ),
            carla.Rotation(yaw=float(rng.uniform(0, 360))),
        )
        w = world.try_spawn_actor(walker_bp, tf)
        if w is None:
            continue
        ctrl = cast(_WalkerCtrl, world.spawn_actor(ctrl_bp, carla.Transform(), w))
        ctrl.start()
        ctrl.set_max_speed(float(rng.uniform(1.2, 1.8)))  # m/s
        dest_v = w.get_location() + w.get_transform().get_forward_vector() * 30
        ctrl.walk_to_location(carla.Location(x=dest_v.x, y=dest_v.y, z=dest_v.z))
    print(f"[traffic] {n_vehicles} vehicles + {n_walkers} walkers via TM/AI")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/kitti_drive")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--npc-vehicles", type=int, default=15)
    ap.add_argument("--npc-walkers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)  # 同步模式红线:TM 必须同步,否则车流冻结

    ego = spawn_ego(world)
    ego.set_autopilot(True, tm.get_port())
    tm.vehicle_percentage_speed_difference(ego, 30.0)  # 70% 速度,防冲撞
    print(f"[ego] autopilot on (TM 8000, 70% speed)")

    spawn_traffic(world, tm, args.npc_vehicles, args.npc_walkers, args.seed)

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    for k, v in CAM_ATTRS.items():
        cam_bp.set_attribute(k, v)
    lid_bp = bp_lib.find("sensor.lidar.ray_cast")
    for k, v in LIDAR_ATTRS.items():
        lid_bp.set_attribute(k, v)
    camera = cast(carla.Sensor, world.spawn_actor(cam_bp, SENSOR_OFFSET, attach_to=ego))
    lidar = cast(carla.Sensor, world.spawn_actor(lid_bp, SENSOR_OFFSET, attach_to=ego))

    img_q: "queue.Queue" = queue.Queue()
    lid_q: "queue.Queue" = queue.Queue()
    camera.listen(img_q.put)
    lidar.listen(lid_q.put)

    for _ in range(5):  # 预热
        world.tick()
        img_q.get(timeout=10)
        lid_q.get(timeout=10)

    k = CameraIntrinsics(
        width=int(CAM_ATTRS["image_size_x"]),
        height=int(CAM_ATTRS["image_size_y"]),
        fov_h_deg=float(CAM_ATTRS["fov"]),
    )
    out = Path(args.out)

    try:
        for i in range(args.frames):
            world.tick()
            image: carla.Image = img_q.get(timeout=10)
            pts: carla.LidarMeasurement = lid_q.get(timeout=10)

            cam_t, lid_t = camera.get_transform(), lidar.get_transform()
            calib_out = KittiCalibOut(
                p2=k.p2(),
                tr_velo_to_cam=tr_velo_to_cam(
                    loc(lid_t), rad(lid_t.rotation), loc(cam_t), rad(cam_t.rotation)
                ),
            )
            labels: list[str] = []
            for a in world.get_actors():
                if not (a.type_id.startswith("vehicle") or a.type_id.startswith("walker")):
                    continue
                bb = a.bounding_box
                box = ActorBox(
                    type_id=a.type_id,
                    extent=(bb.extent.x, bb.extent.y, bb.extent.z),
                    location=(bb.location.x, bb.location.y, bb.location.z),
                    rotation=rad(bb.rotation),
                    actor_location=loc(a.get_transform()),
                    actor_rotation=rad(a.get_transform().rotation),
                )
                line = box_to_gt_line(box, loc(cam_t), rad(cam_t.rotation), k)
                if line:
                    labels.append(line)

            tmp = out / f".tmp_{i}.png"
            image.save_to_disk(str(tmp))
            png = tmp.read_bytes()
            tmp.unlink()

            velo = g.carla_lidar_to_velodyne(
                np.frombuffer(pts.raw_data, dtype=np.float32).reshape(-1, 4)
            )
            write_frame(out, str(i), image_png=png, velodyne=velo, calib=calib_out, labels=labels)
            if (i + 1) % 20 == 0 or i == args.frames - 1:
                print(f"[frame {i + 1}/{args.frames}] ego @ {tuple(round(v, 1) for v in loc(ego.get_transform()))} | {len(labels)} GT")
    finally:
        camera.stop()
        lidar.stop()
        camera.destroy()
        lidar.destroy()
        for a in world.get_actors():
            if a.type_id.startswith("vehicle") or a.type_id.startswith("walker") or a.type_id.startswith("controller"):
                a.destroy()
    print(f"[done] KITTI root: {out.resolve()} ({args.frames} frames)")


if __name__ == "__main__":
    main()

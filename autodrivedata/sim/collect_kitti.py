"""M1a 静态采集:ego 静止 + 摆 NPC + 相机/LiDAR 同步模式 → KITTI root 落盘(raw + GT)。

用法(base env,CARLA 服务器运行中):
  python -m autodrivedata.sim.collect_kitti [--out outputs/kitti_scene] [--frames 10] [--host 127.0.0.1]

落盘布局照 auto3dlabel KittiFrame 契约(验收时 KITTI_OBJECT_ROOT=--out 零改动读入):
  {out}/training/{image_2,velodyne,calib,label_2}/000000.*
GT = 动态 actor(KITTI 类名);点云落盘前做 y 翻转对齐 KITTI velodyne 约定(见 geometry.py)。
"""

from __future__ import annotations

import argparse
import queue
from typing import cast

import carla
import numpy as np

from autodrivedata import geometry as g
from autodrivedata.calib import CameraIntrinsics, KittiCalibOut, tr_velo_to_cam
from autodrivedata.export.kitti import write_frame
from autodrivedata.gt import ActorBox, box_to_gt_line
from autodrivedata.paths import project_path
from autodrivedata.semantic import semantic_to_velodyne_bin
from autodrivedata.sim.carla_common import (
    CAM_ATTRS,
    LIDAR_ATTRS,
    SENSOR_OFFSET,
    loc,
    rad,
    spawn_ego,
    spawn_npcs,
    sync_mode,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/kitti_scene")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--lidar-noise", type=float, default=0.0, help="LiDAR 测距噪声 σ(米),0=关闭")
    ap.add_argument("--lidar-dropoff", type=float, default=0.0, help="LiDAR 随机丢点率 0-1,0=关闭")
    ap.add_argument(
        "--semantic-lidar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="用语义 LiDAR 采集并按语义标签合成 KITTI 式强度(M2-3 实测:几何强度恒定致检出全漏)",
    )
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)

    ego = spawn_ego(world)
    ego_t = ego.get_transform()
    print(f"[ego] vehicle.audi.a2 @ {loc(ego_t)}")

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    for k, v in CAM_ATTRS.items():
        cam_bp.set_attribute(k, v)
    lid_bp = bp_lib.find(
        "sensor.lidar.ray_cast" if not args.semantic_lidar else "sensor.lidar.ray_cast_semantic"
    )
    for k, v in LIDAR_ATTRS.items():
        lid_bp.set_attribute(k, v)
    # 域差距实验旋钮(M2-3):真实传感器有测距噪声与随机丢点,CARLA 默认全 0
    if args.lidar_noise > 0:
        lid_bp.set_attribute("noise_stddev", str(args.lidar_noise))
    if args.lidar_dropoff > 0:
        lid_bp.set_attribute("dropoff_general_rate", str(args.lidar_dropoff))
    camera = cast(carla.Sensor, world.spawn_actor(cam_bp, SENSOR_OFFSET, attach_to=ego))
    lidar = cast(carla.Sensor, world.spawn_actor(lid_bp, SENSOR_OFFSET, attach_to=ego))
    print(
        f"[sensor] camera {CAM_ATTRS['image_size_x']}x{CAM_ATTRS['image_size_y']} "
        f"fov={CAM_ATTRS['fov']} + lidar {LIDAR_ATTRS['channels']}ch {LIDAR_ATTRS['points_per_second']}pps"
    )

    spawn_npcs(world, ego_t)

    img_q: queue.Queue = queue.Queue()
    lid_q: queue.Queue = queue.Queue()
    camera.listen(img_q.put)
    lidar.listen(lid_q.put)

    # 预热:丢前几 tick(传感器首帧常不完整),清空队列
    for _ in range(5):
        world.tick()
        img_q.get(timeout=10)
        lid_q.get(timeout=10)

    k = CameraIntrinsics(
        width=int(CAM_ATTRS["image_size_x"]),
        height=int(CAM_ATTRS["image_size_y"]),
        fov_h_deg=float(CAM_ATTRS["fov"]),
    )
    out = project_path(args.out)

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
                line = box_to_gt_line(box, loc(cam_t), rad(cam_t.rotation), k, max_distance=65.0)
                if line:
                    labels.append(line)

            tmp = out / f".tmp_{i}.png"
            image.save_to_disk(str(tmp))
            png = tmp.read_bytes()
            tmp.unlink()

            raw = np.frombuffer(pts.raw_data, dtype=np.float32)
            if args.semantic_lidar:
                velo = semantic_to_velodyne_bin(raw.reshape(-1, 6), seed=args.frames * 100 + i)
            else:
                velo = g.carla_lidar_to_velodyne(raw.reshape(-1, 4))
            paths = write_frame(out, str(i), image_png=png, velodyne=velo, calib=calib_out, labels=labels)
            print(f"[frame {i}] {len(labels)} GT / {len(velo)} pts -> {paths.image.name}")
    finally:
        camera.stop()
        lidar.stop()
        camera.destroy()
        lidar.destroy()
        for a in world.get_actors():
            if (
                a.type_id.startswith("vehicle")
                or a.type_id.startswith("walker")
                or a.type_id.startswith("controller")
            ):
                a.destroy()
    print(f"[done] KITTI root: {out.resolve()}")


if __name__ == "__main__":
    main()

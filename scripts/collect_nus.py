"""M1b 静态采集:ego 静止 + NPC + 6 相机 + LiDAR → nuScenes 迷你集(scene-0103)。

用法(base env,CARLA 服务器运行中):
  python scripts/collect_nus.py [--out outputs/nus_mini] [--frames 2]

落盘 = 标准 nuScenes dataroot(devkit 直读,auto3dlabel nuscenes-queue 消费):
  {out}/v1.0-mini/*.json(14 表)+ samples/LIDAR_TOP/*.bin((N,5) raw)+ samples/CAM_*/*.png
坐标系:入表数据经 geometry.CARLA_TO_NUS 翻 y(入 nuScenes 全局系 y 左)。
"""

from __future__ import annotations

import argparse
import queue
from pathlib import Path
from typing import cast

import carla
import numpy as np
from carla_common import (
    LIDAR_ATTRS,
    SENSOR_OFFSET,
    loc,
    rad,
    spawn_ego,
    spawn_npcs,
    sync_mode,
)

from autodrivedata import geometry as g
from autodrivedata.export.nuscenes import (
    NUS_CAMERAS,
    NusSample,
    count_points_in_box_nus,
    points_sensor_to_global_nus,
    write_mini_dataset,
)
from autodrivedata.gt import ActorBox, box_center_world, box_heading_world, classify_nus

# 6 相机视角(相对 ego,CARLA yaw 度,左转正):nuScenes 标准通道
CAM_YAW_OFFSET = {
    "CAM_FRONT": 0.0,
    "CAM_FRONT_LEFT": 55.0,
    "CAM_FRONT_RIGHT": -55.0,
    "CAM_BACK": 180.0,
    "CAM_BACK_LEFT": 125.0,
    "CAM_BACK_RIGHT": -125.0,
}
CAM_ATTRS = {
    "image_size_x": "1600",
    "image_size_y": "900",
    "fov": "90",
}  # nuScenes 分辨率


def _actor_to_annotation(a: carla.Actor, points_global: np.ndarray) -> dict | None:
    """actor → nuScenes GT 标注 dict(全局系;类别忽略类返回 None)。"""
    category = classify_nus(a.type_id)
    if category is None:
        return None
    bb = a.bounding_box
    box = ActorBox(
        type_id=a.type_id,
        extent=(bb.extent.x, bb.extent.y, bb.extent.z),
        location=(bb.location.x, bb.location.y, bb.location.z),
        rotation=rad(bb.rotation),
        actor_location=loc(a.get_transform()),
        actor_rotation=rad(a.get_transform().rotation),
    )
    center_nus = g.carla_to_nus_global(box_center_world(box)[None])[0]
    center_xyz = (float(center_nus[0]), float(center_nus[1]), float(center_nus[2]))
    heading_nus = g.CARLA_TO_NUS @ box_heading_world(box)
    yaw_nus = float(np.arctan2(heading_nus[1], heading_nus[0]))
    ex = bb.extent
    size = (2 * ex.y, 2 * ex.x, 2 * ex.z)  # (w,l,h)
    num_pts = count_points_in_box_nus(points_global, center_xyz, size, yaw_nus)
    return {
        "category": category,
        "translation": center_xyz,
        "size": size,
        "yaw_nus": yaw_nus,
        "num_lidar_pts": num_pts,
        "instance_token": f"adinst{a.id:x}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/nus_mini")
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)

    ego = spawn_ego(world)
    ego_t = ego.get_transform()
    print(f"[ego] vehicle.audi.a2 @ {loc(ego_t)}")

    bp_lib = world.get_blueprint_library()
    lid_bp = bp_lib.find("sensor.lidar.ray_cast")
    for k, v in LIDAR_ATTRS.items():
        lid_bp.set_attribute(k, v)
    lidar = cast(carla.Sensor, world.spawn_actor(lid_bp, SENSOR_OFFSET, attach_to=ego))
    cameras: dict[str, carla.Sensor] = {}
    for cam, yaw_off in CAM_YAW_OFFSET.items():
        cam_bp = bp_lib.find("sensor.camera.rgb")
        for k, v in CAM_ATTRS.items():
            cam_bp.set_attribute(k, v)
        tf = carla.Transform(SENSOR_OFFSET.location, carla.Rotation(pitch=0.0, yaw=yaw_off, roll=0.0))
        cameras[cam] = cast(carla.Sensor, world.spawn_actor(cam_bp, tf, attach_to=ego))
    print(
        f"[sensor] lidar {LIDAR_ATTRS['channels']}ch + 6 cameras {CAM_ATTRS['image_size_x']}x{CAM_ATTRS['image_size_y']}"
    )

    spawn_npcs(world, ego_t)

    lid_q: queue.Queue = queue.Queue()
    cam_qs: dict[str, queue.Queue] = {c: queue.Queue() for c in NUS_CAMERAS}
    lidar.listen(lid_q.put)
    for cam, s in cameras.items():
        s.listen(cam_qs[cam].put)

    for _ in range(5):  # 预热
        world.tick()
        lid_q.get(timeout=10)
        for q in cam_qs.values():
            q.get(timeout=10)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    calib_lidar = (
        (SENSOR_OFFSET.location.x, SENSOR_OFFSET.location.y, SENSOR_OFFSET.location.z),
        0.0,
    )
    calib_cameras = {
        c: (calib_lidar[0], g.carla_yaw_to_nus_yaw(np.radians(CAM_YAW_OFFSET[c]))) for c in NUS_CAMERAS
    }

    samples: list[NusSample] = []
    try:
        for i in range(args.frames):
            world.tick()
            pts_raw = np.frombuffer(lid_q.get(timeout=10).raw_data, dtype=np.float32).reshape(-1, 4)
            pts_nus = g.carla_lidar_to_velodyne(pts_raw)  # y 翻转 = nus 传感器约定
            pts5 = np.hstack([pts_nus, np.zeros((len(pts_nus), 1), dtype=np.float32)])  # elongation=0
            lidar_rel = f"samples/LIDAR_TOP/{i:06d}.bin"
            (out / lidar_rel).parent.mkdir(parents=True, exist_ok=True)
            pts5.tofile(out / lidar_rel)

            camera_filenames: dict[str, str] = {}
            for cam in NUS_CAMERAS:
                img = cam_qs[cam].get(timeout=10)
                rel = f"samples/{cam}/{i:06d}.png"
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                img.save_to_disk(str(out / rel))
                camera_filenames[cam] = rel

            ego_t_now = ego.get_transform()
            _ego_nus = g.carla_to_nus_global(np.asarray([loc(ego_t_now)]))[0]
            ego_trans_nus = (float(_ego_nus[0]), float(_ego_nus[1]), float(_ego_nus[2]))
            ego_yaw_nus = g.carla_yaw_to_nus_yaw(np.radians(ego_t_now.rotation.yaw))

            pts_global = points_sensor_to_global_nus(
                pts_nus[:, :3],
                ego_trans_nus,
                ego_yaw_nus,
                calib_lidar[0],
                calib_lidar[1],
            )
            annotations: list[dict] = []
            for a in world.get_actors():
                if not (a.type_id.startswith("vehicle") or a.type_id.startswith("walker")):
                    continue
                ann = _actor_to_annotation(a, pts_global)
                if ann is not None:
                    annotations.append(ann)

            samples.append(
                NusSample(
                    ego_translation=ego_trans_nus,
                    ego_yaw_nus=ego_yaw_nus,
                    lidar_filename=lidar_rel,
                    camera_filenames=camera_filenames,
                    calib_lidar=calib_lidar,
                    calib_cameras=calib_cameras,
                    annotations=annotations,
                    timestamp=1700000000000000 + i * 50000,
                )
            )
            print(f"[sample {i}] {len(pts5)} pts / {len(annotations)} GT / 6 cams")
    finally:
        lidar.stop()
        lidar.destroy()
        for s in cameras.values():
            s.stop()
            s.destroy()
        for a in world.get_actors():
            if (
                a.type_id.startswith("vehicle")
                or a.type_id.startswith("walker")
                or a.type_id.startswith("controller")
            ):
                a.destroy()

    # mini_val = {scene-0103, scene-0916}——两个场景都要有,否则 nuscenes-queue 遍历 KeyError
    write_mini_dataset(
        out,
        "v1.0-mini",
        {"scene-0103": samples, "scene-0916": [samples[0]]},
    )
    print(f"[done] nuScenes dataroot: {out.resolve()} (2 scenes, {len(samples)}+1 samples)")


if __name__ == "__main__":
    main()

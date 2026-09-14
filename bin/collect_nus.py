"""M1b 静态采集:ego 静止 + NPC + 6 相机 + LiDAR + 5 雷达 → nuScenes 迷你集(scene-0103)。

用法(base env,CARLA 服务器运行中):
  python bin/collect_nus.py [--out outputs/nus_mini] [--frames 2]

落盘 = 标准 nuScenes dataroot(devkit 直读,auto3dlabel nuscenes-queue 消费):
  {out}/v1.0-mini/*.json(14 表)+ samples/LIDAR_TOP/*.bin((N,5) raw)
  + samples/RADAR_*/*.pcd(18 字段 radar 点云)+ samples/CAM_*/*.png
坐标系:入表数据经 geometry.CARLA_TO_NUS 翻 y(入 nuScenes 全局系 y 左)。

雷达(2026-09-14 扩展):5 通道照官方 nuScenes 布局(RADAR_FRONT/FRONT_LEFT/
FRONT_RIGHT/BACK_LEFT/BACK_RIGHT),参数复刻官方大陆 ars408(77°×14.2°、range
250);同步模式固定 0.1s → 与 LiDAR 同为 10Hz(官方 13Hz 无法复刻)。挂载 = 官方
calibrated_sensor 位姿的 CARLA 镜像(spawn location y 取负、yaw 与同名相机同号),
入表零转换直接用官方原值 → 雷达锥与同名相机同物理象限。radar 点进 GT 关联:
5 通道点各经 points_sensor_to_global_nus → 全局系合并 → count_points_in_box_nus
计 num_radar_pts。
"""

from __future__ import annotations

import argparse
import math
import queue
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
    NUS_CAMERA_CALIBS,
    NUS_CAMERAS,
    NUS_RADAR_CHANNELS,
    NUS_RADAR_OFFSETS,
    NusSample,
    count_points_in_box_nus,
    points_sensor_to_global_nus,
    write_mini_dataset,
)
from autodrivedata.gt import ActorBox, box_center_world, box_heading_world, classify_nus
from autodrivedata.paths import project_path
from autodrivedata.radar import detections_to_nus18, mask_radar_points, nus18_to_pcd

# 6 相机视角(相对 ego,CARLA yaw 度,左转正):yaw 取官方相机光轴方位角
# (NUS_CAMERA_CALIBS 光轴 = R@+z 的 CARLA 镜像;实测 CAM_FRONT=+0.3°≈0)——
# 保证真实渲染视野与 devkit 深度解释(光轴方向)对齐。
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

# 5 雷达视角(相对 ego,CARLA yaw 度,左转正)——与同名相机同号,保证雷达锥与
# 相机同物理象限;spawn 平移 = 官方 calibrated_sensor 的 CARLA 镜像(y 取负)。
RADAR_YAW_OFFSET = {
    "RADAR_FRONT": 0.0,
    "RADAR_FRONT_LEFT": 45.0,
    "RADAR_FRONT_RIGHT": -45.0,
    "RADAR_BACK_LEFT": 90.0,
    "RADAR_BACK_RIGHT": -90.0,
}
# 官方大陆 ars408 雷达参数(复刻;horizontal 77°/vertical 14.2°/range 250m)。
# **CARLA 0.9.16 两 FOV 属性交叉使用**(编译行为,prebuilt 无法改源码):
#   实际 azi 半角 = vertical_fov/2、实际 alt 半角 = horizontal_fov/2
#   (12 组属性扫描自洽:设 hf=77/vf=14.2 → 实测 azi±7°/alt±38°,与 ars408 完全反了)。
# 故对调属性值获得真实 ars408 锥:设 hf=14.2/vf=77 → 实测 azi±38.1°/alt±7.0° ✓
RADAR_ATTRS = {
    "horizontal_fov": "14.2",
    "vertical_fov": "77",
    "range": "250",
    "points_per_second": "3300",
    "sensor_tick": "0.1",
}


def _latest(q: queue.Queue):
    """取队列里最新一帧(非阻塞);空则 None。

    同步模式下**只有 world.tick() 才产生新帧**,阻塞 get(timeout=10) 是死等待
    (不 tick 数据永远不会来,等满超时抛 Empty)。radar 的 sensor_tick=0.1 与
    同步 tick 同周期,存在相位偶发空 tick(实测 30 tick 2 空,~7%)→ 一律
    drain 式取帧,空由调用方补 tick 处理(C22)。lidar/相机每 tick 必有,
    仍用阻塞 get 保时序。
    """
    frame = None
    while True:
        try:
            frame = q.get_nowait()
        except queue.Empty:
            return frame


def _actor_to_annotation(
    a: carla.Actor,
    points_global: np.ndarray,
    points_radar_global: np.ndarray,
    origin_nus: tuple[float, float, float],
) -> dict | None:
    """actor → nuScenes GT 标注 dict(全局系,已按场景原点平移;类别忽略类返回 None)。

    points_global 是"以场景起点为原点"的 nus 系(采集主循环经 ego_trans_nus
    = ego−origin 平移),GT 框中心必须同口径平移,否则关联计数全空。
    """
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
    center_nus = center_nus - np.asarray(origin_nus, dtype=np.float64)
    center_xyz = (float(center_nus[0]), float(center_nus[1]), float(center_nus[2]))
    heading_nus = g.CARLA_TO_NUS @ box_heading_world(box)
    yaw_nus = float(np.arctan2(heading_nus[1], heading_nus[0]))
    ex = bb.extent
    size = (2 * ex.y, 2 * ex.x, 2 * ex.z)  # (w,l,h)
    num_pts = count_points_in_box_nus(points_global, center_xyz, size, yaw_nus)
    num_radar = count_points_in_box_nus(points_radar_global, center_xyz, size, yaw_nus)
    return {
        "category": category,
        "translation": center_xyz,
        "size": size,
        "yaw_nus": yaw_nus,
        "num_lidar_pts": num_pts,
        "num_radar_pts": num_radar,
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
    # 5 雷达:挂载 = 官方 calibrated_sensor 的 CARLA 镜像(y 取负、yaw 同号)
    radars: dict[str, carla.Sensor] = {}
    for ch in NUS_RADAR_CHANNELS:
        r_bp = bp_lib.find("sensor.other.radar")
        for k, v in RADAR_ATTRS.items():
            r_bp.set_attribute(k, v)
        t_nus, _ = NUS_RADAR_OFFSETS[ch]
        tf = carla.Transform(
            carla.Location(x=t_nus[0], y=-t_nus[1], z=t_nus[2]),
            carla.Rotation(pitch=0.0, yaw=RADAR_YAW_OFFSET[ch], roll=0.0),
        )
        radars[ch] = cast(carla.Sensor, world.spawn_actor(r_bp, tf, attach_to=ego))
    print(
        f"[sensor] lidar {LIDAR_ATTRS['channels']}ch + {len(radars)} radars(ars408) + "
        f"6 cameras {CAM_ATTRS['image_size_x']}x{CAM_ATTRS['image_size_y']}"
    )

    spawn_npcs(world, ego_t)

    lid_q: queue.Queue = queue.Queue()
    cam_qs: dict[str, queue.Queue] = {c: queue.Queue() for c in NUS_CAMERAS}
    radar_qs: dict[str, queue.Queue] = {ch: queue.Queue() for ch in NUS_RADAR_CHANNELS}
    lidar.listen(lid_q.put)
    for cam, s in cameras.items():
        s.listen(cam_qs[cam].put)
    for ch, s in radars.items():
        s.listen(radar_qs[ch].put)

    # 预热:每个 tick 至少取到一帧,才认为同步就绪(雷达 sensor_tick 相位偶发空,
    # 已 drain 丢弃;5 tick 内必收敛——实测最多连续 2 空)
    warmed_radar = {ch: False for ch in NUS_RADAR_CHANNELS}
    for _ in range(10):  # 预热
        world.tick()
        lid_q.get(timeout=10)
        for q in cam_qs.values():
            q.get(timeout=10)
        for ch, q in radar_qs.items():
            if _latest(q) is not None:
                warmed_radar[ch] = True
        if all(warmed_radar.values()):
            break

    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # 官方每场景有独立全局坐标系原点(nuscenes_mini 实测 scene-0061/0103/0553 的
    # 首帧 ego 平移 = 各自场景基准,如 (411,1181,0)/(600,1647,0)/(1316,1039,0))——
    # 我们单场景采集也按此约定:采集中以 ego 首帧为原点,平移量 = 首帧负值,
    # 使本场景坐标系与官方"以场景为单元"的全局一致。
    _ego0 = loc(ego.get_transform())
    nus_origin = g.carla_to_nus_global(np.asarray([_ego0]))[0]
    origin_nus = (float(nus_origin[0]), float(nus_origin[1]), float(nus_origin[2]))
    calib_lidar = (
        (SENSOR_OFFSET.location.x, SENSOR_OFFSET.location.y, SENSOR_OFFSET.location.z),
        0.0,
    )
    calib_cameras = {c: NUS_CAMERA_CALIBS[c] for c in NUS_CAMERAS}
    # calib_radars:translation = 官方 nus 原值(nus 系,零转换),rotation = CARLA
    # yaw 走与相机同一转换路径 → 雷达锥与同名相机同物理象限。
    calib_radars = {
        ch: (NUS_RADAR_OFFSETS[ch][0], g.carla_yaw_to_nus_yaw(math.radians(RADAR_YAW_OFFSET[ch])))
        for ch in NUS_RADAR_CHANNELS
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

            # 5 雷达:raw → 18 字段 nus 点 → 提前按 devkit 过滤 → .pcd 落盘。
            # drain 取最新帧(C22 相位空 tick);空则跳过本帧(该通道本 tick 无点,
            # 写空 pcd——devkit 空编码 (18,0),不阻断采集)。
            radar_filenames: dict[str, str] = {}
            radar_nus18: dict[str, np.ndarray] = {}
            for ch in NUS_RADAR_CHANNELS:
                frame = _latest(radar_qs[ch])
                if frame is None:
                    print(f"  [warn] {ch} 本 tick 空(相位),写空 pcd")
                    nus18 = np.empty((0, 18), dtype=np.float32)
                else:
                    dets = np.frombuffer(frame.raw_data, dtype=np.float32).reshape(-1, 4)
                    nus18 = detections_to_nus18(dets, sensor_id=NUS_RADAR_CHANNELS.index(ch))
                    nus18 = nus18[mask_radar_points(nus18)]
                radar_nus18[ch] = nus18
                rel = f"samples/{ch}/{i:06d}.pcd"
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                (out / rel).write_bytes(nus18_to_pcd(nus18))
                radar_filenames[ch] = rel

            camera_filenames: dict[str, str] = {}
            for cam in NUS_CAMERAS:
                img = cam_qs[cam].get(timeout=10)
                rel = f"samples/{cam}/{i:06d}.png"
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                img.save_to_disk(str(out / rel))
                camera_filenames[cam] = rel

            ego_t_now = ego.get_transform()
            _ego_nus = g.carla_to_nus_global(np.asarray([loc(ego_t_now)]))[0]
            ego_trans_nus = (
                float(_ego_nus[0] - origin_nus[0]),
                float(_ego_nus[1] - origin_nus[1]),
                float(_ego_nus[2] - origin_nus[2]),
            )
            ego_yaw_nus = g.carla_yaw_to_nus_yaw(np.radians(ego_t_now.rotation.yaw))

            pts_global = points_sensor_to_global_nus(
                pts_nus[:, :3],
                ego_trans_nus,
                ego_yaw_nus,
                calib_lidar[0],
                calib_lidar[1],
            )
            # 5 雷达点各自进全局系(calib 非零 yaw 由 points_sensor_to_global_nus 支持)
            # → 合并 → GT 框内计数 num_radar_pts(与官方"当前 sample 全雷达通道命中总数"同口径)
            radar_pts_global: list[np.ndarray] = []
            for ch in NUS_RADAR_CHANNELS:
                radar_pts_global.append(
                    points_sensor_to_global_nus(
                        radar_nus18[ch][:, :3],
                        ego_trans_nus,
                        ego_yaw_nus,
                        calib_radars[ch][0],
                        calib_radars[ch][1],
                    )
                )
            radar_global = np.vstack(radar_pts_global) if radar_pts_global else np.empty((0, 3))
            annotations: list[dict] = []
            for a in world.get_actors():
                if not (a.type_id.startswith("vehicle") or a.type_id.startswith("walker")):
                    continue
                ann = _actor_to_annotation(a, pts_global, radar_global, origin_nus)
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
                    radar_filenames=radar_filenames,
                    calib_radars=calib_radars,
                )
            )
            print(
                f"[sample {i}] {len(pts5)} lidar / {sum(len(v) for v in radar_nus18.values())} radar / "
                f"{len(annotations)} GT / 6 cams"
            )
    finally:
        lidar.stop()
        lidar.destroy()
        for s in cameras.values():
            s.stop()
            s.destroy()
        for s in radars.values():
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

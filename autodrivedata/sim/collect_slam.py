"""B2 SLAM 序列采集:ego 路线行驶 → 逐帧 velodyne + **ego 真值位姿**。

**为什么单开一个采集器**(不复用 `collect_drive.py`):
- `collect_drive.py` 走 autopilot 且**不落位姿** → 旧 `kitti_drive` 的 SLAM 结果
  只能报"首末位姿距离",冒充不了精度指标(Plan2.md P-H 已如实标注该口径缺陷)。
  SLAM 评估(ATE/RPE/回环收益)**必须有真值位姿**——本采集器逐帧写
  `training/pose/{id}.txt`(KITTI 12 数行主序,见 `export/kitti.write_pose`)。
- **autopilot 在这里可用**:A/B 纪律禁 autopilot 是因为帧级配对要求可复现;SLAM 有
  真值位姿后,可复现性不再是前提。且 autopilot 自然走出闭合/重访路线,给后端回环
  检测提供真实测试数据(直线段永远测不出回环收益)。
- **不写图像**:SLAM 只用点云 + 位姿。150 帧图像 ≈ 400M,点云+位姿 ≈ 60M。

用法:
  python -m autodrivedata.sim.collect_slam --frames 400 --out outputs/kitti_slam
  python -m autodrivedata.sim.collect_slam --frames 200 --speed 0 --map Town13 --out outputs/kitti_slam_t13
"""

from __future__ import annotations

import argparse
import json
import queue
from typing import cast

import carla
import numpy as np

from autodrivedata.export.kitti import frame_paths, write_pose
from autodrivedata.geometry import carla_lidar_to_velodyne, carla_rotation_matrix
from autodrivedata.paths import project_path
from autodrivedata.perception.semantic import semantic_to_velodyne_bin
from autodrivedata.sim.carla_common import LIDAR_ATTRS, SENSOR_OFFSET, loc, rad, spawn_ego, sync_mode


def ego_pose_matrix(ego_t: carla.Transform) -> np.ndarray:
    """ego CARLA 位姿 → 4×4(ego 局部系 → 世界系)。

    旋转块走 `geometry.carla_rotation_matrix`(组合顺序 Rz·Ry·Rx,与 pycarla
    `Transform.get_matrix()` 旋转块逐元素一致,有 oracle 单测锁定)——**不要**自己
    按 (roll,pitch,yaw) 拼矩阵:`carla_common.rad()` 的返回序是 **(pitch, yaw, roll)**,
    正是该函数要的序,两者配套;自拼极易错序。
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = carla_rotation_matrix(rad(ego_t.rotation))
    T[:3, 3] = loc(ego_t)
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400, help="采集帧数(10 Hz → 400 帧 = 40 s)")
    ap.add_argument("--out", default="outputs/kitti_slam", help="输出 KITTI root(经 project_path)")
    ap.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help=">0 = 定速直行(确定性,不 autopilot);0 = TM autopilot 走路线(自然重访,测回环)",
    )
    ap.add_argument("--map", default=None, help="切图(缺省=当前图)")
    ap.add_argument("--semantic-lidar", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)
    world = client.load_world(args.map) if args.map else client.get_world()
    sync_mode(world)

    # 清场:残留 actor 会阻塞 spawn point(红线纪律,勿删)。
    # **逐个 try**:destroy 父 actor 会连带销毁其子传感器,快照里的子 actor 再 destroy
    # 就报 "unable to destroy actor: not found"(噪声,非错误)——吞掉即可。
    for a in world.get_actors():
        if a.type_id.startswith("vehicle.") or a.type_id.startswith("sensor."):
            try:
                a.destroy()
            except RuntimeError:
                pass
    world.tick()

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    ego = spawn_ego(world)
    vel = carla.Vector3D(0.0, 0.0, 0.0)
    if args.speed > 0:
        ego.set_autopilot(False)
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))  # 清制动残留
        fwd = ego.get_transform().get_forward_vector()
        vel = carla.Vector3D(x=fwd.x * args.speed, y=fwd.y * args.speed, z=0.0)
        print(f"[ego] 定速 {args.speed} m/s 直行(确定性)")
    else:
        ego.set_autopilot(True, tm.get_port())
        print("[ego] TM autopilot(路线行驶,自然重访 → 后端回环有真实测试数据)")

    bp_lib = world.get_blueprint_library()
    lid_bp = bp_lib.find(
        "sensor.lidar.ray_cast" if not args.semantic_lidar else "sensor.lidar.ray_cast_semantic"
    )
    for k, v in LIDAR_ATTRS.items():
        lid_bp.set_attribute(k, v)
    lidar = cast(carla.Sensor, world.spawn_actor(lid_bp, SENSOR_OFFSET, attach_to=ego))

    lid_q: queue.Queue = queue.Queue()
    lidar.listen(lid_q.put)
    for _ in range(5):  # 预热
        world.tick()
        lid_q.get(timeout=10)

    out = project_path(args.out)
    poses: list[np.ndarray] = []

    try:
        for i in range(args.frames):
            if args.speed > 0:
                ego.set_target_velocity(vel)  # 每帧重发:VehicleControl 一旦设置就每步生效
            world.tick()
            pts: carla.LidarMeasurement = lid_q.get(timeout=10)
            ego_t = ego.get_transform()
            T = ego_pose_matrix(ego_t)
            poses.append(T)

            raw = np.frombuffer(pts.raw_data, dtype=np.float32)
            if args.semantic_lidar:
                velo = semantic_to_velodyne_bin(raw.reshape(-1, 6), seed=args.frames * 100 + i)
            else:
                velo = carla_lidar_to_velodyne(raw.reshape(-1, 4))

            paths = frame_paths(out, str(i))
            paths.velodyne.parent.mkdir(parents=True, exist_ok=True)
            np.asarray(velo, dtype=np.float32).reshape(-1, 4).tofile(paths.velodyne)
            write_pose(out, str(i), T)
            if (i + 1) % 50 == 0 or i == args.frames - 1:
                print(
                    f"[frame {i + 1}/{args.frames}] ego @ {tuple(round(v, 1) for v in loc(ego_t))} "
                    f"| {len(velo)} 点"
                )
    finally:
        lidar.stop()
        lidar.destroy()
        ego.destroy()
        for a in world.get_actors():
            if a.type_id.startswith("vehicle") or a.type_id.startswith("walker"):
                a.destroy()

    # 轨迹摘要(不冒充精度指标:这是**真值**轨迹的几何,供选数据/判回环用)
    P = np.array([T[:3, 3] for T in poses])
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    path_len = float(step.sum())
    yaw = np.array([np.arctan2(T[1, 0], T[0, 0]) for T in poses])
    summary = {
        "dataset": str(args.out),
        "map": world.get_map().name,
        "n_frames": len(poses),
        "mode": "straight" if args.speed > 0 else "autopilot",
        "speed_set": args.speed,
        "path_len_m": round(path_len, 3),
        "start_to_end_m": round(float(np.linalg.norm(P[-1] - P[0])), 3),
        "step_mean_m": round(float(step.mean()), 4),
        "step_median_m": round(float(np.median(step)), 4),
        "step_max_m": round(float(step.max()), 4),
        "yaw_total_deg": round(float(np.degrees(yaw[-1] - yaw[0])), 3),
        "bbox_min": [round(float(x), 2) for x in P.min(0)],
        "bbox_max": [round(float(x), 2) for x in P.max(0)],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "slam_seq_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] {len(poses)} 帧 | 路径 {path_len:.1f} m | 首末 {summary['start_to_end_m']} m")
    print(
        f"  步长 mean {summary['step_mean_m']} / max {summary['step_max_m']} m | yaw 总变化 {summary['yaw_total_deg']}°"
    )
    print(f"  → {out}/training/velodyne + pose,摘要 {out}/slam_seq_summary.json")


if __name__ == "__main__":
    main()

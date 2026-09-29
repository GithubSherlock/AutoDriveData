"""多传感器采集:**6 环视相机 + LiDAR + 5 雷达 + ego 真值位姿**,一次过同步落盘到同一个 root。

## 为什么必须单独一个采集器

各条下游链各要各的,而**磁盘上没有任何一份数据集同时有它们**:

| 链 | 要什么 | 现由谁产出 |
|---|---|---|
| 环视 / MapTR | `cam_*/{fid}.png` + `calib.json` + `ego_pose.json` | `collect_surround` |
| SLAM | `training/velodyne/{fid}.bin` + `training/pose/{fid}.txt` | `collect_slam` |
| 雷达(devkit) | `samples/RADAR_*/*.pcd`(18 字段) | `collect_nus` |

三者**互不支持对方的传感器**(2026-09-28 实测确认)。于是「逐帧环视预测 + SLAM 位姿」
这种组合**根本没有数据可跑** —— `map/stitch_temporal.py --pose slam` 就卡在这。
本采集器把三套落盘写进**同一个 root**,一次采集同时喂三条链。

## 落盘(一个 root,**三种目录布局**)

```
# ① 环视/MapTR 形状(消费方 = assemble_maptr)
<out>/cam_front/000000.png … cam_back_right/000000.png   6 视角(1600×900,逐通道 fov)
<out>/calib.json            逐相机 sensor2ego + intrinsic
<out>/ego_pose.json         逐帧 ego2global
# ② KITTI 形状(消费方 = slam_odometry / eval_slam)
<out>/training/velodyne/000000.bin   KITTI 口径点云
<out>/training/pose/000000.txt       KITTI 12 数位姿(ATE/RPE 的 GT)
# ③ nuScenes devkit 形状(消费方 = AutoLabel 雷达线 / smoke_radar_collect.sh 四判据)
<out>/samples/RADAR_FRONT/000000.pcd …(5 通道,18 字段 nus 点)
```

⚠️ **为什么雷达不合进 ② 的 KITTI 形状**(三种布局共存是**刻意**的,不是没整理):
雷达的消费方是 devkit 口径(AutoLabel 的 `num_radar_pts` 关联、`smoke_radar_collect.sh`
的四条直读判据),它按 **18 字段 nus 点**解释数据;写进 KITTI 形状就没有消费方。
**相机 / LiDAR / 雷达各走各自已有消费方的形状** —— 强行统一成一种,反而三边都不认。

⇒ 下面是**同一段数据**上成立的四条命令(这正是本采集器存在的理由):

```bash
python -m autodrivedata.map.assemble_maptr --surround <out> --map-json training/map/Town10HD_Opt_full.json --out <out>/map_infos.json
python -m autodrivedata.slam.slam_odometry --root <out> --frames 0-N --out <slam_out>
python -m autodrivedata.slam.slam_backend  --traj <slam_out>/traj_raw.json --root <out> --out <slam_out>
python -m autodrivedata.map.stitch_temporal --frames <preds> --infos <out>/map_infos.json \
    --pose slam --slam-traj <slam_out>/traj_pgo.json --out <map_out>
```

## ⚠️ 同步纪律(四条都踩过,别改)

1. **每个 tick 必须逐路抽干队列**。只取被测那一路会让其余积压,下一帧读到的是**更早**的
   陈旧图 —— 症状是"六路里只有第一路对"(§P-M.7 判据 ⑥ 同款坑)。
2. **stride > 1 时中间 tick 也要收齐**。只收最后一 tick 的队头,拿到的不是当帧。
3. **传感器 `get_transform()` 只在 tick 后刷新** —— tick 前读到的是全 0 陈旧值。
4. **雷达有相位空 tick**(`sensor_tick=0.1` 与同步 tick 同周期,实测 ~7%)⇒ 它必须走
   **非阻塞 drain**、空帧写**空 pcd**(devkit 空编码 (18,0)),**不能用阻塞 `get`** ——
   不 tick 数据永远不会来,阻塞就是死等(`collect_nus` 的 `_latest` 就是为此而写,这里复用)。

## 定速口径(照 `collect_ab_route` 的红线,别自创)

`VehicleControl` 一旦设置就**每步生效**:站定用的 `brake=1.0` 不解除,`set_target_velocity`
会打 **0.82 折**(实测命令 8 → 6.60 m/s,P1 四个老数据集全中)。故:先 `apply_control(空)`
清残留,再每 tick 强设速度,并在收尾**用位姿序列反算逐帧实测速度**打印出来 ——
"定速"必须逐帧自证,不能只看命令值。

用法:
  python -m autodrivedata.sim.collect_surround_lidar --frames 200 --out outputs/dual_town10
  python -m autodrivedata.sim.collect_surround_lidar --frames 200 --speed 6 --stride 2 \
      --spawn-index 88 --scene day_clear
  python -m autodrivedata.sim.collect_surround_lidar --frames 50 --no-radar      # 不挂雷达(轻量跑)
"""

from __future__ import annotations

import argparse
import json
import queue
import time
from typing import Any, cast

import carla
import numpy as np

from autodrivedata.calib.camera_rig import NUS_CAMERA_RIG
from autodrivedata.gt.export.kitti import frame_paths, write_pose
from autodrivedata.gt.export.nuscenes import (
    NUS_CAMERA_FOV,
    NUS_CAMERA_HEIGHT,
    NUS_CAMERA_WIDTH,
    NUS_RADAR_CHANNELS,
    NUS_RADAR_MOUNTS_CARLA,
    NUS_RADAR_OFFSETS,
)
from autodrivedata.map.mapviz import calib_from_fov
from autodrivedata.perception.radar import detections_to_nus18, mask_radar_points, nus18_to_pcd
from autodrivedata.sim.carla_common import (
    LIDAR_ATTRS,
    loc,
    spawn_ego,
    spawn_ego_at,
    sync_mode,
)
from autodrivedata.sim.collect_drive import spawn_route_walkers, spawn_traffic
from autodrivedata.sim.collect_nus import RADAR_ATTRS, RADAR_YAW_OFFSET, _latest
from autodrivedata.sim.collect_slam import ego_pose_matrix
from autodrivedata.sim.collect_surround import SURROUND_CAM_ATTRS, SURROUND_CAMS
from autodrivedata.sim.scenarios import SCENES, merged_weather
from autodrivedata.utils.geometry import carla_lidar_to_velodyne
from autodrivedata.utils.paths import project_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出 root(经 project_path)")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=5, help="每帧跨几个 tick(5 = 0.5 s/帧)")
    ap.add_argument("--speed", type=float, default=8.0, help="ego 定速 m/s(红线:定速要逐帧自证)")
    ap.add_argument("--map", default=None, help="加载地图(缺省=不动当前图)")
    ap.add_argument("--scene", default=None, choices=sorted(SCENES), help="天气档")
    ap.add_argument("--spawn-index", type=int, default=None, help="固定用第 N 个 spawn point")
    ap.add_argument("--npc-vehicles", type=int, default=15)
    ap.add_argument("--npc-walkers", type=int, default=6)
    ap.add_argument("--route-walkers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--semantic-lidar", action="store_true", help="用 ray_cast_semantic(32 线 + tag)")
    ap.add_argument(
        "--no-radar",
        action="store_true",
        help="不挂 5 雷达(默认挂;雷达走 devkit 形状 samples/RADAR_*/*.pcd)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.load_world(args.map) if args.map else client.get_world()
    default_map = args.map or world.get_map().name
    sync_mode(world)
    scene = SCENES[args.scene] if args.scene else None
    if scene is not None:
        world.set_weather(carla.WeatherParameters(**merged_weather(scene)))
        print(f"[scene] {scene.name} — 覆写 {sorted(scene.weather)}")

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    ego = spawn_ego_at(world, args.spawn_index) if args.spawn_index is not None else spawn_ego(world)
    ego.set_autopilot(False)  # 定速,不走 TM(红线:autopilot 路线不可复现)
    spawn_traffic(world, tm, args.npc_vehicles, args.npc_walkers, args.seed)
    spawn_route_walkers(world, ego.get_transform(), args.route_walkers)
    bp_lib = world.get_blueprint_library()

    # ---- 传感器:6 环视相机(逐通道 fov,与 collect_surround 同一份 rig 表)----
    cams: dict[str, carla.Sensor] = {}
    qs: dict[str, queue.Queue] = {}
    for name, (mount, rot) in NUS_CAMERA_RIG.items():
        cam_bp = bp_lib.find("sensor.camera.rgb")
        for k, v in SURROUND_CAM_ATTRS.items():
            cam_bp.set_attribute(k, v)
        cam_bp.set_attribute("fov", f"{NUS_CAMERA_FOV[name]:.6f}")
        s = cast(
            carla.Sensor,
            world.spawn_actor(
                cam_bp,
                carla.Transform(
                    carla.Location(x=mount[0], y=mount[1], z=mount[2]),
                    carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]),
                ),
                attach_to=ego,
            ),
        )
        q: queue.Queue = queue.Queue()
        s.listen(q.put)
        cams[name], qs[name] = s, q

    # ---- LiDAR:与 collect_slam 同口径(那边是 SLAM 链已验过的参数)----
    lid_bp = bp_lib.find("sensor.lidar.ray_cast_semantic" if args.semantic_lidar else "sensor.lidar.ray_cast")
    for k, v in LIDAR_ATTRS.items():
        lid_bp.set_attribute(k, v)
    lidar = cast(carla.Sensor, world.spawn_actor(lid_bp, carla.Transform(), attach_to=ego))
    lid_q: queue.Queue = queue.Queue()
    lidar.listen(lid_q.put)

    # ---- 5 雷达:照官方 nuScenes 布局(devkit 形状,见模块头注)----
    # 位姿与偏航**由 `NUS_RADAR_*` 导出**(不是自己写的表):`collect_nus` 那套已经过
    # 十条判据验收(判据②雷达实挂 vs 官方 az / ③点落进自身 FOV 占比),这里只复用。
    radars: dict[str, carla.Sensor] = {}
    radar_qs: dict[str, queue.Queue] = {ch: queue.Queue() for ch in NUS_RADAR_CHANNELS}
    if not args.no_radar:
        for ch in NUS_RADAR_CHANNELS:
            r_bp = bp_lib.find("sensor.other.radar")
            for k, v in RADAR_ATTRS.items():
                r_bp.set_attribute(k, v)
            s = cast(
                carla.Sensor,
                world.spawn_actor(
                    r_bp,
                    carla.Transform(
                        carla.Location(*NUS_RADAR_MOUNTS_CARLA[ch]),
                        carla.Rotation(pitch=0.0, yaw=RADAR_YAW_OFFSET[ch], roll=0.0),
                    ),
                    attach_to=ego,
                ),
            )
            s.listen(radar_qs[ch].put)
            radars[ch] = s
    print(
        f"[sensors] {len(cams)} 相机 @ {NUS_CAMERA_WIDTH}×{NUS_CAMERA_HEIGHT} + LiDAR"
        f"({LIDAR_ATTRS['channels']} 线,semantic={args.semantic_lidar})"
        f" + {len(radars)} 雷达(ars408,{'关' if args.no_radar else 'devkit 形状落盘'})"
    )

    out = project_path(args.out)
    for name in SURROUND_CAMS:
        (out / name.lower()).mkdir(parents=True, exist_ok=True)
    calib: dict[str, Any] = {
        name: {
            "sensor2ego": [mount[0], mount[1], mount[2], rot[1], rot[0], rot[2]],
            "intrinsic": calib_from_fov(NUS_CAMERA_WIDTH, NUS_CAMERA_HEIGHT, NUS_CAMERA_FOV[name])[
                "intrinsic"
            ],
        }
        for name, (mount, rot) in NUS_CAMERA_RIG.items()
    }
    calib.update(
        {
            "map": default_map,
            "spawn_index": args.spawn_index,
            "stride": args.stride,
            "image_size": [NUS_CAMERA_WIDTH, NUS_CAMERA_HEIGHT],
            "route": "const_speed",
            "speed_mps": args.speed,
            "has_lidar": True,  # 溯源:本 root 与 collect_surround 的产物靠这个键区分
            "has_radar": not args.no_radar,
            # 雷达挂点/偏航是**官方 nus 原值**(devkit 口径),不是 CARLA 系 —— 与相机那套
            # (sensor2ego,CARLA 系)不同系,故单独一个键,免得被当成同一种读数。
            "radar_nus_offsets": (
                {ch: list(NUS_RADAR_OFFSETS[ch][0]) for ch in NUS_RADAR_CHANNELS}
                if not args.no_radar
                else None
            ),
        }
    )
    (out / "calib.json").write_text(json.dumps(calib, indent=1), encoding="utf-8")

    # 预热:**每 tick 收齐全部队列**(纪律 1/2)。雷达 sensor_tick 相位偶发空(实测 ~7%),
    # 故它走**非阻塞 drain**、以"每个通道至少出过一帧"为就绪判据(最多 10 tick);
    # 相机/LiDAR 每 tick 必有,仍用阻塞 get 保时序。
    warmed_radar: dict[str, bool] = dict.fromkeys(NUS_RADAR_CHANNELS, False)
    for _ in range(10):
        world.tick()
        for q in qs.values():
            q.get(timeout=10)
        lid_q.get(timeout=10)
        for ch, q in radar_qs.items():
            if not args.no_radar and _latest(q) is not None:
                warmed_radar[ch] = True
        if args.no_radar or all(warmed_radar.values()):
            break
    if not args.no_radar and not all(warmed_radar.values()):
        print(f"  [warn] 预热 10 tick 仍有雷达通道没出帧:{[k for k, v in warmed_radar.items() if not v]}")

    # 清制动残留再定速(红线:`VehicleControl` 残留会让 set_target_velocity 打折)
    ego.apply_control(carla.VehicleControl())
    fwd = ego.get_transform().get_forward_vector()
    fwd_v = carla.Vector3D(x=fwd.x * args.speed, y=fwd.y * args.speed, z=0.0)

    poses: list[dict] = []
    Ts: list[np.ndarray] = []
    t0 = time.monotonic()
    ok = False
    i = -1  # 异常可能在首轮之前抛出;半成品标记要用它,别留未绑定名
    try:
        for i in range(args.frames):
            ego.set_target_velocity(fwd_v)  # 每 tick 强设(速度环不稳)
            drained: dict[str, carla.Image] = {}
            lid: carla.LidarMeasurement | None = None
            for _ in range(args.stride):
                world.tick()
                drained = {name: qs[name].get(timeout=10) for name in SURROUND_CAMS}
                lid = lid_q.get(timeout=10)
            for name, image in drained.items():
                tmp = out / f".tmp_{i}_{name}.png"
                image.save_to_disk(str(tmp))
                tmp.rename(out / name.lower() / f"{i:06d}.png")
            # 5 雷达 → devkit 形状 `samples/RADAR_*/*.pcd`(18 字段 nus 点)。
            # **非阻塞 drain**:相位空 tick 是常态(实测 ~7%),阻塞 get 会死等;
            # 空通道写**空 pcd**(devkit 空编码 (18,0)),不阻断采集 —— 与 `collect_nus` 同口径。
            for ch in NUS_RADAR_CHANNELS:
                if args.no_radar:
                    break
                frame = _latest(radar_qs[ch])
                if frame is None:
                    nus18 = np.empty((0, 18), dtype=np.float32)
                else:
                    dets = np.frombuffer(frame.raw_data, dtype=np.float32).reshape(-1, 4)
                    nus18 = detections_to_nus18(dets, sensor_id=NUS_RADAR_CHANNELS.index(ch))
                    nus18 = nus18[mask_radar_points(nus18)]
                rel = out / "samples" / ch / f"{i:06d}.pcd"
                rel.parent.mkdir(parents=True, exist_ok=True)
                rel.write_bytes(nus18_to_pcd(nus18))
            assert lid is not None
            raw = np.frombuffer(lid.raw_data, dtype=np.float32)
            if args.semantic_lidar:
                from autodrivedata.perception.semantic import semantic_to_velodyne_bin

                velo = semantic_to_velodyne_bin(raw.reshape(-1, 6), seed=args.seed * 1000 + i)
            else:
                velo = carla_lidar_to_velodyne(raw.reshape(-1, 4))
            paths = frame_paths(out, str(i))
            paths.velodyne.parent.mkdir(parents=True, exist_ok=True)
            np.asarray(velo, dtype=np.float32).reshape(-1, 4).tofile(paths.velodyne)
            egot = ego.get_transform()
            T = ego_pose_matrix(egot)  # 与 collect_slam 同一个函数:别自拼矩阵(序易错)
            write_pose(out, str(i), T)
            Ts.append(T)
            poses.append(
                {
                    "frame": i,
                    "tick": i * args.stride,
                    "x": round(egot.location.x, 3),
                    "y": round(egot.location.y, 3),
                    "z": round(egot.location.z, 3),
                    "yaw": round(egot.rotation.yaw, 3),
                    "pitch": round(egot.rotation.pitch, 3),
                    "roll": round(egot.rotation.roll, 3),
                }
            )
            if (i + 1) % 20 == 0 or i == args.frames - 1:
                fps = (i + 1) / (time.monotonic() - t0)
                print(
                    f"[frame {i + 1}/{args.frames}] ego @ {tuple(round(v, 1) for v in loc(egot))}"
                    f" | {len(velo)} 点 | {fps:.1f} fps"
                )
        ok = True
    except RuntimeError as e:
        # 与 collect_slam 同款:半成品必须留记号,否则看起来和完整数据集一样
        (out / "ABORTED.json").write_text(
            json.dumps({"frame": i, "n_frames_written": i + 1, "reason": str(e)}, indent=2),
            encoding="utf-8",
        )
        print(f"[abort] 半成品标记 → {out}/ABORTED.json —— 该目录**不可**当数据集用")
        raise
    finally:
        (out / "ego_pose.json").write_text(json.dumps(poses, indent=1), encoding="utf-8")
        # **雷达必须一起销毁**(2026-09-29 首次真跑漏了):5 个 radar actor 留在世界里,
        # 客户端收尾时报 5 条 `sensor object went out of the scope but the sensor is still
        # alive` 并 core dump。数据当时已完整落盘,但"崩了"会让人误以为采集失败 ——
        # 这类"收尾崩溃掩盖成功"正是 `tools/carla_server.sh` 头注记的同款现象。
        for s in (*cams.values(), lidar, *radars.values()):
            s.stop()
            s.destroy()
        for a in world.get_actors():
            if a.type_id.startswith(("vehicle", "walker", "controller")):
                a.destroy()

    if ok:
        # **定速必须逐帧自证**(红线):从位姿序列反算实测速度,不看命令值。
        # 命令 8 而实测 6.6 就是"制动残留没清"的典型症状(0.82 折)。
        dt = args.stride * 0.1
        p = np.array([T[:3, 3] for T in Ts])
        spd = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt if len(p) > 1 else np.zeros(1)
        print(
            f"[speed] 命令 {args.speed:.2f} m/s → 逐帧实测 中位 {np.median(spd):.2f} / "
            f"均值 {spd.mean():.2f} m/s(取自位姿序列,非命令值)"
        )
    print(
        f"[done] dual root: {out.resolve()} ({args.frames} 帧 × {len(cams)} 相机 + LiDAR,"
        f" stride {args.stride} = {args.frames * args.stride * 0.1:.1f}s 仿真时长)"
    )


if __name__ == "__main__":
    main()

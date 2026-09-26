"""P1-3 单变量 A/B 采集:ego 定速直行 + 路侧静置车(固定位置)→ KITTI 序列。

A/B 纪律(2026-09-08 教训):autopilot/TM 路线失控使帧内容不可对齐(两次
同 seed 采集 GT 分布差 7 倍),场景级比较无法归因。本采集器:
- ego 锚定 spawn point 0(yaw=0,朝世界 +x)+ set_target_velocity 定速直行(无 TM)
- 静置目标车放固定路肩位置(20/35/50/62m,+3.5m 横向),不 autopilot
- A/B 只变 weather(SCENES 场景档,sunset_glare 的 az=90=车头正前 经校准)
→ 帧级配对:同位置同车同角,唯一变量 = 光照(采集史:残留 actor 阻塞
pt0 曾致 fallback 反向出生点、65m 曾卡 GT 阈值——已修,详见 Plan.md §5.5a)。

用法: python -m autodrivedata.sim.collect_ab_route --scene day_clear --frames 220 [--out ...]
      python -m autodrivedata.sim.collect_ab_route --scene sunset_glare --frames 220 [--out ...]
      python -m autodrivedata.sim.collect_ab_route --scene day_clear --speed 4 --frames 140  # 参数扫描(定里程)
"""

from __future__ import annotations

import argparse
import queue
from typing import cast

import carla
import numpy as np

from autodrivedata.calib.core import CameraIntrinsics, KittiCalibOut, tr_velo_to_cam
from autodrivedata.export.kitti import write_frame
from autodrivedata.gt import ActorBox, box_to_gt_line
from autodrivedata.paths import project_path
from autodrivedata.perception.semantic import semantic_to_velodyne_bin
from autodrivedata.sim.carla_common import (
    CAM_ATTRS,
    LIDAR_ATTRS,
    SENSOR_OFFSET,
    loc,
    rad,
    spawn_ego,
    sync_mode,
)
from autodrivedata.sim.scenarios import SCENES, merged_weather

SPEED = 8.0  # m/s 定速
# 2026-09-09:65.0→62.0——第4台车曾恰好卡 GT max_distance=65.0 边界,起步
# 抖动使两侧 GT 数不等(216 vs 219),帧级配对失效;留 3m 裕量
STATIC_OFFSETS = [
    20.0,
    35.0,
    50.0,
    62.0,
]  # 路肩车沿 x 前向距离(m)(-y 侧 spawn 失败,全用 +y 路肩)
STATIC_LATS = [3.5] * 4  # 横向偏移(同侧路肩)
NPC_MODELS = [
    "vehicle.tesla.model3",
    "vehicle.audi.a2",
    "vehicle.ford.mustang",
    "vehicle.toyota.prius",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=sorted(SCENES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=70)  # 70帧≈56m,距第4车9m刹停
    ap.add_argument("--speed", type=float, default=SPEED, help="ego 定速 m/s(参数扫描用;默认= P1 基线 8.0)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()
    scene = SCENES[args.scene]

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)

    # 清场残留 actor:校准脚本曾只毁相机不毁 ego,残留车阻塞 pt0 致
    # spawn_ego fallback 到反向 spawn 点(yaw≈-180,朝 -x)→ A/B 起点失配
    for a in world.get_actors():
        if a.type_id.startswith(("vehicle", "walker", "controller")):
            a.destroy()
    for _ in range(3):
        world.tick()

    world.set_weather(carla.WeatherParameters(**merged_weather(scene)))
    print(f"[scene] {scene.name} — {sorted(scene.weather)}")

    ego = spawn_ego(world)
    ego.set_autopilot(False)
    # 强制锚定 pts[0](yaw=0 朝世界 +x):出生即复现,不依赖 spawn_ego 的
    # 空位探测;锚定后校验,失败立即报错而非静默走错路线
    spawn_pts = world.get_map().get_spawn_points()
    ego.set_transform(carla.Transform(spawn_pts[0].location, carla.Rotation(yaw=0.0)))
    world.tick()
    here = ego.get_location()
    if here.distance(spawn_pts[0].location) > 1.0:
        raise RuntimeError(f"ego 未能锚定 pts[0]: 落在 {here}")
    print(
        f"[ego] 锚定 pts[0] @ ({here.x:.1f}, {here.y:.1f}) 偏航 {ego.get_transform().rotation.yaw:.1f}° (朝世界 +x)"
    )
    ego.apply_control(carla.VehicleControl(brake=1.0))  # 站定
    world.tick()

    # 静置目标车:ego 前方固定路肩位置
    bp_lib = world.get_blueprint_library()
    start = ego.get_location()
    fwd = ego.get_transform().get_forward_vector()
    right = ego.get_transform().get_right_vector()
    placed = []
    for d, lat, i in zip(STATIC_OFFSETS, STATIC_LATS, range(len(STATIC_OFFSETS)), strict=True):
        p = start + fwd * d + right * lat
        pos = carla.Location(x=p.x, y=p.y, z=start.z)
        bp = bp_lib.find(NPC_MODELS[i % len(NPC_MODELS)])
        v = world.try_spawn_actor(
            bp,
            carla.Transform(pos, carla.Rotation(yaw=ego.get_transform().rotation.yaw)),
        )
        if v is not None:
            veh = cast(carla.Vehicle, v)  # carla pyi 桩:try_spawn_actor 标返回 Actor(实为 Actor|None)
            veh.apply_control(carla.VehicleControl(brake=1.0))  # 静置
            placed.append(veh)
    world.tick()
    print(
        f"[statics] {len(placed)}/{len(STATIC_OFFSETS)} 路肩车 @ {[(round((s.get_location() - start).x), round((s.get_location() - start).y)) for s in placed]}"
    )

    # 传感器同 collect_drive
    cam_bp = bp_lib.find("sensor.camera.rgb")
    for k, v in CAM_ATTRS.items():
        cam_bp.set_attribute(k, v)
    lid_bp = bp_lib.find("sensor.lidar.ray_cast_semantic")
    for k, v in LIDAR_ATTRS.items():
        lid_bp.set_attribute(k, v)
    camera = cast(carla.Sensor, world.spawn_actor(cam_bp, SENSOR_OFFSET, attach_to=ego))
    lidar = cast(carla.Sensor, world.spawn_actor(lid_bp, SENSOR_OFFSET, attach_to=ego))
    img_q: queue.Queue = queue.Queue()
    lid_q: queue.Queue = queue.Queue()
    camera.listen(img_q.put)
    lidar.listen(lid_q.put)

    out = project_path(args.out or f"outputs/kitti_ab_{scene.name}")
    k = CameraIntrinsics(
        width=int(CAM_ATTRS["image_size_x"]),
        height=int(CAM_ATTRS["image_size_y"]),
        fov_h_deg=float(CAM_ATTRS["fov"]),
    )

    # 解除"站定"制动:VehicleControl 一旦设置就每步生效,brake=1.0 残留会让
    # set_target_velocity 打折扣(2026-09-09 实测:命令 8 → 实际 6.59 m/s = 0.82×;
    # P1 四个 A/B 数据集实测均为 6.60 m/s,"定速"名不副实且 TTC 归一化偏 18%)
    ego.apply_control(carla.VehicleControl())
    fwd_v = carla.Vector3D(x=fwd.x * args.speed, y=fwd.y * args.speed, z=0.0)
    try:
        for i in range(args.frames):
            # 定速:每 tick 强设速度(车辆控制速度环不稳,直接 velocity)
            ego.set_target_velocity(fwd_v)
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
            velo = semantic_to_velodyne_bin(raw.reshape(-1, 6), seed=args.frames * 100 + i)
            write_frame(
                out,
                str(i),
                image_png=png,
                velodyne=velo,
                calib=calib_out,
                labels=labels,
            )
            if (i + 1) % 25 == 0 or i == args.frames - 1:
                print(f"[frame {i + 1}/{args.frames}] ego x={ego.get_location().x:8.1f} | {len(labels)} GT")
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
    print(f"[done] KITTI root: {out.resolve()} ({args.frames} frames)")


if __name__ == "__main__":
    main()

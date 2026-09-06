"""M1a 静态采集:ego 静止 + 摆 NPC + 相机/LiDAR 同步模式 → KITTI root 落盘(raw + GT)。

用法(base env,CARLA 服务器运行中):
  python scripts/collect_kitti.py [--out outputs/kitti_scene] [--frames 10] [--host 127.0.0.1]

落盘布局照 auto3dlabel KittiFrame 契约(验收时 KITTI_OBJECT_ROOT=--out 零改动读入):
  {out}/training/{image_2,velodyne,calib,label_2}/000000.*
GT = 动态 actor(KITTI 类名);点云落盘前做 y 翻转对齐 KITTI velodyne 约定(见 geometry.py)。
"""
from __future__ import annotations

import argparse
import queue
from pathlib import Path

import carla
import numpy as np

from autodrivedata import geometry as g
from autodrivedata.calib import CameraIntrinsics, KittiCalibOut, tr_velo_to_cam
from autodrivedata.export.kitti import write_frame
from autodrivedata.gt import ActorBox, box_to_gt_line

# 相机对齐 KITTI 口径(1242×375);LiDAR 64 线(与 KITTI velodyne 一致)
CAM_ATTRS = {"image_size_x": "1242", "image_size_y": "375", "fov": "90"}
LIDAR_ATTRS = {
    "channels": "64",
    "range": "70",
    "points_per_second": "200000",
    "rotation_frequency": "10",
    "upper_fov": "10.0",
    "lower_fov": "-30.0",
}
SENSOR_OFFSET = carla.Transform(carla.Location(1.2, 0.0, 1.65))  # 相对 ego 的车顶前装


def _rad(rot: carla.Rotation) -> tuple[float, float, float]:
    """carla.Rotation(度)→ (pitch, yaw, roll) 弧度。"""
    return tuple(np.radians(a) for a in (rot.pitch, rot.yaw, rot.roll))


def _loc(t: carla.Transform) -> tuple[float, float, float]:
    return (t.location.x, t.location.y, t.location.z)


def spawn_npcs(world: carla.World, ego_t: carla.Transform) -> None:
    """ego 前方摆 NPC:同向车 ×2、对向车 ×1、行人 ×2、骑行者 ×1(碰撞失败仅告警)。"""
    fwd = ego_t.get_forward_vector()
    right = ego_t.get_right_vector()
    ego_yaw = ego_t.rotation.yaw

    def place(d: float, off: float, yaw: float, z_off: float = 0.0) -> carla.Transform:
        loc = ego_t.location + fwd * d + right * off
        loc.z += z_off
        return carla.Transform(loc, carla.Rotation(yaw=yaw, pitch=0.0, roll=0.0))

    specs = [
        ("vehicle.tesla.model3", place(12.0, 0.0, ego_yaw)),          # 同车道前车
        ("vehicle.audi.a2", place(22.0, 2.2, ego_yaw)),               # 右邻车道
        ("vehicle.ford.mustang", place(30.0, -3.2, ego_yaw + 180)),   # 对向车
        ("walker.pedestrian.0001", place(8.0, 3.2, ego_yaw)),         # 右侧行人
        ("walker.pedestrian.0002", place(14.0, -3.2, ego_yaw + 90)),  # 左侧行人(面向车道)
        ("vehicle.gazelle.omafiets", place(18.0, 3.6, ego_yaw)),      # 右侧骑行者
    ]
    bp_lib = world.get_blueprint_library()
    for type_id, tf in specs:
        bp = bp_lib.find(type_id)
        actor = world.try_spawn_actor(bp, tf)
        if actor is None:
            print(f"  [warn] NPC spawn 失败(碰撞): {type_id}")
            continue
        if type_id.startswith("walker"):
            ctrl = world.spawn_actor(bp_lib.find("controller.ai.walker"), carla.Transform(), actor)
            ctrl.start()  # 站立不动;M2 再给行走指令
        print(f"  [npc] {type_id} @ {_loc(tf)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/kitti_scene")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.find("vehicle.audi.a2")
    # 出生点逐个尝试(碰撞则换下一个),取第一个成功者
    ego: carla.Vehicle | None = None
    for pt in world.get_map().get_spawn_points():
        ego = world.try_spawn_actor(ego_bp, pt)
        if ego is not None:
            break
    if ego is None:
        raise RuntimeError("所有出生点均 spawn 失败(碰撞)")
    # 同步模式红线:spawn 后必须 tick,actor 位姿才同步到客户端
    # (实测:不 tick 则 get_transform 返回恒等变换 (0,0,0)——NPC 全摆到原点)
    world.tick()
    ego_t = ego.get_transform()
    print(f"[ego] vehicle.audi.a2 @ {_loc(ego_t)}")

    cam_bp = bp_lib.find("sensor.camera.rgb")
    for k, v in CAM_ATTRS.items():
        cam_bp.set_attribute(k, v)
    lid_bp = bp_lib.find("sensor.lidar.ray_cast")
    for k, v in LIDAR_ATTRS.items():
        lid_bp.set_attribute(k, v)
    camera = world.spawn_actor(cam_bp, SENSOR_OFFSET, attach_to=ego)
    lidar = world.spawn_actor(lid_bp, SENSOR_OFFSET, attach_to=ego)
    print(f"[sensor] camera {CAM_ATTRS['image_size_x']}x{CAM_ATTRS['image_size_y']} fov={CAM_ATTRS['fov']} + lidar {LIDAR_ATTRS['channels']}ch")

    spawn_npcs(world, ego_t)

    img_q: "queue.Queue" = queue.Queue()
    lid_q: "queue.Queue" = queue.Queue()
    camera.listen(img_q.put)
    lidar.listen(lid_q.put)

    # 预热:丢前几 tick(传感器首帧常不完整),清空队列
    for _ in range(5):
        world.tick()
        img_q.get(timeout=10)
        lid_q.get(timeout=10)

    k = CameraIntrinsics(width=int(CAM_ATTRS["image_size_x"]), height=int(CAM_ATTRS["image_size_y"]), fov_h_deg=float(CAM_ATTRS["fov"]))
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
                    _loc(lid_t), _rad(lid_t.rotation), _loc(cam_t), _rad(cam_t.rotation)
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
                    rotation=_rad(bb.rotation),
                    actor_location=_loc(a.get_transform()),
                    actor_rotation=_rad(a.get_transform().rotation),
                )
                line = box_to_gt_line(box, _loc(cam_t), _rad(cam_t.rotation), k)
                if line:
                    labels.append(line)

            tmp = out / f".tmp_{i}.png"
            image.save_to_disk(str(tmp))
            png = tmp.read_bytes()
            tmp.unlink()

            velo = g.carla_lidar_to_velodyne(
                np.frombuffer(pts.raw_data, dtype=np.float32).reshape(-1, 4)
            )
            paths = write_frame(out, str(i), image_png=png, velodyne=velo, calib=calib_out, labels=labels)
            print(f"[frame {i}] {len(labels)} GT / {len(velo)} pts -> {paths.image.name}")
    finally:
        camera.stop()
        lidar.stop()
        camera.destroy()
        lidar.destroy()
        for a in world.get_actors():
            if a.type_id.startswith("vehicle") or a.type_id.startswith("walker") or a.type_id.startswith("controller"):
                a.destroy()
    print(f"[done] KITTI root: {out.resolve()}")


if __name__ == "__main__":
    main()

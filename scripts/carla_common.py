"""CARLA 采集公共件(base env,依赖 pycarla):位姿换算 / NPC 摆放 / 传感器参数。"""

from __future__ import annotations

from typing import Protocol, cast

import carla
import numpy as np


class _WalkerCtrl(Protocol):
    """pycarla 桩缺 WalkerAIController 方法标注的最小协议。"""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def set_max_speed(self, speed: float) -> None: ...
    def walk_to_location(self, destination: carla.Location) -> None: ...

# 相机对齐 KITTI 口径(1242×375);LiDAR 64 线(与 KITTI velodyne 一致)
CAM_ATTRS = {"image_size_x": "1242", "image_size_y": "375", "fov": "90"}
LIDAR_ATTRS = {
    "channels": "64",
    "range": "70",
    # 1.3M pps = 真实 HDL-64E 量级(实测每帧 63k 点、360° 全覆盖);
    # 200k pps 时车只有 13~117 点,PointPillars 体素特征不足(M1a-7 实测教训)
    "points_per_second": "1300000",
    "rotation_frequency": "10",
    "upper_fov": "10.0",
    "lower_fov": "-30.0",
}
SENSOR_OFFSET = carla.Transform(carla.Location(1.2, 0.0, 1.65))  # 相对 ego 的车顶前装


def rad(rot: carla.Rotation) -> tuple[float, float, float]:
    """carla.Rotation(度)→ (pitch, yaw, roll) 弧度。"""
    return tuple(np.radians(a) for a in (rot.pitch, rot.yaw, rot.roll))


def loc(t: carla.Transform) -> tuple[float, float, float]:
    return (t.location.x, t.location.y, t.location.z)


def spawn_ego(world: carla.World) -> carla.Vehicle:
    """出生点逐个尝试(碰撞则换下一个),spawn 后 tick 同步位姿。"""
    bp = world.get_blueprint_library().find("vehicle.audi.a2")
    ego: carla.Actor | None = None
    for pt in world.get_map().get_spawn_points():
        ego = world.try_spawn_actor(bp, pt)
        if ego is not None:
            break
    if ego is None:
        raise RuntimeError("所有出生点均 spawn 失败(碰撞)")
    # 同步模式红线:spawn 后必须 tick,actor 位姿才同步到客户端
    # (实测:不 tick 则 get_transform 返回恒等变换 (0,0,0))
    world.tick()
    return cast(carla.Vehicle, ego)


def spawn_npcs(world: carla.World, ego_t: carla.Transform) -> None:
    """ego 前方摆 NPC:同向车 ×2、对向车 ×1、行人 ×2、骑行者 ×1(碰撞失败仅告警)。"""
    fwd = ego_t.get_forward_vector()
    right = ego_t.get_right_vector()
    ego_yaw = ego_t.rotation.yaw

    def place(d: float, off: float, yaw: float, z_off: float = 0.0) -> carla.Transform:
        v = ego_t.location + fwd * d + right * off
        pos = carla.Location(x=v.x, y=v.y, z=v.z + z_off)
        return carla.Transform(pos, carla.Rotation(yaw=yaw, pitch=0.0, roll=0.0))

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
            cast(_WalkerCtrl, ctrl).start()  # 站立不动;动态场景再给行走指令
        print(f"  [npc] {type_id} @ {loc(tf)}")


def sync_mode(world: carla.World, delta: float = 0.1) -> None:
    """开同步模式(固定 tick)。"""
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = delta
    world.apply_settings(settings)

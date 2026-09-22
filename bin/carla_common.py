"""CARLA 采集公共件(base env,依赖 pycarla):位姿换算 / NPC 摆放 / 传感器参数。

灯态 GT 的 carla→纯值归一(traffic_light_frame)与 overlay 绘制
(draw_traffic_lights)也放这里:采集器与实时可视化共用同一条实现,
保证"目检所见 = 落盘口径"。
"""

from __future__ import annotations

from typing import Protocol, cast

import carla
import numpy as np
from PIL import Image, ImageDraw

from autodrivedata.calib import CameraIntrinsics, world_to_img
from autodrivedata.camera_rig import NUS_CAMERA_RIG
from autodrivedata.traffic_light import (
    TrafficLightFrame,
    TrafficLightState,
    in_front,
    normalize_state,
)


class _WalkerCtrl(Protocol):
    """pycarla 桩缺 WalkerAIController 方法标注的最小协议。"""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def set_max_speed(self, speed: float) -> None: ...
    def go_to_location(self, destination: carla.Location) -> None: ...


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
# 官方 nuScenes 相机挂点(x 前 / y 右 / z 上,CARLA 系)——由
# `autodrivedata/camera_rig.NUS_CAMERA_RIG` 导出(y 已翻号,与官方 nus 系 y 左镜像)。
# 真值在 camera_rig,**本表只是"只关心平移的调用方"的别名**;改官方标定只改那一处。
SENSOR_MOUNTS: dict[str, tuple[float, float, float]] = {
    name: mount for name, (mount, _rot) in NUS_CAMERA_RIG.items()
}


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
        ("vehicle.tesla.model3", place(12.0, 0.0, ego_yaw)),  # 同车道前车
        ("vehicle.audi.a2", place(22.0, 2.2, ego_yaw)),  # 右邻车道
        ("vehicle.ford.mustang", place(30.0, -3.2, ego_yaw + 180)),  # 对向车
        ("walker.pedestrian.0001", place(8.0, 3.2, ego_yaw)),  # 右侧行人
        ("walker.pedestrian.0002", place(14.0, -3.2, ego_yaw + 90)),  # 左侧行人(面向车道)
        ("vehicle.gazelle.omafiets", place(18.0, 3.6, ego_yaw)),  # 右侧骑行者
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
    """开同步模式(固定 tick),并 tick 一次刷新客户端 actor 快照。

    实测(2026-09-09):连到**已处于同步模式**的服务器时,首个 get_actors()
    返回空(快照只在 tick 后更新)→ 各采集器"清场残留 actor"的循环会静默漏清,
    残留 ego 阻塞 pts[0] 即复现 A/B 起点失配的老坑。此处统一补一次 tick。
    """
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = delta
    world.apply_settings(settings)
    world.tick()


LIGHT_HEAD_Z = 4.5  # 灯头相对 actor 锚点的高度(实测灯箱 z≈4.0-5.2,状态显示在灯头)
TL_COLOR = {
    "Red": (255, 40, 40),
    "Yellow": (255, 210, 0),
    "Green": (40, 255, 80),
    "Off": (120, 120, 120),
    "Unknown": (120, 120, 120),
}


def traffic_light_frame(
    world: carla.World,
    frame_id: str,
    ego_location: tuple[float, float, float],
    ego_yaw_deg: float,
    horizon: float = 120.0,
    phase_plan: tuple[tuple[str, float], ...] = (),
    forward_only: bool = True,
) -> TrafficLightFrame:
    """世界内信号灯 → 纯值 TrafficLightFrame(状态 + 管制车道 + 停车线)。

    - 灯头位置 = actor 锚点 + LIGHT_HEAD_Z(投影/可视口径)
    - affected_lanes = 路口内管制车道、stop_lanes = 停车线所在车道
      → 状态关联到**流向**(一个灯头管多条道),不是只给灯头坐标
    - 视距 horizon 外剔除(距离字段保留,供下游按需再过滤)
    - forward_only:只收 ego 前方半平面的灯(实测不过滤则 79% 是身后灯)
    """
    lights: list[TrafficLightState] = []
    for actor in world.get_actors().filter("traffic.traffic_light"):
        light = cast(carla.TrafficLight, actor)  # pyi 桩:Actor 无 get_state/get_opendrive_id
        t = light.get_transform()
        head = (t.location.x, t.location.y, t.location.z + LIGHT_HEAD_Z)
        if forward_only and not in_front(head, ego_location, ego_yaw_deg):
            continue
        dist = float(np.hypot(head[0] - ego_location[0], head[1] - ego_location[1]))
        if dist > horizon:
            continue
        affected = tuple(sorted({(wp.road_id, wp.lane_id) for wp in light.get_affected_lane_waypoints()}))
        stops = tuple(
            sorted({(wp.road_id, wp.lane_id, round(float(wp.s), 1)) for wp in light.get_stop_waypoints()})
        )
        lights.append(
            TrafficLightState(
                opendrive_id=str(light.get_opendrive_id()),
                state=normalize_state(str(light.get_state())),
                location=head,
                yaw_deg=float(t.rotation.yaw),
                pole_index=int(light.get_pole_index()),
                elapsed_s=float(light.get_elapsed_time()),
                distance_m=dist,
                affected_lanes=affected,
                stop_lanes=stops,
            )
        )
    lights.sort(key=lambda s: s.distance_m)
    return TrafficLightFrame(
        frame_id=frame_id,
        map_name=world.get_map().name,
        ego_location=ego_location,
        ego_yaw_deg=ego_yaw_deg,
        lights=tuple(lights),
        phase_plan=phase_plan,
    )


def draw_traffic_lights(
    img: Image.Image,
    frame: TrafficLightFrame,
    cam_loc: tuple[float, float, float],
    cam_rot: tuple[float, float, float],
    k: CameraIntrinsics,
) -> Image.Image:
    """灯态 overlay:色点 + `#id 状态 距离`(消费纯值帧 → 所见即落盘口径)。"""
    d = ImageDraw.Draw(img)
    for light in frame.lights:
        uv = world_to_img(light.location, cam_loc, cam_rot, k)
        if uv is None:
            continue
        col = TL_COLOR.get(light.state, TL_COLOR["Unknown"])
        x, y = uv
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=col, outline=(0, 0, 0))
        d.text((x + 8, y - 6), f"#{light.opendrive_id} {light.state} {light.distance_m:.0f}m", fill=col)
    return img

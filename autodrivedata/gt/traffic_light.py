"""交通信号灯状态 GT(动态层,2026-09-09,Plan.md §5.9)。

工业界口径(2026-09-09 调研,见 Plan.md §5.9):
- 灯态是**独立的时序语义层**,与检测框、静态地图几何分离:BDD100K 把灯色挂在
  框属性上(`trafficLightColor: red|green|yellow|none`,最省但无时序)、WOMD 逐帧
  记状态序列(`traffic_light_state/current|future/state`,原始数据 71.7% 缺失或
  unknown)、本模块取后者形态 + 记录**变化点信号**(elapsed_s)与**管制关系**。
- 状态枚举保留 Off/Unknown,**不猜**:下游对"猜错的绿灯"的代价远大于"未知"。
- 关联到**流向**而非灯头:一个灯头管多条车道,规划要的是"我这条道现在能不能走"
  → `affected_lanes`(路口内管制车道)+ `stop_lanes`(停车线所在车道)。

本模块**纯值、不 import carla**(包纪律:两 env 可单测);采集侧
(autodrivedata.sim.carla_common.traffic_light_frame)负责把 carla API 对象归一成本模块值对象。
落盘:KITTI root 扩展 `training/traffic_light/{fid}.json`。

几何口径:`location` = **灯头**位置(actor 锚点 + 4.5m,投影/可视口径;
状态显示在灯头而非杆底),世界系。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

# CARLA TrafficLightState 枚举归一(Off/Unknown 保留,不映射成三色)
STATES = ("Red", "Yellow", "Green", "Off", "Unknown")


def normalize_state(raw: str) -> str:
    """'TrafficLightState.Red' / 'Red' → 'Red';未知值归 'Unknown'(不猜)。"""
    name = raw.rsplit(".", 1)[-1]
    return name if name in STATES else "Unknown"


def in_front(
    point: tuple[float, float, float],
    ego_location: tuple[float, float, float],
    ego_yaw_deg: float,
) -> bool:
    """点是否在 ego 前方半平面(沿行驶方向)。

    灯态 GT 的语义是"**本车相关**的信号"(BDD/WOMD 只标视野内灯):
    纯圆形 horizon 会把身后 120m 的灯一起收进来——实测 90 帧里
    视距内 1046 灯次有 79% 在车后,与当前图像/本车决策无关。
    """
    yaw = math.radians(ego_yaw_deg)
    dx, dy = point[0] - ego_location[0], point[1] - ego_location[1]
    return dx * math.cos(yaw) + dy * math.sin(yaw) > 0.0


@dataclass(frozen=True)
class TrafficLightState:
    """单个信号灯的一帧状态(世界系;opendrive_id 即 xodr 的 signal id)。"""

    opendrive_id: str
    state: str  # Red | Yellow | Green | Off | Unknown
    location: tuple[float, float, float]  # 灯头(杆顶)世界坐标
    yaw_deg: float  # 灯朝向(actor yaw)
    pole_index: int
    elapsed_s: float  # 当前状态已持续秒数 → 变灯时刻可反推
    distance_m: float  # 到 ego 的水平距离
    affected_lanes: tuple[tuple[int, int], ...] = ()  # 管制车道 (road_id, lane_id)
    stop_lanes: tuple[tuple[int, int, float], ...] = ()  # 停车线 (road_id, lane_id, s)


@dataclass(frozen=True)
class TrafficLightFrame:
    """一帧灯态 GT:全图灯状态 + ego 位姿锚定 + 相位计划(受控模式)。"""

    frame_id: str
    map_name: str
    ego_location: tuple[float, float, float]
    ego_yaw_deg: float
    lights: tuple[TrafficLightState, ...] = ()
    phase_plan: tuple[tuple[str, float], ...] = ()  # 空 = 记录模式(未干预灯周期)

    def to_json(self) -> str:
        d = asdict(self)
        d["lights"] = [asdict(l) for l in self.lights]
        d["phase_plan"] = [[name, sec] for name, sec in self.phase_plan]
        return json.dumps(d, ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> TrafficLightFrame:
        d = json.loads(text)
        lights = tuple(
            TrafficLightState(
                opendrive_id=s["opendrive_id"],
                state=s["state"],
                location=tuple(s["location"]),
                yaw_deg=s["yaw_deg"],
                pole_index=s["pole_index"],
                elapsed_s=s["elapsed_s"],
                distance_m=s["distance_m"],
                affected_lanes=tuple((int(r), int(l)) for r, l in s.get("affected_lanes", [])),
                stop_lanes=tuple((int(r), int(l), float(sv)) for r, l, sv in s.get("stop_lanes", [])),
            )
            for s in d.get("lights", [])
        )
        return cls(
            frame_id=d["frame_id"],
            map_name=d.get("map_name", ""),
            ego_location=tuple(d["ego_location"]),
            ego_yaw_deg=d["ego_yaw_deg"],
            lights=lights,
            phase_plan=tuple((str(n), float(s)) for n, s in d.get("phase_plan", [])),
        )


def phase_at(t: float, plan: tuple[tuple[str, float], ...]) -> str:
    """受控模式的相位查表:t 秒落在哪个相位(周期循环,纯值可单测)。"""
    if not plan:
        return "Unknown"
    total = sum(sec for _, sec in plan)
    x = t % total
    acc = 0.0
    for name, sec in plan:
        acc += sec
        if x < acc:
            return name
    return plan[-1][0]

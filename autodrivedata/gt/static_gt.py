"""静态目标/道路特征 GT(CARLA 地图查询源,P2,2026-09-09)。

源裁决(2026-09-09 实测 Town10HD_Opt,详见 Plan.md §5.6a):
- 信号灯/标志 = OpenDRIVE **landmark**(landmark 面比灯 actor 更细:一个灯头管多
  条 lane;灯 actor 见 traffic_light.py 的动态层);landmark 位置 = 地面锚点
- 车道线 = waypoint.lane_marking 实体(type/color/width),沿 lane 中心线采样重建
- semantic LiDAR / RoadRunner 两候选实测出局/挂起(见 §5.6a)

本模块**纯值、不 import carla**(包纪律:两 env 可单测);采集侧负责把
carla API 对象归一成本模块值对象。落盘:KITTI root 扩展
`training/static_gt/{fid}.json`(与帧对齐的自描述格式,KITTI 无静态 GT 先例)。

几何口径:
- landmark 锚点 = 地图事实(xodr),相机 2D 投影仅供目检叠加,不进评测
- 车道线折线 = 世界系点列,沿 ego 行驶向采样(lane 中心左右各 lane_width/2);
  同侧同属性连续段合并为一条 LaneSegment(属性随路段可变的天然表达)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


def landmark_kind(name: str) -> str:
    """landmark name → 归一类别;未知名返回 'unknown'。

    注意:landmark.type 编码与语义名不可靠对应(实测 type 205 名 Sign_Yield、
    type 206 名 Sign_Stop),归一以 name 为准;交通灯允许有无下划线两式
    (Signal_3Light_Post01 / SignalJunction)。
    """
    if name.startswith("Signal"):
        return "traffic_light"
    if name.startswith("Sign_Stop"):
        return "stop"
    if name.startswith("Sign_Yield"):
        return "yield"
    return "unknown"


@dataclass(frozen=True)
class StaticSignal:
    """单个静态信号(landmark 归一)。location = 世界系地面锚点。"""

    landmark_id: str
    kind: str  # traffic_light | stop | yield | unknown
    name: str
    location: tuple[float, float, float]
    yaw_deg: float


@dataclass(frozen=True)
class LaneSegment:
    """一条同属性车道线折线段。points = 世界系点列(含端点,沿行驶向)。"""

    side: str  # left | right(相对 ego 行驶向)
    mark_type: str  # SolidSolid | Solid | Broken | None 等(xodr 原值)
    color: str  # Yellow | White 等(xodr 原值)
    width: float
    points: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class StaticFrame:
    """一帧静态 GT:信号 + 车道线,ego 位姿锚定(世界系)。"""

    frame_id: str
    ego_location: tuple[float, float, float]
    ego_yaw_deg: float
    signals: tuple[StaticSignal, ...] = ()
    lane_lines: tuple[LaneSegment, ...] = ()

    def to_json(self) -> str:
        d = asdict(self)
        d["signals"] = [asdict(s) for s in self.signals]
        d["lane_lines"] = [asdict(l) for l in self.lane_lines]
        return json.dumps(d, ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> StaticFrame:
        d = json.loads(text)
        signals = tuple(
            StaticSignal(
                landmark_id=s["landmark_id"],
                kind=s["kind"],
                name=s["name"],
                location=tuple(s["location"]),
                yaw_deg=s["yaw_deg"],
            )
            for s in d.get("signals", [])
        )
        lines = tuple(
            LaneSegment(
                side=l["side"],
                mark_type=l["mark_type"],
                color=l["color"],
                width=l["width"],
                points=tuple(tuple(p) for p in l["points"]),
            )
            for l in d.get("lane_lines", [])
        )
        return cls(
            frame_id=d["frame_id"],
            ego_location=tuple(d["ego_location"]),
            ego_yaw_deg=d["ego_yaw_deg"],
            signals=signals,
            lane_lines=lines,
        )

    def point_count(self) -> int:
        return sum(len(l.points) for l in self.lane_lines)


def merge_lane_marks(
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """连续同属性采样合并为线段表示(纯值,可单测)。

    samples: 沿行驶向的采样序列(可 left/right 交替),每项 {
      "side": "left"|"right", "mark_type": str, "color": str, "width": float,
      "x": float, "y": float, "z": float}
    left/right 是两条独立的线(采样会交替),**必须分流各自顺序合并**,
    否则线迹被对侧采样打断 → 每段仅 1 点(2026-09-09 实测 bug)。
    输出: {"side", "mark_type", "color", "width", "points": [(x,y,z), ...]}
    """

    def _merge_side(stream: list[dict[str, Any]]) -> list[dict[str, Any]]:
        segs: list[dict[str, Any]] = []
        cur: dict[str, Any] | None = None
        for s in stream:
            if cur is None or (cur["mark_type"], cur["color"], cur["width"]) != (
                s["mark_type"],
                s["color"],
                s["width"],
            ):
                if cur is not None:
                    segs.append(cur)
                cur = {
                    "side": s["side"],
                    "mark_type": s["mark_type"],
                    "color": s["color"],
                    "width": s["width"],
                    "points": [],
                }
            cur["points"].append((s["x"], s["y"], s["z"]))
        if cur is not None:
            segs.append(cur)
        return segs

    return _merge_side([s for s in samples if s["side"] == "right"]) + _merge_side(
        [s for s in samples if s["side"] == "left"]
    )

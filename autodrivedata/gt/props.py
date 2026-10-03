"""静态道具 GT(锥桶 / 路障 / 施工围挡 等 `static.prop.*`)—— 纯值,不 import carla。

## 为什么**不进** `label_2`

建 `label_2` 的循环里有一行显式过滤:

    if not (a.type_id.startswith("vehicle") or a.type_id.startswith("walker")):
        continue

锥桶是被这行**主动跳过**的。这条过滤**不是遗漏,是 A/B 纪律的一部分** —— P1-6b 的
遮挡道具(`static.prop.streetbarrier` / `warningconstruction`)正是靠它被挡在 label_2
之外,才保住了「加墙不改 GT 逐帧条数」这个硬门槛。把道具塞进 label_2 会当场破掉
帧级配对(两侧道具不同 ⇒ 逐帧条数不等),**整条 A/B 结论作废**。

⇒ 静态道具走**独立通道** `training/static_prop_gt/{fid}.json`,与 `label_2` 并列。

## 尺寸**只认 yaw=0 探针**那一读

`Actor.bounding_box` 对**转过**的 actor 给出 `(extent, rotation)` 自相矛盾的读数。
2026-10-01 探针实测(`probe_static_prop_gt`,锥桶 0.3441×0.3441×0.5858 m):

| actor yaw | 直接读回的 `(x, y)` 全长 | 真值 |
|---|---|---|
| 0° | 0.3431 × 0.3450 | 0.3441 × 0.3441 ✓ |
| 30° | **0.1246 × 0.4704** | ✗ |
| 60° | **0.1272 × 0.4697** | ✗ |
| 90° | 0.3450 × 0.3431 | ✓ |

**不是两轴对调** —— 边长 s 的方锥转 θ 后读回 ≈ `(s·|cosθ−sinθ|, s·(cosθ+sinθ))`,
盒被**剪切**了。后果拿渲染轮廓量:3.4% 的物体像素落在投影框**外面**,IoU 0.897 → 0.708。
而 0° 和 90° 都读对 ⇒ **只拿一个 yaw≈0 的样本验一次会得"没问题"**(本仓"yaw≈0 的
相机看着正常"的同款坑)。

⇒ 本模块只存 `size`(**yaw=0 探针**读回的那份,逐资产恒定),朝向单独用 `yaw_deg` 表达;
消费方重建 3D 盒时用「yaw=0 的盒 + actor 位姿」,不许回头去读 `bounding_box`。

## 类别口径

CARLA `static.prop.*` 有几百个资产,KITTI `label_2` 里**没有道具类**
(`classify_kitti` 对非 vehicle 一律给 `Misc`)。所以本模块自带一套归一类别,
**只用于静态道具通道**;nuScenes 官方 23 类里**有** `traffic_cone` 与 `barrier`,
`nus_class_of` 负责把归一类别映到那两个名字(其余返回 `None` = 官方忽略类)。

⚠️ **未识别的一律落 `"prop"`**,不猜。猜错的样子不是崩溃,是**一个看着正常的类别名**。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

Vec3 = tuple[float, float, float]

#: 归一类别的**唯一**合法取值。加新类别必须同时改这里(有单测钉)。
PROP_LABELS: tuple[str, ...] = ("cone", "barrier", "barrel", "sign", "prop")

#: 归一类别 → nuScenes 官方类名(`None` = 官方忽略类,不进 nuScenes 评测)。
#: 依据是 nuScenes 检测 23 类表里**确实存在**的那两个: `traffic_cone` / `barrier`。
PROP_TO_NUS: dict[str, str | None] = {
    "cone": "traffic_cone",
    "barrier": "barrier",
    "barrel": None,
    "sign": None,
    "prop": None,
}

#: 显式表(已实测/常被引用的资产)。**子串规则兜底,但显式表优先** ——
#: 子串会误伤(`static.prop.warningconstruction` 里没有 "barrier" 却有挡板语义)。
_EXPLICIT: dict[str, str] = {
    "static.prop.constructioncone": "cone",
    "static.prop.trafficcone01": "cone",
    "static.prop.trafficcone02": "cone",
    "static.prop.streetbarrier": "barrier",
    "static.prop.warningconstruction": "barrier",
}

#: 子串规则(小写匹配),按顺序第一条命中即用。
_SUBSTRING: tuple[tuple[str, str], ...] = (
    ("cone", "cone"),
    ("barrier", "barrier"),
    ("fence", "barrier"),
    ("barrel", "barrel"),
    ("sign", "sign"),
)


def classify_prop(type_id: str) -> str:
    """CARLA `static.prop.*` type_id → 归一类别;认不出的一律 `"prop"`。

    **不返回 None**:静态道具通道的每一行都必须有个类别名 —— 返回 None 会让调用方
    各自决定"那这行怎么办",而那些决定会不一致。认不出就是 `"prop"`,它是**显式的**
    未知,不是"没有类"。
    """
    if type_id in _EXPLICIT:
        return _EXPLICIT[type_id]
    low = type_id.lower()
    for needle, label in _SUBSTRING:
        if needle in low:
            return label
    return "prop"


def nus_class_of(label: str) -> str | None:
    """归一类别 → nuScenes 官方类名(`None` = 官方忽略类)。"""
    if label not in PROP_TO_NUS:
        raise KeyError(f"未知的静态道具类别 {label!r};合法取值 = {PROP_LABELS}")
    return PROP_TO_NUS[label]


def is_prop(type_id: str) -> bool:
    """该 type_id 是不是静态道具(采集侧清场/收集共用一个谓词,别各写一份)。"""
    return type_id.startswith("static.prop")


@dataclass(frozen=True)
class PropBox:
    """单个静态道具的世界系记录。

    角度一律**度**(与此通道的 `static_gt.StaticSignal` 同口径;C++/KITTI 那边的
    弧度口径是 `gt.core.ActorBox`,两者不混)。

    `size` 是 **yaw=0 探针**读回的 `(长, 宽, 高)` 全长;`box_offset` / `box_rotation_deg`
    同样取自那一读(见模块头注:转过之后读回的是被剪切的错值)。
    """

    type_id: str
    label: str
    location: Vec3  # actor 原点世界坐标
    yaw_deg: float
    size: Vec3
    box_offset: Vec3 = (0.0, 0.0, 0.0)
    box_rotation_deg: Vec3 = (0.0, 0.0, 0.0)
    #: **实例图里的 id**(= spawn 时 CARLA 给的 actor id,采集器落进 `prop_inst/*.png`)。
    #: 判据靠它把这行记录对到那张图上的**渲染轮廓**上 —— 没有它,判据只剩"按投影自己去
    #: 找轮廓"这一条路,而那正是要被检验的东西(自证)。`-1` = 本帧没落 id 图(**显式未知**,
    #: 不是"没有实例" —— 判据遇到它就跳过并报数,不许静默当成 0 像素)。
    instance_id: int = -1

    @property
    def nus_class(self) -> str | None:
        return nus_class_of(self.label)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass(frozen=True)
class CameraPose:
    """落 `prop_inst/{fid}.png` 那台相机的**实读**位姿与内参。

    为什么要把它落进 GT 文件、而不是让判据自己去查:判据住在 `perception/`
    (**层规则禁 carla**),而相机内参的真值在 `sim/carla_common.CAM_ATTRS` ——
    它 import carla。落进文件既绕开了这层依赖,又把"这一帧的 id 图**到底是哪台相机
    拍的**"钉在同一份记录里(判据要复现的就是这条投影链)。

    `rotation_deg` 顺 `PropBox` 的度口径;判据转弧度时用 `gt.core.actor_box_from_prop`
    同一条路,不另写换算。存的是 **capture 之后实读**的位姿,不是规格值。
    """

    location: Vec3
    rotation_deg: Vec3  # (pitch, yaw, roll)
    width: int
    height: int
    fov_deg: float


@dataclass(frozen=True)
class PropFrame:
    """一帧的静态道具 GT。ego 位姿 + 相机位姿一并落盘(判据离线复现投影链要用)。"""

    frame_id: str
    ego_location: Vec3
    ego_yaw_deg: float
    props: tuple[PropBox, ...] = ()
    camera: CameraPose | None = None

    def to_json(self) -> str:
        d: dict[str, Any] = asdict(self)
        d["props"] = [asdict(p) for p in self.props]
        return json.dumps(d, ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> PropFrame:
        d = json.loads(text)
        props = tuple(
            PropBox(
                type_id=p["type_id"],
                label=p["label"],
                location=tuple(p["location"]),  # type: ignore[arg-type]
                yaw_deg=p["yaw_deg"],
                size=tuple(p["size"]),  # type: ignore[arg-type]
                box_offset=tuple(p.get("box_offset", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
                box_rotation_deg=tuple(p.get("box_rotation_deg", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
                instance_id=int(p.get("instance_id", -1)),
            )
            for p in d.get("props", [])
        )
        cam = d.get("camera")
        return cls(
            frame_id=d["frame_id"],
            ego_location=tuple(d["ego_location"]),  # type: ignore[arg-type]
            ego_yaw_deg=d["ego_yaw_deg"],
            props=props,
            camera=(
                CameraPose(
                    location=tuple(cam["location"]),  # type: ignore[arg-type]
                    rotation_deg=tuple(cam["rotation_deg"]),  # type: ignore[arg-type]
                    width=int(cam["width"]),
                    height=int(cam["height"]),
                    fov_deg=float(cam["fov_deg"]),
                )
                if cam
                else None
            ),
        )

    def label_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.props:
            out[p.label] = out.get(p.label, 0) + 1
        return out

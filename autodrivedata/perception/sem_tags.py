"""CARLA 语义相机的 **tag 表 / 调色板 / 三类映射**(纯值,只吃 numpy + PIL)。

## tag 从哪来、编在哪

`sensor.camera.semantic_segmentation` 的**原始帧**里 `B = G = 0`、`A = 255`,
**tag 就在 R 通道**;不 `convert(Raw)` 直接 `save_to_disk` 存下来的是
`CityScapesPalette` 的**上色预览** —— 人眼好看、数值全错。

⚠️ **tag 编号 = `carla.CityObjectLabel`,不是旧版 CARLA 的 CityScapes 顺序**。
拿记忆里的 "6=RoadLine / 7=Road" 去写会全错(那是更早的编号)。本表 2026-10-01
**实测反查**过:把帧里出现的每个 tag 的**调色板色**与 CARLA 源码的 29 色表逐个对,
16/16 全中;`tests/sim/test_sem_tags.py` 再用 `carla.CityObjectLabel` 当 oracle 钉死。

## 为什么落 8 位灰度 PNG

tag 是 0–28 的小整数,灰度 PNG 正好承载:`Image.fromarray(tag)` 就是 `mode="L"`,
读回来还是同一个数组。**不用 RGB 存"红通道装 tag"** —— 那种图任何看图工具都显示成
一片黑,而"看不出来"正是这类产物最坏的失败态。灰度至少能看出结构。

## 三类映射的口径(判据的公平性全在这里)

本项目预测侧是 **YOLOPv2(`da` 可行驶 / `ll` 车道线)+ YOLO11s-seg(car/bus/truck/person)**,
GT 侧就必须是**同一组类**:

| 类 | GT tag | 预测侧来源 |
|---|---|---|
| `drivable` | `Roads`(1) | YOLOPv2 `da` |
| `lane` | `RoadLines`(24) | YOLOPv2 `ll` |
| `obstacle` | `Pedestrians`(12) / `Car`(14) / `Truck`(15) / `Bus`(16) | YOLO11s-seg 四类 |

**两侧都不算** `Rider`(13) / `Motorcycle`(18) / `Bicycle`(19) / `Train`(17) ——
预测器**根本产不出**这些类,放进 GT 就是把"模型没有这个类"记成"模型漏检了"。
被排除的像素占比由 `excluded_share()` 单独报出来(不许静默设上限)。

同理 `drivable` **只取 `Roads`**:`Sidewalks`(2) / `Terrain`(10) / `Ground`(25)
都不是 BDD100K 口径的可行驶面,预测器说它们是 da 就该扣分。这些的占比也一并报。
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

#: tag 名 → 值。**必须与 `carla.CityObjectLabel` 逐项相等**(有 oracle 单测钉)。
SEM_TAGS: dict[str, int] = {
    "NONE": 0,
    "Roads": 1,
    "Sidewalks": 2,
    "Buildings": 3,
    "Walls": 4,
    "Fences": 5,
    "Poles": 6,
    "TrafficLight": 7,
    "TrafficSigns": 8,
    "Vegetation": 9,
    "Terrain": 10,
    "Sky": 11,
    "Pedestrians": 12,
    "Rider": 13,
    "Car": 14,
    "Truck": 15,
    "Bus": 16,
    "Train": 17,
    "Motorcycle": 18,
    "Bicycle": 19,
    "Static": 20,
    "Dynamic": 21,
    "Other": 22,
    "Water": 23,
    "RoadLines": 24,
    "Ground": 25,
    "Bridge": 26,
    "RailTrack": 27,
    "GuardRail": 28,
    "Any": 255,
}

#: CARLA `CityScapesPalette`(取自 CARLA 源码 `CityScapesPalette.h`)。**只用于画图**;
#: 判据不吃颜色。也当 oracle 用:实测帧里每个 tag 的调色板色都要能在本表里对上。
PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (128, 64, 128),
    2: (244, 35, 232),
    3: (70, 70, 70),
    4: (102, 102, 156),
    5: (190, 153, 153),
    6: (153, 153, 153),
    7: (250, 170, 30),
    8: (220, 220, 0),
    9: (107, 142, 35),
    10: (152, 251, 152),
    11: (70, 130, 180),
    12: (220, 20, 60),
    13: (255, 0, 0),
    14: (0, 0, 142),
    15: (0, 0, 70),
    16: (0, 60, 100),
    17: (0, 80, 100),
    18: (0, 0, 230),
    19: (119, 11, 32),
    20: (110, 190, 160),
    21: (170, 120, 50),
    22: (55, 90, 80),
    23: (45, 60, 150),
    24: (157, 234, 50),
    25: (81, 0, 81),
    26: (150, 100, 100),
    27: (230, 150, 140),
    28: (180, 165, 180),
}

#: 判据的三个类(顺序即报数顺序)。与预测侧一一对应,见模块头注那张表。
GT_CLASSES: tuple[str, ...] = ("drivable", "lane", "obstacle")

#: 类 → 构成它的 CARLA tag。
GT_CLASS_TAGS: dict[str, frozenset[int]] = {
    "drivable": frozenset({SEM_TAGS["Roads"]}),
    "lane": frozenset({SEM_TAGS["RoadLines"]}),
    "obstacle": frozenset({SEM_TAGS["Pedestrians"], SEM_TAGS["Car"], SEM_TAGS["Truck"], SEM_TAGS["Bus"]}),
}

#: **两侧都不算**的 tag(预测器产不出这些类)。单独报占比,不许静默丢。
EXCLUDED_TAGS: frozenset[int] = frozenset(
    {SEM_TAGS["Rider"], SEM_TAGS["Motorcycle"], SEM_TAGS["Bicycle"], SEM_TAGS["Train"]}
)

#: 物理上像"地面"但**不属于** BDD100K 可行驶面的 tag。同上报占比。
NON_DRIVABLE_SURFACE_TAGS: frozenset[int] = frozenset(
    {SEM_TAGS["Sidewalks"], SEM_TAGS["Terrain"], SEM_TAGS["Ground"]}
)

#: **地图自带的静态道具**(锥桶/路障/施工围挡……)在语义相机里落的 tag。
#:
#: ★ 这个 tag 集是**实测**定的,不是查文档定的 —— `CityObjectLabel` 只给了名字,
#: 谁是谁没有任何规格说明,而名字本身还**反直觉**(锥桶是静态道具,却打成 `Dynamic`)。
#: 2026-10-01 探针(`sim/probe_static_prop_gt`)把三个已知资产逐个摆到镜头前读:
#:
#: | 资产 | 实例掩膜内 tag |
#: |---|---|
#: | `static.prop.constructioncone` | **21 `Dynamic` 100%** |
#: | `static.prop.streetbarrier` | **21 `Dynamic` 100%** |
#: | `static.prop.warningconstruction` | **21 `Dynamic` 100%** |
#:
#: 3/3 排他。**`Static`(20) 与 `Other`(22) 不并进来** —— 它们各有几千像素,但
#: **没有已知资产能归因**(形态上像"一堆小碎块里混着几个物体"),并进来就是把
#: 没验过的假设写死。要扩这个集合,先拿资产把它钉下来。
#:
#: 为什么值得单独一支:这些 tag **既不在三个 GT 类里、也不在 `EXCLUDED_TAGS` 里**
#: ⇒ 它们原本落在一个**不被报出来的第三桶**。模型若在那里报 obstacle,会被记成 FP
#: 而日志里查不出为什么(COCO 训练集无 traffic cone,当前 YOLO11s-seg 不报 ——
#: 但那是**模型的现状**,不是**判据的性质**)。
PROP_TAGS: frozenset[int] = frozenset({SEM_TAGS["Dynamic"]})

#: **地图自带的**静态道具属于"探测得到但没被建模"的那一类:能出**类掩膜**,出不了实例
#: (它们在 `get_actors()` 里一个都没有 —— 2026-10-01 普查 Town10HD_Opt 实测 0 个,
#: 而同一帧语义相机里有 6999 个 `Dynamic` 像素 ⇒ 它们是**关卡网格**,没有 transform 可查)。
#: 由**采集器摆出来**的道具走另一条路(`gt/props.py` 的独立通道,能出实例 GT)。

TAG_NAMES: dict[int, str] = {v: k for k, v in SEM_TAGS.items()}


def prop_mask(tag: np.ndarray) -> np.ndarray:
    """tag 图 → **静态道具类掩膜**(布尔)。见 `PROP_TAGS` 的口径说明。"""
    return np.isin(tag, list(PROP_TAGS))


def encode_tag_png(tag: np.ndarray) -> bytes:
    """tag 图 `(H, W)` uint8 → **8 位灰度** PNG 字节(无损;tag 直接是像素值)。"""
    arr = np.ascontiguousarray(tag)
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise ValueError(f"tag 图须为 (H,W) uint8,got {arr.shape} {arr.dtype}")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def decode_tag_png(data: bytes) -> np.ndarray:
    """PNG 字节 → tag 图 `(H, W)` uint8。

    **只认 `mode="L"`**:RGB / 调色板图一律报错 —— 那种图是"上色预览存错了",
    而它读回来仍然是个数组、能一路算下去,不拦就会得到一整套错的 IoU。
    """
    with Image.open(io.BytesIO(data)) as im:
        if im.mode != "L":
            raise ValueError(f"语义 tag 图须为 8 位灰度(mode='L'),got {im.mode!r}")
        return np.array(im)


def masks_from_tags(tag: np.ndarray) -> dict[str, np.ndarray]:
    """tag 图 → `{类: 布尔掩膜}`(`GT_CLASSES` 三类,各自二值)。"""
    return {c: np.isin(tag, list(GT_CLASS_TAGS[c])) for c in GT_CLASSES}


def excluded_share(tag: np.ndarray) -> dict[str, float]:
    """**两侧都不算**的像素占比,分三桶报出来 —— 报出来才叫"排除",不报就叫"悄悄设了上限"。

    | 桶 | 内容 | 谁决定的 |
    |---|---|---|
    | `excluded_obstacle` | Rider/Motorcycle/Bicycle/Train | **预测器产不出**,算进去等于冤枉它 |
    | `non_drivable_surface` | Sidewalks/Terrain/Ground | 不是 BDD100K 口径的可行驶面 |
    | `map_prop` | 地图自带的静态道具(见 `PROP_TAGS`) | 三个 GT 类都没建模它 |

    第三桶是 2026-10-01 补的。**它此前就在那儿,只是没人报** —— 而"没报"与"没有"
    在下游长得一样。值是像素占比,不是"做得好不好"。
    """
    n = float(tag.size) or 1.0
    return {
        "excluded_obstacle": float(np.isin(tag, list(EXCLUDED_TAGS)).sum()) / n,
        "non_drivable_surface": float(np.isin(tag, list(NON_DRIVABLE_SURFACE_TAGS)).sum()) / n,
        "map_prop": float(prop_mask(tag).sum()) / n,
    }


def paint_tags(tag: np.ndarray) -> np.ndarray:
    """tag 图 → RGB 预览图(人眼看的,**不是判据输入**)。未知 tag 画成品红以便一眼看见。"""
    out = np.zeros((*tag.shape, 3), dtype=np.uint8)
    for t, rgb in PALETTE.items():
        out[tag == t] = rgb
    unknown = ~np.isin(tag, list(PALETTE))
    out[unknown] = (255, 0, 255)
    return out

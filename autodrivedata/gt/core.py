"""CARLA actor → KITTI label_2 GT 行(纯值接口,不 import carla)。

契约(照 auto3dlabel schema/box3d.py + export/kitti_label.py):
- label_2 的 y = 物体**底部中心**(地面);行格式 "label trunc occl alpha x1 y1 x2 y2 h w l x y z ry"
- 类别直落 KITTI 名(对齐 auto3dlabel COCO_TO_KITTI 语义;评测只算 Car/Pedestrian/Cyclist)
- M1a 口径:occluded=0、alpha=0(由 2D 检测不可得,评测不计);
  truncation = 相机前角点中落图外比例;无前角点(相机后)或全落图外 → 剔除(None)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autodrivedata.calib.core import CameraIntrinsics
from autodrivedata.utils import geometry as g

# CARLA type_id → KITTI 类:精确名优先,兜底前缀规则(未知 vehicle 归 Car,非 vehicle 归 Misc)
_TRUCK_TYPES = frozenset(
    {
        "vehicle.carlamotors.carlacola",
        "vehicle.carlamotors.european_hgv",
        "vehicle.carlamotors.firetruck",
        "vehicle.mitsubishi.fusorosa",  # 公交,KITTI 无 Bus 类 → 归大型车
    }
)
_VAN_TYPES = frozenset(
    {
        "vehicle.ford.ambulance",
        "vehicle.volkswagen.t2",
        "vehicle.volkswagen.t2_2021",
    }
)
# CARLA 摩托车/自行车 type_id 不含 "motorcycle"/"bicycle" 子串,须显式枚举
_MOTORCYCLE_TYPES = frozenset(
    {
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.harley-davidson.low_rider",
    }
)
_BICYCLE_TYPES = frozenset(
    {
        "vehicle.bh.crossbike",
        "vehicle.diamondback.century",
        "vehicle.gazelle.omafiets",
    }
)


def classify_kitti(type_id: str) -> str:
    """CARLA actor type_id → KITTI 类名。"""
    if type_id.startswith("walker"):
        return "Pedestrian"
    if not type_id.startswith("vehicle"):
        return "Misc"
    if type_id in _TRUCK_TYPES or "truck" in type_id:
        return "Truck"
    if type_id in _VAN_TYPES:
        return "Van"
    if type_id in _MOTORCYCLE_TYPES or type_id in _BICYCLE_TYPES:
        return "Cyclist"
    if "bicycle" in type_id or "motorcycle" in type_id:
        return "Cyclist"
    if "tram" in type_id or "train" in type_id:
        return "Tram"
    return "Car"


def classify_nus(type_id: str) -> str | None:
    """CARLA actor type_id → nuScenes 检测类名(None = 官方忽略类,不参与评测)。

    对齐 auto3dlabel NUSCENES_CATEGORY_MAP 的 10 类口径(emergency/debris 等忽略)。
    """
    if type_id.startswith("walker"):
        return "pedestrian"
    if not type_id.startswith("vehicle"):
        return None
    if type_id in _TRUCK_TYPES or "truck" in type_id:
        return "truck"
    if type_id in _VAN_TYPES:
        return None  # ambulance 属官方忽略类(vehicle.emergency)
    if type_id in _MOTORCYCLE_TYPES or "motorcycle" in type_id:
        return "motorcycle"
    if type_id in _BICYCLE_TYPES or "bicycle" in type_id:
        return "bicycle"
    if "tram" in type_id or "train" in type_id:
        return None
    return "car"


@dataclass(frozen=True)
class ActorBox:
    """CARLA actor 的 bounding_box + 位姿(纯值)。角度一律 (pitch, yaw, roll) 弧度。"""

    type_id: str
    extent: tuple[float, float, float]  # (x, y, z) 半尺寸
    location: tuple[float, float, float]  # box 相对 actor 原点的偏移(actor 系)
    rotation: tuple[float, float, float]  # box 相对旋转(actor 系,通常 0)
    actor_location: tuple[float, float, float]
    actor_rotation: tuple[float, float, float]


def box_center_world(box: ActorBox) -> np.ndarray:
    """box 中心的世界坐标:actor 位姿作用在 box 偏移上。"""
    r = g.carla_rotation_matrix(box.actor_rotation)
    return np.asarray(box.actor_location, dtype=np.float64) + r @ np.asarray(box.location, dtype=np.float64)


def box_heading_world(box: ActorBox) -> np.ndarray:
    """box 车头单位向量(世界系):actor 旋转 ∘ box 旋转的第一列。"""
    r = g.carla_rotation_matrix(box.actor_rotation) @ g.carla_rotation_matrix(box.rotation)
    return r[:, 0]


def box_corners_world(box: ActorBox) -> np.ndarray:
    """box 的 8 个世界系角点 (8,3):中心 + R ∘ (±extent) 的 8 种符号组合。

    顺序为 x/y/z 的二进制的位序(bit0 = x),与 `box_to_gt_line` 内部展开同源 ——
    供第三方视角自检"ego 框是否落在画面内"用(纯值,不依赖相机)。
    """
    r = g.carla_rotation_matrix(box.actor_rotation) @ g.carla_rotation_matrix(box.rotation)
    e = np.asarray(box.extent, dtype=np.float64)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    return box_center_world(box) + (signs * e) @ r.T


def box_to_gt_line(
    box: ActorBox,
    cam_location: tuple[float, float, float],
    cam_rotation: tuple[float, float, float],
    intrinsics: CameraIntrinsics,
    max_distance: float | None = None,
) -> str | None:
    """ActorBox → label_2 15 字段行;相机后/全落图外/超距返回 None(剔除)。

    max_distance:框中心到相机原点距离上限(m)——训练数据必须设(=LiDAR 量程余量),
    否则远距无点框(实测 163m)会毒化检测器置信度校准(M3-3 教训)。
    """
    center_k = g.world_to_cam(box_center_world(box)[None], cam_location, cam_rotation)[0]
    if max_distance is not None and float(np.hypot(center_k[0], center_k[2])) > max_distance:
        return None
    h, w, l = box.extent[2] * 2, box.extent[1] * 2, box.extent[0] * 2  # CARLA (x,y,z) 半尺寸 → KITTI h,w,l
    ry = g.heading_to_rotation_y(box_heading_world(box), cam_rotation)
    y_bottom = center_k[1] + h / 2  # 体积中心 → 底部中心(y 向下)

    corners = g.corners_cam_from_bottom(center_k[0], y_bottom, center_k[2], h, w, l, ry)
    hom = np.hstack([corners, np.ones((8, 1))])
    img = (intrinsics.p2() @ hom.T).T
    zc = img[:, 2]
    front = zc > 0
    if not front.any():
        return None  # 相机后
    u = img[front, 0] / zc[front]
    v = img[front, 1] / zc[front]
    inside = (u >= 0) & (u < intrinsics.width) & (v >= 0) & (v < intrinsics.height)
    if not inside.any():
        return None  # 全落图外

    trunc = 1.0 - float(inside.sum() / front.sum())
    label = classify_kitti(box.type_id)
    return (
        f"{label} {trunc:.2f} 0 0.00 "
        f"{u[inside].min():.2f} {v[inside].min():.2f} {u[inside].max():.2f} {v[inside].max():.2f} "
        f"{h:.2f} {w:.2f} {l:.2f} "
        f"{center_k[0]:.2f} {y_bottom:.2f} {center_k[2]:.2f} {ry:.2f}"
    )

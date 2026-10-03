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
from autodrivedata.gt import props
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

#: `box_to_gt_line` 的退化剔除阈值:投影框**两维都**必须至少这么宽(px)。
#: 见该函数内那段注 —— 零面积框配不上,且进出由亚帧抖动决定。
MIN_BOX_SIDE_PX = 1.0

#: KITTI `label_2` 的字段数。`parse_gt_line` 与 `refilter` 的残缺行判据**同引这一个**
#: —— 两边各写一个 15 的话,哪天格式变了会只改一处,而症状是"重筛把好行当残缺删了"。
LABEL_2_FIELDS = 15


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

    ## 静态道具为什么单独一分支(2026-10-01 修)

    原来非 vehicle 一律 `None` —— 对锥桶是**错的**:nuScenes 官方 23 类里**有**
    `traffic_cone` 和 `barrier`,把锥桶记成"官方忽略类"是把官方类表读错了。
    现在委托 [`props`](props.py)(归一类别 → 官方类名)。

    ⚠️ **这条修改不改任何既有数据集**:`collect_nus` 与 `collect_ab_route` 建标注前
    都有一层 `vehicle`/`walker` 过滤,**道具根本走不到这里**。改的是"名字对不对",
    不是"谁进 GT" —— 后者是策略,由各采集器的过滤决定(static prop 走独立通道,
    见 `gt/props.py` 头注)。
    """
    if type_id.startswith("walker"):
        return "pedestrian"
    if props.is_prop(type_id):
        return props.nus_class_of(props.classify_prop(type_id))
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


def actor_box_from_prop(p: props.PropBox) -> ActorBox:
    """静态道具记录(`gt.props.PropBox`,度 + yaw=0 尺寸)→ `ActorBox`(世界系,弧度)。

    ★ **`extent` 一律取 `p.size / 2`**(yaw=0 探针那一读),**绝不回头去读
    `Actor.bounding_box`** —— 转过的 actor 那一读是被剪切的错值(§`gt/props.py` 头注那张表)。

    ⚠️ **这一层是三处口径的汇合点**,每一处错了都不抛异常:
    ① **度/弧度**:`PropBox` 是度、`ActorBox` 是弧度。把度直接递进去,六路里只有
       yaw≈0 的那路看着正常(同投影链那条红线);
    ② **半尺寸**:`PropBox.size` 是全长,`ActorBox.extent` 是半长 —— 差一倍;
    ③ **盒旋转 vs actor 旋转**:盒偏移在**局部系**,actor 位姿是**世界系**,两者不能混。
    三条都有回归钉(`tests/gt/test_props.py::TestActorBoxFromProp`)。
    """
    return ActorBox(
        type_id=p.type_id,
        extent=(p.size[0] / 2.0, p.size[1] / 2.0, p.size[2] / 2.0),
        location=p.box_offset,
        rotation=tuple(float(np.radians(a)) for a in p.box_rotation_deg),  # type: ignore[arg-type]
        actor_location=p.location,
        actor_rotation=(0.0, float(np.radians(p.yaw_deg)), 0.0),
    )


def box_corners_world(box: ActorBox) -> np.ndarray:
    """box 的 8 个世界系角点 (8,3):中心 + R ∘ (±extent) 的 8 种符号组合。

    顺序为 x/y/z 的二进制的位序(bit0 = x),与 `box_to_gt_line` 内部展开同源 ——
    供第三方视角自检"ego 框是否落在画面内"用(纯值,不依赖相机)。
    """
    r = g.carla_rotation_matrix(box.actor_rotation) @ g.carla_rotation_matrix(box.rotation)
    e = np.asarray(box.extent, dtype=np.float64)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    return box_center_world(box) + (signs * e) @ r.T


@dataclass(frozen=True)
class CornerProjection:
    """一个 3D 盒的 8 角点投影读数 —— `box_to_gt_line` 与判据**共用的唯一实现**。

    ★ **`all_inside` 是判据的极性开关**,不是装饰:`u`/`v` 只含**相机前**的角点,而
    `inside` 只含**落在画幅内**的角点。`box_to_gt_line` 的 2D 框是
    `u[inside].min()/max()` 那一套 —— 于是当物体**被画幅边缘裁掉**时,被裁掉的角点
    不参与 min/max,**框会小于物体的真实可见范围**。

    实测(P1 全部归档 `label_2`,3922 个框):**690 个(17.6%)** 有角点落在画幅外,
    其中 370 个两种口径差 **>150 px**(中位 160 px、最大 292 px);被裁的框 80% 在
    **0–8 m**(近处擦身而过),`trunc` 全部 > 0(数据集**知道**它们被截,只是 2D 框
    没按截断语义出)。⇒ 用 `coverage` 这类"框有没有包住物体"的判据时,这些样本
    **按旧口径必然不通过**,不能算作 GT 的失败。

    ★ **2026-10-02 已修**:口径改为「全 front 角点 min/max 再钳到画幅」,判据与出框
    侧共用 `box2d_from_projection`(单一实现)。归档数据的重算走 `rebox_line`,
    见 [`refilter.py`](refilter.py) 的 `--clip2d` 与 Plan4 §P-V18。

    ⚠️ 别把它当成"投影口径可选":两种口径在**全角点都在画幅内**时**逐位相同**
    (`all_inside=True`),只有在裁断时才分叉。
    """

    u: np.ndarray  # 相机前角点的 u(可能落在画幅外)
    v: np.ndarray
    front: np.ndarray  # (8,) 角点在相机前
    inside: np.ndarray  # 相机前且有投影的点里,落在画幅内的
    h: float
    w: float
    l: float
    ry: float
    center_k: np.ndarray
    y_bottom: float

    @property
    def all_inside(self) -> bool:
        """8 个角点**全部**落在画幅内(含相机前)。False = 物体被画幅裁断。"""
        return bool(self.inside.all())


def project_corners(
    box: ActorBox,
    cam_location: tuple[float, float, float],
    cam_rotation: tuple[float, float, float],
    intrinsics: CameraIntrinsics,
) -> CornerProjection | None:
    """`ActorBox` → 8 角点投影读数;相机后 / 全落图外返回 None。

    抽出来是为了让**出框侧与判据吃同一条实现** —— 各写一份的话,两份迟早漂,而
    "漂了"在这里的表现是判据**判的是另一套几何**,数字照样出得来。
    """
    center_k = g.world_to_cam(box_center_world(box)[None], cam_location, cam_rotation)[0]
    h, w, l = box.extent[2] * 2, box.extent[1] * 2, box.extent[0] * 2  # CARLA (x,y,z) → KITTI h,w,l
    ry = g.heading_to_rotation_y(box_heading_world(box), cam_rotation)
    y_bottom = center_k[1] + h / 2  # 体积中心 → 底部中心(y 向下)

    corners = g.corners_cam_from_bottom(center_k[0], y_bottom, center_k[2], h, w, l, ry)
    img = (intrinsics.p2() @ np.hstack([corners, np.ones((8, 1))]).T).T
    zc = img[:, 2]
    front = zc > 0
    if not front.any():
        return None  # 相机后
    u = img[front, 0] / zc[front]
    v = img[front, 1] / zc[front]
    inside = (u >= 0) & (u < intrinsics.width) & (v >= 0) & (v < intrinsics.height)
    if not inside.any():
        return None  # 全落图外
    return CornerProjection(u, v, front, inside, h, w, l, ry, center_k, y_bottom)


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

    ## 2D 框的两句口径(**配套的,拆开任一句都会坏**)

    ① **框** = 全 front 角点的 min/max,**再钳到画幅**(KITTI 口径);
    ② **退化剔除**看的是**未钳**的 `inside` 角点跨度(判据 `MIN_BOX_SIDE_PX`)。

    只做①不做②:「擦过镜头」的物体角点投影发散,钳完变成**接近满幅**的框 ⇒
    与任何预测都能配上 ⇒ **凭空造出 TP**,比原来的"框太小"更毒。
    只做②不做①:被画幅裁掉的物体框偏小 ⇒ 检测器对的框被记成漏检 —— **实测 17.6%**。
    """
    if max_distance is not None:
        center_k = g.world_to_cam(box_center_world(box)[None], cam_location, cam_rotation)[0]
        if float(np.hypot(center_k[0], center_k[2])) > max_distance:
            return None

    proj = project_corners(box, cam_location, cam_rotation, intrinsics)
    if proj is None:
        return None
    u, v, front, inside = proj.u, proj.v, proj.front, proj.inside
    h, w, l, ry, center_k, y_bottom = proj.h, proj.w, proj.l, proj.ry, proj.center_k, proj.y_bottom

    # ★ **退化投影剔除**:近处**擦过镜头**的物体会只留一条线甚至一个点
    #   (`x1==x2` 或 `y1==y2`,实测 depth 0.58–3.5 m)。这种框有两重害:
    #   ① **永远配不上** —— 零面积框与任何预测的 IoU 都是 0,是白送的一次漏检;实测
    #      P1 全部数据集各有 **11/194(5.7%)** 这种框 ⇒ **recall 天花板被压到 94.3%**,
    #      每一份 P1 AP 都带着这个折扣(2026-10-01 发现);
    #   ② **进出由亚帧抖动决定** —— 车的远底角恰落在像面下边缘(v≈375)时,0.24 m 的 ego
    #      偏移就能让整框在「0 px 高」与「164 px 高」之间翻面 ⇒ 两次采集的 GT **逐帧条数**
    #      不再相等,A/B 硬门槛当场破(实测 `kitti_ab_occl_none` 194 vs `_near` 195)。
    #   判据取「两维都 ≥ 1 px」:配合 `max_distance`,本项目最远的框也有十几 px,
    #   1 px 只可能命中这类退化投影。
    box2d = box2d_from_projection(u, v, inside, intrinsics)
    if box2d is None:
        return None
    x1, y1, x2, y2 = box2d

    trunc = 1.0 - float(inside.sum() / front.sum())
    label = classify_kitti(box.type_id)
    return (
        f"{label} {trunc:.2f} 0 0.00 "
        f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
        f"{h:.2f} {w:.2f} {l:.2f} "
        f"{center_k[0]:.2f} {y_bottom:.2f} {center_k[2]:.2f} {ry:.2f}"
    )


def box2d_from_projection(
    u: np.ndarray,
    v: np.ndarray,
    inside: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float, float, float] | None:
    """投影角点 → 2D 框。**`box_to_gt_line`(出框侧)与 `rebox_line`(归档重算侧)的
    唯一实现** —— 各写一份的话两份迟早漂,而漂了的表现是"重算出来的框与出新框的框
    不是同一套口径",数字照样出得来。

    ## 两句口径是**配套的,拆开任一句都会坏**

    ① **框** = 全 front 角点的 min/max,**再钳到画幅**(KITTI 口径);
    ② **退化剔除**看**未钳**的 `inside` 角点跨度(判据 `MIN_BOX_SIDE_PX`)。

    | 只做 | 后果 |
    |---|---|
    | 只①(退化也看钳后框) | 「擦过镜头」的角点投影发散 ⇒ 钳完是**接近满幅**的框 ⇒ 与任何预测都能配上 ⇒ **凭空造出 TP**,比"框太小"更毒 |
    | 只②(不钳) | 被边缘裁掉的框偏小 ⇒ 检测器**对的**框被记成漏检(实测归档 17.6%) |
    """
    if (u[inside].max() - u[inside].min()) < MIN_BOX_SIDE_PX or (
        v[inside].max() - v[inside].min()
    ) < MIN_BOX_SIDE_PX:
        return None
    return (
        float(np.clip(u.min(), 0.0, intrinsics.width - 1.0)),
        float(np.clip(v.min(), 0.0, intrinsics.height - 1.0)),
        float(np.clip(u.max(), 0.0, intrinsics.width - 1.0)),
        float(np.clip(v.max(), 0.0, intrinsics.height - 1.0)),
    )


def parse_gt_line(
    line: str,
) -> tuple[str, float, tuple[float, float, float], tuple[float, float, float, float]]:
    """`label_2` 行 → `(label, trunc, (h,w,l), (x,y_bottom,z,ry))`。

    与 `perception.compare.parse_gt_line`(那条给 `Box7`)**同字段、不同落点**:
    这里要的是**重算 2D 列**所需的原始量,不构造 `Box7`(那会多一层语义)。
    """
    p = line.split()
    if len(p) < LABEL_2_FIELDS:
        raise ValueError(f"label_2 行字段不足:{line[:48]}")
    return (
        p[0],
        float(p[1]),
        (float(p[8]), float(p[9]), float(p[10])),
        (float(p[11]), float(p[12]), float(p[13]), float(p[14])),
    )


def rebox_line(line: str, p2: np.ndarray, width: int, height: int) -> str | None:
    """**按新口径重算一行 `label_2` 的 2D 列**(其余字段原样);变成退化则返回 None。

    存在的理由:2D 框可以从**行自己的字段**重算 —— `label_2` 自带 `h w l x y z ry`,
    calib 自带 `P2` ⇒ **不需要重采**(本文件与 `refilter` 的纪律:重采是投骰子,
    见 [`refilter.py`](refilter.py) 头注)。判据与出框侧**共用** `box2d_from_projection`。

    ⚠️ `x/y/z` 是**相机系**(该行的口径),`p2` 必须是**同一份** calib 的 `P2`。
    两者不同源 ⇒ 框整体偏,而偏多少随 `ry` 变 ⇒ 看着像"框有点歪",不像口径错。

    ## 未裁断的行**原样返回**(不是"重算后恰好相等")

    8 个角点全在画幅内时两种口径**逐位相同** —— 那时再按舍入过的 `x/y/z/ry` 重打一遍,
    只会把 **~0.1 px 的字段精度噪声**注进 82.4% 本来没问题的行(实测归档只有 17.6% 被
    裁断)。⇒ 未裁断走**短路**,输出与输入**逐字节相等**。"只动该动的行"于是成了
    可检验的性质(见 `tests/gt/test_refilter.py::test_unclipped_lines_are_byte_identical`),
    而不是一句承诺。

    判据取 `front.all() and inside.all()`(不是 `inside.all()`):有角点在相机后时
    `u/v` 里根本没有它,`inside` 全 True 也**不代表**没被裁 —— 那种行照常重算。
    """
    label, trunc, (h, w, l), (x, y_bottom, z, ry) = parse_gt_line(line)
    corners = g.corners_cam_from_bottom(x, y_bottom, z, h, w, l, ry)
    img = (np.asarray(p2, dtype=float) @ np.hstack([corners, np.ones((8, 1))]).T).T
    zc = img[:, 2]
    front = zc > 0
    if not front.any():
        return None
    u = img[front, 0] / zc[front]
    v = img[front, 1] / zc[front]
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not inside.any():
        return None
    box2d = box2d_from_projection(u, v, inside, CameraIntrinsics(width, height, 90.0))
    if box2d is None:
        return None
    if front.all() and inside.all():
        return line  # ★ 未裁断 ⇒ 两种口径逐位相同,短路(O(1) 的"没变",不是"算出来没变")
    x1, y1, x2, y2 = box2d
    return (
        f"{label} {trunc:.2f} " + " ".join(line.split()[2:4]) + f" {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
        f"{h:.2f} {w:.2f} {l:.2f} "
        f"{x:.2f} {y_bottom:.2f} {z:.2f} {ry:.2f}"
    )


def is_degenerate_gt_line(line: str) -> bool:
    """一行 `label_2` 是否触发 `MIN_BOX_SIDE_PX` 剔除 —— **出框侧那条例的读侧镜像**。

    存在的理由:出框侧(上面的 `box_to_gt_line`)只对**新采**的数据生效,而磁盘上已经
    躺着一批按旧口径写下的框(§P-V12 实测 P1 每份数据 11/194)。把判据做成同一个函数,
    迁移脚本与采集器就不会各判各的 —— 两边**必须同源**,否则"筛过之后"与"重采一份"
    会给出不同的 GT,而那个差只有在 A/B 逐帧条数对不上时才暴露。

    判据取**文件里印出来的两位小数**(`x2-x1` / `y2-y1`),不是重算几何:
    读侧手上只有文本。⚠️ **不能写成 `x1 == x2`** —— 实测 `wet_road` 帧 57 有一条
    **0.01 px 高**的框,它打印出来两个数不相等,但按阈值早该被剔除;写成相等判断会让
    那一帧的 GT 条数比别的 root 多 1,A/B 硬门槛当场破。
    """
    p = line.split()
    if len(p) < 8:
        return True  # 残缺行:留着只会在评测里当一条永远配不上的 GT
    return min(float(p[6]) - float(p[4]), float(p[7]) - float(p[5])) < MIN_BOX_SIDE_PX

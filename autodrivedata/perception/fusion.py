"""后融合:LiDAR 出几何、相机出类,**在 3D 空间关联**后与 `label_2` GT 比(纯值核心)。

## 为什么是"后融合",以及为什么**雷达不在这张表里**

三模态**不对等**,先写清楚再动手(否则"融合"会变成三个不对等的东西平均一下):

| 模态 | 它**真的**给什么 | 在 P1 数据上的状态 |
|---|---|---|
| LiDAR | 精确的 3D 位置与尺寸,**没有类** | 四台车全看得见(18 m 252 点 → 60 m 4 点) |
| 相机 | **类**(与置信度),没有可靠的深度 | YOLO26 在手 |
| 雷达 | 只出**点 + 径向速度**,出不了目标 | **平台边界**:12/15/18/25 m 逐档实测,静止车上归因回波为 **0**(§P-V17 六) |

⇒ 第一版只做 **LiDAR + 相机**。雷达不进消融表 —— 放一路没有信号的模态进去,
得到的"加不加都一样"说明的是**没有回波**,不是"雷达没用"。

## 消融口径:为什么不是"相机把类拉上去"

P1 的 GT **100% 是 `Car`**(48 框 / 12 帧实测),LiDAR 用尺寸启发式也能全判成 Car
⇒ 「相机出类」这一行会是 0.000 的差,而那个 0 说明的是**没有第二类**。
所以消融量的是**精确率与召回各自的增减**:

| 档 | 内容 | 预期 |
|---|---|---|
| `lidar` | LiDAR 簇 + 尺寸启发式 → 全判 Car | 召回高、**FP 多**(路面残留/墙体碎片) |
| `lidar+cam` | 只保留**被相机 2D 框关联上**的簇 | **FP 降**、召回可能也降 |

判据 = 与 `label_2` 的 3D AP@0.5 **加上逐条 `TP/FP/FN`**(AP 一个数看不出是哪种变化)。

## 三个口径,错了都不报错

1. **`ry` 必须是 `−π/2`**:`label_2` 实测四台车 `ry` 全为 **−1.57**(车头沿相机光轴,
   即沿路)。LiDAR 簇给的是**轴对齐 AABB**,若照搬 `ry=0`,长轴会横过来 ——
   与 GT 的 3D IoU 直接掉到接近 0,而**图上看不出任何异常**。
2. **`l/w/h` 的轴序**:velodyne 系 x=前(车的长轴),映射到相机系后 `l` 仍取 velo 的 x 跨度。
   写反 ⇒ 框变成横向的,同样只表现为 IoU 低。
3. **底部中心**:`Box7.y` 是**底部**(KITTI 口径),velo 的 z 是向上 ⇒ 取 `cz − half_z` 再变换。

## 检出与关联

- **尺寸启发式**:`CAR_EXTENT` 的三维带 + 长宽比门。它是 `lidar` 档的"类",也是
  `lidar+cam` 档里**没被相机确认**的簇的去留判据(见 `fuse` 的 `drop_unconfirmed`)。
- **关联**:把簇的 3D 框投到图像,与相机 2D 框比 IoU,贪心一对一(与 `compare.match_boxes`
  同口径)。**投影用 `P2 ∘ R0_rect ∘ Tr_velo_to_cam`**,与 GT 的相机系同源。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from autodrivedata.perception.compare import Box7

#: 车形簇的尺寸带(米)。刻意给宽 —— 这一档的目标是**高召回、允许大量 FP**,由相机去剪。
#:
#: ★ 2026-10-02 更正:最初照 `label_2` 的 `(h,w,l)` 给(`l ∈ (2.8,5.8)` + 长宽比门),
#: 首跑召回 10%,当时归因成"旧门按**车长**给、而 LiDAR 只看得见车尾 ⇒ 真车全被挡"。
#: **该归因已撤回** —— 那组"`l=1.86–1.89` 的真车簇"来自一个**跨坐标系**的诊断
#: (拿 velo 系的 x 减相机系的 x)。同坐标系重查:真车簇是 `4.28 × 1.90 × 1.13`,
#: **旧门也放行**;旧门只是整体很严(138 簇放 1 个)。
#: ⇒ 真正的召回限制是**远距聚类质量**,不是门。门改宽仍有用(它给出"高召回低精确"的
#: 基线,相机才有东西可剪),但理由不是原来那条。
#:
#: ⚠️ **未做的对照**:旧门单独一档的 TP/FP 可能与"新门 + 相机"几乎相同 ——
#: 若成立,相机的贡献可以被一个更严的门替掉。见 Plan4 §P-V18 三。
CAR_HEIGHT_BAND = (0.8, 3.0)
#: 最大水平跨度(取 velo 的 x/y 里大的那个)—— 车尾迎面时是车宽(≈2 m),侧向可见时是车长。
CAR_SPAN_MAX = (1.2, 6.0)
#: 最小水平跨度:一个"薄片"不是车(路面残留、栏杆、标线的残留点)。
CAR_SPAN_MIN = (0.3, 3.0)
#: **人形簇**的尺寸带。★ 存在的理由就是融合本身:LiDAR 只出几何、**分不出类**,
#: 而"车形/人形"是它**唯一**能不靠相机分开的两族。车形判 `Car`,人形**不给类** ——
#: 交给相机(`fuse` 的 `lidar+cam` 档);`lidar` 档只能全判 `Car`,于是行人与骑行者
#: 必然同时记 FP(错判成 Car)与 FN(真正的类没被检出)。**这个差就是相机的贡献。**
CAR_MIN_HEIGHT, PERSON_HEIGHT_BAND = 0.8, (1.0, 2.3)
PERSON_SPAN = (0.2, 1.3)
#: 关联用的 2D IoU 阈值(与 `compare.match_boxes` 的默认口径一致)。
ASSOC_IOU = 0.3
#: `label_2` 的 `ry`:车头沿相机光轴。**实测四台车全 −1.57**,不是猜的。
RY_ALONG_AXIS = -np.pi / 2


@dataclass(frozen=True)
class Cluster:
    """一个 LiDAR 簇(**velodyne 系**轴对齐 AABB + 点数)。"""

    center: tuple[float, float, float]
    half: tuple[float, float, float]
    n_points: int
    min_dist: float = 0.0

    @property
    def extent(self) -> tuple[float, float, float]:
        """全长 `(l, w, h)` —— velo 系 x=前(长轴)/ y=侧 / z=上。"""
        return (2 * self.half[0], 2 * self.half[1], 2 * self.half[2])


@dataclass(frozen=True)
class CameraDet:
    """一个相机 2D 检测(相机系像素框 + 类 + 置信度)。"""

    label: str
    xyxy: tuple[float, float, float, float]
    conf: float


@dataclass(frozen=True)
class FusedObject:
    """融合表的一行。`sources` 记它被哪几路支持(`("lidar",) / ("lidar","cam")`)。"""

    box: Box7
    conf: float
    sources: tuple[str, ...] = field(default_factory=tuple)
    cluster_points: int = 0


# ---------------------------------------------------------------- 几何


def cluster_to_box7(cl: Cluster, velo_to_cam: np.ndarray, *, prior: dict | None = None) -> Box7:
    """velodyne 系的轴对齐簇 → **相机系** `Box7`(KITTI 口径,`ry = −π/2`)。

    `velo_to_cam` 是 4×4(`R0_rect @ Tr_velo_to_cam`,由调用方从 calib 组装 —— 与
    `label_2` 的相机系**同源**,别在这里再自造一条)。

    ⚠️ 见模块头注「三个口径」:`ry` 恒 **−π/2**、`l` 取 velo 的 **x** 跨度、`y` 取**底部**。
    """
    # ★ **按类尺寸先验补全长**(2026-10-07,默认关)。`prior` 是 `size_prior.fit_from_root`
    #   的产物(`{"dims": {"Car": [l,w,h]}, ...}`)。给了就先**锚在可见面上**再出框 ——
    #   LiDAR 只看得见车的近面,而 GT 是整车(实测 350 个簇只有 10 个 IoU>0.5)。
    #   ⚠️ 默认 `None` ⇒ 归档产物**逐字节复现**(与 `--occluders`/`--depth` 同一条纪律)。
    if prior is not None:
        dims = prior["dims"].get(cl_label_fallback(prior))
        if dims is not None:
            from autodrivedata.perception.size_prior import anchor_cluster

            cl = anchor_cluster(cl, (float(dims[0]), float(dims[1]), float(dims[2])))
    l, w, h = cl.extent
    bottom_velo = np.array([cl.center[0], cl.center[1], cl.center[2] - cl.half[2], 1.0])
    x, y_bottom, z = (velo_to_cam @ bottom_velo)[:3]
    return Box7(label="Car", h=h, w=w, l=l, x=float(x), y=float(y_bottom), z=float(z), ry=RY_ALONG_AXIS)


def cl_label_fallback(prior: dict) -> str:
    """这个先验唯一的那个类(本项目 LiDAR 档只有一类几何)。

    ⚠️ 真写成"猜类"就错了 —— LiDAR 分不出类(`Cyclist` 完全是相机给的,见 §P-V18)。
    所以先验在这里只有**一个**条目,它的名字只是台账。
    """
    keys = sorted(prior["dims"])
    return keys[0] if len(keys) == 1 else "Car"


def strict_gate(cl: Cluster) -> bool:
    """**旧口径**(2026-10-01 第一版):照 `label_2` 的 `(h,w,l)` 给带 + 长宽比门。

    ⚠️ 它**不是"错的"** —— 同坐标系重查证实真车簇(`4.28×1.90×1.13`)它**也放行**,
    只是整体很严(一帧 138 簇放 1 个)。留在这里是当**对照**用的:
    见 Plan4 §P-V18 三 —— 若"严门单独一档"与"宽门 + 相机"给出同样的 TP/FP,
    那么**相机的贡献可以被一个更严的门替掉**,融合那一节的结论要重写。
    """
    l, w, h = cl.extent
    if not (1.0 <= h <= 2.6 and 1.3 <= w <= 2.8 and 2.8 <= l <= 5.8):
        return False
    return 1.25 <= l / max(w, 1e-6) <= 4.0


def is_person_shaped(cl: Cluster) -> bool:
    """人形:竖直、粗壮度小。与车形**互斥判**用调用方各自决定(这里只回答"像不像人")。"""
    l, w, h = cl.extent
    span = max(l, w)
    return PERSON_HEIGHT_BAND[0] <= h <= PERSON_HEIGHT_BAND[1] and PERSON_SPAN[0] <= span <= PERSON_SPAN[1]


def size_plausible(cl: Cluster) -> bool:
    """**任一族的形状**(车形或人形)⇒ 交给相机去分。`lidar` 档只能按车形判类。

    ⚠️ 它不再只放车形 —— 那样行人与骑行者的簇会在**入口**就被丢掉,
    上游再准的相机也没有东西可以确认。**这是"缺类"那一半的修法。**

    ★ **刻意给宽**:这一档的设计目标是**高召回、允许大量 FP**,然后由相机那一档去剪
    (见 [`fuse`])。把它调严会让 `lidar` 档一开始召回就低,于是消融变成"两档都差",
    量不出相机的作用 —— 那正是首跑的教训(见 `CAR_HEIGHT_BAND` 那段)。

    用它判的量都是**遮挡下仍然稳定**的:高度 / 最大水平跨度 / 最小水平跨度。
    **不用长宽比** —— 车尾迎面时 l≈w,比值判"不像车",而那是遮挡不是形状。
    """
    if is_person_shaped(cl):
        return True
    l, w, h = cl.extent
    span_max, span_min = max(l, w), min(l, w)
    return (
        CAR_HEIGHT_BAND[0] <= h <= CAR_HEIGHT_BAND[1]
        and CAR_SPAN_MAX[0] <= span_max <= CAR_SPAN_MAX[1]
        and CAR_SPAN_MIN[0] <= span_min <= CAR_SPAN_MIN[1]
    )


def project_box7_to_image(box: Box7, p2: np.ndarray) -> tuple[float, float, float, float] | None:
    """相机系 `Box7` → 2D 像素框 `(x1,y1,x2,y2)`;全落图外/相机后返回 None。

    `p2` 是 **3×4** 投影矩阵(`label_2` 的 `P2`,已含 `R0_rect` 之后的口径)。
    取 8 角点投影的 min/max —— 与采集侧 `gt.core.project_corners` **同一口径**;
    这里不引它是因为那条链吃 `ActorBox`(世界系 + 弧度),而本模块全程在**相机系**,
    自造一条会更省事也更危险 —— 所以只用它的**结论**(min/max over corners),
    实现在 `perception.geometry` 里没有现成的,故落在本函数(有单测钉)。
    """
    from autodrivedata.utils import geometry as g

    corners = g.corners_cam_from_bottom(box.x, box.y, box.z, box.h, box.w, box.l, box.ry)
    hom = np.hstack([corners, np.ones((8, 1))])
    img = (np.asarray(p2, dtype=float) @ hom.T).T
    zc = img[:, 2]
    front = zc > 0.5
    if not front.any():
        return None
    u = img[front, 0] / zc[front]
    v = img[front, 1] / zc[front]
    return float(u.min()), float(v.min()), float(u.max()), float(v.max())


def box2d_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def associate(
    proj: list[tuple[float, float, float, float] | None],
    dets: list[CameraDet],
    iou_thresh: float = ASSOC_IOU,
) -> list[int | None]:
    """簇 → 相机检测的**贪心一对一**关联;没配上的返回 None。

    与 `compare.match_boxes` 同口径(IoU 降序扫、双方都还空着才收)。
    **只有类匹配的才算** —— 这里两侧都只有一类(`Car`),所以门是纯几何的;
    若将来加类,这一条要一起改(否则"位置对但类错"会被算成关联上)。
    """
    cands: list[tuple[float, int, int]] = []
    for i, p in enumerate(proj):
        if p is None:
            continue
        for j, d in enumerate(dets):
            iou = box2d_iou(p, d.xyxy)
            if iou > iou_thresh:
                cands.append((iou, i, j))
    cands.sort(key=lambda t: -t[0])
    out: list[int | None] = [None] * len(proj)
    used_d: set[int] = set()
    for _, i, j in cands:
        if out[i] is not None or j in used_d:
            continue
        out[i] = j
        used_d.add(j)
    return out


def fuse(
    clusters: list[Cluster],
    dets: list[CameraDet],
    velo_to_cam: np.ndarray,
    p2: np.ndarray,
    *,
    mode: str = "lidar+cam",
    drop_unconfirmed: bool = True,
    gate: Callable[[Cluster], bool] = size_plausible,
    prior: dict | None = None,
) -> list[FusedObject]:
    """按 `mode` 出融合表。

    - `"lidar"`:所有**尺寸像车**的簇都出一行,`conf = f(n_points)`。**不看相机。**
    - `"lidar+cam"`:先按 `lidar` 筛,再与相机 2D 框关联;关联上的 `conf = 相机 conf`
      (相机是更强的证据),`drop_unconfirmed=True` 时**丢掉没关联上的**。

    ⚠️ 两档的 `conf` **来源不同**(一个来自点数、一个来自相机),所以两档之间的 AP
    比较**必须同时看 `TP/FP/FN` 计数** —— 只看 AP 会把"排序变了"读成"检测变好了"。
    """
    if mode not in ("lidar", "lidar+cam"):
        raise ValueError(f"未知 mode {mode!r}(合法值:lidar / lidar+cam)")
    keep = [c for c in clusters if gate(c)]
    boxes = [cluster_to_box7(c, velo_to_cam, prior=prior) for c in keep]
    proj = [project_box7_to_image(b, p2) for b in boxes]

    out: list[FusedObject] = []
    if mode == "lidar":
        for c, b in zip(keep, boxes, strict=True):
            out.append(
                FusedObject(
                    box=b,
                    conf=float(min(1.0, c.n_points / 20.0)),  # 点越多越可信
                    sources=("lidar",),
                    cluster_points=c.n_points,
                )
            )
        return out

    hits = associate(proj, dets)
    for c, b, j in zip(keep, boxes, hits, strict=True):
        if j is None:
            if drop_unconfirmed:
                continue
            out.append(
                FusedObject(
                    box=b,
                    conf=float(min(1.0, c.n_points / 20.0)),
                    sources=("lidar",),
                    cluster_points=c.n_points,
                )
            )
        else:
            # ★ **类由相机给** —— 这是融合这一档的全部价值。LiDAR 只出几何。
            #   盒子按相机给的类重建(`label` 换个名字,几何不动)。
            out.append(
                FusedObject(
                    box=Box7(**{**b.__dict__, "label": dets[j].label}),
                    conf=float(dets[j].conf),
                    sources=("lidar", "cam"),
                    cluster_points=c.n_points,
                )
            )
    return out

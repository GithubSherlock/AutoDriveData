"""失效归因:逐帧 2D 匹配 → 漏检按距离/像素高度/TTC/截断分箱(P1 定量延伸)。

为什么要单独一层(而不是塞进 eval_2d_ab):
- eval_2d_ab 的口径是**全库池化 PR**(标准 AP),回答"天气让 AP 掉多少";
  归因要回答"**哪一帧的哪个 GT 漏了、在什么距离/尺度上漏**" → 必须逐帧匹配,
  且每个 GT 要带上上下文(距离、像素高度、TTC、截断、框内亮度对比度)。
- IoU 与匹配口径必须与 eval_2d_ab 同源(同一个 box_iou2d),否则"检出率"
  与 "AP" 两个数字互相解释不通。

纯值层纪律:不 import carla / PIL / ultralytics。图像诊断字段(lum/grad)由
脚本填充,纯值侧默认 NaN,便于单测与两 env 复用。

两种分箱读法(2026-09-09 实测校准,别混用):
- 距离 / 框高箱 = 图像几何量 → **跨速度/跨天气可比**(决定检出率的是尺度)
- TTC 箱 = 安全余量语义 → **不可跨速度比较检出率**(同一 TTC 箱在不同速度
  对应不同距离,尺度就不同);它的用途是把距离断崖换算成"几秒余量之外检不到"
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# 分箱边界(左闭右开,末箱为 +inf)。口径固定写死:跨脚本/跨速度对比必须同一套箱子。
DISTANCE_EDGES: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, math.inf)
HEIGHT_EDGES: tuple[float, ...] = (0.0, 16.0, 32.0, 64.0, 128.0, math.inf)  # 2D 框高(px)
TTC_EDGES: tuple[float, ...] = (0.0, 1.5, 3.0, 4.5, 6.0, 8.0, math.inf)  # 碰撞时间(s)

GT_CLASSES = ("Car", "Pedestrian", "Cyclist")
_COCO_FALLBACK = {
    "car": "Car",
    "truck": "Car",
    "bus": "Car",
    "van": "Car",
    "person": "Pedestrian",
    "bicycle": "Cyclist",
    "motorcycle": "Cyclist",
}


def norm_cls(name: str) -> str:
    """KITTI / COCO 类名 → 本项目三类(Car/Pedestrian/Cyclist);不相关类返回 ""。"""
    n = name.strip().lower()
    if n in _COCO_FALLBACK:
        return _COCO_FALLBACK[n]
    for c in GT_CLASSES:
        if n == c.lower():
            return c
    return ""


@dataclass(frozen=True)
class GtBox2D:
    """一帧内的一个 GT 目标(2D 框 + 归因所需上下文)。"""

    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    truncation: float  # KITTI 列 1:出画角的比例(0=完全在画面内)
    distance_m: float  # KITTI 列 13 = 相机系 z(纵向距离,不是斜距)

    @property
    def width_px(self) -> float:
        return self.x2 - self.x1

    @property
    def height_px(self) -> float:
        return self.y2 - self.y1

    @property
    def area_px(self) -> float:
        return max(0.0, self.width_px) * max(0.0, self.height_px)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def visible(self) -> bool:
        """完全在画面内(截断=0)——"漏检"与"出画"必须分开算。"""
        return self.truncation <= 0.0


@dataclass(frozen=True)
class Detection:
    """一条 2D 预测(脚本侧从 ultralytics 结果转换而来)。"""

    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def parse_gt_line_2d(line: str) -> GtBox2D | None:
    """label_2 一行 → GtBox2D;字段不足/类别不相关返回 None。"""
    p = line.split()
    if len(p) < 15:
        return None
    cls = norm_cls(p[0])
    if not cls:
        return None
    return GtBox2D(
        cls=cls,
        x1=float(p[4]),
        y1=float(p[5]),
        x2=float(p[6]),
        y2=float(p[7]),
        truncation=float(p[1]),
        distance_m=float(p[13]),
    )


def load_gt_2d(root: str | Path, limit: int | None = None) -> dict[str, list[GtBox2D]]:
    """KITTI root → {frame_id: [GtBox2D]}(frame_id 取 label_2 文件名 stem)。"""
    out: dict[str, list[GtBox2D]] = {}
    files = sorted((Path(root) / "training/label_2").glob("*.txt"))
    if limit:
        files = files[:limit]
    for f in files:
        boxes = [b for line in f.read_text().splitlines() if (b := parse_gt_line_2d(line))]
        out[f.stem] = boxes
    return out


def box_iou2d(a: Sequence[float], b: Sequence[float]) -> float:
    """轴对齐矩形 IoU(2D);无交集返回 0。

    与 eval_2d_ab / 3D 侧 compare 的匹配口径同源:全项目只有这一个 2D IoU 实现。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0.0 else 0.0


def match_frame(
    gt: Sequence[GtBox2D],
    preds: Sequence[Detection],
    iou_thr: float = 0.5,
) -> list[bool]:
    """逐帧贪心匹配(conf 降序,每条预测最多认领一个 GT)→ 每个 GT 是否被检出。

    与 eval_2d_ab.ap_for 的匹配顺序一致,区别只在作用域:那边是**全库池化**,
    这边是**逐帧**(归因必须知道是哪一帧漏的)。
    """
    matched = [False] * len(gt)
    for det in sorted(preds, key=lambda d: -d.conf):
        best_i, best_v = -1, 0.0
        for j, g in enumerate(gt):
            if matched[j] or g.cls != det.cls:
                continue
            v = box_iou2d(g.box, det.box)
            if v > best_v:
                best_i, best_v = j, v
        if best_i >= 0 and best_v >= iou_thr:
            matched[best_i] = True
    return matched


def ttc_s(distance_m: float, speed_mps: float) -> float:
    """碰撞时间 = 纵向距离 / 本车速度(s);静止目标 + 定速直行下 TTC 单调于距离。

    注意 TTC 箱**不可跨速度比较检出率**:4.5-6s 箱在 4/8/12 m/s 下分别对应
    18-24m / 36-48m / 54-72m,尺度差一倍(实测检出率 0.96 / 0.13 / 无样本)。
    它的用途是把距离断崖换算成安全余量——"这个速度下 4.5s 以外基本检不到"。
    """
    if speed_mps <= 0.0:
        return math.inf
    return distance_m / speed_mps


@dataclass(frozen=True)
class GtRecord:
    """一个 GT 的归因记录 = GT 上下文 + 是否被检出 + 图像诊断(脚本填)。"""

    frame: str
    cls: str
    distance_m: float
    height_px: float
    truncation: float
    ttc_s: float
    matched: bool
    lum_mean: float = math.nan  # 框内亮度均值(0-255)
    lum_std: float = math.nan  # 框内亮度标准差(局部对比度代理)
    grad_energy: float = math.nan  # 框内梯度能量(锐度/运动模糊代理)
    contrast: float = math.nan  # (框内均值 − 环带均值)/环带 std(局部信噪比)

    @property
    def visible(self) -> bool:
        """完全在画面内(截断=0)——"漏检"与"出画"必须分开算。"""
        return self.truncation <= 0.0


@dataclass
class BinStat:
    """一个分箱的检出统计。"""

    lo: float
    hi: float
    n_gt: int = 0
    n_matched: int = 0
    heights: list[float] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.n_matched / self.n_gt if self.n_gt else math.nan

    @property
    def height_mean(self) -> float:
        return sum(self.heights) / len(self.heights) if self.heights else math.nan


def bin_stats(
    records: Iterable[GtRecord],
    key: Callable[[GtRecord], float],
    edges: Sequence[float] = DISTANCE_EDGES,
) -> list[BinStat]:
    """按 key(records) 落箱统计检出率;越界(NaN/负值)不计入任何箱。"""
    stats = [BinStat(lo=edges[i], hi=edges[i + 1]) for i in range(len(edges) - 1)]
    for r in records:
        v = key(r)
        if math.isnan(v):
            continue
        for s in stats:
            if s.lo <= v < s.hi:
                s.n_gt += 1
                s.n_matched += int(r.matched)
                s.heights.append(r.height_px)
                break
    return stats


def format_bins(stats: Sequence[BinStat], unit: str = "m", label: str = "距离") -> str:
    """分箱表 → 文本(末箱上界 inf 显示为 +)。"""
    lines = [f"  {label:>10s} {'GT':>5s} {'检出':>5s} {'检出率':>7s} {'框高均值':>9s}"]
    for s in stats:
        hi = "+" if math.isinf(s.hi) else f"{s.hi:g}"
        rate = "—" if math.isnan(s.rate) else f"{s.rate:.2f}"
        h = "—" if math.isnan(s.height_mean) else f"{s.height_mean:.0f}px"
        lines.append(f"  {f'{s.lo:g}-{hi}{unit}':>10s} {s.n_gt:5d} {s.n_matched:5d} {rate:>7s} {h:>9s}")
    return "\n".join(lines)


def closing_speed_series(
    frames: Sequence[Sequence[float]],
    delta_s: float,
    window: int = 5,
) -> list[float]:
    """逐帧接近速度(m/s)——数据自证,不信任命令行标称速度。

    每帧取"最远那台静置车"的距离,窗口内中位差分:匀速段它线性下降;
    ego 超过某车时最大距离突降 → 该差分作废(< 5m 过滤),中位数天然剔除。

    为什么要逐帧而不是单一常数(2026-09-09 实测):
    ① 自证命令速度真的生效——set_target_velocity 4/8/12 m/s 命令 → 4.00/8.00/12.00,
       1–2 帧到稳态(无长加速段);
    ② P1 四个 A/B 数据集因残留 brake=1.0 实际只有 6.60 m/s(标称 8.0),
       用标称值算 TTC 会偏小 18%。
    """
    maxd = [max(f) if f else math.nan for f in frames]
    n = len(maxd)
    out = [math.nan] * n
    for i in range(n):
        lo, hi = max(0, i - window), min(n - 1, i + window)
        drops: list[float] = []
        for j in range(lo, hi):
            if math.isnan(maxd[j]) or math.isnan(maxd[j + 1]):
                continue
            drop = maxd[j] - maxd[j + 1]
            if 0.0 < drop < 5.0:
                drops.append(drop)
        if drops:
            drops.sort()
            out[i] = drops[len(drops) // 2] / delta_s
    return out


def estimate_closing_speed(
    frames: Sequence[Sequence[float]],
    delta_s: float,
) -> float:
    """全程接近速度的代表值(m/s)= 逐帧速度的中位数(忽略 ramp 与超车帧)。"""
    series = [v for v in closing_speed_series(frames, delta_s) if not math.isnan(v)]
    if not series:
        return math.nan
    series.sort()
    return series[len(series) // 2]

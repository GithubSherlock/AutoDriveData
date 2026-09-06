"""伪标签 vs GT 比对层(M2,双输出目的所在):3D IoU 匹配 → 分歧帧 → AP/复核率报表。

纯 numpy(无 carla/无 shapely;凸多边形交集 Sutherland-Hodgman)。
KITTI 7 值 [h,w,l,x,y,z,ry](y=底部中心)口径与 auto3dlabel Box3D/iou3d_list 一致
(oracle 单测对照)。3D IoU = BEV 交集 × y 区间交 ÷ 体积并(同 auto3dlabel iou3d)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from autodrivedata import geometry as g


@dataclass
class Box7:
    """KITTI 7 值框(label 额外;y=底部中心;conf 仅伪标签有,GT 恒 1)。"""

    label: str
    h: float
    w: float
    l: float
    x: float
    y: float
    z: float
    ry: float
    conf: float = 1.0
    fit_points: int = 0
    review_flag: bool = False


# ── 几何 ──────────────────────────────────────────────


def box7_to_quad(box: Box7) -> np.ndarray:
    """7 值框 → BEV 4 角点 (4,2) (x,z) 平面(角点序任意,凸)。"""
    c = g.corners_cam_from_bottom(box.x, box.y, box.z, box.h, box.w, box.l, box.ry)
    return c[:4, [0, 2]]


def polygon_area(poly: np.ndarray) -> float:
    """凸多边形 (N,2) 面积(shoelace,绝对值)。"""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2.0)


def _ccw(poly: np.ndarray) -> np.ndarray:
    """归一为逆时针(shoelace 符号为负则反转)——裁剪假设 CCW。"""
    if len(poly) < 3:
        return poly
    x, y = poly[:, 0], poly[:, 1]
    signed = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) / 2.0)
    return poly[::-1] if signed < 0 else poly


def _clip_edge(poly: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman:poly 对边 a→b(左侧为内侧)裁剪。"""
    if len(poly) == 0:
        return poly
    out: list[np.ndarray] = []

    def inside(p: np.ndarray) -> bool:
        return bool(np.cross(b - a, p - a) >= -1e-12)

    prev = poly[-1]
    prev_in = inside(prev)
    for p in poly:
        cur_in = inside(p)
        if cur_in != prev_in:  # 穿越边界:求线段 prev→p 与直线 ab 的交点参数 t
            denom = float(np.cross(b - a, p - prev))
            # 注意:denom 有符号(穿入为正/穿出为负),不可用 max 钳位——钳位会毁掉负值
            t = float(np.cross(b - a, a - prev)) / denom if abs(denom) > 1e-12 else 0.0
            out.append(prev + t * (p - prev))
        if cur_in:
            out.append(p)
        prev, prev_in = p, cur_in
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, 2))


def polygon_intersection(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """两凸多边形交集(返回 (N,2) 或空);输入方向不限,内部归一 CCW。"""
    poly = _ccw(np.asarray(p1, dtype=np.float64))
    clip = _ccw(np.asarray(p2, dtype=np.float64))
    for i in range(len(clip)):
        poly = _clip_edge(poly, clip[i], clip[(i + 1) % len(clip)])
        if len(poly) == 0:
            break
    return poly


def polygon_iou(p1: np.ndarray, p2: np.ndarray) -> float:
    inter = polygon_area(polygon_intersection(p1, p2))
    if inter <= 0:
        return 0.0
    union = polygon_area(p1) + polygon_area(p2) - inter
    return float(inter / union) if union > 0 else 0.0


def box3d_iou(a: Box7, b: Box7) -> float:
    """3D IoU = BEV 交集 × y 区间交 ÷ 体积并(口径同 auto3dlabel iou3d)。"""
    inter_area = polygon_area(polygon_intersection(box7_to_quad(a), box7_to_quad(b)))
    if inter_area <= 0:
        return 0.0
    a_lo, a_hi = a.y, a.y - a.h  # y 向下:底=大、顶=小
    b_lo, b_hi = b.y, b.y - b.h
    y_inter = max(0.0, min(a_lo, b_lo) - max(a_hi, b_hi))
    if y_inter <= 0:
        return 0.0
    vol_inter = inter_area * y_inter
    vol_a, vol_b = a.h * a.w * a.l, b.h * b.w * b.l
    union = vol_a + vol_b - vol_inter
    return float(vol_inter / union) if union > 0 else 0.0


# ── 解析 ──────────────────────────────────────────────


def parse_gt_line(line: str) -> Box7:
    """label_2 GT 行(15 字段)→ Box7(跳过 DontCare/Misc 由调用方决定)。"""
    p = line.split()
    assert len(p) >= 15, f"GT 行字段不足: {line[:40]}"
    return Box7(
        label=p[0],
        h=float(p[8]), w=float(p[9]), l=float(p[10]),
        x=float(p[11]), y=float(p[12]), z=float(p[13]), ry=float(p[14]),
    )


def parse_pred_line(line: str, conf: float = 1.0) -> Box7:
    """auto3dlabel labels/{fid}.txt 行(15 字段,无 conf)→ Box7。"""
    b = parse_gt_line(line)
    b.conf = conf
    return b


def load_gt_labels(path: str | Path) -> list[Box7]:
    p = Path(path)
    if not p.is_file():
        return []
    return [parse_gt_line(ln) for ln in p.read_text().strip().splitlines() if ln.strip()]


def load_pred_labels(path: str | Path, conf: float = 1.0) -> list[Box7]:
    p = Path(path)
    if not p.is_file():
        return []
    return [parse_pred_line(ln, conf) for ln in p.read_text().strip().splitlines() if ln.strip()]


# ── 匹配与统计 ──────────────────────────────────────────


@dataclass
class MatchResult:
    matches: list[tuple[int, int]]  # (gt_idx, pred_idx) 同类别 IoU≥阈值贪心配对
    missed: list[int]  # 未配对的 GT 索引
    extras: list[int]  # 未配对的伪标签索引

    @property
    def n_matches(self) -> int:
        return len(self.matches)


def match_boxes(gt: list[Box7], pred: list[Box7], iou_thresh: float = 0.5) -> MatchResult:
    """同类别贪心匹配(按 IoU 降序取对)。"""
    pairs: list[tuple[float, int, int]] = []
    for i, a in enumerate(gt):
        for j, b in enumerate(pred):
            if a.label != b.label:
                continue
            iou = box3d_iou(a, b)
            if iou >= iou_thresh:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda t: -t[0])
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int]] = []
    for pair in pairs:
        i, j = pair[1], pair[2]
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        matches.append((i, j))
    return MatchResult(
        matches=matches,
        missed=[i for i in range(len(gt)) if i not in used_gt],
        extras=[j for j in range(len(pred)) if j not in used_pred],
    )


@dataclass
class FrameStats:
    frame_id: str
    gt_count: int
    pred_count: int
    matched: int
    missed: int
    extra: int

    @property
    def divergent(self) -> bool:
        """分歧帧:漏检或误报(复核队列据此跳过/入队)。"""
        return self.missed > 0 or self.extra > 0


def ap11(scores: list[float], matched: list[bool], n_gt: int) -> float:
    """11 点插值 AP(score 降序;n_gt=0 返回 0)。"""
    if n_gt == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    tp = np.cumsum(np.asarray(matched, dtype=bool)[order])
    fp = np.cumsum(~np.asarray(matched, dtype=bool)[order])
    n_pred = len(scores)
    recalls = tp / n_gt
    precisions = tp / np.maximum(tp + fp, 1)
    # 对每个 recall 水平取其后最大 precision(包络)
    for i in range(n_pred - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 11):
        idx = np.searchsorted(recalls, r, side="left")
        if idx < n_pred and recalls[idx] >= r:
            ap += precisions[idx]
        # recall 达不到的水平:该点 precision 贡献 0
    return float(ap / 11.0)


@dataclass
class ClassStats:
    gt_count: int = 0
    pred_count: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ap: float = 0.0


@dataclass
class Report:
    per_class: dict[str, ClassStats] = field(default_factory=dict)
    frames: list[FrameStats] = field(default_factory=list)

    @property
    def review_rate(self) -> float:
        """分歧帧占比(该批数据需人工复核的比例)。"""
        if not self.frames:
            return 0.0
        return sum(1 for f in self.frames if f.divergent) / len(self.frames)

    @property
    def total_gt(self) -> int:
        return sum(f.gt_count for f in self.frames)

    @property
    def total_matched(self) -> int:
        return sum(f.matched for f in self.frames)


def evaluate_frames(
    frames: dict[str, tuple[list[Box7], list[Box7]]],
    classes: list[str],
    iou_thresh: float = 0.5,
) -> Report:
    """frames = {frame_id: (gt, pred)}(pred 需带 conf)→ 分帧统计 + 每类 AP。"""
    rep = Report()
    per_class_scores: dict[str, list[tuple[float, bool]]] = {c: [] for c in classes}
    for fid, (gt, pred) in sorted(frames.items()):
        m = match_boxes(gt, pred, iou_thresh)
        matched_gt = {i for i, _ in m.matches}
        matched_pred = {j for _, j in m.matches}
        rep.frames.append(
            FrameStats(
                frame_id=fid,
                gt_count=len(gt),
                pred_count=len(pred),
                matched=len(m.matches),
                missed=len(m.missed),
                extra=len(m.extras),
            )
        )
        for c in classes:
            gt_c = [i for i, b in enumerate(gt) if b.label == c]
            pred_c = [j for j, b in enumerate(pred) if b.label == c]
            for j in pred_c:
                per_class_scores[c].append((pred[j].conf, j in matched_pred))
            stats = rep.per_class.setdefault(c, ClassStats())
            stats.gt_count += len(gt_c)
            stats.pred_count += len(pred_c)
            stats.tp += sum(1 for i in gt_c if i in matched_gt)
            stats.fn += sum(1 for i in gt_c if i not in matched_gt)
            stats.fp += sum(1 for j in pred_c if j not in matched_pred)
    for c in classes:
        scores = per_class_scores[c]
        n_gt = rep.per_class[c].gt_count
        rep.per_class[c].ap = ap11([s for s, _ in scores], [m for _, m in scores], n_gt)
    return rep


def report_text(rep: Report, iou_thresh: float = 0.5) -> str:
    """人类可读报表:每类 AP/TP/FP/FN + 分歧率。"""
    lines = [f"3D IoU 阈值 {iou_thresh} | 帧数 {len(rep.frames)}"]
    lines.append(f"{'类':14s} {'AP':>6s} {'GT':>5s} {'Pred':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
    for c, s in rep.per_class.items():
        lines.append(
            f"{c:14s} {s.ap:6.3f} {s.gt_count:5d} {s.pred_count:5d} "
            f"{s.tp:5d} {s.fp:5d} {s.fn:5d}"
        )
    lines.append(
        f"合计 GT={rep.total_gt} 匹配={rep.total_matched} "
        f"分歧帧={sum(1 for f in rep.frames if f.divergent)}/{len(rep.frames)} "
        f"(复核率 {rep.review_rate:.1%})"
    )
    return "\n".join(lines)

"""MapTR D 阶段评估纯值:Chamfer 距离匹配 + 多阈值 AP(MapTR 官方口径)。

官方评估口径(MapTR 论文/官方评测代码):
- Chamfer 距离 CD(P, Q) = (Σ_p min_q‖p−q‖ + Σ_q min_p‖p−q‖) / (|P| + |Q|),
  预测折线点与 GT 折线点的双向平均最近距离;
- 匹配:逐类贪婪一对一——每个预测找未占用的最近 GT(按 CD 排序),与 GT 复用
  禁止;CD ≤ 阈值的匹配对计 TP,其余预测 FP、未匹配 GT FN;
- AP = 阈值 {0.5, 1.0, 1.5}m 的 precision 均值(固定匹配,只动阈值),与
  autodrivedata.compare.ap11 的 2D 框 AP 互不混用(那是框,这是折线)。
"""

from __future__ import annotations

import numpy as np

# MapTR 官方 chamfer 阈值口径(米)
CHAMFER_THRESHOLDS = (0.5, 1.0, 1.5)


def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """折线 Chamfer 距离 [米]。pred (P, 2),gt (Q, 2)。"""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.size == 0 or gt.size == 0:
        return float("inf")
    d_pq = np.linalg.norm(pred[:, None] - gt[None], axis=-1)  # (P, Q)
    p2q = d_pq.min(axis=1).sum()
    q2p = d_pq.min(axis=0).sum()
    return float((p2q + q2p) / (pred.shape[0] + gt.shape[0]))


def match_greedy(preds: list[np.ndarray], gts: list[np.ndarray], thr: float) -> tuple[int, int, int]:
    """逐类贪婪一对一匹配 → (TP, FP, FN)(阈值 thr 米)。preds/gts 为折线列表。"""
    if not preds:
        return 0, 0, len(gts)
    if not gts:
        return 0, len(preds), 0
    # 候选代价:pred × gt Chamfer 距离
    cost = np.array([[chamfer_distance(p, g) for g in gts] for p in preds])  # (Np, Ng)
    used = np.zeros(len(gts), dtype=bool)
    tp = 0
    for i in np.argsort(cost.min(axis=1)):  # 按最近 GT 距离排序,近者优先占位
        j = int(cost[i].argmin())
        if cost[i, j] <= thr and not used[j]:
            tp += 1
            used[j] = True
    fp = len(preds) - tp
    fn = len(gts) - used.sum()
    return int(tp), int(fp), int(fn)


def chamfer_ap(
    preds: list[np.ndarray], gts: list[np.ndarray], thresholds: tuple[float, ...] = CHAMFER_THRESHOLDS
) -> float:
    """单类 Chamfer AP = 各阈值 precision 均值(官方口径)。"""
    if not preds:
        return 0.0
    precisions = []
    for thr in thresholds:
        tp, fp, _ = match_greedy(preds, gts, thr)
        precisions.append(tp / max(1, tp + fp))
    return float(np.mean(precisions))


def chamfer_ap_per_class(
    preds_by_class: list[list[np.ndarray]],
    gts_by_class: list[list[np.ndarray]],
    thresholds: tuple[float, ...] = CHAMFER_THRESHOLDS,
) -> tuple[list[float], float]:
    """逐类 AP + 类均值。输入与 head.match_assign 的 gt_by_class 同构(类序一致)。"""
    aps = [chamfer_ap(p, g, thresholds) for p, g in zip(preds_by_class, gts_by_class, strict=True)]
    return aps, float(np.mean(aps))

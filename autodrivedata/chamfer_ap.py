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


def chamfer_cost_matrix(preds: list[np.ndarray], gts: list[np.ndarray], chunk: int = 16) -> np.ndarray:
    """批量 Chamfer 代价矩阵 (Np, Ng)——与逐对 chamfer_distance 完全同口径,向量化加速。

    折线点数可变(GT 裁剪后 2..N):按最长补齐,虚点放在 _PAD_XY 远处——真实点
    距离 < 1e3m,min 永不选虚点,求和后再按掩码置零(避免 inf/NaN 参与运算)。
    距离用平方展开 + BLAS GEMM:‖p−q‖² = ‖p‖² + ‖q‖² − 2·p·qᵀ,分块(chunk 个
    预测/块)限制中间张量 (Nc·Lp × Ng·Lq) 的大小。q2p 是逐对口径(每个 GT 点
    对该预测折线自身点的最小),块内直接算,不可跨块累积。
    """
    np_ = len(preds)
    ng = len(gts)
    cost = np.empty((np_, ng), dtype=np.float64)
    if np_ == 0 or ng == 0:
        return cost
    p_pad, p_mask, p_len = _pad_polylines(preds)
    q_pad, q_mask, q_len = _pad_polylines(gts)
    qf = q_pad.reshape(-1, 2)  # (Ng·Lq, 2)
    q_norm2 = (qf * qf).sum(axis=1)  # (Ng·Lq,)

    for c in range(0, np_, chunk):
        pc = p_pad[c : c + chunk]  # (Nc, Lp, 2)
        pm = p_mask[c : c + chunk]
        nc, lp = pc.shape[:2]
        pf = pc.reshape(-1, 2)  # (Nc·Lp, 2)
        sq = (pf * pf).sum(axis=1)[:, None] + q_norm2[None, :] - 2.0 * (pf @ qf.T)
        d = np.sqrt(np.clip(sq, 0.0, None)).reshape(nc, lp, ng, q_pad.shape[1])
        p2q = np.where(pm[:, :, None], 0.0, d.min(axis=3)).sum(axis=1)  # (Nc, Ng)
        q2p = np.where(q_mask[None, :, :], 0.0, d.min(axis=1)).sum(axis=2)  # (Nc, Ng)
        cost[c : c + chunk] = (p2q + q2p) / (p_len[c : c + chunk][:, None] + q_len[None, :])
    return cost


# 补齐虚点坐标:放在远大于任何真实使用距离的位置(本项目 BEV 窗口 ±30m,
# crop 半径 51.2m,真实点距 < 1e2m 量级),min 永不选中虚点
_PAD_XY = 1e4


def _pad_polylines(polys: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """折线列表 → (补齐坐标 (N, L, 2), 掩码 (N, L), 点数 (N,))。虚点放 _PAD_XY 处。"""
    max_len = max(len(p) for p in polys)
    pad = np.full((len(polys), max_len, 2), _PAD_XY)
    mask = np.ones((len(polys), max_len), dtype=bool)
    for i, p in enumerate(polys):
        pad[i, : len(p)] = p
        mask[i, : len(p)] = False
    return pad, mask, np.array([len(p) for p in polys], dtype=np.float64)


def match_greedy(
    preds: list[np.ndarray], gts: list[np.ndarray], thr: float, cost: np.ndarray | None = None
) -> tuple[int, int, int]:
    """逐类贪婪一对一匹配 → (TP, FP, FN)(阈值 thr 米)。preds/gts 为折线列表。

    cost 可传入预计算代价矩阵(多阈值评估时只算一次,见 chamfer_ap)。
    """
    if not preds:
        return 0, 0, len(gts)
    if not gts:
        return 0, len(preds), 0
    # 候选代价:pred × gt Chamfer 距离
    if cost is None:
        cost = chamfer_cost_matrix(preds, gts)
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
    """单类 Chamfer AP = 各阈值 precision 均值(官方口径)。

    代价矩阵与阈值无关,算一次供三阈值共用(旧实现每阈值重算一遍)。
    """
    if not preds:
        return 0.0
    cost = chamfer_cost_matrix(preds, gts)
    precisions = []
    for thr in thresholds:
        tp, fp, _ = match_greedy(preds, gts, thr, cost=cost)
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

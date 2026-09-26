"""Chamfer 代价矩阵 GPU 实现——与 autodrivedata.chamfer_ap.chamfer_cost_matrix 同口径。

只做代价矩阵:贪婪一对一匹配复用纯值版 match_greedy(cost=...),避免匹配逻辑
双实现分叉。torch 依赖限制在本模块(autodrivedata 纯值纪律:不 import torch;
chamfer_ap 经 cost_fn 注入本实现,依赖方向保持 AutoDriveData 单向)。

精度口径:内部 float32 GEMM(与 numpy 向量化版一致),min 后 float64 累加,
误差 ~1e-3m 量级;与 numpy 版交叉锁定(tests/test_chamfer_gpu.py)。补齐虚点
放在 _pad_polylines 的 1e4m 远处——min 永不选中,求和后掩码置零。
"""

from __future__ import annotations

import numpy as np
import torch

from autodrivedata.map.chamfer_ap import _pad_polylines


def chamfer_cost_matrix_cuda(preds: list[np.ndarray], gts: list[np.ndarray], chunk: int = 32) -> np.ndarray:
    """GPU 版 Chamfer 代价矩阵 (Np, Ng) float64。无 CUDA → RuntimeError(调用方回退 CPU)。"""
    if not torch.cuda.is_available():
        raise RuntimeError("chamfer_cost_matrix_cuda 需要 CUDA,请回退 autodrivedata.chamfer_cost_matrix")
    np_ = len(preds)
    ng = len(gts)
    cost = np.empty((np_, ng), dtype=np.float64)
    if np_ == 0 or ng == 0:
        return cost
    p_pad, p_mask, p_len = _pad_polylines(preds)
    q_pad, q_mask, q_len = _pad_polylines(gts)
    dev = torch.device("cuda")
    q_t = torch.from_numpy(q_pad).to(device=dev, dtype=torch.float32)  # (Ng, Lq, 2)
    q_norm2 = (q_t * q_t).sum(dim=2)  # (Ng, Lq)
    q_mask_t = torch.from_numpy(q_mask).to(dev)
    p_len_t = torch.from_numpy(p_len).to(device=dev, dtype=torch.float32)
    q_len_t = torch.from_numpy(q_len).to(device=dev, dtype=torch.float32)
    zero = torch.zeros((), device=dev)

    for c in range(0, np_, chunk):
        pc = torch.from_numpy(p_pad[c : c + chunk]).to(device=dev, dtype=torch.float32)  # (Nc, Lp, 2)
        pm = torch.from_numpy(p_mask[c : c + chunk]).to(dev)
        nc, lp = pc.shape[:2]
        # 2D GEMM:dot[i·Lp+p, g·Lq+q] = Σ_d pc[i,p,d]·q[g,q,d],行/列序恰为 4D 展平
        dots = (pc.reshape(-1, 2) @ q_t.reshape(-1, 2).T).reshape(nc, lp, ng, q_t.shape[1])
        sq = (pc * pc).sum(dim=2)[:, :, None, None] + q_norm2[None, None] - 2 * dots
        sq.clamp_min_(0.0)
        sq.sqrt_()  # d = sqrt(sq),(Nc, Lp, Ng, Lq),原地省两份 196MB 临时块
        p2q = torch.where(pm[:, :, None], zero, sq.min(dim=3).values).sum(dim=1)  # (Nc, Ng)
        q2p = torch.where(q_mask_t[None, :, :], zero, sq.min(dim=1).values).sum(dim=2)  # (Nc, Ng)
        cost[c : c + chunk] = (
            ((p2q + q2p).double() / (p_len_t[c : c + chunk][:, None] + q_len_t[None, :]).double())
            .cpu()
            .numpy()
        )
    return cost

"""点云地面提取(教程 12 升级)——RANSAC 平面拟合 + 网格法双路线。

教程原始做法:网格法统计 + 图像语义分割 + 先验地图 RANSAC 迭代。
本模块升级:**纯几何双路线**(RANSAC 平面拟合 + 网格统计),零依赖(不 import
carla/sklearn),逐帧可跑;输出地面/非地面二值掩码 + 平面参数。对齐教程三流派
中的「拟合」与「几何规则」两派,择优。

口径:
- 输入:单帧 velodyne bin (N,4) x,y,z,intensity(KITTI velodyne 约定,y 左)
- RANSAC:重复采样 3 点拟合平面(z = a·x + b·y + c),统计内点(inlier = |z_hat − z| < thresh)
- 网格法:按 xy 网格统计每格 z 直方图,取主模式(低 z 峰)为地面
- 输出:ground_mask (N,) bool,plane (a,b,c,d),统计(地面点占比/内点占比/平面倾角)

单测:tests/test_ground.py(手算锚点)。
"""

from __future__ import annotations

import numpy as np

# 默认 RANSAC 参数
RANSAC_ITERS = 60
RANSAC_THRESH = 0.25  # 内点距平面阈值(m)
RANSAC_MIN_INLIER = 0.3  # 内点最小占比(低于则判定无地面)


def ransac_plane(
    points: np.ndarray,
    iters: int = RANSAC_ITERS,
    thresh: float = RANSAC_THRESH,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """RANSAC 平面拟合(z = a·x + b·y + c)。

    返回 (plane, inlier_mask);plane = (a,b,c) 满足 z ≈ a·x + b·y + c;
    inlier_mask = |z − (a·x+b·y+c)| < thresh。内点占比 < RANSAC_MIN_INLIER 时返回 None。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_count = 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        A = np.stack([x[idx], y[idx], np.ones(3)], axis=1)
        if np.linalg.matrix_rank(A) < 3:
            continue
        abc, *_ = np.linalg.lstsq(A, z[idx], rcond=None)
        resid = np.abs(z - (abc[0] * x + abc[1] * y + abc[2]))
        inl = resid < thresh
        c = int(inl.sum())
        if c > best_count:
            best_count = c
            best = (abc, inl)
    if best is None or best_count / n < RANSAC_MIN_INLIER:
        return None
    return best


def grid_ground(
    points: np.ndarray, cell: float = 1.0, thresh: float = 0.3, z_low: float = -2.0
) -> np.ndarray:
    """网格法地面掩码(几何规则派)。

    按 xy 网格(cell m)分桶;每格取 z 直方图主模式(最低峰)作为该格地面高度,
    格内 |z − 主模式| < thresh 的点为地面。低 z(≤z_low,阈值外)视为地面兜底。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    keys = np.floor(pts[:, :2] / cell).astype(np.int64)
    uniq = np.unique(keys, axis=0)
    mask = np.zeros(n, dtype=bool)
    for kx, ky in uniq:
        sel = np.where((keys[:, 0] == kx) & (keys[:, 1] == ky))[0]
        if len(sel) < 5:
            continue  # 稀疏格跳过(可能边缘/噪声)
        zs = pts[sel, 2]
        hist, edges = np.histogram(zs, bins=32, range=(z_low, min(zs.max(), 5.0)))
        peak = np.argmax(hist)
        gz = (edges[peak] + edges[peak + 1]) / 2
        mask[sel] = np.abs(zs - gz) < thresh
    # z ≤ z_low 兜底(路面反光边缘/车道线)
    mask |= pts[:, 2] <= z_low
    return mask


def plane_angle_deg(plane: np.ndarray) -> float:
    """平面 (a,b,c) 相对水平面倾角(°):cos = 1/sqrt(a²+b²+1)。"""
    a, b, _ = plane
    return float(np.degrees(np.arccos(1.0 / np.sqrt(a * a + b * b + 1.0))))


def ground_stats(points: np.ndarray, mask: np.ndarray) -> dict:
    """地面掩码统计(供评估对照)。"""
    n = len(mask)
    n_g = int(mask.sum())
    pts = np.asarray(points, dtype=np.float64)
    return {
        "n_points": n,
        "n_ground": n_g,
        "ground_ratio": round(n_g / max(n, 1), 4),
        "z_mean_ground": round(float(pts[mask, 2].mean()), 3) if n_g else None,
        "z_std_ground": round(float(pts[mask, 2].std()), 3) if n_g else None,
    }

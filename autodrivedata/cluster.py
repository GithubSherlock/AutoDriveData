"""点云聚类障碍物检测(教程 13 升级)——地面分割后欧氏聚类 + 3D 包围盒。

教程原始做法:欧氏聚类/DBSCAN 输出无类别障碍簇。本模块升级:**DBSCAN 密度聚类
(自研,零依赖)+ 轴对齐 3D bbox + 簇内统计(点数/高/距)**;不识别类别,换来
零训练/低算力。教程 13 反思:聚类阈值对远距稀疏点敏感 → 提供距离自适应阈值。

口径:
- 输入:非地面点 (N,4) x,y,z,intensity(KITTI velodyne 约定)
- DBSCAN:eps(半径)/min_samples(簇内最小点数),距离用 3D 欧氏。
  **实现分级**:sklearn 可用(KD-tree,工业标准)时直接用;纯 Python 自研网格
  实现作无 sklearn 环境的降级 + 单测锚点。
- 输出:每个簇 (label, center, bbox 半尺寸, 点数, 最近距离)

单测:tests/test_cluster.py(手算锚点)。
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.cluster import DBSCAN as _SK_DBSCAN  # type: ignore[no-redef]

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _SK_DBSCAN = None  # type: ignore[assignment]
    _HAS_SKLEARN = False


def dbscan(
    points: np.ndarray,
    eps: float = 0.8,
    min_samples: int = 5,
    distance_scale: float | None = None,
) -> np.ndarray:
    """DBSCAN 密度聚类 → 簇标签 (-1=噪声)。

    distance_scale = 距离自适应系数(教程 13 远距稀疏点敏感性):对每点,有效 eps =
    eps · (1 + distance_scale · distance_from_origin)。distance_scale=0 时退化为固定 eps。

    优先 sklearn(KD-tree,工业标准);无 sklearn 时降级自研网格实现(单测锚点同口径)。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if distance_scale is None or distance_scale == 0.0:
        if _HAS_SKLEARN:  # pragma: no cover(依赖 sklearn 环境)
            assert _SK_DBSCAN is not None
            return _SK_DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts[:, :3]).astype(np.int64)
        return _dbscan_grid(pts, eps, min_samples, None)
    # 距离自适应(radius 逐点不同)→ 自研网格实现
    return _dbscan_grid(pts, eps, min_samples, distance_scale)


def _dbscan_grid(
    pts: np.ndarray,
    eps: float,
    min_samples: int,
    distance_scale: float | None,
) -> np.ndarray:
    """自研 DBSCAN:体素网格(边长 2·eps)分桶,桶内+邻居桶暴力密度判据。

    复杂度 O(N·bucketsize);大点云(dist > 5000)或距离自适应时走此路径。
    """
    n = pts.shape[0]
    cell = max(eps * 2.0, 1e-3)
    keys = np.floor(pts[:, :3] / cell).astype(np.int64)
    labels = np.full(n, -1, dtype=np.int64)
    from collections import defaultdict

    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i in range(n):
        k = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
        buckets[k].append(i)

    def _eps_at(j: int) -> float:
        if not distance_scale:
            return eps
        return float(eps * (1.0 + distance_scale * np.linalg.norm(pts[j, :3])))

    def _neighbors(j: int) -> list[int]:
        kj = (int(keys[j, 0]), int(keys[j, 1]), int(keys[j, 2]))
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cand.extend(buckets.get((kj[0] + dx, kj[1] + dy, kj[2] + dz), []))
        if not cand:
            return []
        d = np.linalg.norm(pts[cand, :3] - pts[j, :3], axis=1)
        return [cand[t] for t in np.where(d < _eps_at(j))[0]]

    cid = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        nbrs = _neighbors(i)
        if len(nbrs) < min_samples:
            continue
        labels[i] = cid
        stack = [i]
        while stack:
            j = stack.pop()
            j_nbrs = _neighbors(j)
            for nb in j_nbrs:
                if labels[nb] == -1:
                    labels[nb] = cid
                    nb_nbrs = _neighbors(nb)
                    if len(nb_nbrs) >= min_samples:
                        stack.append(nb)
        cid += 1
    return labels


def cluster_boxes(points: np.ndarray, labels: np.ndarray) -> list[dict]:
    """簇 → 3D bbox 描述(轴对齐)。

    每簇:center = 均值,extent = 半尺寸(±),n_points, min_dist(簇内最近点离原点)。
    返回列表,按 n_points 降序(大概率主导障碍)。
    """
    pts = np.asarray(points, dtype=np.float64)
    out: list[dict] = []
    for c in np.unique(labels):
        if c < 0:
            continue
        sel = pts[labels == c]
        center = sel[:, :3].mean(axis=0)
        half = (sel[:, :3].max(axis=0) - sel[:, :3].min(axis=0)) / 2
        min_dist = float(np.linalg.norm(sel[:, :3], axis=1).min())
        out.append(
            {
                "label": int(c),
                "center": center.tolist(),
                "extent_half": half.tolist(),
                "n_points": int(len(sel)),
                "min_dist": round(min_dist, 2),
            }
        )
    out.sort(key=lambda b: -b["n_points"])
    return out

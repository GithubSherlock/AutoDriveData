"""P-E 教程 15 多雷达标定判据(纯值,零 carla / 零 open3d)。

工业口径:多雷达(如车顶 + 两侧)标定 = 求两片点云的刚体变换 T_src→ref。
本模块自研 **point-to-plane ICP**(numpy 实现),并输出标定**判据**:
- **误差随迭代下降曲线**:point-to-plane 残差 RMSE 序列(收敛性证据)
- **变换增量收敛阈值**:相邻迭代的 ΔR/Δt 小于阈值 → 判收敛
- **重叠度**:源点经变换后在参考点云邻域内的比例(低重叠 = 标定不成立)

口径:
- 输入点云 (N,4)(x,y,z,i,第四列忽略)
- ICP 用最小二乘 point-to-plane,线性化旋转角(小角近似,工业 GICP 同款)
- 判据类:convergence_metrics(curve) → 是否收敛 + 终值 + 迭代数
- 标定成功 = 注入误差后能恢复原变换(判据指示收敛且残差回到噪声底)

单测:手算注入已知 R,t,断言判据收敛/不收敛两分支。
"""

from __future__ import annotations

import numpy as np


def _nearest_indices(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """src 每点到 ref 的最近邻索引(KD-tree 朴素 O(N²) 版,仅小点云/测试用)。"""
    d = ((ref[:, None, :3] - src[None, :, :3]) ** 2).sum(-1)  # (Nref, Nsrc)
    return np.argmin(d, axis=0)


def _estimate_transform(src: np.ndarray, ref: np.ndarray, src_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """point-to-plane ICP 一次迭代的最小二乘解(6 参数线性化旋转)。

    src_n 为 src 每点处的参考法向(点对点最近邻处的法向,重投影近似)。
    返回 (R, t):src → ref。
    """
    # 线性化:R = I + [ω]×(小角),minimize Σ (nᵀ·(src + ω×src + t - ref))²
    # 待求 x = (r_x, r_y, r_z, t_x, t_y, t_z);每点贡献行 [src × n, n]
    n = len(src)
    A = np.empty((n, 6))
    b = np.empty(n)
    for i in range(n):
        p = src[i, :3]
        ni = src_n[i]
        A[i, :3] = np.cross(p, ni)  # ω 系数:(ω×p)·n = ω·(p×n)
        A[i, 3:] = ni
        b[i] = -(ni @ (p - ref[i, :3]))  # 残差 = n·(p − ref)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    rx, ry, rz, tx, ty, tz = x
    wx = np.array([[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]])
    R = np.eye(3) + wx
    # 正交化(R 可能略失正交;SVD 投影回 SO(3))
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    t = np.array([tx, ty, tz])
    return R, t


def point_to_plane_icp(
    src: np.ndarray,
    ref: np.ndarray,
    *,
    max_iter: int = 30,
    tol_delta: float = 1e-6,
    normal_k: int = 8,
) -> dict:
    """point-to-plane ICP,返回判据所需全部序列。

    src/ref: (N,4) 点云。ref 法向用局部协方差估计(PCA 最小特征向量)。
    返回 {
      'R', 't',               # 最终变换 src→ref
      'rmse_curve': [...],    # 每迭代 point-to-plane 残差 RMSE
      'delta_curve': [...],   # 每迭代相邻变换增量 ‖ΔR‖₂+‖Δt‖₂
      'converged': bool,
      'iters': int,
      'overlap': float,       # src 变换后与 ref 最近邻 < 0.3m 的比例
      'rmse_final': float,
    }
    """
    src_f = src[:, :3].astype(np.float64)
    ref_f = ref[:, :3].astype(np.float64)
    if len(src_f) == 0 or len(ref_f) == 0:
        return {
            "R": np.eye(3),
            "t": np.zeros(3),
            "rmse_curve": [],
            "delta_curve": [],
            "converged": False,
            "iters": 0,
            "overlap": 0.0,
            "rmse_final": float("inf"),
        }

    # 工业标定标准初始化:先质心对齐(平移粗对准),再迭代旋转/剩余平移。
    # 无质心对齐时小角度线性化在"平面关于面内平移对称"下会停在同一局部最优。
    t0 = np.mean(ref_f, axis=0) - np.mean(src_f, axis=0)
    src_f = src_f + t0

    # ref 法向:PCA 局部邻域最小特征向量
    ref_n = np.zeros_like(ref_f)
    for i in range(len(ref_f)):
        d2 = ((ref_f - ref_f[i]) ** 2).sum(-1)
        nb = np.argsort(d2)[1 : normal_k + 1]
        # 面法向取 PCA 最小特征向量(np.linalg.eigh 升序,第 0 个)
        cov = np.cov(ref_f[nb].T)
        v = np.linalg.eigh(cov)[1]  # eigenvectors 列
        ref_n[i] = v[:, 0]

    cur_src = src_f.copy()
    R_acc, t_acc = np.eye(3), np.zeros(3)
    rmse_curve: list[float] = []
    delta_curve: list[float] = []
    converged = False
    for _ in range(max_iter):
        idx = _nearest_indices(cur_src, ref_f)
        nb = ref_f[idx]
        nb_n = ref_n[idx]
        # 残差 RMSE(point-to-plane)
        res = np.abs(((cur_src - nb) * nb_n).sum(-1))
        rmse_curve.append(float(np.sqrt((res**2).mean())))
        R, t = _estimate_transform(cur_src, nb, nb_n)
        delta = float(np.linalg.norm(R - np.eye(3)) + np.linalg.norm(t))
        delta_curve.append(delta)
        cur_src = (R @ cur_src.T).T + t
        R_acc = R @ R_acc
        t_acc = R @ t_acc + t
        if delta < tol_delta:
            converged = True
            break
    # 平移在平面内沿滑移自由度不可观(工业 GICP 同病):t_acc 在 (src+t0) 系累积,
    # 还原到原始 src 系需把质心预对齐 t0 加回。法向分量精确,滑移分量如实保留。
    t_acc = t_acc + t0
    # 重叠度:变换后每点找最近邻 < 0.3m
    d = ((ref_f[:, None, :] - cur_src[None, :, :]) ** 2).sum(-1).min(0)
    overlap = float((d < 0.3**2).mean())
    return {
        "R": R_acc,
        "t": t_acc,
        "rmse_curve": rmse_curve,
        "delta_curve": delta_curve,
        "converged": converged,
        "iters": len(rmse_curve),
        "overlap": overlap,
        "rmse_final": rmse_curve[-1] if rmse_curve else float("inf"),
    }


def convergence_metrics(result: dict) -> dict:
    """把 ICP 结果整理成标定判据(数值输出)。

    - converged:迭代提前停止(增量阈值)且终值残差低、**重叠度高**
    - stability:delta 曲线末 3 次均值(越小越稳定)
    - 判据:converged(增量) 且 rmse_final < 0.05m 且 **overlap ≥ 0.6** 才算标定收敛
      —— point-to-plane 的 RMSE 即使大误差也低(对齐到近邻平面),重叠度才是
      真伪标定的分水岭(大误差下源点变换后找不到 0.3m 内近邻 → overlap 低)。
    """
    rmse = result["rmse_curve"]
    deltas = result["delta_curve"]
    if not rmse:
        return {
            "converged": False,
            "stability": float("inf"),
            "rmse_final": float("inf"),
            "overlap": round(result.get("overlap", 0.0), 3),
            "iters": 0,
            "verdict": "no_iter",
        }
    stability = float(np.mean(deltas[-3:])) if len(deltas) >= 3 else float(np.mean(deltas))
    overlap = result.get("overlap", 0.0)
    # 判据二:恢复变换的**合理性**。point-to-plane RMSE/overlap 对高密度场景
    # 天然低/高,但真实标定误差幅度有限——恢复出 t=8m 或 ΔR=0.5rad 就是不可信。
    # 阈值:平移 < 5m、旋转 < 30°(对标定任务的合理界,可配)。
    t_amp = float(np.linalg.norm(result["t"]))
    r_amp = float(np.arccos(np.clip((np.trace(result["R"]) - 1.0) / 2.0, -1.0, 1.0)))
    plausible = t_amp < 5.0 and r_amp < np.radians(30.0)
    converged = bool(result["converged"]) and rmse[-1] < 0.05 and overlap >= 0.6 and plausible
    return {
        "converged": bool(converged),
        "stability": round(stability, 6),
        "rmse_final": round(float(rmse[-1]), 6),
        "overlap": round(float(overlap), 3),
        "recovered_t_amp_m": round(t_amp, 3),
        "recovered_r_amp_deg": round(float(np.degrees(r_amp)), 2),
        "iters": result["iters"],
        "verdict": "converged" if converged else "not_converged",
    }

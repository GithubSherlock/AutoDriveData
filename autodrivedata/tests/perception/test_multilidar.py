"""autodrivedata/multilidar.py 手算锚点单测(P-E 多雷达标定判据)。"""

from __future__ import annotations

import numpy as np

from autodrivedata.perception.multilidar import convergence_metrics, point_to_plane_icp


def _plane_cloud(n: int = 120, noise: float = 0.01):
    """xz 平面上的平面点云(z=0 面,即 y=0),法向 = ±y。"""
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 5, n)
    z = rng.uniform(-5, 5, n)
    y = rng.normal(0, noise, n)
    return np.column_stack([x, y, z, np.ones(n)])


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _apply(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = pts.copy()
    out[:, :3] = (R @ pts[:, :3].T).T + t
    return out


class TestPointToPlaneIcp:
    def test_recovers_known_transform(self):
        # 加 0.1 rad / 0.3m 误差:源点云 = 平面 + 刚体变换,label ICP 应恢复逆变换
        R_true = _rot_y(0.1)
        t_true = np.array([0.3, -0.2, 0.1])
        ref = _plane_cloud()
        src = _apply(ref, R_true, t_true)
        res = point_to_plane_icp(src, ref)
        # 组合误差:src → (res) = R@R_true,应 ≈ I
        R_err = res["R"] @ R_true
        t_err = res["t"] + res["R"] @ t_true
        np.testing.assert_allclose(R_err, np.eye(3), atol=0.05)
        np.testing.assert_allclose(t_err, np.zeros(3), atol=0.05)
        assert res["converged"]

    def test_large_initial_error_no_converge(self):
        # 大幅误差(1.2 rad ≈ 69°)超出线性化域 → 判据指示不收敛(如实报)。
        # 注意:point-to-plane 的 RMSE 对"对齐到近邻平面"本身会下降(残差小),
        # 但判据看**重叠度 + 未收敛布尔**:重叠 <0.6 即标定不成立。
        R_true = _rot_y(1.2)
        ref = _plane_cloud()
        src = _apply(ref, R_true, np.array([2.0, 0.0, 1.0]))
        res = point_to_plane_icp(src, ref, max_iter=20)
        cfg = convergence_metrics(res)
        assert cfg["verdict"] == "not_converged"
        assert cfg["overlap"] < 0.6  # 只有不到 6 成点找到 0.3m 内近邻 → 对齐不成立
        assert not res["converged"]

    def test_curve_descending(self):
        # 小误差:判据曲线单调下降(收敛性核心判据)
        R_true = _rot_y(0.05)
        ref = _plane_cloud()
        src = _apply(ref, R_true, np.array([0.1, 0.0, 0.1]))
        res = point_to_plane_icp(src, ref)
        curve = res["rmse_curve"]
        assert curve[-1] < curve[0]
        # 无回跳:非严格单调,但末段应低于首段
        assert res["converged"]
        cfg = convergence_metrics(res)
        assert cfg["verdict"] == "converged"


class TestConvergenceMetrics:
    def test_empty(self):
        cfg = convergence_metrics({"rmse_curve": [], "delta_curve": [], "overlap": 0.0})
        assert cfg["verdict"] == "no_iter"

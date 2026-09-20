"""激光 SLAM 纯值单测(教程 14 两段式:slam.py 手算锚点)。

风格沿用仓库:类按被测函数分组,手算锚点 + 边界,探测 SE3/网格哈希/ICP/
ScanContext/位姿图 语义。全纯 numpy,零 carla / 零 torch。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.accum import voxel_downsample
from autodrivedata.slam import (
    GRID_OFFSETS,
    Edge,
    GridHash,
    closure_error,
    desc_scan_context,
    estimate_transform_gn,
    icp_odometry,
    match_sc,
    nearest_batch,
    pose_graph_optimize,
    relative_transform,
    ring_key,
    sc_candidates,
    sc_dist,
    twist_exp,
    twist_log,
)


# ---------------------------------------------------------------------------
# SE(3) 原语
# ---------------------------------------------------------------------------
class TestSe3:
    def test_exp_log_roundtrip(self):
        w = np.array([0.1, -0.2, 0.3])
        v = np.array([0.2, 0.4, -0.5])
        T = twist_exp(w, v)
        w2, v2 = twist_log(T)
        np.testing.assert_allclose(w2, w, atol=1e-8)
        np.testing.assert_allclose(v2, v, atol=1e-6)

    def test_exp_identity_zero(self):
        T = twist_exp(np.zeros(3), np.zeros(3))
        np.testing.assert_allclose(T, np.eye(4), atol=1e-9)

    def test_log_identity_zero(self):
        w, v = twist_log(np.eye(4))
        np.testing.assert_allclose(w, np.zeros(3), atol=1e-9)
        np.testing.assert_allclose(v, np.zeros(3), atol=1e-9)

    def test_relative_transform_identity(self):
        T = twist_exp(np.array([0.1, 0.0, 0.0]), np.array([1.0, 2.0, 3.0]))
        # T⁻¹·T = identity(内部 4×4 全等)
        rel = relative_transform(T, T)
        np.testing.assert_allclose(rel, np.eye(4), atol=1e-9)


# ---------------------------------------------------------------------------
# 网格哈希
# ---------------------------------------------------------------------------
class TestGridHash:
    def test_nearest_vs_bruteforce(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(-5, 5, size=(300, 3))
        tree = GridHash(pts, cell=0.5)
        q = rng.uniform(-5, 5, size=(20, 3))
        for x in q:
            i, d2 = tree.nearest(x)
            # 暴力
            bd = float(((pts[:, :3] - x) ** 2).sum(-1).min())
            assert d2 == pytest.approx(bd, rel=1e-9)
            np.testing.assert_allclose(
                pts[i, :3], pts[int(((pts[:, :3] - x) ** 2).sum(-1).argmin())], atol=1e-9
            )

    def test_tie_first_encountered_wins(self):
        # 等距台面:两个点关于查询对称,扫描序先者胜
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        tree = GridHash(pts, cell=100.0)  # 所有点同一格
        # 查询在 (1.5,0,0):等距候选是 idx1(2.0)与 idx2(1.0)?——1.5 距 2.0=0.5,距 1.0=0.5 等距
        i, _ = tree.nearest(np.array([1.5, 0.0, 0.0]))
        # 扫描序 idx0(0,0,0):距 2.25;idx1(2,0,0):0.25;idx2(1,0,0):0.25
        # 等距 0.25 → 先遇到 idx1(扫描序靠前)胜
        assert i == 1

    def test_empty_cloud(self):
        tree = GridHash(np.zeros((0, 3)), cell=0.5)
        i, d = tree.nearest(np.array([0.0, 0.0, 0.0]))
        assert i == -1
        assert d == float("inf")

    def test_rad0_early_stop_regression(self):
        """rad=0 不允许提前停:自身格内有点 ≠ 全局最近,层 1 可能有更近点。

        旧实现层 0 扫完就按 best<cell² 早停,自身格 236 距 0.234 却漏掉层1
        格 869 距 0.169(曾实测 6.7% 次优)。修正后返回 = 层1 的真最近。
        """
        rng = np.random.default_rng(0)
        pts = rng.uniform(-8, 8, (2000, 3))
        q = pts[1908]
        tree = GridHash(pts, cell=0.5)
        i, d2 = tree.nearest(q)
        assert i >= 0
        bd = float(((pts[:, :3] - q) ** 2).sum(-1).min())
        assert d2 == pytest.approx(bd, rel=1e-9)
        # 自身格所有候选都在更远处(层1 格 869 才是真最近)
        cell_c = tuple(np.floor(q / 0.5).astype(int))
        assert all(float(((pts[j, :3] - q) ** 2).sum()) >= d2 for j in tree.buckets.get(cell_c, ()))

    def test_rad0_early_stop_scalar_vs_bruteforce(self):
        """多分辨率随机云:标量最近邻(修正早停)与暴力 0 不一致。"""
        rng = np.random.default_rng(0)
        for cell in (0.5, 1.0):
            pts = rng.uniform(-20, 20, (1000, 3))
            tree = GridHash(pts, cell=cell)
            for q in rng.uniform(-20, 20, (8, 3)):
                i, d2 = tree.nearest(q)
                assert i >= 0
                bd = float(((pts[:, :3] - q) ** 2).sum(-1).min())
                assert d2 == pytest.approx(bd, rel=1e-9)
                np.testing.assert_allclose(
                    pts[i, :3],
                    pts[int(((pts[:, :3] - q) ** 2).sum(-1).argmin()), :3],
                    atol=1e-9,
                )


class TestNearestBatch:
    """nearest_batch vs 标量 GridHash.nearest:逐位一致(位对齐契约探针)。"""

    def _allclose(self, ref, qs, tag, cell=0.5):
        idx_b, d2_b = nearest_batch(ref, qs, cell)
        tree = GridHash(ref, cell=cell)
        idx_s = np.array([tree.nearest(q)[0] for q in qs])
        d2_s = np.array([tree.nearest(q)[1] for q in qs])
        assert (idx_b == idx_s).all(), f"[{tag}] idx mismatch"
        assert (d2_b == d2_s).all(), f"[{tag}] d2 mismatch"

    def test_random_multi_res(self):
        rng = np.random.default_rng(0)
        for seed in (1, 2, 3):
            ref = rng.uniform(-50, 50, (4000, 3))
            qs = rng.uniform(-50, 50, (1500, 3))
            self._allclose(ref, qs, f"rand{seed}")

    def test_dense_grid_ties(self):
        """稠密近等距格子(最易触 tie-break):全对拍一致。"""
        rng = np.random.default_rng(0)
        pts = np.mgrid[-2:2:0.11, -2:2:0.11, -1:1:0.25].reshape(3, -1).T
        pts = pts + rng.uniform(-1e-3, 1e-3, pts.shape)
        self._allclose(pts, rng.uniform(-2, 2, (800, 3)), "dense")

    def test_many_same_bucket(self):
        """全部同格(大桶):批量路径不能只取第一格。"""
        rng = np.random.default_rng(0)
        ref = rng.uniform(-0.4, 0.4, (3000, 3))
        self._allclose(ref, rng.uniform(-0.4, 0.4, (300, 3)), "one-bucket")

    def test_cell_param(self):
        rng = np.random.default_rng(0)
        ref = rng.uniform(-10, 10, (2000, 3))
        qs = rng.uniform(-10, 10, (300, 3))
        self._allclose(ref, qs, "cell1.0", cell=1.0)


# ---------------------------------------------------------------------------
# 点面 ICP
# ---------------------------------------------------------------------------
def _plane_cloud(n: int = 200, z: float = 0.0) -> np.ndarray:
    """xy 平面(z=const)点云 (N,3)。"""
    rng = np.random.default_rng(0)
    x = rng.uniform(-10, 10, n)
    y = rng.uniform(-10, 10, n)
    return np.stack([x, y, np.full(n, z)], 1)


class TestEstimateTransformGn:
    def test_matches_multilidar_estimate_on_plane(self):
        """λ=1e-4 正则化解与 multilidar._estimate_transform(lstsq)在平面夹具上差 <1e-4。"""
        from autodrivedata.multilidar import _estimate_transform

        src = _plane_cloud(150)
        ref = _plane_cloud(150)
        ref += np.array([0.1, -0.2, 0.0])  # 已知平移
        ref_n = np.tile(np.array([0.0, 0.0, 1.0]), (len(ref), 1))
        # multilidar 是 src→ref 累积步(小角线性化)
        R_ml, t_ml = _estimate_transform(src, ref, ref_n)
        R_gn, t_gn = estimate_transform_gn(src, ref, ref_n, lam=1e-4)
        np.testing.assert_allclose(R_gn, R_ml, atol=1e-4)
        np.testing.assert_allclose(t_gn, t_ml, atol=1e-4)

    def test_plane_only_no_NaN(self):
        """纯平面(法向恒定)时正则化解保持有限(不 NaN)。"""
        src = _plane_cloud(100)
        ref = _plane_cloud(100) + np.array([0.5, 0.0, 0.0])
        ref_n = np.tile(np.array([0.0, 0.0, 1.0]), (len(ref), 1))
        R, t = estimate_transform_gn(src, ref, ref_n, lam=1e-4)
        assert np.isfinite(R).all() and np.isfinite(t).all()
        assert np.linalg.norm(R - np.eye(3)) < 1e-3


class TestIcpOdometry:
    def test_recovers_known_transform(self):
        """src 加已知 (R,t) 得 ref ⇒ **点映射 T_delta** 恢复 (R_gt,t_gt),位姿 T = inv(T_delta)。

        两个出口别混:ICP 直接解的是"把 src 点搬到 ref 系"的点映射;`T` 才是位姿
        (= init @ inv(T_delta))。旧版把点映射当位姿返回,本测试曾按 T ≈ (R_gt,t_gt)
        断言 —— 那个断言本身就在固化 bug。
        """
        rng = np.random.default_rng(1)
        src = np.stack([rng.uniform(-8, 8, 300), rng.uniform(-8, 8, 300), rng.uniform(-1, 1, 300)], 1)
        # 已知点映射:ref = R_gt·src + t_gt
        theta = np.radians(10.0)
        R_gt = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        t_gt = np.array([1.5, -0.8, 0.2])
        ref = (R_gt @ src.T).T + t_gt
        res = icp_odometry(src, ref, np.eye(4))
        np.testing.assert_allclose(res["T_delta"][:3, :3], R_gt, atol=0.05)
        np.testing.assert_allclose(res["T_delta"][:3, 3], t_gt, atol=0.05)
        # init=恒等 ⇒ 位姿 = inv(点映射)
        np.testing.assert_allclose(res["T"], np.linalg.inv(res["T_delta"]), atol=1e-12)
        assert res["overlap"] > 0.8

    def test_chain_of_turning_motion_matches_ground_truth(self):
        """**回归(旧版 ATE 大 94× 的根因)**:带转向的链式位姿必须复现 GT。

        旧实现 `T = T_delta @ init_T` 把点映射当位姿左乘:纯平移时"看着像在累加",
        一旦有旋转就发散(实测 206 m 真实序列 ATE 17.5 m vs 修正后 0.19 m)。
        本测试用**每帧 6° 转向**的合成序列把该约定钉死 —— 旧实现下位置误差按帧数线性增长
        (实测 k=7 时 6.35 m),修正后 <0.1 m。
        """
        rng = np.random.default_rng(3)
        g = np.stack([rng.uniform(-6, 6, 300), rng.uniform(-6, 6, 300), np.zeros(300)], 1)
        # 两堵垂直于运动方向的墙:点面残差只在法向有约束,纯地面测不出 x 平移
        w1 = np.stack([rng.uniform(-6, 6, 200), np.full(200, -6.0), rng.uniform(0, 3, 200)], 1)
        w2 = np.stack([rng.uniform(-6, 6, 200), np.full(200, 6.0), rng.uniform(0, 3, 200)], 1)
        base = np.concatenate([g, w1, w2])

        # GT 位姿:每帧前进 0.5 m + 左转 6°(点云 = 世界点投到该帧传感器系)
        gt: list[np.ndarray] = []
        frames: list[np.ndarray] = []
        P = np.eye(4)
        for _ in range(8):
            gt.append(P.copy())
            p = (P[:3, :3].T @ base.T).T + (P[:3, :3].T @ -P[:3, 3])
            frames.append(voxel_downsample(np.hstack([p, np.ones((len(p), 1))]), 0.5)[:, :3])
            c, s = np.cos(np.radians(6.0)), np.sin(np.radians(6.0))
            A = np.eye(4)
            A[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            A[:3, 3] = [0.5, 0.0, 0.0]
            P = P @ A

        poses = [np.eye(4)]
        delta_prev = None  # 恒速先验 = 位姿增量(函数内部取逆)
        for k in range(1, 8):
            res = icp_odometry(frames[k - 1], frames[k], poses[-1], seed=delta_prev)
            poses.append(res["T"])
            delta_prev = relative_transform(poses[-2], poses[-1])

        for k in range(1, 8):
            assert np.linalg.norm(poses[k][:3, 3] - gt[k][:3, 3]) < 0.1
            dR = poses[k][:3, :3].T @ gt[k][:3, :3]
            ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
            assert ang < 0.2

    def test_constant_velocity_prior_chain(self):
        """三级点云(点集每级 +0.5x)链式**位姿**累计 = −1.0。

        点云整体 +0.5x ⟺ 传感器相对世界 −0.5x ⇒ 位姿平移为负。首帧 |t| = 0.5、
        两帧累计 |t| = 1.0(只断模长,方向由场景定义)。
        """
        rng = np.random.default_rng(2)
        base = np.stack([rng.uniform(-6, 6, 250), rng.uniform(-6, 6, 250), rng.uniform(-1, 1, 250)], 1)
        cam0 = base.copy()
        cam1 = base + np.array([0.5, 0.0, 0.0])
        cam2 = base + np.array([1.0, 0.0, 0.0])
        Tm1 = icp_odometry(cam0, cam1, np.eye(4))["T"]
        T_chain = icp_odometry(cam1, cam2, Tm1)["T"]
        assert np.linalg.norm(Tm1[:3, 3]) == pytest.approx(0.5, abs=0.05)
        assert np.linalg.norm(T_chain[:3, 3]) == pytest.approx(1.0, abs=0.05)

    def test_empty_cloud_failed(self):
        res = icp_odometry(np.zeros((0, 4)), np.zeros((0, 4)), np.eye(4))
        assert res["failed"] is True
        assert res["overlap"] == 0.0


# ---------------------------------------------------------------------------
# ScanContext
# ---------------------------------------------------------------------------
class TestScanContext:
    def _ring_cloud(self, radius: float, n: int = 400) -> np.ndarray:
        """半径 radius 圆环点云(点都在该环上,描述子应为单环带宽 1)。"""
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(n)], 1)

    def test_sc_dist_rotated_cloud(self):
        """90° 旋转的点云 → 描述子应滚动等价(shift 90/60 圈 → 距离≈0)。"""
        a = self._ring_cloud(20.0)
        # 绕 z 转 90° 并随机平移(描述子是原点中心,旋转 → 列滚动)
        th = np.radians(90.0)
        b = np.stack(
            [
                20.0 * np.cos(np.linspace(0, 2 * np.pi, 400, endpoint=False) + th),
                20.0 * np.sin(np.linspace(0, 2 * np.pi, 400, endpoint=False) + th),
                np.zeros(400),
            ],
            1,
        )
        da = desc_scan_context(a)
        db = desc_scan_context(b)
        _, _, dist = match_sc(np.stack([da]), db)
        assert dist < 0.05  # 滚动匹配后应几乎相同

    def test_sc_dist_different_cloud(self):
        """不同半径/形状点云 → 距离明显大。"""
        a = self._ring_cloud(20.0)
        b = self._ring_cloud(35.0)
        d = sc_dist(desc_scan_context(a), desc_scan_context(b))
        assert d > 0.5

    def test_ring_key_anchor(self):
        """已知小描述子 → ring_key 的 argmax 扇区。"""
        desc = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        rk = ring_key(desc)
        assert rk.tolist() == [1, 0]

    def test_sc_candidates_min_gap(self):
        """与本帧差 < min_gap 的候选被排除。"""
        descs = [desc_scan_context(self._ring_cloud(r)) for r in (10.0, 20.0, 30.0, 40.0)]
        descs = np.stack(descs)
        q = descs[3]
        cands = sc_candidates(descs, q, node_idx=3, top_n=2, min_gap=2, thresh=0.5)
        # 与 idx3 差 <2 的(0,1)被排除,只剩 2;但 2 差 1 也被排除 → 空
        assert cands == []

    def test_sc_candidates_default_min_gap_is_node_scale(self):
        """默认 min_gap = SC_MIN_GAP_NODES(25 节点),**不是 3**。

        回归:曾把 bin/slam_backend.py 的 `--min-gap` 默认设成 3,在"原地静止"
        数据集上 15 个关键帧互相全部落入候选窗(描述子天然几乎相同)→ 74/75 候选
        全过门 = 假回环。判据:完全相同的描述子序列里,默认参数只允许关键帧号差
        ≥ SC_MIN_GAP_NODES 的对成为候选。
        """
        from autodrivedata.slam import SC_MIN_GAP_NODES

        assert SC_MIN_GAP_NODES >= 10  # 关键帧尺度的下限(KEYFRAME_EVERY=10)
        n = SC_MIN_GAP_NODES + 6
        descs = np.stack([desc_scan_context(self._ring_cloud(20.0)) for _ in range(n)])
        q = descs[-1]
        cands = sc_candidates(descs, q, node_idx=n - 1, top_n=n, thresh=0.5)
        assert cands, "完全相同描述子应产生候选(否则测试无意义)"
        assert all(abs(c - (n - 1)) >= SC_MIN_GAP_NODES for c, _, _ in cands)
        # 相邻关键帧(差 1)必须不在候选里——假回环的典型来源
        assert all(abs(c - (n - 1)) > 1 for c, _, _ in cands)


# ---------------------------------------------------------------------------
# 位姿图 G-N
# ---------------------------------------------------------------------------
class TestPoseGraph:
    def _square_gt(self, size: float = 20.0) -> list[np.ndarray]:
        """正三角形 4 节点位姿 GT(node3 = node0,闭合)。

        node0=(0,0,0°) → node1=(20,0,60°) → node2=(10,17.32,120°) → node3=node0。
        每边 20m,弧长 60;位姿含 yaw(沿边朝向),闭合误差 ≈ 0。
        """
        vertices = [(0.0, 0.0, 0.0), (size, 0.0, 60.0), (size / 2, size * np.sqrt(3) / 2, 120.0)]
        poses = []
        for x, y, yaw_deg in vertices:
            T = np.eye(4)
            th = np.radians(yaw_deg)
            T[:3, :3] = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
            T[:3, 3] = np.array([x, y, 0.0])
            poses.append(T)
        poses.append(poses[0].copy())  # node3 = node0(闭合)
        return poses

    def test_square_closure(self):
        """闭合三角形 4 节点闭合误差 0(首尾同位)。"""
        poses = self._square_gt()
        ce = closure_error(poses)
        assert ce["drift_m"] < 0.05
        assert ce["arc_m"] == pytest.approx(3 * 20.0, rel=0.05)

    def test_pgo_corrects_noisy_odom(self):
        """加噪里程计边 + 1 正确回环 → 优化将闭合误差降到小(≤1m)。"""
        gt = self._square_gt()
        # 里程计边:GT 相对位姿 + 噪声
        rng = np.random.default_rng(0)
        edges = []
        for k in range(3):
            rel = relative_transform(gt[k], gt[k + 1])
            noise = twist_exp(rng.uniform(-0.08, 0.08, 3), rng.uniform(-0.6, 0.6, 3))
            edges.append(Edge(k, k + 1, noise @ rel, weight=1.0))
        # 回环:节点3(==node0 初始) → 节点0
        loop = Edge(3, 0, np.eye(4), weight=0.5)
        edges.append(loop)
        # 初始化:GT + 噪声(2D 平面,60m 弧长 → ±0.6m 平移噪声闭合误差天然 ~1m+
        # 但 PGO 一次 G-N 即把闭合差降到 <0.5m——验证"回环有效"而非"相对改善")
        noisy_gt = self._square_gt()
        noisy = []
        for p in noisy_gt:
            noisy.append(p @ twist_exp(rng.uniform(-0.1, 0.1, 3), rng.uniform(-0.6, 0.6, 3)))
        opt = pose_graph_optimize(noisy, edges)
        ce_before = closure_error(noisy)
        ce_after = closure_error(opt)
        # 优化后闭合误差必须显著小于初始(informative);且绝对 ≤1m(60m 弧长 1.7% 漂移)
        assert ce_after["drift_m"] < max(ce_before["drift_m"] * 0.5, 0.5)
        assert ce_after["drift_m"] < 1.0  # 绝对收敛


# ---------------------------------------------------------------------------
# 端到端烟测(小序列)
# ---------------------------------------------------------------------------
class TestSmoke:
    def test_straight_cloud_sequence_drift_near_zero(self):
        """直线结构化点云序列(点集每帧 +0.5x)→ 链式**位姿**应为直线、横向/高度不漂。

        **场景必须对运动方向可观**:点面残差只在法向有约束。纯水平地面(法向全 =±z)
        对平面内的 x 平移**零约束** → ICP 返回 t=0 才是正确的最小范数解。故此处加两堵
        **垂直于运动方向**的墙,使 x 平移有观测量;地面保留作为 z/y 约束。

        符号:点云整体 +0.5x ⟺ 传感器相对世界 −0.5x ⇒ 位姿 x 累计 **−2.5**(5 步)。
        **开放路径不适用 closure_error**(首末位姿距离 = 弧长,非漂移)。
        """
        rng = np.random.default_rng(3)
        g = np.stack([rng.uniform(-6, 6, 300), rng.uniform(-6, 6, 300), np.zeros(300)], 1)
        w1 = np.stack([rng.uniform(-6, 6, 200), np.full(200, -6.0), rng.uniform(0, 3, 200)], 1)
        w2 = np.stack([rng.uniform(-6, 6, 200), np.full(200, 6.0), rng.uniform(0, 3, 200)], 1)
        base = np.concatenate([g, w1, w2])
        frames = [base + np.array([0.5 * k, 0.0, 0.0]) for k in range(6)]
        # 每帧下采样 + velodyne 4 列拼上强度列
        frames = [voxel_downsample(np.hstack([f, np.ones((len(f), 1))]), 0.5) for f in frames]
        poses = [np.eye(4)]
        for k in range(1, len(frames)):
            res = icp_odometry(frames[k - 1], frames[k], poses[-1])
            poses.append(res["T"])
        x_along = poses[-1][0, 3]  # 累计 x(m);点云 +x ⇒ 位姿 −x
        assert x_along == pytest.approx(-2.5, abs=0.05)
        assert abs(poses[-1][1, 3]) < 0.05  # 横向不漂
        assert abs(poses[-1][2, 3]) < 0.05  # 高度不漂


# ---------------------------------------------------------------------------
# GRID_OFFSETS 字典序(对拍契约探针)
# ---------------------------------------------------------------------------
class TestGridOffsetsOrder:
    def test_lexicographic(self):
        """GRID_OFFSETS 必须为钉死序(z 外循环 → y → x,每个 -1..1 递增)——C++ 硬编码同序。"""
        # _GRID_OFFSETS 按 (dz, dy, dx) 从内到外循环(dz 最外)
        expected = [(dx, dy, dz) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        assert [(int(o[0]), int(o[1]), int(o[2])) for o in GRID_OFFSETS] == expected
        # 相邻两行按 (dz, dy, dx) 排序:先 z 后 y 后 x(z 变化最慢)
        for a, b in zip(GRID_OFFSETS[:-1], GRID_OFFSETS[1:], strict=True):
            assert (int(a[2]), int(a[1]), int(a[0])) <= (int(b[2]), int(b[1]), int(b[0]))

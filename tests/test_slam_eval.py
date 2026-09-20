"""slam_eval.py 单测:Umeyama 对齐 / ATE / RPE 的手算锚点 + 退化保护。

锚点全部可手算(见各 case 注释),不依赖 CARLA。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autodrivedata.slam_eval import (
    LIDAR_LEVER,
    M_FLIP,
    align_trajectory,
    ate,
    eval_trajectory,
    lever_matrix,
    lidar_pose_to_ego,
    rpe,
    umeyama_alignment,
)


def _T(x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    """平移 + 绕 z 旋转的 4×4。"""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = [x, y, z]
    return T


class TestUmeyama:
    def test_identity_when_already_aligned(self):
        pts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
        R, t, s = umeyama_alignment(pts, pts)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-12)
        assert s == 1.0

    def test_recovers_known_rigid_transform(self):
        # 手算锚点:src 绕 z 转 90° → (x,y) → (−y,x);再平移 (10, 20, −3)
        src = np.array([[1.0, 0, 0], [0, 1, 0], [1, 1, 0], [2, -1, 0.5]])
        R_gt = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
        t_gt = np.array([10.0, 20.0, -3.0])
        dst = (R_gt @ src.T).T + t_gt
        R, t, s = umeyama_alignment(src, dst)
        np.testing.assert_allclose(R, R_gt, atol=1e-12)
        np.testing.assert_allclose(t, t_gt, atol=1e-12)
        assert abs(s - 1.0) < 1e-12

    def test_scale_recovered_only_when_requested(self):
        src = np.array([[1.0, 0, 0], [0, 1, 0], [1, 1, 0], [2, -1, 0.5]])
        dst = 2.5 * src  # 纯缩放:最优 R=I, s=2.5
        _, _, s0 = umeyama_alignment(src, dst, with_scale=False)
        _, t1, s1 = umeyama_alignment(src, dst, with_scale=True)
        assert s0 == 1.0  # 不给尺度自由度就固定 1
        assert abs(s1 - 2.5) < 1e-12
        np.testing.assert_allclose(t1, np.zeros(3), atol=1e-12)

    def test_reflection_guard_gives_proper_rotation(self):
        # 构造镜像数据:无保护时 SVD 解会给 det=−1 的"反射",ATE 会假性偏小
        src = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        dst = src.copy()
        dst[:, 0] *= -1  # 沿 x 镜像
        R, _, _ = umeyama_alignment(src, dst)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9  # 必须是真旋转
        assert np.linalg.det(R) > 0

    def test_shape_and_count_guards(self):
        with pytest.raises(ValueError):
            umeyama_alignment(np.zeros((4, 2)), np.zeros((4, 2)))
        with pytest.raises(ValueError):
            umeyama_alignment(np.zeros((4, 3)), np.zeros((5, 3)))
        with pytest.raises(ValueError):
            umeyama_alignment(np.zeros((2, 3)), np.zeros((2, 3)))  # <3 点


class TestAlignTrajectory:
    def test_pose_block_is_rotated_too(self):
        # 位置对齐求出的 R 必须**同时施加到姿态块**:否则 RPE 旋转项会假性偏大
        gt = [_T(0, 0), _T(1, 0, yaw_deg=10), _T(2, 0, yaw_deg=20)]
        R_gt = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90°
        t_gt = np.array([5.0, -7.0, 0.0])
        A = np.eye(4)
        A[:3, :3] = R_gt
        A[:3, 3] = t_gt
        est = [A @ T for T in gt]  # est = 已对齐的 gt → 对齐应把它还原
        out = align_trajectory(est, gt)
        for a, g in zip(out["aligned"], gt, strict=True):
            np.testing.assert_allclose(a, g, atol=1e-9)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            align_trajectory([_T(0, 0), _T(1, 0)], [_T(0, 0)])


class TestATE:
    def test_zero_when_identical(self):
        gt = [_T(i * 1.0, 0) for i in range(5)]
        r = ate(gt, gt)
        assert r["rmse_m"] == 0.0 and r["max_m"] == 0.0 and r["final_m"] == 0.0

    def test_constant_offset_is_removed_by_alignment(self):
        # 纯整体平移:对齐后 ATE 应为 0,不对齐时 = 偏移量
        gt = [_T(i * 1.0, 0) for i in range(6)]
        est = [_T(i * 1.0 + 3.0, 4.0) for i in range(6)]
        assert ate(est, gt, align=True)["rmse_m"] == 0.0
        raw = ate(est, gt, align=False)
        assert abs(raw["rmse_m"] - 5.0) < 1e-12  # 3-4-5

    def test_known_hand_computed_error_raw(self):
        # **不对齐**口径才可手算(对齐后残差依赖 SVD 解,不是手算量)。
        # 非共线 L 形路径(共线路径 Umeyama 退化,见 TestDegenerate)。
        # est 第 k 帧沿 x 偏 d_k = k:raw ATE = sqrt(mean(0² + 1² + 2²)) = sqrt(5/3)
        gt = [_T(0, 0), _T(2, 0), _T(2, 2)]
        est = [_T(0, 0), _T(3, 0), _T(4, 2)]
        r = ate(est, gt, align=False)
        # 容差 1e-4 而非 1e-9:`ate()` 的输出按 5 位小数四舍五入(落盘契约),
        # sqrt(5/3)=1.2909944487 → 1.29099,差值 4.4e-6。手算量只对到舍入位。
        assert abs(r["rmse_m"] - math.sqrt(5.0 / 3.0)) < 1e-4
        assert abs(r["max_m"] - 2.0) < 1e-4

    def test_scale_reported(self):
        # 非共线 L 形:共线时尺度不可辨识(见 TestDegenerate)
        gt = [_T(0, 0), _T(2, 0), _T(2, 2)]
        est = [_T(0, 0), _T(2.2, 0), _T(2.2, 2.2)]  # est = 1.1·gt(估计被拉伸)
        r = ate(est, gt, align=True, with_scale=True)
        # 方向语义:scale 是**施加到 est 上**的因子(est→gt,与 evo/Umeyama 一致)。
        # est 被拉伸 1.1× ⇒ 要乘 1/1.1 才回到 gt ⇒ scale = 0.909。偏离 1 即尺度系统偏差。
        assert abs(r["scale"] - 1.0 / 1.1) < 1e-6
        assert r["rmse_m"] < 1e-9  # 吸收尺度后残差归零

    def test_rotation_only_error_is_counted(self):
        gt = [_T(0, 0, yaw_deg=0), _T(1, 0, yaw_deg=0), _T(2, 0, yaw_deg=0)]
        est = [_T(0, 0, yaw_deg=15), _T(1, 0, yaw_deg=15), _T(2, 0, yaw_deg=15)]
        # 纯全局旋转:位置对齐会把它当旋转消掉 → ATE 0;RPE 旋转项同样为 0
        assert ate(est, gt, align=True)["rmse_m"] < 1e-9
        assert rpe(est, gt, delta=1)["rot_rmse_deg"] < 1e-6

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ate([_T(0, 0)], [_T(0, 0), _T(1, 0)])


class TestRPE:
    def test_zero_when_identical(self):
        gt = [_T(i * 1.0, 0, yaw_deg=i * 3.0) for i in range(6)]
        r = rpe(gt, gt, delta=1)
        assert r["trans_rmse_m"] == 0.0 and r["rot_rmse_deg"] == 0.0

    def test_hand_computed_translation(self):
        # 每帧相对位移 +1m,est 每帧 +1.2m → Δ=1 相对误差 0.2m;GT 路径 5m → per_m 0.2
        gt = [_T(i * 1.0, 0) for i in range(6)]
        est = [_T(i * 1.2, 0) for i in range(6)]
        r = rpe(est, gt, delta=1, align=False)
        assert abs(r["trans_rmse_m"] - 0.2) < 1e-9
        assert abs(r["gt_path_m"] - 5.0) < 1e-9
        assert abs(r["trans_per_m"] - 1.0 / 5.0 * 1.0) < 1e-9  # 1.0m 总误差 / 5m 路径

    def test_hand_computed_rotation(self):
        # 每帧相对 yaw +10°,est +12° → Δ=1 相对旋转误差 2°
        gt = [_T(0, 0, yaw_deg=i * 10.0) for i in range(4)]
        est = [_T(0, 0, yaw_deg=i * 12.0) for i in range(4)]
        r = rpe(est, gt, delta=1, align=False)
        assert abs(r["rot_rmse_deg"] - 2.0) < 1e-9

    def test_delta_changes_accumulation(self):
        # 恒定相对旋转误差 → Δ 越大累积越大(Δ=2 的误差 ≈ 2×)
        gt = [_T(0, 0, yaw_deg=i * 10.0) for i in range(6)]
        est = [_T(0, 0, yaw_deg=i * 11.0) for i in range(6)]
        r1 = rpe(est, gt, delta=1, align=False)
        r2 = rpe(est, gt, delta=2, align=False)
        assert abs(r2["rot_rmse_deg"] - 2.0 * r1["rot_rmse_deg"]) < 1e-6

    def test_delta_guards(self):
        gt = [_T(0, 0), _T(1, 0)]
        with pytest.raises(ValueError):
            rpe(gt, gt, delta=0)
        with pytest.raises(ValueError):
            rpe(gt, gt, delta=2)  # 帧数不足


class TestDegenerate:
    """共线轨迹的退化边界(**契约测试**:谁改了对齐实现,这里会拦下)。

    退化的是**绕轨迹轴的旋转**,不是尺度。构造性证明(见
    `test_collinear_axis_rotation_is_not_identifiable`):共线数据绕轨迹轴转 0°/37°/90°
    对齐残差**恒为 0** —— 位置数据不含绕轴信息,故最优 R 不由数据决定(协方差秩 1,
    非共线秩 2)。

    **后果(B2 口径纪律)**:纯直行序列(如 `collect_slam.py --speed 8`)的
    *对齐后姿态*与 *RPE 旋转项* 不可信 —— 全局 roll 没被数据钉住。要验证旋转精度
    必须走有转向/横向分量的轨迹(autopilot 或 `--speed 0`)。位置 ATE 不受此影响。

    尺度**不退化**:1D 尺度 = 沿线长度比,仍可辨识(曾误记为"共线尺度不可辨识",
    实测证伪)。
    """

    def test_collinear_scale_is_identifiable(self):
        # 1D 尺度 = 沿线长度比 → 共线数据也能算对(est=1.1·gt ⇒ s=1/1.1)
        gt = [_T(i * 1.0, 0) for i in range(6)]
        est = [_T(i * 1.1, 0) for i in range(6)]
        r = ate(est, gt, align=True, with_scale=True)
        assert abs(r["scale"] - 1.0 / 1.1) < 1e-6
        assert r["rmse_m"] < 1e-9  # 吸收尺度后残差归零

    def test_collinear_axis_rotation_is_not_identifiable(self):
        # 位置与 gt 完全相同 → 无论绕轨迹轴转多少,对齐残差都是 0。
        # 这不是实现缺陷,是数据不含该自由度;实现只能给出"某个"合法解。
        gt = np.array([_T(i * 1.0, 0)[:3, 3] for i in range(6)])
        est = gt.copy()
        mu_g, mu_e = gt.mean(0), est.mean(0)
        for ang in (0.0, 37.0, 90.0):
            c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])  # 绕轨迹轴(x)转
            t = mu_g - R @ mu_e
            res = np.linalg.norm((R @ est.T).T + t - gt, axis=1).max()
            assert res < 1e-9  # 残差恒 0 ⇒ R 不可辨识
        # 秩亏的直接证据:协方差秩 1(非共线为 2)
        assert np.linalg.matrix_rank(est - mu_e, tol=1e-9) == 1

    def test_non_collinear_axis_rotation_is_identifiable(self):
        # 对照:L 形同款构造,绕 x 转 37° 残差立刻 0.85 m → R 被数据钉住
        gt = np.array([_T(0, 0)[:3, 3], _T(2, 0)[:3, 3], _T(2, 2)[:3, 3]])
        est = gt.copy()
        mu_g, mu_e = gt.mean(0), est.mean(0)
        c, s = math.cos(math.radians(37.0)), math.sin(math.radians(37.0))
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        t = mu_g - R @ mu_e
        assert np.linalg.norm((R @ est.T).T + t - gt, axis=1).max() > 0.1
        assert np.linalg.matrix_rank(est - mu_e, tol=1e-9) == 2


class TestEvalTrajectory:
    def test_bundle_shape(self):
        gt = [_T(i * 1.0, 0, yaw_deg=i * 2.0) for i in range(30)]
        est = [_T(i * 1.01, 0.02, yaw_deg=i * 2.05) for i in range(30)]
        out = eval_trajectory(est, gt)
        assert set(out) == {"ate_aligned", "ate_raw", "rpe"}
        assert set(out["rpe"]) == {"d1", "d5", "d10"}
        assert out["ate_aligned"]["n"] == 30
        assert out["rpe"]["d10"]["n"] == 20

    def test_aligned_never_worse_than_raw(self):
        # 对齐是优化问题:对齐后 ATE 不可能比不对齐差(同数据同度量)
        rng = np.random.default_rng(0)
        gt = [_T(i * 0.8, math.sin(i * 0.3), yaw_deg=i * 4.0) for i in range(40)]
        est = [T @ _T(*rng.normal(0, 0.05, 3), yaw_deg=float(rng.normal(0, 0.5))) for T in gt]
        out = eval_trajectory(est, gt)
        assert out["ate_aligned"]["rmse_m"] <= out["ate_raw"]["rmse_m"] + 1e-12


class TestLidarPoseToEgo:
    """LiDAR 系位姿 → ego 系位姿的**帧一致性**(2026-09-19 修正的回归锚)。

    原式 `M·T·M @ inv(L)` 是 ATE 对齐口径(把杆臂当恒定偏移,Umeyama 会吸收掉),
    不是 ego 相对位姿。判据不看 ATE(对齐后两式同值 0.1877 m,看不出来),看**锚点**:
    帧 0 必须落在轨迹原点。
    """

    def test_identity_pose_maps_to_identity(self):
        """**核心回归**:`f(I) = I` —— 帧 0 是轨迹原点。

        原式在此给出平移 `[-1.2, 0, -1.65]`(恰是杆臂 `LIDAR_LEVER`),
        即把杆臂当成了恒定偏移。这条测试若红,说明左端的 `L` 又丢了。
        """
        got = lidar_pose_to_ego(np.eye(4))
        np.testing.assert_allclose(got, np.eye(4), atol=1e-12)

    def test_matches_hand_derived_four_factor_form(self):
        """与手写四因子式 `L·M·T·M·inv(L)` 逐元素一致(防实现与 docstring 漂移)。"""
        T = _T(3.0, -2.0, 0.5, yaw_deg=37.0)
        L = lever_matrix(LIDAR_LEVER)
        want = L @ M_FLIP @ T @ M_FLIP @ np.linalg.inv(L)
        np.testing.assert_allclose(lidar_pose_to_ego(T), want, atol=1e-12)

    def test_composes_over_relative_motion(self):
        """语义锚:`f(P_k) = inv(E_0)·E_k` —— 把链式位姿还原成"相对帧 0 的 ego 位姿"。

        取 `E_k = E_0·(平移 5 m + 转 30°)`,则 `f(P_k)` 必须等于那个相对变换本身。
        """
        E0 = _T(100.0, 50.0, 0.0, yaw_deg=20.0)
        dE = _T(5.0, 0.0, 0.0, yaw_deg=30.0)
        Ek = E0 @ dE
        L = lever_matrix(LIDAR_LEVER)
        P_k = M_FLIP @ np.linalg.inv(E0 @ L) @ (Ek @ L) @ M_FLIP
        np.testing.assert_allclose(lidar_pose_to_ego(P_k), dE, atol=1e-9)

    def test_pure_translation_passes_through_unchanged(self):
        """纯平移位姿:`L·M·T·M·inv(L)` 对平移是恒等(共轭把两个原点等量搬走)。

        这是"漏左端 L"最容易蒙混的输入 —— 两式在此只差一个常数 `-lever`。
        """
        T = _T(10.0, 0.0, 0.0)
        np.testing.assert_allclose(lidar_pose_to_ego(T)[:3, 3], [10.0, 0.0, 0.0], atol=1e-12)

    def test_rotation_makes_the_left_lever_visible(self):
        """**带旋转时左端 `L` 不可省**:平移 + 90° 偏航,两式差 `‖lever‖ = 2.0402 m`。

        机理:纯平移时 `L` 与 `inv(L)` 的平移项相消;一旦有旋转,`L` 的平移项被旋转
        块带动,左端那个 `L` 就把"LiDAR 原点 → ego 原点"的偏移补了回来。原式(`M·T·M·inv(L)`)
        在此给出 `[10, 1.2, -1.65]`,四因子式给出 `[11.2, 1.2, 0]` —— 后者的 z = 0 才是
        "ego 在水平面上"的应有结果。
        """
        T = _T(10.0, 0.0, 0.0, yaw_deg=90.0)
        got = lidar_pose_to_ego(T)
        buggy = M_FLIP @ T @ M_FLIP @ np.linalg.inv(lever_matrix(LIDAR_LEVER))
        np.testing.assert_allclose(got[:3, 3], [11.2, 1.2, 0.0], atol=1e-9)
        assert np.linalg.norm(got[:3, 3] - buggy[:3, 3]) == pytest.approx(
            float(np.linalg.norm(LIDAR_LEVER)), abs=1e-9
        )

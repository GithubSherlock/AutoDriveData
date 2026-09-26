"""`calib_probe` 手算锚点单测(纯 numpy,不碰 CARLA)。

覆盖四类**可独立手算**的判据:

1. 射线-平面求交的闭式解(含"沿光轴 vs 沿射线"的差别 —— 45° 下视射线打地面,
   射线长 √2 而光轴深度 1)
2. 投影 / 反投影 / 射线三者互逆
3. 主点估计:注入已知 δ 能否复原、**梯度无变化时是否如实报不可辨识**
4. 轴目标物读数:合成对称掩膜能否精确复原注入的 cx
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autodrivedata.calib import calib_probe as cp
from autodrivedata.calib.core import CameraIntrinsics
from autodrivedata.calib.depth_codec import CONVENTION_CENTER, CONVENTION_CORNER

W, H, FOV = 1242, 375, 90.0
K = CameraIntrinsics(width=W, height=H, fov_h_deg=FOV)
CAM_LOC = (0.0, 0.0, 0.0)
CAM_ROT_FWD = (0.0, 0.0, 0.0)  # CARLA 相机局部系:x 前 / y 右 / z 上
# 相机系 → 世界系(KITTI 相机系 x 右 / y 下 / z 前 ⇒ 光轴落在世界 +x)
ROT = cp.cam_to_world_rot(CAM_ROT_FWD)


def _cam_pts(uv: np.ndarray, z: float | np.ndarray) -> np.ndarray:
    """图像坐标 + 光轴深度 z → **相机系**点(针孔逆投影,`cam_rays` 的单位向量不带 z)。"""
    u = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    zz = np.broadcast_to(np.asarray(z, dtype=np.float64), (u.shape[0],))
    return np.stack([(u[:, 0] - K.cx) * zz / K.fx, (u[:, 1] - K.cy) * zz / K.fy, zz], axis=1)


def _world(uv: np.ndarray, z: float | np.ndarray) -> np.ndarray:
    """图像坐标 + 光轴深度 → **世界系**点(经相机位姿,不是把相机系当世界系用)。"""
    return _cam_pts(uv, z) @ ROT.T


def _samples(e: np.ndarray, g: np.ndarray, z0: float = 20.0) -> cp.DepthSamples:
    """由残差 e 与梯度 g 反造一份 DepthSamples(uv/z 只为占位,不参与估计)。"""
    e = np.asarray(e, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    n = e.size
    return cp.DepthSamples(
        uv=np.zeros((n, 2)),
        z_lidar=np.full(n, z0),
        z_render=z0 + e,
        grad=np.stack([g, np.zeros(n)], axis=1),
    )


def _strip_mask(centre_u: float, width: float, rows: int = 60) -> np.ndarray:
    """按**连续坐标**定义的左右对称竖直条带:像素 i 的中心 i+0.5 落在 [c−w/2, c+w/2] 内。

    故意用连续坐标定义,好让"索引口径中点 = c − 0.5"这条约定关系被测试钉住。
    """
    mask = np.zeros((rows, W), dtype=bool)
    centres = np.arange(W, dtype=np.float64) + 0.5
    mask[:, :] = (centres[None, :] > centre_u - width / 2.0) & (centres[None, :] < centre_u + width / 2.0)
    return mask


class TestPlaneRay:
    def test_head_on_hit(self):
        # 平面 x = 10(法向 +x),相机在原点朝 +x,沿光轴射线 → 深度 10
        pl = cp.Plane(np.array([1.0, 0.0, 0.0]), 10.0)
        assert pl.depth_along_ray(np.zeros(3), np.array([1.0, 0.0, 0.0]), (1.0, 0.0, 0.0)) == pytest.approx(
            10.0
        )

    def test_tilted_ray_ray_length_differs_from_axis_depth(self):
        """45° 下视射线打地面:射线长 √2 m,但**光轴深度**是 1 m。

        这正是"深度语义"混淆的几何来源 —— 用射线长度当深度会随离轴角放大
        (`sec θ`),图像角落可达 +44%。
        """
        pl = cp.Plane(np.array([0.0, 0.0, 1.0]), -1.0)  # 地面 z = −1
        d = np.array([1.0, 0.0, -1.0]) / math.sqrt(2.0)
        axis = np.array([1.0, 0.0, 0.0])
        assert pl.depth_along_ray(np.zeros(3), d, axis) == pytest.approx(1.0)
        # 沿射线距离(axis 取射线自身)则是 √2
        assert pl.depth_along_ray(np.zeros(3), d, d) == pytest.approx(math.sqrt(2.0))

    def test_parallel_ray_is_nan(self):
        pl = cp.Plane(np.array([0.0, 0.0, 1.0]), -1.0)
        assert math.isnan(pl.depth_along_ray(np.zeros(3), np.array([1.0, 0.0, 0.0]), (1.0, 0.0, 0.0)))

    def test_hit_behind_origin_is_nan(self):
        pl = cp.Plane(np.array([1.0, 0.0, 0.0]), -5.0)  # 平面在 x = −5,射线朝 +x
        assert math.isnan(pl.depth_along_ray(np.zeros(3), np.array([1.0, 0.0, 0.0]), (1.0, 0.0, 0.0)))

    def test_batched_matches_scalar(self):
        pl = cp.Plane(np.array([0.3, -0.2, 0.93]) / np.linalg.norm([0.3, -0.2, 0.93]), 4.0)
        rng = np.random.default_rng(0)
        d = rng.normal(size=(50, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        axis = (0.0, 0.0, 1.0)
        want = np.array([pl.depth_along_ray(np.zeros(3), x, axis) for x in d])
        got = pl.depth_along_rays(np.zeros(3), d, axis)
        assert np.allclose(want, got, equal_nan=True)

    def test_world_z_axis_is_not_a_valid_depth_axis(self):
        """反例钉死:相机俯视地面时用世界 z 当深度轴会给出**负深度**(静默错值)。"""
        pl = cp.Plane(np.array([0.0, 0.0, 1.0]), -1.0)
        d = np.array([1.0, 0.0, -1.0]) / math.sqrt(2.0)
        assert pl.depth_along_ray(np.zeros(3), d, (0.0, 0.0, 1.0)) < 0.0


class TestFitPlane:
    def test_recovers_known_plane(self):
        rng = np.random.default_rng(1)
        pts = rng.normal(size=(200, 2))
        z = 0.05 * pts[:, 0] - 0.02 * pts[:, 1] + 3.0
        fit = cp.fit_plane(np.column_stack([pts, z]))
        assert fit is not None
        pl, rms = fit
        assert rms < 1e-9
        # 法向与 (0.05, −0.02, −1) 共线(符号不定)
        ref = np.array([0.05, -0.02, -1.0])
        ref /= np.linalg.norm(ref)
        assert abs(abs(float(pl.normal @ ref)) - 1.0) < 1e-9

    def test_too_few_points_is_none(self):
        assert cp.fit_plane(np.zeros((cp.MIN_PLANE_PTS - 1, 3))) is None

    def test_collinear_points_is_none(self):
        t = np.linspace(0, 1, 40)[:, None]
        assert cp.fit_plane(np.hstack([t, np.zeros_like(t), np.zeros_like(t)])) is None

    def test_noisy_plane_residual_is_reported(self):
        rng = np.random.default_rng(2)
        pts = rng.uniform(-1, 1, size=(400, 2))
        z = 0.0 + rng.normal(scale=0.01, size=400)
        fit = cp.fit_plane(np.column_stack([pts, z]))
        assert fit is not None and 0.008 < fit[1] < 0.012
        assert cp.plane_is_usable(fit[1])
        assert not cp.plane_is_usable(0.5)

    def test_local_planes_on_curved_surface(self):
        """逐点局部平面:球面上每点的法向都应指向球心(半径远大于邻域半径)。"""
        rng = np.random.default_rng(3)
        u = rng.normal(size=(6000, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        pts = u * 20.0
        normals, rms = cp.fit_local_planes(pts, radius=3.0)
        ok = np.isfinite(rms)
        assert ok.sum() > 5000
        cos = np.einsum("ij,ij->i", normals[ok], u[ok])
        assert np.abs(np.abs(cos) - 1.0).max() < 0.01


class TestProjectionInverse:
    def test_ray_and_project_are_inverse(self):
        rng = np.random.default_rng(4)
        uv = np.column_stack([rng.uniform(0, W, 300), rng.uniform(0, H, 300)])
        z = rng.uniform(5, 60, 300)
        pts_world = _world(uv, z)
        back, z_axis = cp.project_world(pts_world, CAM_LOC, CAM_ROT_FWD, K)
        assert np.allclose(back, uv, atol=1e-9)
        assert np.allclose(z_axis, z, rtol=1e-9)

    def test_camera_frame_is_not_world_frame(self):
        """钉死位姿的存在:相机光轴在世界系是 **+x**(KITTI 相机系 z 前 ⇒ 世界 x),
        把相机系射线直接当世界系点用会错位(曾在本文件里犯过)。"""
        assert np.allclose(ROT[:, 2], [1.0, 0.0, 0.0], atol=1e-12)
        rays = cp.cam_rays(K, np.array([[K.cx, K.cy]]))
        assert np.allclose(rays @ ROT.T, [1.0, 0.0, 0.0], atol=1e-12)

    def test_principal_point_maps_to_optical_axis(self):
        rays = cp.cam_rays(K, np.array([[K.cx, K.cy]]))
        assert np.allclose(rays[0], [0.0, 0.0, 1.0], atol=1e-12)

    def test_ray_is_unit_length(self):
        uv = np.array([[0.0, 0.0], [W - 1.0, H - 1.0], [K.cx, K.cy]])
        assert np.allclose(np.linalg.norm(cp.cam_rays(K, uv), axis=1), 1.0)

    def test_convention_does_not_enter_the_ray(self):
        """`cam_rays` 吃的是**图像坐标**,与像素约定无关(约定只在采样时出现)。"""
        uv = np.array([[300.0, 200.0]])
        assert np.array_equal(cp.cam_rays(K, uv), cp.cam_rays(K, uv))


class TestBackproject:
    def test_plane_depth_roundtrip(self):
        """合成一张"正对相机的平面"深度图,反投影后所有点应落在该平面上。"""
        uv = np.array([[K.cx, K.cy], [100.0, 50.0], [1100.0, 300.0]])
        z = 30.0
        depth = np.full((H, W), z)
        pts = cp.backproject_depth(uv, depth, K, CONVENTION_CENTER)
        back, z_axis = cp.project_world(pts, CAM_LOC, CAM_ROT_FWD, K)
        assert np.allclose(back, uv, atol=1e-6)
        assert np.allclose(z_axis, z, rtol=1e-9)

    def test_convention_shift_moves_sampling_by_half_pixel(self):
        """约定确实改变采样位置 —— 用一张阶梯深度图量出这 0.5 px。

        深度图索引 10 起为 5 m,即**图像坐标** 10.5 起为 5 m。
        `center` 约定把图像坐标 10.0 映射到索引 9.5 ⇒ 跨在阶梯上,双线性得半值 2.5;
        `corner` 约定映射到索引 10.0 ⇒ 阶梯**右侧**整值 5。
        (`backproject_depth` 出口是世界系,故用 `project_world` 读回光轴深度。)
        """
        depth = np.zeros((H, W))
        depth[:, 10:] = 5.0

        def z_of(u: float, conv: str) -> float:
            pts = cp.backproject_depth(np.array([[u, 100.0]]), depth, K, conv)
            return float(cp.project_world(pts, CAM_LOC, CAM_ROT_FWD, K)[1][0])

        assert z_of(10.0, CONVENTION_CENTER) == pytest.approx(2.5)
        assert z_of(10.0, CONVENTION_CORNER) == pytest.approx(5.0)
        # 图像坐标 10.5 在两种约定下都落在阶梯右侧(索引 10.0 / 10.5)
        assert z_of(10.5, CONVENTION_CENTER) == pytest.approx(5.0)
        assert z_of(10.5, CONVENTION_CORNER) == pytest.approx(5.0)


class TestLocalRange:
    @pytest.mark.parametrize("radius", [1, 3, 7])
    def test_matches_naive_reference(self, radius):
        rng = np.random.default_rng(5)
        img = rng.random((23, 31))
        got = cp.local_range(img, radius)
        r = radius
        pad = np.pad(img, r, mode="edge")
        want = np.empty_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                win = pad[i : i + 2 * r + 1, j : j + 2 * r + 1]
                want[i, j] = win.max() - win.min()
        assert np.allclose(got, want)

    def test_zero_radius_is_zero(self):
        img = np.random.default_rng(6).random((5, 5))
        assert np.allclose(cp.local_range(img, 0), 0.0)


class TestAxisEstimation:
    def test_recovers_injected_delta(self):
        """注入 δ 与共模偏置 b,回归应同时复原两者。"""
        rng = np.random.default_rng(7)
        g = rng.uniform(-0.05, 0.05, 800)
        delta_true, b_true = 0.5, 0.03
        e = g * delta_true + b_true + rng.normal(scale=0.01, size=800)
        fit = cp.estimate_axis_delta(_samples(e, g), axis=0)
        assert fit.identifiable
        assert fit.delta_px == pytest.approx(delta_true, abs=0.02)
        assert fit.intercept_m == pytest.approx(b_true, abs=0.01)

    def test_constant_gradient_is_unidentifiable(self):
        """梯度无变化 ⇒ δ 与共模偏置完全共线 ⇒ **必须报不可辨识**,不给数字。"""
        rng = np.random.default_rng(8)
        g = np.full(500, 0.02)
        e = g * 0.5 + 0.03 + rng.normal(scale=0.01, size=500)
        fit = cp.estimate_axis_delta(_samples(e, g), axis=0)
        assert not fit.identifiable
        assert math.isinf(fit.sigma_delta_px)

    def test_intercept_absorbs_common_mode_bias(self):
        """b 不被 δ 吸收:纯共模偏置(δ=0)时 δ 估计应 ≈ 0,而 b 复原该偏置。

        σ_δ = σ_e/√(Σ(g−ḡ)²) ≈ 0.0055(n=4000,σ_e=0.01)⇒ 容差取 ~5σ。
        """
        rng = np.random.default_rng(9)
        n = 4000
        g = rng.uniform(-0.05, 0.05, n)
        e = np.full(n, 0.25) + rng.normal(scale=0.01, size=n)
        fit = cp.estimate_axis_delta(_samples(e, g), axis=0)
        assert fit.identifiable
        assert abs(fit.delta_px) < 0.03
        assert fit.intercept_m == pytest.approx(0.25, abs=0.005)

    def test_sigma_shrinks_with_more_samples(self):
        rng = np.random.default_rng(10)

        def sigma(n: int) -> float:
            g = rng.uniform(-0.05, 0.05, n)
            e = g * 0.3 + rng.normal(scale=0.01, size=n)
            return cp.estimate_axis_delta(_samples(e, g), axis=0).sigma_delta_px

        assert sigma(2000) < sigma(200)

    def test_v_axis_is_independent(self):
        rng = np.random.default_rng(11)
        n = 4000
        g = rng.uniform(-0.05, 0.05, (n, 2))
        e = g @ np.array([0.4, -0.2]) + 0.01 + rng.normal(scale=0.01, size=n)
        s = cp.DepthSamples(np.zeros((n, 2)), np.zeros(n), e, g)
        fu, fv = cp.estimate_delta_uv(s)
        assert fu.delta_px == pytest.approx(0.4, abs=0.03)
        assert fv.delta_px == pytest.approx(-0.2, abs=0.03)
        assert (fu.axis, fv.axis) == ("u", "v")

    def test_too_few_samples(self):
        fit = cp.estimate_axis_delta(_samples(np.array([1.0, 2.0]), np.array([0.1, 0.2])), axis=0)
        assert not fit.identifiable and fit.n == 2

    def test_sweep_profile_minimum_matches_regression(self):
        rng = np.random.default_rng(12)
        g = rng.uniform(-0.05, 0.05, 1000)
        e = g * 0.4 + 0.02 + rng.normal(scale=0.005, size=1000)
        s = _samples(e, g)
        deltas = np.linspace(-1.0, 2.0, 301)
        prof = cp.sweep_profile(s, 0, deltas)
        assert deltas[int(np.argmin(prof))] == pytest.approx(cp.estimate_axis_delta(s, 0).delta_px, abs=0.05)

    def test_mad_sigma_is_robust_to_outliers(self):
        rng = np.random.default_rng(13)
        x = rng.normal(scale=1.0, size=2000)
        x[:50] += 500.0
        assert 0.8 < cp.mad_sigma(x) < 1.2


class TestDepthSemantics:
    def test_axis_model_is_flat_and_range_model_is_sec(self):
        theta = np.radians([0.0, 30.0, 46.2])
        z_model, r_model = cp.depth_model_ratio(theta)
        assert np.allclose(z_model, 1.0)
        assert r_model[0] == pytest.approx(1.0)
        assert r_model[2] == pytest.approx(1.0 / math.cos(math.radians(46.2)), rel=1e-9)
        assert r_model[2] - 1.0 > 0.44  # 角落处混淆量级 ~44%

    def test_max_off_axis_angle_anchor(self):
        """1242×375 / fov 90 的角落离轴角 ≈ 46.27°。"""
        assert math.degrees(cp.max_off_axis_angle(K)) == pytest.approx(46.27, abs=0.05)

    def test_off_axis_angle_zero_on_axis(self):
        assert cp.off_axis_angle(np.array([[K.cx, K.cy]]), K)[0] == pytest.approx(0.0, abs=1e-12)


class TestRadialProfile:
    def test_ideal_pinhole_has_zero_slope(self):
        """理想针孔 + 正确 K ⇒ 残差不随半径变化(斜率 ≈ 0)。

        注意:理想针孔下残差的中位**绝对值**是 σ 的 0.6745 倍而非 0 —— 斜率才是判据。
        """
        rng = np.random.default_rng(14)
        n = 3000
        uv = np.column_stack([rng.uniform(0, W, n), rng.uniform(0, H, n)])
        e = rng.normal(scale=0.01, size=n)
        s = cp.DepthSamples(uv, np.zeros(n), e, np.zeros((n, 2)))
        _, _, counts, slope = cp.radial_error_profile(s, K)
        assert counts.sum() >= int(n * 0.98)  # 末尾一箱右开 ⇒ 最高的 ~1% 被舍
        assert abs(slope) < 1e-5

    def test_radial_bias_shows_up_as_slope(self):
        rng = np.random.default_rng(15)
        n = 4000
        uv = np.column_stack([rng.uniform(0, W, n), rng.uniform(0, H, n)])
        r = np.hypot(uv[:, 0] - K.cx, uv[:, 1] - K.cy)
        e = 1e-4 * r + rng.normal(scale=0.005, size=n)
        s = cp.DepthSamples(uv, np.zeros(n), e, np.zeros((n, 2)))
        _, _, _, slope = cp.radial_error_profile(s, K)
        assert slope == pytest.approx(1e-4, rel=0.1)

    def test_empty_samples(self):
        empty = cp.DepthSamples(np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros((0, 2)))
        _, _, counts, slope = cp.radial_error_profile(empty, K)
        assert counts.size == 0 and math.isnan(slope)


class TestAxisTarget:
    def test_midpoints_recover_centre(self):
        """条带按**图像坐标** 621.0 对称 ⇒ 索引口径的中点列 = 620.5(差恰好一个 shift)。"""
        _, mids = cp.mask_row_midpoints(_strip_mask(621.0, width=20.0))
        assert mids.size > 30
        assert np.allclose(mids, 620.5, atol=1e-9)

    def test_midline_deviation_zero_for_symmetric(self):
        _, mids = cp.mask_row_midpoints(_strip_mask(621.0, width=20.0))
        assert cp.midline_deviation(mids) == pytest.approx(0.0, abs=1e-9)

    def test_midline_deviation_flags_one_sided_tilt(self):
        """单侧缺失的行 → 中点整体偏 ⇒ 判据必须非 0(自证目标物不达标)。

        条带列 611..630(中点 620.5);把**前 20 行**(共 60 行)的左边界从 611 削到 618
        ⇒ 这些行中点变 624。中位数仍落在未变的那 40 行上(620.5),故偏离 = 3.5 px。
        若改动超过一半行,中位数自己也会被拖走(那是稳健统计的正常行为,不是缺陷)。
        """
        mask = _strip_mask(621.0, width=20.0)
        mask[:20, :618] = False
        _, mids = cp.mask_row_midpoints(mask, trim=0.0)
        assert cp.midline_deviation(mids) == pytest.approx(3.5, abs=1e-9)

    def test_mirror_asymmetry_zero_for_symmetric(self):
        assert cp.mirror_asymmetry(_strip_mask(621.0, width=20.0), 620.5) == pytest.approx(0.0, abs=1e-9)

    def test_mirror_asymmetry_flags_offset_axis(self):
        # 条带半宽 10 px;镜像面偏 4 px ⇒ 左右半宽差 8 px
        assert cp.mirror_asymmetry(_strip_mask(621.0, width=20.0), 616.5) == pytest.approx(8.0, abs=1e-9)

    def test_mirror_asymmetry_is_about_a_fixed_axis_not_the_midline(self):
        """判据必须**绕固定轴**量 min/max,不能由逐行中点推。

        构造一条绕 620.5 严格对称、但**逐行宽度不同**的掩膜(上半 611..630、下半 617..624):
        逐行中点全是 620.5。若实现写成"中点 ± 半宽"那种自证式子,宽度变化永远看不出来;
        绕固定轴量则确实为 0(它真的对称),而绕错轴(616.5)必须报出非 0 —— 两条一起给。
        """
        mask = _strip_mask(621.0, width=20.0)
        mask[30:, :617] = False
        mask[30:, 625:] = False  # 下半段收窄成 617..624(中点仍是 620.5)
        _, mids = cp.mask_row_midpoints(mask, trim=0.0)
        assert np.allclose(mids, 620.5)
        assert cp.mirror_asymmetry(mask, 620.5) == pytest.approx(0.0, abs=1e-9)
        # 上半半宽 (611..630 绕 616.5) = 5.5 vs 13.5 ⇒ 8.0
        assert cp.mirror_asymmetry(mask, 616.5) == pytest.approx(8.0, abs=1e-9)

    @pytest.mark.parametrize("shift", [0.0, 0.5])
    def test_prop_regression_recovers_cx(self, shift):
        """注入 cx_img = 621.0,回归截距 + shift 应精确复原它(与约定无关)。"""
        cx_img, f_over_z = 621.0, 621.0 / 20.0
        offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        centres = f_over_z * offsets + (cx_img - shift)
        cx_est, slope, resid = cp.prop_axis_regression(offsets, centres, shift)
        assert cx_est == pytest.approx(cx_img, abs=1e-9)
        assert slope == pytest.approx(f_over_z, rel=1e-9)
        assert np.abs(resid).max() < 1e-9

    def test_prop_regression_requires_two_groups(self):
        with pytest.raises(ValueError):
            cp.prop_axis_regression(np.array([0.0]), np.array([621.0]), 0.5)

    def test_prop_regression_residual_flags_moved_prop(self):
        """目标物被物理引擎挪动 ⇒ 残差非 0(该法不可靠的自证)。"""
        offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        centres = 621.0 / 20.0 * offsets + 620.5
        centres[3] += 0.8  # 该组被挪了
        _, _, resid = cp.prop_axis_regression(offsets, centres, 0.5)
        assert np.abs(resid).max() > 0.3


class TestCollectSamples:
    def test_straight_forward_plane_has_zero_residual(self):
        """正对相机的平面:渲染深度图 = 常数,射线-平面求交应精确复原它。"""
        depth = np.full((H, W), 25.0)
        rng = np.random.default_rng(16)
        uv = np.column_stack([rng.uniform(0, W, 400), rng.uniform(0, H, 400)])
        pts = _world(uv, 25.0)  # 平面 = 世界 x = 25,法向 +x
        s = cp.collect_samples(
            pts, np.tile([1.0, 0.0, 0.0], (400, 1)), np.full(400, 25.0), CAM_LOC, CAM_ROT_FWD, K, depth
        )
        # 少数点投影落在图外(边界像素的连续坐标可到 w−0.5 ⇒ 反投影回来刚好出界)
        assert len(s) > 390
        assert np.abs(s.residual).max() < 1e-9
        assert np.abs(s.grad).max() < 1e-12  # 常数深度 ⇒ 零梯度(该场景对主点零信息)

    def test_behind_camera_points_are_dropped(self):
        depth = np.full((H, W), 25.0)
        pts = np.array([[-5.0, 0.0, 0.0]])  # 相机身后(光轴 = 世界 +x)
        s = cp.collect_samples(
            pts, np.array([[-1.0, 0.0, 0.0]]), np.array([5.0]), CAM_LOC, CAM_ROT_FWD, K, depth
        )
        assert len(s) == 0

    def test_out_of_frame_points_are_dropped(self):
        depth = np.full((H, W), 25.0)
        # 极侧向(世界 y = 50)→ 投影到图外
        pts = np.array([[1.0, 50.0, 0.0]])
        s = cp.collect_samples(
            pts, np.array([[1.0, 0.0, 0.0]]), np.array([1.0]), CAM_LOC, CAM_ROT_FWD, K, depth
        )
        assert len(s) == 0

    def test_large_residual_is_dropped(self):
        depth = np.full((H, W), 25.0)
        pts = np.array([[25.0, 0.0, 0.0]])
        s_ok = cp.collect_samples(
            pts, np.array([[1.0, 0.0, 0.0]]), np.array([25.0]), CAM_LOC, CAM_ROT_FWD, K, depth
        )
        assert len(s_ok) == 1
        s_bad = cp.collect_samples(
            pts, np.array([[1.0, 0.0, 0.0]]), np.array([25.0]), CAM_LOC, CAM_ROT_FWD, K, np.full((H, W), 30.0)
        )
        assert len(s_bad) == 0  # 5 m 残差 > MAX_RESIDUAL_M

    def test_occlusion_filter_drops_discontinuity(self):
        """邻域深度极差超阈 ⇒ 该点被剔(视差/遮挡边缘)。"""
        depth = np.full((H, W), 25.0)
        depth[:, 700:] = 5.0  # 一条深度断裂
        uv = np.array([[699.0, 100.0]])  # 紧贴断裂(图像坐标 700.5 起为 5 m)
        pts = _world(uv, 25.0)
        pl_n = np.array([[1.0, 0.0, 0.0]])
        pl_o = np.array([25.0])
        s_no = cp.collect_samples(pts, pl_n, pl_o, CAM_LOC, CAM_ROT_FWD, K, depth)
        assert len(s_no) == 1
        s_oc = cp.collect_samples(pts, pl_n, pl_o, CAM_LOC, CAM_ROT_FWD, K, depth, occlusion_radius_px=8)
        assert len(s_oc) == 0


class TestVisibilityIsOneSided:
    """可见性判据是**单侧**的:渲染比预测更近超过容差才判"被挡住"。

    为什么钉这个(实测踩坑,2026-09-22):旧口径用**对称**的邻域深度极差当遮挡代理,
    在真实街景下窗口半径一大就把合法样本成片误杀 —— r=16 px 时六相机样本从
    89/80/85/40/79 塌到 4/4/8/0/5(半径由 `f·b/参考距` 推出,实测 16.1–36.4 px)。
    物理上"看不见"只有**更近的面挡住**这一种,单侧判据在 r=0 就等价成立。
    """

    def _one(self, z_pred: float, z_rend: float, **kw) -> int:
        depth = np.full((H, W), z_rend)
        uv = np.array([[621.0, 187.0]])
        pts = _world(uv, z_pred)
        n = np.array([[1.0, 0.0, 0.0]])
        return len(cp.collect_samples(pts, n, np.array([z_pred]), CAM_LOC, CAM_ROT_FWD, K, depth, **kw))

    def test_nearer_render_blocks(self):
        """渲染比预测近 2 m(> 容差)⇒ 相机看不见该点,剔除。"""
        assert self._one(25.0, 23.0) == 0

    def test_nearer_render_within_tolerance_kept(self):
        """近一点点(0.1 m < OCCLUSION_TOL_M)⇒ 是深度噪声不是遮挡,保留。"""
        assert self._one(25.0, 24.9) == 1

    def test_farther_render_kept(self):
        """渲染比预测**远**(该点在前、后面才是背景)⇒ 绝不是遮挡,保留。

        幅度取 0.8 m(< `MAX_RESIDUAL_M`,故只考验可见性判据本身;取 15 m 会被
        残差闸拦下,那测的就不是可见性了)。
        """
        assert self._one(25.0, 25.8) == 1

    def test_nearer_neighbour_in_window_does_not_kill(self):
        """窗口里存在更近的面、但**采样像素本身**读到的就是预测值 ⇒ 不该被剔。

        这正是旧对称口径的错杀:窗口极差 20 m > 0.15×25 m,而该点其实清晰可见。
        """
        depth = np.full((H, W), 25.0)
        depth[:, 700:] = 5.0  # 5 m 外的邻居(不在采样点上)
        uv = np.array([[600.0, 100.0]])
        pts = _world(uv, 25.0)
        n = np.array([[1.0, 0.0, 0.0]])
        s = cp.collect_samples(pts, n, np.array([25.0]), CAM_LOC, CAM_ROT_FWD, K, depth)
        assert len(s) == 1
        # 显式开窗口极差口径则被剔 —— 保留这条对照,证明"关掉它"是刻意的
        s_win = cp.collect_samples(
            pts, n, np.array([25.0]), CAM_LOC, CAM_ROT_FWD, K, depth, occlusion_radius_px=120
        )
        assert len(s_win) == 0

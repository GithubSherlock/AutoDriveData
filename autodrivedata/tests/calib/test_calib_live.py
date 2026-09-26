"""`calib_live`(实时标定槽纯值件)单测:手算锚点 + **判据不许报假数字**。

三类锚:

1. **几何**:传感器系点 → 世界系不翻 y;平面截距口径与 `calib_probe.collect_samples` 对齐
   (正对相机的平面残差恒 0)。
2. **像素约定**:着色用 `index = u`(corner),不减 0.5 —— 注入 uv=1.0 必须落在**索引 1**。
3. **统计口径**(本文件的核心):样本数不足时 `median_abs is None`,**不许**拿 3 个样本
   算出一个"漂亮数字";`pooled_median` 是**等权**中位数(某一路样本多 10× 不得拖走总体)。
   这两条对应 CAM_BACK 的自遮挡平台边界(见 `calib_live` 模块头注)。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.calib import calib_live as cl
from autodrivedata.calib import calib_probe as cp
from autodrivedata.calib.core import CameraIntrinsics

W, H, FOV = 640, 360, 90.0
K = CameraIntrinsics(width=W, height=H, fov_h_deg=FOV)
CAM_LOC = (0.0, 0.0, 0.0)
CAM_ROT_FWD = (0.0, 0.0, 0.0)  # CARLA 相机局部系:x 前 / y 右 / z 上


def _samples(n: int, e: float) -> cp.DepthSamples:
    """n 个残差恒为 e 的样本(uv/z 只为占位)。"""
    return cp.DepthSamples(
        uv=np.zeros((n, 2)), z_lidar=np.full(n, 20.0), z_render=np.full(n, 20.0 + e), grad=np.zeros((n, 2))
    )


def _world(uv: np.ndarray, z: float) -> np.ndarray:
    """图像坐标 + 光轴深度 → 世界系点(KITTI 相机系 z 前 ⇒ 光轴落在世界 +x)。"""
    u = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    cam = np.stack([(u[:, 0] - K.cx) * z / K.fx, (u[:, 1] - K.cy) * z / K.fy, np.full(u.shape[0], z)], axis=1)
    return cam @ cp.cam_to_world_rot(CAM_ROT_FWD).T


class TestWorldPoints:
    def test_identity_pose_is_passthrough(self):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = cl.world_points_from_lidar(pts, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        assert np.allclose(out, pts)

    def test_translation_only(self):
        pts = np.array([[1.0, 2.0, 3.0]])
        out = cl.world_points_from_lidar(pts, (10.0, -1.0, 0.5), (0.0, 0.0, 0.0))
        assert np.allclose(out, [[11.0, 1.0, 3.5]])

    def test_y_is_not_flipped(self):
        """y 翻转是 KITTI 落盘口径,与几何无关 —— 在这里翻 y 会让所有点落到镜像位置。"""
        out = cl.world_points_from_lidar(np.array([[0.0, 5.0, 0.0]]), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        assert out[0, 1] == pytest.approx(5.0)

    def test_extra_columns_are_ignored(self):
        """语义 LiDAR 每点 6 列(含标签)、ray_cast 每点 4 列 —— 都只取前 3 列。"""
        pts = np.array([[1.0, 2.0, 3.0, 0.9, 7.0, 4.0]])
        assert np.allclose(
            cl.world_points_from_lidar(pts, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), [[1.0, 2.0, 3.0]]
        )


class TestLivePlanes:
    def _ground(self, extent: float = 5.0, step: float = 0.25, z: float = 0.0) -> np.ndarray:
        g = np.arange(-extent, extent + 1e-9, step)
        xx, yy = np.meshgrid(g, g)
        return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])

    def test_recovers_ground_plane_normal(self):
        pts = self._ground()
        p, n, off = cl.live_planes(pts, np.zeros(3), np.random.default_rng(0), max_samples=600)
        # 1 m 体素把 11×11 的格点降成 121 个,再按半径 2 m 取邻域:边缘点邻域不足 12 个被剔
        assert p.shape[0] > 50
        # 水平地面的法向应沿 ±z;截距 = n·p ⇒ 与 z 同号且量值 ≈ 0
        assert np.abs(np.abs(n[:, 2]) - 1.0).max() < 1e-6
        assert np.abs(off).max() < 1e-6

    def test_intercept_matches_offset_plane(self):
        """抬到 z = 2 的地面:截距必须是 ±2(口径 = n·p,直接进 collect_samples)。"""
        pts = self._ground(z=2.0)
        _, n, off = cl.live_planes(pts, np.zeros(3), np.random.default_rng(0), max_samples=600)
        assert np.abs(off).max() == pytest.approx(2.0, abs=1e-6)

    def test_far_points_are_dropped(self):
        """距离闸是相对 **LiDAR 自身**:30 m 外的点不得进拟合(否则邻域全空、样本塌成 0)。"""
        near = self._ground(extent=3.0)
        far = self._ground(extent=3.0) + np.array([0.0, 0.0, 0.0])
        far[:, 0] += 30.0  # 平移到 30 m 外
        pts = np.vstack([near, far])
        p, _, _ = cl.live_planes(pts, np.zeros(3), np.random.default_rng(0), max_dist=25.0, max_samples=600)
        assert p.shape[0] > 0
        assert p[:, 0].max() < 25.0  # 一个远点都不在

    def test_empty_input_is_empty(self):
        p, n, off = cl.live_planes(np.zeros((0, 3)), np.zeros(3), np.random.default_rng(0))
        assert p.shape == (0, 3) and n.shape == (0, 3) and off.shape == (0,)

    def test_sample_budget_caps_input(self):
        """上限是 O(N²) 的闸:给 5000 点、上限 400 ⇒ 拟合输入不超过 400。"""
        pts = self._ground(extent=12.0, step=0.3)
        assert pts.shape[0] > 4000
        p, _, _ = cl.live_planes(pts, np.zeros(3), np.random.default_rng(0), max_samples=400)
        assert p.shape[0] <= 400


class TestResidualColors:
    @pytest.mark.parametrize(
        ("e", "want"),
        [
            (0.0, cl.GOOD_COLOR),
            (0.049, cl.GOOD_COLOR),
            (0.05, cl.WARN_COLOR),
            (0.149, cl.WARN_COLOR),
            (0.15, cl.BAD_COLOR),
            (5.0, cl.BAD_COLOR),
        ],
    )
    def test_bands_are_left_closed(self, e, want):
        assert cl.residual_colors(np.array([e]))[0].tolist() == list(want)

    def test_sign_does_not_matter(self):
        got = cl.residual_colors(np.array([-0.01, 0.01]))
        assert got[0].tolist() == got[1].tolist() == list(cl.GOOD_COLOR)


class TestPaintResiduals:
    def test_index_equals_coordinate(self):
        """corner 约定:uv 的整数部分**就是**索引(不减 0.5)。"""
        arr = np.zeros((4, 6, 3), dtype=np.uint8)
        assert cl.paint_residuals(arr, np.array([[1.0, 2.0]]), np.array([0.01])) == 1
        assert arr[2, 1].tolist() == list(cl.GOOD_COLOR)
        assert arr[2, 0].tolist() == [0, 0, 0]  # 没有落到隔壁像素

    def test_fractional_uv_floors(self):
        arr = np.zeros((4, 6, 3), dtype=np.uint8)
        cl.paint_residuals(arr, np.array([[3.9, 1.9]]), np.array([0.5]))
        assert arr[1, 3].tolist() == list(cl.BAD_COLOR)

    def test_out_of_range_is_clamped(self):
        arr = np.zeros((4, 6, 3), dtype=np.uint8)
        cl.paint_residuals(arr, np.array([[99.0, -3.0]]), np.array([0.01]))
        assert arr[0, 5].tolist() == list(cl.GOOD_COLOR)

    def test_empty_returns_zero(self):
        arr = np.zeros((4, 6, 3), dtype=np.uint8)
        assert cl.paint_residuals(arr, np.zeros((0, 2)), np.zeros(0)) == 0
        assert not arr.any()

    def test_count_is_number_of_points(self):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        uv = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        assert cl.paint_residuals(arr, uv, np.array([0.01, 0.1, 0.9])) == 3


class TestNearFraction:
    def test_all_far_is_zero(self):
        assert cl.near_fraction(np.full((10, 10), 20.0)) == 0.0

    def test_all_near_is_one(self):
        assert cl.near_fraction(np.full((10, 10), 0.1)) == 1.0

    def test_half(self):
        d = np.full((10, 10), 20.0)
        d[:5] = 0.2
        assert cl.near_fraction(d) == pytest.approx(0.5)

    def test_empty_is_zero(self):
        assert cl.near_fraction(np.zeros((0, 0))) == 0.0

    def test_matches_measured_cam_back(self):
        """实测口径的合成复现:CAM_BACK 0.367(1242×375)/ 0.195(640×360),其余 0.000。"""
        d = np.full((100, 100), 30.0)
        d.ravel()[:3670] = 0.2
        assert cl.near_fraction(d) == pytest.approx(0.367)


class TestSummarize:
    def test_median_reported_when_enough_samples(self):
        st = cl.summarize({"CAM_FRONT": _samples(cl.MIN_CAM_SAMPLES, 0.01)})["CAM_FRONT"]
        assert st.usable and st.median_abs == pytest.approx(0.01)

    def test_below_threshold_reports_none_not_a_number(self):
        """**核心判据**:样本不足时报 `None`,不许拿 3 个样本算出"漂亮数字"。"""
        st = cl.summarize({"CAM_BACK": _samples(cl.MIN_CAM_SAMPLES - 1, 0.0001)})["CAM_BACK"]
        assert st.n == cl.MIN_CAM_SAMPLES - 1
        assert st.median_abs is None and st.p90_abs is None
        assert not st.usable

    def test_zero_samples(self):
        st = cl.summarize({"CAM_BACK": cl.empty_samples()})["CAM_BACK"]
        assert st.n == 0 and st.median_abs is None and not st.usable

    def test_p90_reported(self):
        e = np.linspace(0.0, 1.0, 100)
        s = cp.DepthSamples(np.zeros((100, 2)), np.full(100, 20.0), 20.0 + e, np.zeros((100, 2)))
        st = cl.summarize({"CAM_FRONT": s})["CAM_FRONT"]
        assert st.median_abs == pytest.approx(float(np.median(e)))
        assert st.p90_abs == pytest.approx(float(np.percentile(e, 90)))

    def test_near_fraction_passed_through(self):
        st = cl.summarize({"CAM_BACK": cl.empty_samples()}, {"CAM_BACK": 0.367})["CAM_BACK"]
        assert st.near_fraction == pytest.approx(0.367)

    def test_usable_requires_median(self):
        assert not cl.CameraResidual("X", 100, None, None, 0.0).usable


class TestSelfOccluded:
    """自遮挡判据是**相对**的:样本不足 + 近场占比远高于同批其余相机。

    绝对阈值不成立(实测踩坑):同一台相机同一挂点,近场占比随**画幅宽高比**变 ——
    CAM_BACK 在 1242×375 是 0.367、在 640×360 只有 0.195。写死 `> 0.2` 会让实时槽判不出来。
    """

    @staticmethod
    def _batch(back_near: float, others_near: float = 0.0, back_n: int = 2, other_n: int = 50):
        stats = {
            "CAM_BACK": cl.CameraResidual(
                "CAM_BACK", back_n, None if back_n < cl.MIN_CAM_SAMPLES else 0.001, None, back_near
            )
        }
        for name in ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"):
            stats[name] = cl.CameraResidual(name, other_n, 0.0003, 0.001, others_near)
        return stats

    def test_flags_the_self_occluded_camera(self):
        assert cl.self_occluded_cameras(self._batch(0.367)) == {"CAM_BACK"}

    def test_flags_at_the_live_resolution_fraction(self):
        """0.195(640×360 实测值)也必须判得出来 —— 这正是绝对阈值 > 0.2 会漏掉的。"""
        assert cl.self_occluded_cameras(self._batch(0.195)) == {"CAM_BACK"}

    def test_few_samples_but_no_near_field_is_not_self_occlusion(self):
        """样本少但近场占比正常 ⇒ **不是**自遮挡(不能拿它当借口掩盖真故障)。"""
        assert cl.self_occluded_cameras(self._batch(0.0)) == set()

    def test_enough_samples_is_never_flagged(self):
        """有数据就是可用相机,近场占比再高也不标"自遮挡"(它本来就报得出 median)。"""
        assert cl.self_occluded_cameras(self._batch(0.9, back_n=100)) == set()

    def test_baseline_is_usable_cameras_not_all(self):
        """基准取**可用相机**:若把被挡的那台算进基准,阈值会被自己抬高。"""
        # 全体中位数 = median([0.5, 0.5, 0.0, 0.0, 0.0]) = 0.0 ⇒ 仍判得出
        stats = self._batch(0.5, others_near=0.0)
        stats["CAM_FRONT"] = cl.CameraResidual("CAM_FRONT", 2, None, None, 0.5)
        assert "CAM_BACK" in cl.self_occluded_cameras(stats)

    def test_empty_and_all_unusable(self):
        assert cl.self_occluded_cameras({}) == set()
        only_back = {"CAM_BACK": cl.CameraResidual("CAM_BACK", 2, None, None, 0.367)}
        assert cl.self_occluded_cameras(only_back) == {"CAM_BACK"}


class TestPooledMedian:
    def test_equal_weight_not_sample_weighted(self):
        """等权:某一路样本多 10× 不得把总体中位数拖向它。"""
        stats = {
            "A": cl.CameraResidual("A", 10_000, 0.010, 0.02, 0.0),
            "B": cl.CameraResidual("B", 100, 0.001, 0.002, 0.0),
        }
        assert cl.pooled_median(stats) == pytest.approx(np.median([0.010, 0.001]))

    def test_unusable_cameras_are_excluded(self):
        stats = {
            "A": cl.CameraResidual("A", 100, 0.010, 0.02, 0.0),
            "B": cl.CameraResidual("B", 3, None, None, 0.367),
        }
        assert cl.pooled_median(stats) == pytest.approx(0.010)

    def test_no_data_is_none(self):
        assert cl.pooled_median({"B": cl.CameraResidual("B", 0, None, None, 0.4)}) is None
        assert cl.pooled_median({}) is None


class TestHudLine:
    def test_no_data_says_so_instead_of_zero(self):
        """**必须报"无数据"**:0.000 会被读成"标定完美"。"""
        stats = {"CAM_BACK": cl.CameraResidual("CAM_BACK", 0, None, None, 0.4)}
        line = cl.hud_line(stats, None, 0)
        assert "无数据" in line
        assert "0.0000" not in line

    def test_reports_pooled_and_counts(self):
        stats = {
            "CAM_FRONT": cl.CameraResidual("CAM_FRONT", 100, 0.0003, 0.001, 0.0),
            "CAM_BACK": cl.CameraResidual("CAM_BACK", 0, None, None, 0.367),
        }
        line = cl.hud_line(stats, cl.pooled_median(stats), 230, 3)
        assert "1/2 相机有数据" in line
        assert "0.0003" in line
        assert "平面点 230" in line and "重拟合 3 次" in line

    def test_short_names_are_used(self):
        stats = {n: cl.CameraResidual(n, 50, 0.001, 0.002, 0.0) for n in cl.SHORT_NAMES}
        line = cl.hud_line(stats, 0.001, 10)
        assert "FL" in line and "CAM_FRONT_LEFT" not in line
        assert line.count(" 50") == len(cl.SHORT_NAMES)  # 六路样本数都报出来

    def test_self_occluded_camera_is_annotated(self):
        """自遮挡相机单独标注 —— 免得被当成"标定坏了"。"""
        stats = {
            "CAM_FRONT": cl.CameraResidual("CAM_FRONT", 100, 0.0003, 0.001, 0.0),
            "CAM_BACK": cl.CameraResidual("CAM_BACK", 2, None, None, 0.367),
        }
        line = cl.hud_line(stats, 0.0003, 230)
        assert "自遮挡" in line and "37%" in line

    def test_no_annotation_when_all_usable(self):
        stats = {n: cl.CameraResidual(n, 100, 0.0003, 0.001, 0.0) for n in cl.SHORT_NAMES}
        assert "自遮挡" not in cl.hud_line(stats, 0.0003, 230)

    def test_unknown_camera_name_is_not_dropped(self):
        """未知名字原样出现 —— 静默丢会让 HUD 少一路而看不出来。"""
        stats = {"CAM_SIDE_NEW": cl.CameraResidual("CAM_SIDE_NEW", 50, 0.001, 0.002, 0.0)}
        assert "CAM_SIDE_NEW" in cl.hud_line(stats, 0.001, 10)


class TestSampleCamera:
    def test_empty_planes_gives_empty_samples(self):
        s = cl.sample_camera(
            (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)), (CAM_LOC, CAM_ROT_FWD), K, np.zeros((H, W))
        )
        assert len(s) == 0

    def test_head_on_plane_has_zero_residual(self):
        """平面 x = 25 正对相机、渲染深度图恒 25 ⇒ 残差 0(与离线探针同一口径)。"""
        uv = np.array([[320.0, 180.0], [100.0, 60.0], [500.0, 300.0]])
        pts = _world(uv, 25.0)
        planes = (pts, np.tile([1.0, 0.0, 0.0], (3, 1)), np.full(3, 25.0))
        s = cl.sample_camera(planes, (CAM_LOC, CAM_ROT_FWD), K, np.full((H, W), 25.0))
        assert len(s) == 3
        assert np.abs(s.residual).max() < 1e-9

    def test_occluded_point_is_dropped(self):
        """渲染比预测近得多 ⇒ 被挡住,剔掉(单侧可见性判据)。"""
        uv = np.array([[320.0, 180.0]])
        pts = _world(uv, 25.0)
        planes = (pts, np.array([[1.0, 0.0, 0.0]]), np.array([25.0]))
        s = cl.sample_camera(planes, (CAM_LOC, CAM_ROT_FWD), K, np.full((H, W), 5.0))
        assert len(s) == 0

    def test_writes_onto_the_right_pixels_end_to_end(self):
        """端到端:采样点画到图上,位置与 `project_world` 一致(corner 索引 = u)。"""
        uv = np.array([[320.0, 180.0], [100.0, 60.0]])
        pts = _world(uv, 25.0)
        planes = (pts, np.tile([1.0, 0.0, 0.0], (2, 1)), np.full(2, 25.0))
        s = cl.sample_camera(planes, (CAM_LOC, CAM_ROT_FWD), K, np.full((H, W), 25.0))
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        assert cl.paint_residuals(arr, s.uv, s.residual) == 2
        for u, v in uv.astype(int):
            assert arr[v, u].tolist() == list(cl.GOOD_COLOR)

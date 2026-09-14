"""radar.py 手算锚点 + devkit 往返(纯 numpy;devkit 版本需 nuscenes-devkit)。

需要在 autodrivedata env 跑(nuscenes-devkit==1.2.0 已装)。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.radar import (
    AMBIG_VALID,
    ARS408_VFOV_HALF_DEG,
    INVALID_STATE_VALID,
    IS_QUALITY_VALID,
    detections_to_nus18,
    mask_in_ars408_vfov,
    mask_radar_points,
    nus18_to_pcd,
    valid_mask_nus,
)

# 一个正前方检测:azimuth=0、altitude=0、depth=10、vel=-5(朝传感器接近)
FWD = np.array([[-5.0, 0.0, 0.0, 10.0]], dtype=np.float32)
# 一个偏右+仰角检测:d=20, az=10°, alt=5°, vel=+3(远离)
OFFSET = np.array([[3.0, np.radians(5.0), np.radians(10.0), 20.0]], dtype=np.float32)


class TestDetectionsToNus18:
    def test_front_center(self):
        out = detections_to_nus18(FWD, sensor_id=2)
        assert out.shape == (1, 18)
        assert out.dtype == np.float32
        # x=10, y=0, z=0;vx=+5(反号), vy=0
        np.testing.assert_allclose(out[0, :3], [10.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(out[0, 6:10], [5.0, 0.0, 5.0, 0.0], atol=1e-6)
        # 固定合法字段
        assert out[0, 3] == 0.0  # dyn_prop
        assert out[0, 4] == 2.0  # id = sensor_id
        assert out[0, 10] == IS_QUALITY_VALID
        assert out[0, 11] == AMBIG_VALID
        assert out[0, 14] == INVALID_STATE_VALID
        # 其余 0
        assert out[0, 5] == 0.0  # rcs
        assert out[0, 12] == 0.0 and out[0, 13] == 0.0
        assert out[0, 15] == 0.0 and out[0, 16] == 0.0 and out[0, 17] == 0.0

    def test_offset_hand_anchor(self):
        d, a, e, vr = 20.0, np.radians(10.0), np.radians(5.0), 3.0
        ce, ca, sa, se = np.cos(e), np.cos(a), np.sin(a), np.sin(e)
        out = detections_to_nus18(OFFSET, sensor_id=0)
        np.testing.assert_allclose(out[0, :3], [d * ce * ca, -d * ce * sa, d * se], atol=1e-5)
        np.testing.assert_allclose(out[0, 6:8], [-vr * ce * ca, vr * ce * sa], atol=1e-5)

    def test_n_points(self):
        out = detections_to_nus18(np.vstack([FWD, OFFSET]), sensor_id=0)
        assert out.shape == (2, 18)


class TestValidMaskNus:
    def test_default_filters(self):
        pts = detections_to_nus18(FWD, sensor_id=0)
        # 构造非法:invalid_state != 0 / dyn_prop > 6 / ambig != 3
        bad_inv = pts.copy()
        bad_inv[0, 14] = 7.0
        bad_dyn = pts.copy()
        bad_dyn[0, 3] = 7.0
        bad_amb = pts.copy()
        bad_amb[0, 11] = 1.0
        good = pts.copy()
        assert valid_mask_nus(good)[0]
        assert not valid_mask_nus(bad_inv)[0]
        assert not valid_mask_nus(bad_dyn)[0]
        assert not valid_mask_nus(bad_amb)[0]


class TestMaskInArs408Vfov:
    """垂直锥 ±7.1° 裁剪:判据 |z| <= sin(7.1°)·depth(18 字段无 alt,用 z/depth 反推)。"""

    def _pt(self, d: float, e_deg: float) -> np.ndarray:
        """深度 d、仰角 e_deg 的点(az=0):x=d·cos e, z=d·sin e。"""
        e = np.radians(e_deg)
        return detections_to_nus18(np.array([[0.0, e, 0.0, d]], dtype=np.float32), sensor_id=0)

    def test_within_vfov_kept(self):
        # 仰角 5° < 7.1° → 保留
        assert bool(mask_in_ars408_vfov(self._pt(20.0, 5.0))[0])

    def test_outside_vfov_dropped(self):
        # 仰角 20° >> 7.1° → 丢弃(CARLA 布到 ±38° 的锥外点)
        assert not bool(mask_in_ars408_vfov(self._pt(20.0, 20.0))[0])

    def test_boundary_exact_half(self):
        # 恰在 7.1° → 保留(带 1e-6 容差)
        assert bool(mask_in_ars408_vfov(self._pt(100.0, ARS408_VFOV_HALF_DEG))[0])

    def test_far_distance_not_widened(self):
        # 远距 200m、仰角 20° 仍丢弃(锥角判据不随 depth 放宽)
        assert not bool(mask_in_ars408_vfov(self._pt(200.0, 20.0))[0])

    def test_empty_cloud_mask_false(self):
        # 空点云(全 NaN)→ z/depth 全 NaN → 比较 False → 掩码 False
        m = mask_in_ars408_vfov(np.full((1, 18), np.nan, dtype=np.float32))
        assert not bool(m[0])


class TestMaskRadarPoints:
    def test_combines_both_filters(self):
        # 构造一个垂直锥内但 devkit 非法(ambig=1)的点 → 组合掩码 False
        good = self._pts()
        assert all(mask_radar_points(good))
        bad_amb = good.copy()
        bad_amb[0, 11] = 1.0
        assert not mask_radar_points(bad_amb)[0]

    def _pts(self) -> np.ndarray:
        # 深度 20m、仰角 0°(锥内)、az 0
        return detections_to_nus18(np.array([[0.0, 0.0, 0.0, 20.0]], dtype=np.float32), sensor_id=0)

    def test_outside_vfov_dropped_by_combined(self):
        out = detections_to_nus18(
            np.array([[0.0, np.radians(30.0), 0.0, 20.0]], dtype=np.float32), sensor_id=0
        )
        assert not mask_radar_points(out)[0]


class TestNus18ToPcd:
    def test_roundtrip_via_devkit(self, tmp_path):
        pytest.importorskip("nuscenes")
        from nuscenes.utils.data_classes import RadarPointCloud

        pts = detections_to_nus18(np.vstack([FWD, OFFSET]), sensor_id=0)
        pcd = tmp_path / "radar.pcd"
        pcd.write_bytes(nus18_to_pcd(pts))
        pc = RadarPointCloud.from_file(str(pcd))
        assert pc.points.ndim == 2 and pc.points.shape[0] == 18
        assert pc.points.shape[1] == 2  # 默认过滤不掉光
        np.testing.assert_allclose(pc.points[:3, :].T, pts[:, :3], atol=1e-5)

    def test_empty_pcd_via_devkit(self, tmp_path):
        pytest.importorskip("nuscenes")
        from nuscenes.utils.data_classes import RadarPointCloud

        pcd = tmp_path / "empty.pcd"
        pcd.write_bytes(nus18_to_pcd(np.empty((0, 18))))
        pc = RadarPointCloud.from_file(str(pcd))
        assert pc.points.shape == (18, 0)  # NaN 单点 → devkit 空点云

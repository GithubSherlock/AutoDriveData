"""geometry.py nuScenes 约定手算锚点单测(语义照 auto3dlabel tools/geometry.py)。"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata import geometry as g


class TestCarlaToNus:
    def test_y_flip(self):
        # CARLA (x 前, y 右, z 上) → nuScenes (x 前, y 左, z 上):仅 y 翻号
        np.testing.assert_allclose(
            g.carla_to_nus_global(np.array([[1.0, 2.0, 3.0]])), [[1.0, -2.0, 3.0]], atol=1e-12
        )

    def test_yaw_flip(self):
        # CARLA yaw 左转为正;nuScenes yaw 右转(绕 z 俯视逆时针)为正
        assert g.carla_yaw_to_nus_yaw(np.pi / 2) == pytest.approx(-np.pi / 2)
        assert g.carla_yaw_to_nus_yaw(-np.pi / 4) == pytest.approx(np.pi / 4)
        assert g.carla_yaw_to_nus_yaw(0.0) == pytest.approx(0.0)

    def test_heading_consistency(self):
        # CARLA 车头 (cos ψ, sin ψ) 翻 y 后 = nuScenes 车头 (cos ψ, −sin ψ) = (cos ψ_n, sin ψ_n)
        for psi in [-2.0, -1.0, 0.0, 0.5, 2.0]:
            heading_nus = g.CARLA_TO_NUS @ np.array([np.cos(psi), np.sin(psi), 0.0])
            psi_n = g.carla_yaw_to_nus_yaw(psi)
            np.testing.assert_allclose(
                heading_nus[:2], [np.cos(psi_n), np.sin(psi_n)], atol=1e-12
            )


class TestQuat:
    def test_yaw_to_quat_anchors(self):
        # 照抄 auto3dlabel:quat = (cos(yaw/2), 0, 0, sin(yaw/2))
        assert g.yaw_to_quat(0.0) == pytest.approx((1.0, 0.0, 0.0, 0.0))
        s = np.sqrt(2) / 2
        w, x, y, z = g.yaw_to_quat(np.pi / 2)
        assert (w, x, y, z) == pytest.approx((s, 0.0, 0.0, s))
        w, x, y, z = g.yaw_to_quat(np.pi)
        assert (w, x, y, z) == pytest.approx((0.0, 0.0, 0.0, 1.0))

    def test_round_trip(self):
        for psi in [-np.pi + 0.01, -1.0, 0.0, 1.0, np.pi - 0.01]:
            assert g.quat_to_yaw(g.yaw_to_quat(psi)) == pytest.approx(psi)

    def test_carla_yaw_round_trip_via_quat(self):
        for psi in [-2.5, -1.0, 0.3, 2.0]:
            psi_n = g.carla_yaw_to_nus_yaw(psi)
            assert g.quat_to_yaw(g.carla_yaw_to_nus_quat(psi)) == pytest.approx(psi_n)

    def test_quat_semantics_matches_autolabel(self):
        """quat 语义与 auto3dlabel 一致(单测锁定;oracle 文件另有逐点对照)。"""
        from autodrivedata.geometry import quat_to_yaw as ours

        # auto3dlabel quat_to_yaw 实现:wrap_pi(2*arctan2(z, w))
        q = g.yaw_to_quat(1.234)
        expected = g.wrap_pi(2 * float(np.arctan2(q[3], q[0])))
        assert ours(q) == pytest.approx(expected)

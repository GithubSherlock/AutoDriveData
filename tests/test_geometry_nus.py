"""geometry.py nuScenes 约定手算锚点单测(语义照 auto3dlabel tools/geometry.py)。"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata import geometry as g
from autodrivedata.camera_rig import (
    NUS_CAMERA_CALIBS,
    NUS_CAMERA_RIG,
    NUS_CAMERAS,
    nus_camera_rig,
)


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
            np.testing.assert_allclose(heading_nus[:2], [np.cos(psi_n), np.sin(psi_n)], atol=1e-12)


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

    def test_quat_normalize_makes_matrix_orthogonal(self):
        """官方四元数**不是单位长度** ⇒ 不归一化时闭式解给出的矩阵非正交(姿态误差 ~1e-4 rad)。"""
        q = NUS_CAMERA_CALIBS["CAM_FRONT"][1]
        assert np.linalg.norm(q) != pytest.approx(1.0, abs=1e-9)  # 前提:官方值确实非单位
        raw = g.quat_to_matrix(q)
        fixed = g.quat_to_matrix(g.quat_normalize(q))
        assert np.abs(raw @ raw.T - np.eye(3)).max() > 1e-5  # 未归一化 ⇒ 非正交
        np.testing.assert_allclose(fixed @ fixed.T, np.eye(3), atol=1e-15)

    def test_quat_normalize_is_idempotent_and_rejects_zero(self):
        q = NUS_CAMERA_CALIBS["CAM_BACK_LEFT"][1]
        once = g.quat_normalize(q)
        assert g.quat_normalize(once) == pytest.approx(once, abs=1e-15)
        with pytest.raises(ValueError):
            g.quat_normalize((0.0, 0.0, 0.0, 0.0))


class TestNusCameraRigDerivation:
    """`camera_rig.NUS_CAMERA_RIG` 的推导钉:**yaw_carla = −az_nus**、平移只翻 y。

    **为什么单独钉**(2026-09-22,Plan2.md §P-M.1):旧 `official` rig 把官方方位角**原样抄成
    正数**,四个侧/后相机左右镜像(FRONT_LEFT/RIGHT 差 110.3°、BACK_LEFT/RIGHT 差 217.2°),
    而前/后相机因光轴近自逆"看着对",长期没暴露。本类把这条推导钉到 1e-9。
    """

    @staticmethod
    def _az_nus(name: str) -> float:
        """官方标定 → 该相机视线轴在 nuScenes 全局系的方位角(度)。"""
        r_nus = g.quat_to_matrix(g.quat_normalize(NUS_CAMERA_CALIBS[name][1]))
        boresight = r_nus[:, 2]  # 相机自身系 x右/y下/z前 ⇒ 视线轴 = +z 列
        return float(np.degrees(np.arctan2(boresight[1], boresight[0])))

    def test_yaw_equals_minus_official_azimuth(self):
        """rig 的 yaw 字段 == −az_nus(六个相机逐一,1e-9 度内)。"""
        for name in NUS_CAMERA_CALIBS:
            yaw_carla = NUS_CAMERA_RIG[name][1][1]
            assert yaw_carla == pytest.approx(-self._az_nus(name), abs=1e-9), name

    def test_historical_literals_would_be_off_by_110_and_217_deg(self):
        """历史 bug 的形状:把官方方位角原样抄成正数 ⇒ 四个侧/后相机偏差 110–222°。

        侧/后取 `|历史值 − 正确值|` **不 wrap** —— 它们超过 180°,wrap 会把 217.2° 折成 142.8°
        而看不出"镜像"这件事。前/后相机近自逆 ⇒ 偏差只 0.15–0.32°(且 CAM_BACK 的原始差是
        359.85°、wrap 后才是 0.145°)—— **这正是它长期没暴露的原因**,见 `camera_rig` 模块头注。
        """
        historical = {
            "CAM_FRONT": 0.0,
            "CAM_FRONT_LEFT": 55.0,
            "CAM_FRONT_RIGHT": -55.0,
            "CAM_BACK": 180.0,
            "CAM_BACK_LEFT": 108.6,
            "CAM_BACK_RIGHT": -110.8,
        }
        raw = {n: abs(historical[n] - NUS_CAMERA_RIG[n][1][1]) for n in historical}
        wrapped = {
            n: abs((historical[n] - NUS_CAMERA_RIG[n][1][1] + 180.0) % 360.0 - 180.0) for n in historical
        }
        assert raw["CAM_FRONT_LEFT"] == pytest.approx(110.2, abs=0.1)
        assert raw["CAM_FRONT_RIGHT"] == pytest.approx(111.4, abs=0.1)
        assert raw["CAM_BACK_LEFT"] == pytest.approx(217.2, abs=0.1)
        assert raw["CAM_BACK_RIGHT"] == pytest.approx(221.6, abs=0.1)
        # 前/后:wrap 后偏差小到看不出问题(判据对它们无信息量,故 A1 只查 4 个侧相机)
        assert wrapped["CAM_FRONT"] < 0.4 and wrapped["CAM_BACK"] < 0.4

    def test_translation_flips_y_only(self):
        """平移只翻 y(与姿态翻转同一次基变换);x/z 逐字段照抄官方。"""
        for name, (t_nus, _) in NUS_CAMERA_CALIBS.items():
            t_rig = NUS_CAMERA_RIG[name][0]
            assert t_rig[0] == pytest.approx(t_nus[0], abs=1e-12), name
            assert t_rig[1] == pytest.approx(-t_nus[1], abs=1e-12), name
            assert t_rig[2] == pytest.approx(t_nus[2], abs=1e-12), name

    def test_rotation_is_not_yaw_only(self):
        """6DoF 不可降成 yaw-only:官方 pitch/roll 非零(最大 |pitch| 0.96° / |roll| 0.62°)。

        30 m 处 1° 指向误差 ≈ 0.5 m 横向偏移 —— 对时序建图的位姿链是可观测量级。
        """
        pitches = [abs(NUS_CAMERA_RIG[n][1][0]) for n in NUS_CAMERA_RIG]
        rolls = [abs(NUS_CAMERA_RIG[n][1][2]) for n in NUS_CAMERA_RIG]
        assert max(pitches) > 0.9  # 硬编码 0 会让这条挂
        assert max(rolls) > 0.6

    def test_derivation_reproduces_the_frozen_table(self):
        """展开表与 `nus_camera_rig()` 同源(改官方标定即自动跟随,不存在第二份手抄)。"""
        assert NUS_CAMERA_RIG == nus_camera_rig()
        assert set(NUS_CAMERA_RIG) == set(NUS_CAMERA_CALIBS) == set(NUS_CAMERAS)

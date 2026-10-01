"""nuCarla rig 的**独立复算自证** + 三处差异的回归钉。

## 为什么值得钉

nuCarla 发了四个 BEV 模型的预训练权重(免训 300 GPU-h)。要在**我们的图**上跑泛化,就必须用
**它的 rig** 采我们的图 —— 本项目红线(§P-L.1):**rig 必须与权重训练数据一致**(错配代价实测
122711 vs 135989 px)。所以「我们导出的 `NUS_CAMERA_RIG_NUCARLA` 是否**真的是他们那套**」
是个**必须对表**的问题,不是"看着像"的问题。

## 自证怎么做的(关键:独立实现)

`NUS_CAMERA_RIG_NUCARLA` 是走**我们的**换算链(`nus_camera_rotation_to_carla` →
`rotation_matrix_to_carla`,矩阵路径)导出的;本测试**不复用那条链**,而是照 nuCarla
`sensors.py` 的**四元数路径**另走一遍:

    q_new = yaw_q(−90°) · roll_q(−90°) · nus_q⁻¹   →   yaw_pitch_roll

两条路径**数学上不同**,同解才是证据。与项目既有"用 `_sample_bev` 当 oracle""`rigviz.azimuth_of`
两套独立实现"同一手法。

## 三处差异也钉住

差异不是 bug,是**事实**;不钉的话下一个人会拿我们的 `NUS_CAMERA_CALIBS` 去"简化"这张表。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autodrivedata.calib.camera_rig import (
    NUCARLA_CAMERA_CALIBS,
    NUCARLA_CAMERA_FOV,
    NUCARLA_EGO_ORIGIN_X,
    NUS_CAMERA_CALIBS,
    NUS_CAMERA_RIG,
    NUS_CAMERA_RIG_NUCARLA,
)

# 官方 FOV 表的历史落点在导出侧(采集/导出共用),nuCarla 那张在本模块 —— 见 camera_rig 的注
from autodrivedata.gt.export.nuscenes import NUS_CAMERA_FOV
from autodrivedata.utils.geometry import NUS_EGO_ORIGIN_X, nus_mount_to_carla


# ---------------------------------------------------------------- 独立复算:nuCarla 的数学
def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _axis_angle(axis, deg):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    h = math.radians(deg) / 2.0
    return np.array([math.cos(h), *(a * math.sin(h))])


def _yaw_pitch_roll(q):
    """pyquaternion `Quaternion.yaw_pitch_roll` 的等价实现(Z-Y-X intrinsic)。"""
    w, x, y, z = q
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * x - y * z))))
    roll = math.atan2(2 * (w * y + x * z), 1 - 2 * (x * x + y * y))
    return yaw, pitch, roll


def _nucarla_reference(channel: str, center_to_wheelbase: float = 1.317) -> tuple[tuple, tuple]:
    """照 `nuCarla/data:sensors.py` 逐行复算一个通道的 CARLA `(Location, Rotation)`。"""
    t_nus, q_nus = NUCARLA_CAMERA_CALIBS[channel]
    q_new = _qmul(
        _qmul(_axis_angle([0, 0, 1], -90), _axis_angle([1, 0, 0], -90)),
        np.array([q_nus[0], -q_nus[1], -q_nus[2], -q_nus[3]]),  # inverse
    )
    yaw, pitch, roll = _yaw_pitch_roll(q_new)
    return (
        (t_nus[0] - center_to_wheelbase, -t_nus[1], t_nus[2]),
        tuple(np.degrees([pitch, yaw, roll])),
    )


class TestIndependentRecomputation:
    """★ 主判据:我们导出的 rig == 照 `sensors.py` 独立复算的结果。"""

    @pytest.mark.parametrize("channel", sorted(NUCARLA_CAMERA_CALIBS))
    def test_location_matches(self, channel: str):
        loc_ref, _ = _nucarla_reference(channel)
        loc_ours = NUS_CAMERA_RIG_NUCARLA[channel][0]
        np.testing.assert_allclose(loc_ours, loc_ref, atol=1e-9, err_msg=f"{channel} 挂点")

    @pytest.mark.parametrize("channel", sorted(NUCARLA_CAMERA_CALIBS))
    def test_rotation_matches(self, channel: str):
        _, rot_ref = _nucarla_reference(channel)
        rot_ours = NUS_CAMERA_RIG_NUCARLA[channel][1]
        np.testing.assert_allclose(rot_ours, rot_ref, atol=1e-9, err_msg=f"{channel} 姿态")

    def test_origin_constant_is_used(self):
        """`center_to_wheelbase` 必须真的进换算 —— 用错常量会整体偏 0.0607 m。

        反向自证:把常量换成我们的,结果必须**不同**(否则说明它根本没被读)。
        """
        loc_a, _ = _nucarla_reference("CAM_FRONT", center_to_wheelbase=1.317)
        loc_b, _ = _nucarla_reference("CAM_FRONT", center_to_wheelbase=-NUS_EGO_ORIGIN_X)
        assert abs(loc_a[0] - loc_b[0]) == pytest.approx(0.0607, abs=1e-4)
        assert NUS_CAMERA_RIG_NUCARLA["CAM_FRONT"][0][0] == pytest.approx(loc_a[0], abs=1e-9)


class TestDocumentedDifferences:
    """三处差异是**事实**不是 bug —— 钉住,防下一个人拿我们的表去"简化"。"""

    def test_quaternions_differ_from_ours_by_transcription_rounding(self):
        """他们的 config 是**全精度官方值**;我们的 `NUS_CAMERA_CALIBS` 是**手抄 4 位小数**。

        等价(|Δq| 只有 3–5e-05,即本模块头注记的那次舍入),但**不是同一份数** ——
        所以这张表必须**逐字抄他们**,不能拿我们的代抄。
        """
        deltas = []
        for ch, (_t, q_nc) in NUCARLA_CAMERA_CALIBS.items():
            _t2, q_us = NUS_CAMERA_CALIBS[ch]
            deltas.append(float(np.abs(np.array(q_nc) - np.array(q_us)).max()))
        assert max(deltas) < 1e-4, f"舍入量级变了:{max(deltas):.2e}"
        assert min(deltas) > 1e-6, "两边竟然逐位相同 —— 那就不必另立一张表了"

    def test_cam_front_x_is_the_only_translation_difference(self):
        """**只有 CAM_FRONT 的 x 与官方差 +0.2 m**,其余五路逐位相同。"""
        for ch, (t_nc, _q) in NUCARLA_CAMERA_CALIBS.items():
            t_us, _q2 = NUS_CAMERA_CALIBS[ch]
            dx = t_nc[0] - t_us[0]
            if ch == "CAM_FRONT":
                # 残差 8.8e-06 = **我们**表把官方 1.70079118954 抄成了 1.7008(四位小数),
                # 不是 nuCarla 的差 —— 所以容差按我们的舍入精度给
                assert dx == pytest.approx(0.2, abs=1e-4)
            else:
                # 同上:残差 ≤2.6e-05 全是**我们**表的四位小数舍入
                assert dx == pytest.approx(0.0, abs=1e-4), f"{ch} 的 x 也差了 {dx:.6f}"
            # y/z 六路全同 —— **容差按我们表的舍入精度(四位小数)** 给:
            # `NUS_CAMERA_CALIBS` 是手抄四位(0.0159),他们的 config 是全精度(0.0159456324149)
            assert t_nc[1] == pytest.approx(t_us[1], abs=1e-4)
            assert t_nc[2] == pytest.approx(t_us[2], abs=1e-4)

    def test_origin_constants_differ(self):
        assert NUCARLA_EGO_ORIGIN_X == pytest.approx(-1.3170)
        assert NUS_EGO_ORIGIN_X == pytest.approx(-1.2563)
        assert abs(NUCARLA_EGO_ORIGIN_X - NUS_EGO_ORIGIN_X) == pytest.approx(0.0607, abs=1e-6)


class TestConventionSwap:
    """★ **复现别人的 rig,必须复现别人的换算** —— 只抄标定表不够。

    `sensors.py` 把 pyquaternion 的**右手 Z-Y-X** `(yaw, pitch, roll)` **直接喂进
    `carla.Rotation`(UE 左手)**;我们的 `nus_camera_rotation_to_carla` 走的是**验证过的**
    UE 口径(§P-M.10 ③:`Rz(−yaw)·Ry(−pitch)·Rx(+roll)`)。

    ⇒ 同一条标定,两条链给出**不同的 pitch/roll**。实测关系(六路全部成立):
    **yaw 相同;`nuCarla.pitch == −ours.roll`,`nuCarla.roll == ours.pitch`**。

    **为什么必须复现对方那一套**:模型学的是**他们渲染出来的图**。他们的相机若有 ~0.4° 俯仰偏差,
    我们也要有 —— 否则凭空多一个域差,而那个差会被读成"泛化能力差"。
    """

    def test_yaw_agrees_but_pitch_roll_are_swapped(self):
        from autodrivedata.calib.camera_rig import nus_camera_rig

        ours_chain = nus_camera_rig(NUCARLA_CAMERA_CALIBS, origin_x=NUCARLA_EGO_ORIGIN_X)
        for ch in NUCARLA_CAMERA_CALIBS:
            p_nc, y_nc, r_nc = NUS_CAMERA_RIG_NUCARLA[ch][1]
            p_us, y_us, r_us = ours_chain[ch][1]
            assert y_nc == pytest.approx(y_us, abs=1e-2), f"{ch} yaw 竟不一致"
            assert p_nc == pytest.approx(-r_us, abs=1e-2), f"{ch} pitch ≠ −(我们链的 roll)"
            assert r_nc == pytest.approx(p_us, abs=1e-2), f"{ch} roll ≠ (我们链的 pitch)"

    def test_the_swap_is_not_negligible(self):
        """反向对照:这个差**不小到可以忽略** —— 六路里最大的 pitch 差有 **~1.2°**。

        (若哪天变成 0,说明有人把换算换成我们那条了 —— 而那会让模型的输入凭空偏 ~0.5–1.2°。)
        """
        from autodrivedata.calib.camera_rig import nus_camera_rig

        ours_chain = nus_camera_rig(NUCARLA_CAMERA_CALIBS, origin_x=NUCARLA_EGO_ORIGIN_X)
        worst = max(
            abs(NUS_CAMERA_RIG_NUCARLA[ch][1][0] - ours_chain[ch][1][0]) for ch in NUCARLA_CAMERA_CALIBS
        )
        assert worst > 0.3, f"最大 pitch 差只有 {worst:.3f}° —— 换算被换掉了?"


class TestFovIsTheBiggestDifference:
    """★ **FOV 才是那处会让人白跑一轮的差** —— 它不在标定表里,所以最容易被漏掉。"""

    def test_nucarla_is_flat_65(self):
        assert set(NUCARLA_CAMERA_FOV) == set(NUCARLA_CAMERA_CALIBS)
        assert set(NUCARLA_CAMERA_FOV.values()) == {65.0}

    def test_cam_back_differs_by_24_degrees(self):
        """官方/我们的 CAM_BACK 是 **89.34°**,他们统一 **65°** —— 差 **24.3°**。

        内参由 FOV 导出,**BEV 模型的几何直接依赖它** ⇒ 拿我们的 CAM_BACK 喂它的模型必然错。
        """
        d = NUS_CAMERA_FOV["CAM_BACK"] - NUCARLA_CAMERA_FOV["CAM_BACK"]
        assert d == pytest.approx(24.34, abs=0.01), f"CAM_BACK FOV 差变了:{d:.2f}"

    def test_five_others_differ_by_less_than_a_degree(self):
        """其余五路只差 0.2–0.7° —— **小得多,但不能当零**(内参仍会变)。"""
        for ch in ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"):
            d = abs(NUS_CAMERA_FOV[ch] - NUCARLA_CAMERA_FOV[ch])
            assert 0.0 < d < 1.0, f"{ch} 差 {d:.2f}°"


class TestNoRegressionOnTheOfficialRig:
    """`origin_x` 参数化不许改到官方 rig —— 那是 §P-M.7/.10 十条判据钉过的东西。"""

    def test_default_origin_is_unchanged(self):
        assert nus_mount_to_carla((1.7008, 0.0159, 1.5110)) == (
            1.7008 + NUS_EGO_ORIGIN_X,
            -0.0159,
            1.5110,
        )

    def test_official_rig_values_pinned(self):
        """逐项锚定(抄自 §P-M.10 之后的既成事实):改了这里就是动了冻结口径。"""
        assert NUS_CAMERA_RIG["CAM_FRONT"][0][0] == pytest.approx(0.4445, abs=5e-5)
        assert NUS_CAMERA_RIG["CAM_BACK"][0][0] == pytest.approx(-1.2280, abs=5e-5)
        assert NUS_CAMERA_RIG["CAM_FRONT"][1][1] == pytest.approx(-0.321, abs=1e-3)

    def test_two_rigs_are_not_accidentally_equal(self):
        """反向对照:两套 rig **必须不同** —— 否则说明 nuCarla 那张表没生效。"""
        for ch in NUCARLA_CAMERA_CALIBS:
            a = np.array(NUS_CAMERA_RIG[ch][0])
            b = np.array(NUS_CAMERA_RIG_NUCARLA[ch][0])
            assert np.abs(a - b).max() > 1e-3, f"{ch} 两套 rig 相同?"


class TestRegistryWiring:
    """`nucarla` 必须真的进 `NUS_RIGS` 注册表 —— 否则新 rig 只是"存在但没人能用"。

    注册表是**三处派发**的唯一来源(`_intrinsics` / `camera_calibs` / `camera_fov`,外加
    `collect_nus` 的挂点解析)。挂在这里的判据保证:加一条就是全链路可用,而不是只在
    `camera_rig` 里躺一张表。
    """

    def test_nucarla_is_registered(self):
        from autodrivedata.gt.export.nuscenes import NUS_RIGS

        assert "nucarla" in NUS_RIGS
        assert set(NUS_RIGS) == {"nuscenes", "wide", "nucarla"}

    def test_all_three_dispatch_points_resolve(self):
        from autodrivedata.gt.export.nuscenes import _intrinsics, camera_calibs, camera_fov

        for fn in (camera_calibs, camera_fov, _intrinsics):
            got = fn("nucarla")
            assert set(got) == set(NUCARLA_CAMERA_CALIBS), f"{fn.__name__} 通道集不对"

    def test_unknown_rig_still_raises(self):
        """新加的 `if` 分支不许把"未知 rig 报错"这条挤掉 —— 静默取默认会让落盘与 spawn 分叉。"""
        from autodrivedata.gt.export.nuscenes import _intrinsics, camera_calibs, camera_fov

        for fn in (camera_calibs, camera_fov, _intrinsics):
            with pytest.raises(ValueError):
                fn("definitely_not_a_rig")

    def test_collect_nus_mount_branch_resolves(self):
        """`collect_nus` 的挂点解析必须认 nucarla(它取的是 CARLA 侧展开表,不是标定原值)。"""
        from autodrivedata.sim.collect_nus import rig_tables

        mounts, _calibs, fov = rig_tables("nucarla")
        assert mounts is NUS_CAMERA_RIG_NUCARLA
        assert fov["CAM_BACK"] == 65.0

"""geometry.py nuScenes 约定手算锚点单测(语义照 auto3dlabel tools/geometry.py)。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autodrivedata import geometry as g
from autodrivedata.calib.camera_rig import (
    NUS_CAMERA_CALIBS,
    NUS_CAMERA_RIG,
    NUS_CAMERAS,
    nus_camera_rig,
)
from autodrivedata.export import nuscenes as ne


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
        """**本表手抄的 4 位小数舍入**让四元数非单位 ⇒ 不归一化时闭式解给出的矩阵非正交。

        **归因订正(2026-09-23,Plan2.md §P-M.7.6)**:旧断言写"官方四元数不是单位长度
        (模长 0.99994~1.00005)"——**归因错了**。官方 mini 集 120 条 `calibrated_sensor`
        的 |q| 实测全为 1.000000000000(最大偏离 2.22e-16,IEEE754 正常舍入);非单位的是
        **我们这张表**(把官方值抄成 4 位小数)。断言本身没错(表确实非单位),故保留,
        另加一条钉住"偏离量级 = 手抄舍入"(≥ 1e-5),归因写反时这条会挂。
        """
        q = NUS_CAMERA_CALIBS["CAM_FRONT_LEFT"][1]
        assert abs(np.linalg.norm(q) - 1.0) >= 1e-5  # 手抄 4 位小数 ⇒ 偏离 ~1e-5 量级
        assert abs(np.linalg.norm(q) - 1.0) < 1e-3  # 但远不到"另一套标定"的量级
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

    def test_translation_flips_y_and_shifts_x_origin(self):
        """平移翻 y + **x 平移原点差**(§P-M.10);z 逐字段照抄官方。

        历史缺陷:这里曾写 `t_rig[0] == t_nus[0]`,把两套**不同原点**的 x 当成同一个量 ——
        nus 原点是**后轴中心**,CARLA 车辆 actor 的原点是**车身长度中点**,差 1.2563 m。
        整套 6 相机 + 6 雷达 + LiDAR 因此齐齐偏前 1.2563 m(渲染与声明**一致地**错,查表全绿)。
        """
        for name, (t_nus, _) in NUS_CAMERA_CALIBS.items():
            t_rig = NUS_CAMERA_RIG[name][0]
            assert t_rig[0] == pytest.approx(t_nus[0] + g.NUS_EGO_ORIGIN_X, abs=1e-12), name
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


class TestNusSensorRigDerivation:
    """LiDAR / 雷达的 CARLA 侧推导钉:**同一条对合规则** `yaw_carla = −az_nus`。

    **为什么单独钉**(2026-09-23,Plan2.md §P-M.7.3/.4):`collect_nus.py` 的雷达偏航曾是
    "与同名相机同号"的猜测表 `{0,+45,−45,+90,−90}`,四路角雷达实测差 **94–136°**;
    LiDAR 更是**完全没设 rotation**(= 宣称传感器系 = ego 系,点云绕 z 转 90°)。
    规则本身与相机完全相同,差别只在**传感器自身系**:相机是 x右/y下/z前(多一段
    `CARLA_TO_CAM`),LiDAR/雷达是 x前/y左/z上(两侧同阵 `CARLA_TO_NUS`,对合)。
    """

    @staticmethod
    def _az_nus_sensor(q_nus) -> float:
        """官方标定 → 视线轴在 nuScenes 全局系的方位角(度)。LiDAR/雷达视线轴 = 自身系 +x。"""
        r_nus = g.quat_to_matrix(g.quat_normalize(q_nus))
        boresight = r_nus[:, 0]  # x 前 / y 左 / z 上 ⇒ 视线轴 = +x 列
        return float(np.degrees(np.arctan2(boresight[1], boresight[0])))

    def test_radar_yaw_equals_minus_official_azimuth(self):
        """5 雷达:由官方四元数导出的 CARLA yaw == −az_nus(1e-9 度内)。

        官方雷达四元数 pitch/roll **精确为 0**(实测),故这里也能顺便钉住"yaw-only 无损"。
        """
        for ch, (_, yaw_nus) in ne.NUS_RADAR_OFFSETS.items():
            az_nus = self._az_nus_sensor(g.yaw_to_quat(yaw_nus))
            assert az_nus == pytest.approx(np.degrees(yaw_nus), abs=1e-9), ch
            r_carla = g.nus_sensor_rotation_to_carla(g.yaw_to_quat(yaw_nus))
            pitch, yaw, roll = g.rotation_matrix_to_carla(r_carla)
            assert np.degrees(yaw) == pytest.approx(-az_nus, abs=1e-9), ch
            assert abs(np.degrees(pitch)) < 1e-9 and abs(np.degrees(roll)) < 1e-9, ch

    def test_radar_historical_literals_would_be_off_by_94_to_136_deg(self):
        """历史 bug 的形状:偏航取"与同名相机同号"⇒ 四路角雷达差 94–136°(CARLA 侧口径)。

        偏差必须在 **CARLA 侧**算:历史表 `RADAR_YAW_OFFSET` 是 spawn 用的 CARLA yaw,
        正确值 = `−az_nus`(本仓 `RADAR_YAW_OFFSET` 现在就是这么导出的)。
        """
        historical = {
            "RADAR_FRONT": 0.0,
            "RADAR_FRONT_LEFT": 45.0,
            "RADAR_FRONT_RIGHT": -45.0,
            "RADAR_BACK_LEFT": 90.0,
            "RADAR_BACK_RIGHT": -90.0,
        }
        diff = {
            ch: abs((historical[ch] - (-math.degrees(yaw)) + 180.0) % 360.0 - 180.0)
            for ch, (_, yaw) in ne.NUS_RADAR_OFFSETS.items()
        }
        assert diff["RADAR_FRONT"] < 0.3  # 前雷达近自逆 ⇒ 看不出问题(正是它长期没暴露的原因)
        # 与 Plan2 §P-M.7.4 的 133.36 / 135.98 / 95.59 / 93.89° **逐位同源**
        # (本表的弧度取自官方四元数 `2·atan2(z,w)`,即 A4 要求的 −az_nus 精确值)
        assert diff["RADAR_FRONT_LEFT"] == pytest.approx(133.3600, abs=0.001)
        assert diff["RADAR_FRONT_RIGHT"] == pytest.approx(135.9800, abs=0.001)
        assert diff["RADAR_BACK_LEFT"] == pytest.approx(95.5900, abs=0.001)
        assert diff["RADAR_BACK_RIGHT"] == pytest.approx(93.8900, abs=0.001)

    def test_lidar_rotation_is_not_yaw_only(self):
        """★ LiDAR 的 up 轴倾角 1.4289° 是**可观测量**:丢 pitch/roll ⇒ 复现比值 1.0000 → 0.9097。

        360° 扫描下 yaw **不可观测**(测不出来),但"不可观测"≠"不用写":点云存的是传感器
        自身系,devkit 按 `calibrated_sensor.rotation` 解释。故 `calib_lidar` 必须是四元数。
        """
        r_carla = g.nus_sensor_rotation_to_carla(ne.NUS_LIDAR_CALIB[1])
        pitch, yaw, roll = g.rotation_matrix_to_carla(r_carla)
        assert np.degrees(pitch) == pytest.approx(-0.3380, abs=1e-3)
        assert np.degrees(yaw) == pytest.approx(89.8835, abs=1e-3)
        assert np.degrees(roll) == pytest.approx(-1.3884, abs=1e-3)
        # up 轴倾角 = 官方四元数把 (0,0,1) 转到的方向与竖直的夹角
        up = g.quat_to_matrix(g.quat_normalize(ne.NUS_LIDAR_CALIB[1])) @ np.array([0.0, 0.0, 1.0])
        tilt = math.degrees(math.acos(float(np.clip(up[2], -1.0, 1.0))))
        assert tilt == pytest.approx(1.4289, abs=0.001)
        # 只留 yaw 会把倾角丢掉 ⇒ 这不是可省的自由度
        r_yaw_only = g.carla_rotation_matrix((0.0, yaw, 0.0))
        assert np.abs(r_carla - r_yaw_only).max() > 0.01

    def test_sensor_involution_round_trip(self):
        """`CARLA_TO_NUS` 对合 ⇒ 同一条链能把 CARLA 侧旋转送回 nuScenes 侧(1e-12)。"""
        for q in (ne.NUS_LIDAR_CALIB[1], *(g.yaw_to_quat(y) for _, y in ne.NUS_RADAR_OFFSETS.values())):
            r_carla = g.nus_sensor_rotation_to_carla(q)
            back = g.CARLA_TO_NUS @ r_carla @ g.CARLA_TO_NUS
            np.testing.assert_allclose(back, g.quat_to_matrix(g.quat_normalize(q)), atol=1e-12)


class TestEgoOriginShift:
    """★ **两套系的原点不同**(2026-09-23,Plan2.md §P-M.10)。

    nuScenes ego 原点 = **后轴中心**(地面),CARLA 车辆 actor 原点 = **车身长度中点**。
    两套 `calibrated_sensor` / `ego_pose` / 渲染挂点必须都在**同一套**里,否则
    `ego_pose ⊕ calibrated_sensor` 与世界系真值差 1.2563 m —— 而**逐传感器的判据
    (实挂 vs 声明)**对此结构性失明:错得整齐,两边一致地错。

    本类只钉**换算本身**;`NUS_EGO_ORIGIN_X` 的实测来源见 `geometry.py` 的文档串
    (前轴 +1.2502 / 后轴 −1.2563,四轮 C 值散布 2e-06 m)。
    """

    def test_constant_is_the_measured_rear_axle(self):
        """常量 = 后轴 x(CARLA 车体系),且**不是**前后轴中点(中点会落在 +0.0031 附近 → 实为另一回事)。"""
        assert g.NUS_EGO_ORIGIN_X == pytest.approx(-1.2563, abs=1e-9)
        assert g.NUS_EGO_ORIGIN_X < -1.0  # 后轴在车体中心之后
        assert abs(g.NUS_EGO_ORIGIN_X) != pytest.approx(1.8527, abs=0.5)  # 不是车尾包围盒

    def test_mount_shift_is_the_only_difference_from_the_y_flip(self):
        """`nus_mount_to_carla` == 「y 翻号 + x 平移」,没有第二个隐藏改动的余量。"""
        rng = [(3.412, 0.0, 0.5), (2.422, 0.8, 0.78), (-0.562, -0.618, 0.53), (0.0, 0.0, 0.0)]
        for t in rng:
            assert g.nus_mount_to_carla(t) == pytest.approx(
                (t[0] + g.NUS_EGO_ORIGIN_X, -t[1], t[2]), abs=1e-12
            ), t

    def test_actor_origin_maps_back_to_the_rear_axle(self):
        """CARLA actor 原点 → nus ego 原点:yaw 任意,平移量恒为 |`NUS_EGO_ORIGIN_X`|。"""
        loc = (10.0, -4.0, 0.5)
        for yaw_deg in (0.0, 90.0, 180.0, -90.0, 37.5):
            p = g.carla_actor_origin_to_nus_ego(loc, (0.0, math.radians(yaw_deg), 0.0))
            # 位移只在车头方向,大小恒等于 |origin_x|
            d = math.hypot(p[0] - loc[0], p[1] - loc[1])
            assert d == pytest.approx(abs(g.NUS_EGO_ORIGIN_X), abs=1e-12), yaw_deg
            # yaw=0(车头朝 +x)时后轴纯粹在 −x 侧
        assert g.carla_actor_origin_to_nus_ego(loc, (0.0, 0.0, 0.0))[0] == pytest.approx(
            loc[0] + g.NUS_EGO_ORIGIN_X, abs=1e-12
        )

    def test_ego_translation_is_the_rear_axle_in_the_nus_world_frame(self):
        """`nus_ego_translation` = 后轴点过 `CARLA_TO_NUS`(先车体系平移,再全局基变换)。"""
        loc = (100.0, 200.0, 0.0)
        got = g.nus_ego_translation(loc, (0.0, 0.0, 0.0))
        want = g.carla_to_nus_global(np.array([[loc[0] + g.NUS_EGO_ORIGIN_X, loc[1], loc[2]]]))[0]
        np.testing.assert_allclose(got, want, atol=1e-12)
        assert got[0] == pytest.approx(loc[0] + g.NUS_EGO_ORIGIN_X, abs=1e-12)

    def test_origin_x_is_injectable_so_criteria_can_use_a_measured_value(self):
        """`origin_x` 可注入 = 验收判据能拿**实测**后轴位置去复算(不是拿常量自证常量)。"""
        loc = (1.0, 2.0, 3.0)
        assert g.nus_ego_translation(loc, (0.0, 0.0, 0.0), origin_x=-2.0)[0] == pytest.approx(-1.0, abs=1e-12)
        assert g.carla_actor_origin_to_nus_ego(loc, (0.0, 0.0, 0.0), origin_x=0.0) == pytest.approx(
            loc, abs=1e-12
        )

    def test_round_trip_through_the_declared_tables(self):
        """声明表 → 渲染挂点 → 回推声明表:12 路传感器逐位闭合(1e-12)。"""
        for t_nus, _ in (*(ne.NUS_CAMERA_CALIBS[c] for c in ne.NUS_CAMERAS), ne.NUS_LIDAR_CALIB):
            m = g.nus_mount_to_carla(t_nus)
            assert (m[0] - g.NUS_EGO_ORIGIN_X, -m[1], m[2]) == pytest.approx(t_nus, abs=1e-12)
        for t_nus, _ in ne.NUS_RADAR_OFFSETS.values():
            m = g.nus_mount_to_carla(t_nus)
            assert (m[0] - g.NUS_EGO_ORIGIN_X, -m[1], m[2]) == pytest.approx(t_nus, abs=1e-12)

    def test_ego_rotation_is_the_matrix_conjugate_not_yaw_only(self):
        """★ `nus_ego_rotation` == `M·R_carla·M` 的四元数形式(**含 pitch/roll**)。

        判据是**矩阵相等**(四元数符号 ±q 同义,比向量会假挂),1e-12。
        实测 0.0642° 的悬架俯仰必须真的出现在结果里 —— 纯 yaw 实现下第二条断言会挂。
        """
        m = g.CARLA_TO_NUS
        for deg in ((0.0, 0.0, 0.0), (0.0642, 0.1592, -0.0008), (2.0, 90.0, -3.0), (-1.5, -170.0, 0.5)):
            rot = (math.radians(deg[0]), math.radians(deg[1]), math.radians(deg[2]))
            got = g.quat_to_matrix(g.nus_ego_rotation(rot))
            np.testing.assert_allclose(got, m @ g.carla_rotation_matrix(rot) @ m, atol=1e-12)
        # 悬架俯仰真的在:纯 yaw 的结果与它差 0.0642°
        pitch_only = (math.radians(0.0642), math.radians(0.1592), math.radians(-0.0008))
        yaw_only = (0.0, math.radians(0.1592), 0.0)
        a = g.quat_to_matrix(g.nus_ego_rotation(pitch_only))
        b = g.quat_to_matrix(g.nus_ego_rotation(yaw_only))
        cos_t = (np.trace(a.T @ b) - 1.0) / 2.0
        assert math.degrees(math.acos(float(np.clip(cos_t, -1.0, 1.0)))) == pytest.approx(0.0642, abs=1e-3)

    def test_ego_rotation_pure_yaw_matches_the_legacy_yaw_helper(self):
        """纯 yaw 时与 `carla_yaw_to_nus_quat` 等价 —— 新旧两条路在退化情形必须给出同一个姿态。"""
        for deg in (0.0, 30.0, -120.0, 179.0):
            rot = (0.0, math.radians(deg), 0.0)
            np.testing.assert_allclose(
                g.quat_to_matrix(g.nus_ego_rotation(rot)),
                g.quat_to_matrix(g.carla_yaw_to_nus_quat(rot[1])),
                atol=1e-15,
            )


class TestCameraLocalAxisReorder:
    """★ 相机**局部基**的重排(§P-M.10):CARLA (x 前/y 右/z 上) vs nus (x 右/y 下/z 前)。

    拿 LiDAR/雷达那条 `M·R·M` 去比相机姿态会得到**恒为 120°** 的假误差(轮换阵本征角)——
    判据⑨ 第一次跑正是六路相机齐刷刷 119.93–120.07°。本类把这个重排钉死。
    """

    def test_reorder_is_orthogonal_and_flips_handedness(self):
        """`C` 正交但 **det = −1**(不是旋转)—— 因为 CARLA 相机局部基是**左手**(x 前/y 右/z 上),
        nus 相机局部基是右手(x 右/y 下/z 前)。局部基翻转与全局基翻转(`M`,det 也 = −1)
        **相乘才抵消**:`A = M·R·C` ⇒ det = (−1)·1·(−1) = +1,仍是正当旋转。
        谁要是"顺手把 C 修正成正旋转",判据⑨ 立刻回到那个 120° 的假误差。
        """
        c = g.CARLA_CAM_TO_NUS_CAM
        np.testing.assert_allclose(c @ c.T, np.eye(3), atol=1e-15)
        assert np.linalg.det(c) == pytest.approx(-1.0, abs=1e-15)
        assert np.linalg.det(g.CARLA_TO_NUS @ g.carla_rotation_matrix((0.1, 0.2, 0.3)) @ c) == pytest.approx(
            1.0, abs=1e-12
        )

    def test_columns_are_right_down_forward(self):
        """逐列语义:`C` 的第 0/1/2 列 = nuScenes 相机的「右 / 下 / 前」在 **CARLA 相机自身**系里。

        CARLA 局部轴 (前, 右, 上) = (e0, e1, e2) ⇒ `C = [e1, −e2, e0]`
        ⇒ 作用在局部轴上是 `e0 ↦ e1`(前 → 右? 不,读法见下)、`e1 ↦ −e2`、`e2 ↦ e0`。
        """
        c = g.CARLA_CAM_TO_NUS_CAM
        np.testing.assert_allclose(c, np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]]))
        np.testing.assert_allclose(c[:, 0], [0.0, 1.0, 0.0])  # 第 0 列 = 「右」= carla 的 y
        np.testing.assert_allclose(c[:, 1], [0.0, 0.0, -1.0])  # 第 1 列 = 「下」= −carla 的 z
        np.testing.assert_allclose(c[:, 2], [1.0, 0.0, 0.0])  # 第 2 列 = 「前」= carla 的 x
        # 等价说法:nuScenes 相机系里的"前方"(+z)在 CARLA 相机系里就是 +x —— 两套系的视线轴同向
        np.testing.assert_allclose(c @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0])

    def test_it_explains_the_120_degree_offset(self):
        """**假误差的来源**:不加这个重排时,同一姿态下 `M·R·M` 与真值差恰好 120°。

        用一个"相机水平朝前"(pitch=roll=0)的 CARLA 姿态验:此时 `R_carla = Rz(yaw)`,
        `M·R·M` 与 `M·R·C` 之间只差 `Rᵀ·C` 这个**常数**轮换阵,本征角 = 120°。
        """
        rot = (0.0, math.radians(35.0), 0.0)
        m = g.CARLA_TO_NUS
        r_ue = g.carla_rotation_matrix(rot)
        wrong = m @ r_ue @ m
        right = m @ r_ue @ g.CARLA_CAM_TO_NUS_CAM
        cos_t = (np.trace(wrong.T @ right) - 1.0) / 2.0
        assert math.degrees(math.acos(float(np.clip(cos_t, -1.0, 1.0)))) == pytest.approx(120.0, abs=0.02)

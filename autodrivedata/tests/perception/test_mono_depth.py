"""P-D 单目测距手算锚点单测(ground_intersection 上移 + 迭代深度法)。

风格沿用仓库:手算锚点 + 边界,类按被测函数分组,固定 rng seed。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.calib.core import CameraIntrinsics
from autodrivedata.perception.mono_depth import box_to_ground_distance, ground_plane_distance
from autodrivedata.utils.geometry import ground_intersection, mono_depth_from_box

_K = CameraIntrinsics(1242, 375, 90)


class TestGroundIntersection:
    def test_identity_pose_hand_anchor(self):
        # 恒等位姿:相机在原点无俯仰。图像中心水平射线与 z 平面平行(无交点),
        # 但**中心下方**像素 (v > cy) 的射线向下倾斜会打到地面。
        # 手算锚点:v=300 → yn=(300-187)/621=0.18197;dir_world = R_cw@[0,yn,1],
        #   R_cw = CARLA_TO_CAMᵀ = [[0,0,1],[1,0,0],[0,-1,0]],@ [0,yn,1] = (1, 0, -yn)
        #   从 (0,0,0) 出发,与 z=-0.5: t = 0.5·? dir_world 的 z 分量 -yn,需 -0.5/(-yn)=2.7478
        #   ⇒ 交点 x = 2.7478(手算:0.5 / 0.18197)。精确断言。
        g = ground_intersection(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), _K, _K.cx, 300.0, ground_z=-0.5)
        assert g == pytest.approx((2.7478, 0.0), rel=1e-3, abs=1e-3)
        # 中心像素射线平行地面 → None
        assert (
            ground_intersection(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), _K, _K.cx, _K.cy, ground_z=-0.5) is None
        )

    def test_pitch_down_hand_anchor(self):
        # 相机俯仰 -10°(弧度)看向地面,中心像素射线应打在自己正前方路面。
        # 手算:image 中心 (cx,cy) → 相机系 (0,0,1);反投影世界系方向 = R_cw·(0,0,1)。
        #   R_cw = R_cam_worldᵀ = [[0,-0.1736,0.9848],[1,0,0],[0,-0.9848,-0.1736]]
        #   第3列 = (0.9848, 0, -0.1736) → 射线朝世界系 +x/下(z 负向)。
        #   从 (0,0,2) 出发,与 z=0: t = (0-2)/(-0.1736) ≈ 11.52 → 交点 x = 11.3426。
        g = ground_intersection(
            ((0.0, 0.0, 2.0), (np.radians(-10.0), 0.0, 0.0)),
            _K,
            _K.cx,
            _K.cy,
            ground_z=0.0,
        )
        assert g is not None
        assert g[0] == pytest.approx(11.3426, rel=1e-3)
        assert g[1] == pytest.approx(0.0, abs=1e-6)
        # 往返一致:交点经 world_to_cam 回到相机系,应位于图像中心射线(相机系 x≈0,y≈0)
        from autodrivedata.utils.geometry import world_to_cam

        c = world_to_cam(
            np.asarray([[g[0], g[1], 0.0]]),
            (0.0, 0.0, 2.0),
            (np.radians(-10.0), 0.0, 0.0),
        )[0]
        assert abs(c[0]) < 1e-6 and abs(c[1]) < 1e-6 and c[2] > 0

    def test_ray_up_returns_none(self):
        # 相机 pitch=+10°(抬头),射线不可能打地 → None
        g = ground_intersection(
            ((0.0, 0.0, 2.0), (np.radians(10.0), 0.0, 0.0)),
            _K,
            _K.cx,
            _K.cy,
            ground_z=0.0,
        )
        assert g is None

    def test_camera_below_ground(self):
        # 相机 z < ground_z:射线即使向下也在平面下,返回 None
        g = ground_intersection(
            ((0.0, 0.0, -1.0), (np.radians(-10.0), 0.0, 0.0)),
            _K,
            _K.cx,
            _K.cy,
            ground_z=0.0,
        )
        assert g is None


class TestMonoDepthFromBox:
    def test_hand_anchor(self):
        # z = H·fy / h:车高 1.5m、框高 100px、fy=621 → 9.315m
        assert mono_depth_from_box(100.0, 1.5, _K.fy) == pytest.approx(9.315, rel=1e-3)

    def test_distance_half_frame_doubles_depth(self):
        # 框高减半 → 深度翻倍(反比)
        z1 = mono_depth_from_box(100.0, 1.5, _K.fy)
        z2 = mono_depth_from_box(50.0, 1.5, _K.fy)
        assert z2 == pytest.approx(2 * z1)

    def test_invalid_returns_inf(self):
        assert mono_depth_from_box(0.0, 1.5, _K.fy) == float("inf")
        assert mono_depth_from_box(-10.0, 1.5, _K.fy) == float("inf")
        assert mono_depth_from_box(100.0, 0.0, _K.fy) == float("inf")
        assert mono_depth_from_box(100.0, 1.5, 0.0) == float("inf")


class TestGroundPlaneDistance:
    def test_forward_component(self):
        # 交点在世界系正前方 30m(y=0):前向距离 = 30
        d = ground_plane_distance(0.0, 0.0, 30.0, 0.0, cam_yaw_rad=0.0)
        assert d == pytest.approx(30.0, abs=1e-9)

    def test_yaw_rotates(self):
        # yaw=90°:世界系 +x 方向交点投影到 ego 局部 x 为 0,全在 y(侧向)
        d = ground_plane_distance(0.0, 0.0, 30.0, 0.0, cam_yaw_rad=np.pi / 2)
        assert d == pytest.approx(0.0, abs=1e-9)


class TestBoxToGroundDistance:
    def test_roundtrip_consistent(self):
        # 恒等位姿相机在 (0,0,1.6),框底像素在图像中心 → 无交点(射线水平)
        d = box_to_ground_distance(_K, (0.0, 0.0, 1.6), (0.0, 0.0, 0.0), 0.0, (_K.cx, _K.cy), 0.0)
        assert d is None

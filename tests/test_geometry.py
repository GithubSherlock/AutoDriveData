"""geometry.py 手算锚点单测(纯 numpy,可在任意 env 跑)。

期望值全部手推自轴系定义,锚定语义——不照抄实现公式(防同错)。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata import geometry as g

CAM0 = (0.0, 0.0, 0.0)  # pitch, yaw, roll(弧度)


class TestBasisMatrices:
    def test_carla_to_cam(self):
        # CARLA (x 前, y 右, z 上) → KITTI 相机 (x 右, y 下, z 前)
        v = np.array([3.0, 4.0, 5.0])
        np.testing.assert_allclose(g.CARLA_TO_CAM @ v, [4.0, -5.0, 3.0], atol=1e-12)

    def test_velo_to_cam(self):
        # KITTI velodyne (x 前, y 左, z 上) → 相机系:前→前、左→右、上→下
        v = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(g.VELO_TO_CAM @ v, [-2.0, -3.0, 1.0], atol=1e-12)

    def test_carla_sensor_to_velo_y_flip(self):
        v = np.array([[1.0, 2.0, 3.0, 0.5]])
        np.testing.assert_allclose(g.carla_lidar_to_velodyne(v), [[1.0, -2.0, 3.0, 0.5]])

    def test_basis_matrices_orthogonal(self):
        # CARLA_TO_CAM 含手性翻转(det=−1)但保持正交
        for m in (g.CARLA_TO_CAM, g.VELO_TO_CAM):
            np.testing.assert_allclose(m.T @ m, np.eye(3), atol=1e-12)


class TestCarlaRotationMatrix:
    def test_identity(self):
        np.testing.assert_allclose(g.carla_rotation_matrix(CAM0), np.eye(3), atol=1e-12)

    def test_yaw_90(self):
        # yaw=+90°:x̂(前)→ ŷ(右),即俯视逆时针左转
        np.testing.assert_allclose(
            g.carla_rotation_matrix((0.0, np.pi / 2, 0.0)),
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            atol=1e-12,
        )

    def test_pitch_90_nose_up(self):
        # pitch=+90°:x̂(前)→ ẑ(上)——正 pitch 抬头(实测锁定)
        np.testing.assert_allclose(
            g.carla_rotation_matrix((np.pi / 2, 0.0, 0.0)),
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            atol=1e-12,
        )

    def test_roll_90(self):
        # roll=+90°:ŷ(右)→ ẑ(上)、ẑ → −ŷ
        np.testing.assert_allclose(
            g.carla_rotation_matrix((0.0, 0.0, np.pi / 2)),
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            atol=1e-12,
        )

    def test_composed_30_45_15(self):
        # (pitch 30°, yaw 45°, roll 15°)——与 pycarla 实测矩阵一致(探针锁定)
        m = g.carla_rotation_matrix((np.pi / 6, np.pi / 4, np.pi / 12))
        expected = np.array([[0.612, -0.592, -0.525], [0.612, 0.775, -0.158], [0.5, -0.224, 0.837]])
        np.testing.assert_allclose(m, expected, atol=1e-2)

    def test_det_plus_one(self):
        # 真旋转:任何角组合 det = +1
        m = g.carla_rotation_matrix((0.3, -1.1, 0.7))
        assert abs(np.linalg.det(m) - 1.0) < 1e-9


class TestRotationMatrixToCarla:
    """逆分解回归锚点:曾把 pitch 写成 `asin(−R[2,0])`(静默反号,−12° → +12°)。"""

    @pytest.mark.parametrize(
        "deg",
        [(0, 0, 0), (0, 90, 0), (-12, 0, 0), (5, -55, 3), (30, 45, 15), (-8.5, 179.3, 1.2)],
    )
    def test_round_trip_recovers_angles(self, deg):
        pitch, yaw, roll = (np.radians(d) for d in deg)
        assert g.rotation_matrix_to_carla(g.carla_rotation_matrix((pitch, yaw, roll))) == pytest.approx(
            (pitch, yaw, roll), abs=1e-12
        )

    def test_negative_pitch_keeps_sign(self):
        # 俯角 −12° 必须解出 −12°(写成 asin(−R[2,0]) 会得 +12°,第三方视角变仰视)
        p, _, _ = g.rotation_matrix_to_carla(g.carla_rotation_matrix((np.radians(-12.0), 0.0, 0.0)))
        assert np.degrees(p) == pytest.approx(-12.0, abs=1e-9)

    def test_random_round_trip_elementwise(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            ang = rng.uniform(-np.pi, np.pi, 3)
            m = g.carla_rotation_matrix(tuple(ang))
            back = g.carla_rotation_matrix(g.rotation_matrix_to_carla(m))
            assert np.abs(back - m).max() < 1e-12


class TestWorldToCam:
    def test_cam_at_origin_looking_forward(self):
        # 相机在 (0,0,1.65) 朝 +x_g:等高点 (10,0,1.65) → 相机系 (0,0,10)(正前 10m,无高度差)
        pts = g.world_to_cam(np.array([[10.0, 0.0, 1.65]]), (0.0, 0.0, 1.65), CAM0)
        np.testing.assert_allclose(pts, [[0.0, 0.0, 10.0]], atol=1e-12)

    def test_height_maps_to_y_down(self):
        # 相机原点:上方 1.65m 的点 → y_k = −1.65(相机 y 向下)
        pts = g.world_to_cam(np.array([[5.0, 0.0, 1.65]]), (0.0, 0.0, 0.0), CAM0)
        np.testing.assert_allclose(pts, [[0.0, -1.65, 5.0]], atol=1e-12)

    def test_cam_rotated_yaw_90(self):
        # 相机 yaw=+90°(朝 +y_g):世界点 (0,5,0) 在相机正前 → (0,0,5)
        r = g.camera_rotation_world_to_cam((0.0, np.pi / 2, 0.0))
        np.testing.assert_allclose(r, [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], atol=1e-12)
        pts = g.world_to_cam(np.array([[0.0, 5.0, 0.0]]), (0.0, 0.0, 0.0), (0.0, np.pi / 2, 0.0))
        np.testing.assert_allclose(pts, [[0.0, 0.0, 5.0]], atol=1e-12)

    def test_rotation_orthogonal(self):
        r = g.camera_rotation_world_to_cam((0.3, 1.1, -0.2))
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)


class TestRotationY:
    """KITTI rotation_y 语义:ry=0 车头 +x(右);车头向量 (cos ry, −sin ry)。"""

    def test_actor_facing_forward_cam0(self):
        # actor 朝 +x_g、相机 yaw=0:车头在相机系 = +z(远离相机)→ ry = −π/2
        ry = g.actor_yaw_to_rotation_y(0.0, CAM0)
        assert ry == pytest.approx(-np.pi / 2)

    def test_actor_facing_right_cam0(self):
        # actor 朝 +y_g(相机右)→ ry = 0
        ry = g.actor_yaw_to_rotation_y(np.pi / 2, CAM0)
        assert ry == pytest.approx(0.0)

    def test_actor_facing_camera(self):
        # actor 朝 −x_g(面向相机)→ 相机系车头 (0,0,−1) → ry = +π/2
        ry = g.actor_yaw_to_rotation_y(np.pi, CAM0)
        assert ry == pytest.approx(np.pi / 2)

    def test_actor_facing_left_cam_yaw90(self):
        # 相机 yaw=+90°(朝 +y_g);actor 朝 +x_g = 相机左手边 → ry = ±π(此处 wrap 给 −π)
        ry = g.actor_yaw_to_rotation_y(0.0, (0.0, np.pi / 2, 0.0))
        assert ry == pytest.approx(-np.pi)

    def test_yaw_bev_round_trip(self):
        # yaw_bev 只经 sin/cos 进入 → 往返等价模 2π(±π 边界有跳变,属同义)
        for yb in [-np.pi, -np.pi / 2, 0.0, 1.0, np.pi / 2, np.pi - 0.01]:
            rt = g.rotation_y_to_yaw_bev(g.yaw_bev_to_rotation_y(yb))
            diff = abs((rt - yb + np.pi) % (2 * np.pi) - np.pi)
            assert diff < 1e-12

    def test_wrap_pi_semantics(self):
        # 照抄 auto3dlabel:arctan2(sin, cos);数值上 wrap_pi(−π) = −π
        assert g.wrap_pi(-np.pi) == pytest.approx(-np.pi)
        assert g.wrap_pi(3 * np.pi / 2) == pytest.approx(-np.pi / 2)
        assert g.wrap_pi(0.0) == pytest.approx(0.0)


class TestCorners:
    def test_facing_forward(self):
        # 底心 (0,−1.65,10),h=1.5,w=1.8,l=4,ry=−π/2(车头 +z 远离相机)
        c = g.corners_cam_from_bottom(0.0, -1.65, 10.0, 1.5, 1.8, 4.0, -np.pi / 2)
        assert c.shape == (8, 3)
        # 底 4:前右/前左/后左/后右(车头 +z,右 = −x)
        np.testing.assert_allclose(
            c[:4],
            [[-0.9, -1.65, 12.0], [0.9, -1.65, 12.0], [0.9, -1.65, 8.0], [-0.9, -1.65, 8.0]],
            atol=1e-12,
        )
        # 顶 4:y − h(相机 y 向下,顶面更小)
        np.testing.assert_allclose(c[4:, 1], -3.15, atol=1e-12)

    def test_facing_right(self):
        # ry=0(车头 +x):前 = +x,右 = +z
        c = g.corners_cam_from_bottom(0.0, -1.65, 10.0, 1.5, 1.8, 4.0, 0.0)
        np.testing.assert_allclose(
            c[:4],
            [[2.0, -1.65, 10.9], [2.0, -1.65, 9.1], [-2.0, -1.65, 9.1], [-2.0, -1.65, 10.9]],
            atol=1e-12,
        )

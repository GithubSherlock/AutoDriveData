"""geometry.py vs pycarla oracle 对照(base env,carla 不可用时跳过)。

实测锁定 CARLA 旋转矩阵组合顺序与语义——本文件即"实测标定"的落点
(Plan §3.2 要求 ego 外参 ↔ CARLA get_transform() 换算以实测为准)。
"""

from __future__ import annotations

import numpy as np
import pytest

carla = pytest.importorskip("carla")

from autodrivedata import geometry as g  # noqa: E402

# (pitch, yaw, roll) 度——覆盖单轴、双轴与一般姿态
CASES_DEG = [
    (0, 0, 0),
    (0, 90, 0),
    (-15, 0, 0),
    (90, 0, 0),
    (0, 0, 90),
    (30, 45, 15),
    (-25, 123, -8),
    (12.5, -200.5, 33.3),
]


def _to_rad(deg: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(np.radians(a) for a in deg)


class TestAgainstPycarla:
    @pytest.mark.parametrize("rot_deg", CASES_DEG)
    def test_rotation_matrix_matches(self, rot_deg):
        ours = g.carla_rotation_matrix(_to_rad(rot_deg))
        theirs = np.asarray(carla.Transform(carla.Location(), carla.Rotation(*rot_deg)).get_matrix())[:3, :3]
        np.testing.assert_allclose(ours, theirs, atol=1e-6)

    @pytest.mark.parametrize("rot_deg", CASES_DEG)
    def test_forward_vector_matches(self, rot_deg):
        # 车头 = 旋转阵第一列;get_forward_vector 为 oracle
        fwd = carla.Transform(carla.Location(), carla.Rotation(*rot_deg)).get_forward_vector()
        expected = np.array([fwd.x, fwd.y, fwd.z])
        ours = g.carla_rotation_matrix(_to_rad(rot_deg))[:, 0]
        np.testing.assert_allclose(ours, expected, atol=1e-6)

    def test_cam_pose_consistency(self):
        """随机相机位姿:用我们的链 vs 用 pycarla 矩阵链变换同一批点,结果一致。"""
        rng = np.random.default_rng(42)
        for _ in range(5):
            rot_deg = tuple(rng.uniform(-180, 180, 3).tolist())
            loc = rng.uniform(-50, 50, 3)
            pts_world = rng.uniform(-100, 100, (20, 3))
            # 我们的路径
            ours = g.world_to_cam(pts_world, loc, _to_rad(rot_deg))
            # pycarla 路径:T = get_matrix;world→carla 局部 = T⁻¹ @ p;再 CARLA→KITTI 相机
            # (pycarla float32,坐标 ~100m 时误差 ~1e−5,atol 相应放宽)
            t = np.asarray(carla.Transform(carla.Location(*loc), carla.Rotation(*rot_deg)).get_matrix())
            local = (np.linalg.inv(t) @ np.hstack([pts_world, np.ones((20, 1))]).T).T[:, :3]
            theirs = (g.CARLA_TO_CAM @ local.T).T
            np.testing.assert_allclose(ours, theirs, atol=1e-4)

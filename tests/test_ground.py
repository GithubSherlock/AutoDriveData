"""autodrivedata/ground.py 手算锚点单测。"""

from __future__ import annotations

import numpy as np

from autodrivedata.ground import grid_ground, ground_stats, plane_angle_deg, ransac_plane


def _ground_frame(n=400):
    """典型地面:z = 0 平面 + 少量随机散点(障碍)。"""
    rng = np.random.default_rng(0)
    x = rng.uniform(-20, 20, n)
    y = rng.uniform(-20, 20, n)
    z = np.zeros(n)
    obs = np.array([[1.0, 1.0, 1.5], [2.0, 0.0, 1.0], [-1.0, 2.0, 1.8], [0.5, -1.0, 1.2]])
    pt = np.vstack([np.stack([x, y, z], axis=1), obs])
    return np.hstack([pt, np.ones((pt.shape[0], 1), dtype=np.float32) * 0.5])


class TestRansacPlane:
    def test_horizontal_plane(self):
        pts = _ground_frame()
        res = ransac_plane(pts, seed=1)
        assert res is not None
        plane, inl = res
        # 平面近似水平
        assert plane_angle_deg(plane) < 2.0
        # 障碍点不在内点中(目标 1.5m 高 → 差 > 0.25)
        assert not inl[-4:].any()

    def test_tilted_plane(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-10, 10, 200)
        y = rng.uniform(-10, 10, 200)
        z = 0.5 * x + 0.0 * y  # 平面 z=0.5x
        pts = np.stack([x, y, z], axis=1)
        plane, inl = ransac_plane(pts, seed=0)
        np.testing.assert_allclose(plane[:2], [0.5, 0.0], atol=0.1)
        assert inl.mean() > 0.9

    def test_no_plane(self):
        rng = np.random.default_rng(0)
        pts = np.stack(
            [rng.uniform(-5, 5, 100), rng.uniform(-5, 5, 100), rng.uniform(0, 10, 100)],
            axis=1,
        )
        assert ransac_plane(pts, seed=0) is None or ransac_plane(pts, seed=0)[1].mean() < 0.3


class TestGridGround:
    def test_ground_removed(self):
        pts = _ground_frame()
        mask = grid_ground(pts, cell=10.0)  # ±20 → 4×4 格,每格 ~25 点
        # 障碍点(高 1.0-1.8)不在地面
        assert not mask[-4:].any()
        # 地面点大部分保留
        assert mask[:-4].mean() > 0.95

    def test_empty(self):
        assert grid_ground(np.zeros((0, 4))).shape == (0,)


class TestGroundStats:
    def test_counts(self):
        pts = _ground_frame()
        st = ground_stats(pts, np.zeros(len(pts), dtype=bool))
        assert st["n_points"] == len(pts)
        assert st["n_ground"] == 0
        assert st["ground_ratio"] == 0.0


class TestPlaneAngle:
    def test_horizontal(self):
        assert plane_angle_deg(np.array([0.0, 0.0, 0.0])) < 1e-3

    def test_slope(self):
        assert plane_angle_deg(np.array([1.0, 0.0, 0.0])) > 40.0

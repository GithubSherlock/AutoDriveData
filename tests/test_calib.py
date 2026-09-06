"""calib.py 手算锚点单测(纯 numpy)。"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata import calib
from autodrivedata import geometry as g

CAM0 = (0.0, 0.0, 0.0)


class TestIntrinsics:
    def test_kitti_style_1242x375_fov90(self):
        # fx = 621/tan(45°) = 621;cx=(W−1)/2、cy=(H−1)/2
        k = calib.CameraIntrinsics(width=1242, height=375, fov_h_deg=90.0)
        assert k.fx == pytest.approx(621.0)
        assert k.fy == pytest.approx(621.0)  # 方形像素
        assert k.cx == pytest.approx(620.5)
        assert k.cy == pytest.approx(187.0)

    def test_p2_shape_and_entries(self):
        k = calib.CameraIntrinsics(width=1242, height=375, fov_h_deg=90.0)
        p2 = k.p2()
        assert p2.shape == (3, 4)
        np.testing.assert_allclose(
            p2,
            [[621.0, 0.0, 620.5, 0.0], [0.0, 621.0, 187.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            atol=1e-9,
        )


class TestTrVeloToCam:
    def test_identity_pose_degenerates_to_velo_to_cam(self):
        # LiDAR 与相机同位同向 → Tr 退化为 VELO_TO_CAM 基变换(geometry 手算锚点)
        tr = calib.tr_velo_to_cam((0.0, 0.0, 0.0), CAM0, (0.0, 0.0, 0.0), CAM0)
        np.testing.assert_allclose(tr[:, :3], g.VELO_TO_CAM, atol=1e-12)
        np.testing.assert_allclose(tr[:, 3], [0.0, 0.0, 0.0], atol=1e-12)

    def test_lidar_1m_ahead(self):
        # LiDAR 在相机前 1m(世界 +x):t = (0,0,1)(相机系前方)
        tr = calib.tr_velo_to_cam((1.0, 0.0, 0.0), CAM0, (0.0, 0.0, 0.0), CAM0)
        np.testing.assert_allclose(tr[:, :3], g.VELO_TO_CAM, atol=1e-12)
        np.testing.assert_allclose(tr[:, 3], [0.0, 0.0, 1.0], atol=1e-12)

    def test_point_projection_semantics(self):
        # LiDAR 前 1m、相机原点:velodyne 点 (2,0,0)(lidar 前 2m)→ 相机系 (0,0,3)
        tr = calib.tr_velo_to_cam((1.0, 0.0, 0.0), CAM0, (0.0, 0.0, 0.0), CAM0)
        p_v = np.array([2.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(tr @ p_v, [0.0, 0.0, 3.0], atol=1e-12)

    def test_roof_mount_same_pose(self):
        # 典型车顶安装:相机/LiDAR 同位 (0,0,1.65),t = 0
        tr = calib.tr_velo_to_cam((0.0, 0.0, 1.65), CAM0, (0.0, 0.0, 1.65), CAM0)
        np.testing.assert_allclose(tr[:, 3], [0.0, 0.0, 0.0], atol=1e-12)

    def test_camera_yaw90_rotation(self):
        # 相机 yaw=+90°(朝 +y_g),LiDAR 同向:车顶前向点 → 相机系正前
        tr = calib.tr_velo_to_cam((0.0, 0.0, 0.0), (0.0, np.pi / 2, 0.0), (0.0, 0.0, 0.0), (0.0, np.pi / 2, 0.0))
        p_v = np.array([3.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(tr @ p_v, [0.0, 0.0, 3.0], atol=1e-12)


class TestCalibText:
    def test_exact_text(self):
        k = calib.CameraIntrinsics(width=1242, height=375, fov_h_deg=90.0)
        out = calib.KittiCalibOut(p2=k.p2(), tr_velo_to_cam=calib.tr_velo_to_cam(
            (0.0, 0.0, 0.0), CAM0, (0.0, 0.0, 0.0), CAM0
        ))
        text = out.to_text()
        lines = text.splitlines()
        assert len(lines) == 7
        assert lines[0] == f"P0: {' '.join('0.000000e+00' for _ in range(12))}"
        assert lines[2].startswith("P2: 6.210000e+02 0.000000e+00 6.205000e+02")
        assert lines[4] == "R0_rect: 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00"
        # Tr_velo_to_cam = VELO_TO_CAM(恒等位姿)
        assert lines[5] == (
            "Tr_velo_to_cam: 0.000000e+00 -1.000000e+00 0.000000e+00 0.000000e+00 "
            "0.000000e+00 0.000000e+00 -1.000000e+00 0.000000e+00 "
            "1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00"
        )

    def test_write_creates_file(self, tmp_path):
        k = calib.CameraIntrinsics(width=1242, height=375, fov_h_deg=90.0)
        out = calib.KittiCalibOut(p2=k.p2(), tr_velo_to_cam=calib.tr_velo_to_cam(
            (0.0, 0.0, 0.0), CAM0, (0.0, 0.0, 0.0), CAM0
        ))
        p = out.write(tmp_path / "calib" / "000000.txt")
        assert p.is_file()
        assert p.read_text().splitlines()[2].startswith("P2:")

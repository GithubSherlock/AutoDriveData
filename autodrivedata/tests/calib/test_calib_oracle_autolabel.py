"""calib.py vs auto3dlabel KittiCalib oracle(autolabel env 跑;base env 自动跳过)。

M1a-7 集成验收的前置:标定文本必须被 KittiCalib.from_file 零改动解析,
且投影链(velodyne → cam → 像素)与 auto3dlabel 逐点一致。
"""

from __future__ import annotations

import numpy as np
import pytest

auto3dlabel = pytest.importorskip("auto3dlabel.schema.calib")

from auto3dlabel.schema.calib import KittiCalib  # noqa: E402

from autodrivedata import geometry as g  # noqa: E402
from autodrivedata.calib import core as calib  # noqa: E402

CAM0 = (0.0, 0.0, 0.0)


def _sample_out() -> calib.KittiCalibOut:
    k = calib.CameraIntrinsics(width=1242, height=375, fov_h_deg=90.0)
    return calib.KittiCalibOut(
        p2=k.p2(),
        tr_velo_to_cam=calib.tr_velo_to_cam(
            (0.0, 0.0, 1.65), (0.0, 0.3, 0.0), (0.0, 0.0, 1.65), (0.0, 0.3, 0.0)
        ),
    )


class TestAgainstKittiCalib:
    def test_from_file_parses(self, tmp_path):
        p = _sample_out().write(tmp_path / "000000.txt")
        kc = KittiCalib.from_file(p)
        np.testing.assert_allclose(kc.P2, _sample_out().p2, atol=1e-12)
        np.testing.assert_allclose(kc.R0_rect, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(kc.Tr_velo_to_cam, _sample_out().tr_velo_to_cam, atol=1e-12)

    def test_projection_chain_matches(self, tmp_path):
        """同一批 velodyne 点:auto3dlabel 的 velo_to_cam + P2 投影 = 我们的 Tr + 手算投影。"""
        p = _sample_out().write(tmp_path / "000000.txt")
        kc = KittiCalib.from_file(p)
        rng = np.random.default_rng(7)
        pts_velo = rng.uniform([-30, -5, -3], [30, 5, 3], (200, 3))  # 相机前方区域
        # auto3dlabel 链
        u1, v1, valid1 = kc.project_velo_to_image(pts_velo, 1242, 375)
        # 我们的链:velo → cam(Tr 3x4)→ P2
        cam = (kc.Tr_velo_to_cam @ np.hstack([pts_velo, np.ones((200, 1))]).T).T
        img = (kc.P2 @ np.hstack([cam, np.ones((200, 1))]).T).T
        z = img[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u2 = np.where(z > 0, img[:, 0] / np.maximum(z, 1e-12), 0.0)
            v2 = np.where(z > 0, img[:, 1] / np.maximum(z, 1e-12), 0.0)
        valid2 = (z > 0) & (cam[:, 2] > 0) & (u2 >= 0) & (u2 < 1242) & (v2 >= 0) & (v2 < 375)
        np.testing.assert_array_equal(valid1, valid2)
        np.testing.assert_array_equal(u1[valid1], u2[valid2].astype(np.int32))
        np.testing.assert_array_equal(v1[valid1], v2[valid2].astype(np.int32))

    def test_geometry_wrap_matches_autolabel(self):
        """wrap_pi/ry 换算与 auto3dlabel 函数逐点一致(单一事实源对照)。"""
        from auto3dlabel.tools.geometry import wrap_pi as al_wrap
        from auto3dlabel.tools.geometry import yaw_to_rotation_y as al_ry

        for x in np.linspace(-4 * np.pi, 4 * np.pi, 1000):
            assert g.wrap_pi(float(x)) == pytest.approx(al_wrap(float(x)), abs=1e-12)
            assert g.yaw_bev_to_rotation_y(float(x)) == pytest.approx(al_ry(float(x)), abs=1e-12)

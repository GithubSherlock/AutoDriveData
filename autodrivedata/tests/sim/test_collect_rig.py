"""采集器纯函数手算锚点单测(collect_rig:环绕位姿 + 双目挂点)。

风格沿用仓库:手算锚点 + 边界,类按被测函数分组,探测 yaw/挂点语义。
"""

from __future__ import annotations

import pytest

from autodrivedata.sim.collect_rig import ring_cam_pose, stereo_rig_offsets


class TestRingCamPose:
    def test_i0_facing_center(self):
        # i=0:相机在 +x 侧(cx+r, cy),yaw 应该朝环绕中心 = atan2(0-cy, cx-x) =
        # atan2(0, cx-(cx+r)) = atan2(0, -r) = 180°。
        x, y, _, yaw = ring_cam_pose(0.0, 0.0, 0.0, 6.0, 0, 90)
        assert (x, y) == pytest.approx((6.0, 0.0))
        assert yaw == pytest.approx(180.0)

    def test_ihalf_faces_center_opposite(self):
        # i=n/2(绕到 -x 侧):相机在 (−r, 0),yaw 朝中心 = atan2(0-0, 0-(-6)) = 0°。
        x, y, _, yaw = ring_cam_pose(0.0, 0.0, 0.0, 6.0, 45, 90)
        assert (x, y) == pytest.approx((-6.0, 0.0))
        assert yaw == pytest.approx(0.0)

    def test_iquarter_side(self):
        # n_cams=4, i=1:相机在 (0, +r)(y+ 侧),yaw 朝中心 = atan2(-r, 0) = -90°。
        x, y, z, yaw = ring_cam_pose(0.0, 0.0, 0.0, 6.0, 1, 4)
        assert (x, y) == pytest.approx((0.0, 6.0))
        assert yaw == pytest.approx(-90.0)
        assert z == pytest.approx(1.5)

    def test_z_height_offset(self):
        _, _, z, _ = ring_cam_pose(1.0, 2.0, 3.0, 6.0, 10, 90, height=2.0)
        assert z == pytest.approx(5.0)  # 3.0 + 2.0

    def test_n_cams_zero_raises(self):
        with pytest.raises(ValueError):
            ring_cam_pose(0.0, 0.0, 0.0, 6.0, 0, 0)


class TestStereoRigOffsets:
    def test_half_baseline_sym(self):
        left, right = stereo_rig_offsets(0.4)
        assert left == pytest.approx((1.2, -0.2, 1.65))
        assert right == pytest.approx((1.2, +0.2, 1.65))

    def test_zero_baseline_raises(self):
        with pytest.raises(ValueError):
            stereo_rig_offsets(0.0)

    def test_large_baseline(self):
        left, right = stereo_rig_offsets(2.0)
        assert left == pytest.approx((1.2, -1.0, 1.65))
        assert right == pytest.approx((1.2, +1.0, 1.65))

"""采集器纯函数手算锚点单测(collect_rig:环绕位姿 + 双目挂点)。

风格沿用仓库:手算锚点 + 边界,类按被测函数分组,探测 yaw/挂点语义。
"""

from __future__ import annotations

import pytest

from autodrivedata.sim.collect_rig import ring_cam_pose, stereo_rig_offsets, to_parent_frame


class TestToParentFrame:
    """★ 世界位姿 → 相对挂载父 actor 的位姿(2026-10-07 修,见 §C.0.4 ③)。

    这一段漏掉的代价是**静默**的:相机整体平移 spectator 的世界位姿(实测 79.132 m),
    重建仍然自洽(相对几何没变),但**按世界坐标摆的道具进不了画面** ——
    而症状长得像"道具资产不渲染"。
    """

    def test_subtracts_the_parent(self):
        assert to_parent_frame(10.0, 20.0, 5.0, parent=(3.0, 4.0, 1.5)) == (7.0, 16.0, 3.5)

    def test_identity_parent_is_a_noop(self):
        """反向对照:父在原点时换算必须是**恒等** —— 否则这条会误伤正常情形。"""
        assert to_parent_frame(1.0, -2.0, 3.0, parent=(0.0, 0.0, 0.0)) == (1.0, -2.0, 3.0)

    def test_ring_around_center_becomes_ring_around_origin(self):
        """★ 与 `ring_cam_pose` 合起来的手算锚点 —— 这才是采集器真正的用法。

        相机 attach 在 `spectator = 环心 + 1.5 m` 上 ⇒ 相机的**相对**位姿必须**打平到原点**,
        而**世界**位姿必须仍绕环心。两条同时成立,换算才对。
        """
        center = (100.0, -50.0, 0.6)
        spec_world = (center[0], center[1], center[2] + 1.5)
        for i in (0, 30, 60):
            x, y, z, _ = ring_cam_pose(center[0], center[1], center[2], 6.0, i, 90, height=1.5)
            rx, ry, rz = to_parent_frame(x, y, z, parent=spec_world)
            # ① 相对位姿与环心**无关**(只由半径与角度决定)
            assert abs(rx * rx + ry * ry - 36.0) < 1e-6, "相对位姿的水平模长应当 = 半径"
            assert abs(rz) < 1e-9, "相对高度应当为 0(相机与 spectator 同高)"
            # ② 加上父的世界位姿之后,才是"绕环心的圆"
            assert abs((rx + spec_world[0]) - x) < 1e-9 and abs((ry + spec_world[1]) - y) < 1e-9


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

"""`gs/probe_hole_visibility` 的判据 —— 它推翻了一条**已写进文档的结论**,所以自己必须先站得住。

被推翻的那条(原 §C1):「物体背后的背景在 A 侧**从未被观测过** ⇒ 删掉高斯之后那里没有东西可填」。
探针实测:**中位 104/270 个视角看得到**,**0.0%** 的点一个都看不到;而且 A 侧模型在那块上
**高斯覆盖 93.5%**(删除前后**一模一样**)。

⇒ 那条说法**两头都不成立**。本组钉的就是"这个读数凭什么可信"。
"""

from __future__ import annotations

import numpy as np

from autodrivedata.gs import probe_hole_visibility as P


class TestSampleDepth:
    def test_reads_the_pixel(self):
        d = np.arange(16, dtype=float).reshape(4, 4)
        assert P.sample_depth(d, np.array([[2.0, 1.0]]))[0] == d[1, 2]

    def test_out_of_frame_is_inf_not_zero(self):
        """★ 越界必须返回 `inf`(= 看不见)。返回 0 会被读成"这里有个 0 米的东西" ⇒ 恒判可见。"""
        d = np.ones((4, 4))
        got = P.sample_depth(d, np.array([[-5.0, 0.0], [0.0, 99.0], [99.0, 0.0], [0.0, -1.0]]))
        assert np.all(np.isinf(got)), "越界 = 看不见,不是 0 米"


class TestBackprojectProject:
    """★ 反投影与投影必须**互逆** —— 不互逆的话"投回视角"这一步全错,而它不会报错。"""

    @staticmethod
    def _cam():
        k = np.array([[100.0, 0.0, 8.0], [0.0, 100.0, 6.0], [0.0, 0.0, 1.0]])
        vm = np.eye(4)  # 相机在世界原点、无旋转 ⇒ 世界系 = 相机系
        return k, vm

    def test_round_trip_through_a_constant_depth_plane(self):
        k, vm = self._cam()
        depth = np.full((12, 16), 5.0)
        uv = np.array([[8.0, 6.0], [10.0, 7.0], [4.0, 3.0]])
        pc = P.backproject(depth, uv, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
        assert np.allclose(pc[:, 2], 5.0), "恒深平面 ⇒ 相机系 z 必须恒等于深度"
        back, z = P.project(pc, vm, k)
        assert np.allclose(back, uv, atol=1e-9)
        assert np.allclose(z, 5.0)

    def test_to_world_inverts_the_viewmat(self):
        """`viewmat` 满足 `p_cam = R·p_world + t` ⇒ `to_world` 必须是它的逆。

        ⚠️ **别把 `t` 当成相机位置**:相机在世界 `C`、旋转为单位时 `t = −C`
        (第一版测试就写反了,断言"相机在 (1,2,3) 时 t=(1,2,3)" —— 那对应的相机其实在 **(−1,−2,−3)**)。
        """
        k, _ = self._cam()
        vm = np.eye(4)
        vm[:3, 3] = [-1.0, -2.0, -3.0]  # ⇒ 相机在 (1,2,3)
        pc = np.array([[0.5, -0.5, 4.0]])
        w = P.to_world(pc, vm)
        assert np.allclose(w, [[1.5, 1.5, 7.0]]), "相机在 (1,2,3) ⇒ 相机系 (0.5,-0.5,4) 的世界点 = 相加"
        back, _ = P.project(w, vm, k)
        assert np.allclose(back[0], [k[0, 0] * 0.5 / 4 + k[0, 2], k[1, 1] * -0.5 / 4 + k[1, 2]])

    def test_to_world_and_project_are_inverses(self):
        """★ 互逆是**契约**:探针的整条链就是"反投影出世界点 → 投回视角",不互逆则全错且不报错。"""
        k, _ = self._cam()
        rng = np.random.default_rng(0)
        th = 0.7
        rot = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        vm = np.eye(4)
        vm[:3, :3] = rot
        vm[:3, 3] = np.array([2.0, -1.0, 0.5])
        pc = rng.normal(size=(20, 3)) + np.array([0.0, 0.0, 8.0])
        w = P.to_world(pc, vm)
        back, z = P.project(w, vm, k)
        assert np.allclose(z, pc[:, 2]), "相机系 z 必须还原"
        assert np.allclose(back[:, 0], k[0, 0] * pc[:, 0] / pc[:, 2] + k[0, 2])


class TestCountVisible:
    """★★ 遮挡判据 —— 它是整条结论的承重件,必须有能失败的对照。"""

    @staticmethod
    def _setup(depth_val=5.0, shape=(12, 16)):
        k = np.array([[100.0, 0.0, 8.0], [0.0, 100.0, 6.0], [0.0, 0.0, 1.0]])
        return k, np.eye(4), np.full(shape, depth_val)

    def test_point_on_the_surface_is_seen(self):
        k, vm, d = self._setup()
        # 相机系 z=5 的点,投到主点处
        pts = np.array([[0.0, 0.0, 5.0]])
        assert P.count_visible(pts, [d], np.stack([vm]), np.stack([k]), (12, 16))[0] == 1

    def test_point_behind_the_surface_is_not_seen(self):
        """★ 被挡住(比记录的深度更远)⇒ **不算可见**。这是"遮挡"这两个字的全部内容。"""
        k, vm, d = self._setup()
        pts = np.array([[0.0, 0.0, 9.0]])  # 比 5 m 的墙更远
        assert P.count_visible(pts, [d], np.stack([vm]), np.stack([k]), (12, 16))[0] == 0

    def test_point_behind_the_camera_is_not_seen(self):
        k, vm, d = self._setup()
        assert (
            P.count_visible(np.array([[0.0, 0.0, -3.0]]), [d], np.stack([vm]), np.stack([k]), (12, 16))[0]
            == 0
        )

    def test_out_of_frame_is_not_seen(self):
        k, vm, d = self._setup()
        assert (
            P.count_visible(np.array([[50.0, 0.0, 5.0]]), [d], np.stack([vm]), np.stack([k]), (12, 16))[0]
            == 0
        )

    def test_tightening_the_gate_must_reduce_visibility(self):
        """★★ **反向自证**:把门槛收紧 1 m,可见数必须**掉**。

        ⚠️ 第一版的自证用的是 `+2.0`(**放松**),读数只从 104 动到 105 —— **那不是对照**:
        大多数点在被看到的视角里本来就在最前面,放松门槛几乎不改变任何判决。
        收紧才会真的改判决。这条钉的就是"自证必须是会动的那一种"。
        """
        k, vm, d = self._setup()
        pts = np.array([[0.0, 0.0, 4.5]])  # 比墙近 0.5 m ⇒ 默认可见
        base = P.count_visible(pts, [d], np.stack([vm]), np.stack([k]), (12, 16))
        tight = P.count_visible(pts, [d], np.stack([vm]), np.stack([k]), (12, 16), depth_bias=-1.0)
        assert base[0] == 1 and tight[0] == 0, "收紧 1 m 后这个点应当变成'被挡住'"


class TestVerdict:
    def _rep(self, **kw):
        base = {
            "n_views": 270,
            "visible_median": 104.0,
            "visible_max": 215,
            "frac_zero_visible": 0.0,
            "control_road_median_visible": 48.0,
            "control_tightened_median": 0.0,
            "frac_hole_with_gaussian": 0.935,
        }
        base.update(kw)
        return base

    def test_zero_visible_means_cancel_the_experiment(self):
        v = P.verdict(self._rep(visible_max=0))
        assert "没有视角" in v and "取消" in v

    def test_dead_control_is_undecided_not_negative(self):
        """★ 阳性对照(路面)没起来 ⇒ **未判** —— 不许读成"没视角看得到"。"""
        assert "未判" in P.verdict(self._rep(control_road_median_visible=1.0))

    def test_a_tightening_control_that_does_not_move_is_undecided(self):
        """★ 收紧后可见数没掉 ⇒ 这条判据没在用深度 ⇒ **未判**,不许报"有视角看得到"。"""
        assert "未判" in P.verdict(self._rep(control_tightened_median=104.0))

    def test_full_picture_names_the_real_conclusion(self):
        v = P.verdict(self._rep())
        assert "不成立" in v and "高斯在" in v

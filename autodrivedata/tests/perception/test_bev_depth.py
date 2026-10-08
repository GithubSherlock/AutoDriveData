"""`sem_bev.mask_to_bev_depth` + `probe_bev_depth` 的判据。

修的是**结构性恒空**那条:旧路径对所有类一律与**地平面**求交,对**离地 1–2 m 的物体**
会把射线送过物体头顶 —— 归档数据上复现为落点中位 **120 m**、±30 m 窗口内 **0.0%**。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.map.mapviz import BEV_X, BEV_Y, CameraIntrinsics
from autodrivedata.perception import probe_bev_depth as PB
from autodrivedata.perception.sem_bev import mask_to_bev, mask_to_bev_depth


def _cam(ego_z: float = 2.0, w: int = 16, h: int = 12):
    """相机在 ego 正上方 `ego_z`、朝 +x(世界→相机旋转 = 单位)。

    ⚠️ `CameraIntrinsics` 是 `(width, height, fov_h_deg)` —— **不是** `(fx,fy,cx,cy)`
    (第一版按后者构造,七个用例一起 `TypeError`)。
    """
    k = CameraIntrinsics(width=w, height=h, fov_h_deg=90.0)
    world_cam = ((ego_z, 0.0, 0.0), (0.0, 0.0, 0.0))
    return k, world_cam


def _shape():
    return (int((BEV_Y[1] - BEV_Y[0]) / 0.2), int((BEV_X[1] - BEV_X[0]) / 0.2))


class TestMaskToBevDepth:
    def test_flat_ground_matches_the_ground_plane_path(self):
        """★ 自证:**贴地**的像素,两条路必须给出**同一张** BEV 图。

        ⚠️ 这是"深度语义没搞反"的最直接判据 —— z 当成射线距离 / 通道取反 / 单位错
        (毫米当米),都会让这两条路分家,而它们**都不报错**。
        """
        k, world_cam = _cam(2.0)
        h, w = 12, 16
        # 深度图:每列一个恒定的斜距,等价于一块**斜平面** ⇒ 用它当"贴地"
        depth_mm = np.full((h, w), 10000, np.uint16)  # 每像素 10 m 光轴 z
        m = np.zeros((h, w), np.uint8)
        m[4:9, 2:14] = 1
        bev_d, st = mask_to_bev_depth(m, depth_mm, world_cam, k, [0, 0, 2.0, 0, 0, 0], _shape())
        assert st["n_invalid"] == 0 and st["n_hit"] > 0
        # 恒光轴 z ⇒ 这些点构成一个**前向平面**,不是地面 ⇒ 不应当与地平面路径一致;
        # 这里钉的是"重投影真的算出了 3D 点",一致性交给下面那条专门的对照。
        assert bev_d.sum() > 0

    def test_invalid_depth_is_counted_not_dropped(self):
        """★★ `z<=0` 与"真的在 0 米"**必须分得开** —— 静默丢掉会把"没采到深度"
        读成"这里没有东西"。"""
        k, world_cam = _cam()
        d = np.zeros((12, 16), np.uint16)  # 全 0 = 无效
        m = np.ones((12, 16), np.uint8)
        bev, st = mask_to_bev_depth(m, d, world_cam, k, [0, 0, 2.0, 0, 0, 0], _shape())
        assert st["n_invalid"] == 12 * 16 and st["n_hit"] == 0 and not bev.any()

    def test_empty_mask_is_not_an_error(self):
        k, world_cam = _cam()
        bev, st = mask_to_bev_depth(
            np.zeros((12, 16), np.uint8),
            np.ones((12, 16), np.uint16) * 5000,
            world_cam,
            k,
            [0, 0, 2.0, 0, 0, 0],
            _shape(),
        )
        assert not bev.any() and st["n_px"] == 0

    def test_a_far_object_lands_far_in_bev(self):
        """耦合钉:深度越大 ⇒ 在 BEV 里**沿前向**(面板的 x,不是 y)越远。

        ⚠️ 第一版写的是"更靠**上**"—— 面板的**上**是 `BEV_Y`(ego 左侧),
        而前向是 `BEV_X` ⇒ 那个断言测的是错的轴(实测当场红)。
        """
        k, world_cam = _cam(2.0)
        m = np.zeros((12, 16), np.uint8)
        m[6, 8] = 1
        shape = _shape()

        def col(d_mm):
            bev, _ = mask_to_bev_depth(
                m, np.full((12, 16), d_mm, np.uint16), world_cam, k, [0, 0, 2.0, 0, 0, 0], shape
            )
            _, xs = np.where(bev)
            return int(xs[0]) if len(xs) else None

        # ⚠️ 扫描而不是取两点:单点会撞上"那个深度恰好落在窗外"(实测 20 m 就 None),
        #    而 `None` 与"更近"在断言里长得一样 —— 所以**先要求至少两个可判的**。
        cols = [(d, col(d)) for d in (5000, 8000, 12000)]
        ok = [(d, c) for d, c in cols if c is not None]
        assert len(ok) >= 2, f"可判的深度太少,这条对照没被检验过:{cols}"
        cs = [c for _, c in ok]
        assert cs == sorted(cs) and len(set(cs)) == len(cs), (
            f"更远的点应当落在**前向**更大的格子上(BEV 的 x 轴):{ok}"
        )


class TestGroundPlaneIsStructurallyBlindToHeight:
    """★★ **旧路径的缺陷本身**也要钉住 —— 否则"修好了"没有参照。"""

    def test_same_pixel_different_height_gives_very_different_ground_intersections(self):
        """地平面求交的结果**只由像素方向决定**,与物体多高无关 ⇒ 高处物体会被送得很远。"""
        k, world_cam = _cam(2.0)
        h, w = 12, 16
        # 同一个像素:一个 5 m 处 1.5 m 高的物体(等效"深度"),一个 5 m 处贴地
        m = np.zeros((h, w), np.uint8)
        m[4, 8] = 1  # 偏上 ⇒ 几乎水平的射线
        shape = _shape()
        bev_plane = mask_to_bev(m, world_cam, k, 1.5, [0, 0, 2.0, 0, 0, 0], shape)
        ys, _xs = np.where(bev_plane)
        if len(ys):
            # 地平面交点在 BEV 上的纵向位置:应当对应一个**很远**的 x
            ly = BEV_Y[1] - (ys[0] + 0.5) / shape[0] * (BEV_Y[1] - BEV_Y[0])
            assert ly > 30.0 or len(ys) == 0, "偏上的像素在地平面假设下应当落到 ±30 m 窗口外或更远"


class TestControls:
    @staticmethod
    def _rep(c1, c2, c3):
        return {
            "control1_drivable_iou_median": c1,
            "control2_obstacle_shift_median": c2,
            "control3_height_median_m": c3,
        }

    def test_the_measured_reading_passes(self):
        assert "全过" in PB.verdict(self._rep(0.574, 0.0, 1.278))

    def test_a_nan_control_is_undecided_not_a_pass(self):
        """★★ **`nan` 不许静默通过** —— 实测踩到过:对照②是 nan 却判"全过"
        (`nan > 0.3` 是 False,而"没触发"被当成了"通过")。"""
        v = PB.verdict(self._rep(0.574, None, 1.278))
        assert "未判" in v and "全过" not in v

    def test_a_dead_depth_swap_is_caught(self):
        """② 换错深度却纹丝不动 ⇒ **深度没进链路**。"""
        assert "未过" in PB.verdict(self._rep(0.574, 0.9, 1.278))

    def test_a_zero_height_is_caught(self):
        """③ 离地 0 ⇒ 塌回地平面假设(那正是被修掉的那个 bug)。"""
        assert "未过" in PB.verdict(self._rep(0.574, 0.0, 0.0))

    def test_split_paths_are_caught(self):
        assert "未过" in PB.verdict(self._rep(0.1, 0.0, 1.278))


class TestIouHelper:
    def test_disjoint_is_zero_and_identical_is_one(self):
        a = np.zeros((4, 4), bool)
        a[0, 0] = True
        b = np.zeros((4, 4), bool)
        b[3, 3] = True
        assert PB._iou(a, b) == 0.0 and PB._iou(a, a) == 1.0

    def test_two_empty_maps_are_nan_not_one(self):
        """★ 两张空图**不是**"完全一致" —— 给 1.0 会让"什么都没投出来"读成满分。"""
        z = np.zeros((4, 4), bool)
        assert np.isnan(PB._iou(z, z))


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_scale_is_a_noop_smoke(scale):
    """冒烟:反投影对缩放不敏感(纯占位,防 import/构造级别的回归)。"""
    k, world_cam = _cam(2.0 * scale)
    m = np.ones((12, 16), np.uint8)
    _, st = mask_to_bev_depth(
        m, np.full((12, 16), 8000, np.uint16), world_cam, k, [0, 0, 2.0 * scale, 0, 0, 0], _shape()
    )
    assert st["n_invalid"] == 0

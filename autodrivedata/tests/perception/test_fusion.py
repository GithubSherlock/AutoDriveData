"""后融合的**纯值**判据:几何口径、尺寸门、关联、两档消融的取舍。

## 为什么这个文件是**补**出来的

`fusion.py` 第一版**先跑通、后补测**,首跑召回 10%,当时归因成"尺寸门按 GT 的 `l` 给、把真车挡了"。
**那个归因后来撤回了**(证据来自跨坐标系的诊断;同坐标系重查,真车簇两个门都过,
见 Plan4 §P-V18 三)。**但"先跑通后补测"这件事本身仍然付了代价**:四个几何口径写在注释里
没有钉,谁改错都不报错 —— 这就是它现在在这儿的原因。

⇒ 尺寸门那条用**实跑量到的真车簇形状**当夹具(而不是合成的规整尺寸):
夹具取自真实观测,形状变了才会当场红。

## 四个口径,错了都不报错

| 口径 | 错了的表现 |
|---|---|
| `ry = −π/2` | 长轴横过来 ⇒ 与 GT 的 3D IoU 掉到接近 0,**图上无异常** |
| `l` 取 velo 的 **x** 跨度 | 框变横向,同样只表现为 IoU 低 |
| `Box7.y` 取**底部**(velo 的 `cz − half_z`) | 框整体上下偏一个半高 |
| 尺寸门按**观测量**给,不按 GT 的 `l` | **召回塌掉**(已实测:10%) |
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autodrivedata.perception.compare import Box7, box3d_iou
from autodrivedata.perception.fusion import (
    ASSOC_IOU,
    RY_ALONG_AXIS,
    CameraDet,
    Cluster,
    associate,
    box2d_iou,
    cluster_to_box7,
    fuse,
    is_person_shaped,
    project_box7_to_image,
    size_plausible,
    strict_gate,
)

#: ★ velo→cam 的**轴向**关系:`cam = (y_v, −z_v, x_v)` —— velo 的 z(上)映到 cam 的 −y(下)。
#: **夹具里绝不能拿 `np.eye(4)` 顶替**:那等于把这个轴交换抹掉,而它正是被测的口径之一
#: (第一版就是这么写的,结果"底部中心"那条测的是恒等映射,自然对不上)。
V2C = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
I4 = np.eye(4)
#: KITTI 口径的 P2(1242×375 / fov 90)。投影用例共用。
P2 = np.array([[621.0, 0.0, 620.5, 0.0], [0.0, 621.0, 187.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
#: 一台车在 **velodyne 系**的位置:`(前 20 m, 0, 心高 0.75)`。
CAR_VELO = (20.0, 0.0, 0.75)
#: `label_2` 里那四台车(_000000_)的实测值 —— 尺寸门的"正样本"以它为准。
GT_CAR = Box7(label="Car", h=1.52, w=2.01, l=4.51, x=3.51, y=0.83, z=60.16, ry=RY_ALONG_AXIS)

#: ★ **实跑量到的三个真车簇**(velo 系 `(l, w, h)`)—— 它们当年全被旧门挡掉。
#: 注意 `l` 只有 1.86–1.89:那是**车尾的可见跨度**,不是车长。
REAL_CAR_CLUSTERS = ((1.89, 2.06, 1.53), (1.89, 1.07, 1.55), (1.86, 0.80, 1.95))


def _cl(size: tuple[float, float, float], center=(0.0, 0.0, 0.0), n=50) -> Cluster:
    return Cluster(center=center, half=tuple(v / 2 for v in size), n_points=n)


class TestClusterToBox7:
    def test_ry_is_along_the_optical_axis_not_zero(self):
        """★ `label_2` 实测四台车 `ry` 全 **−1.57**。照搬 0 ⇒ 长轴横过来。"""
        b = cluster_to_box7(_cl((4.0, 2.0, 1.5)), V2C)
        assert b.ry == pytest.approx(RY_ALONG_AXIS)
        assert b.ry == pytest.approx(-math.pi / 2)

    def test_ry_zero_would_collapse_the_iou(self):
        """反向对照:把 `ry` 写成 0,同一个框与 GT 的 3D IoU 必须**塌下来** ——
        否则上一条是死的(说明 `ry` 在这批数据上根本不重要)。"""
        # `good` 与 GT **逐字段相同**(所以 IoU 恰为 1);`bad` 只把 `ry` 翻成 0。
        # ⚠️ `y` 用 GT 自己的 0.83 —— 那是**底部**(`label_2` 的 y 就是底部),
        # 再减一次半高是重复扣(第一版就这么写的,于是 IoU 只有 0.21)。
        good = Box7(
            label="Car",
            h=GT_CAR.h,
            w=GT_CAR.w,
            l=GT_CAR.l,
            x=GT_CAR.x,
            y=GT_CAR.y,
            z=GT_CAR.z,
            ry=RY_ALONG_AXIS,
        )
        bad = Box7(**{**good.__dict__, "ry": 0.0})
        assert box3d_iou(good, GT_CAR) == pytest.approx(1.0)
        assert box3d_iou(bad, GT_CAR) < 0.4, "把 ry 写成 0 之后 IoU 必须**明显**掉下来"

    def test_l_takes_the_velodyne_x_span(self):
        """★ velo 的 x=前(车头方向)⇒ `l` 取 x 跨度。写反只表现为 IoU 低。"""
        long_along_x = cluster_to_box7(_cl((5.0, 1.5, 1.4)), V2C)
        assert (long_along_x.l, long_along_x.w) == pytest.approx((5.0, 1.5))
        long_along_y = cluster_to_box7(_cl((1.5, 5.0, 1.4)), V2C)
        assert (long_along_y.l, long_along_y.w) == pytest.approx((1.5, 5.0)), "w/l 不许对调"

    def test_y_is_the_bottom_not_the_centre(self):
        """★ `Box7.y` 是**底部**;velo 的 z 向上 ⇒ 取 `cz − half_z` 再变换。

        簇心 z=0.9、半高 0.75 ⇒ velo 系的底面在 z=0.15。经 `V2C` 之后
        `cam.y = −z_velo` ⇒ `y` 必须是 **−0.15**;取成中心(0.9)则得 −0.9,差一个半高。
        """
        b = cluster_to_box7(_cl((4.0, 2.0, 1.5), center=(0.0, 0.0, 0.9)), V2C)
        assert b.y == pytest.approx(-(0.9 - 0.75)), "取的是中心(0.9)而不是底部(0.15)"
        assert b.h == pytest.approx(1.5)

    def test_translation_lands_in_the_box(self):
        b = cluster_to_box7(_cl((4.0, 2.0, 1.5), center=CAR_VELO), V2C)
        # velo (20, 0, 0.75) → cam (0, −0.75, 20);底面 = 0.75 − 0.75 = 0
        assert (b.x, b.y, b.z) == pytest.approx((0.0, 0.0, 20.0))


class TestSizePlausible:
    def test_the_real_car_clusters_pass(self):
        """★★ **这条是那次召回塌掉的回归钉** —— 夹具就是当年被误杀的那三个簇。

        旧门口径:`l ∈ (2.8, 5.8)` + 长宽比 `(1.25, 4.0)`。实测簇 `l=1.86–1.89`、
        `l/w ≈ 1.0` ⇒ **三个全挂**,而它们是货真价实的车。
        """
        for size in REAL_CAR_CLUSTERS:
            assert size_plausible(_cl(size)), f"真车簇 {size} 被挡掉了 —— 门又按 GT 的 l 给了"

    def test_thin_slivers_are_rejected(self):
        """反面:薄片不是车(路面残留 / 标线 / 栏杆)。"""
        assert not size_plausible(_cl((0.45, 0.05, 0.00)))
        assert not size_plausible(_cl((0.30, 0.25, 5.90)))

    def test_buildings_and_ground_patches_are_rejected(self):
        assert not size_plausible(_cl((20.0, 12.0, 8.0)))
        assert not size_plausible(_cl((4.0, 2.0, 0.05))), "贴地的薄片不是车"

    def test_the_ratio_gate_is_gone_on_purpose(self):
        """★ 车尾迎面时 `l ≈ w`(遮挡,不是形状)⇒ **长宽比不能当门**。"""
        assert size_plausible(_cl((1.9, 2.1, 1.5)))


class TestProjection:
    def test_a_box_in_front_projects_around_the_principal_point(self):
        box = Box7(label="Car", h=1.5, w=2.0, l=4.0, x=0.0, y=0.75, z=20.0, ry=RY_ALONG_AXIS)
        got = project_box7_to_image(box, P2)
        assert got is not None
        x1, y1, x2, y2 = got
        assert x1 < 620.5 < x2 and y1 < 187.0 < y2
        # 20 m 处宽 2 m ⇒ 约 62 px;只断言量级,不钉像素级(那是另一套判据的事)
        assert 30 < (x2 - x1) < 120

    def test_behind_the_camera_is_none(self):
        box = Box7(label="Car", h=1.5, w=2.0, l=4.0, x=0.0, y=0.75, z=-20.0, ry=RY_ALONG_AXIS)
        assert project_box7_to_image(box, P2) is None

    def test_box2d_iou_hand_computed(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 0.0, 15.0, 10.0)
        assert box2d_iou(a, b) == pytest.approx(50 / 150)
        assert box2d_iou(a, a) == 1.0


class TestAssociate:
    @staticmethod
    def _d(x1, y1, x2, y2, conf=0.9) -> CameraDet:
        return CameraDet(label="Car", xyxy=(x1, y1, x2, y2), conf=conf)

    def test_best_pair_wins_and_is_one_to_one(self):
        proj = [(0.0, 0.0, 10.0, 10.0)]
        dets = [self._d(0, 0, 10, 10), self._d(1, 1, 11, 11)]
        assert associate(proj, dets) == [0], "IoU 高的那个该赢,另一个不该被重复占用"

    def test_below_threshold_is_none(self):
        got = associate([(0.0, 0.0, 10.0, 10.0)], [self._d(50, 50, 60, 60)])
        assert got == [None]

    def test_unprojectable_cluster_is_none(self):
        assert associate([None], [self._d(0, 0, 10, 10)]) == [None]

    def test_two_clusters_two_dets(self):
        # dets[0] 对的是 proj[1]、dets[1] 对的是 proj[0] ⇒ 结果必须是 [1, 0]
        proj = [(0.0, 0.0, 10.0, 10.0), (100.0, 0.0, 110.0, 10.0)]
        dets = [self._d(101, 0, 111, 10), self._d(0, 0, 10, 10)]
        assert associate(proj, dets) == [1, 0]

    def test_threshold_is_the_module_constant(self):
        assert ASSOC_IOU == pytest.approx(0.3)


class TestFuse:
    @staticmethod
    def _dets():
        """与「velo (20, 0, 0.75) 处那台车」的投影**基本重合**的 2D 框。

        先手算过:该框投出来约 `[589, 164, 651, 210]`,所以这里给一个略大的盒子
        —— 但**不能给太大**,否则 2D IoU 掉到 `ASSOC_IOU` 之下,关联会静默失败,
        而症状是"相机什么都没确认",看不出是夹具给大了。
        """
        return [CameraDet(label="Car", xyxy=(585.0, 160.0, 655.0, 215.0), conf=0.77)]

    @staticmethod
    def _car() -> list[Cluster]:
        return [_cl((4.0, 2.0, 1.5), center=CAR_VELO)]

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            fuse([], [], I4, np.eye(3, 4), mode="lidar+radar")

    def test_lidar_mode_ignores_the_camera(self):
        got = fuse(self._car(), [], V2C, P2, mode="lidar")
        assert len(got) == 1 and got[0].sources == ("lidar",)

    def test_lidar_plus_cam_drops_unconfirmed(self):
        """★ 这是消融的**机制本身**:没被相机确认的簇在 `lidar+cam` 档里不出现。"""
        assert len(fuse(self._car(), [], V2C, P2, mode="lidar+cam")) == 0, "没相机确认 ⇒ 丢掉"
        assert len(fuse(self._car(), self._dets(), V2C, P2, mode="lidar+cam")) == 1

    def test_conf_comes_from_a_different_source_per_mode(self):
        """★ 这条钉的是消融的**注意事项**:两档 conf 来源不同 ⇒ 只比 AP 会把
        "排序变了"读成"检测变好了"。计数才是与排序无关的那一半证据。"""
        cl = [_cl((4.0, 2.0, 1.5), center=CAR_VELO, n=7)]
        a = fuse(cl, self._dets(), V2C, P2, mode="lidar")[0]
        b = fuse(cl, self._dets(), V2C, P2, mode="lidar+cam")[0]
        assert a.conf == pytest.approx(7 / 20), "lidar 档的 conf 来自簇点数"
        assert b.conf == pytest.approx(0.77), "lidar+cam 档的 conf 来自相机"

    def test_size_gate_applies_to_both_modes(self):
        junk = [_cl((20.0, 12.0, 8.0))]
        assert fuse(junk, self._dets(), V2C, P2, mode="lidar") == []
        assert fuse(junk, self._dets(), V2C, P2, mode="lidar+cam") == []


class TestPersonShapeAndTheTwoGates:
    """★ 融合**最核心的机制**在这几条里:LiDAR 只出几何、**类由相机给**。

    在这之前 `fuse` 只有"`lidar` 档不看相机 / `lidar+cam` 档丢未确认"两条,**没有一条**
    断言"类真的来自相机" —— 而那正是整个融合的价值所在(summary 里 `Cyclist` AP
    从 0 → 0.182 靠的就是它)。
    """

    def test_person_shaped_distinguishes_the_two_families(self):
        assert is_person_shaped(_cl((0.5, 0.4, 1.7))), "1.7 m 高、0.5 m 宽 = 人"
        assert not is_person_shaped(_cl((4.0, 2.0, 1.5))), "车不是人形"
        assert not is_person_shaped(_cl((0.5, 0.4, 0.4))), "蹲着/贴地的碎片不是人"

    def test_loose_gate_accepts_persons_strict_does_not(self):
        """★★ **两条门的分工**:宽门吃人形(否则行人的簇在**入口**就被丢了,
        再准的相机也没有东西可以确认);严门(旧口径)只吃车形。"""
        person = _cl((0.5, 0.4, 1.7))
        assert size_plausible(person), "宽门必须放行人形簇"
        assert not strict_gate(person), "严门是纯车形口径,不该放人"

    def test_strict_gate_still_accepts_the_real_car_cluster(self):
        """严门**不是错的** —— 它只是整体很严(见 Plan4 §P-V18 三)。"""
        assert strict_gate(_cl((4.28, 1.90, 1.13)))
        assert not strict_gate(_cl((20.0, 12.0, 8.0)))

    #: **人形簇的 2D 检出框**。⚠️ 不能拿车那个框 `(585,160,655,215)` 顶替:
    #: 人形框在 20 m 处只投到约 `[614,134,627,187]`(0.4 m 宽 / 1.7 m 高),
    #: 与车那个框的 2D IoU 只有 **0.084** ⇒ 关联静默失败 ⇒ `lidar+cam` 档空表,
    #: 而症状是"相机什么都没确认",看不出是夹具给大了。
    PERSON_DET = (612.0, 132.0, 630.0, 190.0)

    def test_class_comes_from_the_camera_in_the_fused_arm(self):
        """★★ **这条是整个融合的价值所在**:同一个簇,`lidar` 档只能判 `Car`,
        `lidar+cam` 档拿到**相机给的类**。

        它错了不会报错 —— 只会让 `Cyclist`/`Pedestrian` 全变成 FP,而报表上
        看起来就像"模型检不出自行车"。
        """
        cl = [_cl((0.5, 0.4, 1.7), center=CAR_VELO)]
        dets = [CameraDet(label="Cyclist", xyxy=self.PERSON_DET, conf=0.66)]
        only = fuse(cl, dets, V2C, P2, mode="lidar")
        fused = fuse(cl, dets, V2C, P2, mode="lidar+cam")
        assert len(only) == 1 and len(fused) == 1
        assert only[0].box.label == "Car", "LiDAR 档没有类信息,只能全判 Car"
        assert fused[0].box.label == "Cyclist", "类必须来自相机"
        assert fused[0].conf == pytest.approx(0.66)

    def test_the_camera_label_does_not_move_the_geometry(self):
        """换个类只换名字,盒子几何**逐字段不动** —— 否则"类来自相机"会顺手改了几何。"""
        cl = [_cl((0.5, 0.4, 1.7), center=CAR_VELO)]
        dets = [CameraDet(label="Pedestrian", xyxy=self.PERSON_DET, conf=0.5)]
        a = fuse(cl, dets, V2C, P2, mode="lidar")[0].box
        b = fuse(cl, dets, V2C, P2, mode="lidar+cam")[0].box
        assert (a.h, a.w, a.l, a.x, a.y, a.z, a.ry) == (b.h, b.w, b.l, b.x, b.y, b.z, b.ry)


class TestCocoToKitti:
    """★ 相机类 → KITTI 类。**这一列错了,PQ/AP 的整个类维度就是错的,而它不报错。**"""

    @pytest.mark.parametrize(
        ("coco", "kitti"),
        [(0, "Pedestrian"), (1, "Cyclist"), (3, "Cyclist"), (2, "Car"), (5, "Car"), (7, "Car")],
    )
    def test_mapping(self, coco, kitti):
        from autodrivedata.perception.eval_fusion import COCO_TO_KITTI

        assert COCO_TO_KITTI[coco] == kitti

    def test_unrelated_coco_classes_are_dropped(self):
        from autodrivedata.perception.eval_fusion import COCO_TO_KITTI

        for c in (16, 17, 18, 19):  # dog/cat/horse/sheep —— 不该进检测表
            assert c not in COCO_TO_KITTI

"""gt.py 手算锚点单测(纯 numpy)。

场景:相机 (0,0,1.65) yaw=0(朝 +x_g),内参 1242×375 fov=90(fx=621,cx=620.5,cy=187)。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.calib.core import CameraIntrinsics
from autodrivedata.gt import core as gt

K = CameraIntrinsics(width=1242, height=375, fov_h_deg=90.0)
CAM_LOC = (0.0, 0.0, 1.65)
CAM_ROT = (0.0, 0.0, 0.0)  # 朝 +x_g

CAR = gt.ActorBox(
    type_id="vehicle.audi.a2",
    extent=(2.0, 1.0, 1.0),  # 半尺寸 → h=2, w=2, l=4
    location=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0),
    actor_location=(10.0, 0.0, 0.0),
    actor_rotation=(0.0, 0.0, 0.0),  # 车头 +x_g
)


class TestClassify:
    @pytest.mark.parametrize(
        ("type_id", "expected"),
        [
            ("walker.pedestrian.0001", "Pedestrian"),
            ("vehicle.bmw.grandtourer", "Car"),
            ("vehicle.audi.a2", "Car"),
            ("vehicle.gazelle.omafiets", "Cyclist"),
            ("vehicle.kawasaki.ninja", "Cyclist"),
            ("vehicle.carlamotors.carlacola", "Truck"),
            ("vehicle.carlamotors.european_hgv", "Truck"),
            ("vehicle.tesla.cybertruck", "Truck"),
            ("vehicle.mitsubishi.fusorosa", "Truck"),  # 公交 → 大型车
            ("vehicle.ford.ambulance", "Van"),
            ("static.prop.streetlamp", "Misc"),
        ],
    )
    def test_mapping(self, type_id, expected):
        assert gt.classify_kitti(type_id) == expected


class TestCenterAndHeading:
    def test_center_world_with_offset(self):
        box = gt.ActorBox(
            type_id="vehicle.audi.a2",
            extent=(2.0, 1.0, 1.0),
            location=(0.0, 0.0, 1.0),  # 车体抬升 1m
            rotation=(0.0, 0.0, 0.0),
            actor_location=(10.0, 0.0, 0.0),
            actor_rotation=(0.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(gt.box_center_world(box), [10.0, 0.0, 1.0], atol=1e-12)

    def test_heading_with_box_rotation(self):
        box = gt.ActorBox(
            type_id="vehicle.audi.a2",
            extent=(2.0, 1.0, 1.0),
            location=(0.0, 0.0, 0.0),
            rotation=(0.0, np.pi / 2, 0.0),  # box 相对 actor 转 90° → 车头 +y_g
            actor_location=(10.0, 0.0, 0.0),
            actor_rotation=(0.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(gt.box_heading_world(box), [0.0, 1.0, 0.0], atol=1e-12)


class TestGtLine:
    def test_car_straight_ahead(self):
        """10m 外正前方:底心 (0, 2.65, 10)、ry=−π/2、trunc=0.25(近处底角出画幅下缘)。"""
        line = gt.box_to_gt_line(CAR, CAM_LOC, CAM_ROT, K)
        assert line is not None
        parts = line.split()
        assert len(parts) == 15
        assert parts[0] == "Car"
        # 2D bbox:**全 front 角点** min/max 再钳画幅(2026-10-02 口径修正,旧值是内侧角点那套)。
        # 该框有角点出画(trunc=0.25),旧口径把它裁掉 ⇒ 框偏小。
        assert parts[4] == "542.88" and parts[5] == "220.64"
        assert parts[6] == "698.12" and parts[7] == "374.00", (
            "下缘该钳到画幅(出画角点的投影),不是只取内侧角点"
        )
        # 3D:底心 x/y/z + 尺寸 + ry(车头 +z 远离相机 → −π/2)
        assert [parts[i] for i in (8, 9, 10)] == ["2.00", "2.00", "4.00"]
        assert [parts[i] for i in (11, 12, 13, 14)] == ["0.00", "2.65", "10.00", "-1.57"]
        assert parts[1] == "0.25"  # truncation = 1 − 6/8

    def test_box_offset_and_rotation(self):
        box = gt.ActorBox(
            type_id="vehicle.audi.a2",
            extent=(2.0, 1.0, 1.0),
            location=(0.0, 0.0, 1.0),
            rotation=(0.0, np.pi / 2, 0.0),
            actor_location=(10.0, 0.0, 0.0),
            actor_rotation=(0.0, 0.0, 0.0),
        )
        line = gt.box_to_gt_line(box, CAM_LOC, CAM_ROT, K)
        assert line is not None
        parts = line.split()
        # 中心 (10,0,1) → 相机系 (0, 0.65, 10);底心 y = 0.65+1 = 1.65;车头 +y_g → ry=0
        assert [parts[i] for i in (11, 12, 13, 14)] == ["0.00", "1.65", "10.00", "0.00"]

    def test_max_distance_filter(self):
        # 100m 外目标无 LiDAR 点(M3-3 教训):超距剔除
        far = gt.ActorBox(
            type_id="vehicle.audi.a2",
            extent=(2.0, 1.0, 1.0),
            location=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            actor_location=(100.0, 0.0, 0.0),
            actor_rotation=(0.0, 0.0, 0.0),
        )
        assert gt.box_to_gt_line(far, CAM_LOC, CAM_ROT, K, max_distance=65.0) is None
        # 不加限制时 100m 目标投影入图,会正常出框(训练数据毒化来源)
        assert gt.box_to_gt_line(far, CAM_LOC, CAM_ROT, K) is not None

    def test_behind_camera_dropped(self):
        box = gt.ActorBox(
            type_id="vehicle.audi.a2",
            extent=(2.0, 1.0, 1.0),
            location=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            actor_location=(-10.0, 0.0, 0.0),
            actor_rotation=(0.0, 0.0, 0.0),
        )
        assert gt.box_to_gt_line(box, CAM_LOC, CAM_ROT, K) is None

    def test_fully_outside_dropped(self):
        box = gt.ActorBox(
            type_id="vehicle.audi.a2",
            extent=(2.0, 1.0, 1.0),
            location=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            actor_location=(0.0, 30.0, 0.0),  # 正侧方 30m,镜头外
            actor_rotation=(0.0, 0.0, 0.0),
        )
        assert gt.box_to_gt_line(box, CAM_LOC, CAM_ROT, K) is None

    def test_degenerate_sliver_dropped(self):
        """★ 退化投影(擦过镜头的车)必须**剔除**,不是发一条零面积的框出去。

        复现的是 2026-10-01 在 P1 数据里量到的那一族:车在 ego **正侧 3 m 处**、
        整车落在像面下缘之下,只剩顶面那条边进画幅 —— 投出来 `y1 == y2`。

        危害有二,都不是"少一条框"这么轻:
        ① 零面积框与任何预测的 IoU 恒为 0 ⇒ **白送一次漏检**,P1 每份数据 11/194 条,
           recall 天花板被压到 94.3%;
        ② 它的进出由**亚帧抖动**决定 —— 远底角在 v≈375 上下几 px 摆动,0.24 m 的 ego
           偏移就让整框在「0 px 高」与「164 px 高」之间翻面,于是两次采集的 GT 逐帧
           条数不再相等(A/B 硬门槛当场破)。
        """
        car = gt.ActorBox(
            type_id="vehicle.tesla.model3",
            extent=(2.395, 1.0815, 0.744),
            location=(0.0, 0.0, 0.744),
            rotation=(0.0, 0.0, 0.0),
            actor_location=(3.0, 3.5, 0.0),  # ego 正侧 3.5 m、车头朝 +x_g
            actor_rotation=(0.0, 0.0, 0.0),
        )
        # 相机在 (0,0,1.65)、朝 +x_g:该车整体在相机**右侧且几乎齐平**,只剩顶边进画幅
        assert gt.box_to_gt_line(car, CAM_LOC, CAM_ROT, K) is None

    def test_a_one_pixel_box_still_survives(self):
        """反向对照:1 px 是**下界不是筛子** —— 刚过线的框必须留着。

        否则这条剔除就会静默吃掉远距小目标(而"吃掉了多少"只有数变了才知道)。
        """
        box = gt.ActorBox(
            type_id="vehicle.audi.a2",
            extent=(0.3, 0.3, 0.3),  # h=w=l=0.6 m:30 m 处约 12 px
            location=(0.0, 0.0, 3.0),
            rotation=(0.0, 0.0, 0.0),
            actor_location=(30.0, 0.0, 0.0),
            actor_rotation=(0.0, 0.0, 0.0),
        )
        line = gt.box_to_gt_line(box, CAM_LOC, CAM_ROT, K)
        assert line is not None
        p = line.split()
        assert float(p[6]) - float(p[4]) >= gt.MIN_BOX_SIDE_PX
        assert float(p[7]) - float(p[5]) >= gt.MIN_BOX_SIDE_PX

    @pytest.mark.parametrize(
        ("x1", "y1", "x2", "y2", "degenerate"),
        [
            ("653.33", "189.17", "685.99", "211.57", False),  # 正常框
            ("943.49", "231.32", "1183.14", "231.32", True),  # 0 px 高(线)
            ("1006.45", "239.85", "1006.45", "239.85", True),  # 0×0(点)
            ("947.14", "231.78", "1189.44", "231.79", True),  # ★ 0.01 px 高
            ("900.00", "225.00", "1100.00", "226.00", False),  # 1.00 px 高:边界,留
            ("900.00", "225.00", "1100.00", "225.99", True),  # 0.99 px 高
        ],
    )
    def test_is_degenerate_gt_line(self, x1, y1, x2, y2, degenerate):
        """★ 读侧判据必须与出框侧的 `MIN_BOX_SIDE_PX` **同源**,且**不是** `x1 == x2`。

        第三行那条是 2026-10-01 实测的真数据(`kitti_ab_epic_wet_road` 帧 57):它
        **0.01 px 高**,打印出来两个数**不相等**,但按阈值早该被剔除 —— 写成相等判断
        会让那一帧的 GT 条数比别的 root 多 1,A/B 硬门槛当场破。
        """
        line = f"Car 0.00 0 0.00 {x1} {y1} {x2} {y2} 1.52 2.01 4.51 3.50 0.84 14.47 -1.57"
        assert gt.is_degenerate_gt_line(line) is degenerate

    def test_is_degenerate_gt_line_on_junk(self):
        """残缺行也算"该剔除" —— 留着只会变成一条永远配不上的 GT。"""
        assert gt.is_degenerate_gt_line("Car 0.00 0")
        assert gt.is_degenerate_gt_line("")

    def test_partial_truncation_ground_camera(self):
        """相机贴地 (0,0,0):高车(h=4)在 6m 处,近侧角点(z=4)出画幅 → trunc=0.5。

        手算:center_k=(0,0,6)、y_bottom=2;z∈[4,8]。z=8 的 4 角在画幅内
        (u∈[542.88, 698.12]、v∈[31.75, 342.25]);z=4 的 4 角出界 → trunc = 1 − 4/8 = 0.5。

        ★ **2D 框按新口径**:全 front 角点 min/max **再钳到画幅** ⇒ 上/左钳到 0、
        下钳到 374。旧口径只取内侧 4 角(31.75–342.25),比真实可见范围小。
        """
        box = gt.ActorBox(
            type_id="vehicle.carlamotors.carlacola",
            extent=(2.0, 1.0, 2.0),  # h=4
            location=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            actor_location=(6.0, 0.0, 0.0),
            actor_rotation=(0.0, 0.0, 0.0),
        )
        line = gt.box_to_gt_line(box, (0.0, 0.0, 0.0), CAM_ROT, K)
        assert line is not None
        parts = line.split()
        assert parts[0] == "Truck"
        assert parts[1] == "0.50"
        # ★ 新口径:全 front 角点 min/max **再钳到画幅** ⇒ 钳到 0 / 374。
        #   旧值是内侧 4 角那套 `542.88, 31.75, 698.12, 342.25`,比真实可见范围小。
        assert [parts[i] for i in (4, 5, 6, 7)] == ["465.25", "0.00", "775.75", "374.00"]
        # 底心 (0, 2, 6)
        assert [parts[i] for i in (11, 12, 13)] == ["0.00", "2.00", "6.00"]


class TestClipConventionIsTwoParts:
    """★★ **2D 框的两句口径是配套的,拆开任一句都会坏**(2026-10-02)。

    | 只做 | 后果 |
    |---|---|
    | 只钳画幅,退化剔除也看钳后的框 | 「擦过镜头」的物体角点投影发散 ⇒ 钳完是**接近满幅**的框 ⇒ 与任何预测都能配上 ⇒ **凭空造出 TP** |
    | 不钳画幅 | 被边缘裁掉的物体框偏小 ⇒ 检测器**对的**框被记成漏检(实测归档 17.6%) |

    所以判据必须同时钉住两句。上面 `test_partial_truncation_ground_camera` 钉①,
    这里钉② —— **而且是要它"仍然被剔除"**。
    """

    #: 同一台车的两个距离 —— 5 m 起保留(只被裁)、4 m 起剔除(发散)。实测的转折点。
    @staticmethod
    def _car_at(d: float) -> gt.ActorBox:
        return gt.ActorBox(
            type_id="vehicle.audi.a2",
            extent=(2.0, 1.0, 1.0),
            location=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            actor_location=(d, 0.0, 0.0),
            actor_rotation=(0.0, 0.0, 0.0),
        )

    def test_a_merely_clipped_box_is_kept_and_reaches_the_edge(self):
        """① **只被裁**的框必须保留,且框**要够到画幅边**(10 m 处下缘 374)。"""
        line = gt.box_to_gt_line(self._car_at(10.0), CAM_LOC, CAM_ROT, K)
        assert line is not None
        assert float(line.split()[7]) == 374.00, "下缘必须够到画幅底,而不是停在内侧角点"

    def test_a_near_passing_box_is_still_rejected_not_clipped_to_full_frame(self):
        """② **擦过镜头**的框**必须仍被剔除**(4 m 起)。

        若有人把退化判据也改成看**钳后**的框,这条会红 —— 而那个错在数据里表现为
        「突然多出一批与任何预测都能配上的巨框」,**AP 虚高**,比原来的"框太小"更毒。
        """
        assert gt.box_to_gt_line(self._car_at(4.0), CAM_LOC, CAM_ROT, K) is None
        assert gt.box_to_gt_line(self._car_at(2.0), CAM_LOC, CAM_ROT, K) is None

    def test_the_two_clauses_are_a_pair(self):
        """★ 把两句**并排**钉:同一台车,5 m 保留 / 4 m 剔除 —— 转折点就在这两者之间。

        单看任一条都看不出"配套",所以这条把边界摆在一起;哪天有人只改一句,
        两条断言里必有一条红。
        """
        assert gt.box_to_gt_line(self._car_at(5.0), CAM_LOC, CAM_ROT, K) is not None
        assert gt.box_to_gt_line(self._car_at(4.0), CAM_LOC, CAM_ROT, K) is None


class TestReboxLine:
    """`rebox_line` = **归档数据的离线重算**(不重采)。与出框侧共用 `box2d_from_projection`。

    ## 为什么必须有这一层

    2D 框可以从**行自己的字段**重算 —— `label_2` 自带 `h w l x y z ry`,calib 自带 `P2`。
    所以口径变更不必重采(重采是投骰子,见 `refilter.py` 头注)。但**重算侧与出框侧
    必须是同一句口径** —— 两份实现迟早漂,而漂了的表现是"重算出来的框与新采的框不是
    同一套",数字照样出得来。
    """

    def test_a_real_archived_clipped_line_reaches_the_frame_edge(self):
        """★ **真实归档行 + 实测答案**。

        取自 `outputs/kitti_ab_occl2_full_all/training/label_2/000014.txt` 第 4 行 ——
        即量影响面时那个 **Δmax = 292 px** 的案例(近处右侧被画幅裁掉的车)。
        期望值 `781.1 / 198.3 / 1241.0 / 374.0` 是**独立量出来的**,不是这条代码算出来的:
        先前用另一段脚本按「全角点 min/max 再钳画幅」重算过一遍,两者一致。
        """
        # ⚠️ 这三个字符串是**从归档文件逐字读出**的,不是凭打印重打的 ——
        #   第一版手抄了一行,z 抄成 60.16(实为 48.94),重算立刻差 13.6 px,
        #   而**差异被当成了代码 bug**。抄录即口径。
        line = "Car 0.38 0 0.00 780.87 198.22 948.86 296.86 1.49 2.16 4.79 3.50 1.66 6.97 -1.57"
        got = gt.rebox_line(line, K.p2(), K.width, K.height)
        assert got is not None
        assert [float(v) for v in got.split()[4:8]] == pytest.approx(
            [781.08, 198.27, 1241.0, 374.0], abs=0.02
        )
        # 3D 列**逐位不动** —— 重算只动 2D
        assert line.split()[8:] == got.split()[8:]

    def test_an_unclipped_line_is_returned_verbatim(self):
        """★ **反向对照,而且是逐字节的**:没被裁的行**原样返回**。

        两种口径在「全角点落在画幅内」时逐位相同 —— 那时若还按舍入过的 `x/y/z/ry`
        重打一遍,只会把 ~0.1 px 的**字段精度噪声**注进本来没问题的行(归档里 82.4%)。
        这条同时是 `--clip2d` 的 diff 保证:**只有该动的行会动**(见
        `test_refilter.py::TestClip2d`)。

        ⚠️ 别把它读成"重算不生效" —— 上面那条裁断行会变 292 px,两条合起来才是口径的定义域。
        """
        # ✅ 逐字取自同一份归档文件的**第 1 行**(未出画)。第一版凭打印重打,z 抄成 60.16
        #    (实为 48.94)⇒ 重算差 13.6 px,而**差异被当成了代码 bug**。抄录即口径。
        line = "Car 0.00 0 0.00 650.78 189.04 680.41 209.53 1.52 2.01 4.51 3.50 1.69 48.94 -1.57"
        assert gt.rebox_line(line, K.p2(), K.width, K.height) == line

    def test_a_degenerate_row_is_dropped(self):
        """★ **擦镜头的行必须仍被剔除** —— 重算侧也要守那句配套口径。"""
        line = "Car 0.00 0 0.00 0 0 0 0 2.00 2.00 4.00 0.00 2.65 1.00 -1.57"
        assert gt.rebox_line(line, K.p2(), K.width, K.height) is None

    def test_idempotent_within_the_rounding_of_the_stored_fields(self):
        """★ **幂等**:新口径出的行重算一遍,2D 列必须**基本**不变。

        ⚠️ 不是逐位相同:行里的 `x/y/z/ry` 存的是**两位小数**,重算用的是**舍入后**的值,
        所以有 ~0.1 px 的漂。这是**字段精度**问题,不是口径分叉 —— 判据给 0.5 px 容差
        并写明理由,免得下一个人把它当 bug 去"修"。

        (未裁断的行走短路,**恰好**逐位相同;容差是留给"出框侧刚裁过、重算再裁一次"那类行的。)
        """
        box = gt.ActorBox(
            type_id="vehicle.audi.a2",
            extent=(2.0, 1.0, 1.0),
            location=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            actor_location=(10.0, 0.0, 0.0),
            actor_rotation=(0.0, 0.0, 0.0),
        )
        line = gt.box_to_gt_line(box, CAM_LOC, CAM_ROT, K)
        assert line is not None
        again = gt.rebox_line(line, K.p2(), K.width, K.height)
        assert again is not None
        d = max(
            abs(a - b)
            for a, b in zip(map(float, line.split()[4:8]), map(float, again.split()[4:8]), strict=True)
        )
        assert d < 0.5, f"重算漂了 {d} px —— 用舍入后的字段重算不该超过 0.5 px"

    def test_short_line_raises(self):
        with pytest.raises(ValueError, match="字段不足"):
            gt.rebox_line("Car 0.00 0", K.p2(), K.width, K.height)

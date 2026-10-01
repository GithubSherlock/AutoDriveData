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
        # 2D bbox:内侧角点 u∈[542.88, 698.12](698.125 银行家舍入),v∈[220.64, 324.14]
        assert parts[4] == "542.88" and parts[5] == "220.64"
        assert parts[6] == "698.12" and parts[7] == "324.14"
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

    def test_partial_truncation_ground_camera(self):
        """相机贴地 (0,0,0):高车(h=4)在 6m 处,近侧角点(z=4)出画幅 → trunc=0.5。

        手算:center_k=(0,0,6)、y_bottom=2;z∈[4,8]。z=8 的 4 角在画幅内
        (u∈[542.88, 698.12]、v∈[31.75, 342.25]);z=4 的 4 角出界 → trunc = 1 − 4/8 = 0.5。
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
        assert [parts[i] for i in (4, 5, 6, 7)] == ["542.88", "31.75", "698.12", "342.25"]
        # 底心 (0, 2, 6)
        assert [parts[i] for i in (11, 12, 13)] == ["0.00", "2.00", "6.00"]

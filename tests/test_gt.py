"""gt.py 手算锚点单测(纯 numpy)。

场景:相机 (0,0,1.65) yaw=0(朝 +x_g),内参 1242×375 fov=90(fx=621,cx=620.5,cy=187)。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata import gt
from autodrivedata.calib import CameraIntrinsics

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

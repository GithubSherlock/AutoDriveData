"""gt.py vs auto3dlabel Box3D oracle(autolabel env 跑;base env 自动跳过)。

我们的 GT 行 → Box3D.from_gt_row → line_from_box3d 回写:
3D 字段(label + [8:15])必须往返一致(2 位小数)——即 auto3dlabel 零改动可消费我们的 GT。
"""

from __future__ import annotations

import numpy as np
import pytest

auto3dlabel = pytest.importorskip("auto3dlabel.schema.calib")

from auto3dlabel.export.kitti_label import line_from_box3d  # noqa: E402
from auto3dlabel.schema.box3d import Box3D  # noqa: E402

from autodrivedata import gt  # noqa: E402
from autodrivedata.calib import CameraIntrinsics  # noqa: E402

K = CameraIntrinsics(width=1242, height=375, fov_h_deg=90.0)
CAM_LOC = (0.0, 0.0, 1.65)
CAM_ROT = (0.0, 0.0, 0.0)


def _round_trip(line: str) -> tuple[list[str], list[str]]:
    parts = line.split()
    label = parts[0]
    h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
    x, y, z, ry = float(parts[11]), float(parts[12]), float(parts[13]), float(parts[14])
    box = Box3D.from_gt_row(label, h, w, l, x, y, z, ry)
    re = line_from_box3d(box).split()
    return parts, re


class TestGtRoundTrip:
    def test_car_straight_ahead(self):
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
        parts, re = _round_trip(line)
        assert re[0] == parts[0]
        for i in range(8, 15):
            assert re[i] == parts[i], f"字段 {i}: {re[i]} != {parts[i]}"

    def test_heading_facing_camera(self):
        """车头朝 −x_g(面向相机):ry=+π/2,往返一致。"""
        box = gt.ActorBox(
            type_id="vehicle.audi.a2",
            extent=(2.0, 1.0, 1.0),
            location=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            actor_location=(10.0, 0.0, 0.0),
            actor_rotation=(0.0, np.pi, 0.0),
        )
        line = gt.box_to_gt_line(box, CAM_LOC, CAM_ROT, K)
        assert line is not None
        parts, re = _round_trip(line)
        for i in range(8, 15):
            assert re[i] == parts[i], f"字段 {i}: {re[i]} != {parts[i]}"
        assert parts[14] == "1.57"

"""static_gt.py 单测:landmark kind 归一、JSON roundtrip、车道线段合并。"""

from __future__ import annotations

import json

from autodrivedata.gt.props import CameraPose
from autodrivedata.gt.static_gt import (
    LaneSegment,
    StaticFrame,
    StaticSignal,
    landmark_kind,
    merge_lane_marks,
)


def test_landmark_kind_normalization():
    assert landmark_kind("Signal_3Light_Post01") == "traffic_light"
    assert landmark_kind("SignalJunction") == "traffic_light"
    assert landmark_kind("Sign_Stop_Blue") == "stop"
    assert landmark_kind("Sign_Yield") == "yield"
    assert landmark_kind("random_prop") == "unknown"


def test_static_frame_json_roundtrip():
    sig = StaticSignal(
        landmark_id="960",
        kind="traffic_light",
        name="Signal_3Light_Post01",
        location=(-119.158, 5.093, -0.437),
        yaw_deg=180.0,
    )
    seg = LaneSegment(
        side="right",
        mark_type="Broken",
        color="White",
        width=0.125,
        points=((0.0, 0.0, 0.0), (1.0, 3.5, 0.0), (2.0, 3.5, 0.0)),
    )
    f = StaticFrame(
        frame_id="000000",
        ego_location=(-64.6, 24.5, 0.55),
        ego_yaw_deg=0.0,
        signals=(sig,),
        lane_lines=(seg,),
    )
    g = StaticFrame.from_json(f.to_json())
    assert g == f
    assert g.point_count() == 3
    assert g.camera is None, "不传就不该凭空多出一个位姿"


def test_static_frame_camera_roundtrip():
    """`camera` 是可选的**新增**字段(2026-10-02):判据要靠它离线复现投影链。

    ★ 两个方向都要钉:
    - 传了 → 逐字段往返相等(判据拿到的内参/位姿必须与采集时一致);
    - **没传 → `None`,不是默认内参** —— 2026-10-02 之前采的归档都是这种,
      判据必须能把它认出来并**报错**(拿默认内参硬算会得到一整套看着正常的错数)。
    """
    cam = CameraPose(
        location=(-63.9, 24.5, 1.6),
        rotation_deg=(0.0, 0.16, 0.0),
        width=1242,
        height=375,
        fov_deg=90.0,
    )
    f = StaticFrame(
        frame_id="000000",
        ego_location=(-63.9, 24.5, 0.0),
        ego_yaw_deg=0.16,
        camera=cam,
    )
    g = StaticFrame.from_json(f.to_json())
    assert g.camera == cam
    assert g == f

    # 归档口径:老 json 里**没有** `camera` 这个键
    legacy = json.loads(f.to_json())
    legacy.pop("camera")
    assert StaticFrame.from_json(json.dumps(legacy)).camera is None


def test_merge_lane_marks_joins_same_props():
    samples = [
        {
            "side": "right",
            "mark_type": "Broken",
            "color": "White",
            "width": 0.125,
            "x": 0.0,
            "y": 3.5,
            "z": 0.0,
        },
        {
            "side": "right",
            "mark_type": "Broken",
            "color": "White",
            "width": 0.125,
            "x": 1.0,
            "y": 3.5,
            "z": 0.0,
        },
        # 属性变化 → 断新段
        {
            "side": "right",
            "mark_type": "Solid",
            "color": "White",
            "width": 0.125,
            "x": 2.0,
            "y": 3.5,
            "z": 0.0,
        },
        # 回到原属性 → 也是新段(合并不跨异属性)
        {
            "side": "right",
            "mark_type": "Broken",
            "color": "White",
            "width": 0.125,
            "x": 3.0,
            "y": 3.5,
            "z": 0.0,
        },
    ]
    segs = merge_lane_marks(samples)
    assert [s["points"] for s in segs] == [
        [(0.0, 3.5, 0.0), (1.0, 3.5, 0.0)],
        [(2.0, 3.5, 0.0)],
        [(3.0, 3.5, 0.0)],
    ]
    assert segs[0]["width"] == 0.125


def test_merge_lane_marks_alternating_sides_still_joins():
    """left/right 交替采样时各自独立合并(2026-09-09 实测 bug:全断成 1 点段)。"""
    samples = [
        {
            "side": "right",
            "mark_type": "Broken",
            "color": "White",
            "width": 0.125,
            "x": 0.0,
            "y": 3.5,
            "z": 0.0,
        },
        {
            "side": "left",
            "mark_type": "SolidSolid",
            "color": "Yellow",
            "width": 0.125,
            "x": 0.0,
            "y": -3.5,
            "z": 0.0,
        },
        {
            "side": "right",
            "mark_type": "Broken",
            "color": "White",
            "width": 0.125,
            "x": 5.0,
            "y": 3.5,
            "z": 0.0,
        },
        {
            "side": "left",
            "mark_type": "SolidSolid",
            "color": "Yellow",
            "width": 0.125,
            "x": 5.0,
            "y": -3.5,
            "z": 0.0,
        },
    ]
    segs = merge_lane_marks(samples)
    assert len(segs) == 2  # 不是 4 段
    right = next(s for s in segs if s["side"] == "right")
    left = next(s for s in segs if s["side"] == "left")
    assert len(right["points"]) == 2
    assert len(left["points"]) == 2
    assert right["points"][1] == (5.0, 3.5, 0.0)
    assert left["points"][1] == (5.0, -3.5, 0.0)


def test_merge_lane_marks_splits_by_side():
    samples = [
        {
            "side": "left",
            "mark_type": "SolidSolid",
            "color": "Yellow",
            "width": 0.125,
            "x": 0.0,
            "y": -3.5,
            "z": 0.0,
        },
        {
            "side": "right",
            "mark_type": "Broken",
            "color": "White",
            "width": 0.125,
            "x": 0.0,
            "y": 3.5,
            "z": 0.0,
        },
    ]
    segs = merge_lane_marks(samples)
    assert len(segs) == 2
    assert {s["side"] for s in segs} == {"left", "right"}

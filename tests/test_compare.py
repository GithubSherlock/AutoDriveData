"""compare.py 手算锚点单测(纯 numpy)。"""

from __future__ import annotations

import numpy as np

from autodrivedata.compare import (
    Box7,
    FrameStats,
    ap11,
    box3d_iou,
    box7_to_quad,
    evaluate_frames,
    load_gt_labels,
    load_pred_labels,
    match_boxes,
    parse_gt_line,
    polygon_iou,
    report_text,
)

CAR = Box7(label="Car", h=1.5, w=1.8, l=4.0, x=0.0, y=1.7, z=10.0, ry=-np.pi / 2)


class TestQuad:
    def test_corners(self):
        # ry=−π/2(车头 +z):BEV 角点 x=±0.9、z=10±2
        q = box7_to_quad(CAR)
        np.testing.assert_allclose(
            q,
            [[-0.9, 12.0], [0.9, 12.0], [0.9, 8.0], [-0.9, 8.0]],
            atol=1e-12,
        )


class TestPolygonIoU:
    def test_overlap(self):
        a = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        b = np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])
        assert abs(polygon_iou(a, b) - 1 / 7) < 1e-12

    def test_disjoint(self):
        a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        b = np.array([[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0]])
        assert polygon_iou(a, b) == 0.0

    def test_identical(self):
        a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        assert polygon_iou(a, a) == 1.0

    def test_rotated_overlap(self):
        # 45° 旋转的菱形 vs 轴对齐方形(相交非空、IoU 对称且 <1)
        q = box7_to_quad(Box7(label="Car", h=1.5, w=2.0, l=2.0, x=0.5, y=1.7, z=0.5, ry=np.pi / 4))
        s = np.array([[-0.5, -0.5], [1.5, -0.5], [1.5, 1.5], [-0.5, 1.5]])
        assert 0.0 < polygon_iou(q, s) < 1.0
        assert polygon_iou(q, s) == polygon_iou(s, q)


class TestBox3DIoU:
    def test_identical(self):
        assert box3d_iou(CAR, CAR) == 1.0

    def test_disjoint(self):
        far = Box7(label="Car", h=1.5, w=1.8, l=4.0, x=50.0, y=1.7, z=50.0, ry=0.0)
        assert box3d_iou(CAR, far) == 0.0

    def test_height_only_partial(self):
        # BEV 重合、高度错开(一高一低)→ y 区间无交 → 0
        low = Box7(label="Car", h=1.5, w=1.8, l=4.0, x=0.0, y=5.0, z=10.0, ry=-np.pi / 2)
        assert box3d_iou(CAR, low) == 0.0

    def test_symmetric(self):
        b = Box7(label="Car", h=1.5, w=1.8, l=4.0, x=0.5, y=1.7, z=10.5, ry=-np.pi / 2 + 0.1)
        assert box3d_iou(CAR, b) == box3d_iou(b, CAR)


class TestMatch:
    def test_perfect(self):
        gt = [CAR]
        pred = [Box7(label="Car", h=1.5, w=1.8, l=4.0, x=0.0, y=1.7, z=10.0, ry=-np.pi / 2, conf=0.9)]
        m = match_boxes(gt, pred)
        assert m.n_matches == 1 and m.missed == [] and m.extras == []

    def test_extra_and_missed(self):
        gt = [CAR]
        pred = [Box7(label="Car", h=1.5, w=1.8, l=4.0, x=30.0, y=1.7, z=30.0, ry=0.0, conf=0.5)]
        m = match_boxes(gt, pred)
        assert m.n_matches == 0 and m.missed == [0] and m.extras == [0]

    def test_class_mismatch(self):
        gt = [CAR]
        pred = [Box7(label="Pedestrian", h=1.5, w=1.8, l=4.0, x=0.0, y=1.7, z=10.0, ry=-np.pi / 2, conf=0.9)]
        m = match_boxes(gt, pred)
        assert m.n_matches == 0

    def test_iou_below_threshold(self):
        gt = [CAR]
        pred = [Box7(label="Car", h=1.5, w=1.8, l=4.0, x=3.0, y=1.7, z=10.0, ry=-np.pi / 2, conf=0.9)]
        m = match_boxes(gt, pred, iou_thresh=0.7)
        # BEV 中心偏移 3m,宽 1.8:交集 ≈ (1.8-3)/2… 实际无交 → 不匹配
        assert m.n_matches == 0


class TestAP11:
    def test_perfect(self):
        assert ap11([0.9, 0.8], [True, True], 2) == 1.0

    def test_nothing(self):
        assert ap11([], [], 2) == 0.0

    def test_half(self):
        # 2 GT,1 中 1 漏:R=0.5 处 P=1.0 → 11 点 AP = 6/11
        assert abs(ap11([0.9, 0.5], [True, False], 2) - 6 / 11) < 1e-12


class TestFrameStats:
    def test_divergent(self):
        assert FrameStats("f", 2, 2, 2, 0, 0).divergent is False
        assert FrameStats("f", 2, 1, 1, 1, 0).divergent is True  # 漏检
        assert FrameStats("f", 1, 2, 1, 0, 1).divergent is True  # 误报


class TestEvaluate:
    def test_report(self, tmp_path):
        gt_file = tmp_path / "gt.txt"
        pred_file = tmp_path / "pred.txt"
        gt_file.write_text("Car 0.00 0 0.00 0 0 0 0 1.50 1.80 4.00 0.00 1.70 10.00 -1.57\n")
        pred_file.write_text("Car 0.00 0 0.00 0 0 0 0 1.50 1.80 4.00 0.00 1.70 10.00 -1.57\n")
        gt = load_gt_labels(gt_file)
        pred = load_pred_labels(pred_file, conf=0.9)
        assert len(gt) == 1 and len(pred) == 1
        assert gt[0].label == "Car" and pred[0].conf == 0.9

        rep = evaluate_frames({"000000": (gt, pred)}, classes=["Car", "Pedestrian"])
        assert rep.per_class["Car"].ap == 1.0
        assert rep.review_rate == 0.0
        assert rep.total_matched == 1
        text = report_text(rep)
        assert "Car" in text and "复核率 0.0%" in text

    def test_missed_frame(self):
        gt = [CAR]
        rep = evaluate_frames({"000001": (gt, [])}, classes=["Car"])
        assert rep.per_class["Car"].fn == 1
        assert rep.review_rate == 1.0
        assert rep.frames[0].divergent is True

    def test_parse_roundtrip(self):
        line = "Pedestrian 0.10 1 0.00 100 200 300 400 1.86 0.38 0.38 -27.42 1.56 65.98 -1.57"
        b = parse_gt_line(line)
        assert b.label == "Pedestrian"
        assert (b.h, b.w, b.l) == (1.86, 0.38, 0.38)
        assert (b.x, b.y, b.z, b.ry) == (-27.42, 1.56, 65.98, -1.57)

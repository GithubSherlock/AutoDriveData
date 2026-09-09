"""失效归因纯值层单测(base env,不依赖 carla/PIL/ultralytics)。"""

from __future__ import annotations

import math

import pytest

from autodrivedata import attribution as attr


def gt_line(
    cls: str = "Car",
    box: tuple[float, float, float, float] = (100.0, 150.0, 200.0, 250.0),
    trunc: float = 0.0,
    z: float = 30.0,
) -> str:
    x1, y1, x2, y2 = box
    return f"{cls} {trunc:.2f} 0 0.00 {x1} {y1} {x2} {y2} 1.50 1.60 3.90 0.00 1.70 {z} -1.57"


class TestParse:
    def test_basic(self):
        g = attr.parse_gt_line_2d(gt_line())
        assert g is not None
        assert (g.cls, g.x1, g.y2, g.distance_m) == ("Car", 100.0, 250.0, 30.0)
        assert (g.width_px, g.height_px, g.area_px) == (100.0, 100.0, 10000.0)

    def test_truncation_parsed(self):
        g = attr.parse_gt_line_2d(gt_line(trunc=0.25))
        assert g is not None and g.truncation == 0.25 and not g.visible

    def test_short_line_rejected(self):
        assert attr.parse_gt_line_2d("Car 0.00 0 0.00 1 2 3 4") is None

    def test_irrelevant_class_rejected(self):
        assert attr.parse_gt_line_2d(gt_line(cls="Tram")) is None

    def test_coco_name_normalized(self):
        g = attr.parse_gt_line_2d(gt_line(cls="person"))
        assert g is not None and g.cls == "Pedestrian"


class TestIou2D:
    def test_identical(self):
        b = (0.0, 0.0, 10.0, 10.0)
        assert attr.box_iou2d(b, b) == pytest.approx(1.0)

    def test_disjoint(self):
        assert attr.box_iou2d((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_half_overlap(self):
        # 10×10 与右移 5 的 10×10:交 50,并 150
        assert attr.box_iou2d((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50.0 / 150.0)

    def test_touching_edges_is_zero(self):
        assert attr.box_iou2d((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_zero_area_no_crash(self):
        assert attr.box_iou2d((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


class TestMatchFrame:
    def _gt(self, *boxes):
        return [
            attr.GtBox2D(cls="Car", x1=b[0], y1=b[1], x2=b[2], y2=b[3], truncation=0.0, distance_m=20.0)
            for b in boxes
        ]

    def test_high_conf_wins(self):
        gt = self._gt((0, 0, 10, 10))
        preds = [
            attr.Detection("Car", 0, 0, 10, 10, conf=0.4),
            attr.Detection("Car", 0, 0, 10, 10, conf=0.9),
        ]
        assert attr.match_frame(gt, preds) == [True]  # 只有一条预测能认领

    def test_class_mismatch_not_matched(self):
        gt = self._gt((0, 0, 10, 10))
        preds = [attr.Detection("Pedestrian", 0, 0, 10, 10, conf=0.9)]
        assert attr.match_frame(gt, preds) == [False]

    def test_low_iou_not_matched(self):
        gt = self._gt((0, 0, 10, 10))
        preds = [attr.Detection("Car", 20, 20, 30, 30, conf=0.9)]
        assert attr.match_frame(gt, preds) == [False]

    def test_two_gt_two_pred(self):
        gt = self._gt((0, 0, 10, 10), (50, 50, 60, 60))
        preds = [
            attr.Detection("Car", 50, 50, 60, 60, conf=0.8),
            attr.Detection("Car", 0, 0, 10, 10, conf=0.7),
        ]
        assert attr.match_frame(gt, preds) == [True, True]

    def test_one_pred_does_not_steal_two_gt(self):
        # 两 GT 完全重合、只有一条预测 → 只能认领一个(与 AP 口径一致)
        gt = self._gt((0, 0, 10, 10), (0, 0, 10, 10))
        preds = [attr.Detection("Car", 0, 0, 10, 10, conf=0.9)]
        assert attr.match_frame(gt, preds) == [True, False]


class TestTtc:
    def test_basic(self):
        assert attr.ttc_s(24.0, 12.0) == pytest.approx(2.0)

    def test_zero_speed_is_inf(self):
        assert attr.ttc_s(10.0, 0.0) == math.inf


class TestBinning:
    def _rec(self, dist: float, matched: bool, height: float = 40.0) -> attr.GtRecord:
        return attr.GtRecord(
            frame="000000",
            cls="Car",
            distance_m=dist,
            height_px=height,
            truncation=0.0,
            ttc_s=dist / 8.0,
            matched=matched,
        )

    def test_bins_are_left_closed(self):
        recs = [self._rec(10.0, True), self._rec(9.99, False), self._rec(65.0, False)]
        stats = attr.bin_stats(recs, key=lambda r: r.distance_m)
        by_lo = {s.lo: s for s in stats}
        assert (by_lo[10.0].n_gt, by_lo[10.0].n_matched) == (1, 1)  # 10.0 落 10-20 箱
        assert (by_lo[0.0].n_gt, by_lo[0.0].n_matched) == (1, 0)
        assert by_lo[60.0].n_gt == 1  # 65 落末箱
        assert math.isinf(by_lo[60.0].hi)

    def test_rate_and_height_mean(self):
        recs = [self._rec(25.0, True, height=50.0), self._rec(26.0, False, height=30.0)]
        s = attr.bin_stats(recs, key=lambda r: r.distance_m)[2]  # 20-30
        assert s.n_gt == 2 and s.n_matched == 1
        assert s.rate == pytest.approx(0.5)
        assert s.height_mean == pytest.approx(40.0)

    def test_nan_and_negative_skipped(self):
        recs = [self._rec(-1.0, True), self._rec(math.nan, True)]
        stats = attr.bin_stats(recs, key=lambda r: r.distance_m)
        assert sum(s.n_gt for s in stats) == 0

    def test_empty_rate_is_nan(self):
        s = attr.bin_stats([], key=lambda r: r.distance_m)[0]
        assert math.isnan(s.rate) and math.isnan(s.height_mean)

    def test_format_marks_open_last_bin(self):
        text = attr.format_bins(attr.bin_stats([self._rec(65.0, True)], key=lambda r: r.distance_m))
        assert "60-+m" in text and "1" in text


class TestEstimateClosingSpeed:
    def test_linear_approach(self):
        # 8 m/s × 0.1s = 0.8m/帧;每帧多台车(取最远那台)
        frames = [[d, d - 15.0] for d in (60.0, 59.2, 58.4, 57.6, 56.8)]
        assert attr.estimate_closing_speed(frames, 0.1) == pytest.approx(8.0)

    def test_passing_car_frame_discarded(self):
        # 第 3→4 帧最远车被超过(突降 15m)→ 该帧作废,不影响中位数
        frames = [[60.0], [59.2], [58.4], [43.0], [42.2], [41.4]]
        assert attr.estimate_closing_speed(frames, 0.1) == pytest.approx(8.0)

    def test_empty_is_nan(self):
        assert math.isnan(attr.estimate_closing_speed([], 0.1))
        assert math.isnan(attr.estimate_closing_speed([[], []], 0.1))


class TestClosingSpeedSeries:
    def test_ramp_is_visible_per_frame(self):
        # 前 3 帧加速(0.2m/帧),后 3 帧稳态(0.8m/帧)→ 逐帧速度能看出 ramp
        frames = [[60.0], [59.8], [59.6], [59.4], [58.6], [57.8]]
        series = attr.closing_speed_series(frames, 0.1, window=1)
        assert series[0] == pytest.approx(2.0)
        assert series[-1] == pytest.approx(8.0)

    def test_window_smooths_and_keeps_length(self):
        frames = [[d] for d in (60.0, 59.2, 58.4, 57.6)]
        series = attr.closing_speed_series(frames, 0.1, window=5)
        assert len(series) == len(frames)
        assert all(v == pytest.approx(8.0) for v in series)

    def test_single_frame_has_no_estimate(self):
        assert math.isnan(attr.closing_speed_series([[30.0]], 0.1)[0])

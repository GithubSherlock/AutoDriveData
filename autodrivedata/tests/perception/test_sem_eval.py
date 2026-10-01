"""P2-A 判据的**纯值**部分:累加器、帧号解析、以及"这把尺子真的在量重叠吗"。

不跑模型、不读数据 —— 那些由 `sem_eval --self-test` 在真 root 上验(它有 GT 几何自证)。

## 为什么这几条非钉不可

判据自己是**唯一**能证伪判据的东西,而它塌掉的样子全都不是崩溃:

| 塌法 | 症状 |
|---|---|
| 空掩膜 / 键名对不上 | IoU 全是 `nan`,而 `nan` 在 `mean` 里被当"没样本"跳过 ⇒ **看着像"跑完了"** |
| 把"没样本"算成 0 | mIoU 被空类拉低,越少类越难看 |
| 尺子不量重叠 | 预测偏移半张图,IoU 却纹丝不动 |

最后一条用**扰动**钉:把预测掩膜平移, IoU 必须**单调下降** —— 尺子若恒返回 1.0 或常数,
这里立刻红。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.perception.sem_eval import Counts, Scoreboard, _parse_frames, bev_shape


class TestCounts:
    def test_hand_computed_iou(self):
        """手算:gt 是左边 4 格,pred 是中间 4 格 ⇒ tp=2, fp=2, fn=2 ⇒ IoU=1/3。"""
        gt = np.zeros(10, dtype=bool)
        gt[0:4] = True
        pred = np.zeros(10, dtype=bool)
        pred[2:6] = True
        c = Counts()
        c.add(gt, pred)
        assert (c.tp, c.fp, c.fn) == (2, 2, 2)
        assert c.iou == pytest.approx(1 / 3)

    def test_perfect_overlap_is_one(self):
        m = np.zeros((4, 4), dtype=bool)
        m[1:3, 1:3] = True
        c = Counts()
        c.add(m, m)
        assert c.iou == 1.0

    def test_empty_is_nan_not_zero(self):
        """★ **没样本 ≠ 全错**。`nan` 才有"这一类没被测到"的语义,`0.0` 会被读成"错光了"。"""
        c = Counts()
        c.add(np.zeros((3, 3), dtype=bool), np.zeros((3, 3), dtype=bool))
        assert np.isnan(c.iou)

    def test_accumulates_across_calls(self):
        a = np.zeros(4, dtype=bool)
        a[0] = True
        c = Counts()
        c.add(a, a)
        c.add(a, a)
        assert c.tp == 2 and c.iou == 1.0


class TestScoreboard:
    def test_miou_averages_only_the_classes_with_samples(self):
        """★ 空类被跳过 —— 这是**有意的**,所以更要钉:它等于把 mIoU 的定义悄悄改成
        "有样本的那几类的均值"。真实后果(`sem_eval` 首跑):BEV 的 obstacle 通道结构性
        为空 ⇒ BEV mIoU 实际只是两类的均值,运行时会单独打印警告。"""
        sb = Scoreboard("t")
        ok = np.ones((2, 2), dtype=bool)
        sb.counts["drivable"].add(ok, ok)
        sb.counts["lane"].add(ok, ok)
        # obstacle 一次都不加 ⇒ nan
        assert np.isnan(sb.counts["obstacle"].iou)
        assert sb.miou == 1.0

    def test_all_empty_is_nan(self):
        assert np.isnan(Scoreboard("t").miou)

    def test_line_prints_every_class_in_order(self):
        sb = Scoreboard("图像")
        line = sb.line()
        assert line.index("drivable") < line.index("lane") < line.index("obstacle")


class TestRulerActuallyMeasuresOverlap:
    """★ 尺子的**立论自证**:预测越偏, IoU 越差。恒返回常数的假尺子在这里必红。"""

    @staticmethod
    def _blob(shift: int, size: int = 64) -> np.ndarray:
        m = np.zeros((size, size), dtype=bool)
        m[20:40, 20:40] = True  # 20×20 的块
        if shift:
            m = np.roll(m, shift, axis=1)
        return m

    def test_iou_decreases_monotonically_with_shift(self):
        gt = self._blob(0)
        ious = []
        for shift in (0, 5, 10, 15, 20, 40):
            c = Counts()
            c.add(gt, self._blob(shift))
            ious.append(c.iou)
        assert ious[0] == 1.0
        assert all(b <= a + 1e-12 for a, b in zip(ious[:-1], ious[1:], strict=True)), ious
        assert ious[-1] == 0.0, "整块移开后应当完全不重叠"

    def test_a_shifted_gt_is_not_scored_as_perfect(self):
        """反向对照:位移 10 格(块宽的一半)时 IoU 必须**明显掉下来**,不能还是 1.0。"""
        c = Counts()
        c.add(self._blob(0), self._blob(10))
        assert c.iou < 0.6


class TestBevShapeAndFrames:
    def test_bev_shape_matches_the_frozen_window(self):
        from autodrivedata.map.mapviz import BEV_X, BEV_Y
        from autodrivedata.perception.sem_bev import BEV_PX

        bh, bw = bev_shape()
        assert bh == int((BEV_Y[1] - BEV_Y[0]) / BEV_PX)
        assert bw == int((BEV_X[1] - BEV_X[0]) / BEV_PX)

    @pytest.mark.parametrize(
        ("spec", "want"),
        [("0-3", [0, 1, 2, 3]), ("5", [5]), ("0-1,7,9-10", [0, 1, 7, 9, 10]), ("", [])],
    )
    def test_parse_frames(self, spec, want):
        assert _parse_frames(spec) == want

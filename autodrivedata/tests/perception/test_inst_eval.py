"""实例判据的**纯值**部分 —— 全部在合成掩膜上跑,不碰 CARLA、不碰真数据。

## 为什么必须合成

这里的塌法全不是崩溃,而是**一个看着挺正常的 0–1 比值**:

| 塌法 | 症状 |
|---|---|
| 配对不看类 | 「位置对但类报错」被算成命中 —— PQ 与 mask AP **都**要求类匹配,漏了会同时虚高 |
| 空并集返回 0 而不是 `nan` | "两个空掩膜"被当成"完全不重叠",凭空多一个 FN |
| 分母写成 `TP+FP+FN` | PQ 数值变小,而它仍是个 0–1 的数 |
| `PQ = SQ × RQ` 不成立 | 说明实现与定义不一致 —— 这条等式是 PQ 的**定义性质**,不是近似 |

反面对照同样是合成的:**把预测整体平移**、**把类改错**,读数必须掉下来。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.perception.inst_eval import (
    PredInstance,
    mask_ap,
    mask_iou,
    match_instances,
    panoptic_quality,
)
from autodrivedata.perception.inst_tags import Instance


def _inst(iid: int, cls_name: str, *, x0=0, y0=0, size=10, shape=(40, 40)) -> Instance:
    m = np.zeros(shape, dtype=bool)
    m[y0 : y0 + size, x0 : x0 + size] = True
    tag = {"obstacle": 14, "drivable": 1, "lane": 24}[cls_name]
    return Instance(instance_id=iid, cls_tag=tag, cls_name=cls_name, mask=m, consistent=True)


def _pred(cls_name: str, *, x0=0, y0=0, size=10, conf=1.0, shape=(40, 40)) -> PredInstance:
    m = np.zeros(shape, dtype=bool)
    m[y0 : y0 + size, x0 : x0 + size] = True
    return PredInstance(cls_name=cls_name, mask=m, conf=conf)


class TestMaskIou:
    def test_identical_is_one(self):
        a = _inst(1, "obstacle").mask
        assert mask_iou(a, a) == 1.0

    def test_disjoint_is_zero(self):
        assert mask_iou(_inst(1, "obstacle").mask, _inst(2, "obstacle", x0=20).mask) == 0.0

    def test_two_empty_masks_are_nan_not_zero(self):
        """★ **没有并集 ≠ 完全不重叠**。前者是 `nan`(没样本),后者才该是 0。"""
        e = np.zeros((4, 4), dtype=bool)
        assert np.isnan(mask_iou(e, e))

    def test_half_overlap(self):
        a = _inst(1, "obstacle", x0=0, size=10).mask
        b = _inst(2, "obstacle", x0=5, size=10).mask
        assert mask_iou(a, b) == pytest.approx(50 / 150)


class TestMatching:
    def test_perfect_match(self):
        m = match_instances([_inst(1, "obstacle")], [_pred("obstacle")])
        assert len(m.pairs) == 1 and m.fp == 0 and m.fn == 0

    def test_wrong_class_does_not_match_when_cls_aware(self):
        """★ **类错的预测不算命中** —— PQ 与逐类 AP 都要求这一条。"""
        gt = [_inst(1, "obstacle")]
        pr = [_pred("drivable")]
        assert len(match_instances(gt, pr, cls_aware=True).pairs) == 0
        assert len(match_instances(gt, pr, cls_aware=False).pairs) == 1

    def test_ap_is_blind_to_mask_quality_where_pq_is_not(self):
        """★ PQ 与 mask AP 的**真实差别**(别写成"PQ 多一个语义项" —— 两个都要求类匹配)。

        一个 IoU=1.00 的预测与一个 IoU≈0.55 的预测,在 `AP@0.5` 下**都算命中**(同一个数),
        而 PQ 里前者贡献 1.00、后者只贡献约 0.55。⇒ "掩膜画得糊不糊"只有 PQ 看得出。
        """
        gt = [_inst(1, "obstacle", size=10)]
        perfect = [_pred("obstacle", size=10)]
        # 10×10 与 9×9 错开 1px:IoU = 81/100 = 0.81 —— 过 0.5 阈值但明显不如 1.00
        coarse = [_pred("obstacle", x0=1, size=9)]
        iou = mask_iou(gt[0].mask, coarse[0].mask)
        assert 0.5 < iou < 0.95, f"夹具的 IoU 落在阈值外({iou:.3f}),这条测不出东西"
        assert mask_ap([(gt, perfect)], 0.5) == mask_ap([(gt, coarse)], 0.5), "AP@0.5 对掩膜质量应当不敏感"
        assert panoptic_quality(gt, perfect).pq > panoptic_quality(gt, coarse).pq

    def test_empty_prediction_is_all_fn(self):
        m = match_instances([_inst(1, "obstacle"), _inst(2, "obstacle", x0=20)], [])
        assert m.fn == 2 and m.fp == 0 and not m.pairs

    def test_empty_gt_is_all_fp(self):
        m = match_instances([], [_pred("obstacle")])
        assert m.fp == 1 and m.fn == 0

    def test_greedy_picks_the_best_pair_first(self):
        """一 GT 对两预测:吃掉 IoU 高的那个,另一个记 FP。"""
        gt = [_inst(1, "obstacle", x0=0, size=10)]
        pr = [_pred("obstacle", x0=1, size=10), _pred("obstacle", x0=0, size=10)]
        m = match_instances(gt, pr)
        assert len(m.pairs) == 1 and m.pairs[0][1] == 1 and m.fp == 1

    def test_below_threshold_is_not_a_match(self):
        gt = [_inst(1, "obstacle", x0=0, size=10)]
        pr = [_pred("obstacle", x0=9, size=10)]  # IoU = 1/19 ≈ 0.05
        m = match_instances(gt, pr, iou_thresh=0.5)
        assert not m.pairs and m.fp == 1 and m.fn == 1


class TestPanopticQuality:
    def test_perfect_is_one(self):
        gt = [_inst(1, "obstacle"), _inst(2, "obstacle", x0=20)]
        pr = [_pred("obstacle"), _pred("obstacle", x0=20)]
        r = panoptic_quality(gt, pr)
        assert (r.pq, r.sq, r.rq) == (1.0, 1.0, 1.0)

    def test_pq_equals_sq_times_rq(self):
        """★ **定义性质**,不是近似 —— 实现错了这条会对不上。"""
        gt = [_inst(1, "obstacle"), _inst(2, "obstacle", x0=20), _inst(3, "obstacle", y0=20)]
        pr = [_pred("obstacle"), _pred("obstacle", x0=21), _pred("obstacle", y0=30)]
        r = panoptic_quality(gt, pr)
        assert r.pq == pytest.approx(r.sq * r.rq)

    def test_no_samples_is_nan_not_zero(self):
        """★ **没有样本 ≠ 全错**。`0.0` 会被读成"错光了",`nan` 才是"没测"。"""
        r = panoptic_quality([], [])
        assert np.isnan(r.pq) and np.isnan(r.sq) and np.isnan(r.rq)

    def test_missing_one_object_costs_recall_not_segmentation_quality(self):
        """漏检一个 ⇒ RQ 掉、SQ **不动**(SQ 只量配对上的那些)。两件事必须分得开。"""
        gt = [_inst(1, "obstacle"), _inst(2, "obstacle", x0=20)]
        pr = [_pred("obstacle")]
        r = panoptic_quality(gt, pr)
        assert r.sq == 1.0
        assert r.rq == pytest.approx(1 / 1.5)
        assert r.pq == pytest.approx(1 / 1.5)

    def test_shifted_mask_lowers_sq_and_pq(self):
        """反面对照:把预测挪一点,SQ 必须掉下来 —— 恒 1.0 的假尺子在这里红。"""
        gt = [_inst(1, "obstacle", size=10)]
        base = panoptic_quality(gt, [_pred("obstacle", size=10)])
        shifted = panoptic_quality(gt, [_pred("obstacle", x0=3, size=10)])
        assert shifted.sq < base.sq and shifted.pq < base.pq


class TestMaskAp:
    def test_perfect_is_one(self):
        pairs = [([_inst(1, "obstacle")], [_pred("obstacle")])]
        assert mask_ap(pairs, 0.5) == 1.0

    def test_all_missed_is_zero(self):
        pairs = [([_inst(1, "obstacle")], [_pred("obstacle", x0=20)])]
        assert mask_ap(pairs, 0.5) == 0.0

    def test_no_gt_is_nan(self):
        assert np.isnan(mask_ap([([], [_pred("obstacle")])], 0.5))

    def test_confidence_ordering_matters(self):
        """★ conf 是**模型的原始值**,不许重排 —— 高 conf 的那个命中 vs 没命中,AP 必须不同。"""
        gt = [_inst(1, "obstacle", x0=0, size=10)]
        good = [_pred("obstacle", x0=0, size=10, conf=0.9), _pred("obstacle", x0=20, conf=0.1)]
        bad = [_pred("obstacle", x0=20, conf=0.9), _pred("obstacle", x0=0, size=10, conf=0.1)]
        assert mask_ap([(gt, good)], 0.5) > mask_ap([(gt, bad)], 0.5)

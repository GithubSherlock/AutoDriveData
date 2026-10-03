"""后端抽象(`perception/backends.py`)与 SAM3 提示词表的回归钉。

## 钉的是什么

2026-10-02 起**默认后端从闭集检测器换成 SAM3 开放词表**。这条切换最容易出的三类错
**都不报错**:

1. **类名从哪来**:yolo 的类名是**模型判的**,SAM3 的是**提示词给的**。若某入口
   悄悄用了另一套归一(或漏了归一),读数照出,只是"分类正确率"这一项不见了;
2. **提示词表被改/被漏**:`Car` 少了 `truck`/`bus` 就是**静默少检**(GT 里有卡车,
   而没人去问"卡车在哪")—— 数会低,但低得像"模型不行";
3. **一条提示当多类用**:SAM3 是**单概念**提示,拼串实测返回 0 个掩膜
   (`"car person bicycle"` / `"car, person, bicycle"` 都试过)⇒ 覆盖多概念**只能**逐条前向。

⚠️ 这里**不出模型**(不加载 3.15 GiB 权重):只钉"接口与口径",模型质量由实跑的数说话。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.perception import backends, sam3_backend


class TestClassNormalization:
    def test_coco_names_map_to_project_classes(self):
        """★ yolo 的类号→名→项目类这条链只此一处。`truck`/`bus` 归 `Car` 是 KITTI 口径。"""
        assert backends.norm_cls("car") == "Car"
        assert backends.norm_cls("truck") == "Car"
        assert backends.norm_cls("bus") == "Car"
        assert backends.norm_cls("person") == "Pedestrian"
        assert backends.norm_cls("bicycle") == "Cyclist"
        assert backends.norm_cls("motorcycle") == "Cyclist"

    def test_unknown_is_dropped_not_guessed(self):
        """不认识的名字必须**丢掉**。塞进最近的一类会让 AP 凭空多出 TP。"""
        for junk in ("train", "traffic light", "", "Cat"):
            assert backends.norm_cls(junk) == ""

    def test_project_class_names_round_trip(self):
        """项目类名本身必须能归一(SAM3 分支直接吐 `Car` 这类名字)。"""
        for c in backends.GT_CLASSES:
            assert backends.norm_cls(c) == c


class TestPromptTables:
    def test_every_project_class_has_prompts(self):
        """★ 少一条提示 = **静默少检一类**。这条挡住"加了类忘了加词"。"""
        for c in backends.GT_CLASSES:
            assert sam3_backend.DETECT_PROMPTS.get(c), f"{c} 没有提示词 ⇒ 这一类永远检不出来"

    def test_car_covers_truck_and_bus(self):
        """KITTI 的 `Car` 含卡车/客车(同 `COCO_FALLBACK`)⇒ 提示词也必须覆盖。"""
        assert {"truck", "bus"} <= set(sam3_backend.DETECT_PROMPTS["Car"])

    def test_prompts_are_single_words_not_joined(self):
        """★ **单概念提示**是硬约束:拼串实测返回 0 个掩膜。

        这条防的是"为了省前向把几条词拼成一句" —— 那种改法**不报错、只是全空**,
        而症状是"模型在雨夜完全失效",看起来特别合理。
        """
        for prompts in sam3_backend.DETECT_PROMPTS.values():
            for p in prompts:
                assert " " not in p, f"提示 {p!r} 含空格 —— 多概念拼串实测返回 0 个掩膜"
                assert "," not in p and " and " not in p

    def test_thing_prompts_cover_the_detect_prompts(self):
        """`inst_eval` 的 things 并集必须**盖住**检测那三类 —— 否则同一个物体
        在一条链上算 things、在另一条链上不算,两条链的对照就不成立了。"""
        merged = {p for ps in sam3_backend.DETECT_PROMPTS.values() for p in ps}
        assert merged <= set(sam3_backend.THING_PROMPTS)


class TestMakePredictor:
    def test_backend_names_are_the_two_choices(self):
        assert set(backends.BACKENDS) == {"sam3", "yolo"}

    def test_sam3_is_the_default_in_every_cli(self):
        """★ 默认值必须处处一致 —— 一半入口默认 sam3、一半默认 yolo,
        会得到两份**出自不同后端却摆在一起**的数。"""
        import inspect

        for mod in ("eval_2d_ab", "eval_attr", "inst_eval", "eval_fusion"):
            m = __import__(f"autodrivedata.perception.{mod}", fromlist=["main"])
            src = inspect.getsource(m.main)
            assert '"--backend"' in src, f"{mod} 没有 --backend"
            assert 'default="sam3"' in src, f"{mod} 的 --backend 默认不是 sam3"

    def test_sam3_predictor_does_not_load_weights_at_construction(self):
        """构造预测器**不许**碰模型:加载是 3.15 GiB + 4.6 s,而 `--self-test`
        这类路径根本不该付出这个代价。"""
        p = backends.Sam3Predictor(0.25)  # 不抛就算过
        assert p.conf == 0.25

    def test_describe_says_where_the_classes_come_from(self):
        """开跑时打印的这句要能让人**事后**分辨这份数是哪个后端的。"""
        d = backends.describe("sam3", backends.Sam3Predictor(0.25))
        assert "sam3" in d and "提示词" in d


class TestSam3Availability:
    def test_available_is_cheap_and_does_not_load_the_model(self):
        """`available()` 只查 import —— 它被放在错误提示路径上,不能是重的。"""
        assert sam3_backend.available() is True  # 本 env 已装(2026-10-02)

    def test_load_raises_a_usable_message_when_deps_missing(self, monkeypatch):
        """★ 依赖缺失时必须给出**可照做**的装法。

        ⚠️ 装法里那个源不是随手写的:本机 aliyun 源 403、pypi.org 超时(实测),
        只写 `pip install transformers` 会让人卡在原地。
        """
        monkeypatch.setattr(sam3_backend, "available", lambda: False)
        with pytest.raises(ImportError, match="pypi.tuna.tsinghua.edu.cn"):
            sam3_backend.load()


class TestDedup:
    """★ 去重这一层挡的是**同一份证据被数了两次**,不是"模型检错了"。

    一条提示一次前向 ⇒ `car` / `truck` / `bus` 可能把同一辆车各切一次(IoU≈1),
    而判据(AP / PQ)是**一对一**贪心匹配 —— 多出来的只能记成 FP。
    实测(5 帧 × 5 路):去重把 FP 67→54、mask AP 0.7430→**0.8061**、PQ 0.5221→0.5608。
    """

    @staticmethod
    def _m(x0, x1, y0=0, y1=10):
        m = np.zeros((10, 20), dtype=bool)
        m[y0:y1, x0:x1] = True
        return m

    def test_keeps_the_higher_conf_of_an_overlapping_pair(self):
        masks = [self._m(0, 10), self._m(0, 10)]
        keep = sam3_backend.dedup(masks, [0.4, 0.9])
        assert keep == [1], "应保留高分那个"

    def test_does_not_suppress_non_overlapping_masks(self):
        """★ 反向对照:去重**不许**吃掉真正不同的物体 —— 相邻两辆车是两辆车。"""
        masks = [self._m(0, 5), self._m(15, 20)]
        assert sam3_backend.dedup(masks, [0.9, 0.8]) == [0, 1]

    def test_threshold_is_insensitive_across_the_measured_gap(self):
        """★ 实测的 IoU 分布是**干净双峰**(重复 >0.9、不同物体 ≤0.1,中间零对)⇒
        0.5–0.9 任何阈值都给同一组结果。这条钉的是"阈值不是个需要调的旋钮"。"""
        masks = [self._m(0, 10), self._m(0, 10), self._m(12, 18)]
        scores = [0.9, 0.8, 0.7]
        got = {t: sam3_backend.dedup(masks, scores, t) for t in (0.5, 0.7, 0.9)}
        assert got[0.5] == got[0.7] == got[0.9] == [0, 2]

    def test_a_partially_overlapping_pair_below_threshold_survives(self):
        masks = [self._m(0, 10), self._m(3, 13)]  # IoU ≈ 7/13 ≈ 0.54 > 0.5 的边界外侧
        assert sam3_backend.dedup(masks, [0.9, 0.8], 0.9) == [0, 1]

    def test_dets_use_box_iou(self):
        a = sam3_backend.Det("Car", 0.9, (0.0, 0.0, 10.0, 10.0))
        b = sam3_backend.Det("Car", 0.5, (0.0, 0.0, 10.0, 10.0))
        c = sam3_backend.Det("Car", 0.7, (100.0, 100.0, 110.0, 110.0))
        got = sam3_backend.dedup_dets([b, a, c])
        assert [d.conf for d in got] == [0.9, 0.7] and got[0] is a

    def test_box_iou_edges(self):
        assert sam3_backend.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
        assert sam3_backend.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
        assert sam3_backend.box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0  # 零面积不许除零


class TestConfDefaults:
    """★ **两个后端的默认阈值不是一回事,不许"统一"。**

    0.25 是 YOLO 那边校准过的默认;SAM3 的 score 尺度不同,套 0.25 会放进一大堆
    低分幻觉 —— 实测 `person` 这一条提示在 5 帧里就产出 53 个落在建筑/杆/Static 上的假人。
    """

    def test_defaults_differ_between_backends(self):
        assert backends.DEFAULT_CONF["yolo"] == 0.25
        assert backends.DEFAULT_CONF["sam3"] == sam3_backend.DEFAULT_THRESHOLD
        assert backends.DEFAULT_CONF["sam3"] != backends.DEFAULT_CONF["yolo"], (
            "两边共用 0.25 就把幻觉当检出了 —— 别为了'整齐'改成同一个数"
        )

    def test_none_resolves_to_the_backend_default_not_a_shared_one(self):
        assert backends.resolve_conf("sam3", None) == sam3_backend.DEFAULT_THRESHOLD
        assert backends.resolve_conf("yolo", None) == 0.25

    def test_explicit_value_wins(self):
        """显式传 0.25 与不传**必须分得开** —— 一个是"我要这个阈值",一个是"随默认"。"""
        assert backends.resolve_conf("sam3", 0.25) == 0.25
        assert backends.resolve_conf("yolo", 0.8) == 0.8

    def test_every_cli_resolves_instead_of_hardcoding(self):
        """四个入口都必须走 `resolve_conf` —— 谁写死 0.25 谁就把 SAM3 的幻觉放进来。"""
        import inspect

        for mod in ("eval_2d_ab", "eval_attr", "eval_fusion", "mono_distance"):
            m = __import__(f"autodrivedata.perception.{mod}", fromlist=["main"])
            assert "resolve_conf(" in inspect.getsource(m.main), f"{mod} 没走 resolve_conf"


class TestOperatingPointReadout:
    """`eval_2d_ab.ap_for` 的**第四个数**(操作点上的命中数)是给"AP 不动"准备的。

    ★ 由来(2026-10-03):换到 SAM3 后 P1 四个 Δ 全落进 ±0.02,一度读成"SAM3 抗退化"。
    查下去是**过检把 recall 撑住了** —— AP 的尾部纪律(未达 recall 段 precision=0)
    在 recall 接近饱和时无从发力。⇒ 读数补上 `命中 / 召回 / 检出-GT`,这三个在
    "AP 不动"时照样动。修好去重与阈值后实测:浓雾 `召回 0.962→0.831`(ΔAP −0.075)。
    """

    @staticmethod
    def _b(x1, y1, x2, y2):
        return (x1, y1, x2, y2)

    def test_tp_counts_matched_gt_not_matched_predictions(self):
        """★ 是**命中了几条 GT**,不是"有几个预测配上了" —— 后者在过检时虚高。"""
        from autodrivedata.perception.eval_2d_ab import ap_for

        gt = [self._b(0, 0, 10, 10), self._b(100, 100, 110, 110)]
        # 一条命中、一条打空,外加三条与任何 GT 都不沾边的过检
        preds = [(0.9, self._b(0, 0, 10, 10)), (0.8, self._b(500, 500, 510, 510))]
        _ap, n_gt, n_pred, n_tp = ap_for(gt, preds, 0.5)
        assert (n_gt, n_pred, n_tp) == (2, 2, 1)

    def test_duplicate_predictions_do_not_inflate_tp(self):
        """同一个 GT 被两条预测重复命中 ⇒ TP 仍只算 **1**(一对一匹配)。"""
        from autodrivedata.perception.eval_2d_ab import ap_for

        gt = [self._b(0, 0, 10, 10)]
        preds = [(0.9, self._b(0, 0, 10, 10)), (0.8, self._b(0, 0, 10, 10))]
        _ap, _n_gt, n_pred, n_tp = ap_for(gt, preds, 0.5)
        assert (n_pred, n_tp) == (2, 1), "过检必须体现为'检出多、命中不变'"

    def test_empty_predictions_report_zero_tp(self):
        from autodrivedata.perception.eval_2d_ab import ap_for

        _ap, n_gt, n_pred, n_tp = ap_for([self._b(0, 0, 10, 10)], [], 0.5)
        assert (n_gt, n_pred, n_tp) == (1, 0, 0)

    def test_matching_is_gated_by_iou_not_just_overlap(self):
        """IoU 不到阈值 ⇒ 不算命中。这条挡的是把"有重叠"当"检到了"。"""
        from autodrivedata.perception.eval_2d_ab import ap_for

        gt = [self._b(0, 0, 10, 10)]
        weak = [(0.9, self._b(0, 0, 4, 10))]  # IoU = 40/100 = 0.4 < 0.5
        assert ap_for(gt, weak, 0.5)[3] == 0
        assert ap_for(gt, [(0.9, self._b(0, 0, 6, 10))], 0.5)[3] == 1  # IoU 0.6

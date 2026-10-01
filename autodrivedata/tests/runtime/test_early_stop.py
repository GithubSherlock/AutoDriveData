"""早停判据的回归钉(**纯值,不需要 torch/carla/GPU**)。

判据全是"序列 → 停不停 + 为什么",用合成序列就能钉死 —— 这是选纯值实现的直接好处:
不用真训一轮 9 小时才发现判据写反了。

**重点钉的四类**(前三类今天都真踩过或差一点踩到):

| 钉什么 | 为什么 |
|---|---|
| 持续下降 / 真平台 | 基本功:别在还降的时候停,也别在不降的时候不停 |
| **lr 归零的假平台分型** | 项目文档明写"400 epoch 跑完 lr ~1e-8,平台是 lr 归零不是收敛" —— 停是对的,但理由必须可分辨 |
| **warmup 期不判** | warmup 期 lr 线性升 ⇒ loss 必降,此时判平台无意义 |
| **复核缺失 ⇒ 不敢停** | 安全侧:判据做不了时宁可多跑,不可误停 |
"""

from __future__ import annotations

import pytest

from autodrivedata.map import train_maptr
from autodrivedata.map.mapvec import MAPTR_CLASSES
from autodrivedata.runtime.early_stop import EarlyStopper, PlateauDetector


def _run(det: PlateauDetector, losses: list[float], lr: float = 1e-4, start: int = 1) -> list:
    return [det.update(start + i, v, lr) for i, v in enumerate(losses)]


class TestPlateauDetector:
    def test_always_descending_never_plateaus(self):
        """单调下降(相对改善每次都达标)⇒ 一次都不判平台。"""
        det = PlateauDetector(patience=3, min_improve=1e-3)
        verdicts = _run(det, [10.0 * (0.9**i) for i in range(40)])
        assert not any(v.plateaued for v in verdicts)

    def test_flat_sequence_plateaus_after_patience(self):
        """持平 ⇒ 恰好 patience 个 stale 之后判平台(**不是**提前,也不是拖后)。"""
        det = PlateauDetector(patience=5, min_improve=1e-3)
        _run(det, [1.0])  # 建立 best
        verdicts = _run(det, [1.0] * 10, start=2)
        assert [v.plateaued for v in verdicts[:4]] == [False] * 4
        assert verdicts[4].plateaued, "第 patience 个 stale 就该判平台"

    def test_relative_not_absolute_threshold(self):
        """判据是**相对**改善 —— 同样幅度在不同量级上结论必须一致。

        绝对阈值在 loss=15(K>1 时序模型起步)与 loss=0.05(单帧锚点末期)上不可能同时合适,
        实测两个量级都真实存在(时序 15→2.68 / 单帧要求 →0)。
        """
        for scale in (15.0, 0.05, 1.0):
            det = PlateauDetector(patience=3, min_improve=1e-3)
            _run(det, [scale])
            # 每步降 1%(> min_improve)⇒ 不该判平台
            seq = [scale * (0.99**i) for i in range(1, 12)]
            assert not any(v.plateaued for v in _run(det, seq, start=2)), f"scale={scale}"

    def test_improvement_below_threshold_does_not_reset(self):
        """降幅小于 min_improve 的"改善"不算数 —— 否则噪声就能无限推迟早停。"""
        det = PlateauDetector(patience=3, min_improve=1e-2)
        _run(det, [1.0])
        seq = [1.0 - 1e-4 * i for i in range(1, 10)]  # 每步只降 0.01%,低于 1% 门槛
        assert any(v.plateaued for v in _run(det, seq, start=2))

    def test_lr_exhausted_is_a_distinct_reason(self):
        """★ lr 衰减到初始 1% 以下时的平台 ⇒ 理由必须是 `lr_exhausted`,不是 `converged`。

        两者的处置不同:前者该去**调 lr 计划重跑**,后者才是真收敛。混成一个理由,
        "为什么这次停了"就答不上来。
        """
        det = PlateauDetector(patience=3, min_improve=1e-3, lr_exhausted_frac=0.01)
        _run(det, [1e-4, 1e-4], lr=1e-4)  # 建立 lr_max
        _run(det, [1.0] * 3, lr=1e-9, start=3)  # lr 已衰到 1e-5 倍
        v = det.update(6, 1.0, 1e-9)
        assert v.plateaued and v.reason == "lr_exhausted"

    def test_converged_when_lr_still_healthy(self):
        """lr 没衰减(lr_halve 0 的长训档)时,平台就是 `converged`。"""
        det = PlateauDetector(patience=3, min_improve=1e-3)
        _run(det, [1.0] * 4, lr=1e-4)
        v = det.update(5, 1.0, 1e-4)
        assert v.plateaued and v.reason == "converged"

    def test_warmup_epochs_are_not_judged(self):
        """warmup 期 lr 线性升 ⇒ loss 必降,这段不该参与判平台(否则起步就被记 stale)。"""
        det = PlateauDetector(patience=2, min_improve=1e-3, warmup=5)
        verdicts = _run(det, [1.0] * 5)
        assert all(v.reason == "warmup" and not v.plateaued for v in verdicts)
        assert det.stale == 0


class TestEarlyStopper:
    """第二段:复核编排。**至少两次成功复核才可能判停**(第一次只建立基线)。

    驱动方式:先喂一个 best_loss,再连喂 `patience` 个持平值把状态推过平台线。
    `patience=3` ⇒ 第 3 个 stale 那一 epoch 进平台(实测口径,不是"差不多")。
    """

    PATIENCE = 3

    @classmethod
    def _stopper(cls, aps: list[float | None], **kw):
        """`confirm_fn` 按调用次序依次返回 `aps` 里的值(用完则重复最后一个)。"""
        calls: list[int] = []

        def confirm(epoch: int) -> float | None:
            i = min(len(calls), len(aps) - 1)
            calls.append(epoch)
            return aps[i]

        det = PlateauDetector(patience=cls.PATIENCE, min_improve=1e-3)
        return EarlyStopper(det, confirm, **kw), calls

    def _to_plateau(self, st, start: int = 2) -> int:
        """喂到**恰好进平台**那一 epoch,返回该 epoch。"""
        det = st.det
        det.update(1, 1.0, 1e-4)  # 建立 best
        ep = start
        for _ in range(self.PATIENCE - 1):
            assert not det.update(ep, 1.0, 1e-4).plateaued
            ep += 1
        assert det.update(ep, 1.0, 1e-4).plateaued, "第 patience 个 stale 必须进平台"
        return ep

    def test_first_confirm_only_establishes_baseline(self):
        """第一次复核**不停** —— 没有参照,"还在升吗"无从谈起。"""
        st, _ = self._stopper([0.30], confirm_every=1)
        ep = self._to_plateau(st)
        d = st.update(ep, 1.0, 1e-4)  # 同一 epoch 再判一次 → 这次平台+到节奏 → 复核
        assert d.should_stop is False and d.reason == "ap_still_rising"
        assert d.best_ap == 0.30

    def test_stops_when_ap_no_longer_improves(self):
        """第二次复核 AP 没升到门槛 ⇒ 停,理由沿用 loss 段的分型。"""
        st, _ = self._stopper([0.30, 0.301], confirm_every=1, ap_min_improve=1e-2)
        ep = self._to_plateau(st)
        assert st.update(ep, 1.0, 1e-4).should_stop is False  # 复核 1:建基线
        d = st.update(ep + 1, 1.0, 1e-4)  # 复核 2:没升到门槛
        assert d.should_stop and d.reason == "converged" and d.confirm_ap == pytest.approx(0.301)

    def test_keeps_running_while_ap_rises(self):
        """AP 仍在升(≥ 门槛)⇒ 不停,且 best_ap/best_ap_epoch 跟着走。"""
        st, _ = self._stopper([0.30, 0.35], confirm_every=1, ap_min_improve=1e-2)
        ep = self._to_plateau(st)
        st.update(ep, 1.0, 1e-4)
        d = st.update(ep + 1, 1.0, 1e-4)
        assert not d.should_stop and d.reason == "ap_still_rising"
        assert (d.best_ap, d.best_ap_epoch) == (0.35, ep + 1)

    def test_improvement_below_ap_threshold_counts_as_flat(self):
        """AP 只升 1e-3(< 门槛 1e-2)⇒ 判为平。

        **门槛必须远宽于 2e-3 的复现性下限**,否则判据会在噪声里打转。
        """
        st, _ = self._stopper([0.30, 0.301], confirm_every=1, ap_min_improve=1e-2)
        ep = self._to_plateau(st)
        st.update(ep, 1.0, 1e-4)
        assert st.update(ep + 1, 1.0, 1e-4).should_stop, "1e-3 的升幅低于 1e-2 门槛,应判平"

    def test_missing_confirm_never_stops(self):
        """★ 复核做不了(没配留出选择器 / 评估失败)⇒ **不敢停**。安全侧:宁可多跑。"""
        st, calls = self._stopper([None], confirm_every=1)
        ep = self._to_plateau(st)
        for k in range(4):
            d = st.update(ep + k, 1.0, 1e-4)
            assert not d.should_stop and d.reason == "confirm_unavailable"
        assert st.best_ap is None and len(calls) >= 2, "每次都该真的去试复核,不是只试一次"

    def test_confirm_is_throttled(self):
        """平台区内复核按 `confirm_every` 节流 —— 不然每个 epoch 花 90 s 评估。"""
        st, calls = self._stopper([0.30, 0.30, 0.30], confirm_every=5)
        ep = self._to_plateau(st)
        for k in range(12):
            st.update(ep + k, 1.0, 1e-4)
        assert len(calls) <= 3, f"13 个平台 epoch 最多复核 3 次(节流 5),实测 {len(calls)}"

    def test_no_confirm_before_plateau(self):
        """loss 还在降 ⇒ **一次复核都不做**(复核只在平台候选上触发)。"""
        st, calls = self._stopper([0.30], confirm_every=1)
        st.det.update(1, 10.0, 1e-4)
        for k in range(1, 10):
            st.update(1 + k, 10.0 * (0.9**k), 1e-4)  # 单调下降
        assert calls == [], "没进平台就评估 = 白烧 90 s/次"


class TestTrainHookExemptions:
    """`train_maptr._build_early_stopper` 的**三条不启用路径**。

    为什么值得单钉:这三条都是**静默不启用** —— 不钉住的话,将来某一条失效(比如豁免条件
    写反),表现为"这次怎么没早停",而日志里也看不出原因。它们各自对应一条真实约束:

    | 分支 | 为什么必须不启用 |
    |---|---|
    | 单帧锚点 | 那一段是**故意**跑到 loss→0 的正确性自证,早停会打断它 |
    | 无留出选择器 | 两段式缺第二段 ⇒ 退化成"只看 loss 平台",而那是设计上要避免的误停源 |
    | 显式 `--no-early-stop` | 用户意图优先 |
    """

    class _Args:
        """只需被读到的字段;不启用的分支在碰 model 之前就返回,故 model 传 None 即可。"""

        def __init__(self, **kw):
            self.early_stop = True
            self.eval_seg = self.eval_keep_in_seg = self.eval_exclude_seg = None
            self.out = "/tmp/unused.pt"
            self.patience, self.min_improve, self.confirm_every = 20, 1e-3, 20
            self.ap_min_improve, self.warmup = 1e-2, 0
            self.__dict__.update(kw)

    class _Rl:
        def __init__(self):
            self.notes: list[str] = []

        def note(self, msg: str) -> None:
            self.notes.append(msg)

    @staticmethod
    def _ds(n: int):
        class _D:
            def __len__(self) -> int:
                return n

        return _D()

    def _build(self, args, n_samples: int = 4):
        from autodrivedata.map.train_maptr import _build_early_stopper

        rl = self._Rl()
        return _build_early_stopper(args, self._ds(n_samples), None, rl), rl

    def test_single_frame_anchor_is_exempt(self):
        """★ 单帧锚点必须豁免 —— 否则会打断"故意跑到 loss→0"的正确性自证。"""
        # 豁免判据是 `len(ds) == 1`（见 is_single_frame_anchor 的 docstring）
        st, _ = self._build(self._Args(), n_samples=1)
        assert st is None
        st, _ = self._build(self._Args(eval_seg="seg4"), n_samples=1)
        assert st is None, "单帧锚点即使给了留出选择器也不该启用"

    def test_no_holdout_selector_disables_it(self):
        """没给 --eval-* ⇒ 无 AP 可复核 ⇒ 不启用,且**留一条 note**(事后可查)。"""
        st, rl = self._build(self._Args(eval_seg=None, eval_keep_in_seg=None, eval_exclude_seg=None))
        assert st is None and rl.notes, "不启用时应当留下可追溯的说明"

    def test_explicit_off_wins(self):
        st, _ = self._build(self._Args(early_stop=False, eval_seg="seg4"))
        assert st is None

    def test_enabled_when_both_halves_available(self):
        """三个条件都满足 ⇒ 真的建出 stopper(反向对照:证明上面三条不是恒返回 None)。"""
        st, _ = self._build(self._Args(eval_keep_in_seg="80:100", exclude_seg="seg4"), n_samples=312)
        assert st is not None and isinstance(st, EarlyStopper)


class TestHoldoutClassTags:
    """★ 留出 AP 的**逐类**数值必须落**时间序列**,而不只是 highlight。

    为什么:`runlog.highlight` 是**末次覆盖**(`self._highlights[k] = v`)—— `evaluate()`
    本来就写 `earlystop_AP/<类>`,而**每次复核都盖掉上一次** ⇒ 只剩最后一次,曲线在
    TensorBoard 里完全看不到。"整体 mAP 平台"完全可能是"一类到顶、另一类还在涨"互相抵消,
    不分类就看不出来 —— 而分类别看正是本项目 P1 主线的核心问题。
    """

    RES = {
        "mAP": 0.31,
        "aps": [0.40, 0.20, 0.35, 0.29],
        "classes": list(MAPTR_CLASSES),
        "n_pred": [1308, 210, 900, 640],
        "n_gt": [1200, 300, 880, 700],
    }

    def test_every_class_gets_ap_and_both_counts(self):
        tags = train_maptr.holdout_class_tags(self.RES)
        want = {f"confirm_{k}/{c}" for k in ("AP", "n_pred", "n_gt") for c in MAPTR_CLASSES}
        assert set(tags) == want, f"标签集不对:\n  多 {set(tags) - want}\n  少 {want - set(tags)}"
        assert tags["confirm_AP/divider"] == pytest.approx(0.40)
        assert tags["confirm_AP/centerline"] == pytest.approx(0.29)
        assert tags["confirm_n_pred/divider"] == 1308 and tags["confirm_n_gt/ped_crossing"] == 300

    def test_empty_result_is_spread_safe(self):
        """`res` 缺失/为空时返回空字典 ⇒ 调用方 `**tags` 摊开必须是安全的(不 KeyError)。"""
        for empty in ({}, {"classes": []}, {"classes": None}):
            assert train_maptr.holdout_class_tags(empty) == {}

    def test_length_mismatch_is_loud_not_misattributed(self):
        """★ 逐类数组长度不一致 ⇒ **当场炸**,不许静默错位。

        静默错位会把 `boundary` 的 AP 记到 `ped_crossing` 名下 —— 那种错**图上完全看不出来**
        (两条曲线都平滑、都在合理量级),却会让"哪一类是瓶颈"的判断整个反过来。
        """
        bad = dict(self.RES, aps=self.RES["aps"][:3])
        with pytest.raises(ValueError):
            train_maptr.holdout_class_tags(bad)

    def test_confirm_emits_the_series(self, monkeypatch, tmp_path):
        """端到端:复核跑完 ⇒ **真的**有一条带逐类标签的 metric 行。

        这条是防"函数写好了但没人调"的唯一判据 —— `holdout_class_tags` 自己测得多好都没用,
        只要 `confirm()` 里那行没接上,曲线就是空的。
        """
        from pathlib import Path

        monkeypatch.setattr(train_maptr, "save_map_checkpoint", lambda m, p: Path(p).write_text("x"))
        monkeypatch.setattr(train_maptr, "evaluate", lambda **kw: dict(self.RES))

        args = _Args(eval_keep_in_seg="80:100", out=str(tmp_path / "m.pt"))
        rl = _MetricRl()
        st = train_maptr._build_early_stopper(args, _Ds(312), None, rl)
        assert st is not None

        ap = st.confirm_fn(20)
        assert ap == pytest.approx(0.31)
        assert len(rl.rows) == 1, f"一次复核发一行,实测 {len(rl.rows)}"
        step, kv = rl.rows[0]
        assert step == 20
        assert kv["confirm_AP/divider"] == pytest.approx(0.40)
        assert kv["confirm_n_gt/centerline"] == 700

    def test_confirm_failure_emits_nothing(self, monkeypatch, tmp_path):
        """复核失败 ⇒ 返回 None 且**一行都不发** —— 空的/假的曲线比没有更坏。"""
        from pathlib import Path

        def boom(**kw):
            raise RuntimeError("仿真失败")

        monkeypatch.setattr(train_maptr, "save_map_checkpoint", lambda m, p: Path(p).write_text("x"))
        monkeypatch.setattr(train_maptr, "evaluate", boom)
        args = _Args(eval_keep_in_seg="80:100", out=str(tmp_path / "m.pt"))
        rl = _MetricRl()
        st = train_maptr._build_early_stopper(args, _Ds(312), None, rl)
        assert st is not None and st.confirm_fn(20) is None and rl.rows == []


class _MetricRl:
    """只记 `metric` 的桩(与 `TestTrainHookExemptions._Rl` 分开:那个只记 note)。"""

    def __init__(self) -> None:
        self.rows: list[tuple[int, dict]] = []

    def metric(self, step: int, **kv) -> None:
        self.rows.append((step, kv))

    def note(self, msg: str) -> None:  # `_build_early_stopper` 的不启用分支会调
        pass


class _Args:
    """`_build_early_stopper` 读的**全部**字段。

    `infos`/`root`/`eval_*` 只有**真走到 `confirm()` 里**才会被读(关键字实参在调用前求值)
    ⇒ 上面那组只测豁免分支的用不到,而 `TestHoldoutClassTags` 必须给全 ——
    缺了会以 `AttributeError` **假通过**在 `except Exception` 里(实测踩到:
    `test_confirm_emits_the_series` 报 None,看不出是"没接上"还是"夹具缺字段")。
    """

    def __init__(self, **kw):
        self.early_stop = True
        self.eval_seg = self.eval_keep_in_seg = self.eval_exclude_seg = None
        self.out = "/tmp/unused.pt"
        self.patience, self.min_improve, self.confirm_every = 20, 1e-3, 20
        self.ap_min_improve, self.warmup = 1e-2, 0
        self.infos, self.root = "infos.json", "root"
        self.eval_frames, self.eval_score_thr, self.temporal_window = None, 0.2, 1
        self.__dict__.update(kw)


class _Ds:
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

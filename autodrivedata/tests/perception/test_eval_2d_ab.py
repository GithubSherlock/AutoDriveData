"""`perception/eval_2d_ab` 的判据回归钉 —— 重点是**AP 的台阶分辨率**。

## 这一层为什么值得钉

2026-10-06 实测:`edit/degrade` 的 `blur` 在 COCO 后端下"不单调"(−0.013 / **+0.055** / −0.034)。
查下去**不是模糊的性质,是 11 点插值的台阶**:

    单条边界检测(conf 0.360、bestIoU 0.540)把 `n_tp` 从 107 顶到 108,
    而 `108/119 = 0.908 ≥ 0.90` ⇒ 第 10 个 recall 格点**从"不可达"变成"可达"**,
    那一格的 precision 从 0 跳成 0.857 ⇒ **ΔAP 一步 `+0.078`**。

而**同一个翻转**在真实精度曲线上只值 `p/n_gt`(recall 轴挪了 `1/n_gt`)——
⇒ **11 点插值的放大倍数 ≈ `n_gt/11`**。`n_gt = 119` 时 ≈ 11×。

⇒ 本文件钉三件事:① 悬崖**位置**算得对(`grid_cliff`);
   ② 放大倍数**随 `n_gt` 增长**(合成对照,`n_gt=10` 时两者一致、`n_gt=110` 时差 ~10×);
   ③ `evaluate` 的**明细与脆弱告警**能真的触发。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.perception.eval_2d_ab import (
    AP_GRID_STEP,
    ap_for,
    evaluate,
    fragile_classes,
    grid_cliff,
)


def _gt(n, x=0.0):
    """n 个互不重叠的 100×100 框(间距 200 ⇒ 一个预测最多配上一个)。"""
    return [(x + 200 * i, 0.0, x + 200 * i + 100, 100.0) for i in range(n)]


def _ap_dense(gt, preds, n_points: int, thr: float = 0.5):
    """测试内**独立**实现(故意不复用 `ap_for`):同一批检测换格点数。尾行纪律一致。"""
    from autodrivedata.perception.attribution import box_iou2d

    preds = sorted(preds, key=lambda t: -t[0])
    matched = [False] * len(gt)
    tp = []
    for _, box in preds:
        bi, bv = -1, 0.0
        for j, g in enumerate(gt):
            if matched[j]:
                continue
            v = box_iou2d(g, box)
            if v > bv:
                bi, bv = j, v
        ok = bi >= 0 and bv >= thr
        if ok:
            matched[bi] = True
        tp.append(ok)
    if not preds:
        return 0.0
    ctp = np.cumsum(np.array(tp, float))
    rec = ctp / len(gt)
    pre = ctp / np.arange(1, len(tp) + 1)
    ps = [float(pre[rec >= r].max()) if (rec >= r).any() else 0.0 for r in np.linspace(0, 1, n_points)]
    return float(np.mean(ps))


class TestGridCliff:
    """★ 悬崖位置:`T_j = ceil(j/10·n_gt)`,`n_tp` 每跨过它,那一格才"亮起"。"""

    def test_matches_the_measured_case(self):
        """实测那一点:`n_gt=119` ⇒ 第 10 格(recall 0.9)的门槛是 **108**。

        `107/119 = 0.899 < 0.9`(差一点)、`108/119 = 0.908 ≥ 0.9`(过线)——
        这就是那次 `blur` 从 −0.026 翻成 +0.055 的全部原因。
        """
        assert grid_cliff(119, 106) == (8, 2)
        assert grid_cliff(119, 107) == (8, 1)
        assert grid_cliff(119, 108) == (9, 119 - 108)
        assert grid_cliff(119, 119) == (10, None)

    def test_no_gt_is_undecidable(self):
        """没有 GT 的类 **不可判**(不是通过、也不是不通过)。"""
        assert grid_cliff(0, 5) == (-1, None)
        assert grid_cliff(-1, 0) == (-1, None)

    def test_margin_shrinks_then_jumps(self):
        """★ 余量随 `n_tp` 单调减小,跨格后**重置到满** —— 这是"台阶"的可测形态。"""
        ups = []
        for tp in range(100, 108):
            _, up = grid_cliff(119, tp)
            assert up is not None, "100..107 都还没满格"
            ups.append(up)
        assert ups == sorted(ups, reverse=True), f"余量应当单调减小,实测 {ups}"
        assert ups[-1] == 1, "顶到 107 时应当只剩 1 个 TP"

    def test_tiny_gt_set_is_always_on_a_cliff(self):
        """GT 少到 `T_j` 密成一格一个 ⇒ **每一步都在悬崖上**(余量恒为 1)。"""
        assert all(grid_cliff(2, tp)[1] == 1 for tp in range(0, 2))


class TestApForTailDiscipline:
    """尾行纪律:recall 未达 1 的段 precision=0(低 recall 不注水)。"""

    def test_perfect_is_one(self):
        g = _gt(4)
        ap, n_gt, n_pred, n_tp = ap_for(g, [(0.9, b) for b in g], 0.5)
        assert (n_gt, n_pred, n_tp) == (4, 4, 4)
        assert ap == pytest.approx(1.0)  # 累加 11 次浮点 ⇒ 1.0000000000000002

    def test_no_prediction_is_zero(self):
        assert ap_for(_gt(4), [], 0.5)[0] == 0.0

    def test_half_recall_reports_the_coarse_grid_not_half(self):
        """配上一半 ⇒ AP 是 `6/11`,**不是 0.5** —— 粗格本身就有 0.045 的偏。

        （11 个格点里 6 个落进"可 reach"段 ⇒ `6/11 = 0.545`。这条顺带说明:
        连"一半 recall 值多少分"这种直觉都被格点改写。）
        """
        g = _gt(4)
        ap, _, _, n_tp = ap_for(g, [(0.9, b) for b in g[:2]], 0.5)
        assert n_tp == 2
        assert ap == pytest.approx(6 / 11, abs=0.01), f"实测 {ap}"


class TestStepAmplification:
    """★★ **反向自证**:台阶不是理论,而且它的**放大倍数 = `n_gt/11`**。

    造一对只差**一条边界检测**的预测(`n_tp = T_j−1` vs `T_j`,**`n_pred` 两边相同**),
    在 `n_gt = 10` 与 `n_gt = 110` 上各做一次:

    - 11 点插值下**两次都恰好跳 `1/11`**(台阶与 `n_gt` 无关);
    - 密插值下 `n_gt=10` 跳 ~`1/10`、`n_gt=110` 只跳 ~`1/110`。
    ⇒ **GT 越多,11 点插值越是凭空放大**。这正是本项目那批 `n_gt≈119` 读数的处境。
    """

    @staticmethod
    def _pair(n_gt):
        """`(gt, 命中版, 落空版)` —— 边界那条偏 20 px(IoU 0.667,配上)或 60 px(IoU 0.25,落空)。"""
        g = _gt(n_gt)
        t = -(-9 * n_gt // 10)  # ceil(0.9·n_gt) = 第 10 格的 TP 门槛
        base = [(0.99 - 1e-3 * i, b) for i, b in enumerate(g[: t - 1])]
        x0 = g[t - 1][0]
        ok = base + [(0.5, (x0 + 20.0, 0.0, x0 + 120.0, 100.0))]
        no = base + [(0.5, (x0 + 60.0, 0.0, x0 + 160.0, 100.0))]
        return g, ok, no

    @pytest.mark.parametrize("n_gt", [10, 110])
    def test_flip_is_always_exactly_one_grid_point_at_11(self, n_gt):
        g, ok, no = self._pair(n_gt)
        assert ap_for(g, ok, 0.5)[2] == ap_for(g, no, 0.5)[2], "两次的 n_pred 必须相同"
        assert ap_for(g, ok, 0.5)[3] - ap_for(g, no, 0.5)[3] == 1, "只应当多出 1 个命中"
        d = ap_for(g, ok, 0.5)[0] - ap_for(g, no, 0.5)[0]
        assert d == pytest.approx(AP_GRID_STEP, abs=1e-9), f"11 点下应当恰好跳 1/11,实测 {d}"

    def test_dense_interpolation_scales_with_gt_count(self):
        """★ 同一个翻转,密插值下的量级**随 `n_gt` 变小** —— 放大倍数就是 `n_gt/11`。"""
        d = {}
        for n_gt in (10, 110):
            g, ok, no = self._pair(n_gt)
            d[n_gt] = _ap_dense(g, ok, 101) - _ap_dense(g, no, 101)
        assert d[110] < d[10] / 5, f"GT 多 11 倍,密插值下的同一个翻转应当小 ~10 倍,实测 {d}"
        assert d[10] == pytest.approx(1 / 11, abs=0.02), f"n_gt=10 时两者应当一致,实测 {d[10]}"

    def test_the_amplification_is_the_gt_count_over_11(self):
        """把"放大倍数"直接量出来:`11 点跳幅 / 密插值跳幅 ≈ n_gt/11`。"""
        g, ok, no = self._pair(110)
        s11 = ap_for(g, ok, 0.5)[0] - ap_for(g, no, 0.5)[0]
        s101 = _ap_dense(g, ok, 101) - _ap_dense(g, no, 101)
        assert s11 / s101 == pytest.approx(110 / 11, rel=0.15), f"实测 {s11 / s101:.1f}×"


def _fake_root(tmp_path, n_frames=2, n_gt=3):
    """极小 KITTI root:**图是真 PNG**(`detect` 会打开它),GT 是真 `label_2`。"""
    from PIL import Image

    img_dir = tmp_path / "training" / "image_2"
    lab_dir = tmp_path / "training" / "label_2"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    boxes = _gt(n_gt)
    for i in range(n_frames):
        Image.fromarray(np.full((512, 512, 3), 128, np.uint8)).save(img_dir / f"{i:06d}.png")
        lab_dir.joinpath(f"{i:06d}.txt").write_text(
            "".join(
                f"Car 0 0 0 {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} 1 1 1 1 1 1 1\n" for x1, y1, x2, y2 in boxes
            )
        )
    return tmp_path, boxes


class TestEvaluate:
    """`evaluate` 返回明细 —— `noise_curve` 靠它拿分辨率信息。"""

    def test_reports_cliff_margin_per_class(self, tmp_path):
        root, boxes = _fake_root(tmp_path, n_frames=2, n_gt=3)
        ev = evaluate(root, lambda path: [("Car", 0.9, b) for b in boxes], 0.5, 0.5, None, verbose=False)
        assert ev["n_classes"] == 1  # 只有 Car 有 GT
        car = ev["classes"]["Car"]
        assert (car["n_gt"], car["n_tp"]) == (6, 6)
        assert car["ap"] == pytest.approx(1.0)
        assert (car["cliff_top"], car["cliff_up"]) == (10, None), "满格 ⇒ 无悬崖"
        assert fragile_classes(ev) == []

    def test_fragile_class_is_flagged(self, tmp_path):
        """★★ 造一个**卡在悬崖上**的点(6 GT、3 TP)⇒ 必须被喊出来。

        这不是"检查一个布尔",是**证明这个提醒在该响的时候会响** ——
        否则 `fragile_classes` 永远返回空也能"通过"。
        """
        root, boxes = _fake_root(tmp_path, n_frames=2, n_gt=3)
        # 只在第 0 帧出框 ⇒ det 3 条、GT 6 条(两帧各 3 条,`load_gt` 是池化的)
        ev = evaluate(
            root,
            lambda path: [("Car", 0.9, b) for b in boxes] if path.stem == "000000" else [],
            0.5,
            0.5,
            None,
            verbose=False,
        )
        car = ev["classes"]["Car"]
        assert (car["n_gt"], car["n_tp"]) == (6, 3)
        assert car["cliff_up"] == 1, f"距 recall 0.6 的格点应当差 1 个 TP(tp=3、T_6=4),实测 {car}"
        assert fragile_classes(ev) == [("Car", 1)]

    def test_no_gt_class_is_never_flagged(self, tmp_path):
        """没有 GT 的类 ⇒ `cliff_up=None` ⇒ **不可判** —— 既不算脆、也不能算通过。"""
        root, _ = _fake_root(tmp_path, n_frames=1, n_gt=2)
        ev = evaluate(root, lambda path: [], 0.5, 0.5, None, verbose=False)
        assert ev["classes"]["Pedestrian"]["cliff_up"] is None
        assert "Pedestrian" not in [c for c, _ in fragile_classes(ev)]
        assert ev["mAP"] == 0.0

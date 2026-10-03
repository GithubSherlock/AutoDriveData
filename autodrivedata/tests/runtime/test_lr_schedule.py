"""学习率调度的回归钉(纯值,零 torch)。

## 钉的是什么

`train_maptr` 原来只有"每 N epoch 减半"和"完全不衰减"两种形态,而 **MapQR 官方配方是
AdamW 6e-4 + warmup + CosineAnnealing** —— 两者都不是。§P-V25 八 的四档消融跑在
**恒定 1e-4** 下,而变体末段 loss 的降幅是基线的 **4.6×** ⇒「架构更差」与「没训完」分不开。
这次补 `cosine` 就是为了把这两件事拆开。

**最大的一条是向后兼容**:`halve` 是默认,且 `warmup=0`(默认)时**逐位等于改造前的表达式**。
已归档的 13 次训练全是 `--lr-halve 0`,改错了它们就全部作废。
"""

from __future__ import annotations

import math

import pytest

from autodrivedata.runtime.lr_schedule import PAPER_MIN_LR_RATIO, lr_at, warmup_mult


def _old_halve(epoch: int, base: float, lr_halve: int) -> float:
    """**改造前**的表达式(逐字抄自当时的训练循环)。作向后兼容的对照基准。"""
    lr = base * (0.5 ** ((epoch - 1) // lr_halve)) if lr_halve > 0 else base
    return lr


class TestHalveBackCompat:
    def test_matches_the_old_expression_bit_for_bit(self):
        """★ 默认路径不许动:13 次归档训练用的都是它,差一点就让那些数作废。"""
        for base in (1e-4, 6e-4):
            for lr_halve in (0, 8, 12, 100):
                for e in range(1, 129):
                    assert lr_at(e, base=base, epochs=128, lr_halve=lr_halve) == _old_halve(
                        e, base, lr_halve
                    ), f"base={base} lr_halve={lr_halve} ep={e}"

    def test_zero_lr_halve_is_constant(self):
        """`--lr-halve 0` = 不衰减 —— §P-V11/§P-V25 四档用的就是这个。"""
        assert {lr_at(e, base=1e-4, epochs=128, lr_halve=0) for e in (1, 64, 128)} == {1e-4}

    def test_steps_at_the_right_epochs(self):
        """减半点:第 1 与第 12 帧同值,第 13 帧才落到一半(边界是 `(epoch-1)//N`)。"""
        g = lambda e: lr_at(e, base=1e-4, epochs=128, lr_halve=12)  # noqa: E731
        assert g(1) == g(12) == 1e-4
        assert g(13) == g(24) == pytest.approx(5e-5)


class TestCosine:
    def test_starts_at_base_right_after_warmup(self):
        """★ 退火**从 warmup 结束处拿到峰值**,不是从 epoch 1 —— 否则等于没 warmup。"""
        kw = dict(base=6e-4, epochs=128, schedule="cosine", warmup=5, warmup_ratio=1 / 3)
        assert lr_at(6, **kw) == pytest.approx(6e-4)

    def test_without_warmup_starts_at_base(self):
        """无 warmup 时第一帧就是峰值(mmcv 是 0-based iter,故用 `epoch-1`)。"""
        assert lr_at(1, base=6e-4, epochs=128, schedule="cosine") == pytest.approx(6e-4)

    def test_approaches_the_floor_and_never_reaches_zero(self):
        """★ 退火**趋近** `base·min_lr_ratio`,且末帧严格大于 0。

        ⚠️ 末帧**不恰好等于** floor,这不是 bug 而是**逐 epoch 取值**的必然:每个 epoch 只取
        一个 lr,取的是**该 epoch 开头**的进度 ⇒ 最后一帧是 `prog = 127/128`,
        比 floor 高一点点。官方按 iteration 走,是在**最后一个 iter** 才落到 floor。
        下面用 `epochs+1` 验极限(`prog` 被 `min(...,1.0)` 封顶),两条一起才说明问题:
        极限对 + 末帧不为 0(给成 0 会让末段完全没有梯度)。
        """
        floor = 6e-4 * PAPER_MIN_LR_RATIO
        last = lr_at(128, base=6e-4, epochs=128, schedule="cosine")
        assert lr_at(129, base=6e-4, epochs=128, schedule="cosine") == pytest.approx(floor)
        assert floor < last < floor * 2, f"末帧 {last:.3e} 应贴近但高于 floor {floor:.3e}"
        assert last > 0.0

    def test_monotone_non_increasing_after_warmup(self):
        """★ warmup 之后**单调不增** —— 中间反弹说明 cos 的相位写错了。"""
        vals = [
            lr_at(e, base=6e-4, epochs=128, schedule="cosine", warmup=5, warmup_ratio=1 / 3)
            for e in range(6, 129)
        ]
        assert all(a >= b for a, b in zip(vals, vals[1:], strict=False)), "退火段出现回升"

    def test_warmup_段_starts_at_the_ratio_and_rises(self):
        """★ warmup 起点恰为 `base·ratio`(mmcv 语义),且严格上升 —— 起点写成 0 会浪费掉
        ratio 想买的那点初始梯度。"""
        kw = dict(base=6e-4, epochs=128, schedule="cosine", warmup=5, warmup_ratio=1 / 3)
        assert lr_at(1, **kw) == pytest.approx(6e-4 / 3)
        vals = [lr_at(e, **kw) for e in range(1, 6)]
        assert all(a < b for a, b in zip(vals, vals[1:], strict=False))
        assert vals[-1] < 6e-4, "warmup 末帧应仍低于峰值(mmcv 如此,不凑成 1.0)"

    def test_never_below_the_floor(self):
        """整条曲线不低于 `base·min_lr_ratio`,含 warmup 段之后的每一个 epoch。"""
        floor = 6e-4 * PAPER_MIN_LR_RATIO
        for e in range(1, 129):
            v = lr_at(e, base=6e-4, epochs=128, schedule="cosine", warmup=5, warmup_ratio=1 / 3)
            assert v >= floor - 1e-18

    def test_warmup_longer_than_training_does_not_divide_by_zero(self):
        """退化输入(warmup ≥ epochs)不许抛除零 —— 采集脚本里长度是参数化的。"""
        v = lr_at(3, base=1e-4, epochs=2, schedule="cosine", warmup=10, warmup_ratio=0.0)
        assert math.isfinite(v) and v > 0.0


class TestWarmupMult:
    def test_outside_the_window_is_one(self):
        assert warmup_mult(99, 5, 1 / 3) == 1.0
        assert warmup_mult(1, 0, 1 / 3) == 1.0

    def test_ratio_zero_reproduces_a_pure_linear_ramp(self):
        """ratio=0 ⇒ 从 0 线性升 —— 这是 `--warmup-ratio` 默认值的语义。"""
        assert warmup_mult(1, 5, 0.0) == 0.0
        assert warmup_mult(5, 5, 0.0) == pytest.approx(0.8)


class TestValidation:
    def test_unknown_schedule_raises(self):
        """调度名拼错必须报错 —— 静默退回默认会让"换了配方"悄悄没生效。"""
        with pytest.raises(ValueError, match="未知调度"):
            lr_at(1, base=1e-4, epochs=128, schedule="cosin")

    def test_epoch_is_one_based(self):
        with pytest.raises(ValueError, match="1-based"):
            lr_at(0, base=1e-4, epochs=128)

    def test_non_positive_base_raises(self):
        with pytest.raises(ValueError, match="必须 > 0"):
            lr_at(1, base=0.0, epochs=128)

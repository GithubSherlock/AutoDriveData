"""`edit/video_edit` 的判据回归钉。

这一层的核心是**尺子必须先被证明能测出时序** —— 否则"不闪"和"尺子坏了"长得一样,
而"生成视频不闪"是个极容易被接受的结论。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.edit import video_edit as V


def _frame(shift: int, h=128, w=128):
    """**平移过的平滑随机纹理** —— 造一条**运动量可控**的序列。

    ⚠️ **两版夹具都错了,都留在这里当反例**(它们都让"尺子坏了"伪装成"结果如此"):
    ① 周期 4 的条纹 + `shift=8`(整 2 个周期)⇒ 画面**逐位不变**,读数 **0.000**;
    ② 单根亮条 ⇒ 梯度图**只有两条非零边**,一平移秩相关直接掉到 0 ⇒ 指标**饱和到 1.0**,
       "打乱更大" 于是**永远不成立**。
    ⇒ 只有**稠密且空间相关**的纹理才有连续可比的秩。真实街景本来就是这一类。
    """
    import cv2

    rng = np.random.default_rng(0)
    tex = rng.integers(0, 255, size=(h, w)).astype(np.float32)
    tex = cv2.blur(tex, (15, 15))
    v = np.clip(np.roll(tex, shift, axis=1), 0, 255).astype(np.uint8)
    return np.stack([v] * 3, axis=2)


class TestDescriptor:
    def test_deterministic(self):
        a = _frame(0)
        np.testing.assert_array_equal(V.structure_descriptor(a), V.structure_descriptor(a))

    def test_invariant_to_brightness_offset(self):
        """★ 描述子对**整体亮度平移**不敏感 —— 这正是它能拿来跨退化比的原因。

        雾/夜色都会改整体亮度;若描述子跟着变,量到的就是"亮度差"而不是"结构跳变"。
        """
        a = _frame(0)
        b = np.clip(a.astype(np.int16) + 40, 0, 255).astype(np.uint8)
        d = V.adjacent_diffs([V.structure_descriptor(a), V.structure_descriptor(b)])[0]
        assert d < 0.35, f"整体提亮不该被读成大结构跳变,实测 {d:.3f}"

    def test_detects_a_real_shift(self):
        """反向自证:真的平移了必须读出来(否则尺子对运动无感,"不闪"是假的)。"""
        d = V.adjacent_diffs([V.structure_descriptor(_frame(0)), V.structure_descriptor(_frame(20))])[0]
        assert d > 0.5, f"明显平移应当读出大跳变,实测 {d:.3f}"


class TestAdjacentDiffs:
    def test_identical_sequence_is_zero(self):
        f = _frame(0)
        assert V.adjacent_diffs([V.structure_descriptor(f)] * 3) == [0.0, 0.0]

    def test_metric_is_monotone_in_motion(self):
        """★★ **尺子对"运动量"必须单调** —— 这是它能测闪烁的前提。

        实测(平滑纹理,step 1/2/4/8):`0.28 → 0.59 → 0.75 → 0.94`。
        ⚠️ 这条比"打乱比值"更有力:上一版把两条都压在一条比值断言上,
        而**比值最高只有 2.2×**(梯度图动态范围被并列压扁)⇒ 那条断言永远红。
        """
        meds = [
            float(np.median(V.adjacent_diffs([V.structure_descriptor(_frame(i * s)) for i in range(8)])))
            for s in (1, 2, 4, 8)
        ]
        assert meds == sorted(meds), f"跳变应当随步长单调上升,实测 {meds}"
        assert meds[-1] > 3 * meds[0], f"最大步长应远大于最小步长,实测 {meds}"

    def test_shuffle_raises_the_metric(self):
        """★ **自证**:打乱后跳变必须明显变大 —— 这是"尺子看得出**顺序**"的直接证据。

        ⚠️ 第一版用的是**全同帧**的静态序列 —— 那问的是个空问题:8 张一模一样的图,
        打乱当然还是 0.000。⇒ 夹具必须是"**每帧都不同、但相邻步长很小**"的序列。

        ⚠️ 断言用 **1.5×** 不是 3×:实测 step=1 时是 **2.1×**(blur 15/31/51 三档都是),
        而那已经接近这把尺子的比值上限(见 `SHUF_FLOOR` 头注:动态范围被并列压扁)。
        **把阈值写成 3× 等于让这条判据永远红**,然后被人调绿 —— 那才是真的坏事。
        """
        seq = [V.structure_descriptor(_frame(i)) for i in range(8)]
        d_ord = float(np.median(V.adjacent_diffs(seq)))
        d_shf = float(np.median(V.adjacent_diffs(V.shuffled(seq, seed=0))))
        assert d_shf > 1.5 * d_ord, f"打乱应当明显更大,实测 有序 {d_ord:.3f} vs 打乱 {d_shf:.3f}"

    def test_short_input(self):
        assert V.adjacent_diffs([]) == []
        assert V.adjacent_diffs([np.zeros(4)]) == []


class TestVerdict:
    """三分支:尺子失效 / 不闪 / 闪。**别把第一种读成第二种。**"""

    def test_metric_self_failure(self):
        v = V.verdict({"d_in_median": 0.2, "d_out_median": 0.3, "d_shuf_median": 0.40})
        assert "失效" in v

    def test_consistent(self):
        v = V.verdict({"d_in_median": 0.2, "d_out_median": 0.25, "d_shuf_median": 0.9})
        assert "时序一致" in v

    def test_flicker(self):
        v = V.verdict({"d_in_median": 0.1, "d_out_median": 0.6, "d_shuf_median": 0.9})
        assert "闪烁" in v

    def test_blind_ruler_is_caught_by_an_absolute_floor(self):
        """★ 尺子自证走**绝对下限**,不走比值 —— 比值在梯度图上最高只到 2.2×(见 `SHUF_FLOOR`)。"""
        v = V.verdict({"d_in_median": 0.1, "d_out_median": 0.3, "d_shuf_median": 0.30})
        assert "失效" in v

    def test_floor_is_the_truth_sequence_not_zero(self):
        """★ 与真值序列比、**不与 0 比** —— 场景自己在动,`d_out=0.2` 在 `d_in=0.19` 时不算闪。"""
        assert "时序一致" in V.verdict({"d_in_median": 0.19, "d_out_median": 0.20, "d_shuf_median": 0.9})


class TestCompareSequences:
    def test_length_mismatch_raises(self, tmp_path):
        from PIL import Image

        for i in range(4):
            Image.fromarray(_frame(i)).save(tmp_path / f"a{i:06d}.png")
            Image.fromarray(_frame(i)).save(tmp_path / f"b{i:06d}.png")
        truth = sorted(tmp_path.glob("a*.png"))
        gen = sorted(tmp_path.glob("b*.png"))[:3]
        with pytest.raises(SystemExit, match="帧数不等"):
            V.compare_sequences(truth, gen)

    def test_too_short_raises(self, tmp_path):
        from PIL import Image

        for i in range(2):
            Image.fromarray(_frame(i)).save(tmp_path / f"a{i:06d}.png")
            Image.fromarray(_frame(i)).save(tmp_path / f"b{i:06d}.png")
        with pytest.raises(SystemExit, match="少于 3 帧"):
            V.compare_sequences(sorted(tmp_path.glob("a*.png")), sorted(tmp_path.glob("b*.png")))

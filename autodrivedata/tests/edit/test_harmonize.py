"""`edit/harmonize` 的判据回归钉。

这一格**没有真值靶**,所以判据是**代理** —— 代理指标的风险是"它对什么都说好"。
⇒ 要用测试钉住:**它对本方法(Reinhard)的改善敏感**,且**与对照臂分得开**。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.edit import harmonize as H


def _img(mean, spread=20, h=32, w=32, seed=0):
    """造一张均值可控的彩色图(用固定随机底 + 平移)。"""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h, w, 3)).astype(np.float32)
    base = (base - base.mean()) * (spread / max(base.std(), 1e-6)) + np.asarray(mean, np.float32)
    return np.clip(base, 0, 255).astype(np.uint8)


class TestLabStats:
    def test_constant_image_stats(self):
        m, c = H.lab_stats(np.full((8, 8, 3), 128, np.uint8))
        assert c.max() < 1e-6, "常量图协方差应当为 0"
        assert m.shape == (3,)

    def test_mean_tracks_brightness(self):
        """L 通道随亮度单调 —— 不会单调的话整个"光影冲突"就没在量。"""
        ls = [float(H.lab_stats(np.full((8, 8, 3), v, np.uint8))[0][0]) for v in (30, 128, 220)]
        assert ls == sorted(ls)

    def test_non_rgb_raises(self):
        with pytest.raises(ValueError, match=r"\(H,W,3\)"):
            H.to_lab(np.zeros((4, 4), np.uint8))


class TestStatDistance:
    def test_self_distance_is_zero(self):
        a = _img((128, 128, 128))
        assert H.stat_distance(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_grows_with_mismatch(self):
        """★ 尺子的基本要求:差得越多读数越大(否则它分不出"修好了"与"没修")。"""
        ref = _img((128, 128, 128), seed=1)
        near = _img((140, 128, 128), seed=1)
        far = _img((220, 60, 60), seed=1)
        assert H.stat_distance(near, ref) < H.stat_distance(far, ref)

    def test_cov_term_is_normalized(self):
        """协方差项必须归一 —— 否则它的量纲(σ²)会盖过均值项(第一版就是这么设的)。"""
        low = _img((128, 128, 128), spread=5, seed=2)
        high = _img((128, 128, 128), spread=80, seed=2)
        # 均值几乎相同、方差差 16 倍 ⇒ 距离应当**有限且不爆**
        d = H.stat_distance(low, high)
        assert 0.0 < d < 10.0, f"协方差项没归一(σ² 会爆到 100+),实测 {d:.2f}"


class TestReinhard:
    def test_output_stats_match_reference(self):
        """★★ Reinhard 的**定义性质**:迁移后统计量必须逼近参考。

        这条不成立 ⇒ 方法根本没在做事,后面所有"距离下降"都是假的。
        """
        src = _img((60, 90, 200), spread=15, seed=3)
        ref = _img((180, 130, 120), spread=45, seed=4)
        out = H.reinhard_transfer(src, ref)
        ms, cs = H.lab_stats(out)
        mr, cr = H.lab_stats(ref)
        assert np.allclose(ms, mr, atol=6.0), f"均值没对齐:{ms} vs {mr}"
        assert np.linalg.norm(cs - cr) < 0.35 * (np.linalg.norm(cs) + np.linalg.norm(cr))

    def test_distance_drops(self):
        src = _img((60, 90, 200), spread=15, seed=5)
        ref = _img((180, 130, 120), spread=45, seed=6)
        assert H.stat_distance(H.reinhard_transfer(src, ref), ref) < 0.5 * H.stat_distance(src, ref)

    def test_shape_and_dtype_preserved(self):
        out = H.reinhard_transfer(_img((100, 100, 100)), _img((200, 50, 50)))
        assert out.shape == (32, 32, 3) and out.dtype == np.uint8


class TestMasked:
    """★ 掩膜版 —— `harmonize_target` 用的就是这几个(只改粘贴区、只比粘贴区)。"""

    def test_lab_stats_uses_only_masked_pixels(self):
        """整图统计 vs 掩膜统计**必须不同** —— 否则这个参数是摆设。"""
        img = _img((128, 128, 128), spread=30, seed=7)
        img[:16, :16] = 250  # 左上角一小块全白
        m = np.zeros((32, 32), bool)
        m[:16, :16] = True
        assert not np.allclose(H.lab_stats(img)[0], H.lab_stats(img, m)[0])

    def test_empty_mask_raises(self):
        """掩膜里没有像素 ⇒ 统计量无定义,不许静默返回 NaN。"""
        img = _img((128, 128, 128))
        with pytest.raises(ValueError, match="无定义"):
            H.lab_stats(img, np.zeros((32, 32), bool))

    def test_mask_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="不同画幅"):
            H.lab_stats(_img((128, 128, 128)), np.ones((8, 8), bool))

    def test_dilate_grows_and_context_ring_excludes_the_mask(self):
        m = np.zeros((32, 32), bool)
        m[14:18, 14:18] = True
        d = H.dilate(m, 1)
        # ⚠️ **夹具错过的**一处:第一版我按"方块膨胀"写了 36(=6×6),实测是 **32** ——
        # `dilate` 是 **4 邻域(十字)**,方形块的**四个角不长** ⇒ 6×6 − 4 = 32。
        # 红的是期望值不是代码;十字膨胀是刻意的(不引 scipy,也不让外圈走对角)。
        assert d.sum() == 32, f"4×4 十字膨胀一步 = 6×6 去四角 = 32,实测 {d.sum()}"
        ring = H.context_ring(m, 2)
        assert ring.sum() > 0 and not (ring & m).any(), "外圈不许包含掩膜本身"

    def test_masked_transfer_hits_only_the_mask(self):
        """★★ **掩膜外逐位不变** —— 和谐化不许顺手把周围也改了。"""
        img = _img((80, 90, 200), spread=20, seed=8)
        ref = _img((190, 120, 110), spread=40, seed=9)
        m = np.zeros((32, 32), bool)
        m[8:24, 8:24] = True
        out = H.reinhard_transfer_masked(img, m, ref, ~m)
        np.testing.assert_array_equal(out[~m], img[~m])
        assert not np.array_equal(out[m], img[m])

    def test_masked_transfer_matches_reference_stats_inside_the_mask(self):
        """★ 定义性质:迁移后**掩膜内**的统计逼近 `ref[ref_mask]` 的统计。"""
        img = _img((80, 90, 200), spread=20, seed=10)
        ref = _img((190, 120, 110), spread=40, seed=11)
        m = np.zeros((32, 32), bool)
        m[8:24, 8:24] = True
        out = H.reinhard_transfer_masked(img, m, ref, ~m)
        ms = H.lab_stats(out, m)[0]
        mr = H.lab_stats(ref, ~m)[0]
        assert np.allclose(ms, mr, atol=6.0), f"掩膜内均值没对齐:{ms} vs {mr}"


class TestVerdict:
    """三分支。★ **只过"下降"不算数** —— "把任何图都往中间拉"也会让距离降。"""

    def test_passes_when_both_hold(self):
        v = H.verdict({"before_median": 0.80, "after_median": 0.20, "control_median": 0.55})
        assert "通过" in v

    def test_insensitive_is_caught(self):
        v = H.verdict({"before_median": 0.80, "after_median": 0.79, "control_median": 0.30})
        assert "不敏感" in v

    def test_control_not_separated_is_caught(self):
        """★★ **本模块的核心纪律**:降了、但与"对齐到别的帧"分不开 ⇒ 判据立不起来。"""
        v = H.verdict({"before_median": 0.80, "after_median": 0.40, "control_median": 0.41})
        assert "分不清" in v

"""`edit/depth_cond` 的判据回归钉。

**为什么这组必须有**:这个模块干的事是"把真值深度换算成别人模型的条件图"。
换算写错了**不会报错** —— 它会生成一张**看着像样**的图,ControlNet 照样出图,
只是条件进错了尺度(`1/d` 漏了 = 条件整幅反色)。⇒ 每条判据配一条**反向自证**:
把已知的错误做法造出来,尺子必须把它与正确做法**分开**。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.edit import depth_cond as D


def _ramp(h=8, w=8, near=1.0, far=100.0):
    """一行代表"远处", 末行代表"近处" —— 与地平面视角同形(下近上远)。"""
    return np.linspace(far, near, h, dtype=np.float64)[:, None] * np.ones((1, w))


class TestDisparity:
    def test_is_inverse_metric(self):
        d = np.array([[1.0, 2.0, 4.0]])
        np.testing.assert_allclose(D.disparity_from_metric(d), [[1.0, 0.5, 0.25]])

    def test_near_is_larger(self):
        """视差类量必须**大 = 近**,与 MiDaS 同向。写反了整幅条件图就反了。"""
        disp = D.disparity_from_metric(_ramp(near=1.0, far=100.0))
        assert disp[-1, 0] > disp[0, 0]  # 末行(近) 应更大

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, 1000.0, 999.0])
    def test_invalid_becomes_zero(self, bad):
        d = np.array([[1.0, float(bad)]])
        disp = D.disparity_from_metric(d)
        assert disp[0, 0] == 1.0
        assert disp[0, 1] == 0.0, f"{bad} 应当被当成无效(置 0 = 最远)"

    def test_reverse_selfcheck_without_guard(self):
        """反向自证:不设无效值上界会得到什么 —— 一个**有限但不该有**的视差。

        `1000.0` 在量程(`depth_codec.DEPTH_RANGE_M`)边上,若按有效值处理会给出 0.001,
        与真正的远景不可区分。这条钉住"上界确实在起作用"。
        """
        raw = 1.0 / np.array([[1000.0]])[0, 0]
        assert raw > 0.0
        assert D.disparity_from_metric(np.array([[1000.0]]))[0, 0] == 0.0


class TestMinMax:
    def test_endpoints(self):
        out = D.minmax_uint8(np.array([[-3.0, 0.0, 7.0]]))
        assert out.dtype == np.uint8
        assert out[0, 0] == 0 and out[0, 2] == 255

    def test_constant_raises(self):
        """★ 常量图**必须抛**:静默返回全黑 = 一张"无条件"的条件图,
        而 ControlNet 照样出图、"看着还行" —— 这种错看不出是错的。"""
        with pytest.raises(ValueError, match="退化"):
            D.minmax_uint8(np.full((4, 4), 3.0))

    def test_all_invalid_raises(self):
        """一帧全无效(深度全 0)⇒ 视差全 0 ⇒ 归一化无定义,必须抛而不是给全黑。"""
        with pytest.raises(ValueError):
            D.metric_depth_to_cond(np.zeros((4, 4)))


class TestCondOrientation:
    def test_white_is_near(self):
        """★ 条件图的朝向:近处 = 白。这条是"口径对不对"的第一道闸。"""
        cond = D.metric_depth_to_cond(_ramp(near=1.0, far=100.0))
        assert cond[-1, 0] > cond[0, 0], "近处(末行)应当更白"
        assert cond[-1, 0] == 255 and cond[0, 0] == 0

    def test_inversion_is_separable(self):
        """★★ **反向自证**:把 `1/d` 漏掉(直接归一化 `d`)会得到**反色** ⇒ 这一步是必需的。

        ⚠️ **不是恰好 −1,实测 −0.988** —— 第一版这里写的是 `approx(-1.0)`,当场变红,而
        **红的是测试不是代码**:条件图是 **uint8**,min-max 之后出现**大量并列(tie)**,
        并列会稀释秩相关。⇒ **这把尺子的上限本来就不是 ±1.0**,判据要按实测标定。
        (同族教训:本仓 §P-V24「阈值是操作点不是模型属性」。)

        判别力不受影响:反色是 **−0.99**,而正确 vs 正确是 **+1.00** —— 中间没有灰区。
        """
        d = _ramp()
        correct = D.metric_depth_to_cond(d)
        wrong = D.minmax_uint8(d)  # ← 漏掉 1/d 的写法
        rho_inv = D.spearman(correct, wrong)
        assert rho_inv < -0.95, f"漏掉 1/d 应当明显反相关,实测 {rho_inv:.4f}"
        assert D.spearman(correct, correct) == pytest.approx(1.0)
        assert rho_inv < -0.9 and abs(D.spearman(correct, wrong) - 1.0) > 1.5  # 与"正确"分得开
        assert correct[-1, 0] == 255 and wrong[-1, 0] == 0


class TestSpearman:
    def test_monotone_is_one(self):
        a = np.arange(100.0)
        assert D.spearman(a, a**3 + 5) == pytest.approx(1.0)

    def test_reversed_is_minus_one(self):
        a = np.arange(100.0)
        assert D.spearman(a, -a) == pytest.approx(-1.0)

    def test_shuffle_drops(self):
        """反向自证:打乱之后必须掉下来 —— 否则这把尺子量不出"排序一致"。"""
        rng = np.random.default_rng(0)
        a = np.arange(2000.0)
        shuffled = a.copy()
        rng.shuffle(shuffled)
        assert D.spearman(a, a) == pytest.approx(1.0)
        assert abs(D.spearman(a, shuffled)) < 0.1

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="尺寸不同"):
            D.spearman(np.zeros((2, 2)), np.zeros((2, 3)))

    def test_too_few_valid_raises(self):
        with pytest.raises(ValueError, match="有效像素不足"):
            D.spearman(np.array([np.nan]), np.array([1.0]))


class TestFrameSpec:
    def test_range_and_list(self):
        assert D._parse_frames("0,2-4") == [0, 2, 3, 4]

    def test_empty_raises(self):
        with pytest.raises(SystemExit):
            D._parse_frames(" , ")


class TestLoadGtDepth:
    def test_missing_capture_raises(self, tmp_path):
        with pytest.raises(SystemExit, match="不是"):
            D._load_gt_depth(tmp_path, 0)

    def test_reads_first_pitch(self, tmp_path):
        p = tmp_path / "capture" / "depth" / "p0"
        p.mkdir(parents=True)
        np.save(p / "00007.npy", np.ones((3, 4), dtype=np.float32))
        got = D._load_gt_depth(tmp_path, 7)
        assert got.shape == (3, 4)

    def test_missing_frame_raises(self, tmp_path):
        (tmp_path / "capture" / "depth" / "p0").mkdir(parents=True)
        with pytest.raises(SystemExit, match="缺帧"):
            D._load_gt_depth(tmp_path, 3)


class TestRankUniform:
    """等化到均匀分布。它的**全部意义**在于"保序但换分布" —— 两条都要钉住。"""

    def test_is_monotone(self):
        x = np.array([[3.0, 1.0], [2.0, 4.0]])
        out = D.rank_uniform_uint8(x)
        assert out[0, 1] < out[1, 0] < out[0, 0] < out[1, 1]  # 1<2<3<4

    def test_spans_full_range(self):
        out = D.rank_uniform_uint8(np.linspace(0, 1, 100).reshape(10, 10))
        assert out.min() == 0 and out.max() == 255

    def test_spreads_a_narrow_band(self):
        """★ 这条才是它存在的理由:把挤在窄带里的图铺到全量程。

        真值条件是发灰的平场(实测挤在 131–227),MiDaS 铺满 0–255。
        """
        peaked = np.full((100, 100), 200.0)
        peaked[:50] = np.linspace(131, 227, 50 * 100).reshape(50, 100)
        assert np.ptp(peaked) < 100, "构造的输入本来就该是窄的"
        out = D.rank_uniform_uint8(peaked)
        assert np.ptp(out) == 255

    def test_raises_on_1d(self):
        with pytest.raises(ValueError, match="二维"):
            D.rank_uniform_uint8(np.arange(5.0))

    def test_spearman_invisible_to_normalization(self):
        """★★ **反向自证**:三个归一化与参考图的 Spearman **实质相同**,而**分布差了几倍**。

        ① 三者保序 ⇒ ρ 一致(否则这里会散);
        ② ★ **ρ 这把尺子看不见分布差** —— 所以"ρ 正常"不能当成"条件没问题"。

        ⚠️ 断言用的是**容差不是相等** —— 第一版写的是 `==`,当场红(0.00335 vs 0.00327)。
        **红的是我的断言,不是代码**:uint8 量化下 min-max 在窄带上产生大量**并列**,
        秩等化则铺开,而并列会改秩。差别 ~2%,对"看不见"这个结论毫无影响
        —— **但它说明连"保序 ⇒ ρ 完全不变"这句都得带容差**。
        """
        rng = np.random.default_rng(0)
        x = rng.normal(0.12, 0.01, size=(64, 64))  # 窄带(像 1/d 在远场那一段)
        ref = rng.integers(0, 256, size=(64, 64)).astype(np.uint8)  # 像 MiDaS 的条件
        a = D.minmax_uint8(x)
        b = D.rank_uniform_uint8(x)
        c = D.match_histogram_uint8(x, ref)
        rhos = [D.spearman(v, ref) for v in (a, b, c)]
        assert max(rhos) - min(rhos) < 0.01, f"ρ 应当实质不变,实测 {rhos}"

        # 而分布确实差得多 —— 这才是"ρ 看不见"的对照。
        # ⚠️ 量的是**中段占比**不是 `ptp`:min-max 按定义就铺满 0–255(第一版拿 ptp 断言,当场红)。
        #    "发灰"的真相是**像素堆在中段**,不是范围窄。
        def mid_share(v):
            return float(np.mean((v >= 102) & (v <= 153)))

        assert mid_share(b) == pytest.approx(0.2, abs=0.05), "秩等化后中段应当约占 20%"
        assert mid_share(a) > 2 * mid_share(b), f"min-max 应当把像素堆在中段,实测 {mid_share(a):.3f}"


class TestMatchHistogram:
    def test_output_quantiles_follow_reference(self):
        x = np.linspace(0, 1, 1000).reshape(50, 20)
        ref = np.concatenate([np.zeros(500), np.full(500, 255)]).astype(np.uint8).reshape(50, 20)
        out = D.match_histogram_uint8(x, ref)
        # 参考图只有两个值 ⇒ 输出也只该有两个值,且各占一半
        vals, counts = np.unique(out, return_counts=True)
        assert list(vals) == [0, 255]
        assert counts[0] == 500

    def test_is_monotone(self):
        x = np.array([[1.0, 4.0], [2.0, 3.0]])
        ref = np.arange(256, dtype=np.uint8).reshape(16, 16)
        out = D.match_histogram_uint8(x, ref)
        assert out[0, 0] < out[1, 0] < out[1, 1] < out[0, 1]
        assert out[0, 0] == 0 and out[0, 1] == 255

    def test_too_small_raises(self):
        with pytest.raises(ValueError, match="至少 2 个像素"):
            D.match_histogram_uint8(np.array([[1.0]]), np.array([[1.0]]))


class TestNormalizeSwitch:
    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="未知的归一化"):
            D.metric_depth_to_cond(_ramp(), normalize="zscore")

    def test_all_three_keep_orientation(self):
        """三种归一化都必须保持"白 = 近" —— 换归一化不该把图翻过来。"""
        d = _ramp(near=1.0, far=100.0)
        ref = np.arange(256, dtype=np.uint8).reshape(16, 16)
        for mode in ("minmax", "histeq", "match"):
            c = D.metric_depth_to_cond(d, normalize=mode, ref=ref)
            assert c[-1, 0] > c[0, 0], f"{mode} 把方向弄反了"

    def test_match_without_ref_falls_back_to_minmax(self):
        d = _ramp()
        np.testing.assert_array_equal(
            D.metric_depth_to_cond(d, normalize="match"), D.metric_depth_to_cond(d, normalize="minmax")
        )

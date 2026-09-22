"""`depth_codec` 手算锚点单测。

三件事各自钉死:

1. **通道角色**(B 高位 / G 中位 / R 低位)—— 两份旧实现的注释都写错了,只有代码对;
   这里用**实测数值锚点**把代码的正确性固定下来(B=255 → 996.09 m 近满量程,
   R=255 → 0.0152 m 近 0),以后谁改通道顺序都会红。
2. **编解码互逆** —— `encode_depth` 是 `decode_depth` 的逆,但**只在实数域精确**:
   `decode_depth` 的 float32 出口在 1000 m 处的间距(6.10e-05 m)**大于**量化步
   (5.96e-05 m),故 24 bit 定点往返必然有 ≤1 LSB 的抖动。这条抖动是**输出 dtype
   的代价,不是公式错**,测试按此口径断言(实数域逐字节相等 + API 往返 ≤1 LSB)。
3. **像素索引约定** —— 索引 `i` 的中心在连续坐标 `i + 0.5`;全仓 0.5 只在这里出现。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.depth_codec import (
    _DENOM,
    CONVENTION_CENTER,
    CONVENTION_CORNER,
    DEPTH_RANGE_M,
    continuous_to_index,
    decode_depth,
    encode_depth,
    index_to_continuous,
    sample_bilinear,
    sample_bilinear_many,
)

H, W = 6, 8


def _packed_to_raw(q: np.ndarray) -> np.ndarray:
    """24 bit 整数 → BGRA uint8(B 高位 / G 中位 / R 低位)。"""
    h, w = q.shape
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[:, :, 0] = ((q >> 16) & 0xFF).astype(np.uint8)
    out[:, :, 1] = ((q >> 8) & 0xFF).astype(np.uint8)
    out[:, :, 2] = (q & 0xFF).astype(np.uint8)
    out[:, :, 3] = 255
    return out


def _raw_to_packed(raw: np.ndarray) -> np.ndarray:
    return (
        (raw[:, :, 0].astype(np.uint64) << 16)
        | (raw[:, :, 1].astype(np.uint64) << 8)
        | raw[:, :, 2].astype(np.uint64)
    )


def _single_channel_anchor(channel: int) -> float:
    """只把一个通道置 255、其余 0,读出解码值(实测通道角色)。"""
    raw = np.zeros((1, 1, 4), dtype=np.uint8)
    raw[:, :, 3] = 255
    raw[:, :, channel] = 255
    return float(decode_depth(raw.copy(), 1, 1)[0, 0])


class TestChannelRole:
    """通道角色锚点 —— 旧注释写反了,这里钉代码的正确性。"""

    def test_blue_is_high_bits(self):
        # B=255 → 255·65536/(256³−1)·1000 ≈ 996.09 m(近满量程)
        assert _single_channel_anchor(0) == pytest.approx(996.0938, abs=1e-3)

    def test_green_is_middle_bits(self):
        # G=255 → 255·256/(256³−1)·1000 ≈ 3.891 m
        assert _single_channel_anchor(1) == pytest.approx(3.89099, abs=1e-4)

    def test_red_is_low_bits(self):
        # R=255 → 255/(256³−1)·1000 ≈ 0.0152 m(近 0)
        assert _single_channel_anchor(2) == pytest.approx(0.0151992, abs=1e-6)

    def test_channel_magnitudes_are_ordered(self):
        # 位权单调:B(高位) ≫ G(中位) ≫ R(低位)
        b, g, r = (_single_channel_anchor(i) for i in range(3))
        assert b > 250.0 * g > 250.0 * 250.0 * r

    def test_hand_computed_formula(self):
        # 手算锚点:R=1,G=2,B=3 → (1 + 512 + 196608)/(256³−1)·1000
        raw = np.zeros((1, 1, 4), dtype=np.uint8)
        raw[0, 0, 0], raw[0, 0, 1], raw[0, 0, 2], raw[0, 0, 3] = 3, 2, 1, 255
        expect = (1 + 2 * 256 + 3 * 65536) / _DENOM * DEPTH_RANGE_M
        assert float(decode_depth(raw.copy(), 1, 1)[0, 0]) == pytest.approx(expect, rel=1e-6)


class TestRoundTrip:
    def test_encode_is_exact_inverse_in_float64(self):
        """实数域(不做 float32 出口)编解码**逐字节**互逆。"""
        q = np.random.default_rng(0).integers(0, 1 << 24, size=(H, W)).astype(np.uint64)
        meters = q.astype(np.float64) / _DENOM * DEPTH_RANGE_M
        assert (_raw_to_packed(encode_depth(meters)) == q).all()

    def test_api_roundtrip_within_one_lsb(self):
        """真实 API 往返(decode 的 float32 出口)误差 ≤ 1 LSB。

        不是公式不准:1000 m 处 float32 间距 6.10e-05 m **大于**量化步 5.96e-05 m,
        24 bit 定点本就装不进 float32 ⇒ 必然抖动。这条断言把"上界是 1"钉死,
        任何真正的公式错误(通道颠倒 / 分母写错)都会远远超过 1。
        """
        q = np.random.default_rng(1).integers(0, 1 << 24, size=(H, W)).astype(np.uint64)
        raw = _packed_to_raw(q)
        back = _raw_to_packed(encode_depth(decode_depth(raw.copy(), H, W)))
        assert np.abs(back.astype(np.int64) - q.astype(np.int64)).max() <= 1

    def test_float32_epsilon_exceeds_quantization_step(self):
        """把上一条的**根因**写成断言:float32 出口装不下 24 bit 定点。"""
        step = DEPTH_RANGE_M / _DENOM
        assert np.spacing(np.float32(DEPTH_RANGE_M)) > step

    def test_meters_roundtrip_error_is_sub_quantization(self):
        """米域往返误差 ≤ 2 个量化步(1 步来自 24 bit 截断,1 步来自 float32 出口)。"""
        q = np.random.default_rng(2).integers(0, 1 << 24, size=(H, W)).astype(np.uint64)
        raw = _packed_to_raw(q)
        d = decode_depth(raw.copy(), H, W)
        back = encode_depth(d)
        step = DEPTH_RANGE_M / _DENOM
        assert np.abs(decode_depth(back, H, W).astype(np.float64) - d.astype(np.float64)).max() <= 2.0 * step

    def test_encode_saturates_above_range(self):
        out = encode_depth(np.array([[DEPTH_RANGE_M * 3.0, -5.0]]))
        packed = _raw_to_packed(out)
        assert int(packed[0, 0]) == (1 << 24) - 1  # 超量程 → 满量程
        assert int(packed[0, 1]) == 0  # 负值 → 0

    def test_alpha_channel_is_opaque(self):
        out = encode_depth(np.full((2, 3), 10.0))
        assert (out[:, :, 3] == 255).all()


class TestIndexConvention:
    def test_center_convention_roundtrip(self):
        for i in (0.0, 1.0, 620.0, 621.5):
            assert continuous_to_index(index_to_continuous(i, CONVENTION_CENTER), CONVENTION_CENTER) == i

    def test_corner_convention_is_identity(self):
        assert index_to_continuous(7.0, CONVENTION_CORNER) == 7.0
        assert continuous_to_index(7.0, CONVENTION_CORNER) == 7.0

    def test_conventions_differ_by_half_pixel(self):
        assert (
            index_to_continuous(100.0, CONVENTION_CENTER) - index_to_continuous(100.0, CONVENTION_CORNER)
            == 0.5
        )

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError):
            index_to_continuous(1.0, "nonsense")
        with pytest.raises(ValueError):
            continuous_to_index(1.0, "nonsense")

    def test_image_center_of_even_width_has_no_pixel_center_on_axis(self):
        """1242 是偶数 ⇒ 没有任何像素中心落在 w/2 上(这正是主点口径分裂的根源)。"""
        centres = np.array([index_to_continuous(i) for i in range(1242)])
        assert np.abs(centres - 621.0).min() == pytest.approx(0.5)


class TestSampleBilinear:
    def test_pixel_center_samples_that_pixel(self):
        """按 center 约定在**索引 i 的中心**采样,必须精确取到索引 i 的值。"""
        img = np.arange(H * W, dtype=np.float64).reshape(H, W)
        for r, c in ((0, 0), (2, 3), (H - 1, W - 1), (1, 6)):
            u, v = index_to_continuous(c), index_to_continuous(r)
            assert sample_bilinear(img, u, v, CONVENTION_CENTER) == img[r, c]

    def test_corner_convention_reads_integer_index(self):
        img = np.arange(H * W, dtype=np.float64).reshape(H, W)
        assert sample_bilinear(img, 3.0, 2.0, CONVENTION_CORNER) == img[2, 3]

    def test_midpoint_is_average(self):
        img = np.zeros((H, W), dtype=np.float64)
        img[2, 3] = 10.0
        img[2, 4] = 20.0
        # 索引 3 与 4 的中心 = 连续坐标 3.5 与 4.5,中点 4.0
        assert sample_bilinear(img, 4.0, 2.5, CONVENTION_CENTER) == pytest.approx(15.0)

    def test_out_of_bounds_is_nan(self):
        img = np.ones((H, W))
        assert np.isnan(sample_bilinear(img, -1.0, 2.0))
        assert np.isnan(sample_bilinear(img, 2.0, float(W)))
        assert np.isnan(sample_bilinear(img, 0.0, float(H)))

    def test_boundary_pixel_centre_is_inside(self):
        """两种约定下**末像素中心**都必须采得到(边界不吞像素)。"""
        img = np.ones((H, W))
        for conv in (CONVENTION_CENTER, CONVENTION_CORNER):
            u = index_to_continuous(W - 1, conv)
            v = index_to_continuous(H - 1, conv)
            assert sample_bilinear(img, u, v, conv) == 1.0

    def test_default_convention_is_corner(self):
        """默认约定 = **CARLA 渲染光栅实测口径** corner(裁决见 `bin/probe_calib.py` A3/A4)。

        判据取"索引 3 读到的必须是第 3 列"——corner 下 `u=3.0` 直取;若默认退回 center,
        `u=3.0` 会被插值成第 2/3 列各半(2.5),这条立刻红。
        """
        img = np.arange(H * W, dtype=np.float64).reshape(H, W)
        assert sample_bilinear(img, 3.0, 2.0) == img[2, 3]
        assert sample_bilinear_many(img, np.array([[3.0, 2.0]]))[0] == img[2, 3]


class TestSampleBilinearMany:
    """批量版必须与逐点版**逐点同值** —— 否则离线探针与实时槽会给出两个口径。"""

    @pytest.mark.parametrize("convention", [CONVENTION_CENTER, CONVENTION_CORNER])
    def test_matches_scalar_sampler(self, convention):
        img = np.random.default_rng(0).random((37, 51))
        uv = np.random.default_rng(1).uniform(-3.0, 54.0, size=(4000, 2))
        want = np.array([sample_bilinear(img, float(u), float(v), convention) for u, v in uv])
        got = sample_bilinear_many(img, uv, convention)
        assert np.array_equal(np.isnan(want), np.isnan(got))
        assert np.allclose(want, got, equal_nan=True, atol=0.0)

    def test_empty_input(self):
        assert sample_bilinear_many(np.ones((4, 4)), np.zeros((0, 2))).shape == (0,)

    def test_all_out_of_bounds_is_nan(self):
        got = sample_bilinear_many(np.ones((4, 4)), np.full((3, 2), 99.0))
        assert np.isnan(got).all()


class TestDecodeShape:
    def test_accepts_bytes_buffer(self):
        raw = _packed_to_raw(np.full((H, W), 12345, dtype=np.uint64))
        d = decode_depth(raw.tobytes(), H, W)
        assert d.shape == (H, W)

    def test_accepts_ndarray(self):
        raw = _packed_to_raw(np.full((H, W), 12345, dtype=np.uint64))
        assert np.allclose(decode_depth(raw, H, W), decode_depth(raw.tobytes(), H, W))

    def test_ignores_alpha(self):
        raw = _packed_to_raw(np.full((H, W), 999, dtype=np.uint64))
        raw[:, :, 3] = 0
        assert float(decode_depth(raw, H, W)[0, 0]) == pytest.approx(999.0 / _DENOM * DEPTH_RANGE_M, rel=1e-6)

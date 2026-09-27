"""可变形注意力原语单测:几何正确性 / 分块等价 / 梯度 / **归一化口径**。

本文件的核心是第 1 条 —— 用**既有的** `MapTRDecoderLayer._sample_bev` 当 oracle:
偏移全零 + 权重 one-hot 时,可变形注意力必须**逐点等于**该参考点处的双线性采样。
两套实现互为对照,任何一处归一化写错都会在这里被抓住,而不是等到训练不收敛。

第 5 条是**静默失效的回归钉**:米直接进 `sine_pos_embed` 不报错、只是频率别名,
表现为「训不动」;它必须与归一化后进**不同**。
"""

from __future__ import annotations

import numpy as np
import torch

from autodrivedata.map.maptr.decoder import MapTRDecoderLayer
from autodrivedata.map.maptr.deform_attn import (
    build_sampling_locations,
    multi_scale_deform_attn,
    normalize_ref,
    sine_pos_embed,
)

PC_RANGE = (-15.0, -30.0, 15.0, 30.0)
BEV_H, BEV_W = 20, 10
HEADS, HEAD_DIM = 2, 4
EMBED = HEADS * HEAD_DIM


def _bev(b: int = 1) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(b, EMBED, BEV_H, BEV_W)


def _value_from_bev(bev: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) → (B, H·W, heads, head_dim)(`multi_scale_deform_attn` 的 value 口径)。"""
    b = bev.shape[0]
    return bev.permute(0, 2, 3, 1).reshape(b, BEV_H * BEV_W, HEADS, HEAD_DIM)


def _onehot_weights(b: int, nq: int, k: int, idx: int = 0) -> torch.Tensor:
    w = torch.zeros(b, nq, HEADS, 1, k)
    w[..., idx] = 1.0
    return w


def test_zero_offset_onehot_equals_bilinear_oracle() -> None:
    """★ 偏移全零 + 权重 one-hot(idx 0)⇒ 与 `_sample_bev` 双线性采样逐点一致。"""
    torch.manual_seed(0)
    bev = _bev()
    k = 4
    ref_m = torch.tensor([[0.0, 0.0], [3.0, -6.0], [-14.4, 29.1], [7.5, 12.25], [-0.3, -0.7], [11.0, -20.0]])
    nq = ref_m.shape[0]
    loc = build_sampling_locations(
        normalize_ref(ref_m, PC_RANGE)[None, :, None, :],  # (B=1, Nq, L=1, 2)
        torch.zeros(1, nq, HEADS, 1, k, 2),
        [(BEV_H, BEV_W)],
    )
    got = multi_scale_deform_attn(
        _value_from_bev(bev), [(BEV_H, BEV_W)], loc, _onehot_weights(1, nq, k)
    )  # (1, Nq, C)

    layer = MapTRDecoderLayer(embed_dims=EMBED, n_heads=HEADS, bev_dims=EMBED)
    oracle = layer._sample_bev(bev, ref_m[None, :, None, :], PC_RANGE)  # (Nq, B, P=1, C)
    oracle = oracle.squeeze(2).permute(1, 0, 2)  # (B, Nq, C)

    torch.testing.assert_close(got, oracle, rtol=1e-5, atol=1e-6)


def test_zero_offset_kernel_offsets_shift_by_pixels() -> None:
    """偏移以像素为单位:第 i 个 kernel 点在 x 上偏 1 px ⇒ 归一化坐标恰好挪 1/W。"""
    ref = torch.tensor([[[[0.5, 0.5]]]])  # (B=1, Nq=1, L=1, 2)
    offs = torch.zeros(1, 1, 1, 1, 2, 2)
    offs[..., 1, 0] = 1.0  # 第 1 个采样点在 x 偏 1 px
    loc = build_sampling_locations(ref, offs, [(BEV_H, BEV_W)])
    torch.testing.assert_close(loc[0, 0, 0, 0, 0], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(loc[0, 0, 0, 0, 1], torch.tensor([0.5 + 1.0 / BEV_W, 0.5]))


def test_chunk_matches_unchunked() -> None:
    """分块只影响峰值显存,不改结果。"""
    torch.manual_seed(1)
    bev = _bev()
    nq, k = 9, 4
    ref_m = torch.randn(nq, 2) * torch.tensor([12.0, 24.0])
    loc = build_sampling_locations(
        normalize_ref(ref_m, PC_RANGE)[None, :, None, :],
        torch.randn(1, nq, HEADS, 1, k, 2) * 0.5,
        [(BEV_H, BEV_W)],
    )
    w = torch.softmax(torch.randn(1, nq, HEADS, 1, k), dim=-1)
    val = _value_from_bev(bev)
    full = multi_scale_deform_attn(val, [(BEV_H, BEV_W)], loc, w)
    for chunk in (1, 2, 4):
        torch.testing.assert_close(
            multi_scale_deform_attn(val, [(BEV_H, BEV_W)], loc, w, chunk=chunk), full, rtol=1e-6, atol=1e-7
        )


def test_gradients_reach_value_offsets_and_weights() -> None:
    """梯度能回到 value / 采样偏移 / 注意力权重(三项都可学或需回传)。"""
    torch.manual_seed(2)
    b, nq, k = 1, 5, 4
    val = _value_from_bev(_bev(b)).requires_grad_(True)
    ref = normalize_ref(torch.randn(b, nq, 2) * 5.0, PC_RANGE).unsqueeze(-2)  # (B, Nq, L=1, 2)
    offs = (torch.randn(b, nq, HEADS, 1, k, 2) * 0.1).requires_grad_(True)
    w = torch.softmax(torch.randn(b, nq, HEADS, 1, k), dim=-1).requires_grad_(True)
    loc = build_sampling_locations(ref, offs, [(BEV_H, BEV_W)])
    out = multi_scale_deform_attn(val, [(BEV_H, BEV_W)], loc, w)
    out.sum().backward()
    for name, t in (("value", val), ("offsets", offs), ("weights", w)):
        g = t.grad
        assert g is not None, f"{name} 梯度缺失"
        assert bool(torch.isfinite(g).all()), f"{name} 梯度非有限"
    # value 的梯度必须非全零(采样真的命中了特征,不是全 padding)
    val_grad = val.grad
    assert val_grad is not None and float(val_grad.abs().sum()) > 0.0


def test_multi_level_concatenation() -> None:
    """多 level:各 level 在 N 维顺序拼接,结果等于逐 level 单独算再相加。"""
    torch.manual_seed(3)
    shapes = [(BEV_H, BEV_W), (BEV_H // 2, BEV_W // 2)]
    n = sum(h * w for h, w in shapes)
    val = torch.randn(1, n, HEADS, HEAD_DIM)
    nq, k = 3, 2
    ref = torch.tensor([[[[0.5, 0.5]]]]).expand(1, nq, 1, 2)
    loc = build_sampling_locations(ref, torch.zeros(1, nq, HEADS, len(shapes), k, 2), shapes)
    w = torch.zeros(1, nq, HEADS, len(shapes), k)
    w[..., 0, 0] = 1.0  # 只取 level 0 的第 0 个点
    got = multi_scale_deform_attn(val, shapes, loc, w)
    only0 = multi_scale_deform_attn(val[:, : BEV_H * BEV_W], [shapes[0]], loc[:, :, :, :1], w[:, :, :, :1])
    torch.testing.assert_close(got, only0)


def test_normalize_ref_maps_pc_range_corners_to_unit_interval() -> None:
    """`pc_range` 四角 → `[0,1]` 端点(且与 `_sample_bev` 的 `[−1,1]` 网格是同一线性映射)。"""
    corners = torch.tensor([[-15.0, -30.0], [15.0, 30.0], [-15.0, 30.0], [15.0, -30.0], [0.0, 0.0]])
    norm = normalize_ref(corners, PC_RANGE)
    torch.testing.assert_close(norm[0], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(norm[1], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(norm[2], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(norm[3], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(norm[4], torch.tensor([0.5, 0.5]))
    # `_sample_bev` 的网格 = 2·norm − 1 ⇒ 中心必映射到 0
    torch.testing.assert_close(2 * norm[4] - 1, torch.zeros(2))
    assert float(norm.min()) >= 0.0 and float(norm.max()) <= 1.0


def test_meter_scale_sine_differs_from_normalized() -> None:
    """★ 静默失效回归钉:米**直接**进 `sine_pos_embed` 与归一化后进**必须不同**。

    两者的区别不报错、不崩,只是频率别名 ⇒ 位置嵌入退化成噪声(症状是「训不动」)。
    这条钉住「必须先 `normalize_ref`」这件事本身。
    """
    pts = torch.tensor([[-12.0, -25.0], [0.0, 0.0], [13.0, 27.5]])
    from_norm = sine_pos_embed(normalize_ref(pts, PC_RANGE), dim=EMBED)
    from_meters = sine_pos_embed(pts, dim=EMBED)
    assert not torch.allclose(from_norm, from_meters), "米与归一化产生同一嵌入 ⇒ 归一化被绕过"


def test_sine_pos_embed_shape_and_dim_validation() -> None:
    emb = sine_pos_embed(torch.rand(2, 3, 2), dim=8)
    assert emb.shape == (2, 3, 8)
    assert bool(torch.isfinite(emb).all())
    try:
        sine_pos_embed(torch.rand(2), dim=7)
    except ValueError as e:
        assert "偶数" in str(e)
    else:  # pragma: no cover - 形状/维度校验失败必须抛
        raise AssertionError("奇数 dim 应抛 ValueError")


def test_output_is_zero_where_sampling_falls_outside() -> None:
    """采样点远在 BEV 之外 ⇒ padding 0(与官方 `padding_mode="zeros"` 一致)。"""
    bev = _bev()
    nq, k = 2, 2
    ref_m = torch.tensor([[1e3, 1e3], [-1e3, -1e3]])
    loc = build_sampling_locations(
        normalize_ref(ref_m, PC_RANGE)[None, :, None, :], torch.zeros(1, nq, HEADS, 1, k, 2), [(BEV_H, BEV_W)]
    )
    out = multi_scale_deform_attn(_value_from_bev(bev), [(BEV_H, BEV_W)], loc, _onehot_weights(1, nq, k))
    np.testing.assert_allclose(out.detach().numpy(), 0.0, atol=1e-6)

"""可变形注意力原语 —— 纯 torch,MapQR 移植的两支共用(无 mmcv 依赖)。

**为什么单独成模块**:MapQR 的解码器交叉注意力(`InstancePointAttention`)与 BEV 编码器的
`MSDeformableAttentionKernel` 共用同一套「参考点 + 可学习偏移 → 归一化采样坐标 → 双线性
加权求和」数学,只有**偏移是否可学**不同(解码器可学、编码器用固定 kernel 网格)。
抽成一份实现,两条调用路径就不可能漂移 —— 与 [chamfer_gpu](../../../docs/fileTree.md) 立
`chamfer_ap` 的取向一致。

**签名对齐官方** `mmcv.ops.multi_scale_deform_attn.multi_scale_deformable_attn_pytorch`
(`value` 已按头切分、`sampling_locations` ∈ `[0,1]`、`attention_weights` 已 softmax),
便于逐行对账。

**★ 归一化的唯一落点是本模块**:官方整套数学吃的是**归一化 `[0,1]` 坐标**,而本项目的
坐标口径是**米**(`head.py` 头注「点回归预测绝对坐标」、§5.11d 自实现口径)。米直接进
`sine_pos_embed` 不会报错,只会让正弦频率**别名**、位置嵌入退化成噪声 —— 表现为「训不动」
而不是异常。故:调用方一律先用 `normalize_ref` 转 `[0,1]` 再进本模块,
`tests/map/test_head.py` 有对应回归钉。

**显存**:`grid_sample` 的中间块按 `bs·heads·head_dim·Nq·K` 计。解码器侧 `Nq = 200×20`
量级很小;BEV 编码器交叉注意力的 `Nq` 可达**单相机可见 BEV 数**(数千)。

**`chunk` 的实测结论(2026-09-27,RTX 3090 48G,真实 surround_v2 帧)**:按 query 维分块
在**本项目的 BEV 交叉注意力上并不能降显存** —— `chunk=0` 与 `chunk=8192` 都是 23.6 GiB
(该帧 `max_len = 5074`,小于 8192 ⇒ 只有一块),而 `chunk=4096 / 2048` **反而升到**
26.3 / 29.5 GiB。故 `bevenc` 侧默认不分块,**要降显存请调层数**
(`BEVEncoder.num_layers`,实测每层 ≈ +3.6 GiB @bs2)。参数保留是为了在别的形状上可复测,
不是推荐用法。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def normalize_ref(pts_m: torch.Tensor, pc_range: tuple[float, float, float, float]) -> torch.Tensor:
    """ego 系米坐标 → `[0,1]`。`pts_m` (…, 2) 为 (x 前, y 左);`pc_range` = (xmin, ymin, xmax, ymax)。

    与 [`decoder.MapTRDecoderLayer._sample_bev`](decoder.py) 的 `gx = x / ((xmax−xmin)/2)` **同一条线性映射**
    (`2·normalize_ref − 1` 即该式的 `[−1,1]` 网格坐标),故两者的采样位置可逐点对照。
    """
    xmin, ymin, xmax, ymax = pc_range
    xn = (pts_m[..., 0] - xmin) / (xmax - xmin)
    yn = (pts_m[..., 1] - ymin) / (ymax - ymin)
    return torch.stack([xn, yn], dim=-1)


def sine_pos_embed(pos_norm: torch.Tensor, dim: int) -> torch.Tensor:
    """归一化坐标的正弦位置嵌入(官方 `gen_sineembed_for_position` 的 2 维分支)。

    `pos_norm` (…, 2) ∈ `[0,1]` → (…, `dim`)。`dim` 需为偶数;输出为
    `cat([y 的正弦嵌入, x 的正弦嵌入])`(顺序与官方一致,别调换 —— 调换不报错但语义变了)。

    官方把 `arange(128)` 写死(对应 `embed_dims=256`),此处按 `dim` 参数化,
    使 `embed_dims` 可变。
    """
    if dim % 2 != 0:
        raise ValueError(f"dim 需为偶数(半维给 x、半维给 y),收到 {dim}")
    half = dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half, dtype=torch.float32, device=pos_norm.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half)

    def _embed(v: torch.Tensor) -> torch.Tensor:
        e = v.unsqueeze(-1) * scale / dim_t  # (…, half)
        return torch.stack((e[..., 0::2].sin(), e[..., 1::2].cos()), dim=-1).flatten(-2)

    return torch.cat([_embed(pos_norm[..., 1]), _embed(pos_norm[..., 0])], dim=-1)


def build_sampling_locations(
    reference_points: torch.Tensor,
    sampling_offsets: torch.Tensor,
    spatial_shapes: list[tuple[int, int]],
) -> torch.Tensor:
    """参考点 + 偏移 → 归一化采样坐标(官方同式)。

    `reference_points` (B, Nq, L, 2) ∈ `[0,1]`;`sampling_offsets` (B, Nq, H, L, K, 2)
    以**像素/格**为单位;`spatial_shapes` 每项 (H, W)。返回 (B, Nq, H, L, K, 2) ∈ `[0,1]`。

    偏移除以 (W, H) 才落到 `[0,1]` —— 这一步的前提正是参考点已归一化。
    """
    dev, dt = reference_points.device, reference_points.dtype
    normalizer = torch.tensor([[w, h] for h, w in spatial_shapes], device=dev, dtype=dt)  # (L, 2)
    return (
        reference_points[:, :, None, :, None, :] + sampling_offsets / normalizer[None, None, None, :, None, :]
    )


def multi_scale_deform_attn(
    value: torch.Tensor,
    spatial_shapes: list[tuple[int, int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    chunk: int = 0,
) -> torch.Tensor:
    """多尺度可变形注意力:`Σ_{level,point} w · grid_sample(value, loc)`。

    - `value` (B, N, H, C/H):**已按头切分**(官方签名如此),`N = Σ_l H_l·W_l`,
      各 level 在 `N` 维上顺序拼接;
    - `sampling_locations` (B, Nq, H, L, K, 2) ∈ `[0,1]`;
    - `attention_weights` (B, Nq, H, L, K),**调用方已 softmax**;
    - `chunk` > 0 时按 query 维切分计算,**只影响计算过程不影响结果**(单测钉了等价性);
      ⚠️ 但**实测不降显存、在小分块下反而升**(见模块头注),别拿它当省显存的手段。

    返回 (B, Nq, C)。越界采样由 `padding_mode="zeros"` 记 0(与官方一致)。
    """
    b, _, num_heads, head_dim = value.shape
    num_query = sampling_locations.shape[1]
    if num_query == 0:
        return value.new_zeros(b, 0, num_heads * head_dim)
    starts = [0]
    for sh, sw in spatial_shapes[:-1]:
        starts.append(starts[-1] + sh * sw)
    # [0,1] → [−1,1] 网格(官方 `2 · loc − 1`,`align_corners=False`)
    grid_all = 2.0 * sampling_locations - 1.0
    out = value.new_zeros(b, num_query, num_heads, head_dim)
    step = chunk if chunk > 0 else num_query
    for s in range(0, num_query, step):
        e = min(s + step, num_query)
        nq = e - s
        acc = value.new_zeros(b, num_heads, head_dim, nq)
        for lvl, (h, w) in enumerate(spatial_shapes):
            v_l = value[:, starts[lvl] : starts[lvl] + h * w]
            v_l = v_l.view(b, h, w, num_heads, head_dim).permute(0, 3, 4, 1, 2)
            v_l = v_l.reshape(b * num_heads, head_dim, h, w)
            g_l = grid_all[:, s:e, :, lvl].permute(0, 2, 1, 3, 4)  # (B, H, nq, K, 2)
            k = g_l.shape[3]
            g_l = g_l.reshape(b * num_heads, nq, k, 2)
            sampled = F.grid_sample(v_l, g_l, mode="bilinear", padding_mode="zeros", align_corners=False)
            w_l = attention_weights[:, s:e, :, lvl].permute(0, 2, 1, 3)  # (B, H, nq, K)
            acc += (
                (sampled * w_l.reshape(b * num_heads, 1, nq, k)).sum(dim=-1).view(b, num_heads, head_dim, nq)
            )
        out[:, s:e] = acc.permute(0, 3, 1, 2)
    return out.reshape(b, num_query, num_heads * head_dim)

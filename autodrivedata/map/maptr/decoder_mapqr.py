"""MapQR 解码器层 —— **scatter-and-gather** 口径(ECCV 2024,移植)。

**与默认线的关系**:默认的 [`MapTRDecoderLayer`](decoder.py) 在
[`decoder.py`](decoder.py);本文件是 **MapQR 变体**,两者**输出契约相同**、可互换装配
(见 [`head.MapTRHead`](head.py) 的 `scatter_gather` 开关)。开本支须**从头训练**
(参数名集合与基线不同,旧权重加载会被拦下)。

**为什么值得单独一支**:实测它与我们基线**同源** —— MapQR 是 MapTRv2 上的局部改动,
而它的贡献恰好打在我们自实现线的两处「工程简化」上(见下)。论文自称的
scatter-and-gather 机制上就是:instance query **散**成 P 个点查询 → 各自从 BEV 采样 →
**拼接**聚回 instance。

**必须显式归一化(最大的静默失效点)**:官方整套数学吃**归一化 `[0,1]` 坐标**,而本项目
坐标是**米**。米直接进 `sine_pos_embed` **不抛异常**,只让正弦频率**别名**、位置嵌入退化
成噪声 —— **症状是「训不动」而不是报错**。故本层把 `ref_pts` 先 `normalize_ref` 再用于
① 正弦位置嵌入、② deformable 参考点。`tests/map/test_head.py` 钉**调用点**、
`tests/map/test_deform_attn.py` 钉**语义**。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from autodrivedata.map.maptr.decoder import FFN
from autodrivedata.map.maptr.deform_attn import (
    build_sampling_locations,
    multi_scale_deform_attn,
    normalize_ref,
    sine_pos_embed,
)


class MapQRDecoderLayer(nn.Module):
    """一层解码器 —— MapQR **scatter-and-gather** 口径(默认层的对照实现)。

    与 `MapTRDecoderLayer` 的三处差异(自注意力 / FFN / 残差结构同构):

    1. **散**:点查询 = `实例查询 + pt_query_pos`,其中 `pt_query_pos` 来自**参考点的
       正弦位置嵌入**;默认层用的是可学习的点位置嵌入。
    2. **采**:deformable 注意力(**可学习采样偏移** + 多头注意力权重),替代默认层
       「固定位置双线性采样」(其头注自称 topk=1 简化)。
    3. **聚**:把 P 个点特征 `flatten(2)` **拼接**后经 MLP 压回实例,替代 `mean(dim=2)`。
       这一条是论文的着眼点 —— **均值会抹平同一要素内各点的内容差异**。

    有意分歧(不静默):回归仍在**米**空间(`new_ref = ref + d`),不搬官方的
    `inverse_sigmoid/sigmoid` 归一化构造;`pt_query_pos` 的输入按 `bev_range` **先归一化**
    (见模块头注)。
    """

    def __init__(
        self,
        embed_dims: int = 256,
        n_heads: int = 8,
        bev_dims: int = 256,
        num_pts: int = 20,
        n_attn_heads: int = 8,
        num_points: int = 4,
        num_reg_fcs: int = 2,
    ) -> None:
        super().__init__()
        if embed_dims % n_attn_heads != 0:
            raise ValueError(f"embed_dims({embed_dims}) 必须能被 n_attn_heads({n_attn_heads}) 整除")
        self.embed_dims = embed_dims
        self.num_pts = num_pts  # P:每条折线的点数(gather 与回归头的维度来源)
        self.n_attn_heads = n_attn_heads
        self.num_points = num_points  # K:deformable 每头每 level 的采样点数(≠ P)
        self.ins_attn = nn.MultiheadAttention(embed_dims, n_heads, batch_first=False)
        self.ins_ffn = FFN(embed_dims)
        # 参考点 sine 嵌入 → 点查询位置先验(官方逐层各一份 pt_pos_query_projs)
        self.pt_pos_proj = nn.Linear(embed_dims, embed_dims)
        # 可学习采样偏移 + 注意力权重(官方 InstancePointAttention 口径)
        self.sampling_offsets = nn.Linear(embed_dims, n_attn_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, n_attn_heads * num_points)
        self.value_proj = nn.Linear(bev_dims, embed_dims)
        # gather:P·C 拼接 → MLP → C(官方 output_proj 的三段式)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dims * num_pts, embed_dims * num_pts // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims * num_pts // 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )
        # 实例级回归:一次出全部 P 个点(默认层是逐点 MLP 出 (dx, dy))
        reg: list[nn.Module] = [nn.Linear(embed_dims, embed_dims * 2), nn.ReLU(inplace=True)]
        for _ in range(num_reg_fcs):
            reg += [nn.Linear(embed_dims * 2, embed_dims * 2), nn.ReLU(inplace=True)]
        reg.append(nn.Linear(embed_dims * 2, 2 * num_pts))
        self.reg_branch = nn.Sequential(*reg)
        self._init_weights()

    def _init_weights(self) -> None:
        """官方初始化口径:`sampling_offsets` 权重置零、偏置设成多尺度径向基(逐头转角度、
        逐采样点按距离放大),`attention_weights` 置零 ⇒ 初始时各头采样不同方向且权重均匀。"""
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.pt_pos_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=1.0)
        for m in self.output_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
        thetas = torch.arange(self.n_attn_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_attn_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)  # (heads, 2)
        # 先归一到最大分量 1(view 成 (heads,1,1,2))再沿采样点维 repeat —— 顺序不能反:
        # 直接 view 成 (heads,1,K,2) 元素数对不上(官方同式,`repeat` 不可省)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_attn_heads, 1, 1, 2)
            .repeat(1, 1, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.reshape(-1))

    def forward(
        self,
        q_ins: torch.Tensor,
        ref_pts: torch.Tensor,
        bev: torch.Tensor,
        point_embed: torch.Tensor | None,
        bev_range: tuple[float, float, float, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q_ins (Nq, B, C);ref_pts (B, Nq, P, 2) 米;bev (B, C_bev, H, W)。

        返回 (新 q_ins (Nq, B, C), 新 ref_pts (B, Nq, P, 2))。
        `point_embed` 仅为与 `MapTRDecoderLayer` **同签名**而保留,本层不使用 ——
        点查询的位置先验来自参考点的正弦嵌入(官方口径)。
        """
        b, nq, p = ref_pts.shape[:3]
        bh, bw = bev.shape[-2], bev.shape[-1]
        # ① 实例自注意力(与默认层同)
        q2, _ = self.ins_attn(q_ins, q_ins, q_ins)
        q_ins = self.ins_ffn(q_ins + q2) + q_ins + q2
        # ② 散:实例查询 → P 个点查询(加参考点的正弦位置嵌入)
        ref_norm = normalize_ref(ref_pts, bev_range)  # (B, Nq, P, 2) ∈ [0, 1]
        q_pt = q_ins.permute(1, 0, 2).unsqueeze(2) + self.pt_pos_proj(
            sine_pos_embed(ref_norm, self.embed_dims)
        )
        # ③ 采:可变形注意力(可学习偏移)
        q_flat = q_pt.reshape(b, nq * p, self.embed_dims)
        offsets = self.sampling_offsets(q_flat).view(b, nq * p, self.n_attn_heads, 1, self.num_points, 2)
        weights = self.attention_weights(q_flat)
        weights = weights.softmax(-1).view(b, nq * p, self.n_attn_heads, 1, self.num_points)
        loc = build_sampling_locations(ref_norm.reshape(b, nq * p, 1, 2), offsets, [(bh, bw)])
        value = self.value_proj(bev.flatten(2).transpose(1, 2))  # (B, H·W, C)
        value = value.view(b, bh * bw, self.n_attn_heads, self.embed_dims // self.n_attn_heads)
        attn = multi_scale_deform_attn(value, [(bh, bw)], loc, weights)  # (B, Nq·P, C)
        # ④ 聚:拼接 P 个点特征(不取均值)→ MLP 回实例
        gathered = self.output_proj(attn.view(b, nq, p * self.embed_dims))  # (B, Nq, C)
        q_ins = q_ins + gathered.permute(1, 0, 2)
        # ⑤ 实例级回归(米,绝对坐标口径 — 与默认层同)
        d = self.reg_branch(gathered).view(b, nq, p, 2)
        return q_ins, ref_pts + d

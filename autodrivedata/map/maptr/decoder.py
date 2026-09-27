"""MapTR(v2)解码器层 —— 自实现基线口径(§5.11d)。

**与 MapQR 变体的关系**:本文件是**默认线**,`MapQRDecoderLayer` 在
[`decoder_mapqr.py`](decoder_mapqr.py)(它对三处做了替换 —— 散 / 采 / 聚)。
两者**输出契约相同**、可互换装配(见 [`head.MapTRHead`](head.py) 的 `scatter_gather` 开关),
故下游(`train_maptr` / `eval_maptr` / 逐帧契约)对选哪一支无感。

一层做的事:

- 实例级自注意力 → 点级 BEV 几何采样(参考点 = 上层预测点,**双线性采样**,是官方
  deformable 交叉注意力的 topk=1 简化)→ 点回归 → **点特征均值回聚实例** → FFN。

`MLP` / `FFN` 两个构件放在本文件(MapTRv2 为基),`decoder_mapqr.py` 复用 `FFN` ——
依赖方向单向,不构成环。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """两段 MLP(in → hidden → out,ReLU 激活)。"""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FFN(nn.Module):
    """前馈网络(官方 ffn_ratio=4 口径,残差由调用方加)。两支解码器共用。"""

    def __init__(self, embed_dims: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims * 4, embed_dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MapTRDecoderLayer(nn.Module):
    """一层解码器:实例自注意力 → 点级 BEV 采样 → 点回归 → 实例回聚。"""

    def __init__(self, embed_dims: int = 256, n_heads: int = 8, bev_dims: int = 256) -> None:
        super().__init__()
        self.ins_attn = nn.MultiheadAttention(embed_dims, n_heads, batch_first=False)
        self.ins_ffn = FFN(embed_dims)
        self.pt_proj = nn.Linear(embed_dims, embed_dims)  # 实例特征 → 点查询基
        self.bev_proj = nn.Linear(bev_dims, embed_dims)  # BEV 采样特征 → 点查询空间
        self.agg_proj = nn.Linear(embed_dims, embed_dims)  # 点特征均值 → 实例残差
        self.reg_branch = MLP(embed_dims, embed_dims, 2)  # 点特征 → (dx, dy) 米

    def forward(
        self,
        q_ins: torch.Tensor,
        ref_pts: torch.Tensor,
        bev: torch.Tensor,
        point_embed: torch.Tensor,
        bev_range: tuple[float, float, float, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q_ins (Nq, B, C);ref_pts (B, Nq, P, 2) 米;bev (B, C_bev, H, W)。

        返回 (新 q_ins (Nq, B, C), 新 ref_pts (B, Nq, P, 2))。
        """
        q2, _ = self.ins_attn(q_ins, q_ins, q_ins)
        q_ins = self.ins_ffn(q_ins + q2) + q_ins + q2
        # 点查询:实例查询投影 + 可学习点位置嵌入(官方分层 query 同构)
        q_pt = self.pt_proj(q_ins).unsqueeze(2) + point_embed[None, None]  # (Nq, B, P, C)
        sampled = self._sample_bev(bev, ref_pts, bev_range)  # (Nq, B, P, C_bev)
        q_pt = q_pt + self.bev_proj(sampled)
        # 点回归(绝对坐标)
        d = self.reg_branch(q_pt)  # (Nq, B, P, 2)
        new_ref = ref_pts.permute(1, 0, 2, 3) + d
        new_ref = new_ref.permute(1, 0, 2, 3)
        # 点特征均值回聚实例
        q_ins = q_ins + self.agg_proj(q_pt.mean(dim=2))
        return q_ins, new_ref

    def _sample_bev(
        self, bev: torch.Tensor, ref_pts: torch.Tensor, bev_range: tuple[float, float, float, float]
    ) -> torch.Tensor:
        """参考点(米)→ BEV 双线性采样(官方 deformable 交叉注意力的 topk=1 简化)。

        bev (B, C, H, W) → (Nq, B, P, C)。BEV 网格覆盖 bev_range(xmin,ymin,xmax,ymax),
        与 GKT 的 BEVParams.pc_range 同口径;越界参考点 padding 0。
        输入不 expand(Nq·P 个点打平成 grid 的 H_out 维)——若把 BEV expand 成
        (B·Nq, C, H, W) 会物化 4GB 临时块(3080 Ti 12G 会 OOM)。

        **同时是 `tests/map/test_deform_attn.py` 的 oracle**:MapQR 那支的可变形注意力在
        「偏移全零 + 权重 one-hot」时必须与本方法**逐点相等**(两条独立实现互为对照)。
        """
        b, c = bev.shape[:2]
        nq, p = ref_pts.shape[1], ref_pts.shape[2]
        xmin, ymin, xmax, ymax = bev_range
        gx = ref_pts[..., 0] / ((xmax - xmin) / 2)  # x ∈ [xmin, xmax] → [−1, 1]
        gy = ref_pts[..., 1] / ((ymax - ymin) / 2)
        grid = torch.stack([gx, gy], dim=-1).view(b, nq * p, 1, 2)  # (B, Nq·P, 1, 2)
        out = F.grid_sample(bev, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        # out (B, C, Nq·P, 1) → (Nq, B, P, C)
        return out.squeeze(-1).view(b, c, nq, p).permute(2, 0, 3, 1)

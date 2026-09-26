"""MapTR 分层 query head(参考自实现)——实例级 query + 点级 query + 置换等价匹配。

忠实移植官方 MapTR 核心数学(§5.11d):
- 分层 query:每类 num_vec 个实例查询 + 每实例 num_pts 个点查询(点查询 =
  实例查询线性投影 + 可学习点位置嵌入,与官方 point_embedding 同构);
- 解码器逐层:实例级自注意力 → 点级 BEV 几何采样(参考点 = 上层预测点,双线性
  采样,官方 deformable 交叉注意力的 topk=1 简化)→ 点特征均值回聚实例 → FFN;
- 置换等价匹配:按类匈牙利 + 双向 GT 增强(正/反向各匹配一次取更优,MapTRv2
  §3.2 口径),代价 = 分类 −logit + 5·点级 L1;
- 损失:focal 分类(含背景)+ 点级 L1(仅匹配对,权重 5 = 官方 pts_loss_coef)。

工程简化(§5.11d 自实现口径):
1. 只取末层输出计算损失(官方逐层 aux);
2. 点回归预测绝对坐标(官方预测相对参考点偏移,数学等价——初始参考点可学习)。

类序与 MAPTR_CLASSES 一致:(divider, ped_crossing, boundary, centerline);
logits 类别 0 = 背景,1..num_classes = 上序。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import sigmoid_focal_loss

from autodrivedata.map.maptr.gkt import BEV_DEFAULT

# 官方 MapTR 口径:点损失系数 pts_loss_coef=5(匹配代价与损失共用)
PTS_COST_WEIGHT = 5.0

# 匈牙利未匹配对的填充代价(远大于任何真实代价)
_PAD_COST = 1e6


class MLP(nn.Module):
    """两段 MLP(in → hidden → out,ReLU 激活)。"""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FFN(nn.Module):
    """前馈网络(官方 ffn_ratio=4 口径,残差由调用方加)。"""

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


class MapTRHead(nn.Module):
    """MapTR head:输入 BEV 特征,输出实例分类 + 每实例 num_pts 个折线点。"""

    def __init__(
        self,
        num_classes: int = 4,
        embed_dims: int = 256,
        num_vec: int = 50,
        num_pts: int = 20,
        num_layers: int = 6,
        bev_dims: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_vec = num_vec
        self.num_pts = num_pts
        self.num_queries = num_classes * num_vec
        self.instance_embed = nn.Embedding(self.num_queries, embed_dims)  # 实例查询(每类一组)
        self.point_embed = nn.Embedding(num_pts, embed_dims)  # 点位置嵌入(官方同构,全层共享)
        self.anchor = nn.Parameter(torch.zeros(self.num_queries, num_pts, 2))  # 初始参考点(可学习锚线)
        self.cls_branch = nn.Linear(embed_dims, num_classes + 1)
        self.layers = nn.ModuleList(
            [MapTRDecoderLayer(embed_dims, bev_dims=bev_dims) for _ in range(num_layers)]
        )
        # 锚线初始化:每类 num_vec 个锚中心在 BEV 内均匀散布(10 列网格),点沿 x 轴
        # 铺开(线状先验)。num_vec ≠ 50 时取网格前 num_vec 个(单测用小配置)
        with torch.no_grad():
            rows = max(1, -(-num_vec // 10))  # ceil
            for c in range(num_classes):
                cx = torch.linspace(-10.0, 10.0, 10).repeat(rows)[:num_vec]
                cy = torch.linspace(-24.0, 24.0, rows).repeat_interleave(10)[:num_vec]
                pts = torch.stack(
                    [
                        cx[:, None] + (torch.arange(num_pts) - (num_pts - 1) / 2) * 0.4,
                        cy[:, None].expand(-1, num_pts),
                    ],
                    dim=-1,
                )
                self.anchor[c * num_vec : (c + 1) * num_vec] = pts

    def forward(self, bev: torch.Tensor) -> dict[str, torch.Tensor]:
        """bev (B, C_bev, H, W) → {"pred_logits": (B, Nq, C+1), "pred_points": (B, Nq, P, 2)}。"""
        b = bev.shape[0]
        q_ins = self.instance_embed.weight.unsqueeze(1).expand(-1, b, -1)  # (Nq, B, C)
        ref = self.anchor.unsqueeze(0).expand(b, -1, -1, -1)  # (B, Nq, P, 2)
        for layer in self.layers:
            q_ins, ref = layer(q_ins, ref, bev, self.point_embed.weight, BEV_DEFAULT.pc_range)
        logits = self.cls_branch(q_ins.permute(1, 0, 2))  # (B, Nq, C+1)
        return {"pred_logits": logits, "pred_points": ref}


def match_assign(
    pred_points: np.ndarray,
    pred_logits: np.ndarray,
    gt_by_class: list[list[np.ndarray]],
    num_classes: int,
    num_vec: int,
    pts_cost_weight: float = PTS_COST_WEIGHT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按类匈牙利匹配(MapTRv2 置换等价口径:GT 双向增强) → 训练目标。

    pred_points (Nq, P, 2) 米;pred_logits (Nq, num_classes+1);gt_by_class
    每类 GT 折线 list of (P, 2)。返回:
    - cls_targets (Nq,) int:0 = 背景,1..C = 类别
    - pts_targets (Nq, P, 2):匹配对的 GT 点(取更优方向);未匹配行置 0
    - pts_mask (Nq,) bool:是否参与点损失
    """
    cls_targets = np.zeros(num_classes * num_vec, dtype=np.int64)
    pts_mask = np.zeros(num_classes * num_vec, dtype=bool)
    pts_targets = np.zeros((num_classes * num_vec, pred_points.shape[1], 2), dtype=np.float32)
    for c in range(num_classes):
        gts = gt_by_class[c]
        q0 = c * num_vec
        if not gts:
            continue
        n_gt = len(gts)
        pts = pred_points[q0 : q0 + num_vec]  # (num_vec, P, 2)
        # 分类代价:该 query 组固定属于类 c,−logit(c+1) 越小越该匹配正样本
        cls_cost = -pred_logits[q0 : q0 + num_vec, c + 1][:, None]  # (num_vec, 1)
        # 点代价:双向 GT(置换等价),取更优方向
        pts_cost = np.empty((num_vec, n_gt))
        orient = np.zeros((num_vec, n_gt), dtype=bool)  # True = 反向
        for j, gt in enumerate(gts):
            fwd = np.abs(pts - gt[None]).sum(axis=-1).mean(axis=1)  # (num_vec,)
            rev = np.abs(pts - gt[::-1][None]).sum(axis=-1).mean(axis=1)
            orient[:, j] = rev < fwd
            pts_cost[:, j] = np.minimum(fwd, rev)
        cost = cls_cost + pts_cost_weight * pts_cost  # (num_vec, n_gt)
        # 方阵化(行/列 pad 大代价;列数可能 > num_vec,如 centerline 163 条)
        dim = max(num_vec, n_gt)
        square = np.full((dim, dim), _PAD_COST)
        square[:num_vec, :n_gt] = cost
        rows, cols = linear_sum_assignment(square)
        for k, j in zip(rows, cols, strict=True):
            if k >= num_vec or j >= n_gt:
                continue  # pad 行/列(大代价,正常不会匹配到)
            cls_targets[q0 + k] = c + 1
            pts_mask[q0 + k] = True
            pts_targets[q0 + k] = gts[j][::-1] if orient[k, j] else gts[j]
    return cls_targets, pts_targets, pts_mask


def maptr_loss(
    pred_logits: torch.Tensor,
    pred_points: torch.Tensor,
    cls_targets: torch.Tensor,
    pts_targets: torch.Tensor,
    pts_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """MapTR 损失:focal 分类(含背景)+ 点级 L1(仅匹配对,权重 PTS_COST_WEIGHT)。

    输入按 batch 堆叠:pred_logits (B, Nq, C+1),pred_points (B, Nq, P, 2),
    cls_targets (B, Nq) long,pts_targets (B, Nq, P, 2),pts_mask (B, Nq) bool。
    """
    _, _, c1 = pred_logits.shape
    onehot = F.one_hot(cls_targets.reshape(-1), num_classes=c1).to(pred_logits.dtype)
    cls_loss = sigmoid_focal_loss(
        pred_logits.reshape(-1, c1), onehot, alpha=0.25, gamma=2.0, reduction="mean"
    )
    mask = pts_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_points[..., :1]).reshape(-1)
    pts_loss = F.l1_loss(
        pred_points.reshape(-1, pred_points.shape[-1])[mask], pts_targets.reshape(-1, 2)[mask]
    )
    return {
        "cls": cls_loss,
        "pts": PTS_COST_WEIGHT * pts_loss,
        "total": cls_loss + PTS_COST_WEIGHT * pts_loss,
    }

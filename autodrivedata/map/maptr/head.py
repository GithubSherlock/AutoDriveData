"""MapTR head —— 分层 query 外壳(实例级 query + 置换等价匹配 + 损失)。

**两套解码器按文件分开**(2026-09-27),本文件只做**装配**与共用部分:

| 文件 | 内容 |
|---|---|
| [`decoder.py`](decoder.py) | `MapTRDecoderLayer` + 构件 `MLP` / `FFN` —— **默认线**(MapTRv2 口径) |
| [`decoder_mapqr.py`](decoder_mapqr.py) | `MapQRDecoderLayer` —— **MapQR scatter-and-gather 口径** |
| 本文件 | `MapTRHead`(按 `scatter_gather` 开关选解码器)+ `match_assign` + `maptr_loss` |

两支**输出契约相同**(每层签名一致),故 `MapTRHead` 只换 `layers` 的构件类型,
下游(`train_maptr` / `eval_maptr` / 逐帧契约 `mapvec_pred/1`)对选哪一支无感。
默认 `scatter_gather=False` ⇒ 与加变体之前**逐位一致**;开 MapQR 支须**从头训练**
(参数名集合不同,`load_map_weights` 会把旧权重拦下)。

**共用部分**(两支同构,与选哪支无关):每类 `num_vec` 个实例查询、可学习初始锚线
`anchor`、实例级分类头 `cls_branch`;匹配 = 按类匈牙利 + 双向 GT 增强(MapTRv2 §3.2),
代价 = 分类 −logit + 5·点级 L1;损失 = focal 分类(含背景)+ 点级 L1(仅匹配对)。

工程简化(§5.11d 自实现口径):① 只取末层输出计算损失(官方逐层 aux);
② 点回归预测**绝对坐标**(官方预测相对参考点偏移,数学等价——初始参考点可学习)。

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

from autodrivedata.map.maptr.decoder import MapTRDecoderLayer
from autodrivedata.map.maptr.decoder_mapqr import MapQRDecoderLayer
from autodrivedata.map.maptr.gkt import BEV_DEFAULT

# 官方 MapTR 口径:点损失系数 pts_loss_coef=5(匹配代价与损失共用)
PTS_COST_WEIGHT = 5.0

# 匈牙利未匹配对的填充代价(远大于任何真实代价)
_PAD_COST = 1e6


class MapTRHead(nn.Module):
    """MapTR head:输入 BEV 特征,输出实例分类 + 每实例 num_pts 个折线点。

    `scatter_gather=True` 时换成 MapQR 的 scatter-and-gather 解码器(见
    `MapQRDecoderLayer`);**输出契约与默认分支完全一致**。
    """

    def __init__(
        self,
        num_classes: int = 4,
        embed_dims: int = 256,
        num_vec: int = 50,
        num_pts: int = 20,
        num_layers: int = 6,
        bev_dims: int = 256,
        scatter_gather: bool = False,
        n_attn_heads: int = 8,
        num_points: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_vec = num_vec
        self.num_pts = num_pts
        self.num_queries = num_classes * num_vec
        self.scatter_gather = scatter_gather
        self.instance_embed = nn.Embedding(self.num_queries, embed_dims)  # 实例查询(每类一组)
        self.anchor = nn.Parameter(torch.zeros(self.num_queries, num_pts, 2))  # 初始参考点(可学习锚线)
        self.cls_branch = nn.Linear(embed_dims, num_classes + 1)
        if scatter_gather:
            # 点查询位置先验改用参考点正弦嵌入(官方 instance 口径),可学习点嵌入不再需要
            self.point_embed = None
            self.layers = nn.ModuleList(
                [
                    MapQRDecoderLayer(
                        embed_dims, bev_dims=bev_dims, num_pts=num_pts, n_attn_heads=n_attn_heads
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.point_embed = nn.Embedding(num_pts, embed_dims)  # 点位置嵌入(官方同构,全层共享)
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
        point_embed = None if self.point_embed is None else self.point_embed.weight
        for layer in self.layers:
            q_ins, ref = layer(q_ins, ref, bev, point_embed, BEV_DEFAULT.pc_range)
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

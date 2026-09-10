"""MapTR 模型组装——ResNet50+FPN backbone + GKT BEV 变换 + 分层 query head。

官方 MapTR 结构同构:backbone(ResNet50+FPN)→ GKT 视角变换 → 分层 query head,
中间不设 BEV encoder neck(官方即无此层,BEV 空间推理全部交给 head 的点级采样)。
GKT 采样 FPN 最小 stride 层(P2,1/4),与官方多尺度采样相比是 §5.11d 自实现口径
的工程简化(采样几何链一致,单测锁定)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from maptr_impl.gkt import BEV_DEFAULT, GKT
from maptr_impl.head import MapTRHead

# FPN 特征键:torchvision resnet_fpn_backbone 输出 "0"(1/4)/"1"/"2"/"3"/"pool"
FPN_LEVEL = "0"


class MapTR(nn.Module):
    """MapTR 参考实现:6 相机图像 → 四类矢量折线(实例分类 + 20 点坐标)。"""

    def __init__(
        self,
        num_classes: int = 4,
        embed_dims: int = 256,
        num_vec: int = 50,
        num_pts: int = 20,
        num_layers: int = 6,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=weights)
        self.gkt = GKT(BEV_DEFAULT)
        self.head = MapTRHead(
            num_classes=num_classes,
            embed_dims=embed_dims,
            num_vec=num_vec,
            num_pts=num_pts,
            num_layers=num_layers,
            bev_dims=256,  # FPN P2 通道数
        )
        self.num_classes = num_classes
        self.num_vec = num_vec
        self.num_pts = num_pts

    def forward(
        self,
        images: dict[str, torch.Tensor],
        poses: torch.Tensor,
        calibs: dict[str, dict],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """images 相机名 → (B, 3, H, W)(归一化 RGB);poses (B, 6) 度;calibs = B2 口径。

        返回 (head 输出 {"pred_logits", "pred_points"}, bev_valid (B, 1, H, W))。
        """
        feats = {name: self.backbone(x)[FPN_LEVEL] for name, x in images.items()}
        bev, valid = self.gkt(feats, poses, calibs)
        return self.head(bev), valid

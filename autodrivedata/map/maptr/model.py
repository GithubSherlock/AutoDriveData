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

from autodrivedata.map.maptr.gkt import BEV_DEFAULT, GKT
from autodrivedata.map.maptr.head import MapTRHead
from autodrivedata.map.maptr.temporal import TemporalFusion, warp_bev

# FPN 特征键:torchvision resnet_fpn_backbone 输出 "0"(1/4)/"1"/"2"/"3"/"pool"
FPN_LEVEL = "0"


def load_map_weights(model: MapTR, path: str, dev: torch.device) -> list[str]:
    """载入 state_dict,按**口径**分类报缺失/多余键,返回缺失键(训练脚本用来报热启动)。

    时序版比单帧多一层 `fusion.proj.*`:
    - 单帧权重 → 时序模型:它必然缺失,这是**有意**的热启动,放行;
    - 时序权重 → 单帧模型:多出 `fusion.*` ⇒ **报错**。静默丢掉会把时序权重跑成
      单帧模型,AP 差异看着像"时序没用"。
    其余任何缺失/多余键都是真错(改了结构或拿错文件),一律报错。
    """
    missing, unexpected = model.load_state_dict(torch.load(path, map_location=dev), strict=False)
    if unexpected:
        raise SystemExit(f"{path} 有多余权重 {sorted(unexpected)}(模型口径不匹配?)")
    if missing and not all(k.startswith("fusion.") for k in missing):
        raise SystemExit(f"{path} 缺权重 {sorted(missing)}")
    return sorted(missing)


class MapTR(nn.Module):
    """MapTR 参考实现:6 相机图像 → 四类矢量折线(实例分类 + 20 点坐标)。

    `temporal_window > 1` 时启用 MapTRv2 时序版:过去 K−1 帧的 BEV 扭到当前 ego 系
    后与当前帧融合(见 `autodrivedata/map/maptr/temporal.py`)。**头部一字不改** —— 融合只作用在
    BEV 上,故单帧与时序的差异可完全归因到 BEV 特征,不发生"顺手换了个 head"。
    """

    def __init__(
        self,
        num_classes: int = 4,
        embed_dims: int = 256,
        num_vec: int = 50,
        num_pts: int = 20,
        num_layers: int = 6,
        pretrained: bool = True,
        temporal_window: int = 1,
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
        if temporal_window < 1:
            raise ValueError(f"temporal_window 需 ≥ 1,收到 {temporal_window}")
        self.temporal_window = temporal_window
        self.fusion = TemporalFusion(256, temporal_window - 1) if temporal_window > 1 else None
        self.num_classes = num_classes
        self.num_vec = num_vec
        self.num_pts = num_pts

    def forward(
        self,
        images: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
        poses: torch.Tensor,
        calibs: dict[str, dict],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """images 相机名 → (B, 3, H, W)(归一化 RGB);poses (B, 6) 度;calibs = B2 口径。

        返回 (head 输出 {"pred_logits", "pred_points"}, bev_valid (B, 1, H, W))。

        `calibs` 的 `intrinsic` 是**原图**口径(与 B2 infos 一致);图像尺寸从 `images`
        自取传给 GKT 做 K 缩放——调用方不必也不能自己缩(见 gkt 模块头注坑 2)。

        时序模式:`images` 为**长度 K 的列表**(旧 → 新,见 `dataset.collate`)、`poses`
        为 (B, K, 6);此时 `temporal_window` 必须匹配(不匹配由 `TemporalFusion` 报错)。
        """
        if isinstance(images, list):
            return self._forward_temporal(images, poses, calibs)
        if self.temporal_window > 1:
            raise ValueError(f"模型是时序版(window={self.temporal_window}),但只收到单帧图像")
        bev, valid = self._forward_frames(images, poses, calibs)
        return self.head(bev), valid

    def _forward_frames(
        self, images: dict[str, torch.Tensor], poses: torch.Tensor, calibs: dict[str, dict]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单帧:backbone → GKT → BEV(不做 head)。"""
        first = next(iter(images.values()))
        img_size = (int(first.shape[-1]), int(first.shape[-2]))
        feats = {name: self.backbone(x)[FPN_LEVEL] for name, x in images.items()}
        return self.gkt(feats, poses, calibs, img_size)

    def _forward_temporal(
        self, images: list[dict[str, torch.Tensor]], poses: torch.Tensor, calibs: dict[str, dict]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """K 帧(旧 → 新)→ 历史 BEV 扭到当前系 → 融合 → head。

        **历史帧在 `torch.no_grad()` 下编码**(= 推理期 memory bank 语义,官方
        MapTRv2 同):激活显存 ≈ 单帧 + K−1 个小 BEV,而不是 K 倍单帧。副作用是
        历史帧的 backbone 前向仍会更新 BN running stats(与官方同)—— 不额外处理,
        因为把历史单独切 eval() 会让同一 backbone 在两处用不同 BN 口径,反而引入
        训练/推理不一致。
        """
        if self.fusion is None:
            raise ValueError("单帧模型收到 K>1 帧:请用 temporal_window>1 构造")
        # 必须显式要求 (B, K, 6):给 (K, 6) 时 `poses[:, j]` 会静默取出**第 j 列**(B 个数)
        # 而不是第 j 帧,一路传到 GKT 才因维度不符报错,堆栈指向 cam_world_pose 而不是这里
        if poses.dim() != 3:
            raise ValueError(f"时序模式需要 poses (B, K, 6),收到 {tuple(poses.shape)}")
        n_hist = poses.shape[1] - 1
        if len(images) != poses.shape[1]:
            raise ValueError(f"帧数不一致:images {len(images)} 帧,poses {poses.shape[1]} 帧")
        pose_cur = poses[:, -1]
        hist = []
        with torch.no_grad():
            for j in range(n_hist):
                bev_j, _ = self._forward_frames(images[j], poses[:, j], calibs)
                hist.append(warp_bev(bev_j, poses[:, j], pose_cur, BEV_DEFAULT))
        bev, valid = self._forward_frames(images[-1], pose_cur, calibs)
        return self.head(self.fusion(bev, hist)), valid

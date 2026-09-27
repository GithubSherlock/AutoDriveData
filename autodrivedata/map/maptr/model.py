"""MapTR 模型组装——ResNet50+FPN backbone + GKT BEV 变换 + 分层 query head。

官方 MapTR 结构同构:backbone(ResNet50+FPN)→ GKT 视角变换 → 分层 query head,
中间不设 BEV encoder neck(官方即无此层,BEV 空间推理全部交给 head 的点级采样)。
GKT 采样 FPN 最小 stride 层(P2,1/4),与官方多尺度采样相比是 §5.11d 自实现口径
的工程简化(采样几何链一致,单测锁定)。

**两个 MapQR 变体开关(2026-09-27,默认全关 = 原行为)**:

- `scatter_gather=True` —— head 换成 MapQR 的解码器(散 / 采 / 聚三处,见 `head.py`);
- `bev_encoder="height_kernel"` —— 在 GKT 与 head 之间插一级 BEV 细化(见 `bevenc.py`)。

两者都在**同一套输出契约**下工作,故 `train_maptr` / `eval_maptr` / 逐帧契约无感。
**开了任一开关就必须从头训练**:参数名集合变了,`load_map_weights` 会把旧权重当
「多余/缺失键」拦下(这是有意的,不静默丢键)。

`bev_encoder` 与 `temporal_window > 1` **互斥**:官方的时序发生在 BEV encoder 内部
(`prev_bev`),叠在我们的 `TemporalFusion` 上会让"改的是哪一处 BEV"无从归因。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from autodrivedata.map.maptr.bevenc import BEVEncoder
from autodrivedata.map.maptr.gkt import BEV_DEFAULT, GKT
from autodrivedata.map.maptr.head import MapTRHead
from autodrivedata.map.maptr.temporal import TemporalFusion, warp_bev

BEV_ENCODERS = ("none", "height_kernel")

# FPN 特征键:torchvision resnet_fpn_backbone 输出 "0"(1/4)/"1"/"2"/"3"/"pool"
FPN_LEVEL = "0"


def read_map_state_dict(path: str, dev: torch.device) -> dict:
    """读 checkpoint 的 **state_dict**(兼容新旧两种落盘形态)。

    新形态由 [`variants.save_map_checkpoint`](variants.py) 写出:
    `{"state_dict": …, "model_kwargs": {…}}`;旧形态是**裸 state_dict**(本项目 2026-09-27
    之前的所有权重,如 `maptr_v2_singleF.pt`)。判别键就是字面量 `"state_dict"` ——
    state_dict 的键永远是参数名,不会撞上它。
    """
    obj = torch.load(path, map_location=dev)
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        return obj["state_dict"]
    return obj


def load_map_weights(model: MapTR, path: str, dev: torch.device) -> list[str]:
    """载入 state_dict,按**口径**分类报缺失/多余键,返回缺失键(训练脚本用来报热启动)。

    时序版比单帧多一层 `fusion.proj.*`:
    - 单帧权重 → 时序模型:它必然缺失,这是**有意**的热启动,放行;
    - 时序权重 → 单帧模型:多出 `fusion.*` ⇒ **报错**。静默丢掉会把时序权重跑成
      单帧模型,AP 差异看着像"时序没用"。
    其余任何缺失/多余键都是真错(改了结构或拿错文件),一律报错。
    """
    missing, unexpected = model.load_state_dict(read_map_state_dict(path, dev), strict=False)
    # 开了 MapQR 变体时参数名集合与基线不同 —— 这条报错信息要点明"需从头训练",
    # 否则看到「多余权重」的人会先去怀疑权重文件损坏,白查一轮。
    hint = (
        "  ← 当前模型开了 MapQR 变体(scatter_gather / bev_encoder),参数名与基线不同,"
        "必须从头训练,该权重不可热启动"
        if getattr(model, "scatter_gather", False) or getattr(model, "bev_encoder", None) is not None
        else ""
    )
    if unexpected:
        raise SystemExit(f"{path} 有多余权重 {sorted(unexpected)}(模型口径不匹配?){hint}")
    if missing and not all(k.startswith("fusion.") for k in missing):
        raise SystemExit(f"{path} 缺权重 {sorted(missing)}{hint}")
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
        scatter_gather: bool = False,
        bev_encoder: str = "none",
        bev_encoder_layers: int = 3,
        bev_encoder_heads: int = 8,
        bev_chunk: int = 0,
        num_cams: int = 6,
    ) -> None:
        super().__init__()
        if bev_encoder not in BEV_ENCODERS:
            raise ValueError(f"bev_encoder 需为 {BEV_ENCODERS} 之一,收到 {bev_encoder!r}")
        if bev_encoder != "none" and temporal_window > 1:
            raise ValueError(
                "bev_encoder 与 temporal_window > 1 互斥:官方的时序发生在 BEV encoder 内部,"
                "叠上 TemporalFusion 会让归因无从下手(见模块头注)"
            )
        if temporal_window < 1:
            raise ValueError(f"temporal_window 需 ≥ 1,收到 {temporal_window}")
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=weights)
        self.gkt = GKT(BEV_DEFAULT)
        self.bev_encoder = (
            BEVEncoder(
                num_layers=bev_encoder_layers,
                embed_dims=embed_dims,
                n_heads=bev_encoder_heads,
                num_cams=num_cams,
                # 官方隐式约束 D == heads:直接由 heads 派生,避免配置出不一致的组合
                num_points_in_pillar=bev_encoder_heads,
                chunk=bev_chunk,
            )
            if bev_encoder == "height_kernel"
            else None
        )
        self.head = MapTRHead(
            num_classes=num_classes,
            embed_dims=embed_dims,
            num_vec=num_vec,
            num_pts=num_pts,
            num_layers=num_layers,
            bev_dims=256,  # FPN P2 通道数
            scatter_gather=scatter_gather,
        )
        self.temporal_window = temporal_window
        self.fusion = TemporalFusion(256, temporal_window - 1) if temporal_window > 1 else None
        # 架构参数全部留存 —— `model_kwargs()` 靠它们写出「这份权重是怎么建出来的」
        # (不含 `pretrained`:那是训练期的事,载权重时不该再去拉 ImageNet)。
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_vec = num_vec
        self.num_pts = num_pts
        self.num_layers = num_layers
        self.scatter_gather = scatter_gather
        self.bev_encoder_name = bev_encoder
        self.bev_encoder_layers = bev_encoder_layers
        self.bev_encoder_heads = bev_encoder_heads
        self.bev_chunk = bev_chunk
        self.num_cams = num_cams

    def model_kwargs(self) -> dict:
        """重建本模型所需的**架构参数**(可直接喂回 `MapTR(**kwargs)`)。

        写进 checkpoint ⇒ 权重自带结构说明,eval / viz / studio 不必被额外告知变体
        (见 [`variants.py`](variants.py))。**不含 `pretrained`**:载权重时不该再拉 ImageNet。
        `bev_encoder_name` 落盘时改回构造参数名 `bev_encoder`(模型属性与构造参数同名会
        遮蔽 `self.bev_encoder` 那个子模块,故属性名加了 `_name`)。
        """
        return {
            "num_classes": self.num_classes,
            "embed_dims": self.embed_dims,
            "num_vec": self.num_vec,
            "num_pts": self.num_pts,
            "num_layers": self.num_layers,
            "temporal_window": self.temporal_window,
            "scatter_gather": self.scatter_gather,
            "bev_encoder": self.bev_encoder_name,
            "bev_encoder_layers": self.bev_encoder_layers,
            "bev_encoder_heads": self.bev_encoder_heads,
            "bev_chunk": self.bev_chunk,
            "num_cams": self.num_cams,
        }

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
        """单帧:backbone → GKT → [BEV 细化] → BEV(不做 head)。"""
        first = next(iter(images.values()))
        img_size = (int(first.shape[-1]), int(first.shape[-2]))
        feats = {name: self.backbone(x)[FPN_LEVEL] for name, x in images.items()}
        bev, valid = self.gkt(feats, poses, calibs, img_size)
        if self.bev_encoder is not None:
            # 复用 GKT 的相机特征,不重跑 backbone;`valid` 由 GKT 定义、细化不改可见性
            bev = self.bev_encoder(bev, feats, poses, calibs, img_size)
        return bev, valid

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

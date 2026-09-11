"""B2 infos json → MapTR 训练数据集(图像加载 + 位姿/标定透传 + GT 解析)。

消费 assemble_maptr 输出的 infos(帧级 ego2global + cams 内外参 + annotation
四类矢量,ego 局部系),产出模型前向所需的三件套:
- images: 相机名 → (B, 3, H, W) 归一化 RGB(ImageNet 口径,与 torchvision
  backbone 惯例一致)
- poses: (B, 6) [x, y, z, yaw, pitch, roll] 度
- gts: 每样本 list[list[np.ndarray]] —— 每类 GT 折线 (20, 2) ego 系,类序
  MAPTR_CLASSES(divider, ped_crossing, boundary, centerline)

calibs(批次共享)从首帧 cams 直出,与 GKT 输入口径零转换。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize, to_tensor

from autodrivedata.mapvec import MAPTR_CLASSES

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


class MapTRDataset(Dataset):
    """B2 infos 子集(或全量)数据集;--frames N 取前 N 帧实现单帧过拟合。"""

    def __init__(self, infos: list[dict], root: str | Path, frames: list[int] | None = None) -> None:
        super().__init__()
        self.root = Path(root)
        self.infos = [infos[i] for i in frames] if frames is not None else infos
        if not self.infos:
            raise ValueError("infos 为空")
        self.cam_names = sorted(self.infos[0]["cams"])
        self.calibs = self.infos[0]["cams"]

    def __len__(self) -> int:
        return len(self.infos)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, idx: int) -> dict:
        info = self.infos[idx]
        images = {name: self._load(self.root / info["cams"][name]["data_path"]) for name in self.cam_names}
        pose = torch.tensor(info["ego2global"], dtype=torch.float32)
        ann = info["annotation"]
        gt = [[np.asarray(line, dtype=np.float32) for line in ann[cls]] for cls in MAPTR_CLASSES]
        return {"images": images, "pose": pose, "gt": gt}

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        t = to_tensor(img)  # (3, H, W) [0, 1]
        return normalize(t, IMAGENET_MEAN, IMAGENET_STD)


def collate(batch: list[dict]) -> dict:
    """批次组装:images 按相机堆叠,poses 堆叠,gts 保持列表(匹配在样本级做)。"""
    names = list(batch[0]["images"])
    return {
        "images": {n: torch.stack([b["images"][n] for b in batch]) for n in names},
        "poses": torch.stack([b["pose"] for b in batch]),
        "gts": [b["gt"] for b in batch],
    }

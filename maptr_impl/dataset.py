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

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize, to_tensor

from autodrivedata.mapvec import MAPTR_CLASSES

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


def parse_segs(spec: str | None) -> list[str] | None:
    """`"seg0,seg1"` → `["seg0", "seg1"]`;`None` / 空串 → `None`(= 不按段过滤)。"""
    if not spec:
        return None
    segs = [s.strip() for s in spec.split(",") if s.strip()]
    return segs or None


def parse_frame_range(spec: str | None) -> tuple[int, int] | None:
    """`"80:100"` → `(80, 100)`;`None` / 空串 → `None`。区间**左闭右开**。"""
    if not spec:
        return None
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"--keep-in-seg 应为 A:B,收到 {spec!r}")
    lo, hi = (int(x) for x in parts)
    if not 0 <= lo < hi:
        raise ValueError(f"--keep-in-seg 需 0 <= A < B,收到 {spec!r}")
    return lo, hi


def select_frames(
    infos: list[dict],
    segs: Sequence[str] | None = None,
    exclude_segs: Sequence[str] = (),
    keep_in_seg: tuple[int, int] | None = None,
) -> list[int]:
    """infos → 训练/评测用的**下标**列表(留出划分的唯一落点,train/eval 共用)。

    三个选择器取交集;不带任何选择器时返回全部下标(缺 `seg` 键的**旧单段 infos**
    因此行为一字不变 —— 老命令 `--surround <一个目录> --frames 300` 照旧)。
    `--frames` / `--start` 由调用方在本列表上再截,不走这里。

    为什么划分逻辑放在这里而不是各脚本里:路线级(`--exclude-seg seg4`)与帧级
    (`--keep-in-seg 80:100`)两套留出口径必须**同一份实现** —— 训练与评测各写
    一遍必然漂移,而"训练集与留出集有无交集"是全部 AP 结论有效性的前提。
    段名拼错会**报错而不是静默给 0 帧**(静默空集会训出一个空模型还看着"跑完了")。
    """
    if not infos:
        return []
    # 每个选择器只校验**自己用到的键**:按 seg 过滤才要求 "seg",按帧号才要求
    # "frame_in_seg"。写成"任意选择器都查 seg"会让 keep_in_seg 在旧 infos 上报错,
    # 也会让不带选择器的调用去读不存在的键(后者正是本函数第一版的真 bug)
    by_seg = segs is not None or bool(exclude_segs)
    if by_seg and not all("seg" in i for i in infos):
        raise ValueError("infos 缺 'seg' 键(旧版单段 assemble 产物),不支持段级划分")
    if keep_in_seg is not None and not all("frame_in_seg" in i for i in infos):
        raise ValueError("infos 缺 'frame_in_seg' 键,不支持帧级划分")
    if by_seg:
        known = {i["seg"] for i in infos}
        unknown = (set(segs or ()) | set(exclude_segs)) - known
        if unknown:
            raise ValueError(f"段名不存在:{sorted(unknown)}(本 infos 有 {sorted(known)})")
    sel = []
    for i, info in enumerate(infos):
        if by_seg and (info["seg"] in exclude_segs or (segs is not None and info["seg"] not in segs)):
            continue
        if keep_in_seg is not None:
            f = info["frame_in_seg"]
            if not keep_in_seg[0] <= f < keep_in_seg[1]:
                continue
        sel.append(i)
    return sel


def history_windows(
    infos: list[dict], sel: list[int], window: int
) -> tuple[list[tuple[int, ...]], list[int]]:
    """`(可用窗口, 丢弃下标)`;每个窗口是 infos 下标的元组,长度 window、**旧 → 新**,
    末元素 = 该帧自身。`window=1` 即 `[(i,) for i in sel]`。

    ★ **守卫的边界是"切分"不是"段"**(2026-09-24 由对抗复核纠正,原计划写错了)。

    原计划的不变量是"窗口不跨 `seg`" —— 那条**不够**,而且漏的正是最该防的地方:
    帧级切分画在 `frame_in_seg == 80`,**落在每段内部**,于是留出帧 80 的历史是 79/78,
    而 `--keep-in-seg 0:80` 把 79/78 留在了训练集里(4 段 × 2 帧 = 80 帧留出里的 **8 帧**,
    见 `outputs/surround_v2/map_infos.json` 实测:`frame` 全局连续 0..499、列表序号 == frame、
    段缝在 99→100)。stride 5 = 0.5 s ⇒ 帧间只走 ~3 m、warp 近乎恒等 ⇒ 模型可以直接复制
    它在 78/79 上背下来的地图。**症状**:按 `frame_in_seg` 分箱的 AP 在 80/81 上翘起、
    ~83 回落到基线;而训练日志(损失单调下降、留出确实是 80 帧)看不出任何异常。

    本函数的做法是**结构上不可能泄漏**:窗口只从**本次 `sel`(即本切分自己的帧列表)**
    里取前驱,并且要求前驱落在**同一 `seg`** 内。训练集与留出集各自调用一次、互相看不见
    对方的帧 ⇒ 不需要任何额外的孤立/禁运记账。拿不到完整窗口的帧**被丢弃并计数上报**
    (不是静默截短 —— 静默截短会让"覆盖了全部"变成假话)。

    段内守卫也不能用 `frame` 算:它在段缝处**连续无缺口**(0..499),`infos[idx-1]` 在
    idx=100 处会静默解析到 seg0 的最后一帧。所以键一律是 `(seg, frame_in_seg)`。
    """
    if window < 1:
        raise ValueError(f"window 需 ≥ 1,收到 {window}")
    if window == 1:
        return [(i,) for i in sel], []
    pos = {(infos[i]["seg"], infos[i]["frame_in_seg"]): i for i in sel}
    windows: list[tuple[int, ...]] = []
    dropped: list[int] = []
    for i in sel:
        seg, f = infos[i]["seg"], infos[i]["frame_in_seg"]
        hist: list[int] = []
        for k in range(window - 1, 0, -1):  # 旧 → 近
            j = pos.get((seg, f - k))
            if j is None:  # 前驱不在本切分内(或不在本段内)⇒ 整帧丢弃,不截短
                break
            hist.append(j)
        else:
            windows.append((*hist, i))
            continue
        dropped.append(i)
    return windows, dropped


class MapTRDataset(Dataset):
    """B2 infos 子集(或全量)数据集;--frames N 取前 N 帧实现单帧过拟合。

    `window=1`(默认)返回单帧,结构与既有单帧基线**逐字节相同**;
    `window=K>1` 额外返回 K 帧(本帧 + 前 K−1 帧,旧 → 新)供时序融合用,
    窗口合法性由 `history_windows` 保证(不跨段、不跨切分)。
    """

    def __init__(
        self, infos: list[dict], root: str | Path, frames: list[int] | None = None, window: int = 1
    ) -> None:
        super().__init__()
        self.root = Path(root)
        sel = list(frames) if frames is not None else list(range(len(infos)))
        if window > 1 and not all({"seg", "frame_in_seg"} <= set(infos[i]) for i in sel):
            raise ValueError("window>1 需要 infos 带 'seg'/'frame_in_seg'(旧版单段 assemble 产物不支持)")
        self.window = window
        self.dropped: list[int] = []
        self.windows: list[tuple[int, ...]] | None = None
        if window > 1:
            self.windows, self.dropped = history_windows(infos, sel, window)
            src = infos
        else:
            src = [infos[i] for i in sel]
        self._src = src  # 窗口用:保留原 infos 引用(下标是全局的)
        self.infos = [src[w[-1]] for w in self.windows] if self.windows is not None else src
        if not self.infos:
            raise ValueError(f"infos 为空(窗口 {window} 下没有一帧拿得到完整历史)")
        self.cam_names = sorted(self.infos[0]["cams"])
        self.calibs = self.infos[0]["cams"]

    def __len__(self) -> int:
        return len(self.infos)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def _frames(self, idx: int) -> list[dict]:
        """该样本的帧序列(旧 → 新);window=1 时就是自己一帧。"""
        if self.windows is None:
            return [self.infos[idx]]
        return [self._src[i] for i in self.windows[idx]]

    def __getitem__(self, idx: int) -> dict:
        info = self.infos[idx]
        if self.window == 1:
            return {
                "images": self._images(info),
                "pose": self._pose(info),
                "gt": self._gt(info),
            }
        frames = self._frames(idx)
        return {
            # 旧 → 新:API 顺序由 TemporalFusion 承担(它内部翻成 [当前, t−1, …])
            "images": [self._images(f) for f in frames],
            "poses": torch.stack([self._pose(f) for f in frames]),
            "gt": self._gt(info),  # GT 取窗口**末帧**(当前帧)
        }

    def _images(self, info: dict) -> dict[str, torch.Tensor]:
        return {name: self._load(self.root / info["cams"][name]["data_path"]) for name in self.cam_names}

    @staticmethod
    def _pose(info: dict) -> torch.Tensor:
        return torch.tensor(info["ego2global"], dtype=torch.float32)

    @staticmethod
    def _gt(info: dict) -> list[list[np.ndarray]]:
        ann = info["annotation"]
        return [[np.asarray(line, dtype=np.float32) for line in ann[cls]] for cls in MAPTR_CLASSES]

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        t = to_tensor(img)  # (3, H, W) [0, 1]
        return normalize(t, IMAGENET_MEAN, IMAGENET_STD)


def collate(batch: list[dict]) -> dict:
    """批次组装:images 按相机堆叠,poses 堆叠,gts 保持列表(匹配在样本级做)。

    `window>1` 时 `images` 是**长度 K 的列表**(每项 = 各相机一堆),`poses` 是 (B, K, 6)
    —— 顺序都是旧 → 新,与 `MapTRDataset.__getitem__` 一致。
    """
    first = batch[0]["images"]
    if isinstance(first, list):
        names, k = list(first[0]), len(first)
        images = [{n: torch.stack([b["images"][j][n] for b in batch]) for n in names} for j in range(k)]
        poses = torch.stack([b["poses"] for b in batch])
    else:
        names = list(first)
        images = {n: torch.stack([b["images"][n] for b in batch]) for n in names}
        poses = torch.stack([b["pose"] for b in batch])
    return {"images": images, "poses": poses, "gts": [b["gt"] for b in batch]}

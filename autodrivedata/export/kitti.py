"""KITTI 布局落盘——照 auto3dlabel KittiFrame 路径约定(root 经 KITTI_OBJECT_ROOT 覆盖)。

布局(与 auto3dlabel/data/kitti.py + schema/box3d.py 一致):
  {root}/training/image_2/{id}.png   # 6 位零填充 id
  {root}/training/velodyne/{id}.bin  # float32 (N,4) x,y,z,intensity
  {root}/training/calib/{id}.txt
  {root}/training/label_2/{id}.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autodrivedata.calib import KittiCalibOut


def normalize_frame_id(frame_id: str) -> str:
    """'123' → '000123';已 6 位原样返回(照抄 auto3dlabel normalize_frame_id 语义)。"""
    fid = frame_id.strip()
    if not fid.isdigit():
        raise ValueError(f"非法帧 ID(须为数字): {frame_id!r}")
    return f"{int(fid):06d}"


@dataclass(frozen=True)
class FramePaths:
    """单帧四个文件的路径(镜像 KittiFrame 的属性)。"""

    image: Path
    velodyne: Path
    calib: Path
    label: Path


def frame_paths(root: str | Path, frame_id: str) -> FramePaths:
    """帧 id → 四路径(id 零填充)。"""
    base = Path(root) / "training"
    fid = normalize_frame_id(frame_id)
    return FramePaths(
        image=base / "image_2" / f"{fid}.png",
        velodyne=base / "velodyne" / f"{fid}.bin",
        calib=base / "calib" / f"{fid}.txt",
        label=base / "label_2" / f"{fid}.txt",
    )


def write_frame(
    root: str | Path,
    frame_id: str,
    *,
    image_png: bytes,
    velodyne: np.ndarray,
    calib: KittiCalibOut,
    labels: list[str],
) -> FramePaths:
    """单帧四件套落盘(velodyne 必须已是 KITTI velodyne 约定,见 geometry.carla_lidar_to_velodyne)。"""
    paths = frame_paths(root, frame_id)
    for p in (paths.image, paths.velodyne, paths.calib, paths.label):
        p.parent.mkdir(parents=True, exist_ok=True)
    paths.image.write_bytes(image_png)
    np.asarray(velodyne, dtype=np.float32).reshape(-1, 4).tofile(paths.velodyne)
    calib.write(paths.calib)
    paths.label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    return paths

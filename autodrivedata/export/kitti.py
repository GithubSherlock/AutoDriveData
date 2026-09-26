"""KITTI 布局落盘——照 auto3dlabel KittiFrame 路径约定(root 经 KITTI_OBJECT_ROOT 覆盖)。

布局(与 auto3dlabel/data/kitti.py + schema/box3d.py 一致):
  {root}/training/image_2/{id}.png   # 6 位零填充 id
  {root}/training/velodyne/{id}.bin  # float32 (N,4) x,y,z,intensity
  {root}/training/calib/{id}.txt
  {root}/training/label_2/{id}.txt
  {root}/training/pose/{id}.txt      # 可选:ego 真值位姿(KITTI 12 数 = 3×4 行主序)

**pose 目录是本仓扩展**(KITTI 官方在数据集根放 `poses/{seq}.txt`;这里逐帧一文件,
便于与四件套同帧号对齐)。存在的理由:SLAM 评估(ATE/RPE)必须有真值位姿,
而 `collect_drive.py` 走 autopilot 无真值 → 旧 `kitti_drive` 序列的 `closure.drift_m`
只能当"首末距离",冒充不了精度指标(见 Plan2.md P-H)。SLAM 序列采集器写此目录。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autodrivedata.calib.core import KittiCalibOut


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


def pose_path(root: str | Path, frame_id: str) -> Path:
    """帧 id → 真值位姿文件路径(`training/pose/{id}.txt`)。"""
    return Path(root) / "training" / "pose" / f"{normalize_frame_id(frame_id)}.txt"


def write_pose(root: str | Path, frame_id: str, T: np.ndarray) -> Path:
    """ego 真值位姿落盘:3×4 行主序 12 个数,空格分隔,一行。

    与 KITTI 官方 `poses/*.txt` 同格式(每行 12 数 = 3×4 相机→世界变换,行主序),
    故任何吃 KITTI pose 的工具可直接读。T 须为 (4,4) 或 (3,4)。
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape not in ((4, 4), (3, 4)):
        raise ValueError(f"位姿须为 (4,4) 或 (3,4),got {T.shape}")
    m = T[:3, :4]
    p = pose_path(root, frame_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(" ".join(f"{v:.9f}" for v in m.reshape(-1)) + "\n", encoding="utf-8")
    return p


def read_pose(path: str | Path) -> np.ndarray:
    """`write_pose` 的逆:12 数 → (4,4)。"""
    vals = np.fromstring(Path(path).read_text(encoding="utf-8"), sep=" ", dtype=np.float64)
    if vals.size != 12:
        raise ValueError(f"位姿文件须含 12 个数,got {vals.size}: {path}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = vals.reshape(3, 4)
    return T


def write_frame(
    root: str | Path,
    frame_id: str,
    *,
    image_png: bytes,
    velodyne: np.ndarray,
    calib: KittiCalibOut,
    labels: list[str],
    pose: np.ndarray | None = None,
) -> FramePaths:
    """单帧四件套落盘(velodyne 必须已是 KITTI velodyne 约定,见 geometry.carla_lidar_to_velodyne)。

    `pose` 非 None 时额外写 `training/pose/{id}.txt`(ego 真值位姿,SLAM 评估用)。
    """
    paths = frame_paths(root, frame_id)
    for p in (paths.image, paths.velodyne, paths.calib, paths.label):
        p.parent.mkdir(parents=True, exist_ok=True)
    paths.image.write_bytes(image_png)
    np.asarray(velodyne, dtype=np.float32).reshape(-1, 4).tofile(paths.velodyne)
    calib.write(paths.calib)
    paths.label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    if pose is not None:
        write_pose(root, frame_id, pose)
    return paths

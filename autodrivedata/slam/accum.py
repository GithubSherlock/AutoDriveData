"""累积语义点云建图(教程 11 升级)——多帧 KITTI velodyne 全局累积 + 语义着色。

教程原始做法:每帧点云投影到相机图取语义标签 → GPS 系累积(依赖相机语义分割 + GPS)。
本模块升级:直接用 LiDAR 强度列(语义编码,见 semantic.py)着色,**零依赖投影/分割**,
体素下采样去重 → 全局一致语义点云地图(高精地图雏形)。纯值,不 import carla。

口径:
- 输入:任意帧 velodyne bin (N,4) x,y,z,intensity(KITTI velodyne 约定,y 左)
- 输出:全局系累积点 (M,4)(xyz + 语义着色索引),体素下采样后写 ply
- 语义着色:强度列已按 albedo 编码(车 0.85 / 路面 0.08 / 标牌 0.9)——直接按强度带调色板

单测:tests/test_accum.py(手算锚点)。
"""

from __future__ import annotations

import numpy as np

# 体素边长(m)——下采样网格,去重 + 降密
VOXEL_SIZE = 0.2


def voxel_downsample(points: np.ndarray, voxel_size: float = VOXEL_SIZE) -> np.ndarray:
    """点云 (N,3+) → 体素重心下采样(每格子取重心点,向量化)。

    3D 体素键编码成单个 int64 → np.unique 分组 → np.add.at 求和 → 均值。
    返回降采样后 (M,3+)(前 3 列平均,其余列取首点)。
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return pts
    keys = np.floor(pts[:, :3] / voxel_size).astype(np.int64)
    # 每轴跨度(确保编码无碰撞):x/y ∈ [-400,400],z ∈ [-100,100] 内即可
    kx, ky, kz = keys[:, 0], keys[:, 1], keys[:, 2]
    flat = ((kx + 400) * 801 + (ky + 400)) * 201 + (kz + 100)
    uniq, inverse = np.unique(flat, return_inverse=True)
    m = uniq.shape[0]
    counts = np.bincount(inverse, minlength=m).astype(np.float64)
    sums = np.zeros((m, 3), dtype=np.float64)
    np.add.at(sums, inverse, pts[:, :3])
    out = np.empty((m, pts.shape[1]), dtype=np.float64)
    out[:, :3] = sums / counts[:, None]
    if pts.shape[1] > 3:
        # 其余列取每体素第一个点(np.unique 的 inverse 首现索引)
        first = np.full(m, -1, dtype=np.int64)
        np.minimum.at(first, inverse, np.arange(len(pts)))
        out[:, 3:] = pts[first, 3:]
    return out


def accumulate_global(frames: list[np.ndarray]) -> np.ndarray:
    """多帧 velodyne bin → 全局系累积点云(逐帧拼接)。

    输入帧已是 KITTI velodyne 约定(y 左,与 nuScenes 全局同向)→ 各帧直接拼接即可
    (仅 y 翻转已由采集侧完成)。返回 (N,4) x,y,z,intensity。
    """
    if not frames:
        return np.zeros((0, 4), dtype=np.float64)
    out = np.concatenate([np.asarray(f, dtype=np.float64)[:, :4] for f in frames], axis=0)
    return out


def semantic_color(intensity: float) -> tuple[int, int, int]:
    """强度带 → (R, G, B)。强度 = albedo·cos(见 semantic.py):车 0.85 / 标牌 0.9 /
    路面 0.08 / 行人 0.7。

    调色板(近似视觉语义):
    - 高反(≥0.55)= 车/行人/标牌 → 红
    - 中反(0.35~0.55)= 植被/护栏/低反车 → 橙
    - 低反(<0.35)= 路面/地面 → 灰
    """
    if intensity >= 0.55:
        return (200, 60, 60)
    if intensity >= 0.35:
        return (200, 160, 60)
    return (120, 120, 120)


def colorize(points: np.ndarray) -> np.ndarray:
    """累积点 (N,4) → 着色点 (N,6)(xyz + rgb uint8)。"""
    rgb = np.array([semantic_color(p[3]) for p in points], dtype=np.uint8)
    return np.hstack([points[:, :3], rgb])


def write_ply(path, points: np.ndarray) -> None:
    """着色点 (N,6)(xyz + rgb uint8)或 (N,3) → PLY 落盘。

    格式:PLY binary_little_endian(与多数查看器兼容)。"""
    pts = np.asarray(points)
    if pts.shape[1] == 3:
        pts = np.hstack([pts, np.full((pts.shape[0], 3), 128, dtype=np.uint8)])
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n".encode())
        for prop in ("x", "y", "z"):
            f.write(f"property float {prop}\n".encode())
        for prop in ("red", "green", "blue"):
            f.write(f"property uchar {prop}\n".encode())
        f.write(b"end_header\n")
        out = np.zeros((pts.shape[0], 6), dtype=np.float32)
        out[:, :3] = pts[:, :3]
        out[:, 3:] = pts[:, 3:]
        f.write(out.astype("<f4", copy=False).tobytes())

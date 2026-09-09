"""CARLA 语义 LiDAR 标签 → KITTI 式强度合成(M2-3 域差距修复)。

背景(M1a-7/M2-3 实测):CARLA ray_cast 强度 0.76~1.0(std 0.055)几乎恒定,
PointPillars 学到的"车=高反射+强度对比"特征失效 → 真车全漏、立面误报。
合成数据特权:ray_cast_semantic 每点带语义标签,按标签赋反照率 × 入射角
(Lambertian),复现真实 velodyne 的强度对比(车亮/路暗/标牌高反)。
"""

from __future__ import annotations

import numpy as np

# CARLA 语义标签(cityscapes 口径)→ 反照率(对标真实 velodyne 强度经验值)
CARLA_SEMANTIC_ALBEDO: dict[int, float] = {
    0: 0.3,  # Unlabeled
    1: 0.5,  # Building
    2: 0.4,  # Fence
    3: 0.3,  # Other
    4: 0.7,  # Pedestrian
    5: 0.8,  # Pole
    6: 0.45,  # RoadLine(路面标线反光)
    7: 0.08,  # Road
    8: 0.2,  # Sidewalk
    9: 0.4,  # Vegetation
    10: 0.85,  # Vehicles
    11: 0.5,  # Wall
    12: 0.9,  # TrafficSign(高反)
    13: 0.0,  # Sky
    14: 0.1,  # Ground
    15: 0.3,  # Bridge
    16: 0.3,  # RailTrack
    17: 0.7,  # GuardRail
    18: 0.9,  # TrafficLight(高反)
    19: 0.3,  # Static
    20: 0.6,  # Dynamic
    21: 0.1,  # Water
    22: 0.35,  # Terrain
}


def semantic_intensity(
    tags: np.ndarray, cos_angle: np.ndarray, noise_std: float = 0.05, seed: int | None = None
) -> np.ndarray:
    """语义标签 (N,) + 入射角余弦 (N,) → 强度 (N,) ∈ [0,1]。

    强度 = albedo[tag] · cos_angle + 高斯噪声(确定性 seed)。
    """
    rng = np.random.default_rng(seed)
    tags_i = np.asarray(tags, dtype=np.int64)
    cos = np.clip(np.asarray(cos_angle, dtype=np.float64), 0.0, 1.0)
    albedo = np.array([CARLA_SEMANTIC_ALBEDO.get(int(t), 0.3) for t in tags_i])
    out = albedo * cos + rng.normal(0.0, noise_std, size=len(tags_i))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def semantic_to_velodyne_bin(sem_points: np.ndarray, seed: int | None = None) -> np.ndarray:
    """语义 LiDAR 原始点 (N,6)(x,y,z,cos_angle,obj_idx,tag)→ KITTI velodyne bin (N,4)。

    y 翻转对齐 KITTI velodyne 约定(同 geometry.carla_lidar_to_velodyne);
    强度由语义标签合成(第 4 列 cos_angle 参与 Lambertian)。
    """
    pts = np.asarray(sem_points, dtype=np.float32).reshape(-1, 6)
    out = np.empty((len(pts), 4), dtype=np.float32)
    out[:, 0] = pts[:, 0]
    out[:, 1] = -pts[:, 1]  # y 左对齐
    out[:, 2] = pts[:, 2]
    out[:, 3] = semantic_intensity(pts[:, 5], pts[:, 3], seed=seed)
    return out

"""采集器纯值位姿/挂点计算(零 carla,唯一落点)。

把 `bin/` 采集器里可单测的纯几何抽到这里,bin 脚本只做 carla 编排:
- `ring_cam_pose`:3DGS 360° 环绕相机每帧位姿(collect_3dgs 用)
- `stereo_rig_offsets`:双目 rig 左右相机挂点相对 ego 的偏移(collect_stereo 用)

一致性锚点:
- 与 `carla_common.SENSOR_OFFSET = (1.2, 0.0, 1.65)` 一致(相机挂点 x 前 1.2m、z 高 1.65m)
- 环绕 yaw 口径:相机朝向 world 系 atan2(dy, dx)(从环绕中心指向相机),与
  collect_3dgs 原逻辑一致;度输出(CARLA Rotation 用度)
- 全部函数无 carla 依赖,AST 纪律由 tests/test_paths.py 守护
"""

from __future__ import annotations

import math


def ring_cam_pose(
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    i: int,
    n_cams: int,
    height: float = 1.5,
) -> tuple[float, float, float, float]:
    """360° 环绕第 i 个相机位姿,返回 (x, y, z, yaw_deg)。

    相机绕 (cx, cy, cz) 水平圆周运动,半径 radius,等角步进 360/n_cams;
    高度固定 center.z + height。相机朝向环绕中心(车头指向内)。
    yaw_deg 为 CARLA Rotation 用度(CARLA yaw 左转正),等于
    atan2(dy, dx)(世界系从相机到中心的方向角)。
    """
    if n_cams <= 0:
        raise ValueError(f"n_cams 必须 > 0,收到 {n_cams}")
    yaw_deg = 360.0 * i / n_cams
    rad = math.radians(yaw_deg)
    x = cx + radius * math.cos(rad)
    y = cy + radius * math.sin(rad)
    z = cz + height
    yaw_cam = math.degrees(math.atan2(cy - y, cx - x))
    return x, y, z, yaw_cam


def stereo_rig_offsets(baseline: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """双目基线 → 左右相机相对 ego 的挂点偏移 (x, y, z)[米]。

    与 collect_stereo 原逻辑一致:同一朝向(yaw=0),沿 y 轴 ±baseline/2;
    x/z 用 SENSOR_OFFSET(1.2 前 / 1.65 高)。返回 (left, right)。
    """
    if baseline <= 0:
        raise ValueError(f"baseline 必须 > 0,收到 {baseline}")
    half = baseline / 2.0
    return (1.2, -half, 1.65), (1.2, +half, 1.65)

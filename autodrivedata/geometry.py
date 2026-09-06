"""坐标转换唯一落点(AutoDriveData 侧)——CARLA 系 ↔ KITTI 相机系。

轴系红线(实测 pycarla 0.9.16 + 照抄 auto3dlabel/tools/geometry.py,禁止另立):
- CARLA 全局/传感器系:左手系,x 前 / y 右 / z 上
- CARLA Rotation 组合:**R = Rz(yaw)·Ry(pitch)·Rx(roll)**(矩阵实测锁定);
  正 pitch = 抬头、正 yaw = 左转(俯视逆时针);本模块角度一律**弧度**(CARLA API 是度,
  调用方用 np.radians 换算)
- KITTI 相机系:右手系,x 右 / y 下 / z 前;rotation_y 绕 y 轴(下)
- KITTI velodyne 系:右手系,x 前 / y 左 / z 上(与 CARLA 传感器系差一个 y 符号)
- 内部 yaw_bev(照 auto3dlabel):车头相对 +z 轴、向 +x 为正,车头=(sin yaw_bev, cos yaw_bev)
- 唯一转换点(照抄 auto3dlabel):ry = wrap_pi(yaw_bev − π/2);
  wrap 用 arctan2(sin, cos);数值上 wrap_pi(−π) = −π(±π 同义,与 auto3dlabel 行为一致)

单测:tests/test_geometry.py(手算锚点)+ tests/test_geometry_carla_oracle.py(pycarla 对照)。
"""

from __future__ import annotations

import numpy as np

# CARLA 系(x 前/y 右/z 上)→ KITTI 相机系(x 右/y 下/z 前)基变换。
# 含手性翻转(det = −1),正交:CARLA_TO_CAMᵀ = CARLA_TO_CAM⁻¹。
CARLA_TO_CAM = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64
)

# KITTI velodyne 系(x 前/y 左/z 上)→ KITTI 相机系。
# 与 auto3dlabel 的 GLOBAL_TO_CAM_LIKE 同构(nuScenes 全局系同为 y 左);
# 恒等位姿下 Tr_velo_to_cam 退化到本矩阵(手算锚点)。
VELO_TO_CAM = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64
)

# CARLA 传感器系 → KITTI velodyne 系:仅 y 基轴翻转(轴对齐基变换,非重投影)
CARLA_SENSOR_TO_VELO = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


def wrap_pi(angle: float) -> float:
    """归一化到 [−π, π](语义照抄 auto3dlabel;注意 −π 归一为 +π)。"""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def carla_rotation_matrix(rotation: tuple[float, float, float]) -> np.ndarray:
    """CARLA Rotation (pitch, yaw, roll)[弧度] → 3×3 旋转阵(列 = 局部系轴在世界系的分量)。

    组合顺序 Rz(yaw)·Ry(pitch)·Rx(roll),与 pycarla `Transform.get_matrix()` 旋转块
    逐元素一致(oracle 单测锁定)。
    """
    pitch, yaw, roll = rotation
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def camera_rotation_world_to_cam(cam_rotation: tuple[float, float, float]) -> np.ndarray:
    """相机 CARLA 位姿 → R_camK_world(3×3):世界系向量 → KITTI 相机系向量。

    R = CARLA_TO_CAM @ R_world_camCᵀ(camC = 相机的 CARLA 局部系),正交。
    """
    r_wc = carla_rotation_matrix(cam_rotation)
    return CARLA_TO_CAM @ r_wc.T


def world_to_cam(
    points: np.ndarray,
    cam_location: tuple[float, float, float],
    cam_rotation: tuple[float, float, float],
) -> np.ndarray:
    """世界系点 (N,3) → KITTI 相机系:p_k = R_camK_world @ (p_g − t_cam)。"""
    t = np.asarray(cam_location, dtype=np.float64)
    r = camera_rotation_world_to_cam(cam_rotation)
    return (np.asarray(points, dtype=np.float64)[:, :3] - t) @ r.T


def heading_to_rotation_y(
    heading_world: np.ndarray, cam_rotation: tuple[float, float, float]
) -> float:
    """世界系车头单位向量 → KITTI rotation_y(唯一转换点:ry = wrap_pi(yaw_bev − π/2))。"""
    h = np.asarray(heading_world, dtype=np.float64)[:3]
    hk = camera_rotation_world_to_cam(cam_rotation) @ h
    yaw_bev = float(np.arctan2(hk[0], hk[2]))  # 车头相对 +z、向 +x 为正
    return yaw_bev_to_rotation_y(yaw_bev)


def actor_yaw_to_rotation_y(
    actor_yaw: float, cam_rotation: tuple[float, float, float]
) -> float:
    """CARLA actor yaw[弧度] → 相机系 rotation_y(actor 俯仰/滚转恒 0 场景)。

    CARLA 车头 = Rz(yaw) 第一列 = (cos yaw, sin yaw, 0)(实测锁定)。
    """
    heading = np.array([np.cos(actor_yaw), np.sin(actor_yaw), 0.0])
    return heading_to_rotation_y(heading, cam_rotation)


def yaw_bev_to_rotation_y(yaw_bev: float) -> float:
    """内部 yaw_bev → KITTI rotation_y(照抄 auto3dlabel yaw_to_rotation_y)。"""
    return wrap_pi(yaw_bev - np.pi / 2)


def rotation_y_to_yaw_bev(rotation_y: float) -> float:
    """KITTI rotation_y → 内部 yaw_bev(照抄 auto3dlabel rotation_y_to_yaw)。"""
    return wrap_pi(rotation_y + np.pi / 2)


def corners_cam_from_bottom(
    x: float, y: float, z: float, h: float, w: float, l: float, ry: float
) -> np.ndarray:
    """KITTI 7 值(底心 x,y,z + 尺寸 + ry)→ 相机系 8 角点 (8,3)。

    照 KITTI 官方 computeBoxCorners:x' = x·cos ry + z'·sin ry、z' = −x'·sin ry + z'·cos ry
    (箱体系 +x' 前、+z' 右,ry=0 时车头 +x);相机 y 向下,顶面 y − h。
    角点序:底 4 [前右, 前左, 后左, 后右] → 顶 4 同序。
    """
    xc = np.array([l / 2, l / 2, -l / 2, -l / 2])  # 箱体系 +x'(前)
    zc = np.array([w / 2, -w / 2, -w / 2, w / 2])  # 箱体系 +z'(右)
    cos_r, sin_r = np.cos(ry), np.sin(ry)
    xs = xc * cos_r + zc * sin_r + x
    zs = -xc * sin_r + zc * cos_r + z
    bottom = np.stack([xs, np.full(4, y), zs], axis=1)
    top = np.stack([xs, np.full(4, y - h), zs], axis=1)
    return np.vstack([bottom, top])


def carla_lidar_to_velodyne(points: np.ndarray) -> np.ndarray:
    """CARLA LiDAR 原始点 (N,4)(x 前/y 右/z 上/intensity)→ KITTI velodyne 约定 (N,4)。

    仅 y 符号翻转(轴对齐基变换,无旋转/平移——不是重投影);落盘前调用,
    保证 bin 是标准 KITTI 约定(y 左),可直接混入真实 KITTI 数据训练。
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 4)
    out = pts.copy()
    out[:, 1] = -out[:, 1]
    return out

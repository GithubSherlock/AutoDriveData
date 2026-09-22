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

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover — 仅类型检查,避免几何层反向依赖 calib
    from autodrivedata.calib import CameraIntrinsics

# CARLA 系(x 前/y 右/z 上)→ KITTI 相机系(x 右/y 下/z 前)基变换。
# 含手性翻转(det = −1),正交:CARLA_TO_CAMᵀ = CARLA_TO_CAM⁻¹。
CARLA_TO_CAM = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64)

# KITTI velodyne 系(x 前/y 左/z 上)→ KITTI 相机系。
# 与 auto3dlabel 的 GLOBAL_TO_CAM_LIKE 同构(nuScenes 全局系同为 y 左);
# 恒等位姿下 Tr_velo_to_cam 退化到本矩阵(手算锚点)。
VELO_TO_CAM = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64)

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


def rotation_matrix_to_carla(R: np.ndarray) -> tuple[float, float, float]:
    """3×3 旋转阵 → CARLA Rotation (pitch, yaw, roll)[弧度] —— `carla_rotation_matrix` 的逆。

    逐元素对照 `carla_rotation_matrix` 的矩阵解出:
    `R[2,0] = sin(pitch)`、`R[2,1] = −cos(pitch)·sin(roll)`、`R[1,0]/R[0,0] = tan(yaw)`。
    **符号勿凭记忆**:写成 `pitch = asin(−R[2,0])` 会静默反号(第三方视角俯仰变仰视,
    实测 −12° → +12°);`tests/test_geometry.py` 有往返单测锁定(2.22e-16 级)。
    """
    r = np.asarray(R, dtype=np.float64)
    pitch = float(np.arcsin(np.clip(r[2, 0], -1.0, 1.0)))
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    roll = float(np.arctan2(-r[2, 1], r[2, 2]))
    return pitch, yaw, roll


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


def heading_to_rotation_y(heading_world: np.ndarray, cam_rotation: tuple[float, float, float]) -> float:
    """世界系车头单位向量 → KITTI rotation_y(唯一转换点:ry = wrap_pi(yaw_bev − π/2))。"""
    h = np.asarray(heading_world, dtype=np.float64)[:3]
    hk = camera_rotation_world_to_cam(cam_rotation) @ h
    yaw_bev = float(np.arctan2(hk[0], hk[2]))  # 车头相对 +z、向 +x 为正
    return yaw_bev_to_rotation_y(yaw_bev)


def actor_yaw_to_rotation_y(actor_yaw: float, cam_rotation: tuple[float, float, float]) -> float:
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


# ── nuScenes 约定(M1b;全局系语义照 auto3dlabel/照 nuScenes 官方)──────────────
#
# nuScenes 全局系:x 前 / y 左 / z 上(右手系)——与 CARLA 全局系(y 右)差一个 y 符号。
# 入表前经 CARLA_TO_NUS 翻转,入表后一切照 devkit/auto3dlabel 原生语义,零特判。

CARLA_TO_NUS = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


def carla_to_nus_global(points: np.ndarray) -> np.ndarray:
    """CARLA 全局系点 (N,3) → nuScenes 全局系(x 前/y 左/z 上):仅 y 符号翻转。"""
    pts = np.asarray(points, dtype=np.float64)[:, :3]
    return pts @ CARLA_TO_NUS.T


def carla_yaw_to_nus_yaw(yaw_carla: float) -> float:
    """CARLA yaw(绕 z,左转正)→ nuScenes 全局 yaw:wrap_pi(−yaw_c)。

    推导:CARLA 车头 (cos ψ, sin ψ) → nuScenes 车头 (cos ψ, −sin ψ)
    ⇒ ψ_nus = atan2(−sin ψ, cos ψ) = −ψ。
    """
    return wrap_pi(-yaw_carla)


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """nuScenes 全局系 yaw(绕 z 轴,x 前 y 左)→ 四元数 (w,x,y,z)。

    照抄 auto3dlabel tools/geometry.yaw_to_quat:车头 (cos yaw, sin yaw, 0)
    ⇒ quat = (cos(yaw/2), 0, 0, sin(yaw/2))。
    """
    half = yaw / 2
    return (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))


def quat_to_yaw(quat: tuple[float, float, float, float]) -> float:
    """四元数 (w,x,y,z) → 绕 z 轴 yaw(忽略 x/y 分量;照抄 auto3dlabel)。"""
    return wrap_pi(2 * float(np.arctan2(quat[3], quat[0])))


def carla_yaw_to_nus_quat(yaw_carla: float) -> tuple[float, float, float, float]:
    """CARLA actor yaw → nuScenes 全局四元数(组合 carla_yaw_to_nus_yaw + yaw_to_quat)。"""
    return yaw_to_quat(carla_yaw_to_nus_yaw(yaw_carla))


def quat_to_matrix(quat: tuple[float, float, float, float]) -> np.ndarray:
    """四元数 (w,x,y,z) → 3×3 旋转阵(Hamilton 约定,同 auto3dlabel rot_matrix)。

    **闭式解假定 |q| = 1**:官方 nuScenes 标定的四元数不是单位长度,取用前先过
    `quat_normalize`(见 `nus_camera_rotation_to_carla`)。
    """
    w, x, y, z = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_normalize(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """四元数 (w,x,y,z) 归一化到单位长度。

    **官方 nuScenes 标定存的四元数不是单位长度**(实测模长 0.99994~1.00005),
    而 `quat_to_matrix` 的闭式解假定 |q| = 1 ⇒ 不归一化会带进 ~1e-4 rad 的姿态误差。
    模长为 0 抛错(静默返回原值会让下游拿到非正交矩阵)。
    """
    q = np.asarray(quat, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        raise ValueError(f"四元数模长为 0,无法归一化:{quat}")
    u = q / n
    return (float(u[0]), float(u[1]), float(u[2]), float(u[3]))


def nus_camera_rotation_to_carla(quat_nus: tuple[float, float, float, float]) -> np.ndarray:
    """nuScenes 相机标定四元数 → 该相机在 **CARLA 全局系** 的旋转阵。

    R_carla = CARLA_TO_NUS @ R_nus @ CARLA_TO_CAM

    两段基变换的含义:
    - `R_nus`(由 `quat_to_matrix` 得)把 **nuScenes 相机局部系** 向量变到 nuScenes 全局系;
      nuScenes 相机局部系 = x 右 / y 下 / z 前(实测:NUS_CAMERA_CALIBS 的 CAM_FRONT 四元数
      第三列 = (1.000, 0.006, −0.006) 即朝前,第一列 ≈ −y_global 即朝右),
      与 KITTI 相机系同构 ⇒ CARLA 相机局部系(x 前/y 右/z 上)到它的基变换就是 `CARLA_TO_CAM`。
    - `CARLA_TO_NUS` 把 nuScenes 全局系(y 左)变回 CARLA 全局系(y 右);它是对合阵,转置即自身。

    由此导出的偏航恰好满足 `carla_yaw_to_nus_yaw`(yaw_c = −az_nus),两者互为校验。
    """
    r_nus = quat_to_matrix(quat_normalize(quat_nus))
    return CARLA_TO_NUS @ r_nus @ CARLA_TO_CAM


# ── 单目测距(P-D,教程 08)───────────────────────────────────────────────────
# 口径:相机系 z 向前(与 calib.world_to_img / KITTI 相机系一致);角度一律弧度。


def ground_intersection(
    world_cam: tuple[tuple[float, float, float], tuple[float, float, float]],
    intrinsics: CameraIntrinsics,
    u: float,
    v: float,
    ground_z: float,
) -> tuple[float, float] | None:
    """相机射线与地平面交点(纯值,零 carla 依赖)。

    world_cam = (loc, rot_rad)(mapviz.cam_pose 口径:位置米 / 姿态弧度)。
    像素 (u, v) → 归一化相机系方向 (x/z, y/z) → 世界系射线 → 与 z=ground_z
    平面求交,返回世界系 (x, y)。射线上行 / 相机后 / 与平面平行时返回 None。
    从 bin/sem_bev.py 上移,单一投影实现与采集/实时流共用。
    """
    loc, rot = world_cam
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    # 归一化平面坐标(相机系, z 向前的 pinhole)
    xn = (u - cx) / fx
    yn = (v - cy) / fy
    # 相机系方向 → 世界系(R_camK_world 的转置 = 相机→世界;参考 world_to_cam 逆)
    R_wc = camera_rotation_world_to_cam(rot)  # 世界→相机 旋转矩阵
    R_cw = R_wc.T  # 相机→世界
    dir_cam = np.array([xn, yn, 1.0])
    dir_world = R_cw @ dir_cam
    if dir_world[2] >= 0:  # 射线上行(看不到地面)
        return None
    t = (ground_z - loc[2]) / dir_world[2]
    if t <= 0:
        return None
    p = np.array(loc) + t * dir_world
    return float(p[0]), float(p[1])


def mono_depth_from_box(
    box_height_px: float,
    real_height_m: float,
    fy: float,
) -> float:
    """迭代深度法闭式解:已知真实尺寸 → 单目深度 z = real_height·fy / 框高。

    P-D(教程 08)单目测距;框高法(尺度歧义:单目无法同时知尺寸与深度,假设
    真实尺寸已知,如车高 H≈1.5m)。fy 为焦距像素(方形像素下即 fx)。
    """
    if box_height_px <= 0 or fy <= 0 or real_height_m <= 0:
        return float("inf")
    return float(real_height_m * fy / box_height_px)


def box_2d_from_3d(
    params3d: tuple[float, float, float, float, float, float, float],
    intrinsics: CameraIntrinsics,
) -> tuple[float, float, float, float] | None:
    """KITTI 3D 框(底心 x,y,z + h,w,l,ry)→ 相机图像 2D 框 (x1,y1,x2,y2)。

    与采集器 box_to_gt_line 同投影口径:corners_cam_from_bottom 8 角点 → p2
    投影 → 取**前端**(z>0)角点的 u/v min/max。全在相机后 → None(剔除)。
    P-D(教程 08)的诚实基线:已知 3D 框与位姿的投影,无 2D 模型误差。
    """
    x, y, z, h, w, l, ry = params3d
    corners = corners_cam_from_bottom(x, y, z, h, w, l, ry)
    img = (intrinsics.p2() @ np.hstack([corners, np.ones((8, 1))]).T).T
    zc = img[:, 2]
    front = zc > 0
    if not front.any():
        return None
    u = img[front, 0] / zc[front]
    v = img[front, 1] / zc[front]
    return (float(u.min()), float(v.min()), float(u.max()), float(v.max()))

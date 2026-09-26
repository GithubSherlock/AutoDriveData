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
    from autodrivedata.calib.core import CameraIntrinsics

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


# ---------------------------------------------------------------- 相机通道的朝轴重排
#
# ★ `CARLA_TO_NUS` 只管**全局**基(y 翻号);**相机自身的局部基**另有一个重排,两者不能混:
#   CARLA 相机局部轴 = (x 前, y 右, z 上) —— 与 LiDAR/雷达同一套(UE 系,左手)
#   nuScenes 相机局部轴 = (x 右, y 下, z 前)—— devkit 的 `cam2img` 按这套建 K
# 故「相机局部 → 全局」的完整算子不是 `M·R·M`(LiDAR/雷达那条),而是下面这个 C:
#   `A_cam = CARLA_TO_NUS @ R_carla @ CARLA_CAM_TO_NUS_CAM`
# 逐列读法:`A` 的三列 = 相机「右 / 下 / 前」三个方向在 nus 全局系里的分量,
# 而它们在 CARLA 侧分别是 R_carla 的第 1 列、第 2 列取负、第 0 列。
#
# **为什么必须显式写出来**:拿 `M·R·M` 去比相机的 3×3(或直接比四元数矩阵)会得到一个
# **恒为 120° 的"误差"**(轮换阵的本征角),看着像"相机装反了",其实只是两套局部基不同 ——
# §P-M.10 的判据⑨ 第一次跑就踩了这个(六路相机齐刷刷 119.93–120.07°)。
# LiDAR / 雷达**不走这条**(它们的局部基与 CARLA 同 ⇒ 仍用 `M·R·M`)。
CARLA_CAM_TO_NUS_CAM = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)


def carla_yaw_to_nus_yaw(yaw_carla: float) -> float:
    """CARLA yaw(绕 z,左转正)→ nuScenes 全局 yaw:wrap_pi(−yaw_c)。

    推导:CARLA 车头 (cos ψ, sin ψ) → nuScenes 车头 (cos ψ, −sin ψ)
    ⇒ ψ_nus = atan2(−sin ψ, cos ψ) = −ψ。
    """
    return wrap_pi(-yaw_carla)


# ---------------------------------------------------------------- nuScenes ego 系原点
#
# ★★ **nuScenes 的 `ego_pose` / `calibrated_sensor` 原点 = 后轴中心在地面的投影**;
#    **CARLA 车辆 actor 的原点 = 车身长度中点在地面的投影**。两者差一个纵向平移。
#
# 本项目的整张官方标定表(CAM_*、RADAR_*、LIDAR_TOP)都是**按 nuScenes ego 系写的**,
# 历史实现却把那些 x 原样当成了"距 CARLA actor 原点的距离" ⇒ 全套传感器整体前移
# `|NUS_EGO_ORIGIN_X|` 米(实测 CAM_BACK 因此有 42.999% 的自身车体像素;`RADAR_FRONT`
# 悬在车头前方 1.56 m)。**这不是"抄错数",是两套系的原点不同**——所以修法不是改数,
# 而是把"声明(nus 系)"与"渲染(CARLA 系)"之间的这一条换算**显式写出来**。
#
# 常量值 = `vehicle.audi.a2` 后轴在 CARLA actor 系里的 x(米),**实测**:
#   - 双偏航自解(与轴系约定无关,见 `autodrivedata/calib/verify_nus_calib.py` 判据⑩):
#     `W(ψ) = C + R(ψ)·w₀` 取 ψ 与 ψ+90° 两档解出 4 个轮子的 w₀ 与 C
#     ⇒ 前后轴 x = **+1.2502 / −1.2563**,轴距 2.5065(四轮解出的 C 互差 2e-06 m)
#   - 自洽校验:轴距 2.5065 + 前悬(=前轴到前保险杠 1.8527−1.2502=0.6025)
#     ⇒ 后轴到前保险杠 3.1090 m;同法官方 `RADAR_FRONT x=+3.4120` 落在这**之外** 0.30 m
#     —— 因为 a2(3.705 m)比 nuScenes 的雷诺 Zoe(4.084 m)短 0.38 m,不存在让
#     前后雷达同时落进车身内的刚性映射(如实报出,不靠单传感器微调掩盖)
#   - 佐证:**整车**标定表在同一 Δx 下同时自洽 —— `RADAR_BACK x=−0.562` 落到车尾内侧、
#     `LIDAR_TOP x=+0.9437` 落到车顶行李架、`CAM_BACK` 落到后轴正上方(实测车体像素 0)
#
# ⚠️ **换 ego 蓝图(不是 audi.a2)必须重测本常量**:判据⑩ 会在实机上复测并比对,
# 不一致直接判否(常量腐化 = 静默回到原点错位,那正是本次要根除的失效模式)。
NUS_EGO_ORIGIN_X: float = -1.2563


def nus_mount_to_carla(t_nus: tuple[float, float, float]) -> tuple[float, float, float]:
    """**声明→渲染**的唯一换算:nuScenes 系挂点(原点=后轴)→ CARLA actor 系挂点。

    两步,别再手抄第二份(§P-M.7 的"表对了、图错了"就是手抄出来的):
    ① `x += NUS_EGO_ORIGIN_X`(原点从后轴挪到车身中点);
    ② `y → −y`(nus 系 y 左 → CARLA 系 y 右)。

    ⚠️ z **不动** —— 两个系的原点都在**地面**(实测 a2 包围盒底 z=0.0073、顶 1.5563,
    与实车 A2 高 1.553 m 吻合;官方 nus 的 z 同样是离地高度)。
    """
    return (t_nus[0] + NUS_EGO_ORIGIN_X, -t_nus[1], t_nus[2])


def carla_actor_origin_to_nus_ego(
    actor_location: tuple[float, float, float],
    actor_rotation_rad: tuple[float, float, float],
    origin_x: float = NUS_EGO_ORIGIN_X,
) -> tuple[float, float, float]:
    """CARLA 车辆 actor 位姿 → **nuScenes ego 原点(后轴中心)** 的 CARLA 世界坐标(米)。

    后轴点在 actor 系里是 `(origin_x, 0, 0)`,故世界坐标 = `t_actor + R·(origin_x,0,0)`。
    旋转走 `carla_rotation_matrix`(组合顺序由 oracle 单测锁定),**不自己推车头矢量**
    (正负号写反在 yaw≈0 时看不出来,是本项目踩过的"只有 CAM_FRONT 看着对"同款坑)。

    ⚠️ **收全 6DoF `(pitch, yaw, roll)`[弧度],不收单个 yaw**:`vehicle.audi.a2` 静止后
    悬架沉下来有**实测 +0.0642° 的俯仰**(前后轮自重不均),只传 yaw 会把它丢掉 ——
    `1.2563 m` 的杆臂上就是 1.4e-3 m 的站位误差,整条 `ego_pose ⊕ calibrated_sensor`
    链差 0.064°,而"实挂 vs 声明"逐传感器判据**看不出来**(两边同错)。判据⑨ 之前正是
    卡在这个量级上。**签名只收全量**是故意的:留一个 yaw-only 入口,下一个人就会用它。

    `origin_x` 可注入:判据⑨ 用它把"实机**测出来的**后轴"代进去,从而**不依赖**
    `NUS_EGO_ORIGIN_X` 这个常量本身的取值(常量对不对由判据⑩单独把守)。
    """
    r = carla_rotation_matrix(actor_rotation_rad)
    off = r @ np.array([float(origin_x), 0.0, 0.0], dtype=np.float64)
    return (
        float(actor_location[0] + off[0]),
        float(actor_location[1] + off[1]),
        float(actor_location[2] + off[2]),
    )


def nus_ego_translation(
    actor_location: tuple[float, float, float],
    actor_rotation_rad: tuple[float, float, float],
    origin_x: float = NUS_EGO_ORIGIN_X,
) -> tuple[float, float, float]:
    """CARLA actor 位姿 → nuScenes 全局系的 `ego_pose.translation`(即后轴点的 nus 坐标)。

    **入表前唯一的 ego 位姿来源**:写 `carla_to_nus_global(actor_location)` 会让整条
    传感器链相对 ego_pose 偏 `|NUS_EGO_ORIGIN_X|` 米(见 `NUS_EGO_ORIGIN_X` 头注)。
    """
    p = carla_actor_origin_to_nus_ego(actor_location, actor_rotation_rad, origin_x)
    q = carla_to_nus_global(np.asarray([p], dtype=np.float64))[0]
    return (float(q[0]), float(q[1]), float(q[2]))


def nus_ego_rotation(actor_rotation_rad: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """CARLA actor 姿态 `(pitch, yaw, roll)`[弧度] → nuScenes 全局四元数 `ego_pose.rotation`。

    **为什么不是 `carla_yaw_to_nus_quat(yaw)`**:那只保留 yaw,把整车姿态拍平成纯偏航,
    而 nuScenes 的 `ego_pose` 是 devkit 直接左乘用的**全 6DoF** —— 丢掉 pitch/roll 会让
    `ego_pose ⊕ calibrated_sensor` 与世界系真值差一个 0.064° 量级的旋转(实测值,见
    `carla_actor_origin_to_nus_ego` 的告警)。

    推导(CARLA 车体系 x 前/y 右/z 上;nuScenes ego 系 x 前/y 左/z 上 ⇒ 局部基只差 y 翻号):
    `R_nus = M·R_carla·M`(`M = CARLA_TO_NUS`)。`M² = I` 故共轭可逐因子分配:
    `M·Rz(w)·M = Rz(−w)`、`M·Ry(p)·M = Ry(p)`、`M·Rx(a)·M = Rx(−a)` —— 但**只能代入右手矩阵**。

    **★ 陷阱:`carla_rotation_matrix` 吐的是 UE 左手口径**,它的"pitch/roll"在右手语言里是
    `Ry(−pitch)` / `Rx(−roll)`(纯 pitch 时 `R[2,0] = +sin pitch`、纯 roll 时 `R[2,1] = −sin roll`,
    而右手 `Ry(p)[0,0..]` 是 `R[2,0] = −sin p`)。所以把 `R_carla = Rz(yaw)·Ry(pitch)·Rx(roll)`
    里的 pitch/roll **原样**代进上面三条会**静默反号** —— 2026-09-23 就是这么错的:
    逗号隔开的四项里 yaw 项对得上(故"看着差不多"),而 (0,2)/(2,0)/(1,2)/(2,1) 四个元素
    与矩阵共轭差 2·sin(0.0642°) 量级。正确结果:
    `R_nus = Rz(−yaw) · Ry(−pitch) · Rx(+roll)`(右手口径)。
    用 `quat_mul` 把三个轴角乘起来,**不新写一套欧拉展开**;单测按**矩阵相等**锁死
    (`test_ego_rotation_is_the_matrix_conjugate_not_yaw_only`,比四元数向量会被 ±q 骗过)。
    """
    pitch, yaw, roll = (float(v) for v in actor_rotation_rad)

    def axis_quat(axis: int, a: float) -> tuple[float, float, float, float]:
        sh = float(np.sin(a / 2))
        return (
            float(np.cos(a / 2)),
            sh if axis == 0 else 0.0,
            sh if axis == 1 else 0.0,
            sh if axis == 2 else 0.0,
        )

    return quat_normalize(quat_mul(axis_quat(2, -yaw), quat_mul(axis_quat(1, -pitch), axis_quat(0, roll))))


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

    **闭式解假定 |q| = 1**:本仓表里的四元数是**手抄官方值时舍入到 4 位小数**得到的
    (如 `CAM_FRONT_LEFT` 的 |q|−1 = −5.04e-05),取用前先过 `quat_normalize`
    (见 `nus_camera_rotation_to_carla` / `nus_sensor_rotation_to_carla`)。
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


def quat_mul(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Hamilton 积 `a ⊗ b`(w,x,y,z)——满足 `quat_to_matrix(a⊗b) == quat_to_matrix(a) @ quat_to_matrix(b)`。

    **左乘 = 在固定系里转**:`quat_mul(yaw_to_quat(θ), q)` 等价于把 `q` 的姿态在全局系里
    绕 z 轴再转 θ(局部系的 pitch/roll 不变)。`camera_rig.nus_wide_camera_calibs` 用它做
    "只搬方位角、不动俯仰/横滚"的相机重摆 —— 手写 Rodrigues 会重复这套代数且更易错。
    未归一化(A、B 各自单位则积亦单位),需要时显式过 `quat_normalize`。
    """
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_normalize(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """四元数 (w,x,y,z) 归一化到单位长度。

    **非单位的来源是"手抄舍入",不是官方值**:官方 mini 集 120 条
    `calibrated_sensor` 的 |q| 实测全为 1.000000000000(最大偏离 2.22e-16,IEEE754 正常
    舍入);本仓表把官方值抄成 4 位小数(如 `CAM_FRONT_LEFT` 的 |q|−1 = −5.04e-05),
    而 `quat_to_matrix` 的闭式解假定 |q| = 1 ⇒ 不归一化会带进 ~1e-5 rad 的姿态误差。
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


def nus_sensor_rotation_to_carla(quat_nus: tuple[float, float, float, float]) -> np.ndarray:
    """nuScenes **LiDAR / 雷达** 标定四元数 → 该传感器在 **CARLA 全局系** 的旋转阵。

    R_carla = CARLA_TO_NUS @ R_nus @ CARLA_TO_NUS

    与 `nus_camera_rotation_to_carla` 成对,差别只在**传感器自身系**:
    - 相机自身系 = x 右 / y 下 / z 前(KITTI 同构)⇒ 右边那个基变换是 `CARLA_TO_CAM`;
    - LiDAR / 雷达自身系 = **x 前 / y 左 / z 上**(与 nuScenes 全局系同轴序)⇒ 右边那个
      基变换就是 `CARLA_TO_NUS` 自身。

    `CARLA_TO_NUS = diag(1,−1,1)` 是对合阵(自身即逆),故两侧同阵:整个式子等价于
    把旋转阵的 y 行与 y 列同时翻号 —— 对纯 yaw 即 `yaw_carla = −az_nus`(与相机同一条规则,
    也即 `carla_yaw_to_nus_yaw` 的逆),对 6DoF 则连 pitch/roll 一起正确翻过去。
    LiDAR 的 up 轴倾角 1.4289° 只能靠这条链表达(yaw-only 表达不了)。
    """
    r_nus = quat_to_matrix(quat_normalize(quat_nus))
    return CARLA_TO_NUS @ r_nus @ CARLA_TO_NUS


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

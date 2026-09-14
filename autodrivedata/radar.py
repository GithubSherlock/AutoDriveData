"""CARLA radar 原始检测 → nuScenes 18 字段雷达点云(纯值,不 import carla)。

nuScenes 官方雷达点云是 **18 字段 .pcd** 二进制(不是 6 维 .bin),由
devkit 的 ``RadarPointCloud.from_file`` 消费(devkit 1.2.0 本机实测)。字段表:

    FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid
           ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
    SIZE   4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
    TYPE   F F F I I F F F F F I I I I I I I I

每点 43 字节(struct ``<fffBhfffffBBBBBBBB``)。坐标系:x 前 / y 左 / z 上,
雷达传感器系(与 nuScenes 全系约定一致,非 CARLA 系)。

devkit 默认过滤器(``RadarPointCloud.from_file`` 静默丢点):``invalid_state``
(idx14) 必须 == 0、``dyn_prop`` (idx3) ∈ 0..6、``ambig_state`` (idx11) == 3。
不满足的点即使写盘也会被读回时清掉,所以本模块同时提供 ``valid_mask_nus``
供采集侧提前筛。

空点云编码:单点全 NaN(devkit from_file 读到首点 NaN 即返回空 (18,0),
绕开 ``assert width > 0``)。
"""

from __future__ import annotations

import struct

import numpy as np

# 每点二进制布局:<fff B H fffff BBBBBBBB(3×f32 + u8 + u16 + 5×f32 + 8×u8 = 43B)
RADAR_STRUCT = struct.Struct("<fffBhfffffBBBBBBBB")
# struct 的整数字段(与 RADAR_STRUCT 对齐):dyn_prop(B)、id(h)、
# is_quality_valid..vy_rms(B×8)。写盘前转 int;空点云 NaN 保留(devkit 空编码)。
_INT_FIELDS = (3, 4, 10, 11, 12, 13, 14, 15, 16, 17)
RADAR_NUS_FIELDS = (
    "x y z dyn_prop id rcs vx vy vx_comp vy_comp "
    "is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms"
).split()

# devkit 默认过滤器要求(照 RadarPointCloud 类级默认;见文件头)
INVALID_STATE_VALID = 0  # 仅 valid 保留
DYNPROP_VALID = 0  # 0..6 都保留;0 = moving
AMBIG_VALID = 3  # 仅 unambiguous 保留
IS_QUALITY_VALID = 1  # 质量有效位

# 官方大陆 ars408 垂直 FOV 14.2° → 半角 7.1°。CARLA 0.9.16 把垂直射线布到 ±38°
# (horizontal_fov 属性交叉控制垂直,且不收敛到 14.2°),写 pcd 前必须按真实 ars408
# 垂直锥裁剪,否则 pcd 混入 5% 的锥外点(天空/地面回波,ars408 物理上不存在)。
# 判据用原始 CARLA alt(球坐标直接对应锥角),不反体素化(反算 z 在远距会放宽阈值,
# depth=250m 时 z 界 ±31m,几乎全判锥内——错误)。
ARS408_VFOV_HALF_DEG = 7.1
ARS408_VFOV_HALF_RAD = np.radians(ARS408_VFOV_HALF_DEG)


def _to_radar_alt_z(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """18 字段点 → (depth, z)。18 字段里没有独立 alt 列,用 z/depth 反推锥角:
    |z|/depth = |sin(alt)|。空点云(全 NaN)在此自然得到 False 掩码。
    """
    d = np.hypot(a[:, 0], a[:, 1])
    depth = np.hypot(d, a[:, 2])
    return a[:, 2], depth


def mask_in_ars408_vfov_impl(arr18: np.ndarray) -> np.ndarray:
    """真实 ars408 垂直锥(±7.1°)内布尔掩码的实现(供 mask_in_ars408_vfov 组合调用)。

    CARLA 0.9.16 把垂直射线布到 ±38°(远大于 ars408 的 ±7.1°),写 pcd 前裁剪,
    否则 pcd 混入锥外点(天空/地面回波)。判据用球坐标 alt——注意 **18 字段里没有
    独立 alt 列**,alt 可从 x/depth 反推:alt = arcsin(z/d)。等价的、更稳的判据:
    |z| <= tan(7.1°)·depth(CARLA z 有符号:仰为正/俯为负,对称锥)。

    不反体素化(depth 很大时 tan(7.1°)·depth 会被放宽——250m 处 ±31m,几乎全放行)。
    正确判据必须是**锥角**(alt 或 z/depth 比值),不是绝对 z。
    """
    a = np.asarray(arr18)
    z, depth = _to_radar_alt_z(a)
    sin_lim = np.sin(ARS408_VFOV_HALF_RAD)
    return np.abs(z) <= depth * sin_lim + 1e-6


def detections_to_nus18(
    pts_carla: np.ndarray,
    sensor_id: int = 0,
) -> np.ndarray:
    """CARLA radar 原始检测 → nuScenes 18 字段点 (N,18) float32。

    :param pts_carla: (N,4) float32,每点 ``[vel, altitude, azimuth, depth]``
        (CARLA ``RadarMeasurement.raw_data`` 官方布局,manual_control.py 注释)。
        altitude/azimuth 弧度、depth 米、vel 径向速度(朝传感器为正)。
    :param sensor_id: 写入 18 字段的 ``id``(devkit 只要求唯一;采集侧传通道索引)。

    位置换算(球坐标 → nus 系笛卡尔,推导见 Plan.md §radar):
    d=depth, a=azimuth(CARLA 右转为正), e=altitude →
        x = d·cos e·cos a,  y = −d·cos e·sin a,  z = d·sin e
    速度(径向 → 水平 vx/vy,nus 系;无 vz):
        vx = −vr·cos e·cos a,  vy = +vr·cos e·sin a
    其余 13 字段填 devkit 过滤器合法固定值:dyn_prop=0 / is_quality_valid=1 /
    ambig_state=3 / invalid_state=0 / 其余 0。vx_comp/vy_comp 首版 = 原值
    (无 ego 运动补偿;nuScenes 官方用补偿值,后续要"绝对目标速度"再叠加 ego)。
    """
    arr = np.asarray(pts_carla, dtype=np.float32).reshape(-1, 4)
    n = len(arr)
    vel, alt, azi, depth = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    out = np.zeros((n, 18), dtype=np.float32)
    cos_e = np.cos(alt)
    out[:, 0] = depth * cos_e * np.cos(azi)  # x
    out[:, 1] = -depth * cos_e * np.sin(azi)  # y(nus 左)
    out[:, 2] = depth * np.sin(alt)  # z
    out[:, 3] = DYNPROP_VALID  # dyn_prop
    out[:, 4] = sensor_id  # id
    out[:, 5] = 0.0  # rcs(CARLA 无)
    out[:, 6] = -vel * cos_e * np.cos(azi)  # vx
    out[:, 7] = vel * cos_e * np.sin(azi)  # vy
    out[:, 8] = out[:, 6]  # vx_comp
    out[:, 9] = out[:, 7]  # vy_comp
    out[:, 10] = IS_QUALITY_VALID  # is_quality_valid
    out[:, 11] = AMBIG_VALID  # ambig_state
    out[:, 12] = 0.0  # x_rms
    out[:, 13] = 0.0  # y_rms
    out[:, 14] = INVALID_STATE_VALID  # invalid_state
    out[:, 15] = 0.0  # pdh0
    out[:, 16] = 0.0  # vx_rms
    out[:, 17] = 0.0  # vy_rms
    return out


def valid_mask_nus(arr18: np.ndarray) -> np.ndarray:
    """模拟 devkit 默认过滤器的合法布尔掩码(供采集侧提前筛点;纯值可单测)。

    判据照 ``RadarPointCloud.from_file``:invalid_state(idx14)==0、
    dyn_prop(idx3) ∈ 0..6、ambig_state(idx11)==3。
    """
    a = np.asarray(arr18)
    dyn = a[:, 3].astype(np.int64)
    return (
        (a[:, 14].astype(np.int64) == INVALID_STATE_VALID)
        & (dyn <= 6)
        & (a[:, 11].astype(np.int64) == AMBIG_VALID)
    )


def mask_in_ars408_vfov(arr18: np.ndarray) -> np.ndarray:
    """真实 ars408 垂直锥(±7.1°)内布尔掩码(纯值可单测,供独立调用/单测)。

    **别名**到 mask_in_ars408_vfov_impl(实现);采集组合用 mask_radar_points。
    判据:锥角 |sin(alt)| <= sin(7.1°),即 |z| <= tan(7.1°)·depth。不反体素化。
    """
    return mask_in_ars408_vfov_impl(arr18)


def mask_radar_points(arr18: np.ndarray) -> np.ndarray:
    """采集侧雷达点过滤组合掩码:devkit 默认过滤器(valid_mask_nus) ∩ 垂直锥(mask_in_ars408_vfov_impl)。

    写 pcd 前**必须**两者都过:devkit 读回时静默丢 invalid/dyn_prop/ambig 不合法点;
    垂直锥外点是 CARLA 把射线布到 ±38° 造成的(ars408 物理 ±7.1°),裁剪后才像真实雷达。
    空点云(全 NaN)两掩码都 False → 全滤掉 → 空 pcd 正确。
    """
    a = np.asarray(arr18)
    return valid_mask_nus(a) & mask_in_ars408_vfov_impl(a)


# struct 的整数字段索引(在 (N,18) 中的列):dyn_prop(3)、id(4)、
# is_quality_valid..vy_rms(10..17)——打包前必须转 int(struct 不认 float)。
_INT_FIELDS = (3, 4, 10, 11, 12, 13, 14, 15, 16, 17)


def nus18_to_pcd(points: np.ndarray) -> bytes:
    """(N,18) float32 → 标准 .pcd 二进制(头 + struct 逐点打包),devkit 直读。

    0 点 → 写 WIDTH 1 + 单行全 NaN(devkit 空点云编码:from_file 读到首点
    NaN 返回空 (18,0),绕开 ``assert width > 0``)。
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 18)
    n = len(pts)
    if n == 0:
        pts = np.full((1, 18), np.nan, dtype=np.float32)
        n = 1
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(RADAR_NUS_FIELDS)}\n"
        "SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1\n"
        "TYPE F F F I I F F F F F I I I I I I I I\n"
        "COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    ).encode("ascii")
    payload = bytearray()
    for row in pts:
        packed = [float(v) for v in row]
        for i in _INT_FIELDS:
            v = packed[i]
            packed[i] = 0 if v != v else int(round(v))  # NaN(空点云编码)原样保留
        payload += RADAR_STRUCT.pack(*packed)
    # 官方头以 "DATA binary" 换行结尾,devkit 逐行读、遇 DATA 行即 break
    return header + bytes(payload) + b"\n"

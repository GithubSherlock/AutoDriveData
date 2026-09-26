#!/usr/bin/env python3
"""`collect_nus.py` 全传感器标定的**十条验收判据复现器**(全数值,不目检)。

背景见 Plan2.md §P-M.7:`collect_nus.py` 曾是「**表对了、图错了**」的失效模式 ——
`calibrated_sensor` 写官方正确值,传感器却 spawn 在另一套位姿上。这类缺陷**只查表全绿**、
**只目检"图能出"也全绿**,故必须有独立锚的数值判据。

| 模式 | 判据 | 阈值 |
|---|---|---|
| `--live` | ① 相机实挂 vs 声明(`NUS[_WIDE]_CAMERA_RIG`):平移 / 偏航 | < 1e-3 m / < 1e-3° |
| `--live` | ② 雷达实挂 vs 官方 az | 逐通道 < 0.1° |
| `--offline` | ③ 雷达点落进**自身 FOV** 的占比(以官方 R 为锚) | 五通道全 ≥ 0.80 |
| `--offline` | ④ LiDAR 复现 `num_lidar_pts`(官方集上做锚 + 本仓自洽) | 比值 = 1.0000 |
| `--offline` | ⑤ 相机内参 vs 该 rig 的声明表 | 逐通道 fx/cx/cy < 0.01 px |
| `--live` | ⑥ 渲染 FOV vs 蓝图 fov(轴目标物掩膜质心回归 fx) | 逐通道 < 0.1° |
| `--live` | ⑦ 画幅内自身车体像素 | 逐通道 = 0 px |
| `--live` | ⑧ 相邻相机的共同可见方位 | 每对都被两路看见 |
| `--live` | ⑨ **世界位姿链** `ego_pose ⊕ calibrated_sensor` vs CARLA 实挂 | 12 传感器 < 1e-3 m / 1e-3° |
| `--live` | ⑩ **ego 原点复测**(audi.a2 后轴 x vs `NUS_EGO_ORIGIN_X`) | < 1e-3 m |

**为什么 ①② 全绿还不够(本轮新增 ⑨⑩ 的理由)**:①② 比的是"实挂挂点 vs 声明表",
而**两边都以同一个 ego 为参照** ⇒ ego 原点整体错位时它**恒绿**。2026-09-23 实测正是如此:
① 全绿而 `CAM_BACK` 画幅里 42.999% 是自身车体 —— 因为整套标定表(nus 系,原点=后轴)被
原样当成了 CARLA actor 系(原点=车身中点)的距离,整组传感器前移 1.2563 m。⑨ 把链子对到
**世界系**、⑩ 在实机上**重测**该原点常量,两条合起来才闭环(⑨ 用实测原点 ⇒ 不依赖常量取值,
⑩ 单独把守常量 —— 换 ego 蓝图而忘改常量时会红)。

**判据 ③ 为什么以官方 R 为锚**:点云存的是**传感器自身系**,其 +x 就是光轴 —— 若只问
"点在自己系里落不落进 ±38.1° 的锥",那是恒真的(与标定无关)。故必须把点用**落盘的
`calibrated_sensor.rotation`** 转到 ego 系,再与**官方标定的光轴**比 ⇒ 测的才是
"落盘的 R 与官方 R 是否同一个"。修前(声明 R = 单位阵)四路角雷达掉到 0.00–0.04。

**明令禁止的判据**:`num_radar_pts`(跨 5 通道求和,即使用官方 R + 合并 + 官方框也只有
0.6279)。LiDAR 的 1.0000 是特例,不外推。

## `--rig {nuscenes,wide}`(2026-09-23 增)

判据 ③④(雷达/LiDAR)**与相机无关,原样适用**;①②⑤⑥ 换用该 rig 的声明表;新增两条:

| 模式 | 判据 | 阈值 |
|---|---|---|
| `--live` | ⑦ **画幅内自身车体像素**(instance_seg 里数 ego 的 actor id) | 逐通道 = 0 px |
| `--live` | ⑧ **相邻相机的共同可见方位**(重叠带正中摆锥 → 两路掩膜都命中) | 每对都被两路看见 |

⑦⑧ 的实现落在 `bin/rig_check.py`(同一套相机跑两条判据),那里还记了两条判据的边界:
窄重叠区不摆锥、以及"方位轴重叠"与"有限距离下共同可见"因**挂点视差**而不等(实测差 1.7°)。

用法:
  python bin/verify_nus_calib.py --offline                       # ③④⑤(不需 CARLA)
  bash tools/carla_server.sh start
  python bin/verify_nus_calib.py --live                          # ①②⑥⑦⑧(需 CARLA)
  python bin/verify_nus_calib.py --rig wide --offline --live --dataroot outputs/nus_mini_wide
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

BIN = Path(__file__).resolve().parent
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

from autodrivedata import calib_probe as cp  # noqa: E402
from autodrivedata import geometry as g  # noqa: E402
from autodrivedata.camera_rig import NUS_CAMERA_RIG, NUS_WIDE_CAMERA_RIG  # noqa: E402
from autodrivedata.export.nuscenes import (  # noqa: E402
    NUS_CAMERAS,
    NUS_LIDAR_CALIB,
    NUS_RADAR_CHANNELS,
    NUS_RADAR_OFFSETS,
    NUS_RIGS,
    camera_calibs,
    camera_fov,
    camera_k,
)
from autodrivedata.geometry import quat_normalize, quat_to_matrix  # noqa: E402
from autodrivedata.paths import project_path  # noqa: E402

# 本次运行的 rig(由 `main` 从 `--rig` 写入)。模块级常量而非层层传参:①②⑤⑥ 的
# 声明表选择散在四个函数里,穿参会让"某处忘了换表"变成静默口径分叉(§P-M.7 的教训)。
RIG: str = "nuscenes"


def rig_cameras() -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """当前 rig 的 CARLA 侧逐相机 (挂点, 姿态)(pitch/yaw/roll 度)。"""
    return NUS_WIDE_CAMERA_RIG if RIG == "wide" else NUS_CAMERA_RIG


def rig_intrinsics() -> dict[str, tuple[float, float, float]]:
    """当前 rig 的逐通道 `(fx, cx, cy)`(判据 ⑤ 的比较基准)。"""
    return {cam: camera_k(cam, RIG) for cam in rig_cameras()}


OFFICIAL_MINI = Path("/root/autodl-tmp/Documents/datasets/nuscenes_mini")
VERSION = "v1.0-mini"

# ars408 实测锥(采集侧属性扫描自洽,见 `collect_nus.RADAR_ATTRS`)
ARS408_AZ_HALF_DEG = 38.1
ARS408_EL_HALF_DEG = 7.0
ARS408_RANGE_M = 250.0

# 判据阈值(与 Plan2 §P-M.7.8 的验收表一致,不在这里放宽)
TOL_MOUNT_M = 1e-3
TOL_MOUNT_DEG = 1e-3
TOL_RADAR_DEG = 0.1
MIN_FOV_FRAC = 0.80
TOL_K_PX = 0.01
TOL_FOV_DEG = 0.1

# ⑥ 用的锥体摆放:沿光轴 Z 处、按相机自身"右"轴横移到 ±`FOV_LATERAL_FRAC`·Z
#
# **为什么是"横移 ∝ Z"而不是固定米数**(实测定的,不是拍的):回归 `u = (fx/Z)·x + cx`
# 里 x 的杠杆臂是 `fx·x/Z`,而 `x/Z` 顶到画幅上限(±0.45 时已到半幅的 90%)⇒ 放大杠杆
# 只能靠**缩短 Z**。但 Z 一小,掩膜中点噪声 δ 放大成 fx 误差 `δ·Z/(2x)` 反而变小…
# 实测两者都要:±0.45·Z 的长臂把 CAM_FRONT_LEFT 的 fx 从 1292(dev 0.78°,Z=8/半幅 3.6)
# 拉到 1272.44(dev 0.006°)。故横移按比例取满画幅、Z 取**阶梯**(见 `FOV_Z_LADDER`)。
FOV_LATERAL_FRACS = (-0.45, -0.3375, -0.225, -0.1125, 0.0, 0.1125, 0.225, 0.3375, 0.45)
# Z 阶梯:**从远到近**逐档试,取第一档"量得到"的。远处杠杆臂最长(fx 最准),但**地图遮挡
# 随位姿而变** —— 实测同一相机在不同出生点下 Z=20 可用 / Z=14 不可用都会出现,故必须阶梯
# 兜底(不硬编码某一个 Z,否则换个 spawn point 判据就假失败)。
FOV_Z_LADDER = (20.0, 14.0, 10.0, 8.0, 6.0)
MIN_FOV_SAMPLES = 6  # 9 个锥里至少这么多可用才认这一档(少于则换更近的 Z)
N_FOV_FRAMES = 3  # 逐锥取多帧掩膜中点的**中位数**(单帧偶发遮挡/抗锯齿不参与结论)
MIN_MASK_PX = 8
CAM_W, CAM_H = 1600, 900


# ---------------------------------------------------------------- 纯值:判据 ③④


def radar_points_in_own_fov(
    pts_sensor: np.ndarray,
    r_declared: np.ndarray,
    r_official: np.ndarray,
) -> float:
    """该通道自己的点落进**官方光轴锥**的比例(判据 ③ 的本体)。

    `pts_sensor` 是 `(N,3)` 的**传感器自身系**点(`RadarPointCloud.points[:3].T` 口径)。
    **传 `(3,N)` 会静默只取前 3 个点**(判据看似有数其实无意义),故这里显式校验列数。

    点云在传感器自身系 ⇒ `p_ego = R_declared @ p_sensor`;再退回官方传感器系
    `R_official^T @ p_ego` ⇒ 此时官方光轴就是 +x,锥判据退化成极坐标比较。
    `R_declared == R_official` 时该式恒等于"点在自身系里的锥内占比"(官方集实测
    0.86/0.81/0.85/0.91/0.90);声明 R 写错则塌到 0.00–0.04。
    """
    p = np.asarray(pts_sensor, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"点必须是 (N,3) 的传感器自身系坐标,收到 {p.shape}")
    p = p.T
    if p.shape[1] == 0:
        return float("nan")
    off = np.asarray(r_official, dtype=np.float64).T @ (np.asarray(r_declared, dtype=np.float64) @ p)
    rng = np.linalg.norm(off, axis=0)
    az = np.degrees(np.arctan2(off[1], off[0]))
    el = np.degrees(np.arcsin(np.clip(off[2] / np.maximum(rng, 1e-9), -1.0, 1.0)))
    ok = (np.abs(az) <= ARS408_AZ_HALF_DEG) & (np.abs(el) <= ARS408_EL_HALF_DEG)
    return float((ok & (rng <= ARS408_RANGE_M)).mean())


def count_points_in_box_nus(
    pts_global: np.ndarray, translation: np.ndarray, size: tuple[float, float, float], yaw: float
) -> int:
    """全局系点 → 框内计数(size=(w,l,h),框体轴沿 yaw)。与 `export.nuscenes` 同口径。"""
    pts = np.asarray(pts_global, dtype=np.float64)[:, :3]
    rel = pts - np.asarray(translation, dtype=np.float64)
    cy, sy = math.cos(yaw), math.sin(yaw)
    u = rel[:, 0] * cy + rel[:, 1] * sy
    v = -rel[:, 0] * sy + rel[:, 1] * cy
    w, l, h = size
    inside = (np.abs(u) <= l / 2) & (np.abs(v) <= w / 2) & (np.abs(rel[:, 2]) <= h / 2)
    return int(inside.sum())


def _yaw_of_quat(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def lidar_ratio(dataroot: Path, r_override: np.ndarray | None = None, limit: int = 20) -> dict[str, Any]:
    """LiDAR 复现 `num_lidar_pts` 的比值(判据 ④)。

    `r_override` 给 None 时用**落盘的** `calibrated_sensor.rotation`(自洽性:消费方按表
    复现标注);给一个矩阵时用它替换(消融用,如单位阵 —— 官方集上 1.0000 → 0.1684)。
    """
    from nuscenes.utils.data_classes import LidarPointCloud

    tdir = dataroot / VERSION
    cs = {c["token"]: c for c in json.loads((tdir / "calibrated_sensor.json").read_text())}
    sens = {s["token"]: s for s in json.loads((tdir / "sensor.json").read_text())}
    ego = {e["token"]: e for e in json.loads((tdir / "ego_pose.json").read_text())}
    anns = json.loads((tdir / "sample_annotation.json").read_text())
    sd = json.loads((tdir / "sample_data.json").read_text())

    lid_cs, lid_sds = None, []
    for s in sd:
        c = cs[s["calibrated_sensor_token"]]
        if sens[c["sensor_token"]]["channel"] == "LIDAR_TOP" and s["is_key_frame"]:
            lid_cs, _ = c, lid_sds.append(s)
    if lid_cs is None or not lid_sds:
        return {"ratio": None, "reason": "无 LIDAR_TOP keyframe"}

    r_cal = (
        np.asarray(r_override, dtype=np.float64)
        if r_override is not None
        else quat_to_matrix(quat_normalize(lid_cs["rotation"]))
    )
    t_cal = np.asarray(lid_cs["translation"], dtype=np.float64)[:, None]
    tot_num, tot_gt = 0, 0
    for s in lid_sds[:limit]:
        ps = LidarPointCloud.from_file(str(dataroot / s["filename"])).points[:3]
        e = ego[s["ego_pose_token"]]
        r_e = quat_to_matrix(quat_normalize(e["rotation"]))
        t_e = np.asarray(e["translation"], dtype=np.float64)[:, None]
        pts_g = (r_e @ (r_cal @ ps + t_cal) + t_e).T
        for a in anns:
            if a["sample_token"] != s["sample_token"]:
                continue
            tot_gt += int(a["num_lidar_pts"])
            tot_num += count_points_in_box_nus(
                pts_g, a["translation"], tuple(a["size"]), _yaw_of_quat(a["rotation"])
            )
    return {
        "n_keyframes": len(lid_sds[:limit]),
        "gt_total": tot_gt,
        "reproduced_total": tot_num,
        "ratio": (tot_num / tot_gt) if tot_gt else None,
    }


# ---------------------------------------------------------------- 离线:③④⑤


def _load_tables(dataroot: Path) -> dict[str, Any]:
    tdir = dataroot / VERSION
    return {
        "calib": json.loads((tdir / "calibrated_sensor.json").read_text()),
        "sensor": json.loads((tdir / "sensor.json").read_text()),
        "sample_data": json.loads((tdir / "sample_data.json").read_text()),
    }


def _calib_by_channel(dataroot: Path) -> dict[str, dict]:
    tb = _load_tables(dataroot)
    sens = {s["token"]: s for s in tb["sensor"]}
    out: dict[str, dict] = {}
    for c in tb["calib"]:
        ch = sens[c["sensor_token"]]["channel"]
        out.setdefault(ch, c)
    return out


def official_radar_points() -> dict[str, np.ndarray]:
    """官方 mini 逐通道雷达点 `(N,3)`(传感器自身系)。判据 ③ 的**独立锚**。

    用官方点而不是本仓自采点:判据要测的是"落盘的 R 与官方 R 是不是同一个",
    不该被"本仓这一帧采到几个点"干扰 —— 且这样修前/修后的数字可直接与
    Plan2 §P-M.7.4 的表对齐(0.0000/0.0000/0.0011/0.0015)。
    """
    from nuscenes.utils.data_classes import RadarPointCloud

    tdir = OFFICIAL_MINI / VERSION
    cs = {c["token"]: c for c in json.loads((tdir / "calibrated_sensor.json").read_text())}
    sens = {s["token"]: s for s in json.loads((tdir / "sensor.json").read_text())}
    per: dict[str, list[np.ndarray]] = {ch: [] for ch in NUS_RADAR_CHANNELS}
    for s in json.loads((tdir / "sample_data.json").read_text()):
        ch = sens[cs[s["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        if ch in per:
            pc = RadarPointCloud.from_file(str(OFFICIAL_MINI / s["filename"]))
            per[ch].append(pc.points[:3].T)  # (N,3)
    return {ch: (np.vstack(v) if v else np.empty((0, 3))) for ch, v in per.items()}


def run_offline(dataroot: Path) -> dict[str, Any]:
    rep: dict[str, Any] = {"dataroot": str(dataroot), "official_reference": str(OFFICIAL_MINI)}

    # ③ 雷达点落进自身 FOV:官方点(独立锚)+ 落盘的 R
    ours = _calib_by_channel(dataroot)
    off = _calib_by_channel(OFFICIAL_MINI)
    off_pts = official_radar_points()
    radar: dict[str, Any] = {}
    for ch in NUS_RADAR_CHANNELS:
        pts = off_pts[ch]
        r_dec = quat_to_matrix(quat_normalize(ours[ch]["rotation"]))
        r_off = quat_to_matrix(quat_normalize(off[ch]["rotation"]))
        frac = radar_points_in_own_fov(pts, r_dec, r_off)
        radar[ch] = {
            "n_points": int(pts.shape[0]),
            "fov_frac": frac,
            "pass": bool(frac >= MIN_FOV_FRAC) if frac == frac else False,
        }
    rep["criterion_3_radar_fov_frac"] = {
        "threshold": MIN_FOV_FRAC,
        "note": "官方点 × 本仓落盘 R(以官方 R 为锚);修前 = 0.0000/0.0000/0.0011/0.0015",
        "channels": radar,
        "pass": all(v["pass"] for v in radar.values()),
    }

    # ④ LiDAR 复现 num_lidar_pts:官方锚(有区分度)+ 本仓自洽
    repo_ratio = lidar_ratio(dataroot)
    r4 = repo_ratio["ratio"]
    lid = ours["LIDAR_TOP"]
    t_dev = max(abs(a - b) for a, b in zip(lid["translation"], NUS_LIDAR_CALIB[0], strict=True))
    q_dev = max(abs(a - b) for a, b in zip(lid["rotation"], NUS_LIDAR_CALIB[1], strict=True))
    rep["criterion_4_lidar_num_pts"] = {
        "official_anchor": {
            "persisted_R": lidar_ratio(OFFICIAL_MINI),
            "ablation_identity_R": lidar_ratio(OFFICIAL_MINI, r_override=np.eye(3)),
        },
        "repo": {
            "persisted_R": repo_ratio,
            "mount_dev_m": t_dev,
            "quat_dev": q_dev,
            "mount_matches_official": bool(t_dev < 1e-9 and q_dev < 1e-9),
        },
        "pass": bool(r4 is not None and abs(r4 - 1.0) < 1e-6 and t_dev < 1e-9 and q_dev < 1e-9),
    }

    # ⑤ 相机内参 vs 本 rig 的声明表(**注意基准随 rig 换,含义也跟着换**)
    #   nuscenes:比官方 n015 实测 K(独立锚:真实装配公差)
    #   wide    :比由声明 FoV 反推的 K —— 这是**自洽性锁**,不是独立锚(wide 的 K 就是产物)
    intr: dict[str, Any] = {}
    for cam, (fx, cx, cy) in rig_intrinsics().items():
        k = ours[cam]["camera_intrinsic"]
        d = (abs(k[0][0] - fx), abs(k[0][2] - cx), abs(k[1][2] - cy))
        intr[cam] = {"max_dev_px": max(d), "pass": bool(max(d) < TOL_K_PX), "k": k}
    rep["criterion_5_camera_intrinsics"] = {
        "rig": RIG,
        "threshold_px": TOL_K_PX,
        "k_source": "官方 n015 实测(独立锚)" if RIG == "nuscenes" else "由声明 FoV 反推(自洽性锁,非独立锚)",
        "channels": intr,
        "pass": all(v["pass"] for v in intr.values()),
    }
    return rep


# ---------------------------------------------------------------- 在线:①②⑥


def _radar_carla_spec() -> tuple[
    dict[str, tuple[float, float, float]], dict[str, tuple[float, float, float]]
]:
    """雷达 CARLA 口径 `(mount, (pitch,yaw,roll))`。

    **直接取采集器的导出**(`collect_nus.NUS_RADAR_MOUNTS_CARLA` / `RADAR_YAW_OFFSET`),
    不在此另算一份 —— 就地重写 `(t[0], -t[1], t[2])` 会让"验收"与"采集"各自成立而彼此
    不成立(§P-M.7 的失效模式)。`collect_nus` 顶层 `import carla`,故本函数内延迟导入
    (模块要能在无 CARLA 的 `--offline` 模式下 import)。
    """
    import collect_nus

    mounts = {ch: collect_nus.NUS_RADAR_MOUNTS_CARLA[ch] for ch in NUS_RADAR_CHANNELS}
    rots = {ch: (0.0, collect_nus.RADAR_YAW_OFFSET[ch], 0.0) for ch in NUS_RADAR_CHANNELS}
    return mounts, rots


# ---------------------------------------------------------------- 判据 ⑨⑩:ego 原点

# 传感器世界位姿链的容差。**为什么能这么紧**:两边都是同一台机器上同一个浮点链的两条
# 独立路径(渲染位姿 vs 表 ⊕ ego),量级 1e-6 —— 1e-3 只是给四元数归一化与 UE 单位换算留的余量。
TOL_CHAIN_M = 1e-3
TOL_CHAIN_DEG = 1e-3

# 后轴/前轴复测的容差(米)。双偏航自解在同一台车上是确定的,四轮一致度实测 2e-06 m。
TOL_AXLE_M = 1e-3


def measure_ego_axles(world, ego) -> dict[str, Any]:
    """实机重测 `audi.a2` 的**前后轴在 CARLA 车体系里的 x**(双偏航自解)。

    为什么不能"读一次轮子位置再减 ego 位置":`get_physics_control().wheels[i].position`
    是 **UE 世界坐标(厘米)**,与 CARLA 世界系的关系(是否只差 /100、y 是否翻号)**没有文档
    保证**。本函数只用一个与之无关的事实 —— **刚性**:同一轮子在两个 ego 偏航 ψ₁/ψ₂ 下的
    世界坐标满足 `W(ψ) = C + R(ψ)·w₀`,`C` = ego 原点在世界里的位置。取 ψ₂ = ψ₁ + 90°
    后 `(I − R⁻¹)` 可逆,两个方程解出 `w₀` 与 `C`;四个轮子各自解一遍,**解出的 C 必须一致**
    (实测互差 2e-06 m)—— 这就是判据的自证,不需要先知道 UE↔CARLA 的换算。

    轴别由 `max_steer_angle` 判(前轮 70、后轮 0),**不按轮序猜**(实测按符号分组会直接除零)。
    纵向取前后轴中点的连线方向(UA 世界系里的车头),再把轮子投影上去取 x。测量期间会移动
    ego,函数**结束时恢复原位并 tick**,不改动调用方的世界状态。
    """
    import math as _m

    import carla

    orig = ego.get_transform()

    def wheels() -> list[tuple[tuple[float, float, float], float]]:
        return [
            (
                (wh.position.x / 100.0, wh.position.y / 100.0, wh.position.z / 100.0),
                float(wh.max_steer_angle),
            )
            for wh in ego.get_physics_control().wheels
        ]

    def put(yaw_deg: float) -> None:
        ego.set_transform(
            carla.Transform(
                carla.Location(orig.location.x, orig.location.y, orig.location.z),
                carla.Rotation(yaw=yaw_deg),
            )
        )
        for _ in range(3):
            world.tick()

    try:
        a = wheels()
        put(orig.rotation.yaw + 90.0)
        b = wheels()
    finally:
        ego.set_transform(orig)
        for _ in range(3):
            world.tick()

    # ψ₂ = ψ₁ + 90° ⇒ R(90°) 在 UE 世界系把 (x,y) 映到 (−y,x) 或 (y,−x);两档都试,
    # **以"四轮解出的 C 一致"为准**挑那一档(与轴系手性无关) —— 两档都能解出数字,
    # 只有一档自洽。实测不一致档的四轮 C 散到米级。
    best: dict[str, Any] | None = None
    for sign in (+1, -1):
        cs: list[tuple[float, float]] = []
        ws: list[tuple[float, float]] = []
        for i in range(4):
            tx, ty = b[i][0][0] - a[i][0][0], b[i][0][1] - a[i][0][1]
            if sign > 0:
                wy, wx = -(tx + ty) / 2.0, (ty - tx) / 2.0
            else:
                wy, wx = (tx - ty) / 2.0, (tx + ty) / 2.0
            ws.append((wx, wy))
            cs.append((a[i][0][0] - wx, a[i][0][1] - wy))
        spread = max(_m.dist(cs[0], c) for c in cs)
        if best is None or spread < best["spread_m"]:
            best = {"sign": sign, "spread_m": spread, "c": cs[0], "w": ws, "wheels": a, "wheels2": b}
    assert best is not None
    fr = [i for i in range(4) if best["wheels"][i][1] > 10.0]
    rr = [i for i in range(4) if best["wheels"][i][1] <= 10.0]
    if not fr or not rr:
        raise RuntimeError("前/后轴判据(max_steer_angle>10)未能分出两组轮子")

    def mean_of(idx: list[int]) -> tuple[float, float]:
        return (
            sum(best["wheels2"][i][0][0] for i in idx) / len(idx),
            sum(best["wheels2"][i][0][1] for i in idx) / len(idx),
        )

    fxy, rxy = mean_of(fr), mean_of(rr)
    fx, fy = fxy[0] - rxy[0], fxy[1] - rxy[1]
    n = _m.hypot(fx, fy)
    fx, fy = fx / n, fy / n  # UE 世界系里的车头单位矢量
    cx, cy = best["c"]
    return {
        "front_x_m": (fxy[0] - cx) * fx + (fxy[1] - cy) * fy,
        "rear_x_m": (rxy[0] - cx) * fx + (rxy[1] - cy) * fy,
        "wheelbase_m": _m.dist(fxy, rxy),
        "yaw_ue_deg": _m.degrees(_m.atan2(fy, fx)),
        "c_spread_m": best["spread_m"],
        "rot_sign": best["sign"],
        "ego_yaw_carla_deg": orig.rotation.yaw,
    }


def world_pose_chain(
    sensors: dict[str, Any],
    ego,
    calibs_nus: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]],
    origin_x: float,
) -> dict[str, Any]:
    """判据⑨:逐传感器比 **声明链** `ego_pose ⊕ calibrated_sensor` 与 **CARLA 实挂世界位姿**。

    ⚠️ 为什么必须新增这一条:判据 ①② 是**ego 相对**的(实挂挂点 vs `NUS_CAMERA_RIG`),
    而两者都从同一个 ego 取参照 ⇒ **ego 原点整体错位时它恒绿**。本轮修的就是那个错位
    (整套传感器相对自车偏 1.2563 m),① 全绿而画幅里 42.999% 是自身车体。本判据把链子
    一路对到世界系,并在**报告里回显** `ego_pose`.

    两边都是"表"而不是"渲染"的对照:`declared` 由落盘表 ⊕ **独立测得的**后轴位姿
    (`origin_x` 来自判据⑩的实测,不是常量)算出;`rendered` 由 `sensor.get_transform()`
    经 `CARLA_TO_NUS` 共轭换到 nus 全局系。故**不依赖 `NUS_EGO_ORIGIN_X` 的取值是否对**
    —— 那一条由⑩单独把守,两条合起来才是完整的。
    """
    t_actor = ego.get_transform()
    rot_rad = (
        math.radians(t_actor.rotation.pitch),
        math.radians(t_actor.rotation.yaw),
        math.radians(t_actor.rotation.roll),
    )
    t_ego_nus = g.nus_ego_translation(
        (t_actor.location.x, t_actor.location.y, t_actor.location.z),
        rot_rad,
        origin_x,
    )
    # ★ 全 6DoF:只取 yaw 会漏掉悬架俯仰(实测 +0.0642°),本判据会因此稳定差 0.064°
    # ——**那正是它要抓的东西**,不能用近似把它自己蒙掉。
    r_ego = quat_to_matrix(g.nus_ego_rotation(rot_rad))
    t_e = np.asarray(t_ego_nus, dtype=np.float64)
    m = g.CARLA_TO_NUS

    rows: dict[str, Any] = {}
    for name, s in sensors.items():
        q_nus = calibs_nus[name][1]
        r_cal = quat_to_matrix(quat_normalize(q_nus))
        t_cal = np.asarray(calibs_nus[name][0], dtype=np.float64)
        t_dec = r_ego @ t_cal + t_e
        r_dec = r_ego @ r_cal
        tw = s.get_transform()
        t_rend = m @ np.array([tw.location.x, tw.location.y, tw.location.z], dtype=np.float64)
        r_ue = g.carla_rotation_matrix(
            (math.radians(tw.rotation.pitch), math.radians(tw.rotation.yaw), math.radians(tw.rotation.roll))
        )
        # 局部基的重排**逐类型不同**:相机局部轴是 (x 前,y 右,z 上) 而 nus 相机是 (x 右,y 下,
        # z 前) ⇒ 多一个 `CARLA_CAM_TO_NUS_CAM`;LiDAR/雷达局部基与 CARLA 同 ⇒ 只有 y 翻号。
        # 拿 LiDAR 那条去比相机会得到一个**恒为 120°**的假误差(轮换阵本征角),见其头注。
        r_rend = m @ r_ue @ (g.CARLA_CAM_TO_NUS_CAM if name in NUS_CAMERAS else m)
        dt = float(np.max(np.abs(t_dec - t_rend)))
        dr = float(np.max(np.abs(r_dec - r_rend)))
        # 旋转误差也报成**角度**(矩阵元素差读不出量级):trace(R_dᵀR_r) = 1 + 2cosθ
        cos_t = (np.trace(r_dec.T @ r_rend) - 1.0) / 2.0
        rows[name] = {
            "dev_translation_m": dt,
            "dev_rotation_deg": math.degrees(math.acos(float(np.clip(cos_t, -1.0, 1.0)))),
            "dev_rotation_matrix": dr,
            "pass": bool(dt < TOL_CHAIN_M and dr < 1e-5),
        }
    return {
        "note": "declared = 落盘表 ⊕ **实测**后轴位姿;rendered = CARLA 实挂经 CARLA_TO_NUS 共轭",
        "origin_x_used_m": origin_x,
        "tol_m": TOL_CHAIN_M,
        "sensors": rows,
        "pass": all(v["pass"] for v in rows.values()),
    }


def run_live(host: str, port: int) -> dict[str, Any]:
    import carla
    import collect_nus
    import probe_calib as pc
    from carla_common import spawn_ego, sync_mode
    from live_common import mount_deviation_of, rig_spec

    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)
    ego = spawn_ego(world)
    bp_lib = world.get_blueprint_library()
    rep: dict[str, Any] = {"map": world.get_map().name}

    # ★ 判据⑩ 先测:它要移动 ego(双偏航),放在 spawn 传感器**之前**最干净
    # (测完已恢复原位;见 `measure_ego_axles`)。
    axles = measure_ego_axles(world, ego)
    rep["criterion_10_ego_origin_remeasure"] = {
        "tol_m": TOL_AXLE_M,
        "expected_origin_x_m": g.NUS_EGO_ORIGIN_X,
        "measured_rear_axle_x_m": axles["rear_x_m"],
        "measured_front_axle_x_m": axles["front_x_m"],
        "wheelbase_m": axles["wheelbase_m"],
        "c_consistency_m": axles["c_spread_m"],
        "note": (
            "双偏航自解 W(ψ)=C+R(ψ)·w₀;四轮各自解出的 C 必须一致(自证)。"
            "常量腐化(换 ego 蓝图)= 静默回到整组传感器原点错位,故必须每次复测"
        ),
        "pass": bool(abs(axles["rear_x_m"] - g.NUS_EGO_ORIGIN_X) < TOL_AXLE_M),
    }

    # ★ **等悬架稳定再挂传感器**:`measure_ego_axles` 双偏航自解时把 ego 摆平过(pitch=0),
    # 恢复原位后车体还要**重新下沉** ~20 tick(实测 pitch 0 → 0.148 → 0.083 → 0.064 收敛)。
    # 不等就挂:传感器 `get_transform()` 记的是**挂上那一刻**的姿态,而判据⑨ 读 ego 位姿是在
    # 之后 ⇒ 两边差一个正在变的俯仰,"声明 vs 实挂"的残差会漂(实测 0.0014–0.0031 m 散开,
    # 看着像标定问题,其实是**测量时机**问题 —— 与 Plan.md 红线里"快照陈旧"同族)。
    prev = (ego.get_transform().rotation.pitch, ego.get_transform().rotation.roll)
    settle = 0
    for _ in range(60):
        world.tick()
        settle += 1
        cur = ego.get_transform().rotation
        if max(abs(cur.pitch - prev[0]), abs(cur.roll - prev[1])) < 1e-4:
            break
        prev = (cur.pitch, cur.roll)
    rep["ego_settle"] = {
        "ticks": settle,
        "pitch_deg": ego.get_transform().rotation.pitch,
        "roll_deg": ego.get_transform().rotation.roll,
        "note": "挂传感器前的静置 tick 数;判据⑨⑩ 都要求车体姿态已收敛",
    }

    cams: dict[str, carla.Sensor] = {}
    radars: dict[str, carla.Sensor] = {}
    lidar = None
    try:
        for cam, (mount, rot) in rig_cameras().items():
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(CAM_W))
            bp.set_attribute("image_size_y", str(CAM_H))
            bp.set_attribute("fov", f"{camera_fov(RIG)[cam]:.6f}")
            tf = carla.Transform(
                carla.Location(*mount), carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
            )
            cams[cam] = cast(carla.Sensor, world.spawn_actor(bp, tf, attach_to=ego))
        rmounts, rrots = _radar_carla_spec()
        for ch in NUS_RADAR_CHANNELS:
            bp = bp_lib.find("sensor.other.radar")
            for k, v in collect_nus.RADAR_ATTRS.items():
                bp.set_attribute(k, v)
            m, r = rmounts[ch], rrots[ch]
            tf = carla.Transform(carla.Location(*m), carla.Rotation(pitch=r[0], yaw=r[1], roll=r[2]))
            radars[ch] = cast(carla.Sensor, world.spawn_actor(bp, tf, attach_to=ego))
        # LiDAR 也挂上:判据⑨ 要覆盖全部 12 个传感器(它此前只被判据④ 从**表**那一侧查过,
        # 没有"实挂世界位姿"这一侧)。挂点/姿态取采集器的同一份导出。
        lbp = bp_lib.find("sensor.lidar.ray_cast")
        for k, v in collect_nus.LIDAR_ATTRS.items():
            lbp.set_attribute(k, v)
        lidar = cast(
            carla.Sensor,
            world.spawn_actor(
                lbp,
                carla.Transform(
                    carla.Location(*collect_nus.LIDAR_MOUNT),
                    carla.Rotation(
                        pitch=collect_nus.LIDAR_ROT[0],
                        yaw=collect_nus.LIDAR_ROT[1],
                        roll=collect_nus.LIDAR_ROT[2],
                    ),
                ),
                attach_to=ego,
            ),
        )

        # ★ 必须先 tick:`get_transform()` 在 tick 前是全 0 陈旧值(Plan.md 红线)
        for _ in range(3):
            world.tick()

        cmounts, crots = rig_spec(RIG)
        dev_t, dev_y = mount_deviation_of(cams, ego, cmounts, crots)
        rep["criterion_1_camera_mount"] = {
            "tol_m": TOL_MOUNT_M,
            "tol_deg": TOL_MOUNT_DEG,
            "dev_translation_m": dev_t,
            "dev_yaw_deg": dev_y,
            "pass": bool(dev_t < TOL_MOUNT_M and dev_y < TOL_MOUNT_DEG),
        }
        rdev_t, rdev_y = mount_deviation_of(radars, ego, rmounts, rrots)
        rep["criterion_2_radar_mount"] = {
            "tol_deg": TOL_RADAR_DEG,
            "dev_translation_m": rdev_t,
            "dev_yaw_deg": rdev_y,
            "pass": bool(rdev_y < TOL_RADAR_DEG and rdev_t < TOL_MOUNT_M),
        }
        # ⑨ **世界位姿链**:12 个传感器的"声明 ⊕ ego_pose" vs "CARLA 实挂"。
        # 判据①② 是 ego 相对的 ⇒ **ego 原点整体错位时恒绿**(本轮修的就是那个),本判据补上。
        declared_nus: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {
            cam: camera_calibs(RIG)[cam] for cam in rig_cameras()
        }
        declared_nus["LIDAR_TOP"] = NUS_LIDAR_CALIB
        for ch in NUS_RADAR_CHANNELS:
            t_r, yaw_r = NUS_RADAR_OFFSETS[ch]
            declared_nus[ch] = (t_r, g.yaw_to_quat(yaw_r))
        sensors: dict[str, Any] = {**cams, **radars, "LIDAR_TOP": lidar}
        chain = world_pose_chain(sensors, ego, declared_nus, axles["rear_x_m"])
        rep["criterion_9_world_pose_chain"] = chain
        rep["criterion_6_rendered_fov"] = _rendered_fov(world, ego, pc)

        # ⑦⑧ 同一套 instance_seg 相机:零车体像素 + 相邻共视(实现见 `bin/rig_check.py`)
        import rig_check

        probe = rig_check.instance_probe(world, ego, RIG)
        rep["criterion_7_no_ego_pixels"] = {
            "rig": RIG,
            "ego_actor_id": probe["ego_actor_id"],
            "note": "instance_segmentation 里数 **ego 自己的 actor id**;0 px = 画幅内无自身车体",
            "channels": probe["ego_pixels"],
            "pass": probe["ego_pixels_pass"],
        }
        rep["criterion_8_adjacent_covisibility"] = {
            "rig": RIG,
            "min_overlap_deg": rig_check.MIN_COVIS_OVERLAP_DEG,
            "note": (
                "重叠带正中摆锥(抬高到视轴高度),两路掩膜都命中才算共视;"
                "`common_band_deg` = 该距离下两路**真正共同可见**的方位带(≠ 方位轴重叠,差在挂点视差)"
            ),
            "pairs": probe["covisibility"],
            "pass": probe["covisibility_pass"],
        }
    finally:
        for s in (*cams.values(), *radars.values(), lidar):
            if s is None:
                continue
            s.stop()
            s.destroy()
        for a in world.get_actors():
            if a.type_id.startswith(("vehicle", "walker", "controller")):
                a.destroy()
    return rep


def _rendered_fov(world, ego, pc) -> dict[str, Any]:
    """⑥ 逐相机:沿光轴 Z 处横移锥 → 掩膜中点回归 `u=(fx/Z)x+cx` → `fx_est` → HFOV。

    复用 `probe_calib` 的锥体/掩膜辅助(`try_spawn_cone` / `mask_centre_u` /
    `decode_instance`),**不另写一遍实例分割解码**;横移位置自己算(按 `FOV_LATERAL_FRACS`,
    因为杠杆臂要按比例顶满画幅,`probe_calib.lateral_cone_positions` 是固定米数的 A4 口径)。

    **★ 每 tick 必须抽干所有相机队列**(`drain`):六个相机同时 listen 时,同步模式下每 tick
    每个队列各进一帧。若只取被测相机那一帧,其余五路会积压 —— 轮到它们时读到的是**锥体还没
    spawn 之前**的陈旧帧 ⇒ 掩膜恒 0。实测症状极具迷惑性:六相机里**只有第一个能测出数**,
    其余五路 `instance_hits=0` 且**与距离无关**(像"摆不进去",其实不是)。

    **★ 为什么要 Z 阶梯**:地图遮挡随 ego 出生点而变,同一相机在不同位姿下"Z=20 可用 /
    Z=14 不可用"都会出现。故从远到近逐档试,**取第一档可用样本 ≥ `MIN_FOV_SAMPLES`**;
    远档优先是因为杠杆臂 ∝ Z(远处 fx 更准)。全档都不够 → 该相机如实报 `pass=False` +
    `reason`,**不硬给一个凑合的数**。
    """
    import carla
    from carla_common import loc, rad

    def drain() -> dict[str, bytes]:
        """tick 一次 + **抽干全部队列**,返回逐相机当前帧(不抽干就会积压陈旧帧)。"""
        world.tick()
        return {name: q.get(timeout=10).raw_data for name, (_, q) in iseg.items()}

    out: dict[str, Any] = {"threshold_deg": TOL_FOV_DEG, "cameras": {}}
    bp_lib = world.get_blueprint_library()
    iseg: dict[str, tuple[carla.Sensor, queue.Queue]] = {}
    try:
        for cam, (mount, rot) in rig_cameras().items():
            bp = bp_lib.find("sensor.camera.instance_segmentation")
            bp.set_attribute("image_size_x", str(CAM_W))
            bp.set_attribute("image_size_y", str(CAM_H))
            bp.set_attribute("fov", f"{camera_fov(RIG)[cam]:.6f}")
            tf = carla.Transform(
                carla.Location(*mount), carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
            )
            s = cast(carla.Sensor, world.spawn_actor(bp, tf, attach_to=ego))
            q: queue.Queue = queue.Queue()
            s.listen(q.put)
            iseg[cam] = (s, q)
        for _ in range(5):
            drain()

        for cam in rig_cameras():
            t = iseg[cam][0].get_transform()
            pose = (loc(t), rad(t.rotation))
            r_cam = cp.cam_to_world_rot(pose[1])
            origin = np.asarray(pose[0], dtype=np.float64)
            row: dict[str, Any] = {"z_ladder": list(FOV_Z_LADDER)}
            for z in FOV_Z_LADDER:
                xs = [f * z for f in FOV_LATERAL_FRACS]
                placed: list[tuple[float, carla.Actor]] = []
                for x in xs:
                    a = pc.try_spawn_cone(world, tuple(origin + r_cam @ np.array([x, 0.0, z])))
                    if a is not None:
                        placed.append((x, a))
                centres: dict[int, list[float]] = {a.id: [] for _, a in placed}
                for _ in range(N_FOV_FRAMES):
                    inst = pc.decode_instance(drain()[cam], CAM_H, CAM_W)
                    for _, a in placed:
                        c = pc.mask_centre_u(pc.cone_masks({cam: inst}, a.id)[cam])
                        if c is not None:
                            centres[a.id].append(c)
                samples = [(x, float(np.median(centres[a.id]))) for x, a in placed if centres[a.id]]
                for _, a in placed:
                    a.destroy()
                if len(samples) >= MIN_FOV_SAMPLES:
                    _, slope, resid = cp.prop_axis_regression(
                        np.asarray([s[0] for s in samples]), np.asarray([s[1] for s in samples]), 0.0
                    )
                    fx_est = slope * z
                    hfov = math.degrees(2.0 * math.atan((CAM_W / 2.0) / fx_est))
                    row.update(
                        {
                            "z_used_m": z,
                            "n_placed": len(placed),
                            "n_used": len(samples),
                            "fx_est_px": fx_est,
                            "hfov_rendered_deg": hfov,
                            "hfov_blueprint_deg": camera_fov(RIG)[cam],
                            "dev_deg": abs(hfov - camera_fov(RIG)[cam]),
                            "max_residual_px": float(np.max(np.abs(resid))),
                        }
                    )
                    row["pass"] = bool(row["dev_deg"] < TOL_FOV_DEG)
                    break
                row.setdefault("tried", []).append(
                    {"z_m": z, "n_placed": len(placed), "n_used": len(samples)}
                )
            else:
                row["reason"] = (
                    f"Z 阶梯 {FOV_Z_LADDER} 全档可用样本 < {MIN_FOV_SAMPLES}"
                    "(该出生点地图遮挡过重,换 spawn point 重跑)"
                )
            out["cameras"][cam] = row
    finally:
        for s, _ in iseg.values():
            s.stop()
            s.destroy()
    out["pass"] = all(v.get("pass", False) for v in out["cameras"].values())
    return out


# ---------------------------------------------------------------- 汇总


def summarize(rep: dict[str, Any]) -> tuple[dict[str, bool], str]:
    keys = [k for k in rep if k.startswith("criterion_")]
    verdict = {k: bool(rep[k].get("pass", False)) for k in sorted(keys)}
    lines = [f"  {'✓' if v else '✗'} {k}" for k, v in verdict.items()]
    return verdict, "\n".join(lines)


def main() -> int:
    global RIG  # 模块级 rig 上下文:run_offline/run_live 里的 spawn 与声明表都读它
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="跑判据 ③④⑤(不需 CARLA)")
    ap.add_argument("--live", action="store_true", help="跑判据 ①②⑥⑦⑧(需 CARLA 服务器)")
    ap.add_argument("--rig", choices=NUS_RIGS, default="nuscenes", help="相机 rig(默认官方口径)")
    ap.add_argument("--dataroot", default=None, help="默认按 rig 取 outputs/nus_mini[_wide]")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out", default=None, help="默认按 rig 取 outputs/nus_calib_check/report[_wide].json")
    args = ap.parse_args()
    if not (args.offline or args.live):
        ap.error("至少要给 --offline 或 --live")
    RIG = args.rig
    # 默认落点按 rig 分叉:同一份默认值写死在两个 rig 上会让 wide 静默覆盖官方报告
    if args.dataroot is None:
        args.dataroot = "outputs/nus_mini_wide" if RIG == "wide" else "outputs/nus_mini"
    if args.out is None:
        name = "report_wide.json" if RIG == "wide" else "report.json"
        args.out = f"outputs/nus_calib_check/{name}"
    rep: dict[str, Any] = {"rig": RIG}
    if args.offline:
        rep.update(run_offline(project_path(args.dataroot)))
    if args.live:
        rep.update(run_live(args.host, args.port))

    verdict, text = summarize(rep)
    rep["verdict"] = verdict
    out = project_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(text)
    print(f"[report] {out.resolve()}")
    return 0 if all(verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

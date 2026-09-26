"""实时标定槽(`live_studio --calib`)的纯值件:点云预算 / 残差着色 / 逐相机统计 / HUD 行。

与离线探针 [autodrivedata/calib/probe_calib.py](probe_calib.py) 的关系:那个是**一次性自证**
(静态 ego、训练口径全分辨率、落 `report.json` + `overlay.png`);本模块服务的是
**开着车时的持续监看**。两者共用同一套数值核心
([autodrivedata/calib/selfcheck.py](selfcheck.py)),差别只在预算与呈现:

| | 离线探针 | 实时槽 |
|---|---|---|
| 点云预算 | voxel 0.3 / 半径 1.2 m / 50 m / 上限 8000 | `LIVE_*`(更廉价,见下) |
| 分辨率 | 1242×375(训练口径) | **显示口径**(与 RGB 槽同尺寸,overlay 才能逐像素对齐) |
| 平面重拟合 | 一次(ego 静止 ⇒ 每帧点云相同) | 每 `--calib-refit` tick 一次 |
| 输出 | report.json + overlay.png | HUD 行 + 可选 JSON |

## 为什么平面可以隔几帧才重拟合

拟合出的平面是**世界系**的 —— 它描述的是**场景表面**,不是"这一帧的点云"。ego 移动几米
后同一块路面/墙面仍是同一个平面,拿上一帧的 (点, 法向, 截距) 投到**当前**相机位姿照样
成立(被挡住的点由 `collect_samples` 的**单侧**可见性判据剔掉,不需要重新拟合)。
实测(2026-09-22,640×360)拟合本身 ~100–150 ms/tick 而六相机采样合计仅 ~9 ms
⇒ **瓶颈全在拟合**,故默认每 2 tick 重拟合一次。

## CAM_BACK 的平台边界(实测,必须显式报,不能读成"标定坏了")

官方 nuScenes 的 `CAM_BACK` 挂点 `(x=0.028, y=−0.004, z=1.579)` 只比 CARLA ego **自身的
车顶**(bbox extent z 0.7745 + location z 0.7818 ⇒ z≈1.556)高 **0.023 m** ⇒ 它有很大一部分
画面被**自己的车顶**挡住。实测该相机近场(深度 < 0.5 m)像素占比在 **1242×375 下 0.367**、
在 **640×360 下 0.195**,其余五路均为 **0.000**。

后果:它的可用样本数常年 0–30(其余 50–200),但**残差中位数并不因此变差**(0.0003 m 量级,
与其它相机同级)—— 所以判据必须是"**样本数不足时不许报 median**",而不是"median 大 = 坏标定"。
本模块把这件事做成**数据驱动**:`near_fraction()` 量自遮挡占比,`self_occluded_cameras()`
按**同批其余相机**的近场占比定阈(见该函数 docstring:绝对阈值会被画幅宽高比打穿),
**不按相机名硬编码** —— 换一台 ego、换一个挂点、换一个分辨率,判据自动跟着走。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autodrivedata.calib import selfcheck as sc
from autodrivedata.calib.core import CameraIntrinsics
from autodrivedata.calib.depth_codec import CONVENTION_CORNER
from autodrivedata.slam.accum import voxel_downsample
from autodrivedata.utils.geometry import carla_rotation_matrix

# ---------------------------------------------------------------- 实时预算(实测标定,见模块头注)
# 体素边长 / 邻域平面半径 / 距离上限 / 平面拟合点数上限
LIVE_VOXEL_M = 1.0
LIVE_PLANE_RADIUS_M = 2.0
LIVE_MAX_DIST_M = 25.0
LIVE_MAX_SAMPLES = 1500
LIVE_PLANE_RMS_MAX_M = 0.08

# 相机样本数低于此值 ⇒ 不报 median(报 None)。20 是"少到中位数已无意义"的量级:
# 实测 CAM_BACK 自遮挡下常年 0–30,其余相机 50–200。
MIN_CAM_SAMPLES = 20

# 深度近场判据:渲染深度 < 此值 ⇒ 该像素被**极近的**东西占据(自遮挡的特征)
NEAR_DEPTH_M = 0.5
# 自遮挡判定:**相对**判据(见 `self_occluded_cameras`)。绝对阈值不成立 —— 同一台相机
# 同一挂点,近场占比随**画幅宽高比**变:1242×375(aspect 3.31)实测 0.367,
# 640×360(aspect 1.78)实测 0.195(水平 FOV 都是 90°,竖直 FOV 大得多 ⇒ 车顶占比小)。
NEAR_FRACTION_MIN = 0.05
NEAR_FRACTION_RATIO = 10.0

# 残差着色带(与离线探针同一套,离线侧已改为调用本模块的 `paint_residuals`)
RESIDUAL_GOOD_M = 0.05
RESIDUAL_WARN_M = 0.15
GOOD_COLOR = (0, 255, 0)
WARN_COLOR = (255, 220, 0)
BAD_COLOR = (255, 0, 0)

# 相机名 → HUD 短名(六个全名加起来 80+ 字符,一行放不下)
SHORT_NAMES = {
    "CAM_FRONT": "F",
    "CAM_FRONT_LEFT": "FL",
    "CAM_FRONT_RIGHT": "FR",
    "CAM_BACK": "B",
    "CAM_BACK_LEFT": "BL",
    "CAM_BACK_RIGHT": "BR",
}


def short_name(name: str) -> str:
    """`CAM_FRONT_LEFT` → `FL`;未知名字原样返回(不静默丢,免得 HUD 少一路看不出来)。"""
    return SHORT_NAMES.get(name, name)


# ---------------------------------------------------------------- 点云 → 世界系平面


def world_points_from_lidar(
    points_sensor: np.ndarray,
    location: tuple[float, float, float],
    rotation_rad: tuple[float, float, float],
) -> np.ndarray:
    """LiDAR 传感器系点 `(N, ≥3)` → 世界系 `(N,3)`。

    **不翻 y**:y 翻转是 KITTI 落盘口径(`semantic_to_velodyne_bin`),与几何无关;
    这里要的是世界几何,翻 y 会让所有点落到镜像位置(而残差仍"看着合理"一小段)。
    """
    p = np.asarray(points_sensor, dtype=np.float64).reshape(-1, points_sensor.shape[-1])[:, :3]
    r = carla_rotation_matrix(rotation_rad)
    return p @ r.T + np.asarray(location, dtype=np.float64).reshape(1, 3)


def live_planes(
    pts_world: np.ndarray,
    sensor_loc: np.ndarray,
    rng: np.random.Generator,
    voxel: float = LIVE_VOXEL_M,
    radius: float = LIVE_PLANE_RADIUS_M,
    max_dist: float = LIVE_MAX_DIST_M,
    max_samples: int = LIVE_MAX_SAMPLES,
    rms_max: float = LIVE_PLANE_RMS_MAX_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """世界点 → `(保留点, 逐点法向, 逐点截距)`;截距已按 `n·p = d` 算好可直接进 `collect_samples`。

    **距离一律相对 LiDAR 自身**(`sensor_loc`),不是世界原点:ego 出生点离原点可达 69 m,
    用 `‖P‖` 会把大量点误剔成"远点",平面拟合邻域全空 —— 不报错、只是样本数塌成 0。

    先体素降采样 + 随机抽到 `max_samples` 再拟合:逐点邻域平面是 O(N²),而实时槽每几
    tick 就要跑一次。**抽样在拟合之前**(逐点平面拟合对每点独立,抽样不改变任一保留点
    的判定),先拟合再抽会把"哪些点能过闸"交给运气。
    """
    pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts, pts, np.zeros(0)
    d = np.linalg.norm(pts - np.asarray(sensor_loc, dtype=np.float64).reshape(1, 3), axis=1)
    down = voxel_downsample(pts[d < max_dist], voxel)[:, :3]
    if down.shape[0] > max_samples:
        down = down[rng.choice(down.shape[0], max_samples, replace=False)]
    normals, rms = sc.fit_local_planes(down, radius)
    keep = np.isfinite(rms) & (rms <= rms_max)
    pts_k, nrm_k = down[keep], normals[keep]
    return pts_k, nrm_k, np.einsum("ij,ij->i", nrm_k, pts_k)


def empty_samples() -> sc.DepthSamples:
    """空采样集(某相机一个点都没投进来时的合法返回值,不是 None —— 免得调用方各写一份)。"""
    return sc.DepthSamples(np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros((0, 2)))


def sample_camera(
    planes: tuple[np.ndarray, np.ndarray, np.ndarray],
    cam_pose: tuple[tuple[float, float, float], tuple[float, float, float]],
    intrinsics: CameraIntrinsics,
    depth: np.ndarray,
) -> sc.DepthSamples:
    """单相机采样(实时槽口径:`CONVENTION_CORNER` + **不开**窗口极差)。

    窗口极差那一路(`occlusion_radius_px`)实测在 r=16 px 时把样本塌 95%,而单侧可见性
    判据在 r=0 就等价生效 ⇒ 实时槽一律不传(理由见 `selfcheck.collect_samples`)。
    """
    pts, nrm, off = planes
    if pts.shape[0] == 0:
        return empty_samples()
    return sc.collect_samples(
        pts, nrm, off, cam_pose[0], cam_pose[1], intrinsics, depth, CONVENTION_CORNER, None
    )


# ---------------------------------------------------------------- 残差着色


def residual_colors(residual: np.ndarray) -> np.ndarray:
    """`|残差|` → RGB `(N,3)` uint8(绿 < 0.05 m / 黄 < 0.15 m / 红 其余)。"""
    e = np.abs(np.asarray(residual, dtype=np.float64)).reshape(-1, 1)
    return np.where(
        e < RESIDUAL_GOOD_M,
        np.array(GOOD_COLOR, dtype=np.uint8),
        np.where(
            e < RESIDUAL_WARN_M, np.array(WARN_COLOR, dtype=np.uint8), np.array(BAD_COLOR, dtype=np.uint8)
        ),
    ).astype(np.uint8)


def paint_residuals(arr: np.ndarray, uv: np.ndarray, residual: np.ndarray) -> int:
    """把采样点按 `|残差|` 着色**就地**画进 `arr`(H,W,3 uint8)。返回画上的点数。

    `uv` 是**图像坐标**;换算成索引时取 `floor`(即 `index = u`),与
    `CONVENTION_CORNER`("索引即坐标")一致 —— 这里不能减 0.5,减了才是错的。
    越界钳到图内(采样点已由 `collect_samples` 保证在图内,钳位只为浮点边界那一列)。
    """
    n = int(np.asarray(residual).size)
    if n == 0:
        return 0
    h, w = arr.shape[:2]
    u = np.clip(np.asarray(uv, dtype=np.float64).reshape(-1, 2)[:, 0].astype(np.intp), 0, w - 1)
    v = np.clip(np.asarray(uv, dtype=np.float64).reshape(-1, 2)[:, 1].astype(np.intp), 0, h - 1)
    arr[v, u] = residual_colors(residual)
    return n


def near_fraction(depth: np.ndarray, thr: float = NEAR_DEPTH_M) -> float:
    """渲染深度 < `thr` 的像素占比 —— **自遮挡的可观测量**(近场被极近的几何占据)。

    判据来源:实测 CAM_BACK 0.367 vs 其余五路 0.000(2026-09-22)。用它而不是
    "相机名 == CAM_BACK" 硬编码,是为了换 ego/换挂点后判据仍然成立。
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.size == 0:
        return 0.0
    return float((d < thr).mean())


# ---------------------------------------------------------------- 逐相机统计


@dataclass(frozen=True)
class CameraResidual:
    """单相机的实时标定读数。`median_abs is None` ⇔ 样本不足,**不许**拿它当"标定坏"。"""

    name: str
    n: int
    median_abs: float | None
    p90_abs: float | None
    near_fraction: float

    @property
    def usable(self) -> bool:
        return self.n >= MIN_CAM_SAMPLES and self.median_abs is not None


def summarize(
    samples: dict[str, sc.DepthSamples],
    near: dict[str, float] | None = None,
) -> dict[str, CameraResidual]:
    """逐相机采样 → 统计。样本数 < `MIN_CAM_SAMPLES` 时 median/p90 记 `None`(不报假数字)。"""
    near = near or {}
    out: dict[str, CameraResidual] = {}
    for name, s in samples.items():
        n = len(s)
        if n >= MIN_CAM_SAMPLES:
            e = np.abs(s.residual)
            med: float | None = float(np.median(e))
            p90: float | None = float(np.percentile(e, 90))
        else:
            med = p90 = None
        out[name] = CameraResidual(name, n, med, p90, float(near.get(name, 0.0)))
    return out


def self_occluded_cameras(stats: dict[str, CameraResidual]) -> set[str]:
    """样本不足 **且** 近场占比远高于同批其余相机 ⇒ 判自遮挡(而非"标定不准")。

    **判据必须是相对的**(实测踩坑,2026-09-22):同一台相机、同一挂点,近场占比随**画幅
    宽高比**变 —— `CAM_BACK` 在 1242×375(aspect 3.31)是 **0.367**,在 640×360
    (aspect 1.78)只有 **0.195**(水平 FOV 都是 90°,竖直 FOV 大得多 ⇒ 车顶占比小)。
    写死 `> 0.2` 会让实时槽(640×360)判不出来。故基准取**同批可用相机的近场占比中位数**
    (典型为 0.000),阈值 = `max(NEAR_FRACTION_MIN, NEAR_FRACTION_RATIO × 基准)`。

    基准取"可用相机"而不是全体:若某台自己就是被挡的那台,把它算进基准会抬高阈值。
    全部不可用时**退回绝对下限**(基准取 0.0):此时没有"正常参照",但近场占比本身就远高于
    任何正常相机 —— 退回 0.0 让判据仍给得出结论,而不是因为"找不到参照"就沉默。
    """
    if not stats:
        return set()
    usable_near = [s.near_fraction for s in stats.values() if s.usable]
    base = float(np.median(usable_near)) if usable_near else 0.0
    thr = max(NEAR_FRACTION_MIN, NEAR_FRACTION_RATIO * base)
    return {n for n, s in stats.items() if not s.usable and s.near_fraction > thr}


def pooled_median(stats: dict[str, CameraResidual]) -> float | None:
    """有数据相机的中位数的**中位数**(等权)。

    为什么不是把所有样本汇成一堆取中位:某一路样本多 10× 会把总体中位数拖向它 ——
    这里要的是"六路各自准不准",不是"总样本池准不准",故等权。
    """
    vals = [s.median_abs for s in stats.values() if s.usable and s.median_abs is not None]
    return float(np.median(vals)) if vals else None


def hud_line(
    stats: dict[str, CameraResidual],
    pooled: float | None,
    n_plane_points: int,
    n_refit: int = 0,
) -> str:
    """实时槽的 HUD 第二行(**显式报不可用**,不装作"在跑")。

    `pooled is None` ⇒ 全部相机样本都不足,这一帧**没有标定结论** —— 报"无数据"而不是
    报 0.000(0.000 会被读成"标定完美")。自遮挡的相机单独标注,免得被当成坏标定。
    """
    usable = [s for s in stats.values() if s.usable]
    if pooled is None:
        head = "标定 |e| 无数据(全相机样本不足)"
    else:
        head = f"标定 |e|中位 {pooled:.4f} m({len(usable)}/{len(stats)} 相机有数据)"
    ns = "/".join(f"{short_name(n)} {s.n}" for n, s in stats.items())
    meds = "/".join("–" if s.median_abs is None else f"{s.median_abs:.4f}" for s in stats.values())
    line = f"{head} | 平面点 {n_plane_points}(重拟合 {n_refit} 次) | n {ns} | med {meds}"
    occl = self_occluded_cameras(stats)
    if occl:
        line += " | " + " ".join(
            f"{short_name(n)} 自遮挡 {stats[n].near_fraction * 100:.0f}% 样本 {stats[n].n}"
            for n in sorted(occl)
        )
    return line

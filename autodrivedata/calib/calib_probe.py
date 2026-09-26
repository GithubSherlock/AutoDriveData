"""环视相机标定自证的**数值核心**(纯值,不 import carla)。

回答一个问题:**我们的 (K, 外参) 与 CARLA 渲染器是否一致?**

## 为什么不能用"反投影再重投影"

显然的做法(深度图反投影 → 用我们的 K 重投影 → 比像素)是**循环的**:任何 K 都能重现
自己的像素,误差恒为 0。必须有**独立锚**。

## 本模块的判据:LiDAR 平面 → 相机射线求交 vs 渲染深度

对每个相机 c(世界位姿由 `mapviz.cam_pose` 合成):

1. LiDAR 扫描 → 世界系点 → 邻域**局部平面拟合** π(残差超阈丢弃)
2. 用**我们的 K** 从光心发一条过像素 (u,v) 的射线,与 π 求交 → **预测深度** ẑ
3. 读同帧深度相机的 D(u,v) → **残差** e = D − ẑ

**非循环**:π 来自 LiDAR、D 来自渲染器、K 是待检验量,三者来源独立。
**为何要平面拟合**:LiDAR 挂点与各相机挂点不同(最大视差 ~1.5 m),直接用光轴深度比对
会把视差算成标定误差;射线-平面求交**解析地消掉视差**。

## 为什么是"带截距的回归",不是"取 median |e| 的最小值"

残差不是 K 误差本身的函数,而是

    e = g·δ + b + 噪声,     g = ∂D/∂u(局部深度梯度,米/像素),b = 共同深度偏置

- `b` 必须显式建模:它会吸收深度语义残差 / LiDAR 测距偏置等**共模**项。若强行过原点
  拟合(`e = g·δ`),`b` 会**全部漏进 δ**,得到一个假的"主点偏差"。
- `g` 在无梯度处(正前方路面)趋零 ⇒ **零信息**。故可辨识性不是"样本够多"而是
  "**梯度有变化**":`g` 为常数时 δ 与 b 完全共线,方程组奇异。

故用**二元加权最小二乘** `δ* = (Σw g e − ḡ·Σw e / …)`(闭式见 `estimate_axis_delta`),
自带 `σ_δ = σ_e·√(a₁₁/det)`。**`det ≤ 0` 时返回 `identifiable=False` 与 inf**,
不返回数字 —— **"不可辨识"必须是一个可输出的结论**。

## 深度语义(光轴 z vs 射线距离)

`bin/train_3dgs_mini.py:122` 把解码值**直接当相机系 z 用**,但那是**作者假设**不是证据
(3DGS 从错误深度初始化也会收敛)。该语义是一阶效应:图像角落离轴角
`atan(√(620²+186.5²)/621) ≈ 46.2°`,若是射线距离则 `D/z − 1 = sec θ − 1 = 0.444`
(20 m 处 8.9 m)—— 在这种系统误差上做 cx 扫描是纯垃圾。故本模块另给 `depth_model_ratio`
供**独立探针**离线先行裁决(⚠️ 原注释引用的 `bin/probe_depth_semantics.py` 实际不存在
于磁盘,已如实更正;现由 `autodrivedata/calib/probe_calib.py` 的 A3 锚在"z 深度"假定下间接验证)。

## 像素索引约定:它只出现在"采样"这一步

本模块里所有 `uv` 都是**图像坐标**(与 `CameraIntrinsics.cx/cy` 同一个空间);
`cam_rays` / `project_world` 互为逆,**与约定无关**。约定只在把图像坐标换成**数组索引**
时出现,即 `depth_codec.sample_bilinear(..., convention)`:`index = u − shift(convention)`,
`shift = 0.0`(corner,**CARLA 渲染光栅的实测口径**,本模块默认)或 `0.5`
(center,torch 侧 `grid_sample(align_corners=False)` / FPN 特征图 / gsplat 的约定)。

**已裁决(2026-09-22,`autodrivedata/calib/probe_calib.py` A3/A4)**:CARLA 渲染出的图是 **corner** ——
`(cx=620.5, corner)` 与 `(cx=621.0, center)` 描述**同一张光栅**。A3 在 corner 下
median|e| 0.0003 m、center 下 0.023 m(~70×,六相机一致);A4 掩膜**索引**中点回归
给 `cx = 620.50 = (w−1)/2`。故本模块默认 `corner`,探针显式传 `PIXEL_CONVENTION`。

推论:`(cx, shift)` 是一**对**,本模块量到的永远是**和** `cx + shift`。因此结论表述为
"**在该约定下量得的图像坐标主点**",谁与约定打架谁就错 —— 这正是
[autodrivedata/mapviz.py](../../autodrivedata/mapviz.py) `intrinsics_from_k` 的病根
(只抄 fx、把 cx/cy 丢掉重算,等于用一个约定量出的 cx 去配另一个约定)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from autodrivedata import geometry as g
from autodrivedata.calib.core import CameraIntrinsics
from autodrivedata.calib.depth_codec import (
    CONVENTION_CORNER,
    convention_shift,
    sample_bilinear_many,
)

# 平面拟合:最少点数 / 最大残差(m)/ 平面性判据 s3/s2 上限
MIN_PLANE_PTS = 12
MAX_PLANE_RESIDUAL_M = 0.03
MAX_PLANARITY = 0.05
# 深度残差超出此值判为离群(视差边缘 / 遮挡 / 误配),不进回归
MAX_RESIDUAL_M = 1.0
# 可见性(单侧):渲染深度比 LiDAR 预测**更近**超过容差 ⇒ 该点被更近的面挡住 = 相机看不见它。
# 这是**物理判据**(比"邻域极差"这种代理更直接),故恒开、不受 `occlusion_radius_px` 影响。
OCCLUSION_TOL_M = 0.3
OCCLUSION_TOL_FRAC = 0.05
# 窗口极差(可选):邻域深度极差 > 该比例 × 该点深度 ⇒ 判为落在深度断裂边缘、双线性采样不可靠。
OCCLUSION_RANGE_FRAC = 0.15


@dataclass(frozen=True)
class Plane:
    """世界系平面 `n·p = d`(n 单位法向)。"""

    normal: np.ndarray  # (3,)
    offset: float

    def depth_along_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        axis: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> float:
        """射线 `origin + t·direction` 的交点沿 **axis** 方向的深度。

        `axis` 是"深度"的定义方向,**必须显式给**:CARLA 深度是**相机光轴** z 深度,
        故调用方要传相机光轴在世界系的单位向量;默认的 (0,0,1) 只在"世界竖直向下"
        这一特例下才等价 —— 拿世界 z 当深度会**静默给出负值/错值**(相机前下方看地面时
        世界 z 分量是负的)。平行 / 背向 / 交点在身后 → nan。
        """
        d = np.asarray(direction, dtype=np.float64).reshape(3)
        a = np.asarray(axis, dtype=np.float64).reshape(3)
        denom = float(np.dot(self.normal, d))
        if abs(denom) < 1e-9:
            return float("nan")
        t = (
            self.offset - float(np.dot(self.normal, np.asarray(origin, dtype=np.float64).reshape(3)))
        ) / denom
        if t <= 0.0:
            return float("nan")
        return t * float(np.dot(d, a))

    def depth_along_rays(
        self,
        origin: np.ndarray,
        directions: np.ndarray,
        axis: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> np.ndarray:
        """`depth_along_ray` 的批量版 (N,3) → (N,),语义逐点一致。"""
        d = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        a = np.asarray(axis, dtype=np.float64).reshape(3)
        o = np.asarray(origin, dtype=np.float64).reshape(3)
        denom = d @ self.normal
        num = self.offset - float(o @ self.normal)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / denom
        out = t * (d @ a)
        bad = (np.abs(denom) < 1e-9) | (t <= 0.0)
        out[bad] = np.nan
        return out


def fit_plane(points: np.ndarray) -> tuple[Plane, float] | None:
    """点集 (N,3) → 最小二乘平面 + 拟合残差(RMS)。

    点数不足 / 退化(点共线,如地面上的单条激光环)返回 None。质量判据由调用方施加
    (`MAX_PLANE_RESIDUAL_M` / `MAX_PLANARITY`)。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < MIN_PLANE_PTS:
        return None
    centroid = pts.mean(axis=0)
    _, s, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    if s[0] < 1e-9 or s[1] / s[0] < 1e-6:
        return None
    normal = vt[2]
    resid = (pts - centroid) @ normal
    rms = float(np.sqrt(np.mean(resid**2)))
    return Plane(normal, float(np.dot(normal, centroid))), rms


def fit_local_planes(
    points: np.ndarray, radius: float, min_pts: int = MIN_PLANE_PTS
) -> tuple[np.ndarray, np.ndarray]:
    """逐点邻域平面法向 `(N,3)` + 残差 `(N,)`(半径内点数不足 → nan)。

    **逐点自己的平面**才让"点云不是一张平面"时检验依然成立(台阶、斜坡、多目标场景)。
    朴素 O(N²);探针每帧只跑一次,点云经体素降采样后 N ~ 1e4,可接受。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    normals = np.full((n, 3), np.nan)
    rms = np.full(n, np.nan)
    r2 = float(radius) ** 2
    for i in range(n):
        d2 = np.einsum("ij,ij->i", pts - pts[i], pts - pts[i])
        idx = np.nonzero(d2 <= r2)[0]
        if idx.size < min_pts:
            continue
        fit = fit_plane(pts[idx])
        if fit is None:
            continue
        normals[i], rms[i] = fit[0].normal, fit[1]
    return normals, rms


def plane_is_usable(rms: float, sv: np.ndarray | None = None) -> bool:
    """平面质量判据:残差足够小(可选再加奇异值平面性)。

    只看**质量**(残差 + 奇异值比),不看平面本身 —— 故不收 `Plane` 参数。
    """
    if not np.isfinite(rms) or rms > MAX_PLANE_RESIDUAL_M:
        return False
    if sv is not None and sv[0] > 1e-9 and sv[2] / sv[1] > MAX_PLANARITY:
        return False
    return True


# ---------------------------------------------------------------- 投影 / 射线


def cam_rays(intrinsics: CameraIntrinsics, uv: np.ndarray) -> np.ndarray:
    """**图像坐标** (N,2) → 相机系单位射线 (N,3)。

    与 `project_world` 严格互逆,且**与像素约定无关**(约定只影响采样,见模块 docstring)。
    """
    u = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    x = (u[:, 0] - intrinsics.cx) / intrinsics.fx
    y = (u[:, 1] - intrinsics.cy) / intrinsics.fy
    d = np.stack([x, y, np.ones_like(x)], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def cam_to_world_rot(cam_rotation: tuple[float, float, float]) -> np.ndarray:
    """相机世界姿态 (pitch,yaw,roll) **弧度** → 相机系→世界系旋转阵。

    `geometry.camera_rotation_world_to_cam` 是 world→cam,此处取逆(正交 ⇒ 转置)。
    """
    return g.camera_rotation_world_to_cam(cam_rotation).T


def project_world(
    pts_world: np.ndarray,
    cam_location: tuple[float, float, float],
    cam_rotation: tuple[float, float, float],
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """世界点 (N,3) → **图像坐标** (N,2) + 光轴深度 (N,)。**不做过界过滤**。"""
    c = g.world_to_cam(np.asarray(pts_world, dtype=np.float64).reshape(-1, 3), cam_location, cam_rotation)
    z = c[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intrinsics.fx * (c[:, 0] / z) + intrinsics.cx
        v = intrinsics.fy * (c[:, 1] / z) + intrinsics.cy
    return np.stack([u, v], axis=1), z


def backproject_depth(
    uv: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    convention: str = CONVENTION_CORNER,
    cam_location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cam_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """图像坐标 (N,2) → 世界系点 (N,3):在 uv 处采深度、按光轴 z 语义反投影。

    `bin/train_3dgs_mini.py:122` 的同一条链(`cam = ((u−cx)·z/f, (v−cy)·z/f, z)`),
    此处只是把它挪进纯值层并复用 `sample_bilinear_many` 的采样口径。有效值不足的点
    落在 NaN 上(调用方自行过滤)。
    """
    u = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    z = sample_bilinear_many(depth, u, convention)
    cam = np.stack(
        [
            (u[:, 0] - intrinsics.cx) * z / intrinsics.fx,
            (u[:, 1] - intrinsics.cy) * z / intrinsics.fy,
            z,
        ],
        axis=1,
    )
    r_wc = cam_to_world_rot(cam_rotation)
    return cam @ r_wc.T + np.asarray(cam_location, dtype=np.float64).reshape(1, 3)


# ---------------------------------------------------------------- 深度采样 / 残差


@dataclass
class DepthSamples:
    """一次"LiDAR 平面 → 射线求交 vs 渲染深度"的采样结果。"""

    uv: np.ndarray  # (N,2) 图像坐标
    z_lidar: np.ndarray  # (N,) 射线-平面求交的**预测光轴深度**
    z_render: np.ndarray  # (N,) 渲染深度相机读数(双线性)
    grad: np.ndarray  # (N,2) 渲染深度的局部梯度 [∂D/∂u, ∂D/∂v](米/像素)

    @property
    def residual(self) -> np.ndarray:
        return self.z_render - self.z_lidar

    def __len__(self) -> int:
        return int(self.uv.shape[0])


def local_grad(
    depth: np.ndarray,
    uv: np.ndarray,
    convention: str = CONVENTION_CORNER,
    h: float = 0.5,
) -> np.ndarray:
    """渲染深度在图像坐标 (N,2) 处的梯度 `(N,2)` [米/像素](中心差分,基线 2h 像素)。

    `h = 0.5` ⇒ 基线 1 像素,与"1 像素位移"的尺度一致(δ 的单位就是像素)。
    """
    u = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if u.size == 0:
        return np.zeros((0, 2))
    du = np.empty((u.shape[0], 2))
    for k in range(2):
        lo = u.copy()
        hi = u.copy()
        lo[:, k] -= h
        hi[:, k] += h
        du[:, k] = (
            sample_bilinear_many(depth, hi, convention) - sample_bilinear_many(depth, lo, convention)
        ) / (2.0 * h)
    return du


def local_range(depth: np.ndarray, radius: float) -> np.ndarray:
    """方形窗口 `(2r+1)²` 内的深度极差 max−min(H,W)。

    **可分离**:方形窗口的 max(以及 min)可以分解成"沿行 → 沿列"两次一维窗口,
    代价从 O(r²) 降到 O(r)。r 由视差推出时可达数十像素(1.5 m 基线 / 20 m 处
    f=621 → 46 px),O(r²) 会直接卡死。
    """
    a = np.asarray(depth, dtype=np.float64)
    r = int(math.ceil(float(radius)))
    if r <= 0:
        return np.zeros_like(a)

    def _pass(src: np.ndarray, axis: int, op, init: float) -> np.ndarray:
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(src, pad, mode="edge")
        out = np.full(src.shape, init, dtype=np.float64)
        for k in range(2 * r + 1):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(k, k + src.shape[axis])
            op(out, p[tuple(sl)], out=out)
        return out

    hi = _pass(_pass(a, 1, np.maximum, -np.inf), 0, np.maximum, -np.inf)
    lo = _pass(_pass(a, 1, np.minimum, np.inf), 0, np.minimum, np.inf)
    return hi - lo


def collect_samples(
    pts_world: np.ndarray,
    plane_normals: np.ndarray,
    plane_offsets: np.ndarray,
    cam_location: tuple[float, float, float],
    cam_rotation: tuple[float, float, float],
    intrinsics: CameraIntrinsics,
    depth: np.ndarray,
    convention: str = CONVENTION_CORNER,
    occlusion_radius_px: float | None = None,
) -> DepthSamples:
    """LiDAR 点 + 逐点局部平面 → 深度采样(核心检验)。

    逐点用**该点自己的局部平面**求交(点云非平面时仍成立)。仅保留:投影落在图内、
    深度有效、射线与平面在身前相交、残差未超离群阈,以及**未被更近的面挡住**
    (`OCCLUSION_TOL_M` / `OCCLUSION_TOL_FRAC`,恒开);若再给 `occlusion_radius_px`,
    额外要求邻域深度极差 ≤ `OCCLUSION_RANGE_FRAC × z`(落在深度断裂边缘、双线性采样不可靠)。

    **可见性判据是单侧的,不是"邻域极差"**:物理上"相机看不见该点" ⟺ 同一像素上渲染深度
    **比预测更近**。用对称窗口极差当代理会把大量合法样本误杀 —— 实测(2026-09-22)窗口半径
    r=16px 时样本塌 95%(16.1/18.9/19.4/36.4 px 的半径下只剩 4/4/8/0 个),而单侧判据在
    r=0 就等价生效、样本数与无过滤一致。窗口极差只在 r ≤ 6 时才近似可用,故降级为可选。
    半径是**标量**,由调用方从外参推(基线 / 最近距离);逐点半径会让每点一次不同尺寸的
    邻域查表,代价与收益都不成比例。
    """
    pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    uv, z_axis = project_world(pts, cam_location, cam_rotation, intrinsics)
    h, w = depth.shape[:2]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & (z_axis > 0.5)
    keep = np.nonzero(inside)[0]
    if keep.size == 0:
        return DepthSamples(np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros((0, 2)))

    r_wc = cam_to_world_rot(cam_rotation)
    origin = np.asarray(cam_location, dtype=np.float64)
    uv_keep = uv[keep]
    rays_world = cam_rays(intrinsics, uv_keep) @ r_wc.T
    # 深度 = **相机光轴**方向的投影(与 CARLA 深度编码同语义);光轴 = 相机系 (0,0,1)
    optical_axis = r_wc[:, 2]

    # 逐点自己的平面求交(法向已单位化 ⇒ t 即沿射线距离)
    normals = np.asarray(plane_normals, dtype=np.float64).reshape(-1, 3)[keep]
    offsets = np.asarray(plane_offsets, dtype=np.float64).reshape(-1)[keep]
    denom = np.einsum("ij,ij->i", rays_world, normals)
    num = offsets - origin @ normals.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = num / denom
    z_pred = t * (rays_world @ optical_axis)
    z_rend = sample_bilinear_many(depth, uv_keep, convention)
    grad = local_grad(depth, uv_keep, convention)

    ok = (
        np.isfinite(z_pred)
        & np.isfinite(z_rend)
        & (t > 0.0)
        & (np.abs(denom) > 1e-9)
        & (z_pred > 0.5)
        & (np.abs(z_rend - z_pred) <= MAX_RESIDUAL_M)
        & np.isfinite(grad).all(axis=1)
    )
    # 可见性(单侧):渲染比预测更近超过容差 ⇒ 被更近的面挡住,相机看不见该点。
    # 容差取 `max(绝对, 相对)` —— 近处用绝对项(0.3 m 已远超 0.02 m 量级的残差),
    # 远处用相对项(深度误差随距离线性放大)。
    tol = np.maximum(OCCLUSION_TOL_M, OCCLUSION_TOL_FRAC * z_pred)
    ok &= z_rend >= z_pred - tol
    if occlusion_radius_px is not None and occlusion_radius_px > 0:
        rng_map = local_range(depth, occlusion_radius_px)
        # 窗口查表用的是**索引**,须减回约定平移量(0.5 只由 `convention_shift` 给),
        # 不能直接 `floor(u)` —— 那会把 center 约定下的采样点错位到隔壁像素。
        shift = convention_shift(convention)
        ri = np.floor(np.clip(uv_keep[:, 1] - shift, 0, h - 1)).astype(np.intp)
        ci = np.floor(np.clip(uv_keep[:, 0] - shift, 0, w - 1)).astype(np.intp)
        ok &= rng_map[ri, ci] <= OCCLUSION_RANGE_FRAC * z_rend
    sel = np.nonzero(ok)[0]
    return DepthSamples(uv_keep[sel], z_pred[sel], z_rend[sel], grad[sel])


# ---------------------------------------------------------------- 主点估计


@dataclass(frozen=True)
class AxisFit:
    """单轴(主点偏差 δ / 共模偏置 b)的加权最小二乘解。"""

    delta_px: float
    intercept_m: float
    sigma_delta_px: float
    sigma_intercept_m: float
    n: int
    det: float  # 正规方程行列式;≈0 ⇒ δ 与 b 共线 = 不可辨识
    axis: str

    @property
    def identifiable(self) -> bool:
        """`det > 0` 且解有限 ⇒ 该轴可辨识。

        `det = (Σw g²)(Σw) − (Σw g)²` = `Σw·Σw·Var(g)`:梯度无变化 ⇒ det = 0 ⇒
        δ 与共模偏置完全共线,无论多少样本都定不出 δ。
        """
        return math.isfinite(self.delta_px) and self.det > 0.0


def mad_sigma(x: np.ndarray) -> float:
    """稳健标准差估计 `1.4826·MAD`(抗离群;残差分布未知时优于 std)。"""
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 0.0
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def estimate_axis_delta(samples: DepthSamples, axis: int = 0, sigma_e: float | None = None) -> AxisFit:
    """主点单轴偏差 δ 的加权最小二乘估计(**带截距**)。

    模型 `e = g·δ + b`,法方程

        [Σw g²   Σw g] [δ]   [Σw g e]
        [Σw g    Σw  ] [b] = [Σw e  ]

    两次迭代:先用 `mad_sigma(e)` 定权,解出后再用**拟合残差**的 MAD 重定权与重解,
    避免信号 `g·δ` 自身把 `σ_e` 抬高而虚报精度(常见于 δ 达 0.5 px 时)。

    `det ≤ 0`(梯度无变化)时 `delta_px = nan` 且 `sigma = inf`;调用方读
    `identifiable` 判断 —— **不返回一个硬凑的数**。
    """
    name = ("u", "v")[axis]
    e = samples.residual
    gg = samples.grad[:, axis]
    good = np.isfinite(e) & np.isfinite(gg)
    e, gg = e[good], gg[good]
    n = int(e.size)
    if n < 3:
        return AxisFit(float("nan"), float("nan"), float("inf"), float("inf"), n, 0.0, name)

    w = np.ones(n)
    if sigma_e is not None and sigma_e > 0.0:
        w = np.full(n, 1.0 / sigma_e**2)
    fit = _solve_wls(e, gg, w, name)
    for _ in range(2):  # 用拟合残差重定权(IRLS 的稳健化)
        resid = e - fit.delta_px * gg - fit.intercept_m
        s = mad_sigma(resid)
        if s <= 0.0:
            break
        fit = _solve_wls(e, gg, np.full(n, 1.0 / s**2), name)
    return fit


def _solve_wls(e: np.ndarray, gg: np.ndarray, w: np.ndarray, name: str) -> AxisFit:
    a11 = float(np.sum(w * gg * gg))
    a12 = float(np.sum(w * gg))
    a22 = float(np.sum(w))
    c1 = float(np.sum(w * gg * e))
    c2 = float(np.sum(w * e))
    det = a11 * a22 - a12 * a12
    if not (det > 1e-12) or a22 <= 0.0:
        return AxisFit(float("nan"), float("nan"), float("inf"), float("inf"), int(e.size), det, name)
    delta = (a22 * c1 - a12 * c2) / det
    b = (a11 * c2 - a12 * c1) / det
    sigma_e2 = float(np.sum(w * (e - delta * gg - b) ** 2)) / max(1, int(e.size) - 2)
    sigma_d = math.sqrt(max(0.0, sigma_e2 * a22 / det))
    sigma_b = math.sqrt(max(0.0, sigma_e2 * a11 / det))
    return AxisFit(delta, b, sigma_d, sigma_b, int(e.size), det, name)


def estimate_delta_uv(samples: DepthSamples, sigma_e: float | None = None) -> tuple[AxisFit, AxisFit]:
    """两轴一次给出 `(δu, δv)`(各带截距,互不耦合:法方程按轴独立)。"""
    return estimate_axis_delta(samples, 0, sigma_e), estimate_axis_delta(samples, 1, sigma_e)


def sweep_profile(samples: DepthSamples, axis: int, deltas: np.ndarray) -> np.ndarray:
    """对每个候选 δ 求**剖面似然的 median|残差|** —— 人眼可读的那条 V 形曲线。

    δ 固定时 `b` 的最优解是 `b*(δ) = ē − δ·ḡ`,故残差有闭式
    `r_i(δ) = (e_i − ē) − δ·(g_i − ḡ)`,无需循环里再解一次最小二乘。
    曲线谷底应与 `estimate_axis_delta` 的 `delta_px` 吻合(互为佐证)。
    """
    e = samples.residual
    gg = samples.grad[:, axis]
    good = np.isfinite(e) & np.isfinite(gg)
    e, gg = e[good], gg[good]
    if e.size == 0:
        return np.full(np.asarray(deltas).shape, np.nan)
    ec, gc = e - e.mean(), gg - gg.mean()
    return np.array(
        [float(np.median(np.abs(ec - d * gc))) for d in np.asarray(deltas, dtype=np.float64)],
        dtype=np.float64,
    )


# ---------------------------------------------------------------- 深度语义 / 畸变


def depth_model_ratio(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """离轴角 θ[弧度] → (光轴 z 模型, 射线距离模型) 的 `D/z_axis` 理论值。

    光轴 z: 恒为 1.0;射线距离: sec θ(θ=46.2° 时 **1.444**,即 20 m 处差 8.9 m)。
    """
    t = np.asarray(theta, dtype=np.float64)
    return np.ones_like(t), 1.0 / np.cos(t)


def max_off_axis_angle(intrinsics: CameraIntrinsics) -> float:
    """图像角落的离轴角[弧度] —— 深度语义混淆的**上界**。"""
    x = (intrinsics.width - intrinsics.cx) / intrinsics.fx
    y = (intrinsics.height - intrinsics.cy) / intrinsics.fy
    return float(math.atan(math.hypot(x, y)))


def off_axis_angle(uv: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """图像坐标 (N,2) → 离轴角 θ[弧度]。"""
    u = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    x = (u[:, 0] - intrinsics.cx) / intrinsics.fx
    y = (u[:, 1] - intrinsics.cy) / intrinsics.fy
    return np.arctan(np.hypot(x, y))


def radial_error_profile(
    samples: DepthSamples, intrinsics: CameraIntrinsics, n_bins: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """残差按"距主点像素距离"分箱 → (bin 中心, 中位绝对残差, 计数, 线性斜率)。

    理想针孔 + 正确 K ⇒ 斜率 ≈ 0;存在径向畸变 ⇒ 残差随半径系统变化。
    """
    u = samples.uv
    if len(samples) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int), float("nan")
    r = np.hypot(u[:, 0] - intrinsics.cx, u[:, 1] - intrinsics.cy)
    edges = np.linspace(0.0, float(np.percentile(r, 99)), n_bins + 1)
    centers, meds, counts = [], [], []
    for i in range(n_bins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if not m.any():
            continue
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        meds.append(float(np.median(np.abs(samples.residual[m]))))
        counts.append(int(m.sum()))
    c = np.asarray(centers)
    md = np.asarray(meds)
    slope = float(np.polyfit(c, md, 1)[0]) if c.size >= 2 else float("nan")
    return c, md, np.asarray(counts, dtype=int), slope


# ---------------------------------------------------------------- 轴目标物(主点的直接读数)


def mask_row_midpoints(mask: np.ndarray, trim: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """二值掩膜 → (行号, 该行左右边界的**中点列**)。中点以**索引**表示。

    为什么不用整体质心:阈值质心会被内部明暗梯度整体推移(左亮右暗的锥体推移整像素),
    而"逐行左右边界取中"把**对称**的抗锯齿/边缘偏置一阶抵消。

    `trim` = 上下各裁掉的比例(锥体顶部 1–2 px 不稳、底部椭圆且受光不对称)。
    """
    m = np.asarray(mask, dtype=bool)
    rows = np.nonzero(m.any(axis=1))[0]
    if rows.size == 0:
        return np.zeros(0), np.zeros(0)
    lo = rows.min() + int(trim * rows.size)
    hi = rows.max() - int(trim * rows.size)
    sel = rows[(rows >= lo) & (rows <= hi)]
    mids = np.empty(sel.size, dtype=np.float64)
    for k, r in enumerate(sel):
        cols = np.nonzero(m[r])[0]
        mids[k] = 0.5 * (float(cols.min()) + float(cols.max()))
    return sel.astype(np.float64), mids


def midline_deviation(mids: np.ndarray) -> float:
    """逐行中点相对**中位数**的最大偏离(像素)。

    均匀的亮度梯度把整条中线整体推移(那是我们想保留的**常数**,由回归截距吸收),
    故必须相对中位数取偏离:非零即表示**逐行不同**的左右不对称(自证目标物不达标)。
    """
    m = np.asarray(mids, dtype=np.float64)
    m = m[np.isfinite(m)]
    if m.size < 2:
        return float("nan")
    return float(np.max(np.abs(m - np.median(m))))


def mirror_asymmetry(mask: np.ndarray, axis_col: float, trim: float = 0.2) -> float:
    """以**给定轴列**为镜像面,掩膜左右半宽的差的最大值(像素)。

    这是"目标物够不够对称"的判据:**必须固定候选轴**再量左右宽度。若改用逐行中点当
    镜像面,左右宽度按定义恒等,量出来永远 0 —— 那是自证陷阱。`axis_col` 用估计出的
    主点(索引口径),对称物左右半宽应当相等。
    """
    m = np.asarray(mask, dtype=bool)
    rows = np.nonzero(m.any(axis=1))[0]
    if rows.size == 0:
        return float("nan")
    lo = rows.min() + int(trim * rows.size)
    hi = rows.max() - int(trim * rows.size)
    sel = rows[(rows >= lo) & (rows <= hi)]
    worst = 0.0
    for r in sel:
        cols = np.nonzero(m[r])[0]
        worst = max(worst, abs(float(axis_col - cols.min()) - float(cols.max() - axis_col)))
    return float(worst)


def prop_axis_regression(
    offsets_x: np.ndarray, centres_u: np.ndarray, index_shift: float
) -> tuple[float, float, np.ndarray]:
    """多个已知 x 偏移的对称目标物 → (cx_图像坐标, fx/z, 残差)。

    目标物几何中心在世界系沿**相机横轴**已知偏移 `x_prop`,渲染出的掩膜中点列
    (索引口径)满足 `u_index = (f/z)·x_prop + (cx − index_shift)`。故:

    - **截距**给出该渲染器的主点 x(还原成图像坐标口径,可直接与 K 的 cx 比)
    - **斜率**独立校核 `fx/z`(与 `calib.CameraIntrinsics.fx` 对账)
    - **残差**暴露目标物被物理引擎挪动 / 掩膜不稳(超阈即判该法不可靠,不硬给结论)

    `index_shift` 必须由调用方从 `depth_codec.convention_shift` 取,**不写死 0.5**。
    """
    x = np.asarray(offsets_x, dtype=np.float64).reshape(-1)
    u = np.asarray(centres_u, dtype=np.float64).reshape(-1)
    if x.size < 2 or x.size != u.size:
        raise ValueError(f"至少 2 组且长度一致,收到 {x.size}/{u.size}")
    a = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(a, u, rcond=None)
    resid = u - (slope * x + intercept)
    return float(intercept + index_shift), float(slope), resid

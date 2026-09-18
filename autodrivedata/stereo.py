"""P-F 教程 15 双目立体视觉(纯值,零 carla / 零 opencv)。

工业口径:双目 = 两个水平基线相机,通过**视差**恢复每像素深度。
本模块提供完整双目链路(核心全部 numpy 实现,SGM 用 cv2 可选):
- `triangulate_depth` / `depth_from_disparity`:标准前向平行双目三角测量 z = f·B/d
- `disparity_sgm` / `disparity_ncc`:视差估计(OpenCV SGBM 可选 / 纯 numpy NCC)
- `depth_to_pointcloud`:深度图 + 内参 → 相机系点云
- `normalize_disparity` / `reprojection_loss`:自监督双目深度损失

坐标系与卡口:
- 输入左右图 (H,W) 单通道 float32(灰度),或 (H,W,3) 将内部转灰度
- 返回 depth (H,W) float32,单位米;0 = 无效(视差不可信/遮挡)
- fx=fy 方形像素,CameraIntrinsics 复用 calib.py(不 import carla)

单测手算锚点:纯平面 z=const 的视差应等于 f·B/z;左移右移符号自洽。

实现细节(2026-09-18 修正):
- SGBM 输入必须 CV_8U(灰度换 uint8),单位真延迟 1/16
- NCC 块相关形状要逐块 padding(滑动窗边缘丢弃),不能跨 d 广播
- 重投影方向:右图按视差**右移**采样:左 = 右(x+d),即重建左图
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # opencv 可选:本模块核心三角测量/重建不依赖它,仅 SGM 需要
    import cv2

    _CV2_OK = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_OK = False


def _require_cv2() -> None:
    if not _CV2_OK:
        raise RuntimeError("需要 opencv(cv2)")


def triangulate_depth(disparity: np.ndarray, baseline_m: float, focal_px: float) -> np.ndarray:
    """前向平行双目三角测量:z = f·B / d。

    disparity>0 处深度 = f·B/d;d<=0(无效/无穷远)深度 0。
    """
    d = np.asarray(disparity, dtype=np.float64)
    out = np.zeros_like(d)
    ok = d > 1e-6
    out[ok] = focal_px * baseline_m / d[ok]
    return out.astype(np.float32)


def depth_from_disparity(disparity: np.ndarray, baseline_m: float, focal_px: float) -> np.ndarray:
    """同 triangulate_depth(别名,语义更直白)。"""
    return triangulate_depth(disparity, baseline_m, focal_px)


def _as_gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    return a


def disparity_sgm(
    left: np.ndarray,
    right: np.ndarray,
    *,
    min_disp: int = 0,
    num_disp: int = 64,
    block_size: int = 5,
) -> np.ndarray:
    """OpenCV SGBM 视差(灰度,float32,单位像素;无效处 < min)。

    输入自动转 uint8(SGBM 只收 CV_8U)。16 位真延迟 → 除以 16.0。
    未安装 opencv 时抛 RuntimeError(本模块其余函数不依赖)。
    """
    if cv2 is None:
        raise RuntimeError("disparity_sgm 需要 opencv;其余双目函数纯 numpy 可用")
    _require_cv2()
    L8 = np.clip(_as_gray(left), 0, 255).astype(np.uint8)
    R8 = np.clip(_as_gray(right), 0, 255).astype(np.uint8)
    sgm = cv2.StereoSGBM_create(  # type: ignore[attr-defined]
        minDisparity=min_disp,
        numDisparities=max(16, int(num_disp) // 16 * 16),
        blockSize=max(3, int(block_size) | 1),
        P1=8 * 3 * int(block_size) ** 2,
        P2=32 * 3 * int(block_size) ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    return sgm.compute(L8, R8).astype(np.float32) / 16.0


def _ncc_block(a: np.ndarray, b: np.ndarray, block: int) -> np.ndarray:
    """滑动窗块间归一化互相关;返回滑动窗形状 (H-k+1, W-k+1)。"""
    from numpy.lib.stride_tricks import sliding_window_view

    sw = sliding_window_view(a, (block, block))  # (H-k+1, W-k+1, k, k)
    sw2 = sliding_window_view(b, (block, block))

    def stat(x):
        return x.mean(axis=(-2, -1))

    ma, mb = stat(sw), stat(sw2)
    ac, bc = sw - ma[..., None, None], sw2 - mb[..., None, None]
    num = (ac * bc).mean(axis=(-2, -1))
    den = np.sqrt((ac * ac).mean(axis=(-2, -1)) * (bc * bc).mean(axis=(-2, -1)) + 1e-6)
    return num / den


def disparity_ncc(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_disp: int = 64,
    block: int = 7,
) -> np.ndarray:
    """自研 NCC 局部立体匹配(纯 numpy,无 opencv 依赖)。

    对每个像素在 [1, max_disp] 内搜索,以块 NCC 峰值选视差。
    输出 float32 视差(单位像素);找不到有效块的像素 0。

    实现:同名点右图更左 d → 左图列 x 处的块 = 右图列 x-d 处的块。
    为块对齐,把右图**右移 d** 构造 Rshift(x)=R(x-d),再按同列做块相关。
    (早期版本误把右图"左移 d"当匹配,同名差 2d → 视差系统错,SGBM 正常。
    仅作 SGM 的纯 numpy 参照(教学/无 cv2 环境),精度低于 SGBM。)
    """
    L = _as_gray(left)
    R = _as_gray(right)
    h, w = L.shape
    best = np.full((h, w), -np.inf, dtype=np.float32)
    best_d = np.zeros((h, w), dtype=np.float32)
    for d in range(1, max_disp + 1):
        if d >= w:
            break
        # 右图右移 d → Rshift 列 j = 右图列 j-d → 与左图列 j 同列对齐
        rshift = np.zeros_like(R)
        rshift[:, d:] = R[:, :-d]
        corr = _ncc_block(L, rshift, block)  # (H-k+1, W-k+1)
        cand = np.full((h, w), -np.inf, dtype=np.float32)
        hh, ww = corr.shape
        cand[:hh, :ww] = corr
        better = cand > best
        best = np.where(better, cand, best)
        best_d = np.where(better, d, best_d)
    return best_d


def depth_to_pointcloud(
    depth: np.ndarray, k: Any, *, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """深度图 + 内参 → 相机系点云 (N,3) 与有效掩码 (H,W)。

    k 需带 .fx/.fy/.cx/.cy(calib.CameraIntrinsics);z>0 像素反投影,
    u = (x·fx/z)+cx。返回去无效点的点云 + (H,W) bool 掩码。
    """
    d = np.asarray(depth, dtype=np.float32)
    h, w = d.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    ok = d > 0
    if mask is not None:
        ok &= mask.astype(bool)
    if not ok.any():
        return np.zeros((0, 3), dtype=np.float32), ok
    X = (u[ok] - k.cx) * d[ok] / k.fx
    Y = (v[ok] - k.cy) * d[ok] / k.fy
    Z = d[ok]
    pts = np.column_stack([X, Y, Z]).astype(np.float32)
    return pts, ok


def normalize_disparity(disp: np.ndarray, q: float = 99.0) -> np.ndarray:
    """视差图归一化到 [0,1](自监督光度损失前处理;q 分位裁剪离群)。"""
    d = np.asarray(disp, dtype=np.float32)
    pos = d[d > 0]
    if pos.size == 0:
        return np.zeros_like(d)
    hi = float(np.percentile(pos, q)) + 1e-6
    return np.clip(d / hi, 0.0, 1.0)


def reprojection_loss(
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    *,
    eps: float = 1e-3,
) -> float:
    """自监督双目深度损失:右图按视差重建左图,与真实左图 L1 差。

    双目几何:左 = 右(d 右移)。重建左(u) = 右(u + d);只对视差>0 计。
    """
    L = _as_gray(left)
    R = _as_gray(right)
    d = np.asarray(disparity, dtype=np.float32)
    h, w = L.shape
    xs = np.clip(np.arange(w)[None, :] + np.round(d), 0, w - 1).astype(np.int32)
    recon = R[np.arange(h)[:, None], xs]  # 右图采样出左图
    ok = d > 0
    if not ok.any():
        return 0.0
    err = np.abs(recon - L)[ok]
    return float(np.clip(err - eps, 0, None).mean())

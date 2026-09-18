"""autodrivedata/stereo.py 单测(P-F 双目手算锚点 + 边界)。

锚点:
- 齐次平面 z=10m,f=620,B=0.4m → 视差 ≈ f·B/z = 24.8px → 三角测量回 z=10m
- 右图左移 = 同名点右移? 双目几何:左相机的 x 比右相机 x 大 f·B/z。
  z=10 平面视差 d=24.8px,di、三角测量应回 10m。
- SGBM 在合成(含纹理)图像上恢复近似平面深度。
"""

from __future__ import annotations

import numpy as np

from autodrivedata.calib import CameraIntrinsics
from autodrivedata.stereo import (
    depth_from_disparity,
    depth_to_pointcloud,
    normalize_disparity,
    reprojection_loss,
    triangulate_depth,
)


def _plane_pair(baseline: float = 0.4, f: float = 620.0, z: float = 10.0, size: int = 256):
    """生成一对前向平行(rectified)平面图:视差 d = f·B/z 常量。

    纹理:满幅 [0,255] 平滑随机(scipy 高斯低通)——SGBM 对低对比/周期纹理
    会失效(实测:低对比全 -1,周期棋盘掉周期混淆)。右图 = 左图左移 d。
    双目约定:同名点在左图 x = 右图 x + d(右相机更靠右)。
    """
    from scipy.ndimage import gaussian_filter

    d = f * baseline / z
    rng = np.random.default_rng(7)
    base = rng.standard_normal((size, size))
    tex = gaussian_filter(base, sigma=2.0)
    tex = (tex - tex.min()) / tex.ptp() * 255.0
    left = tex.astype(np.float32)
    shift = int(round(d))
    right = np.zeros_like(left)
    if shift > 0:
        right[:, :-shift] = left[:, shift:]  # 右图 = 左图左移 shift
    else:
        right[:, :] = left
    return left, right, float(d)


class TestTriangulate:
    def test_constant_plane(self):
        L, R, d_true = _plane_pair()
        disp = np.full(L.shape, d_true, dtype=np.float32)
        depth = triangulate_depth(disp, baseline_m=0.4, focal_px=620.0)
        # 视差=常量 → 深度=常量且尽为 z
        assert np.allclose(depth[depth > 0], 10.0, atol=0.5)
        assert depth.shape == L.shape

    def test_invalid_disp_zero_depth(self):
        d = np.zeros((4, 4), dtype=np.float32)
        assert (triangulate_depth(d, 0.4, 620.0) == 0).all()

    def test_depth_inverse_relation(self):
        # 视差越大越近
        d1 = triangulate_depth(np.array([[5.0]]), 0.4, 620.0)[0, 0]
        d2 = triangulate_depth(np.array([[20.0]]), 0.4, 620.0)[0, 0]
        assert d1 > d2 > 0
        assert abs(d2 - 620 * 0.4 / 20.0) < 1e-3

    def test_depth_from_disparity_alias(self):
        d = np.full((2, 2), 24.0, dtype=np.float32)
        assert np.allclose(depth_from_disparity(d, 0.4, 620.0), triangulate_depth(d, 0.4, 620.0))


class TestDisparity:
    def test_sgm_recovers_plane(self):
        try:
            from autodrivedata.stereo import disparity_sgm
        except Exception:
            return  # 无 opencv 跳过
        L, R, d_true = _plane_pair(size=128)
        # 给点噪声,否则左右纯平棋盘中值糊边
        rng = np.random.default_rng(0)
        L = L + 0.05 * rng.standard_normal(L.shape)
        R = R + 0.05 * rng.standard_normal(R.shape)
        disp = disparity_sgm(L, R, num_disp=64, block_size=7)
        # 有效区:x∈[shift, W-shift](右图左移 shift 的边界 0 不可用)
        shift = int(round(d_true))
        valid = disp[50:80, shift + 2 : -2]
        med = float(np.median(valid))
        # SGM 输出像素分辨率,允许 ±3px
        assert abs(med - d_true) <= 3.0

    def test_ncc_ok(self):
        from autodrivedata.stereo import disparity_ncc

        L, R, d_true = _plane_pair(size=96, z=8.0)  # 更近 → 更大视差(31px)
        disp = disparity_ncc(L, R, max_disp=64, block=7)
        shift = int(round(d_true))
        valid = disp[30:66, shift + 2 : -2]
        med = float(np.median(valid))
        assert abs(med - d_true) <= 3.0


class TestPointcloud:
    def test_synthetic_depth(self):
        h, w = 10, 12
        d = np.full((h, w), 8.0, dtype=np.float32)
        k = CameraIntrinsics(width=w, height=h, fov_h_deg=90)
        pts, mask = depth_to_pointcloud(d, k)
        assert pts.shape[0] == h * w and mask.all()
        # 中心像素 z 应 ≈ 8
        mid = np.argmin(np.abs(pts[:, 2] - 8.0))
        assert abs(pts[mid, 2] - 8.0) < 0.01


class TestSelfSupervised:
    def test_normalize_range(self):
        d = np.array([[0, 12, 24, 0], [4, 0, 8, 16]], dtype=np.float32)
        n = normalize_disparity(d)
        assert n.min() >= 0 and n.max() <= 1.0

    def test_reprojection_zero_for_valid(self):
        # 视差 0 且右=左一致 → 损失 0
        L = np.zeros((8, 8), dtype=np.float32)
        R = np.zeros((8, 8), dtype=np.float32)
        d = np.zeros((8, 8), dtype=np.float32)
        assert reprojection_loss(L, R, d) == 0.0

    def test_reprojection_small_with_correct_disp(self):
        # 合成:右图 = 左图平移 d,重投影损失应小;反向 0 应变大
        L, R, d_true = _plane_pair(size=128)
        d_ok = np.full(L.shape, d_true, dtype=np.float32)
        loss_ok = reprojection_loss(L, R, d_ok)
        loss_bad = reprojection_loss(L, R, np.full(L.shape, d_true * 2.0, dtype=np.float32))
        assert loss_ok < loss_bad

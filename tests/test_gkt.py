"""GKT 单测:torch 投影链 vs B3 已验证的 numpy 链(calib.world_to_img)逐点对账。

对账口径(§5.11 C 阶段正确性锚点):投影数学已被 B3 验收(路面率 94.7-100%),
本测试把 torch 实现锁死在同一坐标链上——旋转阵逐元素、像素坐标逐点、BEV
可见性掩码逐单元(仅允许图像边缘 0.01px 内的浮点边界分歧)。

⚠️ **2026-09-22 教训**:上面那句"B3 已验证"当时是错的。B3 探针走的是 numpy 链
(`probe_mapvec_proj.py`,正确顺序 + 全分辨率 K),而本文件的 oracle **复刻了 torch
实现的同一个换序错误**,且拿 375/1242 当特征尺寸 ⇒ 两条 bug 全被单测锁死,
`valid` 覆盖 1.25% 却一路绿灯。现在:
- oracle 的位姿六元组按 `[x,y,z,yaw,pitch,roll]` 解(与 infos 同序);
- 新增 `test_pose_rotation_order_is_carla_convention` 直接钉换序;
- 新增 `test_gkt_scales_k_to_feature_resolution` 钉 K 缩放;
- `test_gkt_valid_coverage_on_real_rig` 用真实 rig 的期望覆盖(~94%)兜底。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from autodrivedata import geometry as g
from autodrivedata.calib import CameraIntrinsics, world_to_img
from autodrivedata.camera_rig import NUS_CAMERA_RIG
from maptr_impl.gkt import (
    _ROT_TO_CARLA,
    GKT,
    BEVParams,
    _carla_rotation_torch,
    cam_world_pose,
    project_pts,
    scale_k,
)

CAM = CameraIntrinsics(width=1242, height=375, fov_h_deg=90.0)  # B1 CAM_ATTRS 口径
K_NP = np.array([[CAM.fx, 0.0, CAM.cx], [0.0, CAM.fy, CAM.cy], [0.0, 0.0, 1.0]])
K_T = torch.tensor(K_NP, dtype=torch.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float32)


def _matrix_to_carla_rot(m: np.ndarray) -> tuple[float, float, float]:
    """Rz(yaw)·Ry(pitch)·Rx(roll) 矩阵 → (pitch, yaw, roll)(|pitch| < π/2 唯一)。"""
    pitch = math.asin(m[2, 0])
    yaw = math.atan2(m[1, 0], m[0, 0])
    roll = math.atan2(-m[2, 1], m[2, 2])
    return float(pitch), float(yaw), float(roll)


def _rot_np(deg6: np.ndarray) -> np.ndarray:
    """infos 六元组尾三 [yaw, pitch, roll](度) → CARLA 旋转阵(正确顺序)。"""
    return g.carla_rotation_matrix((math.radians(deg6[4]), math.radians(deg6[3]), math.radians(deg6[5])))


def _cam_world_np(ego: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    """numpy 参考:相机世界位置 + 合成 CARLA 旋转元组(与 torch cam_world_pose 同式)。"""
    r_e = _rot_np(ego)
    r_s = _rot_np(se)
    t_cam = ego[:3] + r_e @ se[:3]
    return t_cam, _matrix_to_carla_rot(r_e @ r_s)


def _oracle_uv(pts_ego: np.ndarray, ego: np.ndarray, se: np.ndarray) -> list[tuple[float, float] | None]:
    """B3 链 oracle:ego 局部系点 → world_to_img(合成位姿)。"""
    t_cam, cam_rot = _cam_world_np(ego, se)
    r_e = _rot_np(ego)
    pts_world = pts_ego @ r_e.T + ego[:3]
    return [world_to_img(tuple(p), tuple(t_cam), cam_rot, CAM) for p in pts_world]


def test_rotation_matrix_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    rot = rng.uniform(-np.pi, np.pi, (8, 3))
    torch_mat = _carla_rotation_torch(_t(rot))
    np_mat = np.stack([g.carla_rotation_matrix(tuple(r)) for r in rot])
    np.testing.assert_allclose(torch_mat.numpy(), np_mat, atol=1e-6)


def test_pose_rotation_order_is_carla_convention() -> None:
    """换序锚点:`[.., yaw, pitch, roll]` 必须被当成 CARLA 的 (pitch, yaw, roll)。

    反例判据(修前的实际行为):yaw 被塞进 pitch 槽 ⇒ `r_e` 与参考阵最大差 0.5,
    且纯 yaw 的 ego 会得到 `R[2,0] = sin(yaw) ≠ 0`(正确的纯 yaw 阵 R[2,0] = 0)。
    """
    assert _ROT_TO_CARLA == (1, 0, 2)
    ego = np.array([0.0, 0.0, 0.0, 30.0, 0.0, 0.0])  # 纯 yaw=30°
    _, r_e, _ = cam_world_pose(_t(ego), _t(np.zeros(6)))
    ref = _rot_np(ego)
    np.testing.assert_allclose(r_e.numpy(), ref, atol=1e-6)
    assert abs(float(r_e[2, 0])) < 1e-6, "纯 yaw 的 ego 旋转阵不该有 R[2,0](那是 pitch 项)"


def test_scale_k_to_feature_resolution() -> None:
    """K 缩放锚点:fx/fy/cx/cy 按宽高比缩放,末行恒 [0,0,1]。"""
    k = torch.tensor(K_NP, dtype=torch.float32)
    out = scale_k(k, (1242, 375), (311, 94))
    assert out[0, 0] == pytest.approx(K_NP[0, 0] * 311 / 1242)
    assert out[1, 1] == pytest.approx(K_NP[1, 1] * 94 / 375)
    assert out[0, 2] == pytest.approx(K_NP[0, 2] * 311 / 1242)
    assert out[1, 2] == pytest.approx(K_NP[1, 2] * 94 / 375)
    torch.testing.assert_close(out[2], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(scale_k(k, (1242, 375), (1242, 375)), k)  # 同尺寸 = 恒等


def test_project_pts_matches_world_to_img_flat() -> None:
    """平路(ego/sensor pitch=roll=0)+ 随机 yaw:逐点像素对账(≤1e-2 px)。"""
    rng = np.random.default_rng(1)
    ego = np.array(rng.uniform(-200, 200, 3).tolist() + [rng.uniform(-180.0, 180.0), 0.0, 0.0])
    se = np.array([1.2, 0.0, 1.65, rng.uniform(-180.0, 180.0), 0.0, 0.0])
    pts = np.stack([rng.uniform(-40, 40, 25), rng.uniform(-40, 40, 25), np.zeros(25)], axis=1)

    t_cam, _, _ = cam_world_pose(_t(ego), _t(se))
    t_cam_np, _ = _cam_world_np(ego, se)
    np.testing.assert_allclose(t_cam.numpy(), t_cam_np, atol=1e-4)

    uv, depth = project_pts(_t(pts), _t(ego), _t(se), K_T)
    for i, o in enumerate(_oracle_uv(pts, ego, se)):
        u, v, z = float(uv[i, 0]), float(uv[i, 1]), float(depth[i])
        if o is None:
            # oracle 拒绝 ⇒ torch 侧也必须(深度 ≤ 0.5 或出图,容忍浮点边界 1e-3)
            assert z <= 0.5 + 1e-3 or not (0 <= u < CAM.width) or not (0 <= v < CAM.height)
        else:
            assert z > 0.5
            assert abs(u - o[0]) < 1e-2 and abs(v - o[1]) < 1e-2


def test_project_pts_matches_world_to_img_tilted() -> None:
    """带 pitch/roll 的合成位姿(矩阵合成 → 分解回元组):逐点像素对账。"""
    rng = np.random.default_rng(2)
    for _ in range(4):
        ego = np.array(
            [
                *rng.uniform(-200, 200, 3),
                rng.uniform(-180.0, 180.0),
                rng.uniform(-40.0, 40.0),
                rng.uniform(-30.0, 30.0),
            ]
        )
        se = np.array(
            [
                *rng.uniform(-2.5, 2.5, 3),
                rng.uniform(-180.0, 180.0),
                rng.uniform(-25.0, 25.0),
                rng.uniform(-15.0, 15.0),
            ]
        )
        pts = np.stack([rng.uniform(-40, 40, 25), rng.uniform(-40, 40, 25), np.zeros(25)], axis=1)
        uv, depth = project_pts(_t(pts), _t(ego), _t(se), K_T)
        for i, o in enumerate(_oracle_uv(pts, ego, se)):
            u, v, z = float(uv[i, 0]), float(uv[i, 1]), float(depth[i])
            if o is None:
                assert z <= 0.5 + 1e-3 or not (0 <= u < CAM.width) or not (0 <= v < CAM.height)
            else:
                assert z > 0.5
                assert abs(u - o[0]) < 1e-2 and abs(v - o[1]) < 1e-2


def test_bev_cell_centers() -> None:
    b = BEVParams()
    assert b.res == pytest.approx(0.3)
    c = b.cell_centers()
    assert c.shape == (b.bev_h * b.bev_w, 3)
    np.testing.assert_allclose(c[0], (-14.85, -29.85, 0.0), atol=1e-4)
    np.testing.assert_allclose(c[-1], (14.85, 29.85, 0.0), atol=1e-4)
    np.testing.assert_allclose(c[1] - c[0], (0.3, 0.0, 0.0), atol=1e-4)
    np.testing.assert_allclose(c[b.bev_w] - c[0], (0.0, 0.3, 0.0), atol=1e-4)


def _bev_margins_np(
    b: BEVParams, ego: np.ndarray, calibs: dict[str, dict], size: tuple[int, int]
) -> np.ndarray:
    """每 BEV 单元到可见性判决边界的 numpy 距离 (bev_h, bev_w)。

    margin(cell) = max_cam min(u, W−u, v, H−v, z−0.5)(u/v 原始投影,z 深度);
    > 0 ⇔ 该单元至少一个相机可见。浮点分歧只会发生在 |margin| ≤ 0.01 的边缘。
    `size` = 特征图 (宽, 高);K 按同比例缩到该尺寸(与 `scale_k` 同式)。
    """
    fw, fh = size
    fx, cx = CAM.fx * fw / CAM.width, CAM.cx * fw / CAM.width
    fy, cy = CAM.fy * fh / CAM.height, CAM.cy * fh / CAM.height
    pts = b.cell_centers()
    margins = np.full((b.bev_h, b.bev_w), -np.inf)
    for se in calibs.values():
        s2e = np.array(se["sensor2ego"], dtype=np.float64)
        r_e = _rot_np(ego)
        r_s = _rot_np(s2e)
        r_w2c = g.CARLA_TO_CAM @ (r_e @ r_s).T
        t_cam = ego[:3] + r_e @ s2e[:3]
        p_c = (pts @ r_e.T + ego[:3] - t_cam) @ r_w2c.T
        z = p_c[:, 2]
        u = fx * p_c[:, 0] / z + cx
        v = fy * p_c[:, 1] / z + cy
        # 像素中心级覆盖边界(与 GKT mask 同口径):u ∈ [0, W−1]
        m = np.minimum.reduce([u, fw - 1 - u, v, fh - 1 - v, z - 0.5])
        margins = np.maximum(margins, m.reshape(b.bev_h, b.bev_w))
    return margins


def test_gkt_fusion_masks_match_oracle() -> None:
    """端到端:all-ones 特征 → 融合值 0/1;可见性掩码 vs numpy 逐单元对账。

    特征图**故意取 1/4 分辨率**(311×94),以覆盖"K 必须缩放"这条路径——特征图
    与图像同尺寸时缩放是恒等,坑 2 不会暴露。
    """
    b = BEVParams()
    gkt = GKT(b).to(DEVICE)
    poses = torch.tensor(
        [[10.0, 20.0, 0.5, 30.0, 0.0, 0.0], [-5.0, 3.0, 0.5, -120.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=DEVICE,
    )
    calibs = {  # B1 六相机布局
        name: {"sensor2ego": [1.2, 0.0, 1.65, yaw, 0.0, 0.0], "intrinsic": K_NP.tolist()}
        for name, yaw in [
            ("CAM_FRONT", 0.0),
            ("CAM_FRONT_RIGHT", -55.0),
            ("CAM_FRONT_LEFT", 55.0),
            ("CAM_BACK", 180.0),
            ("CAM_BACK_LEFT", 235.0),
            ("CAM_BACK_RIGHT", 125.0),
        ]
    }
    fw, fh = CAM.width // 4, CAM.height // 4  # 311×94 ≈ FPN P2
    feats = {name: torch.ones(2, 3, fh, fw, device=DEVICE) for name in calibs}
    bev, valid = gkt(feats, poses, calibs, (CAM.width, CAM.height))
    assert bev.shape == (2, 3, b.bev_h, b.bev_w)
    assert valid.shape == (2, 1, b.bev_h, b.bev_w)
    # all-ones 输入 → 权重归一化后有效单元恰为 1,无效为 0(float32 容差)
    expanded = valid.expand_as(bev)
    assert bool(torch.isclose(bev[expanded], torch.ones_like(bev[expanded]), atol=1e-5).all())
    assert bool(torch.isclose(bev[~expanded], torch.zeros_like(bev[~expanded]), atol=1e-5).all())
    for bi in range(2):
        margins = _bev_margins_np(b, poses[bi].cpu().numpy(), calibs, (fw, fh))
        mask_np = margins > 0
        diff = valid[bi, 0].cpu().numpy() != mask_np
        assert np.all(~diff | (np.abs(margins) <= 0.01)), (
            f"帧 {bi}: {int(diff.sum())} 单元分歧,{int(diff[np.abs(margins) > 0.01].sum())} 非边缘"
        )


def test_gkt_valid_coverage_on_real_rig() -> None:
    """回归锚点:真实 nuScenes rig + 平路位姿下,BEV 可见率必须 ≈94%(修前 1.25%)。

    这是**端到端**判据——两条坑(换序 / K 未缩放)任一条复发都会把它打到个位数。
    1/4 分辨率特征图 = FPN P2 的实际口径。
    """
    b = BEVParams()
    gkt = GKT(b).to(DEVICE)
    pose = torch.tensor([[10.0, 20.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=DEVICE)
    # NUS_CAMERA_RIG 的旋转是 (pitch,yaw,roll)(CARLA 序);infos 要 [x,y,z,yaw,pitch,roll]
    calibs = {
        name: {"sensor2ego": [*t, rot[1], rot[0], rot[2]], "intrinsic": K_NP.tolist()}
        for name, (t, rot) in NUS_CAMERA_RIG.items()
    }
    fw, fh = CAM.width // 4, CAM.height // 4
    feats = {name: torch.ones(1, 3, fh, fw, device=DEVICE) for name in calibs}
    _, valid = gkt(feats, pose, calibs, (CAM.width, CAM.height))
    cov = float(valid.float().mean())
    assert 0.85 < cov < 1.0, f"真实 rig 的 BEV 可见率 {cov:.2%}(期望 ≈94%,修前 1.25%)"
    margins = _bev_margins_np(b, pose[0].cpu().numpy(), calibs, (fw, fh))
    mask_np = margins > 0
    diff = valid[0, 0].cpu().numpy() != mask_np
    assert np.all(~diff | (np.abs(margins) <= 0.01)), f"{int(diff[np.abs(margins) > 0.01].sum())} 非边缘分歧"


def test_gkt_backward() -> None:
    """可微性锚点:bev 对输入特征有梯度(训练链必须成立)。"""
    b = BEVParams()
    gkt = GKT(b).to(DEVICE)
    poses = torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0]], dtype=torch.float32, device=DEVICE)
    calibs = {
        "CAM_FRONT": {"sensor2ego": [1.2, 0.0, 1.65, 0.0, 0.0, 0.0], "intrinsic": K_NP.tolist()},
        "CAM_BACK": {"sensor2ego": [1.2, 0.0, 1.65, 180.0, 0.0, 0.0], "intrinsic": K_NP.tolist()},
    }
    feats = {
        name: torch.randn(1, 3, CAM.height, CAM.width, device=DEVICE, requires_grad=True) for name in calibs
    }
    bev, _ = gkt(feats, poses, calibs, (CAM.width, CAM.height))
    bev.mean().backward()
    assert all(f.grad is not None and bool(torch.isfinite(f.grad).all()) for f in feats.values())

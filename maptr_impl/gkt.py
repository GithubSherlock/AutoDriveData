"""GKT:Geometry-aware Kernel Transform——环视相机特征 → BEV 特征(纯几何投影采样)。

MapTR 视角变换核心数学:BEV 网格单元中心 → ego 世界 → 各相机 → 双线性采样 →
按 1/深度加权融合。投影链与 B3 验收探针共用同一口径(autodrivedata.calib
world_to_img),tests/test_gkt.py 以该链为 oracle 逐点锁定 torch 实现。

与官方 MapTR GKT 的两处工程差异(核心几何一致;§5.11d 自实现口径):
1. 单高度采样(z = 0 路面)替代官方多高度——地图要素全在路面,单层已充分;
2. 融合取**最近相机独占采样**(官方 topk=1 交叉注意力的几何简化,避免多相机
   重叠区边界混叠),两者对"采样位置由几何投影决定"的口径相同。

BEV 口径(MapTRv2 同款):200×100 @ 0.3 m/pixel,x ∈ [−15, 15](前正)、
y ∈ [−30, 30](左正),即 BEVParams.pc_range = (xmin, ymin, xmax, ymax)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autodrivedata import geometry as g

# CARLA 系 → KITTI 相机系基变换的转置(geometry.CARLA_TO_CAM 正交、det=−1,转置即逆)
_C2K_T = torch.tensor(g.CARLA_TO_CAM.T, dtype=torch.float32)


@dataclass(frozen=True)
class BEVParams:
    """BEV 网格口径:pc_range = (xmin, ymin, xmax, ymax)[米,ego 局部系 x 前/y 左]。"""

    pc_range: tuple[float, float, float, float] = (-15.0, -30.0, 15.0, 30.0)
    bev_w: int = 100  # x 轴(前后)
    bev_h: int = 200  # y 轴(左右)
    z_level: float = 0.0  # 采样高度(路面)

    @property
    def res(self) -> float:
        """像素分辨率 [米/px](x/y 同:30/100 = 60/200 = 0.3)。"""
        return (self.pc_range[2] - self.pc_range[0]) / self.bev_w

    def cell_centers(self) -> np.ndarray:
        """BEV 单元中心 (bev_h*bev_w, 3) 行主序(reshape (bev_h, bev_w, 3) 还原)。"""
        x = np.linspace(self.pc_range[0], self.pc_range[2], self.bev_w, endpoint=False) + self.res / 2
        y = np.linspace(self.pc_range[1], self.pc_range[3], self.bev_h, endpoint=False) + self.res / 2
        xx, yy = np.meshgrid(x, y, indexing="xy")
        return np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, self.z_level)], axis=1).astype(np.float32)


# MapTRv2 同款 BEV 网格默认值(frozen,共享单例安全)
BEV_DEFAULT = BEVParams()


def _carla_rotation_torch(rot_rad: torch.Tensor) -> torch.Tensor:
    """CARLA Rotation (pitch, yaw, roll)[弧度] (…, 3) → (…, 3, 3) 旋转阵。

    组合顺序 Rz(yaw)·Ry(pitch)·Rx(roll),逐元素对应 geometry.carla_rotation_matrix
    (tests/test_gkt.py 锁定)。
    """
    pitch, yaw, roll = rot_rad[..., 0], rot_rad[..., 1], rot_rad[..., 2]
    cy, sy = yaw.cos(), yaw.sin()
    cp, sp = pitch.cos(), pitch.sin()
    cr, sr = roll.cos(), roll.sin()
    return torch.stack(
        [
            cy * cp,
            cy * sp * sr - sy * cr,
            -cy * sp * cr - sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            -sy * sp * cr + cy * sr,
            sp,
            -cp * sr,
            cp * cr,
        ],
        dim=-1,
    ).reshape(*rot_rad.shape[:-1], 3, 3)


def cam_world_pose(
    ego_pose: torch.Tensor, sensor2ego: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """相机世界位姿 + 变换阵(角度 [度],张量 float32)。

    ego_pose (…, 6) [x, y, z, yaw, pitch, roll];sensor2ego (…, 6) 同序(挂点偏移 +
    相对角)。返回:
    - t_cam (…, 3):相机世界位置 t_e + R_e·t_s
    - r_e (…, 3, 3):ego 世界旋转阵(ego 局部系点变世界:pts @ R_eᵀ)
    - r_w2c (…, 3, 3):世界系向量 → KITTI 相机系,与 geometry.world_to_cam 的 R 同式:
      CARLA_TO_CAM @ (R_e·R_s)ᵀ
    """
    rot_e = torch.deg2rad(ego_pose[..., 3:6])
    rot_s = torch.deg2rad(sensor2ego[..., 3:6])
    r_e = _carla_rotation_torch(rot_e)
    r_s = _carla_rotation_torch(rot_s)
    t_cam = ego_pose[..., :3] + torch.einsum("...ij,...j->...i", r_e, sensor2ego[..., :3])
    r_w2c = (r_e @ r_s @ _C2K_T.to(r_e.device)).transpose(-1, -2)
    return t_cam, r_e, r_w2c


def project_pts(
    pts_ego: torch.Tensor, ego_pose: torch.Tensor, sensor2ego: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """ego 局部系点 → 相机像素 + 深度(无阈值过滤,调用方自定可见性)。

    pts_ego (…, N, 3);ego_pose (…, 6) 度;sensor2ego (…, 6) 度;k (3, 3) 内参。
    返回 ((…, N, 2) 像素 (u, v), (…, N) 深度)。像素口径与 calib.world_to_img
    一致:u = fx·x/z + cx;深度 ≤ 0 的点照常输出(调用方判有效)。
    """
    t_cam, r_e, r_w2c = cam_world_pose(ego_pose, sensor2ego)
    p_w = torch.einsum("...nj,...ij->...ni", pts_ego, r_e) + ego_pose[..., :3].unsqueeze(-2)
    p_c = (p_w - t_cam.unsqueeze(-2)) @ r_w2c.transpose(-1, -2)
    z = p_c[..., 2]
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    return torch.stack([fx * p_c[..., 0] / z + cx, fy * p_c[..., 1] / z + cy], dim=-1), z


class GKT(nn.Module):
    """相机特征图 → BEV 特征:几何投影 + 双线性采样 + 1/深度加权融合。

    输入口径(B2 infos 直出):
    - feats: 相机名 → (B, C, H, W) 特征图(如 FPN P2,stride 4)
    - poses: (B, 6) [x, y, z, yaw, pitch, roll] 度
    - calibs: 相机名 → {"sensor2ego": [tx,ty,tz,yaw,pitch,roll] 度,
      "intrinsic": 3×3}(与批次共享)
    输出:
    - bev (B, C, bev_h, bev_w):特征,无相机可见处为 0
    - valid (B, 1, bev_h, bev_w):布尔,至少一个相机可见(诊断/掩码用)
    """

    def __init__(self, bev: BEVParams = BEV_DEFAULT) -> None:
        super().__init__()
        self.bev = bev
        self.register_buffer("grid_ego", torch.from_numpy(bev.cell_centers()), persistent=False)

    def forward(
        self, feats: dict[str, torch.Tensor], poses: torch.Tensor, calibs: dict[str, dict]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feats.keys() != calibs.keys():
            raise ValueError(f"相机集不一致: feats {sorted(feats)} vs calibs {sorted(calibs)}")
        b, _, feat_h, feat_w = next(iter(feats.values())).shape
        n = self.grid_ego.shape[0]
        grid_ego = self.grid_ego.expand(b, n, 3)
        best_d = None
        best_s = None
        for name, feat in feats.items():
            se = torch.tensor(calibs[name]["sensor2ego"], dtype=torch.float32, device=feat.device)
            k = torch.tensor(calibs[name]["intrinsic"], dtype=torch.float32, device=feat.device)
            uv, depth = project_pts(grid_ego, poses, se, k)
            u, v = uv[..., 0], uv[..., 1]
            # 覆盖判据 = 像素中心级 [0, W−1](与 calib.world_to_img 同口径);采样坐标
            # 是连续值,像素 k 中心在 (k+0.5) 处,归一化坐标 = (u+0.5)/W·2−1
            valid = (depth > 0.5) & (u >= 0) & (u <= feat_w - 1) & (v >= 0) & (v <= feat_h - 1)
            grid = torch.stack([(u + 0.5) / feat_w * 2 - 1, (v + 0.5) / feat_h * 2 - 1], dim=-1)
            grid = grid.view(b, self.bev.bev_h, self.bev.bev_w, 2)
            sampled = F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            d = depth.masked_fill(~valid, float("inf")).view(b, 1, self.bev.bev_h, self.bev.bev_w)
            if best_s is None:
                best_d, best_s = d, sampled
            else:
                assert best_d is not None
                take = d < best_d
                best_s = torch.where(take, sampled, best_s)
                best_d = torch.minimum(d, best_d)
        if best_s is None or best_d is None:
            raise ValueError("feats 为空")
        bev = best_s.masked_fill(best_d.isinf(), 0.0)
        return bev, ~best_d.isinf()

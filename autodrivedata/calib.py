"""KITTI 标定生成——照抄 auto3dlabel KittiCalib 解析契约(键名/行列数/缺省值)。

投影链(KittiCalib 语义):velodyne → (Tr_velo_to_cam) cam0 → (R0_rect) rect cam → (P2) 像素。
- P2:3×4 [fx 0 cx 0; 0 fy cy 0; 0 0 1 0];fx=fy=(W/2)/tan(fov_h/2)(CARLA 方形像素)
- R0_rect = I(CARLA 相机为理想针孔,无需整流)
- Tr_velo_to_cam:R = R_camK_world @ R_world_lidarC @ diag(1,−1,1),
  t = R_camK_world @ (t_lidar − t_cam)。
  diag(1,−1,1) = CARLA 传感器系(y 右)→ KITTI velodyne 系(y 左)的基轴翻转,
  与落盘时 carla_lidar_to_velodyne 的 y 翻转配套(两者共同保证 bin 为标准 KITTI 约定)
- P0/P1/P3 = 0(本设备无其他相机,诚实置零;auto3dlabel 只用 P2);Tr_imu_to_velo = [I|0]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from autodrivedata import geometry as g


@dataclass(frozen=True)
class CameraIntrinsics:
    """CARLA 相机内参(方形像素);fov_h_deg 为水平视场角(CARLA 的 fov 属性)。"""

    width: int
    height: int
    fov_h_deg: float

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / np.tan(np.radians(self.fov_h_deg) / 2.0)

    @property
    def fy(self) -> float:
        return self.fx  # CARLA 方形像素,fy = fx

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0

    def p2(self) -> np.ndarray:
        """3×4 投影矩阵(KITTI rectified cam 口径)。"""
        return np.array(
            [
                [self.fx, 0.0, self.cx, 0.0],
                [0.0, self.fy, self.cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )


def tr_velo_to_cam(
    lidar_location: tuple[float, float, float],
    lidar_rotation: tuple[float, float, float],
    cam_location: tuple[float, float, float],
    cam_rotation: tuple[float, float, float],
) -> np.ndarray:
    """LiDAR/相机 CARLA 位姿 → Tr_velo_to_cam 3×4(KITTI velodyne 系 → rectified cam0)。

    位姿角度为 (pitch, yaw, roll) 弧度;恒等位姿退化为 VELO_TO_CAM(手算锚点)。
    """
    r_cam = g.camera_rotation_world_to_cam(cam_rotation)
    r_lid = g.carla_rotation_matrix(lidar_rotation) @ np.diag([1.0, -1.0, 1.0])
    r = r_cam @ r_lid
    t = r_cam @ (np.asarray(lidar_location, dtype=np.float64) - np.asarray(cam_location, dtype=np.float64))
    return np.hstack([r, t.reshape(3, 1)])


def _fmt_3x4(m: np.ndarray) -> str:
    return " ".join(f"{v:.6e}" for v in np.asarray(m, dtype=np.float64).flat)


def _fmt_3x3(m: np.ndarray) -> str:
    return " ".join(f"{v:.6e}" for v in np.asarray(m, dtype=np.float64).flat)


@dataclass(frozen=True)
class KittiCalibOut:
    """KITTI calib txt 内容(字段镜像 auto3dlabel KittiCalib)。"""

    p2: np.ndarray  # 3x4
    tr_velo_to_cam: np.ndarray  # 3x4
    r0_rect: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    p0: np.ndarray = field(default_factory=lambda: np.zeros((3, 4), dtype=np.float64))
    p1: np.ndarray = field(default_factory=lambda: np.zeros((3, 4), dtype=np.float64))
    p3: np.ndarray = field(default_factory=lambda: np.zeros((3, 4), dtype=np.float64))
    tr_imu_to_velo: np.ndarray = field(
        default_factory=lambda: np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1))])
    )

    def to_text(self) -> str:
        """KITTI calib txt(KittiCalib.from_file 可解析;键序照 KITTI 惯例)。"""
        return "\n".join(
            [
                f"P0: {_fmt_3x4(self.p0)}",
                f"P1: {_fmt_3x4(self.p1)}",
                f"P2: {_fmt_3x4(self.p2)}",
                f"P3: {_fmt_3x4(self.p3)}",
                f"R0_rect: {_fmt_3x3(self.r0_rect)}",
                f"Tr_velo_to_cam: {_fmt_3x4(self.tr_velo_to_cam)}",
                f"Tr_imu_to_velo: {_fmt_3x4(self.tr_imu_to_velo)}",
            ]
        )

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_text() + "\n", encoding="utf-8")
        return p

"""P-D 教程 08 单目测距:检测框 → 地平面投影距离 + 迭代深度法(尺寸假设)。

链路:相机图像 2D 检测框 → 相机系距离。两个方法(纯值,autodrivedata 包内实现):
- **地平面投影距离**:框底中心像素经 `geometry.ground_intersection` 投到地平面,
  得到相机在地平面的垂足坐标 → 求"垂足到相机"(ego 局部 x 前向)的投影距离。
  依赖相机安装高度/俯仰已知,是单目测距的几何基线。
- **迭代深度法**:已知目标真实尺寸(车高 H≈1.5m)→ z = H·fy / 框高像素。
  单目存在尺度歧义(尺寸×距离同时未知),迭代法固定尺寸假设,故闭式解即迭代收敛值。

两个方法都不依赖 carla、不依赖模型,检测框从任意 2D 检测器来即可。
口径:与 `calib.world_to_img` 同系(相机系 z 向前);距离 = ego 前向分量(米)。
"""

from __future__ import annotations

import numpy as np

from autodrivedata.calib.core import CameraIntrinsics


def ground_plane_distance(
    cam_ground_x: float,
    cam_ground_y: float,
    ground_x: float,
    ground_y: float,
    cam_yaw_rad: float,
) -> float:
    """相机到地面交点的前向投影距离(ego 局部 x 分量,米)。

    输入为世界系坐标:相机**在地平面的垂足**(cam_ground_x/y)与交点(ground_x/y)。
    先把世界系差向量旋到 ego 局部系(x 前向 / y 左向),取 x 分量——与
    mapviz BEV 面板同一套"世界 → ego 局部"旋转(见 autodrivedata/perception/sem_bev.py 注释)。
    """
    dx = float(ground_x) - float(cam_ground_x)
    dy = float(ground_y) - float(cam_ground_y)
    c, s = np.cos(cam_yaw_rad), np.sin(cam_yaw_rad)
    return float(c * dx + s * dy)  # 世界 → ego 局部系 的 x 分量(前向距离)


def box_to_ground_distance(
    intrinsics: CameraIntrinsics,
    cam_location: tuple[float, float, float],
    cam_rotation_rad: tuple[float, float, float],
    ground_z: float,
    box_bottom_px: tuple[float, float],
    cam_yaw_rad: float,
) -> float | None:
    """2D 检测框底边中心像素 → 地平面投影距离(米)。

    world_cam = (cam_location, cam_rotation_rad)(mapviz.cam_pose 口径)。
    框底中心投到地平面 z=ground_z,再取相机前向距离。相机 z 低于地面
    (相机在地平面以上)或射线不出图时返回 None。
    """
    from autodrivedata.geometry import ground_intersection

    g = ground_intersection(
        (cam_location, cam_rotation_rad),
        intrinsics,
        box_bottom_px[0],
        box_bottom_px[1],
        ground_z,
    )
    if g is None:
        return None
    # 相机地平面垂足 = (cam_x, cam_y)(相机 z 不影响平面内坐标)
    return ground_plane_distance(
        cam_location[0],
        cam_location[1],
        g[0],
        g[1],
        cam_yaw_rad,
    )

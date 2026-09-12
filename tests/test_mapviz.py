"""mapviz.py 手算锚点单测(纯 numpy/PIL)——重点是**旋转单位**的回归锚点。

背景:曾把 `cam_pose` 的 yaw 直接当弧度用(实为度),6 相机里只有 yaw≈0 的
CAM_FRONT 恰好接近正确,侧/后相机 overlay 全画在错位置。判据落在
"命中点方位角是否落在该相机自身 FOV 内",与目检无关。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from autodrivedata import mapviz

# 训练 rig:6 相机名 → 挂点 yaw(度),取自 collect_surround.SURROUND_CAMS
RIG_YAW = (0.0, -55.0, 55.0, 180.0, 235.0, 125.0)
W, H, FOV = 1242, 375, 90.0
OFF = (1.2, 0.0, 1.65)  # carla_common.SENSOR_OFFSET
EGO = [62.1, 24.9, -0.006, 0.185, -0.137, 0.001]  # 真值样例帧(x, y, z, yaw, pitch, roll 度)


def _intrinsics():
    return mapviz.intrinsics_from_k(mapviz.calib_from_fov(W, H, FOV)["intrinsic"], (W, H))


def _se(yaw: float) -> list[float]:
    return [OFF[0], OFF[1], OFF[2], yaw, 0.0, 0.0]


def _at_bearing(bearing_deg: float, dist: float) -> np.ndarray:
    b = math.radians(bearing_deg)
    return np.array([[dist * math.cos(b), dist * math.sin(b)]])


class TestPose:
    def test_cam_pose_exports_radians(self):
        # se yaw 235° → 4.101 rad;若返回度数(235 ≫ 2π),这两条直接炸
        _, rot = mapviz.cam_pose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], _se(235.0))
        assert rot[1] == pytest.approx(math.radians(235.0))
        assert rot[1] < 2.0 * math.pi
        assert rot[0] == pytest.approx(0.0)  # pitch/roll 恒 0 的挂点

    def test_cam_pose_rotates_offset_by_ego_yaw(self):
        loc, _ = mapviz.cam_pose([10.0, 20.0, 1.0, 90.0, 0.0, 0.0], _se(0.0))
        # ego yaw=90° → 挂点 (1.2, 0) 转到世界 (0, 1.2)
        assert loc == pytest.approx((10.0, 21.2, 1.0 + OFF[2]), abs=1e-9)

    def test_ego_to_world_yaw90(self):
        w = mapviz.ego_to_world(np.array([[1.0, 0.0]]), [0.0, 0.0, 1.0, 90.0, 0.0, 0.0])
        assert w[0] == pytest.approx((0.0, 1.0, 1.0), abs=1e-9)


class TestProjection:
    @pytest.mark.parametrize("yaw", RIG_YAW)
    def test_bearing_hits_own_camera(self, yaw: float):
        """方位 = 挂点 yaw 的点(15m 外)必须落在该相机画面里、且近光轴。

        期望 u 手算:挂点前移 1.2m → 相机系里该点 (15 − 1.2cos yaw) 前、
        1.2 sin yaw 侧 → u = cx + fx·侧/前。度数当弧度用时本判据失败
        (实测:6 相机里 5 台只有 0~6% 命中落在自身 FOV)。
        """
        k = _intrinsics()
        pose = mapviz.cam_pose(EGO, _se(yaw))
        (uv,) = mapviz.project_points(_at_bearing(yaw, 15.0), EGO, pose, k)
        assert uv is not None, "方位=挂点 yaw 的点没投进该相机:旋转单位/FOV 口径错"
        u, v = uv
        rad = math.radians(yaw)
        fwd, lat = 15.0 - OFF[0] * math.cos(rad), OFF[0] * math.sin(rad)
        assert u == pytest.approx(k.cx + k.fy * lat / fwd, abs=4.0)
        assert k.cy < v < H  # 路面点必在光轴下方(相机高于路面 OFF[2])

    @pytest.mark.parametrize("yaw", RIG_YAW)
    def test_own_fov_fan_all_hits(self, yaw: float):
        """相机 yaw ±35° 扇面的点应**全部**投进该相机。

        只取 ±35° 而非 ±40°:挂点前移 1.2m 带来视差,侧向相机在 ±40° 处
        实际已越过 45° 半视场(实测 1242 宽图上 u=1251 → 出图)。
        """
        k = _intrinsics()
        pose = mapviz.cam_pose(EGO, _se(yaw))
        fan = np.vstack([_at_bearing(yaw + d, 12.0) for d in np.arange(-35.0, 35.1, 5.0)])
        hits = sum(1 for uv in mapviz.project_points(fan, EGO, pose, k) if uv is not None)
        assert hits == len(fan)

    def test_behind_camera_rejected(self):
        k = _intrinsics()
        pose = mapviz.cam_pose(EGO, _se(0.0))
        assert mapviz.project_lines([np.array([[-20.0, 0.0], [-10.0, 0.0]])], EGO, pose, k) == []

    def test_project_lines_splits_across_camera(self):
        """折线穿过相机平面(前→后→前)→ 断成两段,不跨相机后直连。"""
        k = _intrinsics()
        pose = mapviz.cam_pose(EGO, _se(0.0))
        xs = np.concatenate([np.linspace(25.0, -8.0, 34), np.linspace(-8.0, 25.0, 34)])
        segs = mapviz.project_lines([np.column_stack([xs, np.zeros(68)])], EGO, pose, k)
        assert len(segs) == 2

    def test_intrinsics_from_k_recovers_fov(self):
        k = _intrinsics()
        assert k.fov_h_deg == pytest.approx(FOV)
        assert k.fx == pytest.approx(621.0)


class TestDraw:
    def test_draw_counts_segments_and_paints(self):
        k = _intrinsics()
        pose = mapviz.cam_pose(EGO, _se(0.0))
        img = Image.new("RGB", (W, H), (0, 0, 0))
        line = np.column_stack([np.linspace(5.0, 25.0, 21), np.zeros(21)])
        assert mapviz.draw_projected_lines(ImageDraw.Draw(img), [line], EGO, pose, k) == 1
        # 空黑底上不存在撞色干扰 → 数品红像素即可自证"画上了"
        px = np.asarray(img).reshape(-1, 3)
        assert int((px == np.array(mapviz.PRED_COLOR)).all(axis=1).sum()) > 0

    def test_draw_returns_zero_when_invisible(self):
        k = _intrinsics()
        pose = mapviz.cam_pose(EGO, _se(180.0))  # 背对相机
        img = Image.new("RGB", (W, H), (0, 0, 0))
        line = np.column_stack([np.linspace(5.0, 25.0, 21), np.zeros(21)])
        assert mapviz.draw_projected_lines(ImageDraw.Draw(img), [line], EGO, pose, k) == 0


class TestBevPanel:
    def test_size_and_center_geometry(self):
        preds = [[np.array([[0.0, 0.0], [10.0, 0.0]])]]  # 逐类 → 折线 (P,2)
        panel = mapviz.bev_panel(preds, None, "", (200, 400))
        assert panel.size == (200, 400)
        arr = np.asarray(panel)
        ys, xs = np.nonzero((arr == np.array(mapviz.PRED_COLOR)).all(axis=2))
        assert len(xs) > 0, "预测折线没画上"
        # BEV 中心 = ego 原点 = 面板几何中心;折线沿 +x(前向)→ 只有 u 变、v 恒定在中心
        assert float(xs.min()) == pytest.approx(100.0, abs=2.0)
        assert float(ys.mean()) == pytest.approx(200.0, abs=2.0)
        assert float(xs.max()) < 200.0  # 10m 前向仍在窗口内

    def test_gt_is_optional(self):
        line = [[np.array([[0.0, 0.0], [1.0, 1.0]])]]
        assert mapviz.bev_panel([], None, "t", (64, 64)).size == (64, 64)
        assert mapviz.bev_panel([], [[]], "", (64, 64)).size == (64, 64)
        assert mapviz.bev_panel(line, line, "", (64, 64)).size == (64, 64)


class TestCalib:
    def test_calib_from_fov_matches_kitti_style(self):
        intrinsic = mapviz.calib_from_fov(1242, 375, 90.0)["intrinsic"]
        assert intrinsic[0][0] == pytest.approx(621.0)  # fx
        assert intrinsic[0][2] == pytest.approx(621.0)  # cx = W/2(采集器口径)
        assert intrinsic[1][2] == pytest.approx(187.5)  # cy = H/2

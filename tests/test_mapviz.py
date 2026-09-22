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


class TestBevPoints:
    """在线 SLAM 重建的点散布(散点口径,与 bev_panel 的折线口径不同)。"""

    def test_inside_points_drawn_outside_dropped(self):
        img = Image.new("RGB", (200, 200), (0, 0, 0))
        pts = np.array(
            [
                [0.0, 0.0],  # 窗口中心
                [10.0, 10.0],  # 窗口内
                [100.0, 0.0],  # x 超窗(前向 ±15m)
                [0.0, 100.0],  # y 超窗(左向 ±30m)
            ]
        )
        n = mapviz.bev_points(ImageDraw.Draw(img), pts, (255, 255, 255), (200, 200))
        assert n == 2, "只应画窗口内的 2 个点"
        arr = np.asarray(img)
        ys, xs = np.nonzero((arr == 255).all(axis=2))
        assert len(xs) == 2
        # (0,0) → 面板几何中心;(10,10) → x 前 10m ⇒ px=(10+15)/30·200≈166.7,
        #   y 左 10m ⇒ py=(30−10)/60·200≈66.7(**左为正方向** ⇒ y 越大像素行越小)
        assert float(xs.min()) == pytest.approx(100.0, abs=1.0)
        assert float(ys.max()) == pytest.approx(100.0, abs=1.0)
        assert float(xs.max()) == pytest.approx(166.7, abs=1.5)
        assert float(ys.min()) == pytest.approx(66.7, abs=1.5)

    def test_empty_and_1d_inputs_are_safe(self):
        img = Image.new("RGB", (64, 64), (0, 0, 0))
        assert mapviz.bev_points(ImageDraw.Draw(img), np.zeros((0, 2)), (255, 255, 255), (64, 64)) == 0
        # (N,3) 点云:只用前两列
        assert mapviz.bev_points(ImageDraw.Draw(img), np.zeros((3, 3)), (255, 255, 255), (64, 64)) == 3


class TestBevTrajectory:
    def test_cross_window_segment_not_drawn(self):
        """跨窗相邻点**不得**连线(否则会在面板上拉一条穿越全图的假边)。

        序列 `[窗内, 窗内, 窗外, 窗外, 窗内, 窗内]` 的相邻对逐个数:
        (0,1) 画、(1,2) 跳过、(2,3) 跳过、(3,4) 跳过、(4,5) 画 ⇒ **2 段**。
        若跨窗段被画上会是 5 段,且像素会横穿整个面板。
        """
        img = Image.new("RGB", (200, 200), (0, 0, 0))
        assert (
            mapviz.bev_trajectory(
                ImageDraw.Draw(img), np.array([[0.0, 0.0], [5.0, 0.0]]), (255, 255, 255), 1, (200, 200)
            )
            == 1
        )
        crossed = np.array([[0.0, 0.0], [5.0, 0.0], [500.0, 0.0], [500.0, 5.0], [0.0, 0.0], [5.0, 0.0]])
        img2 = Image.new("RGB", (200, 200), (0, 0, 0))
        n = mapviz.bev_trajectory(ImageDraw.Draw(img2), crossed, (255, 255, 255), 1, (200, 200))
        assert n == 2, f"跨窗段被画了(实得 {n} 段,期望 2)"
        # 画出的像素必须全在面板内(跨窗段若画了会出界)
        arr = np.asarray(img2)
        assert ((arr == 255).all(axis=2)).sum() > 0
        assert ((arr == 255).all(axis=2)).sum() <= 200 * 200

    def test_short_input_returns_zero(self):
        img = Image.new("RGB", (64, 64), (0, 0, 0))
        assert mapviz.bev_trajectory(ImageDraw.Draw(img), np.zeros((1, 2)), (255, 255, 255), 1, (64, 64)) == 0
        assert mapviz.bev_trajectory(ImageDraw.Draw(img), np.zeros((0, 2)), (255, 255, 255), 1, (64, 64)) == 0


class TestBevPanelSlamOverlay:
    def test_points_and_traj_reach_the_panel(self):
        img = mapviz.bev_panel(
            [],
            None,
            "",
            (200, 200),
            points=np.array([[0.0, 0.0], [5.0, 5.0]]),
            traj=np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]]),
        )
        arr = np.asarray(img)
        assert ((arr == np.array(mapviz.MAP_COLOR)).all(axis=2)).sum() > 0, "SLAM 地图点没画上"
        assert ((arr == np.array(mapviz.TRAJ_COLOR)).all(axis=2)).sum() > 0, "轨迹没画上"

    def test_stats_out_param_counts_draws(self):
        """`stats` 就地填绘制计数 —— 在线流靠它做**数值自证**(不靠目检)。

        关键:`n_points` 是**窗内**点数、`n_points_total` 是输入总数 ⇒ 两者之比就是
        "窗内占比"诊断(计划 B4 第 4 项:窗外的点不该出现)。
        """
        stats: dict[str, int] = {}
        pts = np.array([[0.0, 0.0], [5.0, 0.0], [100.0, 0.0], [0.0, 100.0]])  # 2 内 2 外
        mapviz.bev_panel(
            [],
            None,
            "",
            (200, 200),
            points=pts,
            traj=np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]]),
            stats=stats,
        )
        assert stats["n_points"] == 2
        assert stats["n_points_total"] == 4
        assert stats["n_traj_seg"] == 2

    def test_stats_absent_by_default(self):
        """不给 `stats` 时行为不变(现有调用方不必跟着改)。"""
        assert mapviz.bev_panel([], None, "", (64, 64)).size == (64, 64)


class TestBevWindowMask:
    """窗口判据的**单一来源**:画点 / 连轨迹 / 调用方诊断共用它。"""

    def test_boundaries_are_inclusive(self):
        m = mapviz.bev_window_mask(
            np.array(
                [
                    [mapviz.BEV_X[0], mapviz.BEV_Y[0]],  # 角点(含)
                    [mapviz.BEV_X[1], mapviz.BEV_Y[1]],  # 角点(含)
                    [mapviz.BEV_X[1] + 1e-6, 0.0],  # 差一点出界
                    [0.0, mapviz.BEV_Y[0] - 1e-6],
                ]
            )
        )
        assert m.tolist() == [True, True, False, False]

    def test_empty_input_is_safe(self):
        assert mapviz.bev_window_mask(np.zeros((0, 2))).tolist() == []
        assert mapviz.bev_window_mask(np.zeros((0, 3))).tolist() == []

    def test_agrees_with_bev_points_count(self):
        """掩码与 `bev_points` 的实际绘制数必须一致(否则诊断会骗人)。"""
        rng = np.random.default_rng(0)
        pts = np.stack([rng.uniform(-40, 40, 500), rng.uniform(-60, 60, 500)], 1)
        img = Image.new("RGB", (200, 200), (0, 0, 0))
        n_draw = mapviz.bev_points(ImageDraw.Draw(img), pts, (255, 255, 255), (200, 200))
        assert n_draw == int(mapviz.bev_window_mask(pts).sum())


class TestBevPointsBatchPerf:
    def test_large_cloud_draws_in_bounded_time(self):
        """40 万点(累积地图上限量级)必须**批量**提交,不能逐点 `draw.point`。

        判据是**返回计数正确** + 单次绘制耗时有界。逐点版在 40 万点上要数百毫秒
        (实时流 5 fps 的整帧预算只有 200 ms),批量版是毫秒级 —— 阈值取 2 s 留足
        慢机器余量,只拦"逐点 + 巨大点云"这种量级错,不做微基准。
        """
        import time as _time

        rng = np.random.default_rng(0)
        pts = np.stack([rng.uniform(-15, 15, 400_000), rng.uniform(-30, 30, 400_000)], 1)
        img = Image.new("RGB", (420, 420), (0, 0, 0))
        t0 = _time.perf_counter()
        n = mapviz.bev_points(ImageDraw.Draw(img), pts, (255, 255, 255), (420, 420))
        dt = _time.perf_counter() - t0
        assert n == 400_000
        assert dt < 2.0, f"40 万点绘制 {dt:.2f}s —— 疑似退回逐点 draw.point"


class TestCalib:
    def test_calib_from_fov_uses_index_centre_principal_point(self):
        """主点 = **索引约定中心** `(w−1)/2`,不是 `w/2`。

        依据(2026-09-22 实测裁决,见 `bin/probe_calib.py` A4/A6 锚):
        - A4 轴目标物实例分割掩膜的**索引**中点线性回归 → cx = 620.500(shift=0,corner
          约定),残差 0.200 px;若按 center 约定(shift=+0.5)则给 621.000。
        - A3 用 LiDAR 平面点投影 + 双线性采样深度图:corner 约定 median|e| 0.0003 m
          vs center 0.023 m,六相机一致(~70×)。
        `fx` 仍是 `(w/2)/tan(fov/2) = 621.0` —— "半 FOV ↔ 半宽"与"索引中心"是两件事,
        A4 独立测出 f_est = 621.60 px(标称 621.00,差 0.1%),两者并存不矛盾。
        """
        intrinsic = mapviz.calib_from_fov(1242, 375, 90.0)["intrinsic"]
        assert intrinsic[0][0] == pytest.approx(621.0)  # fx = (w/2)/tan(fov/2)
        assert intrinsic[0][2] == pytest.approx(620.5)  # cx = (w−1)/2
        assert intrinsic[1][2] == pytest.approx(187.0)  # cy = (h−1)/2

    def test_calib_from_fov_agrees_with_camera_intrinsics(self):
        """`calib_from_fov` 与 `CameraIntrinsics` 必须**同式**(单一来源,防再次漂移)。"""
        from autodrivedata.calib import CameraIntrinsics

        intr = mapviz.calib_from_fov(1242, 375, 90.0)["intrinsic"]
        k = CameraIntrinsics(width=1242, height=375, fov_h_deg=90.0)
        assert (intr[0][0], intr[1][1]) == pytest.approx((k.fx, k.fy))
        assert (intr[0][2], intr[1][2]) == pytest.approx((k.cx, k.cy))

    def test_intrinsics_from_k_preserves_principal_point(self):
        """`intrinsics_from_k` 必须**直读** K 的 cx/cy,不得丢掉重算(历史缺陷)。

        构造一个主点**刻意偏离** `(w−1)/2` 的 K:若实现退回重算,这条会失败。
        """
        k = [[621.0, 0.0, 617.25], [0.0, 621.0, 190.75], [0.0, 0.0, 1.0]]
        intr = mapviz.intrinsics_from_k(k, (1242, 375))
        assert intr.cx == pytest.approx(617.25)
        assert intr.cy == pytest.approx(190.75)
        assert intr.fx == pytest.approx(621.0)
        assert intr.fov_h_deg == pytest.approx(90.0)

    def test_intrinsics_from_k_falls_back_when_k_has_no_principal_point(self):
        """K 缺主点(2×2 或 3×3 但第三列为 0)⇒ 回落到索引约定中心。"""
        intr = mapviz.intrinsics_from_k([[621.0, 0.0, 0.0], [0.0, 621.0, 0.0], [0.0, 0.0, 1.0]], (1242, 375))
        assert intr.cx == pytest.approx(620.5)
        assert intr.cy == pytest.approx(187.0)

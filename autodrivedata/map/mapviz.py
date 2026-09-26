"""地图矢量投影与绘制(纯值):ego 系折线 → 相机像素 + BEV 面板。

离线目检([autodrivedata/map/viz_maptr_pred.py](viz_maptr_pred.py))与实时流
([autodrivedata/sim/view_stream.py](../sim/view_stream.py) `--maptr-ckpt`)共用同一条链,避免两处
各写一遍投影。坐标链:

  ego 局部系(x 前向 / y 左向 / z=0)
  → CARLA 世界(绕 z 转 yaw + 平移 ego2global)
  → `calib.world_to_img`(内参从 infos 直读、fov 由 fx 反推,不硬编码)

**旋转单位是弧度** —— `cam_pose` 的出口口径与 probe_mapvec_proj 参考实现一致
(`world_to_img`/`carla_rotation_matrix` 都吃弧度)。踩坑记录:曾把度数值直接传
进去,6 相机里只有 yaw≈0 的 CAM_FRONT 恰好接近正确,侧/后相机 overlay 全画在错
位置。数值判据(见 [tests/test_mapviz.py](../tests/map/test_mapviz.py)):命中点方位角
落在该相机 yaw±45°(其 90° FOV)内的比例——弧度口径 91~100%,度数口径 0~6%。

配色(C23 撞色口径):预测品红 / GT 青绿 —— 路面场景与既有 overlay(GT 框
绿/蓝/黄/橙、灯态红/黄/绿)都不含品红。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from autodrivedata.calib.core import CameraIntrinsics, world_to_img
from autodrivedata.utils import fonts

PRED_COLOR = (255, 0, 255)  # 品红:路面场景罕见
GT_COLOR = (0, 255, 255)  # 青绿:同罕见(植被绿与其可区分)
MAP_COLOR = (200, 200, 200)  # 浅灰:SLAM 累积点云(与 pred 品红 / GT 青绿都不撞)
TRAJ_COLOR = (0, 255, 255)  # 青绿:轨迹(与 GT_COLOR 同色系但语义独立,同图不同现)
BEV_X = (-15.0, 15.0)  # BEV 窗口 x(前向,米)——与模型输出系同口径
BEV_Y = (-30.0, 30.0)  # BEV 窗口 y(左向,米)

# (位置, (pitch, yaw, roll) **弧度**)——世界系的相机位姿
CamPose = tuple[tuple[float, float, float], tuple[float, float, float]]


def ego_to_world(pts: np.ndarray, ego: list[float]) -> list[tuple[float, float, float]]:
    """ego 局部系折线点 (N, 2|3) → 世界系 (z = ego 高度)。

    ego = [x, y, z, yaw, pitch, roll] 度(infos.ego2global 与 CARLA transform 同序)。
    """
    a = math.radians(ego[3])
    c, s = math.cos(a), math.sin(a)
    return [
        (c * float(p[0]) - s * float(p[1]) + ego[0], s * float(p[0]) + c * float(p[1]) + ego[1], ego[2])
        for p in pts
    ]


def cam_pose(eg: list[float], se: list[float]) -> CamPose:
    """ego2global + sensor2ego → 相机世界位姿(**位置米 / 姿态弧度**)。

    挂点按 `Rz(yaw_ego)` 旋转后平移;pitch/roll 直接相加——挂点 pitch/roll 恒 0、
    ego 俯仰量级 ~0.1°(30m 处像素偏差 <1px),与 probe_mapvec_proj 参考实现同近似。
    """
    a = math.radians(eg[3])
    c, s = math.cos(a), math.sin(a)
    loc = (
        eg[0] + c * se[0] - s * se[1],
        eg[1] + s * se[0] + c * se[1],
        eg[2] + se[2],
    )
    return loc, (math.radians(eg[4] + se[4]), math.radians(eg[3] + se[3]), math.radians(eg[5] + se[5]))


def intrinsics_from_k(k: list[list[float]], size: tuple[int, int]) -> CameraIntrinsics:
    """infos 内参 3×3 → CameraIntrinsics。

    **K 是权威来源**:fov 由 `k[0][0]`(= fx)反推、**cx/cy 直读** `k[0][2]`/`k[1][2]`
    (不再丢弃重算——历史实现只取 fx 后由 `CameraIntrinsics` 重算主点,一旦落盘的 K 与
    `(w−1)/2` 不同,投影就会静默偏离落盘口径)。`size` 提供画幅;K 里缺 cx/cy 时回落到
    `(w−1)/2`/`(h−1)/2`。
    """
    w, h = size
    fx = float(k[0][0])
    fov_h = math.degrees(2.0 * math.atan((w / 2.0) / fx)) if fx > 0 else 0.0
    # 主点:直读;缺失或为 0(退化的 K)⇒ None,由 CameraIntrinsics 回落到 (w−1)/2
    cx = float(k[0][2]) if len(k[0]) > 2 and k[0][2] else None
    cy = float(k[1][2]) if len(k) > 1 and len(k[1]) > 2 and k[1][2] else None
    return CameraIntrinsics(width=w, height=h, fov_h_deg=fov_h, cx_override=cx, cy_override=cy)


def calib_from_fov(width: int, height: int, fov_deg: float) -> dict[str, Any]:
    """(宽, 高, fov) → B2 口径内参块。**全仓唯一 fov→fx 落点。**

    fx/fy 与主点都取自 `CameraIntrinsics`(单一公式源),不再各写一遍 `w/2`。

    ⚠️ **主点口径已从 `w/2` 改为 `(w−1)/2`**(2026-09-22 实测裁决,`autodrivedata/calib/probe_calib.py`
    A3/A4):CARLA 渲染光栅是 **corner** 约定 —— 索引 i 的连续坐标就是 i,故
    `cx = (1242−1)/2 = 620.5`、`cy = (375−1)/2 = 187.0`。A3 在 corner 下 median|e|
    0.0003 m、center 下 0.023 m(~70×,六相机一致);A4 掩膜索引中点回归给 620.50。
    旧产物/旧权重里的 `621.0 / 187.5` 是同一个物理主点在 center 约定下的**另一种写法**,
    差恰好半像素 —— 新采集一律写 corner 值(旧权重已全部标废弃,重采重训)。
    """
    intr = CameraIntrinsics(width=width, height=height, fov_h_deg=fov_deg)
    return {"intrinsic": [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]]}


def project_points(
    points: np.ndarray, ego: list[float], pose: CamPose, intrinsics: CameraIntrinsics
) -> list[tuple[float, float] | None]:
    """点序列 (N, 2|3) ego 系 → 像素/None(相机后或出图外)。"""
    loc, rot = pose
    return [world_to_img(p, loc, rot, intrinsics) for p in ego_to_world(np.asarray(points), ego)]


def project_lines(
    lines: list[np.ndarray], ego: list[float], pose: CamPose, intrinsics: CameraIntrinsics
) -> list[list[tuple[float, float]]]:
    """折线列表(ego 系)→ [(u, v) 段列表](段 = 连续可见点的相邻对)。

    相机后(深度 ≤ 0.5m)或出图外的点断开成段,不跨遮挡连线。
    """
    segs: list[list[tuple[float, float]]] = []
    for line in lines:
        run: list[tuple[float, float]] = []
        for q in project_points(line, ego, pose, intrinsics):
            if q is None:
                if len(run) >= 2:
                    segs.append(run)
                run = []
            else:
                run.append(q)
        if len(run) >= 2:
            segs.append(run)
    return segs


def draw_projected_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[np.ndarray],
    ego: list[float],
    pose: CamPose,
    intrinsics: CameraIntrinsics,
    color: tuple[int, int, int] = PRED_COLOR,
    width: int = 3,
) -> int:
    """投影并画到图上,返回**画出的段数**(0 = 没画上:数值自证,不靠目检)。"""
    segs = project_lines(lines, ego, pose, intrinsics)
    for seg in segs:
        draw.line([q for p in seg for q in p], fill=color, width=width)
    return len(segs)


def bev_px(x: float, y: float, size: tuple[int, int]) -> tuple[float, float]:
    """ego 系 (x 前 / y 左) → BEV 面板像素。**BEV 窗口的唯一换算处**(bev_panel 也用它)。"""
    w, h = size
    return (x - BEV_X[0]) / (BEV_X[1] - BEV_X[0]) * w, (BEV_Y[1] - y) / (BEV_Y[1] - BEV_Y[0]) * h


def bev_window_mask(pts: np.ndarray) -> np.ndarray:
    """点 (N,2|3) 是否落在 BEV 窗口内 `(N,) bool`。

    **单一来源**:`bev_points`(画点)、`bev_trajectory`(只连窗内相邻点)与调用方
    的"窗内点占比"诊断共用它 —— 三处各写一遍窗口判据迟早漂。
    """
    arr = np.asarray(pts, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)
    arr = arr.reshape(-1, arr.shape[-1])
    return (
        (arr[:, 0] >= BEV_X[0]) & (arr[:, 0] <= BEV_X[1]) & (arr[:, 1] >= BEV_Y[0]) & (arr[:, 1] <= BEV_Y[1])
    )


def bev_points(
    draw: ImageDraw.ImageDraw,
    pts: np.ndarray,
    color: tuple[int, int, int] = MAP_COLOR,
    size: tuple[int, int] = (420, 420),
) -> int:
    """点散布到 BEV 窗口,返回**画出的点数**(0 = 没画上:数值自证,不靠目检)。

    为什么单开一个函数而不是复用 `bev_panel`:`bev_panel` 走 `draw.line` 折线口径
    (每条 ≥2 点),而 SLAM 累积点云 / 原始 LiDAR 是**散点** —— 拿它当折线画会把
    相邻两点连成假线。窗口与 `bev_panel` 共用 `BEV_X`/`BEV_Y` 与 `bev_px`,保证两图可比。

    **逐点 Python 循环改批量**:SLAM 累积地图可达 40 万点,逐点 `draw.point` 单帧要
    几百毫秒(实时流 5 fps 的预算只有 200 ms)。改成 numpy 算像素 + 一次
    `draw.point(list)` 提交;窗口外的点先被 `bev_window_mask` 滤掉(窗内只占少数)。
    """
    arr = np.asarray(pts, dtype=np.float64)
    if arr.size == 0:
        return 0
    arr = arr.reshape(-1, arr.shape[-1])[:, :2]
    arr = arr[bev_window_mask(arr)]
    if len(arr) == 0:
        return 0
    w, h = size
    px = (arr[:, 0] - BEV_X[0]) / (BEV_X[1] - BEV_X[0]) * w
    py = (BEV_Y[1] - arr[:, 1]) / (BEV_Y[1] - BEV_Y[0]) * h
    draw.point([(float(a), float(b)) for a, b in zip(px, py, strict=True)], fill=color)
    return len(arr)


def bev_trajectory(
    draw: ImageDraw.ImageDraw,
    poses_ego: np.ndarray,
    color: tuple[int, int, int] = TRAJ_COLOR,
    width: int = 1,
    size: tuple[int, int] = (420, 420),
) -> int:
    """轨迹折线(ego 系 (N,2|3))画到 BEV 窗口,返回**画出的段数**。

    **只画窗口内的连续段**:跨窗口的相邻点若直接连线,会在面板上拉出一条穿越全图的
    假边(§5.11 B2 GT 粗筛踩过同类"跨窗折线"坑)。
    """
    arr = np.asarray(poses_ego, dtype=np.float64)
    if arr.ndim != 2 or len(arr) < 2:
        return 0
    arr = arr[:, :2]
    inside = bev_window_mask(arr)
    n = 0
    for (x0, y0), (x1, y1), ok0, ok1 in zip(arr[:-1], arr[1:], inside[:-1], inside[1:], strict=True):
        if not (ok0 and ok1):
            continue
        draw.line(
            [bev_px(float(x0), float(y0), size), bev_px(float(x1), float(y1), size)], fill=color, width=width
        )
        n += 1
    return n


def bev_panel(
    preds: list[list[np.ndarray]],
    gts: list[list[np.ndarray]] | None = None,
    title: str = "",
    size: tuple[int, int] = (420, 420),
    points: np.ndarray | None = None,
    traj: np.ndarray | None = None,
    stats: dict[str, int] | None = None,
) -> Image.Image:
    """BEV 面板:pred 品红 / GT 青绿 / SLAM 地图点浅灰 / 轨迹青绿;窗口 BEV_X × BEV_Y。

    `gts=None` 供实时流用(线上无地图 GT:GT 来自 A 阶段矢量库,不在 CARLA 里)。
    `points`/`traj` 供在线 SLAM 重建叠加(ego 系,窗口外的点不画)。

    `stats` 给定时**就地填入绘制计数**(`n_points` / `n_traj_seg` / `n_points_total`),
    供调用方做**数值自证**(画上没画上不靠目检)。做成 out-param 而非改返回值:
    现有调用方(`view_stream` / 测试)不必跟着改签名。
    """
    w, h = size
    img = Image.new("RGB", (w, h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    def px(x: float, y: float) -> tuple[float, float]:
        return bev_px(x, y, size)

    if points is not None:
        n_drawn = bev_points(draw, points, MAP_COLOR, size)
        if stats is not None:
            arr = np.asarray(points, dtype=np.float64)
            stats["n_points"] = n_drawn
            stats["n_points_total"] = 0 if arr.size == 0 else int(arr.reshape(-1, arr.shape[-1]).shape[0])
    for gc in gts or []:
        for g in gc:
            draw.line([q for p in g for q in px(p[0], p[1])], fill=GT_COLOR, width=1)
    for pc in preds:
        for p in pc:
            draw.line([q for pt in p for q in px(pt[0], pt[1])], fill=PRED_COLOR, width=1)
    if traj is not None:
        n_seg = bev_trajectory(draw, traj, TRAJ_COLOR, 1, size)
        if stats is not None:
            stats["n_traj_seg"] = n_seg
    draw.rectangle([(0, 0), (w - 1, h - 1)], outline=(120, 120, 120))
    if title:
        # 走 fonts:直接 draw.text(...) 不给 font= 会用 PIL 内置位图字体(无中文字形)
        fonts.draw_text(draw, (4, 2), title, size=15, fill=(255, 255, 255))
    return img

"""地图矢量投影与绘制(纯值):ego 系折线 → 相机像素 + BEV 面板。

离线目检([bin/viz_maptr_pred.py](bin/viz_maptr_pred.py))与实时流
([bin/view_stream.py](bin/view_stream.py) `--maptr-ckpt`)共用同一条链,避免两处
各写一遍投影。坐标链:

  ego 局部系(x 前向 / y 左向 / z=0)
  → CARLA 世界(绕 z 转 yaw + 平移 ego2global)
  → `calib.world_to_img`(内参从 infos 直读、fov 由 fx 反推,不硬编码)

**旋转单位是弧度** —— `cam_pose` 的出口口径与 probe_mapvec_proj 参考实现一致
(`world_to_img`/`carla_rotation_matrix` 都吃弧度)。踩坑记录:曾把度数值直接传
进去,6 相机里只有 yaw≈0 的 CAM_FRONT 恰好接近正确,侧/后相机 overlay 全画在错
位置。数值判据(见 [tests/test_mapviz.py](tests/test_mapviz.py)):命中点方位角
落在该相机 yaw±45°(其 90° FOV)内的比例——弧度口径 91~100%,度数口径 0~6%。

配色(C23 撞色口径):预测品红 / GT 青绿 —— 路面场景与既有 overlay(GT 框
绿/蓝/黄/橙、灯态红/黄/绿)都不含品红。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from autodrivedata.calib import CameraIntrinsics, world_to_img

PRED_COLOR = (255, 0, 255)  # 品红:路面场景罕见
GT_COLOR = (0, 255, 255)  # 青绿:同罕见(植被绿与其可区分)
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
    """infos 内参 3×3 → CameraIntrinsics(fov 由 fx 反推,不硬编码)。"""
    w, h = size
    fov_h = math.degrees(2.0 * math.atan((w / 2.0) / k[0][0]))
    return CameraIntrinsics(width=w, height=h, fov_h_deg=fov_h)


def calib_from_fov(width: int, height: int, fov_deg: float) -> dict[str, Any]:
    """(宽, 高, fov) → B2 口径内参块(与 collect_surround 同式,单一来源 = carla_common.CAM_ATTRS)。"""
    fx = width / 2.0 / math.tan(math.radians(fov_deg / 2.0))
    return {"intrinsic": [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]]}


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


def bev_panel(
    preds: list[list[np.ndarray]],
    gts: list[list[np.ndarray]] | None = None,
    title: str = "",
    size: tuple[int, int] = (420, 420),
) -> Image.Image:
    """BEV 面板:pred 品红 / GT 青绿;窗口 BEV_X × BEV_Y(与模型输出系同口径)。

    `gts=None` 供实时流用(线上无地图 GT:GT 来自 A 阶段矢量库,不在 CARLA 里)。
    """
    w, h = size
    img = Image.new("RGB", (w, h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    def px(x: float, y: float) -> tuple[float, float]:
        return (x - BEV_X[0]) / (BEV_X[1] - BEV_X[0]) * w, (BEV_Y[1] - y) / (BEV_Y[1] - BEV_Y[0]) * h

    for gc in gts or []:
        for g in gc:
            draw.line([q for p in g for q in px(p[0], p[1])], fill=GT_COLOR, width=1)
    for pc in preds:
        for p in pc:
            draw.line([q for pt in p for q in px(pt[0], pt[1])], fill=PRED_COLOR, width=1)
    draw.rectangle([(0, 0), (w - 1, h - 1)], outline=(120, 120, 120))
    if title:
        draw.text((4, 2), title, fill=(255, 255, 255))
    return img

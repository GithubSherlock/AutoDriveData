"""P2-A 语义 BEV:相机图像 → YOLOPv2(检测+车道线+可行驶)+ YOLO11 实例分割 → BEV 鸟瞰。

对应教程 05(实时语义分割 SegFormer/YOLOPv2 对比)+ 06(分割结果投影 BEV 融合)。
本仓栈落地:autodrivedata 纯值库 + autodrivedata/mapviz 投影链,不 import carla。

模型:
- YOLOPv2(TorchScript,官方 V0.0.1):单模型三合一——2D 检测 + 车道线(ll)+ 可行驶区(da)
- YOLO11-seg(ultralytics):补充"非路面"实例分割(car/bus/truck/person)——YOLOPv2 只
  给路面/车道线/检测框,没有像素级车辆掩膜,语义 BEV 需要物体像素

投影(教程 06 原理):分割掩膜像素 → 利用相机模型把"地平面上的像素"投到世界系
BEV。实现:对每个掩膜像素,沿相机射线与地面 z = ego_z - 0.5m 求交(掩膜像素对应
场景物体,取其与地面交点 → BEV 中的"占位"投影)。投影复用 autodrivedata/mapviz.cam_pose
+ calib.world_to_img(单一投影实现,与采集/实时流共用)。

输出:outputs/sem_bev/{cam}/... 每帧三通道 BEV 语义图(cv2 三色通道):
- 可行驶区(da)   = 绿
- 车道线(ll)     = 黄
- 障碍物实例     = 品红(car/bus/truck/person 掩膜)
+ BEV 合成帧全览图(outputs/sem_bev/bev_{frame}.png:四路投影拼 + 上视角矢量对照)

口径:BEV 窗口 = BEV_RANGE(autodrivedata/mapviz,与 MapTR 面板同窗口);像素 = 0.2m。
离线批处理(CPU/GPU 均可),输出逐帧图 + 一页对照 PDF。

用法:
  PYTHONPATH=$PWD python -m autodrivedata.perception.sem_bev --frames 0-20 [--gpu]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from autodrivedata.geometry import ground_intersection  # noqa: F401 — 语义 BEV 与采集/实时流共用同一投影
from autodrivedata.map.mapviz import BEV_X, BEV_Y, CameraIntrinsics, cam_pose, intrinsics_from_k
from autodrivedata.paths import project_path

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None

DA_COLOR = (0, 200, 0)  # 可行驶区 绿
LL_COLOR = (0, 200, 200)  # 车道线   黄(BGR)
OBJ_COLOR = (200, 0, 200)  # 障碍物   品红(BGR)
DETECT_CLS = {2: "car", 5: "bus", 7: "truck", 0: "person"}

BEV_PX = 0.2  # BEV 每像素 = 0.2m(与 mapviz BEV 面板一致)
GROUND_Z_OFF = 0.5  # 地平面在 ego z 以下 0.5m(车轮着地点近似)


def bev_to_px(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    """BEV 世界系点 (x, y) → 面板像素(与 mapviz.bev_panel.px 同式)。"""
    u = int((x - BEV_X[0]) / (BEV_X[1] - BEV_X[0]) * w)
    v = int((BEV_Y[1] - y) / (BEV_Y[1] - BEV_Y[0]) * h)
    return u, v


def init_camera(calib_cam: dict, size: tuple[int, int]) -> tuple[CameraIntrinsics, list[float]]:
    """calib.json CAM_* → (intrinsics, sensor2ego)。

    内参走 `mapviz.intrinsics_from_k`(**直读** K 的 cx/cy):历史实现只抄 fx 反推 fov、
    把落盘的主点丢掉重算成 `(w−1)/2`,一旦产物里的 K 与它不同(旧产物是 `w/2`)就会
    静默带半像素横移 —— 落盘的 K 才是权威口径。
    """
    return intrinsics_from_k(calib_cam["intrinsic"], size), calib_cam["sensor2ego"]


def project_mask_to_bev(
    mask: np.ndarray,
    world_cam: tuple[tuple[float, float, float], tuple[float, float, float]],
    intrinsics: CameraIntrinsics,
    ground_z: float,
    ego: list[float],
    bev: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    """掩膜像素(1 = 命中)→ BEV 着色(像素级地面投影)。

    ground_intersection 返回**世界系**交点,BEV 面板是 **ego 局部系**(x 前向 /
    y 左向,原点 = ego)→ 投影前先按 ego yaw 旋转回局部系。曾漏掉这步,da 18 万
    像素只有 90 个落进 30m 窗口(世界系坐标当局部系用,远处路面全在窗外)。
    """
    a = math.radians(ego[3])
    c, s = math.cos(a), math.sin(a)
    ex, ey = ego[0], ego[1]
    w, h = bev.shape[1], bev.shape[0]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return
    # 下采样到最多 40k 像素(速度;投影是逐像素射线)
    if len(xs) > 40_000:
        idx = np.random.default_rng(0).choice(len(xs), 40_000, replace=False)
        xs, ys = xs[idx], ys[idx]
    for u, v in zip(xs, ys, strict=True):
        g = ground_intersection(world_cam, intrinsics, float(u), float(v), ground_z)
        if g is None:
            continue
        dx, dy = g[0] - ex, g[1] - ey
        lx, ly = c * dx + s * dy, -s * dx + c * dy  # 世界 → ego 局部系
        bx, by = bev_to_px(lx, ly, w, h)
        if 0 <= bx < w and 0 <= by < h:
            bev[by, bx] = color


def letterbox_sq(img_bgr: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, float, float, float]:
    """yolov5 letterbox:等比缩放 + 中间填充到 imgsz×imgsz(正方形,stride 32 对齐)。

    官方 trace 是 640×640 方形输入;非方形直接 resize 会打乱 anchor grid 尺寸
    (实测 640×360 报 "Got 25 and 24 in dimension 2")。返回 (图, ratio, (dw, dh))。
    """
    h, w = img_bgr.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (imgsz - new_w) / 2, (imgsz - new_h) / 2
    img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, dw, dh


def scale_coords_like(
    boxes: np.ndarray, ratio: float, dw: float, dh: float, orig_shape: tuple[int, int]
) -> np.ndarray:
    """letterbox 反变换:模型输出框 → 原图坐标。"""
    oh, ow = orig_shape
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ow)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, oh)
    return boxes


def yolopv2_predict(model, img_bgr: np.ndarray, device: torch.device, imgsz: int = 640):
    """YOLOPv2 推理(官方 demo.py 同式),返回 (det [N,6], da_mask, ll_mask)。

    img_bgr 原图;内部 letterbox 到 imgsz×imgsz;输出 da/ll = 原图分辨率二值掩膜。
    """
    h, w = img_bgr.shape[:2]
    img, ratio, dw, dh = letterbox_sq(img_bgr, imgsz)
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)
    with torch.no_grad():
        (pred, anchor_grid), seg, ll = model(img)
    # NMS(yolov5 系;参考官方 demo split_for_trace_model + non_max_suppression)
    from utils.utils import non_max_suppression, split_for_trace_model  # noqa: E402  # type: ignore

    pred = split_for_trace_model(pred, anchor_grid)
    dets = non_max_suppression(pred, 0.25, 0.45)[0]
    dets = dets.cpu().numpy() if dets is not None else np.zeros((0, 6))
    if len(dets):
        dets[:, :4] = scale_coords_like(dets[:, :4], ratio, dw, dh, (h, w))
    # 掩膜:seg[:, :, 12:372, :] 可行驶 2 类;ll 车道线 1 类。
    # **seg/ll 输出已是 640×640**(= letterbox 输入分辨率,官方 demo 的 scale_factor=2
    # 是给 1280×720 显示用的,不是模型输出尺寸)——直接 argmax/round,勿再 x2。
    da = seg.argmax(1).squeeze().cpu().numpy()
    llm = torch.round(ll).squeeze().cpu().numpy()
    # 掩膜在 640×640 letterbox 系 → 裁剪去掉 padding → 缩回原图
    y0, y1 = int(dh), int(dh + h * ratio)
    x0, x1 = int(dw), int(dw + w * ratio)
    da = da[y0:y1, x0:x1]
    llm = llm[y0:y1, x0:x1]
    if (h, w) != da.shape:
        da = cv2.resize(da.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        llm = cv2.resize(llm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return dets, da.astype(bool), llm.astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="outputs/surround_train",
        help="surround 数据根(calib.json + cam_*/*.png + ego_pose.json)",
    )
    ap.add_argument("--out", default="outputs/sem_bev", help="输出根")
    ap.add_argument("--frames", default="0-20", help="帧范围 0-20 或逗号列表")
    ap.add_argument(
        "--gpu", action="store_true", help="用 GPU(autodrivedata env torch 无 CUDA 驱动,默认 CPU)"
    )
    ap.add_argument("--imgsz", type=int, default=640, help="YOLOPv2 推理尺寸")
    args = ap.parse_args()

    root = Path(args.root)
    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if YOLO is None:
        raise SystemExit("ultralytics 未装(autodrivedata env)")

    # 解析帧范围
    frames: list[int] = []
    for tok in args.frames.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            frames.extend(range(int(a), int(b) + 1))
        else:
            frames.append(int(tok))

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"[model] YOLOPv2({args.imgsz}) + YOLO11s-seg on {device}")

    # 模型
    yolo11 = YOLO(str(project_path("models/yolo11s-seg.pt")))  # 权重落点 = models/
    # TorchScript archive:autodrivedata env torch 无 CUDA 驱动,jit.load 会做 CUDA 探测
    # 失败(驱动 12.4 vs torch cu130)→ 用 torch.load(weights_only=False) 直接载权重图。
    ckpt = torch.load(str(project_path("outputs/models/yolopv2.pt")), map_location="cpu", weights_only=False)
    yolopv2 = ckpt.to(device).float().eval()

    # 标定 + 姿态
    calib = json.loads((root / "calib.json").read_text())
    ego_poses = json.loads((root / "ego_pose.json").read_text())
    cams = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    cam_dirs = {c: c.lower().replace("cam_", "cam_") for c in cams}  # cam_front ...
    cam_dirs = {c: c.lower() for c in cams}

    # 图尺寸(从第一张图读)
    first_img = next((root / cam_dirs[cams[0]]).glob("*.png"))
    size = (int(cv2.imread(str(first_img)).shape[1]), int(cv2.imread(str(first_img)).shape[0]))
    print(f"[data] {root} | {len(frames)} 帧 × {len(cams)} 相机 | img {size[0]}x{size[1]}")

    for fid in frames:
        token = f"{fid:06d}"
        ep = ego_poses[token] if token in ego_poses else ego_poses[fid]
        ego = [ep["x"], ep["y"], ep["z"], ep["yaw"], ep["pitch"], ep["roll"]]
        bev_h = int((BEV_Y[1] - BEV_Y[0]) / BEV_PX)
        bev_w = int((BEV_X[1] - BEV_X[0]) / BEV_PX)
        bev = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
        ground_z = ego[2] - GROUND_Z_OFF

        panels = {}
        for cam in cams:
            d = cam_dirs[cam]
            p = root / d / f"{token}.png"
            if not p.exists():
                continue
            img = cv2.imread(str(p))
            intrinsics, se = init_camera(calib[cam], size)
            world_cam = cam_pose(ego, se)
            # YOLOPv2:da + ll
            _, da, llm = yolopv2_predict(yolopv2, img, device, args.imgsz)
            # YOLO11:实例掩膜(车/人/卡车/巴士)
            res = yolo11.predict(img, conf=0.35, verbose=False)[0]
            obj_mask = np.zeros(size[::-1], dtype=bool)
            if res.masks is not None and res.boxes is not None:
                cls = res.boxes.cls.cpu().numpy()
                for i, c in enumerate(cls):
                    if int(c) in DETECT_CLS:
                        m = res.masks.data[i].cpu().numpy()
                        if m.shape != size[::-1]:
                            m = cv2.resize(m, (size[0], size[1]), interpolation=cv2.INTER_NEAREST)
                        obj_mask |= m > 0.5
            project_mask_to_bev(da, world_cam, intrinsics, ground_z, ego, bev, DA_COLOR)
            project_mask_to_bev(llm, world_cam, intrinsics, ground_z, ego, bev, LL_COLOR)
            project_mask_to_bev(obj_mask, world_cam, intrinsics, ground_z, ego, bev, OBJ_COLOR)
            # 预览面板(每相机分割原图 + 掩膜投影)
            h, w = size[1], size[0]
            panel = np.hstack(
                [
                    cv2.resize(img, (w, h)),
                    cv2.cvtColor(cv2.resize((da * 255).astype(np.uint8), (w, h)), cv2.COLOR_GRAY2BGR),
                ]
            )
            panels[cam] = panel

        bev_path = out / f"bev_{token}.png"
        cv2.imwrite(str(bev_path), bev)
        print(f"[{token}] bev 写出 {bev_path}")

        # 全览:上 6 相机面板 + 下 BEV
        cell = 480
        row = []
        for cam in cams:
            if cam in panels:
                panel = panels[cam]
                row.append(cv2.resize(panel, (cell, cell)))
            else:
                row.append(np.zeros((cell, cell, 3), dtype=np.uint8))
        top = np.hstack(row)
        bottom = cv2.resize(bev, (cell * len(cams), cell * len(cams) * bev.shape[0] // bev.shape[1]))
        comp = np.vstack([top, bottom])
        comp_path = out / f"panel_{token}.png"
        cv2.imwrite(str(comp_path), comp)
    print(f"[done] 语义 BEV → {out}")


if __name__ == "__main__":
    main()

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

from autodrivedata.map.mapviz import BEV_X, BEV_Y, CameraIntrinsics, cam_pose, intrinsics_from_k
from autodrivedata.utils import runlog
from autodrivedata.utils.geometry import camera_rotation_world_to_cam  # noqa: F401
from autodrivedata.utils.paths import project_path

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


#: 逐像素地面投影的默认像素上限(速度;投影是逐像素射线)。**只在画图时用** ——
#: 判据(`sem_eval`)必须传 `None` 走全量:随机子采样会让 IoU 带一层抽样噪声,
#: 而 GT 与预测的子集还各不相同(`np.where` 出来的顺序不同),那个噪声不会被抵消。
MAX_PROJECT_PIXELS = 40_000


def mask_to_bev(
    mask: np.ndarray,
    world_cam: tuple[tuple[float, float, float], tuple[float, float, float]],
    intrinsics: CameraIntrinsics,
    ground_z: float,
    ego: list[float],
    shape: tuple[int, int],
    *,
    max_pixels: int | None = MAX_PROJECT_PIXELS,
) -> np.ndarray:
    """掩膜像素(非 0 = 命中)→ **BEV 二值图** `(h, w)` bool(像素级地面投影)。

    ground_intersection 返回**世界系**交点,BEV 面板是 **ego 局部系**(x 前向 /
    y 左向,原点 = ego)→ 投影前先按 ego yaw 旋转回局部系。曾漏掉这步,da 18 万
    像素只有 90 个落进 30m 窗口(世界系坐标当局部系用,远处路面全在窗外)。

    这里是**唯一的投影实现**:画图(`project_mask_to_bev`)与判据(`sem_eval`)都走它
    —— 两处各写一份的话,"图看着对"与"数算得对"会各自成立、合起来错。

    **向量化实现**:逐像素调 `ground_intersection` 在判据规模上跑不动 ——
    一帧 6 路可行驶区就有 **360 万**像素,20 帧近 7200 万次 Python 调用。
    这里把同一条链整块用 numpy 算(与标量版**逐位等价**,`test_sem_bev` 有对拍钉):
    归一化 → 相机→世界方向 → 与地平面求交 → 世界→ego 局部 → 栅格化。
    """
    out = np.zeros(shape, dtype=bool)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return out
    if max_pixels is not None and len(xs) > max_pixels:
        idx = np.random.default_rng(0).choice(len(xs), max_pixels, replace=False)
        xs, ys = xs[idx], ys[idx]

    loc, rot = world_cam
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    # 归一化平面 → 相机系方向(与 `ground_intersection` 同一步)
    dir_cam = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs, dtype=np.float64)], axis=1)
    # 相机 → 世界。标量版是 `R_cw @ dir_cam`,其中 `R_cw = R_wc.T`
    # ⇒ 批量形式是 `dir_cam @ R_cw.T` = `dir_cam @ R_wc`。
    dir_world = dir_cam @ camera_rotation_world_to_cam(rot)
    dz = dir_world[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ground_z - loc[2]) / dz
    keep = (dz < 0) & (t > 0)  # 射线上行 / 相机后:与标量版的 None 同判
    if not keep.any():
        return out
    p = np.asarray(loc, dtype=np.float64)[None, :] + t[keep, None] * dir_world[keep]

    a = math.radians(ego[3])
    c, s = math.cos(a), math.sin(a)
    dx, dy = p[:, 0] - ego[0], p[:, 1] - ego[1]
    lx, ly = c * dx + s * dy, -s * dx + c * dy  # 世界 → ego 局部系
    w, h = shape[1], shape[0]
    # `bev_to_px` 的批量形式:同样的 `int()`(向零截断)语义 ⇒ 边界行为与标量版一致
    bx = ((lx - BEV_X[0]) / (BEV_X[1] - BEV_X[0]) * w).astype(np.int64)
    by = ((BEV_Y[1] - ly) / (BEV_Y[1] - BEV_Y[0]) * h).astype(np.int64)
    inb = (bx >= 0) & (bx < w) & (by >= 0) & (by < h)
    out[by[inb], bx[inb]] = True
    return out


def project_mask_to_bev(
    mask: np.ndarray,
    world_cam: tuple[tuple[float, float, float], tuple[float, float, float]],
    intrinsics: CameraIntrinsics,
    ground_z: float,
    ego: list[float],
    bev: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    """`mask_to_bev` 的**画图**封装:命中处涂 `color`(判据走前者,不走这里)。"""
    bev[mask_to_bev(mask, world_cam, intrinsics, ground_z, ego, bev.shape[:2])] = color


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


def yolopv2_predict(model, img_bgr: np.ndarray, device: torch.device, imgsz: int = 640):
    """YOLOPv2 推理,返回 (da_mask, ll_mask) —— 可行驶区 / 车道线,均为原图分辨率二值掩膜。

    img_bgr 原图;内部 letterbox 到 imgsz×imgsz。

    ⚠️ **2026-09-28 删掉一段从未执行过的死代码路径**(它卡死了整个能力面):
    本函数原本还返回检测框 `det`,靠 `from utils.utils import non_max_suppression,
    split_for_trace_model`(YOLOPv2 官方仓库自带的包)做后处理。该包本机不存在,
    而**唯一调用点写的是 `_, da, llm = ...` —— 返回值从来没被消费过**。
    症状:9-27 与 9-28 两次运行都抛 `ModuleNotFoundError: No module named 'utils'`,
    整个语义 BEV 出不了图(两次 runlog 都如实记了,tests/ 里却**没有任何用例覆盖本模块**,
    因此没有一处会报红)。检测框本就走下面的 YOLO11s-seg,故整段删除 ——
    `model(img)` 只取 seg/ll 两路输出,掩膜不需要 NMS。
    """
    h, w = img_bgr.shape[:2]
    img, ratio, dw, dh = letterbox_sq(img_bgr, imgsz)
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)
    with torch.no_grad():
        _, seg, ll = model(img)
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
    return da.astype(bool), llm.astype(bool)


def yolo11_object_mask(yolo11, img_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """YOLO11s-seg → **障碍物实例掩膜**(只留 `DETECT_CLS` 那四类),原分辨率二值。

    与 `yolopv2_predict` 并列抽出来,是为了让判据(`sem_eval`)与出图(`sem_bev`)
    **吃同一份预测** —— 各跑一遍模型的话,报出的 mIoU 与图上看到的可能不是同一次推理。
    """
    res = yolo11.predict(img_bgr, conf=0.35, verbose=False)[0]
    mask = np.zeros(size[::-1], dtype=bool)
    if res.masks is None or res.boxes is None:
        return mask
    cls = res.boxes.cls.cpu().numpy()
    for i, c in enumerate(cls):
        if int(c) in DETECT_CLS:
            m = res.masks.data[i].cpu().numpy()
            if m.shape != size[::-1]:
                m = cv2.resize(m, (size[0], size[1]), interpolation=cv2.INTER_NEAREST)
            mask |= m > 0.5
    return mask


def predict_masks(
    img_bgr: np.ndarray,
    yolopv2,
    yolo11,
    device: torch.device,
    size: tuple[int, int],
    imgsz: int = 640,
) -> dict[str, np.ndarray]:
    """一帧一相机 → `{drivable, lane, obstacle}` 三个**原分辨率布尔掩膜**。

    键名与 GT 侧(`perception.sem_tags.GT_CLASSES`)**逐字相同** —— 判据两侧靠它对上,
    改名就会静默变成"三类全空、mIoU 恒 1"(见 `sem_tags` 头注那张映射表)。
    """
    h, w = img_bgr.shape[:2]
    assert (w, h) == size, f"图像 {w}×{h} 与 size {size} 不符"
    da, llm = yolopv2_predict(yolopv2, img_bgr, device, imgsz)
    return {"drivable": da, "lane": llm, "obstacle": yolo11_object_mask(yolo11, img_bgr, size)}


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
    ap.add_argument("--no-runlog", action="store_true", help="不落 logs/ 三件套(默认每次运行都落)")
    args = ap.parse_args()

    with runlog.run("autodrivedata.perception.sem_bev") as rl:
        rl.input(args.root, "nus-root")
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
        yolo11 = YOLO(
            str(project_path("weights/yolo11s-seg.pt"))
        )  # 权重落点 = weights/(同目录另有 kitti3d_finetune/)
        # TorchScript archive:autodrivedata env torch 无 CUDA 驱动,jit.load 会做 CUDA 探测
        # 失败(驱动 12.4 vs torch cu130)→ 用 torch.load(weights_only=False) 直接载权重图。
        ckpt = torch.load(
            str(project_path("outputs/models/yolopv2.pt")), map_location="cpu", weights_only=False
        )
        yolopv2 = ckpt.to(device).float().eval()

        # 标定 + 姿态
        calib = json.loads((root / "calib.json").read_text())
        ego_poses = json.loads((root / "ego_pose.json").read_text())
        cams = [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
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
                # 三类掩膜(与判据 `sem_eval` 走**同一个** `predict_masks`)
                pred = predict_masks(img, yolopv2, yolo11, device, size, args.imgsz)
                da = pred["drivable"]
                project_mask_to_bev(da, world_cam, intrinsics, ground_z, ego, bev, DA_COLOR)
                project_mask_to_bev(pred["lane"], world_cam, intrinsics, ground_z, ego, bev, LL_COLOR)
                project_mask_to_bev(pred["obstacle"], world_cam, intrinsics, ground_z, ego, bev, OBJ_COLOR)
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
        # 目录产物:记文件数 + 总字节(逐张哈希没有意义,BEV 是中间可观测量)
        rl.highlight("n_frames", len(frames))
        rl.highlight("n_cams", len(cams))
        rl.highlight("imgsz", args.imgsz)
        rl.highlight("device", str(device))
        rl.artifact_dir(out, "sem-bev")


if __name__ == "__main__":
    main()

"""MapTR 预测回投目检:预测/GT BEV 折线 → 6 相机图像 overlay + BEV 面板。

预测折线(模型 BEV 输出系 = ego 局部系,x 前向 / y 左向,z=0)→ 世界 → 各相机
像素,画到原图上:pred 品红 / GT 青绿(路面场景罕见色,C23 撞色口径避让)。
投影链与 B3 探针同式(ego→世界旋转平移 + calib.world_to_img,内参从 infos 直读,
fov 由 fx 反推,不硬编码)。每帧一张拼图:6 相机 3×2 + BEV 面板(窗口同模型
输出系 x∈[−15,15] / y∈[−30,30])。

用法:
  python bin/viz_maptr_pred.py --infos outputs/surround_train/map_infos.json \
      --root outputs/surround_train --ckpt outputs/maptr_400.pt \
      --start 200 --frames 6 --out-dir outputs/viz_maptr
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from autodrivedata.calib import CameraIntrinsics, world_to_img
from maptr_impl.dataset import MAPTR_CLASSES, MapTRDataset
from maptr_impl.model import MapTR

_PRED_COLOR = (255, 0, 255)  # 品红:路面场景罕见
_GT_COLOR = (0, 255, 255)  # 青绿:同罕见(植被绿与其可区分)
_CAM_SCALE = 0.5  # 相机图 1242×375 → 621×187 拼图
_BEV_W, _BEV_H = 420, 420  # BEV 面板像素


def _ego_to_world(pts: list[tuple[float, float, float]], eg: list[float]) -> list[tuple[float, float, float]]:
    """ego 局部系 → CARLA 世界(与 B3 probe_mapvec_proj 同式)。"""
    a = math.radians(eg[3])
    c, s = math.cos(a), math.sin(a)
    return [(c * x - s * y + eg[0], s * x + c * y + eg[1], z + eg[2]) for x, y, z in pts]


def _cam_pose(
    eg: list[float], se: list[float]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """ego2global + sensor2ego → 相机世界位姿(位置, (pitch, yaw, roll) 度)。"""
    yaw_e = math.radians(eg[3])
    loc = (eg[0] + math.cos(yaw_e) * se[0], eg[1] + math.sin(yaw_e) * se[0], eg[2] + se[2])
    return loc, (0.0, eg[3] + se[3], 0.0)


def _intrinsics_from_k(k: list[list[float]], size: tuple[int, int]) -> CameraIntrinsics:
    """infos 内参 3×3 → CameraIntrinsics(fov 由 fx 反推,不硬编码)。"""
    w, h = size
    fx = k[0][0]
    fov_h = math.degrees(2.0 * math.atan((w / 2.0) / fx))
    return CameraIntrinsics(width=w, height=h, fov_h_deg=fov_h)


def _project_lines(
    lines: list[np.ndarray], eg: list[float], cam: dict, intrinsics: CameraIntrinsics
) -> list[list[tuple[float, float]]]:
    """折线列表(ego 系)→ [(u, v) 段列表](段 = 连续可见点的相邻对)。"""
    loc, rot = _cam_pose(eg, cam["sensor2ego"])
    segs: list[list[tuple[float, float]]] = []
    for line in lines:
        world = _ego_to_world([(float(p[0]), float(p[1]), 0.0) for p in line], eg)
        uv = [world_to_img(p, loc, rot, intrinsics) for p in world]
        run: list[tuple[float, float]] = []
        for q in uv:
            if q is None:
                if len(run) >= 2:
                    segs.append(run)
                run = []
            else:
                run.append(q)
        if len(run) >= 2:
            segs.append(run)
    return segs


def _bev_panel(preds: list[list[np.ndarray]], gts: list[list[np.ndarray]], title: str) -> Image.Image:
    """BEV 面板:pred 品红 / GT 青绿,窗口同模型输出系(x∈[−15,15], y∈[−30,30])。"""
    img = Image.new("RGB", (_BEV_W, _BEV_H), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    def px(x: float, y: float) -> tuple[float, float]:
        u = (x + 15.0) / 30.0 * _BEV_W
        v = (30.0 - y) / 60.0 * _BEV_H  # y 左向,图上向上
        return u, v

    for gts_c in gts:
        for g in gts_c:
            draw.line([q for p in g for q in px(p[0], p[1])], fill=_GT_COLOR, width=1)
    for preds_c in preds:
        for p in preds_c:
            draw.line([q for pt in p for q in px(pt[0], pt[1])], fill=_PRED_COLOR, width=1)
    draw.rectangle([(0, 0), (_BEV_W - 1, _BEV_H - 1)], outline=(120, 120, 120))
    draw.text((4, 2), title, fill=(255, 255, 255))
    return img


def _collage(cells: list[tuple[str, Image.Image]], bev: Image.Image) -> Image.Image:
    """6 相机 3×2 + BEV 面板拼图(相机图缩 0.5)。"""
    names = [n for n, _ in cells]
    imgs = {n: i.resize((int(i.width * _CAM_SCALE), int(i.height * _CAM_SCALE))) for n, i in cells}
    cw, ch = next(iter(imgs.values())).size
    canvas = Image.new("RGB", (3 * cw + _BEV_W, max(2 * ch, _BEV_H)), (40, 40, 40))
    for idx, n in enumerate(names):
        r, c = divmod(idx, 3)
        canvas.paste(imgs[n], (c * cw, r * ch))
    canvas.paste(bev, (3 * cw, 0))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True, help="B2 组装 infos json")
    ap.add_argument("--root", required=True, help="图像根目录")
    ap.add_argument("--ckpt", required=True, help="train_maptr.py 输出的 state_dict")
    ap.add_argument("--frames", type=int, default=6, help="可视化帧数")
    ap.add_argument("--start", type=int, default=0, help="起始帧")
    ap.add_argument("--score-thr", type=float, default=0.2, help="实例得分阈值(sigmoid)")
    ap.add_argument("--device", default=None, help="推理设备(默认 cuda 若可用)")
    ap.add_argument("--out-dir", required=True, help="拼图输出目录(每帧一张 png)")
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    infos = json.loads(Path(args.infos).read_text(encoding="utf-8"))
    frames = list(range(args.start, min(args.start + args.frames, len(infos))))
    ds = MapTRDataset(infos, args.root, frames=frames)
    print(f"[data] {len(frames)} 帧 × {len(ds.cam_names)} 相机,设备 {dev}")

    model = MapTR().to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()
    print(f"[model] {args.ckpt} 载入完成")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_names = sorted(ds.cam_names)
    with torch.no_grad():
        for i, item in enumerate(ds):
            t0 = time.perf_counter()
            images = {n: t[None].to(dev) for n, t in item["images"].items()}
            out, _ = model(images, item["pose"][None].to(dev), ds.calibs)
            logits = out["pred_logits"][0].float()
            pts = out["pred_points"][0].float().cpu().numpy()
            scores = torch.sigmoid(logits).cpu().numpy()
            preds = []  # 逐类:得分过滤后的折线列表(ego 系)
            for c in range(len(MAPTR_CLASSES)):
                idx = slice(c * model.num_vec, (c + 1) * model.num_vec)
                keep = scores[idx, c + 1] > args.score_thr
                preds.append(list(pts[idx][keep]))
            n_pred = sum(len(p) for p in preds)
            n_gt = sum(len(g) for g in item["gt"])

            eg = infos[frames[i]]["ego2global"]
            info_cams = infos[frames[i]]["cams"]
            cells: list[tuple[str, Image.Image]] = []
            for name in cam_names:
                cam = info_cams[name]
                img = Image.open(ds.root / cam["data_path"]).convert("RGB")
                k = _intrinsics_from_k(cam["intrinsic"], img.size)
                draw = ImageDraw.Draw(img)
                all_preds = [p for cls in preds for p in cls]
                for seg in _project_lines(all_preds, eg, cam, k):
                    draw.line([q for p in seg for q in p], fill=_PRED_COLOR, width=3)
                all_gts = [g for cls in item["gt"] for g in cls]
                for seg in _project_lines(all_gts, eg, cam, k):
                    draw.line([q for p in seg for q in p], fill=_GT_COLOR, width=3)
                cells.append((name, img))

            canvas = _collage(
                cells, _bev_panel(preds, item["gt"], f"frame {frames[i]} pred {n_pred} / gt {n_gt}")
            )
            path = out_dir / f"frame_{frames[i]:04d}.png"
            canvas.save(path)
            print(f"[viz] {path.name} pred {n_pred} / gt {n_gt} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()

"""MapTR 预测回投目检:预测/GT BEV 折线 → 6 相机图像 overlay + BEV 面板。

预测折线(模型 BEV 输出系 = ego 局部系,x 前向 / y 左向,z=0)→ 世界 → 各相机
像素,画到原图上:pred 品红 / GT 青绿(路面场景罕见色,C23 撞色口径避让)。
投影/绘制走纯值模块 [autodrivedata/map/mapviz.py](mapviz.py)——与实时流
[autodrivedata/sim/view_stream.py](../sim/view_stream.py) 是同一条链(单一投影实现),旋转单位为
**弧度**(单位口径的坑见 mapviz docstring)。每帧一张拼图:6 相机 3×2 + BEV 面板
(窗口同模型输出系 x∈[−15,15] / y∈[−30,30])。

用法:
  python -m autodrivedata.map.viz_maptr_pred --infos outputs/surround_train/map_infos.json \
      --root outputs/surround_train --ckpt outputs/maptr_400.pt \
      --start 200 --frames 6 --out-dir outputs/viz_maptr
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from autodrivedata.map.maptr.dataset import MAPTR_CLASSES, MapTRDataset
from autodrivedata.map.maptr.variants import load_map_model
from autodrivedata.map.mapviz import (
    GT_COLOR,
    PRED_COLOR,
    bev_panel,
    cam_pose,
    draw_projected_lines,
    intrinsics_from_k,
)
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

_CAM_SCALE = 0.5  # 相机图 1242×375 → 621×187 拼图
_BEV_W, _BEV_H = 420, 420  # BEV 面板像素


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
    ap.add_argument("--no-runlog", action="store_true", help="不落 logs/ 三件套(默认每次运行都落)")
    args = ap.parse_args()

    with runlog.run("autodrivedata.map.viz_maptr_pred") as rl:
        # 目检图的**结论全靠"哪份权重 + 哪个阈值"** —— 同一帧换阈值图就换个样
        rl.input(args.ckpt, "ckpt")
        rl.input(args.infos, "infos")
        rl.input(args.root, "root")
        dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        infos = json.loads(Path(args.infos).read_text(encoding="utf-8"))
        frames = list(range(args.start, min(args.start + args.frames, len(infos))))
        ds = MapTRDataset(infos, args.root, frames=frames)
        print(f"[data] {len(frames)} 帧 × {len(ds.cam_names)} 相机,设备 {dev}")

        # 结构由 checkpoint 自带(变体/参数);显式指定会拿错结构,故不给开关
        model, meta = load_map_model(args.ckpt, dev)
        model.eval()
        print(f"[model] {args.ckpt} 载入完成(变体 {meta['name']})")

        out_dir = project_path(args.out_dir)
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
                n_seg = 0
                for name in cam_names:
                    cam = info_cams[name]
                    img = Image.open(ds.root / cam["data_path"]).convert("RGB")
                    k = intrinsics_from_k(cam["intrinsic"], img.size)
                    pose = cam_pose(eg, cam["sensor2ego"])
                    draw = ImageDraw.Draw(img)
                    all_preds = [p for cls in preds for p in cls]
                    n_seg += draw_projected_lines(draw, all_preds, eg, pose, k, color=PRED_COLOR)
                    all_gts = [g for cls in item["gt"] for g in cls]
                    draw_projected_lines(draw, all_gts, eg, pose, k, color=GT_COLOR)
                    cells.append((name, img))

                canvas = _collage(
                    cells,
                    bev_panel(
                        preds, item["gt"], f"frame {frames[i]} pred {n_pred} / gt {n_gt}", (_BEV_W, _BEV_H)
                    ),
                )
                path = out_dir / f"frame_{frames[i]:04d}.png"
                canvas.save(path)
                print(
                    f"[viz] {path.name} pred {n_pred} / gt {n_gt} / 段 {n_seg} ({time.perf_counter() - t0:.1f}s)"
                )
                # 逐帧一行:`n_seg`(投到画面上的**线段数**)是"图看着画出来了"与
                # "投影真落了位"的分界 —— 前者不构成投影正确的证据,后者才是。
                rl.metric(frames[i], frame=frames[i], n_pred=n_pred, n_gt=n_gt, n_seg=n_seg)
        rl.highlight("n_frames", len(frames))
        rl.highlight("score_thr", args.score_thr)
        rl.highlight("start", args.start)
        rl.artifact_dir(out_dir, "viz-collage")


if __name__ == "__main__":
    main()

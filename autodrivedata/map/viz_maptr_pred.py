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

from autodrivedata.calib.camera_rig import camera_grid_order
from autodrivedata.gt.export.nuscenes import NUS_RADAR_CHANNELS
from autodrivedata.map import bev_base
from autodrivedata.map.maptr.dataset import MAPTR_CLASSES, MapTRDataset
from autodrivedata.map.maptr.variants import load_map_model, read_map_meta
from autodrivedata.map.mapviz import (
    BEV_X,
    BEV_Y,
    GT_COLOR,
    PRED_COLOR,
    bev_panel,
    bev_px_transform,
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
    ap.add_argument(
        "--bev-pair",
        action="store_true",
        help="每帧**额外**落两张纯 BEV:`bev/frame_XXXX_pred.png`(只画预测)/ `_gt.png`(只画真值)。拼图里两者是同图叠加的,分开才能逐帧看「谁多谁少」",
    )
    ap.add_argument(
        "--lidar-root",
        default=None,
        help="LiDAR/Radar 点云根目录(**与 --root 同一段数据**)⇒ BEV 面板加一层底图。"
        "⚠️ 底图只是给人看的上下文:模型是纯相机的,点云不进网络",
    )
    ap.add_argument(
        "--no-radar-base", action="store_true", help="底图只画 LiDAR 不画雷达(雷达点稀疏,有时反而碍眼)"
    )
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
        # ★ 时序窗口**从 checkpoint 自带的元信息读**,不给 CLI 开关:窗口是**结构**的一部分
        # (fusion 模块按它构造),调用方指定错了就是"拿单帧喂时序模型"。
        # 2026-09-29 实测:原先恒 window=1,按时序权重会**响亮报错**
        # `模型是时序版(window=3),但只收到单帧图像` —— 好过静默跑错,但 viz 因此出不了图。
        window = int(read_map_meta(args.ckpt).get("model_kwargs", {}).get("temporal_window", 1) or 1)
        ds = MapTRDataset(infos, args.root, frames=frames, window=window)
        if ds.dropped:
            print(f"[data] 窗口 {window} 丢弃 {len(ds.dropped)} 帧(前驱不在本次帧集内)")
        print(f"[data] {len(ds)} 帧 × {len(ds.cam_names)} 相机,设备 {dev},时序窗口 {window}")

        # 结构由 checkpoint 自带(变体/参数);显式指定会拿错结构,故不给开关
        model, meta = load_map_model(args.ckpt, dev)
        model.eval()
        print(f"[model] {args.ckpt} 载入完成(变体 {meta['name']})")

        out_dir = project_path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # 画布行序走**唯一取序入口**(`camera_rig.camera_grid_order`)。原先这里是
        # `sorted(ds.cam_names)` —— 字母序把后三路排到了 `_collage` 的**第一行**、
        # 前一行挤到第二行,且每行内部左右也是反的(2026-09-28 实测)。
        cam_names = camera_grid_order(ds.cam_names)
        with torch.no_grad():
            for i, item in enumerate(ds):
                t0 = time.perf_counter()
                # window=1 → images 是字典、pose 是单帧;window=K>1 → images 是**列表**(旧 → 新 K 帧)、
                # pose 是 `poses`。两种契约都要认(`eval_maptr` 同款分支 —— 那是唯一的另一处消费方)。
                if isinstance(item["images"], list):
                    images = [{n: t[None].to(dev) for n, t in f.items()} for f in item["images"]]
                    pose_in = item["poses"][None].to(dev)
                else:
                    images = {n: t[None].to(dev) for n, t in item["images"].items()}
                    pose_in = item["pose"][None].to(dev)
                out, _ = model(images, pose_in, ds.calibs)
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

                # **必须走 `ds.infos[i]` 而不是 `infos[frames[i]]`**:window>1 时
                # `history_windows` 会丢掉拿不到完整历史的帧,`ds` 的下标与 `frames` 就**错位**了
                # (实测:窗口 3 且只取 3 帧时,唯一留下的样本是第 3 帧,按 frames[i] 取会画错帧)。
                # `ds.infos[i]` 两种窗口下都是该样本自己的帧记录。
                rec = ds.infos[i]
                fid = int(rec["frame"])
                # 底图(**必须在同帧的 ego 系里取**):`fid` 是数据集的帧号,不是 `ds` 的下标 ——
                # window>1 时两者会错位(同上面 `ds.infos[i]` 那条)。
                base = None
                n_lid = n_rad = 0
                if args.lidar_root is not None:
                    lid_B, rad_B = bev_base.frame_points(
                        project_path(args.lidar_root),
                        fid,
                        None if args.no_radar_base else list(NUS_RADAR_CHANNELS),
                    )
                    base, bst = bev_base.base_layer(
                        lid_B,
                        rad_B,
                        bev_px_transform((_BEV_W, _BEV_H)),
                        (_BEV_W, _BEV_H),
                        (BEV_X, BEV_Y),
                    )
                    n_lid, n_rad = bst["n_lidar_drawn"], bst["n_radar_drawn"]
                    if n_lid == 0:
                        # 0 点是**判据**不是噪声:要么 --lidar-root 不是同一段数据,要么窗口/换算错了
                        print(
                            f"  [warn] 帧 {fid} 底图 0 点(总点数 {bst['n_lidar_total']})—— 检查 --lidar-root"
                        )
                eg = rec["ego2global"]
                info_cams = rec["cams"]
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
                        preds,
                        item["gt"],
                        f"frame {fid} pred {n_pred} / gt {n_gt}",
                        (_BEV_W, _BEV_H),
                        base=base,
                    ),
                )
                path = out_dir / f"frame_{fid:04d}.png"
                canvas.save(path)
                if args.bev_pair:
                    # 纯 BEV 各一张:同一份 preds / gt,分别关掉另一半。
                    # `bev_panel` 对空列表安全(`for gc in gts or []`),不必加特判。
                    bd = out_dir / "bev"
                    bd.mkdir(parents=True, exist_ok=True)
                    empty: list[list] = []
                    bev_panel(preds, empty, f"PRED  frame {fid}", (_BEV_W, _BEV_H), base=base).save(
                        bd / f"frame_{fid:04d}_pred.png"
                    )
                    bev_panel(empty, item["gt"], f"GT  frame {fid}", (_BEV_W, _BEV_H), base=base).save(
                        bd / f"frame_{fid:04d}_gt.png"
                    )
                print(
                    f"[viz] {path.name} pred {n_pred} / gt {n_gt} / 段 {n_seg}"
                    f" / 底图 L{n_lid} R{n_rad} ({time.perf_counter() - t0:.1f}s)"
                )
                # 逐帧一行:`n_seg`(投到画面上的**线段数**)是"图看着画出来了"与
                # "投影真落了位"的分界 —— 前者不构成投影正确的证据,后者才是。
                rl.metric(fid, frame=fid, n_pred=n_pred, n_gt=n_gt, n_seg=n_seg)
        rl.highlight("n_frames", len(ds))
        rl.highlight("score_thr", args.score_thr)
        rl.highlight("start", args.start)
        rl.artifact_dir(out_dir, "viz-collage")


if __name__ == "__main__":
    main()

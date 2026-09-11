"""MapTR D 阶段评估:训练权重 → 逐帧推理 → 四类 chamfer AP(MapTR 官方口径)。

评估口径与官方 MapTR map 评测同构:全部评估帧的预测/GT 按类**跨帧汇聚**后做
一次一对一匹配(而非逐帧 AP 平均);AP = 阈值 {0.5, 1.0, 1.5}m 的 precision
均值(autodrivedata.chamfer_ap)。解码:每类 query 取 sigmoid 得分 > --score-thr
的实例(默认 0.2,官方 nuscenes 惯例;阈值可扫)。

用法:
  python bin/eval_maptr.py --infos outputs/surround_train/map_infos.json \
      --root outputs/surround_train --ckpt outputs/maptr.pt --frames 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from autodrivedata.chamfer_ap import chamfer_ap_per_class, chamfer_cost_matrix
from maptr_impl.chamfer_gpu import chamfer_cost_matrix_cuda
from maptr_impl.dataset import MAPTR_CLASSES, MapTRDataset
from maptr_impl.model import MapTR


def _dump_preds(
    path: str,
    infos: str,
    ckpt: str,
    thr: float,
    preds_by_class: list[list[np.ndarray]],
    scores_by_class: list[list[float]],
    gts_by_class: list[list[np.ndarray]],
) -> None:
    """预测产物落盘:json(供 AutoLabel 消费)+ BEV png(目检,红=pred 绿=GT)。

    跨帧汇聚口径(与评估一致,不含帧归属);BEV 窗口与模型输出同系:
    x∈[-15, 15] 前向、y∈[-30, 30] 左向(米)。
    """
    payload = {
        "infos": infos,
        "ckpt": ckpt,
        "score_thr": thr,
        "classes": list(MAPTR_CLASSES),
        "preds": [
            [{"score": float(s), "points": p.tolist()} for s, p in zip(sc, pc, strict=True)]
            for sc, pc in zip(scores_by_class, preds_by_class, strict=True)
        ],
        "gts": [[g.tolist() for g in gc] for gc in gts_by_class],
    }
    Path(path + ".json").write_text(json.dumps(payload), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, name, pc, gc in zip(axes.flat, MAPTR_CLASSES, preds_by_class, gts_by_class, strict=True):
        for g in gc:
            ax.plot(g[:, 0], g[:, 1], color="tab:green", lw=1.2, alpha=0.8)
        for p in pc:
            ax.plot(p[:, 0], p[:, 1], color="tab:red", lw=1.0, alpha=0.9)
        ax.set_title(f"{name}  pred {len(pc)} / gt {len(gc)}")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-30, 30)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.suptitle(f"MapTR pred(red) vs GT(green) | {Path(ckpt).name} score_thr={thr}")
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True, help="B2 组装 infos json")
    ap.add_argument("--root", required=True, help="图像根目录")
    ap.add_argument("--ckpt", required=True, help="train_maptr.py 输出的 state_dict")
    ap.add_argument("--frames", type=int, default=None, help="评估帧数(默认全部)")
    ap.add_argument("--start", type=int, default=0, help="起始帧(留出集评估:训练 0..N-1,评估 --start N)")
    ap.add_argument("--score-thr", type=float, default=0.2, help="实例得分阈值(sigmoid)")
    ap.add_argument("--match", choices=("auto", "cpu", "gpu"), default="auto", help="代价矩阵后端(默认 auto)")
    ap.add_argument("--device", default=None, help="推理设备(默认 cuda 若可用;GPU 被占用时可 --device cpu)")
    ap.add_argument("--out-pred", default=None, help="预测落盘基路径:写 <path>.json + <path>.png(BEV 目检)")
    args = ap.parse_args()

    if args.match == "cpu":
        cost_fn, backend = chamfer_cost_matrix, "cpu"
    elif args.match == "gpu":
        cost_fn, backend = chamfer_cost_matrix_cuda, "gpu"
    else:
        cost_fn, backend = (
            (chamfer_cost_matrix_cuda, "gpu") if torch.cuda.is_available() else (chamfer_cost_matrix, "cpu")
        )

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    infos = json.loads(Path(args.infos).read_text(encoding="utf-8"))
    n = min(args.frames or len(infos), len(infos))
    frames = list(range(args.start, min(args.start + n, len(infos))))
    ds = MapTRDataset(infos, args.root, frames=frames)
    print(f"[data] {len(frames)} 帧 × {len(ds.cam_names)} 相机")

    model = MapTR().to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()
    print(f"[model] {args.ckpt} 载入完成")

    preds_by_class: list[list] = [[] for _ in MAPTR_CLASSES]
    scores_by_class: list[list] = [[] for _ in MAPTR_CLASSES]
    gts_by_class: list[list] = [[] for _ in MAPTR_CLASSES]
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, item in enumerate(ds):
            images = {n: t[None].to(dev) for n, t in item["images"].items()}
            out, _ = model(images, item["pose"][None].to(dev), ds.calibs)
            logits = out["pred_logits"][0].float()  # (Nq, C+1)
            pts = out["pred_points"][0].float().cpu().numpy()  # (Nq, P, 2)
            scores = torch.sigmoid(logits).cpu().numpy()
            for c in range(len(MAPTR_CLASSES)):
                idx = slice(c * model.num_vec, (c + 1) * model.num_vec)
                keep = scores[idx, c + 1] > args.score_thr
                preds_by_class[c].extend(pts[idx][keep])
                scores_by_class[c].extend(scores[idx, c + 1][keep])
                gts_by_class[c].extend(item["gt"][c])
            if (i + 1) % 50 == 0:
                print(f"[infer] {i + 1}/{len(frames)} 帧 ({time.perf_counter() - t0:.1f}s)")

    t1 = time.perf_counter()
    aps, mAP = chamfer_ap_per_class(preds_by_class, gts_by_class, cost_fn=cost_fn)
    print(f"[eval] score_thr={args.score_thr} 后端={backend} 匹配 {time.perf_counter() - t1:.1f}s")
    for cls_name, ap_, preds, gts in zip(MAPTR_CLASSES, aps, preds_by_class, gts_by_class, strict=True):
        print(f"  {cls_name:14s} AP={ap_:.4f}  (pred {len(preds)} / gt {len(gts)})")
    print(f"  {'mAP':14s} = {mAP:.4f}")

    if args.out_pred:
        _dump_preds(
            args.out_pred,
            args.infos,
            args.ckpt,
            args.score_thr,
            preds_by_class,
            scores_by_class,
            gts_by_class,
        )
        print(f"[out] 预测落盘 {args.out_pred}.json / {args.out_pred}.png")


if __name__ == "__main__":
    main()

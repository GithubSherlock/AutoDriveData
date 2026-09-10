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
from pathlib import Path

import torch

from autodrivedata.chamfer_ap import chamfer_ap_per_class
from maptr_impl.dataset import MAPTR_CLASSES, MapTRDataset
from maptr_impl.model import MapTR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True, help="B2 组装 infos json")
    ap.add_argument("--root", required=True, help="图像根目录")
    ap.add_argument("--ckpt", required=True, help="train_maptr.py 输出的 state_dict")
    ap.add_argument("--frames", type=int, default=None, help="评估前 N 帧(默认全部)")
    ap.add_argument("--score-thr", type=float, default=0.2, help="实例得分阈值(sigmoid)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infos = json.loads(Path(args.infos).read_text(encoding="utf-8"))
    frames = list(range(min(args.frames or len(infos), len(infos))))
    ds = MapTRDataset(infos, args.root, frames=frames)
    print(f"[data] {len(frames)} 帧 × {len(ds.cam_names)} 相机")

    model = MapTR().to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()
    print(f"[model] {args.ckpt} 载入完成")

    preds_by_class: list[list] = [[] for _ in MAPTR_CLASSES]
    gts_by_class: list[list] = [[] for _ in MAPTR_CLASSES]
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
                gts_by_class[c].extend(item["gt"][c])
            if (i + 1) % 50 == 0:
                print(f"[infer] {i + 1}/{len(frames)} 帧")

    aps, mAP = chamfer_ap_per_class(preds_by_class, gts_by_class)
    print(f"[eval] score_thr={args.score_thr}")
    for cls_name, ap_, preds, gts in zip(MAPTR_CLASSES, aps, preds_by_class, gts_by_class, strict=True):
        print(f"  {cls_name:14s} AP={ap_:.4f}  (pred {len(preds)} / gt {len(gts)})")
    print(f"  {'mAP':14s} = {mAP:.4f}")


if __name__ == "__main__":
    main()

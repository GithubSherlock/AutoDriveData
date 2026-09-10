"""KITTI root 的 GT vs AutoLabel 伪标签比对报表(比对层 CLI)。

用法(base 或 autolabel env 皆可):
  python bin/eval_kitti.py --root outputs/kitti_scene \
      --pred /root/autodl-tmp/Documents/Projects/AutoLabel/outputs/kitti3d \
      [--frames 3] [--iou 0.5] [--classes Car Pedestrian Cyclist]

- GT:root/training/label_2/{fid}.txt
- 伪标签:pred/labels/{fid}.txt(15 字段,无 conf)
- 置信度:pred/reviews/{fid}_review.json 的 annotations(含 conf)按 3D IoU>0.9
  与 label 行自配对;配不上的行视为 accepted(conf=0.85,复核桶门槛之上)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autodrivedata.compare import (
    Box7,
    box3d_iou,
    evaluate_frames,
    load_gt_labels,
    load_pred_labels,
    report_text,
)


def load_scores(review_json: Path, pred: list[Box7]) -> dict[int, float]:
    """review json annotations → {pred 行索引: confidence}(按 IoU>0.9 自配对)。"""
    if not review_json.is_file():
        return {}
    d = json.loads(review_json.read_text())
    scores: dict[int, float] = {}
    for ann in d.get("annotations", []):
        b = Box7(
            label=str(ann.get("label", "")),
            h=float(ann.get("h", 0)),
            w=float(ann.get("w", 0)),
            l=float(ann.get("l", 0)),
            x=float(ann.get("cx", 0)),
            y=float(ann.get("cy", 0)),
            z=float(ann.get("cz", 0)),
            ry=float(ann.get("rotation_y", 0)),
            conf=float(ann.get("confidence", 1.0)),
        )
        for j, p in enumerate(pred):
            if p.label == b.label and box3d_iou(p, b) > 0.9:
                scores[j] = b.conf
                break
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="KITTI root(GT 所在)")
    ap.add_argument("--pred", required=True, help="AutoLabel out_dir(labels/ + reviews/)")
    ap.add_argument("--frames", type=int, default=0, help="限制前 N 帧(0=全量)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--classes", nargs="+", default=["Car", "Pedestrian", "Cyclist"])
    args = ap.parse_args()

    root = Path(args.root)
    pred_dir = Path(args.pred)
    gt_dir = root / "training" / "label_2"
    labels_dir = pred_dir / "labels"
    reviews_dir = pred_dir / "reviews"
    fids = sorted(p.stem for p in gt_dir.glob("*.txt"))
    if args.frames > 0:
        fids = fids[: args.frames]
    if not fids:
        raise SystemExit(f"无 GT 帧: {gt_dir}")

    frames: dict[str, tuple[list[Box7], list[Box7]]] = {}
    for fid in fids:
        gt = load_gt_labels(gt_dir / f"{fid}.txt")
        pred = load_pred_labels(labels_dir / f"{fid}.txt")
        scores = load_scores(reviews_dir / f"{fid}_review.json", pred)
        for j in range(len(pred)):
            pred[j].conf = scores.get(j, 0.85)  # 未进 review = accepted,阈值之上
        frames[fid] = (gt, pred)

    rep = evaluate_frames(frames, classes=args.classes, iou_thresh=args.iou)
    print(report_text(rep, args.iou))


if __name__ == "__main__":
    main()

"""P1-3 逆光 A/B:冻结 YOLO11s(KITTI 微调)在 A/B 两 KITTI root 的 2D AP 对比。

用法(base env):
  python bin/eval_2d_ab.py --root-a outputs/kitti_ab_day_clear --root-b outputs/kitti_ab_sunset_glare \
      [--limit 150] [--conf 0.25] [--iou 0.5]

注:老 150 帧对(kitti_day_clear / kitti_sunset_glare)中的 kitti_sunset_glare 已于
2026-09-20 清理删除 —— 该对已被 70 帧帧级配对的 kitti_ab_* 取代(见 Plan2.md §10)。
kitti_day_clear 保留(Plan2.md §5 与 bin/slam_diff_test.py 仍引用),但**它现在没有配对的 B**,
要用老口径须显式传一个仍在库的 root。默认值已改为 A/B 新对。

评估口径:GT label_2 2D bbox(列 5-8) vs YOLO 预测(原图尺度),
IoU 贪心匹配(conf 降序,每 GT 一次)→ 逐类 PR 梯形积分 AP;类名归一化
(KITTI Car/Pedestrian/Cyclist ↔ COCO car/person/bicycle 等)。两场景同一模型
权重(冻结)→ 相对差即天气/光照效应。另报画面照度上下文(天空带亮度)。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from autodrivedata.attribution import box_iou2d

GT_CLASSES = ("Car", "Pedestrian", "Cyclist")
COCO_FALLBACK = {
    "car": "Car",
    "truck": "Car",
    "bus": "Car",
    "person": "Pedestrian",
    "bicycle": "Cyclist",
    "motorcycle": "Cyclist",
}


def norm_cls(name: str) -> str:
    n = name.strip().lower()
    if n in COCO_FALLBACK:
        return COCO_FALLBACK[n]
    for c in GT_CLASSES:
        if n == c.lower():
            return c
    return ""


def load_gt(root: Path, limit: int | None = None) -> dict[str, list[tuple[float, float, float, float]]]:
    gt: dict[str, list] = {c: [] for c in GT_CLASSES}
    files = sorted((root / "training/label_2").glob("*.txt"))
    if limit:
        files = files[:limit]
    for f in files:
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) < 15:
                continue
            c = norm_cls(p[0])
            if c:
                gt[c].append(tuple(float(v) for v in p[4:8]))
    return gt


def detect(root: Path, model: YOLO, names: dict[int, str], conf: float, limit: int | None):
    """逐帧推理 → {cls: [(conf, box)]};另返画面天空亮度均值序列。"""
    out: dict[str, list] = {c: [] for c in GT_CLASSES}
    sky_vs: list[float] = []
    files = sorted((root / "training/image_2").glob("*.png"))
    if limit:
        files = files[:limit]
    for f in files:
        img = np.array(Image.open(f).convert("RGB"))
        sky_vs.append(img[: img.shape[0] // 4].mean())
        # predict 返回 union(Iterator | list),先 materialize 再取首帧
        res = cast(Results, list(model.predict(f, conf=conf, verbose=False, device=0))[0])
        for b in res.boxes or []:  # 无检测帧 boxes=None
            c = norm_cls(names[int(b.cls.item())])
            if not c:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out[c].append((float(b.conf.item()), (x1, y1, x2, y2)))
    return out, sky_vs


def ap_for(gt_boxes, preds, iou_thr: float) -> tuple[float, int, int]:
    """conf 降序贪心 IoU 匹配 → 11 点插值 AP(与 3D 侧 compare.ap11 同口径)。

    AP 尾部纪律:recall 未达 1 的部分无预测 → precision=0(低 recall 不注水)。
    2026-09-09 教训:旧实现尾行 `ap += (1-prev_r)*prev_p` 把未达 recall 段仍按
    最后 precision 计入——最后一个是 TP 时低 recall 数据被严重吹高
    (雨夜检出 0.48 却报 AP 0.976,触发修复)。
    """
    preds = sorted(preds, key=lambda t: -t[0])
    matched = [False] * len(gt_boxes)
    tp: list[bool] = []
    for _, box in preds:
        best_i, best_v = -1, 0.0
        for j, g in enumerate(gt_boxes):
            if matched[j]:
                continue
            v = box_iou2d(g, box)
            if v > best_v:
                best_i, best_v = j, v
        if best_i >= 0 and best_v >= iou_thr:
            tp.append(True)
            matched[best_i] = True
        else:
            tp.append(False)
    n_gt, n_pred = len(gt_boxes), len(preds)
    if n_pred == 0 or n_gt == 0:
        return 0.0, n_gt, n_pred
    tp_np = np.array(tp, dtype=float)
    cum_tp = np.cumsum(tp_np)
    recalls = cum_tp / n_gt
    precisions = cum_tp / np.arange(1, n_pred + 1)
    ap = 0.0
    for rq in np.linspace(0.0, 1.0, 11):
        hit = recalls >= rq
        p = float(precisions[hit].max()) if hit.any() else 0.0
        ap += p / 11.0
    return ap, n_gt, n_pred


def report(
    root: Path,
    model: YOLO,
    names: dict[int, str],
    conf: float,
    iou: float,
    limit: int | None,
):
    det, sky = detect(root, model, names, conf, limit)
    gt_all = load_gt(root, limit)
    print(f"\n=== {root.name} (conf={conf} IoU@{iou}) 天空带亮度均值 {np.mean(sky):.0f} ± {np.std(sky):.0f}")
    aps: list[float] = []
    for c in GT_CLASSES:
        ap, n_gt, n_pred = ap_for(gt_all[c], det[c], iou)
        if n_gt > 0:  # 无 GT 的类不稀释 mAP(本项目行人 GT 稀疏,见 collect_drive 局限)
            aps.append(ap)
        print(f"  {c:11s} AP={ap:6.3f}  GT={n_gt:5d}  检出={n_pred:5d}  检出/GT={n_pred / max(n_gt, 1):.2f}")
    m = float(np.mean(aps)) if aps else float("nan")
    print(f"  mAP(有GT的 {len(aps)} 类)={m:.3f}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-a", default="outputs/kitti_ab_day_clear")
    ap.add_argument("--root-b", default="outputs/kitti_ab_sunset_glare")
    ap.add_argument(
        "--weight",
        default=(
            "/root/autodl-tmp/Documents/Projects/AutoLabel/auto2dlabel/weights/"
            "kitti_finetune/yolo11s_kitti/weights/best.pt"
        ),
    )
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = YOLO(args.weight)
    names = model.names
    print("YOLO names:", names)
    ra = report(Path(args.root_a), model, names, args.conf, args.iou, args.limit)
    rb = report(Path(args.root_b), model, names, args.conf, args.iou, args.limit)
    delta = rb - ra
    verdict = (
        f"B 侧更低 → {Path(args.root_b).name} 掉点成立"
        if delta < -0.01
        else ("B 侧更高" if delta > 0.01 else "两测持平")
    )
    print(f"\nΔ mAP (B−A) = {delta:+.3f} —— {verdict}")


if __name__ == "__main__":
    main()

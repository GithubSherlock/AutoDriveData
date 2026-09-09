"""失效归因评估:逐帧 2D 匹配 → 漏检按距离/像素高度/TTC 分箱(Plan §5.10)。

与 eval_2d_ab 的分工(同一 box_iou2d 口径,两个问题):
  eval_2d_ab  全库池化 PR   → "天气让 AP 掉多少"(结论层,已定案)
  eval_attr   逐帧匹配 + 每 GT 上下文 → "漏在哪个距离/尺度/TTC、为什么"(归因层)

每个 GT 记录:距离 / 2D 框高(px)/ TTC / 截断 / 框内亮度·对比度·梯度(图像诊断)。
两种分箱读法别混用:距离/框高箱是图像几何量(跨速度/天气可比,尺度决定检出率);
TTC 箱是安全余量语义(同一箱在不同速度对应不同距离,检出率不可跨速度对比)。

用法(base env,数据已落盘):
  python scripts/eval_attr.py \
      --run day8=outputs/kitti_ab_day_clear:8.0 \
      --run day4=outputs/kitti_sweep_day_clear_4:4.0 \
      --run rain8=outputs/kitti_ab_rain_night:8.0
  # 输出:每跑分箱表 + 漏检画像 + 跨跑距离/框高网格;--json 落原始记录备查
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from autodrivedata.attribution import (
    DISTANCE_EDGES,
    HEIGHT_EDGES,
    TTC_EDGES,
    Detection,
    GtRecord,
    bin_stats,
    closing_speed_series,
    format_bins,
    load_gt_2d,
    match_frame,
    norm_cls,
    ttc_s,
)

DELTA_S = 0.1  # 同步模式固定步长(同 carla_common.sync_mode / 各采集器)


def parse_run(text: str) -> tuple[str, Path, float]:
    """`名字=路径:速度` → (name, root, speed)。"""
    name, sep, rest = text.partition("=")
    root, sep2, speed = rest.rpartition(":")
    if not (name and sep and sep2 and root):
        raise argparse.ArgumentTypeError(f"--run 格式应为 名字=路径:速度,收到 {text!r}")
    try:
        return name, Path(root), float(speed)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--run 速度需为数字: {text!r}") from e


def image_stats(
    gray: np.ndarray,
    box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """框内亮度均值/标准差/梯度能量 + 相对环带对比度(局部信噪比)。

    对比度 = (框内均值 − 外扩 50% 环带均值) / 环带标准差:回答"目标是否被
    局部背景淹没"。梯度能量是运动模糊/失焦的廉价代理(合成图无快门模糊时
    它只反映纹理与尺度)。
    """
    h, w = gray.shape
    x1, y1, x2, y2 = box
    xi1, yi1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    xi2, yi2 = min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2)))
    if xi2 - xi1 < 2 or yi2 - yi1 < 2:
        nan = math.nan  # 显式四元组:tuple 推导是变长,pyright 拒收(T3 类坑)
        return (nan, nan, nan, nan)
    patch = gray[yi1:yi2, xi1:xi2].astype(np.float32)
    lum_mean, lum_std = float(patch.mean()), float(patch.std())
    grad = float(np.abs(np.diff(patch, axis=0)).mean() + np.abs(np.diff(patch, axis=1)).mean())
    mx, my = int(0.5 * (xi2 - xi1)), int(0.5 * (yi2 - yi1))
    ex1, ey1 = max(0, xi1 - mx), max(0, yi1 - my)
    ex2, ey2 = min(w, xi2 + mx), min(h, yi2 + my)
    outer = gray[ey1:ey2, ex1:ex2].astype(np.float32)
    mask = np.ones(outer.shape, dtype=bool)
    mask[yi1 - ey1 : yi2 - ey1, xi1 - ex1 : xi2 - ex1] = False
    ring = outer[mask]
    contrast = float((lum_mean - ring.mean()) / (ring.std() + 1e-6)) if ring.size else math.nan
    return lum_mean, lum_std, grad, contrast


def run_one(
    root: Path,
    speed: float,
    model: YOLO,
    names: dict[int, str],
    conf: float,
    iou: float,
    limit: int | None,
) -> tuple[list[GtRecord], list[float]]:
    """逐帧推理 + 匹配 → 全部 GT 的归因记录 + 逐帧接近速度。

    两趟:第一趟逐帧推理/图像诊断(此时还不知道速度),第二趟用逐帧速度填 TTC。
    逐帧速度而非标称常数:P1 老数据集实测 6.60 m/s(标称 8.0,残留制动),
    详见 attribution.closing_speed_series。
    """
    gt_by_frame = load_gt_2d(root, limit)
    per_frame: list[tuple[str, list]] = []
    dist_seq: list[list[float]] = []
    for fid in sorted(gt_by_frame):
        img_path = root / "training/image_2" / f"{fid}.png"
        gray = np.array(Image.open(img_path).convert("RGB")).mean(axis=2)
        # predict 返回 union(Iterator | list),先 materialize 再取首帧(同 eval_2d_ab)
        res = cast(Results, list(model.predict(img_path, conf=conf, verbose=False, device=0))[0])
        preds: list[Detection] = []
        for b in res.boxes or []:  # 无检测帧 boxes=None
            c = norm_cls(names[int(b.cls.item())])
            if not c:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            preds.append(Detection(c, x1, y1, x2, y2, float(b.conf.item())))
        gts = gt_by_frame[fid]
        dist_seq.append([g.distance_m for g in gts])
        rows: list = []
        for g, ok in zip(gts, match_frame(gts, preds, iou), strict=True):
            lum, lum_std, grad, contrast = image_stats(gray, g.box)
            rows.append((g, ok, (lum, lum_std, grad, contrast)))
        per_frame.append((fid, rows))

    speeds = closing_speed_series(dist_seq, DELTA_S)
    recs: list[GtRecord] = []
    for (fid, rows), v in zip(per_frame, speeds, strict=True):
        v_use = speed if math.isnan(v) else v  # 估不出速度的帧退回标称
        for g, ok, (lum, lum_std, grad, contrast) in rows:
            recs.append(
                GtRecord(
                    frame=fid,
                    cls=g.cls,
                    distance_m=g.distance_m,
                    height_px=g.height_px,
                    truncation=g.truncation,
                    ttc_s=ttc_s(g.distance_m, v_use),
                    matched=ok,
                    lum_mean=lum,
                    lum_std=lum_std,
                    grad_energy=grad,
                    contrast=contrast,
                )
            )
    return recs, speeds


def median_or_nan(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def miss_profile(recs: list[GtRecord]) -> str:
    """命中 vs 漏检的画像对照(中位数)——回答"漏检长什么样"。"""
    hit = [r for r in recs if r.matched]
    miss = [r for r in recs if not r.matched]
    rows = [
        ("距离(m)", lambda rs: [r.distance_m for r in rs]),
        ("框高(px)", lambda rs: [r.height_px for r in rs]),
        ("TTC(s)", lambda rs: [r.ttc_s for r in rs]),
        ("框内亮度", lambda rs: [r.lum_mean for r in rs]),
        ("局部对比度", lambda rs: [r.contrast for r in rs]),
        ("梯度能量", lambda rs: [r.grad_energy for r in rs]),
    ]
    lines = [f"  {'指标':<10s} {'命中中位':>10s} {'漏检中位':>10s}"]
    for label, getter in rows:
        lines.append(
            f"  {label:<10s} {median_or_nan(getter(hit)):>10.2f} {median_or_nan(getter(miss)):>10.2f}"
        )
    return "\n".join(lines)


def report_run(
    name: str,
    root: Path,
    nominal_speed: float,
    recs: list[GtRecord],
    speeds: list[float],
) -> None:
    car_vis = [r for r in recs if r.cls == "Car" and r.truncation <= 0.0]
    n_hit = sum(r.matched for r in car_vis)
    valid = [v for v in speeds if not math.isnan(v)]
    med = statistics.median(valid) if valid else math.nan
    ramp = sum(1 for v in valid if v < 0.9 * med)
    print(f"\n=== [{name}] {root}")
    print(
        f"  GT(全类)={len(recs)}  GT(Car,出画外剔除)={len(car_vis)}  检出={n_hit}  "
        f"检出率={n_hit / max(len(car_vis), 1):.2f}"
    )
    print(
        f"  接近速度(逐帧实测,数据自证):中位 {med:.2f} m/s | 标称 {nominal_speed:.2f} | "
        f"慢速帧 {ramp}/{len(valid)}(< 0.9×中位;应为 0,>0 说明起步/受阻)"
    )
    for label, key, edges, unit in (
        ("距离", lambda r: r.distance_m, DISTANCE_EDGES, "m"),
        ("框高", lambda r: r.height_px, HEIGHT_EDGES, "px"),
        ("TTC", lambda r: r.ttc_s, TTC_EDGES, "s"),
    ):
        print(format_bins(bin_stats(car_vis, key, edges), unit=unit, label=label))
    print("  漏检画像(Car,trunc=0):")
    print(miss_profile(car_vis))


def report_grid(
    results: dict[str, list[GtRecord]],
    key: Callable[[GtRecord], float],
    edges: tuple[float, ...],
    title: str,
    unit: str,
) -> None:
    """跨跑 × 分箱检出率网格(速度/天气对比主表;距离箱=几何,框高箱=尺度)。"""
    per_run = {
        name: bin_stats([r for r in recs if r.cls == "Car" and r.truncation <= 0.0], key, edges)
        for name, recs in results.items()
    }
    head = "  " + f"{'分箱':<10s}" + "".join(f"{n:>12s}" for n in per_run)
    print(f"\n=== {title}(Car,trunc=0;括号内为 GT 数)")
    print(head)
    for i in range(len(edges) - 1):
        hi = "+" if math.isinf(edges[i + 1]) else f"{edges[i + 1]:g}"
        row = f"  {f'{edges[i]:g}-{hi}{unit}':<10s}"
        for stats in per_run.values():
            s = stats[i]
            cell = "—" if s.n_gt == 0 else f"{s.rate:.2f}({s.n_gt})"
            row += f"{cell:>12s}"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        dest="runs",
        type=parse_run,
        action="append",
        required=True,
        metavar="名字=路径:速度",
        help="可重复;速度用于 TTC 归一化(脚本另用 GT 距离自证接近速度)",
    )
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
    ap.add_argument("--json", default=None, help="把原始归因记录落 JSON(备查/复算)")
    args = ap.parse_args()

    model = YOLO(args.weight)
    names = model.names
    results: dict[str, list[GtRecord]] = {}
    dump: dict[str, list[dict]] = {}
    for name, root, speed in args.runs:
        if name in results:
            raise SystemExit(f"--run 名字重复: {name}")
        recs, speeds = run_one(root, speed, model, names, args.conf, args.iou, args.limit)
        report_run(name, root, speed, recs, speeds)
        results[name] = recs
        dump[name] = [asdict(r) for r in recs]

    report_grid(results, lambda r: r.distance_m, DISTANCE_EDGES, "距离分箱检出率网格", "m")
    report_grid(results, lambda r: r.height_px, HEIGHT_EDGES, "框高分箱检出率网格", "px")
    if args.json:
        Path(args.json).write_text(json.dumps(dump, ensure_ascii=False, indent=1))
        print(f"\n[json] {Path(args.json).resolve()}")


if __name__ == "__main__":
    main()

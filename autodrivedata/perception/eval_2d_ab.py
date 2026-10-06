"""P1-3 逆光 A/B:冻结检测器在 A/B 两 KITTI root 的 2D AP 对比。

**两个后端,默认 `sam3`**(2026-10-02 起):
- `--backend sam3`(**默认**):SAM3 开放词表,类别由**提示词**给出
  (`sam3_backend.DETECT_PROMPTS`)。⚠️ 于是 AP 里"分类正确率"那一项退化成
  「提示词写对没有」—— **与 yolo 后端的 AP 不可比**;
- `--backend yolo`(回退):ultralytics 闭集检测器,`--weight` 默认 KITTI 微调的
  `yolo11s_kitti/weights/best.pt`,类名是**模型判的**。
⇒ 引用任何 Δ 都必须写明是哪个后端;P1 归档矩阵是 **yolo 基线**。

用法(base env):
  python -m autodrivedata.perception.eval_2d_ab --root-a outputs/kitti_ab_epic_clip2d_day_clear \
      --root-b outputs/kitti_ab_epic_clip2d_dense_fog [--backend yolo] [--conf 0.25] [--iou 0.5]

注:老 150 帧对(kitti_day_clear / kitti_sunset_glare)中的 kitti_sunset_glare 已于
2026-09-20 清理删除 —— 该对已被 70 帧帧级配对的 kitti_ab_* 取代(见 Plan2.md §10)。
kitti_day_clear 保留(Plan2.md §5 与 autodrivedata/slam/slam_diff_test.py 仍引用),但**它现在没有配对的 B**,
要用老口径须显式传一个仍在库的 root。默认值已改为 A/B 新对。

评估口径:GT label_2 2D bbox(列 5-8) vs 预测(原图尺度),
IoU 贪心匹配(conf 降序,每 GT 一次)→ 逐类 PR 梯形积分 AP;类名归一化
(KITTI Car/Pedestrian/Cyclist ↔ COCO car/person/bicycle 等)。两场景同一**后端**
(冻结)→ 相对差即天气/光照效应。另报画面照度上下文(天空带亮度)。

★ **读数有三列是给"AP 不动"那种情况准备的**:`命中`(操作点上的 TP)/ `召回` / `检出/GT`
(过检率)。换到 SAM3 后 AP 在 P1 上对退化不敏感(过检把 recall 撑住了),而这三列照样动
—— 见 `report` 的 docstring 与 Plan4 §P-V23/§P-V24。

## ★★ AP 是**台阶函数** —— 单条检测翻面最多可换 `1/11 ≈ 0.091`(2026-10-06 实测)

11 点插值的 recall 格点是 `j/10`,格点 `j` **可达** ⟺ `n_tp ≥ ceil(j/10·n_gt)`。
于是每有一个 `n_tp` 跨过这条线,那一格的 precision 就**从 0 跳成正值**,AP 跳 `p_j/11`。

实测(35 帧 / 单类 Car / `n_gt = 119`):把 `blur` 从 `k=6` 换到 `k=7`,**只多了 1 条
TP**(`conf 0.360`、`bestIoU 0.540`,压在 0.5 上),`tp 107 → 108` 使 `108/119 = 0.908 ≥ 0.9`
⇒ 第 10 个格点亮起 ⇒ **ΔAP 从 −0.026 翻成 +0.055**。同一批检测换 **101 点插值**,
台阶**完全消失**(−0.026 → −0.014,单调)。

⇒ 两条纪律:
① **ΔAP 必须与 `n_tp/n_gt` 一起读**。`report` 现在会自己喊出"距格点还剩几个 TP";
② `noise_curve` 这类**要插值/投影**的用途,台阶会直接毁掉拟合 —— 每个点都记下悬崖余量。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from autodrivedata.perception.attribution import box_iou2d
from autodrivedata.perception.backends import (
    DEFAULT_YOLO_WEIGHT,
    GT_CLASSES,
    Predictor,
    describe,
    make_predictor,
    norm_cls,
    resolve_conf,
)
from autodrivedata.utils import runlog


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


def detect(root: Path, predict: Predictor, limit: int | None):
    """逐帧推理 → {cls: [(conf, box)]};另返画面天空亮度均值序列。"""
    out: dict[str, list] = {c: [] for c in GT_CLASSES}
    sky_vs: list[float] = []
    files = sorted((root / "training/image_2").glob("*.png"))
    if limit:
        files = files[:limit]
    for f in files:
        img = np.array(Image.open(f).convert("RGB"))
        sky_vs.append(img[: img.shape[0] // 4].mean())
        for c, score, box in predict(f):
            if c:
                out[c].append((score, box))
    return out, sky_vs


def ap_for(gt_boxes, preds, iou_thr: float, *, n_points: int = 11) -> tuple[float, int, int, int]:
    """conf 降序贪心 IoU 匹配 → 插值 AP(默认 **11 点**,与 3D 侧 `compare.ap11` 同口径)。

    AP 尾部纪律:recall 未达 1 的部分无预测 → precision=0(低 recall 不注水)。
    2026-09-09 教训:旧实现尾行 `ap += (1-prev_r)*prev_p` 把未达 recall 段仍按
    最后 precision 计入——最后一个是 TP 时低 recall 数据被严重吹高
    (雨夜检出 0.48 却报 AP 0.976,触发修复)。

    ## ⚠️ `n_points` 不是调参旋钮,是**分辨率**

    默认 11 是**归档口径**(全仓历史数字都是它,改了不可比)。但 `n_gt` 只有一两百时,
    11 点会在 max-recall 跨格处跳 `p/11 ≤ 0.091`(见模块头注与 `grid_cliff`)。
    ⇒ **凡是要"插值/投影/拟合曲线"的用途,必须用细格**(`edit/calibrate` 用 101),
    否则曲线自己就是台阶状的,拟合出来的东西没有意义。
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
    n_tp = sum(matched)  # ← 操作点上的命中数(`matched` 就是上面那次贪心的结果)
    if n_pred == 0 or n_gt == 0:
        return 0.0, n_gt, n_pred, n_tp
    tp_np = np.array(tp, dtype=float)
    cum_tp = np.cumsum(tp_np)
    recalls = cum_tp / n_gt
    precisions = cum_tp / np.arange(1, n_pred + 1)
    ap = 0.0
    for rq in np.linspace(0.0, 1.0, n_points):
        hit = recalls >= rq
        p = float(precisions[hit].max()) if hit.any() else 0.0
        ap += p / n_points
    return ap, n_gt, n_pred, n_tp


#: 11 点插值的格点数。**台阶的量级由它钉死** —— 单格 `p/11 ≤ 1/11`。见模块头注。
AP_GRID_N = 11

#: 单条检测翻面在 11 点插值下**最多**能换掉的 AP。**这是分辨率,不是误差**。
AP_GRID_STEP = 1.0 / AP_GRID_N


def grid_cliff(n_gt: int, n_tp: int) -> tuple[int, int | None]:
    """11 点插值的 **recall 可达性悬崖**:返回 `(最高可达格点, 距上一格还差几个 TP)`。

    格点 `j ∈ {1..10}` 的阈值是 `T_j = ceil(j/10 · n_gt)`;`n_tp ≥ T_j` 时该格才"亮起"。
    `top == 10`(满格)时第二个值是 `None`;`n_gt == 0` 时返回 `(-1, None)`(**不可判**)。

    ⚠️ **这不是数据噪声,是尺子的台阶**。与"AP 复现性下限 2e-3"(边界实例跨过
    `--score-thr`)是**两个量**:那个是阈值抖动,这个的量级 = `1/11`,比它大 **45 倍**。
    判据侧的含义:高分辨率下 ΔAP 才可比,`n_gt` 只有一两百时**必须**看这个余量。
    """
    if n_gt <= 0:
        return -1, None
    for j in range(1, AP_GRID_N):
        need = -(-j * n_gt // 10)  # ceil(j·n_gt/10),整数算术避开浮点
        if n_tp < need:
            return j - 1, int(need - n_tp)
    return AP_GRID_N - 1, None


def fragile_classes(ev: dict, *, tol: int = 2) -> list[tuple[str, int]]:
    """离悬崖 ≤ `tol` 个 TP 的类(`类名, 还差几个 TP`)。**这些类的 AP 一句话就能翻脸。**"""
    return [
        (c, r["cliff_up"])
        for c, r in ev["classes"].items()
        if r["n_gt"] and r["cliff_up"] is not None and r["cliff_up"] <= tol
    ]


def evaluate(
    root: Path,
    predict: Predictor,
    conf: float,
    iou: float,
    limit: int | None,
    *,
    n_points: int = 11,
    verbose: bool = True,
) -> dict:
    """跑一次评测,**返回全部明细**(`report` 只是它的一个壳)。

    ## ★ 为什么要拆出这一层

    `noise_curve` 要拿每个点的 `n_tp/n_gt` 去判"这个点离悬崖多远" —— 只有返回明细
    才拿得到。**没有它,标定曲线会把插值台阶当成退化效应**(2026-10-06 实测:
    β=0.04 那一点的 ΔAP `+0.0763` 里有 `+0.066` 是台阶)。

    ## 逐类报 **AP + 操作点上的命中/召回/过检**(2026-10-03 加的)

    换到 SAM3 后 P1 的四个 Δ 全落进 ±0.02(而 yolo 口径是 −0.578 / −0.487 / …),
    一度被读成"SAM3 抗退化"。查下去不是:浓雾下**检出数 431→268(−38%)**,退化是真的,
    只是**过检 2.36×GT 把 recall 撑在 1 附近**,AP 的尾部纪律(未达 recall 段
    precision=0)于是无从发力 —— **AP 在 recall 饱和时对退化不敏感**。

    ⚠️ 口径:**都在同一个 conf 操作点上、同一次贪心匹配里出的** —— 不是为了好看另算一套。
    """
    det, sky = detect(root, predict, limit)
    gt_all = load_gt(root, limit)
    rows: dict[str, dict] = {}
    aps: list[float] = []
    for c in GT_CLASSES:
        ap, n_gt, n_pred, n_tp = ap_for(gt_all[c], det[c], iou, n_points=n_points)
        if n_gt > 0:  # 无 GT 的类不稀释 mAP(本项目行人 GT 稀疏,见 collect_drive 局限)
            aps.append(ap)
        top, up = grid_cliff(n_gt, n_tp)
        rows[c] = {
            "ap": ap,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "n_tp": n_tp,
            "recall": (n_tp / n_gt) if n_gt else float("nan"),
            "over": n_pred / max(n_gt, 1),
            "cliff_top": top,
            "cliff_up": up,
        }
    ev = {
        "root": str(root),
        "name": root.name,
        "mAP": float(np.mean(aps)) if aps else float("nan"),
        "n_classes": len(aps),  # 参与 mAP 的类数(不是 `len(GT_CLASSES)`)
        "n_points": n_points,  # ★ 口径必须随读数走 —— "11 点"与"101 点"是两把尺子
        "conf": conf,
        "iou": iou,
        "sky_mean": float(np.mean(sky)) if len(sky) else float("nan"),
        "sky_std": float(np.std(sky)) if len(sky) else float("nan"),
        "classes": rows,
    }
    if verbose:
        print(
            f"\n=== {root.name} (conf={conf} IoU@{iou}) 天空带亮度均值 {ev['sky_mean']:.0f} ± {ev['sky_std']:.0f}"
        )
        print(f"  {'类':<11}{'AP':>7}{'GT':>6}{'检出':>6}{'命中':>6}{'召回':>7}{'检出/GT':>9}")
        for c, r in rows.items():
            print(
                f"  {c:<11}{r['ap']:>7.3f}{r['n_gt']:>6}{r['n_pred']:>6}{r['n_tp']:>6}"
                f"{r['recall']:>7.3f}{r['over']:>9.2f}"
            )
        print(f"  mAP(有GT的 {len(aps)} 类)={ev['mAP']:.3f}")
        frag = fragile_classes(ev)
        if frag:
            detail = "、".join(f"{c} 距上格 {u} TP" for c, u in frag)
            print(f"  ⚠ AP 是台阶函数(11 点 recall 格):{detail} ⇒ 单条检测翻面即可换 ±{AP_GRID_STEP:.3f} AP")
    return ev


def report(root: Path, predict: Predictor, conf: float, iou: float, limit: int | None) -> float:
    """`evaluate` 的薄壳 —— 打印明细,只回 mAP(**老签名,勿改:多处调用方靠它**)。"""
    return evaluate(root, predict, conf, iou, limit)["mAP"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-a", default="outputs/kitti_ab_day_clear")
    ap.add_argument("--root-b", default="outputs/kitti_ab_sunset_glare")
    ap.add_argument(
        "--weight",
        default=DEFAULT_YOLO_WEIGHT,
    )
    ap.add_argument(
        "--backend",
        choices=("sam3", "yolo"),
        default="sam3",
        help="检测后端。**默认 sam3**(开放词表,类别由提示词给出);`yolo` 是回退。"
        "⚠️ **两边的 AP 不可比** —— 换后端必须重跑,引用时必须写明是哪个后端",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=None,
        help="置信度阈值。**不传则按后端取默认**(sam3 0.5 / yolo 0.25)—— "
        "两边的 score 尺度不同,互相套用会把幻觉当检出",
    )
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-runlog", action="store_true", help="不落 logs/ 三件套(默认每次运行都落)")
    args = ap.parse_args()

    with runlog.run("autodrivedata.perception.eval_2d_ab") as rl:
        rl.input(args.weight, "weight")
        rl.highlight("backend", args.backend)  # 权重同名覆盖是常态,不记就没法回溯这份 Δ 是哪份权重出的
        rl.input(args.root_a, "root-a")
        rl.input(args.root_b, "root-b")
        conf = resolve_conf(args.backend, args.conf)  # ⚠️ 解析**一次** —— 下面三处都用它
        predict = make_predictor(args.backend, args.weight, conf)
        print(describe(args.backend, predict))
        print(f"conf = {conf}" + ("(后端默认)" if args.conf is None else "(命令行指定)"))
        ra = report(Path(args.root_a), predict, conf, args.iou, args.limit)
        rb = report(Path(args.root_b), predict, conf, args.iou, args.limit)
        delta = rb - ra
        verdict = (
            f"B 侧更低 → {Path(args.root_b).name} 掉点成立"
            if delta < -0.01
            else ("B 侧更高" if delta > 0.01 else "两测持平")
        )
        print(f"\nΔ mAP (B−A) = {delta:+.3f} —— {verdict}")
        # 无产物脚本:结论就是这两个数与它们的差 —— 断点只看 Δ 不看 conf 会重蹈
        # "AP 数字离开阈值无意义"的坑,故 conf/iou 与 Δ 一起留痕
        rl.highlight("mAP_A", round(ra, 4))
        rl.highlight("mAP_B", round(rb, 4))
        rl.highlight("delta_mAP_B_minus_A", round(delta, 4))
        rl.highlight("conf", conf)
        rl.highlight("iou", args.iou)
        rl.highlight("root_a", Path(args.root_a).name)
        rl.highlight("root_b", Path(args.root_b).name)
        rl.note(f"判据:{verdict}")


if __name__ == "__main__":
    main()

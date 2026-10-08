"""**域差有多大** —— 2×2 迁移矩阵:两个模型 × 两个域。用来决定"域自适应方法侧开不开"。

## 为什么是 2×2 而不是"算一个域距离"

JD 四方向里「域自适应 / 域泛化(方法)」是本项目**唯一方法侧全零**的一格。
开不开它,取决于**域差在这个项目的单位下有多大** —— 而这个项目的单位是**检测 AP**,
不是通用的图像统计量(embedding 距离那类东西在这里没有下游)。

⇒ 直接量**同一个模型换域掉多少**,两个模型各当一次"域内"和"域外":

| 模型 | 训练域 | 在 COCO 上 | 在 CARLA 上 |
|---|---|---|---|
| `coco_native`(`weights/yolo26x.pt`) | **COCO(真实)** | **域内(天花板)** | **域外** |
| `kitti_ft`(AutoLabel 的 `yolo11s_kitti`) | **CARLA→KITTI 合成** | **域外** | **域内** |

⇒ 两条"域内 − 域外"的落差,就是这个项目里**域差的实际大小**。
★ **这也解释了一个已知现象**:红线里记着「yolo 在**无雾的 KITTI** 上微调,域外本就是它的弱项」——
本模块把那个"弱多少"量出来。

## 口径(缺一条都读不出结论)

- **同一个 AP 口径**:`eval_2d_ab.ap_for`,**101 点**(11 点会让小 Δ 落到台阶上,见 §1.10);
- **同一个类表**:COCO 的 80 类经 `backends.COCO_FALLBACK` 归一到本项目的
  `Car` / `Pedestrian` / `Cyclist`(**与两个后端共用同一张表**,不另写一份);
- **同一个 `conf`**:默认 `0.25`(**两个模型都是 yolo 家族**,尺度可比;SAM3 不行,见红线);
- **逐类报 `n_gt` / `n_pred`**,AP 与它们一起读。

## ★ 对照(必须做)

**把 GT 框整体平移 1/3 画幅** ⇒ AP 必须塌到 ≈0。缺这条,"0.35 算低吗"没有参照 ——
而"尺子坏了"与"域差大"长得一样。

⚠️⚠️ **第一版做的是"打乱图与 GT 的配对",当场被自己的读数打红**(2026-10-07):
打乱之后 mAP **一个数没变**(0.750 → 0.750、0.079 → 0.079)。因:
`ap_for` 把**所有图的框池化**之后再做一次全局贪心匹配 ⇒ **"哪张图的框"根本不参与计算**,
重排就是恒等。★ 这正是本仓红线里那条「`eval_2d_ab` 把**所有帧的框池化**后算 AP ⇒
'打乱配对'那类对照**不是对照**」的**第二次现形** —— 而它这次**自己把自己抓了**。
⇒ 有效的对照必须让 **GT 的多重集本身**变(平移框 / 换一个不相交的 GT 池),
不是只换"谁配谁"。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autodrivedata.perception.backends import DEFAULT_YOLO_WEIGHT, norm_cls
from autodrivedata.perception.eval_2d_ab import GT_CLASSES, ap_for
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 两个模型的**标签**,路径在 `main` 里给(默认值只是常见落点,不是契约)。
MODELS = ("coco_native", "kitti_ft")


def load_coco_gt(ann_path: Path) -> dict[int, list[tuple[str, tuple[float, float, float, float]]]]:
    """COCO `instances_*.json` → `{image_id: [(本项目类名, (x1,y1,x2,y2))]}`。

    ⚠️ 类名走 `backends.norm_cls`(**与两个后端同一张表**)。COCO 的 80 类里
    只有 person/bicycle/car/bus/truck/motorcycle 会落进本项目三类,其余**丢弃** ——
    这不是"漏了一半",是本项目的类表就只有三类(`GT_CLASSES`)。
    """
    d = json.loads(ann_path.read_text(encoding="utf-8"))
    cat = {c["id"]: c["name"] for c in d["categories"]}
    out: dict[int, list] = {}
    for a in d["annotations"]:
        cls = norm_cls(cat.get(a["category_id"], ""))
        if not cls:
            continue
        x, y, w, h = a["bbox"]
        out.setdefault(a["image_id"], []).append((cls, (x, y, x + w, y + h)))
    return out


#: 对照臂把 GT 框平移多少(画幅比例)。1/3 ⇒ 框基本离开原位置,但不至于全被钳到边角。
CONTROL_SHIFT_FRAC = 1.0 / 3.0


def _shift_shapes(images: list[Path], gt: dict[int, list]) -> dict[int, list]:
    """★ **有效对照**:把每张图的 GT 框**整体平移** `CONTROL_SHIFT_FRAC × 画幅`。

    ⚠️ 为什么不能靠"打乱配对":`ap_for` 是**池化**之后做一次全局贪心匹配的,
    "哪张图的框"**根本不进计算** ⇒ 重排是恒等(实测 0.750 → 0.750)。见模块头注。
    平移**改变的是 GT 多重集本身**,这才是对照。
    """
    from PIL import Image

    out: dict[int, list] = {}
    for p in images:
        w, h = Image.open(p).size
        dx, dy = CONTROL_SHIFT_FRAC * w, CONTROL_SHIFT_FRAC * h
        out[int(p.stem)] = [
            (c, (x1 + dx, y1 + dy, x2 + dx, y2 + dy)) for c, (x1, y1, x2, y2) in gt.get(int(p.stem), [])
        ]
    return out


def evaluate_coco(
    images: list[Path],
    gt: dict[int, list],
    predict,
    *,
    iou: float = 0.5,
    n_points: int = 101,
    control_shift: bool = False,
    verbose: bool = True,
) -> dict:
    """COCO 侧的 `evaluate` —— 与 `eval_2d_ab.evaluate` **同口径、同返回形状**。

    `control_shift=True` ⇒ 用平移过的 GT(见 `_shift_shapes`)。这是**对照臂**:
    mAP 必须塌。
    """
    if control_shift:
        gt = _shift_shapes(images, gt)
    det: dict[str, list] = {c: [] for c in GT_CLASSES}
    gt_all: dict[str, list] = {c: [] for c in GT_CLASSES}
    for p in images:
        for c, score, box in predict(p):
            if c:
                det[c].append((score, box))
        for c, box in gt.get(int(p.stem), []):
            gt_all[c].append(box)
    rows, aps = {}, []
    for c in GT_CLASSES:
        ap, n_gt, n_pred, n_tp = ap_for(gt_all[c], det[c], iou, n_points=n_points)
        if n_gt > 0:
            aps.append(ap)
        rows[c] = {
            "ap": ap,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "n_tp": n_tp,
            "recall": (n_tp / n_gt) if n_gt else float("nan"),
        }
    ev = {
        "name": "coco_val2017",
        "mAP": float(np.mean(aps)) if aps else float("nan"),
        "n_classes": len(aps),
        "n_points": n_points,
        "iou": iou,
        "n_images": len(images),
        "control_shift": control_shift,
        "classes": rows,
    }
    if verbose:
        for c, r in rows.items():
            print(f"    {c:<11}{r['ap']:>7.3f} GT{r['n_gt']:>6} 检出{r['n_pred']:>6} 召回{r['recall']:>7.3f}")
    return ev


def image_paths(root: Path, ann: dict, limit: int | None) -> list[Path]:
    """只取**有本项目三类 GT** 的图(否则那些图对 mAP 只是稀释),按文件名排序保证可复现。"""
    ids = {i for i, boxes in ann.items() if boxes}
    out = sorted(p for p in root.glob("*.jpg") if int(p.stem) in ids)
    return out[:limit] if limit else out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--coco-root", default="/root/autodl-tmp/Documents/datasets/COCO2017")
    ap.add_argument("--carla-root", default="outputs/kitti_ab_epic_clip2d_day_clear")
    ap.add_argument("--coco-native-weight", default="weights/yolo26x.pt")
    ap.add_argument("--kitti-weight", default=DEFAULT_YOLO_WEIGHT, help="AutoLabel 的 KITTI 微调权重")
    ap.add_argument("--conf", type=float, default=0.25, help="两个模型都是 yolo 家族 ⇒ 同一档可比")
    ap.add_argument("--limit", type=int, default=200, help="每侧最多几张(COCO 侧 GPU 成本线性)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--grid-points", type=int, default=101)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    from autodrivedata.perception.backends import YoloPredictor
    from autodrivedata.perception.eval_2d_ab import evaluate

    coco_root = project_path(args.coco_root)
    ann = load_coco_gt(coco_root / "annotations" / "instances_val2017.json")
    imgs = image_paths(coco_root / "val2017", ann, args.limit)
    if not imgs:
        raise SystemExit(f"{coco_root} 下找不到带本项目三类 GT 的图")
    print(f"[coco] {len(imgs)} 张(有本项目三类 GT 的,按文件名排序取前 {args.limit})")

    with runlog.run("autodrivedata.perception.domain_gap") as rl:
        rl.highlight("n_coco_images", len(imgs))
        rl.highlight("conf", args.conf)
        rl.highlight("grid_points", args.grid_points)
        matrix: dict[str, dict] = {}
        for name, weight in (("coco_native", args.coco_native_weight), ("kitti_ft", args.kitti_weight)):
            if not project_path(weight).is_file():
                raise SystemExit(f"{name} 的权重不在盘上:{weight}")
            predict = YoloPredictor(str(project_path(weight)), args.conf)
            print(f"\n### {name}({Path(weight).name})")
            print("  [COCO(真实)]")
            coco = evaluate_coco(imgs, ann, predict, iou=args.iou, n_points=args.grid_points, verbose=False)
            print(f"    mAP {coco['mAP']:.3f}({coco['n_classes']} 类)")
            print("  [COCO · GT 平移 1/3 的对照]")
            shuf = evaluate_coco(
                imgs,
                ann,
                predict,
                iou=args.iou,
                n_points=args.grid_points,
                control_shift=True,
                verbose=False,
            )
            print(f"    mAP {shuf['mAP']:.3f}  ← **必须塌到 ≈0**")
            print("  [CARLA(合成)]")
            carla = evaluate(
                project_path(args.carla_root),
                predict,
                args.conf,
                args.iou,
                args.limit,
                n_points=args.grid_points,
                verbose=False,
            )
            print(f"    mAP {carla['mAP']:.3f}({carla['n_classes']} 类)")
            matrix[name] = {"coco": coco, "coco_control": shuf, "carla": carla}
            rl.highlight(f"{name}_coco", round(coco["mAP"], 4))
            rl.highlight(f"{name}_coco_control", round(shuf["mAP"], 4))
            rl.highlight(f"{name}_carla", round(carla["mAP"], 4))

        print("\n=== ★ 2×2 迁移矩阵(mAP@0.5,101 点) ===")
        print(f"  {'模型':<16}{'COCO(真实)':>13}{'CARLA(合成)':>14}{'换域落差':>11}")
        for name, m in matrix.items():
            if name == "coco_native":
                d = m["carla"]["mAP"] - m["coco"]["mAP"]
                tag = "真实→合成"
            else:
                d = m["coco"]["mAP"] - m["carla"]["mAP"]
                tag = "合成→真实"
            print(f"  {name:<16}{m['coco']['mAP']:>13.3f}{m['carla']['mAP']:>14.3f}{d:>+11.3f}  ({tag})")
        worst_shuf = max(m["coco_control"]["mAP"] for m in matrix.values())
        # ★ **同类的公平比较**:CARLA 侧只有 `Car` 有 GT(那批数据不带行人/骑行者)
        #   ⇒ 拿 3 类 mAP 比 1 类 mAP **不是一回事**(coco_native 的 3 类 0.750 / 1 类 0.873)。
        #   逐类读才公平,而 `Car` 是唯一两边都有 GT 的类。
        print("\n=== ★ 同类可比(`Car`,唯一两边都有 GT 的类) ===")
        print(
            f"  {'模型':<16}{'COCO Car':>11}{'CARLA Car':>11}{'落差':>10}{'COCO 召回':>11}{'CARLA 召回':>12}"
        )
        for name, m in matrix.items():
            c, k = m["coco"]["classes"]["Car"], m["carla"]["classes"]["Car"]
            print(
                f"  {name:<16}{c['ap']:>11.3f}{k['ap']:>11.3f}{c['ap'] - k['ap']:>+10.3f}"
                f"{c['recall']:>11.3f}{k['recall']:>12.3f}"
            )
            rl.highlight(f"{name}_car_coco", round(c["ap"], 4))
            rl.highlight(f"{name}_car_carla", round(k["ap"], 4))
        print(
            f"\n  对照(GT 平移 1/3)最差臂 mAP = {worst_shuf:.3f}",
            "✅ 尺子立得住" if worst_shuf < 0.05 else "⚠️ 尺子没立住,读数不可用",
        )
        rl.highlight("shuffle_control_worst", round(worst_shuf, 4))

        if args.out:
            p = project_path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(matrix, indent=1, ensure_ascii=False), encoding="utf-8")
            rl.artifact(p, "report")
            print(f"[done] {p.resolve()}")


if __name__ == "__main__":
    main()

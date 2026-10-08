"""**域自适应:先试最笨的办法** —— 测试时把输入对齐到目标域的统计,不训练、不碰权重。

## 它接着哪一条

`perception/domain_gap` 量出:合成(KITTI 口径)上训出来的检测器搬到**真实 COCO** 上,
Car AP 从 **0.647 掉到 0.135**、召回从 0.705 掉到 **0.140**。⇒ 域差是真的、而且很大。

房子纪律(本周已被验证两次):**先量最笨的方法**,再谈要不要开训练线。

## ★ 方向很容易写反(复核当场指出的)

- ✅ **测试时把真实图对齐到目标域统计**再喂给**冻结**的检测器 —— 纯推理,是真正的 training-free;
- ❌ **把合成图对齐之后不重训** —— 检测器一点没变,COCO 的 AP **恒等不变**。那是个无意义的臂
  (第一版就是这么写的)。

## 三条臂 + 两条对照

| 臂 | 是什么 | 期望 |
|---|---|---|
| **baseline** | 原图直接喂 | 0.135(已归档) |
| ★ **align** | COCO 图对齐到 **CARLA 域**的统计 | 若"笨办法有用" ⇒ 明显上升 |
| **wrong-domain** | 对齐到另一个**不相干**的目标(COCO 自己的一半) | 应当**不动** ⇒ 说明"对齐到哪个域"是有影响的 |
| ★ **jitter(地板)** | 对齐到一个**抖动过的** CARLA 参考 | 这是**分辨率地板**:Δ 小于它就不许报(红线:跨权重 σ≈0.025) |

⚠️ **`ultralytics` 的 BGR 坑**:`model.predict(np.ndarray)` 按 **BGR** 解释数组,
喂 RGB 会**静默换 R/B 而 AP 照出**。⇒ 本模块一律**落盘成文件**再喂路径。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from autodrivedata.edit.harmonize import from_lab, lab_stats, reinhard_map, to_lab  # noqa: F401
from autodrivedata.perception.backends import DEFAULT_YOLO_WEIGHT, YoloPredictor
from autodrivedata.perception.domain_gap import evaluate_coco, image_paths, load_coco_gt
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path


def domain_reference(root: Path, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """一个域的 **LAB 参考统计** = 逐图 `lab_stats` 的**均值**。

    ⚠️ 用"数据集级均值"而不是"某一张图的统计":后者需要一个**配对规则**,
    而这里要的是一个**固定的、不需要标签的常量**。
    """
    ms, cs = [], []
    files = sorted((root / "training" / "image_2").glob("*.png"))[:limit]
    if not files:
        raise SystemExit(f"{root} 下没有图 —— 目标域参考取自哪里?")
    for p in files:
        m, c = lab_stats(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
        ms.append(m)
        cs.append(c)
    return np.mean(ms, axis=0), np.mean(cs, axis=0)


def align_image(img: np.ndarray, ref_m: np.ndarray, ref_c: np.ndarray) -> np.ndarray:
    """把一张图的光度对齐到参考统计(**整图**,不用掩膜)。"""
    lab = to_lab(img)
    m, c = lab_stats(img)
    return from_lab(reinhard_map(lab, m, c, ref_m, ref_c))


def materialize(src: list[Path], dst: Path, ref_m: np.ndarray, ref_c: np.ndarray) -> list[Path]:
    """对齐后**落盘**成一等公民的图片文件。

    ★ **刻意落盘而不是喂数组**:ultralytics 的 `predict(ndarray)` 走 BGR,
    喂 RGB 会静默换 R/B 而 AP 照出(复核点名)。落盘让"对齐"与"检测"两段彻底解耦。
    文件名沿用原 stem ⇒ `evaluate_coco` 的 GT 配对**逐字节不变**。
    """
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for p in src:
        img = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        Image.fromarray(align_image(img, ref_m, ref_c)).save(dst / p.name)
        out.append(dst / p.name)
    return out


def jitter_reference(ref_m: np.ndarray, ref_c: np.ndarray, *, scale: float, seed: int):
    """**地板臂**:把参考统计抖动一下 —— 对齐到一个"差不多但不对"的目标。

    ⚠️ 它量的**不是**"抖动会不会掉点",而是**这套流程的分辨率**:
    若 `align` 相对 baseline 的增益**小于** jitter 臂自己的波动,那个增益就不许报。
    """
    rng = np.random.default_rng(seed)
    return ref_m + rng.normal(scale=scale, size=ref_m.shape), ref_c


def run(args) -> dict:
    coco_root = project_path(args.coco_root)
    ann = load_coco_gt(coco_root / "annotations" / "instances_val2017.json")
    imgs = image_paths(coco_root / "val2017", ann, args.limit)
    if not imgs:
        raise SystemExit(f"{coco_root} 下找不到带本项目三类 GT 的图")
    carla_root = project_path(args.carla_root)

    ref_m, ref_c = domain_reference(carla_root, args.ref_images)
    # 对照:对齐到**另一个不相干的目标** —— 用 COCO 自己前一半的统计
    half = imgs[: max(1, len(imgs) // 2)]
    other_m, other_c = domain_reference_of_images(half)
    jit_m, jit_c = jitter_reference(ref_m, ref_c, scale=args.jitter, seed=args.seed)

    work = project_path(args.work)
    arms: dict[str, list[Path]] = {
        "baseline": imgs,
        "align_carla": materialize(imgs, work / "align_carla", ref_m, ref_c),
        "wrong_target": materialize(imgs, work / "wrong_target", other_m, other_c),
        "jitter_floor": materialize(imgs, work / "jitter_floor", jit_m, jit_c),
    }
    predict = YoloPredictor(str(project_path(args.weight)), args.conf)
    out: dict = {}
    for name, paths in arms.items():
        ev = evaluate_coco(paths, ann, predict, iou=args.iou, n_points=args.grid_points, verbose=False)
        # ★ 平移 GT 的那条**有效对照**照旧带上 —— 尺子必须每次都能失败
        ctrl = evaluate_coco(
            paths,
            ann,
            predict,
            iou=args.iou,
            n_points=args.grid_points,
            control_shift=True,
            verbose=False,
        )
        out[name] = {
            "mAP": ev["mAP"],
            "mAP_control_shift": ctrl["mAP"],
            "car": ev["classes"]["Car"],
        }
    base = out["baseline"]["mAP"]
    for name in out:
        out[name]["delta_vs_baseline"] = out[name]["mAP"] - base
    return {"n_images": len(imgs), "ref_images": args.ref_images, "arms": out}


def domain_reference_of_images(images: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    ms, cs = [], []
    for p in images:
        m, c = lab_stats(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
        ms.append(m)
        cs.append(c)
    return np.mean(ms, axis=0), np.mean(cs, axis=0)


#: 平移 GT 的对照**必须**塌到 ≈0;高于这个数说明尺子没立住(同 `domain_gap` 的口径)。
CONTROL_MUST_COLLAPSE = 0.05


def verdict(rep: dict) -> str:
    """★ **先用地板筛一遍,再分符号** —— 增益的**正负是两件完全不同的事**。

    ⚠️ 第一版只比 `|gain| > floor` 就判"有用",而实测 `gain = **−0.0516**`
    ⇒ 它把"**掉了 5 个点**"读成了"**有用**"。**方向反了**。
    """
    a = rep["arms"]
    base, al = a["baseline"]["mAP"], a["align_carla"]["mAP"]
    floor = max(abs(a["jitter_floor"]["mAP"] - base), abs(a["wrong_target"]["mAP"] - base))
    gain = al - base
    # ★ 对照是否**真的塌了**:平移 GT 之后 mAP 应当 ≈0。⚠️ 第一版写的是
    #   `abs(ctrl - mAP) > 0.5` —— **判反了**:尺子**正常**时 ctrl≈0 而 mAP≈0.08,
    #   那个差值只有 0.08(不触发);尺子**坏掉**时 ctrl≈mAP,差值是 **0**(也不触发)
    #   ⇒ 两条路都"通过",这条守卫**从来没有生效过**。判据是 ctrl **本身**高不高。
    ctrl_max = max(a[n]["mAP_control_shift"] for n in a)
    if ctrl_max > CONTROL_MUST_COLLAPSE:
        return f"未判(平移 GT 的对照没塌:最高 {ctrl_max:.4f} ⇒ 这把尺子这次不成立)"
    if -gain > floor:
        return (
            f"★ **最笨的办法不但没用,还有害**:对齐到目标域 **{gain:+.4f}**"
            f"(地板 {floor:.4f},Car AP {a['baseline']['car']['ap']:.3f} → {a['align_carla']['car']['ap']:.3f})"
            " ⇒ **training-free 的光度对齐这条路不通**,要动就得走训练臂"
        )
    if gain > floor:
        return (
            f"★ **最笨的办法有用**:对齐到目标域 {gain:+.4f}(地板 {floor:.4f})"
            " ⇒ 在开训练线之前,先把这条免费的做到位"
        )
    return f"★ **落在噪声里**:{gain:+.4f},而地板是 {floor:.4f} ⇒ 这个方向上量不出东西"


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--coco-root", default="/root/autodl-tmp/Documents/datasets/COCO2017")
    ap.add_argument("--carla-root", default="outputs/kitti_ab_epic_clip2d_day_clear")
    ap.add_argument("--weight", default=DEFAULT_YOLO_WEIGHT, help="KITTI 微调权重(域外那一档)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--ref-images", type=int, default=70, help="目标域参考用多少张 CARLA 图")
    ap.add_argument("--jitter", type=float, default=2.0, help="地板臂的参考抖动(px,LAB 量纲)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--grid-points", type=int, default=101)
    ap.add_argument("--work", default="/tmp/domain_adapt")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    with runlog.run("autodrivedata.perception.domain_adapt") as rl:
        for k in ("limit", "ref_images", "jitter", "conf", "grid_points"):
            rl.highlight(k, getattr(args, k))
        rep = run(args)
        rep["verdict"] = verdict(rep)
        print(
            f"\n=== 域自适应(测试时对齐,冻结权重)===\n  {'臂':<16}{'mAP':>8}{'Δ':>9}{'Car AP':>9}{'Car 召回':>10}"
        )
        for name, a in rep["arms"].items():
            print(
                f"  {name:<16}{a['mAP']:>8.4f}{a['delta_vs_baseline']:>+9.4f}"
                f"{a['car']['ap']:>9.3f}{a['car']['recall']:>10.3f}"
            )
        print(f"  ⇒ {rep['verdict']}")
        for name, a in rep["arms"].items():
            rl.highlight(f"{name}_mAP", round(a["mAP"], 4))
        rl.highlight("verdict", rep["verdict"])
        if args.out:
            p = project_path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
            rl.artifact(p, "report")
            print(f"[done] {p.resolve()}")


if __name__ == "__main__":
    main()

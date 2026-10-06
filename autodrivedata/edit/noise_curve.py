"""**强度 → ΔAP** 的标定曲线:人工注入退化扫一档,给另外两种退化当标尺。

## 它补的是哪一格

`downstream_eval` 量"**生成的**退化 vs **真值的**退化"。两者的**强度都不可控**:
真值的由 CARLA 的场景档决定,生成的由 prompt 决定(实测**极不稳**,同一意图换措辞差 4 倍)。

只有**人工注入**的强度是**我们设的** ⇒ 它能画出一条「强度 → 下游掉点」的单调曲线,
把另外两种退化**投影**上去:

- 投影落得上 ⇒ 它们与注入退化**同一种东西**(只是强度不同)
- 投影落不上(点离曲线很远) ⇒ **不是同一种机制** —— 这正是 §1.6 那条
  「生成的雾把物体画得清清楚楚」的**定量形态**

⚠️ 曲线本身也要有对照:注入强度 0 的那一档必须与基线**逐位相同**(否则"注入"这个动作本身
    就在改数据,后面的 Δ 全不可归因)。这条有判据(`tests/edit/test_degrade.py`)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autodrivedata.edit import degrade as DG
from autodrivedata.perception.backends import DEFAULT_YOLO_WEIGHT
from autodrivedata.perception.eval_2d_ab import describe, evaluate, make_predictor, resolve_conf
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path


def res_summary(ev: dict) -> dict:
    """把 `evaluate` 的明细压成"这一点的分辨率信息"(只留**有 GT 的类**)。

    ★ **必须记** —— 曲线的用途是插值/投影,而 11 点插值下 `n_tp` 每跨过
    `ceil(j/10·n_gt)` 就会**跳一格**(≤1/11 AP)。不记这个,台阶会被读成退化效应:
    实测 β=0.04 那点的 ΔAP `+0.0763` 里有 `+0.066` 是台阶(`tp 106 → 108`)。
    """
    return {
        c: {
            "n_gt": r["n_gt"],
            "n_tp": r["n_tp"],
            "recall": round(r["recall"], 4),
            "cliff_up": r["cliff_up"],  # 距上一个 recall 格点还差几个 TP;None = 已满格
        }
        for c, r in ev["classes"].items()
        if r["n_gt"]
    }


def run_curve(
    base: Path,
    work: Path,
    *,
    kinds: tuple[str, ...] = DG.KINDS,
    levels: dict[str, list[float]] | None = None,
    predict=None,
    conf: float = 0.5,
    iou: float = 0.5,
    frames: list[str] | None = None,
    n_points: int = 11,
) -> dict:
    """跑基线 + 每种注入的每一档。返回 `{kind: [{level, mAP, delta, res}, ...]}`。

    `n_points` = AP 的格点数。**默认 11 是归档口径**;若要拿曲线去拟合/投影,
    用 `edit/calibrate`(它默认 101)—— 11 点下曲线是台阶状的,见 `eval_2d_ab` 头注。
    """
    lv = levels or DG.DEF_LEVELS
    ev0 = evaluate(base, predict, conf, iou, None, n_points=n_points)
    base_map = ev0["mAP"]
    out: dict = {
        "base_mAP": round(base_map, 4),
        "base_res": res_summary(ev0),
        "n_points": n_points,
        "curve": {},
    }
    print(f"\n=== 基线 mAP = {base_map:.4f} ===")
    for kind in kinds:
        rows = []
        for level in lv[kind]:
            dst = work / f"{kind}_{level:g}"
            DG.degrade_root(base, dst, kind=kind, level=level, frames=frames)
            ev = evaluate(dst, predict, conf, iou, None, n_points=n_points)
            m = ev["mAP"]
            rows.append(
                {
                    "level": float(level),
                    "mAP": round(m, 4),
                    "delta": round(m - base_map, 4),
                    "res": res_summary(ev),
                }
            )
            print(f"  [{kind}={level:g}] mAP {m:.4f}  Δ {m - base_map:+.4f}")
        out["curve"][kind] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--base", required=True, help="基线 arm root(如 outputs/edit_p1/clear)")
    ap.add_argument("--work", required=True, help="注入产物落点")
    ap.add_argument("--frames", default="", help="只取这几帧(逗号或 a-b);**必须与另两种退化同一批**")
    ap.add_argument("--kinds", default=",".join(DG.KINDS))
    ap.add_argument(
        "--levels",
        default="",
        help="覆盖默认档位,格式 `kind=l1,l2;kind2=l3`(如 `fogdepth=0.05,0.07,0.09`)。"
        "**细扫只用它** —— 默认档是粗的,而投影对表需要曲线在目标 Δ 附近有足够的采样点",
    )
    ap.add_argument("--backend", choices=("sam3", "yolo"), default="sam3")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--weight", default=DEFAULT_YOLO_WEIGHT)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument(
        "--grid-points",
        type=int,
        default=11,
        help="AP 的格点数。**默认 11 = 归档口径**;要拿曲线去拟合/投影请调大(见 `edit/calibrate`)",
    )
    ap.add_argument("--out", default="", help="曲线 JSON 落点(默认 <work>/curve.json)")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    base, work = project_path(args.base), project_path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    frames = DG._parse_frames(args.frames) if args.frames else None
    with runlog.run("autodrivedata.edit.noise_curve") as rl:
        rl.input(base, "base-root")
        conf = resolve_conf(args.backend, args.conf)
        rl.highlight("backend", args.backend)
        rl.highlight("conf", conf)
        rl.highlight("iou", args.iou)
        rl.highlight("grid_points", args.grid_points)
        rl.highlight("n_frames", len(frames) if frames else "all")
        if frames is None:
            rl.note("未给 --frames ⇒ 与 downstream_eval 的 35 帧**不是同一批**,Δ 不可直接比")
        predict = make_predictor(args.backend, args.weight, conf)
        print(describe(args.backend, predict))
        res = run_curve(
            base,
            work,
            kinds=tuple(args.kinds.split(",")),
            levels=_parse_levels(args.levels),
            predict=predict,
            conf=conf,
            iou=args.iou,
            frames=frames,
            n_points=args.grid_points,
        )
        res["conf"] = conf
        res["iou"] = args.iou
        res["grid_points"] = args.grid_points
        res["n_frames"] = len(frames) if frames else "all"
        dst = project_path(args.out) if args.out else work / "curve.json"
        dst.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        for kind, rows in res["curve"].items():
            rl.highlight(f"delta_{kind}", [r["delta"] for r in rows])
        rl.artifact(dst, "curve")
        print(f"[done] {dst}")


def _parse_levels(spec: str) -> dict[str, list[float]] | None:
    """`kind=l1,l2;kind2=l3` → `{kind: [l1, l2]}`。空串 ⇒ `None`(用默认档)。

    ⚠️ **只覆盖点到名的 kind** —— 没点到的仍走 `DEF_LEVELS`,而不是被清空。
    (那正是"只跑 `--kinds fogdepth` 时另外三种静静消失"这类失效的形态。)
    """
    if not spec.strip():
        return None
    out: dict[str, list[float]] = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"--levels 的每一段要 `kind=l1,l2`,收到 {chunk!r}")
        k, v = chunk.split("=", 1)
        k = k.strip()
        if k not in DG.KINDS:
            raise SystemExit(f"--levels 里的 kind {k!r} 不认识;可选 {list(DG.KINDS)}")
        out[k] = [float(x) for x in v.split(",") if x.strip()]
    return out


def fit_equivalent_level(curve: list[dict], delta: float) -> float | None:
    """把某个 Δ 投影回曲线上 → 等价的注入强度(**线性内插**,超出范围返回 None)。

    这是这张曲线的**全部用途**:`Δ_truth = −0.170` 相当于多少"模糊强度"?
    `Δ_gen = +0.010` 根本**落不上**(它不在退化侧)⇒ 返回 None,**那本身就是结论**。
    """
    pts = sorted((r["level"], r["delta"]) for r in curve)
    if not pts or delta > max(d for _, d in pts):
        return None  # 比最轻的一档还轻 ⇒ 落在曲线外(=没退化)
    for (l0, d0), (l1, d1) in zip(pts, pts[1:], strict=False):
        if (d0 >= delta >= d1) or (d0 <= delta <= d1):
            if d1 == d0:
                return float(l0)
            t = (delta - d0) / (d1 - d0)
            return float(l0 + t * (l1 - l0))
    return float(pts[-1][0]) if delta <= pts[-1][1] else None


if __name__ == "__main__":
    main()

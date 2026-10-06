"""**β 标定**:一条命令把「要解释的退化 → 该施加多强的 `fogdepth`」跑出来。

## 它解决哪件事

`noise_curve` 建曲线之后,用法是**人工看表 → 人工内插 → 人工抄命令**。而曲线是
**底图相关**的(同一个 β 在生成图上掉得更多,见 `docs/edit-image-plan.md` §1.7),
⇒ 每换一个生成基底 / 条件源 / 场景就要重来一遍。这个模块把那三步合成一条命令。

## ★ 为什么默认走 **101 点**而不是归档的 11 点

11 点插值在 `n_gt ≈ 100` 时会让 AP 在 max-recall 跨格处**跳 `p/11 ≤ 0.091`**
(见 `perception/eval_2d_ab` 模块头注)。后果是曲线**不光滑,甚至不单调** ——
实测 `blur` 曲线在 11 点下是 `−0.013 / +0.055 / −0.034`,在 101 点下是单调下降。
⇒ 拿 11 点的曲线去**拟合/内插**,得到的是尺子的台阶,不是退化效应。

⚠️ **归档口径仍是 11 点**(全仓历史数字都是它),两者**不可混比**。所以产物里
**两个口径的数都记**,而 `--grid-points` 决定**哪一条用来拟合**。

## 判据

`fit` 落不上(目标比曲线的**最轻一档还轻**)⇒ 返回 `None`,**那本身就是结论**:
这类退化不在本曲线覆盖的强度范围内(§1.6 的"生成雾比真雾轻一个量级"正是这一支)。

⚠️ **必报的三件东西**(缺一件读数就不可读):① 用的哪个口径与哪个 backend / conf;
② 曲线每个点的 `n_tp/n_gt` 与**距 recall 格点还剩几个 TP**(分辨率);
③ 拟合点的**两侧相邻档**(内插是线性的,跨档太远就是外推)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autodrivedata.edit import degrade as DG
from autodrivedata.edit.noise_curve import fit_equivalent_level, res_summary
from autodrivedata.perception.backends import DEFAULT_YOLO_WEIGHT
from autodrivedata.perception.eval_2d_ab import describe, evaluate, make_predictor, resolve_conf
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 标定曲线的默认档位。**细扫**是刻意的 —— 拟合要的是目标 Δ 附近的采样密度。
DEF_LEVELS = "0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.12"


def is_monotone(curve: list[dict]) -> bool:
    """Δ 随强度**单调不增**。不单调 ⇒ 曲线里混着台阶或噪声,拟合结果不可信。"""
    ds = [r["delta"] for r in sorted(curve, key=lambda r: r["level"])]
    return all(a >= b for a, b in zip(ds, ds[1:], strict=False))


def build_curve(
    base: Path,
    work: Path,
    *,
    kind: str,
    levels: list[float],
    predict,
    conf: float,
    iou: float,
    n_points: int,
    frames: list[str] | None = None,
) -> dict:
    """跑基线 + 每一档,返回 `{base_mAP, base_res, n_points, curve:[...]}`。**不做裁决。**"""
    ev0 = evaluate(base, predict, conf, iou, None, n_points=n_points)
    base_map = ev0["mAP"]
    print(f"\n=== 基线 mAP({n_points} 点) = {base_map:.4f} ===")
    rows = []
    for level in levels:
        dst = work / f"{kind}_{level:g}"
        DG.degrade_root(base, dst, kind=kind, level=level, frames=frames)
        ev = evaluate(dst, predict, conf, iou, None, n_points=n_points)
        rows.append(
            {
                "level": float(level),
                "mAP": round(ev["mAP"], 4),
                "delta": round(ev["mAP"] - base_map, 4),
                "res": res_summary(ev),
            }
        )
        print(f"  [{kind}={level:g}] mAP {ev['mAP']:.4f}  Δ {ev['mAP'] - base_map:+.4f}")
    return {"base_mAP": round(base_map, 4), "base_res": res_summary(ev0), "n_points": n_points, "curve": rows}


def target_delta_from_arms(ref: Path, arm: Path, *, predict, conf: float, iou: float, n_points: int) -> float:
    """`Δ = AP(arm) − AP(ref)` —— **两个 root 必须在同一口径、同一 backend、同一 conf 下评**。"""
    a = evaluate(ref, predict, conf, iou, None, n_points=n_points)["mAP"]
    b = evaluate(arm, predict, conf, iou, None, n_points=n_points)["mAP"]
    print(f"\n  [目标退化] AP(ref)={a:.4f}  AP(arm)={b:.4f}  Δ={b - a:+.4f}")
    return b - a


def recipe(
    base: Path, res: dict, target: float, kind: str, *, n_points: int, backend: str, conf: float
) -> dict:
    """把曲线 + 目标 Δ 拧成**配方**(β + 可粘贴的命令 + 分辨率自述)。"""
    curve = res["curve"]
    beta = fit_equivalent_level(curve, target)
    mono = is_monotone(curve)
    out: dict = {
        "kind": kind,
        "backend": backend,
        "conf": conf,
        "grid_points": n_points,
        "target_delta": round(target, 4),
        "beta": None if beta is None else round(beta, 4),
        "curve_monotone": mono,
        "base_root": str(base),
    }
    if beta is None:
        if not curve:  # 空曲线:没有"最轻的一档"可比,别让 max() 抛在不该抛的地方
            out["verdict"] = "⚠ 曲线为空 ⇒ 无可标定(检查 `--levels` 与 `--work` 是否写对)"
            return out
        lightest = max(r["delta"] for r in curve)
        out["verdict"] = (
            f"★ 目标 Δ {target:+.4f} 比曲线最轻的一档({lightest:+.4f})还轻 ⇒ **落不上曲线**。"
            "这不是「标定失败」,是结论:该退化的强度不在本曲线覆盖的范围内"
        )
        return out
    lo = max((r for r in curve if r["level"] <= beta), key=lambda r: r["level"], default=None)
    hi = min((r for r in curve if r["level"] >= beta), key=lambda r: r["level"], default=None)
    out["bracket"] = {"low": lo, "high": hi}
    if lo is None or hi is None:  # 只可能在曲线为空时发生;不静默假装内插过
        out["verdict"] = "⚠ 曲线为空或档位全在一侧 ⇒ 未内插"
        return out
    out["recipe"] = (
        f"python -m autodrivedata.edit.degrade --src {base} --dst <产物root> --kind {kind} --level {beta:.4f}"
    )
    if not mono:
        out["verdict"] = (
            "⚠ 曲线**不单调** ⇒ 拟合结果不可信。先看每个点的 `res`(台阶余量)"
            f"或把 `--grid-points` 调大(当前 {n_points})"
        )
    else:
        out["verdict"] = f"★ β = {beta:.4f}(在 {lo['level']:g} 与 {hi['level']:g} 之间线性内插)"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--base", required=True, help="要标定的底图 root(校准曲线建在它上面)")
    ap.add_argument("--work", required=True, help="逐档注入产物的落点")
    ap.add_argument("--kind", default="fogdepth", choices=DG.KINDS)
    ap.add_argument("--levels", default=DEF_LEVELS, help=f"逗号分隔的强度档(默认 {DEF_LEVELS})")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--target-delta", type=float, help="直接给要匹配的 ΔAP(如真值浓雾的 −0.209)")
    src.add_argument(
        "--target-arms",
        default="",
        help="`<参考root>:<退化root>` —— 在**同一口径**下现算 Δ = AP(退化) − AP(参考)",
    )
    ap.add_argument(
        "--grid-points", type=int, default=101, help="标定曲线的 AP 格点数(**默认 101**,见模块头注)"
    )
    ap.add_argument("--apply", default="", help="给了就把拟合出的 β 直接施加到该 root")
    ap.add_argument("--frames", default="", help="只取这几帧(**必须与目标 Δ 的评测集同一批**)")
    ap.add_argument("--backend", choices=("sam3", "yolo"), default="yolo")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--weight", default=DEFAULT_YOLO_WEIGHT)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default="", help="配方 JSON 落点(默认 <work>/calibrate.json)")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    base, work = project_path(args.base), project_path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    frames = DG._parse_frames(args.frames) if args.frames else None
    levels = [float(x) for x in args.levels.split(",") if x.strip()]
    with runlog.run("autodrivedata.edit.calibrate") as rl:
        rl.input(base, "base-root")
        conf = resolve_conf(args.backend, args.conf)
        rl.highlight("backend", args.backend)
        rl.highlight("conf", conf)
        rl.highlight("grid_points", args.grid_points)
        rl.highlight("kind", args.kind)
        rl.highlight("n_frames", len(frames) if frames else "all")
        if frames is None:
            rl.note("未给 --frames ⇒ 与任何带 --frames 的目标 Δ **不是同一批评测集**,Δ 不可直接比")
        predict = make_predictor(args.backend, args.weight, conf)
        print(describe(args.backend, predict))

        if args.target_arms:
            if ":" not in args.target_arms:
                raise SystemExit("--target-arms 要 `<参考root>:<退化root>`")
            a, b = (project_path(x) for x in args.target_arms.split(":", 1))
            rl.input(a, "target-ref")
            rl.input(b, "target-arm")
            target = target_delta_from_arms(
                a, b, predict=predict, conf=conf, iou=args.iou, n_points=args.grid_points
            )
        else:
            target = args.target_delta

        res = build_curve(
            base,
            work,
            kind=args.kind,
            levels=levels,
            predict=predict,
            conf=conf,
            iou=args.iou,
            n_points=args.grid_points,
            frames=frames,
        )
        rec = recipe(base, res, target, args.kind, n_points=args.grid_points, backend=args.backend, conf=conf)
        res["recipe"] = rec
        res["target_delta"] = round(target, 4)
        print("\n=== ★ 配方 ===")
        print(f"  目标 Δ(口径 {args.grid_points} 点) = {target:+.4f}")
        print(f"  曲线单调 = {rec['curve_monotone']}")
        print(f"  ⇒ {rec['verdict']}")
        if rec["beta"] is not None:
            b0, b1 = rec["bracket"]["low"], rec["bracket"]["high"]
            print(f"  内插区间:[{b0['level']:g} Δ{b0['delta']:+.4f}] ↔ [{b1['level']:g} Δ{b1['delta']:+.4f}]")
            print(f"  {rec['recipe']}")
        rl.highlight("target_delta", round(target, 4))
        rl.highlight("beta", rec["beta"])
        rl.highlight("curve_monotone", rec["curve_monotone"])
        for r in res["curve"]:
            rl.highlight(f"delta_{r['level']:g}", r["delta"])

        if args.apply and rec["beta"] is not None:
            dst = project_path(args.apply)
            n = DG.degrade_root(base, dst, kind=args.kind, level=rec["beta"], frames=frames)
            print(f"[apply] {rec['beta']} → {dst}({n} 帧)")
            rl.highlight("applied_to", str(dst))

        dst = project_path(args.out) if args.out else work / "calibrate.json"
        dst.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        rl.artifact(dst, "calibrate")
        print(f"[done] {dst}")


if __name__ == "__main__":
    main()

"""编辑判据:删完之后,**离真值靶还有多远**。

## 为什么是三档,少一档就解读不了

| 档 | 读作 |
|---|---|
| `PSNR(A_render, B_img)` | **不编辑的基线** —— 编辑要补的缺口有多大。它太小 ⇒ 这个场景/这个物体**量不出东西**,先换物体 |
| `PSNR(edited_render, B_img)` | 删完之后的实际水平 |
| `PSNR(B_render, B_img)` | ★ **天花板** —— 重建本身的误差。不报它,"20 dB"是好是坏无从判断 |

只报中间那一档是本仓反复记过的形态:**一个孤立的正数读不出好坏**。

## 只在物体的掩膜区里算

全场均值会把窟窿**稀释掉**(物体在 621×187 里只占几百像素)。
掩膜取 **A/B 两张真采图的 RGB 差** —— 它**是真值**(两张实拍相减,不经过任何模型)。

⚠️ **不能用 `inst == instance_id`**(原方案的写法)。**CARLA 的深度 / 语义 / 实例三个 pass
根本不渲染 `static.prop.*`** —— 2026-10-04 实测:A/B 在这三路上**逐像素完全相同**
(深度 0 差、语义 0 差、实例 0 差),而 **RGB 上差 57,664 px**。
⇒ 按 `instance_id` 取掩膜会得到 **0 个像素**,然后被读成"道具没露面",
而真因是"这个通道根本不存在它"。`--mask-source inst` 保留着,就是为了让这个对照随时可复现。

⚠️ **掩膜为空 = 未判**,不是通过也不是失败 —— 与 `static_eval` 的「`n=0` 是不判」同一条。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from autodrivedata.gs import cuda_env  # noqa: F401  —— 与 render_gs 同一纪律(见其头注)
from autodrivedata.gs.render_gs import load_set, mse_to_psnr, parse_frames
from autodrivedata.perception.inst_tags import decode_instance_png
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: A/B 位姿配对的容差(米 / 度)。`ring_cam_pose` 是纯函数 + 同一个 spawn point
#: ⇒ 理论上**恰好 0**,给一个极小的容差只为吸收 JSON 的舍入。比 P1 的 0.07 m 严得多。
PAIR_TOL = 1e-6

#: 掩膜外扩(像素,在渲染分辨率上)。物体的轮廓之外还有一圈软边/阴影,
#: ⚠️ 这是**定标**量:放宽会混进背景像素、把三档一起拉平(同 `static_eval` 的邻域教训)。
MASK_DILATE_PX = 1

#: RGB 差掩膜的阈值(归一化 0–1)。实测道具区最大差 **54/255=0.21**、而 A/B 的
#: 背景噪声远小于 **2/255** ⇒ 取 8/255 在两者之间,既不吃噪声也不漏边缘。
MASK_RGB_THRESH = 8 / 255


def pose_pairing(cap_a: Path, cap_b: Path) -> dict:
    """★ **A/B 配对硬门槛**(C0.3):两次采集的位姿必须逐位相同。

    这是 P1 那条「A/B 只许变一个变量」在 3DGS 层的形态 —— 不成立则**整套判据作废**,
    因为"删完的渲染 vs B 的图"里混进了"两边根本不是同一个视角"。
    ⚠️ 这里能要求**恰好相等**(`ring_cam_pose` 纯函数 + 同一 spawn point),
    比 P1 的 0.07 m 严 —— **别把它放宽成 P1 那个数**:两者量的是不同的东西。
    """
    out = {}
    for name in ("pitches.json",):
        if not (cap_a / name).is_file() or not (cap_b / name).is_file():
            raise SystemExit(f"{cap_a} 或 {cap_b} 不是 capture(缺 {name})")
    pa = json.loads((cap_a / "pitches.json").read_text(encoding="utf-8"))
    pb = json.loads((cap_b / "pitches.json").read_text(encoding="utf-8"))
    if pa != pb:
        raise SystemExit(f"两侧的 pitches 不同:{pa} vs {pb} —— 不是同一套环绕")
    worst, worst_at = 0.0, None
    for p in pa:
        fa = json.loads((cap_a / f"poses_{int(p)}.json").read_text(encoding="utf-8"))
        fb = json.loads((cap_b / f"poses_{int(p)}.json").read_text(encoding="utf-8"))
        if len(fa) != len(fb):
            raise SystemExit(f"pitch {p} 两侧帧数不同:{len(fa)} vs {len(fb)}")
        for ra, rb in zip(fa, fb, strict=True):
            if ra["i"] != rb["i"]:
                raise SystemExit(f"pitch {p} 帧号对不上:{ra['i']} vs {rb['i']}")
            for key in ("x", "y", "z", "pitch", "yaw", "roll"):
                d = abs(float(ra[key]) - float(rb[key]))
                if d > worst:
                    worst, worst_at = d, (int(p), ra["i"], key)
    out["max_abs_delta"] = worst
    out["worst_at"] = worst_at
    out["paired"] = worst <= PAIR_TOL
    return out


def _dilate(m: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return m
    return F.max_pool2d(m.float()[None, None], 2 * k + 1, stride=1, padding=k)[0, 0] > 0.5


def build_mask_rgb_diff(
    cap_a: Path,
    cap_b: Path,
    frames: list,
    pitch_of,
    imgs_a: torch.Tensor,
    imgs_b: torch.Tensor,
    h: int,
    w: int,
    dilate: int,
    thresh: float = MASK_RGB_THRESH,
):
    """★ **A/B 的 RGB 差**当掩膜(渲染分辨率)。真值:两张实拍相减,不经过模型。

    为什么只能是它:见模块头注 —— CARLA 的深度/语义/实例 pass **都不渲染 `static.prop.*`**。
    ⚠️ 它框的是"物体**改变了画面的地方**"(含阴影/软边),不是"物体的轮廓" ——
    对"这次编辑该作用在哪"这个问题,前者才是对的。
    """
    out = []
    for j, _fr in enumerate(frames):
        d = (imgs_a[j] - imgs_b[j]).abs().max(dim=-1)[0]
        out.append(_dilate(d > thresh, dilate))
    return out


def build_mask_inst(cap_a: Path, frames: list, pitch_of, prop_id: int, h: int, w: int, dilate: int):
    """`inst == prop_id`(渲染分辨率)。⚠️ **在本链路上会得到 0 个像素** —— 见模块头注。

    保留它只为让"这个通道不渲染道具"这条对照随时可复现。
    """
    out = []
    for fr in frames:
        p = cap_a / "inst" / f"p{int(pitch_of(fr))}" / f"{fr:05d}.png"
        if not p.is_file():
            raise SystemExit(f"{p} 不存在 —— A 侧没用 `--inst` 采")
        ids = decode_instance_png(p.read_bytes())
        m = torch.from_numpy((ids == prop_id).astype(np.float32))[None, None]
        out.append(_dilate(F.interpolate(m, size=(h, w), mode="nearest")[0, 0] > 0.5, dilate))
    return out


def three_tier(
    rend_a: torch.Tensor,
    rend_edited: torch.Tensor,
    rend_b: torch.Tensor,
    img_b: torch.Tensor,
    masks: list[torch.Tensor],
) -> dict:
    """**纯函数**:三档 PSNR,全局与掩膜内各算一份。"""
    n = rend_a.shape[0]
    rows = []
    for i in range(n):
        m = masks[i]
        n_px = int(m.sum())
        # 显式标注:`None` 是**未判**的第三态(掩膜为空时那一档没有值),不许被当成 0
        row: dict[str, object] = {"i": i, "n_px": n_px}
        for name, r in (("a", rend_a), ("edited", rend_edited), ("b", rend_b)):
            se = ((r[i] - img_b[i]) ** 2).mean(dim=-1)
            row[f"{name}_all"] = mse_to_psnr(float(se.mean()))
            row[f"{name}_mask"] = mse_to_psnr(float(se[m].mean())) if n_px else None
        rows.append(row)

    judged = [r for r in rows if r["n_px"]]
    agg: dict[str, object] = {
        "per_frame": rows,
        "n_judged": len(judged),
        "n_frames": n,
        "n_px_total": int(sum(r["n_px"] for r in rows)),
    }
    if not judged:
        # ⚠️ **未判** —— 不是通过也不是失败。掩膜空 = 这个物体在这些帧里根本没露面。
        agg["verdict"] = "未判"
        agg["reason"] = "掩膜一片空白:道具在这些帧里没有出现(换帧,或查 A 侧是否真的摆上了)"
        return agg
    for name in ("a", "edited", "b"):
        agg[f"{name}_mask_mean"] = float(np.mean([r[f"{name}_mask"] for r in judged]))
        agg[f"{name}_all_mean"] = float(np.mean([r[f"{name}_all"] for r in judged]))
    gap = float(agg["b_mask_mean"]) - float(agg["a_mask_mean"])
    closed = float(agg["edited_mask_mean"]) - float(agg["a_mask_mean"])
    agg["gap_total_db"] = gap
    agg["gap_closed_db"] = closed
    agg["gap_closed_frac"] = (closed / gap) if abs(gap) > 1e-9 else None
    agg["edited_vs_ceiling_db"] = float(agg["edited_mask_mean"]) - float(agg["b_mask_mean"])
    agg["verdict"] = "可判"
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--gs-dir", default="outputs/3dgs")
    ap.add_argument("--capture-a", required=True, help="A 侧 capture(带 --inst 采的)")
    ap.add_argument("--capture-b", required=True, help="B 侧 capture(位姿必须逐位相同)")
    ap.add_argument("--a-tag", required=True, help="A 侧训出的高斯 tag")
    ap.add_argument("--b-tag", required=True, help="B 侧训出的高斯 tag(天花板)")
    ap.add_argument("--edited-tag", required=True, help="`edit_gs --out-tag` 的产物")
    ap.add_argument("--prop-json", required=True, help="`capture_a/prop.json`(读 instance_id)")
    ap.add_argument("--frames", default="45,90,135,180,225", help="评测帧(`0-9` / `0,45` / `all`)")
    ap.add_argument("--mask-dilate", type=int, default=MASK_DILATE_PX)
    ap.add_argument(
        "--mask-source",
        choices=("rgb-diff", "inst"),
        default="rgb-diff",
        help="掩膜来源。`rgb-diff`(**默认**)= A/B 两张真采图的 RGB 差;"
        "`inst` = 实例图 == `instance_id` —— ⚠️ CARLA 的实例/深度/语义 pass **不渲染 "
        "`static.prop.*`**,这条会得到 0 像素(保留只为让该对照可复现)",
    )
    ap.add_argument("--out", default="", help="落 JSON")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    cap_a, cap_b = project_path(args.capture_a), project_path(args.capture_b)
    gs_dir = project_path(args.gs_dir)
    dev = "cuda"

    with runlog.run("autodrivedata.gs.eval_edit") as rl:
        rl.input(args.capture_a, "capture-a")
        rl.input(args.capture_b, "capture-b")
        print(f"[arch] {cuda_env.ARCH_NOTE}")
        print(f"[cuda] {cuda_env.CUDA_NOTE}")
        rl.highlight("cuda_home", cuda_env.home())

        # ★ 硬门槛:配对不成立 ⇒ **不出数**(那些数没法归因)
        pair = pose_pairing(cap_a, cap_b)
        print(f"[pair] 位姿最大差 {pair['max_abs_delta']:.3e}(最差处 {pair['worst_at']})")
        rl.highlight("pair_max_abs_delta", pair["max_abs_delta"])
        if not pair["paired"]:
            raise SystemExit(
                f"A/B 位姿对不上(最大差 {pair['max_abs_delta']:.3e} > {PAIR_TOL})—— "
                "两侧根本不是同一视角,判据作废。**别放宽这个容差**"
            )

        prop = json.loads((cap_a / "prop.json").read_text(encoding="utf-8"))["prop"]
        prop_id = int(prop["instance_id"])
        print(f"[prop] instance_id={prop_id} model={prop['type_id']}")
        rl.highlight("prop_instance_id", prop_id)

        from autodrivedata.gs.train_3dgs_mini import _load_images, _load_poses_and_cams

        poses, _f, h, w, _cx, _cy, viewmats, ks = _load_poses_and_cams(cap_a, downsample=2)
        viewmats, ks = viewmats.to(dev), ks.to(dev)
        idx = parse_frames(args.frames, len(poses))
        pitch_of = lambda k: poses[k]["pitch"]  # noqa: E731
        imgs_b = _load_images(cap_b, poses, h, w).to(dev)
        if args.mask_source == "rgb-diff":
            imgs_a = _load_images(cap_a, poses, h, w).to(dev)
            masks = build_mask_rgb_diff(
                cap_a, cap_b, idx, pitch_of, imgs_a[idx], imgs_b[idx], h, w, args.mask_dilate
            )
        else:
            masks = build_mask_inst(cap_a, idx, pitch_of, prop_id, h, w, args.mask_dilate)
        n_px = int(sum(int(m.sum()) for m in masks))
        print(f"[mask] 源={args.mask_source} | {len(idx)} 帧,掩膜共 {n_px} 像素(外扩 {args.mask_dilate} px)")
        for j, i in enumerate(idx):
            print(f"        帧 {i:>3}(pitch {int(pitch_of(i)):>3}°): {int(masks[j].sum()):>6} px")
        rl.highlight("mask_source", args.mask_source)
        rl.highlight("mask_px_total", n_px)
        if n_px == 0:
            print("[mask] ⚠️ 掩膜为空 ⇒ 这次**未判**(不是通过)")

        with torch.no_grad():
            ra = load_set(gs_dir, args.a_tag).render(idx, viewmats, ks, w, h, dev)
            re_ = load_set(gs_dir, args.edited_tag).render(idx, viewmats, ks, w, h, dev)
            rb = load_set(gs_dir, args.b_tag).render(idx, viewmats, ks, w, h, dev)

        rep = three_tier(ra, re_, rb, imgs_b, masks)
        rep["prop"] = prop
        rep["pairing"] = pair
        rep["frames"] = idx
        rep["mask_dilate"] = args.mask_dilate
        rep["tags"] = {"a": args.a_tag, "edited": args.edited_tag, "b": args.b_tag}

        print(f"[verdict] {rep['verdict']}")
        if rep["verdict"] == "可判":
            print(
                f"[mask 内] 不编辑 {rep['a_mask_mean']:.2f} | 编辑后 {rep['edited_mask_mean']:.2f}"
                f" | 天花板 {rep['b_mask_mean']:.2f} dB"
            )
            print(
                f"[缺口] 总共 {rep['gap_total_db']:+.2f} dB,补上 {rep['gap_closed_db']:+.2f}"
                f"({rep['gap_closed_frac'] * 100:.0f}%)"
            )
            print(f"[离天花板] {rep['edited_vs_ceiling_db']:+.2f} dB")
            rl.highlight("a_mask_mean", round(rep["a_mask_mean"], 3))
            rl.highlight("edited_mask_mean", round(rep["edited_mask_mean"], 3))
            rl.highlight("b_mask_mean", round(rep["b_mask_mean"], 3))
            rl.highlight("edited_vs_ceiling_db", round(rep["edited_vs_ceiling_db"], 3))
            rl.highlight("gap_closed_frac", round(rep["gap_closed_frac"] or 0.0, 4))
        rl.highlight("verdict", rep["verdict"])

        if args.out:
            p = project_path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"[done] {p.resolve()}")
            rl.artifact(p, "eval-edit")
        else:
            print(
                json.dumps({k: v for k, v in rep.items() if k != "per_frame"}, indent=1, ensure_ascii=False)
            )

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

⚠️ **两个掩膜源回答的是两个不同的问题**,不是"哪个对哪个错":

| 源 | 框的是什么 | 2026-10-07 实测(新 capture) |
|---|---|---|
| `rgb-diff`(**默认**) | **物体改变了画面的地方**(含阴影/软边) | 390448 px |
| `inst` | **物体本体的像素** | 3705 px |

⚠️⚠️ **上面那句"CARLA 的实例 pass 不渲染 `static.prop.*` ⇒ 会得 0 个像素"已被推翻**
(2026-10-07):修好位姿口径后采的 capture 里,道具 `instance_id=121` **有像素**,
`inst` 掩膜 **3705 px**(5 帧都有),归属那一步也实打实挂上了 **1100 个高斯**。
当年那条读数的因是 **79.132 m 的相机位姿偏移**(道具在 73 m 外,根本不在视场)——
见 [docs/edit-3dgs-plan.md](../../docs/edit-3dgs-plan.md) §C.0.4 ③ 与 §C.1。
⇒ **`inst` 不再是"对照/反例",它是一个有意义的紧掩膜**;`rgb-diff` 仍然是默认,
因为它含阴影与软边 —— 对"这次编辑该作用在哪"这个问题,那个才是对的。

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

#: ★ **A/B 在道具之外的 p90 亮度差**上限(归一化 0–1;2026-10-07 实测定标)。
#: 干净对 `3dgs_ab` = **0.00**;变了天气的 `3dgs_ab2` = **0.49** ⇒ 取 0.05(≈13/255),
#: **离两边各有约 10×**,不是为了卡着某个数选的。
PAIR_MAX_P90_DIFF = 0.05

#: 判定"A/B 只差道具"时,把道具区域排除掉的外扩量。**必须远大于 `MASK_DILATE_PX`**:
#: 道具的**软边与阴影**也合法地改变画面,拿紧掩膜排除它们会让每一对合法 capture 都被拒判。
PAIR_EXCLUDE_DILATE_PX = 16


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

    为什么**默认**是它(不是"只能是它"):见模块头注 —— 它含**软边与阴影**,
    而那正是"物体改变了画面的地方"。`inst` 是更紧的另一把尺子(2026-10-07 起可用)。
    ⚠️ 它框的是"物体**改变了画面的地方**"(含阴影/软边),不是"物体的轮廓" ——
    对"这次编辑该作用在哪"这个问题,前者才是对的。
    """
    out = []
    for j, _fr in enumerate(frames):
        d = (imgs_a[j] - imgs_b[j]).abs().max(dim=-1)[0]
        out.append(_dilate(d > thresh, dilate))
    return out


def build_mask_inst(cap_a: Path, frames: list, poses: list, prop_id: int, h: int, w: int, dilate: int):
    """`inst == prop_id`(渲染分辨率)。

    ## ★★ 被修掉的那个缺陷(2026-10-07)

    原实现拼路径用的是**全局帧号**:

        f"p{int(pitch_of(fr))}" / f"{fr:05d}.png"      # ← fr 是全局序

    而采集器落盘用的是**逐俯仰的 `i`**(`train_3dgs_mini._frame_path` 取 `pose["i"]`)。
    默认 `--frames 45,90,135,180,225` 在 3×90 的 capture 上,第二个索引 90 会去找
    `p-15/00090.png` —— **那个文件永远不存在**(逐俯仰只有 00000–00089)
    ⇒ 这条路径**在本仓一次都没跑成过**,直接 `SystemExit`。
    ⇒ 「它会给 0 个像素」这句话**从来没被实测过**。

    ⚠️ 现在按 `pose["i"]` 取,并收 `poses` 而不是 `pitch_of` —— 一个闭包只能给一个字段,
    而这里**两个字段都要**(目录用 `pitch`、文件名用 `i`);给一个漏一个正是上面那个错。
    """
    out = []
    for fr in frames:
        pose = poses[fr]
        p = cap_a / "inst" / f"p{int(pose['pitch'])}" / f"{int(pose['i']):05d}.png"
        if not p.is_file():
            raise SystemExit(f"{p} 不存在 —— A 侧没用 `--inst` 采")
        ids = decode_instance_png(p.read_bytes())
        m = torch.from_numpy((ids == prop_id).astype(np.float32))[None, None]
        out.append(_dilate(F.interpolate(m, size=(h, w), mode="nearest")[0, 0] > 0.5, dilate))
    return out


def pair_weather(cap_a: Path, cap_b: Path, *, tol: float = 1e-3) -> dict:
    """两次采集记录的**天气必须一致** —— A/B「只变一个变量」的**正面**判据。

    ★★ 2026-10-07 实测:`outputs/3dgs_ab2` 那对 capture 位姿逐位相同、天气却不同
    (一边下雨一边晴),而当时的 capture **一个字节都没记天气** ⇒ 只能靠差分图肉眼发现。
    现在 `collect_3dgs` 落 `weather.json`(读回值),这条判据才有东西可比。

    ⚠️ **两代产物**:
      - 新 capture 有 `weather.json` ⇒ 比 `effective`(键集合相同,逐键比);
      - 老 capture 没有 ⇒ **未判**(返回 `known=False`),既不是通过也不是失败
        —— 同 `static_eval` 的「`n=0` 是不判」。
    """
    out: dict = {"known": False}
    pa, pb = cap_a / "weather.json", cap_b / "weather.json"
    if not pa.is_file() or not pb.is_file():
        out["reason"] = "至少一侧没有 weather.json(2026-10-07 之前的 capture)⇒ 本判据**未判**"
        return out
    wa = json.loads(pa.read_text(encoding="utf-8"))["effective"]
    wb = json.loads(pb.read_text(encoding="utf-8"))["effective"]
    diff = {k: (float(wa[k]), float(wb[k])) for k in wa if k in wb and abs(wa[k] - wb[k]) > tol}
    out.update(
        {
            "known": True,
            "scene_a": json.loads(pa.read_text(encoding="utf-8")).get("scene"),
            "scene_b": json.loads(pb.read_text(encoding="utf-8")).get("scene"),
            "differing": diff,
            "ok": not diff,
        }
    )
    return out


def pair_outside_diff(
    cap_a: Path,
    frames: list,
    poses: list,
    prop_id: int,
    imgs_a: torch.Tensor,
    imgs_b: torch.Tensor,
    h: int,
    w: int,
    *,
    dilate: int = PAIR_EXCLUDE_DILATE_PX,
) -> dict:
    """A/B 在**道具之外**差多少 —— 这对 capture 是不是"只变了一个变量"?

    ## ★★ 为什么必须有这条(2026-10-07 实测)

    `pose_pairing` 只看位姿 JSON,而 `outputs/3dgs_ab2` 那对**位姿逐位相同、天气却不同**
    (同一帧整帧均值 A=58.2 vs B=128.6;肉眼一望即知一边下雨一边晴)⇒
    **只看位姿的门槛对它完全瞎**,三档 PSNR 里混进了整整一个天气。
    ⚠️ 而这是**静默**的:位姿 JSON 逐位相同、文件齐全、数照样出得来。

    ## ★ 统计量:道具外 `|A−B|`(取通道最大)的 **p90**,再对帧取 max

    为什么**不是**"差了的像素占比":合法的一对里也可能有**局部**差异 ——
    实测干净对 `3dgs_ab` 的帧 90 有 **3132 px**(2.7%)差得明显(某个局部亮斑/软边),
    占比直接被它顶过阈值,而那不是"天气变了"。p90 只在**大片像素一起变**时才动,
    那正是"天气/曝光变了"的形态。

    ⚠️ **外扩必须比 `MASK_DILATE_PX` 大得多**(这里 16 px):道具的阴影/软边也**合法地**
    改变画面,拿紧掩膜去排除它们会把每一对合法 capture 都算成"差了很多"。
    """
    prop_masks = build_mask_inst(cap_a, frames, poses, prop_id, h, w, dilate)
    rows = []
    for j, fr in enumerate(frames):
        d = (imgs_a[j] - imgs_b[j]).abs().max(dim=-1)[0]
        outside = ~prop_masks[j]
        o = d[outside].float()
        rows.append(
            {
                "frame": int(fr),
                "n_outside": int(outside.sum()),
                "prop_px": int(prop_masks[j].sum()),
                # p90 = 「至少 10% 的像素差得比它多」;全局曝光/天气差会把它顶上去
                "p90": float(torch.quantile(o, 0.90)) if o.numel() else None,
                "mean": float(o.mean()) if o.numel() else None,
            }
        )
    judged = [r["p90"] for r in rows if r["p90"] is not None]
    worst = max(judged) if judged else None
    return {
        "per_frame": rows,
        "max_p90": worst,
        "ok": (worst is not None) and worst <= PAIR_MAX_P90_DIFF,
        "exclude_dilate_px": dilate,
        "limit": PAIR_MAX_P90_DIFF,
    }


def three_tier(
    rend_a: torch.Tensor,
    rend_edited: torch.Tensor,
    rend_b: torch.Tensor,
    img_b: torch.Tensor,
    masks: list[torch.Tensor],
) -> dict:
    """**纯函数**:三档 PSNR,全局与掩膜内各算一份。

    ★ `img_b` 必须**逐帧对齐**到 `rend_*` —— 第 `i` 张真值图就是第 `i` 张渲染的**那个视角**。
    调用方手里若是**全量**真值图,必须先按 `idx` 取(`imgs_b[idx]`)。

    ⚠️ **这一条是 2026-10-07 实测踩出来的**:`main()` 曾把**全量** `imgs_b`(270 帧)直接传进来,
    而这里按**循环位置** `img_b[i]` 取 ⇒ 拿「帧 45 的渲染」比「采集 B 的**第 0 帧**」——
    整张三档表全错。⚠️ 症状**完全静默**:数照样出得来、格式也对,而且只评一帧时看不出来
    (默认 5 帧时才露)。唯一的抓手是**归档锚**:`b_all` 逐帧必须等于
    `train_result_<tag>_B.json` 的 `psnr_val`(同一个模型、同一批帧、同一批图,只有口径不同)。
    """
    n = rend_a.shape[0]
    # 形状守卫:三档渲染 + 真值图必须**同长**。不等 = 调用方没对齐 ⇒ **当场停**,
    # 不许"能算就往下算" —— 那正是这个 bug 藏了这么久的原因。
    for name, t in (("rend_edited", rend_edited), ("rend_b", rend_b), ("img_b", img_b)):
        if t.shape[0] != n:
            raise SystemExit(
                f"three_tier: {name} 的首维 {t.shape[0]} != rend_a 的 {n} —— 真值图必须"
                "**按 idx 取**(`imgs_b[idx]`),不许把全量传进来(比的是别的视角)"
            )
    if len(masks) != n:
        raise SystemExit(f"three_tier: masks 长 {len(masks)} != rend_a 的 {n}")
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
    agg["edited_vs_ceiling_db"] = float(agg["edited_mask_mean"]) - float(agg["b_mask_mean"])
    # ★★ **缺口必须是正的**才谈得上"补上多少":
    #   天花板 ≤ 不编辑时,`closed / gap` 的分母非正 ⇒ 比值**没有意义**(会让人以为"越改越差")。
    #   ⇒ 负缺口时报**未判(方向反)**并说清方向,只保留 `编辑后 − 不编辑` 这个**两个数之间**的差。
    #
    # ⚠️ 这条守卫的**触发样例**在 2026-10-07 被查出是**另一件事的症状**,不是它的本意:
    #   当时读到的「inst 紧掩膜 ⇒ 天花板 12.99 < 不编辑 13.46」是上面那条**帧索引 bug** 造的
    #   (渲染帧 45 比的是采集 B 的**第 0 帧**)。修好索引后那个样例**不复存在**。
    #   ⇒ **守卫保留**(它本身是对的 —— 紧掩膜上"B 侧重建本来就不好"是真会发生的),
    #     但**别再把那组数当它的证据**。
    agg["gap_closed_frac"] = (closed / gap) if gap > 1e-9 else None
    agg["verdict"] = "可判" if gap > 1e-9 else "可判(缺口方向反)"
    if gap <= 1e-9:
        agg["reason"] = (
            f"天花板({agg['b_mask_mean']:.2f}) ≤ 不编辑({agg['a_mask_mean']:.2f})"
            " ⇒ 缺口不是正的,**'补上百分之几'读不出来**;能读的是 `gap_closed_db`"
            "(编辑后 − 不编辑)。常见于紧掩膜:物体那几像素上,B 侧的重建本来就不好"
        )
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
        "`inst` = 实例图 == `instance_id`(**物件本体的紧掩膜**,2026-10-07 修好取帧路径后才真的"
        "跑得起来)。⚠️ 旧的 help 说'会得到 0 像素' —— **那句已被推翻**(见模块头注)",
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
        imgs_a = _load_images(cap_a, poses, h, w).to(dev)
        # ★★ 第二条硬门槛:**A/B 只许差"有没有道具"这一件事**(2026-10-07 补)。
        #   `pose_pairing` 只读位姿 JSON,对"两边天气不同"完全瞎 —— 而 `3dgs_ab2` 正是那样一对。
        weather = pair_weather(cap_a, cap_b)
        if not weather["known"]:
            print(f"[weather] ⚠️ {weather['reason']}")
        elif weather["ok"]:
            print(f"[weather] A/B 记录一致(scene={weather['scene_a']})")
        else:
            raise SystemExit(
                f"A/B 记录的天气**不一致**:{weather['differing']} ⇒ 两次采集变的**不止**"
                "'有没有道具',三档 PSNR 里混进了一整个天气。**别放宽这条** —— 重采一对"
            )
        outside = pair_outside_diff(cap_a, idx, poses, prop_id, imgs_a[idx], imgs_b[idx], h, w)
        for r in outside["per_frame"]:
            print(
                f"[A/B] 帧 {r['frame']:>3}:道具外 {r['n_outside']:>6} px(p90 差 "
                f"{(r['p90'] or 0) * 255:5.1f}/255,均值 {(r['mean'] or 0) * 255:5.1f})"
            )
        rl.highlight("pair_max_p90", outside["max_p90"])
        if not outside["ok"]:
            raise SystemExit(
                f"A/B 在道具之外有**全局**光度差(p90 最大 {(outside['max_p90'] or 0) * 255:.1f}/255 "
                f"> {PAIR_MAX_P90_DIFF * 255:.0f}/255)⇒ 两次采集变的**不止**'有没有道具'"
                "(最常见:中间被别的进程改了天气)。**别放宽这个阈值** —— 那不是同一个场景的两个版本"
            )
        if args.mask_source == "rgb-diff":
            masks = build_mask_rgb_diff(
                cap_a, cap_b, idx, pitch_of, imgs_a[idx], imgs_b[idx], h, w, args.mask_dilate
            )
        else:
            masks = build_mask_inst(cap_a, idx, poses, prop_id, h, w, args.mask_dilate)
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

        # ★ 必须 `imgs_b[idx]`:`ra/re_/rb` 只含 `idx` 那几帧,真值图也必须只取那几帧。
        #   传全量会让 `three_tier` 拿"帧 45 的渲染"比"采集 B 的第 0 帧"(2026-10-07 实测)。
        rep = three_tier(ra, re_, rb, imgs_b[idx], masks)
        rep["prop"] = prop
        rep["pairing"] = pair
        rep["pair_weather"] = weather
        rep["pair_outside"] = outside
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


if __name__ == "__main__":
    main()

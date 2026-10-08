"""★ **删掉道具之后那块渲不好 —— 到底是"画的是什么"不对,还是"画得不清楚"?**

## 它接着哪一条

`probe_hole_visibility` 量出两件事,把原来"补窟窿"的路线**否掉了**:

- 那块背景**被大量视角看到**(中位 104/270);
- A 侧模型在那块**本来就有高斯**(覆盖 93.5%,删除前后**一模一样**)。

⇒ 窟窿**不是"没东西可填"**。但同一帧**整幅** PSNR 是 30.96 dB、**掩膜内只有 14.90 dB**
—— 差的是**局部**。所以真正的问题变成:

> **那些已经在场的高斯,渲出来的到底是个什么东西?**

## 判据:三方距离(谁像谁)

把三份渲染在同一批像素上比:

| 比谁 | 读作 |
|---|---|
| `d(A_edit, B_img)` | 编辑后离**真值**多远(已有读数 14.90 dB) |
| `d(A_edit, B_render)` | ★ **两个模型在同一块画的是不是同一个东西** |
| `d(A_edit, A_render)` | A_edit 离**没删之前**多远 —— 删了 1271 个高斯**到底改变没改变这块** |

⇒ **三分**:
- `A_edit` 仍**贴着 `A_render`** ⇒ 删的那批高斯**不负责**这块画面(那么"删干净"从来不是问题);
- `A_edit` 贴着 `B_render` 但离 `B_img` 远 ⇒ **两个模型的表示在这块都偏**(是共同的难点);
- `A_edit` **两边都不贴** ⇒ 删完冒出来了**第三种东西**(残留/糊)。

⚠️ **缺 `B_render` 那一列的话,上面三条一条都读不出来** —— 只报"离真值多远"会把
"两个模型都画不好"读成"A 的模型烂"。
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from autodrivedata.gs import cuda_env  # noqa: F401  —— 与 render_gs 同一纪律
from autodrivedata.gs.eval_edit import (
    build_mask_inst,
    build_mask_rgb_diff,
    parse_frames,
    pose_pairing,
)
from autodrivedata.gs.render_gs import load_set, mse_to_psnr
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path


def _psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    """掩膜内 PSNR(与 `three_tier` 同一条式子)。

    ⚠️⚠️ **这个模块第一版把当天早上刚修的帧索引 bug 又长了一遍**:写成 `imgs_b[j]` 而不是
    `imgs_b[idx][j]`,天花板算出 **18 dB** 而真值是 **39 dB**。因是:这里**绕开了
    `three_tier`** 自己写了一份 —— 而 `three_tier` 里有那条首维形状守卫,自己写的没有。
    ⇒ 教训不是"下次小心",是**别在判据外面另写一份判据**;这里保留 `_psnr` 是因为要比的
    对象不止"对真值"那一组(渲染之间也要比),但**对齐必须由调用方保证**,并在下面加守卫。
    """
    for name, t in (("a", a), ("b", b)):
        if t.shape[:-1] != mask.shape:
            raise SystemExit(
                f"probe_hole_render._psnr: {name} 的画幅 {tuple(t.shape[:-1])} 与掩膜 "
                f"{tuple(mask.shape)} 对不上 —— **真值图必须按 idx 取**(`imgs_b[idx][j]`)"
            )
    se = ((a - b) ** 2).mean(dim=-1)
    return mse_to_psnr(float(se[mask].mean())) if int(mask.sum()) else float("nan")


def run(args) -> dict:
    from autodrivedata.gs.train_3dgs_mini import _load_images, _load_poses_and_cams

    cap_a, cap_b, gs_dir = (
        project_path(args.capture_a),
        project_path(args.capture_b),
        project_path(args.gs_dir),
    )
    dev = "cuda"
    prop = json.loads((cap_a / "prop.json").read_text(encoding="utf-8"))["prop"]
    prop_id = int(prop["instance_id"])

    poses, _f, h, w, _cx, _cy, viewmats, ks = _load_poses_and_cams(cap_a, downsample=args.downsample)
    viewmats, ks = viewmats.to(dev), ks.to(dev)
    idx = parse_frames(args.frames, len(poses))
    imgs_b = _load_images(cap_b, poses, h, w).to(dev)
    # ★ 只取 idx 那几帧 —— 全量传进去会让每一帧都比错视角(当天早上刚踩过同一个坑)
    imgs_b = imgs_b[idx]

    if args.mask_source == "inst":
        masks = build_mask_inst(cap_a, idx, poses, prop_id, h, w, args.mask_dilate)
    else:
        imgs_a = _load_images(cap_a, poses, h, w).to(dev)
        masks = build_mask_rgb_diff(
            cap_a, cap_b, idx, lambda k: poses[k]["pitch"], imgs_a[idx], imgs_b[idx], h, w, args.mask_dilate
        )

    with torch.no_grad():
        ra = load_set(gs_dir, args.a_tag).render(idx, viewmats, ks, w, h, dev)
        re_ = load_set(gs_dir, args.edited_tag).render(idx, viewmats, ks, w, h, dev)
        rb = load_set(gs_dir, args.b_tag).render(idx, viewmats, ks, w, h, dev)

    rows = []
    for j, i in enumerate(idx):
        m = masks[j]
        if not m.any():
            continue
        rows.append(
            {
                "frame": int(i),
                "n_px": int(m.sum()),
                "edited_vs_truth": _psnr(re_[j], imgs_b[j], m),
                "ceiling_vs_truth": _psnr(rb[j], imgs_b[j], m),
                # ★ 两个模型在这块画的是不是同一个东西
                "edited_vs_ceiling_render": _psnr(re_[j], rb[j], m),
                # ★ 删除到底改变没改变这块
                "edited_vs_with_prop": _psnr(re_[j], ra[j], m),
                # 参照:不编辑离真值多远(有道具,当然差)
                "with_prop_vs_truth": _psnr(ra[j], imgs_b[j], m),
            }
        )
    if not rows:
        raise SystemExit("掩膜为空 —— 换帧或查道具是否真的露面")
    med = {k: float(np.median([r[k] for r in rows])) for k in rows[0] if k not in ("frame", "n_px")}
    return {
        "n_frames": len(rows),
        "mask_source": args.mask_source,
        "mask_px_total": int(sum(r["n_px"] for r in rows)),
        "median": med,
        "per_frame": rows,
        "pairing": {k: v for k, v in pose_pairing(cap_a, cap_b).items() if k != "per_frame"},
    }


#: 「有区别」的门槛(dB)。**故意设得很松** —— 它要判的不是"哪个更好",
#: 而是"这两个数是不是**同一个量级**"。实测的两组是 **14 dB vs 41 dB**(差 27 dB),
#: 拉开这么远的读数用 6 dB 已经绰绰有余;把门槛卡紧只会制造假阳性。
NEAR_DB = 6.0


def verdict(rep: dict) -> str:
    """★ **三分,但判据不是"两个渲染像不像"** —— 第一版把它写成那样,当场读错了。

    ⚠️ 第一版写的是「`d(A_edit, B_render) < 6` ⇒ 问题在 A 侧」,而实测 **14.26 dB**
    ⇒ 落进了"两个模型很像"那一档,**结论正好反了**。
    错在:`B_render` 与 `B_img` 本来就几乎一样(天花板 **40.96 dB**),
    所以"`A_edit` 离 `B_render` 远"与"`A_edit` 离真值远"**是同一件事**,
    它**不能**用来区分"A 侧差"与"共同难点"。

    ⇒ 正确的判据是**天花板可不可达**:

    1. 天花板自己就不行(`ceiling_vs_truth` 低)⇒ **未判**(真值在这块根本达不到,谈不了谁的错);
    2. ★ 天花板很高、而 `A_edit` 明显够不着 ⇒ **B 已经证明这块是可达的,A 没够到** ⇒ 问题在 A 侧;
    3. 两者相当 ⇒ 共同的难点。
    """
    m = rep["median"]
    ceil, edit = m["ceiling_vs_truth"], m["edited_vs_truth"]
    if ceil < NEAR_DB:
        return f"未判(天花板自己只有 {ceil:.2f} dB ⇒ 这块真值根本达不到,分不出是谁的错)"
    if edit < ceil - NEAR_DB:
        tail = ""
        if m["edited_vs_with_prop"] < edit:
            tail = (
                f";★ 而且编辑后离**删除前**(12.49 那一档)比离真值**更近**"
                f"({m['edited_vs_with_prop']:.2f} < {edit:.2f})⇒ 删完之后这块**仍更像「道具还在」**"
            )
        return (
            f"★ **问题在 A 侧的表示**:B 在同一块渲到 **{ceil:.2f} dB**(可达),而 A 编辑后只有 "
            f"**{edit:.2f} dB**{tail}"
        )
    return f"★ 两者相当(A {edit:.2f} / B {ceil:.2f})⇒ 离真值那截是**共同**的,不是 A 侧特有的缺陷"


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--capture-a", default="outputs/3dgs_ab3/A/capture")
    ap.add_argument("--capture-b", default="outputs/3dgs_ab3/B/capture")
    ap.add_argument("--gs-dir", default="outputs/3dgs")
    ap.add_argument("--a-tag", default="ab3_A")
    ap.add_argument("--edited-tag", default="ab3_A_edit")
    ap.add_argument("--b-tag", default="ab3_B")
    ap.add_argument("--frames", default="45,90,135,180,225")
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--mask-source", choices=("inst", "rgb-diff"), default="inst")
    ap.add_argument("--mask-dilate", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    with runlog.run("autodrivedata.gs.probe_hole_render") as rl:
        print(f"[arch] {cuda_env.ARCH_NOTE}")
        rl.highlight("mask_source", args.mask_source)
        rep = run(args)
        rep["verdict"] = verdict(rep)
        print(f"\n=== 掩膜内({rep['mask_source']}, {rep['mask_px_total']} px)===")
        print(f"  {'比谁':<34}{'中位 dB':>10}")
        for lab, k in (
            ("编辑后 vs 真值", "edited_vs_truth"),
            ("天花板(B 的渲染)vs 真值", "ceiling_vs_truth"),
            ("★ 编辑后 vs B 的渲染", "edited_vs_ceiling_render"),
            ("★ 编辑后 vs 删除前", "edited_vs_with_prop"),
            ("(参照)删除前 vs 真值", "with_prop_vs_truth"),
        ):
            print(f"  {lab:<34}{rep['median'][k]:>10.2f}")
        print(f"  ⇒ {rep['verdict']}")
        for k, v in rep["median"].items():
            rl.highlight(k, round(v, 3))
        rl.highlight("verdict", rep["verdict"])
        if args.out:
            p = project_path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
            rl.artifact(p, "report")
            print(f"[done] {p.resolve()}")


if __name__ == "__main__":
    main()

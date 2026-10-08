"""把 §1.12 的**真值靶**批量做成训练集:`(composite, truth, mask)` 三元组。

## 它接的是哪一格

§1.12 立起了靶(`harmonize_target`)—— **同一帧跨天气粘贴,真值就是原图** ——
但那时只用来**打分**(把 Reinhard 基线的残差量出来:3.31 → 1.54)。
「**训一条网络**」这一步需要的是**成批的样本**,不是一次评测,所以单开这个模块。

## 切 patch,不整幅

物体在一张 1242×375 的帧里只占几个百分点 ⇒ 整幅训练时**99% 的梯度来自背景**,
而那部分 composite 与 truth **逐位相同**(掩膜外没被动过)。⇒ 按框切 `PATCH` 见方的块。

⚠️ **patch 必须同时含"里面"(要改的)与"外面"(参考)** —— 和谐化的全部信息在
"这块补丁该怎么跟周围对上"。切得贴着框边,网络就只剩"均值回归"一条路。

## 判据(不许缺的那条)

本模块**只产数据,不做裁决**。裁决在 `harmonize_target.verdict` 与
`harmonize.run_harmonize` —— 两条都是"**下降** 且 **与对照臂分得开**"。
⚠️ 训练集与评测集**必须按天气对分开**:同一对既有训练帧又有评测帧 ⇒ 网络可以"背下这次光照差",
而那个差是**全局**的,背下来就会让读数虚高。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from autodrivedata.edit.harmonize_target import boxes_to_mask, load_boxes, paste
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: patch 边长(像素)。128 够装下一台 60 m 处的车(实测 ~23×16 px)**加上**它周围一圈背景。
PATCH = 128

#: ★★ **掩膜在这个 patch 里最多占多少**。超了就**丢掉这个样本**。
#: 理由不是"省显存":和谐化的**全部信息**在"这块补丁该怎么跟周围对上" ——
#: 掩膜铺满 patch 时**根本没有"周围"**,Reinhard 那种"拿外圈当参考"的基线连定义都没有
#: (实测当场抛 `掩膜内只有 0 个像素`)。近处的车就会这样(18 m 处的框 146×80 > 128)。
MAX_MASK_SHARE = 0.5


def patch_slices(cx: float, cy: float, w: int, h: int, patch: int) -> tuple[slice, slice] | None:
    """以 `(cx, cy)` 为中心切 `patch` 见方,**越界就夹回画幅**(不 padding)。

    返回 `None` 表示连夹回之后都放不下(patch 比画幅还大,或中心在画外很远)。
    """
    x0 = min(max(int(cx) - patch // 2, 0), max(w - patch, 0))
    y0 = min(max(int(cy) - patch // 2, 0), max(h - patch, 0))
    if x0 + patch > w or y0 + patch > h:
        return None
    return slice(y0, y0 + patch), slice(x0, x0 + patch)


def iter_patches(
    ctx_root: Path,
    src_root: Path,
    frames: list[str],
    *,
    cls: str = "Car",
    patch: int = PATCH,
):
    """逐帧、逐框产出 `{frame, box_i, composite, truth, mask}`(都是 `patch²` 的块)。

    `skipped` 记在本函数的属性上(生成器不能 return)—— 调用方读完取。
    """
    iter_patches.skipped = 0
    skipped = 0
    for fid in frames:
        boxes = load_boxes(ctx_root, fid, cls)
        if not boxes:
            continue
        ctx = np.array(Image.open(ctx_root / "training/image_2" / f"{fid}.png").convert("RGB"))
        src = np.array(Image.open(src_root / "training/image_2" / f"{fid}.png").convert("RGB"))
        if ctx.shape != src.shape:
            raise SystemExit(f"两张图不同画幅:{ctx.shape} vs {src.shape}")
        comp = paste(ctx, src, boxes)
        m = boxes_to_mask(ctx.shape[:2], boxes)
        h, w = ctx.shape[:2]
        for bi, (x1, y1, x2, y2) in enumerate(boxes):
            sl = patch_slices((x1 + x2) / 2, (y1 + y2) / 2, w, h, patch)
            if sl is None:
                continue
            ys, xs = sl
            pm = m[ys, xs]
            if not pm.any():
                continue  # 框完全在 patch 之外(不该发生,但静默产出空掩膜样本最坏)
            if pm.mean() > MAX_MASK_SHARE:
                # ★ 掩膜铺满 ⇒ **没有"周围"** ⇒ 这个样本对"和谐化"这个问题是空的。丢掉并计数。
                skipped += 1
                # ⚠️ **立刻发布,不能只在 `yield` 前发** —— 全都丢掉时生成器一个 `yield` 都没有,
                #    属性就永远停在 0,而"静默不计数"正是这条要防的。
                iter_patches.skipped = skipped
                continue
            iter_patches.skipped = skipped
            yield {
                "frame": fid,
                "box_i": bi,
                "composite": comp[ys, xs],
                "truth": ctx[ys, xs],
                "mask": pm,
            }


def build_dataset(
    pairs: list[tuple[Path, Path]],
    frames: list[str],
    out: Path,
    *,
    cls: str = "Car",
    patch: int = PATCH,
    limit: int | None = None,
) -> dict:
    """逐对跑,落 `{out}/{ctx名}__{src名}/{frame}_{i:02d}.npz`。**返回统计。**"""
    out.mkdir(parents=True, exist_ok=True)
    n, per_pair = 0, {}
    for ctx_root, src_root in pairs:
        tag = f"{ctx_root.name}__{src_root.name}"
        d = out / tag
        d.mkdir(parents=True, exist_ok=True)
        k = 0
        for p in iter_patches(ctx_root, src_root, frames, cls=cls, patch=patch):
            if limit is not None and n >= limit:
                break
            np.savez_compressed(
                d / f"{p['frame']}_{p['box_i']:02d}.npz",
                composite=p["composite"],
                truth=p["truth"],
                mask=p["mask"],
            )
            k += 1
            n += 1
        per_pair[tag] = k
        print(f"  [{tag}] {k} 个 patch(丢掉掩膜铺满的 {iter_patches.skipped} 个)")
    return {"n_patches": n, "per_pair": per_pair, "patch": patch, "n_frames": len(frames)}


def _parse_pairs(spec: str, root: Path) -> list[tuple[Path, Path]]:
    """`A:B,C:D` → `[(A,B),(C,D)]`。名字按 `root` 解析;`all` = 目录下全部两两有序对。"""
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise SystemExit(f"--pairs 每一段要 `ctx:src`,收到 {chunk!r}")
        a, b = chunk.split(":", 1)
        out.append((root / a, root / b))
    if not out:
        raise SystemExit("--pairs 解析出空列表")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root", default="outputs", help="名字的解析根(默认 outputs/)")
    ap.add_argument(
        "--pairs",
        required=True,
        help="`ctxA:srcA,ctxB:srcB`(**有序** —— 贴谁进谁是自变量)。如 "
        "`kitti_ab_epic_clip2d_day_clear:kitti_ab_epic_clip2d_sunset_glare`",
    )
    ap.add_argument("--frames", default="0-49", help="逗号或 a-b")
    ap.add_argument("--cls", default="Car")
    ap.add_argument("--patch", type=int, default=PATCH)
    ap.add_argument("--limit", type=int, default=None, help="最多产多少个 patch(调试用)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    root, out = project_path(args.root), project_path(args.out)
    pairs = _parse_pairs(args.pairs, root)
    frames = _parse_frames(args.frames)
    with runlog.run("autodrivedata.edit.harmonize_data") as rl:
        for a, b in pairs:
            rl.input(a, "ctx")
            rl.input(b, "src")
        rl.highlight("n_pairs", len(pairs))
        rl.highlight("patch", args.patch)
        rl.highlight("n_frames", len(frames))
        rep = build_dataset(pairs, frames, out, cls=args.cls, patch=args.patch, limit=args.limit)
        print(f"\n⇒ 共 {rep['n_patches']} 个 patch({len(pairs)} 对 × {len(frames)} 帧)")
        (out / "dataset.json").write_text(
            json.dumps({**rep, "pairs": [[str(a), str(b)] for a, b in pairs]}, indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
        rl.highlight("n_patches", rep["n_patches"])
        rl.artifact(out, "dataset")


def _parse_frames(spec: str) -> list[str]:
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(f"{i:06d}" for i in range(int(a), int(b) + 1))
        elif part:
            out.append(f"{int(part):06d}")
    if not out:
        raise SystemExit(f"--frames 解析出空列表:{spec!r}")
    return out


if __name__ == "__main__":
    main()

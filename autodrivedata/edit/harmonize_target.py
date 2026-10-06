"""**和谐化的真值靶** —— 用 A/B 采集造"贴错光照的补丁",**真值就是原图本身**。

## 它补的是哪一格

§1.9 的 `harmonize` 明写着"**这一格没有真值靶**",判据只能是代理(与真值帧的统计距离)。
`docs/edit-image-plan.md` §3 阶段 G 留下的触发条件正是这一条:要**找到真值靶**。

本模块给出的靶**是精确的**,而且只用本项目已有的东西:

| 角色 | 是什么 |
|---|---|
| **背景 / 真值** | A 天气(`day_clear`)的第 f 帧 |
| **补丁来源** | B 天气(`sunset_glare`)的**同一帧** |
| **合成图 X** | A 的帧,把**车辆框区域**换成 B 的同位置像素 ⇒ 一块**光照对不上的补丁** |
| **真值 Y** | A 的帧**原样** |
| **掩膜 M** | A 那一帧的 GT 车辆框(并集) |

⇒ 判据 = `d(和谐化(X), Y)` 在 M 内**必须下降**,且与对照臂分得开。

## ★ 为什么粘得上去:几何由 A/B 硬门槛保证,而且**可以验**

A/B 采集的硬门槛是"两侧 `training/pose/` 逐帧位置差 ≤ ~0.07 m"。在图像上的直接后果是
**同一物体的框坐标几乎相等** —— 实测(`day_clear` vs `sunset_glare`,70 帧):
**中位 0.02 px、max 0.31 px**。`box_shift` 就是这条判据,默认上限 2 px,超了就抛。

⚠️ **它只保证"贴得对",不保证"贴得值"** —— 见下面"边界"。

## 边界(必须随读数一起报)

1. **掩膜是 GT 2D 框,不是紧致轮廓** —— 框里混着车周围的背景,那部分**也被换了**。
   任务因此是"把这一块补丁融进去",而不是"只改车"。**框的像素在掩膜里是已知的**,
   所以判据内部是一致的;
2. **只在掩膜内比** —— 掩膜外的像素两边逐位相同,算进去只会稀释信号;
3. **方法只做统计对齐**(`harmonize.reinhard_map`),**不碰结构** ——
   它能修"光照/色温对不上",修不了"材质对不上"。剩下的残差**就是**那个材质差。

## 对照臂(缺一个都读不出结论)

| 臂 | 参考统计从哪来 | 期望 |
|---|---|---|
| **identity** | 不动 | 基线 `d_before` |
| **方法** | **本图上下文**(粘贴区外圈) | 应当明显下降 |
| **对照·错光源** | **B 天气那张图**的外圈 | 应当不降(甚至更差) |
| **对照·整图全局** | 整张合成图 | 几乎不动(补丁只占几个百分点) |
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from autodrivedata.edit.harmonize import (
    context_ring,
    reinhard_transfer,
    reinhard_transfer_masked,
    stat_distance,
)
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: **对齐判据的上限**(px)。实测 `day_clear`↔`sunset_glare` 是 max 0.31 px,2 px 已经很松。
MAX_BOX_SHIFT_PX = 2.0


def load_boxes(root: Path, fid: str, cls: str = "Car") -> list[tuple[float, float, float, float]]:
    """`label_2` 里某一类的 2D 框(列 5–8)。**按 y1 排序** —— 跨 root 比坐标时必须先定序。"""
    p = root / "training/label_2" / f"{fid}.txt"
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) >= 15 and f[0] == cls:
            out.append(tuple(float(v) for v in f[4:8]))
    return sorted(out, key=lambda b: (b[1], b[0]))


def box_shift(a: list, b: list) -> float:
    """两组框**按坐标定序后**的最大坐标差;条数不等直接抛(那是"配不上",不是"差得多")。

    ⚠️ **排序必须留在这里,不能只留在 `load_boxes` 里** —— 行序在两家 root 之间是不同的
    (CARLA `get_actors()` 跨采集不稳定),逐行 zip 比会得到"差 **215 px**"的假读数
    (2026-10-06 实测踩到,见 `downstream_eval.run_arms` 的头注)。判据自己定序,
    才不依赖调用方记得。
    """
    if len(a) != len(b):
        raise ValueError(f"框条数不等:{len(a)} vs {len(b)}")
    if not a:
        return 0.0
    ka = sorted(a, key=lambda t: (t[1], t[0], t[2], t[3]))
    kb = sorted(b, key=lambda t: (t[1], t[0], t[2], t[3]))
    return float(max(max(abs(x - y) for x, y in zip(p, q, strict=True)) for p, q in zip(ka, kb, strict=True)))


def boxes_to_mask(shape: tuple[int, int], boxes: list, *, dilate_px: int = 0) -> np.ndarray:
    """框 → 二值掩膜。**越界钳到画幅**(框可以伸出去,掩膜不行)。"""
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        xa, xb = max(0, int(np.floor(x1)) - dilate_px), min(w, int(np.ceil(x2)) + dilate_px)
        ya, yb = max(0, int(np.floor(y1)) - dilate_px), min(h, int(np.ceil(y2)) + dilate_px)
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = True
    return m


def paste(ctx: np.ndarray, src: np.ndarray, boxes: list) -> np.ndarray:
    """把 `src` 在 `boxes` 区域内的像素贴到 `ctx` 上 —— **其它像素逐位不变**。"""
    if ctx.shape != src.shape:
        raise ValueError(f"两张图不同画幅:{ctx.shape} vs {src.shape}")
    out = ctx.copy()
    m = boxes_to_mask(ctx.shape[:2], boxes)
    out[m] = src[m]
    return out


def harmonize_pair(ctx: np.ndarray, src: np.ndarray, boxes: list, *, ring: int = 8) -> dict | None:
    """一帧的四臂读数。掩膜为空(该帧没车)⇒ 返回 `None`(由调用方计数)。"""
    mask = boxes_to_mask(ctx.shape[:2], boxes)
    if not mask.any():
        return None
    comp = paste(ctx, src, boxes)
    ring_mask = context_ring(mask, ring)
    if ring_mask.sum() < 32:  # 外圈太小 ⇒ 统计量不稳,别硬算
        ring_mask = ~mask
    fixed = reinhard_transfer_masked(comp, mask, comp, ring_mask)
    ctrl_src = reinhard_transfer_masked(comp, mask, src, ring_mask)
    ctrl_glob = reinhard_transfer(comp, comp)
    return {
        "d_before": stat_distance(comp, ctx, mask),
        "d_method": stat_distance(fixed, ctx, mask),
        "d_ctrl_wrong_source": stat_distance(ctrl_src, ctx, mask),
        "d_ctrl_global": stat_distance(ctrl_glob, ctx, mask),
        "mask_px": int(mask.sum()),
    }


def run_frames(ctx_root: Path, src_root: Path, frames: list[str], *, cls: str = "Car", ring: int = 8) -> dict:
    """逐帧跑;返回**逐帧明细 + 中位数**。对齐不合格**当场抛**(见 `MAX_BOX_SHIFT_PX`)。"""
    rows, skipped, shifts = [], [], []
    for fid in frames:
        a = load_boxes(ctx_root, fid, cls)
        b = load_boxes(src_root, fid, cls)
        shifts.append(box_shift(a, b))
        if not a:
            skipped.append(fid)
            continue
        ctx = np.array(Image.open(ctx_root / "training/image_2" / f"{fid}.png").convert("RGB"))
        src = np.array(Image.open(src_root / "training/image_2" / f"{fid}.png").convert("RGB"))
        r = harmonize_pair(ctx, src, a, ring=ring)
        if r is None:
            skipped.append(fid)
            continue
        r["frame"] = fid
        rows.append(r)
    if not rows:
        raise SystemExit("没有任何一帧有可用的掩膜")
    worst = max(shifts) if shifts else 0.0
    if worst > MAX_BOX_SHIFT_PX:
        raise SystemExit(
            f"对齐判据不合格:两 root 的框坐标最大差 {worst:.2f} px > {MAX_BOX_SHIFT_PX} px。\n"
            "  粘上去的补丁在几何上就不成立 —— 先查 A/B 硬门槛(training/pose 逐帧差)。"
        )
    med = {k: float(np.median([r[k] for r in rows])) for k in rows[0] if k != "frame"}
    return {
        "n_frames": len(rows),
        "n_skipped": len(skipped),
        "skipped": skipped[:20],
        "max_box_shift_px": round(worst, 3),
        "cls": cls,
        "ring": ring,
        "median": med,
        "rows": rows,
    }


def verdict(rep: dict, *, tol: float = 0.02) -> str:
    """**两条都要过**:① 迁移后下降;② 与「对齐到错光源」的对照分得开。

    只过 ① 不算数 —— "把任何图都往中间拉"也会降。**对照臂是本判据的核心纪律。**
    """
    m = rep["median"]
    b, a, c = m["d_before"], m["d_method"], m["d_ctrl_wrong_source"]
    if a >= b - tol:
        return f"★ 方法对**已知的冲突**不敏感(合成 {b:.4f} → 修复 {a:.4f})⇒ 尺子立不起来"
    if a >= c - tol:
        return f"★ 下降低于对照臂(修复 {a:.4f} vs 错光源对照 {c:.4f})⇒ 分不清「用对了上下文」与「随便拉一把」"
    return (
        f"★★ 真值靶通过:掩膜内距离 {b:.4f} → {a:.4f}(降 {b / max(a, 1e-9):.1f}×),"
        f"错光源对照 {c:.4f} ⇒ **方法确实用对了上下文**,但残差就是**材质差**(统计对齐修不了)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--ctx", required=True, help="背景 / 真值天气的 root(如 day_clear)")
    ap.add_argument("--src", required=True, help="补丁来源天气的 root(如 sunset_glare)")
    ap.add_argument("--frames", default="0-39", help="逗号或 a-b;**两侧同帧号必须是同一批**")
    ap.add_argument("--cls", default="Car")
    ap.add_argument("--ring", type=int, default=8, help="上下文外圈的宽度(px)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    ctx, src, out = (project_path(p) for p in (args.ctx, args.src, args.out))
    out.mkdir(parents=True, exist_ok=True)
    frames = _parse_frames(args.frames)
    with runlog.run("autodrivedata.edit.harmonize_target") as rl:
        rl.input(ctx, "ctx-root")
        rl.input(src, "src-root")
        rl.highlight("cls", args.cls)
        rl.highlight("ring", args.ring)
        rl.highlight("n_frames", len(frames))
        rep = run_frames(ctx, src, frames, cls=args.cls, ring=args.ring)
        v = verdict(rep)
        for k, x in rep["median"].items():
            rl.highlight(f"median_{k}", round(float(x), 4))
        rl.highlight("max_box_shift_px", rep["max_box_shift_px"])
        rl.highlight("n_skipped", rep["n_skipped"])
        rl.highlight("verdict", v)
        print(f"\n  掩膜内 LAB 统计距离(中位,{rep['n_frames']} 帧,跳过 {rep['n_skipped']} 帧)")
        print(f"    合成图 vs 真值(未处理) = {rep['median']['d_before']:.4f}")
        print(f"    方法(用本图上下文)     = {rep['median']['d_method']:.4f}")
        print(f"    对照·错光源            = {rep['median']['d_ctrl_wrong_source']:.4f}")
        print(f"    对照·整图全局          = {rep['median']['d_ctrl_global']:.4f}")
        print(f"  对齐:两 root 框坐标最大差 {rep['max_box_shift_px']} px(上限 {MAX_BOX_SHIFT_PX})")
        print(f"  ⇒ {v}")
        (out / "harmonize_target.json").write_text(
            json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        rl.artifact(out / "harmonize_target.json", "summary")


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

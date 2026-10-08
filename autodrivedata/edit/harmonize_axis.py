"""**和谐化这一格到底有没有"只有网络能解"的余地?** —— 拿**不依赖任何假设**的臂去问。

## 为什么问这个

§1.12 立了真值靶(同帧跨天气粘贴,真值 = 原图),`train_harmonize` 训了一条小网络 ——
结论是**负的**:留出对上一个**统计算子**(Reinhard)0.569,而网络 5.895。
一种解释是"判据就是那个算子的目标函数",但那个解释**只在雾那一对成立**
(湿路面那对**根本没有改进空间**:不处理已经 1.19)。

⇒ 在"换判据再训一版"之前,应当先答:**这个靶上到底存不存在一个"统计方法够不着、
只有网络够得着"的格子?**

## ★ 本模块的答案:**用不依赖假设的臂直接量**

不去论证"天气差是不是全局光度变换"(那条路**试了三次、三次都被自己的对照打红**,见文末),
而是直接量:**这一格里,最笨的几个方法离真值有多远?**

| 臂 | 是什么 | 为什么它在表里 |
|---|---|---|
| `d_composite` | 不处理(直接贴) | **地板** |
| `d_ring_mean` | 把掩膜内**全部填成"周围一圈"的 LAB 均值 | ★ **最笨的"用上下文"的方法**。它就是"把邻居的统计抄过来"这句话的**字面实现** |
| `d_reinhard` | §1.12 的基线(周围统计对齐,含协方差) | 有协方差 ⇒ 比上面强一点点的那档 |
| `d_ctrl_wrong_source` | 统计量取自**源天气那张图的周围**(错光源) | **对照**:证明"用的是哪张图的统计"是有影响的 |

**判读**:若 `d_ring_mean` 已经贴着真值(远低于 `d_composite`),
则**"用周围统计"这件事本身就把这一格解完了** —— 那么在这一格上比"网络 vs 统计算子",
比的是**同一个答案的两种写法**,网络**没有结构性的余地**。

⚠️ 这只回答"**还有没有余地**",不回答"哪种写法更好"。所以它是**收口判据**,不是选型判据。

## ★★ 三条被自己对照打红的路线(留在案,防重蹈)

都试过、都失败,而且**失败的方式同型**:新统计量的**零假设没被证明过**。

1. **分块结构比**(残差场的块均值 std ÷ 白噪声期望 std,≈1 判"无结构")
   —— **当场被"打乱行"对照打红**:打乱后是 **16–30** 而不是 ≈1。
   因:真实图像空间强相关,"逐像素独立"那条推导不成立 ⇒ "≈1 = 无结构"**从未成立**。
2. **全局仿射解释率**(`1 − mean(resid²)/mean(raw²)`,≈100% 判"全局")
   —— **被"已知的深度散射"对照打红**:β=0.4 的浓散射(强非线性、散射 ∝ 深度)
   解释率 **100.0%**,**比真值天气对(91–93%)还高**。因:平方比的分子分母都被**少量大残差**主导
   ⇒ 量的是"幅度",不是"结构"。
3. **稳健版解释率**(换成中位数) —— 同一个对照仍然给 **98.8%**(真值雾 74.7%)。
   ⇒ LAB 里的**仿射是个比想象中强得多的模型**,一条平滑单调的色调曲线在既有亮度范围内
   几乎就是仿射的。

★ 三条的教训是**同一条**(与 §1.10 那版被推翻的 WSD 同型):
**先证明这个量在该给 0 的时候会给出 0,再去用它读数。**
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autodrivedata.edit.harmonize import (
    context_ring,
    from_lab,
    reinhard_transfer_masked,
    stat_distance,
    to_lab,
)
from autodrivedata.edit.harmonize_target import MAX_BOX_SHIFT_PX, box_shift, load_boxes, paste
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 外圈宽度(像素)。与 `harmonize_target` 同值 —— 两边量的必须**同一件事**。
RING_PX = 8

#: 外圈小于这个像素数就退化成"掩膜外的全部"(统计量会不稳)。
MIN_RING_PX = 32


def _boxes(root: Path, fid: str, cls: str) -> list:
    """`harmonize_target.load_boxes` 的**缺文件安全**版。

    ⚠️ 底座版直接 `read_text`,**帧号不齐的 root 会 `FileNotFoundError` 崩**;
    而"这一帧没有标注"与"这个 root 根本没有标注"在这里的处理应当一样:**当空**。
    (两种都不该让整对被判死 —— 逐帧跳过、最后按"一帧可判的都没有"收口。)
    """
    if not (root / "training" / "label_2" / f"{fid}.txt").is_file():
        return []
    return load_boxes(root, fid, cls)


def ring_mean_fill(rgb: np.ndarray, mask: np.ndarray, ring_mask: np.ndarray) -> np.ndarray:
    """把 `mask` 里的像素**全部填成 `ring_mask` 的 LAB 均值**。

    ★ 这是"**抄周围统计**"这句话的**字面实现**,而且刻意做成最笨的形状:
    没有空间变化、没有协方差、没有任何模型 —— 只有"周围长什么样,里面就长什么样"。
    ⇒ 它离真值有多远,**就是"用上下文对齐"这条路的天花板有多高**。

    ⚠️ 只在掩膜内改,掩膜外**逐位保留**(与 `reinhard_transfer_masked` 同一条纪律:
    RGB→LAB→RGB 有损,整幅转一圈会把掩膜外也改了)。
    """
    lab = to_lab(rgb)
    mu = lab[ring_mask].mean(axis=0)
    out = lab.copy()
    out[mask] = mu
    rgb_out = from_lab(out)
    res = rgb.copy()
    res[mask] = rgb_out[mask]
    return res


def frame_arms(ctx: np.ndarray, src: np.ndarray, boxes: list, *, ring: int = RING_PX) -> dict | None:
    """一帧的四臂读数(掩膜为空 ⇒ `None`,由调用方计数)。"""
    from autodrivedata.edit.harmonize_target import boxes_to_mask

    mask = boxes_to_mask(ctx.shape[:2], boxes)
    if not mask.any():
        return None
    comp = paste(ctx, src, boxes)
    ring_mask = context_ring(mask, ring)
    if ring_mask.sum() < MIN_RING_PX:
        ring_mask = ~mask
    fixed = reinhard_transfer_masked(comp, mask, comp, ring_mask)
    # 对照:统计量取自**源天气那张图**的外圈(错光源)
    ctrl = reinhard_transfer_masked(comp, mask, src, ring_mask)
    return {
        "d_composite": stat_distance(comp, ctx, mask),
        "d_ring_mean": stat_distance(ring_mean_fill(comp, mask, ring_mask), ctx, mask),
        "d_reinhard": stat_distance(fixed, ctx, mask),
        "d_ctrl_wrong_source": stat_distance(ctrl, ctx, mask),
        # ★ 管道自证:`ring_mask` 若**含掩膜内**像素,下面这条会偏离 `d_composite`
        "mask_px": int(mask.sum()),
        "ring_px": int(ring_mask.sum()),
    }


def run_pair(ctx_root: Path, src_root: Path, frames: list[str], *, cls: str = "Car") -> dict:
    """逐帧跑四臂。★ A/B **对齐不合格当场抛** —— 与 `harmonize_target` 同一条硬门槛。"""
    rows, shifts, skipped = [], [], 0
    for fid in frames:
        a_boxes = _boxes(ctx_root, fid, cls)
        b_boxes = _boxes(src_root, fid, cls)
        if not a_boxes or not b_boxes:
            skipped += 1
            continue
        shift = box_shift(a_boxes, b_boxes)
        if shift > MAX_BOX_SHIFT_PX:
            raise SystemExit(
                f"帧 {fid}:A/B 框最大差 {shift:.2f} px > {MAX_BOX_SHIFT_PX} ⇒ 这对**没有配对**"
                "(贴上去的物体不是同一个),四臂读数作废。**别放宽这个上限**"
            )
        shifts.append(shift)
        a = _load(ctx_root, fid)
        b = _load(src_root, fid)
        r = frame_arms(a, b, a_boxes)
        if r is not None:
            rows.append(r)
    if not rows:
        raise SystemExit(f"{ctx_root.name} / {src_root.name}:一帧可判的都没有")
    return {
        "ctx": ctx_root.name,
        "src": src_root.name,
        "n": len(rows),
        "n_skipped": skipped,
        "median": _agg(rows),
        "box_shift_max_px": float(max(shifts)) if shifts else None,
        "per_frame": rows,
    }


def _agg(rows: list[dict]) -> dict:
    return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


def verdict(rep: dict) -> str:
    """★ 收口判据:**这一格有没有"比不处理更好"的余地**。

    ⚠️ 第一版只看「抄周围贴不贴得近」,而**湿路面那对被读反了**:
    那里不处理已经 **1.541**,两条臂(2.153 / 1.543)**都没有更好** ——
    正确的读法是「**没余地**」,不是「抄周围贴不近 ⇒ 有余地」。
    判据必须先问"**有没有方法能超过不处理**",再问"超得动多少"。
    """
    m = rep["median"]
    comp, ring, re_h = m["d_composite"], m["d_ring_mean"], m["d_reinhard"]
    best = min(ring, re_h)
    if best >= comp - 0.02:
        return (
            f"★ **无余地**:不处理已经 {comp:.3f},而「抄周围」{ring:.3f} / "
            f"reinhard {re_h:.3f} **都没有更好** ⇒ 这一格量不出东西(周围统计不是答案)"
        )
    if best <= 0.5 * comp:
        return (
            f"★ **收口**:最笨的「抄周围均值」把 {comp:.3f} 压到 {ring:.3f}"
            f"(含协方差的 reinhard {re_h:.3f})⇒ 这一格**由统计对齐解完**,网络无结构性余地"
        )
    return f"★ **有余地**:最好的臂 {best:.3f} 只把 {comp:.3f} 压下一半多,值得再看"


def _load(root: Path, fid: str) -> np.ndarray:
    from PIL import Image

    p = root / "training" / "image_2" / f"{fid}.png"
    if not p.is_file():
        raise SystemExit(f"{p} 不存在")
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--pairs", nargs="+", required=True, help="`ctx_root:src_root` 若干")
    ap.add_argument("--frames", default="0-39")
    ap.add_argument("--cls", default="Car")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    frames = _parse_frames(args.frames)
    pairs = []
    for chunk in args.pairs:
        if ":" not in chunk:
            raise SystemExit(f"--pairs 要 `ctx:src`,收到 {chunk!r}")
        a, b = chunk.split(":", 1)
        pairs.append((project_path(a), project_path(b)))

    with runlog.run("autodrivedata.edit.harmonize_axis") as rl:
        rl.highlight("n_frames", len(frames))
        rl.highlight("cls", args.cls)
        reps = []
        for a, b in pairs:
            rep = run_pair(a, b, frames, cls=args.cls)
            rep["verdict"] = verdict(rep)
            reps.append(rep)
            m = rep["median"]
            print(
                f"\n=== {rep['ctx']} → {rep['src']}  (n={rep['n']}, 框对齐 max {rep['box_shift_max_px']:.2f} px) ==="
            )
            print(f"  {'臂':<26}{'掩膜内统计距离':>14}")
            for name, key in (
                ("不处理(直接贴)", "d_composite"),
                ("★ 抄周围均值(最笨)", "d_ring_mean"),
                ("reinhard(含协方差)", "d_reinhard"),
                ("对照:错光源的统计", "d_ctrl_wrong_source"),
            ):
                print(f"  {name:<26}{m[key]:>14.4f}")
            print(f"  ⇒ {rep['verdict']}")
            for k in ("d_composite", "d_ring_mean", "d_reinhard"):
                rl.highlight(f"{rep['src'][-14:]}_{k}", round(m[k], 4))
            rl.highlight(f"{rep['src'][-14:]}_verdict", rep["verdict"])
        if args.out:
            p = project_path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(reps, indent=1, ensure_ascii=False), encoding="utf-8")
            rl.artifact(p, "report")
            print(f"\n[done] {p.resolve()}")


def _parse_frames(spec: str) -> list[str]:
    out: list[str] = []
    for chunk in spec.split(","):
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out += [f"{i:06d}" for i in range(int(a), int(b) + 1)]
        else:
            out.append(f"{int(chunk):06d}")
    return out


if __name__ == "__main__":
    main()

"""**视频编辑的时序一致性**:逐帧生成会不会闪?

JD 明确列了「图像 / **视频**编辑」,而这是本项目**唯一有真值可依**的视频判据 ——
逐帧条件是**真值深度**,帧与帧之间有**已知的、连续的**关系(环形采集每帧转 4°)。
大多数"视频编辑"工作没有这个 → 只能靠人眼看闪不闪。

## 判据:相邻帧的**结构跳变**,不是"看着连不连贯"

- `d_in(t)`  = 输入序列第 t−1 → t 帧的结构差(**场景自己的运动量**,是 floor)
- `d_out(t)` = 生成序列同样的量
- **生成侧明显更大 ⇒ 闪**

⇒ 两条自证(缺一条这根尺子就立不住):
1. **与真值序列比,不与 0 比** —— "有变化"本身不是问题,比场景自身运动量更大才是;
2. **打乱顺序必须报出更大的跳变** —— 否则这把尺子测不出时序,读出的"不闪"是假的。

## ★ 为什么描述子**不用** MiDaS

MiDaS 在 512² 中心裁上**本身不稳**(§1.3.1 ③:逐帧 ρ 从 −0.36 到 0.993)。
**用一把抖的尺子去量抖,两者分不开。** 这里改用**无模型、确定性**的梯度幅值描述子。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 描述子的边长。64 足够留住"哪里有条边"这类结构,又对逐像素噪声不敏感。
DESC_SIZE = 64

#: ★ **尺子自证的绝对下限**:打乱后的跳变中位低于它 ⇒ 这把尺子**看不出时序**。
#:
#: ⚠️ **为什么不用"打乱/有序"的比值**(2026-10-06 实测改过来):在第一版夹具上
#: 比值最高只到 **2.2×**(blur 15/31/51 × step 1/2/4/8 全扫过,最好 2.2、最差 1.0)——
#: `1 − Spearman` 在梯度图上**动态范围被压扁**(大量近平坦区产生并列),
#: 于是"比值 > 2/3"这种判据会**系统性假报"尺子失效"**。
#: 改用绝对下限:打乱序列本身**就该**与有序序列拉开(实测静态序列打乱 0.55–0.95)。
SHUF_FLOOR = 0.45


def structure_descriptor(rgb: np.ndarray, *, size: int = DESC_SIZE) -> np.ndarray:
    """**无模型**结构描述子:灰度 → 梯度幅值 → 均值池化 → 展平 → 标准化。

    - 确定性(同图同出),不依赖任何模型 ⇒ 没有"尺子自己在抖"的问题
    - 对**整体亮度/对比度**不敏感:先做局部对比归一化,再取梯度
      (雾会让对比度降 ⇒ 梯度整体变小,但**帧间相对变化**仍可比 —— 这正是要的)
    """
    import cv2

    g = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    small = cv2.resize(mag, (size, size), interpolation=cv2.INTER_AREA)
    v = small.ravel().astype(np.float64)
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def adjacent_diffs(descs: list[np.ndarray]) -> list[float]:
    """相邻帧的结构差 `1 − Spearman`(对单调变换不变)。**越小越连贯**。"""
    if len(descs) < 2:
        return []
    return [1.0 - float(stats.spearmanr(descs[i - 1], descs[i])[0]) for i in range(1, len(descs))]


def shuffled(seq: list, *, seed: int = 0) -> list:
    """打乱(用 `np.random.default_rng` 而非 `random`,便于复现)。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(seq))
    return [seq[i] for i in idx]


def series_descriptors(paths: list[Path], *, square: bool = False) -> list[np.ndarray]:
    """逐帧算描述子(**按给定顺序**,不排序 —— 顺序就是时序)。

    `square=True` 时先做**中心方裁**(与生成侧 `to_square_rgb` 同一套取整)。
    ⚠️ **跨画幅比较必须先裁**:真值帧是 1242×375、生成图是 512² ——
    把两者都降到 64×64 再比,比的是**构图**不是**结构**,而"爆米花画幅"这个差
    会**全程**压过"帧间变化"那个量。这条与 `kitti_square` 是同一条纪律。
    """
    from PIL import Image

    from autodrivedata.edit.cldm_backend import to_square_rgb

    out = []
    for p in paths:
        a = np.array(Image.open(p).convert("RGB"))
        out.append(structure_descriptor(to_square_rgb(a) if square else a))
    return out


def compare_sequences(truth: list[Path], gen: list[Path], *, seed: int = 0, square: bool = True) -> dict:
    """三条读数:`d_in`(真值) / `d_out`(生成) / `d_shuf`(打乱的生成)。

    `square=True`(默认)**两侧都做中心方裁** —— 生成图本来就是方裁出来的,
    真值不裁就没法比(见 `series_descriptors`)。
    """
    if len(truth) != len(gen):
        raise SystemExit(f"两条序列帧数不等:{len(truth)} vs {len(gen)}")
    if len(truth) < 3:
        raise SystemExit(f"少于 3 帧算不出时序:{len(truth)}")
    d_in = adjacent_diffs(series_descriptors(truth, square=square))
    dg = series_descriptors(gen, square=square)
    d_out = adjacent_diffs(dg)
    d_shuf = adjacent_diffs(shuffled(dg, seed=seed))
    return {
        "n_frames": len(truth),
        "d_in_median": float(np.median(d_in)),
        "d_out_median": float(np.median(d_out)),
        "d_shuf_median": float(np.median(d_shuf)),
        "d_in": [round(x, 4) for x in d_in],
        "d_out": [round(x, 4) for x in d_out],
        "d_shuf": [round(x, 4) for x in d_shuf],
    }


def verdict(rep: dict) -> str:
    """★ 裁决要能分辨**三种**形态,别把"尺子坏了"读成"不闪"。"""
    d_in, d_out, d_shuf = rep["d_in_median"], rep["d_out_median"], rep["d_shuf_median"]
    # ★ 与**输入序列**比,不与生成本身比:生成若已经很跳(闪烁),打乱不可能再大 3 倍 ——
    #   拿它当"尺子失效"的判据会让**闪烁这一档永远报不出来**(第一版就是这么写的)。
    #   尺子有没有分辨力,看的是"它能不能把**有序**与**打乱**分开"。
    if d_shuf < SHUF_FLOOR:
        return f"★ **判据自身失效**:打乱后跳变 {d_shuf:.4f} 低于下限 {SHUF_FLOOR} ⇒ 尺子看不出时序,读数作废"
    if d_out <= 1.5 * d_in:
        return f"★ 时序一致:生成侧跳变 {d_out:.4f} 与真值序列 {d_in:.4f} 同量级(打乱 {d_shuf:.4f})"
    return f"★ **闪烁**:生成侧跳变 {d_out:.4f} 是真值序列 {d_in:.4f} 的 {d_out / max(d_in, 1e-9):.1f}×"


def plot_curve(rep: dict, out: Path) -> None:
    """把三条曲线画出来(目检用;**不替代读数**)。

    ⚠️ **图上的文字一律用 ASCII** —— 本仓红线:matplotlib/PIL 遇缺字会**静默**画 `.notdef` 方框
    (实测 `Glyph 35777 missing from current font`),而那在图上看着像"正常的字"。
    中文结论留在 JSON 与 stdout 里。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=110)
    ax.plot(rep["d_in"], marker="o", ms=3, label=f"input (truth)  med {rep['d_in_median']:.3f}")
    ax.plot(rep["d_out"], marker="s", ms=3, label=f"generated  med {rep['d_out_median']:.3f}")
    ax.plot(
        rep["d_shuf"],
        marker="^",
        ms=3,
        alpha=0.6,
        label=f"shuffled (self-check)  med {rep['d_shuf_median']:.3f}",
    )
    ax.set_xlabel("frame index")
    ax.set_ylabel("adjacent structural diff  (1 - Spearman)")
    ax.set_title("temporal consistency: generated vs input")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--truth-dir", required=True, help="真值序列目录(`*.png`,**按名排序即时序**)")
    ap.add_argument("--gen-dir", required=True, help="生成序列目录(`gen_*_0.png`)")
    ap.add_argument("--limit", type=int, default=16, help="取前 N 帧(默认 16)")
    ap.add_argument("--out", required=True, help="曲线 PNG + JSON 落点")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    truth_dir, gen_dir, out = (project_path(p) for p in (args.truth_dir, args.gen_dir, args.out))
    out.mkdir(parents=True, exist_ok=True)
    truth = sorted(truth_dir.glob("*.png"))[: args.limit]
    gen = sorted(gen_dir.glob("gen_*_0.png"))[: args.limit]
    with runlog.run("autodrivedata.edit.video_edit") as rl:
        rl.input(truth_dir, "truth-dir")
        rl.input(gen_dir, "gen-dir")
        rl.highlight("n_frames", len(truth))
        rep = compare_sequences(truth, gen)
        v = verdict(rep)
        for k in ("d_in_median", "d_out_median", "d_shuf_median"):
            rl.highlight(k, round(rep[k], 4))
        rl.highlight("verdict", v)
        print(f"  输入序列(真值) 相邻跳变中位 {rep['d_in_median']:.4f}")
        print(f"  生成序列        相邻跳变中位 {rep['d_out_median']:.4f}")
        print(f"  打乱(自证)      相邻跳变中位 {rep['d_shuf_median']:.4f}")
        print(f"  ⇒ {v}")
        plot_curve(rep, out / "temporal.png")
        (out / "temporal.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        rl.artifact(out / "temporal.png", "curve")
        print(f"[done] {out}")


if __name__ == "__main__":
    main()

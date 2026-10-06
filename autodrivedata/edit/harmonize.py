"""**和谐化**:先立靶,方法只做基线 —— 解决"合成内容与背景在光影/材质上的视觉冲突"。

## ★ 这一格**没有真值靶**,所以判据只能是代理 —— 但代理也要有对照

JD 那句「解决合成内容与背景在光影、材质、分辨率等方面视觉冲突」,
在**生成整幅图**的场景下没有"正确答案"(不像物体移除有 A/B 成对)。
⇒ 只能立一个**可测的代理**:

**靶** = 生成图与**真值帧**在 **LAB 空间的一阶/二阶统计距离**
(均值 = 光影/色温冲突,协方差 = 反差/色彩结构的冲突)。

**方法**只做最经典的灰度世界基线(`Reinhard` 颜色迁移:把生成图的 LAB 统计搬到参考帧的统计),
它的作用是**证明这把尺子能测出已知的改善** —— 尺子立不起来,后面的方法谈都不用谈。

**判据(两条,缺一条读数就不可读)**:
1. 迁移后距离**必须下降**(尺子对"已知的改善"敏感);
2. ★ 与「对齐到**别的帧**的统计」的**对照臂**分得开 ——
   否则"下降"可能只是"把任何图都往中间拉"。

⚠️ **这是代理指标**,不许报成"和谐化达标"。真靶要等
[docs/edit-image-plan.md](../../docs/edit-image-plan.md) §3 阶段 G 那条
"用真值 depth/seg 重合成当靶"的思路落地。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path


#: ⚠️ **OpenCV 的 LAB 有两套量纲**,混用会让 L 通道彻底错(实测输出 L≈0.47 而参考 153.8):
#: `COLOR_RGB2LAB` 吃 **uint8** 时 L∈[0,255];吃 **float32(输入 [0,1])** 时 L∈[0,100]、a/b∈[−127,127]。
#: 本模块**统一走 float 口径**(输入先 /255),返回 uint8 时再 ×255 —— 两端口径由这两个函数独占,
#: 别在别处直接调 `cv2.cvtColor(...LAB...)`。
def to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 → **CIELAB** float32。`L∈[0,100]`,`a/b∈[−127,127]`(float 口径)。"""
    import cv2

    a = np.asarray(rgb, dtype=np.uint8)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"要 (H,W,3) 的 RGB,收到 {a.shape}")
    return cv2.cvtColor(a.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def from_lab(lab: np.ndarray) -> np.ndarray:
    """CIELAB(float 口径)→ RGB uint8。**与 `to_lab` 严格互逆**。"""
    import cv2

    rgb = cv2.cvtColor(np.asarray(lab, dtype=np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def lab_stats(rgb: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """LAB 的 **(均值 (3,), 协方差 (3,3))** —— 这台尺子的全部内容。

    `mask` 给了就**只统计它框住的那批像素**(`harmonize_target` 要的"只在粘贴区内比")。
    ⚠️ 掩膜内少于 2 个像素时协方差是 NaN —— 调用方先保证掩膜非空(那边有判据)。
    """
    lab = to_lab(rgb).reshape(-1, 3)
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        if m.shape != lab.shape[:1]:
            raise ValueError(f"掩膜与图不同画幅:{m.shape} vs {lab.shape[:1]}")
        lab = lab[m]
        if len(lab) < 2:
            raise ValueError(f"掩膜内只有 {len(lab)} 个像素,统计量无定义")
    return lab.mean(axis=0), np.cov(lab, rowvar=False)


def stat_distance(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """两张图的统计距离:**均值差(L2) + 协方差差的归一化 Frobenius**。

    ⚠️ 两项都**必须归一化**,否则协方差项的量纲(σ²)会盖过均值项。
    协方差按 `‖Σa−Σb‖_F / (‖Σa‖_F + ‖Σb‖_F)` 归一,落在 [0,1]。
    ⚠️ **两张图必须用同一张掩膜** —— 否则比的是"不同像素集",那个数没有意义。
    """
    ma, ca = lab_stats(a, mask)
    mb, cb = lab_stats(b, mask)
    d_mean = float(np.linalg.norm(ma - mb) / 10.0)  # L 量程 0–100、ab ±128 ⇒ /10 让三项同尺
    denom = float(np.linalg.norm(ca, "fro") + np.linalg.norm(cb, "fro"))
    d_cov = float(np.linalg.norm(ca - cb, "fro") / denom) if denom > 0 else 0.0
    return d_mean + d_cov


def reinhard_map(
    lab: np.ndarray, ms: np.ndarray, cs: np.ndarray, mr: np.ndarray, cr: np.ndarray
) -> np.ndarray:
    """LAB 像素按 `(μs,Σs) → (μr,Σr)` 做**白化 → 上色**。**唯一实现**。

    `(L−μs)·Σs^(-1/2)·Σr^(1/2) + μr`。输入形状 `(...,3)`,输出同形。
    两个掩膜版与整图版都走它 —— 三处各写一遍迟早漂(本仓"唯一落点"纪律)。
    """
    ev, evec = np.linalg.eigh(cs)
    w = evec @ np.diag(1.0 / np.sqrt(np.maximum(ev, 1e-6))) @ evec.T
    ev2, evec2 = np.linalg.eigh(cr)
    c2 = evec2 @ np.diag(np.sqrt(np.maximum(ev2, 1e-6))) @ evec2.T
    flat = np.asarray(lab, dtype=np.float32).reshape(-1, 3) - ms
    return ((flat @ w.T @ c2.T) + mr).reshape(np.asarray(lab).shape)


def reinhard_transfer(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """把 `src` 的 LAB 统计搬到 `ref` 上(经典 Reinhard)。返回 uint8 RGB。

    ⚠️ 这是**基线**,不是"好方法" —— 它只对齐一阶/二阶统计,**不碰结构**,
    所以它不可能修"材质冲突"。它的用处是**验尺子**。
    """
    ls = to_lab(src)
    ms, cs = lab_stats(src)
    mr, cr = lab_stats(ref)
    return from_lab(reinhard_map(ls, ms, cs, mr, cr))


def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    """二值掩膜膨胀 `k` 步(4 邻域)。**不引 scipy** —— 只为一个膨胀不值一个依赖。"""
    m = np.asarray(mask, dtype=bool)
    for _ in range(int(k)):
        p = np.pad(m, 1)
        m = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


def context_ring(mask: np.ndarray, k: int = 8) -> np.ndarray:
    """粘贴区**外圈** `k` 像素的那一条带 —— 它就是"周围环境"的可测形态。

    `dilate(mask,k) & ~mask`。**这就是方法唯一能看到的"正确光照"**:
    真值区本身是看不见的(看得见就不叫和谐化问题了)。`k` 太大 ⇒ 混进远处别的材质。
    """
    m = np.asarray(mask, dtype=bool)
    return dilate(m, k) & ~m


def reinhard_transfer_masked(
    img: np.ndarray, mask: np.ndarray, ref_img: np.ndarray, ref_mask: np.ndarray
) -> np.ndarray:
    """**只改 `mask` 内**的像素:把 `ref_img[ref_mask]` 的 LAB 统计搬到 `img[mask]` 上。

    这是"按上下文做和谐化"的**最简形态**:参考区一般取 `context_ring(mask)`
    (同一张图上、粘贴区外圈)。它与整图版共用 `reinhard_map`,所以口径只有一处。

    ⚠️ **掩膜外的像素必须逐位不变** —— 第一版把整幅 `from_lab(...)` 出去,
    而 **RGB→LAB→RGB 是有损的**(量化 + 钳位)⇒ 掩膜外也被改了几个 LSB。
    "只改掩膜内"这句话因此是**假的**。现在把结果只贴回掩膜(判据:
    `tests/edit/test_harmonize.py::TestMasked::test_masked_transfer_hits_only_the_mask`)。
    """
    m = np.asarray(mask, dtype=bool)
    ls = to_lab(img)
    ms, cs = lab_stats(img, m)
    mr, cr = lab_stats(ref_img, ref_mask)
    out = ls.copy()
    out[m] = reinhard_map(ls[m], ms, cs, mr, cr)
    res = img.copy()
    res[m] = from_lab(out)[m]  # ★ 只把掩膜内那批贴回去,掩膜外保持原图逐位
    return res


def run_harmonize(gen: list[Path], ref: list[Path], out_dir: Path) -> dict:
    """逐帧:迁移前距离 / 迁移后距离 / **对照臂**(对齐到另一帧的统计)。"""
    from PIL import Image

    if len(gen) != len(ref):
        raise SystemExit(f"两条序列帧数不等:{len(gen)} vs {len(ref)}")
    n = len(gen)
    before, after, ctrl = [], [], []
    for i in range(n):
        g = np.array(Image.open(gen[i]).convert("RGB"))
        r = np.array(Image.open(ref[i]).convert("RGB"))
        before.append(stat_distance(g, r))
        fixed = reinhard_transfer(g, r)
        Image.fromarray(fixed).save(out_dir / f"harmonized_{i:06d}.png")
        after.append(stat_distance(fixed, r))
        # ★ 对照:对齐到**另一帧**的统计,再与**本帧**真值比
        j = (i + n // 2) % n
        rj = np.array(Image.open(ref[j]).convert("RGB"))
        ctrl.append(stat_distance(reinhard_transfer(g, rj), r))
    return {
        "n_frames": n,
        "before_median": float(np.median(before)),
        "after_median": float(np.median(after)),
        "control_median": float(np.median(ctrl)),
        "before": [round(x, 4) for x in before],
        "after": [round(x, 4) for x in after],
        "control": [round(x, 4) for x in ctrl],
    }


def verdict(rep: dict, *, tol: float = 0.02) -> str:
    """**两条都要过**:① 迁移后下降;② 与对照臂分得开。

    只过 ① 不算数 —— "把任何图都往中间拉"也会让距离降。**这条是本模块的核心纪律。**
    """
    b, a, c = rep["before_median"], rep["after_median"], rep["control_median"]
    if a >= b - tol:
        return f"★ 尺子对本方法的改善**不敏感**(迁移前 {b:.4f} → 后 {a:.4f})⇒ 判据立不起来"
    if a >= c - tol:
        return f"★ 下降低于对照臂(后 {a:.4f} vs 对照 {c:.4f})⇒ 分不清「对齐修好了」与「往中间拉了一下」"
    return f"★ 代理指标通过:距离 {b:.4f} → {a:.4f}(对照 {c:.4f})⇒ 尺子可用,但**这只是代理**"


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--gen-dir", required=True, help="生成图目录(`gen_*_0.png`)")
    ap.add_argument("--ref-dir", required=True, help="真值帧目录(`*.png`,按名同序)")
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    gen_dir, ref_dir, out = (project_path(p) for p in (args.gen_dir, args.ref_dir, args.out))
    out.mkdir(parents=True, exist_ok=True)
    gen = sorted(gen_dir.glob("gen_*_0.png"))[: args.limit]
    ref = sorted(ref_dir.glob("*.png"))[: args.limit]
    with runlog.run("autodrivedata.edit.harmonize") as rl:
        rl.input(gen_dir, "gen-dir")
        rl.input(ref_dir, "ref-dir")
        rl.highlight("n_frames", len(gen))
        rep = run_harmonize(gen, ref, out)
        v = verdict(rep)
        for k in ("before_median", "after_median", "control_median"):
            rl.highlight(k, round(rep[k], 4))
        rl.highlight("verdict", v)
        print(f"  迁移前统计距离(生成 vs 真值) 中位 {rep['before_median']:.4f}")
        print(f"  迁移后                       中位 {rep['after_median']:.4f}")
        print(f"  对照(对齐到别的帧)           中位 {rep['control_median']:.4f}")
        print(f"  ⇒ {v}")
        (out / "harmonize.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        rl.artifact(out / "harmonize.json", "summary")


if __name__ == "__main__":
    main()

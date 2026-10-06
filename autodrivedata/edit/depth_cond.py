"""度量深度 → ControlNet `control_sd15_depth` 的条件图。**纯值**:禁 carla / torch。

## 口径从哪来(不对齐它,生成就是在喂一个尺度不对的条件)

`hdMapGitHub/ControlNet/annotator/midas/__init__.py:MidasDetector.__call__` 的三步:

```python
disparity = model(img)                      # 逆深度类:大 = 近
p = (disparity - disparity.min()) / (disparity - disparity.min()).max()   # ★ 逐图 min-max
cond = (p * 255.0).clip(0, 255).astype(np.uint8)                          # ★ 越白越近
```

本项目有的是**真值度量深度**(CARLA `sensor.camera.depth` → `calib.depth_codec.decode_depth` → 米)
⇒ 要补的只有 **`disparity = 1 / depth`** 这一步,其余逐字相同。

⚠️ **`min-max` 是单调的**,所以"直接归一化 `d`"也能得到一张像样的图 —— 但那**恰好是反的**
(白 = 远)。这一条有反向自证(见 `tests/edit/test_depth_cond.py::TestInversion`):
与 MiDaS 的秩相关会从 **+ρ 翻成 −ρ**。

## 这一层的判据:**秩相关**,不是逐像素相等

真值条件与 MiDaS 条件**不可能逐像素相等** —— MiDaS 给的是**仿射不变的相对视差**,
不是物理的 `1/d`。能问的是**两者排序一致到什么程度**:Spearman ρ,对仿射变换不变,
正对"口径差"这个差异本身。**ρ 高 = 换算没写错;ρ 低 = 换算错 或 真值条件确实带了 MiDaS 没有的信息**
—— 两条都必须报,不许只报一个数(本仓纪律:没有对照的读数读不出东西)。

⚠️ **一处已知的域差,不许当成 bug**:CARLA 深度相机对**天空**给的是一个**有限值**
(实测某帧上 1/8 区域的 max 是 17.08 m —— 那个位置是建筑不是天空;`DEPTH_RANGE_M=1000`
而全场 max 只有 82.077 m),而 MiDaS 会把天空压到视差**最远端**。
⇒ 若某帧真的有天空露出来,ρ 会被这一小块拉低。用法上按"有效像素"算,并把无效占比报出来。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 条件图的分辨率口径(与官方 demo 的 `detect_resolution` 默认同值)。
COND_RES = 512

#: 判定"这个深度值无效"的上界(米)。CARLA 的量程是 1000(`depth_codec.DEPTH_RANGE_M`),
#: 而真实场景的远平面通常远小于它 ⇒ 贴着量程的值按"无效"处理,不给它一个假的视差。
INVALID_DEPTH_M = 999.0


def disparity_from_metric(depth_m: np.ndarray, *, invalid_depth_m: float = INVALID_DEPTH_M) -> np.ndarray:
    """度量深度(米)→ 视差类量 `1/d`。**大 = 近**,与 MiDaS 同向。

    ⚠️ 无效像素(`d <= 0` / `d >= invalid_depth_m` / NaN)被置 **0** ——
    即"最远",与 MiDaS 对天空的处理同向。**置 0 而不是置 NaN**:
    后面 min-max 会吃掉 NaN 且不报错,得到一张**静默错**的图。
    """
    d = np.asarray(depth_m, dtype=np.float64)
    bad = ~np.isfinite(d) | (d <= 0.0) | (d >= float(invalid_depth_m))
    safe = np.where(bad, np.inf, d)  # 避免 1/0 触发 warning;下面统一改回 0
    disp = 1.0 / safe
    disp[bad] = 0.0
    return disp


def minmax_uint8(x: np.ndarray) -> np.ndarray:
    """**逐图** min-max 归一到 0–255 uint8。这就是 MiDaS 那三步的第二三步。

    ⚠️ `max == min`(常量图)**必须抛**,不许静默返回全黑:
    一张全黑的条件图会让 ControlNet 退化成无条件生成,而"生成得还行"看不出来条件其实没进去。
    """
    a = np.asarray(x, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 0.0:
        raise ValueError(
            f"条件图退化:min={lo} max={hi} ⇒ 归一化无定义。"
            "常量深度/全无效帧不能当条件 —— 静默返回全黑会让下游把'无条件生成'读成'条件生效'"
        )
    return ((a - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)


def rank_uniform_uint8(x: np.ndarray) -> np.ndarray:
    """按**秩**映射到均匀分布(直方图等化)。**单调 ⇒ 保序**,与 `minmax_uint8` 看到的是同一个排序。

    ## 为什么需要它(2026-10-05 实测)

    真值深度的 `1/d` 在远场被**强烈压缩**:某帧深度跨 3–82 m,`1/d` 让 10–80 m 全落进
    视差 0.100→0.012,而 3–10 m 占 0.33→0.10 ⇒ min-max 之后**大部分像素挤在 131–227**,
    生成一张**发灰的平场**。而 MiDaS 的条件铺满 0–255。

    ⚠️ **这个差别在秩相关上完全看不见**(min-max 也是单调的)——
    但 **ControlNet 吃的是像素值,不是排序**。一张挤在窄带里的条件图对 control encoder
    是**分布外**的输入。⇒ 这是"ρ 正常但生成不对"的那一类错。

    ⚠️ 保序这一条**有反向自证**(见 tests):等化前后与 MiDaS 的 Spearman **实质不变**。

    ⚠️ 但**不是逐位不变** —— 实测 `minmax` 0.00335 vs `histeq` 0.00327。
    差异只来自 **uint8 量化**:min-max 在窄带上把大量不同值压进同一个字节(并列多),
    秩等化则把它们铺成不同字节(并列少),而**并列会改秩**。
    ⇒ 报"两者相同"时必须给容差;这条本身也是"**ρ 看不见分布**"的证据:
    分布差了几倍,ρ 只动 0.0001。
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"要二维条件图,收到 {a.shape}")
    n = a.size
    if n < 2:
        raise ValueError("少于 2 个像素,秩无定义")
    order = np.argsort(a, axis=None, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return (ranks / (n - 1) * 255.0).reshape(a.shape).astype(np.uint8)


def match_histogram_uint8(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """把 `x` 的**强度分布**搬到 `ref` 的分布上(保序,逐分位点取值)。

    比 `rank_uniform_uint8` 更贴目标:若手上有一张 MiDaS 条件图当**参考分布**,
    就直接对齐到它,而不是对齐到均匀分布(均匀并不等于 MiDaS 的分布)。
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    r = np.sort(np.asarray(ref, dtype=np.float64).ravel())
    if r.size < 2 or a.size < 2:
        raise ValueError("直方图匹配需要两张图各自至少 2 个像素")
    xs = np.sort(a)
    q = np.searchsorted(xs, a, side="left") / (a.size - 1)
    idx = np.clip((q * (r.size - 1)).round().astype(np.int64), 0, r.size - 1)
    return r[idx].reshape(np.asarray(x).shape).astype(np.uint8)


#: 条件图的强度归一化方式。`minmax` = 官方 MiDaS 口径;另两个保序、但把分布铺开。
NORMALIZATIONS = {
    "minmax": lambda x, ref=None: minmax_uint8(x),
    "histeq": lambda x, ref=None: rank_uniform_uint8(x),
    "match": lambda x, ref=None: minmax_uint8(x) if ref is None else match_histogram_uint8(x, ref),
}


def metric_depth_to_cond(
    depth_m: np.ndarray,
    *,
    invalid_depth_m: float = INVALID_DEPTH_M,
    normalize: str = "minmax",
    ref: np.ndarray | None = None,
) -> np.ndarray:
    """CARLA 真值深度(米)→ ControlNet 深度条件图 (H,W) uint8,**白 = 近**。

    `normalize` 见 `NORMALIZATIONS`;**三者保序**,差别只在强度分布(见 `rank_uniform_uint8`)。
    """
    if normalize not in NORMALIZATIONS:
        raise KeyError(f"未知的归一化 {normalize!r};可选 {sorted(NORMALIZATIONS)}")
    disp = disparity_from_metric(depth_m, invalid_depth_m=invalid_depth_m)
    return NORMALIZATIONS[normalize](disp, ref)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """两张条件图的 Spearman 秩相关。对仿射变换不变 ⇒ 正对"口径差"这件事。

    只在**两张都有定义**的像素上算(无效像素已由 `disparity_from_metric` 置 0,
    这里再排掉"两边都是 0"的常量段 —— 它们对秩没有信息,却会稀释相关性)。
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"两张条件图尺寸不同:{x.shape} vs {y.shape}")
    keep = ~(np.isnan(x) | np.isnan(y))
    if keep.sum() < 2:
        raise ValueError("有效像素不足 2 个,算不出秩相关")
    rho, _ = stats.spearmanr(x[keep], y[keep])
    return float(rho)


# --------------------------------------------------------------------------- CLI
def _load_gt_depth(root: Path, fid: int) -> np.ndarray:
    """从采集 root 读某一帧的真值深度(`capture/depth/p0/{fid:05d}.npy`)。"""
    cands = sorted(root.glob("capture/depth/p*/"))
    if not cands:
        raise SystemExit(f"{root} 下没有 capture/depth/p*/,这不是 `collect_3dgs` 的产物")
    p = cands[0] / f"{fid:05d}.npy"
    if not p.exists():
        raise SystemExit(f"缺帧 {p}")
    return np.load(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root", required=True, help="collect_3dgs 的采集 root(含 capture/depth/p*/)")
    ap.add_argument("--frames", default="0", help="帧号:逗号分隔或 a-b")
    ap.add_argument("--out", required=True, help="条件图落盘目录")
    ap.add_argument("--no-runlog", action="store_true", help="不落 logs/ 三件套")
    args = ap.parse_args()

    root = project_path(args.root)
    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fids = _parse_frames(args.frames)

    with runlog.run("autodrivedata.edit.depth_cond") as rl:
        rl.input(root, "capture-root")
        rows = []
        for fid in fids:
            d = _load_gt_depth(root, fid)
            cond = metric_depth_to_cond(d)
            dst = out / f"depthcond_{fid:05d}.png"
            _write_png(dst, cond)
            row = {
                "fid": fid,
                "depth_min_m": float(d.min()),
                "depth_median_m": float(np.median(d)),
                "depth_max_m": float(d.max()),
                "cond_mean": float(cond.mean()),
                "invalid_share": float((disparity_from_metric(d) == 0.0).mean()),
            }
            rows.append(row)
            print(
                f"[{fid:05d}] 深度 {row['depth_min_m']:.2f}–{row['depth_max_m']:.2f} m | "
                f"条件均值 {row['cond_mean']:.1f} | 无效占比 {row['invalid_share']:.4f} → {dst.name}"
            )
            rl.artifact(dst, "depth-cond")
        summ = out / "depth_cond_summary.json"
        summ.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        rl.highlight("n_frames", len(rows))
        rl.highlight("invalid_share_max", round(max(r["invalid_share"] for r in rows), 6) if rows else None)
        rl.artifact(summ, "summary")
        print(f"[done] {summ.resolve()}")


def _write_png(path: Path, gray: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.asarray(gray, dtype=np.uint8)).save(path)


def _parse_frames(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    if not out:
        raise SystemExit(f"--frames 解析出空列表:{spec!r}")
    return out


if __name__ == "__main__":
    main()

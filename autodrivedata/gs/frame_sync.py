"""3DGS capture 的**帧同步判据**(离线纯值:`numpy` + `PIL` 读 `*.npy`/`*.png`)。

## 判的是一件事:第 0 帧是不是一个**采集瞬态**

`collect_3dgs` 原实现有个"一次 tick 只 get 一次"的预热,挂上 `listen` 后一共 tick 了 4 次
却只取走 3 帧 ⇒ 每个队列恒定剩 1 帧陈旧帧,而 FIFO 的 `get()` 取的是**最旧**那帧
⇒ 主循环整段滞后 1 帧。症状(2026-10-03 实测,见
[docs/edit-3dgs-plan.md](../../docs/edit-3dgs-plan.md) §1.5):

| 判据 | 坏的第 0 帧 | 其余 89 帧 |
|---|---|---|
| 与下一帧的平均绝对差 | **75.4** | 30.1(1↔2) |
| 中位深度 | **24.6 m** | 9.4–10.5 m |
| `>50 m` 占比 | **22.6%** | **全部 0.0%** |

**位姿是对的**(第 0 帧与第 1 帧只差 0.42 m / 4°,符合 6 m 半径、4° 步进)—— 错的是图像。
所以判据只能从**图像/深度内容**下,不能从位姿下(位姿上看不出任何异常)。

## 三个读数 + 一条裁决(缺一都读不出结论)

| 量 | 问的是 |
|---|---|
| `over50_first` vs **其余帧的中位** + `FAR_SHARE_TOL` | 首帧的**远点占比**是否超出其余帧的典型水平(§1.5 的 22.6% vs 0.0%) |
| `median_first` vs 其余帧的 **`[p5, p95]`** | 首帧的**中位深度**是否落在其余帧的范围内 |
| `adj_first` vs `k × 其余相邻帧的中位差` | 首帧与下一帧的**内容跳变**是否异常 |

⚠️ **后两条的"其余"取分位/中位,不取 min/max** —— 这是判据的判别力所在。首版用 `max` 当上界,
而"2 帧瞬态"恰好让 `max == 首帧值`(0.2262 vs 0.2264),**判据差点放过去**。90 帧里坏两三帧,
不该能把上界撑到跟坏帧一样高。

**裁决**:三条全过 ⇒ 首帧与其余帧同分布;任一不过 ⇒ 首帧是瞬态,这份 capture 的
`psnr_val` 口径可疑(归档里 `psnr_val` 是 `--val-frames 0` 算的,就是这么来的)。
⚠️ 判据**只看首帧** —— 它不为"这份 capture 没问题"背书,只为"首帧不是瞬态"背书。

## 第三态(未判)

一个俯仰的帧数少于 `MIN_FRAMES`(或压根没有深度文件)时判**未判**(`None`),
不是"通过"也不是"不通过" —— 与 `perception/static_eval` 的「n=0 是不判」同一条纪律:
2 帧的样本里"首帧正常"与"首帧异常"都可能只是噪声,把它读成结论就是**把不知道伪装成结论**。

## 已知边界

- **修好的采集器不会自动让旧 capture 变绿** —— 归档的 `outputs/3dgs/capture` 是 2026-09-18
  采的,首帧照样是瞬态。判据是给**重采的** capture 用的(§1.5 的触发重评条件)。
- 判据**不吃 `poses_*.json`**:位姿在这件事上**没有判别力**(见上)。

用法:
  python -m autodrivedata.gs.frame_sync --capture outputs/3dgs/capture
  python -m autodrivedata.gs.frame_sync --capture outputs/3dgs_sync/capture
  python -m autodrivedata.gs.frame_sync --self-test        # 不出数据,只验尺子
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 远点阈值(m)。CARLA 远平面实测 999.9997 ⇒ 天空也落进这一桶。
FAR_DEPTH_M = 50.0
#: 一个俯仰至少要这么多帧才判(少了是**未判**,见模块头注)。
MIN_FRAMES = 5
#: 首帧相邻差的容忍倍率:超过"其余相邻帧中位差 × 这个数"才算异常跳变。
ADJ_DIFF_MARGIN = 2.0
#: 远点占比的绝对容差。
#:
#: ⚠️ **不是"把判据调绿"的那个容差**(那条红线针对的是"判错了就把阈值放宽").
#: 这一条处理的是**另一个**毛病:占比是个像素计数比例,两帧都该是 0 时会给
#: `3e-6` vs `0.0` 这种差 —— 拿 `<=` 硬比会**在"两边都是 0"的情况下判不通过**,
#: 而那种红会逼着人把判据关掉。1e-4 远小于任何真实差异(§1.5 是 0.226 vs 0.000),
#: 且比"一个像素"在 1242×375 里的占比(2.2e-6)高两个数量级 ⇒ 是"数出来的"而不是"拍的"。
FAR_SHARE_TOL = 1e-4
#: 中位深度区间取下 5% / 上 95% 分位,而不是 min/max。
#: **这是判据的判别力所在**:首个版本用 min/max,而 2 帧瞬态让 max 恰好等于首帧值
#: (0.2262 vs 0.2264),判据**差点放过去**。取分位后,90 帧里坏掉两三帧不再能撑起上界。
DEEP_LO_PCT, DEEP_HI_PCT = 5.0, 95.0


@dataclass(frozen=True)
class FrameStat:
    """单帧的两个内容读数。**不含位姿** —— 位姿在这件事上没有判别力。"""

    index: int
    median_depth: float
    far_share: float


@dataclass
class PitchReport:
    """一个俯仰的汇合读数 + 裁决。"""

    pitch: float
    n: int
    first: FrameStat
    rest_median_lo: float
    rest_median_hi: float
    rest_far_median: float
    adj_first: float
    adj_rest_median: float
    adj_pairs: int

    @property
    def far_ok(self) -> bool:
        return self.first.far_share <= self.rest_far_median + FAR_SHARE_TOL

    @property
    def median_ok(self) -> bool:
        return self.rest_median_lo <= self.first.median_depth <= self.rest_median_hi

    @property
    def adj_ok(self) -> bool:
        return self.adj_first <= ADJ_DIFF_MARGIN * max(self.adj_rest_median, 1e-6)

    def verdict(self) -> tuple[bool | None, str]:
        """`None` = **未判**(帧数不够),不是通过也不是不通过。"""
        if self.n < MIN_FRAMES:
            return None, f"未判(只有 {self.n} 帧 < {MIN_FRAMES})"
        bad = []
        if not self.far_ok:
            bad.append(
                f"远点占比 {self.first.far_share:.4f} > 其余中位 {self.rest_far_median:.4f}+{FAR_SHARE_TOL}"
            )
        if not self.median_ok:
            bad.append(
                f"中位深度 {self.first.median_depth:.1f} m 落在其余 "
                f"[p{DEEP_LO_PCT:.0f}, p{DEEP_HI_PCT:.0f}] "
                f"[{self.rest_median_lo:.1f}, {self.rest_median_hi:.1f}] 之外"
            )
        if not self.adj_ok:
            bad.append(
                f"相邻差 {self.adj_first:.1f} > {ADJ_DIFF_MARGIN}×{self.adj_rest_median:.1f}(其余相邻帧中位)"
            )
        if bad:
            return False, "首帧是采集瞬态 —— " + ";".join(bad)
        return True, "首帧与其余帧同分布"


def frame_stat(idx: int, depth: np.ndarray) -> FrameStat:
    """深度图 → `(中位深度, >50 m 占比)`。**纯函数**。"""
    d = np.asarray(depth, dtype=np.float64)
    if d.size == 0:
        raise ValueError("空深度图")
    return FrameStat(index=idx, median_depth=float(np.median(d)), far_share=float((d > FAR_DEPTH_M).mean()))


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """两帧 RGB 的**平均绝对差**(0–255 量纲;与 §1.5 的 75.4 / 30.1 同口径)。

    ⚠️ 两帧形状必须相同 —— 不同尺寸的 `mean` 会**静默**给出一个看着正常的数。
    """
    x, y = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if x.shape != y.shape:
        raise ValueError(f"两帧形状不同:{x.shape} vs {y.shape}(错配的 mean 不会报错,只会给出错数)")
    return float(np.abs(x - y).mean())


def _load_depth(capture: Path, pitch: float, i: int) -> np.ndarray:
    return np.load(capture / "depth" / f"p{int(pitch)}" / f"{i:05d}.npy")


def _load_image(capture: Path, pitch: float, i: int) -> np.ndarray:
    with Image.open(capture / "images" / f"p{int(pitch)}" / f"{i:05d}.png") as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def audit_pitch(capture: Path, pitch: float) -> PitchReport:
    """一个俯仰圈 → 读数 + 裁决输入。**首帧恒为 i=0**(见模块头注)。"""
    d0 = capture / "depth" / f"p{int(pitch)}"
    if not d0.is_dir():
        raise SystemExit(f"{d0} 不存在 —— 这不是一份 3dgs capture(见 collect_3dgs 落盘说明)")
    idx = sorted(int(p.stem) for p in d0.glob("*.npy"))
    if not idx:
        raise SystemExit(f"{d0} 里没有深度 .npy")
    if idx != list(range(len(idx))):
        # 缺帧会让"相邻差"跨过一段空白而偏大 —— 那不是"瞬态",是数据不全
        raise SystemExit(f"p{int(pitch)} 的帧号不连续:{idx[:5]}…(缺帧会让相邻差读数失真)")
    n = len(idx)
    stats = [frame_stat(i, _load_depth(capture, pitch, i)) for i in idx]
    first = stats[0]
    rest = stats[1:]
    if not rest:
        return PitchReport(
            pitch, n, first, first.median_depth, first.median_depth, first.far_share, 0.0, 0.0, 0
        )
    imgs = [_load_image(capture, pitch, i) for i in idx]
    adj = [mean_abs_diff(imgs[i], imgs[i + 1]) for i in range(n - 1)]
    rest_adj = sorted(adj[1:])  # 去掉首帧那一对
    rm = np.array([s.median_depth for s in rest])
    return PitchReport(
        pitch=pitch,
        n=n,
        first=first,
        rest_median_lo=float(np.percentile(rm, DEEP_LO_PCT)),
        rest_median_hi=float(np.percentile(rm, DEEP_HI_PCT)),
        rest_far_median=float(np.median([s.far_share for s in rest])),
        adj_first=adj[0],
        adj_rest_median=float(np.median(rest_adj)) if rest_adj else 0.0,
        adj_pairs=len(rest_adj),
    )


def audit_capture(capture: Path, pitches: list[float] | None = None) -> dict[str, Any]:
    """整份 capture 跑一遍。`pitches=None` ⇒ 读 `pitches.json` 枚举。"""
    if pitches is None:
        pj = capture / "pitches.json"
        if not pj.is_file():
            raise SystemExit(f"{pj} 不存在 —— 无法枚举俯仰,请显式给 --pitches")
        pitches = [float(x) for x in json.loads(pj.read_text(encoding="utf-8"))]
    return {"capture": str(capture), "reports": {p: audit_pitch(capture, p) for p in pitches}}


def report(res: dict[str, Any]) -> bool:
    """打印 + 裁决。返回"有没有俯仰判不通过"。"""
    print(f"\n=== 3DGS capture 帧同步判据({res['capture']})===")
    print(
        f"{'俯仰':>6}{'帧':>5}{'首帧中位':>10}{'其余中位 p5–p95':>20}{'首帧远点':>11}"
        f"{'其余远点中位':>13}{'首帧相邻差':>12}{'其余相邻中位':>13}   裁决"
    )
    ok_all = True
    n_unjudged = 0
    for p, r in res["reports"].items():
        ok, why = r.verdict()
        if ok is None:
            n_unjudged += 1
            mark = "⚠️ "
        else:
            ok_all &= ok
            mark = "✅ " if ok else "❌ "
        print(
            f"{p:>6.0f}{r.n:>5}{r.first.median_depth:>10.1f}"
            f"{f'[{r.rest_median_lo:.1f}, {r.rest_median_hi:.1f}]':>20}"
            f"{r.first.far_share:>11.4f}{r.rest_far_median:>13.4f}"
            f"{r.adj_first:>12.1f}{r.adj_rest_median:>13.1f}   {mark}{why}"
        )
    if n_unjudged:
        print(f"\n⚠️ {n_unjudged} 个俯仰**未判**(帧数 < {MIN_FRAMES})—— 不计入裁决。")
    print(
        f"\n裁决口径:首帧的远点占比 ≤ 其余帧上界、中位深度落在其余帧区间内、"
        f"首帧相邻差 ≤ {ADJ_DIFF_MARGIN}× 其余相邻帧中位差。"
        "\n⚠️ 判据**只看首帧** —— 它不为「这份 capture 没问题」背书,只为「首帧不是瞬态」背书。"
    )
    return ok_all


def _synth_capture(root: Path, *, broken_first: bool, n: int = 8) -> Path:
    """合成一份 capture:地面深度 8 m;`broken_first` 时首帧整幅换 999(远平面)。"""
    cap = root / "capture"
    (cap / "images/p0").mkdir(parents=True, exist_ok=True)
    (cap / "depth/p0").mkdir(parents=True, exist_ok=True)
    (cap / "pitches.json").write_text("[0]", encoding="utf-8")
    rng = np.random.default_rng(7)
    base = rng.integers(0, 40, size=(8, 12, 3), dtype=np.uint8)
    for i in range(n):
        img = base.copy()
        if broken_first and i == 0:
            img = np.full_like(base, 250)
        elif i:
            img = np.clip(base.astype(np.int16) + rng.integers(-2, 3, base.shape), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(cap / "images/p0" / f"{i:05d}.png")
        d = np.full((8, 12), 8.0, dtype=np.float32)
        if broken_first and i == 0:
            d = np.full((8, 12), 999.0, dtype=np.float32)
        np.save(cap / "depth/p0" / f"{i:05d}.npy", d)
    return cap


def self_test() -> bool:
    """**尺子自证**(不需要 CARLA):坏首帧必须红 / 好首帧必须绿 / 帧太少必须未判。"""
    ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cap_bad = _synth_capture(root / "bad", broken_first=True)
        rb = audit_pitch(cap_bad, 0.0)
        v, why = rb.verdict()
        good = v is False and not rb.far_ok
        print(f"① 坏首帧(整幅 999 m)→ verdict={v} —— {'✅' if good else '❌'}  {why}")
        ok &= good

        cap_ok = _synth_capture(root / "ok", broken_first=False)
        r2 = audit_pitch(cap_ok, 0.0)
        v2, why2 = r2.verdict()
        good = v2 is True
        print(f"② 干净首帧 → verdict={v2} —— {'✅' if good else '❌'}  {why2}")
        ok &= good

        cap_tiny = _synth_capture(root / "tiny", broken_first=True, n=3)
        v3, why3 = audit_pitch(cap_tiny, 0.0).verdict()
        good = v3 is None
        print(f"③ 只有 3 帧(< {MIN_FRAMES})→ verdict={v3}(必须 None)—— {'✅' if good else '❌'}  {why3}")
        ok &= good

        # ④ 形状不等的两帧**必须抛**(错配的 mean 不会报错,只会给出错数)
        try:
            mean_abs_diff(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 5, 3), np.uint8))
            good = False
        except ValueError:
            good = True
        print(f"④ 尺寸不等时 mean_abs_diff 抛 ValueError —— {'✅' if good else '❌'}")
        ok &= good
    return bool(ok)


def main() -> None:
    ap = argparse.ArgumentParser(description="3DGS capture 帧同步判据(首帧是不是采集瞬态)")
    ap.add_argument("--capture", default="outputs/3dgs/capture")
    ap.add_argument("--pitches", default=None, help="逗号分隔;默认读 capture/pitches.json")
    ap.add_argument("--self-test", action="store_true", help="不出数据,只验尺子")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)

    cap = project_path(args.capture)
    pitches = [float(x) for x in args.pitches.split(",")] if args.pitches else None
    with runlog.run("autodrivedata.gs.frame_sync") as rl:
        rl.input(str(cap), "capture")
        rl.highlight("far_depth_m", FAR_DEPTH_M)
        rl.highlight("adj_diff_margin", ADJ_DIFF_MARGIN)
        res = audit_capture(cap, pitches)
        ok = report(res)
        for p, r in res["reports"].items():
            v, _ = r.verdict()
            rl.highlight(f"p{int(p)}.first_far_share", round(r.first.far_share, 4))
            rl.highlight(f"p{int(p)}.rest_far_max", round(r.rest_far_max, 4))
            rl.highlight(f"p{int(p)}.first_median_m", round(r.first.median_depth, 2))
            rl.highlight(f"p{int(p)}.adj_first", round(r.adj_first, 2))
            rl.highlight(f"p{int(p)}.adj_rest_median", round(r.adj_rest_median, 2))
            rl.highlight(f"p{int(p)}.n", r.n)
            rl.highlight(f"p{int(p)}.verdict", "未判" if v is None else ("过" if v else "不过"))
        rl.highlight("pass", ok)
        rl.note("首帧不是瞬态" if ok else "首帧是采集瞬态 —— 该 capture 的 psnr_val 口径可疑")


if __name__ == "__main__":
    main()

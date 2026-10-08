"""在 §1.12 的**真值靶**上训一条小网络 —— 这是"和谐化"那一格真正要的东西。

## 它接的是哪一段

§1.12 立了靶(同帧跨天气粘贴,真值 = 原图)并量出**统计对齐基线**的残差:
掩膜内距离 3.3128 → **1.5385**(错光源对照 2.7054)。
那条基线的**上限是"只对齐一阶/二阶统计"** —— 它不碰结构,也修不了材质。
本模块训一条小 U-Net,看**把残差压到多少**,以及**压下去的是哪一部分**。

## ★ 一条不许省的纪律:**按天气对分开**

训练集与评测集**必须是不同的天气对**。同一对既训又评 ⇒ 网络可以"背下这次光照差",
而那个差是**全局**的 —— 背下来读数会虚高,而它在真实用途里毫无价值。

⇒ 默认:**训练** `day→sunset` + `day→rain`;**评测** `day→fog` + `day→wet`(**留出对**)。

## 三条读数,少一条都读不出结论

| 臂 | 是什么 |
|---|---|
| **composite** | 原样(不处理)—— 地板 |
| **reinhard** | §1.12 的统计对齐基线 —— **要跟它比**,不是跟自己比 |
| **trained** | 本模块训出来的 —— 只有**同时**低于上两条才算数 |

★ 判据 = `stat_distance(臂, truth)` **在掩膜内**;三条都报,并报**相对 reinhard 的增量**。
⚠️ 这是**机制验证**(几百个 patch、一个场景、单卡几分钟),**不是"训好了能用"**。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autodrivedata.edit.harmonize import reinhard_transfer_masked, stat_distance
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path


def _conv(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    """三级 U-Net,**预测一个残差**(输出 + 输入)—— 残差式让"不动"是网络的默认解。

    ⚠️ 直接预测整幅图时,网络一上来就要学会"重建输入",那部分容量和梯度全花在恒等映射上。
    """

    def __init__(self, base: int = 24) -> None:
        super().__init__()
        self.e1, self.e2, self.e3 = _conv(4, base), _conv(base, base * 2), _conv(base * 2, base * 4)
        self.d2, self.d1 = _conv(base * 4 + base * 2, base * 2), _conv(base * 2 + base, base)
        self.head = nn.Conv2d(base, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        u2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2, mode="nearest"), e2], 1))
        u1 = self.d1(torch.cat([F.interpolate(u2, scale_factor=2, mode="nearest"), e1], 1))
        return torch.clamp(x[:, :3] + self.head(u1), 0.0, 1.0)


def load_split(ds: Path, tags: list[str], limit: int | None = None):
    """把若干天气对的 `.npz` 读成 `(comp, truth, mask)` 三个 `(N,3,H,W)/(N,1,H,W)` 张量。"""
    xs, ys, ms = [], [], []
    for t in tags:
        files = sorted((ds / t).glob("*.npz"))
        for f in files[:limit] if limit else files:
            z = np.load(f)
            xs.append(z["composite"].astype(np.float32) / 255.0)
            ys.append(z["truth"].astype(np.float32) / 255.0)
            ms.append(z["mask"].astype(np.float32))
    if not xs:
        raise SystemExit(f"这些天气对下一个样本都没有:{tags}(先跑 `edit.harmonize_data`)")
    # ⚠️ 掩膜是 `(H,W)` **没有通道维**,与图不是同一种形状 —— 第一版把两者套同一个 lambda,
    #    当场 `permute(sparse_coo): input.dim()=3 != len(dims)=4`。
    to_img = lambda a: torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2).contiguous()  # noqa: E731
    to_mask = lambda a: torch.from_numpy(np.stack(a)).unsqueeze(1).contiguous()  # noqa: E731
    return to_img(xs), to_img(ys), to_mask(ms)


def train(
    ds: Path, train_tags: list[str], *, epochs: int, batch: int, lr: float, seed: int, limit: int | None
):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    comp, truth, mask = load_split(ds, train_tags, limit)
    n = comp.shape[0]
    model = TinyUNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"[train] {n} 个 patch  设备 {dev}  参数量 {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k")
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            x, y, m = comp[idx].to(dev), truth[idx].to(dev), mask[idx].to(dev)
            out = model(torch.cat([x, m], 1))
            # 掩膜内是**要改的地方**,外面是**参考** ⇒ 两边都要监督,
            # 但掩膜内给更高权重:只督掩膜外,网络学成恒等即可交差。
            loss = (torch.abs(out - y) * (1.0 + 3.0 * m)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if (ep + 1) % max(1, epochs // 8) == 0 or ep == 0:
            print(f"  [ep {ep + 1:>3}/{epochs}] loss {tot / n:.5f}")
    return model, dev


def evaluate(model, dev: str, ds: Path, tags: list[str], limit: int | None) -> dict:
    """三条臂在**留出天气对**上的掩膜内距离。**逐对报,不池在一起。**

    ## ★★ 为什么必须逐对报(2026-10-07 实测踩到)

    第一版把**所有留出 patch 池在一起取中位**,而四个对的距离是 **双峰**的
    (雾对 4.07、湿对 1.19,差 3.4 倍)⇒ 那个中位 **2.274 不代表任何一对**,
    而它正好落在两峰之间,把"2 对赢 2 对输"读成了"全面更差"。
    ⇒ **池化统计量的前提是"可比的分布"**;当留出集由**不同难度**的子集构成时,
    池化会把结论去掉。至少要把每一对单独报出来。
    """
    out_rows = []
    for t in tags:
        out_rows.append(_eval_one(model, dev, ds, [t], t, limit))
    # 汇总那行**只在所有对同量级时才有意义**;仍然报,但把逐对摆在它前面。
    agg_rows = [r for o in out_rows for r in o["per_patch"]]
    spread = max(o["median"]["composite"] for o in out_rows) / max(
        min(o["median"]["composite"] for o in out_rows), 1e-9
    )
    agg = _summarize(agg_rows)
    agg["composite_spread_across_pairs"] = round(float(spread), 2)
    agg["mixed_pairs"] = spread > 1.5
    return {"per_pair": {o["tag"]: o["median"] for o in out_rows}, "median": agg}


def _eval_one(model, dev: str, ds: Path, tags: list[str], tag: str, limit: int | None) -> dict:
    comp, truth, mask = load_split(ds, tags, limit)
    with torch.no_grad():
        out = model(torch.cat([comp, mask], 1).to(dev)).cpu()

    rows, agg = [], {"composite": [], "reinhard": [], "trained": []}
    n_ring_empty = 0
    for i in range(comp.shape[0]):
        c = (comp[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        t = (truth[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        g = (out[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        m = mask[i, 0].numpy().astype(bool)
        if not m.any():
            continue
        ring = ~m
        if not ring.any():
            # ⚠️ **不静默跳过**:掩膜铺满 ⇒ 这一条臂连定义都没有(没有"周围"当参考)。
            #    数据集那一步已按 `MAX_MASK_SHARE` 滤过,这里是第二道。
            n_ring_empty += 1
            continue
        fixed = reinhard_transfer_masked(c, m, c, ring)
        d = {
            "composite": stat_distance(c, t, m),
            "reinhard": stat_distance(fixed, t, m),
            "trained": stat_distance(g, t, m),
        }
        for k in agg:
            agg[k].append(d[k])
        rows.append(d)
    if not rows:
        raise SystemExit(f"留出集 `{tag}` 上一个可用样本都没有")
    return {"tag": tag, "median": _summarize(rows, n_ring_empty), "per_patch": rows}


def _summarize(rows: list[dict], n_ring_empty: int = 0) -> dict:
    med = {k: float(np.median([r[k] for r in rows])) for k in ("composite", "reinhard", "trained")}
    med["n"] = len(rows)
    med["n_ring_empty_skipped"] = n_ring_empty
    med["trained_vs_reinhard"] = med["trained"] - med["reinhard"]
    med["trained_vs_composite"] = med["trained"] - med["composite"]
    med["trained_wins"] = int(sum(1 for r in rows if r["trained"] < r["reinhard"]))
    med["reinhard_wins"] = med["n"] - med["trained_wins"]
    return med


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--ds", required=True, help="`edit.harmonize_data` 的产物目录")
    ap.add_argument("--train", default="day_clear__sunset_glare,day_clear__rain_night")
    ap.add_argument("--holdout", default="day_clear__dense_fog,day_clear__wet_road")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="每个天气对最多取几个(调试)")
    ap.add_argument("--out-tag", default="harmonize")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    ds = project_path(args.ds)
    tr = [x.strip() for x in args.train.split(",") if x.strip()]
    ho = [x.strip() for x in args.holdout.split(",") if x.strip()]
    avail = {p.name for p in ds.iterdir() if p.is_dir()} if ds.is_dir() else set()
    missing = [t for t in tr + ho if t not in avail]
    if missing:
        raise SystemExit(f"这些天气对不在 {ds} 下:{missing}\n  可用:{sorted(avail)}")

    with runlog.run("autodrivedata.edit.train_harmonize") as rl:
        rl.input(ds, "dataset")
        rl.highlight("train_tags", tr)
        rl.highlight("holdout_tags", ho)
        for k in ("epochs", "batch", "lr", "seed"):
            rl.highlight(k, getattr(args, k))

        model, dev = train(
            ds, tr, epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed, limit=args.limit
        )
        rep = evaluate(model, dev, ds, ho, args.limit)
        m = rep["median"]
        print("\n=== ★ 留出天气对(掩膜内 LAB 统计距离,**逐对**中位)===")
        print(f"  {'留出对':<46}{'不处理':>9}{'reinhard':>10}{'trained':>9}{'n':>5}")
        for tag, r in rep["per_pair"].items():
            print(
                f"  {tag[:46]:<46}{r['composite']:>9.3f}{r['reinhard']:>10.3f}{r['trained']:>9.3f}{r['n']:>5}"
            )
        print(
            f"  {'⇒ 汇总(池化)':<46}{m['composite']:>9.3f}{m['reinhard']:>10.3f}{m['trained']:>9.3f}{m['n']:>5}"
        )
        if m["mixed_pairs"]:
            print(
                f"  ⚠ **各对 'composite' 极差 {m['composite_spread_across_pairs']}×** ⇒ 留出集是**双峰**的,"
                "汇总结论不可读 —— **读逐对那一列**"
            )
        print(
            f"  ⇒ 相对 reinhard {m['trained_vs_reinhard']:+.4f} / 相对不处理 "
            f"{m['trained_vs_composite']:+.4f};逐 patch 胜负 {m['trained_wins']} : {m['reinhard_wins']}"
        )
        for tag, r in rep["per_pair"].items():
            for k in ("composite", "reinhard", "trained"):
                rl.highlight(f"{tag[-20:]}_{k}", round(r[k], 4))
        rl.highlight("mixed_pairs", m["mixed_pairs"])
        for k in ("composite", "reinhard", "trained"):
            rl.highlight(f"median_{k}", round(m[k], 4))
        rl.highlight("trained_vs_reinhard", round(m["trained_vs_reinhard"], 4))
        rl.highlight("wins", f"{m['trained_wins']}:{m['reinhard_wins']}")
        if args.out:
            p = project_path(args.out)
            torch.save(model.state_dict(), p)
            rl.artifact(p, "weights")
        dst = project_path(str(ds / f"train_{args.out_tag}.json"))
        dst.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
        rl.artifact(dst, "report")


if __name__ == "__main__":
    main()

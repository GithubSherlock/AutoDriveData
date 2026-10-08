"""**把「注入式退化」做成一个可交付的训练数据配方**(纯值:禁 carla / torch)。

## 它把哪条负结论翻成正面交付

§1.6 / §1.10 / §1.16 的结论是「**生成的退化不能替代真值退化**」,而且是**结构性**的:
深度条件要求把物体画清楚 ⇒ 生成的雾**必然**保留可检测性。**这条路走到头了。**

⇒ **换赛道**:本仓的**注入式退化**(`degrade.py`,kind 已知、强度可**精确设定**)
是唯一能画「强度 → ΔAP」曲线的东西。把它铺成**训练集**,回答 JD 那句
「**确保仿真数据可用于感知模型训练**」。

## ★★ 验收是"留出集迁移",不是"同 root 的 ΔAP"

**单个 root 的 ΔAP 只证明退化够重,不证明训练有用。** 所以本模块**只负责造数据**,
验收在训练那侧(`docs/domain-adapt-plan.md`):

| 臂 | 是什么 |
|---|---|
| **S0** | 只用 clear 训 |
| **S1** | clear ∪ 本模块铺的标定退化档 |

两边都在**另一套**真值退化 root(GT 冻结、与训练集不同帧)上评。
**通过 ⟺** `AP(S1) − AP(S0) > 该比较的分辨率`(跨权重 σ≈0.025),且操作点召回同向,
且 S1 在 clear 域**不跌破噪声**。

## ★ 交付形态:**一档一个 root**(不是把 N 档塞进一个 root)

下游(`auto2dlabel/tools/train_kitti.py`)会**自己按 `val_ratio` 随机切** train/val。
若把同一张图的 N 档塞进**同一个** root,随机切会把**同一张底图的两个档**
一个切进 train、一个切进 val ⇒ **泄漏**(val 里有 train 的近重复)。
⇒ 每档**独立成一个 root**,manifest 里写明「**全部都是训练数据**;val 必须来自**另一个** root」。

## 自证(缺一条这个配方就不许交出去)

1. ★ **恒等塔基**:`β=0` 那一档必须与 src **逐位相同** —— 证明管线只加了退化、没动别的;
2. ★ **至少一档非恒等** —— 否则配方是个 no-op(和"没接上"长得一样);
3. `fogdepth` 的**硬前置** `training/depth/` 缺了**当场抛**(`degrade.py` 已有这条,这里再挡一次)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from autodrivedata.edit.degrade import KINDS, degrade_root
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 铺档的默认倍率(**围绕 β\\* 上下**)。几档、每档多少 —— §1.6–§1.13 只测过单点 Δ,
#: **没有**测过"训练时铺一度谱"的效果 ⇒ 这是**新实验**的旋钮,不是照抄来的常数。
DEF_LADDER = (0.0, 0.5, 1.0, 2.0)


def level_name(rank: int) -> str:
    """档的目录名。**带序号**而不是带 β 值 —— 浮点进目录名会因舍入对不上。"""
    return f"b{rank:02d}"


def mask_checksum(root: Path) -> str:
    """一个 root 的 `training/image_2` 逐文件 sha256 的合并摘要(**逐位比较用**)。

    ⚠️ 比"逐像素平均差"严:后者对"少了一张图"这种结构性差异不敏感。
    """
    h = hashlib.sha256()
    for p in sorted((root / "training" / "image_2").glob("*.png")):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def augment_root(
    src: Path, dst: Path, *, kind: str, betas: list[float], frames: list[str] | None = None
) -> dict:
    """按 `betas` 铺档,每档一个 root 落在 `dst/b{rank:02d}/`。返回 manifest 的内容。"""
    if kind not in KINDS:
        raise SystemExit(f"未知注入 {kind!r};可选 {list(KINDS)}")
    if kind == "fogdepth" and not (src / "training" / "depth").is_dir():
        raise SystemExit(
            f"{src} 没有 `training/depth/` —— `fogdepth` 是**距离相关**的,没有真值深度就算不出散射。"
            "要么换一个带深度的 root(如 `collect_ab_route --depth` 的产物),要么改用 blur/noise/rain"
        )
    if not betas:
        raise SystemExit("`--betas` 是空的 —— 没有档就没有增强集")

    levels = []
    for rank, beta in enumerate(betas):
        out = dst / level_name(rank)
        n = degrade_root(src, out, kind=kind, level=beta, frames=frames)
        levels.append({"rank": rank, "beta": float(beta), "root": str(out), "n_frames": n})

    # ★ 自证①:β=0 那一档必须与 src **逐位相同**
    zero = [x for x in levels if x["beta"] == 0.0]
    if zero:
        same = mask_checksum(src) == mask_checksum(Path(zero[0]["root"]))
        if not same:
            raise SystemExit(
                "**恒等塔基失败**:β=0 那一档与 src 不逐位相同 ⇒ 管线在退化之外还动了别的,"
                "整条链的 Δ 都无法归因"
            )
        zero[0]["bit_identical_to_src"] = True
    # ★ 自证②:至少一档非恒等
    if all(x["beta"] == 0.0 for x in levels):
        raise SystemExit("所有档都是 β=0 ⇒ 这个配方是 **no-op**(与'代码根本没接上'长得一样)")
    return {
        "kind": kind,
        "base_root": str(src),
        "levels": levels,
        "note": (
            "★ **全部都是训练数据**;val 必须来自**另一个** root —— "
            "同一底图的多个档若被随机切分,会在 val 里留下 train 的近重复(**泄漏**)。"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--src", required=True, help="底图 root(**fogdepth 必须带 `training/depth/`**)")
    ap.add_argument("--dst", required=True, help="档的落点;每档一个子目录")
    ap.add_argument("--kind", default="fogdepth", choices=KINDS)
    ap.add_argument(
        "--betas",
        default="",
        help="逗号分隔的强度。**空则用 `--beta-star` 铺 `--ladder`**",
    )
    ap.add_argument(
        "--beta-star", type=float, default=None, help="标定出来的 β\\*(来自 `edit/calibrate` 的配方)"
    )
    ap.add_argument(
        "--ladder",
        default=",".join(str(v) for v in DEF_LADDER),
        help="围绕 β\\* 的倍率(默认 0,0.5,1,2 —— **含 0 = 恒等塔基**)",
    )
    ap.add_argument("--frames", default=None, help="只处理这些帧(`0-9` / `0,5`;默认全部)")
    ap.add_argument("--manifest", default="", help="配方清单落盘(**建议给,训练侧要吃它**)")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    if args.betas:
        betas = [float(x) for x in args.betas.split(",") if x.strip()]
    elif args.beta_star is not None:
        betas = [round(args.beta_star * float(m), 6) for m in args.ladder.split(",") if m.strip()]
    else:
        raise SystemExit("要么给 `--betas`,要么给 `--beta-star`(配 `--ladder`)")

    frames = _parse_frames(args.frames) if args.frames else None
    with runlog.run("autodrivedata.edit.augment") as rl:
        src, dst = project_path(args.src), project_path(args.dst)
        rl.input(str(src), "src")
        rl.highlight("kind", args.kind)
        rl.highlight("betas", betas)
        rep = augment_root(src, dst, kind=args.kind, betas=betas, frames=frames)

        print(f"\n=== 注入式退化增强配方(`{args.kind}`)===")
        print(f"  底图 {src}")
        for x in rep["levels"]:
            tag = "  ★ 恒等塔基(与 src 逐位相同)" if x.get("bit_identical_to_src") else ""
            print(f"  {level_name(x['rank'])}  β={x['beta']:<9g} {x['n_frames']:>4} 帧 → {x['root']}{tag}")
        print(f"  ⇒ {rep['note']}")
        for x in rep["levels"]:
            rl.highlight(f"{level_name(x['rank'])}_beta", x["beta"])
        if args.manifest:
            m = project_path(args.manifest)
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
            rl.artifact(m, "manifest")
            print(f"[manifest] {m.resolve()}")


def _parse_frames(spec: str) -> list[str]:
    out: list[str] = []
    for tok in spec.split(","):
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(f"{i:06d}" for i in range(int(a), int(b) + 1))
        elif tok:
            out.append(f"{int(tok):06d}")
    return out


__all__ = ["augment_root", "level_name", "mask_checksum", "DEF_LADDER"]


if __name__ == "__main__":
    main()

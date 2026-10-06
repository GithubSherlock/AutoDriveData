"""按实例 id 编辑盘上的一份高斯 —— 目前只有「删」,接口留给后面的「搬/替换」。

## 位置

`attribute_instances` 给每个 Gaussian 一个实例 id ⇒ 本模块按 id 把**指定的那些**拿掉。
它是「改一份高斯 → 渲染它 → 用真值判据量」三步里的**第二步**:

```
attribute_instances → edit_gs → render_gs / eval_edit
       (哪些是它)      (拿掉它)        (拿掉之后离真值多远)
```

## ★ 两条纪律(都对应"删错了但看着像成功")

**① `-1`(未归属)一个都不许动。** 与 `static_eval` 的「`n=0` 是不判」同一条:
把"没测到"当成"没物体",删掉的会是**整片背景** —— 而渲染出来只是"场景没了",
不是报错。故 `--remove -1` 直接拒。

**② 删掉 0 个必须报错。** 请求的 id 在 attr 里不存在(拼错、或那份 attr 是别的 capture 的),
静默成功会让 `eval_edit` 量出"C1 编辑前后完全一样",而那与"这个物体本来就删不掉"长得一样。
⇒ `remove_ids` 对每个请求的 id 分别计数,**任何一个为 0 就抛**。

## 未归属的比例要一起报

`-1` 占多少必须打出来:归属率低的时候(实测 `legacy` 口径只有 24%、中位命中帧数 0),
"删干净了"这件事本身就不成立 —— 而只看"删了 N 个"看不出来。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from autodrivedata.gs import cuda_env  # noqa: F401  —— 与 render_gs 同一纪律(见其头注)
from autodrivedata.gs.render_gs import GaussianSet, load_set, save_set
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 未归属的哨兵值(与 `attribute_instances` 同一口径)。
UNASSIGNED = -1


def read_attr(path: Path) -> np.ndarray:
    """读 `attr*.npy`(每个 Gaussian 一个实例 id)。"""
    a = np.load(path)
    if a.ndim != 1:
        raise SystemExit(f"{path} 的形状是 {a.shape} —— 归属结果是**每高斯一个 id** 的一维数组")
    return a.astype(np.int64)


def id_histogram(attr: np.ndarray) -> dict[int, int]:
    """`id → 高斯数`。**报告用**,也是"该删哪个 id"的唯一可靠来源。"""
    ids, cnt = np.unique(attr, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, cnt, strict=True)}


def check_ids(ids: list[int]) -> None:
    """请求的 id 合不合法 —— **唯一的落点**,`main` 与 `remove_ids` 都调它。

    `main` 在**读盘之前**调:这个错误与数据无关,不该等读完几百 MB 才发现。
    """
    if UNASSIGNED in ids:
        raise SystemExit(
            f"不许删 `{UNASSIGNED}`(未归属)—— 那会把**没测到**当成**没有物体**,"
            "整片背景一起消失,而那不是报错,是「场景没了」"
        )


class RemoveReport(TypedDict):
    """`remove_ids` 的报告。写成 TypedDict 而不是裸 `dict` —— 下面的 `.items()` / 百分比
    都要按字段取值,裸 `dict` 会让静态检查看不出形状(而这里正是最容易写错字段名的地方)。"""

    removed_ids: dict[int, int]
    n_before: int
    n_after: int
    n_removed: int
    unassigned_share: float


def remove_ids(gs: GaussianSet, attr: np.ndarray, ids: list[int]) -> tuple[GaussianSet, RemoveReport]:
    """把 `attr ∈ ids` 的那些高斯拿掉 → `(新的 GaussianSet, 报告)`。**纯函数**。

    ⚠️ 两条纪律见模块头注:`-1` 不许进 `ids`;任何请求的 id 命中 0 个就抛。
    """
    if gs.n != attr.shape[0]:
        raise SystemExit(f"高斯数 {gs.n} 与归属数组 {attr.shape[0]} 对不上 —— 这不是同一份数据")
    check_ids(ids)
    hit = {i: int((attr == i).sum()) for i in ids}
    empty = [i for i, c in hit.items() if c == 0]
    if empty:
        raise SystemExit(
            f"这些 id 在归属结果里一个高斯都没有:{empty}(可用 id 见 `--list`)。"
            "**静默删 0 个**会让下游把「编辑前后一模一样」读成「这个物体删不掉」"
        )
    keep = ~np.isin(attr, ids)
    out = GaussianSet(
        means=gs.means[keep],
        rots=gs.rots[keep],
        scales_lin=gs.scales_lin[keep],
        col=gs.col[keep],
        opac=gs.opac[keep],
    )
    report = {
        "removed_ids": hit,
        "n_before": gs.n,
        "n_after": out.n,
        "n_removed": gs.n - out.n,
        "unassigned_share": float((attr == UNASSIGNED).mean()),
    }
    return out, report


def box_from_prop_json(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """`prop.json` → `(盒中心 世界系, 半尺寸, yaw 弧度)`。

    ⚠️ `PropBox.location` 是**actor 原点**,不是盒中心 —— 盒中心 = 原点 +
    `box_offset`(在 actor 局部系里,见 `gt/props.PropBox`)。本采集摆位 yaw 恒 0,
    所以局部系 = 世界系方向,直接相加。
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    prop = d.get("prop")
    if not isinstance(prop, dict):
        raise SystemExit(f"{path} 里没有 `prop` —— 这份 capture 不是 `--props` 采的")
    loc = np.asarray(prop["location"], dtype=np.float64)
    offs = np.asarray(prop.get("box_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
    size = np.asarray(prop["size"], dtype=np.float64)
    yaw = float(np.radians(float(prop.get("yaw_deg", 0.0))))
    c, sn = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
    return loc + rz @ offs, size / 2.0, yaw


def remove_in_box(
    gs: GaussianSet,
    center: np.ndarray,
    half: np.ndarray,
    yaw: float,
    *,
    margin: float = 0.0,
) -> tuple[GaussianSet, int]:
    """按**世界系 3D 框**删 → `(新 GaussianSet, 删了几个)`。**纯函数**。

    ★ 为什么需要它(而不是按实例 id 删):**CARLA 的实例 / 深度 / 语义三个 pass
    都不渲染 `static.prop.*`** —— 2026-10-04 实测 A/B 在这三路上**逐像素完全相同**
    (而 RGB 上差 57,664 px)⇒ 那个道具**没有实例 id**,`attribute_instances` 指不到它。

    凭什么它是对的:`prop.json` 里的位置与尺寸是本仓**摆位时自证过**的
    (读回残差 ≤ `collect_3dgs.PLACE_TOL_M`),而盒只有 1.3×1.1×1.9 m、摆在环心 ——
    它**不含背景**(背景在盒外好几米)。盒内的高斯只可能是在解释那个道具。

    ⚠️ 删 0 个**也要报错**(同 `remove_ids`):静默成功会让下游把"编辑前后一样"读成结论。
    """
    c, sn = np.cos(-yaw), np.sin(-yaw)
    rzi = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
    local = (np.asarray(gs.means, dtype=np.float64)[:, :3] - center) @ rzi.T
    inside = np.all(np.abs(local) <= (half + margin), axis=1)
    n = int(inside.sum())
    if n == 0:
        raise SystemExit(
            f"框里一个高斯都没有(中心 {np.round(center, 2)}, 半尺寸 {np.round(half, 2)})—— "
            "要么框给错了,要么这份高斯不是这个场景的。**静默删 0 个**会让下游把"
            "「编辑前后一模一样」读成结论"
        )
    keep = ~inside
    return (
        GaussianSet(
            means=gs.means[keep],
            rots=gs.rots[keep],
            scales_lin=gs.scales_lin[keep],
            col=gs.col[keep],
            opac=gs.opac[keep],
        ),
        n,
    )


def ids_from_prop_json(path: Path) -> list[int]:
    """从 `capture/prop.json` 读那个道具的 `instance_id` —— **编辑对象的权威来源**。

    比"人肉挑一个 id"可靠:摆位那一步已经把它记下来了(见 `collect_3dgs.spawn_prop_at`)。
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    prop = d.get("prop")
    if not isinstance(prop, dict) or "instance_id" not in prop:
        raise SystemExit(f"{path} 里没有 `prop.instance_id` —— 这份 capture 不是 `--props` 采的")
    return [int(prop["instance_id"])]


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--gs-dir", default="outputs/3dgs")
    ap.add_argument("--tag", default="", help="源高斯(读 `{means,rots,scales,col,opac}{_tag}.npy`)")
    ap.add_argument(
        "--attr",
        default="",
        help="`attribute_instances` 产出的 attr*.npy(按实例 id 删时才要)",
    )
    ap.add_argument(
        "--bbox-from-prop",
        default="",
        help="★ 按 **`prop.json` 的世界系 3D 框**删。**本链路上这条才是对的** —— "
        "CARLA 的实例/深度/语义 pass 不渲染 `static.prop.*`,那个道具没有实例 id",
    )
    ap.add_argument("--bbox-margin", type=float, default=0.0, help="框外扩(m)")
    ap.add_argument("--remove", default="", help="要删的 id,逗号分隔(与 --prop-json 二选一)")
    ap.add_argument(
        "--prop-json",
        default="",
        help="`capture/prop.json` —— 从中读 `instance_id`,比人肉挑 id 可靠",
    )
    ap.add_argument(
        "--out-tag",
        default="",
        help="编辑后那份的 tag(**别覆盖源那份**)。`--list` 不需要它",
    )
    ap.add_argument("--list", action="store_true", help="只打印 id 直方图,不动数据")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    gs_dir = project_path(args.gs_dir)
    if not args.out_tag and not args.list:
        raise SystemExit("要给 `--out-tag`(编辑后的产物落哪儿;`--list` 才不需要)")
    if not args.attr and not args.bbox_from_prop:
        raise SystemExit("要么给 `--attr`(按实例 id),要么给 `--bbox-from-prop`(按 3D 框)")
    # ★ 按 id 那条路的**入参先解析并校验** —— 它不依赖盘上的高斯,而参数错不该
    #   等读完几百 MB 才报(第一版把校验放在了 load 之后,被回归钉抓回来过)。
    ids: list[int] = []
    if not args.bbox_from_prop:
        ids = (
            ids_from_prop_json(project_path(args.prop_json))
            if args.prop_json
            else [int(x) for x in args.remove.split(",") if x.strip()]
        )
        if not ids:
            raise SystemExit("按 id 删要给 `--remove` 或 `--prop-json`")
        check_ids(ids)

    if args.list:
        if not args.attr:
            raise SystemExit("`--list` 要看的是归属直方图 ⇒ 得给 `--attr`")
        attr = read_attr(project_path(args.attr))
        h = id_histogram(attr)
        total = int(attr.shape[0])
        for i, c in sorted(h.items(), key=lambda kv: -kv[1]):
            tag = "(未归属)" if i == UNASSIGNED else ""
            print(f"  id={i:<8} {c:>8} 个 ({c / total * 100:5.2f}%) {tag}")
        print(f"  合计 {total} 个高斯")
        return

    with runlog.run("autodrivedata.gs.edit_gs") as rl:
        rl.input(args.attr, "attr")
        print(f"[arch] {cuda_env.ARCH_NOTE}")
        print(f"[cuda] {cuda_env.CUDA_NOTE}")
        rl.highlight("cuda_home", cuda_env.home())

        gs = load_set(gs_dir, args.tag)
        if args.bbox_from_prop:
            # ★ **读盘之后**才解析框(它要 prop.json,但"框里 0 个"只有在拿到高斯后才知道)
            center, half, yaw = box_from_prop_json(project_path(args.bbox_from_prop))
            print(
                f"[edit] 按 3D 框删:中心 {np.round(center, 3)} 半尺寸 {np.round(half, 3)} yaw {np.degrees(yaw):.1f}°"
            )
            edited, n_removed = remove_in_box(gs, center, half, yaw, margin=args.bbox_margin)
            print(f"[edit] {gs.n} → {edited.n}(−{n_removed})")
            rl.highlight("bbox_center", [round(float(x), 3) for x in center])
            rl.highlight("n_removed", n_removed)
            rl.highlight("method", "bbox")
        else:
            edited, rep = remove_ids(gs, read_attr(project_path(args.attr)), ids)
            for i, c in rep["removed_ids"].items():
                print(f"[edit] 删 id={i}:{c} 个高斯")
            print(
                f"[edit] {rep['n_before']} → {rep['n_after']}(−{rep['n_removed']});"
                f"未归属占 {rep['unassigned_share'] * 100:.1f}%(**没动**) "
            )
            rl.highlight("n_before", rep["n_before"])
            rl.highlight("unassigned_share", round(rep["unassigned_share"], 4))
            rl.highlight("method", "id")
        written = save_set(edited, gs_dir, args.out_tag)
        print(f"[edit] 落盘 tag={args.out_tag}:{len(written)} 个文件")
        rl.highlight("out_tag", args.out_tag)
        for p in written:
            rl.artifact(p, "gs-set")


if __name__ == "__main__":
    main()

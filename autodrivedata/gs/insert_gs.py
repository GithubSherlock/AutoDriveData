"""**3D 场景编辑的"插入"那一半**:把一个物体的高斯**搬进**另一个场景。

## 为什么这条有**真值靶**(而且免费)

C1 做的是「**删**」。JD 里"3D 场景编辑"真正要的是**插入 / 替换**。
而本项目的采集恰好给了一个**精确的**靶:

| 角色 | 是什么 |
|---|---|
| **底图** | `ab3_B` —— 同一场景、**没有道具** |
| **搬进来的东西** | `ab3_A` 上属于道具的那些高斯(世界坐标**已经是对的**) |
| **真值** | `ab3_A` 的**实拍图** —— 同一场景、同位姿、**有道具** |

⇒ `B + prop` 应当**收敛到 A 本身**。而"收敛到多少"有现成的天花板:
A 自己的重建质量(`psnr(render_A, img_A)`)。

★ **两个捕捉共用同一个世界系与同一串位姿**(A/B 硬门槛保证),所以**搬运 = 直接拼接**,
不需要任何配准 —— 这也是本项目做 3D 编辑的独有便利。

## 判据

用**现成的** `probe_hole_render`(`--capture-a` 与 `--capture-b` **都给 A**,
`--a-tag`/`--b-tag` 也给 `ab3_A`,真值 = A 的实拍):

| 读数 | 读作 |
|---|---|
| `edited_vs_truth` | 插完之后离 A 的实拍多远 |
| `ceiling_vs_truth` | A 自己的重建质量(**上限**) |
| `edited_vs_with_prop` | 插完 vs A 的渲染 —— 应当**很近**(两边都有道具) |

## 控制臂(house rule)

- **错位置插入**(`--shift`):把同一批高斯挪开 ⇒ **必须变差**。
  不然"插进去就行"这句话没有判别力(随便加一堆高斯也可能让某些指标动)。
"""

from __future__ import annotations

import argparse

import numpy as np

from autodrivedata.gs.render_gs import GaussianSet, load_set, save_set
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path


def select_ids(attr: np.ndarray, ids: list[int]) -> np.ndarray:
    """按实例 id 选出一批高斯的**布尔掩膜**。

    ⚠️ `-1`(未归属)不许被请求 —— 它不是"另一个 id",是**没测到**(与 `edit_gs` 同一条)。
    """
    if not ids:
        raise SystemExit("`--add-ids` 是空的 —— 没有要搬的东西")
    if any(i < 0 for i in ids):
        raise SystemExit(f"请求了未归属(-1)…… ids={ids}:`-1` 是**没测到**,不是一个实例")
    mask = np.isin(attr, np.asarray(ids, dtype=attr.dtype))
    if not mask.any():
        raise SystemExit(f"这些 id 一个高斯都没命中:{ids} —— 查归属文件与 tag 是不是同一次训练")
    return mask


def translate(gs: GaussianSet, shift: tuple[float, float, float]) -> GaussianSet:
    """把一份高斯整体平移(**只给控制臂用**:错位置插入)。

    ⚠️ 只动 `means` —— 高斯是**各向异性**的,平移不该改朝向/尺度/颜色。
    """
    d = np.asarray(shift, dtype=gs.means.dtype)
    return GaussianSet(means=gs.means + d, rots=gs.rots, scales_lin=gs.scales_lin, col=gs.col, opac=gs.opac)


def merge_sets(base: GaussianSet, add: GaussianSet, keep: np.ndarray) -> GaussianSet:
    """把 `add` 里 `keep` 选中的那部分**拼到** `base` 上。

    ⚠️ 五个字段**必须一起拼** —— 只拼 `means` 会让新来的高斯**继承前一批的朝向/颜色**,
    而那种错在渲染上只表现为"东西是糊的",看不出根因。
    """
    if keep.shape[0] != add.n:
        raise SystemExit(f"掩膜长 {keep.shape[0]} 与高斯数 {add.n} 对不上 —— 归属文件是不是另一份的?")
    if not keep.any():
        raise SystemExit("掩膜全是 False —— 拼接会是个 no-op,而那与'代码没接上'长得一样")
    cat = lambda a, b: np.concatenate([a, b[keep]], axis=0)  # noqa: E731
    return GaussianSet(
        means=cat(base.means, add.means),
        rots=cat(base.rots, add.rots),
        scales_lin=cat(base.scales_lin, add.scales_lin),
        col=cat(base.col, add.col),
        opac=cat(base.opac, add.opac),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--gs-dir", default="outputs/3dgs")
    ap.add_argument("--base-tag", required=True, help="底图(被插进去的那个场景)")
    ap.add_argument("--add-tag", required=True, help="被搬的物体所在的那份高斯")
    ap.add_argument("--attr", default="", help="`attribute_instances` 的产物(`add-tag` 的归属)")
    ap.add_argument("--add-ids", default="", help="要搬的 id,逗号分隔")
    ap.add_argument("--shift", default="0,0,0", help="整体平移(**只给控制臂**用)")
    ap.add_argument("--out-tag", default="", help="产物 tag(别覆盖源那两份)")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    if not args.out_tag:
        raise SystemExit("必须给 `--out-tag`(别覆盖源那两份)")
    ids = [int(x) for x in args.add_ids.split(",") if x.strip()]
    _sh = [float(x) for x in args.shift.split(",")]
    if len(_sh) != 3:
        raise SystemExit(f"`--shift` 要三个数(x,y,z),收到 {args.shift!r}")
    shift: tuple[float, float, float] = (_sh[0], _sh[1], _sh[2])

    with runlog.run("autodrivedata.gs.insert_gs") as rl:
        gs_dir = project_path(args.gs_dir)
        rl.highlight("base_tag", args.base_tag)
        rl.highlight("add_tag", args.add_tag)
        rl.highlight("add_ids", ids)
        rl.highlight("shift", shift)
        base = load_set(gs_dir, args.base_tag)
        add = load_set(gs_dir, args.add_tag)
        attr = np.load(project_path(args.attr))
        keep = select_ids(attr, ids)
        if shift != (0.0, 0.0, 0.0):
            add = translate(add, shift)
            print(f"[insert] ⚠️ **控制臂**:整批平移 {shift}(应当变差)")
        out = merge_sets(base, add, keep)
        paths = save_set(out, gs_dir, args.out_tag)
        print(f"[insert] {args.base_tag}({base.n})+ {args.add_tag} 的 {int(keep.sum())} 个 → {out.n} 个")
        print(f"[insert] 落盘 tag={args.out_tag}:{len(paths)} 个文件")
        rl.highlight("n_added", int(keep.sum()))
        rl.highlight("n_out", out.n)
        for p in paths:
            rl.artifact(p, "gaussians")


if __name__ == "__main__":
    main()

"""**按类尺寸先验**给 LiDAR 簇补全长 —— 修「簇只覆盖车的可见部分」。

## 它修的是哪一条

Plan4 §P-V18 五:融合消融建立在**低召回**上 —— 350 个簇预测里只有 **10** 个 IoU > 0.5。
簇是 AABB,**只覆盖 LiDAR**看得见的那一面**;** GT 是**整车 4.5 m**。

## ★★ 两条必须先立的纪律(复核指出的)

1. **先验不许与评测同源**(train/test 泄漏)。先验要从**别的 root**上量,
   而 `label_2` 里本来就有真值尺寸 —— 在同一个 root 上量再评,那是**把答案抄进模型**。
   ⇒ 先验文件里**记下它的来源 root**,`eval_fusion` 发现与评测 root 相同时**响亮报警**。
2. **要有"错先验"的对照**。把类别→尺寸打乱,读数**必须变差** ——
   否则分不出"这个先验没用"与"代码根本没接上"。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from autodrivedata.perception.fusion import Cluster
from autodrivedata.utils.paths import project_path

#: `label_2` 的列:类别 + `(h, w, l)` 在 8/9/10 列。
_CLS, _H, _W, _L = 0, 8, 9, 10


def read_dims(root: Path, frames: list[str], cls: str = "Car") -> list[tuple[float, float, float]]:
    """某个 root 的 `label_2` 里某一类的 `(l, w, h)`。**只读真值,不出模型。**"""
    out = []
    for fid in frames:
        p = root / "training" / "label_2" / f"{fid}.txt"
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            f = line.split()
            if len(f) >= 11 and f[_CLS] == cls:
                out.append((float(f[_L]), float(f[_W]), float(f[_H])))
    return out


def fit_from_root(root: Path, frames: list[str], *, cls=("Car",)) -> dict:
    """逐类取**中位**尺寸。中位而不是均值 —— 少数卡车会把均值拉走。

    ⚠️ 返回值里带 `source_root`。**这是防泄漏的那把锁**:`apply_prior` 不看它,
    但 `eval_fusion` 会拿它与评测 root 比,相同就报警。
    """
    dims: dict[str, list[float]] = {}
    for c in cls:
        rows = read_dims(root, frames, c)
        if len(rows) < 3:
            raise SystemExit(f"{root} 的 {c} 只有 {len(rows)} 条 GT —— 先验样本不够(至少 3 条)")
        dims[c] = [float(v) for v in np.median(np.array(rows), axis=0)]
    return {"source_root": str(root), "dims": dims, "n_source_frames": len(frames)}


def load_prior(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def anchor_cluster(cl: Cluster, dims: tuple[float, float, float]) -> Cluster:
    """把簇换成"**按先验全长 + 锚在可见面上**"的盒子。

    ★ **锚哪一面是有讲究的**:LiDAR 看到的是车的**近面**(朝 ego 那侧)与**底面**,
    看不到远面。⇒ 保持**近面与底面不动**、只把**远面往外推**:
    这才是"补上没看见的那一截",而不是"把整辆车平移"。

    ⚠️ `Cluster` 是 **velodyne 系**:x 前向(长轴)、y 侧向、z 上。近面 = `x − half_x`。
    """
    length, width, height = dims
    near_x = cl.center[0] - cl.half[0]
    bottom_z = cl.center[2] - cl.half[2]
    return replace(
        cl,
        center=(near_x + length / 2.0, cl.center[1], bottom_z + height / 2.0),
        half=(length / 2.0, width / 2.0, height / 2.0),
    )


def shuffle_prior(prior: dict, *, seed: int = 0) -> dict:
    """**对照**:把先验**故意弄错** —— 读数必须"变"。

    读数是"变好"还是"变差"都不重要,**重要的是它必须**"变" —— 纹丝不动说明先验没进链路
    (与 `probe_bev_depth` 的对照②同一条纪律)。

    ## ⚠️ 两个变体(实测踩到第一个不够用)

    - **多个类** ⇒ 打乱 `类 → 尺寸` 的配对;
    - ★ **只有一个类** ⇒ 打乱配对是**恒等**(实测:先验里只有 `Car`,打乱后一模一样)
      ⇒ 退化成**打乱分量**(`l/w/h` 轮换) —— 一辆 4.5 m **宽**的车显然是错的,
      而它是**单类**情形下唯一还能"故意弄错"的自由度。

    ⚠️ 本项目的 LiDAR 档**只有一类几何**(类由相机给,见 §P-V18)⇒ **走的是第二个变体**。
    """
    dims = {c: list(v) for c, v in prior["dims"].items()}
    classes = sorted(dims)
    if len(classes) > 1:
        vals = [dims[c] for c in classes]
        order = np.random.default_rng(seed).permutation(len(classes))
        if np.array_equal(order, np.arange(len(classes))):
            order = np.roll(order, 1)  # 别让"打乱"恰好是恒等
        dims = {c: vals[int(i)] for c, i in zip(classes, order, strict=True)}
        how = "class-pairing"
    else:
        c = classes[0]
        dims[c] = [dims[c][1], dims[c][0], dims[c][2]]  # l ↔ w
        how = "component-swap"
    return {**prior, "dims": dims, "shuffled": True, "shuffle_how": how}


def prior_path(out: Path) -> Path:
    return project_path(out)

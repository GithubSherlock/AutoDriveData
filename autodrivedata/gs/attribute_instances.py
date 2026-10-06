"""Gaussian → **实例归属**:把训练出来的每个 Gaussian 认到一个 CARLA 实例 id 上。

## 为什么这件事在本项目是「查出来的」而不是「猜出来的」

3DGS 的每个 Gaussian 只有 `(位置, 尺度, 旋转, 不透明度, 颜色)` —— **没有"属于哪个物体"**。
真实数据上这一步(Gaussian Grouping / 3D 一致性 lifting)要用 SAM + CLIP + 多视角投票去**猜**,
且会错。本项目有 CARLA 的**逐像素实例 id**(`collect_3dgs --inst` 落 `inst/p{p}/{i:05d}.png`,
16 位灰度、像素值 = actor id,口径见 [`perception/inst_tags.py`](../perception/inst_tags.py))
⇒ 把每个 Gaussian 投影回每一帧、读它落点的 id、投票 —— **归属是读出来的**。

## 算法(每个 Gaussian 一个标签)

1. 读 `means{tag}.npy` + `capture/poses_{p}.json`(相机位姿 = **CARLA 真值**);
   内参走 `calib.core.CameraIntrinsics(1242, 375, 90)` —— 与
   `train_3dgs_mini._load_poses_and_cams` **同一口径的 fx/主点**(那边下采样后是
   `fx/D` 与 `cx/D + 0.5`;这里读的是**原生分辨率**的 `inst/*.png`,故直接用原生 K)。
2. 每帧:把 Gaussian 中心投影进该帧(`calib.core.world_to_img` 的口径;
   向量化实现在 [`project_points`],单测与它逐点对拍);**画幅内且相机前方** ⇒
   读 `inst/` 图该像素的 id,**记一票**。
3. 汇总:取**众数 id**;命中帧数 < `min_samples`、或众数占比 < `min_share`、
   或众数是 0(背景)⇒ 判**未归属**(-1)。
   ⚠️ 与 `perception/static_eval` 的「n=0 是不判」同一条纪律:**不许把"没测到"写成"没物体"**。
4. 落盘 `outputs/3dgs/attr{tag}.npy`(每 Gaussian 一个 int64 id,-1 = 未归属)。

## ★ 判据:跨视角标签一致性(**不是准确率**)

把帧集**按视角一分为二**(偶数帧 / 奇数帧),两半**各自独立投票**,比两者的一致率:

| 量 | 问的是 |
|---|---|
| `agree` | 两半独立投出来的标签一致的比例 —— **归属稳不稳定** |
| `chance` | 平凡基线:按标签频率的**碰撞概率** `Σ p_i²`(乱贴标签能达到的一致率) |
| `shuffled` | 对抗对照:把 B 半的标签**随机置换**后的一致率(必须塌到 ≈ `chance`) |
| `kappa` | **裁决量**:`(agree − max(chance, shuffled)) / (1 − max(...))`。"可达到的余量吃掉了多少" |

**裁决**:① `agree ≥ MIN_AGREE`;② `κ ≥ MIN_KAPPA`。

⚠️ **②为什么不是"倍率"**(2026-10-04 实测的判据缺陷):实例 id 的分布**极度倾斜** ——
一个占大头的网格覆盖大部分像素 ⇒ 平凡基线本身就 **0.844**,而一致率上限是 1.0。
`3 × 0.853 = 2.56 > 1.0` ⇒ **任何方法都过不了**,判据永久红、然后被人调阈值调绿。
κ 是同一件事的正确算式(它减的是基线、除的是余量)。倍率仍报出来看绝对水平。
这与 §A.1「16 dB 必须跟复制最近邻比」、`static_eval` 的 `real ≥ 5×对照` 是同一条纪律 ——
**先减掉平凡基线,再谈好坏**;区别只是这里的基线高到让倍率式子无解。

⚠️ **不可用真值的边界**:实例 id 本身是 CARLA 真值,但「哪个 Gaussian 属于哪个实例」
**没有直接真值**(Gaussian 是训练出来的,不是采集来的)⇒ 判据只能是**一致性 + 对抗对照**,
**不是准确率**。任何把这个数读成"归属精度 0.93"的说法都是错的。要准确率只有一条路:
拿**合成**夹具(已知布局 + 已知标签,见 `--self-test`)——那里的"一致率"才是"正确率"。

## 第三态(未判)

两半**都**给出归属的 Gaussian 数为 0(或样本太少)时判**未判**(`None`)——
既不是"过"也不是"不过"。与 `frame_sync` / `static_eval` 同一条纪律。

## ⚠️ 两条边界(实测踩到才写的)

1. **κ 只在"被归属的"子集上算** —— 所以**归属率必须先看**。2026-10-04 实测:同一份 capture 的
   两个模型,`carla` 口径归属 **99.9%**、κ 0.928;`legacy` 口径(相机系错的)只归属 **24.0%**
   (中位命中帧数 **0** —— 大部分 Gaussian 根本投不进任何一帧的画幅),而**被归属的那一小撮**
   κ 照样 **0.917**。⇒ **别只看 κ**:一个只认下 24%(且那 24% 恰好都落在最大那块网格上)的
   方法也能拿高分。判据要连 `n_labeled / N` 一起读。
2. **打乱实例图不一定压低一致率,它压低的是 κ**。2026-10-04 实测:打乱后一致率 **0.9989**
   (比真值 0.9893 还高!因为最大那块网格本来就占 99% 像素),但 κ 从 **0.928 掉到 0.119**。
   ⇒ 反向自证的读数**只能是 κ**,不能是一致率。这也正是裁决式子里必须有 `1 − control` 的原因。

## 与训练口径的关系(★ 2026-10-04 实测,先说在前面)

本模块**必须**用 CARLA 真位姿投影(那些 `inst/*.png` 就是 CARLA 用真位姿渲的)。
而 `train_3dgs_mini._load_poses_and_cams` 的 `rw = ry @ rx` **不是** CARLA 的相机系
(实测:拿 GT 深度按它反投影,点云沿**竖直**方向拉开 69.7 m、水平只有 16.2 m;
用真位姿则是水平的 68.6 m × 竖直 11.2 m —— 城市街道的形状)。
⇒ **用旧口径训出来的 `means*.npy` 不在 CARLA 世界系**,喂进本模块只会得到"归属是噪声"。
`train_3dgs_mini` 因此加了 `--cam-convention {carla,legacy}`
(**2026-10-04 起默认 = `carla`**;`legacy` 只为复现归档的 `means*.npy` 而留)。
⚠️ **喂进来的 `means*.npy` 必须是 `carla` 口径训出来的** —— 归档那份是 `legacy` 的,
直接喂只会得到"归属是噪声"(实测归属率 24.0%、中位命中帧数 **0**;carla 口径是 99.9%)。
见 [docs/edit-3dgs-plan.md](../../docs/edit-3dgs-plan.md) §4 阶段 B。

用法:
  python -m autodrivedata.gs.attribute_instances --capture outputs/3dgs_sync/capture \\
      --means outputs/3dgs/means_syncB.npy --out outputs/3dgs/attr_syncB.npy
  python -m autodrivedata.gs.attribute_instances --self-test          # 合成夹具,只验尺子
  python -m autodrivedata.gs.attribute_instances ... --shuffle-inst   # 反向自证(必须塌)
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autodrivedata.calib.core import CameraIntrinsics
from autodrivedata.perception.inst_tags import decode_instance_png
from autodrivedata.perception.sem_tags import TAG_NAMES, decode_tag_png
from autodrivedata.utils import geometry as g
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 原生分辨率内参(与 `collect_3dgs` 的 `CAM_ATTRS` 一致:`1242×375` / `fov=90`)。
#: 主点按 CARLA 光栅的 **corner** 约定 `(w−1)/2`,与 `world_to_img` 同口径。
SRC_W, SRC_H, SRC_FOV = 1242, 375, 90.0

#: 一个 Gaussian 至少要**这么多帧**命中过一个非零 id 才参与归属。
MIN_SAMPLES = 3
#: 众数占比下限。
MIN_SHARE = 0.5
#: 一致率裁决:① 绝对下限;② **超额一致率**(Cohen's κ 口径)的下限。
MIN_AGREE = 0.80
MIN_KAPPA = 0.50
#: ③ 旧口径(倍率)保留**只为报数**:`agree ≥ MARGIN × control`。
#:
#: ⚠️ **它在这里是不可满足的,别再拿它当裁决**(2026-10-04 实测)。实例 id 的分布**极度倾斜** ——
#: 一个占大头的网格覆盖了大部分像素 ⇒ 平凡基线 `chance = Σp²` 就 **0.844**,而一致率上限是
#: **1.0**。`3 × 0.853 = 2.56 > 1.0` ⇒ **任何方法都不可能过这条**,判据会永久红,然后再被人
#: 把阈值调绿 —— 正是本项目反复吃亏的那种"把不知道伪装成结论"。
#: 处置:改用 **κ = (agree − control)/(1 − control)**,即"把可达到的余量吃掉了多少"。
#: 倍率仍然报出来(它衡量的是"一致率的绝对水平有多高")。
MARGIN = 3.0
#: 分块大小(逐个 Gaussian 分块累计,避免 270 帧 × N 个命中对一次性进内存)。
CHUNK = 200_000
#: 投票键的进位:`key = local_idx * STRIDE + id`(id ≤ 65535 ⇒ 17 位足够)。
STRIDE = 1 << 17


@dataclass(frozen=True)
class FrameRef:
    """一帧的相机位姿(**CARLA 真值**,角度存弧度)。"""

    pitch: float
    index: int
    location: tuple[float, float, float]
    rotation_rad: tuple[float, float, float]  # (pitch, yaw, roll)

    @property
    def key(self) -> str:
        return f"p{int(self.pitch)}_{self.index:05d}"


def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(width=SRC_W, height=SRC_H, fov_h_deg=SRC_FOV)


def load_frames(capture: Path) -> list[FrameRef]:
    """`capture/` → 逐帧相机位姿(按 `pitches.json` 的顺序,与训练侧同一枚举口径)。"""
    pj = capture / "pitches.json"
    if not pj.is_file():
        raise SystemExit(f"{pj} 不存在 —— 这不是一份 3dgs capture")
    out: list[FrameRef] = []
    for p in json.loads(pj.read_text(encoding="utf-8")):
        pp = float(p)
        pf = capture / f"poses_{int(pp)}.json"
        if not pf.is_file():
            raise SystemExit(f"{pf} 不存在 —— pitches.json 与 poses_*.json 对不上")
        for pose in json.loads(pf.read_text(encoding="utf-8")):
            out.append(
                FrameRef(
                    pitch=pp,
                    index=int(pose["i"]),
                    location=(float(pose["x"]), float(pose["y"]), float(pose["z"])),
                    rotation_rad=(
                        float(np.radians(pose["pitch"])),
                        float(np.radians(pose["yaw"])),
                        float(np.radians(pose.get("roll", 0.0))),
                    ),
                )
            )
    return out


def project_points(
    pts: np.ndarray, fr: FrameRef, k: CameraIntrinsics
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """世界点 `(N,3)` → 像素**索引** `(u, v)` + 有效掩膜(`world_to_img` 的向量化形式)。

    **必须与 `calib.core.world_to_img` 逐点等价**(同一个 `utils.geometry.world_to_cam`
    + 同一个 `CameraIntrinsics`),不然"投影进哪一帧的哪个像素"就悄悄换了一套口径,
    而症状只是"归属率低一点"。单测里有逐点对拍。

    ⚠️ 返回的是**像素索引**(`round` 之后),不是连续坐标:`world_to_img` 给的连续坐标
    按 CARLA 光栅的 corner 约定 `index == 连续坐标`,故这里必须 `round` 而不是 `floor`。
    """
    c = g.world_to_cam(np.asarray(pts, dtype=np.float64), fr.location, fr.rotation_rad)
    z = c[:, 2]
    ok = z > 0.5
    safe = np.where(ok, z, 1.0)
    uf = k.fx * (c[:, 0] / safe) + k.cx
    vf = k.fy * (c[:, 1] / safe) + k.cy
    ui = np.round(uf).astype(np.int64)
    vi = np.round(vf).astype(np.int64)
    ok = ok & (ui >= 0) & (ui < k.width) & (vi >= 0) & (vi < k.height)
    return ui, vi, ok


def inst_loader(
    capture: Path, *, shuffle: bool = False, seed: int = 0
) -> Callable[[FrameRef], np.ndarray | None]:
    """帧 → 实例 id 图。`shuffle=True` 时**逐帧独立打乱像素**(反向自证用)。

    打乱后每个像素的 id 与几何无关 ⇒ 投票是噪声 ⇒ 一致率必须塌到平凡基线。
    **这正是"判据能证伪"的形态**:一个永远给出高一致率的判据不是判据。
    """
    rng = np.random.default_rng(seed)

    def load(fr: FrameRef) -> np.ndarray | None:
        p = capture / "inst" / f"p{int(fr.pitch)}" / f"{fr.index:05d}.png"
        if not p.is_file():
            return None
        ids = decode_instance_png(p.read_bytes())
        if shuffle:
            flat = ids.ravel().copy()
            rng.shuffle(flat)
            ids = flat.reshape(ids.shape)
        return ids

    return load


@dataclass
class AttrResult:
    """逐 Gaussian 的归属结果。三个数组同长(与 `means` 同序)。"""

    labels: np.ndarray  # int64,-1 = 未归属
    hits: np.ndarray  # 命中帧数(投进画幅的帧数,含落在背景上的)
    share: np.ndarray  # 众数占比

    @property
    def n_labeled(self) -> int:
        return int((self.labels >= 0).sum())

    @property
    def rate(self) -> float:
        return self.n_labeled / max(len(self.labels), 1)


def attribute(
    means: np.ndarray,
    frames: list[FrameRef],
    load_ids: Callable[[FrameRef], np.ndarray | None],
    *,
    k: CameraIntrinsics | None = None,
    min_samples: int = MIN_SAMPLES,
    min_share: float = MIN_SHARE,
    chunk: int = CHUNK,
) -> AttrResult:
    """投影 + 投票 → 每个 Gaussian 一个实例 id(`-1` = 未归属)。**纯函数**(读图经 `load_ids`)。"""
    k = k or intrinsics()
    n = len(means)
    labels = np.full(n, -1, dtype=np.int64)
    hits = np.zeros(n, dtype=np.int64)
    share = np.zeros(n, dtype=np.float64)
    p = np.asarray(means, dtype=np.float64)[:, :3]

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sub = p[s:e]
        gi_list: list[np.ndarray] = []
        id_list: list[np.ndarray] = []
        for fr in frames:
            ids = load_ids(fr)
            if ids is None:
                continue
            ui, vi, ok = project_points(sub, fr, k)
            if not ok.any():
                continue
            li = np.flatnonzero(ok)
            gi_list.append(li)
            id_list.append(ids[vi[ok], ui[ok]].astype(np.int64))
        m = e - s
        if not gi_list:
            continue
        gi = np.concatenate(gi_list)
        ii = np.concatenate(id_list)
        key = gi * STRIDE + ii
        uniq, cnt = np.unique(key, return_counts=True)
        gid, iid = np.divmod(uniq, STRIDE)
        # 每个 (Gaussian, id) 组的计数;取最大值那一组 = 众数(id 相同则取较小的,确定性)
        order = np.lexsort((iid, -cnt, gid))
        gid, iid, cnt = gid[order], iid[order], cnt[order]
        first = np.empty(len(gid), dtype=bool)
        first[0] = True
        np.not_equal(gid[1:], gid[:-1], out=first[1:])
        heads = np.flatnonzero(first)
        mode_id = np.zeros(m, dtype=np.int64)
        mode_cnt = np.zeros(m, dtype=np.int64)
        mode_id[gid[heads]] = iid[heads]
        mode_cnt[gid[heads]] = cnt[heads]
        tot = np.bincount(gid, weights=cnt.astype(np.float64), minlength=m)
        hits[s:e] = tot.astype(np.int64)
        sh = mode_cnt / np.maximum(tot, 1.0)
        share[s:e] = sh
        good = (hits[s:e] >= min_samples) & (sh >= min_share) & (mode_id > 0)
        labels[s:e] = np.where(good, mode_id, -1)
    return AttrResult(labels=labels, hits=hits, share=share)


@dataclass
class Consistency:
    """两半独立投票的一致率 + 两条对照。`n = 0` ⇒ **未判**(见模块头注)。"""

    n: int = 0
    agree: float = float("nan")
    only_a: int = 0
    only_b: int = 0
    chance: float = float("nan")
    shuffled: float = float("nan")

    @property
    def control(self) -> float:
        """对照里**更高的那个** —— 裁决要比的是"最难超过的那个",不是随便挑一个。"""
        vals = [v for v in (self.chance, self.shuffled) if not np.isnan(v)]
        return max(vals) if vals else float("nan")

    @property
    def kappa(self) -> float:
        """`(agree − control) / (1 − control)` —— 把可达到的余量吃掉了多少。

        `control` 已经含"打乱对照",故它同时排掉了"标签分布本来就很集中"与"对应关系被打乱"两种假象。
        与 §A.1「16 dB 要跟复制最近邻比」同一条纪律:**先减去平凡基线,再谈好坏**。
        """
        c = self.control
        if np.isnan(c) or c >= 1.0:
            return float("nan")
        return (self.agree - c) / (1.0 - c)

    def verdict(self, min_agree: float = MIN_AGREE, min_kappa: float = MIN_KAPPA) -> tuple[bool | None, str]:
        if self.n == 0:
            return None, (f"未判(两半都归属的 Gaussian 为 0;单半归属 A={self.only_a} / B={self.only_b})")
        if self.agree < min_agree:
            return False, f"一致率 {self.agree:.3f} < {min_agree}"
        k = self.kappa
        if np.isnan(k) or k < min_kappa:
            return False, f"超额一致率 κ={k:.3f} < {min_kappa}(对照 {self.control:.3f} 太高,一致率没吃下余量)"
        return True, (
            f"一致率 {self.agree:.3f},κ={k:.3f}(平凡基线 {self.chance:.3f} / 打乱对照 {self.shuffled:.3f})"
        )


def consistency_of_two_splits(
    means: np.ndarray, frames: list[FrameRef], load_ids, *, k=None, seed: int = 0
) -> Consistency:
    """把帧集按**视角序号**一分为二(偶数 / 奇数),两半**各自独立投票**,比一致率。

    对照两条(都不是摆设):
    - `chance` = 按 A 半标签频率算的**碰撞概率** `Σ p_i²` —— 乱贴能蒙到的一致率;
    - `shuffled` = 把 B 半标签**随机置换**后与 A 半比 —— 打乱一切对应关系后的落点。

    ⚠️ **只在两半都归属的 Gaussian 上比**;只被一半归属的单独计数(第三态),不并入裁决。
    """
    k = k or intrinsics()
    ev = [fr for i, fr in enumerate(frames) if i % 2 == 0]
    od = [fr for i, fr in enumerate(frames) if i % 2 == 1]
    ra = attribute(means, ev, load_ids, k=k)
    rb = attribute(means, od, load_ids, k=k)
    both = (ra.labels >= 0) & (rb.labels >= 0)
    n = int(both.sum())
    only_a = int(((ra.labels >= 0) & (rb.labels < 0)).sum())
    only_b = int(((ra.labels < 0) & (rb.labels >= 0)).sum())
    if n == 0:
        return Consistency(n=0, only_a=only_a, only_b=only_b)
    la, lb = ra.labels[both], rb.labels[both]
    cnt = np.bincount(la)
    p = cnt[cnt > 0] / cnt.sum()
    chance = float((p**2).sum())
    rng = np.random.default_rng(seed)
    shuffled = float((la == rng.permutation(lb)).mean())
    return Consistency(
        n=n, agree=float((la == lb).mean()), only_a=only_a, only_b=only_b, chance=chance, shuffled=shuffled
    )


# --------------------------------------------------------------------- 类归属(报告用)
def instance_classes(
    capture: Path, frames: list[FrameRef], *, limit: int = 60
) -> dict[int, tuple[int, float]]:
    """实例 id → `(众数语义 tag, 占比)`。**报告用**,不进裁决。

    为什么要它:阶段的**触发重评条件**是「若归属误差大到无法区分相邻物体,则退回到**类级**编辑」
    —— 那就得先知道现在归属出来的 id 都是些什么东西。tag 名见 `perception/sem_tags.py`。
    只看前 `limit` 帧就够(同一批关卡网格的 id → tag 不随帧变)。
    """
    tally: dict[int, dict[int, int]] = {}
    used = 0
    for fr in frames:
        if used >= limit:
            break
        pi = capture / "inst" / f"p{int(fr.pitch)}" / f"{fr.index:05d}.png"
        ps = capture / "sem" / f"p{int(fr.pitch)}" / f"{fr.index:05d}.png"
        if not (pi.is_file() and ps.is_file()):
            continue
        ids = decode_instance_png(pi.read_bytes()).ravel()
        tags = decode_tag_png(ps.read_bytes()).ravel()
        used += 1
        key = ids.astype(np.int64) * 256 + tags.astype(np.int64)
        uniq, cnt = np.unique(key, return_counts=True)
        gid, tag = np.divmod(uniq, 256)
        for i, t, c in zip(gid.tolist(), tag.tolist(), cnt.tolist(), strict=True):
            if i == 0:
                continue
            tally.setdefault(i, {})
            tally[i][t] = tally[i].get(t, 0) + c
    out: dict[int, tuple[int, float]] = {}
    for i, d in tally.items():
        t, c = max(d.items(), key=lambda kv: kv[1])
        out[i] = (t, c / sum(d.values()))
    return out


# ------------------------------------------------------------------------- 自证(合成夹具)
def _synth_fixture(root: Path, *, shuffle: bool = False, seed: int = 3):
    """合成夹具:3 个 Gaussian(各自一个已知 id)+ 1 个背景点,8 个相机环绕。

    实例图**由已知布局直接画**(不是跑一遍模型),故这里的"一致率"同时是**正确率** ——
    这是本模块唯一能给出准确率的地方(真实数据没有"哪个 Gaussian 属于谁"的真值,见模块头注)。

    返回 `(means, frames, loader)`。
    """
    k = intrinsics()
    # 三个物体:世界系位置(+ 一个在相机**视场外**的点,用来钉"未归属")。
    # ⚠️ 高度必须留在画幅里:相机高 1.5、竖直半 FOV 只有 16.8°(见 `intrinsics`)⇒
    #    远处物体的可容忍高度 ≈ 1.5 + 0.30×水平距离。摆到 6 m 高就整段在画外。
    truth = {11: (0.0, 0.0, 1.5), 22: (2.0, 0.0, 1.0), 33: (-2.0, 0.0, 2.0)}
    means = np.array([*truth.values(), (0.0, 0.0, -40.0)], dtype=np.float64)
    frames: list[FrameRef] = []
    rng = np.random.default_rng(seed)
    for c in range(8):
        yaw = np.radians(360.0 * c / 8)
        loc = (6.0 * np.cos(yaw), 6.0 * np.sin(yaw), 1.5)
        # 相机朝向世界原点:(pitch=0 ⇒ 水平)用 yaw 让相机光轴 +x 指向中心
        yaw_cam = np.degrees(np.arctan2(-loc[1], -loc[0]))
        frames.append(
            FrameRef(pitch=0.0, index=c, location=loc, rotation_rad=(0.0, np.radians(yaw_cam), 0.0))
        )
    # 画实例图:每个 Gaussian 投成一个圆点,像素值 = 它的 id(背景 0)
    painted: dict[FrameRef, np.ndarray] = {}
    for fr in frames:
        img = np.zeros((SRC_H, SRC_W), dtype=np.uint16)
        for bid, pos in truth.items():
            ui, vi, ok = project_points(np.array([pos]), fr, k)
            if not ok[0]:
                continue
            u, v = int(ui[0]), int(vi[0])
            r = 4
            img[max(0, v - r) : v + r + 1, max(0, u - r) : u + r + 1] = bid
        painted[fr] = img

    def load(fr: FrameRef) -> np.ndarray | None:
        img = painted[fr]
        if shuffle:
            flat = img.ravel().copy()
            rng.shuffle(flat)
            img = flat.reshape(img.shape)
        return img

    return means, frames, load


def self_test() -> bool:
    """**尺子自证**(不需要 CARLA / 不需要训练):还原 / 打乱塌 / 未归属 / 未判。"""
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as td:
        means, frames, load = _synth_fixture(Path(td))
        r = attribute(means, frames, load)
        truth = np.array([11, 22, 33, -1])
        good = bool((r.labels == truth).all())
        print(
            f"① 已知布局必须逐点还原:{r.labels.tolist()} vs 真值 {truth.tolist()} —— {'✅' if good else '❌'}"
        )
        ok &= good

        c = consistency_of_two_splits(means, frames, load)
        v, why = c.verdict()
        good = v is True and abs(c.agree - 1.0) < 1e-9
        print(f"② 跨视角一致率 {c.agree:.3f}(n={c.n})→ verdict={v} —— {'✅' if good else '❌'}  {why}")
        ok &= good

        # ③ **反向自证**:打乱实例图 ⇒ 判据必须拒绝给出结论(归属崩塌)
        _, _, load_sh = _synth_fixture(Path(td), shuffle=True)
        r_sh = attribute(means, frames, load_sh)
        cs = consistency_of_two_splits(means, frames, load_sh)
        v3, _ = cs.verdict()
        # 崩塌的**两种**形态都算证伪成立:① 一条归属都给不出(判据拒绝结论);
        # ② 给出了一致率,但落在平凡基线量级。**不要求一定是哪一种** —— 要求的是"必须塌"。
        good = v3 is None or cs.agree < MARGIN * max(cs.chance, cs.shuffled)
        print(
            f"③ 打乱实例图 ⇒ 归属 {r_sh.n_labeled}/{len(means)}(原 {r.n_labeled}/{len(means)}),"
            f"一致率 {cs.agree:.3f}(基线 {cs.chance:.3f})→ verdict={v3} —— {'✅' if good else '❌'}"
        )
        ok &= good

        # ③b 判据内的**随机置换对照**必须真的低 —— 它是"一致率高"这句话的分母,不是摆设
        good = c.shuffled < c.agree and c.shuffled <= max(c.chance * MARGIN, 1.0)
        print(
            f"③b 干净夹具上的一致率 {c.agree:.3f} vs 置换对照 {c.shuffled:.3f}"
            f"(碰撞概率 {c.chance:.3f})—— {'✅' if good else '❌'}"
        )
        ok &= good

        # ④ 只在一帧里命中 ⇒ 未归属(min_samples 生效)
        one = attribute(means, frames[:1], load)
        good = bool((one.labels < 0).all())
        print(f"④ 只给 1 帧(min_samples={MIN_SAMPLES})⇒ 全部未归属 —— {'✅' if good else '❌'}")
        ok &= good

        # ⑤ 两半没有共同归属 ⇒ 未判(第三态)
        cn = consistency_of_two_splits(means, frames[:1], load)
        v5, _ = cn.verdict()
        good = v5 is None
        print(f"⑤ 帧不足以让两半都归属 ⇒ verdict={v5}(必须 None)—— {'✅' if good else '❌'}")
        ok &= good

        # ⑥ 向量化投影必须与 calib.core.world_to_img 逐点等价(同一口径的机械钉)
        from autodrivedata.calib.core import world_to_img

        rng = np.random.default_rng(0)
        pts = rng.normal(0, 8, size=(300, 3))
        fr = frames[0]
        ui, vi, m = project_points(pts, fr, intrinsics())
        bad = 0
        for j, pt in enumerate(pts):
            uv = world_to_img(tuple(pt), fr.location, fr.rotation_rad, intrinsics())
            if (uv is None) == bool(m[j]):
                bad += 1
                continue
            if uv is not None and (int(round(uv[0])) != int(ui[j]) or int(round(uv[1])) != int(vi[j])):
                bad += 1
        good = bad == 0
        print(f"⑥ 向量化投影 vs `calib.core.world_to_img` 300 点不一致 {bad} 个 —— {'✅' if good else '❌'}")
        ok &= good
    return bool(ok)


# ------------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Gaussian → 实例归属(投影 + 投票 + 跨视角一致率)")
    ap.add_argument("--capture", default="outputs/3dgs_sync/capture")
    ap.add_argument("--tag", default="", help="与 `train_3dgs_mini --tag` 同口径(那边拼成 `_tag`)")
    ap.add_argument("--means", default=None, help="默认 outputs/3dgs/means{_tag}.npy")
    ap.add_argument("--out", default=None, help="默认 outputs/3dgs/attr{_tag}.npy")
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    ap.add_argument("--min-share", type=float, default=MIN_SHARE)
    ap.add_argument("--shuffle-inst", action="store_true", help="反向自证:逐帧打乱实例图,一致率必须塌")
    ap.add_argument("--self-test", action="store_true", help="合成夹具,不需要 CARLA / 训练")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)

    tag = f"_{args.tag}" if args.tag else ""
    means_path = project_path(args.means or f"outputs/3dgs/means{tag}.npy")
    out_path = project_path(args.out or f"outputs/3dgs/attr{tag}.npy")
    if not means_path.is_file():
        raise SystemExit(f"{means_path} 不存在 —— 先把模型训出来(--tag 拼成 `_{args.tag}`)")

    capture = project_path(args.capture)
    means = np.load(means_path)
    frames = load_frames(capture)
    load = inst_loader(capture, shuffle=args.shuffle_inst)

    with runlog.run("autodrivedata.gs.attribute_instances") as rl:
        rl.input(str(means_path), "means")
        rl.input(str(capture), "capture")
        rl.highlight("tag", args.tag)
        rl.highlight("shuffle_inst", args.shuffle_inst)
        rl.highlight("min_samples", args.min_samples)
        rl.highlight("min_share", args.min_share)
        rl.highlight("n_gaussians", int(len(means)))
        rl.highlight("n_frames", len(frames))

        res = attribute(means, frames, load, min_samples=args.min_samples, min_share=args.min_share)
        print(
            f"[归属] {len(means)} 个 Gaussian,{len(frames)} 帧 ⇒ 归属 {res.n_labeled}"
            f"({res.rate:.1%}),未归属 {len(means) - res.n_labeled}"
        )
        print(f"[归属] 命中帧数:中位 {np.median(res.hits):.0f} / 众数占比:中位 {np.median(res.share):.3f}")

        c = consistency_of_two_splits(means, frames, load, seed=0)
        v, why = c.verdict()
        lab, cnt = np.unique(res.labels[res.labels >= 0], return_counts=True)
        top = sorted(zip(cnt.tolist(), lab.tolist(), strict=True), reverse=True)[:5]
        print(
            f"\n=== 跨视角一致率(偶数帧 / 奇数帧各投一次)===\n"
            f"  可判 Gaussian {c.n}(只 A 半 {c.only_a} / 只 B 半 {c.only_b};"
            f"**未判**不并入裁决)\n"
            f"  一致率 {c.agree:.4f} | 平凡基线(碰撞概率) {c.chance:.4f} | 打乱对照 {c.shuffled:.4f}"
            f" | 倍率 {c.agree / max(c.control, 1e-9):.2f}× | **κ {c.kappa:.4f}**\n"
            f"  标签分布(前 5 大,共 {len(lab)} 个 id):"
            + ", ".join(f"id{i}={n}({n / max(res.n_labeled, 1):.0%})" for n, i in top)
            + f"\n  裁决:{'未判' if v is None else ('✅ 过' if v else '❌ 不过')} —— {why}"
        )

        cls = instance_classes(capture, frames)
        if cls:
            top = sorted(((i, t, s) for i, (t, s) in cls.items()), key=lambda x: -x[2])[:5]
            print(
                "[类归属] 前几大实例(报告用,不进裁决):"
                + ", ".join(f"id{j}={TAG_NAMES.get(t, t)}({s:.0%})" for j, t, s in top)
            )
        else:
            print("[类归属] 无 `sem/` 或 `inst/` 图 —— 类归属**未判**(不是「没有物体」)")

        np.save(out_path, res.labels)
        print(f"[done] {out_path}(int64,-1 = 未归属)")
        rl.highlight("n_labeled", res.n_labeled)
        rl.highlight("label_rate", round(res.rate, 4))
        rl.highlight("median_hits", float(np.median(res.hits)))
        rl.highlight("consistency_n", c.n)
        rl.highlight("consistency_agree", round(c.agree, 4))
        rl.highlight("consistency_chance", round(c.chance, 4))
        rl.highlight("consistency_shuffled", round(c.shuffled, 4))
        rl.highlight("consistency_kappa", round(c.kappa, 4))
        rl.highlight("consistency_only_a", c.only_a)
        rl.highlight("consistency_only_b", c.only_b)
        rl.highlight("verdict", "未判" if v is None else ("过" if v else "不过"))
        rl.artifact(out_path, "attr-labels")


if __name__ == "__main__":
    main()

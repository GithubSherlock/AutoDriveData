"""LiDAR 的 A/B 判据:**确定性**(阶段 L0)与**「靶有多大」**(阶段 L1)。

纯值:读 `.bin` + 读 `pose/` + 算距离。**不 import carla、不 import torch** ——
它只吃盘上已采好的数据(由 `test_layer_guard.py` 机械强制)。

## 为什么不是「逐点相同」

第一版的想法是比两次采集的点云是否逐点相同。**实测立刻否掉了它**:
`kitti_ab_occl2_none_all` 与 `..._partial_nearest` 的**位姿本来就不同** ——
70 帧里平移差最大 **0.0619 m**(帧 0 是 0.0299),而步长两侧几乎相同(0.7935 vs 0.7934)
⇒ **不是启动瞬态,是持续的亚帧相位差**。

⚠️ **同一个 0.06 m,对相机是「可接受」**(P1 红线记的 A/B 容差是 0.017–0.07 m),
**对 LiDAR 却是整片点云平移 0.06 m**。⇒ 它**直接盖过**遮挡物的效应,
而「点云对不上」本身**不报错**,只会被读成「编辑效果不明显」。

⇒ 本模块据此定三条口径:

1. **先验位姿**,并把它当成**必须报出来的读数**(不是背景假设);
2. 比的是**世界系下的最近邻距离**,不是逐点相等 —— 两次采样落在世界上的**不同点**,
   问「每个 A 点到 B 最近的有多远」才是有意义的问法;
3. ★ **用同一对里的「后方」当对照**:遮挡物在车前,`x<0` 的点**不可能**被它挡住
   ⇒ 那半边的差异**必然是位姿抖动**,是该对自身的 floor。**前方读数只有相对它才可读。**

用法:
  python -m autodrivedata.perception.lidar_ab \
      --root-a outputs/kitti_ab_occl2_none_all --root-b outputs/kitti_ab_occl2_partial_nearest
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 前方/后方的分界(m,车形 x 向前)。0 = 就在传感器处切。**这是定标量**:
#: 往前后挪会把「可能被遮挡」的范围算错(同 `static_eval` 的邻域教训)。
FRONT_X = 0.0


@dataclass(frozen=True)
class PairStats:
    """一对采集里**某一帧**的读数。全部是量出来的,没有一个是推的。"""

    fid: int
    n_a: int
    n_b: int
    pose_dt: float  # 位姿平移最大差(m)
    pose_dr: float  # 位姿旋转元素最大差
    nn_median: float  # A→B 最近邻距离中位(m),全体
    nn_p90: float
    nn_median_rear: float  # ★ 后方(对照):遮挡物够不着
    nn_median_front: float
    d_front: int  # 前方点数差(B−A)
    d_rear: int  # ★ 后方点数差 —— 位姿抖动造成的点数漂移 floor


def load_velodyne(path: Path) -> np.ndarray:
    """KITTI 口径 `(N,4)` float32(x,y,z,intensity)。

    ⚠️ **空点云当场抛**:它与「这个地方没有点」是两回事,静默当成 0 点会让所有统计量失真。
    """
    a = np.fromfile(path, dtype=np.float32)
    if a.size == 0:
        raise SystemExit(f"{path} 是空的 —— 空点云不是「没有点」,是采集或落盘坏了")
    if a.size % 4:
        raise SystemExit(f"{path} 有 {a.size} 个 float,不是 4 的倍数 —— 不是 KITTI velodyne 口径")
    return a.reshape(-1, 4)


def load_pose(path: Path) -> np.ndarray:
    """`(3,4)` 位姿(LiDAR 系 → 世界)。⚠️ 文件是**一行 12 个数**,别忘了 reshape。"""
    v = np.loadtxt(path).reshape(-1)
    if v.size != 12:
        raise SystemExit(f"{path} 有 {v.size} 个数,不是 12 —— 不是 3×4 位姿")
    return v.reshape(3, 4).astype(np.float64)


def pose_delta(pa: np.ndarray, pb: np.ndarray) -> tuple[float, float]:
    """两份位姿的 `(平移最大差 m, 旋转元素最大差)`。"""
    return float(np.abs(pa[:, 3] - pb[:, 3]).max()), float(np.abs(pa[:, :3] - pb[:, :3]).max())


def to_world(pts: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """车形点 `(N,4)` → 世界系 `(N,3)`:`p_w = R @ p_l + t`。"""
    return pts[:, :3].astype(np.float64) @ pose[:, :3].T + pose[:, 3]


def region_of(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """车形点 → `(前方掩膜, 后方掩膜)`。**后方是 A/B 判据的自带对照**(见模块头注)。"""
    front = pts[:, 0] > FRONT_X
    return front, ~front


def nn_distance(a_world: np.ndarray, b_world: np.ndarray) -> np.ndarray:
    """每个 A 点到 B 的最近邻距离(米)。

    ⚠️ **不是逐点相等**:两次采样的位姿本来就差 ~0.06 m,采样落在世界上的**不同点**
    ⇒ 「逐点相等」会把位姿差误报成「编辑效果」。问最近邻才对这个问题有意义。
    """
    if len(b_world) == 0:
        return np.full(len(a_world), np.inf)
    return cKDTree(b_world).query(a_world, k=1, workers=-1)[0]


def frame_stats(
    fid: int, pt_a: np.ndarray, pose_a: np.ndarray, pt_b: np.ndarray, pose_b: np.ndarray
) -> PairStats:
    """一帧的读数。**纯函数** —— 给两组点云 + 两个位姿,出全部 10 个数。"""
    dt, dr = pose_delta(pose_a, pose_b)
    fa, ra = region_of(pt_a)
    fb, rb = region_of(pt_b)
    d = nn_distance(to_world(pt_a, pose_a), to_world(pt_b, pose_b))
    return PairStats(
        fid=fid,
        n_a=int(len(pt_a)),
        n_b=int(len(pt_b)),
        pose_dt=dt,
        pose_dr=dr,
        nn_median=float(np.median(d)),
        nn_p90=float(np.percentile(d, 90)),
        nn_median_rear=float(np.median(d[ra])) if ra.any() else float("nan"),
        nn_median_front=float(np.median(d[fa])) if fa.any() else float("nan"),
        d_front=int(fb.sum() - fa.sum()),
        d_rear=int(rb.sum() - ra.sum()),
    )


def _frames_of(root: Path) -> list[int]:
    d = root / "training/velodyne"
    if not d.is_dir():
        raise SystemExit(f"{d} 不存在 —— 这不是一份带 LiDAR 的 KITTI root")
    return sorted(int(p.stem) for p in d.glob("*.bin"))


def compare_pair(root_a: Path, root_b: Path, frames: list[int] | None = None, *, max_frames: int = 0) -> dict:
    """跑一整对采集 → 逐帧读数 + 汇总。位姿先验、逐帧存档(见模块头注三条口径)。"""
    fa, fb = _frames_of(root_a), _frames_of(root_b)
    if fa != fb:
        raise SystemExit(f"两侧帧号不同:{fa[:3]}… vs {fb[:3]}… —— 不是一对可比的采集")
    todo = fa if frames is None else [f for f in fa if f in set(frames)]
    if max_frames:
        todo = todo[:max_frames]
    rows = [
        frame_stats(
            fid,
            load_velodyne(root_a / "training/velodyne" / f"{fid:06d}.bin"),
            load_pose(root_a / "training/pose" / f"{fid:06d}.txt"),
            load_velodyne(root_b / "training/velodyne" / f"{fid:06d}.bin"),
            load_pose(root_b / "training/pose" / f"{fid:06d}.txt"),
        )
        for fid in todo
    ]
    if not rows:
        return {"verdict": "未判", "reason": "没有可比帧"}
    g = lambda k: np.array([getattr(r, k) for r in rows], dtype=float)  # noqa: E731
    return {
        "verdict": "可判",
        "n_frames": len(rows),
        "pose_dt_max": float(g("pose_dt").max()),
        "pose_dr_max": float(g("pose_dr").max()),
        "n_a_median": float(np.median(g("n_a"))),
        "n_b_median": float(np.median(g("n_b"))),
        "nn_median_median": float(np.median(g("nn_median"))),
        "nn_p90_median": float(np.median(g("nn_p90"))),
        # ⚠️ 全 NaN(某些合成/极端输入里后方为空)时 `nanmedian` 会发警告 ⇒ 先查
        "nn_median_rear_median": float(np.nanmedian(g("nn_median_rear")))
        if np.isfinite(g("nn_median_rear")).any()
        else float("nan"),
        "nn_median_front_median": float(np.nanmedian(g("nn_median_front"))),
        "d_front_median": float(np.median(g("d_front"))),
        "d_rear_median": float(np.median(g("d_rear"))),
        "per_frame": [asdict(r) for r in rows],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root-a", required=True)
    ap.add_argument("--root-b", required=True)
    ap.add_argument("--frames", default="", help="逗号分隔;空 = 全部")
    ap.add_argument("--max-frames", type=int, default=0, help="只跑前 N 帧(冒烟用)")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-runlog", action="store_true", help="不落 logs/ 三件套")
    args = ap.parse_args()

    frames = [int(x) for x in args.frames.split(",") if x.strip()] or None
    with runlog.run("autodrivedata.perception.lidar_ab") as rl:
        rl.input(args.root_a, "root-a")
        rl.input(args.root_b, "root-b")
        rep = compare_pair(
            project_path(args.root_a), project_path(args.root_b), frames, max_frames=args.max_frames
        )
        if rep["verdict"] != "可判":
            print(f"[L0/L1] {rep['verdict']}:{rep['reason']}")
            rl.highlight("verdict", rep["verdict"])
            return
        print(
            f"[L0 位姿] {rep['n_frames']} 帧 | 平移最大差 {rep['pose_dt_max']:.4f} m | 旋转 {rep['pose_dr_max']:.5f}"
        )
        print(f"[L0 点数] A 中位 {rep['n_a_median']:.0f} | B 中位 {rep['n_b_median']:.0f}")
        print(
            f"[L1 靶]  最近邻距离 全体中位 {rep['nn_median_median']:.4f} m | p90 {rep['nn_p90_median']:.4f} m"
        )
        print(f"[★对照]  后方(遮挡够不着) 最近邻中位 {rep['nn_median_rear_median']:.4f} m")
        print(f"         前方              最近邻中位 {rep['nn_median_front_median']:.4f} m")
        print(f"[★对照]  点数差 前方 {rep['d_front_median']:+.0f} | 后方(floor) {rep['d_rear_median']:+.0f}")
        for k in (
            "pose_dt_max",
            "nn_median_median",
            "nn_median_rear_median",
            "d_front_median",
            "d_rear_median",
        ):
            rl.highlight(k, round(rep[k], 5))
        if args.out:
            p = project_path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"[done] {p.resolve()}")
            rl.artifact(p, "lidar-ab")


if __name__ == "__main__":
    main()

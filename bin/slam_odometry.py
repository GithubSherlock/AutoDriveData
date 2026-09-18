"""S1.3 激光 SLAM 前端:逐帧 velodyne → 链式位姿(教程 14 两段式阶段 1)。

对 kitti_drive 的 velodyne 序列做帧间点面 ICP(恒速先验初始化),输出:
- `outputs/slam/traj_raw.json`:链式 T_0→k 位姿 + 每帧残差/重叠/收敛
- `outputs/slam/icp_stats.json`:全程统计(平均 RMSE/重叠/失败帧数/耗时)

判据(阶段 1 验收):150 帧 <5min、零 NaN、漂移率(回环闭合时)>如实报告。
纯值,不 import carla/torch;产物经 paths.project_path 落 outputs/。

用法:
  python bin/slam_odometry.py [--root outputs/kitti_drive] [--frames 0-149]
                              [--voxel 0.5] [--out outputs/slam]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from autodrivedata.accum import voxel_downsample
from autodrivedata.paths import project_path
from autodrivedata.slam import (
    DOWNSAMPLE_VOXEL,
    GRID_CELL,
    closure_error,
    icp_odometry,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/kitti_drive", help="KITTI root(velodyne 所在)")
    ap.add_argument("--frames", default="0-149", help="帧范围 0-149 或逗号列表")
    ap.add_argument("--voxel", type=float, default=DOWNSAMPLE_VOXEL, help="下采样体素边长(m)")
    ap.add_argument("--out", default="outputs/slam", help="输出根(经 project_path)")
    args = ap.parse_args()

    root = Path(args.root)
    frames: list[int] = []
    for tok in args.frames.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            frames.extend(range(int(a), int(b) + 1))
        else:
            frames.append(int(tok))
    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[data] {root} | {len(frames)} 帧 | voxel {args.voxel}m")
    t0 = time.time()

    traj: list[dict] = []
    poses: list[np.ndarray] = []
    prev_down: np.ndarray | None = None
    # 恒速先验:Δ_{k-2→k-1} 初始为恒等(帧0→1 用 identity 先验)
    delta_prev = np.eye(4)
    n_nan = 0
    n_failed = 0
    rmse_sum = 0.0
    overlap_sum = 0.0
    t_icp = 0.0

    for k, fid in enumerate(frames):
        p = root / "training" / "velodyne" / f"{fid:06d}.bin"
        if not p.exists():
            print(f"[skip] {p.name} 不存在")
            continue
        pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
        down = voxel_downsample(pts, args.voxel)  # (M,4),intensity 取首点

        if k == 0:
            prev_down = down
            init = np.eye(4)
            res = _first_result(init)
        else:
            # icp_odometry 契约:init_T = **上一帧链式位姿**(T_0→k−1),seed = 恒速增量 Δ。
            # **不可再乘 delta_prev**——seed 已在 icp_odometry 内作为迭代起点
            # (cur = seed·src)应用一次,init 里再乘一次 = 恒速先验被叠加两次:
            # 实测 0.5m/帧 合成序列累计出 0.5/1.5/3.0/5.0/7.5(真值 0.5..2.5),
            # 轨迹按 k² 发散 → 帧间重叠崩、nearest_batch 活跃集不收缩(单帧 20s+)。
            init = poses[-1]
            t_a = time.time()
            assert prev_down is not None  # k>0 时必有前帧
            res = icp_odometry(prev_down, down, init, seed=delta_prev)
            t_icp += time.time() - t_a
            prev_down = down

        T = res["T"]
        if not np.isfinite(T).all():
            n_nan += 1
            print(f"[{fid}] NaN 位姿,改恒速先验兜底")
            T = init
        poses.append(T)
        if k > 0:
            delta_prev = relative_transform(poses[-2], poses[-1])
            if res["failed"]:
                n_failed += 1
            rmse_sum += res["rmse_final"]
            overlap_sum += res["overlap"]
        traj.append(
            {
                "frame": fid,
                "T": T.tolist(),
                "rmse_final": res["rmse_final"],
                "overlap": res["overlap"],
                "converged": res["converged"],
                "iters": res["iters"],
                "failed": res["failed"],
            }
        )
        if (k % 20) == 0 or k == len(frames) - 1:
            print(f"[{k}/{len(frames)}] t={time.time() - t0:.1f}s")

    # 闭合误差(无回环时 = 末帧相对首帧,开放路径参考口径)
    ce = closure_error(poses)
    stats = {
        "sensor": "velodyne",
        "dataset": str(root),
        "n_frames": len(traj),
        "voxel": args.voxel,
        "cell": GRID_CELL,
        "n_nan": n_nan,
        "n_failed": n_failed,
        "mean_rmse": round(rmse_sum / max(len(traj) - 1, 1), 5),
        "mean_overlap": round(overlap_sum / max(len(traj) - 1, 1), 4),
        "closure": ce,
        "wall_s": round(time.time() - t0, 2),
        "icp_s": round(t_icp, 2),
    }
    traj_json = {
        "sensor": "velodyne",
        "dataset": str(root),
        "n_frames": len(traj),
        "voxel": args.voxel,
        "cell": GRID_CELL,
        "traj": traj,
        "icp_stats": {
            "n_failed": n_failed,
            "mean_rmse": stats["mean_rmse"],
            "mean_overlap": stats["mean_overlap"],
            "n_nan": n_nan,
        },
    }
    (out / "traj_raw.json").write_text(json.dumps(traj_json, indent=2, ensure_ascii=False))
    (out / "icp_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(
        f"[done] {len(traj)} 帧 | 平均 RMSE {stats['mean_rmse']} | "
        f"overlap {stats['mean_overlap']} | failed {n_failed} | NaN {n_nan}"
    )
    print(f"  closure: {json.dumps(ce, ensure_ascii=False)}")
    print(f"  耗时 {stats['wall_s']}s(IC {stats['icp_s']}s)→ {out}/traj_raw.json")


def _first_result(init: np.ndarray) -> dict:
    """首帧:无 src/ref,直接给恒等增量(链式 T_0→0 = identity)。"""
    return {
        "T": np.array(init, dtype=np.float64),
        "rmse_final": 0.0,
        "overlap": 1.0,
        "converged": True,
        "iters": 0,
        "failed": False,
    }


def relative_transform(Ta: np.ndarray, Tb: np.ndarray) -> np.ndarray:
    """T_ab = Ta⁻¹ Tb(链式 Δ 用,避免 import slam.relative_transform 命名冲突)。"""
    return np.linalg.inv(Ta) @ Tb


if __name__ == "__main__":
    main()

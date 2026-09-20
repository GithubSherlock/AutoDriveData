"""S1.4 激光 SLAM 后端:关键帧 + ScanContext 回环 + 位姿图优化(教程 14 两段式阶段 2)。

读阶段 1 的 `traj_raw.json`(链式 T_0→k),在关键帧上:
1. 每 KEYFRAME_EVERY 帧取一个关键帧,建 ScanContext 描述子;
2. 回环候选 = 描述子距离 < SC_SIM_THRESH ∧ 关键帧号差 ≥ min_gap;
3. 候选逐个用**双 yaw 初值 ICP**(0°/180°,ScanContext 对 yaw 有 180° 歧义)验证,
   过门(overlap ≥ LOOP_GATE_OVERLAP ∧ rmse < LOOP_GATE_RMSE ∧ converged)才成为回环边;
4. 位姿图 = 里程计边(相邻关键帧)+ 回环边 → G-N LM 优化。

输出:
- `traj_pgo.json`:优化后的关键帧位姿 + 全帧位姿(每帧 = 其关键帧优化位姿 ∘ 原始相对增量)
- `loops.json`:每条回环(关键帧对/描述子距离/ICP overlap/rmse/位移)
- `slam_summary.json`:关键帧数/回环数/优化前后闭合误差/耗时

**如实报告纪律**:直线段数据回环数天然为 0(n_loops=0 是合法结果,不造回环);
`closure_error` 在开放路径上 = 首末位姿距离,不等于漂移,summary 里两者分开写。

纯值,不 import carla/torch;产物经 paths.project_path 落 outputs/。

用法:
  python bin/slam_backend.py [--traj outputs/slam/traj_raw.json]
                             [--root outputs/kitti_drive] [--out outputs/slam]
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
    KEYFRAME_EVERY,
    LOOP_GATE_CONVERGED,
    LOOP_GATE_OVERLAP,
    LOOP_GATE_RMSE,
    PGO_W_LOOP,
    PGO_W_ODOM,
    SC_MIN_GAP_NODES,
    SC_SIM_THRESH,
    SC_TOP_N,
    Edge,
    closure_error,
    desc_scan_context,
    icp_odometry,
    pose_graph_optimize,
    sc_candidates,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="outputs/slam/traj_raw.json", help="阶段 1 轨迹(读,经 cwd)")
    ap.add_argument("--root", default="outputs/kitti_drive", help="KITTI root(velodyne 所在)")
    ap.add_argument("--out", default="outputs/slam", help="输出根(经 project_path)")
    ap.add_argument("--every", type=int, default=KEYFRAME_EVERY, help="关键帧间隔(帧)")
    ap.add_argument("--min-gap", type=int, default=SC_MIN_GAP_NODES, help="回环候选最小关键帧号差")
    args = ap.parse_args()

    t0 = time.time()
    traj_doc = json.loads(Path(args.traj).read_text())
    frames = [int(f["frame"]) for f in traj_doc["traj"]]
    poses = [np.array(f["T"], dtype=np.float64) for f in traj_doc["traj"]]
    root = Path(args.root)
    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- 关键帧 + 描述子 ---
    kf_idx = list(range(0, len(frames), args.every))
    descs: list[np.ndarray] = []
    kf_clouds: list[np.ndarray] = []
    for i in kf_idx:
        p = root / "training" / "velodyne" / f"{frames[i]:06d}.bin"
        pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
        kf_clouds.append(voxel_downsample(pts, 0.5))
        descs.append(desc_scan_context(kf_clouds[-1]))
    descs_arr = np.stack(descs)
    print(f"[kf] {len(kf_idx)} 关键帧(每 {args.every} 帧)| 描述子 {descs_arr.shape}")

    # --- 回环候选 + 双 yaw ICP 验证 ---
    loops: list[dict] = []
    for n, i in enumerate(kf_idx):
        cands = sc_candidates(
            descs_arr, descs_arr[n], n, top_n=SC_TOP_N, min_gap=args.min_gap, thresh=SC_SIM_THRESH
        )
        for c, shift, dist in cands:
            j = kf_idx[c]
            # 双 yaw 初值:ScanContext 列滚动不变 → 对 180° 旋转有歧义。shift 是描述子
            # 最佳列滚动量(= yaw 的粗估,60 扇 → 6°/扇),仅记录不参与几何(几何由 ICP 定)
            best = None
            for yaw in (0.0, np.pi):
                c_, s_ = np.cos(yaw), np.sin(yaw)
                rot = np.eye(4)
                rot[:3, :3] = np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])
                # **期望点映射**:把候选帧 j 的点云搬进当前帧 i 的坐标系 = P_i⁻¹P_j,
                # 右乘 rot 绕**源云自身** z 轴预旋(ScanContext 列滚动不变 → 180° 歧义)。
                # 这个量正好是 PGO 边要的 Z_ij(= T_i⁻¹T_j),所以直接用 ICP 的 T_delta 出口。
                guess_map = np.linalg.inv(poses[i]) @ poses[j] @ rot
                # seed 契约 = **位姿增量**(icp_odometry 内部取逆换成点映射),故传其逆。
                # **不能用 init_T 当迭代初值**:init_T 只参与出口合成,不影响 ICP 解。
                res = icp_odometry(
                    kf_clouds[c][:, :3], kf_clouds[n][:, :3], np.eye(4), seed=np.linalg.inv(guess_map)
                )
                if best is None or res["rmse_final"] < best["rmse_final"]:
                    best = res
                    best["yaw"] = float(np.degrees(yaw))
            ok = (
                best["overlap"] >= LOOP_GATE_OVERLAP
                and best["rmse_final"] < LOOP_GATE_RMSE
                and (best["converged"] or not LOOP_GATE_CONVERGED)
            )
            loops.append(
                {
                    "kf_i": n,
                    "kf_j": c,
                    "frame_i": frames[i],
                    "frame_j": frames[j],
                    "sc_dist": round(dist, 5),
                    "sc_shift": int(shift),
                    "yaw_deg": best["yaw"],
                    "icp_overlap": round(best["overlap"], 4),
                    "icp_rmse": round(best["rmse_final"], 5),
                    "converged": bool(best["converged"]),
                    "accepted": bool(ok),
                    # 存 **点映射** T_delta(= Z_ij = P_i⁻¹P_j),PGO 边的口径;
                    # best["T"] 是位姿(init=恒等时 = inv(T_delta)),不是边要的量。
                    "T": best["T_delta"].tolist(),
                }
            )
    accepted = [e for e in loops if e["accepted"]]
    print(f"[loop] 候选 {len(loops)} | 过门 {len(accepted)}")

    # --- 位姿图:里程计边 + 回环边 ---
    edges: list[Edge] = []
    for a, b in zip(range(len(kf_idx) - 1), range(1, len(kf_idx)), strict=True):
        rel = np.linalg.inv(poses[kf_idx[a]]) @ poses[kf_idx[b]]
        edges.append(Edge(a, b, rel, weight=PGO_W_ODOM))
    for e in accepted:
        edges.append(Edge(e["kf_i"], e["kf_j"], np.array(e["T"]), weight=PGO_W_LOOP))

    kf_poses = [poses[i] for i in kf_idx]
    ce_pre = closure_error(kf_poses)
    kf_opt = pose_graph_optimize(kf_poses, edges)
    ce_post = closure_error(kf_opt)

    # --- 全帧位姿:每帧 = 其关键帧优化位姿 ∘ 原始相对增量 ---
    full: list[dict] = []
    for k, i in enumerate(kf_idx):
        base = kf_opt[k] @ np.linalg.inv(poses[i])
        hi = kf_idx[k + 1] if k + 1 < len(kf_idx) else len(frames)
        for m in range(i, hi):
            T = base @ poses[m]
            full.append({"frame": frames[m], "kf": k, "T": T.tolist()})

    summary = {
        "traj_in": args.traj,
        "dataset": str(root),
        "n_frames": len(frames),
        "n_keyframes": len(kf_idx),
        "keyframe_every": args.every,
        "n_loop_candidates": len(loops),
        "n_loops": len(accepted),
        "closure_pre": ce_pre,
        "closure_post": ce_post,
        "wall_s": round(time.time() - t0, 2),
        "note": (
            "开放路径下 closure_* 是首末位姿距离(= 路径长度量级),非漂移率;"
            "n_loops=0 是直线段数据的合法结果(不造回环)"
        ),
    }
    (out / "traj_pgo.json").write_text(
        json.dumps(
            {
                "sensor": "velodyne",
                "dataset": str(root),
                "n_frames": len(frames),
                "n_keyframes": len(kf_idx),
                "keyframe_every": args.every,
                "keyframes": [
                    {"kf": k, "frame": frames[i], "T": kf_opt[k].tolist()} for k, i in enumerate(kf_idx)
                ],
                "traj": full,
                "n_loops": len(accepted),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    (out / "loops.json").write_text(
        json.dumps(
            {"n_candidates": len(loops), "n_accepted": len(accepted), "loops": loops},
            indent=2,
            ensure_ascii=False,
        )
    )
    (out / "slam_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(
        f"[done] 回环 {len(accepted)}/{len(loops)} | 闭合 pre {ce_pre['drift_m']:.3f}m → post {ce_post['drift_m']:.3f}m"
    )
    print(f"  耗时 {summary['wall_s']}s → {out}/traj_pgo.json")


if __name__ == "__main__":
    main()

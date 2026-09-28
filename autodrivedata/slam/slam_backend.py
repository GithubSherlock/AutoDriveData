"""S1.4 激光 SLAM 后端:关键帧 + ScanContext 回环 + 位姿图优化(教程 14 两段式阶段 2)。

读阶段 1 的 `traj_raw.json`(链式 T_0→k),在关键帧上:
1. 每 KEYFRAME_EVERY 帧取一个关键帧,建 ScanContext 描述子;
2. 回环候选 = 描述子距离 < SC_SIM_THRESH ∧ 关键帧号差 ≥ min_gap;
3. 候选先过**几何先验闸**(LOOP_PRIOR_MAX_M)与**廉价筛**(LOOP_SCREEN_*,抽稀云 1 次迭代),
   再用**双 yaw 初值 ICP**(0°/180°,ScanContext 对 yaw 有 180° 歧义)验证,
   过门(overlap ≥ LOOP_GATE_OVERLAP ∧ rmse < LOOP_GATE_RMSE ∧ converged)才成为回环边;
   两条闸只用来**拒**,接受判据不受它们影响(口径与实测依据见常量注释);
4. 位姿图 = 里程计边(相邻关键帧)+ 回环边 → G-N LM 优化。

输出:
- `traj_pgo.json`:优化后的关键帧位姿 + 全帧位姿(每帧 = 其关键帧优化位姿 ∘ 原始相对增量)
- `loops.json`:每条回环(关键帧对/描述子距离/ICP overlap/rmse/位移)
- `slam_summary.json`:关键帧数/回环数/优化前后闭合误差/耗时

**如实报告纪律**:直线段数据回环数天然为 0(n_loops=0 是合法结果,不造回环);
`closure_error` 在开放路径上 = 首末位姿距离,不等于漂移,summary 里两者分开写。

纯值,不 import carla/torch;产物经 paths.project_path 落 outputs/。

每次运行落 `logs/` 三件套(脚本/时间/argv/cwd/git/GPU + 输入产物 sha256 + 结论数字),
关键帧循环的累计计数逐段 `metric()` 进 `.jsonl`(逐行 flush)—— 这条链是**分钟到小时级**
且产物只在末尾落盘,中断的跑法靠 `.jsonl` 才能判"跑到哪、卡在哪一档"。`--no-runlog` 关。

用法:
  python -m autodrivedata.slam.slam_backend [--traj outputs/slam/traj_raw.json]
                             [--root outputs/kitti_drive] [--out outputs/slam]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from autodrivedata.slam.accum import voxel_downsample
from autodrivedata.slam.core import (
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
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

# 回环候选的**几何先验闸**(米):前端轨迹给出的两关键帧距离超过它就**不进 ICP**。
#
# **为什么必须有这道闸**:ICP 的成本随失配**爆炸**而非线性增长 —— 实测(12.8k 点云,
# 同一份点云平移后配准):对齐 0.16 s / 10 m **6.8 s** / 40 m **92.7 s** / 100 m 142.6 s。
# 失配云不重叠 ⇒ 迭代跑满且对应搜索退化。而描述子阈值(0.10)只保证"场景像",
# 远处对照样进来(top-5 里常占多数)⇒ 不加闸时 351 个候选 = **小时级**(实测 30 min
# 只跑完不到 25/81 个关键帧)。这条成本路径此前从未被走到:旧阈值在体素口径下恒 0 候选。
#
# **5.0 是量出来的,不是拍的**(802 帧闭环数据、351 个 SC 候选、真值取 `training/pose/`):
#   真回环(GT <2 m,76 个)的前端距离 = **3.05–4.15 m(中位 4.02)** ← 前端在重访处的累积误差
#   闸 2.0 m:放行 0,真对**误杀 76**;  闸 3.0 m:放行 64,真对**误杀 76**
#   闸 5.0 m:放行 196,真对**误杀 0**,远对(GT >15 m)漏进 0   ← 选它
#   闸 20.0 m:放行 333,真对误杀 0,远对漏进 1,但白跑 137 次分钟级 ICP
# ⇒ 闸必须**大于真对的最大前端距离(4.15 m)**且**远低于近邻带**;5.0 对两端都留了余量。
#
# **口径**:只用它**拒**候选,不用它接受任何东西(接受仍由 SC + ICP 双门决定)。
LOOP_PRIOR_MAX_M = 5.0

# **进 ICP 前的廉价筛**(抽稀查询云 + 只跑 1 次迭代)。overlap = "查询点里有近邻的比例",
# 抽稀是它的**无偏估计**,故可在 1/8 的点上量。
#
# 实测(同一批 76 个真回环):抽稀 1/8 筛值 min **0.187** / p10 0.241 / 中位 0.354 ⇒ 0.10 闸
# **0 误杀**(余量 1.9×);而 180° 错分支的筛值恒 **0.01–0.03** ⇒ 一次迭代就出局。
#
# **它省的是什么**:整条链路的成本主项不是"迭代次数",是**失配云上的第一次迭代** ——
# 无近邻 ⇒ `nearest_batch` 的网格早停失效(全层全点),实测 82–125 s/次;而对齐云 0.16 s。
# 套上筛之后错分支 0.2 s 出局(比 82–125 s 省 2–3 个数量级),真分支只多花 0.2 s。
#
# **口径**:筛只用来**跳过某个 yaw 分支**,不参与接受判据(接受仍是 SC + 全量 ICP 双门)。
# **三条已试过、都不行的廉价代理**(勿重走):SC 描述子距离(真对 0.034–0.092 vs 中段
# 0.014–0.099,中段最小值比真对还小)、2 m 粗格占格重叠(真对 0.086–0.180 vs 远对
# 0.001–0.191,区间重叠)、全量云首迭代 overlap(虽能分开,但一次 1.5–18 s,省不了钱)。
LOOP_SCREEN_STRIDE = 8
LOOP_SCREEN_OVERLAP = 0.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="outputs/slam/traj_raw.json", help="阶段 1 轨迹(读,经 cwd)")
    ap.add_argument("--root", default="outputs/kitti_drive", help="KITTI root(velodyne 所在)")
    ap.add_argument("--out", default="outputs/slam", help="输出根(经 project_path)")
    ap.add_argument("--every", type=int, default=KEYFRAME_EVERY, help="关键帧间隔(帧)")
    ap.add_argument("--min-gap", type=int, default=SC_MIN_GAP_NODES, help="回环候选最小关键帧号差")
    ap.add_argument("--no-runlog", action="store_true", help="不落 logs/ 三件套(默认每次运行都落)")
    args = ap.parse_args()

    with runlog.run("autodrivedata.slam.slam_backend") as rl:
        run(args, rl)


def run(args: argparse.Namespace, rl: runlog.RunLogger) -> None:
    """后端主体。抽出来只为让 `main()` 能用 `with runlog.run(...)` 包住全程。"""
    t0 = time.time()
    traj_doc = json.loads(Path(args.traj).read_text())
    frames = [int(f["frame"]) for f in traj_doc["traj"]]
    poses = [np.array(f["T"], dtype=np.float64) for f in traj_doc["traj"]]
    root = Path(args.root)
    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rl.input(args.traj, "slam-traj")
    rl.input(args.root, "kitti-root")

    # --- 关键帧 + 描述子 ---
    kf_idx = list(range(0, len(frames), args.every))
    descs: list[np.ndarray] = []
    kf_clouds: list[np.ndarray] = []
    for i in kf_idx:
        p = root / "training" / "velodyne" / f"{frames[i]:06d}.bin"
        pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
        # ★ 描述子吃**原始点云**,体素云只喂 ICP —— 两个消费者的降采样需求相反:
        #   ScanContext 是**计数直方图**(0.5 m 体素把每格打到 ~10 点、非空格 56%,
        #   Poisson 噪声 ~30% 直接淹掉信号)。实测 802 帧闭环数据、同一地点(帧 300↔700,
        #   GT 距 0.20 m):全点云距 0.039 / 体素后 0.314,而"同一帧不同降采样实现"之间
        #   就有 0.51 ⇒ 真值对全 > SC_SIM_THRESH ⇒ **候选 0**(静默:不报错,只是永不闭环)。
        #   回归钉:tests/slam/test_slam.py 的 AST 判据。
        kf_clouds.append(voxel_downsample(pts, 0.5))
        descs.append(desc_scan_context(pts))
    descs_arr = np.stack(descs)
    print(f"[kf] {len(kf_idx)} 关键帧(每 {args.every} 帧)| 描述子 {descs_arr.shape}")

    # --- 回环候选 + 双 yaw ICP 验证 ---
    # 进度可见(这个循环在闭环数据上是**分钟级到半小时级**的黑盒:351 候选 × 双 yaw,
    # 且坏候选要跑满 ICP_MAX_ITER —— 与收敛候选的成本差一个量级,静默等待无法判断死活)。
    t_loop = time.time()
    n_cand = n_ok = n_prior = n_screened = n_short = 0
    loops: list[dict] = []
    for n, i in enumerate(kf_idx):
        if n % 25 == 0 and n:
            print(
                f"  [loop] {n}/{len(kf_idx)} 关键帧 | SC 候选 {n_cand}(先验拒 {n_prior}"
                f" / 筛出局 {n_screened} / 短路 {n_short})过门 {n_ok} | 耗时 {time.time() - t_loop:.0f}s"
            )
            # 与上面那行 print 同频:卡住时判"卡在第几档"(先验拒/筛出局/短路/全量 ICP)
            rl.metric(
                n,
                kf_done=n,
                sc_candidates=n_cand,
                prior_rejected=n_prior,
                screened_out=n_screened,
                short_circuit=n_short,
                full_icp=len(loops),
                accepted=n_ok,
                loop_s=round(time.time() - t_loop, 1),
            )
        cands = sc_candidates(
            descs_arr, descs_arr[n], n, top_n=SC_TOP_N, min_gap=args.min_gap, thresh=SC_SIM_THRESH
        )
        n_cand += len(cands)
        for c, shift, dist in cands:
            j = kf_idx[c]
            # ★ 几何先验闸(见 LOOP_PRIOR_MAX_M):前端位姿已说两者隔很远 ⇒ 不进 ICP。
            # 必须在 ICP **之前**:失配候选的 ICP 成本是收敛候选的几十到上千倍。
            prior_m = float(np.linalg.norm(poses[i][:3, 3] - poses[j][:3, 3]))
            if prior_m > LOOP_PRIOR_MAX_M:
                n_prior += 1
                continue
            # 双 yaw 初值:ScanContext 列滚动不变 → 对 180° 旋转有歧义。shift 是描述子
            # 最佳列滚动量(= yaw 的粗估,60 扇 → 6°/扇),仅记录不参与几何(几何由 ICP 定)
            best = None
            branches: list[dict] = []
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
                seed = np.linalg.inv(guess_map)
                # ① 廉价筛(见 LOOP_SCREEN_*):抽稀查询云 + 1 次迭代,挡在贵路径之前
                scr = icp_odometry(
                    kf_clouds[c][::LOOP_SCREEN_STRIDE, :3],
                    kf_clouds[n][:, :3],
                    np.eye(4),
                    seed=seed,
                    max_iter=1,
                )
                if scr["overlap"] < LOOP_SCREEN_OVERLAP:
                    branches.append({"yaw_deg": float(np.degrees(yaw)), "skip": "screened"})
                    continue
                # ② 全量 ICP(接受判据只用它)
                res = icp_odometry(kf_clouds[c][:, :3], kf_clouds[n][:, :3], np.eye(4), seed=seed)
                res["yaw"] = float(np.degrees(yaw))
                res["screen_overlap"] = float(scr["overlap"])
                branches.append(res)
                if best is None or res["rmse_final"] < best["rmse_final"]:
                    best = res
                # ③ **短路**:yaw=0 那支已过门 ⇒ 反极分支不再跑。依据:一份点云不可能同时与 0°
                # 和 180° 对齐(实测过门候选的反极支 overlap ≤0.03 / rmse ≥2.4),而反极支是
                # 分钟级开销。**接受判据没动**:反极支仍在"第一支没过门"时才跑(覆盖反向重访)。
                if yaw == 0.0 and (
                    res["overlap"] >= LOOP_GATE_OVERLAP
                    and res["rmse_final"] < LOOP_GATE_RMSE
                    and (res["converged"] or not LOOP_GATE_CONVERGED)
                ):
                    n_short += 1
                    break
            if best is None:  # 两支都被筛掉 ⇒ 这个候选不进 loops(只计数)
                n_screened += 1
                continue
            ok = (
                best["overlap"] >= LOOP_GATE_OVERLAP
                and best["rmse_final"] < LOOP_GATE_RMSE
                and (best["converged"] or not LOOP_GATE_CONVERGED)
            )
            n_ok += int(ok)
            loops.append(
                {
                    "kf_i": n,
                    "kf_j": c,
                    "frame_i": frames[i],
                    "frame_j": frames[j],
                    "sc_dist": round(dist, 5),
                    "sc_shift": int(shift),
                    "prior_m": round(prior_m, 3),
                    "screen_overlap": round(best["screen_overlap"], 4),
                    "skipped_branches": [b_["skip"] for b_ in branches if "skip" in b_],
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
    print(
        f"[loop] SC 候选 {n_cand} → 先验拒 {n_prior} / 筛出局 {n_screened} → 全量 ICP {len(loops)}"
        f" → 过门 {len(accepted)}"
    )

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
        "n_loop_candidates": len(loops),  # 进过全量 ICP 的候选(先验闸 ∧ 筛之后)
        "n_candidates_sc": n_cand,  # 描述子阈值给出的候选(先验闸之前)
        "n_prior_rejected": n_prior,  # 被几何先验拒掉、未进 ICP 的
        "n_screened_out": n_screened,  # 两支 yaw 都被廉价筛出局、未进全量 ICP 的
        "n_short_circuit": n_short,  # yaw=0 支已过门 ⇒ 反极支未跑(次数)
        "loop_prior_max_m": LOOP_PRIOR_MAX_M,
        "loop_screen_stride": LOOP_SCREEN_STRIDE,
        "loop_screen_overlap": LOOP_SCREEN_OVERLAP,
        "n_loops": len(accepted),
        "closure_pre": ce_pre,
        "closure_post": ce_post,
        "wall_s": round(time.time() - t0, 2),
        "note": (
            "开放路径下 closure_* 是首末位姿距离(= 路径长度量级),非漂移率;"
            "n_loops=0 是直线段数据的合法结果(不造回环)。"
            "闭环路径上 closure_* 才是真漂移指标。"
            f"n_candidates_sc → n_loop_candidates 的差由 LOOP_PRIOR_MAX_M={LOOP_PRIOR_MAX_M}m "
            "几何先验 + 廉价筛(LOOP_SCREEN_*,见注释;它只跳过 yaw 分支,不改接受判据)共同砍掉。"
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
            {
                "n_candidates": len(loops),
                "n_candidates_sc": n_cand,
                "n_prior_rejected": n_prior,
                "n_screened_out": n_screened,
                "n_short_circuit": n_short,
                "n_accepted": len(accepted),
                "loops": loops,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    (out / "slam_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(
        f"[done] 回环 {len(accepted)}/{len(loops)} | 闭合 pre {ce_pre['drift_m']:.3f}m → post {ce_post['drift_m']:.3f}m"
    )
    print(f"  耗时 {summary['wall_s']}s → {out}/traj_pgo.json")

    for name, kind in (
        ("traj_pgo.json", "slam-traj-pgo"),
        ("loops.json", "slam-loops"),
        ("slam_summary.json", "slam-summary"),
    ):
        rl.artifact(out / name, kind)
    rl.highlight("dataset", str(root))
    rl.highlight("n_frames", len(frames))
    rl.highlight("n_keyframes", len(kf_idx))
    rl.highlight("keyframe_every", args.every)
    rl.highlight("min_gap", args.min_gap)
    rl.highlight("n_candidates_sc", n_cand)
    rl.highlight("n_prior_rejected", n_prior)
    rl.highlight("n_screened_out", n_screened)
    rl.highlight("n_short_circuit", n_short)
    rl.highlight("n_loop_candidates", len(loops))
    rl.highlight("n_loops", len(accepted))
    rl.highlight("closure_pre_drift_m", round(ce_pre["drift_m"], 3))
    rl.highlight("closure_post_drift_m", round(ce_post["drift_m"], 3))
    rl.highlight("wall_s", summary["wall_s"])
    # 这两个常量决定候选被砍多少 ⇒ `n_loops` 跨版本比较前先看它们(与 AP 看 batch 同理)
    rl.highlight("loop_prior_max_m", LOOP_PRIOR_MAX_M)
    rl.highlight("loop_screen_overlap", LOOP_SCREEN_OVERLAP)
    rl.note(
        "closure_* 在开放路径上 = 首末位姿距离(≠ 漂移);n_loops=0 是直线段数据的合法结果。"
        "n_candidates_sc → n_loop_candidates 的差 = 几何先验闸(LOOP_PRIOR_MAX_M)+ 廉价筛"
    )


if __name__ == "__main__":
    main()

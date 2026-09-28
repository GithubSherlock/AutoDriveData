#!/usr/bin/env python3
"""全能力面验收编排器 —— 一条命令把"这套流水线还活着"重新跑成证据(2026-09-28)。

**它解决什么**:验收的结论散在 18 个 runlog 入口 + 24 个采集器 + 若干可视化脚本里,
每次都要照 CLAUDE.md 的命令清单手敲一遍,敲漏一条没人知道。本脚本把**这次实际跑过、
且逐条验过**的命令固化成阶段,并把新出的图与数字收进 `outputs/showcase/`。

**它不是测试**:真正的回归在 `python -m pytest`(阶段 `baseline` 调它)。
本脚本管的是"端到端还跑不跑得动" —— pytest 全绿而某个 CLI 入口早已因外部依赖缺失
而跑不起来,是完全可能的(2026-09-28 实测:`sem_bev` 就是这么坏的,且零覆盖)。

用法:
  python tools/showcase.py --phase baseline          # ruff + pytest 全量(不需 CARLA)
  python tools/showcase.py --phase offline           # 复用已有 outputs 的离线入口
  python tools/showcase.py --phase online            # 采集器 + 需 CARLA 的探针(**先启服务器**)
  python tools/showcase.py --phase figures           # 只用已有产物出图(不跑任何入口)
  python tools/showcase.py --phase all               # 依次全跑
  python tools/showcase.py --list                    # 只列命令,不执行

**阶段间没有隐式依赖**:`figures` 只读盘上已有的产物,单独跑得动。
`online` 会**改写** CARLA 当前所在的地图与场景(collect_traj 落 Town13 等),跑完记得:
  bash tools/carla_server.sh stop && bash tools/carla_server.sh start   # 回到默认图
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SHOW = PROJECT / "outputs" / "showcase"
LOG = SHOW / "logs"


@dataclass
class Cmd:
    label: str
    argv: list[str]
    note: str = ""
    #: 该命令的产物/结论落在哪 —— 报表里给读者一条可点的线索
    artifact: str = ""


@dataclass
class Phase:
    name: str
    needs_carla: bool
    cmds: list[Cmd] = field(default_factory=list)


PY = sys.executable

# ---------------------------------------------------------------- 阶段定义
# 每条都对应 2026-09-28 验收里**真跑过并核对过数字**的一次调用;改这里前先真跑一遍。

OFFLINE = Phase(
    "offline",
    needs_carla=False,
    cmds=[
        Cmd(
            "eval_2d_ab_sunset",
            [
                PY,
                "-m",
                "autodrivedata.perception.eval_2d_ab",
                "--root-a",
                "outputs/kitti_ab_day_clear",
                "--root-b",
                "outputs/kitti_ab_sunset_glare",
            ],
            "P1 逆光 A/B:期望 ΔmAP ≈ −0.014",
            "logs/perception_eval_2d_ab_*.json",
        ),
        Cmd(
            "eval_2d_ab_rain",
            [
                PY,
                "-m",
                "autodrivedata.perception.eval_2d_ab",
                "--root-a",
                "outputs/kitti_ab_day_clear",
                "--root-b",
                "outputs/kitti_ab_rain_night",
            ],
            "P1 雨夜 A/B:期望 ΔmAP ≈ −0.153(漏检型)",
            "logs/perception_eval_2d_ab_*.json",
        ),
        Cmd(
            "eval_2d_ab_fog",
            [
                PY,
                "-m",
                "autodrivedata.perception.eval_2d_ab",
                "--root-a",
                "outputs/kitti_ab_day_clear",
                "--root-b",
                "outputs/kitti_ab_dense_fog",
            ],
            "P1 浓雾 A/B:期望 ΔmAP ≈ −0.013(FP 型)",
            "logs/perception_eval_2d_ab_*.json",
        ),
        Cmd(
            "eval_attr",
            [
                PY,
                "-m",
                "autodrivedata.perception.eval_attr",
                "--run",
                "day8=outputs/kitti_sweep_day_clear_8:8.0",
                "--run",
                "rain=outputs/kitti_ab_rain_night:8.0",
                "--run",
                "fog=outputs/kitti_ab_dense_fog:8.0",
                "--json",
                "outputs/showcase/02_p1_corner/attr.json",
            ],
            "失效归因:尺度断崖(<32px vs ≥32px)",
            "outputs/showcase/02_p1_corner/attr.json",
        ),
        Cmd(
            "eval_kitti_3d",
            [
                PY,
                "-m",
                "autodrivedata.perception.eval_kitti",
                "--root",
                "outputs/kitti_ab_day_clear",
                "--pred",
                "outputs/kitti3d_ab_day_clear",
            ],
            "LiDAR 3D 基线(天气不敏感的那一侧)",
            "logs/perception_eval_kitti_*.json",
        ),
        Cmd(
            "eval_maptr_frame_holdout",
            [
                PY,
                "-m",
                "autodrivedata.map.eval_maptr",
                "--infos",
                "outputs/surround_v2/map_infos.json",
                "--root",
                "outputs/surround_v2",
                "--ckpt",
                "outputs/maptr_v2_singleF.pt",
                "--exclude-seg",
                "seg4",
                "--keep-in-seg",
                "80:100",
                "--score-thr",
                "0.2",
                "--sweep",
                "0.1,0.2,0.3,0.4",
                "--out-pred",
                "outputs/showcase/03_map/frame_holdout",
            ],
            "MapTR 帧级留出:归档 0.3043,实测 0.3048(±5e-4 是复现性下限)",
            "outputs/showcase/03_map/frame_holdout.png",
        ),
        Cmd(
            "eval_maptr_route_holdout",
            [
                PY,
                "-m",
                "autodrivedata.map.eval_maptr",
                "--infos",
                "outputs/surround_v2/map_infos.json",
                "--root",
                "outputs/surround_v2",
                "--ckpt",
                "outputs/maptr_v2_singleF.pt",
                "--seg",
                "seg4",
                "--score-thr",
                "0.2",
                "--out-pred",
                "outputs/showcase/03_map/route_holdout",
            ],
            "MapTR 路线级留出:归档 0.1114,实测 0.1112",
            "outputs/showcase/03_map/route_holdout.png",
        ),
        Cmd(
            "viz_maptr_pred",
            [
                PY,
                "-m",
                "autodrivedata.map.viz_maptr_pred",
                "--infos",
                "outputs/surround_v2/map_infos.json",
                "--root",
                "outputs/surround_v2",
                "--ckpt",
                "outputs/maptr_v2_singleF.pt",
                "--start",
                "250",
                "--frames",
                "6",
                "--score-thr",
                "0.2",
                "--out-dir",
                "outputs/showcase/03_map/viz_pred",
            ],
            "预测回投 6 相机 + BEV 目检图",
            "outputs/showcase/03_map/viz_pred/",
        ),
        Cmd(
            "export_mapvec_3fmt",
            [PY, "-m", "autodrivedata.map.export_mapvec", "--map", "Town10HD_Opt", "--lanelet2"],
            "三格式出口(opendrive/lanelet2/apollo);读回用 --from",
            "training/map/",
        ),
        Cmd(
            "probe_mapvec_oracle",
            [PY, "-m", "autodrivedata.map.probe_mapvec_oracle", "--map", "Town10HD_Opt", "--samples", "200"],
            "A6 几何对账 vs CARLA 运行时:容差 5cm(**需 CARLA**)",
            "logs/map_probe_mapvec_oracle_*.json",
        ),
        Cmd(
            "slam_eval_pre",
            [
                PY,
                "-m",
                "autodrivedata.slam.eval_slam",
                "--traj",
                "outputs/showcase/04_slam/run/traj_raw.json",
                "--gt",
                "outputs/kitti_loop",
                "--out",
                "outputs/showcase/04_slam/eval_pre.json",
            ],
            "SLAM 前端 ATE:归档 1.4127m",
            "outputs/showcase/04_slam/eval_pre.json",
        ),
        Cmd(
            "slam_eval_post",
            [
                PY,
                "-m",
                "autodrivedata.slam.eval_slam",
                "--traj",
                "outputs/showcase/04_slam/run/traj_pgo.json",
                "--gt",
                "outputs/kitti_loop",
                "--out",
                "outputs/showcase/04_slam/eval_post.json",
            ],
            "SLAM PGO 后 ATE:归档 0.1352m",
            "outputs/showcase/04_slam/eval_post.json",
        ),
        Cmd(
            "slam_diff_test",
            [PY, "-m", "autodrivedata.slam.slam_diff_test"],
            "Python vs C++ ICP 9/9 对拍",
            "logs/slam_slam_diff_test_*.json",
        ),
        Cmd(
            "cluster_obstacles",
            [
                PY,
                "-m",
                "autodrivedata.perception.cluster_obstacles",
                "--root",
                "outputs/kitti_drive",
                "--frames",
                "0-149",
            ],
            "DBSCAN 聚类:归档 117.97 簇/帧",
            "outputs/cluster/summary.json",
        ),
        Cmd(
            "extract_ground",
            [
                PY,
                "-m",
                "autodrivedata.perception.extract_ground",
                "--root",
                "outputs/kitti_drive",
                "--frames",
                "0-149",
            ],
            "地面提取:均值占比 ≈57%",
            "outputs/ground/stats.json",
        ),
        Cmd(
            "build_accum_map",
            [
                PY,
                "-m",
                "autodrivedata.slam.build_accum_map",
                "--root",
                "outputs/kitti_drive",
                "--frames",
                "0-149",
            ],
            "累积建图 → 全局语义点云 PLY",
            "outputs/accum_map/map.ply",
        ),
        Cmd(
            "calib_multilidar",
            [PY, "-m", "autodrivedata.calib.calib_multilidar"],
            "多雷达 ICP 判据:小误差收敛 / 大误差不收敛",
            "outputs/multilidar/icp_result.json",
        ),
        Cmd(
            "sem_bev",
            [
                PY,
                "-m",
                "autodrivedata.perception.sem_bev",
                "--root",
                "outputs/surround_v2/seg0",
                "--out",
                "outputs/showcase/05_perception/sem_bev",
                "--frames",
                "0-5",
                "--gpu",
            ],
            "语义 BEV(**2026-09-28 修好**:原先被一句外部 utils 惰性 import 卡死)",
            "outputs/showcase/05_perception/sem_bev/",
        ),
        Cmd(
            "assemble_traj_hivt",
            [
                PY,
                "-m",
                "autodrivedata.traj.assemble_traj_pt",
                "--traj",
                "outputs/traj_town13/traj.json",
                "--map",
                "Town13",
                "--out",
                "outputs/showcase/08_traj/processed",
                "--steps",
                "5",
            ],
            "CARLA 轨迹 → HiVT TemporalData",
            "outputs/showcase/08_traj/processed/processed/",
        ),
    ],
)

#: 需 CARLA。**顺序有讲究**:采集器会切图,故需 CARLA 的探针排在最后,
#: 且跑完要把服务器停/启一次回到默认图(见模块 docstring)。
ONLINE = Phase(
    "online",
    needs_carla=True,
    cmds=[
        Cmd(
            "smoke",
            [PY, "-m", "autodrivedata.sim.smoke", "--out", "outputs/showcase/09_online/smoke"],
            "M0:连通性 / 同步模式 / 相机 + LiDAR 各取一帧",
            "outputs/showcase/09_online/smoke/",
        ),
        Cmd(
            "collect_kitti",
            [
                PY,
                "-m",
                "autodrivedata.sim.collect_kitti",
                "--out",
                "outputs/showcase/09_online/kitti_static",
                "--frames",
                "10",
            ],
            "M1a 静态采集",
            "outputs/showcase/09_online/kitti_static/",
        ),
        Cmd(
            "collect_drive",
            [
                PY,
                "-m",
                "autodrivedata.sim.collect_drive",
                "--scene",
                "day_clear",
                "--out",
                "outputs/showcase/09_online/kitti_drive",
                "--frames",
                "20",
            ],
            "M2 动态采集(autopilot + 车流 + 行人)",
            "outputs/showcase/09_online/kitti_drive/",
        ),
        Cmd(
            "collect_ab_route",
            [
                PY,
                "-m",
                "autodrivedata.sim.collect_ab_route",
                "--scene",
                "day_clear",
                "--out",
                "outputs/showcase/09_online/kitti_ab",
                "--frames",
                "20",
            ],
            "P1 A/B 采集:锚定起点 + 4 路肩车",
            "outputs/showcase/09_online/kitti_ab/",
        ),
        Cmd(
            "collect_static_gt",
            [
                PY,
                "-m",
                "autodrivedata.sim.collect_static_gt",
                "--frames",
                "10",
                "--out",
                "outputs/showcase/09_online/kitti_static_gt",
            ],
            "P2 静态 GT:信号/标志 + 车道线 + overlay",
            "outputs/showcase/09_online/kitti_static_gt/",
        ),
        Cmd(
            "collect_tl_states",
            [
                PY,
                "-m",
                "autodrivedata.sim.collect_tl_states",
                "--frames",
                "30",
                "--speed",
                "8",
                "--cycle",
                "6,2,6",
                "--out",
                "outputs/showcase/09_online/kitti_tl",
            ],
            "灯色动态 GT:受控切灯 = 确定性变灯序列",
            "outputs/showcase/09_online/kitti_tl/",
        ),
        Cmd(
            "collect_nus",
            [
                PY,
                "-m",
                "autodrivedata.sim.collect_nus",
                "--frames",
                "2",
                "--out",
                "outputs/showcase/09_online/nus_mini",
            ],
            "nuScenes 迷你集:6 相机 + LiDAR + 5 雷达",
            "outputs/showcase/09_online/nus_mini/",
        ),
        Cmd(
            "collect_nus_wide",
            [
                PY,
                "-m",
                "autodrivedata.sim.collect_nus",
                "--rig",
                "wide",
                "--frames",
                "2",
                "--out",
                "outputs/showcase/09_online/nus_mini_wide",
            ],
            "wide rig(挂点后移,画幅内零车体像素)",
            "outputs/showcase/09_online/nus_mini_wide/",
        ),
        Cmd(
            "collect_stereo",
            [
                PY,
                "-m",
                "autodrivedata.sim.collect_stereo",
                "--frames",
                "10",
                "--out",
                "outputs/showcase/09_online/stereo",
            ],
            "双目采集(基线 0.4m)+ 真值深度",
            "outputs/showcase/09_online/stereo/",
        ),
        Cmd(
            "collect_surround_micro_legacy",
            [
                PY,
                "-m",
                "autodrivedata.sim.collect_surround_micro",
                "--out",
                "outputs/showcase/01_calib/micro_legacy",
                "--cam-back",
                "legacy",
                "--frames",
                "10",
            ],
            "布局对照的 A 段(旧后相机布局)",
            "outputs/showcase/01_calib/micro_legacy/",
        ),
        Cmd(
            "collect_surround_micro_official",
            [
                PY,
                "-m",
                "autodrivedata.sim.collect_surround_micro",
                "--out",
                "outputs/showcase/01_calib/micro_official",
                "--cam-back",
                "nuscenes",
                "--frames",
                "10",
            ],
            "布局对照的 B 段(官方后相机布局)",
            "outputs/showcase/01_calib/micro_official/",
        ),
        Cmd(
            "assemble_micro_legacy",
            [
                PY,
                "-m",
                "autodrivedata.map.assemble_maptr",
                "--surround",
                "outputs/showcase/01_calib/micro_legacy",
                "--map-json",
                "training/map/Town10HD_Opt_full.json",
                "--out",
                "outputs/showcase/01_calib/micro_legacy/map_infos.json",
            ],
            "viz_layout_cmp 的前置:两段都要有 map_infos.json",
            "map_infos.json",
        ),
        Cmd(
            "assemble_micro_official",
            [
                PY,
                "-m",
                "autodrivedata.map.assemble_maptr",
                "--surround",
                "outputs/showcase/01_calib/micro_official",
                "--map-json",
                "training/map/Town10HD_Opt_full.json",
                "--out",
                "outputs/showcase/01_calib/micro_official/map_infos.json",
            ],
            "同上",
            "map_infos.json",
        ),
        Cmd(
            "viz_layout_cmp",
            [
                PY,
                "-m",
                "autodrivedata.calib.viz_layout_cmp",
                "--a",
                "outputs/showcase/01_calib/micro_legacy",
                "--b",
                "outputs/showcase/01_calib/micro_official",
                "--out",
                "outputs/showcase/01_calib",
            ],
            "布局对照:同 GT 在两代 rig 上的逐相机覆盖量",
            "outputs/showcase/01_calib/",
        ),
        Cmd(
            "collect_traj",
            [
                PY,
                "-m",
                "autodrivedata.sim.collect_traj",
                "--map",
                "Town13",
                "--frames",
                "60",
                "--out",
                "outputs/showcase/08_traj/carla_traj",
            ],
            "HiVT 轨迹源(多 agent)",
            "outputs/showcase/08_traj/carla_traj/",
        ),
        Cmd(
            "live_studio_video",
            [
                PY,
                "-m",
                "autodrivedata.sim.live_studio",
                "--map",
                "Town10HD_Opt",
                "--npcs",
                "--speed",
                "6",
                "--duration",
                "60",
                "--fps",
                "10",
                "--no-keyboard",
                "--maptr-ckpt",
                "outputs/maptr_v2_singleF.pt",
                "--slam",
                "--video",
                "outputs/showcase/09_online/studio/studio_8view.mp4",
                "--video-fps",
                "0.5",
                "--dump",
                "outputs/showcase/09_online/studio/frame.png",
            ],
            "8 路 studio + 在线 SLAM + 实时 MapTR;同时落八视角视频",
            "outputs/showcase/09_online/studio/studio_8view.mp4",
        ),
        # ---- 需 CARLA 的探针:排在**采集之后**,避免采集器切图影响(见模块 docstring)----
        Cmd(
            "probe_calib",
            [PY, "-m", "autodrivedata.calib.probe_calib"],
            "标定自证 A0–A6:把 CARLA 渲染器当第二把尺子",
            "outputs/calib_check/report.json",
        ),
        Cmd(
            "viz_rig_check_live_nuscenes",
            [
                PY,
                "-m",
                "autodrivedata.calib.viz_rig_check",
                "--rig",
                "nuscenes",
                "--live",
                "--out-dir",
                "outputs/showcase/01_calib",
            ],
            "六视角实拍 + 逐通道 ego 像素数(判据 ①)",
            "outputs/showcase/01_calib/views_nuscenes.png",
        ),
        Cmd(
            "viz_rig_check_live_wide",
            [
                PY,
                "-m",
                "autodrivedata.calib.viz_rig_check",
                "--rig",
                "wide",
                "--live",
                "--out-dir",
                "outputs/showcase/01_calib",
            ],
            "同上,wide rig",
            "outputs/showcase/01_calib/views_wide.png",
        ),
        Cmd(
            "verify_nus_calib_live_nuscenes",
            [
                PY,
                "-m",
                "autodrivedata.calib.verify_nus_calib",
                "--live",
                "--rig",
                "nuscenes",
                "--dataroot",
                "outputs/showcase/09_online/nus_mini",
                "--out",
                "outputs/showcase/01_calib/report_live_nuscenes.json",
            ],
            "十条判据(**必须在默认图 Town10HD_Opt 上跑**,判据 6/8 是场景相关的)",
            "outputs/showcase/01_calib/report_live_nuscenes.json",
        ),
        Cmd(
            "verify_nus_calib_live_wide",
            [
                PY,
                "-m",
                "autodrivedata.calib.verify_nus_calib",
                "--live",
                "--rig",
                "wide",
                "--dataroot",
                "outputs/showcase/09_online/nus_mini_wide",
                "--out",
                "outputs/showcase/01_calib/report_live_wide.json",
            ],
            "十条判据,wide rig",
            "outputs/showcase/01_calib/report_live_wide.json",
        ),
    ],
)

PHASES = {p.name: p for p in (OFFLINE, ONLINE)}


# ---------------------------------------------------------------- 执行
def run_phase(phase: Phase, dry: bool) -> dict:
    LOG.mkdir(parents=True, exist_ok=True)
    results = []
    for c in phase.cmds:
        line = " ".join(c.argv[2:]) if c.argv[:2] == [PY, "-m"] else " ".join(c.argv)
        print(f"\n{'#' * 5} {c.label}\n  $ {line}\n  ↳ {c.note}")
        if dry:
            results.append({"label": c.label, "status": "dry"})
            continue
        t0 = time.time()
        p = subprocess.run(c.argv, cwd=PROJECT, capture_output=True, text=True)
        # ⚠️ 不接 `| tail`:管道会把真实退出码吃成 tail 的 0(2026-09-28 踩过)
        (LOG / f"{c.label}.log").write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8")
        ok = p.returncode == 0
        print(f"  exit={p.returncode} ({time.time() - t0:.0f}s)" + ("" if ok else "  ← 失败,见日志"))
        results.append(
            {
                "label": c.label,
                "status": "ok" if ok else "fail",
                "exit": p.returncode,
                "wall_s": round(time.time() - t0, 1),
                "artifact": c.artifact,
                "note": c.note,
            }
        )
    return {"phase": phase.name, "results": results}


# ---------------------------------------------------------------- 出图
def fig_p1_ab() -> str:
    """P1 A/B:同一帧号下四个天气的对照条 —— GT 框画上去,证明是同一批配对帧。"""
    from PIL import Image, ImageDraw

    from autodrivedata.utils import fonts

    scenes = ["day_clear", "sunset_glare", "rain_night", "dense_fog"]
    fid = 30
    tiles = []
    for sc in scenes:
        root = PROJECT / "outputs" / f"kitti_ab_{sc}" / "training"
        img = Image.open(root / "image_2" / f"{fid:06d}.png").convert("RGB")
        d = ImageDraw.Draw(img)
        lab = root / "label_2" / f"{fid:06d}.txt"
        n = 0
        if lab.exists():
            for ln in lab.read_text().splitlines():
                f = ln.split()
                if not f or f[0] != "Car":
                    continue
                x1, y1, x2, y2 = (float(v) for v in f[4:8])
                d.rectangle([x1, y1, x2, y2], outline=(255, 0, 255), width=2)
                n += 1
        # 文字走 fonts(**不许裸写 d.text**):PIL 遇缺字静默画 .notdef 方框
        fonts.draw_text(d, (8, 8), f"{sc}  Car GT={n}", 20, fill=(255, 255, 0))
        tiles.append(img)
    W, H = tiles[0].size
    out = Image.new("RGB", (W, H * len(tiles)), (0, 0, 0))
    for i, t in enumerate(tiles):
        out.paste(t, (0, i * H))
    p = SHOW / "02_p1_corner" / f"ab_side_by_side_f{fid:06d}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    out.save(p)
    return str(p.relative_to(PROJECT))


def fig_slam_traj() -> str:
    """SLAM:XY 轨迹(pre/post vs GT) + 逐帧位置误差。读 tray_*.json 与 KITTI pose。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def xy_of(traj_json: Path) -> np.ndarray:
        """traj_json 的 `T` 是 **LiDAR 系**位姿 —— 必须先 `lidar_pose_to_ego` 换算。

        **不换算会得到一张骗人的图**:两组轨迹手性相反(`det(U·Vᵀ) = −1`),
        Umeyama 永远对不上(裸对齐 rmse 50 m),画出来像是"定位全错"。
        换算是 `L·M·T·M·inv(L)`(手性共轭 + 杆臂,**端序不能凭直觉**),
        唯一实现在 `slam_eval.lidar_pose_to_ego` —— 这里只许复用,不许重写。
        """
        from autodrivedata.slam.slam_eval import lidar_pose_to_ego

        doc = json.loads(traj_json.read_text())
        out = []
        for r in doc["traj"]:
            e = lidar_pose_to_ego(np.asarray(r["T"], dtype=np.float64))
            out.append([e[0][3], e[1][3], e[2][3]])
        return np.array(out)

    def align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """3D 相似变换对齐(Umeyama)—— **必须做,且必须与 `eval_slam` 同口径**。

        两件事都不做会得到一张骗人的图:
        ① **不对齐**:轨迹差一个固定的原点/朝向,叠画出来是"两条形状相同、位置差几十米
           的轨迹",看图的人只会以为定位崩了(而 ATE 本身是**对齐后**才算的);
        ② **只对齐 xy**:`ATE` 是 **3D** 的。实测 pre 的 3D rmse 1.4126 m 里
           **xy 只占 0.273 m、z 占 1.386 m** —— 也就是前端误差的 98% 是**垂直漂移**。
           只画 xy 会让人以为"这 SLAM 挺好",与引用的 1.4126 对不上。
        """
        mu_s, mu_d = src.mean(0), dst.mean(0)
        s0, d0 = src - mu_s, dst - mu_d
        u, d, vt = np.linalg.svd(d0.T @ s0 / len(src))
        j = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
        r = u @ j @ vt
        var = (s0**2).sum() / len(src)
        sc = float((d * np.diag(j)).sum() / var)
        return (sc * (r @ s0.T)).T + mu_d

    def gt_xyz() -> np.ndarray:
        pts = []
        for f in sorted((PROJECT / "outputs" / "kitti_loop" / "training" / "pose").glob("*.txt")):
            v = [float(x) for x in f.read_text().split()]
            pts.append([v[3], v[7], v[11]])
        return np.array(pts)

    raw0 = xy_of(SHOW / "04_slam" / "run" / "traj_raw.json")
    pgo0 = xy_of(SHOW / "04_slam" / "run" / "traj_pgo.json")
    gt = gt_xyz()
    n = min(len(raw0), len(pgo0), len(gt))
    # 各自与 GT 对齐(3D,与 eval_slam 同口径),再画
    raw, pgo = align(raw0[:n], gt[:n]), align(pgo0[:n], gt[:n])

    def decompose(a: np.ndarray) -> tuple[float, float, float]:
        """→ (3D rmse, xy rmse, z rmse)。3D 那一项应当**逐位等于** `eval_slam` 报的 ATE。"""
        d = a - gt[:n]
        return (
            float(np.sqrt((d**2).sum(1).mean())),
            float(np.sqrt((d[:, :2] ** 2).sum(1).mean())),
            float(np.sqrt((d[:, 2] ** 2).mean())),
        )

    r3, rxy, rz = decompose(raw)
    p3, pxy, pz = decompose(pgo)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot(gt[:n, 0], gt[:n, 1], "k-", lw=2.4, label="GT (CARLA pose)")
    ax[0].plot(raw[:n, 0], raw[:n, 1], "-", color="tab:red", lw=1.4, label="frontend (raw ICP)")
    ax[0].plot(pgo[:n, 0], pgo[:n, 1], "-", color="tab:blue", lw=1.4, label="PGO (loop-closed)")
    ax[0].set_aspect("equal")
    ax[0].set_title(
        f"SLAM trajectory, {n} frames closed loop, aligned to GT (3D Umeyama)\n"
        f"ATE(3D): frontend {r3:.4f} m -> PGO {p3:.4f} m   (Plan2 P-H.3.4: 1.4127 -> 0.1352)"
    )
    ax[0].set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.3)

    ax[1].plot(np.linalg.norm(raw[:n, :2] - gt[:n, :2], axis=1), color="tab:red", lw=1.0, label="frontend xy")
    ax[1].plot(np.linalg.norm(pgo[:n, :2] - gt[:n, :2], axis=1), color="tab:blue", lw=1.0, label="PGO xy")
    ax[1].plot(np.abs(raw[:n, 2] - gt[:n, 2]), color="tab:red", lw=1.0, ls=":", label="frontend z")
    ax[1].plot(np.abs(pgo[:n, 2] - gt[:n, 2]), color="tab:blue", lw=1.0, ls=":", label="PGO z")
    ax[1].set_title(
        f"Error decomposition (aligned): frontend xy {rxy:.3f} / z {rz:.3f} m; PGO xy {pxy:.3f} / z {pz:.3f} m\n"
        f"-> {100 * rz**2 / r3**2:.0f}% of the pre-PGO 3D ATE is VERTICAL drift"
    )
    ax[1].set_xlabel("frame")
    ax[1].set_ylabel("|err| (m)")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.3)

    p = SHOW / "04_slam" / "traj_compare.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return str(p.relative_to(PROJECT))


def fig_hivt_scene() -> str:
    """HiVT:一个场景的 agent 历史/未来 + 车道线(证明组装产物形状对)。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    files = sorted((SHOW / "08_traj" / "processed" / "processed").glob("*.pt"))
    if not files:
        raise FileNotFoundError("先跑 assemble_traj_hivt")
    s = torch.load(files[0], weights_only=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    # ⚠️ `lane_vectors` 是 (N,2) 的**方向**向量,不是线段端点 —— 它是给编码器吃的,
    # 组装器**不落**车道几何(assemble_traj_pt 只写 lane_vectors,见其 docstring)。
    # 因此这里画不出车道底图;硬按端点解包会 `IndexError: invalid index to scalar`(踩过)。
    lv = s["lane_vectors"].numpy()
    pos = s["positions"].numpy()
    for i in range(pos.shape[0]):
        ax.plot(pos[i, :20, 0], pos[i, :20, 1], "-", color="tab:red", lw=1.6, zorder=3)
        ax.plot(pos[i, 19:, 0], pos[i, 19:, 1], "-", color="tab:blue", lw=1.6, zorder=3)
        ax.plot(pos[i, 19, 0], pos[i, 19, 1], "o", color="k", ms=4, zorder=4)
    ax.set_aspect("equal")
    ax.set_title(
        f"HiVT TemporalData scene 0 ({files[0].name})  {pos.shape[0]} agents, {lv.shape[0]} lane vectors\n"
        "red = 20 past / blue = 30 future, AV-centric frame (lane geometry not stored in this .pt)"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.3)
    p = SHOW / "08_traj" / "hivt_scene0.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return str(p.relative_to(PROJECT))


def copy_calib_rig_layouts() -> list[str]:
    """配置图是**纯值**产物(不需 CARLA),单独重出一份到 showcase。"""
    made = []
    for rig in ("nuscenes", "wide"):
        subprocess.run(
            [
                PY,
                "-m",
                "autodrivedata.calib.viz_rig_check",
                "--rig",
                rig,
                "--out-dir",
                "outputs/showcase/01_calib",
            ],
            cwd=PROJECT,
            capture_output=True,
            text=True,
        )
        made.append(f"outputs/showcase/01_calib/rig_layout_{rig}.png")
    return made


FIGS = {
    "p1_ab": fig_p1_ab,
    "slam_traj": fig_slam_traj,
    "hivt_scene": fig_hivt_scene,
}


def run_figures() -> dict:
    made, failed = [], []
    for name, fn in FIGS.items():
        try:
            p = fn()
            made.append(p)
            print(f"  [fig] {name} → {p}")
        except Exception as e:  # 缺前置产物是常见情形,记下来别中断
            failed.append({"fig": name, "error": f"{type(e).__name__}: {e}"})
            print(f"  [fig] {name} 失败: {type(e).__name__}: {e}")
    made += copy_calib_rig_layouts()
    return {"figures": made, "failed": failed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["baseline", "offline", "online", "figures", "all"], default="all")
    ap.add_argument("--list", action="store_true", help="只列命令,不执行")
    args = ap.parse_args()

    SHOW.mkdir(parents=True, exist_ok=True)
    report: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}

    if args.phase in ("baseline", "all"):
        for cmd in [
            Cmd("ruff_check", ["ruff", "check"]),
            Cmd("ruff_format_check", ["ruff", "format", "--check"]),
            Cmd("pytest", [PY, "-m", "pytest", "-q"]),
        ]:
            print(f"\n{'#' * 5} {cmd.label}")
            if not args.list:
                p = subprocess.run(cmd.argv, cwd=PROJECT, capture_output=True, text=True)
                (LOG / f"{cmd.label}.log").write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8")
                print((p.stdout or "").strip().splitlines()[-1] if p.stdout else "")
                report["runs"].append({"label": cmd.label, "exit": p.returncode})

    if args.phase in ("offline", "online", "all"):
        names = ["offline", "online"] if args.phase == "all" else [args.phase]
        for nm in names:
            ph = PHASES[nm]
            if ph.needs_carla and not args.list:
                print(f"\n⚠️ 阶段 {nm} 需 CARLA 服务器:bash tools/carla_server.sh start")
            report["runs"].append(run_phase(ph, args.list))

    if args.phase in ("figures", "all"):
        print(f"\n{'#' * 5} figures")
        report["figures"] = {"figures": [], "failed": []} if args.list else run_figures()

    if not args.list:
        (SHOW / "showcase_run.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"\n[report] {SHOW / 'showcase_run.json'}")


if __name__ == "__main__":
    main()

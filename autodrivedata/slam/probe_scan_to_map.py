"""B2 实测:scan-to-map(ikd-Tree + 局部地图前端)在本仿真数据上值不值得做。

**要回答的问题**:帧间 scan-to-scan 前端之外,再加一个 scan-to-map 前端(FAST-LIO2 的
ikd-Tree 路线),精度会不会更好?

**做法**:用 CARLA 真值位姿构造 **oracle 局部地图**(零里程计漂移)⇒ 测出的是 scan-to-map
的**收益上限**。地图 = 当前帧前 K 帧点云按 GT 摆到预测帧系(两侧都靠近各自传感器原点,
否则点面 G-N 的旋转绕世界原点、力臂 40~80 m 与平移强耦合而发散 —— 实测过 3.5 m 首迭代跳跃)。

**结论(2026-09-19,判 B2 不做)**——误差随地图深度 K **单调变差**,重新体素化救不回来
(全表由本脚本产出,`outputs/s2m_probe/s2m_probe.json`):

| K | 地图 | 点数 | s2m RMS | 相对 s2s | RMSE@ICP | RMSE@GT | 代价降幅 | inlier | 法向稳 | λ1 | 地面比 | 软向投影 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | s2s 基线 | — | 0.0382 | 1.00 | — | — | — | — | — | — | — | — |
| 1 | 拼接 | 5134 | 0.0401 | 1.05 | 0.0984 | 0.0996 | 0.0012 | 0.977 | 0.849 | 1.226e-1 | 0.555 | 0.0174 |
| 1 | 重新体素 | 4089 | 0.0388 | 1.02 | 0.1026 | 0.1038 | 0.0012 | 0.956 | 0.818 | 1.197e-1 | 0.554 | 0.0170 |
| 3 | 拼接 | 15432 | 0.0592 | 1.55 | 0.0562 | 0.0594 | 0.0032 | 0.995 | 0.946 | 1.163e-1 | 0.588 | 0.0347 |
| 3 | 重新体素 | 6628 | 0.0592 | 1.55 | 0.0641 | 0.0688 | 0.0047 | 0.985 | 0.878 | 1.173e-1 | 0.568 | 0.0360 |
| 8 | 拼接 | 41148 | 0.1048 | 2.75 | 0.0419 | 0.0468 | 0.0050 | 0.998 | 0.954 | 9.800e-2 | 0.605 | 0.0438 |
| 8 | 重新体素 | 9752 | 0.1245 | 3.26 | 0.0489 | 0.0629 | 0.0140 | 0.994 | 0.936 | 1.165e-1 | 0.575 | 0.0667 |

**为什么**:四条独立机制证据
1. **残差下降 ≠ 位姿正确**。K 越大 ICP 解处的点面 RMSE 越低(0.0984→0.0419),但拿 GT 点映射
   算的 RMSE 也同步变低(0.0996→0.0468),两者之差(ICP 实际"赚到"的代价降幅)只有
   0.0012→0.0050 m —— 而位姿误差从 0.040 涨到 0.105 m。**代价在解附近变平**:密集地图让每个
   源点都能在 2 m 门内找到"某个"近邻平面,残差机械地变小,但那是地图本身的几何性质,
   不是解更准。
2. **约束方向塌陷**。AᵀA/N(行 = [(p×n)ᵀ, nᵀ])最小特征值 λ1 随 K 从 1.23e-1 掉到 9.80e-2
   (重新体素化档稳在 ~1.17e-1);位姿误差在**最软特征向量平移块**上的投影 0.0174→0.0438 m
   (涨 2.5×),在**最硬方向**上恒 ≤0.0005 m。误差确实沿软方向滑走。
3. **内点被地面稀释**。|n_z|>0.9 的地面点在 2 m 门内占比 0.555→0.605,而地面法向在
   x/y/yaw 上**零信息**(只约束 z/roll/pitch)。额外帧贡献的主要是可滑动的面:它们压低残差、
   不开新约束方向。
4. **法向稳定性不是原因**:法向一致性(8 近邻法向夹角余弦)随 K **上升**(0.849→0.954),
   即"拼接地图法向变糊"假设**被否证**;重新体素化(标准做法)也不能改善(K=8 时 3.26× 反而更差)。

⇒ **帧间重叠 ~90% 时 scan-to-map 没有收益**(与 Plan2 §3 最初的判断一致),本仓不建
ikd-Tree + scan-to-map 前端。**本结论的边界**:单序列、Town10HD_Opt、自动驾驶 400 帧、
体素 0.5、oracle GT 位姿。**换到帧间重叠低的场景(高速、稀疏扫描、大转弯)结论可能翻转** ——
届时本脚本可直接复跑复核。

用法:
  python -m autodrivedata.slam.probe_scan_to_map --root outputs/kitti_slam --frames 0-399 --out outputs/s2m_probe
  python -m autodrivedata.slam.probe_scan_to_map --root <kitti root> --picks 40,70,100 --ks 1,3,8 --radius 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autodrivedata.gt.export.kitti import read_pose
from autodrivedata.slam.accum import voxel_downsample
from autodrivedata.slam.core import (
    GRID_CELL,
    _batch_knn,
    estimate_normals,
    estimate_transform_gn,
    icp_odometry,
    nearest_batch,
)
from autodrivedata.utils.paths import project_path

# CARLA(ego,y 右)→ KITTI(LiDAR,y 左)手性共轭:见 autodrivedata/calib.carla_lidar_to_velodyne
M_FLIP = np.diag([1.0, -1.0, 1.0, 1.0])

# 逐帧累加的量(键名与 s2m_probe.json 的 rows 字段一一对应)
ACC_KEYS = (
    "err",
    "rmse_icp",
    "rmse_gt",
    "inlier",
    "n_points",
    "lambda1",
    "condition",
    "proj_soft",
    "proj_hard",
    "ground",
    "stability",
)


def load_scan(root: Path, fid: int, voxel: float) -> np.ndarray:
    p = root / "training" / "velodyne" / f"{fid:06d}.bin"
    return voxel_downsample(np.fromfile(p, dtype=np.float32).reshape(-1, 4), voxel)[:, :3]


def load_gt(root: Path, fid: int) -> np.ndarray:
    """CARLA 系 GT 位姿 → LiDAR 系(手性共轭)。**帧级配对,缺帧直接报错不插值**。"""
    p = root / "training" / "pose" / f"{fid:06d}.txt"
    if not p.exists():
        raise FileNotFoundError(f"GT 缺帧 {p}(帧级配对不做插值)")
    return M_FLIP @ read_pose(p) @ M_FLIP


def icp_gated(
    src: np.ndarray, ref: np.ndarray, ref_n: np.ndarray, t_pred: np.ndarray, gate: float, max_iter: int = 15
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """带对应距离门的点面 ICP;src/ref 同系(预测帧系)。返回 (位姿, 点映射, inlier 比, RMSE)。

    `icp_odometry` 对**每个**源点强制取最近邻,地图只覆盖 30 m 而源云覆盖 100 m 时地图外的点
    会把解拽走(实测 overlap 0.006、误差 11~16 m)。FAST-LIO2 用 max correspondence distance
    处理这件事;scan-to-scan 前端不需要(相邻帧覆盖范围一致),故生产环没这个门。
    """
    cur, r_acc, t_acc = src.copy(), np.eye(3), np.zeros(3)
    frac, rmse = 0.0, float("nan")
    for _ in range(max_iter):
        idx, d2 = nearest_batch(ref, cur, GRID_CELL)
        keep = (d2 < gate**2) & (np.einsum("ij,ij->i", ref_n[idx], ref_n[idx]) > 0)
        frac = float(keep.mean())
        if int(keep.sum()) < 10:
            break
        res = np.abs(np.einsum("ij,ij->i", ref_n[idx][keep], cur[keep] - ref[idx][keep]))
        rmse = float(np.sqrt((res**2).mean()))
        rt, tt = estimate_transform_gn(cur[keep], ref[idx][keep], ref_n[idx][keep])
        cur = (rt @ cur.T).T + tt
        r_acc, t_acc = rt @ r_acc, rt @ t_acc + tt
        if np.linalg.norm(rt - np.eye(3)) + np.linalg.norm(tt) < 1e-5:
            break
    t_delta = np.eye(4)
    t_delta[:3, :3], t_delta[:3, 3] = r_acc, t_acc
    return t_pred @ t_delta, t_delta, frac, rmse


def rmse_at(src: np.ndarray, ref: np.ndarray, ref_n: np.ndarray, t_delta: np.ndarray, gate: float) -> float:
    """给定 src→ref 的点映射,算点面残差 RMSE(与求解同口径)。"""
    cur = (t_delta[:3, :3] @ src.T).T + t_delta[:3, 3]
    idx, d2 = nearest_batch(ref, cur, GRID_CELL)
    keep = (d2 < gate**2) & (np.einsum("ij,ij->i", ref_n[idx], ref_n[idx]) > 0)
    if int(keep.sum()) < 10:
        return float("nan")
    res = np.abs(np.einsum("ij,ij->i", ref_n[idx][keep], cur[keep] - ref[idx][keep]))
    return float(np.sqrt((res**2).mean()))


def revoxel(pts: np.ndarray, voxel: float) -> np.ndarray:
    """拼接后**重新体素化**(真实局部地图的标准做法);accum.voxel_downsample 取首点强度。"""
    return voxel_downsample(np.hstack([pts, np.ones((len(pts), 1))]), voxel)[:, :3]


def spectrum(
    ref: np.ndarray, ref_n: np.ndarray, src: np.ndarray, t_delta: np.ndarray, gate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """最终对齐处的 AᵀA/N 特征谱(升序)+ 特征向量 + 内点法向;行 A_i = [(p×n)ᵀ, nᵀ]。

    最小特征值 = 该地图给出的**最弱约束方向**;条件数 = 谱动态范围。行数归一化 ⇒ 跨 K 可比。
    """
    cur = (t_delta[:3, :3] @ src.T).T + t_delta[:3, 3]
    idx, d2 = nearest_batch(ref, cur, GRID_CELL)
    keep = (d2 < gate**2) & (np.einsum("ij,ij->i", ref_n[idx], ref_n[idx]) > 0)
    p, n = cur[keep], ref_n[idx][keep]
    a = np.hstack([np.cross(p, n), n])
    w, v = np.linalg.eigh(a.T @ a / len(a))
    return w, v, n


def build_map(
    clouds: dict[int, np.ndarray], gts: dict[int, np.ndarray], i: int, k: int, radius: float
) -> np.ndarray:
    """前 K 帧点云按 GT 摆到世界系,再裁到当前 GT 位置 radius 内。"""
    m = np.concatenate([(gts[j][:3, :3] @ clouds[j].T).T + gts[j][:3, 3] for j in range(i - k, i)])
    return m[np.linalg.norm(m - gts[i][:3, 3], axis=1) < radius]


def to_pred_frame(pts: np.ndarray, t_pred: np.ndarray) -> np.ndarray:
    """世界系 → 预测帧系(靠近传感器原点,G-N 才不发散)。"""
    inv = np.linalg.inv(t_pred)
    return (inv[:3, :3] @ pts.T).T + inv[:3, 3]


def run(args: argparse.Namespace) -> dict:
    root = project_path(args.root)
    lo, hi = (int(v) for v in args.frames.split("-"))
    picks = [int(v) for v in args.picks.split(",")] if args.picks else list(range(lo + 20, hi, 30))
    ks = tuple(int(v) for v in args.ks.split(","))
    need = sorted({j for i in picks for j in range(i - max(ks), i + 1)})
    clouds = {j: load_scan(root, j, args.voxel) for j in need}
    gts = {j: load_gt(root, j) for j in need}

    # 基线:scan-to-scan(生产口径 icp_odometry,源云裁到与地图同半径以对齐覆盖)
    e_ss = []
    for i in picks:
        src = clouds[i][np.linalg.norm(clouds[i], axis=1) < args.radius]
        prev = clouds[i - 1][np.linalg.norm(clouds[i - 1], axis=1) < args.radius]
        r = icp_odometry(prev, src, np.eye(4))
        p_s2s = gts[i - 1] @ np.linalg.inv(r["T_delta"])
        e_ss.append(float(np.linalg.norm(p_s2s[:3, 3] - gts[i][:3, 3])))
    base = float(np.sqrt(np.mean(np.square(e_ss))))

    rows = []
    for k in ks:
        for variant in ("raw", "revox"):
            acc: dict[str, list[float]] = {key: [] for key in ACC_KEYS}
            for i in picks:
                src = clouds[i][np.linalg.norm(clouds[i], axis=1) < args.radius]
                m = build_map(clouds, gts, i, k, args.radius)
                mm = m if variant == "raw" else revoxel(m, args.voxel)
                t_pred = gts[i - 1]
                ref = to_pred_frame(mm, t_pred)
                ref_n = estimate_normals(ref)
                p_s2m, t_delta, f_, r_ = icp_gated(src, ref, ref_n, t_pred, args.gate)
                acc["err"].append(float(np.linalg.norm(p_s2m[:3, 3] - gts[i][:3, 3])))
                acc["rmse_icp"].append(r_)
                acc["rmse_gt"].append(rmse_at(src, ref, ref_n, np.linalg.inv(t_pred) @ gts[i], args.gate))
                acc["inlier"].append(f_)
                acc["n_points"].append(len(mm))
                w, v, n = spectrum(ref, ref_n, src, t_delta, args.gate)
                acc["lambda1"].append(float(w[0]))
                acc["condition"].append(float(w[-1] / max(w[0], 1e-30)))
                d = p_s2m[:3, 3] - gts[i][:3, 3]
                acc["proj_soft"].append(abs(float(v[3:, 0] @ d)))
                acc["proj_hard"].append(abs(float(v[3:, 5] @ d)))
                acc["ground"].append(float((np.abs(n[:, 2]) > 0.9).mean()))
                # 法向一致性:每点到其 8 近邻法向夹角的余弦绝对值均值(越接近 1 越稳)
                knn, _ = _batch_knn(ref, ref, 8, GRID_CELL)
                acc["stability"].append(float(np.mean(np.abs(np.einsum("ij,ikj->ik", ref_n, ref_n[knn])))))
            rms = float(np.sqrt(np.mean(np.square(acc["err"]))))
            rows.append(
                {
                    "K": k,
                    "variant": variant,
                    "n_points": float(np.mean(acc["n_points"])),
                    "s2m_rms_m": rms,
                    "ratio_vs_s2s": rms / base,
                    "rmse_at_icp": float(np.nanmean(acc["rmse_icp"])),
                    "rmse_at_gt": float(np.nanmean(acc["rmse_gt"])),
                    "cost_gain_m": float(np.nanmean(acc["rmse_gt"]) - np.nanmean(acc["rmse_icp"])),
                    "inlier_frac": float(np.mean(acc["inlier"])),
                    "normal_stability": float(np.mean(acc["stability"])),
                    "lambda1": float(np.mean(acc["lambda1"])),
                    "condition": float(np.mean(acc["condition"])),
                    "proj_soft_m": float(np.mean(acc["proj_soft"])),
                    "proj_hard_m": float(np.mean(acc["proj_hard"])),
                    "ground_frac": float(np.mean(acc["ground"])),
                }
            )
    return {
        "root": str(root),
        "frames": args.frames,
        "picks": picks,
        "voxel": args.voxel,
        "radius": args.radius,
        "gate": args.gate,
        "s2s_rms_m": base,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/kitti_slam", help="KITTI root(需 training/{velodyne,pose})")
    ap.add_argument("--frames", default="0-399", help="序列范围(取 --picks 用)")
    ap.add_argument("--picks", default="40,70,100,130,160,200,240,270", help="采样帧(逗号分隔)")
    ap.add_argument("--ks", default="1,3,8", help="地图深度 K(逗号分隔)")
    ap.add_argument("--voxel", type=float, default=0.5)
    ap.add_argument("--radius", type=float, default=30.0, help="地图/源云裁剪半径(m)")
    ap.add_argument("--gate", type=float, default=2.0, help="对应距离门(m)")
    ap.add_argument("--out", default="outputs/s2m_probe", help="产物根(经 project_path)")
    args = ap.parse_args()

    res = run(args)
    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "s2m_probe.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"{len(res['picks'])} 帧 | {res['root']} | 裁 {res['radius']}m | 门 {res['gate']}m | 体素 {res['voxel']}m"
    )
    print(f"  scan-to-scan 位姿误差 RMS {res['s2s_rms_m']:.4f} m\n")
    print(
        f"{'K':>3}{'地图':>9}{'点数':>8}{'s2m RMS':>10}{'比值':>7}{'RMSE@ICP':>10}{'RMSE@GT':>9}"
        f"{'代价降幅':>9}{'inlier':>8}{'法向稳':>8}{'λ1':>10}{'地面比':>8}{'软向':>8}"
    )
    for r in res["rows"]:
        tag = "拼接" if r["variant"] == "raw" else "重新体素"
        print(
            f"{r['K']:>3}{tag:>9}{r['n_points']:>8.0f}{r['s2m_rms_m']:>10.4f}{r['ratio_vs_s2s']:>7.2f}"
            f"{r['rmse_at_icp']:>10.4f}{r['rmse_at_gt']:>9.4f}{r['cost_gain_m']:>9.4f}{r['inlier_frac']:>8.3f}"
            f"{r['normal_stability']:>8.3f}{r['lambda1']:>10.3e}{r['ground_frac']:>8.3f}{r['proj_soft_m']:>8.4f}"
        )
    print(f"\n[out] {out / 's2m_probe.json'}")


if __name__ == "__main__":
    main()

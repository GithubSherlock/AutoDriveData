"""SLAM 轨迹精度评估:LiDAR 系位姿 → ego 系 → ATE/RPE(对齐 evo/KITTI 口径)。

**为什么需要一层坐标换算**:`autodrivedata/slam/slam_odometry.py` 输出的是 **LiDAR 传感器系**位姿,
而 `training/pose/{fid}.txt` 存的是 **ego(车辆)系** GT 位姿。两者差两件事:

1. **手性/朝向约定**:velodyne 点云经 `carla_lidar_to_velodyne` 翻转 y →
   KITTI 约定(x 前 / y 左 / z 上);CARLA ego GT 是 y 右。换算 = 共轭
   `M·X·M`(`M = diag(1,−1,1,1)`,`M²=I`)。
2. **杆臂**:LiDAR 挂在 ego 系 `(1.2, 0, 1.65)`(`carla_common.SENSOR_OFFSET`)。
   `S = E·L` ⇒ `E = S·inv(L)`。

⇒ `ego_pose = M·T_lidar·M @ inv(L)`。实测这一步把 ATE 从 0.4586 m 降到 **0.1869 m**
(不补杆臂时,转向时 LiDAR 的 1.65 m 高度 + 1.2 m 前悬被当成刚体原点误差)。

**ATe 口径**:`ate(..., align=True, with_scale=True)` = Umeyama 对齐后逐帧位置 RMSE;
`scale` 是 est→gt 的尺度因子。纯直行序列的**绕轨迹轴旋转不可辨识**(见
`slam_eval` 模块 docstring),故 ATE 只反映位置;RPE 的旋转项在直行段不可信。

用法:
  python -m autodrivedata.slam.eval_slam --traj outputs/slam_gt/traj_raw.json \
      --gt outputs/kitti_slam --out outputs/slam_gt/eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autodrivedata.gt.export.kitti import pose_path, read_pose
from autodrivedata.slam.slam_eval import LIDAR_LEVER, M_FLIP, eval_trajectory, lidar_pose_to_ego
from autodrivedata.utils.paths import project_path


def load_gt(root: Path, frames: list[int]) -> list[np.ndarray]:
    """按帧号读 GT 位姿(走 `export.kitti` 的落盘口径);**帧级配对是硬门槛,缺帧直接报错不做插值**。"""
    gt = []
    for fid in frames:
        p = pose_path(root, f"{fid:06d}")
        if not p.exists():
            raise FileNotFoundError(f"GT 缺帧 {p}(帧级配对不做插值)")
        gt.append(read_pose(p))
    return gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="outputs/slam_gt/traj_raw.json", help="阶段 1 轨迹(读,经 cwd)")
    ap.add_argument("--gt", default="outputs/kitti_slam", help="GT KITTI root(pose/ 所在)")
    ap.add_argument("--out", default=None, help="评估结果落盘(经 project_path);默认不写")
    ap.add_argument("--no-lever", action="store_true", help="跳过杆臂补偿(仅诊断用)")
    args = ap.parse_args()

    doc = json.loads(Path(args.traj).read_text())
    frames = [int(f["frame"]) for f in doc["traj"]]
    est_lidar = [np.array(f["T"], dtype=np.float64) for f in doc["traj"]]
    gt = load_gt(Path(args.gt), frames)

    if args.no_lever:
        est = [M_FLIP @ T @ M_FLIP for T in est_lidar]
    else:
        est = [lidar_pose_to_ego(T) for T in est_lidar]

    res = eval_trajectory(est, gt)
    a = res["ate_aligned"]
    r = res["rpe"]

    print(f"[eval] {args.traj} | {len(frames)} 帧 | 杆臂 {'off' if args.no_lever else 'on'}")
    print(
        f"  ATE(对齐) {a['rmse_m']:.4f} m | 尺度 {a['scale']:.5f} | "
        f"mean {a['mean_m']:.4f} max {a['max_m']:.4f} final {a['final_m']:.4f}"
    )
    print(f"  ATE(不对齐) {res['ate_raw']['rmse_m']:.4f} m")
    for k, v in r.items():
        print(
            f"  RPE {k}: 平移 {v['trans_rmse_m']:.4f} m / 旋转 {v['rot_rmse_deg']:.4f}° "
            f"| 每米 {v['trans_per_m']:.6f}"
        )
    print(f"  GT 路径长 {r['d1']['gt_path_m']:.2f} m | 相对 {a['rmse_m'] / r['d1']['gt_path_m'] * 100:.3f}%")

    if args.out:
        out = project_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "traj_in": args.traj,
                    "gt_root": args.gt,
                    "n_frames": len(frames),
                    "lever_arm": None if args.no_lever else LIDAR_LEVER.tolist(),
                    "convention": "ego_pose = M·T_lidar·M @ inv(L), M = diag(1,-1,1,1)",
                    **res,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"  → {out}")


if __name__ == "__main__":
    main()

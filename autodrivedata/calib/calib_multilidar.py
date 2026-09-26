"""P-E 教程 15 多雷达标定判据评估:两片点云注入已知误差 → 判据数值。

工业口径:多雷达标定 = 求两片点云的刚体变换。本脚本对现成点云
(kitti_drive 两帧的 non-ground 点,复用 cluster 产物,也可用原始 velodyne)
注入已知 R,t 误差 → 自研 point-to-plane ICP 恢复 → 判据:
- 小误差(0.1rad/0.1m):判据**收敛**且恢复变换 ≈ 注入逆
- 大误差(1.2rad):判据**不收敛**(重叠度 < 0.6)

数据:两帧 = "两个雷达对同一场景的观测"(时间相邻帧重叠大)。
输出:outputs/multilidar/icp_result.json(两分支的判据 + 曲线)。

用法:
  python -m autodrivedata.calib.calib_multilidar [--frames 0,1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autodrivedata.multilidar import convergence_metrics, point_to_plane_icp
from autodrivedata.paths import project_path


def _load_frame(root: Path, frame: int) -> np.ndarray:
    """读 KITTI velodyne bin → 非地面点(直接取原始点,含路面也行,取前 3000 点提速)。"""
    pts = np.fromfile(root / f"{frame:06d}.bin", dtype=np.float32).reshape(-1, 4)
    # 简单去地面:z > 0.5 即非地面(与 extract_ground 口径接近;此处只是给 ICP 一个平面富场景)
    ng = pts[pts[:, 2] > 0.3]
    return ng[:3000]


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _apply(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = pts.copy()
    out[:, :3] = (R @ pts[:, :3].T).T + t
    return out


def run_calib(root: Path, frames: tuple[int, int]) -> dict:
    ref = _load_frame(root, frames[1])  # 参考雷达观测
    src_raw = _load_frame(root, frames[0])  # 待标定雷达观测(先小误差注入)
    out: dict = {"frames": list(frames), "cases": {}}
    for case, R_true, t_true in (
        ("small_error_0.1rad_0.1m", _rot_y(0.1), np.array([0.1, 0.05, -0.08])),
        ("large_error_1.2rad", _rot_y(1.2), np.array([2.0, 0.0, 1.0])),
    ):
        src = _apply(src_raw, R_true, t_true)
        res = point_to_plane_icp(src, ref, max_iter=40)
        cfg = convergence_metrics(res)
        out["cases"][case] = {
            "injected_R": R_true.tolist(),
            "injected_t": t_true.tolist(),
            "recovered_R": res["R"].tolist(),
            "recovered_t": res["t"].tolist(),
            "judge": cfg,
            "rmse_curve_head": res["rmse_curve"][:5],
            "rmse_curve_tail": res["rmse_curve"][-3:],
            "delta_curve_head": res["delta_curve"][:5],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="outputs/kitti_drive/training/velodyne")
    ap.add_argument("--frames", default="0,1")
    ap.add_argument("--json", default="outputs/multilidar/icp_result.json")
    args = ap.parse_args()

    root = project_path(args.root)
    f0, f1 = (int(x) for x in args.frames.split(","))
    out = run_calib(root, (f0, f1))
    out_path = project_path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("== 多雷达标定判据(注入已知误差)==")
    for case, c in out["cases"].items():
        j = c["judge"]
        print(
            f"{case}: {j['verdict']} (rmse尾 {j['rmse_final']}, overlap {j['overlap']}, iters {j['iters']})"
        )
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()

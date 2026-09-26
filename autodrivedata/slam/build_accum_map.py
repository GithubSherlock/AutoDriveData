"""累积语义点云建图(教程 11):kitti_drive 多帧 velodyne → 全局语义地图。

用法:
  python -m autodrivedata.slam.build_accum_map --root outputs/kitti_drive --frames 0-149
  python -m autodrivedata.slam.build_accum_map --root outputs/kitti_drive --frames 0-20 --stride 3

输出:
  outputs/accum_map/map.ply      # 体素下采样后的全局语义点云(PLY 二进制)
  outputs/accum_map/stats.json   # 统计:输入点数/累积点数/下采样点数/去重率/范围
  逐帧进度打印(每 20 帧)。

纯值,不 import carla;产物经 paths.project_path 落 outputs/。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autodrivedata.slam.accum import accumulate_global, colorize, voxel_downsample, write_ply
from autodrivedata.utils.paths import project_path


def parse_range(spec: str) -> list[int]:
    """'0-149' 或 '0,5,10' → 帧 id 列表。"""
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/kitti_drive", help="KITTI root(velodyne 所在)")
    ap.add_argument("--frames", default="0-149", help="帧范围 0-149 或逗号列表")
    ap.add_argument("--stride", type=int, default=1, help="帧间隔采样(降密;1=全用)")
    ap.add_argument("--voxel", type=float, default=0.2, help="体素边长(m)")
    ap.add_argument("--out", default="outputs/accum_map", help="输出根(经 project_path)")
    args = ap.parse_args()

    root = Path(args.root)
    frames = parse_range(args.frames)[:: args.stride]
    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[data] {root} | {len(frames)} 帧(step {args.stride}) | voxel {args.voxel}m")
    buf: list[np.ndarray] = []
    n_in = 0
    for fid in frames:
        p = root / "training" / "velodyne" / f"{fid:06d}.bin"
        if not p.exists():
            print(f"[skip] {p.name} 不存在")
            continue
        pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
        n_in += pts.shape[0]
        buf.append(pts)
        if (len(buf) % 20) == 0:
            print(f"[{len(buf)}/{len(frames)}] 累积中...")

    all_pts = accumulate_global(buf)
    n_all = all_pts.shape[0]
    down = voxel_downsample(all_pts, args.voxel)
    n_down = down.shape[0]
    stats = {
        "input_frames": len(buf),
        "input_points": int(n_in),
        "accum_points": int(n_all),
        "downsample_points": int(n_down),
        "downsample_voxel_m": args.voxel,
        "dedup_ratio": round(1.0 - n_down / max(n_all, 1), 4),
        "x_range": [float(down[:, 0].min()), float(down[:, 0].max())],
        "y_range": [float(down[:, 1].min()), float(down[:, 1].max())],
        "z_range": [float(down[:, 2].min()), float(down[:, 2].max())],
    }
    colored = colorize(down)
    ply_path = out / "map.ply"
    write_ply(ply_path, colored)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"[done] 地图 {n_down} 点(voxel {args.voxel}m)→ {ply_path}")
    print(f"  stats: {json.dumps(stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

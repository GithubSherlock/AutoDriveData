"""聚类障碍物检测(教程 13):地面分割后 → DBSCAN 聚类 → 3D bbox 输出。

用法:
  python -m autodrivedata.perception.cluster_obstacles --root outputs/kitti_drive --frames 0-149
  python -m autodrivedata.perception.cluster_obstacles --root outputs/kitti_drive --frames 0-19 --distance-scale 0.02

输出:
  outputs/cluster/boxes.json   # 逐帧簇列表(label/center/extent_half/n_points/min_dist)
  outputs/cluster/summary.json # 汇总(帧数/平均簇数/平均最大簇点数)
  进度打印(每 20 帧)。

纯值,不 import carla;产物经 paths.project_path 落 outputs/。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autodrivedata.paths import project_path
from autodrivedata.perception.cluster import cluster_boxes, dbscan
from autodrivedata.perception.ground import ransac_plane


def parse_range(spec: str) -> list[int]:
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
    ap.add_argument("--root", default="outputs/kitti_drive")
    ap.add_argument("--frames", default="0-149")
    ap.add_argument("--eps", type=float, default=0.8, help="DBSCAN 半径(m)")
    ap.add_argument("--min-samples", type=int, default=8, help="簇内最小点数")
    ap.add_argument("--distance-scale", type=float, default=0.0, help="远距自适应系数")
    ap.add_argument("--out", default="outputs/cluster")
    args = ap.parse_args()

    root = Path(args.root)
    frames = parse_range(args.frames)
    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_frame: dict[str, dict] = {}
    for i, fid in enumerate(frames):
        p = root / "training" / "velodyne" / f"{fid:06d}.bin"
        if not p.exists():
            continue
        pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
        # 地面去除(RANSAC)
        res = ransac_plane(pts, seed=0)
        if res is None:
            continue
        _, mask = res
        nonground = pts[~mask]
        if len(nonground) < args.min_samples:
            continue
        labels = dbscan(
            nonground, eps=args.eps, min_samples=args.min_samples, distance_scale=args.distance_scale
        )
        boxes = cluster_boxes(nonground, labels)
        per_frame[f"{fid:06d}"] = {
            "nonground_points": int(len(nonground)),
            "n_clusters": len(boxes),
            "boxes": boxes,
        }
        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{len(frames)}] {fid} {len(boxes)} 簇")

    n_clusters = [v["n_clusters"] for v in per_frame.values()]
    summary = {
        "frames": len(per_frame),
        "eps": args.eps,
        "min_samples": args.min_samples,
        "mean_clusters_per_frame": round(sum(n_clusters) / max(len(n_clusters), 1), 2),
        "max_cluster_points": max((b["n_points"] for v in per_frame.values() for b in v["boxes"]), default=0),
    }
    (out / "boxes.json").write_text(json.dumps(per_frame, indent=2, ensure_ascii=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] {len(per_frame)} 帧聚类 → {out}")
    print(f"  均值簇数/帧 {summary['mean_clusters_per_frame']} | 最大簇 {summary['max_cluster_points']} 点")


if __name__ == "__main__":
    main()

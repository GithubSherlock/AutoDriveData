"""地面提取(教程 12):kitti_drive 逐帧 velodyne → 地面/非地面分离 + 统计。

用法:
  python -m autodrivedata.perception.extract_ground --root outputs/kitti_drive --frames 0-149
  python -m autodrivedata.perception.extract_ground --root outputs/kitti_drive --frames 0-19 --method ransac

输出:
  outputs/ground/{fid}_ground.bin   # 地面点(KITTI velodyne 约定,可入后续处理)
  outputs/ground/{fid}_nonground.bin  # 非地面点(障碍)
  outputs/ground/stats.json         # 逐帧统计 + 汇总(平面参数/占比/倾角)
  逐帧进度打印。

纯值,不 import carla;产物经 paths.project_path 落 outputs/。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autodrivedata.paths import project_path
from autodrivedata.perception.ground import grid_ground, ground_stats, ransac_plane


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
    ap.add_argument("--root", default="outputs/kitti_drive", help="KITTI root")
    ap.add_argument("--frames", default="0-149")
    ap.add_argument("--method", default="ransac", choices=["ransac", "grid"])
    ap.add_argument("--out", default="outputs/ground")
    args = ap.parse_args()

    root = Path(args.root)
    frames = parse_range(args.frames)
    out = project_path(args.out)
    (out / "ground").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i, fid in enumerate(frames):
        p = root / "training" / "velodyne" / f"{fid:06d}.bin"
        if not p.exists():
            continue
        pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
        if args.method == "grid":
            mask = grid_ground(pts)
            plane = None
        else:
            res = ransac_plane(pts, seed=0)
            if res is None:
                print(f"[{fid}] 无平面,跳过")
                continue
            plane, mask = res
        pts.tofile(out / "ground" / f"{fid:06d}_ground.bin")
        pts[~mask].tofile(out / "ground" / f"{fid:06d}_nonground.bin")
        st = ground_stats(pts, mask)
        st["frame"] = fid
        st["plane"] = None if plane is None else [float(v) for v in plane]
        rows.append(st)
        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{len(frames)}] {fid} 地面 {st['ground_ratio']:.2%}")

    summary = {
        "frames": len(rows),
        "method": args.method,
        "mean_ground_ratio": round(sum(r["ground_ratio"] for r in rows) / max(len(rows), 1), 4),
    }
    # 补充倾角
    for r in rows:
        if r.get("plane"):
            a, b, _ = r["plane"]
            r["plane_angle_deg"] = round(float(np.degrees(np.arccos(1.0 / np.sqrt(a * a + b * b + 1.0)))), 2)
    (out / "stats.json").write_text(
        json.dumps({"frames": rows, "summary": summary}, indent=2, ensure_ascii=False)
    )
    print(f"[done] {len(rows)} 帧地面提取 → {out}")
    print(f"  均值地面占比 {summary['mean_ground_ratio']:.2%}")


if __name__ == "__main__":
    main()

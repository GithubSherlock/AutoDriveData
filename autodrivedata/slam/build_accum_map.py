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


def render_bev(pts: np.ndarray, rgba: np.ndarray, out_png: Path, stats: dict, cell: float = 0.2) -> Path:
    """全局语义点云 → BEV 俯视图(每格取**平均语义色**,空网格黑)。

    **不是装饰**:画上帧数/点数/体素/包络,看图 ≈ 读 stats.json。
    逐格用 `np.bincount` 聚合(N=200 万点时逐点画像素要几分钟,聚合是毫秒级)。
    """
    from PIL import Image, ImageDraw

    from autodrivedata.utils import fonts

    xy = pts[:, :2].astype(np.float64)
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    nx, ny = int((x1 - x0) / cell) + 1, int((y1 - y0) / cell) + 1
    ix = np.clip(((xy[:, 0] - x0) / cell).astype(np.int64), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - y0) / cell).astype(np.int64), 0, ny - 1)
    flat = ix * ny + iy
    n = nx * ny
    cnt = np.bincount(flat, minlength=n).astype(np.float64)
    acc = np.zeros((n, 3), dtype=np.float64)
    for c in range(3):
        acc[:, c] = np.bincount(flat, weights=rgba[:, c].astype(np.float64), minlength=n)
    hit = cnt > 0
    acc[hit] /= cnt[hit, None]
    # 行序翻转:x 前向画成"上";**空网格留黑**(密度低的地方本来就该是空的,不补色)
    img = acc.reshape(nx, ny, 3)[::-1]
    canvas = Image.fromarray(img.astype(np.uint8))
    head = (
        f"累积语义点云 BEV  {stats['input_frames']} 帧 / {stats['downsample_points']} 点"
        f"(voxel {stats['downsample_voxel_m']} m)  包络 "
        f"{stats['x_range'][1] - stats['x_range'][0]:.0f}×{stats['y_range'][1] - stats['y_range'][0]:.0f} m"
    )
    out = Image.new("RGB", (canvas.width, canvas.height + 32), (0, 0, 0))
    out.paste(canvas, (0, 32))
    fonts.draw_text(ImageDraw.Draw(out), (8, 6), head, size=20, fill=(235, 235, 235))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_png)
    return out_png


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
    # **JSON/PLY 之外同时落一张图**(用户口径 2026-09-28:检图必须有可视化可比对)。
    # PLY 要外部工具才看得开,BEV 图开箱即看。
    png_path = out / "map_bev.png"
    render_bev(down, colored, png_path, stats)

    print(f"[done] 地图 {n_down} 点(voxel {args.voxel}m)→ {ply_path}")
    print(f"  stats: {json.dumps(stats, ensure_ascii=False)}")
    print(f"  [viz] {png_path}")


if __name__ == "__main__":
    main()

"""B3 验收探针:地图矢量投影回 6 视角图像 vs 渲染一致性(数值诊断)。

对指定帧,把 annotation 四类折线点(ego 局部系)→ 世界 → 各相机像素,统计:
- 图像内投影点占比(可见性,环视盲区应有损失)
- 图像内投影点的"路面性"占比:像素非天空(天空 = 亮蓝/白)的比例。
  内外参或坐标系错位 → 投影点系统性落到天空/建筑 → 路面性崩溃,无需目检
  (C23 口径:数值诊断,不做视觉回归)。

用法:
  python bin/probe_mapvec_proj.py --infos outputs/surround_drive/map_infos.json \
      --imgs outputs/surround_drive --frame 0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image

from autodrivedata.calib import CameraIntrinsics, world_to_img


def sky_pixel(px: tuple[int, int, int]) -> bool:
    """天空判据:亮蓝(b-r>25 且 b>140)或高亮云/白(全通道 >220)。"""
    r, g, b = px
    return (b > 140 and b - r > 25) or (r > 220 and g > 220 and b > 220)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--imgs", required=True, help="B1 采集根目录(相对 data_path 的基目录)")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=400, help="每帧采样折线点数(均匀抽样)")
    args = ap.parse_args()

    infos = json.loads(Path(args.infos).read_text(encoding="utf-8"))
    info = next(x for x in infos if x["frame"] == args.frame)
    root = Path(args.imgs)
    eg = info["ego2global"]
    yaw_e = math.radians(eg[3])

    # ego 局部系折线点 → 世界(逆 A5 变换)
    ann = info["annotation"]
    pts_local: list[tuple[float, float, float]] = []
    for cls in ("divider", "ped_crossing", "boundary", "centerline"):
        for line in ann[cls]:
            pts_local.extend((x, y, 0.0) for x, y in line)
    # 均匀抽样(避免顺序偏差 + 控制量)
    step = max(1, len(pts_local) // args.n_samples)
    pts_local = pts_local[::step]
    pts_world = _ego_to_world(pts_local, eg)
    print(f"帧 {args.frame}: 抽样折线点 {len(pts_world)}(全 6 相机投影诊断)")

    for cam_name, cam in info["cams"].items():
        se = cam["sensor2ego"]
        yaw_c = math.radians(eg[3] + se[3])
        loc_cam = (
            eg[0] + math.cos(yaw_e) * se[0],
            eg[1] + math.sin(yaw_e) * se[0],
            eg[2] + se[2],
        )
        k = CameraIntrinsics(width=1242, height=375, fov_h_deg=90.0)  # B1 CAM_ATTRS 口径(1242x375 fov 90)
        img = Image.open(root / cam["data_path"]).convert("RGB")
        hits = 0
        roady = 0
        for p in pts_world:
            uv = world_to_img(p, loc_cam, (0.0, yaw_c, 0.0), k)
            if uv is None:
                continue
            hits += 1
            u, v = int(uv[0]), int(uv[1])
            px = img.getpixel((u, v))
            if isinstance(px, tuple) and len(px) == 3 and not sky_pixel((px[0], px[1], px[2])):
                roady += 1
        rate = hits / len(pts_world)
        road_rate = roady / hits if hits else 0.0
        print(
            f"{cam_name:16s} 图像内 {hits:4d}/{len(pts_world)} ({rate * 100:4.1f}%) | 路面性 {road_rate * 100:4.1f}%"
        )


def _ego_to_world(pts: list[tuple[float, float, float]], eg: list[float]) -> list[tuple[float, float, float]]:
    """ego 局部系 → CARLA 世界(与 mapvec.from_ego_frame 同式,避免构造 MapVec)。"""
    a = math.radians(eg[3])
    c, s = math.cos(a), math.sin(a)
    return [(c * x - s * y + eg[0], s * x + c * y + eg[1], z + eg[2]) for x, y, z in pts]


if __name__ == "__main__":
    main()

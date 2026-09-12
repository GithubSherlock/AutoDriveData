"""A4:地图矢量导出——`training/map/{map}_full.json` + BEV overlay + 可选帧级裁剪版。

全图版折线保持全精度(供目检/oracle/转换器);帧级版按 ego 方形窗口 ±51.2m
裁剪后等距重采样 20 点(MapTR 固定点数口径)。落盘坐标系 = CARLA 世界系
(xodr y 取反,Unreal 左手系),与 B 阶段采集 ego pose 同系,JSON meta.frame 标注。

用法(帧级裁剪版在 --ego 后追加 --fid,±51.2m 窗口 + 20 点重采样):
  python bin/export_mapvec.py --map Town10HD_Opt
  python bin/export_mapvec.py --map Town10HD_Opt --ego 45.6 -23.1 --fid 000000
"""

from __future__ import annotations

import argparse
import glob
import os

from PIL import Image, ImageDraw

from autodrivedata.mapvec import crop_to_ego, extract_mapvec, resample, to_carla, vecs_dump
from autodrivedata.opendrive import parse_xodr
from autodrivedata.paths import project_path

XODR_GLOB = "/root/autodl-tmp/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps/**/*.xodr"
OUT_DIR = "training/map"
# BEV overlay 类色(避 C23 撞色口径:divider 橙/stop_line 红/ped 绿/boundary 蓝)
CLS_COLOR = {
    "divider": (255, 128, 0),
    "boundary": (0, 120, 255),
    "ped_crossing": (0, 200, 80),
    "stop_line": (230, 0, 0),
    "centerline": (160, 160, 160),
    "traffic_light": (200, 0, 220),
}


def find_xodr(name: str) -> str:
    hits = glob.glob(os.path.join(os.path.dirname(XODR_GLOB), f"**/{name}.xodr"), recursive=True)
    if not hits:
        raise SystemExit(f"找不到 {name}.xodr({XODR_GLOB})")
    return hits[0]


def bounds_of(vecs) -> tuple[float, float, float, float]:
    xs = [p[0] for v in vecs for p in v.points]
    ys = [p[1] for v in vecs for p in v.points]
    return min(xs), min(ys), max(xs), max(ys)


def render_bev(vecs, out_png: str, pad: float = 20.0, max_side: int = 2400) -> None:
    """BEV overlay:世界 (x, y) → 图像(x - xmin, ymax - y);忽略 z。"""
    xmin, ymin, xmax, ymax = bounds_of(vecs)
    if xmax - xmin < 1e-6 or ymax - ymin < 1e-6:
        raise SystemExit("矢量包络退化,无有效折线")
    xmin, ymin, xmax, ymax = xmin - pad, ymin - pad, xmax + pad, ymax + pad
    scale = min(max_side / (xmax - xmin), max_side / (ymax - ymin))
    w, h = max(1, int((xmax - xmin) * scale)), max(1, int((ymax - ymin) * scale))
    img = Image.new("RGB", (w, h), (15, 15, 15))
    d = ImageDraw.Draw(img)
    lw = max(2, int(scale * 0.25))
    for v in vecs:
        if len(v.points) < 2:
            continue
        px = [(int((p[0] - xmin) * scale), int((ymax - p[1]) * scale)) for p in v.points]
        d.line(px, fill=CLS_COLOR.get(v.cls, (255, 255, 255)), width=lw)
        if v.cls == "traffic_light":
            d.ellipse([px[0][0] - 2, px[0][1] - 2, px[0][0] + 2, px[0][1] + 2], fill=CLS_COLOR[v.cls])
    img.save(out_png)
    print(f"overlay → {out_png} ({w}x{h}, scale={scale:.2f} px/m)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="Town10HD_Opt")
    ap.add_argument("--ego", nargs=2, type=float, metavar=("X", "Y"), help="帧级裁剪中心(CARLA 系)")
    ap.add_argument("--fid", default="000000", help="帧级文件名片段")
    ap.add_argument("--radius", type=float, default=51.2)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    args.out = str(project_path(args.out))  # 产物锚定项目根(相对路径不随 cwd 漂移)

    vecs = to_carla(extract_mapvec(parse_xodr(find_xodr(args.map))))
    os.makedirs(args.out, exist_ok=True)

    full_json = os.path.join(args.out, f"{args.map}_full.json")
    with open(full_json, "w", encoding="utf-8") as f:
        f.write(vecs_dump(vecs, args.map, "carla_world"))
    n = sum(1 for v in vecs if v.cls != "traffic_light")
    print(f"{full_json}:{len(vecs)} 实例({n} 折线)")
    render_bev(vecs, os.path.join(args.out, f"{args.map}_full.png"))

    if args.ego:
        cropped = tuple(
            resample(v, 20) for v in crop_to_ego(vecs, (args.ego[0], args.ego[1]), radius=args.radius)
        )
        fid_json = os.path.join(args.out, f"{args.map}_{args.fid}.json")
        with open(fid_json, "w", encoding="utf-8") as f:
            f.write(vecs_dump(cropped, args.map, "carla_world"))
        print(f"{fid_json}:{len(cropped)} 实例(裁剪 ±{args.radius}m 后)")
        render_bev(cropped, os.path.join(args.out, f"{args.map}_{args.fid}.png"))


if __name__ == "__main__":
    main()

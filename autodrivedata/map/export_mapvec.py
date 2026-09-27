"""A4:地图矢量导出——`training/map/{map}_full.json` + BEV overlay + 可选帧级裁剪版。

全图版折线保持全精度(供目检/oracle/转换器);帧级版按 ego 方形窗口 ±51.2m
裁剪后等距重采样 20 点(MapTR 固定点数口径)。落盘坐标系 = CARLA 世界系
(xodr y 取反,Unreal 左手系),与 B 阶段采集 ego pose 同系,JSON meta.frame 标注。

**三格式出口(2026-09-27)**:默认 = 上面这份 JSON(`opendrive` 口径,`vecs_dump` **一字不改**);
点名 `--lanelet2` / `--apollo` 则**额外**写一份对应格式的地图(见
[`lanelet2.py`](lanelet2.py) / [`apollo.py`](apollo.py))。两开关**互斥**(同时给报错),
不做"谁覆盖谁"的隐式规则。`--from` 是入口侧的反向开关:三种格式都能**当 GT 源读回**。

⛔ **`--apollo` 有一处语义降级**:Apollo 的 `Map` 没有通用折线要素,`divider` / `boundary` /
`centerline` 借 `lane.central_curve` 承载(**把标线当车道**)。详见 `apollo.py` 模块头注。

用法(帧级裁剪版在 --ego 后追加 --fid,±51.2m 窗口 + 20 点重采样):
  python -m autodrivedata.map.export_mapvec --map Town10HD_Opt
  python -m autodrivedata.map.export_mapvec --map Town10HD_Opt --ego 45.6 -23.1 --fid 000000
  python -m autodrivedata.map.export_mapvec --map Town10HD_Opt --lanelet2
  python -m autodrivedata.map.export_mapvec --map Town10HD_Opt --apollo
  python -m autodrivedata.map.export_mapvec --map Town10HD_Opt --from lanelet2   # 读回当 GT 源
"""

from __future__ import annotations

import argparse
import glob
import os

from PIL import Image, ImageDraw

from autodrivedata.map.apollo import dump_apollo, load_apollo
from autodrivedata.map.lanelet2 import dump_lanelet2, load_lanelet2
from autodrivedata.map.mapvec import (
    MAP_FORMATS,
    crop_to_ego,
    extract_mapvec,
    resample,
    to_carla,
    vecs_dump,
)
from autodrivedata.map.opendrive import parse_xodr
from autodrivedata.map.stitch import parse_stitch_spec, seams, stitch
from autodrivedata.utils.paths import project_path

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


FORMATS = MAP_FORMATS  # 格式名表的唯一落点在 mapvec.py
# 每种格式的落盘后缀;`opendrive` 不在表内 —— 它就是上面那份 JSON(默认路径)
MAP_SUFFIX: dict[str, str] = {"lanelet2": ".osm", "apollo": ".txt"}
_LOAD: dict[str, object] = {"lanelet2": load_lanelet2, "apollo": load_apollo}


def dump_map_text(vecs, fmt: str, map_name: str = "") -> str:
    """`MapVec` 元组 → 该格式的文本。**唯一落点**。

    `export_mapvec` 与 `eval_maptr` 的逐帧地图出口都调它 —— 格式→写出器的映射
    若各写一份,两份迟早漂(本仓"AP 口径只能有一份实现"是同一道理)。
    默认的 `opendrive` **没有**文本写出器:它就是 `vecs_dump` 的 JSON。
    """
    if fmt == "lanelet2":
        return dump_lanelet2(vecs, map_name)
    if fmt == "apollo":
        return dump_apollo(vecs, map_name)
    raise ValueError(f"{fmt!r} 没有文本写出器(opendrive 走 vecs_dump 的 JSON)")


def load_map_text(text: str, fmt: str):
    """该格式的文本 → `MapVec` 元组(写读两端各自的表,见 `_LOAD` 的注)。"""
    return _LOAD[fmt](text)


def _resolve_format(args) -> str | None:
    """两个格式开关 → 出口格式;都不给 = None(只写默认 JSON,现行为)。**互斥报错**。"""
    if args.lanelet2 and args.apollo:
        raise SystemExit("--lanelet2 与 --apollo 互斥:同时给时「谁覆盖谁」没有好默认 —— 要两份就分两次跑")
    if args.lanelet2:
        return "lanelet2"
    if args.apollo:
        return "apollo"
    return None


def _write_format(vecs, fmt: str, base: str, map_name: str = "") -> None:
    """把 `vecs` 按 `fmt` 写到 `base + 后缀`。"""
    path = base + MAP_SUFFIX[fmt]
    with open(path, "w", encoding="utf-8") as f:
        f.write(dump_map_text(vecs, fmt, map_name))
    print(f"{path}:{len(vecs)} 实例({fmt} 口径)")


def _load_one(args, name: str) -> tuple:
    """按 `--from` 取**一张图**的矢量。`opendrive` = 读 CARLA 的 .xodr(即原来的路径)。"""
    if args.from_ == "opendrive":
        return to_carla(extract_mapvec(parse_xodr(find_xodr(name))))
    path = os.path.join(args.out, f"{name}_full{MAP_SUFFIX[args.from_]}")
    with open(path, encoding="utf-8") as f:
        return load_map_text(f.read(), args.from_)


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
    ap.add_argument(
        "--lanelet2", action="store_true", help="额外写 Lanelet2 .osm(标线级;见 lanelet2.py 头注)"
    )
    ap.add_argument(
        "--apollo",
        action="store_true",
        help="额外写 Apollo text-format protobuf(⚠️ divider/boundary/centerline 借 lane 承载,见 apollo.py 头注)",
    )
    ap.add_argument(
        "--stitch",
        default=None,
        help="跨图拼接:`A=0,0,0,0;B=420,0,0,90` 或 placements.json(见 stitch.py;"
        "⚠️ 官方 Town 无真值相对位姿,placement 是**人为摆位**;⚠️ 合并图对 MapTR 训练无用)",
    )
    ap.add_argument(
        "--from",
        dest="from_",  # `from` 是 Python 关键字,属性名必须换(`args.from` 根本写不出来)
        choices=FORMATS,
        default="opendrive",
        help="矢量**来源**格式(默认 CARLA 的 .xodr);三种格式都能当 GT 源读回",
    )
    args = ap.parse_args()
    args.out = str(project_path(args.out))  # 产物锚定项目根(相对路径不随 cwd 漂移)
    fmt = _resolve_format(args)  # 出口格式:None = 只写默认 JSON(现行为)

    os.makedirs(args.out, exist_ok=True)
    if args.stitch:
        spec = parse_stitch_spec(args.stitch)
        entries = [(name, _load_one(args, name), p) for name, p in spec]
        vecs, st = stitch(entries)
        label = "stitched"
        print(
            f"[stitch] {len(spec)} 图 → {st['n_in']} 实例,去重 {st['n_dedup']},"
            f"输出 {st['n_out']}(容差 {st['tol']} m)"
        )
        if st["n_bucket_skipped"]:
            print(f"[stitch] ⚠️ {st['n_bucket_skipped']} 条落在超大桶里**未参与去重**(只报不删)")
        n_seam = len(seams(entries))
        if n_seam:
            print(f"[stitch] 接缝候选 {n_seam} 处(报告非改写;见 stitch.seams)")
    else:
        vecs = _load_one(args, args.map)
        label = args.map

    full_json = os.path.join(args.out, f"{label}_full.json")
    with open(full_json, "w", encoding="utf-8") as f:
        f.write(vecs_dump(vecs, label, "carla_world"))
    n = sum(1 for v in vecs if v.cls != "traffic_light")
    print(f"{full_json}:{len(vecs)} 实例({n} 折线)")
    render_bev(vecs, os.path.join(args.out, f"{label}_full.png"))
    if fmt:
        _write_format(vecs, fmt, os.path.join(args.out, f"{label}_full"), label)

    if args.ego:
        cropped = tuple(
            resample(v, 20) for v in crop_to_ego(vecs, (args.ego[0], args.ego[1]), radius=args.radius)
        )
        fid_json = os.path.join(args.out, f"{args.map}_{args.fid}.json")
        with open(fid_json, "w", encoding="utf-8") as f:
            f.write(vecs_dump(cropped, args.map, "carla_world"))
        print(f"{fid_json}:{len(cropped)} 实例(裁剪 ±{args.radius}m 后)")
        render_bev(cropped, os.path.join(args.out, f"{args.map}_{args.fid}.png"))
        if fmt:
            _write_format(cropped, fmt, os.path.join(args.out, f"{args.map}_{args.fid}"), args.map)


if __name__ == "__main__":
    main()

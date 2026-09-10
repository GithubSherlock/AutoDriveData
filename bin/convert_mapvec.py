"""A5:地图矢量 → MapTRv2 annotation 口径转换器。

输入 A4 帧级 JSON(training/map/{map}_{fid}.json,CARLA 世界系),按 ego 位姿转
ego 局部系,输出 MapTRv2 `custom_nusc_map_converter.VectorizedLocalMap` 同构的
annotation dict(divider/ped_crossing/boundary/centerline 四类,N×2 折线,局部系)。
stop_line/traffic_light 是工程补充类,不进 MapTR 训练口径(留在 full json 供质检)。

坐标口径与官方转换器逐项对齐:rotate(-patch_angle) 后平移 -patch_center,z 丢弃;
训练管线(MapTRv2 vectorize_map)会再 resample 20 点 + 归一化到 [-1,1]。

用法:
  python bin/convert_mapvec.py --in training/map/Town10HD_Opt_000000.json \\
      --ego -64.64 24.47 0.16 --out training/map/Town10HD_Opt_000000_ann.json
"""

from __future__ import annotations

import argparse
import json
import os

from autodrivedata.mapvec import to_ego_frame, to_maptr_annotation, vecs_load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="A4 帧级 JSON")
    ap.add_argument("--ego", nargs=3, type=float, required=True, metavar=("X", "Y", "YAW"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _, frame, vecs = vecs_load(open(args.src, encoding="utf-8").read())
    if frame != "carla_world":
        raise SystemExit(f"输入坐标系应为 carla_world,实际 {frame}")
    local = to_ego_frame(vecs, args.ego[0], args.ego[1], args.ego[2])
    ann = to_maptr_annotation(local)
    n_each = {k: len(v) for k, v in ann.items()}
    skipped = sum(1 for v in local if v.cls not in ann)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=1)
    print(f"{args.out}:{n_each}(skip 非四类 {skipped})")


if __name__ == "__main__":
    main()

"""B2:数据集组装器——环视采集 + 地图矢量 → MapTRv2 infos 同构 json。

消费 B1 的 surround root(6 视角图像 + calib.json + ego_pose.json)与 A 阶段的
全图矢量 json,逐帧组装 MapTRv2 `_fill_trainval_infos` 同构的核心字段:
cams(每相机 data_path/sensor2ego/intrinsic)+ 帧级 ego2global + annotation
(四类矢量 GT,ego 局部系,先 ±radius 预裁剪、再按官方口径裁到 BEV 60×30m
训练窗口后 20 点重采样)。

口径注记:MapTRv2 官方 annotation 在 **LiDAR 局部系**(lidar2global 变换);
本管道无 LiDAR,用 **ego 局部系**(A5 `to_ego_frame` 口径,与 lidar 局部系
同为车辆参考系,训练消费端无差别——lidar2ego 恒等)。

用法:
  python bin/assemble_maptr.py --surround outputs/surround_drive \
      --map-json training/map/Town10HD_Opt_full.json \
      --out outputs/surround_drive/map_infos.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autodrivedata.mapvec import (
    BEV_RANGE,
    clip_to_bev,
    crop_to_ego,
    resample,
    to_ego_frame,
    to_maptr_annotation,
    vecs_load,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surround", required=True, help="B1 采集根目录")
    ap.add_argument("--map-json", required=True, help="A4 全图矢量 json")
    ap.add_argument("--out", required=True, help="输出 infos json")
    ap.add_argument("--radius", type=float, default=51.2, help="ego 窗口裁剪半径(方形)")
    args = ap.parse_args()

    root = Path(args.surround)
    calib = json.loads((root / "calib.json").read_text(encoding="utf-8"))
    poses = json.loads((root / "ego_pose.json").read_text(encoding="utf-8"))
    _, frame, vecs = vecs_load(Path(args.map_json).read_text(encoding="utf-8"))
    if frame != "carla_world":
        raise SystemExit(f"地图矢量坐标系应为 carla_world,实际 {frame}")

    infos: list[dict] = []
    for p in poses:
        i = p["frame"]
        ego = (p["x"], p["y"])
        local = to_ego_frame(crop_to_ego(vecs, ego, radius=args.radius), p["x"], p["y"], p["yaw"])
        # MapTR 官方口径:GT 裁剪到 BEV 训练窗口(60×30m)后再 20 点重采样
        local = tuple(resample(v, 20) for v in clip_to_bev(local, BEV_RANGE))
        infos.append(
            {
                "frame": i,
                "token": f"{i:06d}",
                "ego2global": [p["x"], p["y"], p["z"], p["yaw"], p["pitch"], p["roll"]],
                "cams": {
                    name: {
                        "data_path": f"{name.lower()}/{i:06d}.png",
                        "sensor2ego": c["sensor2ego"],
                        "intrinsic": c["intrinsic"],
                    }
                    for name, c in calib.items()
                },
                "annotation": to_maptr_annotation(local),
            }
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=1)
    n_ann = {k: len(v) for k, v in infos[0]["annotation"].items()}
    print(f"{args.out}:{len(infos)} 帧(首帧 annotation {n_ann})")


if __name__ == "__main__":
    main()

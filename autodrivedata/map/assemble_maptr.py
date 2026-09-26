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
  python -m autodrivedata.map.assemble_maptr --surround outputs/surround_drive \
      --map-json training/map/Town10HD_Opt_full.json \
      --out outputs/surround_drive/map_infos.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autodrivedata.map.mapvec import (
    BEV_RANGE,
    clip_to_bev,
    crop_to_ego,
    resample,
    to_ego_frame,
    to_maptr_annotation,
    vecs_load,
)
from autodrivedata.paths import project_path


def roots_and_prefixes(surround: str | None, segs_dir: str | None) -> list[tuple[Path, str]]:
    """`(--surround X)` → `[(X, "")]`;`(--segs-dir D)` → `[(D/segK, "segK/"), …]`。

    `data_path` 的段名前缀只在这里决定(见 `_seg_infos` 的 path_prefix 说明)——
    单独抽出来是为了能**不依赖地图矢量 json** 单测这条契约:写成"恒定带前缀"
    会让所有旧单段命令静默指错文件,而那种错在训练时表现为"图看着没错、学不动"。
    """
    if (surround is None) == (segs_dir is None):
        raise ValueError("--surround 与 --segs-dir 必须且只能给一个")
    if segs_dir is not None:
        roots = sorted(p for p in Path(segs_dir).glob("seg*") if p.is_dir())
        if not roots:
            raise ValueError(f"{segs_dir} 下没有 seg* 子目录")
        return [(r, f"{r.name}/") for r in roots]  # --root = 父目录
    assert surround is not None  # 上面已排除两者同时为 None
    return [(Path(surround), "")]  # --root = 段目录本身(旧用法不变)


def _seg_infos(
    calib: dict, poses: list[dict], vecs, radius: float, seg: str, base: int, path_prefix: str
) -> list[dict]:
    """单段的 infos(帧号全局递增;`seg`/`frame_in_seg` 供留出划分与时序窗口用)。

    `path_prefix` 让 `data_path` 始终**相对调用方给的 `--root`**:单段(`--surround`)
    时 root 就是段目录、前缀为空(与旧产物逐字节等价);多段(`--segs-dir`)时 root
    是父目录、前缀为 `segK/`。写成"恒定带段名前缀"会让所有旧命令静默指错文件。
    """
    out: list[dict] = []
    for p in poses:
        i, gi = p["frame"], base + p["frame"]
        ego = (p["x"], p["y"])
        local = to_ego_frame(crop_to_ego(vecs, ego, radius=radius), p["x"], p["y"], p["yaw"])
        # MapTR 官方口径:GT 裁剪到 BEV 训练窗口(60×30m)后再 20 点重采样
        local = tuple(resample(v, 20) for v in clip_to_bev(local, BEV_RANGE))
        out.append(
            {
                "frame": gi,
                "seg": seg,
                "frame_in_seg": i,
                # token 必须**全局唯一**:eval_maptr --out-frames 用它做文件名,
                # 多段共用 {i:06d} 会互相覆盖(静默丢帧)
                "token": f"{seg}_{i:06d}",
                "ego2global": [p["x"], p["y"], p["z"], p["yaw"], p["pitch"], p["roll"]],
                "cams": {
                    name: {
                        # 图像路径:多段合并后各段图像在各自目录下,
                        # 不带段名会让所有段都指向同一批文件(参数键不同却同名)
                        "data_path": f"{path_prefix}{name.lower()}/{i:06d}.png",
                        "sensor2ego": c["sensor2ego"],
                        "intrinsic": c["intrinsic"],
                    }
                    # 过滤 meta 键(collect_surround 在 calib 顶层写 "map" 溯源;
                    # 相机键统一 CAM_* 前缀,见 SURROUND_CAMS)
                    for name, c in calib.items()
                    if name.startswith("CAM_")
                },
                "annotation": to_maptr_annotation(local),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surround", default=None, help="B1 单段采集根目录(旧用法;与 --segs-dir 二选一)")
    ap.add_argument(
        "--segs-dir", default=None, help="B1 多段父目录:取其中的 seg* 子目录(与 --surround 二选一)"
    )
    ap.add_argument("--map-json", required=True, help="A4 全图矢量 json")
    ap.add_argument("--out", required=True, help="输出 infos json")
    ap.add_argument("--radius", type=float, default=51.2, help="ego 窗口裁剪半径(方形)")
    args = ap.parse_args()
    args.out = str(project_path(args.out))  # 产物锚定项目根(相对路径不随 cwd 漂移)
    try:
        roots = roots_and_prefixes(args.surround, args.segs_dir)
    except ValueError as e:
        ap.error(str(e))

    _, frame, vecs = vecs_load(Path(args.map_json).read_text(encoding="utf-8"))
    if frame != "carla_world":
        raise SystemExit(f"地图矢量坐标系应为 carla_world,实际 {frame}")

    infos: list[dict] = []
    for root, prefix in roots:
        calib = json.loads((root / "calib.json").read_text(encoding="utf-8"))
        poses = json.loads((root / "ego_pose.json").read_text(encoding="utf-8"))
        # 段名 = 目录名。--surround 旧用法下 infos 只多三个键(seg / frame_in_seg /
        # token 由 {i:06d} 变 {seg}_{i:06d}),cams / ego2global / annotation 逐帧
        # 不变(2026-09-24 用 surround_train 300 帧实测逐帧比对:有差异的键只有这三个)
        infos += _seg_infos(calib, poses, vecs, args.radius, root.name, len(infos), prefix)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=1)
    n_ann = {k: len(v) for k, v in infos[0]["annotation"].items()}
    per_seg = {r.name: sum(1 for i in infos if i["seg"] == r.name) for r, _ in roots}
    print(f"{args.out}:{len(infos)} 帧 {per_seg}(首帧 annotation {n_ann})")


if __name__ == "__main__":
    main()

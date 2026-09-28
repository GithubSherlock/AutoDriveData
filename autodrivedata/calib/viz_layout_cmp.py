"""布局对照数值化:同镜头两种相机布局 → 地图 GT 投影到各相机上的可见性对比。

消费微段 infos(cams/ego2global/annotation)——两段 ego 位姿一致(同起点),GT
完全同一地图矢量 → 画面差异只来自相机**朝向/挂点**。输出:
- 每布局一张 6 相机拼图(GT 青绿投影 overlay);
- stdout 数值:每相机「画出的 GT 折线段数」——旧布局后相机 vs 官方后相机的
  覆盖差异一眼可见。

用法:
  python -m autodrivedata.calib.viz_layout_cmp --a outputs/surround_micro_legacy --b outputs/surround_micro_official

注:两个微采样目录(surround_micro_legacy / surround_micro_official)已于 2026-09-20
清理删除(§P-L.1 结论已归档),本脚本因此没有在库的默认输入 —— 用前先重采:
  python -m autodrivedata.sim.collect_surround_micro --out outputs/surround_micro_legacy   # 及 _official
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from autodrivedata.map.mapvec import MAPTR_CLASSES
from autodrivedata.map.mapviz import (
    GT_COLOR,
    cam_pose,
    draw_projected_lines,
    intrinsics_from_k,
    project_lines,
)


def _gt_lines_for(infos: dict, idx: int) -> list[np.ndarray]:
    """infos[idx] 的四类 GT 折线并成一个列表(投影画线用,类不区分)。"""
    ann = infos[idx]["annotation"]
    lines: list[np.ndarray] = []
    for cls in MAPTR_CLASSES:
        lines.extend(np.asarray(pt, dtype=np.float64) for pt in ann[cls])
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="布局 A micro root(旧 235/125)")
    ap.add_argument("--b", required=True, help="布局 B micro root(官方 108.6/-110.8)")
    ap.add_argument("--frame", type=int, default=0, help="对比帧")
    ap.add_argument("--out", default="outputs/viz_layout_cmp", help="拼图输出目录")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # (tag, root, infos, calib) —— root 一路带到读图处。**曾经它是丢掉的**:
    # 读 infos 用 `--a`/`--b`,读图却拼死路径 `PROJECT_ROOT/outputs/surround_micro_{tag}`
    # (2026-09-28 实测:按 docstring 重采到别处再 `--a/--b` 指过去,数值表照出、读图必
    # `FileNotFoundError`)。与「硬编码源码路径常量」是同一类失效(见 docs/refactor-2026-09.md 阶段 4)。
    tables: list[tuple[str, str, dict, dict]] = []
    for tag, root in (("legacy", args.a), ("official", args.b)):
        infos = json.loads((Path(root) / "map_infos.json").read_text(encoding="utf-8"))
        calib = json.loads((Path(root) / "calib.json").read_text(encoding="utf-8"))
        tables.append((tag, root, infos, calib))

    infos_a = tables[0][2]
    rec = infos_a[args.frame]
    ego = rec["ego2global"]
    gt = _gt_lines_for(infos_a, args.frame)
    cam_names = sorted(rec["cams"])

    print(f"帧 {args.frame}  ego={tuple(round(v, 3) for v in ego[:4])}  GT {len(gt)} 条折线")
    print(f"{'相机':<18}" + "".join(f"{t:>12}" for t, _, _, _ in tables))
    for name in cam_names:
        row = []
        for _tag, _root, infos, calib in tables:
            k = infos[args.frame]["cams"][name]["intrinsic"]
            intrinsics = intrinsics_from_k(k, (1242, 375))
            pose = cam_pose(infos[args.frame]["ego2global"], calib[name]["sensor2ego"])
            segs = project_lines(gt, ego, pose, intrinsics)
            row.append(len(segs))
        print(f"{name:<18}" + "".join(f"{v:>12}" for v in row))

    for tag, root, infos, calib in tables:
        canvases = []
        for name in cam_names:
            data_path = Path(root) / name.lower() / f"{args.frame:06d}.png"
            img = Image.open(data_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            k = infos[args.frame]["cams"][name]["intrinsic"]
            intrinsics = intrinsics_from_k(k, (1242, 375))
            pose = cam_pose(infos[args.frame]["ego2global"], calib[name]["sensor2ego"])
            n = draw_projected_lines(draw, gt, ego, pose, intrinsics, GT_COLOR, width=3)
            canvases.append(img)
            print(f"  [{tag}/{name}] 投影 GT 段数 = {n}")
        wsum = sum(c.width for c in canvases)
        hmax = max(c.height for c in canvases)
        merged = Image.new("RGB", (wsum, hmax), (0, 0, 0))
        x = 0
        for c in canvases:
            merged.paste(c, (x, 0))
            x += c.width
        p = out / f"{tag}_frame{args.frame:06d}.png"
        merged.save(p)
        print(f"[out] {p.resolve()}")


if __name__ == "__main__":
    main()

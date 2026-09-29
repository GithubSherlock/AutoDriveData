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

from autodrivedata.calib.camera_rig import camera_grid_order, camera_grid_rows
from autodrivedata.map.mapvec import MAPTR_CLASSES
from autodrivedata.map.mapviz import (
    GT_COLOR,
    cam_pose,
    draw_projected_lines,
    intrinsics_from_k,
    project_lines,
)
from autodrivedata.utils import fonts


def _gt_lines_for(infos: dict, idx: int) -> list[np.ndarray]:
    """infos[idx] 的四类 GT 折线并成一个列表(投影画线用,类不区分)。"""
    ann = infos[idx]["annotation"]
    lines: list[np.ndarray] = []
    for cls in MAPTR_CLASSES:
        lines.extend(np.asarray(pt, dtype=np.float64) for pt in ann[cls])
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="布局 A micro root(如 outputs/surround_micro_a)")
    ap.add_argument("--b", required=True, help="布局 B micro root(如 outputs/surround_micro_b)")
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
    # 标签只用于产物文件名:a = `--a` / b = `--b`。**别用 legacy/official 当标签** ——
    # 那是已移除的 rig 名,会让人以为这张图还在对照两代标定(2026-09-28 实测:用户看到
    # `legacy_frame000000.png` 就是这么误会的)。
    for tag, root in (("a", args.a), ("b", args.b)):
        infos = json.loads((Path(root) / "map_infos.json").read_text(encoding="utf-8"))
        calib = json.loads((Path(root) / "calib.json").read_text(encoding="utf-8"))
        tables.append((tag, root, infos, calib))

    infos_a = tables[0][2]
    rec = infos_a[args.frame]
    ego = rec["ego2global"]
    gt = _gt_lines_for(infos_a, args.frame)
    # 六路顺序走**唯一取序入口**(`camera_rig.camera_grid_order`)。原先这里是
    # `sorted(rec["cams"])` —— 字母序出 1×6,第二行左右与地理直觉相反,
    # 且与另几张六视角图读不出是同一套。数值表也按同一顺序打印,便于对着图读。
    cam_names = camera_grid_order(rec["cams"])

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
        tiles: dict[str, Image.Image] = {}
        for name in cam_names:
            data_path = Path(root) / name.lower() / f"{args.frame:06d}.png"
            img = Image.open(data_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            k = infos[args.frame]["cams"][name]["intrinsic"]
            intrinsics = intrinsics_from_k(k, (1242, 375))
            pose = cam_pose(infos[args.frame]["ego2global"], calib[name]["sensor2ego"])
            n = draw_projected_lines(draw, gt, ego, pose, intrinsics, GT_COLOR, width=3)
            # 逐格标注:排成 2×3 之后,"哪一格是哪路"不再能靠"一行从左到右数"读出来,必须标。
            # **文字一律走 fonts**(裸 `d.text` 遇缺字静默画 .notdef 方框);标签压白底 chip,
            # 直接压画面会与内容撞色(路面/标线里青绿很常见,而 GT 折线正是青绿)。
            lx, ly, size = 14, 14, 26
            w = fonts.width(name, size) + 20
            draw.rectangle([lx - 8, ly - 6, lx + w, ly + size + 10], fill=(255, 255, 255), outline=(0, 0, 0))
            fonts.draw_text(draw, (lx, ly), name, size, fill=(0, 0, 0))
            tiles[name] = img
            print(f"  [{tag}/{name}] 投影 GT 段数 = {n}")

        # 2 行 × 3 列(`CAMERA_GRID_ROWS`)。六格同尺寸,故直接按格宽高 paste。
        rows = [[tiles[c] for c in row if c in tiles] for row in camera_grid_rows(tiles)]
        rows = [r for r in rows if r]
        tw, th = rows[0][0].size
        merged = Image.new("RGB", (tw * max(len(r) for r in rows), th * len(rows)), (0, 0, 0))
        for ri, row in enumerate(rows):
            for ci, tile in enumerate(row):
                merged.paste(tile, (ci * tw, ri * th))
        p = out / f"{tag}_frame{args.frame:06d}.png"
        merged.save(p)
        print(f"[out] {p.resolve()}")


if __name__ == "__main__":
    main()

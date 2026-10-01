"""把已落盘的 KITTI root 按**当前** GT 口径重筛 `label_2`(口径迁移,不改像素)。

## 为什么是"重筛"而不是"重采"

2026-10-01 给 `gt/core.py` 加了 `MIN_BOX_SIDE_PX`(退化投影剔除,见 §P-V12):车**擦过镜头**
时投出零面积框,它与任何预测的 IoU 恒为 0,是白送的漏检(P1 每份数据 11/194)。这条修在
**出框侧**,只对**新采**的数据生效 —— 磁盘上那批按旧口径写下的框得单独处理。

**重采能修,但结果是投骰子**:退化框的进出由**亚帧抖动**决定(ego 偏 0.24 m 就让整框在
「0 px 高」与「164 px 高」之间翻面),每个 root 有 3 台"被擦过的车",5 个 root 两两比对
共 12 个阈值穿越点 ⇒ 重采后**几乎必然**有一对在某一帧上条数不等(而现有数据里
day_clear 对 dense_fog / rain_night / sunset_glare 三对**已经是 0 帧差**)。

**重筛只动 GT 一个量**:像素 / 点云 / calib / pose 逐字节不变 ⇒ 已经验证过的帧级配对
原样保留。判据与采集器**同源**(`gt.core.is_degenerate_gt_line`),所以"筛出来的"
就是"重采一份会得到的"。

## 用法

    python -m autodrivedata.gt.refilter --root outputs/kitti_ab_epic_day_clear \\
        --out outputs/kitti_ab_epic_gtfix_day_clear

**只写新 root,不就地改**:旧 root 是"旧口径"的物证,删不删由人裁 —— 而这条命令
几乎不占盘(见下),没有非就地不可的理由。

`--out` 走 **硬链接**:四件套里只有 `label_2/` 是重写的,**其余目录不复制一个字节**
(5 个 root × 194 MB 若真拷就是 1 GB)。硬链接对只读消费方与真拷贝无法区分。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from autodrivedata.gt.core import MIN_BOX_SIDE_PX, is_degenerate_gt_line
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 只做硬链接的目录(`label_2/` 是唯一被重写的,单独处理)
_LINK_DIRS = ("image_2", "velodyne", "calib", "pose")


def refilter_root(root: Path, out: Path) -> dict[str, int]:
    """`root` → `out`,只重写 `label_2/`。返回计数(纯副作用函数,判据在调用方)。"""
    src = root / "training"
    dst = out / "training"
    if out.exists() and any(dst.rglob("*.txt")):
        raise FileExistsError(f"{out} 已有内容 —— 重筛不覆盖既有产物,换个 --out 或先删")
    dst.mkdir(parents=True, exist_ok=True)

    n_in = n_kept = n_drop = 0
    labels = sorted((src / "label_2").glob("*.txt"))
    for lab in labels:
        lines = [ln for ln in lab.read_text(encoding="utf-8").splitlines() if ln.strip()]
        kept = [ln for ln in lines if not is_degenerate_gt_line(ln)]
        n_in += len(lines)
        n_kept += len(kept)
        n_drop += len(lines) - len(kept)
        (dst / "label_2").mkdir(parents=True, exist_ok=True)
        (dst / "label_2" / lab.name).write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    for name in _LINK_DIRS:
        s, d = src / name, dst / name
        if not s.is_dir():
            continue  # pose/ 是可选目录(旧数据集没有)
        d.mkdir(parents=True, exist_ok=True)
        for f in s.iterdir():
            if f.is_file():
                os.link(f, d / f.name)

    return {"frames": len(labels), "gt_in": n_in, "gt_kept": n_kept, "gt_dropped": n_drop}


def main() -> None:
    ap = argparse.ArgumentParser(description="按当前 GT 口径重筛 label_2(见模块头注)")
    ap.add_argument("--root", required=True, help="源 KITTI root")
    ap.add_argument("--out", required=True, help="目标 root(非 label 目录走硬链接)")
    args = ap.parse_args()

    root = project_path(args.root)
    with runlog.run("autodrivedata.gt.refilter") as rl:
        rl.input(str(root), "root")
        rl.highlight("min_box_side_px", MIN_BOX_SIDE_PX)
        out = project_path(args.out)
        stats = refilter_root(root, out)
        print(f"[refilter] {stats['gt_in']} → {stats['gt_kept']} 条(剔除 {stats['gt_dropped']})")
        print(f"[refilter] 帧数 {stats['frames']}  |  {out.resolve()}")
        for k, v in stats.items():
            rl.highlight(k, v)
        rl.highlight("out", str(out))


if __name__ == "__main__":
    main()

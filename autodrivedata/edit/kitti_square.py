"""KITTI root → **方裁 root**(生成管线能吃的 512²)。**纯值**:禁 carla / torch。

## 为什么必须有这一层

生成管线吃**方图**(官方 demo 的隐含前提,见 `cldm_backend` 头注第五处),而本项目的
采集帧是 **1242×375**;`label_2` 里的 2D 框是**原画幅**的像素坐标。
⇒ 图裁了而框没裁的话,**不报错** —— 只表现为下游 AP 崩掉,而那与"生成质量差"长得一样。

## 变换(圈图与框**同一套**)

中心方裁 `x0=(W−s)//2, y0=(H−s)//2, s=min(W,H)`,再缩到 `--size`:

```
u' = (u − x0) · size / s      v' = (v − y0) · size / s
```

- `label_2` 的 **2D 列(4–7)** 做上面这个仿射;**3D 列(h w l x y z ry)原样不动**
  —— 它们是**相机系的三维量**,与画幅裁剪无关(动它们等于伪造几何)
- **完全落在外**的整行**丢弃并计数**;部分越界的**钳到画幅**
  (与 KITTI 口径一致:`gt/core.py` 的红线 —— 框 = 全 front 角点 min/max **再钳到画幅**)
- 钳完任一边 < `gt.core.MIN_BOX_SIDE_PX` 的**退化框**单独计数并丢弃
  (那是"擦过镜头"那类,它与任何预测 IoU 恒为 0 ⇒ 白送一次漏检)

## 不落什么

**落 `image_2` + `label_2` + `pose` + `depth`(有则落)**(位姿是世界系的,裁剪不影响它)。
**不落 `calib/` 与 `velodyne/`** —— 那两份在裁剪后**口径变了**(内参主点要平移重缩放),
落一份没改的比不落**更危险**:下游会拿它算出"看起来对"的投影。

⚠️ **不重采**(与 [`gt/refilter.py`](../gt/refilter.py) 同一条纪律):原图 + `label_2`
自带的一切就够,重采是投骰子。
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autodrivedata.gt.core import MIN_BOX_SIDE_PX
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 目标边长(与 `conditioned_gen.COND_OUT` 同值 —— 生成管线吃 512²)。
DEF_SIZE = 512


def crop_geometry(width: int, height: int, size: int = DEF_SIZE) -> tuple[int, int, int, float]:
    """原画幅 → `(x0, y0, s, k)`:中心方裁的**整数**左上角、边长、与缩放系数 `k = size / s`。

    ★ `x0`/`y0` 必须是**整数**,且与**生成侧** `cldm_backend.to_square_rgb` 用同一套取整
    —— 它写的是 `(w - s) // 2`(向下取整)。若这里用 `round((w-s)/2)`,1242×375 会得到
    **434** 而生成侧是 **433** ⇒ 条件图与 GT 框**差 1 px**,而"差 1 px"不报错,
    只在下游表现为"AP 莫名低一点",看不出是这里错的。
    这条一致性有**机械判据**:`tests/edit/test_kitti_square.py::test_crop_matches_generator`。
    """
    if width <= 0 or height <= 0 or size <= 0:
        raise ValueError(f"非法尺寸:width={width} height={height} size={size}")
    s = min(width, height)
    return (width - s) // 2, (height - s) // 2, s, size / float(s)


@dataclass(frozen=True)
class LineOutcome:
    """一行 `label_2` 的处置结果 —— **四态**,不是"留/不留"两态。

    把 `clipped` 与 `kept` 分开是刻意的:被画幅裁断的框**不能与完整框混在一起报**
    (本仓 2D 口径红线:裁断样本必须单独分类,既不能并入裁决、也不能装作没看见)。
    """

    kind: str  # kept | clipped | dropped_out | dropped_degenerate | bad
    line: str | None = None


def transform_line(line: str, *, width: int, height: int, size: int = DEF_SIZE) -> LineOutcome:
    """一行 `label_2` → 方裁后的那一行;该丢的丢,并**说清是哪一种丢**。"""
    p = line.split()
    if len(p) < 15:
        return LineOutcome("bad")
    try:
        x1, y1, x2, y2 = (float(v) for v in p[4:8])
    except ValueError:
        return LineOutcome("bad")
    x0, y0, s, k = crop_geometry(width, height, size)

    # 与画幅求交(未缩放坐标下做,避免把缩放误差混进"落在外"的判定)
    ix1, iy1 = max(x1, x0), max(y1, y0)
    ix2, iy2 = min(x2, x0 + s), min(y2, y0 + s)
    if ix2 <= ix1 or iy2 <= iy1:
        return LineOutcome("dropped_out")  # 完全落在外(或退化成一条线)

    u1, v1 = (ix1 - x0) * k, (iy1 - y0) * k
    u2, v2 = (ix2 - x0) * k, (iy2 - y0) * k
    if (u2 - u1) < MIN_BOX_SIDE_PX or (v2 - v1) < MIN_BOX_SIDE_PX:
        return LineOutcome("dropped_degenerate")

    clipped = (ix1 > x1) or (iy1 > y1) or (ix2 < x2) or (iy2 < y2)
    head = " ".join(p[:4])  # type / truncated / occluded / alpha —— 原样
    tail = " ".join(p[8:])  # h w l x y z ry(3D 列)**原样不动**
    return LineOutcome(
        "clipped" if clipped else "kept",
        f"{head} {u1:.2f} {v1:.2f} {u2:.2f} {v2:.2f} {tail}",
    )


def build_square_root(
    src: Path,
    dst: Path,
    *,
    size: int = DEF_SIZE,
    frames: list[int] | None = None,
    dry_run: bool = False,
) -> dict:
    """把 `src`(KITTI root)方裁成 `dst`(含 `depth/`,若源有)。返回**四态计数**。"""
    img_dir, lab_dir = src / "training/image_2", src / "training/label_2"
    if not img_dir.is_dir() or not lab_dir.is_dir():
        raise SystemExit(f"{src} 不是 KITTI root(缺 training/image_2 或 label_2)")
    poses = sorted(training_pose_dir(src).glob("*.txt"))
    stem = sorted(p.stem for p in lab_dir.glob("*.txt"))
    if frames is not None:
        want = {f"{f:06d}" for f in frames}
        stem = [s for s in stem if s in want]
    if not stem:
        raise SystemExit(f"{src} 里没有可用帧(过滤后为空)")

    from PIL import Image

    counts = {"kept": 0, "clipped": 0, "dropped_out": 0, "dropped_degenerate": 0, "bad": 0}
    n_frames = 0
    for fid in stem:
        ip = img_dir / f"{fid}.png"
        if not ip.exists():
            continue
        im = Image.open(ip)
        w, h = im.size
        # 保存前先按 crop_geometry 裁 —— 与 label_2 用的是**同一个函数**,口径不可能分叉
        # `crop_geometry` 已经返回整数边界(与生成侧 `to_square_rgb` 同一套取整),
        # 这里**直接当像素边界用**,不许再 round 一次 —— 再 round 就是第二次取整,
        # 两次取整的口径没人保证一致。
        bx0, by0, s, _k = crop_geometry(w, h, size)
        box = (bx0, by0, bx0 + s, by0 + s)
        if not dry_run:
            out_img = dst / "training/image_2"
            out_img.mkdir(parents=True, exist_ok=True)
            im.convert("RGB").crop(box).resize((size, size), Image.LANCZOS).save(out_img / f"{fid}.png")
        # ★ **计数与落盘分开** —— `--dry-run` 的意义就是"先说会发生什么"。
        #   第一版把计数放进 `if not dry_run` 里,于是 dry-run 永远报 `kept=0`
        #   (那等于"这个开关只是不写盘",不是"先看数")。
        lines = []
        for line in (lab_dir / f"{fid}.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            out = transform_line(line, width=w, height=h, size=size)
            counts[out.kind] = counts.get(out.kind, 0) + 1
            if out.line is not None:
                lines.append(out.line)
        if not dry_run:
            out_lab = dst / "training/label_2"
            out_lab.mkdir(parents=True, exist_ok=True)
            (out_lab / f"{fid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_frames += 1

    # ★ **深度也要按同一套仿射裁** —— 它是**逐像素几何量**,与 `image_2` 同画幅。
    #   不裁的话它就和图对不上,而"对不上"在下游只表现为**雾/退化的空间分布错了**。
    #   (2026-10-06 补:原来的落盘清单里漏了它,导致 `degrade --kinds fogdepth` 没料可用。)
    dep_src = src / "training/depth"
    if not dry_run and dep_src.is_dir():
        import cv2
        from PIL import Image

        ddir = dst / "training/depth"
        ddir.mkdir(parents=True, exist_ok=True)
        for fid in stem:
            dp = dep_src / f"{fid}.npy"
            if not dp.exists():
                continue
            im = Image.open(img_dir / f"{fid}.png")
            bx0, by0, s, _k = crop_geometry(*im.size, size)
            d = np.load(dp)[by0 : by0 + s, bx0 : bx0 + s].astype(np.float32)
            np.save(ddir / f"{fid}.npy", cv2.resize(d, (size, size), interpolation=cv2.INTER_LINEAR))

    if not dry_run:
        pdst = dst / "training/pose"
        pdst.mkdir(parents=True, exist_ok=True)
        for p in poses:
            if p.stem in set(stem):
                shutil.copyfile(p, pdst / p.name)  # 位姿是世界系的,裁剪不影响

    total = sum(counts.values())
    return {
        "src": str(src),
        "dst": str(dst),
        "size": size,
        "n_frames": n_frames,
        "n_lines": total,
        **counts,
        "has_depth": bool((src / "training/depth").is_dir()),
        "clipped_share": round(counts["clipped"] / total, 4) if total else 0.0,
        "dropped_share": round((counts["dropped_out"] + counts["dropped_degenerate"]) / total, 4)
        if total
        else 0.0,
    }


def training_pose_dir(root: Path) -> Path:
    """`training/pose/`。单独抽出来是因为**归档里有一批 root 根本没有它**
    (`kitti_ab_epic_*` 系列实测无 pose,而 `kitti_ab_occl2_*` 有)——
    缺了就照缺处理,**不从别处猜位姿**。"""
    return root / "training/pose"


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--src", required=True, help="源 KITTI root")
    ap.add_argument("--dst", required=True, help="目标方裁 root")
    ap.add_argument("--size", type=int, default=DEF_SIZE)
    ap.add_argument("--frames", default="", help="逗号分隔或 a-b;空 = 全部")
    ap.add_argument("--dry-run", action="store_true", help="只报四态计数,不落盘")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    src, dst = project_path(args.src), project_path(args.dst)
    frames = _parse_frames(args.frames) if args.frames else None
    with runlog.run("autodrivedata.edit.kitti_square") as rl:
        rl.input(src, "src-root")
        rep = build_square_root(src, dst, size=args.size, frames=frames, dry_run=args.dry_run)
        for k in ("n_frames", "n_lines", "kept", "clipped", "dropped_out", "dropped_degenerate", "bad"):
            print(f"  {k:<20} {rep[k]}")
        print(f"  裁断占比 {rep['clipped_share']:.1%}   丢弃占比 {rep['dropped_share']:.1%}")
        rl.highlight("n_lines", rep["n_lines"])
        rl.highlight("dropped_share", rep["dropped_share"])
        rl.highlight("clipped_share", rep["clipped_share"])
        out = project_path(dst) / "square_summary.json"
        if not args.dry_run:
            out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
            rl.artifact(out, "summary")
            print(f"[done] {out}")


def _parse_frames(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    if not out:
        raise SystemExit(f"--frames 解析出空列表:{spec!r}")
    return out


if __name__ == "__main__":
    main()

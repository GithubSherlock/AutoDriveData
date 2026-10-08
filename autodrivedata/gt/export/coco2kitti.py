"""COCO val2017 → **KITTI 布局**的 root(只含 `training/{image_2,label_2}`)。

## 它服务的是谁

域自适应训练线**已裁决放兄弟仓 `AutoLabel`**(依赖单向 AutoDriveData → AutoLabel)。
本仓只出**数据 + 判据**。这个模块就是数据那一半。

兄弟仓的 2D 微调(`auto2dlabel/tools/train_kitti.py`)与评测
(`auto2dlabel/benchmarks/kitti_benchmark.py`)**只读这两样**:
`training/image_2/{stem}.png` 与 `training/label_2/{stem}.txt` —— 复核逐行核过,
不读 calib / velodyne / pose / ImageSets。所以产出**只落这两样**。

## ★ 一个必须写死的诚实边界:3D 列是**占位**

COCO **没有 3D 标注**。而 `label_2` 是 15 列格式,后 7 列是 `h w l x y z ry`。
本模块**写满 15 列**(占位 `-1 -1 -1 -1000 -1000 -1000 -10`)而不是只写 8 列:

- 下游 `kitti_line_to_yolo` 只读 1/2/5–8 列,**8 列就够**;
- 但本仓既有 root **一律 15 列**,写 8 列会让任何走 `gt.core.parse_gt_line` 的读者崩;
- ⇒ **写满**。⚠️ **但这 7 列是占位,不是真值** —— 任何按 3D 出数的下游**都不许**吃这个 root。

## ★★ 两个会静默出错的点(复核点名,本模块必须报出来)

1. **难度档过滤会大量丢框**:下游 `kitti_difficulty` 的硬门槛是**框高 ≥ 25 px**,
   而 **COCO 小目标极多**。丢掉的框**一个字都不打印** ⇒ 交出去的是"一批数据",
   而没人知道有多少能用。⇒ 本模块报 **框高分布 + 低于 25 px 的条数**。
   (25 px 是 **KITTI 官方**门槛,不是兄弟仓发明的;三档的完整规则见
   `auto2dlabel/benchmarks/kitti_benchmark.py::kitti_difficulty` —— **不在这里复刻**,
   那会变成两份会漂的实现。)
2. **空类**:兄弟仓的类名表是 KITTI 的 8→5 类,而我们这边**只会产出 3 类**
   (Car/Pedestrian/Cyclist)⇒ 他们表里的 `train`/`truck` 实例数恒为 0,**照训不报错**。
   ⇒ 本模块**逐类点实例数**,并把 0 实例的类**明确报出来**。

## 类表:只有一张

**复用** `perception/backends.COCO_FALLBACK`(car/truck/bus→Car、person→Pedestrian、
bicycle/motorcycle→Cyclist)。**不新造第二张** —— 本仓在"两张表迟早漂"上踩过。
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from autodrivedata.gt.export.kitti import normalize_frame_id, write_frame
from autodrivedata.perception.domain_gap import load_coco_gt
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: COCO 的 3D 列占位(`h w l x y z ry`)。**见模块头注:是占位,不是真值。**
PLACEHOLDER_3D = (-1.0, -1.0, -1.0, -1000.0, -1000.0, -1000.0, -10.0)

#: KITTI 官方的 difficulty **硬门槛**(框高 ≥ 25 px)。这里只用它来**报存活风险**,
#: 完整的三档规则在兄弟仓(`kitti_difficulty`),**故意不复刻**(两份实现迟早漂)。
MIN_BOX_H_PX = 25.0

#: 贴到画幅这个边距以内的框,判为**可能被截断**(COCO 没有 truncated 字段)。
TRUNC_EDGE_PX = 2.0


def coco_box_flags(xyxy: tuple[float, float, float, float], w: int, h: int, *, iscrowd: bool):
    """COCO 的框 → `(truncated, occluded)` 两个 KITTI 字段。

    ## ⚠️ 这是**近似**,而且必须写在明面上

    COCO **没有** `truncated` / `occluded`。而下游的难度档判据**要读它们**。
    本模块的规则:

    | KITTI 字段 | 本模块怎么给 | 依据 |
    |---|---|---|
    | `occluded` | `2` if `iscrowd` else `0` | COCO 只有 `iscrowd`(人群/一坨);**没有遮挡分级** |
    | `truncated` | `0.0`,**框贴画幅边**时给 `0.5` | 贴边 ⇒ 大概率被裁;COCO 无截断比例 |

    ⚠️ `0.5` 恰好是 hard 档的上限(`trunc <= 0.5`)⇒ 贴边的框**仍留在 hard 档**,
    这是**刻意**的(贴边不等于没救;真判死会让漏检归因失真)。
    """
    x1, y1, x2, y2 = xyxy
    occluded = 2 if iscrowd else 0
    touches = x1 <= TRUNC_EDGE_PX or y1 <= TRUNC_EDGE_PX or x2 >= w - TRUNC_EDGE_PX or y2 >= h - TRUNC_EDGE_PX
    return (0.5 if touches else 0.0), occluded


def kitti_line(cls: str, xyxy: tuple[float, float, float, float], *, truncated: float, occluded: int) -> str:
    """一条 15 列的 `label_2` 行(**与 `gt.core.box_to_gt_line` 同格式**)。

    ⚠️ `alpha` 恒 `0.00`(与本仓既有导出同口径);后 7 列是**占位**(见模块头注)。
    """
    x1, y1, x2, y2 = xyxy
    parts = [
        cls,
        f"{truncated:.2f}",
        str(int(occluded)),
        "0.00",  # alpha:本仓既有导出也恒 0.00
        f"{x1:.2f}",
        f"{y1:.2f}",
        f"{x2:.2f}",
        f"{y2:.2f}",
        *(f"{v:.2f}" for v in PLACEHOLDER_3D),
    ]
    return " ".join(parts)


def coco_to_kitti(
    coco_root: Path, out_root: Path, *, image_ids: list[int] | None = None, limit: int | None = None
) -> dict:
    """COCO → KITTI。返回**统计**(判据吃它)。`image_ids` 给定就只出这些帧(切分用)。"""
    ann = load_coco_gt(coco_root / "annotations" / "instances_val2017.json")
    raw = json.loads((coco_root / "annotations" / "instances_val2017.json").read_text(encoding="utf-8"))
    crowd = {(a["image_id"], tuple(a["bbox"])) for a in raw["annotations"] if a.get("iscrowd")}

    ids = sorted(i for i, boxes in ann.items() if boxes)
    if image_ids is not None:
        ids = [i for i in ids if i in set(image_ids)]
    if limit:
        ids = ids[:limit]
    if not ids:
        raise SystemExit(f"{coco_root} 下没有可用的图 —— 查 annotations 与类表")

    per_class: dict[str, int] = {}
    heights: list[float] = []
    n_written = 0
    for i in ids:
        src = coco_root / "val2017" / f"{i:012d}.jpg"
        if not src.is_file():
            continue
        # ★ **重编成真 PNG**(而不是把 COCO 的 JPEG 字节改个扩展名写出去)。
        #   理由:`write_frame(image_png=...)` 这条契约**就写着 png** ——
        #   一个叫 `.png` 却是 JPEG 的文件正是本仓专门在抓的那类"静默错"
        #   (PIL/cv2 会嗅探格式、能读,所以它**不会报错**;而严格按扩展名解释的读者会崩)。
        #   代价:体积约 3–5×(实测 200 帧 33 MB)。盘够。
        img = Image.open(src).convert("RGB")
        w, h = img.size
        _buf = io.BytesIO()
        img.save(_buf, format="PNG")
        lines = []
        for cls, xyxy in ann[i]:
            x, y, bw, bh = xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
            iscrowd = (i, (x, y, bw, bh)) in crowd
            trunc, occ = coco_box_flags(xyxy, w, h, iscrowd=iscrowd)
            lines.append(kitti_line(cls, xyxy, truncated=trunc, occluded=occ))
            per_class[cls] = per_class.get(cls, 0) + 1
            heights.append(xyxy[3] - xyxy[1])
        # ⚠️ `write_frame` **不给 velodyne/calib** ⇒ 只落 image_2 + label_2(2D-only root)
        write_frame(
            out_root,
            normalize_frame_id(str(i)),
            image_png=_buf.getvalue(),
            labels=lines,
        )
        n_written += 1
    if not n_written:
        raise SystemExit("一张都没写出来 —— 查 val2017 与标注的 id 对不对得上")

    hh = np.asarray(heights, dtype=np.float64)
    return {
        "n_frames": n_written,
        "n_boxes": int(hh.size),
        "per_class": per_class,
        # ★ 陷阱①:下游按 **框高 ≥25 px** 过滤。COCO 小目标多 ⇒ 这个数必须报出来。
        "box_h_px": {
            "p10": float(np.percentile(hh, 10)),
            "median": float(np.median(hh)),
            "p90": float(np.percentile(hh, 90)),
            "n_below_25px": int((hh < MIN_BOX_H_PX).sum()),
            "frac_below_25px": float((hh < MIN_BOX_H_PX).mean()),
        },
        # ★ 陷阱②:兄弟仓的类表里那些我们产不出来的类
        "empty_classes": sorted(set(("Car", "Pedestrian", "Cyclist")) - set(per_class)),
        "placeholder_3d": "见模块头注:后 7 列是占位,任何按 3D 出数的下游都不许吃这个 root",
    }


def split_ids(ids: list[int], *, holdout_frac: float = 0.5, seed: int = 0) -> tuple[list[int], list[int]]:
    """★ **不相交**地切成 (微调, 测试)。

    ⚠️ 盘上**只有 `val2017`(5000 张),没有 `train2017`** ⇒ 不切就会在同一批图上
    既微调又评测,而那个读数**没有意义**(本仓在"池化口径下打乱配对是恒等"上踩过一次)。
    切分**固定 seed** ⇒ 可复现,且要落进 manifest 让两边看到的是**同一份**切分。
    """
    order = np.random.default_rng(seed).permutation(sorted(ids))
    n_hold = int(round(len(order) * holdout_frac))
    # ⚠️ `permutation` 给的是 **numpy int64**,直接进 `json.dumps` 会
    #    `TypeError: Object of type int64 is not JSON serializable`(实测踩到)。
    return [int(v) for v in order[n_hold:]], [int(v) for v in order[:n_hold]]


def assert_disjoint(a: list[int], b: list[int]) -> None:
    """★ 切分自证:两半**零重叠**。判据必须在**故意造重叠**时会红(测试里钉)。"""
    dup = set(a) & set(b)
    if dup:
        raise SystemExit(f"切分有重叠 {len(dup)} 张(示例 {sorted(dup)[:5]}) —— 微调与测试不许同图")
    if not a or not b:
        raise SystemExit(f"切分出来有一半是空的(微调 {len(a)} / 测试 {len(b)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--coco-root", default="/root/autodl-tmp/Documents/datasets/COCO2017")
    ap.add_argument("--out-root", required=True, help="产出的 KITTI root(只落 image_2 + label_2)")
    ap.add_argument(
        "--split",
        choices=("finetune", "holdout"),
        default="finetune",
        help="出哪一半。★ **两半都必须出成真的 root** —— 兄弟仓要在留出那半上评,"
        "只给 id 列表它读不了。切分是**确定性**的(`--seed`),两次跑出来的切分逐位相同",
    )
    ap.add_argument("--limit", type=int, default=None, help="最多出几张(调试)")
    ap.add_argument("--holdout-frac", type=float, default=0.5, help="测试那半的占比")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default="", help="切分与统计落盘(建议给,两边要对同一份)")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    coco_root = project_path(args.coco_root)
    with runlog.run("autodrivedata.gt.export.coco2kitti") as rl:
        rl.input(str(coco_root), "coco-root")
        rl.highlight("holdout_frac", args.holdout_frac)
        rl.highlight("seed", args.seed)
        # 先切分、再只导出"微调"那一半(测试那半只登记 id,不落图)
        ann = load_coco_gt(coco_root / "annotations" / "instances_val2017.json")
        usable = sorted(i for i, b in ann.items() if b)
        finetune, holdout = split_ids(usable, holdout_frac=args.holdout_frac, seed=args.seed)
        assert_disjoint(finetune, holdout)
        want = finetune if args.split == "finetune" else holdout
        rep = coco_to_kitti(coco_root, project_path(args.out_root), image_ids=want, limit=args.limit)

        print(f"\n=== COCO → KITTI(`{args.out_root}`)===")
        print(f"  切分:微调 {len(finetune)} 张 / 测试 {len(holdout)} 张(**零重叠**,已自证)")
        print(f"  本次出的是 **{args.split}** 那一半")
        print(f"  实写 {rep['n_frames']} 帧 / {rep['n_boxes']} 个框;逐类 {rep['per_class']}")
        b = rep["box_h_px"]
        print(
            f"  ★ 框高分布 p10/中位/p90 = {b['p10']:.1f}/{b['median']:.1f}/{b['p90']:.1f} px;"
            f" **低于 25 px 的占 {b['frac_below_25px'] * 100:.1f}%** —— 下游按这个门槛过滤,**会被丢掉**"
        )
        if rep["empty_classes"]:
            print(f"  ★ **空类**:{rep['empty_classes']} 在本 root 里 0 个实例(兄弟仓的类表里有它们)")
        rl.highlight("n_finetune", len(finetune))
        rl.highlight("n_holdout", len(holdout))
        rl.highlight("per_class", rep["per_class"])
        rl.highlight("frac_below_25px", round(b["frac_below_25px"], 4))
        if args.manifest:
            m = project_path(args.manifest)
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text(
                json.dumps(
                    {"finetune_ids": finetune, "holdout_ids": holdout, "stats": rep, "seed": args.seed},
                    indent=1,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            rl.artifact(m, "manifest")
            print(f"[manifest] {m.resolve()}")


if __name__ == "__main__":
    main()

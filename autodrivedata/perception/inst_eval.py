"""实例分割的判据:逐实例匹配 → **mask AP** 与 **PQ / SQ / RQ**(纯值核心,零 carla)。

## 与类级 `sem_eval` 的分工:两把尺子量的不是同一件事

| 判据 | 问的问题 | 分母 |
|---|---|---|
| `sem_eval`(类级) | 每个**像素**判对了吗 | 像素 |
| 本模块(实例级) | 每个**物体**数对了吗、数得准吗 | 实例 |

一张"把两辆车连成一片"的图,类级 mIoU 可能很高(像素大多是车),而实例级会同时记
1 个 FP 和 1 个 FN。**两个数都要看**,只报一个会漏掉另一半的故事 —— 所以类级口径
**保留不动**,这里只做加法。

## 三个数的定义(panoptic 口径)

先把 GT 与预测按 **mask IoU > 0.5 且同类**贪心配对(类别不同一律配不上),然后:

| 量 | 定义 | 塌了说明 |
|---|---|---|
| `RQ`(recognition quality) | `TP / (TP + 0.5·FP + 0.5·FN)` | 检测层面的好坏(数得对不对) |
| `SQ`(segmentation quality) | 配对上的实例的**平均 IoU** | 掩膜画得准不准 |
| `PQ` | `SQ × RQ`,**等价于** `ΣIoU / (TP + 0.5FP + 0.5FN)` | 两者的乘积 |

`PQ = SQ × RQ` 是 PQ 的定义性质,所以本模块**把这条等式当成自证**:算出来对不上
就是实现错了(而不是"口径不同")。

## PQ 与 mask AP 的真实差别(别写成"PQ 多一个语义项")

两个都要求**类匹配**(AP 是逐类算的,类错了进不了任何一类的 TP),所以"PQ 多的那一项
是语义"是**错的**。真实差别有两条,都实测过(`test_inst_eval.py` 有钉):

1. **AP 对掩膜质量不敏感,PQ 敏感** —— 一个 IoU=1.00 的预测与一个 IoU=0.55 的预测,
   在 `AP@0.5` 下**都算命中**(同一个数),而 PQ 里前者贡献 1.00、后者只贡献 0.55。
   所以"掩膜画得糊不糊"只有 PQ 看得出。
2. **AP 是排序型(吃 conf、做 11 点插值),PQ 不是** —— PQ 在固定阈值上数 TP/FP/FN,
   与模型置信度无关。

⇒ 两个数**互为补充**,只报一个会漏掉另一半:分离"数得对不对"(RQ)与"画得准不准"(SQ)
是 PQ 的用途,而"置信度排序好不好"是 AP 的用途。

## 为什么评测对象只有**可数物体**

见 [`inst_tags.THING_TAGS`](inst_tags.py):CARLA 的实例相机给**每个关卡网格**都发 id,
首采 20 帧实测 `Roads` 有 113 个 id、`Car` 只有 13 个。不筛的话 PQ 的分母里全是路面,
而读数照样是个 0–1 的数 —— **看不出来**。`drivable` / `lane` 是 stuff(面,不是个体),
数"有几块路面"没有意义。

用法:
  python -m autodrivedata.perception.inst_eval --root outputs/surround_inst_demo --frames 0-4
  python -m autodrivedata.perception.inst_eval --root <root> --self-test   # 不出模型,只验尺子
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from autodrivedata.perception.compare import ap11
from autodrivedata.perception.inst_tags import Instance, decode_instance_png
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None

CAMS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
#: panoptic 的配对阈值。**是有定义的**,不是可调旋钮 —— PQ 论文用的是 0.5。
PANOPTIC_IOU = 0.5
#: 最小实例面积(px)。太小的实例(远距、被裁)对两类判据都是噪声源。
MIN_INSTANCE_PX = 32


@dataclass(frozen=True)
class PredInstance:
    """预测侧的一个实例。`cls_name` 用 `sem_tags` 的类名(`drivable`/`lane`/`obstacle`)。"""

    cls_name: str
    mask: np.ndarray
    conf: float = 1.0
    instance_id: int = -1

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class MatchResult:
    """一次配对的结果。`pairs` = `(gt_idx, pred_idx, iou)`。"""

    pairs: list[tuple[int, int, float]] = field(default_factory=list)
    fp: int = 0
    fn: int = 0


@dataclass
class PanopticResult:
    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """两个布尔掩膜的 IoU;并集为空返回 `nan`(**不是 0** —— "没有并集"与"完全不重叠"
    是两件事,后者才该是 0)。"""
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return float(inter) / union if union else float("nan")


def match_instances(
    gt: list[Instance],
    pred: list[PredInstance],
    iou_thresh: float = PANOPTIC_IOU,
    *,
    cls_aware: bool = True,
) -> MatchResult:
    """贪心配对:IoU 降序扫,双方都还空着且同类才收。

    与 `compare.match_boxes` 用同一套贪心口径(阈值化匹配 + 降序),便于两处读数互相对照。
    `cls_aware=False` 退化成"只看掩膜"(那一档量的是纯几何,类错不算错)—— 有单测钉这两档
    必须给出不同的数,否则这个开关是死的。
    """
    res = MatchResult()
    if not gt or not pred:
        res.fp = len(pred)
        res.fn = len(gt)
        return res
    cands: list[tuple[float, int, int]] = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            if cls_aware and g.gt_class != p.cls_name:
                continue
            iou = mask_iou(g.mask, p.mask)
            if not np.isnan(iou) and iou > iou_thresh:
                cands.append((iou, i, j))
    cands.sort(key=lambda t: -t[0])
    used_g: set[int] = set()
    used_p: set[int] = set()
    for iou, i, j in cands:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        res.pairs.append((i, j, iou))
    res.fp = len(pred) - len(used_p)
    res.fn = len(gt) - len(used_g)
    return res


def panoptic_quality(
    gt: list[Instance], pred: list[PredInstance], iou_thresh: float = PANOPTIC_IOU
) -> PanopticResult:
    """`PQ = ΣIoU / (TP + 0.5FP + 0.5FN)`,`SQ = ΣIoU / TP`,`RQ = TP / (TP + 0.5FP + 0.5FN)`。

    三者满足 `PQ = SQ × RQ` —— 这是定义性质,调用方应当拿它自证(空样本除外,
    那时三个数都是 `nan`,**不是 0**:0 会被读成"错光了")。
    """
    m = match_instances(gt, pred, iou_thresh)
    tp = len(m.pairs)
    denom = tp + 0.5 * m.fp + 0.5 * m.fn
    if denom == 0:
        return PanopticResult(float("nan"), float("nan"), float("nan"), 0, 0, 0)
    sq = sum(iou for _, _, iou in m.pairs) / tp if tp else float("nan")
    rq = tp / denom
    return PanopticResult(
        pq=(sum(iou for _, _, iou in m.pairs) / denom),
        sq=sq,
        rq=rq,
        tp=tp,
        fp=m.fp,
        fn=m.fn,
    )


def mask_ap(per_frame: list[tuple[list[Instance], list[PredInstance]]], iou_thresh: float = 0.5) -> float:
    """逐帧配对 → 全局 11 点插值 AP(与 `compare.ap11` **同一口径**,不另写一套)。

    ⚠️ 与项目的 AP 红线一致:这个数**必须带 `iou_thresh` 一起报**;单报一个 mAP 而不写
    阈值 = 无效结论。本模块把它写在返回值里的是调用方的事,`eval_root` 负责带上。
    """
    scores: list[tuple[float, bool]] = []
    n_gt = 0
    for gt, pred in per_frame:
        m = match_instances(gt, pred, iou_thresh)
        hit = {j for _, j, _ in m.pairs}
        n_gt += len(gt)
        scores += [(p.conf, j in hit) for j, p in enumerate(pred)]
    if n_gt == 0:
        return float("nan")
    return ap11([s for s, _ in scores], [h for _, h in scores], n_gt)


def _parse_frames(spec: str) -> list[int]:
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return out


def load_gt(root: Path, frame: int, cams: tuple[str, ...] = CAMS) -> dict[str, list[Instance]]:
    """一帧的 GT:逐相机 `(inst_<cam>/ 的 id 图, sem_<cam>/ 的 tag 图)` → 可数物体实例。

    **类从 `sem_*/` 取,不从实例相机再解一次**:两条路实测逐像素相同
    (`collect_surround` 首帧自证 1.000000 × 6 路),再落一份 R 通道图只是多一处会漂的副本。
    """
    from autodrivedata.perception.inst_tags import instances_from_maps
    from autodrivedata.perception.sem_tags import decode_tag_png

    out: dict[str, list[Instance]] = {}
    for cam in cams:
        p_inst = root / f"inst_{cam.lower()}" / f"{frame:06d}.png"
        p_sem = root / f"sem_{cam.lower()}" / f"{frame:06d}.png"
        if not p_inst.exists() or not p_sem.exists():
            continue
        ids = decode_instance_png(p_inst.read_bytes())
        tag = decode_tag_png(p_sem.read_bytes())
        inst = [i for i in instances_from_maps(ids, tag) if i.area >= MIN_INSTANCE_PX]
        if inst:
            out[cam] = inst
    return out


def eval_root(
    root: Path,
    frames: list[int],
    predict: Callable[[str, int], list[PredInstance]] | None = None,
    cams: tuple[str, ...] = CAMS,
) -> dict[str, Any]:
    """整份 root 的实例判据读数。`predict=None` ⇒ 用 GT 当预测(**只用来验尺子**)。"""
    per_cam: dict[str, list[tuple[list[Instance], list[PredInstance]]]] = {}
    n_inconsistent = 0
    for fid in frames:
        gts = load_gt(root, fid, cams)
        for cam, gt in gts.items():
            n_inconsistent += sum(1 for g in gt if not g.consistent)
            if predict is None:
                pred = [PredInstance(cls_name=g.gt_class or "obstacle", mask=g.mask, conf=1.0) for g in gt]
            else:
                pred = predict(cam, fid)
            per_cam.setdefault(cam, []).append((gt, pred))

    all_pairs = [pair for v in per_cam.values() for pair in v]
    pqs = [panoptic_quality(g, p) for g, p in all_pairs]
    tp = sum(r.tp for r in pqs)
    fp = sum(r.fp for r in pqs)
    fn = sum(r.fn for r in pqs)
    denom = tp + 0.5 * fp + 0.5 * fn
    iou_sum = sum(iou for g, p in all_pairs for _, _, iou in match_instances(g, p).pairs)
    return {
        "root": str(root),
        "n_frames": len(frames),
        "n_cams": len(per_cam),
        "gt_instances": sum(len(g) for g, _ in all_pairs),
        "pred_instances": sum(len(p) for _, p in all_pairs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        # 全局(池化)而不是逐相机平均 —— 与 `sem_eval` 的几何自证同一条理由:
        # 逐帧/逐路平均会被"样本极少的那些帧"带偏。
        "pq": (iou_sum / denom) if denom else float("nan"),
        "sq": (iou_sum / tp) if tp else float("nan"),
        "rq": (tp / denom) if denom else float("nan"),
        "mask_ap@0.5": mask_ap(all_pairs, 0.5),
        "n_inconsistent_instances": n_inconsistent,
        "per_cam": per_cam,
    }


def load_instance_predictor(root: Path, backend: str = "sam3"):
    """真模型预测器:`(cam, frame) → 每实例一份掩膜`。

    ⚠️ 这里**只出 `obstacle` 一类** —— 实例评测的对象是可数物体(见 `inst_tags.THING_TAGS`),
    而 `drivable`/`lane` 是 stuff、没有实例语义。

    **两个后端**(2026-10-02):
    - `sam3`(**默认**):开放词表。⚠️ SAM3 **必须点名概念**,没有"把所有 thing 都切出来"
      这种调用 ⇒ 要**逐概念前向再并集**(`sam3_backend.THING_PROMPTS`),代价 = 概念数 ×
      0.5–1.0 s/帧;结果统一记成 `obstacle`,因为 GT 那边只有这一类(things vs stuff)。
    - `yolo`(回退):`yolo11s-seg`(COCO),一次前向出全部实例。

    ⇒ **两边的 PQ/SQ/RQ 不可比**:yolo 的类由模型判(且这里本就被压成 `obstacle`),
    SAM3 的类由**概念表**决定。引用必须写明后端。
    """

    if backend == "sam3":

        def predict_sam3(cam: str, frame: int) -> list[PredInstance]:
            from autodrivedata.perception import sam3_backend

            p = root / cam.lower() / f"{frame:06d}.png"
            if not p.exists():
                return []
            img = Image.open(p).convert("RGB")
            return [
                PredInstance(cls_name="obstacle", mask=i.mask, conf=i.conf)
                for i in sam3_backend.segment(img, sam3_backend.THING_PROMPTS, cls_name="obstacle")
            ]

        return predict_sam3

    def predict(cam: str, frame: int) -> list[PredInstance]:
        import cv2

        from autodrivedata.perception.sem_bev import yolo11_instances
        from autodrivedata.utils.paths import project_path as _pp

        yolo11 = _YOLO_CACHE.setdefault("m", YOLO(str(_pp("weights/yolo11s-seg.pt"))))
        img = cv2.imread(str(root / cam.lower() / f"{frame:06d}.png"))
        if img is None:
            return []
        size = (img.shape[1], img.shape[0])
        return [
            PredInstance(cls_name="obstacle", mask=m, conf=c) for m, c in yolo11_instances(yolo11, img, size)
        ]

    return predict


#: 模型只加载一次(每个 root 一次预测,别每次调用都从盘上读 20 MB)。
_YOLO_CACHE: dict[str, Any] = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--frames", default="0-4", help="帧范围 0-19 或逗号列表")
    ap.add_argument("--cams", default=",".join(CAMS))
    ap.add_argument(
        "--backend",
        choices=("sam3", "yolo"),
        default="sam3",
        help="分割后端。**默认 sam3**(开放词表,概念由 THING_PROMPTS 给);`yolo` 是回退。"
        "⚠️ **两边 PQ/SQ/RQ 不可比**(类从哪来的口径不同),引用必须写明后端",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="**不出模型**:拿 GT 当预测灌进同一把尺子 ⇒ PQ/SQ/RQ 都必须恰好 1.0(空样本除外)。"
        "用它在怀疑判据本身时先验尺子 —— 键名/配对/分母错了在这里红,而出模型时只是个难看的数",
    )
    args = ap.parse_args()

    with runlog.run("autodrivedata.perception.inst_eval") as rl:
        root = project_path(args.root)
        rl.input(str(root), "root")
        rl.highlight("backend", args.backend)
        frames = _parse_frames(args.frames)
        cams = tuple(args.cams.split(","))
        predictor = None if args.self_test else load_instance_predictor(root, args.backend)
        res = eval_root(root, frames, predict=predictor, cams=cams)

        print(
            f"[data] {root.name} | {res['n_frames']} 帧 × {res['n_cams']} 路 | "
            f"GT 实例 {res['gt_instances']} / 预测 {res['pred_instances']}"
        )
        print(
            f"  TP={res['tp']}  FP={res['fp']}  FN={res['fn']}   "
            f"PQ={res['pq']:.4f}  SQ={res['sq']:.4f}  RQ={res['rq']:.4f}   "
            f"mask AP@{0.5}={res['mask_ap@0.5']:.4f}"
        )
        # ★ 自证:PQ = SQ × RQ 是**定义性质**,不是近似。空样本除外(那时是 nan)。
        if not np.isnan(res["pq"]) and abs(res["pq"] - res["sq"] * res["rq"]) > 1e-9:
            raise SystemExit(
                f"判据自证失败:PQ={res['pq']:.6f} ≠ SQ×RQ={res['sq'] * res['rq']:.6f} —— 实现错了"
            )
        if res["n_inconsistent_instances"]:
            print(
                f"  ⚠️ {res['n_inconsistent_instances']} 个实例的像素**跨多个语义类** —— "
                f"解码或渲染有问题,这些实例的类标签不可信"
            )
        if args.self_test:
            ok = all(abs(v - 1.0) < 1e-9 for v in (res["pq"], res["sq"], res["rq"]) if not np.isnan(v))
            print(f"[self-test] GT 当预测 ⇒ {'通过' if ok else '★ 失败'}(PQ/SQ/RQ 应恰好 1.0)")
            if not ok:
                raise SystemExit("判据自证失败:GT 当预测时 PQ/SQ/RQ 不是 1.0")
        rl.highlight("pq", round(res["pq"], 4) if not np.isnan(res["pq"]) else None)
        rl.highlight("sq", round(res["sq"], 4) if not np.isnan(res["sq"]) else None)
        rl.highlight("rq", round(res["rq"], 4) if not np.isnan(res["rq"]) else None)
        rl.highlight("mask_ap_0.5", round(res["mask_ap@0.5"], 4))
        rl.highlight("n_gt_instances", res["gt_instances"])
        rl.highlight("self_test", bool(args.self_test))


if __name__ == "__main__":
    main()

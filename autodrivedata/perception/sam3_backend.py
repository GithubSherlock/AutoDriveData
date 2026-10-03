"""SAM3 推理后端 —— **开放词表的检测/分割**(2026-10-02 起是本项目的默认后端)。

## ★ 换范式,不是换权重

现模型(`yolo11s_kitti` / `yolo11s-seg`)是**闭集**的:类表写死,模型判"这是什么类"。
SAM3 是**可提示分割基础模型**:你给一句文本,它把该概念的所有实例切出来。
⇒ **类别是提示词给的,不是模型判的。** 这句话的后果必须写死在引用口径里:
`eval_2d_ab` 报的 Car/Pedestrian/Cyclist AP 里那个"分类正确率"这一项,
在 SAM3 后端下退化成「**我的提示词写对没有**」。**跨后端比 AP 是把两件不同的事比大小。**

## 每条提示一次前向(**实测的硬约束**)

SAM3 吃**单概念**提示。拼串实测不可靠(单帧 `kitti_static_sem_v2/000010`):

| 提示 | 返回掩膜 |
|---|---|
| `"traffic cone"` | 2 |
| `"traffic cone and street barrier"` | 5 |
| `"car person bicycle"` | **0** |
| `"car, person, bicycle"` | **0** |

⇒ 想覆盖多个概念只能**逐个前向再并集**。代价是线性的:一帧 6 条提示 ≈ 3–5 s
(单条 0.5–1.0 s,实测)。`DETECT_PROMPTS` / `THING_PROMPTS` 就是为此写死的概念表。

## 依赖

`transformers` + `weights/sam3/`(HF 格式的 `model.safetensors`,3.44 GB)。
⚠️ 装它的时候只从 **清华源** 走 —— 本机 aliyun 源 403、pypi.org 超时(2026-10-02 实测),
`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers`。
⚠️ **加载 4.6 s / 3.15 GiB 常驻**,所以模型在模块级缓存,**不要每帧重建**。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: 权重目录(HF `from_pretrained` 口径,不是 `.pt` 单文件)。
DEFAULT_WEIGHTS = "weights/sam3"

#: 后处理阈值。**SAM3 的 score 与 YOLO 的 conf 不是同一个尺度** —— YOLO 那边 0.25 是
#: 校准过的默认,拿过来直接套会在 SAM3 上放进一大堆低分幻觉(实测见下表)。
#:
#: `surround_inst_demo` 5 帧 × 5 路、6 条 `THING_PROMPTS`,一次采样后离线扫:
#:
#: | conf | PQ | RQ | TP | FP | FN | maskAP |
#: |---|---|---|---|---|---|---|
#: | 0.25 | 0.5221 | 0.5838 | 54 | **67** | 10 | 0.7430 |
#: | 0.40 | 0.6575 | 0.7361 | 53 | 27 | 11 | 0.8061 |
#: | **0.50** | **0.7198** | 0.8000 | 52 | 14 | 12 | **0.8061** |
#: | 0.60 | 0.7434 | 0.8197 | 50 | 8 | 14 | 0.7273 |
#: | 0.70 | **0.7835** | 0.8571 | 48 | **0** | 16 | 0.7273 |
#:
#: ⇒ **阈值是"按指标选"的,不是模型的属性**:PQ 一路涨到 0.7(FP 清零,代价是 FN 10→16),
#: 而 **mask AP 在 0.5 见顶**(AP 吃排序,砍尾反而伤它)。
#: 默认取 **0.5** —— 那里 AP 最高、FP 从 54 掉到 14,是本项目两个指标都站得住的操作点。
#: ⚠️ **引用任何数都必须带上这个阈值**(同"AP 必须带 score_thr"那条红线)。
DEFAULT_THRESHOLD = 0.5

#: 掩膜去重的 IoU 阈值。**0.5 这个取值对结果不敏感** —— 实测同一批数据的两两 IoU 分布是
#: **干净的双峰**:重复掩膜 4 对 >0.9、不同物体 114 对 ≤0.1,**中间一对都没有**。
#: 所以 0.5–0.9 任何值都给出同一组去重结果(NMS 后 40→36)。
DEDUP_IOU = 0.5

#: **项目 GT 类 → 文本提示表**。这是"类别从哪来"的**唯一**出处 ——
#: 改这里等于改评测口径,别在调用点另写一份。
#: ⚠️ `Car` 要三条:COCO 的 truck/bus 在 KITTI 口径里都归 `Car`(见 `eval_2d_ab.COCO_FALLBACK`)。
DETECT_PROMPTS: dict[str, tuple[str, ...]] = {
    "Car": ("car", "truck", "bus"),
    "Pedestrian": ("person",),
    "Cyclist": ("bicycle", "motorcycle"),
}

#: `inst_eval` 的 "可数物体" 概念表。那边 GT 只有 `obstacle` 一类(things vs stuff),
#: 所以这边要**并集**出全部 things 再统一记成 `obstacle`。
THING_PROMPTS: tuple[str, ...] = (
    "car",
    "truck",
    "bus",
    "person",
    "bicycle",
    "motorcycle",
)


@dataclass(frozen=True)
class Det:
    """一条检测:类名(由**提示词**决定)+ 分数 + `xyxy` 像素框。"""

    cls_name: str
    conf: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Inst:
    """一个实例:概念名 + 分数 + 布尔掩膜 `(H, W)`。"""

    cls_name: str
    conf: float
    mask: np.ndarray


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter) / ua if ua > 0 else 0.0


def dedup_dets(dets: list[Det], iou_thr: float = DEDUP_IOU) -> list[Det]:
    """框版去重(同一条物体被 `car`/`truck`/`bus` 两条提示各检一次 ⇒ 判据里记两次)。"""
    order = sorted(range(len(dets)), key=lambda k: -dets[k].conf)
    kept: list[int] = []
    for i in order:
        if all(box_iou(dets[i].box, dets[j].box) <= iou_thr for j in kept):
            kept.append(i)
    return [dets[i] for i in sorted(kept)]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(inter) / union if union else 0.0


def dedup(masks: list[np.ndarray], scores: list[float], iou_thr: float = DEDUP_IOU) -> list[int]:
    """按 score 降序的贪心去重,返回**保留的下标**。

    ★ **为什么必须有这一层**:一条提示一次前向 ⇒ 同一个物体可能被 `car` / `truck` /
    `bus` 里两条同时切出来,它们 IoU≈1。而判据(AP / PQ)是**一对一**贪心匹配的,
    多出来的那些只能记成 FP —— 那不是"模型检错了",是**同一份证据被数了两次**。

    实测(5 帧 × 5 路,`surround_inst_demo`):去重把 **FP 67→54、mask AP 0.7430→0.8061、
    PQ 0.5221→0.5608**(conf 0.25);在 conf 0.5 上是 **PQ 0.699→0.720、maskAP 0.799→0.806**。
    0.063 的 AP 增益远超本项目 0.025 的跨权重选择下限 ⇒ **是实打实的**,不是抖动。
    """
    keep: list[int] = []
    for i in sorted(range(len(masks)), key=lambda k: -scores[k]):
        if all(mask_iou(masks[i], masks[j]) <= iou_thr for j in keep):
            keep.append(i)
    return sorted(keep)


_CACHE: dict[str, Any] = {}


def available() -> bool:
    """依赖在不在。**不加载模型**(那是 3.15 GiB + 4.6 s)。"""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def weights_path(weights: str | Path | None = None) -> Path:
    from autodrivedata.utils.paths import project_path

    return project_path(weights or DEFAULT_WEIGHTS)


def load(weights: str | Path | None = None, device: str = "cuda") -> tuple[Any, Any]:
    """`(model, processor)`,**模块级缓存** —— 每帧重建会把 4.6 s 乘进循环里。

    ⚠️ 缓存键含权重的**绝对路径**:换权重目录必须换一份,否则会静默用回旧的。
    """
    if not available():
        raise ImportError(
            "SAM3 后端要 `transformers`,本项目 env 里没装。装法(⚠️ 只有清华源通):\n"
            "  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers"
        )
    from transformers import Sam3Model, Sam3Processor

    root = str(weights_path(weights).resolve())
    key = f"{root}@{device}"
    if key not in _CACHE:
        model = Sam3Model.from_pretrained(root).to(device).eval()
        _CACHE[key] = (model, Sam3Processor.from_pretrained(root))
    return _CACHE[key]


def _raw(image: Any, text: str, threshold: float, device: str, weights: str | Path | None) -> dict:
    """一条提示一次前向 → `{masks, boxes, scores}`(原样,不解释)。"""
    import torch

    model, proc = load(weights, device)
    inp = proc(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    res = proc.post_process_instance_segmentation(
        out, threshold=threshold, target_sizes=inp["original_sizes"].tolist()
    )[0]
    del inp, out
    return res


def detect(
    image: Any,
    classes: tuple[str, ...] = ("Car", "Pedestrian", "Cyclist"),
    *,
    threshold: float = DEFAULT_THRESHOLD,
    device: str = "cuda",
    weights: str | Path | None = None,
    dedup_iou: float | None = DEDUP_IOU,
) -> list[Det]:
    """文本提示 → `[{cls_name, conf, box}]`。**每个项目类可能对应多条提示**,逐条前向再合并。

    返回的顺序**不保证按 conf 降序** —— 调用方(AP 那套)自己会排。这里不排是为了
    保留"哪条提示给的"这层信息在 `cls_name` 里(合并 truck/bus 时才知道归给了谁)。
    """
    out: list[Det] = []
    for cls in classes:
        prompts = DETECT_PROMPTS.get(cls)
        if prompts is None:
            raise KeyError(f"没有 {cls!r} 的提示词 —— 别在调用点现编,加进 DETECT_PROMPTS")
        for prompt in prompts:
            res = _raw(image, prompt, threshold, device, weights)
            for box, score in zip(res["boxes"].tolist(), res["scores"].tolist(), strict=True):
                out.append(Det(cls, float(score), tuple(float(v) for v in box)))  # type: ignore[arg-type]
    return dedup_dets(out, dedup_iou) if dedup_iou is not None else out


def segment(
    image: Any,
    concepts: tuple[str, ...],
    *,
    cls_name: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    device: str = "cuda",
    weights: str | Path | None = None,
    dedup_iou: float | None = DEDUP_IOU,
) -> list[Inst]:
    """概念表 → `[{cls_name, conf, mask}]`。`cls_name` 给定时**统一改名**
    (`inst_eval` 那边 GT 只有 `obstacle` 一类,多条概念要并成一个类)。"""
    out: list[Inst] = []
    for concept in concepts:
        res = _raw(image, concept, threshold, device, weights)
        name = cls_name if cls_name is not None else concept
        # ⚠️ 掩膜走 `.cpu().numpy()`,**不走 `.tolist()`** —— 后者把 (N,H,W) 展成 Python
        #   嵌套列表,一帧 1242×375 就是百万级对象,比前向本身还慢。
        masks = res["masks"].cpu().numpy() if hasattr(res["masks"], "cpu") else np.asarray(res["masks"])
        for mask, score in zip(masks, res["scores"].tolist(), strict=True):
            out.append(Inst(name, float(score), np.asarray(mask, dtype=bool)))
    if dedup_iou is not None and out:
        keep = dedup([i.mask for i in out], [i.conf for i in out], dedup_iou)
        out = [out[k] for k in keep]
    return out

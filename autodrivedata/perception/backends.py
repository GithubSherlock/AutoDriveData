"""检测/分割的**后端抽象** —— 把"用哪个模型"挡在评测逻辑之外(2026-10-02 起默认 SAM3)。

三个消费者(`eval_2d_ab` / `eval_attr` / `eval_fusion` / `mono_distance`)过去各自
`YOLO(args.weight)` 一次、各读一次 `res.boxes`。抽到这里是为了**只留一处口径** ——
"类名从哪来"这件事在两个后端下**语义不同**,写在四处必然漂。

## ★ 两个后端的差别不是速度,是"类别从哪来"

| | `yolo`(回退) | `sam3`(**默认**) |
|---|---|---|
| 范式 | 闭集检测器 | 可提示分割基础模型 |
| 类名 | **模型判的** | **提示词给的**(`sam3_backend.DETECT_PROMPTS`) |
| 一次前向 | 出全部类 | **一个概念一次**(见 `sam3_backend` 头注的实测) |

⇒ **两边的 AP/PQ 不可比。** yolo 那边 AP 里含一个"分类正确率";sam3 那边这一项
退化成「我的提示词写对没有」。**引用任何数字都必须写明后端。**
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

from PIL import Image

from autodrivedata.perception import sam3_backend

#: 项目 GT 类(与 `eval_2d_ab.GT_CLASSES` 同一套 KITTI 口径)。
GT_CLASSES = ("Car", "Pedestrian", "Cyclist")

#: COCO 名 → 项目类。yolo 分支用(`truck`/`bus` 在 KITTI 口径里都归 `Car`)。
COCO_FALLBACK = {
    "car": "Car",
    "truck": "Car",
    "bus": "Car",
    "person": "Pedestrian",
    "bicycle": "Cyclist",
    "motorcycle": "Cyclist",
}

BACKENDS = ("sam3", "yolo")


def norm_cls(name: str) -> str:
    """类名归一 —— 两个后端都走它,免得各写一份。"""
    n = name.strip().lower()
    if n in COCO_FALLBACK:
        return COCO_FALLBACK[n]
    for c in GT_CLASSES:
        if n == c.lower():
            return c
    return ""


class Predictor(Protocol):
    """给一张图,回 `[(项目类名, conf, xyxy)]`。**类名已归一**(不认识的不返回)。"""

    def __call__(self, path: Path) -> list[tuple[str, float, tuple[float, float, float, float]]]: ...


class YoloPredictor:
    """ultralytics 闭集检测器(**回退后端**)。"""

    def __init__(self, weight: str, conf: float) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weight)
        self.conf = conf
        self.names = self.model.names

    def __call__(self, path: Path):
        from ultralytics.engine.results import Results

        # predict 返回 union(Iterator | list),先 materialize 再取首帧
        res = cast(Results, list(self.model.predict(path, conf=self.conf, verbose=False, device=0))[0])
        out: list[tuple[str, float, tuple[float, float, float, float]]] = []
        for b in res.boxes or []:  # 无检测帧 boxes=None
            c = norm_cls(self.names[int(b.cls.item())])
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append((c, float(b.conf.item()), (x1, y1, x2, y2)))
        return out


class Sam3Predictor:
    """SAM3 开放词表(**默认后端**)。一个项目类 = 多条文本提示,逐条前向。"""

    def __init__(self, conf: float) -> None:
        self.conf = conf

    def __call__(self, path: Path):
        img = Image.open(path).convert("RGB")
        dets = sam3_backend.detect(img, GT_CLASSES, threshold=self.conf)
        return [(d.cls_name, d.conf, d.box) for d in dets]


def make_predictor(backend: str, weight: str, conf: float) -> Predictor:
    """按后端名造预测器。**让"默认走哪条"只有一个出口** —— 各入口自己写 if 迟早不一致。"""
    if backend == "sam3":
        return Sam3Predictor(conf)
    return YoloPredictor(weight, conf)


#: **KITTI 微调的 yolo11s-seg/检测权重**的默认路径。★ **唯一落点** ——
#: 2026-10-06 实测踩到:这个路径原本只写在一个 CLI 的 `add_argument` 里,
#: 而新写的两个入口把 `--weight` 默认成 `""` ⇒ `--backend yolo` **根本跑不起来**
#: (`TypeError: model='' is not a supported model format`),而报错在 ultralytics 深处,
#: **指不到"你没给权重"**。凡是要走 yolo 的入口,默认值**一律取这个常量**。
DEFAULT_YOLO_WEIGHT = (
    "/root/autodl-tmp/Documents/Projects/AutoLabel/auto2dlabel/weights/"
    "kitti_finetune/yolo11s_kitti/weights/best.pt"
)

#: 各后端的**默认置信度阈值**。**两个数不是一回事,不能互相套用**:
#: `0.25` 是 YOLO 那边校准过的默认;SAM3 的 score 尺度不同,套 0.25 会放进一大堆
#: 低分幻觉(实测 `person` 这一条提示单帧就 53 个落在建筑/杆/Static 上的假人)。
#: 定标过程与"阈值按指标选"那条见 `sam3_backend.DEFAULT_THRESHOLD` 的实测表。
DEFAULT_CONF: dict[str, float] = {"sam3": sam3_backend.DEFAULT_THRESHOLD, "yolo": 0.25}


def resolve_conf(backend: str, conf: float | None) -> float:
    """CLI 的 `--conf` 默认给 `None`,在这里落到**该后端自己的**默认值。

    ⚠️ 写成"`--conf` 默认 0.25、sam3 时再改"就晚了 —— 用户显式传 0.25 与不传
    分不开,而那两件事的意图完全不同(一个是我要这个阈值,一个是随默认)。
    """
    return DEFAULT_CONF[backend] if conf is None else conf


def describe(backend: str, predict: Predictor) -> str:
    """开跑时打印"类名从哪来" —— 事后翻日志能直接看出这份数是哪个后端的。"""
    if backend == "sam3":
        return f"backend=sam3(开放词表;提示词 {sam3_backend.DETECT_PROMPTS})"
    return f"backend=yolo(闭集;names={cast(YoloPredictor, predict).names})"

"""MapTR 变体表的**唯一落点** —— 选哪套架构、怎么建、权重归属谁。

**为什么单独成文件**:两支的差异只有「怎么建模型」这一件事。`eval_maptr` /
`viz_maptr_pred` / [`live_common`](../../sim/live_common.py) 里与变体相关的代码**各只有一行**
(构造 `MapTR(...)`),其余全部由 `MapTR.forward` 的输出契约保证与变体无关 ——
两支吐同一个 `{"pred_logits", "pred_points"}`。所以**口径只留一份实现,变体逻辑也只留一份**。

反面做法是把 eval/viz/studio 各复制一份给 MapQR:那会让 `eval_maptr` 的 chamfer AP 出现
**第二份实现**,直接违反 CLAUDE.md 红线「AP 口径只能有一份实现」(这条的来历就是
`train_maptr` 当初不许抄 chamfer AP);studio 两份 800–970 行的复制还要永久同步,
而差异只有一行。

**两个入口**:

- 训练:`--variant {maptrv2,mapqr}`(聚合开关)或细粒度 `--scatter-gather` /
  `--bev-encoder`(**两者同时给报错**,见 `resolve_variant`);
- 评测/可视化:直接读 checkpoint 自带的 `model_kwargs`,**不必也不能**再指定变体。
"""

from __future__ import annotations

from pathlib import Path

import torch

from autodrivedata.map.maptr.model import MapTR, load_map_weights

# 变体名。`maptrv2` = 自实现基线(默认);`mapqr` = 散聚 query + 高度核 BEV 编码器
VARIANT_BASELINE = "maptrv2"
VARIANT_MAPQR = "mapqr"

# **新增变体只改这一张表**:展开成 `MapTR` 的两个细粒度开关
VARIANTS: dict[str, dict[str, object]] = {
    VARIANT_BASELINE: {"scatter_gather": False, "bev_encoder": "none"},
    VARIANT_MAPQR: {"scatter_gather": True, "bev_encoder": "height_kernel"},
}


def variant_names() -> tuple[str, ...]:
    """命令行 choices 用(顺序稳定,便于 `--help`)。"""
    return tuple(VARIANTS)


def expand_variant(name: str) -> dict[str, object]:
    """变体名 → 两个细粒度开关。未知名字**报错**(不静默退回基线)。"""
    if name not in VARIANTS:
        raise SystemExit(f"未知变体 {name!r};可选 {list(VARIANTS)}")
    return dict(VARIANTS[name])


def resolve_variant(
    variant: str | None, *, scatter_gather: bool = False, bev_encoder: str = "none"
) -> dict[str, object]:
    """把「聚合开关」与「细粒度开关」合并成一份 flags。

    **同时给就报错**,不定义"谁覆盖谁"。理由:隐式覆盖规则正是本项目反复吃亏的地方
    (过拟合闸门误判那次的根因就是 `--frames 0` 的语义被类推错了)。要细粒度就别给 `--variant`,
    要做消融(只开一半)就用细粒度开关 —— 两条路都明确。
    """
    if variant is None:
        return {"scatter_gather": scatter_gather, "bev_encoder": bev_encoder}
    if scatter_gather or bev_encoder != "none":
        raise SystemExit(
            f"--variant {variant} 与 --scatter-gather / --bev-encoder 互斥:聚合开关与细粒度开关"
            "同时给时「谁覆盖谁」没有好默认 —— 要细粒度(做消融)就别给 --variant"
        )
    return expand_variant(variant)


def build_map_model(variant: str = VARIANT_BASELINE, **overrides: object) -> MapTR:
    """按变体建模型;`overrides` 覆盖变体默认值(如 `num_vec` / `temporal_window`)。"""
    kwargs = expand_variant(variant)
    kwargs.update(overrides)
    return MapTR(**kwargs)  # type: ignore[arg-type]


def save_map_checkpoint(model: MapTR, path: str | Path) -> None:
    """落盘 `{"state_dict": …, "model_kwargs": …}` —— 权重**自带结构说明**。

    旧形态(裸 state_dict)仍可读(`model.read_map_state_dict` 兼容两种),故本改动
    不破坏任何既有权重(如 `maptr_v2_singleF.pt`)。
    """
    torch.save({"state_dict": model.state_dict(), "model_kwargs": model.model_kwargs()}, path)


def read_map_meta(path: str | Path, dev: torch.device | None = None) -> dict:
    """读 checkpoint 的结构元信息,返回 `{"name", "model_kwargs", "source"}`。

    **裸 state_dict 一律归为基线,这是事实而非猜测**:变体开关是 2026-09-27 才加的,
    在那之前落盘的权重只可能是基线结构。若判错(拿基线结构去载变体权重),
    `load_map_weights` 会因键不匹配**当场报错**,不会静默跑错。
    """
    obj = torch.load(path, map_location=dev or torch.device("cpu"))
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        kwargs = dict(obj.get("model_kwargs") or {})
        return {"name": _name_of(kwargs), "model_kwargs": kwargs, "source": "self-describing"}
    return {"name": VARIANT_BASELINE, "model_kwargs": {}, "source": "legacy-bare-state-dict"}


def _name_of(kwargs: dict) -> str:
    """由 model_kwargs 反查变体名;对不上任何已知变体时返回 `"custom"`。"""
    flags = {
        "scatter_gather": kwargs.get("scatter_gather", False),
        "bev_encoder": kwargs.get("bev_encoder", "none"),
    }
    for name, spec in VARIANTS.items():
        if spec == flags:
            return name
    return "custom"


def load_map_model(path: str | Path, dev: torch.device, **overrides: object) -> tuple[MapTR, dict]:
    """从 checkpoint 建出**结构正确**的模型并载入权重;返回 (model, meta)。

    变体由 checkpoint 自带(旧裸权重 → 基线);`overrides` 覆盖元信息里的同名项
    (eval 侧目前只用它传 `temporal_window`)。**不调 `model.eval()`** ——
    加载器不该替调用方决定 train/eval 模式,三个消费方本来就各自显式调。
    """
    meta = read_map_meta(path, dev)
    kwargs = {**meta["model_kwargs"], **overrides}
    kwargs.pop("pretrained", None)  # 权重从 checkpoint 来,不再拉 ImageNet
    model = MapTR(pretrained=False, **kwargs).to(dev)  # type: ignore[arg-type]
    load_map_weights(model, str(path), dev)
    return model, meta

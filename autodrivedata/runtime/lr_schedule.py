"""训练学习率调度(**纯值**,零 torch):`halve` 阶梯 / `cosine` 退火。

## 为什么要它

`train_maptr` 原来只有两种形态:每 N epoch 减半(`--lr-halve`,默认 12),或**完全不衰减**
(`--lr-halve 0`)。而 **MapQR 官方配方两者都不是** —— 它是
**AdamW lr 6e-4 + 500-iter 线性 warmup(从 lr/3 起) + CosineAnnealing(min_lr_ratio 1e-3)**。

这不是细节。§P-V25 八 的四档消融跑在 `--lr-halve 0`(恒定 1e-4)下,而变体末段 loss 的
降幅是基线的 **4.6×**(③ 0.493 / 基线 0.108) —— **一个"已经平了"的模型与一个"还在快速下降"
的模型,比出来的差不该读成"架构更差"**。本模块把那个**形状**补上,好把「架构」与「没训完」拆开。

## 口径(两处换算写在这里,别在调用点另推一份)

1. **warmup 与总长都以 epoch 计** —— 与既有的 `--lr-halve` 同口径。
   ⚠️ 官方是 **500 iteration**;它的 dataloader 是 ~879 it/ep ⇒ 约 **0.57 ep**。
   我们 320 帧 / batch 3 是 **107 it/ep** ⇒ 500 iter ≈ **4.7 ep**。**同一个 500 换过来差 8 倍**,
   所以命令行给的是 epoch 数,不是照抄 500。
2. **`warmup_ratio` 是起点比例**(mmcv 语义):warmup 段从 `lr·ratio` 线性升到 `lr`。
3. **逐 epoch 取值,且取的是该 epoch 开头的进度** —— 每个 epoch 只有一个 lr。
   后果:cosine 的**末帧略高于 `lr·min_lr_ratio`**(`prog = (epochs-1-warmup)/(epochs-warmup)`),
   要 `prog` 到 1 才恰好落 floor(由 `min(...,1.0)` 封顶)。官方按 iteration 走,
   是在最后一个 iter 才落到 floor —— **两者差一个 epoch 的宽度,是口径不是 bug**。
   钉在 `test_approaches_the_floor_and_never_reaches_zero`。

## 向后兼容

- **`halve` 是默认**,且 `warmup=0`(默认)时与改造前**逐位相同**。
- ⚠️ **唯一的差别在 `--warmup > 0`**:warmup 段的倍率由 `epoch/warmup` 改为
  官方的 `ratio + (1-ratio)·(epoch-1)/warmup`(1-based)。实测**归档里 0 次训练用过
  `--warmup`**(13 次全是 `--lr-halve 0`)⇒ 没有已归档的数受影响。
- **不把默认改成 `cosine`**:已归档的 §P-V11 四档数是 `halve` 口径,改默认等于让它们作废。
"""

from __future__ import annotations

import math

#: 允许的调度名。新增一条要同时改这里与 `train_maptr` 的 `--lr-schedule` choices。
SCHEDULES = ("halve", "cosine")

#: MapQR 官方 `mapqr_nusc_r50_24ep.py` 的配方(写在常量里便于引用与对账,不在调用点硬编码)。
#: ⚠️ 那套是 **batch 4/GPU × 多卡**;我们是 batch 3 单卡,峰值**不能照抄** ——
#: 见 `train_maptr --lr` 的说明。
PAPER_LR = 6e-4
PAPER_WARMUP_ITERS = 500
PAPER_WARMUP_RATIO = 1.0 / 3
PAPER_MIN_LR_RATIO = 1e-3


def warmup_mult(epoch: int, warmup: int, ratio: float) -> float:
    """warmup 段的倍率(epoch 是 1-based);不在 warmup 段返回 `1.0`。

    起点恰为 `ratio`(mmcv 语义:`ratio + (1-ratio)·iter/warmup_iters`,iter 从 0 起),
    末点略低于 1 —— 与官方一致,且**不为了"好看"把它凑成 1.0**。
    """
    if warmup <= 0 or epoch > warmup:
        return 1.0
    return ratio + (1.0 - ratio) * (epoch - 1) / warmup


def lr_at(
    epoch: int,
    *,
    base: float,
    epochs: int,
    schedule: str = "halve",
    lr_halve: int = 12,
    warmup: int = 0,
    warmup_ratio: float = 0.0,
    min_lr_ratio: float = PAPER_MIN_LR_RATIO,
) -> float:
    """第 `epoch`(1-based)的学习率。**纯函数** —— 与训练循环无耦合,可直接钉。

    - `halve`:每 `lr_halve` 个 epoch 减半(`lr_halve=0` ⇒ 恒定);再乘 warmup 倍率;
    - `cosine`:warmup 段线性升,之后从 `base` **余弦退火**到 `base·min_lr_ratio`。

    `epochs` 只被 `cosine` 用(退火终点);`halve` 不依赖它 —— 那条路径是纯阶梯。
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"未知调度 {schedule!r};可选 {list(SCHEDULES)}")
    if epoch < 1:
        raise ValueError(f"epoch 是 1-based,收到 {epoch}")
    if base <= 0.0:
        raise ValueError(f"base lr 必须 > 0,收到 {base}")

    if schedule == "halve":
        peak = base * (0.5 ** ((epoch - 1) // lr_halve)) if lr_halve > 0 else base
        return peak * warmup_mult(epoch, warmup, warmup_ratio)

    if warmup > 0 and epoch <= warmup:
        return base * warmup_mult(epoch, warmup, warmup_ratio)
    # 退火跨度 = 总 epoch 减 warmup;`max(...,1)` 挡住 warmup ≥ epochs 时的除零。
    # `prog` 用 `epoch-1`(与 mmcv 的 0-based iter 同口径):warmup 后第一帧恰为 `base`。
    span = max(epochs - warmup, 1)
    prog = min((epoch - 1 - warmup) / span, 1.0)
    floor = base * min_lr_ratio
    return floor + (base - floor) * (1.0 + math.cos(math.pi * prog)) / 2.0

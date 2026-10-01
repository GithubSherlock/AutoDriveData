"""训练早停:**两段式**平台检测(纯值,不 import torch/carla —— 测试无需 GPU)。

## 为什么是两段,而不是只看 loss

只看 loss 平台会误停,因为 **loss 与 AP 可以背离**:2026-09-29 实测同一次训练里,
K=3b 相对基线**训练集 +47% / 帧级留出 +32%**,而**路线级留出 −0.21%**。
真正要的是"泛化到顶没到顶",那个信号只有一个来源:留出集评估。但留出评估贵
(72 帧 @batch1 ≈ 90 s),不可能每 epoch 做。

故:**loss 平台(便宜、每 epoch 可判)只作候选;触发后才花一次留出评估做裁决。**

## 触发理由**必须分型**(处置不同)

| 理由 | 含义 | 该做什么 |
|---|---|---|
| `converged` | 真收敛:lr 还在,loss 不再降 | 可以收工 |
| `lr_exhausted` | lr 已衰减到初始的 <`lr_exhausted_frac` | **也停**(再跑训不动),但该去调 lr 计划重跑 —— Plan2 已记「400 epoch 跑完 lr ~1e-8,平台是 lr 归零不是收敛」 |

不分型的话,"停"这个动作在两种情形下长得一样,而事后无从判断该不该重跑。

## 与调用方的边界

本模块**不知道 AP 是什么**:复核函数由调用方注入(`confirm_fn: epoch -> float | None`)。
返回 `None` = 这次复核做不了(没配留出选择器 / 评估失败)⇒ **不敢停**,继续跑 ——
这条是安全侧:判据缺失时宁可多跑,不可误停。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class PlateauVerdict:
    """第一段(loss)的判定。`reason` 见模块头注的分型表。"""

    plateaued: bool
    reason: str
    best_loss: float
    best_epoch: int
    stale: int


@dataclass
class StopDecision:
    """第二段(复核后)的最终判定 —— 调用方只消费这个。"""

    should_stop: bool = False
    reason: str = "running"  # running / waiting_for_confirm / confirm_unavailable / ap_still_rising / converged / lr_exhausted
    confirm_ap: float | None = None  # 本次复核值(None = 这轮没做复核)
    best_ap: float | None = None
    best_ap_epoch: int | None = None
    detail: dict = field(default_factory=dict)


class PlateauDetector:
    """第一段:loss 的相对改善是否已经停滞。

    判据 = **相对**改善 `loss < best · (1 − min_improve)` 连续 `patience` 个 epoch 未出现。
    用相对而非绝对,是因为 loss 尺度随配置变(K>1 时序模型与单帧基线不是一个量级,
    实测 15→2.68 对 15→?),写死绝对值必然在某一侧失效。
    """

    def __init__(
        self,
        *,
        patience: int = 20,
        min_improve: float = 1e-3,
        lr_exhausted_frac: float = 0.01,
        warmup: int = 0,
    ) -> None:
        self.patience = patience
        self.min_improve = min_improve
        self.lr_exhausted_frac = lr_exhausted_frac
        self.warmup = warmup  # 前 N 个 epoch 不判(lr 还在爬升,loss 必降)
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.stale = 0
        self.lr_max = 0.0

    def update(self, epoch: int, loss: float, lr: float) -> PlateauVerdict:
        self.lr_max = max(self.lr_max, lr)
        # warmup 期 lr 线性升温 ⇒ loss 必降,此时判平台没有意义
        if epoch <= self.warmup:
            self.best_loss, self.best_epoch = loss, epoch
            return PlateauVerdict(False, "warmup", loss, epoch, 0)

        if loss < self.best_loss * (1.0 - self.min_improve):
            self.best_loss, self.best_epoch = loss, epoch
            self.stale = 0
            return PlateauVerdict(False, "improving", self.best_loss, self.best_epoch, 0)

        self.stale += 1
        if self.stale < self.patience:
            return PlateauVerdict(False, "improving", self.best_loss, self.best_epoch, self.stale)

        # **分型**:平台是"收敛"还是"lr 已经没劲了"—— 后者也停,但处置不同
        lr_dead = self.lr_max > 0 and lr <= self.lr_max * self.lr_exhausted_frac
        return PlateauVerdict(
            True,
            "lr_exhausted" if lr_dead else "converged",
            self.best_loss,
            self.best_epoch,
            self.stale,
        )


class EarlyStopper:
    """第二段:平台候选 → 复核留出指标 → 裁决。

    **至少要两次成功的复核才可能判停**:第一次复核只用来建立 `best_ap` 基线
    (没有参照,"还在升吗"无从谈起)。故实际最少跑 `patience + 2×confirm_every` 个 epoch。
    """

    def __init__(
        self,
        detector: PlateauDetector,
        confirm_fn: Callable[[int], float | None],
        *,
        confirm_every: int = 20,
        ap_min_improve: float = 1e-2,
    ) -> None:
        self.det = detector
        self.confirm_fn = confirm_fn
        self.confirm_every = confirm_every
        # **门槛必须远宽于噪声**:AP 的复现性下限是 2e-3(CLAUDE.md 红线),
        # 判据贴着它会在噪声里打转、动不动就停。
        self.ap_min_improve = ap_min_improve
        self.best_ap: float | None = None
        self.best_ap_epoch: int | None = None
        self.last_confirm_epoch = -(10**9)
        self.n_confirms = 0

    def update(self, epoch: int, loss: float, lr: float) -> StopDecision:
        v = self.det.update(epoch, loss, lr)
        d = StopDecision(
            best_ap=self.best_ap,
            best_ap_epoch=self.best_ap_epoch,
            detail={"plateau_reason": v.reason, "best_loss": v.best_loss, "best_loss_epoch": v.best_epoch},
        )
        if not v.plateaued:
            return d

        # 平台候选,但复核还没到节奏 —— 等(不做评估,免得每个 epoch 都花 90 s)
        if epoch - self.last_confirm_epoch < self.confirm_every:
            d.reason = "waiting_for_confirm"
            return d

        ap = self.confirm_fn(epoch)
        self.last_confirm_epoch = epoch
        if ap is None:
            # **判据缺失 ⇒ 不敢停**(安全侧:宁可多跑,不可误停)
            d.reason = "confirm_unavailable"
            return d
        self.n_confirms += 1
        d.confirm_ap = ap

        if self.best_ap is None or ap - self.best_ap >= self.ap_min_improve:
            self.best_ap, self.best_ap_epoch = ap, epoch
            d.best_ap, d.best_ap_epoch = ap, epoch
            d.reason = "ap_still_rising"  # 第一次复核也走这里:只建立基线,不停
            return d

        d.should_stop = True
        d.reason = v.reason  # converged / lr_exhausted
        return d

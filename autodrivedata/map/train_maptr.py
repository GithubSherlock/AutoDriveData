"""MapTR 训练入口(§5.11 C 阶段):单帧过拟合 = 正确性锚点;多帧 = 常规训练。

单帧模式(--frames 1,默认)目标:总损失压到 ~0(分类全中 + 点回归收敛)。这是
"参考实现数学正确"的最强自证——任何坐标口径 / 匹配 / 损失错误都会把单帧
损失卡在高位,无法过拟合。通过判据:最后 20 步平均 total < 0.5,否则退出码 1。
多帧模式(--frames N > 1,或 --frames 0 = 不截断):常规训练,无过拟合判据,收敛量级看
D 阶段 chamfer AP。**闸门按实际训练样本数判**(见 `is_single_frame_anchor`),不按
`--frames` 标志 —— `--frames 0` 是"不截断",按标志判会把几百帧的训练误判成单帧锚点。

匹配(匈牙利)在 CPU 上做(每步 detach 后),不进入训练图——与官方训练流程一致。

batch 口径:--batch 0(默认)= 自适应实测(空闲显存 × `runtime.device.SAFETY_FACTOR` / 每样本训练步增量,
上限 --max-batch;参考 AutoLabel tools/device.py);--batch N>0 = 显式指定
(显式 > 实测)。自适应探针 = 完整训练步 ×2(batch 1 warmup + batch 2 增量),
含 optimizer.step,见 autodrivedata/runtime/device.py。

用法:
  python -m autodrivedata.map.train_maptr --infos outputs/surround_drive/map_infos.json \
      --root outputs/surround_drive --frames 1 --epochs 400 --out outputs/maptr_overfit.pt

每次运行落 `logs/` 三件套(脚本/时间/argv/git/GPU + 逐 epoch 指标 + 产物带哈希),
`--no-runlog` 关。给了 `--eval-*` 选择器则**训练结束后自动评一次留出 mAP**
(把 `--eval-seg` 设成训练时 `--exclude-seg` 的那一段 = 路线级留出);没给就明说不评。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from autodrivedata.map.eval_maptr import evaluate
from autodrivedata.map.maptr.dataset import (
    MapTRDataset,
    collate,
    parse_frame_range,
    parse_segs,
    select_frames,
)
from autodrivedata.map.maptr.head import maptr_loss, match_assign
from autodrivedata.map.maptr.model import MapTR, load_map_weights
from autodrivedata.map.maptr.variants import resolve_variant, save_map_checkpoint, variant_names
from autodrivedata.runtime.device import SAFETY_FACTOR, get_gpu_free_memory_gb, tune_train_batch_size
from autodrivedata.runtime.early_stop import EarlyStopper, PlateauDetector
from autodrivedata.runtime.lr_schedule import SCHEDULES, lr_at
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path


def _to_device(images: dict | list, dev: torch.device):
    """单帧 dict / 时序 list[dict] 两种结构都搬到 `dev`(保持结构,模型按类型分派)。"""
    if isinstance(images, list):
        return [{n: t.to(dev) for n, t in frame.items()} for frame in images]
    return {n: t.to(dev) for n, t in images.items()}


#: AMP 用 bf16 而不是 fp16:**3090(Ampere)原生支持 bf16,且不需要 `GradScaler`**
#: (fp16 要缩放损失防下溢,那是一条独立的失效模式)。见 `--amp` 的说明。
AMP_DTYPE = torch.bfloat16


def _train_batch(
    model: MapTR,
    opt: torch.optim.Optimizer,
    dev: torch.device,
    ds: MapTRDataset,
    batch: dict,
    amp: bool = False,
) -> tuple[float, float]:
    """跑一个完整训练步(forward+匈牙利匹配+loss+backward+step),返回 (cls, pts) 损失。

    同时用作自适应 batch 的探针步(measure_batch_memory 的 step_fn):探针必须与
    真实训练步完全同构,否则每样本显存增量测不准 —— **`amp` 必须一路传下去**,
    否则探针量的是 fp32 的显存,而真跑用 bf16(或反过来),选出的 batch 当场 OOM。

    ## `amp=True` 时**只**包 forward(2026-10-03)

    实测单步分阶段计时(`num_workers=0`,基线):forward 640 ms / 匹配 18 ms /
    loss+backward 645 ms / opt 5 ms —— GPU 火力全在 forward 与 backward 上,
    所以 autocast 只包这一段。**匹配不在 autocast 里**:它走
    `.cpu().numpy()` + scipy 匈牙利,是 fp32 的 CPU 路径,包不包都一样。
    ⚠️ 实测 `match` 只占 **0.7%** —— 决定 "要不要把匹配挪到 GPU" 之前先看这个数。
    """
    images = _to_device(batch["images"], dev)
    poses = batch["poses"].to(dev)
    with torch.autocast(device_type=dev.type, dtype=AMP_DTYPE, enabled=amp):
        out, _ = model(images, poses, ds.calibs)
    # ★ 发散守卫(**买的是"报错能归因",不是"防发散"**):预测一旦出 NaN/Inf,
    # 原来的报错落在**下一站的 scipy**上 —— `ValueError: matrix contains invalid
    # numeric entries`(实测 lr 6e-4 @ batch 3:warmup 结束那一刻触顶即发散),
    # 那句话既不说哪一帧、也不说跟学习率有关,查起来先怀疑匹配、白绕一圈。
    # 代价是一次 `.item()` 的同步,张量只有 B×Nq×(C+1) 量级(实测可忽略)。
    if not torch.isfinite(out["pred_logits"]).all().item():
        raise RuntimeError(
            "预测出现 NaN/Inf —— 训练已发散。**最可能是学习率超过了这个 batch 能承受的值**"
            "(实测 `lr 6e-4 @ --batch 3` 在 warmup 结束、lr 触顶那一 epoch 发散;"
            "论文的 6e-4 是 batch≈32 的值,按批缩放约为 5.6e-5)。"
        )
    # 匹配在 autocast 之外:`.numpy()` 不接受 bf16(fp32 的 CPU 路径),而且它本来就不吃 GPU
    cls_ts, pts_ts, masks = [], [], []
    for bi in range(poses.shape[0]):
        cls_t, pts_t, mask = match_assign(
            out["pred_points"][bi].detach().float().cpu().numpy(),
            out["pred_logits"][bi].detach().float().cpu().numpy(),
            batch["gts"][bi],
            model.num_classes,
            model.num_vec,
        )
        cls_ts.append(cls_t)
        pts_ts.append(pts_t)
        masks.append(mask)
    with torch.autocast(device_type=dev.type, dtype=AMP_DTYPE, enabled=amp):
        loss = maptr_loss(
            out["pred_logits"],
            out["pred_points"],
            torch.tensor(np.stack(cls_ts), device=dev),
            torch.tensor(np.stack(pts_ts), device=dev),
            torch.tensor(np.stack(masks), device=dev),
        )
        total = loss["total"]
    opt.zero_grad()
    total.backward()
    opt.step()
    return float(loss["cls"].detach()), float(loss["pts"].detach())


def _load_opt_sidecar(opt: torch.optim.Optimizer, args: argparse.Namespace, dev: torch.device) -> None:
    """`--init-ckpt X` 且 X.opt 存在时载入优化器状态(Adam 力矩不重置)。"""
    if args.no_opt or not args.init_ckpt:
        return
    path = Path(str(args.init_ckpt) + ".opt")
    if not path.exists():
        if args.warmup <= 0:
            print(f"[opt] 无侧车 {path}——Adam 力矩重置,建议加 --warmup 抑制重启尖峰")
        return
    side = torch.load(path, map_location=dev)
    opt.load_state_dict(side["optimizer"])
    print(f"[opt] 载入优化器状态 {path}(存盘于 epoch {side.get('epoch', '?')},力矩不重置)")


#: 回评对照的容差。取 **2e-3 = 项目记录的复现性下限**(CLAUDE.md 红线):
#: 同一份权重、同一批帧、同一条评测脚本,换进程也不该动超过这个量。
#: 之所以敢这么紧:回评的输入与复核时被评的那份**逐字节相同**(`copyfile` 是原样拷贝)。
RESTORE_TOL = 2e-3


def _restore_mismatch(claimed: float | None, measured: float | None, tol: float = RESTORE_TOL) -> str | None:
    """回评对照的**判据**(纯值,可单测):返回 `None` = 通过,否则返回人读的失败描述。

    ★ **这是防 §P-M.20 那一类的唯一机械手段。** 那次「恢复 best-AP 权重」被后面的无条件存盘
    静默覆盖:日志照写 0.1557、盘上却是 0.1243,**任何一处的输出都看不出** ——
    唯一的发现方式是手工回评字节、再逐位比 `pred/gt` 计数,纯属凑巧(复核点恰好落在同一 epoch)。

    判据是「**回评值 == 声称值**」,不是「回评值高不高」—— 它防的是**产物与自述不一致**,
    与模型好坏无关。

    三种"判不了"的情形都返回 `None`(**不误报**),由调用方各自明说:
    `claimed is None`(没成功复核过,没有声称值可对)、`measured is None`(回评没做/失败)、
    两者皆非有限数。**"判不了"与"通过"必须由调用方区分开记** —— 静默把判不了当通过,
    就是本函数要消灭的那类误读。
    """
    if claimed is None or measured is None:
        return None
    if not (math.isfinite(claimed) and math.isfinite(measured)):
        return None
    if abs(measured - claimed) <= tol:
        return None
    return (
        f"落盘权重回评 {measured:.4f} ≠ 日志声称的 best-AP {claimed:.4f}"
        f"(差 {abs(measured - claimed):.4f} > 容差 {tol:g})—— **产物与自述不一致**"
    )


def _auto_eval(args: argparse.Namespace, rl: runlog.RunLogger) -> dict | None:
    """训练尾部的留出集评估 —— **转发给 `eval_maptr.evaluate()` 那一份实现**。

    chamfer AP 若在这里抄成第二份,两处必然漂移,而"同一权重在两个脚本里报出不同 AP"
    是最难查的一类结论失效(见 eval_maptr 模块 docstring)。故这里只做参数转发与登记。

    **没给选择器就不评,并且明说** —— 静默跳过会让"这次有日志"被读成"这次有 mAP",
    而这正是引入日志要消灭的那类误读。`--no-eval` 是主动关,措辞与"没给选择器"分开。

    返回 `res`(未评则 `None`)给调用方做**回评对照** —— 这里评的正是**恢复后的 `--out`**,
    所以那次对照**不需要额外再评一遍**(零 GPU 成本)。
    """
    if args.no_eval:
        rl.note("--no-eval:训练结束未评 mAP")
        print("[runlog] --no-eval ⇒ 不自动评 mAP")
        return None
    if not (args.eval_seg or args.eval_keep_in_seg or args.eval_exclude_seg):
        msg = "未给 --eval-seg/--eval-keep-in-seg/--eval-exclude-seg ⇒ 不自动评 mAP"
        rl.note(msg)
        print(f"[runlog] {msg}(要评就补选择器;`--no-eval` 可显式关)")
        return None
    print(f"[runlog] 训练结束 → 留出集评估(score_thr={args.eval_score_thr})")
    res = evaluate(
        infos=args.infos,
        root=args.root,
        ckpt=args.out,  # 评的就是刚落盘的这份权重(恢复之后)
        frames=args.eval_frames,
        seg=args.eval_seg,
        exclude_seg=args.eval_exclude_seg,
        keep_in_seg=args.eval_keep_in_seg,
        score_thr=args.eval_score_thr,
        temporal_window=args.temporal_window,
        rl=rl,
        highlight_prefix="holdout_",  # 训练日志里的 mAP 必须一眼看出是留出集的
    )
    rl.note(f"留出评估 {res['n_frames']} 帧,score_thr={res['score_thr']},后端 {res['backend']}")
    return res


def holdout_class_tags(res: dict, prefix: str = "confirm_") -> dict[str, float]:
    """留出评估的**逐类**结果 → jsonl 的稀疏标签(给曲线用的时间序列)。

    ★ **为什么必须另做这一步**:`evaluate()` 已经把逐类 AP 经 `rl.highlight` 写出去,但
    `runlog.highlight` 是**末次覆盖**(`self._highlights[k] = v`)⇒ 每次复核都盖掉上一次,
    **最后只剩最后那一次的逐类 AP**,时间序列整个丢失,TensorBoard 里什么也看不到。
    整体 mAP 平台完全可能是"一类到顶、另一类还在涨"互相抵消 —— 那不分类就看不出来,
    而"分类别看"正是本项目 P1 主线的核心问题。

    逐类 `n_pred` / `n_gt` 一并带上:它们是 AP 归属判据的第三条(计数不等 = 预测真变了,
    计数相同而 AP 变 = 阈值边界抖动),此前连 highlight 都没写。

    键名带 `/` 是为了在 TensorBoard 里按前缀分组;`classes` 为空(如 `res={}`)⇒ 返回空字典,
    调用方 `**tags` 摊开是安全的。
    """
    classes = res.get("classes") or []
    if not classes:
        return {}
    tags: dict[str, float] = {}
    for cls_name, ap_ in zip(classes, res["aps"], strict=True):
        tags[f"{prefix}AP/{cls_name}"] = round(float(ap_), 4)
    for cls_name, n in zip(classes, res["n_pred"], strict=True):
        tags[f"{prefix}n_pred/{cls_name}"] = int(n)
    for cls_name, n in zip(classes, res["n_gt"], strict=True):
        tags[f"{prefix}n_gt/{cls_name}"] = int(n)
    return tags


def _build_early_stopper(args, ds, model, rl) -> EarlyStopper | None:
    """建早停器。**不适用时返回 None 并把原因打出来** —— 静默不启用等于没人知道它为什么没停。"""
    if not args.early_stop:
        print("[early-stop] 关闭(--no-early-stop)")
        return None
    if is_single_frame_anchor(len(ds)):
        print("[early-stop] 单帧过拟合锚点 ⇒ **不启用**(那一段是故意跑到 loss→0 的正确性自证)")
        return None
    if not (args.eval_seg or args.eval_keep_in_seg or args.eval_exclude_seg):
        print("[early-stop] 未给 --eval-* 留出选择器 ⇒ 没有 AP 可复核(两段式缺第二段)⇒ 不启用")
        rl.note("early_stop:未启用 —— 没有留出选择器可供复核")
        return None

    best_ckpt = Path(str(args.out) + ".best")

    def confirm(epoch: int) -> float | None:
        """复核:把**当前**权重落成临时 ckpt → 评留出 AP → 创新高才留档。

        `evaluate()` 要一个可加载的 ckpt 路径,故必须先落盘(这就是复核的固定成本 ~134 MB 写)。
        留档用"先临时、创高才改名"而不是每次直接覆盖 `.best`:
        否则最后一次复核若 AP 回落,`.best` 存的就是**较差**的那份,与该名字相反。
        """
        tmp = Path(str(args.out) + f".tmp{epoch}")
        save_map_checkpoint(model, str(tmp))
        try:
            res = evaluate(
                infos=args.infos,
                root=args.root,
                ckpt=str(tmp),
                frames=args.eval_frames,
                seg=args.eval_seg,
                exclude_seg=args.eval_exclude_seg,
                keep_in_seg=args.eval_keep_in_seg,
                score_thr=args.eval_score_thr,
                temporal_window=args.temporal_window,
                batch=1,  # **复核必须逐帧**:自适应读空闲显存 ⇒ 数字会随机器状态变
                rl=rl,
                highlight_prefix="earlystop_",
            )
        except Exception as e:  # 复核失败 ⇒ 判据缺失 ⇒ 不敢停(见 early_stop 模块头注)
            print(f"[early-stop] 复核失败({type(e).__name__}: {e}) ⇒ 不敢停,继续跑")
            tmp.unlink(missing_ok=True)
            return None
        # 逐类结果落**时间序列**(整体 mAP 由调用方那行写)。
        # 就地发而非挂在 `StopDecision` 上:`confirm` 就是评估发生的地方,`res` 只在
        # 这里存在;绕一圈回调用方要多穿一层(detail 字段 / 附加属性),而多穿一层的
        # 每一处都是"哪天有人改了那一层,这里就静默不发"的机会。
        rl.metric(epoch, **holdout_class_tags(res, "confirm_"))
        ap = float(res["mAP"])
        prev = getattr(confirm, "best", None)
        if prev is None or ap - prev >= args.ap_min_improve:
            confirm.best = ap  # type: ignore[attr-defined]
            tmp.replace(best_ckpt)
            print(f"[early-stop] epoch {epoch} 复核 mAP={ap:.4f} 创新高 ⇒ 留档 {best_ckpt.name}")
        else:
            tmp.unlink(missing_ok=True)
            print(f"[early-stop] epoch {epoch} 复核 mAP={ap:.4f}(未超 {prev:.4f}+{args.ap_min_improve})")
        return ap

    stopper = EarlyStopper(
        PlateauDetector(patience=args.patience, min_improve=args.min_improve, warmup=args.warmup),
        confirm,
        confirm_every=args.confirm_every,
        ap_min_improve=args.ap_min_improve,
    )
    print(
        f"[early-stop] 启用:patience {args.patience} / loss 相对门槛 {args.min_improve} / "
        f"AP 门槛 {args.ap_min_improve} / 复核间隔 {args.confirm_every}"
    )
    return stopper


def _plot_early_stop(
    hist: list[float],
    lrs: list[float],
    confirms: list[tuple[int, float]],
    stop_epoch: int | None,
    reason: str,
    out_png: Path,
) -> Path | None:
    """早停过程图(**不是装饰**):loss 曲线 + lr 变化点 + 复核点 + 触发点。

    为什么必须有这张图:2026-09-29 我凭 `--epochs` 用尽就判 K=3 "训好了",实际 loss 斜率
    仍是 −0.0255(还在降)—— **"停在平台"和"停在半山腰"从数字列表上读不出来**,图上是一眼的事。
    用英文标签:matplotlib 不认项目那把 CJK 字体,中文会静默画成方框(与 PIL 那边同款坑)。
    """
    if not hist:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = list(range(1, len(hist) + 1))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(ep, hist, "-", color="tab:blue", lw=1.6, label="train loss (epoch mean)")
    # lr 变化点:lr 衰减造成的"平台"不是收敛(见 early_stop 模块头注的分型)
    for i in range(1, len(lrs)):
        if lrs[i] != lrs[i - 1]:
            ax.axvline(ep[i], color="0.7", ls=":", lw=1.0)
            ax.annotate(f"lr→{lrs[i]:.1e}", (ep[i], max(hist)), fontsize=8, color="0.4", rotation=90)
    if confirms:
        cx = [c[0] for c in confirms]
        cy = [c[1] for c in confirms]
        ax2 = ax.twinx()
        ax2.plot(cx, cy, "o-", color="tab:green", lw=1.4, ms=5, label="holdout mAP (confirm)")
        ax2.set_ylabel("holdout mAP", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
    if stop_epoch is not None:
        ax.axvline(stop_epoch, color="tab:red", lw=2.0)
        ax.annotate(
            f"early stop @{stop_epoch}\n{reason}",
            (stop_epoch, max(hist)),
            color="tab:red",
            fontsize=10,
            ha="right",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"early-stop trace (final: {'stopped ' + reason if stop_epoch else 'ran to --epochs'})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def is_single_frame_anchor(n_samples: int) -> bool:
    """过拟合闸门是否适用 —— 判据是**实际训练样本数**,不是 `--frames` 标志。

    `--frames 0` 的语义是「**不截断**」(取全部帧),而闸门原写 `args.frames > 1`
    ⇒ `0 > 1` 为假 ⇒ 几百帧的多帧训练被判成单帧过拟合锚点,打印
    `FAIL:损失卡在 N——匹配/loss/坐标口径存在错误,禁止继续训练` 并以退出码 1 收场
    (2026-09-24 实测:400 帧 128 ep 跑到 2.31 时踩到)。权重在闸门**之前**已存盘,
    故模型没坏 —— 但**假警报会让人以为口径真坏了而白查一轮**,这正是要修的点。
    """
    return n_samples == 1


def _save_opt_sidecar(opt: torch.optim.Optimizer, args: argparse.Namespace, epoch: int) -> None:
    """与 --out 同步写 <out>.opt(仅优化器状态;模型权重留在 --out 供 eval 直读)。"""
    if args.no_opt:
        return
    torch.save({"optimizer": opt.state_dict(), "epoch": epoch}, str(args.out) + ".opt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True, help="B2 组装 infos json")
    ap.add_argument("--root", required=True, help="图像根目录(相对 data_path 的基目录)")
    ap.add_argument("--frames", type=int, default=1, help="取前 N 帧;1 = 单帧过拟合锚点;0 = 不截断")
    ap.add_argument("--seg", default=None, help="只保留这些段(逗号分隔);留出集评测用")
    ap.add_argument("--exclude-seg", default="", help="排除这些段(逗号分隔);路线级留出用")
    ap.add_argument("--keep-in-seg", default=None, help="只保留段内帧号区间 A:B(左闭右开);帧级留出用")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument(
        "--amp",
        action="store_true",
        help="bf16 混合精度(forward 与 loss 走 `torch.autocast`)。**默认关** —— "
        "已归档的数全是 fp32 口径,开了就与它们不可比(四臂内部仍可比)。"
        "⚠️ 用 bf16 不用 fp16:3090 原生支持且**不需要 GradScaler**;"
        "⚠️ 自适应 batch(`--batch 0`)的探针会跟着走 AMP,不会量错显存",
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--lr-schedule",
        choices=SCHEDULES,
        default="halve",
        help="学习率调度。**默认 halve**(每 --lr-halve epochs 减半,已归档数就是它,勿改默认);"
        "cosine = warmup 后余弦退火到 lr·--min-lr-ratio,即 MapQR 官方那个形状"
        "(`runtime/lr_schedule.py` 头注记了「官方 500 iter ↔ 我们 ≈4.7 epoch」的换算)",
    )
    ap.add_argument(
        "--lr-halve",
        type=int,
        default=12,
        help="`--lr-schedule halve` 时 lr 每 N epochs 减半(0=不衰减;长训必须关,否则 lr 提前归零)",
    )
    ap.add_argument("--warmup", type=int, default=0, help="前 N epochs lr 线性升温(重启续训防尖峰;0=关)")
    ap.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="warmup 的**起点比例**(mmcv 语义,官方 1/3);0 = 从 0 升",
    )
    ap.add_argument(
        "--min-lr-ratio",
        type=float,
        default=1e-3,
        help="`--lr-schedule cosine` 的退火终值占峰值的比例(官方 1e-3)",
    )
    ap.add_argument(
        "--early-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="两段式早停(默认开):loss 平台作候选 → 留出 AP 复核 → 都没升才停。"
        "**单帧锚点 / 没给留出选择器时自动不启用**(见运行时的 [early-stop] 行)",
    )
    ap.add_argument("--patience", type=int, default=20, help="loss 连续多少 epoch 无明显改善算平台")
    ap.add_argument("--min-improve", type=float, default=1e-3, help="loss 的**相对**改善门槛(低于它不算改善)")
    ap.add_argument(
        "--ap-min-improve", type=float, default=1e-2, help="AP 复核的改善门槛(须远宽于 2e-3 噪声下限)"
    )
    ap.add_argument("--confirm-every", type=int, default=20, help="平台区内两次 AP 复核的最小间隔(epoch)")
    ap.add_argument("--no-opt", action="store_true", help="不读写优化器状态侧车 <out>.opt")
    ap.add_argument("--batch", type=int, default=0, help="0 = 自适应实测(默认);>0 = 显式指定")
    ap.add_argument("--max-batch", type=int, default=16, help="自适应实测的批大小上限")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader 进程数(多帧训练数据加载是瓶颈)")
    ap.add_argument("--num-vec", type=int, default=50, help="每类实例 query 数(官方 50)")
    ap.add_argument("--no-pretrain", action="store_true", help="backbone 不用 ImageNet 预训练")
    ap.add_argument("--init-ckpt", default=None, help="从既有 state_dict 续训(仅模型权重,优化器重置)")
    ap.add_argument(
        "--temporal-window",
        type=int,
        default=1,
        help="时序窗口 K:1 = 单帧基线(默认);K>1 = MapTRv2 时序版,每样本取本帧 + 前 K−1 帧",
    )
    ap.add_argument(
        "--save-every", type=int, default=0, help="每 N epochs 覆盖存盘 --out(0=仅结束存;长训防中断)"
    )
    ap.add_argument(
        "--variant",
        choices=variant_names(),
        default=None,
        help="一键选架构:mapqr = 散聚 query + 高度核 BEV 编码器(等价于同时打下面两个开关)。"
        "与细粒度开关互斥 —— 要做只开一半的消融就别给 --variant",
    )
    ap.add_argument(
        "--scatter-gather",
        action="store_true",
        help="head 换 MapQR 的 scatter-and-gather 解码器(拼接聚合 + 可学习采样);需从头训练",
    )
    ap.add_argument(
        "--bev-encoder",
        choices=("none", "height_kernel"),
        default="none",
        help="GKT 之后的 BEV 细化;height_kernel = MapQR 的 HeightKernelAttention。需从头训练",
    )
    ap.add_argument("--bev-encoder-layers", type=int, default=3, help="BEV 细化层数(官方 3)")
    ap.add_argument(
        "--bev-encoder-heads",
        type=int,
        default=8,
        help="BEV 细化的注意力头数;高度锚点数由它派生(官方隐式约束 D == heads)",
    )
    ap.add_argument(
        "--bev-chunk",
        type=int,
        default=0,
        help="BEV 交叉注意力按 query 维分块。**实测分块不降反升**(见 Plan2 §P-M.14),默认 0;"
        "要压显存请调 --bev-encoder-layers(实测每层 ≈ +3.6 GiB @bs2)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out", required=True, help="checkpoint 输出路径")
    ap.add_argument(
        "--eval-seg", default=None, help="训练结束后只评这些段(逗号分隔)的 chamfer AP;路线级留出用"
    )
    ap.add_argument("--eval-exclude-seg", default="", help="训练结束后评测排除这些段(逗号分隔)")
    ap.add_argument("--eval-keep-in-seg", default=None, help="训练结束后只评段内帧号区间 A:B;帧级留出用")
    ap.add_argument("--eval-frames", type=int, default=None, help="训练结束后评测帧数(默认全部)")
    ap.add_argument("--eval-score-thr", type=float, default=0.2, help="评测阈值;与 AP 数字一起记录")
    ap.add_argument("--no-eval", action="store_true", help="训练结束后不自动评 mAP")
    ap.add_argument("--no-runlog", action="store_true", help="不落 logs/ 三件套(默认每次运行都落)")
    args = ap.parse_args()
    args.out = str(project_path(args.out))  # 产物锚定项目根(相对路径不随 cwd 漂移)

    with runlog.run("autodrivedata.map.train_maptr") as rl:
        rl.input(args.infos, "infos")
        rl.input(args.root, "root")
        if args.init_ckpt:
            rl.input(args.init_ckpt, "init-ckpt")
        train(args, rl)


def train(args: argparse.Namespace, rl: runlog.RunLogger) -> None:
    """训练主体。抽出来只为让 `main()` 能用 `with runlog.run(...)` 包住全程
    (日志要在 argparse 之前开,才能把参数报错也记下来)。"""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    infos = json.loads(Path(args.infos).read_text(encoding="utf-8"))
    sel = select_frames(
        infos, parse_segs(args.seg), parse_segs(args.exclude_seg) or (), parse_frame_range(args.keep_in_seg)
    )
    if args.frames and args.frames > len(sel):
        raise SystemExit(f"--frames {args.frames} 超过筛选后的帧数 {len(sel)}")
    if args.frames:
        sel = sel[: args.frames]
    if not sel:
        raise SystemExit("筛选后没有任何帧 —— 检查 --seg / --exclude-seg / --keep-in-seg / --frames")
    ds = MapTRDataset(infos, args.root, frames=sel, window=args.temporal_window)
    if ds.dropped:
        # 丢帧必须**显式报出**:静默截短会让"覆盖了全部帧"变成假话(见 dataset.history_windows)
        toks = [infos[i]["token"] for i in ds.dropped]
        print(
            f"[data] 窗口 {args.temporal_window} 丢弃 {len(ds.dropped)} 帧"
            f"(前驱不在本切分内):{toks[:4]}{' …' if len(toks) > 4 else ''}"
        )
    print(f"[data] {len(ds)} 帧({len(ds.cam_names)} 相机),每帧 GT 实例:")
    first = ds[0]["gt"]
    print(
        "  "
        + " ".join(f"{cls}={len(first[i])}" for i, cls in enumerate(("divider", "ped", "boundary", "center")))
    )

    # 聚合开关(--variant)与细粒度开关在此合流;同时给会报错(见 variants.resolve_variant)
    flags = resolve_variant(args.variant, scatter_gather=args.scatter_gather, bev_encoder=args.bev_encoder)
    model = MapTR(
        num_vec=args.num_vec,
        pretrained=not args.no_pretrain,
        temporal_window=args.temporal_window,
        bev_encoder_layers=args.bev_encoder_layers,
        bev_encoder_heads=args.bev_encoder_heads,
        bev_chunk=args.bev_chunk,
        **flags,
    ).to(dev)
    if args.init_ckpt:
        missing = load_map_weights(model, args.init_ckpt, dev)
        # **别在这里断言"优化器重置"**:力矩载不载由 `_load_opt_sidecar` 按 `<init-ckpt>.opt`
        # 是否存在决定,那一步在后面。原打印两行自相矛盾(先"重置"后"力矩不重置",
        # 2026-09-29 实测),会让人误判续训的起步动力学。
        print(f"[model] 从 {args.init_ckpt} 续训(优化器状态见下方 [opt] 行)")
        if missing:
            print(f"[model] 融合层 {len(missing)} 个权重缺失 ⇒ 从零初始化(单帧权重热启动)")
    n_params = sum(p.numel() for p in model.parameters())
    variant = []
    if flags["scatter_gather"]:
        variant.append("scatter_gather")
    if flags["bev_encoder"] != "none":
        variant.append(f"bev_encoder={flags['bev_encoder']}×{args.bev_encoder_layers}")
    print(
        f"[model] MapTR(num_vec={args.num_vec}) @ {dev} | {n_params / 1e6:.1f}M 参数"
        f" | 变体: {', '.join(variant) if variant else '基线(默认)'}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # 优化器状态侧车:续训时 Adam 力矩不重置(重启尖峰的根因是力矩清零后
    # 每步退化为 ~lr·sign(g),大步长下把模型踢出盆地)。侧车与 --out 同步写。
    _load_opt_sidecar(opt, args, dev)

    # batch 解析:显式 > 自适应实测(与 AutoLabel device.py 同口径)
    model.train()
    if args.batch > 0:
        batch_size = args.batch
        print(f"[gpu] batch = {batch_size}(显式指定)")
    else:

        def probe_step(bs: int) -> None:
            # ⚠️ `amp` 必须与真跑一致 —— 否则探针量的是另一种精度下的显存,
            #    选出的 batch 在真跑时当场 OOM(或白保守)
            _train_batch(model, opt, dev, ds, collate([ds[i % len(ds)] for i in range(bs)]), amp=args.amp)

        free_gb = get_gpu_free_memory_gb()
        batch_size = tune_train_batch_size(probe_step, max_batch=args.max_batch)
        free_str = "?" if free_gb is None else f"{free_gb:.1f} GiB"
        print(f"[gpu] batch = {batch_size}(自适应实测:空闲 {free_str} × {SAFETY_FACTOR:.2f} / 每样本增量)")
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=args.workers
    )
    # batch 进 highlights:**换机器 ⇒ 显存变 ⇒ 自适应实测选到不同 batch ⇒ 新旧 AP 不可比**。
    # 旧日志里没这条,才让"换到 24 G 机器后 AP 变了"变成不可归因的问题。
    rl.highlight("batch", batch_size)
    rl.highlight("n_train_frames", len(ds))
    rl.highlight("epochs", args.epochs)
    rl.highlight("seed", args.seed)
    rl.highlight("temporal_window", args.temporal_window)
    # 学习率配方必须逐项记全:峰值/调度/warmup 三件里漏一件,日志里的 AP 就归不到配方上
    # (本条的由来见 `runtime/lr_schedule.py` 头注:恒定 1e-4 下"架构更差"与"没训完"分不开)
    rl.highlight("amp", "bf16" if args.amp else "off")
    rl.highlight("lr", args.lr)
    rl.highlight("lr_schedule", args.lr_schedule)
    rl.highlight("lr_halve", args.lr_halve)
    rl.highlight("warmup", args.warmup)
    rl.highlight("warmup_ratio", args.warmup_ratio)
    rl.highlight("min_lr_ratio", args.min_lr_ratio)
    # 变体开关必须与结论数字同处记录 —— 否则日志里的 AP 归不到具体结构上
    rl.highlight("variant", args.variant or "custom(细粒度开关)")
    rl.highlight("scatter_gather", flags["scatter_gather"])
    rl.highlight("bev_encoder", flags["bev_encoder"])
    if flags["bev_encoder"] != "none":
        rl.highlight("bev_encoder_layers", args.bev_encoder_layers)
        rl.highlight("bev_encoder_heads", args.bev_encoder_heads)
        rl.highlight("bev_chunk", args.bev_chunk)
    stopper = _build_early_stopper(args, ds, model, rl)
    hist: list[float] = []
    lrs: list[float] = []
    confirms: list[tuple[int, float]] = []
    stop_epoch: int | None = None
    stop_reason = ""
    for epoch in range(1, args.epochs + 1):
        # 调度在 `runtime/lr_schedule.py`(**纯值、可单测**);这里只取当帧的值。
        # `halve` 是默认且与改造前逐位相同 —— 已归档的四档数是那个口径。
        lr = lr_at(
            epoch,
            base=args.lr,
            epochs=args.epochs,
            schedule=args.lr_schedule,
            lr_halve=args.lr_halve,
            warmup=args.warmup,
            warmup_ratio=args.warmup_ratio,
            min_lr_ratio=args.min_lr_ratio,
        )
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        ep = {"cls": 0.0, "pts": 0.0}
        for batch in loader:
            c, p = _train_batch(model, opt, dev, ds, batch, amp=args.amp)
            ep["cls"] += c
            ep["pts"] += p
        steps = max(1, len(loader))
        hist.append((ep["cls"] + ep["pts"]) / steps)
        # 逐 epoch 落 .jsonl(**比 `--log-every` 更密**):`.log` 是给人看的、可以稀疏,
        # `.jsonl` 是给画曲线的 —— 稀疏打印会让"哪一段掉下去"看不出拐点。
        rl.metric(
            epoch,
            epoch=epoch,
            loss=round(hist[-1], 6),
            cls=round(ep["cls"] / steps, 6),
            pts=round(ep["pts"] / steps, 6),
            lr=lr,
        )
        lrs.append(lr)
        if stopper is not None:
            d = stopper.update(epoch, hist[-1], lr)
            if d.confirm_ap is not None:
                confirms.append((epoch, d.confirm_ap))
                rl.metric(epoch, confirm_mAP=round(d.confirm_ap, 6), best_ap=round(d.best_ap or 0.0, 6))
            if d.should_stop:
                # **先存盘再退**:`--save-every N` 是周期性的,不补这一刀最后 N 个 epoch 白跑
                save_map_checkpoint(model, args.out)
                _save_opt_sidecar(opt, args, epoch)
                stop_epoch, stop_reason = epoch, d.reason
                print(
                    f"[early-stop] **触发** epoch {epoch}:{d.reason}"
                    f"(best AP {d.best_ap:.4f} @ epoch {d.best_ap_epoch},共 {stopper.n_confirms} 次复核)"
                )
                break
        if args.save_every and epoch % args.save_every == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_map_checkpoint(model, args.out)
            _save_opt_sidecar(opt, args, epoch)
            print(f"[ckpt] epoch {epoch}: 覆盖存盘 {args.out}")
        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[epoch {epoch:4d}/{args.epochs}] total={hist[-1]:.4f} "
                f"cls={ep['cls'] / steps:.4f} pts={ep['pts'] / steps:.4f}"
            )

    last_epoch = len(hist)
    best_ckpt = Path(str(args.out) + ".best")
    if stopper is not None:
        rl.highlight("early_stop_enabled", True)
        rl.highlight("early_stop_epoch", stop_epoch if stop_epoch is not None else last_epoch)
        rl.highlight("early_stop_reason", stop_reason or "ran_to_epochs")
        rl.highlight("early_stop_confirms", stopper.n_confirms)
        if stopper.best_ap is not None:
            rl.highlight("early_stop_best_ap", round(stopper.best_ap, 6))
            rl.highlight("early_stop_best_ap_epoch", stopper.best_ap_epoch)
        png = _plot_early_stop(
            hist,
            lrs,
            confirms,
            stop_epoch,
            stop_reason,
            Path(str(args.out) + ".early_stop.png").with_suffix(".png"),
        )
        if png is not None:
            rl.artifact(png, "early-stop-trace")
            print(f"[early-stop] 过程图 → {png}")

    final = float(np.mean(hist[-20:]))
    print(f"[done] 最后 20 步平均 total = {final:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_map_checkpoint(model, args.out)
    _save_opt_sidecar(opt, args, last_epoch)
    print(f"[save] {args.out}")
    rl.highlight("final_loss", round(final, 4))

    # ★★ best-AP 恢复 —— **必须是最后一次写 `--out`**,故放在上面那次无条件存盘**之后**。
    #
    # **这里踩过一次真 bug(2026-09-30)**:恢复块原先在存盘**之前**,于是
    #   ① `copyfile(best, out)` 把 ep296 的权重放上去,
    #   ② 紧接着 `save_map_checkpoint(model, out)` 把内存里的 **ep329** 又盖回去 ——
    # 日志照写"已把 best-AP 权重恢复",盘上却是触发时刻那份(留出 mAP **0.1557 → 0.1243**)。
    # 更糟的是 `best_ckpt.unlink()` 已删掉唯一副本,而 `--save-every` 覆盖同一个 `--out`、
    # **不留历史** ⇒ 那份权重**不可恢复**。
    #
    # 判据不能靠"看图/看日志"(两处都写着 296),只能靠**源码顺序**,见
    # `tests/map/test_maptr_select.py::TestCheckpointFinalizeOrder`。
    opt_artifact = str(args.out) + ".opt"
    if stop_epoch is not None and best_ckpt.exists():
        import shutil

        if stopper is not None and stopper.best_ap_epoch not in (None, last_epoch):
            shutil.copyfile(best_ckpt, args.out)
            # **`.opt` 与权重不再同源**:侧车记的是另一 epoch 的 Adam 力矩。
            # 留着同名会让"同名即同源"的假设静默破裂 ⇒ 改名留档并明说。
            side = Path(opt_artifact)
            if side.exists():
                opt_artifact = f"{opt_artifact}.stale-ep{last_epoch}"
                side.rename(opt_artifact)
                print(f"[early-stop] `.opt` 侧车(epoch {last_epoch})与恢复的权重不同源 ⇒ 已改名留档")
            print(f"[early-stop] 已把 best-AP 权重(epoch {stopper.best_ap_epoch})恢复为 {args.out}")
        best_ckpt.unlink(missing_ok=True)

    # **产物表在恢复之后才登记**:`rl.artifact` 是**调用时立刻 sha256**(`_digest`),
    # 放在恢复前登记,记下的是**被覆盖掉的那份**的哈希 —— 与本节同一个坑的另一面。
    rl.artifact(args.out, "model")
    if not args.no_opt:
        rl.artifact(opt_artifact, "optimizer-state")

    # ★★ **回评对照**:`_auto_eval` 评的正是恢复后的 `--out` ⇒ 拿它的结果与日志声称的
    # best-AP 比对,**零额外 GPU 成本**就把 §P-M.20 那类"产物与自述不一致"变成自动检查。
    # (那次只能靠手工回评 + 逐位比 `pred/gt` 计数才发现,纯属凑巧。)
    holdout = _auto_eval(args, rl)
    measured = (holdout or {}).get("mAP")
    claimed = stopper.best_ap if (stop_epoch is not None and stopper is not None) else None
    # 三种"判不了"各自留痕 —— **"没验"必须与"验过且通过"可区分**(否则等于没验)
    if measured is None:
        rl.highlight("restore_verified", "unavailable(未评 mAP ⇒ 无法回评对照)")
    elif claimed is None:
        rl.highlight("restore_verified", "n/a(未发生 best-AP 恢复,无声称值可对)")
    else:
        bad = _restore_mismatch(claimed, measured)
        if bad is not None:
            # 权重**已经在盘上**,raise 不会丢东西 —— 它只是"这次交付不成立"的信号。
            # 与单帧锚点的过拟合 FAIL 同一处置(产物已存,退出码非零)。
            rl.note(f"回评对照 **FAIL**:{bad}")
            print(f"\n❌ [verify] {bad}")
            print("   ⇒ 恢复后的 `--out` 与日志声称的 best-AP 不是同一份权重,交付不成立。")
            raise SystemExit(1)
        rl.highlight("restore_verified", f"ok(回评 {measured:.4f} vs 声称 {claimed:.4f})")

    if not is_single_frame_anchor(len(ds)):
        return  # 多帧训练无过拟合判据(损失收敛量级看 D 阶段 AP)
    if final < 0.5:
        print("PASS:单帧过拟合达标(损失 → ~0),参考实现数学链自洽")
    else:
        print(f"FAIL:损失卡在 {final:.3f}——匹配/loss/坐标口径存在错误,禁止继续训练")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

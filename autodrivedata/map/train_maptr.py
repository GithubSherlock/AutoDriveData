"""MapTR 训练入口(§5.11 C 阶段):单帧过拟合 = 正确性锚点;多帧 = 常规训练。

单帧模式(--frames 1,默认)目标:总损失压到 ~0(分类全中 + 点回归收敛)。这是
"参考实现数学正确"的最强自证——任何坐标口径 / 匹配 / 损失错误都会把单帧
损失卡在高位,无法过拟合。通过判据:最后 20 步平均 total < 0.5,否则退出码 1。
多帧模式(--frames N > 1,或 --frames 0 = 不截断):常规训练,无过拟合判据,收敛量级看
D 阶段 chamfer AP。**闸门按实际训练样本数判**(见 `is_single_frame_anchor`),不按
`--frames` 标志 —— `--frames 0` 是"不截断",按标志判会把几百帧的训练误判成单帧锚点。

匹配(匈牙利)在 CPU 上做(每步 detach 后),不进入训练图——与官方训练流程一致。

batch 口径:--batch 0(默认)= 自适应实测(空闲显存 × 0.85 / 每样本训练步增量,
上限 --max-batch;参考 AutoLabel tools/device.py);--batch N>0 = 显式指定
(显式 > 实测)。自适应探针 = 完整训练步 ×2(batch 1 warmup + batch 2 增量),
含 optimizer.step,见 autodrivedata/map/maptr/device.py。

用法:
  python -m autodrivedata.map.train_maptr --infos outputs/surround_drive/map_infos.json \
      --root outputs/surround_drive --frames 1 --epochs 400 --out outputs/maptr_overfit.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from autodrivedata.map.maptr.dataset import (
    MapTRDataset,
    collate,
    parse_frame_range,
    parse_segs,
    select_frames,
)
from autodrivedata.map.maptr.device import SAFETY_FACTOR, auto_tune_batch_size, get_gpu_free_memory_gb
from autodrivedata.map.maptr.head import maptr_loss, match_assign
from autodrivedata.map.maptr.model import MapTR, load_map_weights
from autodrivedata.paths import project_path


def _to_device(images: dict | list, dev: torch.device):
    """单帧 dict / 时序 list[dict] 两种结构都搬到 `dev`(保持结构,模型按类型分派)。"""
    if isinstance(images, list):
        return [{n: t.to(dev) for n, t in frame.items()} for frame in images]
    return {n: t.to(dev) for n, t in images.items()}


def _train_batch(
    model: MapTR, opt: torch.optim.Optimizer, dev: torch.device, ds: MapTRDataset, batch: dict
) -> tuple[float, float]:
    """跑一个完整训练步(forward+匈牙利匹配+loss+backward+step),返回 (cls, pts) 损失。

    同时用作自适应 batch 的探针步(measure_batch_memory 的 step_fn):探针必须与
    真实训练步完全同构,否则每样本显存增量测不准。
    """
    images = _to_device(batch["images"], dev)
    poses = batch["poses"].to(dev)
    out, _ = model(images, poses, ds.calibs)
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
    loss = maptr_loss(
        out["pred_logits"],
        out["pred_points"],
        torch.tensor(np.stack(cls_ts), device=dev),
        torch.tensor(np.stack(pts_ts), device=dev),
        torch.tensor(np.stack(masks), device=dev),
    )
    opt.zero_grad()
    loss["total"].backward()
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
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--lr-halve", type=int, default=12, help="lr 每 N epochs 减半(0=不衰减;长训必须关,否则 lr 提前归零)"
    )
    ap.add_argument("--warmup", type=int, default=0, help="前 N epochs lr 线性升温(重启续训防尖峰;0=关)")
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out", required=True, help="checkpoint 输出路径")
    args = ap.parse_args()
    args.out = str(project_path(args.out))  # 产物锚定项目根(相对路径不随 cwd 漂移)

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

    model = MapTR(
        num_vec=args.num_vec, pretrained=not args.no_pretrain, temporal_window=args.temporal_window
    ).to(dev)
    if args.init_ckpt:
        missing = load_map_weights(model, args.init_ckpt, dev)
        print(f"[model] 从 {args.init_ckpt} 续训(优化器重置)")
        if missing:
            print(f"[model] 融合层 {len(missing)} 个权重缺失 ⇒ 从零初始化(单帧权重热启动)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] MapTR(num_vec={args.num_vec}) @ {dev} | {n_params / 1e6:.1f}M 参数")

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
            _train_batch(model, opt, dev, ds, collate([ds[i % len(ds)] for i in range(bs)]))

        free_gb = get_gpu_free_memory_gb()
        batch_size = auto_tune_batch_size(probe_step, max_batch=args.max_batch)
        free_str = "?" if free_gb is None else f"{free_gb:.1f} GiB"
        print(f"[gpu] batch = {batch_size}(自适应实测:空闲 {free_str} × {SAFETY_FACTOR:.2f} / 每样本增量)")
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=args.workers
    )
    hist: list[float] = []
    for epoch in range(1, args.epochs + 1):
        # 阶梯衰减:每 --lr-halve epochs 减半(0 = 不衰减;长训必须关,
        # 否则 400-epoch 跑的后半程 lr 已衰减到 ~1e-8,平台是 lr 归零不是收敛)
        if args.lr_halve > 0:
            lr = args.lr * (0.5 ** ((epoch - 1) // args.lr_halve))
        else:
            lr = args.lr
        if args.warmup > 0 and epoch <= args.warmup:
            lr *= epoch / args.warmup  # 线性升温:防"力矩重置 + 大步长"的起步尖峰
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        ep = {"cls": 0.0, "pts": 0.0}
        for batch in loader:
            c, p = _train_batch(model, opt, dev, ds, batch)
            ep["cls"] += c
            ep["pts"] += p
        steps = max(1, len(loader))
        hist.append((ep["cls"] + ep["pts"]) / steps)
        if args.save_every and epoch % args.save_every == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
            _save_opt_sidecar(opt, args, epoch)
            print(f"[ckpt] epoch {epoch}: 覆盖存盘 {args.out}")
        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[epoch {epoch:4d}/{args.epochs}] total={hist[-1]:.4f} "
                f"cls={ep['cls'] / steps:.4f} pts={ep['pts'] / steps:.4f}"
            )

    final = float(np.mean(hist[-20:]))
    print(f"[done] 最后 20 步平均 total = {final:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    _save_opt_sidecar(opt, args, args.epochs)
    print(f"[save] {args.out}")
    if not is_single_frame_anchor(len(ds)):
        return  # 多帧训练无过拟合判据(损失收敛量级看 D 阶段 AP)
    if final < 0.5:
        print("PASS:单帧过拟合达标(损失 → ~0),参考实现数学链自洽")
    else:
        print(f"FAIL:损失卡在 {final:.3f}——匹配/loss/坐标口径存在错误,禁止继续训练")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

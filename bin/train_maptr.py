"""MapTR 训练入口(§5.11 C 阶段):单帧过拟合 = 正确性锚点;多帧 = 常规训练。

单帧模式(--frames 1,默认)目标:总损失压到 ~0(分类全中 + 点回归收敛)。这是
"参考实现数学正确"的最强自证——任何坐标口径 / 匹配 / 损失错误都会把单帧
损失卡在高位,无法过拟合。通过判据:最后 20 步平均 total < 0.5,否则退出码 1。
多帧模式(--frames N > 1):常规训练,无过拟合判据,收敛量级看 D 阶段 chamfer AP。

匹配(匈牙利)在 CPU 上做(每步 detach 后),不进入训练图——与官方训练流程一致。

用法:
  python bin/train_maptr.py --infos outputs/surround_drive/map_infos.json \
      --root outputs/surround_drive --frames 1 --epochs 400 --out outputs/maptr_overfit.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from maptr_impl.dataset import MapTRDataset, collate
from maptr_impl.head import maptr_loss, match_assign
from maptr_impl.model import MapTR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True, help="B2 组装 infos json")
    ap.add_argument("--root", required=True, help="图像根目录(相对 data_path 的基目录)")
    ap.add_argument("--frames", type=int, default=1, help="取前 N 帧;1 = 单帧过拟合锚点")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4, help="DataLoader 进程数(多帧训练数据加载是瓶颈)")
    ap.add_argument("--num-vec", type=int, default=50, help="每类实例 query 数(官方 50)")
    ap.add_argument("--no-pretrain", action="store_true", help="backbone 不用 ImageNet 预训练")
    ap.add_argument("--init-ckpt", default=None, help="从既有 state_dict 续训(仅模型权重,优化器重置)")
    ap.add_argument("--save-every", type=int, default=0, help="每 N epochs 覆盖存盘 --out(0=仅结束存;长训防中断)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out", required=True, help="checkpoint 输出路径")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    infos = json.loads(Path(args.infos).read_text(encoding="utf-8"))
    if args.frames > len(infos):
        raise SystemExit(f"--frames {args.frames} 超过 infos 帧数 {len(infos)}")
    ds = MapTRDataset(infos, args.root, frames=list(range(args.frames)))
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.workers
    )
    print(f"[data] {args.frames} 帧({len(ds.cam_names)} 相机),每帧 GT 实例:")
    first = ds[0]["gt"]
    print(
        "  "
        + " ".join(f"{cls}={len(first[i])}" for i, cls in enumerate(("divider", "ped", "boundary", "center")))
    )

    model = MapTR(num_vec=args.num_vec, pretrained=not args.no_pretrain).to(dev)
    if args.init_ckpt:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=dev))
        print(f"[model] 从 {args.init_ckpt} 续训(优化器重置)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] MapTR(num_vec={args.num_vec}) @ {dev} | {n_params / 1e6:.1f}M 参数")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    hist: list[float] = []
    for epoch in range(1, args.epochs + 1):
        # 阶梯衰减:每 12 epochs 减半(长训后期稳定;单帧过拟合不受影响)
        lr = args.lr * (0.5 ** ((epoch - 1) // 12))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        ep = {"cls": 0.0, "pts": 0.0}
        for batch in loader:
            images = {n: t.to(dev) for n, t in batch["images"].items()}
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
            ep["cls"] += float(loss["cls"].detach())
            ep["pts"] += float(loss["pts"].detach())
        steps = max(1, len(loader))
        hist.append((ep["cls"] + ep["pts"]) / steps)
        if args.save_every and epoch % args.save_every == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
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
    print(f"[save] {args.out}")
    if args.frames > 1:
        return  # 多帧训练无过拟合判据(损失收敛量级看 D 阶段 AP)
    if final < 0.5:
        print("PASS:单帧过拟合达标(损失 → ~0),参考实现数学链自洽")
    else:
        print(f"FAIL:损失卡在 {final:.3f}——匹配/loss/坐标口径存在错误,禁止继续训练")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

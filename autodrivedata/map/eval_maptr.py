"""MapTR D 阶段评估:训练权重 → 逐帧推理 → 四类 chamfer AP(MapTR 官方口径)。

评估口径与官方 MapTR map 评测同构:全部评估帧的预测/GT 按类**跨帧汇聚**后做
一次一对一匹配(而非逐帧 AP 平均);AP = 阈值 {0.5, 1.0, 1.5}m 的 precision
均值(autodrivedata.chamfer_ap)。解码:每类 query 取 sigmoid 得分 > --score-thr
的实例(默认 0.2,官方 nuscenes 惯例;阈值可扫)。

**口径警告(实测)**:该 AP 是 precision 均值、**无 recall 项** → 保守操作点
(阈值高、预测少而准)天然占便宜。同一权重 score_thr 0.2→0.4 可达 0.0510→
0.1350(2.6×)。因此**跨权重比较必须固定 --score-thr**,或直接用 --sweep 报告
曲线;绝对数字离开阈值无意义。--sweep 复用同一次推理(阈值只是后处理)。

用法:
  python -m autodrivedata.map.eval_maptr --infos outputs/surround_train/map_infos.json \
      --root outputs/surround_train --ckpt outputs/maptr.pt --frames 300

接 AutoLabel(逐帧契约,见 autodrivedata/map/mapvec_schema.py):
  python -m autodrivedata.map.eval_maptr --infos outputs/surround_train/map_infos.json \
      --root outputs/surround_train --ckpt outputs/maptr_ep512.pt \
      --start 200 --out-frames outputs/surround_pred      # → outputs/surround_pred/{token}.json
      --start 200 --out-frames outputs/surround_pred --map-format lanelet2   # 另落 {token}.osm

**AP 口径的唯一落点**:`evaluate()` 是纯函数形态的实现,`main()` 只是它的 argparse 壳。
`train_maptr --eval-*` 训练尾部的自动评估**调用同一个 `evaluate()`** ——
chamfer AP 若抄成第二份,两处必然漂移,而"同一权重在两个脚本里报出不同 AP"是最难查的失效。
每次运行落 `logs/` 三件套(脚本/时间/argv/git/GPU + 输入产物 + mAP 等结论),`--no-runlog` 关。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from autodrivedata.map.chamfer_ap import chamfer_ap_per_class, chamfer_cost_matrix
from autodrivedata.map.export_mapvec import MAP_SUFFIX, dump_map_text
from autodrivedata.map.maptr.chamfer_gpu import chamfer_cost_matrix_cuda
from autodrivedata.map.maptr.dataset import (
    MAPTR_CLASSES,
    MapTRDataset,
    parse_frame_range,
    parse_segs,
    select_frames,
)
from autodrivedata.map.maptr.variants import load_map_model, read_map_meta
from autodrivedata.map.mapvec import MAP_FORMATS
from autodrivedata.map.mapvec_schema import (
    MapVecFramePred,
    MapVecInstance,
    dump_frame,
    gt_out_of_window,
    make_instance,
    out_of_window,
)
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path


def _dump_preds(
    path: str,
    infos: str,
    ckpt: str,
    thr: float,
    preds_by_class: list[list[np.ndarray]],
    scores_by_class: list[list[float]],
    gts_by_class: list[list[np.ndarray]],
) -> None:
    """预测产物落盘:json(供 AutoLabel 消费)+ BEV png(目检,红=pred 绿=GT)。

    跨帧汇聚口径(与评估一致,不含帧归属);BEV 窗口与模型输出同系:
    x∈[-15, 15] 前向、y∈[-30, 30] 左向(米)。
    """
    payload = {
        "infos": infos,
        "ckpt": ckpt,
        "score_thr": thr,
        "classes": list(MAPTR_CLASSES),
        "preds": [
            [{"score": float(s), "points": p.tolist()} for s, p in zip(sc, pc, strict=True)]
            for sc, pc in zip(scores_by_class, preds_by_class, strict=True)
        ],
        "gts": [[g.tolist() for g in gc] for gc in gts_by_class],
    }
    Path(path + ".json").write_text(json.dumps(payload), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, name, pc, gc in zip(axes.flat, MAPTR_CLASSES, preds_by_class, gts_by_class, strict=True):
        for g in gc:
            ax.plot(g[:, 0], g[:, 1], color="tab:green", lw=1.2, alpha=0.8)
        for p in pc:
            ax.plot(p[:, 0], p[:, 1], color="tab:red", lw=1.0, alpha=0.9)
        ax.set_title(f"{name}  pred {len(pc)} / gt {len(gc)}")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-30, 30)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.suptitle(f"MapTR pred(red) vs GT(green) | {Path(ckpt).name} score_thr={thr}")
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=110)
    plt.close(fig)


def _dump_frame_map(gts, fmt: str, base: Path, ckpt: str) -> None:
    """逐帧裁剪地图 → 该格式;元素 = **该帧的 GT 折线**(2D 包成 `MapVec(z=0)`)。

    与 `{token}.json` 同窗、同坐标系(都是 ego 系米),故下游不必自己再裁一次。
    """
    from autodrivedata.map.mapvec import MapVec

    vecs = tuple(MapVec(g.cls, tuple((x, y, 0.0) for x, y in g.points), (), "", ckpt) for g in gts)
    with open(f"{base}{MAP_SUFFIX[fmt]}", "w", encoding="utf-8") as f:
        f.write(dump_map_text(vecs, fmt))


def evaluate(
    *,
    infos: str,
    root: str,
    ckpt: str,
    frames: int | None = None,
    start: int = 0,
    seg: str | None = None,
    exclude_seg: str = "",
    keep_in_seg: str | None = None,
    score_thr: float = 0.2,
    sweep: str | None = None,
    match: str = "auto",
    temporal_window: int | None = None,  # None = 用 checkpoint 自带的,缺省 1
    device: str | None = None,
    out_pred: str | None = None,
    out_frames: str | None = None,
    map_format: str = "opendrive",  # 逐帧地图的落盘格式(见 mapvec.MAP_FORMATS)
    rl: runlog.RunLogger | None = None,
    highlight_prefix: str = "",
) -> dict:
    """推理 + chamfer AP —— **AP 口径的唯一落点**。

    抽成显式 kwarg 的函数(而不是留在 `main()` 里)是为了让 `train_maptr` 训练结束后
    的自动评估能**调用同一份实现**:chamfer AP 若抄成第二份,两处必然漂移,
    而"同一权重在不同脚本里报出不同 AP"是最难查的一类结论失效。

    `rl` 给定则把输入/产物/结论登记进**调用方的**日志(传 `None` = 全程 no-op)。
    被 `train_maptr` 调用时本函数的 `print()` 落在**训练的**日志里 —— 不另开一份,
    这样"训练 → 留出评估"是一条完整的时间线。

    返回 `{"mAP", "aps", "classes", "n_frames", "score_thr", "backend", "n_pred", "n_gt"}`。
    """
    if match == "cpu":
        cost_fn, backend = chamfer_cost_matrix, "cpu"
    elif match == "gpu":
        cost_fn, backend = chamfer_cost_matrix_cuda, "gpu"
    else:
        cost_fn, backend = (
            (chamfer_cost_matrix_cuda, "gpu") if torch.cuda.is_available() else (chamfer_cost_matrix, "cpu")
        )

    if rl is not None:
        # 输入的 ckpt 必须留痕:权重同名覆盖是常态,"这份 AP 是哪份权重出的"全靠它
        rl.input(ckpt, "ckpt")
        rl.input(infos, "infos")
        rl.input(root, "root")

    if out_pred:
        out_pred = str(project_path(out_pred))  # 产物锚定项目根(不随 cwd 漂移)
    if out_frames:
        out_frames = str(project_path(out_frames))

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    meta = json.loads(Path(infos).read_text(encoding="utf-8"))
    # 段/帧号选择器先过滤,--start / --frames 再在**过滤后的列表上**截 ——
    # 与 train_maptr 共用 select_frames,保证"训练排除了哪一段"与"评测只评哪一段"
    # 是同一份划分逻辑(两处各写一遍必然漂移,而这是 AP 结论有效性的前提)
    sel = select_frames(meta, parse_segs(seg), parse_segs(exclude_seg) or (), parse_frame_range(keep_in_seg))
    if not sel:
        raise SystemExit("筛选后没有任何帧 —— 检查 --seg / --exclude-seg / --keep-in-seg")
    n = min(frames or len(sel), len(sel))
    frame_list = sel[start : start + n]
    # 变体与窗口都从 checkpoint 自带的结构说明里读(旧裸权重 → 基线 / 窗口 1);
    # **必须先解析再建数据集** —— 窗口同时决定 dataset 的取帧与模型结构,两处不一致
    # 会让时序权重按单帧喂进去(键不匹配会报错,但报在模型侧,看着像权重坏了)
    meta_ck = read_map_meta(ckpt, dev)
    window = (
        temporal_window
        if temporal_window is not None
        else int(meta_ck["model_kwargs"].get("temporal_window", 1))
    )
    ds = MapTRDataset(meta, root, frames=frame_list, window=window)
    if ds.dropped:
        print(f"[data] 窗口 {window} 丢弃 {len(ds.dropped)} 帧(前驱不在本切分内)")
    print(f"[data] {len(ds)} 帧 × {len(ds.cam_names)} 相机")

    model, _meta_used = load_map_model(ckpt, dev, temporal_window=window)
    model.eval()
    print(f"[model] {ckpt} 载入完成(变体 {meta_ck['name']},窗口 {window})")

    sweep_list = [float(x) for x in sweep.split(",")] if sweep else []
    floor = min([score_thr, *sweep_list])  # 单次推理收全部 >floor 的实例,阈值纯后处理
    inst_by_class: list[list[tuple[float, np.ndarray]]] = [[] for _ in MAPTR_CLASSES]
    gts_by_class: list[list] = [[] for _ in MAPTR_CLASSES]
    t0 = time.perf_counter()
    n_frame_files = oow_pred = oow_gt = 0
    with torch.no_grad():
        for i, item in enumerate(ds):
            if isinstance(item["images"], list):  # 时序:旧 → 新的 K 帧
                images = [{n: t[None].to(dev) for n, t in f.items()} for f in item["images"]]
                pose = item["poses"][None].to(dev)
            else:
                images = {n: t[None].to(dev) for n, t in item["images"].items()}
                pose = item["pose"][None].to(dev)
            out, _ = model(images, pose, ds.calibs)
            logits = out["pred_logits"][0].float()  # (Nq, C+1)
            pts = out["pred_points"][0].float().cpu().numpy()  # (Nq, P, 2)
            scores = torch.sigmoid(logits).cpu().numpy()
            frame_preds: list[MapVecInstance] = []
            for c in range(len(MAPTR_CLASSES)):
                idx = slice(c * model.num_vec, (c + 1) * model.num_vec)
                sc_c = scores[idx, c + 1]
                keep = sc_c > floor
                inst_by_class[c].extend(zip(sc_c[keep].tolist(), pts[idx][keep], strict=True))
                gts_by_class[c].extend(item["gt"][c])
                if out_frames:  # 逐帧契约按 --score-thr 出(floor 只服务内部扫描)
                    at_thr = sc_c > score_thr  # 勿叫 sel:外层 sel 是帧下标列表
                    frame_preds.extend(
                        make_instance(MAPTR_CLASSES[c], p, s)
                        for s, p in zip(sc_c[at_thr].tolist(), pts[idx][at_thr], strict=True)
                    )
            if out_frames:
                info = ds.infos[i]  # 帧归属:跨帧汇聚产物丢的正是这个
                rec = MapVecFramePred(
                    frame=int(info["frame"]),
                    token=str(info["token"]),
                    score_thr=score_thr,
                    ckpt=ckpt,
                    preds=tuple(frame_preds),
                    gts=tuple(
                        make_instance(MAPTR_CLASSES[c], g)
                        for c in range(len(MAPTR_CLASSES))
                        for g in item["gt"][c]
                    ),
                    map_format=map_format,
                )
                dump_frame(rec, out_frames)
                if map_format != "opendrive":
                    _dump_frame_map(rec.gts, map_format, Path(out_frames) / rec.token, ckpt)
                n_frame_files += 1
                oow_pred += out_of_window(rec)
                oow_gt += gt_out_of_window(rec)
            if (i + 1) % 50 == 0:
                print(f"[infer] {i + 1}/{len(frame_list)} 帧 ({time.perf_counter() - t0:.1f}s)")

    def _at(thr: float) -> tuple[list[list], list[float], float]:
        """按阈值过滤实例 → (逐类折线, 逐类 AP, mAP)。代价矩阵与阈值无关,每档重算。"""
        preds = [[p for s, p in items if s > thr] for items in inst_by_class]
        aps_, m_ = chamfer_ap_per_class(preds, gts_by_class, cost_fn=cost_fn)
        return preds, aps_, m_

    t1 = time.perf_counter()
    if sweep_list:
        print(f"[eval] score 阈值扫描 后端={backend}")
        print("   " + "thr".rjust(5) + "   mAP".ljust(11) + "  ".join(n.rjust(11) for n in MAPTR_CLASSES))
        for thr in sweep_list:
            preds_s, aps_s, m_s = _at(thr)
            counts = "/".join(str(len(p)) for p in preds_s)
            print(
                f"  {thr:5.2f}   {m_s:<11.4f}" + "  ".join(f"{a:11.4f}" for a in aps_s) + f"   pred={counts}"
            )
    preds_by_class, aps, mAP = _at(score_thr)
    scores_by_class = [[s for s, _ in items if s > score_thr] for items in inst_by_class]
    print(f"[eval] score_thr={score_thr} 后端={backend} 匹配 {time.perf_counter() - t1:.1f}s")
    for cls_name, ap_, preds, gts in zip(MAPTR_CLASSES, aps, preds_by_class, gts_by_class, strict=True):
        print(f"  {cls_name:14s} AP={ap_:.4f}  (pred {len(preds)} / gt {len(gts)})")
    print(f"  {'mAP':14s} = {mAP:.4f}")

    if out_pred:
        _dump_preds(
            out_pred,
            infos,
            ckpt,
            score_thr,
            preds_by_class,
            scores_by_class,
            gts_by_class,
        )
        print(f"[out] 预测落盘 {out_pred}.json / {out_pred}.png")

    if out_frames:
        print(
            f"[out] 逐帧契约落盘 {n_frame_files} 个文件 → {out_frames}/{{token}}.json"
            f"(越窗点 pred {oow_pred} / gt {oow_gt})"
        )

    if rl is not None:
        # 结论数字登记到**调用方的**日志:单独跑 = 评测日志;被 train_maptr 调用 = 训练日志
        # `highlight_prefix` 让训练日志里这条叫 `holdout_mAP` —— 不加前缀的话,训练日志的
        # `mAP` 与训练损失并列,极易被读成"训练期算出来的 AP"(它其实是留出集的)。
        rl.highlight(f"{highlight_prefix}mAP", round(mAP, 4))
        for cls_name, ap_ in zip(MAPTR_CLASSES, aps, strict=True):
            rl.highlight(f"{highlight_prefix}AP/{cls_name}", round(ap_, 4))
        rl.highlight(f"{highlight_prefix}score_thr", score_thr)
        rl.highlight(f"{highlight_prefix}n_frames", len(frame_list))
        if out_pred:
            rl.artifact(f"{out_pred}.json", "pred")
            rl.artifact(f"{out_pred}.png", "pred-figure")
        if out_frames:
            rl.artifact_dir(out_frames, "pred-frames")

    return {
        "mAP": mAP,
        "aps": aps,
        "classes": list(MAPTR_CLASSES),
        "n_frames": len(frame_list),
        "score_thr": score_thr,
        "backend": backend,
        "n_pred": [len(p) for p in preds_by_class],
        "n_gt": [len(g) for g in gts_by_class],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True, help="B2 组装 infos json")
    ap.add_argument("--root", required=True, help="图像根目录")
    ap.add_argument("--ckpt", required=True, help="train_maptr.py 输出的 state_dict")
    ap.add_argument("--frames", type=int, default=None, help="评估帧数(默认全部)")
    ap.add_argument(
        "--start",
        type=int,
        default=0,
        help="起始帧(在选择器过滤后的列表上;留出集评估:训练 0..N-1,评估 --start N)",
    )
    ap.add_argument("--seg", default=None, help="只评这些段(逗号分隔);路线级留出用")
    ap.add_argument("--exclude-seg", default="", help="排除这些段(逗号分隔)")
    ap.add_argument("--keep-in-seg", default=None, help="只评段内帧号区间 A:B(左闭右开);帧级留出用")
    ap.add_argument("--score-thr", type=float, default=0.2, help="实例得分阈值(sigmoid)")
    ap.add_argument("--sweep", default=None, help="逗号分隔阈值列表,单次推理出扫描表(如 0.1,0.2,0.3,0.4)")
    ap.add_argument("--match", choices=("auto", "cpu", "gpu"), default="auto", help="代价矩阵后端(默认 auto)")
    ap.add_argument(
        "--temporal-window",
        type=int,
        default=None,
        help="时序窗口 K:不给 = 用 checkpoint 自带的(旧裸权重 → 1);K>1 时每样本取本帧 + 前 K−1 帧",
    )
    ap.add_argument("--device", default=None, help="推理设备(默认 cuda 若可用;GPU 被占用时可 --device cpu)")
    ap.add_argument("--out-pred", default=None, help="预测落盘基路径:写 <path>.json + <path>.png(BEV 目检)")
    ap.add_argument(
        "--out-frames",
        default=None,
        help="逐帧预测落盘目录:<DIR>/{token}.json(mapvec_pred/1 契约,GT 同文件携带;供 AutoLabel 消费)",
    )
    ap.add_argument(
        "--map-format",
        choices=MAP_FORMATS,
        default="opendrive",
        help="逐帧地图的落盘格式(需同时给 --out-frames);默认 opendrive = 只出契约 JSON",
    )
    ap.add_argument("--no-runlog", action="store_true", help="不落 logs/ 三件套(默认每次运行都落)")
    args = ap.parse_args()

    with runlog.run("autodrivedata.map.eval_maptr") as rl:
        evaluate(
            infos=args.infos,
            root=args.root,
            ckpt=args.ckpt,
            frames=args.frames,
            start=args.start,
            seg=args.seg,
            exclude_seg=args.exclude_seg,
            keep_in_seg=args.keep_in_seg,
            score_thr=args.score_thr,
            sweep=args.sweep,
            match=args.match,
            temporal_window=args.temporal_window,
            device=args.device,
            out_pred=args.out_pred,
            out_frames=args.out_frames,
            map_format=args.map_format,
            rl=rl,
        )


if __name__ == "__main__":
    main()

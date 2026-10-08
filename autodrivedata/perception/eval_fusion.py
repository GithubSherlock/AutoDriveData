"""后融合的判据:**LiDAR 单独 vs +相机**,与 `label_2` GT 比 3D AP **加上逐条 TP/FP/FN**。

用法:
  python -m autodrivedata.perception.eval_fusion --root outputs/kitti_ab_occl2_none_all --frames 0-19
  python -m autodrivedata.perception.eval_fusion --root <root> --frames 0-19 --no-camera   # 只出 LiDAR 档

## 为什么必须报 `TP/FP/FN`,不能只报 AP

两档的 `conf` **来源不同**(`lidar` 档来自簇点数,`lidar+cam` 档来自相机置信度)——
只比 AP 会把"排序变了"读成"检测变好了"。计数是**与排序无关**的那一半证据:
`+相机` 这一档要证明的是 **FP 降**(或召回降),那是一个**计数**上的断言。

## 它不做什么

- **不改任何数据**:纯读 `training/{image_2,velodyne,calib,label_2}`。
- **不做雷达**:见 [`fusion`](fusion.py) 头注(平台边界,有实测)。
- **不与 P1 的 A/B 数字比**:那些是 2D 检测的 AP,口径不同。这里报的是**3D** AP。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autodrivedata.perception.backends import BACKENDS, describe, make_predictor, resolve_conf
from autodrivedata.perception.cluster import cluster_boxes, dbscan
from autodrivedata.perception.compare import evaluate_frames, load_gt_labels, report_text
from autodrivedata.perception.fusion import CameraDet, Cluster, fuse, size_plausible, strict_gate
from autodrivedata.perception.ground import ransac_plane
from autodrivedata.perception.size_prior import load_prior
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: `ClassStats` 字段与汇总键的对应(名称不同,写死在这里避免再踩一次 `getattr` 猜名字)。
_STAT_FIELD = {"gt": "gt_count", "pred": "pred_count", "tp": "tp", "fp": "fp", "fn": "fn"}

#: COCO/ultralytics 类 id → **KITTI 类名**。这是融合的**价值所在**:
#: LiDAR 只出几何、**分不出类**;类由相机给。P1 数据上这一列全是 `Car`(所以测不出),
#: `--layout close` 那档带出 `Pedestrian`/`Cyclist`,相机这一列才第一次有活干。
COCO_TO_KITTI = {0: "Pedestrian", 1: "Cyclist", 2: "Car", 3: "Cyclist", 5: "Car", 7: "Car"}
DEFAULT_WEIGHTS = "weights/yolo26n.pt"
#: 点云裁剪:只留自车周围这个半径内的点再做聚类(远距的稀疏点会把簇撑得很大)。
CLIP_RADIUS = 80.0
#: 距离自适应 DBSCAN 系数。**默认 None ⇒ 走 sklearn 的 KD-tree 路径**。
#: ⚠️ 给非零值会切到 `cluster._dbscan_grid`(**纯 Python 网格实现**)—— 教程 13 为远距稀疏点
#: 引入的那条路。2026-10-01 实测:同一帧里 RANSAC 只要 **0.08 s**,而 `lidar_clusters` 整体
#: **129.7 s** ⇒ 20 帧 = 43 min,而 sklearn 路径是秒级。**要开它必须先体素降采样**。
DISTANCE_SCALE: float | None = None
#: 聚类前的体素边长(m)。**只在 `DISTANCE_SCALE` 非零时才降** —— 走到那条纯 Python 路径
#: 就必须先把点数压下来,否则等于挂死;`None` 时原样送 sklearn,不引入降采样误差。
VOXEL = 0.15


def load_calib(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """`calib/{fid}.txt` → `(P2 3×4, R0_rect @ Tr_velo_to_cam 4×4)`。

    **两件都在同一个文件里,别自造** —— `P2` 是 rectified cam0 的口径,
    `R0_rect` 又是 `Tr_velo_to_cam` 之后的一步;漏掉 `R0_rect` 会让投影整体偏,
    而偏多少随 `ry` 变 ⇒ **看着像"检测框有点歪",不像口径错**。
    """
    m: dict[str, np.ndarray] = {}
    for ln in path.read_text().splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            m[k.strip()] = np.array([float(x) for x in v.split()])
    p2 = m["P2"].reshape(3, 4)
    t = np.eye(4)
    t[:3, :4] = m["Tr_velo_to_cam"].reshape(3, 4)
    if "R0_rect" in m:
        r = np.eye(4)
        r[:3, :3] = m["R0_rect"].reshape(3, 3)
        t = r @ t
    return p2, t


def _voxel_down(pts: np.ndarray, size: float) -> np.ndarray:
    """体素降采样(每格取**质心**)。只为把点数压到纯 Python 聚类跑得动。"""
    key = np.floor(pts[:, :3] / size).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    n = int(inv.max()) + 1
    out = np.zeros((n, pts.shape[1]), dtype=np.float64)
    cnt = np.zeros(n, dtype=np.int64)
    np.add.at(out, inv, pts)
    np.add.at(cnt, inv, 1)
    return out / cnt[:, None]


def lidar_clusters(root: Path, fid: int) -> list[Cluster]:
    """`training/velodyne/{fid}.bin` → 车形候选簇(先地面分割,再 DBSCAN)。"""
    p = root / "training/velodyne" / f"{fid:06d}.bin"
    if not p.exists():
        return []
    pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
    if not len(pts):
        return []
    pts = pts[np.linalg.norm(pts[:, :3], axis=1) <= CLIP_RADIUS]
    fit = ransac_plane(pts)
    objects = pts[~fit[1]] if fit is not None else pts
    if len(objects) < 10:
        return []
    if DISTANCE_SCALE:
        objects = _voxel_down(objects, VOXEL)
        if len(objects) < 10:
            return []
    labels = dbscan(objects, eps=0.8, min_samples=5, distance_scale=DISTANCE_SCALE)
    return [
        Cluster(
            center=tuple(b["center"]),
            half=tuple(b["extent_half"]),
            n_points=b["n_points"],
            min_dist=b["min_dist"],
        )
        for b in cluster_boxes(objects, labels)
    ]


def camera_dets(img_path: Path, predict) -> list[CameraDet]:
    """检测 → 2D 检测(只留载具类,全部记作 `Car`)。

    后端由 `--backend` 决定(默认 sam3);`predict` 是 `backends.Predictor`,
    **类名已归一**,所以这里不再看 COCO 类号 —— 那条映射在 `backends.COCO_FALLBACK` 里。
    """
    out: list[CameraDet] = []
    for c, cf, b in predict(img_path):
        if c == "Car":
            out.append(CameraDet(label="Car", xyxy=b, conf=cf))
    return out


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="KITTI root(需 training/{image_2,velodyne,calib,label_2})")
    ap.add_argument("--frames", default="0-19")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--iou", type=float, default=0.5, help="3D IoU 阈值(**报 AP 必须带它**)")
    ap.add_argument(
        "--backend",
        choices=BACKENDS,
        default="sam3",
        help="相机分支的检测后端。**默认 sam3**;`yolo` 是回退。⚠️ 两边 AP 不可比",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=None,
        help="相机分支的置信度阈值。**不传则按后端取默认**(sam3 0.5 / yolo 0.25)—— "
        "两边的 score 尺度不同,互相套用会把幻觉当检出",
    )
    ap.add_argument("--no-camera", action="store_true", help="只跑 lidar 档(不起 YOLO)")
    ap.add_argument(
        "--gates",
        default="loose,strict",
        help="尺寸门的档:loose(默认,宽)/ strict(第一版,严)。**逐档全跑** —— "
        "「严门单独一档」是「相机到底有没有独立贡献」的对照,见 Plan4 §P-V18 三",
    )
    ap.add_argument(
        "--class-prior",
        default="",
        help="按类尺寸先验(JSON,`perception.size_prior.fit_from_root` 的产物)。"
        "**默认空 = 关** ⇒ 归档产物逐字节复现。"
        "⚠️ **先验必须来自别的 root** —— 在评测的同一个 root 上量真值尺寸再评 = 把答案抄进模型;"
        "来源与 `--root` 相同会**响亮报警**。",
    )
    ap.add_argument("--out", default="outputs/fusion", help="逐帧融合表落盘目录;'' = 不落")
    args = ap.parse_args()

    with runlog.run("autodrivedata.perception.eval_fusion") as rl:
        root = project_path(args.root)
        rl.input(str(root), "root")
        # ★★ **防 train/test 泄漏的那把锁**:先验若与评测同源,就是把答案抄进了模型。
        prior = None
        if args.class_prior:
            prior = load_prior(project_path(args.class_prior))
            src = str(prior.get("source_root", ""))
            rl.highlight("class_prior_source", src)
            if Path(src).resolve() == root.resolve():
                raise SystemExit(
                    f"先验来源 `{src}` **就是**评测 root ⇒ **train/test 泄漏**:"
                    "尺寸先验要从**别的 root**上量。换一个 `--prior-from` 的 root 再来"
                )
            print(f"[prior] {src} → {prior['dims']}")
            rl.highlight("class_prior_dims", prior["dims"])
        frames = _parse_frames(args.frames)
        predict = None
        if not args.no_camera:
            predict = make_predictor(
                args.backend, str(project_path(args.weights)), resolve_conf(args.backend, args.conf)
            )
            print(describe(args.backend, predict))

        modes = ["lidar"] if args.no_camera else ["lidar", "lidar+cam"]
        gates = {"loose": size_plausible, "strict": strict_gate}
        want = [g.strip() for g in args.gates.split(",") if g.strip()]
        for g in want:
            if g not in gates:
                raise SystemExit(f"未知门 {g!r}(合法:{sorted(gates)})")
        # 档名 = `门/模态` —— 2×2 一起报,消融才有对照
        per_mode: dict[str, dict[str, tuple[list, list]]] = {f"{g}/{m}": {} for g in want for m in modes}
        n_clusters = n_dets = 0
        for fid in frames:
            gt = load_gt_labels(root / "training/label_2" / f"{fid:06d}.txt")
            p2, v2c = load_calib(root / "training/calib" / f"{fid:06d}.txt")
            cl = lidar_clusters(root, fid)
            n_clusters += len(cl)
            dets = camera_dets(root / "training/image_2" / f"{fid:06d}.png", predict) if predict else []
            n_dets += len(dets)
            for g in want:
                for m in modes:
                    objs = fuse(cl, dets, v2c, p2, mode=m, gate=gates[g], prior=prior)
                    per_mode[f"{g}/{m}"][f"{fid:06d}"] = (gt, [o.box for o in objs])

        if n_clusters == 0:
            raise SystemExit(f"{root} 一帧都没聚出簇 —— 判据没有样本就不是判据,不许报数")
        print(
            f"[data] {len(frames)} 帧 | LiDAR 簇 {n_clusters}({n_clusters / len(frames):.1f}/帧) "
            f"| 相机检出 {n_dets}({n_dets / len(frames):.1f}/帧)"
        )

        # ★ 类从 **GT 实际出现的类**取,不写死 —— 早期数据 100% 是 `Car`,
        #   而 `--layout close` 那档会带出 `Pedestrian` / `Cyclist`。
        #   写死 `Car` 会让新类的目标**一条都不进报表**(而报表照样出得来)。
        gt_classes = sorted({b.label for per in per_mode.values() for gt, _ in per.values() for b in gt})
        # 预测侧**只有 `Car`**(LiDAR 分支按车形簇筛,YOLO 也只取载具类)⇒ 非 Car 的 GT
        # 必然是 FN。**这不是 bug,是"这条链当前只做车"** —— 报表会如实把它记成漏检,
        # 一眼看得出要不要扩。
        print(f"[classes] GT 出现 {gt_classes};预测侧目前只出 `Car`")
        summaries = {}
        for m in per_mode:
            rep = evaluate_frames(per_mode[m], classes=gt_classes, iou_thresh=args.iou)
            print()
            print(f"== 档 {m} (3D IoU {args.iou}) ==")
            print(report_text(rep, args.iou))
            tot = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
            for c in gt_classes:
                st = rep.per_class[c]
                for k in tot:
                    tot[k] += getattr(st, _STAT_FIELD[k])
                rl.highlight(f"{m}/{c}/ap", round(st.ap, 4))
            summaries[m] = {
                "ap_by_class": {c: round(rep.per_class[c].ap, 4) for c in gt_classes},
                "ap": round(sum(rep.per_class[c].ap for c in gt_classes) / len(gt_classes), 4),
                **tot,
            }
            for k, v in summaries[m].items():
                if not isinstance(v, dict):
                    rl.highlight(f"{m}/{k}", v)

        if len(modes) == 2:
            print()
            print(f"{'档(门/模态)':<16}{'AP':>8}{'TP':>6}{'FP':>6}{'FN':>6}{'Pred':>6}")
            for k, v in summaries.items():
                print(f"{k:<16}{v['ap']:>8.4f}{v['tp']:>6}{v['fp']:>6}{v['fn']:>6}{v['pred']:>6}")
            print()
            print(
                "⚠️ 两档 conf **来源不同**(点数 vs 相机置信度)⇒ 跨档 AP 差会把「排序变了」"
                "读成「检测变好了」。**结论看计数。**"
            )
            for g in want:
                a, b = summaries[f"{g}/lidar"], summaries[f"{g}/lidar+cam"]
                print(
                    f"  [{g}] +相机: AP {a['ap']:.4f}→{b['ap']:.4f} | "
                    f"TP {a['tp']}→{b['tp']}  **FP {a['fp']}→{b['fp']}**  FN {a['fn']}→{b['fn']}"
                )
                rl.highlight(f"ablation/{g}/fp_delta", b["fp"] - a["fp"])
                rl.highlight(f"ablation/{g}/tp_delta", b["tp"] - a["tp"])
            if len(want) == 2:
                s_, l_ = summaries["strict/lidar"], summaries["loose/lidar+cam"]
                print()
                print(
                    f"★ **关键对照** 严门单独({s_['tp']}TP/{s_['fp']}FP)vs 宽门+相机"
                    f"({l_['tp']}TP/{l_['fp']}FP) —— 若两者相当,相机的贡献可被更严的门替掉"
                )
                rl.highlight("key/strict_lidar_fp", s_["fp"])
                rl.highlight("key/loose_cam_fp", l_["fp"])

        if args.out:
            out = project_path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "summary.json").write_text(
                json.dumps(
                    {"root": str(root), "frames": frames, "iou": args.iou, "modes": summaries}, indent=1
                ),
                encoding="utf-8",
            )
            print(f"[out] {out / 'summary.json'}")
            rl.highlight("out", str(out / "summary.json"))


if __name__ == "__main__":
    main()

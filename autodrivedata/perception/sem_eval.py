"""P2-A 判据:像素级分割 GT(`sem_*/`)vs `sem_bev` 的预测 —— **图像 / BEV 双口径**。

## 这条补的是 Plan2 §8 记的那个缺口

`sem_bev` 链路 2026-09 就通了,但 Plan2 原话是「**本仓无像素级分割 GT,对比缺裁判口径**」
—— 意思是那张图只能看、不能打分。现在采集侧用 CARLA `sensor.camera.semantic_segmentation`
出了 GT(`collect_surround --sem`),这里就是那个裁判。

## 为什么报两个口径,而不是一个

| 口径 | 量什么 | 塌了说明 |
|---|---|---|
| **图像空间** mIoU | 分割网络在本项目图上的好坏 | 模型不行 |
| **BEV 空间** 逐类 IoU | 分割误差**经投影之后**还剩多少 | 投影/地面假设不行 |

两个都报才分得开这两件事。只报 BEV 的话,分割差与投影差会互相掩盖;只报图像空间的话,
就回答不了"这张 BEV 图到底能不能用"。

## 三类映射与"两侧都不算"的类

见 [`sem_tags.py`](sem_tags.py) 头注 —— 那里是判据公平性的唯一落点。**被排除的 tag 占比
由本模块一并报出**(`excluded_share`),不许静默设上限。

## 自证(每次运行都跑,不是开关)

1. **恒等**:把 GT 掩膜当成预测灌进同一套累加器 ⇒ 两个口径的 mIoU 都必须**恰好 1.0**。
   它同时钉住"预测的键名 == GT 的类名" —— 键名对不上时不是 1.0 而是 0,当场红。
2. **几何**:`Sky` 的像素射线必须**打不到地面**、`Roads` 的必须**几乎全打到**
   (`ground_intersection`,与 BEV 投影**同一个**函数)。翻图 / 转置 / 相机对错位
   全都过不了这一条,而它们在下游只表现为"mIoU 偏低",看不出根因。

用法:
  python -m autodrivedata.perception.sem_eval --root outputs/surround_sem --frames 0-19
  python -m autodrivedata.perception.sem_eval --root <root> --frames 0-19 --self-test  # 不出模型,只验判据
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from autodrivedata.map.mapviz import BEV_X, BEV_Y, cam_pose
from autodrivedata.perception.sem_bev import (
    BEV_PX,
    GROUND_Z_OFF,
    init_camera,
    mask_to_bev,
    predict_masks,
)
from autodrivedata.perception.sem_tags import (
    GT_CLASSES,
    SEM_TAGS,
    decode_tag_png,
    excluded_share,
    masks_from_tags,
    paint_tags,
)
from autodrivedata.utils import runlog
from autodrivedata.utils.geometry import ground_intersection
from autodrivedata.utils.paths import project_path

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None

CAMS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


@dataclass
class Counts:
    """逐类混淆计数。**累加全局 TP/FP/FN 再算 IoU**(micro-average),不平均每帧 IoU
    —— 后者会让"只出现几像素的类"和"占半张图的类"权重相同,而视野里这两者根本不等价。"""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, gt: np.ndarray, pred: np.ndarray) -> None:
        g, p = gt.astype(bool), pred.astype(bool)
        self.tp += int((g & p).sum())
        self.fp += int((~g & p).sum())
        self.fn += int((g & ~p).sum())

    @property
    def iou(self) -> float:
        den = self.tp + self.fp + self.fn
        return float(self.tp) / den if den else float("nan")


@dataclass
class Scoreboard:
    """一个口径的逐类计数。"""

    space: str
    counts: dict[str, Counts] = field(default_factory=lambda: {c: Counts() for c in GT_CLASSES})

    @property
    def miou(self) -> float:
        vals = [v.iou for v in self.counts.values() if not np.isnan(v.iou)]
        return float(np.mean(vals)) if vals else float("nan")

    def line(self) -> str:
        parts = [f"{c}={self.counts[c].iou:.3f}" for c in GT_CLASSES]
        return f"{self.space:>8s}: " + "  ".join(parts) + f"   mIoU={self.miou:.3f}"


def bev_shape() -> tuple[int, int]:
    """BEV 面板尺寸(与 `sem_bev` 同式同常量)。"""
    return int((BEV_Y[1] - BEV_Y[0]) / BEV_PX), int((BEV_X[1] - BEV_X[0]) / BEV_PX)


def geometry_oracle(
    tag: np.ndarray,
    world_cam: tuple[tuple[float, float, float], tuple[float, float, float]],
    intrinsics,
    ground_z: float,
    *,
    n: int = 400,
    seed: int = 0,
) -> dict[str, tuple[int, int]]:
    """★ GT 的**几何自证**:Sky 的射线打不到地面,Roads 的几乎全打到。

    用 `ground_intersection`(与 BEV 投影**同一个函数**)对随机采样的像素打射线。
    翻图 / 转置 / 相机与 tag 对错位都会让这两个比例塌向 0.5,而它们在下游只表现为
    "mIoU 偏低",看不出根因。

    返回 `{"sky": (命中数, 抽样数), "road": (...)}` —— **返回计数而不是比值**:
    比值要在整批上再汇总,而"每路比值求平均"会被侧向相机那种"一个 Sky 像素都没有"
    的情形毁掉(实测同一份数据:按比值平均 0.099、按全局汇总 0.000)。

    ⚠️ 键名只有两个,**由本函数与调用方各自写死**。这里派生过两次键名
    (`roads_hit` vs `road_hit`、`sky_hit_n` vs `sky_n`),两次都是**静默累加 0**,
    只有最终数是 nan 才看得出来 —— 所以调用方对不上就直接 `KeyError`,别用 `get`。
    """
    rng = np.random.default_rng(seed)
    out: dict[str, tuple[int, int]] = {}
    for key, tag_name in (("sky", "Sky"), ("road", "Roads")):
        ys, xs = np.where(tag == SEM_TAGS[tag_name])
        if len(xs) == 0:
            out[key] = (0, 0)
            continue
        idx = rng.choice(len(xs), min(n, len(xs)), replace=False)
        hits = sum(
            1
            for u, v in zip(xs[idx], ys[idx], strict=True)
            if ground_intersection(world_cam, intrinsics, float(u), float(v), ground_z) is not None
        )
        out[key] = (hits, len(idx))
    return out


def _parse_frames(spec: str) -> list[int]:
    frames: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            frames.extend(range(int(a), int(b) + 1))
        elif tok:
            frames.append(int(tok))
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="带 `sem_*/` 的 surround root(collect_surround --sem)")
    ap.add_argument("--frames", default="0-19", help="帧范围 0-19 或逗号列表")
    ap.add_argument("--out", default="outputs/sem_eval", help="输出根(预览图)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="**不出模型**:把 GT 当预测灌进同一套累加器,两个口径的 mIoU 都必须恰好 1.000。"
        "用来在数据没准备好 / 怀疑判据本身时先验尺子(键名对不上时它会红,而出模型时只是数难看)",
    )
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    with runlog.run("autodrivedata.perception.sem_eval") as rl:
        root = project_path(args.root)
        rl.input(str(root), "root")
        calib = json.loads((root / "calib.json").read_text())
        if not calib.get("semantic"):
            raise SystemExit(
                f"{root} 的 calib.json 没有 `semantic` 键 —— 这不是 `collect_surround --sem` 的产物"
            )
        ego_poses = json.loads((root / "ego_pose.json").read_text())
        cams = [c for c in CAMS if (root / f"sem_{c.lower()}").is_dir()]
        if not cams:
            raise SystemExit(f"{root} 下没有 sem_*/ 目录")
        frames = _parse_frames(args.frames)
        first = next((root / f"sem_{cams[0].lower()}").glob("*.png"))
        tag0 = decode_tag_png(first.read_bytes())
        h, w = tag0.shape
        size = (w, h)
        print(f"[data] {root.name} | {len(frames)} 帧 × {len(cams)} 相机 | tag {w}×{h}")

        img_sb = Scoreboard("图像")
        bev_sb = Scoreboard("BEV")
        bshape = bev_shape()
        shares: dict[str, list[float]] = {"excluded_obstacle": [], "non_drivable_surface": []}
        # 几何自证按**全局命中/全局抽样**汇总,**不平均每路的比值** —— 侧向相机常常一个
        # Sky 像素都没有,而"1 个像素打中了"会记成 1.0 把均值整个抬起来(2026-10-01 实测:
        # 同一份数据按比值平均报 0.099、按全局汇总报 0.000)。
        orc: dict[str, int] = {"sky_hit": 0, "sky_n": 0, "road_hit": 0, "road_n": 0}
        preview_done = False
        probe_done = False

        if args.self_test:
            print("[self-test] 不出模型:预测 := GT(两个口径的 mIoU 都必须恰好 1.000)")
            yolo11 = yolopv2 = device = None
        else:
            if YOLO is None:
                raise SystemExit("ultralytics 未装(autodrivedata env)")
            device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
            yolo11 = YOLO(str(project_path("weights/yolo11s-seg.pt")))
            yolopv2 = (
                torch.load(
                    str(project_path("outputs/models/yolopv2.pt")), map_location="cpu", weights_only=False
                )
                .to(device)
                .float()
                .eval()
            )
            print(f"[model] YOLOPv2({args.imgsz}) + YOLO11s-seg on {device}")

        for fid in frames:
            token = f"{fid:06d}"
            ep = ego_poses[token] if token in ego_poses else ego_poses[fid]
            ego = [ep["x"], ep["y"], ep["z"], ep["yaw"], ep["pitch"], ep["roll"]]
            ground_z = ego[2] - GROUND_Z_OFF
            gt_bev = {c: np.zeros(bshape, dtype=bool) for c in GT_CLASSES}
            pr_bev = {c: np.zeros(bshape, dtype=bool) for c in GT_CLASSES}
            for cam in cams:
                # 目录名 = `发"cam"/"sem"_` + 小写相机名(即 `cam_front` / `sem_cam_front`)。
                # ⚠️ 别写成 `f"cam_{cam.lower()}"` —— `cam` 本身已是 `CAM_FRONT`,
                # 那样拼出来是 `cam_cam_front`,文件永远不存在,而**测试照样跑完**
                # (全是 nan),只有自证会红。
                tag_p = root / f"sem_{cam.lower()}" / f"{token}.png"
                rgb_p = root / cam.lower() / f"{token}.png"
                if not tag_p.exists() or not rgb_p.exists():
                    continue
                tag = decode_tag_png(tag_p.read_bytes())
                img = cv2.imread(str(rgb_p))
                intrinsics, se = init_camera(calib[cam], size)
                world_cam = cam_pose(ego, se)
                gt = masks_from_tags(tag)
                if args.self_test:
                    pred = gt
                else:
                    pred = predict_masks(img, yolopv2, yolo11, device, size, args.imgsz)
                    # ★ 键名必须与 GT 的类名**逐字相同** —— 对不上时三类全空,
                    #   mIoU 会是个漂亮的 0 而不是崩,所以这儿当面擂一遍。
                    if set(pred) != set(GT_CLASSES):
                        raise SystemExit(f"预测键名 {sorted(pred)} ≠ GT 类名 {sorted(GT_CLASSES)}")
                if not probe_done:  # ★ 累积器自证(每次运行都跑,不是开关)
                    probe = Scoreboard("恒等")
                    for c in GT_CLASSES:
                        probe.counts[c].add(gt[c], gt[c])
                        # ⚠️ **必须先挡 nan**:`abs(nan - 1.0) > 1e-12` 是 **False**,
                        # 写成只比阈值的话,"一条掩膜都没进来"这种最该红的情况**静默通过**
                        # —— 2026-10-01 就是这么放过了一个路径拼错的(全 nan 全绿)。
                        if np.isnan(probe.counts[c].iou) or abs(probe.counts[c].iou - 1.0) > 1e-12:
                            raise SystemExit(
                                f"判据自证失败:类 {c} 用 GT 当预测时 IoU={probe.counts[c].iou}"
                                f"(tp={probe.counts[c].tp}) —— 空掩膜或键名对不上"
                            )
                    probe_done = True
                for c in GT_CLASSES:
                    img_sb.counts[c].add(gt[c], pred[c])
                    gt_bev[c] |= mask_to_bev(
                        gt[c], world_cam, intrinsics, ground_z, ego, bshape, max_pixels=None
                    )
                    pr_bev[c] |= mask_to_bev(
                        pred[c], world_cam, intrinsics, ground_z, ego, bshape, max_pixels=None
                    )
                for k, v in excluded_share(tag).items():
                    shares[k].append(v)
                o = geometry_oracle(tag, world_cam, intrinsics, ground_z, seed=fid)
                # 命中数与抽样数**分开**累加 ⇒ 最后算全局比值(见 orc 的定义处)。
                # `o` 与 `orc` 的键在这里显式对应:派生过一次,静默错了两回。
                for key, hit_key, n_key in (("sky", "sky_hit", "sky_n"), ("road", "road_hit", "road_n")):
                    hits, ns = o[key]
                    orc[hit_key] += hits
                    orc[n_key] += ns
                if not preview_done:
                    out_dir = project_path(args.out)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    vis = np.hstack(
                        [
                            cv2.resize(img, (w // 2, h // 2)),
                            cv2.resize(paint_tags(tag)[:, :, ::-1], (w // 2, h // 2)),
                        ]
                    )
                    cv2.imwrite(str(out_dir / f"gt_preview_{token}_{cam.lower()}.png"), vis)
                    preview_done = True
            for c in GT_CLASSES:
                bev_sb.counts[c].add(gt_bev[c], pr_bev[c])
            if (fid + 1) % 5 == 0 or fid == frames[-1]:
                print(f"[frame {fid + 1}/{frames[-1] + 1}] " + img_sb.line() + " || " + bev_sb.line())

        if args.self_test:
            ok = abs(img_sb.miou - 1.0) < 1e-9 and abs(bev_sb.miou - 1.0) < 1e-9
            print(
                f"[self-test] 图像 mIoU={img_sb.miou:.6f}  BEV mIoU={bev_sb.miou:.6f} ⇒ {'通过' if ok else '★ 失败'}"
            )
            if not ok:
                raise SystemExit("判据自证失败:GT 当预测时 mIoU 不是 1.0 —— 键名或累加器有问题")
        print()
        print(img_sb.line())
        print(bev_sb.line())
        print(
            f"  被排除类像素占比(均值):"
            f" obstacle 类(Rider/Motorcycle/Bicycle/Train)={np.nanmean(shares['excluded_obstacle']):.4%}"
            f"  非可行驶地面(Sidewalks/Terrain/Ground)={np.nanmean(shares['non_drivable_surface']):.4%}"
        )
        sky_rate = orc["sky_hit"] / orc["sky_n"] if orc["sky_n"] else float("nan")
        road_rate = orc["road_hit"] / orc["road_n"] if orc["road_n"] else float("nan")
        print(
            f"  几何自证(射线打中地面比例,全局汇总):"
            f"Sky={sky_rate:.4f}(应 ≈0,n={int(orc['sky_n'])})"
            f"  Roads={road_rate:.4f}(应 ≈1,n={int(orc['road_n'])})"
        )
        # BEV 窗口与"哪些类真的投影进来了" —— 空类在 mIoU 里被跳过,不报出来就等于
        # 悄悄把 mIoU 的定义改成了"有样本的那几类的均值"。
        print(f"  BEV 窗口:横向 {BEV_X} m、纵向 {BEV_Y} m @ {BEV_PX} m/px ⇒ {bshape[0]}×{bshape[1]} px")
        for c in GT_CLASSES:
            n = bev_sb.counts[c].tp + bev_sb.counts[c].fp + bev_sb.counts[c].fn
            if n == 0:
                print(f"  ⚠️ BEV 类 {c} **一个像素都没有**(GT 与预测都是空)⇒ 未计入 mIoU")
        rl.highlight("image_miou", round(img_sb.miou, 4))
        rl.highlight("bev_miou", round(bev_sb.miou, 4))
        for c in GT_CLASSES:
            rl.highlight(f"image_iou/{c}", round(img_sb.counts[c].iou, 4))
            rl.highlight(f"bev_iou/{c}", round(bev_sb.counts[c].iou, 4))
        rl.highlight("sky_ray_ground_hit", round(sky_rate, 4))
        rl.highlight("road_ray_ground_hit", round(road_rate, 4))
        rl.highlight("excluded_obstacle_share", round(float(np.nanmean(shares["excluded_obstacle"])), 6))
        rl.highlight("n_frames", len(frames))
        rl.highlight("self_test", bool(args.self_test))


if __name__ == "__main__":
    main()

"""P-D 教程 08 单目测距评估:检测框 → 距离,与 KITTI GT 真距对照。

输入:`outputs/kitti_ab_day_clear/training/{image_2,label_2,calib}/`(KITTI 布局)。

检测框口径两种(CLI `--detector` 二选一):
- **project**(默认):**GT 3D 框角点投影框**="2D 检测框 = 已知位姿投影框"的诚实基线。
  背景:采集器输出的 label 2D 列(kitti_day_clear)59/97 是退化框(零宽),无法作检测框;
  改用 8 角点 → p2 投影 → 前端角点 u/v min/max(box_2d_from_3d,与采集器同投影口径)。
- **yolo**:**YOLO 检测框**("生产口径")。逐帧 ultralytics 推理 → 与 GT 投影框 IoU 贪心匹配
  (match_dets_to_gt,conf 降序),命中行的检测框 = YOLO 框;GT 真距仍是 label 列 13。

GT 真距 = label_2 3D 位置的相机系 z(第 **13** 列,米);KITTI 列序 11/12/13/14
  = x/y/z/ry(旧口径"12 列 = z"把 y 当 z,近距全错)。
近距数据质量红线:车直面相机 z<7m 时侧向角点投影把 2D 框拉爆(镜面),且 label
  2D 列成退化框(零宽)→ 剔除不入统计(如实,不注水)。

两种方法:
- **迭代深度法**(尺寸假设):z = H·fy / 框高,H=1.6m(车)。单目测距的尺度基线。
- **地平面投影法**:框底中心像素经 ground_intersection 打到地面,相机前向距离。
  需相机高度(=挂点 z 1.65m)、地面 z(0)、相机俯仰。相机无俯仰 → 返回 None(如实)。

输出:逐框 distance 表格(method, z_pred, z_gt, err_pct) + 汇总统计 + JSON。
评估口径:10-20m 带误差 <10% 为 P-D 验收(7-10m 贴脸区框高被侧向角点拉大,
系统低估,如实排除);20m+ 如实报告(尺度歧义主导)。

用法:
  PYTHONPATH=$PWD python bin/mono_distance.py [--root outputs/kitti_ab_day_clear] [--max-frames 70]
  # 生产口径(YOLO 检测框;需 CUDA)
  PYTHONPATH=$PWD python bin/mono_distance.py --root outputs/kitti_ab_day_clear --max-frames 70 \
    --detector yolo --json outputs/mono_distance/results_yolo.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from autodrivedata import attribution as attr
from autodrivedata.calib.core import CameraIntrinsics
from autodrivedata.geometry import box_2d_from_3d, mono_depth_from_box
from autodrivedata.mono_depth import box_to_ground_distance
from autodrivedata.paths import project_path

# ultralytics 懒加载(yolo 分支才需要):避免基线(纯 numpy)路径触 heavy deps/torch。
_YOLO_IMPORT_ERR: Exception | None = None
try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
except Exception as _e:  # pragma: no cover — 无 ultralytics 环境跑基线不受影响
    _YOLO_IMPORT_ERR = _e
    YOLO = None  # type: ignore[assignment]  # 仅在 yolo 分支(已查 _YOLO_IMPORT_ERR)解引用

REAL_CAR_HEIGHT_M = 1.6  # 尺寸假设:轿车高(迭代法)
GROUND_Z = 0.0  # 地平面 z(CARLA 地面 = 0)
CAM_Z = 1.65  # 相机挂点高度(距地面,与 carla_common.SENSOR_OFFSET 一致)
CAM_YAW_RAD = 0.0  # 相机朝向 = ego 前向 yaw=0(车道直行基线)
CAM_PITCH_RAD = 0.0  # 相机无俯仰:地平面投影法对该 rig 不适用(None)

# YOLO 生产口径默认权重(AutoLabel 项目内的 kitti 微调模型;本仓不携带权重)
DEFAULT_YOLO_WEIGHT = (
    "/root/autodl-tmp/Documents/Projects/AutoLabel/auto2dlabel/weights/kitti_finetune/"
    "yolo11s_kitti/weights/best.pt"
)


def _parse_gt(line: str) -> dict:
    """KITTI label 行 → {cls, bbox2d_raw, params3d, cam_z}。

    bbox2d_raw 保留 label 自带的 2D 框(列 4-7)作对照(部分采集数据该列是
    退化框,零宽,不作检测框用);检测框统一用 box_2d_from_3d 从 3D 框投影。
    KITTI 列序(与 eval_kitti 同一权威):  type trunc occ alpha | x1 y1 x2 y2 |
      h[8] w[9] l[10] | x[11] y[12] z[13] ry[14]。
    """
    f = line.split()
    x1, y1, x2, y2 = (float(v) for v in f[4:8])  # label 2D 列(参考,可能退化)
    h, w, l = (float(v) for v in f[8:11])  # 尺寸
    x, y, z, ry = (float(v) for v in (f[11], f[12], f[13], f[14]))  # 3D 位置 + ry
    return {
        "cls": f[0],
        "bbox2d_raw": (x1, y1, x2, y2),
        "params3d": (x, y, z, h, w, l, ry),
        "cam_z": z,
    }


def _read_fy(calib_txt: str) -> float:
    """KITTI calib P2 矩阵 → fy(1,1 项)。"""
    for line in calib_txt.splitlines():
        if line.startswith("P2:"):
            return float(line.split()[1 + 5])  # 12 值后第 5(= y 焦距项 [1][1])
    raise ValueError("missing P2 in calib")


PYTHON_INTERPOLATION_ERR_PCT = 10.0  # 近距(<20m)验收阈值:迭代深度法误差 < 该值
MIN_GT_Z = 7.0  # 剔除贴脸框:车直面相机(z<7m)时侧向角点投影越界、2D 框被拉爆,测距失真


def eval_mono_distance(
    image_2: Path,
    label_2: Path,
    calib: Path,
    *,
    max_frames: int,
    camera: CameraIntrinsics,
    cam_z: float,
    cam_yaw_rad: float,
) -> dict:
    """对逐帧 GT 2D 框跑两个距离方法,返回逐框记录 + 汇总。

    范:用 GT 3D 框角点投影出的 2D 框("已知位姿投影框"诚实基线,见模块 docstring
        box_2d_from_3d)——生产管线里检测框从模型来,这里用投影框做"已知检测框,
        测距准不准"的诚实基线,且天然是正常 2D 框(不退化)。
    近距剔除 z<7m:车直面相机时侧向角点把 2D 框拉爆(镜面投影),测距失真,
    如实排除(数据质量红线,不注水)。
    """
    del image_2  # image_2 目录保留在签名中(输入完整性),本方法只用 label + calib
    frames = sorted(label_2.glob("*.txt"))[:max_frames]
    rows: list[dict] = []
    for fp in frames:
        fy = _read_fy((calib / fp.name).read_text(encoding="utf-8"))
        k = camera
        for line in fp.read_text(encoding="utf-8").splitlines():
            gt = _parse_gt(line)
            z = gt["cam_z"]
            if not (MIN_GT_Z < z < 40.0 and abs(gt["params3d"][0]) < 6.0):
                continue
            bbox = box_2d_from_3d(gt["params3d"], camera)
            if bbox is None:
                continue  # 全在相机后
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue  # 投影框退化(理论上不发生,保险)不入统计
            z_iter = mono_depth_from_box(y2 - y1, REAL_CAR_HEIGHT_M, fy)
            # 地平面投影:框底中心像素 → 地平面交点 → 前向距离
            z_ground = box_to_ground_distance(
                k,
                (1.2, 0.0, cam_z),
                (CAM_PITCH_RAD, cam_yaw_rad, 0.0),
                GROUND_Z,
                ((x1 + x2) / 2, y2),
                cam_yaw_rad,
            )
            rows.append(
                {
                    "frame": fp.stem,
                    "cls": gt["cls"],
                    "bbox2d": [x1, y1, x2, y2],
                    "z_gt": z,
                    "z_iter": z_iter,
                    "z_ground": z_ground,
                }
            )
    return {"rows": rows, "n_rows": len(rows)}


def _error_row(r: dict, key: str) -> float:
    z = r[key]
    if z is None or not np.isfinite(z):
        return float("nan")  # 该方法不可用(如无俯仰的地面投影)
    return abs(z - r["z_gt"]) / r["z_gt"] * 100.0


def _summarize(res: dict) -> dict:
    rows = res["rows"]
    out: dict = {"n_rows": len(rows), "methods": {}}
    for key, name in (("z_iter", "迭代深度法(尺寸假设)"), ("z_ground", "地平面投影法")):
        r_avail = [r for r in rows if r[key] is not None and np.isfinite(r[key])]
        if not r_avail:
            out["methods"][name] = {"available": 0}
            continue
        errs = np.array([_error_row(r, key) for r in r_avail])
        near = np.array([r["z_gt"] < 20.0 for r in r_avail])
        mid = np.array([10.0 <= r["z_gt"] < 20.0 for r in r_avail])
        out["methods"][name] = {
            "available": len(r_avail),
            "available_frac": round(len(r_avail) / len(rows), 3),
            "mean_err_pct": round(float(np.nanmean(errs)), 2),
            "median_err_pct": round(float(np.nanmedian(errs)), 2),
            "near20_mean_err_pct": round(float(np.nanmean(errs[near])), 2) if near.any() else None,
            "near_ok_frac": round(float((errs[near] < PYTHON_INTERPOLATION_ERR_PCT).mean()), 3)
            if near.any()
            else None,
            # 10-20m 带:避开贴脸车(z<10m)投影框高被侧向角点拉大的系统偏差后,
            # 迭代法误差应 <10%(P-D 验收口径的可信带)
            "z10_20_mean_err_pct": round(float(np.nanmean(errs[mid])), 2) if mid.any() else None,
            "z10_20_ok_frac": round(float((errs[mid] < PYTHON_INTERPOLATION_ERR_PCT).mean()), 3)
            if mid.any()
            else None,
            "max_gt_z": round(float(max(r["z_gt"] for r in r_avail)), 2),
        }
    return out


def _yolo_detect_frames(
    image_2: Path,
    model: Any,
    names: dict[int, str],
    conf: float,
    limit: int | None,
) -> dict[str, list[attr.Detection]]:
    """逐帧 YOLO → {frame_stem: [Detection]}(与 eval_2d_ab.detect 同循环结构)。

    names 是 model.names(COCO 类号→名),检测框转 attr.Detection(cls, x1..y2, conf)。
    无检测帧(res.boxes is None)返回空 list。推理 device=0(GPU)。
    """
    out: dict[str, list[attr.Detection]] = {}
    for f in sorted(image_2.glob("*.png"))[:limit]:
        res = cast(Results, list(model.predict(f, conf=conf, verbose=False, device=0))[0])
        dets: list[attr.Detection] = []
        for b in res.boxes or []:
            c = attr.norm_cls(names[int(b.cls.item())])
            if not c:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            dets.append(attr.Detection(c, x1, y1, x2, y2, float(b.conf.item())))
        out[f.stem] = dets
    return out


def _yolo_rows(
    label_2: Path,
    calib: Path,
    yolo_dets: dict[str, list[attr.Detection]],
    *,
    camera: CameraIntrinsics,
    cam_z: float,
    cam_yaw_rad: float,
    iou_thr: float,
) -> tuple[list[dict], dict]:
    """生产口径行:GT 投影框作匹配锚,命中 YOLO 框即生产框;行结构与基线完全同 shape。

    对每帧:过 z 滤波(同基线 MIN_GT_Z/|x|<6/z<40)且 box_2d_from_3d 非 None 的 GT
    定为"可评估 GT";YOLO 同类检测按 conf 降序贪心认领(match_dets_to_gt,IoU≥iou_thr);
    命中行的 bbox2d 用 YOLO 框、z_gt 取 GT 列 13 真距、z_iter/z_ground 照抄基线。
    返回 (rows, match_stats{n_gt_boxes, n_matched, n_yolo_unmatched})。
    """
    rows: list[dict] = []
    n_gt, n_matched, n_un = 0, 0, 0
    for fp in sorted(label_2.glob("*.txt")):
        fy = _read_fy((calib / fp.name).read_text(encoding="utf-8"))
        gts: list[attr.GtBox2D] = []
        zs: list[float] = []
        for line in fp.read_text(encoding="utf-8").splitlines():
            gt = _parse_gt(line)
            z = gt["cam_z"]
            if not (MIN_GT_Z < z < 40.0 and abs(gt["params3d"][0]) < 6.0):
                continue
            b = box_2d_from_3d(gt["params3d"], camera)
            if b is None or b[2] <= b[0] or b[3] <= b[1]:
                continue  # 相机后/退化投影框 → 不作匹配锚(同基线守卫)
            gts.append(
                attr.GtBox2D(cls=gt["cls"], x1=b[0], y1=b[1], x2=b[2], y2=b[3], truncation=0.0, distance_m=z)
            )
            zs.append(z)
        dets = yolo_dets.get(fp.stem, [])
        m = attr.match_dets_to_gt(gts, dets, iou_thr)
        n_gt += len(gts)
        n_matched += len(m)
        n_un += len(dets) - len(m)
        for i, g in enumerate(gts):
            if i not in m:
                continue  # 无生产框命中 → 不入统计(如实,漏检不注水)
            x1, y1, x2, y2 = m[i].x1, m[i].y1, m[i].x2, m[i].y2
            z_iter = mono_depth_from_box(y2 - y1, REAL_CAR_HEIGHT_M, fy)
            z_ground = box_to_ground_distance(
                camera,
                (1.2, 0.0, cam_z),
                (CAM_PITCH_RAD, cam_yaw_rad, 0.0),
                GROUND_Z,
                ((x1 + x2) / 2, y2),
                cam_yaw_rad,
            )
            rows.append(
                {
                    "frame": fp.stem,
                    "cls": g.cls,
                    "bbox2d": [x1, y1, x2, y2],
                    "z_gt": zs[i],
                    "z_iter": z_iter,
                    "z_ground": z_ground,
                }
            )
    stats = {"n_gt_boxes": n_gt, "n_matched": n_matched, "n_yolo_unmatched": n_un}
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root", default="outputs/kitti_drive")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--json", default="outputs/mono_distance/results.json")
    ap.add_argument(
        "--detector",
        choices=("project", "yolo"),
        default="project",
        help="检测框来源:project=GT 3D 投影框(诚实基线);yolo=YOLO 检测框(生产口径)",
    )
    ap.add_argument("--yolo-weight", default=DEFAULT_YOLO_WEIGHT, help="YOLO 权重(KITTI 微调模型)")
    ap.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO 置信度阈值")
    ap.add_argument("--iou-thr", type=float, default=0.5, help="YOLO 框与 GT 投影框匹配 IoU 阈值")
    args = ap.parse_args()

    root = project_path(args.root)
    cam_k = CameraIntrinsics(width=1242, height=375, fov_h_deg=90)
    image_2 = root / "training" / "image_2"
    label_2 = root / "training" / "label_2"
    calib = root / "training" / "calib"

    match_stats: dict | None = None
    if args.detector == "project":
        res = eval_mono_distance(
            image_2,
            label_2,
            calib,
            max_frames=args.max_frames,
            camera=cam_k,
            cam_z=CAM_Z,
            cam_yaw_rad=CAM_YAW_RAD,
        )
    else:
        # 生产口径:YOLO 推理(懒加载,仅此分支才触 torch/ultralytics)
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("--detector yolo 需要 CUDA(YOLO 推理 device=0);基线用 --detector project")
        if _YOLO_IMPORT_ERR is not None:
            raise RuntimeError(f"ultralytics 导入失败:{_YOLO_IMPORT_ERR}")
        model = cast(Any, YOLO)(args.yolo_weight)  # ultralytics 导入失败已被上面 SystemExit 拦截
        names = model.names
        yolo_dets = _yolo_detect_frames(image_2, model, names, conf=args.yolo_conf, limit=args.max_frames)
        rows, match_stats = _yolo_rows(
            label_2,
            calib,
            yolo_dets,
            camera=cam_k,
            cam_z=CAM_Z,
            cam_yaw_rad=CAM_YAW_RAD,
            iou_thr=args.iou_thr,
        )
        res = {"rows": rows, "n_rows": len(rows)}

    summary = _summarize(res)
    out = {
        "root": str(root),
        "real_car_height_m": REAL_CAR_HEIGHT_M,
        "detector": args.detector,
        "yolo_weight": args.yolo_weight if args.detector == "yolo" else None,
        "yolo_conf": args.yolo_conf if args.detector == "yolo" else None,
        "iou_thr": args.iou_thr if args.detector == "yolo" else None,
        "match_stats": match_stats,
        "summary": summary,
        "rows": res["rows"],
    }
    out_path = project_path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"== 单目测距对比({len(res['rows'])} 框,GT 真距,detector={args.detector}) ==")
    if match_stats:
        print(f"  匹配: {match_stats}")
    for m, st in summary["methods"].items():
        print(f"{m}: {st}")
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()

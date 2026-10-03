"""静态道具 GT 的判据:投影框 vs **渲染轮廓**(离线纯值,**不 import carla**)。

## 这条补的是"静态目标有真值、没判据"那个软处

P2 的静态 GT(`static_gt/`)从一开始就只出真值、不出判据 —— 它是"地图事实"的转写,
没有东西能证伪它。道具通道不一样:**它有一个独立的 oracle** —— 采集时同挂一台
`sensor.camera.instance_segmentation`,把每帧的 actor id 图落成 `prop_inst/*.png`。
于是"GT 框对不对"变成一个可以数的量:**渲染出来的轮廓有没有被投影框包住**。

## 为什么必须拿渲染轮廓当裁判,而不是"投影一遍看看像不像"

`Actor.bounding_box` 对**转过**的 actor 给出 `(extent, rotation)` 自相矛盾的读数
(2026-10-01 实测:锥桶 yaw=0 读 `0.3431×0.3450`、yaw=30° 读 `0.1246×0.4704`,真值
`0.3441×0.3441`)——盒被**剪切**了。后果拿渲染轮廓量:**3.4% 的物体像素落在投影框外**,
IoU 0.897 → 0.708。而这两版投影**都画得出图、都像那么回事**,只有数轮廓分得开。
⇒ 采集侧因此只用 yaw=0 探针那一读(`carla_common.measure_actor_size_yaw0`);
本判据负责**验证那条纪律真的落地了**。

## 三个数

| 量 | 定义 | 应当 |
|---|---|---|
| `coverage` | 轮廓像素落在投影框内的比例 | **≈1.0**(框**包不住**物体时掉下来) |
| `tightness` | 投影框面积 / 轮廓外接矩形面积 | ≈1.0(框**过大**时鼓起来) |
| `iou` | 投影框 vs 轮廓外接矩形 | 越高越好 |

⚠️ **口径差半格**:`box_to_gt_line` 给的是**角点投影的 min/max**(连续坐标),掩膜给的是
**像素索引** min/max。一个像素在索引 `i` 上占的是 `[i, i+1)`。所以 coverage 天然有
±1 px 量级的出入 —— 判据据此设容差,**不要求逐位 1.0000**。容差值由实测定,写在
模块常量上(见 `COVERAGE_TOL`)。

## 自证(每次运行都跑,不是开关)

1. **尺子在量东西**:把投影框平移 `RULER_SHIFT_PX`,coverage 必须下降 —— 恒返回
   常数或恒 1.0 的假尺子在这里立刻红。
2. **编号对得上**:GT 里的 `instance_id` 必须真的能在 id 图里找到(至少有一帧在视野内),
   否则是"GT 与 id 图配错文件/配错帧" —— 那种错在下游只表现为"coverage 全是 nan"。

用法:
  python -m autodrivedata.perception.prop_eval --root outputs/kitti_static_props
  python -m autodrivedata.perception.prop_eval --root <root> --frames 0-19 --out outputs/prop_eval
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from autodrivedata.calib.core import CameraIntrinsics
from autodrivedata.gt.core import actor_box_from_prop, project_corners
from autodrivedata.gt.props import PropBox, PropFrame
from autodrivedata.perception.sem_tags import decode_tag_png, prop_mask
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 轮廓像素少于这个数 ⇒ 该 (道具, 帧) 组合**作废**(不是"通过")。与探针同口径。
MIN_SILHOUETTE_PX = 30
#: coverage 容差:角点投影(连续坐标)与像素索引差半格,边缘像素天然可能差一点点。
#: 实测值见 Plan4 §P-V14;超过它才算"框没包住物体"。
COVERAGE_TOL = 0.005
#: 自证用的平移量(px)。平移后 coverage 必须比不平移低。
RULER_SHIFT_PX = 40


@dataclass
class Row:
    """一个 (道具, 帧) 组合的读数。"""

    frame: str
    instance_id: int
    label: str
    mask_px: int
    coverage: float
    tightness: float
    iou: float
    proj_bbox: tuple[float, float, float, float]
    mask_bbox: tuple[float, float, float, float]
    #: 8 角点是否全在画幅内。False ⇒ 物体被画幅裁断,按当前投影口径框会偏小,
    #: **这条样本不并入裁决**(单独计数报出来),见 `project_gt_box`。
    all_inside: bool = True


@dataclass
class PropStat:
    """一个道具在整个序列上的汇总。"""

    instance_id: int
    label: str
    type_id: str
    n_frames_in_view: int = 0
    min_coverage: float = float("nan")
    min_iou: float = float("nan")
    rows: list[Row] = field(default_factory=list)


def _px_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _iou_box(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else float("nan")


def _coverage(mask: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """轮廓像素落在投影框内的比例 —— 框**包不住**物体时这条会掉下来。"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return float("nan")
    inside = (xs >= box[0]) & (xs <= box[2]) & (ys >= box[1]) & (ys <= box[3])
    return float(inside.mean())


def _tightness(box: tuple[float, float, float, float], mask_bbox: tuple[float, float, float, float]) -> float:
    a = (box[2] - box[0]) * (box[3] - box[1])
    m = (mask_bbox[2] - mask_bbox[0]) * (mask_bbox[3] - mask_bbox[1])
    return float(a / m) if m > 0 else float("nan")


def _shift(box: tuple[float, float, float, float], dx: float) -> tuple[float, float, float, float]:
    return box[0] + dx, box[1], box[2] + dx, box[3]


def project_gt_box(p: PropBox, frame: PropFrame) -> tuple[tuple[float, float, float, float], bool] | None:
    """`PropBox` → `(2D 框, 是否全部角点落在画幅内)`;投影出图返回 None。

    镜头/内参**全部取自 GT 文件自己**(`frame.camera`)—— 判据住在 `perception/`
    (层规则禁 carla),拿不到 `carla_common.CAM_ATTRS` / `SENSOR_OFFSET`。

    ★ 第二个返回值决定这条样本**算不算数**:`box_to_gt_line` 的 2D 框只对**落在画幅内**
    的角点取 min/max,所以物体被画幅边缘裁掉时框会**小于**真实可见范围 —— 那种样本按
    本判据必然不通过,但**不是 GT 的错**(是投影口径的另一件事,见
    `gt.core.CornerProjection` 头注与 Plan4 §P-V14 待决项)。判据把它们单独计数、不并入裁决。
    """
    cam = frame.camera
    if cam is None:
        return None
    k = CameraIntrinsics(cam.width, cam.height, cam.fov_deg)
    rot_rad = cast(tuple[float, float, float], tuple(float(np.radians(a)) for a in cam.rotation_deg))
    proj = project_corners(actor_box_from_prop(p), cam.location, rot_rad, k)
    if proj is None:
        return None
    box = (
        float(proj.u[proj.inside].min()),
        float(proj.v[proj.inside].min()),
        float(proj.u[proj.inside].max()),
        float(proj.v[proj.inside].max()),
    )
    return box, proj.all_inside


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


def eval_root(root: Path, frames: list[int] | None = None) -> dict[str, Any]:
    """整份 root 的判据读数。`frames=None` = 全部帧。"""
    gt_dir = root / "training/static_prop_gt"
    inst_dir = root / "training/prop_inst"
    sem_dir = root / "training/prop_sem"
    if not gt_dir.is_dir() or not inst_dir.is_dir():
        raise SystemExit(f"{root} 不是 `collect_static_gt --props` 的产物(缺 static_prop_gt/ 或 prop_inst/)")

    ids_avail = {int(p.stem) for p in gt_dir.glob("*.json")}
    if frames is not None:
        ids_avail &= set(frames)

    stats: dict[int, PropStat] = {}
    # 类别判据:已知实例的轮廓在**语义 tag 图**里必须全是 `PROP_TAGS`
    sem_rows: list[tuple[str, int, float, int]] = []  # (帧, id, tag 命中率, 轮廓像素)
    ruler_ok: bool | None = None
    ruler_detail = ""
    n_frames = 0

    for fid in sorted(ids_avail):
        token = f"{fid:06d}"
        frame = PropFrame.from_json((gt_dir / f"{token}.json").read_text())
        inst_png = inst_dir / f"{token}.png"
        if not inst_png.exists():
            continue  # 没落 id 图 ⇒ 这一帧判不了(下面按"整段没露过面"报出来)
        ids = np.array(cv2.imread(str(inst_png), cv2.IMREAD_UNCHANGED))
        n_frames += 1

        for p in frame.props:
            if p.instance_id < 0:
                continue
            mask = ids == p.instance_id
            n_px = int(mask.sum())
            st = stats.setdefault(p.instance_id, PropStat(p.instance_id, p.label, p.type_id))
            bbox = _px_bbox(mask)
            pr = project_gt_box(p, frame)
            if n_px < MIN_SILHOUETTE_PX or bbox is None or pr is None:
                continue  # 这一帧不算数(不在视野 / 投影出图)—— 不作"通过"也不作"失败"
            proj, all_inside = pr
            st.n_frames_in_view += 1

            # ★ 类别判据:同一批实例轮廓,在**语义相机**那条独立链上必须是 `PROP_TAGS`。
            #   它证的是 `sem_tags.PROP_TAGS` 那个 tag 集(**地图自带道具**走的就是它),
            #   而几何判据证的是"框套不套得住" —— 两件事。
            sem_png = sem_dir / f"{token}.png"
            if sem_png.exists():
                tag = decode_tag_png(sem_png.read_bytes())
                pm = prop_mask(tag)
                hit = float(pm[mask].mean()) if n_px else float("nan")
                sem_rows.append((token, p.instance_id, hit, n_px))
            cov = _coverage(mask, proj)
            row = Row(
                frame=token,
                instance_id=p.instance_id,
                label=p.label,
                mask_px=n_px,
                coverage=cov,
                tightness=_tightness(proj, bbox),
                iou=_iou_box(proj, bbox),
                proj_bbox=proj,
                mask_bbox=bbox,
                all_inside=all_inside,
            )
            st.rows.append(row)
            st.min_coverage = cov if np.isnan(st.min_coverage) else min(st.min_coverage, cov)
            st.min_iou = row.iou if np.isnan(st.min_iou) else min(st.min_iou, row.iou)

            # ★ 自证(只在第一条**可判**样本上跑一次):把投影框整体平移,coverage 必须下降。
            #   恒返回 1.0 或常数的假尺子在这里立刻红 —— 而它平时**看着完全正常**
            #   (每个数都是个漂亮的 1.000)。
            #   ⚠️ **必须排除裁断样本**:它们的投影框常常巨大(近处角点投影到 ±1e17),
            #   平移 40 px 仍全覆盖 ⇒ coverage 不降 ⇒ 自证**假失败**。2026-10-01 由
            #   `test_prop_eval` 的合成夹具抓到(那段夹具的第一条样本恰好就是裁断的);
            #   真数据上前几条未必是裁断的 —— 但"未必碰得到"不是不修的理由。
            if ruler_ok is None and all_inside:
                cov_shift = _coverage(mask, _shift(proj, RULER_SHIFT_PX))
                ruler_ok = bool(cov_shift < cov)
                ruler_detail = (
                    f"原始 coverage={cov:.4f} → 平移 {RULER_SHIFT_PX}px 后 {cov_shift:.4f}"
                    f"(帧 {token} / id {p.instance_id})"
                )
                if not ruler_ok:
                    raise SystemExit(
                        f"判据自证失败:平移投影框后 coverage 没有下降 —— 尺子不量重叠。{ruler_detail}"
                    )

    # 谁一次都没露过面:GT 与 id 图配错的典型症状(文件对错帧 / instance_id 写错)
    never_seen = [s for s in stats.values() if s.n_frames_in_view == 0]
    all_rows = [r for s in stats.values() for r in s.rows]
    return {
        "root": str(root),
        "n_frames": n_frames,
        "n_props": len(stats),
        "ruler_ok": ruler_ok,
        "ruler_detail": ruler_detail,
        "stats": stats,
        "never_seen": never_seen,
        "n_clipped": sum(1 for r in all_rows if not r.all_inside),
        "sem_rows": sem_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="`collect_static_gt --props` 的产物根")
    ap.add_argument("--frames", default=None, help="帧范围 0-19 或逗号列表;默认全部")
    ap.add_argument("--out", default="outputs/prop_eval", help="输出根(目检图)")
    ap.add_argument("--no-runlog", action="store_true", help="关掉 logs/ 三件套")
    args = ap.parse_args()

    with runlog.run("autodrivedata.perception.prop_eval") as rl:
        root = project_path(args.root)
        rl.input(str(root), "root")
        frames = _parse_frames(args.frames) if args.frames else None
        res = eval_root(root, frames)

        out_dir = project_path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[data] {root.name} | {res['n_frames']} 帧有 id 图 | {res['n_props']} 个道具")
        print(f"[自证] 尺子在量重叠:{res['ruler_detail']} ⇒ {'通过' if res['ruler_ok'] else '★ 失败'}")
        print()

        rows_all: list[Row] = []
        print(
            f"  {'id':>6} {'label':<8} {'在视野':>6} {'可判':>5} {'被裁':>5} "
            f"{'min coverage':>13} {'min IoU':>8} {'最大 mask':>9}"
        )
        for st in sorted(res["stats"].values(), key=lambda s: s.instance_id):
            rows_all += st.rows
            if st.n_frames_in_view == 0:
                print(f"  {st.instance_id:>6} {st.label:<8} {0:>6} —— **一次都没露面**(GT 与 id 图可能配错)")
                continue
            judged = [r for r in st.rows if r.all_inside]
            cov_txt = f"{min(r.coverage for r in judged):>13.4f}" if judged else f"{'—':>13}"
            iou_txt = f"{min(r.iou for r in judged):>8.4f}" if judged else f"{'—':>8}"
            print(
                f"  {st.instance_id:>6} {st.label:<8} {st.n_frames_in_view:>6} "
                f"{len(judged):>5} {st.n_frames_in_view - len(judged):>5} "
                f"{cov_txt} {iou_txt} {max(r.mask_px for r in st.rows):>9}"
            )

        judged_all = [r for r in rows_all if r.all_inside]
        clipped = [r for r in rows_all if not r.all_inside]
        if not judged_all:
            raise SystemExit(
                f"没有一条**可判**样本(全部 {len(rows_all)} 条都被画幅裁断)—— "
                "判据没有样本就不是判据,不许报'通过'"
            )

        min_cov = min(r.coverage for r in judged_all)
        med_tight = float(np.median([r.tightness for r in judged_all]))
        passed = min_cov >= 1.0 - COVERAGE_TOL
        print()
        print(
            f"[判据] 可判样本 {len(judged_all)} 条:最差 coverage = {min_cov:.4f}"
            f"(容差 {COVERAGE_TOL}) | tightness 中位 = {med_tight:.4f}"
            f" ⇒ {'通过' if passed else '★ 失败:有渲染像素落在投影框外'}"
        )
        if clipped:
            worst_clip = min(r.coverage for r in clipped)
            print(
                f"[边界] 另有 {len(clipped)} 条样本物体**被画幅裁断**(角点落到画面外),"
                f"coverage 最低 {worst_clip:.4f} —— **不并入裁决**。\n"
                f"        原因:当前 2D 框只对落在画幅内的角点取 min/max,物体被裁时框必然偏小。\n"
                f"        这不是道具特有的:归档 label_2 里 690/3922(17.6%)同形,"
                f"KITTI 口径应为「全角点 min/max 再钳到画幅」。见 gt.core.CornerProjection 头注。"
            )
        # ★ 类别判据:三个已知资产 3/3 落 `Dynamic` 是**探针**量的(需要 CARLA);
        #   这里把同一条结论钉在**采集产物**上 —— 实例轮廓 ∩ 语义 tag 必须是 100% PROP_TAGS。
        if res["sem_rows"]:
            hits = [h for _, _, h, _ in res["sem_rows"] if not np.isnan(h)]
            bad = [(f, i, round(h, 4)) for f, i, h, _ in res["sem_rows"] if h < 1.0]
            print(
                f"[类别] 实例轮廓落在 `PROP_TAGS` 内的比例:最低 {min(hits):.4f}"
                f" | 样本 {len(hits)} 条 ⇒ {'通过' if not bad else '★ 失败'}"
            )
            if bad:
                print(f"        不合格样本(帧, id, 命中率)前 5:{bad[:5]}")
        else:
            print("[类别] 本 root 没落 `prop_sem/`(采集时未挂语义相机)⇒ 这条判据**未运行**")

        if res["never_seen"]:
            print(
                f"[warn] {len(res['never_seen'])} 个道具一次都没露面:"
                f"{[s.instance_id for s in res['never_seen']]} —— 查 GT 与 id 图是不是同一帧"
            )

        # 目检图:最差那一帧的 mask + 投影框叠在原图上(单看数字分不出"框偏了"与"框没包住")
        worst = min(judged_all, key=lambda r: r.coverage)
        img_p = root / "training/image_2" / f"{worst.frame}.png"
        if img_p.exists():
            img = cv2.imread(str(img_p))
            vis = img.copy()
            # ⚠️ 切片下标**必须转 int** —— `mask_bbox` 是 float 元组,`vis[float:float]`
            # 直接 `TypeError: slice indices must be integers`(2026-10-01 首跑踩)
            mx1, my1, mx2, my2 = (int(v) for v in worst.mask_bbox)
            vis[my1 : my2 + 1, mx1 : mx2 + 1] //= 2  # 轮廓区域压暗,便于看出框有没有套住
            x1, y1, x2, y2 = (int(v) for v in worst.proj_bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 140, 255), 2)  # 橙 = 投影框
            cv2.rectangle(vis, (mx1, my1), (mx2, my2), (255, 0, 255), 1)  # 品红 = 轮廓外接矩形
            cv2.imwrite(str(out_dir / f"worst_{worst.frame}_id{worst.instance_id}.png"), vis)
            print(f"[out] 最差样本目检图:{out_dir / f'worst_{worst.frame}_id{worst.instance_id}.png'}")

        rl.highlight("min_coverage", round(min_cov, 6))
        rl.highlight("passed", bool(passed))
        rl.highlight("n_samples_judged", len(judged_all))
        rl.highlight("n_samples_clipped", len(clipped))
        if res["sem_rows"]:
            rl.highlight(
                "min_prop_tag_hit",
                round(min(h for _, _, h, _ in res["sem_rows"] if not np.isnan(h)), 6),
            )
        rl.highlight("n_props_never_seen", len(res["never_seen"]))
        rl.highlight("ruler_ok", bool(res["ruler_ok"]))


if __name__ == "__main__":
    main()

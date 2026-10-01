"""时序拼接 —— 把**逐帧**的 BEV 矢量预测按位姿拼成一张全局矢量图(纯值,不 import carla/torch)。

## 与 `stitch.py` 的分工(别混)

| | 拼什么 | 位姿从哪来 |
|---|---|---|
| `stitch.py` | 跨**图**(不同 Town) | **人为摆位** —— 各 Town 是独立关卡,没有共同内容可配准 |
| 本模块 | 跨**帧**(同一段路的不同时刻) | **轨迹** —— CARLA 真值 / SLAM 估计 |

几何层**复用** `stitch.place`(它就是 `Rz(yaw)·p + t`,**正好等于 ego→world**)与
`stitch._dedup`(它的判据是"只在**来源不同**之间比") —— 时序场景里每帧就是一个来源,
天然适用;不另写一份几何。

## 位姿源(`--pose`)

| 值 | 来源 | 口径 |
|---|---|---|
| `gt` | `map_infos.json` 的 `ego2global`(按 `token` 关联) | **上界**:位姿无误差,剩下的全是感知误差 |
| `slam` | SLAM 轨迹 json(`traj_pgo.json` 一类) | **车载可做**:含定位误差 |

⚠️ **两者必须来自同一段数据才可比**。见下节。

## 数据前提(`--pose slam`)

需要同一段序列**同时**有 6 环视相机与 LiDAR。**当前磁盘上没有这样的数据集**:
`collect_surround` 只挂相机、`collect_slam` 只挂 LiDAR,两者互不支持对方的传感器
(2026-09-28 实测确认)。要跑 `--pose slam`,先采一段两者兼备的序列 —— 本模块会在
读位姿时**明确报错并给出处置**,不会静默拼出一张假图。

## 按 `seg` 分段是**硬约束**

`surround_v2` 的 5 段是**独立路线**,段缝位移实测 **56.8–109.6 m**(4 处,全在段边界上)。
跨段直拼会被拉成一条横穿全图的直线 ⇒ 本工具**逐段独立**拼接,段间不合并,统计按段报。

## 产出(**JSON + PNG 两份,一向如此**)

| 文件 | 内容 |
|---|---|
| `<out>.json` | 全局矢量图:逐实例 `{class, seg, points(世界系 xyz), n_src}` + 统计 |
| `<out>.png` | 可视化:逐段一栏、按类着色、带图例与**自证信息**(帧数/位姿源/去重统计) |

JSON 只能喂程序;图是给人做**比对**的位置 —— 两类一起出,不省略。

用法:
  # 先出逐帧契约(见 mapvec_schema),再拼
  python -m autodrivedata.map.eval_maptr --infos outputs/surround_v2/map_infos.json \\
      --root outputs/surround_v2 --ckpt outputs/maptr_v2_singleF.pt \\
      --start 200 --out-frames outputs/surround_pred
  python -m autodrivedata.map.stitch_temporal \\
      --frames outputs/surround_pred --infos outputs/surround_v2/map_infos.json \\
      --pose gt --out outputs/temporal_map/gt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from autodrivedata.gt.export.nuscenes import NUS_RADAR_CHANNELS
from autodrivedata.map.mapvec import MAPTR_CLASSES
from autodrivedata.map.mapvec_schema import MapVecFramePred, load_frame
from autodrivedata.map.mapviz import Px
from autodrivedata.map.stitch import DEDUP_CELL as DEDUP_CELL_M
from autodrivedata.map.stitch import _dedup
from autodrivedata.utils import fonts, runlog
from autodrivedata.utils.paths import project_path

#: 按类着色。取**路面场景罕见色**(与 mapviz 的 pred/gt 同族避让):重叠/撞色会让"哪类是哪个"
#: 读不出来 —— 这是本项目绘制体系的既有口径(见 Plan2 §P-M.8 的撞色记录)。
CLASS_COLOR: dict[str, tuple[int, int, int]] = {
    "divider": (255, 96, 96),  # 红:分隔带
    "ped_crossing": (120, 200, 255),  # 蓝:人行横道
    "boundary": (255, 214, 102),  # 黄:路边界
    "centerline": (140, 255, 170),  # 绿:中心线
}

#: 跨帧去重容差(米)。**比 `stitch.DEDUP_TOL`(0.05) 大**:跨图去重比的是"同一段线被导了两遍"
#: (几何几乎逐位相同),而跨帧比的是"同一段路被不同视角各测了一遍" —— 位置本来就有差,
#: 容差取到 0.05 会一条都合并不掉(等于没去重)。
TEMPORAL_DEDUP_TOL = 0.5

#: 三种融合口径 —— **给同一个词消歧**(README/报告里"时序拼接"至少指三件事):
#:   overlay = 逐帧叠画,不清洗(**看原始预测的分布**,不是可用地图)
#:   dedup   = 复用跨图去重(Hausdorff)。**对时序是错配**,只作对照保留
#:   cluster = 真正的时间融合(Chamfer + 朝向门 + 簇内平均)← 默认
FUSION_LABEL = {
    "overlay": "逐帧叠画(未融合)",
    "dedup": "Hausdorff 去重(跨图口径,时序下错配)",
    "cluster": "Chamfer 聚类 + 簇内平均(时序融合)",
}


def load_contract_dir(src: Path) -> list[MapVecFramePred]:
    """契约目录 → 逐帧记录(按 `frame` 升序;文件名 = `{token}.json`)。"""
    files = sorted(src.glob("*.json"))
    if not files:
        raise SystemExit(f"{src} 里没有 `*.json` —— 先用 `eval_maptr --out-frames` 出逐帧契约")
    recs = [load_frame(p) for p in files]
    return sorted(recs, key=lambda r: r.frame)


def poses_from_infos(infos: Path) -> dict[str, list[float]]:
    """`token → ego2global = [x, y, z, yaw, pitch, roll]`(度)。"""
    recs = json.loads(infos.read_text(encoding="utf-8"))
    return {r["token"]: list(r["ego2global"]) for r in recs}


def poses_from_slam(traj: Path, tokens: list[str]) -> dict[str, list[float]]:
    """SLAM 轨迹 → `token → [x, y, z, yaw, 0, 0]`(**ego 相对帧 0 的位姿**)。

    ⚠️ **前提是轨迹与契约帧逐帧对应** —— 若模型/采集两边的帧集不同,这里会直接报错,
    不猜对齐方式(猜错的症状是"拼出一张形状对但整体错位的图",最难查)。
    """
    from autodrivedata.slam.slam_eval import lidar_pose_to_ego

    doc = json.loads(traj.read_text(encoding="utf-8"))
    frames = doc.get("traj") or []
    if len(frames) != len(tokens):
        raise SystemExit(
            f"SLAM 轨迹 {len(frames)} 帧 ≠ 契约 {len(tokens)} 帧 —— 两边**不是同一段数据**。\n"
            "  `--pose slam` 要求同一段序列同时有 6 环视相机与 LiDAR;当前磁盘上没有这样的集\n"
            "(collect_surround 只挂相机 / collect_slam 只挂 LiDAR)。先采一段两者兼备的序列。\n"
            f"  轨迹: {traj}\n  契约: {len(tokens)} 帧"
        )
    out: dict[str, list[float]] = {}
    for tok, rec in zip(tokens, frames, strict=True):
        e = lidar_pose_to_ego(rec["T"] if isinstance(rec["T"], list) else rec["T"])
        # 只取 yaw:`stitch.place` 是平面刚体变换(pitch/roll 丢弃)。
        # 实测 ego 俯仰量级 ~0.1°,30 m 处像素偏差 <1px ⇒ 对矢量图的影响远小于感知误差。
        yaw = math.degrees(math.atan2(e[1][0], e[0][0]))
        out[tok] = [float(e[0][3]), float(e[1][3]), float(e[2][3]), yaw, 0.0, 0.0]
    return out


def frame_to_world(
    rec: MapVecFramePred, ego: list[float], which: str = "pred"
) -> list[tuple[str, list[tuple[float, float]], float]]:
    """该帧的**预测或真值**(ego 系)→ 世界系折线 `(类, 20×(x,y), score)`。

    `which="gt"` 时取同一帧的 GT —— 用于把"预测拼出来是什么样"与"真值拼出来该是什么样"
    并排看。⚠️ GT 是**逐帧窗口裁剪后的并集**,不是完整地图矢量:同一条路被不同帧各裁到
    一段,拼起来才接近完整的;窗口外的部分**本来就不在 GT 里**(不是漏检)。
    """
    src = rec.gts if which == "gt" else rec.preds
    cos, sin = math.cos(math.radians(ego[3])), math.sin(math.radians(ego[3]))
    out = []
    for inst in src:
        if not inst.points:
            continue
        pts = [(ego[0] + x * cos - y * sin, ego[1] + x * sin + y * cos) for x, y in inst.points]
        out.append((inst.cls, pts, float(inst.score) if inst.score is not None else 0.0))
    return out


def _heading(pts: list[tuple[float, float]]) -> float:
    """折线朝向(弧度,−π/2, π/2],只关心**轴**(方向反转视为同向)。"""
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    a = math.atan2(y1 - y0, x1 - x0)
    return a - math.pi if a > math.pi / 2 else (a + math.pi if a <= -math.pi / 2 else a)


def fuse_clusters(
    insts: list[tuple[int, str, list[tuple[float, float]], float]],
    tol: float,
    heading_tol_deg: float,
) -> list[tuple[str, list[tuple[float, float]], float, int]]:
    """**时序融合**(不是去重):同类 + Chamfer ≤ `tol` + 朝向差 ≤ 门限 ⇒ 同一实例。

    为什么不能直接复用 `stitch._dedup`:它的 `_polyline_gap` 是 **Hausdorff**(最坏点主导),
    跨图比"同一段线被导两遍"(几何近乎逐位相同)合适,而跨帧比的是"同一段路被不同视角
    各测一遍" —— 折线两端和采样相位本就有差,一对最坏点就把整条判成不同(实测 tol 0.5 m
    只合掉 **0.7%**;放到 3 m 才 45%,而 3 m 已逼近**车道宽 3.5 m**,开始误合并相邻车道线)。
    故换 **Chamfer(均值)**,并加**朝向门**挡掉"平行但不同"的相邻车道。

    代表几何 = 簇内成员**对齐方向后逐点取均值**(单条观测的抖动被 N 帧平均掉);
    `score` 取均值,并记下**支持帧数** —— 它比 score 更能说明"这条线被看了多少眼"。

    匹配是单遍贪心的(按 score 降序),用 5 m 网格只比邻域 —— O(N·近邻),不做全对。
    """
    import numpy as np

    from autodrivedata.map.chamfer_ap import chamfer_distance

    htol = math.radians(heading_tol_deg)
    cell = DEDUP_CELL_M
    grid: dict[tuple[str, int, int], list[int]] = {}
    clusters: list[dict] = []  # {cls, pts(list), scores(list), heads(list)}
    for _idx, cls, pts, score in sorted(insts, key=lambda t: -t[3]):
        arr = np.asarray(pts, dtype=np.float64)
        head = _heading(pts)
        cx, cy = int(arr[:, 0].mean() // cell), int(arr[:, 1].mean() // cell)
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for ci in grid.get((cls, cx + dx, cy + dy), ()):
                    c = clusters[ci]
                    if abs(c["head"] - head) > htol and abs(abs(c["head"] - head) - math.pi) > htol:
                        continue
                    if chamfer_distance(arr, c["arr"]) <= tol:
                        best = ci
                        break
                if best is not None:
                    break
            if best is not None:
                break
        if best is None:
            clusters.append(
                {"cls": cls, "arr": arr, "pts": list(pts), "head": head, "scores": [score], "members": [arr]}
            )
            grid.setdefault((cls, cx, cy), []).append(len(clusters) - 1)
        else:
            c = clusters[best]
            c["scores"].append(score)
            # **对齐方向再入池**:方向相反的折线逐点平均会得到一条"塌掉"的线
            c["members"].append(arr[::-1] if abs(c["head"] - head) > math.pi / 2 else arr)

    out = []
    for c in clusters:
        stack = np.stack(c["members"])  # (N, 20, 2) —— 同 num_points,可逐点平均
        mean = stack.mean(axis=0)
        out.append(
            (
                c["cls"],
                [(float(x), float(y)) for x, y in mean],
                sum(c["scores"]) / len(c["scores"]),
                len(c["members"]),
            )
        )
    return out


def stitch_segments(
    recs: list[MapVecFramePred],
    poses: dict[str, list[float]],
    seg_of: dict[str, str],
    fusion: str,
    tol: float,
    heading_tol_deg: float,
    which: str = "pred",
) -> dict:
    """逐段拼接(段间不合并,见模块头注)。`fusion` = `overlay` / `dedup` / `cluster`。"""
    by_seg: dict[str, list[MapVecFramePred]] = {}
    for r in recs:
        by_seg.setdefault(seg_of.get(r.token, "?"), []).append(r)

    out: dict[str, dict] = {}
    for seg, rs in sorted(by_seg.items()):
        insts: list[tuple[int, str, list[tuple[float, float]], float]] = []
        for i, r in enumerate(rs):
            ego = poses.get(r.token)
            if ego is None:
                raise SystemExit(f"位姿里没有 token {r.token} —— `--infos` 与契约帧集不一致")
            insts.extend((i, cls, pts, sc) for cls, pts, sc in frame_to_world(r, ego, which))

        n_in = len(insts)
        if fusion == "overlay":
            fused = [(cls, pts, sc, 1) for _, cls, pts, sc in insts]
            n_drop = 0
        elif fusion == "dedup":
            from autodrivedata.map.mapvec import MapVec

            vecs = tuple(
                MapVec(cls, tuple((x, y, 0.0) for x, y in pts), (), "", "") for _, cls, pts, _ in insts
            )
            kept, n_drop, _ = _dedup(vecs, [i for i, *_ in insts], tol)
            fused = [(v.cls, [(p[0], p[1]) for p in v.points], 0.0, 1) for v in kept]
        elif fusion == "cluster":
            fused = fuse_clusters(insts, tol, heading_tol_deg)
            n_drop = n_in - len(fused)
        else:
            raise SystemExit(f"未知 fusion: {fusion}")
        out[seg] = {
            "insts": fused,
            "n_frames": len(rs),
            "n_in": n_in,
            "n_dedup": n_drop,
            "span_m": _span([pts for _, pts, _, _ in fused]),
        }
    return out


def _span(all_pts: list[list[tuple[float, float]]]) -> list[float]:
    """该段世界系包络 (w, h)(米)—— 自证用:拼出来的图跨度该与路线长度同量级。"""
    if not all_pts:
        return [0.0, 0.0]
    xs = [p[0] for pts in all_pts for p in pts]
    ys = [p[1] for pts in all_pts for p in pts]
    return [round(max(xs) - min(xs), 1), round(max(ys) - min(ys), 1)]


def _panel_transform(x0: float, yy1: float, scale: float, ox: float, oy: float) -> Px:
    """世界系 (x, y) → 面板像素 的变换。

    **做成模块级工厂而不是在渲染循环里定义闭包**:循环内定义会绑住当轮的
    `scale/ox/oy/x0/yy1`(ruff B023)。本轮内调用跑起来是对的,但那是巧合 ——
    一旦改成"先收集变换再统一画"就会全体静默用最后一轮的参数。

    现在返回 `mapviz.Px`(值对象)而不是闭包:点云底图要**批量**换算(百万点走 numpy),
    闭包做不到;顺带那条 B023 隐患也从"记得挪出循环"变成"结构上不可能"。
    """
    return Px(sx=scale, sy=-scale, ox=ox - x0 * scale, oy=oy + yy1 * scale)


def segment_world_points(
    recs: list[MapVecFramePred],
    poses: dict[str, list[float]],
    seg_of: dict[str, str],
    root: Path,
    channels: list[str],
    max_points: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """逐段把点云累积到**世界系**(拼接大图的底图)。

    ★ 点数上限是**显式**的:全段 240 帧 × 6.4 万点 = 1500 万点,逐点提交给 PIL 要十几 GB
    的 Python 对象。超限时按**均匀步长**抽样,并把 `dropped` 数返回给调用方**写进产物** ——
    静默截断会让人把"没画全"读成"这里就没有点"(项目对"静默上限"的既有纪律)。

    返回 `({seg: (N,3) float32}, {seg: 被丢弃的点数})`。
    """
    from autodrivedata.map import bev_base

    by_seg: dict[str, list[MapVecFramePred]] = {}
    for r in recs:
        by_seg.setdefault(seg_of.get(r.token, "?"), []).append(r)

    out: dict[str, np.ndarray] = {}
    dropped: dict[str, int] = {}
    for seg, rs in sorted(by_seg.items()):
        chunks: list[np.ndarray] = []
        for r in rs:
            ego = poses.get(r.token)
            if ego is None:
                raise SystemExit(f"位姿里没有 token {r.token} —— `--infos` 与契约帧集不一致")
            lid, rad = bev_base.frame_points(root, int(r.frame), channels)
            chunks.append(bev_base.to_world(np.concatenate([lid, rad], axis=0), ego))
        pts = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=np.float64)
        if len(pts) > max_points:
            keep = np.linspace(0, len(pts) - 1, max_points).astype(np.int64)
            dropped[seg] = len(pts) - max_points
            pts = pts[keep]
        else:
            dropped[seg] = 0
        out[seg] = pts.astype(np.float32)
    return out, dropped


def render(
    stats: dict,
    *,
    pose_source: str,
    fusion: str,
    out_png: Path,
    panel_h: int = 520,
    base: dict[str, np.ndarray] | None = None,
    base_dropped: dict[str, int] | None = None,
    which: str = "pred",
) -> tuple[Path, dict[str, int]]:
    """逐段一栏可视化 + 图例 + 自证信息(**不是装饰**:看图 ≈ 读判据)。

    `base` = 逐段的世界系点云底图(LiDAR+Radar)。**它只是上下文** —— 模型是纯相机的,
    点云不进网络(见 `bev_base` 头注)。返回 `({seg: 画出的点数})` 供调用方做数值自证。
    """
    from autodrivedata.map import bev_base

    segs = [s for s in sorted(stats) if stats[s]["insts"]]
    if not segs:
        raise SystemExit("没有可画的段 —— 拼接结果为空")
    panel_w = 1180
    canvas = Image.new("RGB", (panel_w, panel_h * len(segs) + 96), (16, 16, 20))
    d = ImageDraw.Draw(canvas)
    drawn: dict[str, int] = {}

    for i, seg in enumerate(segs):
        st = stats[seg]
        y0 = 96 + i * panel_h
        # 该段的世界系包络 → 面板像素(等比,按较大的一维适配)。
        # **底图点也参与包络** —— 否则底图会被面板裁掉一半,看着像"那里没点"。
        pts = [p for _, ip, _, _ in st["insts"] for p in ip]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bx = base.get(seg) if base else None
        if bx is not None and len(bx):
            xs = [*xs, float(bx[:, 0].min()), float(bx[:, 0].max())]
            ys = [*ys, float(bx[:, 1].min()), float(bx[:, 1].max())]
        x0, x1, yy0, yy1 = min(xs), max(xs), min(ys), max(ys)
        span = max(x1 - x0, yy1 - yy0, 1e-6)
        scale = (panel_h - 76) / span
        ox = (panel_w - (x1 - x0) * scale) / 2
        oy = y0 + 42

        to_px = _panel_transform(x0, yy1, scale, ox, oy)

        d.rectangle([10, y0 - 30, panel_w - 10, y0 + panel_h - 46], outline=(60, 60, 70))
        # 底图先画(矢量压在上面);`scatter_world` 已按高度分两层着色
        if bx is not None and len(bx):
            drawn[seg] = bev_base.scatter_world(d, bx, to_px)
        else:
            drawn[seg] = 0
        n_drop = (base_dropped or {}).get(seg, 0)
        head = (
            f"[{seg}]  {st['n_frames']} 帧  实例 {st['n_in']} → {len(st['insts'])}"
            f"({FUSION_LABEL.get(fusion, fusion)})  包络 {st['span_m'][0]}×{st['span_m'][1]} m"
            + (f"  底图 {drawn[seg]} 点(抽样丢弃 {n_drop})" if bx is not None and len(bx) else "")
        )
        d.rectangle([10, y0 - 30, 760, y0 - 4], fill=(0, 0, 0))
        fonts.draw_text(d, (16, y0 - 27), head, size=19, fill=(230, 230, 230))
        for cls, ip, _sc, _n in st["insts"]:
            col = CLASS_COLOR.get(cls, (200, 200, 200))
            d.line([to_px(x, y) for x, y in ip], fill=col, width=2)

    # 图例 + 自证(位姿源/预测还是 GT 必须写在图上 —— 换任一项就是另一张图)
    fonts.draw_text(
        d,
        (16, 14),
        f"时序拼接:逐帧 BEV 矢量{'预测' if which == 'pred' else '真值'} → 全局矢量图(按 seg 独立拼)",
        size=26,
        fill=(255, 255, 255),
    )
    fonts.draw_text(
        d,
        (16, 48),
        f"位姿源 = {pose_source}   融合 = {FUSION_LABEL.get(fusion, fusion)}   容差 {TEMPORAL_DEDUP_TOL} m"
        f"   段数 {len(segs)}" + ("   底图 = LiDAR+Radar(不参与模型)" if base else ""),
        size=20,
        fill=(190, 190, 200),
    )
    lx = panel_w - 320
    for j, (cls, col) in enumerate(CLASS_COLOR.items()):
        yy = 16 + j * 24
        d.rectangle([lx, yy + 4, lx + 18, yy + 18], fill=col)
        fonts.draw_text(d, (lx + 26, yy), cls, size=19, fill=(210, 210, 210))
    if base:
        for j, (lbl, col) in enumerate(
            [
                ("lidar 地面", bev_base.LIDAR_GROUND_COLOR),
                ("lidar 离地", bev_base.LIDAR_OBJECT_COLOR),
                ("radar", bev_base.RADAR_COLOR),
            ]
        ):
            yy = 16 + (j + len(CLASS_COLOR)) * 24
            d.rectangle([lx, yy + 4, lx + 18, yy + 18], fill=col)
            fonts.draw_text(d, (lx + 26, yy), lbl, size=19, fill=(210, 210, 210))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    return out_png, drawn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="逐帧契约目录(`eval_maptr --out-frames` 的产物)")
    ap.add_argument("--infos", required=True, help="B2 map_infos.json(取 token→seg 与 GT 位姿)")
    ap.add_argument("--pose", choices=("gt", "slam"), default="gt", help="位姿源;见模块头注")
    ap.add_argument("--slam-traj", default=None, help="`--pose slam` 时的轨迹 json")
    ap.add_argument(
        "--fusion",
        choices=("overlay", "dedup", "cluster"),
        default="cluster",
        help="融合口径;见模块头注与 FUSION_LABEL(cluster = 真正的时序融合)",
    )
    ap.add_argument("--tol", type=float, default=TEMPORAL_DEDUP_TOL, help="融合容差(米)")
    ap.add_argument("--heading-tol", type=float, default=30.0, help="cluster 模式的朝向门(度)")
    ap.add_argument(
        "--which",
        choices=("pred", "gt"),
        default="pred",
        help="拼哪一份:`pred` = 模型预测(默认)/ `gt` = 同帧真值。⚠️ GT 是**逐帧窗口裁剪后的并集**,不是完整地图矢量",
    )
    ap.add_argument(
        "--lidar-root",
        default=None,
        help="LiDAR/Radar 点云根目录(**必须与 --infos/--frames 同一段数据**,按 `frame` 对齐)"
        "⇒ 拼接大图加一层世界系点云底图。⚠️ 底图只是给人看的上下文:模型是纯相机的,点云不进网络",
    )
    ap.add_argument("--no-radar-base", action="store_true", help="底图只画 LiDAR 不画雷达")
    ap.add_argument(
        "--base-max-points",
        type=int,
        default=4_000_000,
        help="底图点数上限(超出按均匀步长抽样);丢弃量会写进 JSON 与图上标题 —— 不静默截断",
    )
    ap.add_argument("--out", required=True, help="输出基路径(落 <out>.json + <out>.png)")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    with runlog.run("autodrivedata.map.stitch_temporal") as rl:
        rl.input(args.frames, "mapvec-pred")
        rl.input(args.infos, "map-infos")
        src = project_path(args.frames)
        recs = load_contract_dir(src)
        info_recs = json.loads(project_path(args.infos).read_text(encoding="utf-8"))
        seg_of = {r["token"]: r["seg"] for r in info_recs}

        if args.pose == "gt":
            poses = poses_from_infos(project_path(args.infos))
        else:
            if not args.slam_traj:
                raise SystemExit("--pose slam 需要 --slam-traj")
            poses = poses_from_slam(project_path(args.slam_traj), [r.token for r in recs])

        stats = stitch_segments(recs, poses, seg_of, args.fusion, args.tol, args.heading_tol, args.which)
        out = project_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)

        # 底图:同一段数据的点云按**同一个位姿源**累到世界系(`--pose slam` 时用 SLAM 轨迹)。
        # ⚠️ 用 GT 位姿画底图、SLAM 位姿画矢量(或反过来)会得到"两张各自都对、叠起来错位"的图。
        base: dict[str, np.ndarray] | None = None
        base_dropped: dict[str, int] = {}
        if args.lidar_root:
            base, base_dropped = segment_world_points(
                recs,
                poses,
                seg_of,
                project_path(args.lidar_root),
                [] if args.no_radar_base else list(NUS_RADAR_CHANNELS),
                args.base_max_points,
            )

        doc = {
            "source": {
                "frames": args.frames,
                "infos": args.infos,
                "pose": args.pose,
                "which": args.which,
                "fusion": args.fusion,
                "tol_m": args.tol,
                "heading_tol_deg": args.heading_tol,
                "lidar_root": args.lidar_root,
                "base_max_points": args.base_max_points if args.lidar_root else None,
            },
            "classes": list(MAPTR_CLASSES),
            "segments": {
                seg: {
                    **{k: v for k, v in st.items() if k != "insts"},
                    "n_out": len(st["insts"]),
                    "instances": [
                        {"class": c, "points": [list(p) for p in pts], "score": round(sc, 4), "n_src": n}
                        for c, pts, sc, n in st["insts"]
                    ],
                }
                for seg, st in stats.items()
            },
            "base_layer": None,  # 真值在下面渲染完再填(drawn 计数要等画完才有)
        }
        # **别用 `with_suffix`**:`--out a/b/k3_v1.0` 的 `.0` 会被当成扩展名吃掉,
        # 产物静默变成 `k3_v1.json`(实测踩到)。一律**追加**后缀。
        out_json = Path(str(out) + ".json")
        out_png = Path(str(out) + ".png")
        # 先画图(拿到 drawn 计数)再落 JSON —— 计数要进 JSON,顺序不能反
        _, drawn = render(
            stats,
            pose_source=args.pose,
            fusion=args.fusion,
            out_png=out_png,
            base=base,
            base_dropped=base_dropped,
            which=args.which,
        )
        doc["base_layer"] = (
            None
            if base is None
            else {
                "note": "LiDAR/Radar 仅为视觉上下文,不参与模型(纯相机)",
                "per_segment": {
                    seg: {"n_drawn": n, "n_dropped": base_dropped.get(seg, 0), "n_total": int(len(base[seg]))}
                    for seg, n in drawn.items()
                },
            }
        )
        out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

        n_in = sum(st["n_in"] for st in stats.values())
        n_out = sum(len(st["insts"]) for st in stats.values())
        print(f"[stitch] {len(recs)} 帧 / {len(stats)} 段 | 实例 {n_in} → {n_out}")
        for seg, st in sorted(stats.items()):
            extra = ""
            if base is not None:
                extra = (
                    f"  底图 {int(len(base[seg]))} 点 → 画 {drawn.get(seg, 0)}"
                    f"(丢弃 {base_dropped.get(seg, 0)})"
                )
            print(
                f"  {seg}: {st['n_frames']} 帧  实例 {st['n_in']} → {len(st['insts'])}"
                f"  包络 {st['span_m'][0]}×{st['span_m'][1]} m{extra}"
            )
        print(f"→ {out_json}\n→ {out_png}")
        rl.highlight("pose_source", args.pose)
        rl.highlight("fusion", args.fusion)
        rl.highlight("n_frames", len(recs))
        rl.highlight("n_segments", len(stats))
        rl.highlight("n_instances_in", n_in)
        rl.highlight("n_instances_out", n_out)
        rl.highlight("tol_m", args.tol)
        rl.highlight("which", args.which)
        if base is not None:
            rl.highlight("base_points_drawn", sum(drawn.values()))
            rl.highlight("base_points_dropped", sum(base_dropped.values()))
        rl.artifact(out_json, "temporal-map-json")
        rl.artifact(out_png, "temporal-map-png")  # 图与 JSON 同列一份,别只报数据


if __name__ == "__main__":
    main()

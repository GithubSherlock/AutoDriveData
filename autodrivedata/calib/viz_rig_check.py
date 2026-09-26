#!/usr/bin/env python3
"""rig 的两件目检交付物:① **配置图**(纯值,随时可出)② **六视角实拍图**(需 CARLA)。

用户口径(2026-09-23):"在 `outputs/calib_check` 中输出自车以及传感器标定配置的可视化图,
包括传感器名称,在自车上的位置,传感器采集方向和范围(如 CAM_FRONT 0 ± 22.5 度)",
以及"检查生成的六视角 FoV 图像,观察是否有重叠区域且不含有车体像素"。

| 产物 | 是什么 | 需要 CARLA |
|---|---|---|
| `rig_layout_{rig}.png` | 配置图:俯视挂点 + 视锥 / 方位环(重叠橙、盲区红带度数)/ 数字表(`az ± fov/2`) | ✗ |
| `views_{rig}.png` | 六视角**原生像素**拼图 + 逐格标注 + 底部**线性方位尺**;车体像素**就地染成品红** | ✓ |
| `report_{rig}.json` | 逐通道 ego 像素数 / 实挂 vs 声明偏差 / 方位覆盖表 / 相邻共视逐对读数 | ✓ |

**为什么图不能替代 `verify_nus_calib`**:这张图证明的是"这一段画面里没有车体",
而"声明 ≠ 渲染"这类失效模式(§P-M.7)在图上**看不见** —— 只有判据 ①⑥ 的数值能抓。
本脚本复用 `autodrivedata/calib/rig_check.py` 的同一套相机与同一份读数,故两者口径不会分叉。

**布局图的分区是硬坐标**(见 `rigviz.draw_rig_layout`),这里只负责喂数据与落盘。

用法:
  python -m autodrivedata.calib.viz_rig_check                          # 只出配置图(不需 CARLA)
  python -m autodrivedata.calib.viz_rig_check --rig wide --live        # 配置图 + 实拍图 + 报告
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from autodrivedata.calib.camera_rig import coverage_table  # noqa: E402
from autodrivedata.calib.rigviz import (  # noqa: E402
    CAM_COLOR,
    CAM_SHORT,
    DIM,
    GAP_COLOR,
    INK,
    draw_rig_layout,
)
from autodrivedata.gt.export.nuscenes import NUS_RIGS, camera_calibs, camera_fov  # noqa: E402

# 进包后不再需要 sys.path 引导(旧 bin/ 非包布局的产物)
from autodrivedata.utils import fonts  # noqa: E402
from autodrivedata.utils.paths import project_path  # noqa: E402

TILE_LABEL_SIZE = 26
# 底部方位尺:6 条泳道(每相机一行)+ 并集/盲区行 + 刻度与注脚
RULER_H = 300
EGO_TINT = (255, 0, 200)  # 车体像素染色:只要非 0 就一定扎眼

# 拼图行序**不沿用 `SURROUND_CAMS` 字典序**(§P-L.6 的教训:字典序会把后三路打乱成
# 一眼读不出的顺序)。按"前三个 / 后三个"排,与俯视图上的左右关系一致。
GRID_ROWS: list[list[str]] = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
]

_COVERAGE_CACHE: dict[str, dict[str, Any]] = {}


def coverage_of(rig: str) -> dict[str, Any]:
    """该 rig 的方位覆盖表(缓存:画标注与写报告用同一份,重算没必要)。"""
    if rig not in _COVERAGE_CACHE:
        _COVERAGE_CACHE[rig] = coverage_table(camera_calibs(rig), camera_fov(rig))
    return _COVERAGE_CACHE[rig]


def layout_image(rig: str) -> Image.Image:
    """配置图(纯值:标定 + FoV 都来自该 rig 的声明表,不碰 CARLA)。"""
    calibs, fov, cov = camera_calibs(rig), camera_fov(rig), coverage_of(rig)
    title = f"自车 + 相机标定配置 · rig = {rig}"
    subtitle = (
        f"{len(fov)} 路环视 | 方位覆盖 {cov['coverage_frac'] * 100:.2f}% "
        f"| 盲区合计 {cov['gap_total_deg']:.3f}° | 画幅 1600×900(与落盘口径一致)"
    )
    return draw_rig_layout(calibs, fov, cov, title, subtitle)


def _az_range(rig: str, name: str) -> tuple[float, float, float, float]:
    """该通道的 (方位角, 半 FoV, 下限, 上限),如 CAM_FRONT → (0.3, 27.5, −27.2, +27.8)。"""
    c = coverage_of(rig)["cameras"][name]
    half = c["fov_deg"] / 2
    return c["az_nus_deg"], half, c["az_nus_deg"] - half, c["az_nus_deg"] + half


def ruler_image(rig: str, width: int) -> Image.Image:
    """线性方位尺:**逐相机各占一条泳道**(上→下按方位角排序)+ 末行并集与盲区。

    与方位环同一份数字(`coverage_table`),只是**线性排开**,于是:
    任一竖线穿过的泳道数 = 该方位被几路同时覆盖(≥2 即重叠,直读);末行灰带 = 并集,
    露出红底的地方就是盲区(度数写在带上)。

    ★ 为什么不是"所有彩带画在同一行":先画的会被后画的**整条盖住** —— 实测
    CAM_BACK_LEFT(145°±55°)的 90–200° 落在 CAM_BACK(120–240°)之下,彩带与短码
    一起被盖,屏幕上只剩下最后画的那条,读数完全失真。泳道版把"谁盖住谁"这个歧义消掉。
    """
    img = Image.new("RGB", (width, RULER_H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    cov = coverage_of(rig)
    order = sorted(cov["cameras"].items(), key=lambda kv: kv[1]["az_nus_360"])
    lane_h, gap_lane_h = 30, 34
    pad, left = 40, 78  # left:左侧留出泳道标签(短码)的位置
    x0, x1 = left, width - pad
    y_top = 52

    def x_of(az360: float) -> float:
        """方位角(度,已在 `[0°,360°]`)→ 画布 x。

        ★ **不能写 `az % 360`**:`camera_rig._sector_intervals` 把跨 0° 的扇区劈成
        `(lo, 360.0)` + `(0.0, …)` 两段,`360.0 % 360 == 0` 会把右端折回**最左**,
        `ImageDraw.rectangle` 拿到 x1<x0 直接抛 `ValueError`(不是画错,是崩)。
        钳到区间即安全,顺带钉死越界值。
        """
        a = min(max(az360, 0.0), 360.0)
        return x0 + (x1 - x0) * a / 360.0

    def seg(lo: float, hi: float) -> list[tuple[float, float]]:
        """度区间 → 不跨 0° 的 (起, 止) 段(跨接缝时劈两段,与扇区同口径)。"""
        return [(lo, hi)] if lo <= hi else [(lo, 360.0), (0.0, hi)]

    # 逐相机泳道(纯色不混白:混白后的 CAM_BACK_LEFT 与盲区红几乎同色,实测一眼分不开)
    for i, (name, c) in enumerate(order):
        y = y_top + i * lane_h
        col = CAM_COLOR.get(name, (80, 80, 80))
        fonts.draw_text(d, (pad, y + lane_h // 2), CAM_SHORT.get(name, name), size=20, fill=INK, anchor="lm")
        for lo, hi in c["span"]:
            d.rectangle([x_of(lo), y + 4, x_of(hi), y + lane_h - 5], fill=col, outline=INK, width=1)
        # 短码写在**该相机自身方位角**处,白底 chip ⇒ 一定压在自家彩带上、一定读得清
        chip_x = x_of(c["az_nus_360"])
        d.rectangle(
            [chip_x - 20, y + 4, chip_x + 20, y + lane_h - 5], fill=(255, 255, 255), outline=col, width=2
        )
        fonts.draw_text(
            d, (chip_x, y + lane_h // 2), CAM_SHORT.get(name, name), size=18, fill=INK, anchor="mm"
        )

    # 末行:并集(灰) + 盲区(红底写度数)
    y = y_top + len(order) * lane_h + 8
    d.rectangle([x0, y, x1, y + gap_lane_h - 10], fill=(228, 228, 232), outline=DIM, width=1)
    for g in cov["gaps"]:
        parts = seg(g["lo"], g["hi"])
        for lo, hi in parts:
            d.rectangle([x_of(lo), y, x_of(hi), y + gap_lane_h - 10], fill=GAP_COLOR, outline=INK, width=1)
        # 度数写在最宽的那一段上 —— 带太窄(< 52 px)就不写,免得字压到邻带(数字在报告表里)
        lo, hi = max(parts, key=lambda p: p[1] - p[0])
        if x_of(hi) - x_of(lo) >= 52:
            fonts.draw_text(
                d,
                ((x_of(lo) + x_of(hi)) / 2, y + (gap_lane_h - 10) // 2),
                f"{g['deg']:.2f}°",
                size=17,
                fill=(255, 255, 255),
                anchor="mm",
            )
    fonts.draw_text(d, (pad, y + gap_lane_h // 2 - 5), "并集", size=20, fill=INK, anchor="lm")

    for az in range(0, 361, 30):
        x = x_of(az)
        d.line([x, y_top - 8, x, y_top], fill=DIM, width=2)
        d.line([x, y + gap_lane_h - 10, x, y + gap_lane_h - 2], fill=DIM, width=2)
        fonts.draw_text(d, (x, 20), f"{az}°", size=18, fill=DIM, anchor="mt")
    # 画在图上的是给人看的字,不写 Markdown 记号(`**` 会原样出现在图里)
    note = (
        "方位尺(线性,0° = 车头,+ 向左):上 6 行 = 逐通道覆盖 az±fov/2,"
        "同一竖线穿过的行数 ≥ 2 即重叠;末行 灰 = 并集、红 = 盲区(写度数)"
    )
    fonts.draw_text(d, (pad, RULER_H - 26), note, size=18, fill=DIM)
    return img


def _tinted(tile: Image.Image, inst: np.ndarray, ego_id: int) -> Image.Image:
    """把 ego 自己的像素染成品红(0 px ⇒ 空操作,画面看不出任何改动)。

    ★ 必须走 `np.array(...)`(拷贝)再 `Image.fromarray` 回包:`np.asarray(PIL 图)`
    拿到的是**只读**视图,就地赋值抛 `assignment destination is read-only`。
    wide rig 上这条分支**从不触发**(0 px),是官方 rig(CAM_BACK 43%)才第一次走到的
    —— "0 px 的 rig 测不到染色路径"本身就是为什么两个 rig 都要出的理由。
    """
    mask = inst == ego_id
    if not mask.any():
        return tile
    arr = np.array(tile)
    arr[mask] = EGO_TINT
    return Image.fromarray(arr)


def views_image(
    rig: str,
    inst: dict[str, np.ndarray],
    imgs: dict[str, Image.Image],
    counts: dict[str, dict[str, Any]],
    ego_id: int,
) -> Image.Image:
    """六视角**原生像素**拼图(不缩放不裁剪)+ 逐格标注 + 底部方位尺。"""
    tiles: list[list[tuple[str, Image.Image]]] = []
    for row_names in GRID_ROWS:
        row: list[tuple[str, Image.Image]] = []
        for name in row_names:
            tile = _tinted(imgs[name], inst[name], ego_id)
            az, half, lo, hi = _az_range(rig, name)
            c = counts[name]
            text = (
                f"{name}   az {az:+.1f}° ± {half:.1f}°   [{lo:+.1f}°, {hi:+.1f}°]\n"
                f"ego px = {c['px']} ({c['frac'] * 100:.4f}%)"
                + ("" if c["px"] == 0 else "   ← 画幅内有自身车体!")
            )
            # 标签连同**白底 chip** 一起画:直接压在画面上会与内容撞色
            # (实测官方 rig 的 CAM_BACK 整片品红,红字压上去几乎读不出)
            d_tile = ImageDraw.Draw(tile)
            lx, ly = 14, tile.height - 78
            w = max(fonts.width(line, TILE_LABEL_SIZE) for line in text.split("\n")) + 20
            d_tile.rectangle([lx - 8, ly - 6, lx + w, ly + 62], fill=(255, 255, 255), outline=INK, width=1)
            fonts.draw_text(
                d_tile,
                (lx, ly),
                text,
                size=TILE_LABEL_SIZE,
                fill=INK if c["px"] == 0 else GAP_COLOR,
            )
            row.append(("", tile))
        tiles.append(row)

    from autodrivedata.sim.live_common import compose_rows

    grid = compose_rows(tiles)
    ruler = ruler_image(rig, grid.width)
    page = Image.new("RGB", (grid.width, grid.height + ruler.height), (255, 255, 255))
    page.paste(grid, (0, 0))
    page.paste(ruler, (0, grid.height))
    return page


def run_live(rig: str, host: str, port: int, out_dir: Path) -> dict[str, Any]:
    """实拍:6× RGB + 6× instance_seg → ego 像素读数 + 拼图 + 报告。**只销毁本次 spawn 的 actor**。"""
    import carla

    from autodrivedata.calib.rig_check import RigCameras, covisibility, decode_ids, ego_pixel_counts
    from autodrivedata.sim.carla_common import spawn_ego, sync_mode
    from autodrivedata.sim.live_common import image_to_pil, mount_deviation_of, rig_spec

    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)
    ego = spawn_ego(world)
    cams = RigCameras(world, ego, rig, ("rgb", "instance_segmentation"))
    try:
        frames = cams.step()  # 两种蓝图共用一套挂点,一次 tick 同时取齐
        inst = decode_ids(frames["instance_segmentation"])
        counts = ego_pixel_counts(inst, ego.id)
        imgs = {n: image_to_pil(img) for n, img in frames["rgb"].items()}
        # 实挂 vs 声明(`spawn_ego` 前已 tick,读到的不是陈旧位姿)
        mounts, rots = rig_spec(rig)
        dev_t, dev_y = mount_deviation_of({n: cams.sensor(n, "rgb") for n in mounts}, ego, mounts, rots)
        cov = coverage_of(rig)
        pairs = [p for p in cov["pairs"] if p["overlap_deg"] > 1e-9]
        rows = covisibility(world, ego, cams, pairs)
    finally:
        cams.close()
        ego.destroy()  # 只销毁自己 spawn 的 ego;锥体由 `covisibility` 内部销毁

    views = views_image(rig, inst, imgs, counts, ego.id)
    views_path = out_dir / f"views_{rig}.png"
    views.save(views_path)
    root = project_path(".")
    return {
        "resolution": [next(iter(imgs.values())).width, next(iter(imgs.values())).height],
        "ego_actor_id": ego.id,
        "ego_pixels": {k: {"px": v["px"], "frac": v["frac"]} for k, v in counts.items()},
        "ego_pixels_pass": bool(all(v["px"] == 0 for v in counts.values())),
        "mount_deviation": {"translation_m": dev_t, "yaw_deg": dev_y},
        "covisibility": rows,
        "covisibility_pass": bool(all(r.get("skipped") or r["both_visible"] for r in rows)),
        "views_png": str(views_path.relative_to(root)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", choices=NUS_RIGS, default="nuscenes")
    ap.add_argument("--live", action="store_true", help="另出六视角实拍图 + 报告(需 CARLA)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out-dir", default="outputs/calib_check")
    args = ap.parse_args()

    root = project_path(".")
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layout = layout_image(args.rig)
    layout_path = out_dir / f"rig_layout_{args.rig}.png"
    layout.save(layout_path)
    rep: dict[str, Any] = {
        "rig": args.rig,
        "generated_by": "autodrivedata/calib/viz_rig_check.py",
        "layout_png": str(layout_path.relative_to(root)),
        "layout_size": list(layout.size),
        "coverage": coverage_of(args.rig),
    }
    print(f"[layout] {layout_path.resolve()}  {layout.width}×{layout.height}")

    rc = 0
    if args.live:
        rep.update(run_live(args.rig, args.host, args.port, out_dir))
        print(f"[views ] {(root / rep['views_png']).resolve()}")
        for name, v in rep["ego_pixels"].items():
            print(f"   {name:16s} ego px={v['px']:8d}  {v['frac'] * 100:.4f}%")
        rc = 0 if (rep["ego_pixels_pass"] and rep["covisibility_pass"]) else 1
    rep_path = out_dir / f"report_{args.rig}.json"
    rep_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[report] {rep_path.resolve()}")
    cov = rep["coverage"]
    print(f"覆盖 {cov['coverage_frac'] * 100:.2f}% / 盲区合计 {cov['gap_total_deg']:.3f}°")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

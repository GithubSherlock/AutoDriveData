#!/usr/bin/env python3
"""标定修正的人工复核图:两张**把判据数字烧进画面**的拼图(自证,不是装饰)。

`autodrivedata/calib/probe_calib.py` 的结论是纯数值的(A0–A6 → `report.json`),它只落一张
`overlay.png`。本脚本补齐"人能对着看"的两张,每张都携带**它自己编码的数字**,
故看图 ≈ 读判据:

| 图 | 编码什么 | 怎么看 |
|---|---|---|
| `check_raw.png` / `check_overlay.png` | **同一帧**的 raw 与残差 overlay | 绿点铺满路面/墙面且几乎无红点 = 外参与内参自洽;两张逐像素差 = **去重后的像素数**(≤ 采样点数,差 = 两点落到同一像素的碰撞数) |
| `check_geometry.png` | **当前** rig 的官方方位角 → yaw 极坐标轮 + 逐相机数字表 + 像素约定读数 | 数字全由 `camera_rig` 现算,与 `verify_nus_calib` 判据①同源 |

⚠️ **原第三张 `check_rig_ab.png` 已于 2026-09-28 删除**:它拿 `nuscenes` 与 `legacy` 并排,
是"镜像 bug 已修"的目视证据 —— 而 `legacy` 口径本身已按用户裁决移除。那份证据冻结在
[docs/legacy-rig-archive.md](../../docs/legacy-rig-archive.md)(图 `assets/legacy-rig/`)。

格式沿用 §P-L.6 的约定:一律 `live_common.compose_rows` **原生像素**拼图,不缩放、不裁剪,
标签画在格**内**(不额外占画布高度)。

用法(需 CARLA 服务器):
  bash tools/carla_server.sh start
  PYTHONPATH=$PWD python autodrivedata/calib/viz_calib_check.py
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
from typing import Any, cast

import carla
import numpy as np
from PIL import Image, ImageDraw

from autodrivedata.calib.camera_rig import NUS_CAMERA_CALIBS, NUS_CAMERA_RIG
from autodrivedata.calib.depth_codec import decode_depth
from autodrivedata.calib.probe_calib import (
    H,
    SensorRig,
    W,
    clean_world,
    depth_residuals,
    draw_residuals,
    lidar_world_points,
    world_planes,
)
from autodrivedata.gt.export.nuscenes import NUS_CAMERA_FOV
from autodrivedata.sim.carla_common import CAM_ATTRS, loc, spawn_ego, sync_mode
from autodrivedata.sim.live_common import compose_rows, image_to_pil, rig_spec
from autodrivedata.utils import fonts
from autodrivedata.utils.geometry import quat_normalize, quat_to_matrix
from autodrivedata.utils.paths import project_path

CAM_COLOR: dict[str, tuple[int, int, int]] = {
    "CAM_FRONT": (255, 255, 255),
    "CAM_FRONT_LEFT": (0, 255, 128),
    "CAM_FRONT_RIGHT": (0, 200, 255),
    "CAM_BACK": (255, 200, 0),
    "CAM_BACK_LEFT": (255, 80, 80),
    "CAM_BACK_RIGHT": (200, 120, 255),
}


# ---------------------------------------------------------------- 画图小件


def stamp(
    img: Image.Image,
    lines: list[str],
    xy: tuple[int, int] = (6, 6),
    color: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
    size: int = 22,
) -> Image.Image:
    """在图上叠一块黑底多行标签 —— 自证数字就烧在画面里,截图即证据。

    字体/字符一律走 `autodrivedata.fonts`:`ImageFont.truetype(DejaVuSans-Bold)`
    不含任何 CJK 字形,中文会**静默**画成豆腐块(本图此前的实际症状)。
    """
    d = ImageDraw.Draw(img)
    lines = [fonts.sanitize(s) for s in lines]
    boxes = [fonts.bbox(d, s, size) for s in lines]
    lh = max(b[3] - b[1] for b in boxes) + 6
    w = max(b[2] - b[0] for b in boxes) + 12
    x, y = xy
    d.rectangle([x, y, x + w, y + lh * len(lines) + 6], fill=bg)
    for i, s in enumerate(lines):
        fonts.draw_text(d, (x + 6, y + 3 + i * lh), s, size=size, fill=color)
    return img


def border(img: Image.Image, color: tuple[int, int, int], px: int = 4) -> Image.Image:
    """给格加一圈边框(只看颜色就能分辨"修正后 / 历史版")。"""
    d = ImageDraw.Draw(img)
    for i in range(px):
        d.rectangle([i, i, img.width - 1 - i, img.height - 1 - i], outline=color)
    return img


def _dir_px(az_nus_deg: float, r: float) -> tuple[float, float]:
    """官方方位角(度,**nus 全局系 y 左**)→ 屏幕偏移(像素,x 右 / y 下)。

    nus 方向 = (cos az, sin az)(y 左)⇒ CARLA 方向 = (cos az, −sin az)(y 右)
    ⇒ 屏幕 x = carla_y·r = −sin(az)·r,屏幕 y = −carla_x·r = −cos(az)·r。
    """
    a = math.radians(az_nus_deg)
    return (-math.sin(a) * r, -math.cos(a) * r)


def _az_nus(name: str) -> float:
    """官方标定 → 该相机视线轴在 nuScenes 全局系的方位角(度)。视线轴 = 相机系 +z 列。"""
    r = quat_to_matrix(quat_normalize(NUS_CAMERA_CALIBS[name][1]))
    b = r[:, 2]
    return float(math.degrees(math.atan2(b[1], b[0])))


# ---------------------------------------------------------------- 图 3:几何裁决(纯值,不需要 CARLA)


def sheet_geometry(report: dict[str, Any] | None) -> Image.Image:
    """**当前**环视 rig 的方位布局裁决:官方方位角 → yaw 的极坐标轮 + 逐相机数字表 + 像素约定读数。

    全部数字由 `autodrivedata/camera_rig.py` 现算,不手抄。

    ⚠️ **2026-09-28**:本图原先还叠一层「修正值 vs 历史字面值」的镜像对照 —— 那依赖
    已移除的 `legacy` 口径。镜像证据已冻结在 [docs/legacy-rig-archive.md](../../docs/legacy-rig-archive.md)
    (图 `assets/legacy-rig/mirror-polar.png`),本图回归它本来的职责:**当前** rig 的几何自证。
    """
    cw, ch = 1600, 900
    img = Image.new("RGB", (cw, ch), (18, 18, 22))
    d = ImageDraw.Draw(img)
    stamp(
        img,
        [
            "check_geometry — 环视 rig 几何裁决(全部数字由 autodrivedata/camera_rig.py 现算)",
            "逐相机:官方方位角 az_nus → CARLA 口径 yaw = −az_nus(见 geometry.carla_yaw_to_nus_yaw)",
        ],
        xy=(16, 14),
        size=24,
    )

    # ---- 左:极坐标方位轮 ----
    cx, cy, r = 400, 500, 250
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(70, 70, 80), width=2)
    d.ellipse([cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2], outline=(48, 48, 56), width=1)
    for label, (dx, dy) in {"前 +x": (1, 0), "后 −x": (-1, 0), "右 +y": (0, 1), "左 −y": (0, -1)}.items():
        ex, ey = cx + dy * r, cy - dx * r
        d.line([cx, cy, ex, ey], fill=(60, 60, 68), width=1)
        # 轴名画在**圆内**(0.78 r):贴上沿会与相机名(1.10 r)叠字 —— 中文字形修好后
        # 才看得出来的排版问题(以前两边都是豆腐块,叠在一起也看不出来)
        ax_, ay_ = cx + dy * r * 0.78, cy - dx * r * 0.78
        fonts.draw_text(d, (ax_ - 30, ay_ - 26 if dx > 0 else ay_ + 8), label, size=20, fill=(150, 150, 160))
    d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(120, 120, 130))

    for name in NUS_CAMERA_RIG:
        az = _az_nus(name)
        col = CAM_COLOR[name]
        # 修正后:实线 + 端点圆点
        ox, oy = _dir_px(az, r * 0.94)
        d.line([cx, cy, cx + ox, cy + oy], fill=col, width=5)
        d.ellipse([cx + ox - 8, cy + oy - 8, cx + ox + 8, cy + oy + 8], fill=col)
        lx, ly = cx + ox * 1.10, cy + oy * 1.10
        fonts.draw_text(d, (lx - 8, ly - 10), name.replace("CAM_", ""), size=19, fill=col)
        fonts.draw_text(d, (lx - 8, ly + 10), f"{az:+.1f}°", size=17, fill=tuple(int(v * 0.8) for v in col))
    fonts.draw_text(
        d, (16, ch - 34), "极坐标:官方方位角 az_nus(度,逆时针为正 = 向左)", size=20, fill=(150, 150, 160)
    )

    # ---- 右:数字表(列位置**显式算**,不靠等宽字体的空格对齐) ----
    # 旧实现是 `f"{name:<16}{az:>12.3f}..."` + 等宽字体 DejaVuSansMono;但等宽字体
    # **一个中文字形都没有**(本图此前的豆腐块症状)。换成能画中文的字体后,它的拉丁
    # 字形是**比例宽** ⇒ 空格对齐必然错列。故列位置由 `fonts.width` 实测累加得出。
    tx, ty = 760, 150
    tsize = 20
    fonts.draw_text(
        d,
        (tx, ty - 34),
        "口径:yaw_carla = −az_nus(漏翻号 = 四个侧/后相机左右镜像 —— 成因与图见 docs/legacy-rig-archive.md)",
        size=tsize,
        fill=(230, 230, 230),
    )
    heads = ["相机", "官方 az_nus", "yaw_carla", "±FoV/2(度)"]
    aligns = ["l", "r", "r", "r"]  # 名称左对齐,数值右对齐(按各自列宽推右缘)
    rows: list[tuple[list[str], str, float]] = []
    for name in NUS_CAMERA_RIG:
        yaw = NUS_CAMERA_RIG[name][1][1]
        rows.append(
            (
                [name, f"{_az_nus(name):+.3f}", f"{yaw:+.3f}", f"±{NUS_CAMERA_FOV[name] / 2:.2f}"],
                name,
                abs(yaw),
            )
        )
    col_w = [
        max([fonts.width(heads[i], tsize)] + [fonts.width(r[0][i], tsize) for r in rows])
        for i in range(len(heads))
    ]
    col_x: list[float] = []
    x = float(tx)
    for w in col_w:
        col_x.append(x)
        x += w + 26.0
    for i, head in enumerate(heads):
        fonts.draw_text(
            d,
            (col_x[i] if aligns[i] == "l" else col_x[i] + col_w[i], ty),
            head,
            size=tsize,
            fill=(255, 255, 0),
            anchor="la" if aligns[i] == "l" else "ra",
        )
    mark_x = col_x[-1] + col_w[-1] + 26.0
    for j, (cells, name, raw) in enumerate(rows):
        y = ty + 34 * (j + 1)
        col = CAM_COLOR[name] if raw > 100.0 else (170, 170, 180)
        for i, cell in enumerate(cells):
            fonts.draw_text(
                d,
                (col_x[i] if aligns[i] == "l" else col_x[i] + col_w[i], y),
                cell,
                size=tsize,
                fill=col,
                anchor="la" if aligns[i] == "l" else "ra",
            )
        if raw > 100.0:
            fonts.draw_text(d, (mark_x, y), "← 镜像", size=19, fill=(255, 90, 90))
    fonts.draw_text(
        d,
        (tx, ty + 34 * 7 + 8),
        "镜像的四台差 110–222°;前/后相机因光轴近自逆只差 0.15–0.32° ⇒ 长期没暴露",
        size=19,
        fill=(200, 200, 210),
    )

    # ---- 右下:像素约定与实测读数(读 report.json,没有就如实说明) ----
    by = ty + 34 * 7 + 60
    if report:
        a4 = report.get("A4_lateral_regression", {})
        a3 = report.get("A3_depth_residual", {})
        nom = report.get("nominal", {})
        meds = [v["median_abs_residual_m"] for v in a3.values() if v.get("median_abs_residual_m")]
        lines = [
            "★ 像素约定 = corner(实测裁决,不是选出来的):",
            f"  A4 掩膜索引中点回归 → cx = {a4.get('cx_corner_px', float('nan')):.2f} px"
            f"(= (w−1)/2 = {nom.get('cx_corner_px', float('nan')):.1f});"
            f" f_est = {a4.get('fx_est_px', float('nan')):.2f} px(标称 {nom.get('fx_px', float('nan')):.2f})",
            f"  A3 深度残差 median|e| 六相机 {min(meds):.4f}–{max(meds):.4f} m"
            f"(center 口径同批为 0.008–0.027 m,差 ~70×)",
            "  判据 A0–A6:"
            + " ".join(f"{k}{'✓' if v else '✗'}" for k, v in report.get("verdict", {}).items()),
        ]
    else:
        lines = ["(未找到 report.json —— 先跑 autodrivedata/calib/probe_calib.py 才有实测读数)"]
    for i, s in enumerate(lines):
        fonts.draw_text(d, (tx, by + 30 * i), s, size=20, fill=(150, 255, 150))

    return img


# ---------------------------------------------------------------- 图 1:raw / overlay 同帧对


def pass_residual_pair(rig: SensorRig, args: argparse.Namespace):
    """A3 的 6 相机残差:**同一帧**出 raw 与 overlay 两张,逐像素差 = 去重后的像素数。"""
    rig.warmup(5)
    frames = rig.capture()
    poses = rig.poses()
    lidar_tf = rig.sensor("lidar:TOP").get_transform()
    lidar_loc = np.asarray(loc(lidar_tf), dtype=np.float64)
    pts_world = lidar_world_points(frames["lidar:TOP"], lidar_tf)
    rng = np.random.default_rng(args.seed)
    pts, nrm, offs = world_planes(pts_world, lidar_loc, rng)
    print(f"[A3] LiDAR {pts_world.shape[0]} 点 → 平面质量保留 {pts.shape[0]} 点")

    tiles_raw: list[tuple[str, Image.Image]] = []
    tiles_over: list[tuple[str, Image.Image]] = []
    stats: dict[str, Any] = {}
    for name in NUS_CAMERA_RIG:
        d_img = decode_depth(frames["depth"][name].raw_data, H, W)
        s = depth_residuals(pts, nrm, offs, poses[name], d_img, args.edge_radius_px)
        e = np.abs(s.residual)
        med = float(np.median(e)) if e.size else float("nan")
        raw = image_to_pil(frames["rgb"][name])
        # 标签**两张都画**(内容逐字相同)⇒ 两张的差集只剩残差点,不含标签像素
        label = [f"{name}  nuscenes(修正后)", f"采样 {len(s)}  median|e| {med:.4f} m"]
        stamp(raw, label, size=20)
        over = raw.copy()
        # `draw_residuals` 返回的是**采样点数**(不是像素数)—— 每个点只写 1 个像素,
        # 两点落到同一像素时两者不等,故判据是 `diff == 去重后的像素数` 且 `diff <= 采样数`。
        n_samples = draw_residuals(over, s)
        diff = int((np.asarray(over) != np.asarray(raw)).any(axis=2).sum())
        u = np.clip(s.uv[:, 0].astype(np.intp), 0, W - 1)
        v = np.clip(s.uv[:, 1].astype(np.intp), 0, H - 1)
        n_px = int(len(np.unique(v * W + u))) if len(s) else 0
        ok = diff == n_px and diff <= n_samples
        stats[name] = {
            "n_samples": len(s),
            "median_abs_m": None if not e.size else med,
            "painted_samples": n_samples,
            "unique_px": n_px,
            "diff_px": diff,
            "diff_equals_unique_px": diff == n_px,
            "collisions": n_samples - n_px,
        }
        tiles_raw.append(("", raw))
        tiles_over.append(("", over))
        print(
            f"  {name:<17} 采样 {len(s):>5}  median|e| {med:.4f} m  去重像素 {n_px:>5}"
            f"  同帧差集 {diff:>5} px  重采样 {n_samples - n_px:>2}"
            f"  {'✓ 一致' if ok else '✗ 不一致'}"
        )

    def rows_of(tiles: list[tuple[str, Image.Image]]) -> list[list[tuple[str, Image.Image]]]:
        return [tiles[i : i + 2] for i in range(0, len(tiles), 2)]

    raw_img = compose_rows(rows_of(tiles_raw))
    over_img = compose_rows(rows_of(tiles_over))
    stamp(raw_img, ["check_raw — 同帧未绘制版(与 check_overlay 逐像素差 = 残差点)"], size=22)
    stamp(over_img, ["check_overlay — 同帧残差 overlay(绿<0.05m / 黄<0.15m / 红其余)"], size=22)
    return raw_img, over_img, stats


# ---------------------------------------------------------------- rig 构建 / 采集


def build_rig(
    world: carla.World, ego: carla.Vehicle, rig: str
) -> dict[str, tuple[carla.Sensor, queue.Queue]]:
    """按 rig 口径挂 6 路 RGB + 实例分割(**rig 可变**,故不复用 `probe_calib.SensorRig`)。"""
    mounts, rots = rig_spec(rig)
    sensors: dict[str, tuple[carla.Sensor, queue.Queue]] = {}
    for kind in ("rgb", "instance_segmentation"):
        for name in NUS_CAMERA_RIG:
            bp = world.get_blueprint_library().find(f"sensor.camera.{kind}")
            for k, v in CAM_ATTRS.items():
                bp.set_attribute(k, v)
            pitch, yaw, roll = rots[name]
            x, y, z = mounts[name]
            s = cast(
                carla.Sensor,
                world.spawn_actor(
                    bp,
                    carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)),
                    attach_to=ego,
                ),
            )
            q: queue.Queue = queue.Queue()
            s.listen(q.put)
            sensors[f"{kind}:{name}"] = (s, q)
    return sensors


def capture_rig(
    world: carla.World, sensors: dict[str, tuple[carla.Sensor, queue.Queue]], warmup: int = 4
) -> dict[str, carla.Image]:
    for _ in range(warmup):
        world.tick()
        for pair in sensors.values():
            pair[1].get(timeout=10)
    world.tick()
    return {k: pair[1].get(timeout=10) for k, pair in sensors.items()}


def destroy_rig(world: carla.World, sensors: dict[str, tuple[carla.Sensor, queue.Queue]]) -> None:
    for pair in sensors.values():
        pair[0].stop()
        pair[0].destroy()
    for _ in range(2):
        world.tick()


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser(description="标定修正的人工复核图(raw/overlay 同帧对 + rig A/B + 几何裁决)")
    ap.add_argument("--out", default="outputs/calib_check")
    ap.add_argument("--edge-radius-px", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--sim-port", type=int, default=2000)
    ap.add_argument()
    args = ap.parse_args()

    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 图 3 是纯值件:先落盘,后面 CARLA 段即使失败也已交付几何裁决
    rep_path = out / "report.json"
    report = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else None
    geo = sheet_geometry(report)
    geo.save(out / "check_geometry.png")
    print(f"[viz] check_geometry.png {geo.width}×{geo.height}")

    client = carla.Client(args.host, args.sim_port)
    client.set_timeout(60.0)
    world = client.get_world()
    sync_mode(world)
    n = clean_world(world)
    print(f"[setup] 清场 {n} 个 actor;map = {world.get_map().name}")
    ego = spawn_ego(world)
    ego.set_autopilot(False)
    for _ in range(10):
        ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        world.tick()
    ego_t = ego.get_transform()
    print(f"[setup] ego @ {loc(ego_t)} yaw={ego_t.rotation.yaw:.3f}°")

    summary: dict[str, Any] = {"map": world.get_map().name, "ego_location": list(loc(ego_t))}

    # ---- 图 1:同帧 raw / overlay ----
    rig = SensorRig(world, ego, ("rgb", "depth"))
    rig.add_lidar(world, ego)
    raw_img, over_img, a3 = pass_residual_pair(rig, args)
    rig.destroy()
    raw_img.save(out / "check_raw.png")
    over_img.save(out / "check_overlay.png")
    summary["a3"] = a3
    print(f"[viz] check_raw.png / check_overlay.png {raw_img.width}×{raw_img.height}")

    # ⚠️ **图 2「rig A/B」已于 2026-09-28 删除**:它拿 `nuscenes` 与已移除的 `legacy` 并排,
    # 是"镜像 bug 已修"的目视证据。证据已冻结在 docs/legacy-rig-archive.md
    # (图 assets/legacy-rig/mirror-ab-views.png),不再由本脚本产出。

    ego.destroy()
    world.apply_settings(carla.WorldSettings())
    with open(out / "viz_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f"\n[done] {out}/check_{{geometry,raw,overlay}}.png + viz_summary.json")
    print("[done] 服务器已恢复异步")
    return 0


if __name__ == "__main__":
    sys.exit(main())

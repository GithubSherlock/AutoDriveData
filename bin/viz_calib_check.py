#!/usr/bin/env python3
"""标定修正的人工复核图:三张**把判据数字烧进画面**的拼图(自证,不是装饰)。

`bin/probe_calib.py` 的结论是纯数值的(A0–A6 → `report.json`),它只落一张
`overlay.png`。本脚本补齐"人能对着看"的三张,每张都携带**它自己编码的数字**,
故看图 ≈ 读判据:

| 图 | 编码什么 | 怎么看 |
|---|---|---|
| `check_raw.png` / `check_overlay.png` | **同一帧**的 raw 与残差 overlay | 绿点铺满路面/墙面且几乎无红点 = 外参与内参自洽;两张逐像素差 = **去重后的像素数**(≤ 采样点数,差 = 两点落到同一像素的碰撞数) |
| `check_rig_ab.png` | 同一 ego、同一批锥体,`nuscenes`(修正后)vs `legacy`(历史镜像) | **左方那个锥出现在哪一路**:修正后必是 `*_LEFT`,历史版跑到 `*_RIGHT` —— 镜像一眼可见 |
| `check_geometry.png` | 官方方位角 → 修正 yaw 的极坐标图 + 历史字面值对比 + 像素约定读数 | 实线(修正)与淡线(历史)在四个侧/后相机上张开 = 镜像;前/后相机两线重合 = 它长期没暴露的原因 |

格式沿用 §P-L.6 的约定:一律 `live_common.compose_rows` **原生像素**拼图,不缩放、不裁剪,
标签画在格**内**(不额外占画布高度)。

用法(需 CARLA 服务器):
  bash tools/carla_server.sh start
  PYTHONPATH=$PWD python bin/viz_calib_check.py
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
from carla_common import CAM_ATTRS, loc, spawn_ego, sync_mode
from live_common import RIG_LEGACY, RIG_NUSCENES, compose_rows, image_to_pil, rig_spec
from PIL import Image, ImageDraw
from probe_calib import (
    WORLD_DIRS,
    H,
    SensorRig,
    W,
    clean_world,
    decode_instance,
    depth_residuals,
    draw_residuals,
    lidar_world_points,
    place_cone_along,
    world_planes,
)

from autodrivedata import fonts
from autodrivedata.camera_rig import NUS_CAMERA_CALIBS, NUS_CAMERA_RIG
from autodrivedata.depth_codec import decode_depth
from autodrivedata.geometry import quat_normalize, quat_to_matrix
from autodrivedata.paths import project_path

# 历史字面值(修前的 `bin/collect_surround.py:SURROUND_CAMS`,已随本次修正删除)——
# 官方方位角被**原样抄成正数**,漏了 `yaw_carla = −az_nus`。列在此处只为让复核图能并排显示
# "当时写的"与"应该写的";真值一律来自 `autodrivedata/camera_rig.py`。
HISTORICAL_YAW: dict[str, float] = {
    "CAM_FRONT": 0.0,
    "CAM_FRONT_LEFT": 55.0,
    "CAM_FRONT_RIGHT": -55.0,
    "CAM_BACK": 180.0,
    "CAM_BACK_LEFT": 108.6,
    "CAM_BACK_RIGHT": -110.8,
}

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
    """官方方位角 → 修正 yaw 的极坐标图 + 历史字面值对比 + 像素约定读数。"""
    cw, ch = 1600, 900
    img = Image.new("RGB", (cw, ch), (18, 18, 22))
    d = ImageDraw.Draw(img)
    stamp(
        img,
        [
            "check_geometry — 环视 rig 几何裁决(全部数字由 autodrivedata/camera_rig.py 现算)",
            "实线 = 修正后(camera_rig.NUS_CAMERA_RIG)  淡线 = 历史字面值(修前 collect_surround.SURROUND_CAMS)",
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
        # 历史字面值:淡线(同一个名字当时指向哪)
        hx, hy = _dir_px(-HISTORICAL_YAW[name], r * 0.94)  # 历史值当 yaw_carla 用 → az = −yaw
        d.line([cx, cy, cx + hx, cy + hy], fill=tuple(int(v * 0.45) for v in col), width=2)
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
        "镜像判据:历史值 = 官方方位角原样抄成正数(漏 yaw_carla = −az_nus)",
        size=tsize,
        fill=(230, 230, 230),
    )
    heads = ["相机", "官方 az_nus", "修正 yaw", "历史字面", "原始差(不 wrap)"]
    aligns = ["l", "r", "r", "r", "r"]  # 名称左对齐,数值右对齐(按各自列宽推右缘)
    rows: list[tuple[list[str], str, float]] = []
    for name in NUS_CAMERA_RIG:
        yaw = NUS_CAMERA_RIG[name][1][1]
        hist = HISTORICAL_YAW[name]
        raw = abs(hist - yaw)  # **不 wrap**:217.2° 折成 142.8° 就看不出"镜像"了
        rows.append(([name, f"{_az_nus(name):+.3f}", f"{yaw:+.3f}", f"{hist:+.1f}", f"{raw:.1f}"], name, raw))
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
        lines = ["(未找到 report.json —— 先跑 bin/probe_calib.py 才有实测读数)"]
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


# ---------------------------------------------------------------- 图 2:legacy vs nuscenes


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


def pass_rig_ab(world: carla.World, ego: carla.Vehicle) -> tuple[Image.Image, dict[str, Any]]:
    """同一 ego、同一批锥体:两代 rig 各拍一遍 → 锥落在哪一路 = 镜像的直接证据。"""
    ego_loc = loc(ego.get_transform())
    cones: list[tuple[str, carla.Actor]] = []
    for direction, dv in WORLD_DIRS.items():
        actor = place_cone_along(world, ego_loc, dv)[0]
        if actor is None:
            print(f"  ✗ {direction} 锥摆不进去(碰撞),该方向跳过")
            continue
        cones.append((direction, actor))
    print(f"[rig A/B] 锥体就位 {len(cones)}/{len(WORLD_DIRS)}:{[c[0] for c in cones]}")

    result: dict[str, Any] = {"cone_dirs": [c[0] for c in cones]}
    for rig in (RIG_NUSCENES, RIG_LEGACY):
        sensors = build_rig(world, ego, rig)
        frames = capture_rig(world, sensors)
        yaws = {
            k.split(":", 1)[1]: sensors[k][0].get_transform().rotation.yaw
            for k in sensors
            if k.startswith("rgb:")
        }
        inst = {
            n: decode_instance(frames[f"instance_segmentation:{n}"].raw_data, H, W) for n in NUS_CAMERA_RIG
        }
        rgb = {n: image_to_pil(frames[f"rgb:{n}"]) for n in NUS_CAMERA_RIG}
        seen = {n: {dd: int((inst[n] == a.id).sum()) for dd, a in cones} for n in NUS_CAMERA_RIG}
        result[rig] = {"mounted_yaw_deg": yaws, "cone_px": seen, "images": rgb}
        print(f"\n[rig A/B] rig={rig}(实挂 yaw 从 CARLA 读回,已 tick)")
        for n in NUS_CAMERA_RIG:
            hits = {dd: px for dd, px in seen[n].items() if px >= 8}
            print(f"  {n:<17} 实挂 yaw {yaws[n]:>+8.3f}°  看见锥 {hits if hits else '(无)'}")

    destroy_rig(world, sensors)
    for a in (c[1] for c in cones):
        a.destroy()

    # 拼图:每格 = nuscenes 在上 / legacy 在下(同名字、同锥、同 ego)
    rows: list[list[tuple[str, Image.Image]]] = []
    row: list[tuple[str, Image.Image]] = []
    for name in NUS_CAMERA_RIG:
        nu = result[RIG_NUSCENES]["images"][name]
        lg = result[RIG_LEGACY]["images"][name]

        def cone_txt(rig: str, cam: str = name) -> str:
            hits = {dd: px for dd, px in result[rig]["cone_px"][cam].items() if px >= 8}
            return "锥 " + (", ".join(f"{dd}:{px}px" for dd, px in hits.items()) if hits else "无")

        stamp(
            nu,
            [
                f"{name}  nuscenes(修正后)  实挂 yaw {result[RIG_NUSCENES]['mounted_yaw_deg'][name]:+.2f}°",
                cone_txt(RIG_NUSCENES),
            ],
            color=(180, 255, 180),
            size=20,
        )
        stamp(
            lg,
            [
                f"{name}  legacy(历史镜像)  实挂 yaw {result[RIG_LEGACY]['mounted_yaw_deg'][name]:+.2f}°",
                cone_txt(RIG_LEGACY),
            ],
            color=(255, 170, 170),
            size=20,
        )
        border(nu, (0, 200, 0), 3)
        border(lg, (220, 0, 0), 3)
        tile = Image.new("RGB", (nu.width, nu.height + lg.height), (0, 0, 0))
        tile.paste(nu, (0, 0))
        tile.paste(lg, (0, nu.height))
        row.append((name, tile))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    img = compose_rows(rows)
    stamp(
        img,
        [
            "check_rig_ab — 同一 ego、同一批锥体,只变 rig 口径(上绿框 = nuscenes 修正后 / 下红框 = legacy 历史镜像)",
            "判据:世界的**左**方锥只该出现在 *_LEFT;历史镜像版把它放进 *_RIGHT(反之亦然)",
        ],
        size=24,
    )
    del result[RIG_NUSCENES]["images"], result[RIG_LEGACY]["images"]  # 图已拼进画布,别留在报告里
    return img, result


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser(description="标定修正的人工复核图(raw/overlay 同帧对 + rig A/B + 几何裁决)")
    ap.add_argument("--out", default="outputs/calib_check")
    ap.add_argument("--edge-radius-px", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--sim-port", type=int, default=2000)
    ap.add_argument(
        "--skip-rig-ab", action="store_true", help="跳过 legacy/nuscenes A/B(省一次 12 相机 spawn)"
    )
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

    # ---- 图 2:rig A/B ----
    if not args.skip_rig_ab:
        ab_img, ab = pass_rig_ab(world, ego)
        ab_img.save(out / "check_rig_ab.png")
        summary["rig_ab"] = ab
        print(f"[viz] check_rig_ab.png {ab_img.width}×{ab_img.height}")

    ego.destroy()
    world.apply_settings(carla.WorldSettings())
    with open(out / "viz_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f"\n[done] {out}/check_{{geometry,raw,overlay,rig_ab}}.png + viz_summary.json")
    print("[done] 服务器已恢复异步")
    return 0


if __name__ == "__main__":
    sys.exit(main())

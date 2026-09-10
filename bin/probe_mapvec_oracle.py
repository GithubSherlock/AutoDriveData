"""A6 验收探针:离线 xodr 解析 vs CARLA 运行时 `get_waypoint_xodr` 几何对账。

抽样 driving 车道 (road, lane_id, s) 随机组,对比:
- 离线:opendrive.lane_centerline_t + road_to_xy / road_heading
- 运行时:map.get_waypoint_xodr(road, lane_id, s).transform

验收口径(§5.11 A6):3D 位置误差 < 5cm。yaw 只打印诊断(个别 road 被 CARLA
导入器按路网拓扑翻转行驶方向,且 MapTR 矢量 GT 是无向几何,不定 yaw 口径)。
此脚本是 A6 验收三件套之一(几何自证 / API oracle / overlay 目检),无落盘产物。

用法:python bin/probe_mapvec_oracle.py --map Town10HD_Opt --samples 200
"""

from __future__ import annotations

import argparse
import glob
import math
import random

import carla

from autodrivedata.opendrive import (
    Road,
    lane_boundary_t,
    lane_centerline_t,
    parse_xodr,
    road_heading,
    road_to_xy,
)

XODR_GLOB = "/root/autodl-tmp/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps/**/*.xodr"


def find_xodr(name: str) -> str:
    hits = [p for p in glob.glob(f"{XODR_GLOB[:-5]}{name}.xodr", recursive=True)]
    if not hits:
        raise SystemExit(f"找不到 {name}.xodr({XODR_GLOB})")
    return hits[0]


def sample_lanes(road: Road, rng: random.Random, k: int) -> list[tuple[int, float]]:
    """road 内随机 (lane_id, s):只抽有宽度的 driving 车道。"""
    pool: list[tuple[int, float, float]] = []  # (lane_id, s_lo, s_hi)
    for i, sec in enumerate(road.lane_sections):
        next_s = road.lane_sections[i + 1].s if i + 1 < len(road.lane_sections) else road.length
        for lane in sec.left + sec.right:
            if lane.type != "driving":
                continue
            try:
                lo, hi = lane_boundary_t(road, sec.s + 0.1, lane.id)
            except ValueError:
                continue
            if hi - lo > 1e-6 and next_s - sec.s > 1.0:
                pool.append((lane.id, sec.s, next_s))
    if not pool:
        return []
    out: list[tuple[int, float]] = []
    for _ in range(k):
        lid, s_lo, s_hi = rng.choice(pool)
        out.append((lid, s_lo + rng.random() * (s_hi - s_lo)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="Town10HD_Opt")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--tol-cm", type=float, default=5.0)
    ap.add_argument("--yaw-tol-deg", type=float, default=0.5)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    xodr = parse_xodr(find_xodr(args.map))
    client = carla.Client(args.host, args.port)
    client.set_timeout(20)
    world = client.get_world()
    cmap = world.get_map()
    rng = random.Random(0)

    candidates = [r for r in xodr.roads.values() if r.junction == -1]
    pos_errs: list[float] = []
    yaw_errs: list[float] = []
    checked = 0
    per_road = max(1, args.samples // max(1, len(candidates)))
    for road in candidates:
        for lid, s in sample_lanes(road, rng, per_road):
            t = lane_centerline_t(road, s, lid)
            x, y, z = road_to_xy(road, s, t)
            y = -y  # CARLA 世界 = xodr 的 y 取反(Unreal 左手系;2026-09-11 oracle 实测标定)
            h = road_heading(road, s)  # yaw 数值与 xodr hdg 一致(镜像 + 左手旋转约定抵消)
            if lid < 0:
                h += math.pi  # 右车道沿 -s 行驶(OpenDRIVE 标准)
            wp = cmap.get_waypoint_xodr(road.id, lid, s)
            if wp is None:
                continue
            loc = wp.transform.location
            dx, dy, dz = loc.x - x, loc.y - y, loc.z - z
            pos_errs.append(math.sqrt(dx * dx + dy * dy + dz * dz))
            yaw = math.radians(wp.transform.rotation.yaw)
            dh = abs(math.atan2(math.sin(h - yaw), math.cos(h - yaw)))
            yaw_errs.append(math.degrees(dh))
            checked += 1

    if not pos_errs:
        raise SystemExit("无有效抽样(地图未加载该图?)")
    tol = args.tol_cm / 100.0
    bad = sum(1 for e in pos_errs if e > tol)
    pos_errs.sort()
    print(f"对账 {args.map}:{checked} 组抽样")
    print(
        f"3D 位置误差: mean={sum(pos_errs) / len(pos_errs) * 100:.2f}cm "
        f"p50={pos_errs[len(pos_errs) // 2] * 100:.2f}cm p95={pos_errs[int(len(pos_errs) * 0.95)] * 100:.2f}cm "
        f"max={pos_errs[-1] * 100:.2f}cm | 超 {tol * 100:.0f}cm: {bad}"
    )
    # yaw 仅诊断:个别 road 的行驶方向被 CARLA 导入器按路网拓扑翻转(实测 road 1
    # 正负 lane 与 +s 的关系整体相反),MapTR 矢量 GT 是无向几何,不定 yaw 验收口径
    print(f"yaw(诊断,非验收): mean={sum(yaw_errs) / len(yaw_errs):.1f}° max={max(yaw_errs):.1f}°")
    if bad:
        raise SystemExit(f"FAIL:{bad} 组超位置容差")
    print("PASS:几何对账在容差内")


if __name__ == "__main__":
    main()

"""L3 物理合理性探针:对照真实大陆 ars408 规格验证 CARLA 雷达点分布(一次性)。

需 CARLA 服务器 + 专用 carla 用户(tools/carla_server.sh)。用法:
    PYTHONPATH=$PWD/bin python bin/probe_radar_l3.py [--channels 1] [--frames 10]

对照表(官方大陆 ars408 规格):
    水平 FOV 77°(±38.5°)、垂直 FOV 14.2°(±7.1°)、range 250m、~3300 pps。
CARLA 0.9.16 两 FOV 属性交叉使用(bin/collect_nus.py RADAR_ATTRS 已对调):
    设 horizontal_fov=14.2 / vertical_fov=77 → 实际 azi±38.1° / alt±7.0°
    (12 组属性扫描自洽,详见 Plan.md §radar)。

统计项:L3 合理性五维——水平/垂直锥角、帧点数、深度分布、地面/天空占比、
前方已知目标命中(与"换向前垂直±38°塞满天空地面"对照)。输出对照表+判定。
帧点数用阻塞取样(与 collect_nus.py 一致,~300/帧);野值(相位回卷 ±1e26° 垃圾)
显式剥离,仅诊断打印不参与判定。
"""

from __future__ import annotations

import argparse
import queue
from typing import cast

import carla
import numpy as np

from autodrivedata.radar import detections_to_nus18, mask_in_ars408_vfov, mask_radar_points

# ars408 规格(工业参照,用于对照表与判定)
SPEC = {
    "azi_half_deg": 77.0 / 2,  # 水平 FOV 77°,半角 38.5°
    "alt_half_deg": 14.2 / 2,  # 垂直 FOV 14.2°,半角 7.1°
    "range_m": 250.0,
    "pps": 3300,
}

# 换向后的 CARLA 属性(与 bin/collect_nus.py RADAR_ATTRS 一致)
RADAR_ATTRS = {
    "horizontal_fov": "14.2",
    "vertical_fov": "77",
    "range": "250",
    "points_per_second": "3300",
    "sensor_tick": "0.1",
}


def _latest(q: queue.Queue):
    """取队列最新一帧(非阻塞,同步 tick 相位偶发空)。"""
    f = None
    while True:
        try:
            f = q.get_nowait()
        except queue.Empty:
            return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    from carla_common import spawn_ego, spawn_npcs, sync_mode

    sync_mode(world)
    ego = spawn_ego(world)
    t0 = ego.get_transform()
    spawn_npcs(world, t0)
    world.tick()

    bp = world.get_blueprint_library().find("sensor.other.radar")
    for k, v in RADAR_ATTRS.items():
        bp.set_attribute(k, v)
    # 前雷达只挂一个(与 collect_nus 的 RADAR_FRONT 同挂点)
    r = cast(
        carla.Sensor, world.spawn_actor(bp, carla.Transform(carla.Location(3.412, 0, 0.5)), attach_to=ego)
    )
    q: queue.Queue = queue.Queue()
    r.listen(q.put)

    # 预热(相位收敛)
    for _ in range(8):
        world.tick()
        _latest(q)

    frames = []
    for _ in range(args.frames):
        world.tick()
        f = q.get(timeout=5)  # 阻塞取(与 collect_nus.py 真实采样一致)
        frames.append(np.frombuffer(f.raw_data, dtype=np.float32).reshape(-1, 4))
    r.destroy()

    if len(frames) == 0:
        print("[fail] 采样 0 帧")
        return 1

    # 转 nus 18 字段(ego 系,前雷达 yaw=0 → R=I)
    all18 = np.vstack([detections_to_nus18(d) for d in frames])
    n = len(all18)

    # —— L3 五维统计 ——
    # 1) 锥角:从原始 CARLA 检测(alt/azi 弧度)读,更直接
    raws = np.vstack(frames)  # (N,4): [vel, altitude, azimuth, depth]
    alt = np.degrees(raws[:, 1])
    azi = np.degrees(raws[:, 2])
    dep = raws[:, 3]
    # **CARLA 0.9.16 相位回卷野值**:每帧混入 2~3 个 alt/azi 为 ±1e26° 量级的
    # 数值垃圾(不是真实目标)。它们破坏锥角/占比统计。显式剥离:
    #   野值 = |alt|>90 或 |azi|>90 或非有限数
    bad = ~np.isfinite(raws[:, 1:3]).all(axis=1) | (np.abs(alt) > 90.0) | (np.abs(azi) > 90.0)
    n_wild = int(bad.sum())
    azi_ok = azi[~bad]
    alt_ok = alt[~bad]
    dep_ok = dep[~bad]
    # 实测半角用**极值**(干净的物理布点边界;99.5 分位会因稀疏布点系统性偏低,
    # 曾把 ±38.1° 测成 ±36.5° 而误报——探针要量"边界在哪",不是"中间密度在哪")
    azi_peak = max(abs(azi_ok.min()), abs(azi_ok.max())) if len(azi_ok) else 0.0
    alt_peak = max(abs(alt_ok.min()), abs(alt_ok.max())) if len(alt_ok) else 0.0

    # 诊断:物理锥角外的"干净"点(没被 |alt|>90 抓到但超出 ars408 锥)。看它们是什么
    # (深度/速度分布),决定水平方向要不要也裁。仅诊断,不参与判定。
    out_azi = np.abs(azi_ok) > SPEC["azi_half_deg"]
    out_alt = np.abs(alt_ok) > SPEC["alt_half_deg"]
    n_out_azi = int(out_azi.sum())
    n_out_alt = int(out_alt.sum())
    out_azi_dep = dep_ok[out_azi]
    out_alt_dep = dep_ok[out_alt]
    print(
        f"[diag] 锥外点 |azi|>{SPEC['azi_half_deg']:.1f}: {n_out_azi} 个"
        f"(depth p50={np.percentile(out_azi_dep, 50) if len(out_azi_dep) else 0:.0f}m,"
        f" max={out_azi_dep.max() if len(out_azi_dep) else 0:.0f}m);"
        f" |alt|>{SPEC['alt_half_deg']:.1f}: {n_out_alt} 个"
        f"(depth max={out_alt_dep.max() if len(out_alt_dep) else 0:.0f}m)"
    )

    # 2) 每帧点数 + 野值数诊断
    n_per_frame = [len(d) for d in frames]
    # 帧点数含野值(drain 或 90° 野值),真实点数以锥内计
    alt_rad_all = np.abs(raws[:, 1])
    n_cone = int((alt_rad_all <= np.radians(SPEC["alt_half_deg"])).sum())

    # 2b) 采集管线口径:与 bin/collect_nus.py 一致的组合过滤
    #     (devkit 默认过滤器 ∩ 垂直锥 → mask_radar_points),这才是会写进 pcd 的点。
    all18_collect = all18[mask_radar_points(all18)]
    n_collect = len(all18_collect)

    # 3) 深度分布(干净数据)
    dep_p = np.percentile(dep_ok, [5, 50, 95])

    # 4) 地面/天空占比:垂直 ±7.1° 锥外 = 物理无效点。判据必须**随深度缩放**
    #    (z 有效界 = tan(7.1°) × depth):固定阈值会把近处真实目标误判成无效。
    #    反体素化用 raw CARLA 检测的 alt/depth 直接判,最准。野值(相位回卷垃圾)
    #    不计入(它们 |alt|>90 必然在锥外,但不应算"真实目标的无效点")。
    alt_rad = np.abs(raws[:, 1])[~bad]  # 剔除野值后的 |alt|
    valid_v = alt_rad <= np.radians(SPEC["alt_half_deg"])  # 垂直锥内
    vfov_bad = float((~valid_v).mean())
    # 交叉验证:18 字段判据(|z| <= sin(7.1°)·depth,采集管线用)在非野值点上应与
    # raw alt 判据逐点一致——一致率 ~1.0 = 锥角口径自洽(反体素化没引入偏差)。
    cone18 = mask_in_ars408_vfov(all18)[~bad]
    agree_cone = float((cone18 == valid_v).mean())

    # 5) 前方已知目标(12m 前车,spawn_nus 固定):8–16m × |y|<1.1 × |z|<nus_z(锥内)
    pe = all18_collect[:, :3]  # 用采集管线过滤后的点(这才是真实 pcd 的点)
    near = (pe[:, 0] > 8) & (pe[:, 0] < 16)
    # 目标框垂直界 = 目标高度 ±1.25m(车高 ~1.6m),不再用"垂直锥"
    box = near & (np.abs(pe[:, 1]) < 1.1) & (np.abs(pe[:, 2]) < 1.25)
    target_hits = int(box.sum())

    # —— 输出对照表 ——
    print(
        f"\nL3 对照表(ars408 规格 vs CARLA 实测,{len(frames)} 帧,每帧 {n // len(frames)} 点,锥内 {n_cone} 点)"
    )
    # 帧点数判定:CARLA 内部每 tick 的射线预算不是 3300pps/10Hz 的精确除法,
    # 实测在两档间抖动:满额 330/帧,或 ~265/帧(服务器负载/场景状态相关)。
    # 判据只拦"异常稀疏"(<200, 配置错误)与"超量"(>340, 物理不可能),
    # 不拦两档正常预算。ars408 标称 330 仅作参照。
    n_frames_mean = n // len(frames)
    frames_ok = 200 <= n_frames_mean <= 340
    rows = [
        (
            "水平半角(°)",
            f"±{SPEC['azi_half_deg']:.1f}",
            f"±{azi_peak:.1f}",
            abs(azi_peak - SPEC["azi_half_deg"]) <= 1.5,
        ),
        (
            "垂直半角(°)",
            f"±{SPEC['alt_half_deg']:.1f}",
            f"±{alt_peak:.1f}",
            abs(alt_peak - SPEC["alt_half_deg"]) <= 0.5,
        ),
        ("range(m)", "250", f"≤{dep.max():.0f}", dep.max() <= 250),
        ("帧点数", "两档 330/265", str(n_frames_mean), frames_ok),
        ("深度 p5/p50/p95(m)", "-", f"{dep_p[0]:.0f}/{dep_p[1]:.0f}/{dep_p[2]:.0f}", dep_p[2] <= 250),
        ("地面/天空占比", "≈0", f"{vfov_bad:.2%}", vfov_bad < 0.05),
        ("前方 12m 车框内点", ">0", str(target_hits), target_hits > 0),
    ]
    print(f"{'指标':<20}{'ars408':<12}{'实测':<16}{'通过'}")
    ok = True
    for name, spec, meas, passed in rows:
        print(f"{name:<20}{spec:<12}{meas:<16}{'✓' if passed else '✗'}")
        ok &= passed

    # 逐帧点数列 + 诊断(干净数据极值;野值数仅诊断,不参与判定)
    print(f"\n每帧点数 {n_per_frame}(锥内 {n_cone} 点,采集管线 {n_collect} 点,野值 {n_wild} 个)")
    print(f"18 字段锥角判据 vs raw alt 判据一致率 {agree_cone:.2%}(~1.0 = 口径自洽)")
    print(f"干净 azi 极值 ±{azi_peak:.1f}° / alt 极值 ±{alt_peak:.1f}°")

    print(f"\n{'=== L3 合理性 ' + ('通过 ✓' if ok else '未达标 ✗')} ===")
    if ok:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

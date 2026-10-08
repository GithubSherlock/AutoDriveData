"""**共轴相机 + 雷达**:同一个挂点、同一个朝向,摆一个已知方位的目标,两边各看一次。

## 它补的是 Plan4「雷达不进融合的消融表」那句的**前置**

Plan4 记的下一步原文:「**相机与雷达同挂点共轴**、摆一个已知方位的目标,直接量
(现数据里相机 FoV 够不着要判别的那几路)」。

**D0 探针(2026-10-07)把"够不着"量出来了** —— 官方 rig 里**五个雷达没有一个与任何相机共轴**:

| 雷达 ↔ 最近相机 | 距离 |
|---|---|
| `RADAR_FRONT` ↔ `CAM_FRONT` | **1.988 m**(雷达前 1.711 m、低 1.011 m) |
| `RADAR_BACK_LEFT` ↔ `CAM_BACK` | 1.356 m |

⇒ 一个 5–10 m 处的目标在这两个挂点上的**方位角本来就不同** —— 拿现数据去问
"雷达在那个方位给不给回波",相机那一路**根本没在看同一个方向**。

## 本探针做什么

**不挂任何父 actor**、两台传感器用**同一个 `carla.Transform`**(世界坐标直接给,不走父系解释 ——
那个坑 2026-10-07 刚在 `collect_3dgs` 上修过一次)。然后按方位扫目标:

| 每条方位读数 | 怎么量 |
|---|---|
| **相机看得见吗** | 语义图里 `Vehicles` tag 的像素数 |
| **同方位雷达给回波吗** | 雷达点里,方位角落在目标 ±`--bearing-tol` 内、且斜距落在 ±`--range-tol` 的点数 |

★ **判据不是"给不给",是"两边一不一致"** —— 这正是"雷达能不能做目标级融合"那个问题本身。
⚠️ 雷达是**随机撒射线**,单帧杂波逐帧不同 ⇒ 每个方位**多帧取中位**,并同时报 `--no-targets`
那一遍(空场)当本底。**不报本底就没法判。**

用法(需 CARLA):
  python -m autodrivedata.sim.probe_radar_camera_coaxial [--bearings -20,-10,0,10,20]
"""

from __future__ import annotations

import argparse
import queue
from typing import cast

import carla
import numpy as np

from autodrivedata.sim.carla_common import CAM_ATTRS, sync_mode
from autodrivedata.sim.collect_nus import RADAR_ATTRS
from autodrivedata.sim.collect_surround import tag_from_semantic_image
from autodrivedata.sim.probe_radar_on_targets import _drain
from autodrivedata.utils import runlog

#: 语义图里载具的 tag(`CityObjectLabel.Vehicles`)。取 14 —— 与 `sem_tags` 同口径。
TAG_VEHICLE = 14
#: 共轴挂点相对**基准位姿**的偏移(CARLA 车体系,米:前 2.0 / 横 0 / 高 1.5)。
#: **两台传感器逐字相同** —— 这就是"共轴"的全部内容。
MOUNT = (2.0, 0.0, 1.5)


def _parse(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def _bearing_of(points_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """雷达自身系 `(N,3)` → 方位角(度)与斜距(米)。CARLA 雷达系:x 前、y 右、z 上。"""
    if not len(points_local):
        return np.zeros(0), np.zeros(0)
    x, y, z = points_local[:, 0], points_local[:, 1], points_local[:, 2]
    bearing = np.degrees(np.arctan2(y, x))
    rng = np.sqrt(x * x + y * y + z * z)
    return bearing, rng


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--center-index", type=int, default=77, help="基准 spawn point(有路面、有朝向)")
    ap.add_argument("--bearings", default="-20,-10,0,10,20", help="目标的横向方位(度)")
    ap.add_argument("--range", type=float, default=8.0, help="目标距离(m)")
    ap.add_argument("--target", default="vehicle.tesla.model3")
    ap.add_argument("--frames", type=int, default=8, help="每个方位取几帧(取中位)")
    ap.add_argument("--bearing-tol", type=float, default=4.0, help="方位命中窗(度)")
    ap.add_argument("--range-tol", type=float, default=2.5, help="斜距命中窗(米)")
    ap.add_argument("--no-targets", action="store_true", help="空场基线 pass(不摆目标)")
    ap.add_argument("--out", default="")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    bearings = _parse(args.bearings)
    sub = "radar_coaxial_floor" if args.no_targets else "radar_coaxial"
    with runlog.run(f"autodrivedata.sim.probe_radar_camera_coaxial.{sub}") as rl:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        world = client.get_world()
        sync_mode(world)
        for a in world.get_actors():
            if a.type_id.startswith(("vehicle", "walker", "controller")):
                a.destroy()
        world.tick()
        lib = world.get_blueprint_library()

        # ★ 基准位姿取一个**合法 spawn point**(有路面、有朝向)。
        #   ⚠️ **不能拿 MOUNT 当世界坐标** —— 第一版就这么错了:`(2,0,1.5)` 落在图原点附近,
        #   目标 spawn 直接失败(位置非法/被占)。
        base = world.get_map().get_spawn_points()[args.center_index]
        bfwd, bright = base.get_forward_vector(), base.get_right_vector()
        mount_loc = base.location + bfwd * MOUNT[0] + carla.Location(0, 0, MOUNT[2])
        # ★ **共轴**:两台传感器同一个 `carla.Transform`、**不挂父 actor**
        #   (挂上去就走父系解释 —— 那个坑见 collect_rig.to_parent_frame 头注)
        tf = carla.Transform(mount_loc, base.rotation)
        rbp = lib.find("sensor.other.radar")
        for k, v in RADAR_ATTRS.items():
            rbp.set_attribute(k, v)
        radar = cast(carla.Sensor, world.spawn_actor(rbp, tf))
        sbp = lib.find("sensor.camera.semantic_segmentation")
        for k, v in CAM_ATTRS.items():
            sbp.set_attribute(k, v)
        sem = cast(carla.Sensor, world.spawn_actor(sbp, tf))
        rq: queue.Queue = queue.Queue()
        sq: queue.Queue = queue.Queue()
        radar.listen(rq.put)
        sem.listen(sq.put)
        print(
            f"[base] spawn {args.center_index} @ ({base.location.x:.2f},{base.location.y:.2f}) | "
            f"共轴偏移 {MOUNT} → 世界 ({mount_loc.x:.2f},{mount_loc.y:.2f},{mount_loc.z:.2f})"
            " —— 两台传感器**同一个 Transform**"
        )
        rl.highlight("mount", list(MOUNT))

        target = None
        try:
            # 预热:刚 spawn 的传感器头几帧不是它自己的视角(TAA + 投递延迟)
            for _ in range(6):
                world.tick()
                _drain(rq)
                _drain(sq)

            rows = []
            for b in bearings:
                rad = np.radians(b)
                loc = carla.Location(
                    x=mount_loc.x + bfwd.x * args.range * np.cos(rad) + bright.x * args.range * np.sin(rad),
                    y=mount_loc.y + bfwd.y * args.range * np.cos(rad) + bright.y * args.range * np.sin(rad),
                    z=base.location.z,
                )
                if not args.no_targets:
                    bp = lib.find(args.target)
                    target = world.try_spawn_actor(bp, carla.Transform(loc, carla.Rotation()))
                    if target is None:
                        # ⚠️ 不静默跳过 —— 那会把"没摆上"读成"雷达看不见"
                        raise SystemExit(f"{args.target} 在方位 {b}° 处 spawn 失败(位置被占?)")
                    world.tick()
                    # 让车落到地面(它在 z=1.5 会掉一小段)
                    for _ in range(3):
                        world.tick()

                r_n, s_n, dist = [], [], []
                for _ in range(args.frames):
                    world.tick()
                    rf = _drain(rq)
                    sf = _drain(sq)
                    if rf is not None:
                        pts = np.array(
                            [
                                [
                                    d.depth * np.cos(d.azimuth) * np.cos(d.altitude),
                                    d.depth * np.sin(d.azimuth) * np.cos(d.altitude),
                                    d.depth * np.sin(d.altitude),
                                ]
                                for d in rf
                            ],
                            dtype=float,
                        ).reshape(-1, 3)
                        bg, rg = _bearing_of(pts)
                        r_n.append(
                            int(
                                (
                                    (np.abs(bg - b) <= args.bearing_tol)
                                    & (np.abs(rg - args.range) <= args.range_tol)
                                ).sum()
                            )
                        )
                    if sf is not None:
                        s_n.append(int((tag_from_semantic_image(sf) == TAG_VEHICLE).sum()))
                    if target is not None:
                        dist.append(float(target.get_location().distance(mount_loc)))

                rows.append(
                    {
                        "bearing_deg": b,
                        "radar_hits_median": float(np.median(r_n)) if r_n else None,
                        "sem_vehicle_px_median": float(np.median(s_n)) if s_n else None,
                        "target_dist_m": round(float(np.median(dist)), 3) if dist else None,
                    }
                )
                r = rows[-1]
                print(
                    f"  [方位 {b:>+6.1f}°] 雷达命中 {r['radar_hits_median']} 点 / "
                    f"语义载具 {r['sem_vehicle_px_median']} px"
                    + (f"  目标实测距离 {r['target_dist_m']} m" if r["target_dist_m"] else "")
                )
                if target is not None:
                    target.destroy()
                    target = None
                    world.tick()

            print("\n=== ★ 两边一不一致 ===")
            for r in rows:
                cam = (r["sem_vehicle_px_median"] or 0) > 0
                rad_hit = (r["radar_hits_median"] or 0) > 0
                mark = "一致" if cam == rad_hit else "★ 不一致"
                print(
                    f"  方位 {r['bearing_deg']:>+6.1f}°: 相机{'见' if cam else '不见'} / "
                    f"雷达{'有回波' if rad_hit else '无回波'} ⇒ {mark}"
                )
                rl.highlight(f"bearing_{r['bearing_deg']:+.0f}", r["radar_hits_median"])
            rl.highlight("no_targets", args.no_targets)
            if args.out:
                import json
                from pathlib import Path

                Path(args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        finally:
            if target is not None and target.is_alive:
                target.destroy()
            for s in (radar, sem):
                s.stop()
                s.destroy()


if __name__ == "__main__":
    main()

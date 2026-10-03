"""雷达**到底打不打得到目标** —— 静止 / 运动 / 空位三臂同场,一次量完(需 CARLA)。

## 这条判据补的是雷达线上最后一个缺口

§P-V17 六量到:`kitti_ab_occl2_*` 那批数据里,雷达对四台**静止车**的归因回波是 **0**
(带阳性对照:同一条投影通路的 LiDAR 有 427/815 点)。但那是**一份数据、一类目标**,
推不出"雷达一律打不到目标"。缺的两件:

1. **空场对照** —— 更早那次探针报过"12 m 静止车 112 点",但那个数**没有对照**
   ("离目标中心 2.5 m 内的点数"会把**路面杂波**算进去),而补对照时 CARLA 客户端
   **两次硬崩**(`msgpack std::bad_cast`,崩在 `car.destroy()`)。**那 112 点至今没分清**
   是回波还是杂波。
2. **运动目标完全没测过。**

## 设计:一场戏三臂,不销毁任何东西

上次崩在 `destroy()`,所以这次**不销毁** —— 同场同时摆三臂,一次 tick 序列同时量:

| 臂 | 位置 | 作用 |
|---|---|---|
| `parked` | 前 20 m / 横 **−3.5 m** | 静止目标 |
| `moving` | 前 20 m / 横 **+3.5 m** | 以 5 m/s **直行前进**(bearing 基本不变,距离缓慢拉大) |
| `empty` | 前 20 m / 横 **0** | **空位对照** —— 同距离、同姿态,唯独没有东西 |

`empty` 那一臂就是"这个距离上的路面杂波有多少"的实测基线。**没有它,"20 个点"读不出任何东西。**

## 换算:**不用我们的任何表**

CARLA 原始检测 `(vel, alt, azi, depth)` 按 CARLA 自己的约定(azimuth 右转为正)还原到
传感器自身系,再用 `sensor.get_transform()` 送世界系。CARLA 自己的 API 就是真值 ——
绕开 §P-V17 那场关于偏航符号的争执(那场争执的结论是 `radar_ego` 曾经镜像)。

## 自证

- **LiDAR 对照**:两臂都要被 LiDAR 看见(否则夹具坏了,读数没有意义);
- **运动车真的在动**:逐帧读它的位移,不动就报错 —— 否则"运动臂"与"静止臂"是同一件事。

用法:
    python -m autodrivedata.sim.probe_radar_on_targets [--frames 12] [--speed 5]
"""

from __future__ import annotations

import argparse
import queue
from typing import cast

import carla
import numpy as np

from autodrivedata.calib.core import CameraIntrinsics, world_to_img
from autodrivedata.gt.export.nuscenes import NUS_RADAR_MOUNTS_CARLA
from autodrivedata.perception.sem_tags import TAG_NAMES
from autodrivedata.sim.carla_common import CAM_ATTRS, LIDAR_ATTRS, SENSOR_OFFSET, spawn_ego, sync_mode
from autodrivedata.sim.collect_nus import RADAR_ATTRS, RADAR_YAW_OFFSET
from autodrivedata.sim.collect_surround import tag_from_semantic_image
from autodrivedata.utils import geometry as g

#: 三臂的 (前向 m, 横向 m)。横向为 ego 右侧为正。
#: 三臂的 (前向 m, 横向 m)+ 蓝图。**`prop` 臂是"大物件"对照** ——
#: 100× 预算下车辆仍是 0、而杂波线性涨,所以要分出"车辆专属"还是"小物体一律不回波"。
#: `static.prop.warningconstruction` 是 2.6×2.1×1.86 m 的施工挡板(比车正面还大)。
ARMS: dict[str, tuple[float, float]] = {
    "parked": (20.0, -3.5),
    "moving": (20.0, 3.5),
    #: `prop` 挪到横向 +7(**位置对照**):若回波跟着道具走,说明那 9501 确实来自道具本身
    "prop": (20.0, 7.0),
    #: `walker` 在横向 0 —— 用来分「vehicle 专属」还是「动态 actor 一律不可见」
    "walker": (20.0, 0.0),
    #: 空位基线**必须与目标同距离** —— 挪到 28 m 会让净额变成跨距离比较(第一版就这么错了)
    "empty": (20.0, -7.0),
}
ARM_BP: dict[str, str] = {
    "parked": "vehicle.tesla.model3",
    "moving": "vehicle.tesla.model3",
    "prop": "static.prop.warningconstruction",
    "walker": "walker.pedestrian.0001",
}
#: 命中半径(m)。**约等于车宽** —— 再大就把隔壁车道也算进来了。
HIT_R = 2.5
#: LiDAR 对照的命中半径,同口径。
LIDAR_HIT_R = 2.5


def _drain(q: queue.Queue):
    """非阻塞取最新帧 —— 雷达有相位空 tick(实测 ~7%),阻塞 `get` 会死等。"""
    f = None
    while True:
        try:
            f = q.get_nowait()
        except queue.Empty:
            return f


def _to_world(tf: carla.Transform, pts: np.ndarray) -> np.ndarray:
    """传感器自身系 `(N,3)` → 世界系(CARLA 自己的位姿,不经任何换算表)。"""
    if not len(pts):
        return np.zeros((0, 3))
    out = [tf.transform(carla.Location(float(p[0]), float(p[1]), float(p[2]))) for p in pts]
    return np.array([[o.x, o.y, o.z] for o in out], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--speed", type=float, default=5.0, help="运动臂的速度(m/s)")
    ap.add_argument(
        "--no-targets",
        action="store_true",
        help="**只摆传感器、不摆任何目标** —— 空场基线 pass。与带目标的那次用**同样的槽位、"
        "同样的帧数**跑,两次相减才是目标的贡献。⚠️ 雷达是**随机撒射线**,单帧杂波逐帧不同,"
        "所以必须靠同帧数相减,不能靠「隔壁有个空位」—— 那条已经咬过两次",
    )
    ap.add_argument(
        "--pps-mult",
        type=float,
        default=1.0,
        help="**射线预算倍数**(乘在 `points_per_second` 上)。默认 1.0 = ARS408 规格。"
        "抬它用来分开「射线预算不够」与「CARLA 压根不为车辆产生回波」这两种解释 —— "
        "**抬过之后就**不再是 ARS408 规格**,读数只能当机制探针,不能当规格结论",
    )
    ap.add_argument(
        "--dump-shape",
        action="store_true",
        help="打印命中点**在雷达自身系里**的形态:逐帧点数、方位/仰角/深度直方图、世界包围盒。"
        "用来分「某个面在求交里被放大」与「被当成一个大反射体」",
    )
    ap.add_argument(
        "--ego-speed",
        type=float,
        default=0.0,
        help="**让 ego 以该速度直行**(m/s)。用来读命中点的**径向速度**:静止目标的径向速度"
        "必须等于自车速度在视线上的投影(`vel ≈ s·(k·u)`,`|k|≈1`)。"
        "这是唯一能证伪「这些点属于那个静止目标」的一步",
    )
    ap.add_argument(
        "--project",
        action="store_true",
        help="**把命中点投进语义相机**并统计它们落在哪个 tag 上 —— "
        "回答「那团点的世界里有什么」。全数值,不目检",
    )
    ap.add_argument(
        "--slot0-bp",
        default="walker.pedestrian.0001",
        help="**横向 0 那个槽位放什么**(默认 walker.pedestrian.0001)。"
        "存在的理由:那团点只在摆上行人时出现,而**横向 0 此前只放过 walker 这一种东西** —— "
        "「行人」与「正前方」两个自变量是混在一起的。换一台车放同一槽位即可分开",
    )
    ap.add_argument(
        "--depth-check",
        action="store_true",
        help="**用同载 LiDAR 当深度裁判**:对每个雷达点,在该方位上找 LiDAR 见到的**最前面**那个面,"
        "比两者的斜距。`d_radar ≈ d_lidar` ⇒ 点在表面上;`d_radar ≫ d_lidar` ⇒ **穿过**了物体。"
        "这是唯一能把「打在物体上」与「物体触发的一团别的东西」分开的测量",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    sync_mode(world)
    for a in world.get_actors():
        if a.type_id.startswith(("vehicle", "walker", "controller")):
            a.destroy()
    world.tick()

    ego = spawn_ego(world)
    et = ego.get_transform()
    fwd, right = et.get_forward_vector(), et.get_right_vector()
    lib = world.get_blueprint_library()

    def at(d: float, lat: float) -> carla.Location:
        v = et.location + fwd * d + right * lat
        return carla.Location(x=v.x, y=v.y, z=et.location.z + 0.2)

    # 传感器:雷达(RADAR_FRONT,已验朝向那一路)+ 64 线 LiDAR 作对照
    rbp = lib.find("sensor.other.radar")
    for k, v in RADAR_ATTRS.items():
        rbp.set_attribute(k, v)
    pps0 = float(RADAR_ATTRS["points_per_second"])
    rbp.set_attribute("points_per_second", str(int(pps0 * args.pps_mult)))
    print(
        f"[雷达参数] points_per_second {pps0:.0f} × {args.pps_mult:g} = "
        f"{pps0 * args.pps_mult:.0f}(sensor_tick {RADAR_ATTRS['sensor_tick']} s "
        f"⇒ 名义 {pps0 * args.pps_mult * float(RADAR_ATTRS['sensor_tick']):.0f} 点/帧)"
    )
    radar = cast(
        carla.Sensor,
        world.spawn_actor(
            rbp,
            carla.Transform(
                carla.Location(*NUS_RADAR_MOUNTS_CARLA["RADAR_FRONT"]),
                carla.Rotation(yaw=RADAR_YAW_OFFSET["RADAR_FRONT"]),
            ),
            attach_to=ego,
        ),
    )
    rq: queue.Queue = queue.Queue()
    radar.listen(rq.put)
    lbp = lib.find("sensor.lidar.ray_cast")
    for k, v in LIDAR_ATTRS.items():
        lbp.set_attribute(k, v)
    lidar = cast(
        carla.Sensor, world.spawn_actor(lbp, carla.Transform(carla.Location(1.2, 0, 1.65)), attach_to=ego)
    )
    lq: queue.Queue = queue.Queue()
    lidar.listen(lq.put)

    sem_cam = None
    sq: queue.Queue = queue.Queue()
    if args.project:
        sbp = lib.find("sensor.camera.semantic_segmentation")
        for k, v in CAM_ATTRS.items():
            sbp.set_attribute(k, v)
        sem_cam = cast(carla.Sensor, world.spawn_actor(sbp, SENSOR_OFFSET, attach_to=ego))
        sem_cam.listen(sq.put)
    kk = CameraIntrinsics(
        width=int(CAM_ATTRS["image_size_x"]),
        height=int(CAM_ATTRS["image_size_y"]),
        fov_h_deg=float(CAM_ATTRS["fov"]),
    )

    # 两臂各摆一台车(空位臂**什么都不摆**)
    cars: dict[str, carla.Actor] = {}
    for name in [] if args.no_targets else ARM_BP:
        d, lat = ARMS[name]
        bp_id = args.slot0_bp if name == "walker" else ARM_BP[name]
        a = world.try_spawn_actor(
            lib.find(bp_id),
            carla.Transform(at(d, lat), carla.Rotation(yaw=et.rotation.yaw)),
        )
        if a is None:
            raise SystemExit(f"{name} 臂 spawn 失败 @ {ARMS[name]} —— 夹具没摆成,不许报数")
        cars[name] = a
    world.tick()  # ★ 读回前必须 tick

    vel = carla.Vector3D(x=fwd.x * args.speed, y=fwd.y * args.speed, z=0.0)
    for _ in range(5):  # 预热:雷达相位收敛 + 传感器队列起势
        world.tick()
        _drain(rq)
        _drain(lq)

    print(f"[夹具] 三臂 @ {ARMS} | 运动臂 {args.speed} m/s | 命中半径 {HIT_R} m")
    print(
        f"{'帧':>3}{'静止臂 雷达':>12}{'运动臂 雷达':>12}{'空位 雷达':>11}"
        f"{'静止臂 LiDAR':>14}{'运动臂 LiDAR':>14}{'运动臂位移':>11}"
    )
    r_hits = {k: 0 for k in ARMS}
    shape: dict[str, list[np.ndarray]] = {k: [] for k in ARM_BP}
    vel_rec: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {k: [] for k in ARM_BP}
    seen_tags: dict[str, list[int]] = {k: [] for k in ARM_BP}
    depth_delta: dict[str, list[np.ndarray]] = {k: [] for k in ARM_BP}
    probe_bin: tuple[float, np.ndarray, np.ndarray] | None = None
    best_bin_n = 0
    probe_raw: np.ndarray | None = None
    all_az: list[np.ndarray] = []
    all_el: list[np.ndarray] = []
    frame_counts: dict[str, list[int]] = {k: [] for k in ARM_BP}
    l_hits = {k: 0 for k in ARM_BP}
    if args.no_targets:
        print("\n[空场 pass] 未摆任何目标 —— 逐臂读数**全部是杂波**,拿来与带目标那次相减")
    prev_moving = None
    moved = 0.0
    n_frames = 0

    for i in range(args.frames):
        if "moving" in cars:
            cars["moving"].set_target_velocity(vel)
        if args.ego_speed:
            ego.set_target_velocity(carla.Vector3D(x=fwd.x * args.ego_speed, y=fwd.y * args.ego_speed, z=0.0))
        world.tick()
        rf = _drain(rq)
        lf = _drain(lq)
        sf = _drain(sq) if sem_cam is not None else None
        if rf is None or lf is None or (args.project and sf is None):
            continue
        n_frames += 1

        a = np.frombuffer(rf.raw_data, np.float32).reshape(-1, 4)
        ok = np.isfinite(a[:, 1]) & (np.abs(np.degrees(a[:, 1])) < 90) & (np.abs(np.degrees(a[:, 2])) < 90)
        a = a[ok]
        # **传感器自身系**的坐标(形态诊断要看的就是它们)
        sensor_xyz = (
            np.stack(
                [
                    a[:, 3] * np.cos(a[:, 1]) * np.cos(a[:, 2]),
                    a[:, 3] * np.cos(a[:, 1]) * np.sin(a[:, 2]),
                    a[:, 3] * np.sin(a[:, 1]),
                ],
                1,
            )
            if len(a)
            else np.zeros((0, 3))
        )
        if len(sensor_xyz):
            all_az.append(np.degrees(np.arctan2(sensor_xyz[:, 1], sensor_xyz[:, 0])))
            all_el.append(
                np.degrees(np.arctan2(sensor_xyz[:, 2], np.hypot(sensor_xyz[:, 0], sensor_xyz[:, 1])))
            )
        pts = _to_world(radar.get_transform(), sensor_xyz)
        lp = np.frombuffer(lf.raw_data, np.float32).reshape(-1, 4)[:, :3]
        lw = _to_world(lidar.get_transform(), lp)

        row = {}
        for name in ARM_BP:
            if name not in cars:
                row[name] = (0, 0)
                continue
            c = cars[name].get_transform().location
            cc = np.array([c.x, c.y, c.z])
            hit = (np.linalg.norm(pts - cc, axis=1) < HIT_R) if len(pts) else np.zeros(0, bool)
            row[name] = (
                int(hit.sum()),
                int((np.linalg.norm(lw - cc, axis=1) < LIDAR_HIT_R).sum()) if len(lw) else 0,
            )
            r_hits[name] += row[name][0]
            l_hits[name] += row[name][1]
            frame_counts[name].append(row[name][0])
            if args.dump_shape and hit.any():
                shape[name].append(sensor_xyz[hit])
            if args.ego_speed and hit.any():
                vel_rec[name].append((sensor_xyz[hit], a[hit, 0].astype(float)))
            if args.depth_check and hit.any() and len(lw):
                # LiDAR 世界点 → 雷达自身系(用雷达自己的位姿求逆,不经任何表)
                rtf = radar.get_transform()
                rr = rtf.rotation
                rmat = g.carla_rotation_matrix(
                    (np.radians(rr.pitch), np.radians(rr.yaw), np.radians(rr.roll))
                )
                lr = (lw - np.array([rtf.location.x, rtf.location.y, rtf.location.z])) @ rmat
                dl = np.linalg.norm(lr, axis=1)
                la = np.degrees(np.arctan2(lr[:, 1], lr[:, 0]))
                le = np.degrees(np.arctan2(lr[:, 2], np.hypot(lr[:, 0], lr[:, 1])))
                # 0.5° 分箱,取每箱**最近**的那个面(即该方位上最前面的表面)
                ka = np.round(la / 0.5).astype(int)
                ke = np.round(le / 0.5).astype(int)
                key = ka * 10000 + ke
                order = np.argsort(dl)
                first: dict[int, float] = {}
                for i in order:
                    first.setdefault(int(key[i]), float(dl[i]))
                hq = sensor_xyz[hit]
                hd = np.linalg.norm(hq, axis=1)
                ha = np.degrees(np.arctan2(hq[:, 1], hq[:, 0]))
                he = np.degrees(np.arctan2(hq[:, 2], np.hypot(hq[:, 0], hq[:, 1])))
                hk = np.round(ha / 0.5).astype(int) * 10000 + np.round(he / 0.5).astype(int)
                near = np.array([first.get(int(k), np.nan) for k in hk])
                depth_delta[name].append(hd - near)
                # 单箱深潜:取**点数最多**的那个 0.05° 方位箱,把它整箱留下来
                kb = np.round(ha / 0.05).astype(int)
                raw_hit = a[hit]
                for b in np.unique(kb):
                    sel_r = hd[kb == b]
                    if len(sel_r) > best_bin_n:
                        best_bin_n = len(sel_r)
                        sel_l = dl[np.round(la / 0.05).astype(int) == b]
                        probe_bin = (b * 0.05, sel_r, sel_l)
                        probe_raw = raw_hit[kb == b]
            if sf is not None and hit.any():
                tag = tag_from_semantic_image(sf)
                ct = sem_cam.get_transform()
                cl, cr = (
                    (ct.location.x, ct.location.y, ct.location.z),
                    tuple(np.radians(v) for v in (ct.rotation.pitch, ct.rotation.yaw, ct.rotation.roll)),
                )
                for pt in pts[hit]:
                    uv = world_to_img((pt[0], pt[1], pt[2]), cl, cr, kk)
                    if uv is None:
                        continue
                    seen_tags[name].append(int(tag[int(uv[1]), int(uv[0])]))
        ce = np.array([at(*ARMS["empty"]).x, at(*ARMS["empty"]).y, at(*ARMS["empty"]).z])
        e_hit = int((np.linalg.norm(pts - ce, axis=1) < HIT_R).sum()) if len(pts) else 0
        r_hits["empty"] += e_hit

        if "moving" in cars:
            cm = cars["moving"].get_transform().location
            moved = (
                0.0
                if prev_moving is None
                else float(np.linalg.norm(np.array([cm.x, cm.y, cm.z]) - prev_moving))
            )
            prev_moving = np.array([cm.x, cm.y, cm.z])
        if i % 3 == 0 or i == args.frames - 1:
            print(
                f"{i:>3}{row['parked'][0]:>12}{row['moving'][0]:>12}{e_hit:>11}"
                f"{row['parked'][1]:>14}{row['moving'][1]:>14}{moved:>11.2f}"
            )

    # ---- 自证 ----
    print()
    ok_lidar = all(v > 0 for v in l_hits.values())
    print(
        "[自证 1] LiDAR 对照:" + " / ".join(f"{k} {v}" for k, v in l_hits.items()) + " 点"
        f" ⇒ {'两臂都看得见,夹具成立' if ok_lidar else '★ 夹具坏了,下面读数不可解读'}"
    )
    print(f"[自证 2] 运动臂逐帧确实在动(末帧位移 {moved:.2f} m > 0)")
    if not ok_lidar:
        raise SystemExit("LiDAR 对照没看见目标 —— 先修夹具,别读数")

    print()
    net = {k: r_hits[k] - r_hits["empty"] for k in ARM_BP}
    # ★★ **位置自证** —— 这是整个探针里唯一从没打过印的变量。命中中心取的是
    #   `actor.get_transform()`,所以"读数大"既可能是**雷达对它敏感**,
    #   也可能是**它其实离雷达很近**(近 ⇒ 立体角大 ⇒ 按几何采样本就该多)。
    #   不打印这一列,两种情形在下游长得一模一样。
    rp = radar.get_transform().location
    rpos = np.array([rp.x, rp.y, rp.z])
    ep = ego.get_transform().location
    epos = np.array([ep.x, ep.y, ep.z])
    print(f"{'臂':<9}{'蓝图':<34}{'离雷达':>9}{'离 ego':>9}{'雷达净额':>10}")
    for name in ARM_BP:
        if name not in cars:
            continue
        c = cars[name].get_transform().location
        cc = np.array([c.x, c.y, c.z])
        print(
            f"{name:<9}{(args.slot0_bp if name == 'walker' else ARM_BP[name]):<34}"
            f"{np.linalg.norm(cc - rpos):>9.2f}"
            f"{np.linalg.norm(cc - epos):>9.2f}{net[name]:>10}"
        )
    if all_az:
        az = np.concatenate(all_az)
        el = np.concatenate(all_el)
        print()
        print("== **全帧点云的角分布** —— 用来量这批探针里雷达的**实际** FOV ==")
        for tag, arr in (("方位 az", az), ("仰角 el", el)):
            print(
                f"  {tag}: min {arr.min():7.2f} | 5% {np.percentile(arr, 5):7.2f} | 中位 {np.median(arr):7.2f} "
                f"| 95% {np.percentile(arr, 95):7.2f} | max {arr.max():7.2f}"
            )
        print(
            f"  ⇒ 实测方位半角 ≈ {max(abs(az.min()), abs(az.max())):.2f}°,"
            f" 仰角半角 ≈ {max(abs(el.min()), abs(el.max())):.2f}°"
        )
        # ★ **方位密度** —— 若射线密度在视轴附近尖峰,那"正前方有、9.9° 没有"就通了。
        #   每 5° 一档,报每档的点数(等宽 ⇒ 点数即密度)。
        edges = np.arange(-40, 45, 5)
        hist, _ = np.histogram(az, bins=edges)
        print("  方位密度(每 5° 一档的点数):")
        for lo, n in zip(edges[:-1], hist, strict=True):
            bar = "█" * int(n / max(hist.max(), 1) * 40)
            print(f"    [{lo:>4.0f},{lo + 5:>4.0f})° {n:>7} {bar}")

    if args.dump_shape:
        print()
        print("== 形态诊断(命中点,雷达自身系)== 方位 az / 仰角 el / 深度 d,单位 度/度/米")
        for name in ARM_BP:
            fc = frame_counts[name]
            if not shape[name]:
                print(f"  {name:<8} 无命中点")
                continue
            q = np.concatenate(shape[name])
            d = np.linalg.norm(q, axis=1)
            az = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
            el = np.degrees(np.arctan2(q[:, 2], np.hypot(q[:, 0], q[:, 1])))
            print(
                f"  {name:<8} 命中 {len(q):>6} 点 | 逐帧 "
                f"min {min(fc)} / max {max(fc)} / 中位 {int(np.median(fc))}"
            )
            for tag, arr in (("方位 az", az), ("仰角 el", el), ("深度 d", d)):
                print(
                    f"           {tag}: 中位 {np.median(arr):7.2f} | 5–95% "
                    f"[{np.percentile(arr, 5):7.2f}, {np.percentile(arr, 95):7.2f}] | "
                    f"极差 {arr.max() - arr.min():7.2f}"
                )
            # ★ 细粒度结构:若雷达按**离散仰角栅格**发射,仰角只有几个离散值,
            #   而那些仰角打到地面就成**弧线** —— 这能解释"一大团地面点"。
            print(
                f"           唯一值:方位 {len(np.unique(np.round(az, 2)))}/{len(az)}"
                f" | 仰角 {len(np.unique(np.round(el, 2)))}/{len(el)}"
                f" | 深度 {len(np.unique(np.round(d, 3)))}/{len(d)}"
            )
            ez = np.unique(np.round(el, 2))
            if len(ez) <= 20:
                print(f"           仰角离散值({len(ez)} 个): {ez.tolist()}")
            else:
                hh, ee = np.histogram(el, bins=np.arange(el.min(), el.max() + 0.25, 0.25))
                top = np.argsort(-hh)[:6]
                print(
                    "           仰角最密的 6 档(0.25° 分箱): "
                    + ", ".join(f"[{ee[j]:.2f},{ee[j] + 0.25:.2f})={hh[j]}" for j in sorted(top))
                )
            w = _to_world(radar.get_transform(), q)
            print(
                f"           世界包围盒 x[{w[:, 0].min():7.2f},{w[:, 0].max():7.2f}] "
                f"y[{w[:, 1].min():7.2f},{w[:, 1].max():7.2f}] z[{w[:, 2].min():7.2f},{w[:, 2].max():7.2f}]"
            )

    if args.depth_check and any(depth_delta.values()):
        print()
        print("== 深度裁判(雷达点斜距 − 该方位上 LiDAR 最前面那个面的斜距)==")
        print(f"{'臂':<9}{'可比点':>8}{'中位 Δ':>9}{'|Δ|<0.3m':>10}{'Δ>1m(穿过)':>13}   判定")
        for name in ARM_BP:
            if not depth_delta[name]:
                print(f"{name:<9}    (无点)")
                continue
            dd = np.concatenate(depth_delta[name])
            dd = dd[np.isfinite(dd)]
            if not len(dd):
                print(f"{name:<9}    (该方位上 LiDAR 无点,判不了)")
                continue
            on = float((np.abs(dd) < 0.3).mean())
            thru = float((dd > 1.0).mean())
            verdict = "✓ 落在表面上" if on > 0.5 else ("★ **穿过物体**(点在它后面)" if thru > 0.5 else "混合")
            print(f"{name:<9}{len(dd):>8}{np.median(dd):>9.2f}{on:>10.0%}{thru:>13.0%}   {verdict}")

    if args.depth_check and probe_bin is not None:
        print()
        print("== 单方位箱深潜(0.05° 箱,取点数最多的那个)==")
        haz, hrd, hld = probe_bin
        print(
            f"  方位 {haz:+.3f}°:雷达 {len(hrd)} 点,深度 "
            f"[{hrd.min():.2f}, {hrd.max():.2f}] 中位 {np.median(hrd):.2f}(跨度 {hrd.ptp():.2f} m)"
        )
        if len(hld):
            print(
                f"                 LiDAR {len(hld)} 点,深度 "
                f"[{hld.min():.2f}, {hld.max():.2f}] 中位 {np.median(hld):.2f}(跨度 {hld.ptp():.2f} m)"
            )
        else:
            print("                 LiDAR 该方位**无点**")
        if probe_raw is not None:
            uniq = np.unique(probe_raw, axis=0)
            print(
                f"  ★ 该箱的**原始检测行**(vel, alt, azi, depth){len(probe_raw)} 条,"
                f"其中**逐位不同**的只有 {len(uniq)} 条"
            )
            if len(uniq) <= 3:
                print(f"    逐位内容: {np.array2string(uniq, precision=4)}")
        print("  雷达该箱的深度值(升序):" + ", ".join(f"{v:.2f}" for v in np.sort(hrd)[:20]))
        if len(hld):
            print(f"  ⇒ 雷达深度跨度 / LiDAR 深度跨度 = {hrd.ptp() / max(hld.ptp(), 1e-6):.1f}×")

    if args.project:
        print()
        print("== 投影检查(命中点 → 语义相机)= 那团点的世界里是什么")
        for name in ARM_BP:
            if not seen_tags[name]:
                print(f"  {name:<8} 无点投进画幅")
                continue
            vals, cnts = np.unique(seen_tags[name], return_counts=True)
            order = np.argsort(-cnts)[:5]
            top = ", ".join(
                f"{TAG_NAMES.get(int(vals[j]), vals[j])}={cnts[j] / sum(cnts):.0%}" for j in order
            )
            print(f"  {name:<8} 投进画幅 {len(seen_tags[name]):>6} 点 | tag 前 5:{top}")

    if args.ego_speed and any(vel_rec.values()):
        print()
        print(f"== 速度旁证(ego {args.ego_speed} m/s)== 静止目标的径向速度必须 = 自车速度投影")
        print(f"{'臂':<9}{'点数':>8}{'|k|':>9}{'R²':>9}{'打乱对照':>10}   判定")
        rng = np.random.default_rng(0)
        for name in ARM_BP:
            if not vel_rec[name]:
                print(f"{name:<9}{0:>8}   (无命中点)")
                continue
            xyz = np.concatenate([v[0] for v in vel_rec[name]])
            meas = np.concatenate([v[1] for v in vel_rec[name]])
            d = np.linalg.norm(xyz, axis=1)
            A = xyz / d[:, None] * args.ego_speed
            k, *_ = np.linalg.lstsq(A, meas, rcond=None)
            res = float(((meas - A @ k) ** 2).sum())
            tot = float(((meas - meas.mean()) ** 2).sum())
            r2 = 1 - res / tot if tot > 0 else float("nan")
            sh = meas.copy()
            rng.shuffle(sh)
            ks, *_ = np.linalg.lstsq(A, sh, rcond=None)
            r2s = 1 - float(((sh - A @ ks) ** 2).sum()) / tot if tot > 0 else float("nan")
            sp = float(np.linalg.norm(k))
            ok = abs(sp - 1.0) < 0.25 and r2 > r2s + 0.3
            print(
                f"{name:<9}{len(meas):>8}{sp:>9.3f}{r2:>9.3f}{r2s:>10.3f}   "
                + ("✓ 与自车速度投影吻合 ⇒ 来自**静止几何**" if ok else "★ 不吻合 ⇒ 这些点不属于那个静止目标")
            )

    print()
    print(f"[判据] {n_frames} 帧合计(**命中半径 {HIT_R} m**,空位臂是同距离的杂波基线):")
    for k in ARMS:
        tag = "(空位,即杂波基线)" if k == "empty" else ""
        netv = net.get(k, 0)
        print(
            f"        雷达 {k:<8} {r_hits[k]:>5} 点 {tag}"
            + (f" ⇒ **归因净额 {netv:+d}**" if k != "empty" else "")
        )
    print()
    verdict = (
        "**有臂超过了空位基线** ⇒ 雷达能打到那类目标:"
        + ", ".join(f"{k}({v:+d})" for k, v in net.items() if v > 0)
        if any(v > 0 for v in net.values())
        else "**所有目标臂都打不赢空位基线** ⇒ 雷达在这套挂点/配置下对目标没有可归因回波"
    )
    print(f"[结论] {verdict}")

    for s_ in [radar, lidar] + ([sem_cam] if sem_cam is not None else []):
        s_.stop()
        s_.destroy()
    for a in cars.values():
        a.destroy()
    ego.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

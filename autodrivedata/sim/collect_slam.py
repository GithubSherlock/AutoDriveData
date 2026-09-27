"""B2 SLAM 序列采集:ego 路线行驶 → 逐帧 velodyne + **ego 真值位姿**。

**为什么单开一个采集器**(不复用 `collect_drive.py`):
- `collect_drive.py` 走 autopilot 且**不落位姿** → 旧 `kitti_drive` 的 SLAM 结果
  只能报"首末位姿距离",冒充不了精度指标(Plan2.md P-H 已如实标注该口径缺陷)。
  SLAM 评估(ATE/RPE/回环收益)**必须有真值位姿**——本采集器逐帧写
  `training/pose/{id}.txt`(KITTI 12 数行主序,见 `export/kitti.write_pose`)。
- **autopilot 在这里可用**:A/B 纪律禁 autopilot 是因为帧级配对要求可复现;SLAM 有
  真值位姿后,可复现性不再是前提。
- **不写图像**:SLAM 只用点云 + 位姿。150 帧图像 ≈ 400M,点云+位姿 ≈ 60M。

## `--route loop`:为什么需要它(实测根因,不是"想加个功能")

**后端(ScanContext 回环 + PGO)在真数据上一次都没触发过** —— 两条已测序列都是 `n_loops=0`。
根因是**路线不是回环**:`outputs/kitti_slam` 400 帧走的是马蹄形,两条平行街相距 **68 m**,
而 `SC_MAX_RANGE = 40 m` 的 range-view 描述子看不到对面那条街(实测自接近最小 58.70 m,
间距 <40 m 的帧对数为 **0**)。且**车在第 320 帧就停了**,后 80 帧原地不动
⇒「多采帧数」是无效解(只是多录静止帧),必须换控制方式。

`loop` 模式 = **路网找环(`route.find_cycle`)+ 纯追踪跑 `--laps` 圈**,让同一处被真的走两次。
纯几何(`Cycle` / `pure_pursuit` / `lap_budget`)在 [`route.py`](route.py),本文件只做编排。

**★ dry-run 是第一道闸**:回环候选要求两次到访帧差 ≥ `SC_MIN_GAP_NODES × KEYFRAME_EVERY`
(= 250 帧),而一圈帧数 = 环长/(速度×tick) ⇒ 换算成**速度上限**。开快了不是"少采几帧",
而是那两圈**根本不可能**触发回环。所以先 `--dry-run` 报预算,不通过就换 `--spawn-index`,
**别先采完 25 分钟才发现白跑**。

用法:
  python -m autodrivedata.sim.collect_slam --frames 400 --out outputs/kitti_slam
  python -m autodrivedata.sim.collect_slam --frames 200 --speed 0 --map Town13 --out outputs/kitti_slam_t13

  # 闭环模式(先 dry-run 看闸门,过了再采)
  python -m autodrivedata.sim.collect_slam --route loop --dry-run --map Town10HD_Opt
  python -m autodrivedata.sim.collect_slam --route loop --laps 2 --map Town10HD_Opt --out outputs/kitti_loop
"""

from __future__ import annotations

import argparse
import json
import math
import queue
from typing import cast

import carla
import numpy as np

from autodrivedata.gt.export.kitti import frame_paths, write_pose
from autodrivedata.perception.semantic import semantic_to_velodyne_bin
from autodrivedata.sim.carla_common import (
    LIDAR_ATTRS,
    SENSOR_OFFSET,
    loc,
    rad,
    spawn_ego,
    spawn_ego_at,
    sync_mode,
)
from autodrivedata.sim.route import (
    NodeKey,
    find_cycle,
    lap_budget,
    lateral_error,
    make_successors,
    pure_pursuit,
    route_closure,
    route_length,
    speed_ceiling,
    track_index,
)
from autodrivedata.slam.core import KEYFRAME_EVERY, SC_MIN_GAP_NODES
from autodrivedata.utils.geometry import carla_lidar_to_velodyne, carla_rotation_matrix
from autodrivedata.utils.paths import project_path

# 一圈至少要跑这么多帧,否则第二圈到访时第一圈才过去不到 SC_MIN_GAP_NODES 个关键帧,
# **连回环候选都进不去**。这是从 slam.core 的两个常量推出来的,不在这里另立数字。
MIN_FRAMES_PER_LAP = SC_MIN_GAP_NODES * KEYFRAME_EVERY

LOOP_SPEED_DEFAULT = 8.0  # `--speed 0` 时的目标速度(还要再被 speed_ceiling 压)
LOOP_SPEED_FLOOR = 3.0  # 低于此速度的环不收:采集时长不可接受,且点云会退化成原地堆叠
LOOP_MAX_LEN_M = 600.0  # 单圈上限:再长则帧数与耗时爆炸,换起点比硬跑划算

# 跟线自证(防"打舵符号反了朝反方向冲出去"这类**看着像在开**的失败)
OFF_ROUTE_ABORT_M = 8.0  # 离路线这么远
OFF_ROUTE_PATIENCE = 40  # 连续这么多帧 ⇒ 中止(不静默产一份跟丢线的数据)


def ego_pose_matrix(ego_t: carla.Transform) -> np.ndarray:
    """ego CARLA 位姿 → 4×4(ego 局部系 → 世界系)。

    旋转块走 `geometry.carla_rotation_matrix`(组合顺序 Rz·Ry·Rx,与 pycarla
    `Transform.get_matrix()` 旋转块逐元素一致,有 oracle 单测锁定)——**不要**自己
    按 (roll,pitch,yaw) 拼矩阵:`carla_common.rad()` 的返回序是 **(pitch, yaw, roll)**,
    正是该函数要的序,两者配套;自拼极易错序。
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = carla_rotation_matrix(rad(ego_t.rotation))
    T[:3, 3] = loc(ego_t)
    return T


# ---------------------------------------------------------------------------
# 闭环模式:路网建图 → 找环 → 预算闸门
# ---------------------------------------------------------------------------
def spawn_index_of(world: carla.World, ego: carla.Vehicle, *, tol: float = 0.2) -> int | None:
    """ego 当前落在哪个 spawn point 上(**记录用,不参与控制**)。

    闭环采集要**可复现**:`spawn_ego` 取的是"第一个空位",而空位取决于清场顺序 ——
    不记下实际用的那个点,下次换台机器/换张图重跑就可能换起点换环,**回归夹具就没了**。
    对不上(自定义摆位 / 悬架沉降超过 tol)返回 `None` —— **不猜**(同 `spawn_ego_at` 的失败即报错纪律)。
    """
    p = ego.get_transform().location
    for i, sp in enumerate(world.get_map().get_spawn_points()):
        if abs(sp.location.x - p.x) < tol and abs(sp.location.y - p.y) < tol:
            return i
    return None


def plan_loop_route(
    world: carla.World,
    ego: carla.Vehicle,
    *,
    step: float,
    laps: int,
    delta: float,
    speed_req: float,
) -> dict:
    """找环 → 算闭合 → 算预算;任何一条闸门不过就 `RuntimeError`(带可执行建议)。

    闸门顺序是刻意的:**先报几何、再报预算** —— 几何不闭合的环连长度都不可信。
    """
    cmap = world.get_map()
    successors, register, cache = make_successors(step)
    start = register(cmap.get_waypoint(ego.get_transform().location))
    # BFS 规模按"找到 max_len 的环"给 2× 余量 + 固定头:真实路网每次 successors 是一次
    # next() 展开,不设限会在街区图里走很远(虽然 libcarla 侧是本地计算,仍要可预期)
    cyc = find_cycle(start, successors, max_nodes=int(2 * LOOP_MAX_LEN_M / step) + 50)
    if cyc is None:
        raise RuntimeError(
            f"从起点 {start} 出发**找不到有向环**(BFS 上限 {int(2 * LOOP_MAX_LEN_M / step) + 50} 节点)。\n"
            "  可能:① 起点在断头/高速段;② 该处 next() 路网连通性有瑕疵。\n"
            "  → 换 `--spawn-index` 重试(路口密集处更容易成环)。"
        )
    keys = [cast(NodeKey, k) for k in (*cyc.prefix, *cyc.loop)]
    # 缓存值是 duck-typed `WaypointLike`(适配层零 carla)⇒ 取坐标时才收回真类型。
    # 同 `cast(carla.Vehicle, v)` 的 pyi 桩纪律:边界上一次 cast,不是到处放宽。
    points = [loc(cast(carla.Waypoint, cache[k]).transform) for k in keys]
    # 单圈几何只看 `loop` 段(prefix 是引路,只跑一次,不该算进单圈长度)
    loop_pts = points[cyc.loop_start :]
    closure = route_closure(loop_pts, tol=step)
    prefix_len = route_length(points[: cyc.loop_start])
    loop_len = closure["length_m"]

    # ① 几何闸:首末接不上 = 这不是环(节点键只保证"图上有回边",不保证"几何闭合")
    gap_tol = max(2.5 * step, 0.05 * loop_len)
    if not closure["n"] or closure["end_gap_m"] > gap_tol:
        raise RuntimeError(
            f"找到的环**几何上不闭合**:首末相距 {closure['end_gap_m']:.2f} m"
            f"(单圈 {loop_len:.1f} m,容差 {gap_tol:.2f} m)。\n"
            "  → 多为路网采样步长与车道长度不整除;换 `--route-step` 或 `--spawn-index`。"
        )
    # ② 长度闸:环太长 ⇒ 帧数与耗时爆炸,换起点比硬跑划算
    if loop_len > LOOP_MAX_LEN_M:
        raise RuntimeError(
            f"单圈 {loop_len:.1f} m > 上限 {LOOP_MAX_LEN_M:.0f} m(BFS 给的是**最短**环,仍这么长)。\n"
            "  → 换 `--spawn-index` 到小路密集处。"
        )

    # ③ ★ 预算闸:一圈帧数 = 环长/(速度·tick) ≥ MIN_FRAMES_PER_LAP,等价于速度上限
    speed_max = speed_ceiling(loop_len, delta=delta, min_frames_per_lap=MIN_FRAMES_PER_LAP)
    speed = speed_req if speed_req > 0 else min(LOOP_SPEED_DEFAULT, speed_max)
    if speed_req > 0 and speed_req > speed_max:
        raise RuntimeError(
            f"`--speed {speed_req}` 超过该环的上限 {speed_max:.2f} m/s"
            f"(单圈 {loop_len:.1f} m ÷ {MIN_FRAMES_PER_LAP} 帧 ÷ {delta} s)。\n"
            f"  开这么快一圈只有 {loop_len / (speed_req * delta):.0f} 帧 < {MIN_FRAMES_PER_LAP},"
            "两次到访的帧差不够,**回环候选必然为空**。\n"
            f"  → 用 `--speed {min(LOOP_SPEED_DEFAULT, speed_max):.1f}` 或更慢;"
            "要先跑快些就换更长的环(`--spawn-index`)。"
        )
    if speed < LOOP_SPEED_FLOOR:
        raise RuntimeError(
            f"自动选出的速度 {speed:.2f} m/s < 下限 {LOOP_SPEED_FLOOR} m/s(单圈仅 {loop_len:.1f} m)。\n"
            "  环太短:要么慢到采集时长不可接受,要么快到回环帧差不够。\n"
            "  → 换 `--spawn-index` 取更长的环。"
        )
    budget = lap_budget(
        loop_len,
        laps=laps,
        delta=delta,
        speed=speed,
        min_frames_per_lap=MIN_FRAMES_PER_LAP,
    )
    return {
        "start_key": list(start),
        "n_nodes": len(points),
        "n_loop_nodes": len(loop_pts),
        "loop_start": cyc.loop_start,
        "prefix_len_m": round(prefix_len, 3),
        "points": [list(p) for p in points],
        "closure": closure,
        "budget": budget,
        "speed": speed,
        "speed_max": speed_max,
    }


def report_loop_plan(plan: dict, *, laps: int, step: float, delta: float) -> None:
    """把闸门结论打印成人能读的块(数字全部来自计划本身,不手抄)。"""
    b, c = plan["budget"], plan["closure"]
    print("[loop] 路网找环 OK")
    print(
        f"  节点 {plan['n_nodes']}(环 {plan['n_loop_nodes']})| 单圈 {c['length_m']:.1f} m "
        f"| 引路 {plan['prefix_len_m']:.1f} m | 首末缺口 {c['end_gap_m']:.2f} m"
    )
    print(
        f"  速度 {plan['speed']:.2f} m/s(上限 {plan['speed_max']:.2f})"
        f"| 一圈 {b['frames_per_lap']:.0f} 帧(门槛 {b['min_frames_per_lap']})"
    )
    print(
        f"  {laps} 圈建议 {b['frames_recommended']} 帧 ≈ "
        f"{b['frames_recommended'] * delta / 60:.1f} min(仿真时间)| 采样步长 {step} m"
    )
    print(f"  起点 spawn point #{plan['spawn_index']}(复跑时用 `--spawn-index {plan['spawn_index']}`)")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frames",
        type=int,
        default=0,
        help="采集帧数;0 = 自动(auto 模式 400;loop 模式用 dry-run 报的预算)",
    )
    ap.add_argument("--out", default="outputs/kitti_slam", help="输出 KITTI root(经 project_path)")
    ap.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="auto: >0 = 定速直行(确定性),0 = TM autopilot 走路线;loop: 目标速度,0 = 自动取 min(8, 环的上限)",
    )
    ap.add_argument("--map", default=None, help="切图(缺省=当前图)")
    ap.add_argument("--semantic-lidar", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--route",
        choices=("auto", "loop"),
        default="auto",
        help="auto = 现有行为(autopilot/定速直行);loop = 路网找环 + 纯追踪跑 laps 圈",
    )
    ap.add_argument("--laps", type=int, default=2, help="loop 模式跑几圈(回环至少 2 圈)")
    ap.add_argument("--route-step", type=float, default=3.0, help="loop 模式的路线采样步长(m)")
    ap.add_argument("--lookahead", type=float, default=8.0, help="loop 模式纯追踪前视距离(m)")
    ap.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="同步模式 tick 间隔(s);同时参与帧预算换算,改了要一起看",
    )
    ap.add_argument(
        "--spawn-index",
        type=int,
        default=None,
        help="指定 spawn point(可复现起点;loop 模式找环不理想时换它重试)",
    )
    ap.add_argument("--dry-run", action="store_true", help="只找环 + 报预算,不采集")
    args = ap.parse_args()
    if args.route == "loop" and args.laps < 2:
        ap.error(f"--route loop 的 --laps 至少为 2(1 圈不存在重访),收到 {args.laps}")

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)
    world = client.load_world(args.map) if args.map else client.get_world()
    sync_mode(world, args.delta)

    # 清场:残留 actor 会阻塞 spawn point(红线纪律,勿删)。
    # **逐个 try**:destroy 父 actor 会连带销毁其子传感器,快照里的子 actor 再 destroy
    # 就报 "unable to destroy actor: not found"(噪声,非错误)——吞掉即可。
    for a in world.get_actors():
        if a.type_id.startswith("vehicle.") or a.type_id.startswith("sensor."):
            try:
                a.destroy()
            except RuntimeError:
                pass
    world.tick()

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    ego = spawn_ego_at(world, args.spawn_index) if args.spawn_index is not None else spawn_ego(world)

    # --- 路线规划(loop 模式)+ 输出目录 ---
    out = project_path(args.out)
    # `plan is not None` = 闭环模式的唯一依据(下面的窄化全部走它;`args.route` 是字符串
    # 比较、类型检查器不拿它当守卫 ⇒ 用它反而要到处 assert)
    plan: dict | None = None
    route_pts: list[list[float]] = []  # 与 plan["points"] 同一对象;主循环用局部量免得每帧查字典
    loop_start = 0
    speed_loop = 0.0
    n_frames = args.frames
    if args.route == "loop":
        plan = plan_loop_route(
            world, ego, step=args.route_step, laps=args.laps, delta=args.delta, speed_req=args.speed
        )
        # 记**实际**起点而不是 `--spawn-index` 的入参:`spawn_ego` 取的是"第一个空位",
        # 不记下来就没有可复现性(回归夹具的前提)。对不上就如实记 None,不猜。
        plan["spawn_index"] = args.spawn_index if args.spawn_index is not None else spawn_index_of(world, ego)
        report_loop_plan(plan, laps=args.laps, step=args.route_step, delta=args.delta)
        route_pts, loop_start, speed_loop = plan["points"], plan["loop_start"], plan["speed"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "route_loop.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.dry_run:
            ego.destroy()
            print(f"[dry-run] 计划已写入 {out}/route_loop.json —— 采集请去掉 --dry-run")
            return
        if n_frames <= 0:
            # 帧预算是**安全上限,不是终止条件** —— 主循环在 `laps_done >= args.laps` 时收工。
            # 实测纯追踪到不了目标速度(目标 8 m/s ⇒ 实际约 6 m/s,0.75×),把预算当终止条件
            # 会永远跑不满 2 圈;×2 只是"真跑飞了也不至于无限采"的兜底。
            n_frames = int(plan["budget"]["frames_recommended"]) * 2
    elif n_frames <= 0:
        n_frames = 400

    # --- ego 控制方式 ---
    vel = carla.Vector3D(0.0, 0.0, 0.0)
    if plan is not None:
        # 纯追踪自己给油/刹车 ⇒ 既不定速也不 autopilot;清制动残留(红线:brake 一旦设置每步生效)
        ego.set_autopilot(False)
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        print(f"[ego] 闭环纯追踪 | 目标速度 {speed_loop:.2f} m/s | 前视 {args.lookahead} m")
    elif args.speed > 0:
        ego.set_autopilot(False)
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))  # 清制动残留
        fwd = ego.get_transform().get_forward_vector()
        vel = carla.Vector3D(x=fwd.x * args.speed, y=fwd.y * args.speed, z=0.0)
        print(f"[ego] 定速 {args.speed} m/s 直行(确定性)")
    else:
        ego.set_autopilot(True, tm.get_port())
        print("[ego] TM autopilot(路线行驶,自然重访 → 后端回环有真实测试数据)")

    bp_lib = world.get_blueprint_library()
    lid_bp = bp_lib.find(
        "sensor.lidar.ray_cast" if not args.semantic_lidar else "sensor.lidar.ray_cast_semantic"
    )
    for k, v in LIDAR_ATTRS.items():
        lid_bp.set_attribute(k, v)
    lidar = cast(carla.Sensor, world.spawn_actor(lid_bp, SENSOR_OFFSET, attach_to=ego))

    lid_q: queue.Queue = queue.Queue()
    lidar.listen(lid_q.put)
    for _ in range(5):  # 预热
        world.tick()
        lid_q.get(timeout=10)

    mode = args.route if args.route == "loop" else ("straight" if args.speed > 0 else "autopilot")
    poses: list[np.ndarray] = []
    lateral: list[float] = []  # 逐帧跟线误差(loop 模式的离线自证)
    steers: list[float] = []
    idx = 0  # 前视点索引(控制用;跨帧单调向前,回绕由 pure_pursuit 的 wrap_at 处理)
    prog = 0  # 进度索引(**自证用**;与前视索引是两个量,别混 —— 见 route.track_index)
    laps_done = 0
    off_run = 0
    i = -1  # 帧号预置:预热段就抛异常时下面的 except 也取得到(-1 = 一帧都没写)

    try:
        for i in range(n_frames):
            if plan is not None:
                # 控制用**上一 tick 之后**的状态(此刻 tick 还没发生)
                t_now = ego.get_transform()
                v_now = ego.get_velocity()
                ctl = pure_pursuit(
                    (t_now.location.x, t_now.location.y),
                    math.radians(t_now.rotation.yaw),
                    route_pts,
                    idx,
                    speed=math.hypot(v_now.x, v_now.y),
                    target_speed=speed_loop,
                    lookahead=args.lookahead,
                    wrap_at=loop_start,
                )
                ego.apply_control(
                    carla.VehicleControl(throttle=ctl.throttle, steer=ctl.steer, brake=ctl.brake)
                )
                idx = ctl.idx
                steers.append(ctl.steer)
                laps_done += 1 if ctl.wrapped else 0  # ★ 圈数唯一依据
            elif args.speed > 0:
                ego.set_target_velocity(vel)  # 每帧重发:VehicleControl 一旦设置就每步生效

            world.tick()
            pts: carla.LidarMeasurement = lid_q.get(timeout=10)
            ego_t = ego.get_transform()  # 位姿用**本 tick** 的状态(与点云同帧)
            T = ego_pose_matrix(ego_t)
            poses.append(T)

            if plan is not None:
                # ★ 自证锚在**进度索引**上,不是 `idx`(前视索引):挂前视上量到的是前视距离,
                # 完美跟线也会读到 8~9 m,与中止阈值重合 ⇒ 正常采集就踩响中止(踩过)。
                prog = track_index(route_pts, (ego_t.location.x, ego_t.location.y), prog)
                err = lateral_error(route_pts, (ego_t.location.x, ego_t.location.y), prog)
                lateral.append(err)
                off_run = off_run + 1 if err > OFF_ROUTE_ABORT_M else 0
                if off_run >= OFF_ROUTE_PATIENCE:
                    raise RuntimeError(
                        f"[frame {i}] 跟线失败:连续 {OFF_ROUTE_PATIENCE} 帧偏离路线 > "
                        f"{OFF_ROUTE_ABORT_M} m(当前 {err:.2f} m,均值 {np.mean(lateral):.2f} m,"
                        f"圈数 {laps_done}/{args.laps},平均打舵 {np.mean(steers):+.3f},"
                        f"进度点 {prog}/{len(route_pts)})。\n"
                        "  平均打舵若持续**同号**且量值大 ⇒ 多为打舵符号反了或前视距离/增益"
                        "与实际轴距不匹配。\n"
                        "  **中止而不产出**是刻意的:跟丢线的序列对回环验证没有价值,"
                        "静默落盘只会得到一份看着完整、实则无重访的数据。"
                    )

            raw = np.frombuffer(pts.raw_data, dtype=np.float32)
            if args.semantic_lidar:
                velo = semantic_to_velodyne_bin(raw.reshape(-1, 6), seed=n_frames * 100 + i)
            else:
                velo = carla_lidar_to_velodyne(raw.reshape(-1, 4))

            paths = frame_paths(out, str(i))
            paths.velodyne.parent.mkdir(parents=True, exist_ok=True)
            np.asarray(velo, dtype=np.float32).reshape(-1, 4).tofile(paths.velodyne)
            write_pose(out, str(i), T)
            if (i + 1) % 50 == 0 or i == n_frames - 1:
                tail = (
                    f"| 圈 {laps_done}/{args.laps} 跟线 {np.mean(lateral[-50:]):.2f} m"
                    if args.route == "loop"
                    else ""
                )
                print(
                    f"[frame {i + 1}/{n_frames}] ego @ {tuple(round(v, 1) for v in loc(ego_t))} "
                    f"| {len(velo)} 点 {tail}"
                )
            if plan is not None and laps_done >= args.laps:
                # 圈数达标即收工:`--frames 0` 给的预算是按**目标**速度算的,实测到不了 ⇒
                # 拿它当终止条件会永远跑不满。多跑的那几帧对回环无增益,只烧 GPU。
                print(f"[loop] 已跑满 {laps_done} 圈(第 {i + 1} 帧收工;帧预算只是安全上限)")
                break
    except RuntimeError as e:
        # 帧是**逐帧落盘**的,所以"不产出"只指不产出**可用的序列** —— 目录里那半截帧
        # 必须留个记号,否则一份没有摘要的半成品看起来和完整数据集完全一样(踩过)。
        (out / "ABORTED.json").write_text(
            json.dumps(
                {"frame": i, "n_frames_written": i + 1, "reason": str(e)}, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        print(f"[abort] 半成品标记已写入 {out}/ABORTED.json —— 该目录**不可**当数据集用")
        raise
    finally:
        lidar.stop()
        lidar.destroy()
        ego.destroy()
        for a in world.get_actors():
            if a.type_id.startswith("vehicle") or a.type_id.startswith("walker"):
                a.destroy()

    # 轨迹摘要(不冒充精度指标:这是**真值**轨迹的几何,供选数据/判回环用)
    P = np.array([T[:3, 3] for T in poses])
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    path_len = float(step.sum())
    yaw = np.array([np.arctan2(T[1, 0], T[0, 0]) for T in poses])
    summary = {
        "dataset": str(args.out),
        "map": world.get_map().name,
        "n_frames": len(poses),
        "mode": mode,
        "speed_set": args.speed,
        "path_len_m": round(path_len, 3),
        "start_to_end_m": round(float(np.linalg.norm(P[-1] - P[0])), 3),
        "step_mean_m": round(float(step.mean()), 4),
        "step_median_m": round(float(np.median(step)), 4),
        "step_max_m": round(float(step.max()), 4),
        "yaw_total_deg": round(float(np.degrees(yaw[-1] - yaw[0])), 3),
        "bbox_min": [round(float(x), 2) for x in P.min(0)],
        "bbox_max": [round(float(x), 2) for x in P.max(0)],
    }
    if plan is not None:
        # ★ 这三项是**不依赖 SLAM** 的第一层判据:圈数够 + 跟线质量好 ⇒ 数据里真有重访。
        # 后端触发与否是第二层(loops.json),两者分开看,别用后者去反推前者。
        summary["laps_done"] = laps_done
        summary["laps_target"] = args.laps
        summary["lateral_err_mean_m"] = round(float(np.mean(lateral)), 4) if lateral else None
        summary["lateral_err_p95_m"] = round(float(np.percentile(lateral, 95)), 4) if lateral else None
        summary["lateral_err_max_m"] = round(float(np.max(lateral)), 4) if lateral else None
        summary["loop"] = {
            "n_nodes": plan["n_nodes"],
            "n_loop_nodes": plan["n_loop_nodes"],
            "prefix_len_m": plan["prefix_len_m"],
            "speed": plan["speed"],
            "speed_max": plan["speed_max"],
            "closure": plan["closure"],
            "budget": plan["budget"],
            "route_step_m": args.route_step,
            "lookahead_m": args.lookahead,
            "spawn_index": plan["spawn_index"],  # 实际起点(不是入参)
        }
    out.mkdir(parents=True, exist_ok=True)
    (out / "slam_seq_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] {len(poses)} 帧 | 路径 {path_len:.1f} m | 首末 {summary['start_to_end_m']} m")
    print(
        f"  步长 mean {summary['step_mean_m']} / max {summary['step_max_m']} m | yaw 总变化 {summary['yaw_total_deg']}°"
    )
    if plan is not None:
        print(
            f"  圈数 {laps_done}/{args.laps} | 跟线 mean {summary['lateral_err_mean_m']} / "
            f"max {summary['lateral_err_max_m']} m"
        )
        if laps_done < args.laps:
            print(
                f"  [warn] 只跑完 {laps_done} 圈 < 目标 {args.laps} —— 回环需要**至少 2 次**到访;"
                "加大 --frames 或提高 --speed 后重采"
            )
    print(f"  → {out}/training/velodyne + pose,摘要 {out}/slam_seq_summary.json")


if __name__ == "__main__":
    main()

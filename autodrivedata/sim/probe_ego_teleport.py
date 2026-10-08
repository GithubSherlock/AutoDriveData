"""★ **把 ego 逐帧瞬移到绝对位姿,读回来到底准不准、可不可复现?**(需 CARLA)

## 它要判的是一条**已归档的否决**

`docs/edit-pointcloud-plan.md` §4 的重评条件 #1 是「位姿逐位可复现」。
它现在的修法建议是「弃用物理驱动、改成逐帧瞬移」。**但本仓已经实测过一件相反的事**:

> `Plan4.md:889` —— 摆雷达探针时用 `vehicle.tesla.model3`,每档 **销毁重建**
> (不 `set_transform` —— **车上物理反复瞬移不可靠**:第一版就是这么写的,
> LiDAR 在 15–62 m 全 0 而它本该看得见)。

⚠️ **那条说的是"目标车",不是 ego** —— 两者的差别是实打实的:目标车没有传感器挂在身上,
而且它"看不见"的机制可能是**它被瞬移到了别处**,不一定读回位姿在抖。
⇒ **本仓缺的正是 ego 这一侧的直接证据**,而这个探针就是补它。

## 判据(两条,缺一不可)

| # | 量什么 | 为什么 |
|---|---|---|
| ① | **同一次内**:请求位姿 vs 读回位姿的逐帧差 | 「瞬移到底准不准」 |
| ② | ★ **两次之间**:同一串位姿跑两遍,用**已有的** `lidar_ab.pose_series_report` 比 | 这才是「**逐位可复现**」的定义 —— **只看①会漏掉"每次都偏同一个量"** |

## 反向自证(房子规矩:判据必须先能失败)

同一次探针里跑**物理驱动**(`set_target_velocity`)的**两遍** —— 它们的②**必须不为 0**。
若物理那对也是 0,说明**这个探针根本量不出差异**,① 的读数就不可信。

## 出口

- ② ≈ 0(且物理对照 ≠ 0)⇒ 「逐帧瞬移」**可行**,点云重评条件 #1 有解;
- ② ≠ 0 ⇒ 把这条**记为已被否**,点云线维持收口,别再花力气改采集器。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import carla
import numpy as np

from autodrivedata.perception.lidar_ab import pose_series_report
from autodrivedata.sim.carla_common import clear_generated_actors, ground_z_at, spawn_ego, sync_mode
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 与 P1 A/B 同量级:8 m/s × 0.1 s tick = 0.8 m/帧。
DEF_SPEED = 8.0
DEF_FRAMES = 70
DEF_TICK = 0.1


def straight_sequence(p0: tuple[float, float, float], speed: float, tick: float, n: int):
    """纯函数:沿世界 **+x** 每帧走 `speed·tick` 米(P1 A/B 就是这条直线)。

    ⚠️ 用**绝对**位姿而不是增量 —— 增量会把"上一帧的误差"累积进去,
    而这里要量的恰恰是"位姿本身准不准"。
    """
    return [(p0[0] + i * speed * tick, p0[1], p0[2]) for i in range(n)]


def write_pose_root(root: Path, poses: list[tuple[float, float, float]], yaw: float = 0.0) -> Path:
    """落成 `training/pose/{i:06d}.txt`(3×4),**为了复用** `lidar_ab.pose_series_report`。"""
    d = root / "training" / "pose"
    d.mkdir(parents=True, exist_ok=True)
    c, s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    for i, p in enumerate(poses):
        m = np.eye(3, 4)
        m[:, :3] = r
        m[:, 3] = p
        np.savetxt(d / f"{i:06d}.txt", m.reshape(1, -1))
    return root


def _run_teleport(world, ego, seq) -> list[tuple[float, float, float]]:
    """逐帧 `set_transform` 到绝对位姿,**逐帧读回**。"""
    got = []
    for p in seq:
        ego.set_transform(carla.Transform(carla.Location(x=p[0], y=p[1], z=p[2])))
        world.tick()
        loc = ego.get_location()
        got.append((loc.x, loc.y, loc.z))
    return got


def _run_physics(world, ego, seq, speed: float) -> list[tuple[float, float, float]]:
    """物理驱动:清一次制动残留,然后逐帧 `set_target_velocity`(**与 `collect_ab_route` 同做法**)。

    ⚠️ **速度要乘上去**:`collect_ab_route` 传的是"模长 = `args.speed` 的矢量"。
    第一版这里写了个**单位矢量**(= 1 m/s),于是对照臂跑在 1/8 的工况上、
    步长只有 **0.100 m** 而应当是 0.8 m —— 那样的对照**比归档那对弱得多**,
    不能拿它去证"这个探针能测出 62 mm 级的差"。
    """
    fwd = carla.Vector3D(speed, 0.0, 0.0)  # 世界 +x,模长 = speed
    got = []
    for _ in seq:
        ego.set_target_velocity(fwd)
        world.tick()
        loc = ego.get_location()
        got.append((loc.x, loc.y, loc.z))
    return got


def _reset(world, ego, p0) -> None:
    """把 ego 摆回起点并让物理静下来 —— 两次之间**必须**回同一个状态,否则比的是"起点不同"。"""
    ego.apply_control(carla.VehicleControl(brake=1.0))
    ego.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    ego.set_transform(carla.Transform(carla.Location(x=p0[0], y=p0[1], z=p0[2])))
    for _ in range(4):
        world.tick()


def run(*, frames: int, speed: float, tick: float, out_root: Path) -> dict:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world, tick)
    clear_generated_actors(world)

    pts = world.get_map().get_spawn_points()
    p0 = pts[0].location
    z0 = ground_z_at(world, p0.x, p0.y, p0.z)
    ego = spawn_ego(world)
    ego.set_autopilot(False)
    _reset(world, ego, (p0.x, p0.y, z0))
    print(f"[ego] 起点 ({p0.x:.2f}, {p0.y:.2f}, {z0:.2f});沿世界 +x,{speed} m/s × {tick} s")

    seq = straight_sequence((p0.x, p0.y, z0), speed, tick, frames)
    try:
        tel = []
        for k in (1, 2):
            _reset(world, ego, (p0.x, p0.y, z0))
            tel.append(_run_teleport(world, ego, seq))
            print(f"[teleport] 第 {k} 遍完成")
        phys = []
        for k in (1, 2):
            _reset(world, ego, (p0.x, p0.y, z0))
            phys.append(_run_physics(world, ego, seq, speed))
            print(f"[physics ] 第 {k} 遍完成")
    finally:
        try:
            ego.destroy()
        except Exception:  # noqa: BLE001 —— 探针收尾,失败不该盖住主读数
            pass

    # ① 同一次内:请求 vs 读回。
    #   ⚠️ **只对瞬移臂有意义** —— `seq` 是"请求序列",而物理驱动**从来没被要求跟它走**
    #   (它按 `set_target_velocity` 自己跑)。第一版把物理臂也这么算,得到 **27.2 m** ——
    #   那个数**是范畴错误**,不是"物理偏了 27 m"。物理臂这一格必须报 N/A。
    req = np.array(seq)
    within = {"teleport": float(np.linalg.norm(np.array(tel[0]) - req, axis=1).max())}
    #  物理臂改报**步长**(它唯一有定义的那个量):应当 ≈ speed×tick
    steps = np.linalg.norm(np.diff(np.array(phys[0]), axis=0), axis=1)
    within["physics_step_median"] = float(np.median(steps)) if len(steps) else float("nan")
    # ② 两次之间:落盘后走**已有**的判据
    roots = {}
    for name, runs in (("teleport", tel), ("physics", phys)):
        for k in (1, 2):
            # ⚠️ `runs[k-1]` 才是**那一次**的位姿序列 —— 传 `runs` 会把两遍一起喂进去,
            #    报出来的是 `setting an array element with a sequence`(实测踩到)。
            roots[f"{name}{k}"] = write_pose_root(out_root / f"{name}{k}", runs[k - 1])
    cross = {}
    for name in ("teleport", "physics"):
        rep = pose_series_report(roots[f"{name}1"], roots[f"{name}2"])
        cross[name] = {k: rep[k] for k in ("d_max", "d_last", "step_gap_median", "is_mostly_along_heading")}
    return {
        "frames": frames,
        "speed": speed,
        "tick": tick,
        # ★ 与"现在归档的物理 A/B"同量级的参照(70 帧实测 0.0619 m,见 edit-pointcloud-plan §1)
        "ref_archived_physics_ab_m": 0.0619,
        "within_run_max_m": within,
        "cross_run": cross,
        "out_root": str(out_root),
    }


#: 判「逐位」的严格门槛 —— 3DGS 那个 spectator 做到的是 `0.000e+00`(`PAIR_TOL = 1e-6`)。
BIT_EXACT_M = 1e-6


def verdict(rep: dict) -> str:
    """★ 三分,**先证明探针能失败**。

    ⚠️ 这里**不能**用"零/非零"二分:实测瞬移的两遍差是 **0.39 mm** —— 严格说不是"逐位可复现",
    但相对**归档物理 A/B 的 61.9 mm** 是 **~160×** 的改善。两个说法都对,给的行动却相反,
    所以裁决要**把两个数都摆出来**再分档。
    """
    ctl = rep["cross_run"]["physics"]
    if ctl["d_max"] < BIT_EXACT_M:
        return "未判(物理驱动的两遍也是 0 ⇒ 这个探针量不出差异,它的读数不作数)"
    tel = rep["cross_run"]["teleport"]["d_max"]
    ref = rep["ref_archived_physics_ab_m"]
    if tel < BIT_EXACT_M:
        return "★ 瞬移**逐位可复现**(两遍 max|Δ| = 0),与 3DGS 的 spectator 同档 ⇒ 条件 #1 有解"
    if tel * 10 < ref:
        return (
            f"★ 瞬移**不是逐位**(两遍 max|Δ| = {tel * 1000:.2f} mm ≠ 0),但相对**归档物理 A/B 的 "
            f"{ref * 1000:.0f} mm** 改善 **{ref / tel:.0f}×** ⇒ 条件 #1 的**量级**问题解决了,"
            "但**严格版**(max|Δ| = 0)没有 ⇒ 要不要开这条线取决于'够不够',不是'能不能'"
        )
    return (
        f"★ 瞬移**不逐位也不够好**:两遍 max|Δ| = {tel * 1000:.2f} mm,"
        f"相对归档 {ref * 1000:.0f} mm 只改善 {ref / tel:.1f}× ⇒ 与 `Plan4.md:889` 的否决一致"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--frames", type=int, default=DEF_FRAMES)
    ap.add_argument("--speed", type=float, default=DEF_SPEED)
    ap.add_argument("--tick", type=float, default=DEF_TICK)
    ap.add_argument("--out-root", default="outputs/probe_ego_teleport")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    root = project_path(args.out_root)
    with runlog.run("autodrivedata.sim.probe_ego_teleport") as rl:
        rl.highlight("frames", args.frames)
        rl.highlight("speed", args.speed)
        rl.highlight("tick", args.tick)
        rep = run(frames=args.frames, speed=args.speed, tick=args.tick, out_root=root)
        rep["verdict"] = verdict(rep)
        print("\n=== ego 瞬移 vs 物理驱动 ===")
        print(
            f"  teleport  ①请求 vs 读回 max {rep['within_run_max_m']['teleport'] * 1000:6.2f} mm"
            f" | ②两遍之间 max|Δ| {rep['cross_run']['teleport']['d_max'] * 1000:6.2f} mm"
        )
        print(
            f"  physics   ① **N/A**(它不跟踪请求序列)         "
            f" | ②两遍之间 max|Δ| {rep['cross_run']['physics']['d_max'] * 1000:6.2f} mm"
            f" | 步长中位 {rep['within_run_max_m']['physics_step_median']:.3f} m"
        )
        print(f"  参照:归档物理 A/B(70 帧) {rep['ref_archived_physics_ab_m'] * 1000:.1f} mm")
        print(f"  ⇒ {rep['verdict']}")
        for name in ("teleport", "physics"):
            rl.highlight(f"{name}_cross_max_mm", round(rep["cross_run"][name]["d_max"] * 1000, 3))
        rl.highlight("verdict", rep["verdict"])
        if args.out:
            p = project_path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
            rl.artifact(p, "report")
            print(f"[done] {p.resolve()}")


if __name__ == "__main__":
    main()

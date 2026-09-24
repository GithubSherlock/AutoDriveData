"""CARLA 多 agent 轨迹采集器(HiVT 训练数据源)。

目标:一条长 drive 连续记录 ego + 固定布局 NPC 的逐帧轨迹,后续滑窗切出
50 帧(20 历史 + 30 未来)场景供 HiVT 训练。多 agent 联合预测,agent-centric
局部坐标(HiVT 数据口径)。

设计(2026-09-16,与用户商讨定案):
- **多 agent 联合**:ego + 同向 2 车 + 对向 1 车 = 4 车(NPC 数可调)
- **可控固定布局**(P1 A/B 纪律):NPC 沿 ego 初始朝向布置在固定距离/车道,
  不用 Traffic Manager(可复现),用 set_target_velocity 定速
- **同步 tick 0.1s**:与 HiVT 时间步(ARG=0.1s)严格对齐,50 帧场景 = 5 秒
- **一条长 drive**:连采 ~550 帧 → 滑窗切 500 个 50 帧场景(场景间渐变连续)
- **落盘**:{out}/traj.json — 每帧所有 vehicle 位姿 + 图名 + ego 标识

用法:
  python bin/collect_traj.py --frames 550 --out outputs/traj_town10 \
      --map Town10HD_Opt [--map Town13 运行时切图]
"""

from __future__ import annotations

import argparse
import json
import time
from typing import cast

import carla
from carla_common import loc, spawn_ego_at, sync_mode

from autodrivedata.paths import project_path
from autodrivedata.scenarios import SCENES, merged_weather


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/traj_drive", help="输出根目录")
    ap.add_argument("--map", default=None, help="目标地图(Town10HD_Opt/Town13;None=服务器当前图)")
    ap.add_argument("--scene", default=None, choices=sorted(SCENES), help="天气档(默认 day_clear)")
    ap.add_argument("--frames", type=int, default=550, help="连续采集帧数")
    ap.add_argument("--delta", type=float, default=0.1, help="同步 tick 步长(秒)= HiVT 时间步")
    ap.add_argument("--npc-ahead", type=int, default=2, help="同向前车数")
    ap.add_argument("--npc-oncoming", type=int, default=1, help="对向车数")
    ap.add_argument("--ego-speed", type=float, default=8.0, help="ego 目标速度 m/s")
    ap.add_argument("--spawn-index", type=int, default=0, help="固定用第 N 个 spawn point(直线段)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    if args.map is not None:
        print(f"[map] load_world {args.map} ...")
        client.load_world(args.map)
        client.set_timeout(60.0)
    world = client.get_world()
    map_name = args.map or world.get_map().name
    sync_mode(world, delta=args.delta)

    # 天气(默认 day_clear;显式 --scene 覆写)
    scene = SCENES[args.scene] if args.scene else SCENES["day_clear"]
    world.set_weather(carla.WeatherParameters(**merged_weather(scene)))
    print(f"[scene] {scene.name} [{scene.group}]")

    # 清场(残留 actor 会阻塞 spawn/干扰轨迹)
    for a in world.get_actors():
        if a.type_id.startswith("vehicle") or a.type_id.startswith("walker"):
            a.destroy()
    world.tick()

    # 清场后再 tick 一次确保快照刷新(sync_mode 已补,防御性再来一发)
    world.tick()

    # ego:spawn 后定速直行(锚定 spawn point 固有朝向;直线段用 --spawn-index)
    ego = spawn_ego_at(world, args.spawn_index)
    ego_t = ego.get_transform()
    ego.set_autopilot(False)
    ego.apply_control(carla.VehicleControl())  # 清残留(§5.10:brake 残留会打 0.82 折)
    ego_target_vel = ego_t.get_forward_vector() * args.ego_speed
    ego.set_target_velocity(ego_target_vel)
    print(
        f"[ego] @ {tuple(round(v, 1) for v in loc(ego_t))} yaw={ego_t.rotation.yaw:.1f} "
        f"target {args.ego_speed} m/s"
    )

    # NPC 固定布局:沿 ego 朝向布置(同向前车 + 对向车),定速
    bp_lib = world.get_blueprint_library()
    fwd = ego_t.get_forward_vector()
    right = ego_t.get_right_vector()
    yaw0 = ego_t.rotation.yaw

    npcs: list[carla.Vehicle] = []
    models = ["vehicle.toyota.prius", "vehicle.audi.a2", "vehicle.chevrolet.impala", "vehicle.ford.mustang"]
    for i in range(args.npc_ahead):
        d = 12.0 + 10.0 * i  # 同向前车 12m/22m
        off = 0.0 if i == 0 else 3.6  # 前车同车道,第二辆邻车道
        pos = ego_t.location + fwd * d + right * off
        tf = carla.Transform(pos, carla.Rotation(yaw=yaw0))
        bp = bp_lib.find(models[i % len(models)])
        v = cast(carla.Vehicle, world.try_spawn_actor(bp, tf))
        if v is not None:
            v.set_autopilot(False)
            v.apply_control(carla.VehicleControl())
            v.set_target_velocity(fwd * args.ego_speed)
            npcs.append(v)
            print(f"  [npc] ahead#{i} {models[i % len(models)]} @ {round(d, 1)}m off={off}")
    for i in range(args.npc_oncoming):
        d = 20.0 + 5.0 * i
        off = -3.6  # 对向车道(右偏)
        pos = ego_t.location + fwd * d + right * off
        tf = carla.Transform(pos, carla.Rotation(yaw=yaw0 + 180))
        bp = bp_lib.find(models[(i + 2) % len(models)])
        v = cast(carla.Vehicle, world.try_spawn_actor(bp, tf))
        if v is not None:
            v.set_autopilot(False)
            v.apply_control(carla.VehicleControl())
            v.set_target_velocity(fwd * args.ego_speed)  # 对向车速度符号由朝向负 yaw 表达
            npcs.append(v)
            print(f"  [npc] oncoming#{i} @ {round(d, 1)}m")

    # 预热(位姿稳定 + 轨迹起点非突变)
    for _ in range(5):
        world.tick()
    print(f"[pre] 预热 5 tick,{len(npcs)} NPC 就位")

    # 主采集:逐帧记录所有 vehicle 位姿
    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_actors = [ego] + npcs
    frames: list[dict] = []
    t0 = time.monotonic()
    # 上一 tick 的定速值(第 0 帧由 spawn 时的 set_target_velocity 建立;每 tick 重发,
    # 防 §5.10 brake 残留 / 碰撞减速把 set_target_velocity 的初值冲掉——Town10 旧数据
    # 550 帧后半程停车 262 帧的根因)。
    last_vel = ego_target_vel
    try:
        for i in range(args.frames):
            world.tick()
            # 每 tick 重发定速(清 brake 残留后速度指令就是纯目标速度);定速期间不读位姿
            # (collision/碰撞检测留给训练自证,HiVT 只消费位姿序列)
            ego.apply_control(carla.VehicleControl())
            ego.set_target_velocity(last_vel)
            rec: dict = {"frame": i, "agents": []}
            for j, a in enumerate(all_actors):
                t = a.get_transform()
                rec["agents"].append(
                    {
                        "id": j,
                        "is_ego": j == 0,
                        "x": round(t.location.x, 4),
                        "y": round(t.location.y, 4),
                        "yaw": round(t.rotation.yaw, 4),
                    }
                )
            frames.append(rec)
            if (i + 1) % 50 == 0 or i == args.frames - 1:
                dt = time.monotonic() - t0
                fps = (i + 1) / dt
                e = frames[-1]["agents"][0]
                print(f"[frame {i + 1}/{args.frames}] ego @ ({e['x']:.1f}, {e['y']:.1f}) | {fps:.1f} fps")
    finally:
        meta = {
            "map": map_name,
            "scene": scene.name,
            "delta": args.delta,
            "ego_speed": args.ego_speed,
            "n_agents": len(all_actors),
            "n_frames": len(frames),
            "npc_layout": {
                "ahead": args.npc_ahead,
                "oncoming": args.npc_oncoming,
                "ahead_dist_m": [12.0 + 10.0 * i for i in range(args.npc_ahead)],
            },
        }
        with open(out / "traj.json", "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "frames": frames}, f, ensure_ascii=False, indent=1)
        for a in all_actors:
            a.destroy()
        for a in world.get_actors():
            if a.type_id.startswith("vehicle") or a.type_id.startswith("walker"):
                a.destroy()
    print(f"[done] traj root: {out.resolve()} ({len(frames)} frames × {len(all_actors)} agents)")


if __name__ == "__main__":
    main()

"""灯色动态 GT 采集器(Plan.md §5.9,2026-09-09)。

工业界口径(§5.9 调研):灯态是**独立时序语义层**,与检测框/静态地图分离。
两种模式:
  记录(默认):不动灯,逐帧记录 CARLA 自身周期(含 elapsed_s)
  受控(--cycle 6,2,6):freeze 全图灯 + 按计划切灯 → **确定性变灯序列**
    (真实数据集最缺"变灯时刻"样本:WOMD 原始数据 71.7% 状态缺失/unknown)

输出(KITTI root 扩展,与帧对齐、ego 位姿锚定):
  training/image_2/{fid}.png           原图(与采集器同一相机链)
  training/overlay/{fid}.png           灯态 overlay(色点 + #id 状态 距离)
  training/traffic_light/{fid}.json    TrafficLightFrame(状态 + 管制车道 + 停车线)

用法(base env,CARLA 服务器运行中):
  python bin/collect_tl_states.py --frames 60 --speed 8
  python bin/collect_tl_states.py --frames 90 --speed 8 --cycle 6,2,6
"""

from __future__ import annotations

import argparse
import queue
from typing import cast

import carla
from carla_common import (
    CAM_ATTRS,
    SENSOR_OFFSET,
    draw_traffic_lights,
    loc,
    rad,
    spawn_ego,
    sync_mode,
    traffic_light_frame,
)
from PIL import Image

from autodrivedata.calib import CameraIntrinsics
from autodrivedata.paths import project_path
from autodrivedata.traffic_light import phase_at

DELTA = 0.1  # 同步模式固定步长(sync_mode 默认)
SPEED_DEFAULT = 8.0
HORIZON_DEFAULT = 120.0  # 灯态视距(灯比 GT 框看得远,且不靠传感器)


def parse_cycle(text: str) -> tuple[tuple[str, float], ...]:
    """'6,2,6' → (('Green',6.0),('Yellow',2.0),('Red',6.0))。"""
    try:
        secs = [float(x) for x in text.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--cycle 需 绿,黄,红 三个秒数: {text}") from e
    if len(secs) != 3 or any(s <= 0 for s in secs):
        raise argparse.ArgumentTypeError(f"--cycle 需 绿,黄,红 三个正数: {text}")
    return tuple(zip(("Green", "Yellow", "Red"), secs, strict=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--out", default="outputs/kitti_tl_demo")
    ap.add_argument("--speed", type=float, default=SPEED_DEFAULT, help="ego 定速直行 m/s")
    ap.add_argument("--cycle", type=parse_cycle, default=None, help="受控切灯:绿,黄,红 秒数")
    ap.add_argument("--horizon", type=float, default=HORIZON_DEFAULT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()
    plan: tuple[tuple[str, float], ...] = args.cycle or ()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)

    # 清场 + 锚定 pt0 固有 rotation(同 collect_static_gt:轨迹可复现、沿车道朝向)
    for a in world.get_actors():
        if a.type_id.startswith(("vehicle", "walker", "controller")):
            a.destroy()
    for _ in range(3):
        world.tick()
    ego = spawn_ego(world)
    ego.set_autopilot(False)
    pts = world.get_map().get_spawn_points()
    ego.set_transform(carla.Transform(pts[0].location, pts[0].rotation))
    world.tick()
    here = ego.get_location()
    if here.distance(pts[0].location) > 1.0:
        raise RuntimeError(f"ego 未能锚定 pts[0]: 落在 {here}")
    print(f"[ego] 锚定 pts[0] @ ({here.x:.1f}, {here.y:.1f}) yaw={ego.get_transform().rotation.yaw:.1f}°")

    lights = [cast(carla.TrafficLight, a) for a in world.get_actors().filter("traffic.traffic_light")]
    if not lights:
        raise RuntimeError("本图无信号灯 actor,灯态 GT 无内容可采")
    print(f"[lights] 全图 {len(lights)} 个信号灯 actor")
    if plan:
        for light in lights:
            light.freeze(True)  # 停掉 CARLA 自动周期,改由本脚本按计划驱动
        print(f"[cycle] 受控切灯 {plan}(freeze 全图灯)")

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    for t, v in CAM_ATTRS.items():
        cam_bp.set_attribute(t, v)
    camera = cast(carla.Sensor, world.spawn_actor(cam_bp, SENSOR_OFFSET, attach_to=ego))
    q: queue.Queue = queue.Queue()
    camera.listen(q.put)
    k = CameraIntrinsics(
        width=int(CAM_ATTRS["image_size_x"]),
        height=int(CAM_ATTRS["image_size_y"]),
        fov_h_deg=float(CAM_ATTRS["fov"]),
    )

    out = project_path(args.out)
    for sub in ("image_2", "overlay", "traffic_light"):
        (out / "training" / sub).mkdir(parents=True, exist_ok=True)

    fwd = ego.get_transform().get_forward_vector()
    fwd_v = carla.Vector3D(x=fwd.x * args.speed, y=fwd.y * args.speed, z=0.0)
    last_phase = ""
    try:
        for i in range(args.frames):
            if plan:
                # 先切灯再 tick:帧内记录的灯态与视觉一致(切灯在下一 tick 生效)
                ph = phase_at(i * DELTA, plan)
                if ph != last_phase:
                    for light in lights:
                        light.set_state(getattr(carla.TrafficLightState, ph))
                    last_phase = ph
            ego.set_target_velocity(fwd_v)
            world.tick()
            image: carla.Image = q.get(timeout=10)

            fid = f"{i:06d}"
            ego_t = ego.get_transform()
            frame = traffic_light_frame(world, fid, loc(ego_t), float(ego_t.rotation.yaw), args.horizon, plan)
            (out / "training/traffic_light" / f"{fid}.json").write_text(frame.to_json())

            png = out / "training/image_2" / f"{fid}.png"
            image.save_to_disk(str(png))
            cam_t = camera.get_transform()
            img = Image.open(png).convert("RGB")
            draw_traffic_lights(img, frame, loc(cam_t), rad(cam_t.rotation), k).save(
                out / "training/overlay" / f"{fid}.png"
            )

            if (i + 1) % 10 == 0 or i == args.frames - 1:
                hist: dict[str, int] = {}
                for light in frame.lights:
                    hist[light.state] = hist.get(light.state, 0) + 1
                print(f"[frame {i + 1}/{args.frames}] 视距内灯 {len(frame.lights)} {hist}")
    finally:
        camera.stop()
        camera.destroy()
        for light in lights:
            light.freeze(False)
            light.reset_group()
        for a in world.get_actors():
            if a.type_id.startswith(("vehicle", "walker", "controller")):
                a.destroy()
    print(f"[done] 灯态 GT root: {out.resolve()} ({args.frames} frames)")


if __name__ == "__main__":
    main()

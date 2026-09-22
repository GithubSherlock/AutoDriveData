"""P-G 教程 3DGS 采集:CARLA 静态场景 360° 环绕 → outputs/3dgs/capture/。

工业口径:3D Gaussian Splatting 训练需多视角图像 + 相机位姿。本脚本:
- 在选定观测点做**360° 环绕**(步进角 = 360/n_cams),采集 RGB + 深度
  (深度用于近端验证;3DGS 训练只用 RGB + 位姿)
- **多俯仰**(--pitches "0,-15,-30"):同一环绕圈跑多组相机俯仰,俯角补顶面/近地
  隐面,提升重建覆盖。落盘按 pitch 分目录 images/p{p}/ + poses_{p}.json
- 相机位姿直接取 CARLA 真值(与 pycolmap 建图对照;未知姿态时用真值初始化)
- 静态场景(无 NPC/车流)——Gaussian 重建要求被摄物不动

实现纪律:
- **单相机逐帧移动**(不批量 spawn 多传感器):CARLA 不允许"无父 actor 的悬空
  sensor",批 spawn 裸 camera 会触发 terminate w/o active exception。逐帧
  set_transform 单挂传感器(attach 到静态 spectator)最稳。
- 采集时用 ego 静态锚点(spectator 不可 attach),故相机直接贴 spectator
  逐帧移动位姿。位姿几何抽成纯函数 autodrivedata/collect_rig.ring_cam_pose。

落盘:
  outputs/3dgs/capture/
    images/p{p}/{i:05d}.png      # 每俯仰一圈独立子目录(i 全局序号)
    depth/p{p}/{i:05d}.npy       # 深度米,验证锚点
    poses_{p}.json               # [{i, loc(x,y,z), rot(pitch,yaw,roll)}]
    pitches.json                 # 本批采集的 pitches 列表(供训练侧枚举)

用法:
  python bin/collect_3dgs.py [--center-index 77] [--n-cams 90] [--radius 6] [--pitches "0,-15,-30"]
"""

from __future__ import annotations

import argparse
import json
import queue
from typing import cast

import carla
import numpy as np
from carla_common import CAM_ATTRS, sync_mode

from autodrivedata.collect_rig import ring_cam_pose
from autodrivedata.depth_codec import decode_depth
from autodrivedata.paths import project_path


def _depth_img_to_meter(dep: carla.Image) -> np.ndarray:
    """CARLA 深度图 → 深度米 (H,W)(**委托** `depth_codec.decode_depth`,同一口径)。"""
    return decode_depth(dep.raw_data, dep.height, dep.width)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/3dgs")
    ap.add_argument("--center-index", type=int, default=77, help="观测中心取第 N 个 spawn point")
    ap.add_argument("--radius", type=float, default=6.0, help="环绕半径(米)")
    ap.add_argument("--pitches", default="0,-15,-30", help="相机俯仰序列(度,逗号分隔,负=朝下)")
    ap.add_argument("--n-cams", type=int, default=90, help="环绕相机数(步进 = 360/n)")
    ap.add_argument("--frames", type=int, default=90, help="每俯仰采集帧数(=相机数)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    pitches = [float(x) for x in args.pitches.split(",")]

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)

    pts = world.get_map().get_spawn_points()
    center = pts[args.center_index % len(pts)].location
    print(f"[3dgs] 中心点 {center} (spawn {args.center_index}) | 半径 {args.radius} | 俯仰 {pitches}°")

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    for k, v in CAM_ATTRS.items():
        cam_bp.set_attribute(k, v)
    depth_bp = bp_lib.find("sensor.camera.depth")
    for k, v in CAM_ATTRS.items():
        depth_bp.set_attribute(k, v)

    # 单相机逐帧移动:attach 到 spectator(静态锚点),逐帧 set_transform。
    # 关键坑(已实测):attach 子 actor 的 set_transform 是**相对父 actor**的位姿。
    # 若不把 spectator 先挪到环绕中心,传世界坐标 = 相对 spectator 默认位置偏移,
    # 环绕实际绕在空场地上(曾致整段采集低纹理、PSNR ~10)。必须先归位 spectator。
    spec = world.get_spectator()
    spec.set_transform(carla.Transform(center + carla.Location(0, 0, 1.5), carla.Rotation(0.0, 0.0, 0.0)))
    world.tick()
    cam = cast(carla.Sensor, world.spawn_actor(cam_bp, carla.Transform(), attach_to=spec))
    dep = cast(carla.Sensor, world.spawn_actor(depth_bp, carla.Transform(), attach_to=spec))
    q: queue.Queue = queue.Queue()
    dq: queue.Queue = queue.Queue()
    cam.listen(q.put)
    dep.listen(dq.put)

    out = project_path(args.out) / "capture"
    for p in pitches:
        (out / "images" / f"p{int(p)}").mkdir(parents=True, exist_ok=True)
        (out / "depth" / f"p{int(p)}").mkdir(parents=True, exist_ok=True)
    (out / "pitches.json").write_text(json.dumps(pitches, indent=1), encoding="utf-8")

    world.tick()
    for _ in range(3):
        world.tick()
        q.get(timeout=10)
        dq.get(timeout=10)

    for p in pitches:
        p_poses: list[dict] = []
        for i in range(args.frames):
            x, y, z, yaw_cam = ring_cam_pose(
                center.x, center.y, center.z, args.radius, i, args.n_cams, height=1.5
            )
            tf = carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch=p, yaw=yaw_cam, roll=0.0))
            cam.set_transform(tf)
            dep.set_transform(tf)
            world.tick()
            img: carla.Image = q.get(timeout=10)
            tmp = out / f".tmp_{int(p)}_{i}.png"
            img.save_to_disk(str(tmp))
            tmp.rename(out / "images" / f"p{int(p)}" / f"{i:05d}.png")
            depth: carla.Image = dq.get(timeout=10)
            np.save(out / "depth" / f"p{int(p)}" / f"{i:05d}.npy", _depth_img_to_meter(depth))
            p_poses.append(
                {
                    "i": i,
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "z": round(z, 3),
                    "pitch": p,
                    "yaw": round(yaw_cam, 3),
                    "roll": 0.0,
                }
            )
            if (i + 1) % 20 == 0 or i == args.frames - 1:
                print(f"[cam {i + 1}/{args.frames}] pitch {p}° yaw {yaw_cam:.1f}° @ ({x:.1f}, {y:.1f})")
        (out / f"poses_{int(p)}.json").write_text(json.dumps(p_poses, indent=1), encoding="utf-8")
        print(f"[pitch {p}°] done, {args.frames} 相机")

    cam.destroy()
    dep.destroy()
    print(f"[done] 3dgs capture: {out.resolve()} ({len(pitches)} 俯仰 × {args.frames} 环绕相机)")


if __name__ == "__main__":
    main()

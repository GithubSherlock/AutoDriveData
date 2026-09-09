"""M0 smoke 测试:CARLA headless 连接 → 同步模式 → 相机 RGB + LiDAR 各取一帧 → 落盘 + BEV 可视化。

用法: python smoke.py [--host 127.0.0.1] [--port 2000] [--out outputs/smoke]
"""

from __future__ import annotations

import argparse
import queue
from pathlib import Path

import carla
import numpy as np

CAM_ATTRS = {"image_size_x": "1242", "image_size_y": "375", "fov": "90"}  # 对齐 KITTI 口径
LIDAR_ATTRS = {
    "channels": "32",
    "range": "70",
    "points_per_second": "100000",
    "rotation_frequency": "10",
    "upper_fov": "10.0",
    "lower_fov": "-30.0",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=2000, type=int)
    ap.add_argument("--out", default="outputs/smoke")
    # 相机:位置偏移(相对出生点,x 前/y 右/z 上)、俯仰角、水平角偏移、视场角
    ap.add_argument("--cam-offset", default="-8,0,4", help="相对出生点的相机偏移 x,y,z(米)")
    ap.add_argument("--pitch", default=-15.0, type=float, help="相机俯仰角(负=低头)")
    ap.add_argument("--yaw-offset", default=0.0, type=float, help="相对出生点朝向的水平旋转")
    ap.add_argument("--fov", default="90", help="相机水平视场角(度),KITTI 约 53")
    # LiDAR:线数、探测距离、每秒点数(每帧点数 = pps / rotation_frequency)
    ap.add_argument("--channels", default="32", help="LiDAR 线数 32/64/128")
    ap.add_argument("--lidar-range", default="70", help="LiDAR 最大探测距离(米)")
    ap.add_argument("--pps", default="100000", help="每秒点数(线数翻倍建议同步翻倍)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    print(f"[1] server version: {client.get_server_version()}")

    world = client.get_world()
    print(f"[2] map: {world.get_map().name}")

    # 同步模式:固定 tick,传感器帧与仿真严格对齐(采集管线的标准姿势)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()

    # 相机:属性必须设在其 blueprint 上(spawn 之前),spawn 后的 actor 只读
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("fov", args.fov)
    for k, v in CAM_ATTRS.items():
        if k != "fov":
            cam_bp.set_attribute(k, v)
    spawn_pt = world.get_map().get_spawn_points()[0]
    ox, oy, oz = (float(v) for v in args.cam_offset.split(","))
    cam_loc = spawn_pt.location + carla.Location(x=ox, y=oy, z=oz)
    cam_rot = carla.Rotation(pitch=args.pitch, yaw=spawn_pt.rotation.yaw + args.yaw_offset)
    camera = world.spawn_actor(cam_bp, carla.Transform(cam_loc, cam_rot))
    print(f"[3] camera spawned at {cam_loc} (fov={args.fov}, pitch={args.pitch})")

    # LiDAR:线数/距离/点数密度按 CLI 参数(属性同样设 blueprint)
    lid_bp = bp_lib.find("sensor.lidar.ray_cast")
    lid_bp.set_attribute("channels", args.channels)
    lid_bp.set_attribute("range", args.lidar_range)
    lid_bp.set_attribute("points_per_second", args.pps)
    for k, v in LIDAR_ATTRS.items():
        if k not in ("channels", "range", "points_per_second"):
            lid_bp.set_attribute(k, v)
    lidar = world.spawn_actor(lid_bp, carla.Transform(cam_loc, cam_rot))
    print(f"[4] lidar spawned ({args.channels}ch, range {args.lidar_range}m, {args.pps} pps)")

    img_q: queue.Queue = queue.Queue()
    lid_q: queue.Queue = queue.Queue()
    camera.listen(img_q.put)
    lidar.listen(lid_q.put)

    world.tick()
    world.tick()

    image = img_q.get(timeout=10)
    pts = lid_q.get(timeout=10)

    img_path = out / "smoke_camera.png"
    image.save_to_disk(str(img_path))

    arr = np.frombuffer(pts.raw_data, dtype=np.float32).reshape(-1, 4)  # x,y,z,intensity
    bin_path = out / "smoke_lidar.bin"
    arr.tofile(bin_path)
    print(f"[5] image -> {img_path}  ({image.width}x{image.height})")
    print(
        f"[6] lidar -> {bin_path}  ({len(arr)} points, 强度范围 [{arr[:, 3].min():.2f},{arr[:, 3].max():.2f}])"
    )

    # BEV 散点可视化:CARLA LiDAR 传感器系 = x 前 / y 右 / z 上,俯视图取 (x, y)
    bev = arr[:, [0, 1]]
    mask = (np.abs(bev[:, 0]) < 40) & (np.abs(bev[:, 1]) < 40)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(bev[mask, 0], bev[mask, 1], s=0.05, c=arr[mask, 3], cmap="plasma")
    ax.set_aspect("equal")
    ax.set_title(f"LiDAR BEV ({len(arr)} pts)")
    bev_path = out / "smoke_bev.png"
    fig.savefig(bev_path, dpi=120)
    print(f"[7] bev plot -> {bev_path}")

    camera.stop()
    lidar.stop()
    camera.destroy()
    lidar.destroy()
    print("[8] DONE")


if __name__ == "__main__":
    main()

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
    for k, v in CAM_ATTRS.items():
        cam_bp.set_attribute(k, v)
    spawn_pt = world.get_map().get_spawn_points()[0]
    cam_loc = spawn_pt.location + carla.Location(x=-8.0, z=4.0)
    cam_rot = carla.Rotation(pitch=-15, yaw=spawn_pt.rotation.yaw)
    camera = world.spawn_actor(cam_bp, carla.Transform(cam_loc, cam_rot))
    print(f"[3] camera spawned at {cam_loc}")

    # LiDAR:同位置,32 线(属性同样设 blueprint)
    lid_bp = bp_lib.find("sensor.lidar.ray_cast")
    for k, v in LIDAR_ATTRS.items():
        lid_bp.set_attribute(k, v)
    lidar = world.spawn_actor(lid_bp, carla.Transform(cam_loc, cam_rot))
    print("[4] lidar spawned (32ch)")

    img_q: "queue.Queue" = queue.Queue()
    lid_q: "queue.Queue" = queue.Queue()
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
    print(f"[6] lidar -> {bin_path}  ({len(arr)} points, 强度范围 [{arr[:,3].min():.2f},{arr[:,3].max():.2f}])")

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

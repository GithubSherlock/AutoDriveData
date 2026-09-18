"""P-F 双目采集:CARLA 双目 rig(基线 0.4m)+ 真值深度 → outputs/stereo/。

工业口径:双目 = 水平基线两相机同时采集左右图。本脚本提供:

- 双目 rig:左/右相机沿 **y 轴 ±0.2m**(乘用车基线 ≈ 0.4m),同为前视(yaw 0),
  其余与 CAM_FRONT(collect_drive 车顶)一致。
- **真值深度**:`sensor.camera.depth` 逐像素(对齐左相机,同一挂点)——
  注意 CARLA 深度是"归一化 0-1 ×1000m",解码回米:z = R×G×B / 255 × 1000。
  给 SGM 验证锚点(双目三角测量 vs 真值)。
- ego 定速直行(yaw=0 车道基线,定速语义与 collect_drive 一致)。
- 前向单人现场(无 NPC)——低纹理/遮挡最少,立体匹配最干净。

落盘:
  outputs/stereo/
    left/000000.png  right/000000.png  depth/000000.npy(depth 米,HxW)
    calib.json      — {baseline, intrinsic(3x3), cam_h_m}
    ego_pose.json   — 逐帧 ego 位姿

用法:
  python bin/collect_stereo.py [--frames 60] [--baseline 0.4]
"""

from __future__ import annotations

import argparse
import json
import queue
from typing import cast

import carla
import numpy as np
from carla_common import CAM_ATTRS, spawn_ego, sync_mode

from autodrivedata.collect_rig import stereo_rig_offsets
from autodrivedata.paths import project_path


def _depth_to_meter(image: carla.Image) -> np.ndarray:
    """CARLA 深度编码(BGR 各 8bit,归一化 ×1000m)→ 深度米 (H,W) float32。"""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    b, g, r = (
        arr[:, :, 0].astype(np.float32),
        arr[:, :, 1].astype(np.float32),
        arr[:, :, 2].astype(np.float32),
    )
    # 归一化 z =  B + G*256 + R*256*256 / 256^3,乘 1000 → 米
    z = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0**3 - 1.0) * 1000.0
    return z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/stereo")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--baseline", type=float, default=0.4, help="双目基线(米)")
    ap.add_argument("--speed", type=float, default=8.0, help="定速直行(米/秒)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)

    ego = spawn_ego(world)
    # 定速直行(collect_ab_route 同口径):先解制动残留,再每 tick 强设速度
    ego.apply_control(carla.VehicleControl())
    fwd = ego.get_transform().get_forward_vector()
    fwd_v = carla.Vector3D(x=fwd.x * args.speed, y=fwd.y * args.speed, z=0.0)

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    for k, v in CAM_ATTRS.items():
        cam_bp.set_attribute(k, v)
    depth_bp = bp_lib.find("sensor.camera.depth")
    for k, v in CAM_ATTRS.items():
        depth_bp.set_attribute(k, v)

    # 左/右相机:同一朝向(yaw=0),沿 y 轴 ±half(车载双目基线);挂点抽成纯函数
    off_l, off_r = stereo_rig_offsets(args.baseline)
    tf_l = carla.Transform(carla.Location(*off_l), carla.Rotation(0.0, 0.0, 0.0))
    tf_r = carla.Transform(carla.Location(*off_r), carla.Rotation(0.0, 0.0, 0.0))
    left_cam = cast(carla.Sensor, world.spawn_actor(cam_bp, tf_l, attach_to=ego))
    right_cam = cast(carla.Sensor, world.spawn_actor(cam_bp, tf_r, attach_to=ego))
    # 真值深度对齐左相机(同一挂点)
    depth_cam = cast(carla.Sensor, world.spawn_actor(depth_bp, tf_l, attach_to=ego))

    ql: queue.Queue = queue.Queue()
    qr: queue.Queue = queue.Queue()
    qd: queue.Queue = queue.Queue()
    left_cam.listen(ql.put)
    right_cam.listen(qr.put)
    depth_cam.listen(qd.put)

    for _ in range(5):  # 预热
        world.tick()
        ql.get(timeout=10)
        qr.get(timeout=10)
        qd.get(timeout=10)

    w, h = int(CAM_ATTRS["image_size_x"]), int(CAM_ATTRS["image_size_y"])
    import math

    fx = w / 2 / math.tan(math.radians(float(CAM_ATTRS["fov"]) / 2))
    intrinsic = [[fx, 0.0, w / 2], [0.0, fx, h / 2], [0.0, 0.0, 1.0]]
    calib = {
        "baseline": args.baseline,
        "focal_px": fx,
        "intrinsic": intrinsic,
        "cam_h_m": 1.65,
        "map": world.get_map().name,
    }
    out = project_path(args.out)
    for sub in ("left", "right", "depth"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "calib.json").write_text(json.dumps(calib, indent=1), encoding="utf-8")

    t0 = __import__("time").monotonic()
    poses: list[dict] = []
    try:
        for i in range(args.frames):
            ego.set_target_velocity(fwd_v)  # 定速(collect_ab_route 同口径,红线)
            world.tick()
            img_l: carla.Image = ql.get(timeout=10)
            img_r: carla.Image = qr.get(timeout=10)
            depth: carla.Image = qd.get(timeout=10)

            for img, name in ((img_l, "left"), (img_r, "right")):
                tmp = out / f".tmp_{i}_{name}.png"
                img.save_to_disk(str(tmp))
                tmp.rename(out / name / f"{i:06d}.png")
            np.save(out / "depth" / f"{i:06d}.npy", _depth_to_meter(depth))

            egot = ego.get_transform()
            v = ego.get_velocity()
            poses.append(
                {
                    "frame": i,
                    "x": round(egot.location.x, 3),
                    "y": round(egot.location.y, 3),
                    "z": round(egot.location.z, 3),
                    "yaw": round(egot.rotation.yaw, 3),
                    "speed": round(float(np.hypot(v.x, v.y)), 2),
                }
            )
            if (i + 1) % 10 == 0 or i == args.frames - 1:
                dt = __import__("time").monotonic() - t0
                print(
                    f"[frame {i + 1}/{args.frames}] speed {poses[-1]['speed']} m/s | {(i + 1) / dt:.1f} fps"
                )
    finally:
        (out / "ego_pose.json").write_text(json.dumps(poses, indent=1), encoding="utf-8")
        for s in (left_cam, right_cam, depth_cam):
            s.stop()
            s.destroy()
        for a in world.get_actors():
            if (
                a.type_id.startswith("vehicle")
                or a.type_id.startswith("walker")
                or a.type_id.startswith("controller")
            ):
                a.destroy()
    print(f"[done] stereo root: {out.resolve()} ({args.frames} frames)")


if __name__ == "__main__":
    main()

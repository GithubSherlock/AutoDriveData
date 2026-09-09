"""场景可视化实时流:真 UE 渲染 + GT overlay → 浏览器 MJPEG(base env)。

为什么不用 CarlaViz / RViz2(2026-09-09 决策,Plan.md §5.8):两者都不是 UE
渲染(Three.js 线框 / RViz 点云 marker),而本项目 P1 的验证对象全是渲染效果
(逆光过曝、雨夜对比度、浓雾);且 carlaviz 官方只到 0.9.15、AutoDL 容器无
docker,ROS2 路线要容器/VM/Mac 三系统联调。本脚本消费与采集器同一条相机链
→ 所见即落盘,且 GT 框走 label_2 同一投影口径(box_to_gt_line)。

用法(base env,CARLA 服务器运行中):
  python scripts/view_stream.py --view follow                 # 跟车视角
  python scripts/view_stream.py --view top --map Town13       # 俯视(看街区/NPC)
  python scripts/view_stream.py --view grid6 --npcs           # nuScenes 6 视角 + 静置 NPC
  python scripts/view_stream.py --scene rain_night --speed 8  # 带天气 + 定速直行
本地:ssh -L 8080:127.0.0.1:8080 <autodl> → 浏览器 http://127.0.0.1:8080

红线:同步模式下 tick 归本脚本,不能与采集脚本同时运行(抢 tick)。
服务只绑 127.0.0.1(经 SSH 隧道访问,不暴露公网)。
"""

from __future__ import annotations

import argparse
import io
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import carla
import numpy as np
from carla_common import (
    SENSOR_OFFSET,
    draw_traffic_lights,
    loc,
    rad,
    spawn_ego,
    spawn_npcs,
    sync_mode,
    traffic_light_frame,
)
from PIL import Image, ImageDraw

from autodrivedata.calib import CameraIntrinsics
from autodrivedata.gt import ActorBox, box_to_gt_line
from autodrivedata.scenarios import SCENES, merged_weather

VIEWS = ("follow", "top", "grid6")
MAX_DISTANCE = 65.0  # 与采集器 GT 口径一致(Plan.md 红线:远距无点框剔除)

# nuScenes 6 视角(同 collect_nus;grid6 按 3×2 拼图)
CAM_YAW_OFFSET = {
    "CAM_FRONT": 0.0,
    "CAM_FRONT_LEFT": 55.0,
    "CAM_FRONT_RIGHT": -55.0,
    "CAM_BACK": 180.0,
    "CAM_BACK_LEFT": 125.0,
    "CAM_BACK_RIGHT": -125.0,
}
CLASS_COLOR = {
    "Car": (0, 255, 80),
    "Pedestrian": (0, 220, 255),
    "Cyclist": (255, 220, 0),
    "Truck": (255, 140, 0),
    "Van": (255, 140, 0),
    "Misc": (170, 170, 170),
}
PAGE = (
    "<!doctype html><meta charset='utf-8'><title>AutoDriveData 实时视图</title>"
    "<body style='margin:0;background:#111;color:#666;font:12px monospace'>"
    "<img src='/stream' style='width:100%;height:auto;display:block'>"
)


def image_to_pil(image: carla.Image) -> Image.Image:
    """carla.Image(BGRA/BGR)→ PIL RGB(不落盘)。

    通道数由 raw_data 长度反推:carla pyi 桩缺 image.channels(T 类坑)。
    """
    ch = len(image.raw_data) // (image.width * image.height)
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, ch)
    return Image.fromarray(arr[:, :, [2, 1, 0]])


def encode_jpeg(img: Image.Image, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def actor_boxes(world: carla.World) -> list[ActorBox]:
    """世界内全部 vehicle/walker → 纯值 ActorBox(GT 框来源)。"""
    out: list[ActorBox] = []
    for a in world.get_actors():
        if not (a.type_id.startswith("vehicle") or a.type_id.startswith("walker")):
            continue
        bb = a.bounding_box
        t = a.get_transform()
        out.append(
            ActorBox(
                type_id=a.type_id,
                extent=(bb.extent.x, bb.extent.y, bb.extent.z),
                location=(bb.location.x, bb.location.y, bb.location.z),
                rotation=rad(bb.rotation),
                actor_location=loc(t),
                actor_rotation=rad(t.rotation),
            )
        )
    return out


def overlay_gt(
    img: Image.Image,
    boxes: list[ActorBox],
    cam_loc: tuple[float, float, float],
    cam_rot: tuple[float, float, float],
    k: CameraIntrinsics,
) -> Image.Image:
    """GT 框:走 box_to_gt_line(label_2 同一口径)→ 所见即导出。"""
    d = ImageDraw.Draw(img)
    for box in boxes:
        line = box_to_gt_line(box, cam_loc, cam_rot, k, max_distance=MAX_DISTANCE)
        if not line:
            continue
        p = line.split()
        cls = p[0]
        x1, y1, x2, y2 = (float(v) for v in p[4:8])
        dist = float(np.hypot(float(p[11]), float(p[13])))  # 相机系 (x, z)
        col = CLASS_COLOR.get(cls, CLASS_COLOR["Misc"])
        d.rectangle([x1, y1, x2, y2], outline=col, width=2)
        d.text((x1 + 2, max(0.0, y1 - 11)), f"{cls} {dist:.0f}m", fill=col)
    return img


def build_cameras(
    world: carla.World, ego: carla.Vehicle, view: str, width: int, height: int
) -> dict[str, tuple[carla.Sensor, CameraIntrinsics]]:
    """视角 → {名称: (相机, 内参)};grid6 用 nuScenes 6 向。"""
    bp_lib = world.get_blueprint_library()
    k = CameraIntrinsics(width=width, height=height, fov_h_deg=90.0)
    cams: dict[str, tuple[carla.Sensor, CameraIntrinsics]] = {}

    def add(name: str, tf: carla.Transform) -> None:
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", "90")
        cams[name] = (cast(carla.Sensor, world.spawn_actor(bp, tf, attach_to=ego)), k)

    if view == "grid6":
        for name, yaw_off in CAM_YAW_OFFSET.items():
            add(
                name,
                carla.Transform(
                    SENSOR_OFFSET.location,
                    carla.Rotation(pitch=0.0, yaw=yaw_off, roll=0.0),
                ),
            )
    elif view == "top":
        # 60m 高:覆盖 ±34m(纵向)× ±60m(横向),容得下 A/B 布局的 20~62m 静置车
        add(
            "TOP",
            carla.Transform(carla.Location(x=0.0, y=0.0, z=60.0), carla.Rotation(pitch=-90.0)),
        )
    else:  # follow
        add(
            "FOLLOW",
            carla.Transform(carla.Location(x=-9.0, y=0.0, z=4.5), carla.Rotation(pitch=-12.0)),
        )
    return cams


def drain(q: queue.Queue) -> carla.Image:
    """取最新一帧(丢弃积压),防渲染慢于 tick 时画面滞后。"""
    img: carla.Image = q.get(timeout=10)
    while not q.empty():
        img = q.get_nowait()
    return img


class FrameSlot:
    """最新帧槽:MJPEG 服务线程只读最新帧,不排队 → 客户端永远看最新画面。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._seq = 0

    def publish(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._seq += 1

    def read(self, last_seq: int) -> tuple[bytes | None, int]:
        with self._lock:
            if self._seq == last_seq:
                return None, last_seq
            return self._jpeg, self._seq


def start_server(slot: FrameSlot, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            del format, args  # 静音 access log(5fps 刷屏);签名照基类(参数名 format)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/stream":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            while True:
                jpeg, seq = slot.read(last)
                if jpeg is None:
                    time.sleep(0.02)
                    continue
                last = seq
                try:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break  # 客户端断开

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def compose_grid(tiles: list[Image.Image], names: list[str], w: int, h: int) -> Image.Image:
    grid = Image.new("RGB", (w * 3, h * 2), (0, 0, 0))
    d = ImageDraw.Draw(grid)
    for idx, (tile, name) in enumerate(zip(tiles, names)):
        row, col = divmod(idx, 3)
        grid.paste(tile, (col * w, row * h))
        d.text((col * w + 6, row * h + 6), name, fill=(255, 255, 0))
    return grid


def dump_pair(path: str, raw: Image.Image, over: Image.Image) -> None:
    """落盘同一帧的 raw + overlay → 差集 = 真实绘制像素(数值诊断)。

    场景自带绿色植被/黄色标线,直接数颜色会误判 overlay 是否画上;
    两帧做差才排得掉干扰。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw.save(p.with_name(f"{p.stem}_raw{p.suffix}"))
    over.save(p)
    print(f"[dump] {p} + {p.stem}_raw{p.suffix}(差集诊断用)")


def draw_hud(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 * len(text) + 8, 16], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", choices=VIEWS, default="follow")
    ap.add_argument("--map", default=None, help="加载地图(默认服务器当前图)")
    ap.add_argument("--scene", default=None, choices=sorted(SCENES), help="天气档")
    ap.add_argument("--npcs", action="store_true", help="ego 前方摆 6 个静置 NPC")
    ap.add_argument("--speed", type=float, default=0.0, help="ego 定速直行 m/s(0=静止)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--width", type=int, default=640, help="单相机/拼图单元宽")
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=0.0, help="秒;0 = 常驻(Ctrl-C 退出)")
    ap.add_argument("--dump", default=None, help="落盘首帧 PATH(overlay)+ PATH_raw(诊断)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--sim-port", type=int, default=2000)
    args = ap.parse_args()

    client = carla.Client(args.host, args.sim_port)
    client.set_timeout(60.0)
    if args.map:
        print(f"[map] 加载 {args.map} ...")
        client.set_timeout(300.0)
        world = client.load_world(args.map)
        client.set_timeout(60.0)
    else:
        world = client.get_world()
    print(f"[map] {world.get_map().name}")

    sync_mode(world)
    cleared = 0
    for a in world.get_actors():
        if a.type_id.startswith(("vehicle", "walker", "controller")):
            a.destroy()
            cleared += 1
    for _ in range(3):
        world.tick()
    if cleared:
        print(f"[clear] 清场 {cleared} 个残留 actor")

    if args.scene:
        scene = SCENES[args.scene]
        world.set_weather(carla.WeatherParameters(**merged_weather(scene)))
        print(f"[scene] {scene.name} — {sorted(scene.weather)}")

    ego = spawn_ego(world)
    ego.set_autopilot(False)
    if args.npcs:
        spawn_npcs(world, ego.get_transform())
    fwd = ego.get_transform().get_forward_vector()
    vel = carla.Vector3D(x=fwd.x * args.speed, y=fwd.y * args.speed, z=0.0)

    cams = build_cameras(world, ego, args.view, args.width, args.height)
    queues: dict[str, queue.Queue] = {name: queue.Queue() for name in cams}
    for name, (cam, _) in cams.items():
        cam.listen(queues[name].put)
    print(f"[sensor] {len(cams)} 相机 @ {args.width}x{args.height}")

    slot = FrameSlot()
    srv = start_server(slot, args.port)
    print(
        f"[stream] http://127.0.0.1:{args.port}  "
        f"(本地:ssh -L {args.port}:127.0.0.1:{args.port} <autodl>)"
    )

    t_end = time.time() + args.duration if args.duration > 0 else float("inf")
    frames, t_report = 0, time.time()
    dumped = False  # frames 每 5s 被 FPS 统计清零,不能用它判"首帧"
    try:
        while time.time() < t_end:
            t0 = time.time()
            if args.speed:
                ego.set_target_velocity(vel)  # 定速(同 collect_ab_route;不走 TM)
            world.tick()
            boxes = actor_boxes(world)
            ego_t = ego.get_transform()
            # 灯态走与采集器同一条实现(carla_common.traffic_light_frame)
            tl_frame = traffic_light_frame(
                world, f"{frames:06d}", loc(ego_t), float(ego_t.rotation.yaw)
            )

            tiles: list[Image.Image] = []
            raw_tiles: list[Image.Image] = []
            for name, (cam, k) in cams.items():
                cam_t = cam.get_transform()
                cam_loc, cam_rot = loc(cam_t), rad(cam_t.rotation)
                raw = image_to_pil(drain(queues[name]))
                img = overlay_gt(raw.copy(), boxes, cam_loc, cam_rot, k)
                img = draw_traffic_lights(img, tl_frame, cam_loc, cam_rot, k)
                raw_tiles.append(raw)
                tiles.append(img)

            frame_img = (
                compose_grid(tiles, list(cams), args.width, args.height)
                if args.view == "grid6"
                else tiles[0]
            )
            if args.dump and not dumped:
                raw_img = (
                    compose_grid(raw_tiles, list(cams), args.width, args.height)
                    if args.view == "grid6"
                    else raw_tiles[0]
                )
                dump_pair(args.dump, raw_img, frame_img.copy())
                dumped = True
            fps_now = frames / max(time.time() - t_report, 1e-6)
            draw_hud(
                frame_img,
                f"{world.get_map().name} | {args.view} | actors={len(boxes)} | "
                f"tl={len(tl_frame.lights)} | {fps_now:.1f}fps",
            )
            slot.publish(encode_jpeg(frame_img))
            frames += 1

            if time.time() - t_report >= 5.0:
                print(f"[stream] {fps_now:.1f} fps | actors={len(boxes)}")
                frames, t_report = 0, time.time()
            dt = time.time() - t0
            if dt < 1.0 / args.fps:
                time.sleep(1.0 / args.fps - dt)
    except KeyboardInterrupt:
        print("\n[stop] Ctrl-C")
    finally:
        srv.shutdown()
        for cam, _ in cams.values():
            cam.stop()
            cam.destroy()
        for a in world.get_actors():
            if a.type_id.startswith(("vehicle", "walker", "controller")):
                a.destroy()
        world.apply_settings(carla.WorldSettings())  # 恢复异步,防服务器冻结
        print("[done] 相机/actor 已清理,服务器恢复异步")


if __name__ == "__main__":
    main()

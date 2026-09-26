"""场景可视化实时流:真 UE 渲染 + GT overlay → 浏览器 MJPEG(autodrivedata env)。

为什么不用 CarlaViz / RViz2(2026-09-09 决策,Plan.md §5.8):两者都不是 UE
渲染(Three.js 线框 / RViz 点云 marker),而本项目 P1 的验证对象全是渲染效果
(逆光过曝、雨夜对比度、浓雾);且 carlaviz 官方只到 0.9.15、AutoDL 容器无
docker,ROS2 路线要容器/VM/Mac 三系统联调。本脚本消费与采集器同一条相机链
→ 所见即落盘,且 GT 框走 label_2 同一投影口径(box_to_gt_line)。

MapTR 实时预测 overlay(`--maptr-ckpt`):在同一 tick 的 6 路环视图上跑一次
前向 → 预测折线(ego 系)按 mapviz 同一投影链回投到各相机(品红),可选角落贴
BEV 面板。**rig 必须与权重训练数据逐字段对齐**(相机名→挂点平移/偏航 / 内参 / 分辨率),
故 rig 与 calib 一律走 `autodrivedata/sim/live_common.py` 的 `build_surround_rig` / `surround_calibs`
(只认 `live_common.rig_spec()` 的两处定义),并做启动自检 `rig_mount_deviation`(平移米 / 偏航度)。
**两代 rig 并存**:`nuscenes`(逐相机 `SENSOR_MOUNTS` + 官方 6DoF 姿态)对 `maptr_600`/`maptr_1000`;
`legacy`(共用 `SENSOR_OFFSET` + 235/125)对 `maptr_ep256`/`maptr_ep512` —— `--rig auto` 按权重名选,
**拿 nuscenes 喂 ep512 是错配**(见 `live_common` 头注对照表)。⚠️ 全部 MapTR 权重已标废弃
(2026-09-22:训练用的 `official` rig 偏航镜像),保留两代仅为兼容既有产物。

共享件(多槽 MJPEG / 拼图 / GT overlay / 环视 rig / 键盘)在 `autodrivedata/sim/live_common.py`,
8 路 studio 见 `autodrivedata/sim/live_studio.py`。

用法(CARLA 服务器运行中):
  python -m autodrivedata.sim.view_stream --view follow                 # 跟车视角
  python -m autodrivedata.sim.view_stream --view top --map Town13       # 俯视(看街区/NPC)
  python -m autodrivedata.sim.view_stream --view grid6 --npcs           # nuScenes 6 视角 + 静置 NPC
  python -m autodrivedata.sim.view_stream --scene rain_night --speed 8  # 带天气 + 定速直行
  python -m autodrivedata.sim.view_stream --view grid6 --maptr-ckpt outputs/maptr_ep512.pt --maptr-bev
本地:ssh -L 8080:127.0.0.1:8080 <autodl> → 浏览器 http://127.0.0.1:8080

红线:同步模式下 tick 归本脚本,不能与采集脚本同时运行(抢 tick)。
服务只绑 127.0.0.1(经 SSH 隧道访问,不暴露公网)。
"""

from __future__ import annotations

import argparse
import queue
import time

import carla
import numpy as np
from PIL import Image, ImageDraw

from autodrivedata.map.mapviz import PRED_COLOR, bev_panel, draw_projected_lines
from autodrivedata.paths import project_path
from autodrivedata.sim.carla_common import (
    draw_traffic_lights,
    loc,
    rad,
    spawn_ego,
    spawn_npcs,
    sync_mode,
    traffic_light_frame,
)
from autodrivedata.sim.live_common import (
    FrameSlot,
    actor_boxes,
    build_cameras,
    build_surround_rig,
    compose_grid,
    drain,
    draw_hud,
    dump_pair,
    encode_jpeg,
    image_to_pil,
    load_maptr,
    maptr_predict,
    overlay_gt,
    resolve_rig,
    rig_frame,
    rig_mount_deviation,
    start_server,
    surround_calibs,
)
from autodrivedata.sim.scenarios import SCENES, merged_weather

VIEWS = ("follow", "top", "grid6")

# 纯显示用的 6 视角偏航(度)。**与采集口径 `camera_rig.NUS_CAMERA_RIG` 无关**:
# 早期显示口径(BACK_LEFT/RIGHT 用 125/−125),且不带头顶 pitch/roll。显示路径
# (不带 --maptr)沿用不变;带 --maptr 时 rig 与 calib 一律走 `live_common.build_surround_rig`
# / `surround_calibs`(只认 `rig_spec()` 的两处定义),不碰本表。
CAM_YAW_OFFSET = {
    "CAM_FRONT": 0.0,
    "CAM_FRONT_LEFT": 55.0,
    "CAM_FRONT_RIGHT": -55.0,
    "CAM_BACK": 180.0,
    "CAM_BACK_LEFT": 125.0,
    "CAM_BACK_RIGHT": -125.0,
}


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
    ap.add_argument("--maptr-ckpt", default=None, help="MapTR state_dict → 实时预测 overlay(需 --view grid6)")
    ap.add_argument("--maptr-thr", type=float, default=0.2, help="预测实例得分阈值(口径同 eval_maptr)")
    ap.add_argument(
        "--maptr-scale", type=float, default=0.5, help="预测时显示缩放(1242×375 全尺寸拼图带宽过大)"
    )
    ap.add_argument("--maptr-bev", action="store_true", help="右下角贴 BEV 面板(线上无地图 GT,只画预测)")
    ap.add_argument("--maptr-device", default=None, help="推理设备(默认 cuda 若可用;与 CARLA 共享 GPU)")
    ap.add_argument(
        "--rig",
        choices=("auto", "nuscenes", "legacy"),
        default="auto",
        help="环视挂点口径;auto 按权重名选(ep256/ep512=legacy,600/1000=nuscenes)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--sim-port", type=int, default=2000)
    args = ap.parse_args()
    if args.dump:
        args.dump = str(project_path(args.dump))  # 产物锚定项目根(相对路径不随 cwd 漂移)
    if args.maptr_ckpt and args.view != "grid6":
        raise SystemExit("--maptr-ckpt 需要 --view grid6:预测 overlay 的口径 = 6 路环视 rig")

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

    maptr = load_maptr(args.maptr_ckpt, args.maptr_device) if args.maptr_ckpt else None
    rig = resolve_rig(args.rig, args.maptr_ckpt)
    calibs: dict[str, dict] = {}
    if maptr:
        # rig 原生画幅(nuscenes = 1600×900,legacy = 1242×375)+ 逐通道 fov;显示侧只缩放
        rig_w, rig_h = rig_frame(rig)[:2]
        cams = build_surround_rig(world, ego, rig=rig)
        # K 必须与上面挂的那套同画幅(corner 主点 ⇒ 画幅不同则 K 不同)
        calibs = surround_calibs(rig, rig_w, rig_h)
        print(f"[rig] {rig} @ {rig_w}×{rig_h}(--rig {args.rig} → 权重 {args.maptr_ckpt})")
        disp_w = int(rig_w * args.maptr_scale)
        disp_h = int(rig_h * args.maptr_scale)
    elif args.view == "grid6":
        # 纯显示环视:同一 rig 定义(只认 rig_spec 的两处),只降分辨率省带宽
        cams = build_surround_rig(world, ego, args.width, args.height, rig=rig)
        disp_w, disp_h = args.width, args.height
    else:
        cams = build_cameras(world, ego, args.view, args.width, args.height)
        disp_w, disp_h = args.width, args.height
    queues: dict[str, queue.Queue] = {name: queue.Queue() for name in cams}
    for name, (cam, _) in cams.items():
        cam.listen(queues[name].put)
    if args.view == "grid6":
        # 传感器 transform 只在 tick 后刷新(C19 同类):tick 前读=全 0 陈旧值,
        # 自检会假报 ~179.8°(=CAM_BACK 规格 180 − ego 固有 yaw 0.159)
        world.tick()
        dev_t, dev_y = rig_mount_deviation(cams, ego, rig)
        print(f"[rig] {rig}:实挂相机 vs 该口径规格 最大偏差:平移 {dev_t:.3f} m / 偏航 {dev_y:.3f}°")
    print(f"[sensor] {len(cams)} 相机 @ {disp_w}x{disp_h}")

    slots = {"main": FrameSlot()}
    srv = start_server(slots, args.port)
    print(f"[stream] http://127.0.0.1:{args.port}  (本地:ssh -L {args.port}:127.0.0.1:{args.port} <autodl>)")

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
            tl_frame = traffic_light_frame(world, f"{frames:06d}", loc(ego_t), float(ego_t.rotation.yaw))

            # 先把本 tick 的 6 路图全部取齐(不跨 tick 混帧),再推理 → 同一帧做 overlay
            raw_by_name = {name: image_to_pil(drain(queues[name])) for name in cams}
            ego_g = [*loc(ego_t), ego_t.rotation.yaw, ego_t.rotation.pitch, ego_t.rotation.roll]
            preds: list[list[np.ndarray]] = []
            if maptr:
                model, dev = maptr
                preds = maptr_predict(model, dev, raw_by_name, ego_g, calibs, args.maptr_thr)
            all_preds = [p for cls in preds for p in cls]

            tiles: list[Image.Image] = []
            raw_tiles: list[Image.Image] = []
            n_seg = 0
            for name, (cam, k) in cams.items():
                cam_t = cam.get_transform()
                cam_loc, cam_rot = loc(cam_t), rad(cam_t.rotation)
                raw = raw_by_name[name]
                img = overlay_gt(raw.copy(), boxes, cam_loc, cam_rot, k)
                img = draw_traffic_lights(img, tl_frame, cam_loc, cam_rot, k)
                if all_preds:
                    # pose 直接用实挂相机世界位姿(弧度),与 mapviz.cam_pose 同口径
                    n_seg += draw_projected_lines(
                        ImageDraw.Draw(img), all_preds, ego_g, (cam_loc, cam_rot), k, color=PRED_COLOR
                    )
                raw_tiles.append(raw)
                tiles.append(img)

            if maptr:
                tiles = [t.resize((disp_w, disp_h)) for t in tiles]
                raw_tiles = [t.resize((disp_w, disp_h)) for t in raw_tiles]
            frame_img = compose_grid(tiles, list(cams), disp_w, disp_h) if args.view == "grid6" else tiles[0]
            if maptr and args.maptr_bev:
                b = min(260, disp_h)  # 右下角贴 BEV 面板(实时无地图 GT:GT 在 A 阶段矢量库,不在 CARLA)
                frame_img.paste(
                    bev_panel(preds, None, "", (b, b)), (frame_img.width - b, frame_img.height - b)
                )
            if args.dump and not dumped:
                raw_img = (
                    compose_grid(raw_tiles, list(cams), disp_w, disp_h)
                    if args.view == "grid6"
                    else raw_tiles[0]
                )
                dump_pair(args.dump, raw_img, frame_img.copy())
                dumped = True
            fps_now = frames / max(time.time() - t_report, 1e-6)
            draw_hud(
                frame_img,
                f"{world.get_map().name} | {args.view} | actors={len(boxes)} | "
                f"tl={len(tl_frame.lights)} | pred={len(all_preds)}/seg={n_seg} | {fps_now:.1f}fps",
            )
            slots["main"].publish(encode_jpeg(frame_img))
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

"""8 路实时可视化 studio:6 相机 + BEV + 第三方视角,单端口多路 MJPEG(autodrivedata env)。

一路 = 一个 `/stream/<name>`(`CAM_FRONT` / `CAM_FRONT_LEFT` / `CAM_FRONT_RIGHT` /
`CAM_BACK` / `CAM_BACK_LEFT` / `CAM_BACK_RIGHT` / `BEV` / `THIRD_PERSON`),另有 `/`
索引页(8 个 `<img>` 网格)与 `grid` 拼图槽(4×2 拼成一张,只开一个隧道时用)。
多路服务/拼图/rig/GT overlay/键盘全部来自 `bin/live_common.py`(与 `view_stream.py` 共用
同一实现,避免两处漂移)。

与 `view_stream.py` 的分工:那个是单视角 + `--maptr` 实时预测 overlay 的既有验证路径
(§5.11f 有文档化像素验收);本脚本面向"人开着车采数据"的多路监看台。

**键盘**:`--keyboard`(默认在 stdin 是 tty 且未给 `--speed` 时开启)把 WASD 折进本
进程的 tick 循环 —— 控制命令在**下次 tick** 生效,故每 tick 重发 `apply_control`。
与 `--speed`(`set_target_velocity`)互斥:两者都写 ego 控制,同时给会**报错退出**,
不静默取一。

**第三方视角**:非 `attach_to` 相机,每 tick 由 `follow_spectator` 按 ego 位姿 ∘ 局部
偏移显式摆位(**必须在 tick 之前设**,渲染用的是 tick 时刻的位姿);画面上额外描 ego
自身的框,便于确认"车在画面里"。

**在线 SLAM(`--slam`,B 期)**:口径已按实测订正(Plan2.md §P-L.2)。原计划的
"ICP 0.78 s/帧 ⇒ 必须 worker 线程"**两条都不成立**:
- 0.78 s 是 400 帧含转弯/重访的**平均值**;在线逐帧(gap 1)只有 **0.15–0.35 s**
  (CARLA 语义 LiDAR 116k 点,`/tmp/probe_live_sweep.py`);
- worker 线程会被 **GIL** 压到 eff 0.04–0.24 —— studio 主线程每 tick 的 overlay/拼图/HUD
  是纯 Python 字节码,持 GIL 不放。同一对点云:worker 线程 2.38 s vs 主线程同步 0.15–0.35 s。

⇒ 默认**同步执行**(`SlamWorker(sync=True)`,ICP 与渲染在同一个 tick 里排队,实测 ~2.4 fps),
`--slam-async` 才起线程(留作对照/将来把 overlay 挪出主线程时用)。两条路径共用同一套
**滞后止损**:帧号差超 `--slam-max-gap` 时那一帧不做 ICP,直接恒速外推(见 `SlamWorker`)。

- LiDAR(`sensor.lidar.ray_cast_semantic`,口径同 `bin/collect_slam.py`)→ `offer`;
- 跨线程只传**点云拷贝 + t_stamp + ego 世界位姿拷贝**;回来只读 `snapshot()`;
- HUD **显式报滞后** `SLAM 滞后 N 帧 / X s`(N = 已 tick 帧号 − 已处理帧号)。**滞后无界
  增长 = 明确故障**:超 `--slam-lag-warn` 帧时 HUD 转红并在周期报告里打印告警,不装作"在跑";
- 退出纪律:`finally` 里**先停 worker 再销毁 world**(顺序反了线程会读到已销毁的 actor)。

BEV 槽 = SLAM 地图点(浅灰)+ 轨迹(青)+ 可选 MapTR 预测(品红)。在线**无地图 GT**
⇒ `bev_panel(gts=None)` 口径不变。`--slam-report` 落 JSON(滞后序列 + 绘制计数 +
「画出的点数 = 窗内点数」自证),供验收**数值自证**而不靠目检。

用法(CARLA 服务器运行中):
  python bin/live_studio.py                          # 8 路 + 键盘(tty)
  python bin/live_studio.py --speed 8 --npcs         # 定速直行(键盘自动关闭)
  python bin/live_studio.py --scene rain_night       # 天气档
  python bin/live_studio.py --maptr-ckpt outputs/maptr_ep512.pt   # BEV 槽出感知结果
  python bin/live_studio.py --slam --speed 8 --duration 90        # 在线 SLAM + 验收报告
本地:ssh -L 8080:127.0.0.1:8080 <autodl> → 浏览器 http://127.0.0.1:8080

红线:同步模式下 tick 归本脚本,不能与采集脚本同时运行(抢 tick)。
服务只绑 127.0.0.1(经 SSH 隧道访问,不暴露公网)。
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from typing import cast

import carla
import numpy as np
from carla_common import (
    CAM_ATTRS,
    LIDAR_ATTRS,
    SENSOR_OFFSET,
    draw_traffic_lights,
    loc,
    rad,
    spawn_ego,
    spawn_npcs,
    sync_mode,
    traffic_light_frame,
)
from collect_slam import ego_pose_matrix
from live_common import (
    FrameSlot,
    KeyboardState,
    actor_box,
    actor_boxes,
    build_spectator,
    build_surround_rig,
    compose_grid,
    drain,
    draw_hud,
    dump_pair,
    ego_box_deviation,
    ego_box_pixels,
    encode_jpeg,
    follow_spectator,
    image_to_pil,
    load_maptr,
    maptr_predict,
    overlay_gt,
    resolve_rig,
    rig_mount_deviation,
    start_server,
    surround_calibs,
)
from PIL import Image, ImageDraw

from autodrivedata.live_slam import LiveSlam, SlamWorker
from autodrivedata.mapviz import PRED_COLOR, bev_panel, bev_window_mask, draw_projected_lines
from autodrivedata.paths import project_path
from autodrivedata.scenarios import SCENES, merged_weather
from autodrivedata.semantic import semantic_to_velodyne_bin
from autodrivedata.slam import DOWNSAMPLE_VOXEL, ICP_MAX_ITER

BEV_NAME = "BEV"
SPECTATOR_NAME = "THIRD_PERSON"
GRID_NAME = "grid"
EGO_BOX_COLOR = (255, 255, 255)  # 第三方视角里的 ego 自身框(白:与 GT 绿/蓝/黄/橙、灯态红/黄/绿都不撞)


def count_color(img: Image.Image, color: tuple[int, int, int]) -> int:
    """图上该颜色的像素数(数值自证:overlay 画上没有,数像素不靠目检)。"""
    arr = np.asarray(img)
    return int(((arr == np.array(color, dtype=np.uint8)).all(axis=2)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=None, help="加载地图(默认服务器当前图)")
    ap.add_argument("--scene", default=None, choices=sorted(SCENES), help="天气档")
    ap.add_argument("--npcs", action="store_true", help="ego 前方摆 6 个静置 NPC")
    ap.add_argument("--speed", type=float, default=0.0, help="ego 定速直行 m/s(0=静止;与键盘互斥)")
    ap.add_argument(
        "--keyboard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="WASD 操控(默认:stdin 是 tty 且未给 --speed 时开启)",
    )
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--width", type=int, default=640, help="单路显示宽(相机路/拼图单元)")
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--bev-size", type=int, default=420, help="BEV 面板边长 px")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=0.0, help="秒;0 = 常驻(Ctrl-C 退出)")
    ap.add_argument("--dump", default=None, help="落盘首帧各路的 PATH 前缀(overlay)+ 同名 _raw(诊断)")
    ap.add_argument("--maptr-ckpt", default=None, help="MapTR state_dict → BEV/相机路出预测折线")
    ap.add_argument("--maptr-thr", type=float, default=0.2, help="预测实例得分阈值(口径同 eval_maptr)")
    ap.add_argument(
        "--maptr-scale", type=float, default=0.5, help="预测时显示缩放(1242×375 全尺寸拼图带宽过大)"
    )
    ap.add_argument("--maptr-device", default=None, help="推理设备(默认 cuda 若可用;与 CARLA 共享 GPU)")
    ap.add_argument("--slam", action="store_true", help="在线 SLAM:LiDAR → BEV 地图点/轨迹")
    ap.add_argument(
        "--slam-voxel", type=float, default=DOWNSAMPLE_VOXEL, help="SLAM 下采样体素边长(m);调大提速"
    )
    ap.add_argument("--slam-max-iter", type=int, default=ICP_MAX_ITER, help="单帧 ICP 最大迭代数")
    ap.add_argument("--slam-queue", type=int, default=3, help="异步模式:LiDAR 有界队列长度(丢旧;越大越滞后)")
    ap.add_argument(
        "--slam-async",
        action="store_true",
        help="SLAM 跑在 worker 线程(默认**同步**)。实测 worker 线程被 GIL 压到 eff 0.24 ⇒ 默认同步",
    )
    ap.add_argument(
        "--slam-max-gap",
        type=int,
        default=3,
        help="帧号差超此值 → 该帧不做 ICP,直接恒速外推(**止损**,防'丢帧→间隙更大→更慢'正反馈)",
    )
    ap.add_argument(
        "--slam-lag-warn",
        type=int,
        default=5,
        help="滞后帧数超此值 → HUD 转红 + 周期报告打告警(**滞后无界增长 = 故障**)",
    )
    ap.add_argument("--slam-report", default=None, help="落盘 JSON:滞后序列 + 绘制计数(经 project_path)")
    ap.add_argument(
        "--rig",
        choices=("auto", "official", "legacy"),
        default="auto",
        help="环视挂点口径;auto 按权重名选(ep256/ep512=legacy,600/1000=official)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--sim-port", type=int, default=2000)
    args = ap.parse_args()

    if args.keyboard is None:
        args.keyboard = args.speed == 0.0 and sys.stdin.isatty()
    if args.keyboard and args.speed:
        raise SystemExit("--keyboard 与 --speed 互斥:两者都写 ego 控制,同时给会互相覆盖")
    if args.dump:
        args.dump = str(project_path(args.dump))  # 产物锚定项目根(相对路径不随 cwd 漂移)

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
    calibs = surround_calibs(rig) if maptr else {}
    if maptr:
        # 模型输入必须与**该权重**训练数据逐字段一致(1242×375 fov90 + 逐相机挂点)
        # ⇒ rig 按训练口径挂,只在显示侧缩放(与 view_stream 同一条已验证路径)
        print(f"[rig] {rig}(--rig {args.rig} → 权重 {args.maptr_ckpt})")
        cams = build_surround_rig(world, ego, rig=rig)
        disp_w = int(int(CAM_ATTRS["image_size_x"]) * args.maptr_scale)
        disp_h = int(int(CAM_ATTRS["image_size_y"]) * args.maptr_scale)
    else:
        cams = build_surround_rig(world, ego, args.width, args.height, rig=rig)
        disp_w, disp_h = args.width, args.height
    spectator, k_spec = build_spectator(world, None, args.width, args.height)

    queues: dict[str, queue.Queue] = {name: queue.Queue() for name in cams}
    for name, (cam, _) in cams.items():
        cam.listen(queues[name].put)
    q_spec: queue.Queue = queue.Queue()
    spectator.listen(q_spec.put)

    # 传感器 transform 只在 tick 后刷新(C19 同类):tick 前读=全 0 陈旧值,
    # 自检会假报 ~179.8°(=CAM_BACK 规格 180 − ego 固有 yaw 0.159)
    follow_spectator(spectator, ego)  # 先摆位再 tick,否则自检读到的还是出生位姿
    world.tick()
    dev_t, dev_y = rig_mount_deviation(cams, ego, rig)
    print(f"[rig] {rig}:实挂相机 vs 该口径规格 最大偏差:平移 {dev_t:.3f} m / 偏航 {dev_y:.3f}°")
    sp_t = spectator.get_transform()
    off_r, wid_r, depth = ego_box_deviation(actor_box(ego), loc(sp_t), rad(sp_t.rotation), k_spec)
    print(
        f"[third] ego 框在第三方画面:中心偏移 {off_r:.3f} 画幅 / 宽 {wid_r:.3f} 画幅 / 深度 {depth:.2f} m"
        f"(判据:偏移 ≤0.5、宽 ∈[0.05,0.6])"
    )
    if not (off_r <= 0.5 and 0.05 <= wid_r <= 0.6):
        print("[third][warn] ego 框不在画面内或占比异常 —— 第三方机位/投影链需检查")
    print(f"[sensor] {len(cams)} 相机 @ {disp_w}x{disp_h} + 第三方 {args.width}x{args.height}")

    # ---- 在线 SLAM:LiDAR + 滞后止损(默认**同步**执行,见文件头"在线 SLAM") ----
    slam = LiveSlam(voxel=args.slam_voxel, max_iter=args.slam_max_iter) if args.slam else None
    worker: SlamWorker | None = None
    lidar: carla.Sensor | None = None
    lidar_q: queue.Queue = queue.Queue()
    if slam is not None:
        bp = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
        for k, v in LIDAR_ATTRS.items():
            bp.set_attribute(k, v)
        lidar = cast(carla.Sensor, world.spawn_actor(bp, SENSOR_OFFSET, attach_to=ego))
        lidar.listen(lidar_q.put)
        worker = SlamWorker(
            slam, maxsize=args.slam_queue, max_gap=args.slam_max_gap, sync=not args.slam_async
        )
        worker.start()
        mode = "worker 线程" if args.slam_async else "同步(主线程)"
        print(
            f"[slam] 开:{mode} / voxel {args.slam_voxel}m / max_iter {args.slam_max_iter} / "
            f"止损阈值 gap > {args.slam_max_gap} 帧 / 滞后告警 > {args.slam_lag_warn} 帧"
        )
        print("[slam] 逐帧 ICP 实测 0.15-0.35 s ⇒ 同步模式帧率约 2-3 fps;HUD 报'滞后 N 帧'")

    names = [*cams, BEV_NAME, SPECTATOR_NAME, GRID_NAME]
    slots = {name: FrameSlot() for name in names}
    srv = start_server(slots, args.port)
    print(
        f"[stream] {len(names)} 路 http://127.0.0.1:{args.port}  (本地:ssh -L {args.port}:127.0.0.1:{args.port} <autodl>)"
    )
    print(f"[keys] {KeyboardState.KEYS}" if args.keyboard else "[keys] 键盘关闭(--speed 或 --no-keyboard)")

    kb = KeyboardState() if args.keyboard else None
    t_start = time.time()
    t_end = t_start + args.duration if args.duration > 0 else float("inf")
    frames, t_report = 0, time.time()
    tick_idx = -1  # 已 tick 的**绝对**帧号(与 frames 不同:后者每 5s 被 FPS 统计清零)
    dumped = False  # frames 每 5s 被 FPS 统计清零,不能用它判"首帧"
    slam_lag_series: list[dict] = []
    slam_stats: dict[str, int] = {}
    bev_diag: dict[str, int] = {}
    last_ego_T: np.ndarray | None = None  # 末帧 ego 世界位姿(报告里算窗内占比用)
    warn_last = 0.0
    try:
        while time.time() < t_end:
            t0 = time.time()
            if kb is not None:
                kb.poll()
                if kb.quit:
                    break
                kb.apply(ego)  # 每 tick 重发:控制命令在下次 tick 生效
            if args.speed:
                ego.set_target_velocity(vel)  # 定速(同 collect_ab_route;不走 TM)
            follow_spectator(spectator, ego)  # 必须在 tick 前设:渲染用的是 tick 时刻的位姿
            world.tick()
            tick_idx += 1
            boxes = actor_boxes(world)
            ego_t = ego.get_transform()
            ego_T = ego_pose_matrix(ego_t)  # CARLA 4×4(ego 局部 → 世界):SLAM 锚定 + 取地图都用它
            last_ego_T = ego_T
            # 灯态走与采集器同一条实现(carla_common.traffic_light_frame)
            tl_frame = traffic_light_frame(world, f"{frames:06d}", loc(ego_t), float(ego_t.rotation.yaw))

            # SLAM 喂帧:**丢旧队列**,渲染慢时只保最新(防滞后无界增长)。
            # 点云与位姿在 offer 内拷贝 —— 下一 tick 的传感器回调会复用缓冲。
            if worker is not None:
                raw_lidar = drain(lidar_q)
                pts_sem = np.frombuffer(raw_lidar.raw_data, dtype=np.float32).reshape(-1, 6)
                velo = semantic_to_velodyne_bin(pts_sem, seed=tick_idx)
                worker.offer(velo, time.time(), ego_T, tick_idx)

            # 先把本 tick 的图全部取齐(不跨 tick 混帧),再推理 → 同一帧做 overlay
            raw_by_name = {name: image_to_pil(drain(queues[name])) for name in cams}
            raw_spec = image_to_pil(drain(q_spec))
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
                    img = img.resize((disp_w, disp_h))
                    raw = raw.resize((disp_w, disp_h))
                slots[name].publish(encode_jpeg(img))

            # 第三方视角:GT/灯态/预测 + ego 自身框(白),便于确认"车在画面里"
            sp_t = spectator.get_transform()
            sp_loc, sp_rot = loc(sp_t), rad(sp_t.rotation)
            spec_img = overlay_gt(raw_spec.copy(), boxes, sp_loc, sp_rot, k_spec)
            spec_img = draw_traffic_lights(spec_img, tl_frame, sp_loc, sp_rot, k_spec)
            if all_preds:
                n_seg += draw_projected_lines(
                    ImageDraw.Draw(spec_img), all_preds, ego_g, (sp_loc, sp_rot), k_spec, color=PRED_COLOR
                )
            ego_rect = ego_box_pixels(actor_box(ego), sp_loc, sp_rot, k_spec)
            if ego_rect is not None:
                ImageDraw.Draw(spec_img).rectangle(ego_rect, outline=EGO_BOX_COLOR, width=2)
            slots[SPECTATOR_NAME].publish(encode_jpeg(spec_img))

            # BEV 槽:MapTR 预测(品红)+ SLAM 地图点(浅灰)/ 轨迹(青)。
            # 地图/轨迹都要用**当前** ego 位姿换算(SLAM 落后于渲染,用快照那份会"甩尾");
            # 未出首帧地图时 map_in_ego_frame 返回空数组,bev_points 画 0 个点。
            bev_pts = None
            bev_traj = None
            if slam is not None:
                bev_pts = slam.map_in_ego_frame(ego_T)
                bev_traj = slam.traj_in_ego_frame(ego_T)
            bev_diag.clear()
            bev = bev_panel(
                preds,
                None,
                "",
                (args.bev_size, args.bev_size),
                points=bev_pts,
                traj=bev_traj,
                stats=bev_diag,
            )
            slots[BEV_NAME].publish(encode_jpeg(bev))

            grid = compose_grid(
                [*tiles, spec_img.resize((disp_w, disp_h)), bev.resize((disp_w, disp_h))],
                [*cams, SPECTATOR_NAME, BEV_NAME],
                disp_w,
                disp_h,
                cols=4,
            )
            if args.dump and not dumped:
                # 逐路 raw/overlay 成对落盘:差集 = 真实绘制像素(场景自带绿植被/黄标线
                # 与类别色撞色,数绝对颜色会误判 ⇒ 必须做差)
                for name, tile, raw_tile in zip(cams, tiles, raw_tiles, strict=True):
                    dump_pair(f"{args.dump}_{name}.png", raw_tile, tile.copy())
                dump_pair(f"{args.dump}_{SPECTATOR_NAME}.png", raw_spec, spec_img.copy())
                dump_pair(f"{args.dump}_{BEV_NAME}.png", bev_panel([], None, "", bev.size), bev.copy())
                raw_grid = compose_grid(
                    [*raw_tiles, raw_spec.resize((disp_w, disp_h))],
                    [*cams, SPECTATOR_NAME],
                    disp_w,
                    disp_h,
                    cols=4,
                )
                dump_pair(f"{args.dump}_{GRID_NAME}.png", raw_grid, grid.copy())
                dumped = True

            fps_now = frames / max(time.time() - t_report, 1e-6)
            hud = (
                f"{world.get_map().name} | actors={len(boxes)} | tl={len(tl_frame.lights)} | "
                f"pred={len(all_preds)}/seg={n_seg} | {fps_now:.1f}fps"
            )
            if kb is not None:
                hud += f" | {kb.hud(ego)}"
            # SLAM 滞后:**显式报**,不装作"在跑"。滞后 = 已 tick 帧号 − 已处理帧号。
            lag_warn = False
            if worker is not None and slam is not None:
                slam_stats = worker.stats(tick_idx)
                snap = slam.snapshot()
                hud += (
                    f" | SLAM 滞后 {slam_stats['lag_frames']}帧/{slam_stats['lag_s']:.1f}s "
                    f"| 已处理 {snap['n_frames']} 丢 {slam_stats['n_dropped']} 止损 {slam_stats['n_dead']} "
                    f"| 地图 {snap['n_map_points']}点 rmse {snap['mean_rmse']:.3f} "
                    f"| BEV 点 {bev_diag.get('n_points', 0)}/{bev_diag.get('n_points_total', 0)}"
                    f" 轨迹段 {bev_diag.get('n_traj_seg', 0)}"
                )
                lag_warn = slam_stats["lag_frames"] > args.slam_lag_warn
                if lag_warn and time.time() - warn_last >= 5.0:
                    warn_last = time.time()
                    print(
                        f"[slam][warn] 滞后 {slam_stats['lag_frames']} 帧(阈值 {args.slam_lag_warn})"
                        f"| 丢 {slam_stats['n_dropped']} 帧 | 止损 {slam_stats['n_dead']} 帧"
                        f"| 队列 {slam_stats['queue_depth']}"
                        " —— 若持续增长说明止损没生效(--slam-voxel 提档可降本)"
                    )
            draw_hud(grid, hud, warn=lag_warn)
            slots[GRID_NAME].publish(encode_jpeg(grid))
            frames += 1
            if worker is not None:
                slam_lag_series.append({"tick": tick_idx, "t": round(time.time() - t_start, 3), **slam_stats})

            if time.time() - t_report >= 5.0:
                print(f"[stream] {fps_now:.1f} fps | actors={len(boxes)} | {hud}")
                frames, t_report = 0, time.time()
            dt = time.time() - t0
            if dt < 1.0 / args.fps:
                time.sleep(1.0 / args.fps - dt)
    except KeyboardInterrupt:
        print("\n[stop] Ctrl-C")
    finally:
        if kb is not None:
            kb.close()  # 还原终端属性(否则退出后终端不回显)
        srv.shutdown()
        # **先停 worker 再销毁 world**(顺序反了:线程还在跑 ICP 时会读到已销毁的 actor)
        if worker is not None:
            joined = worker.stop()
            print(f"[slam] worker 已停({'干净退出' if joined else '超时未退,可能卡在一次 ICP'})")
            if args.slam_report and slam is not None:
                write_slam_report(
                    args, slam, worker, slam_lag_series, bev_diag, tick_idx, t_start, last_ego_T
                )
        if lidar is not None:
            lidar.stop()
            lidar.destroy()
        spectator.stop()
        spectator.destroy()
        for cam, _ in cams.values():
            cam.stop()
            cam.destroy()
        for a in world.get_actors():
            if a.type_id.startswith(("vehicle", "walker", "controller")):
                a.destroy()
        world.apply_settings(carla.WorldSettings())  # 恢复异步,防服务器冻结
        print("[done] 相机/actor 已清理,服务器恢复异步")


def write_slam_report(
    args: argparse.Namespace,
    slam: LiveSlam,
    worker: SlamWorker,
    lag_series: list[dict],
    bev_diag: dict[str, int],
    tick_idx: int,
    t_start: float,
    last_ego_T: np.ndarray | None,
) -> None:
    """落盘验收报告(**数值自证**,不靠目检):滞后序列 + 绘制计数 + 窗内点自证。

    B4 四项判据里三项在这里给出:`lag_max` 稳态有界(非单调增长)、BEV 点/轨迹段像素数 >0、
    画出的地图点**全部**落在 ego 系窗口内(换算没错位)。

    **两个比例别混**(B4-④ 判据订正):累积地图覆盖整段行程(75 s × 8 m/s ≈ 300 m),
    而 BEV 窗口只有 30 m × 60 m,故 `bev_map_in_window_ratio`(窗内 / **全部**地图点)
    远小于 1 是**正常的** —— 早期把它当"应该 ≈100%"是误读。真正的判据是
    `bev_drawn_equals_in_window`:面板上**画出来**的点数必须恰好等于窗内点数
    (窗外的点一个都不该画上)。两个数独立算(`bev_points` 内部过滤 vs 这里 `bev_window_mask`
    重算),相等才说明窗口判据两处一致。

    窗内统计用**末帧** ego 位姿换算:BEV 面板显示的正是那一刻的窗口,拿它统计才与
    "面板上看到的点"一致(用帧 0 的位姿会因车辆已驶远而大量出窗)。
    """
    snap = slam.snapshot()
    lags = [s["lag_frames"] for s in lag_series]
    dropped = [s["n_dropped"] for s in lag_series]
    # "有界"判据:后半段滞后峰值不得显著高于前半段(单调增长会在这里暴露)
    half = max(len(lags) // 2, 1)
    first_half_max = max(lags[:half], default=0)
    second_half_max = max(lags[half:], default=0)
    anchor = last_ego_T if last_ego_T is not None else slam._ego0
    pts = slam.map_in_ego_frame(anchor) if anchor is not None else np.zeros((0, 3))
    in_win = int(bev_window_mask(pts).sum()) if len(pts) else 0
    # 面板**实际画出**的点数(bev_points 内部独立过滤一遍)⇒ 与 in_win 相等才是自证
    n_drawn = int(bev_diag.get("n_points", 0))
    traj_in_win = 0
    if anchor is not None:
        traj = slam.traj_in_ego_frame(anchor)
        if len(traj):
            traj_in_win = int(bev_window_mask(traj).sum())
    report = {
        "slam_voxel": args.slam_voxel,
        "slam_max_iter": args.slam_max_iter,
        "slam_queue": args.slam_queue,
        "slam_async": bool(args.slam_async),
        "slam_max_gap": args.slam_max_gap,
        "lag_warn_threshold": args.slam_lag_warn,
        "wall_s": round(time.time() - t_start, 2),
        "n_ticks": tick_idx + 1,
        "n_processed": snap["n_frames"],
        "n_offered": worker.n_offered,
        "n_dropped": worker.n_dropped,
        "n_dead": worker.n_dead,
        "n_failed": snap["n_failed"],
        "n_nan": snap["n_nan"],
        "lag_max": max(lags, default=0),
        "lag_mean": round(float(np.mean(lags)), 3) if lags else 0.0,
        "lag_first_half_max": first_half_max,
        "lag_second_half_max": second_half_max,
        "lag_bounded": bool(second_half_max <= max(first_half_max * 1.5, args.slam_lag_warn)),
        "dropped_final": dropped[-1] if dropped else 0,
        "n_map_points": snap["n_map_points"],
        "mean_rmse": round(snap["mean_rmse"], 5),
        "mean_overlap": round(snap["mean_overlap"], 5),
        "icp_s": round(snap["icp_s"], 2),
        "bev": dict(bev_diag),
        "bev_in_window_points": in_win,
        "bev_map_in_window_ratio": round(in_win / max(len(pts), 1), 4),
        "bev_drawn_points": n_drawn,
        "bev_drawn_equals_in_window": bool(n_drawn == in_win),
        "bev_traj_in_window": traj_in_win,
        "bev_traj_in_window_ratio": round(traj_in_win / max(len(snap["poses"]), 1), 4),
        "lag_series": lag_series,
    }
    out = project_path(args.slam_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[slam] 报告 → {out}\n"
        f"  滞后 max {report['lag_max']} 帧(前半峰 {first_half_max} / 后半峰 {second_half_max})"
        f" | 有界 {report['lag_bounded']}\n"
        f"  丢 {worker.n_dropped} / 供 {worker.n_offered} 帧 | 处理 {snap['n_frames']} 帧"
        f" | 止损 {worker.n_dead} 帧 | 地图 {snap['n_map_points']} 点 | rmse {report['mean_rmse']}\n"
        f"  BEV 画出 {n_drawn} 点(窗内 {in_win} | 全图 {snap['n_map_points']} ⇒ 占 {report['bev_map_in_window_ratio']})"
        f" | 自证 {report['bev_drawn_equals_in_window']}\n"
        f"  轨迹 {bev_diag.get('n_traj_seg', 0)} 段(窗内 {traj_in_win}/{len(snap['poses'])} 帧"
        f" ⇒ {report['bev_traj_in_window_ratio']})"
    )
    if not report["lag_bounded"]:
        print(f"[slam][warn] 滞后后半段峰值 {second_half_max} > 前半段 {first_half_max} —— 疑似无界增长")


if __name__ == "__main__":
    main()

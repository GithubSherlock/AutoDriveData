"""8 路实时可视化 studio:6 相机 + BEV + 第三方视角,单端口多路 MJPEG(autodrivedata env)。

一路 = 一个 `/stream/<name>`(`CAM_FRONT` / `CAM_FRONT_LEFT` / `CAM_FRONT_RIGHT` /
`CAM_BACK` / `CAM_BACK_LEFT` / `CAM_BACK_RIGHT` / `BEV` / `THIRD_PERSON`),另有 `/`
索引页(8 个 `<img>` 网格)与 `grid` 拼图槽(**三层**拼成一张,只开一个隧道时用)。
多路服务/拼图/rig/GT overlay/键盘全部来自 `bin/live_common.py`(与 `view_stream.py` 共用
同一实现,避免两处漂移)。

**拼图三层(用户口径 2026-09-20)**:① 左前/前/右前 ② 右后/后/左后 ③ 第三方 + BEV。
**每格保持原生像素**(相机 1242×375、第三方 640×360、BEV `--bev-size`),不为了对齐网格而
缩放 —— 走 `live_common.compose_rows`(按行拼、各格原样)。旧实现用等尺寸 `compose_grid`
把 6 路 1242×375 塞进 621×187 的格子,`Image.paste` 在源图大于目标框时**不报错、只贴左上角**
⇒ 右半 + 下半被静默裁掉,而**下半正是地面**(用户报告"6 视角 FoV 缩得看不到地面";
数值判据:拼图格与「源图左上角裁剪」平均绝对差 0.13,与「整幅缩放」差 66.2)。

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

**标定监看槽(`--calib`)**:另挂一套 `sensor.camera.depth`(与 RGB 槽**同挂点/同分辨率**
—— `build_surround_rig(kind="depth")` 只换蓝图,内参/挂点/分辨率三者任一不同着色点就错位),
每 tick 把 LiDAR 点按**当前标定**投影到各相机、按与渲染深度的残差着色画上去
(绿 <0.05 m / 黄 <0.15 m / 红 其余),HUD 第二行报实时中位误差。数值核心是
[autodrivedata/calib_live.py](../autodrivedata/calib_live.py)(纯值、有单测),
与离线自证探针 [bin/probe_calib.py](probe_calib.py) 共用 `autodrivedata/calib_probe.py`。

- **平面隔 tick 重拟合**(`--calib-refit`,默认 2):拟合出的是**世界系**平面(描述场景
  表面,不是"这一帧的点云"),ego 移动几米后同一块路面仍是同一平面,被挡住的点由
  `collect_samples` 的**单侧**可见性判据剔掉。实测拟合 100–150 ms/tick 而六相机采样
  合计仅 ~9 ms ⇒ **瓶颈全在拟合**,故不每帧做。
- **CAM_BACK 是平台边界,不是标定坏了**:官方 nuScenes 挂点 z=1.579 只比 CARLA ego
  **自身车顶**(z≈1.556)高 **0.023 m** ⇒ 实测近场(深度 <0.5 m)像素占比在 1242×375 下
  **0.367**、640×360 下 **0.195**(其余五路 0.000)、可用样本常年 0–30(其余 50–200)。
  它的残差中位数**并不因此变差** ⇒ 判据必须是"**样本 < `MIN_CAM_SAMPLES` 时报 `–`,
  不许报 0.000**",而不是"median 大 = 坏标定"。判据做成**数据驱动**
  (`calib_live.self_occluded_cameras`:相对同批其余相机的近场占比定阈),不按相机名硬编码。
- `--calib-report` 落 JSON:逐相机样本数/中位残差/近场占比 + 逐 tick 中位序列。
- `--dump` 与 `--calib` 同开时,落盘的 raw/overlay 差集 = **真实画上的标定点像素**
  (场景自带绿/黄与着色带撞色,数绝对颜色会误判)。

用法(CARLA 服务器运行中):
  python bin/live_studio.py                          # 8 路 + 键盘(tty)
  python bin/live_studio.py --speed 8 --npcs         # 定速直行(键盘自动关闭)
  python bin/live_studio.py --scene rain_night       # 天气档
  python bin/live_studio.py --calib --speed 8 --duration 30 --calib-report outputs/calib_check/live.json
  python bin/live_studio.py --maptr-ckpt outputs/maptr_ep512.pt   # BEV 槽出感知结果
  python bin/live_studio.py --slam --speed 8 --duration 90        # 在线 SLAM + 验收报告
  # 落一段八视角视频(拼图槽逐帧写 mp4;--video-fps 调到接近实际采集 fps 才是实时播放)
  python bin/live_studio.py --slam --maptr-ckpt outputs/maptr_ep512.pt --npcs \
    --speed 8 --duration 40 --video outputs/videos/studio_8view.mp4 --video-fps 3
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
from pathlib import Path
from typing import Any, cast

import carla
import numpy as np
from carla_common import (
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
    compose_rows,
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
    rig_frame,
    rig_mount_deviation,
    start_server,
    surround_calibs,
)
from PIL import Image, ImageDraw

from autodrivedata import calib_live as cl
from autodrivedata import calib_probe
from autodrivedata.depth_codec import decode_depth
from autodrivedata.live_slam import LiveSlam, SlamWorker
from autodrivedata.mapviz import PRED_COLOR, bev_panel, bev_window_mask, draw_projected_lines
from autodrivedata.paths import project_path
from autodrivedata.scenarios import SCENES, merged_weather
from autodrivedata.semantic import semantic_to_velodyne_bin
from autodrivedata.slam import DOWNSAMPLE_VOXEL, ICP_MAX_ITER

try:  # opencv 只在 `--video` 时需要(与 stereo.py 同一处口径:可选依赖不挡主流程)
    import cv2

    _CV2_OK = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_OK = False


def _fourcc(tag: str) -> int:
    """`cv2.VideoWriter_fourcc(*tag)` 的取用口。

    走 `getattr` 而不是直接点属性:opencv 的 `.pyi` 里只有 `VideoWriter.fourcc`(类方法),
    没有模块级 `VideoWriter_fourcc`(C 绑定实际存在)—— 直接点会让 pyright 报未定义属性。
    """
    assert cv2 is not None
    fn = getattr(cv2, "VideoWriter_fourcc")  # noqa: B009 — 见上:绕 pyi 缺项
    return int(fn(*tag))


def _open_video_writer(path: str, fps: float, size: tuple[int, int]) -> Any:
    """按真实帧尺寸开 mp4 编码器(**首帧才开**,尺寸随 `--video-tile` 变)。"""
    assert cv2 is not None
    w = cv2.VideoWriter(path, _fourcc("mp4v"), fps, size)
    if not w.isOpened():
        raise SystemExit(f"VideoWriter 打不开 {path}(检查扩展名/编码器:mp4v 需要 opencv 带 FFMPEG)")
    return w


BEV_NAME = "BEV"
SPECTATOR_NAME = "THIRD_PERSON"
GRID_NAME = "grid"
EGO_BOX_COLOR = (255, 255, 255)  # 第三方视角里的 ego 自身框(白:与 GT 绿/蓝/黄/橙、灯态红/黄/绿都不撞)

# 拼图布局(用户口径 2026-09-20):三层,行内按**车头朝前**的环视顺序排。
#   ① 左前 / 前 / 右前   ② 右后 / 后 / 左后   ③ 第三方 + BEV
# 不沿用 `SURROUND_CAMS` 的字典序(FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT)
# —— 那样第二行会变成"左后/右后"与地理直觉相反。
GRID_ROWS: tuple[tuple[str, ...], ...] = (
    ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"),
    ("CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"),
    (SPECTATOR_NAME, BEV_NAME),
)


def grid_rows(by_name: dict[str, Image.Image]) -> list[list[tuple[str, Image.Image]]]:
    """布局名 → 按 `GRID_ROWS` 组行;**整行都缺**时跳过该行(`--dump` 的 raw 拼图没有 BEV,
    第三行只剩第三方 —— 不会留一条空行让 `compose_rows` 报错)。"""
    rows = [[(n, by_name[n]) for n in row if n in by_name] for row in GRID_ROWS]
    return [r for r in rows if r]


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
    ap.add_argument(
        "--video",
        default=None,
        help="落盘一段**拼图视频**(mp4,含 8 路 + HUD);需 --duration 或 Ctrl-C 结束(经 project_path)",
    )
    ap.add_argument(
        "--video-fps",
        type=float,
        default=10.0,
        help="视频标称帧率。循环跑不到这个速度时视频会被**加速播放**(结束时打印实际采集 fps)",
    )
    ap.add_argument(
        "--video-tile",
        type=int,
        default=1,
        help="视频里每格放大倍数(1=拼图原始分辨率 3×1242=3726 宽;2 只是插值放大,文件翻倍)",
    )
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
        "--calib",
        action="store_true",
        help="标定监看槽:每 tick 把 LiDAR 点按当前标定投影,按深度残差着色画到 6 相机图上",
    )
    ap.add_argument(
        "--calib-refit",
        type=int,
        default=2,
        help="每 N tick 重拟合一次世界系平面(拟合是瓶颈 ~100-150ms;平面是世界系的,可复用)",
    )
    ap.add_argument(
        "--calib-voxel", type=float, default=cl.LIVE_VOXEL_M, help="标定槽点云体素边长(m);调大提速"
    )
    ap.add_argument(
        "--calib-plane-radius",
        type=float,
        default=cl.LIVE_PLANE_RADIUS_M,
        help="逐点邻域平面拟合半径(m);调小提速(实测 1.5 以下合格点会塌成 0)",
    )
    ap.add_argument(
        "--calib-max-dist", type=float, default=cl.LIVE_MAX_DIST_M, help="平面拟合用的 LiDAR 点距离上限(m)"
    )
    ap.add_argument(
        "--calib-max-samples", type=int, default=cl.LIVE_MAX_SAMPLES, help="平面拟合点数上限(O(N²) 的闸)"
    )
    ap.add_argument("--calib-report", default=None, help="落盘 JSON:逐相机样本数/中位残差(经 project_path)")
    ap.add_argument(
        "--rig",
        choices=("auto", "nuscenes", "legacy"),
        default="auto",
        help="环视挂点口径;auto 按权重名选(ep256/ep512=legacy,600/1000=nuscenes)",
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

    # 视频落盘:拼图槽逐帧写 mp4。为什么是**拼图**而不是 8 个文件:8 路各写一个 mp4
    # 后还得再拼一次,而拼图槽本来就是"一张图看全 8 路"的口径;要单路素材用 `--dump` 落帧。
    # 编码器**首帧才打开**:尺寸取自真实帧(带 --video-tile 缩放),不靠预先推算。
    # `Any` 而不是 `cv2.VideoWriter`:cv2 是可选导入(未装时名字绑到 None),写进类型注解
    # 会连带一串 "None 没有该属性" 的假报。
    vw: Any = None
    vpath: str | None = None
    n_video = 0
    if args.video:
        if not _CV2_OK:
            raise SystemExit("--video 需要 opencv(cv2):本环境未装。可改用 --dump 落帧")
        vpath = str(project_path(args.video))
        Path(vpath).parent.mkdir(parents=True, exist_ok=True)
        print(f"[video] → {vpath} @ {args.video_fps:g} fps(首帧到齐后按实际尺寸打开)")

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
    if maptr:
        # 模型输入必须与**该权重**训练数据逐字段一致(画幅 + 逐通道 fov + 逐相机挂点)
        # ⇒ rig 按训练口径挂,只在显示侧缩放(与 view_stream 同一条已验证路径)。
        # 画幅取自 `rig_frame` 而不是 CAM_ATTRS —— 后者是本仓十余个 KITTI/P1 脚本的共用常量,
        # 与 nuscenes rig 的 1600×900 无关(拿它推显示尺寸会静默按错画幅缩放)。
        rig_w, rig_h, _ = rig_frame(rig)
        cams = build_surround_rig(world, ego, rig=rig)
        disp_w = int(rig_w * args.maptr_scale)
        disp_h = int(rig_h * args.maptr_scale)
    else:
        rig_w, rig_h = args.width, args.height
        cams = build_surround_rig(world, ego, args.width, args.height, rig=rig)
        disp_w, disp_h = args.width, args.height
    # 深度监看槽必须与**上面实际挂的那套 RGB** 逐像素对齐(画幅/挂点/内参三者任一不同即错位)
    # ⇒ 传 rig_w/rig_h 而非 args.width/height(`--maptr` 时两者不同)。
    calibs = surround_calibs(rig, rig_w, rig_h) if maptr else {}
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

    # ---- 标定监看槽:深度相机 rig + 点云平面(与 RGB 槽**同分辨率/同挂点**) ----
    # 深度相机必须与 RGB 逐像素对齐:内参、挂点、分辨率三者任一不同,着色点就整体错位
    # (`build_surround_rig` 的 `kind` 只换蓝图,画幅/挂点/内参全走同一份 `rig_spec`/`rig_frame`)。
    # LiDAR 复用 `--slam` 那台(语义蓝图带 intensity,这里只取前 3 列)⇒ 同开时不重复挂。
    # 点云用 `sensor.lidar.ray_cast`(非语义):语义帧每点 6 个 float32,本槽只关心几何。
    depth_cams: dict[str, tuple[carla.Sensor, Any]] = {}
    depth_qs: dict[str, queue.Queue] = {}
    calib_lidar: carla.Sensor | None = None
    calib_lidar_q: queue.Queue = queue.Queue()
    calib_depth: dict[str, np.ndarray] = {}
    planes: tuple[np.ndarray, np.ndarray, np.ndarray] = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    samples: dict[str, calib_probe.DepthSamples] = {}
    calib_stats: dict[str, cl.CameraResidual] = {}
    calib_med: float | None = None
    calib_refits = 0
    calib_series: list[dict] = []
    calib_rng = np.random.default_rng(0)
    if args.calib:
        depth_cams = build_surround_rig(world, ego, rig_w, rig_h, rig=rig, kind="depth")
        for name, (cam, _) in depth_cams.items():
            q: queue.Queue = queue.Queue()
            cam.listen(q.put)
            depth_qs[name] = q
        if lidar is None:
            bp_l = world.get_blueprint_library().find("sensor.lidar.ray_cast")
            for k, v in LIDAR_ATTRS.items():
                bp_l.set_attribute(k, v)
            calib_lidar = cast(carla.Sensor, world.spawn_actor(bp_l, SENSOR_OFFSET, attach_to=ego))
            calib_lidar.listen(calib_lidar_q.put)
        print(
            f"[calib] 开:6 路深度 @ {args.width}x{args.height}(与 RGB 槽同挂点) / "
            f"体素 {args.calib_voxel}m / 平面半径 {args.calib_plane_radius}m / "
            f"距离<{args.calib_max_dist}m / 上限 {args.calib_max_samples} 点 / "
            f"每 {args.calib_refit} tick 重拟合 / LiDAR {'复用 --slam' if lidar else '新挂 ray_cast'}"
        )
        print(
            "[calib] 判据:|e| 中位数(有数据相机)**等权**取中位;< 0.05 m 绿 / < 0.15 m 黄 / 其余红;"
            f"样本 < {cl.MIN_CAM_SAMPLES} 的相机报 '–' 而不是 0.000"
        )

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
    calib_warn_last = 0.0
    try:
        while time.time() < t_end:
            t0 = time.time()
            t_calib0 = time.time()
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
            raw_lidar: Any = None
            lidar_stride = 0
            if worker is not None:
                raw_lidar = drain(lidar_q)
                lidar_stride = 6  # 语义蓝图:每点 6 个 float32(x,y,z,cos,idx,标签)
                pts_sem = np.frombuffer(raw_lidar.raw_data, dtype=np.float32).reshape(-1, lidar_stride)
                velo = semantic_to_velodyne_bin(pts_sem, seed=tick_idx)
                worker.offer(velo, time.time(), ego_T, tick_idx)
            if args.calib and calib_lidar is not None:
                # `--slam` 未开:本槽自己那台 ray_cast(**每 tick 必取**,不能隔 tick 才取 ——
                # 攒着不取会让队列一直涨;`drain` 只保最新,故取多取少都不影响结果)
                raw_lidar = drain(calib_lidar_q)
                lidar_stride = 4  # ray_cast:每点 4 个 float32(x,y,z,intensity)

            # 先把本 tick 的图全部取齐(不跨 tick 混帧),再推理 → 同一帧做 overlay
            raw_by_name = {name: image_to_pil(drain(queues[name])) for name in cams}
            raw_spec = image_to_pil(drain(q_spec))

            # 标定槽:深度帧 + 点云平面 → 逐相机采样。
            # **平面隔 tick 重拟合**:它是**世界系**的(描述场景表面,不是"这一帧的点云"),
            # ego 移动几米后同一块路面仍是同一平面,被挡住的点由 collect_samples 的**单侧**
            # 可见性判据剔掉,不需要重拟合。实测拟合 ~100-150 ms/tick 而六相机采样仅 ~9 ms
            # ⇒ 瓶颈全在拟合,故默认每 2 tick 一次。
            if args.calib:
                calib_depth = {
                    name: decode_depth(drain(depth_qs[name]).raw_data, args.height, args.width)
                    for name in depth_cams
                }
                if tick_idx % max(int(args.calib_refit), 1) == 0:
                    # 源 LiDAR:`--slam` 同开时复用**本 tick 已取的那一帧**(不重复 drain ——
                    # 队列此刻已空,再 drain 会阻塞到超时)
                    src = calib_lidar if calib_lidar is not None else lidar
                    assert src is not None and raw_lidar is not None, (
                        "calib 槽必须有 LiDAR 帧:新挂 ray_cast 或复用 --slam 的语义帧"
                    )
                    ltf = src.get_transform()
                    pw = cl.world_points_from_lidar(
                        np.frombuffer(raw_lidar.raw_data, dtype=np.float32).reshape(-1, lidar_stride),
                        loc(ltf),
                        rad(ltf.rotation),
                    )
                    planes = cl.live_planes(
                        pw,
                        np.asarray(loc(ltf), dtype=np.float64),
                        calib_rng,
                        voxel=args.calib_voxel,
                        radius=args.calib_plane_radius,
                        max_dist=args.calib_max_dist,
                        max_samples=args.calib_max_samples,
                    )
                    calib_refits += 1
                samples = {}
                for name, (cam, k) in depth_cams.items():
                    cam_t = cam.get_transform()
                    samples[name] = cl.sample_camera(
                        planes, (loc(cam_t), rad(cam_t.rotation)), k, calib_depth[name]
                    )
                calib_stats = cl.summarize(samples, {n: cl.near_fraction(d) for n, d in calib_depth.items()})
                calib_med = cl.pooled_median(calib_stats)
                calib_series.append(
                    {
                        "tick": tick_idx,
                        "pooled_median": calib_med,
                        "plane_points": int(planes[0].shape[0]),
                        "cams": {
                            n: {"n": s.n, "median": s.median_abs, "near": round(s.near_fraction, 4)}
                            for n, s in calib_stats.items()
                        },
                    }
                )

            ego_g = [*loc(ego_t), ego_t.rotation.yaw, ego_t.rotation.pitch, ego_t.rotation.roll]
            preds: list[list[np.ndarray]] = []
            if maptr:
                model, dev = maptr
                preds = maptr_predict(model, dev, raw_by_name, ego_g, calibs, args.maptr_thr)
            all_preds = [p for cls in preds for p in cls]

            tiles: list[Image.Image] = []
            raw_tiles: list[Image.Image] = []
            n_seg = 0
            n_calib_pts = 0
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
                if args.calib:
                    # 深度相机与 RGB 槽**同挂点/同分辨率** ⇒ 采样点的图像坐标可直接当索引
                    # 画到 RGB 图上(绿 <0.05m / 黄 <0.15m / 红 其余)。走 numpy 就地写,
                    # 不经 ImageDraw:每帧上千点,逐点 draw 是纯开销。
                    s = samples.get(name)
                    if s is not None and len(s):
                        arr = np.asarray(img).copy()
                        n_calib_pts += cl.paint_residuals(arr, s.uv, s.residual)
                        img.paste(Image.fromarray(arr))
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

            # 拼图:**三层**,每格保持原生分辨率(不缩放、不裁剪)。
            # 行序按用户口径:左前/前/右前 → 右后/后/左后 → 第三方 + BEV。
            # 第三方与 BEV 不缩到相机格尺寸 —— `compose_rows` 按各格自身像素摆。
            by_name = {**dict(zip(cams, tiles, strict=True)), SPECTATOR_NAME: spec_img, BEV_NAME: bev}
            grid = compose_rows(grid_rows(by_name))
            if args.dump and not dumped:
                # 逐路 raw/overlay 成对落盘:差集 = 真实绘制像素(场景自带绿植被/黄标线
                # 与类别色撞色,数绝对颜色会误判 ⇒ 必须做差)
                for name, tile, raw_tile in zip(cams, tiles, raw_tiles, strict=True):
                    dump_pair(f"{args.dump}_{name}.png", raw_tile, tile.copy())
                dump_pair(f"{args.dump}_{SPECTATOR_NAME}.png", raw_spec, spec_img.copy())
                dump_pair(f"{args.dump}_{BEV_NAME}.png", bev_panel([], None, "", bev.size), bev.copy())
                raw_by = {**dict(zip(cams, raw_tiles, strict=True)), SPECTATOR_NAME: raw_spec}
                raw_grid = compose_rows(grid_rows(raw_by))
                dump_pair(f"{args.dump}_{GRID_NAME}.png", raw_grid, grid.copy())
                dumped = True

            fps_now = frames / max(time.time() - t_report, 1e-6)
            hud = (
                f"{world.get_map().name} | actors={len(boxes)} | tl={len(tl_frame.lights)} | "
                f"pred={len(all_preds)}/seg={n_seg} | {fps_now:.1f}fps"
            )
            if args.calib:
                hud += f" | 标定点 {n_calib_pts}"
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
            if args.calib:
                calib_dt = time.time() - t_calib0
                draw_hud(
                    grid,
                    cl.hud_line(calib_stats, calib_med, int(planes[0].shape[0]), calib_refits)
                    + f" | 本 tick {calib_dt * 1000:.0f} ms",
                    y=16,
                )
                # 全相机样本不足 ⇒ 这一帧**没有标定结论**,必须看得见(否则 0.000 会被读成"完美")
                if calib_med is None and time.time() - calib_warn_last >= 5.0:
                    calib_warn_last = time.time()
                    print(
                        "[calib][warn] 本帧六路样本全部不足 —— 无标定结论"
                        "(点云/深度不同步或平面合格点为 0;--calib-plane-radius 调大可增加合格点)"
                    )
            slots[GRID_NAME].publish(encode_jpeg(grid))

            # 视频:同一帧的拼图放大后写 mp4(**放大而不是缩小** —— 每格 640×360 直接压成
            # 视频会糊到看不清 GT 框)。编码器首帧才按真实尺寸打开(尺寸随 --video-tile 变)。
            if vpath is not None:
                t = max(int(args.video_tile), 1)
                vframe = grid if t == 1 else grid.resize((grid.width * t, grid.height * t))
                if vw is None:
                    vw = _open_video_writer(vpath, args.video_fps, vframe.size)
                    print(f"[video] 编码器已开:{vframe.width}×{vframe.height} @ {args.video_fps:g} fps")
                vw.write(np.asarray(vframe)[:, :, ::-1])  # PIL RGB → cv2 BGR
                n_video += 1

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
        # 视频收尾:**必须在 `--slam-report` 之前**(报告里要带实际落盘帧数/时长)
        if vw is not None:
            vw.release()
            real_fps = n_video / max(time.time() - t_start, 1e-6)
            print(
                f"[video] {n_video} 帧 → {vpath}\n"
                f"  标称 {args.video_fps:g} fps,实际采集 {real_fps:.2f} fps"
                f" ⇒ 播放速度是实时的 {args.video_fps / max(real_fps, 1e-6):.1f}×"
                "(--video-fps 调到接近实际采集 fps 即为实时)"
            )
        # **先停 worker 再销毁 world**(顺序反了:线程还在跑 ICP 时会读到已销毁的 actor)
        if worker is not None:
            joined = worker.stop()
            print(f"[slam] worker 已停({'干净退出' if joined else '超时未退,可能卡在一次 ICP'})")
            if args.slam_report and slam is not None:
                write_slam_report(
                    args,
                    slam,
                    worker,
                    slam_lag_series,
                    bev_diag,
                    tick_idx,
                    t_start,
                    last_ego_T,
                    vpath,
                    n_video,
                )
        if lidar is not None:
            lidar.stop()
            lidar.destroy()
        if calib_lidar is not None:
            calib_lidar.stop()
            calib_lidar.destroy()
        spectator.stop()
        spectator.destroy()
        for cam, _ in cams.values():
            cam.stop()
            cam.destroy()
        for cam, _ in depth_cams.values():
            cam.stop()
            cam.destroy()
        if args.calib and args.calib_report:
            write_calib_report(args, calib_series, calib_stats, calib_refits, tick_idx, t_start)
        for a in world.get_actors():
            if a.type_id.startswith(("vehicle", "walker", "controller")):
                a.destroy()
        world.apply_settings(carla.WorldSettings())  # 恢复异步,防服务器冻结
        print("[done] 相机/actor 已清理,服务器恢复异步")


def write_calib_report(
    args: argparse.Namespace,
    series: list[dict],
    stats: dict[str, cl.CameraResidual],
    n_refit: int,
    tick_idx: int,
    t_start: float,
) -> None:
    """落盘实时标定读数(**数值自证**):逐相机样本数/中位残差/近场占比 + 逐 tick 序列。

    **判据是"样本数够才报 median"**:CAM_BACK 的官方 nuScenes 挂点只比 ego 车顶高 0.023 m
    ⇒ 大面积画面是**自己的车顶**,样本数常年 0–30(其余 50–200)。它的残差中位数并**不**
    因此变差(0.0003 m 量级,与其它相机同级),故 `n < MIN_CAM_SAMPLES` 时这里写 `null`
    而不是写一个从 3 个样本算出来的"漂亮数字"。

    `pooled_median_series` 是逐 tick 的**有数据相机等权中位数**,用来判断"开着车时标定读数
    是否稳定"(相机外参是常量,读数不应随行驶显著漂移)。
    """
    usable = {n: s for n, s in stats.items() if s.usable}
    meds = [s.median_abs for s in usable.values() if s.median_abs is not None]
    series_meds = [r["pooled_median"] for r in series if r["pooled_median"] is not None]
    occl = cl.self_occluded_cameras(stats)
    report = {
        "rig": args.rig,
        "width": args.width,
        "height": args.height,
        "convention": "corner",
        "min_cam_samples": cl.MIN_CAM_SAMPLES,
        "budget": {
            "voxel_m": args.calib_voxel,
            "plane_radius_m": args.calib_plane_radius,
            "max_dist_m": args.calib_max_dist,
            "max_samples": args.calib_max_samples,
            "refit_every_n_ticks": args.calib_refit,
        },
        "wall_s": round(time.time() - t_start, 2),
        "n_ticks": tick_idx + 1,
        "n_refits": n_refit,
        "n_series": len(series),
        "cams": {
            n: {
                "n": s.n,
                "median_abs_m": None if s.median_abs is None else round(s.median_abs, 5),
                "p90_abs_m": None if s.p90_abs is None else round(s.p90_abs, 5),
                "near_fraction": round(s.near_fraction, 4),
                "usable": s.usable,
                "self_occluded": n in occl,
            }
            for n, s in stats.items()
        },
        "self_occluded": sorted(occl),
        "pooled_median_m": round(float(np.median(meds)), 5) if meds else None,
        "n_cams_usable": len(usable),
        "pooled_median_series_m": [None if v is None else round(v, 5) for v in series_meds],
        "pooled_median_series_median_m": round(float(np.median(series_meds)), 5) if series_meds else None,
        "pooled_median_series_max_m": round(max(series_meds), 5) if series_meds else None,
        "series": series,
    }
    out = project_path(args.calib_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pooled = report["pooled_median_m"]
    pooled_txt = "无数据(样本不足)" if pooled is None else f"{pooled:.5f} m"
    print(
        f"[calib] 报告 → {out}\n"
        f"  有数据相机 {len(usable)}/{len(stats)} | 中位 |e| {pooled_txt}\n"
        f"  逐 tick 中位序列 {len(series_meds)} 帧:中位 {report['pooled_median_series_median_m']}"
        f" / 最大 {report['pooled_median_series_max_m']} m"
    )
    if occl:
        print(
            f"[calib] 自遮挡相机 {sorted(occl)}(近场占比 "
            + " / ".join(f"{n} {stats[n].near_fraction * 100:.0f}%" for n in sorted(occl))
            + ")—— 样本不足属**平台边界**,不是标定误差"
        )


def write_slam_report(
    args: argparse.Namespace,
    slam: LiveSlam,
    worker: SlamWorker,
    lag_series: list[dict],
    bev_diag: dict[str, int],
    tick_idx: int,
    t_start: float,
    last_ego_T: np.ndarray | None,
    video_path: str | None = None,
    n_video: int = 0,
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
        "video_path": video_path,
        "video_frames": int(n_video),
        "video_fps_nominal": args.video_fps,
        "video_fps_real": round(n_video / max(time.time() - t_start, 1e-6), 3),
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

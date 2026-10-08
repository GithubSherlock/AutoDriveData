"""B1:环视 6 相机采集(nuScenes 布局)→ 图像 + 内外参 + 逐帧 ego 位姿。

MapTR 端到端训练的输入侧:6 视角图像 + sensor2ego 外参 + 相机内参 + ego2global
位姿(与 MapTRv2 `nuscenes_converter` 的 cams/ego_pose 字段同构)。B2 组装器
消费本输出拼 MapTR 训练格式,地图 GT 来自 A 阶段矢量库。

采集纪律与 collect_drive 相同(同步模式/预热/清场/场景天气档);**不改动
A/B 采集器**(P1 复现性红线)。NPC 布置复用 collect_drive 的既有函数。

nuScenes 相机布局:**直接取官方 calibrated_sensor**(6DoF 四元数 + 平移),
由 [autodrivedata/camera_rig.py](../calib/camera_rig.py) 转成 CARLA 口径
(平移 y 翻号、姿态走 `nus_camera_rotation_to_carla`)。采集器不再自己维护角度表。

⚠️ **历史 bug(2026-09-22 修)**:此前 `SURROUND_CAMS` 把官方**方位角**原样抄成正数,
漏了 CARLA↔nuScenes 的 y 符号翻转(`yaw_carla = −az_nus`)⇒ 四个侧/后相机左右镜像
(FRONT_LEFT 差 110.3°、BACK_LEFT 差 217.2°)。前/后相机因近自逆而"看起来对",
所以长期没暴露。同时 pitch/roll 被硬编码 0(官方实测 |pitch| 最大 0.96°)。
镜像表见 camera_rig 模块头注。

落盘:
  outputs/surround_<scene>/cam_front/000000.png ...(6 视角)
  calib.json        — 每相机 sensor2ego + intrinsic(3x3)
  ego_pose.json     — 逐帧 ego2global(CARLA 世界系)

用法:
  python -m autodrivedata.sim.collect_surround --frames 100 [--scene day_clear] [--npc-vehicles 15]
  python -m autodrivedata.sim.collect_surround --frames 400 --map Town13   # 多图扩数据:运行时切图
  python -m autodrivedata.sim.collect_surround --spawn-index 88 --stride 5 --frames 100  # 指定起点 + 0.5s/帧

多图切图(§5.14 Phase 2):`--map` 用 `client.load_world` 运行时切换(默认不动当前图,
零副作用;每次切换 ~2 分钟加载)。**已采集数据的图标记**:default_map 写进 calib.json
顶层 `"map"` 键,供 assemble/merge 溯源(旧产物无此键 = Town10HD_Opt)。

── 画幅与内参口径(2026-09-23,§P-M.11 冻结;改这里前先读 §P-M.7)──────────────
本采集器**历史上是第三处「声明 ≠ 渲染」**:六路共用一个 `cam_bp` 的 `fov=90`,K 也是
`calib_from_fov(w, h, 90)` 一表六用,而官方逐通道水平 FoV 是 **64.31–64.96°×5 + 89.34°**
—— 渲染视野与落盘口径差 25°。现改为**逐相机蓝图 + 逐通道 fov/K**,与 `collect_nus.py`
(§P-M.7)同一条不变量:**渲染与声明由同一份常量导出**。

- **画幅 1600×900**(nuScenes 官方),不再沿用 KITTI 口径的 1242×375;
- **fov** ← `export.nuscenes.NUS_CAMERA_FOV[cam]`(由官方逐通道 fx 反推);
- **K** ← `calib_from_fov(1600, 900, 该通道 fov)`,即 **fx 取官方值、主点取 corner
  `(w−1)/2`** —— 与 `wide` rig 的 `_wide_intrinsics` 同构造。

⚠️ **为什么不直接落官方 K 的 cx(792–829)**:官方主点是**真实相机的装配公差**,而我们的图
是 **CARLA 渲染栅格**,其光栅中心按 §P-M 的 corner 裁决恒为 `(w−1)/2 = 799.5`。若把官方
cx 写进**我方投影链**(GKT / mapviz / eval)消费的 K,world→image 就会**逐通道不一致地偏
7–27 px**(CAM_FRONT_LEFT 最大),等价 0.05–0.2 m 的 BEV 采样偏置 —— 又变成"声明 ≠ 渲染"。
`collect_nus.py` 落官方 cx 是因为**消费方是 devkit / auto3dlabel**(按官方口径解释);本采集器
的消费方是**我们自己的投影代码**,故取 corner。**两处口径不同是角色不同,不是不一致,别来统一**。

⚠️ **绝不改 `carla_common.CAM_ATTRS`**(1242×375/fov90):它被 KITTI 线、P1 A/B 线、静态 GT、
灯态、`probe_calib`、studio 等十余处引用,**动它就是动 P1 复现性红线**。本表是独立常量。
"""

from __future__ import annotations

import argparse
import json
import queue
import time
from typing import Any, cast

import carla
import numpy as np

from autodrivedata.calib.camera_rig import NUS_CAMERA_RIG, NUS_CAMERA_YAW
from autodrivedata.calib.depth_codec import decode_depth
from autodrivedata.calib.probe_calib import decode_instance
from autodrivedata.gt.export.nuscenes import NUS_CAMERA_FOV, NUS_CAMERA_HEIGHT, NUS_CAMERA_WIDTH
from autodrivedata.map.mapviz import calib_from_fov
from autodrivedata.perception.inst_tags import encode_instance_png
from autodrivedata.perception.sem_tags import encode_tag_png
from autodrivedata.sim.carla_common import assert_synced, load_world, loc, spawn_ego, spawn_ego_at, sync_mode
from autodrivedata.sim.collect_drive import spawn_route_walkers, spawn_traffic
from autodrivedata.sim.scenarios import SCENES, merged_weather
from autodrivedata.utils.paths import project_path

# 相机名 → 相对 ego 的 yaw(度)。**别名**,真值在 `autodrivedata/camera_rig.py`
# (`NUS_CAMERA_RIG` 的完整 (平移, (pitch,yaw,roll)));本表只用于**遍历相机名的顺序**
# 与"只关心偏航"的零散打印 —— 挂载/落盘一律走 `NUS_CAMERA_RIG`(含 pitch/roll)。
SURROUND_CAMS: dict[str, float] = dict(NUS_CAMERA_YAW)

# 本采集器专用画幅(nuScenes 官方 1600×900)。**不是** carla_common.CAM_ATTRS(1242×375,
# 那条是 KITTI/P1 线,勿动,见模块头注)。
SURROUND_CAM_ATTRS = {"image_size_x": str(NUS_CAMERA_WIDTH), "image_size_y": str(NUS_CAMERA_HEIGHT)}


def tag_from_semantic_image(image: carla.Image) -> np.ndarray:
    """语义相机帧 → tag 图 `(H, W)` uint8。

    ⚠️ **必须先 `convert(Raw)`**:默认转换器是 `CityScapesPalette`,不转拿到的是
    **上色预览**(人眼好看、数值全错)。Raw 下 `B=G=0`、`A=255`,**tag 在 R 通道** ——
    2026-10-01 实测确认(每个 tag 的调色板色与 CARLA 源码 29 色表逐个对,16/16 全中),
    不是从文档抄的。见 [`perception/sem_tags.py`](../perception/sem_tags.py) 头注。
    """
    image.convert(carla.ColorConverter.Raw)
    buf = np.frombuffer(image.raw_data, dtype=np.uint8)
    return buf.reshape(image.height, image.width, 4)[:, :, 2].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="输出根目录(默认 outputs/surround_<scene 或 drive>)")
    ap.add_argument("--scene", default=None, choices=sorted(SCENES), help="corner case 场景档(天气覆写)")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--npc-vehicles", type=int, default=15)
    ap.add_argument("--npc-walkers", type=int, default=6)
    ap.add_argument("--route-walkers", type=int, default=6, help="沿 ego 初始朝向布置的行人数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--sem",
        action="store_true",
        help="**逐相机**多挂一路 `sensor.camera.semantic_segmentation`(同挂点同 fov),"
        "落 `sem_<cam>/{fid}.png` = 8 位灰度、tag 即像素值(无损)—— 这是 P2-A 的"
        "**像素级分割 GT**,给 `perception/sem_eval.py` 当裁判用。默认关(对既有管线零改动;"
        "6 路变 12 路,采集与磁盘都翻倍)",
    )
    ap.add_argument(
        "--inst",
        action="store_true",
        help="**逐相机**多挂一路 `sensor.camera.instance_segmentation`(同挂点同 fov),"
        "落 `inst_<cam>/{fid}.png` = 16 位灰度、**像素值 = CARLA actor id**。"
        "R 通道同时带 `CityObjectLabel` 语义类 ⇒ **一台相机同时给「有几个」与「各是什么」**,"
        "正是实例分割 GT(mask AP / PQ 要的就是这两样)。默认关(对既有管线零改动)",
    )
    ap.add_argument(
        "--depth",
        action="store_true",
        help="**逐相机**多挂一路 `sensor.camera.depth`(同挂点同 fov),落 "
        "`depth_<cam>/{fid:06d}.npy` = **uint16 毫米**。默认关(对既有管线零改动)。"
        "★ **为什么必须有它**:判据侧(`perception/sem_bev`)对**所有类**一律拿射线与"
        "**地平面**求交 —— 那对路面是对的,对**离地 1–2 m 的物体**会把射线送过物体头顶,"
        "落到中位 **120 m** 外(实测 ±30 m 窗口内 **0.0%**)⇒ BEV 障碍物通道**结构性恒空**。"
        "修法就是这一路深度(`mask_to_bev_depth`)。",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument(
        "--map",
        default=None,
        help="目标地图(如 Town13/Town15;None = 当前服务器默认图)。运行时 load_world 切图,供多图扩数据",
    )
    ap.add_argument(
        "--spawn-index",
        type=int,
        default=None,
        help="固定用第 N 个 spawn point 出生(多段采集用);None = 沿用 spawn_ego 的首个空位(旧行为)",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="每 N 个 tick 存一帧(1 = 旧行为)。5 ⇒ 0.5s/帧 = nuScenes 关键帧率(2Hz)",
    )
    args = ap.parse_args()
    if args.stride < 1:
        ap.error("--stride 必须 ≥ 1")

    scene = SCENES[args.scene] if args.scene else None
    if scene is not None:
        for key, n in scene.traffic.items():
            setattr(args, key, n)
    if args.out is None:
        args.out = f"outputs/surround_{scene.name}" if scene else "outputs/surround_drive"

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    if args.map is not None:
        # 走 `carla_common.load_world` —— 切图期间的超时**必须**比平时的 60 s 大:
        # 本行原来的注释自己写着「~2min」而超时还是 60 s,大图(Town13 12478 个
        # spawn point)一跑就 `time-out of 60000ms`,而**世界其实加载成功了**
        print(f"[map] load_world {args.map}(运行时切图,~2min)...")
        load_world(client, args.map)
    world = client.get_world()
    default_map = args.map or world.get_map().name
    sync_mode(world)

    if scene is not None:
        world.set_weather(carla.WeatherParameters(**merged_weather(scene)))
        print(f"[scene] {scene.name} [{scene.group}] — 覆写 {sorted(scene.weather)}")

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    ego = spawn_ego_at(world, args.spawn_index) if args.spawn_index is not None else spawn_ego(world)
    ego.set_autopilot(True, tm.get_port())
    tm.vehicle_percentage_speed_difference(ego, 30.0)
    # 采集多样性:红绿灯等待在环视数据里是重复帧(250+ 帧原地,占比拉满),
    # 训练多样性被稀释——采集侧按百分比忽略红灯(不影响 TM 其他车与车道保持)
    tm.ignore_lights_percentage(ego, 100.0)
    print("[ego] autopilot on (TM 8000, 70% speed, 忽略红绿灯)")

    ego_t = ego.get_transform()
    spawn_traffic(world, tm, args.npc_vehicles, args.npc_walkers, args.seed)
    spawn_route_walkers(world, ego_t, args.route_walkers)

    bp_lib = world.get_blueprint_library()

    cams: dict[str, carla.Sensor] = {}
    qs: dict[str, queue.Queue] = {}
    for name, (mount, rot) in NUS_CAMERA_RIG.items():
        x, y, z = mount
        tf = carla.Location(x=x, y=y, z=z)
        # 逐相机蓝图:每路设自己的 fov(官方逐通道 64.31–64.96°×5 + 89.34°),
        # **不能**共用一个 cam_bp —— 那就是历史上的"六路共用 90°"(见模块头注)。
        cam_bp = bp_lib.find("sensor.camera.rgb")
        for k, v in SURROUND_CAM_ATTRS.items():
            cam_bp.set_attribute(k, v)
        cam_bp.set_attribute("fov", f"{NUS_CAMERA_FOV[name]:.6f}")
        s = cast(
            carla.Sensor,
            world.spawn_actor(
                cam_bp,
                carla.Transform(tf, carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])),
                attach_to=ego,
            ),
        )
        q: queue.Queue = queue.Queue()
        s.listen(q.put)
        cams[name], qs[name] = s, q

    # ---- 语义 GT(P2-A;`--sem` 时逐相机多挂一路)----
    # **挂点/fov 必须与 RGB 那路逐字相同** —— 否则 tag 图与 RGB 图不再像素对齐,
    # 而"对不齐"在下游只表现为 mIoU 偏低,看不出是这里错了。
    # `--sem` 与 `--inst` 只差一个蓝图后缀 ⇒ **同一份挂载代码**(挂点/fov/画幅必须逐字
    # 相同,否则两条 GT 的像素不再对齐,而"不齐"在下游只表现为 IoU 偏低)。
    def _spawn_kind(kind: str) -> tuple[dict[str, carla.Sensor], dict[str, queue.Queue]]:
        cs: dict[str, carla.Sensor] = {}
        qq: dict[str, queue.Queue] = {}
        for name, (mount, rot) in NUS_CAMERA_RIG.items():
            bp_k = bp_lib.find(f"sensor.camera.{kind}")
            for k, v in SURROUND_CAM_ATTRS.items():
                bp_k.set_attribute(k, v)
            bp_k.set_attribute("fov", f"{NUS_CAMERA_FOV[name]:.6f}")
            s = cast(
                carla.Sensor,
                world.spawn_actor(
                    bp_k,
                    carla.Transform(
                        carla.Location(x=mount[0], y=mount[1], z=mount[2]),
                        carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]),
                    ),
                    attach_to=ego,
                ),
            )
            q = queue.Queue()
            s.listen(q.put)
            cs[name], qq[name] = s, q
        return cs, qq

    sem_cams: dict[str, carla.Sensor] = {}
    sem_qs: dict[str, queue.Queue] = {}
    if args.sem:
        sem_cams, sem_qs = _spawn_kind("semantic_segmentation")
    inst_cams: dict[str, carla.Sensor] = {}
    inst_qs: dict[str, queue.Queue] = {}
    if args.inst:
        inst_cams, inst_qs = _spawn_kind("instance_segmentation")
    # ★ 深度:与 RGB/sem/inst **同一份** `_spawn_kind`(同挂点、同 fov、同画幅)
    #   ⇒ 深度与 RGB 逐像素对齐。⚠️ 任一处不一致症状只是"BEV 偏低",看不出根因。
    depth_cams: dict[str, carla.Sensor] = {}
    depth_qs: dict[str, queue.Queue] = {}
    if args.depth:
        depth_cams, depth_qs = _spawn_kind("depth")

    fovs = " ".join(f"{n.replace('CAM_', '')}={NUS_CAMERA_FOV[n]:.2f}°" for n in SURROUND_CAMS)
    print(f"[cams] {len(cams)} 环视相机挂载(nuScenes 官方 6DoF 挂点,逐通道 fov):{fovs}")
    if args.sem:
        print(f"[sem]  +{len(sem_cams)} 语义相机(同挂点同 fov)⇒ 像素级分割 GT 落 sem_<cam>/")
    if args.inst:
        print(f"[inst] +{len(inst_cams)} 实例相机(同挂点同 fov)⇒ 实例分割 GT 落 inst_<cam>/")

    # ★ **每一路队列都必须逐 tick 抽干**,包括后来加的 sem/inst —— 传感器队列是 FIFO,
    #   `get()` 取的是**最旧**那帧。漏掉一路的代价不是"少收几帧",而是**那一路整体
    #   滞后 N 帧**(N = 被漏掉的 tick 数),而它与别的路**都"有数据"、帧号也对得上**。
    #   2026-10-01 实测:漏了 `inst_qs` ⇒ 首帧"实例相机 R 通道 vs 语义相机 tag"相符
    #   **0.76–0.84**(应当 1.000),而**是采集器自己的自证把它抓出来的** ——
    #   不看那个数,这条只会在下游表现为"PQ 的语义项莫名其妙偏低"。
    for _ in range(5):  # 预热
        world.tick()
        for q in (*qs.values(), *sem_qs.values(), *inst_qs.values(), *depth_qs.values()):
            q.get(timeout=10)

    w, h = NUS_CAMERA_WIDTH, NUS_CAMERA_HEIGHT
    # sensor2ego = [x, y, z, yaw, pitch, roll] 度(infos 口径)——由官方标定导出,
    # 与上面 spawn 用的是**同一份** NUS_CAMERA_RIG,不存在"布置与落盘两处维护"。
    # intrinsic 逐通道,且取的是**该通道 spawn 时用的那个 fov**(calib_from_fov 的 fx 与
    # CARLA 蓝图 fov 定义同一 ⇒ 渲染视野 == 落盘 K;主点 corner,理由见模块头注)。
    calib: dict[str, Any] = {
        name: {
            "sensor2ego": [mount[0], mount[1], mount[2], rot[1], rot[0], rot[2]],
            "intrinsic": calib_from_fov(w, h, NUS_CAMERA_FOV[name])["intrinsic"],
        }
        for name, (mount, rot) in NUS_CAMERA_RIG.items()
    }

    out = project_path(args.out)
    for name in SURROUND_CAMS:
        (out / name.lower()).mkdir(parents=True, exist_ok=True)
        if args.sem:
            (out / f"sem_{name.lower()}").mkdir(parents=True, exist_ok=True)
        if args.inst:
            (out / f"inst_{name.lower()}").mkdir(parents=True, exist_ok=True)
        if args.depth:
            (out / f"depth_{name.lower()}").mkdir(parents=True, exist_ok=True)
    # 数据溯源(照 `"map"` 键的既有做法):旧产物无这些键 = 1242×375/六路共用 90°/stride 1
    calib["map"] = default_map  # 该采集来自哪张图(旧产物无此键 = Town10HD_Opt)
    calib["spawn_index"] = args.spawn_index  # None = spawn_ego 首空位
    calib["stride"] = args.stride
    calib["image_size"] = [w, h]
    # 语义 GT 的溯源:读的人必须知道"这个 root 有没有 sem_*/、tag 是哪一版编号"。
    # 旧产物无此键 = 没采语义。tag 编号见 `perception/sem_tags.py`(= CARLA CityObjectLabel)。
    calib["semantic"] = "carla.CityObjectLabel/R-channel" if args.sem else None
    # 实例 GT 的溯源。口径见 `perception/inst_tags.py`:id = G + 256·B,R 通道 = CityObjectLabel。
    calib["instance"] = "carla.actor-id(G+256*B)/R-channel-class" if args.inst else None
    # 深度的溯源。**单位必须写死在这里** —— 读的人不会去翻采集器源码。
    calib["depth"] = "uint16 millimetres (.npy, optical-axis z)" if args.depth else None
    with open(out / "calib.json", "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=1)

    poses: list[dict] = []
    t0 = time.monotonic()
    try:
        for i in range(args.frames):
            # stride > 1:中间 tick 也必须**逐路抽干队列** —— 只取被测帧会让其余相机积压,
            # 下一帧读到的是更早的陈旧图(§P-M.7 判据 ⑥ 同款坑)。故先循环 tick 并每 tick 收齐。
            drained: dict[str, carla.Image] = {}
            sem_drained: dict[str, carla.Image] = {}
            inst_drained: dict[str, carla.Image] = {}
            depth_drained: dict[str, carla.Image] = {}
            for _ in range(args.stride):
                world.tick()
                drained = {name: qs[name].get(timeout=10) for name in SURROUND_CAMS}
                sem_drained = {n: sem_qs[n].get(timeout=10) for n in SURROUND_CAMS} if args.sem else {}
                inst_drained = {n: inst_qs[n].get(timeout=10) for n in SURROUND_CAMS} if args.inst else {}
                depth_drained = {n: depth_qs[n].get(timeout=10) for n in SURROUND_CAMS} if args.depth else {}
            # ★ **同 tick 自证**(只查不行为):几路相机的 `frame` 号必须相等。
            #   ⚠️ 它抓的是"几路彼此**不同步**",抓不到"几路一起恒定滞后" ——
            #   后者要靠判据侧那条**跨方法一致**的对照(见 `sem_bev.mask_to_bev_depth` 头注)。
            #   返回的一行人读摘要只在**第一帧**打一次(每帧打会把日志刷满)。
            sync_note = assert_synced(
                [(n, drained[n]) for n in SURROUND_CAMS]
                + [(f"sem/{n}", sem_drained[n]) for n in sem_drained]
                + [(f"inst/{n}", inst_drained[n]) for n in inst_drained]
                + [(f"depth/{n}", depth_drained[n]) for n in depth_drained]
            )
            if i == 0 and sync_note:
                print(f"[sync] {sync_note}")
            for name, image in drained.items():
                tmp = out / f".tmp_{i}_{name}.png"
                image.save_to_disk(str(tmp))
                tmp.rename(out / name.lower() / f"{i:06d}.png")
            for name, image in sem_drained.items():
                # **走 PIL 而不是 `save_to_disk`**:后者写的是当前转换器下的图(默认上色预览),
                # 而我们要的是"tag 即像素值"的 8 位灰度。见 `tag_from_semantic_image`。
                (out / f"sem_{name.lower()}" / f"{i:06d}.png").write_bytes(
                    encode_tag_png(tag_from_semantic_image(image))
                )
            for name, image in depth_drained.items():
                # ⚠️ **不能 `save_to_disk`** —— 那存的是 CARLA 上色后的 PNG(好看不是米)。
                # 解码口径的唯一裁决在 `calib/depth_codec`(纯值)。
                # **落 uint16 毫米**:盘只剩 23 GB,float32 米在 200 帧×6 路下约 7 GB。
                (out / f"depth_{name.lower()}" / f"{i:06d}.npy").write_bytes(
                    (decode_depth(image.raw_data, image.height, image.width) * 1000.0)
                    .clip(0, 65535)
                    .astype(np.uint16)
                    .tobytes()
                )
            for name, image in inst_drained.items():
                # **走 PIL 不走 `save_to_disk`**:实例相机的默认转换器同样是上色预览,
                # 而我们要的是"像素值 = actor id"的 16 位灰度。解码口径的唯一裁决在
                # `calib/probe_calib.decode_instance`(差分实验),这里只是它的落盘侧。
                ids = decode_instance(image.raw_data, image.height, image.width)
                (out / f"inst_{name.lower()}" / f"{i:06d}.png").write_bytes(encode_instance_png(ids))
                # ★ 实测断言(只在首帧做一次):实例相机的 **R 通道 == 语义相机的 tag**。
                #   这条成立时一台相机同时给实例与类(PQ 的语义项白给);不成立时
                #   "拿实例相机当语义相机用"全线是错的,而症状是 **PQ 语义项恒为 0** ——
                #   会被读成"模型的类报得差"。同一帧两路都在时顺手比一次,不额外 tick。
                if i == 0 and args.sem and name in sem_drained:
                    cls_ch = (
                        np.frombuffer(image.raw_data, dtype=np.uint8)
                        .reshape(image.height, image.width, 4)[:, :, 2]
                        .copy()
                    )
                    tag0 = tag_from_semantic_image(sem_drained[name])
                    m = float((cls_ch == tag0).mean())
                    print(f"[inst] 首帧 {name}:实例相机 R 通道 vs 语义相机 tag 逐像素相符 {m:.6f}")

            egot = ego.get_transform()
            poses.append(
                {
                    "frame": i,
                    "tick": i * args.stride,  # 仿真 tick 号(stride>1 时与 frame 不等)
                    "x": round(egot.location.x, 3),
                    "y": round(egot.location.y, 3),
                    "z": round(egot.location.z, 3),
                    "yaw": round(egot.rotation.yaw, 3),
                    "pitch": round(egot.rotation.pitch, 3),
                    "roll": round(egot.rotation.roll, 3),
                }
            )
            if (i + 1) % 10 == 0 or i == args.frames - 1:
                dt = time.monotonic() - t0
                fps = (i + 1) / dt
                print(
                    f"[frame {i + 1}/{args.frames}] ego @ {tuple(round(v, 1) for v in loc(egot))} | {fps:.1f} fps"
                )
    finally:
        with open(out / "ego_pose.json", "w", encoding="utf-8") as f:
            json.dump(poses, f, indent=1)
        # 语义相机也要收 —— 与雷达同一条:`sensor.*` 不在下面那个 vehicle/walker/controller
        # 过滤器里,漏收会留在世界里阻塞下一次采集的 spawn。
        for s in (*cams.values(), *sem_cams.values(), *inst_cams.values()):
            s.stop()
            s.destroy()
        for a in world.get_actors():
            if (
                a.type_id.startswith("vehicle")
                or a.type_id.startswith("walker")
                or a.type_id.startswith("controller")
            ):
                a.destroy()
    print(
        f"[done] surround root: {out.resolve()} "
        f"({args.frames} frames × {len(cams)} cams @ {w}×{h}, stride {args.stride}"
        f" = {args.frames * args.stride * 0.1:.1f}s 仿真时长)"
    )


if __name__ == "__main__":
    main()

"""M1b 静态采集:ego 静止 + NPC + 6 相机 + LiDAR + 5 雷达 → nuScenes 迷你集(scene-0103)。

用法(base env,CARLA 服务器运行中):
  python -m autodrivedata.sim.collect_nus [--out outputs/nus_mini] [--frames 2] [--rig nuscenes|wide]

落盘 = 标准 nuScenes dataroot(devkit 直读,auto3dlabel nuscenes-queue 消费):
  {out}/v1.0-mini/*.json(14 表)+ samples/LIDAR_TOP/*.bin((N,5) raw)
  + samples/RADAR_*/*.pcd(18 字段 radar 点云)+ samples/CAM_*/*.png
坐标系:入表数据经 geometry.CARLA_TO_NUS 翻 y(入 nuScenes 全局系 y 左)。

**全传感器标定口径 = 「渲染位姿」与「声明位姿」同源**(2026-09-23,Plan2.md §P-M.7):
本采集器此前是**表对了、图错了**的新失效模式——`calibrated_sensor` 写官方正确值,而 6 相机
实际 spawn 在 LiDAR 挂点 + 旧镜像偏航表上(四路侧/后相机左右互换)、雷达偏航差 94–136°、
LiDAR 无旋转。现在**每一项都由同一份官方常量导出**:

| 传感器 | 渲染(spawn) | 声明(calibrated_sensor) | 共同来源 |
|---|---|---|---|
| 6 相机 | `--rig` 选中的 rig 表(挂点 + 6DoF 姿态) | 同一 rig 的 calib 表 | `camera_rig`(同一次推导) |
| LiDAR | `LIDAR_MOUNT` + `LIDAR_ROT` | `NUS_LIDAR_CALIB` | `export/nuscenes` |
| 5 雷达 | `NUS_RADAR_OFFSETS` 的 CARLA 镜像 | `NUS_RADAR_OFFSETS` 原值 | `export/nuscenes` |

相机蓝图 `fov` 逐通道由**该 rig 的 K** 反推(官方 rig:64.31–64.96°,CAM_BACK 89.34°)——
否则"标定说 64°、图像是 90°"又是同一类声明≠渲染。判据(全数值)见 `autodrivedata/calib/verify_nus_calib.py`。

★ **ego 原点(2026-09-23,Plan2 §P-M.10)**:上表的"声明"列全部是 **nuScenes ego 系**,
其原点 = **后轴中心在地面**;而 CARLA 车辆 actor 的原点 = **车身长度中点**。故
`x_carla = x_nus + NUS_EGO_ORIGIN_X`(= −1.2563 m,实测 a2 后轴),`ego_pose` 也必须写
**后轴点**的位姿。历史实现两处都按 actor 原点办 ⇒ 6 相机 + 6 雷达 + LiDAR 整体前移 1.26 m
(实测 `CAM_BACK` 画幅 42.999% 是自身车体、`RADAR_FRONT` 悬在车头前 1.56 m)。

`--rig wide`(2026-09-23,Plan2 §P-M.8)= **自定义宽视口口径**(前 55°/后侧 110°/后 120°、
后三路挂点后移到车尾 x=−1.90 ⇒ 零车体像素),与官方口径**并存**、默认仍是官方。
宽视口下相机内参由 FoV 反推且主点走 corner 约定(wide 的图是 CARLA 渲染栅格),见
`export/nuscenes._wide_intrinsics`。**雷达与 LiDAR 不受 rig 影响。**

雷达(2026-09-14 扩展):5 通道照官方 nuScenes 布局(RADAR_FRONT/FRONT_LEFT/
FRONT_RIGHT/BACK_LEFT/BACK_RIGHT),参数复刻官方大陆 ars408(77°×14.2°、range
250);同步模式固定 0.1s → 与 LiDAR 同为 10Hz(官方 13Hz 无法复刻)。挂载 = 官方
calibrated_sensor 位姿的 CARLA 镜像(spawn location y 取负、yaw = −az_nus),
入表零转换直接用官方原值 → 雷达锥与同名相机同物理象限。radar 点进 GT 关联:
5 通道点各经 points_sensor_to_global_nus → 全局系合并 → count_points_in_box_nus
计 num_radar_pts。
"""

from __future__ import annotations

import argparse
import math
import queue
from typing import cast

import carla
import numpy as np

from autodrivedata.calib.camera_rig import NUS_CAMERA_RIG, NUS_WIDE_CAMERA_RIG
from autodrivedata.gt.core import ActorBox, box_center_world, box_heading_world, classify_nus
from autodrivedata.gt.export.nuscenes import (
    NUS_CAMERA_CALIBS,
    NUS_CAMERA_FOV,
    NUS_CAMERAS,
    NUS_LIDAR_CALIB,
    NUS_LIDAR_MOUNT_CARLA,
    NUS_RADAR_CHANNELS,
    NUS_RADAR_MOUNTS_CARLA,
    NUS_RADAR_OFFSETS,
    NUS_RIGS,
    NusSample,
    camera_calibs,
    camera_fov,
    count_points_in_box_nus,
    points_sensor_to_global_nus,
    radar_yaw_offset_carla,
    write_mini_dataset,
)
from autodrivedata.perception.radar import detections_to_nus18, mask_radar_points, nus18_to_pcd
from autodrivedata.sim.carla_common import (
    LIDAR_ATTRS,
    loc,
    rad,
    spawn_ego,
    spawn_npcs,
    sync_mode,
)
from autodrivedata.utils import geometry as g
from autodrivedata.utils.paths import project_path

# 相机蓝图属性:分辨率 + **逐通道 fov**。`fov` 是**水平** FOV,官方 K 反推值见
# `NUS_CAMERA_FOV`;六路共用 90° 会让五路"应该是 64.3°"的相机被渲染成 90°(声明≠渲染)。
CAM_ATTRS_COMMON = {
    "image_size_x": "1600",
    "image_size_y": "900",
}  # nuScenes 分辨率
CAM_FOV: dict[str, float] = dict(NUS_CAMERA_FOV)  # 默认 rig(=官方口径),供只读官方表的调用方


def rig_tables(rig: str) -> tuple[dict, dict, dict]:
    """rig 名 → `(CARLA 侧 spawn rig, nuScenes 侧落盘标定, 逐通道 fov)`。

    **三项必须来自同一个 rig**——采集器只经本函数取表,不许就地写第二份(这正是 §P-M.7
    "表对了、图错了"的成因:spawn 用的表和落盘的表各写一份,对不上时两边都自洽)。

    `nuscenes` = 官方标定(默认,行为不得变);`wide` = 自定义宽视口口径(前 55°/后侧 110°/
    后 120°、后三路挂点后移到车尾,见 `camera_rig` 模块头注 + Plan2 §P-M.8)。
    """
    if rig == "nuscenes":
        return NUS_CAMERA_RIG, NUS_CAMERA_CALIBS, NUS_CAMERA_FOV
    if rig == "wide":
        # 后两表由 `export/nuscenes` 按同一 rig 名派发(wide 的 K 由 FoV 反推,见其注释)
        return NUS_WIDE_CAMERA_RIG, camera_calibs(rig), camera_fov(rig)
    raise ValueError(f"未知相机 rig:{rig!r}(可选 {NUS_RIGS})")


# LiDAR 挂点 / 姿态(CARLA 口径)。
# 挂点**由官方 nus 系标定经 `nus_mount_to_carla` 导出**(x 加 ego 原点差 + y 翻号),
# 见 `export.nuscenes.NUS_LIDAR_MOUNT_CARLA`;姿态由官方四元数导出。两者都不手抄 ——
# 手抄正是"表对了、图错了"的成因(§P-M.7),把 x 当 actor 系距离用是同一个坑的第二代(§P-M.10)。
LIDAR_MOUNT: tuple[float, float, float] = NUS_LIDAR_MOUNT_CARLA
_LIDAR_RPY = g.rotation_matrix_to_carla(g.nus_sensor_rotation_to_carla(NUS_LIDAR_CALIB[1]))
LIDAR_ROT: tuple[float, float, float] = (
    math.degrees(_LIDAR_RPY[0]),
    math.degrees(_LIDAR_RPY[1]),
    math.degrees(_LIDAR_RPY[2]),
)  # ≈ (−0.3380, +89.8835, −1.3884)

# 5 雷达视角(相对 ego,CARLA yaw 度,左转正)—— **由官方标定导出**,不手抄:
# CARLA 侧 `yaw_carla = −az_nus`,与相机同一条规则(`carla_yaw_to_nus_yaw` 的逆)。
# 旧表 {0,+45,−45,+90,−90} 是"与同名相机同号"的猜测,四路角雷达实测差 94–136°。
RADAR_YAW_OFFSET: dict[str, float] = {ch: radar_yaw_offset_carla(ch) for ch in NUS_RADAR_CHANNELS}

# 官方大陆 ars408 雷达参数(复刻;horizontal 77°/vertical 14.2°/range 250m)。
# **CARLA 0.9.16 两 FOV 属性交叉使用**(编译行为,prebuilt 无法改源码):
#   实际 azi 半角 = vertical_fov/2、实际 alt 半角 = horizontal_fov/2
#   (12 组属性扫描自洽:设 hf=77/vf=14.2 → 实测 azi±7°/alt±38°,与 ars408 完全反了)。
# 故对调属性值获得真实 ars408 锥:设 hf=14.2/vf=77 → 实测 azi±38.1°/alt±7.0° ✓
RADAR_ATTRS = {
    "horizontal_fov": "14.2",
    "vertical_fov": "77",
    "range": "250",
    "points_per_second": "3300",
    "sensor_tick": "0.1",
}


def _latest(q: queue.Queue):
    """取队列里最新一帧(非阻塞);空则 None。

    同步模式下**只有 world.tick() 才产生新帧**,阻塞 get(timeout=10) 是死等待
    (不 tick 数据永远不会来,等满超时抛 Empty)。radar 的 sensor_tick=0.1 与
    同步 tick 同周期,存在相位偶发空 tick(实测 30 tick 2 空,~7%)→ 一律
    drain 式取帧,空由调用方补 tick 处理(C22)。lidar/相机每 tick 必有,
    仍用阻塞 get 保时序。
    """
    frame = None
    while True:
        try:
            frame = q.get_nowait()
        except queue.Empty:
            return frame


def _actor_to_annotation(
    a: carla.Actor,
    points_global: np.ndarray,
    points_radar_global: np.ndarray,
    origin_nus: tuple[float, float, float],
) -> dict | None:
    """actor → nuScenes GT 标注 dict(全局系,已按场景原点平移;类别忽略类返回 None)。

    points_global 是"以场景起点为原点"的 nus 系(采集主循环经 ego_trans_nus
    = ego−origin 平移),GT 框中心必须同口径平移,否则关联计数全空。
    """
    category = classify_nus(a.type_id)
    if category is None:
        return None
    bb = a.bounding_box
    box = ActorBox(
        type_id=a.type_id,
        extent=(bb.extent.x, bb.extent.y, bb.extent.z),
        location=(bb.location.x, bb.location.y, bb.location.z),
        rotation=rad(bb.rotation),
        actor_location=loc(a.get_transform()),
        actor_rotation=rad(a.get_transform().rotation),
    )
    center_nus = g.carla_to_nus_global(box_center_world(box)[None])[0]
    center_nus = center_nus - np.asarray(origin_nus, dtype=np.float64)
    center_xyz = (float(center_nus[0]), float(center_nus[1]), float(center_nus[2]))
    heading_nus = g.CARLA_TO_NUS @ box_heading_world(box)
    yaw_nus = float(np.arctan2(heading_nus[1], heading_nus[0]))
    ex = bb.extent
    size = (2 * ex.y, 2 * ex.x, 2 * ex.z)  # (w,l,h)
    num_pts = count_points_in_box_nus(points_global, center_xyz, size, yaw_nus)
    num_radar = count_points_in_box_nus(points_radar_global, center_xyz, size, yaw_nus)
    return {
        "category": category,
        "translation": center_xyz,
        "size": size,
        "yaw_nus": yaw_nus,
        "num_lidar_pts": num_pts,
        "num_radar_pts": num_radar,
        "instance_token": f"adinst{a.id:x}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/nus_mini")
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument(
        "--rig",
        choices=NUS_RIGS,
        default="nuscenes",
        help="相机口径:nuscenes=官方标定(默认)/ wide=自定义宽视口(见 camera_rig 头注)",
    )
    args = ap.parse_args()

    cam_rig, cam_calibs, cam_fov = rig_tables(args.rig)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)

    ego = spawn_ego(world)
    ego_t = ego.get_transform()
    print(f"[ego] {ego.type_id} @ actor {loc(ego_t)} → nus 原点(后轴) x += {g.NUS_EGO_ORIGIN_X:+.4f} m")

    bp_lib = world.get_blueprint_library()
    lid_bp = bp_lib.find("sensor.lidar.ray_cast")
    for k, v in LIDAR_ATTRS.items():
        lid_bp.set_attribute(k, v)
    # LiDAR:挂点 + **姿态**都要设。不设 rotation 等于宣称"传感器系 = ego 系"——
    # 点云存的是传感器自身系,devkit 按 `calibrated_sensor.rotation` 解释,缺了它
    # 整片点云绕 z 转 90°(实测 `num_lidar_pts` 复现比值 1.0000 → 0.0854)。
    lidar = cast(
        carla.Sensor,
        world.spawn_actor(
            lid_bp,
            carla.Transform(
                carla.Location(*LIDAR_MOUNT),
                carla.Rotation(pitch=LIDAR_ROT[0], yaw=LIDAR_ROT[1], roll=LIDAR_ROT[2]),
            ),
            attach_to=ego,
        ),
    )
    # 6 相机:挂点 + 6DoF 姿态 + 逐通道 fov,全部由 `--rig` 选中的那一份表导出
    # (与写进 `calibrated_sensor` 的标定同源推导,见 `rig_tables`)。
    # 样板 = `autodrivedata/sim/collect_surround.py`(那一版从 §P-M 起就是对的)。
    cameras: dict[str, carla.Sensor] = {}
    for cam, (mount, rot) in cam_rig.items():
        cam_bp = bp_lib.find("sensor.camera.rgb")
        for k, v in CAM_ATTRS_COMMON.items():
            cam_bp.set_attribute(k, v)
        cam_bp.set_attribute("fov", f"{cam_fov[cam]:.6f}")
        tf = carla.Transform(
            carla.Location(x=mount[0], y=mount[1], z=mount[2]),
            carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]),
        )
        cameras[cam] = cast(carla.Sensor, world.spawn_actor(cam_bp, tf, attach_to=ego))
    # 5 雷达:挂载 = 官方 calibrated_sensor 的 CARLA 镜像(y 取负、yaw = −az_nus)
    radars: dict[str, carla.Sensor] = {}
    for ch in NUS_RADAR_CHANNELS:
        r_bp = bp_lib.find("sensor.other.radar")
        for k, v in RADAR_ATTRS.items():
            r_bp.set_attribute(k, v)
        t_carla = NUS_RADAR_MOUNTS_CARLA[ch]
        tf = carla.Transform(
            carla.Location(*t_carla),
            carla.Rotation(pitch=0.0, yaw=RADAR_YAW_OFFSET[ch], roll=0.0),
        )
        radars[ch] = cast(carla.Sensor, world.spawn_actor(r_bp, tf, attach_to=ego))
    print(
        f"[sensor] lidar {LIDAR_ATTRS['channels']}ch @ {LIDAR_MOUNT} rot "
        f"({LIDAR_ROT[0]:+.4f}, {LIDAR_ROT[1]:+.4f}, {LIDAR_ROT[2]:+.4f})° + "
        f"{len(radars)} radars(ars408) + 6 cameras "
        f"{CAM_ATTRS_COMMON['image_size_x']}x{CAM_ATTRS_COMMON['image_size_y']} "
        f"(rig={args.rig}, fov {min(cam_fov.values()):.2f}–{max(cam_fov.values()):.2f}°)"
    )

    spawn_npcs(world, ego_t)

    lid_q: queue.Queue = queue.Queue()
    cam_qs: dict[str, queue.Queue] = {c: queue.Queue() for c in NUS_CAMERAS}
    radar_qs: dict[str, queue.Queue] = {ch: queue.Queue() for ch in NUS_RADAR_CHANNELS}
    lidar.listen(lid_q.put)
    for cam, s in cameras.items():
        s.listen(cam_qs[cam].put)
    for ch, s in radars.items():
        s.listen(radar_qs[ch].put)

    # 预热:每个 tick 至少取到一帧,才认为同步就绪(雷达 sensor_tick 相位偶发空,
    # 已 drain 丢弃;5 tick 内必收敛——实测最多连续 2 空)
    warmed_radar = {ch: False for ch in NUS_RADAR_CHANNELS}
    for _ in range(10):  # 预热
        world.tick()
        lid_q.get(timeout=10)
        for q in cam_qs.values():
            q.get(timeout=10)
        for ch, q in radar_qs.items():
            if _latest(q) is not None:
                warmed_radar[ch] = True
        if all(warmed_radar.values()):
            break

    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # 官方每场景有独立全局坐标系原点(nuscenes_mini 实测 scene-0061/0103/0553 的
    # 首帧 ego 平移 = 各自场景基准,如 (411,1181,0)/(600,1647,0)/(1316,1039,0))——
    # 我们单场景采集也按此约定:采集中以 ego 首帧为原点,平移量 = 首帧负值,
    # 使本场景坐标系与官方"以场景为单元"的全局一致。
    _t0 = ego.get_transform()
    nus_origin = np.asarray(g.nus_ego_translation(loc(_t0), rad(_t0.rotation)))
    origin_nus = (float(nus_origin[0]), float(nus_origin[1]), float(nus_origin[2]))
    # 三张标定表**全部直接用原值**(nus 系,y 左),零转换:
    # - LiDAR:官方 translation + **四元数**(含 1.4289° up 轴倾角,是观测量)
    # - 相机:`--rig` 选中那一份的 translation + 6DoF 四元数(与 spawn 的表同源推导)
    # - 雷达:官方 translation + yaw(官方 pitch/roll 精确为 0,yaw-only 无损)
    # 采集侧 spawn 的 CARLA 镜像(y 取负、yaw = −az_nus)是**同一物理挂点的另一套表达**,
    # 由 `RADAR_YAW_OFFSET` / `cam_rig` 从这三张表导出 ⇒ 不存在"布置与落盘两处维护"。
    calib_lidar = NUS_LIDAR_CALIB
    calib_cameras = {c: cam_calibs[c] for c in NUS_CAMERAS}
    calib_radars = {ch: NUS_RADAR_OFFSETS[ch] for ch in NUS_RADAR_CHANNELS}

    samples: list[NusSample] = []
    try:
        for i in range(args.frames):
            world.tick()
            pts_raw = np.frombuffer(lid_q.get(timeout=10).raw_data, dtype=np.float32).reshape(-1, 4)
            pts_nus = g.carla_lidar_to_velodyne(pts_raw)  # y 翻转 = nus 传感器约定
            pts5 = np.hstack([pts_nus, np.zeros((len(pts_nus), 1), dtype=np.float32)])  # elongation=0
            lidar_rel = f"samples/LIDAR_TOP/{i:06d}.bin"
            (out / lidar_rel).parent.mkdir(parents=True, exist_ok=True)
            pts5.tofile(out / lidar_rel)

            # 5 雷达:raw → 18 字段 nus 点 → 提前按 devkit 过滤 → .pcd 落盘。
            # drain 取最新帧(C22 相位空 tick);空则跳过本帧(该通道本 tick 无点,
            # 写空 pcd——devkit 空编码 (18,0),不阻断采集)。
            radar_filenames: dict[str, str] = {}
            radar_nus18: dict[str, np.ndarray] = {}
            for ch in NUS_RADAR_CHANNELS:
                frame = _latest(radar_qs[ch])
                if frame is None:
                    print(f"  [warn] {ch} 本 tick 空(相位),写空 pcd")
                    nus18 = np.empty((0, 18), dtype=np.float32)
                else:
                    dets = np.frombuffer(frame.raw_data, dtype=np.float32).reshape(-1, 4)
                    nus18 = detections_to_nus18(dets, sensor_id=NUS_RADAR_CHANNELS.index(ch))
                    nus18 = nus18[mask_radar_points(nus18)]
                radar_nus18[ch] = nus18
                rel = f"samples/{ch}/{i:06d}.pcd"
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                (out / rel).write_bytes(nus18_to_pcd(nus18))
                radar_filenames[ch] = rel

            camera_filenames: dict[str, str] = {}
            for cam in NUS_CAMERAS:
                img = cam_qs[cam].get(timeout=10)
                rel = f"samples/{cam}/{i:06d}.png"
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                img.save_to_disk(str(out / rel))
                camera_filenames[cam] = rel

            ego_t_now = ego.get_transform()
            # ★ ego_pose 必须是 **nuScenes ego 原点(后轴中心)** 的位姿,不是 CARLA actor 原点
            # (车身长度中点)—— 整套 `calibrated_sensor.translation` 都以后轴为基准,
            # 写 actor 原点会让整组传感器相对自车偏 1.2563 m(§P-M.10)。见 `geometry.NUS_EGO_ORIGIN_X`。
            _ego_nus = g.nus_ego_translation(loc(ego_t_now), rad(ego_t_now.rotation))
            ego_trans_nus = (
                _ego_nus[0] - origin_nus[0],
                _ego_nus[1] - origin_nus[1],
                _ego_nus[2] - origin_nus[2],
            )
            # ★ 全 6DoF:**不能**用 `carla_yaw_to_nus_quat` —— 实测静止时车体俯仰 +0.0642°,
            # 拍平成纯偏航会让 ego_pose ⊕ calibrated_sensor 与世界系真值差这个量级(§P-M.10)。
            ego_rot_nus = g.nus_ego_rotation(rad(ego_t_now.rotation))

            pts_global = points_sensor_to_global_nus(
                pts_nus[:, :3],
                ego_trans_nus,
                ego_rot_nus,
                calib_lidar[0],
                calib_lidar[1],
            )
            # 5 雷达点各自进全局系 → 合并 → GT 框内计数 num_radar_pts
            # (与官方"当前 sample 全雷达通道命中总数"同口径)。
            # 雷达标定只有 yaw(官方 pitch/roll 精确为 0)⇒ 传 `yaw_to_quat(yaw)`,
            # 与 LiDAR 共用**同一个** `points_sensor_to_global_nus`(一条链,不分叉)。
            radar_pts_global: list[np.ndarray] = []
            for ch in NUS_RADAR_CHANNELS:
                radar_pts_global.append(
                    points_sensor_to_global_nus(
                        radar_nus18[ch][:, :3],
                        ego_trans_nus,
                        ego_rot_nus,
                        calib_radars[ch][0],
                        g.yaw_to_quat(calib_radars[ch][1]),
                    )
                )
            radar_global = np.vstack(radar_pts_global) if radar_pts_global else np.empty((0, 3))
            annotations: list[dict] = []
            for a in world.get_actors():
                if not (a.type_id.startswith("vehicle") or a.type_id.startswith("walker")):
                    continue
                ann = _actor_to_annotation(a, pts_global, radar_global, origin_nus)
                if ann is not None:
                    annotations.append(ann)

            samples.append(
                NusSample(
                    ego_translation=ego_trans_nus,
                    ego_rotation_nus=ego_rot_nus,
                    lidar_filename=lidar_rel,
                    camera_filenames=camera_filenames,
                    calib_lidar=calib_lidar,
                    calib_cameras=calib_cameras,
                    annotations=annotations,
                    timestamp=1700000000000000 + i * 50000,
                    radar_filenames=radar_filenames,
                    calib_radars=calib_radars,
                )
            )
            print(
                f"[sample {i}] {len(pts5)} lidar / {sum(len(v) for v in radar_nus18.values())} radar / "
                f"{len(annotations)} GT / 6 cams"
            )
    finally:
        lidar.stop()
        lidar.destroy()
        for s in cameras.values():
            s.stop()
            s.destroy()
        for s in radars.values():
            s.stop()
            s.destroy()
        for a in world.get_actors():
            if (
                a.type_id.startswith("vehicle")
                or a.type_id.startswith("walker")
                or a.type_id.startswith("controller")
            ):
                a.destroy()

    # mini_val = {scene-0103, scene-0916}——两个场景都要有,否则 nuscenes-queue 遍历 KeyError
    write_mini_dataset(
        out,
        "v1.0-mini",
        {"scene-0103": samples, "scene-0916": [samples[0]]},
        rig=args.rig,
    )
    print(f"[done] nuScenes dataroot: {out.resolve()} (2 scenes, {len(samples)}+1 samples, rig={args.rig})")


if __name__ == "__main__":
    main()

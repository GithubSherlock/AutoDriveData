"""P1-3 单变量 A/B 采集:ego 定速直行 + 路侧静置车(固定位置)→ KITTI 序列。

A/B 纪律(2026-09-08 教训):autopilot/TM 路线失控使帧内容不可对齐(两次
同 seed 采集 GT 分布差 7 倍),场景级比较无法归因。本采集器:
- ego 锚定 spawn point 0(yaw=0,朝世界 +x)+ set_target_velocity 定速直行(无 TM)
- 静置目标车放固定路肩位置(20/35/50/62m,+3.5m 横向),不 autopilot
- A/B 只变 weather(SCENES 场景档,sunset_glare 的 az=90=车头正前 经校准)
→ 帧级配对:同位置同车同角,唯一变量 = 光照(采集史:残留 actor 阻塞
pt0 曾致 fallback 反向出生点、65m 曾卡 GT 阈值——已修,详见 Plan.md §5.5a)。

用法: python -m autodrivedata.sim.collect_ab_route --scene day_clear --frames 220 [--out ...]
      python -m autodrivedata.sim.collect_ab_route --scene sunset_glare --frames 220 [--out ...]
      python -m autodrivedata.sim.collect_ab_route --scene day_clear --speed 4 --frames 140  # 参数扫描(定里程)
      python -m autodrivedata.sim.collect_ab_route --scene day_clear --occluders partial    # P1-6b 静态遮挡

## `--occluders`:第二个可变的单变量(2026-10-01)

动机是 P1-6 的 `dense_rush` 候选 —— 但 **`dense_rush` 不能这么做**:它是 `group="traffic"`、
`weather={}`,本采集器只读 `scene.weather`,于是 `--scene dense_rush` 会**静默退化成 day_clear**;
而真上车流要走 Traffic Manager,**违反 A/B 红线**(不可复现)且必然使两侧 GT 计数不等。

所以这条改测**静态遮挡**:`static.prop.*` 道具不是 `vehicle`/`walker`,**结构上进不了 GT**
(见下面收 `label_2` 的类型过滤)⇒ 同目标、逐帧 GT 相等、完全可复现、单一变量。
**产出必须自称"静态遮挡"而不是"密集车流"** —— 后者是这条路走不到的场景。

摆位规则见 [`sim/occlusion.py`](occlusion.py)(纯值、零 carla):遮挡物排在 **ego→车 的视线上**、
长轴垂直于视线,而不是 ego 正前方 —— 排在正前方会因视差在车内侧漏一条全高的缝。
"""

from __future__ import annotations

import argparse
import queue
from typing import Protocol, cast

import carla
import numpy as np

from autodrivedata.calib.core import CameraIntrinsics, KittiCalibOut, tr_velo_to_cam
from autodrivedata.calib.depth_codec import decode_depth
from autodrivedata.gt.core import ActorBox, box_to_gt_line
from autodrivedata.gt.export.kitti import write_frame
from autodrivedata.gt.export.nuscenes import NUS_RADAR_CHANNELS, NUS_RADAR_MOUNTS_CARLA
from autodrivedata.perception.radar import detections_to_nus18, mask_radar_points, nus18_to_pcd
from autodrivedata.perception.semantic import semantic_to_velodyne_bin
from autodrivedata.sim.carla_common import (
    CAM_ATTRS,
    LIDAR_ATTRS,
    SENSOR_OFFSET,
    ground_z_at,
    loc,
    rad,
    spawn_ego,
    sync_mode,
)
from autodrivedata.sim.collect_nus import RADAR_ATTRS, RADAR_YAW_OFFSET, _latest
from autodrivedata.sim.collect_slam import ego_pose_matrix
from autodrivedata.sim.collect_static_gt import assert_synced
from autodrivedata.sim.occlusion import (
    CAM_FWD,
    OCCLUDER_COVER,
    OCCLUDER_DIMS,
    OCCLUDER_GAP,
    OCCLUDER_MODELS,
    occluder_depth,
    project,
    required_span,
    sight_line_occluders,
)
from autodrivedata.sim.scenarios import SCENES, merged_weather
from autodrivedata.utils.paths import project_path

SPEED = 8.0  # m/s 定速


# 2026-09-09:65.0→62.0——第4台车曾恰好卡 GT max_distance=65.0 边界,起步
# 抖动使两侧 GT 数不等(216 vs 219),帧级配对失效;留 3m 裕量
class _WalkerCtrl(Protocol):
    """pycarla 桩缺 WalkerAIController 方法标注的最小协议(与 `carla_common` 同款)。"""

    def start(self) -> None: ...


STATIC_OFFSETS = [
    20.0,
    35.0,
    50.0,
    62.0,
]  # 路肩车沿 x 前向距离(m)(-y 侧 spawn 失败,全用 +y 路肩)
STATIC_LATS = [3.5] * 4  # 横向偏移(同侧路肩)
NPC_MODELS = [
    "vehicle.tesla.model3",
    "vehicle.audi.a2",
    "vehicle.ford.mustang",
    "vehicle.toyota.prius",
]

#: `--layout close` 的目标表:**多类 + 近距**,专为融合消融造的。
#:
#: ★ 为什么需要它(P1 的两条数据事实,见 Plan4 §P-V18 五/六):
#: ① **只有一类目标**(P1 的 GT 100% `Car`)⇒ 相机的**类信息无处发力**,
#:    「相机出类」那一行必然是 0.000 的差;
#: ② **33–60 m 的 LiDAR 回波不足以成框**(实测框内 4/17/49 点)⇒ 四分之三的目标
#:    进不了任何一档。两项都**不是判据能修的**。
#:
#: ⇒ 这一档把目标摆到 **7–24 m**(LiDAR 打得动),并混入**行人/骑行者**
#: (KITTI 类 `Pedestrian`/`Cyclist`,由 `gt.core.classify_kitti` 归一)。
#: ⚠️ **默认仍是 `p1`** —— 那一档的 d/lat 是 P1 已归档矩阵的复现前提,动它就是动红线。
CLOSE_TARGETS: tuple[tuple[str, float, float], ...] = (
    # (蓝图, 前向 m, 横向 m)。横向为正 = ego 右侧,与 P1 同侧。
    ("walker.pedestrian.0001", 7.0, 4.2),
    ("vehicle.tesla.model3", 9.0, 3.5),
    ("vehicle.gazelle.omafiets", 13.0, 4.4),  # 自行车 ⇒ KITTI `Cyclist`
    ("walker.pedestrian.0002", 16.0, 4.2),
    ("vehicle.audi.a2", 21.0, 3.5),
)


def _bbox_top_z(a: carla.Actor) -> float:
    """actor 的**世界**包围盒顶面 z。`bounding_box` 在 actor 局部系,必须经 `get_transform`。"""
    bb = a.bounding_box
    return float(a.get_transform().transform(bb.location).z + bb.extent.z)


def is_cleanup_target(type_id: str) -> bool:
    """这一份采集**自己生成**的 actor 吗?清场与收尾共用同一个判据(纯函数)。

    两处**必须同源**,否则是"清一半"这类不可见失败,本项目已经踩满三次:
    残留车阻塞 pt0(致 fallback 反向出生点)、残留雷达阻塞后续 spawn(2026-09-30)、
    残留**遮挡物**让下一轮 A 侧带着上一轮 B 侧的墙采完 —— 最后这条最阴,数据照出,
    只是 A/B 的差凭空小一截。**注意它宽于 GT 过滤器**:`controller` 要清但不进 GT,
    道具同理;两者混用一个谓词就会把道具写进 `label_2`,A/B 的 GT 相等立刻不成立。
    """
    return type_id.startswith(("vehicle", "walker", "controller")) or type_id in set(OCCLUDER_MODELS.values())


def dims_match(bb: carla.BoundingBox, dims: tuple[float, float, float], *, tol: float = 1e-3) -> bool:
    """实 spawn 的包围盒 `(长, 厚, 高)` 是否就是 `occlusion.OCCLUDER_DIMS` 记的那份。

    纯函数(不需要 CARLA),判据 = 逐项 `max|Δ| <= tol`。挡不挡得住是这三条边的函数,
    资产一换而声明没换 ⇒ **每一个可见性数字都错**,而采集照样跑得完。
    """
    got = (2.0 * bb.extent.x, 2.0 * bb.extent.y, 2.0 * bb.extent.z)
    return max(abs(a - b) for a, b in zip(got, dims, strict=True)) <= tol


def _measure_asset(
    world: carla.World, bp: carla.ActorBlueprint, dims: tuple[float, float, float], *, mode: str
) -> None:
    """把一个临时道具摆在 **yaw=0** 处读回真实尺寸,与声明对表,对不上就报错。

    **为什么必须另起一个 yaw=0 的探针,而不是读实摆的那几个**:`Actor.bounding_box`
    对**转过**的 actor 给出一个 `(extent, rotation)` 自相矛盾的读数 —— 同一个
    streetbarrier,yaw=0 读 `1.215×0.372`、yaw=80.07° 读 `0.157×1.261`,而资产的
    真实尺寸根本不随朝向变。只有 yaw=0(局部系 == 世界系)那一读是无歧义的。

    读回**必须在 `tick()` 之后** —— 快照在 tick 后才刷新,spawn 完立刻读拿到的是
    全 0 的陈旧值(第一版就是这么读到 `0.576×1.133` 的)。
    """
    pts = world.get_map().get_spawn_points()
    loc = pts[-1].location + carla.Location(z=30.0)  # 空中:道具不下落,置一 tick 即收
    probe = world.try_spawn_actor(bp, carla.Transform(loc, carla.Rotation()))
    if probe is None:
        raise RuntimeError(f"{OCCLUDER_MODELS[mode]} 的尺寸探针 spawn 失败 —— 无法自证资产")
    world.tick()
    bb = probe.bounding_box
    got = (2.0 * bb.extent.x, 2.0 * bb.extent.y, 2.0 * bb.extent.z)
    probe.destroy()
    world.tick()
    if not dims_match(bb, dims):
        raise RuntimeError(
            f"{OCCLUDER_MODELS[mode]} 的实测尺寸 {got[0]:.3f}×{got[1]:.3f}×{got[2]:.3f} 与 "
            f"occlusion.OCCLUDER_DIMS 的声明 {dims[0]}×{dims[1]}×{dims[2]} 对不上 —— "
            "摆位数学全建立在声明那份上,必须停"
        )


def _spawn_occluders(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    ego: carla.Actor,
    cars: list[tuple[float, float, float, float]],
    *,
    mode: str,
    gap: float,
    focal_px: float,
) -> list[carla.Actor]:
    """按 [`occlusion`](occlusion.py) 的纯几何摆静态遮挡物,并**读回实测位姿自证**。

    `cars` = 每辆目标车的 `(前向距离 m, 横向偏移 m, 车宽 m, 车顶世界 z)`,前两项是 ego 局部坐标。

    自证为什么必要:摆位全是一串三角函数,**错了不会报错** —— 只是"效果比预期小",
    而那与"模型对遮挡鲁棒"在数据上长得一模一样。所以这里把三件事当着 CARLA 的面核一遍:
    ① 实 spawn 出来的资产尺寸与 `occlusion.OCCLUDER_DIMS` 那份**声明**一致;
    ② 摆完读回的位姿与声明一致;③ 每块的**净空** > 0
    (墙伸进车道的话 ego 会撞上去,而**撞了照样采得完**,只是轨迹不再与 A 侧配对)。
    """
    bp = bp_lib.find(OCCLUDER_MODELS[mode])
    unit_len, unit_thick, unit_h = OCCLUDER_DIMS[mode]
    _measure_asset(world, bp, (unit_len, unit_thick, unit_h), mode=mode)
    ego_t = ego.get_transform()
    cam_z = ego_t.location.z + SENSOR_OFFSET.location.z
    ego_half_w = ego.bounding_box.extent.y

    print(
        f"[occluders] {mode} — {OCCLUDER_MODELS[mode]} "
        f"{unit_len:.3f}(长) × {unit_thick:.3f}(厚) × {unit_h:.3f}(高) m,相机 z={cam_z:.2f}"
    )
    placed: list[carla.Actor] = []
    worst_clear = float("inf")
    for car_d, car_lat, car_w, car_top_z in cars:
        # ★ 深度(沿光轴),不是斜距 —— 成像吃的是深度。混了它 20 m 处整框会算成
        #   49.7 px 而真值 58.6 px(见 occlusion 模块头注那张表)。
        depth_car = car_d - CAM_FWD
        depth_occl = occluder_depth(car_d, car_lat, gap=gap)
        need = required_span(car_w, depth_car, depth_occl)
        slots = sight_line_occluders(
            car_d,
            car_lat,
            gap=gap,
            unit_len=unit_len,
            unit_thick=unit_thick,
            cover=OCCLUDER_COVER,
            needed_width=need,
        )
        # ★ 先 spawn 完、**tick 一次、再读回** —— 快照在 tick 后才刷新:spawn 完立刻读,
        #   `get_transform()` 全 0、`bounding_box` 也是陈旧值(实测读回 0.576×1.133×1.069,
        #   真值 1.215×0.372×1.069)。这正是红线「任何'实挂位姿 vs 规格'的判据都必须先 tick」
        #   那一条,第一版就栽在这里 —— 而**是自证把它拦下的**,不是肉眼。
        pending: list[tuple[carla.Actor, carla.Location, float]] = []
        for s in slots:
            w = ego_t.transform(carla.Location(x=s.x, y=s.y, z=0.0))
            gz = ground_z_at(world, w.x, w.y, ego_t.location.z)
            want = carla.Location(x=w.x, y=w.y, z=gz)
            yaw = ego_t.rotation.yaw + s.yaw_deg
            v = world.try_spawn_actor(bp, carla.Transform(want, carla.Rotation(yaw=yaw)))
            if v is None:
                print(f"  [warn] 遮挡物 spawn 失败 @ ({w.x:.1f}, {w.y:.1f}) —— 该车遮挡不完整")
                continue
            pending.append((v, want, s.min_y))
        world.tick()
        tops: list[float] = []
        for v, want, min_y in pending:
            got = v.get_transform()
            drift = max(
                abs(got.location.x - want.x), abs(got.location.y - want.y), abs(got.location.z - want.z)
            )
            if drift > 0.05:  # 读了就核:摆放与实际分叉时,"声明"那栏的数全是假的
                print(f"  [warn] 遮挡物落点偏移 {drift:.3f} m(声明 {want} vs 实测 {got.location})")
            placed.append(v)
            tops.append(_bbox_top_z(v))  # 只取 z:顶高与朝向无关,yaw 转过的读数里只有它是可信的
            worst_clear = min(worst_clear, min_y - ego_half_w)
        if not tops:
            continue
        pr = project(
            focal_px=focal_px,
            cam_z=cam_z,
            occl_top_z=max(tops),
            car_top_z=car_top_z,
            depth_car=depth_car,
            depth_occl=depth_occl,
        )
        print(
            f"  car @{car_d:4.0f} m(横 {car_lat:.1f},顶 {car_top_z - ego_t.location.z:.2f} m)"
            f" ← {len(tops)} 块 @深度 {depth_occl:5.1f} m 顶 {max(tops) - ego_t.location.z:.2f} m"
            f" ⇒ 可见 {pr.visible_frac * 100:5.1f}%（{pr.full_px:5.1f} px 里剩 {pr.visible_px:5.1f} px）"
        )
    print(f"[occluders] 最小净空 {worst_clear:.3f} m（负 = 伸进车道,ego 会撞）")
    return placed


def make_depth_blueprint(bp_lib: carla.BlueprintLibrary) -> carla.ActorBlueprint:
    """深度相机蓝图:**属性逐字取 `CAM_ATTRS`**(与 RGB 那一路同一份常量)。

    ★ 这是「深度图与 `image_2` **像素对齐**」这条不变量的**唯一落点**(`--depth`)。
    挂点(与 RGB 同 `SENSOR_OFFSET`)与这三项(`image_size_x` / `image_size_y` / `fov`)
    只要有一处不一致,两幅画就不再逐像素对应 —— 而"不齐"在下游**只表现为
    「GT 框与画面配不上」**,看不出是这里错的(同 `collect_3dgs.make_kind_blueprints` 的纪律)。

    抽成函数是为了让单测能拿一个假 `bp_lib` 直接问"你到底给它设了哪些属性"。
    """
    bp = bp_lib.find("sensor.camera.depth")
    for ak, av in CAM_ATTRS.items():
        bp.set_attribute(ak, av)
    return bp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=sorted(SCENES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=70)  # 70帧≈56m,距第4车9m刹停
    ap.add_argument("--speed", type=float, default=SPEED, help="ego 定速 m/s(参数扫描用;默认= P1 基线 8.0)")
    ap.add_argument(
        "--no-radar",
        action="store_true",
        help="不挂 5 雷达(默认挂)。雷达走 devkit 形状 `samples/RADAR_*/*.pcd`,与 "
        "`collect_surround_lidar` **同一套挂点/形状** —— P1 因此能补一条「雷达也不受天气/光照?」,"
        "与已确立的「LiDAR 不受天气光照」对称",
    )
    ap.add_argument(
        "--layout",
        default="p1",
        choices=("p1", "close"),
        help="目标布局。`p1`(默认)= P1 归档矩阵那四台路肩车(20/35/50/62 m)—— "
        "**它的 d/lat 是已归档矩阵的复现前提,勿动**;`close` = **多类 + 近距**"
        "(7–24 m,含行人与骑行者),专为融合消融造,见 `CLOSE_TARGETS` 头注",
    )
    ap.add_argument(
        "--occluders",
        choices=("none", "partial", "full"),
        default="none",
        help="静态遮挡档 —— P1 的**第二个单变量**(默认 none = 与加这个开关前逐位同行为)。"
        "partial = 每辆路肩车前摆 1.07 m 高的护栏墙(低于相机 ⇒ 只截掉车身下一截);"
        "full = 1.86 m 高(高于相机 ⇒ 恒全遮,留作阳性对照:它不掉点就说明摆位代码坏了)。"
        "**不是「密集车流」** —— 道具进不了 GT,产出只有「静态遮挡」这一种解释(见模块头注)",
    )
    ap.add_argument(
        "--occluder-cars",
        choices=("all", "nearest"),
        default="all",
        help="给哪几辆车摆遮挡物。**`all` 会饱和** —— 墙都摆在同一侧,所以每堵墙不只挡它自己"
        "那台,还把更远的几台全挡了:70 帧 × 4 台的几何普查里 partial 档只剩 28 对没被挡,"
        "检出数(26)几乎正好等于它 ⇒ 测出来是「有遮挡 vs 没遮挡」而不是「部分遮挡有多伤」。"
        "`nearest` 只挡最近那台(唯有它落在 21–24 px 断崖上),其余三台**留作同帧内的对照**",
    )
    ap.add_argument(
        "--occluder-gap",
        type=float,
        default=OCCLUDER_GAP,
        help=f"遮挡物沿**视线**到目标车的距离 m(默认 {OCCLUDER_GAP})。越大越不挡:"
        "墙离车越远就离相机越近,能截到车上的俯角段反而更窄",
    )
    ap.add_argument(
        "--depth",
        action="store_true",
        help="多挂一路深度相机(同挂点同 fov → `training/depth/{id}.npy`,米)。"
        "★ **这是「真值深度 + label_2 同源」的唯一来源** —— 既有 root 里 `kitti_ab_*` 有框没深度、"
        "`3dgs_sync/capture` 有深度没框。默认关 ⇒ 关着时产物与归档**逐字节一致**",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()
    scene = SCENES[args.scene]

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)

    # 清场残留 actor:校准脚本曾只毁相机不毁 ego,残留车阻塞 pt0 致
    # spawn_ego fallback 到反向 spawn 点(yaw≈-180,朝 -x)→ A/B 起点失配
    # **遮挡物也要清**:上一次跑崩在中途的话它们会留着,**并且看不见** ——
    # 后一轮的 A 侧会带着上一轮 B 侧的墙采完,而 A/B 的差就凭空小了一截。
    for a in world.get_actors():
        if is_cleanup_target(a.type_id):
            a.destroy()
    for _ in range(3):
        world.tick()

    world.set_weather(carla.WeatherParameters(**merged_weather(scene)))
    print(f"[scene] {scene.name} — {sorted(scene.weather)}")

    ego = spawn_ego(world)
    ego.set_autopilot(False)
    # 强制锚定 pts[0](yaw=0 朝世界 +x):出生即复现,不依赖 spawn_ego 的
    # 空位探测;锚定后校验,失败立即报错而非静默走错路线
    spawn_pts = world.get_map().get_spawn_points()
    ego.set_transform(carla.Transform(spawn_pts[0].location, carla.Rotation(yaw=0.0)))
    world.tick()
    here = ego.get_location()
    if here.distance(spawn_pts[0].location) > 1.0:
        raise RuntimeError(f"ego 未能锚定 pts[0]: 落在 {here}")
    print(
        f"[ego] 锚定 pts[0] @ ({here.x:.1f}, {here.y:.1f}) 偏航 {ego.get_transform().rotation.yaw:.1f}° (朝世界 +x)"
    )
    ego.apply_control(carla.VehicleControl(brake=1.0))  # 站定
    world.tick()

    # 内参:遮挡物档要用 `fx` 把俯角换成像素 ⇒ 提到这里(原本在传感器之后才建)
    k = CameraIntrinsics(
        width=int(CAM_ATTRS["image_size_x"]),
        height=int(CAM_ATTRS["image_size_y"]),
        fov_h_deg=float(CAM_ATTRS["fov"]),
    )

    # 静置目标车:ego 前方固定路肩位置
    bp_lib = world.get_blueprint_library()
    start = ego.get_location()
    fwd = ego.get_transform().get_forward_vector()
    right = ego.get_transform().get_right_vector()
    # (车, 前向距离 d, 横向 lat) —— d/lat 是**ego 局部坐标**,遮挡物档直接吃这两个
    placed: list[tuple[carla.Vehicle, float, float]] = []
    if args.layout == "close":
        # 多类 + 近距档:行人是 `walker.*`,**不给 AI controller 就是站着不动**;
        # 自行车是 `vehicle.*`(静置)。两者的 KITTI 类由 `gt.core.classify_kitti` 归一。
        target_list = [(m, d, lat) for m, d, lat in CLOSE_TARGETS]
    else:
        target_list = [
            (NPC_MODELS[i % len(NPC_MODELS)], d, lat)
            for i, (d, lat) in enumerate(zip(STATIC_OFFSETS, STATIC_LATS, strict=True))
        ]
    for model, d, lat in target_list:
        p = start + fwd * d + right * lat
        pos = carla.Location(x=p.x, y=p.y, z=start.z)
        v = world.try_spawn_actor(
            bp_lib.find(model),
            carla.Transform(pos, carla.Rotation(yaw=ego.get_transform().rotation.yaw)),
        )
        if v is None:
            print(f"  [warn] {model} spawn 失败 @ d={d} lat={lat}")
            continue
        if model.startswith("vehicle"):
            cast(carla.Vehicle, v).apply_control(carla.VehicleControl(brake=1.0))  # 静置
        # ⚠️ **不给 walker 挂 AI controller**(2026-10-02 实测证伪了那条假设):
        #   挂上并 `start()` 之后,`controller.ai.walker` **把它带走了** ——
        #   实测目标从 ego 局部 `(16, 4)` 跑到 `(65, -24)`(相机系 z≈60 m),
        #   而回波**仍然是 0**(18/19 帧)。
        #   (本仓 `carla_common.spawn_npcs` 里同样写法,但那里只是布景,
        #    没有人去核对 walker 的实际落点 —— 这里一核就露了。)
        placed.append((cast(carla.Vehicle, v), d, lat))
    world.tick()
    print(
        f"[statics] {len(placed)}/{len(target_list)} 目标({args.layout}) @ "
        f"{[(round((s.get_location() - start).x), round((s.get_location() - start).y)) for s, _d, _l in placed]}"
    )

    # ---- 静态遮挡(P1-6b;默认 none ⇒ 下面整块不执行,行为与加开关前逐位相同)----
    occluders: list[carla.Actor] = []
    if args.occluders != "none":
        cars = [(d, lat, 2.0 * veh.bounding_box.extent.y, _bbox_top_z(veh)) for veh, d, lat in placed]
        if args.occluder_cars == "nearest":
            cars = cars[:1]  # 最近那台 = 唯一落在 21–24 px 断崖上的(见 occlusion 头注)
        occluders = _spawn_occluders(
            world, bp_lib, ego, cars, mode=args.occluders, gap=args.occluder_gap, focal_px=k.fx
        )
        world.tick()

    # 传感器同 collect_drive
    # ⚠️ 循环变量**不能叫 `k`** —— `k` 是内参(上面已建),`for k, _ in CAM_ATTRS.items()`
    # 会把它悄悄换成字符串,报错点是几十行外的 `k.p2()`(Pyright 能看见,运行期才炸)
    cam_bp = bp_lib.find("sensor.camera.rgb")
    for ak, av in CAM_ATTRS.items():
        cam_bp.set_attribute(ak, av)
    lid_bp = bp_lib.find("sensor.lidar.ray_cast_semantic")
    for ak, av in LIDAR_ATTRS.items():
        lid_bp.set_attribute(ak, av)
    camera = cast(carla.Sensor, world.spawn_actor(cam_bp, SENSOR_OFFSET, attach_to=ego))
    lidar = cast(carla.Sensor, world.spawn_actor(lid_bp, SENSOR_OFFSET, attach_to=ego))
    img_q: queue.Queue = queue.Queue()
    lid_q: queue.Queue = queue.Queue()
    camera.listen(img_q.put)
    lidar.listen(lid_q.put)

    # 深度(**默认关**):★ 挂点与 fov 取**同一份常量**,不许另写一遍 ——
    # 只要挂点/fov 差一处,深度图与 `image_2` 就不再像素对齐,而那在下游**只表现为
    # "GT 框与画面配不上"**,看不出是这里错的(同 `collect_3dgs.make_kind_blueprints` 的纪律)。
    dep: carla.Sensor | None = None
    dep_q: queue.Queue = queue.Queue()
    if args.depth:
        dep = cast(
            carla.Sensor, world.spawn_actor(make_depth_blueprint(bp_lib), SENSOR_OFFSET, attach_to=ego)
        )
        dep.listen(dep_q.put)

    # ---- 5 雷达:与 `collect_surround_lidar` **同一套挂点/形状**(由 NUS_RADAR_* 导出)----
    # 2026-09-30 加:原先 P1 只有相机 + 语义 LiDAR。加雷达是为了补一条与
    # 「LiDAR 不受天气光照」对称的结论 —— 否则 P1 的"非相机模态兜底"只证了一半。
    radars: dict[str, carla.Sensor] = {}
    radar_qs: dict[str, queue.Queue] = {ch: queue.Queue() for ch in NUS_RADAR_CHANNELS}
    if not args.no_radar:
        for ch in NUS_RADAR_CHANNELS:
            r_bp = bp_lib.find("sensor.other.radar")
            for ak, av in RADAR_ATTRS.items():
                r_bp.set_attribute(ak, av)
            s = cast(
                carla.Sensor,
                world.spawn_actor(
                    r_bp,
                    carla.Transform(
                        carla.Location(*NUS_RADAR_MOUNTS_CARLA[ch]),
                        carla.Rotation(pitch=0.0, yaw=RADAR_YAW_OFFSET[ch], roll=0.0),
                    ),
                    attach_to=ego,
                ),
            )
            s.listen(radar_qs[ch].put)
            radars[ch] = s
    print(f"[sensors] 1 相机{' + 深度' if args.depth else ''} + LiDAR(semantic) + {len(radars)} 雷达")

    # 预热:雷达 `sensor_tick` 相位偶发空(实测 ~7%)⇒ 以"每通道至少出过一帧"为就绪判据。
    # **必须在 ego 站定期间做** —— 此处的 tick 不产生 A/B 帧,ego 仍被 brake 钉住。
    warmed: dict[str, bool] = dict.fromkeys(NUS_RADAR_CHANNELS, False)
    for _ in range(10):
        world.tick()
        for ch, q in radar_qs.items():
            if not args.no_radar and _latest(q) is not None:
                warmed[ch] = True
        if args.no_radar or all(warmed.values()):
            break
    if not args.no_radar and not all(warmed.values()):
        print(f"  [warn] 预热 10 tick 仍有雷达通道没出帧:{[k for k, v in warmed.items() if not v]}")

    out = project_path(args.out or f"outputs/kitti_ab_{scene.name}")

    # 解除"站定"制动:VehicleControl 一旦设置就每步生效,brake=1.0 残留会让
    # set_target_velocity 打折扣(2026-09-09 实测:命令 8 → 实际 6.59 m/s = 0.82×;
    # P1 四个 A/B 数据集实测均为 6.60 m/s,"定速"名不副实且 TTC 归一化偏 18%)
    ego.apply_control(carla.VehicleControl())
    fwd_v = carla.Vector3D(x=fwd.x * args.speed, y=fwd.y * args.speed, z=0.0)
    try:
        for i in range(args.frames):
            # 定速:每 tick 强设速度(车辆控制速度环不稳,直接 velocity)
            ego.set_target_velocity(fwd_v)
            world.tick()
            image: carla.Image = img_q.get(timeout=10)
            pts: carla.LidarMeasurement = lid_q.get(timeout=10)
            # ★ **每 tick 每队列都要抽干**(红线第四次现形那一条)。深度那路同样:
            # 漏一次 = 那一路整体滞后 N 帧,而"帧号一张张对得上、图一张张出得来",
            # 症状只是"深度与画面对不上"。
            dep_img: carla.Image | None = dep_q.get(timeout=10) if dep is not None else None
            # ★ **同 tick 自证**:两路的 `frame` 号必须相等。`assert_synced` 管的是
            # "几路彼此同不同步";红线的代价是**恒定滞后**,而它不会自己报错。
            sync_note = assert_synced([("RGB", image), ("深度", dep_img)])

            cam_t, lid_t = camera.get_transform(), lidar.get_transform()
            calib_out = KittiCalibOut(
                p2=k.p2(),
                tr_velo_to_cam=tr_velo_to_cam(
                    loc(lid_t), rad(lid_t.rotation), loc(cam_t), rad(cam_t.rotation)
                ),
            )
            labels: list[str] = []
            for a in world.get_actors():
                if not (a.type_id.startswith("vehicle") or a.type_id.startswith("walker")):
                    continue
                bb = a.bounding_box
                box = ActorBox(
                    type_id=a.type_id,
                    extent=(bb.extent.x, bb.extent.y, bb.extent.z),
                    location=(bb.location.x, bb.location.y, bb.location.z),
                    rotation=rad(bb.rotation),
                    actor_location=loc(a.get_transform()),
                    actor_rotation=rad(a.get_transform().rotation),
                )
                line = box_to_gt_line(box, loc(cam_t), rad(cam_t.rotation), k, max_distance=65.0)
                if line:
                    labels.append(line)

            # 5 雷达 → devkit 形状 `samples/RADAR_*/*.pcd`(与 collect_surround_lidar 同落法)。
            # **非阻塞 drain**:相位空 tick 是常态(实测 ~7%),阻塞 get 会死等;
            # 空通道写**空 pcd**(devkit 空编码),不阻断采集。
            for ch in NUS_RADAR_CHANNELS:
                if args.no_radar:
                    break
                frame = _latest(radar_qs[ch])
                if frame is None:
                    nus18 = np.empty((0, 18), dtype=np.float32)
                else:
                    dets = np.frombuffer(frame.raw_data, dtype=np.float32).reshape(-1, 4)
                    nus18 = detections_to_nus18(dets, sensor_id=NUS_RADAR_CHANNELS.index(ch))
                    nus18 = nus18[mask_radar_points(nus18)]
                rel = out / "samples" / ch / f"{i:06d}.pcd"
                rel.parent.mkdir(parents=True, exist_ok=True)
                rel.write_bytes(nus18_to_pcd(nus18))

            tmp = out / f".tmp_{i}.png"
            image.save_to_disk(str(tmp))
            png = tmp.read_bytes()
            tmp.unlink()
            if dep_img is not None:
                # 解码走 `calib.depth_codec.decode_depth`(**唯一口径**,与 `collect_3dgs` 同)
                dpath = out / "training" / "depth" / f"{i:06d}.npy"
                dpath.parent.mkdir(parents=True, exist_ok=True)
                np.save(dpath, decode_depth(dep_img.raw_data, dep_img.height, dep_img.width))
            raw = np.frombuffer(pts.raw_data, dtype=np.float32)
            velo = semantic_to_velodyne_bin(raw.reshape(-1, 6), seed=args.frames * 100 + i)
            write_frame(
                out,
                str(i),
                image_png=png,
                velodyne=velo,
                calib=calib_out,
                labels=labels,
                pose=ego_pose_matrix(ego.get_transform()),
            )
            if (i + 1) % 25 == 0 or i == args.frames - 1:
                print(
                    f"[frame {i + 1}/{args.frames}] ego x={ego.get_location().x:8.1f} | {len(labels)} GT"
                    + (f" | {sync_note}" if sync_note else "")
                )
    finally:
        # **雷达必须一起收**:留着同名 actor 会在下一次采集的"清场"里被漏掉(它不在
        # vehicle/walker/controller 三类里),阻塞后续 spawn —— 与 `collect_surround_lidar`
        # 首跑漏收 5 个 radar 是同一条。**遮挡物同理**(`static.prop.*` 也不在那三类里)。
        for s in (camera, lidar, dep, *radars.values()):
            if s is None:
                continue  # `dep` 在未开 `--depth` 时是 None
            s.stop()
            s.destroy()
        for a in world.get_actors():
            if is_cleanup_target(a.type_id):
                a.destroy()
    # 末行写清**这一份是哪一档**:A/B 两个 root 的区别只有这一个变量,
    # 而 `eval_2d_ab` 的两个位置参数谁是谁,靠的就是各自 root 名 + 这行
    occ = f"{args.occluders}×{len(occluders)}" if args.occluders != "none" else "无遮挡"
    print(f"[done] KITTI root: {out.resolve()} ({args.frames} frames, {occ})")


if __name__ == "__main__":
    main()

"""CARLA 采集公共件(base env,依赖 pycarla):位姿换算 / NPC 摆放 / 传感器参数。

灯态 GT 的 carla→纯值归一(traffic_light_frame)与 overlay 绘制
(draw_traffic_lights)也放这里:采集器与实时可视化共用同一条实现,
保证"目检所见 = 落盘口径"。
"""

from __future__ import annotations

import queue
from typing import Protocol, cast

import carla
import numpy as np
from PIL import Image, ImageDraw

from autodrivedata.calib.camera_rig import NUS_CAMERA_RIG
from autodrivedata.calib.core import CameraIntrinsics, world_to_img
from autodrivedata.gt.props import is_prop
from autodrivedata.gt.traffic_light import (
    TrafficLightFrame,
    TrafficLightState,
    in_front,
    normalize_state,
)
from autodrivedata.utils import fonts

#: 世界系 3 元组(位置/尺寸/角度)。语义见各函数。
Vec3 = tuple[float, float, float]


class _WalkerCtrl(Protocol):
    """pycarla 桩缺 WalkerAIController 方法标注的最小协议。"""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def set_max_speed(self, speed: float) -> None: ...
    def go_to_location(self, destination: carla.Location) -> None: ...


# 相机对齐 KITTI 口径(1242×375);LiDAR 64 线(与 KITTI velodyne 一致)
CAM_ATTRS = {"image_size_x": "1242", "image_size_y": "375", "fov": "90"}
LIDAR_ATTRS = {
    "channels": "64",
    "range": "70",
    # 1.3M pps = 真实 HDL-64E 量级(实测每帧 63k 点、360° 全覆盖);
    # 200k pps 时车只有 13~117 点,PointPillars 体素特征不足(M1a-7 实测教训)
    "points_per_second": "1300000",
    "rotation_frequency": "10",
    "upper_fov": "10.0",
    "lower_fov": "-30.0",
}
SENSOR_OFFSET = carla.Transform(carla.Location(1.2, 0.0, 1.65))  # 相对 ego 的车顶前装
# 官方 nuScenes 相机挂点(x 前 / y 右 / z 上,CARLA 系)——由
# `autodrivedata/camera_rig.NUS_CAMERA_RIG` 导出(y 已翻号,与官方 nus 系 y 左镜像)。
# 真值在 camera_rig,**本表只是"只关心平移的调用方"的别名**;改官方标定只改那一处。
SENSOR_MOUNTS: dict[str, tuple[float, float, float]] = {
    name: mount for name, (mount, _rot) in NUS_CAMERA_RIG.items()
}


def rad(rot: carla.Rotation) -> tuple[float, float, float]:
    """carla.Rotation(度)→ (pitch, yaw, roll) 弧度。"""
    return tuple(np.radians(a) for a in (rot.pitch, rot.yaw, rot.roll))


# --------------------------------------------------------------------------- 传感器队列同步
#: `assert_synced` 需要相机帧做的**唯一**一件事:自报帧号(写成协议是为了让单测能给假帧)。
class _Framed(Protocol):
    frame: int


def drain(q: queue.Queue) -> int:
    """丢弃队列里**已积压**的帧,返回丢掉几帧。

    同步模式下每 tick 每队列恰一帧,所以"队列里有多少帧"= "相机挂上之后又 tick 了几次"。
    道具通道在开跑前要 tick 好几次(量尺寸 2 次 + 摆位读回 1 次)⇒ **不 drain 的话主循环
    读到的是"道具还没摆好"的那一帧**,而症状是"前几帧 GT 有框、图里没东西" ——
    被眼熟地误读成"模型没检出来"。丢掉的帧数打出来,别静默。
    """
    n = 0
    while True:
        try:
            q.get_nowait()
            n += 1
        except queue.Empty:
            return n


def assert_synced(images: list[tuple[str, _Framed | None]]) -> str:
    """★ **同 tick 自证**:几路传感器这一轮的 `frame` 号必须相等。返回一行人读的摘要。

    这是红线那条「**两路本应逐像素相同的东西比一比**」在这里的形态 —— 比的是帧号。
    只在**全都拿到**时判:某一路本轮没有(没开那个开关)就不管它。
    `frame` 属性是 CARLA 的仿真帧号,**帧号对不上**不报错的症状是"图看着正常、框整体偏",
    而那与"标定错了"长得一样。

    ⚠️ 它抓的是"几路彼此**不同步**",**抓不到"几路一起恒定滞后"** —— 后者的判据是
    "跨方法一致"或"固定掩膜跨帧扫 argmax",见 `collect_static_gt.settle` 头注。

    ★ **为什么从 `collect_static_gt` 搬到这里**(2026-10-07):`collect_surround` 也要用它,
    而那会形成 `collect_surround → collect_static_gt → collect_surround` 的**循环导入**
    (后者要 `tag_from_semantic_image`)。这两个函数**没有领域语义**(只管队列与帧号)
    ⇒ 它们的家在 `carla_common`,不在任何一个采集器里。
    """
    got = [(n, int(im.frame)) for n, im in images if im is not None]
    if len(got) < 2:
        return ""
    frames = {f for _, f in got}
    if len(frames) != 1:
        raise SystemExit(
            "相机不同帧 —— " + ", ".join(f"{n}={f}" for n, f in got) + "。"
            "「每 tick 每队列恰一帧」这条前提没成立,继续采会得到**恒定滞后**的序列"
            "(实测过 1 帧),而帧号一张张对得上、图一张张出得来。"
        )
    return f"frame={got[0][1]}(" + "/".join(n for n, _ in got) + ")"


def clear_generated_actors(world: carla.World) -> int:
    """清掉本仓采集器自己会生成的 actor(车 / 行人 / 控制器 + **全部** `static.prop.*`)。

    返回清掉几个。**清场与收尾共用这一个谓词** —— 两处不同源就是"清一半"这类不可见失败,
    本项目已经踩满三次:残留车阻塞 pt0(致 fallback 反向出生点)、残留雷达阻塞后续 spawn、
    残留**遮挡物/道具**让下一轮 A 侧带着上一轮 B 侧的东西采完 —— **最后这条最阴:数据照出**,
    只是 A/B 的差凭空小一截。

    ⚠️ `static.prop.*` **不在** `vehicle/walker/controller` 三类里,所以要单独带上
    `props.is_prop`(**覆盖全部 `static.prop.*`,不写死我们自己摆的那几个**)。
    谓词**宽于 GT 过滤器**:controller 与道具都不进 GT,混用一个谓词会把道具写进 `label_2`。
    ⚠️ **先 `world.tick()` 再扫** —— `get_actors()` 是陈旧快照,不 tick 会**静默漏清**
    (同 `sync_mode` 那条:"spawn 完立刻读"拿到的是不含新 actor 的旧快照)。
    ⚠️ **只扫一遍**(本函数就是那一遍)。先显式 destroy 再让本函数重扫,会给已经不存在的
    actor 各报一条 `failed to destroy ... not found` —— 数据没错,但"收尾一堆红字"正是
    `tools/carla_server.sh` 头注记的「崩溃掩盖成功」同款误导。
    """
    world.tick()
    n = 0
    for a in world.get_actors():
        if a.type_id.startswith(("vehicle", "walker", "controller")) or is_prop(a.type_id):
            a.destroy()
            n += 1
    return n


def ground_z_at(world: carla.World, x: float, y: float, fallback: float) -> float:
    """`(x, y)` 处的**路面** z;该点不投影到任何车道(路肩外)时退回 `fallback`。

    别拿 ego 的 z 顶替:障碍物能不能挡/道具摆得正不正,对它**高度是线性敏感**的 ——
    路面有起伏时浮空 0.2 m 就等于少挡 0.2 m,而这只在数偏了以后才看得出来
    (与 §P-M.7「声明 = 渲染」同一条纪律)。

    原为 `collect_ab_route` 与 `probe_static_prop_gt` 各一份的局部实现,2026-10-01 上移共享
    —— 三份拷贝里任何一份漂了,"摆位"与"自证"就会用不同的地面,而差 20 cm 谁都看不出来。
    """
    try:
        wp = world.get_map().get_waypoint(carla.Location(x=x, y=y, z=fallback + 2.0))
    except RuntimeError:  # 该点不投影到任何车道(路肩外)
        return fallback
    return float(wp.transform.location.z)


def measure_actor_size_yaw0(world: carla.World, bp: carla.ActorBlueprint) -> tuple[Vec3, Vec3, Vec3]:
    """把蓝图摆在**空中 yaw=0** 处读回 `(全长宽高, 盒偏移, 盒旋转)` —— 唯一无歧义的那一读。

    ## 为什么不能读实摆的那个

    `Actor.bounding_box` 对**转过**的 actor 给出 `(extent, rotation)` 自相矛盾的读数。
    2026-10-01 探针实测(锥桶,真值 0.3441×0.3441×0.5858 m):

    | actor yaw | 直接读回的 (x, y) 全长 | |
    |---|---|---|
    | 0° | 0.3431 × 0.3450 | ✓ |
    | 30° | **0.1246 × 0.4704** | ✗ |
    | 60° | **0.1272 × 0.4697** | ✗ |
    | 90° | 0.3450 × 0.3431 | ✓ |

    不是两轴对调 —— 边长 s 的方锥转 θ 后读回 ≈ `(s·|cosθ−sinθ|, s·(cosθ+sinθ))`,
    盒被**剪切**。后果拿渲染轮廓量:3.4% 的物体像素落在投影框**外面**,IoU 0.897 → 0.708。
    **0° 与 90° 都读对** ⇒ 只拿一个 yaw≈0 的样本验一次会得"没问题"(本仓"yaw≈0 的
    相机看着正常"的同款坑)。

    ## 两个必须

    - **必须摆在空中**:落地会与既有几何碰撞,`try_spawn_actor` 直接返回 None。
    - **必须 `tick()` 之后再读**:快照 tick 后才刷新,spawn 完立刻读拿到的是全 0 的陈旧值
      (第一版就是这么读到 `0.576×1.133` 的)。本函数 spawn 完自己 tick 一次。
    """
    pts = world.get_map().get_spawn_points()
    # ⚠️ `Vector3D` 运算结果**不能直接进 `carla.Transform`**(红线),必须显式包一层
    at = carla.Location(pts[-1].location + carla.Location(z=30.0))
    probe = world.try_spawn_actor(bp, carla.Transform(at, carla.Rotation()))
    if probe is None:
        raise RuntimeError(f"{bp.id} 的 yaw=0 尺寸探针 spawn 失败 —— 无法自证资产尺寸")
    world.tick()
    bb = probe.bounding_box
    dims: Vec3 = (2.0 * bb.extent.x, 2.0 * bb.extent.y, 2.0 * bb.extent.z)
    offs: Vec3 = (bb.location.x, bb.location.y, bb.location.z)
    rot: Vec3 = (bb.rotation.pitch, bb.rotation.yaw, bb.rotation.roll)
    probe.destroy()
    world.tick()
    return dims, offs, rot


def loc(t: carla.Transform) -> tuple[float, float, float]:
    return (t.location.x, t.location.y, t.location.z)


def spawn_ego(world: carla.World) -> carla.Vehicle:
    """出生点逐个尝试(碰撞则换下一个),spawn 后 tick 同步位姿。"""
    bp = world.get_blueprint_library().find("vehicle.audi.a2")
    ego: carla.Actor | None = None
    for pt in world.get_map().get_spawn_points():
        ego = world.try_spawn_actor(bp, pt)
        if ego is not None:
            break
    if ego is None:
        raise RuntimeError("所有出生点均 spawn 失败(碰撞)")
    # 同步模式红线:spawn 后必须 tick,actor 位姿才同步到客户端
    # (实测:不 tick 则 get_transform 返回恒等变换 (0,0,0))
    world.tick()
    return cast(carla.Vehicle, ego)


def spawn_ego_at(world: carla.World, index: int) -> carla.Vehicle:
    """用**指定的** spawn point index 生成 ego(碰撞即报错,不 fallback)。

    与 `spawn_ego` 的分工:那个"逐个试第一个空位",适合单次采集;这个要求**可复现的
    指定起点**,供多段采集(每段一条独立路线,段间起点必须互不相同且可复现)。
    **失败不换点**是刻意的——静默 fallback 会让"我要 88 号点"变成"随便哪条街",
    分段留出集就失去意义(同 `collect_ab_route` 的起点校验纪律)。

    原为 `autodrivedata/sim/collect_traj.py` 局部实现,2026-09-23 上移共享(§P-M.11 下游)。
    """
    pts = world.get_map().get_spawn_points()
    if not 0 <= index < len(pts):
        raise ValueError(f"spawn index {index} 越界(本图共 {len(pts)} 个 spawn point)")
    bp = world.get_blueprint_library().find("vehicle.audi.a2")
    ego = world.try_spawn_actor(bp, pts[index])
    if ego is None:
        raise RuntimeError(f"spawn point {index} 生成失败(碰撞?)")
    # 同步模式红线:spawn 后必须 tick,actor 位姿才同步到客户端(同 spawn_ego)
    world.tick()
    return cast(carla.Vehicle, ego)


def spawn_npcs(world: carla.World, ego_t: carla.Transform) -> None:
    """ego 前方摆 NPC:同向车 ×2、对向车 ×1、行人 ×2、骑行者 ×1(碰撞失败仅告警)。"""
    fwd = ego_t.get_forward_vector()
    right = ego_t.get_right_vector()
    ego_yaw = ego_t.rotation.yaw

    def place(d: float, off: float, yaw: float, z_off: float = 0.0) -> carla.Transform:
        v = ego_t.location + fwd * d + right * off
        pos = carla.Location(x=v.x, y=v.y, z=v.z + z_off)
        return carla.Transform(pos, carla.Rotation(yaw=yaw, pitch=0.0, roll=0.0))

    specs = [
        ("vehicle.tesla.model3", place(12.0, 0.0, ego_yaw)),  # 同车道前车
        ("vehicle.audi.a2", place(22.0, 2.2, ego_yaw)),  # 右邻车道
        ("vehicle.ford.mustang", place(30.0, -3.2, ego_yaw + 180)),  # 对向车
        ("walker.pedestrian.0001", place(8.0, 3.2, ego_yaw)),  # 右侧行人
        ("walker.pedestrian.0002", place(14.0, -3.2, ego_yaw + 90)),  # 左侧行人(面向车道)
        ("vehicle.gazelle.omafiets", place(18.0, 3.6, ego_yaw)),  # 右侧骑行者
    ]
    bp_lib = world.get_blueprint_library()
    for type_id, tf in specs:
        bp = bp_lib.find(type_id)
        actor = world.try_spawn_actor(bp, tf)
        if actor is None:
            print(f"  [warn] NPC spawn 失败(碰撞): {type_id}")
            continue
        if type_id.startswith("walker"):
            ctrl = world.spawn_actor(bp_lib.find("controller.ai.walker"), carla.Transform(), actor)
            cast(_WalkerCtrl, ctrl).start()  # 站立不动;动态场景再给行走指令
        print(f"  [npc] {type_id} @ {loc(tf)}")


#: `load_world` 期间的客户端超时(秒)。**与平时的 60 s 不是一回事**:切大图是一次
#: 同步重活,60 s 会让客户端**单方面放弃等待**(`RuntimeError: time-out of 60000ms
#: while waiting for the simulator`),而**世界其实加载成功了** —— 下一次 `get_world()`
#: 拿到的是新图,只是那次调用白报了错。Town13(12478 个 spawn point)实测 >60 s。
LOAD_WORLD_TIMEOUT_S = 300.0


class ClientLike(Protocol):
    """`carla.Client` 的**最小面**(duck-typed):切图只用这两个方法。

    用 Protocol 而不是直接标 `carla.Client`,理由同 `route.WaypointLike` ——
    让"切图期间超时调高、结束恢复"这条**能在假 client 上跑单测**。
    这一条正是本模块当初没写测试而漏掉的那类错(注释写着 ~2min、超时给 60 s)。

    ⚠️ **形参名必须与 carla 桩逐字一致**(`second` / `map_name`,不是 `seconds` / `name`)——
    Protocol 是按**签名**判兼容的,名字不同就整类不兼容,而报错信息读起来像类型错、
    不像命名错(同 `route.WaypointLike` 里"可读 property vs 可变属性"那条踩坑记录)。
    `reset_settings` 带上是因为实参可能用它;`map_layers` 不必 —— 它后面的都有默认值,
    实现签名多几个可省参数不影响兼容。
    """

    def set_timeout(self, second: float) -> None: ...
    def load_world(self, map_name: str, reset_settings: bool = True) -> carla.World: ...


def load_world(
    client: ClientLike,
    name: str,
    *,
    load_timeout: float = LOAD_WORLD_TIMEOUT_S,
    after_timeout: float = 60.0,
) -> carla.World:
    """`load_world` + **配套超时**(切图期间调高,切完恢复)。

    为什么要包一层:三个采集器各写过一遍 `set_timeout(60)` → `load_world` → `set_timeout(60)`,
    而 `collect_surround` 那处的注释自己写着「~2min」—— **注释与超时值互相矛盾**,
    说明当时就知道慢、只是没把超时对上。Town13 一跑就现形。
    `after_timeout` 由调用方给(pycarla 的 Client 没有 `get_timeout`,取不回原值,
    所以只能**约定**一个后续值,别假装能还原)。
    """
    client.set_timeout(load_timeout)
    try:
        return client.load_world(name)
    finally:
        # 无论成败都恢复:加载失败后还挂着一个 300 s 的超时,会让**后面每一步**
        # 卡到 5 分钟才报错,把一次明确的失败变成一次"看起来像挂死"的失败。
        client.set_timeout(after_timeout)


def sync_mode(world: carla.World, delta: float = 0.1) -> None:
    """开同步模式(固定 tick),并 tick 一次刷新客户端 actor 快照。

    实测(2026-09-09):连到**已处于同步模式**的服务器时,首个 get_actors()
    返回空(快照只在 tick 后更新)→ 各采集器"清场残留 actor"的循环会静默漏清,
    残留 ego 阻塞 pts[0] 即复现 A/B 起点失配的老坑。此处统一补一次 tick。
    """
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = delta
    world.apply_settings(settings)
    world.tick()


LIGHT_HEAD_Z = 4.5  # 灯头相对 actor 锚点的高度(实测灯箱 z≈4.0-5.2,状态显示在灯头)
TL_COLOR = {
    "Red": (255, 40, 40),
    "Yellow": (255, 210, 0),
    "Green": (40, 255, 80),
    "Off": (120, 120, 120),
    "Unknown": (120, 120, 120),
}


def traffic_light_frame(
    world: carla.World,
    frame_id: str,
    ego_location: tuple[float, float, float],
    ego_yaw_deg: float,
    horizon: float = 120.0,
    phase_plan: tuple[tuple[str, float], ...] = (),
    forward_only: bool = True,
) -> TrafficLightFrame:
    """世界内信号灯 → 纯值 TrafficLightFrame(状态 + 管制车道 + 停车线)。

    - 灯头位置 = actor 锚点 + LIGHT_HEAD_Z(投影/可视口径)
    - affected_lanes = 路口内管制车道、stop_lanes = 停车线所在车道
      → 状态关联到**流向**(一个灯头管多条道),不是只给灯头坐标
    - 视距 horizon 外剔除(距离字段保留,供下游按需再过滤)
    - forward_only:只收 ego 前方半平面的灯(实测不过滤则 79% 是身后灯)
    """
    lights: list[TrafficLightState] = []
    for actor in world.get_actors().filter("traffic.traffic_light"):
        light = cast(carla.TrafficLight, actor)  # pyi 桩:Actor 无 get_state/get_opendrive_id
        t = light.get_transform()
        head = (t.location.x, t.location.y, t.location.z + LIGHT_HEAD_Z)
        if forward_only and not in_front(head, ego_location, ego_yaw_deg):
            continue
        dist = float(np.hypot(head[0] - ego_location[0], head[1] - ego_location[1]))
        if dist > horizon:
            continue
        affected = tuple(sorted({(wp.road_id, wp.lane_id) for wp in light.get_affected_lane_waypoints()}))
        stops = tuple(
            sorted({(wp.road_id, wp.lane_id, round(float(wp.s), 1)) for wp in light.get_stop_waypoints()})
        )
        lights.append(
            TrafficLightState(
                opendrive_id=str(light.get_opendrive_id()),
                state=normalize_state(str(light.get_state())),
                location=head,
                yaw_deg=float(t.rotation.yaw),
                pole_index=int(light.get_pole_index()),
                elapsed_s=float(light.get_elapsed_time()),
                distance_m=dist,
                affected_lanes=affected,
                stop_lanes=stops,
            )
        )
    lights.sort(key=lambda s: s.distance_m)
    return TrafficLightFrame(
        frame_id=frame_id,
        map_name=world.get_map().name,
        ego_location=ego_location,
        ego_yaw_deg=ego_yaw_deg,
        lights=tuple(lights),
        phase_plan=phase_plan,
    )


def draw_traffic_lights(
    img: Image.Image,
    frame: TrafficLightFrame,
    cam_loc: tuple[float, float, float],
    cam_rot: tuple[float, float, float],
    k: CameraIntrinsics,
) -> Image.Image:
    """灯态 overlay:色点 + `#id 状态 距离`(消费纯值帧 → 所见即落盘口径)。"""
    d = ImageDraw.Draw(img)
    for light in frame.lights:
        uv = world_to_img(light.location, cam_loc, cam_rot, k)
        if uv is None:
            continue
        col = TL_COLOR.get(light.state, TL_COLOR["Unknown"])
        x, y = uv
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=col, outline=(0, 0, 0))
        fonts.draw_text(
            d,
            (x + 8, y - 15),
            f"#{light.opendrive_id} {light.state} {light.distance_m:.0f}m",
            size=13,
            fill=col,
        )
    return img

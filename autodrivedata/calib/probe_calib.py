#!/usr/bin/env python3
"""环视相机标定自证探针:把 CARLA 渲染器当**第二把尺子**,数值裁决 (K, 外参)。

数值核心全在 [autodrivedata/calib_probe.py](../autodrivedata/calib_probe.py)(纯值、有单测);
本文件只做 CARLA 编排 + 判据 + 落盘。

## 为什么不能"反投影再重投影"

任何 K 都能重现自己的像素,误差恒为 0。必须有**独立锚**。本探针用七个:

| 锚 | 独立来源 | 量什么 | 判据 |
|---|---|---|---|
| **A0 官方方位角** | nuScenes 官方 `calibrated_sensor`(外部数据) | rig 光轴 → az_nus vs 官方表 | < 0.01° |
| **A1 侧别一致性** | 几何常识(侧相机的挂点与朝向必须同侧) | 挂点 y 符号 × 光轴 y 符号 | 全部同号 |
| **A2 世界方向语义** | 世界几何(左/右/前/后的锥该由**哪几个名字**的相机看见) | 锥落在哪些相机里 | 与预期集合逐项一致 |
| **A3 LiDAR 平面 × 渲染深度** | LiDAR(独立传感器)+ 渲染器 | 射线-平面求交 vs 深度图读数 | median \\|e\\| < 0.1 m |
| **A4 横移锥回归** | 渲染器 | 已知横移 x ⇒ u = (f/z)·x + cx | 残差 < 0.3 px |
| **A5 外参对账** | CARLA 实挂位姿 | 实挂 vs spec 的平移/偏航 | < 1e-3 |
| **A6 主点裁决** | A4 的掩膜直读(不经深度) | cx 读数 vs `(w−1)/2` | < 0.5 px |

A2/A4 靠**实例分割**认锥体。CARLA 的实例分割编码官方文档不写通道序,本探针实测裁决为
`id = G + 256·B`(差分法,4/4 命中且跨 actor 类型,详见 `decode_instance`)。

**A3 的遮挡判据是单侧的**(渲染比预测更近超过容差 ⇒ 被挡住),不是"邻域深度极差":
后者是对称代理,窗口半径一大就把合法样本成片误杀(实测 r=16px 时样本塌 95%)。

**A6 同时钉住了像素约定**(它比 A3 更直接):A4 直读 cx=620.50 = `(w−1)/2`。若渲染器
其实是 center 口径,这里会读到 620.00。A3 的 median\\|e\\| 独立佐证(corner 0.0003 m vs
center 0.023 m,差 ~70×)。

**A2 是本探针唯一能证伪"镜像"的锚**(且不依赖任何方位角表):锥放在世界的**左边**
(世界 −y),看见它的相机名字里必须含 `LEFT`;镜像 rig 会让 `RIGHT` 那两台看见它。
历史 bug 的根因是 `SURROUND_CAMS` 把官方方位角原样抄成正数(漏 `yaw_carla = −az_nus`),
前/后相机因近自逆"看着对",只有 A0/A1/A2 三者之一才拦得住。

**A5 需要 tick 后再读**(传感器 `get_transform()` 在 tick 前是全 0 陈旧值,见 Plan.md 红线)。

## 本探针**测不到**什么(如实声明,不硬给结论)

1. **相机外参的旋转本身无法只从图像证伪**。锥体位置与相机挂点用的是**同一份 spec**,
   两者一致地错也照样"对得上" ⇒ 故 A2 用**世界方向**当锚、A0 用**官方外部表**当锚;
   A3/A4 只证明"渲染器与我们的 K/投影链一致",**不证明 spec 是对的**。
2. **pitch/roll 只有粗检**。锥体列中点对绕光轴/俯仰几乎不敏感;只能靠 A5 对账
   (实挂 vs 规格),测不出"真实俯仰应该是多少"。
3. **径向畸变只能证伪到残差剖面的噪声水平**。CARLA 是理想针孔,预期斜率 ≈ 0;
   非零说明 K 有系统误差,而不是"测出了畸变系数"。
4. **主点的 y 轴可辨识性弱于 x 轴**(A4 只测 x;y 靠 A3 的 δv,而正前方路面深度梯度小)。

## 像素约定:**已裁决为 corner**(结论必须带约定,否则无意义)

`depth_codec.CONVENTION_CENTER` 下索引 i 的中心在连续坐标 i+0.5(与
`grid_sample(align_corners=False)` 一致),图像中心 = `w/2 = 621.0`;`CONVENTION_CORNER`
是"索引即坐标",对应 `(w−1)/2 = 620.5`。二者只差 0.5 px,肉眼绝对看不出来,故**必须数值裁决**:

| 证据(2026-09-22 实测) | corner | center | 裁决 |
|---|---|---|---|
| A3 深度残差 median\\|e\\|(六相机) | **0.0003–0.0009 m** | 0.008–0.027 m | corner 好 ~70× |
| A4 掩膜索引中点(x=0 处) | **620.50 = `(w−1)/2`** | 需为 620.00 | corner |
| A3 δu / δv(应趋 0) | −0.35…+0.29 px | −0.59…+0.23 px | 都在噪声级,单独不足以裁决 |

⇒ **`CameraIntrinsics.cx = (w−1)/2` 是对的;错的是拿 center 去采数组**。本探针一律用
`PIXEL_CONVENTION = CONVENTION_CORNER` 采样,并**同时报两种约定的主点读数**,让读者能自己换算。

`fx = (w/2)/tan(fov/2) = 621.00` 与 `cx = (w−1)/2 = 620.50` 并存**不是矛盾**:前者是
"半视场对应半宽"(w/2),后者是"像素索引口径的中心"。A4 独立测得 `f_est = 621.60 px`
(与 621.00 差 0.1%,5 个锥的回归残差 0.20 px),故 `fx` 公式也站得住。
相关病根:`mapviz.intrinsics_from_k` 只抄 fx、把 cx/cy 丢掉重算,等于用一个约定量出的 cx
去配另一个约定。

## 静态 ego 的两个推论(决定了本探针的结构)

- ego 静止 ⇒ 每 tick 的 LiDAR 点云与渲染图**逐帧相同** ⇒ 平面拟合只做一次
  (`--frames` 只用于把深度残差样本取并集,不是为了"多看几帧")。
- 锥体是**后摆的**:先跑完 A3(需要干净的路面点云),再摆锥。

落盘 `outputs/calib_check/`:`report.json`(全部数字)+ `overlay.png`(6 相机原生像素拼图,
LiDAR 点按深度残差着色)+ stdout 数字表。

用法:
  bash tools/carla_server.sh start
  PYTHONPATH=$PWD python autodrivedata/calib/probe_calib.py --frames 2
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
from typing import Any, cast

import carla
import numpy as np
from PIL import Image, ImageDraw

from autodrivedata.calib import calib_live as cl
from autodrivedata.calib import calib_probe as cp
from autodrivedata.calib.camera_rig import NUS_CAMERA_CALIBS, NUS_CAMERA_RIG
from autodrivedata.calib.core import CameraIntrinsics
from autodrivedata.calib.depth_codec import (
    CONVENTION_CENTER,
    CONVENTION_CORNER,
    convention_shift,
    decode_depth,
)
from autodrivedata.sim.carla_common import (
    CAM_ATTRS,
    LIDAR_ATTRS,
    SENSOR_OFFSET,
    loc,
    rad,
    spawn_ego,
    sync_mode,
)
from autodrivedata.sim.live_common import (
    RIG_NUSCENES,
    compose_rows,
    image_to_pil,
    rig_mount_deviation,
    rig_spec,
)
from autodrivedata.slam.accum import voxel_downsample
from autodrivedata.utils import fonts
from autodrivedata.utils.geometry import (
    carla_rotation_matrix,
    carla_to_nus_global,
    quat_normalize,
    quat_to_matrix,
    wrap_pi,
)
from autodrivedata.utils.paths import project_path

W, H, FOV = int(CAM_ATTRS["image_size_x"]), int(CAM_ATTRS["image_size_y"]), float(CAM_ATTRS["fov"])
K = CameraIntrinsics(width=W, height=H, fov_h_deg=FOV)

# CARLA 渲染器的像素约定,**实测裁决**(2026-09-22,见 §"像素约定"的裁决过程):
# 光栅索引 i 对应的图像坐标就是 i ⇒ **corner**。判据是深度残差的 median|e|:
# corner 0.0003 m vs center 0.0229 m(六相机一致,差 ~70×);A4 的掩膜索引中点独立佐证
# (x=0 处中点 = 620.50 = `CameraIntrinsics.cx`,而 center 口径要求它是 620.00)。
# 换句话说 `CameraIntrinsics.cx = (w−1)/2` 是对的,**用 center 去采数组才是错的**。
PIXEL_CONVENTION = CONVENTION_CORNER

# 世界方向(ego 系,CARLA 轴序:x 前 / y 右 / z 上)。**用向量不用 yaw**:
# "+y 到底是左还是右"是历史踩坑点,故运行期用 `get_right_vector()` 实测复核并落进报告。
WORLD_DIRS: dict[str, tuple[float, float, float]] = {
    "FRONT": (1.0, 0.0, 0.0),
    "BACK": (-1.0, 0.0, 0.0),
    "LEFT": (0.0, -1.0, 0.0),
    "RIGHT": (0.0, 1.0, 0.0),
}
# 每个世界方向的锥**应当**被哪些相机看见(名字里的方位词 = 世界方位;这就是 A2 的判据本身)
EXPECTED_SEERS: dict[str, tuple[str, ...]] = {
    "FRONT": ("CAM_FRONT",),
    "BACK": ("CAM_BACK",),
    "LEFT": ("CAM_FRONT_LEFT", "CAM_BACK_LEFT"),
    "RIGHT": ("CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"),
}
CONE_BLUEPRINTS = (
    "static.prop.constructioncone",
    "static.prop.trafficcone01",
    "static.prop.trafficcone02",
)
CONE_DIST_M = 12.0  # A2 世界方向锥的横向距离(m)
CONE_Z_OFF_M = 1.2  # A2 锥心相对 ego 原点的高度(m)—— 抬到相机光轴附近,不受地面坡度影响
CONE_FALLBACK_M = (1.5, -1.5, 3.0)  # A2 摆不进去时的距离回退档(只挪距离不挪方向)
LATERAL_Z_M = 20.0  # A4 横移锥的沿光轴距离(m)
LATERAL_XS = (-5.0, -2.5, 0.0, 2.5, 5.0)  # A4 横移量(m,相机自身"右"为正)
MIN_MASK_PX = 8  # 判定"这台相机看见了锥"的最小掩膜像素
MAX_MIDLINE_DEV_PX = 3.0  # 逐行中点散得过开 = 掩膜不干净,该锥弃用
# 逐点局部平面拟合。**三个常数都是实测标定的,不是拍脑袋**(见 §"为什么是这几个数")。
# 教训:用 `‖P‖`(距世界原点)当距离会把 91% 的点误剔 —— ego 出生点离原点 69 m,
# 而 LiDAR 量程只有 70 m ⇒ "r<30" 实际只留了车前 154 点。距离必须**相对 LiDAR 自身**。
PLANE_RADIUS_M = 1.2  # 逐点局部平面拟合半径(m)
PLANE_MAX_DIST_M = 50.0  # 参与平面拟合的最远点(相对 LiDAR)
PLANE_RMS_MAX_M = 0.08  # 局部平面 RMS 闸(m)。**刻意比 `calib_probe.MAX_PLANE_RESIDUAL_M`(0.03)宽**
# —— 0.03 下 CAM_BACK 只分到 2 个样本(它的最近几何在 15.9 m 外、局部平面普遍更糙),
# 0.08 下六相机各 200–1000 样本而 med|e| 仍只有 0.009–0.028 m(判据 0.1 m)。
# 放宽闸门**不会**把残差判据放松:平面糙 → 预测深度糙 → 残差自己变大,闸门在下一环。
LIDAR_VOXEL_M = 0.3  # 体素边长(m)
# 深度残差采样的点数上限:逐点邻域平面是 O(N²)(voxel 0.3 下 dist<50 有 12808 点),
# 但真正要的是"样本够多且覆盖各相机",不是"用上每一个点"。随机下采样到该数。
# 实测(2026-09-22)2500 点只留 385 个平面点、CAM_BACK 分到 2 个;8000 点 → 2734 个
# 平面点、六相机各 200–1000 样本,med|e| 0.009–0.028 m。故取 8000。
MAX_PLANE_SAMPLES = 8000


# ---------------------------------------------------------------- A0 / A1(纯值,不需要 CARLA)


def anchor_azimuth() -> dict[str, Any]:
    """A0:rig 光轴的世界方位角 vs nuScenes 官方 `calibrated_sensor` 方位角。

    rig 侧走 `live_common.rig_spec`(实时流/采集器共用的那条路)→ `carla_rotation_matrix`
    的第一列(相机局部 +x = 视线轴,与 `geometry.CARLA_TO_CAM` 的第三行一致)→ 翻到
    nuScenes 全局系;官方侧走四元数第三列(相机局部 +z = 视线轴)。两者必须同方位角。

    **这条就是历史 bug 的回归锁**:旧的 `SURROUND_CAMS` 字面表会在这里差 110.3°/217.2°。
    """
    mounts, rots = rig_spec(RIG_NUSCENES)
    rows: list[dict[str, Any]] = []
    for name in NUS_CAMERA_RIG:
        pitch, yaw, roll = (math.radians(v) for v in rots[name])
        fwd_carla = carla_rotation_matrix((pitch, yaw, roll))[:, 0]
        fwd_nus = carla_to_nus_global(fwd_carla[None])[0]
        az_rig = math.degrees(math.atan2(fwd_nus[1], fwd_nus[0]))
        q_nus = NUS_CAMERA_CALIBS[name][1]
        boresight = quat_to_matrix(quat_normalize(q_nus))[:, 2]
        az_off = math.degrees(math.atan2(boresight[1], boresight[0]))
        rows.append(
            {
                "camera": name,
                "mount": list(mounts[name]),
                "az_rig": az_rig,
                "az_official": az_off,
                "diff_deg": math.degrees(wrap_pi(math.radians(az_rig - az_off))),
            }
        )
    return {"rows": rows, "max_abs_diff_deg": max(abs(r["diff_deg"]) for r in rows)}


def anchor_side_consistency(
    spec: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A1:侧相机的挂点 y 符号必须与光轴 y 符号相同(都在左边 or 都在右边)。

    前/后相机(|挂点 y| < 0.1)近自逆,**判据对它们无信息量** → 如实排除并记数,
    不硬算一个假通过。挂点若退化成 6 路共用(legacy rig)则本判据同样退化 —— 这正是
    历史 bug 能藏住的原因之一,故报告里显式给出参与判据的相机数。

    `spec` 缺省走 `live_common.rig_spec`(运行期那条路);可注入 `(mounts, rots)` 供单测
    喂**故意镜像**的 rig,验证本判据真的能拦下历史 bug(而不是恒返回 True)。
    """
    mounts, rots = spec if spec is not None else rig_spec(RIG_NUSCENES)
    rows: list[dict[str, Any]] = []
    for name in NUS_CAMERA_RIG:
        mount, rot = mounts[name], rots[name]
        if abs(mount[1]) < 0.1:
            continue
        pitch, yaw, roll = (math.radians(v) for v in rot)
        fwd = carla_rotation_matrix((pitch, yaw, roll))[:, 0]
        rows.append(
            {
                "camera": name,
                "mount_y": mount[1],
                "boresight_y": float(fwd[1]),
                "same_side": bool(fwd[1] * mount[1] > 0.0),
            }
        )
    return {"rows": rows, "n_checked": len(rows), "all_same_side": all(r["same_side"] for r in rows)}


# ---------------------------------------------------------------- CARLA 侧工具


def decode_instance(raw: bytes, height: int, width: int) -> np.ndarray:
    """CARLA 实例分割 BGRA → actor id 数组(`id = G + 256·B`)。

    官方文档只写"每个实例一个唯一颜色",**不说通道序**。实测裁决(2026-09-22,差分法:
    同一位姿先后各取一帧,只在 actor 投影点邻域比较像素):

        BGRA = [B, G, R, A],  id = arr[1] + 256·arr[0]  (低字节在 G、高字节在 B)

    证据:4/4 命中且跨 actor 类型 —— `static.prop.constructioncone`(id 660 → `[2,148,21,255]`)、
    `vehicle.audi.a2`(661 → `[2,149,14,255]`)、`static.prop.trafficcone01`(662)、
    `walker.pedestrian.0001`(663);并跨 255 边界验证过高字节(垫 300 个 actor 后
    id 354–357 → `[1,98..101,21,255]`)。此前猜的 `R+256G+65536B` / `B+256G+65536R`
    **两种都错** —— id 全 < 256 时高位字节恒 0,两种候选都"看着像",故一直没暴露。

    **`arr[2]`(R 通道)= CARLA `CityObjectLabel` 语义类**(锥/桶 = 21 `Dynamic`、
    车 = 14 `Car`、行人 = 12 `Pedestrians`),`arr[3]` = 255。语义类不参与 id,
    但白送一条独立交叉验证(同一实例的类必须唯一)。

    返回单一口径的 id 数组(不再返回候选字典):口径已裁决,留候选只会让下游再猜一次。
    """
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4).astype(np.int64)
    return arr[:, :, 1] + 256 * arr[:, :, 0]


class SensorRig:
    """挂在同一 ego 上的一组传感器 + 各自队列(每 tick 每队列取一帧)。

    `kinds` 是 CARLA 相机蓝图后缀(如 `"rgb"` / `"depth"` / `"instance_segmentation"`);
    队列键 = `"<kinds 里的原名>:<相机名>"`,故键与蓝图名一致(避免 `instance` 这类缩写
    在蓝图查找处静默找不到)。
    """

    def __init__(self, world: carla.World, ego: carla.Vehicle, kinds: tuple[str, ...]) -> None:
        bp_lib = world.get_blueprint_library()
        self.world = world
        self.sensors: dict[str, tuple[carla.Sensor, queue.Queue]] = {}
        for kind in kinds:
            for name, (mount, rot) in NUS_CAMERA_RIG.items():
                bp = bp_lib.find(f"sensor.camera.{kind}")
                for k, v in CAM_ATTRS.items():
                    bp.set_attribute(k, v)
                tf = carla.Transform(
                    carla.Location(*mount), carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
                )
                s = cast(carla.Sensor, world.spawn_actor(bp, tf, attach_to=ego))
                q: queue.Queue = queue.Queue()
                s.listen(q.put)
                self.sensors[f"{kind}:{name}"] = (s, q)

    def add_lidar(self, world: carla.World, ego: carla.Vehicle) -> None:
        bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        for k, v in LIDAR_ATTRS.items():
            bp.set_attribute(k, v)
        s = cast(carla.Sensor, world.spawn_actor(bp, SENSOR_OFFSET, attach_to=ego))
        q: queue.Queue = queue.Queue()
        s.listen(q.put)
        self.sensors["lidar:TOP"] = (s, q)

    def warmup(self, ticks: int = 5) -> None:
        """预热:每 tick **逐队列各取一帧**。攒着不取会让队列只剩最新帧、并拖慢监听线程。"""
        for _ in range(ticks):
            self.world.tick()
            for pair in self.sensors.values():
                pair[1].get(timeout=10)

    def capture(self) -> dict[str, Any]:
        """tick 一次 + 逐队列取一帧(必须在 tick **之后**读传感器位姿)。"""
        self.world.tick()
        frames: dict[str, Any] = {}
        for key, pair in self.sensors.items():
            data = pair[1].get(timeout=10)
            if key.startswith("lidar:"):
                frames[key] = data
            else:
                kind, name = key.split(":", 1)
                frames.setdefault(kind, {})[name] = data
        return frames

    def poses(self) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """逐相机 (世界位置, (pitch,yaw,roll) 弧度)—— **capture 之后调用**(tick 后位姿才刷新)。"""
        out: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
        for key, pair in self.sensors.items():
            if key.startswith("lidar:"):
                continue
            t = pair[0].get_transform()
            out[key.split(":", 1)[1]] = (loc(t), rad(t.rotation))
        return out

    def sensor(self, key: str) -> carla.Sensor:
        return self.sensors[key][0]

    def destroy(self) -> None:
        for pair in self.sensors.values():
            pair[0].stop()
            pair[0].destroy()
        self.sensors.clear()


def clean_world(world: carla.World) -> int:
    """清场:只清我们可能留下的动态 actor + 锥体(不碰地图固有 prop)。"""
    n = 0
    for a in world.get_actors():
        if a.type_id.startswith(("vehicle", "walker", "controller")) or a.type_id in CONE_BLUEPRINTS:
            a.destroy()
            n += 1
    for _ in range(3):
        world.tick()
    return n


def try_spawn_cone(world: carla.World, location: tuple[float, float, float]) -> carla.Actor | None:
    """在指定世界位置摆一个锥体;所有候选蓝图都被占用/碰撞 → None(由调用方决定怎么办)。"""
    bp_lib = world.get_blueprint_library()
    for type_id in CONE_BLUEPRINTS:
        bp = bp_lib.find(type_id)
        if bp is None:
            continue
        actor = world.try_spawn_actor(bp, carla.Transform(carla.Location(*location), carla.Rotation(yaw=0.0)))
        if actor is not None:
            return actor
    return None


def place_cone_along(
    world: carla.World, origin: tuple[float, float, float], direction: tuple[float, float, float]
) -> tuple[carla.Actor | None, tuple[float, float, float]]:
    """沿 direction 从 origin 起按 `CONE_FALLBACK_M` 逐档试摆 → (actor|None, 实际落点)。

    只回退**距离**不回退方向:A2 的判据是"世界方位 → 哪些相机看得见",方向被挪就失去意义;
    距离 ±3 m 只让方位角变 ~14°,远小于 ±45° 半视场,判据不受影响。
    """
    for d in (0.0, *CONE_FALLBACK_M):
        p = (
            origin[0] + (CONE_DIST_M + d) * direction[0],
            origin[1] + (CONE_DIST_M + d) * direction[1],
            origin[2] + CONE_Z_OFF_M,
        )
        actor = try_spawn_cone(world, p)
        if actor is not None:
            return actor, p
    return None, (
        origin[0] + CONE_DIST_M * direction[0],
        origin[1] + CONE_DIST_M * direction[1],
        origin[2] + CONE_Z_OFF_M,
    )


# ---------------------------------------------------------------- A3:LiDAR 平面 × 渲染深度


def lidar_world_points(raw: carla.LidarMeasurement, lidar_tf: carla.Transform) -> np.ndarray:
    """LiDAR 原始帧 → 世界系点 (N,3)(**不翻 y**:y 翻转是 KITTI 落盘口径,与几何无关)。"""
    pts = np.frombuffer(raw.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    r = carla_rotation_matrix(rad(lidar_tf.rotation))
    return pts @ r.T + np.asarray(loc(lidar_tf), dtype=np.float64)


def world_planes(
    pts_world: np.ndarray, sensor_loc: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """世界点 → (保留点, 逐点法向, 逐点截距)。

    **距离一律相对 LiDAR 自身**(`sensor_loc`),不是世界原点:实测 ego 出生点离原点
    69 m、LiDAR 量程 70 m ⇒ 用 `‖P‖` 会把 91% 的点误剔成"远点"(车前只剩 154 点),
    平面拟合的邻域因此全空。这个坑不报错、只是样本数塌成 0,故写死在签名里。

    点数预算:逐点邻域平面是 O(N²),故先体素降采样 + 随机抽到 `MAX_PLANE_SAMPLES`。
    **采样在平面拟合之前**(而不是在保留点里抽):先拟合再抽会把"哪些点能过闸"这件事
    交给运气,而逐点平面拟合对每个点独立,抽样不改变任一保留点的判定。
    """
    d = np.linalg.norm(pts_world - sensor_loc, axis=1)
    down = voxel_downsample(pts_world[d < PLANE_MAX_DIST_M], LIDAR_VOXEL_M)[:, :3]
    if down.shape[0] > MAX_PLANE_SAMPLES:
        down = down[rng.choice(down.shape[0], MAX_PLANE_SAMPLES, replace=False)]
    normals, rms = cp.fit_local_planes(down, PLANE_RADIUS_M)
    keep = np.isfinite(rms) & (rms <= PLANE_RMS_MAX_M)
    pts, nrm = down[keep], normals[keep]
    return pts, nrm, np.einsum("ij,ij->i", nrm, pts)


def depth_residuals(
    pts: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    cam_pose: tuple[tuple[float, float, float], tuple[float, float, float]],
    depth_img: np.ndarray,
    occlusion_radius_px: float,
) -> cp.DepthSamples:
    """单相机的深度残差采样(射线-平面求交 vs 渲染深度,含遮挡剔除)。"""
    return cp.collect_samples(
        pts, normals, offsets, cam_pose[0], cam_pose[1], K, depth_img, PIXEL_CONVENTION, occlusion_radius_px
    )


def merge_samples(ss: list[cp.DepthSamples]) -> cp.DepthSamples:
    """多帧采样并集(静态 ego 下各帧独立同分布,并集只是把样本量做大)。"""
    if not ss:
        return cp.DepthSamples(np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros((0, 2)))
    return cp.DepthSamples(
        np.concatenate([s.uv for s in ss]),
        np.concatenate([s.z_lidar for s in ss]),
        np.concatenate([s.z_render for s in ss]),
        np.concatenate([s.grad for s in ss]),
    )


def draw_residuals(img: Image.Image, samples: cp.DepthSamples) -> int:
    """把采样点按 |残差| 着色画到图上(绿 <0.05m / 黄 <0.15m / 红 其余)。返回画上的点数。

    着色带与实时槽(`live_studio --calib`)必须**同源**,否则"离线自证看到的图"与"开着车
    看到的图"判读口径不一致 ⇒ 走 `calib_live.paint_residuals`(单一实现)。
    """
    if len(samples) == 0:
        return 0
    arr = np.asarray(img).copy()
    n = cl.paint_residuals(arr, samples.uv, samples.residual)
    img.paste(Image.fromarray(arr))
    return n


# ---------------------------------------------------------------- 锥体(实例分割)辅助


def instance_arrays(frames: dict[str, carla.Image]) -> dict[str, np.ndarray]:
    """逐相机实例分割帧 → actor id 数组(口径见 `decode_instance`)。"""
    return {n: decode_instance(frames[n].raw_data, H, W) for n in NUS_CAMERA_RIG}


def instance_hits(decoded: dict[str, np.ndarray], cone_ids: list[int]) -> int:
    """全部相机上命中给定 actor id 的总像素数。

    必须在**全部相机**上统计:锥可能只落在部分相机里(如 LEFT 方向的锥看不见 CAM_FRONT),
    只看一路会把正确解码判成未命中。
    """
    return sum(int(np.isin(d, cone_ids).sum()) for d in decoded.values())


def require_instance_hits(decoded: dict[str, np.ndarray], cone_ids: list[int]) -> int:
    """口径自检:解码结果必须真的命中锥体 actor id,否则**响亮失败**(不静默给空掩膜)。

    解码口径已由差分实验裁决(`decode_instance`),但 CARLA 版本/渲染后端若改了编码,
    这里会立刻炸出来,而不是让 A2/A4 一路带着 0 掩膜"通过"。
    """
    hits = instance_hits(decoded, cone_ids)
    if hits < MIN_MASK_PX:
        raise RuntimeError(f"实例分割解码 `G+256·B` 未命中锥体 id={cone_ids}(命中数 {hits})")
    return hits


def cone_masks(decoded: dict[str, np.ndarray], cone_id: int) -> dict[str, np.ndarray]:
    """逐相机 → 该锥的布尔掩膜。"""
    return {n: d == cone_id for n, d in decoded.items()}


def mask_centre_u(mask: np.ndarray) -> float | None:
    """锥体掩膜 → 逐行中点列的中位数(**索引**口径);掩膜太小/不稳 → None(不硬给数)。"""
    if int(mask.sum()) < MIN_MASK_PX:
        return None
    mids = cp.mask_row_midpoints(mask)[1]
    if mids.size < 3 or not np.isfinite(cp.midline_deviation(mids)):
        return None
    if cp.midline_deviation(mids) > MAX_MIDLINE_DEV_PX:
        return None
    return float(np.median(mids))


def lateral_cone_positions(cam_pose: tuple[tuple[float, float, float], tuple[float, float, float]]) -> list:
    """沿相机光轴 z=LATERAL_Z_M 处、按相机自身"右"轴横移 LATERAL_XS 的世界点。

    相机局部(KITTI 口径:x 右 / y 下 / z 前)→ 世界走 `calib_probe.cam_to_world_rot`,
    与 `project_world` 严格互逆 —— **不在 bin 里另拼一遍旋转**。
    """
    r = cp.cam_to_world_rot(cam_pose[1])
    o = np.asarray(cam_pose[0], dtype=np.float64)
    return [tuple(o + r @ np.array([x, 0.0, LATERAL_Z_M])) for x in LATERAL_XS]


# ---------------------------------------------------------------- 主流程


def pass_a3(
    rig: SensorRig, args: argparse.Namespace
) -> tuple[dict[str, Any], list[list[tuple[str, Image.Image]]]]:
    """A3:LiDAR 平面 × 渲染深度(逐相机残差 + 主点偏差 + 径向剖面)。"""
    rng = np.random.default_rng(args.seed)
    rig.warmup(5)
    frames = rig.capture()  # 静态 ego:点云与渲染逐帧相同,故平面只拟合一次
    poses = rig.poses()
    lidar_tf = rig.sensor("lidar:TOP").get_transform()
    lidar_loc = np.asarray(loc(lidar_tf), dtype=np.float64)
    pts_world = lidar_world_points(frames["lidar:TOP"], lidar_tf)
    pts, nrm, offs = world_planes(pts_world, lidar_loc, rng)
    print(
        f"[A3] LiDAR {pts_world.shape[0]} 点 → 平面质量保留 {pts.shape[0]} 点"
        f"(相对 LiDAR 截断 {PLANE_MAX_DIST_M} m / 半径 {PLANE_RADIUS_M} m / 体素 {LIDAR_VOXEL_M} m)"
    )

    per_cam: dict[str, list[cp.DepthSamples]] = {n: [] for n in NUS_CAMERA_RIG}
    overlay_rows: list[list[tuple[str, Image.Image]]] = []
    for i in range(args.frames):
        f = frames if i == 0 else rig.capture()
        p = poses if i == 0 else rig.poses()
        row: list[tuple[str, Image.Image]] = []
        for name in NUS_CAMERA_RIG:
            d_img = decode_depth(f["depth"][name].raw_data, H, W)
            s = depth_residuals(pts, nrm, offs, p[name], d_img, args.edge_radius_px)
            per_cam[name].append(s)
            img = image_to_pil(f["rgb"][name])
            draw_residuals(img, s)
            fonts.draw_text(
                ImageDraw.Draw(img), (8, H - 18), f"{name} 采样 {len(s)}", size=15, fill=(255, 255, 0)
            )
            if i == args.frames - 1:
                row.append((name, img))
                if len(row) == 2:
                    overlay_rows.append(row)
                    row = []

    a3: dict[str, Any] = {}
    print(
        f"[A3] 判据 median|e| < 0.1 m(采样约定 {PIXEL_CONVENTION});可见性单侧判据 "
        f"max({cp.OCCLUSION_TOL_M} m, {cp.OCCLUSION_TOL_FRAC:.2f}·z),边缘窗 r={args.edge_radius_px} px"
    )
    print(
        f"  {'相机':<17}{'样本':>7}{'|e|中位':>10}{'|e|P90':>9}{'δu px':>9}{'σδu':>7}"
        f"{'δv px':>9}{'σδv':>7}{'径向斜率':>11}"
    )
    for name, ss in per_cam.items():
        m = merge_samples(ss)
        e = np.abs(m.residual)
        fu, fv = cp.estimate_delta_uv(m)
        slope = cp.radial_error_profile(m, K)[3]
        a3[name] = {
            "n": len(m),
            "median_abs_residual_m": float(np.median(e)) if e.size else None,
            "p90_abs_residual_m": float(np.percentile(e, 90)) if e.size else None,
            "delta_u_px": fu.delta_px if fu.identifiable else None,
            "sigma_delta_u_px": fu.sigma_delta_px if fu.identifiable else None,
            "delta_v_px": fv.delta_px if fv.identifiable else None,
            "sigma_delta_v_px": fv.sigma_delta_px if fv.identifiable else None,
            "delta_u_identifiable": fu.identifiable,
            "delta_v_identifiable": fv.identifiable,
            "radial_slope": slope,
        }
        print(
            f"  {name:<17}{len(m):>7}"
            f"{(a3[name]['median_abs_residual_m'] or float('nan')):>10.4f}"
            f"{(a3[name]['p90_abs_residual_m'] or float('nan')):>9.4f}"
            f"{(a3[name]['delta_u_px'] or float('nan')):>9.3f}"
            f"{(a3[name]['sigma_delta_u_px'] or float('nan')):>7.3f}"
            f"{(a3[name]['delta_v_px'] or float('nan')):>9.3f}"
            f"{(a3[name]['sigma_delta_v_px'] or float('nan')):>7.3f}"
            f"{slope:>11.2e}"
        )
    return a3, overlay_rows


def pass_cones(
    world: carla.World, ego: carla.Vehicle, crig: SensorRig
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """A2 + A4:锥体(世界方向语义 + 横移回归)。返回 (a2, a4, 实际摆放记录)。"""
    crig.warmup(5)
    ego_loc = loc(ego.get_transform())
    spawned: list[carla.Actor] = []
    placed: dict[str, Any] = {}

    def shoot() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """取一帧(锥已摆好),返回 (逐相机 actor id 数组, 位姿)。"""
        f = crig.capture()
        return instance_arrays(f["instance_segmentation"]), {"poses": crig.poses()}

    # ---- A2:世界方向语义 ----
    print("\n[A2] 世界方向语义(锥放在世界的某个方向,看见它的相机名必须含该方位词)")
    a2: dict[str, Any] = {}
    for direction, d in WORLD_DIRS.items():
        for a in spawned:
            a.destroy()
        spawned.clear()
        actor, target = place_cone_along(world, ego_loc, d)
        if actor is None:
            a2[direction] = {"placed": False, "reason": "所有候选蓝图/回退距离均 spawn 失败"}
            print(f"  ✗ {direction:<6} 摆不进去(碰撞),该方向判据跳过")
            continue
        spawned.append(actor)
        inst, meta = shoot()
        hits = require_instance_hits(inst, [actor.id])
        masks = cone_masks(inst, actor.id)
        rows: dict[str, Any] = {}
        for name in NUS_CAMERA_RIG:
            cam_loc, cam_rot = meta["poses"][name]
            uv, z = cp.project_world(np.asarray(target)[None], cam_loc, cam_rot, K)
            u, v = float(uv[0, 0]), float(uv[0, 1])
            in_fov = bool(0.0 <= u < W and 0.0 <= v < H and z[0] > 0.5)
            px = int(masks[name].sum())
            rows[name] = {"u": u, "v": v, "in_fov": in_fov, "mask_px": px, "sees": px >= MIN_MASK_PX}
        seers = tuple(n for n in NUS_CAMERA_RIG if rows[n]["sees"])
        a2[direction] = {
            "placed": True,
            "cone_actor_id": actor.id,
            "target_world": list(target),
            "instance_hits": hits,
            "cameras": rows,
            "seers": list(seers),
            "expected": list(EXPECTED_SEERS[direction]),
            "match": set(seers) == set(EXPECTED_SEERS[direction]),
        }
        print(
            f"  {'✓' if a2[direction]['match'] else '✗'} {direction:<6} 看见的相机 {list(seers) or '(无)'}"
            f"  预期 {list(EXPECTED_SEERS[direction])}"
        )
        for name in NUS_CAMERA_RIG:
            r = rows[name]
            if r["sees"] or r["in_fov"]:
                print(
                    f"      {name:<17} u={r['u']:8.1f} v={r['v']:7.1f} in_fov={int(r['in_fov'])} "
                    f"mask={r['mask_px']:5d}px sees={int(r['sees'])}"
                )

    # ---- A4:横移锥回归 ----
    print(f"\n[A4] 横移锥回归({LATERAL_Z_M:.0f} m 处横移 {LATERAL_XS} m,只测 CAM_FRONT)")
    cam_loc, cam_rot = crig.poses()["CAM_FRONT"]
    positions = lateral_cone_positions((cam_loc, cam_rot))
    for a in spawned:
        a.destroy()
    spawned.clear()
    cones: list[tuple[float, carla.Actor]] = []
    for x, p in zip(LATERAL_XS, positions, strict=True):
        a = try_spawn_cone(world, p)
        if a is None:
            print(f"      x={x:+.1f} m 摆不进去(碰撞)→ 该点弃用")
            continue
        spawned.append(a)
        cones.append((x, a))
    a4: dict[str, Any] = {"requested_x_m": list(LATERAL_XS), "n_placed": len(cones)}
    if len(cones) >= 3:
        decoded, _ = shoot()  # A4 只用 CAM_FRONT 的掩膜,不需要位姿
        hits = require_instance_hits(decoded, [c[1].id for c in cones])
        used_x: list[float] = []
        centres: list[float] = []
        per_cone: list[dict[str, Any]] = []
        for x, a in cones:
            c = mask_centre_u(cone_masks(decoded, a.id)["CAM_FRONT"])
            per_cone.append({"x_m": x, "actor_id": a.id, "centre_u_index": c})
            if c is not None:
                used_x.append(x)
                centres.append(c)
        shift = convention_shift(PIXEL_CONVENTION)
        a4.update(
            {
                "instance_hits": hits,
                "per_cone": per_cone,
                "n_used": len(used_x),
                "offsets_x_m": used_x,
                "centres_u_index": centres,
                "index_shift": shift,
            }
        )
        if len(used_x) >= 3:
            # `prop_axis_regression` 的截距在**索引**口径;+shift 才是图像坐标。故用
            # shift=0 解出**原始截距**,再由它换算两种约定(不两次调回归、不重复拟合)。
            raw_intercept, slope, resid = cp.prop_axis_regression(
                np.asarray(used_x), np.asarray(centres), 0.0
            )
            cx_corner = raw_intercept  # corner:索引即图像坐标
            cx_center = raw_intercept + convention_shift(CONVENTION_CENTER)
            a4.update(
                {
                    "cx_corner_px": cx_corner,
                    "cx_center_px": cx_center,
                    "fx_over_z_px_per_m": slope,
                    "fx_est_px": slope * LATERAL_Z_M,
                    "max_abs_residual_px": float(np.max(np.abs(resid))),
                    "residuals_px": [float(v) for v in resid],
                }
            )
            print(
                f"      用 {len(used_x)} 个锥:cx = {cx_corner:.3f} px(corner 约定,本探针口径)/ "
                f"{cx_center:.3f} px(center 约定);f_est = {slope * LATERAL_Z_M:.2f} px"
                f"(标称 {K.fx:.2f});最大残差 {np.max(np.abs(resid)):.3f} px"
            )
        else:
            a4["verdict"] = f"可用掩膜不足({len(used_x)} < 3),主点不可辨识 —— 不硬给数"
            print(f"      [不可辨识] 只有 {len(used_x)} 个锥的掩膜可用(<3)")
    else:
        a4["verdict"] = f"摆进去的锥不足({len(cones)} < 3),主点不可辨识 —— 不硬给数"
        print(f"      [不可辨识] 只摆进去 {len(cones)} 个锥(<3)")

    placed = {"a2_cone_ids": {k: v.get("cone_actor_id") for k, v in a2.items() if v.get("placed")}}
    for a in spawned:
        a.destroy()
    return a2, a4, placed


def run(args: argparse.Namespace) -> dict[str, Any]:
    client = carla.Client(args.host, args.sim_port)
    client.set_timeout(60.0)
    world = client.get_world()
    sync_mode(world)
    n_cleared = clean_world(world)
    print(f"[setup] 清场 {n_cleared} 个 actor;map = {world.get_map().name}")

    ego = spawn_ego(world)
    ego.set_autopilot(False)
    for _ in range(10):  # 让悬挂/物理沉降稳定(手刹 + 制动,不再漂)
        ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        world.tick()
    ego_t = ego.get_transform()
    right_v = ego_t.get_right_vector()
    print(f"[setup] ego @ {loc(ego_t)} yaw={ego_t.rotation.yaw:.3f}°")
    print(f"[setup] get_right_vector() = ({right_v.x:.3f}, {right_v.y:.3f}, {right_v.z:.3f})")

    report: dict[str, Any] = {
        "map": world.get_map().name,
        "frames": args.frames,
        "ego_location": list(loc(ego_t)),
        "ego_yaw_deg": ego_t.rotation.yaw,
        "right_vector": [right_v.x, right_v.y, right_v.z],
        "right_vector_is_plus_y": bool(right_v.y > 0.9),
    }

    # ---- A0 / A1:纯值锚 ----
    report["A0_azimuth_vs_official"] = anchor_azimuth()
    report["A1_side_consistency"] = anchor_side_consistency()
    print(f"\n[A0] rig 光轴 vs 官方方位角:最大差 {report['A0_azimuth_vs_official']['max_abs_diff_deg']:.6f}°")
    print(
        f"[A1] 侧别一致性:{report['A1_side_consistency']['n_checked']} 台侧相机,"
        f"全部同侧 = {report['A1_side_consistency']['all_same_side']}"
    )

    # ---- Pass 1:RGB + 深度 + LiDAR ----
    rig = SensorRig(world, ego, ("rgb", "depth"))
    rig.add_lidar(world, ego)
    a3, overlay_rows = pass_a3(rig, args)
    report["A3_depth_residual"] = a3

    dev_t, dev_y = rig_mount_deviation(
        {n: (rig.sensor(f"rgb:{n}"), K) for n in NUS_CAMERA_RIG}, ego, RIG_NUSCENES
    )
    report["A5_mount_deviation"] = {"max_translation_m": dev_t, "max_yaw_deg": dev_y}
    print(f"\n[A5] 实挂 vs spec:平移最大偏差 {dev_t:.3e} m,偏航最大偏差 {dev_y:.3e}°")
    rig.destroy()

    # ---- Pass 2/3:锥体(必须先跑完 A3:锥会污染路面点云) ----
    crig = SensorRig(world, ego, ("rgb", "instance_segmentation"))
    a2, a4, placed = pass_cones(world, ego, crig)
    report["A2_world_direction"] = a2
    report["A4_lateral_regression"] = a4
    report["cone_placement"] = placed
    crig.destroy()

    report["nominal"] = {
        "width": W,
        "height": H,
        "fx_px": K.fx,
        "fov_h_deg": FOV,
        "cx_center_px": W / 2.0,
        "cx_corner_px": (W - 1) / 2.0,
        "cy_center_px": H / 2.0,
        "cy_corner_px": (H - 1) / 2.0,
        "occlusion_tol_m": cp.OCCLUSION_TOL_M,
        "occlusion_tol_frac": cp.OCCLUSION_TOL_FRAC,
        "edge_radius_px": args.edge_radius_px,
        "plane_max_dist_m": PLANE_MAX_DIST_M,
        "plane_rms_max_m": PLANE_RMS_MAX_M,
        "max_plane_samples": MAX_PLANE_SAMPLES,
    }

    ego.destroy()
    world.apply_settings(carla.WorldSettings())
    return {"report": report, "overlay_rows": overlay_rows}


def summarize(r: dict[str, Any]) -> tuple[dict[str, bool], str]:
    """把报告折成逐锚的布尔判据(阈值口径写在 docstring 表格里)。"""
    a2_placed = {k: v for k, v in r["A2_world_direction"].items() if v.get("placed")}
    a4 = r["A4_lateral_regression"]
    nom = r["nominal"]
    # A4 的 cx 是**直接读数**(掩膜中点回归,不经深度),故它是主点的独立裁决:
    # 量出来的图像坐标主点必须落在标称值上,否则说明 K 的主点错了。
    cx_est = a4.get("cx_corner_px")
    checks = {
        "A0": r["A0_azimuth_vs_official"]["max_abs_diff_deg"] < 0.01,
        "A1": r["A1_side_consistency"]["all_same_side"],
        "A2": bool(a2_placed) and all(v["match"] for v in a2_placed.values()),
        "A3": all((v["median_abs_residual_m"] or 1e9) < 0.1 for v in r["A3_depth_residual"].values()),
        "A4": (a4.get("max_abs_residual_px") or 1e9) < 0.3 and a4.get("n_used", 0) >= 3,
        "A5": r["A5_mount_deviation"]["max_translation_m"] < 1e-3
        and r["A5_mount_deviation"]["max_yaw_deg"] < 1e-3,
        # A6:主点裁决 —— A4 直读的 cx(corner 口径)必须等于 `CameraIntrinsics.cx = (w−1)/2`。
        # 这条同时钉住**像素约定**:若渲染器其实是 center 口径,A4 会读出 620.00 而不是 620.50。
        "A6": cx_est is not None and abs(cx_est - nom["cx_corner_px"]) < 0.5,
    }
    verdict = (
        f"A4 直读 cx_corner={cx_est} px(标称 {nom['cx_corner_px']});"
        f" 采样约定={PIXEL_CONVENTION}(A3 median|e| 裁决:corner 0.0003 m vs center 0.023 m)"
    )
    return checks, verdict


def main() -> int:
    ap = argparse.ArgumentParser(description="环视相机标定自证探针(CARLA 渲染器当第二把尺子)")
    ap.add_argument("--frames", type=int, default=2, help="A3 取几帧(残差取并集;静态 ego 下逐帧相同)")
    ap.add_argument("--out", default="outputs/calib_check")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--sim-port", type=int, default=2000)
    ap.add_argument(
        "--edge-radius-px",
        type=float,
        default=6.0,
        help="深度断裂边缘剔除的窗口半径(px);0 = 关(只留单侧可见性判据)",
    )
    ap.add_argument("--seed", type=int, default=42, help="平面样本下采样的随机种子(可复现)")
    args = ap.parse_args()

    result = run(args)
    report = result["report"]
    checks, verdict = summarize(report)
    report["verdict"] = checks

    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    if result["overlay_rows"]:
        compose_rows(result["overlay_rows"]).save(out / "overlay.png")
    print(f"\n[done] {out}/report.json + overlay.png")
    print("判据 " + " | ".join(f"{k} {'✓' if v else '✗'}" for k, v in checks.items()))
    print("主点 " + verdict)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

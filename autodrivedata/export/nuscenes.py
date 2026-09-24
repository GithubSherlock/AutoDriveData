"""nuScenes 迷你集生成器——照 devkit NuScenes.__init__ 契约 + auto3dlabel 消费链。

契约钉死点(2026-09-07 实测 devkit 1.2.0,详见 Plan §5.2):
- 表目录 = dataroot/<version>/,14 张表 JSON,每条记录唯一 token;不放 lidarseg/panoptic
- map 表非空,filename 指向的文件必须存在(MapMask assert);category 需 'name'、
  instance 需 category_token、sample_data 需 calibrated_sensor/sensor/modality/channel
- sample['data']/['anns'] 由 devkit 从 sample_data 的 is_key_frame 记录反查,勿手填
- LIDAR 文件 = (N,5) float32 raw(x,y,z,intensity,elongation),filename 相对 dataroot
- 雷达文件 = 18 字段 .pcd(devkit RadarPointCloud 契约),filename 相对 dataroot;
  sample_annotation 带 num_radar_pts(默认 0,无雷达的旧数据兼容)
- 坐标系:入表数据一律 nuScenes 全局系(x 前/y 左/z 上)——CARLA 数据入表前
  经 geometry.CARLA_TO_NUS 翻转(y 符号)。**且该系的原点 = 后轴中心在地面**
  (见 `geometry.NUS_EGO_ORIGIN_X`):`ego_pose.translation` 与 `calibrated_sensor.translation`
  **都**以它为基准,故采集器写 `ego_pose` 时必须走 `geometry.nus_ego_translation`
  (用 CARLA actor 原点会让整组传感器相对自车偏 1.2563 m,§P-M.10)
- 场景名必须 ∈ devkit create_splits_scenes()['mini_val'](auto3dlabel
  generate_review_queue 只遍历 val 场景)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from autodrivedata import geometry as g

# 单点来源,勿在此另抄一份。两个 rig 的**姿态设计**都在 `camera_rig`:
# `nuscenes` = 官方 calibrated_sensor;`wide` = 官方表 + 后移挂点/换轴(见 §P-M.8)。
from autodrivedata.camera_rig import (
    NUS_CAMERA_CALIBS,
    NUS_WIDE_CAMERA_CALIBS,
    NUS_WIDE_CAMERA_FOV,
)

NUS_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

# 相机 rig 名。`nuscenes` = **官方 nuScenes 标定**(默认,行为不得变);
# `wide` = 自定义宽视场口径(前 55°/后侧 110°/后 120°、后三路挂点后移到车尾)。
NUS_RIGS: tuple[str, ...] = ("nuscenes", "wide")
NUS_RIG_DEFAULT = "nuscenes"

# 官方 nuScenes 5 雷达通道(与 nuScenes 官方一致;mini 集只有 5 雷达无 RADAR_BACK)
NUS_RADAR_CHANNELS = (
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
)

# 官方相机 **内参**(1600×900,逐通道 fx=fy / cx / cy)——照 nuscenes_mini 的 n015 车
# (singapore-*,与 `camera_rig.NUS_CAMERA_CALIBS` 的外参同源;**不要混抄 n008/boston-seaport
# 那一套**,逐通道 fx/cx/cy 全都不同,如 CAM_FRONT n015 1266.42 vs n008 1252.81)。
#
# ★ 官方主点逐通道不同且**不等于** `(w−1)/2`(792–829 / 479–501 vs 799.5 / 449.5)。
#   **这不违反 §P-M 的 corner 裁决,也不该被"修正"**:
#   - corner 裁决描述的是 **CARLA 渲染栅格**的索引约定(`cx=(w−1)/2`)——我们渲染时按它采样;
#   - 官方 K 是**真实相机**的装配公差实测值 —— 我们落盘时按它写,因为消费方
#     (devkit / auto3dlabel)按官方口径解释。
#   两者角色不同,不冲突也不可互换。
NUS_CAMERA_INTRINSICS: dict[str, tuple[float, float, float]] = {
    "CAM_FRONT": (1266.417203046554, 816.2670197447984, 491.50706579294757),
    "CAM_FRONT_LEFT": (1272.5979470598488, 826.6154927353808, 479.75165386361925),
    "CAM_FRONT_RIGHT": (1260.8474446004698, 807.968244525554, 495.3344268742088),
    "CAM_BACK": (809.2209905677063, 829.2196003259838, 481.77842384512485),
    "CAM_BACK_LEFT": (1256.7414812095406, 792.1125740759628, 492.7757465151356),
    "CAM_BACK_RIGHT": (1259.5137405846733, 807.2529053838625, 501.19579884916527),
}


NUS_CAMERA_WIDTH, NUS_CAMERA_HEIGHT = 1600, 900


def camera_fov_h_deg(cam: str) -> float:
    """逐通道**水平 FOV**(度)= `2·atan((w/2)/fx)`,由官方 K 反推。

    CARLA 蓝图 `fov` 属性就是水平 FOV,故采集侧直接拿它设蓝图 —— 这样"渲染视野"与
    "落盘 K"由同一个 fx 导出,不存在"标定说 64°、图像是 90°"的声明≠渲染。

    用 `w/2`(半宽)而不是 `(w−1)/2`:FOV 是"半视场↔半宽"的几何关系,与像素索引约定
    (corner/center)无关 —— 后者只影响主点 `cx`,不影响视场角。两者并存不矛盾。
    """
    fx = NUS_CAMERA_INTRINSICS[cam][0]
    return math.degrees(2.0 * math.atan((NUS_CAMERA_WIDTH / 2.0) / fx))


# 逐通道蓝图 fov(度)—— 由 `NUS_CAMERA_INTRINSICS` 导出,不手抄第二份
NUS_CAMERA_FOV: dict[str, float] = {cam: camera_fov_h_deg(cam) for cam in NUS_CAMERAS}


# ── wide rig 的内参(2026-09-23,§P-M.8)────────────────────────────────────
#
# ★ 与官方表的**角色故意不同**,别来"统一":
#   - 官方 K = 真实相机的**装配公差实测值**(逐通道主点 792–829 / 479–501),照抄官方;
#   - wide K = 由**声明的 FoV 反推**,主点取 **corner 约定** `(w−1)/2`。
#
# 为什么 wide 用 corner:wide rig 的图是 **CARLA 渲染栅格**生成的,而 corner 正是 CARLA
# 栅格的索引约定(§P-M 裁决:A3 corner 残差 0.0003 m vs center 0.023 m,70×)。wide rig
# 没有"别人的相机"可言,写官方主点才是错的 —— 那描述的是 nuScenes 的车,不是我们的渲染器。
#
# fx = `(w/2)/tan(hfov/2)`:用 `w/2`(半宽)而非 `(w−1)/2`,与 `camera_fov_h_deg` 同口径
# (视场角是"半视场↔半宽"的几何关系,与像素索引约定无关;索引约定只定主点)。
def _wide_intrinsics() -> dict[str, tuple[float, float, float]]:
    return {
        cam: (
            (NUS_CAMERA_WIDTH / 2.0) / math.tan(math.radians(fov) / 2.0),
            (NUS_CAMERA_WIDTH - 1) / 2.0,
            (NUS_CAMERA_HEIGHT - 1) / 2.0,
        )
        for cam, fov in NUS_WIDE_CAMERA_FOV.items()
    }


NUS_WIDE_CAMERA_INTRINSICS: dict[str, tuple[float, float, float]] = _wide_intrinsics()


def _intrinsics(rig: str = NUS_RIG_DEFAULT) -> dict[str, tuple[float, float, float]]:
    """rig 名 → 逐通道 `(fx, cx, cy)`。未知 rig 抛错(静默取默认会让落盘口径与 spawn 分叉)。"""
    if rig == "nuscenes":
        return NUS_CAMERA_INTRINSICS
    if rig == "wide":
        return NUS_WIDE_CAMERA_INTRINSICS
    raise ValueError(f"未知相机 rig:{rig!r}(可选 {NUS_RIGS})")


def camera_calibs(
    rig: str = NUS_RIG_DEFAULT,
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
    """rig 名 → 逐通道 nuScenes 侧标定 `(translation 米, rotation 四元数 wxyz)`。"""
    if rig == "nuscenes":
        return NUS_CAMERA_CALIBS
    if rig == "wide":
        return NUS_WIDE_CAMERA_CALIBS
    raise ValueError(f"未知相机 rig:{rig!r}(可选 {NUS_RIGS})")


def camera_fov(rig: str = NUS_RIG_DEFAULT) -> dict[str, float]:
    """rig 名 → 逐通道水平 FOV(度),直接给 CARLA 蓝图 `fov` 用。"""
    if rig == "nuscenes":
        return NUS_CAMERA_FOV
    if rig == "wide":
        return NUS_WIDE_CAMERA_FOV
    raise ValueError(f"未知相机 rig:{rig!r}(可选 {NUS_RIGS})")


def camera_fov_h_deg_for(cam: str, rig: str = NUS_RIG_DEFAULT) -> float:
    """逐通道水平 FOV(度)= `2·atan((w/2)/fx)`,由**该 rig 的内参**反推。

    官方 rig 下与 `camera_fov_h_deg` 等价;wide rig 下是"K ↔ fov"的一致性锁
    (wide 的 fov 是输入、K 是导出,这条反向链给出往返判据)。
    """
    fx = _intrinsics(rig)[cam][0]
    return math.degrees(2.0 * math.atan((NUS_CAMERA_WIDTH / 2.0) / fx))


# 官方 LIDAR_TOP 标定(translation 米 / rotation 四元数 wxyz)—— 照 nuscenes_mini n015。
# 四元数含 **1.4289° 的 up 轴倾角**(不是 yaw-only):360° 扫描下 yaw 不可观测,但 pitch/roll
# 由该倾角决定、**是可观测量**(30 m 处 ≈ 0.75 m 高程差)。故 `calib_lidar` 必须走四元数,
# 用 yaw-only 表达不了(见 `points_sensor_to_global_nus` 的签名)。
NUS_LIDAR_CALIB: tuple[tuple[float, float, float], tuple[float, float, float, float]] = (
    (0.943713, 0.0, 1.84023),
    (0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817),
)

# 官方 5 雷达安装位姿(translation 米, yaw_nus 弧度)—— 照 nuscenes_mini n015 实测
# `calibrated_sensor`(nus 系,y 左)。translation 与官方**逐分量完全相等**;rotation 的
# pitch/roll 实测**精确为 0**(官方四元数 x=y=0、|q|=1)⇒ 本表用 yaw 表达**无损**,
# yaw 取自官方四元数的 `2·atan2(z, w)`(**精确值**,不是把 az 抄成 3 位小数再转弧度):
# 这样写进 `calibrated_sensor.rotation` 的四元数与官方**逐位相同**。
# (Plan2 §P-M.7.10 附录把 az 显示为 3 位小数:+0.200 / +88.360 / −90.980 / +174.410 / −176.110°。)
# 采集器 CARLA spawn 用其 CARLA 镜像(y 取负、yaw = −az_nus),这里直接进 calib 表零转换。
NUS_RADAR_OFFSETS: dict[str, tuple[tuple[float, float, float], float]] = {
    "RADAR_FRONT": ((3.412, 0.0, 0.5), 0.003490658503988659),
    "RADAR_FRONT_LEFT": ((2.422, 0.8, 0.78), 1.5421729270621893),  # az +88.360°
    "RADAR_FRONT_RIGHT": ((2.422, -0.8, 0.77), -1.587900553464441),  # az −90.980°
    "RADAR_BACK_LEFT": ((-0.562, 0.628, 0.53), 3.04402874840331),  # az +174.410°
    "RADAR_BACK_RIGHT": ((-0.562, -0.618, 0.53), -3.0736993456872144),  # az −176.110°
}

# ---------------------------------------------------------------- 声明(nus 系)→ 渲染(CARLA 系)
#
# ★ 上面三张表**全是 nuScenes 系**(原点 = 后轴中心在地面);CARLA spawn 用的是 actor 系
# (原点 = 车身长度中点)。换算只有一条:`geometry.nus_mount_to_carla`(x 加
# `NUS_EGO_ORIGIN_X` + y 翻号)。下面两个常量是它的**唯一展开点** —— 采集器与验收器
# 都从这里取,不许就地再写一份 `(t[0], -t[1], t[2])`(那正是 §P-M.7「表对了、图错了」的成因)。
NUS_LIDAR_MOUNT_CARLA: tuple[float, float, float] = g.nus_mount_to_carla(NUS_LIDAR_CALIB[0])
NUS_RADAR_MOUNTS_CARLA: dict[str, tuple[float, float, float]] = {
    ch: g.nus_mount_to_carla(t) for ch, (t, _) in NUS_RADAR_OFFSETS.items()
}


def radar_yaw_offset_carla(channel: str) -> float:
    """雷达 CARLA 侧偏航(度)= `−degrees(官方 yaw_nus)`。与相机的 `yaw_carla = −az_nus` 同一条规则。"""
    return -math.degrees(NUS_RADAR_OFFSETS[channel][1])


# 检测类名 → devkit category_name(入 sample_annotation 经 instance→category 链)
NUS_NAME_TO_CATEGORY = {
    "car": "vehicle.car",
    "pedestrian": "human.pedestrian.adult",
    "bicycle": "vehicle.bicycle",
    "motorcycle": "vehicle.motorcycle",
    "truck": "vehicle.truck",
}


@dataclass(frozen=True)
class NusSample:
    """单个 sample 的落盘内容(表行由 write_mini_dataset 统一生成)。"""

    ego_translation: tuple[float, float, float]  # nuScenes 全局系
    # ★ ego 姿态**四元数**(w,x,y,z),**不是 yaw**:`vehicle.audi.a2` 静止后悬架沉下来有
    # 实测 +0.0642° 的俯仰,拍平成纯偏航会让 `ego_pose ⊕ calibrated_sensor` 与世界系真值
    # 差这个量级(1.2563 m 杆臂上 = 1.4e-3 m),而"实挂 vs 声明"逐传感器判据**看不出来**
    # (两边同错)。来源唯一:`geometry.nus_ego_rotation(actor 的 (pitch,yaw,roll))`。
    ego_rotation_nus: tuple[float, float, float, float]
    lidar_filename: str  # 相对 dataroot
    camera_filenames: dict[str, str]  # 6 通道 → 相对 dataroot
    # (translation, quat wxyz) 传感器→ego。**必须是四元数不是 yaw**:LiDAR 的 1.4289°
    # up 轴倾角表达不了 yaw-only(见 `NUS_LIDAR_CALIB` / `points_sensor_to_global_nus`)。
    calib_lidar: tuple[tuple[float, float, float], tuple[float, float, float, float]]
    # 逐通道相机标定(translation, quat wxyz)。**落盘不读本字段**——写表走模块常量
    # `NUS_CAMERA_CALIBS`(单点来源);本字段供调用方携带"这一帧相机实际用的标定"做溯源。
    calib_cameras: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]]
    annotations: list[dict]  # {'category','translation','size','yaw_nus','num_lidar_pts','instance_token'}
    timestamp: int
    # 5 雷达(2026-09-14 扩展;缺省空 dict,向后兼容旧构造)
    radar_filenames: dict[str, str] = field(default_factory=dict)  # 5 通道 → 相对 dataroot(.pcd)
    # (translation, yaw_nus) —— 官方雷达四元数 pitch/roll 精确为 0,yaw-only 无损
    calib_radars: dict[str, tuple[tuple[float, float, float], float]] = field(default_factory=dict)


def points_sensor_to_global_nus(
    points_sensor: np.ndarray,
    ego_translation: tuple[float, float, float],
    ego_rotation_nus: tuple[float, float, float, float],
    calib_translation: tuple[float, float, float],
    calib_quat: tuple[float, float, float, float],
) -> np.ndarray:
    """传感器系点 (N,3)(x 前/y 左/z 上)→ nuScenes 全局系。

    链(照 auto3dlabel boxes_sensor_to_global):p_g = R_ego @ (R_calib @ p_s + t_calib) + t_ego。

    **两个四元数都不是 yaw**:① `calib_quat` —— LiDAR 的 up 轴倾角(1.4289°)只有四元数能
    表达(实测丢 pitch/roll 后 `num_lidar_pts` 复现比值 1.0000 → 0.9097);② `ego_rotation_nus`
    —— 车体静止后的悬架俯仰(实测 +0.0642°)同理。雷达侧调用点传 `g.yaw_to_quat(yaw)`
    (官方雷达 pitch/roll 精确为 0 ⇒ 无损),于是 LiDAR / 雷达 / 相机共用**这一条链、一个函数**。
    """
    r_ego = g.quat_to_matrix(g.quat_normalize(ego_rotation_nus))
    r_calib = g.quat_to_matrix(g.quat_normalize(calib_quat))
    t_c = np.asarray(calib_translation, dtype=np.float64)[:, None]
    t_e = np.asarray(ego_translation, dtype=np.float64)[:, None]
    pts = np.asarray(points_sensor, dtype=np.float64)[:, :3].T
    return (r_ego @ (r_calib @ pts + t_c) + t_e).T


def count_points_in_box_nus(
    points_global: np.ndarray,
    translation: tuple[float, float, float],
    size: tuple[float, float, float],
    yaw_nus: float,
) -> int:
    """全局系点 → 框内点计数(size=(w,l,h);框体轴沿 yaw_nus 车头)。"""
    pts = np.asarray(points_global, dtype=np.float64)[:, :3]
    w, l, h = size
    c = np.asarray(translation, dtype=np.float64)
    rel = pts - c
    cy, sy = np.cos(yaw_nus), np.sin(yaw_nus)
    u = rel[:, 0] * cy + rel[:, 1] * sy  # 沿车头
    v = -rel[:, 0] * sy + rel[:, 1] * cy  # 右向
    inside = (np.abs(u) <= l / 2) & (np.abs(v) <= w / 2) & (np.abs(rel[:, 2]) <= h / 2)
    return int(inside.sum())


def _tok(kind: str, *idx: int) -> str:
    """确定性 token(devkit 只要求表内唯一,不要求格式)。"""
    return "ad" + kind + "".join(f"{v:x}" for v in idx)


def _quat(yaw_nus: float) -> list[float]:
    return list(g.yaw_to_quat(yaw_nus))


def _quat_of(q: tuple[float, float, float, float]) -> list[float]:
    """已归一化的四元数 → 落盘列表(w,x,y,z)。与 `_quat(yaw)` 分开:**ego_pose 走这条**,
    它必须携带悬架俯仰,不能再经 `yaw_to_quat` 拍平(见 `NusSample.ego_rotation_nus`)。"""
    return list(g.quat_normalize(q))


def camera_k(cam: str, rig: str = NUS_RIG_DEFAULT) -> tuple[float, float, float]:
    """该 rig 的逐通道 `(fx, cx, cy)`。给"按声明口径把世界点投回画幅"的探针用
    (判据 ⑦⑧ 的 in-FOV 诊断:`bin/rig_check.py`)。`camera_intrinsic()` 是它的落盘形式。"""
    return _intrinsics(rig)[cam]


def camera_intrinsic(cam: str, rig: str = NUS_RIG_DEFAULT) -> list[list[float]]:
    """nuScenes devkit 口径内参(1600×900)。

    `[[fx,0,cx],[0,fy,cy],[0,0,1]]`,fx=fy(两个 rig 都如此)。

    `rig="nuscenes"`(默认)= 逐通道**官方 n015 实测 K**,主点来自官方值而**不是** `(w−1)/2`
    —— 理由见 `NUS_CAMERA_INTRINSICS` 的注释:corner 裁决描述的是 CARLA 渲染栅格,
    官方 K 是真实相机装配公差,两者角色不同。历史实现就地写 `800/tan(45°)` 且六路共用一张 K,
    与官方逐通道值差 1.57×。

    `rig="wide"` = 由声明 FoV **反推**的 K、主点取 corner(合成数据自洽),见 `_wide_intrinsics`。
    """
    fx, cx, cy = _intrinsics(rig)[cam]
    return [[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]]


def write_mini_dataset(
    dataroot: str | Path,
    version: str,
    scenes: dict[str, list[NusSample]],
    log_name: str = "ad_log",
    rig: str = NUS_RIG_DEFAULT,
) -> Path:
    """全量落盘:14 张表 + map PNG;返回 dataroot。

    scenes = {场景名: samples}——场景名必须覆盖 devkit val 名单(如 mini_val 的
    scene-0103/scene-0916),否则 auto3dlabel generate_review_queue 遍历会 KeyError。

    `rig` 选相机口径(`nuscenes` 官方 / `wide` 自定义,见 `NUS_RIGS`):相机
    `calibrated_sensor` 的平移/四元数/内参三者**同时**取自该 rig 的表。**必须与采集时
    spawn 用的 rig 一致** —— 否则就回到"表对了、图错了"的老坑(§P-M.7)。
    """
    root = Path(dataroot)
    table_dir = root / version
    table_dir.mkdir(parents=True, exist_ok=True)
    # 相机标定表按 rig 取(未知 rig 在此即抛,不落半套表)
    _calibs = camera_calibs(rig)
    all_samples = [s for ss in scenes.values() for s in ss]
    n = len(all_samples)

    # 固定 token 布局:log/visibility/map 各一;scene 每场景;sample/ego_pose 每 sample
    log_token = _tok("log", 0)
    vis_token = _tok("vis", 0)
    map_token = _tok("map", 0)
    # sample/ego token:两级索引(场景 idx, 样本 idx);sample_scene 记录每 sample 归属场景
    sample_tokens: list[str] = []
    ego_tokens: list[str] = []
    sample_scene: list[str] = []
    scene_token_of: dict[str, str] = {}
    for si, (name, ss) in enumerate(scenes.items()):
        scene_token_of[name] = _tok("scene", si)
        for i in range(len(ss)):
            sample_tokens.append(_tok("sample", si, i))
            ego_tokens.append(_tok("ego", si, i))
            sample_scene.append(scene_token_of[name])
    # instance:按 annotations 的 instance_token 去重保序(跨 sample/场景稳定)
    inst_tokens: list[str] = []
    inst_category: dict[str, str] = {}
    for s in all_samples:
        for a in s.annotations:
            it = a["instance_token"]
            if it not in inst_category:
                inst_category[it] = a["category"]
                inst_tokens.append(it)

    # 1) category(23 类全表,名称照 auto3dlabel NUSCENES_CATEGORY_MAP 键)
    categories = [
        "animal",
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "human.pedestrian.construction_worker",
        "human.pedestrian.personal_mobility",
        "human.pedestrian.police_officer",
        "human.pedestrian.stroller",
        "human.pedestrian.wheelchair",
        "movable_object.barrier",
        "movable_object.debris",
        "movable_object.pushable_pullable",
        "movable_object.trafficcone",
        "static_object.bicycle_rack",
        "vehicle.bicycle",
        "vehicle.bus.bendy",
        "vehicle.bus.rigid",
        "vehicle.car",
        "vehicle.construction",
        "vehicle.emergency.ambulance",
        "vehicle.emergency.police",
        "vehicle.motorcycle",
        "vehicle.trailer",
        "vehicle.truck",
    ]
    category_table = [
        {"token": _tok("cat", i), "name": name, "description": ""} for i, name in enumerate(categories)
    ]
    cat_token = {c["name"]: c["token"] for c in category_table}

    # 2) attribute:空(无人消费)
    attribute_table: list[dict] = []

    # 3) visibility:单条
    visibility_table = [{"token": vis_token, "level": "v0-80", "description": "0-80%"}]

    # 4) instance
    instance_table = [
        {
            "token": it,
            "category_token": cat_token[NUS_NAME_TO_CATEGORY[inst_category[it]]],
            "nbr_annotations": sum(
                1 for s in all_samples for a in s.annotations if a["instance_token"] == it
            ),
            "first_annotation_token": "",
            "last_annotation_token": "",
        }
        for it in inst_tokens
    ]

    # 5) sensor:LIDAR_TOP + 5 雷达 + 6 相机
    # 固定索引布局(sensor/calib/sample_data 三表共用):0 LiDAR、1..5 雷达、6..11 相机
    sensor_table = (
        [{"token": _tok("sens", 0), "channel": "LIDAR_TOP", "modality": "lidar"}]
        + [
            {"token": _tok("sens", i + 1), "channel": ch, "modality": "radar"}
            for i, ch in enumerate(NUS_RADAR_CHANNELS)
        ]
        + [
            {"token": _tok("sens", i + 6), "channel": cam, "modality": "camera"}
            for i, cam in enumerate(NUS_CAMERAS)
        ]
    )

    # 6) calibrated_sensor(LIDAR_TOP + 5 雷达 + 6 相机)
    # 雷达平移/旋转 = NUS_RADAR_OFFSETS 原值(nus 系,y 左)——与 CARLA 采集侧
    # 的镜像(y 取负、yaw = −az_nus)是同一物理挂点的两套表达,devkit 变换链
    # p_g = R_ego(R_rad·p_s + t) + t_ego 直接吃 nus 原值。
    calib_table = (
        [
            {
                "token": _tok("calib", 0),
                "sensor_token": _tok("sens", 0),
                "translation": list(all_samples[0].calib_lidar[0]),
                "rotation": list(all_samples[0].calib_lidar[1]),
                "camera_intrinsic": [[0.0] * 3] * 3,
            }
        ]
        + [
            {
                "token": _tok("calib", i + 1),
                "sensor_token": _tok("sens", i + 1),
                "translation": list(NUS_RADAR_OFFSETS[ch][0]),
                "rotation": _quat(NUS_RADAR_OFFSETS[ch][1]),
                "camera_intrinsic": [[0.0] * 3] * 3,
            }
            for i, ch in enumerate(NUS_RADAR_CHANNELS)
        ]
        + [
            {
                "token": _tok("calib", i + 6),
                "sensor_token": _tok("sens", i + 6),
                "translation": list(_calibs[cam][0]),
                "rotation": list(_calibs[cam][1]),
                "camera_intrinsic": camera_intrinsic(cam, rig),
            }
            for i, cam in enumerate(NUS_CAMERAS)
        ]
    )

    # 7) ego_pose
    ego_table = [
        {
            "token": ego_tokens[i],
            "translation": list(all_samples[i].ego_translation),
            "rotation": _quat_of(all_samples[i].ego_rotation_nus),
            "timestamp": all_samples[i].timestamp,
        }
        for i in range(n)
    ]

    # 8) log
    log_table = [
        {
            "token": log_token,
            "logfile": log_name,
            "vehicle": "ad_ego",
            "date_captured": "2026-09-07",
            "location": "carla_town10",
        }
    ]

    # 9) scene
    scene_table = [
        {
            "token": scene_token_of[name],
            "name": name,
            "description": "AutoDriveData 合成迷你场景",
            "log_token": log_token,
            "nbr_samples": len(ss),
            "first_sample_token": _tok("sample", si, 0),
            "last_sample_token": _tok("sample", si, len(ss) - 1),
        }
        for si, (name, ss) in enumerate(scenes.items())
    ]

    # 10) sample(devkit 反查回填 data/anns,此处空);prev/next 链不跨场景
    sample_table = [
        {
            "token": sample_tokens[i],
            "timestamp": all_samples[i].timestamp,
            "prev": sample_tokens[i - 1] if i > 0 and sample_scene[i - 1] == sample_scene[i] else "",
            "next": sample_tokens[i + 1] if i < n - 1 and sample_scene[i + 1] == sample_scene[i] else "",
            "scene_token": sample_scene[i],
        }
        for i in range(n)
    ]

    # 11) sample_data:每 sample 12 条 keyframe(1 LIDAR + 5 雷达 + 6 相机)
    # token 布局 i*12+j:j=0 LiDAR、j=1..5 雷达、j=6..11 相机(与 sensor/calib 对齐)
    sample_data_table: list[dict] = []
    for i, s in enumerate(all_samples):
        sample_data_table.append(
            {
                "token": _tok("sd", i * 12),
                "sample_token": sample_tokens[i],
                "ego_pose_token": ego_tokens[i],
                "calibrated_sensor_token": _tok("calib", 0),
                "timestamp": s.timestamp,
                "fileformat": "bin",
                "is_key_frame": True,
                "height": 0,
                "width": 0,
                "filename": s.lidar_filename,
                "prev": "",
                "next": "",
                "sensor_token": _tok("sens", 0),
                "channel": "LIDAR_TOP",
                "modality": "lidar",
            }
        )
        for j, ch in enumerate(NUS_RADAR_CHANNELS):
            sample_data_table.append(
                {
                    "token": _tok("sd", i * 12 + j + 1),
                    "sample_token": sample_tokens[i],
                    "ego_pose_token": ego_tokens[i],
                    "calibrated_sensor_token": _tok("calib", j + 1),
                    "timestamp": s.timestamp,
                    "fileformat": "pcd",
                    "is_key_frame": True,
                    "height": 0,
                    "width": 0,
                    "filename": s.radar_filenames.get(ch, ""),
                    "prev": "",
                    "next": "",
                    "sensor_token": _tok("sens", j + 1),
                    "channel": ch,
                    "modality": "radar",
                }
            )
        for j, cam in enumerate(NUS_CAMERAS):
            sample_data_table.append(
                {
                    "token": _tok("sd", i * 12 + j + 6),
                    "sample_token": sample_tokens[i],
                    "ego_pose_token": ego_tokens[i],
                    "calibrated_sensor_token": _tok("calib", j + 6),
                    "timestamp": s.timestamp,
                    "fileformat": "png",
                    "is_key_frame": True,
                    "height": 900,
                    "width": 1600,
                    "filename": s.camera_filenames[cam],
                    "prev": "",
                    "next": "",
                    "sensor_token": _tok("sens", j + 6),
                    "channel": cam,
                    "modality": "camera",
                }
            )

    # 12) sample_annotation(category 经 instance→category_token 链,devkit 装饰 category_name)
    ann_table: list[dict] = []
    for i, s in enumerate(all_samples):
        for a in s.annotations:
            ann_table.append(
                {
                    "token": _tok("ann", len(ann_table)),
                    "sample_token": sample_tokens[i],
                    "instance_token": a["instance_token"],
                    "visibility_token": vis_token,
                    "attribute_tokens": [],
                    "num_lidar_pts": a["num_lidar_pts"],
                    "num_radar_pts": a.get("num_radar_pts", 0),
                    "translation": list(a["translation"]),
                    "size": list(a["size"]),
                    "rotation": _quat(a["yaw_nus"]),
                    "prev": "",
                    "next": "",
                }
            )

    # 13) map:非空 + filename 文件必须存在(写 16×16 黑图)
    map_rel = "maps/ad_map.png"
    map_png = root / map_rel
    map_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(map_png), np.zeros((16, 16, 3), dtype=np.uint8))
    map_table = [
        {
            "token": map_token,
            "log_tokens": [log_token],
            "category": "semantic_prior",
            "filename": map_rel,
        }
    ]

    tables = {
        "category": category_table,
        "attribute": attribute_table,
        "visibility": visibility_table,
        "instance": instance_table,
        "sensor": sensor_table,
        "calibrated_sensor": calib_table,
        "ego_pose": ego_table,
        "log": log_table,
        "scene": scene_table,
        "sample": sample_table,
        "sample_data": sample_data_table,
        "sample_annotation": ann_table,
        "map": map_table,
    }
    for name, rows in tables.items():
        (table_dir / f"{name}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return root

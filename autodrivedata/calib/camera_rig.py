"""环视相机 rig 的**唯一来源**:官方 nuScenes 6 相机标定 → CARLA 采集口径。

**为什么单独成模块(不留在 `autodrivedata/sim/collect_surround.py`)**:同一张表有三个消费面 ——
采集器(spawn)、实时可视化(`live_common.rig_spec`)、导出器(`export/nuscenes`)。
此前 `SURROUND_CAMS`(bin)与 `NUS_CAMERA_CALIBS`(autodrivedata)各写一份,两者
**对不上**(见下),而 `autodrivedata/sim/collect_surround_micro.py` 还抄了第三份。单一来源消除这个面。

## 两张表的关系(踩坑记录)

`NUS_CAMERA_CALIBS` = 官方 `calibrated_sensor` 原值:**nuScenes 全局系**(x 前 / y 左 / z 上,
**原点 = 后轴中心在地面**)。`bin/` 侧 spawn 用的是 **CARLA actor 系**(x 前 / y 右 / z 上,
**原点 = 车身长度中点**)。两者差**两个**基变换:

1. **y 翻号**(nus y 左 → CARLA y 右)。翻转必须同时作用在平移和姿态上,而历史实现只翻了
   平移、把偏航抄了个正数 ⇒ **四个侧/后相机左右镜像**、前后相机因近自逆而"看起来对":

| 相机 | 官方 az_nus | 正确 yaw_carla = −az_nus | 历史 `SURROUND_CAMS` | 差 |
|---|---|---|---|---|
| CAM_FRONT | +0.321° | −0.321° | 0.0 | 0.32° |
| CAM_FRONT_LEFT | +55.165° | **−55.165°** | **+55.0** | **110.3°** |
| CAM_FRONT_RIGHT | −56.402° | **+56.402°** | **−55.0** | **111.4°** |
| CAM_BACK | +179.855° | −179.855° | 180.0 | 0.15° |
| CAM_BACK_LEFT | +108.595° | **−108.595°** | **+108.6** | **217.2°** |
| CAM_BACK_RIGHT | −110.789° | **+110.789°** | **−110.8** | **221.6°** |

2. **x 平移 `NUS_EGO_ORIGIN_X`**(见下节)。这一条曾**整表漏掉**——包括同一个坑的第二代
   修法("全传感器渲染与声明同源")也只对齐了姿态与 y,**两套系的原点差从未写进代码**。

两条都是"图像仍能渲染、GT 框仍能画"的静默错位:前者毁掉"第 i 路图与它学到的语义"的对应,
后者让整组传感器相对自车偏 1.26 m。

## ⚠️ 两套系的原点不同(2026-09-23,Plan2 §P-M.10)

**nuScenes 的 ego 原点 = 后轴中心在地面;CARLA 车辆 actor 的原点 = 车身长度中点。**
官方标定表是按前者写的,历史实现把 x 原样当成了后者 ⇒ 6 相机 + 6 雷达 + LiDAR 整体前移
**1.2563 m**。症状不是"图出不来",而是:实测 `CAM_BACK` 画幅里 **42.999%** 是自身车体
(挂点被推到后轴**之前**,后视相机朝后拍时正对着自己车顶),`RADAR_FRONT` 悬在车头前方 1.56 m。

修法 = **把换算显式写出来**,`geometry.nus_mount_to_carla`(x 加原点差 + y 翻号)是**唯一**换算点;
`nus_camera_rig()` 与 `export/nuscenes` 的雷达/LiDAR 挂点都吃它。常量值与其**实测来源**
(双偏航自解 + 整车自洽校验)写在 `geometry.NUS_EGO_ORIGIN_X` 头注,并在
`autodrivedata/calib/verify_nus_calib.py` 判据⑩ 于实机上**复测比对**。

## 姿态为什么不能只留 yaw

官方四元数是 **6DoF**,归一化后展开含非零 pitch/roll(实测最大 |pitch| 0.96°、
|roll| 0.62°)。挂点姿态直接用 yaw-only 会带进 ~1° 的系统性指向误差 ——
在 30 m 处约 0.5 m 横向偏移,对时序建图的位姿链是**可观测量级**的偏差。
`NUS_CAMERA_RIG` 因此给三元组 (pitch, yaw, roll)(**CARLA 顺序**),
由 `nus_camera_rotation_to_carla` 从官方四元数导出。

⚠️ 本表的四元数**不是单位长度**——但**来源是"手抄舍入",不是官方值**:官方 mini 集 120 条
`calibrated_sensor` 的 |q| 实测全为 1.000000000000(最大偏离 2.22e-16,IEEE754 正常舍入);
这里把官方值抄成 4 位小数(如 `CAM_FRONT_LEFT` 的 |q|−1 = −5.04e-05)才偏离单位。导出前
必须归一化(闭式解假定 |q| = 1),归一化本身无害且必要。

## wide rig(2026-09-23,Plan2 §P-M.8)——**自定义口径,不是官方标定**

用户口径:前三个 55° FoV、左后/右后 110°、后 180°,挂点后移到车尾、**画幅里不含自身车体像素**。
它**不是** nuScenes 官方值(官方 devkit K 反推为 64.31–64.96°×5 + 89.34°;官网口径是五路 70° +
后 110°),故与官方口径**并存**而非替换,由 `collect_nus --rig {nuscenes,wide}` 选。

三条推导理由(**不写下来下一个人会来"修正"**):

1. **零车体像素的解析上限**。挂点落在车身最后点之后时,画幅内每条射线的方位角都必须是**向后**
   的(cos α ≤ 0,即 α∈[90°,270°]),否则该射线必然指向前方的车体。于是

       hfov ≤ 2·min(az − 90°, 270° − az)

   官方轴 az 108.595°/249.211° ⇒ 上限仅 **37.19°/41.58°**,而用户要 110° ⇒ **必须同时换轴**。
   等分后半球得 145/180/215(±55 = 90..200 / 120..240 / 160..270,合起来正好盖满 [90°,270°],
   相邻重叠 80°),此时 110° 才成立。
2. **180° 在针孔模型下 K 奇异**。`fx = (W/2)/tan(hfov/2)` 在 180° 时 `tan(90°) = 1.63e16`
   ⇒ `fx = 4.9e-14 ≈ 0`,K 退化成 `[0,0,cx]/[0,0,cy]/[0,0,1]`,**不可逆**,投影 `u = fx·x/z + cx`
   恒等于主点(整幅塌成一点);CARLA/UE 的投影矩阵同样以 `tan(fov/2)` 当焦距因子,同理退化。
   故 **CAM_BACK 封顶 120°**(fx = 462.0 px);120° 与两路 110° 后侧相机合并仍完整覆盖后半球。
3. **4.7 cm 余量**。车身最后点(实测包围盒)在 x = −1.8527;挂点取
   `NUS_WIDE_REAR_X_CARLA = −1.90` 落在它之后,给 az=90° 的掠射线留余量 ——
   若恰好落在 −1.8527,该族射线会**擦过**车尾面。(该常量是 **CARLA 车体系**的值;
   落盘表在 nus 系,差一个 `NUS_EGO_ORIGIN_X`,见上一节。)

前三个**一动不动**(官方挂点 + 官方方位角):55° 帧 ⊂ 官方帧,而官方帧已实测无车体像素。
价格是覆盖:3×55° = 165° < 180°,前半球必然留 15.16° 盲区(见 Plan2 §P-M.8 的覆盖表)。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from autodrivedata.utils import geometry as g
from autodrivedata.utils.geometry import NUS_EGO_ORIGIN_X

# 官方 nuScenes 6 相机 calibrated_sensor(translation 米, rotation 四元数 w,x,y,z)——
# 照 nuscenes_mini 实测(每通道取第一条记录)。**nusScenes 全局系**(y 左)。
# 保留在此模块:导出器(`export/nuscenes.py`)与采集器共用同一份常量。
NUS_CAMERA_CALIBS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {
    "CAM_FRONT": ((1.7008, 0.0159, 1.5110), (0.4998, -0.5030, 0.4998, -0.4974)),
    "CAM_FRONT_LEFT": ((1.5239, 0.4946, 1.5093), (0.6757, -0.6736, 0.2121, -0.2112)),
    "CAM_FRONT_RIGHT": ((1.5508, -0.4934, 1.4957), (0.2060, -0.2027, 0.6825, -0.6714)),
    "CAM_BACK": ((0.0283, 0.0035, 1.5791), (0.5038, -0.4974, -0.4942, 0.5045)),
    "CAM_BACK_LEFT": ((1.0357, 0.4848, 1.5910), (0.6924, -0.7032, -0.1165, 0.1120)),
    "CAM_BACK_RIGHT": ((1.0149, -0.4806, 1.5624), (0.1228, -0.1324, -0.7004, 0.6905)),
}

NUS_CAMERAS: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def nus_camera_rig(
    calibs: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] | None = None,
    *,
    origin_x: float = NUS_EGO_ORIGIN_X,
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """nuScenes 侧标定 → CARLA 侧 rig:`{相机名: (挂点 (x,y,z) 米, 姿态 (pitch,yaw,roll) 度)}`。

    平移走 `geometry.nus_mount_to_carla`(**x 加 ego 系原点差 + y 翻号**,唯一换算点);
    姿态走 `geometry.nus_camera_rotation_to_carla` → `rotation_matrix_to_carla`(弧度转度)。

    `origin_x` 见 `nus_mount_to_carla` 的说明 —— **换一辆车就要换它**;复现第三方数据集时必须用
    对方的值(见下方 `NUS_CAMERA_RIG_NUCARLA`)。
    """
    out: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
    for name, (t_nus, q_nus) in (calibs or NUS_CAMERA_CALIBS).items():
        r_carla = g.nus_camera_rotation_to_carla(q_nus)
        pitch, yaw, roll = g.rotation_matrix_to_carla(r_carla)
        out[name] = (
            g.nus_mount_to_carla(t_nus, origin_x),
            (math.degrees(pitch), math.degrees(yaw), math.degrees(roll)),
        )
    return out


# 官方 rig 的 CARLA 侧展开表(由上方函数在导入时导出;改官方标定即自动跟随)
NUS_CAMERA_RIG: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = nus_camera_rig()

# ---------------------------------------------------------------- nuCarla rig(第三方数据集口径)
#
# **用途**:nuCarla(arXiv 2511.13744,CARLA **0.9.16** / UE4 —— 与本项目逐位同版本)发布了四个
# BEV 检测模型的预训练权重(免训 300 GPU-h)。要在**我们的图**上跑泛化,就必须用**它的 rig** 采我们的图
# —— §P-L.1 红线:「rig 必须与权重训练数据一致」(错配代价实测 122711 vs 135989 px)。
#
# **来源**:`nuCarla` 仓库 `data` 分支 `config.yaml` 的 sensors 段 **逐字抄录**,
# **不拿我们的 `NUS_CAMERA_CALIBS` 代抄** —— 两者不等价,差在三处(2026-10-01 实测):
#
# | # | 差异 | 量 |
# |---|---|---|
# | ① | **四元数精度**:他们的 config 是**全精度官方值**;我们的 `NUS_CAMERA_CALIBS` 是**手抄 4 位小数**(|Δq| 3–5e-05,即本模块头注记的那次舍入) | 等价,但**不是同一份数** |
# | ② | **CAM_FRONT 的 x 比官方大 0.2 m**(其余五路与官方**逐位相同**) | 0.2 m |
# | ③ | **ego 原点常量**:他们的 `center_to_wheelbase = 1.317`(config 注明 **nissan micra 专用**);我们的 `NUS_EGO_ORIGIN_X = 1.2563` 是**实测 a2 后轴** | 0.0607 m |
#
# ⚠️ **最大的一处差是 FOV,不在表里**:
#
# | | 五路 | **CAM_BACK** |
# |---|---|---|
# | nuCarla | **65.0°** 统一 | **65.0°** |
# | 官方 n015 / 我们 | 64.31–64.96° | **89.34°** |
#
# **CAM_BACK 差 24.3°** —— 内参由 FOV 导出,**BEV 模型的几何直接依赖它**
# ⇒ 拿我们的 CAM_BACK 喂它的模型必然是错的。故 FOV 单独一张表。
NUCARLA_EGO_ORIGIN_X: float = -1.3170

#: nuCarla 六路**统一 65.0°**(与官方逐通道口径不同;见上表)。
#: ⚠️ **官方 FOV 表不在本模块** —— 它在 `gt/export/nuscenes.NUS_CAMERA_FOV`(采集/导出共用
#: 的历史落点)。nuCarla 这张落在这里,是因为它只服务「复现别人的 rig」这一件事,
#: 与挂点表同生共死。比对两套口径时**从两处取**,别以为本模块有全部。
NUCARLA_CAMERA_FOV: dict[str, float] = dict.fromkeys(NUS_CAMERAS, 65.0)

#: nuCarla 的 nuScenes 侧标定(`(平移 nus 系, 四元数)`),**逐字抄自**其 `data` 分支 config.yaml。
NUCARLA_CAMERA_CALIBS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {
    "CAM_FRONT": (
        (1.90079118954, 0.0159456324149, 1.51095763913),
        (0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755),
    ),
    "CAM_BACK": (
        (0.0283260309358, 0.00345136761476, 1.57910346144),
        (0.5037872666382278, -0.49740249788611096, -0.4941850223835201, 0.5045496097725578),
    ),
    "CAM_BACK_LEFT": (
        (1.03569100218, 0.484795032713, 1.59097014818),
        (0.6924185592174665, -0.7031619420114925, -0.11648342771943819, 0.11203317912370753),
    ),
    "CAM_FRONT_LEFT": (
        (1.52387798135, 0.494631336551, 1.50932822144),
        (0.6757265034669446, -0.6736266522251881, 0.21214015046209478, -0.21122827103904068),
    ),
    "CAM_FRONT_RIGHT": (
        (1.5508477543, -0.493404796419, 1.49574800619),
        (0.2060347966337182, -0.2026940577919598, 0.6824507824531167, -0.6713610884174485),
    ),
    "CAM_BACK_RIGHT": (
        (1.0148780988, -0.480568219723, 1.56239545128),
        (0.12280980120078765, -0.132400842670559, -0.7004305821388234, 0.690496031265798),
    ),
}


def _nucarla_camera_rotation(q_nus: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """nuCarla `sensors.py` 的姿态换算 —— **逐行复现,含它的约定**。

    ```
    q_new = yaw_q(−90°) · roll_q(−90°) · q_nus⁻¹      (pyquaternion)
    yaw, pitch, roll = q_new.yaw_pitch_roll           (右手 Z-Y-X)
    carla.Rotation(pitch=..., yaw=..., roll=...)
    ```

    ★ **不能换成 `nus_camera_rotation_to_carla`** —— 两条链**不等价**,差在 pitch/roll:

    | 通道 | 本函数 (pitch, yaw, roll) | `NUS_CAMERA_RIG` 的链 |
    |---|---|---|
    | CAM_BACK | (−0.2292, −179.8612, **0.9594**) | (**0.9594**, −179.8574, −0.2292) |

    **yaw 六路逐位一致,pitch↔roll 互换且一侧反号** ——
    因为 `sensors.py` 把 **pyquaternion 的右手 Z-Y-X** 角**直接喂进 `carla.Rotation`(UE 左手)**,
    而我们的链走的是**经过验证的** UE 口径(§P-M.10 ③ 实测:正确是 `Rz(−yaw)·Ry(−pitch)·Rx(+roll)`,
    `carla_rotation_matrix` 是左手阵,因子分解的右手假设会把 pitch/roll 弄反 —— 那条当时**极隐蔽**,
    因为 yaw 项照样对得上)。

    **为什么仍要用它**:模型学的是**他们渲染出来的图**。他们的相机若有 ~0.4° 的俯仰偏差,
    我们必须**原样复现**,否则凭空多一个域差 —— 这正是 §P-L.1「rig 必须与权重训练数据一致」的极端形态:
    **一致到"连对方的约定偏差也要一致"**。
    """
    yaw_q = _quat_axis_angle((0.0, 0.0, 1.0), -90.0)
    roll_q = _quat_axis_angle((1.0, 0.0, 0.0), -90.0)
    q_new = _quat_mul(_quat_mul(yaw_q, roll_q), _quat_conj(np.asarray(q_nus, dtype=float)))
    w, x, y, z = q_new
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * x - y * z))))
    roll = math.atan2(2 * (w * y + x * z), 1 - 2 * (x * x + y * y))
    return (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(w,x,y,z) 哈达玛积 —— 只在本模块复现 nuCarla 的数学,不外露。"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_axis_angle(axis: tuple[float, float, float], deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    h = math.radians(deg) / 2.0
    return np.array([math.cos(h), *(a * math.sin(h))])


#: nuCarla rig 的 CARLA 侧展开表。**平移**走 `nus_mount_to_carla`(同一个唯一换算点,只是换
#: 原点常量);**姿态**走 `_nucarla_camera_rotation`(见它为什么不能复用我们的链)。
NUS_CAMERA_RIG_NUCARLA: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    ch: (g.nus_mount_to_carla(t_nus, NUCARLA_EGO_ORIGIN_X), _nucarla_camera_rotation(q_nus))
    for ch, (t_nus, q_nus) in NUCARLA_CAMERA_CALIBS.items()
}

# 相机名 → 相对 ego 的偏航(度),CARLA 口径。**历史别名**:`collect_surround.SURROUND_CAMS`
# 的等价物,供只关心偏航的调用方(`live_common.rig_spec`)直接取用。
NUS_CAMERA_YAW: dict[str, float] = {name: rig[1][1] for name, rig in NUS_CAMERA_RIG.items()}

# 六视角图的**画布行序**(2 行 × 3 列)—— **所有六视角拼图的唯一来源**。
#
# 顺序按**方位绕车走一圈**:左前 → 前 → 右前 / 右后 → 后 → 左后。
# **不要改回 `NUS_CAMERAS` / `NUS_CAMERA_RIG` 的字典序**:那是
# `F, FL, FR, B, BL, BR` —— 第一行的左前/前就是反的,第二行按 3 列切出来更是
# `B, BL, BR`,左右与地理直觉完全相反(这条理由最早写在 `live_studio.GRID_ROWS` 上)。
#
# 为什么放在本模块:产出六视角图的地方**都要摆同样的六格**(实时拼图 / 六视角实拍 /
# 布局对照 / 预测回投 / grid6 实时流 / 标定 A/B),各写一份必然漂移 —— 2026-09-28 实测
# 六处里**三处是错的**(一份第二行左右反、一份 1×6 字母序、一份干脆把后三路排到了第一行)。
# 本模块是"环视相机 rig 的唯一来源",方位序天然属于这里,且**纯值**(不 import carla),
# 故离线脚本也能引用。
CAMERA_GRID_ROWS: tuple[tuple[str, ...], ...] = (
    ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"),
    ("CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"),
)


def camera_grid_rows(cams: Iterable[str] | None = None) -> list[list[str]]:
    """画布行序的相机名(2×3 分组);给了 `cams` 就只保留其中出现的。

    **六视角拼图的唯一取序入口** —— 别自己 `sorted(...)`,也别直接遍历 `NUS_CAMERA_RIG`
    (两者的顺序都不是画布顺序,理由见上方 `CAMERA_GRID_ROWS` 注释)。
    """
    keep = None if cams is None else set(cams)
    return [[c for c in row if keep is None or c in keep] for row in CAMERA_GRID_ROWS]


def camera_grid_order(cams: Iterable[str] | None = None) -> list[str]:
    """`camera_grid_rows` 的**展平**版(喂 `compose_grid` 这类一维接口用)。"""
    return [c for row in camera_grid_rows(cams) for c in row]


# ---------------------------------------------------------------- wide rig(见模块头注)

# 后三路挂点的纵向位置(米,**CARLA actor 系** —— 后缀点明系别,别再当 nus 系读)。
# 它描述的是"挂点相对**车体**在哪"(设计理由就是车尾余量),故留在这个系;
# 写进 nuScenes 侧 `calibrated_sensor` 时经 `-_CARLA` 那一侧换算成相对后轴的值。
# 车身最后点实测 −1.8527 ⇒ 本值落在车尾之后 4.7 cm。
NUS_WIDE_REAR_X_CARLA: float = -1.9000

# 后三路的目标**轴方位角**(度,nuScenes 口径:0=车头,+左)。等分后半球 145/180/215;
# 末项写成 −145° 而非 215° 是为了让 `yaw_carla = −az_nus` 直接给出 (pitch, yaw, roll) 三元组
# 里的 yaw(二者是同一个方位)。前三个**不在表内** = 沿用官方方位角。
NUS_WIDE_CAMERA_AZ: dict[str, float] = {
    "CAM_BACK_LEFT": 145.0,
    "CAM_BACK": 180.0,
    "CAM_BACK_RIGHT": -145.0,  # ≡ 真方位角 215°
}

# 用户口径的逐通道水平 FoV(度)。口径由 `autodrivedata/gt/export/nuscenes.py` 的
# `NUS_WIDE_CAMERA_INTRINSICS` 反推(K 由渲染反推,不抄官方)。
NUS_WIDE_CAMERA_FOV: dict[str, float] = {
    "CAM_FRONT": 55.0,
    "CAM_FRONT_LEFT": 55.0,
    "CAM_FRONT_RIGHT": 55.0,
    "CAM_BACK": 120.0,  # 封顶 120°:180° 使针孔 K 奇异(见模块头注 ②)
    "CAM_BACK_LEFT": 110.0,
    "CAM_BACK_RIGHT": 110.0,
}


def max_hfov_no_ego(az_nus_deg: float) -> float:
    """零车体像素的**解析上限** `2·min(az−90°, 270°−az)`(度);自变量取 nuScenes 方位角。

    挂点落在车身最后点之后时,画幅内每条射线必须有向后分量(α∈[90°,270°])才有希望避开自身车体。
    这是**充分条件**(挂点若在车身横向之外、或够高够远,实际上限更大),但它是闭式的、不依赖
    包围盒近似,故用作 rig 设计的硬闸与回归钉(`tests/test_nuscenes_calib_consistency.py`)。

    ⚠️ **返回值在前半球(az∈(270°,360°)∪[0°,90°))为负**:轴朝前的相机无论多窄都拍得到自身
    车体(单条射线就够),故"上限为负"= **无解**,不是笔误、也不该被 clamp 成 0 —— 保留符号
    才能让 `hfov <= bound` 这条判据对前半球相机一律判否。`az = 90°/270°` 恰好 0,`180°` 最大 180°。
    """
    az = az_nus_deg % 360.0
    return 2.0 * min(az - 90.0, 270.0 - az)


def nus_wide_camera_calibs() -> dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float, float]]
]:
    """官方 `calibrated_sensor` → wide 口径(仍是 **nuScenes 系**,故可直接落盘)。

    后三路改两处:平移 x 换 `NUS_WIDE_REAR_X_CARLA − NUS_EGO_ORIGIN_X`
    (设计常量在 CARLA 车体系、本表在 nus 系,差一个原点,见 `geometry.NUS_EGO_ORIGIN_X`);
    姿态左乘**绕世界 up 轴的方位角增量**,即 `q_wide = qz(Δ) ⊗ q_official`、
    `Δ = az_target − az_official`。因 `nus_camera_rig()` 走 `rotation_matrix_to_carla`
    解出 `R = Rz(yaw)Ry(pitch)Rx(roll)`,而左乘 `Rz` **精确保持** pitch/roll、只把 yaw
    平移 −Δ ⇒ CARLA 侧自动得到 `yaw = −az_target` 且 pitch/roll 与官方逐位相同(无需手抄第二份)。

    **本函数是 wide rig 的唯一来源**:CARLA 侧由 `nus_camera_rig(NUS_WIDE_CAMERA_CALIBS)`
    导出(nus 系)与落盘 `calibrated_sensor` 都吃它;`NUS_WIDE_CAMERA_AZ` 是"方位角
    设计表"而非第二份姿态 —— 与官方 `NUS_CAMERA_CALIBS` ↔ `NUS_CAMERA_RIG` 同构。
    """
    out: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
    for name, (t_nus, q_nus) in NUS_CAMERA_CALIBS.items():
        if name not in NUS_WIDE_CAMERA_AZ:
            out[name] = (t_nus, q_nus)
            continue
        boresight = g.quat_to_matrix(g.quat_normalize(q_nus))[:, 2]  # 相机自身系 +z = 视线轴
        az_official = math.degrees(math.atan2(boresight[1], boresight[0]))
        q_wide = g.quat_mul(
            g.yaw_to_quat(math.radians(NUS_WIDE_CAMERA_AZ[name] - az_official)),
            g.quat_normalize(q_nus),
        )
        out[name] = (
            (NUS_WIDE_REAR_X_CARLA - g.NUS_EGO_ORIGIN_X, t_nus[1], t_nus[2]),
            g.quat_normalize(q_wide),
        )
    return out


# wide rig 的 nuScenes 侧标定表(单点来源;落盘 `calibrated_sensor` 直接吃它)
NUS_WIDE_CAMERA_CALIBS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = (
    nus_wide_camera_calibs()
)

# wide rig 的 CARLA 侧展开表(由同一份 calibs 导出;改上方设计常量即自动跟随)
NUS_WIDE_CAMERA_RIG: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = (
    nus_camera_rig(NUS_WIDE_CAMERA_CALIBS)
)

NUS_WIDE_CAMERA_YAW: dict[str, float] = {name: rig[1][1] for name, rig in NUS_WIDE_CAMERA_RIG.items()}


# ---------------------------------------------------------------- 方位覆盖(纯几何)


def camera_azimuth_nus(
    name: str,
    calibs: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] | None = None,
) -> float:
    """相机光轴在 **nuScenes 全局系**的方位角(度,0 = 车头,+ = 左,范围 (−180, 180])。

    光轴取相机自身系 +z 经四元数旋转后的**水平投影**(pitch 不动方位角,故与俯仰无关)。
    这是"用户读官网看到的那个数"的口径 —— `CAL_FRONT 0.321°` 之类;把它与 FoV 混为一谈
    是本项目已踩过的坑(官方 FoV 是 64.31–64.96°×5 + 89.34°,与方位角毫无关系)。
    """
    t_q = (calibs or NUS_CAMERA_CALIBS)[name]
    boresight = g.quat_to_matrix(g.quat_normalize(t_q[1]))[:, 2]
    return math.degrees(math.atan2(boresight[1], boresight[0]))


def _sector_intervals(az_deg: float, fov_deg: float) -> list[tuple[float, float]]:
    """方位角扇区 → `[0°,360°)` 上的 1~2 段闭区间(跨 0° 时劈成两段,不靠 `%` 抹平)。"""
    lo = (az_deg - fov_deg / 2.0) % 360.0
    hi = lo + fov_deg
    return [(lo, hi)] if hi <= 360.0 else [(lo, 360.0), (0.0, hi - 360.0)]


def _merge_intervals(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for lo, hi in sorted(iv):
        if out and lo <= out[-1][1] + 1e-12:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(a, b) for a, b in out]


def _overlap_len(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b)) for lo_a, hi_a in a for lo_b, hi_b in b)


def coverage_table(
    calibs: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]],
    fov: dict[str, float],
) -> dict[str, Any]:
    """逐相机方位扇区 → **相邻重叠 / 盲区**表(纯几何,只算方位轴,不含地面距离)。

    ⚠️ 本表只说"水平方位角上盖没盖住",**不说地面能不能看见** —— 相机装在 1.5 m 高、
    55° 水平 FoV 时近场地面反而进不了画幅(见 §P-M.9 的地面可见距离)。判据分工:
    重叠/盲区看本表,"画幅里有没有自身车体"看实例分割实测(`autodrivedata/calib/rig_check.py`)。

    扇区按方位角**全局排布**求并/交,不假定相机名的字典序 = 方位序 —— 两代 rig 的
    相机名顺序都与方位顺序无关(wide 的后三路是 145/180/215,名字却叫 BACK_LEFT/BACK/BACK_RIGHT)。
    """
    cams: dict[str, Any] = {}
    iv: dict[str, list[tuple[float, float]]] = {}
    for name, f in fov.items():
        az = camera_azimuth_nus(name, calibs)
        iv[name] = _sector_intervals(az, f)
        cams[name] = {
            "az_nus_deg": az,
            "az_nus_360": az % 360.0,
            "fov_deg": float(f),
            "span": [[round(a, 4), round(b, 4)] for a, b in iv[name]],
        }
    names = list(fov)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            # 重叠**区间**也要带出去:共视探针要往重叠区正中摆目标(只给一个度数没法定位)
            segs = [
                (max(lo_a, lo_b), min(hi_a, hi_b))
                for lo_a, hi_a in iv[a]
                for lo_b, hi_b in iv[b]
                if min(hi_a, hi_b) > max(lo_a, lo_b)
            ]
            best = max(segs, key=lambda s: s[1] - s[0]) if segs else None
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "overlap_deg": _overlap_len(iv[a], iv[b]),
                    "span": [round(best[0], 4), round(best[1], 4)] if best else None,
                }
            )
    pairs.sort(key=lambda p: -p["overlap_deg"])
    merged = _merge_intervals([s for v in iv.values() for s in v])
    gaps = [
        {"lo": round(hi, 4), "hi": round(nxt, 4), "deg": round(nxt - hi, 4)}
        for (_, hi), (nxt, _) in zip(merged, merged[1:], strict=False)  # 长度差 1 是刻意的
    ]
    # 接缝(0° = 360°):两端都没盖住时那是**同一个**盲区,合成一段报出(不能报成两段)
    if merged:
        seam, tail = merged[0][0], 360.0 - merged[-1][1]
        if seam > 1e-9 and tail > 1e-9:
            gaps.append(
                {
                    "lo": round(merged[-1][1], 4),
                    "hi": round(merged[0][0], 4),
                    "deg": round(seam + tail, 4),
                    "wraps_zero": True,
                }
            )
        elif seam > 1e-9:
            gaps.insert(0, {"lo": 0.0, "hi": round(seam, 4), "deg": round(seam, 4)})
        elif tail > 1e-9:
            gaps.append({"lo": round(merged[-1][1], 4), "hi": 360.0, "deg": round(tail, 4)})
    covered = sum(b - a for a, b in merged)
    return {
        "cameras": cams,
        "pairs": pairs,
        "gaps": gaps,
        "gap_total_deg": sum(g["deg"] for g in gaps),
        "covered_deg": covered,
        "coverage_frac": covered / 360.0,
    }

"""环视相机 rig 的**唯一来源**:官方 nuScenes 6 相机标定 → CARLA 采集口径。

**为什么单独成模块(不留在 `bin/collect_surround.py`)**:同一张表有三个消费面 ——
采集器(spawn)、实时可视化(`live_common.rig_spec`)、导出器(`export/nuscenes`)。
此前 `SURROUND_CAMS`(bin)与 `NUS_CAMERA_CALIBS`(autodrivedata)各写一份,两者
**对不上**(见下),而 `bin/collect_surround_micro.py` 还抄了第三份。单一来源消除这个面。

## 两张表的关系(踩坑记录)

`NUS_CAMERA_CALIBS` = 官方 `calibrated_sensor` 原值:**nuScenes 全局系**(x 前 / y 左 / z 上)。
`bin/` 侧 spawn 用的是 **CARLA 全局系**(x 前 / y 右 / z 上)。两者只差一个 y 符号翻转,
但**翻转必须同时作用在平移和姿态上**,而历史实现只翻了平移、把偏航抄了个正数:

| 相机 | 官方 az_nus | 正确 yaw_carla = −az_nus | 历史 `SURROUND_CAMS` | 差 |
|---|---|---|---|---|
| CAM_FRONT | +0.321° | −0.321° | 0.0 | 0.32° |
| CAM_FRONT_LEFT | +55.165° | **−55.165°** | **+55.0** | **110.3°** |
| CAM_FRONT_RIGHT | −56.402° | **+56.402°** | **−55.0** | **111.4°** |
| CAM_BACK | +179.855° | −179.855° | 180.0 | 0.15° |
| CAM_BACK_LEFT | +108.595° | **−108.595°** | **+108.6** | **217.2°** |
| CAM_BACK_RIGHT | −110.789° | **+110.789°** | **−110.8** | **221.6°** |

即**四个侧/后相机左右镜像**、前/后相机因近自逆而"看起来对"。
镜像是最隐蔽的错位:图像仍能渲染、GT 框仍能画,只有"第 i 路图与它学到的语义"错位。

## 姿态为什么不能只留 yaw

官方四元数是 **6DoF**,归一化后展开含非零 pitch/roll(实测最大 |pitch| 0.96°、
|roll| 0.62°)。挂点姿态直接用 yaw-only 会带进 ~1° 的系统性指向误差 ——
在 30 m 处约 0.5 m 横向偏移,对时序建图的位姿链是**可观测量级**的偏差。
`NUS_CAMERA_RIG` 因此给三元组 (pitch, yaw, roll)(**CARLA 顺序**),
由 `nus_camera_rotation_to_carla` 从官方四元数导出。

⚠️ 官方四元数**不是单位长度**(模长 0.99994~1.00005),导出前必须归一化。
"""

from __future__ import annotations

import math

from autodrivedata import geometry as g

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
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """官方标定 → CARLA 侧 rig:`{相机名: (挂点 (x,y,z) 米, 姿态 (pitch,yaw,roll) 度)}`。

    平移只翻 y(与姿态翻转同一次基变换);姿态走
    `geometry.nus_camera_rotation_to_carla` → `rotation_matrix_to_carla`(弧度转度)。
    """
    out: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
    for name, (t_nus, q_nus) in (calibs or NUS_CAMERA_CALIBS).items():
        r_carla = g.nus_camera_rotation_to_carla(q_nus)
        pitch, yaw, roll = g.rotation_matrix_to_carla(r_carla)
        out[name] = (
            (float(t_nus[0]), float(-t_nus[1]), float(t_nus[2])),
            (math.degrees(pitch), math.degrees(yaw), math.degrees(roll)),
        )
    return out


# 官方 rig 的 CARLA 侧展开表(由上方函数在导入时导出;改官方标定即自动跟随)
NUS_CAMERA_RIG: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = nus_camera_rig()

# 相机名 → 相对 ego 的偏航(度),CARLA 口径。**历史别名**:`collect_surround.SURROUND_CAMS`
# 的等价物,供只关心偏航的调用方(`live_common.rig_spec`)直接取用。
NUS_CAMERA_YAW: dict[str, float] = {name: rig[1][1] for name, rig in NUS_CAMERA_RIG.items()}

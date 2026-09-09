"""nuScenes 迷你集生成器——照 devkit NuScenes.__init__ 契约 + auto3dlabel 消费链。

契约钉死点(2026-09-07 实测 devkit 1.2.0,详见 Plan §5.2):
- 表目录 = dataroot/<version>/,14 张表 JSON,每条记录唯一 token;不放 lidarseg/panoptic
- map 表非空,filename 指向的文件必须存在(MapMask assert);category 需 'name'、
  instance 需 category_token、sample_data 需 calibrated_sensor/sensor/modality/channel
- sample['data']/['anns'] 由 devkit 从 sample_data 的 is_key_frame 记录反查,勿手填
- LIDAR 文件 = (N,5) float32 raw(x,y,z,intensity,elongation),filename 相对 dataroot
- 坐标系:入表数据一律 nuScenes 全局系(x 前/y 左/z 上)——CARLA 数据入表前
  经 geometry.CARLA_TO_NUS 翻转(y 符号)
- 场景名必须 ∈ devkit create_splits_scenes()['mini_val'](auto3dlabel
  generate_review_queue 只遍历 val 场景)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from autodrivedata import geometry as g

NUS_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

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
    ego_yaw_nus: float
    lidar_filename: str  # 相对 dataroot
    camera_filenames: dict[str, str]  # 6 通道 → 相对 dataroot
    calib_lidar: tuple[tuple[float, float, float], float]  # (translation, yaw_nus) 传感器→ego
    calib_cameras: dict[str, tuple[tuple[float, float, float], float]]
    annotations: list[dict]  # {'category','translation','size','yaw_nus','num_lidar_pts','instance_token'}
    timestamp: int


def points_sensor_to_global_nus(
    points_sensor: np.ndarray,
    ego_translation: tuple[float, float, float],
    ego_yaw_nus: float,
    calib_translation: tuple[float, float, float],
    calib_yaw_nus: float,
) -> np.ndarray:
    """LIDAR 传感器系点 (N,3)(x 前/y 左/z 上)→ nuScenes 全局系。

    链(照 auto3dlabel boxes_sensor_to_global):p_g = R_ego @ (R_calib @ p_s + t_calib) + t_ego。
    """
    r_ego = g.quat_to_matrix(g.yaw_to_quat(ego_yaw_nus))
    r_calib = g.quat_to_matrix(g.yaw_to_quat(calib_yaw_nus))
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


def _intrinsics_1600x900_fov90() -> list[list[float]]:
    fx = 800.0 / np.tan(np.radians(45.0))
    return [[fx, 0.0, 799.5], [0.0, fx, 449.5], [0.0, 0.0, 1.0]]


def write_mini_dataset(
    dataroot: str | Path,
    version: str,
    scenes: dict[str, list[NusSample]],
    log_name: str = "ad_log",
) -> Path:
    """全量落盘:14 张表 + map PNG;返回 dataroot。

    scenes = {场景名: samples}——场景名必须覆盖 devkit val 名单(如 mini_val 的
    scene-0103/scene-0916),否则 auto3dlabel generate_review_queue 遍历会 KeyError。
    """
    root = Path(dataroot)
    table_dir = root / version
    table_dir.mkdir(parents=True, exist_ok=True)
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

    # 5) sensor:LIDAR_TOP + 6 相机
    sensor_table = [{"token": _tok("sens", 0), "channel": "LIDAR_TOP", "modality": "lidar"}] + [
        {"token": _tok("sens", i + 1), "channel": cam, "modality": "camera"}
        for i, cam in enumerate(NUS_CAMERAS)
    ]

    # 6) calibrated_sensor(LIDAR_TOP + 6 相机)
    calib_table = [
        {
            "token": _tok("calib", 0),
            "sensor_token": _tok("sens", 0),
            "translation": list(all_samples[0].calib_lidar[0]),
            "rotation": _quat(all_samples[0].calib_lidar[1]),
            "camera_intrinsic": [[0.0] * 3] * 3,
        }
    ] + [
        {
            "token": _tok("calib", i + 1),
            "sensor_token": _tok("sens", i + 1),
            "translation": list(all_samples[0].calib_cameras[cam][0]),
            "rotation": _quat(all_samples[0].calib_cameras[cam][1]),
            "camera_intrinsic": _intrinsics_1600x900_fov90(),
        }
        for i, cam in enumerate(NUS_CAMERAS)
    ]

    # 7) ego_pose
    ego_table = [
        {
            "token": ego_tokens[i],
            "translation": list(all_samples[i].ego_translation),
            "rotation": _quat(all_samples[i].ego_yaw_nus),
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

    # 11) sample_data:每 sample 7 条 keyframe(LIDAR_TOP + 6 相机)
    sample_data_table: list[dict] = []
    for i, s in enumerate(all_samples):
        sample_data_table.append(
            {
                "token": _tok("sd", i * 7),
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
        for j, cam in enumerate(NUS_CAMERAS):
            sample_data_table.append(
                {
                    "token": _tok("sd", i * 7 + j + 1),
                    "sample_token": sample_tokens[i],
                    "ego_pose_token": ego_tokens[i],
                    "calibrated_sensor_token": _tok("calib", j + 1),
                    "timestamp": s.timestamp,
                    "fileformat": "png",
                    "is_key_frame": True,
                    "height": 900,
                    "width": 1600,
                    "filename": s.camera_filenames[cam],
                    "prev": "",
                    "next": "",
                    "sensor_token": _tok("sens", j + 1),
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

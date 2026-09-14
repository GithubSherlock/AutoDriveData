"""CARLA 环视数据 → 官方 MapTR/MapQR 可直吃的 nuScenes 形状数据集(Plan.md §5.12 B 阶段数据面)。

产物(全落 `outputs/maptr_official/data/`):

| 文件 | 内容 |
|---|---|
| `carla_infos_train.pkl` / `carla_infos_val.pkl` | 官方 infos(默认 200 / 100 帧划分;**含 `annotation` 直供通道**);形状 = `{"infos": [...], "metadata": {"version": …}}`,与官方 `create_data.py` 产物同构 |
| `carla_map_anns_train.json` / `_val.json` | 官方评测 GT(`{"GTs": [...]}`,逐帧 20 点已裁窗口) |
| `maps/expansion/*.json` | nuScenes 地图 expansion 的**空桩**(见下),仅 MapTR v1 需要 |
| `meta.json` | 划分、来源、自证记录(帧号清单 + 投影对齐最大误差) |

**两个"官方数据结构"缺口的补法**(都不是取向官方仓库塞文件):

1. **训练期 GT**:官方**离线**数据集直接从 `info['annotation']` 取矢量(`get_data_info` →
   `input_dict['ann_info']` → `gen_vectorized_samples(map_annotation, ...)`),故 infos 里带
   `annotation` 即可,类名 → 折线列表的结构与我们的 `map_infos.json` 同构。
2. **MapTR v1 的构造期依赖**:v1 **没有**离线数据集,只有上线版 `CustomNuScenesLocalMapDataset`,
   其 `VectorizedLocalMap.__init__` 逐 location 构造 `NuScenesMap(dataroot, loc)` → 要
   `maps/expansion/*.json`。本脚本写一份 19 键、各层为空的**桩**(352 字节):构造期空转、
   运行时该地图对象被项目侧适配器整体替换 → 空地图**从不被查询**。这样 v1 的接入不必往
   `hdMapGitHub/` 塞文件,也不必补下载 4 个真地图(NuScenesMap 只在 `_load_layer` 里
   `json_obj[layer]` 硬取键,`canvas_edge` 须是二元列表,已实测)。

**变换口径**(与 `maptr_impl` 已验证的投影链逐项对齐,推导见 §5.12):

    sensor2lidar_rotation    = carla_rotation_matrix(sensor2ego[3:6]) @ CARLA_TO_CAM.T
    sensor2lidar_translation = sensor2ego[:3]
    lidar2ego                = 恒等(我们的 BEV 系就是 ego 系,故 sensor2lidar ≡ sensor2ego 旋转换基)
    ego2global_rotation      = r_e 的四元数 [w, x, y, z](nuscenes/pyquaternion 口径)

官方 `get_data_info` 会反解这两个字段拼出 `lidar2img`,再与 `cam_intrinsic` 相乘 —— 本脚本
`_selfcheck()` 用**官方那几行代码**逐字复算 lidar2img,和我们自己的 ego→cam 矩阵对比,
超差即报错(这是"转换器写错了"与"官方链路不同口径"的唯一分界线)。

为什么必须复用 `autodrivedata` 的常量而不是另写一套:`CARLA_TO_CAM` 含手性翻转
(det = −1),第二套实现必然在某次重构后漂移。

用法(纯 numpy,autodrivedata env 即可):
    PYTHONPATH=$PWD python bin/prepare_official_dataset.py \
        --surround outputs/surround_train --out outputs/maptr_official/data --train-frames 200
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from autodrivedata.geometry import CARLA_TO_CAM, carla_rotation_matrix
from autodrivedata.mapvec import MAPTR_CLASSES
from autodrivedata.paths import project_path

SCENE_TOKEN = "carla_town10hd_scene_000"
MAP_LOCATION = "carla_town10hd"
CAM_NAMES = tuple(
    f"CAM_{s}" for s in ("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT")
)
FRAME_DT_US = 100_000  # 10 Hz → timestamps[t] = t·0.1s


# ---------- 旋转/四元数 ----------


def quat_from_matrix(r: np.ndarray) -> list[float]:
    """旋转阵 → 四元数 [w, x, y, z](pyquaternion/nuscenes 口径)。

    用 Shepperd 分支法(取迹最大分支,数值稳定);不含符号约定的自由度由 w ≥ 0 定死
    (q 与 −q 表示同一旋转,取 w ≥ 0 使产物可 diff)。
    """
    t = np.trace(r)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return (q if q[0] >= 0 else -q).tolist()


def matrix_from_quat(q) -> np.ndarray:
    """四元数 [w, x, y, z] → 旋转阵(与 `quat_from_matrix` 互逆,单测锁定)。"""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_matrix(deg3) -> np.ndarray:
    """CARLA Rotation 三角度(度,序 **yaw, pitch, roll**)→ 旋转阵。

    我们的 infos 沿用 CARLA 的 yaw/pitch/roll 顺序,而 `carla_rotation_matrix` 吃
    (pitch, yaw, roll) —— 两处顺序不同,故在此收口,调用方不再各自拼元组。
    """
    yaw, pitch, roll = (float(v) for v in deg3)
    return carla_rotation_matrix((pitch, yaw, roll))


def sensor2lidar(sensor2ego: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """`sensor2ego` [tx,ty,tz,yaw,pitch,roll](度)→ 官方 (R_c2l, t_c2l)。"""
    return rot_matrix(sensor2ego[3:6]) @ CARLA_TO_CAM.T, np.asarray(sensor2ego[:3], dtype=np.float64)


# ---------- 官方那几行 lidar2img(逐字复刻,仅用于自证) ----------


def official_lidar2img(cam_info: dict) -> np.ndarray:
    """官方 `get_data_info` 的 lidar2img 计算(逐行照抄,含 `[3, :3]` 这个反常规写法)。"""
    lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
    lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = cam_info["cam_intrinsic"]
    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    return viewpad @ lidar2cam_rt.T


def our_ego2img(sensor2ego: list[float], intrinsic: np.ndarray) -> np.ndarray:
    """我们自己的 ego 点 → 像素矩阵(与 `mapviz`/`geometry.world_to_cam` 同链)。"""
    r_e2c = CARLA_TO_CAM @ rot_matrix(sensor2ego[3:6]).T  # ego 系向量 → KITTI 相机系
    m = np.eye(4)
    m[:3, :3] = r_e2c
    m[:3, 3] = -r_e2c @ np.asarray(sensor2ego[:3], dtype=np.float64)
    return intrinsic @ m[:3, :]


# ---------- 组装 ----------


def build_record(info: dict, root: Path, timestamp_us: int) -> dict:
    """单帧 map_infos 记录 → 官方 info 记录(字段清单来自官方 `get_data_info` + `get_ann_info`)。"""
    x, y, z, yaw, pitch, roll = info["ego2global"]
    r_e = rot_matrix([yaw, pitch, roll])
    ego_t = [float(x), float(y), float(z)]
    ego_q = quat_from_matrix(r_e)

    cams = {}
    for name in CAM_NAMES:
        c = info["cams"][name]
        r_c2l, t_c2l = sensor2lidar(c["sensor2ego"])
        cams[name] = {
            "data_path": str(root / c["data_path"]),  # 官方原样交给 cv2.imread → 必须绝对
            "cam_intrinsic": np.asarray(c["intrinsic"], dtype=np.float64),
            "sensor2lidar_rotation": r_c2l,
            "sensor2lidar_translation": t_c2l,
            "sensor2ego_rotation": quat_from_matrix(rot_matrix(c["sensor2ego"][3:6])),
            "sensor2ego_translation": np.asarray(c["sensor2ego"][:3], dtype=np.float64),
            "ego2global_rotation": ego_q,
            "ego2global_translation": ego_t,
            "timestamp": timestamp_us,
            "sample_data_token": f"{info['token']}_{name}",
        }

    return {
        "token": info["token"],
        "timestamp": timestamp_us,
        "scene_token": SCENE_TOKEN,
        "frame_idx": int(info["frame"]),
        "map_location": MAP_LOCATION,
        "ego2global_translation": ego_t,
        "ego2global_rotation": ego_q,
        "lidar2ego_translation": [0.0, 0.0, 0.0],
        "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
        "lidar_path": "",
        "sweeps": [],
        "can_bus": np.zeros(18, dtype=np.float64),
        "cams": cams,
        # 训练期 GT 的直供通道:官方**离线**数据集 `nuscenes_offlinemap_dataset.py` 的
        # `get_data_info` 把它原样塞进 `input_dict['ann_info']`,`vectormap_pipeline` 再交给
        # `VectorizedLocalMap.gen_vectorized_samples(map_annotation, ...)` —— 后者只做
        # `LineString(np.array(map_annotation[vec_class][i]))`,**不做任何坐标变换**
        # (这正是它相对上线版的全部差异:上线版查 nuScenes map expansion + 全局→ego 变换)。
        # 我们的 `annotation` 已是 ego 系、已裁窗、每线 20 点 → 结构天然同构,零适配。
        # `map_annotation[vec_class]` 按类名取值:四类键必须齐(类别由 config 决定用哪几个)。
        "annotation": {
            cls: [[[float(x), float(y)] for x, y in line] for line in lines]
            for cls, lines in info["annotation"].items()
        },
        # `get_ann_info` 需要(地图任务用不到框,给空数组即可;GT 由 vectormap/适配器供)
        "gt_boxes": np.zeros((0, 7), dtype=np.float64),
        "gt_names": np.array([], dtype="<U16"),
        "gt_velocity": np.zeros((0, 2), dtype=np.float64),
        "num_lidar_pts": np.zeros(0, dtype=np.int64),
        "num_radar_pts": np.zeros(0, dtype=np.int64),
        "valid_flag": np.zeros(0, dtype=bool),
        "ann_info": None,
    }


def gt_payload(records: list[dict], infos: list[dict]) -> dict:
    """官方 GT json:逐帧 `annotation` → `{"GTs": [{"sample_token", "vectors": [...]}]}`。

    `pts_num` 与 `type` 官方都读(`get_cls_results` 只按 `type` 过滤、按 `pts` 重采样),
    类序固定 `MAPTR_CLASSES`,越界类直接报错而不是静默丢帧。
    """
    idx = {name: i for i, name in enumerate(MAPTR_CLASSES)}
    gts = []
    for rec, info in zip(records, infos, strict=True):
        vectors = []
        for cls_name, lines in info["annotation"].items():
            if cls_name not in idx:
                raise ValueError(f"未知类 {cls_name!r}(官方 map_classes 口径:{list(MAPTR_CLASSES)})")
            for pts in lines:
                vectors.append(
                    {
                        "pts": [[float(px), float(py)] for px, py in pts],
                        "pts_num": len(pts),
                        "cls_name": cls_name,
                        "type": idx[cls_name],
                    }
                )
        gts.append({"sample_token": rec["token"], "vectors": vectors})
    return {"GTs": gts}


def link_neighbors(records: list[dict]) -> None:
    """填 prev/next(scene 内相邻帧;首尾自指,与官方 infos 的边界处理一致)。"""
    tokens = [r["token"] for r in records]
    for i, rec in enumerate(records):
        rec["prev"] = tokens[max(0, i - 1)]
        rec["next"] = tokens[min(len(records) - 1, i + 1)]


# nuScenesMap 的 `_load_layer` 是 `self.json_obj[layer]` 硬取键(不是 .get),故空桩也必须
# 带齐全部键:几何层 + 查找层 + 线层给 []、字典层给 {}、canvas_edge 给二元列表。
_MAP_LIST_LAYERS = (
    "polygon", "line", "node", "drivable_area", "road_segment", "road_block", "lane",
    "ped_crossing", "walkway", "stop_line", "carpark_area", "road_divider", "lane_divider",
    "traffic_light", "lane_connector",
)  # fmt: skip
_MAP_DICT_LAYERS = ("arcline_path_3", "connectivity")
MAP_LOCATIONS = ("boston-seaport", "singapore-hollandvillage", "singapore-onenorth", "singapore-queenstown")


def write_map_expansion_stubs(out: Path) -> int:
    """写 4 份空地图 expansion(见模块 docstring 第 2 点);返回写入的 location 数。"""
    stub = {"version": "1.3", "canvas_edge": [500, 500]}  # version 会被字符串比较,须 ≥ '1.3'
    stub.update(dict.fromkeys(_MAP_LIST_LAYERS, []))
    stub.update(dict.fromkeys(_MAP_DICT_LAYERS, {}))
    d = out / "maps" / "expansion"
    d.mkdir(parents=True, exist_ok=True)
    for loc in MAP_LOCATIONS:
        (d / f"{loc}.json").write_text(json.dumps(stub), encoding="utf-8")
    return len(MAP_LOCATIONS)


def _selfcheck(records: list[dict], infos: list[dict], tol: float = 1e-9) -> float:
    """官方 lidar2img 链 vs 我们自己的 ego→img 链,返回最大绝对误差。

    左边走**我们写进 pkl 的字段 + 官方那几行原样代码**,右边走原始 `sensor2ego` +
    我们自己的换基常量 —— 两条路各算各的,只在"口径真的一致"时才相等。
    """
    worst = 0.0
    for rec, info in zip(records, infos, strict=True):
        for name in CAM_NAMES:
            c = rec["cams"][name]
            off = official_lidar2img(c)[:3, :]
            ours = our_ego2img(info["cams"][name]["sensor2ego"], c["cam_intrinsic"])
            worst = max(worst, float(np.abs(off - ours).max()))
    if worst > tol:
        raise SystemExit(f"投影自证失败:官方链与我们链最大偏差 {worst:.3e} > {tol:.1e}")
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surround", required=True, help="B1 环视采集根目录")
    ap.add_argument("--out", default="outputs/maptr_official/data", help="产出目录(项目内)")
    ap.add_argument("--train-frames", type=int, default=200, help="训练帧数(其余归 val)")
    args = ap.parse_args()

    root = Path(project_path(args.surround))
    out = Path(project_path(args.out))
    out.mkdir(parents=True, exist_ok=True)
    infos = json.loads((root / "map_infos.json").read_text(encoding="utf-8"))
    n_train = args.train_frames
    if not 0 < n_train < len(infos):
        raise SystemExit(f"--train-frames 越界:{n_train} ∉ (0, {len(infos)})")

    splits = {"train": infos[:n_train], "val": infos[n_train:]}
    meta = {
        "surround": str(root),
        "n_frames": len(infos),
        "classes": list(MAPTR_CLASSES),
        "map_location": MAP_LOCATION,
        "split": {k: [i["token"] for i in v] for k, v in splits.items()},
    }

    for tag, frame_infos in splits.items():
        records = [
            build_record(info, root, timestamp_us=i * FRAME_DT_US) for i, info in enumerate(frame_infos)
        ]
        link_neighbors(records)
        worst = _selfcheck(records, frame_infos)
        # **必须是 dict**:官方 `NuScenesDataset.load_annotations`(两仓库同源)是
        #     data = mmcv.load(ann_file); data_infos = sorted(data['infos'], key=…timestamp)
        #     self.metadata = data['metadata']; self.version = self.metadata['version']
        # 直接 dump 一个 list 会在数据集构造期就 TypeError(踩过)。形状照官方
        # `create_data.py` 的产物:{"infos": [...], "metadata": {"version": …}}。
        # version 只在 nuscenes-devkit 评测支路(`create_nusc_evaluator`)被读,我们走
        # `map_utils` 那条(不依赖它);取值随 A0-lite 的黄金基准 = mini 转换口径。
        with open(out / f"carla_infos_{tag}.pkl", "wb") as f:
            pickle.dump({"infos": records, "metadata": {"version": "v1.0-mini"}}, f)
        (out / f"carla_map_anns_{tag}.json").write_text(
            json.dumps(gt_payload(records, frame_infos), ensure_ascii=False), encoding="utf-8"
        )
        meta[f"{tag}_selfcheck_max_err"] = worst
        n_vec = sum(len(r["vectors"]) for r in gt_payload(records, frame_infos)["GTs"])
        print(
            f"[{tag}] {len(records)} 帧 → carla_infos_{tag}.pkl + carla_map_anns_{tag}.json"
            f"(GT {n_vec} 条,投影自证 max|Δ| = {worst:.2e})"
        )

    n_stub = write_map_expansion_stubs(out)
    meta["map_expansion_stubs"] = {
        "locations": list(MAP_LOCATIONS),
        "n": n_stub,
        "note": "空桩,仅 v1 构造期需要",
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stub] maps/expansion/ × {n_stub}(空地图,仅 MapTR v1 构造期需要)")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()

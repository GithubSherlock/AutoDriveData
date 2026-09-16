"""CARLA 轨迹 → HiVT TemporalData 组装器(纯值,不 import carla)。

消费 bin/collect_traj.py 的 traj.json(逐帧所有 agent x/y/yaw),滑窗切
50 帧(20 历史 + 30 未来)场景,复用 §5.11 xodr centerline 做 lane 切段
(替代 ArgoverseMap),产出 HiVT `ArgoverseV1Dataset` 同构的 processed/*.pt。

坐标口径(HiVT `process_argoverse` 同构):
- 场景以**当前时刻(第 19 帧)的 AV 位置为原点**,绕 yaw(theta) 旋转到车头朝 +x
- x[:, t] = 相对位移(x[19] 后 = 未来位移);positions = 绝对局部坐标
- lane_vectors 同样转到 ego 局部系,半径 50m 内
- is_intersection:lane 是否过 junction(粗略:起点在 junction 区域内)
- turn_direction / traffic_control:无 xodr 直接对应,默认 0(有损,记录)

滑窗:每 N 步滑一个场景(默认 1),起点 s 取 [0, n_frames-50] 内所有窗口。
带 STEPS 间隔则生成 (n-50)//steps 个场景(去重叠)。

用法:
  python bin/assemble_traj_pt.py \
      --traj outputs/traj_town13/traj.json \
      --map-json training/map/Town13_full.json \
      --map Town13 \
      --out outputs/hivt_carla/val --steps 1 \
      --samples-per-map 250
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from autodrivedata.mapvec import BEV_RANGE, crop_to_ego, to_ego_frame, vecs_load
from autodrivedata.opendrive import parse_xodr
from autodrivedata.paths import project_path

TOTAL_STEPS = 50  # HiVT 时间步:20 历史 + 30 未来
HISTORY = 20
FUTURE = 30
LANE_RADIUS = 50.0


def load_map_centerlines(map_name: str) -> list[np.ndarray]:
    """xodr centerline 折线(**CARLA 世界系**,xy)列表——lane 向量来源。

    **关键:y 取反**(§5.11b A6 oracle:0.00cm 定案「CARLA 世界 = xodr 的 y 取反」)。
    AV 轨迹来自 CARLA(已镜像),centerline 若直接用 xodr road_to_xy 会坐标系错位
    (实测 lane_vectors 全空 = 50m 内查不到 lane)。

    lane 集合沿 road 的 lane_section 可能变化(§5.11 已修同类坑):lane 在
    s 处消失则折断该段,不抛异常。
    """
    xodr = sorted(Path("/root/autodl-tmp/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps").glob(f"**/{map_name}.xodr"))
    if not xodr:
        raise SystemExit(f"找不到 {map_name}.xodr")
    m = parse_xodr(str(xodr[0]))
    out: list[np.ndarray] = []
    for road in m.roads.values():
        for lane in road.lane_sections:
            for l in lane.left + lane.right:
                if l.type != "driving":
                    continue
                pts: list[tuple[float, float]] = []
                for s in np.linspace(0.0, road.length, 40):
                    try:
                        x, y, _ = road_to_xy_ego(road, l, s)
                        pts.append((x, -y))  # y 取反 → CARLA 世界系
                    except ValueError:
                        break  # lane 在 s 消失 → 折断
                if len(pts) >= 2:
                    out.append(np.asarray(pts, dtype=np.float64))
    return out


def road_to_xy_ego(road, lane, s):
    from autodrivedata.opendrive import lane_centerline_t, road_to_xy
    return road_to_xy(road, s, lane_centerline_t(road, s, lane.id))


def rotate_pts(pts: np.ndarray, origin: np.ndarray, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    r = np.array([[c, -s], [s, c]])
    return (pts - origin) @ r.T


def build_scene(
    agents_xy: list[np.ndarray],  # [N, 50, 2] 绝对世界 xy
    agents_yaw: list[np.ndarray],  # [N, 50] 世界 yaw
    centerlines: list[np.ndarray],  # 世界系车道折线
    av_idx: int,
) -> dict:
    """一帧滑窗 → TemporalData 字段(dict of tensor)。"""
    n = len(agents_xy)
    # 原点 = 当前时刻(第 19 帧)AV 位置;theta = AV 朝向
    origin = agents_xy[av_idx][HISTORY - 1]
    av_prev = agents_xy[av_idx][HISTORY - 2]
    theta = math.atan2(origin[1] - av_prev[1], origin[0] - av_prev[0])

    # 全部转 ego 局部系 → numpy 数组(HiVT x[N,50,2] 口径)
    local = np.asarray([rotate_pts(a, origin, theta) for a in agents_xy], dtype=np.float32)  # [N,50,2]

    # x = 相对位移;positions = 绝对局部;padding_mask
    x = np.zeros((n, TOTAL_STEPS, 2), dtype=np.float32)
    positions = np.zeros((n, TOTAL_STEPS, 2), dtype=np.float32)
    padding_mask = np.ones((n, TOTAL_STEPS), dtype=bool)
    for i in range(n):
        positions[i] = local[i]
        # 有效时间步 = 车存在的帧(世界坐标非 0)
        valid = ~(np.abs(agents_xy[i]).sum(axis=1) < 1e-6)
        padding_mask[i] = ~valid
        x[i, 1:] = local[i, 1:] - local[i, :-1]  # 位移
        x[i, 0] = 0.0
    # y = 未来绝对位移(相对当前位置)
    y = np.zeros((n, FUTURE, 2), dtype=np.float32)
    for i in range(n):
        valid_f = ~(np.abs(agents_xy[i][HISTORY:]).sum(axis=1) < 1e-6)
        y[i] = np.where(valid_f[:, None], local[i, HISTORY:] - origin, 0.0)
    # padding_mask 未来:第 19 帧不可见 → 全不可预测
    for i in range(n):
        if padding_mask[i, HISTORY - 1]:
            padding_mask[i, HISTORY:] = True

    # lane:50m 内 centerline 切段
    lane_vecs, lane_positions = [], []
    for cl in centerlines:
        cl2 = cl[:, :2] if cl.shape[1] >= 2 else cl
        for k in range(len(cl2) - 1):
            p = rotate_pts(cl2[k:k + 2], origin, theta)
            if np.linalg.norm(p[0] - local[av_idx][HISTORY - 1]) < LANE_RADIUS:
                lane_vecs.append(p[1] - p[0])
                lane_positions.append(p[0])
    lane_vectors = torch.tensor(lane_vecs, dtype=torch.float) if lane_vecs else torch.zeros((0, 2))
    lane_positions_t = torch.tensor(lane_positions, dtype=torch.float) if lane_positions else torch.zeros((0, 2))
    # lane_actor_index/vectors:50m 内所有 agent↔lane
    node_positions = torch.tensor(positions[:, HISTORY - 1], dtype=torch.float)
    n_lane = lane_vectors.size(0)
    if n_lane:
        lai, lav = [], []
        for i in range(n):
            for j in range(n_lane):
                v = lane_positions_t[j] - node_positions[i]
                if v.norm() < LANE_RADIUS:
                    lai.append([j, i])
                    lav.append(v)
        lane_actor_index = torch.tensor(lai, dtype=torch.long).t().contiguous() if lai else torch.zeros((2, 0), dtype=torch.long)
        lane_actor_vectors = torch.stack(lav) if lav else torch.zeros((0, 2))
    else:
        lane_actor_index = torch.zeros((2, 0), dtype=torch.long)
        lane_actor_vectors = torch.zeros((0, 2))

    # 其他标量
    num_nodes = n
    is_intersections = torch.zeros(lane_vectors.size(0), dtype=torch.uint8)
    turn_directions = torch.zeros(lane_vectors.size(0), dtype=torch.uint8)
    traffic_controls = torch.zeros(lane_vectors.size(0), dtype=torch.uint8)
    # bos_mask(与 HiVT 同构)
    bos_mask = torch.zeros(n, HISTORY, dtype=torch.bool)
    bos_mask[:, 0] = ~torch.tensor(padding_mask[:, 0])
    bos_mask[:, 1:] = torch.tensor(padding_mask[:, :HISTORY - 1]) & ~torch.tensor(padding_mask[:, 1:HISTORY])

    return {
        "x": torch.tensor(x),
        "positions": torch.tensor(positions),
        "y": torch.tensor(y),
        "padding_mask": torch.tensor(padding_mask),
        "bos_mask": bos_mask,
        "rotate_angles": torch.zeros(n, dtype=torch.float),
        "lane_vectors": lane_vectors,
        "is_intersections": is_intersections,
        "turn_directions": turn_directions,
        "traffic_controls": traffic_controls,
        "lane_actor_index": lane_actor_index,
        "lane_actor_vectors": lane_actor_vectors,
        "num_nodes": num_nodes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="collect_traj.py 输出 traj.json")
    ap.add_argument("--map", required=True, help="地图名(找 xodr centerline)")
    ap.add_argument("--out", required=True, help="输出 processed 目录(HiVT dataset 同构)")
    ap.add_argument("--steps", type=int, default=1, help="滑窗步长(>1 去重叠)")
    ap.add_argument("--samples-per-map", type=int, default=250, help="每张图最多生成场景数")
    args = ap.parse_args()

    traj = json.loads(Path(args.traj).read_text())
    frames = traj["frames"]
    meta = traj["meta"]
    n_agents = meta["n_agents"]

    # 组装 N×F×2 轨迹矩阵
    agents_xy = np.zeros((n_agents, len(frames), 2), dtype=np.float64)
    agents_yaw = np.zeros((n_agents, len(frames)), dtype=np.float64)
    for fi, f in enumerate(frames):
        for a in f["agents"]:
            agents_xy[a["id"]][fi] = (a["x"], a["y"])
            agents_yaw[a["id"]][fi] = a["yaw"]

    centerlines = load_map_centerlines(args.map)
    print(f"[lane] {args.map} centerline 折线 {len(centerlines)} 条")

    out = Path(args.out)
    (out / "processed").mkdir(parents=True, exist_ok=True)

    # 滑窗:起点 0..n-50,步长 steps;跳过含全零帧(车已消失)的窗口
    n_frames = len(frames)
    n_scenes = 0
    for s0 in range(0, n_frames - TOTAL_STEPS + 1, args.steps):
        if n_scenes >= args.samples_per_map:
            break
        win = slice(s0, s0 + TOTAL_STEPS)
        # 有效窗口:至少 ego 全程在位
        if np.abs(agents_xy[0][win]).sum() < 1e-6:
            continue
        scene = build_scene(
            [agents_xy[j][win] for j in range(n_agents)],
            [agents_yaw[j][win] for j in range(n_agents)],
            centerlines,
            av_idx=0,
        )
        seq_id = f"{args.map}_{s0:05d}"
        scene["seq_id"] = int(s0)
        torch.save(scene, out / "processed" / f"{seq_id}.pt")
        # HiVT dataset len() = len(os.listdir(raw_dir)),processed 名 = raw 名去后缀
        # → 每个 .pt 必须配同名 .csv 占位(raw 目录),否则 len()=0(空数据集)。
        (out / "data").mkdir(parents=True, exist_ok=True)
        (out / "data" / f"{seq_id}.csv").touch()
        n_scenes += 1
    print(f"[done] {args.out}/processed: {n_scenes} 场景(map={args.map}, 滑窗步长={args.steps})")


if __name__ == "__main__":
    main()

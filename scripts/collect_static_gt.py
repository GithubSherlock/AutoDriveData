"""P2 静态目标/道路特征 GT 采集器(地图查询源,2026-09-09)。

数据源(Plan.md §5.6a 实测裁决):Town10HD_Opt 静态定义全在 OpenDRIVE——
signal/landmark(58×Signal_3Light_Post01 红绿灯 + Sign_Stop/Yield)与
waypoint.lane_marking(车道线实体)。非 actor 无 tag,semantic LiDAR 打不到、
无信号 actor 可用 → 地图查询 API 定案;RoadRunner 挂起至 M4。

输出(每帧,与帧对齐、ego 位姿锚定):
  training/static_gt/{fid}.json   StaticFrame(信号 + 车道线段,世界系)
  training/image_2/{fid}.png      原始相机帧
  training/overlay/{fid}.png      目检叠加图:信号锚点(红) + 车道线段
                                  (白=White 黄=Yellow)投影到图像平面

用法(base env): python scripts/collect_static_gt.py [--frames 40] [--out outputs/kitti_static_demo]
"""

from __future__ import annotations

import argparse
import queue
from pathlib import Path
from typing import cast

import carla
import numpy as np
from carla_common import CAM_ATTRS, SENSOR_OFFSET, loc, rad, spawn_ego, sync_mode
from PIL import Image, ImageDraw

from autodrivedata import geometry as g
from autodrivedata.calib import CameraIntrinsics
from autodrivedata.static_gt import (
    LaneSegment,
    StaticFrame,
    StaticSignal,
    landmark_kind,
    merge_lane_marks,
)

SPEED = 8.0  # m/s 定速直行(A/B 纪律:起点/轨迹可复现)
LANDMARK_HORIZON = 65.0  # 与 GT max_distance 一致的静态锚点视距
LANE_STEP = 5.0  # 车道线采样步长(m)
LANE_STEPS = 13  # 13×5m = 65m 采样长度
MARK_COLOR = {"White": (255, 255, 255), "Yellow": (220, 190, 60)}


def camera_to_img(
    world_pt: tuple[float, float, float],
    cam_loc: tuple[float, float, float],
    cam_rot: tuple[float, float, float],
    k: CameraIntrinsics,
) -> tuple[float, float] | None:
    """世界点 → 图像像素;相机后/图外/深度过近返回 None。"""
    c = g.world_to_cam(np.asarray([world_pt], dtype=np.float64), cam_loc, cam_rot)[0]
    if float(c[2]) <= 0.5:
        return None
    u = k.fx * (c[0] / c[2]) + k.cx
    v = k.fy * (c[1] / c[2]) + k.cy
    if not (0 <= u < k.width and 0 <= v < k.height):
        return None
    return float(u), float(v)


def collect_static_frame(
    world: carla.World, ego: carla.Vehicle, frame_id: str
) -> tuple[StaticFrame, list[StaticSignal], list[LaneSegment]]:
    """地图查询一帧静态 GT:附近信号 landmark + 沿本车道车道线段。"""
    m = world.get_map()
    ego_pt = loc(ego.get_transform())

    # 信号:landmark(世界系锚点 + yaw),按视距过滤。
    # 同一物理信号杆会挂在多条 lane 上(重复锚点)→ 按 (name, 位置) 去重;
    # OpenDRIVE 方位累积出现 -360/-540 类值 → 规范化到 [0, 360)
    sigs: list[StaticSignal] = []
    seen: set[tuple[str, float, float]] = set()
    for lm in m.get_all_landmarks():
        p = lm.transform.location
        if np.linalg.norm([p.x - ego_pt[0], p.y - ego_pt[1]]) > LANDMARK_HORIZON:
            continue
        kind = landmark_kind(str(lm.name))
        if kind == "unknown":
            continue  # 未知 landmark 不输出(防把非信号类当 GT)
        key = (str(lm.name), round(float(p.x), 1), round(float(p.y), 1))
        if key in seen:
            continue
        seen.add(key)
        yaw = float(lm.transform.rotation.yaw) % 360.0
        sigs.append(
            StaticSignal(
                landmark_id=str(lm.id),
                kind=kind,
                name=str(lm.name),
                location=(p.x, p.y, p.z),
                yaw_deg=yaw,
            )
        )

    # 车道线:沿 ego 车道向前采样,mark 点落在车道边缘(中心 ± lane_width/2)
    wp = m.get_waypoint(carla.Location(x=ego_pt[0], y=ego_pt[1], z=ego_pt[2]))
    samples: list[dict[str, float | str]] = []
    cur = wp
    for _ in range(LANE_STEPS):
        rot = g.carla_rotation_matrix(rad(cur.transform.rotation))
        right = rot[:, 1]  # CARLA R 矩阵: 列0=前向 x,列1=右向 y
        b = cur.transform.location
        half = cur.lane_width / 2.0
        for side, sign in (("right", +1.0), ("left", -1.0)):
            mark = cur.right_lane_marking if side == "right" else cur.left_lane_marking
            if mark is None:
                continue
            if (
                str(mark.type) == "NONE"
                or str(mark.color) == "NONE"
                or float(mark.width) <= 0.0
            ):
                continue  # 无实体标线(路口/默认 xodr 占位 w=0),不输出
            px = b.x + right[0] * half * sign
            py = b.y + right[1] * half * sign
            pz = b.z + right[2] * half * sign
            samples.append(
                {
                    "side": side,
                    "mark_type": str(mark.type),
                    "color": str(mark.color),
                    "width": float(mark.width),
                    "x": float(px),
                    "y": float(py),
                    "z": float(pz),
                }
            )
        nxt = cur.next(LANE_STEP)
        if not nxt:
            break
        cur = nxt[0]

    segs: list[LaneSegment] = []
    for s in merge_lane_marks(samples):
        pts = tuple((float(p[0]), float(p[1]), float(p[2])) for p in s["points"])
        segs.append(
            LaneSegment(
                side=str(s["side"]),
                mark_type=str(s["mark_type"]),
                color=str(s["color"]),
                width=float(s["width"]),
                points=pts,
            )
        )

    frame = StaticFrame(
        frame_id=frame_id,
        ego_location=ego_pt,
        ego_yaw_deg=float(ego.get_transform().rotation.yaw),
        signals=tuple(sigs),
        lane_lines=tuple(segs),
    )
    return frame, sigs, segs


def draw_overlay(
    img: Image.Image,
    sigs: list[StaticSignal],
    segs: list[LaneSegment],
    cam_loc: tuple[float, float, float],
    cam_rot: tuple[float, float, float],
    k: CameraIntrinsics,
) -> Image.Image:
    """静态 GT → 目检叠加图(红点=信号锚点,彩线=车道线段)。"""
    d = ImageDraw.Draw(img)
    for seg in segs:
        col = MARK_COLOR.get(seg.color, (170, 170, 170))
        pts_2d: list[tuple[float, float]] = []
        for p in seg.points:
            uv = camera_to_img((p[0], p[1], p[2]), cam_loc, cam_rot, k)
            if uv is not None:
                pts_2d.append(uv)
        if len(pts_2d) >= 2:
            for a, b2 in zip(pts_2d, pts_2d[1:]):
                d.line([a, b2], fill=col, width=3)
        for z in pts_2d:
            d.ellipse([z[0] - 2, z[1] - 2, z[0] + 2, z[1] + 2], fill=col)
    for s in sigs:
        uv = camera_to_img(s.location, cam_loc, cam_rot, k)
        if uv is None:
            continue
        x, y = uv
        d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(255, 40, 40), width=3)
        d.line([x - 11, y, x + 11, y], fill=(255, 40, 40), width=2)
        d.line([x, y - 11, x, y + 11], fill=(255, 40, 40), width=2)
        d.text((x + 10, y - 12), s.kind, fill=(255, 40, 40))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--out", default="outputs/kitti_static_demo")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    sync_mode(world)

    # 清场 + 锚定 pt0(视角固定 + 轨迹可复现,同 collect_ab_route)
    for a in world.get_actors():
        if a.type_id.startswith(("vehicle", "walker", "controller")):
            a.destroy()
    for _ in range(3):
        world.tick()
    ego = spawn_ego(world)
    ego.set_autopilot(False)
    pts = world.get_map().get_spawn_points()
    ego.set_transform(carla.Transform(pts[0].location, carla.Rotation(yaw=0.0)))
    world.tick()
    here = ego.get_location()
    if here.distance(pts[0].location) > 1.0:
        raise RuntimeError(f"ego 未能锚定 pts[0]: 落在 {here}")
    print(f"[ego] 锚定 pts[0] @ ({here.x:.1f}, {here.y:.1f}) 朝世界 +x")

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    for t, v in CAM_ATTRS.items():
        cam_bp.set_attribute(t, v)
    camera = cast(carla.Sensor, world.spawn_actor(cam_bp, SENSOR_OFFSET, attach_to=ego))
    q: queue.Queue = queue.Queue()
    camera.listen(q.put)
    k = CameraIntrinsics(
        width=int(CAM_ATTRS["image_size_x"]),
        height=int(CAM_ATTRS["image_size_y"]),
        fov_h_deg=float(CAM_ATTRS["fov"]),
    )

    out = Path(args.out)
    (out / "training/static_gt").mkdir(parents=True, exist_ok=True)
    (out / "training/image_2").mkdir(parents=True, exist_ok=True)
    (out / "training/overlay").mkdir(parents=True, exist_ok=True)

    fwd = ego.get_transform().get_forward_vector()
    fwd_v = carla.Vector3D(x=fwd.x * SPEED, y=fwd.y * SPEED, z=0.0)
    try:
        for i in range(args.frames):
            ego.set_target_velocity(fwd_v)
            world.tick()
            image: carla.Image = q.get(timeout=10)
            cam_t = camera.get_transform()
            cam_loc, cam_rot = loc(cam_t), rad(cam_t.rotation)
            frame, sigs, segs = collect_static_frame(world, ego, f"{i:06d}")

            fid = f"{i:06d}"
            (out / "training/static_gt" / f"{fid}.json").write_text(frame.to_json())

            png = out / "training/image_2" / f"{fid}.png"
            image.save_to_disk(str(png))
            img = Image.open(png).convert("RGB")
            draw_overlay(img, sigs, segs, cam_loc, cam_rot, k).save(
                out / "training/overlay" / f"{fid}.png"
            )

            if (i + 1) % 10 == 0 or i == args.frames - 1:
                npts = sum(len(s.points) for s in segs)
                kinds = sorted({s.kind for s in sigs})
                print(
                    f"[frame {i + 1}/{args.frames}] 信号 {len(sigs)}（{kinds}）"
                    f" 车道线段 {len(segs)} ({npts} 点)"
                )
    finally:
        camera.stop()
        camera.destroy()
        for a in world.get_actors():
            if a.type_id.startswith(("vehicle", "walker", "controller")):
                a.destroy()
    print(f"[done] static GT root: {out.resolve()} ({args.frames} frames)")


if __name__ == "__main__":
    main()

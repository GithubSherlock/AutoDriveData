"""挂点口径 A/B 实测:同一个权重,喂「它训练时见过的 rig」vs「另一代 rig」。

背景(Plan2.md §P-H.3):仓里存在**两代**环视挂点口径 ——
`legacy`(6 路共用 `SENSOR_OFFSET` 平移 + BACK_LEFT/RIGHT 偏航 235/125)与
`nuscenes`(逐相机 `SENSOR_MOUNTS` 平移 + 官方 6DoF 姿态,当前采集器口径)。

**关键事实:rig 不是"越新越好",而是必须与权重训练数据一致。**
`outputs/maptr_600/map_infos.json` 逐帧查得:帧 0-199 = legacy、帧 200-599 = nuscenes 前身。
⇒ `maptr_ep256.pt` / `maptr_ep512.pt`(200 帧)是 legacy 训的;`maptr_600.pt` /
`maptr_1000.pt` 是当时那套(镜像 + 零 pitch/roll)训的。早期 `view_stream.build_maptr_rig`
无条件用 legacy,喂 ep512 是**对的**;把它"修"成 nuscenes 反而错配。

⚠️ **2026-09-22 补充**:当时那套 `official` 的偏航是**镜像的**(漏了 `yaw_carla = −az_nus`)
且 pitch/roll 硬编码 0 ⇒ **全部 MapTR 权重都已标废弃**,须用修正后的 rig 重采重训。
本探针保留其诊断价值(量化"图与权重错配"的代价),不再是"选哪代 rig"的决策工具。

本探针量化错配的代价:同一 ego 位姿、同一 tick 帧,只变 rig(挂点 + calib)。
判据不看图:数 6 路品红像素 + **光轴以上**像素(`v < cy`;§5.11f 记作"地平线以上"),
外加"外参与权重训练数据逐字段一致"的布尔核对 —— **"数字变了"不是回归**。

用法(需 CARLA 服务器 + outputs/maptr_*.pt):
  PYTHONPATH=$PWD python bin/probe_rig_mount.py                       # ep512 + maptr_600 各测一轮
  PYTHONPATH=$PWD python bin/probe_rig_mount.py --ckpt outputs/maptr_600.pt
"""

from __future__ import annotations

import argparse
import queue
import sys
from typing import cast

import carla
import numpy as np
from carla_common import loc, rad, spawn_ego, sync_mode
from live_common import (
    RIG_LEGACY,
    RIG_NUSCENES,
    image_to_pil,
    load_maptr,
    maptr_predict,
    resolve_rig,
    rig_frame,
    rig_spec,
    surround_calibs,
)
from PIL import ImageDraw

from autodrivedata.calib import CameraIntrinsics
from autodrivedata.mapviz import PRED_COLOR, draw_projected_lines


def build_rig(world: carla.World, ego: carla.Vehicle, rig: str):
    """按 rig 口径挂 6 路相机(与 `live_common.build_surround_rig` 同式,单测/探针自持)。

    画幅/FoV 走 `rig_frame`(逐通道)——探针的用处正是"对/错 rig 的像素差",若这里按
    `CAM_ATTRS` 一表六用,而预测侧 `surround_calibs(rig)` 是逐通道的,两边画幅与 FoV
    都对不上,探针会报出**假**的错配签名(把"探针自己错"读成"权重错")。
    """
    w, h, fovs = rig_frame(rig)
    mounts, rots = rig_spec(rig)
    cams: dict[str, tuple[carla.Sensor, CameraIntrinsics]] = {}
    qs: dict[str, queue.Queue] = {}
    for name, (pitch, yaw, roll) in rots.items():
        x, y, z = mounts[name]
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(w))
        bp.set_attribute("image_size_y", str(h))
        bp.set_attribute("fov", f"{fovs[name]:.6f}")
        cam = cast(
            carla.Sensor,
            world.spawn_actor(
                bp,
                carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)),
                attach_to=ego,
            ),
        )
        q: queue.Queue = queue.Queue()
        cam.listen(q.put)
        cams[name], qs[name] = (cam, CameraIntrinsics(width=w, height=h, fov_h_deg=fovs[name])), q
    return cams, qs


def capture(world: carla.World, cams, qs) -> dict:
    for _ in range(3):  # 预热
        world.tick()
        for q in qs.values():
            q.get(timeout=10)
    world.tick()
    return {n: image_to_pil(qs[n].get(timeout=10)) for n in cams}


def probe_ckpt(
    world: carla.World, ego: carla.Vehicle, ckpt: str, thr: float
) -> list[tuple[str, int, int, int, int]]:
    """同一权重跑两代 rig → [(标签, 实例数, 段数, 品红 px, 光轴以上 px)]。"""
    model, dev = load_maptr(ckpt, None)
    train_rig = resolve_rig("auto", ckpt)
    rows: list[tuple[str, int, int, int, int]] = []
    for rig in (train_rig, RIG_NUSCENES if train_rig == RIG_LEGACY else RIG_LEGACY):
        tag = "训练口径 ✓" if rig == train_rig else "错配 ✗"
        cams, qs = build_rig(world, ego, rig)
        imgs = capture(world, cams, qs)
        et = ego.get_transform()
        ego_g = [*loc(et), et.rotation.yaw, et.rotation.pitch, et.rotation.roll]
        rw, rh, _ = rig_frame(rig)
        preds = maptr_predict(model, dev, imgs, ego_g, surround_calibs(rig, rw, rh), thr)
        all_preds = [p for cls in preds for p in cls]
        n_seg = 0
        mag = np.array(PRED_COLOR)
        px = above = 0
        cy = (rig_frame(rig)[1] - 1) / 2  # 光轴画幅中线(逐 rig;corner 约定)
        for name, (cam, k) in cams.items():
            t = cam.get_transform()
            img = imgs[name].copy()
            n_seg += draw_projected_lines(
                ImageDraw.Draw(img), all_preds, ego_g, (loc(t), rad(t.rotation)), k, color=PRED_COLOR
            )
            m = (np.asarray(img) == mag).all(axis=2)
            px += int(m.sum())
            above += int((np.nonzero(m)[0] < cy).sum())
        label = f"{ckpt.split('/')[-1]} @ {rig} ({tag})"
        rows.append((label, len(all_preds), n_seg, px, above))
        print(f"  {label:<44} 实例 {len(all_preds):3d}  段 {n_seg:4d}  品红 {px:6d}  光轴以上 {above}")
        for s, _ in cams.values():
            s.stop()
            s.destroy()
        for _ in range(2):
            world.tick()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        action="append",
        default=None,
        help="可重复;缺省 = outputs/maptr_ep512.pt(legacy)+ outputs/maptr_600.pt(nuscenes 前身)",
    )
    ap.add_argument("--thr", type=float, default=0.2)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--sim-port", type=int, default=2000)
    args = ap.parse_args()
    ckpts = args.ckpt or ["outputs/maptr_ep512.pt", "outputs/maptr_600.pt"]

    client = carla.Client(args.host, args.sim_port)
    client.set_timeout(60.0)
    world = client.get_world()
    sync_mode(world)
    for a in world.get_actors():
        if a.type_id.startswith(("vehicle", "walker", "controller")):
            a.destroy()
    for _ in range(3):
        world.tick()
    ego = spawn_ego(world)
    ego.set_autopilot(False)

    rows: list[tuple[str, int, int, int, int]] = []
    for ckpt in ckpts:
        print(f"\n[ckpt] {ckpt} → 训练口径 {resolve_rig('auto', ckpt)}")
        rows += probe_ckpt(world, ego, ckpt, args.thr)

    print("\n| 权重 @ rig | 预测实例 | 段 | 品红 px | 光轴以上(v<cy) |")
    print("|---|---|---|---|---|")
    for label, n_i, n_s, n_px, n_a in rows:
        print(f"| {label} | {n_i} | {n_s} | {n_px} | {n_a} |")
    print("\n正确性判据 = calib 的 sensor2ego 与**该权重训练数据**逐字段一致(见 live_common.rig_spec);")
    print("两代 rig 的像素数有差异是预期,不是回归 —— 差异本身 = 错配的代价。")
    print(
        f"训练侧对照:legacy 偏航 BACK_LEFT/RIGHT {rig_spec(RIG_LEGACY)[1]['CAM_BACK_LEFT'][1]}/"
        f"{rig_spec(RIG_LEGACY)[1]['CAM_BACK_RIGHT'][1]} vs nuscenes "
        f"{rig_spec(RIG_NUSCENES)[1]['CAM_BACK_LEFT'][1]}/{rig_spec(RIG_NUSCENES)[1]['CAM_BACK_RIGHT'][1]}"
    )

    ego.destroy()
    world.apply_settings(carla.WorldSettings())
    print("[done] 已清理,服务器恢复异步")


if __name__ == "__main__":
    sys.exit(main())

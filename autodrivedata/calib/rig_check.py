"""rig 的**实例分割实测**:零车体像素(判据 ⑦)+ 相邻 FoV 共视(判据 ⑧)。

## 为什么不能只算盒模型/只算方位角

"画幅里有没有自身车体"能被**解析**算出来(挂点 + 包围盒 + 视锥求交,见 Plan2 §P-M.9 的
`hfov ≤ 2·min(az−90°, 270°−az)`),但那条不等式用的是**盒模型 + 无畸变针孔**两个近似。
渲染是 UE 的真实投影与真实网格 —— 只有把`instance_segmentation` 里 **ego 自己的 actor id**
逐像素数出来,才是"画幅里没有车体"的直接证据。本模块就是这台尺子。

同理,"六视角有重叠"也有两个层次:
- **方位轴重叠**(纯几何,`camera_rig.coverage_table`)= 必要不充分 —— 两条扇区在方位角上
  相交,不代表真有物落在交集里;
- **共视**(本模块)= 往重叠区**正中摆一个锥体**,两条相机各自的掩膜里都数到它。
  这是"重叠区域真的存在"的可判据形式,且与夹具无关(不看相机名、不看假设的相邻序)。

## 判据的**边界**(别把它当万能)

- 锥体只证明**该方位**上共视;80° 重叠的落点必然通过,0.1° 重叠的"重叠区"窄到摆不进去 ⇒
  几何重叠 < `MIN_COVIS_OVERLAP_DEG` 的相邻对**不做探针**,如实报 `skipped`,**不假装测过**。
- 共视探针用**声明口径的位姿/FoV** spawn 相机;若"声明 ≠ 渲染",这里仍然会通过 ——
  声明与渲染是否一致由 `verify_nus_calib` 判据 ①⑥ 分别把守,不重复。
"""

from __future__ import annotations

import math
import queue
from typing import Any, cast

import carla
import numpy as np

from autodrivedata.calib.camera_rig import coverage_table
from autodrivedata.calib.core import CameraIntrinsics, world_to_img
from autodrivedata.calib.probe_calib import (
    CONE_DIST_M,
    CONE_Z_OFF_M,
    MIN_MASK_PX,
    decode_instance,
    place_cone_along,
)
from autodrivedata.gt.export.nuscenes import camera_fov, camera_k
from autodrivedata.sim.carla_common import loc, rad
from autodrivedata.sim.live_common import rig_spec

# 与 nuScenes 落盘口径一致的分辨率(判据里的像素占比才有意义)
CAM_W, CAM_H = 1600, 900
WARMUP_TICKS = 4
PROBE_FRAMES = 3  # 共视探针每档取帧数(掩膜取"任一帧命中即算看见")
MIN_COVIS_OVERLAP_DEG = 5.0  # 低于此重叠度不摆锥:重叠区太窄,锥体必然压在边界上
# 锥心抬到 ego 原点之上多少米(`place_cone_along` 自己再加 `CONE_Z_OFF_M`)。
# **必须抬过车顶**(audi.a2 车顶 ≈1.556 m;默认落点 1.2 m 在车顶之下)——
# 官方侧相机挂点(x=+1.04)落在车身包络内,朝向后方时射线**擦着车顶**过去
# (实测差距 0.03 m),锥体会被自车自己挡住,读数凭"这一路看没看见"变成
# "这一路有没有被自车挡",不再是重叠的证据。抬到视轴高度(1.2+0.45=1.65)后
# 三路后视相机都拿到数百像素。
COVIS_Z_LIFT = 0.45


class RigCameras:
    """一套挂在该 rig **声明位姿/声明 FoV** 上的相机(可多种蓝图并挂)。

    为什么不复用 `probe_calib.SensorRig`:那个写死官方 rig、分辨率走 `carla_common.CAM_ATTRS`
    (1242×375),而本模块要 **1600×900 + 任一 rig**。复用会让 A2 探针的既有口径被本次需求拖着改。
    """

    def __init__(self, world: carla.World, ego: carla.Vehicle, rig: str, kinds: tuple[str, ...]) -> None:
        self.world = world
        self.rig = rig
        self.sensors: dict[str, tuple[carla.Sensor, queue.Queue]] = {}
        mounts, rots = rig_spec(rig)
        fovs = camera_fov(rig)
        bp_lib = world.get_blueprint_library()
        for kind in kinds:
            for name, mount in mounts.items():
                bp = bp_lib.find(f"sensor.camera.{kind}")
                bp.set_attribute("image_size_x", str(CAM_W))
                bp.set_attribute("image_size_y", str(CAM_H))
                bp.set_attribute("fov", f"{fovs[name]:.6f}")
                rot = rots[name]
                tf = carla.Transform(
                    carla.Location(*mount), carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
                )
                s = cast(carla.Sensor, world.spawn_actor(bp, tf, attach_to=ego))
                q: queue.Queue = queue.Queue()
                s.listen(q.put)
                self.sensors[f"{kind}:{name}"] = (s, q)
        for _ in range(WARMUP_TICKS):
            self.step()

    def step(self) -> dict[str, dict[str, Any]]:
        """tick 一次 + **抽干全部队列**(只取一路会让其余五路积压陈旧帧,见 §P-M.7 判据⑥的坑)。"""
        self.world.tick()
        out: dict[str, dict[str, Any]] = {}
        for key, (_, q) in self.sensors.items():
            kind, name = key.split(":", 1)
            out.setdefault(kind, {})[name] = q.get(timeout=10)
        return out

    def sensor(self, name: str, kind: str = "instance_segmentation") -> carla.Sensor:
        return self.sensors[f"{kind}:{name}"][0]

    def close(self) -> None:
        for s, _ in self.sensors.values():
            s.stop()
            s.destroy()
        self.sensors.clear()


def decode_ids(frames: dict[str, Any]) -> dict[str, np.ndarray]:
    """逐相机实例图 → actor id 数组(解码口径由 `probe_calib.decode_instance` 唯一裁决)。"""
    return {name: decode_instance(d.raw_data, CAM_H, CAM_W) for name, d in frames.items()}


def ego_pixel_counts(ids: dict[str, np.ndarray], ego_id: int) -> dict[str, dict[str, Any]]:
    """逐相机统计 **ego 自身 actor id** 的像素数(判据 ⑦ 的读数)。"""
    total = CAM_W * CAM_H
    return {
        cam: {"px": int((arr == ego_id).sum()), "frac": float((arr == ego_id).sum()) / total}
        for cam, arr in ids.items()
    }


def camera_k_carla(rig: str) -> dict[str, CameraIntrinsics]:
    """该 rig **声明口径**的逐通道内参(供"世界点是否落在画幅内"的投影诊断)。

    用声明表而不是"从 fov 重算":官方 rig 的逐通道主点是装配公差实测值
    (`cx` 792–829),重算会把它们一律拉回 `(w−1)/2` —— 那样诊断就测不到
    "声明与渲染是否一致"这件事。`fov_h_deg` 仍由 `camera_fov` 给(`fx` 与它互逆)。"""
    fovs, ks = camera_fov(rig), {c: camera_k(c, rig) for c in camera_fov(rig)}
    return {
        cam: CameraIntrinsics(
            width=CAM_W, height=CAM_H, fov_h_deg=fovs[cam], cx_override=ks[cam][1], cy_override=ks[cam][2]
        )
        for cam in fovs
    }


def in_fov(point: tuple[float, float, float], sensor: carla.Sensor, k: CameraIntrinsics) -> bool:
    """世界点按**该相机当前实际位姿**投回画幅是否落在图内(深度 ≤0.5 m / 出图 → False)。"""
    t = sensor.get_transform()
    return world_to_img(point, loc(t), rad(t.rotation), k) is not None


def covisibility(
    world: carla.World,
    ego: carla.Vehicle,
    cams: RigCameras,
    pairs: list[dict[str, Any]],
    min_overlap_deg: float = MIN_COVIS_OVERLAP_DEG,
) -> list[dict[str, Any]]:
    """相邻对重叠区正中摆一个锥 → 两路掩膜里都数到 = **该重叠区真实存在**。

    摆点用 `probe_calib.place_cone_along`(自带"只退距离不退方向"的回退档):方向被挪就
    失去了"这个方位"的含义,而距离 ±3 m 只让方位角变 ~14°。

    **单侧 0 像素时不许直接下结论**:先把锥心按**声明内参**投回该相机,分三种情形
    (见 `_verdict`)—— 落在画幅外 = 该路本就不覆盖此方位;落在画幅内却 0 像素 =
    **被自车车体遮挡**(官方挂点在车身包络内,侧视相机擦车顶,见 `COVIS_Z_LIFT`)。
    两种情形的工程含义完全不同,混成一句"FAIL"会把遮挡误报成 rig 缺陷。
    """
    t = ego.get_transform()
    origin = loc(t)
    r = np.array(t.get_matrix(), dtype=np.float64)[:3, :3]
    ks = camera_k_carla(cams.rig)
    out: list[dict[str, Any]] = []
    for p in pairs:
        if p["overlap_deg"] < min_overlap_deg or p["span"] is None:
            out.append(
                {
                    "a": p["a"],
                    "b": p["b"],
                    "overlap_deg": p["overlap_deg"],
                    "skipped": True,
                    "reason": f"重叠 {p['overlap_deg']:.2f}° < 阈值 {min_overlap_deg}°,不摆锥(不假装测过)",
                }
            )
            continue
        band = common_band(cams, ks, (p["a"], p["b"]), p["span"], origin, r)
        if band is None:
            out.append(
                {
                    "a": p["a"],
                    "b": p["b"],
                    "overlap_deg": p["overlap_deg"],
                    "span": p["span"],
                    "skipped": True,
                    "reason": (
                        f"方位轴重叠 {p['overlap_deg']:.2f}°,但 12 m 处两路**没有**共同可见方位"
                        "(挂点视差把重叠带推到画幅外)⇒ 不摆锥"
                    ),
                }
            )
            continue
        az = (band[0] + band[1]) / 2.0
        a_rad = math.radians(az)
        direction = r @ np.array([math.cos(a_rad), -math.sin(a_rad), 0.0])  # nus(+左) → CARLA(y 右)
        cone, pos = place_cone_along(
            world, (origin[0], origin[1], origin[2] + COVIS_Z_LIFT), tuple(direction)
        )
        row: dict[str, Any] = {
            "a": p["a"],
            "b": p["b"],
            "overlap_deg": p["overlap_deg"],
            "span": p["span"],
            "common_band_deg": [round(band[0], 3), round(band[1], 3)],
            "common_band_width_deg": round(band[1] - band[0], 3),
            "az_probe_deg": az,
            "cone_pos": [round(v, 3) for v in pos],
            "spawned": cone is not None,
            "px": {},
        }
        if cone is None:
            row["skipped"] = True
            row["reason"] = "锥体三档距离都摆不进去(碰撞),未测"
            out.append(row)
            continue
        seen = {p["a"]: 0, p["b"]: 0}
        try:
            for _ in range(PROBE_FRAMES):
                ids = decode_ids(cams.step()["instance_segmentation"])
                for cam in (p["a"], p["b"]):
                    seen[cam] = max(seen[cam], int((ids[cam] == cone.id).sum()))
            row["px"] = {p["a"]: seen[p["a"]], p["b"]: seen[p["b"]]}
            row["in_fov"] = {cam: in_fov(pos, cams.sensor(cam), ks[cam]) for cam in (p["a"], p["b"])}
            row["verdict"] = {cam: _verdict(seen[cam], row["in_fov"][cam]) for cam in (p["a"], p["b"])}
        finally:
            cone.destroy()
        row["both_visible"] = bool(all(v >= MIN_MASK_PX for v in seen.values()))
        row["min_mask_px"] = MIN_MASK_PX
        out.append(row)
    return out


def common_band(
    cams: RigCameras,
    ks: dict[str, CameraIntrinsics],
    names: tuple[str, str],
    span: list[float],
    origin: tuple[float, float, float],
    r_ego: np.ndarray,
    step_deg: float = 0.25,
) -> tuple[float, float] | None:
    """方位轴重叠带里,**两路在探针距离处真正共同可见**的方位区间;没有 → None。

    ★ 这是本探针的核心订正(2026-09-23 实测):方位轴重叠是**无穷远**口径 —— 它假定
    "光轴方位 + 半 FoV"的扇区相交即重叠,但两个挂点相距最远 3.5 m,而探测点在 12 m 外,
    于是**视差**让同一世界点在两路里的方位角差最多 1.7°。后果实测可见:官方 rig 的
    `CAM_FRONT_LEFT ↔ CAM_BACK_LEFT` 方位轴重叠 11.20°,但重叠带**外沿**那一侧的点
    投回 CAM_FRONT_LEFT 已在画幅外(88.97° > 上限 87.31°)—— 只看方位轴就会得出
    "11.2° 重叠"这个在有限距离下不成立的结论。

    于是本函数沿重叠带扫一遍,用**声明内参 + 相机当前实际位姿**逐点判两路是否都落在画幅内,
    取最长连续命中段。它同时是"共视区有多宽"的直接读数。
    """
    lo, hi = span
    n = max(2, int((hi - lo) / step_deg) + 1)
    azs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    hit: list[float] = []
    for az in azs:
        a_rad = math.radians(az)
        d = r_ego @ np.array([math.cos(a_rad), -math.sin(a_rad), 0.0])
        p = (
            origin[0] + CONE_DIST_M * d[0],
            origin[1] + CONE_DIST_M * d[1],
            origin[2] + COVIS_Z_LIFT + CONE_Z_OFF_M,
        )
        if all(in_fov(p, cams.sensor(nm), ks[nm]) for nm in names):
            hit.append(az)
    if not hit:
        return None
    # 取最长连续段(命中集可能有缝:某个 az 恰好被画幅边界切掉)
    runs: list[list[float]] = [[hit[0]]]
    for az in hit[1:]:
        if az - runs[-1][-1] <= step_deg * 1.5:
            runs[-1].append(az)
        else:
            runs.append([az])
    best = max(runs, key=len)
    return (best[0], best[-1])


def _verdict(px: int, fov_hit: bool) -> str:
    """单路读数 → 结论。三种情形**必须分开报**:混成一句 FAIL 会把遮挡当成 rig 缺陷。"""
    if px >= MIN_MASK_PX:
        return "visible"
    return "occluded_in_fov" if fov_hit else "outside_fov"


def instance_probe(world: carla.World, ego: carla.Vehicle, rig: str) -> dict[str, Any]:
    """判据 ⑦+⑧ 的**唯一执行点**:⑦ 数 ego 像素,⑧ 摆锥测共视(同一套相机,省一次 spawn)。"""
    cov = coverage_table_rig(rig)
    cams = RigCameras(world, ego, rig, ("instance_segmentation",))
    try:
        ids = decode_ids(cams.step()["instance_segmentation"])
        counts = ego_pixel_counts(ids, ego.id)
        pairs = [p for p in cov["pairs"] if p["overlap_deg"] > 1e-9]
        rows = covisibility(world, ego, cams, pairs)
    finally:
        cams.close()
    return {
        "rig": rig,
        "ego_actor_id": ego.id,
        "ego_pixels": counts,
        "ego_pixels_pass": bool(all(v["px"] == 0 for v in counts.values())),
        "covisibility": rows,
        "covisibility_pass": bool(all(r.get("skipped") or r["both_visible"] for r in rows)),
    }


def coverage_table_rig(rig: str) -> dict[str, Any]:
    """rig 名 → 方位覆盖表(纯几何;标定与 FoV 都从该 rig 的声明表取)。"""
    from autodrivedata.gt.export.nuscenes import camera_calibs

    return coverage_table(camera_calibs(rig), camera_fov(rig))

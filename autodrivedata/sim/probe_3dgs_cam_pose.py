"""**相机位姿口径的反向自证**(需 CARLA,一次性探针)。

## 它判什么

`collect_3dgs` 的相机 attach 在 spectator 上 ⇒ `set_transform` 按**父系**解释,
而 `ring_cam_pose` 给的是**世界**坐标。这个缺陷当初的症状**不是报错**:

- 整条环绕链被平移 spectator 的世界位姿(实测 **79.132 m**),
- 重建**仍然自洽**(相对几何没变)⇒ 与 §A/§B 的 27 dB 不矛盾,
- 但**按世界坐标摆的道具进不了画面** —— 看起来像"道具资产不渲染"(§C.0.4 ①)。

⇒ 判据必须能分辨"换算做了"与"换算没做"。本探针**两种写法各测一次**:

| 臂 | 传进去的位姿 | 期望读回 |
|---|---|---|
| **raw**(修复前的写法) | 世界坐标,不换算 | 差 ≈ spectator 的世界位姿 |
| **converted**(现在的写法) | `to_parent_frame(...)` | 差 ≈ **0** |

**只有 raw 那一臂真的偏了**,这条判据才有分辨力 —— 否则"读回等于请求"可能只是因为
这台机器恰好把父放在原点。

用法(需 CARLA):
  python -m autodrivedata.sim.probe_3dgs_cam_pose
"""

from __future__ import annotations

import argparse
import math
from typing import cast

import carla

from autodrivedata.sim.carla_common import CAM_ATTRS, sync_mode
from autodrivedata.sim.collect_rig import ring_cam_pose, to_parent_frame
from autodrivedata.utils import runlog

#: 与 `collect_3dgs` 同值 —— 摆偏**不抛异常**,只让下游每条判据都偏低。
PLACE_TOL_M = 0.05


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--center-index", type=int, default=77, help="环绕中心取第 N 个 spawn point")
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    with runlog.run("autodrivedata.sim.probe_3dgs_cam_pose") as rl:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        world = client.get_world()
        sync_mode(world)
        bp_lib = world.get_blueprint_library()

        pts = world.get_map().get_spawn_points()
        center = pts[args.center_index].location
        print(f"[center] spawn {args.center_index} = ({center.x:.3f}, {center.y:.3f}, {center.z:.3f})")

        spec = world.get_spectator()
        spec.set_transform(carla.Transform(center + carla.Location(0, 0, 1.5), carla.Rotation(0.0, 0.0, 0.0)))
        world.tick()  # ⚠️ 快照 tick 后才刷新;不 tick 读到的是陈旧位姿
        spec_tf = spec.get_transform()
        spec_world = (spec_tf.location.x, spec_tf.location.y, spec_tf.location.z)
        print(f"[spec] 世界位姿 {tuple(round(v, 3) for v in spec_world)}")

        cam_bp = bp_lib.find("sensor.camera.rgb")
        for k, v in CAM_ATTRS.items():
            cam_bp.set_attribute(k, v)
        cam = cast(carla.Sensor, world.spawn_actor(cam_bp, carla.Transform(), attach_to=spec))
        try:
            rows = []
            for i in (0, 1, 2):
                x, y, z, yaw = ring_cam_pose(center.x, center.y, center.z, args.radius, i, 4, height=1.5)
                for label, loc in (
                    ("raw", carla.Location(x, y, z)),
                    ("converted", carla.Location(*to_parent_frame(x, y, z, parent=spec_world))),
                ):
                    cam.set_transform(carla.Transform(loc, carla.Rotation(yaw=yaw)))
                    world.tick()
                    back = cam.get_transform().location
                    err = max(abs(back.x - x), abs(back.y - y), abs(back.z - z))
                    rows.append((i, label, err))
                    print(
                        f"  [i={i} {label:>9}] 请求世界 ({x:8.3f},{y:8.3f},{z:6.3f}) "
                        f"读回 ({back.x:9.3f},{back.y:8.3f},{back.z:6.3f})  差 {err:8.3f} m"
                    )
            raw = max(r[2] for r in rows if r[1] == "raw")
            conv = max(r[2] for r in rows if r[1] == "converted")
            # spectator 的世界位姿模长 —— raw 那一臂的差应当**就是它**
            spec_norm = math.dist(spec_world, (0.0, 0.0, 0.0))
            # ★ 反向自证:如果 raw 与 converted 都 ≈ 0,说明这台机器上父在原点,判据没有分辨力
            sep = "有分辨力" if raw > 1.0 else "★ 无分辨力(raw 也是 0 —— 父恰好在原点?)"
            verdict = (
                f"raw 最大差 {raw:.3f} m(≈ |spectator 世界位姿| = {spec_norm:.3f})、"
                f"converted 最大差 {conv:.3f} m ⇒ {sep}"
            )
            print(f"\n⇒ {verdict}")
            ok = conv <= PLACE_TOL_M and raw > 1.0
            print("⇒ " + ("★ 通过:换算这一半是**必要且充分**的" if ok else "★ 不通过"))
            rl.highlight("raw_err_max", round(raw, 4))
            rl.highlight("converted_err_max", round(conv, 4))
            rl.highlight("spec_world_norm", round(spec_norm, 4))
            rl.highlight("verdict", verdict)
            if not ok:
                raise SystemExit(1)
        finally:
            cam.stop()
            cam.destroy()


if __name__ == "__main__":
    main()

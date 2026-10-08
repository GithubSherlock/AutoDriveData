"""**道具到底渲不渲染** —— 重做 §C.0.4 ①,而且这次把尺子的本底也量出来(需 CARLA)。

## 为什么要重做

§C.0.4 ① 的结论是「**车渲染、道具不渲染**」,它是 B1(用不用车当被删对象)的**承重判据**。
但那条判据有两个问题:

1. **产出它的探针没留下来**(那两张图是一次性脚本出的)⇒ 不可复跑;
2. **它是 2026-10-07 之前量的** —— 当时相机位姿还带着 **79.132 m** 的偏移缺陷
   (见 `probe_3dgs_cam_pose`),而"物体不在画面里"与"物体不渲染"**长得一模一样**。

## 尺子先验本底

§C.0.4 ② 已经证明:逐像素签名那把尺子**有本底** —— 同位姿连续 16 帧的相邻帧差仍有
**2043 px**,且**永不收敛到 0**(TAA 历史效应),而"把物体从 6 m 挪到 14 m"读数**纹丝不动**
⇒ 那个量当时根本不是物体贡献的。

⇒ 本探针**先量本底**,再要求信号 **> 5× 本底** 才算"渲染出来了"。**不量本底就没法判。**

用法(需 CARLA):
  python -m autodrivedata.sim.probe_prop_renders [--out outputs/probe_prop_renders]
"""

from __future__ import annotations

import argparse
import queue
from typing import cast

import carla
import numpy as np
from PIL import Image

from autodrivedata.sim.carla_common import CAM_ATTRS, clear_generated_actors, sync_mode
from autodrivedata.sim.collect_rig import ring_cam_pose
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 逐像素"差 > 8"的阈值(与 §C.0.4 ② 同口径,便于与归档读数对照)。
PIX_DIFF = 8
#: 信号必须超过本底这么多倍才算"真的渲染出来了"。
SIGNAL_FLOOR = 5.0
#: 中心裁剪(与归档那两张图同形)。
CROP = 0.34


def _shoot(cam: carla.Sensor, world: carla.World, q, *, ticks: int = 2) -> np.ndarray:
    """推进 `ticks` 帧,**只取最新的那一张**。

    ⚠️⚠️ **不能"drain 一次再取"**(本仓红线「**抽干≠抽干净**」的第 4 次现形):
    客户端投递是**异步**的,判"队列空"只代表*已经到的*取完了,**还在途的**会在之后补进来。
    换位姿之后尤其致命 —— 拿到的是**旧视角**那一帧,而它**长得完全正常**。

    本探针第一版就是这么错的:换到 pitch −15 后的基线其实是 pitch 0 的画面,
    于是"道具 vs 基线"差 **437864 px(94% 全幅)** —— 而差分图里是**整条街**,
    不是物体。⚠️ **两个完全不同的物体给出几乎相同的差(437864 vs 438030),那就是铁证:
    差不在物体上。**
    """
    img = None
    for _ in range(int(ticks)):
        world.tick()
        while not q.empty():
            img = q.get()
    if img is None:  # 队列还没来 ⇒ 再等一帧
        world.tick()
        img = q.get()
    a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    return a[:, :, 2::-1].copy()  # BGRA → RGB


def _diff(a: np.ndarray, b: np.ndarray) -> int:
    return int((np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) > PIX_DIFF).sum())


def _crop(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    ch, cw = int(h * CROP), int(w * CROP)
    return a[(h - ch) // 2 : (h + ch) // 2, (w - cw) // 2 : (w + cw) // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--center-index", type=int, default=77)
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--prop", default="static.prop.warningconstruction")
    ap.add_argument("--vehicle", default="vehicle.audi.tt")
    ap.add_argument("--out", default="outputs/probe_prop_renders")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with runlog.run("autodrivedata.sim.probe_prop_renders") as rl:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        world = client.get_world()
        sync_mode(world)
        bp_lib = world.get_blueprint_library()
        cleared = clear_generated_actors(world)
        print(f"[clean] 清掉 {cleared} 个本仓生成的 actor")

        center = world.get_map().get_spawn_points()[args.center_index].location
        x, y, z, yaw = ring_cam_pose(center.x, center.y, center.z, args.radius, 0, 4, height=1.5)
        # ★ **不挂 spectator** —— 世界位姿的相机,与 §C.0.4 ② 的探针同口径。
        #   挂上去就会走"父系解释",而那正是 2026-10-07 修掉的缺陷。
        cam_bp = bp_lib.find("sensor.camera.rgb")
        for k, v in CAM_ATTRS.items():
            cam_bp.set_attribute(k, v)
        cam = cast(
            carla.Sensor,
            world.spawn_actor(cam_bp, carla.Transform(carla.Location(x, y, z), carla.Rotation(yaw=yaw))),
        )
        q: queue.Queue = queue.Queue()
        cam.listen(q.put)
        spawned: list[carla.Actor] = []
        try:
            print(f"[cam] 世界位姿 ({x:.3f},{y:.3f},{z:.3f}) 看向环心 ({center.x:.3f},{center.y:.3f})")
            # ⚠️ **warm-up**:刚 spawn 的相机头几帧拿到的不是它自己的视角
            #    (TAA 历史 + 投递延迟)。不预热会把"还没热"读成"每帧都在大变"。
            for _ in range(6):
                _shoot(cam, world, q)
            n_veh = len(world.get_actors().filter("vehicle.*"))
            n_walk = len(world.get_actors().filter("walker.*"))
            print(f"[world] 车 {n_veh} / 行人 {n_walk}(会动的 NPC 会让本底爆掉)")
            shots = [_shoot(cam, world, q) for _ in range(3)]
            diags = [_diff(shots[i], shots[i + 1]) for i in range(2)]
            floor = min(diags)
            print(f"[floor] 预热后连续帧差 {diags}(取最小 {floor})> {PIX_DIFF} 的像素")
            rl.highlight("floor_px", floor)
            rl.highlight("floor_diags", diags)
            rl.highlight("n_vehicles", n_veh)
            rl.highlight("n_walkers", n_walk)
            base_a = shots[-1]

            # ⚠️ **pitch 必须扫** —— 相机在 z=+1.5 m 水平看出去时,6 m 处的地面物体
            #    落在**画面下缘之外**(半 FOV ≈16.8° < 需要的 19.3°)。pitch=0 测出来的
            #    "不渲染"是这个几何造成的,不是资产的锅。采集器本身就用 0/-15/-30。
            rows = []
            for pitch in (0.0, -15.0):
                cam.set_transform(
                    carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch=pitch, yaw=yaw))
                )
                _shoot(cam, world, q)  # ★ 换位姿后的第一张**丢掉**(见 `_shoot` 头注)
                base_p = _shoot(cam, world, q)
                Image.fromarray(_crop(base_p)).save(out / f"baseline_pitch{int(pitch)}.png")
                for label, model in (("prop", args.prop), ("vehicle", args.vehicle)):
                    bp = bp_lib.find(model)
                    a = world.try_spawn_actor(
                        bp, carla.Transform(carla.Location(x=center.x, y=center.y, z=center.z))
                    )
                    if a is None:
                        # ⚠️ **不静默跳过** —— 那会把"没 spawn 出来"读成"不渲染"
                        raise SystemExit(f"{model} 在环心 spawn 失败(位置被占?)—— 不许静默跳过")
                    spawned.append(a)
                    world.tick()
                    img = _shoot(cam, world, q)
                    n = _diff(img, base_p)
                    ratio = n / max(floor, 1)
                    dist = a.get_transform().location.distance(carla.Location(x, y, z))
                    rows.append((f"p{int(pitch)}_{label}", model, n, ratio))
                    Image.fromarray(_crop(img)).save(out / f"{label}_pitch{int(pitch)}_at_center.png")
                    # ★ **差在哪** —— 只看像素数会把"整幅闪烁"读成"物体渲染了"。
                    #   差分图里是一个**连贯的物体形状**才是真的;散在整幅就是闪烁。
                    dv = np.abs(img.astype(np.int16) - base_p.astype(np.int16)).max(axis=2)
                    Image.fromarray(np.clip(dv * 3, 0, 255).astype(np.uint8)).save(
                        out / f"diff_{label}_pitch{int(pitch)}.png"
                    )
                    print(
                        f"[pitch {pitch:>5} {label:>7}] {model} 距相机 {dist:.2f} m  "
                        f"差 {n} px = 本底 {ratio:.1f}×"
                    )
                    a.destroy()
                    spawned.remove(a)
                    world.tick()

            Image.fromarray(_crop(base_a)).save(out / "baseline_at_center.png")
            ok = {r[0]: r[3] > SIGNAL_FLOOR for r in rows}
            verdict = f"本底 {floor} px;" + "".join(f" {r[0]}={r[2]}px({r[3]:.1f}×)" for r in rows)
            print(f"\n⇒ {verdict}")
            for k, v in ok.items():
                print(f"  {k}: {'渲染出来了' if v else '★ 在噪声里 —— 判不了/没渲染'}")
            for r in rows:
                rl.highlight(f"px_{r[0]}", r[2])
            rl.highlight("verdict", verdict)
            rl.artifact(out, "evidence")
        finally:
            for a in spawned:
                if a.is_alive:
                    a.destroy()
            cam.stop()
            cam.destroy()


if __name__ == "__main__":
    main()

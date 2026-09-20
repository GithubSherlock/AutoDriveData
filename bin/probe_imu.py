"""B1 实测:量 CARLA IMU 能否支撑 FAST-LIO2 的 IESKF 预测(教程 14 路线 B 决策点)。

**要回答的问题**:把 IMU 加速度/角速度纯积分成位姿,与 CARLA 真值位姿差多少?
- 噪声全关(默认):差 → 0 说明 IMU 是"完美传感器",IESKF 的预测项在仿真里只是形式价值
- 噪声打开(真实 IMU 量级):差 → 显著说明滤波器有真信号可吃,路线 B 的 2 天值得投

**实测结论(2026-09-19,判 B3/IESKF 不投)**——矩阵(`outputs/imu_probe/*.json`):

| 场景 | 噪声 | 路径 | 位置末/均 | 姿态末/均 |
|---|---|---|---|---|
| 静止 | off | 0.00 m | 0.0863 / 0.0295 m | 0.0000 / 0.0000° |
| 静止 | on | 0.00 m | 0.1774 / 0.0514 m | 1.1396 / 0.5426° |
| 直行 8 m/s | off | 47.19 m | 1.2128 / 0.4238 m | 4.4046 / 1.5282° |
| 直行 8 m/s | on | 47.19 m | 1.6973 / 0.5633 m | 5.9191 / 2.2535° |
| 满舵绕圈 | off | 12.72 m | 0.9956 / 1.1389 m | 3.9931 / 2.8134° |

**关键发现:IMU 的陀螺读数**与**车辆真实朝向**在直行时对不上**。
- `gyro / actor.get_angular_velocity()` 三轴比值恒 ≈1.000(所有场景)→ IMU 陀螺
  **就是物理引擎报的角速度**,不是噪声、不是单位错。
- 但 8 m/s 直行时 `gyro.z = −0.0225 rad/s`(−1.29°/s),而用旋转矩阵差分算出的
  **真实 yaw 速率只有 −2e-5 rad/s**(差 1000×);轨迹相对首末直线的横向偏离 0.0005 m
  (若真以 1.29°/s 转 6 s,47 m 路径应拱起 ≈0.79 m)。**车没转,IMU 说有转。**
- 该伪角速度**完全可复现**(三次运行比值一致到 6 位小数、换图重载一致),且只在
  部分速度档出现(4/9/10/12 m/s 几乎为 0;6/7/8 明显;36+ 又消失)。
- 绕圈场景(真在转)IMU 与 R 差分**吻合**(0.9300 vs 0.9312 rad/s)→ IMU 在**真转时是对的**。
- 后果:纯积分把 4.5° 的假偏航灌进姿态(与实测 `姿态末 4.4046°` 一致)。

**决策判据(单步预测误差,IMU 预测 vs 恒速 CV 预测)**:

| 场景 | CV 位置 | IMU 位置 | CV 姿态 | IMU 姿态 |
|---|---|---|---|---|
| 静止 | 0.000 mm | 0.055 mm | 0.0000° | 0.0000° |
| 直行 8 m/s | **0.120 mm** | 0.326 mm | **0.0041°** | 0.0903° |
| 满舵绕圈 | 15.458 mm | **6.016 mm** | 4.3033° | **0.1162°** |

⇒ **直行段 IMU 预测比"什么都不做"还差 2.7×(位置)/ 22×(姿态)**;只在转弯段 IMU 才赢。
叠加此前已测的"CARLA **不模拟帧内扫描延迟**"(运动畸变校正是空操作),FAST-LIO2 用 IMU
的两个卖点在本仿真里一个为空、一个为负 → **B3(IESKF 紧耦合)不投**。

**纪律**(与采集器同口径,勿绕过):
- `carla_common.sync_mode()` 开同步(内含补 tick,否则首个 get_actors() 为空)
- 传感器 `get_transform()` 只在 tick 后刷新(C19 同类坑)→ 每帧 tick 后再读
- 不用 autopilot(不可复现):定速 + 清制动残留(同 view_stream 口径)
- **重力必须取静止段尾部**:预热期 accel.z 从 9.05 爬到 9.81(落地弹跳),拿首帧估
  会引入 ~0.7 m/s² 偏差 → 静止也积分出 1.5 m 假位移(实测)
- `set_target_angular_velocity()` **不产生转动**(实测 actor 角速度 ~1.6e-3 deg/s、
  陀螺恒 0、GT yaw 只动 0.1°)——要真转用 `apply_control(steer=1.0)`,见 `--turn`

**纯积分口径**(与 FAST-LIO2 `get_f` 一致):
  v ← v + (R·a + g)·dt ; p ← p + v·dt ; R ← R·Exp(ω·dt)
  CARLA IMU 的 accelerometer 是**比力**(含反作用,静止时 ≈ +g 向上);
  gyroscope 是 **rad/s** 的体轴角速度(与 actor 角速度同源,已实测)。

用法:
  python bin/probe_imu.py --frames 60 --speed 8 --out outputs/imu_probe --tag drive
  python bin/probe_imu.py --frames 60 --speed 8 --accel-std 0.05 --gyro-std 0.005 --tag noisy
  python bin/probe_imu.py --frames 80 --turn --tag turn    # 真转向(标定轴向/尺度)
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import time
from typing import cast

import carla
import numpy as np
from carla_common import SENSOR_OFFSET, spawn_ego, sync_mode

from autodrivedata.paths import project_path


def _rot_from_rpy(rpy: tuple[float, float, float]) -> np.ndarray:
    """CARLA (roll,pitch,yaw) 弧度 → R(传感器系 → 世界系)。

    CARLA 系:x 前 / y 右 / z 上;旋转顺序 yaw(Z) → pitch(Y) → roll(X)。
    注意 `carla_common.rad()` 的返回序是 (pitch, yaw, roll),**不是**本函数要的
    (roll, pitch, yaw) —— 调用处直接读 `rotation.roll/pitch/yaw` 转弧度,勿混用。
    """
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _exp_so3(w: np.ndarray) -> np.ndarray:
    """SO(3) 指数映射(小角度稳定版,与 slam._rodrigues 同式)。"""
    th = float(np.linalg.norm(w))
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + (math.sin(th) / th) * K + ((1 - math.cos(th)) / th**2) * (K @ K)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40, help="运动段帧数")
    ap.add_argument("--settle", type=int, default=15, help="静止收敛帧数(用于估重力/陀螺零偏)")
    ap.add_argument("--speed", type=float, default=8.0, help="定速 m/s(0 = 静止)")
    ap.add_argument(
        "--turn",
        action="store_true",
        help="真实物理转向:固定油门+满舵绕圈(标定 gyro 轴向/尺度;直行时世界系与车体系重合,无法区分)",
    )
    ap.add_argument(
        "--throttle",
        type=float,
        default=None,
        help="改用固定油门直行(替代 set_target_velocity)——排查'每帧重发目标速度'是否引入陀螺伪角速度",
    )
    ap.add_argument("--accel-std", type=float, default=0.0, help="noise_accel_stddev_*(m/s²)")
    ap.add_argument("--gyro-std", type=float, default=0.0, help="noise_gyro_stddev_*(rad/s)")
    ap.add_argument("--gyro-bias", type=float, default=0.0, help="noise_gyro_bias_*(rad/s)")
    ap.add_argument("--seed", type=int, default=0, help="noise_seed")
    ap.add_argument("--map", default=None, help="切图(缺省=当前图)")
    ap.add_argument("--out", default="outputs/imu_probe", help="产物根(经 project_path)")
    ap.add_argument("--tag", default=None, help="产物文件名后缀(区分多次运行)")
    args = ap.parse_args()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)
    if args.map:
        world = client.load_world(args.map)
    else:
        world = client.get_world()
    sync_mode(world)  # 同步 + 补 tick(勿绕过)

    # 清场:残留 ego 会阻塞 spawn point
    for a in world.get_actors():
        if a.type_id.startswith("vehicle.") or a.type_id.startswith("sensor."):
            a.destroy()
    world.tick()

    ego = spawn_ego(world)
    ego.set_autopilot(False)
    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))  # 清制动残留

    bp_lib = world.get_blueprint_library()
    imu_bp = bp_lib.find("sensor.other.imu")
    imu_bp.set_attribute("noise_accel_stddev_x", str(args.accel_std))
    imu_bp.set_attribute("noise_accel_stddev_y", str(args.accel_std))
    imu_bp.set_attribute("noise_accel_stddev_z", str(args.accel_std))
    imu_bp.set_attribute("noise_gyro_stddev_x", str(args.gyro_std))
    imu_bp.set_attribute("noise_gyro_stddev_y", str(args.gyro_std))
    imu_bp.set_attribute("noise_gyro_stddev_z", str(args.gyro_std))
    imu_bp.set_attribute("noise_gyro_bias_x", str(args.gyro_bias))
    imu_bp.set_attribute("noise_gyro_bias_y", str(args.gyro_bias))
    imu_bp.set_attribute("noise_gyro_bias_z", str(args.gyro_bias))
    imu_bp.set_attribute("noise_seed", str(args.seed))
    imu = cast(carla.Sensor, world.spawn_actor(imu_bp, SENSOR_OFFSET, attach_to=ego))

    q: queue.Queue = queue.Queue()
    imu.listen(q.put)
    for _ in range(5):  # 预热
        world.tick()
        q.get(timeout=10)

    fwd = ego.get_transform().get_forward_vector()
    vel = carla.Vector3D(x=fwd.x * args.speed, y=fwd.y * args.speed, z=0.0)

    # 真值:位置 + 姿态(每帧 tick 后读,避免陈旧快照)
    gt_t, gt_pos, gt_rpy = [], [], []
    accs, gyros, stamps = [], [], []
    # actor 级自证:get_angular_velocity() 是**物理引擎**的角速度(deg/s),
    # get_velocity() 是线速度(m/s)——判"IMU 读数是否自洽于物理"的一线证据,
    # 不依赖 get_transform() 的欧拉角差分(后者对 <1e-3 deg/s 的量级不可分辨)。
    avs, vels = [], []
    # 静止段:不给速度指令,让车静态收敛(FAST-LIO2 的 IMU 初始化也是静止段估重力/零偏)。
    # **必须静止收敛后再估重力**:预热期 accel.z 从 9.05 爬到 9.81(车落地弹跳),
    # 拿首帧估重力会引入 ~0.7 m/s² 偏差 → 静止时也积分出 4.8 m 假位移。
    try:
        for i in range(args.settle + args.frames):
            moving = i >= args.settle
            if moving and args.turn:
                # 真实物理转向:固定油门 + 满舵绕圈。set_target_angular_velocity 实测
                # **不产生转动**(actor 角速度 ~1.6e-3 deg/s,陀螺恒 0,gt yaw 只动 0.1°)
                # ——该 API 在 CARLA 里不可用,勿再试。绕圈给陀螺一个**与世界系不同向**的
                # 角速度,才能区分 gyro 是车体系还是世界系。
                ego.apply_control(carla.VehicleControl(throttle=0.5, steer=1.0, brake=0.0, hand_brake=False))
            elif moving and args.throttle is not None:
                ego.apply_control(carla.VehicleControl(throttle=args.throttle, steer=0.0, brake=0.0))
            elif moving and args.speed:
                ego.set_target_velocity(vel)
            world.tick()
            m = q.get(timeout=10)
            t = ego.get_transform()
            gt_t.append(time.time())
            gt_pos.append((t.location.x, t.location.y, t.location.z))
            # 直接读 rotation 三轴 → 弧度,序为 (roll, pitch, yaw)(勿用 carla_common.rad,
            # 它的返回序是 (pitch, yaw, roll))
            gt_rpy.append(
                (
                    math.radians(t.rotation.roll),
                    math.radians(t.rotation.pitch),
                    math.radians(t.rotation.yaw),
                )
            )
            accs.append((m.accelerometer.x, m.accelerometer.y, m.accelerometer.z))
            gyros.append((m.gyroscope.x, m.gyroscope.y, m.gyroscope.z))
            av = ego.get_angular_velocity()
            lv = ego.get_velocity()
            avs.append((av.x, av.y, av.z))
            vels.append((lv.x, lv.y, lv.z))
            stamps.append(float(m.timestamp))
            if moving and (i - args.settle + 1) % 10 == 0:
                print(f"  [{i - args.settle + 1}/{args.frames}] ego {tuple(round(v, 2) for v in gt_pos[-1])}")
    finally:
        imu.stop()
        imu.destroy()
        ego.destroy()

    gt_pos = np.array(gt_pos, dtype=np.float64)
    gt_rpy = np.array(gt_rpy, dtype=np.float64)
    accs = np.array(accs, dtype=np.float64)
    gyros = np.array(gyros, dtype=np.float64)
    stamps = np.array(stamps, dtype=np.float64)
    avs = np.array(avs, dtype=np.float64)
    vels = np.array(vels, dtype=np.float64)
    n_all = len(gt_pos)
    if n_all < args.settle + 3:
        raise SystemExit(f"[abort] 帧数不足(需 > settle {args.settle} + 3)")

    # 静止段估重力/零偏(比力反号);积分只跑运动段,静止段仅用于初始化。
    # **取静止段尾部**:预热期 accel.z 从 ~9.05 爬到 9.81(车落地弹跳),拿首帧估重力
    # 会引入 ~0.7 m/s² 偏差 → 静止时也积分出 1.5 m 假位移(实测)。
    tail = max(3, args.settle // 3)
    g_est = -accs[args.settle - tail : args.settle].mean(0)
    gyro_bias = gyros[args.settle - tail : args.settle].mean(0)
    s = args.settle
    gt_pos, gt_rpy = gt_pos[s:], gt_rpy[s:]
    accs, gyros, stamps = accs[s:], gyros[s:], stamps[s:]
    avs, vels = avs[s:], vels[s:]
    gt_t = gt_t[s:]
    n = len(gt_pos)

    # 逐帧 dt:优先传感器时间戳,退化用墙钟
    dt_s = np.diff(stamps)
    dt_w = np.diff(gt_t)
    dt_src = "sensor.timestamp" if (np.isfinite(dt_s).all() and (dt_s > 0).all()) else "wall"
    dt = dt_s if dt_src == "sensor.timestamp" else dt_w
    dt = np.clip(dt, 1e-4, 1.0)

    # --- 纯积分(FAST-LIO2 get_f 同式)---
    # 初值取真值(只测"增量积分"能力,不叠加初始对齐误差)
    R = _rot_from_rpy(tuple(gt_rpy[0]))
    v = np.zeros(3)
    p = gt_pos[0].copy()
    g_used = f"measured(静止段末 {tail} 帧)"

    traj_p = [p.copy()]
    traj_R = [R.copy()]
    for k in range(n - 1):
        a_w = R @ accs[k] + g_est  # 世界系加速度(CARLA 只暴露陀螺零偏,加计零偏不估)
        v = v + a_w * dt[k]
        p = p + v * dt[k]
        R = R @ _exp_so3((gyros[k] - gyro_bias) * dt[k])
        traj_p.append(p.copy())
        traj_R.append(R.copy())
    traj_p = np.array(traj_p)
    traj_R = np.array(traj_R)

    # --- 误差 ---
    pos_err = np.linalg.norm(traj_p - gt_pos, axis=1)
    # 姿态误差:ΔR = R_gtᵀ R_est 的旋转角
    rot_err = []
    for k in range(n):
        dR = _rot_from_rpy(tuple(gt_rpy[k])).T @ traj_R[k]
        c = (np.trace(dR) - 1.0) / 2.0
        rot_err.append(math.degrees(math.acos(float(np.clip(c, -1.0, 1.0)))))
    rot_err = np.array(rot_err)

    # --- IMU 自洽性(不依赖位姿积分,直接问"陀螺读数 vs 物理角速度")---
    # CARLA IMU 的 gyroscope 是 rad/s,actor.get_angular_velocity() 是 deg/s → 先转弧度。
    # 两者独立测量同一物理量,比值应 ≈1。**定速直行时"世界系 vs 车体系"重合,无法区分**
    # (yaw≈0 → 两系同向),必须靠原地自转/转弯场景区分,见 spin90 运行。
    avs_rad = np.radians(avs)
    gyro_vs_av_ratio = np.divide(gyros, avs_rad, out=np.zeros_like(gyros), where=np.abs(avs_rad) > 1e-6)
    gyro_vs_av_ratio = np.divide(gyros, avs_rad, out=np.zeros_like(gyros), where=np.abs(avs_rad) > 1e-6)
    path_len = float(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1).sum())
    gt_end = float(np.linalg.norm(gt_pos[-1] - gt_pos[0]))

    # IMU 自洽性摘要(只用有量级的那几轴,避免除零噪声主导)
    big = np.abs(avs_rad).max(0) > 1e-4
    imu_consistency = {
        "actor_angvel_deg_s_mean": [round(float(x), 6) for x in avs.mean(0)],
        "gyro_rad_s_mean": [round(float(x), 6) for x in gyros.mean(0)],
        "ratio_gyro_over_actor_rad": [
            round(float(gyro_vs_av_ratio[:, k][np.abs(avs_rad[:, k]) > 1e-4].mean()), 6) if big[k] else None
            for k in range(3)
        ],
        "note": "ratio≈1 → gyroscope 与 actor 物理角速度同轴同尺度(rad/s)",
    }

    res = {
        "tag": args.tag,
        "map": world.get_map().name,
        "n_frames": n,
        "speed_set": args.speed,
        "turn": args.turn,
        "throttle": args.throttle,
        "noise": {
            "accel_std": args.accel_std,
            "gyro_std": args.gyro_std,
            "gyro_bias": args.gyro_bias,
            "seed": args.seed,
        },
        "dt_src": dt_src,
        "settle_frames": args.settle,
        "dt_median_s": round(float(np.median(dt)), 6),
        "gravity_used": g_used,
        "gravity_est": [round(float(x), 4) for x in g_est],
        "gyro_bias_est": [round(float(x), 7) for x in gyro_bias],
        "imu_consistency": imu_consistency,
        "path_len_m": round(path_len, 3),
        "gt_start_to_end_m": round(gt_end, 3),
        "pos_err": {
            "final_m": round(float(pos_err[-1]), 4),
            "mean_m": round(float(pos_err.mean()), 4),
            "max_m": round(float(pos_err.max()), 4),
            "per_100m_m": round(float(pos_err[-1] / max(path_len, 1e-6) * 100), 3),
        },
        "rot_err_deg": {
            "final": round(float(rot_err[-1]), 4),
            "mean": round(float(rot_err.mean()), 4),
            "max": round(float(rot_err.max()), 4),
        },
        "accel_mean": [round(float(x), 4) for x in accs.mean(0)],
        "accel_std_meas": [round(float(x), 4) for x in accs.std(0)],
        "gyro_mean": [round(float(x), 6) for x in gyros.mean(0)],
        "gyro_std_meas": [round(float(x), 6) for x in gyros.std(0)],
    }

    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    (out / f"imu_probe{suffix}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    np.savez(
        out / f"imu_probe{suffix}.npz",
        gt_pos=gt_pos,
        gt_rpy=gt_rpy,  # 真值姿态(弧度,roll/pitch/yaw)——判"IMU 与 GT 是否自洽"必需
        gt_t=np.array(gt_t, dtype=np.float64),  # 墙钟(退化 dt 源)
        traj_p=traj_p,
        accs=accs,
        gyros=gyros,
        avs=avs,  # actor 物理角速度(deg/s)——判"陀螺 vs 物理"必需
        vels=vels,  # actor 线速度(m/s)
        pos_err=pos_err,
        rot_err_deg=rot_err,
        dt=dt,
    )

    print(f"\n[imu ] {res['map']} | {n} 帧 | dt {res['dt_median_s']:.4f}s ({dt_src}) | 重力 {g_used}")
    print(f"[meas] accel mean {res['accel_mean']} std {res['accel_std_meas']}")
    print(f"[meas] gyro  mean {res['gyro_mean']} std {res['gyro_std_meas']}")
    print(f"[cons] actor 角速度 mean {imu_consistency['actor_angvel_deg_s_mean']} deg/s")
    print(f"[cons] gyro/actor 比值   {imu_consistency['ratio_gyro_over_actor_rad']}")
    print(f"[gt  ] 路径 {path_len:.2f} m | 首末 {gt_end:.2f} m")
    print(f"[err ] 位置 末 {res['pos_err']['final_m']:.4f} m / 均 {res['pos_err']['mean_m']:.4f} m")
    print(f"[err ] 姿态 末 {res['rot_err_deg']['final']:.4f}° / 均 {res['rot_err_deg']['mean']:.4f}°")
    print(f"→ {out}/imu_probe{suffix}.json")


if __name__ == "__main__":
    main()

"""雷达**速度对表**:径向速度 vs 自车运动(纯值,零 carla)。

## 为什么这条判据重要:它**不需要目标**

前面所有雷达判据都要"目标"才能问("回波落在车上了吗"),而实测那一路在静止车上是 **0**
(§P-V17 六)。于是雷达的可判性一直悬着。

**速度不一样**:静止场景里,每个回波的径向速度**必然**等于自车速度在该视线方向上的投影
—— 这条与"打在什么上"无关。所以它能回答一个更基本的问题:

> **雷达的速度通道有没有物理意义?**

有 ⇒ 即使它出不了目标,它也是一路可用的**自运动/静止性**信号;
没有 ⇒ 这一路连"能判的量"都没有,该从融合里彻底拿掉。

## 口径(两块都**踩过坑才定下来**的)

**① 不假设自车速度沿传感器自己的 x 轴。** 那个假设只对**前置**雷达成立;±90° 的侧雷达,
视线与自车前进方向**垂直**,预期径向速度根本不是 `s·u_x`。用那个模型去判,会把
**模型错**记成"两个侧雷达坏了"。⇒ 改成同时拟合两个系数:

    vel ≈ s · (k_x·u_x + k_y·u_y)

`‖(k_x,k_y)‖` 应当 ≈ 1(回波速度复现了自车速度在该方向的投影);
它的**方向** = 自车前进方向在传感器系里的朝向,可与**声明挂点偏航**对表。

**② R² 必须中心化。** 雷达 FOV 只有 ±38° ⇒ `u_x ∈ [0.79, 1.0]`,自车速度又恒定
⇒ `v` 与 `e` 都是**近似常数**(mean ±7.7,std 仅 0.4)。"绕原点"的 R² 被**均值**主导、
恒 ≈0.999,**连打乱对照也塌不下来**。中心化之后量的才是"解释了多少**方差**"。

## 自证(每次运行都跑)

**打乱对照**:把实测速度随机重排,R² 必须**塌掉**(实测塌到**负** —— 比"用均值"还差)。
没有它,R² 高只能说明"两条曲线都平滑",说明不了它们对得上。

## 判据(三条同时满足)

`|k| ≈ 1` ∧ `R² 高且对照塌` ∧ `推出朝向 ≈ 声明朝向`,最后一条**允许一个全局符号**
(`min(|Δ|, 180−|Δ|)`)—— `vel`「朝传感器为正」的约定让推出朝向整体差 180°,
那是**一个常数**,折掉之后剩下的才是"这一路自己的挂点朝向对不对"。

**实测(`kitti_ab_occl2_none_all`,40 帧、自车 8.01 m/s):达标 3/5。**
FRONT / BACK_LEFT / BACK_RIGHT:`|k|` 0.990–0.993、R² 0.70、对照 −0.27~−0.60。
FRONT_LEFT / FRONT_RIGHT:`|k|` **0.018–0.029**、R² **≈0** —— **这两个通道的速度读数是 0**,
而自车在 8 m/s 前进。⇒ 雷达的速度通道**部分可用**,不是全坏也不是全好。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from autodrivedata.gt.export.nuscenes import NUS_RADAR_CHANNELS
from autodrivedata.map.bev_base import load_radar_nus
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 自车速度低于此值(m/s)的帧**不参与** —— 静止时预期速度恒 0,回归无信息。
MIN_EGO_SPEED = 1.0


def ego_pose(root: Path, fid: int) -> np.ndarray | None:
    """`training/pose/{fid}.txt` → 4×4(3×4 行主序 + 末行)。"""
    p = root / "training/pose" / f"{fid:06d}.txt"
    if not p.exists():
        return None
    vals = [float(x) for ln in p.read_text().splitlines() for x in ln.split()]
    if len(vals) < 12:
        return None
    m = np.eye(4)
    m[:3, :4] = np.array(vals[:12]).reshape(3, 4)
    return m


def ego_speed(root: Path, fid: int, dt: float = 0.1) -> float | None:
    """相邻两帧位姿平移差 → 实测速度(m/s)。

    ⚠️ **不读命令值** —— 本项目红线:「定速必须逐帧自证」(制动残留会把 8.0 打成 6.6)。
    """
    a, b = ego_pose(root, fid), ego_pose(root, fid + 1)
    if a is None or b is None:
        return None
    return float(np.linalg.norm(b[:3, 3] - a[:3, 3]) / dt)


def radial_pairs(root: Path, fid: int, ch: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一帧一通道 → `(u_x, vel_measured, vel_expected)`。

    `vel_measured` 从 18 字段的 `vx` 还原:`vx = −vel·cos e·cos a`(见 `radar.detections_to_nus18`)。
    不重解原始 BGR 二进制 —— 落盘的 18 字段已经是同一条链的产物,再解一次只多一处会漂的口径。
    """
    arr = load_radar_nus(root, fid, ch)
    if not len(arr):
        return np.zeros(0), np.zeros(0), np.zeros(0)
    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
    d = np.sqrt(x * x + y * y + z * z)
    ok = d > 1e-6
    x, y, z, d = x[ok], y[ok], z[ok], d[ok]
    ux, uy = x / d, y / d
    vx = arr[ok, 6]
    with np.errstate(divide="ignore", invalid="ignore"):
        vel = np.where(np.abs(ux) > 1e-3, -vx / np.maximum(ux, 1e-6), np.nan)
    s = ego_speed(root, fid)
    if s is None:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    keep = np.isfinite(vel)
    return np.stack([ux[keep], uy[keep]], 1), vel[keep], np.full(int(keep.sum()), s)


def eval_root(root: Path, frames: list[int], channels: tuple[str, ...] = NUS_RADAR_CHANNELS) -> dict:
    """逐通道拟合 `vel ≈ s·(k_x·u_x + k_y·u_y)` + **打乱对照**。

    ★ **不假设自车速度沿传感器自己的 x 轴**(2026-10-02 踩到):那个假设只对**前置**雷达成立;
    ±90° 的侧雷达,其视线与自车前进方向**垂直**,预期径向速度根本不是 `s·u_x` ——
    用那个模型去判,三个通道会"达标"、两个侧通道会"不达标",而**那两个是模型错,不是雷达错**。
    改成同时拟合两个系数:`‖(k_x,k_y)‖` 应当 ≈ 1(即 `vel` 复现了自车速度在该方向的投影),
    而它的**方向**给出"自车前进方向在传感器系里的朝向",可与声明挂点偏航对表。
    """
    rng = np.random.default_rng(0)
    from autodrivedata.gt.export.nuscenes import radar_yaw_offset_carla

    out: dict[str, dict] = {}
    for ch in channels:
        us, vs, ss = [], [], []
        for fid in frames:
            a, b, c = radial_pairs(root, fid, ch)
            if len(a):
                us.append(a)
                vs.append(b)
                ss.append(c)
        if not us:
            out[ch] = {"n": 0}
            continue
        u = np.concatenate(us)
        v = np.concatenate(vs)
        s_all = np.concatenate(ss)
        # 设计矩阵 = s·u(**同时**吃 u_x 与 u_y),过原点最小二乘
        A = u * s_all[:, None]
        k, *_ = np.linalg.lstsq(A, v, rcond=None)
        pred = A @ k
        ss_res = float(((v - pred) ** 2).sum())
        ss_tot = float(((v - v.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        shuf = v.copy()
        rng.shuffle(shuf)
        k_s, *_ = np.linalg.lstsq(A, shuf, rcond=None)
        ss_res_s = float(((shuf - A @ k_s) ** 2).sum())
        r2_s = 1.0 - ss_res_s / ss_tot if ss_tot > 0 else float("nan")
        speed_ratio = float(np.linalg.norm(k))
        implied_yaw = float(np.degrees(np.arctan2(k[1], k[0])))
        declared = float(radar_yaw_offset_carla(ch))
        diff = (implied_yaw - declared + 180.0) % 360.0 - 180.0
        out[ch] = {
            "n": int(len(v)),
            "k": [round(float(k[0]), 4), round(float(k[1]), 4)],
            "speed_ratio": round(speed_ratio, 3),
            "r2": round(r2, 4),
            "r2_shuffled": round(r2_s, 4),
            "implied_yaw_deg": round(implied_yaw, 1),
            "declared_yaw_deg": round(declared, 1),
            "yaw_diff_deg": round(diff, 1),
        }
    return out


def _ok(r: dict) -> bool:
    """单通道是否达标。**提到模块级**而不是埋在 `main()` 里 —— 它是纯值判据,
    埋在 CLI 里就**测不到**(而"判据自己没测"正是本项目反复付代价的那类欠账)。

    三条同时满足:

    ① `|k| ≈ 1` —— 回波速度复现了自车速度的**量值**;
    ② `R² 高 **且** 打乱对照塌` —— 两条曲线对得上,而不只是"都平滑";
    ③ 推出朝向 ≈ 声明朝向 —— **允许一个全局符号**:`vel`「朝传感器为正」的约定
       让推出朝向整体差 180°,那是**一个常数**、不是五路各自的错;
       `min(|Δ|, 180−|Δ|)` 折掉符号之后,剩下的才是"这一路自己的挂点朝向对不对"。
    """
    if not r.get("n"):
        return False
    d = abs(r["yaw_diff_deg"])
    return abs(r["speed_ratio"] - 1.0) < 0.2 and r["r2"] > r["r2_shuffled"] + 0.3 and min(d, 180.0 - d) < 15.0


def _parse_frames(spec: str) -> list[int]:
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="含 samples/RADAR_* 与 training/pose/ 的 KITTI root")
    ap.add_argument("--frames", default="0-39")
    args = ap.parse_args()

    with runlog.run("autodrivedata.perception.radar_eval") as rl:
        root = project_path(args.root)
        rl.input(str(root), "root")
        frames = _parse_frames(args.frames)
        speeds = [s for s in (ego_speed(root, f) for f in frames) if s is not None]
        if not speeds:
            raise SystemExit(f"{root} 没有 training/pose/ —— 这条判据要自车运动才算得了")
        print(f"[data] {root.name} | {len(frames)} 帧 | 自车实测速度 中位 {np.median(speeds):.2f} m/s")
        if np.median(speeds) < MIN_EGO_SPEED:
            raise SystemExit(f"自车几乎不动({np.median(speeds):.2f} m/s)—— 静止时预期径向速度恒 0,回归无信息")

        res = eval_root(root, frames)
        print()
        print(
            f"{'通道':<18}{'点数':>7}{'|k|':>7}{'R²':>8}{'打乱':>8}{'推出朝向':>10}{'声明朝向':>10}{'差':>8}"
        )
        for ch, r in res.items():
            if not r.get("n"):
                print(f"{ch:<18}{0:>7}   (无回波)")
                continue
            print(
                f"{ch:<18}{r['n']:>7}{r['speed_ratio']:>7.3f}{r['r2']:>8.3f}{r['r2_shuffled']:>8.3f}"
                f"{r['implied_yaw_deg']:>10.1f}{r['declared_yaw_deg']:>10.1f}{r['yaw_diff_deg']:>8.1f}"
            )
            rl.highlight(f"{ch}/speed_ratio", r["speed_ratio"])
            rl.highlight(f"{ch}/r2", r["r2"])
            rl.highlight(f"{ch}/yaw_diff_deg", r["yaw_diff_deg"])

        good = [c for c, r in res.items() if _ok(r)]
        print()
        print(
            f"[判据] `|k|` 应 ≈1、R² 应高且**打乱对照必须塌**、推出朝向应 ≈ 声明朝向。"
            f"(朝向判据允许一个全局符号)达标通道 {len(good)}/{len(res)}:{good}"
        )
        print(
            "        ⚠️ 这条量的是**速度通道的物理意义**,与「打不打得到目标」无关 ——"
            " 所以它能在「回波 0 命中」的情况下照样给结论。"
        )
        rl.highlight("n_channels_ok", len(good))


if __name__ == "__main__":
    main()

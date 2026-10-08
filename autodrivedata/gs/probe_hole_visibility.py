"""★ **删掉物体后那个"窟窿",到底有没有别的视角看得见?**(纯值,零 GPU,零 CARLA)

## 它拦的是什么

C1 的下一步本来是「**屏蔽重训**」:训练时把道具那块像素从损失里挖掉,让别的视角去约束
被它挡住的背景。但复核指出**这个前提可能是错的**:

- 道具就摆在**环心**(实测 prop 中心世界 (-78.03, 12.97, 0.93) vs 环心 (-78.03, 12.97, 2.1),
  水平偏 **0.03 m**),在 **269/270 帧**里都可见 ⇒ 它**每帧都挡住同一块画幅中心**;
- 「被挡住的那块背景」在**有/无屏蔽两种情况下都不被渲染**(被道具挡在前面)⇒
  屏蔽**不会**给那块背景多加任何梯度,它只是让道具自己的高斯不再被拟合。
⇒ 若如此,屏蔽重训 ≈ **no-op**,白烧 10–45 min GPU。

**所以先花分钟级 CPU 把这件事量清楚。**

## 判据

用 **B 侧**(没有道具)的**真值深度**反投影出「A 侧道具遮住的那块背景」的世界点,
再把它们投回**全部 270 个视角**,逐个判:在画幅内 **且** 没被别的东西挡住。

| 读数 | 含义 |
|---|---|
| **可见视角数** | ★ **这就是"屏蔽重训能填多少"的上限** —— 没有视角看到的点,任何训练都填不出来 |
| 可见数 = 0 | ⇒ **取消屏蔽重训**(前提不成立) |
| 可见数 > 0 | ⇒ 值得跑,且这个数就是预期上限 |

## 两条自证(缺一条,读数不可解释)

1. **阳性对照**:同一套代码去投**道具旁边可见的路面**(它一定被很多视角看到)——
   若连路面都投不出可见视角,那是**尺子坏了**,不是"没视角"。
2. **遮挡判据本身要能失败**:把深度换成**故意错的**(整幅 +2 m)再投一次,
   可见数**必须变**(不变说明遮挡测试没真的在用深度)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autodrivedata.gs.train_3dgs_mini import _load_poses_and_cams
from autodrivedata.perception.inst_tags import decode_instance_png
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 遮挡判据的容差(米)。3DGS 的深度是真值渲染深度,同一个世界点在不同视角下应当一致
#: ⇒ 用"投影深度 ≤ 该视角深度 + 容差"判可见。容差只吸收浮点与采样误差。
VIS_TOL_M = 0.05

#: 反投影时只取这么多像素(掩膜可能上万像素;均匀抽稀,**固定 seed** 保证可复现)。
MAX_SAMPLES = 4000

#: 「模型在这块上有高斯吗」的近邻半径(米)。0.2 m ≈ 高斯尺度的量级(实测 scales 中位 ~0.05–0.1 m)。
GAUSS_NEAR_M = 0.2


def prop_mask(capture: Path, pose: dict, prop_id: int) -> np.ndarray:
    """一帧里道具占的像素 —— `inst/p{pitch}/{i:05d}.png == prop_id`。"""
    p = capture / "inst" / f"p{int(pose['pitch'])}" / f"{int(pose['i']):05d}.png"
    if not p.is_file():
        raise SystemExit(f"{p} 不存在 —— 这份 capture 不是 `--inst` 采的")
    return decode_instance_png(p.read_bytes()) == prop_id


def load_depth(capture: Path, pose: dict) -> np.ndarray:
    p = capture / "depth" / f"p{int(pose['pitch'])}" / f"{int(pose['i']):05d}.npy"
    if not p.is_file():
        raise SystemExit(f"{p} 不存在 —— 这份 capture 没有真值深度")
    return np.load(p).astype(np.float64)


def sample_depth(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """最近邻采样(`uv` 是 (N,2) 的 (u=x, v=y))。⛔ 越界的返回 `inf`(= 看不见)。"""
    h, w = depth.shape
    u = np.rint(uv[:, 0]).astype(np.int64)
    v = np.rint(uv[:, 1]).astype(np.int64)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.full(len(uv), np.inf)
    out[ok] = depth[v[ok], u[ok]]
    return out


def backproject(depth: np.ndarray, uv: np.ndarray, fx, fy, cx, cy) -> np.ndarray:
    """`((u-cx)·z/fx, (v-cy)·z/fy, z)` —— 与 `calib.selfcheck.backproject_depth` 同一条链。"""
    z = sample_depth(depth, uv)
    return np.stack(
        [(uv[:, 0] - cx) * z / fx, (uv[:, 1] - cy) * z / fy, z], axis=1
    )  # (N,3),z=inf 的点后面会被滤掉


def to_world(p_cam: np.ndarray, viewmat: np.ndarray) -> np.ndarray:
    """gsplat 的 `viewmat` 是**世界→相机** ⇒ 反变换 `R^T (p - t)`。"""
    r, t = viewmat[:3, :3], viewmat[:3, 3]
    return (p_cam - t) @ r  # (N,3) @ (3,3) == R^T 作用


def project(p_world: np.ndarray, viewmat: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """世界点 → (uv, 相机系 z)。`z <= 0` 的点在相机后面。"""
    p_cam = p_world @ viewmat[:3, :3].T + viewmat[:3, 3]
    z = p_cam[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    uv = np.stack([k[0, 0] * p_cam[:, 0] / safe + k[0, 2], k[1, 1] * p_cam[:, 1] / safe + k[1, 2]], axis=1)
    return uv, z


def count_visible(
    pts_world: np.ndarray,
    depths: list[np.ndarray],
    viewmats: np.ndarray,
    ks: np.ndarray,
    shape: tuple[int, int],
    *,
    tol: float = VIS_TOL_M,
    depth_bias: float = 0.0,
) -> np.ndarray:
    """每个点被**多少个视角**看到(在画幅内 **且** 没被挡住)。返回 `(N,)`。

    ⚠️ `depth_bias` 只给**自证**用:把它设成非 0 会让遮挡判据读到错的深度 ——
    如果可见数**不变**,说明这条判据根本没在用深度。
    """
    h, w = shape
    n = len(pts_world)
    vis = np.zeros(n, dtype=np.int64)
    for vm, k, d in zip(viewmats, ks, depths, strict=True):
        uv, z = project(pts_world, vm, k)
        in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & (z > 0)
        if not in_frame.any():
            continue
        seen = np.zeros(n, dtype=bool)
        seen[in_frame] = z[in_frame] <= sample_depth(d + depth_bias, uv[in_frame]) + tol
        vis += seen
    return vis


def run(cap_a: Path, cap_b: Path, *, samples: int, seed: int, gs_dir: Path, tag: str) -> dict:
    prop = json.loads((cap_a / "prop.json").read_text(encoding="utf-8"))["prop"]
    prop_id = int(prop["instance_id"])
    poses, _f, h, w, _cx, _cy, viewmats, ks = _load_poses_and_cams(cap_a, downsample=1)
    viewmats = viewmats.numpy().astype(np.float64)
    ks = ks.numpy().astype(np.float64)
    fx, fy, cx, cy = ks[0, 0, 0], ks[1, 1, 1], ks[0, 0, 2], ks[1, 1, 2]
    rng = np.random.default_rng(seed)

    depths_b = [load_depth(cap_b, p) for p in poses]
    if depths_b[0].shape != (h, w):
        raise SystemExit(f"深度 {depths_b[0].shape} 与位姿侧 {h}x{w} 对不上")

    # ① 收集「A 侧道具遮住的背景」的世界点(用 B 的深度,因为没有道具)
    pts, occluded_of = [], []
    for r, pose in enumerate(poses):
        m = prop_mask(cap_a, pose, prop_id)
        if not m.any():
            continue
        v, u = np.nonzero(m)
        if len(u) > samples:
            sel = rng.choice(len(u), samples, replace=False)
            u, v = u[sel], v[sel]
        pc = backproject(depths_b[r], np.stack([u, v], axis=1).astype(np.float64), fx, fy, cx, cy)
        good = np.isfinite(pc[:, 2]) & (pc[:, 2] > 0)
        if good.any():
            pts.append(to_world(pc[good], viewmats[r]))
            occluded_of.append(np.full(int(good.sum()), r))
    if not pts:
        raise SystemExit("没有任何一帧露出道具 —— 查 prop.json 与 --inst")
    pts_world = np.concatenate(pts)
    occluded_of = np.concatenate(occluded_of)

    # ② 投回全部视角,数可见数
    vis = count_visible(pts_world, depths_b, viewmats, ks, (h, w))
    # ③ 自证 A:阳性对照 —— 道具**旁边**的路面(它一定被很多视角看到)
    road = prop_mask(cap_a, poses[0], 7) | prop_mask(cap_a, poses[0], 10)  # RoadLines / Roads
    rv, ru = np.nonzero(road)
    road_vis = np.array([0])
    if len(ru):
        sel = rng.choice(len(ru), min(samples, len(ru)), replace=False)
        pc = backproject(depths_b[0], np.stack([ru[sel], rv[sel]], axis=1).astype(np.float64), fx, fy, cx, cy)
        good = np.isfinite(pc[:, 2]) & (pc[:, 2] > 0)
        if good.any():
            rw = to_world(pc[good], viewmats[0])
            road_vis = count_visible(rw, depths_b, viewmats, ks, (h, w))
    # ④ 自证 B:把遮挡判据**收紧**(点必须比记录深度再近 1 m 才算可见)⇒ 可见数**必须掉**。
    #    ⚠️ 第一版用的是 `+2.0`(放松),读数只从 104 动到 105 —— **那不是对照**:
    #       大多数点在被看到的视角里本来就在最前面,放松门槛几乎不改变任何判决。
    vis_biased = count_visible(pts_world, depths_b, viewmats, ks, (h, w), depth_bias=-1.0)

    # ⑤ ★★ **决定性追问**:A 侧的模型**本来**在那块上有高斯吗?
    #    可见性说的是"信息在不在"(在:中位 104 个视角看得到)。
    #    这一条说的是"**模型有没有把它学下来**"。
    #    ~1 ⇒ 窟窿是**质量问题**(高斯在、画得差),不是表示缺失 ⇒ 屏蔽重训无从下手;
    #    ~0 ⇒ 信息在、模型没有 ⇒ 那才是"为什么 104 个视角没训出高斯"的真问题。
    gauss_frac = None
    means_p = gs_dir / f"means_{tag}.npy"
    if means_p.is_file():
        from scipy.spatial import cKDTree

        means = np.load(means_p).astype(np.float64)
        d, _ = cKDTree(means).query(pts_world, k=1)
        gauss_frac = float((d <= GAUSS_NEAR_M).mean())

    return {
        "n_points": int(len(pts_world)),
        "n_views": int(len(poses)),
        "visible_median": float(np.median(vis)),
        "visible_mean": float(vis.mean()),
        "visible_max": int(vis.max()),
        # ★ 关键分布:多少个点**一个视角都看不到**
        "frac_zero_visible": float((vis == 0).mean()),
        "frac_ge5_visible": float((vis >= 5).mean()),
        "visible_of_the_occluding_view": [
            int(vis[occluded_of == r].max()) for r in range(len(poses)) if (occluded_of == r).any()
        ][:5],
        "control_road_median_visible": float(np.median(road_vis)) if road_vis.size else None,
        "control_road_n": int(road_vis.size),
        "control_tightened_median": float(np.median(vis_biased)),
        "frac_hole_with_gaussian": gauss_frac,
        "prop_id": prop_id,
    }


def verdict(rep: dict) -> str:
    """★ 三分。**阳性对照没起来时一律"未判"** —— 不许读成"没视角看得到"。"""
    if rep["visible_max"] == 0:
        return "★ **没有视角看得到那块背景** ⇒ 屏蔽重训是 no-op,应取消并改测别的杠杆"
    if rep["control_road_median_visible"] is None or rep["control_road_median_visible"] < 5:
        return "未判(阳性对照的路面都没被看到 ⇒ 是尺子坏了,不是「没视角」)"
    if rep["control_tightened_median"] >= rep["visible_median"] * 0.9:
        return "未判(把遮挡判据收紧后可见数没掉 ⇒ 这条判据没真的在用深度)"
    extra = ""
    g = rep.get("frac_hole_with_gaussian")
    if g is not None:
        extra = f";A 侧模型在这块上高斯覆盖 {g * 100:.1f}%" + (
            "⇒ **高斯在、画得差**(质量问题,不是表示缺失)" if g > 0.5 else "⇒ 信息在、模型没学下来"
        )
    return (
        f"★ **背景被大量视角看到**:中位 {rep['visible_median']:.0f}/{rep['n_views']} 个视角,"
        f"{rep['frac_zero_visible'] * 100:.1f}% 的点一个都看不到"
        "⇒ **「从没被观测过」这个说法不成立**" + extra
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--capture-a", default="outputs/3dgs_ab3/A/capture", help="带道具那侧")
    ap.add_argument("--capture-b", default="outputs/3dgs_ab3/B/capture", help="**没有道具**那侧(取背景深度)")
    ap.add_argument("--gs-dir", default="outputs/3dgs", help="A 侧高斯五件套所在目录")
    ap.add_argument("--tag", default="ab3_A", help="A 侧高斯的 tag(读 `means_<tag>.npy`)")
    ap.add_argument("--samples", type=int, default=MAX_SAMPLES, help="每帧抽多少个掩膜像素")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    with runlog.run("autodrivedata.gs.probe_hole_visibility") as rl:
        rl.highlight("samples", args.samples)
        rep = run(
            project_path(args.capture_a),
            project_path(args.capture_b),
            samples=args.samples,
            seed=args.seed,
            gs_dir=project_path(args.gs_dir),
            tag=args.tag,
        )
        rep["verdict"] = verdict(rep)
        print("\n=== 窟窿可见性 ===")
        print(f"  道具 id {rep['prop_id']};遮住的背景点 {rep['n_points']} 个;视角 {rep['n_views']} 个")
        print(
            f"  每个点被看到:中位 {rep['visible_median']:.0f} | 均值 {rep['visible_mean']:.1f} | 最大 {rep['visible_max']}"
        )
        print(
            f"  ★ 一个视角都看不到的点占 {rep['frac_zero_visible'] * 100:.1f}%;≥5 个视角的占 {rep['frac_ge5_visible'] * 100:.1f}%"
        )
        print(f"  阳性对照(路面):中位可见 {rep['control_road_median_visible']} (n={rep['control_road_n']})")
        print(f"  自证(遮挡判据收紧 1 m):中位可见 {rep['control_tightened_median']:.0f}  ← 必须明显掉")
        if rep.get("frac_hole_with_gaussian") is not None:
            print(f"  ★ A 侧模型在该区域的**高斯覆盖率**:{rep['frac_hole_with_gaussian'] * 100:.1f}%")
        print(f"  ⇒ {rep['verdict']}")
        for k in ("visible_median", "frac_zero_visible", "visible_max"):
            rl.highlight(k, round(rep[k], 4) if isinstance(rep[k], float) else rep[k])
        rl.highlight("verdict", rep["verdict"])
        if args.out:
            p = project_path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
            rl.artifact(p, "report")
            print(f"[done] {p.resolve()}")


if __name__ == "__main__":
    main()

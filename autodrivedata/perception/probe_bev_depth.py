"""**BEV 障碍物通道那三条对照** —— 房子规矩:新判据先证明它在该失败时会失败。

`sem_eval` 现在对 `obstacle` 走**深度反投影**、对 `drivable`/`lane` 保留地平面 IPM。
读数上去了(障碍物 BEV IoU **0.688**,而旧路是**结构性恒空**)—— 但"上去了"不等于"上对了"。

| # | 对照 | 期望 | 它在防什么 |
|---|---|---|---|
| **①** | `drivable`(确实贴地)同时走**两条路** | 两个 BEV 图**高度重合** | 深度错位 / 解码错(z 当成射线距离)/ 通道取反 —— 这些都会让两条路**分家**,且**边缘像素最敏感** |
| **②** | 把深度换成**另一路相机的** | 障碍物团**必须明显移位** | "深度根本没进链路" —— 换错了却纹丝不动,说明那一路是摆设 |
| **③** | 障碍物点的**离地高度** | 应当 ≈ **车身高**(1–2 m) | ★ **最硬的一条**:地平面假设**按定义**给出高度 0。深度对 ⇒ 点落在车身上;深度错(或退回地平面)⇒ 塌到 0 |

⚠️ 三条**缺一不可**:① 单看会漏"深度整体偏但自洽";② 单看会漏"换了但都错";
③ 是唯一一条**绝对**的(不依赖另一条路)。
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from autodrivedata.map.mapviz import BEV_X, BEV_Y, cam_pose
from autodrivedata.perception.sem_bev import (
    BEV_PX,
    GROUND_Z_OFF,
    init_camera,
    mask_to_bev,
    mask_to_bev_depth,
)
from autodrivedata.perception.sem_tags import decode_tag_png, masks_from_tags
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

CAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else float("nan")


def load_depth(root, cam: str, token: str) -> np.ndarray:
    p = root / f"depth_{cam.lower()}" / f"{token}.npy"
    if not p.is_file():
        raise SystemExit(f"{p} 不存在 —— 这份 root 没带 `--depth`")
    return np.frombuffer(p.read_bytes(), dtype=np.uint16)


def obstacle_points_depth(mask, depth_mm, world_cam, intrinsics, ego, shape) -> np.ndarray:
    """障碍物像素 → **世界点**;②③ 都要用点本身,不只是栅格化后的图。

    ⚠️ 与 `mask_to_bev_depth` **同一条链**(同样先把毫米换成米、同样按光轴 z 反投影)。
    这里再写一遍是因为要拿**点**而不是图;两处的 z 语义必须一致,否则③的高度会系统性偏。
    """
    from autodrivedata.perception.sem_bev import camera_rotation_world_to_cam

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 3))
    h, w = depth_mm.shape
    z = depth_mm.reshape(h, w)[ys, xs].astype(np.float64) / 1000.0
    ok = np.isfinite(z) & (z > 0)
    if not ok.any():
        return np.zeros((0, 3))
    xs, ys, z = xs[ok], ys[ok], z[ok]
    loc, rot = world_cam
    pc = np.stack(
        [
            (xs - intrinsics.cx) * z / intrinsics.fx,
            (ys - intrinsics.cy) * z / intrinsics.fy,
            z,
        ],
        axis=1,
    )
    return pc @ camera_rotation_world_to_cam(rot) + np.asarray(loc, dtype=np.float64)[None, :]


def run(root, frames: list[int]) -> dict:
    calib = json.loads((root / "calib.json").read_text())
    if not calib.get("depth"):
        raise SystemExit(f"{root} 的 calib.json 没有 `depth` 键 —— 不是 `--depth` 采的")
    ego_poses = json.loads((root / "ego_pose.json").read_text())
    cams = [c for c in CAMS if (root / f"sem_{c.lower()}").is_dir()]
    bshape = (int((BEV_Y[1] - BEV_Y[0]) / BEV_PX), int((BEV_X[1] - BEV_X[0]) / BEV_PX))

    rows = []
    for fid in frames:
        ep = ego_poses[fid]
        ego = [ep["x"], ep["y"], ep["z"], ep["yaw"], ep["pitch"], ep["roll"]]
        ground_z = ego[2] - GROUND_Z_OFF
        for cam in cams:
            token = f"{fid:06d}"
            tag_p = root / f"sem_{cam.lower()}" / f"{token}.png"
            if not tag_p.is_file():
                continue
            tag = decode_tag_png(tag_p.read_bytes())
            intrinsics, se = init_camera(calib[cam], (tag.shape[1], tag.shape[0]))
            world_cam = cam_pose(ego, se)
            gt = masks_from_tags(tag)
            d = load_depth(root, cam, token).reshape(tag.shape)
            # 对照②:换**另一路相机**的深度(错位透视)
            other = "CAM_BACK" if cam != "CAM_BACK" else "CAM_FRONT"
            d_other = load_depth(root, other, token).reshape(tag.shape)

            # ① drivable:两条路必须重合
            bev_plane = mask_to_bev(gt["drivable"], world_cam, intrinsics, ground_z, ego, bshape)
            bev_dep, _ = mask_to_bev_depth(gt["drivable"], d, world_cam, intrinsics, ego, bshape)
            # ② obstacle:真深度 vs 错深度
            ob_ok, _ = mask_to_bev_depth(gt["obstacle"], d, world_cam, intrinsics, ego, bshape)
            ob_bad, _ = mask_to_bev_depth(gt["obstacle"], d_other, world_cam, intrinsics, ego, bshape)
            # ③ 障碍物点的离地高度
            pts = obstacle_points_depth(gt["obstacle"], d, world_cam, intrinsics, ego, bshape)
            rows.append(
                {
                    "fid": int(fid),
                    "cam": cam,
                    "control1_drivable_iou": _iou(bev_plane, bev_dep),
                    "control2_obstacle_shift": _iou(ob_ok, ob_bad),
                    "control3_height_median_m": float(np.median(pts[:, 2]) - ground_z) if len(pts) else None,
                    "n_obstacle_pts": int(len(pts)),
                }
            )
    if not rows:
        raise SystemExit("没有可判的帧/相机")

    def med(k):
        """⚠️ 用 **nanmedian** 并**先剔 nan** —— 两条路都空时 `_iou` 给 nan,
        而 `np.median([...nan...])` 会**整个变 nan**,`nan > 0.3` 又是 **False**
        ⇒ nan **静默通过**判据(实测踩到:对照②当时就是 nan 却判"全过")。"""
        v = [r[k] for r in rows if r[k] is not None and np.isfinite(r[k])]
        return float(np.median(v)) if v else None

    return {
        "n": len(rows),
        "control1_drivable_iou_median": med("control1_drivable_iou"),
        "control2_obstacle_shift_median": med("control2_obstacle_shift"),
        "control3_height_median_m": med("control3_height_median_m"),
        "per_row": rows,
    }


#: ① 的门槛。**为什么不是"应当几乎重合"**:drivable 一直铺到天边,而地平面只是一个
#: 名义平面(`ego_z−0.5`),远处几度的差在 BEV 上就是几十像素 ⇒ 远场天然分家。
#: 0.5 是"近场主导"的下限,不是"两条路应当一模一样"。
CONTROL1_MIN_IOU = 0.5


def verdict(rep: dict) -> str:
    c1, c2, c3 = (
        rep["control1_drivable_iou_median"],
        rep["control2_obstacle_shift_median"],
        rep["control3_height_median_m"],
    )
    bad = []
    # ⚠️ 每一格都把 `None`(全 nan ⇒ 一条可判的都没有)单独算**未判**,不许落进"没触发就通过"
    if c1 is None:
        bad.append("① **未判**(一条可判的都没有)")
    elif c1 < CONTROL1_MIN_IOU:
        bad.append(f"① 接地类两条路只重合 {c1:.3f}(深度与地平面**分家了**)")
    if c2 is None:
        bad.append("② **未判**(障碍物团全程为空 ⇒ 这条对照没被检验过)")
    elif c2 > 0.3:
        bad.append(f"② 换成别路深度后障碍物团还重合 {c2:.3f}(**深度可能没进链路**)")
    if c3 is None:
        bad.append("③ **未判**(没有障碍物点)")
    elif not (0.5 <= c3 <= 3.0):
        bad.append(f"③ 障碍物离地 {c3:.2f} m(地平面假设会给 0,车身高应 1–2 m)")
    if bad:
        return "★ **对照未过**: " + ";".join(bad)
    return (
        f"★ **三条对照全过**:① 接地类两条路重合 **{c1:.3f}**;"
        f"② 换深度后障碍物团只剩 **{c2:.3f}**(真的在用它);"
        f"③ 障碍物离地中位 **{c3:.2f} m**(车身高,地平面假设会得 0)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--frames", default="0-4")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()
    frames = _parse(args.frames)
    with runlog.run("autodrivedata.perception.probe_bev_depth") as rl:
        rep = run(project_path(args.root), frames)
        rep["verdict"] = verdict(rep)
        print(f"\n=== BEV 深度投影的三条对照(n={rep['n']})===")
        for lab, k in (
            ("① 接地类:深度 vs 地平面 的 IoU", "control1_drivable_iou_median"),
            ("② 换错深度后障碍物团的 IoU", "control2_obstacle_shift_median"),
            ("③ 障碍物离地高度中位(m)", "control3_height_median_m"),
        ):
            print(f"  {lab:<34}{rep[k]}")
        print(f"  ⇒ {rep['verdict']}")
        for k in (
            "control1_drivable_iou_median",
            "control2_obstacle_shift_median",
            "control3_height_median_m",
        ):
            rl.highlight(k, None if rep[k] is None else round(rep[k], 4))
        rl.highlight("verdict", rep["verdict"])
        if args.out:
            p = project_path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
            rl.artifact(p, "report")
            print(f"[done] {p.resolve()}")


def _parse(spec: str) -> list[int]:
    out: list[int] = []
    for tok in spec.split(","):
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return out


if __name__ == "__main__":
    main()

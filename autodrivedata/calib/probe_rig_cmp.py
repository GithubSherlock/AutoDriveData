"""多 rig **数值对照** + 自证:同一个 ego 上,几套相机口径差在哪、差多少。

## 为什么要它

`NUS_RIGS` 现在有三套口径(`nuscenes` 官方 / `wide` 自定义 / `nucarla` 第三方数据集)。
**换 rig 不是"换个参数"** —— 本项目红线 §P-L.1:rig 必须与权重训练数据一致(错配代价实测
122711 vs 135989 px)。而三套之间的差**不止挂点**:还有 FOV、主点约定、**乃至换算本身的约定**。
光看代码看不出来,得对成表。

## 数值自证(本探针的核心)

对每个 rig,除了列它自己的值,还会跑**该 rig 的独立复算**并报 `max|Δ|`:

| rig | 复算什么 |
|---|---|
| `nuscenes` / `nucarla` | 按 nuScenes 侧标定**另走一条路**重算 CARLA 展开(见下) |
| `wide` | 由 FoV **反推** K,再回推 FoV,闭环 `max|Δfov|` |

`nucarla` 那条尤其要:它的姿态换算与 `nuscenes` **不是同一条链**(见 `camera_rig` 的注),
"看着像"是不够的 —— 必须**逐位对表**。

用法:
  python -m autodrivedata.calib.probe_rig_cmp                       # 全部 rig
  python -m autodrivedata.calib.probe_rig_cmp --rigs nuscenes,nucarla
"""

from __future__ import annotations

import argparse
import json

from autodrivedata.calib.camera_rig import (
    NUCARLA_CAMERA_CALIBS,
    NUS_CAMERA_RIG_NUCARLA,
    coverage_table,
)

# 私名跨模块取用:`_intrinsics` 是 rig→K 的**唯一派发点**(与 `camera_calibs`/`camera_fov` 并列),
# 本项目已有同款先例(`stitch_temporal` 取 `stitch._dedup`)。另两件**公开**入口在下面。
from autodrivedata.gt.export.nuscenes import NUS_CAMERAS, NUS_RIGS, _intrinsics, camera_calibs, camera_fov
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 各 rig 的**独立复算**(与导出路径不同的那条)在下面的 `_selfcheck` 里分派。
#: 判据 = 逐通道 `max|Δ|`,单位随量。


def _selfcheck_nucarla() -> dict[str, float]:
    """nuCarla:照它 `sensors.py` 的**四元数路径**重算,与导出的矩阵路径对表。

    两条路数学不同(见 `camera_rig._nucarla_camera_rotation` 的注),同解才是证据。
    """
    import math

    import numpy as np

    def qmul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    def aa(axis, deg):
        a = np.asarray(axis, float)
        a = a / np.linalg.norm(a)
        h = math.radians(deg) / 2
        return np.array([math.cos(h), *(a * math.sin(h))])

    worst_rot = 0.0
    for ch, (_t_nus, q_nus) in NUCARLA_CAMERA_CALIBS.items():
        q_new = qmul(
            qmul(aa([0, 0, 1], -90), aa([1, 0, 0], -90)),
            np.array([q_nus[0], -q_nus[1], -q_nus[2], -q_nus[3]]),
        )
        w, x, y, z = q_new
        ref = (
            math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * x - y * z))))),
            math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))),
            math.degrees(math.atan2(2 * (w * y + x * z), 1 - 2 * (x * x + y * y))),
        )
        ours = NUS_CAMERA_RIG_NUCARLA[ch][1]
        worst_rot = max(worst_rot, max(abs(a - b) for a, b in zip(ref, ours, strict=True)))
    return {"max|Δ姿态| (度)": worst_rot}


def _selfcheck_wide(k: dict[str, tuple[float, float, float]], fov: dict[str, float]) -> dict[str, float]:
    """wide:K 由 FoV 导出 ⇒ 由 K 反推 FoV 必须回到原值(**闭环**)。"""
    import math

    from autodrivedata.gt.export.nuscenes import NUS_CAMERA_WIDTH

    worst = 0.0
    for cam, (fx, *_rest) in k.items():
        back = math.degrees(2.0 * math.atan((NUS_CAMERA_WIDTH / 2.0) / fx))
        worst = max(worst, abs(back - fov[cam]))
    return {"max|Δfov| (度)": worst}


def row_of(rig: str) -> dict:
    """一个 rig 的全部可对照量 + 自证。"""
    calibs, fov, k = camera_calibs(rig), camera_fov(rig), _intrinsics(rig)
    if rig == "nucarla":
        selfchk = _selfcheck_nucarla()
    elif rig == "wide":
        selfchk = _selfcheck_wide(k, fov)
    else:
        selfchk = {}

    cov = coverage_table(calibs, fov)
    out: dict = {
        "rig": rig,
        "cameras": {},
        "coverage_pct": round(100.0 * cov["coverage_frac"], 2),
        "blind_deg": round(cov["gap_total_deg"], 3),
        "selfcheck": selfchk,
    }
    for cam in NUS_CAMERAS:
        fx, cx, cy = k[cam]
        out["cameras"][cam] = {
            "mount_nus_m": list(calibs[cam][0]),
            "fov_deg": round(fov[cam], 3),
            "fx": round(fx, 3),
            "cx": round(cx, 3),
            "cy": round(cy, 3),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="多 rig 数值对照 + 自证(详见模块头注)")
    ap.add_argument("--rigs", default=None, help=f"逗号分隔;默认全部 {NUS_RIGS}")
    ap.add_argument("--out", default="outputs/calib_check/rig_cmp", help="输出基路径(.json/.md)")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()
    rigs = list(args.rigs.split(",")) if args.rigs else list(NUS_RIGS)

    with runlog.run("autodrivedata.calib.probe_rig_cmp") as rl:
        rows = []
        for r in rigs:
            if r not in NUS_RIGS:
                raise SystemExit(f"未知 rig {r!r}(可选 {NUS_RIGS})")
            rows.append(row_of(r))

        out = project_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.with_suffix(".json")).write_text(
            json.dumps({"rigs": rows}, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        # 表:fov / fx / cx 三个**会直接改变模型输入**的量,按通道排
        lines = [
            "# 相机 rig 数值对照",
            "",
            "| 通道 | " + " | ".join(f"{r}: fov / fx / cx" for r in rigs) + " |",
            "|---|" + "---|" * len(rigs),
        ]
        for cam in NUS_CAMERAS:
            cells = []
            for r in rows:
                c = r["cameras"][cam]
                cells.append(f"{c['fov_deg']:.2f} / {c['fx']:.1f} / {c['cx']:.1f}")
            lines.append(f"| {cam} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "| | " + " | ".join(r["rig"] for r in rows) + " |",
            "|---|" + "---|" * len(rows),
            "| 方位覆盖 % | " + " | ".join(f"{r['coverage_pct']:.2f}" for r in rows) + " |",
            "| 盲区合计 ° | " + " | ".join(f"{r['blind_deg']:.3f}" for r in rows) + " |",
            "| 自证 | "
            + " | ".join(", ".join(f"{k}={v:.2e}" for k, v in r["selfcheck"].items()) or "—" for r in rows)
            + " |",
        ]
        md = "\n".join(lines) + "\n"
        (out.with_suffix(".md")).write_text(md, encoding="utf-8")
        print(md)

        for r in rows:
            rl.highlight(f"{r['rig']}_coverage_pct", r["coverage_pct"])
            rl.highlight(f"{r['rig']}_blind_deg", r["blind_deg"])
        rl.highlight("rigs", ",".join(rigs))
        rl.artifact(str(out.with_suffix(".json")), "rig-cmp-json")
        rl.artifact(str(out.with_suffix(".md")), "rig-cmp-md")


if __name__ == "__main__":
    main()

"""教程 14 阶段 2 位对齐对拍:Python(autodrivedata/slam.py)vs C++(bin/slam_cpp.cpp)。

判据(计划 §对拍契约):全 double;同一 `(prev, cur, init, seed)` 输入下**单次 ICP**
返回的 T 逐位一致 —— 旋转角差 < 1e-3 rad、平移差 < 1e-2 m,实测在 1e-13/1e-14 量级。
唯一允许偏差 = 6×6 求解(Cholesky vs LU)+ 3×3 特征分解(Jacobi vs LAPACK)舍入 ~1e-12;
踩线 = 真 bug,不放松阈值。

**为什么对拍"单次 ICP"而不是"150 帧链式位姿末端"**:链式 T_k = Δ_k·T_{k-1} 对 Δ 的
舍入差是**指数放大**的。最近邻赋值是离散的:1e-16 的 seed 差会在少数点上翻格 → Δ 跳变
~1e-5 → 该差进入下一帧 seed 继续放大,实测 ~2.4×/帧。两条**纯 Python** 链只把 Δ 的求逆
从 `np.linalg.inv`(LU)换成刚体 Rᵀ(C++ 的 `mat_inverse_rigid` 口径),同样在 40 帧内发散到
米级 —— 所以"链式末端差"测的是**放大率**,不是移植正确性。正确判据是把 Python 链每帧的
`(init, seed)` **原样**交给 C++ 的单次 ICP,比对两者的 T。

流程:
1. g++ -O2 -std=c++17 编译 bin/slam_cpp.cpp → /tmp/slam_cpp_diff;
2. 先跑 C++ 侧 `--selftest`(不读数据、解析可验的小例子钉死各步语义)——移植版自身坏掉时
   立刻失败,不必等全量跑完才从对拍差里反推;
3. Python 跑链式(与 bin/slam_odometry.py 同口径)并记录每帧的 (init, seed);
4. 把这些 (init, seed) 喂给 C++ 的 `--icp-seq`(内部对每帧调同一次 icp_odometry);
5. 逐帧算 ΔR = Log(R_pyᵀ R_cpp)、Δt = t_cpp − t_py,超阈即 FAIL 并 exit 1;
6. 另打印**链式末端差**作参考(不判 FAIL):它是放大率指标,量级与帧数强相关。

用法:
  python bin/slam_diff_test.py [--root outputs/kitti_day_clear] [--start 0] [--end 149]
                               [--bin /tmp/slam_cpp_diff] [--tol-rad 1e-3] [--tol-m 1e-2]
                               [--skip-selftest] [--skip-build]

纯值 + subprocess(编译/运行 C++),不 import carla/torch。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from autodrivedata.accum import voxel_downsample
from autodrivedata.slam import _log_so3, icp_odometry

CPP_SRC = Path(__file__).resolve().parent / "slam_cpp.cpp"


def compile_cpp(out_bin: Path) -> None:
    cmd = ["g++", "-O2", "-std=c++17", "-o", str(out_bin), str(CPP_SRC)]
    print(f"[build] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_selftest(bin_path: Path) -> None:
    """C++ 侧自检:不读数据的小例子(下采样/最近邻 tie-break/法向门/ICP 一步)。"""
    print("[selftest] C++ 语义自检")
    r = subprocess.run([str(bin_path), "--selftest"], capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr, end="")
        sys.exit(1)


def run_python(
    root: Path, start: int, end: int, voxel: float
) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray, np.ndarray]]]:
    """Python 链式位姿(与 bin/slam_odometry.py 同口径:init=上一帧链式位姿,seed=恒速增量)。

    返回 (链式位姿列表, 每帧的 (frame, init, seed) —— 后者喂给 C++ 做单次 ICP 对拍)。
    """
    out: list[tuple[int, np.ndarray]] = []
    calls: list[tuple[int, np.ndarray, np.ndarray]] = []
    poses: list[np.ndarray] = []
    delta_prev = np.eye(4)
    prev_down: np.ndarray | None = None
    for fid in range(start, end + 1):
        p = root / "training" / "velodyne" / f"{fid:06d}.bin"
        if not p.exists():
            continue
        pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
        down = voxel_downsample(pts, voxel)
        if prev_down is None:
            T = np.eye(4)
        else:
            init = poses[-1]
            res = icp_odometry(prev_down, down, init, seed=delta_prev)
            T = np.asarray(res["T"], dtype=np.float64)
            calls.append((fid, init.copy(), np.asarray(delta_prev, dtype=np.float64).copy()))
            delta_prev = np.linalg.inv(poses[-1]) @ T
        poses.append(T)
        prev_down = down
        out.append((fid, T))
    return out, calls


def run_cpp_seq(
    root: Path, calls: list[tuple[int, np.ndarray, np.ndarray]], bin_path: Path
) -> dict[int, np.ndarray]:
    """把 (frame, init, seed) 逐帧交给 C++ 单次 ICP,收集返回的 T。"""
    velo = root / "training" / "velodyne"
    in_path, out_path = Path("/tmp/slam_cpp_seq_in.txt"), Path("/tmp/slam_cpp_seq_out.txt")
    lines = [str(len(calls))]
    for fid, init, seed in calls:
        lines.append(str(fid))
        lines.append(" ".join(f"{v:.17g}" for v in init.ravel()))
        lines.append(" ".join(f"{v:.17g}" for v in seed.ravel()))
    in_path.write_text("\n".join(lines) + "\n")
    subprocess.run(
        [str(bin_path), "--icp-seq", str(velo), str(in_path), str(out_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    res: dict[int, np.ndarray] = {}
    for ln in out_path.read_text().splitlines():
        vals = [float(v) for v in ln.split()]
        res[int(vals[0])] = np.array(vals[1:], dtype=np.float64).reshape(4, 4)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/kitti_day_clear", help="KITTI root(读,cwd 口径)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=9)
    ap.add_argument("--voxel", type=float, default=0.5)
    ap.add_argument("--bin", default="/tmp/slam_cpp_diff", help="C++ 可执行输出路径")
    ap.add_argument("--tol-rad", type=float, default=1e-3, help="逐帧旋转角差上限(rad)")
    ap.add_argument("--tol-m", type=float, default=1e-2, help="逐帧平移差上限(m)")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-selftest", action="store_true", help="跳过 C++ 语义自检")
    args = ap.parse_args()

    root = Path(args.root)
    bin_path = Path(args.bin)
    if not args.skip_build:
        compile_cpp(bin_path)
    if not args.skip_selftest:
        run_selftest(bin_path)

    print(f"[py ] {root} frames {args.start}-{args.end}")
    py, calls = run_python(root, args.start, args.end, args.voxel)
    print(f"[cpp] 单次 ICP 对拍 {len(calls)} 帧(同一 init/seed)")
    cpp = run_cpp_seq(root, calls, bin_path)

    n_fail = 0
    print(f"{'frame':>6s} {'rot_err(rad)':>13s} {'trans_err(m)':>13s}  verdict")
    for fid, Tp in py[1:]:  # 首帧无 ICP(恒等),不参与对拍
        if fid not in cpp:
            print(f"{fid:6d} {'-':>13s} {'-':>13s}  MISSING")
            n_fail += 1
            continue
        Tc = cpp[fid]
        dR = Tp[:3, :3].T @ Tc[:3, :3]
        rot = float(np.linalg.norm(_log_so3(dR)))
        trans = float(np.linalg.norm(Tc[:3, 3] - Tp[:3, 3]))
        ok = rot < args.tol_rad and trans < args.tol_m
        if not ok:
            n_fail += 1
        print(f"{fid:6d} {rot:13.3e} {trans:13.3e}  {'PASS' if ok else 'FAIL'}")

    total = len(calls)
    print(f"\n[result] 单次 ICP {total - n_fail}/{total} PASS | tol {args.tol_rad} rad / {args.tol_m} m")
    if py and cpp:
        last_fid = py[-1][0]
        if last_fid in cpp:
            Tp, Tc = py[-1][1], cpp[last_fid]
            cr = float(np.linalg.norm(_log_so3(Tp[:3, :3].T @ Tc[:3, :3])))
            ct = float(np.linalg.norm(Tc[:3, 3] - Tp[:3, 3]))
            print(
                f"[info] 链式末端差(帧 {last_fid},放大率指标,不判 FAIL):rot {cr:.3e} rad / trans {ct:.3e} m"
            )
    # 落一份机器可读结果,便于回归与文档引用
    summary = {
        "root": str(root),
        "start": args.start,
        "end": args.end,
        "n_calls": total,
        "n_fail": n_fail,
        "tol_rad": args.tol_rad,
        "tol_m": args.tol_m,
    }
    Path("/tmp/slam_diff_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if n_fail:
        print("FAIL")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()

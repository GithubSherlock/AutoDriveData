"""SLAM 轨迹精度评估(纯值:零 carla / 零 torch)——ATE / RPE + Umeyama 对齐。

**为什么需要它**:此前 SLAM 只能报 `closure.drift_m`(开放路径首末位姿距离),
那不是精度指标(Plan2.md P-H 已如实标注)。有了真值位姿(采集器写
`training/pose/{id}.txt`)之后,才能算工业界口径的 ATE/RPE。

**口径(与 evo / KITTI odometry 惯例一致)**:
- **ATE**(Absolute Trajectory Error):对齐后逐帧位置差的 RMSE。对齐 = Umeyama
  (SVD 求最优 R,t,可选 s)——**必须对齐**:SLAM 输出与 GT 各自在世界系原点不同
  (SLAM 首帧钉恒等),不对齐比出来的数是坐标系差不是精度。
- **RPE**(Relative Pose Error):固定帧间隔 Δ 的相对位姿误差,平移部分(m)与
  旋转部分(°)。**RPE 不依赖全局对齐**(相对量),所以它对"尺度/整体偏移"不敏感,
  衡量的是局部一致性——漂移率的正口径。
- **同时报对齐/不对齐**:对齐前 ATE 若远大于对齐后,说明主要误差是"整体偏移/尺度",
  不是逐帧漂移;两者都报才看得清误差来源。

**尺度**:`with_scale=False` 是默认(SLAM 输出米制,不该靠缩放拟合);`True` 时
额外报"最优尺度 s"——s 明显偏离 1 就说明存在尺度系统偏差(例如 ICP 体素导致的
尺度收缩),这是**诊断量**而非该被吸收掉的自由度。**方向语义**:s 是施加到 est 上的
因子(est → gt);est 被拉伸 1.1× ⇒ s = 1/1.1 ≈ 0.909(与 evo/Umeyama 一致)。

**退化边界(纯直行序列必读)**:共线轨迹(直线)时 **绕轨迹轴的旋转不可辨识** ——
绕轨迹轴转 0°/37°/90° 的对齐残差**恒为 0**,协方差秩 1(非共线为 2)。故纯直行序列的
*对齐后姿态* 与 *RPE 旋转项* **不可信**(全局 roll 没被数据钉住),位置 ATE 不受影响。
验证旋转精度必须走有转向/横向分量的轨迹。**尺度不退化**(1D 尺度 = 沿线长度比,
仍可辨识;曾误记为"共线尺度不可辨识",实测证伪)。见 `tests/test_slam_eval.py::TestDegenerate`。

纪律:输入须为**等长、逐帧对应**的位姿序列(帧级配对);长度不等直接报错,
不做插值匹配——静默插值会把"丢帧"伪装成"精度好"。
"""

from __future__ import annotations

import math

import numpy as np

# ── LiDAR 系 ↔ ego 系 换算(唯一落点)─────────────────────────────────────────
#
# **为什么下沉到纯值库**:这段换算是 ATE 口径的一部分(实测不带杆臂 ATE 0.4589 m vs
# 带杆臂 0.1877 m,2.44×),原先只在 `autodrivedata/slam/eval_slam.py` 里,在线 SLAM
# (`autodrivedata/live_slam.py`)要用就得复制一份 —— 而"两处各写一遍手性共轭 + 杆臂
# 方向"正是最容易静默漂的地方(方向写成 `L` 而非 `inv(L)` 只差一个符号,数字照样出得来)。
# 故与 `calib.world_to_img` 同例:实现放纯值库,bin 只调用。

# LiDAR 在 ego 系下的挂点(米;= `carla_common.SENSOR_OFFSET` / `collect_slam` 的挂点)
LIDAR_LEVER = np.array([1.2, 0.0, 1.65])
# KITTI(x 前 / y 左 / z 上)↔ CARLA(y 右)的手性共轭矩阵(M² = I)
M_FLIP = np.diag([1.0, -1.0, 1.0, 1.0])


def lever_matrix(lever: np.ndarray = LIDAR_LEVER) -> np.ndarray:
    """LiDAR 挂点 → 齐次矩阵 L(ego→LiDAR 的刚体变换,无旋转)。"""
    L = np.eye(4)
    L[:3, 3] = lever
    return L


def lidar_pose_to_ego(T: np.ndarray, lever: np.ndarray = LIDAR_LEVER) -> np.ndarray:
    """LiDAR 系位姿(KITTI 约定)→ ego 系位姿(CARLA 约定):`L·M·T·M @ inv(L)`。

    **为什么是四段而不是两段**(2026-09-19 修正,原式漏了左端 `L`):
    设 ego 世界位姿 `E_k`(ego 局部 → 世界)、挂点 `L`(LiDAR 局部 → ego 局部)、
    手性共轭 `M`(KITTI y 左 ↔ CARLA y 右)。喂给 ICP 的点云是 KITTI 口径,故
    LiDAR-k 局部 → 世界是 `G_k = E_k·L·M`。链式位姿的语义是
    `P_k = inv(G_0)·G_k = M·inv(E_0·L)·(E_k·L)·M`(帧 0 = 恒等)。

    反解 `E_k`(以帧 0 为参考):`L·M·P_k·M·inv(L) = inv(E_0)·E_k`。

    **判据(锚点)**:`f(I) = I` —— 帧 0 必须落在轨迹原点。原式 `M·T·M @ inv(L)`
    在 `T = I` 时给出平移 `[-1.2, 0, -1.65]`(恰是杆臂),即把杆臂当成了恒定偏移
    ——那是 **ATE 对齐口径**(Umeyama 会吸收常数平移,故对齐后 ATE 0.1877 m 两式相同、
    看不出错),不是 ego 相对位姿。见 `tests/test_slam_eval.py::TestLeverArm`。

    **方向勿凭直觉**:右端是 `inv(L)` 不是 `L` —— 写反 ATE 从 0.1877 m 涨到 0.4589 m
    (2.44×),而轨迹形状看着仍然"像那么回事"。见 Plan2.md §P-H.1。
    """
    L = lever_matrix(lever)
    return L @ M_FLIP @ T @ M_FLIP @ np.linalg.inv(L)


def _as_pose_array(poses: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """list[(4,4)] 或 (N,4,4) → (N,4,4) float64;形状不对直接报错。"""
    a = np.asarray(poses, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (4, 4):
        raise ValueError(f"位姿序列须为 (N,4,4),got {a.shape}")
    return a


def umeyama_alignment(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """Umeyama:求 (R, t, s) 使 s·R·src + t ≈ dst(最小二乘,SVD 闭式解)。

    src/dst 均为 (N,3) 对应点。返回 (R (3,3), t (3,), s)。
    **退化保护**:反射(‖det R‖ = −1)时翻最后一列奇异向量符号,保证输出是真旋转
    (KITTI/evo 同款处理;不处理会得到镜像变换、ATE 假性偏小)。
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"须为同形 (N,3),got {src.shape} / {dst.shape}")
    if len(src) < 3:
        raise ValueError(f"Umeyama 至少需 3 个点,got {len(src)}")
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (sc**2).sum() / len(src)
        s = float((D * np.diag(S)).sum() / var_s) if var_s > 0 else 1.0
    else:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    return R, t, s


def align_trajectory(
    est: list[np.ndarray] | np.ndarray, gt: list[np.ndarray] | np.ndarray, with_scale: bool = False
) -> dict:
    """按位置做 Umeyama 对齐,返回对齐后的 est(位姿整体乘对齐变换)+ 变换参数。

    **只对齐位置**求 (R,t,s),再**整体施加到姿态块**上(est_k ← A·est_k,
    A = [[sR, t],[0,1]])——这是标准做法:位姿序列的平移部分定对齐,姿态随之旋转,
    否则位置对齐了而姿态还差一个全局旋转,RPE 旋转项会假性偏大。
    """
    e = _as_pose_array(est)
    g = _as_pose_array(gt)
    if len(e) != len(g):
        raise ValueError(f"位姿数不等(est {len(e)} vs gt {len(g)}):帧级配对是硬门槛,不做插值")
    R, t, s = umeyama_alignment(e[:, :3, 3], g[:, :3, 3], with_scale=with_scale)
    A = np.eye(4)
    A[:3, :3] = s * R
    A[:3, 3] = t
    return {"aligned": np.array([A @ T for T in e]), "R": R, "t": t, "scale": s}


def ate(
    est: list[np.ndarray] | np.ndarray,
    gt: list[np.ndarray] | np.ndarray,
    *,
    align: bool = True,
    with_scale: bool = False,
) -> dict:
    """绝对轨迹误差(位置,米)。align=False 时不做对齐(报坐标系原始差,作对照)。"""
    e = _as_pose_array(est)
    g = _as_pose_array(gt)
    if len(e) != len(g):
        raise ValueError(f"位姿数不等(est {len(e)} vs gt {len(g)}):帧级配对是硬门槛,不做插值")
    info: dict = {"n": len(e), "aligned": align}
    if align:
        a = align_trajectory(e, g, with_scale=with_scale)
        e = a["aligned"]
        info["scale"] = round(a["scale"], 6)
    err = np.linalg.norm(e[:, :3, 3] - g[:, :3, 3], axis=1)
    return {
        **info,
        "rmse_m": round(float(np.sqrt((err**2).mean())), 5),
        "mean_m": round(float(err.mean()), 5),
        "median_m": round(float(np.median(err)), 5),
        "max_m": round(float(err.max()), 5),
        "final_m": round(float(err[-1]), 5),
    }


def _rot_angle_deg(R: np.ndarray) -> float:
    """旋转阵 → 旋转角(度),数值安全(acos 参数夹紧)。"""
    c = (float(np.trace(R)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def rpe(
    est: list[np.ndarray] | np.ndarray,
    gt: list[np.ndarray] | np.ndarray,
    *,
    delta: int = 1,
    align: bool = True,
) -> dict:
    """相对位姿误差:间隔 delta 帧的相对变换误差(平移 m / 旋转 °)。

    e_k = (gt_k⁻¹ gt_{k+Δ})⁻¹ · (est_k⁻¹ est_{k+Δ}) —— 理想为恒等。
    **相对量,对全局对齐不敏感**:`align=True`(默认)仍先做一次全局对齐,使姿态
    参考系一致(否则旋转项含全局旋转差,假性偏大);平移项与对齐无关。
    **注意**:`rot_rmse_deg` 在**纯直行序列**上不可信——共线数据钉不住绕轨迹轴的旋转
    (见模块头"退化边界"),此时应改看平移项。
    """
    if delta < 1:
        raise ValueError(f"delta 须 ≥1,got {delta}")
    e = _as_pose_array(est)
    g = _as_pose_array(gt)
    if len(e) != len(g):
        raise ValueError(f"位姿数不等(est {len(e)} vs gt {len(g)}):帧级配对是硬门槛,不做插值")
    if align:
        e = align_trajectory(e, g, with_scale=False)["aligned"]
    if len(e) <= delta:
        raise ValueError(f"帧数 {len(e)} 不足以计算 Δ={delta} 的 RPE")
    gt_rel = np.linalg.inv(g[:-delta]) @ g[delta:]
    est_rel = np.linalg.inv(e[:-delta]) @ e[delta:]
    err_T = np.linalg.inv(gt_rel) @ est_rel
    trans = np.linalg.norm(err_T[:, :3, 3], axis=1)
    rot = np.array([_rot_angle_deg(M) for M in err_T[:, :3, :3]])
    # 每米平移误差:Δ 帧的相对平移量做分母(漂移率的常用表达)
    step = np.linalg.norm(gt_rel[:, :3, 3], axis=1)
    per_m = float(trans.sum() / step.sum()) if step.sum() > 0 else float("nan")
    return {
        "delta": delta,
        "n": len(e) - delta,
        "trans_rmse_m": round(float(np.sqrt((trans**2).mean())), 5),
        "trans_mean_m": round(float(trans.mean()), 5),
        "trans_max_m": round(float(trans.max()), 5),
        "rot_rmse_deg": round(float(np.sqrt((rot**2).mean())), 5),
        "rot_mean_deg": round(float(rot.mean()), 5),
        "gt_path_m": round(float(step.sum()), 4),
        "trans_per_m": round(per_m, 6),
    }


def eval_trajectory(
    est: list[np.ndarray] | np.ndarray,
    gt: list[np.ndarray] | np.ndarray,
    *,
    rpe_deltas: tuple[int, ...] = (1, 5, 10),
    with_scale: bool = True,
) -> dict:
    """一次给出全套口径:ATE(对齐/不对齐 + 最优尺度)+ 多个 Δ 的 RPE。"""
    return {
        "ate_aligned": ate(est, gt, align=True, with_scale=with_scale),
        "ate_raw": ate(est, gt, align=False),
        "rpe": {f"d{d}": rpe(est, gt, delta=d) for d in rpe_deltas},
    }

"""教程 14 激光 SLAM:纯值两段式(FAST-LIO2 前端 + SC-PGO 后端降档)。

本模块是 SLAM 的**纯 numpy 核心环**(阶段 1),并作为阶段 2 C++ 移植的 oracle。
对标教程 14 的 FAST-LIO2 + SC-PGO,做**文档化简化**(本仓无 IMU 采集、无 ROS):

- 前端 = **帧间点面 ICP**(替代 FAST-LIO2 的 ikd-tree scan-to-map;帧间重叠 ~90% 时
  等效,后端纠偏)——恒速先验初始化 + λ 正则化法方程(见 estimate_transform_gn)。
- 后端 = **ScanContext 回环**(点计数描述子,列滚动不变)+ **位姿图 G-N**(节点 ≤200,
  纯 numpy,无 g2o)。
- **位对齐纪律**:全 double、网格哈希 tie-break 钉死字典序、体素重心按扫描序累加、
  λ 正则化解代替 lstsq/SVD——保证 C++ 移植逐位对齐(numpy/C++ 对拍唯一允许偏差
  ~1e-12 求解舍入,见 bin/slam_diff_test.py 的 1e-3/1e-2 阈值)。

口径:
- 输入:velodyne bin (N,4) x,y,z,intensity(KITTI velodyne 约定,x 前/y 左/z 上)。
- 输出:链式位姿 T_0→k(4×4,double);判据 = 漂移率(closure_error)与逐帧残差。

单测:tests/test_slam.py(手算锚点)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# 常量(全模块唯一入口;C++ 移植逐条对照)
# ---------------------------------------------------------------------------
GRID_CELL = 0.5  # 网格哈希格子边长(m)
DOWNSAMPLE_VOXEL = 0.5  # 下采样体素边长(m,accum.voxel_downsample 同款)
ICP_MAX_ITER = 15
ICP_TOL_DELTA = 1e-5  # 相邻迭代变换增量阈值(‖ΔR‖₂+‖Δt‖₂)
ICP_NORMAL_K = 8  # 法向 PCA 邻域点数
# 法向可信门(**位对齐必需,不是调参**):邻域退化时最小特征向量方向不定——
# LAPACK 与自写 Jacobi 会在退化子空间里挑到不同向量,差值是 O(1) 而非舍入量级。
# 判据 = λ1 > ICP_NORMAL_PLANARITY · λ2(纯协方差特征值,双方可逐位复算):
#   * 秩 ≤1 的邻域(k≤2 或共线)λ2 = 0 → 恒被拒(法向在二维子空间里可任意转);
#   * 平面内两轴近等大 → 法向在平面内可任意转。
# 实测帧 0:被拒 750/13637 = 5.5%(k<3 有 633、λ1/λ2<1e-9 另有 117);
# 不门控时对拍平移差 3.2cm,远超 1e-2m 阈值。
# 被拒点法向置**精确零向量**,ICP 侧显式剔除(零行对 AtA/Atb 贡献恒为 0,
# 但会白占残差分母 → 不靠数值巧合,显式过滤)。
ICP_NORMAL_PLANARITY = 1e-9  # λ1/λ2 下限(低于此判为退化邻域)
ICP_REG_LAM = 1e-4  # 法方程正则项(约束条件数,位对齐关键)
ICP_OVERLAP_RADIUS = 0.3  # 重叠度判定半径(m)
ICP_FAIL_OVERLAP = 0.3  # 重叠度低于此 → 标记 failed(恒速先验兜底)

SC_NUM_RINGS = 20
SC_NUM_SECTORS = 60
SC_MAX_RANGE = 40.0  # 描述子只看 40m 内(车载感知窗)
SC_SIM_THRESH = 0.15  # 回环门:平均余弦距离 < 此值才候选
SC_TOP_N = 5  # 回环候选最多取前 N
SC_MIN_GAP_NODES = 25  # 候选与本帧关键帧号差 ≥ 此值(避免原地/邻近自环)

PGO_ITERS = 20
PGO_LAMBDA0 = 1e-3  # LM 初阻尼
PGO_DAMP_UP = 3.0  # 拒绝 → λ×3
PGO_DAMP_DOWN = 3.0  # 接受 → λ÷3
PGO_LAM_MAX = 1e6
PGO_CONV_DELTA = 1e-6  # ‖δ‖∞ 收敛阈值
PGO_W_ODOM = 1.0  # 里程计边权重
PGO_W_LOOP = 0.5  # 回环边权重
PGO_ANCHOR_WEIGHT = 1e6  # 节点0 锚到恒等

KEYFRAME_EVERY = 10  # 关键帧间隔(帧)

LOOP_GATE_SC = 0.15
LOOP_GATE_OVERLAP = 0.4  # 双 yaw ICP 重叠度门
LOOP_GATE_RMSE = 0.3  # 双 yaw ICP RMSE 门
LOOP_GATE_CONVERGED = True

# 27 邻域偏移,**字典序钉死**(先 x 后 y 后 z,从 -1..1 递增)。
# C++ 移植必须硬编码同序(对拍 tie-break 的关键,勿改顺序)。
_GRID_OFFSETS = [(dx, dy, dz) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
GRID_OFFSETS = np.array(_GRID_OFFSETS, dtype=np.int64)


def _radius_offsets(r: int) -> list[tuple[int, int, int]]:
    """半径 r 立方体壳偏移(字典序:先 x 后 y 后 z)。C++ 移植硬编码同序。"""
    return [
        (a, b, c2)
        for a in range(-r, r + 1)
        for b in range(-r, r + 1)
        for c2 in range(-r, r + 1)
        if max(abs(a), abs(b), abs(c2)) == r
    ]


# 预生成半径 0..MAXRAD 的壳偏移:最近邻格距可任意大(稀疏云),层判停由 best_d 保证。
GRID_MAX_RAD = 16
_GRID_OFFSETS_BY_LAYER: dict[int, list[tuple[int, int, int]]] = {
    r: _radius_offsets(r) for r in range(GRID_MAX_RAD + 1)
}


# ---------------------------------------------------------------------------
# SE(3) 原语(纯 numpy,无外部依赖)
# ---------------------------------------------------------------------------
def twist_exp(omega: np.ndarray, v: np.ndarray) -> np.ndarray:
    """se(3) (ω, v) → SE(3) 4×4。Rodrigues 公式,数值稳定(‖ω‖→0 用一阶)。"""
    w = np.asarray(omega, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    th = float(np.linalg.norm(w))
    R = np.eye(3)
    if th < 1e-9:
        R = R + _skew(w)
    else:
        wx = _skew(w / th)
        R = np.eye(3) + np.sin(th) * wx + (1 - np.cos(th)) * (wx @ wx)
    # 平移:V = I + (1-cosθ)/θ² [w]× + (θ−sinθ)/θ³ [w]×²
    if th < 1e-9:
        t = vv
    else:
        wx = _skew(w)
        wx2 = wx @ wx
        V = np.eye(3) + (1 - np.cos(th)) / (th * th) * wx + (th - np.sin(th)) / (th**3) * wx2
        t = V @ vv
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def twist_log(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SE(3) → (ω, v) 局部坐标。ω 取旋转轴角,平移用 V⁻¹。"""
    R = np.asarray(T, dtype=np.float64)[:3, :3]
    t = np.asarray(T, dtype=np.float64)[:3, 3]
    th = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if th < 1e-9:
        w = np.zeros(3)
        V_inv = np.eye(3)
    else:
        w = (
            th
            / (2 * np.sin(th))
            * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
        )
        wx = _skew(w)
        # V⁻¹ = I − ½[w]× + (1/θ² − (1+cosθ)/(2θ sinθ)) [w]×²
        wx2 = wx @ wx
        V_inv = np.eye(3) - 0.5 * wx + (1.0 / (th * th) - (1 + np.cos(th)) / (2 * th * np.sin(th))) * wx2
    v = V_inv @ t
    return w, v


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]], dtype=np.float64)


def relative_transform(Ta: np.ndarray, Tb: np.ndarray) -> np.ndarray:
    """T_ab = Ta⁻¹ Tb(把 b 相对 a 的位姿)。"""
    inv = np.linalg.inv(Ta)
    return inv @ Tb


# ---------------------------------------------------------------------------
# 网格哈希(最近邻;tie-break 钉死字典序,与 C++ 对齐)
# ---------------------------------------------------------------------------
class GridHash:
    """体素格子 → 点索引桶。查询扫 27 邻域格子,格内按扫描序,**等距先到者胜**。

    与 C++ 移植的 `std::map<key, std::vector<int>>` + 同序 GRID_OFFSETS 完全对应。
    """

    def __init__(self, points: np.ndarray, cell: float = GRID_CELL) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        self.cell = float(cell)
        self.buckets: dict[tuple[int, int, int], list[int]] = {}
        # 体素键按扫描序入桶(等距 tie 时取先遇到者,与 C++ 同)
        keys = np.floor(self.points[:, :3] / self.cell).astype(np.int64)
        for i, k in enumerate(keys):
            self.buckets.setdefault((int(k[0]), int(k[1]), int(k[2])), []).append(i)

    def nearest(self, q: np.ndarray) -> tuple[int, float]:
        """q (3,) → (最近邻索引, 距离²)。空云 → (-1, inf)。

        半径递增立方体扫描**,稀疏云/空邻域也能命中全局最近**;格内按扫描序、
        等距先到者胜。C++ 移植同式(对拍 tie-break 关键)。

        早停安全判据(2026-09-18 修正):层 rad 内点的坐标相对查询格某轴至少偏
        rad−1 格(order-0 保守下界取 rad−1,对 rad=1 即 ≥0),故扫完层 rad 后只要
        best_d < ((rad−1)·cell)² 即可证明全局最近(rad=0 不满足——best_d 只来自
        自身格,层 1 可能有更近点;rad≥1 才生效)。**旧实现 rad=0 用同一判断提前
        停在自身格,曾返回次优**(实测 vox0.5 帧 856/12840 = 6.7% 次优)。修正后与
        暴力最近邻 0 不一致;tie-break(等距先到者胜)语义不变,C++ 移植同式。
        """
        if len(self.points) == 0:
            return -1, float("inf")
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 1 or q.shape[0] != 3:
            raise ValueError(f"查询点须为 (3,) 数组,got {q.shape}")
        c = np.floor(q / self.cell).astype(np.int64)
        cx, cy, cz = int(c[0]), int(c[1]), int(c[2])
        best_i, best_d = -1, float("inf")
        rad = 0
        empty_layers = 0
        while True:
            if empty_layers >= 2 and rad > 4:  # 稀疏合成云兜底:层 4+ 仍全空 → 全云扫描
                for i in range(len(self.points)):
                    d = float(((self.points[i, :3] - q) ** 2).sum())
                    if d < best_d:
                        best_d, best_i = d, i
                break
            layer_points = 0
            for off in _GRID_OFFSETS_BY_LAYER[rad]:
                cell = (cx + off[0], cy + off[1], cz + off[2])
                layer_points += len(self.buckets.get(cell, ()))
                for i in self.buckets.get(cell, ()):
                    d = float(((self.points[i, :3] - q) ** 2).sum())
                    if d < best_d:  # 严格小于:等距保留先遇到(扫描序)
                        best_d, best_i = d, i
            if rad >= 1 and best_i != -1 and best_d < ((rad - 1) * self.cell) ** 2:
                break
            empty_layers = empty_layers + 1 if layer_points == 0 else 0
            rad += 1
            if rad > GRID_MAX_RAD:  # 超出预生成半径 → 全云扫描(保结果正确)
                for i in range(len(self.points)):
                    d = float(((self.points[i, :3] - q) ** 2).sum())
                    if d < best_d:
                        best_d, best_i = d, i
                break
        return best_i, best_d


# ---------------------------------------------------------------------------
# 批量最近邻(searchsorted 线性键 + rank 折叠;是 GridHash.nearest 的批加速版)
# ---------------------------------------------------------------------------
# 线性键编码:体素键 + base 后逐 lane 左移相加,无进位(lane 宽 21bit,base 2^20,
# 坐标 ±2^20 格 / ≤524km 内安全;x·2^42 + y·2^21 + z)。**必须保证 2×2^20·2^42 < 2^63**
# (实测旧 base=2^21·2^43 溢出成负数、桶错位 → 必须此尺寸)。
_GRID_KEY_BASE = 1 << 20
_GRID_KEY_SHIFT = 21
_GRID_LIN_X = 1 << (2 * _GRID_KEY_SHIFT)  # x 进位单位 = 2^42
_GRID_LIN_Y = 1 << _GRID_KEY_SHIFT  # y 进位单位 = 2^21
_GRID_RANK_SHIFT = 22  # rank 低 22 位存原索引(点 ≤ 4M 无碰撞)
# rank = rad·OFF + 层内偏移序·2^22 + idx,须严格 (rad, 偏移序, idx) 字典序 → 跨层等距
# tie-break 与标量"先到者(低层)胜"一致。OFF 必须 > 最大层偏移数 × 2^22:扫到
# GRID_MAX_RAD=16(层 16 有 6146 偏移,6146·2^22 ≈ 2.58e13),故 OFF = 2^45 安全
# (rad·OFF ≤ 16·2^45 = 2^49,加 r·2^22 与 idx 仍 < 2^63,int64 余量充足)。旧值 2^31
# 只够扫到层 4(386<512 偏移);层 ≥5 偏移数 >512 会使 r·2^22 撞穿 OFF、翻转层优先级。
_GRID_RANK_OFF = 1 << 45  # 层进位单位
# 层偏移预折成线性键增量 (dx·2^42 + dy·2^21 + dz):整层一次性向量化用。
# **不预折 = 逐偏移一次 numpy 调用**:层 16 有 6146 个偏移、绝大多数格为空,
# 空探测的调用开销(~10µs/偏移)会压过真实计算(实测单次 nearest 0.6–1.0s)。
_GRID_DELTA_BY_LAYER: dict[int, np.ndarray] = {
    r: np.array([a * _GRID_LIN_X + b * _GRID_LIN_Y + c for a, b, c in offs], dtype=np.int64)
    for r, offs in _GRID_OFFSETS_BY_LAYER.items()
}
# 整层向量化的分块上限(元素数):单块 (活跃查询 × 层偏移) ≤ 此值 → int64 峰值 ≤ 32MB。
_NB_CHUNK_ELEMS = 1 << 22


def _cell_lin(coords: np.ndarray, cell: float) -> np.ndarray:
    """体素键 (N,3) → 线性键 int64;被 (base+coord) 保证非负,+base 后 lane 无进位。"""
    k = np.floor(coords / cell).astype(np.int64) + _GRID_KEY_BASE
    return k[:, 0] * _GRID_LIN_X + k[:, 1] * _GRID_LIN_Y + k[:, 2]


def nearest_batch(
    ref_all: np.ndarray, queries: np.ndarray, cell: float = GRID_CELL
) -> tuple[np.ndarray, np.ndarray]:
    """批量最近邻:逐查询返回 (最近邻索引, 距离²),与 GridHash.nearest 逐位一致。

    向量化「整层壳扫描 + 每层收窄活跃集」:
    1. 层 0..GRID_MAX_RAD 逐壳 searchsorted 拉候选,逐查询 fold min(d2)/min(rank)
       (rank = rad·_GRID_RANK_OFF + 层内偏移序·2^22 + 原索引,精确复现"等距先到者胜";
       OFF=2^45 保证 (rad, 偏移序, idx) 严格字典序,跨层等距时低层胜);
       一层内的全部偏移**一次 searchsorted 拉平**(分块 _NB_CHUNK_ELEMS 控内存)——
       逐偏移调用会在层 16 的 6146 个空格上白付调用开销;
    2. 每层扫完按**安全早停**收窄:扫完层 rad 后,未扫格必在某轴偏 ≥ rad+1 格 ⇒
       其中任一点距离 ≥ rad·cell(格内最靠前的点还差 (rad+1−1)·cell),故
       best_d < (rad·cell)² 的查询可停(rad≥1)。真实帧云(帧间重叠 ≥85%)层 0-2
       就停掉 ~90%,活跃集迅速收缩;
    3. 剩余活跃查询(孤立/远点,扫完 GRID_MAX_RAD 仍未停)→ 分块暴力全扫,语义 =
       标量 GridHash 的全扫分支:**按原索引升序、严格小于才替换**(等距保留先到者,
       即保留已在 rank 序中胜出的既有最优)。
    返回 (idx (N,), d2 (N,))。
    """
    ref = np.asarray(ref_all[:, :3], dtype=np.float64)
    qs = np.asarray(queries, dtype=np.float64)
    if len(ref) == 0:
        return np.full(len(qs), -1, dtype=np.int64), np.full(len(qs), float("inf"))
    # 排序点表(桶内按 idx 升序 = 扫描序)+ 唯一键桶计数
    ordp = np.lexsort((np.arange(len(ref)), _cell_lin(ref, cell)))
    P = ref[ordp]
    # I:原索引(桶内 idx 升序对应 lexsort 的次键)
    I = np.arange(len(ref))[ordp]
    L = _cell_lin(ref, cell)[ordp]
    U, cnt = np.unique(L, return_counts=True)
    cum = np.concatenate([[0], np.cumsum(cnt)])
    qL = _cell_lin(qs, cell)
    n_q = len(qs)
    best_d = np.full(n_q, np.inf)
    best_rk = np.zeros(n_q, dtype=np.int64)
    active = np.arange(n_q, dtype=np.int64)
    i64max = np.iinfo(np.int64).max
    for rad in range(GRID_MAX_RAD + 1):
        if active.size == 0:
            break
        aL = qL[active]
        n_act = active.size
        deltas = _GRID_DELTA_BY_LAYER[rad]
        # 分块宽度:保证 (n_act × w) 的 int64 中间量 ≤ _NB_CHUNK_ELEMS
        step = max(1, _NB_CHUNK_ELEMS // max(n_act, 1))
        for c0 in range(0, deltas.size, step):
            dl = deltas[c0 : c0 + step]
            w = dl.size
            # (n_act, w) 目标键,行主序拉平 → searchsorted 一次定区间
            tgt = (aL[:, None] + dl[None, :]).ravel()
            lo = np.searchsorted(U, tgt, "left")
            hi = np.searchsorted(U, tgt, "right")
            occ = (lo < hi).reshape(n_act, w)
            if not occ.any():
                continue
            qi, oi = np.nonzero(occ)  # 行主序 → qi 非降(折叠按段切分的前提)
            qq = active[qi]
            st = lo.reshape(n_act, w)[qi, oi]
            # 桶大小 = 该唯一键的点数 cnt[st],**不是 searchsorted 区间长 hi-lo**
            # (存在即恒为 1,会把同格多点漏掉——曾致 6.7% 次优,q1908 只取 236 漏 869)
            cntg = cnt[st]
            tot = int(cntg.sum())
            rep = np.repeat(qq, cntg)
            gst = np.concatenate([[0], np.cumsum(cntg)[:-1]])
            seg = np.arange(tot) - np.repeat(gst, cntg)
            cpos = cum[st].repeat(cntg) + seg
            d2 = ((P[cpos] - qs[rep]) ** 2).sum(1)
            rk = rad * _GRID_RANK_OFF + np.repeat(c0 + oi, cntg) * (1 << _GRID_RANK_SHIFT) + I[cpos]
            gd = np.minimum.reduceat(d2, gst)
            grp = np.repeat(np.arange(len(cntg)), cntg)
            tied = np.where(d2 == gd[grp], rk, i64max)
            grank = np.minimum.reduceat(tied, gst)
            # (候选对 → 查询) 折叠:qi 非降,按变化处切段
            qb = np.concatenate([[0], np.nonzero(np.diff(qi))[0] + 1])
            qd = np.minimum.reduceat(gd, qb)
            qsz = np.diff(np.append(qb, len(gd)))
            qgrp = np.repeat(np.arange(len(qb)), qsz)
            qrank = np.minimum.reduceat(np.where(gd == qd[qgrp], grank, i64max), qb)
            qsel = qq[qb]
            upd = (qd < best_d[qsel]) | ((qd == best_d[qsel]) & (qrank < best_rk[qsel]))
            if upd.any():
                best_d[qsel[upd]] = qd[upd]
                best_rk[qsel[upd]] = qrank[upd]
        # 安全早停:扫完层 rad 后未扫点距离² ≥ (rad·cell)²(rad=0 不成立——层1 可能更近)
        if rad >= 1 and active.size:
            stop = best_d[active] < (rad * cell) ** 2
            if stop.any():
                active = active[~stop]
    # 剩余活跃查询:分块全云扫描,语义 = 标量 GridHash 的全扫分支(按**原索引升序**
    # 严格小于才替换 → 等距保留先到者;已在层扫描中胜出的既有最优若等距则不被替换)。
    if active.size:
        ordI = np.argsort(I, kind="stable")  # P 序 → 原索引升序
        Is = I[ordI]
        for start in range(0, active.size, 512):
            blk = active[start : start + 512]
            d2blk = ((P[None, :, :] - qs[blk, None, :]) ** 2).sum(-1)  # (B,M)
            d2s = d2blk[:, ordI]
            rowmin = d2s.min(1)
            first = np.argmax(d2s == rowmin[:, None], axis=1)  # 原索引序首个达到最小值者
            keep = best_d[blk] <= rowmin  # 既有最优已并列 → 保留(严格小于语义)
            best_d[blk] = np.where(keep, best_d[blk], rowmin)
            best_rk[blk] = np.where(keep, best_rk[blk], Is[first])
    # 折叠:idx = rank 低 22 位(与标量回退写入的纯 idx 一致)
    idx = np.mod(best_rk, (1 << 22))
    return idx, best_d


def _batch_knn(
    ref: np.ndarray, queries: np.ndarray, k: int, cell: float = GRID_CELL
) -> tuple[np.ndarray, np.ndarray]:
    """批量 k-最近邻(供 estimate_normals 批 PCA)。

    用 searchsorted 线性键,27 偏移逐格 gather 候选,一次 lexsort 定序后按段取前 k。
    与标量 estimate_normals 的"27 格候选 + 稳定 sort(d2 升序,等距保持插入序 =
    偏移字典序 → 格内扫描序)取前 k"**语义逐位一致**:
    - 候选集完全相同(同一 27 偏移字典序 + 同格内扫描序);
    - lexsort((ci, co, cd, cq)) 末键为主键 → 优先级 cq(分组) > cd(d2 浮点升序) >
      co(偏移序) > ci(idx),与标量 sort(key=(d2, 偏移序, idx)) 逐位同序。整数 rank
      折叠(d2 量化到 2^-40)会抹掉近等距邻居的浮点序 → 直接按浮点 d2 排序既无量化
      又免 top-k 二次排序。
    返回 (nb_idx (N,k) int64, nb_d2 (N,k) float64);候选不足 k 用 -1 补齐。
    """
    ref = np.asarray(ref[:, :3], dtype=np.float64)
    qs = np.asarray(queries, dtype=np.float64)
    n = len(qs)
    if n == 0 or len(ref) == 0:
        return np.full((n, k), -1, dtype=np.int64), np.full((n, k), np.inf)
    ordp = np.lexsort((np.arange(len(ref)), _cell_lin(ref, cell)))
    P = ref[ordp][:, :3]
    I = np.arange(len(ref))[ordp]
    L = _cell_lin(ref, cell)[ordp]
    U, cnt = np.unique(L, return_counts=True)
    cum = np.concatenate([[0], np.cumsum(cnt)])
    qL = _cell_lin(qs, cell)
    # 候选 gather:27 偏移逐格,桶内多点全部拉入;同时记录偏移序(等距 tie-break 用)
    # **只扫 27 格(rad≤1)** —— 与标量 estimate_normals 同口径(它也只取 3×3×3 邻域,
    # 不扩层);若某查询有 27 格候选但不足 k,语义 = 标量(不足就少填),候选都仍在本窗。
    parts_i: list[np.ndarray] = []
    parts_d: list[np.ndarray] = []
    parts_q: list[np.ndarray] = []
    parts_o: list[np.ndarray] = []
    for o, off in enumerate(GRID_OFFSETS):
        tgt = qL + off[0] * _GRID_LIN_X + off[1] * _GRID_LIN_Y + off[2]
        lo = np.searchsorted(U, tgt, "left")
        hi = np.searchsorted(U, tgt, "right")
        occ = lo < hi
        if not occ.any():
            continue
        qq = np.nonzero(occ)[0]
        st = lo[qq]
        cntg = cnt[st]
        tot = int(cntg.sum())
        if tot == 0:
            continue
        rep = np.repeat(qq, cntg)
        gst = np.concatenate([[0], np.cumsum(cntg)[:-1]])
        seg = np.arange(tot) - np.repeat(gst, cntg)
        cpos = cum[st].repeat(cntg) + seg
        d2 = ((P[cpos] - qs[rep]) ** 2).sum(1)
        parts_i.append(I[cpos])
        parts_d.append(d2)
        parts_q.append(rep)
        parts_o.append(np.full(tot, o, dtype=np.int64))
    if not parts_i:
        return np.full((n, k), -1, dtype=np.int64), np.full((n, k), np.inf)
    ci = np.concatenate(parts_i)
    cd = np.concatenate(parts_d)
    cq = np.concatenate(parts_q)
    co = np.concatenate(parts_o)
    # 每查询候选数(sort 前,候选集幂等)
    widths = np.bincount(cq, minlength=n).astype(np.int64)
    Wk = int(max(widths.max() if widths.size else 0, k))  # ≥k,保证能切出 k 列
    # **一次 lexsort 定序、pad 后直接取前 k 列**(不再折 rank + argpartition)。
    # lexsort 末键为主键 → 优先级 cq(分组) > cd(d2 浮点升序) > co(偏移序) > ci(idx),
    # 与标量 cand.sort(key=(d2, 偏移序, idx)) **逐位同序**。整数 rank 折叠会把 d2 量化到
    # 2^-40、抹掉近等距(<1e-12)邻居的精确浮点序 → 曾致 292 序错 / 190 退化法向翻转;
    # 直接按浮点 d2 排序 + 段内前 k 列既无量化又免去 top-k 二次排序。
    order = np.lexsort((ci, co, cd, cq))
    cs = np.empty(n + 1, dtype=np.int64)
    cs[0] = 0
    np.cumsum(widths, out=cs[1:])
    pos = np.arange(len(order)) - cs[cq[order]]  # 段内秩(0=最近)
    flat = cq[order] * Wk + pos
    fidx = np.full(n * Wk, -1, dtype=np.int64)
    fd2 = np.full(n * Wk, np.inf)
    fidx[flat] = ci[order]
    fd2[flat] = cd[order]
    # 列 0..k-1 已是 (d2, 偏移序, idx) 升序邻居;不足 k 的尾列天然 -1/inf 补齐
    nb_idx = fidx.reshape(n, Wk)[:, :k].copy()
    nb_d2 = fd2.reshape(n, Wk)[:, :k].copy()
    return nb_idx, nb_d2


def estimate_normals(points: np.ndarray, k: int = ICP_NORMAL_K) -> np.ndarray:
    """逐点 PCA 最小特征向量法向(PCA 协方差手算 3×3,不依赖 np.cov 排序)。

    邻域取网格哈希最近 k;协方差用 (p−μ)(p−μ)ᵀ 累加 → eigh 最小特征向量。
    向量化:一次性对整云建哈希,再逐点查——O(N log N) 替代 O(N²) 暴力。
    批版本:27 格候选 gather(_batch_knn)→ 批 PCA(协方差 (p−μ)ᵀ(p−μ) 手算 3×3
    → 批 eigh 最小特征向量)。**与旧标量逐位一致**(候选集/排序/tie-break 同源)。
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[1] > 3:
        pts = pts[:, :3]  # (N,4+) → (N,3)(与 estimate_transform_gn 同口径)
    n = len(pts)
    if n == 0:
        return np.zeros((0, 3))
    nb, _ = _batch_knn(pts, pts, k)
    mu = np.zeros((n, 3), dtype=np.float64)
    cnt_nb = np.zeros(n, dtype=np.int64)
    for j in range(k):
        col = nb[:, j]
        ok = col >= 0
        cnt_nb += ok
        mu[ok] += pts[col[ok]]
    mu /= np.maximum(cnt_nb, 1)[:, None]
    cov = np.zeros((n, 3, 3), dtype=np.float64)
    for j in range(k):
        col = nb[:, j]
        ok = col >= 0
        d = pts[col[ok]] - mu[ok]
        cov[ok] += d[:, :, None] * d[:, None, :]
    cov /= np.maximum(cnt_nb, 1)[:, None, None]
    evals, evecs = np.linalg.eigh(cov)  # (n,3) 升序,第 0 = 最小特征向量
    nrm = evecs[:, :, 0]
    # **可信门(位对齐契约,C++ 同式)**:λ1 ≤ planarity·λ2 的邻域退化,最小特征向量
    # 方向不定(LAPACK/Jacobi 会挑到不同向量,差 O(1))→ 置零向量,ICP 侧剔除。
    ok = evals[:, 1] > ICP_NORMAL_PLANARITY * evals[:, 2]
    nrm = np.where(ok[:, None], nrm, 0.0)
    # **符号规范化**:特征向量正负号任意(LAPACK 与 Jacobi 取号不同),而法向**有号**
    # (A[i,:3]=p×n、b=−n·(p−ref)),翻号会让解翻号 → 旋转完全错。
    # 规则:|分量| 最大者置正,并列取最先出现的分量(双方同式,可逐位复算)。
    kmax = np.argmax(np.abs(nrm), axis=1)
    sgn = np.sign(nrm[np.arange(n), kmax])
    sgn[(sgn == 0.0) | ~ok] = 1.0
    return nrm * sgn[:, None]


def estimate_transform_gn(
    src: np.ndarray, ref: np.ndarray, ref_n: np.ndarray, lam: float = ICP_REG_LAM
) -> tuple[np.ndarray, np.ndarray]:
    """点面 ICP 一步:最小二乘 (ω, v) 解正则化法方程。

    与 multilidar._estimate_transform 同几何,但**不用 lstsq/SVD**——
    (AᵀA + λI)x = Aᵀb,C++ 移植用 Cholesky 位对齐(唯一允许舍入偏差 ~1e-12)。
    A[i,:3] = p_i × n_i, A[i,3:] = n_i;b[i] = −n_i·(p_i − ref_i)。
    """
    src = np.asarray(src, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    ref_n = np.asarray(ref_n, dtype=np.float64)
    if src.ndim == 2 and src.shape[1] > 3:
        src = src[:, :3]  # (N,4+) → (N,3)(与 multilidar 口径一致)
    # **退化邻域剔除**:estimate_normals 把 λ1/λ2 过小的邻域法向置零(方向不定),
    # 这里显式过滤 —— 零行对 AtA/Atb 贡献本就恒为 0,但白占行数会污染残差/overlap 口径。
    keep = np.einsum("ij,ij->i", ref_n, ref_n) > 0.0
    src, ref, ref_n = src[keep], ref[keep], ref_n[keep]
    if len(src) == 0:
        return np.eye(3), np.zeros(3)
    A = np.column_stack([np.cross(src, ref_n), ref_n])  # 向量化,等价逐行 np.cross
    b = -np.einsum("ij,ij->i", ref_n, src - ref[:, :3])
    AtA = A.T @ A + lam * np.eye(6)
    Atb = A.T @ b
    x = np.linalg.solve(AtA, Atb)  # 正则化保证可解(条件数有界)
    w, v = x[:3], x[3:]
    R = _rodrigues(w)
    return R, v


def _rodrigues(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-9:
        return np.eye(3) + _skew(w)
    wx = _skew(w / th)
    return np.eye(3) + np.sin(th) * wx + (1 - np.cos(th)) * (wx @ wx)


def icp_odometry(
    src: np.ndarray,
    ref: np.ndarray,
    init_T: np.ndarray,
    *,
    seed: np.ndarray | None = None,
    max_iter: int = ICP_MAX_ITER,
) -> dict:
    """帧间点面 ICP:src→ref,返回 {T, T_delta, rmse_curve, delta_curve, converged, iters, overlap, rmse_final, failed}。

    **两个出口,语义不同,勿混用(曾因此把 ATE 算大 94×)**:
    - `T_delta` = **点映射 src→ref**:把 src 系坐标搬到 ref 系(`p_ref = T_delta·p_src`)。
      这是 ICP 直接解出的量,`estimate_transform_gn` 的增量就是它。
    - `T` = **位姿**:`init_T` 视为上一帧位姿 P_{k-1},返回 `T = init_T @ inv(T_delta)` = P_k。
      推导:src = 帧 k−1 的点云、ref = 帧 k 的点云 ⇒ T_delta = P_k⁻¹P_{k−1}
      ⇒ P_k = P_{k−1}·(P_{k−1}⁻¹P_k) = P_{k−1}·inv(T_delta)。
      **旧实现写的是 `T_delta @ init_T`,把点映射当位姿左乘** —— 平移无旋转时看着像在
      累加,一转弯就发散(实测 206 m 序列 ATE 17.5 m vs 修正后 0.19 m,差 94×)。

    - **seed = 恒速先验的位姿增量 ΔP = P_{k-2}⁻¹P_{k-1}**(帧0→1 用 None/恒等):函数内部
      取逆得到点映射预测 `cur = inv(ΔP)·src`,把 src 搬到 ref 附近 → 最近邻落在
      rad 0-1 壳,稀疏/转弯段不触发全云扫描(快 ~5×)。**只影响迭代起点,不改解**。
    - 法向对 ref 逐帧算一次(复用 estimate_normals)。
    - overlap < ICP_FAIL_OVERLAP → failed=True(链用恒速先验兜底,不污染)。
    """
    src_f = np.asarray(src[:, :3], dtype=np.float64)
    ref_f = np.asarray(ref[:, :3], dtype=np.float64)
    if len(src_f) == 0 or len(ref_f) == 0:
        return _empty_icp_result(init_T, failed=True)
    ref_n = estimate_normals(ref_f)
    # 累计**点映射** R_acc/t_acc 从 seed 起步(无 seed = 恒等);首迭代 cur = seed_map·src。
    # **seed 是位姿增量 → 这里取逆换成点映射**(方向错会让迭代起点偏 2×,实测收敛到次优)。
    if seed is None:
        R_acc, t_acc = np.eye(3), np.zeros(3)
        cur = src_f.copy()
    else:
        seed_map = np.linalg.inv(np.asarray(seed, dtype=np.float64))
        R_acc, t_acc = seed_map[:3, :3].copy(), seed_map[:3, 3].copy()
        cur = (R_acc @ src_f.T).T + t_acc
    rmse_curve: list[float] = []
    delta_curve: list[float] = []
    converged = False
    for _ in range(max_iter):
        # 最近邻 + 对应残差(批量路径,与标量 GridHash.nearest 逐位一致)
        idx, _ = nearest_batch(ref_f, cur, GRID_CELL)
        nb = ref_f[idx]
        nb_n = ref_n[idx]
        res = np.abs(((cur - nb) * nb_n).sum(-1))
        rmse_curve.append(float(np.sqrt((res**2).mean())))
        R, t = estimate_transform_gn(cur, nb, nb_n)
        delta = float(np.linalg.norm(R - np.eye(3)) + np.linalg.norm(t))
        delta_curve.append(delta)
        cur = (R @ cur.T).T + t
        R_acc = R @ R_acc
        t_acc = R @ t_acc + t
        if delta < ICP_TOL_DELTA:
            converged = True
            break
    T_delta = np.eye(4)
    T_delta[:3, :3] = R_acc
    T_delta[:3, 3] = t_acc
    # 位姿 = init_T @ inv(T_delta)(推导见 docstring;勿改回 T_delta @ init_T)
    T = init_T @ np.linalg.inv(T_delta)
    # 重叠度:变换后点云在 ref 网格 0.3m 内的比例(批量最近邻,O(N) 内存)
    _, d = nearest_batch(ref_f, cur, GRID_CELL)
    overlap = float((d < ICP_OVERLAP_RADIUS**2).mean())
    failed = overlap < ICP_FAIL_OVERLAP
    return {
        "T": T,
        "T_delta": T_delta,
        "rmse_curve": rmse_curve,
        "delta_curve": delta_curve,
        "converged": converged,
        "iters": len(rmse_curve),
        "overlap": overlap,
        "rmse_final": rmse_curve[-1] if rmse_curve else float("inf"),
        "failed": failed,
        "seed_applied": seed is not None,
    }


def _empty_icp_result(init_T: np.ndarray, *, failed: bool) -> dict:
    return {
        "T": np.array(init_T, dtype=np.float64).copy(),
        "T_delta": np.eye(4),  # 空云 = 点映射恒等 → T = init_T
        "rmse_curve": [],
        "delta_curve": [],
        "converged": False,
        "iters": 0,
        "overlap": 0.0,
        "rmse_final": float("inf"),
        "failed": failed,
        "seed_applied": False,
    }


# ---------------------------------------------------------------------------
# ScanContext 回环(点计数描述子)
# ---------------------------------------------------------------------------
def desc_scan_context(
    points: np.ndarray,
    num_rings: int = SC_NUM_RINGS,
    num_sectors: int = SC_NUM_SECTORS,
    max_range: float = SC_MAX_RANGE,
) -> np.ndarray:
    """点云 → ScanContext 描述子 (num_rings, num_sectors) 点计数。

    径向分环(离原点 0..max_range)、周向分扇;每格子 = 命中点数(不取列高,
    平地列高≈噪声)。60 扇 × 20 环 → 行(环)归一化后余弦距离。
    """
    pts = np.asarray(points[:, :3], dtype=np.float64)
    d = np.linalg.norm(pts, axis=1)
    ok = d < max_range
    pts = pts[ok]
    if len(pts) == 0:
        return np.zeros((num_rings, num_sectors))
    az = np.arctan2(pts[:, 1], pts[:, 0])  # [-π, π]
    sec = np.floor((az + np.pi) / (2 * np.pi) * num_sectors).astype(np.int64) % num_sectors
    ring = np.floor(d[ok] / max_range * num_rings).astype(np.int64).clip(0, num_rings - 1)
    desc = np.zeros((num_rings, num_sectors))
    np.add.at(desc, (ring, sec), 1.0)
    return desc


def ring_key(desc: np.ndarray) -> np.ndarray:
    """描述子 → 每环 argmax 扇区(紧凑签名,回环候选粗筛)。"""
    return np.argmax(desc, axis=1)


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    """整描述子余弦距离 [0,1](L2 归一化点计数;行内不归一,保方向信息)。"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - float(np.einsum("ij,ij->", a, b) / (na * nb))


def sc_dist(a: np.ndarray, b: np.ndarray) -> float:
    """两个描述子余弦距离 [0,1];列滚动不变(取 min over shifts)。"""
    if a.shape != b.shape:
        raise ValueError(f"描述子形状不一致 {a.shape} vs {b.shape}")
    n_s = a.shape[1]
    best = 1.0
    for s in range(n_s):
        d = _cos_dist(a, np.roll(b, s, axis=1))
        if d < best:
            best = d
    return float(best)


def match_sc(descs: np.ndarray, query: np.ndarray) -> tuple[int, int, float]:
    """在描述子序列里找与 query 最像的一个 → (索引, 滚动移位, 距离)。"""
    best_i, best_s, best_d = -1, 0, 1.0
    for i, d in enumerate(descs):
        for s in range(d.shape[1]):
            dist = _cos_dist(query, np.roll(d, s, axis=1))
            if dist < best_d:
                best_d, best_i, best_s = dist, i, s
    return best_i, best_s, best_d


def sc_candidates(
    descs: np.ndarray,
    query: np.ndarray,
    node_idx: int,
    *,
    top_n: int = SC_TOP_N,
    min_gap: int = SC_MIN_GAP_NODES,
    thresh: float = SC_SIM_THRESH,
) -> list[tuple[int, int, float]]:
    """候选回环:描述子距离 < thresh ∧ 关键帧号差 ≥ min_gap,取距离最小 top_n。

    返回 [(cand_idx, shift, dist)] 按距离升序。与自身/邻近帧被 min_gap 排除。
    """
    cands: list[tuple[int, int, float]] = []
    for i in range(len(descs)):
        if abs(i - node_idx) < min_gap:
            continue
        for s in range(descs[i].shape[1]):
            dist = _cos_dist(query, np.roll(descs[i], s, axis=1))
            if dist < thresh:
                cands.append((i, s, dist))
    cands.sort(key=lambda e: e[2])
    return cands[:top_n]


# ---------------------------------------------------------------------------
# 位姿图 G-N(纯 numpy,无 g2o)
# ---------------------------------------------------------------------------
@dataclass
class Edge:
    """图边:i→j 相对位姿 T_ij(T_i⁻¹T_j)+ 权重。"""

    i: int
    j: int
    T: np.ndarray  # 4×4
    weight: float = 1.0


def _log_so3(R: np.ndarray) -> np.ndarray:
    th = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if th < 1e-9:
        return np.zeros(3)
    return (
        th
        / (2 * np.sin(th))
        * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    )


def _adjoint(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    tx = _skew(t)
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = tx @ R
    return Ad


def pose_graph_optimize(
    nodes_init: list[np.ndarray],
    edges: list[Edge],
    *,
    iters: int = PGO_ITERS,
    lam0: float = PGO_LAMBDA0,
    anchor_node: int = 0,
) -> list[np.ndarray]:
    """位姿图 G-N(LM):min Σ_e w_e·‖Log(Z_e⁻¹ T_i⁻¹ T_j)‖²,节点0 锚恒等。

    右扰动局部坐标:残差 e_ij = Log(Z_ij⁻¹ T_i⁻¹ T_j),J_i = −Ad(Z_ij⁻¹),J_j = I
    (右扰动 → 无 Ad 共轭项,有限差分单测钉死符号)。LM:(H+λdiagH)δ = −g。
    返回优化后的节点位姿列表(与 nodes_init 同序)。
    """
    # 深拷贝避免改调用方
    nodes = [np.array(T, dtype=np.float64).copy() for T in nodes_init]
    n = len(nodes)
    dim = 6 * n
    lam = lam0
    for _ in range(iters):
        H = np.zeros((dim, dim))
        g = np.zeros(dim)
        # 逐边装配 (i,j) 块:残差 e_ij = Log(Z_ij⁻¹ T_i⁻¹ T_j)(右扰动),
        # J_i = −Ad(Z_ij⁻¹),J_j = I(等价 g2o EdgeSE3 右扰动约定)。
        for e in edges:
            i, j, Z, w = e.i, e.j, e.T, e.weight
            Ti, Tj = nodes[i], nodes[j]
            J_i = -_adjoint(np.linalg.inv(Z))
            J_j = np.eye(6)
            Ti_inv_Tj = np.linalg.inv(Ti) @ Tj
            err = np.concatenate(
                [
                    _log_so3(Z[:3, :3].T @ Ti_inv_Tj[:3, :3]),
                    Z[:3, :3].T @ (Ti_inv_Tj[:3, 3] - Z[:3, 3]),
                ]
            )
            Hii = w * (J_i.T @ J_i)
            Hij = w * (J_i.T @ J_j)
            Hjj = w * (J_j.T @ J_j)
            gi = w * (J_i.T @ err)
            gj = w * (J_j.T @ err)
            si, sj = 6 * i, 6 * j
            H[si : si + 6, si : si + 6] += Hii
            H[si : si + 6, sj : sj + 6] += Hij
            H[sj : sj + 6, si : si + 6] += Hij.T
            H[sj : sj + 6, sj : sj + 6] += Hjj
            g[si : si + 6] += gi
            g[sj : sj + 6] += gj
        # 锚定节点0(到恒等)
        si = 6 * anchor_node
        H[si : si + 6, si : si + 6] += PGO_ANCHOR_WEIGHT * np.eye(6)
        g[si : si + 6] += PGO_ANCHOR_WEIGHT * np.concatenate(
            [_log_so3(nodes[anchor_node][:3, :3]), nodes[anchor_node][:3, 3]]
        )
        # LM 求解
        Hlm = H.copy()
        diag = np.diag(H).copy()
        np.fill_diagonal(Hlm, diag * (1 + lam))
        try:
            delta = np.linalg.solve(Hlm, -g)
        except np.linalg.LinAlgError:
            lam = min(lam * PGO_DAMP_UP, PGO_LAM_MAX)
            continue
        delta = np.asarray(delta).reshape(n, 6)
        # 应用更新(右扰动)
        new_nodes = []
        for k in range(n):
            w_k, v_k = delta[k, :3], delta[k, 3:]
            new_nodes.append(nodes[k] @ twist_exp(w_k, v_k))
        # 收敛检查
        if float(np.max(np.abs(delta))) < PGO_CONV_DELTA:
            nodes = new_nodes
            break
        nodes = new_nodes
        lam = max(lam / PGO_DAMP_DOWN, 1e-12)
    return nodes


def closure_error(poses: list[np.ndarray]) -> dict:
    """轨迹漂移率:‖T_end·T_start⁻¹ 平移‖ / 总弧长。无 GT 时用自洽口径。

    poses = 链式 T_0→k;闭合误差 = 末帧相对首帧的相对位姿平移量(理想 0)。
    返回 {drift_m, arc_m, drift_rate}。
    """
    if len(poses) < 2:
        return {"drift_m": 0.0, "arc_m": 0.0, "drift_rate": float("nan")}
    T_end_start = relative_transform(poses[0], poses[-1])
    drift_m = float(np.linalg.norm(T_end_start[:3, 3]))
    arc = 0.0
    for a, b in zip(poses[:-1], poses[1:], strict=True):
        arc += float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
    return {"drift_m": drift_m, "arc_m": arc, "drift_rate": drift_m / arc if arc > 0 else float("nan")}

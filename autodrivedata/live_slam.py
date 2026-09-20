"""在线激光 SLAM 会话(纯值,不 import carla / torch):tick 循环外的增量重建。

把 `bin/slam_odometry.py` 的链式约定**逐字**封成一个可增量喂帧的会话对象
(`LiveSlam.push`),供 `bin/live_studio.py --slam` 驱动。

**成本口径订正(2026-09-20,Plan2.md §P-L.2)**:原计划写"离线 ICP 0.78 s/帧 ⇒ 必须
worker 线程"。实测 0.78 s 是 400 帧**含转弯/重访的平均值**(`icp_stats.json`),而在线
逐帧(gap 1)只有 **0.15–0.35 s**(`/tmp/probe_live_sweep.py`,CARLA 语义 LiDAR 116k 点)。
更关键的是 worker 线程会被 **GIL** 压到 eff 0.04–0.24(studio 的 overlay/拼图/HUD 是纯
Python 字节码,持 GIL 不放),实测 studio 里同一对点云 ICP 2.38 s vs 主线程同步 0.15–0.35 s。
⇒ **默认同步执行**(`SlamWorker(sync=True)`),异步线程留作可选项。

**链式约定(逐字复用 `bin/slam_odometry.py:87-108`,勿另立)**:

- 帧 0 = 恒等;帧 k = `icp_odometry(prev_down, down, poses[-1], seed=delta_prev)`
- `seed` = **位姿增量** ΔP = P_{k-2}⁻¹P_{k-1}(函数内部取逆当迭代起点);
  **不可再把 seed 乘进 init** —— 那等于恒速先验叠加两次(实测轨迹按 k² 发散)
- 位姿非有限(NaN/Inf)才回退到 `init`;`failed`(重叠不足)只计数,不改链
- 输入点云一律先 `voxel_downsample`(与 `slam_odometry` 同体素口径)

**线程模型**:`push` 与 `snapshot` 可由不同线程调用(同步模式下其实是同一个)。

- ICP 是慢段(0.15–0.35 s),**在锁外算**;只在提交(追加位姿 / 合并地图)时短暂持锁。
  故 `snapshot()` 最多等一次地图合并(~几十 ms),不会等一整个 ICP。
- `_push_lock` 让 push 串行(单 worker 时无争用,多调用方时防状态撕裂)。
- `snapshot()` 返回**不可变副本**(位姿 list 拷贝),调用方拿到后自己动不会污染会话。

**地图参考系 = LiDAR-0 系**(链式位姿的天然出口:`T_0→k` 把帧 k 的点搬到帧 0)。
要画到**当前** ego 系须再过一道 `_lidar0_to_ego()`(CARLA 世界位姿 + 手性共轭 + 杆臂,
与 `bin/eval_slam.py` 同一口径);**渲染循环滞后于 SLAM**,故每次取地图都要传当前 ego
位姿,不能用快照里那份。

**锚定 E_0 是必需的**(2026-09-19 修正):帧 0 的 ego **世界**位姿把"LiDAR-0 系"钉到
世界系上;没有它就只能在 LiDAR-0 系里自说自话,摆不到当前 ego 系。故 `push` 要收
`ego_pose`(CARLA 4×4 ego→world),首帧那份存为 `_ego0`。

**`SlamWorker` 也在这里**(不在 bin):它是纯逻辑(队列 + 止损 + 滞后算术,零 carla),
放纯值库才能单测 —— "止损生效 ⇒ 滞后有界"是 B 期最关键的判据,不能只靠跑一遍看日志。
bin 侧只负责挂 LiDAR、把帧喂进 `offer`、读 `stats`。

单测:tests/test_live_slam.py(合成序列;与 `icp_odometry` 逐帧同输入同输出对拍)。
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from autodrivedata.accum import voxel_downsample
from autodrivedata.slam import DOWNSAMPLE_VOXEL, ICP_MAX_ITER, icp_odometry
from autodrivedata.slam_eval import M_FLIP, lever_matrix, lidar_pose_to_ego

__all__ = ["LiveSlam", "SlamWorker", "ego_from_lidar0", "relative_transform"]


def relative_transform(Ta: np.ndarray, Tb: np.ndarray) -> np.ndarray:
    """T_ab = Ta⁻¹·Tb(链式 Δ 用;与 `bin/slam_odometry.relative_transform` 同式)。"""
    return np.linalg.inv(Ta) @ Tb


def ego_from_lidar0(T_lidar0_ego: np.ndarray, lever: np.ndarray | None = None) -> np.ndarray:
    """LiDAR-0 系位姿 → ego 相对位姿:`L·M·T·M @ inv(L)`(与 `eval_slam` 同口径)。

    **方向勿凭直觉**:右端是 `inv(L)` 不是 `L` —— 不带杆臂 ATE 0.4589 m vs 带杆臂
    0.1877 m(2.44×),实测见 Plan2.md §P-H.1。左端那个 `L` 是 2026-09-19 补的:
    原式漏了它,等价于把杆臂当成恒定偏移(ATE 对齐口径),`f(I)` 会给出 `[-1.2,0,-1.65]`
    而非 `[0,0,0]`。

    实现直接复用 `slam_eval.lidar_pose_to_ego`(同一处定义);`lever=None` 走它的默认挂点。
    """
    if lever is None:
        return lidar_pose_to_ego(np.asarray(T_lidar0_ego, dtype=np.float64))
    return lidar_pose_to_ego(np.asarray(T_lidar0_ego, dtype=np.float64), lever)


class LiveSlam:
    """增量激光 SLAM 会话:逐帧 `push` 点云,随时 `snapshot` 取只读状态。

    `map_voxel` 是**累积地图**的体素边长(默认与逐帧下采样同 0.5 m);调大省内存与
    BEV 画点时间,调小保细节。`map_max_points` 是硬上限:超过就整体重下采样(不是丢点)。
    """

    def __init__(
        self,
        voxel: float = DOWNSAMPLE_VOXEL,
        max_iter: int = ICP_MAX_ITER,
        map_voxel: float | None = None,
        map_max_points: int = 400_000,
    ) -> None:
        self.voxel = float(voxel)
        self.max_iter = int(max_iter)
        self.map_voxel = float(map_voxel if map_voxel else voxel)
        self.map_max_points = int(map_max_points)
        self._push_lock = threading.Lock()
        self._lock = threading.Lock()
        self._poses: list[np.ndarray] = []
        self._map: np.ndarray | None = None  # LiDAR-0 系累积点云 (M,3),KITTI 手性
        self._ego0: np.ndarray | None = None  # 帧 0 的 ego 世界位姿(CARLA 4×4):锚定 LiDAR-0 系
        self._prev_down: np.ndarray | None = None
        self._delta_prev = np.eye(4)
        self._n_nan = 0
        self._n_failed = 0
        self._n_dead = 0
        self._rmse_sum = 0.0
        self._overlap_sum = 0.0
        self._icp_s = 0.0
        self._t0 = time.time()

    # ------------------------------------------------------------ 推进

    def push(
        self,
        points: np.ndarray,
        t_stamp: float = 0.0,
        ego_pose: np.ndarray | None = None,
        dead_reckon: bool = False,
    ) -> dict:
        """喂一帧点云 `(N,3+)`(LiDAR 系,KITTI velodyne 约定)→ 本帧 ICP 结果摘要。

        返回摘要 dict(不是位姿本身 —— 位姿经 `snapshot()` 取,避免两处口径)。
        `t_stamp` 只用于 `lag_s` 计算(studio 传 tick 时刻的 wall time)。

        `ego_pose` = **该帧 ego 的世界位姿**(CARLA 4×4,ego 局部 → 世界)。首帧那份存为
        `_ego0`,把 LiDAR-0 系钉到世界系上;不给则 `map_in_ego_frame`/`traj_in_ego_frame`
        不可用(会报错,而不是静默给错坐标)。后续帧的 ego 位姿不参与链式解算(链是纯
        ICP 的),只在取地图时用**当前**那一份。

        `dead_reckon=True` = **不做 ICP**,位姿直接按恒速先验外推 `poses[-1] @ delta_prev`,
        地图照合并、`prev_down` 照推进。这是**丢帧后的止损路径**(见 `SlamWorker` 的
        "为什么必须止损"):ICP 成本随 `prev_down` 与当前帧的**帧间隙**急剧上升
        (实测体素 1.0 下 0.8 m 间隙 0.2 s → 8 m 间隙 2.8 s → 32 m 间隙 39 s),而丢帧会让
        间隙无界增长 ⇒ "丢帧→更慢→更多丢帧"的正反馈。止损把间隙**强行钉回 1 帧**,
        代价是这一帧的位姿带恒速外推的漂移(如实计入 `n_dead`)。
        """
        down = voxel_downsample(np.asarray(points, dtype=np.float64), self.voxel)
        with self._push_lock:
            k = len(self._poses)
            if k == 0:
                T = np.eye(4)
                res = {
                    "rmse_final": 0.0,
                    "overlap": 1.0,
                    "converged": True,
                    "iters": 0,
                    "failed": False,
                }
            elif dead_reckon:
                # 止损:恒速外推,**不调 ICP**。`_delta_prev` 保持上一次 ICP 的增量不动
                # (它仍是"最近一次真实的帧间位姿增量",比清零更可信)。
                T = self._poses[-1] @ self._delta_prev
                self._n_dead += 1
                res = {
                    "rmse_final": 0.0,
                    "overlap": 0.0,
                    "converged": False,
                    "iters": 0,
                    "failed": False,
                }
            else:
                assert self._prev_down is not None  # k>0 时必有前帧
                init = self._poses[-1]
                t_a = time.time()
                res = icp_odometry(self._prev_down, down, init, seed=self._delta_prev, max_iter=self.max_iter)
                self._icp_s += time.time() - t_a
                T = res["T"]
                if not np.isfinite(T).all():
                    self._n_nan += 1
                    T = init
                if res["failed"]:
                    self._n_failed += 1
                self._rmse_sum += res["rmse_final"]
                self._overlap_sum += res["overlap"]
                self._delta_prev = relative_transform(self._poses[-1], T)
            self._prev_down = down
            with self._lock:
                if self._ego0 is None and ego_pose is not None:
                    self._ego0 = np.asarray(ego_pose, dtype=np.float64).copy()
                self._poses.append(np.asarray(T, dtype=np.float64))
                self._merge_map(down, T)
        return {
            "frame": k,
            "t_stamp": float(t_stamp),
            "rmse_final": float(res["rmse_final"]),
            "overlap": float(res["overlap"]),
            "converged": bool(res["converged"]),
            "iters": int(res["iters"]),
            "failed": bool(res["failed"]),
            "dead_reckon": bool(dead_reckon),
        }

    def _merge_map(self, down: np.ndarray, T: np.ndarray) -> None:
        """把本帧点云按位姿摆到 LiDAR-0 系并并入累积地图(调用方已持 `_lock`)。"""
        pts = (T[:3, :3] @ down[:, :3].T).T + T[:3, 3]
        self._map = pts if self._map is None else np.vstack([self._map, pts])
        # 超上限就整体重下采样(**不是丢点**):体素边长随规模自适应放大
        if self._map.shape[0] > self.map_max_points:
            self.map_voxel *= 2.0
            self._map = voxel_downsample(self._map, self.map_voxel)[:, :3]

    # ------------------------------------------------------------ 读取

    def snapshot(self) -> dict:
        """只读快照:位姿 list **拷贝**、计数、滞后量。UI 线程每次取一份,不共享可变对象。"""
        with self._lock:
            n = len(self._poses)
            poses = [p.copy() for p in self._poses]
            n_pts = 0 if self._map is None else int(self._map.shape[0])
            icp_s = self._icp_s
            has_anchor = self._ego0 is not None
        return {
            "n_frames": n,
            "poses": poses,
            "n_map_points": n_pts,
            "n_nan": self._n_nan,
            "n_failed": self._n_failed,
            "n_dead": self._n_dead,
            "mean_rmse": self._rmse_sum / max(n - 1, 1),
            "mean_overlap": self._overlap_sum / max(n - 1, 1),
            "icp_s": icp_s,
            "wall_s": time.time() - self._t0,
            "has_anchor": has_anchor,
        }

    # ------------------------------------------------------------ 当前 ego 系

    def _lidar0_to_ego(self, ego_pose_world: np.ndarray) -> np.ndarray:
        """LiDAR-0 系 → **当前** ego 系(KITTI 手性)的 4×4 换算矩阵 `T`。

        推导(`E_k` = ego 世界位姿、`L` = 挂点、`M` = 手性共轭、`P_k` = 链式位姿):

            LiDAR-k 局部 → 世界 = `E_k·L·M`
            `P_k = inv(G_0)·G_k = M·inv(E_0·L)·(E_k·L)·M`   (帧 0 = 恒等)

        地图存在 LiDAR-0 系(KITTI 手性),故 `p_world = E_0·L·M·p_map`。要换成
        **当前** ego 系的 KITTI 坐标(`M·inv(E_now)·p_world`):

            `T = M·inv(E_now)·E_0·L·M`

        锚点自检:`E_now = E_0` 时 `T = M·L·M`(即"ego 在 LiDAR-0 系的位置",纯换算,
        不含运动)。实测对合成真值:最近邻 max 0.318 m / mean 0.068 m(体素 0.5 ⇒
        该量级即"正确");**旧式 `inv(ego_from_lidar0(P_k))` 给 max 66.7 / mean 54.7**。
        """
        if self._ego0 is None:
            raise RuntimeError("LiveSlam 缺帧 0 的 ego 世界位姿:push 时须传 ego_pose(用于锚定)")
        L = lever_matrix()
        return M_FLIP @ np.linalg.inv(np.asarray(ego_pose_world, dtype=np.float64)) @ self._ego0 @ L @ M_FLIP

    def map_in_ego_frame(self, ego_pose_world: np.ndarray) -> np.ndarray:
        """累积地图点 → **当前** ego 系的 **KITTI 手性**坐标 `(M,3)`(x 前 / y 左 / z 上)。

        为什么输出 KITTI 手性而不是 CARLA:`mapviz.bev_panel` 的窗口 `BEV_Y` 标注是
        "y 左向"(`bev_px` 把大 y 画在上方)—— 面板口径是 KITTI。SLAM 链是 CARLA 手性,
        故这里补最后一道 `M3`(公式里的右端 `M`)。

        `ego_pose_world` = **当前** ego 世界位姿(CARLA 4×4,`collect_slam.ego_pose_matrix`
        那种)。**渲染循环滞后于 SLAM**,故这个参数每次取地图都要重传;用快照里那份会让
        BEV 上的点按旧位姿摆,视觉上"甩尾"。
        """
        with self._lock:
            m = None if self._map is None else self._map.copy()
        if m is None or m.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        T = self._lidar0_to_ego(ego_pose_world)
        return (T[:3, :3] @ m.T).T + T[:3, 3]

    def traj_in_ego_frame(self, ego_pose_world: np.ndarray) -> np.ndarray:
        """链式位姿序列 → **当前** ego 系的 **KITTI 手性**位置 `(N,3)`,供 BEV 画轨迹。

        每帧的世界位置:`E_0·L·M·P_k·M·inv(L)` 的平移块(把链式位姿还原成 ego 世界
        位姿),再乘 `M·inv(E_now)` 落到当前 ego 系 —— 与 `map_in_ego_frame` 同一出口口径。

        **不是** `T @ P_k[:3,3]`:那会漏掉 `P_k` 的旋转块(它把"原点"从 LiDAR 搬到 ego),
        实测末点差 1.67 m 且漏出 z = 1.629 ≈ 杆臂高度。
        """
        with self._lock:
            poses = [p.copy() for p in self._poses]
            ego0 = None if self._ego0 is None else self._ego0.copy()
        if not poses:
            return np.zeros((0, 3), dtype=np.float64)
        if ego0 is None:
            raise RuntimeError("LiveSlam 缺帧 0 的 ego 世界位姿:push 时须传 ego_pose(用于锚定)")
        L, iL = lever_matrix(), np.linalg.inv(lever_matrix())
        A = M_FLIP @ np.linalg.inv(np.asarray(ego_pose_world, dtype=np.float64))
        out = np.empty((len(poses), 3), dtype=np.float64)
        for k, P in enumerate(poses):
            E_k = ego0 @ L @ M_FLIP @ P @ M_FLIP @ iL  # 链式位姿 → ego 世界位姿
            out[k] = (A @ E_k)[:3, 3]
        return out


class SlamWorker:
    """`LiveSlam.push` 的执行者:**有界丢旧**队列 + 滞后止损,两种模式(纯逻辑,零 carla)。

    **为什么原计划要 worker 线程**(Plan2.md §P-L):离线实测 ICP 0.78 s/帧 vs 渲染
    0.2 s/tick ⇒ 若在 tick 循环里同步跑,帧率会掉到 1 fps 以下。**该前提已被实测推翻**:
    0.78 s 是 400 帧含转弯/重访的**平均值**,而在线逐帧(gap 1)ICP 只有 0.15–0.35 s
    (2026-09-20,`/tmp/probe_live_sweep.py`)。

    **为什么默认同步**(2026-09-20 决定性实验,见 Plan2.md §P-L.2):worker 线程跑同一对
    点云的 ICP,主线程施加不同负载:

    | 主线程负载 | wall | thread-CPU / wall |
    |---|---|---|
    | 空闲 | 0.19–0.26 s | 0.86–1.00 |
    | PIL 画 6 路 + JPEG | 0.24–0.30 s | 0.71–0.79 |
    | CARLA `world.tick()` | 0.24–0.29 s | 0.79–0.90 |
    | **纯 Python 小矩阵自旋** | **13.6–15.3 s** | **0.04** |

    ⇒ 真凶是 **GIL**:studio 主线程每 tick 的 overlay/拼图/HUD 是纯 Python 字节码,
    持 GIL 不放,把 worker 压到 eff 0.24(studio 实测 `process_time` 12.9 s vs wall 2.4 s)。
    `threadpoolctl` 报 OpenBLAS 12 线程,但钉 `OPENBLAS_NUM_THREADS=1` 不改结论 ——
    不是 BLAS 线程池。**把 push 挪回主线程(同步)后 eff 0.95、ICP 0.15–0.35 s、
    44 帧零丢帧、lag 0**。代价 = 帧率 ~2.4 fps(SLAM 的 ICP 与渲染在同一个 tick 里排队)。

    **滞后止损(`max_gap`,两种模式都生效)**:原不变量"有界丢旧队列 ⇒ 有界滞后"
    **对 ICP 不成立** —— 队列有界的是**深度**,不是 `prev_down` 与当前帧的**间隙**,
    而 ICP 成本随间隙爆炸(体素 1.0:0.8 m 0.2 s → 2.4 m 2.1 s → 8 m 2.8 s → 32 m 39 s
    → 49 m 79 s)。丢帧 ⇒ 间隙更大 ⇒ 更慢 ⇒ 更多丢帧,正反馈无界。止损 = 帧号差超
    `max_gap` 时那一帧**不做 ICP**,直接恒速外推(`push(dead_reckon=True)`),`prev_down`
    照推进 ⇒ 下一帧回到 gap 1。被止损的帧如实计入 `n_dead`。

    **滞后口径**:`lag = 已 tick 帧号 − 已处理帧号`。**不能用 `n_offered − n_processed`** ——
    丢旧之后那个差值会随丢帧数无界增长,看着像故障其实队列一直是满的。帧号差才反映
    "SLAM 落后当前时刻多远",且丢旧时它自动收敛(worker 跳过被丢的帧号)。

    **为什么放纯值库而不是 bin**:滞后有界是 B 期最关键的判据,要能单测(合成"消费比生产
    慢"的假 push,断言滞后收敛),不能只靠跑一遍看日志。bin 侧只挂 LiDAR / 喂 `offer` / 读 `stats`。
    """

    def __init__(self, slam: LiveSlam, maxsize: int = 3, max_gap: int = 3, sync: bool = True) -> None:
        self.slam = slam
        self.max_gap = int(max_gap)
        self.sync = bool(sync)
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.n_offered = 0
        self.n_dropped = 0
        self.n_processed = 0
        self.n_failed = 0
        self.n_dead = 0
        self.last_done_idx = -1
        self.last_done_t = 0.0

    def start(self) -> None:
        """同步模式不启线程(`offer` 里就地跑)。"""
        if self.sync:
            return
        self._thread = threading.Thread(target=self._run, name="slam-worker", daemon=True)
        self._thread.start()

    def offer(self, points: np.ndarray, t_stamp: float, ego_pose: np.ndarray, frame_idx: int) -> None:
        """提交一帧。**点云与位姿都拷贝**:跨线程不共享可变对象(调用方的数组下一 tick 会被复用)。

        异步模式下队列满则**丢最旧**再放新的 —— 这是"滞后有界"的实现,不是异常路径。
        同步模式直接就地 `_step`(无队列、无丢帧)。
        """
        if self.sync:
            with self._lock:
                self.n_offered += 1
            self._step(np.asarray(points, dtype=np.float64), t_stamp, ego_pose, int(frame_idx))
            return
        item = (
            np.array(points, dtype=np.float64, copy=True),
            float(t_stamp),
            np.array(ego_pose, dtype=np.float64, copy=True),
            int(frame_idx),
        )
        while True:
            try:
                self._q.put_nowait(item)
                break
            except queue.Full:
                try:
                    self._q.get_nowait()
                    with self._lock:
                        self.n_dropped += 1
                except queue.Empty:  # pragma: no cover — 竞态兜底
                    pass
        with self._lock:
            self.n_offered += 1

    def _step(self, pts: np.ndarray, t_stamp: float, ego_pose: np.ndarray, idx: int) -> None:
        """跑一帧 `LiveSlam.push`,按**帧号差**决定是否止损(两种模式共用)。"""
        with self._lock:
            gap = idx - self.last_done_idx
        dead = self.last_done_idx >= 0 and gap > self.max_gap
        try:
            res = self.slam.push(pts, t_stamp=t_stamp, ego_pose=ego_pose, dead_reckon=dead)
        except Exception as exc:  # 不让 worker 静默死掉(死了 HUD 的滞后会一直涨)
            print(f"[slam][error] push 失败:{exc!r}")
            return
        with self._lock:
            self.n_processed += 1
            self.last_done_idx = idx
            self.last_done_t = time.time()
            if res["failed"]:
                self.n_failed += 1
            if res["dead_reckon"]:
                self.n_dead += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                pts, t_stamp, ego_pose, idx = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            self._step(pts, t_stamp, ego_pose, idx)

    def stop(self, timeout: float = 120.0) -> bool:
        """置停止位并 join。返回是否**真的**退出了(False = 卡在一次 ICP 里)。

        **退出纪律**:调用方必须在销毁 world / 传感器**之前**调它 —— 线程还在跑 ICP 时
        world 被销毁会让它读到已释放的 actor。
        """
        self._stop.set()
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        alive = self._thread.is_alive()
        if alive:
            print(f"[slam][warn] worker {timeout:.0f}s 内未退出(可能卡在一次 ICP)")
        return not alive

    def stats(self, tick_idx: int) -> dict:
        """滞后统计(供 HUD 与验收报告)。`tick_idx` = 当前已 tick 的帧号。"""
        with self._lock:
            done, t_done = self.last_done_idx, self.last_done_t
            offered, dropped, processed, failed = (
                self.n_offered,
                self.n_dropped,
                self.n_processed,
                self.n_failed,
            )
            dead = self.n_dead
        return {
            "lag_frames": max(tick_idx - done, 0) if done >= 0 else 0,
            "lag_s": max(time.time() - t_done, 0.0) if done >= 0 else 0.0,
            "n_offered": offered,
            "n_dropped": dropped,
            "n_processed": processed,
            "n_failed": failed,
            "n_dead": dead,
            "queue_depth": self._q.qsize(),
        }

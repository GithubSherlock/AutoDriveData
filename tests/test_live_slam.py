"""在线 SLAM 会话纯值单测:与离线 `bin/slam_odometry.py` 逐帧对拍 + 线程/边界。

锚点分四类:
1. **等价性**(最重要):同一合成序列,`LiveSlam.push` 的位姿序列必须与
   `bin/slam_odometry.py` 的链式约定**逐帧同输入同输出**。离线那个函数是已验证基线,
   在线实现若偏离(例如把 seed 乘进 init、或漏掉 voxel_downsample)本测试立刻红。
2. **几何**:`ego_from_lidar0` 与 `eval_slam.lidar_pose_to_ego` 同式;帧一致性
   (`f(I) = I`,2026-09-19 修正的回归锚)。
3. **当前 ego 系换算**:`map_in_ego_frame` / `traj_in_ego_frame` 对**合成真值**的
   最近邻误差(带真实杆臂 + 手性,不是退化场景)。
4. **边界**:空云/单帧/超上限重下采样/`snapshot` 是副本(改它不影响会话)。

零 carla / 零 torch。
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from autodrivedata.accum import voxel_downsample
from autodrivedata.geometry import carla_rotation_matrix
from autodrivedata.live_slam import LiveSlam, SlamWorker, ego_from_lidar0, relative_transform
from autodrivedata.slam import icp_odometry
from autodrivedata.slam_eval import LIDAR_LEVER, M_FLIP, lever_matrix, lidar_pose_to_ego


def _synthetic_sequence(n: int = 8, seed: int = 3) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """带转向的合成序列(照 `tests/test_slam.py::test_chain_of_turning_motion` 的场景)。

    返回 (frames, gt):frames[k] = 帧 k 传感器系下的点云;**两堵垂直墙**是必需的 ——
    点面残差只在法向有约束,纯地面测不出 x 平移。
    """
    rng = np.random.default_rng(seed)
    g = np.stack([rng.uniform(-6, 6, 300), rng.uniform(-6, 6, 300), np.zeros(300)], 1)
    w1 = np.stack([rng.uniform(-6, 6, 200), np.full(200, -6.0), rng.uniform(0, 3, 200)], 1)
    w2 = np.stack([rng.uniform(-6, 6, 200), np.full(200, 6.0), rng.uniform(0, 3, 200)], 1)
    base = np.concatenate([g, w1, w2])
    gt: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    P = np.eye(4)
    for _ in range(n):
        gt.append(P.copy())
        p = (P[:3, :3].T @ base.T).T + (P[:3, :3].T @ -P[:3, 3])
        frames.append(voxel_downsample(np.hstack([p, np.ones((len(p), 1))]), 0.5)[:, :3])
        c, s = np.cos(np.radians(6.0)), np.sin(np.radians(6.0))
        A = np.eye(4)
        A[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        A[:3, 3] = [0.5, 0.0, 0.0]
        P = P @ A
    return frames, gt


def _offline_poses(frames: list[np.ndarray], voxel: float = 0.5) -> list[np.ndarray]:
    """`bin/slam_odometry.py` 的链式约定逐字复刻(离线基线)。"""
    poses: list[np.ndarray] = []
    prev_down: np.ndarray | None = None
    delta_prev = np.eye(4)
    for k, pts in enumerate(frames):
        down = voxel_downsample(pts, voxel)
        if k == 0:
            prev_down = down
            T = np.eye(4)
        else:
            assert prev_down is not None
            res = icp_odometry(prev_down, down, poses[-1], seed=delta_prev)
            T = res["T"]
            if not np.isfinite(T).all():
                T = poses[-1]
            prev_down = down
        poses.append(T)
        if k > 0:
            delta_prev = relative_transform(poses[-2], poses[-1])
    return poses


class TestParityWithOffline:
    """在线会话 == 离线 `slam_odometry` 的链式约定(**同输入同输出**)。"""

    def test_pose_sequence_matches_offline_frame_by_frame(self):
        frames, _ = _synthetic_sequence(8)
        slam = LiveSlam(voxel=0.5)
        for f in frames:
            slam.push(f)
        online = slam.snapshot()["poses"]
        offline = _offline_poses(frames)
        assert len(online) == len(offline) == 8
        for k, (a, b) in enumerate(zip(online, offline, strict=True)):
            assert np.abs(a - b).max() < 1e-12, f"帧 {k} 在线/离线位姿不一致"

    def test_recovers_synthetic_ground_truth(self):
        """位姿本身也要对:每帧前进 0.5 m + 左转 6° 的合成 GT。"""
        frames, gt = _synthetic_sequence(8)
        slam = LiveSlam(voxel=0.5)
        for f in frames:
            slam.push(f)
        poses = slam.snapshot()["poses"]
        for k in range(1, 8):
            assert np.linalg.norm(poses[k][:3, 3] - gt[k][:3, 3]) < 0.1

    def test_seed_is_not_double_applied(self):
        """**回归**:seed 是迭代起点,不能再乘进 init —— 叠加两次会让轨迹按 k² 发散。

        判据:在线与离线(离线实现里 init 只当 init 用)逐帧一致;若在线把
        `delta_prev` 也乘进 init,两边的平移会差出一个恒速先验的量级。
        """
        frames, gt = _synthetic_sequence(8)
        slam = LiveSlam(voxel=0.5)
        for f in frames:
            slam.push(f)
        poses = slam.snapshot()["poses"]
        # 末帧位移应接近 GT 的 7×0.5 m 量级,不是 k² 发散
        assert np.linalg.norm(poses[-1][:3, 3]) == pytest.approx(np.linalg.norm(gt[-1][:3, 3]), abs=0.2)


class TestEgoFromLidar0:
    def test_matches_eval_slam_convention(self):
        """`ego_from_lidar0` 与 `slam_eval.lidar_pose_to_ego` 必须逐元素一致。

        两处各写一遍是最容易漂的地方(手性共轭 + 杆臂方向),故直接对拍。
        """
        T = np.eye(4)
        T[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        T[:3, 3] = [12.0, -3.0, 0.5]
        np.testing.assert_allclose(ego_from_lidar0(T), lidar_pose_to_ego(T), atol=1e-12)

    def test_identity_anchor_holds(self):
        """**核心回归**(2026-09-19):`f(I) = I` —— 帧 0 是轨迹原点。

        原式 `M·T·M @ inv(L)` 在此给出平移 `[-1.2, 0, -1.65]`(恰是杆臂),即把杆臂
        当成了恒定偏移(ATE 对齐口径,Umeyama 会吸收掉故对齐 ATE 看不出来)。
        """
        np.testing.assert_allclose(ego_from_lidar0(np.eye(4)), np.eye(4), atol=1e-12)

    def test_lever_shows_up_under_rotation(self):
        """带旋转时左端 `L` 不可省:平移 + 90° 偏航,两式差 `‖lever‖ = 2.0402 m`。

        纯平移时 `L` 与 `inv(L)` 的平移项相消(两式只差常数);一旦有旋转,左端那个
        `L` 就把"LiDAR 原点 → ego 原点"的偏移补了回来 —— z 必须回到 0。
        """
        T = np.eye(4)
        T[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        T[:3, 3] = [10.0, 0.0, 0.0]
        got = ego_from_lidar0(T)
        np.testing.assert_allclose(got[:3, 3], [11.2, 1.2, 0.0], atol=1e-9)

    def test_right_lever_is_inverse(self):
        """右端是 `inv(L)` 不是 `L`:两者差 `2×lever`(实测 ATE 0.4589 vs 0.1877 m,2.44×)。

        **注意别把两端搞混**:四因子式的左端 `L` 与右端 `inv(L)` 在纯平移下相消,
        故正确结果恰好等于 `T` 本身的平移;`inv(L)`/`L` 写反的那两式则各偏移 `∓lever`。
        """
        T = np.eye(4)
        T[:3, 3] = [10.0, 0.0, 0.0]
        got = ego_from_lidar0(T)
        np.testing.assert_allclose(got[:3, 3], [10.0, 0.0, 0.0], atol=1e-12)
        # 写反的那两式(右端用 L / 漏左端 L)相差恰好 2×lever
        right_inv = M_FLIP @ T @ M_FLIP @ np.linalg.inv(lever_matrix(LIDAR_LEVER))
        right_L = M_FLIP @ T @ M_FLIP @ lever_matrix(LIDAR_LEVER)
        assert np.linalg.norm(right_inv[:3, 3] - right_L[:3, 3]) == pytest.approx(
            2 * np.linalg.norm(LIDAR_LEVER), abs=1e-9
        )


def _world_scene(seed: int = 7) -> np.ndarray:
    """非退化世界场景:地面 + 两侧墙(y=±8)+ 三道横向隔墙(x=−20/0/20)。

    **必须非退化**:点面残差只在法向有约束,只有地面 + 平行墙时 x 不受约束,
    ICP 会给出一个看着像"公式错"的 0.67 m 偏移(踩过)。

    点数是**刻意压小**的:ICP 成本随点数涨,单测跑分钟级就没人愿意跑。
    """
    rng = np.random.default_rng(seed)
    g = np.stack([rng.uniform(-30, 30, 1500), rng.uniform(-8, 8, 1500), np.zeros(1500)], 1)
    w1 = np.stack([rng.uniform(-30, 30, 500), np.full(500, -8.0), rng.uniform(0, 4, 500)], 1)
    w2 = np.stack([rng.uniform(-30, 30, 500), np.full(500, 8.0), rng.uniform(0, 4, 500)], 1)
    cross = [
        np.stack([np.full(400, xw), rng.uniform(-8, 8, 400), rng.uniform(0, 4, 400)], 1)
        for xw in (-20.0, 0.0, 20.0)
    ]
    return np.concatenate([g, w1, w2, *cross])


def _ego_pose_world(xyz: tuple[float, float, float], yaw_deg: float) -> np.ndarray:
    """CARLA 口径的 ego 世界位姿 4×4(旋转块走 `geometry.carla_rotation_matrix`)。"""
    T = np.eye(4)
    T[:3, :3] = carla_rotation_matrix((0.0, np.radians(yaw_deg), 0.0))
    T[:3, 3] = xyz
    return T


def _apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return (T[:3, :3] @ p.T).T + T[:3, 3]


def _synthetic_ego_sequence(n: int = 10) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """带真实杆臂 + 手性的合成序列:ego 沿 x 前进 1 m/帧并每帧左转 2°。

    返回 (frames, E, world):frames[k] = 帧 k 的 **KITTI 口径** LiDAR 点云;
    E[k] = 该帧 ego 世界位姿(CARLA);world = 世界系真值点云(比对用)。
    """
    M3, Lm = M_FLIP, lever_matrix(LIDAR_LEVER)
    world = _world_scene()
    E = [_ego_pose_world((-15.0 + 1.0 * k, 0.0, 0.0), 2.0 * k) for k in range(n)]
    frames = []
    for Ek in E:
        p_carla = _apply(np.linalg.inv(Ek @ Lm), world)  # LiDAR-k(CARLA) → 点(CARLA)
        frames.append(_apply(M3, p_carla[np.linalg.norm(p_carla, axis=1) < 45.0]))
    return frames, E, world


class TestCurrentEgoFrame:
    """`map_in_ego_frame` / `traj_in_ego_frame` 对**合成真值**的误差(带杆臂 + 手性)。

    判据是**最近邻**:体素 0.5 m ⇒ 正确实现给出 ~0.3 m 量级;旧式(漏左端 `L`,
    等价于把地图按 ATE 对齐口径摆)给出 ~66 m —— 差两个数量级,不看图也分得清。
    """

    def test_map_lands_on_synthetic_truth(self):
        frames, E, world = _synthetic_ego_sequence(10)
        slam = LiveSlam(voxel=0.5)
        for k, f in enumerate(frames):
            slam.push(f, ego_pose=E[k])
        for know in (0, 4, 9):
            got = slam.map_in_ego_frame(E[know])
            truth = _apply(M_FLIP @ np.linalg.inv(E[know]), world)  # 真值 ego 系(KITTI 手性)
            d = np.linalg.norm(got[:, None, :] - truth[None, :, :], axis=2).min(1)
            assert d.max() < 0.5, f"k={know} 地图点最近邻最大 {d.max():.3f} m(体素 0.5 应 ~0.3)"
            assert d.mean() < 0.15

    def test_map_is_kitti_handed(self):
        """BEV 面板窗口 `BEV_Y` 标注"y 左向" ⇒ 出口必须是 **KITTI 手性**。

        **手性推导别凭直觉**:CARLA 是 y **右**,KITTI 是 y **左**。ego 朝 +x 时,
        世界系 CARLA `y = −4` 的那道墙在 ego 的**左**边 ⇒ 出口(KITTI)y 必须为 **正**。
        若漏掉最后的 `M3` 翻转,符号会整体反过来(BEV 上左右镜像;对称场景看不出来,
        非对称场景才暴露 —— 所以这里用单侧墙)。
        """
        M3, Lm = M_FLIP, lever_matrix(LIDAR_LEVER)
        E0 = _ego_pose_world((0.0, 0.0, 0.0), 0.0)  # 朝 +x
        for wy_carla, sign in ((-4.0, +1.0), (+4.0, -1.0)):
            wall = np.stack([np.full(600, 8.0), np.full(600, wy_carla), np.arange(600) % 4 * 1.0], 1)
            frame = _apply(M3, _apply(np.linalg.inv(E0 @ Lm), wall))
            slam = LiveSlam(voxel=0.5)
            slam.push(frame, ego_pose=E0)
            pts = slam.map_in_ego_frame(E0)
            med = float(np.median(pts[:, 1]))
            assert med * sign > 3.0, (
                f"CARLA y={wy_carla:+.0f} 的墙在 ego 系应为 KITTI y {sign * 4:+.0f},实测 {med:+.2f}"
            )

    def test_traj_matches_ego_positions_in_current_frame(self):
        """轨迹折线 = 链式位姿还原的 ego 世界位置,再落到当前 ego 系(KITTI 手性)。

        **不是** `T @ P_k[:3,3]`:那会漏掉 `P_k` 的旋转块,实测末点差 1.67 m 且漏出
        z = 1.629 ≈ 杆臂高度。
        """
        frames, E, _ = _synthetic_ego_sequence(10)
        slam = LiveSlam(voxel=0.5)
        for k, f in enumerate(frames):
            slam.push(f, ego_pose=E[k])
        for know in (0, 4, 9):
            got = slam.traj_in_ego_frame(E[know])
            want = _apply(M_FLIP @ np.linalg.inv(E[know]), np.array([Ek[:3, 3] for Ek in E]))
            assert np.abs(got - want).max() < 0.05, f"k={know} 轨迹最大差 {np.abs(got - want).max():.4f} m"
            assert abs(got[:, 2]).max() < 0.05, "轨迹应贴地(z≈0),漏旋转块会漏出杆臂高度"

    def test_missing_anchor_raises_instead_of_silently_wrong(self):
        """`push` 不给 `ego_pose` ⇒ 取地图/轨迹必须**报错**,不静默给错坐标系。"""
        frames, _, _ = _synthetic_ego_sequence(2)
        slam = LiveSlam(voxel=0.5)
        for f in frames:
            slam.push(f)  # 无 ego_pose
        assert slam.snapshot()["has_anchor"] is False
        with pytest.raises(RuntimeError, match="ego 世界位姿"):
            slam.map_in_ego_frame(np.eye(4))
        with pytest.raises(RuntimeError, match="ego 世界位姿"):
            slam.traj_in_ego_frame(np.eye(4))


class TestMapAccumulation:
    def test_map_grows_and_lands_in_lidar0_frame(self):
        """累积地图点数单调增;首帧位姿 = 恒等 ⇒ 地图点 = 首帧点云(同系)。"""
        frames, _ = _synthetic_sequence(3)
        slam = LiveSlam(voxel=0.5)
        counts = []
        for f in frames:
            slam.push(f)
            counts.append(slam.snapshot()["n_map_points"])
        assert counts == sorted(counts) and counts[-1] > counts[0]
        # 首帧点云体素下采样后的点数应等于地图初始规模
        first = voxel_downsample(frames[0], 0.5)
        assert counts[0] == len(first)

    def test_map_in_ego_frame_is_finite_and_complete(self):
        """`ego_pose_world` = 帧 0 自身位姿 ⇒ 地图点全部有限、点数与快照一致。

        几何正确性由 `TestCurrentEgoFrame` 对合成真值验;这里只钉"不产生 NaN / 不丢点"。
        """
        frames, E, _ = _synthetic_ego_sequence(3)
        slam = LiveSlam(voxel=0.5)
        for k, f in enumerate(frames):
            slam.push(f, ego_pose=E[k])
        pts = slam.map_in_ego_frame(E[0])
        assert pts.shape[1] == 3 and len(pts) == slam.snapshot()["n_map_points"]
        assert np.isfinite(pts).all()

    def test_over_limit_redownsamples_instead_of_dropping(self):
        """超 `map_max_points` 时整体重下采样(**不是丢点**):点数回落但非零,体素翻倍。"""
        frames, _ = _synthetic_sequence(4)
        slam = LiveSlam(voxel=0.5, map_max_points=500)
        for f in frames:
            slam.push(f)
        snap = slam.snapshot()
        assert 0 < snap["n_map_points"] <= 500
        assert slam.map_voxel > 0.5, "超上限后体素边长应自适应放大"


class TestSnapshotIsolation:
    def test_snapshot_poses_are_copies(self):
        frames, _ = _synthetic_sequence(3)
        slam = LiveSlam(voxel=0.5)
        for f in frames:
            slam.push(f)
        snap = slam.snapshot()
        snap["poses"][0][:] = 999.0  # 篡改快照
        assert np.abs(slam.snapshot()["poses"][0] - np.eye(4)).max() < 1e-12, "快照未拷贝,污染了会话"

    def test_snapshot_does_not_block_on_a_full_icp(self):
        """`snapshot` 在 ICP 慢段**不该**被挡住:ICP 在锁外算,锁只护提交。

        判据:ICP 进行中(用一个慢的假 icp 卡住 push)另开线程取快照,必须在远小于
        一次 ICP 的时间内返回。
        """
        import autodrivedata.live_slam as ls

        frames, _ = _synthetic_sequence(2)
        slam = LiveSlam(voxel=0.5)
        slam.push(frames[0])

        release = threading.Event()
        real = ls.icp_odometry

        def slow_icp(*a, **kw):
            release.wait(timeout=5.0)
            return real(*a, **kw)

        ls.icp_odometry = slow_icp
        try:
            t = threading.Thread(target=lambda: slam.push(frames[1]), daemon=True)
            t.start()
            got: list[dict] = []
            tk = threading.Thread(target=lambda: got.append(slam.snapshot()), daemon=True)
            tk.start()
            tk.join(timeout=1.0)
            assert not tk.is_alive(), "snapshot 被 ICP 阻塞了(锁范围过大)"
            assert got[0]["n_frames"] == 1, "ICP 未提交前不该看到新帧"
            release.set()
            t.join(timeout=10.0)
            assert slam.snapshot()["n_frames"] == 2
        finally:
            ls.icp_odometry = real


class TestEdges:
    def test_empty_cloud_still_advances_with_fallback(self):
        """空云:ICP 返回 failed,链用 init 兜底(**不产生 NaN**),帧计数照走。"""
        slam = LiveSlam(voxel=0.5)
        slam.push(np.zeros((0, 4)))
        slam.push(np.zeros((0, 4)))
        snap = slam.snapshot()
        assert snap["n_frames"] == 2
        assert snap["n_nan"] == 0
        assert snap["n_failed"] >= 1
        assert np.isfinite(snap["poses"][1]).all()

    def test_single_frame_has_no_inter_frame_stats(self):
        """单帧:无帧间对 ⇒ rmse/overlap 统计量按离线口径取 0(**不是 1.0**)。

        `bin/slam_odometry.py` 的 `mean_*` 分母是 `max(n-1, 1)` 且只在 k>0 累加,
        单帧时分子为 0 ⇒ 0.0。别把"没有数据"伪装成"完美重叠"。
        """
        frames, _ = _synthetic_sequence(1)
        slam = LiveSlam(voxel=0.5)
        slam.push(frames[0])
        snap = slam.snapshot()
        assert snap["n_frames"] == 1
        assert snap["mean_rmse"] == 0.0
        assert snap["mean_overlap"] == 0.0

    def test_push_returns_frame_index_and_metrics(self):
        frames, _ = _synthetic_sequence(2)
        slam = LiveSlam(voxel=0.5)
        r0 = slam.push(frames[0])
        r1 = slam.push(frames[1])
        assert (r0["frame"], r1["frame"]) == (0, 1)
        assert r0["rmse_final"] == 0.0
        assert 0.0 <= r1["overlap"] <= 1.0
        assert r1["iters"] >= 1


class _FakeSlam:
    """极简替身:每次 push 睡 `delay` 秒,只记帧号 —— 用来测**队列/滞后算术**而非 ICP。

    用真 `LiveSlam` 测队列会把单测拖到分钟级(单帧 ICP 0.15–0.35 s),而滞后有界是
    **算术**性质,与 ICP 快慢无关:只要"消费比生产慢"这个条件成立,丢旧就该让滞后收敛。
    `dead_reckon` 照 `LiveSlam.push` 的签名收下并回传,便于断言止损路径被走到。
    """

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.pushed: list[int] = []
        self.dead: list[int] = []

    def push(self, points, t_stamp=0.0, ego_pose=None, dead_reckon: bool = False) -> dict:
        time.sleep(self.delay)
        idx = int(np.asarray(points).reshape(-1)[0])
        self.pushed.append(idx)
        if dead_reckon:
            self.dead.append(idx)
        return {"failed": False, "dead_reckon": bool(dead_reckon)}


class TestSlamWorker:
    """`SlamWorker`:`offer`/`stats`/`stop`、**滞后有界**的算术、以及**止损**。

    默认 `sync=True`(实测 worker 线程被 GIL 压到 eff 0.24,见 Plan2.md §P-L.2);
    队列/丢旧那几条用 `sync=False` 显式起线程测。
    """

    def test_drops_oldest_and_lag_stays_bounded(self):
        """**核心判据**(计划 B4 第 2 项):消费远慢于生产时,滞后**有界**而非单调增长。

        场景:生产者瞬间塞 200 帧,消费者每帧 50 ms(总需 10 s)。队列长 3 ⇒ 稳态只
        留最新 3 帧 ⇒ `lag_frames` 必须收敛到个位数,而不是爬到 200。
        丢帧数(`n_dropped`)则如实计数 —— 滞后有界**靠丢帧换来**,两者都要报。
        """
        fake = _FakeSlam(delay=0.05)
        w = SlamWorker(fake, maxsize=3, sync=False)  # type: ignore[arg-type]
        w.start()
        try:
            lags = []
            for idx in range(200):
                w.offer(np.array([[float(idx)]]), time.time(), np.eye(4), idx)
                if idx % 10 == 0:
                    lags.append(w.stats(idx)["lag_frames"])
            # 消费 3 帧(≈0.15 s)后滞后应已收敛,末值远小于 200
            assert lags[-1] < 20, f"滞后未收敛:{lags}"
            st = w.stats(199)
            assert st["n_dropped"] > 0, "队列长 3 + 200 帧 ⇒ 必然丢帧,没丢说明丢旧没生效"
            assert st["n_offered"] == 200
        finally:
            w.stop()

    def test_sync_mode_is_inline_no_queue_no_drop(self):
        """同步模式(默认):`offer` 就地跑完,无队列、无丢帧、滞后恒 0。"""
        fake = _FakeSlam(delay=0.0)
        w = SlamWorker(fake, maxsize=1, sync=True)  # type: ignore[arg-type]
        w.start()  # 同步模式不起线程
        assert w._thread is None, "同步模式不该起线程"
        for idx in range(5):
            w.offer(np.array([[float(idx)]]), time.time(), np.eye(4), idx)
        st = w.stats(4)
        assert fake.pushed == [0, 1, 2, 3, 4], "同步模式必须逐帧处理,不许丢帧"
        assert st["n_processed"] == 5
        assert st["n_dropped"] == 0
        assert st["lag_frames"] == 0, "同步模式下处理完当前帧,滞后必须是 0"
        assert w.stop(timeout=1.0) is True

    def test_max_gap_triggers_dead_reckon(self):
        """**止损判据**:帧号差 > `max_gap` 时不做 ICP,直接恒速外推。

        为什么需要:ICP 成本随 `prev_down` 与当前帧的**帧间隙**爆炸(体素 1.0 下
        0.8 m 0.2 s → 8 m 2.8 s → 32 m 39 s),而丢帧会让间隙无界增长 ⇒
        "丢帧→更慢→更多丢帧" 正反馈。止损把间隙钉回 1 帧。
        """
        fake = _FakeSlam(delay=0.0)
        w = SlamWorker(fake, max_gap=3, sync=True)  # type: ignore[arg-type]
        w.start()
        for idx in (0, 1, 2):  # gap 1,1,1 ⇒ 正常 ICP
            w.offer(np.array([[float(idx)]]), time.time(), np.eye(4), idx)
        assert fake.dead == [], f"gap ≤ max_gap 不该止损,实得 {fake.dead}"
        w.offer(np.array([[9.0]]), time.time(), np.eye(4), 9)  # gap 7 > 3 ⇒ 止损
        assert fake.dead == [9], f"gap 7 应止损,实得 {fake.dead}"
        assert w.stats(9)["n_dead"] == 1
        # 止损后 gap 回到 1,下一帧继续走 ICP
        w.offer(np.array([[10.0]]), time.time(), np.eye(4), 10)
        assert fake.dead == [9], f"止损后应恢复 ICP,实得 {fake.dead}"
        assert w.stop(timeout=1.0) is True

    def test_offer_copies_inputs(self):
        """跨线程不共享可变对象:调用方下一 tick 复用同一数组,不得污染已入队的帧。"""
        fake = _FakeSlam(delay=0.0)
        w = SlamWorker(fake, maxsize=4, sync=False)  # type: ignore[arg-type]
        pts = np.array([[1.0], [2.0]])
        pose = np.eye(4)
        w.offer(pts, 0.0, pose, 0)
        pts[:] = 999.0
        pose[:] = 777.0
        w.start()
        try:
            for _ in range(50):
                if fake.pushed:
                    break
                time.sleep(0.02)
        finally:
            w.stop()
        assert fake.pushed == [1], "入队的点云/位姿被调用方的后续写入污染了"

    def test_lag_uses_frame_index_not_offer_count(self):
        """滞后口径是**帧号差**,不是 `n_offered − n_processed`。

        丢旧之后后者随丢帧数无界增长(看着像故障,其实队列一直是满的);帧号差才反映
        "SLAM 落后当前时刻多远"。构造:只喂 2 帧、队列长 1 ⇒ 丢 1 帧,但处理完后
        `lag_frames` 必须归 0。
        """
        fake = _FakeSlam(delay=0.0)
        w = SlamWorker(fake, maxsize=1, sync=False)  # type: ignore[arg-type]
        w.offer(np.array([[0.0]]), 0.0, np.eye(4), 0)
        w.offer(np.array([[1.0]]), 0.0, np.eye(4), 1)  # 队列满 ⇒ 丢帧 0
        w.start()
        try:
            for _ in range(100):
                if w.stats(1)["n_processed"] >= 1:
                    break
                time.sleep(0.02)
        finally:
            w.stop()
        st = w.stats(1)
        assert st["n_dropped"] == 1
        assert st["lag_frames"] == 0, f"处理到最新帧后滞后应为 0,实得 {st['lag_frames']}"

    def test_stop_joins_the_thread(self):
        """`stop` 必须**真的** join(退出纪律:先停 worker 再销毁 world)。"""
        fake = _FakeSlam(delay=0.0)
        w = SlamWorker(fake, maxsize=2, sync=False)  # type: ignore[arg-type]
        w.start()
        assert w.stop(timeout=5.0) is True
        assert w._thread is not None and not w._thread.is_alive()

    def test_push_exception_does_not_kill_the_worker(self):
        """worker 里 push 抛异常不能静默死掉(死了 HUD 的滞后会一直涨,却看不出原因)。"""

        class _Boom:
            def __init__(self) -> None:
                self.n = 0

            def push(self, points, t_stamp=0.0, ego_pose=None, dead_reckon: bool = False) -> dict:
                self.n += 1
                if self.n == 1:
                    raise RuntimeError("boom")
                return {"failed": False, "dead_reckon": False}

        boom = _Boom()
        w = SlamWorker(boom, maxsize=2, sync=False)  # type: ignore[arg-type]
        w.start()
        try:
            w.offer(np.array([[0.0]]), 0.0, np.eye(4), 0)
            w.offer(np.array([[1.0]]), 0.0, np.eye(4), 1)
            for _ in range(100):
                if w.stats(1)["n_processed"] >= 1:
                    break
                time.sleep(0.02)
            assert w.stats(1)["n_processed"] == 1, "异常后 worker 没继续处理后续帧"
        finally:
            w.stop()

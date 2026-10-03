"""路线纯值单测:路网找环 / 闭环判据 / 前视点 / 纯追踪控制律 / 段起点选点。

风格沿用仓库:手算锚点 + 边界。**全部在合成图上做** —— `find_cycle` 只吃 `successors`
回调(见 route.py 头注),所以不必起 CARLA。

**三条主判据**:
1. **环是真的**:合成方框图 → 找到 4 节点最短环,且首末**几何上接得上**;
2. **★ 进度单调**:前视点索引不许回退 —— 纯追踪最经典的一类失败是车在起点附近来回蹭,
   而这种错在采集时**看着像在开**(有速度、有转向),只在轨迹形状上暴露;
3. **★ 打舵方向**:角度误差为正 ⇒ 目标在右 ⇒ `steer` 取正。符号反了车会**朝反方向冲出去**,
   而两圈跑完才知道 —— 这条必须手算钉死。

第四条在 `TestSpawnPointSelection`:**选点能不能复现已知集合**(`probe_spawn_points --expect`
的自证就靠它),以及 `min_pairwise` 取的是**最近**那一对。
"""

from __future__ import annotations

import math
from typing import cast

import pytest

from autodrivedata.sim.route import (
    Cycle,
    NodeKey,
    farthest_from_centroid,
    find_cycle,
    greedy_maxmin,
    greedy_maxmin_order,
    lap_budget,
    lateral_error,
    lookahead_index,
    make_successors,
    min_pairwise,
    normalize_angle,
    pure_pursuit,
    route_closure,
    route_length,
    speed_ceiling,
    spread_curve,
    track_index,
)

# 合成方框街区:A(0,0) → B(10,0) → C(10,10) → D(0,10) → A,周长 40 m
A, B, C, D = (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)
SQUARE = {A: [B], B: [C], C: [D], D: [A]}


def _succ(graph):
    return lambda n: graph.get(n, [])


def _cycle(start, graph, **kw) -> Cycle:
    """找到环并断言存在(把 `Cycle | None` 收窄,同时给失败一个有意义的断言消息)。"""
    got = find_cycle(start, _succ(graph), **kw)
    assert got is not None, f"本该找到环:{start!r}"
    return got


class TestFindCycle:
    def test_square_yields_the_four_node_cycle(self):
        """★ 主判据:方框图 → 环 = [A,B,C,D](长度 4),且起点在环上 ⇒ 无引路段。"""
        cyc = _cycle(A, SQUARE)
        assert cyc == Cycle(prefix=(), loop=(A, B, C, D))
        assert cyc.route_len == 4 and cyc.loop_start == 0

    def test_cycle_order_is_followed_not_sorted(self):
        """环的**顺序**必须是走后继得到的,不是按坐标排序的巧合(否则会把环读成 8 字)。"""
        assert _cycle(B, SQUARE).loop == (B, C, D, A)

    def test_start_off_cycle_gets_a_prefix(self):
        """起点不在环上 ⇒ 返回「引路 + 环」,引路**不含环入口**(含了就与 loop[0] 重复)。

        S 只通向 A:A 在环上 ⇒ prefix=(S,)、loop=(A,B,C,D) ⇒ 拼起来 = S,A,B,C,D。
        """
        graph = {**SQUARE, (-10.0, -10.0): [A]}
        cyc = _cycle((-10.0, -10.0), graph)
        assert cyc == Cycle(prefix=((-10.0, -10.0),), loop=(A, B, C, D))
        assert cyc.loop_start == 1

    def test_no_cycle_returns_none(self):
        """开链无环 ⇒ None(**不许**造一个"回到起点"的假环)。"""
        assert find_cycle(0, _succ({0: [1], 1: [2], 2: [3]})) is None

    def test_two_node_back_and_forth_is_not_a_cycle(self):
        """★ `A→B→A` 是**掉头**不是环(真实路网里 `next()` 也走不出来)。

        放它过去的话「跑两圈」会退化成原地来回蹭 —— 而那恰好是回环检测的**反面**:
        位置重合但朝向相反、且只隔 2 帧,`SC_MIN_GAP_NODES=25` 也会把它挡掉,白采一趟。
        """
        assert find_cycle("A", _succ({"A": ["B"], "B": ["A"]})) is None

    def test_self_loop_is_rejected(self):
        """自环(长度 1)必须被拒 —— 它会让路线变成 0 米。"""
        assert find_cycle("A", _succ({"A": ["A"]})) is None

    def test_shortest_cycle_wins_over_a_longer_one(self):
        """同时存在 4 环与 6 环时取**短的**(短的 = 小街区 = 圈内帧数够、采集快)。"""
        graph = {**SQUARE, B: [C, "x1"], "x1": ["x2"], "x2": [A]}
        cyc = _cycle(A, graph)
        assert len(cyc.loop) == 4, f"取了长环:{cyc.loop}"

    def test_max_nodes_bounds_an_infinite_chain(self):
        """无环**无限**链上 `max_nodes` 必须兜住(真实路网上每次 `successors` 是一次 RPC)。"""
        succ = lambda n: [n + 1]  # noqa: E731
        assert find_cycle(0, succ, max_nodes=20) is None

    def test_min_len_below_two_raises(self):
        """`min_len < 2` 直接报错:长度为 1 的"环"没有意义,静默接受会造出 0 米路线。"""
        with pytest.raises(ValueError, match="min_len"):
            find_cycle(A, _succ(SQUARE), min_len=1)


class FakeWaypoint:
    """合成 `Waypoint`:方框 `RING` 段路、每段 `LANE_LEN` 米、单向单车道。

    只实现 `WaypointLike` 的那个最小面(road_id / lane_id / s / next),
    于是**适配层本身**能在没有 CARLA 的地方被验证 —— 这正是把 `make_successors`
    放进纯值层的理由。
    """

    RING = 4  # 4 段 ⇒ 周长 120 m 的闭合环;1 = 一条直路(永不重访)
    LANE_LEN = 30.0
    _CORNER = {1: (0.0, 0.0), 2: (30.0, 0.0), 3: (30.0, 30.0), 4: (0.0, 30.0)}
    _DIR = {1: (1.0, 0.0), 2: (0.0, 1.0), 3: (-1.0, 0.0), 4: (0.0, -1.0)}

    def __init__(self, road_id: int, lane_id: int, s: float):
        self._road_id, self._lane_id, self._s = road_id, lane_id, s

    @property
    def road_id(self) -> int:
        return self._road_id

    @property
    def lane_id(self) -> int:
        return self._lane_id

    @property
    def s(self) -> float:
        return self._s

    @property
    def pos(self) -> tuple[float, float]:
        """方框几何(只为手算单圈长度:每段正好 30 m)。"""
        x0, y0 = self._CORNER[self._road_id]
        dx, dy = self._DIR[self._road_id]
        return (x0 + dx * self._s, y0 + dy * self._s)

    def next(self, distance: float):
        s, road = self._s + distance, self._road_id
        if self.RING == 1:  # 直路:s **一直增长**。若在这里也按段长取模,路虽不换但 s 回绕
            return [type(self)(road, self._lane_id, s)]  # ⇒ 照样成一个"环"(实测踩过)
        while s >= self.LANE_LEN:  # 跨段(LANE_LEN=30 / step=3 ⇒ 最多跨一段)
            s -= self.LANE_LEN
            road = road % self.RING + 1
        return [type(self)(road, self._lane_id, s)]


def _cycle_of(start, successors, **kw) -> Cycle:
    """回调版 `_cycle`(合成 Waypoint 不经图字典,直接吃 `successors`)。"""
    got = find_cycle(start, successors, **kw)
    assert got is not None, f"本该找到环:{start!r}"
    return got


def _ring_points(cache, keys) -> list[tuple[float, float]]:
    """键 → 合成 Waypoint 的坐标。

    两次收窄是**测试侧**的事:`find_cycle` 对节点类型不透明(键是 `Hashable`),
    适配层也刻意只承诺 duck-typed `WaypointLike`(零 carla)。
    """
    return [cast(FakeWaypoint, cache[cast(NodeKey, k)]).pos for k in keys]


class TestMakeSuccessors:
    """路网适配层 —— 「键怎么取」「一圈能不能回到起点」**只能靠这条钉**,别只在服务器上试。"""

    def test_ring_road_closes_exactly_and_lap_length_is_right(self):
        """★ 主判据:4 段 × 30 m 方框 ⇒ 40 节点环,**单圈几何 120.0 m**。

        120 = 39×3(相邻节点)+ 3(末点回起点)。这一条同时锁住两件事:
        ① 键里带 `s`(否则根本找不到环);② `route_closure` 把**收尾那一段**算进长度。
        """
        succ, register, cache = make_successors(3.0)
        cyc = _cycle_of(register(FakeWaypoint(1, -1, 0.0)), succ)
        assert cyc.prefix == ()  # 起点就在环上
        assert cyc.route_len == 40
        closure = route_closure(_ring_points(cache, cyc.loop))
        assert closure["length_m"] == pytest.approx(120.0)
        assert closure["end_gap_m"] == pytest.approx(3.0)  # 恰好一个采样步长

    def test_closure_tolerance_has_to_be_step_sized(self):
        """★ 闭合容差该按**采样步长**给,不该按 1 m 给:节点键是分桶量化的,
        收尾那段天然就是一个 `step` 的残差(上面的 3.0 m 就是它)。"""
        succ, register, cache = make_successors(3.0)
        cyc = _cycle_of(register(FakeWaypoint(1, -1, 0.0)), succ)
        pts = _ring_points(cache, cyc.loop)
        assert route_closure(pts, tol=3.0)["closed"]
        assert not route_closure(pts, tol=1.0)["closed"]

    def test_node_key_distinguishes_position_along_the_lane(self):
        """★ `s` 必须进键:同一段路上 s=0 与 s=3 是**两个**节点。

        只留 `(road_id, lane_id)` 的话长路只有一个节点、后继键等于自己 ⇒ `A→A` 自环被
        `min_len` 挡掉 ⇒ **永远找不到环**(而且是静默地找不到)。
        """
        _, register, _ = make_successors(3.0)
        assert register(FakeWaypoint(1, -1, 0.0)) != register(FakeWaypoint(1, -1, 3.0))
        assert register(FakeWaypoint(1, -1, 0.0)) == register(FakeWaypoint(1, -1, 1.4))  # 半桶内同格

    def test_successors_are_resolvable_from_the_cache(self):
        """回调吐出的键必须**当次就进缓存** —— `find_cycle` 下一轮拿它回调回来,
        迟一步填就是 KeyError(适配层最容易漏的一步)。"""
        succ, register, cache = make_successors(3.0)
        keys = succ(register(FakeWaypoint(1, -1, 0.0)))
        assert keys
        assert all(k in cache for k in keys)

    def test_straight_road_yields_no_cycle(self):
        """一条直路(不回绕)⇒ 键一直向前 ⇒ 无环。"""
        succ, register, _ = make_successors(3.0)
        assert find_cycle(register(FakeStraight(1, -1, 0.0)), succ, max_nodes=50) is None

    def test_nonpositive_step_raises(self):
        with pytest.raises(ValueError, match="采样步长"):
            make_successors(0.0)


class FakeStraight(FakeWaypoint):
    """一条不回绕的直路(`next` 只往前推,键永不重逢)。"""

    RING = 1


class TestRouteGeometry:
    def test_closure_length_agrees_whether_or_not_start_is_repeated(self):
        """★ 环长口径钉:末点回到首点那段**必须算进长度**。

        `[A,B,C,D]`(隐含闭合)与 `[A,B,C,D,A]`(显式闭合)都必须给出 40 m ——
        否则闭环路线的长度会被系统性低估整整一条边(方块城市里 = 25%)。
        """
        implicit = route_closure([A, B, C, D], tol=1.0)
        explicit = route_closure([A, B, C, D, A], tol=1.0)
        assert implicit["length_m"] == pytest.approx(40.0)
        assert explicit["length_m"] == pytest.approx(40.0)
        assert explicit["end_gap_m"] == pytest.approx(0.0)
        assert implicit["end_gap_m"] == pytest.approx(10.0)  # |D−A|
        assert explicit["closed"] and not implicit["closed"]

    def test_closure_tolerance_is_the_judgement(self):
        """10 m 的缺口:容差 10 算闭合、容差 1 不算 —— 容差是判据不是装饰。"""
        assert route_closure([A, B, C, D], tol=10.0)["closed"]
        assert not route_closure([A, B, C, D], tol=1.0)["closed"]

    def test_empty_route_is_not_closed(self):
        got = route_closure([])
        assert got == {"n": 0, "length_m": 0.0, "end_gap_m": 0.0, "closed": False}

    def test_route_length_ignores_z(self):
        """长度只算 xy —— z 是路面高程起伏,不该算进行驶里程(否则 791 m 高程图会爆炸)。"""
        pts = [(0.0, 0.0, 0.0), (3.0, 4.0, 100.0)]
        assert route_length(pts) == pytest.approx(5.0)


class TestLapBudget:
    """单圈帧预算 —— **闭环采集的第一道闸**:不通过就不该去采(白跑 25 分钟)。"""

    def test_speed_ceiling_is_inverse_in_loop_length(self):
        """★ 主判据:一圈至少 250 帧 ⇒ `速度 ≤ 环长/(250·tick)`。

        200 m 的环 @10 Hz 上限正好 8.0 m/s;120 m 的环只有 4.8 —— 开 8 m/s 的话
        第二圈到访时第一圈才过去 150 帧,连候选都进不去。**这条必须采集前报出来。**
        """
        b200 = lap_budget(200.0, laps=2, delta=0.1, speed=8.0, min_frames_per_lap=250)
        assert b200["speed_max"] == pytest.approx(8.0)
        assert b200["ok"] is True
        assert b200["frames_per_lap"] == pytest.approx(250.0)

        b120 = lap_budget(120.0, laps=2, delta=0.1, speed=8.0, min_frames_per_lap=250)
        assert b120["speed_max"] == pytest.approx(4.8)
        assert b120["ok"] is False

    def test_speed_ceiling_is_usable_before_the_speed_is_chosen(self):
        """★ 上限必须**独立可算** —— 采集器 `--speed 0` 时要先拿上限再定速度,
        藏进 `lap_budget` 的返回值里就只能靠"猜一个速度"去换,那正是循环依赖。"""
        assert speed_ceiling(200.0, delta=0.1, min_frames_per_lap=250) == pytest.approx(8.0)
        assert speed_ceiling(120.0, delta=0.1, min_frames_per_lap=250) == pytest.approx(4.8)
        with pytest.raises(ValueError):
            speed_ceiling(0.0, delta=0.1, min_frames_per_lap=250)

    def test_frames_recommended_covers_accel_and_tail(self):
        """推荐帧数 = `laps·每圈帧数·余量 + 50`:不带余量会刚好卡在 250 帧上,一轮抖动就掉出去。"""
        b = lap_budget(200.0, laps=2, delta=0.1, speed=8.0, min_frames_per_lap=250)
        assert b["frames_recommended"] == int(2 * 250.0 * 1.25) + 50
        assert b["frames_recommended"] > 2 * b["min_frames_per_lap"]

    def test_degenerate_inputs_raise(self):
        """0 长环 / 0 圈 / 0 速度必须报错,不许返回一个"看着能用"的预算。"""
        for kw in (
            {"loop_len": 0.0, "laps": 2, "delta": 0.1, "speed": 8.0},
            {"loop_len": 200.0, "laps": 0, "delta": 0.1, "speed": 8.0},
            {"loop_len": 200.0, "laps": 2, "delta": 0.0, "speed": 8.0},
            {"loop_len": 200.0, "laps": 2, "delta": 0.1, "speed": 0.0},
        ):
            with pytest.raises(ValueError):
                lap_budget(min_frames_per_lap=250, **kw)


class TestNormalizeAngle:
    def test_wraps_to_the_same_half_open_interval_as_atan2(self):
        """★ 区间口径钉 = `[−π, π)`(与 `math.atan2` 同)。

        π 落成 **−π** 是半开区间的定义,不是怪异行为;写成 `(−π, π]` 就与 `atan2`
        错开一个端点,误差角在正后方附近与目标方位角对不上。
        """
        assert normalize_angle(0.0) == pytest.approx(0.0)
        assert normalize_angle(math.pi) == pytest.approx(-math.pi)
        assert normalize_angle(-math.pi) == pytest.approx(-math.pi)
        assert normalize_angle(3 * math.pi) == pytest.approx(-math.pi)
        # 区间内的角原样保留;多绕整圈回来也对得上
        for a in (0.3, -0.3, 2.5, -2.5):
            assert normalize_angle(a) == pytest.approx(a)
        assert normalize_angle(2.0 * math.pi + 0.3) == pytest.approx(0.3)

    def test_keeps_small_angles(self):
        """★ 小角必须原样保留:若这里被"归一化"成 0,打舵方向就没了。"""
        assert normalize_angle(0.1) == pytest.approx(0.1)
        assert normalize_angle(-0.1) == pytest.approx(-0.1)


class TestLookaheadIndex:
    ROUTE = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (15.0, 0.0)]

    def test_picks_first_point_beyond_the_lookahead(self):
        """手算:(5,0) 距 5 < 6,(10,0) 距 10 ≥ 6 ⇒ 取索引 2。"""
        assert lookahead_index(self.ROUTE, (0.0, 0.0), 0, 6.0) == (2, False)
        assert lookahead_index(self.ROUTE, (0.0, 0.0), 0, 3.0) == (1, False)

    def test_progress_never_goes_backwards(self):
        """★ 进度单调:车已到索引 2,即使索引 1 也够远,也不许退回去。"""
        assert lookahead_index(self.ROUTE, (0.0, 0.0), 2, 1.0) == (2, False)

    def test_wrap_returns_to_the_loop_entry_and_flags_it(self):
        """★ 回绕落到 `wrap_at`(环入口)而不是 0,并**置位 `wrapped`** —— 圈数靠它数。"""
        assert lookahead_index(self.ROUTE, (15.0, 0.0), 3, 1.0, wrap_at=0) == (0, True)

    def test_wrap_at_nonzero_skips_the_prefix(self):
        """`wrap_at=1` ⇒ 第二圈从索引 1 起:引路段(索引 0)**只跑一次**,不回头重跑。"""
        assert lookahead_index(self.ROUTE, (15.0, 0.0), 3, 1.0, wrap_at=1) == (1, True)

    def test_without_wrap_it_clamps_to_the_last_point(self):
        """非闭环路线(采集器不该走到这,但要有个明确行为):钳在末点,不回绕。"""
        assert lookahead_index(self.ROUTE, (15.0, 0.0), 3, 1.0) == (3, False)

    def test_empty_route_raises(self):
        with pytest.raises(ValueError, match="路线为空"):
            lookahead_index([], (0.0, 0.0), 0, 1.0)


def _square_ring(side: float = 12.0, step: float = 3.0) -> list[tuple[float, float]]:
    """闭合方框环(节点间距 `step`,周长 4×side):末点 (0, step) 与首点 (0, 0) 相邻,
    即**环缝**。测试进度索引跨环缝必须用它 —— 直线列表 `%n` 后也会"相邻",但几何是假的。"""
    n = int(side / step)
    ring = [(i * step, 0.0) for i in range(n)]  # 底边 →
    ring += [(side, i * step) for i in range(n)]  # 右边 ↑
    ring += [(side - i * step, side) for i in range(n)]  # 顶边 ←
    ring += [(0.0, side - i * step) for i in range(n)]  # 左边 ↓(i=n 才是起点,故不含重复)
    return ring


class TestTrackIndex:
    """自证用的**进度索引**。与 `lookahead_index` 是两个量 —— 早期把自证挂在前视索引上,
    量到的是前视距离(~8 m),与中止阈值重合 ⇒ 正常跟线踩响中止(见 `TestLateralError`)。"""

    RING = _square_ring()  # 16 节点 / 周长 48 m / 间距 3 m

    def test_finds_the_nearest_point_in_the_window(self):
        """车在 (0, 2):末点 (0,3) 距 1 m 胜出(首点 (0,0) 距 2 m)。"""
        assert track_index(self.RING, (0.0, 2.0), 15) == 15

    def test_follows_the_wrap_across_the_ring_seam(self):
        """★ 回归(real bug):进度索引必须能**跨过环缝**回到环首。

        旧实现把窗口上界钳在 `n`,车跑完第一圈回到环首时索引永远停在 `n-1`,误差变成
        "车到末点的距离"、单调涨到 31 m ⇒ **第二圈刚起步就踩响中止**(实测帧 469,
        进度点 85/86)。车越到 (0, 0.5):首点距 0.5 m 胜过末点的 2.5 m ⇒ 索引回到 0。"""
        assert track_index(self.RING, (0.0, 0.5), 15) == 0

    def test_window_is_bounded_it_does_not_teleport_to_the_global_nearest(self):
        """★ 车跑到 100 m 外时不许"跳到全局最近点"救回来。

        全局最近点是右上角 (12, 12) = 索引 8;窗口(±4/+6)够不到它 ⇒ 返回的索引仍
        贴着环缝那一带。**跟丢必须让上层的偏航自证看见,不能被一次跳跃掩盖。**"""
        idx = track_index(self.RING, (100.0, 100.0), 0)
        assert (idx - 0) % len(self.RING) <= 6

    def test_empty_route_raises(self):
        with pytest.raises(ValueError, match="路线为空"):
            track_index([], (0.0, 0.0), 0)


class TestLateralError:
    ROUTE = [(i * 10.0, 0.0) for i in range(12)]

    def test_on_route_is_zero(self):
        assert lateral_error(self.ROUTE, (10.0, 0.0), 1) == pytest.approx(0.0)

    def test_measures_the_offset_not_the_lookahead_distance(self):
        """★ 回归(real bug):窗口必须是**双侧** ±span。

        旧实现从锚点起**单向向前**取 40 个点,于是"锚点自身有多远"被算进了误差。
        锚点是前视索引(前视 8 m ⇒ 脚下 10~20 m 的点)时,一次**完美**跟线也会读到
        8~9 m —— 而采集器的中止阈值恰好是 8 m / 连续 40 帧:正常采集踩响中止
        (实测探针:最近点偏差 mean 1.12 / max 2.52 m,打印的却是 8.96 m),
        真跟丢时反而不报警。下面 `span=0`(单侧口径)复现旧读数。"""
        assert lateral_error(self.ROUTE, (10.0, 0.0), 2, span=2) == pytest.approx(0.0)
        assert lateral_error(self.ROUTE, (10.0, 0.0), 2, span=0) == pytest.approx(10.0)

    def test_empty_route_is_infinite_not_zero(self):
        """空路线给 `inf` 而不是 0 —— 0 会让"跟线质量"看着完美(假阴性)。"""
        assert lateral_error([], (0.0, 0.0), 0) == float("inf")


class TestPurePursuit:
    def test_straight_ahead_gives_zero_steer(self):
        """车在 +x 直路上面朝 +x ⇒ 无航向误差 ⇒ steer = 0。"""
        c = pure_pursuit(
            (0.0, 0.0),
            0.0,
            [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)],
            0,
            speed=0.0,
            target_speed=8.0,
            lookahead=3.0,
        )
        assert c.steer == pytest.approx(0.0)
        assert c.throttle > 0.0 and c.brake == 0.0

    def test_steer_sign_points_toward_target(self):
        """★ 方向钉:目标在**右前方**(方位角 +45°)⇒ steer **为正**(CARLA 正 = 右转)。

        符号反了车会朝反方向冲出去,而且是"有速度、有转向"地看着正常 —— 只有轨迹暴露。
        这条手算值同时锁住了 route.py 头注里的角度口径。
        """
        c = pure_pursuit((0.0, 0.0), 0.0, [(5.0, 5.0)], 0, speed=0.0, target_speed=8.0, lookahead=1.0)
        assert c.steer > 0.0
        # 目标在左后方(方位角 −135°)⇒ 为负
        c2 = pure_pursuit((0.0, 0.0), 0.0, [(-5.0, -5.0)], 0, speed=0.0, target_speed=8.0, lookahead=1.0)
        assert c2.steer < 0.0

    def test_steer_saturates_at_one(self):
        """增益是有限的:`k_steer=1.5` ⇒ 误差 π/2 时 1.5·1.571 > 1 ⇒ 钳到 1(不许超界)。"""
        c = pure_pursuit((0.0, 0.0), 0.0, [(0.0, 1.0)], 0, speed=0.0, target_speed=8.0, lookahead=1.0)
        assert c.steer == pytest.approx(1.0)

    def test_overspeed_brakes_and_never_both_pedals(self):
        """★ 超速先刹车且**不叠油门** —— 同时给会"一边踩一边刹",车抖且速度测不准。"""
        c = pure_pursuit((0.0, 0.0), 0.0, [(5.0, 0.0)], 0, speed=12.0, target_speed=8.0, lookahead=1.0)
        assert c.throttle == 0.0 and c.brake > 0.0

    def test_pedals_are_never_both_nonzero(self):
        """扫一遍速度区间,`throttle·brake == 0` 恒成立。"""
        for v in (-5.0, 0.0, 7.9, 8.0, 8.1, 20.0):
            c = pure_pursuit((0.0, 0.0), 0.0, [(5.0, 0.0)], 0, speed=v, target_speed=8.0, lookahead=1.0)
            assert c.throttle == 0.0 or c.brake == 0.0, f"v={v} 双踏板"

    def test_wrapped_flag_counts_laps(self):
        """★ 圈数唯一依据:过环末点 ⇒ `wrapped=True` 且索引落到环入口。"""
        route = [(0.0, 0.0), (10.0, 0.0)]
        c = pure_pursuit((10.0, 0.0), 0.0, route, 1, speed=8.0, target_speed=8.0, lookahead=1.0, wrap_at=0)
        assert c.wrapped and c.idx == 0

    def test_idx_is_returned_for_chaining(self):
        """前视点索引回传 ⇒ 下一帧从它继续向前 —— 单调性靠这个跨帧传递,不在函数内保存状态。"""
        route = [(float(i) * 10.0, 0.0) for i in range(5)]
        c = pure_pursuit((0.0, 0.0), 0.0, route, 0, speed=0.0, target_speed=8.0, lookahead=25.0)
        assert c.idx == 3
        assert (
            pure_pursuit((0.0, 0.0), 0.0, route, c.idx, speed=0.0, target_speed=8.0, lookahead=1.0).idx == 3
        )


class TestSpawnPointSelection:
    """多段采集的**段起点选点**(贪心最大最小距离)。

    ★ 这一组真正的赌注是**「能不能复现 v2_epic 的 44/14/15/152/55」** ——
    `probe_spawn_points --expect` 就靠这个自证。复现不了,拿它给新图选的段就没有依据。
    最容易错的两条:**greedy 对种子敏感**(所以 `--expect` 必须遍历种子),
    与 **`min_pairwise` 是「最近的一对」不是「最远的一对」**(写成 max 会让判据永远通过)。
    """

    #: 10 m 小方块 + 一个远处的点 —— 手算得动,且"远点先被选中"一眼可验
    FIVE = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0), (100.0, 100.0)]

    def test_greedy_takes_the_isolated_point_before_the_cluster(self):
        """第一步必须选**离种子最远的**那个(141.4 m 的孤立点),而不是方块里 10 m 的邻居。"""
        assert greedy_maxmin(self.FIVE, 2, 0) == [0, 4]
        assert min_pairwise(self.FIVE, [0, 4]) == pytest.approx(141.4213562, rel=1e-6)

    def test_result_is_sorted_so_expect_comparison_is_order_free(self):
        """★ 返回**升序下标** —— `--expect` 是拿 `got == want` 比的。

        若按"选中顺序"返回,同样的集合会因为起点不同而给出不同序列,
        「复现了没有」这个判断就变成"得先猜对方用什么顺序"。
        """
        for seed in range(len(self.FIVE)):
            got = greedy_maxmin(self.FIVE, 3, seed)
            assert got == sorted(got), f"种子 {seed} 没排序:{got}"

    def test_seed_changes_the_result(self):
        """★ **贪心对种子敏感** —— 这不是缺陷,是 `--expect` 要遍历全部种子的理由。

        记一笔防"只试种子 0 然后说对不上":同样 5 个点、同样 k=3,
        种子 0/3/4 给 `{0,3,4}`,种子 1/2 给 `{1,2,4}` —— 两个不同的集合。
        """
        assert greedy_maxmin(self.FIVE, 3, 0) == [0, 3, 4]
        assert greedy_maxmin(self.FIVE, 3, 1) == [1, 2, 4]

    def test_a_known_set_is_reproduced_only_by_the_right_seed(self):
        """复现 `{0,3,4}` 要种子 0;种子 1 复现不了 ⇒ `--expect` 必须扫全部种子。"""
        want = [0, 3, 4]
        hits = [s for s in range(len(self.FIVE)) if greedy_maxmin(self.FIVE, 3, s) == want]
        assert hits == [0, 3, 4], "应恰好这几个种子能复现"
        assert 1 not in hits, "种子 1 给的是另一个集合 —— 只试它就误报『对不上』"

    def test_min_pairwise_is_the_closest_pair_not_the_farthest(self):
        """★ 取 **min** 不是 max。写成 max 会让「段间够不够远」的判据永远通过。"""
        pts = [(0.0, 0.0), (1.0, 0.0), (100.0, 100.0)]
        assert min_pairwise(pts, [0, 1, 2]) == pytest.approx(1.0)  # 最近的那对是 0–1

    def test_a_single_point_has_infinite_spread_not_zero(self):
        """★ 一个点**谈不上间距** ⇒ `inf`,不是 0。

        写成 0 会让 `--min-gap` 在"只选出一个点"时**误报不通过** ——
        而那其实说明的是 k 给错了,不是这张图太挤。
        """
        assert min_pairwise(self.FIVE, [0]) == float("inf")
        assert min_pairwise(self.FIVE, []) == float("inf")

    def test_spread_curve_never_rises_when_adding_a_point(self):
        """★ 曲线**单调不增**:后面的选集是前面的**超集**(同一个种子、同一段贪心序列),
        多一对点只可能给出更小的最近距离。涨了说明实现不是嵌套的。
        """
        curve = spread_curve(self.FIVE, 4, 0)
        assert [k for k, _ in curve] == [2, 3, 4]
        gaps = [g for _, g in curve]
        assert gaps == sorted(gaps, reverse=True), f"曲线不单调不增:{curve}"
        assert gaps[0] == pytest.approx(141.4213562, rel=1e-6)

    def test_curve_is_empty_below_two_points(self):
        """k<2 谈不上间距曲线 —— 返回空表,不是抛也不是造一个 (1, inf) 的假点。"""
        assert spread_curve(self.FIVE, 1, 0) == []

    def test_k_above_the_point_count_stops_instead_of_looping(self):
        """k 比点数大 ⇒ 给全部点,不许死循环(`best_i < 0` 的那个分支就是为此)。"""
        assert greedy_maxmin(self.FIVE, 10, 0) == [0, 1, 2, 3, 4]

    def test_bad_arguments_raise_instead_of_guessing(self):
        """越界种子 / k<1 必须报错 —— 静默取模或取 0 会让选出的段与记录的种子对不上。"""
        with pytest.raises(ValueError, match="越界"):
            greedy_maxmin(self.FIVE, 2, 5)
        with pytest.raises(ValueError, match="k 必须"):
            greedy_maxmin(self.FIVE, 0, 0)

    def test_pick_order_starts_with_the_seed(self):
        """★ 选择序的**首元素恒为种子**。

        `surround_v2_epic` 记的 `44/14/15/152/55` 是**选择序**(44 = 种子),
        按集合口径排序后是 `14/15/44/55/152`。同一个东西两种写法 ——
        所以两条都要能拿到,而不是让调用方自己猜。
        """
        order = greedy_maxmin_order(self.FIVE, 3, 0)
        assert order[0] == 0 and order == [0, 4, 3]
        assert greedy_maxmin(self.FIVE, 3, 0) == sorted(order)

    def test_pick_order_is_a_prefix_of_a_larger_k(self):
        """★ 嵌套性:`k` 的结果是 `k+1` 的**前缀**(同一实现、同一序列上截断)。

        `spread_curve` 的单调不增就建立在它上面 —— 若不是嵌套,加一个点反而可能
        让最近间距**变大**,曲线就成了锯齿。
        """
        for k in (2, 3, 4):
            assert greedy_maxmin_order(self.FIVE, k + 1, 0)[:k] == greedy_maxmin_order(self.FIVE, k, 0)


class TestFarthestFromCentroid:
    """贪心选点的**种子规则** —— 把"第一个点从哪来"这个自由度消掉。

    ★ 这条规则是**反推出来的**(2026-10-03):`surround_v2_epic` 的选点当时是现算的、
    规则没留档。实测 `Town10HD_Opt` 上只有"离质心最远"能逐位复现那五个点
    (种子 → 44,最近间距 113.994 m,与 milestone2 记的 114 m 吻合)。
    这几条钉的是它的**确定性**与**平局口径** —— 两者一变,新旧两批段就不可比了。
    """

    FIVE = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0), (100.0, 100.0)]

    def test_picks_the_outlier_not_the_cluster(self):
        """质心 (24,24):方块里四个点距离 19.8–33.9,远处那个 107.5 ⇒ 选它。"""
        assert farthest_from_centroid(self.FIVE) == 4

    def test_is_seedless_so_the_same_map_gives_the_same_answer(self):
        """同一份点集调用两次必须同解 —— 规则里不含随机、不含调用方状态。"""
        assert farthest_from_centroid(self.FIVE) == farthest_from_centroid(list(self.FIVE))

    def test_ties_go_to_the_lowest_index(self):
        """平局取下标最小 —— 换成"取最后一个"会让结果变成实现细节(浮点扫描顺序)。"""
        square = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]  # 质心 (5,5),四点等距
        assert farthest_from_centroid(square) == 0
        assert farthest_from_centroid([(0.0, 0.0), (10.0, 0.0)]) == 0

    def test_empty_point_set_raises(self):
        """空集没有质心 —— 报错,不是返回 0(那会静默选一个不存在的点)。"""
        with pytest.raises(ValueError, match="为空"):
            farthest_from_centroid([])

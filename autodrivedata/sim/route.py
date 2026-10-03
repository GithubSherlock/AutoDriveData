"""闭环路线:路网找环 + 纯追踪控制律(零 carla,唯一落点)。

**为什么需要它**:SLAM 后端(ScanContext 回环 + PGO)在真数据上**一次都没触发过** ——
已测的两条序列都是 `n_loops=0`。根因是**路线**:`outputs/kitti_slam` 400 帧走的是**马蹄形**,
两条平行街相距 **68 m**,而 `SC_MAX_RANGE = 40 m` 的 range-view 描述子看不到对面那条街
(实测「自接近最小 58.70 m」,间距 <40 m 的帧对数为 0)。且**车在第 320 帧就停了** ——
⇒「多采帧数」是无效解(只是多录静止帧),必须**换控制方式**。

本模块把采集器里可单测的纯几何抽出来(与 [`collect_rig.py`](collect_rig.py) 同一路子):
CARLA 采集器只做编排(拿 `Waypoint` 建图 + 执行控制),**图算法与控制律在这里,零 carla**。

## 为什么环搜索接受 `successors` 回调而不是直接吃 `carla.Map`

`find_cycle(start, successors)` 只要求一个 `successors(node) -> list[node]`,自己只做 BFS。
于是**可以在合成图上手算锚点单测**(方框图 → 4 节点最短环),而不必起 CARLA。

CARLA 侧的那层适配(`make_successors`,把 `wp.next(step)` 与节点键对上)**也在这里**,
且只吃一个 duck-typed 的 `WaypointLike` —— 于是连适配层本身都能在**合成 Waypoint** 上单测
(见 `tests/sim/test_route.py::TestMakeSuccessors`)。这不只是洁癖:「键里必须带 `s`」
「一圈下来键能不能精确回到起点」这两条最容易写错、又**只有在真服务器上才暴露**的规则,
钉在这里才有回归价值。

## 角度口径(唯一口径,别在别处再推一遍)

与 CARLA `Rotation.yaw` 一致:世界系 `atan2(y, x)`,**+y 是车的右侧**(CARLA 是左手系,
`geometry.carla_rotation_matrix` 列 1 = 局部 y 轴 = `(−sin ψ, cos ψ)`)。
故:**角度误差为正 ⇒ 目标在右 ⇒ `steer` 取正**(CARLA `steer` 正 = 右转)。这条被单测钉死
(`test_steer_sign_points_toward_target`),改之前先看那条。

## 「两圈」怎么实现的

不是新状态 —— 只是 `wrap_at`(回绕落点):跑完一圈后前视点索引回到**环入口**而不是 0,
`Control.wrapped` 是圈数计数的**唯一依据**。`wrap_at` 存在的意义是「起点到环入口那段引路
(prefix)只跑一次」,不该在第二圈重跑(否则车会掉头回去找起点)。

## 另:多段采集的选点也在这里

`greedy_maxmin` / `min_pairwise` / `spread_curve` 是**段起点选点**(贪心最大最小距离)。
放这里而不是放采集器里,同一条理由:CARLA 侧只该做编排,可单测的几何放这边。

⚠️ 贪心**对种子敏感**(第一个点从哪来会改变整组结果)⇒ 种子**不要手挑**,
走 `farthest_from_centroid`。那条规则是**反推出来的** —— 它在 `Town10HD_Opt` 上
逐位复现了 `surround_v2_epic` 现算的那五个点(见 `probe_spawn_points --expect`)。
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Protocol, cast

# 纯追踪:航向误差(弧度)→ steer 的比例增益。误差 0.4 rad(23°)即满舵,留足余量不抖。
K_STEER = 1.5
# 速度误差(m/s)→ throttle / brake 的比例增益
K_THROTTLE = 0.35
K_BRAKE = 0.4


class WaypointLike(Protocol):
    """CARLA `Waypoint` 的**最小面**(duck-typed):节点键 + 后继展开只用这四个成员。

    用 Protocol 而不是 `import carla`,是为了让适配层能在合成 Waypoint 上单测
    (见 `tests/sim/test_route.py::TestMakeSuccessors`)—— 本模块零 carla 的纪律也因此不破。
    """

    # 声明成只读 property 而不是可变属性:carla 的桩里这三个是 property,
    # 而「可读 property」不满足「可变属性」⇒ 写成属性会整类不兼容(实测 Pyright 报
    # "road_id 是固定的,因为它是可变的")。
    @property
    def road_id(self) -> int: ...
    @property
    def lane_id(self) -> int: ...
    @property
    def s(self) -> float: ...

    def next(self, distance: float) -> Sequence[WaypointLike]: ...


# 节点键 = (road_id, lane_id, round(s/step));`s` 是**沿本车道**的里程,不是全局的
NodeKey = tuple[int, int, int]


@dataclass(frozen=True)
class Cycle:
    """路网里的一条有向环 + 从起点到环入口的引路。

    `loop[i]` 的后继是 `loop[i+1]`,`loop[-1]` 的后继是 `loop[0]`(**环是真的闭合的**)。
    `prefix` 是「起点 → 环入口」的 BFS 路径**但不含环入口本身**(含了就与 `loop[0]` 重复);
    起点本身在环上时 `prefix` 为空 —— 网格城市里这通常是常态。
    """

    prefix: tuple[Hashable, ...]
    loop: tuple[Hashable, ...]

    @property
    def route_len(self) -> int:
        return len(self.prefix) + len(self.loop)

    @property
    def loop_start(self) -> int:
        """环在「prefix + loop」拼起来的整条路线里的起始索引(= 回绕落点)。"""
        return len(self.prefix)


def find_cycle(
    start: Hashable,
    successors: Callable[[Hashable], Sequence[Hashable]],
    *,
    max_nodes: int = 4000,
    min_len: int = 4,
) -> Cycle | None:
    """BFS 求**从 `start` 出发可达的最短有向环**;找不到返回 `None`。

    - **有界**:`max_nodes` 限制 BFS 规模(每次 `successors` 在 CARLA 侧是一次 RPC,
      不设上限在真实路网上会跑很久)。
    - **`min_len` 拒绝退化环**:`A→B→A` 这种两点往返在真实路网上不是环(掉头),
      而且它会让「跑两圈」退化成原地来回蹭。默认 4(方块城市一个街区 = 4 段)。
    - **`successors` 只许返回「合法的前向后继」**:CARLA 的 `wp.next()` 天然满足
      (`previous()` 不在此列),所以本算法不会找出逆行环。
    - **不保证环含 `start`**:`start` 不在环上时返回引路 + 环(见 `Cycle`)。

    实现口径:边 `u→v` 且 `v` 已被访问 ⇒ 发现环,长度 = `depth[u] − depth[v] + 1`
    (BFS 树深度差 + 那条回边)。扫完整个 BFS 取最短的,**不用「第一个发现的」** ——
    BFS 出队的深度序不保证回边长度单调。
    """
    if min_len < 2:
        raise ValueError(f"min_len 至少为 2(A→B→A 是最短的环),收到 {min_len}")
    parent: dict[Hashable, Hashable | None] = {start: None}
    depth: dict[Hashable, int] = {start: 0}
    queue: deque[Hashable] = deque([start])
    best: tuple[int, Hashable, Hashable] | None = None  # (环长, 环入口 v, 环出口 u)

    while queue and len(depth) <= max_nodes:
        u = queue.popleft()
        for v in successors(u):
            if v in depth:
                if depth[u] >= depth[v]:  # 只认「回边」;同层/向下的交叉边不成环
                    length = depth[u] - depth[v] + 1
                    if length >= min_len and (best is None or length < best[0]):
                        best = (length, v, u)
            else:
                depth[v] = depth[u] + 1
                parent[v] = u
                queue.append(v)

    if best is None:
        return None
    _, entry, exit_ = best

    loop: list[Hashable] = []
    node: Hashable | None = exit_
    while node != entry:
        if node is None:  # 只可能来自 parent[start] = None,而 entry != start 时不可达
            raise AssertionError(f"环重建失败:回溯越过根(entry={entry!r})")
        loop.append(node)
        node = parent[node]
    loop.append(entry)
    loop.reverse()  # [entry, …, exit]

    prefix_full: list[Hashable] = []
    node = entry
    while node is not None:
        prefix_full.append(node)
        node = parent[node]
    prefix_full.reverse()  # [start, …, entry]
    return Cycle(prefix=tuple(prefix_full[:-1]), loop=tuple(loop))


def make_successors(
    step: float,
) -> tuple[
    Callable[[Hashable], list[Hashable]],
    Callable[[WaypointLike], NodeKey],
    dict[NodeKey, WaypointLike],
]:
    """CARLA 路网 → `find_cycle` 要的 `successors` 回调 + `键 → Waypoint` 缓存。

    返回 `(successors, register, cache)`:开局 `register(起点 Waypoint)` 拿起始键,
    跑完 `find_cycle` 后用 `cache[键]` 换回 Waypoint 取坐标。

    **`s` 必须进键**:只留 `(road_id, lane_id)` 的话一条长路只有一个节点,`next()` 展开出
    来的后继键等于自己 ⇒ `A→A` 自环(被 `min_len` 挡掉)⇒ **永远找不到环**。
    桶宽取 `step`:`next(step)` 的落点与键同格,故一圈能精确回到起点键(偏差 < 半桶)——
    这也是闭环容差该按 `step` 给、不该按 1 m 给的原因。

    `successors` 的签名必须与 `find_cycle` 一样宽(`Hashable`),因为图算法对节点类型
    是**不透明**的;`cache[k]` 必然命中 —— `find_cycle` 只会用 `register` 或本回调
    吐出去的键回调进来(`setdefault` 就在 `append` 之前)。
    """
    if step <= 0.0:
        raise ValueError(f"采样步长必须 > 0,收到 {step}")
    cache: dict[NodeKey, WaypointLike] = {}

    def key_of(wp: WaypointLike) -> NodeKey:
        return (wp.road_id, wp.lane_id, int(round(wp.s / step)))

    def successors(k: Hashable) -> list[Hashable]:
        key = cast(NodeKey, k)
        out: list[Hashable] = []
        for nxt in cache[key].next(step):
            nk = key_of(nxt)
            cache.setdefault(nk, nxt)  # 首见优先:同一键由不同 Waypoint 映射时不翻旧账
            out.append(nk)
        return out

    def register(wp: WaypointLike) -> NodeKey:
        k = key_of(wp)
        cache.setdefault(k, wp)
        return k

    return successors, register, cache


# ---------------------------------------------------------------------------
# 路线几何
# ---------------------------------------------------------------------------
def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def route_length(points: Sequence[Sequence[float]]) -> float:
    """折线长度(米,只算 xy)。"""
    return float(sum(_dist(a, b) for a, b in zip(points, points[1:], strict=False)))


def route_closure(points: Sequence[Sequence[float]], *, tol: float = 1.0) -> dict:
    """闭环判据:首末距离 / 总长 / 是否闭合。

    **这是找环之后的第一道闸**(`--dry-run` 用它):BFS 说「这是个环」不代表几何上首尾接得上 ——
    CARLA 路网连通性有已知瑕疵,节点键虽按位置分桶、但采样步长与真实车道长度不整除时
    首末会有残差。残差超 `tol` 就得报出来,不能假装闭合。
    """
    if not points:
        return {"n": 0, "length_m": 0.0, "end_gap_m": 0.0, "closed": False}
    gap = _dist(points[0], points[-1])
    # 环长要把「末点回到首点」那段算进去,否则系统性低估(方块城市里那是整整一条边)
    return {
        "n": len(points),
        "length_m": route_length(points) + gap,
        "end_gap_m": gap,
        "closed": gap <= tol,
    }


def farthest_from_centroid(points: Sequence[Sequence[float]]) -> int:
    """离**全部点的质心**最远的那个下标 —— 贪心选点的**种子规则**(把种子这个自由度消掉)。

    ★ 这条规则是**反推出来的,不是发明的**(2026-10-03):`surround_v2_epic` 的
    `44/14/15/152/55`(milestone2 记的「贪心最大最小距离……两两最近 114 m」)当时是现算的,
    没留规则。实测 `Town10HD_Opt` 的 155 个 spawn point:按"离质心最远"取种子(→ **44**)
    再贪心,**逐位复现**那五个点(最近间距 113.994 m)。其余候选规则(max x / min y /
    最大模长 / 种子 0)都不命中,种子 55 虽能给出更大的 117.2 m 但**不是**当时那组。

    ⇒ 新图沿用**同一条规则**,新旧两批段的选点口径才可比。

    平局取**下标最小**的(`max` 的语义),故结果与机器无关、可复现。
    """
    if not points:
        raise ValueError("点集为空 —— 没有质心可算")
    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    return max(range(n), key=lambda i: math.hypot(points[i][0] - cx, points[i][1] - cy))


def greedy_maxmin_order(points: Sequence[Sequence[float]], k: int, seed: int) -> list[int]:
    """贪心最大最小距离(**farthest point sampling**),按**选中顺序**返回。

    每步选「到已选集合的**最近**距离」最大的那个。**第一个恒为 `seed`**。

    **为什么用它选多段采集的起点**:段间起点离得近 ⇒ 两段走同一条街,扩的是帧数
    不是**路线多样性**,而 MapTR 线要的恰恰是后者(矢量从 xodr 解析、不依赖外观,
    要的是拓扑多样性)。`surround_v2_epic` 的 `44/14/15/152/55` 就是这么来的 ——
    ⚠️ **那个写法是选择序不是排序**(首元素 44 = 种子),拿它比对时要选对口径。

    平局取**下标最小**的(`>` 严格比较 ⇒ 先到先得),故结果与机器无关。
    """
    if not 0 <= seed < len(points):
        raise ValueError(f"种子 {seed} 越界(共 {len(points)} 个点)")
    if k < 1:
        raise ValueError(f"k 必须 ≥ 1,收到 {k}")
    sel = [seed]
    while len(sel) < k:
        best_i, best_d = -1, -1.0
        for i, p in enumerate(points):
            if i in sel:
                continue
            d = min(_dist(p, points[j]) for j in sel)
            if d > best_d:
                best_i, best_d = i, d
        if best_i < 0:  # k > 点数
            break
        sel.append(best_i)
    return sel


def greedy_maxmin(points: Sequence[Sequence[float]], k: int, seed: int) -> list[int]:
    """同 `greedy_maxmin_order`,但返回**升序下标** —— 供集合比较用。

    下游(`probe_spawn_points --expect`)拿它与已知索引集直接 `==` 比,
    所以必须是集合口径:按选择序返回会让"复现了没有"变成"得先猜对方用什么顺序"。

    ⚠️ **对种子敏感**:第一个点从哪来会改变整组结果(实测同样 5 点、同样 k=3,
    种子 0/3/4 给 `{0,3,4}` 而种子 1/2 给 `{1,2,4}`)。种子**不要手挑** ——
    走 `farthest_from_centroid`(那是复现出已知集合的那条规则)。
    """
    return sorted(greedy_maxmin_order(points, k, seed))


def min_pairwise(points: Sequence[Sequence[float]], idx: Sequence[int]) -> float:
    """所选点两两之间的**最近**距离(米)—— 就是「段间离得够不够远」那个数。

    少于两点时返回 `inf`:**一个点谈不上间距**,写成 0 会被下游读成"两点重合",
    于是 `--min-gap` 判据在只有一个点时**误报不通过**。
    """
    if len(idx) < 2:
        return math.inf
    return min(_dist(points[i], points[j]) for i, j in combinations(sorted(idx), 2))


def spread_curve(points: Sequence[Sequence[float]], k: int, seed: int) -> list[tuple[int, float]]:
    """`k' = 2..k` 的最近间距曲线 —— 判「这张图撑不撑得起 k 段」。

    只看最终那一个数看不出"再加一段会掉多少"。曲线掉得陡 ⇒ 这张图的路网本来就
    不大,再多分段只是把同一片街区切细(段数涨、多样性不涨)。
    """
    out: list[tuple[int, float]] = []
    for kk in range(2, k + 1):
        out.append((kk, min_pairwise(points, greedy_maxmin(points, kk, seed))))
    return out


def speed_ceiling(loop_len: float, *, delta: float, min_frames_per_lap: int) -> float:
    """一圈至少 `min_frames_per_lap` 帧 ⇒ 允许的最高速度(m/s)。**与 `speed` 无关**,单列出来。

    采集器要在**选速度之前**拿到它(`--speed 0` = 自动取 `min(默认, 上限)`),
    所以它不能藏在 `lap_budget` 的返回值里跟着一个"猜的速度"一起算。
    """
    if loop_len <= 0.0 or delta <= 0.0 or min_frames_per_lap <= 0:
        raise ValueError(f"参数必须 > 0:loop_len={loop_len} delta={delta} {min_frames_per_lap=}")
    return loop_len / (min_frames_per_lap * delta)


def lap_budget(
    loop_len: float,
    *,
    laps: int,
    delta: float,
    speed: float,
    min_frames_per_lap: int,
    accel_margin: float = 1.25,
) -> dict:
    """单圈帧预算 + **速度上限** —— 闭环采集的第一道闸(采集前算,别采完才发现白跑)。

    **为什么是速度上限而不是帧数上限**:回环候选要求两次到访的帧差 ≥
    `SC_MIN_GAP_NODES × KEYFRAME_EVERY`(= 250 帧)。一圈跑多少帧 = `环长 / (速度 × tick)`,
    所以**环越短就必须开得越慢** —— 开快了不是"少采几帧",而是那两圈**根本不可能**触发回环
    (第二圈到访时第一圈才过去 150 帧,连候选都进不去)。这条约束必须**在采集前**报出来。

    `min_frames_per_lap` 由调用方从 `slam.core` 的常量算好传进来 —— 本模块零 slam 依赖,
    那两个常数是 SLAM 后端的判据,不该在这里复制一份(复制就会漂)。
    """
    if laps < 1:
        raise ValueError(f"laps 至少为 1,收到 {laps}")
    if speed <= 0.0:
        raise ValueError(f"speed 必须 > 0,收到 {speed}")
    speed_max = speed_ceiling(loop_len, delta=delta, min_frames_per_lap=min_frames_per_lap)
    frames_per_lap = loop_len / (speed * delta)
    return {
        "loop_len_m": loop_len,
        "speed": speed,
        "speed_max": speed_max,
        "frames_per_lap": frames_per_lap,
        "min_frames_per_lap": min_frames_per_lap,
        # 加速段 + 收尾余量;不乘余量会刚好卡在 250 帧上,一轮抖动就掉出门槛
        "frames_recommended": int(laps * frames_per_lap * accel_margin) + 50,
        "ok": speed <= speed_max,
    }


def normalize_angle(a: float) -> float:
    """弧度归一化到 `[−π, π)`(**与 `math.atan2` 同区间**,不是 `(−π, π]`)。

    口径写错不影响打舵方向(只有「正后方」那一个点会取到不同符号,而那里本来就左右等价),
    但**必须与 `atan2` 一致** —— 否则误差角在 π 附近的连续性会与目标方位角对不上。
    """
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def lookahead_index(
    points: Sequence[Sequence[float]],
    pos: Sequence[float],
    from_idx: int,
    lookahead: float,
    *,
    wrap_at: int | None = None,
    max_skip: int = 60,
) -> tuple[int, bool]:
    """从 `from_idx` 向前找**第一个距 `pos` ≥ `lookahead` 的点** → `(索引, 是否回绕)`。

    **进度单调是硬要求**:只向前找、绝不返回 `from_idx` 之前 —— 否则车会在起点附近
    来回蹭(纯追踪最经典的一类失败)。回绕(`i >= len(points)`)时落到 `wrap_at`
    (环入口)而不是 0,这样「起点 → 环入口」的引路段只跑一次;`wrap_at is None`(非闭环
    路线)则钳在末点。

    `max_skip` 兜底:车若已严重偏离(前视点怎么找都不够远),不至于空转整个路线;
    找不到就返回最后的候选,让上层的**偏航自证**去报错(不在这里静默吞掉)。
    """
    n = len(points)
    if n == 0:
        raise ValueError("路线为空")
    i = from_idx % n
    for _ in range(max_skip):
        if _dist(points[i], pos) >= lookahead:
            return i, False
        i += 1
        if i >= n:
            if wrap_at is None:
                return n - 1, False
            return wrap_at, True
    return i % n, False


def track_index(
    points: Sequence[Sequence[float]],
    pos: Sequence[float],
    from_idx: int,
    *,
    back: int = 4,
    forward: int = 6,
) -> int:
    """自证用的**进度索引**:在 `from_idx` 附近(环上 `±` 有界窗口)取距 `pos` 最近的路线点。

    **与 `lookahead_index` 是两个不同的量,不要混用**(踩过):前视索引取的是「≥ lookahead
    的前方点」,拿它当自证锚点量到的是**前视距离**而不是横向偏差 —— 完美跟线也会读到
    8~9 m,而中止阈值恰好是 8 m,一次**正常**的采集就能踩响中止,真跟丢时反而不报警。
    自证的锚点必须是"车现在走到哪了",即本函数。

    两条硬约束,各有实测的失败对应:

    - **窗口必须绕环(`% n`)**:把上界钳在 `n` 会让车跑完第一圈回环首时索引**永远停在
      `n-1`**,误差变成"车到末点的距离"、单调涨到 31 m ⇒ **第二圈刚起步就踩响中止**
      (实测帧 469,进度点 85/86)。闭环的进度索引必须能跨过环缝。
    - **窗口必须有界**(`back`/`forward`,不是全局最近点):环会自我靠近,全局最近点会在
      第二圈的对应位置给出 ≈0,把"跟丢了"掩盖成完美(与 `lateral_error` 同一个陷阱)。
      `forward` 取每帧名义前进量的几十倍(3 m 节点、0.6 m/帧 ⇒ 0.2 节点/帧)即可 ——
      它同时是**前进量上限**:车若"被判前进"几十个节点,那是跟丢了,该让上层的
      偏航自证看到不断变大的误差,而不是被一次跳跃救回来。
    """
    n = len(points)
    if n == 0:
        raise ValueError("路线为空")
    cur = from_idx % n
    window = [(cur + k) % n for k in range(-back, forward + 1)]
    return min(window, key=lambda k: _dist(points[k], pos))


def lateral_error(
    points: Sequence[Sequence[float]],
    pos: Sequence[float],
    from_idx: int,
    *,
    span: int = 2,
) -> float:
    """车到路线在**进度索引**附近 `±span` 个点的最小距离(米)= 横向偏差。采集器的自证用它。

    `from_idx` 必须是 `track_index` 的结果,**不是** `lookahead_index` 的(见 `track_index`)。
    `span` 只用来吃掉**节点量化**:3 m 步长下,车恰好卡在两节点中间时点到点距离会虚高
    ~1.5 m,取 ±2 个点后这个虚假分量消失。**不要放大到前视那种量级** —— 那量的是前视距离。
    """
    n = len(points)
    if n == 0:
        return float("inf")
    idx = [(from_idx + k) % n for k in range(-span, span + 1)]
    return float(min(_dist(points[k], pos) for k in idx))


@dataclass(frozen=True)
class Control:
    """一帧的控制量 + 前视点信息(采集器直接喂 `carla.VehicleControl`)。"""

    throttle: float
    steer: float
    brake: float
    idx: int  # 本帧前视点索引(下一帧从它继续向前找)
    wrapped: bool  # ★ 本帧是否回绕 = 圈数计数的唯一依据


def pure_pursuit(
    pos: Sequence[float],
    yaw: float,
    points: Sequence[Sequence[float]],
    from_idx: int,
    *,
    speed: float,
    target_speed: float,
    lookahead: float,
    wrap_at: int | None = None,
    k_steer: float = K_STEER,
    k_throttle: float = K_THROTTLE,
    k_brake: float = K_BRAKE,
) -> Control:
    """纯追踪:朝前视点打舵 + 按速度误差给油/刹车。角度口径见模块头注。

    - `yaw` 单位**弧度**,世界系 `atan2(y, x)`(与 `Rotation.yaw` 同向,只是换了单位)。
    - **误差为正 ⇒ 目标在右 ⇒ `steer` 为正**(CARLA 正 = 右转)。
    - 定速优先:超速先刹车、不叠油门(`throttle` 与 `brake` 不同时给,否则车会「一边踩一边刹」)。
    """
    idx, wrapped = lookahead_index(points, pos, from_idx, lookahead, wrap_at=wrap_at)
    target = points[idx]
    err = normalize_angle(math.atan2(target[1] - pos[1], target[0] - pos[0]) - yaw)
    steer = max(-1.0, min(1.0, k_steer * err))
    dv = target_speed - speed
    if dv >= 0.0:
        throttle, brake = max(0.0, min(1.0, k_throttle * dv)), 0.0
    else:
        throttle, brake = 0.0, max(0.0, min(1.0, k_brake * -dv))
    return Control(throttle, steer, brake, idx, wrapped)

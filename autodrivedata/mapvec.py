"""地图矢量 GT(MapTR 口径)提取与采样(纯值,不 import carla)。

§5.11 A2+A3:把 `opendrive.py` 的解析结果映射为 MapTR 三类 + 工程补充,
输出世界系折线实例;等距重采样与 ego 窗口裁剪为独立纯函数。

要素映射(xodr → 类;口径实测于 Town10HD_Opt,2026-09-11):
- divider   = 同向 driving 车道间的标线(mark 挂**外缘**,OpenDRIVE 标准)
             + center 车道(lane 0)的 mark = 双向道路中心线(归 divider,
             超出定案"同向"字面,MapTR 语义含对向分隔;attrs 记 centerline=yes)
- boundary  = curb mark(路缘石)+ driving 最外实线 + 最外两侧兜底
             (sidewalk/shoulder 外缘);无 mark 的内部边(如 driving-shoulder)
             不输出
- ped_crossing = <object type="crosswalk"> 的 outline 4 角(5 点闭合,
             世界系;cornerLocal 按 s/t/hdg 变换)
- stop_line = <object name="StopLine"> 的 outline 首末 2 点(实测为沿 hdg
             的线段,3 点/15 点形态均取首末)
- centerline = 每条非 center 车道中心线(t = (lo+hi)/2 链;attrs 带
             lane_type/lane_id,消费方可过滤)
- traffic_light = <signal>(validity 车道关联 + 位置;复用 P2 landmark 口径)

采端口径:divider/boundary/centerline 沿 s 以 0.5m 步长折线化(A2),
`resample` 等距 20 点(A3);crop_to_ego 为方形窗口 ±radius 裁剪(跨窗折线
裁成窗口内子段并分裂实例)。
"""

from __future__ import annotations

from dataclasses import dataclass

from .opendrive import Lane, LaneSection, OpenDriveMap, Road, lane_boundary_t, road_to_xy

_S_STEP = 0.5  # A2 折线化步长(m)
_EDGE_EPS = 1e-2  # 相邻区间判定容差(m)
_DIVIDER_MARKS = frozenset({"solid", "broken", "solid solid"})


@dataclass(frozen=True)
class MapVec:
    """一条地图矢量实例:类 + 世界系折线 + 属性 + 实例 id + 溯源。"""

    cls: str  # divider | boundary | ped_crossing | stop_line | centerline | traffic_light
    points: tuple[tuple[float, float, float], ...]
    attrs: tuple[tuple[str, str], ...] = ()  # 有序 (key, value),允许重复键(如 validity 多车道对)
    id: str = ""
    src: str = ""

    def with_points(self, points: tuple[tuple[float, float, float], ...], id_suffix: str = "") -> MapVec:
        return MapVec(self.cls, points, self.attrs, self.id + id_suffix, self.src)


# ---------- 几何辅助 ----------


def _t_zero(_s: float) -> float:
    """center 车道 mark 恒在 reference 线(t=0)。"""
    return 0.0


def _outer_t(lane: Lane, road: Road, s: float) -> float:
    """lane 在 s 处的外缘 t(离 reference 远的一侧;lane 无边界 raise)。"""
    lo, hi = lane_boundary_t(road, s, lane.id)
    if lo > hi:  # 宽度为负的 lane(CARLA 单向段),归一化
        lo, hi = hi, lo
    return hi if abs(hi) >= abs(lo) else lo


def _lane_intervals(road: Road, sec: LaneSection, s_probe: float) -> list[tuple[Lane, float, float]]:
    """section 内各非 center 车道的 t 区间 [(lane, lo, hi)](已归一 lo<=hi);无宽度的跳过。"""
    out: list[tuple[Lane, float, float]] = []
    for lane in sec.left + sec.right:
        try:
            lo, hi = lane_boundary_t(road, s_probe, lane.id)
        except ValueError:
            continue
        if lo > hi:
            lo, hi = hi, lo
        if hi - lo > 1e-6:
            out.append((lane, lo, hi))
    return sorted(out, key=lambda x: x[1])


@dataclass
class _Edge:
    """一条边界线:两侧 lane 列表(最外单边 1 条),共享 t 值,是否最外。"""

    t: float
    lanes: tuple[Lane, ...]
    outermost: bool


def _section_edges(road: Road, sec: LaneSection) -> list[_Edge]:
    """section 的全部边界线:相邻车道共享边 + 最外两侧(含 center 两侧缝隙)。"""
    iv = _lane_intervals(road, sec, sec.s + 0.1)
    if not iv:
        return []
    edges: list[_Edge] = []
    for (la, _, hia), (lb, lob, _) in zip(iv, iv[1:], strict=False):
        if abs(hia - lob) < _EDGE_EPS:
            edges.append(_Edge(t=hia, lanes=(la, lb), outermost=False))
    edges.append(_Edge(t=iv[0][1], lanes=(iv[0][0],), outermost=iv[0][1] < -_EDGE_EPS))
    edges.append(_Edge(t=iv[-1][2], lanes=(iv[-1][0],), outermost=iv[-1][2] > _EDGE_EPS))
    return edges


def _mark_segments(lane: Lane, sec: LaneSection, next_s: float) -> tuple[tuple[float, float, str, str], ...]:
    """lane 的 mark 生效段 [(s_start, s_end, type, color)],相邻同属性段合并;无实体段返回 ()。"""
    marks = [mk for mk in lane.marks if mk.type != "none"]
    if not marks:
        return ()
    segs: list[tuple[float, float, str, str]] = []
    for i, mk in enumerate(marks):
        s0 = sec.s + mk.s_offset
        s1 = sec.s + marks[i + 1].s_offset if i + 1 < len(marks) else next_s
        if s1 - s0 <= _EDGE_EPS:
            continue
        if segs and segs[-1][2] == mk.type and segs[-1][3] == mk.color and abs(segs[-1][1] - s0) < _EDGE_EPS:
            segs[-1] = (segs[-1][0], s1, mk.type, mk.color)
        else:
            segs.append((s0, s1, mk.type, mk.color))
    return tuple(segs)


def _sample_boundary(road: Road, s0: float, s1: float, t_fn) -> tuple[tuple[float, float, float], ...]:
    """沿 s 以 _S_STEP 折线化边界几何 t_fn(s) → 世界系点列(含两端点)。"""
    pts: list[tuple[float, float, float]] = []
    s = s0
    while s < s1 + 1e-9:
        pts.append(road_to_xy(road, s, t_fn(s)))
        s = min(s + _S_STEP, s1)
        if s >= s1 - 1e-9 and (not pts or pts[-1] != pts[0]):
            break
    last = road_to_xy(road, s1, t_fn(s1))
    if len(pts) < 2 or abs(pts[-1][0] - last[0]) > 1e-9 or abs(pts[-1][1] - last[1]) > 1e-9:
        pts.append(last)
    return tuple(pts)


# ---------- 六类提取 ----------


def _extract_lane_marks(m: OpenDriveMap, out: list[MapVec], nxt: list[int]) -> None:
    """divider(boundary 一并判定):lane 外缘 mark + center 车道 mark。"""
    for road in m.roads.values():
        secs = road.lane_sections
        for si, sec in enumerate(secs):
            next_s = secs[si + 1].s if si + 1 < len(secs) else road.length
            if next_s - sec.s <= _EDGE_EPS:
                continue
            edges = _section_edges(road, sec)
            for e in edges:
                # 共享边:外缘在该 t 的 lane 承载 mark;单边:唯一 lane 的外缘
                holder = next(
                    (ln for ln in e.lanes if abs(_outer_t(ln, road, sec.s + 0.1) - e.t) < _EDGE_EPS),
                    None,
                )
                if holder is None:
                    continue
                sides_driving = all(ln.type == "driving" for ln in e.lanes) and len(e.lanes) == 2
                for s0, s1, mtype, color in _mark_segments(holder, sec, next_s):
                    pts = _sample_boundary(
                        road, s0, min(s1, next_s), lambda s, h=holder, r=road: _outer_t(h, r, s)
                    )
                    if len(pts) < 2:
                        continue
                    nxt[0] += 1
                    attrs = [("mark_type", mtype), ("color", color), ("road_id", str(road.id))]
                    if sides_driving and mtype in _DIVIDER_MARKS:
                        cls = "divider"
                    elif mtype == "curb" or e.outermost:
                        cls = "boundary"
                    elif mtype in _DIVIDER_MARKS and len(e.lanes) == 2:
                        cls = "boundary"  # driving 外缘实线 / 与非 driving 相邻的标线
                    else:
                        continue
                    if cls == "divider":
                        attrs.append(("same_dir", "yes"))
                    out.append(
                        MapVec(
                            cls=cls,
                            points=pts,
                            attrs=tuple(attrs),
                            id=f"{cls}_{nxt[0]:05d}",
                            src=f"road {road.id} lane {holder.id} s[{s0:.2f},{s1:.2f}]",
                        )
                    )
            # center 车道 mark = 双向道路中心线(divider)
            for cl in sec.center:
                for s0, s1, mtype, color in _mark_segments(cl, sec, next_s):
                    pts = _sample_boundary(road, s0, min(s1, next_s), _t_zero)
                    if len(pts) < 2 or mtype not in _DIVIDER_MARKS:
                        continue
                    nxt[0] += 1
                    out.append(
                        MapVec(
                            cls="divider",
                            points=pts,
                            attrs=(
                                ("mark_type", mtype),
                                ("color", color),
                                ("centerline", "yes"),
                                ("same_dir", "no"),
                                ("road_id", str(road.id)),
                            ),
                            id=f"divider_{nxt[0]:05d}",
                            src=f"road {road.id} center-lane s[{s0:.2f},{s1:.2f}]",
                        )
                    )


def _object_world_pts(road: Road, obj, local: tuple[float, float, float]) -> tuple[float, float, float]:
    """cornerLocal (u, v, z) → 世界系(绕 object hdg 旋转,沿 s/t 平移)。"""
    import math

    u, v, z = local
    ch, sh = math.cos(obj.hdg), math.sin(obj.hdg)
    s = obj.s + u * ch - v * sh
    t = obj.t + u * sh + v * ch
    x, y, z0 = road_to_xy(road, s, t)
    return x, y, z0 + z + obj.z_offset


def _extract_objects(m: OpenDriveMap, out: list[MapVec], nxt: list[int]) -> None:
    """ped_crossing(outline 4 角 5 点闭合)与 stop_line(首末 2 点)。"""
    for road in m.roads.values():
        for obj in road.objects:
            pts: list[tuple[float, float, float]] = []
            if obj.type == "crosswalk" and obj.outline:
                pts = [_object_world_pts(road, obj, c) for c in obj.outline]
                if len(pts) >= 4:
                    pts = pts[:4]
                    pts.append(pts[0])  # 闭合
            elif "StopLine" in obj.name:
                if len(obj.outline) >= 2:
                    pts = [
                        _object_world_pts(road, obj, obj.outline[0]),
                        _object_world_pts(road, obj, obj.outline[-1]),
                    ]
                else:
                    import math

                    ch, sh = math.cos(obj.hdg), math.sin(obj.hdg)
                    pts = [
                        road_to_xy(road, obj.s - obj.length / 2 * ch, obj.t - obj.length / 2 * sh),
                        road_to_xy(road, obj.s + obj.length / 2 * ch, obj.t + obj.length / 2 * sh),
                    ]
            else:
                continue
            nxt[0] += 1
            cls = "ped_crossing" if obj.type == "crosswalk" else "stop_line"
            out.append(
                MapVec(
                    cls=cls,
                    points=tuple(pts),
                    attrs=(("road_id", str(road.id)), ("s", f"{obj.s:.2f}"), ("t", f"{obj.t:.2f}")),
                    id=f"{cls}_{nxt[0]:05d}",
                    src=f"road {road.id} object {obj.id}",
                )
            )


def _center_t(road: Road, lane: Lane, s: float) -> float:
    lo, hi = lane_boundary_t(road, s, lane.id)
    return (lo + hi) / 2.0


def _extract_centerlines(m: OpenDriveMap, out: list[MapVec], nxt: list[int]) -> None:
    """centerline:每条非 center 车道中心线(全 lane type,attrs 带 lane_type 供过滤)。"""
    for road in m.roads.values():
        secs = road.lane_sections
        for si, sec in enumerate(secs):
            next_s = secs[si + 1].s if si + 1 < len(secs) else road.length
            if next_s - sec.s <= _EDGE_EPS:
                continue
            for lane in sec.left + sec.right:
                try:
                    lo, hi = lane_boundary_t(road, sec.s + 0.1, lane.id)
                except ValueError:
                    continue
                if hi - lo <= 1e-6:
                    continue
                pts = _sample_boundary(
                    road, sec.s, min(next_s, road.length), lambda s, l=lane, r=road: _center_t(r, l, s)
                )
                if len(pts) < 2:
                    continue
                nxt[0] += 1
                out.append(
                    MapVec(
                        cls="centerline",
                        points=pts,
                        attrs=(
                            ("road_id", str(road.id)),
                            ("lane_id", str(lane.id)),
                            ("lane_type", lane.type),
                        ),
                        id=f"centerline_{nxt[0]:05d}",
                        src=f"road {road.id} lane {lane.id}",
                    )
                )


def _extract_signals(m: OpenDriveMap, out: list[MapVec], nxt: list[int]) -> None:
    """traffic_light:signal 位置 + validity 车道关联(复用 P2 landmark 口径)。"""
    for road in m.roads.values():
        for sig in road.signals:
            if sig.type not in ("traffic_light", "1000001"):  # 1000001 = OpenDRIVE 数字码
                continue
            nxt[0] += 1
            attrs: list[tuple[str, str]] = [
                ("road_id", str(road.id)),
                ("name", sig.name),
                ("subtype", sig.subtype),
                ("dynamic", sig.dynamic),
                ("s", f"{sig.s:.2f}"),
                ("t", f"{sig.t:.2f}"),
            ]
            for fl, tl in sig.validity:
                attrs.append(("validity", f"{fl}->{tl}"))
            out.append(
                MapVec(
                    cls="traffic_light",
                    points=(road_to_xy(road, sig.s, sig.t),),
                    attrs=tuple(attrs),
                    id=f"traffic_light_{nxt[0]:05d}",
                    src=f"road {road.id} signal {sig.id}",
                )
            )


def extract_mapvec(m: OpenDriveMap) -> tuple[MapVec, ...]:
    """A2:全图六类要素实例(世界系折线)。"""
    out: list[MapVec] = []
    nxt = [0]
    _extract_lane_marks(m, out, nxt)
    _extract_objects(m, out, nxt)
    _extract_centerlines(m, out, nxt)
    _extract_signals(m, out, nxt)
    return tuple(out)


# ---------- A3:重采样与裁剪 ----------


def resample(v: MapVec, n: int = 20) -> MapVec:
    """折线弧长等距重采样为 n 点(含端点);点数不足或 cls 固定形态(灯)原样返回。"""
    if len(v.points) < 2 or v.cls in ("traffic_light",):
        return v
    import numpy as np

    p = np.asarray(v.points, dtype=float)
    segs = np.linalg.norm(np.diff(p, axis=0), axis=1)
    total = segs.sum()
    if total <= 1e-9:
        return v
    targets = np.linspace(0.0, total, n)
    cum = np.concatenate(([0.0], np.cumsum(segs)))
    idx = np.searchsorted(cum, targets, side="right") - 1
    idx = np.clip(idx, 0, len(segs) - 1)
    # np.where 两分支都算,零长线段(裁剪后重合点)会 0/0 → 先全算再掩码
    frac = np.divide(
        targets - cum[idx],
        segs[idx],
        out=np.zeros_like(targets),
        where=segs[idx] > 1e-12,
    )
    pts = p[idx] + (p[idx + 1] - p[idx]) * frac[:, None]
    pts[-1] = p[-1]
    return v.with_points(tuple((float(x), float(y), float(z)) for x, y, z in pts))


def crop_to_ego(
    vecs: tuple[MapVec, ...], pose_xy: tuple[float, float], radius: float = 51.2
) -> tuple[MapVec, ...]:
    """方形窗口 |x-px|,|y-py| <= radius 裁剪(Liang-Barsky);跨窗折线裁成窗口内子段(可能分裂实例)。"""
    cx, cy = pose_xy
    out: list[MapVec] = []
    for v in vecs:
        pts = list(v.points)
        if v.cls == "ped_crossing" and len(pts) > 1:
            pts = pts + [pts[0]]  # 显式闭合,遍历边不重不漏
        pieces: list[list[tuple[float, float, float]]] = []
        cur: list[tuple[float, float, float]] | None = None
        for i in range(len(pts) - 1):
            r = _liang_barsky(pts[i], pts[i + 1], cx, cy, radius)
            if r is None:
                continue
            p0, p1 = r
            if cur is not None and _near(cur[-1], p0):
                cur.append(p1)  # 相邻段连续,合并
            else:
                if cur is not None:
                    pieces.append(cur)
                cur = [p0, p1]
        if cur is not None:
            pieces.append(cur)
        for j, p in enumerate(pieces):
            if len(p) >= 2:
                out.append(v.with_points(tuple(p), "" if len(pieces) == 1 else f"_{j}"))
    return tuple(out)


def _liang_barsky(
    a: tuple[float, float, float], b: tuple[float, float, float], cx: float, cy: float, radius: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """线段 a→b 与方形窗口的交集段;无交集返回 None(含两端点均窗外的穿角情形)。"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, a[0] - (cx - radius)),
        (dx, cx + radius - a[0]),
        (-dy, a[1] - (cy - radius)),
        (dy, cy + radius - a[1]),
    ):
        if p == 0.0:
            if q < 0.0:
                return None  # 平行且在窗外
        else:
            t = q / p
            if p < 0.0:
                if t > t1:
                    return None
                t0 = max(t0, t)
            else:
                if t < t0:
                    return None
                t1 = min(t1, t)
    p0 = (a[0] + dx * t0, a[1] + dy * t0, a[2] + (b[2] - a[2]) * t0)
    p1 = (a[0] + dx * t1, a[1] + dy * t1, a[2] + (b[2] - a[2]) * t1)
    return p0, p1


def _near(p: tuple[float, float, float], q: tuple[float, float, float]) -> bool:
    return abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6


# ---------- A4:坐标变换与 JSON 往返(纯值,落盘/A5 转换器共用) ----------


def flip_y(v: MapVec) -> MapVec:
    """xodr 系 → CARLA 世界系:y 取反(Unreal 左手系;A6 oracle 实测 0.00cm 定案)。"""
    return v.with_points(tuple((x, -y, z) for x, y, z in v.points))


def to_carla(vecs: tuple[MapVec, ...]) -> tuple[MapVec, ...]:
    """整组折线从 xodr 系翻到 CARLA 世界系(幂等:再翻一次回原值)。"""
    return tuple(flip_y(v) for v in vecs)


def vec_to_dict(v: MapVec) -> dict:
    """实例 → JSON dict;attrs 保持有序列表(允许重复键,如 validity 多车道对)。"""
    return {
        "cls": v.cls,
        "pts": [[round(x, 3), round(y, 3), round(z, 3)] for x, y, z in v.points],
        "attrs": [list(a) for a in v.attrs],
        "id": v.id,
        "src": v.src,
    }


def vec_from_dict(d: dict) -> MapVec:
    return MapVec(
        d["cls"],
        tuple((float(p[0]), float(p[1]), float(p[2])) for p in d["pts"]),
        tuple((str(k), str(v)) for k, v in d["attrs"]),
        d.get("id", ""),
        d.get("src", ""),
    )


def vecs_dump(vecs: tuple[MapVec, ...], map_name: str, frame: str = "carla_world") -> str:
    """整图矢量 → JSON 文本(顶层 meta + vecs 列表;落盘 mm 精度)。"""
    import json

    return json.dumps(
        {"map": map_name, "frame": frame, "vecs": [vec_to_dict(v) for v in vecs]},
        ensure_ascii=False,
        indent=1,
    )


def vecs_load(text: str) -> tuple[str, str, tuple[MapVec, ...]]:
    """JSON 文本 → (map_name, frame, vecs);与 vecs_dump 严格往返。"""
    import json

    d = json.loads(text)
    return d["map"], d["frame"], tuple(vec_from_dict(v) for v in d["vecs"])

"""OpenDRIVE 1.4 解析(纯值,不 import carla)。

地图矢量管道 §5.11 的离线数据源:把 CARLA 官方地图 .xodr 解析成道路几何
(planView/elevation/lanes/roadMark/object/junction/signal),核心 = s-t
车道坐标 → 世界 xyz 的 `road_to_xy`。与运行时 `map.get_waypoint_xodr`
的对账在 §5.11-A6 做,本模块只忠实还原 xodr 文件原坐标系。

实测口径(CARLA 0.9.16 全 17 图,2026-09-10 摸底):
- 几何类型只有 line 54460 / arc 33841 / spiral 262(仅 Town15);
  poly3 / paramPoly3 无实际用例,仍实现(闭式多项式,防御性)
- 车道 id 惯例 = **左正右负**(与 OpenDRIVE 标准相反,与 CARLA 运行时
  lane_id 同向);t 正 = 行驶向左侧,hdg 逆时针为正
- laneOffset 全 0(仍解析);roadMark 的 width 属性可缺省(记 0.0)
- superelevation(lateralProfile)不解析:矢量 GT 用 z 只看 elevationProfile
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

_EPS = 1e-9

# 各 kind 的 params 顺序
# line: ()                       arc: (curvature,)
# spiral: (curv_start, curv_end)
# poly3: (a, b, c, d)            paramPoly3: (aU, bU, cU, dU, aV, bV, cV, dV)


@dataclass(frozen=True)
class Geometry:
    """planView 中的一段几何;s = 起点桩号(road 系,自 road 起点)。"""

    kind: str  # line | arc | spiral | poly3 | paramPoly3
    s: float
    x: float
    y: float
    hdg: float
    length: float
    params: tuple[float, ...]
    p_range: str = "arcLength"  # paramPoly3 专用:arcLength | normalized


@dataclass(frozen=True)
class Elevation:
    s: float
    a: float
    b: float
    c: float
    d: float


@dataclass(frozen=True)
class LaneOffset:
    """车道整体横向偏移(多项式;CARLA 图全 0)。"""

    s: float
    a: float
    b: float
    c: float
    d: float


@dataclass(frozen=True)
class WidthRec:
    """车道宽度段;s_offset 相对所属 laneSection.s。"""

    s_offset: float
    a: float
    b: float
    c: float
    d: float


@dataclass(frozen=True)
class MarkRec:
    """车道标线段;s_offset 相对所属 laneSection.s,生效到下一段起点。"""

    s_offset: float
    type: str
    color: str
    width: float  # 缺省 0.0


@dataclass(frozen=True)
class Lane:
    id: int  # 左正右负(CARLA 惯例)
    type: str
    level: bool
    widths: tuple[WidthRec, ...]
    marks: tuple[MarkRec, ...]
    link: tuple[int, int] | None  # (predecessor_id, successor_id),无 <link> 为 None


@dataclass(frozen=True)
class LaneSection:
    s: float
    left: tuple[Lane, ...]
    center: tuple[Lane, ...]
    right: tuple[Lane, ...]


@dataclass(frozen=True)
class ObjectRec:
    """road 内对象(crosswalk / StopLine / 路面花纹等)。"""

    id: str
    name: str
    type: str
    s: float
    t: float
    z_offset: float
    hdg: float
    width: float
    length: float
    outline: tuple[tuple[float, float, float], ...]  # cornerLocal (u, v, z) 局部系


@dataclass(frozen=True)
class SignalRec:
    id: str
    name: str
    type: str
    subtype: str
    dynamic: str
    s: float
    t: float
    z_offset: float
    hdg: float
    validity: tuple[tuple[int, int], ...]  # (fromLane, toLane)


@dataclass(frozen=True)
class Road:
    id: int
    junction: int  # -1 = 普通道路
    name: str
    length: float
    geometries: tuple[Geometry, ...]  # 按 s 升序
    elevations: tuple[Elevation, ...]
    lane_offsets: tuple[LaneOffset, ...]
    lane_sections: tuple[LaneSection, ...]
    objects: tuple[ObjectRec, ...]
    signals: tuple[SignalRec, ...]
    predecessor: tuple[int, str] | None  # (element_id, contact_point),仅 road
    successor: tuple[int, str] | None


@dataclass(frozen=True)
class Junction:
    id: int
    name: str
    connections: tuple[tuple[int, int, str], ...]  # (incoming, connecting, contact_point)
    lane_links: tuple[tuple[int, int], ...]  # 全部 connection 的 (from, to) 并集


@dataclass(frozen=True)
class OpenDriveMap:
    roads: dict[int, Road]
    junctions: dict[int, Junction]


def _polyval(coeffs: tuple[float, ...], ds: float) -> float:
    a, b, c, d = coeffs
    return a + ds * (b + ds * (c + ds * d))


def _spiral_local(c0: float, c1: float, length: float, ds: float) -> tuple[float, float, float]:
    """spiral 局部系(未旋转):heading(s') = c0*s' + (c1-c0)*s'^2/(2L),Simpson 积分。"""
    if ds <= _EPS:
        return 0.0, 0.0, 0.0
    n = 128
    ss = np.linspace(0.0, ds, n + 1)
    th = c0 * ss + (c1 - c0) * ss * ss / (2.0 * length)
    w = np.full(n + 1, 4.0 / 3.0)
    w[::2] = 2.0 / 3.0
    w[0] = w[-1] = 1.0 / 3.0
    u = float(np.sum(w * np.cos(th))) * ds / n
    v = float(np.sum(w * np.sin(th))) * ds / n
    dh = c0 * ds + (c1 - c0) * ds * ds / (2.0 * length)
    return u, v, dh


def geo_local(geo: Geometry, ds: float) -> tuple[float, float, float]:
    """几何局部系结果 (u, v, dh):s 轴 = +x、t 轴 = +y(左),未按 hdg 旋转。"""
    kind = geo.kind
    if kind == "line":
        return ds, 0.0, 0.0
    if kind == "arc":
        c = geo.params[0]
        if abs(c) < _EPS:
            return ds, 0.0, 0.0
        return math.sin(c * ds) / c, (1.0 - math.cos(c * ds)) / c, c * ds
    if kind == "spiral":
        c0, c1 = geo.params
        return _spiral_local(c0, c1, geo.length, ds)
    if kind == "poly3":
        a, b, c, d = geo.params
        v = _polyval((a, b, c, d), ds)
        dh = math.atan2(b + 2.0 * c * ds + 3.0 * d * ds * ds, 1.0)
        return ds, v, dh
    if kind == "paramPoly3":
        aU, bU, cU, dU, aV, bV, cV, dV = geo.params
        p = ds if geo.p_range == "arcLength" else ds / geo.length
        u = _polyval((aU, bU, cU, dU), p)
        v = _polyval((aV, bV, cV, dV), p)
        du = bU + 2.0 * cU * p + 3.0 * dU * p * p
        dv = bV + 2.0 * cV * p + 3.0 * dV * p * p
        return u, v, math.atan2(dv, du)
    raise ValueError(f"未知几何类型 {kind!r}")


def _geo_at(road: Road, s: float) -> tuple[Geometry, float]:
    """s 所属几何与段内偏移;越界 raise(调用方采样应保持在 [0, length] 内)。"""
    if s < -_EPS or s > road.length + _EPS:
        raise ValueError(f"road {road.id}: s={s} 超出 [0, {road.length}]")
    geo = road.geometries[0]
    for g in road.geometries:
        if g.s <= s + _EPS:
            geo = g
        else:
            break
    return geo, max(s - geo.s, 0.0)


def road_xy(road: Road, s: float) -> tuple[float, float]:
    """planView 平面坐标 (x, y)。"""
    geo, ds = _geo_at(road, s)
    u, v, _ = geo_local(geo, ds)
    ch, sh = math.cos(geo.hdg), math.sin(geo.hdg)
    return geo.x + u * ch - v * sh, geo.y + u * sh + v * ch


def road_heading(road: Road, s: float) -> float:
    geo, ds = _geo_at(road, s)
    _, _, dh = geo_local(geo, ds)
    return geo.hdg + dh


def road_z(road: Road, s: float) -> float:
    """elevationProfile 高程(不含 superelevation 倾角)。"""
    _geo_at(road, s)
    elev = road.elevations[0]
    for e in road.elevations:
        if e.s <= s + _EPS:
            elev = e
        else:
            break
    return _polyval((elev.a, elev.b, elev.c, elev.d), s - elev.s)


def road_to_xy(road: Road, s: float, t: float) -> tuple[float, float, float]:
    """s-t → 世界 (x, y, z);t 正 = 行驶向左侧,可外推。"""
    x, y = road_xy(road, s)
    h = road_heading(road, s)
    return x - t * math.sin(h), y + t * math.cos(h), road_z(road, s)


def lane_offset_at(road: Road, s: float) -> float:
    if not road.lane_offsets:
        return 0.0
    off = road.lane_offsets[0]
    for o in road.lane_offsets:
        if o.s <= s + _EPS:
            off = o
        else:
            break
    return _polyval((off.a, off.b, off.c, off.d), s - off.s)


def _section_at(road: Road, s: float) -> tuple[LaneSection, float]:
    if s < -_EPS or s > road.length + _EPS:
        raise ValueError(f"road {road.id}: s={s} 超出 [0, {road.length}]")
    sec = road.lane_sections[0]
    for sc in road.lane_sections:
        if sc.s <= s + _EPS:
            sec = sc
        else:
            break
    return sec, s - sec.s


def _lane_in(sec: LaneSection, lane_id: int) -> Lane | None:
    for lane in sec.left + sec.center + sec.right:
        if lane.id == lane_id:
            return lane
    return None


def lane_width_at(road: Road, s: float, lane_id: int) -> float:
    """lane_id 在 s 处的宽度(含 width 多项式)。"""
    sec, ds = _section_at(road, s)
    lane = _lane_in(sec, lane_id)
    if lane is None:
        raise ValueError(f"road {road.id} s={s}: lane {lane_id} 不存在")
    rec = lane.widths[0]
    for w in lane.widths:
        if w.s_offset <= ds + _EPS:
            rec = w
        else:
            break
    return _polyval((rec.a, rec.b, rec.c, rec.d), ds - rec.s_offset)


def lane_boundary_t(road: Road, s: float, lane_id: int) -> tuple[float, float]:
    """lane 的横向边界 (t_lo, t_hi),含 laneOffset;t 正 = 左。

    惯例:左正右负;t_lo < t_hi 恒成立(左车道 t_lo=内沿,右车道 t_hi=内沿)。
    """
    sec, _ = _section_at(road, s)
    if _lane_in(sec, lane_id) is None:
        raise ValueError(f"road {road.id} s={s}: lane {lane_id} 不存在")
    offset = lane_offset_at(road, s)
    side = 1 if lane_id > 0 else -1
    t_inner = offset
    for lane in sorted((l for l in sec.left if l.id > 0), key=lambda l: l.id):
        w = lane_width_at(road, s, lane.id)
        if side == 1 and lane.id == lane_id:
            return t_inner, t_inner + w
        t_inner += w
    t_inner = offset
    for lane in sorted((l for l in sec.right if l.id < 0), key=lambda l: -l.id):
        w = lane_width_at(road, s, lane.id)
        if side == -1 and lane.id == lane_id:
            return t_inner - w, t_inner
        t_inner -= w
    raise ValueError(f"road {road.id} s={s}: lane {lane_id} 不在行车侧")  # center 车道无边界


def lane_centerline_t(road: Road, s: float, lane_id: int) -> float:
    lo, hi = lane_boundary_t(road, s, lane_id)
    return (lo + hi) / 2.0


def mark_at(road: Road, s: float, lane_id: int) -> MarkRec | None:
    """s 处生效的标线段(按 sOffset 分段);该 lane 无标线记录返回 None。"""
    sec, ds = _section_at(road, s)
    lane = _lane_in(sec, lane_id)
    if lane is None or not lane.marks:
        return None
    rec = lane.marks[0]
    for m in lane.marks:
        if m.s_offset <= ds + _EPS:
            rec = m
        else:
            break
    return rec


# ---------- XML 解析 ----------


def _f(el: ET.Element, name: str, default: float = 0.0) -> float:
    return float(el.get(name, default))


def _parse_geometry(el: ET.Element) -> Geometry:
    kind = "line"  # 缺省:空 <geometry/> 视为直线
    params: tuple[float, ...] = ()
    p_range = "arcLength"
    if (child := el.find("arc")) is not None:
        kind = "arc"
        params = (_f(child, "curvature"),)
    elif (child := el.find("spiral")) is not None:
        kind = "spiral"
        params = (_f(child, "curvStart"), _f(child, "curvEnd"))
    elif (child := el.find("poly3")) is not None:
        kind = "poly3"
        params = (_f(child, "a"), _f(child, "b"), _f(child, "c"), _f(child, "d"))
    elif (child := el.find("paramPoly3")) is not None:
        kind = "paramPoly3"
        params = tuple(_f(child, k) for k in ("aU", "bU", "cU", "dU", "aV", "bV", "cV", "dV"))
        p_range = child.get("pRange", "arcLength")
    return Geometry(
        kind=kind,
        s=_f(el, "s"),
        x=_f(el, "x"),
        y=_f(el, "y"),
        hdg=_f(el, "hdg"),
        length=_f(el, "length"),
        params=params,
        p_range=p_range,
    )


def _parse_width(el: ET.Element) -> WidthRec:
    return WidthRec(_f(el, "sOffset"), _f(el, "a"), _f(el, "b"), _f(el, "c"), _f(el, "d"))


def _parse_mark(el: ET.Element) -> MarkRec:
    return MarkRec(
        s_offset=_f(el, "sOffset"),
        type=el.get("type", "none"),
        color=el.get("color", ""),
        width=_f(el, "width"),
    )


def _parse_side(sec: ET.Element, tag: str) -> tuple[Lane, ...]:
    node = sec.find(tag)
    return tuple(_parse_lane(l) for l in node) if node is not None else ()


def _parse_lane(el: ET.Element) -> Lane:
    link = None
    lk = el.find("link")
    if lk is not None:
        p, s2 = lk.find("predecessor"), lk.find("successor")
        if p is not None and s2 is not None:
            link = (int(p.get("id", 0)), int(s2.get("id", 0)))
    widths = tuple(_parse_width(w) for w in el.findall("width"))
    marks = tuple(_parse_mark(m) for m in el.findall("roadMark"))
    return Lane(
        id=int(el.get("id", 0)),
        type=el.get("type", "none"),
        level=el.get("level", "false") == "true",
        widths=tuple(sorted(widths, key=lambda w: w.s_offset)),
        marks=tuple(sorted(marks, key=lambda m: m.s_offset)),
        link=link,
    )


def _parse_object(el: ET.Element) -> ObjectRec:
    outline: tuple[tuple[float, float, float], ...] = ()
    ol = el.find("outline")
    if ol is not None:
        outline = tuple((_f(c, "u"), _f(c, "v"), _f(c, "z")) for c in ol.findall("cornerLocal"))
    return ObjectRec(
        id=el.get("id", ""),
        name=el.get("name", ""),
        type=el.get("type", ""),
        s=_f(el, "s"),
        t=_f(el, "t"),
        z_offset=_f(el, "zOffset"),
        hdg=_f(el, "hdg"),
        width=_f(el, "width"),
        length=_f(el, "length"),
        outline=outline,
    )


def _parse_signal(el: ET.Element) -> SignalRec:
    validity = tuple((int(v.get("fromLane", 0)), int(v.get("toLane", 0))) for v in el.findall("validity"))
    return SignalRec(
        id=el.get("id", ""),
        name=el.get("name", ""),
        type=el.get("type", ""),
        subtype=el.get("subtype", ""),
        dynamic=el.get("dynamic", "no"),
        s=_f(el, "s"),
        t=_f(el, "t"),
        z_offset=_f(el, "zOffset"),
        hdg=_f(el, "hdg"),
        validity=validity,
    )


def _parse_road(el: ET.Element) -> Road:
    link = el.find("link")
    pre = suc = None
    if link is not None:
        p, s2 = link.find("predecessor"), link.find("successor")
        if p is not None and p.get("elementType", "road") == "road":
            pre = (int(p.get("elementId", 0)), p.get("contactPoint", ""))
        if s2 is not None and s2.get("elementType", "road") == "road":
            suc = (int(s2.get("elementId", 0)), s2.get("contactPoint", ""))
    pv = el.find("planView")
    geometries = tuple(sorted((_parse_geometry(g) for g in pv), key=lambda g: g.s)) if pv is not None else ()
    ep = el.find("elevationProfile")
    elevations = (
        tuple(
            sorted(
                (Elevation(_f(e, "s"), _f(e, "a"), _f(e, "b"), _f(e, "c"), _f(e, "d")) for e in ep),
                key=lambda e: e.s,
            )
        )
        if ep is not None
        else ()
    )
    lanes_el = el.find("lanes")
    lane_offsets = lane_sections = ()
    if lanes_el is not None:
        lane_offsets = tuple(
            sorted(
                (
                    LaneOffset(_f(o, "s"), _f(o, "a"), _f(o, "b"), _f(o, "c"), _f(o, "d"))
                    for o in lanes_el.findall("laneOffset")
                ),
                key=lambda o: o.s,
            )
        )
        sections: list[LaneSection] = []
        for sec in lanes_el.findall("laneSection"):
            sections.append(
                LaneSection(
                    _f(sec, "s"),
                    _parse_side(sec, "left"),
                    _parse_side(sec, "center"),
                    _parse_side(sec, "right"),
                )
            )
        lane_sections = tuple(sorted(sections, key=lambda sc: sc.s))
    objs = tuple(_parse_object(o) for o in el.findall("objects/object") + el.findall("object"))
    sigs = tuple(_parse_signal(s) for s in el.findall("signals/signal") + el.findall("signal"))
    return Road(
        id=int(el.get("id", 0)),
        junction=int(el.get("junction", "-1")),
        name=el.get("name", ""),
        length=_f(el, "length"),
        geometries=geometries,
        elevations=elevations,
        lane_offsets=lane_offsets,
        lane_sections=lane_sections,
        objects=objs,
        signals=sigs,
        predecessor=pre,
        successor=suc,
    )


def _parse_junction(el: ET.Element) -> Junction:
    connections: list[tuple[int, int, str]] = []
    lane_links: list[tuple[int, int]] = []
    for c in el.findall("connection"):
        connections.append(
            (int(c.get("incomingRoad", 0)), int(c.get("connectingRoad", 0)), c.get("contactPoint", ""))
        )
        lane_links.extend((int(l.get("from", 0)), int(l.get("to", 0))) for l in c.findall("laneLink"))
    return Junction(
        id=int(el.get("id", 0)),
        name=el.get("name", ""),
        connections=tuple(connections),
        lane_links=tuple(lane_links),
    )


def _parse_root(root: ET.Element) -> OpenDriveMap:
    roads = {r.id: r for el in root.findall("road") if (r := _parse_road(el)) is not None}
    junctions = {j.id: j for el in root.findall("junction") if (j := _parse_junction(el)) is not None}
    return OpenDriveMap(roads=roads, junctions=junctions)


def parse_xodr(path: str) -> OpenDriveMap:
    """解析 .xodr 文件为 OpenDriveMap(ElementTree,整文件一次读入)。"""
    return _parse_root(ET.parse(path).getroot())


def parse_xodr_text(text: str) -> OpenDriveMap:
    """解析 .xodr 文本(测试/内嵌场景用)。"""
    return _parse_root(ET.fromstring(text))

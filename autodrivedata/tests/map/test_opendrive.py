"""opendrive.py 解析器单测(纯值,base env)。

闭式解手算锚定 line/arc/poly3/paramPoly3/螺旋退化;真实文件测试用
CARLA 官方图做计数锚点(grep 复核过),机器上无 CARLA 时自动 skip。
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import pytest

from autodrivedata.map.opendrive import (
    Road,
    lane_boundary_t,
    lane_centerline_t,
    lane_width_at,
    mark_at,
    parse_xodr,
    parse_xodr_text,
    road_heading,
    road_to_xy,
    road_xy,
    road_z,
)

CARLA_MAPS = "/root/autodl-tmp/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps"

# ---------- 构建器 ----------


def xodr_doc(roads_xml: str, junctions_xml: str = "") -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n<OpenDRIVE>\n{roads_xml}\n{junctions_xml}\n</OpenDRIVE>\n'
    )


def one_road(
    plan_view: str,
    lanes: str = "",
    elevation: str = '<elevation s="0" a="0" b="0" c="0" d="0"/>',
    extra: str = "",
    length: float = 100.0,
) -> str:
    return (
        f'<road name="" id="1" junction="-1" length="{length}">'
        f"<planView>{plan_view}</planView>"
        f"<elevationProfile>{elevation}</elevationProfile>"
        f"<lanes>{lanes}</lanes>{extra}</road>"
    )


def parse_one(road_xml: str) -> Road:
    return parse_xodr_text(xodr_doc(road_xml)).roads[1]


def geo(
    kind_xml: str, s: float = 0.0, x: float = 0.0, y: float = 0.0, hdg: float = 0.0, length: float = 100.0
) -> str:
    return f'<geometry s="{s}" x="{x}" y="{y}" hdg="{hdg}" length="{length}">{kind_xml}</geometry>'


def lane_xml(lid: int, widths: str, marks: str = "", type_: str = "driving") -> str:
    return f'<lane id="{lid}" type="{type_}" level="false">{widths}{marks}</lane>'


def width_xml(s_offset: float, a: float, b: float = 0.0, c: float = 0.0, d: float = 0.0) -> str:
    return f'<width sOffset="{s_offset}" a="{a}" b="{b}" c="{c}" d="{d}"/>'


def lanes_xml(right: str = "", left: str = "", center: str = "", offset: str = "") -> str:
    return f'{offset}<laneSection s="0"><left>{left}</left><center>{center}</center><right>{right}</right></laneSection>'


# ---------- 几何闭式解 ----------


class TestLine:
    def test_axis_aligned(self) -> None:
        road = parse_one(one_road(geo("<line/>")))
        assert road_to_xy(road, 5.0, 2.0) == pytest.approx((5.0, 2.0, 0.0))
        assert road_heading(road, 5.0) == pytest.approx(0.0)

    def test_rotated(self) -> None:
        road = parse_one(one_road(geo("<line/>", hdg=math.pi / 2)))
        assert road_to_xy(road, 3.0, 0.0) == pytest.approx((0.0, 3.0, 0.0))
        assert road_to_xy(road, 3.0, 1.0) == pytest.approx((-1.0, 3.0, 0.0))  # t 正 = 左

    def test_lateral_extrapolation(self) -> None:
        road = parse_one(one_road(geo("<line/>", length=10.0)))
        assert road_to_xy(road, 5.0, -7.0) == pytest.approx((5.0, -7.0, 0.0))


class TestArc:
    def test_left_turn_closed_form(self) -> None:
        road = parse_one(one_road(geo('<arc curvature="1"/>', length=10.0)))
        assert road_to_xy(road, math.pi / 2, 0.0) == pytest.approx((1.0, 1.0, 0.0))
        assert road_heading(road, math.pi / 2) == pytest.approx(math.pi / 2)

    def test_right_turn_closed_form(self) -> None:
        road = parse_one(one_road(geo('<arc curvature="-1"/>', length=10.0)))
        assert road_to_xy(road, math.pi / 2, 0.0) == pytest.approx((1.0, -1.0, 0.0))
        assert road_heading(road, math.pi / 2) == pytest.approx(-math.pi / 2)

    def test_rotated_and_offset(self) -> None:
        # 独立公式重算:u = sin(c ds)/c,v = (1-cos(c ds))/c,再绕 hdg 旋转
        hdg, x0, y0, c, ds = 0.7, 10.0, -3.0, 0.4, 2.5
        road = parse_one(one_road(geo(f'<arc curvature="{c}"/>', x=x0, y=y0, hdg=hdg, length=10.0)))
        u, v = math.sin(c * ds) / c, (1.0 - math.cos(c * ds)) / c
        want = (x0 + u * math.cos(hdg) - v * math.sin(hdg), y0 + u * math.sin(hdg) + v * math.cos(hdg))
        assert road_xy(road, ds) == pytest.approx(want)
        assert road_heading(road, ds) == pytest.approx(hdg + c * ds)

    def test_zero_curvature_is_line(self) -> None:
        road = parse_one(one_road(geo('<arc curvature="0"/>')))
        assert road_to_xy(road, 5.0, 0.0) == pytest.approx((5.0, 0.0, 0.0))


class TestSpiral:
    def test_constant_curvature_equals_arc(self) -> None:
        spiral = parse_one(one_road(geo('<spiral curvStart="0.5" curvEnd="0.5"/>', length=10.0)))
        arc = parse_one(one_road(geo('<arc curvature="0.5"/>', length=10.0)))
        for s in (0.0, 1.7, 4.3, 9.9):
            assert road_xy(spiral, s) == pytest.approx(road_xy(arc, s), abs=1e-6)
            assert road_heading(spiral, s) == pytest.approx(road_heading(arc, s), abs=1e-6)

    def test_zero_curvature_is_line(self) -> None:
        road = parse_one(one_road(geo('<spiral curvStart="0" curvEnd="0"/>')))
        assert road_to_xy(road, 5.0, 0.0) == pytest.approx((5.0, 0.0, 0.0))

    def test_heading_closed_form_and_turn_direction(self) -> None:
        # heading(s) = c0 s + (c1-c0) s^2/(2L):c0=0, c1=0.1, L=10 → heading(10)=0.5
        road = parse_one(one_road(geo('<spiral curvStart="0" curvEnd="0.1"/>', length=10.0)))
        assert road_heading(road, 10.0) == pytest.approx(0.5)
        x, y, _ = road_to_xy(road, 10.0, 0.0)
        assert x > 0 and y > 0  # 左转:终态在起点右前上方

    def test_matches_fine_integration(self) -> None:
        # 独立细网格积分(Simpson, N=4000)对照
        c0, c1, length, s = 0.02, 0.12, 30.0, 17.3
        road = parse_one(one_road(geo(f'<spiral curvStart="{c0}" curvEnd="{c1}"/>', length=length)))
        n = 4000
        ss = np.linspace(0.0, s, n + 1)
        th = c0 * ss + (c1 - c0) * ss * ss / (2.0 * length)
        w = np.full(n + 1, 4.0 / 3.0)
        w[::2] = 2.0 / 3.0
        w[0] = w[-1] = 1.0 / 3.0
        u = float(np.sum(w * np.cos(th))) * s / n
        v = float(np.sum(w * np.sin(th))) * s / n
        assert road_xy(road, s) == pytest.approx((u, v), abs=1e-4)


class TestPolynomial:
    def test_poly3(self) -> None:
        # v(u) = u^2:u=s → (2,4),heading = atan(2u)
        road = parse_one(one_road(geo('<poly3 a="0" b="0" c="1" d="0"/>', length=10.0)))
        assert road_xy(road, 2.0) == pytest.approx((2.0, 4.0))
        assert road_heading(road, 2.0) == pytest.approx(math.atan(4.0))

    def test_param_poly3_arc_length(self) -> None:
        # u = p = s,v = 0 → 直线
        k = '<paramPoly3 aU="0" bU="1" cU="0" dU="0" aV="0" bV="0" cV="0" dV="0" pRange="arcLength"/>'
        road = parse_one(one_road(geo(k, length=10.0)))
        assert road_to_xy(road, 4.0, 0.0) == pytest.approx((4.0, 0.0, 0.0))
        assert road_heading(road, 4.0) == pytest.approx(0.0)


class TestElevation:
    def test_poly(self) -> None:
        road = parse_one(
            one_road(geo("<line/>"), elevation='<elevation s="0" a="10" b="2" c="0.5" d="0.25"/>')
        )
        assert road_z(road, 2.0) == pytest.approx(10 + 4 + 2 + 2)  # a + b s + c s^2 + d s^3

    def test_segments(self) -> None:
        road = parse_one(
            one_road(
                geo("<line/>"),
                elevation='<elevation s="0" a="0" b="0" c="0" d="0"/><elevation s="5" a="30" b="1" c="0" d="0"/>',
            )
        )
        assert road_z(road, 3.0) == pytest.approx(0.0)
        assert road_z(road, 7.0) == pytest.approx(32.0)


# ---------- 车道 / 标线 ----------


class TestLaneGeometry:
    def test_right_lanes_negative_t(self) -> None:
        lanes = lanes_xml(right=lane_xml(-1, width_xml(0, 3.5)) + lane_xml(-2, width_xml(0, 3.0)))
        road = parse_one(one_road(geo("<line/>"), lanes=lanes))
        assert lane_boundary_t(road, 5.0, -1) == pytest.approx((-3.5, 0.0))
        assert lane_boundary_t(road, 5.0, -2) == pytest.approx((-6.5, -3.5))
        assert lane_centerline_t(road, 5.0, -2) == pytest.approx(-5.0)

    def test_left_lanes_positive_t(self) -> None:
        lanes = lanes_xml(left=lane_xml(1, width_xml(0, 3.0)))
        road = parse_one(one_road(geo("<line/>"), lanes=lanes))
        assert lane_boundary_t(road, 5.0, 1) == pytest.approx((0.0, 3.0))
        assert lane_centerline_t(road, 5.0, 1) == pytest.approx(1.5)

    def test_width_poly(self) -> None:
        # lane -1 宽度 3 + 0.5 s:s=2 → 4.0
        lanes = lanes_xml(right=lane_xml(-1, width_xml(0, 3.0, b=0.5)))
        road = parse_one(one_road(geo("<line/>"), lanes=lanes))
        assert lane_width_at(road, 2.0, -1) == pytest.approx(4.0)
        assert lane_boundary_t(road, 2.0, -1) == pytest.approx((-4.0, 0.0))

    def test_lane_offset_shift(self) -> None:
        lanes = lanes_xml(
            right=lane_xml(-1, width_xml(0, 3.0)), offset='<laneOffset s="0" a="1" b="0" c="0" d="0"/>'
        )
        road = parse_one(one_road(geo("<line/>"), lanes=lanes))
        assert lane_boundary_t(road, 5.0, -1) == pytest.approx((-2.0, 1.0))

    def test_center_lane_has_no_boundary(self) -> None:
        lanes = lanes_xml(center=lane_xml(0, width_xml(0, 0.0), type_="none"))
        road = parse_one(one_road(geo("<line/>"), lanes=lanes))
        with pytest.raises(ValueError):
            lane_boundary_t(road, 5.0, 0)

    def test_unknown_lane_raises(self) -> None:
        lanes = lanes_xml(right=lane_xml(-1, width_xml(0, 3.5)))
        road = parse_one(one_road(geo("<line/>"), lanes=lanes))
        with pytest.raises(ValueError):
            lane_width_at(road, 5.0, -9)


class TestMarkAt:
    def test_segment_lookup_and_persist(self) -> None:
        marks = '<roadMark sOffset="0" type="solid" color="white" width="0.12"/>'
        marks += '<roadMark sOffset="10" type="broken" color="white" width="0.12"/>'
        lanes = lanes_xml(right=lane_xml(-1, width_xml(0, 3.5), marks=marks))
        road = parse_one(one_road(geo("<line/>"), lanes=lanes))
        assert mark_at(road, 5.0, -1).type == "solid"
        assert mark_at(road, 10.0, -1).type == "broken"
        assert mark_at(road, 50.0, -1).type == "broken"  # 末段延续

    def test_no_marks_returns_none(self) -> None:
        lanes = lanes_xml(right=lane_xml(-1, width_xml(0, 3.5)))
        road = parse_one(one_road(geo("<line/>"), lanes=lanes))
        assert mark_at(road, 5.0, -1) is None

    def test_missing_width_attr_defaults_zero(self) -> None:
        lanes = lanes_xml(right=lane_xml(-1, width_xml(0, 3.5), marks='<roadMark sOffset="0" type="none"/>'))
        road = parse_one(one_road(geo("<line/>"), lanes=lanes))
        assert mark_at(road, 5.0, -1).width == 0.0


# ---------- 道路结构与边界 ----------


class TestRoadStructure:
    def test_geometry_continuity_at_boundary(self) -> None:
        hdg = 0.3
        x1, y1 = 1.0 + 5.0 * math.cos(hdg), 2.0 + 5.0 * math.sin(hdg)
        pv = geo("<line/>", x=1.0, y=2.0, hdg=hdg, length=5.0) + geo(
            '<arc curvature="0.2"/>', s=5.0, x=x1, y=y1, hdg=hdg, length=10.0
        )
        road = parse_one(one_road(pv, length=15.0))
        # s=5 处直线段公式与弧段起点一致,heading 连续
        assert road_xy(road, 5.0) == pytest.approx((x1, y1))
        assert road_heading(road, 5.0) == pytest.approx(hdg)

    def test_s_out_of_range_raises(self) -> None:
        road = parse_one(one_road(geo("<line/>", length=10.0), length=10.0))
        with pytest.raises(ValueError):
            road_xy(road, -0.01)
        with pytest.raises(ValueError):
            road_xy(road, 10.01)

    def test_road_link(self) -> None:
        link = '<link><predecessor elementType="road" elementId="3" contactPoint="end"/>'
        link += '<successor elementType="road" elementId="10" contactPoint="start"/></link>'
        road = parse_one(one_road(geo("<line/>"), extra=link))
        assert road.predecessor == (3, "end")
        assert road.successor == (10, "start")

    def test_junction_link_is_none(self) -> None:
        link = '<link><predecessor elementType="junction" elementId="7"/></link>'
        road = parse_one(one_road(geo("<line/>"), extra=link))
        assert road.predecessor is None


class TestObjectsSignalsJunctions:
    def test_crosswalk_object_with_outline(self) -> None:
        obj = (
            '<objects><object id="1140" name="ContinentalCrosswalk" s="4.566" t="-0.0949" zOffset="0"'
            ' hdg="1.583" width="2.921" length="19.405" type="crosswalk">'
            '<outline><cornerLocal u="9.1" v="1.46" z="0"/>'
            '<cornerLocal u="9.7" v="-1.03" z="0"/>'
            '<cornerLocal u="-9.7" v="-1.46" z="0"/>'
            '<cornerLocal u="-9.0" v="1.06" z="0"/>'
            '<cornerLocal u="9.1" v="1.46" z="0"/></outline></object></objects>'
        )
        road = parse_one(one_road(geo("<line/>"), extra=obj))
        o = road.objects[0]
        assert o.type == "crosswalk" and o.name == "ContinentalCrosswalk"
        assert (o.s, o.t, o.hdg, o.width, o.length) == pytest.approx((4.566, -0.0949, 1.583, 2.921, 19.405))
        assert len(o.outline) == 5
        assert o.outline[0] == pytest.approx((9.1, 1.46, 0.0))

    def test_direct_child_object(self) -> None:
        # 无 <objects> 包裹的直接子元素
        obj = '<object id="1" name="StopLine" s="10" t="0.2" zOffset="0" hdg="0" width="0.3" length="8"/>'
        road = parse_one(one_road(geo("<line/>"), extra=obj))
        assert len(road.objects) == 1
        assert road.objects[0].name == "StopLine" and road.objects[0].outline == ()

    def test_signal_wrapped_with_validity(self) -> None:
        sig = (
            '<signals><signal id="944" name="Signal_3Light_Post01" s="8.06" t="1.19" zOffset="-0.445"'
            ' hdg="-2.34" dynamic="yes" type="1000001" subtype="-1">'
            '<validity fromLane="-2" toLane="-1"/><validity fromLane="-1" toLane="-1"/>'
            "</signal></signals>"
        )
        road = parse_one(one_road(geo("<line/>"), extra=sig))
        s = road.signals[0]
        assert s.id == "944" and s.dynamic == "yes"
        assert s.validity == ((-2, -1), (-1, -1))

    def test_junction_connections(self) -> None:
        jx = (
            '<junction id="5" name="J5">'
            '<connection id="0" incomingRoad="4" connectingRoad="6" contactPoint="start">'
            '<laneLink from="-1" to="-1"/><laneLink from="-2" to="-2"/></connection></junction>'
        )
        m = parse_xodr_text(xodr_doc(one_road(geo("<line/>")), jx))
        j = m.junctions[5]
        assert j.connections == ((4, 6, "start"),)
        assert j.lane_links == ((-1, -1), (-2, -2))


# ---------- 真实文件(无 CARLA 机器自动 skip)----------


HAS_CARLA = os.path.isdir(CARLA_MAPS)
pytestmark_real = pytest.mark.skipif(not HAS_CARLA, reason="本机无 CARLA 官方图")


def official_xodr_files() -> list[str]:
    return sorted(glob.glob(os.path.join(CARLA_MAPS, "OpenDrive", "*.xodr"))) + sorted(
        glob.glob(os.path.join(CARLA_MAPS, "*", "OpenDrive", "*.xodr"))
    )


class TestRealFiles:
    @pytestmark_real
    def test_town10hd_anchors(self) -> None:
        m = parse_xodr(os.path.join(CARLA_MAPS, "OpenDrive", "Town10HD_Opt.xodr"))
        assert len(m.roads) == 108
        assert len(m.junctions) == 9
        road0 = m.roads[0]
        assert len(road0.geometries) == 5
        g = road0.geometries[0]
        assert g.kind == "line"
        assert (g.x, g.y, g.hdg) == pytest.approx((104.68, 9.37, 1.5639764844735413))
        assert sum(len(r.objects) for r in m.roads.values()) == 60  # 16 crosswalk + 44 路面花纹
        assert sum(1 for r in m.roads.values() for o in r.objects if o.type == "crosswalk") == 16
        assert sum(len(r.signals) for r in m.roads.values()) == 21
        assert (
            sum(
                len(l.marks)
                for r in m.roads.values()
                for sc in r.lane_sections
                for l in sc.left + sc.center + sc.right
            )
            == 2802
        )
        kinds = [g.kind for r in m.roads.values() for g in r.geometries]
        assert kinds.count("line") == 458
        assert kinds.count("arc") == 149

    @pytestmark_real
    def test_town15_spiral_anchors(self) -> None:
        m = parse_xodr(os.path.join(CARLA_MAPS, "Town15", "OpenDrive", "Town15.xodr"))
        assert len(m.roads) == 723
        assert len(m.junctions) == 99
        kinds = [g.kind for r in m.roads.values() for g in r.geometries]
        assert kinds.count("spiral") == 262
        # 螺旋终点 heading 闭式:c0 L + (c1-c0) L/2
        for r in m.roads.values():
            for g in r.geometries:
                if g.kind == "spiral":
                    c0, c1 = g.params
                    want = g.hdg + c0 * g.length + (c1 - c0) * g.length / 2.0
                    assert road_heading(r, g.s + g.length) == pytest.approx(want, abs=1e-9)
                    assert all(math.isfinite(v) for v in road_xy(r, g.s + g.length / 2.0))
                    break
            else:
                continue
            break

    @pytestmark_real
    def test_all_official_maps_totals(self) -> None:
        totals: dict[str, int] = {}
        n_roads = 0
        for path in official_xodr_files():
            m = parse_xodr(path)
            assert len(m.roads) > 0, path
            n_roads += len(m.roads)
            for r in m.roads.values():
                assert r.geometries, f"{path} road {r.id} 无 planView"
                for g in r.geometries:
                    totals[g.kind] = totals.get(g.kind, 0) + 1
        assert totals == {"line": 54460, "arc": 33841, "spiral": 262}
        assert n_roads == 21900

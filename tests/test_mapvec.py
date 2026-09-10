"""mapvec.py 单测:构造 xodr 手算锚定 + 真实文件计数锚点 + 重采样/裁剪。"""

from __future__ import annotations

import glob

import pytest

from autodrivedata.mapvec import MapVec, crop_to_ego, extract_mapvec, resample
from autodrivedata.opendrive import parse_xodr_text

# 双向四车道 + sidewalk/curb + center 双黄线 + crosswalk + StopLine + signal
_XODR = """<?xml version="1.0"?>
<OpenDRIVE>
<header revMajor="1" revMinor="4" name="t"/>
<road name="r0" length="100" id="7" junction="-1">
  <link/>
  <planView><geometry s="0" x="0" y="0" hdg="0" length="100"><line/></geometry></planView>
  <elevationProfile><elevation s="0" a="0" b="0" c="0" d="0"/></elevationProfile>
  <lanes>
    <laneSection s="0">
      <left>
        <lane id="3" type="sidewalk" level="false"><width sOffset="0" a="2" b="0" c="0" d="0"/>
          <roadMark sOffset="0" type="curb" color="grey" width="0.2"/></lane>
        <lane id="2" type="driving" level="false"><width sOffset="0" a="3.5" b="0" c="0" d="0"/>
          <roadMark sOffset="0" type="solid" color="white" width="0.15"/></lane>
        <lane id="1" type="driving" level="false"><width sOffset="0" a="3.5" b="0" c="0" d="0"/>
          <roadMark sOffset="0" type="broken" color="white" width="0.15"/></lane>
      </left>
      <center>
        <lane id="0" type="none" level="false"><roadMark sOffset="0" type="solid solid" color="yellow" width="0.15"/></lane>
      </center>
      <right>
        <lane id="-1" type="driving" level="false"><width sOffset="0" a="3.5" b="0" c="0" d="0"/>
          <roadMark sOffset="0" type="broken" color="white" width="0.15"/></lane>
        <lane id="-2" type="driving" level="false"><width sOffset="0" a="3.5" b="0" c="0" d="0"/>
          <roadMark sOffset="0" type="solid" color="white" width="0.15"/></lane>
        <lane id="-3" type="sidewalk" level="false"><width sOffset="0" a="2" b="0" c="0" d="0"/>
          <roadMark sOffset="0" type="curb" color="grey" width="0.2"/></lane>
      </right>
    </laneSection>
  </lanes>
  <objects>
    <object id="1" name="cw" type="crosswalk" s="50" t="0" zOffset="0" hdg="0" width="4" length="10">
      <outline>
        <cornerLocal u="5" v="2" z="0"/><cornerLocal u="5" v="-2" z="0"/>
        <cornerLocal u="-5" v="-2" z="0"/><cornerLocal u="-5" v="2" z="0"/>
      </outline>
    </object>
    <object id="2" name="StopLine" type="none" s="20" t="0" zOffset="0" hdg="0" width="0" length="6">
      <outline>
        <cornerLocal u="3" v="0" z="0"/><cornerLocal u="-3" v="0" z="0"/>
      </outline>
    </object>
  </objects>
  <signals>
    <signal id="s1" name="TL1" type="traffic_light" subtype="regular" dynamic="yes" s="80" t="0" zOffset="5" hdg="0">
      <validity fromLane="1" toLane="-1"/>
    </signal>
  </signals>
</road>
</OpenDRIVE>
"""


@pytest.fixture(scope="module")
def vecs() -> tuple[MapVec, ...]:
    return extract_mapvec(parse_xodr_text(_XODR))


def by_cls(vecs: tuple[MapVec, ...], cls: str) -> list[MapVec]:
    return [v for v in vecs if v.cls == cls]


def test_divider_count_and_geometry(vecs: tuple[MapVec, ...]) -> None:
    dv = by_cls(vecs, "divider")
    # 同向 2 条 broken(lane1/lane-1 外缘)+ center 双黄线 1 条
    assert len(dv) == 3
    center = next(v for v in dv if dict(v.attrs).get("centerline") == "yes")
    assert dict(center.attrs)["color"] == "yellow"
    pts = center.points
    assert abs(pts[0][0] - 0.0) < 1e-6 and abs(pts[0][1]) < 1e-6  # t=0 起点
    assert abs(pts[-1][0] - 100.0) < 1e-6
    same_dir = [v for v in dv if dict(v.attrs).get("centerline") != "yes"]
    ys = sorted({round(p[1], 1) for v in same_dir for p in v.points})
    assert ys == [-3.5, 3.5]  # 车道间虚线在两侧 3.5m


def test_boundary_count_and_geometry(vecs: tuple[MapVec, ...]) -> None:
    bv = by_cls(vecs, "boundary")
    # lane2/lane-2 最外 solid + lane3/lane-3 外缘 curb(最外两侧边)
    assert len(bv) == 4
    curbs = [v for v in bv if dict(v.attrs).get("mark_type") == "curb"]
    assert len(curbs) == 2
    assert sorted({round(p[1], 1) for v in curbs for p in v.points}) == [-9.0, 9.0]
    solids = [v for v in bv if dict(v.attrs).get("mark_type") == "solid"]
    assert sorted({round(p[1], 1) for v in solids for p in v.points}) == [-7.0, 7.0]


def test_ped_crossing_world_points(vecs: tuple[MapVec, ...]) -> None:
    cw = by_cls(vecs, "ped_crossing")
    assert len(cw) == 1
    pts = cw[0].points
    assert len(pts) == 5 and pts[0] == pts[4]  # 4 角 + 闭合
    # hdg=0、s=50、t=0:长轴沿 s,世界 y 为 ±2
    assert sorted({round(p[1], 1) for p in pts}) == [-2.0, 2.0]
    assert sorted({round(p[0], 1) for p in pts}) == [45.0, 55.0]


def test_stop_line_two_points(vecs: tuple[MapVec, ...]) -> None:
    sl = by_cls(vecs, "stop_line")
    assert len(sl) == 1
    pts = sl[0].points
    assert len(pts) == 2
    assert sorted(round(p[0], 1) for p in pts) == [17.0, 23.0]  # 沿 hdg 的线段


def test_centerline_all_lanes(vecs: tuple[MapVec, ...]) -> None:
    cl = by_cls(vecs, "centerline")
    # 4 条 driving + 2 条 sidewalk(全 lane type,attrs 供过滤)
    assert len(cl) == 6
    ys = sorted({round(p[1], 2) for v in cl for p in v.points})
    assert ys == [-8.0, -5.25, -1.75, 1.75, 5.25, 8.0]
    assert all(dict(v.attrs).get("lane_type") for v in cl)


def test_traffic_light_validity(vecs: tuple[MapVec, ...]) -> None:
    tl = by_cls(vecs, "traffic_light")
    assert len(tl) == 1
    assert tl[0].points[0] == (80.0, 0.0, 0.0)
    attrs = dict(tl[0].attrs)
    assert attrs["validity"] == "1->-1" and attrs["dynamic"] == "yes"


def test_resample_uniform_20() -> None:
    # 直线折线:弧长等距 = 欧氏等距(拐角折线欧氏距 < 弧长,属几何事实非 bug)
    v = MapVec("divider", ((0.0, 0.0, 0.0), (15.0, 0.0, 0.0)), (), "d_1", "")
    r = resample(v, 20)
    assert len(r.points) == 20
    assert r.points[0] == v.points[0] and r.points[-1] == v.points[-1]
    import math

    d = [math.dist(r.points[i], r.points[i + 1]) for i in range(19)]
    assert max(d) - min(d) < 1e-9  # 等距


def test_resample_keeps_fixed_classes() -> None:
    tl = MapVec("traffic_light", ((1.0, 2.0, 3.0),), (), "t_1", "")
    assert resample(tl, 20) is tl


def test_crop_three_states() -> None:
    inner = MapVec("divider", ((1.0, 1.0, 0.0), (5.0, 1.0, 0.0)), (), "i", "")
    outer = MapVec("divider", ((60.0, 60.0, 0.0), (70.0, 70.0, 0.0)), (), "o", "")
    cross = MapVec("divider", ((-60.0, 0.0, 0.0), (0.0, 0.0, 0.0)), (), "c", "")
    out = crop_to_ego((inner, outer, cross), (0.0, 0.0), radius=51.2)
    assert {v.id for v in out} == {"i", "c"}
    c = next(v for v in out if v.id == "c")
    assert len(c.points) == 2
    assert abs(c.points[0][0] + 51.2) < 1e-6 and abs(c.points[1][0]) < 1e-6


def test_crop_splits_crossing_polyline() -> None:
    # 中间点在窗外:穿窗两次的折线应分裂为两段
    v = MapVec("divider", ((-60.0, 10.0, 0.0), (0.0, 100.0, 0.0), (60.0, 10.0, 0.0)), (), "sp", "")
    out = crop_to_ego((v,), (0.0, 0.0), radius=51.2)
    assert len(out) == 2
    assert {o.id for o in out} == {"sp_0", "sp_1"}


@pytest.mark.skipif(
    not glob.glob("/root/autodl-tmp/CARLA_0.9.16/**/*Town10HD_Opt.xodr", recursive=True), reason="需本机 xodr"
)
def test_town10_counts() -> None:
    from autodrivedata.opendrive import parse_xodr

    path = glob.glob("/root/autodl-tmp/CARLA_0.9.16/**/*Town10HD_Opt.xodr", recursive=True)[0]
    vecs = extract_mapvec(parse_xodr(path))
    by = {
        c: len(by_cls(vecs, c))
        for c in ("divider", "boundary", "ped_crossing", "stop_line", "centerline", "traffic_light")
    }
    # 与 2026-09-10/11 摸底对账
    assert by["ped_crossing"] == 16
    assert by["stop_line"] == 21
    assert 15 <= by["traffic_light"] <= 21
    center_div = [v for v in vecs if v.cls == "divider" and dict(v.attrs).get("centerline") == "yes"]
    assert len(center_div) >= 80  # center mark 586 段,同属性相邻段合并后实测 106
    assert by["divider"] > len(center_div)  # 同向 divider 非空
    assert by["centerline"] > 0 and by["boundary"] > 0
    # 折线至少 2 点
    for v in vecs:
        if v.cls not in ("traffic_light",):
            assert len(v.points) >= 2

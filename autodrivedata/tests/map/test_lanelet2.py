"""Lanelet2 适配器单测:往返保真(含 z)/ 标准 OSM 结构 / 投影 / 第三方文件尽力读回。

**主判据是往返**:`MapVec` → `.osm` → `MapVec` 后**逐字段全等**(六类 + 重复键 attrs +
**非零 z** + 退化折线)。往返不成立的话,这个格式就只是"单向写出去的死文件",不能当 GT 源接回管线。

**两条曾经翻车的钉**(2026-09-27 修,别再退回去):
1. 早先的往返断言写的是 `for (ax, ay, _), (bx, by, _)` —— **故意跳过 z**,所以 z 被丢光它也不红。
   现在**含 z 逐分量比较**。
2. 早先的写出把坐标**内联**进 `<nd lat= lon=/>`(非标准)⇒ 真实 lanelet2 读者只认 `ref`,
   会把每条 way 读成空几何。现在钉标准结构。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from autodrivedata.map.lanelet2 import (
    dump_lanelet2,
    from_latlon,
    load_lanelet2,
    to_latlon,
)
from autodrivedata.map.mapvec import MapVec

# 六类各一条:含**非零 z**(高程是 2026-09-27 修的那个丢 z 缺陷的正对靶)、
# 重复键 attrs、空 src、闭合折线(斑马线)、单点(信号灯)
VECS = (
    MapVec(
        "divider",
        ((0.0, 0.0, 1.5), (10.0, 1.0, 2.5)),
        (("mark", "solid"), ("validity", "0:-1"), ("validity", "1:-2")),
        "d1",
        "road:0",
    ),
    MapVec("boundary", ((0.0, 5.0, -3.25), (10.0, 6.0, 0.0)), (), "b1", ""),
    MapVec("centerline", ((0.0, 2.5, 12.0), (10.0, 3.5, 13.0)), (("lane_id", "-1"),), "c1", "sec:0"),
    MapVec(
        "ped_crossing",
        ((0.0, 0.0, 0.1), (2.0, 0.0, 0.2), (2.0, 2.0, 0.3), (0.0, 2.0, 0.4), (0.0, 0.0, 0.1)),
        (),
        "p1",
        "",
    ),
    MapVec("stop_line", ((3.0, 0.0, 0.0), (3.0, 2.0, 0.5)), (), "s1", ""),
    MapVec("traffic_light", ((5.0, 1.0, 4.75),), (("state", "Green"),), "t1", "xodr:949"),
)


def test_roundtrip_is_exact_for_all_six_classes() -> None:
    """★ 往返主判据:六类 + attrs(含重复键)+ id + src + **z** 逐字段全等,顺序不变。

    坐标按 `vec_to_dict` 的 mm 口径比;**z 必须在比较范围内**(这是被修过的那个缺陷)。
    """
    back = load_lanelet2(dump_lanelet2(VECS, "Town10HD_Opt"))
    assert len(back) == len(VECS)
    for a, b in zip(VECS, back, strict=True):
        assert (a.cls, a.id, a.src, a.attrs) == (b.cls, b.id, b.src, b.attrs), f"{a.cls} 字段不一致"
        assert len(a.points) == len(b.points)
        for pa, pb in zip(a.points, b.points, strict=True):
            for i, axis in enumerate("xyz"):
                assert abs(pa[i] - pb[i]) < 1e-6, f"{a.cls} {axis} 漂移 {pa[i]} → {pb[i]}"


def test_elevation_survives_including_large_values() -> None:
    """高程单独钉一条:CARLA 图的高程跨度极大(实测 Town11 到 **791 m**)。

    丢 z 会把桥/隧道/立体交叉**叠成一条** —— 这是拼接多层地图时最致命的一类静默错误。
    """
    v = MapVec("divider", ((0.0, 0.0, 791.05), (10.0, 0.0, 20.57)), (), "d", "")
    back = load_lanelet2(dump_lanelet2((v,)))[0]
    assert [p[2] for p in back.points] == pytest.approx([791.05, 20.57], abs=1e-5)


def test_z_defaults_to_zero_when_ele_missing() -> None:
    """第三方文件没有 `ele` ⇒ z 取 0,不报错(缺省而非拒收)。"""
    txt = """<?xml version="1.0"?>
<osm version="0.6">
  <node id="1" lat="31.230000" lon="121.470000"/>
  <node id="2" lat="31.230100" lon="121.470100"/>
  <way id="1"><nd ref="1"/><nd ref="2"/><tag k="type" v="line_thin"/></way>
</osm>
"""
    back = load_lanelet2(txt, origin=(31.23, 121.47))
    assert [p[2] for p in back[0].points] == [0.0, 0.0]


def test_output_is_standard_osm_structure() -> None:
    """★ 结构钉:**坐标挂在 `<node>` 上,`<way>` 里只有 `<nd ref>`**。

    早先版本把坐标内联在 `<nd lat= lon=/>` 上 —— 那是**破坏性**偏离:真实 lanelet2 读者
    只认 `ref`,会把每条 way 读成空几何(整图变空)。这条防的就是退回去。
    """
    root = ET.fromstring(dump_lanelet2(VECS, "T"))
    ids = {n.get("id") for n in root.findall("node")}
    assert len(ids) == len(root.findall("node")), "node id 有重复"
    for way in root.findall("way"):
        for nd in way.findall("nd"):
            assert set(nd.attrib) == {"ref"}, f"<nd> 只该有 ref,实测 {set(nd.attrib)}"
            assert nd.get("ref") in ids, f"<nd ref={nd.get('ref')}> 指向不存在的 node"
    for node in root.findall("node"):
        assert node.get("lat") is not None and node.get("lon") is not None


def test_output_node_and_way_counts_match() -> None:
    """格式合法性(结构自证):能解析、计数与输入一致。

    ⚠️ 本机无 `lanelet2` 包,**不声称通过官方工具校验** —— 这条只证明结构与自洽。
    """
    root = ET.fromstring(dump_lanelet2(VECS, "T"))
    n_line = sum(1 for v in VECS if v.cls != "traffic_light")
    n_pts = sum(len(v.points) for v in VECS if v.cls != "traffic_light")
    n_light = sum(1 for v in VECS if v.cls == "traffic_light")
    assert len(root.findall("way")) == n_line
    assert len(root.findall("node")) == n_pts + n_light  # 几何节点 + 信号灯节点
    types = {t.get("v") for w in root for t in w.findall("tag") if t.get("k") == "type"}
    assert {"line_thin", "line_thick", "crosswalk", "stop_line"} <= types


def test_duplicate_attr_keys_survive() -> None:
    """`attrs` 允许重复键,而 OSM tag key 必须唯一 ⇒ 靠 `:N` 后缀往返。

    这条单独拎出来,是因为它是本格式**唯一的有损风险点**:后缀方案一旦写错,
    `validity` 这类多车道对就会被静默合并成一条。
    """
    v = MapVec(
        "divider",
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        (("validity", "a"), ("validity", "b"), ("validity", "c")),
    )
    back = load_lanelet2(dump_lanelet2((v,)))[0]
    assert back.attrs == (("validity", "a"), ("validity", "b"), ("validity", "c"))


def test_attr_key_containing_colon_is_not_mangled() -> None:
    """键名自身含 `:`(如 `mark:type`)不许被当成"序号后缀"剥掉。"""
    v = MapVec("divider", ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (("mark:type", "solid"),))
    assert load_lanelet2(dump_lanelet2((v,)))[0].attrs == (("mark:type", "solid"),)


def test_degenerate_lines_are_dropped_and_counted() -> None:
    """单点折线在 OSM 里无合法表达 ⇒ **丢弃但计数上报**(注释里),不静默。

    静默丢要素是这类导出器最难发现的一类错 —— 图看着对,只是少了几条。
    """
    vecs = (
        MapVec("divider", ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (), "ok", ""),
        MapVec("divider", ((9.0, 9.0, 0.0),), (), "single", ""),  # 单点 → 丢
    )
    txt = dump_lanelet2(vecs)
    assert "autodrivedata-dropped: 1" in txt
    assert [v.id for v in load_lanelet2(txt)] == ["ok"]


def test_projection_roundtrip_and_origin_matters() -> None:
    """投影往返误差 < 1e-4 m;且**换 origin 结果必须不同**(防"投影根本没生效")。"""
    origin = (31.23, 121.47)
    for x, y in ((0.0, 0.0), (1234.5, -678.9), (-20000.0, 5000.0)):
        lat, lon = to_latlon(x, y, origin)
        bx, by = from_latlon(lat, lon, origin)
        assert abs(bx - x) < 1e-4 and abs(by - y) < 1e-4
    assert to_latlon(100.0, 100.0, origin) != to_latlon(100.0, 100.0, (0.0, 0.0))


def test_origin_travels_with_the_file() -> None:
    """origin 记在文件注释里 ⇒ 读回不需要调用方记得同一个 origin(否则就是隐式约定)。"""
    vecs = (MapVec("divider", ((100.0, 200.0, 0.0), (150.0, 260.0, 0.0)), (), "d", ""),)
    txt = dump_lanelet2(vecs)
    assert "autodrivedata-origin" in txt
    back = load_lanelet2(txt)  # 不传 origin
    for pa, pb in zip(vecs[0].points, back[0].points, strict=True):
        assert abs(pa[0] - pb[0]) < 1e-6 and abs(pa[1] - pb[1]) < 1e-6


def test_missing_origin_errors_instead_of_guessing() -> None:
    """没有 origin 注释又没显式给 ⇒ **报错**,不默默按 (0,0) 算(那会把整图挪到几内亚湾)。"""
    with pytest.raises(ValueError, match="origin"):
        load_lanelet2('<?xml version="1.0"?><osm version="0.6"></osm>')


def test_reads_third_party_lanelet2_by_semantic_tags() -> None:
    """**非本仓产出**的 lanelet2(只有语义标签、无自定义 tag)按 `type` 尽力还原。

    两种 `<nd>` 形态都要收:**标准**(`ref` 指向 `<node>`)与本模块早先的**内联**写法
    (`<nd lat lon>`,有些工具这么写)—— 只认标准形态的话,内联文件会**整图读成空**。
    """
    standard = """<?xml version="1.0"?>
<osm version="0.6">
  <node id="1" lat="31.230000" lon="121.470000"/>
  <node id="2" lat="31.230100" lon="121.470100"/>
  <way id="1"><nd ref="1"/><nd ref="2"/><tag k="type" v="line_thick"/></way>
  <way id="2"><nd ref="2"/><nd ref="1"/><tag k="type" v="pedestrian_marking"/></way>
</osm>
"""
    inlined = """<?xml version="1.0"?>
<osm version="0.6">
  <way id="1">
    <nd lat="31.230000" lon="121.470000"/>
    <nd lat="31.230100" lon="121.470100"/>
    <tag k="type" v="line_thin"/>
  </way>
</osm>
"""
    for txt, want in ((standard, ["boundary", "ped_crossing"]), (inlined, ["divider"])):
        back = load_lanelet2(txt, origin=(31.23, 121.47))
        assert [v.cls for v in back] == want
        assert all(v.attrs == () for v in back)

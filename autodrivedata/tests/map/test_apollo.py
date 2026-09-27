"""Apollo 适配器单测:往返保真 / 顺序 / 成分解析 / 第三方文件回退。

**主判据是往返**:`MapVec` → `base_map.txt` → `MapVec` 后**逐字段全等**。
Apollo 侧比 lanelet2 多一层风险 —— 文本是按**要素种类分块**写的,而 `load_apollo` 若按
种类分组读,原始**顺序会丢**;故顺序单独钉一条。
"""

from __future__ import annotations

import pytest

from autodrivedata.map.apollo import dump_apollo, load_apollo, parse_textproto
from autodrivedata.map.mapvec import MapVec

VECS = (
    MapVec(
        "divider",
        ((0.0, 0.0, 0.0), (10.0, 1.0, 0.0)),
        (("mark", "solid"), ("validity", "0:-1"), ("validity", "1:-2")),
        "d1",
        "road:0",
    ),
    MapVec(
        "ped_crossing",
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 0.0)),
        (),
        "p1",
        "",
    ),
    MapVec("boundary", ((0.0, 5.0, 0.0), (10.0, 6.0, 0.0)), (), "b1", ""),
    MapVec("stop_line", ((3.0, 0.0, 0.0), (3.0, 2.0, 0.0)), (), "s1", ""),
    MapVec("centerline", ((0.0, 2.5, 0.0), (10.0, 3.5, 0.0)), (("lane_id", "-1"),), "c1", "sec:0"),
    MapVec("traffic_light", ((5.0, 1.0, 0.0),), (("state", "Green"),), "t1", "xodr:949"),
)


def test_roundtrip_is_exact_and_order_preserved() -> None:
    """★ 往返主判据:逐字段全等 **且顺序与导出时一致**(按种类分组读会打乱顺序)。"""
    back = load_apollo(dump_apollo(VECS, "Town10HD_Opt"))
    assert [v.cls for v in back] == [v.cls for v in VECS], "元素顺序被打乱"
    for a, b in zip(VECS, back, strict=True):
        assert (a.cls, a.id, a.src, a.attrs) == (b.cls, b.id, b.src, b.attrs), f"{a.cls} 字段不一致"
        assert a.points == b.points, f"{a.cls} 点不一致"


def test_split_classes_never_mix_across_blocks() -> None:
    """三类线走 `lane.central_curve`、三类走各自要素 —— 同一条线**不许**出现在两个块里。

    钉的是"借 lane 承载"这个降级手段没把要素**重复**输出(重复会给下游制造幽灵实例)。
    """
    root = parse_textproto(dump_apollo(VECS))
    assert len(root["lane"]) == 3  # divider / boundary / centerline
    assert len(root["crosswalk"]) == 1
    assert len(root["stop_line"]) == 1
    assert len(root["signal"]) == 1
    assert len(load_apollo(dump_apollo(VECS))) == len(VECS)


def test_lane_geometry_does_not_absorb_boundary_points() -> None:
    """★ `lane` 只取 `central_curve` —— 第三方 lane 带左右边界时不许把边界点吸进来。

    整节点递归收点会把 `left_boundary.curve` 的点并进 central_curve,折线悄悄变长;
    这类"多出点"比"少了要素"更难发现(点还在,只是不再是那条线)。
    """
    txt = """
lane {
  id { id: "thirdparty-lane-1" }
  central_curve { segment { line_segment { point { x: 0 y: 0 z: 0 } point { x: 10 y: 0 z: 0 } } } }
  left_boundary  { curve { segment { line_segment { point { x: 0 y: 5 z: 0 } point { x: 10 y: 5 z: 0 } } } } }
  right_boundary { curve { segment { line_segment { point { x: 0 y: -5 z: 0 } point { x: 10 y: -5 z: 0 } } } } }
}
"""
    back = load_apollo(txt)
    assert len(back) == 1
    assert len(back[0].points) == 2, f"central_curve 被污染成 {len(back[0].points)} 个点"
    assert [p[0] for p in back[0].points] == [0.0, 10.0]


def test_third_party_map_falls_back_to_kind() -> None:
    """id 前缀不是本适配器写的 ⇒ 按要素种类判类(真车道 → `centerline`,语义最近的一类)。"""
    txt = """
crosswalk { id { id: "cw-1" } polygon { point { x: 0 y: 0 z: 0 } point { x: 1 y: 0 z: 0 } } }
stop_line { id { id: "sl-1" } segment { line_segment { point { x: 0 y: 0 z: 0 } point { x: 1 y: 0 z: 0 } } } }
lane { id { id: "lane-7" } central_curve { segment { line_segment { point { x: 0 y: 0 z: 0 } point { x: 2 y: 0 z: 0 } } } } }
"""
    back = load_apollo(txt)
    assert [v.cls for v in back] == ["ped_crossing", "stop_line", "centerline"]
    assert all(v.attrs == () for v in back)
    assert [v.id for v in back] == ["cw-1", "sl-1", "lane-7"]


def test_empty_geometry_is_dropped_and_counted() -> None:
    """空点集的要素丢弃并**计数上报**(注释),不静默。"""
    txt = dump_apollo(
        (
            MapVec("divider", ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (), "ok", ""),
            MapVec("divider", (), (), "empty", ""),
        )
    )
    assert "autodrivedata-dropped: 1" in txt
    assert [v.id for v in load_apollo(txt)] == ["ok"]


def test_textproto_parser_handles_nesting_and_comments() -> None:
    """成分解析器:嵌套 message 成列表、`#` 注释被剥、引号内的 `#` **不算**注释。"""
    msg = parse_textproto(
        """
# 顶层注释
a { b: 1 b: 2 }
c { d { e: "x#y" } }   # 行尾注释
f: -1.5e-3
"""
    )
    assert msg["a"][0]["b"] == ["1", "2"]
    assert msg["c"][0]["d"][0]["e"] == ["x#y"]
    assert msg["f"] == ["-1.5e-3"]


def test_textproto_parser_rejects_malformed_input() -> None:
    """坏输入**报错**,不静默返回半个解析结果。"""
    with pytest.raises(ValueError):
        parse_textproto("a { b: 1")  # 括号不闭合
    with pytest.raises(ValueError):
        parse_textproto("a b: 1")  # 字段后既非 : 也非 {


def test_map_name_lives_in_a_comment_not_the_schema() -> None:
    """`map_name` 写注释而非私有字段 —— 往 schema 里塞字段就不算"合法 Apollo"了。"""
    txt = dump_apollo(VECS, "Town10HD_Opt")
    assert "# map_name: Town10HD_Opt" in txt
    assert "map_name {" not in txt

"""跨图拼接单测:恒等不变 / 计数守恒 / z 保真 / 去重 / 变换口径 / 接缝报告。

**两条主判据**:
1. **恒等 placement ⇒ 单图产物逐位不变**(合并不该改变单图)—— 靠 `place()` 直接返回同一对象保证;
2. **z 保真** —— 拼接不许碰高程。12 张能提的 CARLA 图里 10 张有非平凡高程(最大 791 m),
   丢 z 会把桥/隧道叠成一条,而那种错在图上**看着是对的**。
"""

from __future__ import annotations

import math

import pytest

from autodrivedata.map.mapvec import MapVec
from autodrivedata.map.stitch import (
    Placement,
    parse_placement,
    parse_stitch_spec,
    place,
    seams,
    stitch,
)

A = (
    MapVec("divider", ((0.0, 0.0, 1.5), (10.0, 0.0, 2.5)), (("m", "solid"),), "d1", "road:1"),
    MapVec("ped_crossing", ((0.0, 3.0, 0.2), (2.0, 3.0, 0.4), (2.0, 5.0, 0.6)), (), "p1", ""),
)
B = (MapVec("divider", ((0.0, 0.0, 9.0), (10.0, 0.0, 8.0)), (("m", "dashed"),), "d1", "road:9"),)


def test_identity_placement_returns_the_same_object() -> None:
    """★ 恒等 placement **原样返回同一对象**(不复制)⇒ 单图产物逐位不变。"""
    assert place(A, Placement()) is A
    assert place(A, Placement(0.0, 0.0, 0.0, 0.0)) is A


def test_single_map_stitch_changes_nothing() -> None:
    """单图 + 恒等 ⇒ 几何与 id **逐位不变**(多图才需要加前缀,单图加前缀纯属噪声)。"""
    merged, stats = stitch([("A", A, Placement())])
    assert merged == A
    assert [v.id for v in merged] == [v.id for v in A]
    assert stats == {**stats, "n_in": 2, "n_out": 2, "n_dedup": 0}


def test_count_conserved_when_maps_do_not_overlap() -> None:
    """不重合的两图:实例数 = 各图之和(id 加前缀以防撞车)。"""
    merged, stats = stitch([("A", A, Placement()), ("B", B, Placement(1000.0, 0.0))])
    assert len(merged) == len(A) + len(B)
    assert stats["n_dedup"] == 0
    assert [v.id for v in merged] == ["d1@A", "p1@A", "d1@B"]


def test_overlapping_maps_dedup_to_one() -> None:
    """★ 重合的两图去重成一条,且去重后**容差内不再有同类重复对**。

    只断言"少了一条"不够 —— 真正要保证的是去重跑完之后**没有残留**。
    """
    merged, stats = stitch([("A", A, Placement()), ("B", B, Placement())])
    assert stats["n_dedup"] == 1
    assert stats["n_in"] == 3 and stats["n_out"] == 2
    divs = [v for v in merged if v.cls == "divider"]
    assert len(divs) == 1
    assert stats["n_bucket_skipped"] == 0


def test_dedup_respects_tolerance() -> None:
    """容差是判据:0.04 m 的错位在 0.05 容差内算重复;2 m 的错位不算。"""
    base = MapVec("divider", ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)), (), "d", "")
    near = MapVec("divider", ((0.0, 0.04, 0.0), (10.0, 0.04, 0.0)), (), "d2", "")
    far = MapVec("divider", ((0.0, 2.0, 0.0), (10.0, 2.0, 0.0)), (), "d3", "")
    assert stitch([("A", (base,), Placement()), ("B", (near,), Placement())])[1]["n_dedup"] == 1
    assert stitch([("A", (base,), Placement()), ("B", (far,), Placement())])[1]["n_dedup"] == 0


def test_different_classes_are_not_deduped() -> None:
    """只有**同类**才算重复 —— 重叠边界上 divider 与 boundary 贴在一起是常态,别误删。"""
    d = MapVec("divider", ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)), (), "d", "")
    b = MapVec("boundary", ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)), (), "b", "")
    merged, stats = stitch([("A", (d,), Placement()), ("B", (b,), Placement())])
    assert stats["n_dedup"] == 0 and len(merged) == 2


def test_z_is_preserved_through_placement_and_stitch() -> None:
    """★ z 保真:变换只加 `dz`,拼接不碰高程。

    CARLA 图高程跨度可达 791 m(Town11),桥/隧道/立体交叉全靠 z 区分 ——
    丢了它,不同层级的道路会在图上**看着正常地**叠成一条。
    """
    moved = place(A, Placement(1.0, 2.0, 100.0, 90.0))
    assert [p[2] for p in moved[0].points] == [101.5, 102.5]
    merged, _ = stitch([("A", A, Placement()), ("B", B, Placement(500.0, 0.0, -100.0))])
    by_src = {v.src: v for v in merged}
    assert [p[2] for p in by_src["road:1"].points] == [1.5, 2.5]
    assert [p[2] for p in by_src["road:9"].points] == [-91.0, -92.0]


def test_yaw_matches_the_documented_formula() -> None:
    """变换**口径**钉:`x' = dx + x·cosθ − y·sinθ`、`y' = dy + x·sinθ + y·cosθ`。

    靠感觉判断"往哪转"在这套坐标里不可靠(以公式为准),所以用 90° 手算值钉死。
    """
    v = (MapVec("divider", ((1.0, 0.0, 0.0),), (), "d", ""),)
    got = place(v, Placement(10.0, 20.0, 0.0, 90.0))[0].points[0]
    assert got[0] == pytest.approx(10.0 + 1.0 * math.cos(math.pi / 2))
    assert got[1] == pytest.approx(20.0 + 1.0 * math.sin(math.pi / 2))


def test_placement_actually_applied() -> None:
    """两个不同 placement ⇒ 结果必须不同(防"参数收了但没用")。"""
    a = place(A, Placement())
    b = place(A, Placement(100.0, 0.0))
    assert a[0].points != b[0].points


def test_attrs_and_src_survive_stitching() -> None:
    """拼接只加 id 前缀,**不动** attrs / src / cls / 点数。"""
    merged, _ = stitch([("A", A, Placement()), ("B", B, Placement(900.0, 0.0))])
    by_src = {v.src: v for v in merged}
    assert by_src["road:1"].attrs == (("m", "solid"),)
    assert by_src["road:9"].cls == "divider"


def test_parse_placement_and_spec() -> None:
    assert parse_placement("1,2") == Placement(1.0, 2.0, 0.0, 0.0)
    assert parse_placement("1,2,3,90") == Placement(1.0, 2.0, 3.0, 90.0)
    assert parse_stitch_spec("A=0,0;B=420,0,0,90") == [
        ("A", Placement()),
        ("B", Placement(420.0, 0.0, 0.0, 90.0)),
    ]
    with pytest.raises(ValueError, match="1–4 个数"):
        parse_placement("1,2,3,4,5")
    with pytest.raises(ValueError, match="缺 '='"):
        parse_stitch_spec("A0,0")


def test_dedup_never_fires_within_one_map() -> None:
    """★ 去重**只在「来源图不同」之间比,同图内部一律不去重**(实测逼出来的判据)。

    在真实图上开"同图也去重"会删掉 `Town10HD_Opt` 的 10 条、`Town01` 的 16 条 ——
    全是**不同 road 的中心线恰好重合**(实测 `road 90` vs `road 89`,互距 0.0000 m)。
    那是**真实的道路结构**(分隔带两侧、被拆成多个 road id 的同一条路),不是重复。
    单张图自身自洽,重复只可能来自两张图在交叠区各导了一遍。
    """
    a = MapVec("divider", ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)), (), "road89", "")
    b = MapVec("divider", ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)), (), "road90", "")
    merged, stats = stitch([("A", (a, b), Placement())])
    assert stats["n_dedup"] == 0, "同图内重合的两条是不同要素,不能删"
    assert len(merged) == 2


def test_single_point_elements_participate_in_dedup() -> None:
    """★ 单点要素(信号灯)必须参与去重 —— 漏掉它们会静默残留重复。

    实测:同一张图复制两份同位时,把 1 点要素排除在去重之外会残留 **36/495** 条,
    而那 36 恰好就是信号灯的条数。
    """
    light = MapVec("traffic_light", ((5.0, 1.0, 4.0),), (("state", "Green"),), "t1", "")
    merged, stats = stitch([("A", (light,), Placement()), ("B", (light,), Placement())])
    assert stats["n_dedup"] == 1 and len(merged) == 1
    # 错开位置则不去重
    _, st2 = stitch([("A", (light,), Placement()), ("B", (light,), Placement(500.0, 0.0))])
    assert st2["n_dedup"] == 0


# ---------------- 接缝报告(只对"本来就该相接"的图有意义)----------------


def test_seams_finds_adjacent_endpoints() -> None:
    """把 B 平移成与 A 首尾相接,应报出接缝;离远了则无。"""
    assert seams([("A", A, Placement())]) == [], "同图内部不算接缝"
    got = seams([("A", A, Placement()), ("B", B, Placement(0.0, 0.0))])
    assert got, "同一位置的两图端点该报接缝"
    assert all(s.dist <= 1.0 for s in got)


def test_seams_empty_when_far_apart() -> None:
    assert seams([("A", A, Placement()), ("B", B, Placement(10_000.0, 0.0))]) == []


def test_seams_respects_heading_filter() -> None:
    """朝向差超阈值的端点对**不算接缝**(路口处正交的两条线不该被当成一个口子)。"""
    a = (MapVec("divider", ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)), (), "a", ""),)
    b = (MapVec("divider", ((0.0, 0.0, 0.0), (0.0, 10.0, 0.0)), (), "b", ""),)
    assert seams([("A", a, Placement()), ("B", b, Placement())]) == []
    assert seams([("A", a, Placement()), ("B", b, Placement())], heading_tol_deg=180.0)

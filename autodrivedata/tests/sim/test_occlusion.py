"""静态遮挡的**纯几何**判据(零 carla)。用 P1 冻结布局的真数手算锚定。

## 钉的是什么

`collect_ab_route --occluders` 的失败模式**全是不可见的**:墙摆偏一点、矮一点、离车道
近一点 —— 采集照样跑完、GT 照样相等、图看着也对,只有"效果比预期小"这一个症状,
而它与"模型对遮挡鲁棒"在数据上长得一模一样。所以每条几何都得有个能离线手算的锚:

| 判据 | 错了会怎样 |
|---|---|
| `TestFrozenLayout` 净空 > 0 | 墙伸进车道 ⇒ ego 撞上去 **照样采得完**,只是轨迹不再与 A 侧配对 |
| `TestFrozenLayout` 可见高度落在断崖区 | 落不到 21–24 px 那一段 ⇒ 这一档问不出问题 |
| `TestFrozenLayout` full 档恒 0 px | 阳性对照失效 ⇒ "没掉点"分不清是模型强还是墙没摆上 |
| `TestSightLineVsForward` 视线上摆放全盖住车 | 排 ego 正前方会漏一条全高的缝(视差),遮挡退化成"看运气" |
| `TestDepths` 深度 ≠ 斜距 | **2026-10-01 真踩过**:全程用斜距,20 m 处整框算成 49.7 px 而 `label_2` 写着 58.6 px |

车宽/车高用**真实资产实测值**(`NPC_MODELS` 那四台),不是估计值。
"""

from __future__ import annotations

import math

import pytest

from autodrivedata.sim.occlusion import (
    CAM_FWD,
    OCCLUDER_COVER,
    OCCLUDER_DIMS,
    OCCLUDER_GAP,
    OCCLUDER_MODELS,
    Occluder,
    occluder_depth,
    project,
    required_span,
    sight_line_occluders,
    visible_fraction,
)

# --- P1 冻结布局 + 资产实测(2026-10-01 从 CARLA 资产 bbox 量)--------------
STATIC_OFFSETS = (20.0, 35.0, 50.0, 62.0)
STATIC_LAT = 3.5
CAM_Z = 1.65  # SENSOR_OFFSET.location.z(相对 ego 原点);相机 pitch=0 ⇒ 光轴水平
FOCAL_PX = (1242 / 2.0) / math.tan(math.radians(90.0) / 2.0)  # = 621.0
EGO_HALF_W = 2.163 / 2.0  # 道具在车顶前装,ego 是 vehicle.tesla.model3

#: (前向 d, 横向 lat, 车宽 W, 车顶相对 ego 原点);四台依次对应 NPC_MODELS。
#: 车顶 = 资产 bbox 的 `bb.location.z + bb.extent.z`(tesla 1.480 / audi 1.556 /
#: mustang 1.301 / prius 1.487)—— **实采时读回的是这几个数再加 ~0.19 m**
#: (车比 ego 原点略高),见 `TestFrozenLayout.test_the_regime_holds_across_car_heights`。
CARS = (
    (20.0, STATIC_LAT, 2.163, 1.480),  # tesla.model3
    (35.0, STATIC_LAT, 1.789, 1.556),  # audi.a2
    (50.0, STATIC_LAT, 1.895, 1.301),  # ford.mustang
    (62.0, STATIC_LAT, 2.007, 1.487),  # toyota.prius
)


def _depths(car: tuple[float, float, float, float]) -> tuple[float, float]:
    d, lat = car[0], car[1]
    return d - CAM_FWD, occluder_depth(d, lat, gap=OCCLUDER_GAP)


def _slots(mode: str, car: tuple[float, float, float, float]) -> list[Occluder]:
    d, lat, w, _top = car
    unit_len, unit_thick, _h = OCCLUDER_DIMS[mode]
    depth_car, depth_occl = _depths(car)
    return sight_line_occluders(
        d,
        lat,
        gap=OCCLUDER_GAP,
        unit_len=unit_len,
        unit_thick=unit_thick,
        cover=OCCLUDER_COVER,
        needed_width=required_span(w, depth_car, depth_occl),
    )


def _proj(mode: str, car: tuple[float, float, float, float]):
    depth_car, depth_occl = _depths(car)
    return project(
        focal_px=FOCAL_PX,
        cam_z=CAM_Z,
        occl_top_z=OCCLUDER_DIMS[mode][2],
        car_top_z=car[3],
        depth_car=depth_car,
        depth_occl=depth_occl,
    )


class TestRegistry:
    """注册表本身:**道具必须落在 GT 过滤器的反面**,否则 A/B 的硬门槛当场失效。"""

    def test_every_mode_has_model_and_dims(self):
        assert set(OCCLUDER_MODELS) == set(OCCLUDER_DIMS) == {"partial", "full"}

    @pytest.mark.parametrize("mode", sorted(OCCLUDER_MODELS))
    def test_model_is_a_static_prop(self, mode: str):
        """★ 核心判据:`static.prop.*` —— 与 `collect_ab_route` 收 `label_2` 的类型过滤
        (`vehicle*` / `walker*`)**无交集**,所以遮挡物结构上进不了 GT。"""
        mid = OCCLUDER_MODELS[mode]
        assert mid.startswith("static.prop."), mid
        assert not mid.startswith(("vehicle", "walker")), f"{mid} 会被写进 label_2 —— A/B 作废"

    def test_dims_are_plausible_bounds(self):
        """尺寸是摆位数学的唯一输入,写反一个数(长宽互换)不报错、只让墙变窄。"""
        for mode, (ln, th, h) in OCCLUDER_DIMS.items():
            assert 0.2 < th < ln < 4.0, f"{mode} 的长/厚序反了"
            assert 0.5 < h < 3.0, f"{mode} 的高不合理"

    def test_only_full_tower_over_the_camera(self):
        """★ `full` 档当阳性对照的**全部依据**就是这一条:它比相机高。

        `partial` 若哪天被换成一根比相机高的杆,"部分遮挡"就悄悄变成了"全遮",
        而两档的产出仍各自成立、只是**测的是同一件事**。
        """
        assert OCCLUDER_DIMS["full"][2] > CAM_Z > OCCLUDER_DIMS["partial"][2]


class TestDepths:
    """★ 深度 vs 斜距 —— 2026-10-01 真踩过的那一处。"""

    def test_camera_is_ahead_of_the_ego_origin(self):
        assert CAM_FWD == pytest.approx(1.2)  # == SENSOR_OFFSET.location.x

    def test_depth_is_not_the_slant_range(self):
        """20 m 处:深度 18.8,斜距 19.12 —— 差 0.32 m,而整框高度对它是**线性**的。"""
        d, lat = 20.0, 3.5
        assert d - CAM_FWD == pytest.approx(18.8)
        assert math.hypot(d, lat) == pytest.approx(20.304, abs=1e-3)
        assert occluder_depth(d, lat, gap=OCCLUDER_GAP) == pytest.approx(15.851, abs=1e-3)

    def test_the_two_conventions_give_visibly_different_px(self):
        """反向对照:深度 18.8 ⇒ 整框 **48.89 px**;斜距 20.30 ⇒ **45.27 px**。差 **8%**。

        方向是**斜距偏小**(距离更大 ⇒ 更矮),而 `label_2` 里那一帧实测 **58.6 px** ——
        两支都在它下面,因为 GT 报的是整条 4.79 m 车长投影出来的 AABB(近端面比 18.8 近
        约 2.4 m)。所以这条**不能拿"谁离 GT 近"来判**,判据只有成像公式本身:
        `v = cy + fy·Y/Z`,Z 是沿光轴的那一个。
        """
        d, lat, _w, top = CARS[0]
        kw = {
            "focal_px": FOCAL_PX,
            "cam_z": CAM_Z,
            "occl_top_z": OCCLUDER_DIMS["partial"][2],
            "car_top_z": top,
        }
        by_depth = project(depth_car=d - CAM_FWD, depth_occl=occluder_depth(d, lat, gap=OCCLUDER_GAP), **kw)
        by_slant = project(depth_car=math.hypot(d, lat), depth_occl=math.hypot(d, lat) - OCCLUDER_GAP, **kw)
        assert by_depth.full_px == pytest.approx(48.89, abs=0.05)
        assert by_slant.full_px == pytest.approx(45.27, abs=0.05)
        assert by_depth.full_px > by_slant.full_px * 1.05, "两种口径居然差不多 —— 那这条坑就不值一提了"


class TestFrozenLayout:
    """★ P1 四个距离上的真数 —— 手算锚,改摆位数学这里必红。"""

    @pytest.mark.parametrize("mode", sorted(OCCLUDER_MODELS))
    def test_clearance_stays_positive(self, mode: str):
        """净空 > 0:墙不许伸进 ego 走的那条线。

        为什么不是"跑一遍看撞没撞":**撞了也采得完**(只是轨迹不再与 A 配对),
        而"轨迹不配对"要到 `eval_2d_ab` 出数之后才看得出来。
        """
        for car in CARS:
            clear = min(s.min_y for s in _slots(mode, car)) - EGO_HALF_W
            assert clear > 0.2, f"{mode} @{car[0]:.0f} m 净空只有 {clear:+.3f} m"

    def test_partial_at_20m_is_the_calibration_anchor(self):
        """部分遮挡 @20 m:48.9 px 的车只剩 **17.2 px**(实采车身略高 ⇒ 约 23 px)。

        两个数都在 21–24 px 断崖的**下沿到中心**那一段,也就是这一档唯一有信息量的车
        (35/50/62 m 的整框本来就只有 28.6/16.6/15.2 px)。
        """
        p = _proj("partial", CARS[0])
        assert p.full_px == pytest.approx(48.89, abs=0.05)
        assert p.visible_px == pytest.approx(17.15, abs=0.05)
        assert p.visible_frac == pytest.approx(0.3515, abs=1e-3)

    def test_the_regime_holds_across_car_heights(self):
        """★ 20 m 那台(tesla)不管按资产尺寸(1.48)还是实采读回(1.67)都落在断崖带里。

        单点锚容易被"只有那一个输入好看"骗过 —— 这条横扫该车**实际可能取到**的车顶高度,
        要求可见高度落在 [15, 30] px:低于 15 就跟全遮没区别(**这一档白做**),
        高过 30 就越过断崖、模型照检不误(**还是白做**)。
        """
        for top in (1.44, 1.48, 1.56, 1.67, 1.75):
            px = _proj("partial", (20.0, STATIC_LAT, 2.163, top)).visible_px
            assert 15.0 <= px <= 30.0, f"车顶 {top} m ⇒ 可见 {px:.1f} px,掉出断崖带"

    def test_a_very_low_car_is_no_longer_partial(self):
        """反向对照(记录边界):车顶只有 1.30 m 时可见只剩 **11.2 px** —— 那已近似全遮。

        这条不是"顺带一测",是把这一档的**适用边界**写下来:低矮车(以及远距离下整框本就
        只有十几 px 的那几台)从这一档里问不出"部分遮挡"的信息,别把它们的结果读成
        "模型对部分遮挡也鲁棒"。
        """
        assert _proj("partial", (20.0, STATIC_LAT, 2.163, 1.301)).visible_px == pytest.approx(11.20, abs=0.05)
        assert _proj("partial", CARS[2]).visible_px < 4.0  # 50 m 那台整框才 16.6 px

    def test_full_is_zero_everywhere(self):
        """★ 阳性对照:1.86 m > 相机 1.65 m ⇒ 四个距离上**可见都是 0**。

        它要是不为 0,这条实验的"没掉点"就分不清是模型强还是墙没摆上。
        """
        for car in CARS:
            p = _proj("full", car)
            assert p.visible_px == 0.0, f"full @{car[0]:.0f} m 还剩 {p.visible_px:.2f} px"
            assert p.visible_frac == 0.0

    def test_partial_never_hides_the_whole_car(self):
        """反向对照:部分遮挡档**不许**全遮 —— 否则两档就退化成同一档。"""
        for car in CARS:
            assert _proj("partial", car).visible_frac > 0.15, f"partial @{car[0]:.0f} m 只剩一点"

    def test_two_blocks_per_car_at_the_default_gap(self):
        """跨度是**算出来的**、不是写死的:默认参数下四个距离都需要 2 块。"""
        for car in CARS:
            assert len(_slots("partial", car)) == 2
            assert len(_slots("full", car)) == 2

    def test_block_count_follows_the_required_span(self):
        """反向对照:把所需跨度放大一倍就会多要一块 —— 说明块数是**算**出来的。"""
        d, lat, w, _ = CARS[0]
        depth_car = d - CAM_FWD
        need = required_span(w, depth_car, occluder_depth(d, lat, gap=OCCLUDER_GAP))
        more = sight_line_occluders(
            d,
            lat,
            gap=OCCLUDER_GAP,
            unit_len=1.215,
            unit_thick=0.372,
            cover=OCCLUDER_COVER,
            needed_width=need * 2,
        )
        assert len(more) == 4


class TestSightLineVsForward:
    """★ 为什么排在**视线**上而不是 ego 正前方 —— 直接把那条缝算出来。

    比的是**同一块墙**(同跨度、同"比车近 gap 米"),只把中心从视线上挪到 ego 正前方:
    视差让它在像面上整体外移,车**内侧**于是露出一条全高的缝。
    """

    @staticmethod
    def _angles(points: list[tuple[float, float]]) -> list[float]:
        return sorted(math.degrees(math.atan2(y, x - CAM_FWD)) for x, y in points)

    @staticmethod
    def _corners(s: Occluder) -> list[tuple[float, float]]:
        """一块遮挡物的**长轴两端**(不是中心 —— 覆盖看的是两端点到哪)。"""
        hx = math.cos(math.radians(s.yaw_deg)) * s.length / 2
        hy = math.sin(math.radians(s.yaw_deg)) * s.length / 2
        return [(s.x - hx, s.y - hy), (s.x + hx, s.y + hy)]

    def test_sight_line_covers_the_whole_car_at_20m(self):
        """墙的**端点**角跨度必须包住车的两条边。

        (第一版拿**中心**去比,量出来只有 2.76° —— 中心间距是 `step` 不是 `span`,
        那是把"墙有多宽"记成了"块中心离多远"。)
        """
        d, lat, w, _top = CARS[0]
        wall = self._angles([p for s in _slots("partial", CARS[0]) for p in self._corners(s)])
        car = self._angles([(d, lat - w / 2), (d, lat + w / 2)])
        assert wall[0] <= car[0] + 0.05 and wall[-1] >= car[-1] - 0.05, (
            f"墙 [{wall[0]:.2f},{wall[-1]:.2f}] 没盖住车 [{car[0]:.2f},{car[-1]:.2f}]"
        )

    def test_forward_placement_leaks_a_full_height_slit(self):
        """排 ego 正前方(横向偏移 = 车的 3.5 m)会因视差在车**内侧**漏一条全高的缝。

        同一个 `span` 搬到正前方:内边角从车的 **7.33°** 外移到墙的 **8.50°**,
        即车宽的 **18.3%** 整条全高可见。排视线上则是 0%。
        """
        d, lat, w, _ = CARS[0]
        _dc, depth_occl = _depths(CARS[0])
        span = max(OCCLUDER_COVER * required_span(w, d - CAM_FWD, depth_occl), OCCLUDER_DIMS["partial"][0])
        fwd = self._angles([(d - OCCLUDER_GAP, lat - span / 2), (d - OCCLUDER_GAP, lat + span / 2)])
        car = self._angles([(d, lat - w / 2), (d, lat + w / 2)])
        leak = (fwd[0] - car[0]) / (car[1] - car[0])  # 内侧那条缝占车宽的比例
        assert leak == pytest.approx(0.183, abs=0.01)
        assert fwd[0] > car[0], "正前方摆放竟然盖住了内边 —— 说明视差那一项没了"
        assert fwd[0] < car[1], "缝比车还宽 —— 那就不是'缝'了,是没挡住"


class TestOccluderPlacement:
    """摆位本身的代数性质:同一平面上、对称于视线、首尾相接铺满。"""

    def test_blocks_share_one_plane_and_are_symmetric(self):
        """各块沿视线**同一个距离**(是一堵墙不是一个楔子),且对称铺在视线两侧。

        坐标必须**以相机为原点**量:视线是从**相机**出发的,拿 ego 原点去量会多出一个
        恒定的 `−lat·CAM_FWD/r ≈ −0.22 m` 偏置,于是"对称"这条被测成红的(第一版就是这样)。
        """
        slots = _slots("partial", CARS[0])
        d, lat = CARS[0][0], CARS[0][1]
        r = math.hypot(d - CAM_FWD, lat)
        ux, uy = (d - CAM_FWD) / r, lat / r
        along = [(s.x - CAM_FWD) * ux + s.y * uy for s in slots]
        perp = [-(s.x - CAM_FWD) * uy + s.y * ux for s in slots]
        assert max(along) - min(along) == pytest.approx(0.0, abs=1e-9), "各块不在同一平面上"
        assert sum(perp) == pytest.approx(0.0, abs=1e-9), "各块没对称铺在视线两侧"
        # 反向对照:沿视线真的推进了 gap 那么多(不是"都在同一处"这种平凡满足)
        assert along[0] == pytest.approx(r - OCCLUDER_GAP, abs=1e-9)

    def test_long_axis_is_perpendicular_to_the_sight_line(self):
        """长轴垂直 ⇒ 墙顶在像面上是水平的 ⇒ 可见高度沿车宽恒定。"""
        d, lat = CARS[0][0], CARS[0][1]
        for blk in _slots("partial", CARS[0]):
            ax, ay = math.cos(math.radians(blk.yaw_deg)), math.sin(math.radians(blk.yaw_deg))
            cos_to_sight = abs((ax * (d - CAM_FWD) + ay * lat) / math.hypot(d - CAM_FWD, lat))
            assert cos_to_sight == pytest.approx(0.0, abs=1e-9), f"长轴与视线余弦 {cos_to_sight:.3e}"

    def test_blocks_tile_the_span_end_to_end(self):
        """n 块的**并集**长度 == 声明跨度(不留缝、也不算重)。

        并集要沿**垂线**量(墙的展开方向),不是沿 `x` —— 垂线方向是 `(−uy, ux)`。
        """
        for car in CARS:
            d, lat, w, _ = car
            r = math.hypot(d - CAM_FWD, lat)
            ux, uy = (d - CAM_FWD) / r, lat / r
            unit_len = OCCLUDER_DIMS["partial"][0]
            _dc, depth_occl = _depths(car)
            span = max(OCCLUDER_COVER * required_span(w, d - CAM_FWD, depth_occl), unit_len)
            ends = [
                -p[0] * uy + p[1] * ux
                for s in _slots("partial", car)
                for p in TestSightLineVsForward._corners(s)
            ]
            assert max(ends) - min(ends) == pytest.approx(span, rel=1e-9)

    def test_occluder_depth_matches_the_placed_blocks(self):
        """闭式解的深度必须等于**实摆各块深度的均值** —— 两处各算一遍就会各错各的。

        用均值而不是"找中间那块":两块时没有 `o=0` 的块,按 `x` 挑最接近的挑到的是
        偏心的那一块(实测差 0.098 m),而那 0.098 会被当成公式错。
        """
        for car in CARS:
            slots = _slots("partial", car)
            mean_depth = sum(s.depth for s in slots) / len(slots)
            assert mean_depth == pytest.approx(occluder_depth(car[0], car[1], gap=OCCLUDER_GAP), abs=1e-9)

    @pytest.mark.parametrize("gap", [-1.0, -1e-9, math.hypot(20.0 - 1.2, 3.5), 100.0])
    def test_bad_gap_raises(self, gap: float):
        """`gap >= 斜距` 会把墙放到相机后面/车体内 —— **必须报错**,不能静默出个奇怪位置。"""
        with pytest.raises(ValueError):
            sight_line_occluders(
                20.0,
                3.5,
                gap=gap,
                unit_len=1.215,
                unit_thick=0.372,
                cover=OCCLUDER_COVER,
                needed_width=1.5,
            )


class TestRequiredSpan:
    def test_it_is_bigger_than_the_car_width_when_the_wall_is_far(self):
        """墙比车**离相机近** ⇒ 同样的视角只要更小的一块(62 m 处:2.007 → 1.908)。"""
        w, depth_car, depth_occl = 2.007, 60.8, occluder_depth(62.0, 3.5, gap=OCCLUDER_GAP)
        assert required_span(w, depth_car, depth_occl) < w
        assert required_span(w, depth_car, depth_occl) == pytest.approx(w * depth_occl / depth_car)

    def test_using_the_raw_width_would_over_cover(self):
        """反向对照:直接拿车宽当跨度会**多盖** ~5% —— 少吃的是车道净空。"""
        w, depth_car = 2.007, 60.8
        over = w / required_span(w, depth_car, occluder_depth(62.0, 3.5, gap=OCCLUDER_GAP)) - 1.0
        assert 0.02 < over < 0.15, f"多盖 {over:.1%}"


class TestProjection:
    """三个量的方向与单调性 —— 单调性反了,调 `--occluder-gap` 会朝反方向走。"""

    _KW = {
        "focal_px": 621.0,
        "cam_z": CAM_Z,
        "occl_top_z": 1.069,
        "car_top_z": 1.48,
        "depth_car": 18.8,
    }

    def test_taller_occluder_hides_more(self):
        lo = project(depth_occl=16.0, **self._KW)
        hi = project(
            occl_top_z=1.86, depth_occl=16.0, **{k: v for k, v in self._KW.items() if k != "occl_top_z"}
        )
        assert hi.visible_px < lo.visible_px
        assert hi.visible_frac < lo.visible_frac

    def test_larger_gap_hides_less(self):
        """★ 方向判据:墙离车越远 ⇒ 离相机越近,但它要盖住的那段俯角**反而更窄**。

        实测 20 m 处:gap=3 ⇒ 可见 23.4 px,gap=6 ⇒ 28.6 px。反了的话 `--occluder-gap`
        就是个陷阱 —— 现场想"挡得更多"的人会把旋钮拧向相反一侧。
        """
        near = project(depth_occl=18.8 - 3.0, **self._KW)
        far = project(depth_occl=18.8 - 6.0, **self._KW)
        assert far.visible_px > near.visible_px

    def test_wall_above_camera_and_car_is_total(self):
        """★ **墙顶高于相机、且高于车顶** ⇒ 恒 0,与车多远都无关。

        两个条件缺一不可:墙顶 1.86 m 只要高过车顶,从地面到车顶的每一条视线都在墙后
        落地 ⇒ 全遮。这是 `full` 档当阳性对照的全部依据,而 P1 四台车(1.30–1.56 m)
        全在 1.86 m 之下。末组 top=1.8 是贴着墙顶的边界。
        """
        for depth_car, top in ((5.0, 1.4), (18.8, 1.0), (60.8, 1.6), (30.0, 1.8)):
            kw = {**self._KW, "occl_top_z": 1.86, "car_top_z": top, "depth_car": depth_car}
            assert project(depth_occl=depth_car - 3.0, **kw).visible_frac == 0.0, f"d={depth_car} top={top}"

    def test_where_the_leak_starts_is_set_by_the_wall_being_closer(self):
        """★ 要漏光,车顶得高过**一个比墙顶更高的阈值** —— 因为墙离相机更近。

        `(cam_z − occl_top)/d_occl > (cam_z − car_top)/d_car`
        ⇒ `car_top > 1.65 + 0.21 × 30/27 = **1.8833 m**`。

        所以"车顶 1.87 > 墙顶 1.86"**并不漏**(第一版就是这么写错的)。这条同时钉住
        `full` 档为什么对 1.30–1.56 m 的车恒 0 —— 离阈值还差 0.3 m 以上。
        """
        base = {k: v for k, v in self._KW.items() if k != "car_top_z"}
        kw = {**base, "occl_top_z": 1.86, "depth_car": 30.0, "depth_occl": 27.0}
        thr = CAM_Z - (CAM_Z - 1.86) * 30.0 / 27.0
        assert thr == pytest.approx(1.8833, abs=1e-4)
        assert project(car_top_z=thr - 0.02, **kw).visible_frac == 0.0
        assert project(car_top_z=thr + 0.02, **kw).visible_frac > 0.0

    def test_no_occluder_is_one(self):
        """墙顶低到车底以下 ⇒ 一点没挡(钳位不许把"没挡"写成 >1 或负数)。"""
        assert (
            visible_fraction(cam_z=CAM_Z, occl_top_z=0.0, car_top_z=1.48, depth_car=18.8, depth_occl=16.0)
            == 1.0
        )

    def test_tall_car_over_a_lower_wall_keeps_only_its_top_slice(self):
        """★ 车顶高过相机、**且高过墙顶**(公交那种)⇒ 只剩车顶那条 —— 公式照样成立。

        第一版把"车顶高过相机"当成退化去特判(返回 1.0),这里实测是 **21.5%**:
        公式本来就对,错的是那个特判。
        """
        f = visible_fraction(cam_z=CAM_Z, occl_top_z=1.86, car_top_z=2.4, depth_car=30.0, depth_occl=27.0)
        assert f == pytest.approx(0.2154, abs=5e-4)
        assert f == pytest.approx(
            (math.atan2(CAM_Z - 1.86, 27.0) - math.atan2(CAM_Z - 2.4, 30.0))
            / (math.atan2(CAM_Z, 30.0) - math.atan2(CAM_Z - 2.4, 30.0))
        )

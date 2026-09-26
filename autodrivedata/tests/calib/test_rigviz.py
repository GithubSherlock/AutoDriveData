"""配置图与覆盖表的回归钉(纯值:PIL + numpy,不 import carla)。

**为什么要有这个文件(2026-09-23,Plan2.md §P-M.9)**:`outputs/calib_check/rig_layout_*.png`
与 `views_*.png` 是**给人看的交付物**,而"图看着对"从来不是判据 —— 本文件把图里
**能用数值验的部分**钉住:

| 钉什么 | 怎么钉 |
|---|---|
| 覆盖表本身(盲区/重叠的度数) | 与设计预算逐项相等(wide 15.1561° / 官方 0°),不是"大概" |
| 图**确实把几何画出来了** | 盲区红弧像素数 0(官方)vs 868(wide);重叠橙弧两代都 > 0 |
| `rigviz.azimuth_of` 与 `camera_azimuth_nus` | 两套**独立实现**必须给出同一个数(交叉验证,不是自己对自己) |
| 底尺的跨 0° 画法 | 踩过的崩:`360.0 % 360 == 0` 会把扇区右端折回最左 ⇒ `rectangle` 抛 `ValueError` |
| 底尺读数 = 覆盖表读数 | 盲区红**列数** / 尺宽 ≈ Σ盲区度数 / 360°(图上量与表上量自洽) |

**不钉什么**:文字排版、有无错别字、标签位置。那些是目检的事,写成断言只会得到
"改个字就红"的脆测试。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from autodrivedata.calib import viz_rig_check as vrc  # noqa: E402
from autodrivedata.calib.camera_rig import (  # noqa: E402
    camera_azimuth_nus,
    coverage_table,
)
from autodrivedata.calib.rigviz import (  # noqa: E402
    CAM_COLOR,
    CAM_SHORT,
    FOOTER,
    FOOTER_LH,
    FOOTER_SIZE,
    GAP_COLOR,
    OVERLAP_COLOR,
    azimuth_of,
    draw_rig_layout,
)
from autodrivedata.gt.export.nuscenes import camera_calibs, camera_fov  # noqa: E402

# 进包后不再需要 sys.path 引导(旧 bin/ 非包布局的产物)
from autodrivedata.utils import fonts  # noqa: E402

RIGS = ("nuscenes", "wide")
# wide rig 的设计预算(§P-M.9):三个盲区 + 合计 + 覆盖率。改 rig 就必须改这里,不许放宽。
WIDE_GAPS = ((82.6647, 90.0, 7.3353), (270.0, 276.0984, 6.0984), (331.0984, 332.8207, 1.7224))
WIDE_GAP_TOTAL = 15.1561
WIDE_COVERAGE = 0.9579
WIDE_OVERLAPS = {
    ("CAM_BACK", "CAM_BACK_LEFT"): 80.0,
    ("CAM_BACK", "CAM_BACK_RIGHT"): 80.0,
    ("CAM_BACK_LEFT", "CAM_BACK_RIGHT"): 40.0,
    ("CAM_FRONT", "CAM_FRONT_LEFT"): 0.1561,
}


def _cov(rig: str) -> dict:
    return coverage_table(camera_calibs(rig), camera_fov(rig))


def _mask(img: Image.Image, color: tuple[int, int, int]) -> np.ndarray:
    return (np.array(img) == np.array(color)).all(-1)


# ---------------------------------------------------------------- 覆盖表


class TestCoverageTable:
    def test_wide_gaps_match_the_design_budget(self):
        """三个盲区的**位置与度数**逐个相等(纯几何量,与相机名/字典序无关)。"""
        got = [(g["lo"], g["hi"], g["deg"]) for g in _cov("wide")["gaps"]]
        assert len(got) == len(WIDE_GAPS)
        for (lo, hi, deg), (wlo, whi, wdeg) in zip(got, WIDE_GAPS, strict=True):
            assert (lo, hi) == pytest.approx((wlo, whi), abs=1e-3)
            assert deg == pytest.approx(wdeg, abs=1e-3)

    def test_wide_totals(self):
        cov = _cov("wide")
        assert cov["gap_total_deg"] == pytest.approx(WIDE_GAP_TOTAL, abs=1e-3)
        assert cov["coverage_frac"] == pytest.approx(WIDE_COVERAGE, abs=1e-4)
        # 容差 2e-3 而非 1e-9:盲区端点落盘前按 4 位小数取整(3 段 6 个端点),
        # 而 `covered_deg` 由未取整的区间求和 ⇒ 两者相加与 360° 差 ~1e-4 是**取整**造成的
        assert cov["covered_deg"] + cov["gap_total_deg"] == pytest.approx(360.0, abs=2e-3)

    def test_wide_pairs_are_the_expected_ones(self):
        """wide 只有**四个**正重叠:后三路互相 80/80/40,前视对 FL↔F 只剩 0.1561°(设计代价)。"""
        got = {(p["a"], p["b"]): p["overlap_deg"] for p in _cov("wide")["pairs"] if p["overlap_deg"] > 0}
        assert set(got) == set(WIDE_OVERLAPS)
        for pair, deg in WIDE_OVERLAPS.items():
            assert got[pair] == pytest.approx(deg, abs=1e-3)

    def test_official_rig_covers_the_full_circle(self):
        """官方 rig 方位轴无盲区 —— 它是"另一条基线",不是本次要改的对象。"""
        cov = _cov("nuscenes")
        assert cov["gaps"] == []
        assert cov["coverage_frac"] == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("rig", RIGS)
    def test_pair_spans_lie_inside_both_sectors(self, rig: str):
        """重叠**区间**必须落在两个相机各自的扇区里(共视探针按它摆锥,错了就摆到画幅外)。"""
        cov = _cov(rig)
        for p in cov["pairs"]:
            if p["span"] is None:
                assert p["overlap_deg"] == 0.0
                continue
            lo, hi = p["span"]
            for name in (p["a"], p["b"]):
                assert any(slo <= lo + 1e-6 and hi <= shi + 1e-6 for slo, shi in cov["cameras"][name]["span"])

    @pytest.mark.parametrize("rig", RIGS)
    def test_gap_total_equals_sum_of_gaps(self, rig: str):
        cov = _cov(rig)
        assert cov["gap_total_deg"] == pytest.approx(sum(g["deg"] for g in cov["gaps"]), abs=1e-6)


# ---------------------------------------------------------------- 独立实现交叉验证


class TestAzimuthIndependentImplementation:
    @pytest.mark.parametrize("rig", RIGS)
    def test_rigviz_matches_camera_rig(self, rig: str):
        """`rigviz.azimuth_of` 与 `camera_rig.camera_azimuth_nus` 是两套独立实现。

        钉它们相等 = 图上标的角度与覆盖表算的角度是同一个数;任一处改了四元数展开顺序,
        这里会红(而"图看着还是像个锥"不会)。
        """
        cal = camera_calibs(rig)
        for name in cal:
            assert azimuth_of(cal, name) == pytest.approx(camera_azimuth_nus(name, cal), abs=1e-9)

    @pytest.mark.parametrize("rig", RIGS)
    def test_coverage_azimuth_matches_table(self, rig: str):
        cal = camera_calibs(rig)
        for name, c in _cov(rig)["cameras"].items():
            assert c["az_nus_deg"] == pytest.approx(camera_azimuth_nus(name, cal), abs=1e-9)


# ---------------------------------------------------------------- 配置图


class TestRigLayoutFigure:
    @pytest.mark.parametrize("rig", RIGS)
    def test_canvas_is_rgb_of_the_requested_size(self, rig: str):
        img = draw_rig_layout(camera_calibs(rig), camera_fov(rig), _cov(rig), f"t {rig}", "s", (1800, 1200))
        assert img.size == (1800, 1200)
        assert img.mode == "RGB"

    def test_gap_arc_only_when_there_is_a_gap(self):
        """盲区红弧是**几何的**读数:官方 rig 0 个盲区 ⇒ 图里一个红像素都不该有;

        wide 有 15.156° 盲区 ⇒ 必须有。只断言"有/无"不断言像素数:环是非线性的(半径固定、
        弧长 ∝ 度数),像素数还受线宽影响,写死数字只会在改线宽时变成噪声。
        """
        assert not _mask(_draw("nuscenes"), GAP_COLOR).any()
        assert _mask(_draw("wide"), GAP_COLOR).sum() > 0

    @pytest.mark.parametrize("rig", RIGS)
    def test_overlap_arc_present_for_both_rigs(self, rig: str):
        """两代 rig 都有相邻重叠(官方 6 对 / wide 4 对),橙弧必须画出来。"""
        assert _mask(_draw(rig), OVERLAP_COLOR).sum() > 0

    @pytest.mark.parametrize("rig", RIGS)
    def test_every_camera_has_a_colour_and_a_short_code(self, rig: str):
        """六个通道都要有画色与短码 —— 缺一个就是图上少一路,而"少一路"在图里看不出来。"""
        assert set(camera_fov(rig)) == set(CAM_COLOR) == set(CAM_SHORT)

    def test_footer_fits_inside_the_canvas(self):
        """★ 页脚是按实测宽折行后画的:**折完每一行都必须装得下**。

        踩过的坑:页脚曾是单行 `draw_text`,1800 px 画布装不下 —— PIL 不报错、**静默裁掉**,
        图上看着像"这行写完了"。所以判据是"折行后的实测宽 ≤ 画布 − 左右边距",不是
        "页脚里有几个字"。用 `FOOTER` 常量而非就地字面量,正是为了让这条断言够得着原文。
        """
        canvas_w = 1800
        lines = fonts.wrap(FOOTER, FOOTER_SIZE, canvas_w - 80)
        assert len(lines) >= 2  # 单行就是没折,说明 `wrap` 没被用上
        for line in lines:
            assert fonts.width(line, FOOTER_SIZE) <= canvas_w - 80
        # 折行后逐行从底边往上排:最上面一行仍要在数字表(底 1100)之下
        top = 1200 - 34 - (len(lines) - 1) * FOOTER_LH
        assert top > 1100


def _draw(rig: str) -> Image.Image:
    return draw_rig_layout(camera_calibs(rig), camera_fov(rig), _cov(rig), f"t {rig}")


# ---------------------------------------------------------------- 底尺(views 图)


class TestRulerLanes:
    """`viz_rig_check.ruler_image` 的纯值回归(它不 spawn 任何东西,可直接调)。"""

    WIDTH = 2400

    def test_wraps_at_zero_does_not_crash(self):
        """★ 踩过的崩:`CAM_FRONT` 的扇区是 `(332.82, 360.0)` + `(0.0, 27.82)`,
        `az % 360` 把 `360.0` 折成 0 ⇒ `rectangle` 拿到 x1<x0 直接抛 `ValueError`
        (不是画错,是崩)。wide rig 正好有这一路,故这里必须过。
        """
        img = vrc.ruler_image("wide", self.WIDTH)
        assert img.size == (self.WIDTH, vrc.RULER_H)

    @pytest.mark.parametrize("rig", RIGS)
    def test_gap_columns_agree_with_the_coverage_table(self, rig: str):
        """盲区红**列数** / 尺宽 ≈ Σ盲区角度 / 360° —— 图与表自洽(不是"画了个红条")。

        底尺里只有并集行用 `GAP_COLOR`,泳道用的是逐通道纯色(含 CAM_BACK_LEFT 的
        (200,60,60),与盲区红 (220,40,40) 不同值)⇒ 精确匹配只命中盲区。

        **容差是 6% 相对值,不是 ±2 px**:红带描了 1 px 的 `INK` 边框(每段两侧各吃 1 列),
        加上 `rectangle` 把浮点框取整,实测比"度数 × 尺宽 / 360°"少 ~3%。这条钉的是
        "图与表同一个量级(4.2% 的圆周)",不是像素级相等 —— 像素级相等只会在改线宽时变噪声。
        """
        img = vrc.ruler_image(rig, self.WIDTH)
        cols = int(np.unique(np.where(_mask(img, GAP_COLOR))[1]).size)
        span = self.WIDTH - 40 - 78  # 与 ruler_image 的 pad / left 一致
        want = span * _cov(rig)["gap_total_deg"] / 360.0
        assert cols == pytest.approx(want, rel=0.06)

    @pytest.mark.parametrize("rig", RIGS)
    def test_every_lane_is_drawn(self, rig: str):
        """六个泳道各自成行:每路纯色都要在图上出现(漏画一路 = 少一行,肉眼易忽略)。"""
        img = vrc.ruler_image(rig, self.WIDTH)
        for name, col in CAM_COLOR.items():
            assert _mask(img, col).sum() > 0, f"{name} 的泳道没画出来"

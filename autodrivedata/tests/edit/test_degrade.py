"""`edit/degrade` + `edit/noise_curve` 的判据回归钉。

这一层的两条要害:
1. **强度 0 必须是恒等** —— 否则"注入"这个动作本身就在改数据,后面所有 Δ 不可归因;
2. **逐帧 seed 固定** —— 否则"两帧之间的差"里混着注入自身的随机性,
   而那会被读成"时序抖动"(P3 正要量那个)。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.edit import degrade as DG
from autodrivedata.edit import noise_curve as NC


def _img(h=32, w=32):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _checker(h=64, w=64, period=2):
    """高频棋盘 —— 模糊会让它的方差**单调下降**,是"真的糊了"的可测形态。"""
    y, x = np.mgrid[0:h, 0:w]
    v = (((x // period + y // period) % 2) * 255).astype(np.uint8)
    return np.stack([v] * 3, axis=2)


class TestIdentity:
    def test_blur_identity_at_low_k(self):
        a = _img()
        np.testing.assert_array_equal(DG.motion_blur(a, 1), a)
        np.testing.assert_array_equal(DG.motion_blur(a, 0), a)

    def test_noise_identity_at_zero_sigma(self):
        a = _img()
        np.testing.assert_array_equal(DG.gaussian_noise(a, 0.0, seed=1), a)

    def test_rain_identity_at_zero(self):
        a = _img()
        np.testing.assert_array_equal(DG.rain_streaks(a, 0, seed=1), a)

    @pytest.mark.parametrize("kind,zero", [("blur", 1), ("noise", 0), ("rain", 0), ("fog", 0)])
    def test_apply_identity_at_floor(self, kind, zero):
        """★ 曲线的**零档必须与基线逐位相同** —— 这是整条曲线的地基。"""
        a = _img()
        np.testing.assert_array_equal(DG.apply_degradation(a, kind, zero, seed=7), a)


class TestReproducible:
    def test_same_seed_bitwise(self):
        a = _img()
        for kind, lv in (("noise", 25), ("rain", 800)):
            np.testing.assert_array_equal(
                DG.apply_degradation(a, kind, lv, seed=3), DG.apply_degradation(a, kind, lv, seed=3)
            )

    def test_different_seed_differs(self):
        """反向对照:换 seed 必须换图,否则"逐帧固定"这句话没有内容。"""
        a = _img()
        assert not np.array_equal(
            DG.apply_degradation(a, "noise", 25, seed=3), DG.apply_degradation(a, "noise", 25, seed=4)
        )

    def test_blur_is_deterministic(self):
        """模糊没有随机性 ⇒ 同 level 必须逐位相同(与 seed 无关)。"""
        a = _img()
        np.testing.assert_array_equal(DG.motion_blur(a, 7), DG.motion_blur(a, 7))


class TestActuallyDegrades:
    def test_blur_reduces_high_frequency_variance_monotonically(self):
        """★★ **反向自证**:棋盘的高频方差必须随核长**单调下降**。

        若这条不成立,`motion_blur` 可能什么都没做(而"Δ≈0"会被读成
        "模型对运动模糊鲁棒" —— 与 §1.6 那个"假退化"同一种误读)。
        """
        c = _checker()
        vs = [float(np.var(DG.motion_blur(c, k)[..., 0])) for k in (1, 3, 7, 15)]
        assert vs == sorted(vs, reverse=True), f"方差应当单调下降,实测 {vs}"
        assert vs[-1] < 0.2 * vs[0], "最大档应当把高频基本抹掉"

    def test_noise_increases_variance_on_mid_gray(self):
        """★ 用**中灰底**量 —— 不能用棋盘。

        第一版拿棋盘(0/255 饱和)去量,当场红:**加噪声再 clip 到 0/255 只会把方差拉低**
        (两极已经到顶了)。**红的是测试的口径不是代码** —— 换到中灰底(不触顶)才量得对。
        """
        g = np.full((64, 64, 3), 128, np.uint8)
        v0 = float(np.var(g))
        v1 = float(np.var(DG.gaussian_noise(g, 50, seed=0)))
        assert v1 > v0 * 10, f"σ=50 应当显著抬高方差,实测 {v0:.1f} → {v1:.1f}"

    def test_fog_veils_toward_airlight_monotonically(self):
        """★★ 合成雾:**亮度必须随遮蔽系数单调上升**,且 k=1 时趋近大气光。

        这条是它能当"可调退化"的前提 —— 不单调就没法用它比"强度"。
        """
        a = _img()
        m = [float(DG.veil_fog(a, k).mean()) for k in (0.0, 0.2, 0.4, 0.6, 1.0)]
        assert m == sorted(m), f"亮度应当单调上升,实测 {m}"
        assert abs(m[-1] - 230) < 1.0, "k=1 应当全幅趋近大气光"

    def test_atmospheric_fog_is_distance_dependent(self):
        """★★ **这是本模块的核心判据**:同一张图,`atmospheric_fog` 必须

        ① 让**远处**(d 大)比**近处**(d 小)被遮得更狠 —— 这正是它与 `veil_fog` 的分水岭,
           也是实测里"全局遮蔽掉不动、距离相关掉 0.26"的机制;
        ② 随 β 单调(β 大 ⇒ 更白)。
        """
        img = np.full((8, 8, 3), 0, np.uint8)  # 全黑底:遮出来的亮度就是(1−t)·A
        d = np.tile(np.linspace(2, 80, 8, dtype=np.float32), (8, 1))  # 左近右远
        far = DG.atmospheric_fog(img, d, 0.05)
        assert far[:, -1].mean() > far[:, 0].mean() + 50, "远端应当被遮得明显更狠"
        lv = [float(DG.atmospheric_fog(img, d, b).mean()) for b in (0.02, 0.05, 0.10)]
        assert lv == sorted(lv), f"应当随 β 单调变白,实测 {lv}"

    def test_atmospheric_fog_identity_at_zero_beta(self):
        a = _img()
        d = np.full((32, 32), 30.0, np.float32)
        np.testing.assert_array_equal(DG.atmospheric_fog(a, d, 0.0), a)

    def test_atmospheric_fog_shape_mismatch_raises(self):
        """深度与图不同画幅 ⇒ 抛。静默广播会把雾按错误的几何铺上去。"""
        with pytest.raises(ValueError, match="不同画幅"):
            DG.atmospheric_fog(_img(8, 8), np.zeros((4, 4), np.float32), 0.05)

    def test_rain_changes_pixels(self):
        a = _img()
        assert not np.array_equal(DG.rain_streaks(a, 500, seed=0), a)


class TestGuards:
    def test_unknown_kind_raises(self):
        """静默返回原图 = 假阴性(读者会以为"这个退化不掉点")。"""
        with pytest.raises(KeyError, match="未知注入"):
            DG.apply_degradation(_img(), "jpeg", 10)

    def test_degrade_root_keeps_gt_untouched(self, tmp_path):
        from PIL import Image

        src = tmp_path / "src"
        (src / "training/image_2").mkdir(parents=True)
        (src / "training/label_2").mkdir(parents=True)
        for i in range(3):
            Image.fromarray(_img(8, 8)).save(src / "training/image_2" / f"{i:06d}.png")
            (src / "training/label_2" / f"{i:06d}.txt").write_text(f"Car 0 0 0 0 0 1 1 1 1 1 0 0 0 0 {i}\n")
        dst = tmp_path / "dst"
        assert DG.degrade_root(src, dst, kind="blur", level=3) == 3
        for i in range(3):
            assert (dst / f"training/label_2/{i:06d}.txt").read_text() == (
                src / f"training/label_2/{i:06d}.txt"
            ).read_text()

    def test_degrade_root_mismatch_raises(self, tmp_path):
        from PIL import Image

        src = tmp_path / "s"
        (src / "training/image_2").mkdir(parents=True)
        (src / "training/label_2").mkdir(parents=True)
        Image.fromarray(_img(8, 8)).save(src / "training/image_2/000000.png")
        with pytest.raises(SystemExit, match="不规整"):
            DG.degrade_root(src, tmp_path / "d", kind="blur", level=3)


class TestLevelOverrideWiring:
    """★ **`--levels` 必须真的传到 `run_curve`** —— 2026-10-06 实测踩到。

    加完参数与解析函数之后,`main` 里那一行**没改上**(替换字符串被 ruff 重排过),
    于是 `--levels fogdepth=0.04,...` 跑出来**还是默认的三档** ——
    **不报错、不提示**,只是你给的值被静默丢掉。判据走 AST:源码级,跑一次也不报错的那种。
    """

    def test_main_passes_levels_to_run_curve(self):
        import ast
        from pathlib import Path as _P

        src = (_P(__file__).resolve().parents[2] / "edit" / "noise_curve.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        calls = [
            n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "run_curve"
        ]
        assert calls, "找不到 run_curve 调用"
        assert any(any(k.arg == "levels" for k in c.keywords) for c in calls), (
            "`main` 没把 `levels` 传给 `run_curve` —— `--levels` 会被静默忽略"
        )


class TestLevelOverride:
    def test_empty_is_none(self):
        assert NC._parse_levels("  ") is None

    def test_parses_multi_kind(self):
        got = NC._parse_levels("fogdepth=0.05,0.07; blur=3,9")
        assert got == {"fogdepth": [0.05, 0.07], "blur": [3.0, 9.0]}

    def test_unknown_kind_raises(self):
        with pytest.raises(SystemExit, match="不认识"):
            NC._parse_levels("jpeg=1")

    def test_bad_chunk_raises(self):
        with pytest.raises(SystemExit, match="每一段"):
            NC._parse_levels("fogdepth")


class TestFitEquivalent:
    """曲线的**唯一用途**:把别的退化投影上来。"""

    CURVE = [
        {"level": 3.0, "delta": -0.05},
        {"level": 7.0, "delta": -0.20},
        {"level": 15.0, "delta": -0.60},
    ]

    def test_interpolates_inside(self):
        assert NC.fit_equivalent_level(self.CURVE, -0.20) == pytest.approx(7.0)
        assert NC.fit_equivalent_level(self.CURVE, -0.125) == pytest.approx(5.0)

    def test_beyond_max_returns_max(self):
        assert NC.fit_equivalent_level(self.CURVE, -0.90) == pytest.approx(15.0)

    def test_lighter_than_lightest_is_none(self):
        """★★ **最关键的一支**:`Δ_gen = +0.010` 比最轻的一档还轻 ⇒ **落不上曲线**。

        返回 None 本身就是结论:**生成的"雾"不是这段曲线上的任何一档** ——
        它压根不在退化侧。这与 §1.6 的目检(物体仍然清晰)是同一条。
        """
        assert NC.fit_equivalent_level(self.CURVE, 0.010) is None
        assert NC.fit_equivalent_level(self.CURVE, -0.01) is None

    def test_empty_curve_is_none(self):
        assert NC.fit_equivalent_level([], -0.2) is None

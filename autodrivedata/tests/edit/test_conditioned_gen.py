"""`edit/conditioned_gen` 的判据回归钉(**不出模型、不下权重**)。

这一层最容易出的错是**条件源张冠李戴**:`gt-depth` 走错分支、忘了方裁、
或把某一帧的条件用到另一帧上 —— 三种都**不报错**,只是生成图跟条件无关,
而"条件没进去"与"这个模型就这水平"在肉眼上分不开。⇒ 用**可判定的等价关系**钉住。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.edit import conditioned_gen as G
from autodrivedata.edit import depth_cond as DC


def _depth(h=16, w=32, near=2.0, far=50.0):
    """上远下近的地平面深度(米)。"""
    return np.linspace(far, near, h, dtype=np.float32)[:, None] * np.ones((1, w), dtype=np.float32)


class TestBuildConditionDepth:
    def test_gt_depth_matches_depth_cond_when_already_cond_size(self):
        """★ `gt-depth` 必须**逐位等于** `depth_cond.metric_depth_to_cond`。

        用已经等于 `COND_OUT` 的方图 ⇒ `_to_cond_size` 是空操作 ⇒ 能逐位对。
        这条是"两个模块同口径"的硬钉:任一侧改了裁剪/换算/重采样,这里当场红。
        """
        n = G.COND_OUT
        d = _depth(h=n, w=n)
        got = G.build_condition("gt-depth", rgb=np.zeros((n, n, 3), np.uint8), gt_depth=d)
        np.testing.assert_array_equal(got, DC.metric_depth_to_cond(d.astype(np.float64)))

    def test_square_crop_happens_before_conversion(self):
        """长条输入必须**中心方裁**后再换算 —— 裁错边会拿另一块像素当条件。"""
        n = G.COND_OUT
        d = _depth(h=n, w=3 * n)
        got = G.build_condition("gt-depth", rgb=np.zeros((n, 3 * n, 3), np.uint8), gt_depth=d)
        assert got.shape == (n, n)
        # 中心那一块应当与整张裁出来的一致
        np.testing.assert_array_equal(got, DC.metric_depth_to_cond(d[:, n : 2 * n].astype(np.float64)))

    def test_requires_depth(self):
        with pytest.raises(ValueError, match="需要传 gt_depth"):
            G.build_condition("gt-depth", rgb=np.zeros((4, 4, 3), np.uint8))

    def test_blank_is_a_constant_grey_condition(self):
        """★ `blank` 是**对照组**(没有结构约束),不是"一个更弱的条件"。

        它必须是**常量灰**而不是全黑:全黑会被判成退化条件,且 SD 对全黑的响应不可控。
        """
        out = G.build_condition("blank", rgb=np.zeros((97, 401, 3), np.uint8))
        assert out.shape == (G.COND_OUT, G.COND_OUT)
        assert out.dtype == np.uint8
        assert (out == 128).all()

    def test_unknown_kind_raises(self):
        with pytest.raises(KeyError, match="未接的条件源"):
            G.build_condition("normal", rgb=np.zeros((4, 4, 3), np.uint8))


class TestCondSizeUnification:
    def test_every_kind_returns_cond_out(self):
        """★★ **三条路的尺寸必须一致**。不齐的话保真度判据要么抛、要么被"顺手 resize"糊过去。

        这条是 §"三个条件源统一到 COND_OUT" 那条设计决定的**可执行版本**。
        """

        class FakeDet:
            def __call__(self, rgb):
                return np.zeros(rgb.shape[:2], np.uint8), np.zeros_like(rgb)

        rgb = np.zeros((97, 401, 3), np.uint8)  # 故意用非 64 倍数、非方
        d = _depth(h=97, w=401)
        for kind, kw in (
            ("gt-depth", {"gt_depth": d}),
            ("midas-depth", {"midas_detector": FakeDet()}),
            ("canny", {}),
        ):
            out = G.build_condition(kind, rgb=rgb, **kw)
            assert out.shape == (G.COND_OUT, G.COND_OUT), f"{kind} 尺寸不对:{out.shape}"
            assert out.dtype == np.uint8

    def test_already_sized_is_not_resampled(self):
        a = np.arange(G.COND_OUT * G.COND_OUT, dtype=np.uint8).reshape(G.COND_OUT, G.COND_OUT)
        assert G._to_cond_size(a) is a


class TestBuildConditionMidas:
    def test_detector_receives_resized_image(self):
        """★ 假 detector 直接问"你收到了什么形状"。

        期望 **512×512**(不是输入的 10×40,也不是方裁后的 10×10)——
        因为 `MidasDetector` **自己不 resize**,必须先过官方的 `resize_image(...,512)`。
        这条是适配第六处的可执行版本:**去掉那一步这里当场红**。
        """

        class FakeDet:
            def __init__(self):
                self.seen = None

            def __call__(self, rgb):
                self.seen = rgb.shape
                return np.zeros(rgb.shape[:2], np.uint8), np.zeros_like(rgb)

        det = FakeDet()
        G.build_condition("midas-depth", rgb=np.zeros((10, 40, 3), np.uint8), midas_detector=det)
        assert det.seen == (512, 512, 3), f"detector 收到的应是 resize 后的方图,实际 {det.seen}"

    def test_requires_detector(self):
        with pytest.raises(ValueError, match="已构造的 detector"):
            G.build_condition("midas-depth", rgb=np.zeros((4, 4, 3), np.uint8))


class TestGenerateGuards:
    def test_non_square_raises_before_touching_model(self):
        """方图检查必须在**用模型之前** —— 否则会先花 100 s 建模再报形状错。"""
        with pytest.raises(ValueError, match="必须是方的"):
            G.generate(None, np.zeros((8, 16), np.uint8))

    def test_hint_is_three_channel(self):
        """★★ **反向自证**:hint 张量必须 3 通道。

        ControlNet 的 `input_hint_block` 首层是 `Conv2d(3,16,3)`。条件源是**灰度**的,
        官方 demo 靠 `HWC3()` 复制成三通道 —— 漏掉这一步会在**很深的地方**报
        `expected input[2, 1, 512, 512] to have 3 channels`,**不提"条件图是单通道"**。
        本轮真的漏了(跑起来才炸),所以钉在这里。

        用假 cldm 直接截住喂进 sampler 的 control 张量,不问 torch 要模型。
        """

        class FakeSampler:
            def __init__(self):
                self.control = None

            def sample(self, steps, n, shape, c, **kw):
                self.control = c["c_concat"][0]
                raise _Stop()

        class _Stop(Exception):
            pass

        class FakeModel:
            def get_learned_conditioning(self, prompts):
                return prompts

        class FakeCldm:
            device = "cpu"

            def __init__(self):
                self.sampler = FakeSampler()

            @property
            def model(self):
                return FakeModel()

        fake = FakeCldm()
        with pytest.raises(_Stop):
            G.generate(fake, np.full((64, 64), 128, np.uint8))
        assert fake.sampler.control.shape == (2, 3, 64, 64), (
            f"hint 应当是 (n,3,h,w),实际 {tuple(fake.sampler.control.shape)}"
        )


class TestFidelityGuard:
    def test_requires_detector(self):
        with pytest.raises(ValueError, match="需要 MiDaS detector"):
            G.condition_fidelity(
                np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.uint8), midas_detector=None
            )


class TestFrameIo:
    def test_rgb_missing_capture_raises(self, tmp_path):
        with pytest.raises(SystemExit, match="不是"):
            G.load_frame_rgb(tmp_path, 0)

    def test_rgb_reads(self, tmp_path):
        from PIL import Image

        p = tmp_path / "capture" / "images" / "p0"
        p.mkdir(parents=True)
        Image.fromarray(np.zeros((5, 7, 3), np.uint8)).save(p / "00003.png")
        assert G.load_frame_rgb(tmp_path, 3).shape == (5, 7, 3)


class TestSavePng:
    def test_uint8_roundtrip(self, tmp_path):
        from PIL import Image

        a = np.arange(16, dtype=np.uint8).reshape(4, 4)
        p = tmp_path / "c.png"
        G.save_png(p, a)
        np.testing.assert_array_equal(np.array(Image.open(p)), a)

    def test_float_is_clipped_not_wrapped(self, tmp_path):
        """★ 非 uint8 输入必须 **clip** 而不是截断成低 8 位 ——
        `astype(uint8)` 会把 256 变成 0,图会**静默反相**(黑变白),看不出是错的。"""
        from PIL import Image

        p = tmp_path / "c.png"
        G.save_png(p, np.full((2, 2), 300.0))
        np.testing.assert_array_equal(np.array(Image.open(p)), np.full((2, 2), 255, np.uint8))
        G.save_png(p, np.full((2, 2), -5.0))
        np.testing.assert_array_equal(np.array(Image.open(p)), np.zeros((2, 2), np.uint8))

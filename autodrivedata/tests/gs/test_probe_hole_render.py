"""`gs/probe_hole_render` 的判据。

★★ 这个模块有**两次**值得留档的翻车,都由它自己的读数抓出来:

1. **它把当天早上刚修的帧索引 bug 又长了一遍** —— `imgs_b[j]` 而不是 `imgs_b[idx][j]`,
   天花板算出 **18 dB** 而真值是 **41 dB**。因是它**绕开了 `three_tier`** 自己写了一份判据。
2. **它的裁决方向写反了** —— 第一版用「两个渲染像不像」判"是不是 A 侧的错",
   而 `B_render ≈ B_img`(天花板 41 dB)⇒ 那两个距离**是同一件事**,区分不了任何东西。

⇒ 本组钉的就是这两条。
"""

from __future__ import annotations

import pytest
import torch

from autodrivedata.gs import probe_hole_render as P


class TestPsnrGuard:
    """★ 翻车①的守卫 —— 形状不对**当场抛**,不许"能算就往下算"。"""

    def test_mismatched_shape_raises_with_the_reason(self):
        a = torch.zeros(4, 6, 3)
        b = torch.zeros(4, 6, 3)
        mask = torch.ones(3, 5, dtype=torch.bool)  # 故意不同
        with pytest.raises(SystemExit, match="按 idx 取"):
            P._psnr(a, b, mask)

    def test_matching_shape_is_fine(self):
        a = torch.zeros(4, 6, 3)
        b = torch.zeros(4, 6, 3)
        assert P._psnr(a, b, torch.ones(4, 6, dtype=torch.bool)) > 100.0

    def test_empty_mask_is_nan_not_zero(self):
        """空掩膜是**未判**(`nan`),不是 0 dB —— 把"没测到"读成"最差"是同族的老坑。"""
        z = torch.zeros(4, 6, 3)
        assert P._psnr(z, z, torch.zeros(4, 6, dtype=torch.bool)) != P._psnr(
            z, z, torch.zeros(4, 6, dtype=torch.bool)
        )


class TestVerdict:
    """★ 翻车②:判据必须是"**天花板可不可达**",不是"两个渲染像不像"。"""

    @staticmethod
    def _rep(ceil, edit, with_prop=None):
        m = {"ceiling_vs_truth": ceil, "edited_vs_truth": edit}
        if with_prop is not None:
            m["edited_vs_with_prop"] = with_prop
        return {"median": m}

    def test_the_real_reading_says_the_problem_is_on_a_side(self):
        """★★ 实测那一档:B 渲到 41 dB 而 A 只有 14 dB ⇒ **A 侧的表示**。"""
        v = P.verdict(self._rep(40.95, 14.20, with_prop=12.49))
        assert "A 侧的表示" in v
        assert "道具还在" in v, "12.49 < 14.20 ⇒ 删完那块仍更像道具 —— 这条必须一起报"

    def test_a_bad_ceiling_is_undecided(self):
        """天花板自己就不行 ⇒ **未判** —— 真值在这块根本达不到,谈不了是谁的错。"""
        assert "未判" in P.verdict(self._rep(3.0, 2.0))

    def test_comparable_values_mean_a_shared_difficulty(self):
        assert "共同" in P.verdict(self._rep(30.0, 29.0))

    def test_the_first_version_would_have_read_it_backwards(self):
        """★★ 反向自证:把旧判据(两个渲染像不像)**喂进新裁决**必须**不会**给出"A 侧"。

        旧的错在于:`B_render ≈ B_img`,所以 `d(A_edit, B_render)` 与 `d(A_edit, B_img)`
        本来就该几乎相等 —— 拿它当"两个模型像不像"用,**恒真**,毫无判别力。
        这里用一个"两个模型确实一致"的读数(两者都离真值 14 dB)验证新裁决不会误判。
        """
        v = P.verdict(self._rep(14.0, 13.9))
        assert "共同" in v
        # ⚠️ 别断言 `"A 侧" not in v` —— "共同"那一档的措辞里**就含**"不是 A 侧特有的缺陷",
        #    子串断言会假红(第一版就这么写的)。要断言的是**裁决本身**。
        assert "问题在" not in v

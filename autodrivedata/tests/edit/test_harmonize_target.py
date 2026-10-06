"""`edit/harmonize_target` 的判据回归钉 —— **真值靶**那一层。

§1.9 的 `harmonize` 只能给代理指标(与真值帧的统计距离,但没有"正确答案")。
这一层用 A/B 采集造出**精确的真值**:把 B 天气的车辆区域贴进 A 天气的同帧,
真值就是 A 那一帧原样。于是判据是"**修复后离真值更近**",不再是代理。

三条要害:
1. ★ **行序**:两家的 `label_2` **行序不同**(CARLA `get_actors()` 跨采集不稳定)——
   逐行 zip 会得到"差 215 px"的假读数。`box_shift` 必须先排序(本仓 2026-10-06 实测踩到);
2. ★ **只改掩膜内**:粘贴与和谐化都不许碰掩膜外的像素(否则"改善"里混着别的东西);
3. ★ **对齐是判据的前提**,不是假设 —— 超限**当场抛**,不许"差不多就行"地算下去。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from autodrivedata.edit import harmonize_target as HT


def _boxes(a, b, c, d):
    return [(a, b, c, d)]


class TestBoxShift:
    """★ 跨 root 比框坐标 —— **必须按多重集比**。"""

    def test_same_boxes_in_a_different_order_read_as_zero(self):
        """★★ 反向自证:同一批框、**行序不同** ⇒ 必须是 0,不是几百 px。

        实测踩到过:两家 root 的框内容一样但行序不同,逐行 zip 比出 **215 px**,
        差一点被读成"两个采集位置不一样"。
        """
        a = _boxes(10, 10, 30, 30) + _boxes(100, 40, 160, 90)
        b = list(reversed(a))
        assert HT.box_shift(a, b) == 0.0

    def test_detects_a_real_shift(self):
        a = _boxes(10, 10, 30, 30)
        b = _boxes(13, 10, 33, 30)
        assert HT.box_shift(a, b) == pytest.approx(3.0)

    def test_count_mismatch_raises(self):
        """条数不等 = "配不上",不是"差得多" —— 混在一起会把口径问题读成对齐问题。"""
        with pytest.raises(ValueError, match="条数不等"):
            HT.box_shift(_boxes(0, 0, 1, 1), [])

    def test_empty_is_zero(self):
        assert HT.box_shift([], []) == 0.0


class TestMaskAndPaste:
    def test_out_of_frame_is_clipped(self):
        m = HT.boxes_to_mask((20, 20), [(-5.0, -5.0, 8.0, 8.0)])
        assert m[:8, :8].all() and m.sum() == 64

    def test_empty_boxes_give_empty_mask(self):
        assert not HT.boxes_to_mask((20, 20), []).any()

    def test_paste_touches_only_the_mask(self):
        """★★ **掩膜外必须逐位不变** —— 否则后面量到的"改善"里混着别的东西。"""
        rng = np.random.default_rng(0)
        ctx = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        src = np.full_like(ctx, 255)
        out = HT.paste(ctx, src, _boxes(5, 5, 15, 15))
        m = HT.boxes_to_mask((32, 32), _boxes(5, 5, 15, 15))
        np.testing.assert_array_equal(out[~m], ctx[~m])
        assert (out[m] == 255).all()

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="不同画幅"):
            HT.paste(np.zeros((8, 8, 3), np.uint8), np.zeros((4, 4, 3), np.uint8), _boxes(0, 0, 2, 2))


def _scene(h=64, w=64, seed=0):
    """有纹理的底图(**别用纯色** —— 纯色下协方差项无定义,测出来的是空转)。"""
    rng = np.random.default_rng(seed)
    return rng.integers(60, 200, (h, w, 3), dtype=np.uint8)


class TestHarmonizePair:
    def test_identity_when_source_is_the_truth(self):
        """★★ **自证**:补丁取自真值那张图本身 ⇒ 合成图**就是**真值 ⇒ 距离必须是 0。

        这条证明的是"管道接对了" —— 没有它,后面所有下降都可能是脚本自己在动。
        """
        ctx = _scene()
        r = HT.harmonize_pair(ctx, ctx, _boxes(20, 20, 40, 40))
        assert r is not None
        assert r["d_before"] == pytest.approx(0.0, abs=1e-9)

    def test_wrong_illumination_patch_is_measurably_wrong_and_fixable(self):
        """★ 造一块**光照对不上**的补丁(整体偏蓝),方法必须把距离拉下来。

        ⚠️ **夹具必须让 `src` 在掩膜外也不同**(这里是整幅偏蓝)—— 第一版只改掩膜内,
        于是"错光源对照"的参考区**与真值区逐位相同**,对照臂退化成方法臂
        (两个读数一模一样,断言当场红)。真实数据里两种天气**整幅都不同**。
        """
        ctx = _scene(seed=1)
        src = np.clip(ctx.astype(np.int16) + np.array([-50, -20, 50]), 0, 255).astype(np.uint8)
        r = HT.harmonize_pair(ctx, src, _boxes(20, 20, 44, 44))
        assert r is not None
        assert r["d_before"] > 0.5, f"这块补丁应当明显对不上,实测 {r['d_before']:.3f}"
        assert r["d_method"] < r["d_before"], "统计对齐应当把距离拉下来"
        # ★ 对照:拿**错光源那张图**的外圈当参考(那是另一种光照),不该有同样的改善
        assert r["d_ctrl_wrong_source"] > r["d_method"]

    def test_empty_boxes_returns_none(self):
        ctx = _scene()
        assert HT.harmonize_pair(ctx, ctx, []) is None


class TestVerdict:
    def _rep(self, **kw):
        base = {"d_before": 3.0, "d_method": 1.0, "d_ctrl_wrong_source": 2.5, "d_ctrl_global": 3.0}
        base.update(kw)
        return {"median": base}

    def test_both_conditions_pass(self):
        assert "真值靶通过" in HT.verdict(self._rep())

    def test_insensitive_is_caught(self):
        assert "不敏感" in HT.verdict(self._rep(d_method=2.99))

    def test_control_not_separated_is_caught(self):
        assert "分不清" in HT.verdict(self._rep(d_method=2.5, d_ctrl_wrong_source=2.52))


class TestRunFramesAlignmentGate:
    @staticmethod
    def _root(tmp_path, name, dx=0.0, n=3):
        root = tmp_path / name
        (root / "training/image_2").mkdir(parents=True)
        (root / "training/label_2").mkdir(parents=True)
        img = _scene(48, 48, seed=hash(name) % 1000)
        for i in range(n):
            Image.fromarray(img).save(root / f"training/image_2/{i:06d}.png")
            (root / f"training/label_2/{i:06d}.txt").write_text(
                f"Car 0 0 0 {10 + dx:.2f} 10 {34 + dx:.2f} 34 1 1 1 1 1 0 0 0 0\n", encoding="utf-8"
            )
        return root

    def test_aligned_pair_runs(self, tmp_path):
        ctx, src = self._root(tmp_path, "ctx"), self._root(tmp_path, "src", dx=0.3)
        rep = HT.run_frames(ctx, src, ["000000", "000001"])
        assert rep["n_frames"] == 2 and rep["max_box_shift_px"] == pytest.approx(0.3)

    def test_misaligned_pair_raises(self, tmp_path):
        """★★ **对齐是前提不是假设** —— 差 10 px 必须当场抛,不许"差不多"地算下去。"""
        ctx, src = self._root(tmp_path, "ctx2"), self._root(tmp_path, "src2", dx=10.0)
        with pytest.raises(SystemExit, match="对齐判据不合格"):
            HT.run_frames(ctx, src, ["000000"])

    def test_frames_without_boxes_are_counted_not_silently_dropped(self, tmp_path):
        ctx, src = self._root(tmp_path, "ctx3"), self._root(tmp_path, "src3")
        (ctx / "training/label_2/000001.txt").write_text("", encoding="utf-8")
        (src / "training/label_2/000001.txt").write_text("", encoding="utf-8")
        rep = HT.run_frames(ctx, src, ["000000", "000001"])
        assert (rep["n_frames"], rep["n_skipped"]) == (1, 1)
        assert rep["skipped"] == ["000001"]

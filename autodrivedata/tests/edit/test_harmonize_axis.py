"""`edit/harmonize_axis` 的判据 —— **收口判据**,所以它自己必须先站得住。

这一层有两条要害,都是**实测踩出来的**:

1. ★ **"抄周围均值"必须真的只在掩膜内改** —— 与 `reinhard_transfer_masked` 同一条纪律:
   RGB→LAB→RGB **有损**,整幅转一圈会把掩膜外的像素也改了,而"只改掩膜内"这句话就变成假的;
2. ★ **裁决方向**:第一版只看「抄周围贴不贴得近」,把**湿路面那对读反了** ——
   那里不处理已经 1.541、两条臂都没有更好(正确读法是**没余地**),
   却被读成"抄周围贴不近 ⇒ 有余地"。判据必须先问"**有没有方法能超过不处理**"。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.edit import harmonize_axis as HA


class TestRingMeanFill:
    def _img(self, val: int, h: int = 32, w: int = 32) -> np.ndarray:
        return np.full((h, w, 3), val, np.uint8)

    def test_only_touches_inside_the_mask(self):
        """★★ 掩膜外必须**逐位不变** —— 整幅 LAB 往返会把它们改了,而那是静默的。"""
        img = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), bool)
        mask[10:20, 10:20] = True
        out = HA.ring_mean_fill(img, mask, ~mask)
        np.testing.assert_array_equal(out[~mask], img[~mask])
        assert not np.array_equal(out[mask], img[mask]), "掩膜内得真的被改了"

    def test_fills_with_the_ring_mean(self):
        """掩膜内被填成外圈的 LAB 均值 ⇒ 色彩应当明显被拉向外圈。"""
        img = np.zeros((32, 32, 3), np.uint8)
        ring = np.zeros((32, 32), bool)
        ring[:, :8] = True
        img[:, :8] = 200  # 外圈亮
        mask = np.zeros((32, 32), bool)
        mask[10:20, 20:30] = True  # 掩膜内是 0(黑)
        out = HA.ring_mean_fill(img, mask, ring)
        assert out[mask].mean() > 100, "填完之后掩膜内应当接近外圈的亮度"

    def test_no_model_is_used(self):
        """★ 定义性质:填进去的是**常量** ⇒ 掩膜内应当只剩**一种**颜色。

        ⚠️ 别用 `out[mask].std()` 测:那是**跨通道**一起算的,常量色 (151,131,127)
        照样给出 std ≈ 10(实测踩到)。要钉的是"每个像素都一样"。
        """
        img = np.random.default_rng(1).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), bool)
        mask[8:24, 8:24] = True
        out = HA.ring_mean_fill(img, mask, ~mask)
        assert len(np.unique(out[mask], axis=0)) == 1, "填进去的是常量 ⇒ 掩膜内只有一种颜色"


class TestVerdict:
    """★ 裁决方向 —— 湿路面那对把第一版读反了,这里把两种情形都钉住。"""

    def _rep(self, comp, ring, re_h):
        return {"median": {"d_composite": comp, "d_ring_mean": ring, "d_reinhard": re_h}}

    def test_statistics_solve_it_is_a_close_out(self):
        v = HA.verdict(self._rep(3.71, 2.15, 1.54))  # 实测 dense_fog
        assert "收口" in v and "没有" not in v.split("收口")[0]

    def test_nothing_beats_doing_nothing_is_no_headroom(self):
        """★★ 湿路面:**不处理已经最好** ⇒ 「没余地」,不许读成「有余地」。"""
        v = HA.verdict(self._rep(1.541, 2.153, 1.543))
        assert "无余地" in v and "有余地" not in v.replace("无余地", "")

    def test_a_real_gap_is_flagged(self):
        """既有可观的下降空间、又没被压到一半 ⇒ 值得再看。"""
        v = HA.verdict(self._rep(10.0, 8.0, 7.0))
        assert "有余地" in v


class TestPairGate:
    def test_box_shift_beyond_the_limit_raises(self, tmp_path):
        """A/B 框对不上 ⇒ **当场抛**,不许拿"贴的不是同一个物体"的读数出结论。"""
        for root, x in (("A", 0.0), ("B", 40.0)):
            d = tmp_path / root / "training"
            (d / "label_2").mkdir(parents=True)
            (d / "image_2").mkdir(parents=True)
            (d / "label_2" / "000000.txt").write_text(
                f"Car 0 0 0 {x} 10 {x + 20} 30 1 1 1 1 1 0 0 0 0\n", encoding="utf-8"
            )
        with pytest.raises(SystemExit, match="没有配对|作废"):
            HA.run_pair(tmp_path / "A", tmp_path / "B", ["000000"])

    def test_no_judgeable_frame_raises(self, tmp_path):
        for root in ("A", "B"):
            (tmp_path / root / "training" / "label_2").mkdir(parents=True)
        with pytest.raises(SystemExit, match="一帧可判的都没有"):
            HA.run_pair(tmp_path / "A", tmp_path / "B", ["000000"])

"""`edit/harmonize_data` + `edit/train_harmonize` 的判据。

这一层的两条要害,都是 2026-10-07 实测踩出来的:

1. ★ **掩膜铺满 patch 的样本必须丢掉** —— 那不是"省显存":和谐化的**全部信息**在
   "这块补丁该怎么跟周围对上",掩膜铺满时**根本没有"周围"**,
   连"按上下文对齐"这条基线的定义都不存在(实测当场抛 `掩膜内只有 0 个像素`)。
2. ★ **掩膜是 `(H,W)` 没有通道维,与图不是同一种形状** —— 第一版把两者套同一个
   `permute(0,3,1,2)`,当场 `input.dim()=3 != len(dims)=4`。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from autodrivedata.edit import harmonize_data as HD
from autodrivedata.edit import train_harmonize as TH


def _root(tmp_path, name, *, box, n=2, val=60):
    """极小 KITTI root:`label_2` 一条 Car + 常量图。"""
    r = tmp_path / name
    (r / "training/image_2").mkdir(parents=True)
    (r / "training/label_2").mkdir(parents=True)
    img = np.full((256, 256, 3), val, np.uint8)
    x1, y1, x2, y2 = box
    for i in range(n):
        Image.fromarray(img).save(r / f"training/image_2/{i:06d}.png")
        (r / f"training/label_2/{i:06d}.txt").write_text(
            f"Car 0 0 0 {x1} {y1} {x2} {y2} 1 1 1 1 1 0 0 0 0\n", encoding="utf-8"
        )
    return r


class TestPatchSlices:
    def test_center_patch(self):
        ys, xs = HD.patch_slices(128, 128, 256, 256, 128)
        assert (ys.start, ys.stop, xs.start, xs.stop) == (64, 192, 64, 192)

    def test_clamps_at_the_border_without_padding(self):
        """靠边的框要**夹回画幅**,不 padding —— padding 会造出本来不存在的像素。"""
        ys, xs = HD.patch_slices(2, 2, 256, 256, 128)
        assert (ys.start, xs.start) == (0, 0)
        ys2, xs2 = HD.patch_slices(255, 255, 256, 256, 128)
        assert (ys2.stop, xs2.stop) == (256, 256)

    def test_too_small_canvas_returns_none(self):
        assert HD.patch_slices(10, 10, 64, 64, 128) is None


class TestIterPatches:
    def _pair(self, tmp_path, box=(100, 100, 156, 156)):
        ctx = _root(tmp_path, "ctx", box=box, val=60)
        src = _root(tmp_path, "src", box=box, val=200)
        return ctx, src

    def test_composite_differs_only_inside_the_mask(self, tmp_path):
        ctx, src = self._pair(tmp_path)
        p = next(HD.iter_patches(ctx, src, ["000000"], patch=128))
        m = p["mask"]
        np.testing.assert_array_equal(p["composite"][~m], p["truth"][~m])
        assert not np.array_equal(p["composite"][m], p["truth"][m])

    def test_full_mask_patch_is_dropped_and_counted(self, tmp_path):
        """★★ 掩膜铺满 ⇒ **没有"周围"** ⇒ 丢掉,且必须**计数**(不许静默)。"""
        ctx, src = self._pair(tmp_path, box=(0, 0, 256, 256))  # 整个画幅都是车
        got = list(HD.iter_patches(ctx, src, ["000000"], patch=128))
        assert got == []
        assert HD.iter_patches.skipped >= 1, "丢掉的样本必须被计数"

    def test_normal_box_is_kept(self, tmp_path):
        ctx, src = self._pair(tmp_path)
        got = list(HD.iter_patches(ctx, src, ["000000"], patch=128))
        assert len(got) == 1
        assert got[0]["mask"].mean() < HD.MAX_MASK_SHARE


class TestParsePairs:
    def test_parses_ordered_pairs(self, tmp_path):
        got = HD._parse_pairs("a:b,c:d", tmp_path)
        assert got == [(tmp_path / "a", tmp_path / "b"), (tmp_path / "c", tmp_path / "d")]

    def test_order_matters(self, tmp_path):
        """★ 有序 —— "把谁贴进谁"是自变量,反了就是另一个样本。"""
        fwd = HD._parse_pairs("a:b", tmp_path)[0]
        rev = HD._parse_pairs("b:a", tmp_path)[0]
        assert fwd != rev

    def test_bad_chunk_raises(self, tmp_path):
        with pytest.raises(SystemExit, match="ctx:src"):
            HD._parse_pairs("ab", tmp_path)


class TestModel:
    def test_output_shape_and_range(self):
        m = TH.TinyUNet(base=4)
        x = torch.rand(2, 4, 32, 32)
        y = m(x)
        assert y.shape == (2, 3, 32, 32)
        assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0

    def test_zeroed_head_makes_it_an_identity(self):
        """★ 残差式的**定义性质**:把 head 清零 ⇒ 整个模型**恒等**。

        ⚠️ 第一版写的是"随机初始化下输出接近输入"(atol=0.35)—— **那是错的夹具**:
        `head` 是 1×1 conv,默认初始化下它乘在 ReLU 后的特征上,**偏移本来就可以很大**。
        残差结构说的是"输出 = 输入 + 残差",不是"残差初值小"。
        """
        m = TH.TinyUNet(base=4)
        with torch.no_grad():
            m.head.weight.zero_()
            m.head.bias.zero_()
        x = torch.rand(1, 4, 32, 32)
        assert torch.allclose(m(x), x[:, :3]), "head 清零后必须逐位等于输入"


class TestLoadSplit:
    def test_mask_gets_a_channel_dim(self, tmp_path):
        """★★ 掩膜是 `(H,W)`,与图**不是同一种形状** —— 第一版套同一个 `permute` 当场炸。"""
        d = tmp_path / "pair"
        d.mkdir()
        for i in range(3):
            np.savez_compressed(
                d / f"{i:06d}_00.npz",
                composite=np.zeros((16, 16, 3), np.uint8),
                truth=np.full((16, 16, 3), 255, np.uint8),
                mask=np.ones((16, 16), bool),
            )
        comp, truth, mask = TH.load_split(d.parent, ["pair"])
        assert comp.shape == (3, 3, 16, 16)
        assert truth.shape == (3, 3, 16, 16)
        assert mask.shape == (3, 1, 16, 16), "掩膜必须补一个通道维"

    def test_empty_tags_raise(self, tmp_path):
        (tmp_path / "pair").mkdir()
        with pytest.raises(SystemExit, match="一个样本都没有"):
            TH.load_split(tmp_path, ["pair"])

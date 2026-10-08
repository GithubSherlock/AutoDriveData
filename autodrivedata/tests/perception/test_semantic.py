"""autodrivedata/semantic.py 手算锚点单测。"""

from __future__ import annotations

import numpy as np

from autodrivedata.perception.semantic import (
    CARLA_SEMANTIC_ALBEDO,
    semantic_intensity,
    semantic_to_velodyne_bin,
    semantic_to_velodyne_tags,
)


class TestSemanticIntensity:
    def test_albedo_contrast(self):
        # 车(10)远亮于路面(7)——真实 velodyne 的关键对比
        tags = np.array([10, 7, 12])
        cos = np.ones(3)
        out = semantic_intensity(tags, cos, noise_std=0.0)
        assert out[0] > out[1] * 5
        assert out[2] > out[0]  # 标牌最高反

    def test_lambertian(self):
        tags = np.array([10, 10, 10])
        out = semantic_intensity(tags, np.array([1.0, 0.5, 0.0]), noise_std=0.0)
        np.testing.assert_allclose(out, [0.85, 0.425, 0.0], atol=1e-6)

    def test_clip_and_deterministic(self):
        tags = np.array([10] * 50)
        a = semantic_intensity(tags, np.ones(50), seed=42)
        b = semantic_intensity(tags, np.ones(50), seed=42)
        np.testing.assert_array_equal(a, b)
        assert a.min() >= 0.0 and a.max() <= 1.0
        # 噪声存在(非恒定)
        assert a.std() > 1e-4

    def test_unknown_tag_default(self):
        assert 999 not in CARLA_SEMANTIC_ALBEDO
        out = semantic_intensity(np.array([999]), np.ones(1), noise_std=0.0)
        np.testing.assert_allclose(out, [0.3], atol=1e-6)


class TestSemanticToBin:
    def test_y_flip_and_shape(self):
        pts = np.array([[1.0, 2.0, 3.0, 0.8, 0.0, 10.0]], dtype=np.float32)
        out = semantic_to_velodyne_bin(pts, seed=0)
        assert out.shape == (1, 4)
        assert out[0, 0] == 1.0 and out[0, 1] == -2.0 and out[0, 2] == 3.0
        assert 0.0 < out[0, 3] < 1.0


class TestSemanticToTags:
    """★ **标签版**(2026-10-07):`semantic_to_velodyne_bin` 把 tag 合成强度之后就丢掉了
    ⇒ 盘上没有任何"每点的语义类"可读,而按语义类编辑 LiDAR 正需要它
    (`docs/edit-pointcloud-plan.md` §4 重评条件 #3)。
    """

    def test_tag_lands_in_the_fourth_column(self):
        pts = np.array([[1.0, 2.0, 3.0, 0.8, 5.0, 10.0]], dtype=np.float32)
        out = semantic_to_velodyne_tags(pts)
        assert out.shape == (1, 4)
        assert out[0, 3] == 10.0, "第 4 列必须是**原始 tag**,不是合成强度"

    def test_row_aligned_with_the_intensity_version(self):
        """★★ 两个文件**逐行是同一个点**:前三列必须逐位相同。

        ⚠️ 这条是**契约**:读的人会按行号把 `velodyne/{fid}.bin` 与
        `semantic_velodyne/{fid}.bin` 对起来用;错位了不会报错,只会静默给错标签。
        """
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(50, 6)).astype(np.float32)
        a = semantic_to_velodyne_bin(pts, seed=0)
        b = semantic_to_velodyne_tags(pts)
        np.testing.assert_array_equal(a[:, :3], b[:, :3])

    def test_y_is_flipped_like_the_intensity_version(self):
        pts = np.array([[0.0, 2.0, 0.0, 1.0, 0.0, 7.0]], dtype=np.float32)
        assert semantic_to_velodyne_tags(pts)[0, 1] == -2.0

    def test_tag_is_not_averaged_or_filtered(self):
        """逐点独立,不去重、不合并 —— 同一类在不同点必须各自保留。"""
        pts = np.zeros((4, 6), dtype=np.float32)
        pts[:, 5] = [1.0, 7.0, 7.0, 10.0]
        np.testing.assert_array_equal(semantic_to_velodyne_tags(pts)[:, 3], [1.0, 7.0, 7.0, 10.0])

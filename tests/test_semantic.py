"""autodrivedata/semantic.py 手算锚点单测。"""

from __future__ import annotations

import numpy as np

from autodrivedata.semantic import CARLA_SEMANTIC_ALBEDO, semantic_intensity, semantic_to_velodyne_bin


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

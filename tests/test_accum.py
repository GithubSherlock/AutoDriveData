"""autodrivedata/accum.py 手算锚点单测。"""

from __future__ import annotations

import numpy as np

from autodrivedata.accum import (
    accumulate_global,
    colorize,
    semantic_color,
    voxel_downsample,
    write_ply,
)


def _frame(*rows) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


class TestVoxelDownsample:
    def test_empty(self):
        assert voxel_downsample(np.zeros((0, 4))).shape == (0, 4)

    def test_single_voxel_centroid(self):
        pts = np.array([[0.0, 0.0, 0.0, 0.9], [0.05, 0.0, 0.0, 0.9], [0.1, 0.0, 0.0, 0.9]])
        out = voxel_downsample(pts)
        assert out.shape == (1, 4)
        # 同一体素内均值
        np.testing.assert_allclose(out[0, :3], [0.05, 0.0, 0.0], atol=1e-9)
        # 强度列取首点
        assert out[0, 3] == 0.9

    def test_two_voxels(self):
        pts = np.array([[0.0, 0.0, 0.0, 0.9], [0.3, 0.0, 0.0, 0.1]])
        out = voxel_downsample(pts)
        assert out.shape == (2, 4)


class TestAccumulateGlobal:
    def test_concat_same_frame(self):
        f1 = _frame([1.0, 2.0, 3.0, 0.5])
        f2 = _frame([4.0, 5.0, 6.0, 0.1])
        out = accumulate_global([f1, f2])
        assert out.shape == (2, 4)
        np.testing.assert_allclose(out[1, :3], [4.0, 5.0, 6.0])

    def test_empty_list(self):
        assert accumulate_global([]).shape == (0, 4)


class TestSemanticColor:
    def test_high_reflectivity_red(self):
        # 车(0.85)/标牌(0.9)高反
        assert semantic_color(0.6) == (200, 60, 60)
        assert semantic_color(0.9) == (200, 60, 60)

    def test_mid_green(self):
        assert semantic_color(0.4) == (200, 160, 60)

    def test_low_road_grey(self):
        assert semantic_color(0.1) == (120, 120, 120)


class TestColorize:
    def test_shape_and_value(self):
        pts = np.array([[0.0, 0.0, 0.0, 0.9], [1.0, 1.0, 1.0, 0.1]])
        out = colorize(pts)
        assert out.shape == (2, 6)
        # rgb 列在 0-255(经 float64 hstack 归一)
        np.testing.assert_allclose(out[0, 3:], [200.0, 60.0, 60.0])
        np.testing.assert_allclose(out[1, 3:], [120.0, 120.0, 120.0])


class TestWritePly:
    def test_writes_vertex_count(self, tmp_path):
        p = tmp_path / "m.ply"
        write_ply(p, np.array([[0.0, 0.0, 0.0, 200, 60, 60]], dtype=np.float32))
        head = p.read_bytes().split(b"end_header\n")[0]
        assert b"element vertex 1" in head
        assert b"property float x" in head
        assert b"property uchar red" in head

    def test_writes_binary(self, tmp_path):
        p = tmp_path / "m.ply"
        write_ply(p, np.zeros((2, 6)))
        data = p.read_bytes()
        # 3 float + 3 uchar = 18 bytes/顶点 → 36 bytes 数据
        assert len(data.split(b"end_header\n")[1]) == 2 * (4 * 6)

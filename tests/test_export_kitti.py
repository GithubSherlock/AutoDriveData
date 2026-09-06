"""export/kitti.py 布局与落盘单测(镜像 auto3dlabel KittiFrame 路径契约)。"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.calib import CameraIntrinsics, KittiCalibOut, tr_velo_to_cam
from autodrivedata.export.kitti import frame_paths, normalize_frame_id, write_frame

CAM0 = (0.0, 0.0, 0.0)


class TestNormalize:
    def test_zero_padding(self):
        assert normalize_frame_id("123") == "000123"
        assert normalize_frame_id(" 7 ") == "000007"
        assert normalize_frame_id("000123") == "000123"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            normalize_frame_id("12a")


class TestFramePaths:
    def test_layout(self, tmp_path):
        p = frame_paths(tmp_path, "5")
        assert p.image == tmp_path / "training" / "image_2" / "000005.png"
        assert p.velodyne == tmp_path / "training" / "velodyne" / "000005.bin"
        assert p.calib == tmp_path / "training" / "calib" / "000005.txt"
        assert p.label == tmp_path / "training" / "label_2" / "000005.txt"


class TestWriteFrame:
    def _calib(self) -> KittiCalibOut:
        k = CameraIntrinsics(width=1242, height=375, fov_h_deg=90.0)
        return KittiCalibOut(p2=k.p2(), tr_velo_to_cam=tr_velo_to_cam(
            (0.0, 0.0, 0.0), CAM0, (0.0, 0.0, 0.0), CAM0
        ))

    def test_round_trip_contents(self, tmp_path):
        png = b"\x89PNG fake"
        velo = np.array([[1.0, -2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.7]], dtype=np.float32)
        labels = ["Car 0.00 0 0.00 1 2 3 4 1.5 1.6 3.9 0 1.7 10 -1.57"]
        paths = write_frame(tmp_path, "42", image_png=png, velodyne=velo,
                            calib=self._calib(), labels=labels)
        assert paths.image.read_bytes() == png
        got = np.fromfile(paths.velodyne, dtype=np.float32).reshape(-1, 4)
        np.testing.assert_allclose(got, velo)
        assert paths.calib.read_text().splitlines()[2].startswith("P2:")
        assert paths.label.read_text() == labels[0] + "\n"

    def test_empty_labels_writes_empty_file(self, tmp_path):
        # 空 label 文件存在但无行(auto3dlabel load_gt3d 读到 [] 语义)
        paths = write_frame(tmp_path, "0", image_png=b"x", velodyne=np.zeros((1, 4), np.float32),
                            calib=self._calib(), labels=[])
        assert paths.label.read_text() == ""

    def test_frame_ids_increment(self, tmp_path):
        p1 = frame_paths(tmp_path, "1")
        p2 = frame_paths(tmp_path, "2")
        assert p1.image.name == "000001.png"
        assert p2.image.name == "000002.png"

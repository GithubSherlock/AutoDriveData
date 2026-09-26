"""export/kitti.py 布局与落盘单测(镜像 auto3dlabel KittiFrame 路径契约)。"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.calib.core import CameraIntrinsics, KittiCalibOut, tr_velo_to_cam
from autodrivedata.gt.export.kitti import (
    frame_paths,
    normalize_frame_id,
    pose_path,
    read_pose,
    write_frame,
    write_pose,
)

CAM0 = (0.0, 0.0, 0.0)


def _calib() -> KittiCalibOut:
    """KITTI calib 工厂(TestWriteFrame / TestPose 共用)。"""
    k = CameraIntrinsics(width=1242, height=375, fov_h_deg=90.0)
    return KittiCalibOut(
        p2=k.p2(), tr_velo_to_cam=tr_velo_to_cam((0.0, 0.0, 0.0), CAM0, (0.0, 0.0, 0.0), CAM0)
    )


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
    def test_round_trip_contents(self, tmp_path):
        png = b"\x89PNG fake"
        velo = np.array([[1.0, -2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.7]], dtype=np.float32)
        labels = ["Car 0.00 0 0.00 1 2 3 4 1.5 1.6 3.9 0 1.7 10 -1.57"]
        paths = write_frame(tmp_path, "42", image_png=png, velodyne=velo, calib=_calib(), labels=labels)
        assert paths.image.read_bytes() == png
        got = np.fromfile(paths.velodyne, dtype=np.float32).reshape(-1, 4)
        np.testing.assert_allclose(got, velo)
        assert paths.calib.read_text().splitlines()[2].startswith("P2:")
        assert paths.label.read_text() == labels[0] + "\n"

    def test_empty_labels_writes_empty_file(self, tmp_path):
        # 空 label 文件存在但无行(auto3dlabel load_gt3d 读到 [] 语义)
        paths = write_frame(
            tmp_path,
            "0",
            image_png=b"x",
            velodyne=np.zeros((1, 4), np.float32),
            calib=_calib(),
            labels=[],
        )
        assert paths.label.read_text() == ""

    def test_frame_ids_increment(self, tmp_path):
        p1 = frame_paths(tmp_path, "1")
        p2 = frame_paths(tmp_path, "2")
        assert p1.image.name == "000001.png"
        assert p2.image.name == "000002.png"

    def test_pose_is_optional(self, tmp_path):
        # 不传 pose → 不产生 pose 文件(既有四件套调用方零影响)
        write_frame(
            tmp_path,
            "0",
            image_png=b"x",
            velodyne=np.zeros((1, 4), np.float32),
            calib=_calib(),
            labels=[],
        )
        assert not pose_path(tmp_path, "0").exists()

    def test_pose_written_when_given(self, tmp_path):
        write_frame(
            tmp_path,
            "0",
            image_png=b"x",
            velodyne=np.zeros((1, 4), np.float32),
            calib=_calib(),
            labels=[],
            pose=np.eye(4),
        )
        np.testing.assert_allclose(read_pose(pose_path(tmp_path, "0")), np.eye(4), atol=1e-9)


class TestPose:
    """ego 真值位姿(KITTI 12 数行主序)——SLAM 评估的输入契约。"""

    def test_pose_path_layout(self, tmp_path):
        assert pose_path(tmp_path, "7") == tmp_path / "training" / "pose" / "000007.txt"

    def test_round_trip(self, tmp_path):
        T = np.eye(4)
        T[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        T[:3, 3] = [12.5, -3.25, 0.75]
        write_pose(tmp_path, "3", T)
        got = read_pose(pose_path(tmp_path, "3"))
        np.testing.assert_allclose(got, T, atol=1e-9)

    def test_text_is_twelve_rowmajor_numbers(self, tmp_path):
        # 格式契约:单行 12 数,行主序 —— 与 KITTI 官方 poses/*.txt 同格式,
        # 吃 KITTI pose 的外部工具可直接读(改格式会静默破坏该兼容性)
        T = np.arange(12, dtype=np.float64).reshape(3, 4)
        write_pose(tmp_path, "0", T)
        txt = pose_path(tmp_path, "0").read_text().strip()
        assert len(txt.split()) == 12
        assert "\n" not in txt
        np.testing.assert_allclose(np.array([float(v) for v in txt.split()]), T.reshape(-1), atol=1e-9)

    def test_accepts_3x4_and_rejects_other_shapes(self, tmp_path):
        write_pose(tmp_path, "1", np.zeros((3, 4)))
        np.testing.assert_allclose(read_pose(pose_path(tmp_path, "1"))[:3, :3], np.zeros((3, 3)))
        with pytest.raises(ValueError):
            write_pose(tmp_path, "2", np.zeros((3, 3)))

    def test_read_pose_rejects_wrong_count(self, tmp_path):
        p = pose_path(tmp_path, "9")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("1 2 3\n")
        with pytest.raises(ValueError):
            read_pose(p)

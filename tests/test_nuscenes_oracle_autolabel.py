"""export/nuscenes.py vs devkit + auto3dlabel oracle(autolabel env 跑;base 自动跳过)。

验证:我们生成的 dataroot 能被 devkit NuScenes 构造、auto3dlabel 数据层消费、
NusBox 往返一致——M1b-10 nuscenes-queue 验收的前置。
"""

from __future__ import annotations

import numpy as np
import pytest

auto3dlabel = pytest.importorskip("auto3dlabel")
nuscenes_devkit = pytest.importorskip("nuscenes.nuscenes")

from auto3dlabel.data.nuscenes import (  # noqa: E402
    box3d_dict_to_nusbox,
    gt_boxes_of_sample,
    load_lidar_points,
    load_nuscenes,
    nusbox_to_box3d_dict,
    samples_of_scene,
)

from autodrivedata.export import nuscenes as ne  # noqa: E402


def _build_dataroot(tmp_path) -> None:
    pts = np.array(
        [[5.0, 0.0, -1.0, 0.9, 0.0], [5.5, 0.5, -0.8, 0.8, 0.0]], dtype=np.float32
    )
    samples = [
        ne.NusSample(
            ego_translation=(0.0, 0.0, 0.0),
            ego_yaw_nus=0.0,
            lidar_filename=f"samples/LIDAR_TOP/{i:06d}.bin",
            camera_filenames={c: f"samples/{c}/{i:06d}.png" for c in ne.NUS_CAMERAS},
            calib_lidar=((1.2, 0.0, 1.65), 0.0),
            calib_cameras={c: ((1.2, 0.0, 1.65), 0.0) for c in ne.NUS_CAMERAS},
            annotations=[
                {
                    "category": "car",
                    "translation": (10.0, 0.0, 0.0),
                    "size": (1.9, 4.5, 1.5),
                    "yaw_nus": 0.0,
                    "num_lidar_pts": 2,
                    "instance_token": "adinst1",
                },
                {
                    "category": "pedestrian",
                    "translation": (8.0, 3.0, 0.0),
                    "size": (0.6, 0.6, 1.8),
                    "yaw_nus": 0.5,
                    "num_lidar_pts": 1,
                    "instance_token": "adinst2",
                },
            ],
            timestamp=1700000000000000 + i,
        )
        for i in range(2)
    ]
    ne.write_mini_dataset(tmp_path, "v1.0-mini", {"scene-0103": samples})
    for i in range(2):
        (tmp_path / f"samples/LIDAR_TOP/{i:06d}.bin").parent.mkdir(
            parents=True, exist_ok=True
        )
        pts.tofile(tmp_path / f"samples/LIDAR_TOP/{i:06d}.bin")
        for c in ne.NUS_CAMERAS:
            p = tmp_path / f"samples/{c}/{i:06d}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x89PNG fake")


class TestDevkitConsumes:
    def test_nuscenes_constructs_and_samples(self, tmp_path):
        _build_dataroot(tmp_path)
        nusc = load_nuscenes(root=tmp_path, version="v1.0-mini")
        samples = samples_of_scene(nusc, "scene-0103")
        assert len(samples) == 2
        assert samples[0]["next"] == samples[1]["token"]

    def test_gt_boxes_and_lidar(self, tmp_path):
        _build_dataroot(tmp_path)
        nusc = load_nuscenes(root=tmp_path, version="v1.0-mini")
        sample = samples_of_scene(nusc, "scene-0103")[0]
        boxes = gt_boxes_of_sample(nusc, sample["token"])
        assert len(boxes) == 2
        car = next(b for b in boxes if b.label == "car")
        assert car.track_id == "adinst1"
        np.testing.assert_allclose(car.translation, (10.0, 0.0, 0.0), atol=1e-9)
        pts = load_lidar_points(sample, nusc, tmp_path)
        assert pts.shape == (2, 5)

    def test_nusbox_round_trip_via_autolabel(self, tmp_path):
        """我们的 GT → NusBox → auto3dlabel 渲染 dict → 逆变换:语义全保。"""
        _build_dataroot(tmp_path)
        nusc = load_nuscenes(root=tmp_path, version="v1.0-mini")
        sample = samples_of_scene(nusc, "scene-0103")[0]
        for box in gt_boxes_of_sample(nusc, sample["token"]):
            d = nusbox_to_box3d_dict(box, ego_translation=(0.0, 0.0, 0.0))
            back = box3d_dict_to_nusbox(d, ego_translation=(0.0, 0.0, 0.0))
            assert back.label == box.label
            np.testing.assert_allclose(back.translation, box.translation, atol=1e-9)
            np.testing.assert_allclose(back.size, box.size, atol=1e-9)
            np.testing.assert_allclose(back.quaternion, box.quaternion, atol=1e-9)
            assert back.track_id == box.track_id

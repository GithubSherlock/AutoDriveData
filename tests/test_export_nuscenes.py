"""export/nuscenes.py 手算锚点单测(纯 numpy,任意 env 可跑)。"""

from __future__ import annotations

import json

import numpy as np

from autodrivedata.export import nuscenes as ne

SIZE = (2.0, 4.0, 2.0)  # (w,l,h)
ORIGIN = (0.0, 0.0, 0.0)


class TestSensorToGlobal:
    def test_identity_chain(self):
        pts = np.array([[1.0, 2.0, 3.0]])
        out = ne.points_sensor_to_global_nus(pts, ORIGIN, 0.0, ORIGIN, 0.0)
        np.testing.assert_allclose(out, pts, atol=1e-12)

    def test_calib_translation(self):
        pts = np.array([[2.0, 3.0, 4.0]])
        out = ne.points_sensor_to_global_nus(pts, ORIGIN, 0.0, (1.0, 0.0, 1.65), 0.0)
        np.testing.assert_allclose(out, [[3.0, 3.0, 5.65]], atol=1e-12)

    def test_ego_pose_full_chain(self):
        # ego (10,20,0) yaw 0;calib t=(1,0,1.65) → p_g = p_s + t_c + t_e
        pts = np.array([[2.0, 3.0, 4.0]])
        out = ne.points_sensor_to_global_nus(pts, (10.0, 20.0, 0.0), 0.0, (1.0, 0.0, 1.65), 0.0)
        np.testing.assert_allclose(out, [[13.0, 23.0, 5.65]], atol=1e-12)

    def test_ego_yaw_rotation(self):
        # ego yaw π/2(车头 +y):传感器点 (1,0,0) → 全局 (0,1,0)
        pts = np.array([[1.0, 0.0, 0.0]])
        out = ne.points_sensor_to_global_nus(pts, ORIGIN, np.pi / 2, ORIGIN, 0.0)
        np.testing.assert_allclose(out, [[0.0, 1.0, 0.0]], atol=1e-12)

    def test_calib_yaw_rotation(self):
        # calib yaw π/2:传感器点 (1,0,0) → ego 系 (0,1,0) → 全局同(ego 恒等)
        pts = np.array([[1.0, 0.0, 0.0]])
        out = ne.points_sensor_to_global_nus(pts, ORIGIN, 0.0, ORIGIN, np.pi / 2)
        np.testing.assert_allclose(out, [[0.0, 1.0, 0.0]], atol=1e-12)


class TestCountInBox:
    def test_yaw0_inside_outside(self):
        pts = np.array([[1.0, 0.5, 0.5], [2.1, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.1], [1.0, 0.5, -0.5]])
        assert ne.count_points_in_box_nus(pts, ORIGIN, SIZE, 0.0) == 2

    def test_yaw90(self):
        # 车头 +y:沿 y 是长轴、沿 x 是宽轴
        pts = np.array([[0.0, 1.5, 0.0], [1.5, 0.0, 0.0]])
        assert ne.count_points_in_box_nus(pts, ORIGIN, SIZE, np.pi / 2) == 1

    def test_offset_center(self):
        pts = np.array([[10.5, 0.0, 0.0], [12.5, 0.0, 0.0]])
        assert ne.count_points_in_box_nus(pts, (10.0, 0.0, 0.0), SIZE, 0.0) == 1


def _sample(i: int) -> ne.NusSample:
    return ne.NusSample(
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
                "num_lidar_pts": 42,
                "instance_token": "adinst1",
            }
        ],
        timestamp=1700000000000000 + i,
    )


class TestMiniDataset:
    def test_tables_written(self, tmp_path):
        ne.write_mini_dataset(tmp_path, "v1.0-mini", {"scene-0103": [_sample(0), _sample(1)]})
        table_dir = tmp_path / "v1.0-mini"
        names = {
            "category",
            "attribute",
            "visibility",
            "instance",
            "sensor",
            "calibrated_sensor",
            "ego_pose",
            "log",
            "scene",
            "sample",
            "sample_data",
            "sample_annotation",
            "map",
        }
        assert {p.stem for p in table_dir.glob("*.json")} == names
        assert (tmp_path / "maps" / "ad_map.png").is_file()

    def test_tokens_unique_per_table(self, tmp_path):
        ne.write_mini_dataset(tmp_path, "v1.0-mini", {"scene-0103": [_sample(0), _sample(1)]})
        for p in (tmp_path / "v1.0-mini").glob("*.json"):
            rows = json.loads(p.read_text())
            tokens = [r["token"] for r in rows]
            assert len(tokens) == len(set(tokens)), p.name

    def test_sample_chain_and_scene(self, tmp_path):
        ne.write_mini_dataset(tmp_path, "v1.0-mini", {"scene-0103": [_sample(0), _sample(1)]})
        samples = json.loads((tmp_path / "v1.0-mini" / "sample.json").read_text())
        assert samples[0]["next"] == samples[1]["token"]
        assert samples[1]["prev"] == samples[0]["token"]
        scene = json.loads((tmp_path / "v1.0-mini" / "scene.json").read_text())[0]
        assert scene["name"] == "scene-0103"
        assert scene["first_sample_token"] == samples[0]["token"]

    def test_annotation_links(self, tmp_path):
        ne.write_mini_dataset(tmp_path, "v1.0-mini", {"scene-0103": [_sample(0), _sample(1)]})
        anns = json.loads((tmp_path / "v1.0-mini" / "sample_annotation.json").read_text())
        assert len(anns) == 2
        assert anns[0]["instance_token"] == "adinst1"
        inst = json.loads((tmp_path / "v1.0-mini" / "instance.json").read_text())
        cat = json.loads((tmp_path / "v1.0-mini" / "category.json").read_text())
        cat_name = next(c["name"] for c in cat if c["token"] == inst[0]["category_token"])
        assert cat_name == "vehicle.car"

    def test_sample_data_keyframes(self, tmp_path):
        ne.write_mini_dataset(tmp_path, "v1.0-mini", {"scene-0103": [_sample(0), _sample(1)]})
        sd = json.loads((tmp_path / "v1.0-mini" / "sample_data.json").read_text())
        assert len(sd) == 14  # 2 samples × (1 lidar + 6 cams)
        assert all(r["is_key_frame"] for r in sd)
        lidar_rec = next(r for r in sd if r["channel"] == "LIDAR_TOP")
        assert lidar_rec["filename"].endswith("000000.bin")

    def test_two_scenes_chain_not_crossing(self, tmp_path):
        ne.write_mini_dataset(
            tmp_path,
            "v1.0-mini",
            {"scene-0103": [_sample(0), _sample(1)], "scene-0916": [_sample(2)]},
        )
        samples = json.loads((tmp_path / "v1.0-mini" / "sample.json").read_text())
        scenes = {
            s["token"]: s["name"] for s in json.loads((tmp_path / "v1.0-mini" / "scene.json").read_text())
        }
        assert len(samples) == 3
        # 场景内成链、场景边界断开
        assert samples[0]["next"] == samples[1]["token"] and samples[1]["prev"] == samples[0]["token"]
        assert samples[1]["next"] == "" and samples[2]["prev"] == ""
        assert scenes[samples[2]["scene_token"]] == "scene-0916"

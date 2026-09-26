"""export/nuscenes.py 手算锚点单测(纯 numpy,任意 env 可跑)。"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from autodrivedata import geometry as g
from autodrivedata.calib.camera_rig import NUS_WIDE_REAR_X_CARLA
from autodrivedata.export import nuscenes as ne

SIZE = (2.0, 4.0, 2.0)  # (w,l,h)
ORIGIN = (0.0, 0.0, 0.0)
IDQ = (1.0, 0.0, 0.0, 0.0)  # 单位四元数


class TestSensorToGlobal:
    """`calib_quat` 是**四元数**不是 yaw —— LiDAR 的 up 轴倾角只能这样表达。"""

    def test_identity_chain(self):
        pts = np.array([[1.0, 2.0, 3.0]])
        out = ne.points_sensor_to_global_nus(pts, ORIGIN, IDQ, ORIGIN, IDQ)
        np.testing.assert_allclose(out, pts, atol=1e-12)

    def test_calib_translation(self):
        pts = np.array([[2.0, 3.0, 4.0]])
        out = ne.points_sensor_to_global_nus(pts, ORIGIN, IDQ, (1.0, 0.0, 1.65), IDQ)
        np.testing.assert_allclose(out, [[3.0, 3.0, 5.65]], atol=1e-12)

    def test_ego_pose_full_chain(self):
        # ego (10,20,0) yaw 0;calib t=(1,0,1.65) → p_g = p_s + t_c + t_e
        pts = np.array([[2.0, 3.0, 4.0]])
        out = ne.points_sensor_to_global_nus(pts, (10.0, 20.0, 0.0), IDQ, (1.0, 0.0, 1.65), IDQ)
        np.testing.assert_allclose(out, [[13.0, 23.0, 5.65]], atol=1e-12)

    def test_ego_yaw_rotation(self):
        # ego yaw π/2(车头 +y):传感器点 (1,0,0) → 全局 (0,1,0)
        pts = np.array([[1.0, 0.0, 0.0]])
        out = ne.points_sensor_to_global_nus(pts, ORIGIN, g.yaw_to_quat(np.pi / 2), ORIGIN, IDQ)
        np.testing.assert_allclose(out, [[0.0, 1.0, 0.0]], atol=1e-12)

    def test_calib_yaw_rotation(self):
        # calib yaw π/2:传感器点 (1,0,0) → ego 系 (0,1,0) → 全局同(ego 恒等)
        pts = np.array([[1.0, 0.0, 0.0]])
        out = ne.points_sensor_to_global_nus(pts, ORIGIN, IDQ, ORIGIN, g.yaw_to_quat(np.pi / 2))
        np.testing.assert_allclose(out, [[0.0, 1.0, 0.0]], atol=1e-12)

    def test_calib_quat_carries_tilt_not_just_yaw(self):
        """★ 四元数签名存在的理由:非 yaw 的姿态分量必须真的生效(否则回到 yaw-only 的坑)。

        官方 LiDAR_TOP 的 up 轴有 1.4289° 倾角 ⇒ 传感器点 (0,0,1)(正上方)经 R_calib
        后**不再**是 (0,0,1)。yaw-only 实现下这个测试必然失败(它只能绕 z 转)。
        """
        pts = np.array([[0.0, 0.0, 1.0]])
        out = ne.points_sensor_to_global_nus(pts, ORIGIN, IDQ, ORIGIN, ne.NUS_LIDAR_CALIB[1])
        assert not np.allclose(out, pts, atol=1e-3)
        # 倾角 = 与竖直方向的夹角;官方 up 轴实测 1.4289°
        tilt = math.degrees(math.acos(float(np.clip(out[0, 2] / np.linalg.norm(out[0]), -1, 1))))
        assert tilt == pytest.approx(1.4289, abs=0.01)


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
        ego_rotation_nus=IDQ,
        lidar_filename=f"samples/LIDAR_TOP/{i:06d}.bin",
        camera_filenames={c: f"samples/{c}/{i:06d}.png" for c in ne.NUS_CAMERAS},
        calib_lidar=(ne.NUS_LIDAR_CALIB[0], ne.NUS_LIDAR_CALIB[1]),
        calib_cameras={c: ((1.2, 0.0, 1.65), (1.0, 0.0, 0.0, 0.0)) for c in ne.NUS_CAMERAS},
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
        assert len(sd) == 24  # 2 samples × (1 lidar + 5 radar + 6 cams)
        assert all(r["is_key_frame"] for r in sd)
        lidar_rec = next(r for r in sd if r["channel"] == "LIDAR_TOP")
        assert lidar_rec["filename"].endswith("000000.bin")

    def test_radar_rows(self, tmp_path):
        ne.write_mini_dataset(tmp_path, "v1.0-mini", {"scene-0103": [_sample(0)]})
        sd = json.loads((tmp_path / "v1.0-mini" / "sample_data.json").read_text())
        radar_recs = [r for r in sd if r["modality"] == "radar"]
        assert len(radar_recs) == 5
        assert {r["channel"] for r in radar_recs} == set(ne.NUS_RADAR_CHANNELS)
        assert all(r["fileformat"] == "pcd" for r in radar_recs)
        # 旧 sample 未传 radar_filenames → filename 空串(向后兼容)
        assert all(r["filename"] == "" for r in radar_recs)
        # sensor/calib 链
        sens = json.loads((tmp_path / "v1.0-mini" / "sensor.json").read_text())
        calib = json.loads((tmp_path / "v1.0-mini" / "calibrated_sensor.json").read_text())
        assert {s["channel"] for s in sens if s["modality"] == "radar"} == set(ne.NUS_RADAR_CHANNELS)
        front = next(
            c
            for c in calib
            if c["sensor_token"] == next(s["token"] for s in sens if s["channel"] == "RADAR_FRONT")
        )
        assert front["translation"] == list(ne.NUS_RADAR_OFFSETS["RADAR_FRONT"][0])
        # 旋转 = 官方 n015 原值(旧表是 ±45/±90 的猜测值,四路角雷达实测差 94–136°)
        assert front["rotation"] == pytest.approx(list(g.yaw_to_quat(ne.NUS_RADAR_OFFSETS["RADAR_FRONT"][1])))
        back_left = next(
            c
            for c in calib
            if c["sensor_token"] == next(s["token"] for s in sens if s["channel"] == "RADAR_BACK_LEFT")
        )
        assert back_left["rotation"] == pytest.approx(
            list(g.yaw_to_quat(ne.NUS_RADAR_OFFSETS["RADAR_BACK_LEFT"][1]))
        )

    def test_lidar_calib_written_as_official_quaternion(self, tmp_path):
        """LIDAR_TOP 的 rotation 必须是**官方四元数原值**(含 1.4289° up 轴倾角),不是单位。

        历史缺陷:写 `_quat(0.0)` = 单位四元数 ⇒ devkit 按"传感器系 = ego 系"解释点云,
        整片点云绕 z 转 90°(`num_lidar_pts` 复现比值 1.0000 → 0.0854)。
        """
        ne.write_mini_dataset(tmp_path, "v1.0-mini", {"scene-0103": [_sample(0)]})
        calib = json.loads((tmp_path / "v1.0-mini" / "calibrated_sensor.json").read_text())
        sens = json.loads((tmp_path / "v1.0-mini" / "sensor.json").read_text())
        lid = next(
            c
            for c in calib
            if c["sensor_token"] == next(s["token"] for s in sens if s["channel"] == "LIDAR_TOP")
        )
        assert lid["translation"] == list(ne.NUS_LIDAR_CALIB[0])
        assert lid["rotation"] == pytest.approx(list(ne.NUS_LIDAR_CALIB[1]))
        assert not np.allclose(lid["rotation"], [1.0, 0.0, 0.0, 0.0], atol=1e-3)

    def test_camera_intrinsics_match_official_n015_per_channel(self, tmp_path):
        """`camera_intrinsic` 逐通道 == 官方 n015 实测 K(1600×900)。

        官方值**在测试里独立硬编码**(不从 `ne.NUS_CAMERA_INTRINSICS` 取)——
        否则断言恒真,拦不住"表被改错"。历史缺陷:六路共用 `fx=800 / cx=799.5`,
        与官方逐通道值差 **1.57×**(CAM_BACK fx 809.22 除外,它恰好接近)。
        """
        official = {
            "CAM_FRONT": (1266.417203046554, 816.2670197447984, 491.50706579294757),
            "CAM_FRONT_LEFT": (1272.5979470598488, 826.6154927353808, 479.75165386361925),
            "CAM_FRONT_RIGHT": (1260.8474446004698, 807.968244525554, 495.3344268742088),
            "CAM_BACK": (809.2209905677063, 829.2196003259838, 481.77842384512485),
            "CAM_BACK_LEFT": (1256.7414812095406, 792.1125740759628, 492.7757465151356),
            "CAM_BACK_RIGHT": (1259.5137405846733, 807.2529053838625, 501.19579884916527),
        }
        ne.write_mini_dataset(tmp_path, "v1.0-mini", {"scene-0103": [_sample(0)]})
        calib = json.loads((tmp_path / "v1.0-mini" / "calibrated_sensor.json").read_text())
        sens = json.loads((tmp_path / "v1.0-mini" / "sensor.json").read_text())
        token_of = {s["channel"]: s["token"] for s in sens}
        for cam, (fx, cx, cy) in official.items():
            k = next(c for c in calib if c["sensor_token"] == token_of[cam])["camera_intrinsic"]
            assert k[0] == pytest.approx([fx, 0.0, cx], abs=1e-9), cam
            assert k[1] == pytest.approx([0.0, fx, cy], abs=1e-9), cam
            assert k[2] == [0.0, 0.0, 1.0], cam

    def test_camera_fov_matches_official_intrinsics(self):
        """蓝图 fov(逐通道)== `2·atan((w/2)/fx)`,由**测试里独立硬编码**的官方 fx 反推。

        这条是"渲染视野 vs 落盘 K"的一致性锁:CARLA 蓝图 `fov` 是水平 FOV,设成 90
        就会渲染出 90° 而标定说 64.3° —— 同一类"声明≠渲染"。
        """
        for cam, fx in (
            ("CAM_FRONT", 1266.417203046554),
            ("CAM_FRONT_LEFT", 1272.5979470598488),
            ("CAM_FRONT_RIGHT", 1260.8474446004698),
            ("CAM_BACK", 809.2209905677063),
            ("CAM_BACK_LEFT", 1256.7414812095406),
            ("CAM_BACK_RIGHT", 1259.5137405846733),
        ):
            expect = math.degrees(2.0 * math.atan((1600 / 2.0) / fx))
            assert ne.NUS_CAMERA_FOV[cam] == pytest.approx(expect, abs=1e-9), cam
            # 与 Plan2 §P-M.7.10 附录的 6 位小数字面值同源(差 ≤ 2e-4°)
            assert ne.NUS_CAMERA_FOV[cam] == pytest.approx(ne.camera_fov_h_deg(cam), abs=1e-12)
        assert ne.NUS_CAMERA_FOV["CAM_BACK"] > 89.0  # 唯一宽视场通道(官方 89.34°)
        assert max(ne.NUS_CAMERA_FOV[c] for c in ne.NUS_CAMERAS if c != "CAM_BACK") < 65.0

    def test_num_radar_pts_default(self, tmp_path):
        ne.write_mini_dataset(tmp_path, "v1.0-mini", {"scene-0103": [_sample(0)]})
        anns = json.loads((tmp_path / "v1.0-mini" / "sample_annotation.json").read_text())
        assert anns[0]["num_radar_pts"] == 0  # 旧 annotation 无此键 → 默认 0

    def test_wide_rig_writes_its_own_calibration(self, tmp_path):
        """`rig="wide"` 落盘的相机标定 == wide 表(平移/四元数/内参**三项同时**换)。

        "同时"是判据本体:只要有一项漏换,就回到 §P-M.7 的"表对了、图错了"(声明与渲染分叉)。
        wide 的目标值在测试里**独立硬编码**(fx 由 `(w/2)/tan(hfov/2)` 闭式算,挂点 x = −1.90)。
        """
        wide_fx = {
            "CAM_FRONT": 1536.78566,
            "CAM_FRONT_LEFT": 1536.78566,
            "CAM_FRONT_RIGHT": 1536.78566,
            "CAM_BACK": 461.88018,
            "CAM_BACK_LEFT": 560.16604,
            "CAM_BACK_RIGHT": 560.16604,
        }
        ne.write_mini_dataset(tmp_path, "v1.0-mini", {"scene-0103": [_sample(0)]}, rig="wide")
        calib = json.loads((tmp_path / "v1.0-mini" / "calibrated_sensor.json").read_text())
        sens = json.loads((tmp_path / "v1.0-mini" / "sensor.json").read_text())
        token_of = {s["channel"]: s["token"] for s in sens}
        for cam, fx in wide_fx.items():
            row = next(c for c in calib if c["sensor_token"] == token_of[cam])
            assert row["translation"] == pytest.approx(list(ne.NUS_WIDE_CAMERA_CALIBS[cam][0])), cam
            assert row["rotation"] == pytest.approx(list(ne.NUS_WIDE_CAMERA_CALIBS[cam][1])), cam
            k = row["camera_intrinsic"]
            assert k[0] == pytest.approx([fx, 0.0, 799.5], abs=1e-4), cam  # corner 主点
            assert k[1] == pytest.approx([0.0, fx, 449.5], abs=1e-4), cam
            assert k[2] == [0.0, 0.0, 1.0], cam
        # 后三路挂点确实后移、且不在车身内(车身最后点 x = −1.8527)
        # ★ 落盘声明值是 **nus 后轴原点系**(= 物理落点 − `NUS_EGO_ORIGIN_X`,负负得正),不是物理 x
        for cam in ("CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"):
            row = next(c for c in calib if c["sensor_token"] == token_of[cam])
            phys = NUS_WIDE_REAR_X_CARLA
            assert row["translation"][0] == pytest.approx(phys - g.NUS_EGO_ORIGIN_X, abs=1e-12), cam
            assert phys < -1.8527, f"{cam} 挂点回到车身内了({phys})"
            assert row["translation"][0] != pytest.approx(-1.9, abs=1e-3), f"{cam} 落盘值少了原点平移"

    def test_wide_rig_does_not_disturb_the_official_tables(self, tmp_path):
        """落一次 wide,官方口径的表**一字节不变**(wide 是分支不是替换)。"""
        ne.write_mini_dataset(tmp_path, "v1.0-mini", {"scene-0103": [_sample(0)]}, rig="wide")
        for cam in ne.NUS_CAMERAS:
            assert ne.camera_intrinsic(cam) == ne.camera_intrinsic(cam, "nuscenes"), cam
            assert ne.camera_intrinsic(cam, "nuscenes")[0][0] == pytest.approx(
                ne.NUS_CAMERA_INTRINSICS[cam][0], abs=1e-12
            ), cam
        with pytest.raises(ValueError, match="未知相机 rig"):
            ne.write_mini_dataset(tmp_path / "x", "v1.0-mini", {"scene-0103": [_sample(0)]}, rig="wide2")

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

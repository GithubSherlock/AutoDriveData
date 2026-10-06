"""`gs/attribute_instances` 的回归钉(纯值,不连 CARLA、不出模型)。

三类:
1. **投影口径** —— 向量化 `project_points` 必须与 `calib.core.world_to_img` 逐点一致;
2. **算法** —— 已知布局必须还原、未归属的第三态、阈值生效;
3. **判据** —— 跨视角一致率在高一致时要过、在打乱时必须塌、样本不足时必须**未判**。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from autodrivedata.calib.core import CameraIntrinsics, world_to_img
from autodrivedata.gs.attribute_instances import (
    MIN_SAMPLES,
    FrameRef,
    attribute,
    consistency_of_two_splits,
    inst_loader,
    instance_classes,
    intrinsics,
    load_frames,
    project_points,
)


def _ring(n_cams: int = 8, radius: float = 6.0, height: float = 1.5) -> list[FrameRef]:
    """一圈朝向世界原点的相机(与 `collect_rig.ring_cam_pose` 同构,但只依赖 numpy)。"""
    out = []
    for c in range(n_cams):
        a = 2 * np.pi * c / n_cams
        loc = (radius * np.cos(a), radius * np.sin(a), height)
        yaw = float(np.degrees(np.arctan2(-loc[1], -loc[0])))
        out.append(FrameRef(pitch=0.0, index=c, location=loc, rotation_rad=(0.0, np.radians(yaw), 0.0)))
    return out


def _paint(frames: list[FrameRef], objs: dict[int, tuple[float, float, float]], r: int = 4) -> dict:
    """已知布局 → 逐帧实例图(投成小圆点)。**这是唯一能给出"正确率"的夹具。**"""
    k = intrinsics()
    out = {}
    for fr in frames:
        img = np.zeros((k.height, k.width), dtype=np.uint16)
        for bid, pos in objs.items():
            ui, vi, ok = project_points(np.array([pos]), fr, k)
            if not ok[0]:
                continue
            u, v = int(ui[0]), int(vi[0])
            img[max(0, v - r) : v + r + 1, max(0, u - r) : u + r + 1] = bid
        out[fr.key] = img
    return out


#: 三个物体(**高度必须留在画幅里**:相机高 1.5、竖直半 FOV 只有 16.8°)+ 一个视场外的点。
OBJS = {11: (0.0, 0.0, 1.5), 22: (2.0, 0.0, 1.0), 33: (-2.0, 0.0, 2.0)}
OUTSIDE = (0.0, 0.0, -40.0)


class TestProjectPoints:
    def test_matches_world_to_img_pointwise(self):
        """★ **同一口径的机械钉** —— 投影进哪一帧的哪个像素,只允许有一套实现。"""
        rng = np.random.default_rng(0)
        pts = rng.normal(0, 8, size=(400, 3))
        k = intrinsics()
        fr = _ring(3)[0]
        ui, vi, ok = project_points(pts, fr, k)
        for j, pt in enumerate(pts):
            uv = world_to_img(tuple(pt), fr.location, fr.rotation_rad, k)
            assert (uv is None) != bool(ok[j]), f"第 {j} 点有效性与 world_to_img 不一致"
            if uv is not None:
                assert (int(round(uv[0])), int(round(uv[1]))) == (int(ui[j]), int(vi[j]))

    def test_behind_camera_is_invalid(self):
        """相机身后(深度 ≤ 0.5 m)必须判无效 —— 投影发散会造出**看着正常**的假命中。"""
        k = intrinsics()
        fr = FrameRef(0.0, 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        # 相机朝世界 +x;把点放在 −x 侧(身后)
        _, _, ok = project_points(np.array([[-5.0, 0.0, 0.0]]), fr, k)
        assert not ok[0]

    def test_beyond_fov_is_invalid(self):
        k = intrinsics()
        fr = FrameRef(0.0, 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        _, _, ok = project_points(np.array([[5.0, 500.0, 0.0]]), fr, k)
        assert not ok[0]


class TestAttribute:
    def test_recovers_known_layout(self):
        """已知布局必须逐点还原 —— 这里的"一致率"同时是**正确率**。"""
        frames = _ring()
        painted = _paint(frames, OBJS)
        means = np.array([*OBJS.values(), OUTSIDE], dtype=np.float64)
        r = attribute(means, frames, lambda fr: painted[fr.key])
        assert r.labels.tolist() == [11, 22, 33, -1]
        assert r.n_labeled == 3
        # 未归属的那个是**真的没看见**(hits=0),不是"看见了但不敢认"
        assert r.hits[3] == 0

    def test_min_samples_blocks_single_view(self):
        frames = _ring()
        painted = _paint(frames, OBJS)
        means = np.array([*OBJS.values()], dtype=np.float64)
        r = attribute(means, frames[:1], lambda fr: painted[fr.key], min_samples=MIN_SAMPLES)
        assert (r.labels < 0).all()

    def test_min_share_blocks_split_votes(self):
        """众数占比不足时必须**未归属** —— 一个 Gaussian 一半在一辆车上一半在墙上,不该硬认。"""
        frames = _ring()
        k = intrinsics()

        def load(fr: FrameRef):
            img = np.zeros((k.height, k.width), dtype=np.uint16)
            # 交替把同一片像素涂成两个 id ⇒ 每帧一票轮流,众数占比 ~0.5
            bid = 11 if fr.index % 2 == 0 else 22
            ui, vi, ok = project_points(np.array([OBJS[11]]), fr, k)
            u, v = int(ui[0]), int(vi[0])
            img[max(0, v - 4) : v + 5, max(0, u - 4) : u + 5] = bid
            return img

        means = np.array([OBJS[11]], dtype=np.float64)
        r_loose = attribute(means, frames, load, min_share=0.4)
        r_strict = attribute(means, frames, load, min_share=0.6)
        assert r_loose.labels[0] >= 0 and r_strict.labels[0] < 0

    def test_missing_instance_frames_give_all_unattributed(self):
        """实例图缺席 ⇒ 全部未归属(**不是**"这个场景没有物体")。"""
        means = np.array([*OBJS.values()], dtype=np.float64)
        r = attribute(means, _ring(), lambda fr: None)
        assert (r.labels < 0).all() and r.n_labeled == 0


class TestConsistency:
    def test_perfect_when_layout_is_exact(self):
        frames = _ring()
        painted = _paint(frames, OBJS)
        means = np.array([*OBJS.values()], dtype=np.float64)
        c = consistency_of_two_splits(means, frames, lambda fr: painted[fr.key])
        assert c.agree == 1.0
        assert c.verdict()[0] is True

    def test_permutation_control_is_far_below_agreement(self):
        """★ 判据内的**随机置换对照**必须真的低 —— 它是"一致率高"的分母。

        ⚠️ 这里用 6 个物体而不是 3 个:**3 个元素时随机置换有 1/3 的概率恰好是恒等**,
        对照会偶然跳到 1.0 —— 那不是"对照失效",是**夹具太小**。
        """
        objs = {**OBJS, 44: (0.0, 2.0, 1.2), 55: (0.0, -2.0, 1.8), 66: (3.0, 0.0, 2.2)}
        frames = _ring()
        painted = _paint(frames, objs)
        means = np.array([*objs.values()], dtype=np.float64)
        c = consistency_of_two_splits(means, frames, lambda fr: painted[fr.key], seed=1)
        assert c.shuffled < c.agree
        assert c.chance < 0.35  # 6 个 id ⇒ 碰撞概率 ≈ 1/6

    def test_shuffled_images_collapse(self):
        """★ **反向自证**:把实例图逐像素打乱 ⇒ 归属/一致率必须塌。"""
        frames = _ring()
        painted = _paint(frames, OBJS)
        rng = np.random.default_rng(5)
        shuffled = {kk: rng.permutation(v.ravel()).reshape(v.shape) for kk, v in painted.items()}
        means = np.array([*OBJS.values()], dtype=np.float64)
        r_ok = attribute(means, frames, lambda fr: painted[fr.key])
        r_sh = attribute(means, frames, lambda fr: shuffled[fr.key])
        c_sh = consistency_of_two_splits(means, frames, lambda fr: shuffled[fr.key])
        assert r_ok.n_labeled == 3
        assert r_sh.n_labeled < r_ok.n_labeled
        assert c_sh.verdict()[0] is None or c_sh.agree < 3.0 * max(c_sh.chance, c_sh.shuffled)

    def test_no_common_labels_is_undecided(self):
        """★ 第三态:两半没有共同归属 ⇒ `None`,不是通过也不是不通过。"""
        frames = _ring()
        painted = _paint(frames, OBJS)
        means = np.array([*OBJS.values()], dtype=np.float64)
        c = consistency_of_two_splits(means, frames[:1], lambda fr: painted[fr.key])
        assert c.n == 0
        assert c.verdict()[0] is None

    def test_low_agreement_fails(self):
        from autodrivedata.gs.attribute_instances import Consistency

        c = Consistency(n=10, agree=0.5, chance=0.33, shuffled=0.31)
        assert c.verdict()[0] is False

    def test_agreement_must_beat_the_control(self):
        """一致率看似很高但没超过对照 ⇒ **不过**(缺这条,"到处都归同一个网格"会被读成很稳)。"""
        from autodrivedata.gs.attribute_instances import Consistency

        c = Consistency(n=10, agree=0.85, chance=0.8, shuffled=0.79)
        assert c.verdict()[0] is False


class TestCaptureIO:
    def _write_capture(self, root, pitches=(0.0, -15.0), frames=3):
        cap = root / "capture"
        cap.mkdir(parents=True, exist_ok=True)
        (cap / "pitches.json").write_text(json.dumps(list(pitches)), encoding="utf-8")
        for p in pitches:
            poses = [
                {"i": i, "x": float(i), "y": 1.0, "z": 2.0, "pitch": p, "yaw": 30.0 * i, "roll": 0.0}
                for i in range(frames)
            ]
            (cap / f"poses_{int(p)}.json").write_text(json.dumps(poses), encoding="utf-8")
        return cap

    def test_load_frames_flattens_pitches(self, tmp_path):
        cap = self._write_capture(tmp_path)
        fr = load_frames(cap)
        assert len(fr) == 6
        assert [f.pitch for f in fr] == [0.0] * 3 + [-15.0] * 3
        assert fr[1].location == (1.0, 1.0, 2.0)
        assert fr[1].rotation_rad[1] == pytest.approx(np.radians(30.0))

    def test_missing_poses_raises(self, tmp_path):
        cap = self._write_capture(tmp_path)
        (cap / "poses_-15.json").unlink()
        with pytest.raises(SystemExit, match="对不上"):
            load_frames(cap)

    def test_inst_loader_reads_uint16_png(self, tmp_path):
        from autodrivedata.perception.inst_tags import encode_instance_png

        cap = tmp_path / "capture"
        (cap / "inst/p0").mkdir(parents=True)
        ids = np.array([[1, 2], [3, 65535]], dtype=np.uint16)
        (cap / "inst/p0/00000.png").write_bytes(encode_instance_png(ids))
        load = inst_loader(cap)
        fr = FrameRef(0.0, 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        assert np.array_equal(load(fr), ids)
        assert load(FrameRef(0.0, 9, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))) is None

    def test_instance_classes_needs_both_channels(self, tmp_path):
        """缺 `sem/` 时类归属**未判**(空字典),不是"没有物体"。"""
        cap = self._write_capture(tmp_path)
        assert instance_classes(cap, load_frames(cap)) == {}


class TestIntrinsics:
    def test_matches_full_resolution_corner_convention(self):
        k = intrinsics()
        ref = CameraIntrinsics(width=1242, height=375, fov_h_deg=90.0)
        assert (k.fx, k.fy, k.cx, k.cy) == (ref.fx, ref.fy, ref.cx, ref.cy)
        assert (k.width, k.height) == (1242, 375)


class TestCamConvention:
    """★ 钉住"训练侧相机系"与"本模块投影口径"是**同一套**(§4 阶段 B 的硬前置)。"""

    def test_geometry_helper_is_the_inverse_of_world_to_cam(self):
        from autodrivedata.utils import geometry as g

        rng = np.random.default_rng(3)
        for _ in range(20):
            rot = tuple(rng.uniform(-0.6, 0.6, size=3))
            r_wc = g.carla_cam2world(rot)
            assert np.allclose(r_wc @ g.camera_rotation_world_to_cam(rot), np.eye(3), atol=1e-12)

    def test_trainer_carla_convention_matches_this_module(self):
        """`train_3dgs_mini --cam-convention carla` 的 viewmat 必须与 `world_to_img` 同解。

        ⚠️ 这条测试**必须能红**:同一段代码在 `legacy` 下给定一个 yaw≠0 的位姿会给出
        完全不同的投影 —— 那是 2026-10-04 之前 `means*.npy` 不在 CARLA 世界系的根因。
        """
        pytest.importorskip("torch")
        import tempfile
        from pathlib import Path

        from autodrivedata.gs.train_3dgs_mini import _load_poses_and_cams

        with tempfile.TemporaryDirectory() as td:
            cap = Path(td) / "capture"
            cap.mkdir(parents=True)
            # ⚠️ `_load_poses_and_cams` 的 pitch 取 **pitches.json 那一份**(不是 poses 里的)
            (cap / "pitches.json").write_text("[-15]", encoding="utf-8")
            (cap / "poses_-15.json").write_text(
                json.dumps(
                    [{"i": 0, "x": 1.0, "y": 2.0, "z": 1.5, "pitch": -15.0, "yaw": 40.0, "roll": 0.0}]
                ),
                encoding="utf-8",
            )
            got = {}
            for conv in ("carla", "legacy"):
                poses, f, H, W, cx, cy, vm, ks = _load_poses_and_cams(cap, downsample=1, cam_convention=conv)
                got[conv] = (vm[0].numpy(), f, cx, cy)

        # 相机在 (1,2,1.5)、yaw 40° ⇒ 光轴 ≈ (0.766, 0.643, 0);点沿光轴前 5 m、低 1 m
        pt = (1.0 + 5.0 * np.cos(np.radians(40.0)), 2.0 + 5.0 * np.sin(np.radians(40.0)), 0.5)
        k = intrinsics()
        uv_true = world_to_img(pt, (1.0, 2.0, 1.5), (np.radians(-15.0), np.radians(40.0), 0.0), k)
        assert uv_true is not None
        for conv in ("carla", "legacy"):
            vm0, f, cx, cy = got[conv]
            pc = vm0[:3, :3] @ np.array(pt) + vm0[:3, 3]
            # viewmat/K 走 gsplat 的 **center** 约定(像素中心 = 索引 + 0.5)⇒ 减回来再比
            uv = (f * pc[0] / pc[2] + cx - 0.5, f * pc[1] / pc[2] + cy - 0.5)
            if conv == "carla":
                assert uv[0] == pytest.approx(uv_true[0], abs=1e-3)
                assert uv[1] == pytest.approx(uv_true[1], abs=1e-3)
            else:
                assert abs(uv[0] - uv_true[0]) > 20.0, "legacy 口径居然和真值对上了 —— 这条测试失去意义"

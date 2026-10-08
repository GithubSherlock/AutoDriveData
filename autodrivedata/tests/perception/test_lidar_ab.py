"""`perception/lidar_ab` 的判据回归钉。

**为什么这组测试必须存在**:本模块给出的第一个结论就是**否定性的**
（"这份 A/B 量不出点云编辑的效果"）。一个给出否定结论的仪器,
**自己必须先被证明能把肯定的情况认出来** —— 否则"没信号"与"尺子坏了"长得一样。

⇒ 每条判据都配一条**反向自证**:把已知的信号造出来,尺子必须报出来。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autodrivedata.perception import lidar_ab as L


def _cloud(n=200, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(-40, 40, size=(n, 3)).astype(np.float32)
    p[:, 0] = np.abs(p[:, 0])  # 都放前方,便于分前后
    return np.column_stack([p, rng.random(n).astype(np.float32)])


def _pose(t=(0.0, 0.0, 0.0)):
    m = np.eye(3, 4)
    m[:, 3] = t
    return m


class TestLoad:
    def test_reads_kitty_velodyne(self, tmp_path):
        p = tmp_path / "a.bin"
        _cloud(10).tofile(p)
        assert L.load_velodyne(p).shape == (10, 4)

    def test_empty_file_raises(self, tmp_path):
        """★ 空点云**不是**「这里没有点」—— 静默当成 0 点会让所有统计量失真。"""
        p = tmp_path / "e.bin"
        p.write_bytes(b"")
        with pytest.raises(SystemExit, match="空的"):
            L.load_velodyne(p)

    def test_wrong_stride_raises(self, tmp_path):
        p = tmp_path / "w.bin"
        np.zeros(7, dtype=np.float32).tofile(p)
        with pytest.raises(SystemExit, match="4 的倍数"):
            L.load_velodyne(p)

    def test_pose_is_reshaped_from_one_line(self, tmp_path):
        """★ 位姿文件是**一行 12 个数** —— 忘了 reshape 会让后面全部按 1-D 算错。"""
        p = tmp_path / "p.txt"
        p.write_text(" ".join(str(x) for x in np.eye(3, 4).ravel()) + "\n")
        assert L.load_pose(p).shape == (3, 4)

    def test_pose_with_wrong_count_raises(self, tmp_path):
        p = tmp_path / "p.txt"
        p.write_text("1 2 3\n")
        with pytest.raises(SystemExit, match="不是 12"):
            L.load_pose(p)


class TestGeometry:
    def test_pose_delta_identity_is_zero(self):
        assert L.pose_delta(_pose(), _pose()) == (0.0, 0.0)

    def test_pose_delta_reports_the_shift(self):
        dt, dr = L.pose_delta(_pose(), _pose((0.1, 0.0, 0.0)))
        assert dt == pytest.approx(0.1)
        assert dr == 0.0

    def test_to_world_identity_is_a_noop(self):
        c = _cloud(20)
        assert np.allclose(L.to_world(c, _pose()), c[:, :3])

    def test_to_world_applies_translation(self):
        c = _cloud(20)
        w = L.to_world(c, _pose((1.0, 2.0, 3.0)))
        assert np.allclose(w, c[:, :3] + np.array([1.0, 2.0, 3.0]))

    def test_region_split_puts_x_positive_in_front(self):
        c = _cloud(20)
        f, r = L.region_of(c)
        assert f.sum() + r.sum() == len(c)
        assert (c[f, 0] > 0).all()
        assert (c[r, 0] <= 0).all()


class TestNnDistance:
    def test_identical_clouds_give_zero(self):
        c = _cloud(50)
        assert L.nn_distance(c[:, :3], c[:, :3]).max() == 0.0

    def test_reverse_proof_shift_is_detected(self):
        """★ **反向自证**:把 B 整体平移 1 mm ⇒ 最近邻距离中位必须 ≈ 1 mm,不许是 0。

        少了这条,"最近邻距离是 0" 就分不清「真的重合」与「尺子写死了 0」。
        """
        c = _cloud(300, seed=1)
        d = L.nn_distance(c[:, :3], c[:, :3] + np.array([0.001, 0.0, 0.0]))
        assert np.median(d) == pytest.approx(0.001, abs=2e-4)

    def test_empty_b_returns_inf_not_zero(self):
        """★ B 侧空 ⇒ 返回 `inf`(「量不出来」),**不是 0**(「完全重合」)。"""
        c = _cloud(10)
        assert np.isinf(L.nn_distance(c[:, :3], np.zeros((0, 3)))).all()


class TestFrameStats:
    def test_no_change_means_no_difference(self):
        c, p = _cloud(80), _pose()
        s = L.frame_stats(0, c, p, c, p)
        assert s.nn_median == 0.0
        assert s.d_front == 0 and s.d_rear == 0
        assert s.pose_dt == 0.0

    def test_reverse_proof_added_object_shows_up_in_front(self):
        """★ **反向自证**:在**前方**加一簇点(模拟"编辑"的动作) ⇒ 前方点数差必须非零。"""
        c, p = _cloud(80), _pose()
        extra = np.tile(np.array([[20.0, 0.0, 0.0, 0.5]], dtype=np.float32), (30, 1))
        s = L.frame_stats(0, c, p, np.vstack([c, extra]), p)
        # `d_front = B前方 − A前方`;**B 多 30 点 ⇒ +30**(第一版把符号写反了)
        assert s.d_front == +30
        assert s.d_rear == 0, "加的簇在前方,后方点数不该动"

    def test_front_and_rear_are_tracked_separately(self):
        """★ 后方是**对照**:只改后方时,前方那格必须纹丝不动。"""
        c, p = _cloud(80), _pose()
        back = np.tile(np.array([[-20.0, 0.0, 0.0, 0.5]], dtype=np.float32), (12, 1))
        s = L.frame_stats(0, c, p, np.vstack([c, back]), p)
        assert s.d_rear == +12
        assert s.d_front == 0


def _write_root(root: Path, frames, poses, clouds):
    (root / "training/velodyne").mkdir(parents=True, exist_ok=True)
    (root / "training/pose").mkdir(parents=True, exist_ok=True)
    for i, (p, c) in enumerate(zip(poses, clouds, strict=True)):
        c.tofile(root / "training/velodyne" / f"{i:06d}.bin")
        (root / "training/pose" / f"{i:06d}.txt").write_text(
            " ".join(str(x) for x in np.asarray(p).ravel()) + "\n"
        )


class TestComparePair:
    def _pair(self, tmp_path, *, shift=0.0, extra_front=0):
        a, b = tmp_path / "a", tmp_path / "b"
        poses = [_pose((i * 0.8, 0.0, 0.0)) for i in range(3)]
        ca = [_cloud(120, seed=i) for i in range(3)]
        cb = [c + np.array([shift, 0, 0, 0], dtype=np.float32) for c in ca]
        if extra_front:
            add = np.tile(np.array([[15.0, 0.0, 0.0, 0.5]], dtype=np.float32), (extra_front, 1))
            cb = [np.vstack([c, add]) for c in cb]
        _write_root(a, poses, poses, ca)
        _write_root(b, poses, poses, cb)
        return a, b

    def test_identical_pair_is_clean(self, tmp_path):
        a, b = self._pair(tmp_path)
        r = L.compare_pair(a, b)
        assert r["verdict"] == "可判"
        assert r["nn_median_median"] == 0.0
        assert r["d_front_median"] == 0 and r["d_rear_median"] == 0

    def test_reverse_proof_object_in_front_is_reported(self, tmp_path):
        """★ 反向自证:前方多 40 个点 ⇒ 前方点数差必须报出来(这是"能测到编辑"的最小形态)。"""
        a, b = self._pair(tmp_path, extra_front=40)
        r = L.compare_pair(a, b)
        assert r["d_front_median"] == +40
        assert r["d_rear_median"] == 0

    def test_pose_shift_is_reported_as_such(self, tmp_path):
        """★ 位姿差必须**单独报** —— 它会让整片点云错位,与"编辑"是两回事。"""
        a, b = self._pair(tmp_path, shift=0.05)
        r = L.compare_pair(a, b)
        assert r["nn_median_median"] > 0.04, "平移 5 cm 必须体现在最近邻距离上"
        assert r["d_front_median"] == 0, "纯平移不改点数"

    def test_frame_mismatch_raises(self, tmp_path):
        a, b = self._pair(tmp_path)
        (b / "training/velodyne/000009.bin").write_bytes(np.zeros(4, dtype=np.float32).tobytes())
        with pytest.raises(SystemExit, match="帧号不同"):
            L.compare_pair(a, b)

    def test_missing_velodyne_dir_raises(self, tmp_path):
        with pytest.raises(SystemExit, match="不是一份带 LiDAR"):
            L.compare_pair(tmp_path / "nope", tmp_path / "nope2")


class TestCli:
    def test_main_writes_json(self, tmp_path, monkeypatch, capsys):
        """端到端:CLI 跑得通、落得下 JSON,且**降采样开关**生效。"""
        import sys

        a, b = TestComparePair()._pair(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "lidar_ab",
                "--root-a",
                str(a),
                "--root-b",
                str(b),
                "--max-frames",
                "2",
                "--out",
                str(tmp_path / "r.json"),
                "--no-runlog",
            ],
        )
        L.main()
        d = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
        assert d["n_frames"] == 2
        assert capsys.readouterr().out.count("[L0") >= 1


class TestPoseSeriesReport:
    """★ 重评条件 #1「位姿逐位可复现」的**诊断器** —— 光知道"差 0.06 m"没法修。

    2026-10-07 实测把原来的判读**改了**:`edit-pointcloud-plan` §1 读成「亚帧相位差」,
    实测是「**启动瞬态 + 0.24% 的恒定速度差沿航向累积**」(见 `pose_series_report` 头注)。
    本组把两种形状**分开钉住**:真无限价对照的话,"拆开"这件事本身就没被验过。
    """

    @staticmethod
    def _root(tmp_path, name, ts):
        d = tmp_path / name / "training" / "pose"
        d.mkdir(parents=True)
        for i, t in enumerate(ts):
            m = np.eye(3, 4)
            m[:, 3] = [t[0], t[1], t[2]]
            np.savetxt(d / f"{i:06d}.txt", m.reshape(1, -1))
        return tmp_path / name

    def test_identical_trajectories_have_zero_offset(self, tmp_path):
        ts = [(i * 0.8, 0.0, 0.0) for i in range(20)]
        r = L.pose_series_report(self._root(tmp_path, "A", ts), self._root(tmp_path, "B", ts))
        assert r["d_max"] == pytest.approx(0.0, abs=1e-9)

    def test_constant_phase_lag_is_not_flat_but_is_along_heading(self, tmp_path):
        """纯滞后:位置差**恒定**、步长**完全相同** ⇒ 这才是"相位差"。"""
        ts = [(i * 0.8, 0.0, 0.0) for i in range(20)]
        lag = [(t[0] - 0.05, 0.0, 0.0) for t in ts]
        r = L.pose_series_report(self._root(tmp_path, "A", ts), self._root(tmp_path, "B", lag))
        assert r["d_first"] == pytest.approx(r["d_last"], abs=1e-6), "纯滞后 ⇒ 位置差是常量"
        assert r["step_gap_median"] == pytest.approx(0.0, abs=1e-9), "纯滞后 ⇒ 步长完全相同"

    def test_speed_difference_grows_monotonically(self, tmp_path):
        """★ 速度差:位置差**单调增长**、步长**持续不等** —— 实测那对是这个形状。"""
        ta = [(i * 0.8, 0.0, 0.0) for i in range(20)]
        tb = [(i * 0.8009, 0.0, 0.0) for i in range(20)]  # 快 0.11%
        r = L.pose_series_report(self._root(tmp_path, "A", ta), self._root(tmp_path, "B", tb))
        assert r["d_last"] > r["d_first"] * 10
        assert r["step_gap_median"] > 0, "速度差 ⇒ 步长必然不等(这正是它和滞后的分界)"
        assert r["step_gap_rel"] < 0.01

    def test_lateral_offset_is_flagged_as_not_along_heading(self, tmp_path):
        """横移 ≠ 沿航向 —— 判据必须分得开,否则"沿航向"这句话没有判别力。"""
        ta = [(i * 0.8, 0.0, 0.0) for i in range(20)]
        tb = [(i * 0.8, 0.05, 0.0) for i in range(20)]
        r = L.pose_series_report(self._root(tmp_path, "A", ta), self._root(tmp_path, "B", tb))
        assert not r["is_mostly_along_heading"]

    def test_frame_mismatch_raises(self, tmp_path):
        with pytest.raises(SystemExit, match="不齐"):
            L.pose_series_report(
                self._root(tmp_path, "A", [(0.0, 0, 0)] * 3), self._root(tmp_path, "B", [(0.0, 0, 0)] * 2)
            )

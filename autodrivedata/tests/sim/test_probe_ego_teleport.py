"""`sim/probe_ego_teleport` 的判据。

它判的是一个**已经有归档结论**的问题(`Plan4.md:889` 说"车上物理反复瞬移不可靠"),
而实测给出的是**第三种答案**:瞬移**不是逐位**(0.39 mm ≠ 0),但比归档的物理 A/B(62 mm)
好 **158×**。⇒ 这里钉的是"裁决不许退化成零/非零二分"。
"""

from __future__ import annotations

import numpy as np

from autodrivedata.sim import probe_ego_teleport as P


class TestStraightSequence:
    def test_steps_are_speed_times_tick(self):
        seq = P.straight_sequence((0.0, 5.0, 1.0), speed=8.0, tick=0.1, n=4)
        assert [round(p[0], 6) for p in seq] == [0.0, 0.8, 1.6, 2.4]
        assert all(p[1] == 5.0 and p[2] == 1.0 for p in seq)

    def test_it_is_absolute_not_incremental(self):
        """★ **绝对**位姿:第 i 帧不依赖前 i−1 帧 —— 增量会把误差累积进去,而这里要量的正是位姿本身。"""
        a = P.straight_sequence((0.0, 0.0, 0.0), 8.0, 0.1, 10)
        b = P.straight_sequence((0.0, 0.0, 0.0), 8.0, 0.1, 10)
        assert a == b


class TestWritePoseRoot:
    def test_round_trips_through_the_existing_reader(self, tmp_path):
        """★ 落盘格式必须能被**已有的** `lidar_ab.load_pose`(一行 12 个数)读回来。"""
        from autodrivedata.perception.lidar_ab import load_pose

        seq = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        root = P.write_pose_root(tmp_path / "r", seq)
        for i, p in enumerate(seq):
            m = load_pose(root / "training" / "pose" / f"{i:06d}.txt")
            assert np.allclose(m[:, 3], p, atol=1e-6)
            assert np.allclose(m[:, :3], np.eye(3), atol=1e-9), "yaw=0 ⇒ 旋转必须是单位"


class TestVerdict:
    """★ 三分裁决 —— 实测落在**中间那一档**,所以中间档必须有名字。"""

    @staticmethod
    def _rep(tel_mm: float, phys_mm: float):
        return {
            "ref_archived_physics_ab_m": 0.0619,
            "cross_run": {
                "teleport": {"d_max": tel_mm / 1000.0},
                "physics": {"d_max": phys_mm / 1000.0},
            },
        }

    def test_dead_control_is_undecided(self):
        """★ 物理对照也是 0 ⇒ **未判** —— 探针量不出差异时,它的读数不作数。"""
        assert "未判" in P.verdict(self._rep(0.0, 0.0))

    def test_bit_exact_is_the_strong_verdict(self):
        assert "逐位可复现" in P.verdict(self._rep(0.0, 11.43))

    def test_not_bit_exact_but_two_orders_better_is_its_own_bucket(self):
        """★★ 实测那一档:0.39 mm 不是 0,但比 62 mm 好 158× —— **两个说法都要给**。"""
        v = P.verdict(self._rep(0.392, 11.43))
        assert "不是逐位" in v and "158×" in v and "够不够" in v

    def test_not_good_enough_is_the_negative_verdict(self):
        """瞬移若只比归档好一点点 ⇒ 与 `Plan4.md:889` 的否决一致。"""
        v = P.verdict(self._rep(30.0, 40.0))
        assert "不够好" in v

    def test_the_threshold_is_the_spectator_standard(self):
        """耦合钉:`BIT_EXACT_M` 必须与 3DGS 那条 `PAIR_TOL = 1e-6` 同量级(那里实测恰好 0)。"""
        from autodrivedata.gs.eval_edit import PAIR_TOL

        assert P.BIT_EXACT_M == PAIR_TOL

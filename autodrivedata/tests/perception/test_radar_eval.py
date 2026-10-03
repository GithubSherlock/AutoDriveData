"""雷达**速度对表**的判据自证 —— 在**合成 root** 上跑,不碰 CARLA。

## 为什么必须合成

这条判据的塌法全是"出一个看着合理的数":

| 塌法 | 症状 |
|---|---|
| 自车速度读成常数(不逐帧算) | 预期值全相等 ⇒ 回归退化成常数比,**R² 照样漂亮** |
| `vel` 还原时符号写反 | 斜率翻号,而|k| 仍是 1 |
| R² 用"绕原点"口径 | **近似常数**向量下恒 ≈0.999,**连打乱对照都塌不下来**(实测踩过) |
| 打乱对照没生效 | R² 高只能说明"两条曲线都平滑",说明不了它们对得上 |

⇒ 合成的好处是**答案是我摆的**:我按 `vel = −s·u_x` 造一个静止场景,判据**必须**回出
`|k| ≈ 1`;再把速度打乱,它**必须**回出 `|k| ≈ 0`。两条都要钉。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.gt.export.nuscenes import NUS_RADAR_CHANNELS
from autodrivedata.perception.radar import RADAR_NUS_FIELDS, nus18_to_pcd
from autodrivedata.perception.radar_eval import ego_pose, ego_speed, eval_root, radial_pairs

CH = NUS_RADAR_CHANNELS[0]
SPEED = 8.0
DT = 0.1


def _pose(x: float) -> str:
    """沿世界 +x 平移 `x` 的 3×4 位姿(行主序,与 `collect_slam.ego_pose_matrix` 同布局)。"""
    m = np.eye(4)
    m[0, 3] = x
    return "\n".join(" ".join(f"{v:.6f}" for v in row) for row in m[:3])


def _make_root(base, *, n_frames=3, per_frame=200, seeded=True, static=True):
    """合成 root:自车以 `SPEED` 沿 +x 走,雷达点按**静止场景**给速度。

    `static=True` ⇒ 每个点的径向速度 = `−s·u_x`(U 系里,`vel` 朝传感器为正的约定);
    `static=False` ⇒ 速度**随机**,判据必须**认不出来**。
    """
    (base / "training/pose").mkdir(parents=True, exist_ok=True)
    (base / "samples" / CH).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        (base / "training/pose" / f"{i:06d}.txt").write_text(_pose(i * SPEED * DT))
        # 传感器系里撒一批点:方位 ±30°、俯仰 ±5°、距离 5–60 m
        a = np.radians(rng.uniform(-30, 30, per_frame))
        e = np.radians(rng.uniform(-5, 5, per_frame))
        d = rng.uniform(5, 60, per_frame)
        x = d * np.cos(e) * np.cos(a)
        y = -d * np.cos(e) * np.sin(a)  # nus 系 y 左(与 detections_to_nus18 同口径)
        z = d * np.sin(e)
        ux = x / np.sqrt(x * x + y * y + z * z)
        vel = -SPEED * ux if static else rng.uniform(-1, 1, per_frame)
        arr = np.zeros((per_frame, len(RADAR_NUS_FIELDS)), dtype=np.float32)
        arr[:, 0], arr[:, 1], arr[:, 2] = x, y, z
        arr[:, 6] = -vel * ux  # vx = −vel·cos e·cos a(与 detections_to_nus18 同式)
        arr[:, 11] = 3  # ambig_state(devkit 过滤要合法值)
        (base / "samples" / CH / f"{i:06d}.pcd").write_bytes(nus18_to_pcd(arr))
    return base


class TestEgoSpeed:
    def test_reads_the_real_speed_not_a_constant(self, tmp_path):
        """★ 速度必须**逐帧算**:写死常数会让预期值全相等、回归退化成常数比。"""
        root = _make_root(tmp_path)
        got = [ego_speed(root, i) for i in range(2)]
        assert got == pytest.approx([SPEED, SPEED], abs=1e-3)

    def test_a_stationary_ego_reads_zero(self, tmp_path):
        (tmp_path / "training/pose").mkdir(parents=True)
        for i in range(2):
            (tmp_path / "training/pose" / f"{i:06d}.txt").write_text(_pose(0.0))
        assert ego_speed(tmp_path, 0) == pytest.approx(0.0, abs=1e-6)

    def test_missing_pose_returns_none_not_zero(self, tmp_path):
        """★ **没有真值 ≠ 速度为 0** —— 返回 0 会让"静止"与"没数据"混成一个读数。"""
        (tmp_path / "training/pose").mkdir(parents=True)
        (tmp_path / "training/pose" / "000000.txt").write_text(_pose(0.0))
        assert ego_speed(tmp_path, 0) is None
        assert ego_pose(tmp_path, 5) is None


class TestRadialPairs:
    def test_recovers_the_static_scene_velocity(self, tmp_path):
        """★ 合成时按 `vel = −s·u_x` 造 ⇒ 还原出来必须**逐点**相等。"""
        root = _make_root(tmp_path, n_frames=2)
        u, vel, s = radial_pairs(root, 0, CH)
        assert len(u) == 200
        assert s == pytest.approx(np.full(200, SPEED), abs=1e-3)
        assert vel == pytest.approx(-SPEED * u[:, 0], abs=1e-4)

    def test_no_pose_gives_empty_not_garbage(self, tmp_path):
        root = _make_root(tmp_path, n_frames=1)
        u, vel, s = radial_pairs(root, 0, CH)  # 第 1 帧没有 pose
        assert len(u) == 0 and len(vel) == 0 and len(s) == 0


class TestEvalRootIsFalsifiable:
    def test_a_static_scene_gives_unit_speed_ratio(self, tmp_path):
        """★ 正样本:`|k| ≈ 1`(|s| 复现了自车速度量值)、R² 高、对照塌。"""
        root = _make_root(tmp_path, seeded=False)
        r = eval_root(root, [0, 1], channels=(CH,))[CH]
        assert r["speed_ratio"] == pytest.approx(1.0, abs=0.02)
        assert r["r2"] > 0.9
        assert r["r2_shuffled"] < 0.2, "打乱之后必须塌 —— 否则 R² 量的不是它们对不对得上"

    def test_random_velocities_are_not_recognised(self, tmp_path):
        """★★ **反面对照**:速度随机时 `|k|` 必须掉到 0 附近 —— 这是这条判据的立论。"""
        root = _make_root(tmp_path, seeded=False, static=False)
        r = eval_root(root, [0, 1], channels=(CH,))[CH]
        assert r["speed_ratio"] < 0.3, f"随机速度被认出来了({r['speed_ratio']})—— 判据不成立"

    def test_centered_r2_collapses_when_the_association_is_destroyed(self, tmp_path):
        """★ 这条钉的是**中心化**那个口径:绕原点的 R² 在近似常数向量下恒 ≈1、对照不塌。"""
        root = _make_root(tmp_path, seeded=False)
        r = eval_root(root, [0, 1], channels=(CH,))[CH]
        assert r["r2"] - r["r2_shuffled"] > 0.5, (
            f"真 R² {r['r2']} 与打乱 {r['r2_shuffled']} 分不开 —— R² 口径又退回绕原点了"
        )


class TestVerdictCriterion:
    """`_ok` 的三条:量值 / 可分性 / 朝向(允许一个全局符号)。"""

    @staticmethod
    def _row(**kw):
        base = {"n": 100, "speed_ratio": 1.0, "r2": 0.7, "r2_shuffled": -0.5, "yaw_diff_deg": 180.0}
        return {**base, **kw}

    def test_a_180_degree_flip_still_passes(self):
        """★ `vel`「朝传感器为正」让推出朝向整体差 180° —— 那是**一个常数**,不该算错。"""
        from autodrivedata.perception.radar_eval import _ok

        assert _ok(self._row(yaw_diff_deg=180.0))
        assert _ok(self._row(yaw_diff_deg=-179.0))

    def test_a_90_degree_error_fails(self):
        from autodrivedata.perception.radar_eval import _ok

        assert not _ok(self._row(yaw_diff_deg=-88.0))

    def test_low_speed_ratio_fails(self):
        from autodrivedata.perception.radar_eval import _ok

        assert not _ok(self._row(speed_ratio=0.03))

    def test_a_non_collapsing_control_fails(self):
        from autodrivedata.perception.radar_eval import _ok

        assert not _ok(self._row(r2=0.7, r2_shuffled=0.69))

    def test_empty_channel_fails(self):
        from autodrivedata.perception.radar_eval import _ok

        assert not _ok({"n": 0})

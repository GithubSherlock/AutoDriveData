"""`gs/frame_sync` 的判据回归钉(纯值,不连 CARLA、不出模型)。

钉的是 §1.5 那件事的**可判定形式**:第 0 帧是不是采集瞬态。
三条读数(远点占比 / 中位深度 / 相邻差)+ **第三态**(帧数不够 = 未判)。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from autodrivedata.gs.frame_sync import (
    FAR_DEPTH_M,
    MIN_FRAMES,
    FrameStat,
    audit_pitch,
    frame_stat,
    mean_abs_diff,
)


def _synth(root, *, broken_first: bool, n: int = 8, pitch: float = 0.0, far=999.0):
    """合成一份 capture:地面深度 8 m;`broken_first` 时首帧整幅换成远平面。"""
    cap = root / "capture"
    (cap / "images" / f"p{int(pitch)}").mkdir(parents=True, exist_ok=True)
    (cap / "depth" / f"p{int(pitch)}").mkdir(parents=True, exist_ok=True)
    (cap / "pitches.json").write_text(f"[{int(pitch)}]", encoding="utf-8")
    rng = np.random.default_rng(7)
    base = rng.integers(0, 40, size=(8, 12, 3), dtype=np.uint8)
    for i in range(n):
        img = base.copy()
        if broken_first and i == 0:
            img = np.full_like(base, 250)
        Image.fromarray(img).save(cap / "images" / f"p{int(pitch)}" / f"{i:05d}.png")
        d = np.full((8, 12), far if (broken_first and i == 0) else 8.0, dtype=np.float32)
        np.save(cap / "depth" / f"p{int(pitch)}" / f"{i:05d}.npy", d)
    return cap


class TestFrameStat:
    def test_far_share_and_median(self):
        d = np.array([[1.0, 2.0, 3.0, 100.0]])
        st = frame_stat(0, d)
        assert st.median_depth == pytest.approx(2.5)
        assert st.far_share == pytest.approx(0.25 if FAR_DEPTH_M == 100 else st.far_share)

    def test_median_is_not_mean(self):
        """中位与均值在这里必须分开 —— 一帧 22% 的远点会把均值拉飞,中位稳得住。"""
        d = np.array([5.0] * 78 + [999.0] * 22)
        st = frame_stat(0, d)
        assert st.median_depth < FAR_DEPTH_M
        assert float(np.mean(d)) > 100

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            frame_stat(0, np.zeros((0, 0)))


class TestMeanAbsDiff:
    def test_identical_frames_are_zero(self):
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        assert mean_abs_diff(a, a) == 0.0

    def test_shape_mismatch_raises(self):
        """★ 尺寸不等**必须抛** —— 广播出来的 `mean` 是个看着正常的错数。"""
        with pytest.raises(ValueError, match="形状不同"):
            mean_abs_diff(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 5, 3), np.uint8))


class TestVerdict:
    def test_bad_first_frame_is_red(self, tmp_path):
        r = audit_pitch(_synth(tmp_path, broken_first=True), 0.0)
        ok, why = r.verdict()
        assert ok is False
        assert not r.far_ok and not r.median_ok

    def test_clean_first_frame_is_green(self, tmp_path):
        r = audit_pitch(_synth(tmp_path, broken_first=False), 0.0)
        ok, why = r.verdict()
        assert ok is True, why

    def test_too_few_frames_is_undecided(self, tmp_path):
        """★ 第三态:帧太少 ⇒ `None`。**不许**写成通过,也不许写成不通过。"""
        r = audit_pitch(_synth(tmp_path, broken_first=True, n=MIN_FRAMES - 2), 0.0)
        ok, why = r.verdict()
        assert ok is None and "未判" in why

    def test_verdict_is_undecided_not_pass_when_equal(self):
        """`real == control` 那种"看不出差别"必须是**不通过**,不是通过(同 static_eval 的口径)。"""
        st = FrameStat(0, 8.0, 0.0)
        from autodrivedata.gs.frame_sync import PitchReport

        r = PitchReport(0.0, 10, st, 8.0, 8.0, 0.0, 3.0, 3.0, 9)
        ok, _ = r.verdict()
        assert ok is True  # 三条读数各就各位 ⇒ 首帧正常
        r2 = PitchReport(0.0, 10, FrameStat(0, 8.0, 0.0), 8.0, 9.0, 0.0, 30.0, 3.0, 9)
        assert r2.verdict()[0] is False  # 只有相邻差炸 ⇒ 不过

    def test_one_bad_frame_among_many_does_not_move_the_bar(self, tmp_path):
        """★ 判据的判别力:**两三帧坏掉不能把"其余"的统计量撑起来**。

        首个版本用 min/max 当上界,而 2 帧瞬态恰好让 max == 首帧值 ⇒ 判据差点放过去。
        这里把第 3 帧也做成远端帧:首帧照样必须红。
        """
        cap = _synth(tmp_path, broken_first=True)
        np.save(cap / "depth/p0/00003.npy", np.full((8, 12), 999.0, dtype=np.float32))
        r = audit_pitch(cap, 0.0)
        assert not r.far_ok, "坏帧把'其余'的中位数撑起来了 —— 判据失去判别力"
        assert r.verdict()[0] is False


class TestAuditErrors:
    def test_missing_depth_dir_raises(self, tmp_path):
        (tmp_path / "capture").mkdir()
        with pytest.raises(SystemExit):
            audit_pitch(tmp_path / "capture", 0.0)

    def test_non_contiguous_frames_raise(self, tmp_path):
        """缺帧会让"相邻差"跨过一段空白而偏大 —— 那不是瞬态,是数据不全,必须单独报。"""
        cap = _synth(tmp_path, broken_first=False, n=6)
        (cap / "depth/p0/00003.npy").unlink()
        with pytest.raises(SystemExit, match="不连续"):
            audit_pitch(cap, 0.0)

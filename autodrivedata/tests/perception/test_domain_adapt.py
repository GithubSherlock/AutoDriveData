"""`perception/domain_adapt` 的判据。

★★ 这个模块的裁决**方向写反过一次**,而且是被自己的读数抓出来的:
第一版只比 `|gain| > floor` 就判"有用",而实测 `gain = −0.0516`
⇒ 它把"**掉了 5 个点**"读成了"**有用**"。**增益的正负是两件完全不同的事。**
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from autodrivedata.perception import domain_adapt as D


class TestAlignImage:
    def test_output_is_a_valid_rgb_image(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)
        out = D.align_image(img, np.array([50.0, 0.0, 0.0]), np.eye(3))
        assert out.shape == img.shape and out.dtype == np.uint8

    def test_aligning_to_its_own_statistics_is_near_identity(self):
        """★ 自证:把一张图对齐到**它自己的**统计 ⇒ 应当几乎不动(LAB 往返只有量化损失)。"""
        rng = np.random.default_rng(1)
        img = rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)
        m, c = D.lab_stats(img)
        out = D.align_image(img, m, c)
        assert np.abs(out.astype(int) - img.astype(int)).mean() < 2.0

    def test_a_strong_reference_actually_moves_the_image(self):
        """★ 反向自证:对齐到**明显不同**的统计 ⇒ 必须真的动(不动说明变换没接上)。"""
        img = np.full((24, 32, 3), 120, np.uint8)
        out = D.align_image(img, np.array([90.0, -20.0, 30.0]), np.eye(3) * 4.0)
        assert np.abs(out.astype(int) - img.astype(int)).mean() > 10.0


class TestMaterialize:
    def test_files_keep_the_stem_so_gt_pairing_is_unchanged(self, tmp_path):
        """★★ GT 是按**文件名**配的 ⇒ 改名会让预测配错 GT,而且不报错。"""
        src = tmp_path / "in"
        src.mkdir()
        for i in range(3):
            Image.fromarray(np.full((8, 8, 3), 100, np.uint8)).save(src / f"{i:012d}.jpg")
        out = D.materialize(
            sorted(src.glob("*.jpg")), tmp_path / "out", np.array([50.0, 0.0, 0.0]), np.eye(3)
        )
        assert [p.name for p in out] == [p.name for p in sorted(src.glob("*.jpg"))]

    def test_output_is_written_as_a_real_image_file(self, tmp_path):
        """★ **刻意落盘**:ultralytics 的 `predict(ndarray)` 走 BGR,喂 RGB 会静默换 R/B。"""
        src = tmp_path / "in"
        src.mkdir()
        Image.fromarray(np.full((8, 8, 3), 100, np.uint8)).save(src / "000000000000.jpg")
        out = D.materialize([src / "000000000000.jpg"], tmp_path / "o", np.array([50.0, 0.0, 0.0]), np.eye(3))
        assert out[0].is_file()
        assert np.asarray(Image.open(out[0])).shape == (8, 8, 3)


class TestVerdict:
    @staticmethod
    def _rep(base, align, jitter, wrong, ctrl_ok=True):
        def arm(v):
            # 尺子正常 ⇒ 平移 GT 之后 ≈0;坏掉 ⇒ 平移了也照样高分(实测的坏法是 ~0.5)
            return {"mAP": v, "mAP_control_shift": 0.0 if ctrl_ok else 0.5, "car": {"ap": v, "recall": v}}

        return {
            "arms": {
                "baseline": arm(base),
                "align_carla": arm(align),
                "jitter_floor": arm(jitter),
                "wrong_target": arm(wrong),
            }
        }

    def test_the_measured_reading_says_harmful_not_useful(self):
        """★★ 实测那一档:−0.0516、地板 0.0502 ⇒ **有害**,不是"有用"。

        第一版在这里判"有用" —— 它只看了绝对值,没看符号。
        """
        v = D.verdict(self._rep(0.0794, 0.0278, 0.0292, 0.0624))
        assert "有害" in v
        assert "有用" not in v.replace("没用", ""), "负增益绝不许读成'有用'"

    def test_a_real_positive_gain_is_useful(self):
        assert "有用" in D.verdict(self._rep(0.0794, 0.2000, 0.0800, 0.0790)).replace("没用,还", "")

    def test_gain_within_the_floor_is_noise(self):
        v = D.verdict(self._rep(0.0794, 0.0804, 0.0744, 0.0844))  # gain +0.0010,地板 0.0050
        assert "噪声" in v

    def test_a_dead_control_makes_it_undecided(self):
        """★ 平移 GT 的对照没塌 ⇒ **未判** —— 尺子这次不成立,读数不许用。"""
        assert "未判" in D.verdict(self._rep(0.0794, 0.0278, 0.0292, 0.0624, ctrl_ok=False))

    def test_the_floor_is_the_max_of_both_controls(self):
        """耦合钉:地板取**抖动臂与错目标臂里更差的那个** —— 只取一个会把另一个的波动漏掉。"""
        rep = self._rep(0.10, 0.20, 0.09, 0.05)
        v = D.verdict(rep)
        assert "0.0500" in v, "地板应当是 max(|0.09−0.10|, |0.05−0.10|) = 0.05"


class TestDomainReference:
    def test_uses_the_mean_over_images_not_one_image(self, tmp_path):
        """★ 数据集级常量 —— 用单张图需要一个配对规则,而这里要的是**固定的、无标签的**常量。"""
        d = tmp_path / "training" / "image_2"
        d.mkdir(parents=True)
        for i, v in enumerate((40, 200)):
            Image.fromarray(np.full((8, 8, 3), v, np.uint8)).save(d / f"{i:06d}.png")
        m, _ = D.domain_reference(tmp_path, 10)
        solo, _ = D.lab_stats(np.full((8, 8, 3), 40, np.uint8))
        assert not np.allclose(m, solo), "两图取均值 ⇒ 不该等于任何单张"

    def test_empty_root_raises(self, tmp_path):
        (tmp_path / "training" / "image_2").mkdir(parents=True)
        with pytest.raises(SystemExit, match="目标域参考"):
            D.domain_reference(tmp_path, 10)

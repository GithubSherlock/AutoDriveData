"""融合判据(`eval_fusion.py`)的纯值部分:标定装配、帧号解析、**键名对表**。

## 里面有一条是**回归钉**,不是预防

`_STAT_FIELD` 是我在跑到一半才补的:原先的汇总写成

    tot[k] += getattr(st, k if k != "gt" else "gt_count")

—— 而 `ClassStats` 的字段是 `gt_count` / **`pred_count`**(不是 `pred`)。
「gt」被特判了,「pred」没有 ⇒ 跑到第二个档就 `AttributeError` 崩掉,**整轮白跑**。

⇒ `test_every_mapped_field_exists_on_class_stats` 钉的就是这条:键名对表**必须**
逐个 `hasattr` 通过。它塌掉的样子是"跑到一半崩",而崩之前已经烧掉的时间不可恢复。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.perception.compare import ClassStats
from autodrivedata.perception.eval_fusion import _STAT_FIELD, _parse_frames, load_calib


class TestStatFieldMap:
    def test_every_mapped_field_exists_on_class_stats(self):
        """★★ **回归钉**:`getattr` 猜名字的写法在这里会当场红,而不是跑到一半崩。"""
        for key, field in _STAT_FIELD.items():
            assert hasattr(ClassStats(), field), f"汇总键 {key!r} → 字段 {field!r} 在 ClassStats 上不存在"

    def test_the_five_aggregate_keys_are_covered(self):
        assert set(_STAT_FIELD) == {"gt", "pred", "tp", "fp", "fn"}

    def test_pred_maps_to_pred_count_not_pred(self):
        """★ 这正是当年崩掉的那一处:字段名是 `pred_count`。"""
        assert _STAT_FIELD["pred"] == "pred_count"
        assert _STAT_FIELD["gt"] == "gt_count"


class TestParseFrames:
    @pytest.mark.parametrize(
        ("spec", "want"),
        [("0-2", [0, 1, 2]), ("5", [5]), ("0-1,7,9-10", [0, 1, 7, 9, 10]), ("", [])],
    )
    def test_parse(self, spec, want):
        assert _parse_frames(spec) == want


class TestLoadCalib:
    @staticmethod
    def _write(tmp_path, *, r0=None):
        p2 = "6.210000e+02 0 6.205000e+02 0 0 6.210000e+02 1.870000e+02 0 0 0 1 0"
        tv = "1 0 0 0 0 1 0 -1.65 0 0 1 0.1"
        lines = [f"P2: {p2}", f"Tr_velo_to_cam: {tv}"]
        if r0 is not None:
            lines.append(f"R0_rect: {r0}")
        f = tmp_path / "calib.txt"
        f.write_text("\n".join(lines) + "\n")
        return f

    def test_p2_shape(self, tmp_path):
        p2, _ = load_calib(self._write(tmp_path))
        assert p2.shape == (3, 4)
        assert p2[0, 0] == pytest.approx(621.0)

    def test_no_r0_rect_still_works(self, tmp_path):
        _, t = load_calib(self._write(tmp_path))
        assert t.shape == (4, 4)
        assert t[2, 3] == pytest.approx(0.1)

    def test_r0_rect_is_applied_on_the_left(self, tmp_path):
        """★ **顺序**:`R0_rect @ Tr_velo_to_cam`。写反 ⇒ 投影整体偏,而偏多少随 `ry` 变
        ⇒ 看着像"检测框有点歪",不像口径错。

        用一个可分辨的 `R0_rect`(绕 y 转 180° ⇒ x、z 取负)把顺序量出来:
        左乘会让平移项 `(0, -1.65, 0.1)` 变成 `(-0, -1.65, -0.1)`;右乘则不动。
        """
        r0 = "-1 0 0 0 1 0 0 0 -1"
        _, t = load_calib(self._write(tmp_path, r0=r0))
        assert t[0, 3] == pytest.approx(0.0, abs=1e-9)
        assert t[2, 3] == pytest.approx(-0.1, abs=1e-9), "R0_rect 右乘了(平移项没被旋转)"
        assert t[1, 3] == pytest.approx(-1.65, abs=1e-9)

    def test_a_vendored_identity_r0_rect_changes_nothing(self, tmp_path):
        """反向对照:`R0_rect` 是单位阵时,有没有它必须**逐位相同**。"""
        _, a = load_calib(self._write(tmp_path))
        _, b = load_calib(self._write(tmp_path, r0="1 0 0 0 1 0 0 0 1"))
        assert np.allclose(a, b)


class TestCocoToKitti:
    """相机类 → KITTI 类。**这一列错了,类维度整个是错的,而它不报错。**"""

    def test_mapping_has_only_known_kitti_classes(self):
        from autodrivedata.perception.eval_fusion import COCO_TO_KITTI

        assert set(COCO_TO_KITTI.values()) <= {"Car", "Pedestrian", "Cyclist"}

    def test_person_bicycle_motorcycle_are_covered(self):
        """★ 融合的价值就在这三类上:LiDAR 分不出它们,类只能由相机给。"""
        from autodrivedata.perception.eval_fusion import COCO_TO_KITTI

        assert COCO_TO_KITTI[0] == "Pedestrian"
        assert COCO_TO_KITTI[1] == "Cyclist"
        assert COCO_TO_KITTI[3] == "Cyclist"

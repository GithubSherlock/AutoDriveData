"""语义 tag 表的**纯值**判据(不 import carla):编解码、三类映射、排除口径。

编号本身与 `carla.CityObjectLabel` 的等价性在 `tests/sim/test_sem_tags_oracle.py`
(那里才允许 import carla)。这里钉的是**下游会算错的那几处**:

| 判据 | 漏了会怎样 |
|---|---|
| 灰度往返 | 读写不是无损 ⇒ 每个 IoU 都带一层看不见的噪声 |
| `decode` 只认 `mode="L"` | 上色预览图读回来**仍是个数组**,能一路算完 —— 得到一整套错的数 |
| 排除集与三类**不相交** | 同一个 tag 既算 GT 又被排除 ⇒ 报出的占比与算出的 IoU 互相矛盾 |
| 掩膜按**逐类**拆 | 三类互相串味(把车道线算进可行驶)在总数上看不出来 |
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from autodrivedata.perception import sem_tags as st


class TestTables:
    def test_palette_covers_every_tag_zero_to_twenty_eight(self):
        assert sorted(st.PALETTE) == list(range(29))
        assert len(set(st.PALETTE.values())) == 29, "调色板有两个 tag 同色 ⇒ 反查编号不再唯一"

    def test_tag_names_is_the_inverse(self):
        for name, v in st.SEM_TAGS.items():
            assert st.TAG_NAMES[v] == name
        assert st.SEM_TAGS["Any"] == 255

    def test_gt_classes_are_exactly_the_three(self):
        assert st.GT_CLASSES == ("drivable", "lane", "obstacle")
        assert set(st.GT_CLASS_TAGS) == set(st.GT_CLASSES)
        for c in st.GT_CLASSES:
            assert st.GT_CLASS_TAGS[c], f"{c} 是空集"
            assert st.GT_CLASS_TAGS[c] <= set(st.SEM_TAGS.values())

    def test_classes_do_not_overlap_each_other(self):
        """★ 三类必须两两不交 —— 相交的话同像素会被算进两个类,IoU 各自都虚高。"""
        seen: set[int] = set()
        for c in st.GT_CLASSES:
            assert not (st.GT_CLASS_TAGS[c] & seen), f"{c} 与前面的类共用了 tag"
            seen |= set(st.GT_CLASS_TAGS[c])

    def test_excluded_tags_are_disjoint_from_every_class(self):
        """★ 排除集与三类不相交 —— 既算 GT 又报"被排除",占比与 IoU 会自相矛盾。"""
        in_classes = set().union(*st.GT_CLASS_TAGS.values())
        assert not (st.EXCLUDED_TAGS & in_classes)
        assert not (st.NON_DRIVABLE_SURFACE_TAGS & st.GT_CLASS_TAGS["drivable"])

    def test_the_excluded_obstacle_tags_are_exactly_the_ones_the_predictor_lacks(self):
        """把"预测器没有的类"**写死成清单** —— 预测侧加了 bicycle 就必须回来改这里,
        否则 GT 少一类、mIoU 会凭空变好。"""
        assert st.EXCLUDED_TAGS == {
            st.SEM_TAGS["Rider"],
            st.SEM_TAGS["Motorcycle"],
            st.SEM_TAGS["Bicycle"],
            st.SEM_TAGS["Train"],
        }


class TestPngCodec:
    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_round_trip_is_lossless(self, seed):
        tag = np.random.default_rng(seed).integers(0, 29, size=(37, 61)).astype(np.uint8)
        back = st.decode_tag_png(st.encode_tag_png(tag))
        np.testing.assert_array_equal(back, tag)
        assert back.dtype == np.uint8

    def test_round_trip_keeps_the_hard_tags(self):
        """0(未标注)与 255(Any)是最容易被"当背景吞掉"的两个值。"""
        tag = np.array([[0, 255], [29, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(st.decode_tag_png(st.encode_tag_png(tag)), tag)

    def test_encode_rejects_non_uint8_or_3d(self):
        with pytest.raises(ValueError):
            st.encode_tag_png(np.zeros((4, 4), dtype=np.int16))
        with pytest.raises(ValueError):
            st.encode_tag_png(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_decode_rejects_the_coloured_preview(self):
        """★ 最要命的一条:调色板图(3 通道)读回来**照样是个数组**,不拦就一路算完。"""
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        with pytest.raises(ValueError, match="灰度"):
            st.decode_tag_png(buf.getvalue())


class TestMasksAndShares:
    TAG = np.array(
        [
            [st.SEM_TAGS["Roads"], st.SEM_TAGS["Roads"], st.SEM_TAGS["RoadLines"]],
            [st.SEM_TAGS["Car"], st.SEM_TAGS["Sky"], st.SEM_TAGS["Bicycle"]],
            [st.SEM_TAGS["Pedestrians"], st.SEM_TAGS["Bus"], st.SEM_TAGS["Sidewalks"]],
        ],
        dtype=np.uint8,
    )

    def test_masks_split_by_class(self):
        m = st.masks_from_tags(self.TAG)
        assert set(m) == set(st.GT_CLASSES)
        assert m["drivable"].sum() == 2
        assert m["lane"].sum() == 1
        assert m["obstacle"].sum() == 3  # Car + Pedestrians + Bus,Bicycle 不算
        for c in st.GT_CLASSES:
            assert m[c].dtype == bool and m[c].shape == self.TAG.shape

    def test_excluded_share_counts_the_bicycle_and_the_sidewalk(self):
        s = st.excluded_share(self.TAG)
        assert s["excluded_obstacle"] == pytest.approx(1 / 9)  # Bicycle
        assert s["non_drivable_surface"] == pytest.approx(1 / 9)  # Sidewalks

    def test_excluded_share_never_exceeds_one(self):
        assert 0.0 <= st.excluded_share(self.TAG)["excluded_obstacle"] <= 1.0


class TestPaint:
    def test_known_tags_get_their_palette_colour(self):
        img = st.paint_tags(np.array([[1, 24]], dtype=np.uint8))
        assert tuple(img[0, 0]) == st.PALETTE[1]
        assert tuple(img[0, 1]) == st.PALETTE[24]

    def test_unknown_tag_is_flagged_not_silently_black(self):
        """未知 tag 画成品红 —— 画成黑的话它就跟 `NONE(0)` 混在一起,看不出图坏了。"""
        img = st.paint_tags(np.array([[200]], dtype=np.uint8))
        assert tuple(img[0, 0]) == (255, 0, 255)
        assert tuple(img[0, 0]) != st.PALETTE[0]

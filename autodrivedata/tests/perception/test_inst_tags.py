"""实例 id/类 的**纯值**核心:`inst_tags.py` 的编解码、实例抽取、类归属。

不碰 CARLA。这两条来自**实测**(2026-10-01 首采 `surround_inst_demo`),不是文档抄的:

① CARLA 的实例相机给**每个关卡网格**都发 id —— 首采 20 帧里 `Roads` 有 **113 个 id**、
   `Car` 只有 13 个。不筛,`instances_from_maps` 会返回几百个"实例",PQ 的分母里全是路面,
   而读数照样是个 0–1 的数。
② 实例相机的 R 通道 == 语义相机的 tag,**逐像素 1.000000 × 6 路**(采集器首帧自证)。
   这条成立,实例评测的类标签才能从 `sem_*/` 取而不必再落一份。

两条塌掉的样子都不是崩溃,所以都要钉。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.perception.inst_tags import (
    MAX_INSTANCE_ID,
    THING_TAGS,
    class_channel_matches_semantic,
    decode_instance_png,
    encode_instance_png,
    gt_class_names,
    instances_from_maps,
    is_scored_class,
    is_thing,
)
from autodrivedata.perception.sem_tags import SEM_TAGS


def _maps() -> tuple[np.ndarray, np.ndarray]:
    ids = np.zeros((10, 10), dtype=np.uint16)
    ids[0:2, 0:2] = 3  # 一块路面(关卡网格也有 id)
    ids[3:6, 3:6] = 9  # 一辆车
    cls = np.zeros((10, 10), dtype=np.uint8)
    cls[0:2, 0:2] = SEM_TAGS["Roads"]
    cls[3:6, 3:6] = SEM_TAGS["Car"]
    return ids, cls


class TestCodec:
    def test_roundtrip_is_lossless_including_the_extremes(self):
        """★ 0 与 65535 是最容易被"当背景吞掉"的两个值 —— 必须原样往返。"""
        a = np.array([[0, 1, 255, 256, MAX_INSTANCE_ID]], dtype=np.uint16)
        assert np.array_equal(decode_instance_png(encode_instance_png(a)), a)

    def test_out_of_range_raises_instead_of_truncating(self):
        """★ 截断会把两个物体并成一个,症状是"实例数少了几个" —— 而"少了几个"与
        "本来就没那么多"在报表上长得一样。"""
        with pytest.raises(ValueError, match="越界"):
            encode_instance_png(np.array([[MAX_INSTANCE_ID + 1]], dtype=np.int64))

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="越界"):
            encode_instance_png(np.array([[-1]], dtype=np.int64))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match=r"\(H,W\)"):
            encode_instance_png(np.zeros((2, 2, 3), dtype=np.uint16))

    def test_decode_rejects_a_non_16bit_mode(self):
        """★ 读错 mode **不会报错**,只会算出一套错的实例数(而且"看着正常")。"""
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(buf, format="PNG")
        with pytest.raises(ValueError, match="I;16"):
            decode_instance_png(buf.getvalue())


class TestInstancesFromMaps:
    def test_keeps_only_countable_things_by_default(self):
        """★ 实测:关卡网格也有 id。不筛的话 PQ 的分母里全是路面。"""
        ids, cls = _maps()
        got = instances_from_maps(ids, cls)
        assert [i.instance_id for i in got] == [9]
        assert got[0].cls_name == "Car" and got[0].gt_class == "obstacle"

    def test_things_only_false_keeps_everything(self):
        """反向对照:开关必须**真的有用**,否则它是死代码。"""
        ids, cls = _maps()
        got = instances_from_maps(ids, cls, things_only=False)
        assert [i.instance_id for i in got] == [3, 9]
        assert got[0].gt_class == "drivable"

    def test_background_id_zero_is_never_an_instance(self):
        ids, cls = _maps()
        assert all(i.instance_id != 0 for i in instances_from_maps(ids, cls, things_only=False))

    def test_mask_area_and_bbox(self):
        ids, cls = _maps()
        car = instances_from_maps(ids, cls)[0]
        assert car.area == 9
        assert car.bbox() == (3, 3, 5, 5)

    def test_consistent_flag_catches_a_mixed_instance(self):
        """★ 一个实例的像素跨多个语义类 = 解码或渲染有问题。判据必须**看得见**它,
        而不是拿众数继续算(那会把"解码错了"变成"看着正常的一个类")。"""
        ids, cls = _maps()
        cls[5, 5] = SEM_TAGS["Pedestrians"]  # 同一实例里混进另一个类
        car = instances_from_maps(ids, cls)[0]
        assert car.consistent is False

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="尺寸"):
            instances_from_maps(np.zeros((4, 4), np.uint16), np.zeros((4, 5), np.uint8))


class TestClassChannel:
    def test_identical_maps_match_perfectly(self):
        """★ 实测结论(1.000000 × 6 路)在这里被固化成可回归的断言。"""
        a = np.array([[1, 2], [14, 21]], dtype=np.uint8)
        assert class_channel_matches_semantic(a, a)["match"] == 1.0

    def test_one_pixel_off_is_detected(self):
        a = np.zeros((2, 2), dtype=np.uint8)
        b = a.copy()
        b[0, 0] = 14
        r = class_channel_matches_semantic(a, b)
        assert r["match"] == pytest.approx(0.75) and r["n"] == 4

    def test_shape_mismatch_raises(self):
        """两张图不同画幅时比的是**两个世界**,这条判据本身就没意义 —— 必须拦。"""
        with pytest.raises(ValueError, match="尺寸"):
            class_channel_matches_semantic(np.zeros((2, 2), np.uint8), np.zeros((3, 3), np.uint8))


class TestClassBuckets:
    def test_thing_tags_are_the_obstacle_class(self):
        assert THING_TAGS == frozenset(
            {SEM_TAGS["Pedestrians"], SEM_TAGS["Car"], SEM_TAGS["Truck"], SEM_TAGS["Bus"]}
        )

    def test_roads_and_lanes_are_stuff_not_things(self):
        """`drivable`/`lane` 是面不是个体 —— 数"有几块路面"没有意义。"""
        assert not is_thing(SEM_TAGS["Roads"])
        assert not is_thing(SEM_TAGS["RoadLines"])
        assert is_scored_class(SEM_TAGS["Roads"])  # 但**类级**判据里算

    def test_map_props_are_neither(self):
        """地图自带道具(`Dynamic`)两边都不算 —— 与 `sem_tags` 的口径必须一致。"""
        assert not is_thing(SEM_TAGS["Dynamic"])
        assert not is_scored_class(SEM_TAGS["Dynamic"])

    def test_gt_class_names_is_the_three_class_order(self):
        assert gt_class_names() == ("drivable", "lane", "obstacle")

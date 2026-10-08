"""`perception/domain_gap` 的判据。

两条要害,都是**实测踩出来的**:

1. ★★ **"打乱图与 GT 的配对"不是对照**:`ap_for` 把所有图的框**池化**之后再做一次全局贪心匹配
   ⇒ "哪张图的框"根本不进计算,重排是**恒等**(第一版实测 0.750 → 0.750)。有效对照必须
   让 **GT 多重集本身**变(平移框)。这是红线里那条的**第二次现形**。
2. ★ **类表必须走 `backends.norm_cls`**(与两个后端同一张表);
   COCO 的 80 类里只有 6 类落进本项目三类,其余**丢弃**是对的,不是漏了一半。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from autodrivedata.perception import domain_gap as DG


def _coco(tmp_path, n=3, size=64):
    """极小 COCO:`annotations/instances_val2017.json` + 若干 jpg。"""
    root = tmp_path / "COCO"
    (root / "annotations").mkdir(parents=True)
    (root / "val2017").mkdir()
    cats = [
        {"id": 1, "name": "person"},
        {"id": 3, "name": "car"},
        {"id": 2, "name": "bicycle"},
        {"id": 4, "name": "motorcycle"},
        {"id": 5, "name": "airplane"},  # 不落本项目三类 ⇒ 必须被丢
    ]
    anns = []
    for i in range(n):
        Image.fromarray(np.zeros((size, size, 3), np.uint8)).save(root / "val2017" / f"{i:012d}.jpg")
        anns.append({"image_id": i, "category_id": 3, "bbox": [10.0, 10.0, 20.0, 20.0]})
        anns.append({"image_id": i, "category_id": 5, "bbox": [0.0, 0.0, 5.0, 5.0]})
    (root / "annotations" / "instances_val2017.json").write_text(
        json.dumps({"categories": cats, "annotations": anns}), encoding="utf-8"
    )
    return root


class TestLoadCocoGt:
    def test_maps_names_through_the_shared_table(self, tmp_path):
        gt = DG.load_coco_gt(_coco(tmp_path) / "annotations" / "instances_val2017.json")
        assert gt[0] == [("Car", (10.0, 10.0, 30.0, 30.0))]

    def test_xywh_becomes_xyxy(self, tmp_path):
        """★ COCO 是 `[x,y,w,h]`,本项目全线是 `x1y1x2y2` —— 换错不会报错,只会全错。"""
        gt = DG.load_coco_gt(_coco(tmp_path) / "annotations" / "instances_val2017.json")
        x1, y1, x2, y2 = gt[0][0][1]
        assert (x2 - x1, y2 - y1) == (20.0, 20.0), "w/h 必须被加成 x2/y2"

    def test_classes_outside_the_project_table_are_dropped(self, tmp_path):
        """`airplane` 不在三类里 ⇒ 丢弃。这不是"漏了一半",是类表就只有三类。"""
        gt = DG.load_coco_gt(_coco(tmp_path) / "annotations" / "instances_val2017.json")
        assert all(c in ("Car", "Pedestrian", "Cyclist") for c, _ in gt[0])


class TestImagePaths:
    def test_only_images_with_project_gt(self, tmp_path):
        root = _coco(tmp_path)
        ann = DG.load_coco_gt(root / "annotations" / "instances_val2017.json")
        assert len(DG.image_paths(root / "val2017", ann, None)) == 3

    def test_limit_and_order_are_deterministic(self, tmp_path):
        root = _coco(tmp_path)
        ann = DG.load_coco_gt(root / "annotations" / "instances_val2017.json")
        got = DG.image_paths(root / "val2017", ann, 2)
        assert [p.stem for p in got] == sorted(p.stem for p in got)[:2]


class TestEvaluateCoco:
    @staticmethod
    def _perfect(img_path):
        """每张图都给出正好落在 GT 上的框。"""
        return [("Car", 0.9, (10.0, 10.0, 30.0, 30.0))]

    def test_perfect_detections_score_high(self, tmp_path):
        root = _coco(tmp_path)
        ann = DG.load_coco_gt(root / "annotations" / "instances_val2017.json")
        ev = DG.evaluate_coco(DG.image_paths(root / "val2017", ann, None), ann, self._perfect, verbose=False)
        assert ev["classes"]["Car"]["ap"] > 0.9 and ev["classes"]["Car"]["recall"] == 1.0

    def test_shifted_gt_control_collapses(self, tmp_path):
        """★★ **有效对照**:GT 平移 1/3 画幅 ⇒ AP 必须塌。

        ⚠️ 反面就是第一版的错:换成"打乱配对"读数**一个数不变**(池化 ⇒ 重排是恒等)。
        这条与下面 `test_pairing_shuffle_is_a_noop` 是**一对**,合起来证明
        "我们选对了对照的形状"。
        """
        root = _coco(tmp_path)
        ann = DG.load_coco_gt(root / "annotations" / "instances_val2017.json")
        imgs = DG.image_paths(root / "val2017", ann, None)
        ok = DG.evaluate_coco(imgs, ann, self._perfect, verbose=False)
        bad = DG.evaluate_coco(imgs, ann, self._perfect, control_shift=True, verbose=False)
        assert ok["classes"]["Car"]["ap"] > 0.9
        assert bad["classes"]["Car"]["ap"] < 0.1, "GT 挪走了 ⇒ 完美检测也应当配不上"

    def test_pairing_shuffle_is_a_noop(self, tmp_path):
        """★★ **留在案**:池化口径下,"谁配谁"不进计算 —— 所以那类对照**不是对照**。

        这条不是在测功能,是在**钉住一个已经踩过的坑**:如果哪天有人"顺手加回一个打乱臂",
        下面这个断言会告诉他那个臂**恒等于基线**,给不出任何信息。
        """
        root = _coco(tmp_path)
        ann = DG.load_coco_gt(root / "annotations" / "instances_val2017.json")
        imgs = DG.image_paths(root / "val2017", ann, None)
        # 同一批预测、同一批 GT,只把 GT 换成"下一张图的"⇒ 池化之后多重集完全相同
        rotated = {int(p.stem): ann[int(imgs[(k + 1) % len(imgs)].stem)] for k, p in enumerate(imgs)}
        a = DG.evaluate_coco(imgs, ann, self._perfect, verbose=False)
        b = DG.evaluate_coco(imgs, rotated, self._perfect, verbose=False)
        assert a["classes"]["Car"]["ap"] == pytest.approx(b["classes"]["Car"]["ap"]), (
            "池化口径下重排是恒等 —— 这类对照给不出信息(这正是第一版被自己打红的原因)"
        )

    def test_class_without_gt_does_not_dilute_map(self, tmp_path):
        """没有 GT 的类不计入 mAP(同 `eval_2d_ab.evaluate`)。"""
        root = _coco(tmp_path)
        ann = DG.load_coco_gt(root / "annotations" / "instances_val2017.json")
        ev = DG.evaluate_coco(DG.image_paths(root / "val2017", ann, None), ann, self._perfect, verbose=False)
        assert ev["n_classes"] == 1, "只有 Car 有 GT ⇒ 只算 1 类"

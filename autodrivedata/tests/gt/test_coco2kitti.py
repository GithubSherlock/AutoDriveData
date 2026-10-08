"""`gt/export/coco2kitti` + 放宽后的 `write_frame` 的判据。

这一层服务的是一次**跨仓交付**(数据给兄弟仓 `auto2dlabel` 的 2D 微调),而它的失效模式**全是静默的**:

- 2D-only root 产不出来 ⇒ 只能另写一个写入器 ⇒ **两张布局实现迟早漂**;
- 下游按**框高 ≥25 px** 过滤而**一个字不打印** ⇒ 交出去不知道多少能用;
- 声明了却**恒为 0 的类** ⇒ 照训不报错;
- 微调与测试**同一批图** ⇒ 读数好看但没有意义。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from autodrivedata.gt.export import coco2kitti as C
from autodrivedata.gt.export.kitti import frame_paths, write_frame


class TestWriteFrameRelaxed:
    """★ 放宽 `write_frame` 是为了产 2D-only root —— 但**既有路径必须逐字节不变**。"""

    def _calib(self):
        from autodrivedata.calib.core import KittiCalibOut

        # ⚠️ 字段名是**小写**的(`p2` / `tr_velo_to_cam`),不是 KITTI txt 里的大写
        return KittiCalibOut(p2=np.eye(3, 4), tr_velo_to_cam=np.eye(3, 4))

    def test_2d_only_root_writes_exactly_two_files(self, tmp_path):
        write_frame(
            tmp_path,
            "000000",
            image_png=b"PNG",
            labels=["Car 0.00 0 0.00 1 2 3 4 -1 -1 -1 -1000 -1000 -1000 -10"],
        )
        assert (tmp_path / "training/image_2/000000.png").is_file()
        assert (tmp_path / "training/label_2/000000.txt").is_file()
        # ⚠️ **不许建空目录** —— 空目录会让下游的 glob 以为"这一路存在"
        assert not (tmp_path / "training/velodyne").exists()
        assert not (tmp_path / "training/calib").exists()

    def test_giving_only_one_of_the_pair_raises(self, tmp_path):
        """★★ 只给一半会产出**既不是 3D root 也不是 2D root** 的目录,而两边都不报错。"""
        with pytest.raises(SystemExit, match="要么都给"):
            write_frame(
                tmp_path, "000000", image_png=b"PNG", labels=[], velodyne=np.zeros((1, 4), np.float32)
            )

    def test_full_3d_path_is_unchanged(self, tmp_path):
        """★ 给齐 velodyne + calib 时,产物与放宽前**逐字段相同**。"""
        p = write_frame(
            tmp_path,
            "000001",
            image_png=b"PNG",
            labels=["Car 0 0 0 1 2 3 4 1 1 1 1 1 1 0"],
            velodyne=np.arange(8, dtype=np.float32).reshape(2, 4),
            calib=self._calib(),
        )
        assert p.velodyne.is_file() and p.calib.is_file()
        assert np.fromfile(p.velodyne, dtype=np.float32).tolist() == list(range(8))
        assert (tmp_path / "training/velodyne").is_dir()

    def test_frame_paths_is_still_the_single_layout_owner(self, tmp_path):
        """布局由 `frame_paths` 拥有 —— 新路径不许自己拼目录名。"""
        assert frame_paths(tmp_path, "7").image == tmp_path / "training/image_2/000007.png"


class TestKittiLine:
    def test_fifteen_columns_with_placeholders(self):
        line = C.kitti_line("Car", (1.0, 2.0, 3.0, 4.0), truncated=0.0, occluded=0)
        f = line.split()
        assert len(f) == 15 and f[0] == "Car"
        assert f[8:11] == ["-1.00", "-1.00", "-1.00"], "3D 尺寸列是占位"
        assert f[11:14] == ["-1000.00", "-1000.00", "-1000.00"], "3D 位置列是占位"

    def test_it_matches_the_existing_line_format(self):
        """★ 与 `gt.core.box_to_gt_line` **同口径**:列数相同、类名与 2D 列位置相同。"""
        from autodrivedata.gt.core import LABEL_2_FIELDS

        assert len(C.kitti_line("Car", (1, 2, 3, 4), truncated=0.0, occluded=0).split()) == LABEL_2_FIELDS


class TestCocoBoxFlags:
    def test_iscrowd_becomes_occluded_2(self):
        assert C.coco_box_flags((10, 10, 50, 50), 100, 100, iscrowd=True)[1] == 2
        assert C.coco_box_flags((10, 10, 50, 50), 100, 100, iscrowd=False)[1] == 0

    def test_edge_touching_becomes_truncated(self):
        """COCO 没有截断字段 ⇒ 贴边当近似。⚠️ 给 0.5 = hard 档的**上限**,是刻意的。"""
        assert C.coco_box_flags((0, 10, 50, 50), 100, 100, iscrowd=False)[0] == 0.5
        assert C.coco_box_flags((0.5, 10, 50, 50), 100, 100, iscrowd=False)[0] == 0.5
        assert C.coco_box_flags((10, 10, 50, 50), 100, 100, iscrowd=False)[0] == 0.0

    def test_truncation_stays_inside_the_hard_band(self):
        """★ `0.5` 必须**不把框判死** —— hard 档的上限就是 `trunc <= 0.5`。

        ⚠️ 这里**不**去 import 兄弟仓的 `kitti_difficulty`:那会形成反向依赖
        (依赖单向 AutoDriveData → AutoLabel)。钉的是**契约那一侧**:贴边给 0.5 而不是 1.0。
        """
        assert C.coco_box_flags((0, 10, 50, 50), 100, 100, iscrowd=False)[0] <= 0.5


class TestSplit:
    def test_halves_are_disjoint_and_cover_everything(self):
        ids = list(range(100))
        a, b = C.split_ids(ids, holdout_frac=0.5, seed=0)
        assert len(a) + len(b) == 100 and not (set(a) & set(b))

    def test_it_is_seeded_and_reproducible(self):
        ids = list(range(50))
        assert C.split_ids(ids, seed=3) == C.split_ids(ids, seed=3)
        assert C.split_ids(ids, seed=3) != C.split_ids(ids, seed=4)

    def test_ids_are_plain_ints_not_numpy(self):
        """★ `permutation` 给的是 **numpy int64** ⇒ 直接进 `json.dumps` 会 `TypeError`(实测踩到)。"""
        a, _ = C.split_ids(list(range(10)), seed=0)
        assert all(type(v) is int for v in a)

    def test_overlap_is_caught(self):
        """★★ **反向自证**:故意造重叠,自证**必须红** —— 否则它从来没在判。"""
        with pytest.raises(SystemExit, match="重叠"):
            C.assert_disjoint([1, 2, 3], [3, 4])

    def test_an_empty_half_is_caught(self):
        with pytest.raises(SystemExit, match="空的"):
            C.assert_disjoint([1, 2, 3], [])


class TestCocoToKitti:
    @staticmethod
    def _coco(tmp_path, *, boxes, size=(200, 200)):
        """极小 COCO:一张图 + 若干框。`boxes` = [(cat_id, x, y, w, h, iscrowd)]"""
        root = tmp_path / "COCO"
        (root / "annotations").mkdir(parents=True)
        (root / "val2017").mkdir()
        Image.fromarray(np.zeros((size[1], size[0], 3), np.uint8)).save(root / "val2017/000000000001.jpg")
        cats = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}, {"id": 4, "name": "bicycle"}]
        anns = [
            {"image_id": 1, "category_id": c, "bbox": [x, y, w, h], "iscrowd": int(cr)}
            for c, x, y, w, h, cr in boxes
        ]
        (root / "annotations/instances_val2017.json").write_text(
            __import__("json").dumps({"categories": cats, "annotations": anns}), encoding="utf-8"
        )
        return root

    def test_it_writes_only_the_two_dirs_and_reports_counts(self, tmp_path):
        coco = self._coco(tmp_path, boxes=[(3, 10, 10, 60, 60, False), (1, 10, 100, 50, 50, False)])
        rep = C.coco_to_kitti(coco, tmp_path / "out")
        assert (tmp_path / "out/training/image_2/000001.png").is_file()
        assert not (tmp_path / "out/training/velodyne").exists()
        assert rep["per_class"] == {"Car": 1, "Pedestrian": 1}
        assert rep["n_boxes"] == 2

    def test_it_reports_the_small_box_fraction(self, tmp_path):
        """★★ **陷阱①的核心判据**:下游按 25 px 过滤会丢框,而它不打印 ⇒ 我们必须报。
        夹具里放一个 10 px 高的框 ⇒ `n_below_25px` 必须是 1。"""
        coco = self._coco(tmp_path, boxes=[(3, 10, 10, 60, 60, False), (3, 10, 100, 60, 10, False)])
        rep = C.coco_to_kitti(coco, tmp_path / "out")
        assert rep["box_h_px"]["n_below_25px"] == 1
        assert rep["box_h_px"]["frac_below_25px"] == pytest.approx(0.5)

    def test_it_reports_declared_but_absent_classes(self, tmp_path):
        """★★ **陷阱②**:COCO 里只有车 ⇒ 另外两类**必须被点名**,否则"0 实例的类"照训不报错。"""
        coco = self._coco(tmp_path, boxes=[(3, 10, 10, 60, 60, False)])
        rep = C.coco_to_kitti(coco, tmp_path / "out")
        assert rep["empty_classes"] == ["Cyclist", "Pedestrian"]

    def test_crowd_becomes_occluded_2_in_the_written_line(self, tmp_path):
        coco = self._coco(tmp_path, boxes=[(1, 10, 10, 60, 60, True)])
        C.coco_to_kitti(coco, tmp_path / "out")
        line = (tmp_path / "out/training/label_2/000001.txt").read_text().strip()
        assert line.split()[2] == "2", "iscrowd ⇒ 遮挡 2"

    def test_empty_annotation_raises(self, tmp_path):
        coco = self._coco(tmp_path, boxes=[(5, 10, 10, 5, 5, False)])  # 5 = airplane,不在类表
        with pytest.raises(SystemExit, match="没有可用的图"):
            C.coco_to_kitti(coco, tmp_path / "out")


class TestThatTheClassTableIsNotDuplicated:
    """★ 本仓在"两张表迟早漂"上踩过 ⇒ 类表只许有一张。"""

    def test_it_imports_the_shared_normalizer(self):
        import inspect

        src = inspect.getsource(C)
        assert "domain_gap import load_coco_gt" in src, "COCO 读取器必须复用,不许再写一份"
        assert "COCO_FALLBACK" not in src.replace("perception/backends.COCO_FALLBACK", ""), (
            "类表只许引用,不许在本模块里复制一份"
        )

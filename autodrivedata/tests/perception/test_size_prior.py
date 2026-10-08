"""`perception/size_prior` 的判据。

它修的是「LiDAR 簇只覆盖车的**可见部分**」,而两条纪律是**复核当场指出的**:

1. ★ **先验不许与评测同源**(train/test 泄漏)—— `label_2` 里本来就有真值尺寸,
   在同一个 root 上量再评就是**把答案抄进模型**;
2. ★ **要有"错先验"的对照** —— 打乱类别→尺寸后读数必须**变**,
   否则分不出"这个先验没用"与"代码根本没接上"。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.perception import size_prior as SP
from autodrivedata.perception.fusion import Cluster, cluster_to_box7


def _cluster(cx=10.0, l=2.0, w=1.8, h=1.5, cz=None):
    """一个"只见近面"的小簇:x 跨 2 m 而车长 4.5 m。"""
    cz = h / 2 if cz is None else cz
    return Cluster(center=(cx, 0.0, cz), half=(l / 2, w / 2, h / 2), n_points=50)


class TestAnchorCluster:
    def test_prior_length_is_used(self):
        cl = SP.anchor_cluster(_cluster(l=2.0), (4.5, 1.9, 1.6))
        assert cl.extent[0] == pytest.approx(4.5)

    def test_the_near_face_stays_put(self):
        """★★ **锚哪一面是有讲究的**:LiDAR 看到的是**近面**(朝 ego 那侧)。
        补全长应当是"把远面往外推",**不是把整辆车平移**。"""
        cl0 = _cluster(cx=10.0, l=2.0)
        near0 = cl0.center[0] - cl0.half[0]
        cl = SP.anchor_cluster(cl0, (4.5, 1.9, 1.6))
        near = cl.center[0] - cl.half[0]
        assert near == pytest.approx(near0), "近面必须不动"

    def test_the_bottom_stays_put(self):
        """底面(车轮着地)也必须在原地 —— 补的是高度方向的**上半截**。"""
        cl0 = _cluster(h=1.0, cz=0.5)
        bottom0 = cl0.center[2] - cl0.half[2]
        cl = SP.anchor_cluster(cl0, (4.5, 1.9, 1.6))
        assert cl.center[2] - cl.half[2] == pytest.approx(bottom0)

    def test_center_moves_away_from_ego(self):
        cl0 = _cluster(cx=10.0, l=2.0)
        assert SP.anchor_cluster(cl0, (4.5, 1.9, 1.6)).center[0] > cl0.center[0]


class TestFitFromRoot:
    def _root(self, tmp_path, dims):
        d = tmp_path / "r" / "training" / "label_2"
        d.mkdir(parents=True)
        for i, (l, w, h) in enumerate(dims):
            (d / f"{i:06d}.txt").write_text(f"Car 0 0 0 0 0 1 1 {h} {w} {l} 1 1 1 0\n", encoding="utf-8")
        return tmp_path / "r"

    def test_takes_the_median_not_the_mean(self, tmp_path):
        """★ 少数卡车会把**均值**拉走,中位不会。"""
        root = self._root(tmp_path, [(4.5, 1.9, 1.6), (4.6, 1.9, 1.6), (12.0, 2.5, 3.5)])
        p = SP.fit_from_root(root, ["000000", "000001", "000002"])
        assert p["dims"]["Car"][0] == pytest.approx(4.6), "中位,不是 6.9 的均值"

    def test_records_the_source_root(self, tmp_path):
        """★★ **这是防泄漏的那把锁** —— `eval_fusion` 拿它和评测 root 比,相同就报警。"""
        root = self._root(tmp_path, [(4.5, 1.9, 1.6)] * 4)
        assert SP.fit_from_root(root, [f"{i:06d}" for i in range(4)])["source_root"] == str(root)

    def test_too_few_samples_raises(self, tmp_path):
        root = self._root(tmp_path, [(4.5, 1.9, 1.6)] * 2)
        with pytest.raises(SystemExit, match="样本不够"):
            SP.fit_from_root(root, ["000000", "000001"])


class TestShufflePrior:
    def test_it_actually_permutes(self):
        """★ 打乱**必须真的变** —— 恰好恒等的"打乱"会给出假对照。"""
        prior = {"source_root": "x", "dims": {"Car": [4.5, 1.9, 1.6], "Truck": [8.0, 2.5, 3.0]}}
        sh = SP.shuffle_prior(prior, seed=0)
        assert sh["dims"] != prior["dims"] and sh["shuffled"] is True

    def test_the_value_set_is_preserved(self):
        prior = {"source_root": "x", "dims": {"Car": [4.5, 1.9, 1.6], "Truck": [8.0, 2.5, 3.0]}}
        sh = SP.shuffle_prior(prior, seed=3)
        assert sorted(map(tuple, sh["dims"].values())) == sorted(map(tuple, prior["dims"].values()))

    def test_single_class_falls_back_to_component_swap(self):
        """★★ 实测踩到:**单类时打乱配对是恒等**(本项目 LiDAR 档只有一类几何)
        ⇒ 那条对照**退化**,必须换成**打乱分量**(一辆 4.5 m 宽的车显然是错的)。"""
        prior = {"source_root": "x", "dims": {"Car": [4.5, 1.9, 1.6]}}
        sh = SP.shuffle_prior(prior, seed=0)
        assert sh["dims"] != prior["dims"], "单类时恒等 ⇒ 对照退化成假对照"
        assert sh["shuffle_how"] == "component-swap"
        assert sh["dims"]["Car"] == [1.9, 4.5, 1.6], "l ↔ w"


class TestClusterToBox7Wiring:
    def test_prior_none_reproduces_the_old_output(self):
        """★ **默认关 ⇒ 归档产物逐字节复现** —— 与 `--occluders`/`--depth` 同一条纪律。"""
        v2c = np.eye(4)
        cl = _cluster(l=2.0)
        assert cluster_to_box7(cl, v2c).l == pytest.approx(2.0)
        assert cluster_to_box7(cl, v2c, prior=None).l == pytest.approx(2.0)

    def test_prior_changes_the_length(self):
        v2c = np.eye(4)
        cl = _cluster(l=2.0)
        got = cluster_to_box7(cl, v2c, prior={"dims": {"Car": [4.5, 1.9, 1.6]}})
        assert got.l == pytest.approx(4.5) and got.w == pytest.approx(1.9)

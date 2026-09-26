"""autodrivedata/cluster.py 手算锚点单测。"""

from __future__ import annotations

import numpy as np

from autodrivedata.perception.cluster import cluster_boxes, dbscan


def _two_clusters() -> np.ndarray:
    """两个明显分离簇(车/行人):簇 A 中心 (2,0,1),簇 B 中心 (-3,1,0.5)。"""
    rng = np.random.default_rng(0)
    a = np.column_stack([rng.normal(2, 0.2, 40), rng.normal(0, 0.2, 40), rng.normal(1, 0.1, 40)])
    b = np.column_stack([rng.normal(-3, 0.2, 25), rng.normal(1, 0.2, 25), rng.normal(0.5, 0.1, 25)])
    # 噪声
    noise = np.column_stack([rng.uniform(-5, 5, 10), rng.uniform(-5, 5, 10), rng.uniform(0, 2, 10)])
    return np.vstack([a, b, noise])


class TestDbscan:
    def test_two_clusters(self):
        pts = _two_clusters()
        labels = dbscan(pts, eps=0.6, min_samples=3)
        # 恰 2 个聚类
        uniq = [c for c in np.unique(labels) if c >= 0]
        assert len(uniq) == 2
        # 两簇点数
        n0 = int((labels == 0).sum())
        n1 = int((labels == 1).sum())
        assert min(n0, n1) >= 20  # 各簇至少 20 点

    def test_isolated_noise(self):
        pts = np.array([[0.0, 0.0, 0.0, 1.0], [10.0, 10.0, 10.0, 1.0], [10.1, 10.1, 10.1, 1.0]])
        labels = dbscan(pts, eps=0.5, min_samples=2)
        # A 孤立为噪声,BC 一簇
        assert labels[0] == -1
        assert labels[1] == labels[2] != -1

    def test_empty(self):
        assert dbscan(np.zeros((0, 4))).shape == (0,)


class TestClusterBoxes:
    def test_orders_by_size(self):
        labels = np.array([0, 0, 0, 0, 1, 1])
        pts = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.1, 0.0, 0.0, 1.0],
                [0.2, 0.0, 0.0, 1.0],
                [0.3, 0.0, 0.0, 1.0],
                [5.0, 5.0, 5.0, 1.0],
                [5.1, 5.0, 5.0, 1.0],
            ]
        )
        boxes = cluster_boxes(pts, labels)
        assert len(boxes) == 2
        assert boxes[0]["n_points"] == 4  # 大簇在前
        assert boxes[0]["min_dist"] == 0.0

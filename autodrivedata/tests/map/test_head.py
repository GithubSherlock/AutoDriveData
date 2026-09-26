"""MapTR head 单测:匹配正确性 / 置换等价 / 代价语义 / 损失 / 前向形状。"""

from __future__ import annotations

import numpy as np
import torch

from autodrivedata.map.maptr.head import MapTRHead, maptr_loss, match_assign


def _line_forward() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)


def _line_reversed() -> np.ndarray:
    return _line_forward()[::-1]


def test_match_assign_basic() -> None:
    """类 1 一条 GT:点精确的 query 匹配成功,其余背景;类 0 无 GT 全背景。"""
    num_classes, num_vec = 2, 3
    gt = [[], [_line_forward()]]
    pred_points = np.zeros((num_classes * num_vec, 4, 2), dtype=np.float32)
    pred_points[4] = _line_forward()  # 类 1 组第 1 个 query 点精确
    pred_logits = np.full((num_classes * num_vec, num_classes + 1), -10.0)
    pred_logits[4, 2] = 10.0  # 类 1 logit 高
    cls_t, pts_t, mask = match_assign(pred_points, pred_logits, gt, num_classes, num_vec)
    np.testing.assert_array_equal(cls_t, [0, 0, 0, 0, 2, 0])
    np.testing.assert_array_equal(mask, [False, False, False, False, True, False])
    np.testing.assert_allclose(pts_t[4], _line_forward())


def test_match_assign_permutation_equivariant() -> None:
    """GT 存为反向:双向增强仍能零代价匹配,且目标取与预测一致的朝向(置换等价)。"""
    num_classes, num_vec = 1, 1
    gt = [[_line_reversed()]]
    pred_points = np.zeros((1, 4, 2), dtype=np.float32)
    pred_points[0] = _line_forward()
    pred_logits = np.array([[10.0, 10.0]])  # 类 1 logit 高
    cls_t, pts_t, mask = match_assign(pred_points, pred_logits, gt, num_classes, num_vec)
    assert cls_t[0] == 1 and mask[0]
    np.testing.assert_allclose(pts_t[0], _line_forward())


def test_match_assign_more_gt_than_queries() -> None:
    """GT 数 > query 数:方阵化 pad 后只匹配 num_vec 个(匈牙利不匹配 pad 行/列)。"""
    num_classes, num_vec = 1, 3
    gt = [[np.array([[i, 0.0], [i + 1, 0.0]], dtype=np.float32) for i in range(5)]]
    pred_points = np.zeros((3, 2, 2), dtype=np.float32)
    pred_points[1] = gt[0][2]
    pred_logits = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    _, _, mask = match_assign(pred_points, pred_logits, gt, num_classes, num_vec)
    assert mask.sum() == num_vec
    assert mask[1]  # 点精确的 query 必被匹配


def test_match_assign_cost_tradeoff() -> None:
    """代价 = −logit + 5·L1:点更准但置信低的 query 会被高置信 query 抢走匹配。"""
    num_classes, num_vec = 1, 2
    gt = [[_line_forward()]]
    pred_points = np.zeros((2, 4, 2), dtype=np.float32)
    pred_points[0] = _line_forward()  # 点完美
    pred_points[1] = _line_forward() + 10.0  # 点偏 10m
    pred_logits = np.array([[0.0, -20.0], [0.0, 100.0]])  # q0 类 0 置信极低,q1 极高
    cls_t, _, mask = match_assign(pred_points, pred_logits, gt, num_classes, num_vec)
    # q0: −logit = 20 + 0 = 20;q1: −100 + 5·10 = −50 → q1 胜
    assert mask[1] and not mask[0]
    assert cls_t[1] == 1


def test_maptr_loss() -> None:
    pred_logits = torch.randn(2, 4, 3)
    pred_points = torch.randn(2, 4, 4, 2)
    cls_targets = torch.tensor([[0, 1, 0, 0], [2, 0, 0, 0]])
    pts_mask = torch.tensor([[False, True, False, False], [True, False, False, False]])
    pts_targets = pred_points.detach().clone()  # 与预测一致 → pts loss ≈ 0
    out = maptr_loss(pred_logits, pred_points, cls_targets, pts_targets, pts_mask)
    assert set(out) == {"cls", "pts", "total"}
    assert float(out["pts"]) < 1e-5
    assert bool(torch.isfinite(out["total"]))
    assert (
        float(out["total"]) == float(out["cls"] + out["pts"])
        or abs(float(out["total"] - out["cls"] - out["pts"])) < 1e-6
    )


def test_head_forward_shapes_and_backward() -> None:
    head = MapTRHead(num_classes=2, num_vec=3, num_pts=4, num_layers=2, embed_dims=32, bev_dims=16)
    bev = torch.randn(1, 16, 20, 10)
    out = head(bev)
    assert out["pred_logits"].shape == (1, 6, 3)
    assert out["pred_points"].shape == (1, 6, 4, 2)
    (out["pred_logits"].sum() + out["pred_points"].sum()).backward()
    assert head.anchor.grad is not None and bool(torch.isfinite(head.anchor.grad).all())


def test_anchor_init_within_bev() -> None:
    head = MapTRHead(num_classes=4, num_vec=50, num_pts=20)
    a = head.anchor.detach()
    assert a.shape == (200, 20, 2)
    # 锚线在 BEV 范围内(x ∈ [−15, 15], y ∈ [−30, 30])
    assert bool(torch.all(a[..., 0] >= -15.0)) and bool(torch.all(a[..., 0] <= 15.0))
    assert bool(torch.all(a[..., 1] >= -30.0)) and bool(torch.all(a[..., 1] <= 30.0))

"""Chamfer AP 单测:距离语义 / 匹配语义 / 阈值口径 / 边界情形。

注意:Chamfer 距离 ≠ 平移量。3 点折线平移 1m 时内点重合,CD = 2/6 = 1/3;
要得到"CD = 平移量"的精确情形用单点折线。
"""

from __future__ import annotations

import numpy as np

from autodrivedata.chamfer_ap import (
    chamfer_ap,
    chamfer_ap_per_class,
    chamfer_cost_matrix,
    chamfer_distance,
    match_greedy,
)


def _line() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)


def _pt(x: float) -> np.ndarray:
    return np.array([[x, 0.0]], dtype=np.float32)


def test_chamfer_distance_basic() -> None:
    assert chamfer_distance(_line(), _line()) == 0.0
    # 单点:CD = 欧氏距离
    assert chamfer_distance(_pt(0.0), _pt(1.0)) == 1.0
    # 3 点折线平移 1m:内点重合,CD = (1+0+0 + 0+0+1)/6 = 1/3
    assert abs(chamfer_distance(_line(), _line() + [1.0, 0.0]) - 1 / 3) < 1e-9
    # 对称性(同一距离矩阵,行和 + 列和互换)
    assert chamfer_distance(_line(), _line() + [2.0, 3.0]) == chamfer_distance(_line() + [2.0, 3.0], _line())
    # 空输入 → inf
    assert np.isinf(chamfer_distance(np.zeros((0, 2)), _line()))


def test_match_greedy_semantics() -> None:
    gt = [_line(), _line() + [10.0, 0.0]]
    # 完美命中两个 → 2 TP 0 FP 0 FN
    assert match_greedy([_line(), _line() + [10.0, 0.0]], gt, thr=0.5) == (2, 0, 0)
    # 偏移 1m(CD=1/3):阈值 0.3 全不中,阈值 0.4 全中
    shifted = [_line() + [1.0, 0.0], _line() + [11.0, 0.0]]
    assert match_greedy(shifted, gt, thr=0.3) == (0, 2, 2)
    assert match_greedy(shifted, gt, thr=0.4) == (2, 0, 0)
    # GT 不复用:两个预测争同一个 GT,近者(CD=0.1)得,远者(CD=0.5)FP
    tp, fp, fn = match_greedy([_line() + [0.5, 0.0], _line() + [0.1, 0.0]], [_line()], thr=1.0)
    assert (tp, fp, fn) == (1, 1, 0)
    # 无 GT → 全 FP;无预测 → 全 FN
    assert match_greedy([_line()], [], thr=0.5) == (0, 1, 0)
    assert match_greedy([], [_line()], thr=0.5) == (0, 0, 1)


def test_chamfer_ap_thresholds() -> None:
    # 完美预测 → AP 1.0;无预测 → 0.0
    assert chamfer_ap([_line()], [_line()]) == 1.0
    assert chamfer_ap([], [_line()]) == 0.0
    # 单点偏移 0.8m:阈值 0.5 不中、1.0/1.5 中 → AP = (0 + 1 + 1) / 3 = 2/3
    ap = chamfer_ap([_pt(0.8)], [_pt(0.0)])
    assert abs(ap - 2 / 3) < 1e-9
    # 偏移 2m:三阈值全不中 → 0
    assert chamfer_ap([_pt(2.0)], [_pt(0.0)]) == 0.0


def test_chamfer_ap_per_class() -> None:
    preds = [[_line()], [], [_pt(0.8)], []]
    gts = [[_line()], [_line()], [_pt(0.0)], []]
    aps, mean = chamfer_ap_per_class(preds, gts)
    assert aps[0] == 1.0 and aps[1] == 0.0 and aps[3] == 0.0
    assert abs(aps[2] - 2 / 3) < 1e-9
    assert abs(mean - (1.0 + 0.0 + 2 / 3 + 0.0) / 4) < 1e-9


def _rnd_poly(rng: np.random.Generator, n_pts: int) -> np.ndarray:
    return rng.uniform(-5, 5, size=(n_pts, 2))


def test_chamfer_cost_matrix_matches_pairwise() -> None:
    """向量化代价矩阵与逐对 chamfer_distance 同口径(chunk=3 跨块验证 q2p 累积)。

    矩阵内部 float32、逐对口径 float64,允许 ~1e-3m 误差;语义错误是 O(1)
    量级(曾现 2× 偏差),1e-3 容差足够锁定。
    """
    rng = np.random.default_rng(7)
    preds = [_rnd_poly(rng, 20) for _ in range(9)]
    gts = [_rnd_poly(rng, int(rng.integers(1, 26))) for _ in range(7)]
    fast = chamfer_cost_matrix(preds, gts, chunk=3)
    slow = np.array([[chamfer_distance(p, g) for g in gts] for p in preds])
    assert np.allclose(fast, slow, atol=2e-3, rtol=1e-3)


def test_chamfer_cost_matrix_empty() -> None:
    assert chamfer_cost_matrix([], []).shape == (0, 0)
    assert chamfer_cost_matrix([_line()], []).shape == (1, 0)
    assert chamfer_cost_matrix([], [_line()]).shape == (0, 1)


def test_match_greedy_precomputed_cost_equivalent() -> None:
    rng = np.random.default_rng(3)
    preds = [_rnd_poly(rng, 20) for _ in range(5)]
    gts = [_rnd_poly(rng, 12) for _ in range(4)]
    cost = chamfer_cost_matrix(preds, gts)
    for thr in (0.5, 1.5, 4.0):
        assert match_greedy(preds, gts, thr, cost=cost) == match_greedy(preds, gts, thr)

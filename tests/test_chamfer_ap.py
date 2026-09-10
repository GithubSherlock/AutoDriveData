"""Chamfer AP 单测:距离语义 / 匹配语义 / 阈值口径 / 边界情形。

注意:Chamfer 距离 ≠ 平移量。3 点折线平移 1m 时内点重合,CD = 2/6 = 1/3;
要得到"CD = 平移量"的精确情形用单点折线。
"""

from __future__ import annotations

import numpy as np

from autodrivedata.chamfer_ap import (
    chamfer_ap,
    chamfer_ap_per_class,
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

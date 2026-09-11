"""GPU chamfer 代价矩阵单测:与 numpy 纯值版交叉锁定。

CUDA 用例 skipif 保护;无 CUDA 报错路径在任意机器上可测(monkeypatch)。
GPU 内部 float32、CPU 版同为 float32 内部口径,交叉容差 2e-3 与
test_chamfer_ap 的逐对锁定同族;语义错误 O(1) 量级必被抓。
"""

import numpy as np
import pytest
import torch

from autodrivedata.chamfer_ap import chamfer_cost_matrix, match_greedy
from maptr_impl.chamfer_gpu import chamfer_cost_matrix_cuda

HAS_CUDA = torch.cuda.is_available()


def _rnd_poly(rng: np.random.Generator, n_pts: int) -> np.ndarray:
    return rng.uniform(-5, 5, size=(n_pts, 2))


def test_gpu_cost_requires_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    line = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(RuntimeError, match="CUDA"):
        chamfer_cost_matrix_cuda([line], [line])


@pytest.mark.skipif(not HAS_CUDA, reason="需要 CUDA")
def test_gpu_cost_matches_cpu() -> None:
    """GPU 代价矩阵与 numpy 版一致(chunk=3 跨块,变长折线)。"""
    rng = np.random.default_rng(11)
    preds = [_rnd_poly(rng, 20) for _ in range(9)]
    gts = [_rnd_poly(rng, int(rng.integers(1, 26))) for _ in range(7)]
    gpu = chamfer_cost_matrix_cuda(preds, gts, chunk=3)
    cpu = chamfer_cost_matrix(preds, gts, chunk=3)
    assert np.allclose(gpu, cpu, atol=2e-3, rtol=1e-3)


@pytest.mark.skipif(not HAS_CUDA, reason="需要 CUDA")
def test_gpu_cost_empty() -> None:
    assert chamfer_cost_matrix_cuda([], []).shape == (0, 0)
    line = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert chamfer_cost_matrix_cuda([line], []).shape == (1, 0)
    assert chamfer_cost_matrix_cuda([], [line]).shape == (0, 1)


@pytest.mark.skipif(not HAS_CUDA, reason="需要 CUDA")
def test_gpu_cost_feeds_greedy_equivalently() -> None:
    """GPU 代价矩阵喂给共享的 match_greedy,三阈值判定与 CPU 版一致。"""
    rng = np.random.default_rng(5)
    preds = [_rnd_poly(rng, 20) for _ in range(6)]
    gts = [_rnd_poly(rng, 12) for _ in range(4)]
    gpu_cost = chamfer_cost_matrix_cuda(preds, gts)
    for thr in (0.5, 1.5, 4.0):
        assert match_greedy(preds, gts, thr, cost=gpu_cost) == match_greedy(preds, gts, thr)

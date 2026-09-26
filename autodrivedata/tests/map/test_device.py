"""autodrivedata/map/maptr/device.py 单测:自适应显存实测的确定性回退路径 + CUDA 冒烟。

确定性路径(OOM 回退)在有无 GPU 的机器上都走同一分支;真实测量路径
(每样本增量 > 0 → 顶到 max_batch)需 CUDA,skipif 保护。
"""

from collections.abc import Callable

import pytest
import torch

from autodrivedata.map.maptr.device import auto_tune_batch_size, get_gpu_free_memory_gb, measure_batch_memory

HAS_CUDA = torch.cuda.is_available()


def _tiny_step(dev: str = "cuda", oom: int | None = None) -> Callable[[int], None]:
    """返回 step_fn(bs);oom 为 None 时正常,为 N 时 bs==N 抛 CUDA OOM。"""

    def step(bs: int) -> None:
        if oom is not None and bs == oom:
            raise RuntimeError("CUDA out of memory. Tried to allocate 9.00 GiB")
        # 1<<20 元素 = 4MB/样本:必须足够大,否则 batch 2 的激活会复用 batch 1
        # 释放的缓存块,memory_reserved 增量为 0,测量退化为 None
        x = torch.randn(bs, 1 << 20, device=dev, requires_grad=True)
        (x @ x.t()).sum().backward()

    return step


def test_get_gpu_free_memory_gb_shape():
    gb = get_gpu_free_memory_gb()
    if HAS_CUDA:
        assert gb is not None and gb >= 0.0
    else:
        assert gb is None


def test_measure_warmup_oom_returns_none():
    # warmup 步 OOM:无论有无 GPU 都确定返回 None
    assert measure_batch_memory(_tiny_step(oom=1)) is None


def test_auto_tune_warmup_oom_falls_back_to_min():
    assert auto_tune_batch_size(_tiny_step(oom=1), min_batch=1, max_batch=16) == 1


def test_auto_tune_probe_oom_falls_back_to_min():
    # probe 步(batch=2)OOM:measure 返回保守上界 → 回退 min_batch
    assert auto_tune_batch_size(_tiny_step(oom=2), min_batch=1, max_batch=16) == 1


def test_measure_probe_oom_conservative_upper_bound():
    # probe OOM → (batch=1 全量增量, 1),调用方据此回退 batch=1
    measured = measure_batch_memory(_tiny_step(oom=2))
    if HAS_CUDA:
        assert measured is not None and measured[1] == 1
    else:
        assert measured is None


@pytest.mark.skipif(not HAS_CUDA, reason="需要 CUDA")
def test_auto_tune_cuda_reaches_max_batch():
    # 真实测量:小模型每样本增量极微,预算远大于增量 → 顶到 max_batch 钳制
    assert auto_tune_batch_size(_tiny_step(), max_batch=4) == 4


@pytest.mark.skipif(not HAS_CUDA, reason="需要 CUDA")
def test_auto_tune_cuda_explicit_min_batch():
    # 显式 min_batch=3:即便预算充裕也不得低于 min(钳制下界)
    assert auto_tune_batch_size(_tiny_step(), min_batch=3, max_batch=4) == 4

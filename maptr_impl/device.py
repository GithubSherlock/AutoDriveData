"""GPU 显存自适应实测工具——参考 AutoLabel auto2dlabel/tools/device.py 的思路。

训练场景与 AutoLabel 推理探针的差异:每样本显存增量测的是 forward+backward
(激活 + 激活梯度),探针步走完整训练步(含 optimizer.step——优化器状态与
batch 无关,属一次性常驻内存,不影响线性外推)。

口径(与 AutoLabel 一致):
- 增量法:memory_reserved 只增不减,只测差值、从不 reset;
- budget = 空闲显存 × safety_factor(留余量给其他进程与估计误差);
- bs = 1 + floor(budget / 每样本字节)(batch=1 已驻留,每加一份花 per_sample)。
"""

from __future__ import annotations

from collections.abc import Callable

# 显存预算安全系数:留 15% 余量给缓存碎片与其他进程(与 AutoLabel 同口径)
SAFETY_FACTOR = 0.85


def get_gpu_free_memory_gb() -> float | None:
    """当前空闲显存(GB);synchronize + empty_cache 后读取;无 CUDA 返回 None。"""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info(0)[0] / (1024**3)


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


def measure_batch_memory(step_fn: Callable[[int], None]) -> tuple[int, int] | None:
    """实测每样本训练步显存增量,返回 (per_sample_bytes, probe_batch)。

    step_fn(bs) 跑一个完整训练步(forward+backward+step)。warmup 步 batch=1
    同时驻留优化器状态与 cudnn kernel;probe 步 batch=2 的线性增量 = 每样本
    激活成本。probe OOM → 返回 (batch=1 全量增量, 1) 作保守上界(调用方
    回退 batch=1)。无 CUDA / warmup OOM / 增量 <= 0 → None。
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    r0 = torch.cuda.memory_reserved(0)
    try:
        step_fn(1)
        torch.cuda.synchronize()
    except RuntimeError as e:
        if _is_oom(e):
            return None
        raise
    r1 = torch.cuda.memory_reserved(0)

    try:
        step_fn(2)
        torch.cuda.synchronize()
    except RuntimeError as e:
        if _is_oom(e):
            torch.cuda.empty_cache()
            return (r1 - r0, 1)
        raise
    r2 = torch.cuda.memory_reserved(0)

    per = r2 - r1
    if per <= 0:
        return None
    return per, 2


def auto_tune_batch_size(
    step_fn: Callable[[int], None],
    min_batch: int = 1,
    max_batch: int = 16,
    safety_factor: float = SAFETY_FACTOR,
) -> int:
    """实测推荐训练 batch;无 GPU / 失败 / 探测仅容单图 → min_batch。

    budget = 空闲显存 × safety_factor;bs = 1 + floor(budget / per_sample);
    钳制到 [min_batch, max_batch]。
    """
    try:
        import torch
    except ImportError:
        return min_batch
    if not torch.cuda.is_available():
        return min_batch

    measured = measure_batch_memory(step_fn)
    if measured is None:
        return min_batch
    per_sample, probe_batch = measured
    if probe_batch == 1:
        return min_batch

    torch.cuda.empty_cache()
    budget = torch.cuda.mem_get_info(0)[0] * safety_factor
    bs = 1 + int(budget // per_sample)
    return max(min_batch, min(max_batch, bs))

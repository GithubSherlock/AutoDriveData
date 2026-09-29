"""autodrivedata/runtime/device.py 单测:自适应显存实测的确定性回退路径 + CUDA 冒烟。

确定性路径(OOM 回退)在有无 GPU 的机器上都走同一分支;真实测量路径
(每样本增量 > 0 → 顶到 max_batch)需 CUDA,skipif 保护。
"""

from collections.abc import Callable

import pytest
import torch

from autodrivedata.runtime.device import (
    get_gpu_free_memory_gb,
    measure_train_batch_memory,
    tune_train_batch_size,
)

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
    assert measure_train_batch_memory(_tiny_step(oom=1)) is None


def test_auto_tune_warmup_oom_falls_back_to_min():
    assert tune_train_batch_size(_tiny_step(oom=1), min_batch=1, max_batch=16) == 1


def test_auto_tune_probe_oom_falls_back_to_min():
    # probe 步(batch=2)OOM:measure 返回保守上界 → 回退 min_batch
    assert tune_train_batch_size(_tiny_step(oom=2), min_batch=1, max_batch=16) == 1


def test_measure_probe_oom_conservative_upper_bound():
    # probe OOM → (batch=1 全量增量, 1),调用方据此回退 batch=1
    measured = measure_train_batch_memory(_tiny_step(oom=2))
    if HAS_CUDA:
        assert measured is not None and measured[1] == 1
    else:
        assert measured is None


@pytest.mark.skipif(not HAS_CUDA, reason="需要 CUDA")
def test_auto_tune_cuda_reaches_max_batch():
    # 真实测量:小模型每样本增量极微,预算远大于增量 → 顶到 max_batch 钳制
    assert tune_train_batch_size(_tiny_step(), max_batch=4) == 4


@pytest.mark.skipif(not HAS_CUDA, reason="需要 CUDA")
def test_auto_tune_cuda_explicit_min_batch():
    # 显式 min_batch=3:即便预算充裕也不得低于 min(钳制下界)
    assert tune_train_batch_size(_tiny_step(), min_batch=3, max_batch=4) == 4


# ================================================================ 2026-09-29 并入时的新增钉


class TestDeterministicFallbacks:
    """无 GPU / 探针失败时的确定性回退 —— 这批在**任何**机器上都走同一分支。"""

    def test_no_cuda_returns_min(self, monkeypatch):
        from autodrivedata.runtime.device import tune_batch_size

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert tune_batch_size(_tiny_step(dev="cpu")) == 1
        assert measure_train_batch_memory(_tiny_step(dev="cpu")) is None
        assert get_gpu_free_memory_gb() is None

    def test_explicit_batch_wins(self):
        """`resolve_batch_params` 的优先级:**显式 > 实测 > 逐图**。"""
        from autodrivedata.runtime.device import resolve_batch_params

        bs, _ = resolve_batch_params(None, [], explicit_batch=7, explicit_workers=0)
        assert bs == 7

    def test_explicit_workers_respected(self):
        from autodrivedata.runtime.device import resolve_batch_params

        _bs, nw = resolve_batch_params(None, [], explicit_batch=1, explicit_workers=9)
        assert nw == 9

    def test_workers_recommendation_is_bounded(self):
        from autodrivedata.runtime.device import recommend_num_workers

        assert 0 <= recommend_num_workers() <= 4


class TestTf32:
    """TF32 口径:**批量与逐帧结果可比的前提**(见 runtime/device.py 头注的实测表)。

    不钉"关掉之后 AP 会怎样"(那要真权重 + 真数据,归集成测),只钉**开关本身被关掉** ——
    它是全局副作用,漏掉时症状是"换 batch 就换结果",极难归因。
    """

    def test_disable_tf32_turns_both_switches_off(self):
        from autodrivedata.runtime.device import disable_tf32

        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        disable_tf32()
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is False

    def test_idempotent(self):
        from autodrivedata.runtime.device import disable_tf32

        disable_tf32()
        disable_tf32()
        assert torch.backends.cudnn.allow_tf32 is False


class TestDeviceDetection:
    def test_get_device_returns_a_known_backend(self):
        from autodrivedata.runtime.device import get_device

        assert get_device() in {"cuda", "mps", "cpu"}

    def test_cuda_machine_reports_cuda(self):
        """本机是 CUDA 机器 —— 顺带证明探测不是恒返回 cpu 的哑实现。"""
        from autodrivedata.runtime.device import get_device

        if HAS_CUDA:
            assert get_device() == "cuda"


class TestStackBatchShapes:
    """`eval_maptr._stack_batch` 的**形状契约**钉(两个窗口模式不同,错了报在模型内部)。

    ⚠️ 形状写错的症状是堆栈**指向模型**(`TemporalFusion` 报窗口不匹配 / GKT 报维度),
    看着像权重坏了 —— 故这里用合成样本把两种契约钉在**入口处**,不需要真模型与真数据。
    """

    CAMS = ["CAM_FRONT", "CAM_BACK"]

    @staticmethod
    def _item(k: int | None) -> dict:
        one = {c: torch.zeros(3, 8, 8) for c in TestStackBatchShapes.CAMS}
        if k is None:  # 单帧:dict + pose
            return {"images": one, "pose": torch.zeros(6), "gt": {}}
        return {"images": [dict(one) for _ in range(k)], "poses": torch.zeros(k, 6), "gt": {}}

    def test_single_frame_stacks_to_B(self):
        from autodrivedata.map.eval_maptr import _stack_batch

        images, poses = _stack_batch([self._item(None) for _ in range(3)], self.CAMS, torch.device("cpu"))
        assert isinstance(images, dict)
        assert all(t.shape == (3, 3, 8, 8) for t in images.values())
        assert tuple(poses.shape) == (3, 6)

    @pytest.mark.parametrize("k", [2, 3])
    def test_temporal_stacks_to_BK(self, k: int):
        """时序必须给出 **list[K]** 与 **(B, K, 6)** —— 给 (B,6) 会被模型按 K=1 解释。"""
        from autodrivedata.map.eval_maptr import _stack_batch

        images, poses = _stack_batch([self._item(k) for _ in range(2)], self.CAMS, torch.device("cpu"))
        assert isinstance(images, list), "时序模式下 images 必须是长度 K 的 list"
        assert len(images) == k
        assert all(t.shape == (2, 3, 8, 8) for t in images[0].values())
        assert tuple(poses.shape) == (2, k, 6)

    def test_missing_camera_is_a_loud_error(self):
        """相机名对不上必须**当场报错**:静默少一路 = 第 i 路图与它学过的语义错位。"""
        from autodrivedata.map.eval_maptr import _stack_batch

        with pytest.raises(KeyError):
            _stack_batch([self._item(None)], ["CAM_FRONT", "CAM_NOT_EXIST"], torch.device("cpu"))


class TestModuleLocation:
    """落点回归钉:`runtime` 是**唯一**的设备工具落点(2026-09-29 用户裁决)。

    防两种复发:① 又拷一份回 `utils/`(那层 `_PURE` 禁 torch,拷回去守卫会红);
    ② **两份并在**、调用方各 import 一个 —— 那才是真正危险的形态(`SAFETY_FACTOR` 分叉)。
    """

    def test_no_second_device_module(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        hits = sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("device.py")
            if "__pycache__" not in p.parts and "ipynb_checkpoints" not in str(p)
        )
        assert hits == ["runtime/device.py"], f"设备工具应当只有一份,实测 {hits}"

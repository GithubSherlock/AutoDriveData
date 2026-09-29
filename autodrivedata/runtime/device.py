"""运行时设备工具:设备探测 / TF32 口径 / 显存体检 / 批量超参实测。

**为什么单独成一个能力面**(2026-09-29 用户裁决):这些是**跨能力面**的 torch 运行时问题 ——
`map/maptr` 用它调**训练** batch、`eval_maptr` 用它调**推理** batch、`live_studio` 用它
做设备探测。放 `utils/` 不行(**那一层是 `_PURE`,禁 torch**,有守卫强制),放 `map/maptr/`
则把"设备工具"挂在了地图能力下。故新建本目录,在 `test_layer_guard.LAYER_RULES` 里
声明为「允 torch、禁 carla」。

来源:AutoLabel `auto2dlabel/tools/device.py` + 本项目 `map/maptr/device.py`(原训练版)。
合并时做了三件**必须**的事(见下),不是简单堆在一起。

## ⚠️ 合并时的三处处置(每一处都是"不处置就会静默出错")

1. **两版 `auto_tune_batch_size` 同名不同签名** —— 训练版的探针是 `step_fn(bs)`(跑
   forward+backward+step),推理版的探针是 `infer_fn(paths)`(只前向)。同名会让调用方
   传错探针而**两边都不报错**(都只是"测出来偏小/偏大")。现改名为
   `tune_train_batch_size` / `tune_infer_batch_size`,**名字里带用途**。
2. **`SAFETY_FACTOR` 两处不同值**(训练版 0.95 / AutoLabel 0.85)。同名常量两个值是
   最坏的一种分叉:调用方读到的取决于 import 哪个模块。现**统一为 `SAFETY_FACTOR` 一个**,
   取 **0.95** —— 那是本项目 `--batch 0` 的**现行行为**,改它等于静默变更默认档位。
   (原文 docstring 写"留 15%"与 0.95 自相矛盾,一并订正为"留 5%"。)
3. **`resolve_batch_params` 依赖 `auto2dlabel.schema.task_plan`** —— 那是 AutoLabel 的包,
   **在 autodrivedata env 不可导入**(实测 `find_spec` 抛 ModuleNotFoundError)⇒ 一调即崩。
   现**去掉静态表兜底**(它本来就是"模型未加载时的规划阶段"才走,本项目没有那个阶段),
   只保留「显式 > 实测 > 逐图」三段。

## TF32 口径(本模块最有价值的一条)

`cudnn.allow_tf32` 在 torch 2.x **默认为 True**,而全仓原本无人设置它。实测(2026-09-29,
`maptr_v2_singleF.pt`、帧级留出 80 帧、`--score-thr 0.2`):

| `cudnn.allow_tf32` | divider | boundary | mAP |
|---|---|---|---|
| **True**(现状默认) | 0.3167 | 0.3077 | 0.3048 |
| **False** | **0.3152** | **0.3075** | **0.3043** |

**False 那一行与 Plan2 §P-M.12 归档逐位相同**(0.3152 / 0.3075 / 0.3043)⇒ 归档是在
TF32 关闭的口径下测的,今天的默认值把它**静默改掉了**。`pred/gt` 计数两边完全一致
(1343/907),变的只是**匹配时的数值边界**。

⇒ **`disable_tf32()` 不是"精度洁癖",是把口径钉死的一步**。批量推理尤其依赖它:
TF32 会让 batch=1 与 batch>1 走**不同的 cudnn kernel**,于是"批量结果 == 逐图结果"
这条闸门在不关它时**根本不成立**。**新增批量推理入口时,先调它。**
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

#: 显存预算安全系数:budget = 空闲显存 × 它。
#: **取 0.95 = 留 5% ±给缓存碎片与其他进程** —— 这是本项目 `--batch 0` 的现行行为,
#: 别照 AutoLabel 的 0.85 改回去(那会静默变更默认档位选出的 batch)。
SAFETY_FACTOR = 0.95

#: 启动体检阈值(GB):低于此值黄字提醒。
#: 实测洞:上次运行 Ctrl+Z 挂起/未正常退出 ⇒ 残留进程占着显存 ⇒ 本次跑到一半神秘 OOM。
#: (与红线「停训练必须连 DataLoader worker 一起收 —— 判据是 `nvidia-smi` 归零」同一族问题)
_HEADROOM_WARN_GB = 2.0


# ---------------------------------------------------------------- 设备探测


def get_device() -> str:
    """最佳可用设备:`"cuda"` > `"mps"` > `"cpu"`。无 torch 时回落 `"cpu"`。"""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_info() -> str:
    """可读描述,如 `CUDA (NVIDIA GeForce RTX 3090)` / `CPU`。"""
    try:
        import torch
    except ImportError:
        return "CPU"
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "MPS (Apple Silicon GPU)"
    return "CPU"


def print_device() -> None:
    """打印设备,并**先关 TF32**(入口统一口径;见模块头注)。"""
    disable_tf32()
    print(f"[device] {get_device_info()}")


# ---------------------------------------------------------------- 数值口径


def disable_tf32() -> None:
    """关闭 TF32,让 batch=1 与 batch>1 走同一套 kernel(**批量推理正确性的前提**)。

    TF32 是 Ampere+ 的 19-bit 尾数加速格式。开启时同一模型对不同 batch 会选**不同的
    cudnn kernel**,首层卷积即产生 ~1e-5 量级差异,经百层 + attention softmax + 阈值化
    放大后能翻转具体实例。AutoLabel 侧实测(yolo26x-obb):关闭后 raw 差 <1e-4、结果逐位
    一致;本项目实测见模块头注表(帧级留出 mAP 0.3048 → **0.3043 = 归档值**)。

    幂等;无 torch 时静默跳过。吞吐代价 ~10%(正确性优先)。
    """
    try:
        import torch
    except ImportError:
        return
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def get_gpu_free_memory_gb() -> float | None:
    """当前空闲显存(GB);`synchronize` + `empty_cache` 后读取;无 CUDA 返回 None。"""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info(0)[0] / (1024**3)


def check_gpu_headroom() -> bool:
    """启动显存体检:空闲 < 阈值时提醒(**不阻断**,用户可能有意共存)。返回是否充足。"""
    free_gb = get_gpu_free_memory_gb()
    if free_gb is None:
        return True  # 无 CUDA 无需体检
    if free_gb >= _HEADROOM_WARN_GB:
        return True
    print(
        f"⚠ GPU 显存紧张(空闲 {free_gb:.2f} GiB < {_HEADROOM_WARN_GB:.0f} GiB)"
        " —— 可能上次运行未正常退出、仍有进程占用。"
    )
    print("  请 `nvidia-smi` 检查并清理残留进程后重跑;若属正常共存可忽略。")
    return False


def recommend_num_workers() -> int:
    """DataLoader workers 按 CPU 核数推荐(与显存无关):每 4 核 1 个,上限 4。"""
    return min(4, max(0, (os.cpu_count() or 1) // 4))


# ---------------------------------------------------------------- 批量超参实测


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


def _probe_two_largest(image_paths: list[str]) -> list[str]:
    """选面积最大的 2 张不同图作显存探针(批显存 ≈ 面积和,最大图最保守)。"""
    from PIL import Image

    sizes: list[tuple[int, str]] = []
    for p in image_paths:
        try:
            with Image.open(p) as im:
                sizes.append((im.size[0] * im.size[1], p))
        except Exception:
            continue  # 读不出来的图不参与探测
    sizes.sort(reverse=True)
    unique: list[str] = []
    for _area, p in sizes:  # noqa: B007 — 面积只用于排序
        if p not in unique:
            unique.append(p)
        if len(unique) == 2:
            break
    return unique


def _linear_increment(run: Callable[[int], None]) -> tuple[int, int] | None:
    """增量法公共骨架:`memory_reserved` 只增不减,只测差值、从不 reset。

    `run(k)` 跑一次 batch=k 的负载(训练步或推理)。warmup(batch=1)同时驻留优化器状态 /
    cudnn kernel,B 步的线性增量 = 每单位成本。B 步 OOM → 返回 (batch=1 全量增量, 1)
    作保守上界(调用方回退 batch=1)。无 CUDA / warmup OOM / 增量 ≤ 0 → None。
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    r0 = torch.cuda.memory_reserved(0)
    try:
        run(1)
        torch.cuda.synchronize()
    except RuntimeError as e:
        if _is_oom(e):
            return None
        raise
    r1 = torch.cuda.memory_reserved(0)

    try:
        run(2)
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


def _bs_from_budget(per_unit: int, min_batch: int, max_batch: int, safety_factor: float) -> int:
    """`bs = 1 + floor(空闲×安全系数 / 每单位)`(batch=1 已驻留),钳到 [min, max]。"""
    import torch

    torch.cuda.empty_cache()
    budget = torch.cuda.mem_get_info(0)[0] * safety_factor
    return max(min_batch, min(max_batch, 1 + int(budget // per_unit)))


def tune_batch_size(
    run: Callable[[int], None],
    min_batch: int = 1,
    max_batch: int = 64,
    safety_factor: float = SAFETY_FACTOR,
) -> int:
    """**通用**批量实测:`run(bs)` 跑一次 batch=bs 的负载 → 推荐 batch。

    ★ 这一版才是核心,下面两个"训练/推理"函数只是它的薄封装。原设计里训练版收
    `step_fn(bs)`、推理版收 `infer_fn(paths)` —— **后者其实更窄**:本项目的推理载荷是
    "数据集样本(6 相机 + 位姿)",不是"图路径列表",照那个签名根本接不上。统一成
    `run(bs)` 之后两边都只是"怎么把一个 batch 跑起来"的差异,不再有两套探针契约。

    无 GPU / 失败 / 探测仅容单样本 → `min_batch`。
    """
    measured = _linear_increment(run)
    if measured is None:
        return min_batch
    per_unit, probe_batch = measured
    if probe_batch == 1:
        return min_batch
    return _bs_from_budget(per_unit, min_batch, max_batch, safety_factor)


def measure_train_batch_memory(step_fn: Callable[[int], None]) -> tuple[int, int] | None:
    """实测每样本**训练步**显存增量(forward + backward + step)。

    训练与推理的每单位成本**不是一回事**:训练要驻留激活梯度与优化器状态,推理不要。
    故两个探针分开,别拿推理的公式去猜训练的 batch。
    """
    return _linear_increment(step_fn)


def tune_train_batch_size(
    step_fn: Callable[[int], None],
    min_batch: int = 1,
    max_batch: int = 16,
    safety_factor: float = SAFETY_FACTOR,
) -> int:
    """实测推荐**训练** batch(薄封装:`step_fn(bs)` 跑一个完整训练步)。"""
    return tune_batch_size(step_fn, min_batch, max_batch, safety_factor)


def measure_infer_batch_memory(
    infer_fn: Callable[[list[str]], Any], image_paths: list[str]
) -> tuple[int, int] | None:
    """实测每张图**推理**显存增量;返回 `(per_img_bytes, probe_batch)`。

    未指定路径后缀过滤 —— 显存只取决于尺寸,取面积最大的两张即最保守(AutoLabel 口径)。
    """
    probes = _probe_two_largest(image_paths)
    if not probes:
        return None
    return _linear_increment(lambda k: infer_fn(probes[:k]))


def tune_infer_batch_size(
    infer_fn: Callable[[list[str]], Any],
    image_paths: list[str],
    min_batch: int = 1,
    max_batch: int = 64,
    safety_factor: float = SAFETY_FACTOR,
) -> int:
    """实测推荐**推理** batch(薄封装:探针取面积最大的 2 张图,`infer_fn(paths)`)。

    ⚠️ 这个签名(`infer_fn(paths)`)只适合**按图路径**推理的入口;本项目 MapTR 的推理
    载荷是数据集样本(6 相机 + 位姿),那边直接用更通用的 `tune_batch_size(run)`。
    """
    probes = _probe_two_largest(image_paths)
    if not probes:
        return min_batch
    return tune_batch_size(lambda k: infer_fn(probes[:k]), min_batch, max_batch, safety_factor)


def resolve_batch_params(
    infer_fn: Callable[[list[str]], Any] | None,
    image_paths: list[str],
    explicit_batch: int | None = None,
    explicit_workers: int | None = None,
    max_batch: int = 64,
) -> tuple[int, int]:
    """批量**推理**超参统一入口:`显式 > 实测 > 逐图`。返回 `(batch, workers)`。

    与 AutoLabel 原版的差别:去掉了「静态表兜底」那一段 —— 它依赖
    `auto2dlabel.schema.task_plan`,而**本 env 装不了那个包**(见模块头注第 3 条)。
    本项目没有"模型未加载的规划阶段",静态表本来也用不上。

    **一律先关 TF32**:这是"批量结果 == 逐图结果"成立的前提,放在统一入口里,
    调用方就不必各自记得。
    """
    disable_tf32()
    nw = recommend_num_workers() if explicit_workers is None else max(0, explicit_workers)
    if explicit_batch is not None:
        return max(1, explicit_batch), nw

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False
    if infer_fn is None or not cuda_ok:
        return 1, nw
    return tune_infer_batch_size(infer_fn, image_paths, max_batch=max_batch), nw

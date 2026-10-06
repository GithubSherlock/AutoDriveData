"""`import gsplat` **之前**必须做完的两件事 —— 本包唯一的落点。

## 为什么单独立一个模块

gsplat 会在**第一次真正用到光栅化**时 JIT 编译 CUDA 内核,而那之前有两件事必须先定好,
定了就晚了:

1. **`TORCH_CUDA_ARCH_LIST`** —— 决定内核编给哪个架构。本机 shell/direnv 预置的是
   `7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0;12.1+PTX`,含 torch 2.6 **不认识**的 `10.3`
   ⇒ gsplat 内部抛 `Unknown CUDA arch (10.3)`,**报错点完全指不到是环境变量的事**
   (2026-10-03 实测)。旧写法只 `setdefault`,挡不住"预置了个坏值"。
2. **`CUDA_HOME`** —— torch 的扩展缓存按**构建 flags 的 hash** 定位 `.so`:环境不同 ⇒
   hash 不同 ⇒ 它**重编并覆盖规范那份**,而不是报"环境不对"。实测代价 ≈1 h,
   见 [docs/edit-3dgs-plan.md](../../docs/edit-3dgs-plan.md) §A.4.0。

**这两件事原先写在 `train_3dgs_mini` 里**。2026-10-04 加了 `render_gs` / `edit_gs` /
`eval_edit` 三个同样要 import gsplat 的入口 ⇒ 再复制三份就是"同一个口径四套实现"。
抽到这里,并由 `tests/gs/test_cuda_env.py` **AST 钉住**:本包内凡出现 `import gsplat`
的模块,**必须先 import 本模块**(无论 `import` 还是 `from ... import`)。

## 怎么用

```python
from autodrivedata.gs import cuda_env  # noqa: F401  —— 必须在 import gsplat 之前

import gsplat  # noqa: E402
```

导入顺序即语义:本模块在导入时**就**把环境设好(副作用是有意的),
`ARCH_NOTE` / `CUDA_NOTE` 供各入口在 runlog 里留痕(`cuda_env.note()`)。
"""

from __future__ import annotations

import os
import shutil

import torch


def _arch_list_usable(spec: str) -> bool:
    """这份 `TORCH_CUDA_ARCH_LIST` 本机 torch 认不认 —— **直接问那个会抛错的函数**。

    不另写一份"已知架构表":那张表会随 torch 版本漂,而真正决定成败的就是
    `_get_cuda_arch_flags()` 自己。私有 API 若改名,这里会 `AttributeError` **当场暴露**,
    好过静默选错架构(那要等内核编译出来才报,且报的是误导性的共享内存错误)。
    """
    import torch.utils.cpp_extension as _ext

    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = spec
    try:
        _ext._get_cuda_arch_flags()
        return True
    except ValueError:
        return False
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


#: 本卡架构与最终生效的 `TORCH_CUDA_ARCH_LIST`(供启动时打印,便于事后归因)。
ARCH_NOTE = ""
#: `CUDA_HOME` 的实况(或"没设"的告警)。
CUDA_NOTE = ""

if torch.cuda.is_available():
    _cc = torch.cuda.get_device_capability()
    _detected = f"{_cc[0]}.{_cc[1]}"
    _preset = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if _preset and not _arch_list_usable(_preset):
        os.environ["TORCH_CUDA_ARCH_LIST"] = _detected
        ARCH_NOTE = (
            f"预置的 TORCH_CUDA_ARCH_LIST={_preset!r} 本机 torch 不接受 ⇒ "
            f"改用本卡架构 {_detected}(shell/direnv 预置的全量列表常含未来架构)"
        )
    else:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _detected)
        ARCH_NOTE = f"TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}"

_HOME = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if _HOME:
    CUDA_NOTE = f"CUDA_HOME={_HOME}"
else:
    # ⚠️ **只报一行**,不写成因 —— "那次为什么会重编"没查实,能说的是
    #    「没按头注「用法」给 CUDA_HOME 时,这个风险就在」。
    CUDA_NOTE = f"未设 CUDA_HOME / CUDA_PATH ⇒ nvcc 将用 PATH 上的 {shutil.which('nvcc') or '未找到'}"
    print(
        f"[cuda] ⚠️ {CUDA_NOTE};gsplat 可能被重编(含一次性探针 `import gsplat`)"
        "—— 规范用法要 CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH,"
        "见 train_3dgs_mini 模块头注「用法」"
    )


def ensure_gsplat():
    """**在环境设好之后** import gsplat 并把模块返回。本包内取 gsplat 的**唯一入口**。

    ⚠️ **不要写 `import gsplat`** —— 那让正确性依赖"这个文件里两行 import 的先后",
    而 `ruff format --fix`(isort 规则)**会把 `import gsplat` 挪到 first-party 的
    `autodrivedata.*` 之前**,于是 cuda_env 还没设架构、gsplat 就先进来了 ——
    实测就这么被自动修坏过一次。顺序写成调用,格式化器就动不了它。
    判据 = `tests/gs/test_cuda_env.py` 的 AST 钉(本包内除本模块外不许直接 import gsplat)。
    """
    import gsplat

    return gsplat


def note() -> str:
    """两行归属信息(`[arch]` / `[cuda]`),各入口照原样打印 + 进 runlog。"""
    return f"[arch] {ARCH_NOTE}\n[cuda] {CUDA_NOTE}"


def home() -> str | None:
    """runlog 的 `cuda_home` 字段用(跨 build 的数不许混比,那要求事后查得出来)。"""
    return _HOME

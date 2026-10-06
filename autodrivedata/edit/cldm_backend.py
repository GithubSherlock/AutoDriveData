"""ControlNet 官方 repo 的加载器 —— **唯一落点**。

外部 repo 克隆在 `hdMapGitHub/ControlNet`,**保持 pristine**;
每一次"本机环境与 2021 年的 repo 对不上"的适配都收在本模块,**不在 repo 里改一行**。

## 四处适配(每一处都"报错指不到真因",详录见 [docs/edit-image-plan.md](../../docs/edit-image-plan.md) §1.2)

| # | 现象 | 真因 |
|---|---|---|
| 1 | `No module named 'pytorch_lightning'` | `cldm.cldm` → `ldm.models.diffusion.ddpm` → `import pytorch_lightning`。**单文件 AST 扫描抓不到传递链** |
| 2 | `No module named 'pytorch_lightning.utilities.distributed'` | PL 2.x 把它挪到 `.utilities.rank_zero`。推理链上只用到 `rank_zero_only` ⇒ **模块替身,不降级 PL** |
| 3 | `UnpicklingError` | **torch 2.6 的 `torch.load` 默认 `weights_only=True`**,而这份 ckpt 里带 omegaconf 对象 |
| 4 | `Missing key(s): cond_stage_model.transformer.final_layer_norm.*` | **CLIP 文本塔的包装层被上游改了**:2021 年 ckpt 带 `text_model.` 前缀,本机 `transformers` 的键是裸的(196 个) |

⚠️ **还有第五处不是 import 问题、但同样"指不到真因"**:官方 demo 假定输入是**方图**
(Gradio 画布产出的就是方的)。喂长条图时 `resize_image` 按长边缩,得到的 hint 与 latent 对不上,
报 `RuntimeError: size of tensor a (64) must match tensor b (208)` ——
**它只说形状不匹配,没说"你喂了张长条图"**。⇒ 本方提供 `to_square_rgb`,生成侧必须过它。

⚠️ **第六处(2026-10-05 实测撞上)**:**`MidasDetector` 直接吃原图,自己不 resize**。
官方 demo 是 `apply_midas(resize_image(HWC3(img), 512))` —— 短边缩到 512 **且边长取 64 的倍数**。
把一张 375×375 原样喂进去,ViT 会算出 23×23 的 patch 网格,而位置嵌入按 24×24 存的,报

```
RuntimeError: The size of tensor a (577) must match the size of tensor b (530)
```

—— 报错在 `annotator/midas/midas/vit.py:forward_flex` 的 `x = x + pos_embed`,
**既没提"你没 resize",也没提是哪个尺寸错了**。⇒ `midas_depth()` 内部走官方那两步,调用方不必知道。

## 权重完整性

`weights/controlnet/MANIFEST.txt` 记 sha256;`verify_weight` **对不上直接抛**。
这不是洁癖:本项目实测过**三次**"size 完全正确、sha256 不对"的坏文件
(同一文件上换下载器/并发写导致),不校验就发现不了,见 MANIFEST 头注。
"""

from __future__ import annotations

import functools
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np

from autodrivedata.utils.paths import PROJECT_ROOT

#: 外部 repo 相对项目根的位置(**不入库**,`hdMapGitHub/` 已被 `.gitignore` 覆盖)。
REPO_REL = Path("hdMapGitHub/ControlNet")
WEIGHTS_REL = Path("weights/controlnet")
MANIFEST_NAME = "MANIFEST.txt"

#: 已接的变体 → 权重文件名。**每个都是 5.71 GB**,盘上各占一份。
VARIANT_WEIGHTS = {
    "canny": "control_sd15_canny.pth",
    "depth": "control_sd15_depth.pth",
}

#: MiDaS 估计器权重(官方 `annotator/ckpts/` 的那份;本仓改落 `weights/`)。
MIDAS_WEIGHT = "dpt_hybrid-midas-501f0c75.pt"


def repo_dir() -> Path:
    d = PROJECT_ROOT / REPO_REL
    if not d.is_dir():
        raise FileNotFoundError(
            f"没找到 ControlNet 官方 repo:{d}\n"
            "  本模块**不 clone** 它(第三方内容不入库)—— 需要时手工 clone 到该位置。"
        )
    return d


def weights_dir() -> Path:
    return PROJECT_ROOT / WEIGHTS_REL


# --------------------------------------------------------------------------- 权重校验
def read_manifest(path: Path | None = None) -> dict[str, str]:
    """读 `MANIFEST.txt` → {文件名: sha256}。格式:`<name>  size=<n>  sha256=<hex>`。"""
    p = path or (weights_dir() / MANIFEST_NAME)
    if not p.exists():
        raise FileNotFoundError(f"缺 MANIFEST:{p} —— 没有它就无法判断权重是否完整")
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split()[0]
        for tok in line.split():
            if tok.startswith("sha256="):
                out[name] = tok.split("=", 1)[1]
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_weight(path: Path, *, expected: str | None = None) -> str:
    """校验权重的 sha256,返回实际值;**对不上直接抛**。

    结果按 `(size, mtime)` 缓存进 `<权重>.sha256` 边车 —— 5.71 GB 每次重算要十几秒。

    ⚠️ **边车只加速,不替代判据**,但它有一条**明说的边界**:
    命中缓存的唯一条件是 `(size, mtime)` 都没变 **且** 边车里的哈希 == MANIFEST 里那个。
    所以 —— **能改 mtime 的人可以骗过它**(`touch -r` 就能保住 mtime)。
    **本判据防的是"下载坏了",不是"有人篡改"**:威胁模型里没有后者,
    而且真要防篡改,边车本身也得有签名,那是另一件事。
    边车内容坏(哈希对不上 MANIFEST)时**回退去重算**,不算命中。
    """
    if not path.exists():
        raise FileNotFoundError(f"缺权重:{path}")
    exp = expected or read_manifest().get(path.name)
    if exp is None:
        raise KeyError(f"MANIFEST 里没有 {path.name} —— 新下的权重必须先把 sha256 登记进去")
    st = path.stat()
    side = path.with_name(path.name + ".sha256")
    if side.exists():
        try:
            rec = json.loads(side.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rec = {}
        if rec.get("size") == st.st_size and rec.get("mtime") == int(st.st_mtime):
            if rec.get("sha256") == exp:
                return exp
    got = _sha256(path)
    if got != exp:
        raise ValueError(
            f"权重校验失败:{path.name}\n  期望 {exp}\n  实际 {got}\n"
            "  ⇒ 这个文件**不能用**。本项目踩过三次'尺寸正确但内容坏'的下载"
            "(同一文件上换下载器/并发写),重下请用单一 aria2c --checksum。"
        )
    side.write_text(
        json.dumps({"size": st.st_size, "mtime": int(st.st_mtime), "sha256": got}), encoding="utf-8"
    )
    return got


# --------------------------------------------------------------------------- import 适配
_READY = False

#: `_ensure_offline` 是否已经报过一次(只报一次,别刷屏)。
_OFFLINE_NOTED = False


def _ensure_offline() -> None:
    """**本机够不着 `huggingface.co`** ⇒ 让 transformers 直接用缓存,别去试。

    ## 为什么这不是"顺手加个环境变量"

    2026-10-06 实测:不设 `HF_ENDPOINT` / `HF_HUB_OFFLINE` 时,`CLIPTextModel.from_pretrained`
    会去够 `huggingface.co` —— 而本机对它**不可达**。它**不抛异常**,而是**挂在重试上**:

    ```
    进程 STAT=Sl  WCHAN=wait_woken   %CPU 5.9   已跑 5:17   GPU 6191 MiB / 利用率 0%
    ```

    ⇒ 界面看起来就是"**还在加载模型**",而它永远不会结束。这类错最贵的地方是
    **它长得像正常等待** —— 我这次就等了 5 分钟才去查。

    ## 处置:只报不拦(同 `train_3dgs_mini` 的 `CUDA_NOTE` 那套纪律)

    两个都没设时,把 `HF_HUB_OFFLINE=1` 定下来并**打一行说明**;
    想真去下载的人**显式设 `HF_ENDPOINT` 即可覆盖**(那时这一行不动它)。
    """
    global _OFFLINE_NOTED
    if _OFFLINE_NOTED:
        return
    import os

    if os.environ.get("HF_ENDPOINT") or os.environ.get("HF_HUB_OFFLINE"):
        _OFFLINE_NOTED = True
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    _OFFLINE_NOTED = True
    print(
        "[cldm] 未设 HF_ENDPOINT / HF_HUB_OFFLINE ⇒ 定为 HF_HUB_OFFLINE=1(纯用本地缓存)。\n"
        "       本机够不着 huggingface.co,而 transformers 会**挂在网络重试上**"
        "(看着像'还在加载模型')。要真去下载就显式设 HF_ENDPOINT=https://hf-mirror.com。",
        flush=True,
    )


def ensure_importable(*, midas_weight: Path | None = None) -> Path:
    """把官方 repo 接进 `sys.path` 并补上两处替身。**幂等**。返回 repo 目录。

    - 替身 ①:`pytorch_lightning.utilities.distributed`(PL 2.x 挪走了它)
    - 替身 ②:`annotator.midas.api.ISL_PATHS` 指向本仓的 `weights/`
      —— 官方预期权重在 `<repo>/annotator/ckpts/`,而**往 pristine 克隆里写文件**是本仓禁止的
    - 替身 ③:见 `_ensure_offline()` 的头注(本机够不着 huggingface.co,而 transformers
      **会挂在网络重试上**,界面看起来就是"还在加载模型")
    """
    _ensure_offline()
    global _READY
    repo = repo_dir()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    if "pytorch_lightning.utilities.distributed" not in sys.modules:
        try:
            from pytorch_lightning.utilities.rank_zero import rank_zero_only
        except ImportError as e:  # pragma: no cover - 环境缺 PL
            raise ImportError(
                "需要 pytorch_lightning(`ldm.models.diffusion.ddpm` 顶层 import 它);"
                " 装:pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytorch_lightning"
            ) from e
        shim = types.ModuleType("pytorch_lightning.utilities.distributed")
        shim.rank_zero_only = rank_zero_only
        sys.modules["pytorch_lightning.utilities.distributed"] = shim

    if not _READY:
        from annotator.midas import api as midas_api

        w = midas_weight or (weights_dir() / MIDAS_WEIGHT)
        if not w.exists():
            raise FileNotFoundError(
                f"缺 MiDaS 权重 {w}\n"
                "  官方 demo 会自己去 huggingface.co 下(本机不通);本仓要求预置在 weights/ 下。"
            )
        # 官方那份叫 dpt_hybrid;**别改键名**,`MiDaSInference(model_type=...)` 按它查表
        midas_api.ISL_PATHS["dpt_hybrid"] = str(w)
        _READY = True
    return repo


def to_square_rgb(rgb: np.ndarray) -> np.ndarray:
    """中心方裁到 min(H,W)。**方图是官方 demo 的隐含前提**,不裁会在很深处才报形状错。"""
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"要 (H,W,3) 的 RGB,收到 {a.shape}")
    h, w = a.shape[:2]
    if h == w:
        return a
    s = min(h, w)
    return a[(h - s) // 2 : (h + s) // 2, (w - s) // 2 : (w + s) // 2]


# --------------------------------------------------------------------------- 条件估计器(官方)
def canny_edges(rgb: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """官方 `annotator.canny.CannyDetector` 的包装。返回 uint8 (H,W)。"""
    ensure_importable()
    from annotator.canny import CannyDetector

    return CannyDetector()(to_square_rgb(rgb), low, high)


def midas_depth(rgb: np.ndarray, detector=None, *, detect_resolution: int = 512) -> np.ndarray:
    """官方 depth 条件图:先方裁 → `resize_image(..., detect_resolution)` → MiDaS。

    ⚠️ **`resize_image` 这一步不能省**(头注第六处):官方 demo 是
    `apply_midas(resize_image(HWC3(img), 512))`,而 `MidasDetector` **自己不 resize**。
    直接喂原图会在 ViT 里报 `a (577) must match b (530)`。

    ⚠️ **detector 请由调用方传入并复用** —— 每次构造要重新加载 470 MB 权重。
    """
    ensure_importable()
    from annotator.util import HWC3, resize_image

    if detector is None:
        from annotator.midas import MidasDetector

        detector = MidasDetector()
    resized = resize_image(HWC3(to_square_rgb(rgb)), detect_resolution)
    depth_u8, _normal = detector(resized)
    return np.asarray(depth_u8, dtype=np.uint8)


# --------------------------------------------------------------------------- 模型加载
class CldmModel:
    """加载好的 ControlNet 句柄:`model` / `sampler` / `variant`。"""

    __slots__ = ("variant", "model", "sampler", "weight", "sha256", "device")

    def __init__(self, variant, model, sampler, weight, sha256, device):
        self.variant = variant
        self.model = model
        self.sampler = sampler
        self.weight = weight
        self.sha256 = sha256
        self.device = device


def load_cldm(variant: str = "canny", *, device: str = "cuda", verify: bool = True) -> CldmModel:
    """加载 `control_sd15_<variant>`。**先校验权重,再建模型**(校验失败不该花 100 s 建图)。"""
    if variant not in VARIANT_WEIGHTS:
        raise KeyError(f"未接的变体 {variant!r};已接 {sorted(VARIANT_WEIGHTS)}")
    repo = ensure_importable()
    weight = weights_dir() / VARIANT_WEIGHTS[variant]
    sha = verify_weight(weight) if verify else ""

    import torch  # noqa: PLC0415 — torch 只在这条路径上需要
    from cldm.ddim_hacked import DDIMSampler
    from cldm.model import create_model, load_state_dict

    model = create_model(str(repo / "models" / "cldm_v15.yaml"))
    # 适配 ③:`weights_only` 的作用域**只包住这一次读取**,不给全局留副作用
    orig_load = torch.load
    torch.load = functools.partial(orig_load, weights_only=False)
    try:
        sd = load_state_dict(str(weight), location=device)
    finally:
        torch.load = orig_load

    sd = remap_clip_keys(sd, model.state_dict())
    model.load_state_dict(sd)
    model = model.to(device).eval()
    return CldmModel(variant, model, DDIMSampler(model), weight, sha, device)


def remap_clip_keys(sd: dict, model_sd: dict) -> dict:
    """适配 ④:**只**去掉 CLIP 文本塔的 `text_model.` 前缀,并丢掉老版 buffer。

    ⚠️ **只动这一个前缀**,不是"把不匹配的键都猜一遍" —— 那样会把真正的缺键也掩盖掉。
    返回新 dict,不改入参。
    """
    out: dict = {}
    n_prefix = 0
    n_drop = 0
    for k, v in sd.items():
        if k.endswith(".embeddings.position_ids"):
            n_drop += 1
            continue
        alt = k.replace(".transformer.text_model.", ".transformer.")
        if k not in model_sd and alt in model_sd:
            out[alt] = v
            n_prefix += 1
        else:
            out[k] = v
    missing = [k for k in model_sd if k not in out]
    if missing:
        raise RuntimeError(
            f"重映射后仍有 {len(missing)} 个键缺失,前 5 个:{missing[:5]}\n"
            "  ⇒ **不要**把这里改成'宽松加载':缺键说明 transformers 的包装层又变了,"
            "  要按新布局再加一条**明确**的重映射规则。"
        )
    return out


def load_rgb(path: Path) -> np.ndarray:
    """读一张采集帧 → uint8 HxWx3 RGB。"""
    from PIL import Image

    a = np.array(Image.open(path).convert("RGB"))
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"{path} 不是三通道 RGB,shape={a.shape}")
    return a

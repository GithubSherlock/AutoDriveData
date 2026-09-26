#!/usr/bin/env bash
# 官方 MapTR/MapQR 对照栈环境引导(Plan.md §5.12 阶段 0)。
#
# 用途:在**数据盘**新建 py3.8 env,装官方老栈并编译官方自带 mmdet3d 与 GKT op,
# 供 A′(口径复算)/ B(同数据复线)使用。系统盘只有 8.2G,env 必须落数据盘。
#
# 与官方 install.md 的差异(均为本机实测约束,不是偏好):
#   - torch 1.9.1+cu111 用**阿里云 pytorch-wheels 镜像**(pypi 上只有 cu102 版;
#     conda/tuna 的 py3.8_cuda11.1 变体在 classic solver 下解算过慢,实测放弃)。
#     **不要改回 download.pytorch.org**:实测该直链在本机下到 1.1G/2.04G 后停滞
#     (20 分钟 0 字节),阿里云同文件 HTTP 206 稳定 ~4MB/s → 全程约 9 分钟
#   - mmcv-full 1.4.0 用官方**预编译 wheel**(cu111/torch1.9.0),免源码编译
#   - 编译用系统 CUDA 11.8(torch cu111 只校验 major 版本)+ gcc-9
#     (torch1.9 头文件与 GCC 11 不兼容,实测先用 gcc-9)
#
# 用法:bash autodrivedata/map/maptr_official/setup_maptr_official.sh [system|conda|pip|ckpt|build|verify|all]
# 幂等:已完成的步骤会跳过;日志落 outputs/maptr_official/logs/。
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${MAPTR_ENV:-/root/autodl-tmp/envs/maptr_official}"
LOGDIR="$PROJ/outputs/maptr_official/logs"
# 大件缓存与解包目录都放**数据盘**:系统盘只剩 5.4G,torch 单包 2.04G + pip 解包必爆。
WHEELDIR="$PROJ/outputs/maptr_official/wheels"
export TMPDIR="$PROJ/outputs/maptr_official/tmp"
export PIP_CACHE_DIR="$PROJ/outputs/maptr_official/pip-cache"  # 同因:/root/.cache 在系统盘
PY="$ENV/bin/python"
PIP="$ENV/bin/pip"
CONDA=/root/miniconda3/bin/conda
MIRROR=https://mirrors.aliyun.com/pytorch-wheels/cu111   # 见文件头:官方直链会停滞
TORCH_WHEEL=$MIRROR/torch-1.9.1%2Bcu111-cp38-cp38-linux_x86_64.whl        # 2.04 GB
TV_WHEEL=$MIRROR/torchvision-0.10.1%2Bcu111-cp38-cp38-linux_x86_64.whl   # 20.6 MB
MMCV_WHEEL=https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/mmcv_full-1.4.0-cp38-cp38-manylinux1_x86_64.whl
REPO_MAIN="$PROJ/hdMapGitHub/MapTR"   # main = MapTR v1
REPO_QR="$PROJ/hdMapGitHub/MapQR"     # MapTRv2 系(MapQR 自述与 MapTRv2 同设置)

export CUDA_HOME=/usr/local/cuda-11.8
export CC=gcc-9 CXX=g++-9
export FORCE_CUDA=1 MMCV_WITH_OPS=1
# **必须显式钉住**:宿主 conda 环境导出了 `TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0;…;12.1+PTX`,
# 而 torch 1.9.1 的 `_get_cuda_arch_flags` 只认到 8.6 → 编译期直接
# `ValueError: Unknown CUDA arch (8.9) or GPU not supported`。本机卡是 3080 Ti(sm_86),
# 取 8.6+PTX:出 sm_86 SASS + PTX 兜底(driver 可在更新架构上 JIT),且只编一个架构、快得多。
export TORCH_CUDA_ARCH_LIST=8.6+PTX
export CUDAARCHS=86-real

mkdir -p "$LOGDIR" "$WHEELDIR" "$TMPDIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# 断点续传下载(取代 `pip install <URL>`):pip 直装遇停滞只能整包重来,这里每个
# 网络中断点都保留已下字节,重跑本阶段即从断点续。停在 `--speed-limit` 以下
# 45 秒即判停滞并续传(实测 download.pytorch.org 会静默停在半途,不发 EOF)。
# 结果路径经全局 `_WHL` 回传(log 也写 stdout,不能用命令替换取返回值)。
fetch_wheel() {  # $1=url $2=期望字节数(0 = 跳过校验)
  # 逐行 local:`local a=1 b=$a` 里 b 看不到 a(参数整体先展开,且 $a 未定义时
  # 报错发生在命令替换的子 shell 里 → 只会静默拿到空串,不会中断脚本)
  local url="$1"
  local want="${2:-0}"
  local n=0 got f
  # URL 里的 %2B 要还原成 '+',否则 pip 按文件名解析版本号时会看到 "%" 而拒装
  f="$WHEELDIR/$(basename "${url%%\?*}" | sed 's/%2B/+/g')"
  if [ "$want" != 0 ] && [ -f "$f" ] && [ "$(stat -c%s "$f")" = "$want" ]; then
    log "  命中缓存:$(basename "$f")(完整)"
    _WHL="$f"
    return 0
  fi
  while :; do
    got="$(stat -c%s "$f" 2>/dev/null || echo 0)"
    log "  下载 $(basename "$f")(从 $got 字节续)"
    curl -L --fail --silent --show-error --connect-timeout 20 -C - \
      --speed-limit 20480 --speed-time 45 -o "$f" "$url" && break
    n=$((n + 1))
    if [ "$n" -ge 15 ]; then log "  ✗ 下载失败(已试 $n 次):$f"; return 1; fi
    sleep 3
  done
  if [ "$want" != 0 ] && [ "$(stat -c%s "$f")" != "$want" ]; then
    log "  ✗ 大小不符:$(stat -c%s "$f") != $want"
    return 1
  fi
  log "  ✓ $(basename "$f") $(du -h "$f" | cut -f1)"
  _WHL="$f"
}

# av2(Argoverse 2 SDK)垫片 —— 只补"能被 import"这一层。
#
# 起因:官方**两个**仓库的 `datasets/__init__.py` 都无条件
# `from .av2_map_dataset import CustomAV2LocalMapDataset`,而那两个模块顶层 import 了
# av2 SDK 的 6 个符号 → 不补则整条 `projects.mmdet3d_plugin` 导入链断裂(实测:v1 补完
# numba/IPython 后卡在这里;v2/MapQR 走 train.py 的 plugin_dir 加载,同样要过这关)。
#
# 为什么不装真包:`pip install --dry-run av2` 实测解析器持续回溯 >10min 无输出(pip 24.2),
# 与老 torch/numpy 的 pin 冲突面大,还可能带进第二份 cv2(与 opencv-python-headless 抢
# `cv2` 名)。而本项目**永远不用 AV2 数据**(数据源是 CARLA),为一行 import 付这个代价不值。
#
# 边界写清楚:占位类**实例化即抛 NotImplementedError**,不静默返回假对象 ——
# 将来真要跑 AV2 必须 `pip install av2` 并删掉 `site-packages/av2`(脚本幂等,不会重建)。
install_av2_stub() {
  local sp
  sp="$("$PY" -c 'import site;print(site.getsitepackages()[0])')"
  if [ -d "$sp/av2" ]; then
    log "  av2 垫片已存在:$sp/av2"
    return 0
  fi
  "$PY" - "$sp" <<'PYEOF'
import pathlib
import sys

HEAD = "# 由 AutoDriveData 的 autodrivedata/map/maptr_official/setup_maptr_official.sh 生成的 av2 垫片,不是真 SDK。\n"
GUARD = '''"""AV2 垫片公共占位:真实调用必须炸,不静默返回假结果。"""


def unused(name):
    class _Unused:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                f"{name} 是 av2 垫片占位(见 autodrivedata/map/maptr_official/setup_maptr_official.sh 的 install_av2_stub)。"
                " 本项目数据是 CARLA、不用 AV2;要真跑 AV2 得 `pip install av2` "
                "并删除 site-packages/av2。"
            )

    _Unused.__name__ = name
    return _Unused
'''
files = {
    "__init__.py": "",
    "_unused.py": GUARD,
    "datasets/__init__.py": "",
    "datasets/sensor/__init__.py": "",
    "datasets/sensor/av2_sensor_dataloader.py": (
        "from av2._unused import unused\n\nAV2SensorDataLoader = unused('AV2SensorDataLoader')\n"
    ),
    "map/__init__.py": "",
    "map/lane_segment.py": (
        "from av2._unused import unused\n\n"
        "LaneMarkType = unused('LaneMarkType')\nLaneSegment = unused('LaneSegment')\n"
    ),
    "map/map_api.py": ("from av2._unused import unused\n\nArgoverseStaticMap = unused('ArgoverseStaticMap')\n"),
    "geometry/__init__.py": "",
    "geometry/se3.py": "from av2._unused import unused\n\nSE3 = unused('SE3')\n",
    # 只被官方注释掉的那行 `# interp_utils.interp_arc(...)` 引用 → 空模块(真调用是
    # AttributeError,同样是响的)
    "geometry/interpolate.py": "",
}
root = pathlib.Path(sys.argv[1]) / "av2"
for rel, body in files.items():
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(HEAD + body, encoding="utf-8")
print(f"  写入 {len(files)} 个文件到 {root}")
PYEOF
}

stage_system() {
  log "STAGE system: 编译链 gcc-9/g++-9(CUDA 11.8 + torch1.9 头文件兼容)"
  if command -v gcc-9 >/dev/null && command -v g++-9 >/dev/null; then
    log "  已装:$(gcc-9 --version | head -1)"
  else
    apt-get install -y gcc-9 g++-9
  fi
}

stage_conda() {
  log "STAGE conda: $ENV (python3.8)"
  if [ ! -x "$PY" ]; then
    "$CONDA" create -y -p "$ENV" python=3.8
  else
    log "  已存在:$("$PY" -V)"
  fi
}

stage_pip() {
  log "STAGE pip: torch 1.9.1+cu111 / torchvision 0.10.1(阿里云镜像,自带 cuDNN)"
  if "$PY" -c "import torch" 2>/dev/null; then
    log "  已装:torch $("$PY" -c 'import torch;print(torch.__version__)')"
  else
    fetch_wheel "$TORCH_WHEEL" 2041339783   # 落 $_WHL
    t="$_WHL"
    fetch_wheel "$TV_WHEEL" 20580631
    tv="$_WHL"
    "$PIP" install "$t" "$tv"
  fi
  log "STAGE pip: mmcv-full 1.4.0(预编译 wheel,cu111/torch1.9.0)"
  if ! "$PY" -c "import mmcv._ext" 2>/dev/null; then
    fetch_wheel "$MMCV_WHEEL" 0 && "$PIP" install "$_WHL"
  else
    log "  已装:mmcv $("$PY" -c 'import mmcv;print(mmcv.__version__)') + _ext"
  fi
  log "STAGE pip: mmdet 2.14.0 / mmsegmentation 0.14.1 / timm 及 mmdet3d 运行时依赖"
  "$PIP" install "numpy==1.23.5" "mmdet==2.14.0" "mmsegmentation==0.14.1" timm \
    "shapely==1.8.5.post1" scipy opencv-python-headless nuscenes-devkit pycocotools terminaltables
  # yapf 必须 <0.40:mmcv 1.4.0 的 `Config.pretty_text` 调
  # `yapf.FormatCode(text, style_config=..., verify=True)`,而 `verify` 参数在 yapf 0.40
  # 被删 → 默认装的 0.43 会在 `tools/train.py` 开头 `cfg.dump()` 处
  # `TypeError: FormatCode() got an unexpected keyword argument 'verify'`,训练根本起不来。
  "$PIP" install "yapf==0.32.0"
  # mmdet3d 的 requirements/runtime.txt **不能整份照装**:里面两个 pin 是 mmdet3d 0.x 时代的
  # 遗留(注释自己写着 "we may unlock the verion of numba in the future"):
  #   numba==0.48.0 + numpy<1.20.0 → 会把 numpy 降到 1.19.5,而本栈的 opencv-python-headless 5.x
  #   / shapely 1.8.5 都是按新 ABI 出的轮子 → 静默降级后 cv2/shapely 导入即炸。
  # 逐项列出:lyft_dataset_sdk 是**导入期硬依赖**(`mmdet3d/datasets/__init__.py` → lyft_dataset.py
  # 顶层 import),numba 用与 numpy 1.23.5 相容的 0.56.4(0.48 上限只到 numpy 1.18)。
  # ipython:MapTR v1 的 `datasets/nuscnes_eval.py:53` 顶层留着 `from IPython import embed`
  # (调试残留),不装则在导入 `projects.mmdet3d_plugin.datasets` 时 ModuleNotFoundError。
  # 只影响 v1 那条线(官方仓库一个字符都不改,靠装包补齐)。
  "$PIP" install lyft_dataset_sdk "networkx>=2.2,<2.3" "numba==0.56.4" plyfile scikit-image tensorboard \
    "trimesh>=2.35.39,<2.35.40" ipython
  # numba.errors 垫片(mmdet3d 0.17.2 的**最后一处** legacy pin):
  # `datasets/pipelines/data_augment_utils.py:5` 写死 `from numba.errors import
  # NumbaPerformanceWarning`,而 `numba.errors` 在 numba 0.49 就并进了 `numba.core.errors`
  # → numba 0.56.4 下导入 mmdet3d.datasets 直接 `ModuleNotFoundError: No module named
  # 'numba.errors'`(实测:卡在 bridge 导入的第三层)。其余 numba 用法(jit/njit/prange/cuda)
  # 在 0.56 都还在,只有这一处。
  # 解法是**给 env 补垫片**,不动官方仓库一行(hdMapGitHub 保持 pristine):
  # 该模块把 numba.core.errors 的名字原样转出,mmdet3d 那行 import 照常成立。
  local numba_dir shim
  numba_dir="$("$PY" -c 'import numba,os;print(os.path.dirname(numba.__file__))')"
  shim="$numba_dir/errors.py"
  if [ ! -f "$shim" ]; then
    cat > "$shim" <<'PYSHIM'
"""Compatibility shim for `numba.errors`(已并入 `numba.core.errors`)。

由 AutoDriveData 的 `autodrivedata/map/maptr_official/setup_maptr_official.sh` 生成,不是 numba 自带文件。
原因:官方 mmdet3d 0.17.2 的 `datasets/pipelines/data_augment_utils.py` 仍按
numba<0.49 的包布局 `from numba.errors import NumbaPerformanceWarning`,而
mmdet3d 钉的 `numba==0.48.0` 只能配 numpy<1.18(会连带降级 numpy、破坏
opencv/shapely 的 ABI)。本栈保留 numba 0.56.4 + numpy 1.23.5,故补此垫片。
重装 numba 会覆盖掉本文件 → 重跑 `bash autodrivedata/map/maptr_official/setup_maptr_official.sh pip` 即可恢复。
"""

import numba.core.errors as _core_errors

globals().update({k: v for k, v in vars(_core_errors).items() if not k.startswith("_")})
PYSHIM
    log "  补 numba.errors 垫片:$shim"
  else
    log "  numba.errors 垫片已存在"
  fi
  install_av2_stub
}

# 主干预训练权重(ImageNet ResNet-50)搬进项目。三份 config 的 `model.pretrained` 覆盖
# 说明见各文件;这里只负责把文件放到那个绝对路径上。来源按序回退:
#   ① 项目内(已就位)② 本机 torch hub 缓存 ③ download.pytorch.org(实测该域名在本机
#   会静默停滞,故只当兜底,不指望它)
stage_ckpt() {
  local dst="$PROJ/outputs/maptr_official/ckpts"
  local name=resnet50-0676ba61.pth
  local size=102530333
  mkdir -p "$dst"
  if [ -f "$dst/$name" ]; then
    log "  已就位:$dst/$name"
    return 0
  fi
  local src
  for src in "${TORCH_HUB_CKPT:-$HOME/.cache/torch/hub/checkpoints}/$name" "$WHEELDIR/$name"; do
    if [ -f "$src" ]; then
      cp "$src" "$dst/$name"
      log "  已从 $src 复制"
      return 0
    fi
  done
  log "  本地无副本 → 下载 https://download.pytorch.org/models/$name(约 100MB)"
  fetch_wheel "https://download.pytorch.org/models/$name" "$size" || return 1
  cp "$_WHL" "$dst/$name"
  log "  已下载并复制到 $dst/"
}

stage_build() {
  # mmdet3d 只装一次(两仓库同为 mmdet3d fork);GKT op 各仓库自带一份、
  # setup.py 产出的模块同名 → 用 MapQR(MapTRv2 系)那份(实测两仓库 op 目录逐字节一致,
  # `diff -r` 可复核),MapTR v1 侧直接复用。
  log "STAGE build: mmdet3d develop(CUDA_HOME=$CUDA_HOME, CC=$CC)"
  if "$PY" -c "import mmdet3d" 2>/dev/null; then
    log "  已装:mmdet3d $("$PY" -c 'import mmdet3d;print(mmdet3d.__version__)')"
  else
    # **必须 --no-deps**:`setup.py develop` 走 setuptools 老式 easy_install 解析,
    # 它**看不见 pip 的 pin**,会照 mmdet3d 0.17.2 的 install_requires 自行装
    # `numpy<1.20.0` + `numba==0.48.0`(实测:把 numpy 1.23.5 旁路成
    # site-packages/numpy-1.19.5-py3.8-linux_x86_64.egg 并写进 easy-install.pth,
    # 随后被 nuscenes-devkit 的 `numpy>=1.22` 撞停 → 整个 build 阶段 EXIT=1)。
    # 运行期依赖已在 stage_pip 里逐项装好,这里只要编译产物。
    (cd "$REPO_QR/mmdetection3d" && "$PY" setup.py develop --no-deps)
  fi
  # 模块名是 setup.py 里 `extension(...)` 的第一参数 = **GeometricKernelAttention**;
  # `geometric_kernel_attn_cuda` 只是它内部导出的 C++ 函数名 —— 拿后者当 import 名
  # 会永远探不到已装好的 op(反复重编)。
  local op="$REPO_QR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn"
  log "STAGE build: GKT op($op)"
  if "$PY" -c "import GeometricKernelAttention" 2>/dev/null; then
    log "  已装:GeometricKernelAttention"
  else
    # op 的 setup.py 没有 install_requires(只有未传入 setup() 的 `requirements` 列表)
    # → `build install` 不会触发依赖解析,可以放心用官方原命令。
    (cd "$op" && "$PY" setup.py build install)
  fi
}

stage_verify() {
  log "STAGE verify: 环境自证 → outputs/maptr_official/env_check.json"
  # 判据是 **GKT op 真跑一次前向+反向**,不是"import 成功":编译缺陷 / 架构不匹配
  # 都在 kernel 启动那一刻才显形(`import` 只证明 .so 能被加载)。op 自带 test.py 是
  # 空文件,故这里自建最小用例;Func 按**文件路径**加载 —— `geometric_kernel_attn_func.py`
  # 只 import torch 与 op,沿包路径导入反而会拉起整个模型栈(projects/__init__.py)。
  # 形状口径照 op 自己的约定:value(bs, spatial, heads, channels)、
  # sampling_loc(bs, q, heads, levels, npts, 2) —— C++ 侧 `num_heads = value.size(2)`
  # 而 sampling_loc 的第 3 维被当作 num_heads 用,所以**参考点个数 nz 必须 == num_heads**。
  (cd "$PROJ" && "$PY" - "$REPO_QR" <<'EOF'
import importlib.util, json, platform, pathlib, sys
import torch, mmcv, mmdet, mmseg, mmdet3d
import GeometricKernelAttention as GKA

func_py = pathlib.Path(sys.argv[1]) / (
    "projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn/function/geometric_kernel_attn_func.py"
)
spec = importlib.util.spec_from_file_location("gkt_func_selfcheck", func_py)
gkt_func = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gkt_func)

torch.manual_seed(0)
dev = "cuda"
bs, heads, levels, npts, ch, h, w, nq = 2, 8, 1, 5, 32, 16, 32, 12
value = torch.randn(bs, levels * h * w, heads, ch, device=dev)
shapes = torch.tensor([[h, w]], dtype=torch.long, device=dev)
start = torch.tensor([0], dtype=torch.long, device=dev)
# +1 起步、留 2 格余量:kernel 内部对采样点有 ±1 取整,贴边会越界
loc = torch.rand(bs, nq, heads, levels, npts, 2, device=dev) * torch.tensor(
    [w - 2, h - 2], device=dev
) + 1
# **采样点必须是 int64**:kernel 侧 `sampling_loc.data<int64_t>()` 是按 int64 硬读的
# (官方模块里也确实是 `.round().long()`,见 geometry_kernel_attention.py 的
# `sampling_locations = (...).round().long()`)。传 float 会在 kernel 启动处直接
# `RuntimeError: expected scalar type Long but found Float` —— 实测踩过。
loc = loc.long()
weight = torch.rand(bs, nq, heads, levels, npts, device=dev)
weight = weight / weight.sum(-1, keepdim=True)
value.requires_grad_(True)
weight.requires_grad_(True)

out = gkt_func.GeometricKernelAttentionFunc.apply(
    value, shapes, start, loc.contiguous(), weight, 64
)
# 显式连续梯度:`out.sum().backward()` 给的是**广播**梯度,反向入口有
# `AT_ASSERTM(grad_output.is_contiguous())`,非连续会直接 INTERNAL ASSERT FAILED
# (探针产物,不是环境问题 —— 真实训练里下游给的就是连续梯度)。
out.backward(torch.ones_like(out))
torch.cuda.synchronize()

rec = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    "mmcv": mmcv.__version__,
    "mmcv_ext": True,
    "mmdet": mmdet.__version__,
    "mmseg": mmseg.__version__,
    "mmdet3d": mmdet3d.__version__,
    "gkt_op": GKA.__file__,
    "gkt_forward_shape": list(out.shape),
    "gkt_forward_finite": bool(torch.isfinite(out).all()),
    "gkt_backward_grad_value_finite": bool(torch.isfinite(value.grad).all()),
    "gkt_backward_grad_weight_finite": bool(torch.isfinite(weight.grad).all()),
    "gcc": __import__("os").environ.get("CC"),
    "torch_cuda_arch_list": __import__("os").environ.get("TORCH_CUDA_ARCH_LIST"),
}
assert rec["cuda_available"] and rec["gkt_forward_finite"], rec
assert rec["gkt_backward_grad_value_finite"] and rec["gkt_backward_grad_weight_finite"], rec
out_path = pathlib.Path("outputs/maptr_official/env_check.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(rec, ensure_ascii=False, indent=2))
EOF
  )
}

for s in "${@:-all}"; do
  case "$s" in
    all)    stage_system; stage_conda; stage_pip; stage_ckpt; stage_build; stage_verify ;;
    system|conda|pip|ckpt|build|verify) "stage_$s" ;;
    *) echo "未知阶段:$s(可选 system|conda|pip|ckpt|build|verify|all)" >&2; exit 2 ;;
  esac
done
log "完成"

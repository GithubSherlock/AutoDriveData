"""P-G 3DGS mini 训练:CARLA 360° 环绕采集 → Gaussian Splatting → outputs/3dgs/。

输入:`outputs/3dgs/capture/`(autodrivedata/sim/collect_3dgs.py 产物:多俯仰采集,每 pitch 一圈)。
  - 采集现按 pitch 分目录 images/p{p}/{i}.png + poses_{p}.json(见 pitches.json 枚举);
    本脚本跨 pitch 平铺(全局序号 = p_idx × frames_per_pitch + i),位姿仍用 CARLA 真值
    (定位降级:pycolmap SfM 对齐误差 ~5.7m,见 outputs/3dgs/sfm_eval.json)。
- 初始化:真值深度(sensor.camera.depth)稠密网格反投影,降采样至 `_N_INIT_TOTAL` 点。
- 训练:gsplat 光栅化 + Adam,后激活 RGB 颜色(sh_degree=None 口径)。
- 留出帧(`--val-frames`,不参与训练)算 PSNR(如实报告)。
  ⚠️ **默认值曾是 `0`,而第 0 帧是采集瞬态**(图像与位姿不符)⇒ 全部归档的 `psnr_val`
  都算在一张坏帧上,**那个数一直无意义**。本轮起曲线用 `--val-frames 45`。见
  [docs/edit-3dgs-plan.md](../../docs/edit-3dgs-plan.md) §1.5。

落盘:
  outputs/3dgs/means{tag}.npy / scales / col / opac      (训练参数)
  outputs/3dgs/gaussians{tag}.ply                         标准 3DGS PLY
  outputs/3dgs/train_result{tag}.json                     PSNR + 帧数统计
  outputs/3dgs/render_compare{tag}.png                    GT|渲染|差值(帧0 与帧45)

用法(必须先建好 gsplat 扩展,见 CLAUDE.md 环境注意):
  CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH \
    python -m autodrivedata.gs.train_3dgs_mini [--iters 1500] [--tag ep1500] [--scale 0.05]

## ★ 环境两个坑,2026-10-03 实测(报错都指向别处)

**① `CUDA_HOME` 必须显式给,而且 PATH 上的 `nvcc` 必须是 11.8 那一个。**
本机 `PATH` 上的 `nvcc` 是 conda 的 **13.0**(`/root/autodl-tmp/miniconda3/bin/nvcc`),
而**盘上只有 CUDA 11.8 的 `libcudart`** ⇒ 用 13.0 编出来的扩展在 import 期抛
`ImportError: libcudart.so.13: cannot open shared object file`(`.so` 已生成,看着像环境坏了)。
更早一步的报错更误导 —— 不设 `CUDA_HOME` 时是
`fatal error: crt/host_defines.h: No such file or directory`,**编译错误**,但真因是
`/root/miniconda3/include/cuda_runtime_api.h` 是个**指向 11.8 的软链**,
配对的 `crt/` 目录却不在 include 路径上。

**② `TORCH_CUDA_ARCH_LIST` 的 shell 预置值不能盲信。**
本机预置了 `7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0;12.1+PTX`,含 torch 2.6 **不认识**的 `10.3`,
而报错点是 gsplat 内部的 `ValueError: Unknown CUDA arch (10.3)` —— **看不出是环境变量的事**。
故本模块的架构处理**不再只 `setdefault`**:预置值先过一遍 `_get_cuda_arch_flags()`,
不可用就换成实卡架构并在启动时打印 `[arch]`。**不要照抄某个数字** ——
这台机器换过卡(3080 Ti sm_86 → 4080 SUPER sm_89 → 3090 sm_86),写死的 `8.9` 在 sm_86 上
会让内核起不来,报的却是 `Failed to set maximum shared memory size ... try lowering tile_size`
(2026-09-28 实测:它提示的方向是误导,真因是**架构不符**)。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from autodrivedata.calib.core import CameraIntrinsics
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path


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

# ★ 必须在 `import gsplat` **之前**设好 —— 它决定 CUDA 内核的编译目标架构。
if torch.cuda.is_available():
    _cc = torch.cuda.get_device_capability()
    _detected = f"{_cc[0]}.{_cc[1]}"
    _preset = os.environ.get("TORCH_CUDA_ARCH_LIST")
    # ⚠️ **`setdefault` 不够**(2026-10-03 实测):本机 shell 预置了
    #   `7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0;12.1+PTX`,含 torch 2.6 不认识的 `10.3`
    #   ⇒ gsplat 的 JIT 编译抛 `Unknown CUDA arch (10.3)`,**而报错点在 gsplat 内部**,
    #   看不出是环境变量预置导致的(旧注释还以为"外部显式指定 = 有意为之")。
    #   ⇒ 预置值**先验一遍**,不可用就换成实卡架构并留下 `ARCH_NOTE`。
    if _preset and not _arch_list_usable(_preset):
        os.environ["TORCH_CUDA_ARCH_LIST"] = _detected
        ARCH_NOTE = (
            f"预置的 TORCH_CUDA_ARCH_LIST={_preset!r} 本机 torch 不接受 ⇒ "
            f"改用本卡架构 {_detected}(shell/direnv 预置的全量列表常含未来架构)"
        )
    else:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _detected)
        ARCH_NOTE = f"TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}"

import gsplat  # noqa: E402  —— 必须排在上面那段设置之后

_DOWNSAMPLE = 2  # 1242x375 → 621x187
#: 阶段 A 诊断用的**深度分箱**(m)。天空单列一桶 —— CARLA 远平面实测 **999.9997**。
#: **全部 270 帧实测的占比**:`<5m 38.5% | 5–20m 60.4% | 20–50m 0.6% | >50m 0.5% | 天空 0.3%`
#: ⇒ 这是个 **~20 m 的「近场气泡」**(99% 的像素在 20 m 内),所以
#: **"排除天空"几乎是个空操作**(天空 0.3%)—— 分箱的价值在于看**误差落在哪一段**,
#: 而不是靠掐掉一小撮像素去抬全场那个数。
#: ⚠️ **别拿单帧算这个分布**:`p0` 的第 0 帧量出来是「>50 m 占 22.6%、天空 1.1%」,
#: 而其余 89 帧 **全部 0.0%** —— 那一帧是采集瞬态(图像与位姿不符,见
#: [docs/edit-3dgs-plan.md](../../docs/edit-3dgs-plan.md) §1.5)。
#: 见同文档 §1.4 / §4 阶段 A。
PSNR_DEPTH_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 5.0),
    (5.0, 20.0),
    (20.0, 50.0),
    (50.0, 999.0),
)
SKY_DEPTH = 999.0
_N_INIT_PER_FRAME = 800  # 每帧深度采样点数(270 帧 → ~216k 候选,再降采样)
_N_INIT_TOTAL = 120000  # 多俯仰扩点:3 倍视角需 ~3 倍点预算才不稀释原环覆盖
_N_VIEWS_PER_STEP = 12
_DEPTH_LOWER = 1.0  # 深度初始化的下限(m,滤掉近身灰尘);上限 = 采集范围


def _load_poses_and_cams(capture: Path, src_h: int = 375, src_w: int = 1242, downsample: int = _DOWNSAMPLE):
    """跨 pitch 平铺位姿(全局序 = p_idx × n_per_pitch + i)。

    pitches.json 是本批俯仰序列;每俯仰读 poses_{p}.json。返回平铺后的 poses,
    viewmat/内参序与 poses 对齐(后续 _load_images/_depth_init_points 同此序)。
    多俯仰时每 pose 带 (pitch, i) 以定位图像文件;pitch 存原始 float(供 Rotation 用)。
    """
    pitches = json.loads((capture / "pitches.json").read_text(encoding="utf-8"))
    # fov→fx 与主点走全仓唯一落点(`CameraIntrinsics`),不再就地重写公式。
    # 历史写法 `1242/2/(2·tan45°)` 数值上恰好等于 621/2(tan45°=1 把多写的那个 2 抵掉),
    # 是巧合而非推导 —— 换成别的 fov 就会错。
    #
    # **两套空间的主点必须各自推对**(实测裁决见 autodrivedata/calib/probe_calib.py A3/A4:
    # CARLA 渲染光栅是 **corner** 约定 —— 索引 i 的连续坐标就是 i,`cx = (w−1)/2 = 620.5`):
    # - 深度图/图像是 CARLA 光栅 → 下采样索引 j ↔ 原图索引 D·j ↔ 原图连续坐标 D·j;
    # - `ks` 喂给 gsplat(torch 原生光栅器,与 `grid_sample(align_corners=False)` 同族,
    #   **center** 约定:像素 j 覆盖 [j, j+1),中心 j+0.5)。
    # 令 gsplat 把原图索引 D·j 的点画到像素 j:`fx_g·(D·j−cx_orig)/fx_orig + cx_g = j + 0.5`
    # ⇒ `fx_g = fx_orig/D`、`cx_g = cx_orig/D + 0.5`。
    #
    # ⚠️ **通式的 `+0.5` 不是 `+1/D`**(2026-10-03 修)。旧写法 `(cx_orig+1)/D` **只在 D=2 上
    # 恰好相等**(`(cx+1)/2 = cx/2+0.5`,那个 `1/D` 与 corner→center 的 `0.5` 撞上了),
    # D=1 差 **0.5 px**、D=4 差 0.25 px —— 又是"数值上恰好成立"那类坑,只是高一层
    # (同本函数上面记的 `1242/2/(2·tan45°)`)。**改 `--downsample` 之前先看这条。**
    _k = CameraIntrinsics(width=src_w, height=src_h, fov_h_deg=90.0)
    H, W = src_h // downsample, src_w // downsample
    f = _k.fx / downsample
    cx, cy = _k.cx / downsample + 0.5, _k.cy / downsample + 0.5
    poses, viewmats, ks = [], [], []
    for p in pitches:
        pp = float(p)
        for pose in json.loads((capture / f"poses_{int(pp)}.json").read_text(encoding="utf-8")):
            pose = {**pose, "pitch": pp}  # 归一化 float,pitch 用于定位图像 + 建 viewmat
            poses.append(pose)
            yaw = math.radians(pose["yaw"])
            ry = torch.tensor(
                [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]],
                dtype=torch.float32,
            )
            rx = torch.tensor(
                [
                    [1, 0, 0],
                    [0, math.cos(math.radians(pp)), -math.sin(math.radians(pp))],
                    [0, math.sin(math.radians(pp)), math.cos(math.radians(pp))],
                ],
                dtype=torch.float32,
            )
            rw = ry @ rx  # cam→world 旋转
            tw = torch.tensor([pose["x"], pose["y"], pose["z"]], dtype=torch.float32)
            v = torch.eye(4, dtype=torch.float32)
            v[:3, :3] = rw.T
            v[:3, 3] = -rw.T @ tw  # world→cam
            viewmats.append(v)
            ks.append(torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=torch.float32))
    return poses, f, H, W, cx, cy, torch.stack(viewmats), torch.stack(ks)


def _frame_path(capture: Path, pose: dict, sub: str) -> Path:
    """位姿 → 采集文件路径(images/p{p}/{i}.png / depth/p{p}/{i}.npy)。"""
    name = f"{int(pose['i']):05d}"
    ext = "png" if sub == "images" else "npy"
    return capture / sub / f"p{int(pose['pitch'])}" / f"{name}.{ext}"


def _mse_to_psnr(mse: float | torch.Tensor) -> float:
    """MSE → PSNR(dB)。**训练循环与评估共用这一份** —— 两处各写一份必然漂。

    收 `Tensor` 是为了让调用点不必先 `.item()`(那会多一次同步);
    内部统一转 `float`,故标量张量与 Python float 走同一条路。
    """
    return float(-10 * math.log10(max(float(mse), 1e-8)))


def psnr_by_depth(
    se_per_pixel: torch.Tensor,
    depth: torch.Tensor,
    *,
    sky: float = SKY_DEPTH,
    bins: tuple[tuple[float, float], ...] = PSNR_DEPTH_BINS,
) -> dict[str, dict[str, float]]:
    """逐像素平方误差 × 真值深度 → **按深度分箱的 PSNR**。**纯函数,可单测。**

    `se_per_pixel` 与 `depth` 同形 `(N, H, W)`(前者是每像素的均方误差,已在 RGB 上求过均值)。
    返回 `{桶名: {"psnr", "share", "n_px"}}`,外加 `all` / `no_sky` 两桶。

    ## 为什么分箱比"排除天空"有信息量

    实测(`depth/p0/00000.npy`):天空(远平面 999.9997)只占 **1.1%**,
    而 >50 m 的像素占 **22.6%** —— 全场 PSNR 的误差大头在**远景**,
    掐掉那 1.1% 几乎不动这个数。**分箱才看得出"误差堆在哪一段深度上"**,
    而它直接决定编辑实验够不够用:物体移除/插入发生在**近场**。

    ⚠️ **桶内没有像素时给 `nan` 而不是 0** —— 与 `sem_eval` 的"没样本是 nan"同一条纪律:
    写成 0 会让"这一段没测到"被读成"这一段错得离谱"。
    """
    if se_per_pixel.shape != depth.shape:
        raise ValueError(f"形状必须相同,收到 {tuple(se_per_pixel.shape)} vs {tuple(depth.shape)}")
    total = se_per_pixel.numel()
    out: dict[str, dict[str, float]] = {}
    edges = [*bins, (sky, float("inf"))]
    names = [f"{int(lo)}-{int(hi)}m" for lo, hi in bins] + ["sky"]
    for name, (lo, hi) in zip(names, edges, strict=True):
        m = (depth >= lo) & (depth < hi)
        n = int(m.sum())
        out[name] = {
            "psnr": _mse_to_psnr(float(se_per_pixel[m].mean())) if n else float("nan"),
            "share": n / total if total else float("nan"),
            "n_px": float(n),
        }
    out["all"] = {"psnr": _mse_to_psnr(float(se_per_pixel.mean())), "share": 1.0, "n_px": float(total)}
    nosky = se_per_pixel[depth < sky]
    out["no_sky"] = {
        "psnr": _mse_to_psnr(float(nosky.mean())) if nosky.numel() else float("nan"),
        "share": (nosky.numel() / total) if total else float("nan"),
        "n_px": float(nosky.numel()),
    }
    return out


def _load_depths(capture: Path, poses: list, H: int, W: int) -> torch.Tensor:
    """与 `_load_images` **同序同尺寸**的深度张量 `(N,H,W)`(米)。

    分箱统计要逐像素对齐,顺序错一帧就会把两张图的误差拼在一起 —— 而**结果看着只是"分箱不好看"**,
    不会报错。故与 `_depth_init_points` 用**同一个** `_frame_path` 与同一个 `_DOWNSAMPLE`;
    那里用 `mode="nearest"` 是有意的:深度是测距,双线性会在物体边缘插出**不存在的深度**。
    """
    ds = []
    for pose in poses:
        d = torch.from_numpy(np.load(_frame_path(capture, pose, "depth")).astype(np.float32))
        ds.append(d)
    return F.interpolate(torch.stack(ds)[:, None], size=(H, W), mode="nearest")[:, 0]


def _load_images(capture: Path, poses: list, H: int, W: int) -> torch.Tensor:
    imgs = []
    for pose in poses:
        im = (
            torch.from_numpy(np.asarray(Image.open(_frame_path(capture, pose, "images")).convert("RGB")))
            .permute(2, 0, 1)
            .float()
            .contiguous()
            / 255.0
        )
        imgs.append(F.interpolate(im[None], size=(H, W), mode="bilinear", align_corners=False)[0])
    return torch.stack(imgs).permute(0, 2, 3, 1).contiguous()  # (N,H,W,3)


def _depth_init_points(
    capture: Path,
    poses: list,
    viewmats: torch.Tensor,
    H: int,
    W: int,
    f: float,
    cx: float,
    cy: float,
) -> torch.Tensor:
    """深度图 → 相机系点(与 `ks` **同一套连续坐标口径**)。

    `px/py` 是 `torch.nonzero` 给出的**光栅索引**;先 `+0.5` 换成连续坐标再减主点 ——
    这是把"索引"与"连续坐标"两套口径接起来的那一步,漏掉就是恒定半像素外推
    (30 m 处约 5 cm 横向偏差,随深度线性放大)。`cx/cy` 由 `_load_poses_and_cams` 按
    "CARLA corner 光栅 → 下采样 → gsplat center 光栅"推出,不在本函数里再推一遍。
    """
    depths = []
    for pose in poses:
        d = torch.from_numpy(np.load(_frame_path(capture, pose, "depth")).astype(np.float32))
        depths.append(d)
    depths = F.interpolate(torch.stack(depths)[:, None], size=(H, W), mode="nearest")[:, 0]  # (N,H,W)
    pts = []
    for i, d in enumerate(depths):
        valid = d > _DEPTH_LOWER
        if not valid.any():
            continue
        idx = torch.nonzero(valid)
        perm = torch.randperm(idx.shape[0])[:_N_INIT_PER_FRAME]
        px, py = idx[perm, 1].float(), idx[perm, 0].float()
        zv = d[py.long(), px.long()]
        cam_pts = torch.stack(
            [(px + 0.5 - cx) * zv / f, (py + 0.5 - cy) * zv / f, zv],
            1,
        ).float()
        v = viewmats[i].cpu()
        rw, tw = v[:3, :3].T, -v[:3, :3].T @ v[:3, 3]
        pts.append(cam_pts @ rw.T + tw)
    out = torch.cat(pts)
    out = out[torch.isfinite(out).all(1)]
    return out[torch.randperm(out.shape[0])[:_N_INIT_TOTAL]]


def _standard_ply(pos, col, opac, scales):
    """导出标准 3DGS PLY(二进制 little-endian,SH0 颜色口径)。"""
    n = pos.shape[0]
    fdc = (col.astype(np.float64) - 0.5) / 0.2820947917
    op_logit = np.log(np.clip(opac / (1.0 - opac), 1e-7, 1.0 - 1e-7))
    rot = np.zeros((n, 4))
    rot[:, 0] = 1.0
    out = np.column_stack([pos, np.zeros((n, 3)), fdc, op_logit, scales, rot])
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float f_rest_0\nproperty float f_rest_1\nproperty float f_rest_2\n"
        "property float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        "end_header\n"
    )
    return header.encode() + out.astype("<f4").tobytes()


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--capture", default="outputs/3dgs/capture")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--tag", default="")
    ap.add_argument("--val-frames", type=str, default="0", help="留出帧(不参与训练,逗号分隔,全局序)")
    ap.add_argument("--scale", type=float, default=0.05, help="高斯初始尺度(米,默认 0.05)")
    ap.add_argument(
        "--downsample",
        type=int,
        default=_DOWNSAMPLE,
        help="图像/深度的下采样倍率(默认 2 ⇒ 621×187)。**1 = 原生 1242×375**。"
        "⚠️ 内参那一带的通式 2026-10-03 才修对(旧写法只在 D=2 上碰巧成立),见 `_load_poses_and_cams`",
    )
    ap.add_argument(
        "--densify",
        action="store_true",
        help="开**自适应密度控制**(gsplat `DefaultStrategy`:clone/split/prune/reset)。"
        "**默认关,关时与归档逐位一致**;开时尺度改存 log(策略用 `exp`)。"
        "标准 3DGS 的核心组件,本实现原先没有 —— 见 docs/edit-3dgs-plan.md §A.3",
    )
    ap.add_argument(
        "--scene-scale",
        type=float,
        default=20.0,
        help="场景尺度(米),`DefaultStrategy.initialize_state` 用;本项目是 ~20 m 的近场气泡(§1.4)",
    )
    ap.add_argument("--no-runlog", action="store_true", help="不落 logs/ 三件套(默认每次运行都落)")
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子。**本模块原先全程没有 seed** —— 每步取哪 12 个视角(`np.random.choice`)、"
        "初始颜色(`torch.rand`)、深度初始化抽样(`torch.randperm`)全是自由随机 ⇒ "
        "同一份代码两次跑出的 val_psnr 实测差 4.5 dB(15.74 vs 11.29),**结论判不了**。"
        "2026-10-04 补;`--seed -1` 回到旧的自由随机行为",
    )
    args = ap.parse_args()

    with runlog.run("autodrivedata.gs.train_3dgs_mini") as rl:
        rl.input(args.capture, "capture")
        # ★ 三处随机源全部固定(见 `--seed` 的说明)。放在最前:数据加载里的 `torch.randperm`
        #   与颜色初始化都在它之后。
        if args.seed >= 0:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        rl.highlight("seed", args.seed)
        # 架构口径必须留痕:换卡/换 torch 后"内核编不出来"的那类报错**指向误导**(见模块头注)
        print(f"[arch] {ARCH_NOTE}")
        rl.highlight("cuda_arch", os.environ.get("TORCH_CUDA_ARCH_LIST", "?"))
        dev = "cuda"
        torch.manual_seed(0)
        np.random.seed(0)
        capture = project_path(args.capture)
        poses, f, H, W, cx, cy, viewmats_all, ks_all = _load_poses_and_cams(
            capture, downsample=args.downsample
        )
        n = len(poses)
        viewmats_all, ks_all = viewmats_all.to(dev), ks_all.to(dev)

        val_idx = [int(x) for x in args.val_frames.split(",")]
        train_idx = [i for i in range(n) if i not in val_idx]
        imgs = _load_images(capture, poses, H, W).to(dev)
        # 与 imgs **同序同尺寸**的深度,只给收尾的分箱诊断用(不动训练)
        depths = _load_depths(capture, poses, H, W).to(dev)

        points = _depth_init_points(capture, poses, viewmats_all, H, W, f, cx, cy).to(dev)
        n_init = points.shape[0]
        means = nn.Parameter(points.detach().clone())
        # 尺度存储口径:**开致密化存 log**(策略要 `exp`),关则线性(归档口径)
        _s0 = math.log(args.scale) if args.densify else args.scale
        scales = nn.Parameter(torch.full((n_init, 3), _s0, device=dev))
        quats = nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(n_init, 1))
        opac = nn.Parameter(torch.full((n_init,), 0.5, device=dev).log())
        col = nn.Parameter(torch.rand(n_init, 3, device=dev) * 0.5)
        # ⚠️ **lr 是配着尺度参数化走的,不能只换参数化不换 lr**(2026-10-04 实测)。
        # 单一 lr=1e-3 是给**线性**尺度配的:每步相对变化 1e-3/0.05 = **2%**。
        # 换成 log 尺度后同样 1e-3 只对应 **0.1%/步**(慢 20×)⇒ 实测**尺度几乎不动**
        # (落盘中位 0.0627、范围 0.028~0.105,而初始就是 0.05),模型只能靠挪位置与改颜色拟合,
        # 训练损失 **0.057 vs 无致密化的 0.008**(差 7×)。故开致密化时改用**标准 3DGS 的分组 lr**。
        if args.densify:
            # ⚠️ **每个参数一个独立优化器**,不是一个 Adam 开五个 param_group ——
            #    `DefaultStrategy.check_sanity` 断言 `len(optimizer.param_groups) == 1`
            #    (它要按 key 精确地替换"新长出来的"张量,多 group 的优化器它换不动)。
            opts_list = [
                torch.optim.Adam([{"params": [means], "lr": 1.6e-4}]),
                torch.optim.Adam([{"params": [scales], "lr": 5e-3}]),
                torch.optim.Adam([{"params": [quats], "lr": 1e-3}]),
                torch.optim.Adam([{"params": [opac], "lr": 5e-2}]),
                torch.optim.Adam([{"params": [col], "lr": 2.5e-3}]),
            ]
        else:
            opts_list = [torch.optim.Adam([{"params": [means, quats, scales, opac, col], "lr": 1e-3}])]

        # ---- 自适应密度控制(§A.3)。**默认关 ⇒ 与归档逐位一致** ----
        # ⚠️ `DefaultStrategy` 用 `torch.exp(params["scales"])` ⇒ **开致密化时尺度必须存 log**;
        #    关时保持线性(归档口径)。渲染一律取线性,故这里是唯一的换算点。
        strategy = None
        # 三个名字在两条路径上都要**有定义**(只在使用侧 `if strategy is not None` 不够 —— 类型
        # 检查看不到那层关联,`possibly unbound` 会一直挂着;运行期也确实更稳)
        strat_state: dict = {}
        # ★ 参数一律经 `_params` 读写 —— 密度控制会**替换字典里的张量对象**(不是原地改),
        #   谁还攥着旧的局部变量,谁就还在更新那份已被丢弃的参数。见下面 rasterize 的注释。
        # ★ **颜色也必须进 `_params`**(键名用 gsplat 惯例的 `sh0`):密度控制在
        #   增长/剪枝时对 `params` 的**每个 key** 做同样的增删(`ops` 里 `names=None` ⇒
        #   全量),漏掉谁,谁就与 `means` 形状对不上 —— 表现为 gsplat `rasterization`
        #   里一句关于 **颜色** 形状的断言失败(`AssertionError: torch.Size([N,3])`),
        #   而报错点完全指不到"密度控制改了形状"这件事(2026-10-04 实测)。
        _params: dict = {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opac,
            "sh0": col,
        }
        _opts: dict[str, torch.optim.Optimizer] = {}
        if args.densify:
            from gsplat.strategy import DefaultStrategy

            strategy = DefaultStrategy(verbose=False)
            # `check_sanity` 强制 `optimizers` 与 `params` **同键**,且**每个优化器只有一个
            # param_group`(见上面的建法)。颜色(`col`)不进策略 —— 它不参与密度决策。
            # `_opts` 必须与 `_params` **同键**(`check_sanity` 强制),故 5 个全给
            _opts = dict(zip(_params, opts_list, strict=True))
            strategy.check_sanity(_params, _opts)
            strat_state = strategy.initialize_state(scene_scale=args.scene_scale)

        def scale_lin() -> torch.Tensor:
            """渲染用的**线性**尺度(存储口径见上)。"""
            sc = _params["scales"]
            return sc.exp() if args.densify else sc

        def rasterize(idx):
            """→ `(colors, alphas, info)`。**info 不能丢** —— 密度控制要从它读
            `means2d` 的梯度 / `radii` / `gaussian_ids`。"""
            return gsplat.rasterization(
                _params["means"],
                _params["quats"],
                scale_lin(),
                _params["opacities"].sigmoid().clamp(max=0.9),
                # ⚠️ 颜色也必须走 `_params`(键 `sh0`)—— 密度控制把它与 `means` 一起增删,
                #   传局部变量就会形状对不上;而 gsplat 的断言报的是**颜色**形状,
                #   完全指不到"密度控制改了形状"这件事(2026-10-04 实测)
                _params["sh0"],
                viewmats_all[idx],
                ks_all[idx],
                W,
                H,
                near_plane=0.1,
                far_plane=1000.0,
            )

        def render(idx) -> torch.Tensor:
            return rasterize(idx)[0]

        best_val = float("inf")
        for it in range(args.iters):
            for o in opts_list:
                o.zero_grad()
            bidx = torch.tensor(np.random.choice(train_idx, _N_VIEWS_PER_STEP, replace=False)).to(dev)
            rend, _, rinfo = rasterize(bidx)
            loss = F.mse_loss(rend, imgs[bidx])
            if strategy is not None:
                # 顺序是硬约束:`pre` 要在 backward 前登记梯度钩子,`post` 在 `optim.step()` 后
                strategy.step_pre_backward(_params, _opts, strat_state, it, rinfo)
            loss.backward()
            for o in opts_list:
                o.step()
            if strategy is not None:
                # ⚠️ **`packed` 必须与 `rasterization` 实际返回的形式对上**(2026-10-04 实测):
                #    gsplat 1.5.3 返回的是 **packed** 形式 —— `means2d/radii/gaussian_ids`
                #    都是 `[nnz, …]`(nnz = 未被剔除的 (相机,高斯) 对),而策略**默认走
                #    non-packed 分支**(它要求 `radii` 是 `[C, N]` 网格)⇒
                #    `sel = (radii>0).all(-1)` 退化成 1 维、`torch.where(sel)[1]` **越界**
                #    (`IndexError: tuple index out of range`,报错点完全看不出是形状口径问题)。
                #    按实际维度判而不写死:换 gsplat 版本时返回形式可能变。
                strategy.step_post_backward(
                    _params, _opts, strat_state, it, rinfo, packed=rinfo["means2d"].dim() == 2
                )
            if it % 200 == 0 or it == args.iters - 1:
                with torch.no_grad():
                    vloss = F.mse_loss(render(val_idx), imgs[val_idx]).item()
                best_val = min(best_val, vloss)
                print(
                    f"[iter {it}/{args.iters}] train {loss.item():.4f} | val psnr {_mse_to_psnr(vloss):.2f}",
                    flush=True,
                )
                # **每 200 步那条也进 .jsonl**(与打印同节奏):3DGS 的指标是 PSNR 不是 mAP,
                # 按脚本实际指标落,不套 mAP 字段凑格式
                rl.metric(
                    it,
                    iter=it,
                    train_loss=round(loss.item(), 6),
                    val_psnr=round(-10 * math.log10(max(vloss, 1e-8)), 3),
                )

        # 全帧评估 + 留出帧 PSNR
        with torch.no_grad():
            rend_all = render(list(range(n)))
        psnrs = np.array([_mse_to_psnr(F.mse_loss(rend_all[i], imgs[i])) for i in range(n)])

        # ★ 阶段 A 诊断:**误差堆在哪一段深度上**。
        # 全场 PSNR 一个数看不出"是近场没建好还是远景糊了",而物体编辑发生在近场 ——
        # 若近场 PSNR 明显高于全场,编辑实验就不必等全场达标。见 docs/edit-3dgs-plan.md §4 阶段 A。
        by_depth = psnr_by_depth(((rend_all - imgs) ** 2).mean(dim=-1), depths)

        tag = f"_{args.tag}" if args.tag else ""
        out_dir = project_path("outputs/3dgs")
        # ⚠️ **尺度一律按线性落盘**(开致密化时内部存的是 log)⇒ 两种模式下 `.npy`/PLY 里
        #    `scales` 的含义一致,下游不必知道这次开没开致密化。
        #    ⚠️ 已知偏差:标准 3DGS 的 PLY 约定 `scale_*` 是 **log** 尺度,这里写的是线性 ——
        #    归档的 PLY 一直如此,不在本轮改动范围内(改它会让既有产物语义变)。
        scales_out = scale_lin().detach().cpu().numpy()
        for name, arr in (
            ("means", _params["means"]),
            # ⚠️ 颜色走 `_params["sh0"]` —— 密度控制把它与 means 一起增删,用局部变量会在
            #    `np.column_stack` 处报"长度不一致"(实测 882975 vs 120000),而那句报错
            #    仍然指不到"密度控制改了形状"(2026-10-04)
            ("col", _params["sh0"]),
            ("opac", _params["opacities"].sigmoid()),
        ):
            np.save(out_dir / f"{name}{tag}.npy", arr.detach().cpu().numpy())
        np.save(out_dir / f"scales{tag}.npy", scales_out)
        (out_dir / f"gaussians{tag}.ply").write_bytes(
            _standard_ply(
                _params["means"].detach().cpu().numpy(),
                _params["sh0"].detach().cpu().numpy(),
                _params["opacities"].sigmoid().detach().cpu().numpy(),
                scales_out,
            )
        )

        n_final = int(_params["means"].shape[0])  # 开了致密化就不再等于 n_init
        result = {
            "n_gaussians": n_final,
            "n_gaussians_init": n_init,
            "densify": bool(args.densify),
            "n_frames": n,
            "n_train": len(train_idx),
            "psnr_all_mean": round(float(psnrs.mean()), 2),
            "psnr_all_min": round(float(psnrs.min()), 2),
            "psnr_val": {str(i): round(float(psnrs[i]), 2) for i in val_idx},
            "iters": args.iters,
            "init": "truth-depth 网格反投影",
            "init_std": args.scale,
            "downsample": args.downsample,
            "val_frames": list(val_idx),
            # 阶段 A 诊断:误差堆在哪一段深度上(见 docs/edit-3dgs-plan.md §4)
            "psnr_by_depth": {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in by_depth.items()
            },
        }
        (out_dir / f"train_result{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        # 对比图:上=帧0(留出),下=训练集末帧(取中后帧保证多样);左 GT | 中 渲染 | 右 差值放大
        def to_uint8(t: torch.Tensor) -> np.ndarray:
            return (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        panes = []
        for i in (val_idx[0], n - 1):
            gt, rd = to_uint8(imgs[i]), to_uint8(rend_all[i])
            diff = np.abs(gt.astype(np.float32) - rd.astype(np.float32))
            diff = (diff / (diff.max() + 1e-6) * 255).astype(np.uint8)
            panes.append(np.concatenate([gt, rd, diff], axis=1))
        Image.fromarray(np.concatenate(panes, axis=0)).save(out_dir / f"render_compare{tag}.png")
        print(f"[done] psnr_all={result['psnr_all_mean']} val={result['psnr_val']} | {out_dir.resolve()}")
        print("  按深度分箱(占比 | PSNR):")
        for k, v in by_depth.items():
            print(f"    {k:>9}  {v['share'] * 100:5.1f}%  {v['psnr']:6.2f} dB")
        rl.highlight("psnr_all_mean", result["psnr_all_mean"])
        rl.highlight("psnr_all_min", result["psnr_all_min"])
        rl.highlight("best_val_loss", round(best_val, 6))
        rl.highlight("n_gaussians", n_init)
        rl.highlight("iters", args.iters)
        rl.highlight("tag", args.tag)
        for name in ("means", "scales", "col", "opac"):
            rl.artifact(out_dir / f"{name}{tag}.npy", f"gaussians-{name}")
        rl.artifact(out_dir / f"gaussians{tag}.ply", "gaussians-ply")
        rl.artifact(out_dir / f"train_result{tag}.json", "train-result")
        rl.artifact(out_dir / f"render_compare{tag}.png", "render-compare")


if __name__ == "__main__":
    main()

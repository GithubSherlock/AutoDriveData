"""P-G 3DGS mini 训练:CARLA 360° 环绕采集 → Gaussian Splatting → outputs/3dgs/。

输入:`outputs/3dgs/capture/`(bin/collect_3dgs.py 产物:多俯仰采集,每 pitch 一圈)。
  - 采集现按 pitch 分目录 images/p{p}/{i}.png + poses_{p}.json(见 pitches.json 枚举);
    本脚本跨 pitch 平铺(全局序号 = p_idx × frames_per_pitch + i),位姿仍用 CARLA 真值
    (定位降级:pycolmap SfM 对齐误差 ~5.7m,见 outputs/3dgs/sfm_eval.json)。
- 初始化:真值深度(sensor.camera.depth)稠密网格反投影,降采样至 ~40k 点。
- 训练:gsplat 光栅化 + Adam,后激活 RGB 颜色(sh_degree=None 口径)。
- 留出帧 0 不参与训练,作 val 算 PSNR(如实报告)。

落盘:
  outputs/3dgs/means{tag}.npy / scales / col / opac      (训练参数)
  outputs/3dgs/gaussians{tag}.ply                         标准 3DGS PLY
  outputs/3dgs/train_result{tag}.json                     PSNR + 帧数统计
  outputs/3dgs/render_compare{tag}.png                    GT|渲染|差值(帧0 与帧45)

用法(必须先建好 gsplat 扩展,见 CLAUDE.md 环境注意):
  CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.9 \
    python bin/train_3dgs_mini.py [--iters 1500] [--tag ep1500] [--scale 0.05]

环境注意:gsplat 通过 torch JIT 一次性编译(sm_89 本机缓存)。每次进程启动时
TORCH_CUDA_ARCH_LIST 必须与本机 arch 一致,否则 import 即抛 ValueError。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gsplat
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from autodrivedata.calib import CameraIntrinsics
from autodrivedata.paths import project_path

_DOWNSAMPLE = 2  # 1242x375 → 621x187
_N_INIT_PER_FRAME = 800  # 每帧深度采样点数(270 帧 → ~216k 候选,再降采样)
_N_INIT_TOTAL = 120000  # 多俯仰扩点:3 倍视角需 ~3 倍点预算才不稀释原环覆盖
_N_VIEWS_PER_STEP = 12
_DEPTH_LOWER = 1.0  # 深度初始化的下限(m,滤掉近身灰尘);上限 = 采集范围


def _load_poses_and_cams(capture: Path):
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
    # **两套空间的主点必须各自推对**(实测裁决见 bin/probe_calib.py A3/A4:
    # CARLA 渲染光栅是 **corner** 约定 —— 索引 i 的连续坐标就是 i,`cx = (w−1)/2 = 620.5`):
    # - 深度图/图像是 CARLA 光栅 → 下采样索引 j ↔ 原图索引 2j ↔ 原图连续坐标 2j;
    # - `ks` 喂给 gsplat(torch 原生光栅器,与 `grid_sample(align_corners=False)` 同族,
    #   **center** 约定:像素 j 覆盖 [j, j+1),中心 j+0.5)。
    # 令 gsplat 把原图索引 2j 的点画到像素 j:`fx_g·(2j−cx_orig)/fx_orig + cx_g = j + 0.5`
    # ⇒ `fx_g = fx_orig/2`、`cx_g = (cx_orig+1)/2`。于是
    #   fx = 621/2 = 310.5、cx = (620.5+1)/2 = 310.75、cy = (187+1)/2 = 94.0
    # (cx/cy 分别差 0.25/0.5 下采样像素 —— W 偶 H 奇,故两者不对称;历史写 W/2,H/2)
    _k = CameraIntrinsics(width=1242, height=375, fov_h_deg=90.0)
    H, W = 375 // _DOWNSAMPLE, 1242 // _DOWNSAMPLE
    f = _k.fx / _DOWNSAMPLE
    cx, cy = (_k.cx + 1.0) / _DOWNSAMPLE, (_k.cy + 1.0) / _DOWNSAMPLE
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
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    np.random.seed(0)
    capture = project_path(args.capture)
    poses, f, H, W, cx, cy, viewmats_all, ks_all = _load_poses_and_cams(capture)
    n = len(poses)
    viewmats_all, ks_all = viewmats_all.to(dev), ks_all.to(dev)

    val_idx = [int(x) for x in args.val_frames.split(",")]
    train_idx = [i for i in range(n) if i not in val_idx]
    imgs = _load_images(capture, poses, H, W).to(dev)

    points = _depth_init_points(capture, poses, viewmats_all, H, W, f, cx, cy).to(dev)
    n_init = points.shape[0]
    means = nn.Parameter(points.detach().clone())
    scales = nn.Parameter(torch.full((n_init, 3), args.scale, device=dev))
    quats = nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(n_init, 1))
    opac = nn.Parameter(torch.full((n_init,), 0.5, device=dev).log())
    col = nn.Parameter(torch.rand(n_init, 3, device=dev) * 0.5)
    optim = torch.optim.Adam([{"params": [means, quats, scales, opac, col], "lr": 1e-3}])

    def render(idx) -> torch.Tensor:
        out = gsplat.rasterization(
            means,
            quats,
            scales,
            opac.sigmoid().clamp(max=0.9),
            col,
            viewmats_all[idx],
            ks_all[idx],
            W,
            H,
            near_plane=0.1,
            far_plane=1000.0,
        )
        return out[0]

    best_val = float("inf")
    for it in range(args.iters):
        optim.zero_grad()
        bidx = torch.tensor(np.random.choice(train_idx, _N_VIEWS_PER_STEP, replace=False)).to(dev)
        rend = render(bidx)
        loss = F.mse_loss(rend, imgs[bidx])
        loss.backward()
        optim.step()
        if it % 200 == 0 or it == args.iters - 1:
            with torch.no_grad():
                vloss = F.mse_loss(render(val_idx), imgs[val_idx]).item()
            best_val = min(best_val, vloss)
            print(
                f"[iter {it}/{args.iters}] train {loss.item():.4f} | val psnr {-10 * math.log10(max(vloss, 1e-8)):.2f}",
                flush=True,
            )

    # 全帧评估 + 留出帧 PSNR
    with torch.no_grad():
        rend_all = render(list(range(n)))
    psnrs = np.array([-10 * math.log10(max(F.mse_loss(rend_all[i], imgs[i]).item(), 1e-8)) for i in range(n)])

    tag = f"_{args.tag}" if args.tag else ""
    out_dir = project_path("outputs/3dgs")
    for name, arr in (("means", means), ("scales", scales), ("col", col), ("opac", opac.sigmoid())):
        np.save(out_dir / f"{name}{tag}.npy", arr.detach().cpu().numpy())
    (out_dir / f"gaussians{tag}.ply").write_bytes(
        _standard_ply(
            means.detach().cpu().numpy(),
            col.detach().cpu().numpy(),
            opac.sigmoid().detach().cpu().numpy(),
            scales.detach().cpu().numpy(),
        )
    )

    result = {
        "n_gaussians": n_init,
        "n_frames": n,
        "n_train": len(train_idx),
        "psnr_all_mean": round(float(psnrs.mean()), 2),
        "psnr_all_min": round(float(psnrs.min()), 2),
        "psnr_val": {str(i): round(float(psnrs[i]), 2) for i in val_idx},
        "iters": args.iters,
        "init": "truth-depth 网格反投影",
        "init_std": args.scale,
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


if __name__ == "__main__":
    main()

"""离线渲染器:**加载一份存盘的高斯** + 一份 capture → 渲染指定帧。

## 为什么要有它(2026-10-04)

编辑的**公共出口**。在这之前全仓**没有任何离线渲染入口** ——
`train_3dgs_mini.main()` 里那个 `render()` 是**局部闭包**,训练一结束就没了
⇒ "渲染一份被改过的高斯"这件事根本做不到,编辑链路第一步就断。

它同时是 C0.1 的判据载体:对一份**未编辑**的高斯,本模块复算的逐帧 PSNR
必须与 `train_result{tag}.json` 对上(`--verify-json`,容差 ≤1e-3 dB)。
这条专抓「**两套渲染口径**」那类不报错、只是数偏的失效。

## ★ 存盘口径(编辑链路的公共接口)

盘上一份高斯 = **五个 `.npy`**(`outputs/3dgs/`):

| 文件 | 形状 | 口径 |
|---|---|---|
| `means{tag}.npy` | (N,3) | 世界系,米 |
| `rots{tag}.npy` | (N,4) | 四元数 `(w,x,y,z)`,**未归一化**(gsplat 内部归) |
| `scales{tag}.npy` | (N,3) | **线性**米(两种训练模式下都是;开致密化时内部存 log,落盘已 `exp` 回来) |
| `col{tag}.npy` | (N,3) | SH0 颜色(渲染直接用,不再做 `+0.5`) |
| `opac{tag}.npy` | (N,) | **已 sigmoid** 的不透明度 |

⚠️ **`rots` 原先没存**(2026-10-04 补)。在那之前盘上**没有任何一份产物能重建渲染** ——
`.npy` 缺 quats,而 `gaussians*.ply` 的 `rot` 被硬写成单位四元数(`_standard_ply` 的
签名里根本没有 quats)。见 `docs/edit-3dgs-plan.md` §4 阶段 C0。
⚠️ `opac` 存的是**激活后**的值,而渲染只做 `clamp(max=0.9)` —— **绝不能再 sigmoid 一次**
(那是静默的双激活:图看着像回事,数全错)。

## 唯一口径

`rasterize()` 从 `train_3dgs_mini` **抽出来共用**(原先在 `main()` 里):
训练时看到的渲染与编辑后看到的渲染必须是**同一套**。`train_3dgs_mini` 反过来 import 本模块。

⚠️ **本模块(以及 `edit_gs` / `eval_edit`)同样要 `import gsplat`** ⇒ 必须先
`from autodrivedata.gs import cuda_env`(它设 `TORCH_CUDA_ARCH_LIST`,并在没给
`CUDA_HOME` 时告警)。顺序由 `tests/gs/test_cuda_env.py` AST 钉住。

用法:
  # 渲染一份训好的模型(读 outputs/3dgs 的五个 .npy)
  python -m autodrivedata.gs.render_gs --capture outputs/3dgs_sync/capture \\
      --tag 3dgs_mini_30000 --frames 0-9 --out outputs/3dgs/render
  # 复算 PSNR 与产物 JSON 对账(C0.1 的判据)
  python -m autodrivedata.gs.render_gs --capture outputs/3dgs_sync/capture \\
      --tag 3dgs_mini_30000 --verify-json outputs/3dgs/train_result_3dgs_mini_30000.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from autodrivedata.gs import cuda_env
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

# ★ gsplat **只能经这里取** —— 顺序由函数保证,不靠文件里 import 的先后
#   (原先写成两行 `import`,`ruff format --fix` 会把 gsplat 挪到前面去,
#    cuda_env 就白设了;2026-10-04 实测被自动修坏过一次)。
gsplat = cuda_env.ensure_gsplat()

#: 与 `train_3dgs_mini` 的 `rasterize` 同一组近远平面。抽出来是为了**只有一个落点**。
NEAR_PLANE = 0.1
FAR_PLANE = 1000.0

#: 盘上高斯集的文件后缀(`f"{name}{tag}.npy"`)。
SET_FIELDS = ("means", "rots", "scales", "col", "opac")

#: `train_result*.json` 里的 PSNR **都是 `round(..., 2)` 存的**(见 `train_3dgs_mini` 的 `result`)——
#: 所以拿"未舍入的复算值"去比"已舍入的存值",**差最多 0.005 是必然的**,与渲染对不对无关。
#: 实测:一条完全正确的渲染被判红过(逐帧 Δ **全 0**,只有均值差 0.0013)。
#: ⇒ 本判据的分辨力由**产物的存储精度**决定,不是由渲染决定。要更细就得先改产物的落盘精度。
ROUND_DP = 2
TOL_DB = 0.5 * 10**-ROUND_DP


def mse_to_psnr(mse: float | torch.Tensor) -> float:
    """MSE → PSNR(dB)。**全仓唯一落点**(原先叫 `train_3dgs_mini._mse_to_psnr`)。"""
    return float(-10.0 * math.log10(max(float(mse), 1e-12)))


def rasterize(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales_lin: torch.Tensor,
    opac_act: torch.Tensor,
    col: torch.Tensor,
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    w: int,
    h: int,
):
    """→ `(colors, alphas, info)`。**info 不能丢** —— 密度控制要从它读
    `means2d` 的梯度 / `radii` / `gaussian_ids`。

    ⚠️ `opac_act` 是**已经 sigmoid 过**的不透明度,本函数据只做 `clamp(max=0.9)`。
    调用方不许再 sigmoid 一次 —— 存盘口径也是激活后的(见模块头注)。
    """
    return gsplat.rasterization(
        means,
        quats,
        scales_lin,
        opac_act.clamp(max=0.9),
        col,
        viewmats,
        ks,
        w,
        h,
        near_plane=NEAR_PLANE,
        far_plane=FAR_PLANE,
    )


@dataclass(frozen=True)
class GaussianSet:
    """一份高斯 —— **盘上五个 `.npy` 的完整对应物**。

    `opac` 是**激活后**的值(与 `opac{tag}.npy` 逐位一致),见模块头注那张表。
    """

    means: np.ndarray
    rots: np.ndarray
    scales_lin: np.ndarray
    col: np.ndarray
    opac: np.ndarray

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    def to_tensors(self, dev: str | torch.device) -> dict[str, torch.Tensor]:
        return {k: torch.as_tensor(v, dtype=torch.float32, device=dev) for k, v in self._asdict().items()}

    def _asdict(self) -> dict[str, np.ndarray]:
        return {
            "means": self.means,
            "rots": self.rots,
            "scales_lin": self.scales_lin,
            "col": self.col,
            "opac": self.opac,
        }

    def render(self, idx, viewmats: torch.Tensor, ks: torch.Tensor, w: int, h: int, dev) -> torch.Tensor:
        """渲染帧集 `idx` → `(len(idx), h, w, 3)`。"""
        p = self.to_tensors(dev)
        return rasterize(
            p["means"],
            p["rots"],
            p["scales_lin"],
            p["opac"],
            p["col"],
            viewmats[idx],
            ks[idx],
            w,
            h,
        )[0]


def set_path(out_dir: Path, name: str, tag: str) -> Path:
    """`outputs/3dgs/{name}{tag}.npy` —— 与 `train_3dgs_mini` 的落盘口径同一处规则。"""
    suffix = f"_{tag}" if tag else ""
    return out_dir / f"{name}{suffix}.npy"


def load_set(out_dir: Path, tag: str) -> GaussianSet:
    """读盘上一份高斯。**缺任何一个字段都当场抛** —— 缺 `rots` 的旧产物渲染不出正确结果,
    而"少了它照样能跑"正是那种不报错的错(见模块头注)。
    """
    got: dict[str, np.ndarray] = {}
    for name in SET_FIELDS:
        p = set_path(out_dir, name, tag)
        if not p.is_file():
            raise SystemExit(
                f"{p} 不存在 —— 盘上这份高斯不完整。⚠️ `rots` 是 2026-10-04 才补的,"
                "在那之前训出的产物**无法重建渲染**(quats 只被写进过 PLY,而 PLY 里那份是"
                "硬写的单位四元数);要重训,或显式给齐五个 --*-npy"
            )
        got[name] = np.load(p)
    n = got["means"].shape[0]
    bad = {k: v.shape for k, v in got.items() if v.shape[0] != n}
    if bad:
        raise SystemExit(f"字段长度不一致(means={n}):{bad} —— 这份高斯是被手工改坏的")
    return GaussianSet(
        means=got["means"].astype(np.float32),
        rots=got["rots"].astype(np.float32),
        scales_lin=got["scales"].astype(np.float32),
        col=got["col"].astype(np.float32),
        opac=got["opac"].astype(np.float32),
    )


def save_set(gs: GaussianSet, out_dir: Path, tag: str) -> list[Path]:
    """写盘上一份高斯(五个 `.npy`)。`edit_gs` 的产物走这条路,口径与训练侧逐字相同。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, arr in (
        ("means", gs.means),
        ("rots", gs.rots),
        ("scales", gs.scales_lin),
        ("col", gs.col),
        ("opac", gs.opac),
    ):
        p = set_path(out_dir, name, tag)
        np.save(p, arr.astype(np.float32))
        written.append(p)
    return written


def parse_frames(spec: str, n: int) -> list[int]:
    """`"0-9"` / `"0,45,90"` → 帧号列表。`"all"` = 全部。

    ⚠️ 本仓 `perception/` 下有 **8 份**同样的解析(既有重复,本轮不动它)。
    新增的三个 `gs/` 模块**只留这一份** —— 谁再写第四份就是又开一个口径。
    """
    if spec.strip().lower() == "all":
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i < n]
    if bad:
        raise SystemExit(f"帧号 {bad} 超出 capture 的 {n} 帧 —— 别静默截断")
    return sorted(set(out))


def _load_poses(capture: Path):
    """**复用训练侧的位姿/内参口径** —— 不在这里另推一遍。

    惰性 import:`train_3dgs_mini` 反过来 import 本模块(`rasterize`),
    模块级互相 import 会成环;函数级没有这个问题。
    """
    from autodrivedata.gs.train_3dgs_mini import _load_images, _load_poses_and_cams

    return _load_poses_and_cams, _load_images


def verify_against_json(gs: GaussianSet, capture: Path, result: dict, dev: str) -> dict:
    """★ C0.1 的判据:复算 `psnr_all_mean` 与 `psnr_val`,与产物 JSON 对账。

    判据的形态是「**两套渲染口径**」那类静默失效 —— 不报错,只是数偏。
    """
    load_poses, load_images = _load_poses(capture)
    poses, _f, H, W, _cx, _cy, viewmats, ks = load_poses(
        capture, downsample=result.get("downsample", 2), cam_convention=result["cam_convention"]
    )
    viewmats, ks = viewmats.to(dev), ks.to(dev)
    imgs = load_images(capture, poses, H, W).to(dev)

    n = len(poses)
    with torch.no_grad():
        rend = gs.render(list(range(n)), viewmats, ks, W, H, dev)
        psnrs = np.array([mse_to_psnr(torch.nn.functional.mse_loss(rend[i], imgs[i])) for i in range(n)])

    val_idx = [int(x) for x in result["val_frames"]]
    got_val = {str(i): round(float(psnrs[i]), 2) for i in val_idx}
    return {
        "n": n,
        "psnr_all_mean": round(float(psnrs.mean()), 2),
        "psnr_all_min": round(float(psnrs.min()), 2),
        "psnr_val": got_val,
        "ref": {
            "psnr_all_mean": result["psnr_all_mean"],
            "psnr_all_min": result["psnr_all_min"],
            "psnr_val": result["psnr_val"],
        },
        "delta": {
            "psnr_all_mean": round(float(psnrs.mean()) - result["psnr_all_mean"], 4),
            "val": {i: round(got_val[str(i)] - result["psnr_val"][str(i)], 4) for i in val_idx},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--capture", required=True, help="capture 目录(位姿/内参与训练侧同一口径)")
    ap.add_argument("--tag", default="", help="读 outputs/3dgs/{means,rots,scales,col,opac}{_tag}.npy")
    ap.add_argument("--gs-dir", default="outputs/3dgs", help="高斯五件套所在目录")
    ap.add_argument("--frames", default="0-9", help="`0-9` / `0,45` / `all`")
    ap.add_argument("--out", default="", help="落 PNG/NPY 的目录(默认 <gs-dir>/render/{tag})")
    ap.add_argument(
        "--verify-json",
        default="",
        help="★ 复算 PSNR 与 train_result*.json 对账(忽略 --frames,跑全帧)",
    )
    ap.add_argument("--no-runlog", action="store_true", help="不落 logs/ 三件套")
    args = ap.parse_args()

    capture = project_path(args.capture)
    gs_dir = project_path(args.gs_dir)
    dev = "cuda"

    with runlog.run("autodrivedata.gs.render_gs") as rl:
        rl.input(args.capture, "capture")
        print(f"[arch] {cuda_env.ARCH_NOTE}")
        print(f"[cuda] {cuda_env.CUDA_NOTE}")
        rl.highlight("cuda_home", cuda_env.home())
        rl.highlight("tag", args.tag)

        gs = load_set(gs_dir, args.tag)
        print(f"[gs] {gs.n} 个高斯 | 尺度中位 {float(np.median(gs.scales_lin)):.4f} m")
        rl.highlight("n_gaussians", gs.n)

        if args.verify_json:
            result = json.loads(project_path(args.verify_json).read_text(encoding="utf-8"))
            got = verify_against_json(gs, capture, result, dev)
            print(f"[verify] 复算 psnr_all={got['psnr_all_mean']} vs 产物 {got['ref']['psnr_all_mean']}")
            print(f"[verify] 逐帧 val Δ(复算 − 产物):{got['delta']['val']}")
            print(f"[verify] psnr_all_min 复算 {got['psnr_all_min']} vs 产物 {got['ref']['psnr_all_min']}")
            rl.highlight("verify_psnr_all_mean", got["psnr_all_mean"])
            rl.highlight("verify_delta_all", got["delta"]["psnr_all_mean"])
            rl.highlight("verify_delta_val_max", max(abs(v) for v in got["delta"]["val"].values()))
            # 容差 = **产物存储精度的一半**(见 `TOL_DB` 的头注)——
            # ⚠️ 逐帧 val 的 Δ 才是强证据:它若**恒为 0**,说明两条渲染路径逐位相同,
            #    均值那点差只是"产物的值被舍入过"。
            worst = max(
                [abs(got["delta"]["psnr_all_mean"])]
                + [abs(v) for v in got["delta"]["val"].values()]
                + [abs(got["psnr_all_min"] - got["ref"]["psnr_all_min"])]
            )
            exact_frames = all(v == 0.0 for v in got["delta"]["val"].values())
            tol = TOL_DB
            verdict = "✅" if worst <= tol else "❌"
            print(f"[verify] 逐帧 val Δ 是否**逐位为 0**:{exact_frames}")
            print(f"[verify] 最大偏差 {worst:.6f} dB(容差 {tol}) {verdict}")
            if worst > tol:
                raise SystemExit(
                    f"渲染口径对不上:最大偏差 {worst:.6f} dB > {tol} —— "
                    "训练侧与本模块的渲染不是同一套(见模块头注「唯一口径」)"
                )
        else:
            load_poses, _ = _load_poses(capture)
            poses, _f, H, W, _cx, _cy, viewmats, ks = load_poses(
                capture, downsample=2, cam_convention="carla"
            )
            viewmats, ks = viewmats.to(dev), ks.to(dev)
            idx = parse_frames(args.frames, len(poses))
            out_dir = project_path(args.out) if args.out else gs_dir / "render" / (args.tag or "untagged")
            out_dir.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                rend = gs.render(idx, viewmats, ks, W, H, dev)
            for j, i in enumerate(idx):
                p = poses[i]
                key = f"p{int(p['pitch'])}_{int(p['i']):05d}"
                np.save(out_dir / f"{key}.npy", rend[j].cpu().numpy())
                img = (rend[j].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                from PIL import Image as _I

                _I.fromarray(img).save(out_dir / f"{key}.png")
            print(f"[done] {len(idx)} 帧 → {out_dir.resolve()}")
            rl.artifact(out_dir, "render")


if __name__ == "__main__":
    main()

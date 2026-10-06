"""条件可控生成:把**本项目的真值条件**喂给 ControlNet,并量「条件到底进没进去」。

## 为什么这条线对本项目有意义

ControlNet 的 8 种条件里,`depth` / `seg` 在本项目**有真值**(`collect_3dgs --sem` 的语义 tag、
`depth/*.npy` 的度量深度),而官方 demo 只能拿 MiDaS / Uniformer 去**估计**
⇒ "用真值条件替代估计条件"是本项目在这条线上唯一站得住的差异点。

★ **而且只有 `depth` 能做到 JD 要的那件事**:canny 依赖纹理与光照,
**天气一变 canny 就变** ⇒ 它无法把「布局」与「外观」分开。`depth` 才能。

## 判据:**条件保真度**,不是"看着像"

对 depth 条件,正确的问题是"生成图里还剩下多少输入条件的结构"。做法是把生成图**重新过一遍
MiDaS** 得到它的深度条件,再与输入条件算 **Spearman 秩相关**(`depth_cond.spearman`)。

⚠️ **必须与"换一张条件"的臂比**。单看 ρ 读不出东西 —— 生成图天然是街景,
任何街景的深度图之间都有一定相关性。**本仓纪律:没有对照的读数读不出东西。**

⚠️ 阈值/步数/尺寸**必须随读数一起报**(同本仓 SAM3 那条:不带阈值的数不可比)。

## ★ 这把尺子的三条已知边界(2026-10-05 实测,不写清就会误读)

**① 整图 Spearman 被"共享先验"撑满。** 环形采集每一帧都是"下近上远",
跨帧对照实测 **0.9633 > 同帧 0.0360** —— 一个**没有判别力**的读数。
⇒ 要读横向结构(哪边近哪边远)必须**先减掉逐行均值**(去趋势),见 `map_probe` 的做法。

**② ρ 看不见条件的强度分布。** 这是最阴的一条:min-max 与秩等化
**保序**⇒ 与任何参考图的 ρ 几乎相同(实测 0.00335 vs 0.00327),
但真值条件被 `1/d` **挤在 131–227 的窄带**(发灰平场),而 MiDaS 铺满 0–255。
**ControlNet 吃的是像素值不是排序** ⇒ 一张 OOD 的条件图,而 ρ 一路正常。

**③ MiDaS 本身在 512² 中心裁上不稳定。** 实测同一换算下逐帧 ρ 从 **−0.36 到 0.993**,
且跨度由**场景深度范围**决定(0.7–21 m 的近景帧 ρ≈0.9,3–82 m 的远景帧 ρ≈0.03)。
⇒ **MiDaS 不能当"标准答案"**;它是官方 demo 的选择,不是可依赖的参照系。

★ **三条合起来的结论**:这条线上"ρ 高"**既不充分也不必要**。
判据要有意义,必须走**配对识别**(见 `docs/edit-image-plan.md` 阶段 E),
而不是拿一个整图 ρ 报数。

## 接线踩的坑(逐个留痕)

| 现象 | 真因 |
|---|---|
| `RuntimeError: Given groups=1, weight of size [16, 3, 3, 3], expected input[2, 1, 512, 512] to have 3 channels` | **hint 必须是 3 通道**。ControlNet 的 `input_hint_block` 首层是 `Conv2d(3,16,3)`,而条件源是**灰度**的 ⇒ 官方 demo 靠 `HWC3()` 把灰度复制成三通道。报错在 `ldm/modules/diffusionmodules/openaimodel.py`,`**不提**"你的条件图是单通道"`。本轮**我漏了这一步**、跑起来才炸 —— 见 `generate()` 里那句断言 |
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from autodrivedata.edit import cldm_backend as B
from autodrivedata.edit import depth_cond as DC
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: cond.py 与官方 demo 同口径的默认采样参数。
DEF_STEPS = 20
DEF_SCALE = 9.0
DEF_PROMPT = "a photo of a city street with buildings and a road, daytime, photorealistic"
DEF_NEG = "lowres, bad anatomy, worst quality, blurry, watermark"

#: 已接的条件源 → 需要哪个变体的权重。
CONDITION_KINDS = {
    "gt-depth": "depth",  # ★ 本项目的真值
    "midas-depth": "depth",  # 官方估计器(对照)
    "canny": "canny",
    # ★ **空白条件 = 没有结构约束**。它不是"一个更弱的条件",是**对照组**:
    #   §1.6 的假设是"**深度条件在保护被退化的物体**" —— 若换成空白条件后
    #   同一 prompt 生成出的图**掉点明显更多**,这条假设就成立。
    #   ⚠️ 用**常量灰**而不是全黑:全黑会被 `minmax_uint8` 判成退化条件而抛,
    #   且 SD 对全黑条件的响应不可控。灰 = "有图但没结构"。
    "blank": "depth",
}

#: ★ **所有条件源统一到这个方尺寸**。理由不是"好看":
#: 三个源天然产出**不同尺寸**(真值深度方裁后是 375²、MiDaS 是 512²、canny 是 375²),
#: 而**保真度判据要拿它们互相比较** —— 尺寸不齐会在 `spearman` 里抛,
#: 更坏的情况是有人"顺手 resize 一下"把两边的重采样口径弄成不一样,读数就没有意义了。
#: 512 与官方 demo 的 `detect_resolution` 默认同值。
COND_OUT = 512


@dataclass
class GenResult:
    """一次生成的产物与成本(全部实测,没有一个是推的)。"""

    cond: np.ndarray
    images: list[np.ndarray] = field(default_factory=list)
    seconds: float = 0.0
    peak_gib: float = 0.0


#: 两种采集布局。**别把它们的取帧路径写在一处 if 里** —— 两者的目录名与零填充位数都不同
#: (`collect_3dgs` 是 `capture/images/p0/00000.png` 5 位;KITTI root 是 `training/image_2/000000.png` 6 位),
#: 混着写迟早会出现"读数路径对、写盘路径错"这类只错一半的形态。
LAYOUTS = ("capture", "kitti")


def load_frame_rgb(root: Path, fid: int, layout: str = "capture") -> np.ndarray:
    """读一帧 RGB。`capture` = `collect_3dgs` 的 `capture/images/p*/{fid:05d}.png`;
    `kitti` = KITTI root 的 `training/image_2/{fid:06d}.png`(即 `collect_ab_route --depth` 的产物)。"""
    p = _frame_path(root, fid, layout, "image")
    if not p.exists():
        raise SystemExit(f"缺帧 {p}")
    return B.load_rgb(p)


def _frame_path(root: Path, fid: int, layout: str, kind: str) -> Path:
    """`kind` ∈ {image, depth};返回该帧的文件路径(**只负责拼路径,不检查存在**)。"""
    if layout == "capture":
        sub = {"image": "capture/images/p", "depth": "capture/depth/p"}[kind]
        cands = sorted(root.glob(f"{sub}*/"))
        if not cands:
            raise SystemExit(f"{root} 下没有 {sub}*/,这不是 `collect_3dgs` 的产物")
        return cands[0] / f"{fid:05d}.{'png' if kind == 'image' else 'npy'}"
    if layout == "kitti":
        sub = {"image": "training/image_2", "depth": "training/depth"}[kind]
        return root / sub / f"{fid:06d}.{'png' if kind == 'image' else 'npy'}"
    raise KeyError(f"未知布局 {layout!r};可选 {list(LAYOUTS)}")


def load_frame_depth(root: Path, fid: int, layout: str = "capture") -> np.ndarray:
    p = _frame_path(root, fid, layout, "depth")
    if not p.exists():
        raise SystemExit(
            f"缺真值深度 {p}\n"
            "  ⚠️ **归档里没有一份 root 同时有 `label_2` 与深度** —— "
            "要 KITTI 布局带深度,得用 `collect_ab_route --depth` 采。"
        )
    return np.load(p)


def build_condition(
    kind: str,
    *,
    rgb: np.ndarray,
    gt_depth: np.ndarray | None = None,
    midas_detector=None,
    normalize: str = "minmax",
    ref_cond: np.ndarray | None = None,
) -> np.ndarray:
    """造条件图**统一为 `COND_OUT` 见方**的 uint8。

    三个源的做法各自对齐官方 demo(方裁是 demo 的隐含前提;`midas` 那一路的
    `resize_image` 是**必需**的,见 `cldm_backend` 头注第六处)。

    `normalize` **只作用于 `gt-depth`** —— 另外两路给的是模型输出的 uint8 图,
    本来就落在 ControlNet 的训练分布里;需要修的恰恰是真值那一路
    (它被 `1/d` 挤在窄带里,见 `depth_cond.rank_uniform_uint8`)。
    """
    if kind not in CONDITION_KINDS:
        raise KeyError(f"未接的条件源 {kind!r};已接 {sorted(CONDITION_KINDS)}")
    if kind == "gt-depth":
        if gt_depth is None:
            raise ValueError("gt-depth 需要传 gt_depth(采集的 depth/*.npy)")
        # 真值深度与 RGB 同挂点同 fov ⇒ 用**同一个方裁**才是同一块像素
        sq = B.to_square_rgb(np.repeat(gt_depth[:, :, None], 3, axis=2))[:, :, 0]
        # 先换算再重采样:与官方 demo 的 `cv2.resize(detected_map, ...)` 同序
        # ⚠️ 归一化必须在**重采样之前**(在原始像素上做秩),否则插值会造出本来没有的中间值
        cond = DC.metric_depth_to_cond(sq.astype(np.float64), normalize=normalize, ref=ref_cond)
        return _to_cond_size(cond)
    if kind == "midas-depth":
        if midas_detector is None:
            raise ValueError("midas-depth 需要传入已构造的 detector(构造一次 470 MB,别每帧重建)")
        return _to_cond_size(B.midas_depth(rgb, midas_detector))
    if kind == "blank":
        return np.full((COND_OUT, COND_OUT), 128, np.uint8)
    return _to_cond_size(B.canny_edges(rgb))


def _to_cond_size(img: np.ndarray, out: int = COND_OUT) -> np.ndarray:
    """把条件图重采样到 `COND_OUT` 见方。已经是该尺寸就**原样返回**(不引入一次无谓的重采样)。"""
    import cv2

    a = np.asarray(img)
    if a.shape[:2] == (out, out):
        return a
    return cv2.resize(a, (out, out), interpolation=cv2.INTER_LINEAR)


def generate(
    cldm,
    cond: np.ndarray,
    *,
    prompt: str = DEF_PROMPT,
    n: int = 2,
    seed: int = 42,
    steps: int = DEF_STEPS,
    scale: float = DEF_SCALE,
) -> GenResult:
    """按条件生成 `n` 张。返回 `GenResult`(含实测耗时/显存)。"""
    import einops
    import torch

    h, w = cond.shape
    if h != w:
        raise ValueError(f"条件图必须是方的(官方 demo 的前提),收到 {cond.shape}")
    # ★ **必须是 3 通道** —— ControlNet 的 `input_hint_block` 第一层是 `Conv2d(3,16,3)`,
    #   喂单通道会在很深处报 `expected input[2, 1, 512, 512] to have 3 channels`
    #   (报错在 `ldm/modules/diffusionmodules/openaimodel.py`,**不提"你的条件图是灰度的"**)。
    #   官方 demo 是 `HWC3(detected_map)` 干的这件事,这里显式做同一件事。
    gray = np.repeat(np.asarray(cond, dtype=np.float32)[:, :, None] / 255.0, 3, axis=2)
    control = torch.from_numpy(gray).to(cldm.device)  # (h,w,3)
    control = einops.rearrange(control, "h w c -> 1 c h w").repeat(n, 1, 1, 1)
    if control.shape[1] != 3:
        raise AssertionError(f"hint 必须 3 通道,实际 {tuple(control.shape)}")
    torch.manual_seed(seed)
    t0 = time.time()
    with torch.no_grad():
        c = {"c_concat": [control], "c_crossattn": [cldm.model.get_learned_conditioning([prompt] * n)]}
        uc = {"c_concat": [control], "c_crossattn": [cldm.model.get_learned_conditioning([DEF_NEG] * n)]}
        latent = h // 8
        samples, _ = cldm.sampler.sample(
            steps,
            n,
            (4, latent, latent),
            c,
            verbose=False,
            eta=0.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
        )
        x = cldm.model.decode_first_stage(samples)
    imgs = (
        (einops.rearrange(x, "b c h w -> b h w c") * 127.5 + 127.5)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )
    return GenResult(
        cond=np.asarray(cond),
        images=[imgs[i] for i in range(n)],
        seconds=time.time() - t0,
        peak_gib=torch.cuda.max_memory_allocated() / 2**30,
    )


def condition_fidelity(gen_rgb: np.ndarray, ref_cond: np.ndarray, *, midas_detector) -> float:
    """**条件保真度**:把生成图重新估一遍深度条件,与参考条件算 Spearman。

    只用 `depth` 变体时有意义(要 MiDaS 去估计生成图的深度)。
    """
    if midas_detector is None:
        raise ValueError("条件保真度需要 MiDaS detector(它给的是'生成图里还剩多少深度结构')")
    back = build_condition("midas-depth", rgb=gen_rgb, midas_detector=midas_detector)
    return DC.spearman(ref_cond, back)


def save_png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = a.clip(0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--root", required=True, help="采集 root(capture 布局 = collect_3dgs;kitti 布局 = collect_ab_route)"
    )
    ap.add_argument(
        "--layout",
        default="capture",
        choices=LAYOUTS,
        help="取帧布局。`kitti` 用于 `collect_ab_route --depth` 的产物(★ 唯一同时有 label_2 与真值深度的来源)",
    )
    ap.add_argument("--frames", default="0", help="逗号分隔或 a-b")
    ap.add_argument("--kind", default="gt-depth", choices=sorted(CONDITION_KINDS))
    ap.add_argument(
        "--normalize",
        default="minmax",
        choices=sorted(DC.NORMALIZATIONS),
        help="**只作用于 gt-depth**。真值条件被 1/d 挤在窄带里,minmax 会出一张发灰的平场;"
        "histeq 保序但把强度铺开。三者与 MiDaS 的 Spearman 几乎相同 ⇒ **ρ 看不出这个差别**",
    )
    ap.add_argument("--prompt", default=DEF_PROMPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=2, help="每帧生成几张")
    ap.add_argument("--steps", type=int, default=DEF_STEPS)
    ap.add_argument("--scale", type=float, default=DEF_SCALE)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--fidelity",
        action="store_true",
        help="额外量**条件保真度**(生成图重估深度 → 与输入条件的 Spearman)。"
        "⚠️ 会同时给**对照臂**(与另一帧的条件比)—— 单看自比那个 ρ 读不出东西",
    )
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    root, out = project_path(args.root), project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fids = DC._parse_frames(args.frames)
    variant = CONDITION_KINDS[args.kind]

    with runlog.run("autodrivedata.edit.conditioned_gen") as rl:
        rl.input(root, "capture-root")
        rl.highlight("condition_kind", args.kind)
        # ⚠️ 这三个必须随读数一起报(SAM3 那条纪律)
        rl.highlight("steps", args.steps)
        rl.highlight("guidance_scale", args.scale)
        rl.highlight("seed", args.seed)

        cldm = B.load_cldm(variant)
        print(f"[model] {variant} 已加载  sha256={cldm.sha256[:12]}…  设备 {cldm.device}")

        # MiDaS 在这三种情况下要构造:`midas-depth` 本身 / 量保真度 / `match` 归一化要参考分布
        need_midas = args.kind == "midas-depth" or args.fidelity or args.normalize == "match"
        midas = None
        if need_midas:
            from annotator.midas import MidasDetector

            midas = MidasDetector()

        # 先把每帧的条件都造出来 —— 保真度的**对照臂**要拿"别的帧的条件",必须全部就位
        conds: dict[int, np.ndarray] = {}
        rgba: dict[int, np.ndarray] = {}
        for fid in fids:
            rgba[fid] = load_frame_rgb(root, fid, args.layout)
            gt = load_frame_depth(root, fid, args.layout) if args.kind == "gt-depth" else None
            # `match` 的参考分布取**本帧的 MiDaS 条件** —— 这是"若对齐到估计器的分布会怎样"的上界版;
            # ⚠️ 它**不可部署**(部署就没有 MiDaS 了),只用来回答"分布是不是问题"
            ref = None
            if args.normalize == "match" and args.kind == "gt-depth":
                ref = build_condition("midas-depth", rgb=rgba[fid], midas_detector=midas)
            conds[fid] = build_condition(
                args.kind,
                rgb=rgba[fid],
                gt_depth=gt,
                midas_detector=midas,
                normalize=args.normalize,
                ref_cond=ref,
            )
            save_png(out / f"cond_{args.kind}_{args.normalize}_{fid:05d}.png", conds[fid])

        # ⚠️ 这里**不要**再定义一次 `need_midas` —— 上面那个是"要不要构造 detector",
        #    这里要问的是"要不要算保真度"。第一版两处同名,后一处把前一处覆盖掉了。
        want_fidelity = args.fidelity and variant == "depth"
        assert midas is not None or not want_fidelity

        rows = []
        for fid in fids:
            res = generate(
                cldm,
                conds[fid],
                prompt=args.prompt,
                n=args.n,
                seed=args.seed,
                steps=args.steps,
                scale=args.scale,
            )
            for i, im in enumerate(res.images):
                save_png(out / f"gen_{args.kind}_{fid:05d}_{i}.png", im)
            row = {
                "fid": fid,
                "cond_kind": args.kind,
                "cond_mean": float(conds[fid].mean()),
                "s": round(res.seconds, 2),
                "peak_gib": round(res.peak_gib, 2),
                "n": len(res.images),
            }
            if want_fidelity and len(fids) > 1:
                other = next(f for f in fids if f != fid)
                # ★ 两条臂**必须成对报**:自比高不等于条件进去了 —— 生成图天然都是街景,
                #   任何两张街景深度图之间本来就有相关性,那才是 floor
                rho_self = [condition_fidelity(im, conds[fid], midas_detector=midas) for im in res.images]
                rho_ctrl = [condition_fidelity(im, conds[other], midas_detector=midas) for im in res.images]
                row["rho_self"] = [round(r, 4) for r in rho_self]
                row["rho_control_other_frame"] = [round(r, 4) for r in rho_ctrl]
                row["rho_margin"] = round(float(np.mean(rho_self) - np.mean(rho_ctrl)), 4)
                rl.highlight(f"rho_self_f{fid}", round(float(np.mean(rho_self)), 4))
                rl.highlight(f"rho_ctrl_f{fid}", round(float(np.mean(rho_ctrl)), 4))
            rows.append(row)
            extra = ""
            if "rho_margin" in row:
                extra = (
                    f" | 保真度 自比 {np.mean(row['rho_self']):.4f}"
                    f" vs 对照 {np.mean(row['rho_control_other_frame']):.4f}"
                    f" ⇒ 余量 {row['rho_margin']:+.4f}"
                )
            print(
                f"[{fid:05d}] 条件均值 {row['cond_mean']:6.1f} | 生成 {res.seconds:.1f}s | "
                f"峰值 {res.peak_gib:.2f} GiB{extra}"
            )
            rl.metric(fid, gen_seconds=round(res.seconds, 3), peak_gib=round(res.peak_gib, 3))

        summ = out / "gen_summary.json"
        summ.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1), encoding="utf-8")
        rl.highlight("n_frames", len(rows))
        rl.artifact(summ, "summary")
        print(f"[done] {summ.resolve()}")


if __name__ == "__main__":
    main()

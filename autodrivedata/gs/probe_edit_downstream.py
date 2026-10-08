"""**编辑接感知闭环**:这个 3D 编辑,让下游模型**看见**了吗?

## 它回答的是 JD 那句原话

「**确保仿真数据可用于感知模型训练**」。前面所有编辑判据量的都是**画质**(PSNR),
而这一条量的是**下游能不能用**:编辑**前后**,同一个检测器在同一批视角上**检出数变了没有**。

## ★ 为什么必须用**开放词表**后端

被编辑的道具是 `static.prop.warningconstruction`(**施工围挡** —— 一类路障)。
它**不在** KITTI 三类、也**不在** COCO 80 类里 ⇒ 闭集检测器**按定义检不出它**,
"编辑前后都没检出"会把这件事读成"编辑没生效",而实际是**模型没有这个概念**。

⇒ 用 **SAM3 的开放词表**:提示词直接给 `barrier`。本仓已实测它在这类道具上
`barrier` **30/30** 命中(见 CLAUDE.md 的 SAM3 条目)。

## 判据

| 臂 | 期望 |
|---|---|
| `ab3_B`(**没有**道具) | `barrier` 检出 **0** |
| `ab3_B + prop`(插入后) | `barrier` 检出 **> 0** |

★ **控制臂**——**错位置插入**(`insert_gs --shift`):检出应当**掉下来**。
少了这条,"插完就检出了"可能只是"随便加一堆高斯都会让检测器乱报"。

⚠️ **渲染分辨率只有 621×187**(downsample 2),道具在 6 m 处 —— 先报**框高像素**,
别拿一个 3 px 的框去谈检出。
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from autodrivedata.gs import cuda_env  # noqa: F401
from autodrivedata.gs.eval_edit import parse_frames
from autodrivedata.gs.render_gs import load_set
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 被编辑物体的**开放词表**提示词。⚠️ 与 `sam3_backend.DETECT_PROMPTS` 无关 ——
#: 那是**项目三类的**提示词表;这里是"这一类道具叫什么",是**这次编辑实验**的自变量。
PROP_PROMPT = "barrier"


def detections(images: np.ndarray, *, prompt: str, threshold: float) -> list[dict]:
    """对一批 RGB 帧跑 SAM3 开放词表。返回每帧的 `{n, boxes, max_conf}`。

    ⚠️ 走 **`sam3_backend.segment`** 而不是 `detect` —— 后者的 `DETECT_PROMPTS`
    只认**项目三类**且**故意拒绝调用点现编**(那条守卫是对的:类名表是项目口径)。
    而这里的提示词是**这次实验的自变量**(道具叫什么),所以用**开放词表**那一路
    (`segment` 直接吃 concepts)。**不另写前向链**(那会漂)。
    """
    from autodrivedata.perception import sam3_backend

    out = []
    for im in images:
        # SAM3 吃 0–255 的 RGB uint8(与 `Sam3Predictor.__call__` 同口径)
        found = sam3_backend.segment(im.astype(np.uint8), (prompt,), threshold=threshold, dedup_iou=None)
        out.append(
            {
                "n": len(found),
                "max_conf": float(max((d.conf for d in found), default=0.0)),
                # 掩膜的外接框高度(像素)—— 用来判"太小了没检出"还是"真没检出"
                "heights": [round(float(np.ptp(np.where(d.mask)[0])), 1) for d in found if np.any(d.mask)],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--capture", default="outputs/3dgs_ab3/A/capture")
    ap.add_argument("--gs-dir", default="outputs/3dgs")
    ap.add_argument("--tags", required=True, help="逗号分隔:基线 tag,编辑后 tag[,错位置 tag]")
    ap.add_argument("--frames", default="45,90,135,180,225")
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--prompt", default=PROP_PROMPT)
    ap.add_argument("--threshold", type=float, default=0.5, help="SAM3 默认阈值")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    with runlog.run("autodrivedata.gs.probe_edit_downstream") as rl:
        from autodrivedata.gs.train_3dgs_mini import _load_poses_and_cams

        cap = project_path(args.capture)
        gs_dir = project_path(args.gs_dir)
        poses, _f, h, w, _cx, _cy, viewmats, ks = _load_poses_and_cams(cap, downsample=args.downsample)
        dev = "cuda"
        viewmats, ks = viewmats.to(dev), ks.to(dev)
        idx = parse_frames(args.frames, len(poses))
        rl.highlight("prompt", args.prompt)
        rl.highlight("threshold", args.threshold)

        rep: dict = {"prompt": args.prompt, "threshold": args.threshold, "arms": {}}
        print(f"\n=== 编辑 → 下游(SAM3 开放词表,提示词「{args.prompt}」,阈值 {args.threshold})===")
        for tag in [t.strip() for t in args.tags.split(",") if t.strip()]:
            with torch.no_grad():
                imgs = load_set(gs_dir, tag).render(idx, viewmats, ks, w, h, dev).cpu().numpy()
            d = detections(imgs * 255.0, prompt=args.prompt, threshold=args.threshold)
            n_total = sum(x["n"] for x in d)
            heights = [hh for x in d for hh in x["heights"]]
            rep["arms"][tag] = {
                "n_total": n_total,
                "n_frames_with_det": sum(1 for x in d if x["n"]),
                "max_conf": float(max((x["max_conf"] for x in d), default=0.0)),
                "det_heights_px": heights,
                "per_frame": [x["n"] for x in d],
            }
            print(
                f"  {tag:<20} 检出 {n_total:>3}(含检出的帧 {rep['arms'][tag]['n_frames_with_det']}/{len(idx)})"
                f"  最高 conf {rep['arms'][tag]['max_conf']:.3f}"
                + (f"  框高 {heights} px" if heights else "")
            )
            rl.highlight(f"{tag}_detections", n_total)
        rep["verdict"] = verdict(rep)
        print(f"  ⇒ {rep['verdict']}")
        rl.highlight("verdict", rep["verdict"])
        if args.out:
            p = project_path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
            rl.artifact(p, "report")
            print(f"[done] {p.resolve()}")


def verdict(rep: dict) -> str:
    """★ 三分。**先看基线是不是 0**,再看编辑后有没有抬起来,最后看控制臂掉没掉。"""
    tags = list(rep["arms"])
    if len(tags) < 2:
        return "未判(至少要两个臂:基线 + 编辑后)"
    base, edited = rep["arms"][tags[0]]["n_total"], rep["arms"][tags[1]]["n_total"]
    if base != 0:
        return f"未判(基线**本来就有** {base} 个检出 ⇒ 这一格里'编辑让它出现'没有判别力)"
    if edited == 0:
        return "★ **编辑没能让下游看见**:插入后仍然 0 个检出 —— 但先查框高(太小可能只是分辨率不够)"
    ctrl = f";控制臂 {rep['arms'][tags[2]]['n_total']} 个" if len(tags) > 2 else "(**无控制臂**)"
    return f"★ **编辑让下游看见了**:基线 0 → 编辑后 **{edited}** 个{ctrl}"


if __name__ == "__main__":
    main()

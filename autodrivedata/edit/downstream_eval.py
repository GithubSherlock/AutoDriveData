"""下游闭环:**「生成的退化」vs「真值的退化」** —— JD 那句「确保仿真数据可用于感知模型训练」的落点。

## 问的是什么

JD 那句话的可测形式是:**同一套 GT 配上不同来源的退化图,看检测器掉多少**。

## 四臂(★ **同一后端、同一 conf、同一次会话** —— 跨会话/跨后端不可比,是本仓红线)

| 臂 | `image_2` | `label_2` |
|---|---|---|
| **A** clear | 真值晴天方裁帧 | 真值 GT |
| **B** truth-fog | 真值浓雾方裁帧 | 真值 GT |
| **C** gen-fog | **生成**浓雾(条件 = 晴天帧的**真值深度**) | 真值 GT |
| **D** gen-rot | **与 C 同一批图,但整体旋转 K 帧** | **与 A 逐条相同** |

★ **D 是判据的分辨力来源**:没有它,就分不清"生成掉点"与"GT 本来就配不上"。
   `AP_C` 明显高于 `AP_D` ⇒ 生成图确实与 GT 对得上(布局保住了)。
   ⚠️ **旋转的是图,不是 GT** —— 四臂的 GT **按构造逐条相同**,差里才只有"布局"这一个变量。

## 三档解读(**方向必须能分辨两头的错**)

- `Δ_truth ≈ Δ_gen` ⇒ 生成的退化**可替代**真值退化 ← JD 要的答案
- `Δ_gen ≪ Δ_truth` ⇒ 生成图**过毒**(掉得比真退化还狠),不能用于训练
- `Δ_gen ≈ 0`       ⇒ **假退化**(生成根本没加雾)

⚠️ 两种"看着都一样"的失败,只有靠 D 臂才分得开:
   ① 生成图把布局丢了 ⇒ C 与 D 一起低;
   ② 生成图过毒      ⇒ C 低但 D 更低不了(它是地板)。

## 边界(必须随读数一起报)

中心方裁要付代价:实测 183 条 GT 里 **kept 115 / clipped 55 / dropped 13**。
被裁断的框按本仓 2D 口径红线**单独分类**,不并入裁决(此处只是沿用方裁 root 的产物)。

⚠️ **生成图尺寸必须与方裁 root 一致**(512²)—— `kitti_square` 与生成侧
`conditioned_gen.COND_OUT` 同值,且两者的中心裁起点用**同一套取整**(见
`kitti_square.crop_geometry` 的头注与 `test_crop_matches_generator`)。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from autodrivedata.perception.backends import DEFAULT_YOLO_WEIGHT
from autodrivedata.perception.eval_2d_ab import describe, make_predictor, report, resolve_conf
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: D 臂默认平移多少帧(取半程)。见 `run_arms` 头注:平移的不是"配对"是"图集合"。
DEF_ROTATE = 35


def build_arm(
    dst: Path,
    *,
    image_src: Path | None = None,
    image_files: list[Path] | None = None,
    label_src: Path,
    image_shift: int = 0,
    frames: list[str] | None = None,
) -> int:
    """组一个 arm root(`training/{image_2,label_2}`)。返回帧数。

    `image_shift` = K 表示"从第 K 张图开始取",配第 0.. 条 GT —— **D 臂就是它**。

    ## ⚠️ 为什么是"平移图"而不是"旋转图"(2026-10-06 两版实测)

    **第一版**旋转 `label_2`:D 臂的 GT **条数都变了**(实测 20 vs 40,因为
    `eval_2d_ab --limit N` 只读前 N 个 label,旋转后那 N 个来自另一段帧)⇒ 不可比。

    **第二版**旋转 `image_2`(`(i+K) mod N`):**D 与 C 逐位相同**(mAP 0.8072 = 0.8072、
    检出 174 = 174)。真因不是代码错,是 **`eval_2d_ab` 把**所有帧的框**池化**后算 AP**
    (见 `load_gt` / `detect`:它们把逐帧结果 append 进同一个 list)——
    ⇒ **配对顺序对池化统计毫无影响,重排 = 恒等**。
    我那时还核了 sha256 确认旋转真的落盘了(D 的第 0 张确实等于 C 的第 35 张),所以
    "没生效"这个怀疑方向是错的 —— **错的不是旋转,是"配对"这件事在这个口径下不存在**。

    ⇒ 正确的对照必须让**图的集合**都不一样:取**后半程**的图配**前半程**的 GT
    (ego 已开出 ~28 m,车的位置全变了)。调用方相应地只评前半程,四臂 GT 才相等。
    """
    if (image_src is None) == (image_files is None):
        raise SystemExit("`image_src` 与 `image_files` 必须**恰好给一个**")
    imgs = sorted(image_src.glob("*.png")) if image_src is not None else sorted(image_files)
    labs = sorted(label_src.glob("*.txt"))
    n_img_all = len(imgs)
    if frames is not None:
        want = set(frames)
        labs = [p for p in labs if p.stem in want]
        # ⚠️ **只对 `image_src`(KITTI 布局目录)按 `frames` 过滤**。两条都错过的坑:
        #    ① `image_shift != 0` 时过滤会把"D 臂要用的后半程"先滤掉;
        #    ② ★ `image_files` 的名字是 `gen_{kind}_{fid}_{i}.png`,`stem` **不是帧号**
        #       ⇒ 过滤后必然为空,而报错说的是"为空",**完全指不到真因**
        #       (2026-10-06 实测就在这里卡了一轮)。列表由调用方给,这里不再筛。
        if image_src is not None and image_shift == 0:
            imgs = [p for p in imgs if p.stem in want]
    if not imgs or not labs:
        which = "图" if not imgs else "GT"
        raise SystemExit(
            f"组 arm 失败:{which}为空(图 {n_img_all}→{len(imgs)} 张、GT {len(labs)} 条;"
            f"源 = {image_src or image_files})"
        )
    n = len(labs)
    # 只有 **KITTI 布局目录**(`image_src`)才要求"图的张数 == GT 条数" ——
    # 那是"A/B 配对前提"的检查;`image_files` 是调用方给的显式列表,由下面的 shift 逻辑管。
    if image_src is not None and len(imgs) != n:
        raise SystemExit(f"图与 GT 条数不等:{len(imgs)} vs {n} —— A/B 配对的前提没了")
    if len(imgs) < image_shift + n:
        raise SystemExit(f"图不够:{len(imgs)} 张,要从第 {image_shift} 张起取 {n} 张")
    used = imgs[image_shift : image_shift + n]

    (dst / "training/image_2").mkdir(parents=True, exist_ok=True)
    (dst / "training/label_2").mkdir(parents=True, exist_ok=True)
    for i, lp in enumerate(labs):
        # GT **原样**(所有臂共用同一套) —— 只有图按 `image_shift` 取
        shutil.copyfile(lp, dst / "training/label_2" / f"{i:06d}.txt")
        shutil.copyfile(used[i], dst / "training/image_2" / f"{i:06d}.png")
    return n


def gen_image_files(gen: Path) -> list[Path]:
    """从 `conditioned_gen` 的**扁平**产物里挑出"每帧第一张"(`--n 1` 的那张)。

    ⚠️ **必须按后缀 `_0.png` 挑,不许 `glob("gen_*.png")` 全收** —— `--n > 1` 时
    同一个条件会有多张,全收会让帧数与 GT 条数不等(那一步会抛,但抛的地方离真因很远)。
    ⚠️ 也**不许**用 `glob("*.png")`:同目录里躺着 `cond_*.png`,拿它当生成图评出来的 AP
    照样像个数。
    """
    files = sorted(gen.glob("gen_*_0.png"))
    if not files:
        raise SystemExit(
            f"{gen} 下没有 `gen_*_0.png` —— 生成那一步先跑:\n"
            "  python -m autodrivedata.edit.conditioned_gen --layout kitti ... --n 1"
        )
    return files


def run_arms(clear: Path, fog: Path, gen: Path, work: Path, *, rotate: int | None = None) -> dict:
    """组四臂。返回每臂的帧数(全部实测)。

    ★ **`rotate` 一给,四臂就都只评前半程**(`n // 2` 帧),D 的图取后半程 ——
    这是"图集合不同"这条对照的**代价**:少用一半帧。不给 `rotate` 时四臂各 70 帧、无对照臂。
    ⚠️ 四臂的帧数**必须一致**(否则 GT 集不同,Δ 不可归因)—— 所以这里由函数统一切,不靠调用方记得。

    ## ★★ GT 必须**冻结成一套**(2026-10-06 实测踩到)

    第一版让 B 臂带 `fog` 自己的 `label_2`(其余三臂带 `clear` 的),想当然地以为
    "A/B 是同一批物体 ⇒ 两份 GT 逐条相同"。**实测不是**:

    | 量 | 实测 |
    |---|---|
    | 条数 / 帧号 | **完全相等**(119 条、35 帧) |
    | 2D 框(排序后) | 差 **≤ 0.78 px** |
    | 3D 位置(排序后) | 差 **≤ 0.06 m**(正是 A/B 硬门槛 `≤ 0.07 m` 那一档) |
    | **行序** | **不同**(CARLA `get_actors()` 的顺序跨采集不稳定)\
    ⇒ 逐行 zip 对比会误报"全错 215 px",**必须按多重集比** |

    这 0.78 px 足以让一条压在 `IoU=0.5` 上的检测翻面,于是同一份预测换一套 GT 就
    差 **0.009 AP**(实测 `0.6521` vs `0.6429`)—— 与本仓"AP 是台阶函数"同族。
    ⇒ 四臂**共用 `clear` 那一套 GT**,单变量只剩"图"。
    """
    gfiles = gen_image_files(gen)
    n_all = min(len(gfiles), len(sorted((clear / "training/label_2").glob("*.txt"))))
    if rotate is None:
        rotate = n_all // 2
    n_keep = n_all - rotate
    if n_keep <= 0:
        raise SystemExit(f"帧数不足以切出对照臂:{n_all} 帧、平移 {rotate}")
    keep = [f"{i:06d}" for i in range(n_keep)]
    gt = clear / "training/label_2"  # ★ **唯一 GT 源** —— 四臂共用,别换成各臂自己的
    arms = {
        "A_clear": dict(image_src=clear / "training/image_2", label_src=gt, frames=keep),
        "B_truth_fog": dict(image_src=fog / "training/image_2", label_src=gt, frames=keep),
        "C_gen_fog": dict(image_files=gfiles, label_src=gt, frames=keep),
        # ★ 对照臂:用**后半程**的生成图(ego 已开出 ~28 m)配前半程的 GT
        "D_gen_shift": dict(image_files=gfiles, label_src=gt, frames=keep, image_shift=rotate),
    }
    # ⚠️ **返回的键必须恰好是四个臂名** —— 第一版在这里塞了 `_n_keep` / `_shift` 两个元数据键,
    #    而 `main` 是 `for name in arms:` 逐臂评测的 ⇒ 它去评了一个叫 `_n_keep` 的目录,
    #    得到 `mAP=nan` 并**继续往下跑**(不报错)。元数据由 `main` 从臂里读。
    out: dict = {}
    for name, kw in arms.items():
        out[name] = {"n_frames": build_arm(work / name, **kw), "shift": rotate, "n_keep": n_keep}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--clear", required=True, help="晴天方裁 root(kitti_square 的产物)")
    ap.add_argument("--fog", required=True, help="真值浓雾方裁 root")
    ap.add_argument("--gen", required=True, help="生成浓雾 root(conditioned_gen 的产物)")
    ap.add_argument("--work", required=True, help="四臂 root 的落点")
    ap.add_argument("--rotate", type=int, default=None, help="D 臂平移帧数(默认半程);给了则四臂只评前半程")
    ap.add_argument("--backend", choices=("sam3", "yolo"), default="sam3")
    ap.add_argument("--conf", type=float, default=None, help="不传则按后端取默认")
    ap.add_argument("--weight", default=DEFAULT_YOLO_WEIGHT)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    clear, fog, gen, work = (project_path(p) for p in (args.clear, args.fog, args.gen, args.work))
    with runlog.run("autodrivedata.edit.downstream_eval") as rl:
        for p, k in ((clear, "clear"), (fog, "fog"), (gen, "gen")):
            rl.input(p, k)
        # ⚠️ 这三个必须随读数一起报(本仓 SAM3 那条:不带阈值的数不可比)
        conf = resolve_conf(args.backend, args.conf)
        rl.highlight("backend", args.backend)
        rl.highlight("conf", conf)
        rl.highlight("iou", args.iou)
        if args.rotate:
            rl.highlight("shift_frames", args.rotate)

        arms = run_arms(clear, fog, gen, work, rotate=args.rotate)
        predict = make_predictor(args.backend, args.weight, conf)
        print(describe(args.backend, predict))

        aps: dict[str, float] = {}
        for name in arms:
            aps[name] = report(work / name, predict, conf, args.iou, args.limit)
            rl.highlight(f"mAP_{name}", round(aps[name], 4))

        d_truth = aps["B_truth_fog"] - aps["A_clear"]
        d_gen = aps["C_gen_fog"] - aps["A_clear"]
        floor = aps["D_gen_shift"] - aps["A_clear"]
        print("\n=== ★ 判据 ===")
        print(f"  Δ 真值退化 (B−A) = {d_truth:+.4f}")
        print(f"  Δ 生成退化 (C−A) = {d_gen:+.4f}")
        print(f"  对照臂 (D−A)     = {floor:+.4f}   ← **地板**:生成图与 GT 完全对不上时该有的读数")
        verdict = _verdict(d_truth, d_gen, floor)
        print(f"  ⇒ {verdict}")
        rl.highlight("delta_truth", round(d_truth, 4))
        rl.highlight("delta_gen", round(d_gen, 4))
        rl.highlight("delta_control", round(floor, 4))
        rl.highlight("verdict", verdict)

        summ = work / "downstream_summary.json"
        summ.write_text(
            json.dumps(
                {
                    "args": {k: str(v) for k, v in vars(args).items()},
                    "conf": conf,
                    "mAP": {k: round(v, 4) for k, v in aps.items()},
                    "delta_truth": round(d_truth, 4),
                    "delta_gen": round(d_gen, 4),
                    "delta_control": round(floor, 4),
                    "arms": arms,
                    "verdict": verdict,
                },
                indent=1,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        rl.artifact(summ, "summary")
        print(f"[done] {summ}")


def _verdict(d_truth: float, d_gen: float, floor: float, *, tol: float = 0.05) -> str:
    """三档裁决 + **两头都能分辨**。

    ⚠️ 用**地板**当参照:若 `Δ_gen` 与地板不可分辨 ⇒ 生成图与 GT 对不上,
    那时"掉得多"不是"退化重",是**布局丢了** —— 这条区分正是 D 臂存在的理由。
    """
    if d_gen <= floor + tol:
        return "★ 生成图与 GT 对不上(Δ_gen 与对照地板不可分辨)⇒ 布局没保住,不能当退化数据"
    if d_gen > -tol:
        return "★ 假退化:生成的退化几乎不掉点 ⇒ 没真的加雾"
    if abs(d_gen - d_truth) <= max(0.1 * abs(d_truth), tol):
        return "★★ 同量级 ⇒ 生成的退化**可替代**真值退化(JD 要的答案)"
    if d_gen < d_truth:
        return "★ 生成图**过毒**:比真值退化掉得更多 ⇒ 不能直接当训练数据"
    return "生成退化比真值退化轻 ⇒ 退化偏弱"


if __name__ == "__main__":
    main()

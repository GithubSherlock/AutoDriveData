"""把已落盘的 KITTI root 按**当前** GT 口径重筛 `label_2`(口径迁移,不改像素)。

## 为什么是"重筛"而不是"重采"

2026-10-01 给 `gt/core.py` 加了 `MIN_BOX_SIDE_PX`(退化投影剔除,见 §P-V12):车**擦过镜头**
时投出零面积框,它与任何预测的 IoU 恒为 0,是白送的漏检(P1 每份数据 11/194)。这条修在
**出框侧**,只对**新采**的数据生效 —— 磁盘上那批按旧口径写下的框得单独处理。

**重采能修,但结果是投骰子**:退化框的进出由**亚帧抖动**决定(ego 偏 0.24 m 就让整框在
「0 px 高」与「164 px 高」之间翻面),每个 root 有 3 台"被擦过的车",5 个 root 两两比对
共 12 个阈值穿越点 ⇒ 重采后**几乎必然**有一对在某一帧上条数不等(而现有数据里
day_clear 对 dense_fog / rain_night / sunset_glare 三对**已经是 0 帧差**)。

**重筛只动 GT 一个量**:像素 / 点云 / calib / pose 逐字节不变 ⇒ 已经验证过的帧级配对
原样保留。判据与采集器**同源**(`gt.core.is_degenerate_gt_line`),所以"筛出来的"
就是"重采一份会得到的"。

## 用法

    python -m autodrivedata.gt.refilter --root outputs/kitti_ab_epic_day_clear \\
        --out outputs/kitti_ab_epic_gtfix_day_clear

**只写新 root,不就地改**:旧 root 是"旧口径"的物证,删不删由人裁 —— 而这条命令
几乎不占盘(见下),没有非就地不可的理由。

`--out` 走 **硬链接**:只有 `label_2/` 是重写的,**其余产物一个字节都不复制**(5 个 root
若真拷就是 1 GB)。硬链接对只读消费方与真拷贝无法区分。

⚠️ **旁路产物按谓词全量硬链,不列清单** —— 原实现写死 `("image_2","velodyne","calib","pose")`,
**漏了 `samples/RADAR_*`**,于是五份 `gtfix` root 的雷达静默消失(重筛前 70 帧、重筛后 0)。
"只动 label_2"是这条工具的定义,写成 `is_rewritten()` 而不是一份会过期的名单。

## `--clip2d`:同一个工具的**第二个修正**(2026-10-02)

退化剔除修的是"**该删的没删**";裁断口径修的是"**留下来的框画小了**"。两条都是口径,
所以**同一条命令一次做完** —— 分两次跑会得到一个"剔了退化但框仍偏小"的中间代 root,
而它与前后两代都不可比(红线:**跨口径不可混比**),白白多一代垃圾。

    python -m autodrivedata.gt.refilter --root outputs/kitti_ab_epic_day_clear \\
        --out outputs/kitti_ab_epic_clip2d_day_clear --clip2d

**重算只需两个本就在 root 里的量**:`label_2` 每行自带 `h w l x y z ry`,`calib/{fid}.txt`
自带 `P2` ⇒ 画幅从 `image_2/{fid}.png` 的 **IHDR 头**(前 24 字节)读,**不引 PIL**。
⚠️ 缺 calib 或缺图**直接抛**,不回落旧口径 —— 一份 root 里混两套口径,是这里最坏的结果
(逐帧条数照对、A/B 硬门槛照过,只是框偏,而偏多少随 `ry` 变)。
"""

from __future__ import annotations

import argparse
import errno
import os
import shutil
import struct
from pathlib import Path

import numpy as np

from autodrivedata.gt.core import LABEL_2_FIELDS, MIN_BOX_SIDE_PX, is_degenerate_gt_line, rebox_line
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path


def png_size(path: Path) -> tuple[int, int]:
    """PNG 的 `(宽, 高)` —— 只读 IHDR(前 24 字节),**不引 PIL**。

    存在理由:`gt` 是 `_PURE` 层,画幅这一件事不该为它拉一个图像库进来;而 PNG 头是
    规范钉死的定长结构(`\\x89PNG\\r\\n\\x1a\\n` + 长度 + `IHDR` + 两个大端 uint32)。
    """
    with path.open("rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"不是 PNG(或文件截断):{path}")
    w, h = struct.unpack(">II", head[16:24])
    return int(w), int(h)


def load_p2(path: Path) -> np.ndarray:
    """`calib/{fid}.txt` → `P2` 3×4。

    与 `perception/eval_fusion.load_calib` **不是重复**:那边要的是
    `(P2, R0_rect @ Tr_velo_to_cam)` 做点云投影,这里只要 `P2` 一个;而 `gt` 是 `_PURE`
    层(禁 torch),反向依赖 `perception` 会把 torch 拖进 GT 通道。
    """
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("P2:"):
            return np.array([float(x) for x in ln.split(":", 1)[1].split()]).reshape(3, 4)
    raise ValueError(f"calib 里没有 P2:{path}")


def _zero_counts() -> dict[str, int]:
    """逐帧计数的键集合。**一处定义**,`refilter_root` 的汇总与 `main` 的打印都从它取 ——
    加一个计数项时只改这里,不会出现"算了但没报"(那个数就等于不存在)。"""
    return {"in": 0, "kept": 0, "dropped": 0, "malformed": 0, "reboxed": 0}


def rebox_label_file(lab: Path, calib_dir: Path, image_dir: Path, *, clip2d: bool) -> tuple[list[str], dict]:
    """一帧的 `label_2` → 新行列表 + 计数。**纯函数**(不写盘),便于分别钉两个开关。"""
    lines = [ln for ln in lab.read_text(encoding="utf-8").splitlines() if ln.strip()]
    fid = lab.stem
    p2 = size = None
    if clip2d:
        # ⚠️ 缺件**抛**,不回落旧口径 —— 混口径的 root 比缺帧更毒(见模块头注)。
        p2 = load_p2(calib_dir / f"{fid}.txt")
        size = png_size(image_dir / f"{fid}.png")

    kept: list[str] = []
    c = _zero_counts()
    for ln in lines:
        c["in"] += 1
        if len(ln.split()) < LABEL_2_FIELDS:
            # 与 `is_degenerate_gt_line` 同一句话(它只管到 8 列),但这里要走到 `rebox_line`,
            # 残缺行会**抛**而不是返回 None —— 先按字段数挡掉,免得一帧坏行终止整批迁移。
            c["malformed"] += 1
            continue
        if is_degenerate_gt_line(ln):
            c["dropped"] += 1
            continue
        if clip2d:
            assert p2 is not None and size is not None
            new = rebox_line(ln, p2, size[0], size[1])
            if new is None:
                # 几何判据比"印出来的两位小数"更严:实测有 0.996 px 那种行,打印成 1.00
                # 骗过了文本判据。**以几何为准**(它才是出框侧的那句口径),并单独计数。
                c["dropped"] += 1
                continue
            c["reboxed"] += new != ln
            kept.append(new)
        else:
            kept.append(ln)
        c["kept"] += 1
    return kept, c


def is_rewritten(rel: Path) -> bool:
    """该相对路径是否落在**被重写**的目录里 —— 重筛**只动 `label_2/`**,其余一律原样硬链。

    ★ 这里原来是写死的目录清单 `("image_2", "velodyne", "calib", "pose")`,**漏了
    `samples/RADAR_*`** ⇒ 五份 `kitti_ab_epic_gtfix_*` 的雷达**静默消失**(2026-10-01
    实测:重筛前 70 帧雷达 pcd、重筛后 0)。数据照出、帧号照对,只是**少了一路**。
    改成**谓词**而不是顺着清单再补一项:名单是会过期的产物,而"只动 label_2"是这条
    工具的定义 —— 定义写成代码,下次加通道时不必记得回来改这里。
    """
    return rel.parts[:2] == ("training", "label_2")


def refilter_root(root: Path, out: Path, *, clip2d: bool = False) -> dict[str, int]:
    """`root` → `out`,只重写 `label_2/`。返回计数(纯副作用函数,判据在调用方)。

    `clip2d=True` 时**一次做完两个修正**(退化剔除 + 裁断口径重算,见模块头注);
    `False` 时行为与 2026-10-01 那版**逐字节相同**(默认值即旧口径)。
    """
    src = root / "training"
    dst = out / "training"
    if out.exists() and any(dst.rglob("*.txt")):
        raise FileExistsError(f"{out} 已有内容 —— 重筛不覆盖既有产物,换个 --out 或先删")
    dst.mkdir(parents=True, exist_ok=True)

    totals = _zero_counts()
    labels = sorted((src / "label_2").glob("*.txt"))
    (dst / "label_2").mkdir(parents=True, exist_ok=True)
    for lab in labels:
        kept, c = rebox_label_file(lab, src / "calib", src / "image_2", clip2d=clip2d)
        for k in totals:
            totals[k] += c[k]
        (dst / "label_2" / lab.name).write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    # ⚠️ 走**整个 root** 而不是 `training/` —— `samples/RADAR_*` 是 `training/` 的**兄弟**,
    #    只扫 `training/` 会让雷达照样漏掉(第一版改完就是这样,被本文件的测试当场抓住)。
    n_linked = n_copied = 0
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(root)
        if is_rewritten(rel):
            continue  # training/label_2/ 已在上面重写
        d = out / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(f, d)
            n_linked += 1
        except OSError as e:
            # ★ **跨设备**(`--out` 在另一个盘/`/tmp`)时硬链不可能 —— 退化成拷贝,但
            #   **分开计数**:`linked` 与 `copied` 是两个不同的性质(前者零占盘、改一个
            #   另一个跟着变),混成一个数就分不出来了。原先这里直接抛 `EXDEV`,
            #   而报错信息是"Invalid cross-device link",看不出"该怎么办"。
            if e.errno != errno.EXDEV:
                raise
            shutil.copy2(f, d)
            n_copied += 1

    return {
        "frames": len(labels),
        "gt_in": totals["in"],
        "gt_kept": totals["kept"],
        "gt_dropped": totals["dropped"],
        # `--clip2d` 专属:被画幅裁断、**真的重画过**的行数(未裁断的走短路、逐字节不变)。
        # 与 `gt_dropped` 分开报 —— 一个是"删了",一个是"改了",混在一起看不出这轮做了哪种。
        "gt_malformed": totals["malformed"],
        "gt_reboxed": totals["reboxed"],
        # 旁路产物**逐文件**计数:少了一路时这个数会变小(而不是静默)。
        # `linked` 与 `copied` 分开报 —— 零占盘 vs 占盘,是两回事。
        "linked": n_linked,
        "copied": n_copied,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="按当前 GT 口径重筛 label_2(见模块头注)")
    ap.add_argument("--root", required=True, help="源 KITTI root")
    ap.add_argument("--out", required=True, help="目标 root(非 label 目录走硬链接)")
    ap.add_argument(
        "--clip2d",
        action="store_true",
        help="同时按「全角点 min/max 再钳画幅」重算 2D 列(需 calib/ 与 image_2/,默认关=旧口径)",
    )
    args = ap.parse_args()

    root = project_path(args.root)
    with runlog.run("autodrivedata.gt.refilter") as rl:
        rl.input(str(root), "root")
        rl.highlight("min_box_side_px", MIN_BOX_SIDE_PX)
        rl.highlight("clip2d", args.clip2d)
        out = project_path(args.out)
        stats = refilter_root(root, out, clip2d=args.clip2d)
        print(
            f"[refilter] {stats['gt_in']} → {stats['gt_kept']} 条"
            f"(剔除 {stats['gt_dropped']} 退化 / {stats['gt_malformed']} 残缺)"
        )
        if args.clip2d:
            print(f"[refilter] 其中 {stats['gt_reboxed']} 条被画幅裁断,已按新口径重画 2D 框")
        print(f"[refilter] 帧数 {stats['frames']}  |  {out.resolve()}")
        for k, v in stats.items():
            rl.highlight(k, v)
        rl.highlight("out", str(out))


if __name__ == "__main__":
    main()

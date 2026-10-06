"""**人工注入**退化:运动模糊 / 高斯噪声 / 雨痕。**纯值**:禁 carla / torch。

## 为什么这一路是"人工注入"而不是"生成"

本仓红线:「CARLA **无运动模糊**(退化只能人工注入)」——
即平台本身不建模这一类退化,**只能我们造**。而"传感器噪声"正是 JD 列的四类变异之一。

## 它在下游闭环里的位置

`downstream_eval` 量的是"**生成的**退化 vs **真值的**退化"。
这一路补的是第三个量:**强度可精确设定的**退化 —— 它是唯一能把
"**退化强度 → 下游掉点**"这条曲线画出来的东西(生成那一侧强度不可控)。

⇒ 判据:**注入曲线**给出一条"强度 → ΔAP"的标定;把生成退化与真值退化**投影到这条曲线上**,
看它们落在哪一档。落不上曲线 ⇒ 生成/真值的退化**不是同一种东西**。

⚠️ **注入必须在**原图尺寸上做(不是先减分辨率),且**同一个 seed 逐帧固定**
—— 否则"两帧之间的差"里混着注入本身的随机性。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 三种注入。键 → (说明, 默认强度列表)。**强度是单调可比的量**(模糊核 / 噪声 σ / 雨痕条数)。
KINDS = ("blur", "noise", "rain", "fog", "fogdepth")

#: 默认扫描档。**不是"挑一个好看的"**,是要覆盖到"明显掉点"为止 ——
#: 曲线的用途是给另外两种退化当标尺,截在中途就标不了。
DEF_LEVELS = {
    "blur": [3, 7, 15],
    "noise": [10, 25, 50],
    "rain": [200, 800, 2000],
    # `fog` 的 level = **遮蔽系数** k(`0` = 原图,`1` = 全白)。见 `veil_fog`。
    "fog": [0.2, 0.4, 0.6],
    # ★ `fogdepth` 的 level = **消光系数 β**(1/m)。**需要 `training/depth/`**。见 `atmospheric_fog`。
    "fogdepth": [0.02, 0.05, 0.10],
}


def motion_blur(img: np.ndarray, k: int) -> np.ndarray:
    """水平方向**线性运动模糊**,核长 `k`(像素)。`k <= 1` 原样返回。

    用 `cv2.filter2D` 而不是"逐像素平均"—— 后者是 O(k) 循环,慢且边界口径不好定。
    """
    import cv2

    k = int(k)
    if k <= 1:
        return img
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k  # 一条横线 = 水平运动
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def gaussian_noise(img: np.ndarray, sigma: float, *, seed: int = 0) -> np.ndarray:
    """加性高斯噪声(σ 为 0–255 尺度)。`seed` **逐帧固定** ⇒ 重跑逐位相同。"""
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, float(sigma), size=img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def rain_streaks(img: np.ndarray, n: int, *, seed: int = 0) -> np.ndarray:
    """画 `n` 条**斜向雨痕**(亮线 + 轻微模糊尾)。

    ⚠️ 雨痕是**画上去的**,不是物理合成的 —— 它改变的是"镜头上有水"这一类退化,
    与 CARLA 的 `rain`(它改的是环境光照)不是一回事。报数时必须写明。
    """
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    rng = np.random.default_rng(seed)
    for _ in range(int(n)):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        ln = int(rng.integers(8, 28))
        dx, dy = int(rng.integers(1, 4)), ln  # 略斜
        c = int(rng.integers(120, 220))
        cv2.line(out, (x, y), (min(x + dx, w - 1), min(y + dy, h - 1)), (c, c, c), 1, cv2.LINE_AA)
    return out


def veil_fog(img: np.ndarray, k: float, *, airlight: int = 230) -> np.ndarray:
    """**合成雾 = 大气光遮蔽**:`I' = (1−k)·I + k·A`(A = 大气光,默认 230)。

    ## 为什么用这个模型,而不是完整的 `I = J·t + A·(1−t)`

    完整的散射模型要**逐像素透射率 `t = exp(−β·d)`**,而 **`d` 正是我们没有的**
    (生成图没有深度;真值图在这条链上也没带 `depth/`)。用全局常数遮蔽 =
    **只保留大气光那一项、丢掉距离项**,是最简的"看得见的雾"。

    ⚠️ **代价必须写明**:它**不产生真雾那种"远处先糊"的梯度**,只是整体泛白。
    ⇒ 它是"**同一种退化类型的最简形式**",不是"真雾的等价物"。
    它的用处正是**当一个纯可调的对照**:对生成图与真值图**施加同一个模型**,
    两边退化强度就**逐位可比**。
    """
    k = float(np.clip(k, 0.0, 1.0))
    a = np.asarray(img, np.float32)
    return np.clip(a * (1.0 - k) + float(airlight) * k, 0, 255).astype(np.uint8)


def atmospheric_fog(img: np.ndarray, depth_m: np.ndarray, beta: float, *, airlight: int = 230) -> np.ndarray:
    """**距离相关的真雾模型**:`I' = J·t + A·(1−t)`,`t = exp(−β·d)`。

    ## 为什么必须有这一路(2026-10-06 实测的机制)

    全局遮蔽(`veil_fog`)把亮度做到 **186**(真值浓雾 196)**依然一点都不掉点**(Δ−0.002);
    而 CARLA 的真雾掉 **0.186**。⇒ **杀伤力不在"有多白",在"退化是否随距离增长"** ——
    真雾按 `exp(−β·d)` 让 **40–60 m 的目标先没**,而全局遮蔽对所有距离一视同仁。

    ★ 而 **`d` 正是本项目的真值深度** —— 这一路是"用本项目独有件造退化"的落点。

    ⚠️ `beta` 是**消光系数**(1/m),不是无量纲强度:`β=0.05` 时 40 m 处 `t=0.14`(几乎全遮)、
    10 m 处 `t=0.61`。**报 β 必须连同"多少米处 t 是多少"一起报**。
    """
    d = np.asarray(depth_m, np.float32)
    if d.shape[:2] != np.asarray(img).shape[:2]:
        raise ValueError(f"深度与图不同画幅:{d.shape} vs {np.asarray(img).shape}")
    d = np.nan_to_num(d, nan=0.0, posinf=1e4, neginf=0.0)
    t = np.exp(-float(beta) * np.clip(d, 0.0, None))[..., None]
    a = np.asarray(img, np.float32)
    return np.clip(a * t + float(airlight) * (1.0 - t), 0, 255).astype(np.uint8)


INJECTORS = {
    "blur": motion_blur,
    "noise": gaussian_noise,
    "rain": rain_streaks,
    "fog": veil_fog,
    # `fogdepth` 的签名多一个 `depth` —— 下面 `apply` 里单列,不进 `INJECTORS` 的统一签名
}


def apply_degradation(img: np.ndarray, kind: str, level: float, *, seed: int = 0) -> np.ndarray:
    """按 `kind` / `level` 注入。**`kind` 不认识就抛**(静默返回原图 = 假阴性)。"""
    if kind not in INJECTORS:
        raise KeyError(f"未知注入 {kind!r};可选 {list(KINDS)}")
    if kind == "blur":
        return motion_blur(img, int(level))
    if kind == "noise":
        return gaussian_noise(img, level, seed=seed)
    if kind == "fog":
        return veil_fog(img, level)
    if kind == "fogdepth":
        raise KeyError("`fogdepth` 需要逐帧深度 ⇒ 走 `degrade_root`(它知道 `training/depth/`)")
    return rain_streaks(img, int(level), seed=seed)


def degrade_root(src: Path, dst: Path, *, kind: str, level: float, frames: list[str] | None = None) -> int:
    """把一个 arm root 的 `training/image_2` 逐帧注入 → 新 root。GT 原样照搬。"""
    import shutil

    from PIL import Image

    imgs = sorted((src / "training/image_2").glob("*.png"))
    labs = sorted((src / "training/label_2").glob("*.txt"))
    if frames is not None:
        want = set(frames)
        imgs = [p for p in imgs if p.stem in want]
        labs = [p for p in labs if p.stem in want]
    n = len(imgs)
    if n == 0 or len(labs) != n:
        raise SystemExit(f"源 root 不规整:图 {len(imgs)} 张、GT {len(labs)} 条")
    (dst / "training/image_2").mkdir(parents=True, exist_ok=True)
    (dst / "training/label_2").mkdir(parents=True, exist_ok=True)
    if kind == "fogdepth" and not (src / "training/depth").is_dir():
        raise SystemExit(
            f"{kind} 需要逐帧真值深度,而 {src}/training/depth 不存在。\n"
            "  用 `collect_ab_route --depth` 采,或用 `kitti_square` 从带深度的 root 方裁过来。"
        )
    for i, (ip, lp) in enumerate(zip(imgs, labs, strict=True)):
        arr = np.array(Image.open(ip).convert("RGB"))
        if kind == "fogdepth":
            d = np.load(src / "training/depth" / f"{ip.stem}.npy")
            out = atmospheric_fog(arr, d, level)
        else:
            out = apply_degradation(arr, kind, level, seed=1000 + i)  # ★ seed 逐帧固定
        Image.fromarray(out).save(dst / "training/image_2" / f"{i:06d}.png")
        shutil.copyfile(lp, dst / "training/label_2" / f"{i:06d}.txt")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--src", required=True, help="源 arm root(有 training/image_2 + label_2)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--level", type=float, required=True, help="强度:模糊核长 / 噪声 σ / 雨痕条数")
    ap.add_argument("--frames", default="", help="逗号分隔或 a-b;空 = 全部")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    src, dst = project_path(args.src), project_path(args.dst)
    frames = _parse_frames(args.frames) if args.frames else None
    with runlog.run("autodrivedata.edit.degrade") as rl:
        rl.input(src, "src-root")
        rl.highlight("kind", args.kind)
        rl.highlight("level", args.level)
        n = degrade_root(src, dst, kind=args.kind, level=args.level, frames=frames)
        print(f"[{args.kind}={args.level:g}] {n} 帧 → {dst}")
        rl.highlight("n_frames", n)
        rl.artifact(dst / "training/image_2", "degraded")


def _parse_frames(spec: str) -> list[str]:
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(f"{i:06d}" for i in range(int(a), int(b) + 1))
        elif part:
            out.append(f"{int(part):06d}")
    if not out:
        raise SystemExit(f"--frames 解析出空列表:{spec!r}")
    return out


if __name__ == "__main__":
    main()

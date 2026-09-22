"""CARLA 深度相机编解码(纯值,不 import carla)。

**单一来源**:此前 `bin/collect_stereo.py:38` 与 `bin/collect_3dgs.py:44` 各有一份逐字
重复的实现(公式相同、注释各自有误),本模块合并它们。

CARLA `sensor.camera.depth` 的编码:BGRA 四通道,深度值以 24 bit 定点的**归一化**
形式打包在 B/G/R 三个通道里,量程 1000 m:

    D[m] = (R + 256·G + 65536·B) / (256³ − 1) × 1000

CARLA 的 BGRA 布局里 `arr[:, :, 0]` 是 **B**(高位,×65536)、`[:, :, 1]` 是 G、
`[:, :, 2]` 是 **R**(低位)。实测锚点:`B=255` 其余 0 → **996.09 m**(近满量程)、
`R=255` 其余 0 → **0.0152 m**(近 0)—— 见 `tests/test_depth_codec.py`。

⚠️ 两处旧注释都写错了(`bin/collect_stereo.py:39` 写 `B + G·256 + R·256²`,
其行 13 又写 `/255`),而**代码是对的**;本模块以代码 + 上述实测锚点为准。

**深度语义 = 光轴 z 深度**(不是射线距离):`bin/train_3dgs_mini.py:122` 反投影时把
解码值**直接当相机系 z 用**(`cam_pts = [(px−W/2)·zv/f, (py−H/2)·zv/f, zv]`)。
⚠️ 该结论**尚未由本仓的独立探针复核**:此前注释引用的 `bin/probe_depth_semantics.py`
**并不存在于磁盘**(文档与代码不一致,已如实更正)。若要钉死,判据应是
"渲染深度 / LiDAR 预测光轴 z − 1 ≈ 0" vs "…/ 射线距离 − 1 ≈ sec θ − 1"
(θ 为离轴角,边缘可差 44%),见 `autodrivedata/calib_probe.py:41` 的同款推理;
`bin/probe_calib.py` 的 A3 锚已在**假定 z 深度**下给出 median|e| 0.0003 m,间接支持该语义。

量化精度:24 bit / 1000 m ≈ **0.06 mm**,可忽略。
"""

from __future__ import annotations

import numpy as np

# 深度量程(m):CARLA 归一化编码的满量程
DEPTH_RANGE_M = 1000.0
_DENOM = float(256**3 - 1)


def decode_depth(raw: bytes | np.ndarray, height: int, width: int) -> np.ndarray:
    """CARLA 深度 BGRA 原始缓冲 → 深度米 (H, W) float32。

    `raw` 可为 `carla.Image.raw_data`(bytes)或已 reshape 的 (H, W, 4) 数组。
    """
    if isinstance(raw, np.ndarray):
        arr = raw.reshape(height, width, 4)
    else:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    b = arr[:, :, 0].astype(np.float32)  # CARLA B 通道 = 高位(×65536)
    g = arr[:, :, 1].astype(np.float32)  # G 通道 = 中位
    r = arr[:, :, 2].astype(np.float32)  # CARLA R 通道 = 低位
    return (r + g * 256.0 + b * 65536.0) / _DENOM * DEPTH_RANGE_M


def encode_depth(depth_m: np.ndarray) -> np.ndarray:
    """深度米 → CARLA BGRA 编码 (H, W, 4) uint8(**解码的逆**,供单测往返)。

    alpha 通道固定 255(CARLA 亦如此);超出量程的值饱和到满量程。
    """
    d = np.clip(np.asarray(depth_m, dtype=np.float64), 0.0, DEPTH_RANGE_M) / DEPTH_RANGE_M * _DENOM
    q = np.round(d).astype(np.uint64)
    h, w = q.shape
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[:, :, 0] = ((q >> 16) & 0xFF).astype(np.uint8)  # B = 高位
    out[:, :, 1] = ((q >> 8) & 0xFF).astype(np.uint8)  # G = 中位
    out[:, :, 2] = (q & 0xFF).astype(np.uint8)  # R = 低位
    out[:, :, 3] = 255
    return out


# ---------------------------------------------------------------- 像素索引约定

# **本仓的 CARLA 渲染光栅实测是 `CORNER`**(索引 i 的连续坐标就是 i)——见
# `bin/probe_calib.py` 的 A3/A4 锚:corner 约定下 LiDAR-平面-深度图残差 median|e| 0.0003 m,
# center 约定 0.023 m(~70×,六相机一致);轴目标物掩膜**索引**中点回归给 cx = 620.50
# = `(w−1)/2`,而 center 约定给 621.00。
# `CENTER`(索引 i 覆盖 [i, i+1),中心在 i+0.5)是 **torch 侧**的约定
# (`grid_sample(align_corners=False)`、FPN 特征图、gsplat)——**不是 CARLA 的**。
# ⇒ 采样 CARLA 渲染出的图(深度/语义/实例分割/RGB)默认走 `CORNER`;
#   采样 torch 特征图时调用方显式传 `CENTER`(如 `maptr_impl/gkt.py` 的 grid 归一化)。
CONVENTION_CORNER = "corner"  # CARLA 光栅:索引 i 即连续坐标 i
CONVENTION_CENTER = "center"  # torch 光栅:索引 i 的中心在连续坐标 i + 0.5

# 两种约定的**唯一数值差异**:索引 → 连续坐标的平移量。标量/批量两条路都取它,
# 不存在第二处硬编码 0.5。
_CONVENTION_SHIFT = {CONVENTION_CENTER: 0.5, CONVENTION_CORNER: 0.0}


def _shift(convention: str) -> float:
    if convention not in _CONVENTION_SHIFT:
        raise ValueError(f"未知像素约定 {convention!r}(应为 {CONVENTION_CENTER}/{CONVENTION_CORNER})")
    return _CONVENTION_SHIFT[convention]


def convention_shift(convention: str = CONVENTION_CORNER) -> float:
    """该约定的索引平移量(corner 0.0 / center 0.5)。

    给**索引口径与图像坐标口径混用**的调用方取用:例如"渲染出的掩膜中点列(索引) →
    图像坐标主点"必须加回这个平移量,而不是在调用点再写一个 0.5。
    """
    return _shift(convention)


def index_to_continuous(i: float, convention: str = CONVENTION_CENTER) -> float:
    """像素**索引** → 连续坐标。**全仓唯一落点**(0.5 只在这里出现)。

    GPU 光栅化把索引 `i` 覆盖的连续区间定为 `[i, i+1)`,故其中心在 `i + 0.5`
    (与 `torch.nn.functional.grid_sample(align_corners=False)` 一致:索引 i ↔ 坐标
    i+0.5)。`corner` 是"索引即坐标"的口径 —— **CARLA 渲染光栅实测就是它**
    (见本段上方常量处的裁决);本函数默认仍取 `center`(它描述的是通用 torch 光栅),
    CARLA 侧调用方显式传 `corner`。
    """
    return float(i) + _shift(convention)


def continuous_to_index(u: float, convention: str = CONVENTION_CENTER) -> float:
    """连续坐标 → 像素**索引**(`index_to_continuous` 的逆)。"""
    return float(u) - _shift(convention)


def sample_bilinear(
    img: np.ndarray,
    u: float,
    v: float,
    convention: str = CONVENTION_CORNER,
) -> float:
    """在**连续坐标** (u, v) 处双线性采样 CARLA 渲染图。

    默认 `CORNER`(CARLA 光栅的实测口径);越界返回 NaN(调用方自行过滤)。
    **不要用整数索引读深度图**:那等于隐式钉死一个 0.5 px 约定,而本模块存在的意义
    正是把这个约定显式化。
    """
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape[:2]
    x = continuous_to_index(u, convention)
    y = continuous_to_index(v, convention)
    if not (0.0 <= x <= w - 1.0 and 0.0 <= y <= h - 1.0):
        return float("nan")
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    top = a[y0, x0] * (1.0 - fx) + a[y0, x1] * fx
    bot = a[y1, x0] * (1.0 - fx) + a[y1, x1] * fx
    return float(top * (1.0 - fy) + bot * fy)


def sample_bilinear_many(
    img: np.ndarray,
    uv: np.ndarray,
    convention: str = CONVENTION_CORNER,
) -> np.ndarray:
    """在**连续坐标** (N,2) 处批量双线性采样 CARLA 渲染图 → (N,) float64。

    `sample_bilinear` 的批量版(**同一套边界/插值口径**,不是另写一份):离线探针一次
    要采上万个点、实时槽每 tick 也要采,逐点 Python 循环是纯开销。越界返回 NaN。
    默认 `CORNER`(同 `sample_bilinear`)。
    """
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape[:2]
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        return np.zeros(0, dtype=np.float64)
    x = pts[:, 0] - _shift(convention)
    y = pts[:, 1] - _shift(convention)
    out = np.full(pts.shape[0], np.nan, dtype=np.float64)
    inside = (x >= 0.0) & (x <= w - 1.0) & (y >= 0.0) & (y <= h - 1.0)
    if not inside.any():
        return out
    xi, yi = x[inside], y[inside]
    x0 = np.floor(xi).astype(np.intp)
    y0 = np.floor(yi).astype(np.intp)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = xi - x0
    fy = yi - y0
    top = a[y0, x0] * (1.0 - fx) + a[y0, x1] * fx
    bot = a[y1, x0] * (1.0 - fx) + a[y1, x1] * fx
    out[inside] = top * (1.0 - fy) + bot * fy
    return out

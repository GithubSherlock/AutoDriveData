"""跨图拼接 —— 把几张图按**放置表**摆到同一坐标系里合并(纯值,不 import carla/torch)。

## 这件事的本质(先读这段,否则会误解产出)

**CARLA 各 Town 是独立 UE 关卡、原点任意,彼此没有真值相对位姿。** 所以拼官方 Town
**不是配准问题** —— 没有共享内容可对齐 —— 而是**人为摆位**。placement 表里填的就是人的决定,
不是算出来的。只有拼**你自己设计的分段图**时,接缝处才有真值(车道口本就该对齐)。

由此决定本模块的边界:
- **几何层**(`place` + `stitch`):两种场景都适用,是"把几坨几何摆到一张图上并去重"。
- **拓扑层**(`seams`):**只对"本来就该相接"的图有意义**。而且注意 [`MapVec`](mapvec.py)
  **不携带道路图**(没有 `predecessor` / `successor` / junction —— 那些在 `opendrive.Road` 上,
  提取时没带出来)。所以本模块给的是**接缝候选报告**(跨图的端点配对 + 朝向差),
  **不是**把 link 改写上去。要真接拓扑得先把道路图带进 MapVec,那是另一件事。

## 坐标载体:纯平面米系,**不走 WGS84**

拼接是摆位问题,引入经纬度只会再叠一层等距圆柱近似误差(该近似在离 origin 几十公里处
已不可忽略,而高程跨度实测可到 **791 m**,摆开后平面跨度也可能很大)。
要对外交付再在最后一步过 [`lanelet2`](lanelet2.py) 的投影。

## 变换定义(唯一口径,别在别处再写一遍)

```text
x' = dx + x·cosθ − y·sinθ
y' = dy + x·sinθ + y·cosθ
z' = dz + z                    θ = radians(yaw_deg)
```

即 (x, y) 平面上的标准 2D 旋转。CARLA 世界系是**左手系**(xodr y 取反而来),
所以若你的直觉是"往右转",这里的正 yaw 可能是反过来 —— **以公式为准**,不要靠感觉。

## ⚠️ 合并图的用途边界

**对 MapTR 训练基本无用**:训练仍是逐帧 `BEV_RANGE`(±15/±30)裁剪,合并图超窗口。
它的价值在**全图质检 / 可视化 / 对外交付**。别把它当训练数据源。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mapvec import MapVec

# 去重的空间哈希格子边长(米):先按质心分桶,只在桶内两两比,避免 O(n²)
DEDUP_CELL = 5.0
# 两条同类折线互距 ≤ 此值即视为重复(米)
DEDUP_TOL = 0.05
# 单个桶内的配对数上限:超了就**只报不删**(宁可留下可疑的重复,也不做 O(k²) 卡死)
_BUCKET_PAIR_CAP = 200_000


@dataclass(frozen=True)
class Placement:
    """一张图在合并系里的位姿(见模块头注的变换定义)。"""

    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    yaw_deg: float = 0.0

    @property
    def is_identity(self) -> bool:
        return self.dx == 0.0 and self.dy == 0.0 and self.dz == 0.0 and self.yaw_deg == 0.0

    def to_dict(self) -> list[float]:
        return [self.dx, self.dy, self.dz, self.yaw_deg]


def parse_placement(text: str) -> Placement:
    """`"dx,dy"` / `"dx,dy,dz"` / `"dx,dy,dz,yaw"` → `Placement`(缺省补 0)。"""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not 1 <= len(parts) <= 4:
        raise ValueError(f"placement 需要 1–4 个数(dx,dy[,dz[,yaw]]),收到 {text!r}")
    vals = [float(p) for p in parts] + [0.0] * (4 - len(parts))
    return Placement(*vals)


def parse_stitch_spec(spec: str) -> list[tuple[str, Placement]]:
    """`"A=0,0,0,0;B=420,0,0,90"` 或一个 JSON 文件路径 → `[(图名, Placement)]`。

    逗号在 placement 内部用,故条目之间只能以 `;` 分隔(**不要用逗号分隔条目** ——
    `A=0,0;B=1,1` 拆不开)。
    """
    if spec.endswith(".json") and Path(spec).exists():
        raw = json.loads(Path(spec).read_text(encoding="utf-8"))
        return [(str(k), parse_placement(",".join(str(x) for x in v))) for k, v in raw.items()]
    out: list[tuple[str, Placement]] = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        name, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"条目 {item!r} 缺 '='(形如 NAME=dx,dy,dz,yaw)")
        out.append((name.strip(), parse_placement(val)))
    if not out:
        raise ValueError(f"--stitch 没解析出任何图:{spec!r}")
    return out


def place(vecs: tuple[MapVec, ...], p: Placement) -> tuple[MapVec, ...]:
    """按 `p` 做刚体变换(见模块头注的公式)。

    **恒等 placement 直接原样返回**(不复制、不改一个 bit) —— 这是"拼接不改变单图"的
    机械保证:恒等档下产物必须与不拼接时**逐位相同**。
    """
    if p.is_identity:
        return vecs
    th = math.radians(p.yaw_deg)
    c, s = math.cos(th), math.sin(th)
    out = []
    for v in vecs:
        pts = tuple((p.dx + x * c - y * s, p.dy + x * s + y * c, p.dz + z) for x, y, z in v.points)
        out.append(v.with_points(pts))
    return tuple(out)


def _polyline_gap(
    a: tuple[tuple[float, float, float], ...], b: tuple[tuple[float, float, float], ...]
) -> float:
    """两条折线的**互距**(Hausdorff):双向「每个点到对方最近点」的最大值。

    比"首末点距离"稳(采样相位不同也能认出同一段线),比 Chamfer 严(不会因为一长一短
    就给出小值)。
    """
    pa = np.asarray(a, dtype=np.float64)[:, :2]
    pb = np.asarray(b, dtype=np.float64)[:, :2]
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def _cell_of(v: MapVec) -> tuple[str, int, int]:
    """元素质心所在的哈希格(**键里带 `cls`**)。

    两点都必须:**类参与键** —— 重叠路段的 divider 与 boundary 贴在一起是常态,
    不分类会把不同要素当重复删掉(实测被测试当场抓住)。**必须探邻域**
    ([[`_dedup`]] 里探 3×3):重合的两条线若质心恰好落在格的分界两侧,只看本格会比不到。
    """
    xs = [p[0] for p in v.points]
    ys = [p[1] for p in v.points]
    n = len(v.points)
    return v.cls, int(sum(xs) / n / DEDUP_CELL), int(sum(ys) / n / DEDUP_CELL)


def _dedup(vecs: tuple[MapVec, ...], origins: list[int], tol: float) -> tuple[tuple[MapVec, ...], int, int]:
    """**跨图**同类且折线互距 ≤ `tol` 的视为重复,只留先出现的那条。

    返回 (去重后, 删掉数, 跳过数)。

    **★ 只在「来源图不同」之间比 —— 同图内部一律不去重。** 这条是实测逼出来的:
    在 `Town10HD_Opt` 上开去重会删掉 10 条,全是**不同 road 的中心线恰好重合**
    (`road 90` vs `road 89`,互距 0.0000 m)—— 那是**真实的道路结构**(分隔带两侧、
    被拆成多个 road id 的同一条路),不是重复。单张图自身是自洽的,重复只可能来自
    **两张图在交叠区各导了一遍同一段线**;把同图内的重合也当重复 = 静默删真实要素。
    """
    grid: dict[tuple[int, int], list[int]] = {}  # **只装保留下来的**元素
    dropped: set[int] = set()
    skipped = 0
    for i, v in enumerate(vecs):
        if not v.points:
            continue  # 空几何没法比;**单点要素(traffic_light)必须参与** —— 实测漏掉它们
            # 会让"同一张图复制两份同位"残留 36/495 条(正是信号灯的条数)
        cls, cx, cy = _cell_of(v)
        hit = False
        candidates = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for k in grid.get((cls, cx + dx, cy + dy), ()):
                    candidates += 1
                    if candidates > _BUCKET_PAIR_CAP:
                        break
                    if origins[k] != origins[i] and _polyline_gap(v.points, vecs[k].points) <= tol:
                        hit = True
                        break
                if hit or candidates > _BUCKET_PAIR_CAP:
                    break
            if hit or candidates > _BUCKET_PAIR_CAP:
                break
        if candidates > _BUCKET_PAIR_CAP:
            skipped += 1  # 邻域内候选过多:只报不删,不硬撑
        if hit:
            dropped.add(i)
        else:
            grid.setdefault((cls, cx, cy), []).append(i)
    kept = tuple(v for i, v in enumerate(vecs) if i not in dropped)
    return kept, len(dropped), skipped


def stitch(
    entries: list[tuple[str, tuple[MapVec, ...], Placement]],
    tol: float = DEDUP_TOL,
    prefix_ids: bool = True,
) -> tuple[tuple[MapVec, ...], dict]:
    """各图 place 后求并集 → id 加图名前缀 → 重叠去重。返回 (矢量, 统计)。

    **多图时必须加图名前缀**:不同图都会产出 `divider_00001` 这种 id,不加前缀拼起来后会
    假成"同一条"(去重、对账都会错)。**单图不加**(没有撞车风险,加前缀纯属噪声),
    这样"单图 + 恒等 placement ⇒ 产物逐位不变"才成立。

    统计字典:`n_in` / `n_out` / `n_dedup` / `n_bucket_skipped` / `per_map` —— 去重与跳过
    **都要报**,静默少几条是这类工具最难发现的一类错。
    """
    placed: list[MapVec] = []
    origins: list[int] = []
    per_map: dict[str, int] = {}
    for mi, (name, vecs, p) in enumerate(entries):
        got = place(vecs, p)
        per_map[name] = len(got)
        if prefix_ids and len(entries) > 1:
            got = tuple(v.with_points(v.points, id_suffix=f"@{name}") for v in got)
        placed.extend(got)
        origins.extend([mi] * len(got))
    merged, n_dedup, n_skipped = _dedup(tuple(placed), origins, tol)
    stats = {
        "n_in": len(placed),
        "n_out": len(merged),
        "n_dedup": n_dedup,
        "n_bucket_skipped": n_skipped,
        "per_map": per_map,
        "tol": tol,
    }
    return merged, stats


# ---------- 接缝候选(拓扑层的**报告**,不是改写) ----------


@dataclass(frozen=True)
class Seam:
    """跨图的一对近邻端点:可能是"本该接上的车道口"。"""

    cls_a: str
    id_a: str
    cls_b: str
    id_b: str
    dist: float
    heading_diff_deg: float


def _heading(p0, p1) -> float:
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))


def seams(
    entries: list[tuple[str, tuple[MapVec, ...], Placement]],
    tol: float = 1.0,
    heading_tol_deg: float = 30.0,
) -> list[Seam]:
    """找出**不同图之间**距离 ≤ `tol` 且朝向差 ≤ `heading_tol_deg` 的端点对。

    **这是报告,不是拓扑拼接**:`MapVec` 不带道路图(没有 predecessor/successor),
    没法把 link 接上去(见模块头注)。此函数回答的是"哪些口子看起来该接、接得上吗",
    供人判断 —— 别把它当成已经接好了。

    **只在"本来就该相接的图"之间用**(你自己设计的分段图)。官方 Town 之间摆多近都会
    冒出一堆候选,那是噪声不是信号。
    """
    ends: list[tuple[str, MapVec, bool]] = []
    for name, vecs, p in entries:
        for v in place(vecs, p):
            if len(v.points) >= 2:
                ends.append((name, v, False))
                ends.append((name, v, True))
    # 按端点位置分桶(格边长 = tol):只跟同桶与相邻桶比 —— 直连 O(n²) 在真图上(数万条)
    # 跑不动,而接缝本来就是**局部**现象,分桶不丢候选。
    grid: dict[tuple[int, int], list[int]] = {}
    for i, (_n, v, rev) in enumerate(ends):
        q = v.points[-1] if rev else v.points[0]
        grid.setdefault((int(q[0] // tol), int(q[1] // tol)), []).append(i)

    def _pairs(i: int):
        _n, v, rev = ends[i]
        q = v.points[-1] if rev else v.points[0]
        cx, cy = int(q[0] // tol), int(q[1] // tol)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), ()):
                    if j > i:
                        yield j

    out: list[Seam] = []
    for i, (na, va, ra) in enumerate(ends):
        pa = va.points[-1] if ra else va.points[0]
        ha = _heading(va.points[-2 if ra else 1], pa)
        for j in _pairs(i):
            nb, vb, rb = ends[j]
            if na == nb:
                continue  # 同图内部不算接缝
            pb = vb.points[-1] if rb else vb.points[0]
            dist = math.dist(pa[:2], pb[:2])
            if dist > tol:
                continue
            hb = _heading(vb.points[-2 if rb else 1], pb)
            diff = abs((ha - hb + 180.0) % 360.0 - 180.0)
            if diff <= heading_tol_deg:
                out.append(Seam(va.cls, va.id, vb.cls, vb.id, dist, diff))
    out.sort(key=lambda s: s.dist)
    return out

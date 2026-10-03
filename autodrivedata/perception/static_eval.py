"""静态 GT(P2:信号/标志 + 车道线)的判据 —— **投影进语义图,看渲染里有没有**(离线纯值)。

## 这条补的是"静态 GT 只有真值源、没有打分模块"那个软处

`static_gt/` 从 2026-09-09 起就只出真值:信号锚点来自 OpenDRIVE 的 landmark、车道线
来自 `waypoint.lane_marking`。**它是地图事实的转写,没有东西能证伪它** —— 地图里写的位置
与 CARLA 真渲染出来的东西对不上时,GT 照样一份份地出,数字照样好看。
(同 §P-V14 道具的处境,那次的解法是**同挂一台实例/语义相机当 oracle**;这里照搬。)

2026-10-02 起 `collect_static_gt --sem` 同挂一台语义相机 → `training/static_sem/{fid}.png`,
并在 `static_gt/{fid}.json` 里落**实读**的相机位姿(见 `StaticFrame.camera` 头注 —— 判据
住在 `perception/`,层规则**禁 carla**,拿不到 `CAM_ATTRS`/`SENSOR_OFFSET`)。

## ★ 判据形状是**量出来的**,不是设计的(2026-10-02 实测 40 帧)

第一版想当然的形状是"锚点那个像素的 tag 是不是 TrafficLight" —— **全红**,40/40 个信号
锚点投出来都落在 `Roads`(1) 上。原因是 landmark 的位置是**地面锚点**(灯杆与地面的
交点),而灯头在它上方 5 m 处。同理车道线用单像素采样只有 21.7%(实线)/27.8%(虚线)
命中 —— 渲染出来的线只有 ~2 px 宽,采样点差一两个像素就落空。

于是两条判据都改成「**锚定 + 邻域**」,邻域大小按实测定:

| 判据 | 形状 | 真命中 | **横移对照** | **同 v 随机基准率** |
|---|---|---|---|---|
| 车道线 | 采样点 ±`LANE_WINDOW_PX` 窗内有 `RoadLines` | **0.996** | 0.039(横移半车道宽) | 0.139 |
| 信号/标志 | 锚点上方 `SIGNAL_WINDOW_M` 米的**竖带**内有期望 tag | **1.000** | 0.000(左右各移 40 px) | 见运行报告 |

**窗宽怎么定的**(不是拍的):
- 车道线 ±4 px 是**扫描出来的拐点** —— ±1 → 0.696、±2 → 0.887、±4 → **0.996**、再大
  只是把基准率一起抬上去;
- 信号 6 m 是**几何量出来的** —— 实测期望 tag 的最高像素离锚点中位 **5.1 m**、max **5.2 m**,
  取 6 m 覆盖 100% 且留 0.8 m 余量。⚠️ 窗放到 8 m 以上时横移对照就爬到 0.225–0.30
  (抬高到树/天那一带会撞上邻近杆上的别的灯),**判别力反而塌**。
- 竖带半宽 1 px(3 px 宽):把 6 m 窗下 0.950 补到 **1.000**,而对照仍是 0.000。

## 三个数(每类都给,缺一个都读不出结论)

| 量 | 含义 |
|---|---|
| `real` | 真采样点/真锚点的命中率 —— **xodr 与渲染对不对得上** |
| `shift` | 横移对照(车道线移半车道宽 / 信号移 ±40 px)—— 位置信息有没有起作用 |
| `rand` | **同 v 随机 u** 的基准率 —— 这个命中率是不是随便一指就有 |

⚠️ `rand` **必须配 v 分布**:车道线采样点全在画面下半(路面),拿全图均匀随机当基准
会低到 0.01 而显得判据很强。判据取**同一个 v、u 随机**。

**两条裁决**(缺第二条,第一条就可能是"路面本来就长这样"):
① `real ≥ MIN_HIT_RATE`;
② `real ≥ MIN_MARGIN × max(shift, rand)` —— 位置本身携带信息。

## 自证(每次运行都跑,不是开关)

1. **尺子在量东西**:`--self-test` 用合成 tag 图 + 解析式相机,钉死"窗真的横跨
   `SIGNAL_WINDOW_M` 米"、"带外 1 px 不计"、"横移后必须掉"、"真命中==随机时必须判不通过"。
2. **位姿在不在**:`StaticFrame.camera is None`(2026-10-02 之前采的 GT)**必须报错**,
   不许拿默认内参硬算 —— 那会得到一整套看着正常的错数。

用法:
  python -m autodrivedata.sim.collect_static_gt --sem --frames 40 --out outputs/kitti_static_sem
  python -m autodrivedata.perception.static_eval --root outputs/kitti_static_sem
  python -m autodrivedata.perception.static_eval --self-test        # 不出数据,只验尺子
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from autodrivedata.calib.core import CameraIntrinsics
from autodrivedata.gt.props import CameraPose
from autodrivedata.gt.static_gt import StaticFrame
from autodrivedata.perception.sem_tags import SEM_TAGS, TAG_NAMES, decode_tag_png
from autodrivedata.utils import geometry as g
from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 车道线采样点的邻域半宽(px)。**扫描出来的拐点**,见模块头注的表。
LANE_WINDOW_PX = 4

#: 信号锚点向上的搜索高度(m)。实测最高 tag 像素离锚点 5.2 m ⇒ 6 m 覆盖 100% 且留余量。
#: ⚠️ 放到 8 m 以上横移对照就爬到 0.225–0.30,**判别力反而塌** —— 别"顺手放宽"。
SIGNAL_WINDOW_M = 6.0

#: 信号竖带的半宽(px)。1 px 把 6 m 窗下的真命中从 0.950 补到 1.000,对照仍 0.000。
SIGNAL_BAND_PX = 1

#: 横移对照量:车道线移**半个车道宽**(落在车道中间 ⇒ 那里不该有标线)。
LANE_SHIFT_M = 1.75
#: 横移对照量:信号左右各移这么多像素。
SIGNAL_SHIFT_PX = 40

#: 裁决阈值。① 真命中率下限;② 真命中率相对**对照里更高的那个**的倍率。
MIN_HIT_RATE = 0.90
MIN_MARGIN = 5.0

#: landmark kind → 它在语义图里**应当**是哪个 tag。
#: ⚠️ `stop`/`yield` 都是 `Sign_*` 系列 ⇒ 归 `TrafficSigns`,**不是** `TrafficLight`。
SIGNAL_TAG_OF_KIND: dict[str, int] = {
    "traffic_light": SEM_TAGS["TrafficLight"],
    "stop": SEM_TAGS["TrafficSigns"],
    "yield": SEM_TAGS["TrafficSigns"],
}


def project_raw(
    pt: tuple[float, float, float], cam: CameraPose, k: CameraIntrinsics
) -> tuple[float, float] | None:
    """世界点 → 像素 (u, v),**不做"落在图外"过滤**(只挡相机后)。

    为什么不复用 `calib.core.world_to_img`:它把"图外"也返回 None,而**判据恰恰需要
    画外的 v** —— 信号窗的上端点在近处目标上会跑到画面上边缘之外,那时要拿这个 v 去
    **钳到 0**,而不是判成"这个锚点看不见"。
    """
    c = g.world_to_cam(
        np.asarray([pt], dtype=np.float64), cam.location, tuple(np.radians(a) for a in cam.rotation_deg)
    )[0]
    if float(c[2]) <= 0.5:
        return None
    return float(k.fx * c[0] / c[2] + k.cx), float(k.fy * c[1] / c[2] + k.cy)


def in_frame(u: float, v: float, k: CameraIntrinsics) -> bool:
    return 0.0 <= u < k.width and 0.0 <= v < k.height


def window_hit(tag: np.ndarray, u: float, v_top: int, v_bot: int, want: int, half: int) -> bool | None:
    """`[v_top, v_bot)` 行、`u ± half` 列里有没有 `want` 这个 tag。**空窗返回 None**。

    None 表示"这次没法判"(窗完全在画外),调用方**必须单独计数**,不许当 False 吞掉 ——
    "判不了"与"判错了"在下游长得一样,而前者是夹具问题、后者是数据问题。
    """
    x = int(round(u))
    if v_bot <= v_top or not (0 <= x < tag.shape[1]):
        return None
    x0, x1 = max(0, x - half), min(tag.shape[1], x + half + 1)
    v0, v1 = max(0, v_top), min(tag.shape[0], v_bot)
    if v1 <= v0 or x1 <= x0:
        return None
    return bool((tag[v0:v1, x0:x1] == want).any())


def lane_window_hit(
    tag: np.ndarray, pt: tuple[float, float, float], cam: CameraPose, k: CameraIntrinsics, want: int
) -> bool | None:
    """车道线采样点:`pt` 投影处 ±`LANE_WINDOW_PX` 的方窗里有没有 `want`。"""
    uv = project_raw(pt, cam, k)
    if uv is None or not in_frame(uv[0], uv[1], k):
        return None  # 画外:判不了,不计入分母
    u, v = uv
    return window_hit(
        tag, u, int(round(v)) - LANE_WINDOW_PX, int(round(v)) + LANE_WINDOW_PX + 1, want, LANE_WINDOW_PX
    )


def signal_v_span(
    anchor: tuple[float, float, float], cam: CameraPose, k: CameraIntrinsics
) -> tuple[float, float] | None:
    """锚点 → `(v_top, v_anchor)`:`v_top` 是**锚点上方 `SIGNAL_WINDOW_M` 米**那一点的 v。

    纯几何,可解析核对:`Δv ≈ SIGNAL_WINDOW_M / depth × fy`(见 `--self-test`)。
    上端点**钳到画幅**——近处目标上它会跑到画面外,那是正常的,不是"判不了"。
    """
    uv_a = project_raw(anchor, cam, k)
    if uv_a is None:
        return None
    uv_t = project_raw((anchor[0], anchor[1], anchor[2] + SIGNAL_WINDOW_M), cam, k)
    v_top = 0.0 if uv_t is None else min(uv_t[1], float(k.height))
    return v_top, uv_a[1]


def signal_window_hit(
    tag: np.ndarray, anchor: tuple[float, float, float], cam: CameraPose, k: CameraIntrinsics, want: int
) -> bool | None:
    """信号:`anchor → anchor + SIGNAL_WINDOW_M` 的竖带(`±SIGNAL_BAND_PX`)里有没有 `want`。"""
    uv = project_raw(anchor, cam, k)
    if uv is None or not in_frame(uv[0], uv[1], k):
        return None
    span = signal_v_span(anchor, cam, k)
    if span is None:
        return None
    v_top, v_bot = span
    return window_hit(tag, uv[0], int(np.floor(v_top)), int(round(v_bot)) + 1, want, SIGNAL_BAND_PX)


def lane_perp(points: tuple[tuple[float, float, float], ...], i: int) -> np.ndarray | None:
    """折线段上第 `i` 点的**水平横向**单位向量(用于"横移半车道宽"对照)。

    取相邻两点的切向再转 90°。**必须用点自己的切向**,不能用整段的平均方向 ——
    弯道上平均方向会把横移量摊到切向上,"移了半个车道宽"这句话就不成立了。
    """
    n = len(points)
    if n < 2:
        return None
    j, m = min(i + 1, n - 1), max(i - 1, 0)
    t = np.array([points[j][0] - points[m][0], points[j][1] - points[m][1]], dtype=float)
    norm = float(np.linalg.norm(t))
    if norm < 1e-6:
        return None
    return np.array([-t[1], t[0]]) / norm


@dataclass
class Sample:
    """一条样本的三个读数 + 它属于哪一类。`real is None` = 判不了。

    `off` 记**为什么判不了**,分两种 —— 混成一个数会把两种完全不同的处置方式糊在一起:
    - `"behind"`:这个点在**相机后面**。透镜原理上就看不见,与判据无关;
    - `"fov"`:在相机前方但落在**视场/画幅外**。这一条才是"这套 rig 覆盖不到",
      与静态 GT 的 `LANDMARK_HORIZON` 是**圆形**过滤(会把车后的也收进来)直接相关。
    """

    group: str
    real: bool | None
    shift: bool | None
    rand: bool | None
    off: str | None = None


@dataclass
class GroupStat:
    """一类的汇合读数。区分**画外**与**判错** —— 两者混一起就读不出结论了。"""

    group: str
    real: list[bool] = field(default_factory=list)
    shift: list[bool] = field(default_factory=list)
    rand: list[bool] = field(default_factory=list)
    off_frame: int = 0
    off_behind: int = 0
    off_fov: int = 0

    def add(self, s: Sample) -> None:
        if s.real is None:
            self.off_frame += 1
            if s.off == "behind":
                self.off_behind += 1
            elif s.off == "fov":
                self.off_fov += 1
            return
        self.real.append(s.real)
        if s.shift is not None:
            self.shift.append(s.shift)
        if s.rand is not None:
            self.rand.append(s.rand)

    @staticmethod
    def _rate(xs: list[bool]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    @property
    def n(self) -> int:
        return len(self.real)

    @property
    def real_rate(self) -> float:
        return self._rate(self.real)

    @property
    def shift_rate(self) -> float:
        return self._rate(self.shift)

    @property
    def rand_rate(self) -> float:
        return self._rate(self.rand)

    @property
    def control(self) -> float:
        """对照里**更高的那个** —— 裁决要比的是"最难超过的那个",不是随便挑一个。"""
        vals = [v for v in (self.shift_rate, self.rand_rate) if not np.isnan(v)]
        return max(vals) if vals else float("nan")

    def verdict(
        self, min_hit: float = MIN_HIT_RATE, min_margin: float = MIN_MARGIN
    ) -> tuple[bool | None, str]:
        """`None` = **未判**(一条可判样本都没有),不是"通过"也不是"不通过"。

        ★ 这条与 `sem_eval` 对空类的处置**同源**(那边"某类 BEV 一个像素都没有时
        mIoU 会跳过它并单独打印警告"):一个**判不了**的组不能算过 —— 那会把"没测到"
        读成"没问题";也不该算不过 —— 那会让判据在"这条路正好有条画外的黄双实线"时
        永远红,然后被调阈值调绿。**两种都是把"不知道"伪装成结论。**
        """
        if self.n == 0:
            return None, f"未判(全在画外:车后 {self.off_behind} / 视场外 {self.off_fov})"
        if self.real_rate < min_hit:
            return False, f"真命中 {self.real_rate:.3f} < {min_hit}"
        c = self.control
        if not np.isnan(c) and self.real_rate < min_margin * c:
            # ★ **这条才是判据**。少了它,`real=0.4` 在一个"随便一指就有 0.4"的场景里
            #   也会被判通过 —— 而那不是"xodr 对得上",那是"这场景到处都是标线"。
            return False, f"真命中 {self.real_rate:.3f} 未超过对照 {c:.3f} 的 {min_margin}×"
        return True, f"真命中 {self.real_rate:.3f}(对照 {c:.3f})"


def eval_frame(frame: StaticFrame, tag: np.ndarray, rng: np.random.Generator) -> list[Sample]:
    """一帧 → 逐样本读数。**纯函数**(给合成夹具用,也是单测的入口)。"""
    if frame.camera is None:
        raise ValueError(
            "这份 static_gt 没落相机位姿(`camera` 为空)—— 2026-10-02 之前采的 GT 都是这样。"
            "判据**不许**拿默认内参硬算(会得到一整套看着正常的错数)。"
            "重采:python -m autodrivedata.sim.collect_static_gt --sem"
        )
    cam = frame.camera
    k = CameraIntrinsics(cam.width, cam.height, cam.fov_deg)
    out: list[Sample] = []

    def off_reason(pt: tuple[float, float, float]) -> str | None:
        """为什么这条判不了 —— `None` 表示**判得了**。见 `Sample.off` 那两条的差别。"""
        uv = project_raw(pt, cam, k)
        if uv is None:
            return "behind"
        return None if in_frame(uv[0], uv[1], k) else "fov"

    for seg in frame.lane_lines:
        want = SEM_TAGS["RoadLines"]
        pts = seg.points
        for i, pt in enumerate(pts):
            real = lane_window_hit(tag, pt, cam, k, want)
            perp = lane_perp(pts, i)
            shift: bool | None = None
            if perp is not None:
                hs: list[bool] = []
                for sgn in (+1.0, -1.0):
                    q = (pt[0] + perp[0] * LANE_SHIFT_M * sgn, pt[1] + perp[1] * LANE_SHIFT_M * sgn, pt[2])
                    h = lane_window_hit(tag, q, cam, k, want)
                    if h is not None:
                        hs.append(h)
                shift = any(hs) if hs else None
            # rand 与 real **同一个 v**:基准率必须配着"采样点落在画面哪一带"来算
            uv = project_raw(pt, cam, k)
            rand = (
                None
                if uv is None or not in_frame(uv[0], uv[1], k)
                else window_hit(
                    tag,
                    float(rng.integers(0, k.width)),
                    int(round(uv[1])) - LANE_WINDOW_PX,
                    int(round(uv[1])) + LANE_WINDOW_PX + 1,
                    want,
                    LANE_WINDOW_PX,
                )
            )
            out.append(Sample(f"lane:{seg.mark_type}", real, shift, rand, off_reason(pt)))

    for sig in frame.signals:
        want = SIGNAL_TAG_OF_KIND.get(sig.kind)
        if want is None:
            continue  # unknown 已经不进 GT(见 static_gt.landmark_kind),这里是第二道闸
        real = signal_window_hit(tag, sig.location, cam, k, want)
        # 横移对照在**图像空间**做(锚点世界坐标不动,但按 ±40 px 查列)——
        # 那正是"位置信息有没有起作用"的直接问法。
        uv = project_raw(sig.location, cam, k)
        shift: bool | None = None
        rand: bool | None = None
        span = signal_v_span(sig.location, cam, k)
        if uv is not None and span is not None:
            v_top, v_bot = span
            ss: list[bool] = []
            for dx in (+SIGNAL_SHIFT_PX, -SIGNAL_SHIFT_PX):
                h = window_hit(
                    tag, uv[0] + dx, int(np.floor(v_top)), int(round(v_bot)) + 1, want, SIGNAL_BAND_PX
                )
                if h is not None:
                    ss.append(h)
            shift = any(ss) if ss else None
            rand = window_hit(
                tag,
                float(rng.integers(0, k.width)),
                int(np.floor(v_top)),
                int(round(v_bot)) + 1,
                want,
                SIGNAL_BAND_PX,
            )
        out.append(Sample(f"signal:{sig.kind}", real, shift, rand, off_reason(sig.location)))
    return out


def eval_root(root: Path, frames: str | None = None) -> dict[str, Any]:
    """整个 root 跑一遍。返回汇合结果(纯副作用只有读文件,判据在调用方)。"""
    gt_dir = root / "training/static_gt"
    sem_dir = root / "training/static_sem"
    if not sem_dir.is_dir():
        raise SystemExit(
            f"{sem_dir} 不存在 —— 判据要语义图当 oracle。"
            f"采集时加 `--sem`:python -m autodrivedata.sim.collect_static_gt --sem ..."
        )
    fids = sorted(p.stem for p in gt_dir.glob("*.json"))
    if frames:
        a, _, b = frames.partition("-")
        lo, hi = int(a), int(b) if b else int(a)
        fids = [f for f in fids if lo <= int(f) <= hi]
    rng = np.random.default_rng(20261002)
    groups: dict[str, GroupStat] = {}
    for fid in fids:
        frame = StaticFrame.from_json((gt_dir / f"{fid}.json").read_text(encoding="utf-8"))
        tag = decode_tag_png((sem_dir / f"{fid}.png").read_bytes())
        for s in eval_frame(frame, tag, rng):
            groups.setdefault(s.group, GroupStat(s.group)).add(s)
    return {"frames": len(fids), "groups": groups}


def _report(res: dict[str, Any]) -> bool:
    print(f"\n=== 静态 GT 判据({res['frames']} 帧)===")
    print(f"{'组':<22}{'n':>5}{'真命中':>9}{'横移对照':>10}{'随机基准':>10}{'车后':>6}{'视场外':>7}   裁决")
    ok_all = True
    n_unjudged = 0
    for name in sorted(res["groups"]):
        gs: GroupStat = res["groups"][name]
        ok, why = gs.verdict()
        if ok is None:
            n_unjudged += 1
            mark = "⚠️ "
        else:
            ok_all &= ok
            mark = "✅ " if ok else "❌ "
        print(
            f"{name:<22}{gs.n:>5}{gs.real_rate:>9.3f}{gs.shift_rate:>10.3f}"
            f"{gs.rand_rate:>10.3f}{gs.off_behind:>6}{gs.off_fov:>7}   {mark}{why}"
        )
    if n_unjudged:
        # ★ 单独喊一声 —— "未判"既不进通过也不进不通过,不喊就会被读成"这类做得好"
        #   (同 `sem_eval` 对空类的处置:跳过 + 单独打印警告)。
        print(f"\n⚠️ {n_unjudged} 个组**未判**(一条可判样本都没有)—— 不计入裁决。")
    print(
        f"\n裁决口径:真命中 ≥ {MIN_HIT_RATE},且 ≥ {MIN_MARGIN}× max(横移, 随机)"
        "\n⚠️ tag 名对照:"
        + ", ".join(f"{t}={TAG_NAMES.get(t, t)}" for t in sorted(set(SIGNAL_TAG_OF_KIND.values())))
        + f", {SEM_TAGS['RoadLines']}={TAG_NAMES[SEM_TAGS['RoadLines']]}"
    )
    return ok_all


def self_test() -> bool:
    """**尺子自证**(不需要任何数据):窗真的横跨 6 m / 带外不计 / 横移掉 / 阈值会红。"""
    ok = True
    k = CameraIntrinsics(100, 100, 90.0)
    # 相机在原点、朝向 +x(世界),即 carla yaw=0 ⇒ 相机系 z 轴朝前
    cam = CameraPose(
        location=(0.0, 0.0, 0.0), rotation_deg=(0.0, 0.0, 0.0), width=100, height=100, fov_deg=90.0
    )

    # ① 窗真的横跨 SIGNAL_WINDOW_M 米:解析式 Δv = Δz / depth × fy
    depth = 20.0
    span = signal_v_span((depth, 0.0, 0.0), cam, k)
    assert span is not None, "锚点在相机正前方,不该判不出窗"
    dv = span[1] - span[0]
    want_dv = SIGNAL_WINDOW_M / depth * k.fy
    good = abs(dv - want_dv) < 0.05 * want_dv
    print(f"① 窗高 {dv:.1f} px,解析式 {want_dv:.1f} px —— {'✅' if good else '❌'}")
    ok &= good

    # ② 带外不计:tag 摆在窗**上方** 1 px ⇒ 必须不命中
    tag = np.zeros((100, 100), dtype=np.uint8)
    tl = np.uint8(SEM_TAGS["TrafficLight"])
    anchor = (depth, 0.0, 0.0)
    uv = project_raw(anchor, cam, k)
    assert uv is not None
    u, v0 = int(round(uv[0])), int(round(uv[1]))
    inside = tag.copy()
    inside[v0 - 5, u] = tl
    outside = tag.copy()
    outside[int(np.floor(span[0])) - 1, u] = tl
    a = signal_window_hit(inside, anchor, cam, k, int(tl))
    b = signal_window_hit(outside, anchor, cam, k, int(tl))
    good = a is True and b is False
    print(f"② 窗内命中={a} / 窗上一像素={b} —— {'✅' if good else '❌'}")
    ok &= good

    # ③ 真命中 == 随机时**必须判不通过**(margin 那条不是摆设)
    gs = GroupStat("x", [True] * 10, [True] * 10, [True] * 10, 0)
    passed, _ = gs.verdict()
    good = not passed
    print(f"③ real==control 时 verdict={passed} —— {'✅' if good else '❌'}")
    ok &= good

    # ④ 没落位姿必须抛,不许默认内参硬算
    try:
        eval_frame(
            StaticFrame(frame_id="0", ego_location=(0, 0, 0), ego_yaw_deg=0.0), tag, np.random.default_rng(0)
        )
        good = False
    except ValueError:
        good = True
    print(f"④ camera 为空时抛 ValueError —— {'✅' if good else '❌'}")
    ok &= good

    # ⑤ 横向单位向量:一条沿 +y 的直线,横向必须是 ±x
    perp = lane_perp(((0.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 10.0, 0.0)), 1)
    good = perp is not None and abs(abs(float(perp[0])) - 1.0) < 1e-9 and abs(float(perp[1])) < 1e-9
    print(f"⑤ 沿 +y 的线,横向 = {None if perp is None else perp.round(3)} —— {'✅' if good else '❌'}")
    ok &= good
    return bool(ok)


def main() -> None:
    ap = argparse.ArgumentParser(description="静态 GT(信号/车道线)vs 语义渲染的判据")
    ap.add_argument("--root", default="outputs/kitti_static_sem")
    ap.add_argument("--frames", default=None, help="如 0-19")
    ap.add_argument("--self-test", action="store_true", help="不出数据,只验尺子")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)

    root = project_path(args.root)
    with runlog.run("autodrivedata.perception.static_eval") as rl:
        rl.input(str(root), "root")
        rl.highlight("lane_window_px", LANE_WINDOW_PX)
        rl.highlight("signal_window_m", SIGNAL_WINDOW_M)
        rl.highlight("signal_band_px", SIGNAL_BAND_PX)
        res = eval_root(root, args.frames)
        ok = _report(res)
        for name in sorted(res["groups"]):
            gs = res["groups"][name]
            rl.highlight(f"{name}.real", round(gs.real_rate, 4))
            rl.highlight(f"{name}.shift", round(gs.shift_rate, 4))
            rl.highlight(f"{name}.rand", round(gs.rand_rate, 4))
            rl.highlight(f"{name}.n", gs.n)
            rl.highlight(f"{name}.off_behind", gs.off_behind)
            rl.highlight(f"{name}.off_fov", gs.off_fov)
        rl.highlight("frames", res["frames"])
        rl.highlight("pass", ok)
        rl.note("xodr 与渲染对得上" if ok else "有组未过 —— 见上面表格")


if __name__ == "__main__":
    main()

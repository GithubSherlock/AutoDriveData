"""静态遮挡 A/B 的**纯几何**(零 carla):遮挡物摆哪、挡住多少、剩多少像素。

## 为什么要单独一层

`collect_ab_route` 的 A/B 纪律是「只变一个变量」。天气那几条变的是 `scene.weather`
(渲染侧),而**遮挡**变的是**场景里多出来的物体** —— 一旦它摆错位置,现象不是崩溃,
是"效果比预期小一半"或"根本没挡住",而这两个都与"模型对遮挡不鲁棒"长得一模一样。
所以摆放规则必须是**可离线手算、可单测**的纯值,不能埋在采集器里靠肉眼看图。

## 两个约定,混一个数就全错

| 量 | 定义 | 谁用 |
|---|---|---|
| **深度 depth** | 沿**相机光轴**的距离 = `ego 局部 x − CAM_FWD` | 像面行/列(`y = cy + f·tan(俯角)`) |
| **斜距 slant** | 到相机的直线距离 = `hypot(x − CAM_FWD, y)` | 摆位(沿视线取点、垂线方向) |

**这条不是学究**:2026-10-01 第一版全程用斜距去算像面高度,20 m 处那台车算出"整框
49.7 px",而同一帧的 `label_2` 写着 **58.6 px**(差 18%)。成像吃的是深度,不是斜距 ——
车横着挪 3.5 m 并不会让它变矮。

## 关键决定:遮挡物排在哪

**排在 相机→车 的视线上、长轴垂直于视线**(不是排在 ego 正前方)。理由是可算的:

车在路侧(横向 +3.5 m),若把遮挡物排在 ego **正前方**同一横向偏移处,它在像面上
因**视差**比车更靠外 —— 20 m 处车的**内侧边**在 7.33°,同一个 `span` 搬到正前方后
墙的内端跑到了 **8.50°**,于是车**内侧整整一条全高的缝**露出来(占车宽 **18.3%**),
遮挡就从"定量"退化成"看运气"。排在视线上则是 0%,剩下的**只有竖直方向**一个自由度。

## 三个量,全部由几何直接给出

- `sight_line_occluders`:遮挡物摆哪 —— 由「沿视线距车 `gap` 米」+「横向盖住
  `required_span`」唯一确定,顺带给出每块的**净空**(`Occluder.min_y`,别撞上 ego);
- `visible_fraction`:挡完还剩多少 —— 相机、遮挡物顶、车顶三者各一条俯角,
  可见段 = `[车顶角, 遮挡物顶角]`;
- `project`:同一组量的像素版(判据用:可见高度落在 21–24 px 断崖上时最可能掉点)。

三条都**不读 carla**,资产尺寸只吃 `OCCLUDER_DIMS` 那一份**声明**;采集时读回**已 spawn
的 actor** 对表(声明 = 渲染,`collect_ab_route._spawn_occluders`),对不上就停。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: 遮挡档位 → blueprint id。**只挑 `static.prop.*`** —— 判据不是"看着像道具",是 GT 过滤器:
#: `collect_ab_route` 收 `label_2` 时只认 `vehicle*` / `walker*` 前缀,道具**结构上不可能**
#: 进 GT ⇒ A/B 两侧 GT 逐帧相等这条硬门槛自动成立(见 Plan4 §P-V12)。
OCCLUDER_MODELS: dict[str, str] = {
    # 实测 bbox 1.215 × 0.372 × 1.069 高 ⇒ **低于相机**(1.65 m)⇒ 只截掉车身下一截 = 部分遮挡
    "partial": "static.prop.streetbarrier",
    # 实测 bbox 1.305 × 1.055 × 1.860 高 ⇒ **高于相机** ⇒ 视线无论俯仰都被截断 = 恒全遮
    # (这条不是估计:见 `visible_fraction` 的两条钳位与 `TestProjection` 的阈值钉)
    "full": "static.prop.warningconstruction",
}

#: 各档资产的**实测**尺寸 `(长 = 沿长轴, 厚, 高)` m,量自 CARLA 0.9.16 资产 bbox。
#: 摆位数学只吃这一份 —— 采集时另起一个 **yaw=0** 的探针读回对表,对不上就停。
#: (探针必须另起:转过的 actor 读回的 `bounding_box` 是 `extent/rotation` 自相矛盾的,
#:  同一个 streetbarrier yaw=0 读 1.215×0.372、yaw=80.07° 读 0.157×1.261。)
OCCLUDER_DIMS: dict[str, tuple[float, float, float]] = {
    "partial": (1.215, 0.372, 1.069),
    "full": (1.305, 1.055, 1.860),
}

#: 相机相对 ego 原点的**前移量** = `SENSOR_OFFSET.location.x`。深度与斜距之差就是它;
#: 少了这一项,20 m 处那台车整框算成 49.7 px 而真值 58.6 px。
CAM_FWD = 1.2

#: 遮挡物中心沿**视线**到目标车的距离(m)。**越大越不挡**:墙离车越远就离相机越近,
#: 但它要盖住的那段俯角反而更窄(20 m 处那台车:gap=3 ⇒ 可见 23.4 px,gap=6 ⇒ 28.6 px)。
#:
#: **3.0 是照 21–24 px 断崖定的**(P1 已确立:框高掉到那一段检出率从 ~0.8 跌到 0.15–0.47)。
#: 2026-10-01 定档时的初值是 6.0 —— 那是在"斜距 vs 深度"那处换算修好**之前**算的,
#: 同一份几何按深度重算是 28.6 px(已越过断崖);照**意图**(落在断崖上)反解就是 3.0。
#: 顺带把最小净空从 0.37 m 抬到 0.86 m。
OCCLUDER_GAP = 3.0

#: 遮挡物**总跨度 / 理论所需跨度**。>1 才有横向余量;1.25 = 两侧各留 12.5%。
OCCLUDER_COVER = 1.25


@dataclass(frozen=True)
class Occluder:
    """一个待摆的遮挡物:**ego 局部坐标**(x 前 / y 右),`yaw_deg` 只绕竖轴。

    `yaw_deg` 使长轴落在视线的垂线上 —— 这样墙顶在像面上是**水平**的,可见高度沿车宽恒定;
    否则顶边在像面上是斜的,"挡住多少"就变成逐列不同的量。

    `min_y` = 这块的 footprint 在 ego 局部系里的**最小横向坐标**(= 离 ego 中轴最近处)。
    它不是"顺便算的":ego 定速走 y≈0 的直线,`min_y` 减去 ego 半车宽就是**净空** ——
    净空为负意味着墙伸进车道、ego 会撞上去,而**撞了照样能采完**,只是轨迹跟 A 侧不再配对。
    """

    x: float
    y: float
    yaw_deg: float
    length: float
    min_y: float

    @property
    def depth(self) -> float:
        """这块的**深度**(沿相机光轴)= 摆位数学与成像之间的那一处坐标换算。"""
        return self.x - CAM_FWD


def sight_line_occluders(
    car_d: float,
    car_lat: float,
    *,
    cam_fwd: float = CAM_FWD,
    gap: float,
    unit_len: float,
    unit_thick: float,
    cover: float,
    needed_width: float,
) -> list[Occluder]:
    """沿 **相机→车** 视线铺遮挡物,横向总跨度 `cover × needed_width`,返回各块的中心与朝向。

    输入 `car_d / car_lat` 是 **ego 局部坐标**(与 `collect_ab_route` 摆路肩车用的是同一套),
    相机在 `(cam_fwd, 0, z)` —— 视线从**相机**出发,不是从 ego 原点出发。
    `needed_width` 是**该车投到遮挡物所在平面上的**跨度(`required_span` 算),不是车的物理宽度。
    """
    dx, dy = car_d - cam_fwd, car_lat
    r = math.hypot(dx, dy)
    if not 0.0 <= gap < r:
        raise ValueError(f"gap={gap} 必须落在 [0, {r:.3f}) 内(否则遮挡物跑到相机后面或与车重叠)")
    ux, uy = dx / r, dy / r  # 视线单位向量
    px, py = -uy, ux  # 视线的垂线 —— 长轴方向
    cx, cy = car_d - gap * ux, car_lat - gap * uy
    yaw = math.degrees(math.atan2(py, px))

    span = max(cover * needed_width, unit_len)  # 至少要放得下一块
    n = max(1, math.ceil(span / unit_len))
    step = (span - unit_len) / (n - 1) if n > 1 else 0.0  # n 块首尾相接正好铺满 span
    out: list[Occluder] = []
    for k in range(n):
        o = (k - (n - 1) / 2.0) * step
        min_y = cy + (o - unit_len / 2.0) * py - (unit_thick / 2.0) * abs(uy)
        out.append(Occluder(cx + o * px, cy + o * py, yaw, unit_len, min_y))
    return out


def occluder_depth(car_d: float, car_lat: float, *, cam_fwd: float = CAM_FWD, gap: float) -> float:
    """遮挡物所在平面的**深度**(沿光轴)= `(car_d − cam_fwd)·(1 − gap/斜距)`。

    单列出来是因为**摆位要先知道它才知道要多宽**,而它是摆位公式里就能闭式解出的
    (垂线偏移不改变沿视线的那个分量)。让调用方各自再算一遍,就等于把"深度 vs 斜距"
    那条坑重新挖一次。
    """
    depth_car = car_d - cam_fwd
    return depth_car * (1.0 - gap / math.hypot(depth_car, car_lat))


def required_span(car_width: float, depth_car: float, depth_occl: float) -> float:
    """侧向投影:车宽 `car_width` 投到**遮挡物所在的深度**上占多宽(`w · d_occl / d_car`)。

    墙比车**离相机近** ⇒ 同样的视角只要更小的一块:2.163 m 的车在 18.8 m、墙在 12.9 m,
    墙只要 **1.484 m**。直接拿 `car_width` 会**多盖** 46% —— 那不会漏缝,但会白白吃掉
    车道净空(墙越宽越往 ego 那边伸),而净空是"撞了也照样采得完"的那类不可见失败。
    """
    return car_width * depth_occl / depth_car


def angle_below_horizon(height: float, cam_z: float, depth: float) -> float:
    """相机看向 (深度 `depth`, 高 `height`) 那一点的**俯角**(弧度,正 = 在水平面下)。

    `depth` 是**沿光轴**的距离,不是斜距 —— 见模块头注那张表。
    """
    return math.atan2(cam_z - height, depth)


def visible_fraction(
    *, cam_z: float, occl_top_z: float, car_top_z: float, depth_car: float, depth_occl: float
) -> float:
    """挡完还剩**竖直方向的占比**(0 = 全遮,1 = 一点没挡)。

    可见段 = `[车顶的俯角, 遮挡物顶的俯角]`。两处钳位都有实义,不是保险丝:

    - `occl_top_z > cam_z`(墙顶在水平面**之上**)⇒ `a_occ < 0` ⇒ 分子为负 ⇒ 钳到 **0**
      = "视线无论往上往下都被这堵墙截断"。`full` 档(1.86 m)恒 0 正是靠这条;
    - `a_occ > a_bot`(墙顶在像面上低过车底)⇒ 占比 > 1 ⇒ 钳到 **1** = "一点没挡"。

    **别把"车顶高过相机"当退化去特判** —— 那时 `a_top < 0`(车有一部分在水平面之上),
    公式照样对:可见段从**地平线以上**的车顶一路算到墙顶。
    """
    a_top = angle_below_horizon(car_top_z, cam_z, depth_car)
    a_bot = angle_below_horizon(0.0, cam_z, depth_car)
    a_occ = angle_below_horizon(occl_top_z, cam_z, depth_occl)
    if a_bot <= a_top:  # 真退化:车顶落到地面以下,分母 ≤ 0,占比无定义
        return 1.0
    return max(0.0, min(1.0, (a_occ - a_top) / (a_bot - a_top)))


def px_span(focal_px: float, a_img_bot: float, a_img_top: float) -> float:
    """两个俯角之间的**像面高度**(px);`a_img_bot` 是像面上**靠下**那条边(俯角更大)。

    相机 pitch=0(`SENSOR_OFFSET` 只有平移)⇒ 光轴水平 ⇒ `y = cy + f·tan(俯角)`,
    两次作差即得。方形像素 ⇒ `fx == fy`,只传一个 `focal_px`。
    """
    return focal_px * (math.tan(a_img_bot) - math.tan(a_img_top))


@dataclass(frozen=True)
class Projection:
    """一辆车在**某一瞬间**的可见性。三个数一起报,"挡住了多少"才不会被读成绝对值。"""

    full_px: float  # 不遮时整辆车多高
    visible_px: float  # 遮住之后可见那截多高
    visible_frac: float  # 可见竖直占比


def project(
    *, focal_px: float, cam_z: float, occl_top_z: float, car_top_z: float, depth_car: float, depth_occl: float
) -> Projection:
    """把几何量换成像素。**判据看 `visible_px`**:它落在 21–24 px 断崖上时最可能掉点。"""
    a_top = angle_below_horizon(car_top_z, cam_z, depth_car)
    a_bot = angle_below_horizon(0.0, cam_z, depth_car)
    a_occ = angle_below_horizon(occl_top_z, cam_z, depth_occl)
    return Projection(
        full_px=px_span(focal_px, a_bot, a_top),
        visible_px=max(0.0, px_span(focal_px, a_occ, a_top)),
        visible_frac=visible_fraction(
            cam_z=cam_z,
            occl_top_z=occl_top_z,
            car_top_z=car_top_z,
            depth_car=depth_car,
            depth_occl=depth_occl,
        ),
    )

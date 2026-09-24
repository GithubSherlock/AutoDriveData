"""自车 + 传感器标定配置的**配置图**(纯值,不 import carla / torch)。

三张图拼一页,回答"这套 rig 到底怎么装的、看哪儿、盖住多少":

- **左上 · 俯视**:车体轮廓(实测包围盒)+ 逐相机挂点(带符号标记)+ 各自的视锥(颜色 + 短码);
- **右上 · 方位环**:逐相机方位扇区沿整圈排开 —— **重叠**自动加深、**盲区**用红弧标出并写度数;
- **右下 · 数字表**:通道名 / 挂点 x,y,z / 方位角 az / FoV / **"az ± fov/2"**(如 `+0.3° ± 27.5°`)。

## 为什么要单独成模块

用户口径要的是"能拿去排计划"的配置图,而它**只依赖纯值表**(`camera_rig` 的标定 +
`export/nuscenes` 的 FoV)。若画在两代 rig 共用的 CARLA 脚本里,想画"还没采过的候选 rig"
就必须起服务器;纯值落点可以只改常量就出图(这也是本模块接受**显式 calibs/fov 参数**、
而不是自己去查 rig 名的原因 —— 调用方给什么就画什么,候选 rig 无需登记)。

⚠️ **图上的方位角是 `az_nus`(0°=车头,+ = 左)**,与官网标定表同口径;屏幕映射为
`x_screen = −sin(az)`、`y_screen = −cos(az)`(nus 的 y 左 → CARLA 的 y 右 → 屏幕 x 右),
故 az 增大在屏幕上是**逆时针**,与 `bin/viz_calib_check.sheet_geometry` 同一约定。
"""

from __future__ import annotations

import math
from typing import Any

from PIL import Image, ImageDraw

from autodrivedata import fonts
from autodrivedata.camera_rig import nus_camera_rig
from autodrivedata.geometry import NUS_EGO_ORIGIN_X, quat_normalize, quat_to_matrix

# 车体实测包围盒(vehicle.audi.a2):半长 / 半宽(米)。与 `camera_rig` 头注的
# "车身最后点 x=−1.8527" 同源 —— 那里是后挂点余量的依据,这里是画轮廓的依据。
EGO_HALF_LEN = 1.8527
EGO_HALF_WID = 0.8943

# nuScenes ego 原点(后轴中心)在 CARLA 车体系里的 x —— 与 `geometry.NUS_EGO_ORIGIN_X`
# **同一来源**(不另抄数)。整套 `calibrated_sensor` 的 x 都以它为基准,而 CARLA 车辆
# actor 的原点是车身长度中点,故俯视图上必须画出来:否则"挂点为什么在这儿"看不出来。
ORIGIN_COLOR = (205, 55, 150)

# 逐通道画色(与 `bin/viz_calib_check.CAM_COLOR` 同族:前后左右各一色,一眼分得开)
CAM_COLOR: dict[str, tuple[int, int, int]] = {
    "CAM_FRONT": (0, 90, 200),
    "CAM_FRONT_LEFT": (0, 160, 110),
    "CAM_FRONT_RIGHT": (0, 175, 215),
    "CAM_BACK": (215, 150, 0),
    "CAM_BACK_LEFT": (200, 60, 60),
    "CAM_BACK_RIGHT": (150, 90, 200),
}

# 通道名 → 俯视图上的短码(全名在左侧图例里,锥尖只放短码以免三个前视标签叠在一起)
CAM_SHORT: dict[str, str] = {
    "CAM_FRONT": "F",
    "CAM_FRONT_LEFT": "FL",
    "CAM_FRONT_RIGHT": "FR",
    "CAM_BACK": "B",
    "CAM_BACK_LEFT": "BL",
    "CAM_BACK_RIGHT": "BR",
}

INK = (20, 20, 24)
DIM = (110, 110, 120)
GAP_COLOR = (220, 40, 40)
OVERLAP_COLOR = (250, 140, 0)

# 页脚(口径说明)。**必须是模块常量**:折行是按实测像素宽做的(见 `fonts.wrap`),
# 单测要能对同一份文字断言"折完每一行都装得下",写字面量在函数里就断言不到。
FOOTER = (
    "方位角 az_nus:0° = 车头,+ = 左(与 nuScenes 官网标定表同口径)。方向与范围 = az ± fov/2。"
    " 表内方位角用 (-180°,180°] 标记、环上按 [0°,360°) 排布(如 CAM_BACK_RIGHT 的 -145° 即环上 215°)。"
    " 车体轮廓 = vehicle.audi.a2 实测包围盒,图上车体中心 = CARLA actor 原点(车身长度中点)。"
    " nuScenes 官方标定表的挂点是**后轴中心为原点**的坐标,而 CARLA 车辆 actor 的原点在中点 ——"
    " 两套系差 1.2563 m。洋红线 = nuScenes 原点(后轴),空心灰圈 = 修正前(少减这一截)的挂点位置。"
)
FOOTER_SIZE = 18
FOOTER_LH = 26  # 行距(折行后逐行画,从底边往上排,免得压到数字表)


def _dir_screen(az_deg: float) -> tuple[float, float]:
    """nus 方位角(度) → 屏幕单位方向(x 右 / y 下)。az=0 朝上,az=+90 朝左。"""
    a = math.radians(az_deg)
    return (-math.sin(a), -math.cos(a))


def _sector_poly(
    centre: tuple[float, float],
    az_lo: float,
    az_hi: float,
    r_in: float,
    r_out: float,
    step: float = 1.0,
) -> list[tuple[float, float]]:
    """环形扇区多边形(az_lo→az_hi 采样外弧,再逆序回来内弧)。`r_in=0` 即实心扇形。"""
    n = max(2, int(abs(az_hi - az_lo) / step) + 1)
    azs = [az_lo + (az_hi - az_lo) * i / (n - 1) for i in range(n)]
    outer = [(centre[0] + r_out * _dir_screen(a)[0], centre[1] + r_out * _dir_screen(a)[1]) for a in azs]
    inner = [
        (centre[0] + r_in * _dir_screen(a)[0], centre[1] + r_in * _dir_screen(a)[1]) for a in reversed(azs)
    ]
    return outer + inner


def _alpha_poly(
    img: Image.Image, pts: list[tuple[float, float]], color: tuple[int, int, int], alpha: int
) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).polygon(pts, fill=(*color, alpha))
    img.alpha_composite(layer)


def _chip(
    d: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    size: int,
    fill: tuple[int, int, int],
    anchor: str = "mm",
    pad: int = 3,
) -> None:
    """带白色衬底的文字(俯视图里挂点/锥尖/车头三组标签必然互压,衬底是唯一低成本的解)。

    ⚠ 走 `fonts.width` / `fonts.bbox` 量宽 —— **不能按字符数估**:本机 CJK 字体是
    **比例宽**(见 Plan2 §P-M.8),`len(text) * size` 估出来的框与真实字宽无关。
    """
    w = fonts.width(text, size)
    bb = fonts.bbox(d, text, size)
    h = bb[3] - bb[1]
    x, y = xy
    if anchor == "mm":
        box = [x - w / 2, y - h / 2 - 2, x + w / 2, y + h / 2 + 2]
    elif anchor == "ms":  # 底边贴着 y(文字在 y 之上)
        box = [x - w / 2, y - h - pad, x + w / 2, y + 1]
    else:  # "lm" 左中
        box = [x - pad, y - h / 2 - 2, x + w + pad, y + h / 2 + 2]
    d.rectangle(box, fill=(255, 255, 255))
    fonts.draw_text(d, xy, text, size=size, fill=fill, anchor=anchor)


def panel_top_view(
    img: Image.Image,
    mounts_carla: dict[str, tuple[float, float, float]],
    az: dict[str, float],
    fov: dict[str, float],
    box: tuple[int, int, int, int],
    scale: float = 66.0,
    wedge_m: float = 1.55,
) -> None:
    """俯视图:车体 + 挂点 + 视锥(短码 + 度数)。**全名与 az±fov/2 在图例/表里**,锥尖只放短码
    —— 前三个挂点相距不到 1 m,把全名摆在锥尖必然互压。"""
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def to_screen(x_c: float, y_c: float) -> tuple[float, float]:
        """CARLA 车体系 (x 前, y 右) → 屏幕;前 = 上,y 右 = 右。"""
        return (cx + scale * y_c, cy - scale * x_c)

    # 车体轮廓 + 车头三角(**标在车头**,不是车尾:前 = 屏幕上方)
    hl, hw = EGO_HALF_LEN * scale, EGO_HALF_WID * scale
    d.rounded_rectangle(
        [cx - hw, cy - hl, cx + hw, cy + hl], radius=10, fill=(238, 240, 244), outline=INK, width=3
    )
    d.polygon([(cx, cy - hl - 18), (cx - 11, cy - hl + 2), (cx + 11, cy - hl + 2)], fill=INK)
    _chip(d, (cx + 16, cy - hl - 20), "车头", 19, (60, 60, 70), anchor="lm")

    # ★ nuScenes ego 原点(后轴中心)横线。整套 `calibrated_sensor` 的 x 都以它为基准,
    # 而 CARLA actor 的原点(= 本图的车体中心,两轴交点)是车身长度中点 —— 差
    # `NUS_EGO_ORIGIN_X`。不画这条线,"挂点为什么在后窗上方而不是车头"就看不出来。
    ax, ay = to_screen(NUS_EGO_ORIGIN_X, 0.0)
    d.line([ax - hw - 6, ay, ax + hw + 6, ay], fill=ORIGIN_COLOR, width=3)
    # 标签放**车体左侧**(不压 CAM_BACK 的挂点:CAM_BACK 恰在后轴正上方,只差 0.03 m ≈ 2 px)
    _chip(d, (x0 + 4, ay), "nuScenes 原点 · 后轴", 17, ORIGIN_COLOR, anchor="lm")

    # 视锥 + 短码
    for name, m in mounts_carla.items():
        px, py = to_screen(m[0], m[1])
        col = CAM_COLOR.get(name, (80, 80, 80))
        half = fov[name] / 2.0
        _alpha_poly(
            img, _sector_poly((px, py), az[name] - half, az[name] + half, 0.0, wedge_m * scale), col, 64
        )
        d = ImageDraw.Draw(img)
        ux, uy = _dir_screen(az[name])
        _chip(d, (px, py - 9), CAM_SHORT.get(name, "?"), 21, col, anchor="ms")
        _chip(
            d,
            (px + wedge_m * scale * 0.66 * ux, py + wedge_m * scale * 0.66 * uy),
            f"{az[name]:+.1f}°",
            17,
            (70, 70, 80),
        )
        d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=col, outline=INK)

    # 空心圈 = **修正前**的挂点位置(整套少的正是这一截 `NUS_EGO_ORIGIN_X`)。画在锥体之上,
    # 一眼看出"改了什么":实心点整体后移 1.2563 m 落到后轴线上。修前 12 路传感器全都偏前
    # 1.2563 m —— 这是 §P-M.10 的整条结论,留在图上省得下一个人又去"修正"回去。
    for m in mounts_carla.values():
        bx, by = to_screen(m[0] - NUS_EGO_ORIGIN_X, m[1])
        d.ellipse([bx - 7, by - 7, bx + 7, by + 7], outline=(120, 120, 130), width=3)
        d.line([bx - 10, by, bx + 10, by], fill=(120, 120, 130), width=2)
        d.line([bx, by - 10, bx, by + 10], fill=(120, 120, 130), width=2)

    # 比例尺(2 m)
    sx, sy = x0 + 16, y1 - 30
    d.line([sx, sy, sx + 2 * scale, sy], fill=INK, width=4)
    fonts.draw_text(d, (sx + scale, sy - 24), "2 m", size=18, fill=INK, anchor="mm")


def panel_legend(
    img: Image.Image,
    mounts_carla: dict[str, tuple[float, float, float]],
    az: dict[str, float],
    fov: dict[str, float],
    xy: tuple[int, int],
) -> None:
    """图例:色块 + **全名** + `az ± 半视场`(用户点名要的"传感器名称 + 方向与范围")。"""
    d = ImageDraw.Draw(img)
    x, y = xy
    fonts.draw_text(d, (x, y - 34), "传感器(颜色 = 俯视图/方位环)", size=21, fill=INK)
    for i, name in enumerate(mounts_carla):
        ly = y + i * 46
        col = CAM_COLOR.get(name, (80, 80, 80))
        d.rectangle([x, ly + 3, x + 18, ly + 17], fill=col, outline=INK)
        fonts.draw_text(d, (x + 26, ly), name, size=20, fill=INK)
        fonts.draw_text(d, (x + 26, ly + 23), f"{az[name]:+.1f}° ± {fov[name] / 2.0:.1f}°", size=18, fill=DIM)


def panel_azimuth_ring(
    img: Image.Image,
    az: dict[str, float],
    fov: dict[str, float],
    overlaps: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    centre: tuple[float, float],
    r_in: float = 110.0,
    r_out: float = 170.0,
    r_mark: float = 186.0,
    overlap_min_deg: float = 0.5,
) -> None:
    """方位环:逐相机扇区 + 重叠(橙弧,标度数)+ 盲区(红弧,标度数)+ 角度刻度。"""
    d = ImageDraw.Draw(img)
    for name, f in fov.items():
        a = az[name]
        _alpha_poly(
            img,
            _sector_poly(centre, a - f / 2.0, a + f / 2.0, r_in, r_out),
            CAM_COLOR.get(name, (80, 80, 80)),
            92,
        )
        d = ImageDraw.Draw(img)
        tx = centre[0] + (r_in + r_out) / 2.0 * _dir_screen(a)[0]
        ty = centre[1] + (r_in + r_out) / 2.0 * _dir_screen(a)[1]
        fonts.draw_text(d, (tx, ty), CAM_SHORT.get(name, "?"), size=24, fill=(255, 255, 255), anchor="mm")

    # 盲区 / 重叠:同一圈上互斥,不会互相遮盖
    for g in gaps:
        _alpha_poly(img, _sector_poly(centre, g["lo"], g["hi"], r_mark, r_mark + 14), GAP_COLOR, 255)
        d = ImageDraw.Draw(img)
        mid = g["lo"] + g["deg"] / 2.0
        tx = centre[0] + (r_mark + 52) * _dir_screen(mid)[0]
        ty = centre[1] + (r_mark + 52) * _dir_screen(mid)[1]
        fonts.draw_text(d, (tx, ty), f"盲区 {g['deg']:.1f}°", size=17, fill=GAP_COLOR, anchor="mm")
    for p in overlaps:
        if p["overlap_deg"] < overlap_min_deg or p["span"] is None:
            continue
        _alpha_poly(
            img, _sector_poly(centre, p["span"][0], p["span"][1], r_mark, r_mark + 14), OVERLAP_COLOR, 255
        )

    d = ImageDraw.Draw(img)
    d.ellipse(
        [centre[0] - r_out, centre[1] - r_out, centre[0] + r_out, centre[1] + r_out], outline=DIM, width=2
    )
    d.ellipse([centre[0] - r_in, centre[1] - r_in, centre[0] + r_in, centre[1] + r_in], outline=DIM, width=2)
    for a in range(0, 360, 30):
        ux, uy = _dir_screen(a)
        d.line(
            [
                centre[0] + (r_out + 4) * ux,
                centre[1] + (r_out + 4) * uy,
                centre[0] + (r_out + 14) * ux,
                centre[1] + (r_out + 14) * uy,
            ],
            fill=DIM,
            width=2,
        )
        # 刻度字**必须落在标注环带(橙/红,r_mark..r_mark+14)之外** —— 否则与重叠/盲区度数互压
        fonts.draw_text(
            d,
            (centre[0] + (r_out + 34) * ux, centre[1] + (r_out + 34) * uy),
            f"{a}°",
            size=17,
            fill=DIM,
            anchor="mm",
        )
    fonts.draw_text(d, (centre[0], centre[1]), "方位角", size=20, fill=INK, anchor="mm")
    fonts.draw_text(d, (centre[0], centre[1] + 22), "0°=车头 + = 左", size=17, fill=DIM, anchor="mm")
    fonts.draw_text(
        d,
        (centre[0], centre[1] + r_mark + 62),
        "橙 = 相邻重叠  红 = 盲区",
        size=18,
        fill=DIM,
        anchor="mm",
    )


def panel_table(
    img: Image.Image,
    mounts_carla: dict[str, tuple[float, float, float]],
    az: dict[str, float],
    fov: dict[str, float],
    box: tuple[int, int, int, int],
    coverage: dict[str, Any],
) -> None:
    """数字表:通道 / 挂点 / 方位角 / FoV / `az ± fov/2` / 覆盖区间(逐格显式列坐标)。"""
    d = ImageDraw.Draw(img)
    x0, y0, x1, _ = box
    cols = [x0, x0 + 250, x0 + 470, x0 + 600, x0 + 700]
    head = ["通道", "挂点 x / y / z (m)", "方位角", "FoV", "方向与范围"]
    fonts.draw_text(d, (x0 - 30, y0 - 44), "逐相机标定读数", size=22, fill=INK)
    for cx_, t in zip(cols, head, strict=True):
        fonts.draw_text(d, (cx_, y0), t, size=19, fill=INK)
    d.line([x0, y0 + 28, x1, y0 + 28], fill=DIM, width=2)
    ry = y0 + 40
    for name, m in mounts_carla.items():
        col = CAM_COLOR.get(name, (80, 80, 80))
        d.rectangle([x0 - 26, ry + 3, x0 - 8, ry + 17], fill=col, outline=INK)
        cells = [
            name,
            f"{m[0]:+.3f} / {m[1]:+.3f} / {m[2]:+.3f}",
            f"{az[name]:+.2f}°",
            f"{fov[name]:.1f}°",
            f"{az[name]:+.1f}° ± {fov[name] / 2.0:.1f}°",
        ]
        for cx_, t in zip(cols, cells, strict=True):
            fonts.draw_text(d, (cx_, ry), t, size=19, fill=INK if cx_ == cols[0] else (60, 60, 70))
        ry += 30
    ry += 10
    top = coverage["pairs"][0]
    fonts.draw_text(
        d,
        (x0 - 30, ry),
        f"方位覆盖 {coverage['covered_deg']:.2f}° / 360° ({coverage['coverage_frac'] * 100:.2f}%)"
        f"    盲区合计 {coverage['gap_total_deg']:.2f}°({len(coverage['gaps'])} 段)",
        size=20,
        fill=INK,
    )
    ry += 30
    fonts.draw_text(
        d,
        (x0 - 30, ry),
        f"最大重叠 {top['overlap_deg']:.1f}°({top['a']} ↔ {top['b']})    ⚠ 本表只算方位轴,不含地面可见距离",
        size=18,
        fill=DIM,
    )


def draw_rig_layout(
    calibs: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]],
    fov: dict[str, float],
    coverage: dict[str, Any],
    title: str,
    subtitle: str = "",
    size: tuple[int, int] = (1800, 1200),
) -> Image.Image:
    """自车 + 传感器标定配置图(俯视 + 图例 | 方位环 | 数字表)。返回 RGB 图,不落盘。

    **分区是硬坐标、互不重叠**(图例压锥 / 方位环压表都踩过):上排 = 俯视(左)+ 方位环(右),
    下排 = 数字表 + 覆盖汇总,页脚横贯底部。改画布尺寸时三个 `panel_*` 的位置都要跟着挪,
    单测 `test_rigviz.py` 用"任一非背景像素只属于一个分区"钉住。"""
    img = Image.new("RGBA", size, (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    mounts_carla = {n: m for n, (m, _) in nus_camera_rig(calibs).items()}
    az = {n: float(c["az_nus_deg"]) for n, c in coverage["cameras"].items()}

    fonts.draw_text(d, (40, 24), title, size=32, fill=INK)
    if subtitle:
        fonts.draw_text(d, (40, 66), subtitle, size=20, fill=DIM)

    panel_top_view(img, mounts_carla, az, fov, (30, 150, 520, 640))
    panel_legend(img, mounts_carla, az, fov, (30, 730))
    panel_azimuth_ring(img, az, fov, coverage["pairs"], coverage["gaps"], (1010, 400))
    panel_table(img, mounts_carla, az, fov, (620, 800, 1780, 1100), coverage)

    # 文本一律先过 `fonts.sanitize`(由 `fonts.draw_text` 内部调用):本机字体缺 U+2212(减号)
    # 等字形,`fonts.REPLACE` 把它们换成等价 ASCII —— **这不是降级,是设计**。故此处不报"缺字":
    # 判据是"最终画面里有没有豆腐块",而 `missing()` 只看字形有无、不看有无 ASCII 替身(会误报)。
    flines = fonts.wrap(FOOTER, FOOTER_SIZE, size[0] - 80)
    for i, line in enumerate(reversed(flines)):
        fonts.draw_text(d, (40, size[1] - 34 - i * FOOTER_LH), line, size=FOOTER_SIZE, fill=DIM)
    return img.convert("RGB")


def azimuth_of(
    calibs: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]],
    name: str,
) -> float:
    """光轴方位角(度,nuScenes 系)。与 `camera_rig.camera_azimuth_nus` 同一算式,
    此处保留一份**独立实现**以作交叉验证(单测断言两者相等)。"""
    boresight = quat_to_matrix(quat_normalize(calibs[name][1]))[:, 2]
    return math.degrees(math.atan2(boresight[1], boresight[0]))

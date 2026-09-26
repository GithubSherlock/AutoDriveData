"""实时可视化的共享件:多路 MJPEG 服务 / 拼图 / 环视 rig / 第三方视角 / 键盘。

从 `autodrivedata/sim/view_stream.py` 抽出,供 `view_stream.py` 与 `autodrivedata/sim/live_studio.py` 共用
(对标已有 `autodrivedata/sim/carla_common.py` 的先例:bin 只留 carla 编排,共享件单独成文件)。

**多路服务的口径**:一个 HTTP 端口 + 路径分路(`/stream/<name>`),不是 N 个端口。
理由:SSH 隧道只需转发一个端口,新增/删除一路流不改网络配置;`/` 出索引页把 N 路
`<img>` 拼在一页,浏览器端仍是一个连接面。

**rig 口径(两代,必须与权重的训练数据一致 —— 不是"越新越好")**:

| rig | 平移 | 偏航(BACK_LEFT / BACK_RIGHT) | 姿态 | 训练数据 |
|---|---|---|---|---|
| `nuscenes`(当前) | `SENSOR_MOUNTS[name]` 逐相机 | −108.6 / +110.8 | 逐相机 6DoF(pitch/roll 非 0) | `surround_p3` / `surround_town13` |
| `legacy`(早期) | 6 路**共用** `SENSOR_OFFSET`(1.2, 0, 1.65) | 235 / 125 | 仅偏航 | `surround_train` / `surround_drive` |

**`nuscenes` 曾名 `official`,且当时的值是镜像的**(2026-09-22 修):`SURROUND_CAMS` 把官方
方位角原样抄成正数,漏了 `yaw_carla = −az_nus`(见 `geometry.carla_yaw_to_nus_yaw`)⇒ 四个侧/后
相机左右互换(FRONT_LEFT 差 110.3°、BACK_LEFT 差 217.2°),pitch/roll 也被硬编码 0。前/后相机
因近自逆而"看着对",长期没暴露。真值现由 `autodrivedata/camera_rig.NUS_CAMERA_RIG` 单点提供
(官方四元数导出),采集器与本文件**同源**。

早期 `view_stream.build_maptr_rig` 给 6 路**共用** `SENSOR_OFFSET` 平移(与逐相机差最多 1.5 m),
且 BACK_LEFT/BACK_RIGHT 偏航与官方布局**恰好互换**。它**不是无条件 bug**:`maptr_ep512.pt`
就是在这套 rig 上训出来的,拿 nuscenes 喂它反而是错配。故本文件同时保留两套口径,
由 `--rig {auto,nuscenes,legacy}` 选(`auto` 按权重文件名查 `LEGACY_CKPTS`)。

**判据不看图**:`rig_mount_deviation()` 直接量"实挂位姿 vs **该 rig 规格**"的平移/偏航偏差,
修正前预期 ~1.5 m(旧口径下偏差反而 ~0)。**必须先 tick 再读**(传感器 `get_transform()`
在 tick 前是全 0 陈旧值,见 Plan.md 红线)。

**第三方视角**:非 `attach_to` 相机,每 tick 由 `follow_spectator()` 显式 `set_transform`。
不用 attach 的原因:attach 子 actor 的 `set_transform` 是**相对父位姿**的增量语义,
"归位"极易写错(§5.11 3DGS 采集踩过:不归位会绕空场地)。
"""

from __future__ import annotations

import html
import io
import math
import queue
import select
import sys
import termios
import threading
import time
import tty
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, cast

import carla
import numpy as np
from PIL import Image, ImageDraw

from autodrivedata.calib.camera_rig import NUS_CAMERA_RIG, NUS_WIDE_CAMERA_RIG
from autodrivedata.calib.core import CameraIntrinsics, world_to_img
from autodrivedata.gt.core import ActorBox, box_center_world, box_corners_world, box_to_gt_line
from autodrivedata.gt.export.nuscenes import NUS_CAMERA_HEIGHT, NUS_CAMERA_WIDTH, camera_fov
from autodrivedata.map.mapviz import calib_from_fov
from autodrivedata.sim.carla_common import CAM_ATTRS, SENSOR_MOUNTS, SENSOR_OFFSET, loc, rad
from autodrivedata.utils import fonts
from autodrivedata.utils.geometry import carla_rotation_matrix, rotation_matrix_to_carla, world_to_cam

if TYPE_CHECKING:  # pragma: no cover — 仅类型检查:torch/模型只在 --maptr 路径真需要
    import torch

    from autodrivedata.map.maptr.model import MapTR

# ---------------------------------------------------------------- rig 口径表

RIG_NUSCENES = "nuscenes"
RIG_LEGACY = "legacy"
# 自定义 wide rig(后移挂点 + 55/110/120 口径,见 `autodrivedata/camera_rig.py` 头注)。
# 与 legacy 同构:这里只登记**挂点与姿态**,FoV/内参由 `export/nuscenes` 按 rig 分派。
RIG_WIDE = "wide"

# 早期布局:6 路共用 SENSOR_OFFSET 平移 + 这套偏航(BACK_LEFT/RIGHT 与官方**互换**)
LEGACY_CAM_YAW: dict[str, float] = {
    "CAM_FRONT": 0.0,
    "CAM_FRONT_RIGHT": -55.0,
    "CAM_FRONT_LEFT": 55.0,
    "CAM_BACK": 180.0,
    "CAM_BACK_LEFT": 235.0,
    "CAM_BACK_RIGHT": 125.0,
}

# 权重文件名 → 它的训练数据用的是 legacy rig(见 `outputs/maptr_600/map_infos.json`:
# 帧 0-199 旧布局 / 200-599 官方布局)。新权重一律 nuscenes。
LEGACY_CKPTS = ("maptr_ep256", "maptr_ep512")

# legacy 画幅/FoV = KITTI 口径那套(**旧权重的口径,勿改**;由 CAM_ATTRS 导出,不手抄)
LEGACY_FRAME = (int(CAM_ATTRS["image_size_x"]), int(CAM_ATTRS["image_size_y"]))
LEGACY_FOV = float(CAM_ATTRS["fov"])


def rig_spec(
    rig: str,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, tuple[float, float, float]]]:
    """rig 名 → (逐相机平移, 逐相机姿态 (pitch,yaw,roll) 度)。`nuscenes` = 采集器口径。"""
    if rig == RIG_NUSCENES:
        return dict(SENSOR_MOUNTS), {name: rot for name, (_, rot) in NUS_CAMERA_RIG.items()}
    if rig == RIG_WIDE:  # 后三路挂点后移到车尾 + 轴方位角重排;**前三个与官方逐位相同**
        return (
            {name: m for name, (m, _) in NUS_WIDE_CAMERA_RIG.items()},
            {name: rot for name, (_, rot) in NUS_WIDE_CAMERA_RIG.items()},
        )
    shared = (SENSOR_OFFSET.location.x, SENSOR_OFFSET.location.y, SENSOR_OFFSET.location.z)
    return {name: shared for name in LEGACY_CAM_YAW}, {
        name: (0.0, yaw, 0.0) for name, yaw in LEGACY_CAM_YAW.items()
    }


def rig_frame(
    rig: str, width: int | None = None, height: int | None = None, fov: float | None = None
) -> tuple[int, int, dict[str, float]]:
    """rig 名 + 可选覆盖 → (画幅 w, h, **逐通道** fov 度)。**实时侧的画幅/FoV 唯一落点**
    (采集侧的对应物 = `collect_surround.SURROUND_CAM_ATTRS` + `export.nuscenes.NUS_CAMERA_FOV`)。

    - `nuscenes` / `wide`:**1600×900** + **逐通道** fov(由该 rig 的 K 导出)。
      逐通道是硬要求 —— 六路共用 90° 是"声明 ≠ 渲染"(§P-M.7)的第三种形态:模拟器里
      25° 的视野差会让第 i 路图与权重学过的语义错位,症状比挂点镜像更隐蔽。
    - `legacy`:1242×375 + 六路共用 90°(**旧权重口径,勿改** —— 它服务的是已废弃的
      `maptr_ep512`,不是"省带宽的小分辨率档")。

    `width/height` 显式传入只换光栅尺寸、**不换 FoV**(FoV 是相机属性,不是光栅属性),
    故仍逐通道;只有显式传 `fov` 才把六路抹平成一个值(纯显示路径)。
    """
    names = list(rig_spec(rig)[1])
    if rig in (RIG_NUSCENES, RIG_WIDE):
        w0, h0 = NUS_CAMERA_WIDTH, NUS_CAMERA_HEIGHT
        all_fov = camera_fov(rig)
        fovs = {name: float(all_fov[name]) for name in names}
    else:
        w0, h0 = LEGACY_FRAME
        fovs = dict.fromkeys(names, LEGACY_FOV)
    if fov is not None:
        fovs = dict.fromkeys(names, float(fov))
    return int(width or w0), int(height or h0), fovs


def resolve_rig(choice: str, ckpt: str | None) -> str:
    """`auto` 按权重文件名定 rig;显式给 `nuscenes`/`legacy` 时不猜。

    **为什么要按权重选**:外参必须与权重**训练时见过的一致**,否则第 i 路图与它学过的
    语义错位。实测 `maptr_ep512.pt` ← `surround_train`(legacy)、`maptr_600.pt` /
    `maptr_1000.pt` ← `surround_p3` + `surround_town13`(当时的 nuscenes 前身)。
    """
    if choice != "auto":
        return choice
    if ckpt and any(tag in Path(ckpt).stem for tag in LEGACY_CKPTS):
        return RIG_LEGACY
    return RIG_NUSCENES


# ---------------------------------------------------------------- 图像


def image_to_pil(image: carla.Image) -> Image.Image:
    """carla.Image(BGRA/BGR)→ PIL RGB(不落盘)。

    通道数由 raw_data 长度反推:carla pyi 桩缺 image.channels(T 类坑)。
    """
    ch = len(image.raw_data) // (image.width * image.height)
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, ch)
    return Image.fromarray(arr[:, :, [2, 1, 0]])


def encode_jpeg(img: Image.Image, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


HUD_H = 16  # 条带高(px)—— 第二行 HUD 传 y=HUD_H
HUD_FONT_SIZE = 13  # 字号:13 px 下 ink 高 ~13 px,恰好落进 HUD_H 条带

# 拼图**格内**的相机名角标(不额外占画布高度)。这三个常量是给测试用的契约:
# `TILE_LABEL_BOTTOM` 之下必须与源图逐像素相同 —— 换字号/字体时它自动跟着变,
# 不再是散在测试里的魔数(曾经写死 20,换成真字体后角标下沿到 26,测试就会假失败)。
TILE_LABEL_XY = (6, 6)
TILE_LABEL_SIZE = 16
TILE_LABEL_BOTTOM = (
    TILE_LABEL_XY[1]
    + fonts.bbox(ImageDraw.Draw(Image.new("L", (1, 1))), "CAM_FRONT_LEFT", TILE_LABEL_SIZE)[3]
    + 1
)


def draw_hud(img: Image.Image, text: str, warn: bool = False, y: int = 0) -> Image.Image:
    """左上角 HUD 条;`warn=True` 转红底(用于"滞后无界"这类必须看见的故障)。

    `y` = 条带顶边的像素偏移(默认 0)。多行 HUD(如 `--calib` 的第二行)靠它叠加,
    不必为此再造一个函数;条带高 `HUD_H` px,故第二行传 16。

    底条宽度按**实测**文本宽度定:旧实现是 `7 * len(text) + 8`,那个 7 是 PIL 内置位图字体
    的经验字宽 —— 换真字体、或文本含中文(CJK 字宽 ≈ 2× ASCII)后常数必然错。
    **字体一律走 `autodrivedata.fonts`**:直接 `d.text(...)` 不传 `font=` 会用内置位图字体,
    中文整行画成豆腐块(见 [autodrivedata/utils/fonts.py](../autodrivedata/utils/fonts.py))。
    """
    d = ImageDraw.Draw(img)
    bg = (140, 0, 0) if warn else (0, 0, 0)
    d.rectangle([0, y, min(img.width, int(fonts.width(text, HUD_FONT_SIZE)) + 9), y + HUD_H], fill=bg)
    fonts.draw_text(d, (4, y + 1), text, size=HUD_FONT_SIZE, fill=(255, 255, 255))
    return img


def compose_grid(tiles: list[Image.Image], names: list[str], w: int, h: int, cols: int = 3) -> Image.Image:
    """等尺寸拼图:cols 列,行数由 tile 数决定(6 格 = 3×2,8 格 = 4×2 传 cols=4)。

    **每格必须是 `w`×`h`**:尺寸不符直接报错,不静默裁。PIL 的 `paste` 在源图大于目标
    框时**只贴左上角、超出部分无声丢弃**(见 `compose_rows` docstring 的踩坑记录)。
    需要混合尺寸(相机全幅 + 第三方 + BEV)请用 `compose_rows`。
    """
    rows = math.ceil(len(tiles) / cols)
    grid = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))
    d = ImageDraw.Draw(grid)
    for idx, (tile, name) in enumerate(zip(tiles, names, strict=True)):
        if tile.size != (w, h):
            raise ValueError(
                f"compose_grid: 第 {idx} 格 {name} 尺寸 {tile.size} ≠ 格 {w}×{h}"
                "(paste 会静默裁掉超出部分;混合尺寸改用 compose_rows)"
            )
        row, col = divmod(idx, cols)
        grid.paste(tile, (col * w, row * h))
        fonts.draw_text(
            d,
            (col * w + TILE_LABEL_XY[0], row * h + TILE_LABEL_XY[1]),
            name,
            size=TILE_LABEL_SIZE,
            fill=(255, 255, 0),
        )
    return grid


def compose_rows(
    rows: list[list[tuple[str, Image.Image]]],
    bg: tuple[int, int, int] = (0, 0, 0),
    center: bool = True,
) -> Image.Image:
    """按行拼图:**每格按自身像素原样摆**,不统一尺寸、不缩放、不裁剪。

    行高 = 该行最高的一格;行宽 = 该行各格宽之和;整幅宽 = 最宽的那行,窄行居中
    (也可以左对齐)。名称标签画在各格左上角。

    为什么需要它(实测踩坑):`compose_grid` 要求所有格同尺寸,调用方为了把 6 路
    1242×375 相机图 + 第三方 640×360 + BEV 420×420 塞进一个 4×2 网格,只好先把大图
    缩到小格;而一旦漏缩(或想"少缩一点"),`Image.paste` 在源图大于目标框时**不报错、
    只贴左上角**——实测把 CAM_FRONT 的 1242×375 裁成左上角 621×187,右半 + 下半全丢,
    而**下半正是地面**(用户报告"6 视角 FoV 缩得看不到地面")。数值判据:拼图格与
    「源图左上角裁剪」平均绝对差 0.13,与「整幅缩放」差 66.2 ⇒ 是裁剪不是缩放。

    ⇒ 要"不为了整齐而缩减像素尺寸"就用本函数:每格保持原生分辨率。
    """
    if not rows or any(not row for row in rows):
        raise ValueError("compose_rows: rows 不能为空,且不允许空行")
    widths = [sum(t.width for _, t in row) for row in rows]
    heights = [max(t.height for _, t in row) for row in rows]
    total_w, total_h = max(widths), sum(heights)
    canvas = Image.new("RGB", (total_w, total_h), bg)
    d = ImageDraw.Draw(canvas)
    y = 0
    for r, row in enumerate(rows):
        x = (total_w - widths[r]) // 2 if center else 0
        for name, tile in row:
            canvas.paste(tile, (x, y))
            if name:
                fonts.draw_text(
                    d,
                    (x + TILE_LABEL_XY[0], y + TILE_LABEL_XY[1]),
                    name,
                    size=TILE_LABEL_SIZE,
                    fill=(255, 255, 0),
                )
            x += tile.width
        y += heights[r]
    return canvas


def dump_pair(path: str, raw: Image.Image, over: Image.Image) -> None:
    """落盘同一帧的 raw + overlay → 差集 = 真实绘制像素(数值诊断)。

    场景自带绿色植被/黄色标线,直接数颜色会误判 overlay 是否画上;
    两帧做差才排得掉干扰。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw.save(p.with_name(f"{p.stem}_raw{p.suffix}"))
    over.save(p)
    print(f"[dump] {p} + {p.stem}_raw{p.suffix}(差集诊断用)")


# ---------------------------------------------------------------- 多路 MJPEG 服务


class FrameSlot:
    """最新帧槽:MJPEG 服务线程只读最新帧,不排队 → 客户端永远看最新画面。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._seq = 0

    def publish(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._seq += 1

    def read(self, last_seq: int) -> tuple[bytes | None, int]:
        with self._lock:
            if self._seq == last_seq:
                return None, last_seq
            return self._jpeg, self._seq


def index_page(names: list[str], port: int) -> bytes:
    """索引页:N 路 `<img>` 网格(每路独立 `<img src='/stream/<name>'>`)。"""
    cells = "".join(
        f"<figure style='margin:0'><img src='/stream/{n}' style='width:100%;display:block'>"
        f"<figcaption style='color:#888;font:11px monospace'>{n}</figcaption></figure>"
        for n in names
    )
    return (
        "<!doctype html><meta charset='utf-8'><title>AutoDriveData 实时视图</title>"
        "<body style='margin:0;background:#111;color:#666;font:12px monospace'>"
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:2px'>"
        f"{cells}</div>"
        f"<p style='padding:4px'>共 {len(names)} 路 @ 127.0.0.1:{port}</p>"
    ).encode()


def start_server(slots: dict[str, FrameSlot], port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """单端口多路服务:`/` 索引页、`/stream/<name>` 取该路 MJPEG。"""
    page = index_page(list(slots), port)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            del format, args  # 静音 access log(5fps × N 路刷屏);签名照基类(参数名 format)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            name = self.path[len("/stream/") :] if self.path.startswith("/stream/") else None
            slot = slots.get(name) if name else None
            if slot is None:
                # ⚠️ 中文只能进 `explain`(**body**,UTF-8),**绝不能进 `message`** ——
                # 后者会被 `send_response` 拼进 HTTP **状态行**,而 `http.server` 用
                # **latin-1** 编码状态行 ⇒ `UnicodeEncodeError` 抛在请求线程里,
                # 客户端收到的是「连接被重置」而不是 404(实测:curl 得 0 字节)。
                # `name` 来自 URL,进 HTML 前必须转义。
                self.send_error(404, explain=f"未知流 {html.escape(name or '')!r};可选: {', '.join(slots)}")
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            while True:
                jpeg, seq = slot.read(last)
                if jpeg is None:
                    time.sleep(0.02)
                    continue
                last = seq
                try:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break  # 客户端断开

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def drain(q: queue.Queue) -> carla.Image:
    """取最新一帧(丢弃积压),防渲染慢于 tick 时画面滞后。"""
    img: carla.Image = q.get(timeout=10)
    while not q.empty():
        img = q.get_nowait()
    return img


# ---------------------------------------------------------------- GT overlay

MAX_DISTANCE = 65.0  # 与采集器 GT 口径一致(Plan.md 红线:远距无点框剔除)

CLASS_COLOR = {
    "Car": (0, 255, 80),
    "Pedestrian": (0, 220, 255),
    "Cyclist": (255, 220, 0),
    "Truck": (255, 140, 0),
    "Van": (255, 140, 0),
    "Misc": (170, 170, 170),
}


def actor_box(a: carla.Actor) -> ActorBox:
    """单个 carla actor → 纯值 ActorBox(GT 框来源)。"""
    bb = a.bounding_box
    t = a.get_transform()
    return ActorBox(
        type_id=a.type_id,
        extent=(bb.extent.x, bb.extent.y, bb.extent.z),
        location=(bb.location.x, bb.location.y, bb.location.z),
        rotation=rad(bb.rotation),
        actor_location=loc(t),
        actor_rotation=rad(t.rotation),
    )


def actor_boxes(world: carla.World) -> list[ActorBox]:
    """世界内全部 vehicle/walker → 纯值 ActorBox 列表。"""
    return [
        actor_box(a)
        for a in world.get_actors()
        if a.type_id.startswith("vehicle") or a.type_id.startswith("walker")
    ]


def overlay_gt(
    img: Image.Image,
    boxes: list[ActorBox],
    cam_loc: tuple[float, float, float],
    cam_rot: tuple[float, float, float],
    k: CameraIntrinsics,
) -> Image.Image:
    """GT 框:走 box_to_gt_line(label_2 同一口径)→ 所见即导出。"""
    d = ImageDraw.Draw(img)
    for box in boxes:
        line = box_to_gt_line(box, cam_loc, cam_rot, k, max_distance=MAX_DISTANCE)
        if not line:
            continue
        p = line.split()
        cls = p[0]
        x1, y1, x2, y2 = (float(v) for v in p[4:8])
        dist = math.hypot(float(p[11]), float(p[13]))  # 相机系 (x, z)
        col = CLASS_COLOR.get(cls, CLASS_COLOR["Misc"])
        d.rectangle([x1, y1, x2, y2], outline=col, width=2)
        fonts.draw_text(d, (x1 + 2, max(0.0, y1 - 14)), f"{cls} {dist:.0f}m", size=13, fill=col)
    return img


# ---------------------------------------------------------------- 环视 rig


def build_surround_rig(
    world: carla.World,
    ego: carla.Vehicle,
    width: int | None = None,
    height: int | None = None,
    fov: float | None = None,
    rig: str = RIG_NUSCENES,
    kind: str = "rgb",
) -> dict[str, tuple[carla.Sensor, CameraIntrinsics]]:
    """环视 rig:挂点/内参/分辨率与 `collect_surround.py` **逐字段**对齐。

    模型是按"相机名 → 该名挂点"记语义的:name→挂点与训练不一致 = 第 i 路图与它学过的
    第 i 路语义错位(**侧后相机镜像**是最隐蔽的一种:早期 235/125 与官方 −108.6/+110.8
    恰好互换)。故这里只认 `rig_spec()` 的两处定义。

    `rig` 选口径:默认 `nuscenes` = 当前采集器(1600×900 + **逐通道** fov);
    喂旧权重(`maptr_ep512.pt` 一类)必须传 `legacy`(1242×375 + 六路共用 90°),
    否则图与权重错配(见模块头注的对照表)。画幅/FoV 一律走 `rig_frame`,不在此另立。

    `width/height` 缺省 = 该 rig 的原生画幅;纯显示路径(`view_stream --view grid6`
    不带 `--maptr`)可传小分辨率省带宽 —— 挂点与 FoV 不受影响。

    `kind` 是相机蓝图后缀(`rgb` / `depth` / `instance_segmentation`)。**实时标定槽**
    (`live_studio --calib`)要挂 `depth` 并与 RGB 槽**逐像素对齐**,故它必须传与 RGB
    完全相同的 `width/height/fov` —— 内参、挂点、分辨率三者任一不同,着色点都会错位。
    """
    w, h, fovs = rig_frame(rig, width, height, fov)
    mounts, rots = rig_spec(rig)
    bp_lib = world.get_blueprint_library()
    cams: dict[str, tuple[carla.Sensor, CameraIntrinsics]] = {}
    for name, (pitch, yaw, roll) in rots.items():
        x, y, z = mounts[name]
        # 逐相机蓝图:每路设自己的 fov(六路共用一个 bp 就是"声明 ≠ 渲染",见 rig_frame)
        bp = bp_lib.find(f"sensor.camera.{kind}")
        bp.set_attribute("image_size_x", str(w))
        bp.set_attribute("image_size_y", str(h))
        bp.set_attribute("fov", f"{fovs[name]:.6f}")
        tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))
        k = CameraIntrinsics(width=w, height=h, fov_h_deg=fovs[name])
        cams[name] = (cast(carla.Sensor, world.spawn_actor(bp, tf, attach_to=ego)), k)
    return cams


def surround_calibs(
    rig: str = RIG_NUSCENES, width: int | None = None, height: int | None = None
) -> dict[str, dict]:
    """环视 rig 的 calib(与训练 infos 同结构:内参 3×3 + sensor2ego 6 元组)。

    内参走 `rig_frame` **逐通道**(render 与落盘同源,不允许"六路一张 K");
    `width/height` 必须与 `build_surround_rig` 传的**相同**,否则画幅与 K 分叉。
    sensor2ego = `[x, y, z, yaw, pitch, roll]`(infos 口径,**注意不是 rig_spec 的
    (pitch,yaw,roll)**)—— 逐字段对齐权重训练数据里的那份,不是"最新那份"。
    """
    w, h, fovs = rig_frame(rig, width, height, None)
    mounts, rots = rig_spec(rig)
    return {
        name: {
            "sensor2ego": [*mounts[name], rot[1], rot[0], rot[2]],
            "intrinsic": calib_from_fov(w, h, fovs[name])["intrinsic"],
        }
        for name, rot in rots.items()
    }


def mount_deviation_of(
    actors: Mapping[str, carla.Actor],
    ego: carla.Vehicle,
    mounts: Mapping[str, tuple[float, float, float]],
    rots: Mapping[str, tuple[float, float, float]],
) -> tuple[float, float]:
    """实挂传感器相对 ego 的平移/偏航 vs **给定规格**的最大偏差 → (米, 度)。**调用前必须 tick**。

    传感器 `get_transform()` 只在 tick 后刷新:tick 前读到的是全 0 陈旧值,
    会假报 ~179.8°(= CAM_BACK 规格 180 − ego 固有 yaw 0.159,见 Plan.md 红线)。

    **矩阵顺序是 ego⁻¹·cam**(把世界系的相机位姿换算到 ego 系);写成 `cam·ego⁻¹`
    会把 ego 的**世界坐标**混进平移块,ego 离原点越远偏差越大(实测 138 m,偏航却
    恰好仍为 0 —— 只看偏航自检会漏掉,故本函数同时报平移)。

    偏航对账**用规格旋转阵自身解出的 yaw**,不是规格里那个 yaw 字段——nuscenes rig
    带非零 pitch/roll,yaw 字段与矩阵解出的 yaw 在 Rz·Ry·Rx 组合下**不是**同一个数。

    相机与雷达共用本函数(两者都是 attach 到 ego 的刚体,判据同形);`rots` 一律是
    CARLA 口径的 `(pitch, yaw, roll)` 度。
    """
    inv_ego = np.array(ego.get_transform().get_inverse_matrix(), dtype=np.float64)
    dev_t = dev_y = 0.0
    for name, actor in actors.items():
        T = inv_ego @ np.array(actor.get_transform().get_matrix(), dtype=np.float64)
        spec = np.array(mounts[name], dtype=np.float64)
        dev_t = max(dev_t, float(np.linalg.norm(T[:3, 3] - spec)))
        pitch, yaw, roll = (math.radians(v) for v in rots[name])
        r_spec = carla_rotation_matrix((pitch, yaw, roll))
        yaw_spec = math.degrees(math.atan2(r_spec[1, 0], r_spec[0, 0]))
        yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
        dev_y = max(dev_y, abs((yaw - yaw_spec + 180.0) % 360.0 - 180.0))
    return dev_t, dev_y


def rig_mount_deviation(
    cams: dict[str, tuple[carla.Sensor, CameraIntrinsics]],
    ego: carla.Vehicle,
    rig: str = RIG_NUSCENES,
) -> tuple[float, float]:
    """实挂相机 vs **该 rig 规格** 的 (平移米, 偏航度) 最大偏差。**调用前必须 tick**。

    薄封装:`rig_spec(rig)` 取规格后委托 `mount_deviation_of`(相机与雷达共用同一实现,
    避免验收脚本里再抄一遍 `ego⁻¹·cam` 的矩阵顺序)。
    """
    mounts, rots = rig_spec(rig)
    return mount_deviation_of({n: c for n, (c, _) in cams.items()}, ego, mounts, rots)


def build_cameras(
    world: carla.World, ego: carla.Vehicle, view: str, width: int, height: int
) -> dict[str, tuple[carla.Sensor, CameraIntrinsics]]:
    """单相机视角(`follow` 跟车 / `top` 俯视);`grid6` 走 `build_surround_rig`。"""
    bp_lib = world.get_blueprint_library()
    k = CameraIntrinsics(width=width, height=height, fov_h_deg=90.0)
    cams: dict[str, tuple[carla.Sensor, CameraIntrinsics]] = {}

    def add(name: str, tf: carla.Transform) -> None:
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", "90")
        cams[name] = (cast(carla.Sensor, world.spawn_actor(bp, tf, attach_to=ego)), k)

    if view == "top":
        # 60m 高:覆盖 ±34m(纵向)× ±60m(横向),容得下 A/B 布局的 20~62m 静置车
        add(
            "TOP",
            carla.Transform(carla.Location(x=0.0, y=0.0, z=60.0), carla.Rotation(pitch=-90.0)),
        )
    else:  # follow
        add(
            "FOLLOW",
            carla.Transform(carla.Location(x=-9.0, y=0.0, z=4.5), carla.Rotation(pitch=-12.0)),
        )
    return cams


def ego_box_pixels(
    box: ActorBox,
    cam_loc: tuple[float, float, float],
    cam_rot: tuple[float, float, float],
    k: CameraIntrinsics,
) -> tuple[float, float, float, float] | None:
    """box 的 8 角点投影到该相机 → 像素框 (x1,y1,x2,y2);无角点可见返回 None。

    供第三方视角自检「ego 在画面内」用:`box_to_gt_line` 在"全落图外"时直接剔除、
    拿不到中间量,而自检需要的就是**框在哪**(中心/宽度占比),故单开一个。
    """
    corners = box_corners_world(box)
    uv = [world_to_img(tuple(p), cam_loc, cam_rot, k) for p in corners]
    vis = [q for q in uv if q is not None]
    if not vis:
        return None
    xs = [q[0] for q in vis]
    ys = [q[1] for q in vis]
    return min(xs), min(ys), max(xs), max(ys)


def ego_box_deviation(
    box: ActorBox,
    cam_loc: tuple[float, float, float],
    cam_rot: tuple[float, float, float],
    k: CameraIntrinsics,
) -> tuple[float, float, float]:
    """第三方视角自检量 → (框中心 u 相对画幅中心的偏移比, 框宽占画幅比, 框中心深度 m)。

    判据不靠目检:中心偏移比 ∈ [0, 0.5] 且宽度占比 ∈ [0.05, 0.6] 才算"ego 在画面内
    且不是贴脸/针尖"。全落图外时返回 (inf, 0, inf) —— 明确失败而非静默通过。
    """
    rect = ego_box_pixels(box, cam_loc, cam_rot, k)
    if rect is None:
        return float("inf"), 0.0, float("inf")
    x1, _, x2, _ = rect
    c = g_world_to_cam(box_center_world(box), cam_loc, cam_rot)
    return abs((x1 + x2) / 2.0 - k.width / 2.0) / k.width, (x2 - x1) / k.width, float(c[2])


def g_world_to_cam(
    world_pt: np.ndarray, cam_loc: tuple[float, float, float], cam_rot: tuple[float, float, float]
) -> np.ndarray:
    """世界点 → 相机系(x 右 / y 下 / z 前);薄封装,免在 bin 里再 import geometry。"""
    return world_to_cam(np.asarray(world_pt, dtype=np.float64)[None], cam_loc, cam_rot)[0]


# ---------------------------------------------------------------- 第三方视角

# 默认机位:ego 后 9 m / 高 5 m / 俯角 12°(与 follow 同构,但**不 attach**)
SPECTATOR_OFFSET = carla.Transform(carla.Location(x=-9.0, y=0.0, z=5.0), carla.Rotation(pitch=-12.0))


def build_spectator(
    world: carla.World, offset: carla.Transform | None = None, width: int = 640, height: int = 360
) -> tuple[carla.Sensor, CameraIntrinsics]:
    """第三方视角相机:**不 attach**,由 `follow_spectator()` 每 tick 摆位。"""
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", "90")
    cam = cast(carla.Sensor, world.spawn_actor(bp, offset or SPECTATOR_OFFSET))
    return cam, CameraIntrinsics(width=width, height=height, fov_h_deg=90.0)


def follow_spectator(
    cam: carla.Sensor, ego: carla.Vehicle, offset: carla.Transform | None = None
) -> carla.Transform:
    """把第三方相机摆到 ego 位姿 ∘ 局部偏移处,返回实设位姿。

    手算复合(不走 attach):世界位姿 = ego 位姿 @ 局部偏移。
    旋转块走 `geometry.carla_rotation_matrix`(组合序 Rz·Ry·Rx,与 pycarla
    `Transform.get_matrix()` 旋转块逐元素一致,有 oracle 单测锁定)——**不要**自己按
    (roll,pitch,yaw) 拼矩阵:`carla_common.rad()` 的返回序是 (pitch, yaw, roll),正是
    该函数要的序,两者配套;自拼极易错序。
    """
    off = offset or SPECTATOR_OFFSET
    ego_t = ego.get_transform()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = carla_rotation_matrix(rad(ego_t.rotation))
    T[:3, 3] = loc(ego_t)
    O = np.eye(4, dtype=np.float64)
    O[:3, :3] = carla_rotation_matrix(rad(off.rotation))
    O[:3, 3] = loc(off)
    W = T @ O
    tf = carla.Transform(
        carla.Location(x=float(W[0, 3]), y=float(W[1, 3]), z=float(W[2, 3])),
        carla.Rotation(*np.degrees(_rotation_from_matrix(W[:3, :3])).tolist()),
    )
    cam.set_transform(tf)
    return tf


def _rotation_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rz·Ry·Rx 分解 → (pitch, yaw, roll) 弧度(与 `carla_common.rad` 同序)。

    实现落在 `geometry.rotation_matrix_to_carla`(纯值库,有往返单测);
    这里只做 ndarray 包装,免 bin 里再拼一遍三角函数。
    """
    return np.array(rotation_matrix_to_carla(R), dtype=np.float64)


# ---------------------------------------------------------------- MapTR(可选路径)


def load_maptr(ckpt: str, device: str | None = None) -> tuple[MapTR, torch.device]:
    """MapTR state_dict → eval 模式(与 `autodrivedata/map/eval_maptr.py` 同口径:num_vec/预处理都不改)。

    `torch`/`maptr_impl` 在此**延迟 import**:`live_common` 被 `drive_ego.py` 这类
    不用模型的路径 import,模块级拉 torch 会白等数秒。
    """
    import torch

    from autodrivedata.map.maptr.model import MapTR

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MapTR().to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    print(f"[maptr] {ckpt} @ {dev}(num_vec={model.num_vec})")
    return model, dev


def maptr_predict(
    model: MapTR,
    dev: torch.device,
    images: dict[str, Image.Image],
    ego_g: list[float],
    calibs: dict,
    thr: float,
) -> list[list[np.ndarray]]:
    """单帧 6 路环视 → 逐类预测折线(ego 系)。解码口径与 `autodrivedata/map/eval_maptr.py` 逐行一致。"""
    import torch
    from torchvision.transforms.functional import normalize, to_tensor

    from autodrivedata.map.maptr.dataset import IMAGENET_MEAN, IMAGENET_STD

    imgs = {n: normalize(to_tensor(t), IMAGENET_MEAN, IMAGENET_STD)[None].to(dev) for n, t in images.items()}
    pose = torch.tensor([ego_g], dtype=torch.float32, device=dev)
    with torch.no_grad():
        out, _ = model(imgs, pose, calibs)
    scores = torch.sigmoid(out["pred_logits"][0].float()).cpu().numpy()
    pts = out["pred_points"][0].float().cpu().numpy()
    preds: list[list[np.ndarray]] = []
    for c in range(model.num_classes):
        idx = slice(c * model.num_vec, (c + 1) * model.num_vec)
        preds.append(list(pts[idx][scores[idx, c + 1] > thr]))
    return preds


# ---------------------------------------------------------------- 键盘


class KeyboardState:
    """非阻塞键盘 → ego 控制状态机(从 `autodrivedata/sim/drive_ego.py` 抽出,行为不变)。

    tty 下 cbreak 模式单键即响应;stdin 为管道/重定向时按行读(测试路径)。
    `apply(ego)` 每 tick 调一次 —— sync 模式下控制命令在**下次 tick** 生效,必须持续喂。
    """

    KEYS = "w/s 油门刹车 · a/d 转向 · x 滑行 · 空格 手刹 · q 退出"

    def __init__(self) -> None:
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.handbrake = False
        self.quit = False
        self._is_tty = sys.stdin.isatty()
        self._old = termios.tcgetattr(sys.stdin.fileno()) if self._is_tty else None
        if self._is_tty:
            tty.setcbreak(sys.stdin.fileno())

    def close(self) -> None:
        """还原终端属性(必须在 finally 里调,否则退出后终端不回显)。"""
        if self._is_tty and self._old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)
            self._old = None

    def poll(self) -> None:
        """非阻塞读一键并更新状态;`quit` 置位表示用户按了 q。"""
        if self._is_tty:
            if not select.select([sys.stdin], [], [], 0.0)[0]:
                return
            ch = sys.stdin.read(1)
            ch = ch.lower() if ch else ""
        else:
            line = sys.stdin.readline()
            if not line:  # 管道 EOF
                self.quit = True
                return
            ch = line.strip().lower()[:1]
        if ch == "w":
            self.throttle, self.brake, self.handbrake = 0.8, 0.0, False
        elif ch == "s":
            self.throttle, self.brake, self.handbrake = 0.0, 0.8, False
        elif ch == "a":
            self.steer = -0.45
        elif ch == "d":
            self.steer = 0.45
        elif ch == "x":
            self.throttle, self.brake, self.steer, self.handbrake = 0.0, 0.0, 0.0, False
        elif ch == " ":
            self.brake, self.handbrake = 1.0, True
        elif ch == "q":
            self.quit = True

    def apply(self, ego: carla.Vehicle) -> None:
        ego.apply_control(
            carla.VehicleControl(
                throttle=self.throttle, steer=self.steer, brake=self.brake, hand_brake=self.handbrake
            )
        )

    def hud(self, ego: carla.Vehicle) -> str:
        v = ego.get_velocity()
        kmh = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z) * 3.6
        return (
            f"{kmh:5.1f}km/h 油门{self.throttle:.1f} 刹车{self.brake:.1f} "
            f"转向{self.steer:+.2f} 手刹{int(self.handbrake)}"
        )

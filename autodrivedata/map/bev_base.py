"""BEV 底图:把**同一段序列的 LiDAR / Radar 点云**画成矢量图下方的上下文层。

## ⚠️ 先读这条:底图**不是**"多模态融合"

MapTR/MapQR 是**纯相机**模型 —— LiDAR/Radar **一个点都没进网络**。本模块只做一件事:
把同一帧(或整段)的点云按位姿投到**与矢量图完全相同的那个平面系**里,叠在下面当参考。
看图时按"这里真的有一面墙/一辆车"去读,不要读成"模型用了点云"。

## 坐标系(三层,错一层就是"图能画出来但整体偏")

| 层 | 定义 | 谁定的 |
|---|---|---|
| `B`(**BEV ego 系**) | x 前 / y 左 / z 上,原点 = **CARLA actor 原点(车身中点)** | MapTR 的 BEV 输出系;`mapviz.bev_px` 与 GT 矢量都用它 |
| velodyne | 同上,**原点 = LiDAR** | `carla_lidar_to_velodyne` 落盘时的约定 |
| nus sensor | 同上,**原点 = 该雷达** | `detections_to_nus18` 的输出 |

**B 的原点为什么是车身中点而不是后轴**:B2 线(`collect_surround` / 本采集器)的
`ego_pose.json` 与 `assemble_maptr` 的 `ego2global` 都直接取 `ego.get_transform()`,
那是 CARLA actor 原点。nuScenes 的"后轴原点"是**另一条线**(`collect_nus` → devkit)的口径,
两者靠 `geometry.NUS_EGO_ORIGIN_X` 换算 —— 本模块把这一步**显式写出来**,不猜。

推导(两步,`lidar_ego` / `radar_ego`):
- LiDAR:`p_B = p_velo + LIDAR_LEVER`(杆臂取 `slam_eval.LIDAR_LEVER`,**不手抄**)
- Radar:`p_B = nus_ego_to_bev(R_yaw · p_radar + t_nus)`,`(t_nus, yaw_nus)` 取
  `gt.export.nuscenes.NUS_RADAR_OFFSETS`(**官方值,不手抄**)

**自证**(`tests/map/test_bev_base.py`):两条链各自把"传感器原点"映回 B 系后,必须与
**已验证的表**逐分量相等 —— `carla_common.SENSOR_OFFSET` 与 `NUS_RADAR_MOUNTS_CARLA`(y 翻号)。
写错系/错符号时散点图看着仍然"像那么回事"(本项目的"yaw≈0 的相机看着正常"同款坑),
故判据必须是**数值对表**,不是目检。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from autodrivedata.gt.export.nuscenes import NUS_RADAR_OFFSETS
from autodrivedata.map.mapviz import Px
from autodrivedata.perception.radar import RADAR_NUS_FIELDS, RADAR_STRUCT
from autodrivedata.slam.slam_eval import LIDAR_LEVER
from autodrivedata.utils.geometry import NUS_EGO_ORIGIN_X

__all__ = ["Px"]  # 再导出:调用方从底图模块拿 Px 也成立(定义在 mapviz,见那里的头注)

#: 底图三色。**比矢量色暗一档**才对:底图是上下文,抢了主色的对比度就本末倒置。
LIDAR_GROUND_COLOR = (62, 66, 76)  # 地面点(近灰蓝)
LIDAR_OBJECT_COLOR = (124, 128, 140)  # 离地点(墙/车/杆,亮一档)
RADAR_COLOR = (255, 138, 60)  # 雷达(橙;点稀疏,必须显眼才看得见)

#: 离地切分高度(米,**ego 系**)—— 低于它算地面。
#:
#: 取 0.4 是**实测**定的,不是拍的。ego 系里地面就是 z≈0(actor 原点在地面,LiDAR 挂高
#: 1.65 已在 `lidar_ego` 里加回)。实测一帧 63937 点的 z 直方图(2026-09-30,`dual_run/seq`):
#:
#: | 区间 | 点数 | |
#: |---|---|---|
#: | [0.0, 0.2) | **38140** | 路面 |
#: | [0.2, 0.4) | 636 | ← 谷 |
#: | [0.4, 0.6) | 617 | ← 谷 |
#: | [0.6, 1.0) | 1310 | 路缘/车底 |
#: | [1.0, 1.5) | 10399 | 车顶/树干 |
#: | ≥1.5 | 12835 | 墙/树冠 |
#:
#: 两个数量级的**天然谷**落在 0.2–0.6,0.4 在其中间 ⇒ 分层的稳健性不依赖精确阈值
#: (路面有坡度、车身有俯仰时它都不会翻)。原来 docstring 写的 "z ∈ [−1.7, −1.3]"
#: 是 **LiDAR 系**的值,在 ego 系里是错的 —— 已按实测订正。
OBJECT_Z_MIN = 0.4


# ------------------------------------------------------------------ 读盘


def load_lidar_velo(root: Path, frame: int) -> np.ndarray:
    """`training/velodyne/{frame:06d}.bin` → (N,4) float32(**velodyne 约定**)。

    无文件返回 (0,4) —— 采集被中断的序列尾部会是这种,静默跳过好过崩在可视化上。
    """
    p = Path(root) / "training" / "velodyne" / f"{frame:06d}.bin"
    if not p.exists():
        return np.zeros((0, 4), dtype=np.float32)
    return np.fromfile(p, dtype=np.float32).reshape(-1, 4)


def load_radar_nus(root: Path, frame: int, channel: str) -> np.ndarray:
    """`samples/{channel}/{frame:06d}.pcd` → (M,18) float32(**nus sensor 系**)。

    走 `radar.RADAR_STRUCT` 逐点解包 —— **与写盘侧同一个 struct**,不另写一份字段表
    (两侧各写一份迟早漂)。

    ★ 空点云的 devkit 编码是"WIDTH 1 + 单点全 NaN"(`nus18_to_pcd`),**在这里就滤成 0 点**:
    官方 devkit 的 `from_file` 读到首点 NaN 同样返回 `(18,0)`。放在读侧而不是消费侧,
    是为了让"空"这件事对**所有**消费者一致 —— 漏一个消费方就会多画一个假点在原点。
    (2026-09-30 单测实测:原先只在 `radar_ego` 里滤,`load_radar_nus` 直接返回 1 个 NaN 点。)
    """
    p = Path(root) / "samples" / channel / f"{frame:06d}.pcd"
    if not p.exists():
        return np.zeros((0, 18), dtype=np.float32)
    blob = p.read_bytes()
    marker = b"DATA binary\n"
    i = blob.find(marker)
    if i < 0:
        return np.zeros((0, 18), dtype=np.float32)
    body = blob[i + len(marker) :]
    size = RADAR_STRUCT.size
    n = len(body) // size
    arr = np.array([RADAR_STRUCT.unpack_from(body, k * size) for k in range(n)], dtype=np.float32)
    return _drop_nan(arr.reshape(-1, len(RADAR_NUS_FIELDS)))


# ------------------------------------------------------------------ 换算


def _nus_ego_to_bev(pts: np.ndarray) -> np.ndarray:
    """nus ego 系(y 左,**原点=后轴**)→ B(y 左,原点=车身中点):只挪 x。

    两个系**同手性**(都是 y 左),故没有翻号 —— 翻号只出现在 `nus_mount_to_carla`
    那条给 CARLA spawn 用的路上。这里只有 `x += NUS_EGO_ORIGIN_X`。
    """
    out = np.asarray(pts, dtype=np.float64).reshape(-1, 3).copy()
    out[:, 0] += NUS_EGO_ORIGIN_X
    return out


def lidar_ego(root: Path, frame: int) -> np.ndarray:
    """该帧 LiDAR → B 系 (N,3)。杆臂取 `slam_eval.LIDAR_LEVER`(与 SLAM 评估同一常量)。"""
    pts = load_lidar_velo(root, frame)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return pts[:, :3].astype(np.float64) + np.asarray(LIDAR_LEVER, dtype=np.float64)


def radar_ego(arr18: np.ndarray, channel: str) -> np.ndarray:
    """`(M,18)` nus sensor 系 → B 系 (M,3):先 `R_yaw·p + t`,再 `nus_ego_to_bev`。

    这里**再滤一次 NaN**:读盘入口 `load_radar_nus` 已经滤过,但本函数是公开的,
    调用方可能自己拼数组(测试就是这么用的)。NaN 一旦漏进去,`R_yaw·NaN` 静默传播,
    最后只有一个"画不出来的点",没有报错 —— 两道滤网的代价是零。
    """
    arr = _drop_nan(np.asarray(arr18, dtype=np.float64).reshape(-1, len(RADAR_NUS_FIELDS)))
    if len(arr) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    t, yaw_nus = NUS_RADAR_OFFSETS[channel]
    # ★★ **旋转角必须取 `−yaw_nus`**(2026-10-01 实测裁定,此前镜像了整整一版)。
    #   判据不是"看着对不对",而是**与 CARLA 自己的 `sensor.get_transform()` 比位移** ——
    #   把同一批检测同时喂两条路送进世界系,量中位位移(米):
    #
    #   | 通道 | `+yaw_nus`(旧) | `−yaw_nus`(今) |
    #   |---|---|---|
    #   | RADAR_FRONT | 1.18 m | 1.14 m |
    #   | RADAR_FRONT_LEFT | **44.13 m** | 2.33 m |
    #   | RADAR_FRONT_RIGHT | **24.36 m** | 2.30 m |
    #   | RADAR_BACK_LEFT | 2.10 m | 0.62 m |
    #   | RADAR_BACK_RIGHT | 3.16 m | 2.34 m |
    #
    #   为什么能活一版:① `RADAR_FRONT` 的 yaw 只有 **0.2°** ⇒ 翻符号几乎不动;
    #   ② 原先记这条链的**唯一**判据是"雷达点落在 LiDAR 表面上",而 LiDAR 是 11.7 万点的
    #   **密云** —— 44 m 的位移照样落在"某个"表面附近(实测 ≤1 m 占比 52%,**看着完全正常**)。
    #   密度越高的参考越没判别力 ⇒ 正确仲裁是 CARLA 自己的位姿,它精确且不依赖任何表。
    #   (本仓"yaw≈0 的相机看着正常"那个坑的第四次现形。)
    yaw = -yaw_nus
    c, s = np.cos(yaw), np.sin(yaw)
    xyz = arr[:, :3]
    x = xyz[:, 0] * c - xyz[:, 1] * s + t[0]
    y = xyz[:, 0] * s + xyz[:, 1] * c + t[1]
    z = xyz[:, 2] + t[2]
    return _nus_ego_to_bev(np.stack([x, y, z], axis=1))


def _drop_nan(arr: np.ndarray) -> np.ndarray:
    """空点云的 devkit 编码 = 单点全 NaN(见 `radar.nus18_to_pcd`)。"""
    if len(arr) == 0:
        return arr
    return arr[~np.isnan(arr).any(axis=1)]


def frame_points(root: Path, frame: int, channels: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """一帧的 `(lidar_B (N,3), radar_B (M,3))`。`channels` 为空/None = 不读雷达。"""
    lid = lidar_ego(root, frame)
    rad = (
        np.zeros((0, 3), dtype=np.float64)
        if not channels
        else np.concatenate([radar_ego(load_radar_nus(root, frame, ch), ch) for ch in channels], axis=0)
    )
    return lid, rad


def to_world(pts_B: np.ndarray, ego: list[float]) -> np.ndarray:
    """B 系 (N,3) → 世界系 (N,3)。**与 `stitch.place` / `frame_to_world` 同式**
    (`Rz(yaw)·p + t`,z 原样平移)—— 单一几何实现:矢量与底图必须落在同一张图上。"""
    arr = np.asarray(pts_B, dtype=np.float64).reshape(-1, 3)
    if len(arr) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    th = np.radians(ego[3])
    c, s = np.cos(th), np.sin(th)
    return np.stack(
        [
            ego[0] + arr[:, 0] * c - arr[:, 1] * s,
            ego[1] + arr[:, 0] * s + arr[:, 1] * c,
            ego[2] + arr[:, 2],
        ],
        axis=1,
    )


# ------------------------------------------------------------------ 绘制


def _paint(draw, xyz: np.ndarray, px: Px, *, colors: tuple = (LIDAR_GROUND_COLOR, LIDAR_OBJECT_COLOR)) -> int:
    """按高度分两层批量画点,返回**画出的点数**(0 = 没画上 ⇒ 数值自证,不靠目检)。"""
    if len(xyz) == 0:
        return 0
    pix = px.arr(xyz[:, :2])
    low = xyz[:, 2] < OBJECT_Z_MIN
    n = 0
    for mask, col in ((low, colors[0]), (~low, colors[1])):
        if not mask.any():
            continue
        draw.point([(float(a), float(b)) for a, b in pix[mask]], fill=col)
        n += int(mask.sum())
    return n


def scatter_ego(
    draw, pts_B: np.ndarray, px: Px, window: tuple[tuple[float, float], tuple[float, float]]
) -> int:
    """ego 系点云 → 像素,返回画出的点数。**只画窗口内的点** —— 窗外点提交给 PIL 会被
    静默裁掉,"画了没画上"就分不清。"""
    arr = np.asarray(pts_B, dtype=np.float64).reshape(-1, 3)
    if len(arr) == 0:
        return 0
    (x0, x1), (y0, y1) = window
    m = (arr[:, 0] >= x0) & (arr[:, 0] <= x1) & (arr[:, 1] >= y0) & (arr[:, 1] <= y1)
    return _paint(draw, arr[m], px)


def scatter_world(draw, pts_world: np.ndarray, px: Px) -> int:
    """世界系点云 → 像素(拼接大图用)。窗口由**调用方**按面板包络保证(全局范围,
    再套 ego 窗口会把它切碎)。"""
    return _paint(draw, np.asarray(pts_world, dtype=np.float64).reshape(-1, 3), px)


def scatter_radar(
    draw,
    pts: np.ndarray,
    px: Px,
    window: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> int:
    """雷达散点(橙)。**单独一个函数是为了让它画在 LiDAR 之后** —— 点数少,压在底下就没了。

    `window` 给定时只画窗内点并只数窗内点。**这条不是优化而是判据**:实测同一帧
    5 雷达共 1398 点,其中相当一部分在 ±30 m 窗口外(侧向雷达 y 最远到 87 m、
    后向最远 −245 m)—— 不滤的话 `n_radar_drawn` 会**虚报**成"全画上了",
    而实际上 PIL 早已把它们裁掉。
    """
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(arr) == 0:
        return 0
    if window is not None:
        (x0, x1), (y0, y1) = window
        arr = arr[(arr[:, 0] >= x0) & (arr[:, 0] <= x1) & (arr[:, 1] >= y0) & (arr[:, 1] <= y1)]
        if len(arr) == 0:
            return 0
    pix = px.arr(arr[:, :2])
    draw.point([(float(a), float(b)) for a, b in pix], fill=RADAR_COLOR)
    return len(arr)


def base_layer(
    lidar_B: np.ndarray,
    radar_B: np.ndarray,
    px: Px,
    size: tuple[int, int],
    window: tuple[tuple[float, float], tuple[float, float]],
    *,
    bg: tuple[int, int, int] = (20, 20, 20),
) -> tuple[Image.Image, dict]:
    """把一帧点云画成**一张图**,交给调用方当底板(`bev_panel(base=...)`)。

    为什么返回图而不是"接收 draw 就地画":`mapviz` 定义 `Px`、本模块 import 它 ——
    若再让 `mapviz.bev_panel` 反过来调本模块的散点函数就成环了。把"画"的入口留在
    本模块、把"往哪画"的决定权交给调用方,依赖方向保持单向。

    返回的 `stats` 是**数值自证**:`n_lidar_drawn` 为 0 就说明窗口/换算写错了
    (而不是"这段路本来就没点")。
    """
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    n_lid = scatter_ego(d, lidar_B, px, window)
    n_rad = scatter_radar(d, radar_B, px, window)
    stats = {
        "n_lidar_drawn": n_lid,
        "n_lidar_total": 0 if len(lidar_B) == 0 else int(np.asarray(lidar_B).reshape(-1, 3).shape[0]),
        "n_radar_drawn": n_rad,
        "n_radar_total": 0 if len(radar_B) == 0 else int(np.asarray(radar_B).reshape(-1, 3).shape[0]),
    }
    return img, stats

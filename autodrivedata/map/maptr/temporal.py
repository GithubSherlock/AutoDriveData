"""MapTRv2 时序版:历史帧 BEV 扭转到当前 ego 系 + 残差融合(§P-M.12 阶段 4)。

**为什么是"只融合 BEV"**:官方 MapTRv2 的时序在 `TemporalSelfAttention` 里做,挂在
BEV encoder 之后、decoder 之前;本实现的 `model.MapTR.forward` 是
`backbone → GKT → head`,`head` 只吃 BEV ⇒ 在 GKT 与 head 之间插一层 BEV 融合
**不需要动 head 一行**。这与 `gkt.py` 用"最近相机独占采样"简化官方 topk=1 交叉
注意力是同级别的工程简化(核心几何/时序语义一致)。

**两处口径必须钉死(都会静默出错)**:

1. **变换顺序 = `R_prevᵀ·(R_cur·p + t_cur − t_prev)`,即 `T_prev⁻¹·T_cur`**。
   `gkt.cam_world_pose` 的 `r_e` 是 **ego→world**(`p_w = R_e·p_ego + t_e`),
   所以 world→prev 要左乘 `R_prev⁻¹ = R_prevᵀ`。

   ⚠️ **这里有两个不同的错法,误差律差几个数量级,不能混为一谈**(2026-09-24 用
   `outputs/surround_v2` 的 **200 对真实相邻帧**实测):

   | 错法 | 误差律 | 实测格位移(格宽 0.3 m) |
   |---|---|---|
   | **漏转置**(用 `R_prev` 代 `R_prevᵀ`) | `2·abs(w)·sin(ψ_prev)` | 中位 **138 格 = 41 m**,95% 分位 227 格 |
   | **错乘积**(`R_prev·R_curᵀ` 作用在 `p_cur`) | `2·abs(p_cur)·sin(Δψ)` | Δψ 中位 0.46° 时 ≈ 2 格;转弯时爆炸 |

   漏转置的盲点是 **`ψ_prev = 0`**(不是 `Δψ = 0`)—— 这正是它最容易骗过测试的地方:
   姿态全零时 `R_prev = I`,`R_prev` 与 `R_prevᵀ` 恒等,写反完全测不出来
   (`tests/test_temporal.py` 第一版就是这么假绿的)。错乘积则在**直行**时近乎无害、
   只在转弯处显著 ⇒ 两条都必须用**非零绝对偏航**的姿态去钉。

2. **BEV 两轴不许对调**:`gkt.BEVParams` 是 `bev_h=200` 沿 **ego y**、
   `bev_w=100` 沿 **ego x**,`cell_centers()` 行主序 ⇒ 张量的 H 索引是 y、
   W 索引是 x(H 增大的方向是车体**右侧**,见 `geometry.py` 的 `x 前/y 右`;
   `gkt` 里"y 左"的措辞是**标签写反**的注释错,不影响几何)。故 `grid_sample`
   的 grid 末维必须是 **(gx, gy)** 而不是 (gy, gx)。

归一化口径与 `gkt.GKT.forward` 采样图像时同式(`align_corners=False`):栅格索引 i
的连续坐标是 **i**(§P-M.11 CORNER 约定),归一化坐标 `g = (u+0.5)/N·2−1`。反解
BEV 单元中心 `x = xmin + (w+0.5)·res` 代入,得 `gx = (x − xmin)/(xmax − xmin)·2 − 1`
—— 对称区间 `(−15, 15)` 下即 `x/15`(实测与逐格索引反解精确吻合)。
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from autodrivedata.map.maptr.gkt import _ROT_TO_CARLA, BEV_DEFAULT, BEVParams, _carla_rotation_torch


def ego_rotations(poses: torch.Tensor) -> torch.Tensor:
    """infos 六元组 (…, 6)[x,y,z,yaw,pitch,roll] 度 → ego→world 旋转阵 (…, 3, 3)。

    与 `gkt.cam_world_pose` 的 `r_e` **同源同序** —— `_ROT_TO_CARLA` 是换序的唯一
    落点,这里再写一遍换序就会重演 gkt 坑 1(5/6 相机指向错且 yaw≈0 的看不出)。

    ★ **输入先升到 float64 再算,返回值也是 float64**。位姿张量在训练里是 float32,
    直接算出来的旋转阵 `Rᵀ·R` 与单位阵差 **1.45e-6**(float64 下是 1e-16)。这个量级
    看着无害,但 `_norm_xy` 的 `(gx+1)` 是灾难性抵消,会把 1.4e-6 放大成画幅边缘
    1e-4 格的位置误差 —— 表现为"两帧姿态相同却映射出不同结果",而这类偏差在训练里
    只会安静地退化成噪声。3×3 的升精度开销可以忽略。
    """
    return _carla_rotation_torch(torch.deg2rad(poses.to(torch.float64)[..., 3:6])[..., _ROT_TO_CARLA])


def _assert_matches_gkt(bev: BEVParams, centers: torch.Tensor) -> None:
    """自检:本模块解析重建的网格点必须与 `gkt.cell_centers()` 一致(float32 精度内)。

    这条把"**同一份网格口径写了两遍**"从静默风险变成响错误:若日后 `gkt` 改了
    `pc_range` 的解释(比如换 `endpoint=True`、或把 z 挪走),融合层会在**第一次
    前向时**就地炸,而不是安静地按另一套格采样、训练出一个 AP 略低的模型。
    """
    ref = torch.from_numpy(bev.cell_centers()).double()
    err = (centers - ref).abs().max().item()
    # 阈值 1e-5:`cell_centers()` 出口是 float32,30 附近的 eps ≈ 2e-6,实测 7.6e-7
    if err > 1e-5:
        raise AssertionError(f"BEV 网格重建与 gkt.cell_centers() 不符,max|e|={err:.3e}")


@lru_cache(maxsize=8)
def _cell_centers(bev: BEVParams) -> torch.Tensor:
    """BEV 单元中心 (bev_h*bev_w, 3) **float64** CPU(行主序,H=y/W=x)。

    与 `gkt.BEVParams.cell_centers()` 同式(`x = xmin + (i+0.5)·res`,行主序展平),
    但**在 float64 里解析重建**而不是拿现成结果转类型 —— 后者出口是 float32,
    舍入让 -14.85 变成 -14.850000381469727,归一化后采样点偏 1.3e-6 格,于是
    "整格对齐时逐元素相等"这类精确断言拿不到,只能退化成带容差的弱判据(实测
    max|e| 1.8e-5)。调用方在 `warp_bev` 里按特征图 dtype 降精度,训练行为不变。

    缓存是因为它每步前向都要用而 numpy meshgrid 不便宜;`BEVParams` 是 frozen
    dataclass ⇒ 可哈希,lru_cache 安全。
    """
    res_x = (bev.pc_range[2] - bev.pc_range[0]) / bev.bev_w
    res_y = (bev.pc_range[3] - bev.pc_range[1]) / bev.bev_h
    x = bev.pc_range[0] + (torch.arange(bev.bev_w, dtype=torch.float64) + 0.5) * res_x
    y = bev.pc_range[1] + (torch.arange(bev.bev_h, dtype=torch.float64) + 0.5) * res_y
    out = torch.empty(bev.bev_h * bev.bev_w, 3, dtype=torch.float64)
    out[:, 0] = x.repeat(bev.bev_h)  # 行主序:W(=x)是最快变化的索引
    out[:, 1] = y.repeat_interleave(bev.bev_w)
    out[:, 2] = bev.z_level
    _assert_matches_gkt(bev, out)
    return out


def _norm_xy(p_ego: torch.Tensor, bev: BEVParams) -> torch.Tensor:
    """ego 系 (…, 3)[米] → 该 ego 系 BEV 的归一化采样坐标 (…, 2)[gx, gy]。

    `align_corners=False` 口径:单元中心 x = xmin + (w+0.5)·res 反解得
    `gx = (x − xmin)/(xmax − xmin)·2 − 1`(对称区间即 x/15)。**末维顺序是 (gx, gy)**
    —— BEV 张量是 (B, C, bev_h, bev_w),H 索引吃 gy、W 索引吃 gx,写反即两轴对调。
    """
    xmin, ymin, xmax, ymax = bev.pc_range
    gx = (p_ego[..., 0] - xmin) / (xmax - xmin) * 2 - 1
    gy = (p_ego[..., 1] - ymin) / (ymax - ymin) * 2 - 1
    return torch.stack([gx, gy], dim=-1)


def warp_bev(
    bev_prev: torch.Tensor,
    pose_prev: torch.Tensor,
    pose_cur: torch.Tensor,
    bev: BEVParams = BEV_DEFAULT,
) -> torch.Tensor:
    """把 `bev_prev`(prev ego 系)重采样到 **cur ego 系**。

    - `bev_prev` (B, C, bev_h, bev_w)
    - `pose_prev` / `pose_cur` (B, 6)[x, y, z, yaw, pitch, roll] 度
    - 返回同形状张量:每个 **cur 系** BEV 单元取 prev 系里对应位置的读数,
      落不到 prev 画幅内的单元补 0(`padding_mode="zeros"`)。

    语义 = 「上一帧的特征图,按两帧相对位姿搬到当前视角」。**逐点映射**(不是网格
    重采样):cur 系每个单元中心 → world → prev ego 系 → 归一化 → 双线性采样。
    整条链按**完整 3D 刚体变换**走(含 pitch/roll),只在最后取 (x, y) 投影 ——
    这一步丢的是"prev 系里的高度",等价于把 prev 的路面当平面,与 GKT 单高度采样
    口径一致(地图要素全在路面,见 gkt 模块头注)。

    ⚠️ **一个反直觉但有意义的推论**:丢 z 之后,**两帧 pitch/roll 相同**时它们对
    (x,y) 完全不产生影响(单元中心 z=0 ⇒ 前向旋转的第三列乘的是 0;反向旋转的
    第三列只把 z 馈回 x/y,而 z 恰好不含 pitch/roll 的净效应)。**只有相对 pitch/roll
    才有位移**。这不是 bug —— 车整体俯仰而路面跟着俯仰,路面上的点当然不动。
    实测:两帧同为 0 时,把 `pose_cur` 单独转 15° 对 (x,y) 位移**恰好 0**;
    而 prev 与 cur 差 15° 时位移 0.506 m(见 `tests/test_temporal.py`)。
    """
    b = bev_prev.shape[0]
    # 三个张量的 batch 维必须一致:写成"以 pose 的 batch 去 expand"会在二者不等时
    # 直接 view 失败(或更糟:静默按错维度 reshape)。不给广播留口子。
    for name, t in (("pose_prev", pose_prev), ("pose_cur", pose_cur)):
        if t.shape[0] != b:
            raise ValueError(f"{name} 的 batch {t.shape[0]} ≠ bev_prev 的 {b}")

    # ★ 整条链在 **float64** 里算,只在最后一步把 grid 降到特征图 dtype。
    # 理由:`_norm_xy` 的 `(gx+1)/2·N − 0.5` 是**灾难性抵消** —— gx 在 −1 附近时
    # `gx+1` 只剩 ~0.01,float32 的 6e-8 绝对误差被放大成 ~3e-5,再乘 N/2 得
    # **u 偏 ~7e-5 格**(实测)。训练上无害,但"整格对齐时逐元素相等"这类精确判据
    # 就只能退化成弱容差。在 float64 里算完再降精度,u 的误差 ~1e-13 格、
    # 且降到 float32 后恰好落在整格上,两边都拿到。
    # (grid 只有 20000×2 个数,这个 upcast 的开销可以忽略。)
    dtype = bev_prev.dtype
    r_prev = ego_rotations(pose_prev).to(torch.float64)
    r_cur = ego_rotations(pose_cur).to(torch.float64)
    t_prev = pose_prev[..., :3].to(torch.float64).unsqueeze(-2)
    t_cur = pose_cur[..., :3].to(torch.float64).unsqueeze(-2)
    cen = _cell_centers(bev).to(device=bev_prev.device).expand(b, -1, -1)

    # cur ego → world(R 是 ego→world:行向量形式 p_w = R·p + t)
    p_w = torch.einsum("bij,bnj->bni", r_cur, cen) + t_cur
    # world → prev ego:左乘 R_prev⁻¹ = R_prevᵀ(写成 R_prev 即反向旋转,静默错)
    p_prev = torch.einsum("bji,bnj->bni", r_prev, p_w - t_prev)

    grid = _norm_xy(p_prev, bev).view(bev_prev.shape[0], bev.bev_h, bev.bev_w, 2).to(dtype)
    return F.grid_sample(bev_prev, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class TemporalFusion(nn.Module):
    """历史 BEV(已扭到当前系)与当前 BEV 的融合:拼接 → 1×1 conv → **残差加回当前 BEV**。

    `proj` **零初始化** ⇒ 初始化时该模块恒等(`out = bev_cur`),时序模型在第 0 步
    与单帧模型**逐位相同**。这样"时序带来了多少增益"是学出来的,不是结构自带的
    —— 也排除了"只是多了一层卷积所以变了"的解释。

    ⚠️ **ReLU 必须在 `proj` 之前,不能写成 `cur + relu(proj(cat))`**:后者零初始化下
    `proj(cat) = 0` ⇒ `relu'(0) = 0` ⇒ `proj.weight.grad ≡ 0`,**历史分支的梯度被
    彻底饿死,模块永久停在恒等**(实测:不加这条断言时,梯度回归会红)。现写法
    `cur + proj(relu(cat))` 里 `relu` 作用在输入侧(非零),故 `proj` 的权重/偏置
    在第一步就有非零梯度,而输出仍精确为 0。回归钉:`TestTemporalFusion::test_gradients_reach_history_branch`。

    通道块序 = **[当前, t−1, t−2, …]**(按时间**由近及远**),便于消融时按块置零;
    而对外 API 一律 **oldest → newest**(构造窗口最自然),翻序只在本模块内做一次。
    """

    def __init__(self, channels: int, n_hist: int = 2) -> None:
        super().__init__()
        if n_hist < 1:
            raise ValueError(f"n_hist 需 ≥ 1,收到 {n_hist}")
        self.n_hist = n_hist
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(channels * (n_hist + 1), channels, kernel_size=1)
        # 零初始化 ⇒ 初始化时该模块恒等(out = bev_cur,见类文档)。
        # assert 只为收窄 Conv2d.weight/bias 的 `Tensor | None` 桩类型,无运行时语义。
        assert self.proj.weight is not None and self.proj.bias is not None
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, bev_cur: torch.Tensor, hist_warpped: list[torch.Tensor]) -> torch.Tensor:
        """`hist_warpped` = 已扭到当前系的 BEV 列表,**旧 → 新**;长度须 = n_hist。"""
        if len(hist_warpped) != self.n_hist:
            raise ValueError(f"历史帧数 {len(hist_warpped)} ≠ 模型配置的 {self.n_hist}")
        # 通道块序按时间由近及远:当前在前,历史倒序。ReLU 在 proj 之前(见类文档)
        return bev_cur + self.proj(self.act(torch.cat([bev_cur, *reversed(hist_warpped)], dim=1)))

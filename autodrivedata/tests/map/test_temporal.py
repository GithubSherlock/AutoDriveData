"""时序 BEV 扭正的回归钉(§P-M.12 阶段 4)。

这个模块的错法**全是静默型**:写反矩阵顺序、对调 BEV 两轴、归一化仿射差一点,
都不会崩、都会照常训练出一个能报 AP 的模型。所以这里不用"跑得通"当判据,
全部用**手算可验的闭式 oracle**:

- 线性场在均匀栅格上双线性插值**恒等** ⇒ 采样值必须等于解析值(把归一化仿射、
  align_corners、两轴顺序一次全钉死);
- 纯平移下逐格位移 = Δ/res 格,**整数格时逐元素相等**(不需要 oracle);
- 纯 yaw 下 p_prev = R(θ_cur − θ_prev)·p_cur,用平面旋转阵手算。

⚠️ **四条测试自身踩过的坑(都已钉住,别再犯)**:

1. **转置/顺序陷阱的对照必须给 prev 非零绝对偏航** —— `pose_prev` 全零时
   `R_prev = I`,`R_prev` 与 `R_prevᵀ` 恒等、对照实现与真实现同解,测试假绿。
   `test_zero_prev_yaw_makes_transpose_indistinguishable` 把这个前提本身也钉住。
   (两种错法——漏转置 vs 错乘积——的误差律与量级差几个数量级,见
   `autodrivedata/map/maptr/temporal.py` 模块头注的表。)
2. **解析断言只能在 `in_support` 内比** —— 旋出 prev 画幅的格走 zero-padding 补 0,
   解析式在那里不成立,拿全图 max 比会得到"实现错了"的假结论(实测 6.94 的假误差)。
3. **手算 oracle 必须走完整 3D**:单元中心 z=0,但前向旋转会给出非零 z,反向旋转的
   第三列再把它馈回 x/y。只取 2×2 分块会漏掉这一项,在 pitch/roll 非零时假红。
4. **判"pitch/roll 有没有被丢掉"要用依赖 y 的场**:场只依赖 x 时,roll 引起的
   (x, y) 位移在双线性插值里被**同列两点的相同值**抵消,读数恒 0 —— 那是场的问题,
   不是 roll 没用。用 `x + y` 才测得到。
"""

from __future__ import annotations

import pytest
import torch

from autodrivedata.map.maptr.gkt import BEV_DEFAULT as BEV
from autodrivedata.map.maptr.temporal import TemporalFusion, ego_rotations, warp_bev

H, W = BEV.bev_h, BEV.bev_w
RES = BEV.res
# 网格点经"归一化 → 采样 → 反归一化"往返后不是逐位精确(实测 u 差 1.4e-14 格),
# 故精确类断言带这个容差:仍比 float32 的 eps(~1e-7)低两个量级。
ATOL = 1e-9


def _centers3d() -> torch.Tensor:
    """(H*W, 3) float64 单元中心 [x 前后, y 左右, z=路面],行主序(H=y / W=x)。"""
    res_x = (BEV.pc_range[2] - BEV.pc_range[0]) / W
    res_y = (BEV.pc_range[3] - BEV.pc_range[1]) / H
    cx = BEV.pc_range[0] + (torch.arange(W, dtype=torch.float64) + 0.5) * res_x
    cy = BEV.pc_range[1] + (torch.arange(H, dtype=torch.float64) + 0.5) * res_y
    out = torch.zeros(H * W, 3, dtype=torch.float64)
    out[:, 0] = cx.repeat(H)
    out[:, 1] = cy.repeat_interleave(W)
    return out


def _centers2d() -> torch.Tensor:
    """(H, W, 2) —— 铺线性场用(双线性插值对线性场恒等)。"""
    return _centers3d()[:, :2].view(H, W, 2)


def _pose(x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0, roll=0.0) -> torch.Tensor:
    """单帧 infos 口径 (1, 6)[x,y,z,yaw,pitch,roll] 度。

    ⚠️ 默认 **float32** —— 与真实 infos 一致。要"逐格精确"类断言必须用 `_exact_pose`,
    因为 4·res = 1.2 在 float32 里是 1.2000000476837158,那个位移**根本不在整格上**
    (实测残差 1.4e-6,曾据此误判实现有问题)。
    """
    return torch.tensor([[x, y, z, yaw, pitch, roll]])


def _exact_pose(x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0, roll=0.0) -> torch.Tensor:
    """float64 版位姿 —— 只给"给定精确输入则精确输出"这类断言用(见 `_pose` 注)。"""
    return torch.tensor([[x, y, z, yaw, pitch, roll]], dtype=torch.float64)


def _prev_coords(pose_prev: torch.Tensor, pose_cur: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """手算 (p_prev (N,2), in_support (N,)):cur 单元中心(3D)→ world → prev ego。

    走**完整 3D**(见模块头注坑 3);`in_support` 圈定双线性对线性场精确的区间 ——
    采样点必须落在 prev 栅格单元中心张成的矩形内,旋出去的点补 0、解析式不成立。
    """
    cen = _centers3d()
    r_prev = ego_rotations(pose_prev).double()[0]
    r_cur = ego_rotations(pose_cur).double()[0]
    p_w = cen @ r_cur.T + pose_cur[0, :3].double()
    p_prev = (p_w - pose_prev[0, :3].double()) @ r_prev  # 行向量右乘 R ≡ 左乘 Rᵀ
    hx = cen[:, 0].abs().max()
    hy = cen[:, 1].abs().max()
    return p_prev[:, :2], (p_prev[:, 0].abs() <= hx) & (p_prev[:, 1].abs() <= hy)


def _warp_missing_transpose(
    bev_prev: torch.Tensor, pose_prev: torch.Tensor, pose_cur: torch.Tensor
) -> torch.Tensor:
    """故意**漏掉转置**(用 `R_prev` 代 `R_prevᵀ`)的对照实现 —— 只为证明它可被区分。

    ⚠️ 这不是"错乘积"那一支:它的误差律是 `2·abs(w)·sin(ψ_prev)`,盲点是
    **`ψ_prev = 0`**。所以用它做对照时 `pose_prev` 的**绝对偏航必须非零**,
    否则 `R_prev = I`、正反同解,对照恒等、测试假绿(见模块头注坑 1)。
    """
    cen = _centers3d().unsqueeze(0)
    r_prev = ego_rotations(pose_prev).double()
    r_cur = ego_rotations(pose_cur).double()
    p_w = torch.einsum("bij,bnj->bni", r_cur, cen) + pose_cur[..., :3].double().unsqueeze(-2)
    p_prev = torch.einsum("bij,bnj->bni", r_prev, p_w - pose_prev[..., :3].double().unsqueeze(-2))
    xmin, ymin, xmax, ymax = BEV.pc_range
    grid = torch.stack(
        [(p_prev[..., 0] - xmin) / (xmax - xmin) * 2 - 1, (p_prev[..., 1] - ymin) / (ymax - ymin) * 2 - 1], -1
    )
    return torch.nn.functional.grid_sample(
        bev_prev.double(), grid.view(1, H, W, 2), mode="bilinear", padding_mode="zeros", align_corners=False
    )


def _xy_field(a: float = 0.7, b: float = -1.3, c: float = 2.5) -> torch.Tensor:
    """(1, 1, H, W) 的 `a·x + b·y + c` —— 双线性插值对线性场**恒等**,故可闭式验。"""
    cen = _centers2d()
    return (a * cen[..., 0] + b * cen[..., 1] + c).view(1, 1, H, W)


class TestWarpBeV:
    """`warp_bev`:`bev_prev`(prev ego 系)重采样到 cur ego 系。"""

    def test_linear_field_is_reproduced_exactly(self):
        """线性场 ⇒ 闭式 oracle:采样值必须等于解析值。

        一条断言同时钉死三件事:归一化仿射(`(x−xmin)/(xmax−xmin)·2−1`,对称区间即
        x/15)、`align_corners=False`、以及 grid 末维是 **(gx, gy)** 而非 (gy, gx)。
        姿态取两帧都非零且含 pitch/roll,顺带覆盖完整 3D 链。
        """
        prev = _xy_field().double()
        pose_prev = _pose(yaw=40.0, pitch=6.0, roll=3.0)
        pose_cur = _pose(x=3.0, y=-1.0, yaw=25.0, pitch=2.0, roll=-1.0)
        out = warp_bev(prev, pose_prev, pose_cur, BEV).double()

        p_prev, ok = _prev_coords(pose_prev, pose_cur)
        expect = (0.7 * p_prev[:, 0] - 1.3 * p_prev[:, 1] + 2.5).view(1, 1, H, W)
        assert int(ok.sum()) > 10000, f"画幅内样本仅 {int(ok.sum())} 个,断言没有覆盖面"
        err = (out - expect).abs().view(-1)[ok].max().item()
        assert err < 1e-6, f"线性场未被精确复现 max|e|={err:.3e}(归一化/轴序/align_corners 有错)"

    def test_pure_translation_shifts_by_exact_cell_count(self):
        """纯平移、无旋转 ⇒ 逐格位移 = Δ/res 格。取整数格位移以做到**逐元素相等**。

        这条不需要 oracle:整格位移下双线性采样恰好命中单元中心,读数必须一字不差。
        用 `_exact_pose`(float64):见 `_pose` 注 —— float32 位姿下位移不在整格上,
        本来就不该期望精确相等(那种情况由 `test_float32_pose_quantization_is_bounded` 管)。
        """
        dx, dy = 4 * RES, 3 * RES  # 恰好 4 格(列)与 3 格(行)
        prev = torch.randn(1, 3, H, W, dtype=torch.float64)
        out = warp_bev(prev, _exact_pose(), _exact_pose(x=dx, y=dy), BEV)

        # 车前进 +x/+y ⇒ 世界物相对车后移:cur 格 (h, w) 读到 prev 格 (h+3, w+4)
        assert (out[:, :, : H - 3, : W - 4] - prev[:, :, 3:, 4:]).abs().max().item() < ATOL
        # 越界在**高索引**一侧:cur 格 h 读 prev 格 h+3,故尾部 H−3 行/末 4 列落空补 0。
        # 注意不能断言 `== 0.0`:采样点落在画幅外 1e-14 格处时,最近邻权重还残留
        # ~1e-14 的读数(实测 1.85e-14)—— 那是 padding 的正常边界行为,不是漏补。
        assert out[:, :, H - 3 :, :].abs().max().item() < ATOL
        assert out[:, :, :, W - 4 :].abs().max().item() < ATOL

    def test_float32_pose_quantization_is_bounded(self):
        """float32 位姿下的位置误差**上界** —— 这条把"精确断言为什么必须用 float64 位姿"钉住。

        真实 infos 的位姿是 float32:位移 1.2 只能存成 1.2000000476837158(相对误差 4e-8),
        换算到采样点是 ~1.6e-7 格。这不是缺陷,是输入编码的极限;判据是它**有界且远小于
        一个格**(实测 < 1e-4,而格宽 0.3 m ⇒ 位置误差 < 3e-5 m)。
        """
        dx, dy = 4 * RES, 3 * RES
        prev = torch.randn(1, 3, H, W, dtype=torch.float64)
        out = warp_bev(prev, _pose(), _pose(x=dx, y=dy), BEV).double()
        err = (out[:, :, : H - 3, : W - 4] - prev[:, :, 3:, 4:]).abs().max().item()
        assert 0.0 < err < 1e-4, f"float32 位姿残差 {err:.3e} 超出一个格的量级"

    def test_float32_features_are_the_precision_floor(self):
        """float32 **特征图**的往返精度地板 —— 单测里不能对 float32 断言精确相等。

        根因是 `_norm_xy` 的 `(gx+1)/2·N − 0.5` 在 gx≈−1 处灾难性抵消:float32 的
        −0.99 存成 −0.99000000953674316,反解出 u 偏 3.8e-6 格(隔离实测:float64 输入
        下同一式子只偏 7e-15,读数误差 7e-14 vs float32 的 3.6e-5)。训练上无关紧要,
        但任何"逐元素相等"的断言都只能在 float64 上成立。
        """
        prev = torch.randn(2, 4, H, W)  # float32
        out = warp_bev(prev, _pose().repeat(2, 1), _pose().repeat(2, 1), BEV)
        err = (out - prev).abs().max().item()
        assert 1e-9 < err < 1e-3, f"float32 恒等映射残差 {err:.3e} 不在预期量级(实测定标用)"

    def test_pure_yaw_matches_planar_rotation(self):
        """纯 yaw、无平移 ⇒ p_prev = R(θ_cur − θ_prev)·p_cur,用平面旋转阵手算比对。

        场 = x 坐标(线性)⇒ 采样值就是 p_prev 的 x 分量本身。这条独立于 gkt 的 3D
        旋转 helper(手写 2D 旋转),能抓到 helper 与 warp 之间的口径错配。
        """
        prev = _centers2d()[..., 0].unsqueeze(0).unsqueeze(0).double()
        th_prev, th_cur = 40.0, 25.0
        out = warp_bev(prev, _pose(yaw=th_prev), _pose(yaw=th_cur), BEV).double()
        p_prev, ok = _prev_coords(_pose(yaw=th_prev), _pose(yaw=th_cur))
        err = (out - p_prev[:, 0].view(1, 1, H, W)).abs().view(-1)[ok].max().item()
        assert err < 1e-6, f"纯 yaw 未复现平面旋转 max|e|={err:.3e}"

    def test_missing_transpose_is_detected(self):
        """漏转置(`R_prev` 代 `R_prevᵀ`)必须与正确实现**显著不同** —— 防回归的核心。

        真实帧上这个偏差中位 138 格(41 m),但它在 `ψ_prev = 0` 处**恒为零**;
        这个测试给 prev 一个明显非零的绝对偏航把差距放大到可断言(实测差 29.4 m)。
        """
        prev = _centers2d()[..., 0].unsqueeze(0).unsqueeze(0).double()
        pose_prev, pose_cur = _pose(yaw=40.0, pitch=6.0, roll=3.0), _pose(yaw=25.0)
        ok = warp_bev(prev, pose_prev, pose_cur, BEV).double()
        bad = _warp_missing_transpose(prev, pose_prev, pose_cur)
        assert (ok - bad).abs().max().item() > 1.0, "反向顺序未被区分 ⇒ 测试本身失效"

    def test_zero_prev_yaw_makes_transpose_indistinguishable(self):
        """把上一条的**前提**也钉住:`pose_prev` 偏航为 0 时 `R_prev = I`,转置与否同解。

        这是"测试假绿"的根因 —— 对照测试必须给 prev 非零**绝对**偏航(不是非零相对转角)。
        """
        prev = _centers2d()[..., 0].unsqueeze(0).unsqueeze(0).double()
        pose_prev, pose_cur = _pose(), _pose(x=3.0, yaw=25.0, pitch=6.0)
        ok = warp_bev(prev, pose_prev, pose_cur, BEV).double()
        assert torch.allclose(ok, _warp_missing_transpose(prev, pose_prev, pose_cur), atol=1e-6)

    def test_batch_mismatch_raises(self):
        """pose 与 bev 的 batch 不一致必须**显式报错**,不能靠 view 失败或静默错位。"""
        with pytest.raises(ValueError, match="batch"):
            warp_bev(torch.randn(2, 4, H, W), _pose(), _pose(), BEV)

    def test_no_motion_is_identity(self):
        """零相对位姿 ⇒ 恒等(采样点恰好落在单元中心)。"""
        prev = torch.randn(2, 4, H, W, dtype=torch.float64)
        out = warp_bev(prev, _exact_pose().repeat(2, 1), _exact_pose().repeat(2, 1), BEV)
        assert (out - prev).abs().max().item() < ATOL

    def test_axes_are_not_swapped(self):
        """x 平移只改**列**、y 平移只改**行**(BEV 张量 H 索引 = ego y、W 索引 = ego x)。

        ⚠️ 方向:车沿 +x 前进一格 ⇒ 世界物相对车**后移**一格,故尖峰列号 **−1**
        (不是 +1)。写成 +1 是把"车的位移"当成了"物的位移"。
        """
        prev = torch.zeros(1, 1, H, W)
        prev[0, 0, 100, 50] = 1.0
        x_only = warp_bev(prev, _pose(), _pose(x=RES), BEV)  # 车 +x 前进一格
        assert float(x_only[0, 0, 100, 49]) == 1.0  # 行不变、列 −1
        y_only = warp_bev(prev, _pose(), _pose(y=RES), BEV)  # 车 +y 左移一格
        assert float(y_only[0, 0, 99, 50]) == 1.0  # 列不变、行 −1

    def test_relative_pitch_and_roll_displace_points(self):
        """★ **相对** pitch/roll 必须真的搬动采样点,不能被当成 0 丢掉。

        防的是"只用了 yaw"—— 平路上几乎看不出来。场取 `x + y`(见模块头注坑 4:
        只用 x 时 roll 的位移会被同列两点抵消,读数恒 0)。两帧各自 20° 的相对俯仰/侧倾
        下位移上限 = (1−cos20°)·半幅 = 0.90 m(pitch) / 1.80 m(roll)。
        """
        prev = _xy_field(1.0, 1.0, 0.0).double()
        flat = warp_bev(prev, _pose(), _pose(), BEV).double()
        for kw in ({"pitch": 20.0}, {"roll": 20.0}):
            tilted = warp_bev(prev, _pose(), _pose(**kw), BEV).double()
            assert (tilted - flat).abs().max().item() > 0.5, f"相对 {kw} 未产生影响"

    def test_common_pitch_and_roll_are_a_noop(self):
        """★ 反过来:两帧**相同**的 pitch/roll 对 (x,y) **恰好无影响**。

        这不是漏改,是几何事实 —— 丢 z 之后,车整体俯仰时路面上的点不动。这条把
        "为什么这个模块在近水平路面上仍然必须带 pitch/roll"讲清楚:带它是为了
        **悬架俯仰逐帧变化**(§P-M.10 实测 +0.0642°),不是因为车有固定倾角。
        上一版把这条当 bug 追过(测试用 `x` 单变量场 + 同类姿态,读数 0.0000)。
        """
        prev = _xy_field().double()
        base = warp_bev(prev, _pose(), _pose(), BEV).double()
        kw = {"pitch": 6.0, "roll": 3.0}
        same = warp_bev(prev, _pose(**kw), _pose(**kw), BEV).double()
        assert (same - base).abs().max().item() < ATOL


class TestTemporalFusion:
    """融合层:零初始化恒等、通道块序、梯度可达、长度校验。"""

    def test_zero_init_is_identity(self):
        """`proj` 零初始化 ⇒ 初始化时 `out = bev_cur`。

        这是"时序增益是学出来的、不是结构自带"的前提:第 0 步时序模型与单帧模型
        **逐位相同**,故二者 AP 之差不能归因于多了一层卷积。
        """
        f = TemporalFusion(8, n_hist=2)
        cur = torch.randn(2, 8, H, W)
        hist = [torch.randn(2, 8, H, W), torch.randn(2, 8, H, W)]
        assert torch.equal(f(cur, hist), cur)

    def test_gradients_reach_the_history_branch(self):
        """★ 零初始化**不能**把历史分支饿死。

        `cur + relu(proj(cat))` 在 init 下 `proj(cat) = 0` ⇒ `relu'(0) = 0`
        ⇒ `proj.weight.grad ≡ 0`,历史分支永远学不进来、模块永久恒等(实测)。
        现写法 `cur + proj(relu(cat))` 把 ReLU 放在输入侧,`proj` 第一步就有非零梯度。

        ⚠️ 注意 init 时 **`hist.grad` 恒为 0 是必然的**(`d(out)/d(hist) = proj.weightᵀ = 0`),
        不是缺陷 —— 权重一旦离开 0 它立刻变非零。所以判据分两段:第一步看
        `proj.weight.grad`,走一步 optimizer 之后再看 `hist.grad`。
        """
        f = TemporalFusion(4, n_hist=1)
        opt = torch.optim.SGD(f.parameters(), lr=0.5)
        cur = torch.randn(1, 4, H, W)
        hist = torch.randn(1, 4, H, W, requires_grad=True)

        f(cur, [hist]).sum().backward()
        assert f.proj.weight.grad is not None
        assert f.proj.weight.grad.abs().sum().item() > 0, "proj.weight 梯度为 0 ⇒ 历史分支被饿死"

        opt.step()
        opt.zero_grad()
        hist.grad = None  # 第一步累积的 0 不是判据,清掉
        f(cur, [hist]).sum().backward()
        assert hist.grad is not None and hist.grad.abs().sum().item() > 0, "权重离开 0 后历史输入仍无梯度"

    def test_channel_order_is_recent_first(self):
        """通道块序 = [当前, t−1, t−2]:消融按块置零时顺序不能含糊。"""
        f = TemporalFusion(1, n_hist=2)
        with torch.no_grad():
            assert f.proj.weight.shape[1] == 3
            f.proj.weight.zero_()
            f.proj.bias.zero_()
            f.proj.weight[0, 0, 0, 0] = 1.0  # 只连"当前"块
        cur = torch.full((1, 1, H, W), 7.0)
        hist = [torch.full((1, 1, H, W), 100.0), torch.full((1, 1, H, W), 200.0)]
        # ReLU 在 proj 之前(输入侧),故输出 = cur + cur 的正部
        assert torch.allclose(f(cur, hist), cur + torch.relu(cur))

    def test_wrong_history_count_raises(self):
        f = TemporalFusion(8, n_hist=2)
        with pytest.raises(ValueError, match="历史帧数"):
            f(torch.randn(1, 8, H, W), [torch.randn(1, 8, H, W)])

    def test_rejects_n_hist_below_one(self):
        with pytest.raises(ValueError, match="n_hist"):
            TemporalFusion(8, n_hist=0)


class TestEgoRotations:
    def test_matches_gkt_cam_world_pose_convention(self):
        """与 `gkt.cam_world_pose` 的 `r_e` 必须同源同序 —— 换序写两遍就会重演
        gkt 坑 1(5/6 相机指向错,yaw≈0 的看不出)。"""
        from autodrivedata.map.maptr.gkt import cam_world_pose

        pose = _pose(x=1.0, y=2.0, z=1.5, yaw=33.0, pitch=-4.0, roll=2.0)
        _, r_e, _ = cam_world_pose(pose, torch.zeros(1, 6))
        # gkt 那条链全程 float32,而 ego_rotations 现在返回 float64(见其文档)⇒ 降精度比
        assert torch.allclose(ego_rotations(pose).float(), r_e, atol=1e-6)

    @pytest.mark.parametrize(
        "kw",
        [{"yaw": 40.0}, {"pitch": 6.0}, {"roll": 3.0}, {"yaw": 40.0, "pitch": 6.0, "roll": 3.0}],
    )
    def test_is_proper_rotation(self, kw):
        """det = +1、正交 —— `R_prevᵀ = R_prev⁻¹` 才成立(逆用转置是本模块的前提)。"""
        r = ego_rotations(_pose(**kw)).double()[0]
        assert abs(torch.linalg.det(r).item() - 1.0) < 1e-6
        assert (r.T @ r - torch.eye(3, dtype=torch.float64)).abs().max().item() < 1e-6


def _fake_rig(n_cams: int = 2, size: tuple[int, int] = (64, 64)) -> dict:
    """最小环视 rig:相机绕 z 均布、`sensor2ego` 是**六元组** [x,y,z,yaw,pitch,roll] 度。

    ⚠️ `sensor2ego` 是 6 元组不是 4×4(与 infos/calib.json 同口径,gkt 直接传给
    `cam_world_pose`)。写成矩阵会在 `cam_world_pose` 里报维度错。
    """
    import numpy as np

    k = np.array([[60.0, 0.0, (size[0] - 1) / 2], [0.0, 60.0, (size[1] - 1) / 2], [0.0, 0.0, 1.0]])
    return {
        f"CAM_{i}": {
            "sensor2ego": [0.0, 0.0, 1.5, 360.0 * i / n_cams, 0.0, 0.0],
            "intrinsic": k.tolist(),
        }
        for i in range(n_cams)
    }


def _frames(k: int, n_cams: int = 2, size: tuple[int, int] = (64, 64), seed: int = 0) -> list[dict]:
    """造 k 帧形如 `dataset.collate` 的 images 列表(旧 → 新),**每次调用逐位相同**(固定种子)。

    帧数**显式给**,不从 poses 的形状推 —— 曾经的写法按 `poses.shape[1]` 推,于是
    传错形状的 poses 会先造出错帧数、再在 GKT 里报一个指向别处的维度错(实测)。

    ⚠️ 用 randn 而**不是 zeros**:全零图会让 GKT 输出的 BEV 整片 ≤ 0,于是
    `relu(cat)` 恒 0、`fusion.proj.weight.grad ≡ 0` —— 梯度回归会以"历史分支断了"
    的假象红掉,而真实数据(照片)不会全零。
    """
    g = torch.Generator().manual_seed(seed)
    return [{f"CAM_{i}": torch.randn(1, 3, *size, generator=g) for i in range(n_cams)} for _ in range(k)]


class TestModelSeam:
    """`MapTR` 的时序接缝:融合层零初始化 ⇒ **第 0 步与单帧模型逐位相同**。

    这条是整个时序 A/B 可比性的前提:如果初始化不恒等,后面"时序涨了多少"就分不清
    是时序带来的还是"多了一层卷积"带来的。
    """

    @staticmethod
    def _pair(window: int = 3):
        from autodrivedata.map.maptr.model import MapTR

        # ⚠️ `num_vec` 必须够大:head 的锚线按 `num_vec` 铺格(见 MapTRHead.init),
        # `num_vec=4` 时 rows=1 ⇒ 全部锚点落在 **y=−24** 的 BEV 边缘,而本测试的最小
        # rig 恰好不覆盖那片区域 ⇒ head 的输出对它**免疫**,接缝判据恒假(实测 head
        # 输出差 0.0 vs num_vec=50 的 28.3)。用生产默认值 50 才是有效判据。
        cfg = {"num_vec": 50, "num_layers": 2, "pretrained": False}
        m3 = MapTR(temporal_window=window, **cfg).eval()
        m1 = MapTR(**cfg).eval()
        # 单帧模型吃时序模型的全部非融合权重 ⇒ 两边除融合层外逐位相同
        m1.load_state_dict({k: v for k, v in m3.state_dict().items() if not k.startswith("fusion.")})
        return m1, m3

    def test_temporal_equals_single_frame_at_init(self):
        calibs = _fake_rig()
        # (B=1, K=3, 6):时序模式的 poses 形状与 collate 一致
        poses = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 20.0, 1.0, -2.0],
                    [1.0, 0.0, 0.0, 22.0, -1.0, 1.0],
                    [3.0, 0.5, 0.0, 25.0, 0.0, 0.0],
                ]
            ]
        )
        m1, m3 = self._pair()
        with torch.no_grad():
            out1, v1 = m1(_frames(3)[-1], poses[:, -1], calibs)
            out3, v3 = m3(_frames(3), poses, calibs)
        assert torch.equal(v1, v3)  # valid 只由当前帧决定
        for key in ("pred_logits", "pred_points"):
            assert torch.equal(out1[key], out3[key]), f"{key} 在零初始化下就该逐位相同"

    def test_gradient_reaches_the_fusion_module(self):
        """融合层必须在梯度路径上:ReLU 若写在 proj **之后**(`cur + relu(proj(cat))`),
        零初始化下 `proj` 的偏置梯度也恒 0,模块永久停在恒等(见 temporal 头注)。

        判据用 **bias 的梯度**(= Σ dL/dout),**不用 weight 的**:`dL/d(weight) = Σ dL/dout·relu(cat)`
        还要求 head 采样的 BEV 单元落在 ReLU 激活区。实测随机特征下二者支撑可以完全不相交、
        读数恰好 0(weight 0 / bias 834)—— 那是特征符号的偶然,不是缺陷;拿它当判据会假红。
        """
        calibs = _fake_rig()
        poses = torch.tensor([[[0.0, 0.0, 0.0, 20.0, 0.0, 0.0], [2.0, 0.0, 0.0, 21.0, 0.0, 0.0]]])
        _, m3 = self._pair(window=2)
        m3.train()
        out, _ = m3(_frames(2), poses, calibs)
        loss = out["pred_logits"].abs().sum() + out["pred_points"].abs().sum()
        loss.backward()
        assert m3.fusion is not None and m3.fusion.proj.bias is not None
        grad = m3.fusion.proj.bias.grad
        assert grad is not None and float(grad.abs().sum()) > 0

    def test_history_frames_change_the_output(self):
        """换掉**历史帧**(当前帧逐位不变)必须改变输出 —— 证明历史 BEV 真的流到了 head。

        零初始化下恒等,所以先把 `proj.weight` 填成非零再比:这一条是"历史分支接通了"
        的直接证据,堵住"窗口拼好了、模型却只看了当前帧"的静默失效。
        """
        calibs = _fake_rig()
        poses = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 20.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 21.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0, 22.0, 0.0, 0.0],
                ]
            ]
        )
        _, m3 = self._pair()
        assert m3.fusion is not None and m3.fusion.proj.weight is not None
        with torch.no_grad():
            # 前提:head 真的读 BEV。不成立时下面两条恒假(测的是 rig 覆盖,不是融合),
            # 故先自证——`test_history…` 曾因 num_vec 太小而假绿。
            bev = m3._forward_frames(_frames(3, seed=0)[-1], poses[:, -1], calibs)[0]
            assert not torch.equal(
                m3.head(bev)["pred_points"], m3.head(torch.zeros_like(bev))["pred_points"]
            ), "本 rig 下 head 对 BEV 免疫,接缝判据无效"
            m3.fusion.proj.weight.fill_(0.01)
            cur = _frames(3, seed=0)[-1]  # 当前帧:两次调用逐位相同
            a = m3(_frames(3, seed=0), poses, calibs)[0]["pred_points"]
            b = m3(_frames(2, seed=1) + [cur], poses, calibs)[0]["pred_points"]
        assert not torch.allclose(a, b), "历史帧换了但输出没变 ⇒ 历史 BEV 没进 head"

    def test_rejects_2d_poses(self):
        """(K, 6) 必须报错:`poses[:, j]` 会静默取第 j **列**(见 `_forward_temporal` 注)。"""
        calibs = _fake_rig()
        _, m3 = self._pair(window=2)
        with pytest.raises(ValueError, match=r"\(B, K, 6\)"):
            m3(_frames(2), torch.zeros(2, 6), calibs)

    def test_口径不匹配要报错(self):
        from autodrivedata.map.maptr.model import MapTR

        calibs = _fake_rig()
        poses = torch.tensor([[[0.0, 0.0, 0.0, 20.0, 0.0, 0.0], [2.0, 0.0, 0.0, 21.0, 0.0, 0.0]]])
        single = MapTR(num_vec=4, num_layers=2, pretrained=False).eval()
        with pytest.raises(ValueError, match=r"temporal_window>1"):
            single(_frames(2), poses, calibs)
        temporal = MapTR(num_vec=4, num_layers=2, pretrained=False, temporal_window=2).eval()
        with pytest.raises(ValueError, match="单帧"):
            temporal(_frames(2)[-1], poses[:, -1], calibs)


class TestCheckpoint口径:
    """`load_map_weights` 的两种错配必须**分类**:一种是有意热启动,一种是口径用错。

    静默丢掉多余键会把"时序权重"跑成"单帧模型",AP 差异看着像"时序没用" —— 正是本轮
    A/B 最容易被误读的失效,故用一条单测把它钉死。
    """

    @staticmethod
    def _models():
        from autodrivedata.map.maptr.model import MapTR

        cfg = {"num_vec": 4, "num_layers": 1, "pretrained": False}
        return MapTR(**cfg), MapTR(temporal_window=2, **cfg)

    def test_single_frame_weights_are_a_legal_warm_start(self, tmp_path):
        from autodrivedata.map.maptr.model import load_map_weights

        m1, m3 = self._models()
        p = tmp_path / "single.pt"
        torch.save(m1.state_dict(), p)
        missing = load_map_weights(m3, str(p), torch.device("cpu"))
        assert missing and all(k.startswith("fusion.") for k in missing)
        # 非融合权重必须逐位载入(否则"热启动"是假话)
        assert torch.equal(m3.head.cls_branch.weight, m1.head.cls_branch.weight)

    def test_temporal_weights_into_single_frame_model_raise(self, tmp_path):
        from autodrivedata.map.maptr.model import load_map_weights

        m1, m3 = self._models()
        p = tmp_path / "temporal.pt"
        torch.save(m3.state_dict(), p)
        with pytest.raises(SystemExit, match="多余权重"):
            load_map_weights(m1, str(p), torch.device("cpu"))

    def test_missing_backbone_key_raises(self, tmp_path):
        from autodrivedata.map.maptr.model import load_map_weights

        _, m3 = self._models()
        p = tmp_path / "broken.pt"
        torch.save({k: v for k, v in m3.state_dict().items() if not k.startswith("backbone.")}, p)
        with pytest.raises(SystemExit, match="缺权重"):
            load_map_weights(m3, str(p), torch.device("cpu"))

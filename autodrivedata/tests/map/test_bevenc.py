"""BEV 编码器(MapQR HeightKernelAttention)单测。

核心是**投影 oracle**:`project_bev_anchors`(经由 `build_lidar2img` 的 4×4 齐次链)
必须与项目既有的 [`gkt.project_pts`](../../map/maptr/gkt.py)(GKT 与 B3 验收探针共用的手写投影链)
**逐点一致**。两条链独立写、互为对照 —— 参数序写反(如 `scale_k` 的 (w,h))、
位姿换序、内参不缩放这三类错都会在这里被抓住,而不是等到训练不收敛。

其余各条钉住:高度锚点的 `linspace(0, Z, D)` 口径(含可学习偏移的换算)、
可见掩码语义、kernel 网格偏移、以及**官方那处隐式约束 `D == n_heads`**。
"""

from __future__ import annotations

import pytest
import torch

from autodrivedata.map.maptr.bevenc import (
    PC_RANGE_3D,
    HeightKernelAttention,
    MSDeformableAttentionKernel,
    build_lidar2img,
    get_reference_points_2d,
    get_reference_points_3d,
    project_bev_anchors,
)
from autodrivedata.map.maptr.gkt import BEVParams, project_pts, scale_k

IMG_SIZE = (64, 32)  # (宽, 高) —— 与 gkt.scale_k 同序
FEAT_SHAPE = (8, 4)  # (高, 宽) —— 与本模块其余函数同序
CAM_NAMES = ["CAM_BACK", "CAM_FRONT"]
# 玩具相机:焦距取小 ⇒ 视场够宽,地面上的锚点(z近 0、相机高 1.6 m)**在画幅内**。
# 焦距若按真实 KITTI 口径取(f≈800 / 1600 px),竖直视场只有十几度,地面点会全部落在画幅外,
# 于是「可见性」相关的用例会以"什么都没看见"的方式假通过 —— 所以这里的取值是有意的。
_INTRINSIC = [[50.0, 0.0, 32.0], [0.0, 50.0, 16.0], [0.0, 0.0, 1.0]]
CAM_Z = 1.6
CALIBS = {
    # CARLA 口径:sensor2ego = [tx, ty, tz, yaw, pitch, roll] 度
    "CAM_FRONT": {"sensor2ego": [0.0, 0.0, CAM_Z, 0.0, 0.0, 0.0], "intrinsic": _INTRINSIC},
    "CAM_BACK": {"sensor2ego": [0.0, 0.0, CAM_Z, 180.0, 0.0, 0.0], "intrinsic": _INTRINSIC},
}
POSES = torch.zeros(1, 6)
SMALL_BEV = BEVParams(pc_range=(-15.0, -30.0, 15.0, 30.0), bev_w=10, bev_h=20)


def _k_scaled(name: str) -> torch.Tensor:
    k = torch.tensor(CALIBS[name]["intrinsic"], dtype=torch.float32)
    return scale_k(k, IMG_SIZE, (FEAT_SHAPE[1], FEAT_SHAPE[0]))


def _se(name: str) -> torch.Tensor:
    return torch.tensor(CALIBS[name]["sensor2ego"], dtype=torch.float32)


# **非恒等**的 ego 位姿(取自真实 surround_v2 帧首):漏掉 ego→world 那一步时,
# 恒等位姿下两个实现在数值上恰好同解,只有非恒等位姿才暴露 —— 所以这条必须在。
POSES_MOVED = torch.tensor([[109.516, 89.56, -0.011, -89.992, 0.554, 0.049]])


def test_ego_to_image_is_invariant_to_ego_pose() -> None:
    """★ BEV 锚点与相机**同挂在 ego 上**,故 ego 位姿在「ego→图像」里**必然抵消** ——
    投影矩阵对 ego 位姿不敏感,且这正说明复合做对了。

    代数:相机中心 `t_cam = t_e + R_e·t_s` ⇒ `t_e − t_cam = −R_e·t_s`,于是
    `p_c = R_w2c·(R_e·p_ego + t_e − t_cam) = R_w2c·R_e·(p_ego − t_s)`,
    又 `R_w2c = C2K·R_sᵀ·R_eᵀ` ⇒ `R_w2c·R_e = C2K·R_sᵀ` —— **与 ego 位姿无关**。

    这条钉的是"复合做对了":少做或多做任何一步 ego 变换,抵消都不成立,
    矩阵会随位姿漂(原有的实现正是这个形态)。与
    `test_projection_matches_gkt_oracle_per_point[ego@real]` 互补 ——
    那条用**非恒等位姿对拍真值**,这条钉**不变量**,单看任一条都盖不全。
    """
    moved = POSES.clone()
    moved[0, 0] += 5.0  # 平移
    moved[0, 3] += 30.0  # 转 yaw
    a = build_lidar2img(POSES, CALIBS, CAM_NAMES, IMG_SIZE, FEAT_SHAPE)
    b = build_lidar2img(moved, CALIBS, CAM_NAMES, IMG_SIZE, FEAT_SHAPE)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("poses", [POSES, POSES_MOVED], ids=["ego@origin", "ego@real"])
def test_projection_matches_gkt_oracle_per_point(poses: torch.Tensor) -> None:
    """★ `project_bev_anchors` 与 `gkt.project_pts` 逐点一致(同一 3D 点、同一相机)。

    **两套位姿各跑一遍**:恒等位姿只覆盖「相机外参 + 内参缩放」,非恒等位姿才覆盖
    `ego→world` 的复合(见上一条)。

    **限域是口径决定的,不是放水**:本模块照官方把相机系 z **钳到 eps**
    (`z.clamp(min=eps)`,见 `point_sampling`)后再做除法;`project_pts` 不钳。
    两者只在 `z > eps` 的区域按定义一致 —— 那里才是"点在相机前方"的合法域,
    也正是掩码判 `z > eps` 的那个域。
    """
    torch.manual_seed(0)
    b, n, d = 1, 40, 3
    base = torch.rand(b, n, 1, 3) * torch.tensor([30.0, 60.0, 6.0]) + torch.tensor([-15.0, -30.0, -3.0])
    ref_m = base.expand(b, n, d, 3).contiguous()  # 三个同位置锚点(只为凑 D 维)

    lidar2img = build_lidar2img(poses, CALIBS, CAM_NAMES, IMG_SIZE, FEAT_SHAPE)
    ref_cam, _mask = project_bev_anchors(ref_m, lidar2img, FEAT_SHAPE)

    for ci, name in enumerate(CAM_NAMES):
        pts = torch.cat([ref_m[0, :, 0, :], torch.ones(n, 1)], dim=-1)
        z_hom = (pts @ lidar2img[0, ci].T)[:, 2]  # 齐次投影的第三分量 = 相机系 z
        uv, z = project_pts(ref_m[0, :, 0, :], poses[0], _se(name), _k_scaled(name))
        # 深度一致性(全定义域)
        torch.testing.assert_close(z_hom, z, rtol=1e-4, atol=1e-4)
        legal = z > 1e-3  # 留出 eps 附近的余量,避免踩到钳位区
        assert int(legal.sum()) >= 5, "随机点里合法域样本太少,这条 oracle 会形同虚设"
        want = torch.stack([uv[:, 0] / FEAT_SHAPE[1], uv[:, 1] / FEAT_SHAPE[0]], dim=-1)
        torch.testing.assert_close(ref_cam[0, ci, legal, 0, :], want[legal], rtol=1e-4, atol=1e-5)
        # 域外(z ≤ eps)两者**必须**分歧 —— 否则说明钳位被去掉了(那会让掩码判据失真)
        if int((~legal).sum()) > 0:
            assert not torch.allclose(ref_cam[0, ci, ~legal, 0, :], want[~legal], rtol=1e-5, atol=1e-6), (
                "z ≤ eps 的区域不该与不钳位的实现一致"
            )


def test_projection_is_not_transposed() -> None:
    """★ 防「(w,h) 传反」:`scale_k` 的入参序与 `feat_shape` 的 (h,w) 不同位 ——
    若把 `build_lidar2img` 内部的翻转去掉,非方形特征图上 u/v 会互换,这里立刻失败。"""
    assert FEAT_SHAPE[0] != FEAT_SHAPE[1], "本仓库的防传反钉要求非方形特征图"
    ref_m = torch.tensor([[[[10.0, 0.0, 1.6]]]])  # 正前方 10 m 且与相机同高 ⇒ 落在主点上
    lidar2img = build_lidar2img(POSES, CALIBS, ["CAM_FRONT"], IMG_SIZE, FEAT_SHAPE)
    ref_cam, _ = project_bev_anchors(ref_m, lidar2img, FEAT_SHAPE)
    u = float(ref_cam[0, 0, 0, 0, 0]) * FEAT_SHAPE[1]
    v = float(ref_cam[0, 0, 0, 0, 1]) * FEAT_SHAPE[0]
    # 主点 ⇒ u/v 各自落在特征图**自己那一维**的中心;两维宽度不同(4 vs 8),
    # 故"传反"时 u 会落到 4(宽的中心变成高的中心),两个区间必有一个失守
    assert 0.45 * FEAT_SHAPE[1] < u < 0.55 * FEAT_SHAPE[1], f"u={u} 不在特征图宽度中心"
    assert 0.45 * FEAT_SHAPE[0] < v < 0.55 * FEAT_SHAPE[0], f"v={v} 不在特征图高度中心"


def test_visibility_mask_semantics() -> None:
    """可见掩码:前下方可见、相机后方不可见、镜头外不可见、浅深度大离轴不可见。

    **末条刻意钉住一个口径差异**:本模块照官方只判 `z > eps`,**没有** GKT 那道
    `depth > 0.5 m` 近平面;故"贴着镜头且在主点上"的点仍算可见。这是移植带来的
    与 GKT 不同的约定,如实钉住而不是当成 bug 抹平。
    """
    ref_m = torch.tensor(
        [
            [
                [[10.0, 0.0, 0.0]],  # 前相机前方 10 m 地面(相机高 1.6 m)→ 可见
                [[-10.0, 0.0, 0.0]],  # 前相机后方 → 不可见
                [[10.0, -500.0, 0.0]],  # 横向远出画幅 → 不可见
                [[0.1, 0.5, 0.0]],  # 浅深度 + 大离轴 ⇒ 像素跑出画幅 → 不可见
                [[10.0, 0.5, 0.0]],  # 10 m 处小幅离轴 → 仍在画幅内 → 可见
            ]
        ]
    )
    lidar2img = build_lidar2img(POSES, CALIBS, ["CAM_FRONT"], IMG_SIZE, FEAT_SHAPE)
    _ref_cam, mask = project_bev_anchors(ref_m, lidar2img, FEAT_SHAPE)
    got = mask[0, 0, :, 0].tolist()
    assert got == [True, False, False, False, True], got


def test_height_anchors_span_the_pillar() -> None:
    """高度锚点初值(偏移恒零)恰为 `linspace(zmin, zmax, D)` —— MapQR 的 `linspace(0, Z, D)` 口径。"""
    d, heads = 4, 4
    attn = HeightKernelAttention(embed_dims=8, n_heads=heads, num_cams=2, num_points_in_pillar=d)
    # 偏置初值恒零 —— 这是官方 `constant_init(height_offsets, 0)` 的口径
    assert float(attn.height_offsets.bias.abs().max()) == 0.0
    assert float(attn.height_offsets.weight.abs().max()) == 0.0
    n = 6
    query = torch.zeros(1, n, 8)
    ref3d = get_reference_points_3d(SMALL_BEV.bev_h, SMALL_BEV.bev_w, attn.height_range, d, torch.float32)
    ref_m = attn.refine_heights(query, ref3d[:n])  # (1, n, D, 3)
    z = ref_m[0, 0, :, 2]
    torch.testing.assert_close(z, torch.linspace(PC_RANGE_3D[2], PC_RANGE_3D[5], d))
    # xy 落在 pc_range 内
    assert float(ref_m[..., 0].min()) >= PC_RANGE_3D[0] - 1e-5
    assert float(ref_m[..., 0].max()) <= PC_RANGE_3D[3] + 1e-5


def test_learned_height_offset_shifts_by_normalized_units() -> None:
    """`height_offsets` 输出是**归一化**单位:加 1.0 ⇒ z 抬升整个柱高。"""
    attn = HeightKernelAttention(embed_dims=8, n_heads=2, num_cams=2, num_points_in_pillar=2)
    with torch.no_grad():
        attn.height_offsets.bias.fill_(0.0)
        attn.height_offsets.weight.zero_()
    query = torch.zeros(1, 1, 8)
    ref3d = torch.zeros(1, 2, 3)  # (N=1, D=2, 3) 全零 → 反归一化后 z 全 = zmin
    base = attn.refine_heights(query, ref3d)[0, 0, :, 2]
    torch.testing.assert_close(base, torch.full((2,), PC_RANGE_3D[2]))
    with torch.no_grad():
        attn.height_offsets.bias.fill_(1.0)  # 归一化 +1 ⇒ 米制 +z_span
    shifted = attn.refine_heights(query, ref3d)[0, 0, :, 2]
    torch.testing.assert_close(shifted, base + attn.height_range)


def test_height_offsets_get_gradient_from_bev() -> None:
    """可学习高度偏移真的进计算图(端到端一层)。

    **必须挑到确实可见的 BEV 单元**:若全部锚点都落在视场外,`slots` 恒为零、
    高度偏移自然拿不到梯度 —— 那时这条用例会以"梯度是 None"失败,而病根却在夹具
    选点上。故这里显式取**前方车道**上的单元,并先断言掩码非空。
    """
    attn = HeightKernelAttention(embed_dims=8, n_heads=2, num_cams=2, num_points_in_pillar=2)
    b = 1
    ref_all = get_reference_points_3d(SMALL_BEV.bev_h, SMALL_BEV.bev_w, attn.height_range, 2, torch.float32)
    # (iy=10 → y≈+1.5),ix∈{6,7,8} → x≈4.5/7.5/10.5,即相机正前方那片
    pick = torch.tensor([10 * SMALL_BEV.bev_w + k for k in (6, 7, 8)])
    ref3d = ref_all[pick]
    n = ref3d.shape[0]
    query = torch.randn(b, n, 8, requires_grad=True)
    feats = torch.randn(b, 2, 8, FEAT_SHAPE[0], FEAT_SHAPE[1])
    lidar2img = build_lidar2img(POSES, CALIBS, CAM_NAMES, IMG_SIZE, FEAT_SHAPE)
    _ref_cam, mask = project_bev_anchors(attn.refine_heights(query.detach(), ref3d), lidar2img, FEAT_SHAPE)
    assert bool(mask.any()), "夹具选点全在视场外 ⇒ 本用例失去意义,先修夹具"
    out = attn(query, ref3d, feats, lidar2img, FEAT_SHAPE)
    assert out.shape == query.shape
    out.sum().backward()
    for name, p in (
        ("height_offsets.weight", attn.height_offsets.weight),
        ("height_offsets.bias", attn.height_offsets.bias),
        ("output_proj.weight", attn.output_proj.weight),
    ):
        assert p.grad is not None, f"{name} 梯度缺失(该路径没进图)"
        assert float(p.grad.abs().sum()) > 0.0, f"{name} 梯度全零"


def test_kernel_grid_offsets_axes_not_transposed() -> None:
    """★ `kernel_size=(2,4)` ⇒ 恰 8 个偏移、互不相同,且**末维是 (dx, dy)**:
    第 0 列跨 4 个值(宽)、第 1 列跨 2 个值(高)。

    这条钉的是"把 (dx,dy) 写成 (dy,dx)"—— 该错不报错,只会让宽高方向的采样偏移互换
    (非方形特征图上立刻错、方形特征图上永远看不出来)。`(3,3)` 作为对照:两列都跨 3 个值,
    中心点 (0,0) 在集合内。
    """
    m = MSDeformableAttentionKernel(embed_dims=8, n_heads=2, kernel_size=(2, 4))
    offs = m.grid_offsets
    assert offs.shape == (8, 2)
    assert len({tuple(o.tolist()) for o in offs}) == 8, "偏移有重复"
    assert {int(v) for v in offs[:, 0]} == {-2, -1, 0, 1}, "第 0 列应跨 kernel 的**宽**(4 值)"
    assert {int(v) for v in offs[:, 1]} == {-1, 0}, "第 1 列应跨 kernel 的**高**(2 值)"
    m3 = MSDeformableAttentionKernel(embed_dims=8, n_heads=2, kernel_size=(3, 3))
    assert m3.grid_offsets.shape == (9, 2)
    assert {int(v) for v in m3.grid_offsets[:, 0]} == {-1, 0, 1}
    assert any(torch.equal(o, torch.zeros(2)) for o in m3.grid_offsets), "(3,3) 应含中心点"


def test_kernel_requires_anchor_count_equal_heads() -> None:
    """★ 官方那处隐式约束:`D == n_heads`(靠广播对齐),这里用断言显式化。

    不钉住的话,维度不符时广播会静默拼出 7 维张量,报错点会离病根很远。
    """
    with pytest.raises(ValueError, match="num_points_in_pillar"):
        HeightKernelAttention(embed_dims=8, n_heads=4, num_cams=2, num_points_in_pillar=3)
    m = MSDeformableAttentionKernel(embed_dims=8, n_heads=4, kernel_size=(2, 2))
    with pytest.raises(ValueError, match="高度锚点数"):
        m.sampling_locations(
            torch.zeros(1, 3, 3, 2),  # D=3 ≠ heads=4
            torch.zeros(1, 3, 8),
            [(FEAT_SHAPE[0], FEAT_SHAPE[1])],
        )


def test_reference_points_layout_and_range() -> None:
    """参考点:2D 是 (N,1,2)、3D 是 (N,D,3),都落在 [0,1];xy 的归一化分母与 BEV 网格一致。"""
    ref2 = get_reference_points_2d(SMALL_BEV.bev_h, SMALL_BEV.bev_w, torch.float32)
    assert ref2.shape == (SMALL_BEV.bev_h * SMALL_BEV.bev_w, 1, 2)
    assert float(ref2.min()) > 0.0 and float(ref2.max()) < 1.0
    ref3 = get_reference_points_3d(SMALL_BEV.bev_h, SMALL_BEV.bev_w, 4.0, 4, torch.float32)
    assert ref3.shape == (SMALL_BEV.bev_h * SMALL_BEV.bev_w, 4, 3)
    assert float(ref3.min()) >= 0.0 and float(ref3.max()) <= 1.0
    # 3D 的 z 维恰为 linspace(0, Z, D)/Z(官方 `linspace(0., Z, D)` 口径)
    torch.testing.assert_close(ref3[0, :, 2], torch.linspace(0.0, 1.0, 4))
    # 2D 与 3D 的 xy 必须逐点同序同值(两者都按 H 外/W 内展开)
    torch.testing.assert_close(ref2[:, 0, :], ref3[:, 0, :2])


def test_bev_encoder_rejects_mismatched_pc_range() -> None:
    """BEV 网格与 3D 高度范围的 xy 必须同源,否则两者会错位 —— 构造期直接报错。"""
    from autodrivedata.map.maptr.bevenc import BEVEncoder

    bad = BEVParams(pc_range=(-10.0, -20.0, 10.0, 20.0), bev_w=10, bev_h=20)
    with pytest.raises(ValueError, match="pc_range"):
        BEVEncoder(num_layers=1, embed_dims=8, n_heads=2, num_cams=2, num_points_in_pillar=2, bev=bad)

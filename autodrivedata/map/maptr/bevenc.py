"""BEV 编码器 —— MapQR 的 `HeightKernelAttention` + `with_height_refine`(纯 torch 移植)。

**它在链路里的位置**:GKT 负责"图像 → BEV"的提升(在 MapQR 的 config 里对应 `LSSTransform`),
本模块在其后做 **BEV 细化**:每层 = BEV 上的可变形自注意力 → `HeightKernelAttention`
(BEV 查询 → 图像特征的交叉注意力,带**可学习高度偏移**)→ FFN。**GKT 一字不改** ——
它的投影链是既有回归测试的 oracle(`tests/map/test_gkt.py`),动它会连带毁掉全链自证。

**`with_height_refine` 的真机制**(官方 `bevformer/modules/encoder.py` 的 23 行 diff):
原版把 BEV 柱上的 D 个高度锚点**固定**为 `linspace(0.5, Z−0.5, D)`,并在
`SpatialCrossAttention` 里先投影再交给注意力模块;MapQR 改成 `linspace(0, Z, D)`,
且**投影推迟到注意力模块内部**做,好让查询先加上一个可学习的 `height_offsets`
再投影。`height_offsets` 初值常零 ⇒ 训练起点仍是均匀锚点。

**★ 一处必须显式化的隐式约束**:官方 `MSDeformableAttentionKernel` 里参考点是
**(D 个高度锚点)**、偏移是 **(heads 个注意力头)**,两者靠广播对齐 —— 故必须
`num_heads == num_points_in_pillar`(官方 nuScenes config 正是 8 = 8)。语义即
**第 h 个头专门采第 h 个高度锚点**。这不是笔误(它产生了论文数字),如实移植,
但用断言把约束显式化:否则广播会静默拼出 7 维张量,报错点离病根很远。

**z 跨度取 (−2, 2)** = MapQR nuScenes config 的 `point_cloud_range`。该 config 的 BEV 网格
(x ∈ [−15,15] × y ∈ [−30,30],200×100 @ 0.3 m/px)与我们的 `BEV_DEFAULT` **逐位相同**,
只差 z —— 故这里不自己发明参数,引它。

**与官方的一处刻意分歧(正确性优先)**:官方用 `mask_per_img[0]` 算可见 query 下标,
即**所有 batch 元素都套用第 0 个样本的可见集合**(`bs=1` 时等价,`bs>1` 时错)。
本实现按 **per-batch** 算,并在文档里记明。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from autodrivedata.map.maptr.deform_attn import multi_scale_deform_attn
from autodrivedata.map.maptr.gkt import BEV_DEFAULT, BEVParams, cam_world_pose, scale_k

# MapQR nuScenes config 的 6 元组 (xmin, ymin, zmin, xmax, ymax, zmax)
PC_RANGE_3D: tuple[float, float, float, float, float, float] = (-15.0, -30.0, -2.0, 15.0, 30.0, 2.0)


def get_reference_points_3d(
    bev_h: int, bev_w: int, z_span: float, num_points_in_pillar: int, dtype: torch.dtype
) -> torch.Tensor:
    """BEV 单元 → 柱上 D 个高度锚点(**归一化** `[0,1]`),返回 (bev_h·bev_w, D, 3)。

    维度序取 (N, D, 3) 而非官方的 (D, N, 3) —— 本文件各处都用前者,**语义与官方一致**。
    归一化分母:z 用 `z_span`、x 用 `bev_w`、y 用 `bev_h`,与官方逐字对应
    (`zs` 的 `linspace(0, Z, D)` 是 MapQR 相对原版 BEVFormer 的改动,见模块头注)。
    """
    n = bev_h * bev_w
    zs = torch.linspace(0.0, z_span, num_points_in_pillar, dtype=dtype) / z_span
    xs = torch.linspace(0.5, bev_w - 0.5, bev_w, dtype=dtype) / bev_w
    ys = torch.linspace(0.5, bev_h - 0.5, bev_h, dtype=dtype) / bev_h
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W),行主序 = y outer / x inner
    ref = torch.stack(
        [
            xx.reshape(-1)[:, None].expand(n, num_points_in_pillar),
            yy.reshape(-1)[:, None].expand(n, num_points_in_pillar),
            zs[None, :].expand(n, num_points_in_pillar),
        ],
        dim=-1,
    )
    return ref.contiguous()  # (N, D, 3)


def get_reference_points_2d(bev_h: int, bev_w: int, dtype: torch.dtype) -> torch.Tensor:
    """BEV 单元中心的归一化坐标,返回 (bev_h·bev_w, 1, 2) —— 自注意力的参考点。"""
    ys = torch.linspace(0.5, bev_h - 0.5, bev_h, dtype=dtype) / bev_h
    xs = torch.linspace(0.5, bev_w - 0.5, bev_w, dtype=dtype) / bev_w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)[:, None, :].contiguous()


def denorm_3d(ref_norm: torch.Tensor, pc_range3d: tuple[float, ...]) -> torch.Tensor:
    """归一化 `[0,1]` 的 3D 点 → 米(官方 `point_sampling` 开头那三行的逆)。"""
    scale = ref_norm.new_tensor(
        [pc_range3d[3] - pc_range3d[0], pc_range3d[4] - pc_range3d[1], pc_range3d[5] - pc_range3d[2]]
    )
    shift = ref_norm.new_tensor([pc_range3d[0], pc_range3d[1], pc_range3d[2]])
    return ref_norm * scale + shift


def build_lidar2img(
    poses: torch.Tensor,
    calibs: dict[str, dict],
    cam_names: list[str],
    img_size: tuple[int, int],
    feat_shape: tuple[int, int],
) -> torch.Tensor:
    """(B, N_cam, 4, 4):**ego 局部系**点 → **特征图**像素(齐次)。

    **★ 必须把 ego→world 折进来**。BEV 锚点是 **ego 局部**坐标(我们的 pc_range 就是
    ego 系的),而 `gkt` 那条链是两步:`p_w = p_ego @ r_e + t_e` →
    `p_c = (p_w − t_cam) @ r_w2cᵀ`。只做第二步、把 ego 局部点当世界点投,**在 ego 位姿
    非恒等时全错**;而玩具夹具里位姿恒等,这个错**看不出来**(本仓库踩过的同类坑:
    `test_deform_attn` 头注记的"yaw≈0 的相机看着正常")。故这里显式复合两步。

    列向复合式(本函数给的是 `P34 @ [p_ego; 1]`):
    `R_col = r_w2c · r_eᵀ`、`t_col = r_w2c · (t_e − t_cam)`;再左乘按**特征分辨率**
    缩放的 K(缩放走既有 `scale_k`,与 GKT 同一步)。**不另写一套投影**。

    `img_size` / `feat_shape` 都是 **(宽, 高)**,与 `gkt.scale_k` 同序;本模块其余函数
    (`project_bev_anchors` 等)的 `feat_shape` 是 **(高, 宽)**,故此处显式翻转 ——
    传反不会报错,只会让投影错一位,`tests/map/test_bevenc.py` 有逐点对拍钉死。
    """
    feat_h, feat_w = feat_shape
    rows = []
    for name in cam_names:
        se = torch.tensor(calibs[name]["sensor2ego"], dtype=torch.float32, device=poses.device)
        k = torch.tensor(calibs[name]["intrinsic"], dtype=torch.float32, device=poses.device)
        k = scale_k(k, img_size, (feat_w, feat_h))  # 图像口径 K → 特征图口径(与 GKT 同一步)
        t_cam, r_e, r_w2c = cam_world_pose(poses, se)  # (B,3), (B,3,3), (B,3,3)
        # ego 局部 → 相机:两步复合。**约定不要混**:`gkt.project_pts` 是**行向**,
        # 且 `p_w` 那一步写的是 `einsum("...nj,...ij->...ni", p_ego, r_e)` = `p_ego @ r_eᵀ`
        # (列向即 `r_e @ p_ego`);本函数给的是列向 `P34 @ [p_ego; 1]`。两次转置里
        # 少一次/多一次都**不报错**,只会把锚点投到镜像或旋转错位的位置 ——
        # `tests/map/test_bevenc.py` 的「ego@real」逐点 oracle 与
        # 「投影矩阵必须随 ego 位姿变」两条钉死它(与 `test_deform_attn` 记的
        # "位姿换序,而 yaw≈0 的相机看着正常"是同一类坑)。
        r_col = r_w2c @ r_e  # (B,3,3);列向的 p_w = r_e·p_ego + t_e
        t_col = torch.einsum("bij,bj->bi", r_w2c, poses[:, :3] - t_cam)  # (B,3)
        rt = torch.cat([r_col, t_col[..., None]], dim=-1)  # (B,3,4)
        p34 = k @ rt  # (B,3,4)
        m = torch.zeros(p34.shape[0], 4, 4, dtype=p34.dtype, device=p34.device)
        m[:, :3, :4] = p34
        m[:, 3, 3] = 1.0
        rows.append(m)
    return torch.stack(rows, dim=1)  # (B, N_cam, 4, 4)


def project_bev_anchors(
    ref_m: torch.Tensor, lidar2img: torch.Tensor, feat_shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """米制 3D 锚点 (B, N, D, 3) → 每相机归一化像素 (B, C, N, D, 2) + 可见掩码 (B, C, N, D)。

    归一化到**特征图**尺寸(不是原图):`lidar2img` 用的就是特征分辨率的内参,
    两者配套才落在 `[0,1]`;特征图是原图的均匀降采样,故 `[0,1]` 在两种口径下等价。
    可见判据与官方同:`z > eps` 且 `0 < u,v < 1`。
    """
    feat_h, feat_w = feat_shape
    ones = torch.ones_like(ref_m[..., :1])
    pts = torch.cat([ref_m, ones], dim=-1)  # (B, N, D, 4)
    proj = torch.einsum("bcij,bndj->bcndi", lidar2img, pts)  # (B, C, N, D, 4)
    eps = 1e-5
    z = proj[..., 2:3]
    mask = (z > eps).squeeze(-1)
    uv = proj[..., :2] / z.clamp(min=eps)
    u = uv[..., 0] / feat_w
    v = uv[..., 1] / feat_h
    mask = mask & (v > 0.0) & (v < 1.0) & (u > 0.0) & (u < 1.0)
    return torch.stack([u, v], dim=-1), mask


class MSDeformableAttentionKernel(nn.Module):
    """BEV 查询 → 图像特征的交叉注意力(**固定** kernel 网格偏移,只学注意力权重)。

    与解码器那支([`decoder_mapqr.MapQRDecoderLayer`](decoder_mapqr.py))**不同**:官方这里把 `sampling_offsets`
    注释掉了,改用 `kernel_size=(2,4)` 的固定网格 —— "kernel" 一词的来由。
    """

    def __init__(
        self,
        embed_dims: int = 256,
        n_heads: int = 8,
        num_levels: int = 1,
        kernel_size: tuple[int, int] = (2, 4),
        dilation: int = 1,
        chunk: int = 0,
    ) -> None:
        super().__init__()
        if embed_dims % n_heads != 0:
            raise ValueError(f"embed_dims({embed_dims}) 必须能被 n_heads({n_heads}) 整除")
        self.embed_dims = embed_dims
        self.n_heads = n_heads
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.num_points = kernel_size[0] * kernel_size[1]
        self.chunk = chunk
        self.attention_weights = nn.Linear(embed_dims, n_heads * num_levels * self.num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        gh, gw = kernel_size
        ky = torch.arange(gh) - gh // 2
        kx = torch.arange(gw) - gw // 2
        # **末维必须是 (dx, dy)** —— 偏移要与参考点的 (u, v) 相加并除以 (W, H),
        # 写反不会报错,只会把宽高方向的采样偏移互换(非方形特征图上立刻错)。
        # 序与官方 `stack(meshgrid(x, y)).permute(1,2,0).reshape(-1,2)` 等价。
        offs = torch.stack(torch.meshgrid(kx, ky, indexing="ij"), dim=-1).reshape(-1, 2) * dilation
        self.register_buffer("grid_offsets", offs, persistent=False)  # (K, 2) 末维 (dx, dy)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=1.0)

    def sampling_locations(
        self, reference_points: torch.Tensor, query: torch.Tensor, spatial_shapes: list[tuple[int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """构造采样坐标 —— **注意与解码器那支的构造不同**(见模块头注)。

        `reference_points` (B, Nq, D, 2) 的 **D 维**与 `query` 导出的 **heads 维**广播对齐,
        故断言 `D == n_heads`:第 h 个头采第 h 个高度锚点。
        """
        b, nq, d, _ = reference_points.shape
        if d != self.n_heads:
            raise ValueError(
                f"高度锚点数 D={d} 必须等于注意力头数 n_heads={self.n_heads}"
                "(官方靠广播把两维对齐,见 bevenc 模块头注)"
            )
        k = self.grid_offsets.shape[0]
        so = self.grid_offsets.to(query.dtype).view(1, 1, 1, 1, k, 2)
        so = so.expand(b, nq, self.n_heads, self.num_levels, k, 2)
        w = self.attention_weights(query).view(b, nq, self.n_heads, self.num_levels * k)
        w = w.softmax(-1).view(b, nq, self.n_heads, self.num_levels, k)
        norm = torch.tensor([[sw, sh] for sh, sw in spatial_shapes], dtype=query.dtype, device=query.device)
        # 参考点 (B,Nq,D,2) → (B,Nq,D,1,1,2),与 (B,Nq,heads,1,K,2) 广播(D == heads,见断言)
        loc = reference_points[:, :, :, None, None, :] + so / norm[None, None, None, :, None, :]
        return loc, w

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """`query` (B, Nq, C);`value` (B, ΣH·W, C);`reference_points` (B, Nq, D, 2) ∈ [0,1]。"""
        loc, w = self.sampling_locations(reference_points, query, spatial_shapes)
        v = self.value_proj(value).view(value.shape[0], value.shape[1], self.n_heads, -1)
        return multi_scale_deform_attn(v, spatial_shapes, loc, w, chunk=self.chunk)


class BEVDeformableSelfAttn(nn.Module):
    """BEV 序列上的可变形自注意力(官方 `TemporalSelfAttention` 在单帧下的退化形式:
    没有 `prev_bev` ⇒ 退化为对当前 BEV 的自注意力)。偏移**可学习**(与 kernel 那支相反)。"""

    def __init__(self, embed_dims: int = 256, n_heads: int = 8, num_points: int = 4) -> None:
        super().__init__()
        if embed_dims % n_heads != 0:
            raise ValueError(f"embed_dims({embed_dims}) 必须能被 n_heads({n_heads}) 整除")
        self.embed_dims = embed_dims
        self.n_heads = n_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims, n_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, n_heads * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        for m in (self.sampling_offsets, self.attention_weights):
            nn.init.constant_(m.weight, 0.0)
            nn.init.constant_(m.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=1.0)

    def forward(self, bev_flat: torch.Tensor, ref_2d: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        """`bev_flat` (B, N, C);`ref_2d` (N, 1, 2) ∈ [0,1];`shape` = (H, W)。返回 (B, N, C)。"""
        b, n, _ = bev_flat.shape
        offs = self.sampling_offsets(bev_flat).view(b, n, self.n_heads, 1, self.num_points, 2)
        w = self.attention_weights(bev_flat)
        w = w.softmax(-1).view(b, n, self.n_heads, 1, self.num_points)
        sh, sw = shape
        norm = torch.tensor([[sw, sh]], dtype=bev_flat.dtype, device=bev_flat.device)
        loc = ref_2d[None, :, None, :, None, :] + offs / norm[None, None, None, :, None, :]
        v = self.value_proj(bev_flat).view(b, n, self.n_heads, self.embed_dims // self.n_heads)
        return multi_scale_deform_attn(v, [shape], loc, w)


class HeightKernelAttention(nn.Module):
    """BEV 查询 → 图像特征的空间交叉注意力,带**可学习高度偏移**(MapQR `with_height_refine`)。

    流程(官方同序):可学习高度偏移 → BEV 柱上 D 锚点 → 投影到各相机 → 按相机 rebatch
    (每台相机只与它可见的 BEV 查询交互,官方省显存手法)→ kernel 注意力 → 散加回
    BEV 查询槽并按"见过它的相机数"取均值 → 输出投影 + 残差。
    """

    def __init__(
        self,
        embed_dims: int = 256,
        n_heads: int = 8,
        num_cams: int = 6,
        num_points_in_pillar: int = 8,
        kernel_size: tuple[int, int] = (2, 4),
        pc_range3d: tuple[float, ...] = PC_RANGE_3D,
        dropout: float = 0.1,
        chunk: int = 0,
    ) -> None:
        super().__init__()
        if num_points_in_pillar != n_heads:
            raise ValueError(
                f"num_points_in_pillar({num_points_in_pillar}) 必须等于 n_heads({n_heads})"
                " —— 官方靠这一相等做广播对齐(见 bevenc 模块头注)"
            )
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range3d = pc_range3d
        self.height_range = pc_range3d[5] - pc_range3d[2]  # 柱高(米);偏移=归一化单位,乘它才是米
        self.dropout = nn.Dropout(dropout)
        self.height_offsets = nn.Linear(embed_dims, num_points_in_pillar)
        self.attention = MSDeformableAttentionKernel(
            embed_dims=embed_dims, n_heads=n_heads, kernel_size=kernel_size, chunk=chunk
        )
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        # 官方 `constant_init(height_offsets, 0)` ⇒ 训练起点是均匀高度锚点
        nn.init.constant_(self.height_offsets.weight, 0.0)
        nn.init.constant_(self.height_offsets.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=1.0)

    def refine_heights(self, query: torch.Tensor, ref3d_norm: torch.Tensor) -> torch.Tensor:
        """可学习高度偏移叠加 + 反归一化,返回**米制** 3D 锚点 (B, N, D, 3)。

        独立成方法是为了可单测:偏移初值恒零时,输出必须恰为 `linspace(0, Z, D)` 的均匀柱。
        """
        b = query.shape[0]
        d = self.num_points_in_pillar
        n = ref3d_norm.shape[0]
        heights = self.height_offsets(query).permute(0, 2, 1)  # (B, D, N)
        ref = ref3d_norm.permute(1, 0, 2)[None].expand(b, d, n, 3).clone()  # (B, D, N, 3)
        ref[..., 2] = ref[..., 2] + heights
        return denorm_3d(ref, self.pc_range3d).permute(0, 2, 1, 3)  # (B, N, D, 3)

    def forward(
        self,
        query: torch.Tensor,
        ref3d_norm: torch.Tensor,
        feats: torch.Tensor,
        lidar2img: torch.Tensor,
        feat_shape: tuple[int, int],
    ) -> torch.Tensor:
        """`query` (B,N,C);`ref3d_norm` (N,D,3);`feats` (B,Cam,C,h,w);`lidar2img` (B,Cam,4,4)。"""
        b, n, c = query.shape
        cam = feats.shape[1]
        if cam != self.num_cams:
            raise ValueError(f"相机数 {cam} 与模块声明的 {self.num_cams} 不一致")
        d = self.num_points_in_pillar
        ref_m = self.refine_heights(query, ref3d_norm)  # (B, N, D, 3) 米
        ref_cam, mask = project_bev_anchors(ref_m, lidar2img, feat_shape)  # (B,Cam,N,D,2), (B,Cam,N,D)
        # 按相机 rebatch:每台相机只与它可见的 query 交互(官方省显存手法)
        # ★ 官方此处用 `mask_per_img[0]` 即**所有 batch 套用第 0 样本**;本实现按 per-batch
        #   算(bs=1 时两者等价,bs>1 时官方那份是错的),见模块头注。
        indexes: list[list[torch.Tensor]] = []
        max_len = 1
        for bi in range(b):
            row = []
            for ci in range(cam):
                idx = mask[bi, ci].sum(-1).nonzero().squeeze(-1)
                row.append(idx)
                max_len = max(max_len, int(idx.numel()))
            indexes.append(row)
        q_re = query.new_zeros(b, cam, max_len, c)
        r_re = ref_cam.new_zeros(b, cam, max_len, d, 2)
        for bi in range(b):
            for ci in range(cam):
                idx = indexes[bi][ci]
                if idx.numel():
                    q_re[bi, ci, : idx.numel()] = query[bi, idx]
                    r_re[bi, ci, : idx.numel()] = ref_cam[bi, ci, idx]
        # kernel 注意力(固定网格偏移;六台相机并成一批算)
        value = feats.flatten(3).transpose(2, 3).reshape(b * cam, -1, c)  # (B·Cam, h·w, C)
        out = self.attention(
            q_re.reshape(b * cam, max_len, c),
            value,
            r_re.reshape(b * cam, max_len, d, 2),
            [feat_shape],
        ).view(b, cam, max_len, c)
        # 散加回 BEV 查询槽 + 按"见过它的相机数"取均值
        slots = query.new_zeros(b, n, c)
        for bi in range(b):
            for ci in range(cam):
                idx = indexes[bi][ci]
                if idx.numel():
                    slots[bi, idx] += out[bi, ci, : idx.numel()]
        count = (mask.sum(-1) > 0).sum(dim=1).clamp(min=1)  # (B, N):至少一个锚点可见的相机数
        slots = self.output_proj(slots / count[..., None])
        return self.dropout(slots) + query


class BEVEncoderLayer(nn.Module):
    """官方 `operation_order=('self_attn','norm','cross_attn','norm','ffn','norm')`,
    按 pre-norm 残差实现:`x = x + blk(norm(x))`。"""

    def __init__(
        self,
        embed_dims: int = 256,
        n_heads: int = 8,
        num_cams: int = 6,
        num_points_in_pillar: int = 8,
        kernel_size: tuple[int, int] = (2, 4),
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        chunk: int = 0,
    ) -> None:
        super().__init__()
        self.self_attn = BEVDeformableSelfAttn(embed_dims, n_heads=n_heads)
        self.cross_attn = HeightKernelAttention(
            embed_dims,
            n_heads=n_heads,
            num_cams=num_cams,
            num_points_in_pillar=num_points_in_pillar,
            kernel_size=kernel_size,
            dropout=dropout,
            chunk=chunk,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * ffn_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * ffn_ratio, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        x: torch.Tensor,
        ref_2d: torch.Tensor,
        ref3d_norm: torch.Tensor,
        feats: torch.Tensor,
        lidar2img: torch.Tensor,
        bev_shape: tuple[int, int],
        feat_shape: tuple[int, int],
    ) -> torch.Tensor:
        """`bev_shape` = BEV 栅格 (H, W);`feat_shape` = **相机特征图** (h, w)。

        **两个形状不能混**:自注意力在 BEV 栅格上采(分辨率的分子是 BEV 尺寸),
        交叉注意力在相机特征图上采(分母是特征图尺寸)。混用不会立刻报错维度,
        但会在 `value` 的 reshape 处炸 —— 或更糟:两者数值恰好可比时**静默采错位置**。
        """
        x = x + self.self_attn(self.norm1(x), ref_2d, bev_shape)
        x = x + self.cross_attn(self.norm2(x), ref3d_norm, feats, lidar2img, feat_shape)
        return x + self.ffn(self.norm3(x))


class BEVEncoder(nn.Module):
    """GKT 之后的 BEV 细化:N 层(自注意力 + HeightKernelAttention + FFN)。

    `bev` 直接来自 GKT —— 官方此处是**可学习 BEV 嵌入 + 学习位置编码**,我们用 GKT 输出
    顶替(再叠一个可学习位置编码)。这是**有意分歧**:GKT 已注入几何先验,故本模块是
    "细化"而非"提升";也正因如此 GKT 与其回归测试可以一字不动。

    `feats` 用相机名 → (B, C, h, w) 的 dict(与 GKT 同口径),内部按 `sorted` 定序成批次维,
    故调用方不必关心顺序。
    """

    def __init__(
        self,
        num_layers: int = 3,
        embed_dims: int = 256,
        n_heads: int = 8,
        num_cams: int = 6,
        num_points_in_pillar: int = 8,
        kernel_size: tuple[int, int] = (2, 4),
        bev: BEVParams = BEV_DEFAULT,
        pc_range3d: tuple[float, ...] = PC_RANGE_3D,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        chunk: int = 0,
    ) -> None:
        super().__init__()
        if (pc_range3d[0], pc_range3d[1], pc_range3d[3], pc_range3d[4]) != bev.pc_range:
            raise ValueError(
                f"bev.pc_range {bev.pc_range} 与 pc_range3d {pc_range3d} 的 xy 不一致 —— "
                "两者必须同源,否则 BEV 网格与高度锚点会错位"
            )
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.bev = bev
        self.z_span = pc_range3d[5] - pc_range3d[2]
        self.pc_range3d = pc_range3d
        self.pos_embed = nn.Parameter(torch.zeros(bev.bev_h * bev.bev_w, embed_dims))
        self.layers = nn.ModuleList(
            [
                BEVEncoderLayer(
                    embed_dims,
                    n_heads=n_heads,
                    num_cams=num_cams,
                    num_points_in_pillar=num_points_in_pillar,
                    kernel_size=kernel_size,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    chunk=chunk,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(embed_dims)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        dtype = self.pos_embed.dtype
        # 参考点由 BEV 网格与 pc_range 唯一决定 ⇒ 一次性算好存 buffer(不是参数)
        self.register_buffer(
            "ref3d_norm",
            get_reference_points_3d(bev.bev_h, bev.bev_w, self.z_span, num_points_in_pillar, dtype),
            persistent=False,
        )
        self.register_buffer("ref_2d", get_reference_points_2d(bev.bev_h, bev.bev_w, dtype), persistent=False)

    def forward(
        self,
        bev_feat: torch.Tensor,
        feats: dict[str, torch.Tensor],
        poses: torch.Tensor,
        calibs: dict[str, dict],
        img_size: tuple[int, int],
    ) -> torch.Tensor:
        """`bev_feat` (B, C, bev_h, bev_w) ← GKT;`feats`/`calibs`/`poses`/`img_size` 与 GKT 同口径。

        返回同形 BEV 特征 —— **契约与 GKT 输出一致**,故 head 无感。
        """
        if feats.keys() != calibs.keys():
            raise ValueError(f"相机集不一致: feats {sorted(feats)} vs calibs {sorted(calibs)}")
        b, c, bh, bw = bev_feat.shape
        if (bh, bw) != (self.bev.bev_h, self.bev.bev_w):
            raise ValueError(f"BEV 尺寸不符:收到 {(bh, bw)},期望 {(self.bev.bev_h, self.bev.bev_w)}")
        cam_names = sorted(feats)
        if len(cam_names) != self.num_cams:
            raise ValueError(f"相机数 {len(cam_names)} 与模块构造时声明的 {self.num_cams} 不一致")
        feat_shape = (feats[cam_names[0]].shape[-2], feats[cam_names[0]].shape[-1])
        for name in cam_names:
            if (feats[name].shape[-2], feats[name].shape[-1]) != feat_shape:
                raise ValueError(f"相机特征图尺寸不齐:{name} {feats[name].shape[-2:]} vs {feat_shape}")
        feats_stack = torch.stack([feats[n] for n in cam_names], dim=1)  # (B, Cam, C, h, w)
        lidar2img = build_lidar2img(poses, calibs, cam_names, img_size, feat_shape)
        # GKT 输出 + 可学习位置编码(官方此处是可学习 BEV 嵌入,我们用 GKT 顶替,见类文档)
        x = bev_feat.flatten(2).transpose(1, 2) + self.pos_embed[None]  # (B, N, C)
        for layer in self.layers:
            x = layer(x, self.ref_2d, self.ref3d_norm, feats_stack, lidar2img, (bh, bw), feat_shape)
        return self.out_norm(x).transpose(1, 2).reshape(b, c, bh, bw)

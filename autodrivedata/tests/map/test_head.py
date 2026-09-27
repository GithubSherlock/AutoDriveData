"""MapTR head 单测:匹配正确性 / 置换等价 / 代价语义 / 损失 / 前向形状 / 两套解码器。

末段是 **MapQR scatter-and-gather** 变体的钉:输出契约与默认分支同形(默认分支的参数名
集合不许变 —— 既有权重仍要能加载)、聚合是**拼接**而非均值(论文着眼点的结构钉)、
以及**归一化接线钉**(喂给 `sine_pos_embed` 的必须是 `[0,1]` 而**不是米**)。
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from autodrivedata.map.maptr.decoder import MapTRDecoderLayer
from autodrivedata.map.maptr.decoder_mapqr import MapQRDecoderLayer
from autodrivedata.map.maptr.gkt import BEV_DEFAULT
from autodrivedata.map.maptr.head import MapTRHead, maptr_loss, match_assign


def _line_forward() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)


def _line_reversed() -> np.ndarray:
    return _line_forward()[::-1]


def test_match_assign_basic() -> None:
    """类 1 一条 GT:点精确的 query 匹配成功,其余背景;类 0 无 GT 全背景。"""
    num_classes, num_vec = 2, 3
    gt = [[], [_line_forward()]]
    pred_points = np.zeros((num_classes * num_vec, 4, 2), dtype=np.float32)
    pred_points[4] = _line_forward()  # 类 1 组第 1 个 query 点精确
    pred_logits = np.full((num_classes * num_vec, num_classes + 1), -10.0)
    pred_logits[4, 2] = 10.0  # 类 1 logit 高
    cls_t, pts_t, mask = match_assign(pred_points, pred_logits, gt, num_classes, num_vec)
    np.testing.assert_array_equal(cls_t, [0, 0, 0, 0, 2, 0])
    np.testing.assert_array_equal(mask, [False, False, False, False, True, False])
    np.testing.assert_allclose(pts_t[4], _line_forward())


def test_match_assign_permutation_equivariant() -> None:
    """GT 存为反向:双向增强仍能零代价匹配,且目标取与预测一致的朝向(置换等价)。"""
    num_classes, num_vec = 1, 1
    gt = [[_line_reversed()]]
    pred_points = np.zeros((1, 4, 2), dtype=np.float32)
    pred_points[0] = _line_forward()
    pred_logits = np.array([[10.0, 10.0]])  # 类 1 logit 高
    cls_t, pts_t, mask = match_assign(pred_points, pred_logits, gt, num_classes, num_vec)
    assert cls_t[0] == 1 and mask[0]
    np.testing.assert_allclose(pts_t[0], _line_forward())


def test_match_assign_more_gt_than_queries() -> None:
    """GT 数 > query 数:方阵化 pad 后只匹配 num_vec 个(匈牙利不匹配 pad 行/列)。"""
    num_classes, num_vec = 1, 3
    gt = [[np.array([[i, 0.0], [i + 1, 0.0]], dtype=np.float32) for i in range(5)]]
    pred_points = np.zeros((3, 2, 2), dtype=np.float32)
    pred_points[1] = gt[0][2]
    pred_logits = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    _, _, mask = match_assign(pred_points, pred_logits, gt, num_classes, num_vec)
    assert mask.sum() == num_vec
    assert mask[1]  # 点精确的 query 必被匹配


def test_match_assign_cost_tradeoff() -> None:
    """代价 = −logit + 5·L1:点更准但置信低的 query 会被高置信 query 抢走匹配。"""
    num_classes, num_vec = 1, 2
    gt = [[_line_forward()]]
    pred_points = np.zeros((2, 4, 2), dtype=np.float32)
    pred_points[0] = _line_forward()  # 点完美
    pred_points[1] = _line_forward() + 10.0  # 点偏 10m
    pred_logits = np.array([[0.0, -20.0], [0.0, 100.0]])  # q0 类 0 置信极低,q1 极高
    cls_t, _, mask = match_assign(pred_points, pred_logits, gt, num_classes, num_vec)
    # q0: −logit = 20 + 0 = 20;q1: −100 + 5·10 = −50 → q1 胜
    assert mask[1] and not mask[0]
    assert cls_t[1] == 1


def test_maptr_loss() -> None:
    pred_logits = torch.randn(2, 4, 3)
    pred_points = torch.randn(2, 4, 4, 2)
    cls_targets = torch.tensor([[0, 1, 0, 0], [2, 0, 0, 0]])
    pts_mask = torch.tensor([[False, True, False, False], [True, False, False, False]])
    pts_targets = pred_points.detach().clone()  # 与预测一致 → pts loss ≈ 0
    out = maptr_loss(pred_logits, pred_points, cls_targets, pts_targets, pts_mask)
    assert set(out) == {"cls", "pts", "total"}
    assert float(out["pts"]) < 1e-5
    assert bool(torch.isfinite(out["total"]))
    assert (
        float(out["total"]) == float(out["cls"] + out["pts"])
        or abs(float(out["total"] - out["cls"] - out["pts"])) < 1e-6
    )


def test_head_forward_shapes_and_backward() -> None:
    head = MapTRHead(num_classes=2, num_vec=3, num_pts=4, num_layers=2, embed_dims=32, bev_dims=16)
    bev = torch.randn(1, 16, 20, 10)
    out = head(bev)
    assert out["pred_logits"].shape == (1, 6, 3)
    assert out["pred_points"].shape == (1, 6, 4, 2)
    (out["pred_logits"].sum() + out["pred_points"].sum()).backward()
    assert head.anchor.grad is not None and bool(torch.isfinite(head.anchor.grad).all())


def test_anchor_init_within_bev() -> None:
    head = MapTRHead(num_classes=4, num_vec=50, num_pts=20)
    a = head.anchor.detach()
    assert a.shape == (200, 20, 2)
    # 锚线在 BEV 范围内(x ∈ [−15, 15], y ∈ [−30, 30])
    assert bool(torch.all(a[..., 0] >= -15.0)) and bool(torch.all(a[..., 0] <= 15.0))
    assert bool(torch.all(a[..., 1] >= -30.0)) and bool(torch.all(a[..., 1] <= 30.0))


# --------------------------- MapQR scatter-and-gather 变体 ---------------------------

_SG_KW = dict(num_classes=2, num_vec=3, num_pts=4, num_layers=2, embed_dims=32, bev_dims=16)


def test_scatter_gather_output_contract_matches_default() -> None:
    """两套解码器的输出契约逐字段同形 —— 下游(`train_maptr`/`eval_maptr`/逐帧契约)无感。"""
    torch.manual_seed(0)
    default = MapTRHead(**_SG_KW, scatter_gather=False)(torch.randn(2, 16, 20, 10))
    torch.manual_seed(0)
    sg = MapTRHead(**_SG_KW, scatter_gather=True)(torch.randn(2, 16, 20, 10))
    assert set(default) == set(sg) == {"pred_logits", "pred_points"}
    for k in default:
        assert default[k].shape == sg[k].shape, k
        assert bool(torch.isfinite(sg[k]).all()), k
    assert default["pred_logits"].shape == (2, 6, 3)
    assert default["pred_points"].shape == (2, 6, 4, 2)


def test_default_branch_param_names_unchanged() -> None:
    """默认分支的参数名集合不变 ⇒ **既有权重仍可加载**;MapQR 专属参数只在开分支时出现。

    这是「加变体不破坏存量」的机械钉:`load_map_weights` 对多余/缺失键报错,
    所以参数名一漂移,旧权重立刻加载失败。
    """
    default = set(dict(MapTRHead(**_SG_KW).named_parameters()))
    assert "point_embed.weight" in default
    assert not any("pt_pos_proj" in n or "sampling_offsets" in n for n in default)
    sg = set(dict(MapTRHead(**_SG_KW, scatter_gather=True).named_parameters()))
    assert "point_embed.weight" not in sg
    assert any("pt_pos_proj" in n for n in sg)
    assert any("sampling_offsets" in n for n in sg)
    # 两分支共享的部分(instance_embed / anchor / cls_branch / ins_attn)必须同名
    shared = {"instance_embed.weight", "anchor", "cls_branch.weight", "cls_branch.bias"}
    assert shared <= default and shared <= sg


def test_scatter_gather_aggregation_is_concat_not_mean() -> None:
    """★ 结构钉:聚合输入的维度必须是 **P·C**(拼接),不是 C(均值)。

    钉的就是论文的着眼点本身 —— 均值会抹平同一要素内各点的内容差异。
    若把 `output_proj` 换回「先 mean 再线性」,输入维会掉回 C,这里立刻失败。
    """
    head = MapTRHead(
        num_classes=1, num_vec=1, num_pts=4, num_layers=1, embed_dims=32, bev_dims=16, scatter_gather=True
    )
    assert head.layers[0].output_proj[0].in_features == 4 * 32  # P · C
    assert head.layers[0].reg_branch[-1].out_features == 2 * 4  # 一次出全 P 个点


def test_scatter_gather_learned_sampling_and_sine_prior_get_gradients() -> None:
    """可学习采样偏移、参考点正弦位置先验、点查询投影都真的进了计算图。"""
    torch.manual_seed(0)
    head = MapTRHead(**_SG_KW, scatter_gather=True)
    out = head(torch.randn(2, 16, 20, 10))
    (out["pred_logits"].sum() + out["pred_points"].sum()).backward()
    layer = head.layers[0]
    for name, param in (
        ("sampling_offsets.weight", layer.sampling_offsets.weight),
        ("sampling_offsets.bias", layer.sampling_offsets.bias),
        ("attention_weights.weight", layer.attention_weights.weight),
        ("pt_pos_proj.weight", layer.pt_pos_proj.weight),
    ):
        g = param.grad
        assert g is not None, f"{name} 梯度缺失(该路径没进图)"
        assert bool(torch.isfinite(g).all()), f"{name} 梯度非有限"
    anchor_grad = head.anchor.grad
    assert anchor_grad is not None and bool(torch.isfinite(anchor_grad).all())


def test_scatter_gather_feeds_unit_interval_to_sine(monkeypatch) -> None:
    """★ 归一化**接线**钉:解码器喂给 `sine_pos_embed` 的必须是 `[0,1]`,而不是米。

    与 `test_deform_attn.test_meter_scale_sine_differs_from_normalized` 互补 ——
    那条钉 `sine_pos_embed` 自己的语义,这条钉**调用点**有没有先归一化。
    漏掉归一化不会报错,只会让正弦频率别名(症状是「训不动」),所以必须机械钉死。

    **替换的是 `decoder_mapqr` 模块**(解码器所在文件)的符号,不是 `head` ——
    本钉绑的是**调用点**,换错模块会静静地钉到空气上(替换到未被调用的符号,
    断言 `"pos" in seen` 会失败,但如果连那个断言也写错就成了假绿)。
    """
    import autodrivedata.map.maptr.decoder_mapqr as decoder_mod

    seen: dict[str, torch.Tensor] = {}
    real = decoder_mod.sine_pos_embed

    def spy(pos, dim):  # noqa: ANN001, ANN202 - 测试替身,签名与真身一致
        seen["pos"] = pos.detach().clone()
        return real(pos, dim)

    monkeypatch.setattr(decoder_mod, "sine_pos_embed", spy)
    layer = MapQRDecoderLayer(embed_dims=32, n_heads=2, bev_dims=16, num_pts=4, n_attn_heads=2, num_points=2)
    # 参考点**取在 pc_range 内**,这样归一化后必然落在 [0,1]
    xmin, ymin, xmax, ymax = BEV_DEFAULT.pc_range
    span = torch.tensor([xmax - xmin, ymax - ymin])
    ref = torch.rand(2, 3, 4, 2) * span + torch.tensor([xmin, ymin])
    layer(torch.randn(3, 2, 32), ref, torch.randn(2, 16, 20, 10), None, BEV_DEFAULT.pc_range)

    assert "pos" in seen, "sine_pos_embed 未被调用 ⇒ 位置先验路径断了"
    assert float(seen["pos"].min()) >= 0.0 and float(seen["pos"].max()) <= 1.0, (
        "喂进 sine_pos_embed 的不是 [0,1] 归一化坐标(疑似直接用了米)"
    )


def test_scatter_gather_reference_points_stay_in_meters() -> None:
    """回归输出仍是**米**(有意分歧):参考点的数量级由 pc_range 决定,不是 [0,1]。"""
    torch.manual_seed(0)
    head = MapTRHead(
        num_classes=2, num_vec=3, num_pts=4, num_layers=2, embed_dims=32, bev_dims=16, scatter_gather=True
    )
    pts = head(torch.randn(1, 16, 20, 10))["pred_points"].detach()
    # 初始锚线铺在 ±15/±30 内 ⇒ 若回归被误改成归一化空间,数值会塌到 0~1
    assert float(pts.abs().max()) > 2.0, f"参考点疑似落进 [0,1] 归一化空间:{float(pts.abs().max())}"


# --------------------- 变体的文件分离(结构钉) ---------------------


def _imported_modules(path: str) -> list[str]:
    """文件里**真实 import** 的模块名(AST 解析)。

    不能用子串匹配:docstring 里提到 `decoder_mapqr` 是**文档**,不是依赖 ——
    拿文本当依赖判据会把"描述"误判成"耦合"(本文件头注就点了名)。
    """
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    return mods


def test_variants_live_in_separate_modules() -> None:
    """★ 两套解码器**各在各的文件**,且依赖**单向**(默认线不 import MapQR 线)。

    这条把「各写各的文件、要用时可以选择」这个结构本身钉住:把两支塞回同一个文件、
    或让默认线反向依赖 MapQR 线(那样删掉默认线就带崩 MapQR 线),这里立刻失败。
    `head.py` 是装配层,它 import 两者是**允许的**。
    """
    from autodrivedata.map.maptr import decoder as decoder_mod
    from autodrivedata.map.maptr import decoder_mapqr as mapqr_mod

    assert MapTRDecoderLayer.__module__ == "autodrivedata.map.maptr.decoder"
    assert MapQRDecoderLayer.__module__ == "autodrivedata.map.maptr.decoder_mapqr"
    base_imports = _imported_modules(decoder_mod.__file__)
    assert not any("decoder_mapqr" in m for m in base_imports), (
        f"默认线不得 import MapQR 线(依赖须单向),实测 import 了 {base_imports}"
    )
    # MapQR 线复用默认线的 FFN 构件 —— 单向依赖的**正向**这一半也要成立
    assert any(m == "autodrivedata.map.maptr.decoder" for m in _imported_modules(mapqr_mod.__file__))


def test_head_picks_decoder_class_by_flag() -> None:
    """开关换的确实是**解码器类**,不是只置了个没人读的字段。"""
    assert isinstance(MapTRHead(**_SG_KW).layers[0], MapTRDecoderLayer)
    assert isinstance(MapTRHead(**_SG_KW, scatter_gather=True).layers[0], MapQRDecoderLayer)
    assert len({type(x) for x in MapTRHead(**_SG_KW).layers}) == 1

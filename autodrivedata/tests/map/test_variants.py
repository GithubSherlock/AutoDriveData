"""变体表单测:展开 / 冲突规则 / 权重归属 / **结构必须自洽**。

核心是**归属往返**:用 `--variant mapqr` 训出来的权重,消费方(eval / viz / studio)
必须能**只凭 checkpoint** 建出同一个结构。这条不成立的话,那些权重就没法评 ——
而失败形态是"键不匹配报错"而非静默,所以这里既钉**正向能认出**,也钉**反向报错**。
"""

from __future__ import annotations

import pytest
import torch

from autodrivedata.map.maptr.model import MapTR, load_map_weights
from autodrivedata.map.maptr.variants import (
    VARIANT_BASELINE,
    VARIANT_MAPQR,
    VARIANTS,
    _name_of,
    build_map_model,
    expand_variant,
    load_map_model,
    read_map_meta,
    resolve_variant,
    save_map_checkpoint,
    variant_names,
)

# 小配置:只为把结构建起来比参数名,不做前向。`pretrained=False` 免下载 ImageNet。
SMALL = dict(
    num_classes=2,
    embed_dims=16,
    num_vec=2,
    num_pts=2,
    num_layers=1,
    bev_encoder_layers=1,
    bev_encoder_heads=8,
)


def _param_names(model: MapTR) -> set[str]:
    return set(dict(model.named_parameters()))


def test_variant_table_expands_to_the_two_switches() -> None:
    assert expand_variant(VARIANT_MAPQR) == {"scatter_gather": True, "bev_encoder": "height_kernel"}
    assert expand_variant(VARIANT_BASELINE) == {"scatter_gather": False, "bev_encoder": "none"}
    assert variant_names() == (VARIANT_BASELINE, VARIANT_MAPQR)


def test_unknown_variant_errors_instead_of_falling_back() -> None:
    """未知变体**报错**,不静默退回基线 —— 退回会让"我明明选了 mapqr"变成假象。"""
    with pytest.raises(SystemExit, match="未知变体"):
        expand_variant("mapqr_v2")


def test_aggregate_and_fine_grained_switches_are_mutually_exclusive() -> None:
    """聚合开关与细粒度开关**同时给报错**:不定义"谁覆盖谁"(隐式规则是本项目反复吃亏处)。"""
    assert resolve_variant(None) == {"scatter_gather": False, "bev_encoder": "none"}
    assert resolve_variant(None, scatter_gather=True) == {"scatter_gather": True, "bev_encoder": "none"}
    assert resolve_variant(VARIANT_MAPQR) == {"scatter_gather": True, "bev_encoder": "height_kernel"}
    with pytest.raises(SystemExit, match="互斥"):
        resolve_variant(VARIANT_MAPQR, scatter_gather=True)
    with pytest.raises(SystemExit, match="互斥"):
        resolve_variant(VARIANT_MAPQR, bev_encoder="height_kernel")


def test_model_kwargs_rebuilds_the_same_architecture() -> None:
    """`model_kwargs()` 必须**完整** —— 漏一项就会建出别的结构(而键不匹配只在事后才报)。"""
    for variant in variant_names():
        m = build_map_model(variant, pretrained=False, **SMALL)
        again = MapTR(pretrained=False, **m.model_kwargs())
        assert _param_names(again) == _param_names(m), f"{variant}: model_kwargs 往返后结构变了"
        for k, v in m.state_dict().items():
            assert again.state_dict()[k].shape == v.shape, f"{variant}: {k} 形状不一致"


@pytest.mark.parametrize("variant", [VARIANT_BASELINE, VARIANT_MAPQR])
def test_checkpoint_roundtrip_identifies_the_variant(tmp_path, variant: str) -> None:
    """★ 归属往返:存盘 → 只凭 checkpoint 读懂结构 → 建出**同名参数集**的模型。"""
    m = build_map_model(variant, pretrained=False, **SMALL)
    path = tmp_path / f"{variant}.pt"
    save_map_checkpoint(m, path)

    meta = read_map_meta(path)
    assert meta["name"] == variant
    assert meta["source"] == "self-describing"

    loaded, meta2 = load_map_model(path, torch.device("cpu"))
    assert meta2["name"] == variant
    assert _param_names(loaded) == _param_names(m)


def test_legacy_bare_state_dict_needs_explicit_architecture(tmp_path) -> None:
    """**向后兼容的边界**(钉住真实行为,不是许愿):裸 state_dict **不带**架构信息。

    - 变体归为基线是**事实**不是猜测:开关是 2026-09-27 才加的,之前落盘的权重只可能是基线结构;
    - 但 `num_vec` / `embed_dims` 这类**没说出口的**参数它同样不带 ⇒ 裸权重必须由调用方
      用 `**overrides` 补上。
    - **补不齐会当场报错**(形状不符),不会静默跑错 —— 这条与正向兼容同等重要。
    """
    m = build_map_model(VARIANT_BASELINE, pretrained=False, **SMALL)
    path = tmp_path / "legacy.pt"
    torch.save(m.state_dict(), path)  # 旧形态:裸 state_dict

    meta = read_map_meta(path)
    assert meta["name"] == VARIANT_BASELINE
    assert meta["model_kwargs"] == {}  # 架构信息为零 —— 这正是必须显式补的原因
    assert meta["source"] == "legacy-bare-state-dict"

    loaded, _ = load_map_model(path, torch.device("cpu"), **SMALL)  # 显式补架构
    assert _param_names(loaded) == _param_names(m)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_map_model(path, torch.device("cpu"))  # 不给 = 按默认结构建 ⇒ 响亮地失败


def test_wrong_structure_is_rejected_loudly(tmp_path) -> None:
    """★ 反向钉:拿**基线结构**去载 MapQR 权重必须报错,而不是静默跑错模型。"""
    m = build_map_model(VARIANT_MAPQR, pretrained=False, **SMALL)
    path = tmp_path / "mapqr.pt"
    save_map_checkpoint(m, path)

    baseline = build_map_model(VARIANT_BASELINE, pretrained=False, **SMALL)
    with pytest.raises(SystemExit, match="权重"):
        load_map_weights(baseline, str(path), torch.device("cpu"))


def test_custom_flag_combination_is_not_mislabelled() -> None:
    """只开一半的消融配置**不许**被标成 `mapqr` —— 标错会让日志里的归属失真。"""
    assert _name_of({"scatter_gather": True, "bev_encoder": "height_kernel"}) == VARIANT_MAPQR
    assert _name_of({"scatter_gather": False, "bev_encoder": "none"}) == VARIANT_BASELINE
    assert _name_of({"scatter_gather": True, "bev_encoder": "none"}) == "custom"
    assert _name_of({}) == VARIANT_BASELINE  # 缺键 = 全默认


def test_variants_cover_the_model_switches() -> None:
    """表里的键必须都是 `MapTR` 认识的构造参数 —— 写错键会在建模型时才炸,离病根很远。"""
    from inspect import signature

    params = set(signature(MapTR.__init__).parameters)
    for name, spec in VARIANTS.items():
        assert set(spec) <= params, f"{name} 含未知构造参数 {set(spec) - params}"

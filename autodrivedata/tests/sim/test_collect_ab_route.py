"""`collect_ab_route` 的**结构回归钉**:A/B 硬门槛不许被新功能挤掉。

## 钉的是什么

P1-6b 往采集器里加了**会改变场景内容**的一档(`--occluders`)。它踩得最深的两处都不是
"会崩"的那类:

1. **道具进了 GT** ⇒ A/B 两侧 GT 计数不等 ⇒ 帧级配对失效(红线第一条)。防线是那条
   `type_id.startswith(("vehicle","walker"))` 过滤 —— 它一旦被"顺手放宽"就静默破防。
   本组用 AST 把过滤器的**前缀集合**读出来对表(`TestGtFilter`),而不是跑一遍看数。
2. **`--occluders` 默认不是 `none`** ⇒ 所有**历史** P1 数据集与开关前的口径不再可比
   (那正是"跨档不可混比"那条)。判据 = argparse 里 `default` 逐字是 `"none"`。

再钉两条纯函数:`is_cleanup_target`(清场/收尾**同一个谓词**,漏一处就是"清一半")与
`dims_match`(声明 ≠ 渲染时要**报错**而不是继续)。
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from autodrivedata.sim.collect_ab_route import dims_match, is_cleanup_target
from autodrivedata.sim.occlusion import OCCLUDER_DIMS, OCCLUDER_MODELS

_PKG = Path(__file__).resolve().parents[2]  # autodrivedata/
_COLLECTOR = _PKG / "sim" / "collect_ab_route.py"
_SRC = _COLLECTOR.read_text(encoding="utf-8")
_TREE = ast.parse(_SRC)


def _add_argument_calls(name: str) -> list[ast.Call]:
    """源码里所有 `add_argument("<name>", ...)` 调用(可能不止一处)。"""
    out = []
    for node in ast.walk(_TREE):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        a0 = node.args[0]
        if isinstance(a0, ast.Constant) and a0.value == name:
            out.append(node)
    return out


def _kw(call: ast.Call, key: str):
    for k in call.keywords:
        if k.arg == key:
            return k.value
    return None


def _startswith_prefix_sets() -> list[frozenset[str]]:
    """源码里每个 `X.startswith((...))` 的**字面量前缀集合**(非字面量的调用跳过)。"""
    out: list[frozenset[str]] = []
    for node in ast.walk(_TREE):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "startswith" or not node.args:
            continue
        a0 = node.args[0]
        elts = a0.elts if isinstance(a0, ast.Tuple) else [a0]
        if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in elts):
            out.append(frozenset(e.value for e in elts))
    return out


class TestGtFilter:
    """★ A/B 硬门槛:`label_2` 的准入前缀必须是 `{vehicle, walker}` 且**只有**这两个。"""

    def test_the_gt_filter_exists_and_excludes_controller(self):
        """收 GT 的是 `startswith("vehicle") or startswith("walker")` **两个调用**;
        `controller` 只在清场那一处 —— 两者混用就会把道具或红绿灯写进 GT。"""
        sets = _startswith_prefix_sets()
        assert frozenset({"vehicle"}) in sets, "找不到收 GT 的 vehicle 分支"
        assert frozenset({"walker"}) in sets, "找不到收 GT 的 walker 分支"
        assert frozenset({"controller"}) not in sets, "controller 进了 GT 过滤器"

    def test_cleanup_filter_is_wider_than_the_gt_filter(self):
        assert frozenset({"vehicle", "walker", "controller"}) in _startswith_prefix_sets()

    @pytest.mark.parametrize("mode", sorted(OCCLUDER_MODELS))
    def test_occluder_blueprints_are_outside_the_gt_filter(self, mode: str):
        """道具 `static.prop.*` 与前缀集**无交集** ⇒ 结构上进不了 `label_2`。

        这条不是"读代码确认过了":它把 `OCCLUDER_MODELS` 与**从源码 AST 读出来的**
        前缀集当面对表 —— 以后谁把 `static.prop` 加进过滤元组,这里当场红。
        """
        mid = OCCLUDER_MODELS[mode]
        assert not mid.startswith(("vehicle", "walker")), (
            f"{mid} 会命中 GT 过滤器 —— A/B 的 GT 相等立刻不成立"
        )
        # 与源码里**每一个**字面量前缀集都对一遍:将来谁把 `static.prop` 加进其中任何一个
        # (包括更宽的那个收尾过滤器),这里都当场红。
        for prefixes in _startswith_prefix_sets():
            assert not mid.startswith(tuple(prefixes)), (
                f"{mid} 会命中 `startswith{sorted(prefixes)}` —— 确认那处是不是在收 GT"
            )


class TestDefaults:
    """新开关的默认值 = **历史口径**。默认一变,库里所有老 P1 数据集就不再可比。"""

    def test_occluders_defaults_to_none(self):
        calls = _add_argument_calls("--occluders")
        assert len(calls) == 1, f"--occluders 应当只注册一次,实际 {len(calls)}"
        d = _kw(calls[0], "default")
        assert isinstance(d, ast.Constant) and d.value == "none", "默认必须是 none(加开关前的行为)"

    def test_occluders_choices_cover_the_registry(self):
        calls = _add_argument_calls("--occluders")
        ch = _kw(calls[0], "choices")
        vals = {e.value for e in ch.elts}
        assert vals == {"none", *OCCLUDER_MODELS}, f"choices 与注册表对不上:{vals}"

    def test_occluder_cars_defaults_to_all(self):
        calls = _add_argument_calls("--occluder-cars")
        assert len(calls) == 1
        d = _kw(calls[0], "default")
        assert isinstance(d, ast.Constant) and d.value == "all", "默认必须是加这个开关前的行为"

    def test_nearest_keeps_the_closest_car_only(self):
        """★ `nearest` 的判据是**按前向距离取最近的**,不是按列表顺序 —— 顺序一变,
        被挡的那台就从"唯一落在断崖上的"变成别的车,而产出照样跑得完、照样出数。"""
        offs = []
        for node in ast.walk(_TREE):
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "STATIC_OFFSETS" for t in node.targets
            ):
                offs = [e.value for e in node.value.elts]
        assert offs == [20.0, 35.0, 50.0, 62.0], f"P1 冻结布局变了:{offs}"
        assert offs == sorted(offs), "STATIC_OFFSETS 不再按前向距离递增 ⇒ `cars[:1]` 不再是最近那台"

    def test_gap_takes_its_default_from_the_geometry_module(self):
        """`--occluder-gap` 的默认值必须**引用** `occlusion.OCCLUDER_GAP`,不许另写一个字面量
        —— 两处各写一份的话,调了模块常量而 CLI 没跟着变,声明与实摆就分叉了。"""
        calls = _add_argument_calls("--occluder-gap")
        assert len(calls) == 1
        d = _kw(calls[0], "default")
        assert isinstance(d, ast.Name) and d.id == "OCCLUDER_GAP", ast.dump(d)


class TestCleanupPredicate:
    """清场与收尾共用一个谓词 —— 漏一处就是"清一半"这类不可见失败。"""

    @pytest.mark.parametrize("mid", sorted(OCCLUDER_MODELS.values()))
    def test_occluders_are_cleanup_targets(self, mid: str):
        assert is_cleanup_target(mid), f"{mid} 不会被清掉 —— 下一轮 A 侧会带着上一轮的墙"

    @pytest.mark.parametrize("tid", ["vehicle.tesla.model3", "walker.pedestrian.0001", "controller.aiwalker"])
    def test_the_three_original_classes_still_covered(self, tid: str):
        assert is_cleanup_target(tid)

    @pytest.mark.parametrize("tid", ["traffic.traffic_light", "spectator", "sensor.camera.rgb"])
    def test_world_assets_are_not_touched(self, tid: str):
        """反向对照:`traffic.*` 是**地图自带的**(实测默认世界 23 个 actor 全是它们),
        收走它们就改了场景,而这条只在 A/B 出数后看得出来。"""
        assert not is_cleanup_target(tid)

    def test_the_predicate_is_actually_used_twice(self):
        """清场 + 收尾**都**要走它(两处 inline 展开的话,改一处漏一处)。"""
        assert _SRC.count("is_cleanup_target(") >= 3, "调用点少于 2 处(第 3 次是定义处)"


class TestDimsMatch:
    """声明 = 渲染:实 spawn 的尺寸对不上就**停**,不许带着错的可见性数字采完。"""

    @staticmethod
    def _bb(ln: float, th: float, h: float) -> SimpleNamespace:
        return SimpleNamespace(extent=SimpleNamespace(x=ln / 2, y=th / 2, z=h / 2))

    @pytest.mark.parametrize("mode", sorted(OCCLUDER_DIMS))
    def test_exact_dims_pass(self, mode: str):
        assert dims_match(self._bb(*OCCLUDER_DIMS[mode]), OCCLUDER_DIMS[mode])

    def test_swapped_length_and_thickness_fails(self):
        """★ 反向对照:长/厚互换**不报错也不会崩**,只会让墙变窄 —— 必须被拦下。"""
        ln, th, h = OCCLUDER_DIMS["partial"]
        assert not dims_match(self._bb(th, ln, h), (ln, th, h))

    def test_tiny_drift_passes_but_centimetre_does_not(self):
        ln, th, h = OCCLUDER_DIMS["partial"]
        assert dims_match(self._bb(ln + 5e-4, th, h), (ln, th, h))
        assert not dims_match(self._bb(ln + 0.01, th, h), (ln, th, h))


class _BP:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attrs: dict = {}

    def set_attribute(self, k: str, v: object) -> None:
        self.attrs[k] = v


class _Lib:
    """假的 `carla.BlueprintLibrary`:只记"谁被建了、设了哪些属性"。"""

    def __init__(self) -> None:
        self.made: list[_BP] = []

    def find(self, name: str) -> _BP:
        b = _BP(name)
        self.made.append(b)
        return b


class TestDepthChannel:
    """`--depth`(2026-10-06 加):「真值深度 + `label_2` 同源」的唯一来源。

    它坏掉的症状**不是报错**,而是"GT 框与画面配不上" —— 而那与"标定错了"长得一样。
    ⇒ 两条必须机械钉住:**默认关**(否则与归档不可比)、**属性与 RGB 逐字相同**。
    """

    def test_default_is_off(self):
        """★ 默认关 ⇒ 关着时产物与归档**逐字节一致**。
        默认一旦翻了,所有历史 P1 数据集与开关后的口径不再可比(同 `--occluders` 那条)。"""
        calls = _add_argument_calls("--depth")
        assert len(calls) == 1
        a = _kw(calls[0], "action")
        assert isinstance(a, ast.Constant) and a.value == "store_true", "`--depth` 应当是 store_true 开关"
        assert _kw(calls[0], "default") is None, "store_true 本来就没有 default(即 False),别多写一个"

    def test_depth_blueprint_takes_exactly_cam_attrs(self):
        """★ 挂点与 fov 是「像素对齐」的**唯一落点**:三项逐字取 `CAM_ATTRS`(RGB 那路同一份常量)。"""
        from autodrivedata.sim import collect_ab_route as M
        from autodrivedata.sim.carla_common import CAM_ATTRS

        bp = M.make_depth_blueprint(_Lib())
        assert bp.name == "sensor.camera.depth"
        assert bp.attrs == dict(CAM_ATTRS)
        assert set(bp.attrs) == {"image_size_x", "image_size_y", "fov"}, "多了/少了属性都算改了口径"

    def test_depth_uses_the_same_mount_offset_as_rgb(self):
        """挂点也必须是**同一份常量**。源码级判据:spawn 那一行不许出现别的 transform。"""
        assert "world.spawn_actor(make_depth_blueprint(bp_lib), SENSOR_OFFSET, attach_to=ego)" in _SRC, (
            "深度相机必须与 RGB 用同一个 SENSOR_OFFSET;另写一个 transform 就失去像素对齐"
        )

    def test_every_queue_is_drained_each_tick(self):
        """★ 红线「每 tick 每队列都要抽干」的第 5 个可能现形点:深度队列。

        漏 `get` ⇒ 那一路**整体滞后 N 帧**;漏 `assert_synced` ⇒ 没人发现。
        而两者的症状都是"帧号一张张对得上、图一张张出得来" —— 这正是这条最毒的地方。
        """
        assert "dep_q.get(timeout=10)" in _SRC, "深度队列没有被抽干"
        assert 'assert_synced([("RGB", image), ("深度", dep_img)])' in _SRC, "两路没有同 tick 自证"

    def test_depth_is_written_next_to_image_2(self):
        """落点必须是 `training/depth/{fid:06d}.npy`,解码走 `depth_codec`(**唯一口径**)。"""
        assert 'out / "training" / "depth"' in _SRC
        assert "decode_depth(dep_img.raw_data" in _SRC, "解码必须走 calib.depth_codec,不许另写一份"

    def test_destroy_includes_the_optional_sensor(self):
        """收尾必须带上深度那路 —— 留着同名 sensor actor 会在下一轮"清场"里被漏掉,
        与"5 个 radar 忘收"是同一条(见主循环 `finally` 的头注)。"""
        assert "(camera, lidar, dep, *radars.values())" in _SRC

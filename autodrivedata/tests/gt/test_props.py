"""静态道具 GT 的**纯值**部分:类别映射、nuScenes 口径、JSON 往返。

不碰 CARLA —— "资产摆出来长什么样"由 `probe_static_prop_gt` 实测,"读回来的尺寸
可不可信"由 `perception/test_prop_eval.py` 拿渲染轮廓当裁判。这里只管表本身。

## 这几条为什么非钉不可

类别映射塌掉的样子全都**不是崩溃**,是**一个看着正常的类别名**:

| 塌法 | 症状 |
|---|---|
| 子串规则顺序错 | `warningconstruction` 被 "sign" 之类的规则抢走,变成另一类 |
| 认不出时返回 `None` | 调用方各自决定"这行怎么办",决定之间不一致 |
| 抄 nuScenes 类表抄错 | 锥桶记成"官方忽略类"被静默丢弃(这正是 2026-10-01 修掉的那个 bug) |
"""

from __future__ import annotations

import math

import pytest

from autodrivedata.gt import props
from autodrivedata.gt.core import actor_box_from_prop, classify_nus


class TestClassifyProp:
    @pytest.mark.parametrize(
        ("type_id", "want"),
        [
            ("static.prop.constructioncone", "cone"),
            ("static.prop.trafficcone01", "cone"),
            ("static.prop.trafficcone02", "cone"),
            ("static.prop.streetbarrier", "barrier"),
            ("static.prop.warningconstruction", "barrier"),  # 显式表优先于子串
        ],
    )
    def test_known_assets(self, type_id, want):
        """★ 前三项与后两项都是 **2026-10-01 实 spawn 过**的资产(探针量过尺寸与 tag)。"""
        assert props.classify_prop(type_id) == want

    def test_unknown_lands_on_an_explicit_bucket_not_none(self):
        """★ **认不出 ≠ 没有类**。返回 None 会让每个调用方自己发挥,而那些发挥会不一致。"""
        got = props.classify_prop("static.prop.someassetnobodyhasseen")
        assert got == "prop"
        assert got is not None

    def test_every_returned_label_is_declared(self):
        """返回值必须落在 `PROP_LABELS` 里 —— 新增类别忘了改那个元组,这里就红。"""
        samples = [
            "static.prop.constructioncone",
            "static.prop.streetbarrier",
            "static.prop.warningconstruction",
            "static.prop.barrel01",
            "static.prop.sign_whatever",
            "static.prop.mystery",
        ]
        for tid in samples:
            assert props.classify_prop(tid) in props.PROP_LABELS

    def test_substring_rules_are_reachable(self):
        """反向对照:显式表之外的名字也得能命中子串规则,否则规则是死代码。"""
        assert props.classify_prop("static.prop.some_cone_v3") == "cone"
        assert props.classify_prop("static.prop.trafficbarrel02") == "barrel"

    def test_is_prop_predicate(self):
        assert props.is_prop("static.prop.constructioncone")
        assert not props.is_prop("vehicle.audi.a2")
        assert not props.is_prop("walker.pedestrian.0001")
        assert not props.is_prop("traffic.traffic_light")


class TestNusClass:
    def test_cone_and_barrier_are_official_nuscenes_classes(self):
        """★ nuScenes 23 类里**确实有** `traffic_cone` 与 `barrier` —— 这条是官方类表口径。"""
        assert props.nus_class_of("cone") == "traffic_cone"
        assert props.nus_class_of("barrier") == "barrier"

    def test_the_rest_are_official_ignore_classes(self):
        for label in ("barrel", "sign", "prop"):
            assert props.nus_class_of(label) is None

    def test_unknown_label_raises_instead_of_returning_none(self):
        """★ 未知类别**抛异常**,不许静默返回 None —— 那会与"官方忽略类"混成同一个值。"""
        with pytest.raises(KeyError):
            props.nus_class_of("not_a_label")


class TestClassifyNusRegression:
    """★ 2026-10-01 修的那个 bug:非 vehicle 一律 `None`,把锥桶记成了"官方忽略类"。"""

    def test_cone_is_no_longer_silently_dropped(self):
        assert classify_nus("static.prop.constructioncone") == "traffic_cone"

    def test_barrier_too(self):
        assert classify_nus("static.prop.streetbarrier") == "barrier"

    def test_non_prop_non_vehicle_still_none(self):
        """反向对照:修的是道具,**不是**把忽略类的口子开大。"""
        assert classify_nus("traffic.traffic_light") is None
        assert classify_nus("traffic.stop") is None

    def test_vehicle_and_walker_paths_untouched(self):
        assert classify_nus("vehicle.audi.a2") == "car"
        assert classify_nus("walker.pedestrian.0001") == "pedestrian"
        assert classify_nus("vehicle.ford.ambulance") is None  # 官方忽略类


class TestPropFrameJson:
    @staticmethod
    def _frame() -> props.PropFrame:
        return props.PropFrame(
            frame_id="000003",
            ego_location=(1.5, -2.5, 0.6),
            ego_yaw_deg=0.16,
            props=(
                props.PropBox(
                    type_id="static.prop.constructioncone",
                    label="cone",
                    location=(10.0, 0.3, 0.55),
                    yaw_deg=90.0,
                    size=(0.3441, 0.3441, 0.5858),
                    box_offset=(0.0, 0.0, 0.2929),
                ),
            ),
        )

    def test_roundtrip_preserves_everything(self):
        f = self._frame()
        back = props.PropFrame.from_json(f.to_json())
        assert back == f

    def test_roundtrip_survives_a_rotated_prop(self):
        """★ 转过 yaw 的道具必须原样往返 —— 尺寸存的是 yaw=0 那份,朝向单独存
        (`gt/props.py` 头注:`bounding_box` 转过后读回的是被剪切的错值)。"""
        f = self._frame()
        back = props.PropFrame.from_json(f.to_json())
        assert back.props[0].yaw_deg == 90.0
        assert back.props[0].size == (0.3441, 0.3441, 0.5858)

    def test_label_counts(self):
        assert self._frame().label_counts() == {"cone": 1}

    def test_empty_frame_roundtrips(self):
        f = props.PropFrame(frame_id="000000", ego_location=(0.0, 0.0, 0.0), ego_yaw_deg=0.0)
        assert props.PropFrame.from_json(f.to_json()) == f

    def test_nus_class_property(self):
        assert self._frame().props[0].nus_class == "traffic_cone"


class TestActorBoxFromProp:
    """★ **三处口径的汇合点**,每一处错了都不抛异常 —— 只让投影框**默默不对**。

    ① 度/弧度(`PropBox` 度、`ActorBox` 弧度)② 全长/半长 ③ 盒偏移(局部系)vs actor 位姿(世界系)。
    """

    @staticmethod
    def _prop(**kw) -> props.PropBox:
        base = dict(
            type_id="static.prop.constructioncone",
            label="cone",
            location=(10.0, 2.0, 0.0),
            yaw_deg=0.0,
            size=(0.4, 0.4, 0.6),
            box_offset=(0.0, 0.0, 0.3),
        )
        return props.PropBox(**{**base, **kw})

    def test_extent_is_half_of_size(self):
        """★ 坑②:size 是**全长**,extent 是**半长**。差一倍,而投影出来只是"框大了一点"。"""
        b = actor_box_from_prop(self._prop())
        assert b.extent == (0.2, 0.2, 0.3)

    def test_yaw_is_converted_to_radians(self):
        """★ 坑①:度直接当弧度传进去时,六路里**只有 yaw≈0 的那路看着正常**
        (同投影链那条红线)。这里钉具体数值,不做视觉判断。"""
        b = actor_box_from_prop(self._prop(yaw_deg=90.0))
        assert b.actor_rotation[1] == pytest.approx(math.pi / 2)
        assert b.actor_rotation[0] == 0.0 and b.actor_rotation[2] == 0.0

    def test_box_rotation_is_converted_too(self):
        b = actor_box_from_prop(self._prop(box_rotation_deg=(0.0, 180.0, 0.0)))
        assert b.rotation[1] == pytest.approx(math.pi)

    def test_actor_location_is_passed_through_unchanged(self):
        """★ 坑③:actor 位姿是**世界系**原样,盒偏移是**局部系**原样 —— 谁也不许被"顺手"旋转。"""
        b = actor_box_from_prop(self._prop(location=(3.0, -4.0, 0.5)))
        assert b.actor_location == (3.0, -4.0, 0.5)
        assert b.location == (0.0, 0.0, 0.3)

    def test_negative_yaw_for_a_reversed_prop(self):
        b = actor_box_from_prop(self._prop(yaw_deg=270.0))
        assert b.actor_rotation[1] == pytest.approx(3 * math.pi / 2)


class TestInstanceIdAndCamera:
    def test_instance_id_defaults_to_explicit_unknown(self):
        """★ `-1` = **显式未知**(没落 id 图),不是"没有实例" —— 判据遇到它要跳过并报数,
        不许静默当成 0 像素(那会与"真的没渲染出来"混成同一个读数)。"""
        p = props.PropBox(type_id="t", label="cone", location=(0, 0, 0), yaw_deg=0.0, size=(1, 1, 1))
        assert p.instance_id == -1

    def test_instance_id_roundtrips(self):
        f = props.PropFrame(
            frame_id="000001",
            ego_location=(0.0, 0.0, 0.0),
            ego_yaw_deg=0.0,
            props=(
                props.PropBox(
                    type_id="t",
                    label="cone",
                    location=(0, 0, 0),
                    yaw_deg=0.0,
                    size=(1, 1, 1),
                    instance_id=65535,
                ),
            ),
        )
        assert props.PropFrame.from_json(f.to_json()).props[0].instance_id == 65535

    def test_camera_roundtrips_including_none(self):
        base = dict(frame_id="0", ego_location=(0.0, 0.0, 0.0), ego_yaw_deg=0.0)
        assert props.PropFrame(**base).to_json()  # camera=None 也能序列化
        assert props.PropFrame.from_json(props.PropFrame(**base).to_json()).camera is None
        cam = props.CameraPose(
            location=(1.0, 2.0, 1.6), rotation_deg=(0.1, 0.2, -0.0), width=1242, height=375, fov_deg=90.0
        )
        f = props.PropFrame(**base, camera=cam)
        assert props.PropFrame.from_json(f.to_json()).camera == cam

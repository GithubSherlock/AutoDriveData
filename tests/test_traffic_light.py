"""traffic_light.py 纯值单测(不依赖 carla)。"""

from __future__ import annotations

from dataclasses import replace

from autodrivedata import traffic_light as tl

PLAN = (("Green", 6.0), ("Yellow", 2.0), ("Red", 6.0))


class TestNormalizeState:
    def test_strips_carla_enum_prefix(self):
        assert tl.normalize_state("TrafficLightState.Red") == "Red"
        assert tl.normalize_state("TrafficLightState.Yellow") == "Yellow"
        assert tl.normalize_state("TrafficLightState.Green") == "Green"

    def test_bare_name_passthrough(self):
        assert tl.normalize_state("Off") == "Off"

    def test_unknown_values_fall_back_not_guessed(self):
        # 枚举外的值不许映射成三色(工业口径:宁 Unknown 不猜)
        assert tl.normalize_state("TrafficLightState.Blinking") == "Unknown"
        assert tl.normalize_state("") == "Unknown"
        assert tl.normalize_state("TrafficLightState.Red.Yellow") == "Yellow"


class TestInFront:
    EGO = (0.0, 0.0, 0.0)

    def test_along_heading(self):
        # yaw=0 朝 +x
        assert tl.in_front((10.0, 0.0, 0.0), self.EGO, 0.0)
        assert not tl.in_front((-10.0, 0.0, 0.0), self.EGO, 0.0)

    def test_rotated_heading(self):
        # yaw=90 朝 +y(用明确的后方点:cos(radians(90))=6.1e-17 不是 0,
        # 正侧方在旋转朝向下是浮点刀口,判据本身不为此加 epsilon)
        assert tl.in_front((0.0, 10.0, 0.0), self.EGO, 90.0)
        assert not tl.in_front((0.0, -10.0, 0.0), self.EGO, 90.0)

    def test_perpendicular_is_not_front(self):
        # 正侧方(半平面边界)归"不在前方":身后的灯实测占 79%,宁可漏收不滥收
        assert not tl.in_front((0.0, 5.0, 0.0), self.EGO, 0.0)

    def test_relative_to_ego_position(self):
        assert tl.in_front((110.0, 5.0, 0.0), (100.0, 5.0, 0.0), 0.0)
        assert not tl.in_front((90.0, 5.0, 0.0), (100.0, 5.0, 0.0), 0.0)

    def test_z_ignored(self):
        # 灯头比 ego 高 4.5m 是常态,高度不参与前后判据
        assert tl.in_front((10.0, 0.0, 4.5), self.EGO, 0.0)


class TestPhaseAt:
    def test_phase_boundaries(self):
        # 周期 6+2+6=14s;边界取左侧相位(Green 6s → Yellow 2s → Red 6s)
        assert tl.phase_at(0.0, PLAN) == "Green"
        assert tl.phase_at(5.9, PLAN) == "Green"
        assert tl.phase_at(6.0, PLAN) == "Yellow"
        assert tl.phase_at(7.9, PLAN) == "Yellow"
        assert tl.phase_at(8.0, PLAN) == "Red"
        assert tl.phase_at(13.9, PLAN) == "Red"

    def test_wraps_around(self):
        assert tl.phase_at(14.0, PLAN) == "Green"
        assert tl.phase_at(20.0, PLAN) == "Yellow"

    def test_empty_plan_is_unknown(self):
        assert tl.phase_at(3.0, ()) == "Unknown"


class TestFrameJson:
    def _frame(self) -> tl.TrafficLightFrame:
        return tl.TrafficLightFrame(
            frame_id="000003",
            map_name="Carla/Maps/Town10HD_Opt",
            ego_location=(-64.6, 24.5, 0.3),
            ego_yaw_deg=0.2,
            lights=(
                tl.TrafficLightState(
                    opendrive_id="949",
                    state="Red",
                    location=(-31.6, 33.6, 4.8),
                    yaw_deg=180.0,
                    pole_index=3,
                    elapsed_s=1.4,
                    distance_m=34.2,
                    affected_lanes=((315, -1), (255, -2)),
                    stop_lanes=((19, -1, 8.0),),
                ),
            ),
            phase_plan=PLAN,
        )

    def test_round_trip(self):
        f = self._frame()
        back = tl.TrafficLightFrame.from_json(f.to_json())
        assert back == f  # 含嵌套 tuple 与 phase_plan 的完整往返

    def test_record_mode_has_empty_plan(self):
        f = tl.TrafficLightFrame(
            frame_id="000000", map_name="m", ego_location=(0.0, 0.0, 0.0), ego_yaw_deg=0.0
        )
        assert f.phase_plan == ()
        assert tl.TrafficLightFrame.from_json(f.to_json()).phase_plan == ()

    def test_unknown_state_survives_serialization(self):
        # 不猜原则要落进 json:Unknown/Off 原样保留(不被三色归一)
        f = self._frame()
        text = replace(f, lights=(replace(f.lights[0], state="Unknown"),)).to_json()
        assert '"state": "Unknown"' in text
        assert tl.TrafficLightFrame.from_json(text).lights[0].state == "Unknown"

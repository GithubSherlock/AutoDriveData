"""scenarios.py 场景目录与可模拟性矩阵的静态校验(P1)。

纯逻辑单测:场景档的键必须落在实测的 WeatherParameters 字段内
(防打错字静默失效——天气覆写失败会直接产出"假晴天"基线数据)。
"""

from __future__ import annotations

import pytest

from autodrivedata.sim.scenarios import (
    BASE_WEATHER,
    SCENES,
    WEATHER_KEYS,
    merged_weather,
    validate_scene,
)

WEATHER_SCENES = [
    "day_clear",
    "sunset_glare",
    "night_clear",
    "heavy_rain",
    "rain_night",
    "dense_fog",
    "wet_road",
]


class TestCatalog:
    def test_base_weather_is_full(self):
        # BASE_WEATHER 必须是 pycarla WeatherParameters 全字段(实测 14 个数值字段)
        assert set(BASE_WEATHER) == WEATHER_KEYS
        assert len(WEATHER_KEYS) == 14

    def test_all_scenes_valid(self):
        for scene in SCENES.values():
            validate_scene(scene)

    def test_merged_inherits_full_base(self):
        """merge 后每场景键集合 = 基线全集(继承不能丢字段)。"""
        for scene in SCENES.values():
            m = merged_weather(scene)
            assert set(m) == WEATHER_KEYS
            for k, v in scene.weather.items():
                assert m[k] == v, f"{scene.name}.{k}"

    def test_weather_override_lands(self):
        assert merged_weather(SCENES["sunset_glare"])["sun_altitude_angle"] == 6.0
        assert merged_weather(SCENES["dense_fog"])["fog_density"] == 100.0
        assert merged_weather(SCENES["day_clear"]) == BASE_WEATHER  # 基线场景零覆写

    def test_typo_key_rejected(self):
        """打错天气键必须抛错,不许静默出"假晴天"。"""
        from autodrivedata.sim.scenarios import Scene

        bad = Scene(name="bad", group="weather", weather={"sun_altidude": 1.0})
        with pytest.raises(KeyError):
            merged_weather(bad)

    def test_bad_traffic_key_rejected(self):
        from autodrivedata.sim.scenarios import Scene

        bad = Scene(name="bad", group="traffic", traffic={"npc_car": 3})
        with pytest.raises(KeyError):
            validate_scene(bad)

    def test_lighting_scenes_present(self):
        for name in WEATHER_SCENES:
            assert name in SCENES

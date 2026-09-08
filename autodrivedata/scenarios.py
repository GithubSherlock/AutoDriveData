"""Corner case 场景目录 + 可模拟性矩阵(P1,2026-09-08)。

纯数值定义、**不 import carla**(autodrivedata 包纪律:两 env 可单测)。
采集脚本运行时映射:`carla.WeatherParameters(**weather)` 与相机蓝图属性。

场景 = weather(天气/光照)× traffic(车流)两维度;SCENES 每项可覆写任一维,
未覆写字段继承 BASE_WEATHER(= 生产当前配置的 ClearNoon 实测值)。

可模拟性矩阵(对每个场景如实标注保真边界,防把简化渲染当真实):
  可模拟: 逆光(低角度正前太阳——无镜头光学,只出"高反差+天空过曝+目标剪影")、
           黄昏/黎明、夜(路灯车灯起效)、雨、雨夜、浓雾、湿/积水反光、
           密集车流;LiDAR 退化只能**人工注入**(丢点/噪声开关,非物理模拟)
  不可模拟: 雪/冰(0.9.16 无雪粒子)、lens flare/真实镜头光学(逆光"眼瞎"只量化
           相机 AP 掉点 + LiDAR 兜底差值,不做视觉真实感)、LiDAR 雨/尘物理回波
  待验证:   鬼探头(walker NAV 网格约束——M3 原点毒化教训,需遮挡+路径脚本,排后续)

实验纪律(2026-09-07 教训):行人只放有效 spawn 点、不导航;GT max_distance=65;
对比实验同 seed/同路线/同车流,只变 weather —— 见 P1-3 逆光 A/B。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 生产当前配置 = ClearNoon 实测值(2026-09-08 pycarla 直查),作为一切场景的基底
BASE_WEATHER: dict[str, float] = {
    "cloudiness": 5.0,
    "precipitation": 0.0,
    "precipitation_deposits": 0.0,
    "wind_intensity": 10.0,
    "sun_azimuth_angle": -1.0,
    "sun_altitude_angle": 45.0,
    "fog_density": 2.0,
    "fog_distance": 0.75,
    "fog_falloff": 0.1,
    "wetness": 0.0,
    "scattering_intensity": 1.0,
    "mie_scattering_scale": 0.03,
    "rayleigh_scattering_scale": 0.0331,
    "dust_storm": 0.0,
}

# pycarla WeatherParameters 全部可设字段(校验 weather override 键用)
WEATHER_KEYS: frozenset[str] = frozenset(BASE_WEATHER)

_TRAFFIC_KEYS = ("npc_vehicles", "npc_walkers", "route_walkers")


@dataclass(frozen=True)
class Scene:
    name: str
    group: str  # lighting | weather | traffic | combo
    weather: dict[str, float] = field(default_factory=dict)  # override,键 ⊂ WEATHER_KEYS
    traffic: dict[str, int] = field(default_factory=dict)  # override,键 ⊂ _TRAFFIC_KEYS
    exposure: dict[str, str] = field(default_factory=dict)  # 相机蓝图属性覆写(曝光)
    fidelity: str = ""  # 保真度评注:能模拟什么、边界在哪
    sensor_note: str = ""  # 预期传感器影响(哪类"瞎"、哪类不受影响)


def merged_weather(scene: Scene) -> dict[str, float]:
    """BASE_WEATHER + scene.weather override(键校验,防打错字静默失效)。"""
    bad = set(scene.weather) - WEATHER_KEYS
    if bad:
        raise KeyError(f"{scene.name}: 非法 weather 键 {sorted(bad)}")
    return {**BASE_WEATHER, **scene.weather}


SCENES: dict[str, Scene] = {
    # ---- lighting ----
    "day_clear": Scene(
        name="day_clear",
        group="lighting",
        fidelity="基线 = 生产当前配置(ClearNoon);A/B 对照的晴天侧",
        sensor_note="无退化,基准",
    ),
    "sunset_glare": Scene(
        name="sunset_glare",
        group="lighting",
        weather={"sun_altitude_angle": 6.0, "cloudiness": 0.0, "fog_density": 5.0},
        fidelity=(
            "逆光实测(2026-09-08,数值诊断):日盘真实渲染但**AE 压至 ~205 不饱和**"
            "(天空 p99 205 vs 背阳 170);天空均值与太阳方位弱相关(散射天光主导,"
            "AE 全局曝光)。azimuth→世界方向(ego yaw=0/朝+x 时):日盘峰 az≈290±40。"
            "CARLA 无 flare/镜头光学——'眼瞎'不可用天空曝光目检证明,必须 P1-3 "
            "A/B 用模型 AP 量化(相机 AP 掉点 + LiDAR 兜底差值)"
        ),
        sensor_note="相机:逆光高反差(待 A/B 定量);LiDAR 不受光照影响(物理正确,融合兜底侧)",
    ),
    "night_clear": Scene(
        name="night_clear",
        group="lighting",
        weather={"sun_altitude_angle": -35.0, "cloudiness": 10.0},
        fidelity="夜间:路灯/车前灯 0.9.16 起效;整体亮度低、无月光补助",
        sensor_note="相机:低光噪点少(渲染理想化)+ 动态范围压缩;LiDAR 不受影响",
    ),
    # ---- weather ----
    "heavy_rain": Scene(
        name="heavy_rain",
        group="weather",
        weather={
            "cloudiness": 100.0,
            "precipitation": 100.0,
            "wind_intensity": 50.0,
            "sun_altitude_angle": 30.0,
        },
        fidelity="暴雨:雨丝/水花渲染可见,积水自动出现(wetness 随降水)",
        sensor_note="相机:对比下降;LiDAR:CARLA 不模拟雨回波(与真车差距)——如要雨损需人工注入",
    ),
    "rain_night": Scene(
        name="rain_night",
        group="combo",
        weather={
            "cloudiness": 100.0,
            "precipitation": 80.0,
            "sun_altitude_angle": -25.0,
            "fog_density": 20.0,
        },
        fidelity="雨夜:低照度 + 湿路面车灯/路灯反光(真实驾驶最难的组合之一)",
        sensor_note="相机:暗 + 反光光斑;LiDAR:无雨退化(平台边界)",
    ),
    "dense_fog": Scene(
        name="dense_fog",
        group="weather",
        weather={
            "fog_density": 100.0,
            "fog_distance": 0.05,
            "fog_falloff": 0.3,
            "cloudiness": 60.0,
        },
        fidelity="浓雾:fog_distance=0.05km ≈ 50m 可见度(fog_density 0-100 拉满);实测(数值):中带高频梯度 -31%(day_clear 19.2→13.3),细节掩蔽生效",
        sensor_note="相机:远处目标被雾掩;LiDAR:雾模拟有限,不可做真物理衰减",
    ),
    "wet_road": Scene(
        name="wet_road",
        group="weather",
        weather={
            "cloudiness": 60.0,
            "precipitation_deposits": 100.0,  # 雨后积水保持,不降水
            "sun_altitude_angle": 15.0,
        },
        fidelity="雨后湿路面(积水反光 + 低角度阳光):反光/眩光最重的光照×路面组合",
        sensor_note="相机:路面镜面反光可能产生伪影区域",
    ),
    # ---- traffic ----
    "dense_rush": Scene(
        name="dense_rush",
        group="traffic",
        traffic={"npc_vehicles": 40, "npc_walkers": 12},
        fidelity="密集拥堵车流(40 车 + 12 行人);TM 同步模式下交互风险高,速度差调小防追尾",
        sensor_note="相机/LiDAR:遮挡多——FP 与漏检都预期上升,测遮挡鲁棒性",
    ),
}


def validate_scene(scene: Scene) -> None:
    merged_weather(scene)  # weather 键校验
    bad = set(scene.traffic) - set(_TRAFFIC_KEYS)
    if bad:
        raise KeyError(f"{scene.name}: 非法 traffic 键 {sorted(bad)}")


def list_scenes() -> str:
    lines = []
    for s in SCENES.values():
        validate_scene(s)
        lines.append(f"- {s.name} [{s.group}] 天气覆写 {sorted(s.weather)} 交通覆写 {s.traffic or '无'}")
    return "\n".join(lines)

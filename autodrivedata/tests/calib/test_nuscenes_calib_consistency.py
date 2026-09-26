"""`collect_nus.py` 全传感器标定一致性钉:**「渲染位姿」与「声明位姿」必须同源**。

**为什么要有这个文件(2026-09-23,Plan2.md §P-M.7)**:`collect_nus.py` 曾出现**表对了、图错了**
的新失效模式——写进 `calibrated_sensor` 的是官方正确值,而传感器实际 spawn 在另一套位姿上:

| 项 | 历史(错) | 现在(对) |
|---|---|---|
| 6 相机 spawn | 一律挂 LiDAR 挂点 + 旧镜像偏航表 `{0,±55,180,±125}` | `NUS_CAMERA_RIG` 逐相机 6DoF |
| 5 雷达 spawn 偏航 | `{0,±45,±90}`(与同名相机同号的**猜测**) | `−az_nus`(四路角雷达差 94–136°) |
| LiDAR spawn | **无 rotation**、挂点 `(1.2,0,1.65)` | 官方挂点 + 1.4289° up 轴倾角 |
| 相机蓝图 fov | 六路共用 `90°` | 逐通道 `64.31–64.96°`(CAM_BACK `89.34°`) |

这类缺陷**查表拦不住**(表是对的)、**目检"图能出"也拦不住**(图确实出得来)。故本文件只钉
"两侧由同一份常量导出"这件事本身——**断言的是等价关系,不是某次采样的数值**。

约定:`pytest.importorskip("carla")`。autodrivedata env 装有 carla ⇒ 实际**不会 skip**,不是占位。
(阶段 3 起本文件在包内,已摘掉旧的 `sys.path.insert(0, BIN)` 引导,改直连 `autodrivedata.calib.*`。)
"""

from __future__ import annotations

import ast
import math

import pytest

pytest.importorskip("carla")

# 进包后不再需要 sys.path 引导(旧 bin/ 非包布局的产物);ROOT 改走 paths.PROJECT_ROOT ——
# 不再用 `Path(__file__).parents[N]`,那条路径会随本文件在包内挪动而**静默指错**
# (阶段 3 本文件从 tests/ 挪到 autodrivedata/tests/calib/,parents[1] 就从仓库根变成了 tests/)。
from autodrivedata import geometry as g
from autodrivedata import paths
from autodrivedata.calib import verify_nus_calib as vnc
from autodrivedata.calib.camera_rig import (
    NUS_CAMERA_CALIBS,
    NUS_CAMERA_RIG,
    NUS_WIDE_CAMERA_AZ,
    NUS_WIDE_CAMERA_CALIBS,
    NUS_WIDE_CAMERA_FOV,
    NUS_WIDE_CAMERA_RIG,
    NUS_WIDE_CAMERA_YAW,
    NUS_WIDE_REAR_X_CARLA,
    max_hfov_no_ego,
)
from autodrivedata.export import nuscenes as ne
from autodrivedata.sim import (
    collect_nus,
    collect_surround,
)
from autodrivedata.sim import live_common as lc

ROOT = paths.PROJECT_ROOT
CALIB = ROOT / "autodrivedata" / "calib"  # 阶段 3 起标定层在这里(源码检查按路径读)
SIM = ROOT / "autodrivedata" / "sim"  # 阶段 2 起采集器在这里


def _call_name(node: ast.AST) -> str:
    """`ast.Call` → 被调名的点分字符串(如 `world.tick`);不是 Call / 名字取不到 → 空串。"""
    if not isinstance(node, ast.Call):
        return ""
    parts: list[str] = []
    cur: ast.AST = node.func
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


# 官方 n015 逐通道 fx(1600×900)。**独立硬编码**——从 `ne.NUS_CAMERA_INTRINSICS` 取会让
# 下面两条断言恒真,拦不住"表被改错"。
OFFICIAL_FX = {
    "CAM_FRONT": 1266.417203046554,
    "CAM_FRONT_LEFT": 1272.5979470598488,
    "CAM_FRONT_RIGHT": 1260.8474446004698,
    "CAM_BACK": 809.2209905677063,
    "CAM_BACK_LEFT": 1256.7414812095406,
    "CAM_BACK_RIGHT": 1259.5137405846733,
}

# 历史镜像表(§P-M.7.2/.3 的实测缺陷值)——用于反回归,见 TestMirrorTablesGone
HISTORICAL_CAM_YAW = {
    "CAM_FRONT": 0.0,
    "CAM_FRONT_LEFT": 55.0,
    "CAM_FRONT_RIGHT": -55.0,
    "CAM_BACK": 180.0,
    "CAM_BACK_LEFT": 108.6,
    "CAM_BACK_RIGHT": -110.8,
}
HISTORICAL_RADAR_YAW = {
    "RADAR_FRONT": 0.0,
    "RADAR_FRONT_LEFT": 45.0,
    "RADAR_FRONT_RIGHT": -45.0,
    "RADAR_BACK_LEFT": 90.0,
    "RADAR_BACK_RIGHT": -90.0,
}


class TestMirrorTablesGone:
    """① 镜像表**不再存在**(或已被单点来源取代)——反回归的第一道闸。

    `CAM_YAW_OFFSET` 整张删除;`RADAR_YAW_OFFSET` 保留同名但改为**导出**。名字还在不代表
    值还对,故两条分开断言:前者查"属性不存在",后者查"逐通道 == 导出值"。
    """

    def test_cam_yaw_offset_deleted(self):
        assert not hasattr(collect_nus, "CAM_YAW_OFFSET"), (
            "CAM_YAW_OFFSET 已删除(相机偏航由 NUS_CAMERA_RIG 单点提供);"
            "重新引入 = 回到「相机挂 LiDAR 挂点 + 镜像偏航」的失效模式"
        )

    def test_cam_attrs_split(self):
        """`CAM_ATTRS`(六路共用含 fov=90)拆成 COMMON + 逐通道 FOV。"""
        assert not hasattr(collect_nus, "CAM_ATTRS"), "CAM_ATTRS 已拆成 CAM_ATTRS_COMMON + CAM_FOV"
        assert "fov" not in collect_nus.CAM_ATTRS_COMMON, "fov 必须逐通道,不能留在共用属性里"
        assert collect_nus.CAM_ATTRS_COMMON["image_size_x"] == "1600"
        assert collect_nus.CAM_ATTRS_COMMON["image_size_y"] == "900"

    def test_radar_yaw_is_not_the_historical_guess(self):
        """四路角雷达不再等于历史猜测值(±45/±90);偏差 ≥ 90°。"""
        for ch, old in HISTORICAL_RADAR_YAW.items():
            new = collect_nus.RADAR_YAW_OFFSET[ch]
            raw = abs((old - new + 180.0) % 360.0 - 180.0)
            if ch == "RADAR_FRONT":  # 前雷达近自逆 ⇒ 历史值恰好接近正确值(它长期没暴露的原因)
                assert raw < 0.3
            else:
                assert raw > 90.0, f"{ch} 与历史猜测值仅差 {raw:.2f}°,疑似回退"


class TestSpawnPosesSingleSourced:
    """② spawn 位姿由官方常量**导出**,不存在第二份手抄。"""

    def test_camera_spawn_uses_nus_camera_rig(self):
        """`NUS_CAMERA_RIG` 就是 spawn 用的表(与写进 calibrated_sensor 的同源推导)。"""
        assert collect_nus.NUS_CAMERA_RIG is NUS_CAMERA_RIG
        # 与落盘表逐通道一致:camera_rig 里 rig 与 calibs 是同一推导的两套表达
        for cam, (mount, _) in NUS_CAMERA_RIG.items():
            t_nus = ne.NUS_CAMERA_CALIBS[cam][0]
            # ★ x **不是**照抄:**两套系的原点不同**(nus 原点=后轴、CARLA actor 原点=车身中点)
            assert mount[0] == pytest.approx(t_nus[0] + g.NUS_EGO_ORIGIN_X, abs=1e-12), cam
            assert mount[1] == pytest.approx(-t_nus[1], abs=1e-12), cam  # CARLA y 右 = nus y 左取负
            assert mount[2] == pytest.approx(t_nus[2], abs=1e-12), cam
            assert mount[0] != pytest.approx(t_nus[0], abs=1e-3), f"{cam} 的 x 少了原点平移"

    def test_camera_yaw_equals_minus_official_azimuth(self):
        """六个相机 spawn 偏航 == `−az_nus`(1e-9 度内)——镜像 bug 的判据本体。

        四个侧/后相机另查"与历史镜像值差 > 90°"(前/后相机近自逆,历史值恰好接近正确值,
        对它们无信息量 —— 这正是镜像 bug 长期没暴露的原因,见 `camera_rig` 模块头注)。
        """
        for cam, (_, rot) in NUS_CAMERA_RIG.items():
            r_nus = g.quat_to_matrix(g.quat_normalize(ne.NUS_CAMERA_CALIBS[cam][1]))
            boresight = r_nus[:, 2]  # 相机自身系 x右/y下/z前 ⇒ 视线轴 = +z 列
            az_nus = math.degrees(math.atan2(boresight[1], boresight[0]))
            assert rot[1] == pytest.approx(-az_nus, abs=1e-9), cam
            if cam not in ("CAM_FRONT", "CAM_BACK"):
                raw = abs((HISTORICAL_CAM_YAW[cam] - rot[1] + 180.0) % 360.0 - 180.0)
                assert raw > 90.0, f"{cam} 与历史镜像值仅差 {raw:.2f}°,疑似回退"

    def test_radar_spawn_yaw_derived_from_official_table(self):
        """`RADAR_YAW_OFFSET` == `−degrees(NUS_RADAR_OFFSETS[ch].yaw)`(逐通道,1e-12)。"""
        for ch in ne.NUS_RADAR_CHANNELS:
            expect = -math.degrees(ne.NUS_RADAR_OFFSETS[ch][1])
            assert collect_nus.RADAR_YAW_OFFSET[ch] == pytest.approx(expect, abs=1e-12), ch
        assert set(collect_nus.RADAR_YAW_OFFSET) == set(ne.NUS_RADAR_CHANNELS)

    def test_radar_spawn_translation_mirrors_official(self):
        """雷达 spawn 平移 = 官方 nus 原值**翻 y 并减后轴偏移**(与 `nus_mount_to_carla` 同口径)。

        官方逐通道值(n015,米):`(3.412, 0, 0.5) / (2.422, ±0.8, 0.78|0.77) /
        (−0.562, ±0.62, 0.53)` —— 表里是**后轴原点**系,采集侧 spawn 走
        `geometry.nus_mount_to_carla`(x 平移 + y 取负)。
        """
        official = {
            "RADAR_FRONT": (3.412, 0.0, 0.5),
            "RADAR_FRONT_LEFT": (2.422, 0.8, 0.78),
            "RADAR_FRONT_RIGHT": (2.422, -0.8, 0.77),
            "RADAR_BACK_LEFT": (-0.562, 0.628, 0.53),
            "RADAR_BACK_RIGHT": (-0.562, -0.618, 0.53),
        }
        for ch, t_official in official.items():
            assert ne.NUS_RADAR_OFFSETS[ch][0] == pytest.approx(t_official, abs=1e-12), ch
            assert collect_nus.NUS_RADAR_MOUNTS_CARLA[ch] == pytest.approx(
                g.nus_mount_to_carla(t_official), abs=1e-12
            ), ch
            assert collect_nus.NUS_RADAR_MOUNTS_CARLA[ch][0] == pytest.approx(
                t_official[0] + g.NUS_EGO_ORIGIN_X, abs=1e-12
            ), ch


class TestLidarMountAndRotation:
    """③ LiDAR 挂点/姿态(历史:挂点 `(1.2,0,1.65)`、**无 rotation**)。"""

    def test_mount_constant(self):
        """spawn 挂点 = 官方 nus 值**减去后轴到车身中点的距离**(不是照抄)。

        官方 `(0.943713, 0, 1.84023)` 是**后轴原点系**的;CARLA actor 原点在中点 ⇒ 渲染落点
        `x = 0.943713 − 1.2563 = −0.312587`(LiDAR 从车顶前部退到车顶中部)。照抄会让整套
        传感器偏前 1.2563 m —— 这正是 §P-M.10 的缺陷本体。
        """
        official = (0.943713, 0.0, 1.84023)
        assert ne.NUS_LIDAR_CALIB[0] == pytest.approx(official, abs=1e-12)  # 声明侧仍是官方原值
        assert collect_nus.LIDAR_MOUNT == pytest.approx(
            (official[0] + g.NUS_EGO_ORIGIN_X, -official[1], official[2]), abs=1e-12
        )
        assert collect_nus.LIDAR_MOUNT == ne.NUS_LIDAR_MOUNT_CARLA
        assert collect_nus.LIDAR_MOUNT[0] == pytest.approx(-0.312587, abs=1e-6)
        assert collect_nus.LIDAR_MOUNT[0] != pytest.approx(official[0], abs=1e-3), "少了原点平移"

    def test_rotation_is_derived_not_hand_copied(self):
        """`LIDAR_ROT` == 由 `NUS_LIDAR_CALIB` 四元数导出的 CARLA (pitch, yaw, roll),1e-12。

        **不写成字面量**的理由:字面量会与四元数构成同一物理量的**两份手抄**——正是本次要
        根除的失效模式。实测导出值 ≈ `(−0.338027, +89.8835, −1.3884)`,与 §P-M.7.5 一致。
        """
        r_carla = g.nus_sensor_rotation_to_carla(ne.NUS_LIDAR_CALIB[1])
        p, y, r = g.rotation_matrix_to_carla(r_carla)
        expect = (math.degrees(p), math.degrees(y), math.degrees(r))
        assert collect_nus.LIDAR_ROT == pytest.approx(expect, abs=1e-12)
        assert collect_nus.LIDAR_ROT == pytest.approx((-0.3380, 89.8835, -1.3884), abs=1e-3)
        assert collect_nus.LIDAR_ROT[1] != pytest.approx(0.0, abs=1.0)  # 不是"无 rotation"

    def test_lidar_mount_differs_from_kitti_offset(self):
        """确认没退回 `carla_common.SENSOR_OFFSET = (1.2, 0, 1.65)`(KITTI 线口径,勿动)。"""
        from autodrivedata.sim.carla_common import SENSOR_OFFSET  # noqa: PLC0415

        kitti = (SENSOR_OFFSET.location.x, SENSOR_OFFSET.location.y, SENSOR_OFFSET.location.z)
        assert collect_nus.LIDAR_MOUNT != pytest.approx(kitti, abs=1e-6)
        assert SENSOR_OFFSET.location.z == pytest.approx(1.65, abs=1e-6)  # KITTI 线未被改动


class TestCameraFovMatchesIntrinsics:
    """④ 蓝图 fov 与落盘 K **由同一个 fx 导出**——"标定说 64°、图像是 90°"的锁。

    fx 表**在本文件独立硬编码**(`OFFICIAL_FX`),故断言非恒真。
    """

    def test_per_channel_fov(self):
        for cam, fx in OFFICIAL_FX.items():
            expect = math.degrees(2.0 * math.atan((1600 / 2.0) / fx))
            assert collect_nus.CAM_FOV[cam] == pytest.approx(expect, abs=1e-9), cam

    def test_fov_is_not_shared_90(self):
        """历史缺陷:六路共用 90°。至少五路必须显著窄于 90°。"""
        narrow = [c for c in OFFICIAL_FX if collect_nus.CAM_FOV[c] < 70.0]
        assert len(narrow) == 5
        assert collect_nus.CAM_FOV["CAM_BACK"] > 89.0  # 唯一宽视场通道(官方 89.34°)
        assert collect_nus.CAM_FOV["CAM_BACK"] != pytest.approx(90.0, abs=0.5)

    def test_fov_dict_is_the_export_table(self):
        """采集侧的 fov 就是导出侧的 `NUS_CAMERA_FOV`(不另抄一份)。"""
        assert collect_nus.CAM_FOV == ne.NUS_CAMERA_FOV


class TestFovCriterionShape:
    """判据 ⑥ 的**几何/取帧口径**钉(2026-09-23 两个实测坑的回归锁,纯静态、不连 CARLA)。

    `bin/verify_nus_calib.py` 模块级**不 import carla**(只在 `run_live` 里惰性 import),
    故 `import verify_nus_calib` 在哪个 env 都成立(本文件自身仍走模块头的 `importorskip`,
    与同目录其它采集器测试同款)。钉的是两个**只看代码看不出来、跑起来才暴露**的坑:

    1. **取帧必须"每 tick 抽干全部相机队列"**。六相机同时 listen 时每 tick 每队列各进一帧;
       只取被测相机那一帧 ⇒ 其余五路积压 ⇒ 轮到它们时读到的是**锥体 spawn 之前**的陈旧帧,
       掩膜恒 0。症状极具迷惑性:六相机里**只有第一个能测出数**,其余全 0 且**与距离无关**,
       看着像"摆不进去"。锁法:`world.tick()` 在 `_rendered_fov` 里**只能出现在 `drain` 内**。
    2. **Z 不能写死**。地图遮挡随 ego 出生点而变(实测同一相机 Z=20 可用 / Z=14 不可用都出现过)
       ⇒ 必须阶梯兜底,否则换个 spawn point 判据就假失败。
    """

    def _fov_fn(self) -> ast.FunctionDef:
        src = (CALIB / "verify_nus_calib.py").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name == "_rendered_fov":
                return node
        raise AssertionError("bin/verify_nus_calib.py 里找不到 `_rendered_fov`")

    def test_tick_only_inside_drain(self):
        fn = self._fov_fn()
        drains = [n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name == "drain"]
        assert len(drains) == 1, "`_rendered_fov` 里应当只有一个取帧助手 `drain`"
        ticks = [n for n in ast.walk(fn) if _call_name(n) == "world.tick"]
        assert len(ticks) == 1, (
            f"`_rendered_fov` 里有 {len(ticks)} 处 `world.tick()`;必须只有 `drain` 内一处 —— "
            "每 tick 抽干全部相机队列是硬要求(否则未测相机积压陈旧帧 ⇒ 掩膜恒 0)"
        )
        assert any(n is ticks[0] for n in ast.walk(drains[0])), "`world.tick()` 必须落在 `drain` 内部"

    def test_frames_read_through_the_one_drain(self):
        """队列读取只出现在 `drain` 内(别处再读一次就又回到"取帧不同步"的坑)。

        只数 `q.get(...)` 这一种:函数里还有 `dict.get("pass")` 之类**同名但无关**的调用,
        按 `.get` 通配会误报。
        """
        fn = self._fov_fn()
        qgets = [n for n in ast.walk(fn) if _call_name(n) == "q.get"]
        assert len(qgets) == 1, f"`_rendered_fov` 里有 {len(qgets)} 处 `q.get`;应只有 `drain` 内一处"
        drains = [n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name == "drain"]
        assert any(n is qgets[0] for n in ast.walk(drains[0])), "`q.get` 必须落在 `drain` 内部"

    def test_z_ladder_is_a_ladder(self):
        zs = vnc.FOV_Z_LADDER
        assert len(zs) >= 2, "Z 阶梯至少两档,否则退化成写死"
        assert all(zs[i] > zs[i + 1] for i in range(len(zs) - 1)), "阶梯必须**从远到近**(远处杠杆臂长)"
        assert all(z > 0.0 for z in zs)
        assert vnc.MIN_FOV_SAMPLES >= 3, "样本数 < 3 时回归退化,判据无意义"

    def test_lateral_fracs_span_the_frame_symmetrically(self):
        """横移按 `frac·Z` 取:必须关于 0 对称、且不越出半幅(0.5)。"""
        fr = vnc.FOV_LATERAL_FRACS
        assert len(fr) >= 5
        assert all(abs(f) < 0.5 for f in fr), "|frac| 必须 < 0.5(否则锥心出画幅)"
        assert max(abs(f) for f in fr) > 0.4, "杠杆臂要顶到接近半幅,否则 fx 精度不够(见模块 docstring)"
        for f in fr:
            assert any(abs(-f - g) < 1e-12 for g in fr), f"横移表不对称:{f}"


# ── wide rig(2026-09-23,Plan2 §P-M.8)——自定义宽视口口径 ────────────────────
#
# 设计目标值**独立硬编码**(从 `camera_rig` 取会让断言恒真)。用户口径 = 前三个 55°、
# 后侧 110°、后 120°(180° 被针孔退化否掉)、后三路挂点后移到车尾且**零车体像素**。
WIDE_AZ = {"CAM_BACK_LEFT": 145.0, "CAM_BACK": 180.0, "CAM_BACK_RIGHT": -145.0}
WIDE_FOV = {
    "CAM_FRONT": 55.0,
    "CAM_FRONT_LEFT": 55.0,
    "CAM_FRONT_RIGHT": 55.0,
    "CAM_BACK": 120.0,
    "CAM_BACK_LEFT": 110.0,
    "CAM_BACK_RIGHT": 110.0,
}
# `fx = (1600/2)/tan(hfov/2)` 的闭式结果(1600×900 下),独立算出
WIDE_FX = {
    "CAM_FRONT": 1536.78566,
    "CAM_FRONT_LEFT": 1536.78566,
    "CAM_FRONT_RIGHT": 1536.78566,
    "CAM_BACK": 461.88018,
    "CAM_BACK_LEFT": 560.16604,
    "CAM_BACK_RIGHT": 560.16604,
}
# 车身最后点(实测 `vehicle.audi.a2` 包围盒 x_min,米;CARLA 系)。挂点必须**在它之后**。
EGO_REAR_X = -1.8527
FRONT_CAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")


def _az_of_calib(q_nus) -> float:
    """标定四元数 → 视线轴在 nus 系的方位角(度)。相机自身系 +z = 视线轴。"""
    boresight = g.quat_to_matrix(g.quat_normalize(q_nus))[:, 2]
    return math.degrees(math.atan2(boresight[1], boresight[0]))


class TestWideRigDerivation:
    """wide rig 与官方 rig 的**等价/差别关系**(纯值,不连 CARLA)。

    这是"wide 是官方表的**扩展**而非第二份手抄"的判据:凡是声明"不动"的量必须**逐位相等**,
    凡是声明"改了"的量必须**恰好等于目标值**。两边都断言,才拦得住"顺手改了别的"。
    """

    def test_front_three_are_bit_identical(self):
        """前三个相机**一动不动**(挂点 + 四元数 + 姿态逐位相等)。"""
        for cam in FRONT_CAMS:
            assert NUS_WIDE_CAMERA_CALIBS[cam] == NUS_CAMERA_CALIBS[cam], f"{cam} 标定被改动"
            assert NUS_WIDE_CAMERA_RIG[cam] == NUS_CAMERA_RIG[cam], f"{cam} spawn 姿态被改动"

    def test_rear_three_move_to_the_rear_overhang(self):
        """后三路挂点:`carla_x == NUS_WIDE_REAR_X_CARLA` 且**在车身最后点之后**;`y/z` 与官方逐位相等。

        ★ 两个系的 x 都要钉:`NUS_WIDE_REAR_X_CARLA` 是**物理落点**(CARLA actor 系,决定画幅里
        有没有自己),落盘的声明值则是它减去 `NUS_EGO_ORIGIN_X`。只钉一个就会漏掉另一侧的换算。
        """
        assert NUS_WIDE_REAR_X_CARLA < EGO_REAR_X, "挂点必须在车身最后点之后(否则掠射线擦过车尾面)"
        for cam in WIDE_AZ:
            mount, rot = NUS_WIDE_CAMERA_RIG[cam]
            mount_nus = NUS_WIDE_CAMERA_CALIBS[cam][0]
            assert mount[0] == pytest.approx(NUS_WIDE_REAR_X_CARLA, abs=1e-12), cam
            assert mount[0] < EGO_REAR_X, cam
            assert mount_nus[0] == pytest.approx(NUS_WIDE_REAR_X_CARLA - g.NUS_EGO_ORIGIN_X, abs=1e-12), cam
            t_off = NUS_CAMERA_CALIBS[cam][0]
            assert mount_nus[1] == pytest.approx(t_off[1], abs=1e-12), cam  # nus 系 y 保留官方
            assert mount_nus[2] == pytest.approx(t_off[2], abs=1e-12), cam  # z 保留官方
            assert rot[0] == pytest.approx(NUS_CAMERA_RIG[cam][1][0], abs=1e-12), cam  # pitch 不变
            assert rot[2] == pytest.approx(NUS_CAMERA_RIG[cam][1][2], abs=1e-12), cam  # roll 不变

    def test_rear_azimuth_hits_the_design_target(self):
        """后三路轴方位角 == 设计目标 145/180/215(由四元数**独立反解**,不是读设计表)。

        同时钉 `yaw_carla == −az_nus` 精确成立 —— 这正是"左乘 `Rz(Δ)` 精确保持 pitch/roll"
        的推论(见 `nus_wide_camera_calibs` 的 docstring)。末项设计值写 −145°(= 真方位 215°)。
        """
        for cam, az in WIDE_AZ.items():
            got = _az_of_calib(NUS_WIDE_CAMERA_CALIBS[cam][1])
            assert got == pytest.approx(az, abs=1e-9), cam
            assert NUS_WIDE_CAMERA_YAW[cam] == pytest.approx(-az, abs=1e-9), cam
            assert max_hfov_no_ego(az) >= WIDE_FOV[cam], f"{cam} 超出零车体像素的解析上限"

    def test_azimuths_partition_the_rear_hemisphere_evenly(self):
        """等分后半球:相邻间隔 35°/35°,两端到 90°/270° 各留 35°(110° 半角正好贴边)。"""
        assert NUS_WIDE_CAMERA_AZ == WIDE_AZ, "设计表被改动(上方 WIDE_AZ 是独立硬编码的目标值)"
        az = sorted(v % 360.0 for v in WIDE_AZ.values())
        assert az == pytest.approx([145.0, 180.0, 215.0], abs=1e-12)
        assert az[0] - 90.0 == pytest.approx(55.0, abs=1e-12)
        assert 270.0 - az[-1] == pytest.approx(55.0, abs=1e-12)

    def test_no_ego_pixel_bound_is_tight_for_the_side_cameras(self):
        """解析上限在后侧两路**恰好取等**(110°)——再宽 1° 就必然拍到自己。"""
        for cam in ("CAM_BACK_LEFT", "CAM_BACK_RIGHT"):
            assert max_hfov_no_ego(WIDE_AZ[cam]) == pytest.approx(WIDE_FOV[cam], abs=1e-12), cam

    def test_analytic_bound_formula(self):
        """上限公式本体 `2·min(az−90, 270−az)`(纯函数,防被"简化"成别的)。"""
        assert max_hfov_no_ego(180.0) == pytest.approx(180.0)
        assert max_hfov_no_ego(90.0) == pytest.approx(0.0)
        assert max_hfov_no_ego(270.0) == pytest.approx(0.0)
        assert max_hfov_no_ego(108.595) == pytest.approx(37.19, abs=0.01)  # 官方轴的实测上限
        # 前半球**上限为负 = 无解**(轴朝前的相机多窄都拍得到自己)—— 保留符号,不 clamp 成 0
        assert max_hfov_no_ego(0.0) == pytest.approx(-180.0)
        assert max_hfov_no_ego(45.0) == pytest.approx(-90.0)
        for az in range(0, 90):
            assert max_hfov_no_ego(float(az)) < 0.0, az


class TestWideIntrinsics:
    """wide 的 K 由渲染反推(corner 主点),且 **CAM_BACK 不得回退到 180° 的奇异 K**。"""

    def test_fx_matches_closed_form(self):
        for cam, fx in WIDE_FX.items():
            assert ne.NUS_WIDE_CAMERA_INTRINSICS[cam][0] == pytest.approx(fx, abs=1e-4), cam

    def test_principal_point_is_corner_convention(self):
        """wide 的图是 CARLA 渲染栅格 ⇒ 主点走 corner 约定 `((w−1)/2, (h−1)/2)`。"""
        for cam in ne.NUS_CAMERAS:
            cx, cy = ne.NUS_WIDE_CAMERA_INTRINSICS[cam][1:]
            assert (cx, cy) == pytest.approx((799.5, 449.5), abs=1e-12), cam
            assert cx != pytest.approx(800.0, abs=1e-9), cam  # 与"重算主点"的历史缺陷划清界限

    def test_fov_round_trips_through_the_intrinsics(self):
        for cam, fov in WIDE_FOV.items():
            assert NUS_WIDE_CAMERA_FOV[cam] == pytest.approx(fov, abs=1e-12), cam
            assert ne.camera_fov_h_deg_for(cam, "wide") == pytest.approx(fov, abs=1e-9), cam

    def test_back_camera_is_not_the_singular_180(self):
        """★ 反回归:`CAM_BACK` **必须窄于 180°**,K 非奇异。

        `hfov=180°` ⇒ `fx = (w/2)/tan(90°) ≈ 4.9e-14` ⇒ `det(K) ≈ 0`,投影 `u = fx·x/z + cx`
        恒等于主点(整幅塌成一点),且 `inv(K)` 不存在 ⇒ devkit/auto3dlabel 的
        `cam2img` / `lidar2cam` 链直接失效。**这条断言是"别再改回 180°"的唯一闸门。**
        """
        assert WIDE_FOV["CAM_BACK"] < 180.0
        for cam in ne.NUS_CAMERAS:
            K = ne.camera_intrinsic(cam, "wide")
            det = K[0][0] * K[1][1] - K[0][1] * K[1][0]
            assert K[0][0] > 1.0, f"{cam} 的 fx 塌了({K[0][0]})"
            assert det > 1.0, f"{cam} 的 K 奇异(det={det})"

    def test_official_rig_is_untouched_by_the_wide_branch(self):
        """官方口径逐位不变(wide 是新增分支,不是替换)。"""
        for cam, fx in OFFICIAL_FX.items():
            assert ne.camera_intrinsic(cam)[0][0] == pytest.approx(fx, abs=1e-9), cam
            assert ne.camera_intrinsic(cam) == ne.camera_intrinsic(cam, "nuscenes"), cam
        assert ne.NUS_RIG_DEFAULT == "nuscenes"
        assert set(ne.NUS_RIGS) == {"nuscenes", "wide"}


class TestCollectNusRigSelection:
    """`collect_nus.rig_tables(rig)` 是采集器取表的**唯一入口**,三项必须同一 rig。"""

    def test_wide_branch_returns_the_wide_tables(self):
        carla_rig, calibs, fov = collect_nus.rig_tables("wide")
        assert carla_rig is NUS_WIDE_CAMERA_RIG
        assert calibs is NUS_WIDE_CAMERA_CALIBS
        assert fov is NUS_WIDE_CAMERA_FOV

    def test_default_branch_returns_the_official_tables(self):
        carla_rig, calibs, fov = collect_nus.rig_tables("nuscenes")
        assert carla_rig is NUS_CAMERA_RIG
        assert calibs is NUS_CAMERA_CALIBS
        assert fov is ne.NUS_CAMERA_FOV

    def test_unknown_rig_raises(self):
        with pytest.raises(ValueError, match="未知相机 rig"):
            collect_nus.rig_tables("wide2")

    def test_rig_choices_cover_the_export_side(self):
        """`--rig` 的 choices 就是导出侧的 `NUS_RIGS`(不另写一份元组)。"""
        src = (SIM / "collect_nus.py").read_text(encoding="utf-8")
        assert "choices=NUS_RIGS" in src, "`--rig` 的 choices 必须直接引自 `NUS_RIGS`"


class TestSurroundRigMatchesStudio:
    """`collect_surround.py`(训练数据)与 `live_common`(实时权重输入)**必须同口径**。

    **为什么单列一类(2026-09-23,§P-M.11 下游)**:`collect_surround.py` 是"声明 ≠ 渲染"的
    **第三处**残留 —— 六路共用一个 `cam_bp` 的 `fov=90`、K 也是 `calib_from_fov(w,h,90)`
    一表六用(官方逐通道是 64.31–64.96°×5 + 89.34°,差 25°)。后果不是"图难看",而是
    **实时 overlay 与权重错配**:`live_common.build_surround_rig` 若仍按 1242×375/90° 挂相机,
    它喂给新权重的图就与训练数据不同分布 —— 症状极像"权重训坏了"。故这里同时钉
    **两侧**("采集器逐通道" + "实时流跟着采集器走"),只钉一侧拦不住分叉。
    """

    def test_surround_has_its_own_frame_table(self):
        """采集器不共用 `CAM_ATTRS`(那是 KITTI/P1 线,十余处引用,动它就是动 A/B 红线)。"""
        assert not hasattr(collect_surround, "CAM_ATTRS"), (
            "collect_surround 不应再引用 carla_common.CAM_ATTRS(1242×375/fov90)"
        )
        assert collect_surround.SURROUND_CAM_ATTRS["image_size_x"] == "1600"
        assert collect_surround.SURROUND_CAM_ATTRS["image_size_y"] == "900"
        assert "fov" not in collect_surround.SURROUND_CAM_ATTRS, "fov 必须逐通道,不能留在共用属性里"

    def test_kitti_frame_constant_is_untouched(self):
        """`carla_common.CAM_ATTRS` 逐位不变(P1/KITTI 线的复现性红线)。"""
        from autodrivedata.sim.carla_common import CAM_ATTRS  # noqa: PLC0415

        assert CAM_ATTRS == {"image_size_x": "1242", "image_size_y": "375", "fov": "90"}

    def test_per_channel_fov_is_official(self):
        """落盘 fov(逐通道)就是官方 `NUS_CAMERA_FOV`,且至少五路显著窄于 90°。"""
        for cam, fx in OFFICIAL_FX.items():
            expect = math.degrees(2.0 * math.atan((1600 / 2.0) / fx))
            assert ne.NUS_CAMERA_FOV[cam] == pytest.approx(expect, abs=1e-9), cam
        narrow = [c for c in OFFICIAL_FX if ne.NUS_CAMERA_FOV[c] < 70.0]
        assert len(narrow) == 5, "六路共用 90° 的失效模式回来了"

    def test_intrinsics_use_corner_principal_point_not_official_cx(self):
        """★ 训练数据的 K:**fx 取官方(即 fov 取官方)、主点取 corner**。

        **这条是刻意的口径分歧,不是疏漏**(与 `collect_nus.py` 落官方 cx 不同,别来"统一"):
        - 官方 cx(792–829)是**真实相机的装配公差**,消费方是 devkit / auto3dlabel;
        - 本采集器的消费方是**我们自己的投影链**(GKT / `mapviz` / `eval_maptr`),而我们的图是
          **CARLA 渲染栅格**,其光栅中心按 §P-M 的 corner 裁决恒为 `(w−1)/2 = 799.5`。
        - 若落官方 cx:world→image 会**逐通道不一致地偏 7–27 px**(CAM_FRONT_LEFT 最大,
          cx=826.6 vs 799.5),等价 0.05–0.2 m 的 BEV 采样偏置 —— 又是一次"声明 ≠ 渲染"。

        故这里断言 ①主点 == corner;②**不等于**官方 cx(至少一路差 > 5 px,防被"顺手统一")。
        """
        for cam in ne.NUS_CAMERAS:
            K = collect_surround.calib_from_fov(1600, 900, ne.NUS_CAMERA_FOV[cam])["intrinsic"]
            assert (K[0][2], K[1][2]) == pytest.approx((799.5, 449.5), abs=1e-12), cam
            assert K[0][0] == pytest.approx(OFFICIAL_FX[cam], abs=1e-6), f"{cam} 的 fx 没跟官方走"
        off = abs(ne.NUS_CAMERA_INTRINSICS["CAM_FRONT_LEFT"][1] - 799.5)
        assert off > 5.0, "官方 cx 与 corner 的差别消失了?这条断言失去区分度"

    def test_studio_rig_frame_agrees_with_the_collector(self):
        """`live_common.rig_frame(nuscenes)` == 采集器的 (画幅, 逐通道 fov) —— 分叉即错配。"""
        w, h, fovs = lc.rig_frame(lc.RIG_NUSCENES)
        assert (w, h) == (1600, 900)
        assert (str(w), str(h)) == (
            collect_surround.SURROUND_CAM_ATTRS["image_size_x"],
            collect_surround.SURROUND_CAM_ATTRS["image_size_y"],
        )
        assert fovs == ne.NUS_CAMERA_FOV

    def test_studio_calibs_match_the_collector_calib_math(self):
        """实时流的 K 与采集器**同一算式、同一画幅**(逐位相等)。"""
        calibs = lc.surround_calibs(lc.RIG_NUSCENES, 1600, 900)
        for cam in ne.NUS_CAMERAS:
            got = calibs[cam]["intrinsic"]
            want = collect_surround.calib_from_fov(1600, 900, ne.NUS_CAMERA_FOV[cam])["intrinsic"]
            assert len(got) == len(want) == 3, cam
            for r_got, r_want in zip(got, want, strict=True):  # approx 不支持嵌套 list,逐行比
                assert r_got == pytest.approx(r_want, abs=1e-12), cam

    def test_legacy_rig_is_untouched(self):
        """`legacy`(**旧权重口径**)仍 1242×375 + 六路共用 90° —— 它服务已废弃的 ep512,不许被改。"""
        w, h, fovs = lc.rig_frame(lc.RIG_LEGACY)
        assert (w, h) == (1242, 375)
        assert set(fovs.values()) == {90.0}

    def test_rig_frame_overrides_only_the_raster(self):
        """显式 `width/height` 只换光栅、**不换 FoV**;显式 `fov` 才抹平六路。

        语义重要:FoV 是**相机属性**,改显示分辨率不该顺手改视野 —— 那会让"小分辨率显示档"
        静默变成"另一种相机口径"。
        """
        native = lc.rig_frame(lc.RIG_NUSCENES)
        small = lc.rig_frame(lc.RIG_NUSCENES, 640, 360)
        assert small[:2] == (640, 360)
        assert small[2] == native[2], "换画幅不该换 FoV"
        flat = lc.rig_frame(lc.RIG_NUSCENES, fov=90.0)
        assert set(flat[2].values()) == {90.0}, "显式 fov 才抹平"
        assert flat[:2] == native[:2], "只给 fov 不该动画幅"

    def test_spawn_index_and_stride_defaults_are_backward_compatible(self):
        """两个新参数的缺省必须**复现旧行为**(否则历史采集命令的语义静默变了)。"""
        src = (SIM / "collect_surround.py").read_text(encoding="utf-8")
        assert '"--spawn-index"' in src and '"--stride"' in src
        assert 'type=int,\n        default=None,\n        help="固定用第 N 个 spawn point' in src, (
            "--spawn-index 缺省必须是 None(= 沿用 spawn_ego 首空位)"
        )
        assert 'default=1,\n        help="每 N 个 tick 存一帧' in src, "--stride 缺省必须是 1"

    def test_stride_drains_every_camera_each_tick(self):
        """stride > 1 时**每个中间 tick 都要抽干全部相机队列**(§P-M.7 判据 ⑥ 同款坑)。

        只取"要存的那一帧"会让其余相机积压 ⇒ 下一帧读到更早的图(症状:图像与 ego 位姿
        差一拍,且**与 stride 无关地**偶发)。锁法:保存循环里不再出现 `qs[...].get`。
        """
        src = (SIM / "collect_surround.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
        gets = [n for n in ast.walk(main) if _call_name(n).endswith("].get") or _call_name(n) == "q.get"]
        # 唯一一处队列读取必须在 `for _ in range(args.stride)` 的 tick 循环体里
        assert len(gets) == 1, f"main() 里有 {len(gets)} 处队列读取;只允许 stride 循环内一处"
        assert any(n is gets[0] for n in ast.walk(main)), "队列读取不在 main 内?"

    def test_calib_json_carries_provenance(self):
        """落盘带 `spawn_index` / `stride` / `image_size`(旧产物无此键 = 旧口径)。"""
        src = (SIM / "collect_surround.py").read_text(encoding="utf-8")
        for key in ('calib["spawn_index"]', 'calib["stride"]', 'calib["image_size"]'):
            assert key in src, f"calib.json 缺溯源键 {key}"


class TestVerifyRigSelection:
    """`verify_nus_calib` 的判据表必须**跟着模块级 `RIG` 走**。

    ★ 为什么用 `monkeypatch` 拨这个全局:①⑤⑥ 的声明表选择散在四个函数里,若某处漏读
    `RIG`(写死 `NUS_CAMERA_RIG`),wide 的验收会**静默地**拿官方表去比 —— 判据全绿,
    而它验的根本不是这次要交付的 rig。这类失效只有"拨一下全局看结果变不变"才拦得住。
    """

    def test_rig_helpers_follow_the_module_global(self, monkeypatch):
        for rig, cameras in (("nuscenes", NUS_CAMERA_RIG), ("wide", NUS_WIDE_CAMERA_RIG)):
            monkeypatch.setattr(vnc, "RIG", rig)
            assert vnc.rig_cameras() is cameras, rig
            got = vnc.rig_intrinsics()
            assert set(got) == set(cameras), rig
            for cam in got:
                assert got[cam] == ne.camera_k(cam, rig), f"{rig}/{cam} 的内参没跟着 rig 走"

    def test_reports_are_written_per_rig(self):
        """`--rig wide` 的报告与数据根不许落回官方口径的路径(否则 wide 会**覆盖**官方报告)。"""
        src = (CALIB / "verify_nus_calib.py").read_text(encoding="utf-8")
        assert "nus_mini_wide" in src and "report_wide.json" in src
        assert "choices=NUS_RIGS" in src, "`--rig` 的 choices 必须直接引自 `NUS_RIGS`"

"""多传感器采集器的**结构回归钉**:LiDAR 挂点与自述字段(`calib.json` 的溯源)。

## 钉的是什么

2026-09-29 真踩到:采集器把 LiDAR 挂在 `carla.Transform()`(ego 原点),而
`slam_eval.LIDAR_LEVER = [1.2, 0, 1.65]` 是**照 `collect_slam` 的 `SENSOR_OFFSET` 定的**。
后果不是崩溃、不是报错,而是 `eval_slam` 补了一个**错的**杆臂 ⇒ ATE 带系统性偏差
(那条数据上实测 0.877 m,不可作干净参考)。与红线「ICP 的 T_delta 是点映射」同族:
**静默偏**,只有对上游挂点才知道数不对。

## 为什么写成两条,而不是"跑一遍采集看数"

跑真采集要 CARLA + 十几分钟,而且**判据本身还是数不对才知道**。这两条是纯静态的:
- 常数耦合:两个定义必须相等(任一侧被改动即红);
- 源码结构:采集器里 LiDAR 的 `spawn_actor` 第二个位置参数**必须是** `SENSOR_OFFSET`。

第二条**用 AST 不用字符串扫描** —— 上面的注释里就写着 `carla.Transform()`,
纯文本匹配会被自己的注释骗过(本项目已有的教训)。

## 第三组:`calib.json` 的**自述字段**

2026-09-30 踩到:`route` / `speed_mps` **恒写** `"const_speed"` + 默认 8.0,连
`--autopilot` 的 run 也照写 ⇒ 后来读这段数据的人会以为它是"定速可复现"的。
这组用**纯函数**(`collect_surround_lidar.provenance`)钉,不需要 CARLA。
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest

from autodrivedata.sim.collect_surround_lidar import AUTOPILOT_SPEED_PCT
from autodrivedata.sim.collect_surround_lidar import provenance as _provenance

_PKG = Path(__file__).resolve().parents[2]  # autodrivedata/
_COLLECTOR = _PKG / "sim" / "collect_surround_lidar.py"


def test_sensor_offset_equals_slam_lever():
    """`carla_common.SENSOR_OFFSET` 与 `slam_eval.LIDAR_LEVER` 必须逐分量相等。

    两者是**同一个物理事实的两份表达**(车顶前装 LiDAR 相对 ego 原点的偏移),
    分居两层(`sim/` 与 `slam/`)故无法共用一个常量 —— 那就用这条测试钉住耦合。
    """
    carla = pytest.importorskip("carla")
    from autodrivedata.sim.carla_common import SENSOR_OFFSET
    from autodrivedata.slam.slam_eval import LIDAR_LEVER

    assert isinstance(SENSOR_OFFSET, carla.Transform)
    got = [SENSOR_OFFSET.location.x, SENSOR_OFFSET.location.y, SENSOR_OFFSET.location.z]
    for g, want, axis in zip(got, LIDAR_LEVER, "xyz", strict=True):
        assert g == pytest.approx(float(want), abs=1e-6), f"杆臂 {axis} 不一致:{g} vs {want}"


def _lidar_spawn_transform_arg() -> ast.expr:
    """返回采集器里 LiDAR `spawn_actor(...)` 的**位姿实参**节点;找不到即失败。"""
    tree = ast.parse(_COLLECTOR.read_text(encoding="utf-8"))
    found: list[ast.expr] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "spawn_actor"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        # LiDAR 的蓝图变量名形如 `lid_bp`;相机是 `cam_bp`、雷达是 `r_bp` —— 只挑 lidar 那条
        if not (isinstance(first, ast.Name) and first.id.startswith("lid")):
            continue
        assert len(node.args) >= 2, "spawn_actor(bp, transform, attach_to=...) 至少两个位置参数"
        found.append(node.args[1])
    assert len(found) == 1, f"应当恰好有一处 LiDAR spawn_actor,实测 {len(found)}"
    return found[0]


def test_lidar_is_spawned_at_sensor_offset():
    """★ LiDAR 的 spawn 位姿必须是 `SENSOR_OFFSET`,**不是** `carla.Transform()`(ego 原点)。

    反向自证见 `test_bare_transform_would_be_caught` —— 证明这个判据真的会红,
    而不是"恰好也能过"。
    """
    arg = _lidar_spawn_transform_arg()
    assert isinstance(arg, ast.Name) and arg.id == "SENSOR_OFFSET", (
        f"LiDAR 挂点实参是 {ast.dump(arg)},必须是 SENSOR_OFFSET"
        f" —— 挂到 ego 原点会让 slam_eval 的杆臂补偿失效(静默偏差)"
    )


def test_bare_transform_would_be_caught():
    """反向自证:把判据喂给合成源码,坏写法必须被拒。

    `carla.Transform()` 全默认 —— 它 AST 上是 `Call`,不是 `Name`,上面那条会直接红。
    """
    tree = ast.parse("lidar = world.spawn_actor(lid_bp, carla.Transform(), attach_to=ego)\n")
    arg = next(
        n.args[1]
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "spawn_actor"
    )
    assert not (isinstance(arg, ast.Name) and arg.id == "SENSOR_OFFSET"), "坏写法竟然通过了判据"
    assert isinstance(arg, ast.Call), "裸 carla.Transform() 在 AST 上是 Call"


class TestProvenance:
    """`calib.json` 的自述字段:**跑的是什么就必须写什么**。

    这类字段错起来**不报错、不崩溃,只是把人引到错的做法上** —— 一段 autopilot 数据
    被标成 `const_speed` 之后,读方会当它是可复现的、拿去做 A/B(那是红线禁止的)。
    """

    @staticmethod
    def _args(**kw) -> argparse.Namespace:
        base = dict(autopilot=False, speed=8.0, no_radar=False, spawn_index=None, stride=5)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_autopilot_run_is_not_labelled_const_speed(self):
        """★ 核心判据:`--autopilot` 的 run **不得**出现 `const_speed` / 具体速度。"""
        p = _provenance(self._args(autopilot=True), "Town10HD_Opt")
        assert p["route"] == "autopilot"
        assert p["speed_mps"] is None, "autopilot 没有'设定速度'这回事 —— 写个数就是编的"
        assert p["reproducible"] is False, "下游必须能一眼看出它不能进 A/B"
        assert p["autopilot_speed_pct"] == AUTOPILOT_SPEED_PCT

    def test_const_speed_run_records_its_actual_speed(self):
        """反向对照:定速 run 必须把**真实速度**写进去(不是永远 8.0)。"""
        p = _provenance(self._args(autopilot=False, speed=6.5), "Town13")
        assert p["route"] == "const_speed" and p["speed_mps"] == 6.5 and p["reproducible"] is True
        # 两个模式的自述字段必须**互斥地**有值,不能两边都留一个像模像样的数
        assert p["autopilot_speed_pct"] is None

    def test_autopilot_speed_pct_is_the_spawn_value(self):
        """`calib.json` 记的百分比与 spawn 时设的必须是**同一个常量**导出。

        分两处写数的症状:改了 `vehicle_percentage_speed_difference` 的参数、
        忘了改自述字段 ⇒ 数据上写着 70%、实际跑的是 50% —— 全程无人报错。
        """
        src = _COLLECTOR.read_text(encoding="utf-8")
        assert "100.0 - AUTOPILOT_SPEED_PCT" in src, "spawn 侧必须由该常量导出,不许写死数字"
        assert "AUTOPILOT_SPEED_PCT if args.autopilot else None" in src, "自述侧必须取自同一常量"

    def test_has_lidar_key_distinguishes_from_collect_surround(self):
        """`has_lidar` 是"本 root 与 `collect_surround` 产物"的区分键 —— 恒 True 不许省。"""
        assert _provenance(self._args(), "Town10HD_Opt")["has_lidar"] is True

    def test_truthy_provenance_is_ast_visible(self):
        """`provenance()` 必须真的被 `main()` 调用 —— 否则字段全是空转。"""
        tree = ast.parse(_COLLECTOR.read_text(encoding="utf-8"))
        called = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "provenance"
            for n in ast.walk(tree)
        )
        assert called, "`provenance()` 定义了却没被调用 = 自述字段根本没写进产物"

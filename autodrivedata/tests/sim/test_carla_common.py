"""`carla_common` 的公共件单测 —— 目前只有 `load_world`(切图超时)。

**为什么要有这个文件(2026-10-03 实测踩坑)**:三个采集器各写过一遍
`set_timeout(60)` → `load_world` → `set_timeout(60)`,而 `collect_surround` 那处的
注释**自己写着**「~2min」—— 注释与超时值互相矛盾,当时就知道慢、只是没把超时对上。

一跑 Town13(12478 个 spawn point)就现形:

```
RuntimeError: time-out of 60000ms while waiting for the simulator,
              make sure the simulator is ready and connected to 127.0.0.1:2000
```

而**世界其实加载成功了** —— 紧接着 `get_world()` 拿到的是 `Carla/Maps/Town13/Town13`。
⇒ 这是"报错但没坏"的那类失效:照错误提示去查服务器,会白查一小时。

零 CARLA 服务器依赖(duck-typed 假 client;`carla` 包缺失时整体跳过,
与 `test_live_common` 同一路子)。
"""

from __future__ import annotations

from typing import cast

import carla
import pytest

from autodrivedata.sim.carla_common import (
    LOAD_WORLD_TIMEOUT_S,
    load_world,
)

#: 假 `load_world` 的返回值。做成**同一个哨兵对象**而不是字符串 ——
#: 比 `==` 更能说明"原样直通"(字符串相等可能来自一次多余的复制/转换)。
SENTINEL = object()


class FakeClient:
    """只实现 `ClientLike` 的两个方法;`calls` 记下**调用当时的超时值**。"""

    def __init__(self) -> None:
        self.timeout: float | None = None
        self.calls: list[tuple] = []

    def set_timeout(self, second: float) -> None:
        self.timeout = second
        self.calls.append(("timeout", second))

    def load_world(self, map_name: str, reset_settings: bool = True) -> carla.World:
        self.calls.append(("load", map_name, self.timeout))
        return cast(carla.World, SENTINEL)


class BoomClient(FakeClient):
    """加载失败(超时/名字写错)—— 用来看 `finally` 有没有跑。"""

    def load_world(self, map_name: str, reset_settings: bool = True) -> carla.World:
        self.calls.append(("load", map_name, self.timeout))
        raise RuntimeError("time-out of 60000ms while waiting for the simulator")


class TestLoadWorld:
    def test_timeout_is_raised_while_loading(self):
        """★ 加载那一刻的超时必须**远大于**平时的 60 s —— 这正是这个包装存在的理由。"""
        c = FakeClient()
        load_world(c, "Town13")
        assert c.calls[0] == ("timeout", LOAD_WORLD_TIMEOUT_S)
        assert c.calls[1] == ("load", "Town13", LOAD_WORLD_TIMEOUT_S)
        assert LOAD_WORLD_TIMEOUT_S > 60.0, "不大于 60 s 就等于没修 —— 大图照样超时"

    def test_timeout_is_restored_after_a_successful_load(self):
        """加载完恢复成后续惯用值,否则后面每一步都挂着 5 分钟的超时。"""
        c = FakeClient()
        load_world(c, "Town13", after_timeout=42.0)
        assert c.timeout == 42.0

    def test_timeout_is_restored_even_when_the_load_fails(self):
        """★ 失败也要恢复(`finally`)—— 否则一次明确的失败会被放大成
        "每一步都卡 5 分钟才报错",读起来像挂死而不是像报错。"""
        c = BoomClient()
        with pytest.raises(RuntimeError, match="time-out"):
            load_world(c, "Town13", after_timeout=42.0)
        assert c.timeout == 42.0, "异常路径没走 finally"

    def test_returns_what_load_world_returns(self):
        """返回值**原样直通** —— 调用方要拿它当场用(`world.get_map()`),不能被包成别的形状。"""
        assert load_world(FakeClient(), "Town13") is SENTINEL

    def test_custom_load_timeout_is_honoured(self):
        """两个超时都可覆盖:更慢的图(或更快的机器)不该改常量。"""
        c = FakeClient()
        load_world(c, "Town13", load_timeout=999.0, after_timeout=5.0)
        assert c.calls[1] == ("load", "Town13", 999.0)
        assert c.timeout == 5.0

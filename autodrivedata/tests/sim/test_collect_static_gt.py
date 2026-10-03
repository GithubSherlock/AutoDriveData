"""`collect_static_gt` 的**同 tick 自证**与**稳态排空**的回归钉(纯值,不连 CARLA)。

## 钉的是什么

2026-10-02 实测的真 bug:`image_2/{f}.png` 与**同一个采集器**写出的 `prop_inst/{f}.png`、
`prop_sem/{f}.png`、`static_prop_gt/{f}.json` **差 1 帧**。症状是"图看着正常、框整体偏",
与"标定错了"长得一样;而帧号一张张对得上、图也一张张出得来。判据(固定掩膜跨帧 argmax)
与修法见 `collect_static_gt.settle` 头注。

两条:
1. **帧号不等必须抛** —— 这是红线那条「两路本应逐像素相同的东西比一比」在这里的形态;
2. **排空一遍不够** —— 客户端投递是异步的,判"队列空"只代表*已经到的*取完了,
   **在途的**帧会补进来 ⇒ 恒定滞后。`settle` 要到"一次 tick 后每队列恰好 1 帧"才收。
"""

from __future__ import annotations

import queue

import pytest

from autodrivedata.sim.collect_static_gt import SETTLE_TRIES, assert_synced, drain, settle


class _Img:
    """`carla.Image` 的最小替身 —— 判据只用 `.frame`。"""

    def __init__(self, frame: int) -> None:
        self.frame = frame


class TestAssertSynced:
    def test_equal_frame_numbers_pass_and_report(self):
        note = assert_synced([("RGB", _Img(2165)), ("实例", _Img(2165)), ("语义", _Img(2165))])
        assert "2165" in note and "RGB" in note

    def test_differing_frames_raise(self):
        """★ 核心那条:三路帧号不等 ⇒ **当场停**,不许继续采。"""
        with pytest.raises(SystemExit, match="相机不同帧"):
            assert_synced([("RGB", _Img(2163)), ("实例", _Img(2165)), ("语义", _Img(2165))])

    def test_the_message_carries_every_leg_name_and_frame(self):
        """报错要能直接定位是哪一路掉队 —— 只写"不同帧"等于没报。"""
        with pytest.raises(SystemExit) as ei:
            assert_synced([("RGB", _Img(1)), ("实例", _Img(2))])
        msg = str(ei.value)
        assert "RGB=1" in msg and "实例=2" in msg

    def test_single_leg_is_not_checked(self):
        """只有一路(没开 `--props`/`--sem`)时**不能**判 —— 一条腿谈不上"同步"。"""
        assert assert_synced([("RGB", _Img(1))]) == ""

    def test_absent_legs_are_skipped(self):
        """`--props` 关着时实例那一路是 `None`,不该把它算成"帧号 0"。"""
        note = assert_synced([("RGB", _Img(7)), ("实例", None), ("语义", _Img(7))])
        assert "7" in note and "实例" not in note


class TestSettle:
    def test_converges_when_each_queue_gets_exactly_one(self):
        qs = [queue.Queue() for _ in range(3)]
        world = _World(qs)
        for q in qs:
            for f in (1, 2, 3):  # 积压 3 帧
                q.put(_Img(f))
        got = settle(world, qs)
        assert got == [1, 1, 1]
        assert all(q.qsize() == 0 for q in qs), "settle 收尾必须把队列留空,主循环才从干净态起步"

    def test_keeps_trying_while_frames_are_still_in_flight(self):
        """★ 这条钉的是**原实现的缺陷**:排空后仍有在途帧 ⇒ 第一轮 tick 会看到 2 帧。

        假世界在第 1 轮 tick 时往队列里塞 2 帧(模拟"排空那一刻还在途"的那一帧),
        之后恢复每 tick 1 帧 ⇒ `settle` 必须**再试一轮**才收。
        """
        qs = [queue.Queue() for _ in range(2)]
        world = _World(qs, extra_first_tick=1)  # 除当 tick 那帧外多塞 1 帧
        got = settle(world, qs)
        assert got == [1, 1], "在途帧没被排掉 —— 主循环会一直读到陈旧帧(实测滞后 1 帧)"
        assert world.ticks >= 2

    def test_gives_up_and_reports_rather_than_looping_forever(self):
        """收敛不了要**返回实况**让调用方喊出来,不是静默接受、也不是死循环。"""
        qs = [queue.Queue() for _ in range(2)]
        world = _World(qs, extra_every_tick=1)  # 永远多一帧,不可能收敛
        got = settle(world, qs)
        assert got == [2, 2]
        assert world.ticks == SETTLE_TRIES


class _World:
    """每次 `tick()` 给每个队列投一帧(可配置多投,模拟在途帧)。"""

    def __init__(
        self, queues: list[queue.Queue], *, extra_first_tick: int = 0, extra_every_tick: int = 0
    ) -> None:
        self.queues = queues
        self.ticks = 0
        self.frame = 100
        self._extra_first = extra_first_tick
        self._extra_every = extra_every_tick

    def tick(self) -> None:
        self.ticks += 1
        self.frame += 1
        for q in self.queues:
            q.put(_Img(self.frame))
            for _ in range(self._extra_every):
                q.put(_Img(self.frame))
            if self.ticks == 1:
                for _ in range(self._extra_first):
                    q.put(_Img(self.frame))


class TestDrain:
    def test_drain_returns_count_and_leaves_empty(self):
        q: queue.Queue = queue.Queue()
        for i in range(5):
            q.put(_Img(i))
        assert drain(q) == 5
        assert q.qsize() == 0
        assert drain(q) == 0

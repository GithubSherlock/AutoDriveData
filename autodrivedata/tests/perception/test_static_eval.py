"""`perception/static_eval` 的回归钉 —— 合成夹具,不碰真数据、不连 CARLA。

## 钉的是什么

这条判据的失败模式**全是"看起来在量、其实没量"**:

1. **窗画在错的地方**:信号锚点是**地面**点(实测 40/40 投出来落在 `Roads`),
   窗必须从锚点**往上**跨 `SIGNAL_WINDOW_M` 米。窗开到天上 = 真命中虚高;
   窗只有几像素 = 真命中虚低,而两者都"跑得出数"。
2. **对照没有判别力**:`real` 高不代表判据强 —— 一个到处都是标线的场景,
   随机指也有两位数。少了 `MIN_MARGIN` 那条,判据会在一片"路面本就这样"里判通过。
3. **"判不了"被当成"判错了"**:画外样本混进分母会把真命中率压下来,读成"xodr 与渲染不符"。
4. **没落位姿时硬算**:2026-10-02 之前采的 `static_gt` 没有 `camera` 键,拿默认内参算
   一整套**看着正常**的错数 —— 必须抛。

判据口径与定标过程见 `perception/static_eval.py` 头注。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.gt.props import CameraPose
from autodrivedata.gt.static_gt import LaneSegment, StaticFrame, StaticSignal
from autodrivedata.perception import static_eval as se
from autodrivedata.perception.sem_tags import SEM_TAGS

K = se.CameraIntrinsics(100, 100, 90.0)
#: 相机在原点、朝向世界 +x(CARLA yaw=0 ⇒ 相机系 z 轴朝前,与 `world_to_cam` 一致)。
CAM = CameraPose(location=(0.0, 0.0, 0.0), rotation_deg=(0.0, 0.0, 0.0), width=100, height=100, fov_deg=90.0)
TL, TS, RL = SEM_TAGS["TrafficLight"], SEM_TAGS["TrafficSigns"], SEM_TAGS["RoadLines"]


def _tag(shape=(100, 100)) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


class TestProjectionContract:
    def test_raw_projection_keeps_points_outside_the_frame(self):
        """★ 判据要的是**画外的 v**(信号窗上端点在近处目标上会跑出画面)。

        拿 `world_to_img` 那条(图外返 None)做,近处锚点会被判成"看不见",
        而它其实**看得见**,只是窗顶伸出去了。
        """
        # 20 m 前方、40 m 高:fov 90° ⇒ f = 50 px,Δv = 40/20×50 = 100 px ⇒ 远超上边缘
        far_up = se.project_raw((20.0, 0.0, 40.0), CAM, K)
        assert far_up is not None
        assert not se.in_frame(far_up[0], far_up[1], K), "这个点本就该在画外"
        assert far_up[1] < 0.0

    def test_behind_the_camera_is_none(self):
        assert se.project_raw((-5.0, 0.0, 0.0), CAM, K) is None


class TestSignalWindow:
    def test_the_window_really_spans_signal_window_metres(self):
        """★ 窗高**解析可核**:`Δv = SIGNAL_WINDOW_M / depth × fy`。

        这条挡的是"窗画对了位置但高度是拍脑袋的" —— 那种错只会让真命中率
        慢慢漂,不会红。
        """
        depth = 20.0
        span = se.signal_v_span((depth, 0.0, 0.0), CAM, K)
        assert span is not None
        v_top, v_bot = span
        assert v_bot - v_top == pytest.approx(se.SIGNAL_WINDOW_M / depth * K.fy, rel=1e-6)

    def test_hit_inside_the_window_miss_one_pixel_above_it(self):
        """★ 边界:窗内命中;窗**上方 1 px** 不算 —— 窗的上界真被用上了。"""
        tag = _tag()
        anchor = (20.0, 0.0, 0.0)
        uv = se.project_raw(anchor, CAM, K)
        assert uv is not None
        u = int(round(uv[0]))
        span = se.signal_v_span(anchor, CAM, K)
        assert span is not None
        v_top = int(np.floor(span[0]))

        inside = tag.copy()
        inside[v_top + 1, u] = np.uint8(TL)
        above = tag.copy()
        above[v_top - 1, u] = np.uint8(TL)

        assert se.signal_window_hit(inside, anchor, CAM, K, int(TL)) is True
        assert se.signal_window_hit(above, anchor, CAM, K, int(TL)) is False

    def test_one_pixel_lateral_offset_still_hits_but_forty_does_not(self):
        """★ 竖带半宽 1 px 的**理由**:同类灯箱宽 2–3 px,差 1 px 不该判错;
        而 40 px(横移对照量)必须判否 —— 否则那条对照就没有判别力。"""
        tag = _tag()
        anchor = (20.0, 0.0, 0.0)
        uv = se.project_raw(anchor, CAM, K)
        assert uv is not None
        u, v0 = int(round(uv[0])), int(round(uv[1]))
        tag[v0 - 5, u + se.SIGNAL_BAND_PX] = np.uint8(TL)
        assert se.signal_window_hit(tag, anchor, CAM, K, int(TL)) is True
        assert se.signal_window_hit(tag, (20.0, 0.0, 0.0), CAM, K, int(TL)) is True
        tag2 = _tag()
        tag2[v0 - 5, u + se.SIGNAL_SHIFT_PX] = np.uint8(TL)
        assert se.signal_window_hit(tag2, anchor, CAM, K, int(TL)) is False

    def test_stop_and_yield_expect_traffic_signs_not_traffic_light(self):
        """★ `Sign_*` 归 `TrafficSigns` —— 读成 TrafficLight 是把两类混成一类。"""
        assert se.SIGNAL_TAG_OF_KIND["stop"] == SEM_SIGNS
        assert se.SIGNAL_TAG_OF_KIND["yield"] == SEM_SIGNS
        assert se.SIGNAL_TAG_OF_KIND["traffic_light"] == TL

    def test_unknown_kind_is_skipped_not_scored(self):
        frame = StaticFrame(
            frame_id="0",
            ego_location=(0.0, 0.0, 0.0),
            ego_yaw_deg=0.0,
            signals=(StaticSignal("1", "unknown", "X", (20.0, 0.0, 0.0), 0.0),),
            camera=CAM,
        )
        assert se.eval_frame(frame, _tag(), np.random.default_rng(0)) == []


SEM_SIGNS = SEM_TAGS["TrafficSigns"]


class TestLaneWindow:
    def test_lane_points_are_scored_by_a_neighbourhood_not_a_single_pixel(self):
        """★ 单像素采样只有 21.7%/27.8%(渲染的线 ~2 px 宽)。±4 px 窗是**定标出来的**。"""
        assert se.LANE_WINDOW_PX == 4
        tag = _tag()
        anchor = (20.0, 0.0, 0.0)
        uv = se.project_raw(anchor, CAM, K)
        assert uv is not None
        u, v = int(round(uv[0])), int(round(uv[1]))
        tag[v, u + 2] = np.uint8(RL)  # 差 2 px:单像素判否,±4 窗判是
        assert se.lane_window_hit(tag, anchor, CAM, K, int(RL)) is True
        assert se.window_hit(tag, u, v, v + 1, int(RL), 0) is False

    def test_perp_is_per_point_not_segment_average(self):
        """★ 弯道上必须用**该点自己的**切向 —— 用整段平均方向,移的量就不是半个车道宽。"""
        # 折线在第二点处拐 90°:第一段沿 +y(横 = ±x),第二段沿 +x(横 = ±y)
        pts = ((0.0, 0.0, 0.0), (0.0, 10.0, 0.0), (10.0, 10.0, 0.0))
        p0 = se.lane_perp(pts, 0)
        p2 = se.lane_perp(pts, 2)
        assert p0 is not None and p2 is not None
        assert abs(abs(float(p0[0])) - 1.0) < 1e-9  # 第一点:横 = ±x
        assert abs(abs(float(p2[1])) - 1.0) < 1e-9  # 末点:横 = ±y

    def test_single_point_segment_has_no_perp(self):
        assert se.lane_perp(((0.0, 0.0, 0.0),), 0) is None


class TestVerdict:
    def test_perfect_real_with_equally_perfect_control_must_fail(self):
        """★ **这条是判据的判据**:真命中 1.0 而随机基准也是 1.0 ⇒ 位置没携带信息。

        少了它,"这场景到处都是标线"会被读成"xodr 与渲染完美吻合"。
        """
        gs = se.GroupStat("x", [True] * 20, [True] * 20, [True] * 20, 0)
        ok, why = gs.verdict()
        assert not ok and "对照" in why

    def test_high_real_with_empty_control_passes(self):
        gs = se.GroupStat("x", [True] * 20, [False] * 20, [False] * 20, 0)
        ok, _ = gs.verdict()
        assert ok

    def test_real_below_the_floor_fails_even_with_a_weak_control(self):
        gs = se.GroupStat("x", [True] * 5 + [False] * 15, [False] * 20, [False] * 20, 0)
        ok, why = gs.verdict()
        assert not ok and "真命中" in why

    def test_no_samples_is_unjudged_not_a_pass(self):
        """★ 空样本 → **`None`(未判)**,既不是通过也不是不通过。

        两种错误处置都试过,都不能要:
        - 判**过** ⇒ "没测到"被读成"没问题"(最坏);
        - 判**不过** ⇒ 一条画外的黄双实线会让判据永远红,然后被调阈值调绿。
        口径同 `sem_eval` 对空类:跳过 + **单独喊一声**。`--self-test` ④ 之外,
        `_report` 里那个 `n_unjudged` 计数就是"喊一声"。
        """
        ok, why = se.GroupStat("x").verdict()
        assert ok is None and "未判" in why

    def test_off_frame_reason_is_split_into_behind_and_fov(self):
        """★ "车后"与"视场外"是两回事:前者与判据无关,后者是这套 rig 覆盖不到。

        静态 GT 的 `LANDMARK_HORIZON` 是**圆形**过滤(车后的 landmark 也收),
        实测 65% 的信号锚点在相机后 —— 不看这个拆分,会误以为判据漏了一半数据。
        """
        gs = se.GroupStat("x")
        gs.add(se.Sample("x", None, None, None, "behind"))
        gs.add(se.Sample("x", None, None, None, "fov"))
        assert (gs.off_behind, gs.off_fov, gs.off_frame) == (1, 1, 2)
        assert "车后 1 / 视场外 1" in gs.verdict()[1]

    def test_control_takes_the_higher_of_the_two(self):
        gs = se.GroupStat("x", [True] * 20, [False] * 20, [True] * 4 + [False] * 16, 0)
        assert gs.control == pytest.approx(gs.rand_rate)

    def test_off_frame_samples_are_counted_separately_not_as_misses(self):
        """★ 画外 = "判不了",不许混进分母 —— 混进去会把夹具问题读成数据问题。"""
        gs = se.GroupStat("x")
        gs.add(se.Sample("x", None, None, None))
        gs.add(se.Sample("x", True, False, False))
        assert (gs.n, gs.off_frame) == (1, 1)
        assert gs.real_rate == 1.0


class TestCameraPoseRequired:
    def test_missing_pose_raises_instead_of_using_defaults(self):
        """★ 2026-10-02 之前采的 `static_gt` 没有 `camera` 键。

        拿默认内参硬算会得到一整套**看着正常**的错数(投影链全偏,而命中率只是低一点),
        必须当场抛。
        """
        frame = StaticFrame(frame_id="0", ego_location=(0.0, 0.0, 0.0), ego_yaw_deg=0.0)
        assert frame.camera is None
        with pytest.raises(ValueError, match="没落相机位姿"):
            se.eval_frame(frame, _tag(), np.random.default_rng(0))


class TestSelfTest:
    def test_the_bundled_self_test_passes(self):
        """判据自带自证(`--self-test`),这里保证它不会随改动静默腐化。"""
        assert se.self_test() is True


class TestSyntheticFrameEndToEnd:
    def test_a_frame_where_the_answer_is_known(self):
        """★ 端到端:合成一帧 —— 车道线上真有 `RoadLines`、锚点上方真有 `TrafficLight` ⇒ 全过。

        再来一帧把 tag 抹掉 ⇒ 真命中 0 ⇒ 判不过。两帧合起来证明这条链**两头都能动**。
        """
        lane = LaneSegment(
            side="right",
            mark_type="Solid",
            color="White",
            width=0.125,
            points=((15.0, -1.75, 0.0), (20.0, -1.75, 0.0), (25.0, -1.75, 0.0)),
        )
        sig = StaticSignal("1", "traffic_light", "Signal_3Light_Post01", (25.0, 2.0, 0.0), 0.0)
        frame = StaticFrame(
            frame_id="0",
            ego_location=(0.0, 0.0, 0.0),
            ego_yaw_deg=0.0,
            signals=(sig,),
            lane_lines=(lane,),
            camera=CAM,
        )

        good = _tag()
        for pt in lane.points:  # 把车道线**画在投影位置**上
            uv = se.project_raw(pt, CAM, K)
            assert uv is not None
            good[int(round(uv[1])), int(round(uv[0]))] = np.uint8(RL)
        span = se.signal_v_span(sig.location, CAM, K)
        uv_s = se.project_raw(sig.location, CAM, K)
        assert span is not None and uv_s is not None
        good[int(np.floor(span[0])) + 5, int(round(uv_s[0]))] = np.uint8(TL)

        stats = {s.group: se.GroupStat(s.group) for s in se.eval_frame(frame, good, np.random.default_rng(0))}
        for s in se.eval_frame(frame, good, np.random.default_rng(0)):
            stats[s.group].add(s)
        assert stats["lane:Solid"].real_rate == 1.0
        assert stats["signal:traffic_light"].real_rate == 1.0

        blank = _tag()  # 渲染里什么都没有 ⇒ 真命中必须掉到 0
        stats2 = {}
        for s in se.eval_frame(frame, blank, np.random.default_rng(0)):
            stats2.setdefault(s.group, se.GroupStat(s.group)).add(s)
        assert stats2["lane:Solid"].real_rate == 0.0
        assert stats2["signal:traffic_light"].real_rate == 0.0
        assert not stats2["lane:Solid"].verdict()[0]

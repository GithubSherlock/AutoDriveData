"""时序拼接的回归钉 —— 重点是 2026-09-30 加的三处(`--which` / 底图 / `Px` 换算)。

## 为什么这几条必须存在

这三处全是**"图能画出来"但结论会悄悄错**的类型:

| 改动 | 错了会怎样 |
|---|---|
| `--which {pred,gt}` | 取错来源 ⇒ 拼出来的是预测却标成 GT(或反过来),**图完全正常** |
| 底图的世界系换算 | 偏 1–3 m,散点图上看不出来(无 GT 可对) |
| `_panel_transform` → `Px` | 矢量与点云用两套像素公式 ⇒ 两张各自都对、**叠起来错位** |

判据一律是**手算可验的数值**,不是目检。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.map import bev_base
from autodrivedata.map import stitch_temporal as st
from autodrivedata.map.mapvec import MAPTR_CLASSES
from autodrivedata.map.mapvec_schema import MapVecFramePred, make_instance
from autodrivedata.map.mapviz import Px


def _rec(token: str = "seg0_000000", frame: int = 0) -> MapVecFramePred:
    """一帧:pred 与 GT 的**坐标刻意不同**,取错来源会被判据抓住。"""
    pred = make_instance("divider", [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]] + [[3.0, 0.0]] * 17, score=0.9)
    gt = make_instance("divider", [[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]] + [[12.0, 0.0]] * 17)
    return MapVecFramePred(frame=frame, token=token, score_thr=0.2, ckpt="x.pt", preds=(pred,), gts=(gt,))


class TestWhich:
    def test_pred_and_gt_are_not_interchangeable(self):
        """★ `--which` 必须真的换来源 —— 两边的坐标刻意不同,取错当场露馅。"""
        r, ego = _rec(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        xs_pred = st.frame_to_world(r, ego, "pred")[0][1][0][0]
        xs_gt = st.frame_to_world(r, ego, "gt")[0][1][0][0]
        assert xs_pred == pytest.approx(1.0) and xs_gt == pytest.approx(10.0)

    def test_gt_instances_have_no_score_and_do_not_crash(self):
        """GT 的 `score` 是 None ⇒ 必须回落 0.0,不能 `float(None)` 崩。"""
        r, ego = _rec(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert st.frame_to_world(r, ego, "gt")[0][2] == 0.0

    def test_default_is_pred(self):
        """缺省必须是 pred —— 旧的调用方(不传 which)行为逐位不变。"""
        r, ego = _rec(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert st.frame_to_world(r, ego) == st.frame_to_world(r, ego, "pred")


class TestPanelTransform:
    def test_matches_bev_base_px_semantics(self):
        """`_panel_transform` 返回的 `Px` 必须与手算的 `ox + (x−x0)·s` / `oy + (yy1−y)·s` 一致。

        这条是**两种绘制路径同源**的保证:矢量走它、点云走 `Px.arr`,公式分叉 = 两张图错位。
        """
        tx = st._panel_transform(x0=100.0, yy1=50.0, scale=2.0, ox=7.0, oy=11.0)
        assert isinstance(tx, Px)
        assert tx(100.0, 50.0) == pytest.approx((7.0, 11.0))
        assert tx(110.0, 40.0) == pytest.approx((27.0, 31.0))

    def test_is_a_value_not_a_late_bound_closure(self):
        """★ 值对象**没有绑定时机可言** —— 每个实例各带自己的参数。

        这是把循环内闭包(ruff B023,"全体静默用最后一轮的参数")换成 `Px` 的**直接收益**,
        故值得钉:两个不同参数造出来的变换必须互不影响。
        """
        a = st._panel_transform(0.0, 0.0, 1.0, 0.0, 0.0)
        b = st._panel_transform(0.0, 0.0, 3.0, 5.0, 6.0)
        assert a(1.0, 1.0) == pytest.approx((1.0, -1.0))
        assert b(1.0, 1.0) == pytest.approx((8.0, 3.0))
        assert a(1.0, 1.0) == pytest.approx((1.0, -1.0)), "a 被 b 的构造改掉了 = 又变回闭包"


class TestSegmentWorldPoints:
    """底图累积:点云 → 世界系。**几何用能闭式验的摆位**。"""

    @staticmethod
    def _root(tmp_path, frame: int, pts: list[tuple[float, float, float]]) -> None:
        d = tmp_path / "training" / "velodyne"
        d.mkdir(parents=True, exist_ok=True)
        arr = np.zeros((len(pts), 4), dtype=np.float32)
        arr[:, :3] = pts
        arr.tofile(d / f"{frame:06d}.bin")

    def test_world_geometry_is_hand_verifiable(self, tmp_path):
        """ego 在原点、yaw=0 ⇒ 世界坐标 = **velodyne 坐标 + 杆臂**。

        点在 velodyne 原点 ⇒ 世界坐标恰为 `LIDAR_LEVER`(`slam_eval` 的常量)。
        这条一眼可验,却是整条底图链的锚:任何一处漏了杆臂/翻错号都会偏。
        """
        from autodrivedata.slam.slam_eval import LIDAR_LEVER

        self._root(tmp_path, 0, [(0.0, 0.0, 0.0)])
        rec = _rec()
        poses = {"seg0_000000": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        pts, dropped = st.segment_world_points([rec], poses, {"seg0_000000": "seg0"}, tmp_path, [], 1000)
        np.testing.assert_allclose(pts["seg0"][0], LIDAR_LEVER, atol=1e-5)
        assert dropped == {"seg0": 0}

    def test_yaw_90_rotates_forward_into_plus_y(self, tmp_path):
        """ego yaw=90° ⇒ ego 前方的点落到世界 +y。**翻错手性/写成 −yaw 会在这里红。**"""
        self._root(tmp_path, 0, [(10.0 - 1.2, 0.0, 0.0)])  # 加回杆臂后 ego 系 (10,0,0)
        rec = _rec()
        poses = {"seg0_000000": [0.0, 0.0, 0.0, 90.0, 0.0, 0.0]}
        pts, _ = st.segment_world_points([rec], poses, {"seg0_000000": "seg0"}, tmp_path, [], 1000)
        got = pts["seg0"][0]
        assert got[0] == pytest.approx(0.0, abs=1e-5)
        assert got[1] == pytest.approx(10.0, abs=1e-5)

    def test_missing_pose_raises_instead_of_guessing(self, tmp_path):
        """位姿缺失**必须报错** —— 静默跳过会拼出一张"少了几段"的图,而图看着正常。"""
        self._root(tmp_path, 0, [(0.0, 0.0, 0.0)])
        with pytest.raises(SystemExit, match="token"):
            st.segment_world_points([_rec()], {}, {"seg0_000000": "seg0"}, tmp_path, [], 1000)

    def test_subsample_is_explicit_not_silent(self, tmp_path):
        """★ 超上限时按步长抽样,并把**丢弃量报出来** —— 静默截断会让人把"没画全"
        读成"这里就没有点"(项目对"静默上限"的既有纪律)。"""
        many = [(float(i), 0.0, 0.0) for i in range(100)]
        self._root(tmp_path, 0, many)
        rec = _rec()
        poses = {"seg0_000000": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        pts, dropped = st.segment_world_points([rec], poses, {"seg0_000000": "seg0"}, tmp_path, [], 25)
        assert len(pts["seg0"]) == 25 and dropped == {"seg0": 75}

    def test_segments_do_not_bleed_into_each_other(self, tmp_path):
        """★ 按 seg 分段是硬约束(段缝位移 56.8–109.6 m)⇒ 每段的点只进自己那一桶。"""
        for f in (0, 1):
            self._root(tmp_path, f, [(float(f), 0.0, 0.0)])
        recs = [_rec("a_000000", 0), _rec("b_000000", 1)]
        poses = {"a_000000": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "b_000000": [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        pts, _ = st.segment_world_points(
            recs, poses, {"a_000000": "segA", "b_000000": "segB"}, tmp_path, [], 99
        )
        assert set(pts) == {"segA", "segB"} and len(pts["segA"]) == 1 and len(pts["segB"]) == 1
        assert pts["segA"][0][0] < 10.0 < pts["segB"][0][0], "两段的点串了"


class TestRender:
    def test_base_layer_is_drawn_and_counted(self, tmp_path):
        """`render(base=...)` 必须**真的画上并回报点数** —— 0 是判据不是噪声。"""
        stats = {
            "seg0": {
                "insts": [("divider", [(0.0, 0.0), (10.0, 0.0)], 0.9, 1)],
                "n_frames": 1,
                "n_in": 1,
                "n_dedup": 0,
                "span_m": [10.0, 0.0],
            }
        }
        base = {"seg0": np.array([[0.0, 0.0, 0.0], [5.0, 1.0, 2.0]], dtype=np.float32)}
        out = tmp_path / "m.png"
        png, drawn = st.render(
            stats, pose_source="gt", fusion="cluster", out_png=out, base=base, which="pred"
        )
        assert png.exists() and drawn["seg0"] == 2

    def test_without_base_nothing_is_claimed(self, tmp_path):
        """不传底图时 `drawn` 全 0 —— 自述字段不许凭空有数。"""
        stats = {
            "seg0": {
                "insts": [("divider", [(0.0, 0.0), (1.0, 1.0)], 0.9, 1)],
                "n_frames": 1,
                "n_in": 1,
                "n_dedup": 0,
                "span_m": [1.0, 1.0],
            }
        }
        _, drawn = st.render(stats, pose_source="gt", fusion="cluster", out_png=tmp_path / "n.png")
        assert drawn["seg0"] == 0


class TestPalette:
    def test_every_class_has_a_color(self):
        """★ 少一个类色 ⇒ `CLASS_COLOR.get(cls, 灰)` 静默退回灰 ⇒ **两类同色**,
        图上分不出谁是谁。这条在加类/改名时是唯一的哨兵。"""
        missing = [c for c in MAPTR_CLASSES if c not in st.CLASS_COLOR]
        assert not missing, f"按时序拼接配色表缺类:{missing}"

    def test_base_palette_does_not_collide_with_vector_palette(self):
        """底图三色不得与矢量四色相同 —— 撞色违反本项目的绘制口径(见 mapviz 头注)。"""
        overlap = set(bev_base.LIDAR_GROUND_COLOR for _ in [0]) | {
            bev_base.LIDAR_OBJECT_COLOR,
            bev_base.RADAR_COLOR,
        }
        assert not (overlap & set(st.CLASS_COLOR.values())), "底图色与矢量色撞了"

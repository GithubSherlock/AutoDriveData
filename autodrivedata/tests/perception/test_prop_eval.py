"""道具判据的**纯值**部分 —— 用**合成 root** 把整条判据跑通,不碰 CARLA、不碰真数据。

## 为什么必须合成

判据塌掉的样子全都不是崩溃:

| 塌法 | 症状 |
|---|---|
| 尺子恒返回 1.0 | 每条 coverage 都是漂亮的 1.000,**永远绿** |
| 空掩膜被算成"通过" | 道具根本没渲染出来,却记成"框套住了" |
| 裁断样本并入裁决 | 物体被画幅裁掉时框按口径必然偏小 ⇒ **天天红**,然后被人调大容差调绿 |
| `instance_id` 对不上 | 全 root 一条样本都没有,而"没有样本"会被当成"没问题" |

合成 root 的好处是**答案是我摆出来的**:盒子摆在哪儿、掩膜画在哪儿、哪个角点出画,
全都是已知量。真数据上这些都得靠猜。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from autodrivedata.gt.props import CameraPose, PropBox, PropFrame
from autodrivedata.perception.prop_eval import (
    COVERAGE_TOL,
    RULER_SHIFT_PX,
    _coverage,
    _iou_box,
    _px_bbox,
    _shift,
    eval_root,
    project_gt_box,
)

W, H, FOV = 100, 100, 90.0  # fx = 50,主点 = (W-1)/2 = 49.5
#: 相机在**世界原点、零旋转**。⚠️ 此时**世界系 ≠ 相机系** —— 实测
#: `world_to_cam`:世界 +x → 相机 +z(**前**)、世界 +y → 相机 +x(**右**)、世界 +z → 相机 −y(**上**)。
#: 所以"摆在正前方 d 米"= `box_offset=(d, 0, …)`,不是 z。摆错轴的症状是**投影出画**(None),
#: 而不是报错 —— 本文件第一版就摆错成 z,四条测试同时红在"None"上。
CAM = CameraPose(location=(0.0, 0.0, 0.0), rotation_deg=(0.0, 0.0, 0.0), width=W, height=H, fov_deg=FOV)


def _prop(dist: float, *, iid: int, half: float = 0.5, lateral: float = 0.0) -> PropBox:
    """actor 原点放在世界原点,靠 `box_offset` 把**盒心**推到正前方 `dist` 米(底面贴地)。

    `box_offset` 是**局部系**偏移 —— actor yaw=0 时局部系 == 世界系,故这一读无歧义。
    盒心抬 `half` 是为了让底面落在 y_down = 0(相机水平面)上,与真实摆位同形。
    """
    return PropBox(
        type_id="static.prop.constructioncone",
        label="cone",
        location=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        size=(2 * half, 2 * half, 2 * half),
        box_offset=(dist, lateral, half),
        instance_id=iid,
    )


def _frame(props: tuple[PropBox, ...]) -> PropFrame:
    return PropFrame(
        frame_id="000000", ego_location=(0.0, 0.0, 0.0), ego_yaw_deg=0.0, props=props, camera=CAM
    )


def _paint_inside(ids: np.ndarray, box, iid: int, *, margin: int = 1) -> int:
    """把投影框**内缩 `margin` 像素**的那块涂成 `iid`,返回涂了几个像素。

    内缩是刻意的:`coverage` 的口径是"像素索引落在连续坐标的 min/max 之间",边缘差半格
    —— 贴着边涂会让测试变成在量那半格。**从框本身派生掩膜**,夹具就不会因为我手算错
    一个像素而红(第一版就是手算的,红了两次)。
    """
    x1, y1 = (int(np.ceil(v)) + margin for v in box[:2])
    x2, y2 = (int(np.floor(v)) - margin for v in box[2:])
    ids[y1 : y2 + 1, x1 : x2 + 1] = iid
    return int((y2 - y1 + 1) * (x2 - x1 + 1))


def _paint_outside(ids: np.ndarray, box, iid: int, *, width: int = 6) -> int:
    """在框**右边** `width` 像素处涂一块同样高的区域 —— 反面对照用。"""
    y1, y2 = int(np.ceil(box[1])), int(np.floor(box[3]))
    a = int(np.floor(box[2])) + 8  # 框右边界外 8 px 起,确保整块都在框外
    ids[y1 : y2 + 1, a : a + width] = iid
    return int((y2 - y1 + 1) * width)


def _write_root(tmp_path, frame: PropFrame, ids: np.ndarray):
    """把一帧写成 `collect_static_gt --props` 的最小产物结构。"""
    gt = tmp_path / "training/static_prop_gt"
    inst = tmp_path / "training/prop_inst"
    gt.mkdir(parents=True)
    inst.mkdir(parents=True)
    (gt / "000000.json").write_text(frame.to_json())
    Image.fromarray(ids.astype(np.uint16)).save(inst / "000000.png")
    return tmp_path


class TestRulerItself:
    """★ 尺子的**立论自证**:预测越偏,读数越差。恒返回常数的假尺子在这里必红。"""

    def test_coverage_drops_monotonically_under_shift(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:20, 10:20] = True
        box = (10.0, 10.0, 19.0, 19.0)
        covs = [_coverage(mask, _shift(box, dx)) for dx in (0, 5, 10, 20)]
        assert covs[0] == 1.0
        assert all(b <= a + 1e-12 for a, b in zip(covs[:-1], covs[1:], strict=True)), covs
        assert covs[-1] == 0.0

    def test_coverage_of_an_empty_mask_is_nan_not_one(self):
        """★ **没有轮廓 ≠ 框套住了**。`nan` 才有"没样本"的语义,`1.0` 会被读成"完美"。"""
        assert np.isnan(_coverage(np.zeros((4, 4), dtype=bool), (0.0, 0.0, 3.0, 3.0)))

    def test_px_bbox_of_an_empty_mask_is_none(self):
        assert _px_bbox(np.zeros((4, 4), dtype=bool)) is None

    def test_iou_of_identical_boxes_is_one(self):
        b = (1.0, 2.0, 3.0, 4.0)
        assert _iou_box(b, b) == 1.0


class TestProjectGtBox:
    def test_a_box_in_front_projects_around_the_principal_point(self):
        """正前方 10 m、半边长 0.5、fx=50 ⇒ 框约 5×5 px,落在主点附近。

        数值由 `project_corners` 实算(不是手推的)—— 手推容易把"半长/全长"记错一倍,
        而那正好是被测对象的经典错误之一。
        """
        p = _prop(10.0, iid=1)
        box, all_inside = project_gt_box(p, _frame((p,)))
        assert all_inside is True
        assert box == pytest.approx((46.87, 44.24, 52.13, 49.50), abs=0.05)

    def test_a_very_close_box_is_flagged_as_clipped(self):
        """★ 正前方 0.5 m 时角点投影到 ±1e17(画幅外)⇒ `all_inside=False`。

        这条**必须**单独分类:裁断样本按当前投影口径**必然** coverage 偏低,把它并进裁决
        会得到一个永远红色的判据 —— 而红久了就会被调容差调绿,那比没有判据更坏。
        """
        p = _prop(0.5, iid=1)
        box, all_inside = project_gt_box(p, _frame((p,)))
        assert all_inside is False
        assert box[2] < W  # 框右侧确实小得包不住物体 —— 这正是它要被单独分类的理由

    def test_a_box_behind_the_camera_returns_none(self):
        p = _prop(-5.0, iid=1)
        assert project_gt_box(p, _frame((p,))) is None

    def test_no_camera_in_the_frame_returns_none(self):
        p = _prop(10.0, iid=1)
        f = PropFrame(frame_id="0", ego_location=(0.0, 0.0, 0.0), ego_yaw_deg=0.0, props=(p,))
        assert project_gt_box(p, f) is None


class TestEvalRootOnSyntheticData:
    def test_a_mask_inside_the_box_passes(self, tmp_path):
        p = _prop(10.0, iid=7, half=1.25)  # 盒子够大,内缩后仍有 ≥30 px
        f = _frame((p,))
        ids = np.zeros((H, W), dtype=np.uint16)
        box, _ = project_gt_box(p, f)
        n = _paint_inside(ids, box, 7)
        assert n >= 30, f"夹具本身太小({n} px),测不出东西"
        res = eval_root(_write_root(tmp_path, f, ids))
        rows = res["stats"][7].rows
        assert len(rows) == 1 and rows[0].coverage == 1.0 and rows[0].all_inside

    def test_a_mask_that_sticks_out_of_the_box_is_caught(self, tmp_path):
        """★ 反面对照:轮廓**故意画到框外**,coverage 必须掉下来 —— 否则尺子是假的。"""
        p = _prop(10.0, iid=7, half=1.25)
        f = _frame((p,))
        ids = np.zeros((H, W), dtype=np.uint16)
        box, _ = project_gt_box(p, f)
        n_in = _paint_inside(ids, box, 7)
        n_out = _paint_outside(ids, box, 7)
        res = eval_root(_write_root(tmp_path, f, ids))
        row = res["stats"][7].rows[0]
        assert row.coverage == pytest.approx(n_in / (n_in + n_out))
        assert row.coverage < 1.0 - COVERAGE_TOL

    def test_an_id_that_never_renders_is_reported_not_counted(self, tmp_path):
        """★ `instance_id` 对不上是**静默**的:它只表现为"这条一次都没露面"。
        判据必须把它单独报出来 —— 那是"GT 与 id 图配错帧"的典型症状。"""
        p = _prop(10.0, iid=4242)
        ids = np.zeros((H, W), dtype=np.uint16)  # 画了别的 id
        ids[10:20, 10:20] = 99
        res = eval_root(_write_root(tmp_path, _frame((p,)), ids))
        assert res["stats"][4242].n_frames_in_view == 0
        assert [s.instance_id for s in res["never_seen"]] == [4242]

    def test_a_silhouette_below_the_minimum_is_not_a_pass(self, tmp_path):
        """★ **样本太少 ⇒ 作废**,不是"通过"。判据没有样本就不是判据。"""
        p = _prop(10.0, iid=7)
        ids = np.zeros((H, W), dtype=np.uint16)
        ids[47, 49] = 7  # 1 px,远小于 MIN_SILHOUETTE_PX
        res = eval_root(_write_root(tmp_path, _frame((p,)), ids))
        assert res["stats"][7].n_frames_in_view == 0

    def test_clipped_samples_are_counted_separately(self, tmp_path):
        """★ 裁断样本要**计数报出来**,不能并入裁决、也不能消失。"""
        p = _prop(0.5, iid=7)
        ids = np.zeros((H, W), dtype=np.uint16)
        ids[0:60, 50:100] = 7  # 贴右边缘的一大块
        res = eval_root(_write_root(tmp_path, _frame((p,)), ids))
        assert res["n_clipped"] == 1
        assert res["stats"][7].rows[0].all_inside is False

    def test_a_missing_root_shape_raises(self, tmp_path):
        with pytest.raises(SystemExit):
            eval_root(tmp_path)

    def test_columns_are_read_back_consistently(self, tmp_path):
        """落盘的 json 必须能被判据原样读回(帧号、相机、instance_id)。"""
        p = _prop(10.0, iid=7)
        ids = np.zeros((H, W), dtype=np.uint16)
        ids[46:48, 48:50] = 7
        root = _write_root(tmp_path, _frame((p,)), ids)
        raw = json.loads((root / "training/static_prop_gt/000000.json").read_text())
        assert raw["camera"]["fov_deg"] == FOV
        assert raw["props"][0]["instance_id"] == 7
        assert RULER_SHIFT_PX > 0  # 自证用的平移量必须非零,否则自证恒真

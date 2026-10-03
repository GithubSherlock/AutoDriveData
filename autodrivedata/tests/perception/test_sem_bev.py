"""sem_bev 的 YOLOPv2 推理链回归钉(2026-09-28 实测故障,此前本模块**零覆盖**)。

**实际故障**:`yolopv2_predict` 里有一句**函数体内**的
`from utils.utils import non_max_suppression, split_for_trace_model`(YOLOPv2 官方仓库自带的包),
本机没有该包 ⇒ 整个语义 BEV 出不了图。9-27 与 9-28 两次运行都抛
`ModuleNotFoundError: No module named 'utils'`,runlog 两次都如实记了,**但没有任何用例会报红**。
而那段代码**从来没被执行过**:唯一调用点写的是 `_, da, llm = ...`,返回值被直接丢弃。

这正是本项目重构文档列的「**方法体内的惰性 import**(逃过 `--collect-only`)」那一类,
所以本钉用**桩模型**跑通整条掩膜链 —— 只要有人把那个 import 加回来,这里立刻 `ModuleNotFoundError`。

**顺带钉住一直没人测的几何**:letterbox 填充 → 裁掉 padding → 缩回原图。曾踩过的坑是
「seg/ll 输出已是 640×640,官方 demo 的 `scale_factor=2` 是给 1280×720 显示用的,不是模型输出尺寸」,
所以这里按 640×640 喂,并断言可行驶带**落到了正确的行区间**(裁错/缩放错都会立刻偏出)。

**不钉什么**:掩膜像素级精度(INTER_NEAREST 的取整边界)。断言只取带宽中心区,边缘留余量。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from autodrivedata.map.mapviz import BEV_X, BEV_Y, CameraIntrinsics, cam_pose
from autodrivedata.perception import sem_bev
from autodrivedata.perception.sem_bev import DETECT_CLS, yolo11_instances, yolo11_object_mask

# KITTI 口径输入(与采集器一致);letterbox 到 640×640 后 r≈0.5153、dh≈223.5
W, H, IMGSZ = 1242, 375, 640


class _StubYoloPv2(torch.nn.Module):
    """官方 TorchScript 的输出契约:`((pred, anchor_grid), seg, ll)`。

    `pred`/`anchor_grid` 对掩膜链无意义(检测框那条路已删),给 None 即可 ——
    这一条本身就是回归钉:若有人把检测后处理加回来,它会去碰这两个 None。
    """

    def __init__(self, seg: torch.Tensor, ll: torch.Tensor) -> None:
        super().__init__()
        self._seg, self._ll = seg, ll

    def forward(self, x: torch.Tensor):  # noqa: ARG002
        return (None, None), self._seg, self._ll


def _stub_with_drivable_band(row_lo: int = 230, row_hi: int = 410) -> _StubYoloPv2:
    """seg 在 640 行里 [row_lo, row_hi) 判为可行驶(类别 1),其余背景;ll 全 1。"""
    seg = torch.zeros(1, 2, IMGSZ, IMGSZ)
    seg[0, 1, row_lo:row_hi, :] = 1.0
    ll = torch.ones(1, 1, IMGSZ, IMGSZ)
    return _StubYoloPv2(seg, ll)


def test_returns_two_masks_at_original_resolution():
    """签名契约:返回 (da, ll) 两个掩膜 —— 旧版第三个返回值 `det` 从未被消费。"""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    da, ll = sem_bev.yolopv2_predict(_stub_with_drivable_band(), img, torch.device("cpu"))
    assert da.shape == (H, W) and ll.shape == (H, W)
    assert da.dtype == bool and ll.dtype == bool


def test_letterbox_crop_resize_puts_the_band_in_the_right_rows():
    """几何钉:640 行里的 [230,410) 带 → 原图里的中心行区间;上下边缘必须干净。"""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    da, ll = sem_bev.yolopv2_predict(_stub_with_drivable_band(), img, torch.device("cpu"))

    # 带中心区(两侧各留 ≥25 行余量,避开 INTER_NEAREST 的取整边界)
    assert da[60:300, :].all(), "可行驶带没落到预期的行区间 —— 裁 padding 或缩回原图错了"
    # 上下边缘:640 行里 0..223 是 letterbox 填充(灰色),223..230 是背景 ⇒ 原图首行必须为 False
    assert not da[0, :].any(), "原图首行被判成可行驶 —— padding 没裁掉"
    assert not da[-1, :].any(), "原图末行被判成可行驶 —— 缩放越界"
    # ll 全 1 ⇒ 全图 True(同理,裁/缩错会露出 False)
    assert ll.all(), "车道线掩膜未覆盖全图 —— 裁剪或缩放不对"


def test_no_import_of_the_external_yolopv2_utils_package():
    """把「函数体内惰性 import 外部 `utils` 包」这件事本身钉住(见模块 docstring)。

    `utils.utils` 是 YOLOPv2 官方仓库自带的包,**不在本仓、也不在依赖里**。
    注意别与 `traj/convert_hivt_pt.py` 的 `from utils import TemporalData` 混淆 ——
    那个 `utils` 是 HiVT 仓库的,只在 hivt env 下有意义,是**同名多义**。
    """
    import ast
    import inspect

    src = inspect.getsource(sem_bev)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "utils.utils", "外部 YOLOPv2 utils 包又回来了"
        elif isinstance(node, ast.Import):
            assert all(a.name != "utils" for a in node.names), "外部 utils 包又回来了"


def test_band_shifts_with_the_band_position():
    """**对照**(反例):把带挪到 640 行里的另一处,输出必须跟着挪 —— 否则上一条可能是假过。"""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    da_low, _ = sem_bev.yolopv2_predict(_stub_with_drivable_band(230, 410), img, torch.device("cpu"))
    da_high, _ = sem_bev.yolopv2_predict(_stub_with_drivable_band(420, 600), img, torch.device("cpu"))
    assert not np.array_equal(da_low, da_high)
    assert da_low[60:300, :].all() and not da_high[60:300, :].any()


# ---------------------------------------------------------------------------
# 投影的**向量化 vs 标量对拍**(2026-10-01)
# ---------------------------------------------------------------------------
# `mask_to_bev` 当天从"逐像素调 ground_intersection"改成整块 numpy:判据(P2-A)要全量跑,
# 一帧 6 路可行驶区 360 万像素 × 20 帧 = 7200 万次 Python 调用,不向量化跑不完。
# 这类改写的典型失败是"图看着对但点整体偏了半个像素" —— 在 BEV 图上**完全看不出来**。
# 所以拿仍在的标量版(`ground_intersection` + `bev_to_px`)当 oracle,逐点对拍。

_EGO = [12.0, -3.5, 0.6, 17.0, 0.4, -0.3]  # x, y, z, yaw°, pitch°, roll°


def _bev_shape(pad: int = 0) -> tuple[int, int]:
    from autodrivedata.perception.sem_bev import BEV_PX

    h = int((BEV_Y[1] - BEV_Y[0]) / BEV_PX) + pad
    w = int((BEV_X[1] - BEV_X[0]) / BEV_PX) + pad
    return h, w


def _scalar_reference(mask, world_cam, intrinsics, ground_z, ego, shape) -> np.ndarray:
    """**标量**参考:逐像素走 `ground_intersection` + `bev_to_px`(判据用的那两个函数)。"""
    from autodrivedata.map.mapviz import bev_px_transform
    from autodrivedata.utils.geometry import ground_intersection

    px = bev_px_transform((shape[1], shape[0]))
    out = np.zeros(shape, dtype=bool)
    a = math.radians(ego[3])
    c, s = math.cos(a), math.sin(a)
    for v, u in zip(*np.where(mask > 0), strict=True):
        g = ground_intersection(world_cam, intrinsics, float(u), float(v), ground_z)
        if g is None:
            continue
        dx, dy = g[0] - ego[0], g[1] - ego[1]
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        bx, by = sem_bev.bev_to_px(lx, ly, shape[1], shape[0])
        assert (bx, by) == tuple(int(t) for t in px(lx, ly)), "两处栅格不一致"
        if 0 <= bx < shape[1] and 0 <= by < shape[0]:
            out[by, bx] = True
    return out


def _mask_scene(seed: int, n_px: int = 900, size=(640, 360)):
    """随机相机位姿 + 随机像素(偏向画面下半,多半打得到地面)。"""
    rng = np.random.default_rng(seed)
    se = (0.9, 0.4, 1.6, float(rng.uniform(-180, 180)), float(rng.uniform(-2, 2)), 0.0)
    intr = CameraIntrinsics(width=size[0], height=size[1], fov_h_deg=90.0)
    mask = np.zeros((size[1], size[0]), dtype=bool)
    mask[rng.integers(size[1] // 2, size[1], n_px), rng.integers(0, size[0], n_px)] = True
    return mask, cam_pose(_EGO, se), intr


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_vectorised_projection_matches_the_scalar_one(seed):
    """★ 两条实现对同一组像素必须给出**逐点相同**的 BEV 栅格。"""
    mask, world_cam, intr = _mask_scene(seed)
    shape = _bev_shape()
    gz = _EGO[2] - 0.5
    got = sem_bev.mask_to_bev(mask, world_cam, intr, gz, _EGO, shape, max_pixels=None)
    want = _scalar_reference(mask, world_cam, intr, gz, _EGO, shape)
    assert got.any(), "这次抽样一个点都没落进 BEV 窗口 —— 先换个场景再谈对不对"
    np.testing.assert_array_equal(got, want)


def test_vectorised_projection_matches_on_a_wider_window():
    """换窗口尺寸(多出边界几格)也必须一致 —— 栅格化的截断只在边界上露馅。"""
    mask, world_cam, intr = _mask_scene(7)
    shape = _bev_shape(pad=3)
    gz = _EGO[2] - 0.5
    got = sem_bev.mask_to_bev(mask, world_cam, intr, gz, _EGO, shape, max_pixels=None)
    want = _scalar_reference(mask, world_cam, intr, gz, _EGO, shape)
    assert got.any() and np.array_equal(got, want)


def test_upward_rays_draw_nothing():
    """画面上缘(射线朝上)`ground_intersection` 返回 None ⇒ 两条实现都不该画出东西。"""
    mask = np.zeros((360, 640), dtype=bool)
    mask[0:10, :] = True
    _m, world_cam, intr = _mask_scene(11)
    shape = _bev_shape()
    gz = _EGO[2] - 0.5
    assert not sem_bev.mask_to_bev(mask, world_cam, intr, gz, _EGO, shape, max_pixels=None).any()
    assert not _scalar_reference(mask, world_cam, intr, gz, _EGO, shape).any()


def test_subsampling_is_seeded_and_is_a_subset_of_the_full_path():
    """画图路径会下采样(40k 上限):必须**可复现**,且是**全量的子集**。

    判据走全量、画图走子采样 —— 若两条不是同一套投影,图与数就会互相矛盾,
    而那种矛盾看起来只是"图上稀疏一点"。
    """
    mask = np.zeros((360, 640), dtype=bool)
    mask[180:, :] = True
    _m, world_cam, intr = _mask_scene(19)
    shape = _bev_shape()
    gz = _EGO[2] - 0.5
    sub_a = sem_bev.mask_to_bev(mask, world_cam, intr, gz, _EGO, shape, max_pixels=500)
    sub_b = sem_bev.mask_to_bev(mask, world_cam, intr, gz, _EGO, shape, max_pixels=500)
    full = sem_bev.mask_to_bev(mask, world_cam, intr, gz, _EGO, shape, max_pixels=None)
    np.testing.assert_array_equal(sub_a, sub_b)
    assert sub_a.any() and full.any()
    assert not (sub_a & ~full).any(), "子采样画出了全量没有的格子 —— 两条路径不是同一套投影"
    assert sem_bev.MAX_PROJECT_PIXELS == 40_000, "默认上限是画图口径,改小会静默改变既有出图"


class _FakeTensor:
    """最小张量桩:只需 `.cpu().numpy()`(两条实现都只用这两个方法)。"""

    def __init__(self, arr):
        self._a = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _FakeYOLO:
    """最小 ultralytics 结果桩 —— 只回放给定的 `masks.data` / `boxes.cls` / `boxes.conf`。"""

    def __init__(self, masks, cls, conf):
        self._masks = type("M", (), {"data": [_FakeTensor(m) for m in masks]})()
        self._boxes = type("B", (), {"cls": _FakeTensor(cls), "conf": _FakeTensor(conf)})()

    def predict(self, *a, **k):
        return [type("R", (), {"masks": self._masks, "boxes": self._boxes})()]


class TestObjectMaskIsTheUnionOfInstances:
    """★ 类级掩膜必须是实例掩膜的**并集** —— 两条链看的必须是同一次推理。

    2026-10-01 把 `yolo11_object_mask` 重写成 `yolo11_instances` 的并集(实例级判据也要
    吃同一份预测)。合并**顺序无关**,所以旧实现(逐框 `|=`)与新实现应当**逐位相同**;
    这条钉的是重构没有偷偷改变类级判据的输入 —— 各跑一遍模型的话,报出的 mIoU 与图上
    看到的可能不是同一次推理,而"不是同一次"在下游看不出来。
    """

    @staticmethod
    def _yolo(n_keep: int = 2):
        m0 = np.zeros((8, 16), dtype=np.float32)
        m0[1:3, 1:3] = 1.0
        m1 = np.zeros((8, 16), dtype=np.float32)
        m1[4:6, 4:6] = 1.0
        keep = sorted(DETECT_CLS)[:n_keep]
        # 第 3 个框用类别 99(不在 DETECT_CLS 里)⇒ 两条实现都必须丢掉它
        return _FakeYOLO([m0, m1, m0], [keep[0], keep[1], 99], [0.9, 0.8, 0.7])

    def test_union_matches_the_old_per_box_loop(self):
        size, img = (16, 8), np.zeros((8, 16, 3), dtype=np.uint8)
        y = self._yolo()
        union = yolo11_object_mask(y, img, size)
        old = np.zeros(size[::-1], dtype=bool)
        for m, _ in yolo11_instances(y, img, size):
            old |= m
        assert np.array_equal(union, old)
        assert union.sum() == 8  # 两个 2×2 的块;类别 99 那个被丢掉

    def test_instances_keep_identity_and_conf(self):
        """★ 实例级判据要的就是这两样:每个实例**分开**,且带模型的**原始 conf**。"""
        size, img = (16, 8), np.zeros((8, 16, 3), dtype=np.uint8)
        inst = yolo11_instances(self._yolo(), img, size)
        assert len(inst) == 2, "类别 99 不该进实例表"
        assert [c for _, c in inst] == [0.9, 0.8]
        assert not np.array_equal(inst[0][0], inst[1][0]), "两个实例的掩膜必须分开,不许并成一块"

    def test_crowded_scene_would_be_indistinguishable_after_union(self):
        """反向对照:两个实例**并起来**与**分开**在类级掩膜上完全一样 —— 这正是实例级判据
        存在的理由(类级看不出"两辆车连成了一片")。"""
        size, img = (16, 8), np.zeros((8, 16, 3), dtype=np.uint8)
        y = self._yolo()
        inst = yolo11_instances(y, img, size)
        union = yolo11_object_mask(y, img, size)
        assert union.sum() == sum(m.sum() for m, _ in inst)  # 不重叠时两者一致
        assert len(inst) == 2 and union.ndim == 2

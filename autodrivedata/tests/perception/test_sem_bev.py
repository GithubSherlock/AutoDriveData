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

import numpy as np
import torch

from autodrivedata.perception import sem_bev

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

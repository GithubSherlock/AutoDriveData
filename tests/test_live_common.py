"""实时可视化拼图单测:等尺寸 `compose_grid` 的**防静默裁剪**守卫 + 混合尺寸 `compose_rows`。

**为什么要有这个文件(实测踩坑,2026-09-20)**:`PIL.Image.paste` 在源图**大于**目标框时
不报错、不缩放,**只贴左上角**——超出部分无声丢弃。studio 旧实现用等尺寸 `compose_grid`
把 6 路 1242×375 相机图塞进 621×187 的格子(`--maptr-scale 0.5`),一旦某格漏缩或想
"少缩一点",右半 + 下半就被裁掉,而**下半正是地面**。用户看到的现象是"6 视角摄像头的
FoV 缩小得都看不到地面了"。数值判据(当时实测):拼图格与「源图左上角裁剪」平均绝对差
**0.128**,与「整幅缩放」差 **66.18** ⇒ 是裁剪不是缩放。

本文件把这个坑钉成回归:
1. `compose_grid` 尺寸不符必须 `ValueError`(以前是静默裁);
2. `compose_rows` 按各格**原生像素**摆,整幅尺寸 = 行宽/行高的精确和,且**每格内容逐像素
   等于源图**(既不缩放也不裁剪)——这是"不为了整齐而缩减像素尺寸"的可判据形式。

零 CARLA 服务器依赖(只 import 模块;carla 包缺失时整体跳过,同 `test_geometry_carla_oracle`)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("carla")

BIN = Path(__file__).resolve().parents[1] / "bin"
if str(BIN) not in sys.path:  # bin/ 不是包(采集/可视化脚本),按脚本目录加路径
    sys.path.insert(0, str(BIN))

# `bin/` 不是包,静态分析跟不到上面那句运行时 `sys.path.insert` → 就地标注,不改全局 pyright 配置
from live_common import compose_grid, compose_rows  # noqa: E402  # pyright: ignore[reportMissingImports]

# studio 真实的三层布局(与 `live_studio.GRID_ROWS` 同口径):相机 1242×375 / 第三方 640×360 / BEV 420×420
CAM_W, CAM_H = 1242, 375
SPEC_W, SPEC_H = 640, 360
BEV = 420


def _solid(w: int, h: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def _textured(w: int, h: int, seed: int = 0) -> Image.Image:
    """非均匀图:任何缩放/裁剪都会改变像素 ⇒ 可逐像素判等。"""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


class TestComposeGridRejectsMismatch:
    """等尺寸拼图的唯一职责是"防静默裁剪":尺寸不符必须炸,不能悄悄裁。"""

    def test_wrong_size_raises(self):
        # 旧行为:1242×375 贴进 621×187 → 只留左上角,不报错(这就是用户看到的问题)
        with pytest.raises(ValueError, match="静默裁"):
            compose_grid([_solid(CAM_W, CAM_H, (1, 2, 3))], ["CAM_FRONT"], CAM_W // 2, CAM_H // 2)

    def test_oversize_raises_even_by_one_pixel(self):
        with pytest.raises(ValueError):
            compose_grid([_solid(10, 10, (0, 0, 0))], ["x"], 9, 10)

    def test_exact_size_ok_and_roundtrips(self):
        tile = _textured(8, 6, seed=1)
        grid = compose_grid([tile, tile], ["a", "b"], 8, 6, cols=2)
        assert grid.size == (16, 6)
        # 标签画在每格左上角 (6, 6) 起 → 只有那几个像素与源图不同,其余必须逐像素相同
        for col in (0, 8):
            cell = np.asarray(grid.crop((col, 0, col + 8, 6)), float)
            diff = np.abs(cell - np.asarray(tile, float)).mean(axis=2)
            assert (diff > 0).sum() <= 8 * 6 * 0.05, "格内容被改动(缩放/裁剪了)"


class TestComposeRowsNativePixels:
    """混合尺寸按行拼:**每格原样**,不缩放不裁剪 —— "不为了整齐而缩减像素尺寸"。"""

    def _layout(self):
        return [
            [("CAM_FRONT_LEFT", _textured(CAM_W, CAM_H, 1)), ("CAM_FRONT", _textured(CAM_W, CAM_H, 2))],
            [("CAM_BACK", _textured(CAM_W, CAM_H, 3))],
            [("THIRD_PERSON", _textured(SPEC_W, SPEC_H, 4)), ("BEV", _textured(BEV, BEV, 5))],
        ]

    def test_canvas_size_is_sum_of_rows(self):
        img = compose_rows(self._layout())
        # 宽 = 最宽行(2×1242);高 = 375 + 375 + 420
        assert img.size == (2 * CAM_W, CAM_H * 2 + BEV)

    def test_every_tile_is_pixel_identical(self):
        """核心判据:每格区域与源图逐像素相同(既没被缩放,也没被裁)。"""
        rows = self._layout()
        img = compose_rows(rows)
        assert img.size == (2 * CAM_W, CAM_H * 2 + BEV)
        y = 0
        for row in rows:
            x = (img.width - sum(t.width for _, t in row)) // 2  # 窄行居中
            for _, tile in row:
                cell = np.asarray(img.crop((x, y, x + tile.width, y + tile.height)), float)
                src = np.asarray(tile, float)
                # 左上角标签会盖住几个像素 → 用"未覆盖区域完全相同"判等
                assert np.array_equal(cell[20:, :], src[20:, :]), "格内容与源图不一致(被缩放/裁剪了)"
                x += tile.width
            y += max(t.height for _, t in row)

    def test_narrow_row_centered(self):
        img = compose_rows([[("a", _solid(10, 4, (255, 0, 0)))], [("b", _solid(30, 4, (0, 255, 0)))]])
        assert img.size == (30, 8)
        # 窄行居中:两侧各留 10 px 背景(黑)
        assert np.asarray(img)[0, 0].tolist() == [0, 0, 0]
        assert np.asarray(img)[0, 15].tolist() == [255, 0, 0]
        assert np.asarray(img)[0, 29].tolist() == [0, 0, 0]

    def test_left_align_option(self):
        img = compose_rows(
            [[("a", _solid(10, 4, (255, 0, 0)))], [("b", _solid(30, 4, (0, 255, 0)))]], center=False
        )
        assert np.asarray(img)[0, 0].tolist() == [255, 0, 0]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compose_rows([])
        with pytest.raises(ValueError):
            compose_rows([[]])

    def test_no_name_skips_label(self):
        """`--dump` 的 raw 拼图没有 BEV 行;缺名的行由调用方过滤后再传。"""
        img = compose_rows([[("", _solid(10, 4, (1, 1, 1)))]])
        assert np.asarray(img)[0, 0].tolist() == [1, 1, 1]  # 没画标签,原色保留


class TestStudioGridRows:
    """studio 的三层布局口径(名 → 行序)与用户要求一致。"""

    def test_row_order_matches_user_spec(self):
        import live_studio  # pyright: ignore[reportMissingImports]

        assert live_studio.GRID_ROWS == (
            ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"),
            ("CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"),
            (live_studio.SPECTATOR_NAME, live_studio.BEV_NAME),
        )

    def test_missing_names_are_skipped(self):
        """`--dump` 的 raw 拼图没有 BEV:整行缺了就丢行,不留空行。"""
        import live_studio  # pyright: ignore[reportMissingImports]

        t = _solid(4, 4, (0, 0, 0))
        rows = live_studio.grid_rows({"CAM_FRONT": t, live_studio.SPECTATOR_NAME: t})
        assert rows == [[("CAM_FRONT", t)], [(live_studio.SPECTATOR_NAME, t)]]

    def test_grid_rows_builds_three_rows_with_full_name_set(self):
        import live_studio  # pyright: ignore[reportMissingImports]

        names = [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
            live_studio.SPECTATOR_NAME,
            live_studio.BEV_NAME,
        ]
        by_name = {n: _solid(4, 4, (0, 0, 0)) for n in names}
        rows = live_studio.grid_rows(by_name)
        assert [len(r) for r in rows] == [3, 3, 2]
        assert [n for n, _ in rows[0]] == ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
        assert [n for n, _ in rows[1]] == ["CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]

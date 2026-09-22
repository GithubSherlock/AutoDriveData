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

import math
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


class TestRigSpec:
    """`rig_spec` / `resolve_rig`:两代 rig 的口径与"按权重选"的判据。

    **为什么钉**(2026-09-22,Plan2.md §P-M):外参必须与权重**训练时见过的一致**,否则第 i 路
    图与它学过的语义错位 —— 这是"看着能出图却全错"的典型,肉眼查不出来。
    """

    def test_nuscenes_spec_is_the_camera_rig_table(self):
        import live_common as lc  # pyright: ignore[reportMissingImports]

        from autodrivedata.camera_rig import NUS_CAMERA_RIG

        mounts, rots = lc.rig_spec(lc.RIG_NUSCENES)
        assert {n: mounts[n] for n in NUS_CAMERA_RIG} == {n: v[0] for n, v in NUS_CAMERA_RIG.items()}
        assert {n: rots[n] for n in NUS_CAMERA_RIG} == {n: v[1] for n, v in NUS_CAMERA_RIG.items()}

    def test_legacy_is_a_distinct_rig_not_a_near_copy(self):
        """legacy(180±55 / 共用挂点 / pitch=roll=0)与 nuscenes 是**两套口径**,不是近似。

        **实测差异形状**(2026-09-22 复核,别按直觉猜):前侧两台差 **110°**(镜像的形状),
        后侧两台只差 **14–16°**(legacy 的 `180±55` 恰好落在官方 `±108.6/110.8` 附近)——
        所以"差得多不多"不是判据,**挂点是否逐相机 + 有无 pitch/roll** 才是。
        """
        import live_common as lc  # pyright: ignore[reportMissingImports]

        leg_mounts, leg_rots = lc.rig_spec(lc.RIG_LEGACY)
        nus_mounts, nus_rots = lc.rig_spec(lc.RIG_NUSCENES)
        assert leg_rots["CAM_BACK_LEFT"][1] == 235.0 and leg_rots["CAM_BACK_RIGHT"][1] == 125.0

        def wrap(d: float) -> float:
            return abs((d + 180.0) % 360.0 - 180.0)

        # 前侧两台:镜像形状的 110° 偏差
        for name in ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"):
            assert wrap(nus_rots[name][1] - leg_rots[name][1]) > 100.0, name
        # 后侧两台:只差 ~15°(故"偏差大小"不能当判据)
        for name in ("CAM_BACK_LEFT", "CAM_BACK_RIGHT"):
            assert 5.0 < wrap(nus_rots[name][1] - leg_rots[name][1]) < 30.0, name

        # 真正的判别式:legacy 共用同一挂点平移 + pitch/roll 恒 0;nuscenes 逐相机 + 6DoF
        assert len({m for m in leg_mounts.values()}) == 1
        assert len({m for m in nus_mounts.values()}) == 6
        assert all(r[0] == 0.0 and r[2] == 0.0 for r in leg_rots.values())
        assert any(r[0] != 0.0 or r[2] != 0.0 for r in nus_rots.values())

    def test_auto_picks_legacy_for_legacy_ckpts(self):
        import live_common as lc  # pyright: ignore[reportMissingImports]

        for tag in lc.LEGACY_CKPTS:
            assert lc.resolve_rig("auto", f"outputs/{tag}.pt") == lc.RIG_LEGACY
        assert lc.resolve_rig("auto", "outputs/maptr_9999.pt") == lc.RIG_NUSCENES
        assert lc.resolve_rig("auto", None) == lc.RIG_NUSCENES  # 无权重 ⇒ 采集口径

    def test_explicit_choice_is_not_overridden(self):
        """显式给 rig 时不猜 —— 否则"我明明指定了 legacy"会被文件名静默改掉。"""
        import live_common as lc  # pyright: ignore[reportMissingImports]

        assert lc.resolve_rig(lc.RIG_LEGACY, "outputs/maptr_9999.pt") == lc.RIG_LEGACY
        assert lc.resolve_rig(lc.RIG_NUSCENES, "outputs/maptr_ep512.pt") == lc.RIG_NUSCENES


class _FakeTransform:
    """只带 4×4 位姿的桩(免起 CARLA 服务器)。"""

    def __init__(self, matrix: np.ndarray):
        self._m = np.asarray(matrix, dtype=np.float64)

    def get_matrix(self) -> list[list[float]]:
        return self._m.tolist()

    def get_inverse_matrix(self) -> list[list[float]]:
        return np.linalg.inv(self._m).tolist()


class _FakeActor:
    def __init__(self, matrix: np.ndarray):
        self._t = _FakeTransform(matrix)

    def get_transform(self) -> _FakeTransform:
        return self._t


def _pose(mount: tuple[float, float, float], rot_deg: tuple[float, float, float]) -> np.ndarray:
    """(平移米, (pitch,yaw,roll) 度) → 4×4 位姿,与 CARLA `get_matrix()` 同口径。"""
    from autodrivedata.geometry import carla_rotation_matrix

    m = np.eye(4)
    pitch, yaw, roll = (math.radians(v) for v in rot_deg)
    m[:3, :3] = carla_rotation_matrix((pitch, yaw, roll))
    m[:3, 3] = mount
    return m


class TestRigMountDeviation:
    """`rig_mount_deviation`:实挂 vs 规格的对账,**矩阵顺序写成 `cam·ego⁻¹` 必须被抓出来**。

    **为什么钉**(Plan2.md 红线 / milestone2 踩坑 1):写成 `cam·ego⁻¹` 会把 ego 的**世界坐标**
    混进平移块 —— ego 离原点越远偏差越大(实测 138 m),而**偏航仍恰好 0°**。只看偏航自检会
    静默放过,故本函数必须**同时报平移**,本类把这条钉死。
    """

    @staticmethod
    def _cams_and_ego(rig: str, ego_xy: tuple[float, float], order: str = "correct"):
        import live_common as lc  # pyright: ignore[reportMissingImports]

        mounts, rots = lc.rig_spec(rig)
        ego_m = _pose((ego_xy[0], ego_xy[1], 0.0), (0.0, 0.0, 0.0))
        cams = {}
        for name in mounts:
            spec = _pose(mounts[name], rots[name])
            world = ego_m @ spec if order == "correct" else spec @ ego_m
            cams[name] = (_FakeActor(world), None)
        return cams, _FakeActor(ego_m)

    def test_zero_deviation_when_rig_matches(self):
        import live_common as lc  # pyright: ignore[reportMissingImports]

        cams, ego = self._cams_and_ego(lc.RIG_NUSCENES, (100.0, -50.0))
        dev_t, dev_y = lc.rig_mount_deviation(cams, ego, lc.RIG_NUSCENES)
        assert dev_t < 1e-9 and dev_y < 1e-9

    def test_wrong_matrix_order_shows_in_translation_not_yaw(self):
        """**核心判据**:顺序写反 ⇒ 平移爆掉(≈ ego 到原点的距离)、偏航仍 ~0。"""
        import live_common as lc  # pyright: ignore[reportMissingImports]

        cams, ego = self._cams_and_ego(lc.RIG_NUSCENES, (100.0, -50.0), order="wrong")
        dev_t, dev_y = lc.rig_mount_deviation(cams, ego, lc.RIG_NUSCENES)
        assert dev_t > 100.0  # ≈ ‖ego 世界坐标‖ = 111.8 m
        assert dev_y < 0.01  # 只看偏航会判"没问题" —— 所以必须有平移项

    def test_deviation_grows_with_distance_from_origin(self):
        """同一份错顺序,ego 离原点越远偏差越大(故不能用"绝对值小"当判据)。"""
        import live_common as lc  # pyright: ignore[reportMissingImports]

        near = lc.rig_mount_deviation(
            *self._cams_and_ego(lc.RIG_NUSCENES, (1.0, 0.0), "wrong"), lc.RIG_NUSCENES
        )
        far = lc.rig_mount_deviation(
            *self._cams_and_ego(lc.RIG_NUSCENES, (500.0, 0.0), "wrong"), lc.RIG_NUSCENES
        )
        assert far[0] > near[0] * 100.0

    def test_catches_a_mirrored_rig(self):
        """把 legacy 的实挂位姿拿去对 nuscenes 规格 ⇒ 必须报出大偏差(镜像 rig 的判据形式)。"""
        import live_common as lc  # pyright: ignore[reportMissingImports]

        cams, ego = self._cams_and_ego(lc.RIG_LEGACY, (0.0, 0.0))
        dev_t, dev_y = lc.rig_mount_deviation(cams, ego, lc.RIG_NUSCENES)
        assert dev_t > 0.1  # legacy 共用挂点 vs nuscenes 逐相机,平移就对不上
        assert dev_y > 100.0  # 前侧相机镜像的 110° 量级

    def test_stale_zero_transforms_are_not_silently_zero_deviation(self):
        """tick 前 `get_transform()` 返回全 0(陈旧值)⇒ 必须**报出偏差**,不能判成"通过"。"""
        import live_common as lc  # pyright: ignore[reportMissingImports]

        ego = _FakeActor(_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        cams = {n: (_FakeActor(np.zeros((4, 4))), None) for n in lc.rig_spec(lc.RIG_NUSCENES)[0]}
        dev_t, dev_y = lc.rig_mount_deviation(cams, ego, lc.RIG_NUSCENES)
        assert dev_t > 0.5  # 挂点本身就有 ~1.5 m 的量级
        assert dev_y > 1.0  # 全 0 矩阵解出的 yaw 与规格不符 ⇒ 不会被当成"自检通过"


class TestDrawHudSecondLine:
    """`draw_hud(..., y=)` 的第二行:`--calib` 槽靠它叠 HUD,不能把第一行覆盖掉。"""

    def test_second_line_leaves_first_line_intact(self):
        import live_common as lc  # pyright: ignore[reportMissingImports]

        img = _solid(240, 40, (0, 0, 0))
        lc.draw_hud(img, "first", warn=False)
        first_row = np.asarray(img)[0].copy()
        lc.draw_hud(img, "second", warn=False, y=16)
        assert np.asarray(img)[0].tolist() == first_row.tolist()  # 第一行逐像素没变
        assert np.asarray(img)[16:].any()  # 第二行确实画上了

    def test_default_y_is_the_first_line(self):
        """不传 y 必须与旧行为逐像素一致(不然所有既有调用点的外观都会变)。"""
        import live_common as lc  # pyright: ignore[reportMissingImports]

        a, b = _solid(240, 40, (0, 0, 0)), _solid(240, 40, (0, 0, 0))
        lc.draw_hud(a, "same text")
        lc.draw_hud(b, "same text", y=0)
        assert np.array_equal(np.asarray(a), np.asarray(b))

    def test_warn_flag_changes_color(self):
        import live_common as lc  # pyright: ignore[reportMissingImports]

        a, b = _solid(240, 40, (0, 0, 0)), _solid(240, 40, (0, 0, 0))
        lc.draw_hud(a, "x")
        lc.draw_hud(b, "x", warn=True)
        assert not np.array_equal(np.asarray(a), np.asarray(b))

"""留出划分与多段组装的回归钉(§P-M.12)。

三条被测契约都是**静默失效型**:
- `select_frames`:训练集与留出集若有交集,AP 看着更高但结论作废 —— 不会报错;
- `roots_and_prefixes`:data_path 前缀写错会让所有旧单段命令指错文件 ——
  训练时表现为"图看着没错、学不动",也不会报错;
- `history_windows`:时序窗口若跨切分取前驱,留出帧的历史就在训练集里 —— 见 §P-M.12 阶段 4。

故这里的断言都针对"错法会不会被静默接受",不是覆盖率。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BIN = Path(__file__).resolve().parents[1] / "bin"
if str(BIN) not in sys.path:  # bin/ 不是包(脚本目录),与 test_live_common.py 同款
    sys.path.insert(0, str(BIN))

import assemble_maptr  # noqa: E402  # pyright: ignore[reportMissingImports]

from maptr_impl.dataset import (  # noqa: E402
    MapTRDataset,
    collate,
    history_windows,
    parse_frame_range,
    parse_segs,
    select_frames,
)


def _infos(segs: dict[str, int]) -> list[dict]:
    """{段名: 帧数} → 与 assemble_maptr 同构的最小 infos(只留选择器用到的键)。"""
    out = []
    for seg, n in segs.items():
        out += [
            {"seg": seg, "frame_in_seg": i, "frame": len(out) + i, "token": f"{seg}_{i:06d}"}
            for i in range(n)
        ]
    return out


class TestSelectFrames:
    def test_no_selector_returns_everything(self):
        assert select_frames(_infos({"seg0": 3, "seg1": 2})) == [0, 1, 2, 3, 4]

    def test_seg_keeps_only_that_segment(self):
        assert select_frames(_infos({"seg0": 3, "seg1": 2}), segs=["seg1"]) == [3, 4]

    def test_exclude_seg_is_the_route_level_holdout(self):
        # 路线级:训练 seg0..seg3、留出 seg4 —— 两侧必须无交集且并集为全集
        infos = _infos({"seg0": 2, "seg1": 2, "seg2": 2, "seg3": 2, "seg4": 2})
        train = select_frames(infos, exclude_segs=["seg4"])
        val = select_frames(infos, segs=["seg4"])
        assert not set(train) & set(val)
        assert sorted(train + val) == list(range(10))
        assert len(train) == 8 and len(val) == 2

    def test_keep_in_seg_is_the_frame_level_holdout(self):
        # 帧级:每段 [0,2) 训练、[2,4) 留出,段间交错但必须互斥
        infos = _infos({"seg0": 4, "seg1": 4})
        train = select_frames(infos, keep_in_seg=(0, 2))
        val = select_frames(infos, keep_in_seg=(2, 4))
        assert train == [0, 1, 4, 5]
        assert val == [2, 3, 6, 7]
        assert not set(train) & set(val)

    def test_boundary_is_half_open(self):
        infos = _infos({"seg0": 4})
        assert select_frames(infos, keep_in_seg=(1, 3)) == [1, 2]

    def test_selectors_intersect(self):
        infos = _infos({"seg0": 4, "seg1": 4})
        assert select_frames(infos, segs=["seg1"], keep_in_seg=(0, 2)) == [4, 5]

    def test_unknown_segment_raises_instead_of_returning_empty(self):
        # 段名拼错必须报错:静默给 0 帧会训出一个空模型还看着"跑完了"
        infos = _infos({"seg0": 2, "seg1": 2})
        with pytest.raises(ValueError, match="段名不存在"):
            select_frames(infos, segs=["seg9"])
        with pytest.raises(ValueError, match="段名不存在"):
            select_frames(infos, exclude_segs=["seg9"])

    def test_legacy_infos_without_seg_key_raise_only_when_selectors_used(self):
        # 旧版单段 infos 无 seg 键:不带选择器照旧可用(老命令不受影响),
        # 一旦用段级选择器就必须报错而不是把整份数据静默丢光
        legacy = [{"frame": i, "token": f"{i:06d}"} for i in range(3)]
        assert select_frames(legacy) == [0, 1, 2]  # 不带选择器必须照旧可用
        with pytest.raises(ValueError, match="缺 'seg' 键"):
            select_frames(legacy, exclude_segs=["seg4"])
        with pytest.raises(ValueError, match="缺 'frame_in_seg' 键"):
            select_frames(legacy, keep_in_seg=(0, 2))

    def test_empty_infos(self):
        assert select_frames([]) == []


class TestArgParsing:
    def test_parse_segs(self):
        assert parse_segs("seg0, seg1") == ["seg0", "seg1"]
        assert parse_segs(None) is None
        assert parse_segs("") is None
        assert parse_segs("  ") is None

    def test_parse_frame_range(self):
        assert parse_frame_range("80:100") == (80, 100)
        assert parse_frame_range(None) is None

    @pytest.mark.parametrize("bad", ["80", "80:100:2", "100:80", "5:5"])
    def test_parse_frame_range_rejects_bad_specs(self, bad):
        with pytest.raises(ValueError):
            parse_frame_range(bad)


class TestHistoryWindows:
    """时序窗口守卫:窗口的边界是**切分**不是段(§P-M.12 阶段 4)。

    这是本项目里最贵的静默失效:帧级切分画在段**内部**(`frame_in_seg == 80`),而
    stride 5 ⇒ 帧间只走 ~3 m、warp 近乎恒等 ⇒ 留出帧 80 的历史 79/78 若在训练集里,
    模型可以直接复制它背下来的地图。**症状**是按 `frame_in_seg` 分箱的 AP 在 80/81
    翘起,而训练日志(损失下降、留出确实是 80 帧)看不出任何异常。故这里钉的是
    "留出窗口里绝不出现训练帧",不是覆盖率。
    """

    def test_window_one_is_identity(self):
        infos = _infos({"seg0": 3})
        assert history_windows(infos, [0, 1, 2], 1) == ([(0,), (1,), (2,)], [])

    def test_window_is_oldest_to_newest_and_ends_at_self(self):
        infos = _infos({"seg0": 5})
        windows, dropped = history_windows(infos, [0, 1, 2, 3, 4], 3)
        assert windows == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
        assert dropped == [0, 1]  # 拿不到 2 个前驱 ⇒ 整帧丢弃,不静默截短

    def test_window_never_crosses_a_segment(self):
        infos = _infos({"seg0": 3, "seg1": 3})
        windows, dropped = history_windows(infos, list(range(6)), 3)
        assert windows == [(0, 1, 2), (3, 4, 5)]
        assert dropped == [0, 1, 3, 4]  # 段首两帧拿不到同段前驱

    def test_train_and_val_windows_are_disjoint(self):
        # 帧级切分落在段内部:训练 [0,5) / 留出 [5,10)。窗口只在本切分自己的帧列表里取
        # 前驱 ⇒ 留出帧 5/6 作为**目标**被丢(它们的前驱 3/4 在训练集),而 7/8/9 的窗口
        # 里只有留出帧自己(5/6 虽不当目标,仍是留出集的数据,当历史合法)。
        infos = _infos({"seg0": 10})
        train_sel = select_frames(infos, keep_in_seg=(0, 5))
        val_sel = select_frames(infos, keep_in_seg=(5, 10))
        tw, _ = history_windows(infos, train_sel, 3)
        vw, vdropped = history_windows(infos, val_sel, 3)
        assert tw == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
        assert vw == [(5, 6, 7), (6, 7, 8), (7, 8, 9)]  # 末元素才是目标帧
        assert vdropped == [5, 6]
        # ★ 本条是整个守卫的全部意义:窗口帧**全部**落在本切分自己的池里
        assert {i for w in tw for i in w} <= set(train_sel)
        assert {i for w in vw for i in w} <= set(val_sel)
        assert not set(train_sel) & set(val_sel)

    def test_reported_drop_count_matches_the_frame_level_split(self):
        # 真实口径:5 段 × 100 帧、切分在段内 80 ⇒ 每段丢 80/81 两帧 = 10 帧,
        # 留出集从 100 帧降到 90 帧(不静默:调用方拿 dropped 打日志)
        infos = _infos({f"seg{s}": 100 for s in range(5)})
        val_sel = select_frames(infos, keep_in_seg=(80, 100))
        windows, dropped = history_windows(infos, val_sel, 3)
        assert len(val_sel) == 100 and len(dropped) == 10 and len(windows) == 90
        assert {infos[i]["frame_in_seg"] for i in dropped} == {80, 81}

    def test_frame_numbers_are_not_used_as_history_key(self):
        # `frame` 在段缝处**连续无缺口**(0..499),拿它算前驱会在 idx=100 处静默解析到
        # seg0 的最后一帧 —— 所以键必须是 (seg, frame_in_seg)
        infos = _infos({"seg0": 3, "seg1": 3})
        assert [i["frame"] for i in infos] == [0, 1, 2, 3, 4, 5]
        windows, _ = history_windows(infos, [3], 2)
        assert windows == []  # seg1 的首帧没有同段前驱,尽管它的 frame 3 有"前一帧" 2

    def test_rejects_window_below_one(self):
        with pytest.raises(ValueError, match="window 需 ≥ 1"):
            history_windows(_infos({"seg0": 3}), [0, 1], 0)

    def test_missing_frames_in_sel_are_treated_as_absent(self):
        # 前驱在 infos 里存在但**不在本次切分里**(--frames 截断后就是这种情形)⇒ 同样丢弃
        infos = _infos({"seg0": 6})
        windows, dropped = history_windows(infos, [3, 4, 5], 3)
        assert windows == [(3, 4, 5)]
        assert dropped == [3, 4]


class TestWindowDataset:
    """`MapTRDataset(window=K)` 的返回结构 —— 单帧路径必须逐字节不变(单帧基线红线)。"""

    @staticmethod
    def _mock(monkeypatch):
        """避开真实图像 IO:`_images` 返回小张量,`_gt` 返回空标注。"""
        import torch

        monkeypatch.setattr(MapTRDataset, "_images", lambda self, info: {"CAM_FRONT": torch.zeros(3, 2, 2)})
        monkeypatch.setattr(MapTRDataset, "_gt", staticmethod(lambda info: [[], [], [], []]))

    def _infos_with_cams(self, n: int) -> list[dict]:
        return [
            {
                "seg": "seg0",
                "frame_in_seg": i,
                "frame": i,
                "token": f"seg0_{i:06d}",
                "cams": {"CAM_FRONT": {"data_path": f"{i}.jpg"}},
                "ego2global": [float(i), 0.0, 0.0, 0.0, 0.0, 0.0],
                "annotation": {},
            }
            for i in range(n)
        ]

    def test_window_one_structure_is_unchanged(self, monkeypatch, tmp_path):
        infos = self._infos_with_cams(4)
        self._mock(monkeypatch)
        ds = MapTRDataset(infos, tmp_path, frames=[0, 1, 2, 3])
        item = ds[0]
        assert set(item) == {"images", "pose", "gt"}  # 单帧口径:dict + 单个 pose
        assert item["pose"].shape == (6,)
        assert ds.dropped == []
        batch = collate([ds[0], ds[1]])
        assert batch["poses"].shape == (2, 6)

    def test_window_returns_k_frames_oldest_first(self, monkeypatch, tmp_path):
        infos = self._infos_with_cams(5)
        self._mock(monkeypatch)
        ds = MapTRDataset(infos, tmp_path, window=3)
        assert len(ds) == 3 and ds.dropped == [0, 1]
        item = ds[0]
        assert set(item) == {"images", "poses", "gt"}
        assert isinstance(item["images"], list) and len(item["images"]) == 3
        assert item["poses"].shape == (3, 6)
        # 旧 → 新:末帧的 x 最大(本测试里 x == 帧号)
        assert [float(p[0]) for p in item["poses"]] == [0.0, 1.0, 2.0]
        assert ds.infos[0]["frame_in_seg"] == 2  # GT 归属 = 窗口末帧

    def test_collate_stacks_window(self, monkeypatch, tmp_path):
        infos = self._infos_with_cams(6)
        self._mock(monkeypatch)
        ds = MapTRDataset(infos, tmp_path, window=3)
        batch = collate([ds[0], ds[1]])
        assert isinstance(batch["images"], list) and len(batch["images"]) == 3
        assert batch["images"][0]["CAM_FRONT"].shape == (2, 3, 2, 2)
        assert batch["poses"].shape == (2, 3, 6)
        assert len(batch["gts"]) == 2

    def test_window_requires_seg_keys(self, tmp_path):
        legacy = [{"cams": {"CAM_FRONT": {"data_path": "0.jpg"}}, "ego2global": [0.0] * 6}]
        with pytest.raises(ValueError, match="需要 infos 带 'seg'"):
            MapTRDataset(legacy, tmp_path, window=3)

    def test_all_frames_dropped_raises(self, tmp_path):
        infos = self._infos_with_cams(2)
        with pytest.raises(ValueError, match="没有一帧拿得到完整历史"):
            MapTRDataset(infos, tmp_path, window=3)


class TestRootsAndPrefixes:
    """data_path 前缀契约:单段 root = 段目录(前缀空),多段 root = 父目录(前缀 segK/)。"""

    def test_single_segment_has_empty_prefix(self, tmp_path):
        seg = tmp_path / "surround_drive"
        seg.mkdir()
        assert assemble_maptr.roots_and_prefixes(str(seg), None) == [(seg, "")]

    def test_multi_segment_prefixes_are_segment_names(self, tmp_path):
        for name in ("seg1", "seg0", "seg2"):
            (tmp_path / name).mkdir()
        got = assemble_maptr.roots_and_prefixes(None, str(tmp_path))
        assert got == [
            (tmp_path / "seg0", "seg0/"),
            (tmp_path / "seg1", "seg1/"),
            (tmp_path / "seg2", "seg2/"),
        ]

    def test_glob_ignores_files_and_unrelated_dirs(self, tmp_path):
        (tmp_path / "seg0").mkdir()
        (tmp_path / "seg_note.txt").write_text("x")  # 文件,不是目录
        (tmp_path / "other").mkdir()  # 不匹配 seg*
        assert assemble_maptr.roots_and_prefixes(None, str(tmp_path)) == [(tmp_path / "seg0", "seg0/")]

    def test_exactly_one_source_required(self, tmp_path):
        with pytest.raises(ValueError, match="只能给一个"):
            assemble_maptr.roots_and_prefixes(None, None)
        with pytest.raises(ValueError, match="只能给一个"):
            assemble_maptr.roots_and_prefixes(str(tmp_path), str(tmp_path))

    def test_empty_segs_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="没有 seg"):
            assemble_maptr.roots_and_prefixes(None, str(tmp_path))


class TestOverfitGate:
    """过拟合闸门的适用判据 —— 按**实际训练样本数**判,不按 `--frames` 标志。

    这是"假警报"型缺陷(§P-M.12):`--frames 0` 的语义是「**不截断**」(取全部帧),而
    闸门原写 `args.frames > 1` ⇒ `0 > 1` 为假 ⇒ **几百帧的多帧训练落进单帧锚点闸门**,
    打印 `FAIL:损失卡在 N——匹配/loss/坐标口径存在错误,禁止继续训练` 并以退出码 1 收场
    (2026-09-24 实测:400 帧 128 ep 跑到 2.31 时踩到)。权重在闸门**之前**已存盘故模型
    没坏,但假警报会让人以为口径真坏了而白查一轮 —— 故判据与"不许写回"都要钉住。
    """

    @staticmethod
    def _fn():
        # 惰性:该模块拉 torch/torchvision,不值得让整个文件收集变慢
        import train_maptr  # pyright: ignore[reportMissingImports]

        return train_maptr.is_single_frame_anchor

    def test_only_exactly_one_sample_is_the_anchor(self):
        fn = self._fn()
        assert fn(1) is True
        # 0 = 时序窗口把仅有的帧也丢了(不会走到闸门,但语义上不是锚点);
        # 2/320/400/500 = 真实的多帧训练集大小
        for n in (0, 2, 320, 400, 500):
            assert fn(n) is False

    def test_gate_is_not_keyed_on_the_cli_flag(self):
        """静态根因钉:闸门一旦写回 `args.frames > 1`,`--frames 0` 的老病立刻复发。

        判据是**源码形态**而不是"跑一遍 400 帧看看" —— 后者要 4 h GPU。
        只禁"与整数字面量比较":`args.frames > len(sel)`(截断越界检查)是合法的,
        故不能一概禁掉 `args.frames`。
        """
        import ast
        import inspect

        import train_maptr  # pyright: ignore[reportMissingImports]

        src = inspect.getsource(train_maptr)
        assert "is_single_frame_anchor(len(ds))" in src
        bad = [
            node
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Attribute)
            and node.left.attr == "frames"
            and any(isinstance(c, ast.Constant) and isinstance(c.value, int) for c in node.comparators)
        ]
        assert not bad, "过拟合闸门不许按 --frames 标志判定(应当按实际训练样本数)"

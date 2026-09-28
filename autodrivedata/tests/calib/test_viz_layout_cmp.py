"""viz_layout_cmp 的 `--a`/`--b` 回归钉(2026-09-28 实测故障)。

**实际故障**:本脚本读 `map_infos.json` / `calib.json` 用的是 `--a`/`--b`,**读图却拼死路径**
`<项目根>/outputs/surround_micro_{tag}` —— 于是"按 docstring 重采到别处、再用 `--a/--b` 指过去"
这条**它自己文档所教的用法**必然崩:数值表照常打印(那几行用的是 `--a/--b`),
一到拼图就 `FileNotFoundError: .../surround_micro_legacy/cam_back/000000.png`。
与「硬编码源码路径常量」是同一类失效(见 docs/refactor-2026-09.md 阶段 4 的 11 类引用形态)。

**判据不看代码改没改**:两个 root 的图**染成互不相同的纯色**,断言两张输出拼图各自取自
自己那个 root。路径一旦写死,两张图必然同色 —— 红蓝一看就分得开,不需要人眼比对。

**不钉什么**:拼图排版、GT 投影段数(那是投影链的事,归 tests/map/test_mapviz.py)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from autodrivedata.calib import viz_layout_cmp

CAMS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
K = [[621.0, 0.0, 620.5], [0.0, 621.0, 187.0], [0.0, 0.0, 1.0]]
SENSOR2EGO = [1.2, 0.0, 1.65, 0.0, 0.0, 0.0]
W, H = 1242, 375


def _make_root(root: Path, rgb: tuple[int, int, int]) -> None:
    """最小可跑 root:infos + calib + 6 路纯色图(无 annotation → 不画 GT)。"""
    root.mkdir(parents=True, exist_ok=True)
    cams = {
        c: {"data_path": f"{c.lower()}/000000.png", "sensor2ego": SENSOR2EGO, "intrinsic": K} for c in CAMS
    }
    rec = {
        "frame": 0,
        "seg": "seg0",
        "frame_in_seg": 0,
        "token": "000000",
        "ego2global": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "cams": cams,
        "annotation": {c: [] for c in ("divider", "ped_crossing", "boundary", "centerline")},
    }
    (root / "map_infos.json").write_text(json.dumps([rec]), encoding="utf-8")
    (root / "calib.json").write_text(
        json.dumps({c: {"sensor2ego": SENSOR2EGO} for c in CAMS}), encoding="utf-8"
    )
    for c in CAMS:
        d = root / c.lower()
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((H, W, 3), rgb, dtype=np.uint8)).save(d / "000000.png")


def test_images_are_read_from_the_given_roots(tmp_path, monkeypatch):
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    _make_root(a, (255, 0, 0))  # 布局 A = 纯红
    _make_root(b, (0, 0, 255))  # 布局 B = 纯蓝
    monkeypatch.setattr("sys.argv", ["viz_layout_cmp", "--a", str(a), "--b", str(b), "--out", str(out)])
    viz_layout_cmp.main()

    # (tag, 该 root 的图应当占主导的 RGB 通道) —— 红=0 / 蓝=2
    for tag, want in (("legacy", 0), ("official", 2)):
        img = np.asarray(Image.open(out / f"{tag}_frame000000.png").convert("RGB"), dtype=np.float64)
        assert img.shape[:2] == (H, W * len(CAMS)), f"{tag} 拼图尺寸不对: {img.shape}"
        own = img[..., want].mean()
        others = [img[..., i].mean() for i in range(3) if i != want]
        assert own > 200, f"{tag} 的图没取自它自己的 root(通道 {want} 均值仅 {own:.1f})"
        assert max(others) < 50, f"{tag} 的图串到了另一个 root(其余通道 {others})"

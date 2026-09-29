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
from autodrivedata.calib.camera_rig import CAMERA_GRID_ROWS

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

# 逐相机的**互不相同**的纯色 —— 用来断言"第 (r,c) 格摆的到底是哪一路"。
# 取色避开白/黑:那是标签 chip 的底色/字色,采样点落在 chip 上会误判。
_CAM_RGB: dict[str, tuple[int, int, int]] = {
    "CAM_FRONT_LEFT": (200, 0, 0),
    "CAM_FRONT": (0, 200, 0),
    "CAM_FRONT_RIGHT": (0, 0, 200),
    "CAM_BACK_RIGHT": (200, 200, 0),
    "CAM_BACK": (0, 200, 200),
    "CAM_BACK_LEFT": (200, 0, 200),
}


def _make_root(root: Path, rgb: tuple[int, int, int] | None = None) -> None:
    """最小可跑 root:infos + calib + 6 路纯色图(无 annotation → 不画 GT)。

    `rgb` 给一个颜色 = 六路同色(用于"图取自哪个 root"那类断言);
    给 `None` = **逐相机不同色**(用于"哪一格摆的是哪路"那类断言,见 `TestTileGrid`)。
    """
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
        color = rgb if rgb is not None else _CAM_RGB[c]
        Image.fromarray(np.full((H, W, 3), color, dtype=np.uint8)).save(d / "000000.png")


def test_images_are_read_from_the_given_roots(tmp_path, monkeypatch):
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    _make_root(a, (255, 0, 0))  # 布局 A = 纯红
    _make_root(b, (0, 0, 255))  # 布局 B = 纯蓝
    monkeypatch.setattr("sys.argv", ["viz_layout_cmp", "--a", str(a), "--b", str(b), "--out", str(out)])
    viz_layout_cmp.main()

    # (tag, 该 root 的图应当占主导的 RGB 通道) —— 红=0 / 蓝=2
    for tag, want in (("a", 0), ("b", 2)):
        img = np.asarray(Image.open(out / f"{tag}_frame000000.png").convert("RGB"), dtype=np.float64)
        assert img.shape[:2] == (H * 2, W * 3), f"{tag} 拼图不是 2 行 3 列: {img.shape}"
        own = img[..., want].mean()
        others = [img[..., i].mean() for i in range(3) if i != want]
        assert own > 200, f"{tag} 的图没取自它自己的 root(通道 {want} 均值仅 {own:.1f})"
        assert max(others) < 50, f"{tag} 的图串到了另一个 root(其余通道 {others})"


class TestTileGrid:
    """**六视角画布布局**回归钉(2026-09-28 用户口径)。

    要求:2 行 × 3 列;行优先读作「左前 / 前 / 右前」「**右后** / 后 / 左后」。
    原先这里是 `sorted(rec["cams"])` 的**字母序 + 一行六列**(`BACK, BACK_LEFT, BACK_RIGHT,
    FRONT, ...`)—— 第二行的左右与地理直觉相反,且与 `live_studio` 的实时拼图口径不一致。

    判据不看图:六路染成**互不相同的纯色**,逐格采样断言"第 (r,c) 格 = 该行的第 c 路"。
    采样点取**格中心**(标签 chip 画在左上角,采到那里会误判)。
    """

    def test_tiles_are_placed_in_the_declared_grid_order(self, tmp_path, monkeypatch):
        a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
        _make_root(a)  # 逐相机异色
        _make_root(b)
        monkeypatch.setattr("sys.argv", ["viz_layout_cmp", "--a", str(a), "--b", str(b), "--out", str(out)])
        viz_layout_cmp.main()

        img = np.asarray(Image.open(out / "a_frame000000.png").convert("RGB"))
        rows = CAMERA_GRID_ROWS
        assert len(rows) == 2 and all(len(r) == 3 for r in rows), f"画布不是 2×3: {rows}"
        for r, row in enumerate(rows):
            for c, cam in enumerate(row):
                cy, cx = r * H + H // 2, c * W + W // 2
                got = tuple(int(v) for v in img[cy, cx])
                assert got == _CAM_RGB[cam], (
                    f"第 ({r},{c}) 格应当是 {cam}{_CAM_RGB[cam]},实测 {got} —— 格位或行序错了"
                )

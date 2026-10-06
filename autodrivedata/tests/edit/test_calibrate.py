"""`edit/calibrate` 的判据回归钉 —— 把 #12 那三步(扫 / 拟合 / 出配方)钉成一条命令。

## 这一层为什么值得钉

#12 的用法原来是**人工看表 → 人工内插 → 人工抄命令**,而曲线**底图相关**(每换一个
生成基底都要重标)。自动化本身不难,**难的是三件容易静默错的事**:

1. ★ **口径**:11 点插值下曲线是**台阶状的、甚至不单调**(实测 `blur`:
   `−0.013 / +0.055 / −0.034`;101 点下单调) —— 拿它拟合 = 把尺子的台阶当退化效应;
2. ★ **落不上要说话**:目标比曲线最轻一档还轻 ⇒ `β=None`,**那是结论不是失败**
   (§1.6 的"生成雾比真雾轻一个量级"正是这一支);
3. ★ **不单调要报警**:拟合值照样算得出来,只是不可信 —— 静默返回一个数最坑。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.edit import calibrate as C

CURVE = [
    {"level": 0.04, "delta": 0.01, "res": {}},
    {"level": 0.06, "delta": -0.05, "res": {}},
    {"level": 0.08, "delta": -0.20, "res": {}},
    {"level": 0.10, "delta": -0.30, "res": {}},
]


def _res(curve):
    return {"base_mAP": 0.80, "base_res": {}, "n_points": 101, "curve": curve}


class TestMonotone:
    def test_decreasing_is_monotone(self):
        assert C.is_monotone(CURVE)

    def test_a_bump_breaks_it(self):
        """★★ 实测量到的那种形状:`blur` 在 11 点下是 `− / + / −`,中间那档反而涨。"""
        assert not C.is_monotone(
            [
                {"level": 3, "delta": -0.013},
                {"level": 7, "delta": +0.055},
                {"level": 15, "delta": -0.034},
            ]
        )

    def test_flat_is_monotone(self):
        assert C.is_monotone([{"level": 1, "delta": 0.0}, {"level": 2, "delta": 0.0}])


class TestRecipe:
    def test_interpolates_and_emits_a_runnable_command(self):
        r = C.recipe(
            C.Path("base_root"), _res(CURVE), -0.10, "fogdepth", n_points=101, backend="yolo", conf=0.25
        )
        # −0.10 落在 0.06(−0.05)与 0.08(−0.20)之间,**线性内插**给 0.06 + 1/3·0.02
        assert r["beta"] == pytest.approx(0.0667, abs=1e-4)
        assert r["curve_monotone"] is True
        # ★ 配方必须**可直接粘贴** —— 含 --src / --kind / --level 三件
        assert "--src base_root" in r["recipe"] and "--kind fogdepth" in r["recipe"]
        assert f"--level {r['beta']:.4f}" in r["recipe"]
        assert r["bracket"]["low"]["level"] == 0.06 and r["bracket"]["high"]["level"] == 0.08

    def test_lighter_than_lightest_says_so(self):
        """★ `+0.05` 比最轻一档(`+0.01`)还轻 ⇒ 落不上。**这是结论,不是失败。**"""
        r = C.recipe(C.Path("b"), _res(CURVE), +0.05, "fogdepth", n_points=101, backend="yolo", conf=0.25)
        assert r["beta"] is None
        assert "落不上" in r["verdict"]
        assert "recipe" not in r, "落不上就不该给出可粘贴的命令"

    def test_non_monotone_warns_even_though_it_can_fit(self):
        """★★ **最容易静默出错的一支**:台阶曲线照样内插得出来,只是不可信。"""
        bumpy = [
            {"level": 3, "delta": -0.013, "res": {}},
            {"level": 7, "delta": +0.055, "res": {}},
            {"level": 15, "delta": -0.034, "res": {}},
        ]
        r = C.recipe(C.Path("b"), _res(bumpy), +0.02, "fogdepth", n_points=11, backend="yolo", conf=0.25)
        assert r["beta"] is not None, "算得出来 —— 正因为算得出来才必须报"
        assert r["curve_monotone"] is False
        assert "不单调" in r["verdict"]

    def test_empty_curve_does_not_pretend(self):
        r = C.recipe(C.Path("b"), _res([]), -0.2, "fogdepth", n_points=101, backend="yolo", conf=0.25)
        assert r["beta"] is None

    def test_records_the_gauge_it_used(self):
        """口径必须随读数走 —— 11 点与 101 点是两把尺子。"""
        r = C.recipe(C.Path("b"), _res(CURVE), -0.10, "fogdepth", n_points=101, backend="yolo", conf=0.25)
        assert (r["grid_points"], r["backend"], r["conf"]) == (101, "yolo", 0.25)


class TestWiring:
    """★ 源码级接线判据 —— `--grid-points` 这类参数最容易"加了但没传下去"而**不报错**。

    本仓 2026-10-06 实测同族:`noise_curve --levels` 就是这样被静默忽略的
    (`main` 里那一行没改上,给了参数照样跑默认档)。
    """

    def _src(self):
        import ast
        from pathlib import Path

        return ast.parse((Path(__file__).resolve().parents[2] / "edit" / "calibrate.py").read_text("utf-8"))

    def test_main_passes_grid_points_into_build_curve(self):
        import ast

        calls = [
            n
            for n in ast.walk(self._src())
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "build_curve"
        ]
        assert calls, "找不到 build_curve 调用"
        assert any(any(k.arg == "n_points" for k in c.keywords) for c in calls), (
            "`main` 没把 `n_points` 传给 `build_curve` —— `--grid-points` 会被静默忽略"
        )

    def test_default_gauge_is_the_fine_one(self):
        """默认必须是 **101** —— 11 点下曲线是台阶状的,拟合出来的东西没有意义。"""
        assert C.DEF_LEVELS and "0.04" in C.DEF_LEVELS
        src = (C.__file__ and open(C.__file__, encoding="utf-8").read()) or ""
        assert "default=101" in src, "`--grid-points` 的默认值应当是 101"


def _fake_root(tmp_path, n_frames=3, n_gt=4):
    """带真值深度的极小 root(`fogdepth` 必须有 `training/depth/`)。"""
    from PIL import Image

    img_dir = tmp_path / "training" / "image_2"
    lab_dir = tmp_path / "training" / "label_2"
    dep = tmp_path / "training" / "depth"
    for d in (img_dir, lab_dir, dep):
        d.mkdir(parents=True)
    boxes = [(200.0 * i, 0.0, 200.0 * i + 100.0, 100.0) for i in range(n_gt)]
    for i in range(n_frames):
        # 恒定 `d = 25 m` ⇒ `t = exp(−β·25)`:β=0.02 → 0.61、β=0.10 → 0.082
        Image.fromarray(np.full((256, 256, 3), 100, np.uint8)).save(img_dir / f"{i:06d}.png")
        np.save(dep / f"{i:06d}.npy", np.full((256, 256), 25.0, np.float32))
        lab_dir.joinpath(f"{i:06d}.txt").write_text(
            "".join(f"Car 0 0 0 {a:.1f} {b:.1f} {c:.1f} {d:.1f} 1 1 1 1 1 1 1\n" for a, b, c, d in boxes)
        )
    return tmp_path, boxes


class TestBuildCurve:
    """端到端:注入 → 评测 → 曲线。**注入真的改变了下游读数**(否则这条测的是空转)。"""

    def test_brighter_image_loses_boxes_so_the_curve_is_monotone(self, tmp_path):
        """伪造的判据:**越亮 ⇒ 认出的框越少** —— 于是 fogdepth 曲线单调下降。

        ⚠️ 这里量的是**这条管线**(注入 → 评测 → 记录)接得对不对,**不是**模型行为。
        """
        root, boxes = _fake_root(tmp_path)
        work = tmp_path / "work"
        work.mkdir()

        from PIL import Image

        def predict(path):
            img = np.array(Image.open(path).convert("RGB"))
            mean = float(img.mean())
            keep = len(boxes) if mean <= 120 else (len(boxes) // 2 if mean <= 160 else 0)
            return [("Car", 0.9, b) for b in boxes[:keep]]

        res = C.build_curve(
            root,
            work,
            kind="fogdepth",
            levels=[0.02, 0.05, 0.10],
            predict=predict,
            conf=0.25,
            iou=0.5,
            n_points=101,
        )
        ds = [r["delta"] for r in res["curve"]]
        assert ds == sorted(ds, reverse=True), f"曲线应当单调不增,实测 {ds}"
        assert ds[0] > ds[-1], "最强档必须掉得更多(否则注入没生效)"
        assert res["n_points"] == 101
        # ★ 每个点都要带分辨率自述,否则读不出这个 Δ 可不可信
        assert all("res" in r and r["res"] for r in res["curve"]), "每个点都必须记 n_gt/n_tp/悬崖余量"
        assert res["base_res"]["Car"]["n_gt"] == 3 * 4

    def test_target_from_arms_is_a_difference(self, tmp_path):
        root, boxes = _fake_root(tmp_path, n_frames=2, n_gt=4)
        deg = tmp_path / "deg"
        from autodrivedata.edit import degrade as DG

        DG.degrade_root(root, deg, kind="fogdepth", level=0.10)

        def predict(path):
            return (
                [("Car", 0.9, b) for b in boxes]
                if "deg" not in str(path)
                else [("Car", 0.9, b) for b in boxes[:2]]
            )

        d = C.target_delta_from_arms(root, deg, predict=predict, conf=0.25, iou=0.5, n_points=11)
        assert d < 0, f"退化臂应当更低,实测 Δ={d}"

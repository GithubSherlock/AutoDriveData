"""`runtime/tb_export` 的回归钉:jsonl → TensorBoard event。

## 钉什么

两处**不会报错、只会让曲线说谎**的地方:

| 坑 | 错了会怎样 |
|---|---|
| **标签稀疏**(复核行没有 `loss`) | 固定取键 ⇒ KeyError;**缺的补 0** ⇒ 曲线上拉出**假台阶** |
| **多个 jsonl 合并成一个 run** | 两段不同的训练交替进同一条曲线,而**曲线看着正常** |

`step` 轴的取法也钉 —— 它是 runlog 的契约("`step` 就是迭代标识"),取错会让 x 轴错位
(而 TB 里 x 轴错位不像 JSON 那样一眼可见)。

## 怎么验

**写完读回来**(`EventAccumulator`):这是唯一能证明"TB 真能看到"的判据 ——
只断言"文件建出来了"在 writer API 变更时会假绿。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("tensorboard")

from autodrivedata.runtime import tb_export  # noqa: E402


def _write_jsonl(p: Path, rows: list[dict]) -> Path:
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _read_back(d: Path) -> dict[str, list]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(d))
    ea.Reload()
    return {t: ea.Scalars(t) for t in ea.Tags()["scalars"]}


class TestSparseTags:
    """★ 最要命的一条:早停复核行与 epoch 行**同 step 但键不同**。"""

    ROWS = [
        {"t": 135.1, "step": 1, "epoch": 1, "loss": 18.04, "cls": 0.039, "pts": 18.0, "lr": 1e-4},
        {"t": 264.7, "step": 2, "epoch": 2, "loss": 17.80, "cls": 0.047, "pts": 17.75, "lr": 1e-4},
        {"t": 900.2, "step": 20, "confirm_mAP": 0.31, "best_ap": 0.31},  # ← 没有 loss/lr
    ]

    def test_tags_are_per_row_not_a_fixed_list(self, tmp_path):
        rows = [tb_export.tags_of(r) for r in self.ROWS]
        assert set(rows[0]) == {"loss", "cls", "pts", "lr", "wall_s"}
        assert set(rows[2]) == {"confirm_mAP", "best_ap", "wall_s"}, "复核行不该凭空多出 loss"

    def test_missing_keys_are_absent_not_zero(self, tmp_path):
        """★ 缺的键**不许补 0** —— 补 0 会在曲线上拉出一条掉到 0 再弹回的假台阶。

        数值用 `approx`:TB 的 scalar **存 float32**(实测 `18.04` 读回 `18.040000915527344`)。
        对本项目够用 —— AP 量级 0.3 时 float32 仍有 ~1e-7 分辨率,远细于 2e-3 的复现性下限 ——
        但"图和 jsonl 逐位一致"这个假设**不成立**,别拿 TB 读数去比 1e-6。
        """
        src = _write_jsonl(tmp_path / "a.jsonl", self.ROWS)
        out = tmp_path / "tb"
        tb_export.export_one(src, "r", out, _FakeRl())
        scal = _read_back(out / "r")
        loss = [(s.step, s.value) for s in scal["loss"]]
        assert [p[0] for p in loss] == [1, 2], f"loss 只该有 2 个点,实测 {loss}"
        assert [p[1] for p in loss] == pytest.approx([18.04, 17.80])
        assert [(s.step, s.value) for s in scal["confirm_mAP"]] == [(20, pytest.approx(0.31))]
        assert [(s.step, s.value) for s in scal["best_ap"]] == [(20, pytest.approx(0.31))]

    def test_rows_without_a_step_are_counted_not_silently_dropped(self, tmp_path):
        """没有步标识的行**无处安放**,但必须**数出来**并上报 —— 静默丢 = "曲线少了一段"。"""
        rows = [*self.ROWS, {"loss": 1.0}]  # 无 step/epoch/frame/t
        src = _write_jsonl(tmp_path / "b.jsonl", rows)
        st = tb_export.export_one(src, "r", tmp_path / "tb", _FakeRl())
        assert st["n_skipped_no_step"] == 1 and st["n_rows"] == 3

    def test_empty_jsonl_is_not_an_error(self, tmp_path):
        """空 jsonl 是**合法产物**(那次跑没有逐迭代指标);整文件无步标识同理。"""
        src = tmp_path / "empty.jsonl"
        src.write_text("", encoding="utf-8")
        st = tb_export.export_one(src, "r", tmp_path / "tb", _FakeRl())
        assert st["n_rows"] == 0 and st["n_scalars"] == 0


class TestStepAxis:
    def test_step_key_wins(self):
        """`step` 是 runlog 契约里的迭代标识 ⇒ 优先于 `epoch`(两者通常同值,但契约优先)。"""
        assert tb_export.step_of({"step": 7, "epoch": 7, "t": 1.0}) == 7.0
        assert tb_export.step_of({"epoch": 7, "t": 1.0}) == 7.0  # 缺席时回落
        assert tb_export.step_of({"frame": 12, "t": 1.0}) == 12.0
        assert tb_export.step_of({"t": 3.5}) == 3.5
        assert tb_export.step_of({"loss": 1.0}) is None

    def test_wall_clock_is_kept_as_a_tag(self):
        """`t` 不做 x 轴,但**不丢** —— 它在 TB 里能看出"哪一段卡住了"。"""
        assert tb_export.tags_of({"step": 1, "t": 42.0})["wall_s"] == 42.0


class TestSpecParsing:
    def test_name_suffix(self, tmp_path):
        p = _write_jsonl(tmp_path / "run.jsonl", [])
        assert tb_export.parse_spec(f"{p}=myname") == (str(p), "myname")

    def test_bare_path(self, tmp_path):
        p = _write_jsonl(tmp_path / "run.jsonl", [])
        assert tb_export.parse_spec(str(p)) == (str(p), None)

    def test_bad_spec_reports_instead_of_silently_exporting_nothing(self):
        with pytest.raises(SystemExit, match="没有匹配到"):
            tb_export.expand("logs/definitely-not-here-*.jsonl")

    def test_glob_matches_become_separate_runs(self, tmp_path):
        """★ 一个名字配多个文件 ⇒ **必须拆成多个 run**,不许合并。

        合并的症状:两段不同的训练交替写进同一条曲线,x 轴还会因为两边 step 都从 1 开始
        而**互相覆盖** —— 曲线看着正常,结论全错。
        """
        for i in range(2):
            _write_jsonl(tmp_path / f"r{i}.jsonl", [])
        pairs = tb_export.expand(f"{tmp_path}/r*.jsonl=base")
        assert len(pairs) == 2 and len({run for _, run in pairs}) == 2, "两个文件必须两个 run 名"
        assert all(run.startswith("base__") for _, run in pairs)

    def test_single_hit_keeps_the_explicit_name(self, tmp_path):
        _write_jsonl(tmp_path / "only.jsonl", [])
        pairs = tb_export.expand(f"{tmp_path}/only.jsonl=base")
        assert pairs == [(tmp_path / "only.jsonl", "base")]


class TestMeta:
    def test_highlights_ride_along_with_the_curve(self, tmp_path):
        """结论数字(batch / 变体 / AP)必须与曲线同屏 —— 否则看图的人无从归属。"""
        _write_jsonl(tmp_path / "c.jsonl", [{"step": 1, "loss": 1.0}])
        (tmp_path / "c.json").write_text(
            json.dumps({"script": "train_maptr", "highlights": {"batch": 3, "variant": "mapqr"}}),
            encoding="utf-8",
        )
        meta = tb_export.sibling_meta(tmp_path / "c.jsonl")
        assert meta["highlights"] == {"batch": 3, "variant": "mapqr"}

    def test_missing_or_broken_sibling_json_is_not_fatal(self, tmp_path):
        p = _write_jsonl(tmp_path / "d.jsonl", [{"step": 1, "loss": 1.0}])
        assert tb_export.sibling_meta(p) == {}
        (tmp_path / "d.json").write_text("{ not json", encoding="utf-8")
        assert tb_export.sibling_meta(p) == {}


class _FakeRl:
    """`export_one` 只用到 `rl.metric`(逐 run 一行统计)—— 用桩把测试与 runlog 解耦。"""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def metric(self, step: int, **kv) -> None:
        self.rows.append({"step": step, **kv})

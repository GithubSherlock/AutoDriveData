"""运行日志三件套的契约(Plan2.md §P-M.13)。

**为什么值得单测**:这个模块的失效模式几乎全是**静默**的 ——
它不产出任何"结果",它产出的是"证据"。它坏掉时训练照跑、AP 照出,
只是**日志里的东西是错的或没有**:
- tee 把 `sys.stdout` 换掉却恢复成 `sys.__stdout__` ⇒ pytest 的 `capsys` 被永久破坏(测别的东西才炸);
- `_Tee` 不代理 `isatty`/`encoding` ⇒ 在管道里"恰好"不炸、在有 tty 的跑法里 `AttributeError`;
- env 名判据写成 `sys.prefix == sys.base_prefix` ⇒ **每份日志都标成 `base`**(本机实测踩到);
- 产物登记虚报存在 ⇒ "这份 AP 是哪份权重出的"重新变成不可判。

所以下面钉的不是"能写文件",而是**这几条判据本身**。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from autodrivedata.utils import runlog


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


def _paths(rl: runlog.RunLogger) -> tuple[Path, Path, Path]:
    """三件套路径(`enabled=False` 时才是 `None`,而本文件的用法都开着)。"""
    log_p, metrics_p, summary_p = rl.log_path, rl.metrics_path, rl.summary_path
    assert log_p is not None and metrics_p is not None and summary_p is not None
    return log_p, metrics_p, summary_p


def _read_json(rl: runlog.RunLogger) -> dict:
    return json.loads(_paths(rl)[2].read_text(encoding="utf-8"))


class TestTrio:
    """三件套:同 stem、同目录、都建出来。"""

    def test_three_files_share_one_stem(self, log_dir: Path):
        with runlog.run("autodrivedata.map.train_maptr", out_dir=log_dir) as rl:
            pass
        logs, jsonl, js = _paths(rl)
        assert logs.exists() and jsonl.exists() and js.exists()
        assert len({logs.stem, jsonl.stem, js.stem}) == 1
        assert logs.stem.startswith("map_train_maptr_"), "文件名须含「能力_模块_时间戳」"
        assert logs.parent == jsonl.parent == js.parent == log_dir

    def test_latest_symlink_points_at_this_run(self, log_dir: Path):
        with runlog.run("autodrivedata.map.eval_maptr", out_dir=log_dir) as rl:
            pass
        link = log_dir / "latest" / "map_eval_maptr.log"
        assert link.is_symlink()
        assert link.resolve() == _paths(rl)[0].resolve()

    def test_same_second_reruns_do_not_overwrite(self, log_dir: Path):
        """连跑 / 测试循环会撞同一秒 —— 撞了就**不许**互相覆盖(否则留痕断链)。"""
        with runlog.run("autodrivedata.map.eval_maptr", out_dir=log_dir) as a:
            pass
        with runlog.run("autodrivedata.map.eval_maptr", out_dir=log_dir) as b:
            pass
        pa, pb = _paths(a)[0], _paths(b)[0]
        assert pa != pb
        assert pa.exists() and pb.exists()


class TestHeader:
    def test_header_records_script_time_argv_cwd(self, log_dir: Path):
        import re

        with runlog.run("autodrivedata.map.train_maptr", out_dir=log_dir) as rl:
            pass
        text = _paths(rl)[0].read_text(encoding="utf-8")
        assert "autodrivedata.map.train_maptr" in text
        assert re.search(r"started\s+\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text), "缺年月日时分秒"
        assert "argv" in text and "cwd" in text

    def test_fingerprint_fields_exist(self, log_dir: Path):
        with runlog.run("autodrivedata.m.f", out_dir=log_dir) as rl:
            pass
        summary = _read_json(rl)
        assert {"git", "env", "gpu", "host"} <= set(summary)
        assert summary["env"]["python"]
        assert summary["argv"], "argv 必须原样落日志(复现一次跑法最直接的东西)"

    def test_fingerprint_sampled_once(self, log_dir: Path, monkeypatch):
        """`.json` 的 `gpu` 与 `.log` 头块必须是**同一份**快照。

        旧实现里 `_summary()` 重采一次指纹 ⇒ 同一次跑的 `used_mib` 头块写 1、`.json`
        写 12772(本进程**自己还活着**的分配),而 `.json["gpu"]` 会被当成
        "这次跑在什么机器上"来读。换机器正是靠这条证据判「显存变 ⇒ batch 变 ⇒
        旧 AP 不可比」—— 让它随时间漂移等于把证据毁掉。
        """
        n = 0

        def fake_fp():
            nonlocal n
            n += 1
            return {
                "git": {"rev": f"rev{n}"},
                "env": {"python": "3"},
                "gpu": {"name": f"GPU{n}"},
                "host": "h",
            }

        monkeypatch.setattr(runlog, "_fingerprint", fake_fp)
        with runlog.run("autodrivedata.m.f", out_dir=log_dir) as rl:
            pass
        assert n == 1, "指纹只许采一次(重采会让 .json 与头块各说各话)"
        assert _read_json(rl)["gpu"]["name"] == "GPU1"
        assert "GPU1" in _paths(rl)[0].read_text(encoding="utf-8"), "头块与 .json 必须是同一份"

    def test_env_name_comes_from_interpreter_path(self):
        """**根因钉**:判据不许写成 `sys.prefix == sys.base_prefix`。

        本项目 `autodrivedata` env 的真身在数据盘、`/root/miniconda3/envs/` 下只是符号链接,
        故它报 `base_prefix == prefix == .../envs/autodrivedata` —— 用"相等"判 base
        会把**每一份日志**都标成 `base`,而且不报错。改用路径里的 `envs/<名>` 段。
        """
        assert runlog._env_name("/root/miniconda3/envs/autodrivedata") == "autodrivedata"
        assert runlog._env_name("/root/miniconda3/envs/autolabel") == "autolabel"
        # hivt 未注册进 envs_dirs,prefix 是数据盘路径 —— 同一判据照样认得出
        assert runlog._env_name("/root/autodl-tmp/x/envs/hivt") == "hivt"
        # 真·base conda 安装:路径里没有 `envs` 段,且 prefix 就是 base_prefix
        assert runlog._env_name("/opt/miniconda3", base_prefix="/opt/miniconda3") == "base"
        # ★ 本机的坑就地记:本机 `sys.base_prefix` == `.../envs/autodrivedata`,
        #   所以"相等即 base"的老判据在这里**必然**把 env 读成 `base`。
        #   这一条把它钉住(判定用的是解释器路径,两条路都得对)。
        assert runlog._env_name(sys.prefix) != "" and runlog._env_name(sys.prefix) is not None

    def test_fingerprint_survives_missing_gpu_and_git(self, monkeypatch):
        """没 GPU / 不在 git 仓 ⇒ 取 `None`,**不许抛**(日志功能不许成为新的失败点)。"""
        monkeypatch.setattr(runlog, "_capture", lambda *a, **k: None)
        fp = runlog._fingerprint()
        assert fp["gpu"] is None
        assert fp["git"] == {"rev": None, "dirty": None, "dirty_files": None}


class TestTee:
    def test_print_inside_block_lands_in_stdout_and_file(self, log_dir: Path, capsys):
        with runlog.run("autodrivedata.m.f", out_dir=log_dir) as rl:
            print("标记串-Zx9 中文")
        assert "标记串-Zx9 中文" in capsys.readouterr().out, "脚本自己的 print 必须照常到终端"
        assert "标记串-Zx9 中文" in _paths(rl)[0].read_text(encoding="utf-8")

    def test_stdout_is_restored_to_the_exact_object(self, log_dir: Path):
        """恢复的是**替换时抓的那个对象**,不是 `sys.__stdout__`。

        写错的话:在 pytest `capsys` 下退出后 `sys.stdout` 变成真终端流 ⇒
        **后续所有测试的捕获全失效**(而本测试自己是绿的,极难归因)。
        """
        before_out, before_err = sys.stdout, sys.stderr
        with runlog.run("autodrivedata.m.f", out_dir=log_dir):
            assert isinstance(sys.stdout, runlog._Tee), "块内应已是 tee"
            assert sys.stdout is not before_out
        assert sys.stdout is before_out
        assert sys.stderr is before_err

    def test_tee_proxies_stream_attributes(self, log_dir: Path):
        """`isatty` / `encoding` / `fileno` 必须代理到内层流 —— tqdm、ultralytics 会问。"""
        with runlog.run("autodrivedata.m.f", out_dir=log_dir) as rl:
            tee, inner = sys.stdout, rl._stdout
            assert isinstance(tee, runlog._Tee) and inner is not None
            assert tee.isatty() == inner.isatty()
            assert tee.encoding == inner.encoding
            # 未知名照样抛 AttributeError(不是 `__getattr__` 把一切都吞成 None)
            assert not hasattr(tee, "no_such_attr_anywhere")

    def test_tee_detaches_instead_of_raising_when_log_file_dies(self, log_dir: Path):
        """盘满 / 日志被删 ⇒ 摘掉自己让运行继续,**不许**让每次 print 都抛。"""
        before = sys.stdout
        with runlog.run("autodrivedata.m.f", out_dir=log_dir):
            tee = sys.stdout
            assert isinstance(tee, runlog._Tee) and tee._fh is not None
            tee._fh.close()  # 模拟文件句柄失效
            print("这次 print 不许抛")
            assert sys.stdout is before, "写文件失败后应把自己从 sys.stdout 摘掉"


class TestRegistries:
    def test_metric_rows_are_jsonl_and_flushed(self, log_dir: Path):
        n, rows = 25, []
        with runlog.run("autodrivedata.map.train_maptr", out_dir=log_dir) as rl:
            for i in range(n):
                rl.metric(i, loss=1.0 / (i + 1), lr=1e-4)
            # 块内就读 —— 逐行 flush 的意义就是"中断也有已写出的行"
            rows = [json.loads(ln) for ln in _paths(rl)[1].read_text().splitlines()]
        assert len(rows) == n
        assert {"t", "step", "loss", "lr"} <= set(rows[0])
        assert rows[0]["step"] == 0 and rows[-1]["step"] == n - 1
        assert _read_json(rl)["metrics_count"] == n

    def test_artifact_hashes_real_files(self, log_dir: Path, tmp_path: Path):
        w = tmp_path / "w.pt"
        w.write_bytes(b"weights" * 100)
        with runlog.run("autodrivedata.m.f", out_dir=log_dir, tee=False) as rl:
            rl.artifact(w, "model")
        rec = _read_json(rl)["artifacts"][0]
        assert rec["kind"] == "model" and rec["bytes"] == 700
        assert rec["sha256"] and len(rec["sha256"]) == 12

    def test_missing_artifact_is_not_reported_as_produced(self, log_dir: Path, tmp_path: Path):
        """「这次没产出」必须与「产出了」可区分 —— 否则日志会替脚本吹牛。"""
        with runlog.run("autodrivedata.m.f", out_dir=log_dir, tee=False) as rl:
            rl.artifact(tmp_path / "never_written.pt", "model")
        summary = _read_json(rl)
        assert summary["artifacts"][0]["missing"] is True
        assert "sha256" not in summary["artifacts"][0]
        assert any("never_written" in n for n in summary["notes"])

    def test_artifact_dir_counts_files_and_bytes(self, log_dir: Path, tmp_path: Path):
        d = tmp_path / "frames"
        (d / "sub").mkdir(parents=True)
        (d / "a.json").write_bytes(b"x" * 10)
        (d / "sub" / "b.json").write_bytes(b"y" * 5)
        with runlog.run("autodrivedata.m.f", out_dir=log_dir, tee=False) as rl:
            rl.artifact_dir(d, "pred")
        rec = _read_json(rl)["artifacts"][0]
        assert (rec["n_files"], rec["bytes"]) == (2, 15)

    def test_input_resolves_against_cwd_not_project_root(self, log_dir: Path, tmp_path: Path, monkeypatch):
        """输入按 cwd 解析(与「读路径不锚定」口径一致)、产物按项目根 —— 两者不许混。"""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "in.json").write_text("{}", encoding="utf-8")
        with runlog.run("autodrivedata.m.f", out_dir=log_dir, tee=False) as rl:
            rl.input("in.json", "infos")
        assert _read_json(rl)["inputs"][0]["path"] == str(tmp_path / "in.json")

    def test_repeated_registration_of_one_file_stays_one_row(self, log_dir: Path, tmp_path: Path):
        """同一文件登两次 ⇒ `inputs`/`artifacts` 里**只有一行**。

        `train_maptr` 训练尾部调 `evaluate(rl=rl)`,`evaluate` 会再登一遍 infos/root ——
        旧实现把这些追加成 4 条(两条重复)。`inputs` 是"读了哪些文件"的**集合**,不是
        登记调用流水账:重复行会被读成"读了两遍 / 读的是另一个副本"。
        """
        f = tmp_path / "a.bin"
        f.write_bytes(b"x" * 3)
        with runlog.run("autodrivedata.m.f", out_dir=log_dir, tee=False) as rl:
            rl.input(f, "infos")
            rl.input(f, "infos")
            rl.input(f, "ckpt")  # 同路径不同 kind ⇒ 是另一条事实,不许并
            rl.artifact(f, "model")
            rl.artifact(f, "model")
        summary = _read_json(rl)
        assert [r["kind"] for r in summary["inputs"]] == ["infos", "ckpt"]
        assert len(summary["artifacts"]) == 1


class TestExitStatus:
    def test_exception_is_logged_reraised_and_marked_error(self, log_dir: Path):
        rl: runlog.RunLogger | None = None
        with pytest.raises(ValueError, match="boom"):
            with runlog.run("autodrivedata.m.f", out_dir=log_dir) as rl:
                raise ValueError("boom")
        assert rl is not None, "日志对象应在异常前已建好(否则这次退出连日志都没有)"
        text = _paths(rl)[0].read_text(encoding="utf-8")
        assert "Traceback" in text and "ValueError: boom" in text
        summary = _read_json(rl)
        assert summary["status"] == "error" and summary["exit_code"] == 1

    def test_system_exit_records_the_code(self, log_dir: Path):
        """过拟合闸门的 `raise SystemExit(1)` 必须在 `.json` 里留痕。

        旧形态下这次退出只活在终端最后一行,还被 `| tee` 的退出码吃掉 ⇒
        "这次是闸门判 FAIL 退出的"事后查不出来。
        """
        rl: runlog.RunLogger | None = None
        with pytest.raises(SystemExit):
            with runlog.run("autodrivedata.map.train_maptr", out_dir=log_dir) as rl:
                print("FAIL:损失卡在 2.314")
                raise SystemExit(1)
        assert rl is not None
        summary = _read_json(rl)
        assert (summary["status"], summary["exit_code"]) == ("failed", 1)
        assert "FAIL:损失卡在 2.314" in _paths(rl)[0].read_text(encoding="utf-8")

    def test_system_exit_zero_is_ok(self, log_dir: Path):
        rl: runlog.RunLogger | None = None
        with pytest.raises(SystemExit):
            with runlog.run("autodrivedata.m.f", out_dir=log_dir) as rl:
                raise SystemExit(0)
        assert rl is not None
        assert _read_json(rl)["status"] == "ok"

    def test_stop_is_idempotent(self, log_dir: Path):
        """`__exit__` 与 `atexit` 都会调 stop —— 二次调用不许把 `.json` 写坏/写两遍。"""
        with runlog.run("autodrivedata.m.f", out_dir=log_dir) as rl:
            rl.highlight("mAP", 0.3043)
        rl.stop()
        rl.stop()
        assert _read_json(rl)["highlights"]["mAP"] == 0.3043


class TestSwitches:
    def test_env_var_disables_everything(self, log_dir: Path, monkeypatch):
        monkeypatch.setenv("AUTODRIVEDATA_RUNLOG", "0")
        with runlog.run("autodrivedata.m.f", out_dir=log_dir) as rl:
            print("照样能打印")
            rl.metric(1, loss=0.5)  # no-op 不许抛
            rl.highlight("mAP", 1.0)
        assert not log_dir.exists(), "关掉时一个文件都不许建"

    def test_cli_flag_disables_everything(self, log_dir: Path, monkeypatch):
        """`--no-runlog` 只能认**原始 argv 字面量** —— 日志在 argparse 之前就开了。"""
        monkeypatch.setattr(sys, "argv", ["train_maptr.py", "--out", "x.pt", "--no-runlog"])
        with runlog.run("autodrivedata.m.f", out_dir=log_dir):
            pass
        assert not log_dir.exists()

    def test_cli_flag_absent_keeps_logging(self, log_dir: Path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train_maptr.py", "--out", "x.pt"])
        with runlog.run("autodrivedata.m.f", out_dir=log_dir) as rl:
            pass
        assert _paths(rl)[0].exists()

    def test_unwritable_dir_degrades_instead_of_raising(self, monkeypatch):
        """`logs/` 建不出来(只读盘)⇒ 退化成"只打印",不许把运行本身搞挂。"""
        monkeypatch.setattr(
            runlog.RunLogger, "_open", lambda self, *_a, **_k: setattr(self, "enabled", False)
        )
        with runlog.run("autodrivedata.m.f") as rl:
            print("仍然能跑")
            rl.metric(1, loss=0.5)
        assert rl.log_path is None


class TestEnvChain:
    """★ **会改变读数、却原本完全不进指纹的环境变量**(2026-10-07,§5 #7)。

    每一条都有实测后果:`CUDA_HOME` 不设 ⇒ torch 扩展缓存 hash 不命中 ⇒ **重编并静默覆盖
    规范 `.so`**(≈1 h);`PYTHONPATH` 被 `.bashrc` 塞进 ~90 条 ROS2 路径 ⇒
    **提交前闸门 `python -m pytest -q` 直接崩**,只好 `env -u PYTHONPATH` 绕。
    ⇒ 同一份代码在不同 shell 里**行为不同**,这个差必须留痕,否则"这数哪来的"重新变成不可判。
    """

    def test_every_listed_var_is_recorded(self, monkeypatch):
        from autodrivedata.utils import runlog

        monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda-11.8")
        monkeypatch.delenv("PYTHONPATH", raising=False)
        chain = runlog._env_chain()
        assert set(chain) == set(runlog.FINGERPRINT_ENV_VARS)
        # 设了的记结构,没设的记 None —— **"没设"与"设成空"必须可分**
        assert chain["CUDA_HOME"]["value"] == "/usr/local/cuda-11.8"
        assert chain["PYTHONPATH"] is None

    def test_it_records_the_entry_count_not_just_the_string(self, monkeypatch):
        """★ `PYTHONPATH` 的**病根是条目数**(~90 条 ROS2 路径),不是它的字面值。"""
        from autodrivedata.utils import runlog

        monkeypatch.setenv("PYTHONPATH", "/a:/b::/c")
        chain = runlog._env_chain()
        assert chain["PYTHONPATH"]["n_entries"] == 3, "空段不算条目"

    def test_long_values_are_truncated_but_still_identifiable(self, monkeypatch):
        """长路径串只留摘要 + 前 200 字符 —— 指纹不该把日志撑成几 MB。"""
        from autodrivedata.utils import runlog

        monkeypatch.setenv("LD_LIBRARY_PATH", "/x" * 5000)
        e = runlog._env_chain()["LD_LIBRARY_PATH"]
        assert len(e["value"]) <= 200 and len(e["sha1_12"]) == 12

    def test_the_two_vars_with_measured_consequences_are_listed(self):
        """耦合钉:`CUDA_HOME` 与 `PYTHONPATH` **必须**在表里 —— 这两条是本仓实测过的。"""
        from autodrivedata.utils import runlog

        assert {"CUDA_HOME", "PYTHONPATH"} <= set(runlog.FINGERPRINT_ENV_VARS)

    def test_chain_lands_in_the_summary(self, log_dir: Path):
        with runlog.run("autodrivedata.m.chain", out_dir=log_dir) as rl:
            pass
        assert "chain" in _read_json(rl)["env"]

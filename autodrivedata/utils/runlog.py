"""运行日志三件套 —— 训练/推理脚本的"跑过就留痕"基础设施(Plan2.md §P-M.13)。

**为什么需要**:包内全部可执行入口只 `print()` 到 stdout,零处 `import logging`。
留下来的"日志"是 shell 里临时 `| tee outputs/xxx.log` —— 跑过即无痕,模型与预测产物
无法回溯到「哪次跑、哪份参数、哪个代码版本、哪块 GPU」。换机器后显存变了会让
`auto_tune_batch_size` 选到不同 batch,**旧数字就不可比了**,而旧日志里没记 GPU。

**三件套**(同 stem,同目录,由 [`paths.project_path`](paths.py) 锚定到项目根的 `logs/`):
- `<stem>.log`   —— 全量文本:**tee `sys.stdout`/`sys.stderr`** ⇒ 脚本里既有的一百多处
  `print()` 一字不改就进日志。头块(脚本/时间/argv/cwd/git/python/GPU/host)+ 尾块
  (状态/耗时/产物表/highlights/notes)。
- `<stem>.jsonl` —— 逐迭代指标,一行一个 JSON(训练脚本用 `metric()`)。
  逐行 flush ⇒ **中断的跑法也有已写出的行**(长训防中断是常态)。
- `<stem>.json`  —— 汇总:环境指纹 + 输入/产物(**带 sha256**)+ highlights + 退出码。

**用法**(脚本 `main()` 只动三处,其余 `print()` 全部保留):

    from autodrivedata.utils import runlog

    def main() -> None:
        with runlog.run("autodrivedata.map.eval_maptr") as rl:
            ap = argparse.ArgumentParser()
            ...
            args = ap.parse_args()
            rl.input(args.ckpt, "ckpt")          # 登记输入(读路径,按 cwd 解析)
            ...
            rl.highlight("mAP", mAP)             # 登记结论数字

**两条接线纪律**:
- **输入按 cwd 解析、产物按项目根解析** —— 与 `paths.py` 的既有口径逐条对齐
  ("读路径不锚定,写盘路径锚定项目根")。日志里两者都记**绝对路径**,所以
  "这次到底读了哪个文件"永远可判。
- **开关**:默认总开。`AUTODRIVEDATA_RUNLOG=0` 关;`--no-runlog` 关。
  后者**只能扫 `sys.argv` 字面量** —— 日志在 argparse 之前就要开(否则参数报错无日志),
  那时 parse 结果还不存在。故 `--no-runlog` 必须在 `sys.argv` 里原样出现才认。

**为什么住在 `utils/`**:准入判据是「无项目领域语义、无 carla/torch 依赖」。
本模块只用 stdlib + `paths`(层守卫 `LAYER_RULES["utils"] = _PURE` 机械强制)。
GPU 指纹因此走 `nvidia-smi` 子进程而**不是** `torch.cuda` —— 顺带的好处是它在
没装 torch 的跑法(采集/标定)里同样可用。
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from hashlib import sha1, sha256
from pathlib import Path
from typing import Any, TextIO

from autodrivedata.utils.paths import PROJECT_ROOT, project_path

__all__ = ["RunLogger", "run", "start"]

_ENV_SWITCH = "AUTODRIVEDATA_RUNLOG"
_CLI_SWITCH = "--no-runlog"
_HASH_LIMIT = 512 * 1024 * 1024  # 超此大小只记字节数(权重/侧车 ~100–250 M,亚秒级)
_RULE = "=" * 100
_THIN = "-" * 100


# --------------------------------------------------------------------------- 环境指纹
def _capture(cmd: list[str], timeout: float = 10.0) -> str | None:
    """跑一条只读命令取 stdout;任何失败(缺可执行文件/超时/非 0 退出)一律 `None`。

    **指纹采集一律软失败**:在没装 git、没 GPU、命令挂住的机器上,
    "日志功能"绝不许变成"新的失败点"。
    """
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJECT_ROOT)
    except Exception:
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def _git_info() -> dict[str, Any]:
    """HEAD 短哈希 + 工作区是否脏。「改动未提交 ≠ 待办」是项目纪律,日志里带上
    就能让任何一份 AP/PSNR 数字回溯到确切的代码版本。"""
    rev = _capture(["git", "rev-parse", "--short", "HEAD"])
    if rev is None:
        return {"rev": None, "dirty": None, "dirty_files": None}
    porcelain = _capture(["git", "status", "--porcelain"]) or ""
    n = len([ln for ln in porcelain.splitlines() if ln.strip()])
    return {"rev": rev, "dirty": bool(n), "dirty_files": n}


def _env_name(prefix: str, base_prefix: str | None = None) -> str:
    """从解释器前缀反推 conda env 名。

    **不许用 `sys.prefix == sys.base_prefix` 判 base** —— 本项目实测会判错:
    `autodrivedata` env 的真身在数据盘、于 `/root/miniconda3/envs/` 下只是个符号链接,
    其解释器因此报 `base_prefix == prefix == .../envs/autodrivedata`,
    于是"是不是 base"的判据把它读成 base,**每份日志的环境指纹都错**,还不报错。
    改用路径里的 `envs/<名>` 段 —— 它同时覆盖 `hivt`(未注册进 `envs_dirs`,
    只能绝对路径调,prefix 是数据盘路径)与 base(路径里无 `envs`)。

    `base_prefix` 可注入:`== sys.base_prefix` 那条只对**当前解释器**有意义,
    可注入才测得了(否则用例只能跟着机器变绿变红)。
    """
    parts = Path(prefix).parts
    if "envs" in parts:
        idx = parts.index("envs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    base = sys.base_prefix if base_prefix is None else base_prefix
    return "base" if prefix == base else (Path(prefix).name or "unknown")


#: ★ **会改变读数、却又完全不进指纹的三条环境变量**(2026-10-07 补,§5 #7 收口)。
#: 每一条都有**本仓实测**的后果:
#:   - `CUDA_HOME` / `LD_LIBRARY_PATH` —— 不设时 torch 的扩展缓存 hash 不命中 ⇒
#:     **重编并静默覆盖规范 `.so`**(实测代价 ≈1 h;见 `gs/cuda_env` 头注)。
#:     这条原先只在 `train_3dgs_mini` 里单独补过,**其余 21 个入口都在裸奔**。
#:   - `PYTHONPATH` —— 本机 `.bashrc` 无条件 `source ros2_humble/install/setup.bash`,
#:     往每个新 shell 里塞 ~90 条路径;而 `env -u PYTHONPATH` 跑 pytest 才不崩
#:     (见 §5 #7)。⇒ **同一份代码在不同 shell 里行为不同**,这个差必须留痕。
#: 记**摘要**不记全文:路径串可能很长,而归属只需要"是哪一条链"。
FINGERPRINT_ENV_VARS = ("CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "TORCH_CUDA_ARCH_LIST")


def _env_chain() -> dict[str, Any]:
    """上表那几条变量的**值 + 条目数 + 摘要**(软失败:取不到就记 `None`,不影响运行)。"""
    out: dict[str, Any] = {}
    for k in FINGERPRINT_ENV_VARS:
        v = os.environ.get(k)
        if v is None:
            out[k] = None
            continue
        entry: dict[str, Any] = {"n_entries": len([p for p in v.split(os.pathsep) if p])}
        entry["sha1_12"] = sha1(v.encode()).hexdigest()[:12]
        # 短的直接记全文(便于一眼认出);长的只记条目数 + 摘要
        entry["value"] = v if len(v) <= 200 else v[:197] + "..."
        out[k] = entry
    return out


def _env_info() -> dict[str, Any]:
    """python 版本 + conda env 名 —— 3D 检测必须在 `autolabel` env 下跑,
    记下来能一眼看出跑错环境。

    `env` 取自**解释器路径**(权威),`env_activated` 另记 conda 的激活变量 ——
    两者不一致本身就是个信号(在 A env 下用 B env 的解释器跑,是容易犯又难查的错)。
    """
    return {
        "python": sys.version.split()[0],
        "env": _env_name(sys.prefix),
        "env_activated": os.environ.get("CONDA_DEFAULT_ENV"),
        "prefix": sys.prefix,
        "executable": sys.executable,
        # ★ 见 `FINGERPRINT_ENV_VARS` 的注释 —— 这四条会静默改变读数
        "chain": _env_chain(),
    }


_CUDA_RE = re.compile(r"CUDA Version:\s*([0-9.]+)")
_INT_RE = re.compile(r"(\d+)")


def _first_int(text: str) -> int | None:
    m = _INT_RE.search(text)
    return int(m.group(1)) if m else None


def _gpu_info() -> dict[str, Any] | None:
    """GPU 型号 / 显存 / 驱动 / CUDA(驱动口径)。

    `cuda` 取自 `nvidia-smi` 头部 = **驱动支持的最高 CUDA**,与 `torch.version.cuda`
    (torch 编译期版本)不是一回事 —— 本层禁 torch,故只能取前者,如实标为驱动口径。
    """
    q = _capture(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader",
            "-i",
            "0",
        ]
    )
    if not q:
        return None
    parts = [p.strip() for p in q.split(",")]
    if len(parts) < 4:
        return None
    total, used = _first_int(parts[1]), _first_int(parts[2])
    m = _CUDA_RE.search(_capture(["nvidia-smi", "-i", "0"]) or "")
    return {
        "name": parts[0],
        "total_mib": total,
        "used_mib": used,
        "free_mib": None if total is None or used is None else total - used,
        "driver": parts[3],
        "cuda": m.group(1) if m else None,
    }


def _upsert(rows: list[dict[str, Any]], rec: dict[str, Any]) -> None:
    """按 (kind, path) 去重登记,重复的**后登记者覆盖**(它带着落盘后的真实大小/哈希)。

    为什么要去重:`inputs` 是"这次跑读了哪些文件"这个**集合**,不是登记调用的流水账。
    `train_maptr` 尾部调 `evaluate(rl=rl)` 会让 infos/root 各被登记两次 —— 读的人看到
    同名两条只会怀疑"是不是读了两遍/读的是不同副本",而它们其实是同一条事实。
    """
    key = (rec.get("kind"), rec.get("path"))
    for i, old in enumerate(rows):
        if (old.get("kind"), old.get("path")) == key:
            rows[i] = rec
            return
    rows.append(rec)


def _fingerprint() -> dict[str, Any]:
    """全量指纹。逐项软失败 —— 单条采集炸了不影响其余,更不影响运行本身。"""
    out: dict[str, Any] = {}
    for key, fn in (("git", _git_info), ("env", _env_info), ("gpu", _gpu_info)):
        try:
            out[key] = fn()
        except Exception:
            out[key] = None
    try:
        import platform

        out["host"] = platform.node()
        out["platform"] = platform.platform()
    except Exception:
        out["host"] = out["platform"] = None
    return out


def _rel(path: Path) -> str:
    """项目内 → 仓库根相对路径;项目外 → 绝对路径(照实记,不假装在项目里)。"""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _digest(path: Path) -> dict[str, Any]:
    """(bytes, sha256 前 12 位)。超 `_HASH_LIMIT` 只给字节数并注明。

    **权重必须带哈希**:`--out` 同名覆盖是常态,"这份 AP 是哪份权重出的"、
    "这个模型还是上次那个吗" 没有哈希就不可判定。
    """
    info: dict[str, Any] = {"bytes": None, "sha256": None, "note": None}
    try:
        size = path.stat().st_size
    except OSError as exc:
        info["note"] = f"stat 失败: {exc}"
        return info
    info["bytes"] = size
    if size > _HASH_LIMIT:
        info["note"] = f"size {size} > 上限 {_HASH_LIMIT} ⇒ 未哈希"
        return info
    h = sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError as exc:
        info["note"] = f"读取失败: {exc}"
        return info
    info["sha256"] = h.hexdigest()[:12]
    return info


def _dir_stats(d: Path) -> dict[str, Any]:
    """目录:文件数 + 总字节(逐帧契约 / 拼图目录动辄上千文件,不逐个哈希)。"""
    n = total = 0
    for f in d.rglob("*"):
        if f.is_file():
            n += 1
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return {"n_files": n, "bytes": total}


# --------------------------------------------------------------------------- tee
class _Tee:
    """把写往 `stream` 的一切复制一份到 `fh`。

    **`__getattr__` 全量代理不是可选项**:tqdm / ultralytics / rich 之类会问
    `sys.stdout.isatty()` / `.fileno()` / `.encoding`。只实现 `write`/`flush` 的包装
    在有 tty 的跑法里 `AttributeError`,而在管道里"恰好"不炸 ⇒ 典型的环境相关偶发。

    **盘满 / 日志被删不许拖垮运行**:写文件失败就把自己从 `sys.stdout` 上摘下来
    (原流照常工作),而不是让每一次 `print()` 都抛 `OSError` 打断 3 小时训练。
    """

    def __init__(self, stream: TextIO, fh: TextIO) -> None:
        self._stream = stream
        self._fh: TextIO | None = fh

    def write(self, s: str) -> int:
        fh = self._fh
        if fh is not None:
            try:
                fh.write(s)
            except (OSError, ValueError):
                self._detach()
        return self._stream.write(s)

    def flush(self) -> None:
        fh = self._fh
        if fh is not None:
            try:
                fh.flush()
            except (OSError, ValueError):
                self._detach()
        self._stream.flush()

    def _detach(self) -> None:
        """日志文件死了:摘掉 tee,运行继续(日志功能的失败不许升级成运行的失败)。"""
        self._fh = None
        if sys.stdout is self:
            sys.stdout = self._stream
        if sys.stderr is self:
            sys.stderr = self._stream

    def __getattr__(self, name: str) -> Any:
        # `_` 开头直接拒 —— 否则 `self._stream` 未赋值时 `__getattr__("_stream")`
        # 会递归到爆栈(这正是"只剩 write/flush 就够"的写法踩的坑)。
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._stream, name)


# --------------------------------------------------------------------------- 主体
class RunLogger:
    """一次运行的日志三件套。经 `run()` / `start()` 构造,不要直接 `RunLogger(...)`。"""

    def __init__(
        self,
        script: str,
        *,
        out_dir: str | Path = "logs",
        enabled: bool = True,
        tee: bool = True,
    ) -> None:
        self.script = script
        self.enabled = enabled
        self.when = datetime.now()
        # 先给默认值:`start()` + `atexit` 路径没有 `__exit__`,`_summary()` 照样要能读
        self.status = "ok"
        self.exit_code = 0
        self._fp: dict[str, Any] | None = None  # 起跑指纹(`_header()` 采一次,`_summary()` 复用)
        self._t0 = time.perf_counter()
        self._inputs: list[dict[str, Any]] = []
        self._artifacts: list[dict[str, Any]] = []
        self._highlights: dict[str, Any] = {}
        self._notes: list[str] = []
        self._n_metrics = 0
        self._stopped = False
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None
        self._fh: TextIO | None = None
        self._mh: TextIO | None = None
        self.dir: Path | None = None
        self.log_path: Path | None = None
        self.metrics_path: Path | None = None
        self.summary_path: Path | None = None
        if self.enabled:
            self._open(out_dir, tee)

    # ------------------------------------------------------------------ 开/关
    def _open(self, out_dir: str | Path, tee: bool) -> None:
        try:
            self.dir = project_path(out_dir)
            self.dir.mkdir(parents=True, exist_ok=True)
            stem = self._unique(_stem_of(self.script, self.when))
            self.log_path = self.dir / f"{stem}.log"
            self.metrics_path = self.dir / f"{stem}.jsonl"
            self.summary_path = self.dir / f"{stem}.json"
            self._fh = self.log_path.open("w", encoding="utf-8")
            self._mh = self.metrics_path.open("w", encoding="utf-8")
            self._link_latest(stem)
        except OSError:
            # logs/ 建不出来(只读盘、权限)⇒ 退化成"只打印",不抛
            self.enabled = False
            self._fh = self._mh = None
            return
        if tee:
            # 抓的是**替换时的那个对象**,不是 `sys.__stdout__` ——
            # pytest 的 capsys / 外层重定向会把 sys.stdout 换掉,写错就把外层捕获永久破坏。
            out, err, fh = sys.stdout, sys.stderr, self._fh
            self._stdout, self._stderr = out, err
            sys.stdout = _Tee(out, fh)
            sys.stderr = _Tee(err, fh)
        self._header()
        atexit.register(self.stop)

    def _unique(self, base: str) -> str:
        """同一秒内二次运行不许互相覆盖(测试与连跑都会撞)。"""
        assert self.dir is not None
        stem, i = base, 1
        while (self.dir / f"{stem}.log").exists():
            stem = f"{base}-{i}"
            i += 1
        return stem

    def _link_latest(self, stem: str) -> None:
        """`logs/latest/<能力_模块>.<ext>` 相对软链 → 最新一次,方便 `tail -f` 上一跑。"""
        assert self.dir is not None
        tag = stem[: stem.rfind("_")]
        latest = self.dir / "latest"
        try:
            latest.mkdir(parents=True, exist_ok=True)
            for ext in ("log", "jsonl", "json"):
                link = latest / f"{tag}.{ext}"
                link.unlink(missing_ok=True)
                os.symlink(f"../{stem}.{ext}", link)
        except OSError as exc:
            self._notes.append(f"latest 软链建立失败(fs 不支持?){exc}")

    def _header(self) -> None:
        # 指纹**只采一次**,存下来给 `_summary()` 复用 —— 头块与 `.json` 必须说同一句话。
        # (曾经在 `_summary()` 里重采:同一次跑的 `used_mib` 头块写 1、`.json` 写 2504,
        #  而 `.json` 的 `gpu` 会被当成"这次跑在什么机器上"来读 —— 重采等于让这个键
        #  随时间漂移,且漂移的正是"换机器 ⇒ batch 变 ⇒ AP 不可比"最需要的那条证据。)
        fp = self._fp = _fingerprint()
        git, env, gpu = fp.get("git") or {}, fp.get("env") or {}, fp.get("gpu") or {}
        dirty = "clean" if git.get("dirty") is False else f"dirty: {git.get('dirty_files')} files"
        gpu_line = "无(nvidia-smi 不可用 / 无 GPU)"
        if gpu:
            gpu_line = (
                f"{gpu.get('name')} | {gpu.get('total_mib')} MiB "
                f"(used {gpu.get('used_mib')}) | CUDA {gpu.get('cuda')} | driver {gpu.get('driver')}"
            )
        self._raw(
            "\n".join(
                [
                    _RULE,
                    f"run       {self.script}",
                    f"started   {self.when:%Y-%m-%d %H:%M:%S}",
                    _RULE,
                    f"argv      {' '.join(sys.argv)}",
                    f"cwd       {os.getcwd()}",
                    f"git       {git.get('rev')} ({dirty})",
                    f"python    {env.get('python')} | env={env.get('env')} | {env.get('prefix')}",
                    f"gpu       {gpu_line}",
                    f"host      {fp.get('host')}",
                    _THIN,
                    "",
                ]
            )
        )

    # ------------------------------------------------------------------ 登记
    def input(self, path: str | Path, kind: str = "input") -> None:
        """登记**输入**。按 cwd 解析(与"读路径不锚定"口径一致),记绝对路径。

        输入可能不在项目内(如 `/root/miniconda3/envs/.../weights/*.pth`),故不做
        "必须落在项目内"的断言 —— 照实记它到底是哪个文件。
        """
        p = Path(path)
        abs_p = p if p.is_absolute() else Path.cwd() / p
        rec: dict[str, Any] = {"kind": kind, "path": str(abs_p), "exists": abs_p.exists()}
        if abs_p.exists() and abs_p.is_file():
            rec.update(_digest(abs_p))
        _upsert(self._inputs, rec)

    def artifact(self, path: str | Path, kind: str = "artifact") -> None:
        """登记**产物**。按项目根解析(与 `project_path` 口径一致)。

        不存在的路径**不许虚报**:记 `missing: True` + 一条 note。
        (脚本常在某个分支里没落盘 —— 那种"这次没产出"必须与"产出了"可区分。)
        """
        p = project_path(path)
        if p.is_dir():
            self.artifact_dir(p, kind)
            return
        if not p.is_file():
            _upsert(self._artifacts, {"kind": kind, "path": _rel(p), "missing": True})
            self._notes.append(f"产物不存在(未落盘?){_rel(p)}")
            return
        rec: dict[str, Any] = {"kind": kind, "path": _rel(p), "missing": False}
        rec.update(_digest(p))
        _upsert(self._artifacts, rec)

    def artifact_dir(self, path: str | Path, kind: str = "dir") -> None:
        """登记**目录**产物(逐帧契约 / 拼图):文件数 + 总字节,不逐个哈希。"""
        p = project_path(path)
        if not p.is_dir():
            self._artifacts.append({"kind": kind, "path": _rel(p), "missing": True})
            self._notes.append(f"产物目录不存在(未落盘?){_rel(p)}")
            return
        self._artifacts.append({"kind": kind, "path": _rel(p), "missing": False, **_dir_stats(p)})

    def metric(self, step: int, **kv: Any) -> None:
        """逐迭代指标 → `.jsonl` 一行(逐行 flush ⇒ 中断也有已写出的行)。"""
        self._n_metrics += 1
        if self._mh is None:
            return
        row = {"t": round(time.perf_counter() - self._t0, 3), "step": step, **kv}
        try:
            self._mh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._mh.flush()
        except (OSError, ValueError):
            self._mh = None

    def highlight(self, key: str, value: Any) -> None:
        """登记**结论数字**(mAP / PSNR / ATE / final loss)。

        这是让日志"可扫"的那一层:不必读 400 行也能知道这次跑出了什么。
        """
        self._highlights[key] = value
        print(f"[runlog] {key} = {value}", flush=True)

    def note(self, text: str) -> None:
        self._notes.append(text)
        print(f"[runlog] note: {text}", flush=True)

    # ------------------------------------------------------------------ 收尾
    def stop(self) -> None:
        """幂等收尾:尾块 + `.json` + 摘 tee。`__exit__` 与 `atexit` 都会调。"""
        if self._stopped:
            return
        self._stopped = True
        if self.enabled and self._fh is not None:
            self._footer()
        self._close()

    def _footer(self) -> None:
        elapsed = time.perf_counter() - self._t0
        lines = [_THIN, f"finished  {datetime.now():%Y-%m-%d %H:%M:%S}  (elapsed {elapsed:.1f}s)"]
        for label, items in (("inputs", self._inputs), ("artifacts", self._artifacts)):
            if items:
                lines.append(f"{label}:")
                for rec in items:
                    size = rec.get("bytes")
                    bits = [f"  {rec.get('kind'):8s} {rec.get('path')}"]
                    if size is not None:
                        bits.append(f"{size / 1e6:.1f} MB")
                    if rec.get("sha256"):
                        bits.append(f"sha256:{rec['sha256']}")
                    if rec.get("missing"):
                        bits.append("[缺失]")
                    lines.append(" ".join(bits))
        if self._highlights:
            lines.append("highlights:")
            lines += [f"  {k} = {v}" for k, v in self._highlights.items()]
        if self._notes:
            lines.append("notes:")
            lines += [f"  - {n}" for n in self._notes]
        self._raw("\n".join(lines) + "\n" + _RULE + "\n")

    def _close(self) -> None:
        for stream, attr in ((self._stdout, "stdout"), (self._stderr, "stderr")):
            if stream is not None and getattr(sys, attr) is not stream:
                setattr(sys, attr, stream)
        if self.enabled and self.summary_path is not None:
            try:
                self.summary_path.write_text(
                    json.dumps(self._summary(), indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except OSError:
                pass
        for fh in (self._fh, self._mh):
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
        self._fh = self._mh = None

    def _summary(self) -> dict[str, Any]:
        # 用头块那份指纹(不重采);**另外**补一份收尾快照 —— 退出时的
        # `used_mib` 就是"这次跑有没有把显存还回去"的证据(本项目红线:
        # 停训练必须 `nvidia-smi` 归零,孤儿 worker 会继续占显存)。
        fp = self._fp or _fingerprint()
        return {
            "script": self.script,
            "started": f"{self.when:%Y-%m-%dT%H:%M:%S}",
            "finished": f"{datetime.now():%Y-%m-%dT%H:%M:%S}",
            "elapsed_s": round(time.perf_counter() - self._t0, 1),
            "status": self.status,
            "exit_code": self.exit_code,
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "git": fp.get("git"),
            "env": fp.get("env"),
            # 与 .log 头块**同一份**快照。曾经在这里重采一次 ⇒ 同一次跑的 used_mib
            # 头块写 1、`.json` 写 12772,而 `.json["gpu"]` 会被当成"这次跑在什么机器上"
            # 读 —— 换机器正是靠这条证据判「显存变 ⇒ batch 变 ⇒ 旧 AP 不可比」。
            #
            # **不另记收尾显存**:`stop()` 在本进程内跑,采到的必然含本进程**自己还活着**的
            # 分配(实测 5 epoch 冒烟报 12772 MiB)。红线讲的"停训练须 nvidia-smi 归零"看的是
            # **进程消失后**的机器,这里采不到那个时刻 —— 记一个 12 GB 只会被读成"泄漏了"。
            "gpu": fp.get("gpu"),
            "host": fp.get("host"),
            "platform": fp.get("platform"),
            "log": None if self.log_path is None else _rel(self.log_path),
            "metrics": None if self.metrics_path is None else _rel(self.metrics_path),
            "metrics_count": self._n_metrics,
            "inputs": self._inputs,
            "artifacts": self._artifacts,
            "highlights": self._highlights,
            "notes": self._notes,
        }

    # ------------------------------------------------------------------ 写
    def _raw(self, text: str) -> None:
        """只写日志文件(不走 stdout)—— 头/尾块不必在终端再刷一遍。"""
        if self._fh is None:
            return
        try:
            self._fh.write(text)
            self._fh.flush()
        except (OSError, ValueError):
            self._fh = None

    # ------------------------------------------------------------------ 上下文
    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        self.exit_code, self.status = 0, "ok"
        if isinstance(exc, SystemExit):
            code = exc.code
            self.exit_code = 0 if code is None else (code if isinstance(code, int) else 1)
            self.status = "ok" if self.exit_code == 0 else "failed"
        elif exc_type is not None:
            self.exit_code, self.status = 1, "error"
        if exc is not None:
            # 完整 traceback **只进文件**:终端那边 Python 自己的 excepthook 还会再打一次,
            # 两边都打就是双份。终端只留一行摘要。
            # `SystemExit` 不印栈 —— 它是"正常收场"信号(argparse 的 `--help`/参数报错、
            # 闸门 `raise SystemExit(1)`),Python 自己也不印,印出来只是噪声。
            if self._fh is not None and not isinstance(exc, SystemExit):
                try:
                    self._fh.write("\n")
                    traceback.print_exception(exc_type, exc, tb, file=self._fh)
                    self._fh.flush()
                except (OSError, ValueError):
                    pass
            name = "异常" if exc_type is None else exc_type.__name__
            print(f"[runlog] {self.status}(exit {self.exit_code}): {name}: {exc}", flush=True)
        self.stop()
        return False  # 异常照常抛出,不吞


# --------------------------------------------------------------------------- 入口
def _stem_of(script: str, when: datetime) -> str:
    """`autodrivedata.map.train_maptr` → `map_train_maptr_20260927-013045`。"""
    parts = [p for p in script.split(".") if p and p != "autodrivedata"]
    tag = "_".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "run")
    return f"{tag}_{when:%Y%m%d-%H%M%S}"


def _disabled_by_env() -> bool:
    return os.environ.get(_ENV_SWITCH, "1").strip().lower() in ("0", "false", "no", "off")


def run(script: str, out_dir: str | Path = "logs", *, tee: bool = True) -> RunLogger:
    """上下文管理器入口 —— 起日志 + tee stdout。见模块头注用法。

    `--no-runlog` 与 `AUTODRIVEDATA_RUNLOG=0` 任一命中 ⇒ 返回**不落盘**的 logger
    (方法照常可调,全部 no-op),故调用侧无需分支。
    """
    enabled = not _disabled_by_env() and _CLI_SWITCH not in sys.argv
    return RunLogger(script, out_dir=out_dir, enabled=enabled, tee=tee)


def start(script: str, out_dir: str | Path = "logs", *, tee: bool = True) -> RunLogger:
    """非上下文用法(需自己 `stop()`);`atexit` 已兜底,忘调也不会丢 `.json`。"""
    return run(script, out_dir, tee=tee)

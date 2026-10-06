"""`gs/cuda_env` —— **"必须在 import gsplat 之前做完那两件事"**的机械判据。

## 为什么这条要写成 AST 判据,而不是"记得写对顺序"

那段逻辑(设 `TORCH_CUDA_ARCH_LIST`、检查 `CUDA_HOME`)原先只长在 `train_3dgs_mini` 里,
靠的是**两行 import 的先后**。2026-10-04 新增三个同样要 gsplat 的入口时,按原样放
`import gsplat` —— 结果 **`ruff format --fix`(isort) 把它挪到了 first-party 的
`autodrivedata.*` 之前**,`cuda_env` 还没跑、gsplat 就先进来了。**格式化器悄悄毁掉一条不变量**,
而症状只在"架构不对/扩展被重编"时才露头,且报错点完全指不到这里。

⇒ 改成 `cuda_env.ensure_gsplat()`:**顺序由函数保证,不靠文件排布**。本判据守住它。

## 坏掉的样子(两种都不报错)

| 写法 | 症状 |
|---|---|
| 顶层裸 `import gsplat` 排在 `cuda_env` 前 | `ValueError: Unknown CUDA arch (10.3)` —— 报在 **gsplat 内部**,看不出是环境变量 |
| 顶层裸 `import gsplat` 且没设 `CUDA_HOME` | torch 按 flags 的 hash 找不到缓存 ⇒ **重编并静默覆盖规范 `.so`**(实测代价 ≈1 h) |
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from autodrivedata.gs import cuda_env

GS_DIR = Path(cuda_env.__file__).parent
#: 唯一允许直接 import gsplat 的模块(而且它也只在 `ensure_gsplat()` 里懒 import)。
EXEMPT = {"cuda_env.py"}


def _gs_modules() -> list[Path]:
    return sorted(p for p in GS_DIR.glob("*.py") if p.name != "__init__.py")


def _toplevel_gsplat_imports(tree: ast.Module) -> list[int]:
    """顶层(非函数体)里对 gsplat 的 import 行号。函数内的懒 import 不算。"""
    hits = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "gsplat" for a in node.names):
            hits.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "gsplat":
            hits.append(node.lineno)
    return hits


class TestNoBareGsplatImport:
    def test_only_cuda_env_may_import_gsplat_at_toplevel(self):
        """★ 全包扫一遍 —— 新加一个 `gs/` 入口时,这条会替人记住顺序。"""
        offenders = {}
        for p in _gs_modules():
            if p.name in EXEMPT:
                continue
            tree = ast.parse(p.read_text(encoding="utf-8"))
            lines = _toplevel_gsplat_imports(tree)
            if lines:
                offenders[p.name] = lines
        assert not offenders, (
            f"顶层裸 import gsplat(会被 isort 挪到 cuda_env 之前):{offenders} —— "
            "改走 `cuda_env.ensure_gsplat()`"
        )

    def test_every_module_that_uses_gsplat_pulls_in_cuda_env(self):
        """凡**运行时**碰 gsplat 的模块,都必须让 `cuda_env` 先跑过。

        没有这条的话,「删掉 `ensure_gsplat()`、改成函数内懒 `import gsplat`」能绕过上面
        那条顶层检查 —— 而它同样是错的顺序。

        判定宽松一档是**有意**的:`train_3dgs_mini` 自己不经手 gsplat(渲染走
        `render_gs.rasterize`,`gsplat.strategy` 是 `main()` 里的懒 import),它只要
        import 了 `cuda_env` 就够 —— 而那个 import 是它文件里的第一条。
        """
        missing = []
        for p in _gs_modules():
            if p.name in EXEMPT:
                continue
            src = p.read_text(encoding="utf-8")
            if "gsplat." in src and "cuda_env" not in src:
                missing.append(p.name)
        assert not missing, f"碰了 gsplat 却连 `cuda_env` 都没 import:{missing}"


class TestEnsureGsplat:
    def test_returns_the_module(self):
        g = cuda_env.ensure_gsplat()
        assert hasattr(g, "rasterization"), "拿回来的不是 gsplat"

    def test_is_idempotent(self):
        assert cuda_env.ensure_gsplat() is cuda_env.ensure_gsplat()


class TestNotes:
    def test_arch_and_cuda_notes_are_both_present(self):
        """两个 NOTE 是**归属信息** —— 跨 build 的数不许混比,而那要求事后查得出来。"""
        assert cuda_env.ARCH_NOTE
        assert cuda_env.CUDA_NOTE

    def test_note_names_the_culprit_when_cuda_home_is_absent(self, monkeypatch):
        """没给 `CUDA_HOME` 时,那条 NOTE 必须点名"会用哪个 nvcc" —— 只报"环境不对"没用。"""
        monkeypatch.delenv("CUDA_HOME", raising=False)
        monkeypatch.delenv("CUDA_PATH", raising=False)
        import importlib

        m = importlib.reload(cuda_env)
        try:
            assert "CUDA_HOME" in m.CUDA_NOTE
            assert "nvcc" in m.CUDA_NOTE
        finally:
            importlib.reload(cuda_env)

    def test_note_records_the_home_when_set(self, monkeypatch):
        monkeypatch.setenv("CUDA_HOME", "/opt/fake-cuda")
        import importlib

        m = importlib.reload(cuda_env)
        try:
            assert "/opt/fake-cuda" in m.CUDA_NOTE
            assert m.home() == "/opt/fake-cuda"
        finally:
            importlib.reload(cuda_env)

    def test_note_is_printable_two_lines(self):
        lines = cuda_env.note().splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("[arch]")
        assert lines[1].startswith("[cuda]")


class TestArchHandling:
    def test_arch_list_usable_rejects_unknown_arch(self):
        """★ 直接问那个会抛错的函数 —— 不维护"已知架构表"(它会随 torch 版本漂)。"""
        assert not cuda_env._arch_list_usable("10.3")
        assert cuda_env._arch_list_usable("8.6")

    def test_probe_restores_the_previous_value(self, monkeypatch):
        """探针**不许留下副作用** —— 它只是问一句,不是设置。"""
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.6")
        cuda_env._arch_list_usable("10.3")
        import os

        assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.6"
        monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
        cuda_env._arch_list_usable("8.6")
        assert "TORCH_CUDA_ARCH_LIST" not in os.environ

    def test_arch_list_usable_is_not_reimplemented(self):
        """`train_3dgs_mini` 里那份已搬走 —— 再长回去就是两套。"""
        from autodrivedata.gs import train_3dgs_mini as T

        assert "def _arch_list_usable" not in inspect.getsource(T)
        assert callable(cuda_env._arch_list_usable)


@pytest.mark.parametrize("name", sorted(EXEMPT))
def test_exempt_modules_really_exist(name):
    """豁免名单不许是死链 —— 名字改了而名单没改,这条守卫会静默放行一切。"""
    assert (GS_DIR / name).is_file()

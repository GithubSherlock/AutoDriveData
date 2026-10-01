"""**下游仓库**对本包的引用必须可解析 —— 重构 checklist 缺失的第 12 类引用形态。

## 为什么有这条(不是假想,是 2026-09-30 实测)

`mapvec_pred/1` 是**复制而非 import** 的跨仓契约(红线:AutoLabel 不反向 import 本包)。
为保证复制版不漂移,AutoLabel 侧写了 `test_mapvec_crosscheck.py` **逐函数比对原实现** ——
那是这条契约**唯一的**漂移保护网。

2026-09-26 本包重构把模块收进能力子目录(`chamfer_ap.py` → `map/chamfer_ap.py` 等),
**AutoLabel 那边当场 `ModuleNotFoundError`,保护网断了 4 天,两边都不知道**:

    ModuleNotFoundError: No module named 'autodrivedata.chamfer_ap'
    Interrupted: 1 error during collection

根因是**覆盖面**:[CLAUDE.md](../../CLAUDE.md) 记的 11 类引用形态、
[docs/refactor-2026-09.md](../../docs/refactor-2026-09.md) 的扫描清单、
[test_docs.py](test_docs.py) 的坏链判据 —— **全部只覆盖本仓**。
"改了自己的目录"与"下游还指着旧路径"之间没有任何机制。

## 判据

AST 扫下游仓库的 `.py`,取所有 `import autodrivedata.X` / `from autodrivedata.X import …`,
逐个 `importlib.util.find_spec`。**只判可解析,不判语义** —— 语义由下游自己的交叉验证负责
(那条现在也活了)。

## 有意不覆盖的形态(如实记)

下游还有**路径字符串**引用(如 `finetune_config.py` 里写死的
`.../AutoDriveData/outputs/kitti_ft/`)。本测试**不查**它们:那属**数据**可用性,
不是代码引用;写进单测会让测试依赖数据集在不在盘上。
⇒ 这类要另判,别以为这里绿了就代表全部。
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]  # 项目根

#: 下游仓库(消费方)。新增消费方时在这里加一行 —— 判据会自动覆盖。
_DOWNSTREAMS: dict[str, Path] = {
    "AutoLabel": _ROOT.parent / "AutoLabel",
}

#: 扫描时跳过的目录名(第三方/缓存)。
_SKIP_DIRS = frozenset({"__pycache__", ".git", ".ipynb_checkpoints", "weights", "node_modules"})


def _pkg_refs(src: str) -> set[str]:
    """源码里所有引用本包的**模块路径**(`autodrivedata` 及其子模块)。"""
    out: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names if a.name.split(".")[0] == "autodrivedata")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] == "autodrivedata":
                out.add(node.module)
    return out


def _scan(root: Path) -> dict[Path, set[str]]:
    found: dict[Path, set[str]] = {}
    for p in root.rglob("*.py"):
        if _SKIP_DIRS & set(p.parts):
            continue
        try:
            refs = _pkg_refs(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue  # 坏文件不参与(它自己会以别的方式暴露)
        if refs:
            found[p] = refs
    return found


@pytest.mark.parametrize("name", sorted(_DOWNSTREAMS))
def test_downstream_imports_resolve(name: str):
    root = _DOWNSTREAMS[name]
    if not root.is_dir():
        pytest.skip(f"本机无下游仓库 {name}({root}),跳过")

    found = _scan(root)
    # **自证**:一个引用都没扫到 ⇒ 判据是死的(目录改了 / 仓库清空了),不是"全通过"
    assert found, (
        f"{name} 里一条 `autodrivedata.*` 引用都没扫到 —— 判据失效了。"
        f"确认扫描根 {root} 是否仍然正确(换过仓库布局?)"
    )

    missing: list[str] = []
    for p, refs in sorted(found.items()):
        for ref in sorted(refs):
            if importlib.util.find_spec(ref) is None:
                missing.append(f"  {p.relative_to(root)}: {ref}")
    assert not missing, (
        f"{name} 引用了本包里**不存在**的模块(本包改过目录结构?):\n"
        + "\n".join(missing)
        + "\n⇒ 要么把下游的路径一起改(并核对符号未改名),要么别动那部分结构。"
        "\n   本次教训见本文件头注:断过 4 天,两边都不知道。"
    )


def test_scanner_catches_a_broken_ref():
    """**扫描器自证**:把一条真会失效的引用喂进去,必须认出来。

    没有这条,`_pkg_refs` 哪天被改坏(比如只认 `import` 不认 `from … import`)
    会**静默空过** —— 而"判据本身失效"是本项目最爱记的一类。
    """
    src = "import autodrivedata.map.chamfer_ap\nfrom autodrivedata.no_such_module import x\n"
    refs = _pkg_refs(src)
    assert refs == {"autodrivedata.map.chamfer_ap", "autodrivedata.no_such_module"}
    assert importlib.util.find_spec("autodrivedata.map.chamfer_ap") is not None
    assert importlib.util.find_spec("autodrivedata.no_such_module") is None

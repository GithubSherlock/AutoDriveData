"""★ **全包每一个带 `main()` 的模块都必须有 `__main__` 守卫。**

## 为什么单独钉这一条

2026-10-06 实测踩到:**一次清理脚本把两个模块的守卫删掉了**,后果是

```
python -m autodrivedata.edit.noise_curve --base ... --work ...   ⇒  无输出、退出码 0
```

—— **不报错、不提示、什么都不做**。比崩溃恶劣得多:脚本在批处理里会被当成"跑过了",
而产物目录是空的(甚至没建)。

## ★★ 2026-10-07:同一条**又现形一次**,而当时的判据**覆盖不到**

`python -m autodrivedata.gs.eval_edit ...` 同样静默无输出、`rc=0`。
根因一模一样 —— `gs/eval_edit.py` 少了守卫。

**但当时已经有一条判据了**(`tests/edit/test_edit_entrypoints.py`),
它写的理由正是「后果是 CLI 静默无输出」…… 而它**只扫 `autodrivedata/edit/`**
⇒ `gs/` 那一整套 CLI 从来没被扫过。

⇒ 本仓那条教训的**第三次**现形:「**判据的覆盖范围本身也是判据的一部分**」
(前两次:① 只扫 3 个 `.md` 漏了包内 docstring 的链接;② 只扫 3 份文档漏了 edit 计划)。
**这一版扫全包**,并把自证的阈值拎出来 —— 扫到 0 个模块时它恒绿,那等于没有。
"""

from __future__ import annotations

import ast
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]  # autodrivedata/


def _has_main(tree: ast.Module) -> bool:
    return any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body)


def _has_guard(tree: ast.Module) -> bool:
    for n in tree.body:
        if not isinstance(n, ast.If):
            continue
        t = n.test
        if isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__":
            return True
    return False


def _modules() -> list[tuple[Path, ast.Module]]:
    out = []
    for f in sorted(_PKG.rglob("*.py")):
        if "tests" in f.parts or "__pycache__" in f.parts:
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        if _has_main(tree):
            out.append((f, tree))
    return out


def test_the_scan_actually_found_modules():
    """自证:**扫到 0 个模块时本判据恒绿,等于没有**。实测全包 ~88 个。"""
    mods = _modules()
    assert len(mods) >= 80, f"只扫到 {len(mods)} 个带 main() 的模块 —— 判据可能扫空了"


def test_every_main_has_a_guard():
    missing = [str(f.relative_to(_PKG)) for f, t in _modules() if not _has_guard(t)]
    assert not missing, (
        "这些模块有 `main()` 却没有 `__main__` 守卫 ⇒ `python -m` 进去**什么都不做、退出码 0**:\n  "
        + "\n  ".join(missing)
    )


def test_the_checker_can_actually_fail():
    """反向自证:把守卫拿掉,上面那条判据必须能红。

    不看行为、只看这个检查器本身 —— 一条永远说"没问题"的判据比没有更坏。
    """
    src = "def main():\n    pass\n"
    assert not _has_guard(ast.parse(src)), "无守卫的源码被判成有守卫 ⇒ 判据恒绿"
    assert _has_guard(ast.parse(src + '\nif __name__ == "__main__":\n    main()\n'))

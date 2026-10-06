"""★ **每个带 `main()` 的 `edit/` 模块都必须有 `__main__` 守卫。**

## 为什么单独钉这一条

2026-10-06 实测踩到:**一次清理脚本把两个模块的守卫删掉了**,后果是

```
python -m autodrivedata.edit.noise_curve --base ... --work ...   ⇒  无输出、退出码 0
```

—— **不报错、不提示、什么都不做**。比崩溃恶劣得多:脚本在批处理里会被当成"跑过了",
而产物目录是空的(甚至没建)。

判据是**源码级**的(AST 扫 `if __name__ == "__main__"` 节点),不看行为 ——
行为判据要真跑一个 CLI,而那正是"跑一次也不报错"的那种。
"""

from __future__ import annotations

import ast
from pathlib import Path

_PKG = Path(__file__).resolve().parents[2]  # autodrivedata/
_EDIT = _PKG / "edit"


def _modules_with_main() -> list[Path]:
    out = []
    for f in sorted(_EDIT.glob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        if any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body):
            out.append(f)
    return out


def test_there_is_at_least_one_cli():
    """自证:这条判据必须**真的扫到了东西** —— 扫到 0 个模块时它恒绿,等于没有。"""
    assert len(_modules_with_main()) >= 6, "edit/ 下的 CLI 模块不该这么少(判据可能扫空了)"


def test_every_main_has_a_guard():
    missing = []
    for f in _modules_with_main():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        has = any(
            isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            and getattr(n.test.left, "id", "") == "__name__"
            for n in tree.body
        )
        if not has:
            missing.append(f.name)
    assert not missing, f"这些模块有 main() 但没有 `__main__` 守卫(跑起来静默无输出、退出码 0):{missing}"


class TestYoloWeightIsShared:
    """★★ **凡是要走 yolo 的入口,`--weight` 默认值必须取同一个常量。**

    2026-10-06 实测:这个路径原本只写在一个 CLI 的 `add_argument` 里,而新写的两个入口
    把 `--weight` 默认成 `""` ⇒ **`--backend yolo` 根本跑不起来**:

        TypeError: model='' is not a supported model format

    —— 报错在 ultralytics 深处,**完全指不到"你没给权重"**。
    """

    def test_no_module_hardcodes_the_kitti_yolo_path(self):
        """路径只许出现在 `perception/backends.py` 的常量里 —— 别处一律引用它。"""
        offenders = []
        for f in sorted(_PKG.rglob("*.py")):
            if f.name == "backends.py" or "tests" in f.parts:
                continue
            src = f.read_text(encoding="utf-8")
            if "kitti_finetune/yolo11s_kitti" in src:
                offenders.append(str(f.relative_to(_PKG)))
        assert not offenders, f"这些文件又写死了 yolo 权重路径(应当引用 DEFAULT_YOLO_WEIGHT):{offenders}"

    def test_edit_clis_default_to_the_shared_constant(self):
        for name in ("downstream_eval.py", "noise_curve.py"):
            src = (_EDIT / name).read_text(encoding="utf-8")
            assert "default=DEFAULT_YOLO_WEIGHT" in src, f"{name} 的 --weight 默认值没接常量"

"""文档不腐守卫:**CLAUDE.md / README.md 里写的路径必须真能解析**。

**为什么要有这个文件(2026-09-26 实测教训)**:一次目录重构之后,CLAUDE.md 里
**57 条文件引用有 21 条失效**(37%)、markdown 坏链 8/24。全部是**静默**的 ——
文档不会报错,只有照着做的人才会拿到 `FileNotFoundError` / `ImportError`。
其中 `python -m pytest tests/ -q` 指向已删除的 `tests/`,而它是**提交前两件套里的一件**。

失效模式与代码里的 `paths.py::parents[1]` 同族:**"写的时候对、挪了之后静默错"**。
所以判据不能靠人定期自查,必须机械可执行。

**只扫机械可判的两类**(散文里的裸文件名如 `core.py` 不算,它们本就不表示路径):
1. **markdown 链接目标** —— `[文字](路径)`,相对本文件解析后必须存在;
2. **`python -m autodrivedata.<...>` 模块** —— 必须能被 `importlib.util.find_spec` 解析。

**第三类(2026-09-26 补)**:包内 `.py` **docstring 里的 markdown 链接**。
第一版只扫了上面三个 `.md`,于是 16 条坏链在「全文档坏链 0」的结论下**整体漏网**
—— **判据的覆盖范围本身也是判据的一部分**。典型一条在 `autodrivedata/utils/fonts.py`:
链接目标写作 `../autodrivedata/sim/live_common.py`(文字是仓库根相对路径、目标却按
文件相对解析,谁按它找过去都扑空)。

> 本文件里**不要**把坏链写成 `链接语法` 的样子 —— 下面的守卫会把它当真链接抓。
> (初版就在这里翻过车:举例的那条把守卫自己测红了。)
"""

from __future__ import annotations

import importlib.util
import re

import pytest

from autodrivedata.utils import paths

# 文档的权威落点。**新增顶层文档时加进来** —— 否则它不在守卫范围内。
DOCS = ("CLAUDE.md", "README.md", "docs/fileTree.md")
ROOT = paths.PROJECT_ROOT

_MD_LINK = re.compile(r"\]\(([^)\s]+)\)")
# 末位必须是字母/数字,且后面不能再跟词字符或 `*` —— 否则会把散文里的**通配符**
# `python -m autodrivedata.sim.collect_*` 抓成模块名 `...collect_`(实测踩过)。
_MOD_CMD = re.compile(r"python -m (autodrivedata\.[a-z_0-9.]*[a-z_0-9])(?![a-z_0-9.*])")


def _resolvable(module: str) -> bool:
    """模块名能否解析成真实文件(注意 `find_spec` 对带数字的段同样有效)。"""
    return importlib.util.find_spec(module) is not None


class TestDocLinksResolve:
    """文档里的相对链接必须指向存在的文件。"""

    @pytest.mark.parametrize("rel", DOCS)
    def test_markdown_links_point_at_real_files(self, rel: str) -> None:
        doc = ROOT / rel
        if not doc.is_file():
            pytest.skip(f"{rel} 不存在")
        bad = [
            target
            for target in _MD_LINK.findall(doc.read_text(encoding="utf-8"))
            # 跳过外链、页内锚点,以及含空格的散文(如 `[…](yaw=0)`)
            if not target.startswith(("http", "#"))
            and " " not in target
            and not (doc.parent / target).exists()
        ]
        assert not bad, f"{rel} 有指向不存在文件的链接: {bad}"


class TestDocCommandsResolve:
    """文档里 `python -m autodrivedata.X` 的模块必须真的存在。

    **这是本轮踩到的真缺陷的直接回归钉**:重构后 `bin/` 被删,而文档命令段一度仍写
    `python -m pytest tests/ -q`(指已删除的目录)。模块入口同理会随目录移动而失效。
    """

    @pytest.mark.parametrize("rel", DOCS)
    def test_module_entrypoints_are_importable_targets(self, rel: str) -> None:
        doc = ROOT / rel
        if not doc.is_file():
            pytest.skip(f"{rel} 不存在")
        mods = sorted(set(_MOD_CMD.findall(doc.read_text(encoding="utf-8"))))
        if not mods:
            # 合法:纯索引类文档(如 docs/fileTree.md)本来就不含命令行。
            # 「扫描器有没有失效」由下面那条自证单测管,不在这里要求每个文档都有命令。
            pytest.skip(f"{rel} 不含 `python -m autodrivedata.*` 命令(索引类文档),跳过")
        bad = [m for m in mods if not _resolvable(m)]
        assert not bad, f"{rel} 里这些模块入口已失效(查 §2 与 docs/fileTree.md): {bad}"

    def test_scan_actually_finds_the_documented_entrypoints(self) -> None:
        """扫描器自证:CLAUDE.md 的常用命令段应有足够多的模块入口(防正则腐化后空过)。"""
        mods = _MOD_CMD.findall((ROOT / "CLAUDE.md").read_text(encoding="utf-8"))
        assert len(set(mods)) >= 10, f"只抽到 {len(set(mods))} 个模块入口,正则可能已失效"


# 只有**文件后缀白名单**里的目标才判。docstring 里 `[m](米/像素)`、`[左正](左正)` 这类
# 是**单位/图例注记**,不是路径;不加这道过滤会当场多出 9 条假阳性(实测)。
_PATH_EXT = (".py", ".md", ".png", ".json", ".sh", ".cpp", ".txt", ".mp4", ".pt")


def _docstring_links() -> list[tuple[str, int, str]]:
    """包内 `.py` 里形如 `](...)` 且目标带文件后缀的相对链接 → [(相对路径, 行号, 目标)]。"""
    hits: list[tuple[str, int, str]] = []
    for py in sorted((ROOT / "autodrivedata").rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        rel = py.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            for tgt in _MD_LINK.findall(line):
                if tgt.startswith(("http", "#")) or not tgt.endswith(_PATH_EXT):
                    continue
                hits.append((rel, lineno, tgt))
    return hits


class TestPackageDocstringLinks:
    """包内 `.py` docstring 的 markdown 链接必须指向真文件。

    **为什么单独立一条**:它的失效模式和 §模块一完全不同 —— 文档里的坏链只有
    "照着做的人"会踩到,而 **docstring 里的坏链是给改代码的人看的导航**:
    `autodrivedata/utils/fonts.py` 里那条指向 `../autodrivedata/sim/live_common.py`,
    谁按它找过去都会扑空。重构后实测 **16 条**,全部是「文字写仓库根相对路径、
    目标却按文件相对解析」的同一形态。
    """

    def test_docstring_links_point_at_real_files(self) -> None:
        bad = [
            f"{rel}:{lineno} -> {tgt}"
            for rel, lineno, tgt in _docstring_links()
            if not (ROOT / rel).parent.joinpath(tgt).exists()
        ]
        assert not bad, "包内 docstring 有指向不存在文件的链接(目标须相对**本文件**):\n" + "\n".join(bad)

    def test_scan_actually_finds_the_cross_module_links(self) -> None:
        """扫描器自证:包内跨文件链接数量足够多(防正则/后缀表腐化后空过)。"""
        n = len(_docstring_links())
        assert n >= 15, f"只扫到 {n} 条包内链接,后缀白名单或正则可能已失效"

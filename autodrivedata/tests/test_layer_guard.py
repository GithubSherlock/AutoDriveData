"""层守卫:`autodrivedata/**` 的每个目录允许 import 什么(Plan_fileTree.md §3)。

**这是项目最有价值的那条不变量的可执行版本。** 旧版(`test_paths.py` 的
`test_package_stays_pure_value`)只有一条「整包不许 carla/torch」,它有两个洞、一个表达力缺陷:

1. **马甲库**:判据是 `mods & {"carla", "torch"}` 按模块名字面匹配,而 `from ultralytics import YOLO`
   实测会拉起 torch(`import ultralytics` 后 `'torch' in sys.modules == True`),字面上却不含 `torch`
   ⇒ **静默溜过**(本次审计实测 3 个文件如此:`eval_2d_ab` / `eval_attr` / `finetune_synth`)。
2. **字面量动态导入**:`importlib.import_module("torch")` / `__import__("torch")` 在 AST 上看不到
   import 语句,静态扫描的天然盲区。
3. **表达力**:写不出「`gt/` 不许依赖 `sim/`」这类规则 —— 只有一条全局拒绝清单。

**诚实说明:新版不是全面更强。** 包根 / `utils` / `gt` / `slam` 的保护力度与旧版等价
(它们本来就整包禁 carla/torch)。真正新增的是上面三条,加上「让这次目录重构在结构上成为可能」
—— 旧版用 `pkg.rglob("*.py")` 递归全包,`sim/` 一建就会红,而代码一个字没坏。
"""

from __future__ import annotations

import ast
from pathlib import Path

from autodrivedata.utils import paths

# --------------------------------------------------------------------------- 规则表
# 键 = 目录(相对 autodrivedata/,`/` 分隔;"" = 包根);值 = 该目录下【禁止】import 的三方顶层名。
# 匹配 = **最长前缀匹配**(故 "map/maptr" 比 "map" 更具体,前者胜)。

# torch 的马甲:import 它们会拉起 torch —— 必须与 torch 同列,否则红线形同虚设
_MASQ = frozenset({"ultralytics", "mmdet3d", "mmcv", "lightning"})
_PURE = frozenset({"carla", "torch"}) | _MASQ  # 纯值层:两者都不许
_CARLA_OK = frozenset({"torch"}) | _MASQ  # 许 carla,不许 torch
_TORCH_OK = frozenset({"carla"})  # 许 torch,不许 carla
_ANY = frozenset()  # 不设限

LAYER_RULES: dict[str, frozenset[str]] = {
    "": _PURE,  # 包根:迁移期尚未进入能力子目录的模块
    "utils": _PURE,  # 通用数学 / 基础设施(geometry, paths, fonts)
    "gt": _PURE,  # GT 生成
    "slam": _PURE,  # SLAM 纯值部分
    "export": _PURE,  # 落盘(kitti / nuscenes)
    "calib": _CARLA_OK,  # 标定:探针要连 CARLA,但不需要 GPU
    "perception": _TORCH_OK,
    "traj": _TORCH_OK,
    "gs": _TORCH_OK,
    "map/maptr": _TORCH_OK,
    "map": _ANY,  # 含 maptr/(torch) 与 probe_mapvec_oracle(carla)
    "sim": _ANY,  # 含 live_common(carla+torch)
    "tests": _ANY,  # 测试跟着被测对象走
}

_DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__"})


def rule_key_for(rel_dir: str) -> str:
    """命中规则的**键**(最长前缀匹配);返回 `""` 表示只命中了包根回落。"""
    hits = [k for k in LAYER_RULES if k == "" or rel_dir == k or rel_dir.startswith(k + "/")]
    return max(hits, key=len)


def rule_for(rel_dir: str) -> frozenset[str]:
    """给定相对目录(包根为 ""),返回其禁用集合。"""
    return LAYER_RULES[rule_key_for(rel_dir)]


def _abs_of_relative(py_file: Path, pkg_root: Path, level: int, module: str | None) -> str:
    """把相对导入解析成绝对模块名(`from .a import b` → `autodrivedata.<所在包>.a`)。"""
    pkg_parts = ("autodrivedata", *py_file.relative_to(pkg_root).parts[:-1])
    up = min(level - 1, len(pkg_parts))
    parts = list(pkg_parts[: len(pkg_parts) - up])
    if module:
        parts += module.split(".")
    return ".".join(parts)


def imported_toplevels(src: str, py_file: Path, pkg_root: Path) -> set[str]:
    """AST 取源码里**所有**可以被检查到的三方顶层 import 名。

    覆盖四种形态:绝对 `import x` / `from x import y`、相对导入(解析成绝对名)、
    以及**字面量**动态导入(`importlib.import_module("x")` / `__import__("x")`)。
    变量参数的动态导入在静态上不可判定,本函数**有意**不猜。
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    names.add(node.module.split(".")[0])
            else:
                # 相对导入按定义是包内引用,解析后顶层恒为 `autodrivedata` ⇒ 永不命中禁用集合。
                # 这里显式解析**不是为了抓违规**,而是让「扫了哪些节点」没有盲区:
                # 一旦将来有人把包内模块命名为与禁用模块同名(如 `sim/torch.py`),这条路径不会静默跳过。
                names.add(_abs_of_relative(py_file, pkg_root, node.level, node.module).split(".")[0])
        elif isinstance(node, ast.Call):
            fn = node.func
            is_dyn = (isinstance(fn, ast.Attribute) and fn.attr in _DYNAMIC_IMPORTERS) or (
                isinstance(fn, ast.Name) and fn.id in _DYNAMIC_IMPORTERS
            )
            if is_dyn and node.args and isinstance(node.args[0], ast.Constant):
                arg = node.args[0].value
                if isinstance(arg, str):
                    names.add(arg.split(".")[0])
    return names


def scan_package_layers(pkg_root: Path, injected: dict[str, str] | None = None) -> dict[str, list[str]]:
    """扫描包内所有 `.py`,返回 {违规文件(相对 pkg_root): 命中的禁用模块}。

    `injected` = {相对路径: 合成源码}。**给定时只扫这些**,用于守卫自身的反向自证
    (不往仓库里写坏代码就能证明规则真的会红)。
    """
    offenders: dict[str, list[str]] = {}
    if injected is not None:
        sources = {pkg_root / rel: src for rel, src in injected.items()}
    else:
        sources = {
            f: f.read_text(encoding="utf-8")
            for f in sorted(pkg_root.rglob("*.py"))
            if "__pycache__" not in f.parts
        }

    for path, src in sources.items():
        rel = path.relative_to(pkg_root)
        rel_dir = "" if rel.parent == Path(".") else rel.parent.as_posix()
        forbidden = rule_for(rel_dir)
        if not forbidden:
            continue
        hit = imported_toplevels(src, path, pkg_root) & forbidden
        if hit:
            offenders[rel.as_posix()] = sorted(hit)
    return offenders


# --------------------------------------------------------------------------- 守卫本体
class TestPackageLayers:
    """真实包必须满足规则表。"""

    def test_package_respects_layer_rules(self):
        offenders = scan_package_layers(paths.PROJECT_ROOT / "autodrivedata")
        assert not offenders, "目录层级违规(见 Plan_fileTree.md §3.2;改规则前先想清楚为什么):\n" + "\n".join(
            f"  {f}: {mods}" for f, mods in sorted(offenders.items())
        )

    def test_every_subdirectory_has_an_explicit_rule(self):
        """每个子目录必须被**包根之外的**一条规则覆盖。

        漏声明的后果是**静默回落到包根规则** —— 而"静默回落的默认值"正是本守卫要消灭的东西:
        新目录若需要 torch 却回落到 `_PURE`,你会得到一条莫名其妙的红;若需要纯值却回落到 `_ANY`,
        你会得到**假绿**。后者更坏,所以这里强制显式。

        ⚠️ **判据是「命中的键 != ""」而不是「目录名在表里」** —— 这条区别有实际意义:
        `tests/sim/`、`map/maptr/` 这类**子目录**本就该由父规则覆盖,不是遗漏。
        (本条第一次写成「目录名在表里」,阶段 2 建出 `autodrivedata/tests/sim/` 时误报,
        是守卫自己抓出来的判据缺陷。)
        """
        pkg = paths.PROJECT_ROOT / "autodrivedata"
        dirs = (
            p.relative_to(pkg).as_posix()
            for p in pkg.rglob("*")
            if p.is_dir() and p.name != "__pycache__" and not p.name.startswith(".")
        )
        undeclared = sorted(d for d in dirs if rule_key_for(d) == "")
        assert not undeclared, (
            f"这些子目录只命中了包根回落规则,未显式声明(见本文件顶部 LAYER_RULES): {undeclared}"
        )


class TestLayerGuardSelfCheck:
    """**守卫自身的立论自证**。

    一条"永远绿"的守卫比没有守卫更坏 —— 它给的是假信心。这里用**合成源码注入**
    (不往仓库里写坏代码)证明:三类违规各自都能被抓住,且规则**能区分**而不是"见 carla 就红"。
    """

    @staticmethod
    def _scan(rel: str, src: str) -> dict[str, list[str]]:
        return scan_package_layers(paths.PROJECT_ROOT / "autodrivedata", injected={rel: src})

    def test_catches_carla_in_pure_dir(self):
        assert self._scan("utils/probe.py", "import carla\n") == {"utils/probe.py": ["carla"]}

    def test_catches_masquerade_library(self):
        """马甲库:字面不含 torch,但 import 即拉起 torch —— 旧守卫的洞。"""
        got = self._scan("utils/probe.py", "from ultralytics import YOLO\n")
        assert got == {"utils/probe.py": ["ultralytics"]}, "马甲库必须与 torch 同列,否则红线形同虚设"

    def test_catches_literal_dynamic_import(self):
        """字面量动态导入:AST 里没有 import 语句 —— 静态扫描的盲区。"""
        got = self._scan("utils/probe.py", "import importlib\nimportlib.import_module('torch')\n")
        assert got.get("utils/probe.py") == ["torch"]

    def test_catches_builtin_dunder_import(self):
        assert self._scan("utils/probe.py", "__import__('mmdet3d')\n").get("utils/probe.py") == ["mmdet3d"]

    def test_allows_carla_where_declared(self):
        """立论的反面:规则必须**能区分**,不是"见 carla 就红"的蠢规则。"""
        assert self._scan("sim/collect.py", "import carla\n") == {}

    def test_subdirectory_is_covered_by_its_parent_rule(self):
        """父规则覆盖子目录 —— 与「未显式声明」是两回事(判据缺陷的回归钉)。

        阶段 2 建出 `autodrivedata/tests/sim/` 时,旧判据(「目录名在表里」)把它误报为遗漏;
        实际它由 `tests` 规则正确覆盖。真正的遗漏是**只命中包根回落**(`rule_key_for == ""`)。
        """
        assert rule_key_for("tests/sim") == "tests"
        assert rule_key_for("map/maptr") == "map/maptr"  # 更具体者胜
        assert rule_key_for("map") == "map"
        assert rule_key_for("brand_new_dir") == "", "新目录若无人声明,必须落回包根(这才叫遗漏)"

    def test_allows_torch_where_declared(self):
        assert self._scan("perception/eval.py", "import torch\n") == {}

    def test_catches_torch_where_only_carla_allowed(self):
        """`calib/` 许 carla 不许 torch —— 单向规则必须真的单向。"""
        assert self._scan("calib/probe.py", "import torch\n") == {"calib/probe.py": ["torch"]}

    def test_catches_carla_where_only_torch_allowed(self):
        assert self._scan("perception/eval.py", "import carla\n") == {"perception/eval.py": ["carla"]}

    def test_longest_prefix_wins(self):
        """`map/maptr` 比 `map` 更具体 ⇒ 前者胜:`maptr/` 禁 carla,`map/` 不设限。"""
        assert self._scan("map/maptr/model.py", "import carla\n") == {"map/maptr/model.py": ["carla"]}
        assert self._scan("map/other.py", "import carla\n") == {}

    def test_relative_import_is_resolved_not_skipped(self):
        """相对导入**不产生**三方顶层名,但也不该被静默跳过 —— 解析后顶层恒为 autodrivedata。"""
        got = self._scan("utils/probe.py", "from .fonts import get_font\nfrom ..geometry import x\n")
        assert got == {}, "相对导入是包内引用,本就不该命中禁用集合"

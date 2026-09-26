"""paths.py 的产出纪律锚点:产物必须落在项目内,且与 cwd 无关。"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

from autodrivedata.utils import paths


class TestProjectRoot:
    def test_root_is_repo_root(self):
        """项目根 = 仓库根。判据必须**三者齐备**,不能只看 `pyproject.toml`。

        ⚠️ 不要退回「只断言含 pyproject.toml」:`paths._find_project_root()` 的定义就是
        「向上找第一个含 pyproject.toml 的目录」,单这一条是**恒真**的,钉不住任何东西。
        `.git` 与 `autodrivedata/utils/paths.py` 才是真判据 —— 若哪天包内混进一个 pyproject.toml,
        向上搜索会停在那里,这两条立刻红(见 Plan_fileTree.md §5.6)。
        """
        assert (paths.PROJECT_ROOT / "pyproject.toml").is_file()
        assert (paths.PROJECT_ROOT / "autodrivedata" / "utils" / "paths.py").is_file()
        assert (paths.PROJECT_ROOT / ".git").exists(), "PROJECT_ROOT 必须是仓库根,不是包目录"
        assert paths.OUTPUTS == paths.PROJECT_ROOT / "outputs"
        # 反例钉:产物目录绝不能落在包内(§5.6 的实际故障形态)
        assert paths.OUTPUTS != paths.PROJECT_ROOT / "autodrivedata" / "outputs"

    def test_relative_is_anchored_to_project_not_cwd(self, tmp_path):
        """核心回归:换 cwd 后相对输出仍落在项目内(曾散到项目外)。"""
        old = Path.cwd()
        try:
            os.chdir(tmp_path)  # 一个与项目无关的目录
            got = paths.project_path("outputs/foo/bar.pt")
            assert got == paths.PROJECT_ROOT / "outputs" / "foo" / "bar.pt"
            assert not str(got).startswith(str(tmp_path))
        finally:
            os.chdir(old)

    def test_absolute_passthrough(self, tmp_path):
        """显式绝对路径 = 用户明确要写到别处,不劫持。"""
        target = tmp_path / "explicit" / "x.json"
        assert paths.project_path(target) == target

    def test_ensure_parent_creates_dir(self, tmp_path):
        got = paths.ensure_parent(tmp_path / "a" / "b" / "c.pt")
        assert got.parent.is_dir()
        assert got.name == "c.pt"

    # 「包不 import carla/torch」的守卫**已迁到 test_layer_guard.py**,并升级为按目录的规则表
    # (旧版 `test_package_stays_pure_value` 只有一条全局拒绝清单,且漏马甲库与动态导入)。
    # 见 Plan_fileTree.md §3。

    @pytest.mark.parametrize("name", ["outputs/carla/libmhookshim.so", "outputs/carla/nvidia-compat"])
    def test_state_paths_are_inside_project(self, name: str):
        assert str(paths.project_path(name)).startswith(str(paths.PROJECT_ROOT))


class TestWeightPaths:
    """权重路径回归钉(Plan_fileTree.md §5.1)。

    实际故障:`autodrivedata/perception/sem_bev.py` 写 `project_path("yolo11s-seg.pt")`(行内注释也说「权重放项目根」),
    而文件实际在 `models/` —— **代码 / 文档 / 文件位置三处同口径地错**,谁都不报错,
    跑起来只是 ultralytics 静默联网重下(不可复现 + 污染工作区)。

    判据不靠人眼:AST 取代码里所有 `project_path("<...>.pt")` 字面量,指向不存在时,
    若**同名文件在仓库别处存在**即判为路径写错;仓库里也没有才是合法的「权重未入库」。
    """

    # 权重【未入库】在本仓可能出现的落点(见 docs/fileTree.md);空串 = 项目根
    _WEIGHT_ROOTS = ("", "models", "outputs/models")

    def _pt_literals(self):
        """扫 `bin/`(迁移期遗留)与 `autodrivedata/` 下所有 project_path("*.pt") 字面量。"""
        for base in ("bin", "autodrivedata"):
            for src in sorted((paths.PROJECT_ROOT / base).rglob("*.py")):
                if "__pycache__" in src.parts:
                    continue
                for node in ast.walk(ast.parse(src.read_text(encoding="utf-8"))):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "project_path"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                        and node.args[0].value.endswith(".pt")
                    ):
                        yield src, node.args[0].value

    def test_weight_paths_point_at_their_real_location(self):
        checked, offenders = 0, []
        for src, rel in self._pt_literals():
            checked += 1
            if (paths.PROJECT_ROOT / rel).is_file():
                continue  # 指向真实文件 → 通过
            name = Path(rel).name
            elsewhere = [
                Path(root) / name
                for root in self._WEIGHT_ROOTS
                if (paths.PROJECT_ROOT / root / name).is_file() and (Path(root) / name) != Path(rel)
            ]
            if elsewhere:  # 同名文件在别处存在 ⇒ 路径写错,不是「未入库」
                offenders.append(
                    f"{src.relative_to(paths.PROJECT_ROOT)}: project_path({rel!r}) 不存在,"
                    f"但同名文件在 {elsewhere[0]!s}"
                )
        assert checked >= 2, f"只扫到 {checked} 处 project_path('*.pt') —— 扫描失效,测试形同虚设"
        assert not offenders, "权重路径写错(见 Plan_fileTree.md §5.1):\n" + "\n".join(offenders)


class TestGitignoreTraps:
    """`.gitignore` 陷阱回归钉(Plan_fileTree.md §5.2–§5.4)。

    三条都是**静默**失效:`git add` 不报错 —— 该忽略的被悄悄纳入(1.2 G 侧车),
    该入库的被悄悄跳过(assets/ 里的图)。人眼与代码评审都看不见。
    """

    @staticmethod
    def _ignored(rel: str) -> bool:
        """`git check-ignore` 对不存在的路径同样有效,故无需真建文件。"""
        return (
            subprocess.run(
                ["git", "check-ignore", "-q", rel],
                cwd=paths.PROJECT_ROOT,
                capture_output=True,
            ).returncode
            == 0
        )

    def test_optimizer_sidecars_are_ignored(self):
        """§5.2:`*.pt` 不匹配 `.pt.opt`,而侧车每个约 250 M、合计比权重还大。"""
        assert self._ignored("models/maptr_ep512.pt"), "权重本身应被忽略"
        assert self._ignored("models/maptr_ep512.pt.opt"), "优化器侧车(.pt.opt)必须被忽略"

    def test_logs_dir_is_ignored(self):
        """§5.3:根 `logs/` 曾不被忽略 ⇒ 训练日志会直接入库。"""
        assert self._ignored("logs/train.log")

    def test_assets_is_trackable_but_outputs_still_ignored(self):
        """§5.4:全局 `*.png` 会静默吞掉 assets/;放行时不许连带放开产物目录。"""
        assert not self._ignored("assets/logo.png"), "assets/ 下的静态资源必须可入库"
        assert not self._ignored("assets/data/icon.svg"), "assets/ 需支持子目录"
        assert self._ignored("outputs/logo.png"), "产物目录下的图仍须被忽略(不能误放行)"

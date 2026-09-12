"""paths.py 的产出纪律锚点:产物必须落在项目内,且与 cwd 无关。"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from autodrivedata import paths


class TestProjectRoot:
    def test_root_is_repo_root(self):
        # 项目根判据 = 同时有 pyproject.toml 与 autodrivedata/ 包
        assert (paths.PROJECT_ROOT / "pyproject.toml").is_file()
        assert (paths.PROJECT_ROOT / "autodrivedata" / "paths.py").is_file()
        assert paths.OUTPUTS == paths.PROJECT_ROOT / "outputs"

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

    def test_package_stays_pure_value(self):
        """包纪律:autodrivedata 全包不 import carla/torch(纯值,任何 env 可单测)。

        用 AST 取真实 import 名,不用子串匹配——文档字符串里写着"不 import carla"
        的子串匹配会假红(本测试第一版就踩了)。
        """
        pkg = paths.PROJECT_ROOT / "autodrivedata"
        offenders: dict[str, set[str]] = {}
        for src_file in sorted(pkg.rglob("*.py")):
            mods: set[str] = set()
            tree = ast.parse(src_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods.add(node.module.split(".")[0])
            hit = mods & {"carla", "torch"}
            if hit:
                offenders[str(src_file.relative_to(pkg))] = hit
        assert not offenders, f"autodrivedata 包必须纯值,违规: {offenders}"

    @pytest.mark.parametrize("name", ["outputs/carla/libmhookshim.so", "outputs/carla/nvidia-compat"])
    def test_state_paths_are_inside_project(self, name: str):
        assert str(paths.project_path(name)).startswith(str(paths.PROJECT_ROOT))

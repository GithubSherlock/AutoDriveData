"""`edit/augment` 的判据 —— 它交出去的是一份**训练数据配方**,而失效模式全是静默的。

三条自证,缺一条这个配方就不许交出去:

1. ★ **恒等塔基**:`β=0` 那一档必须与 src **逐位相同** —— 否则管线在退化之外还动了别的,
   整条链的 Δ 都无法归因;
2. ★ **至少一档非恒等** —— 否则配方是 no-op,而 no-op 与"代码根本没接上"长得一样;
3. ★ **一档一个 root** —— 同一底图的多个档塞进同一个 root,下游随机切分会在 val 里
   留下 train 的**近重复**(泄漏)。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from autodrivedata.edit import augment as A


def _root(tmp_path, name, *, n=3, with_depth=True, val=120):
    r = tmp_path / name
    (r / "training/image_2").mkdir(parents=True)
    (r / "training/label_2").mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.full((20, 30, 3), val, np.uint8)).save(r / f"training/image_2/{i:06d}.png")
        (r / f"training/label_2/{i:06d}.txt").write_text(
            "Car 0 0 0 1 1 5 5 1 1 1 1 1 1 0\n", encoding="utf-8"
        )
    if with_depth:
        (r / "training/depth").mkdir(parents=True)
        for i in range(n):
            np.save(r / f"training/depth/{i:06d}.npy", np.full((20, 30), 12.0))
    return r


class TestIdentityRung:
    def test_beta_zero_is_bit_identical_to_src(self, tmp_path):
        """★ 恒等塔基 —— 比"逐像素平均差"更严:比的是**逐文件 sha256**。"""
        src = _root(tmp_path, "src")
        rep = A.augment_root(src, tmp_path / "out", kind="fog", betas=[0.0, 0.5])
        zero = rep["levels"][0]
        assert zero["bit_identical_to_src"] is True
        assert A.mask_checksum(Path_of(zero)) == A.mask_checksum(src)

    def test_a_nonzero_rung_actually_differs(self, tmp_path):
        src = _root(tmp_path, "src")
        rep = A.augment_root(src, tmp_path / "out", kind="fog", betas=[0.0, 0.5])
        assert A.mask_checksum(Path_of(rep["levels"][1])) != A.mask_checksum(src)

    def test_all_zero_rungs_raise(self, tmp_path):
        """★ 全是 β=0 ⇒ **no-op**,而 no-op 与"没接上"长得一样 ⇒ 当场抛。"""
        src = _root(tmp_path, "src")
        with pytest.raises(SystemExit, match="no-op"):
            A.augment_root(src, tmp_path / "out", kind="fog", betas=[0.0, 0.0])


class TestLayout:
    def test_one_root_per_rung_not_one_root_with_all_rungs(self, tmp_path):
        """★★ 一档一个 root —— 塞进同一个 root 会让下游的随机切分产生**近重复泄漏**。"""
        src = _root(tmp_path, "src")
        rep = A.augment_root(src, tmp_path / "out", kind="fog", betas=[0.0, 0.5, 1.0])
        roots = [Path_of(x) for x in rep["levels"]]
        assert len(set(roots)) == 3
        for r in roots:
            assert (r / "training/image_2").is_dir() and (r / "training/label_2").is_dir()

    def test_manifest_states_all_of_it_is_training_data(self, tmp_path):
        src = _root(tmp_path, "src")
        rep = A.augment_root(src, tmp_path / "out", kind="fog", betas=[0.0, 0.5])
        assert "训练数据" in rep["note"] and "另一个" in rep["note"]

    def test_rung_names_are_indexed_not_float(self):
        """★ 目录名带**序号**不带 β —— 浮点进目录名会因舍入对不上。"""
        assert A.level_name(3) == "b03"


class TestHardPrecondition:
    def test_fogdepth_without_depth_raises(self, tmp_path):
        """★ `fogdepth` 是**距离相关**的,没有真值深度就算不出散射 ⇒ 当场抛,不静默退化成别的。"""
        src = _root(tmp_path, "src", with_depth=False)
        with pytest.raises(SystemExit, match="depth"):
            A.augment_root(src, tmp_path / "out", kind="fogdepth", betas=[0.05])

    def test_unknown_kind_raises(self, tmp_path):
        src = _root(tmp_path, "src")
        with pytest.raises(SystemExit, match="未知注入"):
            A.augment_root(src, tmp_path / "out", kind="nope", betas=[1.0])

    def test_empty_betas_raise(self, tmp_path):
        src = _root(tmp_path, "src")
        with pytest.raises(SystemExit, match="空的"):
            A.augment_root(src, tmp_path / "out", kind="fog", betas=[])


def Path_of(level):
    from pathlib import Path

    return Path(level["root"])

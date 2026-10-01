"""GT 口径迁移工具(`gt/refilter.py`)的回归钉 —— 在**合成 root** 上跑,不碰真数据。

## 钉的是什么

这个工具只做一件事:把 `label_2/` 里触发 `MIN_BOX_SIDE_PX` 的行删掉,**别的什么都不动**。
两类"不报错的错"值得钉:

1. **多删 / 少删**:判据必须与出框侧同源(`is_degenerate_gt_line`),边界(1.00 px)不能手滑;
2. **动了不该动的东西**:`image_2` / `velodyne` / `calib` / `pose` 必须是**硬链接**而不是
   拷贝更不是重写 —— 重写会静默改掉像素,而这个工具存在的全部理由就是"只动 GT 一个量"。
   判据用 `st_ino` 相等(硬链接的文件系统证据),不是"文件大小一样"。
"""

from __future__ import annotations

import os

import pytest

from autodrivedata.gt.refilter import refilter_root

GOOD = "Car 0.00 0 0.00 653.33 189.17 685.99 211.57 1.52 2.01 4.51 3.50 0.84 44.96 -1.57"
SLIVER = "Car 0.75 0 0.00 947.14 231.78 1189.44 231.79 1.52 2.01 4.51 3.50 0.84 2.50 -1.57"


def _fake_root(base, *, frames=2):
    """造一个最小 KITTI root:每帧 1 条正常 GT + 1 条退化,外加三个旁路文件。"""
    for i in range(frames):
        fid = f"{i:06d}"
        for sub, payload in (
            ("image_2", b"\x89PNG fake"),
            ("velodyne", b"\x00" * 16),
            ("calib", b"P2: 1 0 0 0\n"),
            ("pose", b"1 0 0 0 0 1 0 0 0 0 1 0\n"),
        ):
            d = base / "training" / sub
            d.mkdir(parents=True, exist_ok=True)
            (
                d / f"{fid}.{'png' if sub == 'image_2' else ('bin' if sub == 'velodyne' else 'txt')}"
            ).write_bytes(payload)
        lab = base / "training" / "label_2"
        lab.mkdir(parents=True, exist_ok=True)
        (lab / f"{fid}.txt").write_text(f"{GOOD}\n{SLIVER}\n", encoding="utf-8")
    return base


class TestRefilterRoot:
    def test_drops_only_the_degenerate_lines(self, tmp_path):
        src = _fake_root(tmp_path / "src")
        stats = refilter_root(src, tmp_path / "out")
        assert stats == {"frames": 2, "gt_in": 4, "gt_kept": 2, "gt_dropped": 2}
        for i in range(2):
            got = (tmp_path / "out" / "training" / "label_2" / f"{i:06d}.txt").read_text()
            assert got == GOOD + "\n", f"帧 {i} 的剩余内容不对:{got!r}"

    def test_side_dirs_are_hardlinks_not_copies(self, tmp_path):
        """★ 判据是 `st_ino` 相等 —— "大小一样"对拷贝也成立,证明不了没重写。"""
        src = _fake_root(tmp_path / "src", frames=1)
        refilter_root(src, tmp_path / "out")
        for sub, name in (("image_2", "000000.png"), ("velodyne", "000000.bin"), ("calib", "000000.txt")):
            a = os.stat(src / "training" / sub / name)
            b = os.stat(tmp_path / "out" / "training" / sub / name)
            assert a.st_ino == b.st_ino, f"{sub} 不是硬链接(被真拷了)"
            assert b.st_nlink >= 2

    def test_label_is_a_real_rewrite_not_a_link(self, tmp_path):
        """反向对照:`label_2` **必须**是新 inode —— 否则"重筛"会就地改掉源 root。"""
        src = _fake_root(tmp_path / "src", frames=1)
        refilter_root(src, tmp_path / "out")
        a = os.stat(src / "training" / "label_2" / "000000.txt")
        b = os.stat(tmp_path / "out" / "training" / "label_2" / "000000.txt")
        assert a.st_ino != b.st_ino
        assert len((src / "training" / "label_2" / "000000.txt").read_text().splitlines()) == 2, (
            "源 root 被改了 —— 迁移工具不许动源"
        )

    def test_refuses_to_overwrite_existing_output(self, tmp_path):
        """跑第二次必须报错而不是把上一轮的结果混进去(`--out` 写错时最危险)。"""
        src = _fake_root(tmp_path / "src", frames=1)
        refilter_root(src, tmp_path / "out")
        with pytest.raises(FileExistsError):
            refilter_root(src, tmp_path / "out")

    def test_empty_root_is_not_an_error(self, tmp_path):
        """一个帧都没有(空 root)也要给出 0 而不是抛 —— 便于批量脚本一路跑下去。"""
        (tmp_path / "src" / "training" / "label_2").mkdir(parents=True)
        assert refilter_root(tmp_path / "src", tmp_path / "out")["gt_in"] == 0

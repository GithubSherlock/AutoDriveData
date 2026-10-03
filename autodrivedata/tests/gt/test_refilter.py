"""GT 口径迁移工具(`gt/refilter.py`)的回归钉 —— 在**合成 root** 上跑,不碰真数据。

## 钉的是什么

这个工具只做一件事:把 `label_2/` 里触发 `MIN_BOX_SIDE_PX` 的行删掉,**别的什么都不动**。
两类"不报错的错"值得钉:

1. **多删 / 少删**:判据必须与出框侧同源(`is_degenerate_gt_line`),边界(1.00 px)不能手滑;
2. **动了不该动的东西**:旁路产物必须是**硬链接**而不是拷贝更不是重写 —— 重写会静默改掉
   像素,而这个工具存在的全部理由就是"只动 GT 一个量"。判据用 `st_ino` 相等(硬链接的
   文件系统证据),不是"文件大小一样"。
3. **★ 一路都不能漏**(2026-10-01 补):原实现写死 `("image_2","velodyne","calib","pose")`,
   **漏了 `samples/RADAR_*`** ⇒ 五份 `kitti_ab_epic_gtfix_*` 的雷达**静默消失**(重筛前
   70 帧 pcd、重筛后 0)。数据照出、帧号照对,只是少了一路。现在按谓词全量硬链,
   `test_radar_channel_is_carried_over` 钉死它。
"""

from __future__ import annotations

import os
import struct

import pytest

from autodrivedata.gt.refilter import png_size, refilter_root

GOOD = "Car 0.00 0 0.00 653.33 189.17 685.99 211.57 1.52 2.01 4.51 3.50 0.84 44.96 -1.57"
SLIVER = "Car 0.75 0 0.00 947.14 231.78 1189.44 231.79 1.52 2.01 4.51 3.50 0.84 2.50 -1.57"

#: 真归档的 `P2` 与画幅(`outputs/kitti_ab_occl2_full_all/training/{calib,image_2}/000000.*`)。
#: ⚠️ **抄录即口径** —— 从磁盘逐字读出来的,不是按 f=621 手打的:f / cx / cy 任一个错,
#: 重算出的框会整体偏,而**看着仍像个框**(同 §P-M.10 那条"能画出图不是投影正确的证据")。
REAL_P2 = (
    "6.210000e+02 0.000000e+00 6.205000e+02 0.000000e+00 "
    "0.000000e+00 6.210000e+02 1.870000e+02 0.000000e+00 "
    "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00"
)
REAL_W, REAL_H = 1242, 375

#: 同上文件的第 4 行 —— **被画幅右/下边缘裁断**的近处车(旧口径下框只到 948.86/296.86)。
CLIPPED = "Car 0.38 0 0.00 780.87 198.22 948.86 296.86 1.49 2.16 4.79 3.50 1.66 6.97 -1.57"


def _png_header(w: int, h: int) -> bytes:
    """24 字节的合法 PNG 头(签名 + IHDR 长度 + 类型 + 宽高)—— 刚好是 `png_size` 读的范围。"""
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", w, h)


def _fake_root(base, *, frames=2):
    """造一个最小 KITTI root:每帧 1 条正常 GT + 1 条退化,外加三个旁路文件。"""
    for i in range(frames):
        fid = f"{i:06d}"
        for sub, payload in (
            ("image_2", b"\x89PNG fake"),
            ("velodyne", b"\x00" * 16),
            ("calib", b"P2: 1 0 0 0\n"),
            ("pose", b"1 0 0 0 0 1 0 0 0 0 1 0\n"),
            ("samples/RADAR_FRONT", b"pcd fake"),  # ★ 雷达在 samples/ 下,不是 training/
        ):
            d = base / (sub if sub.startswith("samples/") else f"training/{sub}")
            d.mkdir(parents=True, exist_ok=True)
            ext = {"image_2": "png", "velodyne": "bin"}.get(sub, "pcd" if "RADAR" in sub else "txt")
            (d / f"{fid}.{ext}").write_bytes(payload)
        lab = base / "training" / "label_2"
        lab.mkdir(parents=True, exist_ok=True)
        (lab / f"{fid}.txt").write_text(f"{GOOD}\n{SLIVER}\n", encoding="utf-8")
    return base


class TestRefilterRoot:
    def test_radar_channel_is_carried_over(self, tmp_path):
        """★ 2026-10-01 修的那个洞:`samples/RADAR_*` 以前不在硬链名单里。

        症状不是报错,是**那一路整个没了** —— 而"少了一路"与"这个数据集本来就没雷达"
        在目录列表上长得一样。判据取 `st_ino`(硬链接证据)+ 文件确实存在。
        """
        src = _fake_root(tmp_path / "src", frames=2)
        stats = refilter_root(src, tmp_path / "out")
        for i in range(2):
            got = tmp_path / "out" / "samples" / "RADAR_FRONT" / f"{i:06d}.pcd"
            assert got.exists(), "雷达通道没被带过来 —— 融合线会拿到 0 帧雷达"
            assert got.stat().st_ino == (src / "samples" / "RADAR_FRONT" / f"{i:06d}.pcd").stat().st_ino
        # 旁路产物**逐文件计数**:少一路时这个数会变小,而不是静默。
        # 合成 root 每帧 5 个旁路文件(image_2/velodyne/calib/pose + 雷达)× 2 帧 = 10
        assert stats["linked"] == 10
        assert stats["copied"] == 0  # 同设备必须**全走硬链**,不许悄悄退化成拷贝

    def test_cross_device_falls_back_to_copy_not_crash(self, tmp_path, monkeypatch):
        """★ `--out` 在另一个盘(或 `/tmp`)时硬链不可能 —— 退化成拷贝,**分开计数**。

        原先直接抛 `OSError: Invalid cross-device link`,看不出"该怎么办"。而静默当成
        链接更坏:`linked` 与 `copied` 是两个不同的性质(零占盘 + 改一个另一个跟着变
        vs 占盘 + 各走各的),混成一个数下游就分不出来了。
        """
        import errno
        import os as _os

        src = _fake_root(tmp_path / "src", frames=1)
        real_link = _os.link

        def fake_link(a, b):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        monkeypatch.setattr(_os, "link", fake_link)
        stats = refilter_root(src, tmp_path / "out")
        monkeypatch.setattr(_os, "link", real_link)
        assert stats["linked"] == 0 and stats["copied"] == 5
        assert (tmp_path / "out" / "samples" / "RADAR_FRONT" / "000000.pcd").exists()

    def test_drops_only_the_degenerate_lines(self, tmp_path):
        src = _fake_root(tmp_path / "src")
        stats = refilter_root(src, tmp_path / "out")
        assert stats["frames"] == 2
        assert (stats["gt_in"], stats["gt_kept"], stats["gt_dropped"]) == (4, 2, 2)
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


def _clip_root(base, lines, *, calib=True):
    """`--clip2d` 需要的那个最小 root:真 `P2` + 合法 PNG 头 + 指定行。"""
    tr = base / "training"
    for sub in ("label_2", "calib", "image_2"):
        (tr / sub).mkdir(parents=True, exist_ok=True)
    (tr / "label_2" / "000000.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if calib:
        (tr / "calib" / "000000.txt").write_text(f"P2: {REAL_P2}\n", encoding="utf-8")
    (tr / "image_2" / "000000.png").write_bytes(_png_header(REAL_W, REAL_H))
    return base


class TestClip2d:
    """`--clip2d`:同一个工具的**第二个修正**(裁断口径),与退化剔除一次做完。

    ## 钉的三件事

    1. **该动的动** —— 被画幅裁断的行,新框要**贴到画幅边缘**(旧口径只到物体自己的角点);
    2. **不该动的一个字节都不动** —— 未裁断的行两种口径逐位相同,重打一遍只会注入
       ~0.1 px 的字段舍入噪声(归档里 82.4%)。"只动该动的行"要**可检验**,不能是承诺;
    3. **缺件不许回落旧口径** —— 回落会让一份 root 里混两套口径,而它俩在 A/B 硬门槛下
       **长得一样**(逐帧条数照对),只有框偏,偏多少还随 `ry` 变。
    """

    def test_unclipped_lines_are_byte_identical(self, tmp_path):
        """★ 判据 2:**逐字节相等**,不是"重算后差得少"。"""
        src = _clip_root(tmp_path / "src", [GOOD])
        stats = refilter_root(src, tmp_path / "out", clip2d=True)
        got = (tmp_path / "out" / "training" / "label_2" / "000000.txt").read_text()
        assert got == GOOD + "\n"
        assert stats["gt_reboxed"] == 0, "未裁断的行被重画了 —— diff 会白涨 82%"
        assert (stats["gt_kept"], stats["gt_dropped"]) == (1, 0)

    def test_a_clipped_line_reaches_the_frame_edge(self, tmp_path):
        """★ 判据 1:真归档的裁断行 → 框贴到 `x2 = W-1 = 1241` / `y2 = H-1 = 374`。

        期望值与 `test_gt.py::TestReboxLine` 那条**同一个案例、同一组数** —— 出框侧与
        重算侧共用 `box2d_from_projection`,所以两处给的答案必须是同一个。
        """
        src = _clip_root(tmp_path / "src", [CLIPPED])
        stats = refilter_root(src, tmp_path / "out", clip2d=True)
        got = (tmp_path / "out" / "training" / "label_2" / "000000.txt").read_text().split("\n")[0]
        assert [float(v) for v in got.split()[4:8]] == pytest.approx(
            [781.08, 198.27, 1241.0, 374.0], abs=0.02
        )
        assert stats["gt_reboxed"] == 1

    def test_the_two_corrections_are_done_in_one_pass(self, tmp_path):
        """退化行在 `--clip2d` 下**照样被剔除** —— 两个修正不是二选一。"""
        src = _clip_root(tmp_path / "src", [GOOD, SLIVER, CLIPPED])
        stats = refilter_root(src, tmp_path / "out", clip2d=True)
        assert (stats["gt_in"], stats["gt_kept"], stats["gt_dropped"]) == (3, 2, 1)
        assert stats["gt_reboxed"] == 1
        names = (tmp_path / "out" / "training" / "label_2" / "000000.txt").read_text().splitlines()
        assert [ln.split()[0] for ln in names] == ["Car", "Car"]

    def test_multiple_frames_each_use_their_own_calib(self, tmp_path):
        """★ 逐帧取**自己那份** `calib/{fid}.txt` —— 用错一帧的 `P2` 不会报错,只会让框偏。"""
        src = _clip_root(tmp_path / "src", [CLIPPED])
        tr = src / "training"
        (tr / "label_2" / "000001.txt").write_text(GOOD + "\n", encoding="utf-8")
        # 帧 1 给一份**明显不同**的 P2(cx 挪到 100)⇒ 若错用了这一份,帧 0 的框会整体左移。
        (tr / "calib" / "000001.txt").write_text(f"P2: {REAL_P2.replace('6.205000e+02', '1.000000e+02')}\n")
        (tr / "image_2" / "000001.png").write_bytes(_png_header(REAL_W, REAL_H))
        refilter_root(src, tmp_path / "out", clip2d=True)
        frame0 = (tmp_path / "out" / "training" / "label_2" / "000000.txt").read_text()
        assert [float(v) for v in frame0.split()[4:8]] == pytest.approx(
            [781.08, 198.27, 1241.0, 374.0], abs=0.02
        )

    def test_missing_calib_or_image_raises_instead_of_falling_back(self, tmp_path):
        """★ 判据 3:缺件**抛**,不静默回落旧口径(混口径的 root 在 A/B 下看不出来)。"""
        src = _clip_root(tmp_path / "src", [CLIPPED], calib=False)
        with pytest.raises(FileNotFoundError):
            refilter_root(src, tmp_path / "out", clip2d=True)

        src2 = _clip_root(tmp_path / "src2", [CLIPPED])
        (src2 / "training" / "image_2" / "000000.png").write_bytes(b"\x89PNG\r\n\x1a\n truncated")
        with pytest.raises(ValueError, match="不是 PNG"):
            refilter_root(src2, tmp_path / "out2", clip2d=True)

    def test_off_by_default_keeps_the_previous_generation_byte_for_byte(self, tmp_path):
        """★ **默认值即旧口径**:不传 `--clip2d` 时,产物与 2026-10-01 那版逐字节相同。

        这条是"跨口径不可混比"红线的工具侧保障 —— 新口径必须**显式索取**,免得批量脚本
        一路跑下去、悄悄换掉了某一代 root 的口径。
        """
        src = _clip_root(tmp_path / "src", [GOOD, CLIPPED, SLIVER])
        stats = refilter_root(src, tmp_path / "out", clip2d=False)
        got = (tmp_path / "out" / "training" / "label_2" / "000000.txt").read_text()
        assert got == GOOD + "\n" + CLIPPED + "\n"  # 只有退化行被删,没有被重画的行
        assert (stats["gt_reboxed"], stats["gt_dropped"]) == (0, 1)


class TestPngSize:
    """`png_size` —— `gt` 是 `_PURE` 层,为读一个画幅不该拉进 PIL。"""

    def test_reads_the_real_archived_frame(self):
        from pathlib import Path

        p = Path("outputs/kitti_ab_occl2_full_all/training/image_2/000000.png")
        if not p.exists():
            pytest.skip("归档数据不在盘上")
        assert png_size(p) == (REAL_W, REAL_H)

    def test_rejects_a_non_png(self, tmp_path):
        bad = tmp_path / "x.png"
        bad.write_bytes(b"not a png at all, but long enough to pass 24 bytes")
        with pytest.raises(ValueError, match="不是 PNG"):
            png_size(bad)

"""`edit/kitti_square` 的判据回归钉。

这一层坏掉的症状**不是报错**,是**下游 AP 崩掉** —— 而"框没跟着裁"与"生成质量差"
在数字上长得一模一样。⇒ 每条判据都配一条**反向自证**:把已知的错误做法造出来,分得开才算数。
"""

from __future__ import annotations

import numpy as np
import pytest

from autodrivedata.edit import kitti_square as K
from autodrivedata.gt.core import MIN_BOX_SIDE_PX

#: 采集帧的真实画幅(全仓一致)。
W, H = 1242, 375
#: 手算锚:该画幅下中心方裁到 512 的几何。
X0, Y0, S, KSCALE = 433, 0, 375, 512.0 / 375.0
# ⚠️ 常量名**不许叫 `K`** —— 那会把模块别名(`kitti_square as K`)覆盖掉,
#    24 条用例一起报 `'float' object has no attribute 'transform_line'`(第一版就是这么错的)。

#: 真实 `label_2` 的第一行(取自定义采集),用于手算锚点。
REAL_LINE = "Car 0.00 0 0.00 645.35 178.94 668.81 195.29 1.52 2.01 4.51 3.50 0.77 60.16 -1.57"


def _line(x1, y1, x2, y2, *, cls="Car", rest="1.52 2.01 4.51 3.50 0.77 60.16 -1.57"):
    return f"{cls} 0.00 0 0.00 {x1} {y1} {x2} {y2} {rest}"


class TestCropGeometry:
    def test_real_frame_becomes_center_square(self):
        """1242×375 → 裁中间 375 宽的一条(x0=433.5),缩到 512 ⇒ k=512/375。"""
        assert K.crop_geometry(W, H, 512) == (X0, Y0, S, KSCALE)

    def test_tall_frame_crops_vertically(self):
        """高图裁上下(与宽图裁左右对称)—— 写反了会把画面主体裁掉。"""
        assert K.crop_geometry(300, 900, 300) == (0, 300, 300, 1.0)

    def test_square_passthrough(self):
        assert K.crop_geometry(512, 512, 512) == (0, 0, 512, 1.0)

    def test_crop_matches_generator(self):
        """★★ **与生成侧同一套取整** —— 这是"条件图与 GT 框对齐"的机械判据。

        `cldm_backend.to_square_rgb` 写的是 `(w−s)//2`(向下取整);本模块若用
        `round((w−s)/2)`,1242×375 会得 **434** 而生成侧 **433** ⇒ **差 1 px**,
        而"差 1 px"不报错,只让下游 AP 莫名低一点。
        判据不看数字,看**实际裁掉了哪些列** —— 与 `to_square_rgb` 逐列比。
        """
        from autodrivedata.edit.cldm_backend import to_square_rgb

        img = np.zeros((H, W, 3), np.uint8)
        img[0, :, 0] = np.arange(W, dtype=np.uint8)  # 每列一个可分辨的值
        kept = to_square_rgb(img)  # 生成侧实际保留的列
        x0, y0, s, _k = K.crop_geometry(W, H, 512)
        np.testing.assert_array_equal(kept[0, :, 0], img[0, x0 : x0 + s, 0], "方裁起点与生成侧不一致")
        assert kept.shape[:2] == (s, s)

    @pytest.mark.parametrize("whs", [(0, 10, 8), (10, 0, 8), (10, 10, 0)])
    def test_bad_size_raises(self, whs):
        with pytest.raises(ValueError, match="非法尺寸"):
            K.crop_geometry(*whs)


class TestTransformLine:
    def test_hand_computed_anchor(self):
        """★ **手算锚点**(不是拿同一份公式再算一遍)。

        真实第一行的框 `(645.35, 178.94, 668.81, 195.29)`:
        `u' = (u − 433)·512/375` ⇒ `(289.93, 244.32, 321.96, 266.64)`。
        ⚠️ x0 是 **433(向下取整)** 不是 433.5 —— 见 `crop_geometry` 的取整口径。
        第一版这里写 289.24,是**手算错了 0.01**;而 `approx(abs=0.02)` 那条**照样过**,
        是下面那条精确比对(`== "289.93"`)才把它抓出来。
        全在画幅内 ⇒ `kept`。
        """
        out = K.transform_line(REAL_LINE, width=W, height=H, size=512)
        assert out.kind == "kept"
        assert out.line is not None
        got = [float(v) for v in out.line.split()[4:8]]
        assert got == pytest.approx([289.93, 244.32, 321.96, 266.64], abs=0.02)

    def test_three_d_columns_are_untouched(self):
        """★★ **3D 列(h w l x y z ry)原样不动** —— 它们是**相机系的三维量**,
        与画幅裁剪无关。动了它们 = **伪造几何**,而且不报错。
        """
        out = K.transform_line(REAL_LINE, width=W, height=H, size=512)
        assert out.line is not None
        assert out.line.split()[8:] == REAL_LINE.split()[8:]

    def test_type_and_truncation_preserved(self):
        out = K.transform_line(REAL_LINE, width=W, height=H, size=512)
        assert out.line is not None
        assert out.line.split()[:4] == REAL_LINE.split()[:4]

    def test_box_left_of_crop_is_dropped(self):
        """框整个在裁切区**左侧**(x<433.5)⇒ 丢弃,且要说清是 `dropped_out`。"""
        out = K.transform_line(_line(100, 100, 200, 150), width=W, height=H, size=512)
        assert out.kind == "dropped_out" and out.line is None

    def test_box_partially_outside_is_clipped_not_dropped(self):
        """部分越界 ⇒ **钳到画幅**(KITTI 口径:框 = 角点 min/max 再钳画幅),不是丢。"""
        out = K.transform_line(_line(400, 100, 500, 150), width=W, height=H, size=512)
        assert out.kind == "clipped" and out.line is not None
        u1, v1, u2, v2 = (float(v) for v in out.line.split()[4:8])
        assert u1 == pytest.approx(0.0, abs=0.02), "左边界应当被钳到 0"
        assert u2 > 0

    def test_frame_edge_box_kept_verbatim(self):
        """恰好贴边(全在内)⇒ 归 `kept`,不该被误报成 `clipped`。"""
        out = K.transform_line(_line(X0, 0.0, X0 + S, float(H)), width=W, height=H, size=512)
        assert out.kind == "kept"

    def test_degenerate_after_clip_is_dropped_separately(self):
        """钳完只剩几个像素 ⇒ **单独计数**(`dropped_degenerate`),不并进 `dropped_out`
        —— "擦过镜头"那类框与任何预测 IoU 恒为 0,白送一次漏检(本仓红线)。"""
        n = MIN_BOX_SIDE_PX / 2
        out = K.transform_line(_line(X0 - n, 100, X0 + n, 150), width=W, height=H, size=512)
        assert out.kind == "dropped_degenerate" and out.line is None

    def test_short_line_is_bad(self):
        assert K.transform_line("Car 0.00 0 0.00 1 2 3", width=W, height=H).kind == "bad"

    def test_reverse_selfcheck_missing_offset(self):
        """★★ **反向自证**:漏掉 `−x0`(只缩放不平移)会得到**完全不同的框**。

        这正是这一层最可能被写错的地方 —— 而它不报错,只让下游 AP 崩。
        """
        ok = K.transform_line(REAL_LINE, width=W, height=H, size=512)
        wrong_u1 = 645.35 * KSCALE  # ← 漏掉 −x0 的写法
        assert ok.line is not None
        got_u1 = float(ok.line.split()[4])
        assert abs(got_u1 - wrong_u1) > 500, "漏掉平移与正确做法应当差得很远"


class TestBuildRoot:
    def _mk_root(self, tmp_path):
        from PIL import Image

        src = tmp_path / "src"
        for sub in ("image_2", "label_2", "pose", "calib", "velodyne"):
            (src / "training" / sub).mkdir(parents=True)
        for fid in ("000000", "000001"):
            Image.fromarray(np.full((H, W, 3), 128, np.uint8)).save(src / "training/image_2" / f"{fid}.png")
            (src / "training/label_2" / f"{fid}.txt").write_text(REAL_LINE + "\n", encoding="utf-8")
            (src / "training/pose" / f"{fid}.txt").write_text(" ".join(["0"] * 12) + "\n", encoding="utf-8")
            (src / "training/calib" / f"{fid}.txt").write_text("P2: 1 0 0 0\n", encoding="utf-8")
            (src / "training/velodyne" / f"{fid}.bin").write_bytes(b"\x00" * 16)
        return src

    def test_crops_images_and_labels(self, tmp_path):
        from PIL import Image

        src = self._mk_root(tmp_path)
        dst = tmp_path / "dst"
        rep = K.build_square_root(src, dst, size=512)
        assert rep["n_frames"] == 2 and rep["kept"] == 2
        assert Image.open(dst / "training/image_2/000000.png").size == (512, 512)
        assert (dst / "training/label_2/000000.txt").read_text().split()[4] == "289.93"

    def test_pose_is_copied_unchanged(self, tmp_path):
        """位姿是**世界系**的 ⇒ 裁剪不影响它,原样搬过去。"""
        src = self._mk_root(tmp_path)
        dst = tmp_path / "dst"
        K.build_square_root(src, dst, size=512)
        assert (dst / "training/pose/000000.txt").read_text() == (
            src / "training/pose/000000.txt"
        ).read_text()

    def test_depth_is_cropped_with_the_same_affine(self, tmp_path):
        """★★ **深度也要按同一套仿射裁** —— 它是逐像素几何量,与 `image_2` 同画幅。

        2026-10-06 补:原来落盘清单里漏了它 ⇒ `degrade --kinds fogdepth` 没料可用。
        判据用**已知的线性深度坡**验:裁完仍应是同一条坡,且**逐分量等于手算的裁切**。
        """
        src = self._mk_root(tmp_path)
        (src / "training/depth").mkdir()
        h, w = H, W
        ramp = np.tile(np.arange(w, dtype=np.float32), (h, 1))  # 只随列变
        for fid in ("000000", "000001"):
            np.save(src / "training/depth" / f"{fid}.npy", ramp)
        dst = tmp_path / "dst"
        K.build_square_root(src, dst, size=512)
        got = np.load(dst / "training/depth/000000.npy")
        assert got.shape == (512, 512)
        # 首行应当是"中心裁那一列段的线性插值" —— 与 image_2 用同一个 crop_geometry
        x0, _y0, s, _k = K.crop_geometry(w, h, 512)
        assert got[0, 0] == pytest.approx(x0, abs=1.0), f"左端应对应原图第 {x0} 列"
        assert got[0, -1] == pytest.approx(x0 + s - 1, abs=1.0), "右端应对应原图第 x0+s-1 列"

    def test_depth_absent_is_fine(self, tmp_path):
        """源没有 `depth/` 时照旧 —— 不许因为它缺失就抛。"""
        src = self._mk_root(tmp_path)
        dst = tmp_path / "dst"
        rep = K.build_square_root(src, dst, size=512)
        assert rep["has_depth"] is False
        assert not (dst / "training/depth").exists()

    def test_calib_and_velodyne_are_deliberately_absent(self, tmp_path):
        """★★ **calib / velodyne 一律不落** —— 裁剪后主点要平移重缩放,
        落一份没改的比不落**更危险**(下游会拿它算出"看起来对"的投影)。"""
        src = self._mk_root(tmp_path)
        dst = tmp_path / "dst"
        K.build_square_root(src, dst, size=512)
        assert not (dst / "training/calib").exists(), "calib 在裁剪后口径变了,不许照搬"
        assert not (dst / "training/velodyne").exists()

    def test_frame_filter(self, tmp_path):
        src = self._mk_root(tmp_path)
        dst = tmp_path / "dst"
        rep = K.build_square_root(src, dst, size=512, frames=[1])
        assert rep["n_frames"] == 1
        assert (dst / "training/label_2/000001.txt").exists()
        assert not (dst / "training/label_2/000000.txt").exists()

    def test_dry_run_writes_nothing(self, tmp_path):
        src = self._mk_root(tmp_path)
        dst = tmp_path / "dst"
        rep = K.build_square_root(src, dst, size=512, dry_run=True)
        assert rep["kept"] == 2 and not dst.exists()

    def test_missing_dirs_raises(self, tmp_path):
        with pytest.raises(SystemExit, match="不是 KITTI root"):
            K.build_square_root(tmp_path, tmp_path / "o", size=512)

    def test_empty_after_filter_raises(self, tmp_path):
        src = self._mk_root(tmp_path)
        with pytest.raises(SystemExit, match="没有可用帧"):
            K.build_square_root(src, tmp_path / "o", size=512, frames=[99])


class TestFrameSpec:
    def test_range_and_list(self):
        assert K._parse_frames("0,2-4") == [0, 2, 3, 4]

    def test_empty_raises(self):
        with pytest.raises(SystemExit):
            K._parse_frames(" , ")

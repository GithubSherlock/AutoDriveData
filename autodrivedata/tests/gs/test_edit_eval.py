"""`gs/edit_gs` 与 `gs/eval_edit` 的判据。

三条被测契约都是**删错了但看着像成功**:

- **`-1`(未归属)被顺手删掉** ⇒ 整片背景消失,而那不是报错,是"场景没了";
- **请求的 id 一个都没命中,却静默成功** ⇒ 下游把"编辑前后一模一样"读成"这个物体删不掉";
- **A/B 位姿其实对不上** ⇒ 三档 PSNR 里混进了"两边不是同一个视角",数照样出得来。

外加一条**第三态**:掩膜为空(`n=0`)是**未判**,既不是通过也不是失败。
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch

from autodrivedata.gs import edit_gs, eval_edit
from autodrivedata.gs.render_gs import GaussianSet
from autodrivedata.perception.inst_tags import encode_instance_png


def _gs(n: int, seed: int = 0) -> GaussianSet:
    rng = np.random.default_rng(seed)
    return GaussianSet(
        means=rng.normal(size=(n, 3)).astype(np.float32),
        rots=np.tile(np.array([[1.0, 0, 0, 0]], dtype=np.float32), (n, 1)),
        scales_lin=np.full((n, 3), 0.05, dtype=np.float32),
        col=rng.random((n, 3)).astype(np.float32),
        opac=np.full(n, 0.8, dtype=np.float32),
    )


class TestRemoveIds:
    def test_removes_exactly_the_requested_id(self):
        gs = _gs(10)
        attr = np.array([7, 7, 9, 9, 9, -1, -1, 3, 3, 7], dtype=np.int64)
        out, rep = edit_gs.remove_ids(gs, attr, [7])
        assert out.n == 7
        assert rep["removed_ids"] == {7: 3}  # 7 出现在 0/1/9 三处
        # 留下的必须是**同一批行**,不是"随便 7 个"
        keep = attr != 7
        assert np.array_equal(out.means, gs.means[keep])
        assert np.array_equal(out.col, gs.col[keep])
        assert np.array_equal(out.opac, gs.opac[keep])

    def test_unassigned_is_refused(self):
        """★ `-1` 不是"另一个 id",它是**没测到** —— 删它等于删背景。"""
        gs = _gs(4)
        attr = np.array([5, 5, -1, -1], dtype=np.int64)
        with pytest.raises(SystemExit, match="未归属"):
            edit_gs.remove_ids(gs, attr, [-1])

    def test_id_with_zero_hits_raises(self):
        """★ 静默删 0 个 ⇒ 下游读成"这个物体本来就删不掉"。"""
        gs = _gs(4)
        attr = np.array([5, 5, 5, 5], dtype=np.int64)
        with pytest.raises(SystemExit, match="一个高斯都没有"):
            edit_gs.remove_ids(gs, attr, [999])

    def test_partial_miss_still_raises(self):
        """命中一个、漏一个 —— **也要抛**:漏的那个会被静默忽略。"""
        gs = _gs(4)
        attr = np.array([5, 5, 7, 7], dtype=np.int64)
        with pytest.raises(SystemExit, match="一个高斯都没有"):
            edit_gs.remove_ids(gs, attr, [5, 999])

    def test_length_mismatch_raises(self):
        """attr 与高斯数对不上 ⇒ 那是**另一份数据**的归属结果。"""
        with pytest.raises(SystemExit, match="对不上"):
            edit_gs.remove_ids(_gs(4), np.array([1, 1, 1], dtype=np.int64), [1])

    def test_reports_unassigned_share(self):
        """未归属占比必须报出来 —— 它高的时候"删干净了"这件事本身就不成立。"""
        gs = _gs(4)
        attr = np.array([5, 5, -1, -1], dtype=np.int64)
        _, rep = edit_gs.remove_ids(gs, attr, [5])
        assert rep["unassigned_share"] == pytest.approx(0.5)
        assert rep["n_removed"] == 2


class TestRemoveInBox:
    """★ **本链路上"删道具"只能走这条** —— 那个道具没有实例 id(见模块头注)。

    凭什么它是对的:`prop.json` 的位置与尺寸是**摆位时自证过**的,而盒只有 1.3×1.1×1.9 m、
    摆在环心,背景在盒外好几米 ⇒ 盒内的高斯只可能是在解释那个道具。
    """

    def _gs(self, pts):
        return GaussianSet(
            means=np.asarray(pts, dtype=np.float32),
            rots=np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (len(pts), 1)),
            scales_lin=np.full((len(pts), 3), 0.05, np.float32),
            col=np.zeros((len(pts), 3), np.float32),
            opac=np.full(len(pts), 0.8, np.float32),
        )

    def test_only_gaussians_inside_the_box_are_removed(self):
        # ⚠️ [0.3,0.2,1.2] 的 z=1.2 > 半高 0.93 —— **它在盒外**,别数错(第一版就数错过)
        gs = self._gs([[0, 0, 0.9], [0.3, 0.2, 1.2], [-0.4, 0.1, 0.5], [5, 0, 0.9], [0, 5, 0.9], [0, 0, 9.0]])
        out, n = edit_gs.remove_in_box(gs, np.zeros(3), np.array([0.65, 0.53, 0.93]), 0.0)
        assert n == 2
        assert out.n == 4
        # 留下的必须是**同一批行**:盒内是 0 与 2 号,所以留 1/3/4/5
        assert np.array_equal(out.means, gs.means[[1, 3, 4, 5]])

    def test_yaw_rotates_the_box_not_the_points(self):
        """盒转 90° ⇒ 两个点**换边**。⚠️ 夹具要让"换边"真的发生(半宽得一个宽一个窄)。"""
        gs = self._gs([[0.6, 0.0, 0.9], [0.0, 0.6, 0.9]])
        half = np.array([0.5, 0.8, 0.93])  # x 窄 y 宽 ⇒ 转 90° 后两者的内外互换
        out0, n0 = edit_gs.remove_in_box(gs, np.zeros(3), half, 0.0)
        assert n0 == 1 and np.allclose(out0.means[0], [0.6, 0.0, 0.9])  # 删了 y 那个
        out90, n90 = edit_gs.remove_in_box(gs, np.zeros(3), half, np.pi / 2)
        assert n90 == 1 and np.allclose(out90.means[0], [0.0, 0.6, 0.9])  # 转后删 x 那个

    def test_empty_box_raises(self):
        """★ 框里 0 个必须抛 —— 静默删 0 个会让下游把「编辑前后一样」读成结论。"""
        gs = self._gs([[10, 10, 10]])
        with pytest.raises(SystemExit, match="一个高斯都没有"):
            edit_gs.remove_in_box(gs, np.zeros(3), np.array([0.65, 0.53, 0.93]), 0.0)

    def test_margin_widens_the_box(self):
        gs = self._gs([[0.7, 0.0, 0.9]])
        half = np.array([0.65, 0.53, 0.93])
        with pytest.raises(SystemExit):
            edit_gs.remove_in_box(gs, np.zeros(3), half, 0.0)
        assert edit_gs.remove_in_box(gs, np.zeros(3), half, 0.0, margin=0.1)[1] == 1

    def test_box_center_uses_the_actor_origin_plus_offset(self, tmp_path):
        """★ `PropBox.location` 是 **actor 原点**,不是盒中心 —— 直接当中心用会整体偏 0.93 m。"""
        p = tmp_path / "prop.json"
        p.write_text(
            json.dumps(
                {
                    "prop": {
                        "location": [-78.0, 13.0, 0.0],
                        "box_offset": [0.0, 0.0, 0.9294],
                        "size": [1.305, 1.055, 1.860],
                        "yaw_deg": 0.0,
                    }
                }
            ),
            encoding="utf-8",
        )
        center, half, yaw = edit_gs.box_from_prop_json(p)
        assert np.allclose(center, [-78.0, 13.0, 0.9294])
        assert np.allclose(half, [0.6525, 0.5275, 0.93])
        assert yaw == 0.0


class TestCliWiring:
    """★ 两条都是**冒烟才抓到**的接线问题(单测不到 CLI 就看不见)。

    它们不是"顺手改好看",各自对应一个真会被读错的输出:
    `--list` 被 `--out-tag` 卡住 ⇒ 想看一眼 id 直方图得先编一个产物名;
    `-1` 的拒绝发生在**读盘之后** ⇒ 参数错却先报"文件不存在"。
    """

    def test_check_ids_rejects_unassigned_early(self):
        with pytest.raises(SystemExit, match="未归属"):
            edit_gs.check_ids([3, -1])

    def test_check_ids_passes_normal_ids(self):
        edit_gs.check_ids([3, 8])  # 不抛即可

    def test_main_checks_ids_before_loading(self):
        """★ `check_ids` 必须排在 `load_set` **之前** —— 参数错不该等读完几百 MB 才报。

        (这条**抓到过一次回归**:加 3D 框那条路时我把校验挪进了 else 分支,排到了读盘之后。)
        """
        src = inspect.getsource(edit_gs.main)
        assert src.index("check_ids(ids)") < src.index("load_set(")

    def test_list_mode_does_not_require_out_tag(self):
        """`--out-tag` 是**运行期**要求,不是 argparse 的 `required=True`。"""
        src = inspect.getsource(edit_gs.main)
        i = src.index('"--out-tag"')
        block = src[i : src.index("add_argument(", i + 10)]
        assert "required=True" not in block, "`--list` 会被 required=True 卡住"
        assert "out_tag" in src.split("if args.list:")[1], "`--list` 之后没有 out-tag 的运行期校验"


class TestHistogram:
    def test_counts_every_id_including_unassigned(self):
        h = edit_gs.id_histogram(np.array([3, 3, 3, -1, 8], dtype=np.int64))
        assert h == {3: 3, -1: 1, 8: 1}


class TestPropJson:
    def test_reads_instance_id(self, tmp_path):
        p = tmp_path / "prop.json"
        p.write_text(json.dumps({"prop": {"instance_id": 4242}}), encoding="utf-8")
        assert edit_gs.ids_from_prop_json(p) == [4242]

    def test_missing_key_raises_not_defaults(self, tmp_path):
        """缺字段**报错不猜** —— 猜一个 id 出来会删掉另一个物体,而渲染看着仍像回事。"""
        p = tmp_path / "prop.json"
        p.write_text(json.dumps({"prop": {"type_id": "static.prop.x"}}), encoding="utf-8")
        with pytest.raises(SystemExit, match="instance_id"):
            edit_gs.ids_from_prop_json(p)


def _capture(root, poses, pitches=(0.0,)):
    root.mkdir(parents=True, exist_ok=True)
    (root / "pitches.json").write_text(json.dumps(list(pitches)), encoding="utf-8")
    for p in pitches:
        (root / f"poses_{int(p)}.json").write_text(json.dumps(poses), encoding="utf-8")
    return root


def _pose(i, x=0.0):
    return {"i": i, "x": x, "y": 0.0, "z": 1.5, "pitch": 0.0, "yaw": 0.0, "roll": 0.0}


class TestPosePairing:
    def test_identical_poses_are_paired(self, tmp_path):
        a = _capture(tmp_path / "a", [_pose(0), _pose(1)])
        b = _capture(tmp_path / "b", [_pose(0), _pose(1)])
        r = eval_edit.pose_pairing(a, b)
        assert r["paired"]
        assert r["max_abs_delta"] == 0.0

    def test_a_small_pose_drift_is_caught(self, tmp_path):
        """★ 这条是硬门槛的**直接反例**:差 1 mm 也不许放行。

        P1 那边容差是 **0.07 m**(那边的位姿本来就来自物理过程);这边是纯函数算出来的,
        **能要求恰好相等** —— 所以容差更严是**对的**,不是"顺手调紧"。
        """
        a = _capture(tmp_path / "a", [_pose(0), _pose(1)])
        b = _capture(tmp_path / "b", [_pose(0), _pose(1, x=1e-3)])
        r = eval_edit.pose_pairing(a, b)
        assert not r["paired"]
        assert r["worst_at"] == (0, 1, "x")

    def test_different_pitches_raise(self, tmp_path):
        a = _capture(tmp_path / "a", [_pose(0)], pitches=(0.0,))
        b = _capture(tmp_path / "b", [_pose(0)], pitches=(0.0, -15.0))
        with pytest.raises(SystemExit, match="pitches"):
            eval_edit.pose_pairing(a, b)


class TestMaskSource:
    """★ **掩膜只有一个可靠来源:A/B 两张真采图的 RGB 差。**

    原方案写的是"`inst == instance_id`",实测**取不到东西** —— CARLA 的深度 / 语义 / 实例
    三个 pass **根本不渲染 `static.prop.*`**(A/B 在那三路上逐像素完全相同,而 RGB 差 57,664 px)。
    这一组钉的就是"默认走 RGB 差",以及"实例那条路仍在、且仍然是对的它自己的事"。
    """

    def _inst(self, root, ids: np.ndarray):
        p = root / "inst" / "p0"
        p.mkdir(parents=True, exist_ok=True)
        (p / "00000.png").write_bytes(encode_instance_png(ids))
        return root

    def test_rgb_diff_selects_the_changed_region(self, tmp_path):
        """两张真采图只在中间一块不同 ⇒ 掩膜就是那一块。"""
        a = torch.zeros(1, 8, 8, 3)
        b = torch.zeros(1, 8, 8, 3)
        b[0, 2:6, 2:6] = 0.5  # 差 0.5 ≫ 阈值 8/255
        m = eval_edit.build_mask_rgb_diff(tmp_path, tmp_path, [0], lambda _k: 0.0, a, b, 8, 8, dilate=0)
        assert int(m[0].sum()) == 16

    def test_rgb_diff_ignores_subthreshold_noise(self, tmp_path):
        """★ 噪声不许进掩膜 —— 阈值就定在"实测背景噪声"与"实测信号"之间。"""
        a = torch.zeros(1, 8, 8, 3)
        b = torch.full((1, 8, 8, 3), 1 / 255)  # 1/255 < 8/255 阈值
        m = eval_edit.build_mask_rgb_diff(tmp_path, tmp_path, [0], lambda _k: 0.0, a, b, 8, 8, dilate=0)
        assert int(m[0].sum()) == 0

    def test_threshold_sits_between_measured_noise_and_signal(self):
        """耦合钉:实测噪声 ≤2/255、信号 54/255 ⇒ 阈值必须夹在中间。"""
        assert 2 / 255 < eval_edit.MASK_RGB_THRESH < 54 / 255

    def test_dilation_widens_the_mask(self, tmp_path):
        a = torch.zeros(1, 8, 8, 3)
        b = torch.zeros(1, 8, 8, 3)
        b[0, 4, 4] = 0.5
        m0 = eval_edit.build_mask_rgb_diff(tmp_path, tmp_path, [0], lambda _k: 0.0, a, b, 8, 8, dilate=0)
        m2 = eval_edit.build_mask_rgb_diff(tmp_path, tmp_path, [0], lambda _k: 0.0, a, b, 8, 8, dilate=2)
        assert int(m2[0].sum()) > int(m0[0].sum())

    def test_inst_source_still_works_and_is_kept_for_the_control(self, tmp_path):
        """实例那条路**保留** —— 它正是"这个通道不渲染道具"那组对照的复现手段。"""
        ids = np.zeros((8, 8), dtype=np.uint16)
        ids[2:6, 2:6] = 77
        cap = self._inst(tmp_path, ids)
        m = eval_edit.build_mask_inst(cap, [0], lambda _k: 0.0, 77, 4, 4, dilate=0)
        assert int(m[0].sum()) == 4

    def test_default_mask_source_is_rgb_diff(self):
        """默认必须是 RGB 差 —— 另一个来源在这条链上恒为 0 像素。"""
        assert eval_edit.MASK_RGB_THRESH > 0
        src = inspect.getsource(eval_edit.main)
        i = src.index('"--mask-source"')
        block = src[i : src.index("add_argument(", i + 10)]
        assert 'default="rgb-diff"' in block


class TestThreeTier:
    """三档读数的手算锚点:全掩膜、误差全部在掩膜内 ⇒ 全局与掩膜内应当**相同**。"""

    def _case(self, da: float, de: float, db: float, n_px: int = 4, n: int = 1):
        h = w = 4
        img = torch.zeros(n, h, w, 3)
        ra, re_, rb = img + da, img + de, img + db
        mask = torch.zeros(h, w, dtype=torch.bool)
        if n_px:
            mask.view(-1)[:n_px] = True
        return eval_edit.three_tier(ra, re_, rb, img, [mask] * n)

    def test_gap_and_closed_fraction_are_hand_checkable(self):
        # 误差 0.1 / 0.01 / 0.001 ⇒ PSNR 20 / 40 / 60 dB
        r = self._case(0.1, 0.01, 0.001)
        assert r["verdict"] == "可判"
        assert r["a_mask_mean"] == pytest.approx(20.0, abs=1e-6)
        assert r["edited_mask_mean"] == pytest.approx(40.0, abs=1e-6)
        assert r["b_mask_mean"] == pytest.approx(60.0, abs=1e-6)
        assert r["gap_total_db"] == pytest.approx(40.0, abs=1e-6)
        assert r["gap_closed_db"] == pytest.approx(20.0, abs=1e-6)
        assert r["gap_closed_frac"] == pytest.approx(0.5, abs=1e-6)
        assert r["edited_vs_ceiling_db"] == pytest.approx(-20.0, abs=1e-6)

    def test_empty_mask_is_undecided_not_passed(self):
        """★ 第三态:掩膜空 ⇒ **未判**。写成"通过"就是把"没测到"读成"做到了"。"""
        r = self._case(0.1, 0.01, 0.001, n_px=0)
        assert r["verdict"] == "未判"
        assert "a_mask_mean" not in r
        assert r["n_px_total"] == 0

    def test_mask_psnr_differs_from_global_when_error_is_local(self):
        """掩膜**真的在分区**:掩膜内的误差与全局不同 ⇒ 两个数必须分开。"""
        h = w = 4
        img = torch.zeros(1, h, w, 3)
        ra = img.clone()
        ra.view(-1, 3)[:4] = 0.5  # 只在掩膜区里错
        mask = torch.zeros(h, w, dtype=torch.bool)
        mask.view(-1)[:4] = True
        r = eval_edit.three_tier(ra, img, img, img, [mask])
        assert r["a_mask_mean"] < r["a_all_mean"], "误差全在掩膜内 ⇒ 掩膜内该更差"

    def test_partial_frames_count_only_the_judged_ones(self):
        """有的帧掩膜空、有的不空 ⇒ 只按**可判的那些帧**平均,空的那些记 `None`。"""
        h = w = 4
        img = torch.zeros(2, h, w, 3)
        ra = img + 0.1
        empty = torch.zeros(h, w, dtype=torch.bool)
        full = torch.ones(h, w, dtype=torch.bool)
        r = eval_edit.three_tier(ra, img + 0.01, img + 0.001, img, [empty, full])
        assert r["n_judged"] == 1
        assert r["per_frame"][0]["a_mask"] is None
        assert r["per_frame"][1]["a_mask"] is not None

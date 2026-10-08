"""`gs/edit_gs` 与 `gs/eval_edit` 的判据。

三条被测契约都是**删错了但看着像成功**:

- **`-1`(未归属)被顺手删掉** ⇒ 整片背景消失,而那不是报错,是"场景没了";
- **请求的 id 一个都没命中,却静默成功** ⇒ 下游把"编辑前后一模一样"读成"这个物体删不掉";
- **A/B 位姿其实对不上** ⇒ 三档 PSNR 里混进了"两边不是同一个视角",数照样出得来;
- ★ **真值图没按 `idx` 取** ⇒ 拿「帧 45 的渲染」比「采集 B 的**第 0 帧**」(2026-10-07 实测,
  整张 C1 表因此作废)。**这一条的既有测试构造上抓不到** —— 它们传的全是已对齐张量。

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
    """★ 两个掩膜源**回答两个不同的问题**,不是"哪个对哪个错"。

    - `rgb-diff`(**默认**)= **物体改变了画面的地方**(含阴影/软边);
    - `inst` = **物体本体的像素**(更紧)。

    ⚠️⚠️ **2026-10-07 推翻了一条承重说法**:本组原先写着「CARLA 的实例 pass 根本不渲染
    `static.prop.*` ⇒ 按 `instance_id` 取掩膜得 **0 个像素**」。修好相机位姿口径后重采的
    capture 上,**道具 `instance_id=121` 有像素,`inst` 掩膜 3705 px**。
    当年那条读数的因是 **79.132 m 的位姿偏移**(道具在 73 m 外、不在视场),
    而**那条代码路径本身一次都没跑成过** —— 它按**全局帧号**拼文件名,而磁盘上是**逐俯仰编号**
    ⇒ 默认帧表上第二帧去找 `p-15/00090.png`(永不存在)直接 `SystemExit`。
    ⇒ 「它会得 0 个像素」这句话**从未被实测**。已修:收 `poses`,两个字段(`pitch` 取目录、`i` 取名)都对上。
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

    def test_inst_source_picks_the_object_pixels(self, tmp_path):
        """`inst` 掩膜 = `实例图 == prop_id` 的那些像素(在自己的分辨率上取,再缩到渲染分辨率)。"""
        ids = np.zeros((8, 8), dtype=np.uint16)
        ids[2:6, 2:6] = 77
        cap = self._inst(tmp_path, ids)
        m = eval_edit.build_mask_inst(cap, [0], [{"pitch": 0.0, "i": 0}], 77, 8, 8, dilate=0)
        assert int(m[0].sum()) == 16

    def test_inst_source_uses_the_per_pitch_index_not_the_global_one(self, tmp_path):
        """★★ **文件名用 `pose["i"]`,不是全局帧号** —— 就是那个"一次都没跑成过"的缺陷。

        构造一个"全局序 ≠ 逐俯仰序"的 capture:只有 `p-15/00000.png` 存在。
        传 `frames=[1]`(全局序 1 ⇒ pitch −15、i=0),**必须找到**它。
        旧实现会去找 `p-15/00001.png` 而抛 `SystemExit`。
        """
        ids = np.zeros((8, 8), dtype=np.uint16)
        ids[1:3, 1:3] = 9
        d = tmp_path / "inst" / "p-15"
        d.mkdir(parents=True)
        (d / "00000.png").write_bytes(encode_instance_png(ids))
        m = eval_edit.build_mask_inst(
            tmp_path, [1], [{"pitch": 0.0, "i": 0}, {"pitch": -15.0, "i": 0}], 9, 8, 8, 0
        )
        assert int(m[0].sum()) == 4

    def test_default_mask_source_is_rgb_diff(self):
        """默认是 RGB 差 —— 它含**阴影与软边**,而"编辑该作用在哪"问的正是那个。"""
        assert eval_edit.MASK_RGB_THRESH > 0
        src = inspect.getsource(eval_edit.main)
        i = src.index('"--mask-source"')
        block = src[i : src.index("add_argument(", i + 10)]
        assert 'default="rgb-diff"' in block


class TestPairWeather:
    """★★ **A/B 记录的天气必须一致**(2026-10-07 补)—— 「只变一个变量」的**正面**判据。

    `3dgs_ab2` 那对位姿逐位相同、天气却不同,而当时的 capture **一个字节都没记天气**
    ⇒ 只能靠差分图肉眼发现。这条判据是那条教训的直接产物。
    """

    @staticmethod
    def _w(root, eff, scene="day_clear"):
        root.mkdir(parents=True, exist_ok=True)
        (root / "weather.json").write_text(json.dumps({"scene": scene, "effective": eff}), encoding="utf-8")
        return root

    def test_identical_weather_passes(self, tmp_path):
        w = {"cloudiness": 5.0, "sun_altitude_angle": 45.0}
        r = eval_edit.pair_weather(self._w(tmp_path / "A", w), self._w(tmp_path / "B", w))
        assert r["known"] and r["ok"] and r["differing"] == {}

    def test_different_weather_is_refused_and_named(self, tmp_path):
        """★ 必须**点名**是哪个字段差多少 —— 只报"不一致"没法定位。"""
        a = {"cloudiness": 5.0, "sun_altitude_angle": 45.0}
        b = {"cloudiness": 60.0, "sun_altitude_angle": 15.0}
        r = eval_edit.pair_weather(self._w(tmp_path / "A", a), self._w(tmp_path / "B", b))
        assert not r["ok"]
        assert set(r["differing"]) == {"cloudiness", "sun_altitude_angle"}

    def test_small_float_noise_is_tolerated(self, tmp_path):
        """CARLA 回读的浮点尾数不该把一对合法 capture 判死。"""
        a = {"cloudiness": 5.0, "fog_falloff": 0.1}
        b = {"cloudiness": 5.0, "fog_falloff": 0.1 + 1e-9}
        assert eval_edit.pair_weather(self._w(tmp_path / "A", a), self._w(tmp_path / "B", b))["ok"]

    def test_legacy_capture_without_weather_json_is_undecided(self, tmp_path):
        """★ 老 capture 没有 `weather.json` ⇒ **未判**,不是通过也不是失败。"""
        (tmp_path / "A").mkdir()
        r = eval_edit.pair_weather(tmp_path / "A", self._w(tmp_path / "B", {"cloudiness": 5.0}))
        assert not r["known"] and "未判" in r["reason"]


class TestPairOutsideDiff:
    """★★ **A/B 只许差"有没有道具"这一件事**(2026-10-07 补)。

    坏掉的症状**完全静默**:`3dgs_ab2` 那对 capture 的位姿 JSON **逐位相同**、
    文件齐全、三档 PSNR 照样出得来 —— 而整帧均值 A=58.2 vs B=128.6,一边下雨一边晴。
    `pose_pairing` 只读位姿 JSON,对这种对**完全瞎**。

    ⚠️ 尺子的难点在**误伤**:合法的一对里也可能有局部差异(实测干净对 `3dgs_ab` 的帧 90
    有 3132 px 差得明显)⇒ 判据只能用**对大片像素一起变敏感**的统计量(p90),
    不能用"差了的像素占比"。下面 `test_local_blob_is_not_a_weather_change` 就是钉这一条的。
    """

    @staticmethod
    def _cap(root, ids: np.ndarray, n_frames: int = 1):
        """极小 capture:`inst/p0/{k:05d}.png` 各一张。"""
        p = root / "inst" / "p0"
        p.mkdir(parents=True, exist_ok=True)
        for k in range(n_frames):
            (p / f"{k:05d}.png").write_bytes(encode_instance_png(ids))
        return root

    def _run(self, tmp_path, a, b, n_frames=1, ids=None, dilate=1):
        """⚠️ `dilate=1`:真实默认是 **16 px**,在 16×16 的玩具上会把"道具之外"整个吃掉。
        真实默认值由 `test_threshold_sits_between_the_measured_pair` 与实测定标覆盖。"""
        hw = a.shape[1]
        if ids is None:
            ids = np.zeros((8, 8), dtype=np.uint16)  # 没有道具露出 ⇒ 整幅都在"道具之外"
        cap = self._cap(tmp_path / "A", ids, n_frames)
        poses = [{"pitch": 0.0, "i": k} for k in range(n_frames)]
        return eval_edit.pair_outside_diff(cap, list(range(n_frames)), poses, 77, a, b, hw, hw, dilate=dilate)

    def test_identical_pair_passes(self, tmp_path):
        a = torch.full((1, 16, 16, 3), 0.4)
        r = self._run(tmp_path, a, a.clone())
        assert r["ok"] and r["max_p90"] == pytest.approx(0.0, abs=1e-9)

    def test_global_shift_is_rejected(self, tmp_path):
        """★ 天气/曝光变了 ⇒ 大片像素一起变 ⇒ 必须拒判。用的是实测那对的整帧均值。"""
        a = torch.full((1, 16, 16, 3), 58 / 255)
        b = torch.full((1, 16, 16, 3), 129 / 255)
        r = self._run(tmp_path, a, b)
        assert not r["ok"] and r["max_p90"] > eval_edit.PAIR_MAX_P90_DIFF

    def test_local_blob_is_not_a_weather_change(self, tmp_path):
        """★★ **反向自证**:一**小块**像素差得再多,也不许判成"变了天气"。

        这正是不能用"差了的像素占比"的原因 —— 实测干净对 `3dgs_ab` 帧 90 有 3132 px
        (2.7%)差得明显,占比会被它顶过任何合理阈值。p90 只在**大片一起变**时才动。
        """
        a = torch.zeros(2, 16, 16, 3)
        b = a.clone()
        b[0, 0:4, 0:4] = 1.0  # 16/256 = 6.25% 的像素差了满量程
        assert self._run(tmp_path, a, b, n_frames=2)["ok"], "局部小块不是'变了天气'"
        # 对照:同样多的像素,**铺满整幅**(全局变)就应当被拒
        c = torch.zeros(2, 16, 16, 3)
        d = torch.full((2, 16, 16, 3), 0.2)
        assert not self._run(tmp_path / "g", c, d, n_frames=2)["ok"]

    def test_difference_inside_the_prop_is_excluded(self, tmp_path):
        """道具区域内的差异是**这次编辑本身**,不许算进"A/B 一致性"。"""
        ids = np.zeros((8, 8), dtype=np.uint16)
        ids[2:6, 2:6] = 77  # 道具在中间;缩到 16×16 后约 rows/cols 4–12
        a = torch.zeros(1, 16, 16, 3)
        b = a.clone()
        b[0, 5:11, 5:11] = 1.0  # 差异全落在道具上
        r = self._run(tmp_path, a, b, ids=ids)
        assert r["per_frame"][0]["prop_px"] > 0, "构造:道具得真的露面"
        assert r["ok"], "道具自己的差异不属于'A/B 不一致'"

    def test_max_over_frames_not_mean(self, tmp_path):
        """★ 逐帧取 **max**:只有一帧变了天气,整对就不能用(实测 `3dgs_ab2` 的帧 45/225 就是好的)。"""
        a = torch.zeros(2, 16, 16, 3)
        b = a.clone()
        b[1] = 0.6  # 只有第二帧变了
        assert not self._run(tmp_path, a, b, n_frames=2)["ok"], "一帧坏 ⇒ 整对拒判(取 max 不取 mean)"

    def test_threshold_sits_between_the_measured_pairs(self):
        """耦合钉:阈值必须夹在实测的干净对(0.004)与坏对(0.494)之间。"""
        assert 0.004 < eval_edit.PAIR_MAX_P90_DIFF < 0.494


class TestNegativeGap:
    """★★ **天花板低于基线时,"补上百分之几"读不出来**(2026-10-07 实测踩到)。

    紧掩膜(`inst`)那一档:天花板 12.99 **低于**不编辑 13.46 ⇒ 缺口 = −0.47,
    而旧代码照样算 `0.28 / (−0.47) = −59%` —— 那个数会让人读成"越改越差"。
    """

    def _case(self, a_db, e_db, b_db):
        # 用 PSNR 反推需要的误差:psnr = 10·log10(1/mse) ⇒ mse = 10^(−psnr/10)
        h = w = 4
        img = torch.zeros(1, h, w, 3)
        f = lambda p: float(10 ** (-p / 10)) ** 0.5  # noqa: E731
        mask = torch.ones(h, w, dtype=torch.bool)
        return eval_edit.three_tier(img + f(a_db), img + f(e_db), img + f(b_db), img, [mask])

    def test_normal_gap_keeps_the_fraction(self):
        r = self._case(20.0, 40.0, 60.0)
        assert r["verdict"] == "可判" and r["gap_closed_frac"] is not None

    def test_negative_gap_says_it_cannot_be_read(self):
        r = self._case(13.46, 13.73, 12.99)
        assert r["gap_total_db"] < 0
        assert r["gap_closed_frac"] is None, "负缺口下'补上百分之几'没有意义,不许给数"
        assert "方向反" in r["verdict"] and "读不出来" in r["reason"]
        # ★ 能读的仍然是那两个数之间的差
        assert r["gap_closed_db"] > 0


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


class TestFrameAlignment:
    """★★ **真值图必须按 `idx` 取**(2026-10-07 实测踩到,整套 C1 数字因此作废)。

    `main()` 曾把**全量** `imgs_b`(270 帧)直接传进 `three_tier`,而后者按**循环位置**取
    ⇒ 拿「帧 45 的渲染」比「采集 B 的**第 0 帧**」。⚠️ 症状**完全静默**:数照样出得来、
    格式也对;而且**只评一帧时看不出来**(那时循环位置 0 恰好是唯一那帧)。

    真正的抓手是**归档锚**:`b_all` 逐帧必须等于 `train_result_<tag>_B.json` 的 `psnr_val`
    —— 同一个模型、同一批帧、同一批图,只有口径不同。归档实测对不上
    (`12.90/12.64/13.70/14.02/14.33` vs `26.61/28.73/27.72/33.83/29.72`)。
    """

    @staticmethod
    def _frames(n: int = 4, hw: int = 4) -> torch.Tensor:
        """第 i 帧整幅填 `i/10` ⇒ 一旦取错帧,PSNR 会掉得很明显。"""
        return (torch.arange(n, dtype=torch.float32) / 10.0).view(n, 1, 1, 1) * torch.ones(n, hw, hw, 3)

    def test_full_tensor_is_refused(self):
        """★ 守卫:传全量 ⇒ **当场抛**,不许"能算就往下算"(那正是它藏了这么久的原因)。"""
        imgs = self._frames(4)
        mask = torch.ones(4, 4, dtype=torch.bool)
        with pytest.raises(SystemExit, match="按 idx 取"):
            eval_edit.three_tier(imgs[:2], imgs[:2], imgs[:2], imgs, [mask, mask])

    def test_mismatched_mask_count_is_refused(self):
        imgs = self._frames(4)
        mask = torch.ones(4, 4, dtype=torch.bool)
        with pytest.raises(SystemExit, match="masks"):
            eval_edit.three_tier(imgs[:2], imgs[:2], imgs[:2], imgs[:2], [mask])

    def test_psnr_follows_the_index_not_the_loop_position(self):
        """★ 取对帧的正向锚:渲染与真值**同取 `idx`** ⇒ 逐位相同 ⇒ PSNR 极高。

        原 bug 下第 j 张渲染会去比第 j 帧真值(而渲染取自 `idx[j]`),两者**不等** ⇒ 数会很小。
        """
        imgs = self._frames(4)
        sel = imgs[[1, 3]]
        mask = torch.ones(4, 4, dtype=torch.bool)
        r = eval_edit.three_tier(sel, sel, sel, imgs[[1, 3]], [mask, mask])
        assert r["a_all_mean"] > 100.0, "同一帧对同一帧 ⇒ PSNR 应接近 120 dB 的上限"
        assert r["gap_closed_db"] == pytest.approx(0.0, abs=1e-9)

    def test_wrong_index_is_measurably_different(self):
        """反向自证:同样两张渲染,真值换成**别的帧** ⇒ 读数必须塌下来。"""
        imgs = self._frames(4)
        sel, wrong = imgs[[1, 3]], imgs[[0, 2]]
        mask = torch.ones(4, 4, dtype=torch.bool)
        good = eval_edit.three_tier(sel, sel, sel, sel, [mask, mask])
        bad = eval_edit.three_tier(sel, sel, sel, wrong, [mask, mask])
        assert good["a_all_mean"] - bad["a_all_mean"] > 10.0

    def test_main_passes_idx_not_the_full_tensor(self):
        """结构钉:调用点必须写 `imgs_b[idx]`。

        整条采集/渲染链才碰得到这里,而那正是这个 bug 藏了这么久的原因 ——
        **已有的行为测试传的全是已对齐张量**,构造上抓不到。
        """
        src = inspect.getsource(eval_edit.main)
        assert "three_tier(ra, re_, rb, imgs_b[idx], masks)" in src
        assert "three_tier(ra, re_, rb, imgs_b, masks)" not in src

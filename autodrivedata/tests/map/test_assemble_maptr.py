"""`assemble_maptr --map-json auto`(双图池)+ **池内 rig 同源自证**的回归钉。

## 钉的是什么

2026-10-03 起,"把图池扩到第二张图"靠 `--map-json auto`:矢量 json **逐段**由该段
`calib.json` 的 `map` 字段推。这条链上有四类错**都不报错**:

1. **`map` 字段的形态** —— 两个采集器落的东西不一样(`Town13` vs `Carla/Maps/Town10HD_Opt`),
   平铺大图还多一层(`Carla/Maps/Town13/Town13`)。取错 → 拼出的文件名不存在;
2. **旧命令被改坏** —— 显式给文件时必须**逐字节不变**。单图池的新旧产物对拍是唯一判据;
3. **缺 `map` 字段** —— 静默沿用上一段的图 ⇒ 整段用错 GT,而症状只是"学不动";
4. **★ 池内混 rig** —— `MapTRDataset` 是 `self.calibs = self.infos[0]["cams"]`,
   **从首帧取一份给全池共用**。混了两种 rig **不会抛异常**,只会拿 A 图的内参投影 B 图的
   像素。组装收尾的 `assert_same_rig` 是唯一能拦住它的地方,反向自证见
   `test_mixed_rig_pool_is_rejected`。

零 CARLA 依赖:夹具是合成 root + 迷你矢量 json(经 `vecs_dump` 造,与真实产物同构)。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autodrivedata.map import assemble_maptr
from autodrivedata.map.mapvec import MapVec, vecs_dump

#: 两代 rig 各一个 **fx**(旧 621 / 新 ≈1266)。判据读的就是它 —— 见 CLAUDE.md 那条
#: 「2026-09-16 前后两代相机口径」。
FX_NEW = 1266.4
FX_OLD = 621.0


def _map_json(path: Path, label: str) -> Path:
    """迷你全图矢量:一条横跨 ego 窗口的 boundary(够 `to_maptr_annotation` 出四类键)。"""
    pts = tuple((float(x), 5.0, 0.0) for x in range(-20, 21, 2))
    path.write_text(vecs_dump((MapVec("boundary", pts, (), "b0", "test"),), label), encoding="utf-8")
    return path


def _seg(root: Path, name: str, *, map_raw: str | None, fx: float = FX_NEW, n: int = 3) -> Path:
    """一个合成段目录:`calib.json`(**带 map 溯源键**)+ `ego_pose.json`(+ 空的 cam 目录)。"""
    d = root / name
    (d / "cam_front").mkdir(parents=True)
    calib: dict = {
        "CAM_FRONT": {
            "sensor2ego": [0.44, -0.02, 1.51, -0.32, -0.32, -0.04],
            "intrinsic": [[fx, 0.0, 799.5], [0.0, fx, 449.5], [0.0, 0.0, 1.0]],
        }
    }
    if map_raw is not None:
        calib["map"] = map_raw
    (d / "calib.json").write_text(json.dumps(calib), encoding="utf-8")
    poses = [
        {"frame": i, "x": i * 3.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        for i in range(n)
    ]
    (d / "ego_pose.json").write_text(json.dumps(poses), encoding="utf-8")
    return d


class TestMapLabel:
    def test_handles_all_three_recorded_forms(self):
        """★ 三种形态都取 basename —— 两个采集器 + 平铺大图各一种。

        写成"去掉 `Carla/Maps/` 前缀"会在平铺大图上剩 `Town13/Town13`,
        拼出来的文件名不存在,而症状只是"找不到矢量 json"。
        """
        assert assemble_maptr.map_label("Town13") == "Town13"
        assert assemble_maptr.map_label("Carla/Maps/Town10HD_Opt") == "Town10HD_Opt"
        assert assemble_maptr.map_label("Carla/Maps/Town13/Town13") == "Town13"


class TestResolveMapJson:
    def test_explicit_spec_is_passed_through_untouched(self, tmp_path):
        """显式给文件 ⇒ **原样放行**(不做任何 basename/目录推导)。"""
        spec = str(tmp_path / "whatever.json")
        got = assemble_maptr.resolve_map_json(spec, {"map": "Ignored"}, "seg0", tmp_path)
        assert got == Path(spec)

    def test_auto_resolves_both_map_field_forms(self, tmp_path):
        """★ `Town13` 与 `Carla/Maps/Town05_Opt` 都要能推对文件名。"""
        for raw, want in (("Town13", "Town13"), ("Carla/Maps/Town05_Opt", "Town05_Opt")):
            _map_json(tmp_path / f"{want}_full.json", want)
            got = assemble_maptr.resolve_map_json(assemble_maptr.AUTO, {"map": raw}, "seg0", tmp_path)
            assert got.name == f"{want}_full.json"

    def test_missing_map_field_raises(self, tmp_path):
        """★ 缺 `map` **必须报错,不许沿用上一段** —— 用错图的 GT 只表现为"学不动"。"""
        with pytest.raises(SystemExit, match="缺 `map` 字段"):
            assemble_maptr.resolve_map_json(assemble_maptr.AUTO, {}, "seg0", tmp_path)

    def test_absent_vector_json_raises_with_the_path(self, tmp_path):
        """矢量 json 不存在时报错要**带上路径** —— 否则只看到"找不到",不知它在找哪儿。"""
        with pytest.raises(SystemExit, match="Town09_full.json"):
            assemble_maptr.resolve_map_json(assemble_maptr.AUTO, {"map": "Town09"}, "seg0", tmp_path)


class TestAssemble:
    def test_auto_and_explicit_are_byte_identical(self, tmp_path):
        """★ **旧命令零改动**的判据:池内只有一张图时,`auto` 与显式给那一份的产物相同。

        这一条挡的是"为了支持多图顺手把单图路径改了" —— 那种改法让所有已归档
        `kitti_static_*` / `surround_*` 的复现命令静默产出不同的 infos。
        """
        segs = tmp_path / "pool"
        _seg(segs, "seg0", map_raw="Town05_Opt")
        _seg(segs, "seg1", map_raw="Carla/Maps/Town05_Opt")
        mj = _map_json(tmp_path / "Town05_Opt_full.json", "Town05_Opt")
        roots = assemble_maptr.roots_and_prefixes(None, str(segs))

        auto, _ = assemble_maptr.assemble(roots, map_spec=assemble_maptr.AUTO, radius=51.2, map_dir=tmp_path)
        explicit, _ = assemble_maptr.assemble(roots, map_spec=str(mj), radius=51.2, map_dir=tmp_path)
        assert auto == explicit

    def test_data_path_prefix_follows_the_segment_name(self, tmp_path):
        """`--segs-dir` 下 `data_path` 必须带 `segK/` —— 不带就让所有段指向同一批文件。"""
        segs = tmp_path / "pool"
        _seg(segs, "seg0", map_raw="Town05_Opt")
        _map_json(tmp_path / "Town05_Opt_full.json", "Town05_Opt")
        infos, _ = assemble_maptr.assemble(
            assemble_maptr.roots_and_prefixes(None, str(segs)),
            map_spec=assemble_maptr.AUTO,
            radius=51.2,
            map_dir=tmp_path,
        )
        assert infos[0]["cams"]["CAM_FRONT"]["data_path"] == "seg0/cam_front/000000.png"

    def test_each_segment_is_tagged_with_its_own_map(self, tmp_path):
        """多图池的**事后溯源**:`seg → 图` 必须逐段对得上。"""
        segs = tmp_path / "pool"
        _seg(segs, "seg0", map_raw="Town10HD_Opt")
        _seg(segs, "seg1", map_raw="Town05_Opt")
        _map_json(tmp_path / "Town10HD_Opt_full.json", "Town10HD_Opt")
        _map_json(tmp_path / "Town05_Opt_full.json", "Town05_Opt")
        infos, seg_map = assemble_maptr.assemble(
            assemble_maptr.roots_and_prefixes(None, str(segs)),
            map_spec=assemble_maptr.AUTO,
            radius=51.2,
            map_dir=tmp_path,
        )
        assert seg_map == {"seg0": "Town10HD_Opt", "seg1": "Town05_Opt"}
        assert len(infos) == 6  # 两段 × 3 帧

    def test_a_vector_json_is_read_once_per_map(self, tmp_path, monkeypatch):
        """★ 矢量 json **按路径缓存**:同一张图的段共用一份。

        `Town13_full.json` 有 412 MB —— 逐段重读会让组装从秒级变成分钟级,
        而"变慢"不会报错,只会让人以为图太大没办法。
        """
        segs = tmp_path / "pool"
        for k in range(4):
            _seg(segs, f"seg{k}", map_raw="Town05_Opt")
        _map_json(tmp_path / "Town05_Opt_full.json", "Town05_Opt")

        calls: list[Path] = []
        real = assemble_maptr.vecs_load

        def counting(text: str):
            calls.append(Path("counted"))
            return real(text)

        monkeypatch.setattr(assemble_maptr, "vecs_load", counting)
        assemble_maptr.assemble(
            assemble_maptr.roots_and_prefixes(None, str(segs)),
            map_spec=assemble_maptr.AUTO,
            radius=51.2,
            map_dir=tmp_path,
        )
        assert len(calls) == 1, f"同一张图被读了 {len(calls)} 次"

    def test_wrong_frame_in_vector_json_raises(self, tmp_path):
        """坐标系必须是 `carla_world` —— 别的系会让 GT 整体错位而**照样出图**。"""
        segs = tmp_path / "pool"
        _seg(segs, "seg0", map_raw="Town05_Opt")
        pts = tuple((float(x), 5.0, 0.0) for x in range(-20, 21, 2))
        (tmp_path / "Town05_Opt_full.json").write_text(
            vecs_dump((MapVec("boundary", pts, (), "b0", "t"),), "Town05_Opt", frame="ego"), "utf-8"
        )
        with pytest.raises(SystemExit, match="carla_world"):
            assemble_maptr.assemble(
                assemble_maptr.roots_and_prefixes(None, str(segs)),
                map_spec=assemble_maptr.AUTO,
                radius=51.2,
                map_dir=tmp_path,
            )


class TestSameRigGuard:
    """★ `MapTRDataset` 只认 **首帧那一份 calib** —— 混 rig 是"不报错的错"的典型。

    这一组是那个守卫的**反向自证**:喂一份人工造坏的池子,它必须**抛**;
    喂正常的池子,它必须**不抛**。少了反向那一半,"守卫绿了"可能只是它从没被走到。
    """

    def test_same_rig_pool_passes(self, tmp_path):
        """★ 同 rig 的多段池必须**放行** —— 而且它的 `data_path` 是逐段不同的。

        这条同时钉住本守卫第一版的 bug:那时它整份 `cams` 比,而 `data_path` 带段名
        前缀(`seg0/…` vs `seg1/…`)**必然不同** ⇒ **正确的池子被判成"混 rig"而停住**。
        判据必须只取 `(sensor2ego, intrinsic)`。
        """
        segs = tmp_path / "pool"
        _seg(segs, "seg0", map_raw="Town05_Opt", fx=FX_NEW)
        _seg(segs, "seg1", map_raw="Town05_Opt", fx=FX_NEW)
        _map_json(tmp_path / "Town05_Opt_full.json", "Town05_Opt")
        infos, _ = assemble_maptr.assemble(
            assemble_maptr.roots_and_prefixes(None, str(segs)),
            map_spec=assemble_maptr.AUTO,
            radius=51.2,
            map_dir=tmp_path,
        )
        assert len(infos) == 6
        # 前提:两段的 data_path 确实不同(否则这条测试什么也没证明)
        assert infos[0]["cams"]["CAM_FRONT"]["data_path"] != infos[3]["cams"]["CAM_FRONT"]["data_path"]
        assert infos[0]["cams"] != infos[3]["cams"], "整份 cams 本就不同 —— 所以不能整份比"
        assert assemble_maptr.rig_of(infos[0]["cams"]) == assemble_maptr.rig_of(infos[3]["cams"])

    def test_mixed_rig_pool_is_rejected(self, tmp_path):
        """★ 反向自证:两段 fx 不同(旧 621 / 新 1266)⇒ **必须抛,且点名是哪两段**。"""
        segs = tmp_path / "pool"
        _seg(segs, "seg0", map_raw="Town10HD_Opt", fx=FX_NEW)
        _seg(segs, "seg1", map_raw="Town05_Opt", fx=FX_OLD)
        _map_json(tmp_path / "Town10HD_Opt_full.json", "Town10HD_Opt")
        _map_json(tmp_path / "Town05_Opt_full.json", "Town05_Opt")
        with pytest.raises(SystemExit) as ei:
            assemble_maptr.assemble(
                assemble_maptr.roots_and_prefixes(None, str(segs)),
                map_spec=assemble_maptr.AUTO,
                radius=51.2,
                map_dir=tmp_path,
            )
        msg = str(ei.value)
        assert "seg1" in msg and "seg0" in msg, f"报错必须点名是哪两段,实际:{msg}"

    def test_rig_of_calib_and_rig_of_agree(self, tmp_path):
        """采集侧(`calib.json`)与组装侧(`infos.cams`)必须给出**同一形状**的 rig ——
        两张输入要能过同一份比较(逐段快速检查跑在组装之前,吃的是前者)。"""
        seg = _seg(tmp_path, "seg0", map_raw="Town05_Opt")
        calib = json.loads((seg / "calib.json").read_text(encoding="utf-8"))
        assert assemble_maptr.rig_of_calib(calib) == {
            "CAM_FRONT": (
                calib["CAM_FRONT"]["sensor2ego"],
                calib["CAM_FRONT"]["intrinsic"],
            )
        }
        assert set(assemble_maptr.rig_of_calib(calib)) == {"CAM_FRONT"}, "非 CAM_* 的溯源键不许混进来"

    def test_mixed_rig_is_caught_before_reading_the_vector_json(self, tmp_path, monkeypatch):
        """★ 便宜的检查必须排在**贵的加载之前**(2026-10-03 由真实数据的负向对照发现)。

        矢量 json 动辄几百 MB(`Town13_full.json` 有 **412 MB**)—— 原来把 rig 检查
        放在全部段组装**之后**,混 rig 要先把所有 json 读完才报错:实测那样跑 **300 s
        都出不来**(超时被杀),看起来像"卡死"而不是"数据错了"。
        ⇒ 现在逐段比、且在 `resolve_map_json` 之前。这条钉住那个顺序。
        """
        segs = tmp_path / "pool"
        _seg(segs, "seg0", map_raw="Town10HD_Opt", fx=FX_NEW)
        _seg(segs, "seg1", map_raw="Town05_Opt", fx=FX_OLD)
        _map_json(tmp_path / "Town10HD_Opt_full.json", "Town10HD_Opt")
        _map_json(tmp_path / "Town05_Opt_full.json", "Town05_Opt")

        real = assemble_maptr.vecs_load
        reads: list[int] = []

        def counting(text: str):
            reads.append(1)
            return real(text)

        monkeypatch.setattr(assemble_maptr, "vecs_load", counting)
        with pytest.raises(SystemExit, match="rig 不一致"):
            assemble_maptr.assemble(
                assemble_maptr.roots_and_prefixes(None, str(segs)),
                map_spec=assemble_maptr.AUTO,
                radius=51.2,
                map_dir=tmp_path,
            )
        assert len(reads) == 1, f"首段的 json 读一次就够;实际读了 {len(reads)} 次 —— 第二段的没拦住"

    def test_guard_compares_every_frame_not_just_segment_heads(self, tmp_path):
        """比的是**每一帧**:段内串帧同样让模型看到错的内参,而它比"每段首帧"更便宜不到哪去。"""
        segs = tmp_path / "pool"
        _seg(segs, "seg0", map_raw="Town05_Opt")
        bad = _seg(segs, "seg1", map_raw="Town05_Opt")
        _map_json(tmp_path / "Town05_Opt_full.json", "Town05_Opt")
        infos, _ = assemble_maptr.assemble(
            assemble_maptr.roots_and_prefixes(None, str(segs)),
            map_spec=assemble_maptr.AUTO,
            radius=51.2,
            map_dir=tmp_path,
        )
        assert len(infos) == 6

        # 只把 seg1 的**最后一帧**改坏 —— 首帧仍与 seg0 相同
        poisoned = [dict(r) for r in infos]
        last = poisoned[-1]
        last["cams"] = json.loads(json.dumps(last["cams"]))
        last["cams"]["CAM_FRONT"]["intrinsic"][0][0] = FX_OLD
        with pytest.raises(SystemExit, match="rig 不一致"):
            assemble_maptr.assert_same_rig(poisoned)
        assert bad.exists()  # 夹具确实建出来了(不做无意义的自证)

"""mapvec_pred/1 契约:往返、硬校验、越窗计数、与模型/GT 常量的锚定。

契约的用途是**跨项目**(AutoLabel 侧不读本项目代码即可消费),所以这里的断言
当"接口守卫"用:字段名/类序/点数/坐标系一旦漂移必须红。
"""

from __future__ import annotations

import inspect
import json
import math

import pytest

from autodrivedata.map import mapvec
from autodrivedata.map import mapvec_schema as mvs


def _inst(cls: str = "divider", dx: float = 0.0, score: float | None = 0.5) -> mvs.MapVecInstance:
    pts = [(float(i) + dx, float(i % 3)) for i in range(mvs.NUM_POINTS)]
    return mvs.make_instance(cls, pts, score)


def _rec(n_pred: int = 2, n_gt: int = 1, token: str = "000123") -> mvs.MapVecFramePred:
    return mvs.MapVecFramePred(
        frame=123,
        token=token,
        score_thr=0.2,
        ckpt="outputs/maptr_ep512.pt",
        preds=tuple(_inst(dx=0.1 * i) for i in range(n_pred)),
        gts=tuple(_inst(score=None, dx=0.05 * i) for i in range(n_gt)),
    )


class TestContractSurface:
    def test_schema_id_and_constants_match_gt_side(self):
        # 类序/窗口只有一个来源(mapvec 纯值层),契约层不得另立一份
        assert mvs.MAPTR_CLASSES == mapvec.MAPTR_CLASSES
        assert tuple(mvs.BEV_RANGE) == mapvec.BEV_RANGE
        assert mvs.SCHEMA_ID == "mapvec_pred/1"
        assert mvs.COORD == "ego"

    def test_num_points_matches_model_default(self):
        """契约声明 20 点;模型默认值漂移则文件与消费者口径脱节。"""
        from autodrivedata.map.maptr.model import MapTR

        assert inspect.signature(MapTR.__init__).parameters["num_pts"].default == mvs.NUM_POINTS

    def test_top_level_keys_are_exact(self):
        assert set(mvs.frame_to_dict(_rec())) == {
            "schema",
            "frame",
            "token",
            "classes",
            "coord",
            "bev_range",
            "num_points",
            "score_thr",
            "ckpt",
            "preds",
            "gts",
        }

    def test_instance_keys_and_gt_has_no_score(self):
        d = mvs.frame_to_dict(_rec())
        assert set(d["preds"][0]) == {"class", "points", "score"}
        assert set(d["gts"][0]) == {"class", "points"}  # GT 无置信度

    def test_points_are_rounded_for_stable_diffs(self):
        inst = mvs.make_instance("boundary", [(1.2345678, -9.87654321), *[(0.0, 0.0)] * 19])
        assert mvs.inst_to_dict(inst)["points"][0] == [1.235, -9.877]


class TestRoundTrip:
    def test_dict_roundtrip_is_stable(self):
        d1 = mvs.frame_to_dict(_rec())
        d2 = mvs.frame_to_dict(mvs.frame_from_dict(json.loads(json.dumps(d1))))
        assert d1 == d2

    def test_dump_and_load(self, tmp_path):
        rec = _rec(token="000007")
        path = mvs.dump_frame(rec, tmp_path / "out")
        assert path == tmp_path / "out" / "000007.json"  # 文件名 = {token}.json
        assert mvs.load_frame(path) == rec

    def test_token_with_different_frames_do_not_collide(self, tmp_path):
        a, b = _rec(token="000001"), _rec(token="000002")
        assert mvs.dump_frame(a, tmp_path) != mvs.dump_frame(b, tmp_path)
        assert len(list(tmp_path.glob("*.json"))) == 2


class TestHardValidation:
    @pytest.mark.parametrize(
        ("mutate", "frag"),
        [
            (lambda d: d.update(schema="mapvec_pred/2"), "schema 不匹配"),
            (lambda d: d.update(coord="global"), "coord 不符"),
            (lambda d: d.update(num_points=10), "num_points 不符"),
            (
                lambda d: d.update(classes=["divider", "boundary", "ped_crossing", "centerline"]),
                "classes 顺序不符",
            ),
        ],
    )
    def test_rejects_wrong_header(self, mutate, frag):
        d = mvs.frame_to_dict(_rec())
        mutate(d)
        with pytest.raises(ValueError, match=frag):
            mvs.frame_from_dict(d)

    def test_rejects_unknown_class(self):
        rec = _rec()
        bad = mvs.MapVecFramePred(
            frame=rec.frame,
            token=rec.token,
            score_thr=0.2,
            ckpt=rec.ckpt,
            preds=(mvs.make_instance("stop_line", [(0.0, 0.0)] * mvs.NUM_POINTS, 0.5),),
            gts=(),
        )
        with pytest.raises(ValueError, match="未知类"):
            mvs.validate_frame(bad)

    def test_rejects_wrong_point_count(self):
        bad = mvs.MapVecFramePred(
            frame=0,
            token="000000",
            score_thr=0.2,
            ckpt="c",
            preds=(mvs.make_instance("divider", [(0.0, 0.0)] * 7, 0.5),),
            gts=(),
        )
        with pytest.raises(ValueError, match="点数"):
            mvs.validate_frame(bad)

    @pytest.mark.parametrize("score", [-0.01, 1.01])
    def test_rejects_score_out_of_range(self, score):
        bad = mvs.MapVecFramePred(
            frame=0,
            token="000000",
            score_thr=0.2,
            ckpt="c",
            preds=(_inst(score=score),),
            gts=(),
        )
        with pytest.raises(ValueError, match="score 越界"):
            mvs.validate_frame(bad)

    def test_rejects_missing_score_and_non_finite_points(self):
        no_score = mvs.MapVecFramePred(
            frame=0, token="0", score_thr=0.2, ckpt="c", preds=(_inst(score=None),), gts=()
        )
        with pytest.raises(ValueError, match="缺 score"):
            mvs.validate_frame(no_score)
        nan = mvs.make_instance("divider", [(math.nan, 0.0)] * mvs.NUM_POINTS, 0.5)
        bad = mvs.MapVecFramePred(frame=0, token="0", score_thr=0.2, ckpt="c", preds=(nan,), gts=())
        with pytest.raises(ValueError, match="非有限数"):
            mvs.validate_frame(bad)

    def test_dump_refuses_invalid_record(self, tmp_path):
        """写前校验:坏记录不落盘(否则消费方拿到半截契约)。"""
        bad = mvs.MapVecFramePred(
            frame=0, token="000000", score_thr=0.2, ckpt="c", preds=(_inst(score=2.0),), gts=()
        )
        with pytest.raises(ValueError):
            mvs.dump_frame(bad, tmp_path)
        assert list(tmp_path.glob("*.json")) == []
        assert mvs.dump_frame(_rec(token="000001"), tmp_path).is_file()  # 好记录照常落盘


class TestWindowDiagnostics:
    def test_out_of_window_counts_pred_only(self):
        """pred 无裁剪(实测可越窗)、GT 恒在窗内 → 只诊断,不拒收。"""
        xmin, ymin, xmax, ymax = mvs.BEV_RANGE
        inside = [(0.0, 0.0)] * (mvs.NUM_POINTS - 2)
        outside = [(xmax + 1.0, 0.0), (0.0, ymin - 1.0)]
        rec = mvs.MapVecFramePred(
            frame=0,
            token="000000",
            score_thr=0.2,
            ckpt="c",
            preds=(mvs.make_instance("divider", inside + outside, 0.5),),
            gts=(mvs.make_instance("divider", inside + [(xmin, ymax), (0.0, 0.0)], None),),
        )
        mvs.validate_frame(rec)  # 越窗不报错
        assert mvs.out_of_window(rec) == 2
        assert mvs.gt_out_of_window(rec) == 0

    def test_out_of_window_flags_gt_leak(self):
        """GT 越窗 = 上游 clip_to_bev 被绕过(应恒为 0),计数必须能抓到。"""
        rec = mvs.MapVecFramePred(
            frame=0,
            token="000000",
            score_thr=0.2,
            ckpt="c",
            preds=(),
            gts=(mvs.make_instance("boundary", [(0.0, 99.0)] * mvs.NUM_POINTS, None),),
        )
        assert mvs.gt_out_of_window(rec) == mvs.NUM_POINTS

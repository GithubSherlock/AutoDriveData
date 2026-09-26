"""A′ 口径复算:同一份逐帧预测/GT,并排算"我们的 chamfer AP"与"官方 eval_map"。

Plan.md §5.12 的探查结论:官方 AP(`map_utils/mean_ap.py`)是**按 score 排序 →
累加 tp/fp → 算 recalls/precisions → PR 曲线积分**,**含 recall 项**;我们的
`autodrivedata.chamfer_ap` 是 3 阈值(0.5/1.0/1.5m)precision 均值、**无 recall 项**。
两者不是同一口径,本脚本量化差值并逐项归因(重采样 / recall 项 / score 排序)。

做法:读 `mapvec_pred/1` 逐帧产物 → 转成官方两个 json:
  - GT:`{"GTs": [{"sample_token": …, "vectors": [{"pts", "pts_num", "cls_name", "type"}]}]}`
  - pred:`{"meta": …, "results": [{…, "vectors": [{"pts", "pts_num", "cls_name",
           "type", "confidence_level"}]}]}`
两边**按序号对齐**(官方实现的对齐键是 `sample_id` 序号,不是 token:见
`_evaluate_single` 里 `assert len(gen_results) == len(annotations)`),因此输出文件
必须由同一份、同一顺序的 records 生成,本脚本一次生成两份以杜绝错位。

用法(在 maptr_official env 下,官方代码只读引用、不修改):
  PYTHONPATH=$PWD /root/autodl-tmp/envs/maptr_official/bin/python -m autodrivedata.map.eval_official_metric \
      --pred-dir outputs/surround_pred --repo hdMapGitHub/MapTR \
      --out outputs/maptr_official/a_prime

**语法下限 = py3.8**(脚本要在两个 env 里都跑:autodrivedata 是 3.11,官方 env 是 3.8)。
故禁用 3.9+ 语法:`zip(strict=)` / `dict|dict` 之类注解都靠 `from __future__ import
annotations` 兜住,但**运行时**构造(如 `zip(strict=True)`)在 3.8 直接 TypeError —— 用
`_pair_names` 代替。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np

from autodrivedata.map.chamfer_ap import chamfer_ap_per_class, chamfer_cost_matrix
from autodrivedata.map.mapvec import BEV_RANGE
from autodrivedata.map.mapvec_schema import load_frame
from autodrivedata.utils.paths import project_path

DEFAULT_CLASSES = ("divider", "ped_crossing", "boundary", "centerline")

_DEPS_READY = False


def _ensure_official_deps() -> None:
    """让官方 AP 模块在没有 mmcv/mmdet 的环境里也能导入(如 autodrivedata py3.11)。

    A′ 只跑实测口径,**被测代码一行不改**:被替换掉的只有 `mmcv.Timer` / `mmcv.dump` /
    `mmcv.mkdir_or_exist` / `mmcv.utils.print_log` 四个日志落盘工具,以及 chamfer 分支
    永不触达的 `bbox_overlaps`(只有 metric='iou' 会调)。真栈(maptr_official env)
    里这些包齐全,本函数是 no-op。幂等:两个口径各调一次,第二次直接返回。
    """
    global _DEPS_READY
    if _DEPS_READY:
        return

    class _Timer:
        def __init__(self) -> None:
            self._t0 = self._last = time.time()

        def since_start(self) -> float:
            return time.time() - self._t0

        def since_last_check(self) -> float:
            now = time.time()
            dt, self._last = now - self._last, now
            return dt

    def _dump(obj, path, **kwargs) -> None:
        path = str(path)
        if path.endswith(".pkl"):
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        else:
            Path(path).write_text(json.dumps(obj), encoding="utf-8")

    def _stub(name: str, **attrs) -> types.ModuleType:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    if importlib.util.find_spec("mmcv") is None:
        utils = _stub("mmcv.utils", print_log=lambda msg, logger=None: None)
        _stub(
            "mmcv",
            Timer=_Timer,
            dump=_dump,
            mkdir_or_exist=lambda p: Path(p).mkdir(parents=True, exist_ok=True),
            utils=utils,
        )

    if importlib.util.find_spec("terminaltables") is None:

        class _AsciiTable:
            """官方只用它渲染逐类汇总表;本脚本 print_log 为 no-op,表格不进产出。"""

            def __init__(self, data) -> None:
                self.data = data

            @property
            def table(self) -> str:
                return "\n".join("\t".join(str(c) for c in row) for row in self.data)

        _stub("terminaltables", AsciiTable=_AsciiTable)

    if importlib.util.find_spec("mmdet") is None:
        _stub("mmdet")
        _stub("mmdet.core")
        _stub("mmdet.core.evaluation")
        _stub(
            "mmdet.core.evaluation.bbox_overlaps",
            bbox_overlaps=lambda *a, **k: (_ for _ in ()).throw(
                NotImplementedError("bbox_overlaps 仅在官方 metric='iou' 分支使用,A′ 走 chamfer")
            ),
        )

    _DEPS_READY = True


def _ensure_shapely1_semantics() -> None:
    """官方 chamfer 代码按 **shapely 1.x** 语义用 STRtree:

    `tpfp_chamfer.py:48` 是 `for o in tree.query(pline): if o.intersects(pline)`,再靠
    `index_by_id[id(o)]` 反查 pred 下标 —— 即 `query()` 必须返回**几何对象**。shapely 2.x
    改成返回**索引数组** → 官方代码在 2.x 下直接 AttributeError。

    这里把返回值映射回几何:2.x 的 `tree.geometries()` 与索引同序,且返回的是构造时传入的
    同一批对象(`id()` 语义与 1.x 一致)。真栈 maptr_official env 里 pin 的是
    shapely==1.8.5(官方 requirements 口径),本函数是 no-op。
    """
    import shapely
    from shapely.strtree import STRtree

    if shapely.__version__.startswith("1.") or getattr(STRtree, "_a_prime_shapely1_compat", False):
        return
    _orig_query = STRtree.query

    def query(self, geometry, *args, **kwargs):
        res = _orig_query(self, geometry, *args, **kwargs)
        if isinstance(res, np.ndarray) and res.dtype.kind in "iu":
            geoms = self.geometries  # 2.0 是属性;个别版本是方法
            if callable(geoms):
                geoms = geoms()
            return [geoms[int(i)] for i in res]
        return res

    STRtree.query = query  # type: ignore[method-assign]
    STRtree._a_prime_shapely1_compat = True  # type: ignore[attr-defined]


def _load_official_mean_ap(repo: Path):
    """按**文件路径**加载官方 `map_utils.mean_ap`,不执行沿途的 `__init__.py`。

    官方的 `projects/mmdet3d_plugin/__init__.py` 会 `import` 整个模型栈(匈牙利匹配器、
    mmdet builder…),而 A′ 只用到 `map_utils/` 这一支纯 numpy/shapely 的评测代码。
    做法:把沿途包注册成带 `__path__` 的空模块(等于绕开 `__init__`),再让 Python
    正常解析叶子模块——`mean_ap.py` 内部的 `from .tpfp import …` 因此照常工作。
    """
    base = repo / "projects" / "mmdet3d_plugin" / "datasets"
    for name, path in (
        ("projects", repo / "projects"),
        ("projects.mmdet3d_plugin", repo / "projects" / "mmdet3d_plugin"),
        ("projects.mmdet3d_plugin.datasets", base),
        ("projects.mmdet3d_plugin.datasets.map_utils", base / "map_utils"),
    ):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules.setdefault(name, pkg)
    return importlib.import_module("projects.mmdet3d_plugin.datasets.map_utils.mean_ap")


def _pair_names(names, results):
    """`zip(names, results)` + 长度校验。

    不用 `zip(..., strict=True)`:本脚本要在**官方 env(py3.8)**里复算,而 `strict=` 是
    py3.10 才加的(3.8 直接 `TypeError: zip() takes no keyword arguments`)。语义照旧 ——
    数量不符必须炸,不能静默截断。
    """
    if len(names) != len(results):
        raise RuntimeError(f"官方返回 {len(results)} 项,期望 {len(names)} 项:{list(names)}")
    return zip(names, results)  # noqa: B905 —— strict= 是 py3.10 才有的(见上);长度已自查


def _resample(points: list[list[float]], n: int) -> list[list[float]]:
    """折线等弧长重采样 n 点(与官方 `get_cls_results` 的 shapely 口径一致)。"""
    from shapely.geometry import LineString

    line = LineString(points)
    if line.length == 0:
        return [list(points[0])] * n
    dists = np.linspace(0.0, line.length, n)
    return [list(line.interpolate(float(d)).coords[0]) for d in dists]


def build_payloads(records, cls_names: tuple[str, ...], resample: int | None):
    """records(帧序)→ (官方 GT json, 官方 pred json);重采样只影响几何,不改类/分值。

    `cls_names` 是**取子集**用的:产物可能有多类(如 4 类模型的预测 + 4 类 GT),
    而对照 v1/MapQR 只比共享 3 类。类外实例在此处丢弃 —— 与官方
    `format_res_gt_by_classes`(按 cls_names 的**索引**过滤)同机制,故两边等价。
    """
    idx = {name: i for i, name in enumerate(cls_names)}
    gts, preds = [], []
    for rec in records:

        def vec(inst, with_score: bool) -> dict:
            pts = [[x, y] for x, y in inst.points]
            if resample:
                pts = _resample(pts, resample)
            d = {"pts": pts, "pts_num": len(pts), "cls_name": inst.cls, "type": idx[inst.cls]}
            if with_score:
                d["confidence_level"] = float(inst.score)
            return d

        def keep(i) -> bool:  # 类外实例不进 payload(见 docstring)
            return i.cls in idx

        gts.append({"sample_token": rec.token, "vectors": [vec(i, False) for i in rec.gts if keep(i)]})
        preds.append({"sample_token": rec.token, "vectors": [vec(i, True) for i in rec.preds if keep(i)]})
    return {"GTs": gts}, {"meta": {"use_camera": True}, "results": preds}


def official_map(
    repo: Path,
    out_dir: Path,
    gt_payload: dict,
    pred_payload: dict,
    cls_names: tuple[str, ...],
    pc_range: list[float],
    flag: bool,
    nproc: int,
) -> dict:
    """调官方 eval_map(3 阈值 0.5/1.0/1.5m);返回逐阈值/逐类 AP 与 mAP。"""
    _ensure_official_deps()
    _ensure_shapely1_semantics()
    mean_ap = _load_official_mean_ap(repo)
    eval_map, format_res_gt_by_classes = mean_ap.eval_map, mean_ap.format_res_gt_by_classes

    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "official_results.json"
    res_path.write_text(json.dumps(pred_payload), encoding="utf-8")
    (out_dir / "official_gts.json").write_text(json.dumps(gt_payload), encoding="utf-8")

    gen_results, annotations = pred_payload["results"], gt_payload["GTs"]
    cls_gens, cls_gts = format_res_gt_by_classes(
        str(res_path),
        gen_results,
        annotations,
        cls_names=list(cls_names),
        num_pred_pts_per_instance=20,
        eval_use_same_gt_sample_num_flag=flag,
        pc_range=pc_range,
    )
    per_thr: dict[str, dict] = {}
    for thr in (0.5, 1.0, 1.5):
        mAP, results = eval_map(
            gen_results,
            annotations,
            cls_gens,
            cls_gts,
            threshold=thr,
            cls_names=list(cls_names),
            logger="silent",
            pc_range=pc_range,
            metric="chamfer",
            num_pred_pts_per_instance=20,
            nproc=nproc,
        )
        per_thr[f"{thr:.1f}"] = {
            "mAP": float(mAP),
            "ap": {r_cls: float(res["ap"]) for r_cls, res in _pair_names(cls_names, results)},
            "num_gts": int(sum(res["num_gts"] for res in results)),
        }
    overall = float(np.mean([v["mAP"] for v in per_thr.values()]))
    # 逐类 AP 同样取跨阈值均值 —— 与官方 `_evaluate_single` 里 `cls_aps.mean(0)` 的上报口径一致
    ap_mean = {c: float(np.mean([per_thr[t]["ap"][c] for t in per_thr])) for c in cls_names}
    return {"mAP": overall, "ap": ap_mean, "per_threshold": per_thr}


def our_map(records, cls_names: tuple[str, ...]) -> dict:
    """我们自己的 chamfer AP(无 recall 项、无重采样)——同输入对照。"""
    preds = [
        [np.asarray(i.points, dtype=float) for r in records for i in r.preds if i.cls == c] for c in cls_names
    ]
    gts = [
        [np.asarray(i.points, dtype=float) for r in records for i in r.gts if i.cls == c] for c in cls_names
    ]
    aps, mAP = chamfer_ap_per_class(preds, gts, cost_fn=chamfer_cost_matrix)
    return {
        "mAP": float(mAP),
        "ap": {c: float(a) for c, a in _pair_names(cls_names, aps)},
        "num_pred": {c: len(p) for c, p in _pair_names(cls_names, preds)},
        "num_gt": {c: len(g) for c, g in _pair_names(cls_names, gts)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="mapvec_pred/1 逐帧产物目录")
    ap.add_argument("--repo", default="hdMapGitHub/MapTR", help="官方仓库根(只读引用)")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help="类序(须与产物类序一致)")
    ap.add_argument("--out", default="outputs/maptr_official/a_prime", help="产出目录(项目内)")
    ap.add_argument("--nproc", type=int, default=8, help="官方 eval_map 的进程数")
    ap.add_argument("--num-sample", type=int, default=100, help="官方口径重采样点数")
    args = ap.parse_args()

    pred_dir = Path(project_path(args.pred_dir))
    repo = Path(project_path(args.repo))
    out_dir = Path(project_path(args.out))
    cls_names = tuple(args.classes.split(","))
    records = sorted((load_frame(p) for p in pred_dir.glob("*.json")), key=lambda r: r.frame)
    if not records:
        raise SystemExit(f"未找到逐帧产物:{pred_dir}")
    thr_set = {r.score_thr for r in records}
    # 官方形态 = 6 元 [xmin, ymin, zmin, xmax, ymax, zmax](z 沿用官方 config 口径)。
    # 实测评测链里 pc_range 只在被 `if False:` 关掉的目检块用到 → 不参与裁剪,故不存在
    # "官方把越窗点裁回窗口"这一差值来源。
    pc_range = [BEV_RANGE[0], BEV_RANGE[1], -2.0, BEV_RANGE[2], BEV_RANGE[3], 2.0]
    print(f"[data] {len(records)} 帧 × {len(cls_names)} 类;score_thr={thr_set};pc_range={pc_range}")

    summary: dict = {
        "pred_dir": str(pred_dir),
        "repo": str(repo),
        "classes": list(cls_names),
        "frames": len(records),
        "score_thr": sorted(thr_set),
        "pc_range": pc_range,
    }
    for tag, flag in (("official_flag_raw", False), ("official_flag_resample", True)):
        gt_payload, pred_payload = build_payloads(records, cls_names, args.num_sample if flag else None)
        summary[tag] = official_map(
            repo, out_dir / tag, gt_payload, pred_payload, cls_names, pc_range, flag, args.nproc
        )
    summary["ours_raw20"] = our_map(records, cls_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metric_compare.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== A′ 口径对照(同一份 ep512 逐帧产物)===")
    print(f"{'口径':32s} {'mAP':>8s}  " + "  ".join(f"{c:>10s}" for c in cls_names))
    for tag, label in (
        ("ours_raw20", "我们 chamfer AP(20 点 raw)"),
        ("official_flag_raw", "官方 eval_map(20 点 raw)"),
        ("official_flag_resample", f"官方 eval_map({args.num_sample} 点重采样)"),
    ):
        d = summary[tag]
        print(f"{label:32s} {d['mAP']:8.4f}  " + "  ".join(f"{d['ap'][c]:10.4f}" for c in cls_names))
    print(f"\n[out] {out_dir}/metric_compare.json")


if __name__ == "__main__":
    main()

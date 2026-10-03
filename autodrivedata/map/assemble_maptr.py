"""B2:数据集组装器——环视采集 + 地图矢量 → MapTRv2 infos 同构 json。

消费 B1 的 surround root(6 视角图像 + calib.json + ego_pose.json)与 A 阶段的
全图矢量 json,逐帧组装 MapTRv2 `_fill_trainval_infos` 同构的核心字段:
cams(每相机 data_path/sensor2ego/intrinsic)+ 帧级 ego2global + annotation
(四类矢量 GT,ego 局部系,先 ±radius 预裁剪、再按官方口径裁到 BEV 60×30m
训练窗口后 20 点重采样)。

口径注记:MapTRv2 官方 annotation 在 **LiDAR 局部系**(lidar2global 变换);
本管道无 LiDAR,用 **ego 局部系**(A5 `to_ego_frame` 口径,与 lidar 局部系
同为车辆参考系,训练消费端无差别——lidar2ego 恒等)。

用法:
  python -m autodrivedata.map.assemble_maptr --surround outputs/surround_drive \
      --map-json training/map/Town10HD_Opt_full.json \
      --out outputs/surround_drive/map_infos.json

## `--map-json auto`:**把图池扩到第二张图**(2026-10-03)

`--segs-dir` 收的是**一个父目录下的 `seg*`**,而 `--map-json` 原来只吃**一个**文件
⇒ 两张图的段进不了同一个 infos。`auto` 让矢量 json **逐段**由该段自己的
`calib.json["map"]` 推,于是"两张图的段放进同一个父目录"就能直接合成一个池子:

```
outputs/multimap/
  seg0..seg4  -> ../../surround_v2_epic/seg{0..4}       # Town10HD_Opt
  seg5..seg9  -> ../../surround_town05_epic/seg{0..4}   # Town05_Opt
python -m autodrivedata.map.assemble_maptr --segs-dir outputs/multimap \
    --map-json auto --out outputs/multimap/map_infos.json
```

⚠️ **显式给文件时行为逐字节不变** —— 旧命令一字不改(池内只有一张图时,`auto`
与显式给那一份的产物**逐字节相同**,有回归钉)。

⚠️ **同一份矢量 json 只读一次**:`Town13_full.json` 有 412 MB,逐段重读会很慢。

## ★ 池内 rig 必须同源(`assert_same_rig`)

`MapTRDataset.__init__` 是 `self.calibs = self.infos[0]["cams"]` —— **从首帧取一份
给全池共用**。混了两种 rig 的池子**不会抛异常**,只会拿 A 图的内参去投影 B 图的像素
⇒ 症状是"训练学不动",而数据看着没毛病。所以组装收尾逐段比一次 `cams`,不等就停下并
**点名是哪两段**(同 `select_frames` 段名拼错要报错不静默的纪律)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autodrivedata.map.mapvec import (
    BEV_RANGE,
    clip_to_bev,
    crop_to_ego,
    resample,
    to_ego_frame,
    to_maptr_annotation,
    vecs_load,
)
from autodrivedata.utils.paths import project_path

#: `--map-json` 的自动取值:逐段由该段 `calib.json` 的 `map` 推 `{map_dir}/{name}_full.json`。
AUTO = "auto"
#: `auto` 的默认查找目录。**经 `project_path` 锚定项目根**(与 `--out` 同口径)——
#: 这里有别于"读路径不锚定"的惯例,因为路径**不是用户给的**,而是由 calib 里的
#: 地图名推出来的:用户没有机会用 cwd 表达意图,不锚定就只会在别的 cwd 下静默找不到。
MAP_DIR = "training/map"


def roots_and_prefixes(surround: str | None, segs_dir: str | None) -> list[tuple[Path, str]]:
    """`(--surround X)` → `[(X, "")]`;`(--segs-dir D)` → `[(D/segK, "segK/"), …]`。

    `data_path` 的段名前缀只在这里决定(见 `_seg_infos` 的 path_prefix 说明)——
    单独抽出来是为了能**不依赖地图矢量 json** 单测这条契约:写成"恒定带前缀"
    会让所有旧单段命令静默指错文件,而那种错在训练时表现为"图看着没错、学不动"。
    """
    if (surround is None) == (segs_dir is None):
        raise ValueError("--surround 与 --segs-dir 必须且只能给一个")
    if segs_dir is not None:
        roots = sorted(p for p in Path(segs_dir).glob("seg*") if p.is_dir())
        if not roots:
            raise ValueError(f"{segs_dir} 下没有 seg* 子目录")
        return [(r, f"{r.name}/") for r in roots]  # --root = 父目录
    assert surround is not None  # 上面已排除两者同时为 None
    return [(Path(surround), "")]  # --root = 段目录本身(旧用法不变)


def map_label(raw: str) -> str:
    """`calib.json` 的 `map` 字段 → 地图标签。**三种形态都要认**。

    两个采集器落的**不是同一种东西**,而平铺大图还多一层:

    | 来源 | `map` 的值 | `Path(...).name` |
    |---|---|---|
    | `collect_surround`(落 `args.map`) | `Town13` | `Town13` |
    | `collect_surround_lidar`(落 `world.get_map().name`) | `Carla/Maps/Town10HD_Opt` | `Town10HD_Opt` |
    | 同上的平铺大图 | `Carla/Maps/Town13/Town13` | `Town13` |

    取 basename 三种都对。**不要**写成"去掉 `Carla/Maps/` 前缀" —— 平铺大图那层
    会剩下 `Town13/Town13`,拼出来的文件名不存在,而症状是"找不到矢量 json"。
    """
    return Path(raw).name


def resolve_map_json(spec: str, calib: dict, seg: str, map_dir: Path) -> Path:
    """`--map-json` 的取值:显式文件**原样放行**,`auto` 由该段的 calib 推。

    `auto` 下缺 `map` 字段**必须报错,不许沿用上一段** —— 用错图的 GT 在下游只表现为
    **学不动**(矢量与像素对不上,损失降不下去),看不出是这里错的。
    """
    if spec != AUTO:
        return Path(spec)
    raw = calib.get("map")
    if not raw:
        raise SystemExit(
            f"段 {seg!r} 的 calib.json 缺 `map` 字段 —— `--map-json auto` 靠它解析矢量 json。"
            f"(该字段由采集器写入;手工造的 root 要自己补)"
        )
    p = map_dir / f"{map_label(raw)}_full.json"
    if not p.is_file():
        raise SystemExit(f"段 {seg!r} 的地图 {raw!r} ⇒ 找不到 {p}(用 export_mapvec 生成全图矢量)")
    return p


#: 混 rig 的报错尾巴 —— 两处(逐段快速检查 / 末尾逐帧复查)共用一份,免得两处口径漂
_RIG_HINT = (
    "MapTRDataset 只用 infos[0]['cams'] 一份标定去投影**所有**段 —— 混 rig 不报错,只让训练学不动。\n"
    "  确认各段是**同一代 rig**采的(盘上 2026-09-16 前后两代:"
    "旧 == fx 621/1242×375,新 == 逐相机 fx≈1260/1600×900)。"
)


def rig_of(cams: dict) -> dict:
    """infos 的 `cams` → **属于 rig 的那部分** —— 每相的 `(sensor2ego, intrinsic)`。

    ⚠️ **不能整份 `cams` 比**:里面的 `data_path` 带段名前缀(`seg0/cam_front/…`
    vs `seg1/cam_front/…`),逐段**必然不同** —— 那是 `--segs-dir` 的设计,不是 rig 不一致。
    本守卫的第一版就是整份比,当场把**正确的**多段池判成"混 rig"而停住
    (被 `test_same_rig_pool_passes` 抓到)。判据取"错在哪"而不是"哪里不同":
    用错标定的后果只由这两项决定,`data_path` 不参与投影。
    """
    return {n: (c["sensor2ego"], c["intrinsic"]) for n, c in cams.items()}


def rig_of_calib(calib: dict) -> dict:
    """`calib.json` → **同一形状**的 rig(采集侧只有 `CAM_*` 是相机键,其余是溯源 meta)。

    与 `rig_of` 同形是为了**同一份比较**能同时吃采集侧与组装侧的输入 ——
    逐段快速检查必须在组装之前跑(见 `assemble` 里那条顺序注释)。
    """
    return {n: (c["sensor2ego"], c["intrinsic"]) for n, c in calib.items() if n.startswith("CAM_")}


def assert_rig_matches(got: dict, seg: str, head_seg: str, head: dict) -> None:
    """单次比较:`seg` 与首段 `head_seg` 的 rig 不等就停。**与首段同名时直接放行。**"""
    if seg != head_seg and got != head:
        raise SystemExit(f"池内 rig 不一致:段 {seg!r} 的相机标定与段 {head_seg!r} 的不同。\n  " + _RIG_HINT)


def assert_same_rig(infos: list[dict]) -> None:
    """★ 对**已组装**的 infos 逐帧复查:与首段不等就停,并点名是哪两段。

    **为什么必须有**:`MapTRDataset.__init__` 的 `self.calibs = self.infos[0]["cams"]`
    —— **从首帧取一份给全池共用**。混了两种 rig 的池子不会抛异常,只会拿 A 图的内参
    投影 B 图的像素 ⇒ 训练"学不动",而数据看着没毛病。

    比的是**每一帧**而不只是每段首帧:段内串帧同样会让模型看到错的内参,
    而多比 500 次字典相等是免费的。

    ⚠️ 它**不是唯一的关卡**:`assemble` 里还有一道**逐段、且在贵加载之前**的同类检查。
    两道都留着 —— 这道是"组装结果对不对",那道是"别等读完 412 MB 才报错"
    (实测:只留末尾这道时,混 rig 的负向对照跑 300 s 都报不出来)。
    """
    if not infos:
        return
    head = infos[0]
    for r in infos:
        assert_rig_matches(rig_of(r["cams"]), r["seg"], head["seg"], rig_of(head["cams"]))


def assemble(
    roots: list[tuple[Path, str]],
    *,
    map_spec: str,
    radius: float,
    map_dir: Path,
) -> tuple[list[dict], dict[str, str]]:
    """逐段组装 → `(infos, {段名: 地图标签})`。

    矢量 json **按路径缓存** —— 同一张图的段共用一份(412 MB 的 Town13 逐段重读会很慢)。
    """
    cache: dict[Path, tuple] = {}
    seg_map: dict[str, str] = {}
    infos: list[dict] = []
    head_seg: str | None = None
    head_rig: dict = {}
    for root, prefix in roots:
        calib = json.loads((root / "calib.json").read_text(encoding="utf-8"))
        poses = json.loads((root / "ego_pose.json").read_text(encoding="utf-8"))
        # ★ 顺序是刻意的:**便宜的比较排在贵的加载之前**。矢量 json 动辄几百 MB
        #   (Town13 有 412 MB),排在末尾统一比的话,混 rig 要先把全部 json 读完才报错 ——
        #   实测那样跑 300 s 都出不来。数据本来就是错的,越早停越好。
        raw_rig = rig_of_calib(calib)
        if head_seg is None:
            head_seg, head_rig = root.name, raw_rig
        else:
            assert_rig_matches(raw_rig, root.name, head_seg, head_rig)
        mj = resolve_map_json(map_spec, calib, root.name, map_dir)
        if mj not in cache:
            _, frame, vecs = vecs_load(mj.read_text(encoding="utf-8"))
            if frame != "carla_world":
                raise SystemExit(f"地图矢量坐标系应为 carla_world,实际 {frame}({mj})")
            cache[mj] = vecs
        seg_map[root.name] = mj.stem.removesuffix("_full")
        infos += _seg_infos(calib, poses, cache[mj], radius, root.name, len(infos), prefix)
    assert_same_rig(infos)
    return infos, seg_map


def _seg_infos(
    calib: dict, poses: list[dict], vecs, radius: float, seg: str, base: int, path_prefix: str
) -> list[dict]:
    """单段的 infos(帧号全局递增;`seg`/`frame_in_seg` 供留出划分与时序窗口用)。

    `path_prefix` 让 `data_path` 始终**相对调用方给的 `--root`**:单段(`--surround`)
    时 root 就是段目录、前缀为空(与旧产物逐字节等价);多段(`--segs-dir`)时 root
    是父目录、前缀为 `segK/`。写成"恒定带段名前缀"会让所有旧命令静默指错文件。
    """
    out: list[dict] = []
    for p in poses:
        i, gi = p["frame"], base + p["frame"]
        ego = (p["x"], p["y"])
        local = to_ego_frame(crop_to_ego(vecs, ego, radius=radius), p["x"], p["y"], p["yaw"])
        # MapTR 官方口径:GT 裁剪到 BEV 训练窗口(60×30m)后再 20 点重采样
        local = tuple(resample(v, 20) for v in clip_to_bev(local, BEV_RANGE))
        out.append(
            {
                "frame": gi,
                "seg": seg,
                "frame_in_seg": i,
                # token 必须**全局唯一**:eval_maptr --out-frames 用它做文件名,
                # 多段共用 {i:06d} 会互相覆盖(静默丢帧)
                "token": f"{seg}_{i:06d}",
                "ego2global": [p["x"], p["y"], p["z"], p["yaw"], p["pitch"], p["roll"]],
                "cams": {
                    name: {
                        # 图像路径:多段合并后各段图像在各自目录下,
                        # 不带段名会让所有段都指向同一批文件(参数键不同却同名)
                        "data_path": f"{path_prefix}{name.lower()}/{i:06d}.png",
                        "sensor2ego": c["sensor2ego"],
                        "intrinsic": c["intrinsic"],
                    }
                    # 过滤 meta 键(collect_surround 在 calib 顶层写 "map"/"spawn_index" 等溯源;
                    # 相机键统一 CAM_* 前缀,见 SURROUND_CAMS)
                    for name, c in calib.items()
                    if name.startswith("CAM_")
                },
                "annotation": to_maptr_annotation(local),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surround", default=None, help="B1 单段采集根目录(旧用法;与 --segs-dir 二选一)")
    ap.add_argument(
        "--segs-dir", default=None, help="B1 多段父目录:取其中的 seg* 子目录(与 --surround 二选一)"
    )
    ap.add_argument(
        "--map-json",
        required=True,
        help="A4 全图矢量 json;**`auto` = 逐段由该段 `calib.json` 的 `map` 推**"
        "`{--map-dir}/{name}_full.json`(多图池用这个;单图时与显式给该文件逐字节相同)",
    )
    ap.add_argument(
        "--map-dir",
        default=MAP_DIR,
        help=f"`--map-json auto` 的查找目录(默认 {MAP_DIR},经 project_path 锚定项目根)",
    )
    ap.add_argument("--out", required=True, help="输出 infos json")
    ap.add_argument("--radius", type=float, default=51.2, help="ego 窗口裁剪半径(方形)")
    args = ap.parse_args()
    args.out = str(project_path(args.out))  # 产物锚定项目根(相对路径不随 cwd 漂移)
    try:
        roots = roots_and_prefixes(args.surround, args.segs_dir)
    except ValueError as e:
        ap.error(str(e))

    infos, seg_map = assemble(
        roots,
        map_spec=args.map_json,
        radius=args.radius,
        map_dir=project_path(args.map_dir),
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=1)
    n_ann = {k: len(v) for k, v in infos[0]["annotation"].items()}
    per_seg = {r.name: sum(1 for i in infos if i["seg"] == r.name) for r, _ in roots}
    maps = sorted(set(seg_map.values()))
    print(f"{args.out}:{len(infos)} 帧 {per_seg}(首帧 annotation {n_ann})")
    # 多图池时把「哪段是哪张图」打出来 —— 池子一旦混了图,这是唯一的事后溯源
    print(f"  地图:{maps}" + (f"  段→图 {seg_map}" if len(maps) > 1 else ""))


if __name__ == "__main__":
    main()

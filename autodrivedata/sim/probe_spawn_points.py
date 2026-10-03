"""多段采集的 spawn point 选点:贪心最大最小距离(farthest point sampling)。

## 为什么需要它

多段采集的**段间起点必须尽量远** —— 两段走同一条街,扩的是帧数不是**路线多样性**,
而 MapTR 线要的恰恰是后者(CLAUDE.md:「矢量从 xodr 解析、不依赖外观;**要的是拓扑多样性**」)。

`surround_v2_epic` 的 `44/14/15/152/55` 就是这么选的(milestone2:「贪心最大最小距离选
spawn point……两两最近 114 m」),但**当时是现算的、没留成工具** ⇒ 换一张图就得再算一遍,
而"再算一遍"如果实现与当时不是同一种贪心,新旧两批段的**选点口径就不可比**。

## ★ 种子规则是**反推出来的**(2026-10-03)

贪心对**种子**敏感(第一个点从哪来会改变整组结果),而当时的**起始规则没留档**。
反推过程:在 `Town10HD_Opt`(155 个 spawn point)上逐个试候选规则,只有
「**种子 = 离全部点质心最远的那个**」(→ 44)能**逐位复现** `44/14/15/152/55`
(两两最近 **113.994 m**,milestone2 记的 114 m 就是它)。其余候选(max x / min y /
最大模长 / 种子 0)都不命中;种子 55 能给出更大的 117.2 m 但**不是**当时那组。

⇒ 规则固化在 `route.farthest_from_centroid`,**新图沿用同一条**,新旧两批段口径才可比。
`--seed-index` 只在要显式覆盖时才给。

## 自证

`--expect`:默认规则必须复现那五个索引。复现不了 ⇒ 实现与当时不是同一种贪心,
用它给 Town13 选的段**没有依据**,要先对齐口径再采。对不上时会**遍历所有种子**
报出最接近的一组 —— 不做"差不多就算过"。

算法本身在 [`route.py`](route.py)(零 carla,可单测);本模块只做 CARLA 编排:
取 spawn point → 转 xy → 交给纯函数 → 打印。

用法:
  # 自证:复现 v2_epic 的五个点(需要 CARLA 在跑)
  python -m autodrivedata.sim.probe_spawn_points --map Town10HD_Opt --k 5 \
      --expect 44,14,15,152,55
  # 选新图(分期的段起点)
  python -m autodrivedata.sim.probe_spawn_points --map Town13 --k 5 --out outputs/town13_spawns.json

⚠️ 本探针**只读**(不 spawn、不 tick、不清场):`get_spawn_points()` 是地图查询,
不经过 actor 快照,所以不受「首个 `get_actors()` 为空」那条坑影响(那条管的是 actor,
这条不读 actor)。唯一副作用是 `--map` 会 `load_world`(~2 min,会重置世界)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import carla

from autodrivedata.sim.route import (
    farthest_from_centroid,
    greedy_maxmin,
    greedy_maxmin_order,
    min_pairwise,
    spread_curve,
)
from autodrivedata.utils.paths import project_path

#: `--expect` 未过时「遍历所有种子」诊断的点数上限。遍历是 O(N²·k²),
#: Town10HD_Opt(155 个点)秒出;Town13 有 12478 个 ⇒ 会跑到天荒地老。
#: 自证本来就只有老图(选点规则反推的那张)才做得了,大图明说跳过而不是挂死。
MAX_SEED_SCAN = 2000


def spawn_xy(world: carla.World) -> tuple[list[tuple[float, float]], list[carla.Transform]]:
    """地图的 spawn point → `(xy 列表, 原始 Transform 列表)`。**只读地图,不碰 actor。**"""
    pts = world.get_map().get_spawn_points()
    return [(p.location.x, p.location.y) for p in pts], list(pts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=None, help="目标地图(如 Town13);缺省 = 不动当前图")
    ap.add_argument("--k", type=int, default=5, help="要选几个点(= 采集几段)")
    ap.add_argument(
        "--seed-index",
        type=int,
        default=-1,
        help="贪心的种子点。**默认 -1 = 离质心最远**,即反推出来的那条规则"
        "(它在 Town10HD_Opt 上逐位复现了 v2_epic 的选点)。给非负值 = 显式覆盖",
    )
    ap.add_argument(
        "--expect",
        default=None,
        help="逗号分隔的已知索引(如 `44,14,15,152,55`):默认规则**必须**复现它。"
        "对不上时遍历所有种子报出最接近的一组 —— 不做『差不多就算过』",
    )
    ap.add_argument("--min-gap", type=float, default=100.0, help="判据:两两最近距离下限(米)")
    ap.add_argument("--out", default=None, help="可选:把选定索引落 json(经 project_path)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    if args.map:
        print(f"[map] load_world {args.map}(运行时切图,~2min)...")
        world = client.load_world(args.map)
    else:
        world = client.get_world()
    xy, spawns = spawn_xy(world)
    print(f"[map] {world.get_map().name}:{len(xy)} 个 spawn point")

    # ---- 自证模式:默认规则必须复现;对不上才遍历种子做诊断 ----
    if args.expect is not None:
        want = sorted(int(x) for x in args.expect.split(","))
        if len(want) != args.k:
            ap.error(f"--expect 给了 {len(want)} 个索引,与 --k {args.k} 不符")
        got = greedy_maxmin(xy, args.k, farthest_from_centroid(xy))
        if got == want:
            print(f"\n✅ 自证通过:默认规则(离质心最远)复现出 {got}")
            print(f"   两两最近 {min_pairwise(xy, got):.3f} m")
            return
        print(f"\n❌ 自证未过:默认规则给出 {got},与期望的 {want} 不符")
        # 遍历诊断是 O(N²·k²):Town10HD_Opt(155)秒出,但大图(Town13 有 12478 个)会跑到天荒地老。
        # 自证本来就不该在大图上做 —— 与其挂死,不如明说跳过了。
        if len(xy) > MAX_SEED_SCAN:
            print(f"   跳过遍历诊断:{len(xy)} 个点超过上限 {MAX_SEED_SCAN}(遍历是 O(N²) 量级)")
            raise SystemExit(1)
        best: tuple[int, float, list[int]] | None = None  # (种子, 最近间距, 选中集)
        for s in range(len(xy)):
            g2 = greedy_maxmin(xy, args.k, s)
            if g2 == want:
                print(f"   (但种子 {s} 能复现 —— 说明默认规则被改过,见 route.farthest_from_centroid)")
                raise SystemExit(1)
            gap2 = min_pairwise(xy, g2)
            if best is None or gap2 > best[1]:
                best = (s, gap2, g2)
        assert best is not None
        s, gap2, got2 = best
        print(f"   遍历 {len(xy)} 个种子,**没有一个**复现 {want};最接近(种子 {s}):{got2}")
        print("   ⇒ 实现与当时选 44/14/15/152/55 的**不是同一种**。先对齐口径再拿它选新图。")
        raise SystemExit(1)

    # ---- 选点模式 ----
    seed = args.seed_index if args.seed_index >= 0 else farthest_from_centroid(xy)
    how = "显式指定" if args.seed_index >= 0 else "离质心最远(默认规则)"
    sel = greedy_maxmin(xy, args.k, seed)
    gap = min_pairwise(xy, sel)
    print(f"\n选中 {len(sel)} 个 —— 种子 {seed}({how})")
    # 选择序也打出来:milestone2 记的「44/14/15/152/55」是**选择序**(首元素 = 种子),
    # 拿它与 sorted 后的集合比对会读成"对不上"
    print(f"   选择顺序:{' → '.join(str(i) for i in greedy_maxmin_order(xy, args.k, seed))}")
    print(f"   升序集合:{' '.join(str(i) for i in sel)}")
    print(f"\n  {'idx':>5}{'x':>10}{'y':>10}{'yaw':>8}   到已选集合最近距离")
    chosen: list[tuple[float, float]] = []
    for i in sel:
        d = min((min_pairwise([xy[i], c], [0, 1]) for c in chosen), default=float("inf"))
        print(f"  {i:>5}{xy[i][0]:>10.1f}{xy[i][1]:>10.1f}{spawns[i].rotation.yaw:>8.1f}   {d:>8.1f}")
        chosen.append(xy[i])

    print("\n  最近间距曲线(k → 两两最近距离,米;掉得陡=这张图路网本来就不大):")
    for kk, g in spread_curve(xy, args.k, seed):  # ⚠️ 传**解析后**的 seed,不是 args.seed_index
        print(f"    k={kk}  {g:>8.1f}" + ("   ← 本 k" if kk == args.k else ""))

    ok = gap >= args.min_gap
    print(
        f"\n判据:两两最近 {gap:.1f} m vs 下限 {args.min_gap:.0f} m —— "
        + ("通过" if ok else "**不通过**:段间太近,缩 k 或换图(v2_epic 实测 114 m)")
    )
    print("\n采集命令(每段一条,`--spawn-index` 依次取):" + " ".join(str(i) for i in sel))

    if args.out:
        out = project_path(args.out)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(
            json.dumps(
                {
                    "map": world.get_map().name,
                    "k": len(sel),
                    "seed_index": args.seed_index,
                    "indices": sel,
                    "min_pairwise_m": gap,
                    "points": [
                        {"index": i, "x": xy[i][0], "y": xy[i][1], "yaw": spawns[i].rotation.yaw} for i in sel
                    ],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"[out] {out}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

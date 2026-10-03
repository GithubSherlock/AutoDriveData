"""静态道具(锥桶等)**能不能出 GT** 的实测探针 —— 一次把三件事量完(需 CARLA 服务器)。

## 起因

`collect_ab_route.py` 建 label_2 时有一行显式过滤:

    if not (a.type_id.startswith("vehicle") or a.type_id.startswith("walker")):
        continue

锥桶(`static.prop.*`)是被这行**主动跳过**的,不是查不到。但它到底是"设计上不要",
还是"能力上要不到",那行代码本身答不了。本探针回答三个问题:

| # | 问题 | 判据 |
|---|---|---|
| ① | 地图里**烘死**的静态道具在不在 `get_actors()` 里 | 世界 `static.prop.*` actor 普查(零 spawn 时先数) |
| ② | 摆出来的锥桶,`bounding_box` 读得准吗 | 逐 yaw 读回 vs **yaw=0 探针**读回,并各自投影与**渲染轮廓**对拍 |
| ③ | 语义相机把锥桶打成哪个 tag | 实例掩膜内像素的 tag 直方图(两代相机各一次) |

## ② 为什么必须拿"渲染轮廓"当裁判,而不是看读回来的数

`Actor.bounding_box` 对**转过**的 actor 给出 `(extent, rotation)` 自相矛盾的读数
(实测同一个 streetbarrier:yaw=0 读 `1.215×0.372`、yaw=80.07° 读 `0.157×1.261`)。
但"读出来的数不一样"还不等于"GT 框是错的" —— 有可能错的那份投影出来照样包住物体。
**唯一能裁决的是渲染**:实例分割相机给出锥桶的**真实像素轮廓**,把两种读法各自
`box_to_gt_line` 投影到同一张图上,谁的框包不住轮廓,谁就是错的。

判据三条(全数值):
- `coverage` = 轮廓像素落在投影框内的比例(**应当 ≈1.0**;包不住的框会掉下来)
- `tightness` = 投影框面积 / 轮廓外接矩形面积(**应当 ≈1.0**;过大的框会鼓起来)
- `iou` = 投影框 vs 轮廓外接矩形

## ③ 为什么顺带量 tag

`probe_calib.decode_instance` 头注记着"实例相机 R 通道 = `CityObjectLabel`,
锥/桶 = 21 `Dynamic`" —— 那是**实例相机**上观察到的。P2-A 用的是**语义相机**
(`collect_surround --sem`),两代相机是否同源**没人验过**。而 `Dynamic`(21) 既不在
P2-A 的三个 GT 类里、也不在 `EXCLUDED_TAGS` 里 ⇒ 若真如此,锥桶像素落在
**第三个没被报出来的桶**里,模型把它预测成 obstacle 就白记一次 FP。

用法:
    python -m autodrivedata.sim.probe_static_prop_gt            # Town10HD_Opt
    python -m autodrivedata.sim.probe_static_prop_gt --map Town01
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import carla
import cv2
import numpy as np

from autodrivedata.calib.core import CameraIntrinsics
from autodrivedata.calib.probe_calib import decode_instance
from autodrivedata.calib.rig_check import CAM_H, CAM_W, RigCameras
from autodrivedata.gt.core import ActorBox, box_to_gt_line, classify_kitti, classify_nus
from autodrivedata.gt.export.nuscenes import camera_fov
from autodrivedata.perception.sem_tags import SEM_TAGS
from autodrivedata.sim.carla_common import (
    ground_z_at,
    loc,
    measure_actor_size_yaw0,
    rad,
    spawn_ego,
    sync_mode,
)
from autodrivedata.sim.collect_surround import tag_from_semantic_image
from autodrivedata.sim.occlusion import OCCLUDER_MODELS
from autodrivedata.utils.paths import project_path

RIG = "nuscenes"
#: 静态道具蓝图(`static.prop.*` 通用,不写死某一个 —— 有的图没有 constructioncone)
CONE_BP_CANDIDATES = (
    "static.prop.constructioncone",
    "static.prop.trafficcone01",
    "static.prop.trafficcone02",
)
#: 目标车/相机在 ego 系里的前移量(与 collect_ab_route.CAM_FWD 同源含义)
CAM_FWD = 1.2
#: 逐锥的 (前向距离 m, 横向偏移 m, yaw 偏角 deg) —— yaw 拉开就是为了踩 ② 那个坑
CONE_SLOTS = ((6.0, -0.90, 0.0), (8.0, -0.30, 30.0), (10.0, 0.30, 60.0), (12.0, 0.90, 90.0))
MIN_MASK_PX = 30  # 轮廓像素少于此 ⇒ 该行**作废**(不是"通过")——判据没有样本就不是判据


def _count_static_props(world: carla.World) -> dict[str, int]:
    """普查世界里的 `static.prop.*` actor(**spawn 任何东西之前**调用)。"""
    out: dict[str, int] = {}
    for a in world.get_actors():
        if a.type_id.startswith("static.prop"):
            out[a.type_id] = out.get(a.type_id, 0) + 1
    return out


def _level_bbs_census(
    world: carla.World, near_xy: tuple[float, float], radius: float = 30.0
) -> dict[str, Any]:
    """`get_level_bbs` 普查:**烘在关卡里的**静态几何(不是 actor)有哪些。

    与 `_count_static_props` 是两回事,必须都问:
    - `static.prop.*` actor = 本会话 spawn 出来的道具(可查 `get_transform`)
    - level BBS        = 关卡自带网格(楼房/路面/也可能是地图自带的锥桶/路障)

    ⚠️ **第一版这个函数报了个不可解读的数**(66600 个盒里"62884 个最大边长 < 1.5 m"),
    看着像"满地都是锥桶"。真相是 `get_level_bbs` 返回的是**整张图每个网格的包围盒**,
    绝大多数是路面/路缘这类碰撞基元 —— "最大边 < 1.5 m" 对它们恒真。所以改成
    **按道具尺寸带 (0.2–1.2 m) 且落在 ego 附近 30 m 内**来数,这才对应"看得见的摆设"。
    """
    try:
        bbs = world.get_level_bbs(carla.CityObjectLabel.Any)
    except (RuntimeError, TypeError) as e:  # pycarla 桩/版本差异:拿不到就如实报,不假装 0
        return {"error": f"{type(e).__name__}: {e}"}
    has_tag = any(hasattr(b, "tag") for b in bbs[:64])
    n_near_prop = 0
    for bb in bbs:
        c = bb.location
        if (c.x - near_xy[0]) ** 2 + (c.y - near_xy[1]) ** 2 > radius**2:
            continue
        side = 2.0 * max(bb.extent.x, bb.extent.y, bb.extent.z)
        if 0.2 <= side <= 1.2:
            n_near_prop += 1
    return {
        "n_total": len(bbs),
        f"n_prop_sized_within_{int(radius)}m": n_near_prop,
        "has_tag_attr": has_tag,
    }


def _tag_attribution(
    world: carla.World, cams: RigCameras, ego_t: carla.Transform, bp_ids: tuple[str, ...]
) -> list[dict[str, Any]]:
    """逐个资产摆一台、读它落在哪个语义 tag —— 回答"类掩膜 GT 该取哪些 tag"。

    为什么必须实测而不是查文档:CARLA 的 `CityObjectLabel` 只给了**名字**
    (`Static`/`Dynamic`/`Other`),谁是谁**没有任何规格说明**,而名字本身还会误导
    (锥桶是静态道具,却被打成 `Dynamic`)。三个 tag 的边界只能靠"摆一个已知资产、
    看它落在哪"来钉。
    """
    placed: list[carla.Actor] = []
    fails: list[dict[str, Any]] = []
    for i, bid in enumerate(bp_ids):
        bp = world.get_blueprint_library().find(bid)
        if bp is None:
            # 「蓝图不存在」与「spawn 失败」是**两件事**。合并成一个"未测"会把
            # "本图根本没这个资产"藏起来,而两者的处置完全不同(换资产 vs 挪位置)。
            fails.append({"type_id": bid, "status": "no_blueprint"})
            continue
        lat = (i - (len(bp_ids) - 1) / 2.0) * 1.5  # 横向拉开,掩膜不互串
        w = ego_t.transform(carla.Location(x=7.0, y=lat, z=0.0))
        gz = ground_z_at(world, w.x, w.y, ego_t.location.z)
        a = world.try_spawn_actor(bp, carla.Transform(carla.Location(x=w.x, y=w.y, z=gz), carla.Rotation()))
        if a is None:
            fails.append({"type_id": bid, "status": "spawn_failed", "at": [round(w.x, 2), round(w.y, 2)]})
            continue
        placed.append(a)
    # ⚠️ **不许在这里裸敲 `world.tick()`** —— 传感器队列是 FIFO,`cams.step()` 取的是
    # **最旧**那一帧。外面多 tick 一次就多积一帧,`step()` 交回来的就是"道具还没出生"
    # 的那帧 ⇒ 掩膜恒 0,症状是"spawn 成功但镜头里数不到轮廓"(2026-10-01 实测)。
    # `RigCameras` 的纪律是**每次 tick 都走 `step()`**,读回的新鲜度由它保证。
    frames = cams.step()
    inst = frames["instance_segmentation"]["CAM_FRONT"]
    sem = frames["semantic_segmentation"]["CAM_FRONT"]
    ids = decode_instance(inst.raw_data, CAM_H, CAM_W)
    cls_ch = np.frombuffer(inst.raw_data, dtype=np.uint8).reshape(CAM_H, CAM_W, 4)[:, :, 2].copy()
    tag = tag_from_semantic_image(sem)
    name_of = {v: k for k, v in SEM_TAGS.items()}

    out: list[dict[str, Any]] = list(fails)
    for a in placed:
        mask = ids == a.id
        n = int(mask.sum())
        if n == 0:
            out.append({"type_id": a.type_id, "status": "no_mask"})
            continue
        iv, ic = np.unique(cls_ch[mask], return_counts=True)
        tv, tc = np.unique(tag[mask], return_counts=True)
        out.append(
            {
                "type_id": a.type_id,
                "status": "ok",
                "mask_px": n,
                "instance_seg_R_tag": int(iv[int(np.argmax(ic))]),
                "semantic_cam_tag": int(tv[int(np.argmax(tc))]),
                "tag_name": name_of.get(int(tv[int(np.argmax(tc))]), "?"),
                "semantic_tag_share": round(float(tc.max() / tc.sum()), 4),
                "consistent": bool(len(iv) == 1 and len(tv) == 1),
            }
        )
    for a in placed:
        a.destroy()
    cams.step()  # 抽干(销毁也是一次世界变化,不 drain 会把这一帧留给下一次 capture)
    return out


#: 语义 tag 里的"杂项"三兄弟。名字是 CARLA 给的,**没有任何规格说明**,而名字还会
#: 误导(锥桶是静态道具,却被打成 `Dynamic`)⇒ 边界只能靠实测钉(见 `_tag_attribution`)。
MISC_TAGS = ("Static", "Dynamic", "Other")


def _tag_morphology(tag: np.ndarray, names: tuple[str, ...] = MISC_TAGS) -> dict[str, Any]:
    """杂项 tag 的**连通域形态** —— 判断它们是"物体"还是"杂散像素"。

    只报像素数是不够的:6999 个 `Dynamic` 像素既可能是几百个道具,也可能是一圈抗锯齿
    边缘。**连通域尺寸**才分得开这两件事,而"这到底是不是物体"决定了能不能叫它静态道具。
    """
    out: dict[str, Any] = {}
    for name in names:
        m = (tag == SEM_TAGS[name]).astype(np.uint8)
        n = int(m.sum())
        if n == 0:
            out[name] = {"px": 0}
            continue
        ncc, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        areas = np.sort(stats[1:, cv2.CC_STAT_AREA])[::-1]  # 丢掉背景标签 0
        out[name] = {
            "px": n,
            "n_components": int(ncc - 1),
            "largest_px": int(areas[0]),
            "median_px": int(np.median(areas)),
            "n_ge_100px": int((areas >= 100).sum()),
        }
    return out


def _cone_blueprint_id(world: carla.World) -> str:
    """挑一个本图真的有的锥桶蓝图;一个都没有就报错退出(不许静默退化成"没测")。"""
    lib = world.get_blueprint_library()
    for cand in CONE_BP_CANDIDATES:
        if lib.find(cand) is not None:
            return cand
    raise SystemExit(f"蓝图库里没有 {CONE_BP_CANDIDATES} 任何一个 —— 本图无锥桶资产")


Vec3 = tuple[float, float, float]


def _px_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """布尔掩膜 → 像素外接矩形 `(x1, y1, x2, y2)`;空掩膜返回 None。

    ⚠️ 与 GT 侧的**投影框**口径差半格:`box_to_gt_line` 走的是角点 `min/max`,
    这里走的是**像素索引** `min/max`。差 1 px 量级,对 60 px 的尺子可忽略 ——
    但结论要按"这个量级以下不解读"来读。
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _iou_box(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else float("nan")


def _parse_gt_box(line: str) -> tuple[float, float, float, float]:
    """label_2 行 → 2D 框 `(x1, y1, x2, y2)`(列 4–7)。"""
    p = line.split()
    return float(p[4]), float(p[5]), float(p[6]), float(p[7])


def _coverage(mask: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """轮廓像素落在投影框内的比例 —— 框**包不住**物体时这条会掉下来。"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return float("nan")
    inside = (xs >= box[0]) & (xs <= box[2]) & (ys >= box[1]) & (ys <= box[3])
    return float(inside.mean())


def _tightness(box: tuple[float, float, float, float], mask_bbox: tuple[float, float, float, float]) -> float:
    """投影框面积 / 轮廓外接矩形面积 —— **过大的框会鼓起来**(远大于 1)。

    单独成函数而不是闭包:闭包会捕获循环变量,ruff B023 指出来的正是"定义时不绑定 ⇒
    每次调用读到的是**最后一次循环**的值",而那个错**不抛异常、只是一行数字错**。
    """
    a = (box[2] - box[0]) * (box[3] - box[1])
    m = (mask_bbox[2] - mask_bbox[0]) * (mask_bbox[3] - mask_bbox[1])
    return float(a / m) if m > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="Town10HD_Opt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out", default="outputs/probe_static_prop_gt")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.load_world(args.map)
    sync_mode(world)

    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ① 普查:零 spawn 时世界里有什么静态道具(**先数,再 spawn 任何东西**)
    census = _count_static_props(world)
    print(f"[① 普查] {args.map} 世界内 `static.prop.*` actor:{sum(census.values())} 个")
    for tid, n in sorted(census.items()):
        print(f"          {tid} × {n}")
    # ⚠️ actor 数为 0 **不等于**"地图里没有锥桶" —— 烘在关卡里的网格根本不是 actor。
    #    换 `get_level_bbs` 问一遍(在 ego spawn 之后,要按"离 ego 多近"筛)。

    # 清掉地图自带车/行人(否则挡视线),再 spawn ego
    for a in world.get_actors():
        if a.type_id.startswith(("vehicle", "walker", "controller")):
            a.destroy()
    world.tick()
    ego = spawn_ego(world)
    ego0 = loc(ego.get_transform())
    print(f"[ego] spawn @ {tuple(round(v, 2) for v in ego0)}")

    level = _level_bbs_census(world, (ego0[0], ego0[1]))
    print(
        f"[① 关卡几何] `get_level_bbs` 共 {level['n_total']} 个盒"
        f"(每网格一个,绝大多数是路面/路缘的碰撞基元,不是道具);"
        f"其中**道具尺寸带(0.2–1.2 m)且落在 ego 30 m 内**的有 "
        f"{level['n_prop_sized_within_30m']} 个;该 API 带 tag 属性 = {level['has_tag_attr']}"
    )

    bp_id = _cone_blueprint_id(world)
    bp = world.get_blueprint_library().find(bp_id)
    dims0, offs0, rot0 = measure_actor_size_yaw0(world, bp)
    print(
        f"[② 资产] {bp_id} yaw=0 读回 **长×厚×高 = "
        f"{dims0[0]:.4f} × {dims0[1]:.4f} × {dims0[2]:.4f} m**,偏移 {offs0},盒旋转 {rot0}"
    )
    print(f"          classify_kitti → {classify_kitti(bp_id)} | classify_nus → {classify_nus(bp_id)}")

    cams = RigCameras(world, ego, RIG, ("instance_segmentation", "semantic_segmentation"))
    ego_t = ego.get_transform()
    intrinsics = CameraIntrinsics(CAM_W, CAM_H, camera_fov(RIG)["CAM_FRONT"])

    # ★ 基线帧:锥桶 spawn **之前**的画面。地图自带的道具若真存在,这里就该有 tag 21
    #   像素 —— 它是"烘死的道具到底看不看得见"的唯一直接证据(普查只数 actor,数不到网格)。
    base_tag = tag_from_semantic_image(cams.step()["semantic_segmentation"]["CAM_FRONT"])
    bv, bc = np.unique(base_tag, return_counts=True)
    name_of_all = {v: k for k, v in SEM_TAGS.items()}
    base_hist = {
        name_of_all.get(int(t), str(int(t))): int(n)
        for t, n in sorted(zip(bv.tolist(), bc.tolist(), strict=True), key=lambda p: -p[1])
    }
    print(f"[① 基线帧] 锥桶 spawn 前 CAM_FRONT 的 tag 分布(前 8):{dict(list(base_hist.items())[:8])}")
    # ★ 杂项三兄弟:像素数**不够**,还要看连通域 —— "是物体"与"是一圈抗锯齿"在像素数上
    #   长得一样,只有尺寸分布分得开。这决定了它们能不能被叫作静态道具。
    morph = _tag_morphology(base_tag)
    for name in MISC_TAGS:
        s = morph[name]
        if s["px"] == 0:
            print(f"           {name:<8} 0 px")
            continue
        print(
            f"           {name:<8} {s['px']:>7} px | 连通域 {s['n_components']:>5} 个"
            f"(最大 {s['largest_px']:>5} px、中位 {s['median_px']:>3} px、"
            f"≥100 px 的 {s['n_ge_100px']:>3} 个)"
        )

    # 逐锥 spawn(直接按目标 yaw 生成,不依赖 set_transform —— 静态道具能不能挪是另一件事)
    cones: list[carla.Actor] = []
    for d, lat, dyaw in CONE_SLOTS:
        w = ego_t.transform(carla.Location(x=d, y=lat, z=0.0))
        gz = ground_z_at(world, w.x, w.y, ego_t.location.z)
        c = world.try_spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=w.x, y=w.y, z=gz), carla.Rotation(yaw=ego_t.rotation.yaw + dyaw)
            ),
        )
        if c is None:
            print(f"  [warn] 锥桶 spawn 失败 @ d={d} lat={lat} —— 该槽位无数据")
            continue
        cones.append(c)
    # ★ 读回的新鲜度由 `cams.step()` 的那次 tick 提供(**不要**在前面再裸敲一次 tick,
    #   否则队列积一帧、`step()` 交回旧帧 —— 见 `_tag_attribution` 里那段实测记录)
    print(f"[② 摆位] spawn 成功 {len(cones)}/{len(CONE_SLOTS)} 个锥桶")

    frames = cams.step()
    inst = frames["instance_segmentation"]["CAM_FRONT"]
    sem = frames["semantic_segmentation"]["CAM_FRONT"]
    ids = decode_instance(inst.raw_data, CAM_H, CAM_W)
    cls_ch = np.frombuffer(inst.raw_data, dtype=np.uint8).reshape(CAM_H, CAM_W, 4)[:, :, 2].copy()
    tag = tag_from_semantic_image(sem)
    cam_t = cams.sensor("CAM_FRONT").get_transform()
    cam_loc, cam_rot = loc(cam_t), rad(cam_t.rotation)

    rows: list[dict[str, Any]] = []
    print()
    print("[② 投影 vs 渲染轮廓]  coverage/tightness/iou 都应当 ≈1.0")
    print(
        f"  {'id':>6} {'dyaw':>5} {'maskpx':>7} "
        f"{'cov(yaw0)':>10} {'tight(yaw0)':>12} {'iou(yaw0)':>10} | "
        f"{'cov(读回)':>10} {'tight(读回)':>12} {'iou(读回)':>10}"
    )
    for (d, lat, dyaw), cone in zip(CONE_SLOTS, cones, strict=False):
        mask = ids == cone.id
        bb = _px_bbox(mask)
        row: dict[str, Any] = {
            "id": cone.id,
            "dyaw": dyaw,
            "dist": d,
            "lateral": lat,
            "type_id": cone.type_id,
        }
        if bb is None or int(mask.sum()) < MIN_MASK_PX:
            row["status"] = "no_mask"
            print(
                f"  {cone.id:>6} {dyaw:>5.0f} {int(mask.sum()):>7} —— **轮廓不足 {MIN_MASK_PX} px,该行作废**"
            )
            rows.append(row)
            continue

        ct = cone.get_transform()
        actor_loc, actor_rot = loc(ct), rad(ct.rotation)
        cbb = cone.bounding_box

        # 正确读法:尺寸/偏移/盒旋转取 **yaw=0 探针**那份,只让 actor 位姿随朝向变
        box_ok = ActorBox(
            type_id=cone.type_id,
            extent=tuple(v / 2.0 for v in dims0),  # type: ignore[arg-type]
            location=offs0,
            rotation=rot0,
            actor_location=actor_loc,
            actor_rotation=actor_rot,
        )
        # 陷阱读法:原样吃 `Actor.bounding_box` 当前那一读
        box_bad = ActorBox(
            type_id=cone.type_id,
            extent=(cbb.extent.x, cbb.extent.y, cbb.extent.z),
            location=(cbb.location.x, cbb.location.y, cbb.location.z),
            rotation=rad(cbb.rotation),
            actor_location=actor_loc,
            actor_rotation=actor_rot,
        )
        line_ok = box_to_gt_line(box_ok, cam_loc, cam_rot, intrinsics)
        line_bad = box_to_gt_line(box_bad, cam_loc, cam_rot, intrinsics)
        if line_ok is None or line_bad is None:
            row["status"] = "projected_out"
            print(f"  {cone.id:>6} {dyaw:>5.0f} {int(mask.sum()):>7} —— 投影出图,该行作废")
            rows.append(row)
            continue

        b_ok, b_bad = _parse_gt_box(line_ok), _parse_gt_box(line_bad)

        row |= {
            "status": "ok",
            "mask_px": int(mask.sum()),
            "mask_bbox": [round(v, 2) for v in bb],
            "proj_bbox_yaw0": [round(v, 2) for v in b_ok],
            "proj_bbox_read": [round(v, 2) for v in b_bad],
            "read_extent": [
                round(2.0 * cbb.extent.x, 4),
                round(2.0 * cbb.extent.y, 4),
                round(2.0 * cbb.extent.z, 4),
            ],
            "cov_yaw0": round(_coverage(mask, b_ok), 4),
            "tight_yaw0": round(_tightness(b_ok, bb), 4),
            "iou_yaw0": round(_iou_box(b_ok, bb), 4),
            "cov_read": round(_coverage(mask, b_bad), 4),
            "tight_read": round(_tightness(b_bad, bb), 4),
            "iou_read": round(_iou_box(b_bad, bb), 4),
        }
        rows.append(row)
        print(
            f"  {cone.id:>6} {dyaw:>5.0f} {int(mask.sum()):>7} "
            f"{row['cov_yaw0']:>10.4f} {row['tight_yaw0']:>12.4f} {row['iou_yaw0']:>10.4f} | "
            f"{row['cov_read']:>10.4f} {row['tight_read']:>12.4f} {row['iou_read']:>10.4f}"
        )
        # ★ 自证:同一实例的语义类必须唯一(多类 ⇒ 解码口径错,全表作废)
        vals, cnts = np.unique(cls_ch[mask], return_counts=True)
        if len(vals) != 1:
            print(f"        ⚠️ 该实例的语义类不唯一:{dict(zip(vals.tolist(), cnts.tolist(), strict=True))}")
        row["instance_seg_R_tag"] = int(vals[int(np.argmax(cnts))])
        tv, tc = np.unique(tag[mask], return_counts=True)
        row["semantic_cam_tag"] = int(tv[int(np.argmax(tc))])
        row["instance_tag_share"] = round(float(cnts.max() / cnts.sum()), 4)
        row["semantic_tag_share"] = round(float(tc.max() / tc.sum()), 4)

    # ③ 两代相机的 tag 是否同源(逐实例比对)
    print()
    print("[③ 语义 tag] 实例掩膜内的众数 tag")
    name_of = {v: k for k, v in SEM_TAGS.items()}
    for r in rows:
        if r.get("status") != "ok":
            continue
        a, b = r["instance_seg_R_tag"], r["semantic_cam_tag"]
        same = "同" if a == b else "★ 不同"
        print(
            f"  id={r['id']:>6} dyaw={r['dyaw']:>3.0f}  "
            f"实例相机 R={a}({name_of.get(a, '?')},{r['instance_tag_share']:.1%})  "
            f"语义相机={b}({name_of.get(b, '?')},{r['semantic_tag_share']:.1%})  [{same}]"
        )

    # ④ tag 归属:逐个资产摆一台,看它落在哪个 tag(类掩膜 GT 的 tag 集由此定)
    for c in cones:
        c.destroy()
    cones = []  # 已销毁 ⇒ 收尾循环里别再碰(重复 destroy 会抛)
    cams.step()  # 抽干(同 `_tag_attribution` 里那条:裸 tick 会让下一次 capture 读到旧帧)
    print()
    print("[④ tag 归属] 逐个资产摆在 CAM_FRONT 前 7 m,读它的语义 tag")
    # 资产表复用 `occlusion.OCCLUDER_MODELS`(遮挡道具就是静态道具,不另抄一份)
    attr = _tag_attribution(world, cams, ego_t, (bp_id, *OCCLUDER_MODELS.values()))
    why = {
        "no_blueprint": "**本图没有这个蓝图**(资产缺失,不是位姿问题)",
        "spawn_failed": "蓝图在但 spawn 失败(碰撞/位姿)",
        "no_mask": "spawn 成功但镜头里数不到轮廓(被挡?出画?)",
    }
    for r in attr:
        if r.get("status") != "ok":
            print(f"  {r['type_id']:<38} —— {why.get(r['status'], r['status'])}")
            continue
        flag = "" if r["consistent"] else "  ⚠️ 掩膜内 tag 不唯一"
        print(
            f"  {r['type_id']:<38} mask={r['mask_px']:>6} px  "
            f"tag={r['semantic_cam_tag']}({r['tag_name']},{r['semantic_tag_share']:.1%}){flag}"
        )

    # 汇总:两种读法各自的通过率
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    summary: dict[str, Any] = {
        "map": args.map,
        "census_static_props": census,
        "level_bbs": level,
        "baseline_tag_hist": base_hist,
        "baseline_tag_morphology": morph,
        "cone_bp": bp_id,
        "yaw0_dims": [round(v, 4) for v in dims0],
        "n_cones": len(cones),
        "n_valid": len(ok_rows),
        "tag_attribution": attr,
        "rows": rows,
    }
    if ok_rows:
        summary["yaw0_min_coverage"] = round(min(r["cov_yaw0"] for r in ok_rows), 4)
        summary["read_min_coverage"] = round(min(r["cov_read"] for r in ok_rows), 4)
        # ⚠️ **先把 tag 集合取出来再排** —— 不是为了好看:`tests/calib/test_rigviz` 有一条
        #   AST 守卫,扫"`sorted(...)` 的实参里出现 cam"就报警(防**拿字母序当画布序**,
        #   那种错会让拼图第二行左右颠倒)。这里排的是 **tag 名**(Dynamic/…),只因那个
        #   字段叫 `semantic_cam_tag` 就被误伤 —— 把集合提到 `sorted` 外面,守卫的实参
        #   里就没有 `cam` 了。**别"顺手合并回去"**,那会让守卫重新变红,而它是对的。
        inst_tags_seen = {r["instance_seg_R_tag"] for r in ok_rows}
        sem_tags_seen = {r["semantic_cam_tag"] for r in ok_rows}
        summary["tag_names"] = {
            "instance_seg_R": sorted(name_of.get(t, "?") for t in inst_tags_seen),
            "semantic_cam": sorted(name_of.get(t, "?") for t in sem_tags_seen),
        }
        print()
        print(
            f"[结论] 有效锥桶 {len(ok_rows)}/{len(CONE_SLOTS)}"
            f" | **yaw=0 读法** 最差 coverage = {summary['yaw0_min_coverage']:.4f}"
            f" | **读回读法** 最差 coverage = {summary['read_min_coverage']:.4f}"
        )

    (out_dir / "report.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"[out] {out_dir / 'report.json'}")

    for c in cones:
        c.destroy()
    cams.close()
    ego.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

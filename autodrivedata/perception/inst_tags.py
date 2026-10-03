"""实例分割的**纯值核心**:实例 id 图的编解码 + 实例抽取 + 类归属(零 carla,零 torch)。

## 实例 GT 从哪来:一台相机同时给"有几个"和"各是什么"

`sensor.camera.instance_segmentation` 的原始 BGRA 里有两样东西:

| 通道 | 是什么 |
|---|---|
| `G + 256·B` | **实例 id**(口径由 `calib/probe_calib.decode_instance` 的差分实验裁决:4/4 命中且跨 actor 类型、跨 255 边界验过高字节) |
| `R` | **`CityObjectLabel` 语义类**,与 [`sem_tags.SEM_TAGS`](sem_tags.py) **同一张表** |

第二行意味着**实例相机把语义相机包了**:一台就够同时回答"这个像素属于哪个物体"和
"这个物体是什么类"。这正是 PQ(全景质量)要的两样东西 —— PQ 比 mask AP 多的那一项
"类报对了没",在这里是**白给的**。

⚠️ 但"R 通道 == 语义相机"这条**此前只在标定探针里被观察过**(数 ego 像素),**采集链上
没人验过**。所以本模块给判据留了 [`class_channel_matches_semantic`],**每次跑都验**
(不是开关)—— 它塌掉的样子是"PQ 的语义项恒为 0",而那会被读成"模型的类报得差"。

## 为什么落 uint16 PNG

`id = G + 256·B` 的上界恰是 **65535** ⇒ uint16 无损承载。落地形态是**已解码的 id**,
不是 BGRA 原图:判据直接读,不必再引一次解码口径(多一处口径就多一处会漂的地方)。

## 为什么只认一种 PNG mode

与 `sem_tags.decode_tag_png` 同一条纪律:`I;16` 之外的 mode **一律报错**。RGB / `L` /
调色板图读回来**仍然是个数组**,能一路算下去 —— 不拦就会得到一整套错的实例数
(而且是"看着正常"的那种错)。同理,id 超过 `MAX_INSTANCE_ID` 时**抛**而不是截断:
截断会把两个物体并成一个,表现为"数量少了一些",不表现为崩溃。
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from autodrivedata.perception.sem_tags import GT_CLASS_TAGS, GT_CLASSES, SEM_TAGS, TAG_NAMES

#: `id = G + 256·B` 的上界(B 最大 255)⇒ 恰好是 uint16 的上界。超了就是解码口径变了。
MAX_INSTANCE_ID = 65535

#: tag → 三个 GT 类,由 `GT_CLASS_TAGS` 反查(单一真值,不另抄一份)。
_TAG_TO_CLASS: dict[int, str] = {t: c for c, tags in GT_CLASS_TAGS.items() for t in tags}

#: **可数物体**(panoptic 口径里的 "things")。实例评测只对这些类有定义。
#:
#: ★ 这条是**实测**逼出来的:CARLA 的实例相机给**每一个关卡网格**都发 id ——
#: 2026-10-01 首采 20 帧,`Roads` 有 **113 个 id**、`Sidewalks`/`Poles`/`Buildings` 各有其数,
#: 而真正的 `Car` 只有 13 个。若不筛,`instances_from_maps` 会返回**几百个"实例"**,
#: 于是 PQ 的分母里全是路面 —— 而读数照样是个 0–1 的数,看不出来。
#:
#: 反过来,`drivable`(Roads)与 `lane`(RoadLines)是 panoptic 的 "stuff":它们是**面**,
#: 不是**可数的个体** —— 数"有几块路面"没有意义。所以实例评测只取 [`THING_TAGS`],
#: 而类级评测(`sem_eval`)照旧三类都算。**两套判据的分母不同是角色不同,不是不一致。**
THING_TAGS: frozenset[int] = GT_CLASS_TAGS["obstacle"]


@dataclass(frozen=True)
class Instance:
    """一个实例:它在 id 图里的编号、它落哪个 GT 类、以及它的布尔掩膜。

    `cls_tag` / `cls_name` 是**该实例像素的众数语义类**;`consistent` = 该实例的所有
    像素是否**只**落一个类。`consistent=False` 表示解码口径或渲染出了问题 ——
    判据遇到它应当**拒绝**这个样本,而不是拿着众数继续算(那会把"解码错了"变成
    "看着正常的一个类")。
    """

    instance_id: int
    cls_tag: int
    cls_name: str
    mask: np.ndarray
    consistent: bool

    @property
    def gt_class(self) -> str | None:
        """落在三个 GT 类中的哪一个;不属于任何一类(如地图自带道具)返回 None。"""
        return _TAG_TO_CLASS.get(self.cls_tag)

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def bbox(self) -> tuple[int, int, int, int]:
        """像素外接矩形 `(x1, y1, x2, y2)`(闭区间,像素索引口径)。"""
        ys, xs = np.where(self.mask)
        if len(xs) == 0:
            return (0, 0, -1, -1)
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def encode_instance_png(ids: np.ndarray) -> bytes:
    """实例 id 图 `(H, W)` 整型 → **16 位灰度** PNG 字节(无损)。

    ⚠️ 越界**抛异常**,不截断 —— 截断会把两个物体并成一个,症状是"实例数少了几个",
    而"少了几个"与"本来就没那么多"在报表上长得一样。
    """
    arr = np.asarray(ids)
    if arr.ndim != 2:
        raise ValueError(f"实例 id 图须为 (H,W),got {arr.shape}")
    if arr.min() < 0 or arr.max() > MAX_INSTANCE_ID:
        raise ValueError(f"实例 id 越界 [{arr.min()}, {arr.max()}],uint16 只到 {MAX_INSTANCE_ID}")
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint16)).save(buf, format="PNG")
    return buf.getvalue()


def decode_instance_png(data: bytes) -> np.ndarray:
    """PNG 字节 → 实例 id 图 `(H, W)` uint16。

    **只认 `mode="I;16"`** —— 见模块头注最后一段(读错 mode 不会报错,只会算出一套错的数)。
    """
    with Image.open(io.BytesIO(data)) as im:
        if im.mode != "I;16":
            raise ValueError(f"实例 id 图须为 16 位灰度(mode='I;16'),got {im.mode!r}")
        return np.array(im)


def instances_from_maps(ids: np.ndarray, cls_ch: np.ndarray, *, things_only: bool = True) -> list[Instance]:
    """`(实例 id 图, R 通道类图)` → 实例列表(按 id 分组,**不连通域拆分**)。

    为什么按 id 分组而不是按连通域:实例分割的定义就是"同一个 id = 同一个物体";
    被遮挡物切断的一块仍属于**同一个**实例,拆成两块会把实例数**虚高**。
    (需要"每个可见碎块"的场景是另一回事,那时用连通域,不在这里。)

    id 0 = 背景,跳过。

    `things_only=True`(默认)只留可数物体(见 `THING_TAGS`)—— 关卡网格也有 id,
    不筛就会把几百块路面算成实例。
    """
    if ids.shape != cls_ch.shape:
        raise ValueError(f"id 图与类图尺寸不符:{ids.shape} vs {cls_ch.shape}")
    out: list[Instance] = []
    for iid in np.unique(ids):
        iid = int(iid)
        if iid == 0:
            continue
        mask = ids == iid
        tags, counts = np.unique(cls_ch[mask], return_counts=True)
        top = int(tags[int(np.argmax(counts))])
        if things_only and top not in THING_TAGS:
            continue
        out.append(
            Instance(
                instance_id=iid,
                cls_tag=top,
                cls_name=TAG_NAMES.get(top, f"tag{top}"),
                mask=mask,
                consistent=bool(len(tags) == 1),
            )
        )
    return out


def class_channel_matches_semantic(cls_ch: np.ndarray, tag: np.ndarray) -> dict[str, float]:
    """★ **实测断言**:实例相机的 R 通道 == 语义相机的 tag 图,逐像素。

    这条成立时,一台实例相机就同时给实例与类(PQ 的语义项白给);不成立时**任何**
    "实例相机顺便当语义相机用"的写法都是错的,而它的症状是 **PQ 的语义项恒为 0** ——
    会被读成"模型的类报得差",不会读成"我们拿错了通道"。

    ⚠️ 两张图必须**同挂点同 fov**(`collect_surround --inst` 与 `--sem` 都是这样),
    否则比的是两个画幅,这条判据本身就没意义。

    返回 `{"match": 逐像素相等比例, "n": 像素数}`;`match < 1.0` 时调用方应当**拒绝**
    那个样本,而不是按比例打折。
    """
    if cls_ch.shape != tag.shape:
        raise ValueError(f"类图与 tag 图尺寸不符:{cls_ch.shape} vs {tag.shape}")
    return {"match": float((cls_ch == tag).mean()), "n": float(tag.size)}


def gt_class_names() -> tuple[str, ...]:
    """三个 GT 类(顺序即报数顺序)—— 转出去免得判据各处重抄。"""
    return GT_CLASSES


def is_scored_class(tag: int) -> bool:
    """该 tag 是否属于三个 GT 类之一(`sem_eval` 的口径)。

    `sem_tags.PROP_TAGS`(地图自带道具)与 `EXCLUDED_TAGS`(预测器产不出的类)
    都在此返回 False —— 否则两套判据会对同一张图给出互相矛盾的"该算多少"。
    """
    return tag in _TAG_TO_CLASS


def is_thing(tag: int) -> bool:
    """该 tag 是不是**可数物体**(实例评测的口径,见 `THING_TAGS`)。"""
    return tag in THING_TAGS


__all__ = [
    "MAX_INSTANCE_ID",
    "SEM_TAGS",
    "THING_TAGS",
    "Instance",
    "class_channel_matches_semantic",
    "decode_instance_png",
    "encode_instance_png",
    "gt_class_names",
    "instances_from_maps",
    "is_scored_class",
    "is_thing",
]

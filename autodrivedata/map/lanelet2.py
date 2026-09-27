"""Lanelet2(.osm XML)地图格式适配器 —— 导出与读回(纯 stdlib,不 import carla/torch)。

**角色**:与 [`opendrive.py`](opendrive.py) 同为「地图格式适配器」,只是方向不同 ——
`opendrive.py` 只读(CARLA `.xodr` → `OpenDriveMap`),本模块**双向**
(`MapVec` 元组 ⇄ Lanelet2 `.osm`)。产出方/消费方见 [`export_mapvec.py`](export_mapvec.py)。

**★ 本模块只到「标线级」**(用户 2026-09-27 裁决)。Lanelet2 的核心原语本是
**lanelet**(`relation type=lanelet` = 左边界 + 右边界),而我们六类要素的每一类
都**天然就是一条 `way`**(见下表)—— 故不做 lane 合成。**升级到真 lanelet 级
(选项二)需要什么**:

1. 把 [`mapvec.py`](mapvec.py) 内部的车道边界结构 `_Edge`(每条边界线 + 它两侧的
   lane 列表)提升为**公开模型**;
2. 按 section 为每个 driving lane 找它左右两侧的边界,分别引用第 1 步的 `way` id;
3. 输出 `relation type=lanelet` + `<member type="way" ref=… role="left"/"right"/>`。
   数据源已具备(源 xodr 里车道是完备的),缺的只是"把 lane↔边界 的关联带出来"。

**坐标**:Lanelet2 的 `.osm` 按标准存 **lat/lon**(由投影器从 UTM 反算)。我们的
`MapVec` 是 CARLA 世界**米**。本模块用**等距圆柱投影**(equirectangular)互转:

    lat = lat0 + y / R          lon = lon0 + x / (R·cos(lat0))

`origin = (lat0, lon0)` 默认取矢量**质心**;并且**写进输出文件的一行 XML 注释**里 ——
经纬度按 **12 位小数**落盘(纬度上 ≈ 1e-4 mm,远严于 `vec_to_dict` 的 mm 口径)——
外部工具会忽略注释,而我们的 `load_lanelet2` 读得回来,于是**往返不依赖文件外的隐式约定**
(优先级:显式参数 > 文件内注释 > 质心)。

⚠️ **投影是近似**(等距圆柱,非 UTM),在**单张 CARLA 图**的尺度上误差可忽略;
**跨图拼接、或与真实 GNSS 对齐时不适用** —— 那时应改用真正的 UTM 投影器。

**元素映射(标线级)**:

| `MapVec.cls` | lanelet2 元素 |
|---|---|
| `divider` | `<way>` + `type=line_thin` |
| `boundary` | `<way>` + `type=line_thick` |
| `centerline` | `<way>` + `type=line_thin` / `subtype=virtual` |
| `ped_crossing` | `<way>` + `type=crosswalk`(闭合) |
| `stop_line` | `<way>` + `type=stop_line` |
| `traffic_light` | `<node>` + `type=traffic_light` |

**保真手段**:每个元素都附自定义 tag `autodrivedata:cls` / `autodrivedata:src` /
`autodrivedata:id`,且 `attrs` 以 `attrs:<key>` 形式原样带上。**读取时优先用自定义 tag,
缺失才回退到上面的语义标签** —— 即既能无损耗地读回本仓的导出,也能**尽力**读第三方
Lanelet2 地图(只认得语义标签时类名可辨、attrs 为空)。

**结构与高程(2026-09-27 修)**:
- **标准 OSM 结构**:`<node id= lat= lon=/>` 元素 + `<way><nd ref=…/></way>`。
  早先版本把坐标**内联**在 `<nd lat= lon=/>` 上(非标准)⇒ 真实 lanelet2 读者只认 `ref`,
  会把每条 way 读成**空几何**(整图变空)。那是**破坏性**偏离,已改。
- **高程走 `<node>` 的 `ele` 属性**。OSM 标准里 `ele` 惯例是 tag 而非属性;写成属性是
  **无害扩展**(严格解析器忽略它,文件仍完全可读)。之所以非保不可:12 张能提的 CARLA 图里
  **10 张有非平凡高程**(Town07 9.4 m 到 **Town11 791 m**),桥/隧道/立体交叉全靠 z 区分 ——
  丢 z 会把不同层级叠成一条。`ele` 缺失时读回 0。
- **元素顺序靠 `autodrivedata:seq`**:文件里节点与 way 分块写,读取若按类型分组会**打乱顺序**;
  seq = 该元素在入参里的下标,读回按它排序。

**已知有损点**:
- `MapVec.attrs` 允许**重复键**(如 `validity` 多车道对),而 OSM tag 的 key 必须唯一 ⇒
  重复键写时加序号后缀(`attrs:validity:2`),读时**按后缀还原**;真冲突(第三方文件里
  本就有同 key 加后缀)会如实读成两条而非静默丢弃。
- OSM 的 `node` id 是整数,而 `MapVec.id` 是字符串 ⇒ 用 `autodrivedata:id` 原样带回。
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable

from .mapvec import MapVec

# 地球平均半径(米)。等距圆柱投影用;**不是** WGS84 椭球,见模块头注的适用边界。
EARTH_R = 6371000.0

# 自定义 tag 前缀:本模块的保真通道(读时优先,语义标签是回退)
_NS = "autodrivedata:"
CLS_TAG = _NS + "cls"
SRC_TAG = _NS + "src"
ID_TAG = _NS + "id"
SEQ_TAG = _NS + "seq"  # 元素顺序(节点与 way 分块写,读取要按它还原)
ATTR_PREFIX = "attrs:"
ORIGIN_PREFIX = "autodrivedata-origin"

# 类 → (type, subtype);subtype 为 None 表示不写
_CLS_TO_TAGS: dict[str, tuple[str, str | None]] = {
    "divider": ("line_thin", "solid"),
    "boundary": ("line_thick", None),
    "centerline": ("line_thin", "virtual"),
    "ped_crossing": ("crosswalk", None),
    "stop_line": ("stop_line", None),
}
# 语义标签 → 类(读第三方文件时的回退判据)
_TAGS_TO_CLS: dict[str, str] = {
    "line_thin": "divider",
    "line_thick": "boundary",
    "crosswalk": "ped_crossing",
    "pedestrian_marking": "ped_crossing",
    "stop_line": "stop_line",
    "traffic_light": "traffic_light",
}


# ---------- 投影 ----------


def to_latlon(x: float, y: float, origin: tuple[float, float]) -> tuple[float, float]:
    """CARLA 世界米 (x, y) → (lat, lon)。`origin` = 该局部系的 (lat0, lon0)。"""
    lat0, lon0 = origin
    lat = lat0 + math.degrees(y / EARTH_R)
    lon = lon0 + math.degrees(x / (EARTH_R * math.cos(math.radians(lat0))))
    return lat, lon


def from_latlon(lat: float, lon: float, origin: tuple[float, float]) -> tuple[float, float]:
    """`to_latlon` 的逆(同一 origin 下往返误差只在 float 精度)。"""
    lat0, lon0 = origin
    y = math.radians(lat - lat0) * EARTH_R
    x = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    return x, y


def _default_origin(vecs: tuple[MapVec, ...]) -> tuple[float, float]:
    """矢量质心处的经纬度;空集退化为 (0, 0)。"""
    pts = [p for v in vecs for p in v.points]
    if not pts:
        return 0.0, 0.0
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    return math.degrees(my / EARTH_R), math.degrees(mx / EARTH_R)


# ---------- 保真 tag 的编解码 ----------


def _encode_attrs(attrs: tuple[tuple[str, str], ...]) -> list[tuple[str, str]]:
    """attrs → 唯一 key 的 tag 列表:重复键加 `:2` `:3` 后缀(OSM tag key 必须唯一)。"""
    seen: dict[str, int] = {}
    out: list[tuple[str, str]] = []
    for k, v in attrs:
        n = seen.get(k, 0) + 1
        seen[k] = n
        suffix = "" if n == 1 else f":{n}"
        out.append((f"{ATTR_PREFIX}{k}{suffix}", v))
    return out


def _decode_attrs(tags: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """tag 列表 → attrs,按 `:N` 后缀还原重复键(保持原顺序)。"""
    out: list[tuple[str, str]] = []
    for k, v in tags:
        if not k.startswith(ATTR_PREFIX):
            continue
        key = k[len(ATTR_PREFIX) :]
        # 只剥**末尾**的 `:<数字>`,键名自身含冒号(如 `mark:type`)不受影响
        head, sep, tail = key.rpartition(":")
        if sep and tail.isdigit():
            key = head
        out.append((key, v))
    return tuple(out)


def _cls_from_tags(tags: dict[str, str]) -> str | None:
    """读回判据:优先自定义 tag,回退语义 `type`;都不认得返回 None(静默丢弃会被计数)。"""
    if (cls := tags.get(CLS_TAG)) is not None:
        return cls
    return _TAGS_TO_CLS.get(tags.get("type", ""))


# ---------- 导出 ----------


def dump_lanelet2(
    vecs: tuple[MapVec, ...], map_name: str = "", origin: tuple[float, float] | None = None
) -> str:
    """`MapVec` 元组 → Lanelet2 `.osm` 文本(**标准 OSM 结构**,见模块头注)。

    `origin` 不给则取矢量质心;写出的 origin 会**同时**记进文件注释(供 `load_lanelet2` 读回)。
    """
    org = origin if origin is not None else _default_origin(vecs)
    head = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- {ORIGIN_PREFIX} lat={org[0]:.12f} lon={org[1]:.12f} map={map_name!r} -->",
        '<osm version="0.6" generator="autodrivedata.lanelet2">',
    ]
    node_lines: list[str] = []  # 几何节点(无 tag)
    light_lines: list[str] = []  # 信号灯节点(带 tag)
    way_lines: list[str] = []
    node_id = 0
    way_id = 0
    n_dropped = 0
    for seq, v in enumerate(vecs):
        if v.cls == "traffic_light":
            if not v.points:
                n_dropped += 1
                continue
            x, y, z = v.points[0]
            lat, lon = to_latlon(x, y, org)
            node_id += 1
            light_lines.append(f'  <node id="{node_id}" lat="{lat:.12f}" lon="{lon:.12f}" ele="{z:.6f}">')
            _emit_tags(light_lines, v, extra=[("type", "traffic_light")], seq=seq)
            light_lines.append("  </node>")
            continue
        if len(v.points) < 2:
            # 单点折线在 OSM 里没有合法表达(way 至少两个 nd)—— **计数上报**,不静默
            n_dropped += 1
            continue
        refs: list[int] = []
        for x, y, z in v.points:
            lat, lon = to_latlon(x, y, org)
            node_id += 1
            refs.append(node_id)
            node_lines.append(f'  <node id="{node_id}" lat="{lat:.12f}" lon="{lon:.12f}" ele="{z:.6f}"/>')
        way_id += 1
        way_lines.append(f'  <way id="{way_id}">')
        way_lines.extend(f'    <nd ref="{r}"/>' for r in refs)
        _emit_tags(way_lines, v, extra=_cls_tags(v.cls), seq=seq)
        way_lines.append("  </way>")
    tail = ["</osm>"]
    if n_dropped:
        # 丢弃必须可见 —— "少导几条"是最难发现的一类错
        head.append(f"  <!-- autodrivedata-dropped: {n_dropped} (退化/单点要素,OSM way 无法表达) -->")
    return "\n".join([*head, *node_lines, *light_lines, *way_lines, *tail]) + "\n"


def _cls_tags(cls: str) -> list[tuple[str, str]]:
    t, st = _CLS_TO_TAGS.get(cls, ("line_thin", None))
    tags = [("type", t)]
    if st:
        tags.append(("subtype", st))
    return tags


def _emit_tags(lines: list[str], v: MapVec, extra: list[tuple[str, str]], seq: int) -> None:
    """写 tag。元信息为空时**不占位**(读回同样缺省 ⇒ 往返一致);**attrs 一律写**,
    因为空值 attrs 是用户数据,丢掉就是静默有损。`autodrivedata:seq` 用来还原元素顺序。"""
    meta = [(CLS_TAG, v.cls), (SEQ_TAG, str(seq))]
    meta += [(t, x) for t, x in ((ID_TAG, v.id), (SRC_TAG, v.src)) if x]
    for k, val in extra + meta + _encode_attrs(v.attrs):
        lines.append(f'    <tag k="{_esc(k)}" v="{_esc(val)}"/>')


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


# ---------- 读回 ----------


def _parse_origin_comment(text: str) -> tuple[float, float] | None:
    """从文件注释里取 origin(找不到返回 None)。

    注释由 `dump_lanelet2` 写出,形如
    `<!-- autodrivedata-origin lat=12.34 lon=56.78 map='Town' -->`。
    **外部工具会忽略注释**,故这不影响 `.osm` 的格式合法性。
    """
    m = re.search(rf"{ORIGIN_PREFIX}\s+lat=(\S+)\s+lon=(\S+)", text)
    if m is None:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None


def load_lanelet2(text: str, origin: tuple[float, float] | None = None) -> tuple[MapVec, ...]:
    """Lanelet2 `.osm` 文本 → `MapVec` 元组(**顺序与导出时一致**)。

    `origin` 优先级:**显式参数 > 文件内注释 > 报错**(无 origin 就无从还原米坐标,
    静默用 (0,0) 会把整张图挪到几内亚湾)。

    先建 `node id → (x, y, z)` 表再解析 way 的 `<nd ref>` —— 标准 OSM 的坐标只挂在
    `<node>` 上(`<nd>` 只有 `ref`)。`ele` 缺失 ⇒ z 取 0。
    """
    org = origin if origin is not None else _parse_origin_comment(text)
    if org is None:
        raise ValueError(
            "lanelet2 文本里没有 origin 注释(非本仓导出?),必须显式传 origin=(lat, lon)"
            " —— 否则无从把 lat/lon 还原成米坐标"
        )
    root = ET.fromstring(text)
    nodes: dict[str, tuple[float, float, float]] = {}
    for el in root.findall("node"):
        x, y = from_latlon(float(el.get("lat", 0.0)), float(el.get("lon", 0.0)), org)
        nodes[el.get("id", "")] = (x, y, float(el.get("ele", 0.0)))

    found: list[tuple[int, MapVec]] = []
    fallback = 0
    for el in root:
        if el.tag not in ("node", "way"):
            continue
        tags = {t.get("k", ""): t.get("v", "") for t in el.findall("tag")}
        cls = _cls_from_tags(tags)
        if cls is None:
            continue
        if el.tag == "node":
            p = nodes.get(el.get("id", ""))
            if p is None:
                continue
            pts = (p,)
        else:
            pts = tuple(_nd_point(nd, nodes, org) for nd in el.findall("nd"))
            pts = tuple(p for p in pts if p is not None)
            if len(pts) < 2:
                continue
        seq = int(tags[SEQ_TAG]) if tags.get(SEQ_TAG, "").isdigit() else None
        key = seq if seq is not None else 10_000 + fallback  # 无 seq(第三方)⇒ 按遇到次序排在后面
        fallback += 1
        found.append((key, _build(cls, pts, tags)))
    found.sort(key=lambda kv: kv[0])
    return tuple(v for _, v in found)


def _nd_point(
    nd: ET.Element, nodes: dict[str, tuple[float, float, float]], org: tuple[float, float]
) -> tuple[float, float, float] | None:
    """一个 `<nd>` → 米坐标。**两种形态都收**:

    - 标准 OSM:`<nd ref="…"/>`(坐标挂在 `<node>` 上)—— 本模块的导出形态;
    - **内联**:`<nd lat="…" lon="…"/>` —— 有些工具这么写(本模块 2026-09-27 之前也这么写)。
      只认标准形态的话,这类文件会**整图读成空**,静默丢数据。
    """
    if (ref := nd.get("ref")) is not None:
        return nodes.get(ref)
    if nd.get("lat") is not None and nd.get("lon") is not None:
        x, y = from_latlon(float(nd.get("lat", 0.0)), float(nd.get("lon", 0.0)), org)
        return (x, y, float(nd.get("ele", 0.0)))
    return None


def _build(cls: str, pts: tuple[tuple[float, float, float], ...], tags: dict[str, str]) -> MapVec:
    """tag 字典 → `MapVec`(attrs 走 `_decode_attrs`,重复键按 `:N` 后缀还原)。"""
    return MapVec(cls, pts, _decode_attrs(tags.items()), tags.get(ID_TAG, ""), tags.get(SRC_TAG, ""))

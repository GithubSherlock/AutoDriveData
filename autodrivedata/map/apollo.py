"""Apollo HD Map(text-format protobuf)适配器 —— 导出与读回(纯 stdlib,不 import carla/torch)。

**格式选择**:Apollo 官方的地图产物有两种形态 —— 编译后的二进制 `.pb`,以及
**text-format protobuf**(惯例文件名 `base_map.txt` / `sim_map.txt`,人可读)。本模块走后者:
本机**没有 Apollo 的 `.proto`**(实测),走 text-format 既不需要 `protoc`、
也不引入任何新依赖,同时保持"是 Apollo 的 schema"这一点。

**角色**:与 [`opendrive.py`](opendrive.py) 同为「地图格式适配器」。产出方/消费方见
[`export_mapvec.py`](export_mapvec.py)。

---

## ★ 唯一的语义降级(必须知道)

**Apollo 的 `Map` 没有「任意折线」要素**。它的要素是 `lane`(central_curve + 左右边界)、
`crosswalk`(polygon)、`stop_line`(segment)、`signal`、`junction`(polygon)、`road`……。
而我们的源数据(`MapVec`)是**标线中心的**六类折线。天然对应的只有三类:

| `MapVec.cls` | Apollo 要素 |
|---|---|
| `ped_crossing` | `crosswalk`(polygon)—— **天然对应** |
| `stop_line` | `stop_line`(segment)—— **天然对应** |
| `traffic_light` | `signal` —— **天然对应** |
| `divider` / `boundary` / `centerline` | **无落点** |

本模块对这三类做**降级表达**:借 `lane.central_curve` 承载(即"一条只有中心曲线的 lane"),
类名编进 `id.id` 前缀 —— 于是**保形且可往返**,但语义上确实是"把标线当车道"。
**这是本适配器唯一的有语义代价之处**,读本模块的人必须知道。

**升级到真 lane(选项二)需要什么**:

1. 给 [`mapvec.py`](mapvec.py) 的 `centerline` 实例的 `attrs` 补上 `road_id` / `lane_id` / `s`
   (`_extract_centerlines` 内部已经知道这些,只是没带出来);
2. 导出时用 [`opendrive.lane_boundary_t`](opendrive.py) 按 `(road, s, lane_id)` **重算左右边界**,
   写进 `lane.left_boundary.curve` / `lane.right_boundary.curve`;
3. 此时 `divider` / `boundary` 不再单独出 —— 它们已作为车道边界被表达,单独再出会**重复**
   (现在是靠 id 前缀区分;升级后要改成"只出 lane")。

---

## 保真约定(读回用)

Apollo 的 message 里**没有通用的 KV 槽**(`Id` 只有 `id` / `int_id`),而我们有
`attrs`(允许重复键)、`id`、`src` 三样要带。故本模块把它们编进 `id.id`:

    {cls}:{序号}[;@id={原 id}][;@src={src}][;{k}={v}]...

- `{序号}` **全局递增**,读回时按它**还原元素顺序** —— 否则"按要素种类分组"会把原始顺序打乱;
- `@id` / `@src` 是**保留键**(字面量含 `@`);
- 各值一律经 [`urllib.parse.quote(safe="")`](https://docs.python.org/3/library/urllib.parse.html)
  转义 ⇒ 用户 attr 的键名里的 `@` 会变成 `%40`,**不可能与保留键撞名**。

**这仍是合法的 Apollo `Id` 字符串**,不影响格式合法性;但它**是本适配器的约定,不是 Apollo 规范** ——
读第三方地图时前缀对不上,`load_apollo` 回退到按要素种类判类(见 `_KIND_TO_CLS`),
attrs 为空、id 取原串、顺序按遇到次序。

**已知有损点**:第三方 Apollo 地图的 `lane.left_boundary` / `right_boundary` / `junction` /
`overlap` / `road` 等**不读**(本模块只承诺读回本仓导出的子集,见 `load_apollo` docstring)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote, unquote

from .mapvec import MapVec

# id 串里的**保留键**:字面量含 `@`,而 attr 的 key 一律经 quote(safe="") ⇒
# `@` 会变成 `%40`,故保留键与用户 attr **不可能撞名**。
_RES_ID = "@id"
_RES_SRC = "@src"

# 类 → Apollo 要素种类;反向见 _KIND_TO_CLS
_CLS_TO_KIND: dict[str, str] = {
    "divider": "lane",
    "boundary": "lane",
    "centerline": "lane",
    "ped_crossing": "crosswalk",
    "stop_line": "stop_line",
    "traffic_light": "signal",
}
# 读第三方地图时的回退判据(序:先精确后宽松)
_KIND_TO_CLS: dict[str, str] = {
    "crosswalk": "ped_crossing",
    "stop_line": "stop_line",
    "signal": "traffic_light",
    "lane": "centerline",  # 真车道的 central_curve 就是中心线 —— 语义上最近的一类
}
# 每类要素「几何挂在哪个字段」—— **必须逐类指定**:第三方 lane 带 left/right_boundary,
# 若对整个 node 递归收点会把边界点一起吸进来(静默多出折线,极难发现)
_KIND_GEOM_KEY: dict[str, str] = {
    "lane": "central_curve",
    "crosswalk": "polygon",
    "stop_line": "segment",
    "signal": "position",
}


# ---------- id 的编解码(attrs/src/顺序 的保真通道) ----------


@dataclass(frozen=True)
class _IdParts:
    """`id.id` 里编着的全部信息(见模块头注的保真约定)。"""

    cls: str | None  # None ⇒ 前缀不是本适配器写的,调用方走种类回退
    seq: int | None  # 全局序号,用来还原元素顺序
    orig_id: str
    src: str
    attrs: tuple[tuple[str, str], ...]


def _encode_id(v: MapVec, seq: int) -> str:
    """`MapVec` → Apollo `id.id` 字符串(见模块头注的约定)。"""
    parts = [f"{v.cls}:{seq}"]
    if v.id:
        parts.append(f"{_RES_ID}={quote(v.id, safe='')}")
    if v.src:
        parts.append(f"{_RES_SRC}={quote(v.src, safe='')}")
    parts.extend(f"{quote(k, safe='')}={quote(val, safe='')}" for k, val in v.attrs)
    return ";".join(parts)


def _decode_id(raw: str) -> _IdParts:
    """Apollo `id.id` → `_IdParts`。前缀对不上(第三方文件)时 `cls` / `seq` 为 None。"""
    head, *rest = raw.split(";")
    name, sep, num = head.partition(":")
    cls = name if sep and name in _CLS_TO_KIND else None
    seq = int(num) if sep and num.isdigit() else None
    orig_id = src = ""
    attrs: list[tuple[str, str]] = []
    for item in rest:
        k, sep2, val = item.partition("=")
        if not sep2:
            continue
        if k == _RES_ID:
            orig_id = unquote(val)
        elif k == _RES_SRC:
            src = unquote(val)
        else:
            attrs.append((unquote(k), unquote(val)))
    return _IdParts(cls, seq, orig_id, src, tuple(attrs))


# ---------- text-format 的极小词法/语法(只覆盖我们导出的子集) ----------

_COMMENTLESS = re.compile(r'("(?:[^"\\]|\\.)*")|#[^\n]*')
_TOKEN = re.compile(
    r"""\s*(?:
      (?P<lbrace>\{)
    | (?P<rbrace>\})
    | (?P<colon>:)
    | (?P<string>"(?:[^"\\]|\\.)*")
    | (?P<number>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)
    | (?P<ident>[A-Za-z_][A-Za-z0-9_.-]*)
    )""",
    re.VERBOSE,
)


def _strip_comments(text: str) -> str:
    """去掉 `#` 注释(**引号内的 `#` 不算**;我们的导出里没有,但读别人的文件要稳)。"""
    return _COMMENTLESS.sub(lambda m: m.group(1) or "", text)


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        m = _TOKEN.match(text, pos)
        if m is None:
            raise ValueError(f"text-format 在偏移 {pos} 处无法解析: {text[pos : pos + 40]!r}")
        out.append(m.group(0).strip())
        pos = m.end()
    return out


def _scalar(tok: str) -> str:
    if tok.startswith('"') and tok.endswith('"'):
        return tok[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return tok


def _parse_message(tokens: list[str], i: int) -> tuple[dict[str, list], int]:
    """`{...}` 内的一段 → {字段: [值, …]}(重复字段成列表);值 = 标量 str 或嵌套 dict。"""
    msg: dict[str, list] = {}
    while i < len(tokens):
        if tokens[i] == "}":
            return msg, i + 1
        name = tokens[i]
        if i + 1 >= len(tokens):
            raise ValueError(f"字段 {name!r} 后没有值")
        nxt = tokens[i + 1]
        if nxt == "{":
            sub, i = _parse_message(tokens, i + 2)
            msg.setdefault(name, []).append(sub)
        elif nxt == ":":
            if i + 2 >= len(tokens):
                raise ValueError(f"字段 {name!r} 的冒号后缺值")
            msg.setdefault(name, []).append(_scalar(tokens[i + 2]))
            i += 3
        else:
            raise ValueError(f"字段 {name!r} 后既非 ':' 也非 '{{': {nxt!r}")
    raise ValueError("text-format 括号不闭合")


def parse_textproto(text: str) -> dict[str, list]:
    """text-format protobuf 文本 → `{顶层字段: [值, …]}`。

    顶层**没有外层花括号**(protobuf text-format 就是这样),故补一个哨兵 `}`
    让 `_parse_message` 的终止条件成立 —— 比给解析器加一个 `top` 开关更少分支。
    """
    tokens = _tokenize(_strip_comments(text))
    msg, end = _parse_message([*tokens, "}"], 0)
    if end != len(tokens) + 1:
        raise ValueError(f"text-format 顶层提前结束(解析到 {end}/{len(tokens) + 1})")
    return msg


# ---------- 导出 ----------


def dump_apollo(vecs: tuple[MapVec, ...], map_name: str = "") -> str:
    """`MapVec` 元组 → Apollo text-format protobuf 文本(`base_map.txt` 风格)。

    `map_name` 写进**注释**(`Map.header` 没有这个字段) —— 不往 schema 里塞私有字段,
    否则"这是合法 Apollo"这一条就不成立了。
    """
    out: list[str] = [
        "# Apollo HD Map (text-format protobuf) —— 由 autodrivedata.map.apollo 导出",
        f"# map_name: {map_name}",
        "# ★ divider/boundary/centerline 借 lane.central_curve 承载(见模块头注的降级说明)",
        "header {",
        '  version: "1.0"',
        '  projection { proj: "local" }',
        "}",
    ]
    n_dropped = 0
    for seq, v in enumerate(vecs):
        kind = _CLS_TO_KIND.get(v.cls)
        if kind is None or not v.points:  # 未知类 / 空几何:计数上报,不静默丢
            n_dropped += 1
            continue
        out.append(f"{kind} {{")
        out.append(f'  id {{ id: "{_encode_id(v, seq)}" }}')
        out.append(_geometry(v, kind))
        if kind == "lane":
            out.append(f"  length: {_polyline_len(v.points):.6f}")
        out.append("}")
    if n_dropped:
        out.insert(0, f"# autodrivedata-dropped: {n_dropped} (未知类/空几何)")
    return "\n".join(out) + "\n"


def _polyline_len(pts: tuple[tuple[float, float, float], ...]) -> float:
    return sum(sum((b[k] - a[k]) ** 2 for k in range(3)) ** 0.5 for a, b in zip(pts, pts[1:], strict=False))


def _point(p: tuple[float, float, float]) -> str:
    return f"point {{ x: {p[0]:.9f} y: {p[1]:.9f} z: {p[2]:.9f} }}"


def _geometry(v: MapVec, kind: str) -> str:
    """按要素种类写几何 —— 与 `_KIND_GEOM_KEY` 一一对应(读写两侧同一张表,不会漂)。"""
    if kind == "crosswalk":
        body = "\n".join(f"    {_point(p)}" for p in v.points)
        return f"  polygon {{\n{body}\n  }}"
    if kind == "signal":
        x, y, z = v.points[0]
        return f"  position {{ x: {x:.9f} y: {y:.9f} z: {z:.9f} }}"
    if kind == "stop_line":
        return f"  segment {{ line_segment {{ {' '.join(_point(p) for p in v.points)} }} }}"
    body = "\n".join(f"      {_point(p)}" for p in v.points)
    return f"  central_curve {{\n    segment {{\n      line_segment {{\n{body}\n      }}\n    }}\n  }}"


# ---------- 读回 ----------


def _points_of(node: dict) -> tuple[tuple[float, float, float], ...]:
    """递归收集嵌套里的 `point { x y z }`(顺序 = 出现顺序)。"""
    pts: list[tuple[float, float, float]] = []
    for key, vals in node.items():
        for val in vals:
            if not isinstance(val, dict):
                continue
            if key == "point":
                pts.append(
                    (
                        float(val.get("x", ["0"])[0]),
                        float(val.get("y", ["0"])[0]),
                        float(val.get("z", ["0"])[0]),
                    )
                )
            else:
                pts.extend(_points_of(val))
    return tuple(pts)


def _geom_points(node: dict, kind: str) -> tuple[tuple[float, float, float], ...]:
    """按**该类自己的几何字段**取点 —— 不整节点递归(否则第三方 lane 的左右边界会被吸进来)。"""
    key = _KIND_GEOM_KEY[kind]
    field = node.get(key)
    if not field or not isinstance(field[0], dict):
        return ()
    if kind == "signal":  # position 是单个 PointENU,不是 message 列表
        p = field[0]
        return ((float(p.get("x", ["0"])[0]), float(p.get("y", ["0"])[0]), float(p.get("z", ["0"])[0])),)
    return _points_of(field[0])


def _first_str(node: dict, key: str) -> str:
    vals = node.get(key)
    return vals[0] if vals and isinstance(vals[0], str) else ""


def load_apollo(text: str) -> tuple[MapVec, ...]:
    """Apollo text-format 文本 → `MapVec` 元组(**元素顺序与导出时一致**)。

    **只承诺读回本模块导出的子集**:`lane` / `crosswalk` / `stop_line` / `signal` 四种;
    `header` / `junction` / `overlap` / `road` 等**原样跳过**。读第三方地图时 id 前缀对不上,
    回退到按种类判类(`_KIND_TO_CLS`),attrs 为空。
    """
    root = parse_textproto(text)
    found: list[tuple[tuple[int, int], MapVec]] = []
    for order, (kind, cls_fallback) in enumerate(_KIND_TO_CLS.items()):
        for i, node in enumerate(root.get(kind, [])):
            if not isinstance(node, dict):
                continue
            id_field = node.get("id")
            raw_id = _first_str(id_field[0], "id") if id_field and isinstance(id_field[0], dict) else ""
            ids = _decode_id(raw_id) if raw_id else _IdParts(None, None, "", "", ())
            cls = ids.cls or cls_fallback
            pts = _geom_points(node, kind)
            if not pts:
                continue
            if cls == "traffic_light":
                pts = pts[:1]
            # 有序号 ⇒ 按序号还原全局顺序;无序号(第三方)⇒ 按种类再按遇到次序
            key = (0, ids.seq) if ids.seq is not None else (1, order * 10_000 + i)
            found.append((key, MapVec(cls, pts, ids.attrs, ids.orig_id or raw_id, ids.src)))
    found.sort(key=lambda kv: kv[0])
    return tuple(v for _, v in found)

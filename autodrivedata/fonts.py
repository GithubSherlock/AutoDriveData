"""覆盖层文本的**唯一字体落点**(纯值,不 import carla)。

## 为什么需要它(实测,不是推测)

`ImageDraw.text()` 遇到字体里没有的码位时,FreeType 画 `.notdef` 方框(豆腐块),
**不报错、不告警**;"中文全变方块"这个症状因此可以长期静默存在。本机踩到的是
**两个独立成因**,只修一个不够:

1. 本机字体只有 DejaVu / Quicksand / Ubuntu 三族,**一族的 CJK 字形都没有**
   (`fc-list` 查不到任何中文字体),而绘制代码直接 `ImageFont.truetype(DejaVuSans-Bold)`;
2. **PIL 没有字体回退链** —— `ImageDraw.text()` 只吃单个 `font` 对象(Pillow 12.3.0
   无 `font_chain`/`font_stack`)。**不传 `font=` 就用内置位图字体**,同样整行豆腐,
   而且只有 ~11 px。[autodrivedata/sim/live_common.py](../autodrivedata/sim/live_common.py) 的 HUD、
   [mapviz.py](mapviz.py) 的面板标题、[autodrivedata/calib/probe_calib.py](../autodrivedata/calib/probe_calib.py)
   的标签原本全属这一类。

## 判据(全部数值,不目检)

"这个字体能不能画中文"**不看文件名、不看文件在不在、也不看 `fc-list`**,而是**渲染探针**:

- `NOTDEF_PROBE` = `U+10FFFF`(noncharacter,任何字体都不该有它的字形)的像素签名,
  即该字体的 **`.notdef` 签名**;
- 某字符签名 == `.notdef` 签名 ⇒ 该字体**没有**这个字形 ⇒ 画出来是豆腐块;
- `has_cjk()` = 四个互不相同的常用汉字(``CJK_PROBE``)签名**互不相同** ——
  DejaVu 会把它们全画成同一个 `.notdef` 框 ⇒ 签名全同 ⇒ 判否。

实测:DejaVuSans-Bold 下 `文相机字` 四字签名**全等于** `U+10FFFF` 的签名;
CARLA 随包的 `DroidSansFallback.ttf` 四字签名互不相同、且都与 `.notdef` 不同。
判据不依赖 `fontTools`(非本项目依赖),只用 PIL。

## 字体从哪来

本机唯一可用的中文字体是 **CARLA 随包的 Slate 回退字体**
(`Engine/Content/Slate/Fonts/DroidSansFallback.ttf`,Apache-2.0)。这不是"顺手拿到外部资源":
CARLA 本来就是本项目的硬依赖(`tools/carla_server.sh`)。候选表仍按
「`AUTODRIVEDATA_FONT` 覆盖 → 系统 CJK → CARLA 随包 → DejaVu 兜底」排,换机器自动受益。

DroidSansFallback 仍缺 4 个码位(实测):`−` U+2212 / `∘` U+2218 / `⚠` U+26A0 /
`⁻` U+207B —— `sanitize()` 把它们换成等价 ASCII,**绝不把豆腐块留给用户**。
本轮排查里真正画进画面、又踩中这条的只有 `−`(见 [autodrivedata/calib/viz_calib_check.py](../autodrivedata/calib/viz_calib_check.py)
的 `yaw_carla = −az_nus` 与 `(w−1)/2`),其余三个只在注释/文档串里出现。

## 退化行为(诚实报,不假装)

找不到任何含中文字形的字体时(换机器、字体被删),`sanitize()` 把画不出的字符换成
`?` 而不是留豆腐块,并在首次解析时 `warnings.warn` 一次。判据见
[tests/test_fonts.py](../tests/test_fonts.py)。
"""

from __future__ import annotations

import glob
import hashlib
import os
import warnings
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# PIL 的 `fill` 口径:颜色元组 / 调色板索引 / 颜色名(与 `ImageDraw.text` 一致)
Ink = int | str | tuple[int, ...] | None

# noncharacter(Unicode 永久保留、不分配给任何字符)⇒ 任何字体都不该有它的字形
NOTDEF_PROBE = "\U0010ffff"
# 四个互不相同的常用汉字:能画中文的字体必须给出四个互不相同的签名
CJK_PROBE = "文相机字"
# 探针渲染字号:字形覆盖与字号无关,固定一个即可(且让缓存键稳定)
PROBE_SIZE = 24

FONT_ENV = "AUTODRIVEDATA_FONT"

# 系统 CJK 字体(本机**一个都没有**;列出来是为了换机器时自动受益)
SYSTEM_CJK_PATHS: tuple[str, ...] = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)
# CARLA 随包(硬依赖)的 Slate 回退字体:本机唯一真正能画中文的那一个
CARLA_FONT_GLOBS: tuple[str, ...] = (
    "/root/autodl-tmp/CARLA_*/Engine/Content/Slate/Fonts/DroidSansFallback.ttf",
    "/opt/CARLA_*/Engine/Content/Slate/Fonts/DroidSansFallback.ttf",
    "/usr/local/CARLA_*/Engine/Content/Slate/Fonts/DroidSansFallback.ttf",
)
DEJAVU_FALLBACK = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# 字体缺、但画面里真要用的码位 → 等价 ASCII。**只收实测缺的**(见模块 docstring)。
REPLACE: dict[str, str] = {
    "−": "-",  # MINUS SIGN —— 公式里的减号
    "∘": "o",  # RING OPERATOR —— 函数复合
    "⚠": "!",  # WARNING SIGN
    "⁻": "-",  # SUPERSCRIPT MINUS
}


# ---------------------------------------------------------------- 渲染探针


def _render_signature(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, ch: str) -> str:
    """单字渲染到**固定画布** → 像素哈希。

    固定画布(而不是比 `font.getmask` 的字节)是为了与 FreeType 的内部分配解耦:
    实测 `bytes(getmask(ch))` 会把 `U+E000` 与真正缺失的字符判成不同签名,而
    「渲染到同样大小的画布再比像素」两者一致 —— 后者才是"画出来长什么样"的口径。
    """
    n = PROBE_SIZE * 2
    img = Image.new("L", (n, n), 0)
    ImageDraw.Draw(img).text((PROBE_SIZE // 2, PROBE_SIZE // 4), ch, font=font, fill=255)
    return hashlib.sha256(img.tobytes()).hexdigest()


DEFAULT_KEY = ""  # 哨兵:没有字体可用 ⇒ PIL 内置位图字体(除 ASCII 外一律豆腐)


@lru_cache(maxsize=8)
def _font_at(path: str, size: int = PROBE_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)


@lru_cache(maxsize=8)
def _probe(path: str) -> tuple[str, tuple[str, ...]]:
    """(该字体的 `.notdef` 签名, `CJK_PROBE` 各字的签名)。按路径缓存 ⇒ 只开一次文件。"""
    font = _font_at(path)
    return _render_signature(font, NOTDEF_PROBE), tuple(_render_signature(font, c) for c in CJK_PROBE)


def candidate_paths() -> list[str]:
    """候选字体路径(按优先级;不一定存在)。诊断/测试用,正常路径请用 `font_path()`。"""
    return _candidates()


def notdef_signature(path: str) -> str:
    """该字体的 `.notdef` 签名(豆腐块的像素指纹)。"""
    return _probe(path)[0]


def has_cjk(path: str) -> bool:
    """该字体**实测**能画中文吗:四个互不相同的汉字必须给出四个互不相同的签名。"""
    if not Path(path).is_file():
        return False
    try:
        _, sigs = _probe(path)
    except OSError:  # 文件损坏 / 不是字体
        return False
    return len(set(sigs)) == len(CJK_PROBE)


@lru_cache(maxsize=8192)
def renders_in(path: str | None, ch: str) -> bool:
    """`ch` 在**指定字体**下会画出真字形吗(签名 ≠ 该字体的 `.notdef` 签名)。"""
    key = path or DEFAULT_KEY
    if not key:
        # PIL 内置位图字体:8 bit 位图,没有 CJK 一说
        return ch.isascii() and ch.isprintable()
    try:
        return _render_signature(_font_at(key), ch) != notdef_signature(key)
    except OSError:  # pragma: no cover - 字体在缓存后损坏
        return False


def renders(ch: str) -> bool:
    """`ch` 在**当前生效的字体**下会画出真字形吗。"""
    return renders_in(font_path(), ch)


def missing_in(text: str, path: str | None) -> str:
    """`text` 里**指定字体**画不出来的字符(去重、保序)。"""
    out: list[str] = []
    for ch in text:
        if ch not in out and not renders_in(path, ch):
            out.append(ch)
    return "".join(out)


def missing(text: str) -> str:
    """`text` 里**当前生效的字体**画不出来的字符(去重、保序)。"""
    return missing_in(text, font_path())


# ---------------------------------------------------------------- 字体解析


def _candidates() -> list[str]:
    return list(SYSTEM_CJK_PATHS) + sorted(p for g in CARLA_FONT_GLOBS for p in glob.glob(g))


@lru_cache(maxsize=1)
def font_path() -> str | None:
    """解析出**含中文字形**的字体路径;实在没有就回 DejaVu(并 warn 一次)。"""
    env = os.environ.get(FONT_ENV)
    if env:
        if not Path(env).is_file():
            raise FileNotFoundError(f"{FONT_ENV}={env} 不存在")
        if not has_cjk(env):
            warnings.warn(
                f"{FONT_ENV}={env} 实测不含中文字形 ⇒ 中文会退化成 '?'(见 {__name__})",
                stacklevel=2,
            )
        return env
    for p in _candidates():
        if has_cjk(p):
            return p
    if Path(DEJAVU_FALLBACK).is_file():
        warnings.warn(
            f"未找到含中文字形的字体 —— 中文会退化成 '?'(可用 {FONT_ENV}=<路径> 指定,或装 Noto Sans CJK)",
            stacklevel=2,
        )
        return DEJAVU_FALLBACK
    return None


@lru_cache(maxsize=128)
def get_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """按字号取字体(进程内缓存 —— HUD/拼图每帧都取,不能每次开文件)。"""
    path = font_path()
    if path is not None:
        try:
            return ImageFont.truetype(path, size)
        except OSError:  # pragma: no cover - 字体在解析后被删
            pass
    return ImageFont.load_default(size=size)


# ---------------------------------------------------------------- 绘制入口


def sanitize(text: str) -> str:
    """把当前字体画不出的字符换成等价 ASCII —— 兜底 `?`,**不留豆腐块**。"""
    if text.isascii():
        return text
    bad = [ch for ch in text if not renders(ch)]
    if not bad:
        return text
    return "".join(ch if renders(ch) else REPLACE.get(ch, "?") for ch in text)


def width(text: str, size: int = 20) -> float:
    """`text` 在当前字体下的**实际**像素宽(已 sanitize)——拼图/HUD 底条按它定宽。"""
    return float(get_font(size).getlength(sanitize(text)))


def wrap(text: str, size: int, max_width: float) -> list[str]:
    """按**实测像素宽**折行(已 sanitize),返回逐行文本。

    ★ 为什么不是按字符数折:本机字体下**比例拉丁(数字/字母窄)+ 全宽 CJK** 混排,
    一行里两类字符的宽度差 ~2×,"N 字一行"必然把长行推出画布 —— 实测页脚按字符数估的
    宽度溢出 1800 px 画布(字被**静默裁掉**,图上看着像"写完了")。
    判据只能是 `width()`,与底条定宽用的是同一把尺。

    断行偏好:优先在 ASCII 空格/中文标点处断,否则硬断(长英文单词/长数字串)。
    """
    if not text:
        return [""]
    lines: list[str] = []
    cur = ""
    for ch in text:
        if cur and width(cur + ch, size) > max_width:
            # 回退到最近的断点(仅在断点不太靠前时才回退,免得把行压得过短)
            cut = max(cur.rfind(" "), cur.rfind("、"), cur.rfind("。"), cur.rfind(","), cur.rfind(")"))
            if cut > len(cur) // 2:
                lines.append(cur[: cut + 1].rstrip())
                cur = cur[cut + 1 :].lstrip()
            else:
                lines.append(cur)
                cur = ""
            cur = cur.lstrip()  # 折行后不吃行首空格(band 宽度会白算一格)
        cur += ch
    if cur:
        lines.append(cur)
    return lines


def bbox(draw: object, text: str, size: int = 20) -> tuple[int, int, int, int]:
    """`text` 相对于绘制原点 `(0, 0)` 的包围盒(已 sanitize),与 `ImageDraw.textbbox` 同口径。"""
    d = draw
    return d.textbbox((0, 0), sanitize(text), font=get_font(size))  # type: ignore[attr-defined]


def draw_text(
    draw: object,
    xy: tuple[float, float],
    text: str,
    size: int = 20,
    fill: Ink = (255, 255, 255),
    anchor: str | None = None,
) -> None:
    """**所有** `ImageDraw.text()` 都应走这里 —— 唯一的字体 + sanitize 落点。

    直接用 `d.text(...)`(尤其是不传 `font=`)是本模块要根除的写法:前者 tofu 静默、
    后者连字号都不是自己定的。
    """
    d = draw
    d.text(xy, sanitize(text), font=get_font(size), fill=fill, anchor=anchor)  # type: ignore[attr-defined]


def diagnostics() -> dict[str, object]:
    """给探针/测试用:当前解析结果与它画不出的字符(中文退化时一眼看得见)。"""
    path = font_path()
    return {
        "font_path": path,
        "env_override": os.environ.get(FONT_ENV),
        "has_cjk": bool(path) and has_cjk(path),
        "probe_ok": bool(path) and has_cjk(path),
        "missing_in_use": missing(CJK_PROBE + "ABC-123"),
    }

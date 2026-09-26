"""字体落点的回归钉:**中文字形不许静默变成豆腐块**。

症状是 `outputs/calib_check/check_geometry.png` 里中文字符位全是方框。根因有两条,
只修一条不够(见 [autodrivedata/fonts.py](../autodrivedata/fonts.py) 的模块 docstring):
① 本机字体族**一个 CJK 字形都没有**,绘制代码却硬写 `DejaVuSans-Bold`;
② **PIL 没有字体回退链**,不传 `font=` 就用内置位图字体(同样没有 CJK)。

判据全部数值化、不目检、不依赖 `fontTools`:
`U+10FFFF`(noncharacter,任何字体都不该有字形)的像素签名 = 该字体的 `.notdef` 签名;
某字符签名与它相同 ⇒ 该字体没有这个字形 ⇒ 画出来就是豆腐块。
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from autodrivedata import fonts

ROOT = Path(__file__).resolve().parents[1]

# 会把文字画到画面上的模块(新增绘制脚本请加进来)
DRAWING_MODULES: tuple[str, ...] = (
    "autodrivedata/sim/live_common.py",
    "autodrivedata/sim/live_studio.py",
    "autodrivedata/calib/viz_calib_check.py",
    "autodrivedata/calib/probe_calib.py",
    "autodrivedata/sim/carla_common.py",
    "autodrivedata/sim/collect_static_gt.py",
    "autodrivedata/map/mapviz.py",
    "autodrivedata/calib/calib_live.py",
)
# 画文字的函数名(`<obj>.text(...)` / `stamp(...)` / `fonts.draw_text(...)` / HUD 构造器)
DRAW_CALLS: frozenset[str] = frozenset({"text", "stamp", "draw_text", "draw_hud", "hud_line", "hud"})

DEJAVU_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _callee_name(node: ast.Call) -> str:
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")


def _drawn_literals(rel: str) -> list[tuple[int, str]]:
    """AST 抽出该文件里**会画上屏**的字符串常量(含 f-string 的静态片段)。

    两个来源,缺一个都会漏:
    ① 绘制调用的实参(`d.text(...)` / `stamp(...)` / `draw_hud(...)`);
    ② HUD 构造器**函数体**(`hud_line` 之类是 `return` 一个长 f-string,不是调用点)——
       漏了它,`calib_live` 那一整行中文 HUD 就没进检查。

    只扫这两处,注释与文档串不算 —— 免得测试对无关字符报假失败。
    """
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    roots: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _callee_name(node) in DRAW_CALLS:
            roots.extend(list(node.args) + [k.value for k in node.keywords])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in DRAW_CALLS:
            roots.append(node)
    out: list[tuple[int, str]] = []
    for root in roots:
        for sub in ast.walk(root):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                out.append((getattr(sub, "lineno", 0), sub.value))
    return out


DRAWN: list[tuple[str, int, str]] = [
    (rel, ln, s) for rel in DRAWING_MODULES for ln, s in _drawn_literals(rel)
]


def _canvas(ch: str, size: int = 24) -> Image.Image:
    """单字渲染到固定画布 —— 与 `fonts._render_signature` 同口径,用来比"画出来一样吗"。"""
    n = size * 2
    img = Image.new("L", (n, n), 0)
    ImageDraw.Draw(img).text((size // 2, size // 4), ch, font=fonts.get_font(size), fill=255)
    return img


# ---------------------------------------------------------------- 探针本身可信吗


class TestProbe:
    def test_noncharacter_renders_as_notdef(self):
        """`U+10FFFF` 在**任何**字体下都该落到 `.notdef`(探针的立论基础)。"""
        assert not fonts.renders("\U0010ffff")

    def test_dejavu_is_rejected_as_cjk_font(self):
        """DejaVu 有 `−`/`°`/`★`,却是**豆腐块**画中文 —— 这正是原缺陷的成因。"""
        assert Path(DEJAVU_BOLD).is_file()
        assert fonts.has_cjk(DEJAVU_BOLD) is False
        assert fonts.missing_in("文相机字", DEJAVU_BOLD) == "文相机字"
        assert fonts.missing_in("−°★", DEJAVU_BOLD) == ""  # 符号它反而有 ⇒ 判据不是"文件在不在"

    def test_resolved_font_really_draws_cjk(self):
        """实际生效的字体必须**实测**能画中文:四个互不相同的汉字给出四个签名。"""
        path = fonts.font_path()
        assert path, "没有解析到任何字体 —— 中文会退化成 '?'(见 fonts.font_path 的 warn)"
        assert fonts.has_cjk(path), f"{path} 不含中文字形"
        assert fonts.missing(CJK := "文相机字中文测试") == "", f"缺字:{fonts.missing(CJK)}"


# ---------------------------------------------------------------- 端到端:画出来真的不一样


class TestRendering:
    def test_distinct_cjk_chars_paint_distinct_pixels(self):
        """最强的钉子:两个不同的汉字在**画布上**必须像素不同。

        原缺陷下它们会逐像素相同(同一个 `.notdef` 方框)—— 这正是"豆腐块"的定义。
        """
        a, b = _canvas("相"), _canvas("机")
        assert np.asarray(a).any(), "什么都没画出来"
        assert not np.array_equal(np.asarray(a), np.asarray(b))

    def test_sanitized_text_leaves_no_glyph_missing(self):
        probe = "中文测试 −∘⚠⁻ –—°±×✓✗←→★ 0-9 Aa ./,()"
        assert fonts.missing(fonts.sanitize(probe)) == ""

    def test_replace_targets_are_themselves_renderable(self):
        """替换目标必须自己画得出来,否则只是把豆腐块换成另一个豆腐块。"""
        for src, dst in fonts.REPLACE.items():
            assert fonts.renders(dst), f"{src!r} → {dst!r}:目标字形也不存在"

    def test_ascii_and_known_cjk_pass_through_untouched(self):
        for s in ("actors=12 | 1.4fps", "文相机字"):
            assert fonts.sanitize(s) == s

    def test_sanitize_never_lengthens_the_run(self):
        """替换是逐字符的 ⇒ 长度不变(排版不会因此错位)。"""
        s = "yaw_carla = −az_nus (w−1)/2"
        assert len(fonts.sanitize(s)) == len(s)

    def test_width_measures_cjk_as_double(self):
        """CJK 字宽 ≈ 2× ASCII —— HUD 底条按实测宽度定,不能再写死 `7 * len(text)`。"""
        assert fonts.width("中", 20) > 1.8 * fonts.width("a", 20)

    def test_bbox_matches_what_is_drawn(self):
        d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        assert fonts.bbox(d, "中文", 20) == d.textbbox(
            (0, 0), fonts.sanitize("中文"), font=fonts.get_font(20)
        )


class TestWrap:
    """`fonts.wrap`:按**实测像素宽**折行。

    踩过的坑(2026-09-23,§P-M.9):配置图页脚是单行 `draw_text`,超出画布被 PIL **静默裁掉**
    —— 不报错、不告警,图上看着像"这行写完了"。根因是不能按字符数估宽:同一行里比例拉丁
    (数字/字母窄)与全宽 CJK 的宽度差 ~2×。
    """

    MIXED = "方位角 az_nus:0° = 车头,方向与范围 = az ± fov/2。(与 nuScenes 官网标定表同口径)"

    def test_every_line_fits_the_budget(self):
        """硬判据:折出来的**每一行**实测宽 ≤ 预算(超了就等于又被裁一次)。"""
        for max_w in (600, 900, 1400, 1720):
            for line in fonts.wrap(self.MIXED, 18, max_w):
                assert fonts.width(line, 18) <= max_w, f"{max_w}: {line!r} 超宽"

    def test_content_is_preserved_except_whitespace(self):
        """折行不丢字(只许吃掉断点处的空白)—— 丢字比溢出更难发现。"""
        got = "".join(fonts.wrap(self.MIXED, 18, 600)).replace(" ", "")
        assert got == self.MIXED.replace(" ", "")

    def test_breaks_at_a_space_when_one_is_available(self):
        """有空格断点就不硬断词:整串 196 px 放不进 160 px,但"到 gamma 为止"放得进。

        断在 `gamma` 之后(而不是把 `gamma` 砍成 `gamm`)才是对的 —— 硬断只该是**没有**
        断点时的兜底。
        """
        assert fonts.wrap("alpha beta gamma delta", 18, 160) == ["alpha beta gamma", "delta"]

    def test_single_long_token_is_hard_split(self):
        """没有断点的超长串必须硬断,不能整串吐出来。"""
        lines = fonts.wrap("x" * 200, 18, 200)
        assert len(lines) > 1
        assert all(fonts.width(line, 18) <= 200 for line in lines)

    def test_empty_and_unbreakable_inputs(self):
        assert fonts.wrap("", 18, 100) == [""]
        assert fonts.wrap("短", 18, 100) == ["短"]


# ---------------------------------------------------------------- 全仓:画上屏的字符串都不许有豆腐块


class TestDrawnStringsAreRenderable:
    def test_scan_actually_found_something(self):
        """扫描器本身要自证有效 —— 否则下面的用例会空过(假绿)。

        **不是**每个绘制模块都有字面量:`mapviz` 的面板标题、`collect_static_gt` 的
        地标类别都是调用方传进来的变量(无字面量可抽),故只要求总量与覆盖广度。
        """
        assert len(DRAWN) >= 30, f"只抽到 {len(DRAWN)} 条绘制字符串,扫描器可能坏了"
        assert len({rel for rel, _, _ in DRAWN}) >= 4, "抽到的模块太少,扫描器可能坏了"
        assert any(not s.isascii() for _, _, s in DRAWN), "没抽到任何非 ASCII 字符串?"
        assert any(s == "采样" or "标定" in s for _, _, s in DRAWN), "没抽到 HUD/标签里的中文?"

    def test_no_drawn_string_needs_dropping_to_nothing(self):
        """每个会画上屏的字符串,sanitize 之后都必须**零缺字**。"""
        bad = [
            (rel, ln, s, fonts.missing(fonts.sanitize(s)))
            for rel, ln, s in DRAWN
            if fonts.missing(fonts.sanitize(s))
        ]
        assert not bad, "这些绘制字符串仍有画不出的字符:" + repr(bad[:5])

    def test_candidate_fonts_cover_every_drawn_character(self):
        """换任何候选字体(系统 CJK / CARLA 随包),画上屏的字符都得**有解**。

        有解 = 字体自带,或在 `REPLACE` 表里 —— 两者都没有才是缺陷。
        """
        chars = "".join(s for _, _, s in DRAWN)
        unresolved: dict[str, set[str]] = {}
        for path in fonts.candidate_paths():
            if not Path(path).is_file():
                continue
            gaps = set(fonts.missing_in(chars, path)) - set(fonts.REPLACE)
            if gaps:
                unresolved[path] = gaps
        assert not unresolved, f"候选字体画不出、REPLACE 也没兜住:{unresolved}"

    def test_no_module_draws_with_a_bare_text_call(self):
        """**根因回归钉**:绘制脚本不许再出现不带 `font=` 的 `d.text(...)`。

        `ImageDraw.text()` 不传 `font=` 会用 PIL 内置位图字体(无 CJK、且只有 ~11 px),
        这是原缺陷的第二条成因。所有绘制都必须经 `autodrivedata.fonts`。
        """
        offenders: list[str] = []
        for rel in DRAWING_MODULES:
            src = (ROOT / rel).read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(src)):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr != "text":
                    continue
                # `autodrivedata/fonts.py` 自己就是那个落点,不在此列(它不在 DRAWING_MODULES 里)
                if not any(k.arg == "font" for k in node.keywords):
                    offenders.append(f"{rel}:{node.lineno}")
        assert not offenders, "这些位置没传 font=(会用内置位图字体,中文变豆腐块):" + repr(offenders)


@pytest.mark.parametrize("rel", DRAWING_MODULES)
def test_drawing_module_imports_the_font_module(rel: str) -> None:
    """**碰文字渲染的模块**必须经 `autodrivedata.fonts` 落点 —— 否则会悄悄退回豆腐块。

    判据 = 「源码里出现 `fonts`」⇒ 必须真的 `from autodrivedata.fonts import ...`。
    只**委托**绘制、自己不碰文字的模块跳过(`live_studio.py` 只调 `draw_hud`,
    `calib_live.py` 只构造 `hud_line` 字符串)—— 它们进 `DRAWING_MODULES` 是为了
    **字面量扫描**,不是为了这个判据。

    ⚠️ **旧判据长期"因错误的原因"全绿,值得记下**:它是
    `n.module in ("autodrivedata", "autodrivedata.fonts")` ——
    **任何** `from autodrivedata import X` 都算通过。`live_studio.py` 里那句
    `from autodrivedata import calib_live as cl` 恰好把它喂饱了。
    阶段 3 把该行改精确成 `from autodrivedata.calib import ...` 后,假通过才暴露。
    教训:**判据里的"或"每放宽一档,都要问它会不会被无关代码满足**。
    """
    src = (ROOT / rel).read_text(encoding="utf-8")
    if "fonts" not in src:
        pytest.skip(f"{rel} 不碰文字渲染(仅委托绘制),无需 import 字体落点")
    # 两种惯用形式都得认:`from autodrivedata.fonts import ...` 与 `from autodrivedata import fonts`。
    # 关键是**后者只认别名恰为 `fonts` 的那一种** —— 旧判据在这里失守:
    # 它接受 `from autodrivedata import <任何东西>`,于是被无关的 import 喂饱。
    imported = any(
        isinstance(n, ast.ImportFrom)
        and (
            n.module == "autodrivedata.fonts"
            or (n.module == "autodrivedata" and any(a.name == "fonts" for a in n.names))
        )
        for n in ast.walk(ast.parse(src))
    )
    assert imported, f"{rel} 用了 fonts 却没有 import autodrivedata.fonts"

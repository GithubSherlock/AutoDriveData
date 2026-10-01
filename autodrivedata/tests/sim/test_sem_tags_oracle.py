"""用 **`carla` 当 oracle** 钉死语义 tag 表 —— 所以本文件住在 `tests/sim/`(那层才许 import carla)。

## 为什么非要有这一条

`perception/sem_tags.py` 里那张 `SEM_TAGS` 是**手抄**的。抄错的代价不是崩溃,是**每一个
IoU 都算在错的类上**,而且它仍然是个 0–1 的数,看不出来。

2026-10-01 真差一点抄错:凭记忆写的是旧版 CARLA 的 CityScapes 顺序("6=RoadLine / 7=Road"),
而 0.9.16 的 `CityObjectLabel` 是 **`Roads=1 / RoadLines=24 / Car=14`**。当时的实测反查是
"把帧里每个 tag 的调色板色与源码 29 色表逐个对" —— 那条能证伪,但**跑一次就没了**。
这条把它固化成每次 `pytest` 都跑一遍的判据。

`carla.CityObjectLabel` 是**静态枚举**,`import carla` **不需要服务器**(pycarla 是客户端库)。
"""

from __future__ import annotations

import carla
import pytest

from autodrivedata.perception.sem_tags import PALETTE, SEM_TAGS


def _carla_labels() -> dict[str, int]:
    """`carla.CityObjectLabel` 的全部成员(含 `NONE`/`Any`)。"""
    out = {}
    for name in dir(carla.CityObjectLabel):
        if name.startswith("_"):
            continue
        v = getattr(carla.CityObjectLabel, name)
        if isinstance(v, int):
            out[name] = int(v)
    return out


class TestSemTagsMatchCarla:
    def test_every_carla_label_is_present_with_the_same_value(self):
        """★ 主判据:我们抄的表必须**逐项**等于 CARLA 的枚举。"""
        ours, theirs = SEM_TAGS, _carla_labels()
        missing = sorted(set(theirs) - set(ours))
        assert not missing, f"CARLA 有而我们没抄:{missing}"
        bad = {k: (ours[k], theirs[k]) for k in theirs if ours[k] != theirs[k]}
        assert not bad, f"编号抄错(我们, CARLA):{bad}"

    def test_we_add_nothing_carla_does_not_have(self):
        """反向对照:我们的表里不许有 CARLA 没有的名字 —— 那多半是抄串了行。"""
        extra = sorted(set(SEM_TAGS) - set(_carla_labels()))
        assert not extra, f"我们多出来的标签:{extra}"

    def test_the_table_is_not_accidentally_the_old_numbering(self):
        """★ 反向对照(防"抄成旧版"复发):旧版 CARLA 是 `6=RoadLine / 7=Road`。

        真抄成那一版,上面两条**照样能过**(旧版自洽),只有这条会红。
        """
        assert SEM_TAGS["Roads"] == 1 and SEM_TAGS["RoadLines"] == 24
        assert SEM_TAGS["Roads"] != 7, "抄成旧版编号了(旧版 Roads=7)"

    @pytest.mark.parametrize("name", ["Roads", "RoadLines", "Car", "Pedestrians", "Bus", "Truck"])
    def test_the_six_tags_the_judge_depends_on(self, name: str):
        """判据只用到这几个 —— 单独点名,红了能一眼看出是哪个类塌了。"""
        assert SEM_TAGS[name] == int(getattr(carla.CityObjectLabel, name))


class TestPaletteShape:
    def test_palette_has_one_colour_per_tag(self):
        """调色板是 0–28 全覆盖的:漏一个,那类的像素在图里就是纯黑,与 `NONE` 混同。"""
        assert sorted(PALETTE) == list(range(29))

    def test_palette_colours_are_distinct(self):
        """两色相同 ⇒ "用调色板色反查 tag"那套自证法失效(2026-10-01 就是靠它验的编号)。"""
        assert len(set(PALETTE.values())) == 29

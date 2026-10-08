"""`edit/downstream_eval` 的判据回归钉(**不出模型、不跑检测器**)。

两个必须钉住的东西:
1. **四臂的组法** —— `D` 臂靠"图平移半程"造对照。**两版都错过**(先旋转 GT ⇒ GT 条数变了;
   再旋转图 ⇒ 与 C 臂逐位相同,因为 `eval_2d_ab` 把框**池化**、配对顺序不起作用) ——
   现在的版本是"取后半程的图配前半程的 GT",**图集合真的不同**。
2. **裁决的分支** —— 三档解读要能分辨"过毒"与"假退化"**两头**的错,
   而这两头的读数都不是"AP 低",是**相对地板的**位置。
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from autodrivedata.edit import downstream_eval as D
from autodrivedata.perception import eval_2d_ab as AB


class TestBuildArm:
    def _src(self, tmp_path, n=3):
        from PIL import Image

        imgs, labs = tmp_path / "img", tmp_path / "lab"
        imgs.mkdir()
        labs.mkdir()
        for i in range(n):
            Image.fromarray(np.full((4, 4, 3), i, np.uint8)).save(imgs / f"{i:06d}.png")
            (labs / f"{i:06d}.txt").write_text(f"Car 0 0 0 0 0 1 1 1 1 1 0 0 0 0 {i}\n", encoding="utf-8")
        return imgs, labs

    def test_copies_aligned_by_default(self, tmp_path):
        imgs, labs = self._src(tmp_path)
        dst = tmp_path / "arm"
        assert D.build_arm(dst, image_src=imgs, label_src=labs) == 3
        assert sorted(p.name for p in (dst / "training/image_2").glob("*.png")) == [
            "000000.png",
            "000001.png",
            "000002.png",
        ]
        # 对齐时第 i 帧配第 i 条
        assert (dst / "training/label_2/000001.txt").read_text().split()[-1] == "1"

    def test_shift_takes_later_images_but_same_gt(self, tmp_path):
        """★★ **D 臂的组法**:GT **原样**,图从第 `K` 张起取(所以**图集合真的不同**)。

        ⚠️ 两版都错过,都留在这里当反例:
        ① 旋转 GT ⇒ D 的 GT **条数变了**(实测 20 vs 40,`--limit N` 只读前 N 个 label);
        ② 旋转图 `(i+K) mod N` ⇒ **D 与 C 逐位相同** —— 因为 `eval_2d_ab` 把框**池化**,
           配对顺序对统计毫无影响,**"重排"就是恒等**。
        """
        imgs, labs = self._src(tmp_path, n=4)
        dst = tmp_path / "arm"
        keep = ["000000", "000001", "000002"]  # 与真实用法同形:`frames` 只筛 GT,图靠 shift 取
        files = sorted(imgs.glob("*.png"))  # 显式列表(与"生成臂"同形)
        assert D.build_arm(dst, image_files=files, label_src=labs, image_shift=1, frames=keep) == 3
        # 只取 3 张(3 条 GT)
        assert len(list((dst / "training/image_2").glob("*.png"))) == 3
        # GT 逐条不变
        for i in range(3):
            assert (dst / f"training/label_2/{i:06d}.txt").read_text().split()[-1] == str(i)
        # 图从第 1 张起:第 0 张 GT 配的是原第 1 张图(整幅值 = 1)
        from PIL import Image

        assert int(np.array(Image.open(dst / "training/image_2/000000.png"))[0, 0, 0]) == 1

    def test_shift_zero_is_a_noop(self, tmp_path):
        """反向对照:`K=0` 时与 `C` 臂**逐字节相同** —— 那就不是对照臂。"""
        imgs, labs = self._src(tmp_path)
        a, b = tmp_path / "a", tmp_path / "b"
        D.build_arm(a, image_src=imgs, label_src=labs)
        D.build_arm(b, image_src=imgs, label_src=labs, image_shift=0)
        for i in range(3):
            assert (a / f"training/label_2/{i:06d}.txt").read_text() == (
                b / f"training/label_2/{i:06d}.txt"
            ).read_text()
            assert (a / f"training/image_2/{i:06d}.png").read_bytes() == (
                b / f"training/image_2/{i:06d}.png"
            ).read_bytes()

    def test_not_enough_images_raises(self, tmp_path):
        """图不够平移 ⇒ 抛,不静默少取。"""
        imgs, labs = self._src(tmp_path)
        with pytest.raises(SystemExit, match="图不够"):
            D.build_arm(tmp_path / "a", image_src=imgs, label_src=labs, image_shift=2)

    def test_count_mismatch_raises(self, tmp_path):
        """图与 GT 条数不等 ⇒ A/B 配对的前提没了,必须抛(不许截断到较短的那个)。"""
        imgs, labs = self._src(tmp_path, n=3)
        (labs / "000003.txt").write_text("Car 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="图与 GT 条数不等"):
            D.build_arm(tmp_path / "arm", image_src=imgs, label_src=labs)

    def test_frame_filter(self, tmp_path):
        imgs, labs = self._src(tmp_path)
        dst = tmp_path / "arm"
        assert D.build_arm(dst, image_src=imgs, label_src=labs, frames=["000001"]) == 1

    def test_empty_source_raises(self, tmp_path):
        (tmp_path / "e").mkdir()
        with pytest.raises(SystemExit, match="为空"):
            D.build_arm(tmp_path / "arm", image_src=tmp_path / "e", label_src=tmp_path / "e")

    def test_exactly_one_image_source(self, tmp_path):
        """`image_src` 与 `image_files` **恰好给一个** —— 都不给会静默组出空臂。"""
        imgs, labs = self._src(tmp_path)
        with pytest.raises(SystemExit, match="恰好给一个"):
            D.build_arm(tmp_path / "a", label_src=labs)
        with pytest.raises(SystemExit, match="恰好给一个"):
            D.build_arm(tmp_path / "a", image_src=imgs, image_files=[], label_src=labs)

    def test_explicit_file_list_works(self, tmp_path):
        imgs, labs = self._src(tmp_path)
        dst = tmp_path / "arm"
        files = sorted(imgs.glob("*.png"))[::-1]  # 显式列表(顺序即配对顺序)
        assert D.build_arm(dst, image_files=files, label_src=labs) == 3


class TestRunArmsKeys:
    def test_returned_keys_are_exactly_the_arm_names(self, tmp_path):
        """★★ **`run_arms` 的键必须恰好是臂名**。

        2026-10-06 实测:第一版在这里塞了 `_n_keep` / `_shift` 两个元数据键,
        而 `main` 是 `for name in arms:` 逐臂评测的 ⇒ 它去评了一个叫 `_n_keep` 的目录,
        拿到 `mAP=nan` **并继续往下跑完**(不报错、退出码 0)。元数据改为挂在每臂里。
        """
        from PIL import Image

        def mk(root, n=4):
            for sub in ("image_2", "label_2"):
                (root / "training" / sub).mkdir(parents=True, exist_ok=True)
            for i in range(n):
                Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(root / f"training/image_2/{i:06d}.png")
                (root / f"training/label_2/{i:06d}.txt").write_text(
                    "Car 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0\n", encoding="utf-8"
                )

        clear, fog, gen = tmp_path / "c", tmp_path / "f", tmp_path / "g"
        mk(clear)
        mk(fog)
        gen.mkdir()
        for i in range(4):
            Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(gen / f"gen_gt-depth_{i:05d}_0.png")
        keys = set(D.run_arms(clear, fog, gen, tmp_path / "w").keys())
        assert keys == {"A_clear", "B_truth_fog", "C_gen_fog", "D_gen_shift"}, f"键不对:{keys}"
        assert not any(k.startswith("_") for k in keys), "元数据键不许混进臂字典"


class TestSharedGt:
    """★★ **四臂必须共用一套 GT** —— 2026-10-06 实测踩到。

    第一版让 B 臂带自己的 `label_2`,理由是"A/B 同一批物体 ⇒ 逐条相同"。实测:
    条数/帧号相等、排序后 2D 差 **≤0.78 px**、3D 差 **≤0.06 m**(正是 A/B 硬门槛的量级),
    **行序还不同**(CARLA `get_actors()` 跨采集不稳定)。这 0.78 px 足以让一条压在
    `IoU=0.5` 上的检测翻面 ⇒ 同一份预测换一套 GT 差 **0.009 AP**。
    ⇒ 单变量只剩"图"这条前提,靠**冻结 GT** 保证,不能靠"以为它们一样"。
    """

    @staticmethod
    def _mk(root, n=4, dx=0.0):
        from PIL import Image

        for sub in ("image_2", "label_2"):
            (root / "training" / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(root / f"training/image_2/{i:06d}.png")
            (root / f"training/label_2/{i:06d}.txt").write_text(
                f"Car 0 0 0 {dx:.2f} 0 {dx + 1:.2f} 1 1 1 1 1 0 0 0 0\n", encoding="utf-8"
            )

    def _run(self, tmp_path):
        from PIL import Image

        clear, fog, gen = tmp_path / "c", tmp_path / "f", tmp_path / "g"
        self._mk(clear)
        self._mk(fog, dx=0.78)  # ← A/B 实测的那种抖动:同一物体差 0.78 px
        gen.mkdir()
        for i in range(4):
            Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(gen / f"gen_gt-depth_{i:05d}_0.png")
        work = tmp_path / "w"
        D.run_arms(clear, fog, gen, work)
        return work

    def test_all_four_arms_carry_the_same_gt_bytes(self, tmp_path):
        work = self._run(tmp_path)
        dirs = sorted(
            (work / a / "training/label_2") for a in ("A_clear", "B_truth_fog", "C_gen_fog", "D_gen_shift")
        )
        first = {p.name: p.read_text(encoding="utf-8") for p in sorted(dirs[0].glob("*.txt"))}
        for d in dirs[1:]:
            assert {p.name: p.read_text(encoding="utf-8") for p in sorted(d.glob("*.txt"))} == first, (
                f"{d.parent.name} 的 GT 与 A_clear 不同 —— 单变量被污染了"
            )

    def test_fog_arms_own_gt_is_deliberately_ignored(self, tmp_path):
        """反向自证:把 B 侧 GT 改到**明显不同**,四臂仍必须逐字节相同(否则测的是空转)。"""
        work = self._run(tmp_path)
        b_gt = (work / "B_truth_fog/training/label_2/000000.txt").read_text(encoding="utf-8")
        a_gt = (work / "A_clear/training/label_2/000000.txt").read_text(encoding="utf-8")
        assert b_gt == a_gt
        assert "0.78" not in b_gt, "冻结失败 —— 带上了 fog 侧那份抖动过的 GT"


class TestGenImageFiles:
    def test_only_picks_index_zero(self, tmp_path):
        """★ 只挑 `_0.png` —— `--n > 1` 时全收会让帧数与 GT 不等;
        而误收 `cond_*.png` 则会"拿条件图当生成图评",AP 照样像个数。"""
        for n in ("gen_gt-depth_00000_0.png", "gen_gt-depth_00000_1.png", "cond_gt-depth_minmax_00000.png"):
            (tmp_path / n).write_bytes(b"")
        got = D.gen_image_files(tmp_path)
        assert [p.name for p in got] == ["gen_gt-depth_00000_0.png"]

    def test_missing_raises_with_the_command(self, tmp_path):
        with pytest.raises(SystemExit, match="conditioned_gen"):
            D.gen_image_files(tmp_path)


class TestVerdict:
    """三档裁决。**两头都要能分辨** —— 那是这一层存在的全部理由。"""

    def test_replaceable(self):
        assert "可替代" in D._verdict(-0.30, -0.28, -0.90)

    def test_too_poisonous(self):
        """生成掉得比真值退化更狠 ⇒ 过毒。"""
        assert "过毒" in D._verdict(-0.30, -0.70, -0.90)

    def test_fake_degradation(self):
        """几乎不掉点 ⇒ 假退化(没真的加雾)。"""
        assert "假退化" in D._verdict(-0.30, -0.01, -0.90)

    def test_layout_lost_is_reported_as_such(self):
        """★★ **最关键的一条**:Δ_gen 与**对照地板**不可分辨时,
        不能说"过毒",要说"**布局丢了**"。两者都是低 AP,但含义完全相反
        (一个是"退化太重",一个是"这根本不是同一个场景")。"""
        v = D._verdict(-0.30, -0.88, -0.90)
        assert "对不上" in v and "过毒" not in v

    def test_weaker_than_truth(self):
        assert "偏弱" in D._verdict(-0.50, -0.20, -0.90)

    def test_floor_arm_is_never_read_as_a_real_effect(self):
        """自证:把地板原样当读数喂进去,裁决必须是"对不上",不是任何"有效果"的档。"""
        assert "对不上" in D._verdict(-0.4, -0.9, -0.9)


class TestFlatGenFilenames:
    """★ 回归钉:**扁平生成名的 `stem` 不是帧号,不许按 `frames` 筛**。

    2026-10-06 实测卡了一轮:`gen_gt-depth_00000_0.png` 的 stem 是
    `gen_gt-depth_00000_0`,拿它跟 `"000000"` 比必然不中 ⇒ 图被滤成空,
    而报错只说"为空",**指不到真因**。
    """

    def test_image_files_survive_frame_filter(self, tmp_path):
        from PIL import Image

        labels = tmp_path / "lab"
        labels.mkdir()
        for i in range(4):
            (labels / f"{i:06d}.txt").write_text("Car 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0\n", encoding="utf-8")
        gen = tmp_path / "gen"
        gen.mkdir()
        files = []
        for i in range(4):
            p = gen / f"gen_gt-depth_{i:05d}_0.png"
            Image.fromarray(np.full((4, 4, 3), i, np.uint8)).save(p)
            files.append(p)
        dst = tmp_path / "arm"
        keep = ["000000", "000001", "000002"]
        n = D.build_arm(dst, image_files=files, label_src=labels, frames=keep)
        assert n == 3
        assert len(list((dst / "training/image_2").glob("*.png"))) == 3


class TestGridPointsIsWired:
    """★★ **`--grid-points` 必须真的传到 `evaluate`**(2026-10-07 补)。

    为什么不加这条不行:`report` 的默认是 **11 点**(归档口径),
    而 §1.10 量过 11 点下 AP 是**台阶函数** —— 同一批检测、同一批图,
    `gt-depth` 那一格实测从 **+0.0032(11 点)翻成 −0.0617(101 点)**,**符号都翻了**。
    ⇒ 参数没接上时**不报错、照样出数**,只是结论可能反号。
    """

    def test_report_forwards_n_points(self, monkeypatch, tmp_path):
        called = {}

        def fake_evaluate(root, predict, conf, iou, limit, *, n_points=11, verbose=True):
            called["n_points"] = n_points
            return {"mAP": 0.0}

        monkeypatch.setattr(AB, "evaluate", fake_evaluate)
        AB.report(tmp_path, object(), 0.25, 0.5, None, n_points=101)
        assert called["n_points"] == 101

    def test_report_defaults_to_the_archive_grid(self, monkeypatch, tmp_path):
        """默认必须是 11 —— 改了会让已引用的 §1.6 四臂数**不可比**。"""
        called = {}

        def fake_evaluate(root, predict, conf, iou, limit, *, n_points=11, verbose=True):
            called["n_points"] = n_points
            return {"mAP": 0.0}

        monkeypatch.setattr(AB, "evaluate", fake_evaluate)
        AB.report(tmp_path, object(), 0.25, 0.5, None)
        assert called["n_points"] == 11

    def test_main_passes_the_cli_value_through(self):
        """结构钉:调用点必须写 `n_points=args.grid_points`(漏了只会静默按 11 点跑)。"""
        src = inspect.getsource(D.main)
        assert "--grid-points" in src
        assert "n_points=args.grid_points" in src

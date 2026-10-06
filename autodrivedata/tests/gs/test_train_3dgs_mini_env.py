"""`gs/train_3dgs_mini` 的**训练环境**回归钉(纯值,不连 CARLA、不训模型)。

两组都钉同一类失效:**不报错、不改变任何输出形状,只让结果悄悄变了**。
两条底层的机制也一样 —— **设置与使用之间隔着别的东西**。

**① `--seed` 是假的(2026-10-04 修)**:这个模块原先**全程没有 seed**,同一份代码两次跑
val_psnr 实测差 4.5 dB(15.74 vs 11.29)⇒ 任何 A/B 都判不了。补 `--seed` 时补丁**只加在前面**,
`main` 里原有的 `torch.manual_seed(0)` / `np.random.seed(0)` 原封不动留在后面 ⇒
**把刚设的种子静默覆盖回 0**:`--seed 5` 与 `--seed 0` 逐位相同(参数是摆设)、
`--seed -1`(help 写"回到旧的自由随机行为")同样被确定化 —— 两处说明都是假的。

**② `CUDA_HOME` 不设会**静默换掉规范 `.so`**(2026-10-04 修,实测代价 ≈1 h)**:
torch 的扩展缓存按**构建 flags 的 hash** 定位 `.so` —— 环境不同 ⇒ hash 不同 ⇒
它**重编并覆盖**,而不是报"环境不对"。补了 CUDA 13 头文件路径那一次**构建是成功的**,
产出的 `.so` 要 `libcudart.so.13`;**「上次能跑」不构成任何保证**。详见
[docs/edit-3dgs-plan.md](../../../docs/edit-3dgs-plan.md) §A.4.0。

⇒ 判据据此分三层:① helper/常量的**行为**;② **源码顺序**(设置必须排在被使用之前,
同 `tests/map/test_maptr_select.py::TestCheckpointFinalizeOrder` 的纪律);
③ 归属信息**进了 runlog**(跨 build 的数不许混比,而那要求事后查得出是哪次 build)。
"""

from __future__ import annotations

import inspect

import numpy as np
import torch

from autodrivedata.gs import train_3dgs_mini as T


class TestSeedEverything:
    def test_pins_torch_and_numpy(self):
        T._seed_everything(5)
        assert torch.initial_seed() == 5
        # numpy 没有"读回当前种子"的接口 ⇒ 与一次干净的 seed(5) 比抽出来的序列。
        drawn = np.random.rand(4)
        np.random.seed(5)
        assert np.array_equal(drawn, np.random.rand(4))

    def test_different_seeds_give_different_streams(self):
        """★ 这一条就是原来那个 bug 的直接反例:换 seed 必须换结果。"""
        T._seed_everything(0)
        a = np.random.rand(4)
        T._seed_everything(5)
        b = np.random.rand(4)
        assert not np.array_equal(a, b)

    def test_negative_seed_leaves_state_alone(self):
        """`--seed -1` = 不动上游状态(help 里许诺的"自由随机")。"""
        np.random.seed(7)
        before = np.random.rand()
        np.random.seed(7)
        T._seed_everything(-1)
        assert np.random.rand() == before


class TestNoClobber:
    """★ 源码顺序钉:`_seed_everything` 之后、读数据之前,不许再有任何 seed 调用。"""

    def _window(self) -> list[str]:
        src = inspect.getsource(T.main)
        call = "_seed_everything(args.seed)"
        assert call in src, "main 里没调 _seed_everything —— 种子根本没设"
        start = src.index(call) + len(call)
        end = src.index("_load_poses_and_cams(", start)
        # 只看代码行:注释里提到 `manual_seed` 是**解释**这个坑,不是在犯它。
        return [ln for ln in src[start:end].splitlines() if not ln.strip().startswith("#")]

    def test_no_seeding_between_seed_call_and_data_load(self):
        # 认**调用**而不是认词:`rl.highlight("seed", args.seed)` 里有 "seed" 却无害,
        # 而真正的覆盖长成 `xxx.seed(...)` 或 `manual_seed(...)`。
        offenders = [ln.strip() for ln in self._window() if "manual_seed" in ln or ".seed(" in ln]
        assert not offenders, f"`_seed_everything` 之后又被 seed 了一次(会覆盖它):{offenders}"


class TestCudaNote:
    """CUDA 环境的**归属信息**。

    ⚠️ 2026-10-04 这块**从 `train_3dgs_mini` 搬去了 `gs/cuda_env.py`** —— 因为同样要
    import gsplat 的入口多了三个(`render_gs` / `edit_gs` / `eval_edit`)。判据也跟着搬:
    在这里钉的是「**训练侧确实用了那一份**」,口径本身的判据在 `test_cuda_env.py`。
    """

    def test_train_uses_the_shared_cuda_env(self):
        """★ 训练侧不许自带一套:它得用 `cuda_env` 的两个 NOTE,并且 `home()` 进了 runlog。"""
        src = inspect.getsource(T)
        assert "cuda_env.ARCH_NOTE" in src
        assert "cuda_env.CUDA_NOTE" in src
        assert 'rl.highlight("cuda_home", cuda_env.home())' in src
        # 反过来钉:这两条常量**不许**再长回训练模块里
        assert 'CUDA_NOTE = ""' not in src
        assert 'ARCH_NOTE = ""' not in src

    def test_gsplat_is_taken_through_the_guard(self):
        """★ 裸 `import gsplat` 会被 `ruff format --fix` 挪到 `cuda_env` 之前 —— 实测坏过一次。"""
        # ⚠️ 只认**代码行**:注释与 docstring 里提 `import gsplat` 是在**解释**这个坑。
        offenders = [
            ln.strip() for ln in inspect.getsource(T).splitlines() if ln.strip().startswith("import gsplat")
        ]
        assert not offenders, f"训练侧必须经 `cuda_env.ensure_gsplat()` 取 gsplat,而不是 {offenders}"


class TestArtifactProvenance:
    """★ `train_result{tag}.json` 是**随模型一起走的那份说明书**(runlog 是另一条链)。

    口径类字段(`cam_convention` / `densify` / `downsample` / `val_frames`)一直都有;
    **`seed` 原先缺**(2026-10-04 补)—— 没有它,拿到文件也复不出同一次训练。
    这里钉的是"这几个字段不许再掉出去",不钉取值(取值由真训练产出)。
    """

    REQUIRED = ("seed", "cuda_home", "cam_convention", "densify", "downsample", "val_frames")

    def test_result_dict_carries_the_provenance_keys(self):
        src = inspect.getsource(T)
        i = src.index("result = {")
        block = src[i : src.index("\n        }", i)]
        missing = [k for k in self.REQUIRED if f'"{k}":' not in block]
        assert not missing, f"`train_result*.json` 掉了归属字段:{missing}"


class TestDefaultConvention:
    """★ 相机系的默认值(`legacy` → `carla`,2026-10-04 用户裁决)。

    这个默认值**翻过一次**,而且两处各写一个字面量正是"改了一处、另一处还是旧值"的
    静默失效 —— 所以它被抽成 `DEFAULT_CAM_CONVENTION`,这里钉的就是"**只有一个落点**"。
    """

    def test_constant_is_carla(self):
        assert T.DEFAULT_CAM_CONVENTION == "carla"

    def test_function_signature_takes_the_constant(self):
        p = inspect.signature(T._load_poses_and_cams).parameters["cam_convention"]
        assert p.default == "carla"
        assert p.default is T.DEFAULT_CAM_CONVENTION or p.default == T.DEFAULT_CAM_CONVENTION

    def test_cli_default_is_the_constant_not_a_literal(self):
        src = inspect.getsource(T)
        i = src.index('"--cam-convention"')
        # 只取这一条 add_argument 的范围:下一条 add_argument 之前。
        block = src[i : src.index("add_argument(", i + 10)]
        assert "default=DEFAULT_CAM_CONVENTION" in block, "CLI 默认没有引用常量"
        # 反过来钉:任何**写死**的默认值都不许再出现(那正是本次要修的形态)。
        for lit in ('default="legacy"', "default='legacy'", 'default="carla"', "default='carla'"):
            assert lit not in src, f"又出现了写死的默认值:{lit}"

    def test_legacy_is_still_selectable(self):
        """翻默认 ≠ 删开关:归档的 `means*.npy` 还要靠它复现。"""
        assert 'choices=("carla", "legacy")' in inspect.getsource(T)

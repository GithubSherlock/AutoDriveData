"""`gs/render_gs` 的回归钉。

两条被测契约都是**不报错的错**:

- **存盘口径**:`opac` 存的是**激活后**的值,而渲染只 `clamp` —— 谁来"顺手再 sigmoid 一次",
  图看着仍像回事,数全错。`rots` 缺了更狠:渲染**照样能跑**(gsplat 不在乎你给的是不是
  训练出来的四元数),只是结果不是那个模型。
- **两份实现**:训练时的渲染与编辑后的渲染必须是同一套 —— 否则"编辑后看起来不对"
  会被归因到编辑本身。

夹具用**极少量高斯**(几十个),显存需求可忽略 ⇒ 与 MapTR 重训等大任务并存也能跑。
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest
import torch

from autodrivedata.gs import render_gs
from autodrivedata.gs import train_3dgs_mini as T
from autodrivedata.gs.render_gs import GaussianSet, load_set, mse_to_psnr, parse_frames, save_set


def _tiny(n: int = 16, seed: int = 0) -> GaussianSet:
    rng = np.random.default_rng(seed)
    return GaussianSet(
        means=(rng.normal(size=(n, 3)) * 0.5).astype(np.float32),
        rots=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1)),
        scales_lin=np.full((n, 3), 0.05, dtype=np.float32),
        col=rng.random((n, 3)).astype(np.float32),
        opac=np.full(n, 0.8, dtype=np.float32),
    )


class TestFrameSpec:
    def test_range_list_all(self):
        assert parse_frames("0-2", 5) == [0, 1, 2]
        assert parse_frames("0,4", 5) == [0, 4]
        assert parse_frames("all", 3) == [0, 1, 2]

    def test_dedupes_and_sorts(self):
        assert parse_frames("2,0,2", 5) == [0, 2]

    def test_out_of_range_raises_not_truncates(self):
        """★ 静默截断会让"我只渲了前 10 帧"被读成"这份模型只有 10 帧好"。"""
        with pytest.raises(SystemExit, match="超出"):
            parse_frames("0-9", 5)


class TestSetRoundTrip:
    def test_save_then_load_is_bit_identical(self, tmp_path):
        gs = _tiny()
        save_set(gs, tmp_path, "t")
        back = load_set(tmp_path, "t")
        for f in ("means", "rots", "scales_lin", "col", "opac"):
            assert np.array_equal(getattr(gs, f), getattr(back, f)), f

    def test_missing_rots_raises(self, tmp_path):
        """★ 这条是 2026-10-04 那个缺陷的**直接反例**:缺 `rots` 必须当场抛。

        在那之前盘上真就没有 `rots` —— 而缺了它渲染**照样跑得起来**,只是结果
        不是训练出来的那个模型。判据必须是"缺了就停",不是"跑了就算"。
        """
        gs = _tiny()
        save_set(gs, tmp_path, "t")
        (tmp_path / "rots_t.npy").unlink()
        with pytest.raises(SystemExit, match="rots"):
            load_set(tmp_path, "t")

    def test_ragged_fields_raise(self, tmp_path):
        gs = _tiny(8)
        save_set(gs, tmp_path, "t")
        np.save(tmp_path / "col_t.npy", np.zeros((7, 3), dtype=np.float32))
        with pytest.raises(SystemExit, match="长度不一致"):
            load_set(tmp_path, "t")


#: ★ 凡是**真调 gsplat** 的用例,都必须先有规范的 CUDA 环境。
#:
#: 不给 `CUDA_HOME` 时 torch 的扩展缓存**按 flags 的 hash 定位 `.so`** —— 环境不同 ⇒
#: hash 不同 ⇒ **它重编并（可能）覆盖规范那份**。2026-10-04 同一天踩中三次
#: （一次训练、一次探针、一次就是这个测试套件），累积代价 ≈1 h。
#: ⇒ 条件不满足时**跳过并说明**,不是静默通过 —— 跳过在 pytest 里是**可见的第三态**,
#: 而"跑了但跑的是被重编过的东西"看不出来。
_NEEDS_CUDA_ENV = pytest.mark.skipif(
    not (os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")),
    reason="未设 CUDA_HOME ⇒ torch 会重编 gsplat 扩展、可能覆盖规范 .so;"
    "带 `CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH` 再跑",
)


@_NEEDS_CUDA_ENV
class TestRasterize:
    def test_render_shape_and_nonzero(self):
        """几十个高斯、一张小图 —— 只验管道通,不验画质。"""
        gs = _tiny(64, seed=1)
        dev = "cuda"
        p = gs.to_tensors(dev)
        vm = torch.eye(4, dtype=torch.float32, device=dev)[None].repeat(1, 1, 1)
        vm[0, 2, 3] = 3.0  # 相机在 z=3 处朝向 -z(gsplat 的 +z 是光轴)
        vm[0, :3, :3] = torch.tensor(
            [[-1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]], dtype=torch.float32, device=dev
        )
        # ⚠️ gsplat 要的是 `Ks [C,3,3]` —— 与 viewmats 的批维对齐,不是裸 `[3,3]`
        k = torch.tensor([[[50.0, 0, 32.0], [0, 50.0, 24.0], [0, 0, 1.0]]], device=dev)
        colors = render_gs.rasterize(
            p["means"], p["rots"], p["scales_lin"], p["opac"], p["col"], vm, k, 64, 48
        )[0]
        assert colors.shape == (1, 48, 64, 3)
        assert float(colors.max()) > 0.0, "一个高斯都没画出来 —— 位姿/内参的口径对不上"

    def test_opacity_is_not_double_activated(self):
        """★ `opac_act` 是**已 sigmoid** 的 ⇒ 渲染只 clamp。

        判据不看实现看**行为**:把同一组不透明度按"再 sigmoid 一次"喂进去,结果必须不同。
        这条钉的是"没人能顺手改回双重激活"。
        """
        gs = _tiny(32, seed=2)
        dev = "cuda"
        p = gs.to_tensors(dev)
        vm = torch.eye(4, dtype=torch.float32, device=dev)[None]
        vm[0, 2, 3] = 3.0
        vm[0, :3, :3] = torch.tensor(
            [[-1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]], dtype=torch.float32, device=dev
        )
        # ⚠️ gsplat 要的是 `Ks [C,3,3]` —— 与 viewmats 的批维对齐,不是裸 `[3,3]`
        k = torch.tensor([[[50.0, 0, 32.0], [0, 50.0, 24.0], [0, 0, 1.0]]], device=dev)
        right = render_gs.rasterize(
            p["means"], p["rots"], p["scales_lin"], p["opac"], p["col"], vm, k, 64, 48
        )[0]
        wrong = render_gs.rasterize(
            p["means"], p["rots"], p["scales_lin"], p["opac"].sigmoid(), p["col"], vm, k, 64, 48
        )[0]
        assert not torch.allclose(right, wrong), (
            "双重激活与单次激活结果相同 —— 那说明夹具的 opac 恰好在 0/1 附近,"
            "这条判据失去分辨力,要换夹具而不是放过它"
        )


class TestPsnr:
    def test_known_value(self):
        assert mse_to_psnr(0.01) == pytest.approx(20.0, abs=1e-6)

    def test_zero_mse_is_finite(self):
        assert np.isfinite(mse_to_psnr(0.0))


class TestVerifyTolerance:
    """★ C0.1 判据的容差**由产物的存储精度决定**,不是由渲染决定。

    第一版把容差写成 `1e-3` —— 一条**完全正确**的渲染被它判红过:
    逐帧 Δ **全 0**,只有均值差 0.0013,因为 `train_result*.json` 里的 PSNR 是
    `round(..., 2)` 存的,拿未舍入的复算值去比已舍入的存值,**差最多 0.005 是必然的**。
    ⇒ 判据必须比在**存值的精度**上,否则它会逼人把容差一路放宽到失去意义。
    """

    def test_tolerance_is_half_the_stored_quantum(self):
        assert render_gs.ROUND_DP == 2
        assert render_gs.TOL_DB == pytest.approx(0.005, abs=1e-12)

    def test_round_dp_still_matches_how_the_artifact_rounds(self):
        """**耦合钉**:产物换了落盘精度,这条会红,提醒改 `ROUND_DP`(否则容差悄悄失配)。"""
        src = inspect.getsource(T)
        assert "round(float(psnrs.mean()), 2)" in src
        assert "round(float(psnrs.min()), 2)" in src
        assert "round(float(psnrs[i]), 2)" in src

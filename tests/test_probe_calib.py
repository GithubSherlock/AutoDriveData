"""`bin/probe_calib.py` 的纯值部分单测:实例分割解码口径 + 锚点判据的合成数据自证。

**为什么钉实例分割编码**(实测裁决,2026-09-22):CARLA 官方文档只说"每个实例一个唯一颜色",
**不写通道序**。旧实现猜了两种(`R+256G+65536B` / `B+256G+65536R`)且**两种都错**,后果是
A2/A4 一路带着空掩膜 —— 直到 `instance_decode_mode` 抛 `RuntimeError` 才暴露。差分法实测
(同一位姿先后取帧,只比 actor 投影点邻域的像素)裁决为:

    BGRA = [B, G, R, A]   ⇒   id = G + 256·B   (低字节在 G、高字节在 B)

证据:4/4 命中且跨类型(锥 660 / 车 661 / 交通锥 662 / 行人 663),并跨 255 边界验证高字节。
本文件用**合成缓冲**把这条公式钉死 —— 不依赖 CARLA 服务器(只 import 模块,缺 carla 整体跳过)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("carla")

BIN = Path(__file__).resolve().parents[1] / "bin"
if str(BIN) not in sys.path:  # bin/ 不是包,按脚本目录加路径(同 test_live_common)
    sys.path.insert(0, str(BIN))

import bin.probe_calib as pc  # noqa: E402


def _bgra(ids: np.ndarray, semantic: int = 21) -> bytes:
    """按实测编码把 id 数组打包成 CARLA 实例分割 BGRA 缓冲。"""
    flat = np.asarray(ids, dtype=np.int64).ravel()
    arr = np.zeros((flat.size, 4), dtype=np.uint8)
    arr[:, 0] = (flat >> 8) & 0xFF  # B = 高字节
    arr[:, 1] = flat & 0xFF  # G = 低字节
    arr[:, 2] = semantic  # R = CityObjectLabel
    arr[:, 3] = 255  # A
    return arr.tobytes()


class TestDecodeInstance:
    def test_recovers_ids_roundtrip(self):
        ids = np.array([[0, 1, 45], [255, 256, 660]])
        got = pc.decode_instance(_bgra(ids), 2, 3)
        assert got.shape == (2, 3)
        assert np.array_equal(got, ids)

    def test_high_byte_is_the_blue_channel(self):
        """跨 255 边界:id 300 → BGRA `[1, 44, *, 255]`。旧候选口径在这里会给出 44 或 11264。"""
        raw = _bgra(np.array([[300]]))
        assert raw[:4] == bytes([1, 44, 21, 255])
        assert int(pc.decode_instance(raw, 1, 1)[0, 0]) == 300

    def test_old_candidate_orders_would_disagree(self):
        """反例钉死:两个旧候选口径在 id ≥ 256 时与真值不同(故当初"自证"选不出对的)。"""
        arr = np.frombuffer(_bgra(np.array([[660]])), dtype=np.uint8).astype(np.int64).reshape(1, 1, 4)
        b, g, r = (int(arr[0, 0, k]) for k in (0, 1, 2))
        assert pc.decode_instance(_bgra(np.array([[660]])), 1, 1)[0, 0] == 660
        assert r + 256 * g + 65536 * b != 660
        assert b + 256 * g + 65536 * r != 660

    def test_semantic_channel_does_not_leak_into_id(self):
        """R 通道是语义类,不参与 id —— 换语义类 id 必须不变。"""
        ids = np.array([[1234]])
        assert pc.decode_instance(_bgra(ids, semantic=14), 1, 1)[0, 0] == 1234
        assert pc.decode_instance(_bgra(ids, semantic=21), 1, 1)[0, 0] == 1234

    def test_hits_and_masks(self):
        decoded = {"CAM_FRONT": pc.decode_instance(_bgra(np.array([[7, 7, 9]])), 1, 3)}
        assert pc.instance_hits(decoded, [7]) == 2
        assert pc.cone_masks(decoded, 7)["CAM_FRONT"].sum() == 2

    def test_missing_hit_fails_loudly(self):
        """解码口径若与渲染器不符 ⇒ 响亮失败,而不是静默给空掩膜。"""
        decoded = {"CAM_FRONT": pc.decode_instance(_bgra(np.array([[7] * 20])), 1, 20)}
        with pytest.raises(RuntimeError, match="未命中"):
            pc.require_instance_hits(decoded, [999])


class TestAnchorsArePureValue:
    """A0/A1 是纯值锚(不需要 CARLA 服务器),且历史镜像 bug 必须在这里被拦下。"""

    def test_a0_azimuth_matches_official(self):
        r = pc.anchor_azimuth()
        assert r["max_abs_diff_deg"] < 0.01
        assert len(r["rows"]) == 6

    def test_a1_all_side_cameras_same_side(self):
        r = pc.anchor_side_consistency()
        assert r["n_checked"] == 4  # 前/后近自逆,判据对它们无信息量 → 如实排除
        assert r["all_same_side"]

    def test_a1_would_catch_the_mirror_bug(self):
        """把四个侧相机的偏航取反(历史 bug 的形状)⇒ A1 必须翻转成 False。"""
        mounts, rots = pc.rig_spec(pc.RIG_NUSCENES)
        mirrored = {n: (rots[n][0], -rots[n][1], rots[n][2]) for n in rots}
        r = pc.anchor_side_consistency(spec=(mounts, mirrored))
        assert not r["all_same_side"]

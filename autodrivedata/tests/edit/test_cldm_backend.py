"""`edit/cldm_backend` 的判据回归钉(**不出模型、不读 5.71 GB 权重**)。

这个模块的每一次适配都是**静默型**的:CLIP 前缀重映射错了不会报错(它会走到
`load_state_dict` 报一堆不相干的键),权重校验图快的话会把"尺寸对但内容坏"放过去。
⇒ 每条判据配一条**反向自证**。
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from autodrivedata.edit import cldm_backend as B


class TestManifest:
    def test_parse(self, tmp_path):
        m = tmp_path / B.MANIFEST_NAME
        m.write_text(
            "# 注释行\na.pth  size=12  sha256=deadbeef\n\nb.pt  size=3  sha256=cafe\n",
            encoding="utf-8",
        )
        got = B.read_manifest(m)
        assert got == {"a.pth": "deadbeef", "b.pt": "cafe"}

    def test_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="缺 MANIFEST"):
            B.read_manifest(tmp_path / B.MANIFEST_NAME)


class TestVerifyWeight:
    def _mk(self, tmp_path, data=b"hello", sha=None, name="w.pth"):
        p = tmp_path / name
        p.write_bytes(data)
        (tmp_path / B.MANIFEST_NAME).write_text(
            f"{name}  size={len(data)}  sha256={sha or hashlib.sha256(data).hexdigest()}\n", encoding="utf-8"
        )
        return p

    def test_ok(self, tmp_path):
        p = self._mk(tmp_path)
        assert B.verify_weight(p, expected=hashlib.sha256(b"hello").hexdigest())

    def test_mismatch_raises(self, tmp_path):
        """★ 核心判据:内容坏必须抛。"""
        p = self._mk(tmp_path, data=b"hello")
        with pytest.raises(ValueError, match="权重校验失败"):
            B.verify_weight(p, expected="0" * 64)

    def test_unregistered_raises(self, tmp_path):
        """新下的权重没登记 sha256 ⇒ 抛。不登记就等于没校验。"""
        p = tmp_path / "new.pth"
        p.write_bytes(b"x")
        (tmp_path / B.MANIFEST_NAME).write_text("other.pth  sha256=aa\n", encoding="utf-8")
        with pytest.raises(KeyError, match="MANIFEST 里没有"):
            B.verify_weight(p)

    def test_sidecar_caches_but_not_masks(self, tmp_path):
        """★ 边车**只加速不替代**:改了文件(mtime 变)必须重新算并报错。"""
        p = self._mk(tmp_path, data=b"hello")
        exp = hashlib.sha256(b"hello").hexdigest()
        B.verify_weight(p, expected=exp)
        side = p.with_name(p.name + ".sha256")
        assert side.exists()

        # 缓存命中:不再重算(把 expected 换个错的也照样返回缓存值 —— 说明走的是缓存路径)
        assert B.verify_weight(p, expected=exp) == exp

        # 内容被换掉 + mtime 变 ⇒ 缓存失效 ⇒ 必须抛
        p.write_bytes(b"tampered")
        import os
        import time

        os.utime(p, (time.time() + 10, time.time() + 10))
        with pytest.raises(ValueError, match="权重校验失败"):
            B.verify_weight(p, expected=exp)

    def test_bad_sidecar_falls_back_to_rehash(self, tmp_path):
        """边车自己坏了(哈希与 MANIFEST 不一致)⇒ **不算命中**,回退去重算。

        ⚠️ 第一版这里写的是 `pytest.raises` —— **红的是测试不是代码**:
        文件本身是好的,坏边车**不该**让校验失败,它只该让缓存失效。
        (区分点:抛错说的是"文件坏",回退说的是"缓存不可信"。)
        """
        p = self._mk(tmp_path, data=b"hello")
        exp = hashlib.sha256(b"hello").hexdigest()
        st = p.stat()
        side = p.with_name(p.name + ".sha256")
        side.write_text(
            json.dumps({"size": st.st_size, "mtime": int(st.st_mtime), "sha256": "f" * 64}), encoding="utf-8"
        )
        assert B.verify_weight(p, expected=exp) == exp
        assert json.loads(side.read_text())["sha256"] == exp, "回退后应当把边车修正"

    def test_sidecar_boundary_mtime(self, tmp_path):
        """★ **明说的边界**:保住 mtime + 边车声称 MANIFEST 的哈希 ⇒ **会被接受而不重算**。

        这条**不是在测一个缺陷**,是在把边界**钉住**:
        本判据防的是"下载坏了",不是"有人篡改"(威胁模型里没有后者)。
        哪天要防篡改,这里会红 —— 那正是"该给边车加签名"的提示。
        """
        p = self._mk(tmp_path, data=b"hello")
        exp = hashlib.sha256(b"hello").hexdigest()
        st = p.stat()
        # ★ 篡改必须**等长** —— 第一版这里写成 `b"tampered!"`(9 字节 vs 5 字节),
        #   size 一变缓存自然失效、判据照常抛出,于是这条测的不是"mtime 边界"而是"size 变了"。
        p.write_bytes(b"HELLO")
        import os

        os.utime(p, (st.st_atime, st.st_mtime))  # 但 size 与 mtime 都保住了
        p.with_name(p.name + ".sha256").write_text(
            json.dumps({"size": st.st_size, "mtime": int(st.st_mtime), "sha256": exp}), encoding="utf-8"
        )
        assert B.verify_weight(p, expected=exp) == exp  # ← 已知边界,接受

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="缺权重"):
            B.verify_weight(tmp_path / "nope.pth", expected="aa")


class TestRemapClipKeys:
    def test_strips_text_model_prefix(self):
        model_sd = {"cond_stage_model.transformer.encoder.w": 1, "unet.w": 2}
        sd = {"cond_stage_model.transformer.text_model.encoder.w": 3, "unet.w": 4}
        out = B.remap_clip_keys(sd, model_sd)
        assert out["cond_stage_model.transformer.encoder.w"] == 3
        assert out["unet.w"] == 4

    def test_drops_position_ids_buffer(self):
        model_sd = {"a": 1}
        sd = {"a": 2, "cond_stage_model.transformer.text_model.embeddings.position_ids": 3}
        assert B.remap_clip_keys(sd, model_sd) == {"a": 2}

    def test_does_not_mutate_input(self):
        sd = {"cond_stage_model.transformer.text_model.a": 1}
        model_sd = {"cond_stage_model.transformer.a": 1}
        B.remap_clip_keys(sd, model_sd)
        assert list(sd) == ["cond_stage_model.transformer.text_model.a"]

    def test_missing_keys_raises(self):
        """★★ **反向自证**:缺键必须抛。

        这不是"多一层保险" —— 若改成宽松加载(把缺的键静默跳过),
        `load_state_dict` 会把它们当随机初始化,**模型照样能跑、照样出图**,
        只是 CLIP 那一半是白噪声。**这类错没有任何症状。**
        """
        with pytest.raises(RuntimeError, match="仍有 1 个键缺失"):
            B.remap_clip_keys({"only_this": 1}, {"only_this": 1, "missing_that": 2})

    def test_unrelated_keys_pass_through(self):
        """**只动那一个前缀**,不碰别的键 —— 别的键不匹配时应当交给 load_state_dict 报错。"""
        model_sd = {"weird.key": 1}
        sd = {"weird.key": 2}
        assert B.remap_clip_keys(sd, model_sd) == {"weird.key": 2}


class TestToSquare:
    def test_center_crops(self):
        a = np.zeros((4, 10, 3), dtype=np.uint8)
        out = B.to_square_rgb(a)
        assert out.shape == (4, 4, 3)

    def test_square_passthrough_is_identical(self):
        a = np.arange(3 * 3 * 3, dtype=np.uint8).reshape(3, 3, 3)
        assert B.to_square_rgb(a) is a

    def test_real_frame_shape(self):
        """本项目的采集帧是 1242×375 ⇒ 裁成 375×375。"""
        a = np.zeros((375, 1242, 3), dtype=np.uint8)
        assert B.to_square_rgb(a).shape == (375, 375, 3)

    def test_non_three_channel_raises(self):
        with pytest.raises(ValueError, match=r"\(H,W,3\)"):
            B.to_square_rgb(np.zeros((4, 4, 4), dtype=np.uint8))


class TestVariantTable:
    def test_variants_have_distinct_weights(self):
        """两个变体必须指向**不同**的文件 —— 同名会让"换了变体"变成空操作。"""
        names = list(B.VARIANT_WEIGHTS.values())
        assert len(names) == len(set(names))

    def test_unknown_variant_raises(self, tmp_path, monkeypatch):
        with pytest.raises(KeyError, match="未接的变体"):
            B.load_cldm("openpose_hand", verify=False)


class TestLoadRgb:
    def test_reads_png(self, tmp_path):
        from PIL import Image

        p = tmp_path / "a.png"
        Image.fromarray(np.zeros((5, 7, 3), dtype=np.uint8)).save(p)
        assert B.load_rgb(p).shape == (5, 7, 3)

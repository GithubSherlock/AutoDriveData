"""SLAM 能力面:两段式激光 SLAM(ICP 前端 + ScanContext/PGO 后端)+ 精度评估 + 累积建图。

层次见 [docs/refactor-2026-09.md](../../docs/refactor-2026-09.md) §2;本目录**纯值**(禁 carla 与 torch),
由 `tests/test_layer_guard.py` 的 `LAYER_RULES["slam"]` 强制。
"""

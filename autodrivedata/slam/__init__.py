"""SLAM 能力面:两段式激光 SLAM(ICP 前端 + ScanContext/PGO 后端)+ 精度评估 + 累积建图。

层次见 [Plan_fileTree.md](../../Plan_fileTree.md) §2;本目录**纯值**(禁 carla 与 torch),
由 `tests/test_layer_guard.py` 的 `LAYER_RULES["slam"]` 强制。
"""

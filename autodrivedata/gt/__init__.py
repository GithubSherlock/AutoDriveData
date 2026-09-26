"""GT 能力面:动态目标(actor→label_2)+ 静态目标/道路特征(地图查询)+ 灯态时序层 + 落盘导出。

层次见 [Plan_fileTree.md](../../Plan_fileTree.md) §2;本目录**纯值**(禁 carla 与 torch),
由 `tests/test_layer_guard.py` 的 `LAYER_RULES["gt"]` 强制。
"""

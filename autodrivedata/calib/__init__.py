"""标定能力面:标定原语 + 自证探针 + 实时监看 + 配置图。

层次见 [docs/refactor-2026-09.md](../../docs/refactor-2026-09.md) §2;允许 import 什么由
`tests/test_layer_guard.py` 的 `LAYER_RULES["calib"]` 强制(许 carla,禁 torch)。
"""

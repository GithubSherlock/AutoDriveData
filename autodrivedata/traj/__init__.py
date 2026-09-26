"""轨迹能力面:多 agent 轨迹 → HiVT 口径(TemporalData)组装与转换。

层次见 [docs/refactor-2026-09.md](../../docs/refactor-2026-09.md) §2;本目录**不 import carla**(许 torch),
由 `tests/test_layer_guard.py` 的 `LAYER_RULES["traj"]` 强制。
采集侧在 `sim/collect_traj.py`。
"""

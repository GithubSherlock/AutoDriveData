"""感知能力面:2D/3D 检测、单双目、雷达、语义分割、点云处理(地面/聚类)。

层次见 [Plan_fileTree.md](../../Plan_fileTree.md) §2;本目录**不 import carla**
(许 torch/ultralytics),由 `tests/test_layer_guard.py` 的 `LAYER_RULES["perception"]` 强制。
需要 CARLA 服务器的传感器探针在 `sim/`(如 `probe_radar_l3` / `probe_imu`)。
"""

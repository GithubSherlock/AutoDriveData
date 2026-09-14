#!/usr/bin/env bash
# 雷达采集冒烟(需 CARLA 服务器 + GPU,12 GiB;建议在 v1 训练结束后跑)。
# 三个"可直喂 devkit"判据:① RadarPointCloud.from_file 18 维/点数>0/过滤不掉光;
# ② NuScenes().map_pointcloud_to_image 端到端(表链 + 坐标系);③ 方向自证(前雷达 x>0)。
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-outputs/nus_mini}"
PY=/root/autodl-tmp/envs/autodrivedata/bin/python
$PY bin/collect_nus.py --out "$OUT" --frames 2

echo "=== [1/4] devkit 直读:RadarPointCloud.from_file × 5 通道(两帧内各通道至少一帧非空) ==="
$PY - "$OUT" <<'EOF'
import sys
from nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
root = sys.argv[1]
nusc = NuScenes(version="v1.0-mini", dataroot=root, verbose=False)
chans = ("RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT")
counts = {ch: [] for ch in chans}
for s in nusc.sample:
    for ch in chans:
        sd = nusc.get("sample_data", s["data"][ch])
        pc = RadarPointCloud.from_file(root + "/" + sd["filename"])
        assert pc.points.shape[0] == 18, f"{ch} dims != 18"
        counts[ch].append(pc.points.shape[1])
        # 采样帧 0 的雷达可能整帧为空(同步 tick 相位,sensor_tick=0.1 与 tick 同周期,
        # 偶发空 tick 写空 pcd)——允许样本间为空的通道,但两帧必须至少有一帧非空
for ch in chans:
    npts = max(counts[ch])
    print(f"  {ch:16s} modality=radar  npts per sample={counts[ch]}  max={npts:5d}")
    assert npts > 0, f"{ch} empty in all samples after devkit default filters!"
EOF

echo "=== [2/4] 端到端:map_pointcloud_to_image(雷达→相机,表链+坐标系) ==="
$PY - "$OUT" <<'EOF'
import sys
from nuscenes import NuScenes
root = sys.argv[1]
nusc = NuScenes(version="v1.0-mini", dataroot=root, verbose=False)
# devkit 1.2.0 把投影放进了 NuScenesExplorer:nusc.explorer.map_pointcloud_to_image
for i, n in enumerate(nusc.sample):
    pts, coloring, im = nusc.explorer.map_pointcloud_to_image(
        n["data"]["RADAR_FRONT"], n["data"]["CAM_FRONT"])
    print(f"  sample[{i}] projected {pts.shape[1]} pts on CAM_FRONT, image {im.size}")
    if pts.shape[1] > 0:
        break
else:
    raise AssertionError("radar points didn't land in front cam in any sample")
EOF

echo "=== [3/4] 方向自证:前雷达朝前、侧雷达在正确一侧 ==="
$PY - "$OUT" <<'EOF'
import sys
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
root = sys.argv[1]
nusc = NuScenes(version="v1.0-mini", dataroot=root, verbose=False)
s = next(s for s in nusc.sample if any(
    RadarPointCloud.from_file(root + "/" + nusc.get("sample_data", s["data"][ch])["filename"]).points.shape[1]
    for ch in ("RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT")
))
def _qmat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
def fr_ego(ch):
    """雷达传感器系点 → ego 系(nus: x 前/y 左/z 上):p_ego = R_calib @ p_s。"""
    sd = nusc.get("sample_data", s["data"][ch])
    cal = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pc = RadarPointCloud.from_file(root + "/" + sd["filename"])
    R = _qmat(cal["rotation"])
    return (R @ pc.points[:3, :]).T  # (N,3)
fwd = fr_ego("RADAR_FRONT")
# 前雷达点应基本都在 z=0 水平面前方(x>0)
frac_fwd = float((fwd[:, 0] > 0).mean()) if len(fwd) else 0.0
fl = fr_ego("RADAR_FRONT_LEFT")   # ego 系 y>0 = 左
fr = fr_ego("RADAR_FRONT_RIGHT")  # ego 系 y<0 = 右
frac_fl = float((fl[:, 1] > 0).mean()) if len(fl) else 0.0
frac_fr = float((fr[:, 1] < 0).mean()) if len(fr) else 0.0
print(f"  RADAR_FRONT x>0: {frac_fwd:.2f}")
print(f"  RADAR_FRONT_LEFT y>0: {frac_fl:.2f}  (期望 >0.5)")
print(f"  RADAR_FRONT_RIGHT y<0: {frac_fr:.2f}  (期望 >0.5)")
assert frac_fwd > 0.5 and frac_fl > 0.5 and frac_fr > 0.5
EOF

echo "=== [4/4] GT 关联:num_radar_pts 非负且靠近前车的框有命中 ==="
$PY - "$OUT" <<'EOF'
import sys, json
root = sys.argv[1]
anns = json.load(open(f"{root}/v1.0-mini/sample_annotation.json"))
assert all("num_radar_pts" in a for a in anns)
assert all(a["num_radar_pts"] >= 0 for a in anns)
n_hit = sum(1 for a in anns if a["num_radar_pts"] > 0)
print(f"  {len(anns)} annotations, {n_hit} with radar hits")
assert n_hit > 0, "no GT box has radar points — association broken?"
EOF

echo "=== 冒烟全部通过 ✓ ==="

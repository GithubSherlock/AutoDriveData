#!/usr/bin/env bash
# 组装 + 合并 600 帧(MapTR 扩数据管道 B2 链,输出自校验)
set -euo pipefail
cd "$(dirname "$0")/.."
NEW=${1:-outputs/surround_p3}      # 新采集根(官方布局)
FBASE=${2:-200}                    # 新数据起始帧号
OUT=outputs/maptr_600/map_infos.json
OLD=outputs/surround_train/map_infos.json
echo "[1/3] assemble $NEW"
python bin/assemble_maptr.py --surround "$NEW" --map-json training/map/Town10HD_Opt_full.json \
  --out "$NEW/map_infos.json"
echo "[2/3] merge old(0-199) + new($FBASE..$((FBASE+399)))"
python bin/merge_train_infos.py --old "$OLD" --old-n 200 --new "$NEW/map_infos.json" \
  --new-fbase "$FBASE" --new-img-root "$NEW" --out "outputs/maptr_600/map_infos.json"
echo "[3/3] 自校验:600 帧图像齐全"
python - <<'PYEOF'
import json
from pathlib import Path
inf = json.loads(Path("outputs/maptr_600/map_infos.json").read_text())
roots = {0: "outputs/surround_train", 200: "outputs/surround_p3"}
miss = [
    (r["frame"], c["data_path"])
    for r in inf
    for c in r["cams"].values()
    if not (Path(roots.get(r["frame"], "outputs/surround_p3")) / c["data_path"]).is_file()
]
assert not miss, f"缺 {len(miss)} 图: {miss[:3]}"
print(f"OK: {len(inf)} 帧, 图像全齐")
PYEOF

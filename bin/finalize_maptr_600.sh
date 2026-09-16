#!/usr/bin/env bash
# 600 帧 MapTR 数据集定稿:assemble p3 → merge → 图像集中 → infos data_path 对齐
set -euo pipefail
cd "$(dirname "$0")/.."
NEW=${1:-outputs/surround_p3}
FBASE=${2:-200}
OUT=outputs/maptr_600/map_infos.json
IMG_ROOT=outputs/maptr_600/images

bash bin/assemble_and_merge.sh "$NEW" "$FBASE"

python - "$OUT" "$IMG_ROOT" <<'PYEOF'
import json, shutil, sys
from pathlib import Path
out, img_root = Path(sys.argv[1]), Path(sys.argv[2])
inf = json.loads(out.read_text())
cam_names = sorted(inf[0]["cams"])
for c in cam_names:
    (img_root / c.lower()).mkdir(parents=True, exist_ok=True)
for r in inf:
    i = r["frame"]
    src_root = "outputs/surround_train" if i < 200 else sys.argv[1].replace("outputs/", "outputs/")
    # src 文件名:旧段 000000..000199 / 新段 000000..000399
    for c in cam_names:
        if i < 200:
            src = Path("outputs/surround_train") / c.lower() / f"{r['token']}.png"
        else:
            src = Path(sys.argv[1]) / c.lower() / f"{i - 200:06d}.png"
        dst = img_root / c.lower() / f"{i:06d}.png"
        shutil.copy2(src, dst)
        r["cams"][c]["data_path"] = f"{c.lower()}/{i:06d}.png"
out.write_text(json.dumps(inf, ensure_ascii=False, indent=1))
miss = [
    (r["frame"], c["data_path"])
    for r in inf
    for c in r["cams"].values()
    if not (img_root / c["data_path"]).is_file()
]
assert not miss, f"缺 {len(miss)} 图: {miss[:3]}"
print(f"OK: {len(inf)} 帧集中到 {img_root}, 图像全齐")
PYEOF

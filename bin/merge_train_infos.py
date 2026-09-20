"""B2 链专用:拼接旧新两组环视训练数据 → 600 帧 infos(新数据帧号连续重排)。

目标:扩数据长尾(200 现有 + 400 新采集→600 帧,仅 Town10HD_Opt)。两侧都已是
assemble_maptr.py 的同构产物,这里做纯组装:旧 0–199 + 新 0–399,新侧帧号 +200
平移(避免 token 冲突),合并段落连续性由 ego 位姿保证(段落内 z/yaw 自然连续)。

用法:
  python bin/merge_train_infos.py \
      --old outputs/surround_train/map_infos.json \
      --new outputs/surround_p2/map_infos.json \
      --new-fbase 200 --out outputs/maptr_600/map_infos.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autodrivedata.paths import project_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="旧采集 infos")
    ap.add_argument("--old-n", type=int, default=-1, help="旧数据保留前 N 帧训练段(默认全量)")
    ap.add_argument("--new", required=True, help="新采集 infos(重排到新帧号)")
    ap.add_argument(
        "--new-img-root", default=None, help="新数据图像根目录(默认 = --new 所在目录,需含 cam_* 子目录)"
    )
    ap.add_argument("--new-fbase", type=int, default=200, help="新数据起始帧号")
    ap.add_argument("--out", required=True, help="合并后 infos 输出(锚定项目根)")
    args = ap.parse_args()
    args.out = str(project_path(args.out))

    old = json.loads(Path(args.old).read_text(encoding="utf-8"))
    if args.old_n > 0:
        old = old[: args.old_n]
    new = json.loads(Path(args.new).read_text(encoding="utf-8"))

    fbase = args.new_fbase
    for rec in new:
        i = rec["frame"] + fbase
        rec["frame"] = i
        rec["token"] = f"{i:06d}"
        # data_path **不随帧号重写**:图像文件名是采集时的原始帧号
        # (0..N-1),训练 root 指向新采集目录即可。会话级 meta(帧号)平移,
        # 文件级引用(路径)保持原样。合并进 600 帧 infos 后,
        # 训练时 root = 新采集目录(surround_p3)即正确。
    # (旧版曾把 data_path 一起平移,导致文件与文件名脱节——已修)

    infos = old + new
    dup = [i for i in range(len(infos) - 1) if infos[i]["frame"] >= infos[i + 1]["frame"]]
    if dup:
        raise SystemExit(f"合并后帧号非单调@掉 {dup}")
    if args.new_img_root is not None:
        img_root = Path(args.new_img_root)
        missing = [
            (r["frame"], c["data_path"])
            for r in new
            for c in r["cams"].values()
            if not (img_root / c["data_path"]).is_file()
        ]
        if missing:
            raise SystemExit(
                f"新数据图像缺失 {len(missing)} 条(如 {missing[:2]})——先跑 collect_surround 再合并"
            )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=1)
    n_ann = {k: len(v) for k, v in infos[0]["annotation"].items()}
    where_img = str(project_path(args.new_img_root)) if args.new_img_root is not None else "<--new 所在目录>"
    print(
        f"{args.out}:{len(infos)} 帧(旧 {len(old)} 保留 0–{len(old) - 1} / 新 {len(new)} 重排到 "
        f"{fbase}–{fbase + len(new) - 1};首帧 annotation {n_ann};新图像根 = {where_img})"
    )


if __name__ == "__main__":
    main()

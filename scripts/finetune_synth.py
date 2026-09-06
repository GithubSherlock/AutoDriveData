"""合成 KITTI 数据 → pointpillars_kitti 微调(AutoLabel train3d 五函数复用)。

用法(autolabel env,CARLA 服务器已停以腾显存):
  KITTI_OBJECT_ROOT=outputs/kitti_ft python3 scripts/finetune_synth.py --dry-run  # 数据准备+config
  KITTI_OBJECT_ROOT=outputs/kitti_ft python3 scripts/finetune_synth.py           # + 训练(GPU)

与 AutoLabel train3d 的差异:ImageSets 按本数据集实际帧号写(0..n_train-1 / 训练余下为 val),
而非官方 7481 帧的固定划分;其余步骤(create_data/subsample/config/train)原样复用。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from auto3dlabel.configs.kitti import DEFAULT_KITTI_ROOT, MMDET3D_CONFIG_DIR, WEIGHTS_DIR
from auto3dlabel.configs.model_catalog import DETECTOR3D_NAMES
from auto3dlabel.tools import train3d


def _run_create_data(kitti_root: Path) -> None:
    """create_data 自修复版:create_data.py import `tools.dataset_converters` →
    PYTHONPATH 必须含 .mim(AutoLabel 原版指 .mim/tools,首次实际执行即炸——本脚本修正)。"""
    required = [
        "kitti_infos_train.pkl",
        "kitti_infos_val.pkl",
        "kitti_infos_test.pkl",
        "kitti_infos_trainval.pkl",
        "kitti_dbinfos_train.pkl",
    ]
    velodyne = kitti_root / "training" / "velodyne_reduced"
    if all((kitti_root / f).is_file() for f in required) and any(velodyne.iterdir()):
        print("② create_data 产物齐全,幂等跳过")
        return
    tools_dir = train3d.mim_tools_dir()
    cmd = [
        sys.executable,
        "create_data.py",
        "kitti",
        "--root-path",
        str(kitti_root),
        "--out-dir",
        str(kitti_root),
        "--extra-tag",
        "kitti",
    ]
    env = {**os.environ, "PYTHONPATH": str(tools_dir.parent)}
    subprocess.run(cmd, cwd=tools_dir, env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.0003)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(DEFAULT_KITTI_ROOT)
    if not (root / "training" / "image_2").is_dir():
        raise SystemExit(f"KITTI root 无 training/image_2: {root}(KITTI_OBJECT_ROOT 指向合成数据 root)")

    # ① ImageSets:按实际帧号(train 0..n-1 / val n..n+m-1 / test 同 val)
    sets_dir = root / "ImageSets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    n_total = args.n_train + args.n_val
    (sets_dir / "train.txt").write_text(
        "".join(f"{i:06d}\n" for i in range(args.n_train)), encoding="utf-8"
    )
    (sets_dir / "val.txt").write_text(
        "".join(f"{i:06d}\n" for i in range(args.n_train, n_total)), encoding="utf-8"
    )
    (sets_dir / "test.txt").write_text(
        "".join(f"{i:06d}\n" for i in range(args.n_train, n_total)), encoding="utf-8"
    )
    print(f"① ImageSets: train 000000-{args.n_train-1:06d} / val {args.n_train:06d}-{n_total-1:06d}")

    # ② create_data(幂等;首跑几分钟)
    _run_create_data(root)

    # ③ subsample infos
    out = train3d.FINETUNE_DIR
    out.mkdir(parents=True, exist_ok=True)
    train_infos = out / f"kitti_infos_train_{args.n_train}.pkl"
    val_infos = out / f"kitti_infos_val_{args.n_val}.pkl"
    n1 = train3d.subsample_infos(root / "kitti_infos_train.pkl", train_infos, args.n_train)
    n2 = train3d.subsample_infos(root / "kitti_infos_val.pkl", val_infos, args.n_val)
    print(f"③ subsample: train {n1} / val {n2}")

    # ④ config 改造
    entry = DETECTOR3D_NAMES["pointpillars_kitti"]
    official = MMDET3D_CONFIG_DIR / entry["config"]
    checkpoint = WEIGHTS_DIR / entry["weights_dir"] / entry["checkpoint"]
    config_path = out / "finetune_config.py"
    train3d.build_finetune_config(
        official_config=official,
        kitti_root=root,
        train_infos=train_infos,
        val_infos=val_infos,
        checkpoint=checkpoint,
        out_path=config_path,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
    )
    print(f"④ config: {config_path}")

    if args.dry_run:
        print("dry-run 完成(未训练)")
        return
    # ⑤ 训练
    train3d.train(config_path, out)
    print(f"⑤ 训练完成,work_dir: {out}")


if __name__ == "__main__":
    main()

"""assemble_traj_pt.py 产出的 plain dict → HiVT TemporalData 转换器(在 hivt env 跑)。

背景:assemble_traj_pt.py 用 `torch.save(scene, ...)` 保存 plain dict,PyG DataLoader
对 dict 回退 default_collate(变长 lane_actor_index 崩);HiVT 官方 process 保存的是
TemporalData(Data 子类,__inc__ 触发图感知 collate)。本脚本把既有 processed/*.pt
原地转为 TemporalData,补齐官方字段:

- edge_index       [2, N(N-1)] 全排列(global/local encoder 的 subgraph 源)
- rotate_angles    [N] 各 agent 在 ego 系下的朝向(官方:历史最后两步 heading atan2;
                   静止 agent = 0)。assemble 原存全 0 是"全部按 ego 朝向"的简化,
                   此处按官方口径补算(forward 会把 y 也转到 agent 系,指标口径一致)
- agent_index      = av_index = 0(ego 是 agents[0];PyG 1.7.2 对 int 属性按 num_nodes
                   增量,ego 在 batch 中的位置 = 0,4,8,... 正确)
- city             = 图名(仅存无消费)
- origin           = [0,0](assemble 已中心化;eval 链不消费 origin/theta)
- theta            = 0

文件名不变(raw/*.csv 与 processed/*.pt 一一对应关系保持)。

用法(hivt env,无 GPU):
  CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/envs/hivt/bin/python \
      autodrivedata/traj/convert_hivt_pt.py outputs/hivt_carla/train outputs/hivt_carla/val
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch

# 本脚本以**文件路径**跑(`<hivt>/bin/python autodrivedata/traj/convert_hivt_pt.py ...`),
# 此时 `sys.path[0]` = 脚本所在目录(autodrivedata/traj),**项目根不在路径上** ⇒
# `import autodrivedata` 抛 ModuleNotFoundError。故先把项目根插进去 —— 与下面插 HiVT
# 路径同一手法,两条都是"这个脚本要在别的 env 里按路径跑"的代价。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, "/root/autodl-tmp/Documents/Projects/AutoDriveData/hdMapGitHub/HiVT")
from utils import TemporalData  # noqa: E402

from autodrivedata.utils import runlog


def convert(pt_path: Path, map_name: str) -> None:
    d = torch.load(pt_path, map_location="cpu")
    if isinstance(d, TemporalData):
        return  # 已转换过
    n = int(d["num_nodes"])

    # edge_index:全排列(官方 edge_index = [2, N*(N-1)])
    edge_index = torch.tensor(list(itertools.permutations(range(n), 2)), dtype=torch.long).t().contiguous()

    # rotate_angles:各 agent 在 ego 系下的朝向(静止 → 0)
    rotate_angles = torch.zeros(n, dtype=torch.float)
    for i in range(n):
        h = d["positions"][i, 19] - d["positions"][i, 18]
        if h.norm() > 1e-6:
            rotate_angles[i] = torch.atan2(h[1], h[0])

    data = TemporalData(
        x=d["x"][:, :20],  # 官方 x = 历史 20 步相对位移
        positions=d["positions"],
        edge_index=edge_index,
        y=d["y"],
        num_nodes=n,
        padding_mask=d["padding_mask"],
        bos_mask=d["bos_mask"],
        rotate_angles=rotate_angles,
        lane_vectors=d["lane_vectors"],
        is_intersections=d["is_intersections"],
        turn_directions=d["turn_directions"],
        traffic_controls=d["traffic_controls"],
        lane_actor_index=d["lane_actor_index"],
        lane_actor_vectors=d["lane_actor_vectors"],
        seq_id=d["seq_id"],
        av_index=0,
        agent_index=0,
        city=map_name,
        origin=torch.zeros(1, 2, dtype=torch.float),
        theta=0.0,
    )
    torch.save(data, pt_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="dataset root(train/val),每 root 转 processed/*.pt")
    ap.add_argument("--no-runlog", action="store_true", help="不落 logs/ 三件套(默认每次运行都落)")
    args = ap.parse_args()

    with runlog.run("autodrivedata.traj.convert_hivt_pt") as rl:
        n_total = 0
        for root in args.roots:
            rl.input(root, "dataset-root")
            root = Path(root)
            processed = root / "processed"
            if not processed.exists():
                print(f"[skip] {root} 无 processed/")
                continue
            pts = sorted(processed.glob("*.pt"))
            n_done = 0
            for p in pts:
                convert(p, map_name=root.name)
                n_done += 1
            print(f"[done] {root}: {n_done}/{len(pts)} 场景已转 TemporalData")
            n_total += n_done
            # 转换是**原地覆盖** —— 记 n_files 能看出这次到底动了几个文件
            rl.artifact_dir(processed, f"temporal-data:{root.name}")
        rl.highlight("n_scenes_converted", n_total)


if __name__ == "__main__":
    main()

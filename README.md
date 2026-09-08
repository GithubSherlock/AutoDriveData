# AutoDriveData

CARLA 0.9.16 → [AutoLabel](https://github.com/GithubSherlock/AutoLabel) 自动驾驶数据输出流水线:自定义地图/场景采集车检测动态/静态目标与道路特征,构建 0→1 数据输出。raw 输出供 AutoLabel 验证场 + 长尾数据源。

## 核心成果

**P1 Corner Case 场景矩阵**(帧级配对 A/B,统一 11 点插值 AP 口径):

| corner case | 相机 2D Δ AP | 掉点型 | LiDAR 3D Δ AP |
|---|---|---|---|
| 逆光(sunset_glare) | -0.014 | 轻掉点(AE 补偿) | +0.003 噪声 |
| 雨夜(rain_night) | **-0.153** | 漏检型(检出 0.72→0.48) | +0.017 噪声 |
| 浓雾(dense_fog) | -0.013 | FP 型(检出 0.72→0.85) | 0.000 |

结论:相机在三型真实驾驶长尾上均有可量化掉点;**LiDAR 兜底不受天气/光照**(平台边界:雨/雾对合成 LiDAR 无物理回波,退化只能人工注入)。

**P2 静态 GT**:地图查询 API(landmark 信号 + lane_marking 车道线)→ `training/static_gt/{fid}.json` + overlay 目检图,与天气/光照解耦。

**地图池**:17 图(Town01-10 + AdditionalMaps Town11/12/13/15,零构建)。

## 快速开始

```bash
# 1. CARLA 服务器(headless,GPU 修复栈;专用用户 carla)
bash scripts/carla_server.sh

# 2. 场景采集(KITTI root:image_2 + label_2 GT + velodyne + calib)
python scripts/collect_drive.py --scene rain_night --frames 70
python scripts/collect_ab_route.py --scene sunset_glare --frames 70   # P1 A/B 专用

# 3. 静态 GT(landmark + 车道线,含 overlay 目检图)
python scripts/collect_static_gt.py --frames 40

# 4. 2D A/B 评估
python scripts/eval_2d_ab.py --root-a outputs/kitti_ab_day_clear --root-b outputs/kitti_ab_sunset_glare

# 5. 3D LiDAR 检测(AutoLabel autolabel env;cwd 必须在 AutoLabel 根)
cd /root/autodl-tmp/Documents/Projects/AutoLabel && KITTI_OBJECT_ROOT=<abs kitti root> \
  /root/miniconda3/envs/autolabel/bin/auto3dlabel run 000000-000069 "检测汽车" \
  --det-model pointpillars_kitti --batch --no-viz --out-dir <abs out>
python scripts/eval_kitti.py --root outputs/kitti_ab_x --pred outputs/kitti3d_ab_x

# 6. 测试
python -m pytest tests/ -q   # base env,131 passed
```

## 环境(双环境,勿新建)

| 环境 | Python | 用途 |
|---|---|---|
| **base** | 3.10.8 | pycarla + ultralytics;采集、2D 评估、3D 比对、全部单测 |
| **autolabel** `/root/miniconda3/envs/autolabel` | 3.11.15 | mmdet3d;3D 检测、oracle 对比 |

纪律:autodrivedata 包**不 import carla**(纯值,两 env 可单测);依赖单向 AutoDriveData → AutoLabel,禁止反向。

## 项目结构

- `autodrivedata/` — 纯值库(geometry/calib/gt/static_gt/semantic/export/compare/scenarios),不 import carla
- `scripts/` — carla 采集器(collect_drive/collect_ab_route/collect_static_gt/collect_nus)+ 评估(eval_2d_ab/eval_kitti)+ `carla_server.sh`
- `tests/` — 单测(base env,131 passed)
- `outputs/` — 采集产物(不进 git)

## 文档

- [Plan.md](Plan.md) — **单一事实源**:方案定案/执行记录/待办,改决策先读再改
- [docs/milestone.md](docs/milestone.md) — 里程碑时间线与验收结论速览
- [docs/testLog.md](docs/testLog.md) — 测试与踩坑日志(现象 → 修复 → 回归保护)
- [CLAUDE.md](CLAUDE.md) — 项目会话说明(A/B 实验纪律与红线)

## 平台边界(如实记录)

- CARLA 0.9.16 无镜头光学(flare/动态范围)、无雪;雨/雾只影响渲染不影响合成 LiDAR → 传感器退化只能人工注入
- 定制街道需 CARLA 源码构建(~170G 磁盘/Epic 授权),已降级挂起;零构建扩地图池替代
- 新图采集约束:Town11/12 禁采集(spawn camera segfault)、Town13 TM 车流降级、可用 Town13/15

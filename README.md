# AutoDriveData

CARLA 0.9.16 仿真数据输出流水线:在自定义地图/场景中采集**车检测动态/静态目标与道路特征**,产出「raw 传感器数据 + 结构化 GT」双输出,供 [AutoLabel](https://github.com/GithubSherlock/AutoLabel) 做验证场与长尾数据源。

项目同时是一条**定量验证链**:corner case 退化能不能量化、退化归因到哪一层、感知模型在长尾上掉多少点,都在本仓库内有可复现的 A/B 口径与数字,而不是"看起来更难了"。

## 能力总览

### P1 · Corner Case 场景矩阵(帧级配对 A/B)

同一 ego 锚定起点、同一静置车布局,**只变天气/光照**逐帧配对采集,统一 11 点插值 AP 口径评估:

| corner case | 相机 2D ΔAP | 掉点型 | LiDAR 3D ΔAP |
|---|---|---|---|
| 逆光 `sunset_glare` | −0.014 | 轻掉点(AE 补偿) | +0.003 噪声 |
| 雨夜 `rain_night` | **−0.153** | 漏检型(检出 0.72→0.48) | +0.017 噪声 |
| 浓雾 `dense_fog` | −0.013 | FP 型(检出 0.72→0.85) | 0.000 |

**失效归因**(逐帧匹配 + 每 GT 上下文,与 AP 共用同一 IoU 口径):

- **尺度主导** —— 框高 <32px 检出率一律 0.15–0.47,≥32px 一律 0.78–1.00,断崖 ≈21–24px(30–40m);漏检框内亮度与命中几乎相同 ⇒ 漏的是"小",不是"暗"
- **天气只是把断崖前移** —— 雨夜零检出从 40–50m 提前到 30–40m,雾反而最晚
- **平台边界** —— CARLA 无运动模糊(4/8/12 m/s 梯度能量 35.6/35.2/34.8,检出率无趋势)、雨雾对合成 LiDAR 无物理回波 ⇒ 这两类退化只能人工注入,不能靠改场景参数复现

### P2 · 静态 GT 与灯色动态 GT

- **静态 GT** —— 信号/标志是 landmark、车道线是 lane_marking 实体,semantic LiDAR 打不到 ⇒ 走**地图查询 API**,落 `training/static_gt/{fid}.json` + overlay 目检图,与天气/光照解耦
- **灯色动态 GT** —— 灯态是独立时序语义层(工业口径:Off/Unknown **不猜**),落 `training/traffic_light/{fid}.json`(逐帧状态 + 管制车道/停车线 + 相位计划);`--cycle 6,2,6` 给出**确定性变灯序列**。真值取自 actor API 而非视觉——镜片 0.2m 在 f=621 下 30m 处仅约 4px,视觉回归不成立

### MapTR 矢量管道(自实现)

OpenDRIVE 解析 → 矢量 GT 提取 → 环视采集/组装/投影验收 → **参考自实现**(GKT + 分层 query,单帧过拟合锚定正确性)→ chamfer AP 评估 → 逐帧契约落盘。

- 留出集 chamfer AP @0.2 = **0.0674**(@0.3 0.0904 / @0.4 0.1280),训练集对照 0.2603 ⇒ **泛化间隙 3.9×**,下轮收益靠扩数据而非继续长训
- 该口径是 3 阈值 precision 均值、**无 recall 项**,跨权重比较必须固定 `--score-thr`,单报一个 mAP 数字而不写阈值 = 无效结论
- 逐帧契约 `mapvec_pred/1`(`outputs/surround_pred/{token}.json`)schema/帧归属/坐标系/窗口/阈值/溯源齐全,GT 同文件携带,供 AutoLabel 消费(**消费方未接**)

### 8 路实时 studio + 在线 SLAM

`bin/live_studio.py` 单端口多槽 MJPEG:6 相机 + BEV + 第三方视角 + 拼图,浏览器直接看采集链所见画面(真 UE 渲染 + GT 框/灯色 overlay,不依赖 carlaviz/RViz2)。

- `--keyboard` WASD 操控(折进 tick 循环)、`--slam` 在线语义 LiDAR SLAM、`--maptr-ckpt` 实时 MapTR overlay、`--video` 落八视角视频段
- **在线 SLAM 验收**:400 帧在线 vs 离线链式位姿逐元素差 **0.0**,ATE **0.18768 m**;滞后口径 = 已 tick 帧号 − 已处理帧号
- 关键机制:有界丢旧队列**有界的是深度不是帧间隙**,而 ICP 成本随间隙超线性 ⇒ 必须按**帧号差止损**(`--slam-max-gap`),否则"丢帧→间隙更大→更慢→更多丢帧"是无界正反馈。另:纯 Python 主循环持 GIL,worker 线程 ICP 效率被压到 0.04–0.24 ⇒ 默认同步执行

### 地图池

17 图零构建扩展(Town01–10 + 开源 AdditionalMaps 的 Town11/12/13/15),默认 Town10HD_Opt;每张新图入池前先过 spawn/车流可用性验证(见「边界」)。

### 教程能力线

单双目测距 / 多雷达标定 / 累积建图 / 地面提取 / 点云聚类 / 3DGS / HiVT 轨迹预测(CARLA→Argoverse 口径,minADE 6.08)/ 激光 SLAM(前端点面 ICP + ScanContext 回环 PGO,含 evo·KITTI 口径 ATE/RPE 评估)。详见 [docs/milestone2.md](docs/milestone2.md)。

## 快速开始

> 进入项目目录自动激活 **autodrivedata** conda env(direnv + `.envrc`,首次需 `direnv allow`)。

```bash
# 1. CARLA 服务器(headless,GPU 修复栈 + Vulkan 兼容层自愈;专用用户 carla)
bash bin/carla_server.sh

# 2. 场景采集(KITTI root:image_2 + label_2 GT + velodyne + calib)
python bin/collect_drive.py --scene rain_night --frames 70
python bin/collect_ab_route.py --scene sunset_glare --frames 70   # P1 A/B 专用:锚定起点 + 静置车布局

# 3. 静态 GT / 灯色动态 GT
python bin/collect_static_gt.py --frames 40
python bin/collect_tl_states.py --frames 90 --speed 8 --cycle 6,2,6   # 受控切灯 = 确定性变灯序列

# 4. 8 路实时 studio(浏览器需本地 ssh -L 8080:127.0.0.1:8080 <host>)
python bin/live_studio.py --speed 8 --npcs
python bin/live_studio.py --maptr-ckpt outputs/maptr_ep512.pt --slam --speed 8

# 5. 评估:2D A/B + 失效归因
python bin/eval_2d_ab.py --root-a outputs/kitti_ab_day_clear --root-b outputs/kitti_ab_sunset_glare
python bin/eval_attr.py --run day8=outputs/kitti_sweep_day_clear_8:8.0 \
                        --run rain=outputs/kitti_ab_rain_night:8.0 --json outputs/attr.json

# 6. MapTR:逐帧推理 + chamfer AP + 逐帧契约落盘
PYTHONPATH=$PWD python bin/eval_maptr.py --infos outputs/surround_train/map_infos.json \
  --root outputs/surround_train --ckpt outputs/maptr_ep512.pt --start 200 --out-frames outputs/surround_pred

# 7. 测试
python -m pytest tests/ -q          # 510 passed / 3 skipped
```

## 环境

| 环境 | Python | 用途 |
|---|---|---|
| **autodrivedata**(本项目) | 3.11.16 | pycarla + ultralytics;全部采集器、2D 评估、全部单测 |
| **autolabel** | 3.11.15 | mmdet3d;3D 检测(`auto3dlabel run`)、oracle 对比 |
| **hivt** | 3.8.20 | HiVT 复现栈(torch1.8 / pl1.5 / pyg1.7),CPU 推理;**只能用绝对路径调**(未注册进 `envs_dirs`) |
| **base** | 3.10.8 | conda 底座 + direnv,不承担项目职责 |

**硬纪律**:`autodrivedata/` 包**绝不 import carla**(纯值,任何 env 可单测);依赖单向 AutoDriveData → AutoLabel,**禁止反向**。

## 项目结构

```txt
autodrivedata/    纯值库:几何/标定/GT/场景目录/地图矢量/归因/SLAM 评估(不 import carla)
maptr_impl/       MapTR 参考自实现(ResNet50+FPN + GKT + 分层 query head)
bin/              可执行入口:采集器 / 组装转换 / 训练 / 评估 / 可视化 / 探针
tests/            单测 + oracle 对比(510 passed)
docs/             文件级索引 / 里程碑 / 测试日志 / 16 篇 CARLA 教程
outputs/          【未入库】全部产物的唯一落点(权重/数据集/可视化)
```

> 📁 **文件级索引见 [docs/fileTree.md](docs/fileTree.md)** —— 每个文件/脚本/产物的职责与依赖方向,加/改文件后需回来补一行。

## 文档

| 文档 | 内容 |
|---|---|
| [Plan.md](Plan.md) | **方案定案 + 历史执行记录**(契约/架构/里程碑归档)。已冻结,只增不改 |
| [Plan2.md](Plan2.md) | **新计划的制定地**(2026-09-19 起):执行项/进度/遗留缺口。改决策先读它 |
| [docs/fileTree.md](docs/fileTree.md) | 文件级索引与维护约定 |
| [docs/milestone.md](docs/milestone.md) · [milestone2.md](docs/milestone2.md) | 版本里程碑 / 教程能力线里程碑 |
| [docs/testLog.md](docs/testLog.md) | 测试与踩坑日志(现象 → 修复 → 回归保护) |
| [CLAUDE.md](CLAUDE.md) | AI 协作入口:A/B 实验纪律与已踩坑红线 |

## 边界(已实测的平台限制)

这些不是待办,是**已验证的能力边界**——决定哪些结论不能靠改场景参数得到:

- **合成 LiDAR 无天气/光照物理回波** ⇒ 雨雾对 LiDAR 的退化只能人工注入,不能靠 weather 参数复现
- **CARLA 无运动模糊** ⇒ 速度不改变图像质量,高速退化同理只能注入
- **灯态不做视觉回归** ⇒ 灯色 GT 取自 actor API(逻辑层),真值口径是"时序语义层"
- **MapTR 权重与 rig 必须配对** ⇒ `legacy` / `official` 两代环视挂点并存,用错 rig 喂权重会显著掉点(`--rig auto` 按权重名选)
- **Town11/12 禁采集**(spawn camera segfault)、Town13 TM 车流降级 0 NPC

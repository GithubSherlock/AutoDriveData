# AutoDriveData — CARLA 仿真数据生成 Pipeline(方案定案 v0.7)

> 定案日期:2026-09-06(v0.1 架构)→ 09-06(v0.2 M0+KITTI 契约)→ 09-07(v0.3 M1 编排)→ 09-07/08(v0.4-0.6 M1a/M1b/M2/M3 执行记录)。
> 状态:**M0-M3 ✅ 工具链闭环;微调专项挂起;最终目标剩余三缺口拆 P1→P2→M4 三工作包推进中(§5.5,2026-09-08 用户拍板)**。
>
> **⚠️ 计划制定已迁移(2026-09-19 用户拍板)**:自即日起,**新项目计划一律在 [Plan2.md](Plan2.md) 中制定**,
> 本文件转为**方案定案 + 历史执行记录**(§1~§4 契约/架构 + §5 里程碑归档 + §6 骨架 + §7 待办快照)。
> 具体:① **CARLA 采集/场景/教程能力线**的后续规划(含教程 04 IPM 缺口、SLAM 阶段 2、多图扩数据等)
> 全在 Plan2.md §3/§8;② Plan2.md 的里程碑记录落 docs/milestone2.md;③ 本文件 §5.x 记录**不再新增**,
> 只在既有条目上补"已迁移"指针;④ 红线纪律、接口契约、环境表仍以本文件为准(Plan2.md §6 复述)。
>
> **⚠️ 路径口径声明(2026-09-26)**:本文档是**冻结的历史记录**,其正文里的文件路径是
> **2026-09-26 目录重构之前**的口径,故 `bin/` 与 `tests/` 之类的路径**现在已不存在**。
> 映射:`bin/x.py` → `autodrivedata/<能力>/x.py`、`tests/test_x.py` → `autodrivedata/tests/<能力>/test_x.py`、
> `maptr_impl/` → `autodrivedata/map/maptr/`、包根模块 → 各能力目录(`calib.py`→`calib/core.py` 等)。
> **不修改正文**(冻结纪律),完整映射与理由见 [docs/refactor-2026-09.md](docs/refactor-2026-09.md) §2。
> 查**当前**路径请用 [docs/fileTree.md](docs/fileTree.md)。

## 1. 定位

CARLA 仿真数据源 + 真值旁路,供 **AutoLabel** 消费:

- **AutoLabel 侧现状**(`auto3dlabel` v0.4):mmdet3d PointPillars 三引擎统一工厂,已通 KITTI + nuScenes 端到端闭环(复核队列 + Web HITL 编辑),输入契约明确。
- **AutoDriveData 定位(已确认)**:raw + GT **双输出**;用途 = ① AutoLabel 在未知合成域的验证场与回归基准(GT vs 伪标签自动比对,分歧帧才人工复核);② 长尾场景补充源(合成数据后续并入 mmdet3d 训练)。

```
AutoDriveData(CARLA)
  场景层   现有 Town10HD(+RoadRunner 定制,远期)
  采集层   同步相机 RGB / LiDAR / semantic LiDAR(GT 校验)
  GT 层    动态 actor box + 静态(信号灯/标志状态)+ 道路特征(车道线/限速)
  桥接层   KITTI 目录 / nuScenes 语义 → 落盘 raw + GT
        │
        ▼
AutoLabel 预测 → 伪标签
  比对层  F4/代码级:伪标签 vs GT → 分歧帧入复核队列(review_flag 机制已有)
  人工复核 Web(:8765)少量分歧帧 → 真值
  回归基准 每版引擎变更重跑合成域对比 GT(量化 2D/3D AP 与复核率)
```

## 2. 可行性结论

技术高度可行:各模块均有成熟方案;CARLA 官方 sensor suite 带完美同步与真值 API。主要工作量在**格式桥接层**与**域差距校准**,两处均已探明落地路径(§3、§5)。

风险排序(2026-09-07 更新,磁盘阻塞已解除):
1. 非 actor 静态目标(CARLA 网格类物体)无 API 真值 → semantic LiDAR/segmentation 后处理,或 RoadRunner 建为资产(**延后到 M4**)
2. 合成域掉点:3D 点云模型通常可控;2D 检测在渲染图上掉点更明显 → M1a 验收 + M2 用 GT 实测量化,不猜
3. RoadRunner 收费 + 建模规范学习成本(延后,先用免费 trial 验证链路)

## 3. 与 AutoLabel 的接口契约(决定成败,先固化)

全部照抄 `auto3dlabel` 已有约定,禁止另立格式。

### 3.1 KITTI 域(已钉死,2026-09-06 读码确认)

**数据根目录可配置**:`KITTI_OBJECT_ROOT` 环境变量已实现(`auto3dlabel/configs/kitti.py`),AutoDriveData 输出目录直接作为 root,**AutoLabel 零改动**。

布局(照 `schema/box3d.py` `KittiFrame` + `data/kitti.py`):
- `{root}/training/image_2/{id}.png` — id 为 **6 位零填充数字**(`normalize_frame_id`,`frame_ids_from_dir` 只收 `*.png` 数字 stem);顺延 `frame_ids_by_range(start,end)` 语义
- `{root}/training/velodyne/{id}.bin` — float32 (N,4): x,y,z,intensity(读时剔 NaN/Inf)
- `{root}/training/calib/{id}.txt` — `KittiCalib.from_file`:P2/R0_rect/Tr_velo_to_cam 投影链(CARLA extrinsics/intrinsics 在此生成)
- `{root}/training/label_2/{id}.txt` — 15 字段;**GT 消费行 `[h,w,l,x,y,z,ry]`,y = 物体底部中心**(地面),内部归一 cy = y − h/2;跳过 `DontCare/Misc`;评测只算 **Car/Pedestrian/Cyclist**(Truck/Tram 导出不评)

GT 类别直落 KITTI 名(`COCO_TO_KITTI` 已存:car→Car/person→Pedestrian/bicycle→Cyclist/truck→Truck/tram→Tram)。

### 3.2 nuScenes 域(auto3dlabel v0.4 P2 约定,单一事实源)

- 复核队列键 `dataset:"nuscenes"` 分派;scene/sample token 语义照 `export/nuscenes_queue.py`
- box 语义照 devkit:**全局四元数 + velocity + track_id**;nusbox ↔ box3d_dict 双向转换在 `data/nuscenes.py` 已有 → AutoDriveData 生成器复用它作**校验基准**
- 坐标系约定(P2 唯一落点 `tools/geometry.py`):`cam_like = M @ (p_global − t_ego)`,`M = [[0,−1,0],[0,0,−1],[1,0,0]]`;`yaw_bev = −yaw_g`;`(h,w,l) = (size[2],size[0],size[1])`;`rotation_y = −yaw_g − π/2`
- ego 轨迹/相机外参 ↔ CARLA `get_transform()` 换算以**实测标定**为准(写单测,附断言)

### 3.3 伪标签 vs GT 比对(M2 开发,双输出目的所在)

- 输入:AutoLabel `*_review.json` 三档分流产物 + AutoDriveData GT(同 frame key)
- 规则:类别 + 3D IoU(复用 `tools/geometry.py`)+ fit_points 数量级 → 一致帧跳过复核(人工量预期 <5%),分歧帧注入 `review_flag`
- 输出 2:合成域回归基准(每 engine/config 的 AP + 复核率报表)

## 4. M0 环境(✅ 完成)

### 4.1 硬件匹配结论(已实测,2026-09-07 更新)

| 项 | 本机 | 要求 | 结论 |
|---|---|---|---|
| GPU | RTX 3080 Ti 12GB,driver 595.58 | ≥8GB VRAM | ✅(Low 画质实占 ~5G) |
| OS | Ubuntu 22.04 x86_64 | Linux | ✅ |
| Python | autodrivedata env 3.11(本项目)/ autolabel env 3.11 | API 支持 3.7–3.12 | ✅ 双 wheel |
| 内存/CPU | 440GB / 12 核 | 8GB+ | ✅ |
| 磁盘 | autodl-tmp **190G(余 76G)** | ~20G | ✅(已扩容) |

### 4.2 版本选型:**0.9.16**(官方 latest,2025-09-16)

- 弃 0.10.0(UE5 新线,周边生态滞后);0.9.16 = 0.9 线集大成 + 关键修复:instance segmentation 细粒度标注、**SensorData frame/timestamp/transform 与图像严格匹配**(GT 同步刚需)、Synchronized actor BoundingBox

### 4.3 安装与启动(已执行,速查)

```bash
# 安装位置 /root/autodl-tmp/CARLA_0.9.16/(19G);pycarla 装 autodrivedata env: pip install carla==0.9.16
# 启动(必须专用用户 carla,UE4 拒绝 root;headless 适配):
su - carla -c "cd /root/autodl-tmp/CARLA_0.9.16 && ./CarlaUE4.sh -RenderOffScreen -quality-level=Low"
```

### 4.4 环境共存策略(已定)

采集脚本独立进程跑在 **autodrivedata env**(3.11):只 import pycarla + numpy;绝不装进 `autolabel` env(mmdet3d CUDA13 编译环境脆弱)。落盘后再由 AutoLabel 侧读取(跨进程边界即解耦点)。

**2026-09-10 环境迁移定案**(用户拍板):项目 env 从 base(3.10)→ 新建 **autodrivedata**(3.11.16),pycarla/ultralytics 全量迁入,base 仅剩 conda 底座 + direnv;direnv + .envrc 进入目录自动激活,VSCode 解释器指向该 env;依赖钉死在 requirements.txt(版本对齐 base 已验证组合,cu130)。MapTR/MapQR(§5.11 C 阶段)需 py3.8 + torch1.9 + mmcv-full1.4 老栈,与 autodrivedata 不兼容 → 预留独立 maptr env(未建)。

### 4.5 M0 执行记录(2026-09-06 ✅)

- 磁盘扩容后下载 8.35GB(29 分,5.1MB/s)→ 解压归拢 `CARLA_0.9.16/`(坑① tar 平铺无顶层目录)
- 坑② UE4 拒绝 root → 专用用户 `carla`(home `/root/autodl-tmp/carla_home`);坑③ `/root` 700 → `chmod 711`
- 首启 ~8 分钟(shader 编译);ALSA 报错可忽略;版本握手 0.9.16 ✅
- 坑④ 传感器属性设 blueprint(spawn 前);坑⑤ **轴系红线**:ray_cast LiDAR 默认 360° 旋转,原始数据为传感器系(x 前/y 右/z 上)≠ KITTI 相机系(x 右/y 下/z 前)
- 验收:`outputs/smoke/` + `bin/smoke.py`(已参数化 --channels/--pps/--cam-offset/--pitch/--fov/--lidar-range)

## 5. 里程碑与编排

> ⚠️ **本节为历史归档,不再新增条目**(2026-09-19 起计划制定移至 [Plan2.md](Plan2.md);
> 后续里程碑记录见 Plan2.md §7 + docs/milestone2.md)。

| 阶段 | 内容 | 验收 |
|---|---|---|
| **M0 环境** ✅ | 安装 + headless 适配 + smoke | 单帧 raw 落盘 + 可视化 |
| **M1a KITTI 闭环** ✅ | 步骤 1–7(见 §5.1) | 数据层零改动读入 + pointpillars 出伪标签 |
| **M1b nuScenes 语义** ✅ | 步骤 8–10(见 §5.2) | nuscenes-queue 分派通 |
| **M2 闭环验证** ✅(带已知缺口) | autopilot 短途采集(数百帧)→ 比对层 → 复核队列 | 报表出;分歧率 <10% **未达**(见 §5.3) |
| **M3 场景参数化** ✅(工具链;天气欠账→P1) | Traffic Manager/微调排查/数据卫生;天气·光照·长尾指令集未做(归 P1) | 场景矩阵脚本归 P1 |
| **M4 定制街道** | 拆三包:P1 场景矩阵 → P2 静态 GT → M4 RoadRunner(§5.5) | 逐包验收 |

**2026-09-07 决策**(用户拍板):M1 拆 M1a→M1b 分步交付;M1a 验收口径 = 数据层 + 跑一次 pointpillars;项目现在建本地 git 仓库(Conventional Commits + feature/ 分支,每步骤一 commit)。

### 5.1 M1a — KITTI 静态一帧闭环(步骤 1–7)

| # | 步骤 | 产出/验收 |
|---|---|---|
| 1 | 项目骨架:git + pyproject + `autodrivedata/` 包(纯逻辑无 carla 依赖) | ✅ 骨架就绪,首次 commit |
| 2 | `geometry.py`:CARLA 系 → KITTI 相机系 + rotation_y/yaw_bev 换算 | 手工算例单测(方位角象限全覆盖) |
| 3 | `calib.py`:fov+分辨率 → P2;R0_rect=I;LiDAR→相机外参 → Tr_velo_to_cam | KITTI 格式 calib txt + 单测 |
| 4 | `gt.py`:actor bounding_box/transform → label_2 15 字段(类别映射、y=底心、视野过滤 + truncation) | 与 CARLA 真值手验一致 |
| 5 | `export/kitti.py`:image_2/velodyne/calib/label_2 落盘(6 位零填充) | 目录结构单测 |
| 6 | `bin/collect_kitti.py`:ego 静止 + 摆 NPC,同步模式采 N 帧 | 输出一个 KITTI root |
| 7 | **集成验收**:`KITTI_OBJECT_ROOT=... auto3dlabel run`(autolabel env) | 数据层读入全通 + pointpillars 伪标签落盘 |

### 5.1a M1a 执行记录(2026-09-07 ✅ 验收通过,含域差距首测)

**验收通过项**:
- 单测 72 passed(base)+ oracle 5 passed(autolabel env:KittiCalib 解析/投影链逐点一致、GT 往返一致)
- 数据层零改动读入:`KITTI_OBJECT_ROOT=outputs/kitti_scene` → `resolve_frame` + `load_calib/load_gt3d/load_velodyne` 全通
- PointPillars 推理跑通:伪标签 + review 队列落盘(`采纳 0 / 复核 3`)
- 几何链自检:点云投影全部落入 GT 2D 框(每目标 11~450 点)

**关键坑(全部已解决,代码带回归注释)**:
- ① tar 解压平铺 ② UE4 拒绝 root → 专用用户 carla ③ /root 700 → 711 ④ 传感器属性设 blueprint ⑤ **同步模式 spawn 后必须 tick**,否则 get_transform 返回恒等(实测)⑥ TaskStop 杀不死 UE4 子进程 → 端口占用崩新实例,须 pkill ⑦ auto3dlabel CLI 的 config 路径相对 cwd,须在 AutoLabel 根目录跑 ⑧ LiDAR 200k pps 车簇仅 13~117 点 → 1.3M pps(实测 63k 点/帧、360° 全覆盖)

**域差距首测(M2 比对层的先导数据)**:
- A/B 对照:同一检测器在真实 KITTI 000000 检出 GT 行人 IoU 0.60 ✅;在我们的合成帧上真车全漏、建筑立面误报(伪标签 3 框 IoU 全 0)
- 疑似成因(按嫌疑排序):① 强度通道退化(std 0.055 vs 真实 velodyne 大幅变化)② CARLA 默认零噪声/零 dropout(真实传感器有 ~10% 丢失 + 测距噪声)③ 点密度仍低 7 倍(63k vs 真实 130k+/帧)④ 合成表面过于光滑
- 后续动作归 M2:比对层建好后系统量化;候选数据侧修复(LiDAR noise_stddev/dropoff 属性、强度合成、更高 pps)在 M2 起验证

**M1a 设计要点**(已定):
- **静态场景不开车**——ego 停着 + 手动摆 NPC;autopilot 短途是 M2 内容,不提前
- **点云不做重投影**——LiDAR 挂装位姿即 KITTI velodyne 位姿,点云保持传感器系原样落盘,`Tr_velo_to_cam` = LiDAR→相机外参(投影链 `x_cam = R0_rect @ Tr @ x_velo` 天然成立)
- GT 只含动态 actor(静态目标 GT 延后 M2+);遮挡字段 M1a 填 0,truncation 按 8 角点投影计算

### 5.1b M1b 执行记录(2026-09-07 ✅ 验收通过)

- geometry 补 nuScenes 约定:CARLA_TO_NUS(y 翻号)、yaw↔quat(照抄 auto3dlabel 语义)、quat_to_matrix
- export/nuscenes.py:14 表 + map png(devkit 构造契约),多场景支持(scenes dict)
- **关键坑**:mini_val = {scene-0103, scene-0916}——generate_review_queue 遍历 val 名单,缺场景直接 KeyError → 生成器强制两场景
- oracle 3/3:devkit 构造、GT 读取、NusBox 往返一致;验收:nuscenes-queue → 3 复核队列(采纳 0/复核 5,域差距同 KITTI 侧)

### 5.2 M1b — nuScenes 语义(步骤 8–10)

| # | 步骤 | 产出/验收 |
|---|---|---|
| 8 | `geometry.py` 补 P2 约定(cam_like 矩阵、尺寸换序、全局四元数) | 单测(对照 §3.2 公式) |
| 9 | `export/nuscenes.py`:nusbox + scene/sample token 落盘 | 用 auto3dlabel `data/nuscenes.py` 双向转换做**往返 oracle 校验** |
| 10 | 验收:`auto3dlabel nuscenes-queue` 分派通 | 复核队列生成 |

### 5.3 M2 执行记录(2026-09-07 ✅ 工具链闭环,验收缺口如实记录)

**工具链交付**(全部落盘 + 单测):
- `collect_drive.py`:ego autopilot + TM 车流 + AI 行走行人,同步模式采 150 帧(验收用)
- `compare.py`:Sutherland-Hodgman 多边形 IoU + 贪心匹配 + 11 点 AP + 分歧帧/复核率报表(纯 numpy)
- `eval_kitti.py`:GT vs 伪标签 CLI(含 review JSON 置信度自配对);`semantic.py`:语义强度合成

**域差距修复实验(6 变体 × 5 帧,PointPillars KITTI)**:

| 变体 | Car TP/15 | 结论 |
|---|---|---|
| 几何强度基线/噪声/丢点/2.6M/5.2M pps | 全部 0 | 密度/噪声不是主因(实测 pps 线性无上限:5.2M→253k 点) |
| **语义 LiDAR + 反照率×入射角强度合成** | **11/15,AP 0.529** | ✅ **决定性修复**:车亮(0.85)/路暗(0.08)/标牌高反(0.9) |

**M2-4 全链路实测(150 帧动态)**:
- 采集:ego 行驶穿越街区,GT 233 Car(行人 0——随机 spawn 未入相机视野)
- 检测:采纳 76 / 复核 340;比对:Car **AP 0.294(TP 118/233)**,复核率 100%
- **验收缺口**:分歧率 <10% **未达**——根因 ① 伪标签 FP 高(车 154 FP/行人 144 FP,合成场景立面/杆件误报)② 动态场景更难(AP 0.53→0.29)③ 行人/骑行者 GT 缺失无法评估
- **通往 <10% 的路径(归 M3)**:AutoLabel `train3d` 在合成 KITTI 数据上微调 PointPillars(域内训练,预期 FP 大幅下降);行人用固定路径布置(勿随机 spawn);复核率与 AP 随微调重测

**已知限制(如实记录)**:
- CARLA headless 稳定性:客户端销毁传感器后服务器反复 segfault(exit 139)——重启可恢复,采集脚本 try/finally 清理已就位;待 M3 排查(疑似 Vulkan/离屏渲染 GC 路径)
- 语义 LiDAR 将行人/骑行者标为 Unlabeled(tag=0);walker 身体点稀疏(8m 处 ~25 点)
- dropoff_general_rate>0 时点数反常上升(105k vs 64k),机制未查明,已弃用

### 5.4 M3 执行记录(2026-09-07 ✅ 工具链交付,微调质量未达标——如实记录)

**交付**(全部可复现):
- `finetune_synth.py`:合成 KITTI → train3d 五步(ImageSets 自适应帧号、create_data PYTHONPATH 修正、testing/ 符号链接)
- 训练 ~5 分钟/轮(300 帧 × 4 epoch,batch 2,3080 Ti 0.07s/iter),loss 0.87→0.05 正常收敛
- 3 次微调尝试全部在驱动帧上塌缩(40 点 AP easy:官方 78.3 → 微调 9.5/5.5/0.0)——静态帧上微调模型反而更干净(3/3 全对、无 FP)

**根因排查(3 个毒化源,已修 2 个,1 个记录待查)**:
1. ✅ **行人 NAV 失败 → 原点聚集**:AI 控制器 go_to_location 偏离导航网格时把行人导向地图原点,528 个行人 GT 挤在原点(已修:只放有效 spawn 点 + 不导航)
2. ✅ **远距无点 GT**:相机 FOV 内 163m 的 GT 框远超 LiDAR 量程 70m,零支撑点的正样本毒化校准(已修:box_to_gt_line max_distance=65)
3. 📋 **疑似残留**:微调后模型在驱动帧分数/位置全面塌缩但静态帧正常——训练域(稀疏 1-2 GT/帧 + 50% 空帧)与推理域分布差 + 灾难性遗忘;需要更大数据量 + 训练中 val 监控 + 更多 epoch 才能定论

**M3-4 行人布置**:修复 + 验证(480 行人 GT 分布在 90 个唯一位置,不再聚集)
**M3-5 headless segfault**:结论=teardown 阶段偶发(数据从不丢,采集已完整落盘);纪律=每次采集前 `carla_server.sh start` 干净启动;已交付辅助脚本

**当前生产配置定论**:官方 pointpillars_kitti + 语义强度合成(静态 Car AP 0.53;驱动 40 点 easy 78.3/mod 54.1)+ max_distance=65 GT 过滤。微调留作后续专项。

### 5.5 M4 编排(2026-09-08 定案:三工作包 P1→P2→M4)

用户拍板(2026-09-08 全选)齐推最终目标三缺口:**Corner Case 场景矩阵 + 静态目标/道路特征 GT + RoadRunner 定制街道**。按工程依赖排定:

**P1 — Corner Case 场景矩阵**(Town10HD,零外部依赖 → 先行)
- P1-1 corner case 目录 + **可模拟性矩阵**(对每个场景如实标注 CARLA 保真度,防做假)
  - 光照:逆光(低角度正前阳光,可做)、黄昏、夜(负太阳高度角,可做)、隧道阴影(可做)
  - 天气:雨/湿地面、雾(CARLA 原生参数,可做);雪/冰(0.9.16 无,不可)
  - 车流:密集拥堵、鬼探头(walker 路径可控,可做);逆行/违章(需脚本驱动,部分)
  - 传感器:lidar 丢点注入(已有 --lidar-noise/dropoff 开关);真实镜头光学(flare/动态范围 **CARLA 无,不可**——逆光"眼瞎"只量化相机 AP 掉点 + LiDAR 兜底差值,不做视觉真实感)
- P1-2 场景参数档库(weather/光照 preset dict)+ 矩阵采集脚本(复用 collect_drive 同步/GT/semantic 基础设施)
- P1-3 **逆光 A/B 定量实验**(首个 corner case):晴天 vs 低角度正前阳光,各 N 帧 → PointPillars AP 对比 → 相机"瞎多少"量化 + 融合差值
- 验收:≥1 corner case 定量 A/B + 参数档可复用库

### 5.5a P1 执行记录(2026-09-09 ✅ P1 验收)

**P1-1**(37b5359)场景目录 + 单测;可模拟性矩阵写在 scenarios.py 模块 docstring(无 flare/镜头光学、无雪、LiDAR 雨损不可模拟——防把简化渲染当真实)。

**P1-2**(f0003da)collect_drive --scene + 8 场景档实测确认(day_clear 为生产基底,其余覆写校验防打错字)。

**P1-3 逆光 A/B**(collect_ab_route.py + eval_2d_ab.py,新增)——关键坑与最终数:

方法学(三次修正,每步都是纪律教训):
1. autopilot 路线不可控 → 弃;改固定 ego 定速直行 + 路肩 4 静置车(20/35/50/**62**m),只变 weather
2. **方位校准**:az=300 误采是顺光(太阳在车后,对比 +0.054 假"逆光更强");azscan 修正 az=90(=东=+x=车头正前,判据:日盘在 FOV 时全图过曝最低——AE 压得最狠)
3. **起点锚定**:校准脚本残留 ego 阻塞 spawn point 0 → spawn_ego fallback 到反向点(朝 -x),轨迹失配;现清场 + 强制锚定 pts[0](yaw=0) + 起点校验,重采即复现
4. **GT 边缘卡边**:第 4 台车原放 65.0m = GT max_distance 阈值,起步抖动致两侧 GT 216 vs 219 不可比;改 62m 留裕量 → 重采 GT **218 = 218 帧级完全配对**

最终数(同帧配对,唯一变量 = 光照;**2026-09-09 AP 口径修复后重算**,
见 §5.7b 尾部 bug 说明):
| 传感器 | 模型 | day_clear | sunset_glare | Δ |
|---|---|---|---|---|
| 相机 2D | YOLO11s kitti_finetune | Car AP 0.606 | **0.592** | **-0.014** |
| LiDAR 3D | pointpillars_kitti | 0.473 | 0.476 | +0.003(噪声内)|

- 相机逆光**掉点成立但幅度小**(Δ-0.014);天空带亮度 136→94 确认光照确实变了
- LiDAR 兜底**不受光照影响**(+0.003)——融合兜底逻辑成立
- 幅度小的根因如实记录(平台边界):CARLA 0.9.16 无镜头光学/高光饱和,AE 全局曝光自动补偿——"逆光眼瞎"在仿真里只能量出轻度掉点,真车镜头的大反差截断不可模拟(已在 scenarios.py fidelity 标注)
- 3D 侧 AP 绝对值低(0.47)是 pointpillars 预训练权重 × CARLA 点云域差,与本实验归因无关(两侧同差)

**P2 — 静态目标 + 道路特征 GT**(Town10 验证方法学)
- P2-1 技术选型裁决 §2 风险①:semantic LiDAR 后处理 vs RoadRunner 资产化
- P2-2 静态 GT 格式设计:信号灯(actor 有状态 API)/限速标志/车道线(semantic tag,号值以实测为准)→ 扩展 gt.py 静态类 + KITTI/nuScenes 落盘取舍
- 验收:一类静态目标 + 车道线样例输出,人工目检通过

### 5.6a P2-1 选型裁决(2026-09-09,probe 实测定案)

Town10HD_Opt 静态 GT 源三候选实测:
1. **semantic LiDAR 后处理——出局**:信号灯/标志**不是可打 tag 的实体**
   (无 traffic.traffic_light 蓝图可 spawn;世界内 15 个灯 actor 是地图
   dynamic 信号实例,非 semantic 实体),车道线非实体无 tag——LiDAR
   打不到,无从后处理
   〔2026-09-09 更正:原记"世界 0 信号 actor"有误,见 §5.8 事实更正〕
2. **RoadRunner 资产化——挂起降级**:可行但依赖 M4;且车道线/信号定义本就在
   xodr,RR 只是换地图时的载体 → M4-1 定制街道时复用
3. **地图查询 API(定案,0 外部依赖)**:landmark 65 个(58×Signal_3Light_Post01
   红绿灯 + 6×Sign_Stop + 1×Sign_Yield,含世界位姿/类型/id);车道线 =
   waypoint.lane_marking 实体(type SolidSolid/Broken × color Yellow/White ×
   width 0.125,沿 lane 中心采样重建折线)。上帝视角、传感器解耦、与帧对齐
   由 ego 位姿锚定。

边界如实记录:Town10HD_Opt **有 15 个信号灯 actor**(灯色可读,见 §5.8 更正),
但灯色属**动态** GT、本阶段不落盘(P2 只做静态);本图无限速牌
(landmark 面仅信号灯/停/让);车道线几何 = xodr 事实,若渲染 mesh 与 xodr
不符需目检兜底(第 5.6b 验收含目检图)。

### 5.6b P2-2 执行记录(2026-09-09 ✅ P2 验收)

**格式**(autodrivedata/static_gt.py,纯值不 import carla,沿用 gt.py 纪律):
- `training/static_gt/{fid}.json` = 一帧 StaticFrame:{}  signals(landmark
  归一:traffic_light/stop/yield,世界系锚点 + yaw)+ lane_lines(车道线段:
  side/type/color/width + 世界系点列);ego 位姿锚定,与帧对齐。
- 车道线 : 沿 ego 车道向前 5m×13 采样,mark 点落在车道边缘(lane_width/2),
  同侧同属性连续段合并(merge_lane_marks)。

**collect_static_gt.py**(采集器):锚定 pt0 定速直行,每帧查询 landmark(65m
视距,与 GT max_distance 一致)+ 车道线采样 → static_gt json + 原图 +
overlay 目检图(红圈=信号锚点,彩线=车道线段投影)。

**实测 bug 3 个(均已修 + 回归测试)**:
1. 信号虚高 21→5:同一物理信号杆挂在多条 lane(landmark 按 lane 引用,
   id 不同位置同)→ 按 (name,位置) 去重
2. yaw 怪值 (-360/-540):OpenDRIVE 方位累积 → % 360
3. 车道线全断成 1 点段:left/right 采样交替打断 merge → 分流各自合并(回归
   test_merge_lane_marks_alternating_sides_still_joins)
外加:无实体标线(type/color NONE 或 w=0 xodr 占位)过滤。

**验收(人工目检通过,2026-09-09)**:demo 40 帧
outputs/kitti_static_demo/——帧 10 画面可见 2 个红圈标注路口两侧信号杆
锚点(**真实画面里对面即有 3 根黄灯杆**,锚点在其基座处);帧 39 黄(左
Solid Yellow)白(右 Broken White)双线沿路缘延伸、透视收敛正确;信号集
跨帧一致性验证(视距内集合相同,出视距自然裁掉)✓

**边界如实记录**:静态 GT 为地图事实(xodr),与天气/光照/渲染**解耦**——
这是特性(静止目标真 GT),但换地图即换真值(M4 RoadRunner 时复用本链路)。
信号灯**灯色状态**不在本版本——灯 actor 存在(15 个,§5.8 更正),但灯色属
**动态** GT 范畴,静态 json 不收;实时可视化已画(actor API 直读)。

**M4 — RoadRunner 定制街道**(外部依赖,末位)
- M4-0 RoadRunner 可用性调研(trial 渠道/平台约束/0.9.16 USD-xodr 导入链路),P1 完成后启动
- M4-1 小型路网 → 定制地图端到端(P1/P2 方法学直接复用)

### 5.7 M4-0 调研结论(2026-09-09,Web 调研 + 本机核实)

**RoadRunner 侧(易解决)**:
- 商业授权:30 天 trial(MathWorks 账号,一次性)/校园·客户授权;Linux(Ubuntu 22.04)官方支持
  (R2023a 起 Ubuntu 22.04 有变数,R2023b 实测可装,缺 libssl1.1 需补)
- **RoadRunner 不是关键**:它只是 xodr 编辑器;xodr 可用免费替代
  (手写 OpenDRIVE XML / Road2Sim / esmini 生态)生成

**CARLA 消费侧(关键瓶颈,本机核实)**:
- 本机 0.9.16 prebuilt 包(19G,`/root/autodl-tmp/CARLA_0.9.16`):
  Binaries/Linux 仅 `CarlaUE4-Linux-Shipping`(**无 UnrealEditor 二进制**);
  Import/ 目录为空,ImportAssets.sh 只解包 UE 导出产物;**PythonAPI 无运行时
  xodr 入口**(仅 load_world/reload_world;get_traffic_light_from_opendrive_id 只做查询);
  HDMaps/ 为 Town01-07 的 .pcd 参考点云,与定制无关
- **结论:prebuilt 无法导入自定义地图**;官方导入链(make import /
  RoadRunner Importer 插件)全部要求 **CARLA 源码构建**
- 源码构建成本(官方文档):UE4.26 CARLA fork ~91G(CARLA 专用 fork,
  **需 Epic GitHub 账号关联授权**)+ CARLA 源码 1.2G + 资产 31G +
  编译输出 → **~170G 磁盘**(本机数据盘 190G 已用 121G/余 70G → **差 ~100G**,
  需清盘或增盘)+ 编译数小时 + **GPU 驱动栈回归**(现行 shim/EGL 修复
  glvnd 方案不一定覆盖源码版 UE)

**裁决:M4 的真正成本在 CARLA 源码构建,不在 RoadRunner** → M4-1 路线
待用户拍板(§5.7a)。

### 5.7b P1-4 雨夜 A/B(2026-09-09 ✅)+ **AP 实现 bug 修复**

**AP 口径 bug(2026-09-09 发现,重要)**:eval_2d_ab.py 旧 ap_for 尾行
`ap += (1-prev_r)*prev_p` 把"未达 recall=1 的部分"仍按最后 precision 计入
——最后一个预测是 TP 时,低 recall 数据被严重吹高(雨夜检出率 0.48 却报
AP 0.976 触发怀疑)。修复为 11 点插值(与 3D compare.ap11 同口径,尾部=0)。
**P1-3 数字随修复重算**(§5.5a):Δ-0.020→-0.014,方向不变。

**P1-4 雨夜(dense 雨夜场景,唯一变量 = 天气):**
| 传感器 | 模型 | day_clear | rain_night | Δ |
|---|---|---|---|---|
| 相机 2D | YOLO11s kitti_finetune | Car AP 0.606 | **0.453** | **-0.153 掉点成立** |
| LiDAR 3D | pointpillars_kitti | 0.473 | 0.490 | +0.017(噪声级)|

- 相机:强掉点(Δ-0.153)+ 检出/GT 0.72→0.48 —— **漏检型 corner case**
  (暗+湿反光对比度下降);与逆光的"轻掉点"互补,两型都量化到了
- LiDAR:点云物理不变(CARLA 雨丝只渲染、不断回波),3D AP 差=噪声级
  —— 融合兜底两实验一致成立
- 帧级配对:GT 219 vs 218,差 1 条(帧 28 rain 侧"贴相机掠影"GT:
  trunc=0.75/z=0.05m/单像素,0.5m 轨迹差所致边缘情形,对 AP 影响 ≤0.5%,
  如实记录)

### 5.7c P1-5 浓雾 A/B(dense_fog,唯一变量 = 天气)

| 传感器 | 模型 | day_clear | dense_fog | Δ |
|---|---|---|---|---|
| 相机 2D | YOLO11s kitti_finetune | Car AP 0.606 | **0.592** | **-0.013 掉点成立(FP 型)** |
| LiDAR 3D | pointpillars_kitti | 0.473 | 0.473 | 0.000(点云未变)|

- 相机:**FP 型掉点**——检出/GT 0.72→0.85(更多低质检测),AP 微降;
  与逆光(轻掉点)、雨夜(漏检型)互补,三型都量化到
- 雾对合成 LiDAR 零影响(AP 0.473=0.473):CARLA 雾渲染掩蔽、ray_cast 无
  雾衰减——LiDAR 雨/雾退化只能人工注入(与 scenarios.py fidelity 标注一致)
- GT 218=218 帧级完全配对 ✓

**P1 扩展三 A/B 汇总(2026-09-09,统一 11 点插值口径):**
| corner case | 相机 Δ | 相机型 | LiDAR Δ |
|---|---|---|---|
| 逆光 | -0.014 | 轻掉点(AE 补偿) | +0.003 噪声 |
| 雨夜 | -0.153 | 漏检型(检出 0.72→0.48) | +0.017 噪声 |
| 浓雾 | -0.013 | FP 型(检出 0.72→0.85) | 0.000 |

结论:相机在三个真实驾驶长尾上都有可量化掉点(三型各不同);LiDAR 兜底
在全部三个上都不受天气/光照影响(平台边界:雨/雾无物理回波效果,只能注入)。

### 5.7a 用户裁决(2026-09-09):**降级 M4 → 扩展 P1**

M4(定制街道)挂起,理由:M4-0 显示道路封闭 = 源码构建
(~170G 磁盘/Epic 授权/天级编译/驱动回归)对"一条演示街道"收益过载。
精力转入 P1 扩展:在现有平台再跑 corner case 定量 A/B
(P1-4 雨夜,P1-5+ 待定)——最终目标缺口中"场景矩阵"缺口直接壮大;
定制街道留待日后(换盘/有需求)按 M4-0 沉淀路线重开。

### 5.7d 地图池扩展 AdditionalMaps(2026-09-09 ✅)

降级 M4 后选定零构建扩地图池:官方 AdditionalMaps_0.9.16(14.8G)下载
(断点续传)+ 解压合并 → **Town11/12/13/15 入池**(13+4 = 17 图,磁盘余 45G)。
`load_world` 验证 4 图全通过(15s/50s/123s/188s);默认图仍 Town10HD_Opt
(DefaultGame.ini,重启即恢复)。

**归档处置(2026-09-12)**:安装包 `AdditionalMaps_0.9.16.tar.gz`(14G,数据盘)已删——
4 图已实装(本体 17 张 umap)且采集验证通过,要重装按官方 release 重下即可。

**采集验证(probe 分层定位,两型新问题 + 1 个采集器 bug)**:
1. **Town11/12 禁采集**:sync + ego + tick 均正常,spawn camera(attach_to=ego)
   瞬间 segfault(Signal 11)——该二图渲染资源与 headless GPU shim 栈冲突
   (机制未明);同链 Town10HD_Opt/Town13/15 全通(已记 testLog C20)
2. **Town13 TM 车流过载**:15 车 + TM 8000 首次 tick 把服务器打满
   (153% CPU,client 30s 无响应,非崩溃);降级 0 NPC(仅 ego autopilot)
   12 帧采集通(TM 大图实用性边界,testLog C21)
3. **锚定 yaw bug(已修)**:collect_static_gt 硬编码 yaw=0,Town10HD_Opt
   pts[0] 固有 yaw=0.16° 时恰好成立;Town13 pts[0] yaw=125.9° → 车道线
   采样沿 lane 方向走到车后、overlay 全空(投影深度全负,数值诊断发现)。
   改锚定用 **spawn point 固有 rotation**(地图作者设定的沿车道朝向):
   原图回归行为不变(yaw 0.2°),Town13 重采 overlay 恢复
   (每帧 988-1108 标注像素;静态 GT 本体 1 yield + 2 段 22 点车道线,
   地图查询链路新图自动生效)。collect_ab_route 的 yaw=0 锚定不动
   (P1 已验证基线,Town10HD 专用)。

**结论**:新图采集约束 = Town13/15 可用、Town13 TM 车流降级、Town11/12
禁采集;P1-6 候选场景与后续长尾数据源可在地图池内选图。

### 5.8 场景可视化实时流(2026-09-09 ✅)

**决策(两条外部候选 vs 自建)**:carlaviz(Three.js 线框)/ ROS2-Bridge+RViz2
都不选,三条理由:
1. **都不是 UE 渲染**——P1 的验证对象全是渲染效果(逆光过曝/雨夜对比度/浓雾),
   线框与点云 marker 看不到要验的东西
2. **版本/依赖不成立**:carlaviz 官方最高 0.9.15(无 0.9.16);AutoDL 容器无
   docker 无 ROS,ROS2 路线要容器/VM/Mac 三系统联调
3. 自建更省且口径同源:直接消费采集器同一条相机链 → **所见即落盘**,GT 框走
   label_2 同一投影函数(box_to_gt_line)
定案:自建 MJPEG 流(bin/view_stream.py,base env,当天可用)。

**实现**:
- `bin/view_stream.py`:3 视角(follow / top / grid6=nuScenes 6 向 3×2 拼图)
  + `--scene` 天气档 + `--npcs` 静置 NPC + `--speed` 定速直行 + HUD;
  框/行人/骑行者按类别着色,信号灯画灯色圆点;MJPEG 服务只绑 127.0.0.1
  (本地 ssh -L 隧道),最新帧槽不排队(客户端永远看最新画面)
- `world_to_img` 上移 `autodrivedata/calib.py`(+5 单测),采集器 overlay 与实时流
  **共用单一投影实现**;collect_static_gt 改用它(回归:红圈 504 px、白/黄车道线
  均绘制、static_gt json 内容不变)

**验收(数值诊断;`--dump` 落同帧 raw+overlay 做差)**:

| 视角 | 差集 | Car | Ped | Cyc | 信号灯 |
|---|---|---|---|---|---|
| follow | 3023 px | 1261 | 246 | 173 | 291 |
| top(60m) | 1995 px | 603 | 111 | 108 | 194 |

流:31/30 段 JPEG、首帧 640×360 有效、4.8/4.9 fps;退出清理干净(相机/actor
销毁 + 恢复异步)。

**坑(testLog C23)**:直接数 overlay 颜色会被场景自带绿(植被)/黄(标线)干扰——
top 视角曾误判"Car 仅 3 px";真因是 35m 高度只覆盖 ±20m(22/30m 的车在画面外),
高度提到 60m(覆盖 ±34m×±60m)后 Car 603 px 与框周长量级吻合。
**overlay 验证必须做差集,不能数绝对颜色。**

**事实更正(§5.6a 撤回一句,testLog C24)**:Town10HD_Opt **有** 15 个
`traffic.traffic_light` actor(Red/Green 状态,opendrive id 943–962,位置与
Signal landmark 重合 0.1m)——原记"世界 0 信号 actor"有误。xodr 实证:21 条
`<signal>` 定义 = **17 个 dynamic=yes**(15 个 Signal_3Light_Post01 + 2 个无名)
+ 4 个 dynamic=no(Sign_Stop/Yield);CARLA 只把**动态信号**实例化为 actor
(15 个,2 个无名的未实例化)。不变的两点:① 蓝图库确实无 traffic light 可 spawn
(原判断成立);② P2 静态源仍用 landmark(56×Signal + Stop/Yield 比 15 个灯头更
细),灯色属**动态** GT 故不进 P2 json;可视化侧用 actor API 实时画灯色。

**边界**:只读可视化,无交互控制;同步模式 tick 归本脚本,与采集脚本互斥。

### 5.9 灯色动态 GT(2026-09-09 ✅,工业口径对齐)

**工业口径(调研定案)**:灯态是**独立的时序语义层**,与检测框/静态地图几何分离。
- BDD100K:`trafficLightColor: red|green|yellow|none` 挂在框属性上(最省,无时序)
- WOMD:`traffic_light_state/current|future/state`,逐帧序列——**原始数据 71.7% 缺失或
  unknown**,靠地图+轨迹+环形规则补全后闯红灯估计 15.7%→2.9% → 说明①允许 unknown/
  遮挡(宁 Unknown 不猜)②有价值的是**变化点**与时间一致性③状态关联到**流向**而非灯头
- nuScenes:map expansion 的 `traffic_light` 层只有静态几何(位姿/灯泡/停车线),无逐帧状态

本项目取 WOMD 形态:逐帧状态 + 变化点 + 管制关系;Off/Unknown 保留不归一成三色。

**实现(三层)**:
- `autodrivedata/traffic_light.py`(纯值,不 import carla):`normalize_state` /
  `in_front` / `phase_at` + `TrafficLightState`(状态/灯头位置/管制车道/停车线/
  elapsed)+ `TrafficLightFrame`(逐帧 JSON 往返)
- `bin/carla_common.py`:`traffic_light_frame()` 把 actor 归一成纯值帧、
  `draw_traffic_lights()` 画色点——**采集器与实时流共用同一条实现**(目检所见 = 落盘口径)
- `bin/collect_tl_states.py`:记录模式(默认,不动灯)/ 受控模式(`--cycle 6,2,6`
  → freeze 全图灯 + 按 `phase_at(i·0.1s)` 驱动 → **确定性变灯序列**,真实数据集最缺的样本)

**落盘**:KITTI root 扩展 `training/traffic_light/{fid}.json` + `image_2/` + `overlay/`。

**证据归档(2026-09-12 清盘)**:采集时落在 `/tmp` 的四套灯态运行(受控 tl_static 90 帧、
记录 tl_record 40 帧、演示 tl_demo/tl_demo2 各 90 帧,合计 594M 的 image_2/overlay)已清,
**GT 全量迁入项目**:`outputs/kitti_tl_{static,record,demo,demo2}/training/traffic_light/`;
受控那套另存变化帧 overlay(帧 000000/000001/000059-61/000079-81)+ 切灯对比网格 + 灯头裁剪
诊断图。复核:受控序列变化帧 = **60/80**,与本节验收一致。

**验收(受控 6/2/6 @0.1s 步长,90 帧)**:
- 状态变化点 = 帧 **0 / 60 / 80**,与计划逐帧吻合(灯 949/957/958/959 一致)
- `affected_lanes`/`stop_lanes` 非空(如 #950:4 条管制车道 + 2 条停车线)
- overlay 差集与"画面内灯数"相关 **0.99**(≈412 px/灯:圆点 + 文字)→ 绘制口径正确
- view_stream 回归:差集 3522 px(车 1257/人 246/骑行 174/灯点 三色共 707)

**前向过滤(实测驱动的修正)**:纯圆形 horizon 把身后 120m 的灯一起收进来——90 帧里
视距内 **1046 灯次有 79% 在车后**(前方仅 221,画面内 131)。加 ego 前向半平面过滤
(`in_front`,默认开)后为 221 灯次。工业数据集(BDD/WOMD)同样只标视野内/本车相关灯。

**平台事实(受控探针实测,testLog C25–C27)**:
- 渲染**确实跟随** `set_state`:相机正对镜片时依次拍到 **红上/黄中/绿下** 点亮
  (带网点发光纹理)。但镜片直径 0.2m,在 KITTI 口径相机(f=621)下 30m 处仅约 4px、
  60m 处 2px → **"按像素颜色做 GT↔图像一致性验证"在 ego 视角不成立**:12–30m 处
  采到的是**黄色灯箱外壳**(实测 RGB≈(255,237,0),与黄灯镜片同色相),一致性率被
  拉成 23% 的假象。结论:灯态 GT 是**逻辑层**,图像不提供强视觉证据。
- 一个灯 actor 管**多个灯头**(`get_light_boxes()` 实测 4–5 个,分布在不同 x/y/z:
  杆顶 z≈4.05 + 悬臂 z≈5.15);json 里 `location` = actor 锚点 + 4.5m 的近似灯头,
  管的是"哪个流向"靠 `affected_lanes` 而非坐标。
- `elapsed_s` 语义:红灯相位**恒 0**,绿/黄相位自相位起点计时(实测 0.3→0.8);
  受控模式恒 0(取值 {0.0,0.2})→ **变灯时刻以状态序列变化点为准**(受控模式另有
  `phase_plan` 可精确反推),不要把它当通用"当前状态已持续时长"。

**同步模式快照坑(testLog C28)**:连到已处于同步模式的服务器时,首个 `get_actors()`
返回空(快照只在 tick 后刷新)→ 采集器"清场"循环会静默漏清。已统一修在
`carla_common.sync_mode()`(应用设置后补一次 tick)。

**边界**:灯态来自 actor API(逻辑真值),不依赖渲染;图像侧镜片过小,不做视觉回归。

### 5.10 参数扫描 + 失效归因(2026-09-09 ✅,P1 定量延伸)

**问题**:§5.7c 三 corner case 只回答"AP 掉多少"。"漏在哪个距离/尺度/安全余量、为什么漏"
是另一层问题——必须逐帧匹配 + 每个 GT 带上下文。

**口径(与 eval_2d_ab 同源、分工不同)**:
- `eval_2d_ab` 全库池化 PR → 结论层(AP,已定案)
- `eval_attr`(新)逐帧贪心匹配(conf 降序,每预测认领一个 GT)+ 每 GT 记录
  距离/框高/TTC/截断/框内亮度·对比度·梯度 → 归因层
- 两者共用同一个 `box_iou2d`(`autodrivedata/attribution.py` = 全项目唯一 2D IoU 实现;
  eval_2d_ab 的本地副本已删)→ **检出率与 AP 两个数字互相解释得通**

**交付**:`autodrivedata/attribution.py`(纯值,不 import carla/PIL/ultralytics)+
`bin/eval_attr.py`(`--run 名字=路径:速度` 可重复 → 每跑分箱表 + 漏检画像 +
跨跑距离/框高网格 + `--json`)+ `tests/test_attribution.py`(28 例)+
数据 `outputs/kitti_sweep_day_clear_{4,8,12}`(同 56m 里程)+ `outputs/attr_all.json`。

**发现 1:定速失效(brake 残留)→ 已修,老数据集速度需换算**

collect_ab_route 放静置车时设 `brake=1.0` 站定,进采集循环后**未解除**——VehicleControl
每步生效,`set_target_velocity` 被制动抵消(probe 实测):

| 命令 m/s | 无残留 | 有残留 |
|---|---|---|
| 4.0 | 4.00 | 2.60(0.65×) |
| 8.0 | 8.00 | 6.59(0.82×) |
| 12.0 | 12.00 | 10.59(0.88×) |

→ **P1 四个 A/B 数据集实际均为 6.60 m/s**(标称 8.0),TTC 用标称值偏小 18%。
修法:采集循环前 `apply_control(carla.VehicleControl())` 清残留。新数据集实测
4.00/8.00/12.00,1–2 帧到稳态(**曾据 ego-x 打印误判"2 秒加速段",逐帧速度序列证伪**;
报告标签 加速段→慢速帧,`closing_speed_series` 逐帧自证取代命令行标称值)。

**发现 2:CARLA 无运动模糊(平台边界)**

速度扫描 4/8/12 m/s(同里程同布局):同距离箱的**梯度能量几乎相同**(10-20m
35.6/35.2/34.8;30-40m 61.6/61.0/60.8),池化 10-40m 检出率 0.914/0.886/0.909
→ **速度不改变图像质量**(合成渲染逐帧静态曝光,不模拟快门/运动模糊)。与"雨雾对
合成 LiDAR 无物理回波"同类:速度退化只能人工注入,不能指望采集器给。

**发现 3:决定检出率的是尺度,不是距离/速度/亮度**

框高分箱(Car,trunc=0;括号内 GT 数):

| 框高 | day4 | day8 | day12 | p1_day | sunset | rain | fog |
|---|---|---|---|---|---|---|---|
| 16-32px | 0.36(128) | 0.27(62) | 0.38(40) | 0.31(77) | 0.28(78) | **0.15**(79) | 0.47(77) |
| 32-64px | 0.94(111) | 0.93(56) | 0.97(38) | 0.99(68) | 0.90(67) | 0.78(68) | 1.00(68) |
| 64-128px | 0.96(72) | 0.97(35) | 0.92(24) | 0.92(37) | 0.97(37) | 0.83(35) | 1.00(37) |
| 128+px | 1.00(20) | 0.80(10) | 1.00(7) | 0.90(10) | 0.92(12) | **0.58**(12) | 1.00(10) |

- **<32px 一律 0.15–0.47,≥32px 一律 0.78–1.00**——距离断崖在 30-40m(框高均值 28px)
  / 40-50m(21px),换算阈值 ≈ **21–24px**
- 漏检画像(中位):命中 ~20m / 漏检 ~40–48m;框内亮度几乎无差(day 118 vs 134,
  rain 33 vs 37)、对比度差也小 → **主因是尺度,不是"暗"或"对比度低"**

**发现 4:天气只是把断崖前移;雨夜另有近场异常**

距离箱(6.60 m/s,同布局):

| 距离 | day | sunset | rain | fog |
|---|---|---|---|---|
| 20-30m | 0.98 | 0.87 | 0.71 | 1.00 |
| 30-40m | 0.69 | 0.66 | **0.32** | 0.91 |
| 40-50m | 0.07 | 0.00 | 0.03 | 0.21 |

- 雨夜把零检出从 40-50m **提前到 30-40m**;雾反而最晚(0.91)——与 §5.7c 的
  "雨夜漏检型 / 雾 FP 型"一致,这里给出了距离维度的落点
- **雨夜独有形态:近场也掉**(0-10m 0.56、128+px 0.58),与 sunset/fog 的距离单调
  衰减不同类;疑似湿地面反射/大灯眩光,未验证,不作结论
- TTC 读法(安全余量):8 m/s 下 4.5-6s(36-48m)已掉到 0.13 → 本数据集"6 秒余量"
  实际不可靠;同一箱在 4 m/s(18-24m)是 0.96 → **TTC 箱不可跨速度比较检出率**
  (模块 docstring 已写明,避免误用)

**边界**:①单图单场景(day_clear + 3 天气),结论是"尺度主导"的定量证据,非跨地图泛化;
②静置目标只有 Car,行人/骑行者样本不足;③速度维度因平台无运动模糊只能证伪,真实
速度退化需人工注入(与 §5.7a 同思路)。

### 5.11 地图矢量管道(MapTR/MapQR 口径)编排(2026-09-10 定案,待执行)

**需求**:已有 2D 检测 GT(label_2)、3D LiDAR GT、静态 GT(P2)、灯色 GT,**缺 BEV 矢量地图 GT**。
MapTR/MapQR 类架构(端到端 vectorized map)的输入 = 多视角环视图像 → BEV 特征 → 实例级折线
(类 + 固定点数折线),本工作包补齐这条输出管道。

**用户裁决(2026-09-10)**:① 目标形态 = **分阶段,A 先 B 紧随**;② 要素范围 = **MapTR 三类 +
工程补充**;③ 输出格式 = **KITTI root 扩展 + 转换器**;④ 验收 = **几何自证 + overlay 目检**。

**核心架构决策:离线 xodr 解析为主,运行时 API 为 oracle**

1. 地图矢量是**整图静态事实**,不随帧/天气/光照变化 → 不绑进采集循环(采集器要清场/同步/
   tick,重且不可复现)
2. `autodrivedata` 不 import carla 的纪律 → 纯值解析器天然契合,任何 env 可单测(与 attribution.py
   同性质)
3. **21 个 xodr 已在本机磁盘**(`CARLA_0.9.16/CarlaUE4/Content/Carla/Maps/**/OpenDrive/*.xodr`,
   覆盖全部 17 图)→ 零服务器依赖、毫秒级、可复现
4. **CARLA 自带 oracle**:`map.get_waypoint_xodr(road_id, lane_id, s)` 返回运行时几何 → 离线
   解析的采样点逐个对账(最强形式的"几何自证")
5. **意外收益**:Town11/12 **禁采集**(C20 spawn camera segfault)但**地图矢量照样出**——
   离线路径不碰渲染

**要素映射(xodr 语义 → MapTR 口径;实测于 2026-09-10)**

| MapTR 类 | 来源(xodr) | 本机覆盖 |
|---|---|---|
| `divider` | 同向车道间的 `<roadMark>`(solid / broken / solid solid,**排除 curb**) | 全图,2.8k–40k 条/图 |
| `boundary` | 道路外沿:`type="curb"` roadMark + lane type `sidewalk`/`border` 外边界 | 全图 |
| `ped_crossing` | `<object type="crosswalk">` 的 `<outline>` 4 角多边形(5 点闭合) | Town03 73 / Town05 66 / Town04 27 / Town10HD_Opt 16 / Town06 14 / Town07 7;Town01/02/11/12/13/15 = 0(地图作者未放置,**非提取失败**) |
| 补充 `stop_line` | `<object name="StopLine">`(与 crosswalk 同路径,含 s/t/hdg/width/length) | 待摸底确认 outline 结构 |
| 补充 `centerline` | lane 中心线(t = ±w/2 中点链) | 全图 |
| 补充 灯-车道关联 | `<signal>` 的 `<validity>`(复用 P2 landmark 口径) | 全图 |

**阶段 A(离线矢量库,无服务器,交付物见下)**

| 步骤 | 产物 | 验收 |
|---|---|---|
| A1 解析器 | `autodrivedata/opendrive.py`:ElementTree 解析 `<geometry>`(line/arc/spiral/poly3/paramPoly3)+ elevationProfile + lanes/width + roadMark + junction + object;核心 `road_to_xy(road, s, t)` | 单测(直线/圆弧闭式解手算)+ API oracle 对账 |
| A2 要素提取 | `autodrivedata/mapvec.py`:上表映射 → 实例(类 + 折线 + 属性 + 实例 id) | 每类计数/拓扑自证 |
| A3 采样与裁剪 | 等距重采样(divider/boundary 20 点;ped_crossing 4 角 → 2 点长轴)、`crop_to_ego(pose, ±51.2m)` | 采样间距/点数断言 |
| A4 导出 + 目检 | `bin/export_mapvec.py` → `training/map/{map}_full.json` + `{fid}.json` + BEV overlay 图 | overlay 目检(同帧差集口径,C23) |
| A5 转换器 | `bin/convert_mapvec.py` → MapTR 目录结构(annotation json + 可选 BEV 渲染) | 往返断言 |
| A6 验收三件套 | — | ①几何自证(闭合/自交/曲率/点在可行驶域)②API 交叉验证(`get_waypoint_xodr` 抽样 < 5cm)③overlay 目检 |

**阶段 B(多视角采集,紧随)**:B1 `collect_drive.py` 扩环视相机(6 视角)+ 内外参导出;
B2 数据集组装器(图像 + ego pose + map GT → MapTR 训练格式);B3 验收 = 矢量投影回各视角
图像 vs 渲染一致性(数值诊断,不做视觉回归)。

**边界/风险**:①环视 6 相机 + LiDAR 单卡吞吐未测(B 阶段先探 FPS);②xodr → MapTR 三类是
**有损映射**(xodr 语义更细),映射表进文档,**不静默丢要素**;③依赖方向单向:
MapTR 实现不反向依赖 AutoLabel;④不改 A/B 采集纪律。

### 5.11d C/D 阶段实现路径定案(2026-09-11 用户裁决)

- **先参考自实现**:忠实移植 MapTR 核心数学(GKT BEV 变换 + 分层 query + 置换等价
  匹配 + focal/L1),基础设施用现代栈(torch2.x,autodrivedata env),不动 autolabel
  生产环境
- **老栈降级 optional**:`maptr` env(py3.8 + torch1.9 + mmcv-full1.4)不再作为 C 阶段
  前置;**若后续需要官方权重对标**(绝对性能数字),再启动官方对照(工作记录在案)
  → **已于 2026-09-13 启动**(见 §5.12;实测官方权重不可得,口径修正为"官方代码对标")
- **正确性口径**:无官方权重对照 → 实现正确性靠**单帧过拟合测试**(小数据上 loss
  必须压到 ~0)锚定;性能数字只做**内部对比**(场景/天气间)
- 结构:新建顶层包 `maptr_impl/`(torch 依赖,不进 autodrivedata 纯值包),评估
  chamfer AP 入 D 阶段

### 5.11e C 阶段执行记录(2026-09-11 ~,参考自实现)

- **C1 GKT**(`maptr_impl/gkt.py`):BEV 200×100 @ 0.3m(x∈[−15,15] 前向 / y∈[−30,30]),
  相机链投影 + bilinear 采样 + **topk=1 最近相机独占融合**(贴近官方交叉注意力口径;
  加权平均在相机边界混叠出 0.5 值,弃用)。6 单测与 calib.world_to_img oracle 锁定
  (平/斜/端到端 <0.01px)
- **C2 分层 query head**(`maptr_impl/head.py`):实例 query + 点 query + 6 层解码器
  (点级 BEV 采样 → 回归 → 均值回聚);置换等价匹配(按类匈牙利 + GT 双向增强,
  代价 = −logit + 5·L1);focal + 5·L1 损失(官方 pts_loss_coef)。7 单测
- **C3 组装 + 训练入口**(`maptr_impl/model.py` / `dataset.py` / `bin/train_maptr.py`):
  ResNet50+FPN(P2)+GKT+head 33.2M,ImageNet 预训练;单帧过拟合 = 正确性锚点
  (判据:最后 20 步平均 total < 0.5)。修 `_sample_bev` expand 物化 4GB 临时块
  OOM(grid 打平进 H_out 维)
- **C4 单帧过拟合** ✅:两轮失败归因后通过。①lr=1e-4 × 400 → 14.9→5.9(优化量不足);
  ②lr=5e-4 × 1000 → 8.5→0.90 噪声平台——**根因 = B2 GT 未裁 BEV 窗口**(首帧
  GT 点 ~80% 在 60×30m 窗外不可观测,L1 卡 0.19m 是几何事实非实现错误);修
  `clip_to_bev`(bb3292d)重装 infos 后 ③同参重跑:**最后 20 步平均 total =
  0.395 < 0.5**(cls 0.0002 / 点 L1 0.08m)。**数学链自洽锚定通过**
- **C5 正式训练**:300 帧采集(0.6 fps autopilot,8.3 分钟)+ infos 组装(44.9 万
  GT 点,窗外 0);训练帧 0–199,留出 200–299 供 D 阶段评估。首轮 batch 2 × 24
  epochs(lr 5e-4)损失平坦 10.94→10.53;续训 +48 epochs(阶梯 lr 衰减 +
  `--save-every` 防长训中断丢进度)至 72 累计,最后 20 步平均 10.24——仍在
  高位,损失未收敛
- **D 阶段评估**(72-epoch ckpt,score-thr 0.2,官方 chamfer AP 口径):

  | 集合 | mAP | divider / ped / boundary / center |
  |---|---|---|
  | 训练集对照(帧 100–199) | **0.0061** | 0.0059 / 0.0030 / 0.0050 / 0.0104 |
  | 留出集(帧 200–299) | **0.0045** | 0.0041 / 0.0007 / 0.0050 / 0.0083 |

  双低且无过拟合间隙 → **欠训练**(损失 10 高位,不是管道故障——C4 锚点已
  证明数学链自洽)。Pareto 定案:C4 锚点 + 测量装置 + 诚实基线数字 = D 阶段
  交付物;长训收敛(400+ epochs)列为可选后续
- **D 纯值**(`autodrivedata/chamfer_ap.py`):chamfer 距离 + 贪婪一对一匹配 +
  阈值 {0.5,1.0,1.5} AP(官方口径),7 单测(单点/多点/边界/向量化等价);
  `bin/eval_maptr.py` 评估入口(--score-thr 可扫,--start 留出集,--match 后端)。
  代价矩阵向量化:GEMM 平方展开 + 补齐虚点 1e4m + 三阈值共用 + float32
  内部口径(min 后 float64 累加,误差 ~1e-3m 远小于 0.5m 阈值),随机交叉
  验证与逐对口径 1e-3 等价(语义错误 O(1) 量级必被抓)
- **GPU 匹配**(`maptr_impl/chamfer_gpu.py`):代价矩阵 CUDA 版(torch 约束在
  maptr_impl,autodrivedata 纯值纪律不破——chamfer_ap 经 cost_fn 注入,
  贪婪配对复用纯值 match_greedy);4 GPU 单测交叉锁定。held-out 复验
  mAP 0.0045 与 CPU 慢路径**逐位一致**;评估 75min → **~30s**(推理 25s /
  匹配 134s → 1.7s)
- **自适应 GPU batch**(`maptr_impl/device.py`,参考 AutoLabel tools/device.py):
  实测增量法——batch 1 warmup + batch 2 增量测 forward+backward 每样本显存,
  budget = 空闲 × 0.85,bs 钳制 [1, --max-batch];train_maptr `--batch 0`
  (默认)= 自适应、N>0 显式优先。真实模型冒烟:实测 batch = 4(空闲 10.0 GiB)
- **长训 400 epochs 启动**(2026-09-11 夜):72-epoch 权重续训(优化器重置)→
  `outputs/maptr_400.pt`(72e 基线保留),lr 2e-4 / batch 自适应实测 **5**(空闲
  11.2 GiB × 0.85)/ --save-every 12 / --log-every 6。启动 7.2 分钟到 epoch 6+
  → **~72s/epoch,全程 ≈ 8h**;前 6 epoch 损失 11.23(从 10.24 起步,无 NaN)。
  判据:训练集对照 + 留出集 mAP 双升且显著超 72e 基线(0.0061 / 0.0045)
- **lr 归零根因**(epoch 150 发现):train 脚本阶梯衰减 lr 每 **12** epochs 减半,
  epoch 144 时 lr ≈ 2e-4×2⁻¹² ≈ 5e-8 ≈ 0——损失 7.72 平台**不是收敛而是 lr 归零**
  (同理解释 C4 早期"优化量不足"的 1e-4×400 失败:后半程 lr 已归零)。修:
  train_maptr 加 `--lr-halve`(默认 12 保持旧行为,0=不衰减);期间用户看 live
  场景(暂停/恢复代价 ≈ 每次续训优化器重置回跳 ~1-2 损失单位)
- **长训 lr 扫描**(2026-09-12 凌晨,三次重启数据点,均从 e132 备份 loss 7.73 起):
  | lr | 重启尖峰 | 恢复行为 | 结论 |
  |---|---|---|---|
  | 5e-5 | +2.2(9.97) | **立即单调降**(-0.03~0.05/6ep) | ✅ 最终采用 |
  | 1e-4 | +2.4(10.15) | 平 6ep 后缓降 | 弃 |
  | 2e-4 | +2.9(10.65) | 爬升到 10.65 平台打滑 | 弃 |

  规律:优化器重置(Adam 力矩清零)后的尖峰随 lr 增大,**大步长在该损失盆地边缘
  打滑**——小步长慢但稳。教训:--save-every 覆盖存盘把 epoch-144(7.72)冲掉,幸有
  /tmp 快照备份——**对 --out 文件做破坏性操作前先 cp 备份**(已落
  outputs/maptr_400_e132.pt);后续可加优化器状态侧车文件使暂停/恢复免尖峰
- **长训最终结果 ✅**(2026-09-12 凌晨):lr 2e-4 每 **48** epochs 减半 × 256,
  从 72e 重启(尖峰 11.24),前半程缓降(108ep 时 9.83 疑似停滞),**lr 降到
  5e-5 后加速下冲**:终值 **5.99**(最后 20 步 6.34,首轮 7.72 大幅超越),
  结束时仍在陡降。评估(score-thr 0.2):

  | 权重 | 训练对照 | 留出集 mAP | divider / ped / boundary / center |
  |---|---|---|---|
  | 72e 基线 | 0.0061 | 0.0045 | — |
  | snap≈ep150 | — | 0.0075 | — |
  | snap≈ep200 | — | 0.0351 | — |
  | **ep256 终值** | **0.1071** | **0.0510** | 0.0566 / 0.0033 / 0.0493 / 0.0948 |

  轨迹单调升 = 终值即最优,**损失仍在下行 → 续训继续获益**(判据:留出集爬升
  未停);训练/留出间隙 2.1× 是 200 帧小数据的必然,扩数据或早停由快照轨迹裁决。
  产物:mapvec_pred_final_held.{json,png}(pred 1654/399/3058/4372)
- **评估口径固化**(2026-09-12,阈值敏感度实测):同一权重 score_thr 0.2→0.4 给出
  **0.0510→0.1350(2.6×)**。根因:`chamfer_ap` 是 3 阈值 precision 均值、**无 recall
  项** → 保守操作点(预测少而准)天然占优;默认 0.2 是任意选的。**规则:跨权重比较
  固定 --score-thr,绝对数字必须带阈值**;看曲线用 `eval_maptr --sweep`(单次推理,
  阈值纯后处理,已验证扫描行与单阈值路径逐位一致)。留出集对照:
  | score_thr | 72e 基线 | ep256 | 倍数 |
  |---|---|---|---|
  | 0.2 | 0.0045 | 0.0510 | 11× |
  | 0.3 | 0.0062 | 0.0952 | 15× |
  | 0.4 | 0.0078 | 0.1350 | 17× |

  结论不变(提升在所有阈值成立),但**绝对值只在带阈值时可引用**
- **重启尖峰根因修复**(`bin/train_maptr.py`,2026-09-12):尖峰 = 优化器状态丢失
  (Adam 力矩清零后每步退化为 ~lr·sign(g))× 步长。e132 三档 lr 全败即此——该 ckpt
  结束 lr≈2e-7,盆地极窄,任何有效步长都把它踢出去。修:①**优化器状态侧车**
  `<out>.opt`(与 --out 同步写、--init-ckpt 自动载入、--no-opt 关闭);②**--warmup N**
  lr 线性升温。实测:ep256 → lr 2e-5/warmup 3 重启,尖峰仅 **+0.8**(6.77@ep6)且
  ep12 即回落 6.68,对比 e132 同量级 lr 的 +2.2~+2.9
- **第二轮续训启动**(2026-09-12 中午):ep256 冻结为 `outputs/maptr_ep256.pt`(留出集
  0.0510 的参考产物)→ lr 2e-5 每 128ep 减半 × 256 epochs(batch 5,~34s/ep ≈ 2.4h),
  out=`outputs/maptr_ep512.pt`,30min 快照循环 + 侧车同步落盘
- **live 场景间隙交付**(训练暂停期):`bin/viz_maptr_pred.py` 预测回投目检
  (pred 品红 / GT 青绿 → 6 相机拼图 + BEV 面板,投影链与 B3 同式,内参从 infos
  直读;epoch-132 权重 6 帧产物 `outputs/viz_maptr_e120/frame_*.png`,raw 零
  撞色数值验证);`bin/drive_ego.py` 终端驾驶(WASD + 手刹,持续 apply_control
  喂 view_stream 的同步 tick,ego 识别=挂相机的车,快照重试)
- **预测落盘产物化**(`bin/eval_maptr.py --out-pred`):跨帧汇聚口径(与评估一致)
  落 `<path>.json`(classes + preds 含 score/points + gts,供 AutoLabel 消费)+
  `<path>.png`(matplotlib BEV 2×2 目检图,红=pred 绿=GT,窗口与模型输出同系
  x∈[−15,15] / y∈[−30,30])。新增 `--device`(GPU 被长训占用时 CPU 逃生)。
  20 帧留出集 CPU 端到端验证通过:产物 `outputs/mapvec_pred_72e_held20.{json,png}`
  (JSON 结构 + 数值色检红 31.6k / 绿 20.4k 像素)
- 提交 62f9121 / bb3292d / e73adcd / fd08442 / 7f1bd8a / f3a7478 / 0888e3a / 257e031

### 5.11f C 阶段收尾(2026-09-12 ✅):第二轮结果 + 实时 overlay + 两处根因

**① 第二轮续训(ep512,`outputs/maptr_ep512.pt`)结果——口径分化**

| score_thr | ep256 | ep512 | 变化 |
|---|---|---|---|
| 0.2 | 0.0510 | **0.0674** | **+32%** ✅ |
| 0.3 | 0.0952 | 0.0904 | −5% |
| 0.4 | 0.1350 | 0.1280 | −5% |

- 快照轨迹单调:r2_1329(≈ep145)0.0514 → r2_1400(≈ep230)0.0616 → ep512 **0.0674**
  → 终值即最优(选 ep512 的理由)
- **训练集对照 0.2603(帧 100–199)** vs 留出集 0.0674 → **泛化间隙 3.9×**
  (ep256 时是 2.1×)。结合"低阈值升、高阈值不升":模型在**保守操作点**上更敢输出
  (覆盖收益),但高阈值精度无增益 → **下一轮收益应来自扩数据,不是继续长训**
- 评估口径提醒见 §5.11e:绝对数字必须带 `--score-thr`,跨权重固定阈值比较

**② 投影单位 bug(已提交的 viz 里,用户发现的第三交付项带出的)**

- 症状:`bin/viz_maptr_pred.py` 的 `_cam_pose` 把**度**直接传进吃**弧度**的
  `calib.world_to_img`/`carla_rotation_matrix`。6 相机里只有 yaw≈0 的 CAM_FRONT
  恰好接近正确 → 目检不炸、数值不查则漏
- 判据(与目检无关):**命中点方位角落在该相机自身 yaw±45°(其 90° FOV)内的比例**
  ——弧度口径 91–100%,度数口径 **0–6%**(一次性诊断脚本曾放 /tmp,已废弃;
  该判据现固化为 `tests/test_mapviz.py`)
- 修:抽 `autodrivedata/mapviz.py`(纯值:**位置米 / 姿态弧度**出口口径,离线 viz 与
  实时流共用一条链)+ `tests/test_mapviz.py` 23 例(主锚点 = 旋转单位,含解析期望
  u = cx + fy·lat/fwd 手算);`viz_maptr_pred` 复验逐相机 GT 分段 23/26/24/16/7/13
  **全非零**(修复前侧/后相机近零)。教训:**"能跑出图"不是投影正确的证据**,要数值判据

**③ 实时 overlay 交付**(`bin/view_stream.py --maptr-ckpt/--maptr-thr/--maptr-scale/--maptr-bev`)

- rig 只认 `collect_surround.SURROUND_CAMS` **一处定义**(BACK_LEFT 235 / BACK_RIGHT 125
  与显示用 CAM_YAW_OFFSET 正好镜像互换——错位则第 i 路图与其学过的语义错位);推理用
  **实挂相机世界位姿(弧度)**,与 mapviz.cam_pose 同口径
- 实况数值验证(40 s / grid6 / ep512,`--dump` 做差集):
  overlay 品红 **25553 px** vs raw **0**(场景本身不含品红)✓;6 瓦片全覆盖(2047–7963 px)✓;
  **地平线以上 0 / 18496**(地面矢量必须全部 v > cy),v_min 109–127 与解析值
  "30m 地面点 v ≈ cy + fy·1.65/30 = 110.6" 吻合 ✓;BEV 内嵌 7057 px ✓
- FPS 1.1–1.4(6 路 1242×375 推理 + CARLA 共享 GPU,显示侧 621×187);够看、不够流畅
- **自检假阳性修复**:`rig_yaw_deviation` 在 tick 前读传感器 transform(全 0 陈旧值)
  → 假报 179.841°(= CAM_BACK 规格 180 − ego 固有 yaw 0.159)。加一次 `world.tick()`
  后 **0.000°**(实测逐台 dev 全 0)。与 C19"快照只在 tick 后刷新"同类坑,勿在 tick 前读 actor

**④ CARLA 渲染停摆根因:宿主驱动升版后 Vulkan ICD 加载失败(环境级,非本项目代码)**

- 现象:GameThread timed out waiting for RenderThread (60 s) + Signal 11、显存恒 **0 MiB**;
  端口 2000 能通(RPC 起得来)、首启无 shader 编译
- 根因(`VK_LOADER_DEBUG=all`):`ERROR: libnvidia-gpucomp.so.580.105.08: cannot open
  shared object file` → `loader_icd_scan: Failed to add ICD JSON libGLX_nvidia.so.0` →
  **枚举 0 个 Vulkan 设备** → UE4 渲染线程无从初始化。镜像里 580.76.05/580.82.07 的
  gpucomp 是**真实 72 MB 文件**、580.105.08 **完全缺失**(宿主驱动已升到该版本)
- 处置:`bin/carla_server.sh` 加 `setup_gpucompat()`——检测当前驱动版本是否缺 gpucomp,
  缺则用镜像自带副本按缺失 SONAME 顶名到项目私有目录(`outputs/carla/nvidia-compat`,
  见 §5.11g)+ `LD_LIBRARY_PATH` 注入(**不动 /usr/lib**,驱动回退后自动免用);
  判据脚本 `bin/probe_vulkan.py`:无兼容层只剩 llvmpipe(1 个),有兼容层
  NVIDIA+llvmpipe(2 个)且应列出 `NVIDIA GeForce RTX 3080 Ti`(对照 lavapipe 验证探针本身)
- 修复后:server 0.9.16 / 5036–5629 MiB / 日志零崩溃标记 / view_stream 实况通过

### 5.11g 产出归拢项目内(2026-09-12 ✅,用户要求)

- 缘起:用户将清理系统盘(/)与数据盘(`/root/autodl-tmp`)。历史口径是"相对 cwd 的
  `outputs/...`"——从别处 cwd 调用就把权重/可视化散到项目外,清盘时无从分辨;运行支撑物
  (shim、Vulkan 兼容层、服务器日志)散在 `/tmp` 与 `carla_home`,**清盘即失效**
- 做法:`autodrivedata/paths.py`(纯值,只 pathlib)提供 `PROJECT_ROOT`/`OUTPUTS`/
  `project_path()`/`ensure_parent()`——**写盘相对路径一律解释为相对项目根**,绝对路径
  原样放行;**读路径不锚定**(输入沿用 cwd 口径);16 个脚本的 `args.out*` 全部过 `project_path`
- 运行支撑物一并迁入 `outputs/carla/`:shim(`libmhookshim.so`,缺失时 `ensure_shim()`
  现编 `bin/gpu_fix/mhookshim.c`)、`nvidia-compat/`、`carla_server.log`;
  `bin/gpu_fix/install.sh` 同步改为编到项目内(原 `/tmp/libmhookshim.so` 与
  `carla_home/nvidia-compat` 副本已废弃)
- 验收:清空 `outputs/carla/` 后 `bash bin/carla_server.sh start` → 现编 shim + 重建兼容层
  + server 0.9.16(零崩溃标记);从 `/tmp` cwd 跑 `viz_maptr_pred.py --out-dir outputs/viz_check`
  → 图落在项目内、`/tmp/outputs` 不存在;`tests/test_paths.py` 6 例(含 chdir 锚定回归 + 包纯度)

未提交清单(用户手动提交):见对话末尾提醒。

### 5.11h 预测的逐帧契约 `mapvec_pred/1`(2026-09-13 ✅,接 AutoLabel 前置)

**缘起(接入缺口)**:`--out-pred` 的单一 json 是**跨帧汇聚**产物——100 帧的实例堆成
按类的池子,实例里**没有帧归属**。消费方(AutoLabel)无法把一条预测对应到哪一帧/
哪张图,作不了帧级对账。补的不是字段,是接口形态。

**做法**:
- 新纯值模块 `autodrivedata/mapvec_schema.py`(不 import carla/torch,纪律同 mapvec):
  `MapVecInstance` / `MapVecFramePred` + `validate_frame` + `dump_frame`/`load_frame`,
  逐帧一文件 `<DIR>/{token}.json`;**GT 与 pred 同文件携带**(消费方不必解析
  `map_infos.json` 就能对账;要对回图像按 token 去 `<root>/cam_*/{token}.png`)
- 头部字段即契约:`schema`(固定 `mapvec_pred/1`,不匹配拒收)/`frame`/`token`/
  `classes`(类序取 `mapvec.MAPTR_CLASSES`,**不另立一份**)/`coord`(固定 `ego`)/
  `bev_range`(取 `mapvec.BEV_RANGE`)/`num_points`/`score_thr`(绝对数字必须带阈值)/
  `ckpt`(溯源)
- 入口:`bin/eval_maptr.py --out-frames DIR`(复用同一次推理,`--score-thr` 决定落盘阈值;
  `floor` 只服务内部 `--sweep`);写盘路径过 `project_path`(产出纪律)
- **设计判据(实测反推)**:越窗**不算错误**。GT 经 `clip_to_bev` 恒在窗内(实测越窗 0),
  而 pred **无裁剪** x 达 19.75 / y 达 32.35(窗口 15/30)→ 越窗是模型行为,只计数
  (`out_of_window`)供诊断。硬校验只覆盖结构性不变量(类序/点数/score 域/有限数)

**验收(ep512,留出集 200–299,score_thr 0.2)**:
- 100 个文件落 `outputs/surround_pred/`,**逐帧实例数与跨帧汇聚逐位相等**
  (pred 11508 / gt 7160)——逐帧化没有丢任何实例
- 每帧 `frame` 与 `map_infos.json` 同 token 的帧号一致(帧归属连通性)
- mAP **0.0674** 与 §5.11f 记录一致 → 汇聚口径未被改坏(回归)
- 越窗点 pred 14045 / gt 0(≈6% 的 pred 点;佐证"pred 不裁窗"的平台事实)
- `tests/test_mapvec_schema.py` 20 例(往返/头部拒收/类序守卫/点数/score 域/
  越窗计数/落盘前校验/`NUM_POINTS` 与模型默认值锚定)

**边界与下一步**:本项目只负责"产出 + 契约";消费方(AutoLabel 侧只读 loader /
benchmark / agent 数据集登记)**未接**,按依赖单向纪律在其侧实现,不得反向 import。

**消费侧已接通 ✅(2026-09-15,AutoLabel `mapvec-report`)**:
- AutoLabel 新增对账 CLI(独立工具,零 Web 前端改动):`schema/mapvec.py`(契约消费方
  硬校验副本)+ `schema/mapvec_proj.py`(CAM_FRONT 投影链,复制 mapviz 纯值)+
  `tools/mapvec_compare.py`(chamfer 比对,复制 chamfer_ap)+ `export/mapvec_report.py`
  + `cli.py` 子命令 `mapvec-report`。产物:overlay(品红 pred/青绿 GT)+ BEV 面板 +
  `mapvec_report.md/json`(逐类 AP/CD 中位/最差 5 帧,强制带 score_thr)+
  `{token}_review.json` 复核队列(REVIEW3D_DIR 即见,`review-save` 空 annotations 无副作用)
- 交叉验证测试锁定复制代码不漂移(同一输入 → 与原版逐元素一致);28 例单测全绿
- **验收(ep512 留出集 200–299,score_thr 0.2)**:100 帧比对,渲染段 pred 3589 / gt 2637;
  CD 中位 **0.37m**(p25 0.30 / p75 0.43,418 对);AP divider 0.054 / ped_crossing 0.006 /
  boundary 0.076 / centerline 0.132(与官方口径数量级一致);100 个 review 队列文件
- 复制纪律:AutoLabel 不 import AutoDriveData,纯值逻辑复制(版本锁定注释 +
  `test_mapvec_crosscheck` 对照),依赖仍单向

### 5.11i 600 帧扩数据轮(2026-09-16 ✅,flywheel 第一圈手动原型)

**链路**:布局微对照(官方 108.6/-110.8 vs 旧 235/125,10 帧同镜头 GT 投影段数:
后左 45→29 / 后右 19→57 / 前三无差 → 后向覆盖从"后左偏重"改"后右为主",两布局
360° 无盲区)→ 重采官方布局 400 帧(`surround_p3`)→ merge 旧 0-199 + 新 200-599
(`bin/merge_train_infos.py`:帧号平移只动 meta,data_path 不随帧号重写)→ 图像集中
`maptr_600/images`(600×6 帧)→ 续训。

**训练**:600 帧 × 256 epochs,从 `maptr_ep512.pt` 续训,lr 2e-5/128ep 减半,warmup 3,
batch 5(自适应),epoch 256 完成 **loss 3.57**(ep512 末轮 6.0 起步,扩数据后降到 3.6)。
权重 `outputs/maptr_600.pt`(+ `.opt` 侧车)。

**评估(score_thr 0.2,GPU 后端)**:

| 集合 | ep512(200帧) | **maptr_600(600帧)** | 变化 |
|---|---|---|---|
| 留出集(帧 200-299) | 0.0674 | **0.1607** | **+138%** ✅ |
| 训练集对照(帧 100-199) | 0.2603 | 0.2120 | −19% |
| **泛化间隙** | **3.9×** | **1.32×** | **大幅收窄** ✅ |

四类留出集 AP:divider **0.1559**(4007/2474)/ ped_crossing **0.0241**(982/175)/
boundary **0.1888**(4615/2912)/ centerline **0.2740**(4989/5698)。

**结论**:扩数据(200→600 帧)大幅收窄泛化间隙(3.9×→1.32×),留出集 AP +138% ——
§5.11f"下轮收益靠扩数据"判据成立。训练集对照略降(0.26→0.21)= 模型不再死记训练帧,
与留出集上升**相向而行 = 过拟合减轻的教科书信号**。

**口径诚实声明**:ep512 与 maptr_600 评估数据**布局不同**(旧 235/125 vs 官方
108.6/-110.8),留出集/训练集对照都是各自布局下评估,绝对数差异含布局混杂;但泛化
间隙是同一评估内的相对量,3.9×→1.32× 的量级变化在布局混杂下依然稳健 → 结论可靠。

**下一步**:AutoLabel `mapvec-report` 在 600 帧权重重跑(消费链复核);多图扩数据
(§5.14 Phase 2)。

**环境变更**:GPU 已从 RTX 3080 Ti 12GB 换到 **RTX 4080 SUPER 32GB**(无卡模式换卡机)。

### 5.11j 1000 帧跨图扩数据轮(2026-09-17 ✅,多图验证)

**链路**:布局微对照(官方 108.6/-110.8 vs 旧 235/125,10 帧同镜头 GT 投影段数:
后左 45→29 / 后右 19→57 / 前三无差 → 后向覆盖从"后左偏重"改"后右为主",两布局
360° 无盲区)→ 重采官方布局 400 帧(`surround_p3`)→ merge 旧 0-199 + 新 200-599
(`bin/merge_train_infos.py`:帧号平移只动 meta,data_path 不随帧号重写)→ 图像集中
`maptr_600/images`(600×6 帧)→ 续训。

**Town13 跨图接入**(§5.14 Phase 2 第一枪):
- `collect_surround.py` 加 `--map Town13`(运行时 `load_world` 切图,零副作用)——多图采集基础设施
- Town13 采集 400 帧(@官方布局,0.8 fps)→ `assemble_maptr`(Town13 xodr → infos)
- **切图采集坑**:服务器已在 Target 图时 `load_world` 重复切换 → 60s 超时;改用"缺省
  `--map` = 用当前图"避免。Town13 无 NPC 车流(§5.7d TM 降级纪律)
- **Town13 xodr lane 边界坑**:lane 集合在相邻 lane_section 间变化(road 3279 lane-4
  在 s=2.24 处消失)→ `_sample_boundary` 加 ValueError 截断(标线在 lane 消失处折断),
  不抛异常(§5.11b 同类坑,Town13 触发)
- **crop_to_ego 粗筛优化**:全图 3.3 万矢量 × 400 帧暴力裁剪 23min → 加包围盒粗筛
  `_rough_in_window` → **90s**(窗口外实例直接滤掉,量级 ~15×)
- 组装 1000 帧 infos(600 Town10 + 400 Town13 → 帧号 0-999,data_path 全指
  `maptr_1000/images`,6000 图集中,逐张存在性断言)
  - ⚠️ **订正(2026-09-19)**:`outputs/maptr_1000/`(images 6000 图 + infos,5.5 G)已随磁盘
    清理删除。**权重 `outputs/maptr_1000.pt` 保留**;复现训练需重跑组装链(源图在 `surround_train` 等)

**训练**:1000 帧 × 256 epochs,从 `maptr_600.pt` 续训,lr 2e-5/128ep 减半,warmup 3,
**batch 12**(4080 SUPER 自适应,空闲 24.7GiB×0.95)——600 帧时仅 5。epoch 256 完成
**loss 3.02**(600 帧末轮 3.57 → 再降一档)。权重 `outputs/maptr_1000.pt`。

**评估(score_thr 0.2,GPU 后端)**:

| 集合 | ep512(200帧) | 600帧(Town10) | **1000帧(Town10+Town13)** | 变化 |
|---|---|---|---|---|
| 留出集(帧 200-299) | 0.0674 | 0.1607 | **0.1988** | **+24%** ✅ |

留出集四类:divider **0.1941**(4203/2474)/ ped_crossing **0.0591**(541/175)/
boundary **0.2169**(4650/2912)/ centerline **0.3252**(4996/5698)。

**结论**:跨图扩数据(600→1000)继续收窄泛化(留出集 +24%),**ped_crossing 单类
+146%(0.024→0.059)**——Town13 补进稀疏类样本提升最显著。轨迹 0.0674→0.1607→
**0.1988** = 数据量 + 多样性价值三连验证。§5.14 Phase 2"多图扩数据"判据成立,
Town13/15 可采图池证实可用。

**边界**:留出集仍是单图(Town10 帧 200-299),跨图泛化还没在"Town13 上出预测"验证
——那是验收口径的分支,后续可加 Town13 留出集评估补全。

### 5.12 官方 MapTR/MapQR 栈对照(2026-09-13 定案 → **2026-09-14 终止**,见文末"终止记录")

**缘起**:§5.11d 把老栈降级 optional,条件是"若后续需要官方权重对标再启动"。2026-09-13
用户裁决启动,范围 = **A′(口径复算)+ B(同数据复线)**,三个实现都跑(MapTRv2 4 类 /
MapQR / MapTR v1 3 类),**重训权重必须落项目内**。

**资源实况(2026-09-13 实测,定义了方案边界)**:

| 依赖 | 实况 |
|---|---|
| GitHub / HuggingFace 直连 | ✗ 超时;`ghfast.top` 代理 ✓(`git ls-remote` 两仓库均通) |
| 官方仓库 | **已在项目内** `hdMapGitHub/{MapTR,MapQR}`(`.gitignore:64`,外部资产)。MapTR main `a6872d8` + `origin/maptrv2` `e03f097`;MapQR `d1d9f38` |
| 官方权重 | **不可得**——MapTR 挂 Google Drive(✗)、MapQR 挂 CUHK SharePoint(链接 404);两仓库内均无 ckpt |
| nuScenes | `datasets/nuscenes_mini` 完整(404 关键帧 ×6 相机 @1600×900 + sweeps + v1.0-mini + 4 张地图 png) |
| **map expansion** | **缺** `<mini>/maps/expansion/*.json`(maps/ 只有 png)→ 官方 GT 生成(`NuScenesMap`,nuscenes_map_dataset.py:527)走不通。**已解**:v1 构造期的地图库只需"能构造"(其 `__init__` 硬访问 15 个列表层 + 2 个字典层 + `version>=1.3` + `canvas_edge` 二元列表),352B 空桩即可;运行时该空地图**从不被查询**(v1 用 bridge 换掉 `self.vector_map`)。A0 完整(mini 训练/评测)仍阻塞 |
| 老栈 wheel | torch 1.9.1+cu111 / torchvision 0.10.1 走 **阿里云 pytorch-wheels 镜像**(~4MB/s);**不要用 download.pytorch.org 直链**——实测下到 1.1G/2.04G 后静默停滞(20 分钟 0 字节,不发 EOF);mmcv-full 1.4.0 **预编译** wheel(cu111/torch1.9.0 ✓,免源码编译,0.3MB/s);mmdet 2.14.0 / mmseg 0.14.1 走 aliyun ✓ |
| 编译链 | 系统 CUDA 11.8 ✓(torch cu111 只校验 major 版本)、gcc-9 可 apt(huaweicloud 源)✓、R50 ImageNet 预训练已在 torch hub 缓存 ✓ |

**口径修正(本次探查的重要发现)**:官方 AP 是**按 score 排序 → 累加 tp/fp → PR 曲线积分**
(`map_utils/mean_ap.py:287-306`,chamfer 阈值 [0.5, 1.0, 1.5]),**含 recall 项**;我们的
`chamfer_ap` 是 3 阈值 precision 均值、**无 recall 项** → §5.11f 那条"高阈值天然占优、
0.2→0.4 翻 2.6×"是**我们实现的口径特性,不是官方口径**。A′ 必须量化两者差值并归因
(recall 项 / `pc_range` 裁剪 / score 排序),据此修正本文件内所有 mAP 数字的口径声明。

**阶段 0/数据面踩坑(2026-09-13 实测,都已修入 `bin/setup_maptr_official.sh` / 转换器)**:

1. **`pip install -r mmdet3d/requirements/runtime.txt` 不能整份照装**:里面 `numba==0.48.0`
   + `numpy<1.20.0` 是 mmdet3d 0.x 遗留 pin(该文件自己写着 "we may unlock the verion of numba
   in the future")→ 会把 numpy 降到 1.19.5,而 opencv-python-headless 5.x/shapely 1.8.5 都是
   新 ABI 轮子,静默降级后导入即炸。改为逐项装(lyft_dataset_sdk 是 `mmdet3d/datasets/__init__.py`
   的**导入期硬依赖**;numba 用与 numpy 1.23.5 相容的 0.56.4)
2. **infos pkl 必须是 `{"infos": [...], "metadata": {"version": …}}`** —— 官方
   `NuScenesDataset.load_annotations` 是 `data['infos']` + `data['metadata']['version']`;
   直接 dump 一个 list 会在**数据集构造期**就 TypeError(转换器已改,`--train-frames` 划分照旧)
3. **v1 的 detection GT 是空数组也必须类型正确**:v1 pipeline 有 `LoadAnnotations3D`
   + `ObjectNameFilter` + `use_valid_flag=True`,官方 `get_ann_info` 走
   `gt_boxes[valid_flag]` 掩码索引 → `gt_boxes/gt_names/valid_flag` 必须是**等长 ndarray**
   (我们的转换器已按 (0,7)/(0,)/(0,) bool 写)。真正的 GT 在 `vectormap_pipeline` 里被
   **整体覆写**成地图矢量(`example['gt_labels_3d'] = DC(gt_vecs_label)`),所以
   `filter_empty_gt` 检查的是"本帧有没有地图矢量"——我们有,不会被滤掉
4. **v2/MapQR 侧无需改代码**:离线数据集 `get_data_info` 直吃 `info['annotation']`,
   `map_ann_file` 存在时 `_format_bbox` 跳过 `_format_gt`(即不碰 map expansion)

**阶段与判据**:
- **0 环境**:`/root/autodl-tmp/envs/maptr_official`(py3.8;系统盘仅 8.2G,env 必须落数据盘)。
  判据 = `import mmdet3d` + GKT op 前向跑通 + `outputs/maptr_official/env_check.json`
- **A0 数据链(mini)**:官方 `tools/create_data.py nuscenes --version v1.0-mini` 生成
  `nuscenes_infos_*.pkl`。A0-lite = 作为**我们转换器的格式黄金基准**(逐字段比对);
  A0 完整(含 mini 训练/评测)= 阻塞于 map expansion
- **A′ 口径复算**:ep512 的 100 帧 preds/GT → 官方 `{'GTs':…}` + result json,
  用官方 `eval_map` 复算 → 报出差值与归因
- **B 同数据复线**:三个实现在同样 200 帧训练、留出 100 帧评估;**预算 = 等预算**
  (2026-09-14 定为 **256 ep / 51,200 样本 / 10,240 优化步** = 自实现基线**第一轮**的完整
  口径,样本与步数双等;原定的 512 ep × 200 帧 = 102,400 样本是基线**两轮之和**,已裁,
  理由见下"预算裁剪");v1 只比共享 3 类;`work_dir` 落 `outputs/maptr_official/<impl>/`。
  → **2026-09-14 用户裁决整条中止,本项未执行完**(见文末"终止记录")
- **C 归档**:结论 + 三仓库 commit 号 + `environment.yml` 进项目,复现命令一条不漏

**A′ 执行记录(2026-09-13 ✅)**:`bin/eval_official_metric.py`(项目侧,官方代码只读引用)。
同一份 ep512 逐帧产物(`outputs/surround_pred`,100 留出帧)分别喂我们的口径与官方 `eval_map`:

| 口径 | 20 点 raw | 100 点重采样 | 说明 |
|---|---|---|---|
| 我们 `chamfer_ap`(4 类) | 0.0672 | — | 基线口径 |
| 官方 `eval_map`(4 类) | 0.0586 | **0.0699** | 官方 config 是 `eval_use_same_gt_sample_num_flag=True` → **0.0699 才是参照值** |

- 3 类子集(与 v1/MapQR 可比的口径,`--classes divider,ped_crossing,boundary`):
  我们 0.0456 / 官方 raw 0.0485 / 官方 100 点 0.0557(ped_crossing 三类口径下 AP≈0.0006,
  仅 118 条 GT → 该类的三方对照**不具区分度**,不据此下结论)
- 归因:**差异几乎全部来自 GT 重采样点数**(官方把 GT 插成 100 点,chamfer 距离因此变小),
  而非 recall 项或 score 排序——两口径在同一数据上的类序(centerline 最高、ped 最低)一致
- 复算脚本在 autodrivedata env 可跑:官方 `mean_ap.py` 只依赖 mmcv 的 Timer/dump/print_log
  (已 stub)+ shapely **1.x 语义**(`STRtree.query` 返回几何对象;2.x 返回索引 → 已补 shim)

**补口径说明**:§5.11f 那条"高阈值天然占优、0.2→0.4 翻 2.6×"是**我们实现的口径特性**
(无 recall 项),不是官方口径;跨权重比较仍固定 `--score-thr`,但对外报数须写明口径与点数。

**阶段 B 执行记录(2026-09-13/14 → 2026-09-14 **中止**;保留为"若重启该从哪继续"的完整记录)**:

*账本更正(经产物核实)*:原记"自实现 ep512 × 200 帧 × bs2 = 51,200 步"**为错**。
`outputs/maptr_ep512.pt.opt` 里 AdamW 的 `step` 计数 = **10,242** ⇒ 末轮(256 ep)是
**batch 5**(200/5 = 40 步/epoch × 256 = 10,240,余 2 为自适应探针步),与 §5.11 的
"batch 5,~34s/ep" 互证。可跨轮次对账的量是**样本数** 512 × 200 = **102,400**(与 batch
无关);步数随 batch 变(基线末轮 10,240 步)。三天后仍可核对的判据:每个 checkpoint 里
AdamW 的 step。

*显存边界(实测;卡 12GB,可用 11.63 GiB)*:bs2@1242×375 需 ~13 GiB → **OOM**;
bs1@1242×375 峰值 **8,147 MiB** ✓;bs2@0.5 分配 8,906 / 峰值 11,497 MiB(贴边,不用)。
**并证伪"降分辨率换吞吐"**:0.5 缩放 1.98 s/iter vs 1.0 的 1.75 s/iter —— 成本在 BEV
transformer 不在主干(MSDeformableAttention3D 每 query 每 level 固定采 8 点,与特征图大小
无关;20k query × 6 层 × 6 相机才是主项)→ 账本里"0.5 换 4× 加速"的假设**错误**。

*等有效 batch 的实现(纯配置,官方代码一行未改)*:`samples_per_gpu=1` +
`GradientCumulativeFp16OptimizerHook(cumulative_iters=5)` —— mmcv 的累积 hook 每步先
`loss = loss / loss_factor` 再 backward → 梯度 = 5 个微批的**均值**,与真 bs5 同式(非求和)。
写 `fp16 = None` 仅为绕开官方 train 脚本把 hook 类型**硬编码**成 `Fp16OptimizerHook` 的那条
分支(`fp16_cfg is not None`);fp16 包装仍由 `Fp16OptimizerHook.before_run` 完成。两处已知
偏差:①fp16 后端 torch GradScaler → mmcv LossScaler(同为 static 512 + 溢出跳过);②BN 统计
按微批而非有效批(每通道空间样本 2.4 万+,可忽略)。**自证**:1 epoch 后 checkpoint 里 AdamW
的 step 必须恰为 200/5 = **40**(实测 ✓)。口径 = 512 ep × 200 帧 ÷ 5 = **20,480 步**
(= 基线末轮 10,240 步 × 2 轮)。

*吞吐与预算(bs1,原生 1242×375,实测)*:

| 线 | s/iter | 512 ep(200 帧) |
|---|---|---|
| MapTR v1(3 类) | 0.30–0.36 | ≈ 9.4 h |
| MapQR(3 类,SGQ + GKT-h) | 1.0–1.2 | ≈ 31 h |
| MapTRv2(4 类) | 1.55–2.13 | ≈ 50 h |

合计 ≈ 90 h(3.8 天),单卡串行;**链式顺序 = 便宜的先跑**(v1 → mapqr → maptrv2)。

*链式阻断缺陷(两个,都属"开了在线评测才会炸"型,已在挂链前修掉)*:

1. **v1 桥的同位姿帧**(`maptr_official/bridge.py`):留出集末尾 16 帧采集车停在同一位姿
   → 位姿键完全相同,原守卫一见重复即抛 `ValueError` ⇒ **开验证连数据集都构造不出来**。
   准则改为"位姿查表无歧义的充要条件是**查出来的 GT 一样**":同键帧只要 `annotation` 与
   `map_location` 全等就放行(实测这 16 帧 GT 1e-9 全等;同位姿帧图像不同,但 GT 只由位姿
   决定),有一帧不同才报错
2. **RGBA 图读成 4 通道**(三份 config 的 `LoadMultiViewImageFromFiles`):本 env 的 mmdet3d
   其实是 **MapQR 自带的 vendored 副本**(`hdMapGitHub/MapQR/mmdetection3d`),其 loader 默认
   `color_type='unchanged'`(IMREAD_UNCHANGED),而我们的导出图是 **RGBA PNG** → 读成 4 通道;
   `NormalizeMultiviewImage` 的 mean 只有 3 通道 → `cv2.subtract` 尺寸不匹配。**train 侥幸能跑**
   只因 `PhotoMetricDistortion` 在前(其 `bgr2hsv` 静默吞掉 alpha);`test_pipeline` 没有它 →
   **在线评测一开就炸**(若不修,epoch 64 才发现,白等 63 轮)。修法 = 三份 config 显式钉
   `color_type="color"`:官方 nuScenes 图是 3 通道 JPEG → **3 通道才是官方契约**,且与自实现
   基线(`cv2.imread` 默认口径)**逐像素同源**,alpha 本无信息

*链路脚本*:`bin/run_official.sh` 补齐三件 —— `--resume`(有 `latest.pth` 就接着训)、
`--bg`(`setsid nohup` 自重入,SSH 断线不死)、`chain`(三线串行,**失败即停**;`<impl>.done`
标记让重跑自动跳过已完成的线;每线训完自动补 best/final 两次官方评测 → 链跑完时所有记账
数字都已在盘上)。

*在线评测路径已端到端验证(2026-09-14)*:三条线各跑 1 epoch + `evaluation.interval=1` 探针,
全部 `EXIT=0` —— val 走满 100 帧、`NuscMap_chamfer/mAP` 键存在、`best_NuscMap_chamfer/
mAP_epoch_1.pth` 落盘、官方评测 json 落 `<work_dir>/<时间戳>/pts_bbox/nuscmap_results.json`
(v1 0.0048 / v2 0.0019 / mapqr 0.0003,均为 1 epoch 随机权重,量级合理);显存峰值
v1 5077 / v2 8149 / mapqr 7743 MiB(bs1)。

*独立评测(`tools/test.py`)另有四个坑,已全部封进 `bin/run_official.sh test`*:

1. **单卡分支被官方写死 `assert False`**(三仓库 test.py:225)→ 只能走分布式路径
2. **分布式初始化强制 `spawn`**(`mmcv/runner/dist_utils.py:16`:`init_dist` 在 start method
   未设时 `mp.set_start_method('spawn')`)→ DataLoader 要 pickle 整个 dataset,而官方 dataset
   挂着 `eval_detection_configs`(nuScenes `DetectionConfig`,内含 `dict_keys`)→
   `TypeError: cannot pickle 'dict_keys' object`。**训练期 EvalHook 走 fork,从不暴露**。
   绕法 = `--cfg-options data.workers_per_gpu=0`(不开 worker 进程就不 pickle dataset;
   100 帧顺序读图代价可忽略)+ `RANK/WORLD_SIZE/MASTER_*` 环境变量自举单进程分布式
3. **`jsonfile_prefix='test/...'` 与 `args.tmpdir` 都是相对 cwd** → 从仓库根跑会把 `test/`、
   `.dist_test/` 写进 **pristine 的官方仓库** → cwd 刻意设 work_dir(插件导入靠 PYTHONPATH,
   与 cwd 无关)
4. 官方 `tools/dist_test.sh` 尾巴**硬编码 `--eval bbox`**(argparse 同名参数取最后一次)→
   会把我们要的 chamfer 覆盖掉,故不经它启动

*两条路数字对账*:同一 ckpt(v1 探针的 best)分别走训练期 EvalHook 与独立 `test`,
mAP **逐位一致**(`0.004777971396429671`)→ 独立评测可信,阶段 C 可用它复核任一 ckpt。

*预算裁剪(2026-09-14,用户质疑"90 小时会不会太长"后重算)*:原定每条线 512 ep;实测吞吐
0.303 / 1.07 / 1.44 s/iter ⇒ 三线合计 ≈ **81 h**(比先前估的 90 h 略短 —— 先前按 v2 最坏
2.13 s/iter 估)。裁到 **256 ep**:v1 ≈ 4.3 h + mapqr ≈ 15.2 h + maptrv2 ≈ 20.4 h ≈
**40 h(−50%)**,且**对照强度不降** —— 可比的基线记录本来就是**两轮各 256 ep**(见上文
"账本更正":第一轮 → `maptr_ep256.pt`,留出集 @0.2 **0.0510** / @0.3 0.0952 / @0.4 0.1350
三档齐全;第二轮是 **lr 2e-5** 每 128 ep 减半的**低 lr 续训** → @0.2 0.0674)。官方线
256 ep 与基线**第一轮**在**样本数(51,200)与优化步数(10,240)上双双相等**;而拿 512 ep
对齐恰是拿"低 lr 续训"当参照,平白多一个 schedule 形状(lr 2e-5 vs base 6e-4)的混淆项。
**且基线自身的结论就是"ep256→ep512 只在保守操作点 +32%、高阈值持平略降 → 下轮收益靠
扩数据而非继续长训"**(§5.11f)⇒ 256 ep 已足以给三条官方线定序,512 ep 的边际信息不值
40 h 的 GPU 独占(这 40 h 正是 CARLA 采数据要用的)。**延长路径留着**:每 64 ep 一次的
在线评测会显示曲线,若 256 ep 处仍在陡升,再对**三条线同时**补第二轮 +256(与基线第二轮
同型:低 lr 续训、从 `latest.pth` 续,不重跑),届时才付第二段。

*启动/中断记录(2026-09-14)*:①**00:29:55** 首挂 `MAPTR_EPOCHS=512` 的链,跑 13 min
(9 epoch、**未落任何 checkpoint**)后因上述裁剪手动停,该次 text log 已删(零产物损失);
②**停训练必须连 DataLoader worker 一起收** —— `workers_per_gpu=4` 的 worker 是 fork 出来的,
而 fork 发生在 CUDA 初始化**之后** ⇒ 它们**继承 CUDA 上下文**,父进程被杀后变成 PPID=1 的
孤儿**继续占显存**(nvidia-smi 仍把 5068 MiB 挂在已死的父 PID 名下,`/proc` 里早就没有它了)
→ 显式收掉那 4 个孤儿后才回到 **0 MiB**。**判据:`nvidia-smi` 归零才算停干净,不是"父进程
没了"**;③**01:34:30** 以 256 ep 重挂:`MAPTR_EPOCHS=256 MAPTR_IMG_SCALE=1.0
bash bin/run_official.sh chain --bg`(三份 config 的默认 `MAPTR_EPOCHS` 同步改成 256,
免得日后手跑时静默偏离账本)。顺序仍 **v1 → mapqr → maptrv2**(便宜的先跑),每 64 ep 一次
在线评测 + 存盘,每线训完自动补 best/final 两次独立评测,**任一失败即停**并留 `.done`
标记(重跑 chain 自动跳过已完成的线,当前线靠 `latest.pth` 续训)。进度总览
`outputs/maptr_official/logs/chain_chain_boot_<TS>.log`,各线明细 `<impl>_{train,test}_*_<TS>.log`。

**纪律**:`hdMapGitHub/` 保持 pristine(在 .gitignore 内,改了什么 git 也看不见)——转换器/
配置/脚本一律写项目内受版本控制的位置,配置用绝对路径写 `_base_` / `plugin_dir`,不往仓库塞文件;
官方仓库不改逻辑。产出(权重/日志/评测 json/可视化)一律 `outputs/maptr_official/`。

**终止记录(2026-09-14 02:10,用户裁决)**:

- **决策**:用户裁定 41 h 仍过长 → **整条官方复线中止**。既不做 256 ep、也不做第二轮延长;
  停顿点 = 重挂后跑到 **epoch 29 / 256**(02:05:59),`checkpoint_config.interval=64`
  ⇒ **未落任何 checkpoint**,不存在"半成品权重"需要处置
- **已删**(共回收 **15 GB**,磁盘 `30G → 45G` 可用):
  ① 官方环境 `/root/autodl-tmp/envs/maptr_official`(5.8 G);
  ② 官方栈产物 `outputs/maptr_official/` 整目录(11 G:探针 work_dir 的 432 MB×N 权重
  ≈7.5 G、`wheels/` 2.0 G、`pip-cache/` 0.44 G、`ckpts/` 98 M、A′ 复算产物 332 M、
  infos 数据集 20 M、日志 1.1 M)。**数据目录是零风险删除**:其 20 M 全是 infos/map 桩,
  图像按路径引用 `outputs/surround_train`(符号链接 0 个、拷贝 0 份)
  - ⚠️ **订正(2026-09-19)**:上述 ① 当日**实际未执行**——`envs/maptr_official` 目录一直在
    (5.8 G,py3.8 + torch1.9.1+cu111 + mmcv1.4.0 仍可运行),故当日真实回收为 **~9 GB 而非 15 GB**。
    该目录已于 **2026-09-19 补删**(磁盘清理脚本阶段 3),至此 ①② 才全部落地。
    教训:删除动作要**当场 `test -e` 复核**,不能只凭执行意图写归档
- **未删(刻意保留)**:① 自实现线全部资产(`outputs/surround_train` 1800 图、
  `surround_pred`、`maptr_ep256.pt`、`maptr_ep512.pt` + `.opt` 侧车)——本次删除**不触碰**
  自实现任何产物;② `hdMapGitHub/` 三个官方仓库(583 M,§5.11 起的既有外部资产);
  ③ 项目源码层:`maptr_official/`(configs + bridge)、`bin/run_official.sh`、
  `bin/setup_maptr_official.sh`、`bin/prepare_official_dataset.py`、
  `bin/eval_official_metric.py`;**待用户裁决**(见下)
- **残留(未处理,刻意不动系统层)**:搭建时 `apt-get install gcc-9 g++-9`(与保留项
  CUDA 11.8 配套,重装一条命令即可:`apt-get install -y gcc-9 g++-9`);conda 包缓存
  `/root/miniconda3/pkgs` 2.2 G 为**各 env 共享**,未清(如需回收用 `conda clean`)。
  `/etc/pip.conf`(8 月 4 日,先于本项目)、base env 的 torch 2.13/mmdet 3.2/nuscenes-devkit
  **均非本次所装**(搭建脚本只写 `ENV=...` 与 env 内 site-packages),不动
- **知识不随产物消失**:本次两个真 bug(RGBA→4 通道炸在线评测、v1 同位姿帧)与其修法都已在
  项目源码与本文档内;A′ 口径复算结论(0.0586 raw / **0.0699** 100 点、差异来自 GT 重采样
  点数而非 recall 项)在 §5.12 上表;重建路径 = `bin/setup_maptr_official.sh all` + 本节各条
- **对结论的影响**:表 C 的"官方实现 vs 自实现"三方对照**不存在**,A′ 的**口径**结论
  (两套 mAP 口径的关系)仍成立且已固化。**自实现线不受影响**——§5.11 的 ep512 权重、
  逐帧契约 `mapvec_pred/1`、实时 overlay 全部照旧可用

**终止后探针:实现对齐验证(2026-09-14 ✅,用户裁决重启,非整条复线)**:

终止后重开一条**轻量 v1 探针**,目的 = 验证我们的数据/转换/评测链与官方实现**对齐**,
不重启完整对照。协议仍钉 §5.12 四钉子(config `color_type="color"`、bridge 同位姿放行、
独立评测四坑、产物落 `outputs/maptr_official/`)。口径:**128 ep / bs2×accum2(有效 bs4)/
online eval / seed 0 / IMG_SCALE 1.0**。起点 2026-09-14 16:37,18:41 训完(≈2 h)。

在线评测曲线(官方 `NuscMap_chamfer/mAP`,3 类 = divider/ped_crossing/boundary):

| epoch | divider_AP | ped_crossing_AP | boundary_AP | mAP |
|---|---|---|---|---|
| 32 | 0.0701 | 0.0010 | 0.0397 | 0.0370 |
| 64 | 0.0537 | 0.0581 | 0.0694 | 0.0604 |
| 96 | 0.0594 | 0.0537 | 0.0854 | **0.0662**(best) |
| 128 | 0.0574 | 0.0591 | 0.0785 | 0.0650(final) |

- **判定(±2× 锚点)**:A′ 锚点 = 官方 eval_map 3 类 100 点 **0.0557**(§5.12 A′ 表)。best
  0.0662 / final 0.0650 均在锚点 ±2× 范围内(比值 1.19 / 1.17)→ **实现对齐成立**。
  绝对值低 = 200 帧数据规模限制(§5.11e 同款信号),不是链路缺陷
- **收敛形状**:96 见顶、128 小幅回落(0.0662 → 0.0650,~0.065 处饱和);尾部 lr 已衰减到
  ~7e-7(128 ep),不涨反微落 —— 与 §5.11 基线"ep256→ep512 只 +32% 且高阈值持平"同一信号:
  **下轮收益靠扩数据,不靠长训**
- **账本闭合(AdamW step 从 ckpt 直读)**:epoch 64 = 3200 / 96 = 4800 / 128 = 6400,恰为
  50 步/epoch × epoch(200 帧 ÷ 有效 bs4 = 50 iter/ep)→ bs2×accum2 协议与记账一致;
  与基线 `maptr_ep512.pt.opt`(step=10,242)"step 才是可对账量"互证(§5.12 账本更正)
- **产物**:权重 `outputs/maptr_official/maptr_v1/epoch_{64,96,128}.pth`(432 M 各一)+
  `latest.pth → epoch_128.pth`;训练日志 `outputs/maptr_official/logs/maptr_v1_train_20260914_163751.log`

### 5.13 CARLA 雷达 → 真实 ars408 口径(L3 物理合理性,2026-09-14 ✅)

**目标**:5 雷达(sensor.other.radar)接入 nuScenes 输出的点分布要像真实大陆 ars408
(水平 FOV 77° / 垂直 14.2° / range 250m / ~3300pps),L0(格式)与 L1(devkit 直读)
已过,本小节补 L3。

**核心发现:CARLA 0.9.16 两 FOV 属性交叉使用**(prebuilt 编译行为,无源码可改)。
12 组属性扫描自洽,一行映射:

```
azi 半角(水平) = vertical_fov / 2
alt 半角(垂直) = horizontal_fov / 2
```

设 `hf=77, vf=14.2` → 实测 azi±7° / alt±38°(与 ars408 完全反了);对调设
`hf=14.2, vf=77` → 实测 azi±38.1° / alt±7.0° = 真实 ars408 ✓。

**12 组属性扫描证据**(实测 `azi/alt` 半角,`hf`/`vf` 为蓝图属性值,度):

| hf | vf | azi 半角 | alt 半角 | 解读 |
|---|---|---|---|---|
| 77 | 14.2 | ±7.0 | ±38.4 | 当前参数,**交叉使用,完全反了** |
| 77 | 77 | ±38.1 | ±38.4 | 双维同值时"交叉"不可见(hf 生效) |
| 60 | 60 | ±29.6 | ±29.9 | 同上,线性缩放 |
| 90 | 90 | ±44.5 | ±44.9 | 同上 |
| 120 | 120 | ±59.5 | ±59.9 | 同上,线性可达 ±60° 以上 |
| 100 | 100 | ±49.5 | ±49.9 | 同上 |
| 100 | 14.2 | ±7.0 | ±49.9 | hf 越大 alt 越大,vf 锁死 azi |
| 77 | 40 | ±19.7 | ±38.4 | vf 越大 azi 越大(hf 锁定 alt) |
| 40 | 14.2 | ±7.0 | ±19.9 | hf 线性控制 alt |
| 77 | 14 | ±6.9 | ±38.4 | vf 控制 azi 线性成立 |
| 80 | 14.2 | ±7.0 | ±39.9 | hf 线性控制 alt 成立 |
| 70 | 14.2 | ±7.0 | ±34.9 | 同上 |
| **14.2** | **77** | **±38.1** | **±7.0** | **对调后 = 真实 ars408 ✓** |

全部 12 组 + 换向验证组(13 组)都满足一条映射 `azi=vf/2、alt=hf/2`,无一例外;
`hf/hf` 对角组合线性放大到 ±60° 也成立 → 不是"hf 不生效",是 **hf 与 vf 交叉**。

**解法(零源码、零后处理)**:`bin/collect_nus.py` RADAR_ATTRS 两值对调 +
垂直锥裁剪(见下)。prebuilt 无法重编译 `ARadar::SendLineTraces`,故不改源码;
Explore 代理确认本机 prebuilt-only。

**垂直锥裁剪**:换向后 CARLA 布点 alt ∈ ±7.1°(物理对),但实测偶发跑出锥外的
干净点(不触发 |alt|>90 野值剥离判据)→ 写 pcd 前统一按 ars408 锥角裁剪。
判据 = 锥角而非绝对 z:**18 字段无 alt 列,用 |z| ≤ sin(7.1°)·depth**(等价
|sin(alt)| ≤ sin(7.1°));反体素化成绝对 z 阈值会在远距放宽(250m 处 ±31m,几乎
全放行)= 错误。实现 = `autodrivedata/radar.py`:`ARS408_VFOV_HALF_DEG=7.1`、
`mask_in_ars408_vfov`(别名到 `_impl`)、组合 `mask_radar_points` = devkit 默认
过滤器(`valid_mask_nus`) ∩ 垂直锥。空点云(全 NaN)两掩码都 False → 空 pcd 正确。

**L3 探针**(`bin/probe_radar_l3.py`,一次性)实测 20 帧 vs ars408 规格:

| 指标 | ars408 | 实测 | 通过 |
|---|---|---|---|
| 水平半角(°) | ±38.5 | ±38.4(极值) | ✓ |
| 垂直半角(°) | ±7.1 | ±7.1(极值) | ✓ |
| range(m) | 250 | ≤229 | ✓ |
| 帧点数 | 3300pps/10Hz≈330 | 两档 330/265 | ✓ |
| 深度 p5/p50/p95(m) | - | 1/6/74 | ✓ |
| 地面/天空占比 | ≈0 | 0.00% | ✓ |
| 前方 12m 车框内点 | >0 | 153 | ✓ |

探针口径修正三处(记录以避复踩):① 锥角用**极值**而非 99.5 分位——稀疏布点
令分位系统性偏低,曾把 ±38.1° 测成 ±36.5° 误报;② 帧点数判据放宽到 200–340
(CARLA 每 tick 射线预算在两档 330/265 间抖动,与场景负载相关),只拦异常稀疏/
超量;③ 18 字段锥角判据与 raw alt 判据逐点一致率 100% = 反体素化没引入偏差。

**冒烟复跑**(`bin/smoke_radar_collect.sh outputs/nus_mini_l3`):四判据全过。
裁剪后各通道单帧 249–330 点、方向自证 1.00/1.00/1.00、GT 关联 186 注解中 2 框有
radar 命中(近前车框,框内点更纯)。回归 `python -m pytest tests/ -q` = 330 passed /
3 skipped(radar 单测 13 项含锥内/锥外/边界/远距不放宽/空云/组合掩码)。

**遗留**:帧 0 偶发相位空 tick 写空 pcd(同步 sensor_tick=0.1 与 tick 同周期,
drain 已丢弃,`collect_nus.py` C22 既有行为,不影响 devkit 消费)。水平方向未做
±38.5° 裁剪(实测干净数据从不越界,0 个锥外点)。

### 5.14 长期工作目标:合成数据驱动的数据闭环(2026-09-16,工业界对照定案)

**缘起**:用户要求把 Bosch / Momenta / 地平线等自动驾驶公司的数据方法论(影子模式 /
自动标注 / 场景库 / 数据闭环)对照本项目,固化为**长期工作目标**与**分步路线**。

#### 5.14a 工业界方法论对照(2026-09-16 调研)

| 环节 | 工业做法(Bosch/Momenta/地平线等) | 本项目现状 | 缺口 |
|---|---|---|---|
| 采集 | 影子模式(量产车边侧难例筛选回传)/ 定向路采 / 云端仿真(百万公里/天) | 全合成定向采集(600 帧 MapTR / P1 A/B / 17 图池) | 无边侧难例判定;场景靠人工挑选 |
| GT | 自动标注(多帧融合重建)+ 人工只审边界例;自训练(teacher-student 伪标签) | 自动 GT 全覆盖(xodr 地图查询 0 误差 / actor API / oracle) | 无人工复核环(合成域可接受,真实域必需);无伪标签回灌 |
| 长尾 | 场景库参数化 + 数据驱动挖掘(从量产数据自动挖 corner case)+ 每日回归套件 | P1 三 corner case 定量 A/B + 参数档库 + 固定 score_thr 评估口径 | 挖掘是常识驱动(P1-6 候选 = 人想),非数据驱动;回归未自动化 |
| 闭环 | flywheel 基础设施(发现失败 → 挖掘 → 回灌 → 重训 → 评测 → 上线,每晚 CI) | 手动闭环(600 帧 → 留出集 → 人工决定扩数据) | 无自动编排;扩数据决策靠人工判断 |

**核心诚实结论**:A/B 帧级配对纪律、自动 GT、逐帧契约、固定阈值评估口径在工业界反而是
多数团队做不齐的(受控实验纪律);真正的缺口是**数据驱动挖掘**——用大规模数据告诉你
"该补什么场景",而不是人猜。

#### 5.14b 长期目标(定位)

把本项目从"手动工作流"升级为**合成数据驱动的迷你数据闭环**,方法论对齐工业界。
边界如实声明:**100% 合成 → 仿真保真度是全部可信度来源**(与 M4 / CARLA 0.10 迁移
决策联动,见 §5.14d)。

#### 5.14c 分步路线(Phase 1 → 3)

**Phase 1(近期):把现有手动闭环固化为可重复测量基准**

- **P1.1 回归套件**:固定场景集 × 固定 `--score-thr 0.2` 的"一键回归"——聚合
  2D/3D/mapvec AP + 复核率 → 单份报告。各条 eval 链已存在,补一个聚合入口
- **P1.2 场景库参数化**:P1-2 参数档库扩成网格(天气 × 光照 × 速度 × 车流密度),
  补 P1-6 候选(wet_road 眩光 / dense_rush 遮挡)→ corner case 矩阵从 3 型扩到 5 型
- **P1.3 口径冻结**:逐帧契约 + score_thr 已成对外口径(✅ 已有),纳入回归套件

验收:一条命令出一份跨 2D/3D/mapvec 回归报告;corner case 定量 A/B ≥ 5 型。

**Phase 2(中期):场景挖掘数据化 —— 从"人猜"到"数据告诉你该补什么"**

- **P2.1 探测器引导挖掘**:当前最强权重在 17 图地图池 × 参数网格上批量预测,按
  漏检率/AP 热力图自动排序场景组合 → 替代人工挑 P1-6(直接回答"下一个 corner case
  该做哪个")
- **P2.2 长尾注入管线**:挖掘结果 → 采集参数(weather/光照/NPC 布局)→ collect_drive
  批量采集;参数与数据一一可溯源
- **P2.3 难例边侧判定器原型**:纯值函数(低置信度 / TTC 冲突 / 罕见场景组合)给每帧打
  "该不该收"标记——对应影子模式的边侧筛选,合成侧用在线过滤器模拟

验收:自动产出 ≥1 个"当前模型最弱场景"并完成采集 + 重训,AP 提升可复现。

**Phase 3(远期):闭环自动化 + 真实性增强**

- **P3.1 flywheel 编排**:发现失败 → 挖掘 → 采集 → 重训 → 回归 → 上线,脚本链(或 CI)
  串联,跑完出一份"闭环报告"(含扩数据前后泛化间隙变化——把现在 600 帧扩数据的
  人工判断自动化)
- **P3.2 伪标签回灌(teacher-student)**:AutoLabel 3D 检测在合成数据微调,伪标签进
  MapTR/2D 训练(对应工业自训练;§5.11h 契约已就绪,消费侧已接 mapvec-report)
- **P3.3 真实性阶梯**:按"保真度 vs 成本"阶梯升级仿真(CARLA 0.10/UE5 + 嘉定地图,
  Lumen/Nanite 显存饥饿;或真实数据锚点)——**只在 Phase 2 判据触发时动**(如"挖掘
  发现某类场景仿真做不了")
- **P3.4 真实数据接入(可选)**:少量真实帧作分布锚点,与合成数据混合训练——从
  "100% 合成"升级为工业主流形态(真实为主 + 合成为补充)的前提

#### 5.14d 决策联动

- 嘉定 / CARLA 0.10(§7 候选):归 **Phase 3.3 真实性阶梯**,不提前;GPU 升级(4090/24GB)
  只在真上 0.10 时是刚需,当前 600 帧训练不需要
- 当前 600 帧扩数据:是 **P1.1 回归套件的首批输入**,也是 flywheel 的第一圈**手动原型**
- 与 §5.12 官方复线:已终止,本路线**不需要官方权重**,自实现线全资产可直接用

#### 5.14e 岗位 JD 有益项筛选(2026-09-16:什么值得做、什么故意不碰)

**缘起**:对照某数据挖掘岗 JD(海量数据挖掘/图·生成·迁移·强化·多模态/预测·规划·仿真·
数据闭环/向量库检索·图文检索/工具链平台)评估本项目差距后,再按"能否解决我们真实
痛苦"过滤一遍——**不为适配岗位,只为项目本身**。

**✅ 立刻立项:难例挖掘原型(§5.14 Phase 2 提前)——三件同源的事**

1. **探测引导采集(价值最大)**:痛点不是"数据不够"而是"不知道下一批采什么"(600 帧
   里大量近冗余直路帧,留出集 16 帧同位姿即证据)。最小形态 = 当前最强权重在"廉价
   生成池"(参数网格 × 地图池)上批量预测 → 按漏检率/失败置信区间排序场景组合 →
   只采前 N → 重训 → 冒烟 AP 对比。一次实验即验证(弱场景 200 帧 vs 随机 200 帧),
   不新增基础设施(eval 链 + 批量采集全有)
2. **Embedding 检索/样本召回(轻量 FAISS + 现成 encoder)**:合成数据生成便宜、筛选
   昂贵;采集易、判断"哪些帧值得留"难。CPU FAISS + 现成特征(ImageNet/中途特征)三个
   立刻动作 = **去重**(省训练时间)、**多样性覆盖检查**(检索"哪区域无帧"→ 告诉采集器
   去哪)、**复核队列增强**(按相似度召回同族失败帧)。与 #1 同根:#1 决定"采哪"、#2 决定
   "采完怎么挑"——合起来就是缺的"样本召回"。1 万帧量级 CPU 可扛,不碰可扩展性
3. **规则挖掘自动化(把结论变资产)**:已人工总结两条规则(尺度断崖 <32px、天气前移断崖)
   但是一次性观察。把 eval_attr 特征表喂简单决策树/规则学习器 → 自动"漏检判定规则"→
   直接当 **Phase 2.3 边侧判定器**原型(采集在线打"该不该收"标记)+ 新实验自动出失败画像。
   小脚本,成本近零,把"归因能力"从报告升级为可用资产

验收 = 一次完整闭环(采弱场景 → 重训 → 留出集 AP 提升可复现)。

**🔜 未来有用,现在做是分心**

- **图学习(lane/scene graph)**:xodr 图结构(road/lane/link)已解析在手(§5.11 A 阶段),
  却只当几何用;做 lane 连通图/场景关系图是前瞻方向,但属研究非工程需求,等 Phase 2
  原型跑通再谈
- **参数空间覆盖引导(生成式的轻量版)**:用不确定性/覆盖度评分引导参数搜索(weather ×
  布局 × 图池网格自动探索),比 #1 只多一层评分函数;**先做 #1 即可**,diffusion 造场景
  属远期(等"挖掘发现某类场景仿真做不了"触发——§5.14 P3.3)

**🚫 明确不碰(伪需求)**

- **RL 对抗场景生成**:训练循环重资产;人工 corner case 在 5 场景内仍可控,为其付一个
  RL 项目换"自动造冲突",ROI 负
- **多模态理解/文搜图·图搜图**:产物语义由 GT + 场景参数完全可控,不需要"看图说话"层;
  等真实数据/开放域需求再说
- **分布式/大规模平台**:1 万帧量级 CPU FAISS 就够,P3.1 flywheel 用脚本链即可;企业级
  平台是被数据量逼到那步才有意义

**一句话**:值得正式立项的只有一件——把 §5.14 Phase 2 提前成"难例挖掘原型"
(探测引导为主、embedding 召回为辅、规则学习当边侧判定器);其余按上面三档各安其位。

### 5.15 轨迹预测对标:HiVT 复现(2026-09-16 ✅,补"预测"能力面)

> **📦 已迁出(2026-09-19)**:完整执行记录(评测结果表 / 环境沉淀 / 四个踩坑 / 对项目的意义)
> 见 **[Plan2.md](Plan2.md) §9.1**。此处仅留标题作历史索引,内容不再维护。

### 5.11a A1 执行记录(2026-09-10 ✅)

**交付**:`autodrivedata/opendrive.py`(纯值,stdlib+numpy)+ `tests/test_opendrive.py`(35 例)。

- 解析:planView(line/arc/spiral/poly3/paramPoly3)+ elevationProfile + lanes
  (laneOffset/width/roadMark/link)+ objects(outline 4 角)+ signals(validity)+
  junction(connection/laneLink);核心 `road_to_xy(s, t)`(闭式解;spiral Simpson N=128)
- **实测口径修正**(比 §5.11 摸底更细):
  - 全库几何类型总计 = line **54460** / arc **33841** / spiral **262**(仅 Town15)——§5.11 摸底
    的 65570/39301 是 glob `**/` 把顶层 OpenDrive 目录双计了,以本行为准
  - 车道 id 惯例 = **左正右负**(与 OpenDRIVE 标准相反,与 CARLA 运行时 lane_id 同向)
  - signal 在 `<signals>` 包裹内(曾 `findall("signal")` 直取 → 0 条,已修)
  - Town10HD_Opt 锚点:road 108 / roadMark 2802 / crosswalk 16 / signal 21 / object 60
    (16 crosswalk + 44 路面花纹)
- 测试:闭式解手算锚定(line/arc 圆方程/螺旋退化=arc/细网格独立积分对照)+
  真实文件计数锚点(逐项与 grep 复核一致,20 文件全解析;无 CARLA 机器自动 skip)

### 5.11b A2+A3+A6 执行记录(2026-09-11 ✅)

**交付**:`autodrivedata/mapvec.py`(纯值,六类提取/重采样/裁剪)+ `tests/test_mapvec.py`(11 例)+
`bin/probe_mapvec_oracle.py`(A6 API oracle 探针)。

- **要素提取口径**(xodr 实测修正,比 §5.11 摸底更细):
  - `divider` = driving-driving 共享边缘标记(**标记附着于车道外侧边缘**,统计证明:23 例
    broken/solid 模式 + curb 位于 sidewalk/shoulder 上)∪ center lane(id=0)标记;center lane
    标记 = 双向道路中心线(solid solid yellow 对向分隔,attrs `centerline=yes, same_dir=no`)
  - `boundary` = curb ∪ 最外侧实线 ∪ 最外侧兜底(沿 t 区间排序共享端点 ε=1cm 合边)
  - `stop_line` = `<object name="StopLine">` outline(2/3/15 点,沿 hdg 的线段);`ped_crossing`
    = outline 4 角 + 闭合(5 点);signal 类型为**数字编码** `"1000001"`(='traffic_light',
    Town10HD_Opt 17 个),字符串 `"traffic_light"` 匹配会漏光
  - Town10HD_Opt 计数锚点:ped_crossing 16 / stop_line 21 / traffic_light 15–21 /
    center_divider 106(586 段同属性相邻合并)
- **裁剪与重采样**:`crop_to_ego(pose, ±51.2m)` 用 **Liang-Barsky**(修掉"两端点在窗外
  但线段穿窗"的漏段 + V 形折线分裂为多段);`resample(v, n=20)` 弧长等距,traffic_light
  定点类原样返回
- **A6 oracle 对账**(`probe_mapvec_oracle.py --map Town10HD_Opt --samples 200`):
  - **CARLA 世界 = xodr 的 y 取反**(Unreal 左手系;镜像后 3D 位置误差 **0.00cm**,
    184 组全通过,验收口径 <5cm ✅)
  - **yaw 仅诊断、不定验收口径**:个别 road 的行驶方向被 CARLA 导入器按路网拓扑整体翻转
    (实测 road 1 正负 lane 与 +s 的关系和 road 0/2 相反;xodr 侧 link 无一致判据)。
    MapTR 矢量 GT 是**无向几何**折线,行驶方向非产物需求;若 B 阶段需要车道方向,
    从 CARLA 运行时 API 拿
- **A4 导出**(`bin/export_mapvec.py`):`training/map/{map}_full.json`(全精度)+
  `{map}_{fid}.json`(ego ±51.2m 裁剪后 20 点重采样)+ 两档 BEV overlay PNG。
  **落盘坐标系 = CARLA 世界系**(xodr y 取反,`meta.frame="carla_world"` 标注),与
  B 阶段采集 ego pose 同系;`mapvec.py` 增 `flip_y/to_carla/vecs_dump/vecs_load`
  纯值 JSON 往返(attrs 保持有序列表,validity 多车道对不丢)。Town10HD_Opt 实测
  816 实例(799 折线),帧级裁剪 332 实例;resample 零长线段 0/0 除零已修(np.divide
  where 掩码)。目检:路口/斑马线/停止线/红绿灯空间关系符合交通拓扑,C23 撞色口径
  避让(divider 橙 / stop_line 红 / ped 绿 / boundary 蓝)
- **A5 转换器**(`bin/convert_mapvec.py`):帧级 JSON → MapTRv2 annotation 口径
  (与官方 `custom_nusc_map_converter.VectorizedLocalMap` 同构:四类
  divider/ped_crossing/boundary/centerline 的 N×2 折线,**ego 局部系** =
  rotate(-yaw) 后平移,z 丢弃;训练管线再 resample 20 + 归一化 [-1,1])。
  stop_line/traffic_light 工程补充类不进训练口径(留在 full json)。`mapvec.py`
  增 `to_ego_frame/from_ego_frame`(往返断言)+ `to_maptr_annotation`。
  Town10HD_Opt@spawn0 实测:divider 72 / ped 7 / boundary 82 / centerline 163,
  局部坐标 ±51.3m 与裁剪窗口吻合,ped 闭合 20 点
- **A6 三件套全过** ✅:①几何自证(ped 未闭合 0 / 折线自交 0 / 车道线类回折 0;
  回折 10 例全在 ped 闭合多边形角点,属地图作者几何)②API oracle 0.00cm ③overlay
  目检(路网/路口/斑马线/停止线/灯空间关系正确)。**第 2 步(A1–A6)完成**,提交
  b16d2bd / dd8aa5c / 80821dd / 0a66da6(本小节为合并记录)

### 5.11c B 阶段执行记录(2026-09-11 ✅,第 3 步)

- **B1 环视采集**(`bin/collect_surround.py`):6 相机 nuScenes 布局(FRONT 0 /
  FRONT_RIGHT -55 / FRONT_LEFT +55 / BACK 180 / BACK_LEFT 235 / BACK_RIGHT 125,
  共用挂点 SENSOR_OFFSET),落盘 6 视角 png + calib.json(sensor2ego + intrinsic)
  + ego_pose.json。**不改 A/B 采集器**(P1 复现红线),NPC 布置复用 collect_drive。
  smoke 20 帧 × 6 视角完整。**FPS 实测 0.6**(单相机 47.3、6 相机 1.7s/帧,非线性
  回读瓶颈)——§5.11 风险① 探明:100 帧 ≈ 3 分钟,可接受,不降分辨率
- **B2 组装器**(`bin/assemble_maptr.py`):surround root + 全图矢量 → MapTRv2 infos
  同构 json(逐帧 cams + ego2global + annotation 四类 ego 局部系)。口径注记:
  MapTRv2 官方 annotation 在 **LiDAR 局部系**,本管道无 LiDAR → **ego 局部系**
  (lidar2ego 恒等,训练消费端无差别)。20 帧 smoke 首帧 annotation 与 A5 完全一致
  (divider 72 / ped 7 / boundary 82 / centerline 163,交叉验证通过)
- **B3 验收**(`bin/probe_mapvec_proj.py`):矢量折线点投影回 6 视角图像,数值诊断
  (不做视觉回归):图像内投影占比 12.8–30.1%(环视 fov 90 合理),**路面性
  94.7–100%**(投影点像素非天空比例)——内外参 + 坐标系链正确的最强实证。
  **第 3 步(B1–B3)完成**,提交 fda2286(本小节合并记录 B2/B3 提交)

### 5.6 测试环境策略(已定)

- **纯数学单测**:autodrivedata env(手算断言,不依赖 carla 与 auto3dlabel)
- **oracle 对比脚本**:autolabel env 跑,直接 import auto3dlabel 的 geometry/data 模块当单一事实源(双向转换往返断言)
- 两边互不污染,遵循 AutoLabel"依赖方向单向"纪律

## 6. 项目骨架

```
AutoDriveData/
├── Plan.md                # 本文件(单一事实源)
├── pyproject.toml         # autodrivedata 包:纯逻辑,numpy only
├── autodrivedata/         # 包:不 import carla(可在任何 env 测试)
│   ├── geometry.py        # 坐标转换唯一落点(照 auto3dlabel 纪律)
│   ├── calib.py           # 内参/外参 → KITTI calib txt
│   ├── gt.py              # actor → label_2 行
│   └── export/kitti.py    # KITTI 布局落盘
├── bin/               # 采集入口(依赖 pycarla,autodrivedata env 跑)
│   └── smoke.py           # ✅ M0;collect_kitti.py 待建(步骤 6)
├── tests/                 # 单测(autodrivedata)+ oracle 脚本(autolabel env)
└── outputs/               # 数据落盘(不进 git)
```

## 7. 待办/依赖(2026-09-07 更新;⚠️ **后续待办一律在 Plan2.md §3/§8 维护,本表冻结**)

- [x] CARLA 版本选型 0.9.16;下载/安装/headless 适配/pycarla/隔离验证(M0 ✅)
- [x] `resolve_frame`/frame_id 规则与 `KITTI_OBJECT_ROOT` 覆盖(§3.1)
- [x] 磁盘扩容(190G,余 76G)
- [x] git init + 骨架 + 决策(§5 三决策)
- [x] M1a 步骤 1–7(geometry/calib/gt/export/collect/集成验收)✅
- [x] M1b 步骤 8–10(geometry P2 约定/nuscenes 生成器/oracle/nuscenes-queue 验收)✅
- [x] M2(collect_drive/compare/eval_kitti/域差距修复实验/150 帧闭环验收)✅——分歧率 <10% 未达,微调路径归 M3
- [x] M3 工具链(train3d 微调 3 败排查/行人布置/headless 纪律)✅——微调专项挂起,天气·长尾转 P1
- [x] **P1** corner case 场景矩阵:参数档库 + 逆光 A/B 定量(§5.5a ✅ 2026-09-09)——相机 Δ-0.020 掉点成立、LiDAR Δ+0.003 兜底不受影响
- [x] **P2** 静态目标 + 道路特征 GT:选型(§5.6a 地图查询定案)+ 格式 + 样例(§5.6b ✅ 2026-09-09,目检通过)
- [ ] **P1-4+** 第二 corner case 定量 A/B(雨夜;隧道/遮挡候选)——M4 降级后新主线
- [~] **M4** 定制街道:M4-0 调研(§5.7 ✅)+ M4-1 用户裁决降级(§5.7a,源码构建成本过载,挂起重开条件:换盘/有需求)
- [x] **地图池扩展**(§5.7d ✅ 2026-09-09):AdditionalMaps Town11/12/13/15 入池(17 图);约束:Town11/12 禁采集(spawn camera segfault)、Town13 TM 车流降级、锚定 yaw bug 已修(spawn point 固有 rotation)
- [x] **可视化实时流**(§5.8 ✅ 2026-09-09):自建 MJPEG(view_stream.py,3 视角 + GT overlay + 灯色);carlaviz/RViz2 出局(非 UE 渲染 + 版本/依赖不成立);`world_to_img` 上移 calib.py 共用
- [x] **灯色动态 GT**(§5.9 ✅ 2026-09-09):traffic_light.py 纯值层 + collect_tl_states.py(记录/受控切灯)+ 前向过滤(修掉 79% 身后灯)+ 渲染探针实证(镜片 30m 处仅 4px,不做视觉回归)
- [x] 工程规范(2026-09-09):`[tool.ruff]` 定死(110 列 / E,F,I,UP,B / ignore E501,E741)+ 存量 25 违规清零 + 全仓 `ruff format`(26 文件 419 行),单 `style:` 提交 fc9f592 + `.git-blame-ignore-revs`;pre-commit 未装 → 不引入,纪律落到 CLAUDE.md 命令行
- [x] 参数扫描 + 失效归因(§5.10 ✅ 2026-09-09):距离×速度网格 + 逐帧漏检归因;三大结论 = 尺度主导(<32px 0.15-0.47 vs ≥32px 0.78-1.00)、CARLA 无运动模糊(速度不改图像)、天气只前移断崖;顺带修掉 collect_ab_route 的 brake 残留(老数据集实速 6.60 而非 8.0)
- [ ] **P1-6 候选**:wet_road 眩光 / dense_rush 遮挡(待用户定)
- [x] **环境迁移**(§4.4,2026-09-10 用户拍板):项目 env base(3.10)→ autodrivedata(3.11.16,pycarla/ultralytics 全量迁入);requirements.txt 钉版本;direnv + .envrc 自动激活;base 仅剩 conda 底座;验收 = 213 单测 + 采集冒烟;env 迁数据盘(软链)避开系统盘
- [x] **scripts→bin 改名**(第 1 步 ✅):`git mv scripts bin` + 全仓 69 处 `scripts/` 引用 sed 统一替换(代码 18 + 文档 45 + 其余),残留 0
- [~] **地图矢量管道 A+B 阶段**(§5.11 ✅ 2026-09-11,第 2/3 步完成):A 阶段 opendrive/mapvec/export/convert + A6 三件套;B 阶段 collect_surround(6 相机,实测 0.6 fps)+ assemble_maptr(infos 同构)+ B3 投影路面性 94.7–100%;C/D 阶段 = **参考自实现**(§5.11d 用户裁决 2026-09-11),`maptr_impl/` 顶层包 + 单帧过拟合锚定正确性;老栈 maptr env 降级 optional(需要官方权重对标再启动)
- [~] **嘉定地图候选**(2026-09-16 调研,挂起):嘉定路网仅存在于 CARLA **0.10.0(UE5)** 附加资产包。升级 = 引擎迁移(UE4.26→UE5.5,Lumen/Nanite 默认开启)+ 采集栈重验证(雷达 FOV 交叉 bug/同步 tick 语义/相机 SENSOR_MOUNTS spawn)+ **Town10 被重新建模 → 600 帧已采数据语义不可沿用**。GPU 非硬门槛:官方 0.9 口径 6-8GB,0.10 headless 关 Lumen/Nanite 12GB 可试;真需求才上 4090/24GB。等 600 帧扩数据结论(泛化间隙是否收窄)再定值不值
- [~] **长期工作目标:合成数据驱动的数据闭环**(§5.14 ✅ 2026-09-16,工业界对照定案):Phase 1 回归套件+场景库参数化 → Phase 2 探测器引导挖掘(数据告诉你该补什么场景)+ 边侧判定器原型 → Phase 3 flywheel 编排 + 伪标签回灌 + 真实性阶梯(嘉定/0.10 归此)。当前 600 帧扩数据 = flywheel 第一圈手动原型
- [x] **600 帧扩数据轮**(§5.11i ✅ 2026-09-16):官方相机布局重采 400 帧 + 旧 200 帧合并续训 256ep(loss 6.0→3.57);留出集 AP **0.0674→0.1607(+138%)**,泛化间隙 **3.9×→1.32× 大幅收窄**;权重 `outputs/maptr_600.pt`。下一步 = AutoLabel mapvec-report 消费复核 + 多图扩数据(Phase 2)
- [x] **教程能力 P-D~P-G 落地**(§9.3 ✅ 2026-09-18;原 §5.17 已迁出):单目(基线+生产口径)/双目/多雷达判据/累积建图/地面/聚类/3DGS 链路 7 项能力,采集器纯函数下沉(collect_rig.py)。剩教程 14(FAST-LIO2 外部 ROS 栈)列待办

### 5.16 CARLA 轨迹 → HiVT 训练管线(2026-09-18 ✅,教程 02 升级)

> **📦 已迁出(2026-09-19)**:完整执行记录(采集/组装/转换三段管线、train/val 数据口径、
> 跨图泛化弃用原因、minADE 6.08 与模式退化诊断)见 **[Plan2.md](Plan2.md) §9.2**。

### 5.17 教程能力 P-D~P-G 落地(2026-09-18 ✅,教程 08~16)

> **📦 已迁出(2026-09-19)**:完整执行记录(P-D 单目 / P-E 多雷达 / P-F 双目 / P-G 3DGS 含多俯仰
> 调优表 / 累积建图·地面·聚类 / 采集器纯函数下沉)见 **[Plan2.md](Plan2.md) §9.3**。
> 教程能力线的**后续计划与缺口**(教程 04 IPM、教程 14 阶段 2、P2-A 分割 GT)在 Plan2.md §3/§8。

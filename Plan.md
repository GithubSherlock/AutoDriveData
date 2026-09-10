# AutoDriveData — CARLA 仿真数据生成 Pipeline(方案定案 v0.7)

> 定案日期:2026-09-06(v0.1 架构)→ 09-06(v0.2 M0+KITTI 契约)→ 09-07(v0.3 M1 编排)→ 09-07/08(v0.4-0.6 M1a/M1b/M2/M3 执行记录)。
> 状态:**M0-M3 ✅ 工具链闭环;微调专项挂起;最终目标剩余三缺口拆 P1→P2→M4 三工作包推进中(§5.5,2026-09-08 用户拍板)**。

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
**有损映射**(xodr 语义更细),映射表进文档,**不静默丢要素**;③不做 MapTR 训练/推理
(依赖方向单向,AutoLabel 侧);④不改 A/B 采集纪律。

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

## 7. 待办/依赖(2026-09-07 更新)

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
- [~] **地图矢量管道 A+B 阶段**(§5.11 ✅ 2026-09-11,第 2/3 步完成):A 阶段 opendrive/mapvec/export/convert + A6 三件套;B 阶段 collect_surround(6 相机,实测 0.6 fps)+ assemble_maptr(infos 同构)+ B3 投影路面性 94.7–100%;C 阶段(MapTR/MapQR 预测,独立 maptr env)+ D 阶段(chamfer AP 评估)待做

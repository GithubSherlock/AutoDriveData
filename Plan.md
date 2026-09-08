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
| Python | base 3.10 / autolabel env 3.11 | API 支持 3.7–3.12 | ✅ 双 wheel |
| 内存/CPU | 440GB / 12 核 | 8GB+ | ✅ |
| 磁盘 | autodl-tmp **190G(余 76G)** | ~20G | ✅(已扩容) |

### 4.2 版本选型:**0.9.16**(官方 latest,2025-09-16)

- 弃 0.10.0(UE5 新线,周边生态滞后);0.9.16 = 0.9 线集大成 + 关键修复:instance segmentation 细粒度标注、**SensorData frame/timestamp/transform 与图像严格匹配**(GT 同步刚需)、Synchronized actor BoundingBox

### 4.3 安装与启动(已执行,速查)

```bash
# 安装位置 /root/autodl-tmp/CARLA_0.9.16/(19G);pycarla 在 base(3.10): pip install carla==0.9.16
# 启动(必须专用用户 carla,UE4 拒绝 root;headless 适配):
su - carla -c "cd /root/autodl-tmp/CARLA_0.9.16 && ./CarlaUE4.sh -RenderOffScreen -quality-level=Low"
```

### 4.4 环境共存策略(已定)

采集脚本独立进程跑在 base(3.10):只 import pycarla + numpy;绝不装进 `autolabel` env(mmdet3d CUDA13 编译环境脆弱)。落盘后再由 AutoLabel 侧读取(跨进程边界即解耦点)。

### 4.5 M0 执行记录(2026-09-06 ✅)

- 磁盘扩容后下载 8.35GB(29 分,5.1MB/s)→ 解压归拢 `CARLA_0.9.16/`(坑① tar 平铺无顶层目录)
- 坑② UE4 拒绝 root → 专用用户 `carla`(home `/root/autodl-tmp/carla_home`);坑③ `/root` 700 → `chmod 711`
- 首启 ~8 分钟(shader 编译);ALSA 报错可忽略;版本握手 0.9.16 ✅
- 坑④ 传感器属性设 blueprint(spawn 前);坑⑤ **轴系红线**:ray_cast LiDAR 默认 360° 旋转,原始数据为传感器系(x 前/y 右/z 上)≠ KITTI 相机系(x 右/y 下/z 前)
- 验收:`outputs/smoke/` + `scripts/smoke.py`(已参数化 --channels/--pps/--cam-offset/--pitch/--fov/--lidar-range)

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
| 6 | `scripts/collect_kitti.py`:ego 静止 + 摆 NPC,同步模式采 N 帧 | 输出一个 KITTI root |
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

**P1-1**(7624b2b)场景目录 + 单测;可模拟性矩阵写在 scenarios.py 模块 docstring(无 flare/镜头光学、无雪、LiDAR 雨损不可模拟——防把简化渲染当真实)。

**P1-2**(7c9b493)collect_drive --scene + 8 场景档实测确认(day_clear 为生产基底,其余覆写校验防打错字)。

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
1. **semantic LiDAR 后处理——出局**:信号灯/标志**不是 actor**(0.9.16_Opt 无
   traffic.traffic_light 蓝图,世界 0 信号 actor),车道线非实体无 tag——LiDAR
   打不到,无从后处理
2. **RoadRunner 资产化——挂起降级**:可行但依赖 M4;且车道线/信号定义本就在
   xodr,RR 只是换地图时的载体 → M4-1 定制街道时复用
3. **地图查询 API(定案,0 外部依赖)**:landmark 65 个(58×Signal_3Light_Post01
   红绿灯 + 6×Sign_Stop + 1×Sign_Yield,含世界位姿/类型/id);车道线 =
   waypoint.lane_marking 实体(type SolidSolid/Broken × color Yellow/White ×
   width 0.125,沿 lane 中心采样重建折线)。上帝视角、传感器解耦、与帧对齐
   由 ego 位姿锚定。

边界如实记录:Town10HD_Opt 无信号灯 actor → **无灯色状态周期**(状态属动态
范畴;若需灯光状态须换非 Opt 地图或 actor 注入,排后续);本图无限速牌
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
信号灯**灯色状态**不在本版本(Opt 无 actor),状态属动态 GT 范畴。

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

### 5.6 测试环境策略(已定)

- **纯数学单测**:base env(手算断言,不依赖 carla 与 auto3dlabel)
- **oracle 对比脚本**:autolabel env 跑,直接 import auto3dlabel 的 geometry/data 模块当单一事实源(双向转换往返断言)
- 两边互不污染,遵循 AutoLabel"依赖方向单向"纪律

## 6. 项目骨架

```
AutoDriveData/
├── Plan.md                # 本文件(单一事实源)
├── pyproject.toml         # autodrivedata 包:纯逻辑,numpy only
├── autodrivedata/         # 包:不 import carla(可在两 env 测试)
│   ├── geometry.py        # 坐标转换唯一落点(照 auto3dlabel 纪律)
│   ├── calib.py           # 内参/外参 → KITTI calib txt
│   ├── gt.py              # actor → label_2 行
│   └── export/kitti.py    # KITTI 布局落盘
├── scripts/               # 采集入口(依赖 pycarla,base env 跑)
│   └── smoke.py           # ✅ M0;collect_kitti.py 待建(步骤 6)
├── tests/                 # 单测(base)+ oracle 脚本(autolabel env)
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

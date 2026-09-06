# AutoDriveData — CARLA 仿真数据生成 Pipeline(方案定案 v0.3)

> 定案日期:2026-09-06(v0.1 架构)→ 09-06(v0.2 M0 调研 + KITTI 契约钉死)→ **09-07(v0.3 M1 编排定案)**。
> 状态:**M0 ✅ 完成;M1a 进行中**。

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
| **M1b nuScenes 语义** | 步骤 8–10(见 §5.2) | nuscenes-queue 分派通 |
| **M2 闭环验证** | autopilot 短途采集(数百帧)→ 比对层 → 复核队列 | 分歧率实测 <10%;AP/复核率报表 |
| **M3 场景参数化** | Traffic Manager 车流/天气/光照;长尾指令集 | 场景矩阵脚本 + 采集量可配 |
| **M4 定制街道** | RoadRunner trial 小型路网 | 定制地图跑通端到端 |

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

### 5.2 M1b — nuScenes 语义(步骤 8–10)

| # | 步骤 | 产出/验收 |
|---|---|---|
| 8 | `geometry.py` 补 P2 约定(cam_like 矩阵、尺寸换序、全局四元数) | 单测(对照 §3.2 公式) |
| 9 | `export/nuscenes.py`:nusbox + scene/sample token 落盘 | 用 auto3dlabel `data/nuscenes.py` 双向转换做**往返 oracle 校验** |
| 10 | 验收:`auto3dlabel nuscenes-queue` 分派通 | 复核队列生成 |

### 5.3 测试环境策略(已定)

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
- [ ] M1a 步骤 2–7(geometry/calib/gt/export/collect/集成验收)
- [ ] M1b 步骤 8–10
- [ ] 静态目标 GT(semantic LiDAR 提取)技术选型——M2 之后启动

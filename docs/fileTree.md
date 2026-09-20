# 文件树:AutoDriveData 仓库结构与文件职责

> **本文档的用处**:仓库的**文件级索引**——"某个文件是干什么的、该改哪、产物落在哪"。
> 定位是**导航**,不是事实源:方案定案看 [Plan.md](../Plan.md),新计划/待办看 [Plan2.md](../Plan2.md),
> 里程碑看 [milestone.md](milestone.md) / [milestone2.md](milestone2.md),AI 协作纪律看 [CLAUDE.md](../CLAUDE.md)。
>
> **用法**:
> 1. **找文件**——先在本文件的树里定位目录,再看该目录的职责表;
> 2. **加文件**——新增代码/脚本后**必须回来补一行**(见下方「维护约定」),否则本文件失效;
> 3. **找产物**——只看 `outputs/` 那一段;产物目录**不展开子文件**(数量大、随采集变动,不可控),
>    只说明"装什么、由谁产出、能否删"。
>
> **维护约定**:
> - 只收录**顶层与二级**条目;`outputs/` 下的具体数据集目录不逐一登记,只列类别。
> - 每行格式:`路径 — 一句话职责`;新增/改名/删除时同步本表,与代码改动同一个 commit。
> - 标注 **【未入库】** 的条目被 `.gitignore` 排除(权重/产物/本地配置),克隆后不存在,需自行生成。

## 1 顶层结构

```txt
AutoDriveData/
├── CLAUDE.md                    # AI 协作入口:纪律/红线/常用命令/环境表(改决策先读它)
├── README.md                    # 项目对外简介(定位 + 快速上手)
├── Plan.md                      # 方案定案 + 历史执行记录(§1~§4 契约/架构、§5 里程碑归档、§6 骨架、§7 待办快照)
├── Plan2.md                     # 新计划的制定地(2026-09-19 起):执行项/进度/遗留缺口 + §9 迁移归档
├── pyproject.toml               # 包定义 + 钉死 [tool.ruff] 规则集(110 列 / E,F,I,UP,B)
├── requirements.txt             # autodrivedata env 的 Python 依赖钉版本
├── .envrc                       # direnv:进目录自动激活 autodrivedata env(首次需 `direnv allow`)
├── .gitignore                   # 排除权重/产物/图像/本地配置(见各条【未入库】)
│
├── autodrivedata/               # ★ 纯值库:不 import carla,任何 env 可单测
├── maptr_impl/                  # ★ MapTR 参考自实现(torch2.x 现代栈)
├── maptr_official/              #   官方 MapTR/MapQR 项目侧胶水(仅适配器与配置)
├── bin/                         # ★ 可执行入口:采集 / 评估 / 可视化 / 探针(依赖 pycarla)
├── tests/                       # ★ 单测 + oracle 对比(autodrivedata env)
├── docs/                        #   文档:教程 / 里程碑 / 本文件
│
├── outputs/                     # 【未入库】全部产物的唯一落点(不展开,见 §7)
├── training/                    # 【未入库】地图矢量全量导出(数据侧,非训练代码)
├── lightning_logs/              # 【未入库】HiVT 训练日志与 ckpt(PyTorch Lightning 默认输出)
├── auto3dlabel/weights/         # 【未入库】3D 检测微调权重(AutoLabel 侧消费)
├── hdMapGitHub/                 # 【未入库】上游开源仓库克隆(HiVT/MapTR/MapQR,保持 pristine)
│
├── yolo11s-seg.pt               # 【未入库】实例分割权重(sem_bev.py 用,放项目根)
├── clear_cache.sh               # 【未入库】本机磁盘清理脚本(环境维护,非项目代码)
├── gitpush.sh                   # 【未入库】本机推送辅助脚本(环境维护,非项目代码)
├── build/                       # 【未入库】构建残留(setuptools 产物,可删)
└── .vscode/ .claude/ .pytest_cache/ .ruff_cache/ .ipynb_checkpoints/   # 【未入库】本地工具配置与缓存
```

**依赖方向(硬纪律)**:`bin/` → `autodrivedata/` / `maptr_impl/`;`autodrivedata/` **绝不 import carla**;
项目整体 → AutoLabel(3D 检测消费方)单向,**禁止反向**。

## 2 `autodrivedata/` — 纯值库(不 import carla)

| 文件 | 职责 |
|---|---|
| `geometry.py` | 坐标转换唯一落点:CARLA 系 ↔ KITTI 相机系 |
| `calib.py` | KITTI 标定生成(内参/外参 → calib txt,含 `world_to_img` 投影共用件) |
| `gt.py` | CARLA actor → KITTI label_2 GT 行 |
| `static_gt.py` | 静态目标/道路特征 GT(地图查询源:P2) |
| `traffic_light.py` | 交通信号灯状态 GT(动态时序层:状态归一/前向判据/相位查表) |
| `semantic.py` | CARLA 语义 LiDAR 标签 → KITTI 式强度合成(域差距修复) |
| `opendrive.py` | OpenDRIVE 1.4 解析(planView/lanes/objects/signals/junction) |
| `mapvec.py` | 地图矢量 GT 提取与采样(MapTR 口径:六类要素 + 裁剪 + 重采样) |
| `mapvec_schema.py` | 矢量预测对外契约 `mapvec_pred/1`(schema 校验/JSON 往返) |
| `mapviz.py` | 矢量投影与绘制:ego 系折线 → 相机像素 + BEV 面板 |
| `chamfer_ap.py` | MapTR 评估纯值:Chamfer 距离匹配 + 多阈值 AP(官方口径) |
| `compare.py` | 伪标签 vs GT 比对层:3D IoU 匹配 → 分歧帧 → AP/复核率报表 |
| `attribution.py` | 失效归因:逐帧 2D 匹配 → 漏检按距离/框高/TTC 分箱 |
| `scenarios.py` | Corner case 场景目录 + 可模拟性矩阵(P1) |
| `radar.py` | CARLA radar 原始检测 → nuScenes 18 字段雷达点云 |
| `paths.py` | 项目路径锚定(`project_path()`:相对路径 = 相对项目根) |
| `collect_rig.py` | 采集器纯值位姿/挂点计算(环绕位姿 / 双目挂点;AST 纪律守护) |
| `mono_depth.py` | 单目测距:检测框 → 地平面投影距离 + 迭代深度法 |
| `stereo.py` | 双目立体视觉:三角测量 / SGBM 视差 / NCC 匹配 / 自监督损失 |
| `multilidar.py` | 多雷达标定:point-to-plane ICP + overlap/plausible 判据 |
| `slam.py` | 激光 SLAM 纯值两段式(帧间点面 ICP 前端 + ScanContext 回环/PGO 后端);`icp_odometry` 双出口 = 位姿 `T` / 点映射 `T_delta` |
| `slam_eval.py` | 轨迹精度评估纯值:Umeyama 对齐 / ATE / RPE(evo·KITTI 口径);**纯直行序列的绕轴旋转不可辨识**见模块 docstring |
| `live_slam.py` | **在线 SLAM 会话**(纯值):`LiveSlam.push` 逐帧增量重建(链式约定逐字复用 `slam_odometry`)+ `map_in_ego_frame`/`traj_in_ego_frame` 换到当前 ego 系;**`SlamWorker` = 有界丢旧队列 + 帧间隙止损(`max_gap`,防"丢帧→间隙更大→ICP 更慢"正反馈),默认同步执行(worker 线程被 GIL 压到 eff 0.04–0.24)** —— 滞后有界的判据靠它单测 |
| `accum.py` | 累积语义点云建图:多帧 velodyne 全局累积 + 语义着色 |
| `ground.py` | 点云地面提取:RANSAC 平面拟合 + 网格法双路线 |
| `cluster.py` | 点云聚类障碍物检测:欧氏聚类 + 3D 包围盒 |
| `export/kitti.py` | KITTI 布局落盘(root 经 `KITTI_OBJECT_ROOT` 覆盖) |
| `export/nuscenes.py` | nuScenes 迷你集生成器(照 devkit 契约) |

## 3 `maptr_impl/` — MapTR 参考自实现(§5.11 C/D 阶段)

| 文件 | 职责 |
|---|---|
| `model.py` | 模型组装:ResNet50+FPN backbone + GKT BEV 变换 + 分层 query head |
| `gkt.py` | GKT(Geometry-aware Kernel Transform):环视相机特征 → BEV 特征 |
| `head.py` | 分层 query head:实例级 query + 点级 query + 置换等价匹配 |
| `dataset.py` | B2 infos json → 训练数据集(图像加载 + 位姿/标定透传 + GT 解析) |
| `chamfer_gpu.py` | Chamfer 代价矩阵 GPU 实现(与 `chamfer_ap` 同口径) |
| `device.py` | GPU 显存自适应实测工具 |

`maptr_official/`(**已终止线,勿主动重提**)仅存项目侧胶水:`bridge.py`(v1 数据源桥)+ `configs/`
(`maptr_v1`/`maptrv2`/`mapqr` 三份 CARLA config),官方仓库本体在 `hdMapGitHub/` 且保持 pristine。

## 4 `bin/` — 可执行入口(依赖 pycarla)

### 4.1 采集器(产出 KITTI / nuScenes / 专项数据集)

| 文件 | 职责 |
|---|---|
| `collect_kitti.py` | 静态采集:ego 静止 + 摆 NPC + 同步模式 → KITTI root(raw + GT) |
| `collect_drive.py` | 动态采集:ego autopilot + TM 车流 + 行人 → KITTI 序列 |
| `collect_ab_route.py` | P1 A/B 专用:ego 定速直行 + 路侧静置车(固定位置,帧级配对) |
| `collect_nus.py` | nuScenes 迷你集:6 相机 + LiDAR + 5 雷达 |
| `collect_surround.py` | 环视 6 相机采集(nuScenes 布局)→ 图像 + 内外参 + 逐帧 ego 位姿 |
| `collect_surround_micro.py` | 环视微采样(10 帧)→ 相机布局对照微实验 |
| `collect_static_gt.py` | 静态目标/道路特征 GT(地图查询源,含 overlay 目检图) |
| `collect_tl_states.py` | 灯色动态 GT(记录模式 / `--cycle` 受控切灯) |
| `collect_traj.py` | 多 agent 轨迹采集(HiVT 训练数据源) |
| `collect_stereo.py` | 双目 rig 采集(基线 0.4m)+ 真值深度 |
| `collect_3dgs.py` | 静态场景 360° 环绕采集(RGB + 真值深度,支持多俯仰) |
| `collect_slam.py` | SLAM 数据集采集:ego 定速巡游 → `training/velodyne/` + **`training/pose/`(ego 真值位姿,ATe/RPE 评估的 GT)** |

### 4.2 组装 / 转换(采集产物 → 训练口径)

| 文件 | 职责 |
|---|---|
| `assemble_maptr.py` | 环视采集 + 地图矢量 → MapTRv2 infos 同构 json |
| `merge_train_infos.py` | 拼接多组环视训练数据 → 合并 infos(帧号连续重排) |
| `convert_mapvec.py` | 地图矢量 → MapTRv2 annotation 口径 |
| `export_mapvec.py` | 地图矢量导出全量/帧级裁剪 json + BEV overlay |
| `assemble_traj_pt.py` | CARLA 轨迹 → HiVT TemporalData 组装(纯值) |
| `convert_hivt_pt.py` | plain dict → HiVT TemporalData(在 hivt env 跑) |
| `prepare_official_dataset.py` | 环视数据 → 官方栈可直吃的 nuScenes 形状数据集(已终止线) |

### 4.3 训练

| 文件 | 职责 |
|---|---|
| `train_maptr.py` | MapTR 训练入口(单帧过拟合 = 正确性锚点;多帧 = 常规训练) |
| `train_3dgs_mini.py` | 3DGS mini 训练(gsplat 光栅化) |
| `finetune_synth.py` | 合成 KITTI → pointpillars_kitti 微调(复用 AutoLabel train3d) |

### 4.4 评估

| 文件 | 职责 |
|---|---|
| `eval_2d_ab.py` | P1 逆光 A/B:冻结 YOLO11s 在两个 KITTI root 的 2D AP 对比 |
| `eval_attr.py` | 失效归因评估:多跑 × 距离/框高/TTC 网格 + 漏检画像 |
| `eval_kitti.py` | GT vs AutoLabel 伪标签比对报表(比对层 CLI) |
| `eval_maptr.py` | MapTR 评估:权重 → 逐帧推理 → 四类 chamfer AP(+ 逐帧契约落盘) |
| `eval_official_metric.py` | A′ 口径复算:并排算"自实现 chamfer AP"与"官方 eval_map" |
| `mono_distance.py` | 单目测距评估(检测框 → 距离,与 KITTI GT 真距对照) |
| `calib_multilidar.py` | 多雷达标定判据评估(注入已知误差 → 判据数值) |
| `slam_odometry.py` | SLAM 前端:逐帧 velodyne → 链式位姿 `T_k = P_{k-1}·inv(T_delta)`(双出口契约见 `slam.py`) |
| `slam_backend.py` | SLAM 后端:关键帧 + ScanContext 回环候选 + 双 yaw ICP 验证 + PGO(边存点映射 `Z_ij`) |
| `slam_diff_test.py` | 前端位对齐对拍:numpy vs `slam_cpp` 同一 `(prev,cur,init,seed)` 下比单次 ICP |
| `eval_slam.py` | SLAM 精度评估:LiDAR 系位姿 → ego 系(手性共轭 `M·T·M` + 杆臂 `inv(L)`)→ ATE/RPE |
| `build_accum_map.py` | 累积语义建图(多帧 velodyne → 全局语义地图) |
| `extract_ground.py` | 地面提取(逐帧点云 → 地面/非地面分离 + 统计) |
| `cluster_obstacles.py` | 聚类障碍物检测(地面分割 → 聚类 → 3D bbox) |
| `sem_bev.py` | 语义 BEV:图像 → YOLOPv2 + YOLO11s-seg → BEV 鸟瞰 |

### 4.5 可视化

| 文件 | 职责 |
|---|---|
| `view_stream.py` | 场景实时流:真 UE 渲染 + GT/预测 overlay → 浏览器 MJPEG |
| `live_common.py` | **实时可视化共享件**(从 view_stream 抽出):多槽 MJPEG 服务(单端口 `/stream/<name>` + `/` 索引页)/ **拼图(两套:`compose_grid` 等尺寸 + **尺寸守卫**,不符即 `ValueError` —— `paste` 源图大于目标框时只贴左上角、静默裁;`compose_rows` 按行拼、每格**原生像素**,studio 三层用它)** / GT overlay / **环视 rig(两代口径 `rig_spec`:`official` 逐相机 `SENSOR_MOUNTS` + 108.6/−110.8,`legacy` 共用 `SENSOR_OFFSET` + 235/125;`resolve_rig` 按权重名选)** / 第三方视角 / 键盘 —— view_stream 与 live_studio 共用 |
| `live_studio.py` | **8 路 studio**:6 相机 + BEV + 第三方 + `grid` 拼图槽,各占一路;`--keyboard` 折进 tick 循环(WASD 开采集);第三方非 attach 每 tick 摆位。**拼图 = `GRID_ROWS` 三层**(①左前/前/右前 ②右后/后/左后 ③第三方 + BEV),**每格原生像素不缩放**(相机 1242×375 / 第三方 640×360 / BEV `--bev-size` ⇒ 画布 3726×1170;旧 4×2 等尺寸布局把相机图裁到 621×187 丢掉地面,见 Plan2.md §P-L.6)。**`--slam` 接在线 SLAM**(挂语义 LiDAR → `SlamWorker`,BEV 槽画地图点/轨迹;`--slam-async` / `--slam-voxel` / `--slam-max-gap` / `--slam-report` 落验收 JSON)。**`--video` 落八视角视频段**(拼图槽逐帧写 mp4,cv2/mp4v 惰性开编码器;`--video-fps` 标称帧率 / `--video-tile` 放大倍数) |
| `viz_maptr_pred.py` | 预测回投目检:预测/GT 折线 → 6 相机 overlay + BEV 面板 |
| `viz_layout_cmp.py` | 相机布局对照数值化(同镜头两布局的可见性对比) |
| `drive_ego.py` | live 手动驾驶(服务器终端 WASD 遥控);薄封装 `live_common.KeyboardState`(studio 内置键盘是首选) |

### 4.6 探针 / 诊断(一次性验收与故障定位)

| 文件 | 职责 |
|---|---|
| `probe_mapvec_oracle.py` | 离线 xodr 解析 vs CARLA 运行时几何对账(0.00cm 验收) |
| `probe_mapvec_proj.py` | 矢量投影回 6 视角图像的路面性验收(数值诊断) |
| `probe_radar_l3.py` | 雷达物理合理性探针(对照真实 ars408 规格) |
| `probe_vulkan.py` | Vulkan 设备枚举——CARLA 渲染停摆的一线判据 |
| `probe_imu.py` | **B1 实测**:CARLA IMU 能否支撑 FAST-LIO2 的 IESKF 预测(结论:直行段 IMU 预测比恒速先验更差 → B3 不投) |
| `probe_scan_to_map.py` | **B2 实测**:oracle GT 局部地图下 scan-to-map vs scan-to-scan(结论:误差随地图深度 K 单调变差 1.05×→2.75×,重新体素化救不回 → 不建 ikd-Tree 前端;含代价/谱/地面占比三条机制证据) |
| `probe_rig_mount.py` | **环视挂点口径 A/B 实测**:同一权重喂「它训练时见过的 rig」vs「另一代 rig」→ 品红像素/段数差 = 错配代价(结论见 Plan2.md §P-L.1) |
| `smoke.py` | M0 smoke:CARLA headless 连接 → 同步模式 → 各取一帧落盘 |

### 4.7 共用件与运维脚本

| 文件 | 职责 |
|---|---|
| `carla_common.py` | 采集公共件:位姿换算 / NPC 摆放 / 传感器参数 / 同步模式 / 灯态归一与绘制 |
| `carla_server.sh` | CARLA 服务器启动/停止(GPU 修复栈 + Vulkan 兼容层自愈) |
| `gpu_fix/` | GPU 修复栈:LD_PRELOAD shim 源码 + 安装脚本 |
| `assemble_and_merge.sh` | 组装 + 合并 infos 的批量编排 |
| `finalize_maptr_600.sh` | 600 帧扩数据轮的收尾编排 |
| `smoke_radar_collect.sh` | 雷达采集冒烟 |
| `setup_maptr_official.sh` / `run_official.sh` | 官方栈环境搭建与运行(**已终止线**,重建设路子在 Plan.md §5.12) |

## 5 `tests/` — 单测与 oracle 对比(autodrivedata env)

| 类别 | 文件 | 说明 |
|---|---|---|
| 纯值库单测 | `test_geometry.py` `test_calib.py` `test_gt.py` `test_compare.py` `test_paths.py` `test_scenarios.py` `test_static_gt.py` `test_traffic_light.py` `test_semantic.py` `test_radar.py` | 手算断言,不依赖 carla / AutoLabel |
| 地图矢量线 | `test_opendrive.py` `test_mapvec.py` `test_mapvec_schema.py` `test_mapviz.py` `test_chamfer_ap.py` `test_chamfer_gpu.py` | 含闭式解手算锚点与真实 xodr 计数锚点 |
| 教程能力线 | `test_mono_depth.py` `test_stereo.py` `test_multilidar.py` `test_slam.py` `test_accum.py` `test_ground.py` `test_cluster.py` `test_collect_rig.py` | 各含手算锚点;`collect_rig` 兼作采集器回归先例 |
| SLAM 精度/在线线 | `test_slam_eval.py` `test_live_slam.py` | `slam_eval`:ATE/RPE 手算锚点 + 杆臂方向(不补杆臂 ATE 2.44×);`live_slam`:与离线 `slam_odometry` **逐帧同输入同输出**(<1e-12)+ `SlamWorker` 滞后有界/止损/同步模式 |
| 实时可视化 | `test_live_common.py` | `compose_grid` **尺寸守卫**(不符必抛,防 `paste` 静默裁)+ `compose_rows` 每格**原生像素**逐像素等于源图 + studio `GRID_ROWS` 三层行序。import `bin/` 模块需运行时加 `sys.path`(非包),静态分析跟不到 ⇒ 就地 `pyright: ignore[reportMissingImports]` |
| MapTR 自实现 | `test_gkt.py` `test_head.py` `test_device.py` | 单帧过拟合正确性锚定 |
| 落盘契约 | `test_export_kitti.py` `test_export_nuscenes.py` | 路径/字段与消费方契约一致 |
| oracle 对比 | `test_geometry_carla_oracle.py` `test_geometry_nus.py` `test_calib_oracle_autolabel.py` `test_gt_oracle_autolabel.py` `test_nuscenes_oracle_autolabel.py` | **需 autolabel env / CARLA 机器**,缺失时自动 skip |

## 6 `docs/` — 文档

| 文件 | 职责 |
|---|---|
| `fileTree.md` | **本文件**:文件级索引与维护约定 |
| `milestone.md` | 版本里程碑(P1 / MapTR 线) |
| `milestone2.md` | 教程能力线里程碑(HiVT-CARLA / 语义 BEV / 单双目 / 建图 / SLAM / 3DGS) |
| `testLog.md` | 测试与验证日志 |
| `PRD.md` / `TRD.md` | 需求/技术文档(**当前为空占位**) |
| `Carla_Sim_Tutorial_01..16.md` | 16 篇 Carla 仿真教程(ros-bridge 旧栈),Plan2.md 的能力对照来源 |

## 7 `outputs/` — 产物目录(唯一落点,**不展开子文件**)

> 全部写盘路径经 `autodrivedata/paths.project_path()`(相对路径 = 相对项目根,不随 cwd 漂移)。
> 下列目录**均【未入库】**,克隆后不存在,由对应脚本重新生成。

| 目录 | 装什么 | 产出者 |
|---|---|---|
| `kitti_*` | KITTI root 结构数据集(`image_2` / `velodyne` / `calib` / `label_2`):`kitti_day_clear` `kitti_drive` `kitti_sweep_*` 等 | `collect_drive.py` / `collect_kitti.py` |
| `kitti_ab_*` | P1 A/B 帧级配对序列(day_clear / sunset_glare / rain_night / dense_fog) | `collect_ab_route.py` |
| `kitti3d_ab_*` | 上列 A/B 的 3D 伪标签输出(AutoLabel 消费) | `auto3dlabel run` + `eval_kitti.py` |
| `surround_*` | 环视 6 相机数据集(`surround_train` / `surround_p3` / `surround_town13` / `surround_pred` 逐帧契约) | `collect_surround.py` / `assemble_maptr.py` / `eval_maptr.py` |
| `traj_town*` | 多 agent 轨迹数据集(HiVT 输入) | `collect_traj.py` / `assemble_traj_pt.py` |
| `maptr_*.pt` | MapTR 权重(`maptr_600` / `maptr_1000` / `ep256` / `ep512` + `.opt` 优化器态) | `train_maptr.py` |
| `maptr_600` / `maptr_1000` | MapTR 扩数据轮的 infos 与图像。**两轮的 `images/` 均已清理**(`maptr_1000` 于 2026-09-19 删 5.5 G;`maptr_600/images` 3600 图同期发现已不存在),现仅存 `maptr_600/map_infos.json`(33 M);`maptr_1000/` 已整目录不存在。**权重 `.pt` 均保留**。复现训练需重跑组装链(`surround_train` + `surround_p3` + `training/map/Town10HD_Opt_full.json` → `bin/finalize_maptr_600.sh`) | `assemble_maptr.py` / `merge_train_infos.py` |
| `kitti_sweep_day_clear_8` | **仅存**的速度档(70 帧 @8 m/s)。`_4` / `_12` 于 2026-09-20 清理(§5.10 结论已归档);重采 `collect_ab_route.py --speed N` | `collect_ab_route.py` |
| `kitti_wet_road` / `kitti_dense_rush` | **P1-6 候选**场景探针(各 12 帧,尚无 A/B 版)。同批的 `rain_night` / `dense_fog` 因已有 70 帧 A/B 版而列入待清;`heavy_rain` / `night_clear` **无** A/B 版,12 帧版是唯一数据 | `collect_drive.py` |
| `kitti_slam` / `slam_gt/` | SLAM 数据集(velodyne + **pose 真值**)与轨迹/精度产物(`traj_raw.json` / `icp_stats.json` / `eval_*.json` / **B 期验收 `accept_sync|accept_async|accept_parity_full.json`**) | `collect_slam.py` / `slam_odometry.py` / `eval_slam.py` / `live_studio.py --slam-report` |
| `sem_bev/` `mono_distance/` `stereo/` `multilidar/` `accum_map/` `ground/` `cluster/` | 教程能力线各产物的图/点云/结果 json | 各自 `bin/*.py` |
| `3dgs/` | 3DGS 环绕采集帧 + 真值深度位姿 + `.ply` 高斯 + 训练结果 json | `collect_3dgs.py` / `train_3dgs_mini.py` |
| `hivt_carla/` | HiVT 训练用 TemporalData 与场景划分 | `convert_hivt_pt.py` |
| `carla/` | **运行支撑物**:服务器日志 + LD_PRELOAD shim + Vulkan 兼容层 | `carla_server.sh` |
| `models/` | 推理权重(YOLOPv2 等) | 下载/转换 |
| `nus_mini*` | nuScenes 迷你集(含 L3 雷达口径变体) | `collect_nus.py` |

**可否删**:数据集与权重删前先确认 Plan2.md §5「数据资产」是否仍被引用。

> ⚠️ **已删目录已从本表移除**(2026-09-20,详见 Plan2.md §10):`dumps/` `viz_check` `viz_maptr_e120`
> `kitti_{slam,town13,town13_static}_probe` `maptr_600_pred` `mapvec_pred_*` `kitti_sunset_glare`
> `kitti_sweep_day_clear_{4,12}` `surround_micro_{legacy,official}` `zzz_probe.txt` `videos/`(空目录)。
> 这些目录**克隆后本来就不存在**(【未入库】),由对应 `bin/*.py` 重新生成 —— 需要时重跑即可。
> 保留 `kitti_sweep_day_clear_8`(速度档只剩 `_8` 一档)。

## 8 非代码目录(环境/上游)

| 目录 | 职责 |
|---|---|
| `training/map/` | 【未入库】地图矢量**全量**导出(`{map}_full.json` + BEV overlay + 帧级裁剪版),供 MapTR 训练组装消费 |
| `lightning_logs/` | 【未入库】HiVT 训练日志与 ckpt(PyTorch Lightning 默认落点) |
| `auto3dlabel/weights/` | 【未入库】3D 检测微调权重(AutoLabel 消费方) |
| `hdMapGitHub/` | 【未入库】上游开源仓库克隆:`HiVT` / `MapTR` / `MapTR_maptrv2` / `MapQR`,**保持 pristine**,项目侧改动一律放 `maptr_impl/` / `maptr_official/` |
| `.vscode/` `.claude/` `build/` `*_cache/` | 【未入库】本地 IDE 配置、AI 会话配置、构建与测试缓存 |

# Plan2:Carla 系列教程能力补全路线图(2026-09-19 更新)

> **文档分工(2026-09-19 用户拍板,Plan.md 头部同步声明)**:本文件 = **项目计划的制定地**
> (后续所有新计划、待办、排期都在这里);Plan.md = **方案定案 + 历史执行记录**(§1~§4 契约/架构、
> §5 里程碑归档、§6 骨架、§7 待办快照,均**冻结不再新增**)。
> 起点是「16 篇教程能力 → 本仓库栈补全」的路线图,现已扩展为**全项目计划载体**(§3 执行项、
> §7 执行进度、§8 遗留缺口);红线纪律/接口契约/环境表仍以 Plan.md 为准(§6 复述其要点)。
>
> **⚠️ 路径口径(2026-09-26 目录重构后)**:本文件 §P-L / §P-M 等**执行记录**里的 `bin/x.py`、`tests/test_x.py`、
> `maptr_impl/` 是**重构前口径**,现已不存在。映射:`bin/x.py` → `python -m autodrivedata.<能力>.x`、
> `tests/test_x.py` → `autodrivedata/tests/<能力>/test_x.py`、`maptr_impl/` → `autodrivedata/map/maptr/`。
> **历史记录不逐一回改**(它们记的是当时怎么跑的);**新写的计划请用当前路径** —— 当前路径看
> [docs/fileTree.md](docs/fileTree.md),结构与理由看 [docs/refactor-2026-09.md](docs/refactor-2026-09.md)。

## 1 定位

16 篇 Carla 仿真教程(`docs/Carla_Sim_Tutorial_01..16.md`)是 **ros-bridge / ROS1 py2.7 旧技术栈**
的实战系列。本仓库现在是 **纯值库 + carla 0.9.16 自研同步采集栈**(不依赖 ROS)。Plan2 =
在**现有仓库栈上**把教程能力逐项补齐,重复能力标「已有」,缺口列执行项。

**执行前提(2026-09-18 用户拍板,已过期)**:当时服务器**无 GPU**(`/dev/nvidia*` 不存在)、
`autodrivedata` env 迁移中 → 各执行项等 GPU/CARLA 恢复后启动。**2026-09-18 会话已恢复**:
GPU 可用(autodrivedata env,cuda=True)、CARLA 可起,下面 §3 执行项与 §7 进度按恢复后口径。
**2026-09-19**:torch 换装 2.6.0+cu124(旧 2.5 直连 pytorch.org 下载卡死 → aria2 **单线程**
从 R2 取包并比对官方索引 sha256;多线程 `-x 8` 会切坏 S3 multipart 边界、zip 能过但 hash 不符,
**勿再用**)。`autodrivedata` env 单测全绿(353 passed / 3 skipped)。

## 2 能力映射(Explore 2026-09-18,只读)

| # | 教程能力 | 本仓库对应 | 状态 |
|---|---|---|---|
| 01 | ROS py2.7↔py3.10 跨环境通信 | 不适用(本仓 CARLA 直连同 tick,无需 ROS) | 非本仓范式 |
| 02 | YOLOv8 实时检测 | `autodrivedata/perception/eval_2d_ab.py`(YOLO11s 2D AP)+ `attribution.py` | 已有(离线评估) |
| 03 | 4 相机安装 + 标定网格 | `collect_surround.py` + `SENSOR_MOUNTS`(**6 相机超集**);纯函数下沉 `autodrivedata/collect_rig.py`(`ring_cam_pose` / `stereo_rig_offsets`) | 已有 |
| 04 | 4 相机标定 + BEV 环视拼接(单应/IPM) | `calib.py` + `mapviz.py` 有标定/矢量投影;`sem_bev.py` 走 ground_intersection 射线投影;**无像素级 IPM 拼接** | 缺口(见 §8) |
| 05 | 实时图像语义分割(SegFormer / YOLOPv2) | `autodrivedata/perception/sem_bev.py`(YOLOPv2 检测+车道线+可行驶;YOLO11s-seg 实例掩膜) | ✅ |
| 06 | BEV + 语义分割融合 | `autodrivedata/perception/sem_bev.py` 像素级语义 BEV(ground_intersection 投影 + 世界→ego 旋转) | ✅ |
| 07 | 相机+LiDAR 融合,点云→图像 | `calib.world_to_img` / `tr_velo_to_cam` + `geometry.py` 完整投影链(含单测) | 已有 |
| 08 | 单目测距(4 法) | `autodrivedata/perception/mono_distance.py` + `autodrivedata/perception/mono_depth.py` / `geometry.mono_depth_from_box`(迭代深度法)+ `box_2d_from_3d`(GT 3D 投影框基线) | ✅ |
| 09 | 双目测距(视差) | `autodrivedata/sim/collect_stereo.py` 双目 rig + `autodrivedata/perception/stereo.py`(SGBM/NCC/三角测量) | ✅ |
| 10 | 上帝视角可视化(OpenDRIVE+NPC) | `view_stream.py --view top` + `opendrive.py` + `mapviz` | 已有 |
| 11 | LiDAR+语义建点云地图 | `autodrivedata/slam/accum.py` + `autodrivedata/slam/build_accum_map.py`(多帧累积,时序证据加权) | ✅ |
| 12 | 点云地面提取 | `autodrivedata/perception/ground.py` + `autodrivedata/perception/extract_ground.py`(RANSAC 平面拟合) | ✅ |
| 13 | 点云障碍物检测(聚类) | `autodrivedata/perception/cluster.py` + `autodrivedata/perception/cluster_obstacles.py`(欧氏聚类) | ✅ |
| 14 | FAST-LIO2 + SC-PGO SLAM | `autodrivedata/slam/core.py`(纯 numpy 两段式降档:帧间点面 ICP 前端 + ScanContext 回环/PGO 后端)+ `autodrivedata/slam/slam_odometry.py` + `autodrivedata/slam/slam_backend.py` + `autodrivedata/slam/slam_cpp.cpp`(阶段 2 位对齐对拍)+ `autodrivedata/sim/route.py`(回环路线的纯值侧) | ✅(阶段 1+2)⚠️ **口径 = 「教程能力闭环」,不是「ROS 栈跑通」** —— 无 ROS / 无 C++ 后端 / 无 IMU 融合,是**降档**(依据见 P-H.2);**回环分支已在真数据上端到端验证**(P-H.3.4:闭环序列 802 帧、114 条回环边、闭合误差 12.386→**10.808** m 对 GT 10.994 m、`ate_aligned` **1.4127→0.1352 m**;真回环召回 49/76) |
| 15 | 多激光雷达标定 | `autodrivedata/perception/multilidar.py`(point-to-plane ICP + overlap/plausible 判据)+ `autodrivedata/calib/calib_multilidar.py` | ✅ |
| 16 | 3DGS 重建 | `autodrivedata/sim/collect_3dgs.py`(环绕采集)+ `autodrivedata/gs/train_3dgs_mini.py`(gsplat 训练) | ✅(链路) |

图例:✅ = 链路已交付(详见 §7);「已有」= 补全前本仓已具备;「缺口」= 见 §8 遗留缺口。

## 3 执行项(2026-09-18 GPU/CARLA 已恢复;下列各项均已执行完毕,状态见 §7)

### P2-A 语义 BEV(教程 05+06)——✅ 链路已通(见 §7)
- **实际落地**:YOLOPv2(检测+车道线+可行驶三合一)+ YOLO11s-seg(实例掩膜)——**未跑 SegFormer**
  (原计划"两模型对比"降级为单模型链路;本仓无像素级分割 GT,对比缺裁判口径)
- **输入数据**:现成 `surround_train/cam_*`(Town10 300 帧 ±)+ `kitti_day_clear/training/image_2`(150 帧)——无需 CARLA 重采
- **投影链**:分割结果 → 复用 `calib.world_to_img` + `mapviz` 投影 → BEV 鸟瞰(教程 06)
- **验收**:✅ 21 帧像素级语义 BEV 图 + 与既有 MapTR 矢量 BEV 面板对照成立
- **备注**:本仓无像素级分割 GT,链路已通;GT 后续用 CARLA semantic camera 合成(P2-A 的后续子项)

### P2-B 3DGS 重建(教程 16)——✅ 已并入 P-G 链路闭环
- 原「范围一(拍板)」= 静态场景 360° 采集 + 建图 + 训练跑通 → **已由 P-G 交付**(gsplat mini
  训练替代 LiteGS,位姿用 CARLA 真值;见上文 P-G 与 outputs/3dgs/)
- **范围二(待定)仍成立**:多角度俯仰采集、效果调优(PSNR 由 17.9 再提升)——待定,不承诺排期
- 原依赖项更新:COLMAP CPU 已装(pycolmap 4.2.0),但其 SfM 在本场景退化(Sim3 误差 5.7m/79.6°,
  见 sfm_eval.json)→ 真值位姿为主口径

### P2-C 激光 SLAM(教程 14)——✅ 阶段 1+2 均已交付
- **原裁决**:外部 C++/ROS 栈(FAST-LIO2 + SC-PGO)与本仓无 ROS 范式冲突 → 列待办暂不启动
- **实际走向(降档)**:未搭 ROS,改在**本仓栈内**实现纯 numpy 两段式等价环(帧间点面 ICP 前端 +
  ScanContext 回环/PGO 后端)→ 见 §7 **P-H**,单测 27 passed。原"搭 ROS 工作量大、风险高"的判断成立,
  降档后链路已闭环;`kitti_*/training/velodyne/*.bin`(64 线语义强度)直接喂入,无需重采
- **阶段 2 口径(2026-09-19 交付)**:不做 ROS 移植,改做 **C++ 位对齐对拍**——
  `autodrivedata/slam/slam_cpp.cpp`(C++17 单文件零依赖)独立实现前端,`autodrivedata/slam/slam_diff_test.py` 在**同一
  `(prev, cur, init, seed)`** 下逐帧比对单次 ICP 的 T → 149/149 PASS(最差 1.9e-14 rad /
  7.0e-12 m),证明"纯 numpy 环 = C++ 环"在数值上等价,即阶段 1 的环可直接作为原生栈的 oracle

## 4 挂起项(衔接 Plan.md 冻结区)

- **HiVT-CARLA 轨迹预测主线** —— ✅ **已闭环**(见 §7 末条 P-K 与 §9.2)。原卡点(env 迁移未完成
  + 无 GPU)已随 2026-09-18 会话恢复解除,val minADE 6.08。
- **激光 SLAM 阶段 2(C++ 位对齐对拍)** —— ✅ **已交付**(2026-09-19)。未做 ROS 原生栈移植,
  改以 `autodrivedata/slam/slam_cpp.cpp` 独立 C++ 实现 + `autodrivedata/slam/slam_diff_test.py` 单次 ICP 位对齐对拍收口
  (149/149 PASS);阶段 1 的纯 numpy 环由此被确认为可移植的 oracle(见 §7 P-H)。

## 5 数据资产(可复用,无需重采)

> ⚠️ **2026-09-20 清理后**,下列清单已按实况订正(详见 §10)。

- 图像:`surround_train`(300 帧 × 6)/ `surround_p3` / `surround_town13` / `kitti_*` image_2
- 点云:`kitti_*/training/velodyne/*.bin`(64 线语义强度)/ `nus_mini` LIDAR_TOP + RADAR
  - **SLAM 输入**:`kitti_drive/training/velodyne/`(150 帧,`autodrivedata/slam/slam_odometry.py` 的默认输入)
- 标定:`calib.json`(surround 精确 sensor2ego + intrinsic)/ `training/calib`
- 轨迹:`traj_town10`(550 帧 4 agents)/ `traj_town13_clean`(175 帧)
- 3DGS:`outputs/3dgs/`(单俯仰 90 帧 + 多俯仰 270 帧 3×90,含真值深度与位姿)
- 双目:`outputs/stereo/`(基线 0.4m,left/right/depth 40 帧 + depth_pc 点云)
- **速度档**:`kitti_sweep_day_clear_8`(70 帧 @8 m/s)是**仅存**的一档(4/12 档已清,可重采)
- **P1-6 候选数据**:`kitti_wet_road` / `kitti_dense_rush`(各 12 帧,尚无 A/B 版)

## 6 纪律

- 本仓栈(纯值 + carla 0.9.16)不引入 ROS/rosbridge(旧栈教程能力一律在本栈上重新实现或标注不适用)
- 产物一律落 `outputs/`(`paths.project_path`,不随 cwd 漂移)
- 语义分割 GT:需要像素级 GT 时用 CARLA `sensor.camera.semantic_segmentation` 合成(采集侧),不手工标注
- 与 Plan.md 同律:Conventional Commits、提交不附 AI 署名、`ruff check && ruff format`、相关单测

## 7 执行进度(2026-09-19 更新;最新条目见 §P-M)

> **★ 标定口径已冻结(2026-09-23 用户裁决,§P-M.11)**:后续**采集 / 导出 / 训练 / 评测一律按该节**,
> 不再逐轮重新推导;该节同时列出**本口径下作废的旧结论**(别照旧文档"修正"回去)。

> 单测口径:`autodrivedata` env 全量 **353 passed / 3 skipped**(2026-09-19 torch 2.6.0+cu124 换装后复跑);
> 本节各条目的"N passed"是该模块自身测试文件的口径。**2026-09-22**:`pytest --collect-only` 已收集
> **712 用例**(§P-M 新增 `test_calib_live.py` 51 / `test_calib_probe.py` 55 / `test_depth_codec.py` 29 /
> `test_probe_calib.py` 9 / `test_nuscenes_cali_sensors.py` 33 与 `test_geometry_nus` 14 / `test_live_common` 24
> 的扩充),下方各节的"N passed"未逐条回填。

### P2-A 语义 BEV(教程 05+06)——✅ 链路已通
- `autodrivedata/perception/sem_bev.py`:YOLOPv2(检测+车道线+可行驶)+ YOLO11s-seg(实例掩膜)→ 像素级 BEV 投影
- 修 3 个坑:jit.load→torch.load(weights_only=False)、ego_pose 列表访问、投影坐标帧(世界→ego 局部旋转)
- 输出 `outputs/sem_bev/bev_{000000..000020}.png` + `panel_*.png`(21 帧样例),三色均值 da≈6754 / ll≈2560 / obj≈8146 px
- **验收通过**:像素级语义 BEV 图可生成,与 MapTR 矢量 BEV 面板对照成立

### P-D 单目测距(教程 08)——✅ 链路已通 + 评估修复
- `autodrivedata/perception/mono_distance.py` + 两个纯值模块:`geometry.mono_depth_from_box`(迭代深度法 z=H·fy/框高,H=1.6m)
  + `autodrivedata/perception/mono_depth.py:box_to_ground_distance`(地平面投影法)+ `geometry.box_2d_from_3d`(GT 3D 框角点投影框基线)
- **修复**:label 2D 列 59/97 是零宽退化框 → 检测框改走 GT 3D 框投影(与采集器 `box_to_gt_line` 同投影口径)="2D 检测框 = GT 3D 投影框"诚实基线
- **实测**(147 框):全距 mean 8.56% / median 7.8%;**10-20m 带 mean 8.47%、70% 框 <10%** 达标;7-10m 贴脸区系统低估(侧向角点拉大框高)如实排除;地平面投影法相机无俯仰不适用(None)
- 输出 `outputs/mono_distance/results.json`;单测 tests/test_mono_depth.py 10 passed

### P-E 多雷达标定(教程 15)——✅ 链路已通 + 判据修复
- `autodrivedata/perception/multilidar.py`(point-to-plane ICP,纯 numpy,零 carla/零 open3d)+ `autodrivedata/calib/calib_multilidar.py`(注入已知误差 → 判据)+ autodrivedata/tests/perception/test_multilidar.py 4 passed
- **修复**:收敛 = converged(增量阈值)∧ rmse_final<0.05m ∧ **overlap≥0.6** ∧ **plausible(t<5m、r<30°)**
- **实测**:small(0.1rad/0.1m)**converged**(overlap 0.991、iter 5、恢复 t 0.15m/r 5.7°);large(1.2rad/2m)**not_converged**(overlap 仍 0.991,recovered t 10.26m/r 68.8° 被 plausible 否决)→ RMSE 对平面场景天然低,overlap+合理性双闸分开真伪标定
- 输出 `outputs/multilidar/icp_result.json`

### P-F 双目测距(教程 09)——✅ 链路已通
- `autodrivedata/sim/collect_stereo.py` CARLA 双目 rig(基线 0.4m,同朝向 yaw=0、y 轴 ±0.2m)+ `autodrivedata/perception/stereo.py` 纯值链路(z=f·B/d 三角测量、SGBM 视差可选、NCC 纯 numpy、自监督 `reprojection_loss`)+ autodrivedata/tests/perception/test_stereo.py(手算锚点 10 passed)
- **实测**:定速 5.98 m/s(逐帧自证);SGM 近物点云 z≈5.5m ↔ GT 深度同值(三角测量链与真值深度对得上);depth_check.png 目检图已生成
- 输出 `outputs/stereo/`(calib.json + left/right/depth 40 帧 + depth_pc 点云)

### P-G 3DGS 重建(教程 16,降档链路验证)——✅ 链路闭环
- `autodrivedata/sim/collect_3dgs.py` 静态场景 360° 环绕采集(spectator 归位修复:attach 子 actor set_transform 是**相对父**位姿,不归位会绕空场地、PSNR~10)+ `autodrivedata/gs/train_3dgs_mini.py` gsplat mini 训练
- 初始化真值深度网格反投影 ~40k 点;**位姿用 CARLA 真值**(定位降级:pycolmap SfM Sim3 对齐误差 ~5.7m/79.6° → outputs/3dgs/sfm_eval.json),留出帧 0 作 val
- **实测**(train_result_ep1500.json):**psnr_all_mean 17.9、val 帧 0 13.56**;GT|渲染|差值三栏目检图 render_compare_ep1500.png 已生成
- 输出 `outputs/3dgs/`(capture 90 帧 + gaussians_ep1500.ply + train_result_ep1500.json)
- **调优(2026-09-18,多俯仰)**:`collect_3dgs.py --pitches "0,-15,-30"`(分 pitch 目录 images/p{p}/ + poses_{p}.json + pitches.json)+ `train_3dgs_mini.py` 跨 pitch 平铺 / `--scale` / `_DEPTH_LOWER`。120k 点 × 270 帧:psnr_all **17.9→18.32**(3000 iters),val 帧0 **13.56→10.45** 不升反降(多俯仰分走观测,留出单水平视角被稀释);消融与单环对照复现 val~10。多俯仰价值 = 补顶面/近地隐面(psnr_all ↑),代价 = 留出帧视角外推变难。产物 `gaussians_mp3_120k_v2.ply` / `train_result_mp3_120k_v2.json`

### P-H 激光 SLAM(教程 14,纯 numpy 两段式降档)——✅ 阶段 1+2 链路闭环
> ⚠️ **`FAST-LIO2` / `SC-PGO` 是对标名词,不是文件名**(按名字搜会搜空)——本线
> **无 ROS / 无 C++ 后端 / 无 IMU 融合**,是纯 numpy 降档,依据是 P-H.2 的两条实测否定。
> 且「✅ 闭环」指的是**两段式链路**跑通;**回环分支在真数据上端到端未验证**(P-H.3)。

- `autodrivedata/slam/core.py`(原 `slam.py`):**纯 numpy 核心环**(零 carla/零 torch/零 ROS),并作为阶段 2 C++ 移植的 oracle
  - 前端 = **帧间点面 ICP**(替代 FAST-LIO2 的 ikd-tree scan-to-map;帧间重叠 ~90% 时等效)+ 恒速先验初始化 + λ 正则化法方程
  - 后端 = **ScanContext 回环**(点计数描述子,列滚动不变)+ **位姿图 G-N**(节点 ≤200,纯 numpy,无 g2o)
- `autodrivedata/slam/slam_odometry.py`(S1.3 前端):逐帧 velodyne → 链式位姿 `T_0→k`,落 `outputs/slam/{traj_raw,icp_stats}.json`
- **位对齐纪律**(阶段 2 对拍前提):全 double、网格哈希 tie-break 钉字典序、体素重心按扫描序累加、
  λ 正则化解代替 lstsq/SVD —— numpy/C++ 对拍只允许 ~1e-12 求解舍入偏差
- 单测 `autodrivedata/tests/slam/test_slam.py` **27 passed**(手算锚点:exp/log 往返、多分辨率近邻、rad0 早停回归、批量 nearest vs 标量逐位一致、
  恒速先验链、SC 描述子旋转不变、SC 默认 min_gap 回归、PGO 纠偏、直线序列零漂移)
- **阶段 1 验收(2026-09-19 端到端实测,输入 `outputs/kitti_drive/training/velodyne/` 150 帧)**:
  `autodrivedata/slam/slam_odometry.py --root outputs/kitti_drive --frames 0-149 --out outputs/slam` →
  **wall 242.42 s(< 5 min 达标)、NaN 0、failed 1 帧、平均 RMSE 0.19593 m、平均 overlap 0.779**;
  产物 `outputs/slam/traj_raw.json` + `icp_stats.json`
  - **轨迹形状**:起点 (0,0,0) → 终点 (−42.21, −50.56, −1.40),路径长 **94.73 m**,yaw 0° → 121.12°,
    逐帧步长 mean 0.636 m / median 0.524 m —— **不是直线**(绕行约 121°),故 closure 的
    `drift_m 65.88` 是**开放路径首末位姿距离,不是漂移率**(源数据是 `collect_drive.py` 的 ego autopilot,
    无 GT 位姿文件可比 → 真漂移率不可得,如实标注)
  - **failed 帧 = 帧 3**:逐帧 ICP 步长 9.517 m / rmse 2.153 / overlap 0.022。已核为**数据侧 ego 瞬移**
    (相机帧 2→3 `mean|d|` 60.16、平均亮度 159.0→128.7,远超其余相邻对 12–24;GT 标注数 2→1;
    纯 x 平移 −1..10 m 暴力扫描 overlap 上限仅 0.128)——**非 ICP 缺陷**,链从帧 4 正常续上(overlap 0.875)
- **阶段 2(2026-09-19,C++ 位对齐对拍)**:`autodrivedata/slam/slam_cpp.cpp`(C++17 单文件,零外部依赖,含 `--selftest`
  语义自检)+ `autodrivedata/slam/slam_diff_test.py`
  - **判据订正**:不比"150 帧链式位姿末端"——链式 `T_k = Δ_k·T_{k-1}` 对 Δ 的舍入差**指数放大**(最近邻赋值
    是离散的:1e-16 的 seed 差翻格 → Δ 跳 ~1e-5 → 进入下一帧 seed,实测 ~2.4×/帧;两条纯 Python 链只把求逆
    从 `np.linalg.inv`(LU)换成刚体 Rᵀ 就能在 40 帧内发散到米级)。正确判据 = **同一 `(prev, cur, init, seed)`
    下比对单次 ICP 的 T**
  - **实测**:`autodrivedata/slam/slam_diff_test.py --root outputs/kitti_day_clear --start 0 --end 149` →
    **149/149 PASS**(tol 1e-3 rad / 1e-2 m),最差旋转 **1.9e-14 rad**、最差平移 **7.0e-12 m**;
    链式末端差(仅作放大率参考,不判 FAIL)rot 0 / trans 5.1e-14 m
- **后端实测**(`autodrivedata/slam/slam_backend.py`):15 关键帧(每 10 帧)、**候选 0 / 过门 0** —— 该序列是**开放路径无重访**,
  `n_loops=0` 是合法结果(**不造回环**);闭合 pre 63.185 m → post 63.185 m(无回环边时 PGO 不动,
  即"零信息时保持原样"的预期行为)。产物 `outputs/slam/{traj_pgo,loops,slam_summary}.json`

#### P-H.1 精度口径修正 + 带 GT 的 400 帧基线(2026-09-19)

此前所有 P-H 数字都建在**没有 GT 位姿**的 `outputs/kitti_drive`(ego autopilot)上,只能看轨迹形状。
本次用 `autodrivedata/sim/collect_slam.py` 重采 **400 帧带 GT 位姿**的序列(Town10HD_Opt,autopilot,路径 206.24 m,
`outputs/kitti_slam/`,位姿落 `training/pose/{fid}.txt`),才第一次能算真 ATE/RPE。

- **位姿约定 bug(94×)**——`icp_odometry` 原出口是 `T_delta @ init_T`,把**点映射当位姿左乘**:
  纯平移时看着像在累加,一转弯就发散。实测同一 400 帧序列 **ATE 17.53 m → 修正后 0.187 m**。
  正解 = `T = init_T @ inv(T_delta)`(推导与 src/ref 方向记忆法见 `autodrivedata/slam/core.py::icp_odometry`
  docstring + `autodrivedata/tests/slam/test_slam.py::TestIcpOdometry::test_chain_of_turning_motion_matches_ground_truth`)。
  **教训:ICP 解出的 `T_delta` 是点映射(src→ref),不是位姿;方向由哪个帧当 src 决定,别按形状记。**
- **坐标系换算**——LiDAR 系位姿 → ego 系:CARLA(ego,y 右)与 KITTI(LiDAR,y 左)手性差 = 共轭 `M·T·M`
  (`M = diag(1,−1,1,1)`);LiDAR 挂点 `(1.2, 0, 1.65)` ⇒ `ego_pose = M·T_lidar·M @ inv(L)`。
  **不带杆臂 ATE 0.4589 m vs 带杆臂 0.1877 m(2.44×)** —— 方向是 `inv(L)` 不是 `L`。
- **基线**(`autodrivedata/slam/eval_slam.py --traj outputs/slam_gt/traj_raw.json --gt outputs/kitti_slam`,产物 `outputs/slam_gt/eval.json`):

  | 指标 | 值 |
  |---|---|
  | 前端 wall / NaN / failed | 310.01 s(IC 300.76)/ 0 / 0 |
  | 平均 RMSE / 平均 overlap | 0.1306 m / 0.8177 |
  | **ATE(对齐后)** | **0.1877 m**(mean 0.1798 / max 0.3099 / final 0.1957),尺度 1.00829 |
  | ATE(原始,未对齐) | 67.98 m(首帧即差一个世界系偏移,对齐后才有意义) |
  | RPE Δ=1 / 5 / 10 | 0.0216 / 0.0568 / 0.0948 m;0.0481° / 0.1153° / 0.1768° |
  | 相对误差 | 路径长的 **0.091%** |
  | 分段步长比(直行/转弯/回程/"静止") | 0.993 / 0.996 / 0.990 / **1.145** |

  - 末段"静止"步长比 1.145:序列尾段 ego 已停,GT 几乎不动而估计仍有微小位移 ⇒ 比值虚高,
    **不是漂移**(该段 GT 位移本身接近 0,比值口径在此失真)。
- **后端(诚实报告)**:40 关键帧、**候选 0 / 接受 0**,闭包前 58.688 m → 闭包后 58.688 m。
  根因已数值验证:最小 ScanContext 距离 0.5855(门 0.15),几何上最近的关键帧对相隔 58.69 m ——
  该路线是折返路线但**从未在空间上重新靠近**。`n_loops=0` 是正确结果,不是缺陷。
- **阶段 2 对拍复核**:`autodrivedata/slam/slam_diff_test.py --root outputs/kitti_slam --start 0 --end 99` →
  **99/99 PASS**(最差平移 2.2e-11 m,旋转 0),修正后的位姿约定在 numpy/C++ 两侧一致。

#### P-H.2 路线 B(B1 IMU / B2 scan-to-map)实测裁决 —— **两项均不投**(2026-09-19)

用户指令"做 B1 的实测,没问题的话直接上 B2"。两项都做了实测,结论都是**负面**,故 B3/B4 不启动。

- **B1(CARLA IMU 能否支撑 IESKF 预测)⇒ 不投**(`autodrivedata/sim/probe_imu.py`,产物 `outputs/imu_probe/*.json`):
  - IMU 陀螺读数**就是物理引擎报的角速度**(三轴比值恒 ≈1.000),但 8 m/s 直行时
    `gyro.z = −1.29°/s` 而旋转矩阵差分算出的真实 yaw 速率只有 −2e-5 rad/s(**差 1000×**);
    该伪角速度完全可复现(三次运行一致到 6 位小数),只在部分速度档出现(6/7/8 m/s 明显,4/9/10/12 几乎为 0)。
    **车没转,IMU 说有转** ⇒ 纯积分把 4.5° 假偏航灌进姿态。
  - **单步预测误差(决策判据)**:直行 8 m/s 时 IMU 位置 0.326 mm vs 恒速 CV **0.120 mm**(差 2.7×)、
    姿态 0.0903° vs **0.0041°**(差 22×);只在满舵绕圈时 IMU 才赢(6.016 vs 15.458 mm)。
  - 叠加已测的"CARLA **不模拟帧内扫描延迟**"(运动畸变校正是空操作):FAST-LIO2 用 IMU 的两个卖点
    在本仿真里**一个为空、一个为负** ⇒ **IESKF 不投**。
- **B2(ikd-Tree + scan-to-map 前端)⇒ 不投**(`autodrivedata/slam/probe_scan_to_map.py`,产物 `outputs/s2m_probe/s2m_probe.json`):
  - 用 **oracle GT 位姿**构造局部地图(零里程计漂移 ⇒ 测出的是**收益上限**),8 采样帧、裁 30 m、
    对应距离门 2.0 m、体素 0.5 m。误差随地图深度 K **单调变差**:K=1 **1.05×**、K=3 **1.55×**、
    K=8 **2.75×**(重新体素化更差:1.02× / 1.55× / **3.26×**)。
  - 四条机制证据:**① 残差下降 ≠ 位姿正确**(RMSE@ICP 0.0984→0.0419 与 RMSE@GT 0.0996→0.0468 同步降,
    ICP 净赚的代价降幅仅 0.0012→0.0050 m,而位姿误差 0.040→0.105 m ⇒ 代价在解附近变平);
    **② 约束方向塌陷**(AᵀA/N 最小特征值 1.23e-1→9.80e-2,误差在**最软**特征向量平移块上的投影
    0.0174→0.0438 m,在最硬方向恒 ≤0.0005 m);**③ 内点被地面稀释**(|n_z|>0.9 占比 0.555→0.605,
    而地面法向在 x/y/yaw 上零信息);**④ 法向稳定性假设被否证**(法向一致性 0.849→0.954 **上升**,
    重新体素化也救不回来)。
  - ⇒ 与 §3 最初的判断一致(**帧间重叠 ~90% 时 scan-to-map 无收益**),本仓不建 ikd-Tree 前端。
    **结论边界**:单序列、Town10HD_Opt、autopilot 400 帧、体素 0.5、oracle GT 位姿;
    换到帧间重叠低的场景(高速、稀疏扫描、大转弯)**可能翻转**,届时本脚本可直接复跑复核。
- **副产品**:`autodrivedata/slam/eval_slam.py`(ATE/RPE,含双坐标换算)+ `autodrivedata/slam_eval.py`(纯值)
  + `autodrivedata/sim/collect_slam.py`(带 GT 位姿的采集)+ `tests/test_slam_eval.py`。

#### P-H.3 回环链路端到端 —— 路线侧修复(路网找环 + 纯追踪跑圈)(2026-09-27)

**问题**(不是新发现的缺陷,是 P-H.1 已如实标注的状态):后端(ScanContext 回环 + PGO)
**在真数据上一次都没触发过** —— 两段已测序列(`kitti_drive` 150 帧、`kitti_slam` 400 帧)
都是「候选 0 / 接受 0」。回环链路当时只有**合成单测**证据 + **两次"正确地不动作"**的证据,
**没有端到端真数据证据**。

**根因**(本次实测,决定方案):`outputs/kitti_slam` 400 帧的真值轨迹是**马蹄形** ——
右侧下行 → 沿底边横穿 → 左侧上行,**两条平行街相距 68 m**。而 `SC_MAX_RANGE = 40 m`
的 range-view 描述子**看不到对面那条街**。逐帧自证:

| 量 | 实测 |
|---|---|
| 自接近(关键帧差 ≥25 = 帧差 ≥250)最小距离 | **58.70 m** |
| 间距 < 40 / 30 / 20 / 10 / 5 / 3 m 的帧对数 | **0 / 0 / 0 / 0 / 0 / 0** |
| 车停下的帧 | **第 320 帧**(后 80 帧原地不动) |

⇒ **「多采帧数」是无效解**(只是多录静止帧)。要触发回环只能换**控制方式**:让同一处真的被走两次。

**裁决**(用户 2026-09-27):「车道图找环 + 纯追踪跑两圈」。另两个备选是直线往返(便宜)与
合成倒序序列(零成本),**都否掉**:这个方案**确定性可复现 ⇒ 能变成长期回归夹具**,
且覆盖**任意朝向的重访**(最接近真实场景);而往复/倒序只能覆盖"原路反向"这一种,
朝向差恰好 180°,正落在 ScanContext 列滚动不变的**盲点区**(要靠 `slam_backend` 的双 yaw 初值 ICP 才救得回来)。

**实现**:`autodrivedata/sim/route.py`(纯值,零 carla)+ `collect_slam.py --route loop`(采集器只做编排)。

| 构件 | 职责 | 钉住的东西 |
|---|---|---|
| `find_cycle(start, successors)` | BFS 求从起点可达的**最短有向环**;吃 `successors` 回调 ⇒ 图算法与 CARLA 解耦,可在合成图上单测 | `min_len=4` 拒退化环(`A→B→A` 是**掉头**不是环);`max_nodes` 兜住无限链;取**最短**而非"第一个发现的"(BFS 出队的深度序不保证回边长度单调) |
| `make_successors(step)` | CARLA `wp.next()` → 节点键 `(road_id, lane_id, round(s/step))`;**只吃 duck-typed `WaypointLike`** | ★ **`s` 必须进键** —— 只留 `(road_id, lane_id)` 时一条长路只有一个节点、后继键等于自己 ⇒ 自环被 `min_len` 拒 ⇒ **静默地永远找不到环**;合成 Waypoint 上钉「4 段 × 30 m ⇒ 40 节点环、单圈几何 **120.0 m**」(= 39×3 + 收尾那 3 m,同时钉住"闭环长度要把末点回首点那段算进去") |
| `pure_pursuit` | 前视点 + 航向误差比例控制 + 定速(超速只刹车,**不叠油门**) | ★ **进度单调**(前视点索引不许回退 —— 否则车在起点附近来回蹭,而那**看着像在开**:有速度、有转向,只在轨迹形状上暴露);★ **打舵符号**(目标在右 ⇒ `steer` 为正;符号反了车朝反方向冲出去,两圈跑完才知道) |
| `lap_budget` / `speed_ceiling` | **★ 采集前的第一道闸** | 一圈帧数 = `环长/(速度 × tick)` 必须 ≥ `SC_MIN_GAP_NODES × KEYFRAME_EVERY` = **250** ⇒ 换算成**速度上限**:200 m 的环 8.0 m/s、**120 m 的环只有 4.8**。开快了不是"少采几帧",而是两次到访**连候选都进不去** |

**为什么这道闸门必须在采集前**:一次闭环采集 ≈ **25–35 min**(采集 2–3 / 前端 ~12 / 后端 5–10)。
不先算,采完才发现帧差不够 = 白押一轮 GPU。`--dry-run` 只找环 + 报预算,**不落数据**。

**闸门顺序**(`plan_loop_route`,任一不过即 `RuntimeError` 并带可执行建议):
① **几何闭合** —— `route_closure` 首末缺口 ≤ `max(2.5·step, 5%·环长)`。容差**必须按采样步长给**:
节点键是分桶量化的,收尾那段天然带一个 `step` 的残差,按 1 m 给会**假红**;
② 单圈长度 ≤ 600 m(BFS 给的已是最短环,仍这么长就该换起点);③ 速度上限。
采集时另有**逐帧跟线自证**:偏离路线 > 8 m 连续 40 帧即**中止且不落盘**
(跟丢线的序列对回环验证没有价值,静默落盘只会得到一份"看着完整、实则无重访"的数据)。

**验证协议(判据先写死,跑完不许改)**:

| 序 | 判据 | 可信度 | 结果 |
|---|---|---|---|
| 1 | **几何自证(完全不依赖 SLAM)**:用 GT 位姿直接算「第二圈第 k 帧 ↔ 第一圈对应帧」的距离 ≈ 0 ⇒ 数据里**真有重访** | 最高 | ✅ 帧差 ≥400 时 97.5% <2 m(中位 0.216 m),真重访帧差 ≈404 |
| 2 | `loops.json`:`n_candidates > 0` **且** `n_accepted > 0` | 高 | ✅ 196 / **114**;但**真回环只占 49**(召回 49/76,见 .4 的构成表) |
| 3 | `slam_summary.json`:`closure_pre` **第一次成为真漂移指标** —— 路径真闭合后首末距离不再等于路径长度,P-H.1 明确标注的「开放路径口径缺陷」在这条序列上消失 | 中 | ✅ 对 GT 10.994 m:pre 12.386(+1.392)→ post 10.808(−0.186) |
| 4 | `eval_slam` 前后对比:**ATE 不劣化** | 中 | ✅ `ate_aligned` 1.4127 → **0.1352 m**;代价是 d10 旋转 0.24°→0.47° |

> ⚠️ **边界(不写会被读成注水结论)**:**CARLA 不漂移** —— 前端 400 帧 ATE 仅 **0.1877 m**(P-H.1),
> 累积漂移本来就小,**PGO 的"收益"在这类数据上天然很小**。判据**不能**是"ATE 大幅下降",
> 而是「回环链路**真的触发**」+「PGO 收到回环边后**不劣化**」+「回环边残差 / 闭合误差」。
> `n_loops=0` 若再现,**要给为什么**(帧差不够?ICP 过不了门?),**不许改门限硬凑**。

**实现与离线验证**:`autodrivedata/tests/sim/test_route.py` **45 passed**(含合成 Waypoint 的适配层回归)。

##### P-H.3.1 采集实测(2026-09-27,`outputs/kitti_loop`)

`collect_slam --route loop --laps 2 --speed 8`(dry-run 报环 86 节点 / 269.36 m / 建议帧预算 891):

| 量 | 实测 |
|---|---|
| 采出帧数 / 圈数 | **802 帧** / `laps_done=2` |
| 逐帧跟线偏差(进度索引口径) | mean **1.253** / p95 2.214 / **max 2.620 m**(中止阈 8 m,未触发) |
| 总里程 / step | **511.65 m** / 中位 **0.647 m/帧** ⇒ **实际 6.47 m/s**(目标 8.0,**0.81×**:纯追踪到不了目标速度) |
| 末帧距起点 | 10.34 m(在一个前视距离内 ⇒ 圈数计数与"差 10 m 没停回起点"是自洽的) |

##### P-H.3.2 判据 1 —— 数据里真有重访(GT 口径,完全不依赖 SLAM)

| 约束 | <2 m 重访占比 | 最近重访距离 中位 / max |
|---|---|---|
| 帧差 ≥ 250(=25 关键帧) | 71.0% | 0.260 m / 78.649 m |
| 帧差 ≥ 400(=40 关键帧) | **97.5%** | 0.216 m / 8.409 m |

**真重访帧差 ≈ 404**(= 40 关键帧 > `SC_MIN_GAP_NODES=25`);帧 410–415(s=259 m)回到起点 2 m 内
⇒ **单圈实际驱动里程 ≈ 259 m**(比纯值侧的 269.36 m 短 3.8%,纯追踪切弯);帧 800 距帧 396 **0.016 m**。
71% 那一行被**第一圈帧**稀释(它们按构造不存在 250 帧外的重访),不是数据缺陷。

> ⚠️ **这条判据我算错过两次,记在这里免得再犯**(两次都是"拿前端轨迹当真值"的变体):
> 第一次把 `traj_raw.json` 的 `T`(**前端估计**)同时当 prior 和真值 ⇒ 得到"真回环 **0 个**"的假象;
> 第二次把约束写成 `P[:n-gap]`,于是"最近邻"永远落在 `i-1`(相隔 1 帧)上 ⇒ 得到"**100%**、中位 **0.000 m**"的假象。
> **真值只认 `training/pose/`;约束必须是 `j ≤ i-gap`。** 两次都因数字"太好"而回头查才逮到。

##### P-H.3.3 成本诊断 —— 回环验证的真实开销与三条走不通的廉价代理

后端此前**跑不完**(实测 30 min 只跑完 <25/81 个关键帧)。根因不是迭代次数,是**失配云上的第一次迭代**:

| ICP 情形 | 单次耗时 |
|---|---|
| 对齐(同一份点云) | 0.16 s |
| 平移 10 / 40 / 100 m | 6.8 / **92.7** / 142.6 s |
| 真回环(yaw=0,9–12 次迭代才收敛) | 3–10 s |
| 真回环的**反极分支**(yaw=π,永不收敛) | **82–125 s** |

失配云无近邻 ⇒ `nearest_batch` 的网格早停失效(全层全点);而 SC 给的是 top-5、其中只有一部分是真回环
⇒ 351 个候选 × 双 yaw 全落在这条成本曲线上。

**三条廉价代理,全试过、全不行**(勿重走):

| 代理 | 真回环 vs 干扰 | 结论 |
|---|---|---|
| SC 描述子距离 | 真对 0.034–0.092 vs 中段 0.014–0.099 | 中段**最小值比真对还小** ⇒ 分不开(SC 量的是"场景像"而非"位置近") |
| 2 m 粗格占格重叠 | 真对 0.086–0.180 vs 远对 0.001–0.191 | 区间重叠;且 425 ms/候选,一点都不便宜 |
| 全量云首迭代 overlap | 真对 0.226–0.555 vs 反极支 0.01–0.03 | **能分开**,但一次 1.5–18 s ⇒ 省不了钱 |

**最后落地的三条闸(都只用来看/拒,不进接受判据)**:

| 常量 | 值 | 依据(实测) |
|---|---|---|
| `LOOP_PRIOR_MAX_M` | **5.0**(原 20.0) | 真对的**前端**距离 **3.05–4.15 m**(中位 4.02 = 关键帧相位量化 ≤0.5×6.5 m + 一圈前端漂移 ~1.5 m)。闸 2/3 m ⇒ **误杀全部 76 个真对**;5 m ⇒ 误杀 **0**、远对(GT >15 m)漏进 **0**;20 m ⇒ 多白跑 137 次分钟级 ICP |
| `LOOP_SCREEN_STRIDE` / `_OVERLAP` | 8 / 0.10 | **抽稀查询云 + 1 次迭代**的预筛(overlap 是比例,抽稀是无偏估计):真对筛值 min **0.187**(余量 1.9×)⇒ **0/76 误杀**,成本 **0.2 s**(vs 全量首迭代 1.5–18 s);反极支筛值恒 0.01–0.03 ⇒ 0.2 s 出局 |
| (短路,无新常量) | — | yaw=0 支**已过门** ⇒ 不跑反极支(一份点云不可能同时与 0° 和 180° 对齐;过门候选的反极支 overlap ≤0.03、rmse ≥2.4)。**接受判据没动**:反极支仍在"第一支没过门"时才跑(覆盖反向重访) |

**实测成本分解**(669 s):筛 196 次 + 82 次反极支筛(0.2 s 级)≈ 1 min;
**196 次全量 ICP 全部落在 yaw=0(已对齐)那一支上**(前端相对位姿给的初值本就对,错的是 180° 那支)
⇒ 平均 **3.4 s/次**,合计 ≈ 10 min。**失配路径(82–125 s/次)被完全消掉了** ——
这正是"跑不完 → 11 min"的全部原因。剩下的 196 次对齐 ICP 是**固有代价**:粗几何与描述子都分不开
"真重访"与"同街近邻"(见 .4 的构成表),只有真配准能判。

> ⚠️ **边界修正**(本小节初稿写的「CARLA 不漂移 ⇒ PGO 收益天然很小」**被实测否掉了**,见 .4):
> CARLA 的**真值**不漂移,但我们的**前端**(纯 numpy ICP 里程计)在 512 m / 802 帧上累积了
> **3.05–4.15 m** 的重访相对误差 —— 这就是**回环边残差 = PGO 真正要纠的量**(其中 ~1.5 m 是漂移,
> ~2.6 m 以内是关键帧相位量化)。

##### P-H.3.4 判据 ②③④ —— 端到端结果(2026-09-27,`outputs/slam_loop`)

**闸门链与耗时**:整条后端 **669 s = 11.1 min**(此前"跑不完"、粗估小时级)。

| 阶段 | 数 | 说明 |
|---|---|---|
| SC 候选 | **351** | 描述子 top-5 ∧ `min_gap=25` 关键帧 |
| 先验闸拒 | 155 | `LOOP_PRIOR_MAX_M = 5.0` |
| 廉价筛(候选级出局) | **0** | 但 **82 个候选的反极支**被筛在 0.2 s 上(逐候选记 `skipped_branches`) |
| 进全量 ICP | 196 | |
| 反极支被短路 | **114** | yaw=0 支已过门 ⇒ 不跑反极支 |
| **过门(=`n_loops`)** | **114** | **判据 ② ✓** |

两条优化各贡献一半(短路 114 次 × ~50 s + 筛 82 次 × ~50 s);无它们 ≈ **2.7 h**,实测 **11 min**。

**判据 ② 的构成(GT 核对 —— 数值达标 ≠ "检出的都是真回环")**:

| | 条数 | GT <2 m(真重访) | GT 2–15 m(同街近邻) |
|---|---|---|---|
| **接受** | 114 | **49** | 65 |
| **拒绝** | 82 | **27** | 55 |

- **真回环召回 = 49/76 = 64.5%**(27 条真回环被 ICP 门拒掉)
- ★ **接受/拒绝两组的 GT 分布几乎重合**(中位 2.10 vs 2.38 m)⇒ **在 5 m 先验闸之内,ICP 门对
  "是不是真重访"几乎不区分** —— 它判的是"能不能对齐",而 4 m 内的两片云本来就都能对齐。
  **区分力主要来自 SC 描述子 + 先验闸,不是 ICP 门**(此性质本身值得记:它说明"过门数"不能当"检出数"报)
- 提示(未用于任何闸):真回环 `sc_dist` 中位 0.0554 vs 近邻 0.0746 —— 有信号但重叠严重(见 .3 负面结论表)

**判据 ③ 闭合误差(第一次成为真漂移指标,不再是"路径长度量级")**:

| | 值 | 对 GT 的误差 |
|---|---|---|
| **GT 参考**(kf0=帧 0 ↔ kf80=帧 800) | **10.994 m** | — |
| `closure_pre`(前端) | 12.386 m | **+1.392 m** |
| `closure_post`(PGO) | 10.808 m | **−0.186 m** |

⇒ **这 1.58 m 修正可 100% 归因于回环边**:只含里程计边的位姿图按构造以链式轨迹为最优解(残差恒 0)
⇒ 它**不可能**移动闭合误差。

**判据 ④ `eval_slam` 前后**(802 帧,`lever_arm=True`,对齐含尺度):

| 指标 | pre | post | 判 |
|---|---|---|---|
| **`ate_aligned` rmse** | **1.4127 m** | **0.1352 m** | **改善 10.4×** |
| `ate_aligned` mean / max / final | 1.2605 / 2.4246 / 2.4246 m | 0.1239 / 0.2548 / 0.1297 m | 全线改善 |
| `ate_aligned` scale | 1.00773 | 1.01018 | — |
| `rpe` d1 trans rmse | 0.0222 m | 0.0229 m | 略劣化(+3%) |
| `rpe` d10 **rot** rmse | 0.2414° | **0.4696°** | **劣化 ~2×**(绝对值仍 <0.5°) |
| `ate_raw`(未对齐,被原点偏移主导) | 68.53 m | 68.58 m | **口径陷阱,勿引用** |

⇒ **判据 ④ 通过**(ATE 不劣化 —— 实际大幅改善);代价是**局部旋转一致性略劣化**(d10 0.24°→0.47°),
这是位姿图把误差全局重分配的正常代价,**如实记下**。

> ⚠️ **边界修正(第二次,推翻了本小节初稿的预测)**:
> 初稿写「CARLA 不漂移 ⇒ PGO 收益天然很小」——**实测否掉**。CARLA 的**真值**确实不漂移,
> 但**我们的前端**(纯 numpy ICP 里程计)在 512 m / 802 帧上 `ate_aligned` = **1.41 m**(0.28%),
> PGO 收到回环边后压到 **0.135 m**。P-H.1 的「前端 400 帧 ATE 0.1877 m」是**另一条序列**
> (autopilot 直线段为主、无 2 圈闭环),**不能拿来预测本条** —— 环路弯多、单圈 259 m,
> 前端累积漂移大得多。**教训:跨序列借用精度数字 = 换了个口径,和跨机型比 AP 同病。**

### P-I 累积语义建图 / 地面提取 / 聚类(教程 11+12+13)——✅ 链路闭环
- 累积:`autodrivedata/slam/accum.py` + `autodrivedata/slam/build_accum_map.py`(ego 位姿变换累积到全局系 + 时序证据加权)→
  `outputs/accum_map/map.ply`(150 帧);`tests/test_accum.py` 11 passed。累积显著抑制单帧伪影
- 地面:`autodrivedata/perception/ground.py` + `autodrivedata/perception/extract_ground.py`(RANSAC 平面拟合 + 内点掩码)→
  `outputs/ground/`;`tests/test_ground.py` 8 passed
- 聚类:`autodrivedata/perception/cluster.py` + `autodrivedata/perception/cluster_obstacles.py`(DBSCAN 风格邻域密度连通)→
  `outputs/cluster/`(150 帧,均值 117.97 簇/帧);`tests/test_cluster.py` 4 passed
- 注:上一会话"聚类任务失败"是误报——产物齐全,仅末行 bash 因 /tmp 配额满报错

### P-J 采集器纯函数下沉(回归测试先例)——✅
- `autodrivedata/collect_rig.py`(零 carla,AST 纪律守护)+ `tests/test_collect_rig.py` 8 passed
  - `ring_cam_pose`(3DGS 环绕位姿)+ `stereo_rig_offsets`(双目挂点 ±baseline/2)
- `autodrivedata/sim/collect_3dgs.py` / `autodrivedata/sim/collect_stereo.py` 改为 import 纯函数,bin 只剩 carla 编排
- 匹配纯函数 `match_dets_to_gt` 并入 `attribution.py`(P-D 生产口径共用,IoU 与 AP 评估同口径)

### P-K HiVT-CARLA 轨迹预测主线——✅ 管线闭环(原 §4 挂起项)
- 重采 Town10 运动段(v2/v3/v4,ego 8m/s 直线)+ Town13 v2(运动段)
- train=Town10 250 场景 / val=Town10 211 场景(同图时间外推;跨图 Town13 因静止分布弃用)
- HiVT-64 CPU 100 epoch → val minADE **6.08** / minFDE 15.85 / minMR 1.000(逐场景最优 mode)
- 模式退化诊断:各 mode 终点收敛 6-8m(恒速 3s=24m 的 1/3),train reg_loss 过拟合下降 → 数据规模/多样性问题,非管线问题
- 权重 `lightning_logs/version_0/epoch=99-step=3199.ckpt`;详见 §9.2

### P-L 8 路实时可视化 studio + 键盘操控 + 在线 SLAM(A 期 ✅ 2026-09-19;B 期 ✅ 2026-09-20)

用户目标(原话):「输出 六相机视角 + BEV 视角 + 第三方视角 共 8 个终端可视化输出,
我操控汽车便可采集动静态目标和道路特征,输出感知结果的同时也做 slam 重建。」

**分期裁决**:① 分两期,先 A(8 路 + 键盘 + 第三方)后 B(在线 SLAM);② 抽
`autodrivedata/sim/live_common.py` + 新 `autodrivedata/sim/live_studio.py`(**不动**已验证的 `--maptr` 路径)。

**A 期交付**:

| 文件 | 动作 | 要点 |
|---|---|---|
| `autodrivedata/sim/live_common.py` | 新增 | 多槽 MJPEG(单端口 `/stream/<name>` + `/` 索引页)/ 拼图 / GT overlay / 环视 rig / 第三方视角 / `KeyboardState` / MapTR 懒加载 |
| `autodrivedata/sim/live_studio.py` | 新增 | 9 槽 = 6 相机 + `BEV` + `THIRD_PERSON` + `grid`(拼图,当时是 4×2;2026-09-20 改**三层**且不缩像素,见 §P-L.6);`--keyboard` 折进 tick 循环 |
| `autodrivedata/sim/view_stream.py` | 改 | 薄编排:共享件全部改走 `live_common`;新增 `--rig` |
| `autodrivedata/sim/drive_ego.py` | 改 | 薄封装 `live_common.KeyboardState`(独立进程遥控用法保留,studio 内置键盘是首选) |

**A 期验收(全部数值,不靠目检)**:

1. **挂点自检**:`[rig] 实挂相机 vs 该口径规格 最大偏差:平移 0.000 m / 偏航 0.000°`
   —— 矩阵顺序踩坑见下。
2. **第三方视角 ego 在画面内**:`中心偏移 0.001 画幅 / 宽 0.116 画幅 / 深度 9.71 m`
   (判据 偏移 ≤0.5、宽 ∈[0.05,0.6])。**注意 `follow_spectator` 必须在 tick 之前设**
   (渲染用 tick 时刻的位姿),自检读 spectator 位姿必须在 tick 之后。
3. **8 路各自独立且内容正确**(`--maptr` + `--npcs`,`--dump` 同帧 raw/overlay 差集):

   | 路 | 尺寸 | 差集 px | 占比 |
   |---|---|---|---|
   | CAM_FRONT | 1242×375 | 22533 | 4.84% |
   | CAM_FRONT_RIGHT | 1242×375 | 36891 | 7.92% |
   | CAM_FRONT_LEFT | 1242×375 | 20898 | 4.49% |
   | CAM_BACK | 1242×375 | 14540 | 3.12% |
   | CAM_BACK_LEFT | 1242×375 | 32988 | 7.08% |
   | CAM_BACK_RIGHT | 1242×375 | 11333 | 2.43% |
   | THIRD_PERSON | 640×360 | 25619 | 11.12% |
   | BEV | 420×420 | 17574 | 9.96% |
   | grid | 2484×374 | 150109 | 16.16% |

   6 相机路差集 >0 且量级一致(2.4–7.9%),第三方路画 ego 自身白框、BEV 路出预测面板。
4. **回归**:`--view grid6 --maptr-ckpt outputs/maptr_ep512.pt --maptr-bev --dump` 复跑,
   品红像素 overlay **6002 px**(6 相机路,已屏蔽右下 BEV 子区域)vs raw **0**、
   **光轴以上 0/6002**、6 格 `v_min` 109–131(与解析值 `cy + fy·1.65/30 = 110.6` 吻合);
   BEV 面板区域另计 7128 px。§5.11f 记的 25553 px 是**另一帧另一内容**(场景 NPC/朝向不同),
   验收判据(raw 0 / 地平线以上 0 / 各格非零)**全部成立**。

**踩坑 1 —— 矩阵顺序 `inv_ego @ cam`(不是 `cam @ inv_ego`)**:把世界系相机位姿换算到
ego 系必须左乘 ego 逆。写成右乘会把 ego 的**世界坐标**混进平移块,实测偏差 **138.243 m**;
而**偏航恰好仍是 0.000°** ⇒ 只看偏航自检会漏掉,故 `rig_mount_deviation` 同时报平移。

**踩坑 2 —— `rotation_matrix_to_carla` 的 pitch 符号**:写成 `pitch = asin(−R[2,0])`
会静默反号(第三方视角俯仰 −12° → +12°)。已落 `autodrivedata/utils/geometry.py` 纯值实现 +
`tests/test_geometry.py::TestRotationMatrixToCarla` 往返单测(200 随机旋转逐元素 <1e-12)。

#### P-L.1 环视 rig **两代并存** —— rig 必须匹配权重训练数据(2026-09-19 实测订正)

**原计划前提只对了一半。** 计划说"`view_stream.build_maptr_rig` 给 6 路共用 `SENSOR_OFFSET`
是真 bug,修成逐相机挂点即可"。实测发现:**它取决于哪个权重**。

| rig | 平移 | BACK_LEFT/BACK_RIGHT 偏航 | 训练数据 | 对应权重 |
|---|---|---|---|---|
| `legacy` | 6 路共用 `SENSOR_OFFSET`(1.2, 0, 1.65) | 235 / 125 | `surround_train` / `surround_drive` | `maptr_ep256.pt` / `maptr_ep512.pt` |
| `official` | 逐相机 `SENSOR_MOUNTS[name]` | 108.6 / −110.8 | `surround_p3` / `surround_town13` | `maptr_600.pt` / `maptr_1000.pt` |

证据链:
- `outputs/surround_train/map_infos.json`(ep512 的训练 infos)六路 `sensor2ego` 全为
  `[1.2, 0.0, 1.65]` + BACK_LEFT 235 / BACK_RIGHT 125 ⇒ **legacy**;
  `surround_p3` / `surround_town13` 与 `SURROUND_CAMS` + `SENSOR_MOUNTS` **逐字段一致** ⇒ official。
- `outputs/maptr_600/map_infos.json` **逐帧查得**:帧 0-199 = legacy、帧 200-599 = official
  (600 帧轮是"重采 400 帧官方布局 + merge 旧 200 帧")⇒ `maptr_600.pt` 以 official 为主。
- 引入时点:`SENSOR_MOUNTS` 与 108.6/−110.8 布局在 commit `38cfe90`(2026-09-16)引入,
  **该 commit 未改 `autodrivedata/sim/view_stream.py`** ⇒ 旧 rig 一直是 ep512 的正确口径。

**A/B 探针**(`autodrivedata/calib/probe_rig_mount.py`,同一 ego 位姿同一 tick 帧,只变 rig):

| 权重 @ rig | 预测实例 | 段 | 品红 px | 光轴以上 |
|---|---|---|---|---|
| `maptr_ep512.pt` @ legacy(训练口径 ✓) | 153 | 269 | 122711 | 0 |
| `maptr_ep512.pt` @ official(错配 ✗) | 148 | 290 | 135989 | 0 |
| `maptr_600.pt` @ official(训练口径 ✓) | 163 | 326 | 173711 | 0 |
| `maptr_600.pt` @ legacy(错配 ✗) | 158 | 282 | 140562 | 0 |

**处置**:`live_common` 同时保留两套口径(`rig_spec`),`view_stream` / `live_studio` 加
`--rig {auto,nuscenes,legacy}`;`auto` 按权重名查 `LEGACY_CKPTS` 选(**默认喂对**)。
**正确性判据 = 外参与该权重训练数据逐字段一致**,不是"数字变了"也不是"用最新布局"。
⇒ `CLAUDE.md` / Plan.md §5.11f 里 `--maptr-ckpt outputs/maptr_ep512.pt` 的命令现在经
`auto` 自动走 legacy,**行为与文档化验收一致**。

> ⚠️ **2026-09-22 订正(见 §P-M):上表的 `official` 那一代 rig 本身是错的**——
> 它的偏航是官方方位角的**镜像**(漏了 `yaw_carla = −az_nus`),pitch/roll 硬编码 0。
> 故 `maptr_600.pt` / `maptr_1000.pt`(以及**全部** MapTR 权重)已标废弃,须重采重训。
> 本节保留的价值 = "rig 必须匹配训练数据"这条**方法论**,以及错配代价的量级。

**B 期交付(在线 SLAM,2026-09-20 ✅)**:

| 文件 | 动作 | 要点 |
|---|---|---|
| `autodrivedata/live_slam.py` | 新增(纯值) | `LiveSlam.push/snapshot`(链式约定逐字复用 `slam_odometry`)+ `map_in_ego_frame`/`traj_in_ego_frame`(LiDAR-0 系 → 当前 ego 系)+ **`SlamWorker`**(有界丢旧队列 + 帧间隙止损) |
| `autodrivedata/sim/live_studio.py` | 改 | `--slam` 挂语义 LiDAR → `SlamWorker`;BEV 槽画地图点(灰)+ 轨迹(青);HUD 显式报滞后;`--slam-report` 落验收 JSON;`finally` 先 join 再销毁 world |
| `autodrivedata/map/mapviz.py` | 改 | `bev_points`(散点,批量像素)/ `bev_trajectory`(只连窗内相邻点)/ `bev_window_mask`(窗口判据单一来源) |
| `tests/test_live_slam.py` | 新增 | 26 passed:与离线 `slam_odometry` **逐帧同输入同输出**(<1e-12)+ 滞后有界/止损/同步模式 |

#### P-L.2 在线 SLAM 的两条原计划前提**都不成立**(2026-09-20 决定性实验)

**前提 1「离线 ICP 0.78 s/帧 ⇒ 必须 worker 线程」** —— 0.78 s 是 400 帧**含转弯/重访的
平均值**(`icp_stats.json`: wall 310.01 / icp 300.76 / n=400),而**在线逐帧(gap 1)只有
0.15–0.35 s**(CARLA 语义 LiDAR 116k 点 → voxel 1.0 下采样 ~5–6k)。

**前提 2「worker 线程能救帧率」** —— 恰恰相反,worker 线程被 **GIL 饿死**。同一对点云
(studio 里 2.301 s vs 离线复算 0.344 s),主线程施加不同负载:

| 主线程负载 | wall | thread-CPU / wall(eff) |
|---|---|---|
| 空闲 | 0.19–0.26 s | 0.86–1.00 |
| PIL 画 6 路 + JPEG | 0.24–0.30 s | 0.71–0.79 |
| CARLA `world.tick()` | 0.24–0.29 s | 0.79–0.90 |
| **纯 Python 小矩阵自旋** | **13.6–15.3 s** | **0.04** |

判据:studio 主线程每 tick 的 GT overlay / `compose_grid` / HUD / 灯态绘制是**纯 Python
字节码**,持 GIL 不放。钉 `OPENBLAS_NUM_THREADS=1` **不改结论** ⇒ 不是 BLAS 线程池
(`threadpoolctl` 报 OpenBLAS 12 线程是无关项)。studio 实测 `process_time` 12.9 s vs wall 2.4 s
⇒ eff 0.24,与上表一致。

**处置**:`SlamWorker(sync=True)` **默认同步**(push 在主线程就地跑,eff 0.90–1.00、
ICP 0.15–0.35 s、零丢帧、滞后 0),代价 = 帧率 ~2.4 fps;`--slam-async` 保留为对照路径。

#### P-L.3 有界丢旧队列**不足以**保证滞后有界 —— 必须按**帧间隙**止损

原不变量「有界丢旧队列 ⇒ 有界滞后」**对 ICP 是假的**:队列有界的是**深度**,不是
`prev_down` 与当前帧的**间隙**,而 ICP 成本随间隙**爆炸**(体素 1.0:CARLA 序列实测
0.8 m 间隙 0.2 s → 2.4 m 2.1 s → 8 m 2.8 s → 32 m 39 s → 49 m 79 s)。丢帧 ⇒ 间隙更大
⇒ 更慢 ⇒ 更多丢帧 = **无界正反馈**。

**实测证据(异步模式复跑,`accept_async.json`)**:voxel 2.0、70 s、259 tick —— 处理
**8 帧**、丢 **248 帧**、滞后从 0 单调涨到 **157 帧(45.2 s)**、`lag_bounded False`;
`n_processed` 在 tick 11 之后**再没动过**(最后 4 帧 ICP 各耗 ~10 s)。这正是"没有稳态"的形态。

**修复 = 按帧号差止损**:`frame_idx − last_done_idx > max_gap`(默认 3)时该帧**不做 ICP**,
直接恒速外推 `poses[-1] @ delta_prev`(`push(dead_reckon=True)`),`prev_down` 照推进
⇒ 下一帧回到 gap 1。被止损的帧如实计入 `n_dead`(位姿带外推漂移,不掩饰)。

**滞后口径也订正**:`lag = 已 tick 帧号 − 已处理帧号`,**不能用 `n_offered − n_processed`**
—— 丢旧之后那个差值随丢帧数无界增长,看着像故障其实队列一直是满的;帧号差才反映
"SLAM 落后当前时刻多远",且丢旧时它自动收敛。

#### P-L.4 B 期验收(全部数值)

1. **与离线基线对拍 ✅** —— 400 帧 / voxel 0.5 / Town10HD_Opt,`accept_parity_full.json`:

   | 量 | 在线 `LiveSlam` | 离线 `traj_raw.json` | 参照 `eval_fixed.json` |
   |---|---|---|---|
   | ATE(对齐) | **0.18768 m** | 0.18768 m | 0.18768 m |
   | 尺度 | 1.00829 | 1.00829 | 1.00829 |
   | RPE 平移(d1) | 0.02161 m | 0.02161 m | — |
   | 平均 RMSE / overlap | 0.13060 / 0.8177 | 0.1306 / 0.8177 | — |

   链式位姿**逐元素最大差 0.0**(400 帧)、ATE 差 **0.0 m** —— 在线会话与离线基线**完全同解**。
   在线 400 帧耗时 342.9 s(ICP 332.5 s),离线 310.0 s(ICP 300.8 s),差 10% 来自逐帧
   `voxel_downsample` 与地图合并(离线不建累积地图)。

2. **滞后有界 ✅** —— 同步模式 `accept_sync.json`:136 tick / 136 处理 / **丢 0 / 止损 0**、
   `lag_max 0`、前半峰 0 / 后半峰 0、`lag_bounded True`、ICP 33.2 s、rmse 0.106、overlap 0.608。
   异步对照(`accept_async.json`)如实记为 `lag_bounded False` —— **保留这条反例**作为
   "为什么默认同步 + 为什么必须止损"的证据。

3. **停干净 ✅** —— 退出后 `nvidia-smi` **6394 MiB = 基线**(服务器仍在跑,故不是 0)、
   残留 `live_studio` 进程 **0** 个、日志有 `[slam] worker 已停(干净退出)` 与
   `[done] 相机/actor 已清理,服务器恢复异步`。

4. **BEV 数值自证 ✅** —— `bev_points` 画出 **90611 点**、轨迹 **20 段**;
   `bev_drawn_equals_in_window True`(面板画出的点数 90611 = 用 `bev_window_mask` 独立重算的
   窗内点数 90611,两处判据一致 ⇒ 窗外点一个都没画上)。

   **判据订正**:原写"地图点在 ego 系窗口内的比例 ≈ 100%",这是**误读** —— 累积地图覆盖
   整段行程(75 s × 8 m/s ≈ 300 m),BEV 窗口只有 30 m × 60 m,故
   `bev_map_in_window_ratio`(窗内 / 全部地图点)= 0.241 是**正常的**。真正的自证是
   "画出的 = 窗内的",不是"窗内占比高"。

**B 期结论**:8 路 studio + 键盘 + 在线 SLAM 全部打通,用户目标(6 相机 + BEV + 第三方共 8 路,
操控采集的同时做 SLAM 重建)达成。**性能边界如实报告**:同步模式帧率 ~2.4 fps(voxel 1.0)/
~1.5 fps(voxel 0.5),ICP 与渲染在同一个 tick 里排队;要提速只有两条路 —— 提 voxel
(1.5 → ICP 0.10 s)或把 overlay/拼图移出主线程(纯 Python 是 GIL 争用的根源)。

#### P-L.5 八视角**视频段**输出 `--video`(2026-09-20,用户问"能输出八视角可视化的一段检测")

**需求**:用户提交 GitHub 后问「现在怎么玩 Carla?可以输出八视角可视化的一段检测?」——
即把 studio 的拼图槽逐帧编码成一段 mp4,而不是只留浏览器里的实时流。

**实现**(`autodrivedata/sim/live_studio.py`):`--video <path>` / `--video-fps` / `--video-tile`。

- 编码器 **cv2(mp4v)**,惰性打开(**首帧到齐才开**,尺寸随 `--video-tile` 变,避免先猜尺寸);
  `vw.release()` 在 `finally` 里(与 worker 停止同一段,顺序:停 worker → 关编码器)。
- 环境事实:本机**无 ffmpeg 二进制、无 imageio/av**,但 autodrivedata env 有 **cv2 4.11.0**
  (`VideoWriter`/`VideoWriter_fourcc` 齐全)。cv2 的 `.pyi` 只声明了类方法
  `VideoWriter.fourcc`、**没有**模块级 `VideoWriter_fourcc` C 绑定 ⇒ 走
  `getattr(cv2, "VideoWriter_fourcc")` 包一层(noqa B009),否则 pyright 报未定义属性。
- `--video-tile` 默认 **1**。曾默认 2,实测只是**插值放大**(30 帧 5120×1440 / 40 MB),
  信息量不变、文件翻倍 ⇒ 改回 1 并在 help 里写明。
- **标称 fps vs 实际采集 fps 分开报**:循环跑不到 `--video-fps` 时视频会被**加速播放**,
  结束时打印 `实际采集 X fps ⇒ 播放速度是实时的 N×`,用户据此把 `--video-fps` 调到接近实测值。
  `--slam-report` JSON 同步落 `video_path` / `video_frames` / `video_fps_nominal` / `video_fps_real`。

**实测**(`outputs/videos/studio_8view.mp4`,MapTR ep512 + SLAM + 6 NPC 定速 6 m/s):

| 量 | 值 |
|---|---|
| 帧数 / 尺寸 / 时长 | **43 帧** / 2484×374 / 86.0 s(标称 0.5 fps) |
| 实际采集帧率 | **0.43 fps** ⇒ 播放 1.2×(近似实时) |
| 文件 | 10.3 MB |
| 8 格内容(非黑占比) | 84–100%(逐格核过,首末帧不同) |
| HUD(中途) | `pred=156/seg=354`、`SLAM 滞后 0帧`、`已处理 25 丢 0 止损 0`、`地图 369390点 rmse 0.177`、`BEV 点 95810/369390 轨迹段 24` |

**帧率口径(如实)**:MapTR + SLAM 同开时主线程每 tick 要跑 6 路推理 + 6 路 overlay +
拼图 + ICP,实测 **0.3–0.5 fps**;只开 GT overlay(无 MapTR/SLAM)约 2 fps。
尺寸 2484×374 = MapTR 路径下 `disp_w = 1242×0.5 = 621` × 4 列 × 2 行(无 MapTR 时是 2560×720)。
**该 4×2 等尺寸布局已于 2026-09-20 废弃**——它把相机图裁到 621×187 丢掉下半(地面),见 §P-L.6;
现布局为三层、每格原生像素,尺寸 3726×1170。

**结论**:「八视角检测视频」这条路径打通(`--video` 一行开关);**不是新能力**,是既有 8 路
studio 的**录制出口**——检测框/灯色/BEV 地图点与轨迹/HUD 全部按实时流同一份渲染写进视频。

#### P-L.6 拼图改**三层**且不缩像素 —— 用户报告"FoV 缩得看不到地面"的根因是 `paste` 静默裁剪(2026-09-20)

**用户报告**(看完 `studio_8view.mp4`):「6 视角摄像头的 FoV 缩小得都看不到地面了,能否改成三层,
以第一层是摄像头的左前、前和右前,第二层是右后、后和左后,第三层是第三视角和 bev,
且都不要为了整齐而缩减像素尺寸。」

**根因 = `PIL.Image.paste` 在源图大于目标框时不报错、不缩放,只贴左上角**。旧拼图是等尺寸
`compose_grid`(`cols=4`),MapTR 路径下 `disp_w = 1242×0.5 = 621`,6 路 1242×375 相机图被塞进
**621×187** 的格子 ⇒ 右半 + **下半(正是地面)** 被无声丢弃。

| 假设 | 与实测拼图格的平均绝对差 |
|---|---|
| 源图**左上角裁剪**到 621×187 | **0.128** |
| 源图**整幅缩放**到 621×187 | **66.18** |

⇒ 是**裁剪**不是缩放(逐相机复算 0.12–0.21 vs 60–86;直接对 1242×375 → 621×187 做一次
`paste` 复现,差 **0.000**)。这解释了"FoV 变窄"的观感:不是 FoV 变了,是画面被切走了一半。

**改动**:

- `autodrivedata/sim/live_common.py` 新增 **`compose_rows(rows, bg, center=True)`**:按行拼,**每格按自身原生像素
  原样摆**,行高 = 该行最高格、行宽 = 该行各格宽之和、整幅宽 = 最宽行、窄行居中。
- `compose_grid` 加**永久回归守卫**:格尺寸不符直接 `ValueError`(措辞含"静默裁"与"改用
  compose_rows"),把这一类坑钉死不再静默复发。`view_stream.py --view grid6` 的调用不受影响
  (它的格子全是 `disp_w × disp_h`)。
- `autodrivedata/sim/live_studio.py` 新增布局常量 **`GRID_ROWS`**(**不沿用 `SURROUND_CAMS` 的字典序** ——
  那样第二行会变成"左后/右后"与地理直觉相反)+ `grid_rows()`(整行缺名则丢该行,`--dump` 的
  raw 拼图没有 BEV 时不留空行);tick 里改调 `compose_rows`。
- 画布 **2484×374 → 3726×1170**(3×1242 宽;375+375+420 高)。`--video-tile` help 同步订正。

**验收(数值,不靠目检)**:实跑 `--npcs --speed 6 --duration 12 --maptr-ckpt outputs/maptr_ep512.pt
--dump outputs/dumps/lay3.png`:

| 判据 | 结果 |
|---|---|
| 拼图尺寸 | **3726×1170** = 3×1242 × (375+375+420) |
| 逐格 vs 源图(非标签区) | **最大差 0.0**(逐像素完全相同:既没缩放也没裁) |
| 第一行 x | 0 / 1242 / 2484 = 左前 / 前 / 右前 |
| 第二行 x | 0 / 1242 / 2484 = 右后 / 后 / 左后 |
| 第三行 | 第三方 640×360 @ x=1333 + BEV 420×420 @ x=1973(行居中 `(3726−1060)//2`) |
| **地面区域** | 相机图下半 **最大差 0.0**、均值 148.6 / 152.4 / 163.0(不再被裁) |

回归:`tests/test_live_common.py`(**新增 12 用例**)= ① `compose_grid` 尺寸不符必抛
(含"多 1 px 也抛");② `compose_rows` 画布尺寸 = 行宽/行高精确和、**每格逐像素等于源图**
(核心判据"不为了整齐而缩减像素尺寸")、窄行居中/左对齐/空行报错;③ studio 三层行序与用户口径
一致、缺名丢行。`ruff check && ruff format` 通过。

### P-M 环视相机标定修正(nuScenes 口径)+ 七锚自证 + 实时监看槽(2026-09-22 ✅)

> **触发**:用户要求环视 6 相机按 **nuScenes 官方硬件布局**标定 —— 这是**时序建图**(MapQR/MapTRv2
> temporal;StreamMapNet/MapTracker 因非 SOTA 被用户明确降级)的**前置**,且用户设了硬顺序
> **"先修完标定,才讨论时序建图"**。

#### P-M.1 根因:rig 镜像(`yaw_carla = −az_nus` 漏翻)+ pitch/roll 硬编码 0

旧 `official` rig 的偏航是**官方方位角原样抄的正数**,漏了 CARLA/nuScenes 的符号转换
(`yaw_carla = −az_nus`,见 [autodrivedata/utils/geometry.py](autodrivedata/utils/geometry.py) `carla_yaw_to_nus_yaw`)
⇒ **四个侧/后相机左右互换**,pitch/roll 还硬编码 `0`:

| 相机 | 旧值(bug) | 应为(−az_nus) | 偏差 |
|---|---|---|---|
| CAM_FRONT_RIGHT | −55.0 | **+56.40** | — |
| CAM_FRONT_LEFT | 55.0 | **−55.16** | 两者差 **110.3°** |
| CAM_BACK_LEFT | 108.6 | **−108.60** | — |
| CAM_BACK_RIGHT | −110.8 | **+110.79** | 两者差 **217.2°** |
| CAM_FRONT / CAM_BACK | ≈0 / 180 | −0.32 / −179.85 | 近自逆 ⇒ **长期没暴露** |

**为什么长期没被发现**:前/后两台相机在镜像下"看着对"(光轴近自逆),而"能画出图"从来不是
投影正确的证据(与 §5.11f 同一条教训)。**真值改为单点提供**:[autodrivedata/camera_rig.py](autodrivedata/calib/camera_rig.py)
从官方 `calibrated_sensor` 四元数导出 `NUS_CAMERA_RIG`(导出前**归一化** —— 官方存储的四元数
不是单位长度),采集器 / 实时流 / 导出器同源;`official` 更名 **`nuscenes`**。

**修复面**:`collect_surround.py` / `collect_surround_micro.py`(后者 `OFFICIAL_CAMS`→`NUSCENES_CAMS`)、
`carla_common.SENSOR_MOUNTS["CAM_FRONT"]`(占位 `(1.2,0.0,1.65)` → 官方 `(1.7008, 0.0159, 1.5110)`)、
两处 calib dict 里 `sensor2ego` 硬编码的 `0.0, 0.0`、`live_common`/`view_stream`/`live_studio`/
`probe_rig_mount` 的 rig 名与文档。

#### P-M.2 七锚自证探针(`autodrivedata/calib/probe_calib.py` → `outputs/calib_check/report.json`)

**判据全数值,不目检。** 静态 ego、训练口径全分辨率(1242×375)、spawn 6 RGB + 6 depth + LiDAR +
施工锥;`verdict` 七项**全 true**:

| 锚 | 检什么 | 实测 | 判据 |
|---|---|---|---|
| A0 | 实挂光轴方位角 vs 官方 | `max_abs_diff_deg = **7.105e-15**` | 数值一致 |
| A1 | 侧别一致性(挂点 y 与光轴 y 同号) | **4/4 同侧** | 全部同侧 |
| A2 | 世界方向:四个方位锥各自被"该看到"的相机看到 | 四对**全 match**(FRONT 354 px / BACK 254 / LEFT 991 / RIGHT 747) | 集合相等 |
| A3 | LiDAR 平面 × 深度图交叉验证 | 逐相机 median \|e\| **0.00032–0.00090 m** | < 0.1 m |
| A4 | 轴目标物掩膜质心回归主点 | `cx = **620.5**`、`fx_est = 621.6 px`、残差 max **0.200 px** | < 0.5 px |
| A5 | 实挂位姿 vs 规格 | 平移 **3.84e-06 m** / 偏航 **4.49e-05°** | < 1e-3 |
| A6 | 主点锁定 `(w−1)/2` | `cx_est == 620.5` | < 0.5 px |

**★ A3/A4 联合裁决了像素约定 = CORNER**(整个工作流的主干):CARLA 渲染出的栅格**索引 i 的连续
图像坐标恰为 i**,故 `cx = (w−1)/2 = 620.5`、`cy = (h−1)/2 = 187.0`。证据:A3 在 corner 口径下
median \|e\| = **0.0003 m** vs center 口径 **0.023 m**(约 **70×**,六相机一致);A4 用**实例掩膜
索引**中点回归,独立测得 `fx = 621.6 px`(偏差 0.1%)。
**`fx = (w/2)/tan(fov/2) = 621.0` 与 `cx = 620.5` 并存不是矛盾** —— 前者是"半 FOV ↔ 半宽",
后者是索引约定中心。

**按约定给角色分类(防再犯)**:A = 采样 CARLA 渲染栅格(深度/语义/实例分割/RGB)⇒ **必须 corner**;
B = 采样 **torch** 栅格(FPN 特征图 / `grid_sample(align_corners=False)` / gsplat)⇒ 索引 i ↔ 坐标 i+0.5,
故 `(u+0.5)/W*2−1` 是**正确**的;C = 纯绘制(PIL)⇒ 与约定无关;D = 读内参 / fov→fx ⇒ fx 公式
**唯一落点** `CameraIntrinsics.fx`,cx/cy **必须从 K 直读**而不重算;E = 混用索引与连续坐标 ⇒ 真缺陷。

#### P-M.3 全局口径统一

- [autodrivedata/map/mapviz.py](autodrivedata/map/mapviz.py) `intrinsics_from_k` 改为**直读 K 的 cx/cy**
  (旧实现只读 fx、把 cx/cy 丢掉重算 —— 纯缺陷,与主点裁决无关,必须修)
- fov→fx 公式收敛到 `mapviz.calib_from_fov` **全仓唯一落点**(消除 `collect_surround*` 里的内联重复实现)
- **已导出的 KITTI `calib.txt` P2 不动**,只统一代码侧新导出的口径

#### P-M.4 实时监看槽(`live_studio --calib`)+ CAM_BACK 平台边界

`autodrivedata/calib_live.py`(纯值)+ `autodrivedata/sim/live_studio.py --calib`:另挂 6 深度相机(同挂点/同内参/
同分辨率,否则 overlay 无法逐像素对齐)+ `sensor.lidar.ray_cast` ⇒ LiDAR→世界系平面→投影回相机→
按深度残差着色画进各相机槽;`draw_hud` **第二行**报 pooled |e| 与逐路样本数。离线探针与实时槽
**共用同一套色带**(`calib_live.paint_residuals`)。

**验收(640×360,`--speed 8`)**:5/6 相机可用,pooled median **0.00032–0.00044 m**,时序中位数 0.00033 /
最大 **0.00044 m**(行驶中稳定,77 tick 与 63 tick 两次运行同量级),退出后 `nvidia-smi` 回基线、残留进程 0。

- **第一次运行(修复前)**正是**发现 K3 的那次**:`self_occluded` 报 `False` —— 而 CAM_BACK 的
  `near_fraction = 0.195` 明明自遮挡。**判据静默失效,不报错**。
- **修复后复跑**:HUD/报告显式输出 `自遮挡相机 ['CAM_BACK'](近场占比 CAM_BACK 20%)—— 样本不足属
  **平台边界**,不是标定误差`。**这条输出本身就是验收判据**:平台边界必须被显式报出来,
  而不是让读表的人自己从"样本 2 / median –"里猜。

**★ 成本预算(实测)**:平面拟合 ~**100–150 ms/tick**、六相机采样合计仅 ~**9 ms** ⇒ 瓶颈全在拟合,
故 `--calib-refit` 默认 2。**平面是"世界系"的**(描述场景表面,不是"这一帧的点云")⇒ ego 移动几米后
仍成立、可跨 tick 复用,被挡住的点由单侧可见性判据剔掉。可靠最低配置 = voxel 1.0 / 半径 2.0 /
dist<20 / 上限 1500(半径 1.5 或更低 ⇒ 邻域低于 `MIN_PLANE_PTS=12`,**0 个平面合格点**)。

**★ CAM_BACK 自遮挡 = 平台边界,必须显式报,不能读成"标定坏了"**:官方 `CAM_BACK` 挂点
`(x=0.0283, y=−0.0035, z=1.5791)` 只比 CARLA ego **自身车顶**(bbox extent z 0.7745 + location
z 0.7818 ⇒ z≈1.556)高 **0.023 m** ⇒ 相当一部分画面被**自己的车顶**挡住。实测近场(深度 < 0.5 m)
像素占比:**1242×375 下 0.367、640×360 下 0.195**,其余五路 **0.000**。

- **后果**:它可用样本数常年 0–30(其余 50–200),但**残差中位数并不因此变差**(0.0003 m 量级,
  与其它相机同级)⇒ 判据必须是"**样本不足时不许报 median**"(`median_abs = None`,HUD 报"无数据"),
  **不是**"median 大 = 坏标定"。
- **★ 踩坑(实时运行才暴露)**:`self_occluded` 曾写死绝对阈值 `> 0.2` —— 该阈值在 1242×375 下成立、
  在 640×360 下**静默失效**(0.195 < 0.2)。根因是**近场占比随画幅宽高比变**(两者水平 FOV 都是 90°,
  但 640×360 竖直 FOV 大得多 ⇒ 车顶占比小)。修复 = **相对判据** `self_occluded_cameras(stats)`:
  基准取**同批可用相机**近场占比的中位数(典型 0.000),阈值 `max(NEAR_FRACTION_MIN=0.05,
  NEAR_FRACTION_RATIO=10.0 × 基准)`;**不按相机名硬编码** ⇒ 换 ego / 换挂点 / 换分辨率判据自动跟着走。
  回归钉 `tests/test_calib_live.py::TestSelfOccluded::test_flags_at_the_live_resolution_fraction`
  (0.195 这个实测值必须判得出来)。

**交付文件**:`autodrivedata/calib_live.py`(新)、`tests/test_calib_live.py`(新,51 用例)、
`autodrivedata/sim/live_studio.py`、`autodrivedata/sim/live_common.py`(`build_surround_rig(kind=)` + `draw_hud(y=)`)、
`autodrivedata/calib/probe_calib.py`。产物 `outputs/calib_check/{report.json,overlay.png,live.json}`。

**回归测试(本次新增/扩展,三条锁)**:

| 文件 | 新增 | 钉住什么 |
|---|---|---|
| `tests/test_geometry_nus.py` | `TestQuat` +2、`TestNusCameraRigDerivation` +5(**不 skip**) | 官方四元数**非单位长度**(不归一化 ⇒ 闭式解矩阵非正交 >1e-5);`yaw_carla = −az_nus` 到 **1e-9**;平移只翻 y;6DoF **不可**降 yaw-only(最大 \|pitch\| 0.96°/\|roll\| 0.62°);历史字面表偏差 **110.2/111.4/217.2/221.6°**(取**不 wrap** 的原始差 —— wrap 会把 217.2° 折成 142.8° 而丢掉"镜像"这件事) |
| `tests/test_live_common.py` | `TestRigSpec` +4、`TestRigMountDeviation` +5、`TestDrawHudSecondLine` +3 | `rig_spec(nuscenes)` == `NUS_CAMERA_RIG`;legacy 与 nuscenes 的**真实差异形状**(前侧 110°、后侧仅 14–16° ⇒ **"差得多不多"不是判据**,逐相机挂点 + 有无 pitch/roll 才是);`resolve_rig` 显式指定不被文件名覆盖;**`rig_mount_deviation` 规格对账**(矩阵顺序写反 ⇒ 平移爆掉而**偏航仍 ~0**、偏差随 ego 离原点变远而放大、tick 前全 0 陈旧位姿**不许**判成"通过");`draw_hud(y=)` 第二行不动第一行 |
| `tests/test_calib_live.py` | 51 用例(前次会话) | 自遮挡**相对**判据(0.195 实时分辨率实测值也必须判得出来);样本不足 ⇒ `median_abs is None` **不许报假数字** |

#### P-M.5 MapTR 权重处置:全部标废弃,重采重训(重采前置已齐 ✅ —— 见 §P-M.11)

`maptr_ep256 / ep512 / 600 / 1000` **全部废弃** —— 采集数据本身就错(错误 rig),不是训练问题;
**不做数值修补**,重新采集(修正后 rig)+ 重新训练。与 §P-L.1 那条订正框呼应:
本节保留的价值 = **"rig 必须匹配训练数据"** 这条方法论 + 错配代价的量级。

**legacy 路径回归(本次改动没误伤它的证据)**:`view_stream.py --view grid6
--maptr-ckpt outputs/maptr_ep512.pt --maptr-bev --speed 6 --duration 20 --dump` 复跑:

| 判据 | 结果 |
|---|---|
| rig 自检 | `[rig] legacy(--rig auto → 权重 outputs/maptr_ep512.pt)`、实挂 vs 规格 `平移 0.000 m / 偏航 0.000°` |
| 画布 | **1863×374**(= 3×621 × 2×187,该视图的 `--maptr-scale 0.5` 口径,与 studio 三层拼图无关) |
| overlay 品红 vs raw | **6370 px vs 0**(场景本身不含品红 ⇒ 差集判据成立) |
| 6 瓦片覆盖 | 1089 / 1932 / 1544 / 207 / 663 / 935 px(逐格非零) |

⇒ 本次口径改动(主点 corner、`intrinsics_from_k` 直读、`calib_from_fov` 统一)只作用在
**新导出/新采集**路径上;legacy 权重走的是 infos 里已落盘的 K,行为未变。

> **✅ 状态(2026-09-23,§P-M.11 口径冻结)**:重采重训的**前置条件已全部满足** —— 标定三处缺陷全修
> (§P-M.7「声明 ≠ 渲染」/ §P-M.9 wide rig / §P-M.10 挂点原点),`nus_mini` 与 `nus_mini_wide` 已按新口径
> **重采覆盖**,十条判据在两代 rig 上全过。**旧权重(ep256/ep512/600/1000)与旧数据一律不可复用**;
> 新训练数据的采集口径 = §P-M.11 的冻结表。

#### P-M.6 时序建图阶段前置事项(用户已要求过,此处落笔)

硬顺序:**标定 ✅ → 才讨论 online HD mapping**。已定:

1. **实现优先序 = MapQR 或 MapTRv2 的时序建图**;StreamMapNet / MapTracker 是 2023/24 架构、
   已非 SOTA,**用户明确降级**。
2. ~~**数据必须先重采**(§P-M.5)—— 现有一切 MapTR 数据/权重都在错误 rig 下产生,**不可复用**。~~
   **✅ 已完成(2026-09-23,§P-M.11)**:标定冻结 + `nus_mini` 按新口径重采。硬顺序的**第一段已闭合**
   ⇒ 可以开始讨论 online HD mapping(但训练数据是 `nus_mini`(2 帧样例口径),正式训练集仍需按
   §P-M.11 的冻结表**大规模重采**,规模与图池见下条待定项)。
3. 标定侧已备好的接口:逐帧契约 `mapvec_pred/1`(`eval_maptr.py --out-frames`,GT 同文件携带,
   **消费方 AutoLabel 尚未接**);实时 overlay(`view_stream.py --maptr-ckpt ... --maptr-bev`)。
4. 待用户定:时序窗口长度 / 是否引入 ego 运动补偿 / 评估口径沿用 chamfer AP(须**固定
   `--score-thr`**,见红线)。

#### P-M.7 全传感器标定对账:相机表 ✅,其余全 ❌ + 采集器「声明 ≠ 渲染」(2026-09-23 ✅ 已修)

> **触发**:用户追问「现在的标定设置和 nuscenes 每个传感器(摄像头 + LiDAR + RADAR)的
> translation / rotation 一样(或者相差不大)吗?」——**答案是否**。§P-M 只修了**相机声明表**
> 与 `collect_surround` 一条链;本次把 **LiDAR / 5 雷达 / 内参 / 全部采集器**一起对账。
> 对账脚本:`/tmp/audit_final.py`(官方表 vs 本仓表)、`/tmp/audit_fov.py`(判据 ①②)、
> `/tmp/audit_radar_agg.py`(判据 ③),**均从 `nuscenes_mini/` 下跑**(devkit 表在
> `v1.0-mini/`、点云在 `samples/`,路径口径见 §P-M.7.6)。

##### P-M.7.1 相机声明表 ✅(唯一对得上的那一块)

`camera_rig.NUS_CAMERA_CALIBS` vs 官方 n015(singapore-onenorth / hollandvillage 同一套):

| 相机 | Δt(逐分量最大) | Δaz | Δel |
|---|---|---|---|
| CAM_FRONT | 4.6e-05 m | −0.0047° | +0.0022° |
| CAM_FRONT_LEFT | 3.1e-05 m | +0.0040° | +0.0003° |
| CAM_FRONT_RIGHT | 4.8e-05 m | −0.0043° | −0.0017° |
| CAM_BACK | 4.9e-05 m | −0.0025° | −0.0028° |
| CAM_BACK_LEFT | 3.0e-05 m | −0.0016° | −0.0052° |
| CAM_BACK_RIGHT | 3.2e-05 m | +0.0005° | +0.0026° |

⇒ 差值全部来自**手抄时的 4 位小数舍入**,不是另一套标定。**这一块可以不动**。

##### P-M.7.2 ★ `autodrivedata/sim/collect_nus.py`「声明位姿 ≠ 实际渲染位姿」— 新失效模式

`collect_nus.py` **没被 §P-M 的修复扫到**(它既不 import `camera_rig`,也没在修复清单里):

- **相机渲染**:6 台一律 spawn 在 `SENSOR_OFFSET = (1.2, 0, 1.65)`(LiDAR 挂点!),
  yaw 取自**旧的镜像表** `CAM_YAW_OFFSET = {0, +55, −55, 180, +125, −125}`、`pitch = roll = 0`
- **相机声明**:`calib_cameras = {c: NUS_CAMERA_CALIBS[c]}` —— 写的是**官方正确表**

⇒ 落盘的 nuScenes 数据集里,**每一帧图像的位姿与它自己的 `calibrated_sensor` 记录不一致**:

| 相机 | 实际 spawn(挂点 / CARLA yaw) | 声明表(挂点 / az_carla) | Δt | Δyaw |
|---|---|---|---|---|
| CAM_FRONT | (1.2, 0, 1.65) / +0.00° | (1.7008, −0.0159, 1.5110) / −0.32° | 0.5200 m | 0.32° |
| CAM_FRONT_LEFT | (1.2, 0, 1.65) / **+55.00°** | (1.5239, −0.4946, 1.5093) / **−55.16°** | 0.6077 m | **110.16°** |
| CAM_FRONT_RIGHT | (1.2, 0, 1.65) / **−55.00°** | (1.5508, +0.4934, 1.4957) / **+56.40°** | 0.6248 m | **111.40°** |
| CAM_BACK | (1.2, 0, 1.65) / +180.00° | (0.0283, −0.0035, 1.5791) / −179.85° | 1.1738 m | 0.15° |
| CAM_BACK_LEFT | (1.2, 0, 1.65) / **+125.00°** | (1.0357, −0.4848, 1.5910) / **−108.60°** | 0.5153 m | **126.40°** |
| CAM_BACK_RIGHT | (1.2, 0, 1.65) / **−125.00°** | (1.0149, +0.4806, 1.5624) / **+110.79°** | 0.5224 m | **124.21°** |

- **为什么比 §P-M.1 更危险**:§P-M.1 是「表错了」,这次是「**表对了、图错了**」——
  表看上去完全正确(A0 探针若只查表会全绿),错在**渲染**。二者都靠"图看着能出"通过目检。
- **`autodrivedata/sim/collect_surround.py` 是对照组(正确实现)**:line 46 import `NUS_CAMERA_RIG`、
  line 117 逐相机 spawn `(mount, (pitch, yaw, roll))`、line 142 用**同一张表**写 `sensor2ego`
  ⇒ 不存在"布置与落盘两处维护"。**修 `collect_nus.py` 就照它抄。**

##### P-M.7.3 LiDAR ❌:挂点差 0.2865–0.3192 m,旋转靠帧约定而非"对"

| 项 | 官方(n015 / n008) | 本仓(`SENSOR_OFFSET` + 未设 rotation) |
|---|---|---|
| translation | (0.9437, 0, 1.8402) / (0.9858, 0, 1.8402) | (1.2, 0, 1.65) |
| **Δt** | — | **0.3192 m** / **0.2865 m** |
| 光轴 az | −89.883° / −90.031° | 0.000° |
| 俯仰 el | −0.338° / −0.169° | 0.000° |
| up 轴 `R@(0,0,1)` | (0.0242, −0.0058, **0.9997**) / (0.0462, −0.0030, 0.9989) | (0, 0, 1) |
| up 轴倾角 | **1.429°**(n015) | 0° |

**判据(决定性,复现官方 `num_lidar_pts`)**:用官方 R 反投影计数 = 官方标注值 **1.0000**。
**消融表**(20 个 keyframe / GT 合计 65416 点,逐个组合跑)——**这张表推翻了一个想当然的结论**:

| t | R | 比值 |
|---|---|---|
| 官方 (0.9437, 0, 1.8402) | 官方(6DoF) | **1.0000** |
| 官方 | **identity** | **0.1684** |
| 官方 | yaw-only(−89.883°) | 0.9097 |
| **本仓 (1.2, 0, 1.65)** | 官方(6DoF) | 0.8266 |
| 本仓 | identity | 0.0854 |
| 本仓 | yaw-only | 0.8279 |

- **★ 旋转比平移更要紧**:丢旋转(identity)⇒ 只剩 **0.1684**;只丢 pitch/roll(yaw-only)⇒ 0.9097;
  只错平移 ⇒ 0.8266。**"360° 扫描 ⇒ yaw 不可观测"说的是"测不出来",不是"不用写"** ——
  点云存的是**传感器自身系**,devkit 按 `calibrated_sensor.rotation` 解释;不写就等于宣称
  传感器系 = ego 系,把整片点云**绕 z 转了 90°**。**别把这个"不可观测"读成"可以省"**。
- **真正不可观测的只有 yaw 这一个自由度**(360° 旋转对称 ⇒ 扫描图案不变,无法从数据反解);
  pitch/roll 由 up 轴倾角 **1.429°** 决定,是**可观测量**(在 30 m 处 ≈ 0.75 m 高程差)。

##### P-M.7.4 RADAR ❌:平移全对,四个角雷达偏航差 93.89–135.98°

`export/nuscenes.NUS_RADAR_OFFSETS` 的 translation 与官方**逐分量完全相等**(Δt_max = 0.00e+00),
但 yaw 全错:

| 通道 | 官方 az_nus | 正确 yaw_nus(= az_nus) | 本仓 yaw_nus | Δ |
|---|---|---|---|---|
| RADAR_FRONT | +0.200° | +0.200° | 0.000° | 0.20° ✓ |
| RADAR_FRONT_LEFT | +88.360° | +88.360° | **+45.000°** | **−43.36°** |
| RADAR_FRONT_RIGHT | −90.980° | −90.980° | **−45.000°** | **+45.98°** |
| RADAR_BACK_LEFT | +174.410° | +174.410° | **+90.000°** | **−84.41°** |
| RADAR_BACK_RIGHT | −176.110° | −176.110° | **−90.000°** | **+86.11°** |

⇒ 换到 **CARLA 侧**看就是 `collect_nus.RADAR_YAW_OFFSET` 的实际渲染偏航差
**133.36 / 135.98 / 95.59 / 93.89°**(FRONT 0.20°)。

**★ 判据(无框、最干净):每通道自己的点落进自己 FOV 的比例。**
ars408 实测锥 = 方位角 ±38.1° / 俯仰 ±7.0° / range 250 m(采集侧属性扫描自洽,见 `collect_nus` 头注):

| 通道 | 点数 | 官方挂法锥内占比 | 本仓挂法锥内占比 |
|---|---|---|---|
| RADAR_FRONT | 158536 | 0.8607 | 0.8611 |
| RADAR_FRONT_LEFT | 33224 | **0.8135** | **0.0000** |
| RADAR_FRONT_RIGHT | 51384 | **0.8458** | **0.0000** |
| RADAR_BACK_LEFT | 136749 | **0.9076** | **0.0011** |
| RADAR_BACK_RIGHT | 139692 | **0.8996** | **0.0015** |

**★ 踩坑:不要把 `num_radar_pts` 当标定判据**(我一度想这么用,实测推翻)。它是**跨 5 通道求和**
(单通道计数永远对不上),而且**即使用官方 R + 5 通道合并 + 官方框**,复现比值也只有 **0.6279**
(单通道最大之和 0.5814;框放大 2× 才到 0.9070)⇒ 官方那个数不是"盒内点计数"这一件事能复现的,
**别拿它做判据**。LiDAR 侧的 1.0000 是特例(`num_lidar_pts` 确实就是盒内计数),别外推。

##### P-M.7.5 相机内参 ❌:焦距差 1.57×,主点逐通道不同

官方 1600×900(逐通道,`calibrated_sensor.camera_intrinsic`):

| 通道 | fx = fy | cx | cy | HFOV |
|---|---|---|---|---|
| CAM_FRONT | 1266.42 | 816.27 | 491.51 | 64.561° |
| CAM_FRONT_LEFT | 1272.60 | 826.62 | 479.75 | 64.310° |
| CAM_FRONT_RIGHT | 1260.85 | 807.97 | 495.33 | 64.790° |
| CAM_BACK | **809.22** | 829.22 | 481.78 | **89.343°** |
| CAM_BACK_LEFT | 1256.74 | 792.11 | 492.78 | 64.959° |
| CAM_BACK_RIGHT | 1259.51 | 807.25 | 501.20 | 64.845° |

本仓 `_intrinsics_1600x900_fov90()`:**fx = fy = 800、cx = 799.5、cy = 449.5(HFOV 90°)**,
**六路共用一张 K**。

- **这不是笔误**:CARLA 蓝图 `fov` 属性是**水平 FOV**,我们设的是 `"90"`(为了匹配官方
  CAM_BACK 的 89.3°),于是五路"应该是 64.3°"的相机被渲染成 90°。**要真的对齐官方内参,
  必须改蓝图 `fov` 为逐通道 64.31–64.96°(CAM_BACK 89.34°)**,光改 K 会得到"标定说 64°、
  图像是 90°"的**新一版声明 ≠ 渲染**。
- **官方 K 也分两套**(与 §P-M.7.1 的外参同构):n015(singapore-*)一套、n008(boston-seaport)
  另一套,**逐通道 fx/cx/cy 全都不同**(如 CAM_FRONT n015 `(1266.42, 816.27, 491.51)` vs
  n008 `(1252.81, 826.59, 469.98)`;CAM_BACK 更是 `809.22` vs `796.89`)。上表抄的是 **n015**,
  与 §P-M.7.1 的外参同源 —— **不要混着两套抄**。
- **连带**:官方 cx/cy 逐通道不同且**不等于** `(w−1)/2`(792–829 / 479–501 vs 799.5 / 449.5)
  ⇒ 官方这批内参含逐台装配公差。§P-M 的 corner 裁决(`cx=(w−1)/2`)描述的是 **CARLA 渲染栅格**,
  与官方实测 K 是**两件事**,不冲突但也**不可互换**。
- **★ 裁决(2026-09-23 用户定)= 追官方:逐通道 fov + 官方 K**。
  - 官方 jpg **实测就是 1600×900**(`Image.open` 复核,与表里 w/h 一致)⇒ 内参口径对得上,
    改 `fov` **不涉及改分辨率**。
  - 蓝图 `fov` 逐通道设为 **64.310 / 64.561 / 64.790 / 64.959 / 64.845 / 89.343°**,
    `camera_intrinsic` 写**官方 n015 逐通道实测 K**(含其非中心主点)。
  - **注意 K 的主点 ≠ `(w−1)/2`**:这**不违反** §P-M 的 corner 裁决 —— corner 说的是
    **CARLA 渲染栅格**的索引约定;官方 K 是**真实相机**的装配公差实测值。两者角色不同:
    我们**渲染用 CARLA 栅格**(corner 仍然正确),**落盘的 K 用官方实测值**(因为消费方
    devkit / auto3dlabel 按官方口径解释)。**这一条必须写进代码注释,否则下一个人会来"修正"它**。
  - **代价(如实记录)**:六路画面从 HFOV 90° 收窄到 64.3°(CAM_BACK 到 89.3°),
    与 KITTI 线(`carla_common.CAM_ATTRS` 的 1242×375/fov 90)**口径不同** —— 这是刻意的,
    `collect_nus` 服务 nuScenes 消费链,`collect_drive` 服务 KITTI 消费链。
  - **连带动作**:`export/nuscenes._intrinsics_1600x900_fov90()` 要换成**逐通道官方 K 表**
    (函数名里的 `fov90` 随之改名);`NusSample.calib_cameras` 的注释「只作溯源」不再成立
    —— 它现在是**落盘真值**。

##### P-M.7.6 两处一致性缺陷(与标定数值无关,但必须一并修)

1. **`NUS_CAMERA_CALIBS` 两份**:[autodrivedata/camera_rig.py](autodrivedata/calib/camera_rig.py) 与
   [autodrivedata/gt/export/nuscenes.py](autodrivedata/gt/export/nuscenes.py) 各一份,**实测逐字节相等**,
   但 `export/nuscenes.py` **不 import `camera_rig`**(`autodrivedata/sim/collect_nus.py` 也不 import)
   ⇒ 单点来源在 §P-M 建立后**没被接上**,下次改官方表会静默分叉。
2. **`camera_rig` 头注的四元数模长断言对本数据集为假**:头注写「官方四元数不是单位长度
   (模长 0.99994~1.00005),导出前必须归一化」。实测 mini 集 **120 条 `calibrated_sensor`
   的 |q| 全为 1.000000000000**(最大偏离 **2.22e-16**)——那是 IEEE754 的正常舍入,
   不是"官方值非单位"。**真正的来源是我们表里手抄的 4 位小数**
   (如 `CAM_FRONT_LEFT` 的 `|q|−1 = −5.04e-05`)。
   `tests/test_geometry_nus.py::TestQuat::test_quat_normalize_makes_matrix_orthogonal`
   的 `assert np.linalg.norm(q) != pytest.approx(1.0, abs=1e-9)` **之所以通过,靠的正是这个舍入**
   —— 断言本身没错(表确实非单位)、**头注的归因错了**。归一化照做(无害且必要),
   但要把归因改成"手抄舍入"。

##### P-M.7.7 处置裁决(2026-09-23 用户定)

| 项 | 裁决 | 依据 |
|---|---|---|
| 相机内参口径 | **追官方:逐通道 `fov` + 官方 n015 K** | §P-M.7.5;官方 jpg 实测即 1600×900 |
| 旧产物 `outputs/nus_mini`(33M) | **标废弃,重采** | 与 §P-M.5 对 MapTR 权重的处置同口径:数据本身错(相机镜像 + 错挂点 + 雷达偏航错),不修补 |

**★ 标废弃的理由(为什么不能"只补表")**:这两个产物的**表是对的、图是错的**
(§P-M.7.2)⇒ 不存在"改几行标定就能救"的路径;能救的只有**重采**。

**处置落地(2026-09-23)**:`outputs/nus_mini` **已重采覆盖**(36M,canonical 路径 ——
`tests/test_nuscenes_cali_sensors.REPO_MINI` 与 `smoke_radar_collect.sh` 都读它);
`nus_mini_l3`(35M)**已删除**(与 §P-M.5 对旧权重的口径一致:不修补、不删文件)。
修前证据已固化在本节 §P-M.7.2–.5 的审计表与 `outputs/nus_calib_check/report_prefix.json` 里。

##### P-M.7.8 修正清单(本轮开工项,判据全数值)

| # | 改动 | 落点 |
|---|---|---|
| 1 | 相机 spawn 改走 `NUS_CAMERA_RIG`(挂点 + 6DoF 姿态),删 `CAM_YAW_OFFSET` | `autodrivedata/sim/collect_nus.py:57,177-182` |
| 2 | 相机蓝图 `fov` 逐通道(64.310/64.561/64.790/64.959/64.845/89.343) | `autodrivedata/sim/collect_nus.py:65-69`(新增逐通道表) |
| 3 | 雷达 spawn 偏航改官方(`−az_nus`):FRONT −0.20 / FRONT_LEFT −88.36 / FRONT_RIGHT +90.98 / BACK_LEFT −174.41 / BACK_RIGHT +176.11 | `autodrivedata/sim/collect_nus.py:73-79` |
| 4 | `NUS_RADAR_OFFSETS` 的 yaw 改官方 n015 值(弧度) | `autodrivedata/gt/export/nuscenes.py:65-71` |
| 5 | LiDAR spawn 挂点改官方 `(0.9437, 0, 1.8402)` + 旋转 `(pitch −0.338, yaw +89.884, roll −1.388)`;`calib_lidar` 同步 | `autodrivedata/sim/collect_nus.py:175,234-237` |
| 6 | `camera_intrinsic` 换**逐通道官方 n015 K 表**,函数改名(去掉 `_fov90`) | `autodrivedata/gt/export/nuscenes.py:147-156,296` |
| 7 | `NUS_CAMERA_CALIBS` **去重**:`export/nuscenes.py` 改为 `from autodrivedata.camera_rig import NUS_CAMERA_CALIBS` | `autodrivedata/gt/export/nuscenes.py:53-60` |
| 8 | `camera_rig` 头注的四元数归因改成"手抄 4 位小数舍入"(官方原值 |q|=1.000000000000) | `autodrivedata/camera_rig.py:34` |

**★ 第 5 项隐含一个签名改动(别漏)**:`NusSample.calib_lidar` 现在是
`(translation, yaw_nus)` 的 **yaw-only** 二元组,`points_sensor_to_global_nus(..., calib_yaw_nus)` 也只吃 yaw
⇒ **表达不了 LiDAR 的 1.4289° up 轴倾角**。两条路:

- **(推荐)** 把 `calib_lidar` 的第二元从 `float` 改成**四元数**、`points_sensor_to_global_nus` 的
  `calib_yaw_nus` 改成 `calib_quat`;雷达侧调用点传 `g.yaw_to_quat(yaw)`(雷达官方四元数
  pitch/roll **实测精确为 0**,yaw-only 无损)⇒ **一条链、一个函数**,不重复实现。
- **(不推荐)** 新增一个 quaternion 版函数并存 ⇒ 两条链会分叉,违反本仓"单点来源"纪律。

**连带改动(签名一改,这三处必须同 commit 跟上)**:
`autodrivedata/gt/export/nuscenes.py:275-276`(写表时 `_quat(yaw)` → 直写官方四元数)、
`tests/test_export_nuscenes.py:66` 与 `tests/test_nuscenes_oracle_autolabel.py:35`(测试构造点)。

**验收判据(全数值,不许目检)**:

| 判据 | 阈值 | 复现方式 |
|---|---|---|
| 相机实挂 vs 声明(平移 / 偏航) | 平移 **< 1e-3 m**、偏航 **< 1e-3°** | 复用 `live_common.rig_mount_deviation` |
| 雷达实挂 vs 官方 az | 逐通道 **< 0.1°** | 同上 + 官方 az 表 |
| 雷达点落进自身 FOV 的占比 | 五通道**全部 ≥ 0.80**(修前 0.0000/0.0000/0.0011/0.0015) | `/tmp/audit_fov.py` 同口径 |
| LiDAR 复现官方 `num_lidar_pts` | 比值 **= 1.0000**(修前 0.0854) | `/tmp/audit_fov.py` 同口径 |
| 相机内参 vs 官方 n015 | 逐通道 fx/cx/cy **< 0.01 px** | 落盘 `calibrated_sensor.json` 直读 |
| 渲染 FOV vs 蓝图 fov | 逐通道 **< 0.1°**(轴目标物掩膜质心回归 fx) | `probe_calib.py` A4 同法 |

**★ 验收结果(2026-09-23 实测,六条全过)**——复现器 `autodrivedata/calib/verify_nus_calib.py`
(落 `outputs/nus_calib_check/report.json`),重采 `outputs/nus_mini`(2 scene / 3 sample):

| 判据 | 修前 | 修后 | 阈值 | 结论 |
|---|---|---|---|---|
| ① 相机实挂 vs 声明 | 挂点差 **0.5153–1.1738 m**、偏航差 **0.15–126.40°** | 平移 **4.34e-06 m** / 偏航 **3.95e-05°** | < 1e-3 m / < 1e-3° | ✅ 余量 230× / 25× |
| ② 雷达实挂 vs 官方 az | 四路角雷达差 **93.89–135.98°** | 平移 **3.34e-06 m** / 偏航 **2.85e-05°** | 逐通道 < 0.1° | ✅ 余量 3500× |
| ③ 雷达点落进自身 FOV | 0.8611 / **0.0000** / **0.0000** / **0.0011** / **0.0015** | 0.8607 / 0.8135 / 0.8458 / 0.9076 / 0.8996 | 五通道 ≥ 0.80 | ✅ 与官方挂法逐位吻合 |
| ④ LiDAR 复现 `num_lidar_pts` | 0.0854(本仓挂点+单位阵) | **1.0000**(本仓 3 keyframe / 18873 点,挂点+四元数偏差 **0.0**) | = 1.0000 | ✅ 官方锚 1.0000 / 单位阵消融 0.1684 |
| ⑤ 相机内参 vs 官方 n015 | **32.278–472.598 px** | **0.0 px**(六通道) | < 0.01 px | ✅ 逐位相同 |
| ⑥ 渲染 FOV vs 蓝图 fov | 六路共用 90°(与声明 64.3° 差 **25.7°**) | dev **0.0024–0.0672°** | < 0.1° | ✅ |

落盘表 vs 官方(逐通道最大偏差):相机 `Δt ≤ 4.9e-05 m / Δq ≤ 5.0e-05`(仍是 §P-M.7.1 的
**手抄 4 位小数**);**LiDAR 与 5 雷达 `Δt = Δq = 0.0`(逐位相同)** —— 因为它们的表是
官方原值直接落盘,不经手抄。`bash autodrivedata/sim/smoke_radar_collect.sh` 四判据全过(前雷达 x>0 占
1.00、左右雷达同侧 1.00、devkit `RadarPointCloud.from_file` 五通道可读、GT 关联 2/18 命中)。

**★ 判据 ⑥ 的两个实测坑(写进 `verify_nus_calib` 的注释与回归测试)**:
1. **取帧必须每 tick 抽干全部相机队列**。六相机同时 listen 时每 tick 每队列各进一帧;
   只取被测相机那一帧 ⇒ 其余五路积压 ⇒ 轮到它们时读到的是**锥体 spawn 之前**的陈旧帧。
   症状极具迷惑性:六相机里**只有第一个能测出数**,其余五路 `instance_hits=0` 且
   **与距离无关**(看着像"摆不进去",其实不是)。修后 6/6 可测。
2. **Z 不能写死**。地图遮挡随 ego 出生点而变(实测 `CAM_BACK_RIGHT` 同一脚本两次跑:
   一次 Z=20 可用、一次 Z=20/14 都不可用要退到 Z=10)⇒ 改 **Z 阶梯 `(20,14,10,8,6)`**
   取第一档可用样本 ≥ 6,全档不够则如实报 `pass=False` + `reason`,不凑数。
   横移也从固定 5 m 改成 **±0.45·Z**(杠杆臂 ∝ `fx·x/Z`,`x/Z` 已顶到画幅上限)——
   这一项把 `CAM_FRONT_LEFT` 的 fx 从 1292(dev 0.78°)拉到 1272.54(dev 0.0024°)。

**回归测试(新增,必须与代码同一 commit)**:

| 文件 | 钉什么 |
|---|---|
| `tests/test_nuscenes_calib_consistency.py`(新,**不 skip**) | ① `collect_nus.CAM_YAW_OFFSET` / `RADAR_YAW_OFFSET` **不再存在**(防回退到镜像表);② spawn 位姿由 `NUS_CAMERA_RIG` / 官方雷达表**单点导出**(AST 或常量等价断言);③ LiDAR 挂点常量 == 官方 `(0.9437, 0, 1.8402)`;④ 相机 `fov` 表逐通道值 == 由官方 K 反推的 HFOV |
| `tests/test_export_nuscenes.py`(扩) | `calibrated_sensor` 的 `camera_intrinsic` **逐通道**等于官方 n015 表(现在只测了雷达 translation,内参零覆盖) |
| `tests/test_geometry_nus.py`(扩) | 雷达/LiDAR 的 `R_carla = CARLA_TO_NUS @ R_nus @ CARLA_TO_NUS`(对合)推出 `yaw_carla = −az_nus`,与相机同一规则;`camera_rig` 头注归因订正后,四元数非单位的**来源**断言改为"手抄舍入 ≥ 1e-5" |
| `tests/test_nuscenes_cali_sensors.py`(扩) | `test_repo_lidar_quat_is_identity` 现在**测的是缺陷**:应改为"本仓 LIDAR_TOP 四元数 == 官方 n015" |

**文档同步义务**:`docs/fileTree.md`(`collect_nus.py` / `export/nuscenes.py` / `camera_rig.py` 三行的描述要改);
本 §P-M.7 的 🔴 待修 → ✅ 已修;`CLAUDE.md` 的 MapTR/环境表若引用 `nus_mini` 需同步。

##### P-M.7.9 复现命令(判据全可重跑)

```bash
# 修后:六条判据的**唯一复现器**(离线 ③④⑤ + 在线 ①②⑥,落 outputs/nus_calib_check/report.json)
python -m autodrivedata.calib.verify_nus_calib --offline --live
python -m autodrivedata.sim.collect_nus --frames 2      # 重采(需 CARLA)
bash autodrivedata/sim/smoke_radar_collect.sh                           # devkit 直读四判据
```

> **`/tmp/audit_*.py` 三个审计脚本已随重启消失**(它们是修前一次性对账用的)。数字全部
> 固化在 §P-M.7.1–.5 的表与本附录里;要重跑对账,`verify_nus_calib.py --offline` 的
> `criterion_3/4/5` 就是同一口径(且判据 ③ 额外以**官方集**为锚,比 `/tmp` 版更干净)。
> 历史命令留档(已失效):
>
> ```bash
> cd /root/autodl-tmp/Documents/datasets/nuscenes_mini     # 表在 v1.0-mini/,点云在 samples/
> /root/autodl-tmp/envs/autodrivedata/bin/python /tmp/audit_final.py      # §P-M.7.1/.3/.4 的 Δ 表
> /root/autodl-tmp/envs/autodrivedata/bin/python /tmp/audit_fov.py        # §P-M.7.3 LiDAR 比值 + .4 锥内占比
> /root/autodl-tmp/envs/autodrivedata/bin/python /tmp/audit_radar_agg.py  # §P-M.7.4 的 num_radar_pts 反例
> ```

**小结**:「完全一样」**只对 6 个相机的声明表成立**;LiDAR 挂点差 **0.29–0.32 m**、
雷达四路偏航差 **94–136°**、相机内参焦距差 **1.57×**;且 `collect_nus.py` 用**旧镜像 rig 渲染
却写官方表**。修正范围从"环视相机"扩到**全传感器 + 采集器一致性**。
**修后(2026-09-23)**:六条判据全过(见 §P-M.7.8 验收表),LiDAR/雷达落盘表与官方**逐位相同**,
相机表仍是手抄 4 位小数(§P-M.7.1 口径,本轮未动也不需要动)。

##### P-M.7.10 附录:官方 n015 全传感器原值(抄录,免重跑数据集)

> 来源:`nuscenes_mini/v1.0-mini/{calibrated_sensor,sensor,sample_data,sample,scene,log}.json`,
> location = `singapore-hollandvillage`(= n015,与 §P-M.7.1/.5 同源)。**`/tmp` 的审计脚本会随重启消失,
> 上表数字按本附录即可复算,不必依赖它们。**

**相机**(translation 米 / rotation 四元数 wxyz / K):

| 通道 | translation | rotation | fx=fy | cx | cy |
|---|---|---|---|---|---|
| CAM_FRONT | (1.70079118954, 0.0159456324149, 1.51095763913) | (0.4998015430569128, −0.5030316162024876, 0.4997798114386805, −0.49737083824542755) | 1266.417203046554 | 816.2670197447984 | 491.50706579294757 |
| CAM_FRONT_LEFT | (1.52387798135, 0.494631336551, 1.50932822144) | (0.6757265034669446, −0.6736266522251881, 0.21214015046209478, −0.21122827103904068) | 1272.5979470598488 | 826.6154927353808 | 479.75165386361925 |
| CAM_FRONT_RIGHT | (1.5508477543, −0.493404796419, 1.49574800619) | (0.2060347966337182, −0.2026940577919598, 0.6824507824531167, −0.6713610884174485) | 1260.8474446004698 | 807.968244525554 | 495.3344268742088 |
| CAM_BACK | (0.0283260309358, 0.00345136761476, 1.57910346144) | (0.5037872666382278, −0.49740249788611096, −0.4941850223835201, 0.5045496097725578) | 809.2209905677063 | 829.2196003259838 | 481.77842384512485 |
| CAM_BACK_LEFT | (1.03569100218, 0.484795032713, 1.59097014818) | (0.6924185592174665, −0.7031619420114925, −0.11648342771943819, 0.11203317912370753) | 1256.7414812095406 | 792.1125740759628 | 492.7757465151356 |
| CAM_BACK_RIGHT | (1.0148780988, −0.480568219723, 1.56239545128) | (0.12280980120078765, −0.132400842670559, −0.7004305821388234, 0.690496031265798) | 1259.5137405846733 | 807.2529053838625 | 501.19579884916527 |

**LiDAR_TOP**:
- translation = `(0.943713, 0.0, 1.84023)`
- rotation = `(0.7077955119163518, −0.006492242056004365, 0.010646214713995808, −0.7063073142877817)`
- ⇒ CARLA 侧 `(pitch, yaw, roll) = (−0.3380°, +89.8835°, −1.3884°)`(经
  `R_carla = CARLA_TO_NUS @ R_nus @ CARLA_TO_NUS`,往返 max|ΔR| = 1.665e-16)

**雷达**(translation / rotation;pitch=roll 精确为 0):

| 通道 | translation | rotation | az_nus | CARLA yaw |
|---|---|---|---|---|
| RADAR_FRONT | (3.412, 0.0, 0.5) | (0.9999984769132877, 0, 0, 0.0017453283658983088) | +0.200° | −0.200° |
| RADAR_FRONT_LEFT | (2.422, 0.8, 0.78) | (0.7171539204983457, 0, 0, 0.6969148113750004) | +88.360° | −88.360° |
| RADAR_FRONT_RIGHT | (2.422, −0.8, 0.77) | (0.7010337393110511, 0, 0, −0.7131281065471795) | −90.980° | +90.980° |
| RADAR_BACK_LEFT | (−0.562, 0.628, 0.53) | (0.04876260733128922, 0, 0, 0.9988103964848657) | +174.410° | −174.410° |
| RADAR_BACK_RIGHT | (−0.562, −0.618, 0.53) | (0.0339401344459428, 0, 0, −0.9994238676726663) | −176.110° | +176.110° |

**蓝图 fov 逐通道值**(= 由官方 K 反推的 HFOV,`2·atan(800/fx)`):
`CAM_FRONT 64.561365` / `CAM_FRONT_LEFT 64.309754` / `CAM_FRONT_RIGHT 64.789650` /
`CAM_BACK 89.343457` / `CAM_BACK_LEFT 64.959021` / `CAM_BACK_RIGHT 64.844784`

#### P-M.8 覆盖层中文全变豆腐块:字体落点收敛(2026-09-23 ✅)

**症状与裁决**:用户报告 `outputs/calib_check/check_geometry.png` 里"大量方格符替代了原来的字符位",
问是否系统不认中文,**不能解决就用英文替代**。排查结论:**是字体问题、不是编码问题,而且本机可解**
⇒ **不降级英文**(降级会把"中文 HUD 本可正确"这一能力白白丢掉)。

**两条独立成因,只修一条都不够**(这是本条的要点):

1. 本机 `fc-list` 查不到**任何**中文字体(只有 DejaVu / Quicksand / Ubuntu 三族),
   而 [autodrivedata/calib/viz_calib_check.py](autodrivedata/calib/viz_calib_check.py) 绘制时硬写 `ImageFont.truetype(DejaVuSans-Bold)`;
2. **PIL 没有字体回退链** —— `ImageDraw.text()` 只吃单个 `font` 对象(Pillow 12.3.0 无
   `font_chain`/`font_stack`)。**不传 `font=` 就用内置位图字体**,同样整行豆腐、而且只有 ~11 px。
   [autodrivedata/sim/live_common.py](autodrivedata/sim/live_common.py)(HUD/拼图标签,**4 处**)、[autodrivedata/map/mapviz.py](autodrivedata/map/mapviz.py)、
   [autodrivedata/calib/probe_calib.py](autodrivedata/calib/probe_calib.py)、[autodrivedata/sim/carla_common.py](autodrivedata/sim/carla_common.py)、
   [autodrivedata/sim/collect_static_gt.py](autodrivedata/sim/collect_static_gt.py) 原本全属这一类。**"修了 DejaVu 就完事"是错的。**

**判据(全数值,不目检,不依赖非本项目依赖)**:新建 [autodrivedata/utils/fonts.py](autodrivedata/utils/fonts.py)
用**渲染探针**判"这个字体能不能画中文":
`U+10FFFF`(noncharacter,Unicode 永久保留 ⇒ 任何字体都不该有它的字形)渲染到固定画布取像素 SHA-256
= 该字体的 **`.notdef` 签名**(豆腐块的像素指纹);某字符签名与之相同 ⇒ 画出来就是豆腐块。
`has_cjk(path)` 要求 `文相机字` 四个互不相同的汉字给出**四个互不相同的签名**。

- **踩坑 1:`bytes(font.getmask(ch))` 不能当判据**。实测 DejaVu 下 `U+E000`(PUA)的 mask 字节与
  真正缺失的字符**不同**(`68c44148fa` vs `13218`)⇒ 会把 PUA 判成"有字形"。**"渲染到同样大小的画布再比像素"
  两者一致** —— 后者才是"画出来长什么样"的口径。
- **踩坑 2:判据不看文件名、不看文件在不在、不看 `fc-list`**。DejaVu 有 `−`/`°`/`★`,却画不了中文;
  所以"字体文件存在"与"能画这个字符"是两件事(测试里专门钉了这条)。

**字体从哪来**:本机唯一能画中文的是 **CARLA 随包的 Slate 回退字体**
`Engine/Content/Slate/Fonts/DroidSansFallback.ttf`(Apache-2.0)。这不是"顺手引入外部资源":
CARLA 本就是硬依赖(`tools/carla_server.sh`)。候选顺序
「`AUTODRIVEDATA_FONT` → 系统 CJK(Noto/文泉驿…) → CARLA 随包 → DejaVu 兜底」——换机器自动受益。
Droid 仍缺 4 个码位(`−` U+2212 / `∘` U+2218 / `⚠` U+26A0 / `⁻` U+207B),
`sanitize()` 换成等价 ASCII;**本轮真正画进画面的只有 `−`**(`yaw_carla = −az_nus` 与 `(w−1)/2`)。
**退化行为诚实报**:找不到任何含中文字形的字体时,`sanitize()` 把画不出的字符换成 `?`(而不是留豆腐块)并 `warnings.warn` 一次
—— "看起来正常"的假绿比报错更危险。

**改动落点**:新建 `fonts.py`(`font_path`/`has_cjk`/`missing`/`sanitize`/`get_font`/`draw_text`/`width`/`bbox`/`diagnostics`);
[autodrivedata/calib/viz_calib_check.py](autodrivedata/calib/viz_calib_check.py) 删掉 `_font()` 与 `DejaVu` 常量、全部 `d.text` 改走 `fonts.draw_text`;
上列 5 个绘制文件同样迁移。回归钉 [tests/test_fonts.py](autodrivedata/tests/utils/test_fonts.py) —— 最强的一条是
`test_distinct_cjk_chars_paint_distinct_pixels`(两个不同汉字**画布上必须像素不同**;豆腐块下它们逐像素相同),
外加 **AST 根因钉** `test_no_module_draws_with_a_bare_text_call`(不许再出现不带 `font=` 的 `d.text(...)`)。

**排版连带坑(字体一换全暴露)**:

- **Droid 数字等宽但拉丁比例**(size 22 下数字一律 12.125,而 `i=5.672`/`W=19.422`)
  ⇒ 旧 `f"{name:<16}"` 的**空格补位对齐在任何 CJK 字体下都不成立**,数值表改成**显式列 x 坐标**(`anchor="la"/"ra"`);
  `check_geometry.png` 画布宽度随之 1560 → **1600**(表右缘 1538.5)。
- **HUD 底条宽度原来按内置位图字体估**(`7 * len(text) + 8`)—— 换成真字体后必须按**实测宽度**定宽,否则条带与文字不匹配。
- **`tests/test_live_common.py` 的硬编码 `20` 失效**:新字体 size 16 下拼图标签**墨迹底边到 26**,
  `cell[20:, :]` 的逐像素相等断言会挂。改从 live_common 导出 `TILE_LABEL_BOTTOM`(=27)由它推导
  —— **不再有魔数,后续调字体/字号自动跟随**。
- **`前 +x` 轴标签与 `FRONT` 相机标签撞车**(两者以前都是豆腐块,重叠了也看不出来)⇒ 轴标签改画在 `0.78 * r`(轮内)。

**验收**:`ruff check` / `ruff format --check` 均干净;`test_fonts + test_live_common + test_mapviz + test_calib_live + test_probe_calib + test_collect_rig` **151 passed**;
用 CARLA 重新生成 `outputs/calib_check/check_{geometry,raw,overlay,rig_ab}.png` 四张并逐张确认
—— 中文正确、数值表列对齐、`−` 显示为 `-`;实时 HUD 串(旧版 **100% 豆腐**)实测宽 1318.3 px / 条带 1327 px / `missing(hud) == ""`。

#### P-M.9 宽视场 wide rig:挂点后移 + 新 FoV「画幅内零车体像素」(2026-09-23 ✅)

**用户口径**:前三个 55° FoV、左后/右后 110°、后 180°;**挂点后移到车尾/车顶后方**;
**画幅里不能有自身车体像素**。要检查两件事:① 数据上能否对齐 ② 六视角图是否**有重叠区且不含车体像素**;
另外**在 `outputs/calib_check` 出配置图**(传感器名称 / 自车上的位置 / 采集方向与范围,如 `CAM_FRONT 0 ± 22.5 度`)。

**两处离线复算先推翻了原始 spec 的一部分**(纯几何,未动 CARLA):

1. **180° 在针孔下 K 奇异**:`fx = (w/2)/tan(hfov/2)` 在 180° 时 `tan(90°) ≈ 1.6e16` ⇒ `fx ≈ 4.9e-14`,
   `det(K) ≈ 0` ⇒ 投影 `u = fx·x/z + cx` **恒等于主点**(整幅塌成一点),`inv(K)` 不存在。
   devkit / auto3dlabel 两侧都**只读表**(`cam2img ← calib["camera_intrinsic"]`,`lidar2cam = inv(cam_pose)@lidar_pose`)
   ⇒ 55/110/120° 都不影响对齐,**只有 180° 不行**。
   实测 CARLA 渲染侧另有一道崖:**≤138° 忠实渲染;139° 起中央带消失;180° 全黑**(0.9.16 无鱼眼蓝图)。
2. **零车体像素的解析上限**:要求画幅内**每条**射线都有向后分量 ⇒
   `hfov ≤ 2·min(az−90°, 270°−az)`(实现 `camera_rig.max_hfov_no_ego`)。后侧 110° 配官方轴方位角
   108.595°/249.211° 时上限只有 37.19°/41.58° ⇒ **必须同时改轴方位角**,只挪挂点不够。
3. **订正一条外部事实**:官网那三个数(**55/110/180**)是**方位角**,不是 FoV ——
   官方 devkit 内参反推 FoV 是 **64.31–64.96°(五路)+ 89.34°(CAM_BACK)**。
   ⇒ 本 rig 是**自定义标定,不是 nuScenes 官方口径**,故与官方口径**并存**(官方仍是默认)。

**用户三项裁决**:① `CAM_BACK` 封顶 **120°**;② 后三轴改 **145/180/215**(等分后半球);
③ 走 `collect_nus.py --rig wide` **新增分支**,官方口径保持默认且逐位不变。

**定死的 wide rig**(`autodrivedata/camera_rig.py`:`NUS_WIDE_REAR_X=-1.90` / `NUS_WIDE_CAMERA_AZ` /
`NUS_WIDE_CAMERA_FOV`;前三个**逐位等于官方**):

| 通道 | 挂点 CARLA x/y/z | az_nus | FoV | fx(1600×900) | 车体像素 |
|---|---|---|---|---|---|
| CAM_FRONT / FRONT_LEFT / FRONT_RIGHT | 官方值不变 | +0.321 / +55.165 / −56.402° | 55° | 1536.79 | 0 |
| CAM_BACK_LEFT | **−1.900** / −0.485 / +1.591 | **+145°** | 110° | 560.17 | 0 |
| CAM_BACK | **−1.900** / −0.004 / +1.579 | **+180°** | **120°**(非 180°) | 461.88 | 0 |
| CAM_BACK_RIGHT | **−1.900** / +0.481 / +1.562 | **+215°**(记 −145°) | 110° | 560.17 | 0 |

只改两项:后三路 `x → −1.9000`(车身最后点 −1.8527 **之后 4.7 cm**,给 az=90° 的掠射线留余量)、后三路轴方位角。
`y/z` 与 `pitch/roll` 保留官方值 ⇒ 新姿态恰为 `Rz(Δaz_nus)@R_official`,而 `R = Rz(yaw)Ry(pitch)Rx(roll)`
对**左乘** `Rz` 精确保持 pitch/roll,故 `yaw_carla_new = −az_nus_target`(闭式断言,不靠目检)。
wide 的 K 由**渲染反推**(`fx=(w/2)/tan(hfov/2)`、`cx=(w−1)/2`、`cy=(h−1)/2`,corner 约定)——
与官方 rig 用**实测装配 K** 是两种有意的口径;**写盘 `calibrated_sensor` 与 spawn 仍由同一份常量导出**(§P-M.7 的不变量)。

**方位覆盖表**(纯几何,`camera_rig.coverage_table`;不看相机名字典序,按方位角全局排布):

- 盲区 **1.7224°(331.10–332.82)+ 7.3353°(82.66–90.00)+ 6.0984°(270.00–276.10)= 15.1561°= 4.21%**,
  覆盖 **344.84°/360° = 95.79%**;官方 rig 同一口径是 **0 盲区 / 100%**。
- 重叠:后三路互相 **80 / 80 / 40°**;前视对 `CAM_FRONT↔CAM_FRONT_LEFT` 只剩 **0.1561°**。
  两个正侧方盲区是"前视 55° + 后视不越 90°"的**结构性代价**(唯一杠杆是前视加宽:FL ≥ 70° 时 90.165° 恰好接上,
  或后视允许越 90° 而**支付车体像素**)—— 本轮按用户 spec 不动,**如实报出**。

**验收(八条判据,全数值;`autodrivedata/calib/verify_nus_calib.py --rig wide --offline --live` → `outputs/nus_calib_check/report_wide.json`)**:

| 判据 | 结果 |
|---|---|
| ① 相机实挂 vs 声明 | 平移 `4.70e-06 m` / 偏航 `5.28e-05°` |
| ② 雷达实挂(P-M.7 修后口径) | `3.34e-06 m` / `2.85e-05°` |
| ③④ 雷达 FOV / LiDAR `num_lidar_pts` | **与相机无关,原样达标**:0.8135–0.9076 / 1.0000 |
| ⑤ 内参 vs 声明表 | 逐通道 **0.0 px**(标注 `k_source = 由声明 FoV 反推(自洽性锁,非独立锚)` —— 不冒充独立锚) |
| ⑥ 渲染 FOV vs 蓝图 fov | dev `0.0004–0.0588°`(CAM_BACK 120° 也在崖内) |
| ⑦ **画幅内自身车体像素** | 六路 **全 0 px**(instance_seg 里数 ego 自己的 actor id) |
| ⑧ **相邻共视** | 后三路成对可见,共同可见带宽 **66.5 / 66.5 / 29.0°**;前视对 0.1561° < 阈值 ⇒ **如实 `skipped`,不假装测过** |

**⑦⑧ 的实现**(`autodrivedata/calib/rig_check.py`)与**两条边界**:

- ⑦ 用**实例分割**数 `ego.id` 的像素 —— 解析上限那条不等式用的是"盒模型 + 无畸变"两个近似,
  只有渲染侧逐像素计数才是直接证据。**对照**:官方 rig 同一探针实测 `CAM_BACK = 619189 px = 42.9992%`
  (与 P-M.4 记录的 43.00% 逐位复核),其余五路 0 ⇒ **"改前有、改后没有"同一次测量给出**。
  > **⚠️ 该对照已过时(§P-M.10,2026-09-23)**:那个 43% 是**挂点原点修正前**(整套传感器偏前 1.2563 m、
  > CAM_BACK 落在车身中部)的读数。修正后官方 rig 同探针也是**六路全 0 px** ⇒ ⑦ 不再是 wide 的区分度。
  > wide rig 的取舍因此要说清楚:满足用户指定的 FoV(55/110/120),但**覆盖率更差**(95.79% vs 100%)
  > —— 买到的是那个 FoV spec,不是"零车体像素"(官方 rig 修正后同样零)。
- ⑧ 往重叠区**正中摆锥**,两路掩膜都命中才算共视。**锥心必须抬到视轴高度**(+0.45 m):
  官方挂点在车身包络内,侧视相机朝后时射线**擦车顶**(余量 0.03 m),锥体会被自车挡住 ——
  那时读数变成"这一路有没有被自车挡",不再是重叠的证据。
- **★ 方位轴重叠 ≠ 有限距离下的共同可见**(本条是本轮最重要的订正):方位轴重叠是**无穷远**口径,
  而两挂点最远相距 3.5 m、探针在 12 m ⇒ **视差**让同一世界点两路的方位角差最多 **1.7°**。
  实测:官方 `FL↔BL` 方位轴重叠 11.20°,但带外沿那侧的点投回 CAM_FRONT_LEFT 已在画幅外
  (88.97° > 上限 87.31°)⇒ 12 m 处真正共视只剩 **4.838°**;官方 `B↔BL` 的 5.89° 更是**整个消失**。
  故 ⑧ 报的是 `common_band_deg`(沿重叠带用声明内参逐点判两路是否都落画幅内,取最长连续段),
  不是方位轴重叠。**只看方位轴会系统性高估重叠**。

**交付物**(用户要的两方面;`autodrivedata/calib/viz_rig_check.py`,两代 rig 各一套 → `outputs/calib_check/`):

- `rig_layout_{nuscenes,wide}.png`:**配置图**(俯视挂点 + 视锥 / 方位环,重叠橙、盲区红带度数 / 数字表
  `通道 · 挂点 x,y,z · 方位角 · FoV · az ± fov/2`)。纯值落点(`autodrivedata/rigviz.py`,不碰 CARLA 就能出)。
- `views_{rig}.png`:六视角**原生像素**拼图(不缩放不裁剪,§P-L.6 口径)+ 逐格 `az ± fov/2` 与 ego 像素读数
  + 底部**线性方位尺**(逐相机一条泳道 ⇒ 同一竖线穿过的行数 = 该方位被几路覆盖,≥2 即重叠;末行并集 + 红盲区带度数)。
  **车体像素就地染成品红** —— wide 上是空操作(0 px),官方 rig 的 CAM_BACK 下半幅整片品红。
- `report_{rig}.json`:ego 像素逐通道 / 实挂偏差 / 覆盖表 / 共视逐对读数。

**图上踩的三个坑(都已写进测试)**:

- **`360.0 % 360 == 0`**:`_sector_intervals` 把跨 0° 的扇区劈成 `(lo, 360.0)`+`(0.0, …)`,
  而 `x = f(az % 360)` 会把右端折回**最左** ⇒ `ImageDraw.rectangle` 拿到 `x1 < x0` **直接抛 `ValueError`**
  (不是画错,是崩)。wide 的 `CAM_FRONT` 正好是这一路,官方 rig 却碰不到 —— 又一个"只有新 rig 才走到"的分支。
- **`np.asarray(PIL 图)` 是只读视图**:染色时就地赋值抛 `assignment destination is read-only`,
  必须 `np.array` 拷贝 + `Image.fromarray` 回包。**wide 上这条分支从不触发**(0 px),是官方 rig 才第一次走到的
  —— "0 px 的 rig 测不到染色路径"本身就是为什么**两个 rig 都要出图**。
- 另外:所有彩带画在同一行时,先画的会被后画的**整条盖住**(`CAM_BACK_LEFT` 的 90–200° 全在 `CAM_BACK` 的 120–240° 之下),
  彩带与短码一起被盖 ⇒ 改**泳道**。
- **页脚单行超画布被 PIL 静默裁掉**:页脚是单行 `draw_text`,1800 px 画布装不下后面的字,而 PIL **不报错、不告警**
  —— 图上看着像"这行写完了"。根因是**不能按字符数估宽**:同一行里比例拉丁(数字/字母窄)与全宽 CJK 的宽度差 ~2×。
  修法 = `fonts.wrap(text, size, max_width)` 按**实测像素宽**折行(与底条定宽同一把尺),再逐行从底边往上排;
  页脚文字提为模块常量 `rigviz.FOOTER` 以便单测对原文断言"折完每行都装得下"。

**回归钉**:`tests/test_rigviz.py`(覆盖表逐项等于设计预算 / 盲区红弧"有当且有、无当且无" /
`rigviz.azimuth_of` 与 `camera_azimuth_nus` 两套独立实现相等 / 底尺红列数 ∝ Σ盲区度数 且**跨 0° 不崩** /
页脚折行后每行实测宽 ≤ 画布且不压数字表)、`tests/test_fonts.py`(新增 `TestWrap`:折行每行 ≤ 预算 /
不丢字 / 有断点就断在词间 / 无断点才硬断)、
`tests/test_nuscenes_calib_consistency.py`(新增 `TestVerifyRigSelection`:拨模块级 `RIG` 后内参表必须跟着走
—— 防"wide 的验收静默拿官方表去比")。

**范围外**:不动 `live_common.rig_spec` 的 studio rig 注册表(wide 目前只服务 `collect_nus` 线);
不改前三个的 55°/官方轴向(正侧方盲区如实报出,加宽前视是用户的杠杆)。

#### P-M.10 挂点原点修正:整套传感器后移 1.2563 m(后轴对齐)+ 判据⑨⑩(2026-09-23 ✅)

**用户发现**:看 `outputs/calib_check/rig_layout_nuscenes.png` 时指出 **CAM_FRONT 落在车头盖/引擎盖前方**
而非车顶前部,以官方 `nuscenes-cam-fov.png` 为参照(六相机整体大致居中),要求把整套环视挂点
**向车尾方向平移**,使 CAM_FRONT/CAM_BACK 中点落在车体中心线上。

**根因(纯几何,与任何一个标定数值都无关)**——两套系的**原点定义不同**:

- CARLA 车辆 actor 的原点 = **车身长度中点**(a2 实测包围盒 x ∈ [−1.8527, +1.8527]);
- nuScenes 官方标定表的挂点坐标(以及 `ego_pose`)以**后轴中心**为原点。

⇒ 把官方表值当 CARLA 局部坐标直接用 = 整组 **12 路传感器偏前 1.2563 m**。官方 `CAM_FRONT` x = 1.7008
落在 CARLA **+1.7008**(车头 1.8527 附近,即引擎盖上方),而应在 **+0.4445** —— 与用户目检完全吻合。

**平移量 = 后轴对齐 Δx = −1.2563 m**(用户裁决,**不是**"中点规则"的 −0.8646)。中点规则确实能让
CAM_FRONT/BACK 的中点在车体中心,**但会让 ego 原点与真实后轴错位 0.39 m** ⇒ `ego_pose` 与官方 devkit 语义
不再一致(下游 `lidar2cam`/`ego_pose ⊕ calibrated_sensor` 全依赖这个原点)。本项目把"ego 原点 = 后轴"
当**可测量事实**而非约定 ⇒ 必须对齐真后轴。实测(判据⑩,双偏航自解 `W(ψ) = C + R(ψ)·w₀`):
后轴 **−1.2562963447285285** / 前轴 **+1.2502001429339191** / 轴距 **2.506496487662447** /
四轮各自解出的 C 一致到 **1.73e-06 m**(自证:不是拟合出来的一个数)。

**平移范围 = 相机 + 5 雷达 + LiDAR 全套**(用户裁决):刚体平移 12 路同步;只挪相机而留 LiDAR 在原位,
跨模态融合会再次错位 1.2563 m —— 那正是 §P-M.7 修过一次的同一类病。

**落点(单点真值)**:`geometry.NUS_EGO_ORIGIN_X = −1.2563` 是唯一换算口,
`nus_ego_translation()` / `carla_actor_origin_to_nus_ego()` 两个函数的入口;
采集器 / 导出器 / 实时流 / 验收探针 / 配置图全部经它 ⇒ **无第二处手抄**。
wide rig 的后三路 `x` 是 **CARLA 口径的 −1.9000**(车身最后点之后 4.7 cm,§P-M.9),故常量改名
`NUS_WIDE_REAR_X_CARLA`;它落盘的 nus 侧值 = **−0.6437**(= −1.9000 − (−1.2563))。

**两条新判据**(`autodrivedata/calib/verify_nus_calib.py`)——**为什么必须要新判据**:①(实挂 vs 声明)与②(雷达实挂 vs 声明)
都相对**同一个 ego** 比,**在结构上对原点误差是盲的**;①②全绿而整组传感器偏 1.2563 m 是可能的。

- **⑨ 世界系链**:`declared = 落盘表 ⊕ 实测后轴位姿` vs `rendered = CARLA 实挂经共轭`,12 路逐位比;
- **⑩ 独立复测后轴**:**不读常量**,现场用双偏航自解测出后轴 x 再与 `NUS_EGO_ORIGIN_X` 比。
  常量腐化(换 ego 蓝图、换车)会**静默**把整组传感器原点错位 ⇒ 必须每次复测。

**★ ⑨ 第一次跑出来的两个"假故障"与一个真故障**:

1. **六路相机齐刷刷 `119.93–120.07°`** —— 不是"相机装反了",是**相机局部基不同**:
   CARLA 相机局部轴 = (x 前, y 右, z 上),nuScenes 相机局部轴 = (x 右, y 下, z 前)。
   拿 LiDAR/雷达那条 `M·R·M` 去比相机,必得**轮换阵的本征角**(恒 120°,六路一致)。
   修法 = 显式给出 `CARLA_CAM_TO_NUS_CAM`(`= [e1, −e2, e0]`;正交但 **det = −1** —— 局部基翻转,
   与全局基翻转 `M` 相乘**才抵消**成正当旋转,单测同时钉 `det(M·R·C) = +1`)。**LiDAR/雷达不走这条**。
2. **雷达/LiDAR 齐刷刷 `0.0846°`** —— 真故障,但是**第二处「声明 ≠ 渲染」**(与 §P-M.7 同类不同处):
   `ego_pose` 只写了纯偏航,**丢掉实测的悬架俯仰 +0.0642°**(roll −0.0005°;挂传感器前静置 8 tick 收敛)。
   在 1.2563 m 力臂上不是小数。修法 = `geometry.nus_ego_rotation()` 出**全 6DoF** 四元数
   (`ego_pose.rotation` 归它;`carla_yaw_to_nus_quat` 降为纯 yaw 的退化特例,单测锁 pitch=roll=0 时两者相等);
   采集器挂传感器前先让车**静置收敛**(报告里的 `ego_settle`)。
3. **★ 同一个 0.0642° 又暴露一个符号陷阱**:因子分解 `M·Rz(w)·M = Rz(−w)`、`M·Ry(p)·M = Ry(p)`、
   `M·Rx(a)·M = Rx(−a)` **只对右手矩阵成立**,而 `carla_rotation_matrix` 吐的是 **UE 左手口径**
   (纯 pitch 时 `R[2,0] = +sin p`;右手语言里那其实是 `Ry(−pitch)`)。把 pitch/roll 原样代入 ⇒
   正确结果是 `Rz(−yaw)·Ry(−pitch)·Rx(+roll)`,而不是 `Rz(−yaw)·Ry(pitch)·Rx(−roll)`。
   症状极隐蔽:**yaw 项对得上**(所以"看着差不多"),只有 (0,2)/(2,0)/(1,2)/(2,1) 四个元素差
   `2·sin(0.0642°)`。回归 `test_ego_rotation_is_the_matrix_conjugate_not_yaw_only` **按矩阵相等**判
   —— 比四元数向量会被 `±q` 骗过。

**验收**(`autodrivedata/calib/verify_nus_calib.py --offline --live` → `outputs/nus_calib_check/report.json` /
`report_wide.json`;**十条判据全过**,两代 rig 各跑一遍):

| 判据 | 修正前 | 修正后 |
|---|---|---|
| ⑦ 画幅内自身车体像素 | `CAM_BACK = 619189 px = 42.9992%` | **六路全 0 px**(两代 rig 都是) |
| ⑨ 世界系链 · 相机 | `119.93–120.07°`(局部基混用) | 旋转 `1.5e-05–4.0e-05°` / 平移 `3.8e-06–9.5e-06 m` |
| ⑨ 世界系链 · 雷达+LiDAR | `0.0846°`(ego 丢俯仰) | 旋转 `4.2e-06–2.6e-05°` / 平移 `1.2e-06–1.1e-05 m` |
| ⑩ 后轴复测 | —(本轮新增) | `−1.2562963447285285` vs 常量 `−1.2563`(差 `3.7e-06 m`),C 一致 `1.73e-06 m` |
| ① 相机实挂 vs 声明 | — | 平移 `3.88e-06 m` / 偏航 `3.71e-05°` |

⑨ 的 `origin_x_used_m` 用**实测**值(−1.2562963447285285)而**不是**常量 —— 判据自证链不引用被检验的东西。
其余 ②–⑥ 原样达标;wide 的覆盖仍 **95.79% / 盲区 15.156°**,与 §P-M.9 逐位相同 ⇒ **无回归**。

**订正 §P-M.4 的一条结论**:**「CAM_BACK 自遮挡 = 平台边界」是修正前挂点落在车身中部时的读数**。
`live_studio --calib` 重测(51 tick,640×360):**六路 `near_fraction` 全 0.0**(修正前 CAM_BACK 0.367 @1242×375);
pooled 中位 |e| `0.00037 m`,时序中位 `0.00033` / 最大 `0.00044 m`。
CAM_BACK 样本 `n = 11 < 20` ⇒ **报「无数据」而不是 `0.000`**(§P-M.4 既有的口径没变)——这是监看槽的
样本不足,与标定准确度不是一回事。
- 另记一条**与原点无关的平台边界**:`RADAR_FRONT` 落在 a2 前保险杠**外侧 0.30 m**(`+2.1557` vs `+1.8527`),
  因为 a2(3.705 m)比官方用的 Zoe(4.084 m)**短 0.38 m** —— 任何刚体映射都修不掉,**如实报出**。

**交付物**(用户要求的两张图;`autodrivedata/calib/viz_rig_check.py` → `outputs/calib_check/`,两代 rig 各一套):

- `rig_layout_nuscenes.png`:俯视图新增**后轴标记线**(洋红,label「nuScenes 原点 · 后轴」)
  + **空心灰圈 = 修正前挂点位置**(整体后移 1.2563 m 落到后轴线上,一眼看出改了什么;
  CAM_BACK 恰在后轴正上方只差 0.03 m ≈ 2 px ⇒ label 放**车体左侧**不压挂点)。页脚记两套系差 1.2563 m 的来由。
- `views_nuscenes.png`:六视角**原生像素**拼图,逐格读数 `ego px = 0 (0.0000%)`。
- `report_nuscenes.json`:覆盖 `100.00%` / 盲区 `0.000°`(官方口径不变)、ego 像素全 0、
  实挂偏差 `7.58e-06 m / 3.56e-05°`。

**回归钉**:`tests/test_geometry_nus.py`(后轴换算往返 / `CARLA_CAM_TO_NUS_CAM` 正交且 **det = −1**、
与 M 相乘后 det = +1 / 三个列语义(右·下·前)/ 120° 假误差自证 / `nus_ego_rotation` 等于矩阵共轭 /
纯 yaw 退化等价)、`tests/test_nuscenes_calib_consistency.py`(挂点 = 官方表 + `NUS_EGO_ORIGIN_X`;
wide 后三路 x 的 CARLA 口径与 nus 口径**两处都钉**)、`tests/test_export_nuscenes.py`
(`ego_pose.rotation` 走 6DoF,不再经 `yaw_to_quat` 拍平)。相关单测 170 用例全过。

**数据处置**:`outputs/nus_mini` 与 `outputs/nus_mini_wide` **已重采覆盖**(旧版是"表对了、原点错了",
与 §P-M.7 同类:只能重采,不修补);`bash autodrivedata/sim/smoke_radar_collect.sh` 四判据全过。
**改动未提交,待用户手动 `git commit`**。

#### P-M.11 ★ 标定口径**冻结**(2026-09-23 用户裁决:「以后就按照这样进行」)

用户验收 `outputs/calib_check/{rig_layout,views}_nuscenes.png` 后拍板:**本条即标定最终口径**。
后续采集 / 导出 / 训练 / 评测一律引用本节,**不再逐轮重新推导**;新增任何传感器或改 rig 之前,
先读本节的不变量与作废清单。

**冻结配置 —— 只有三个单点真值,其余全部由它们导出(无第二处手抄)**:

| 项 | 冻结值 | 唯一落点 |
|---|---|---|
| 传感器组 | 6 相机 + 5 雷达 + 1 LiDAR,**12 路刚体同移** | `camera_rig.NUS_CAMERA_RIG` / `NUS_RADAR_OFFSETS` / `LIDAR_ROT` |
| **ego 原点** | **后轴中心**,`NUS_EGO_ORIGIN_X = −1.2563`(实测 −1.2562963447285285,判据⑩ 每次复测) | `geometry.NUS_EGO_ORIGIN_X` |
| **ego 姿态** | **6DoF**,含实测悬架俯仰 `+0.0642°`(挂传感器前静置 8 tick 收敛),**不是纯 yaw** | `geometry.nus_ego_rotation()` |
| **相机局部基** | CARLA (前,右,上) → nus (右,下,前);`CARLA_CAM_TO_NUS_CAM` **det = −1**(与全局翻转 `M` 相乘才抵消) | `geometry` |
| **像素约定** | **CORNER**:`cx = (w−1)/2`、`cy = (h−1)/2`;CARLA 栅格索引即连续坐标 | `mapviz.intrinsics_from_k`(直读,不重算) |
| K 来源 | 官方 rig = 官方 n015 **实测装配 K**;wide rig = **渲染反推** `fx = (w/2)/tan(fov/2)` —— 两种口径**有意不同** | `export/nuscenes.camera_intrinsic*` |
| **官方 rig** | **默认口径**(`collect_nus.py` 不带 `--rig`) | `NUS_CAMERA_RIG` |
| wide rig | 并行口径(`--rig wide`;前三个逐位同官方,后三路 `x = −1.90`) | `NUS_WIDE_CAMERA_RIG` |
| 雷达 / LiDAR 落盘表 | 与官方**逐位相同**;旋转由官方四元数 / `−az_nus` **导出**,不手抄 | `collect_nus.py` |

**三条不变量(每次改采集/导出都要过)**:

1. **spawn 与落盘由同一份常量导出** —— §P-M.7 的教训是「**表对了、图错了**」,这类缺陷只查表全绿、
   只目检"图能出"也全绿;**图不能替代 `verify_nus_calib.py`**。
2. **任何"实挂 vs 声明"判据先 tick** —— 快照陈旧(`get_transform()` 在 tick 前全为 0)会假报。
3. 写盘一律经 `paths.project_path()`;产物落 `outputs/`。

**验收口径(唯一)**:`autodrivedata/calib/verify_nus_calib.py --offline --live` —— **十条判据**,**两代 rig 各跑一遍**。
⑨⑩ 是挂点原点误差的**唯一**探针(①② 相对同一个 ego 比,对该误差**结构上盲**);
**判据不达标如实报数字,不调阈值凑过**。

**本口径下作废 / 保留的旧结论**(防下一个人照旧文档"修正"回去):

| 旧结论 | 处置 | 理由 |
|---|---|---|
| 「CAM_BACK 自遮挡 = 平台边界」(§P-M.4) | ❌ **作废** | 那是挂点落在车身中部时的读数;修正后六路 `near_fraction` 全 0.0 |
| wide rig 用 ⑦「零车体像素」对官方 rig 的区分度(§P-M.9) | ❌ **作废** | 修正后官方 rig 同样六路 0 px;wide 的代价是**覆盖率 95.79% vs 100%** |
| 「官网 55/110/180 是 FoV」(§P-M.9) | ❌ **订正** | 那是**方位角**;官方 FoV 实为 64.31–64.96°×5 + 89.34° |
| rig 必须与权重训练数据一致(§P-L.1) | ✅ **保留** | 不受本次修正影响;**全部旧权重仍不可复用**(§P-M.5) |

**本条解锁的任务**:§P-M.5 / §P-M.6 的**重采重训前置条件已全部满足**
(标定冻结 + `nus_mini` / `nus_mini_wide` 重采完毕 + 十条判据全过)——「标定 ✅ → 才讨论 online HD mapping」
这条硬顺序的**第一段闭合**,下游时序建图按 §P-M.6 启动。

#### P-M.12 重训范围与采样口径(2026-09-24,§P-M.11 冻结口径下的执行)

**范围裁决**(用户 AskUserQuestion):数据规模 **~500 帧**、Town10HD_Opt **多 spawn point** 不跨图;
时序窗口 **3 帧 (t, t−1, t−2)**;留出划分 **路线级 + 帧级两种都报**;实现优先序 **MapTRv2 时序版**。

**★ 采样步长必须一起改(本轮最关键的实测发现)**:旧训练集 `outputs/surround_train` 300 帧,
实测**每帧位移中位 0.520 m、累计路线仅 139.2 m**(`sync_mode(delta=0.1)` + TM 70%)。再往上堆帧数
只是把同一条街采得更密;而留出集是**按帧切**的 —— 同一条街上相隔 0.5 m 的两帧几乎相同
⇒ §5.11 记录的「泛化间隙 3.9×」里有相当一部分是**泄漏**,不是真实泛化差距。
改 **stride 5 = 0.5 s/帧 = nuScenes 关键帧率(2 Hz)**,既对齐工业口径又把路线总长拉起来。

**分辨率 1600×900 是几何决定,不是"追官方"**:1242×375 的宽高比 3.31 配官方 64.3° 水平 FoV
⇒ 垂直 FoV 只有 **33.6°**(官方 38.9°)。**换分辨率会改掉投影几何本身**,不是降采样。
`fov → K` 的落点仍是 `mapviz.calib_from_fov`,即 **fx 取官方、主点取 corner `(w−1)/2`** ——
与 `wide` rig 的 `_wide_intrinsics` 同构造。**为什么不落官方 K 的 cx(792–829)**:那是真实相机
装配公差,消费方是 devkit(`collect_nus.py` 写它是对的);我们的图是 CARLA 渲染栅格、中心恒为
`(w−1)/2 = 799.5`,落官方 cx 会让 GKT 的 world→image **逐通道不一致地偏 7–27 px**
(CAM_FRONT_LEFT 最大)⇒ 又是一次「声明 ≠ 渲染」。两条是**角色不同**,不许"统一"。

**采集(阶段 2 ✅,2026-09-24)**:贪心最大最小距离从 Town10HD_Opt 的 155 个 spawn point 选出 5 个
(两两最近 114 m),`--stride 5 --frames 100` 各采一段 → `outputs/surround_v2/seg{0..4}`。

| 段 | spawn index | 坐标 (x, y) | 路线长 | 帧数 |
|---|---|---|---|---|
| seg0 | 44 | (109.5, 89.8) | 260.1 m | 100 |
| seg1 | 14 | (−113.4, −25.8) | 255.8 m | 100 |
| seg2 | 15 | (−56.9, 140.5) | 259.3 m | 100 |
| seg3 | 152 | (57.6, −67.9) | 245.9 m | 100 |
| seg4 | 55 | (−4.0, 28.1) | 259.6 m | 100 |

合计 **1280.7 m**(旧集 300 帧只有 139.2 m,**9.2×**),中位步长 2.63–2.70 m/帧;采集吞吐
**0.93 s/tick**(1600×900,6 路),5 段共 46 min。

**采集器第三处「声明 ≠ 渲染」已修**:`collect_surround.py` 曾是六路共用 `cam_bp` 的 `fov=90`
+ 一表六用的 K,而官方逐通道 FoV 是 64.31–64.96°×5 + **89.34°** ⇒ 渲染视野与官方差 **25°**。
现改为逐相机蓝图 + 逐通道 `NUS_CAMERA_FOV[name]`;`live_common.rig_frame()` 是**实时侧画幅/FoV
的唯一落点**,`legacy` 口径(1242×375/fov90,服务旧权重)一字未动。**绝不改 `carla_common.CAM_ATTRS`**
—— 它被 KITTI 线 / P1 A/B 线 / 静态 GT / 灯态 / `probe_calib` / 3DGS / 双目 / studio 共 16 处引用。

**多段留出(阶段 3.1 ✅)**:`assemble_maptr.py --segs-dir` 收 `seg*`,每帧带 `seg` / `frame_in_seg`,
`token` 改 `{seg}_{i:06d}`(**全局唯一** —— `--out-frames` 拿它做文件名,多段共用会静默覆盖);
`data_path` 前缀由 `roots_and_prefixes()` 决定(**单段前缀空 / 多段 `segK/`**),实测旧单段
`--surround surround_train` 300 帧重组装后**只有 `seg`/`frame_in_seg`/`token` 三个键变**,
`cams`/`ego2global`/`annotation` 逐帧不变。留出划分收敛到 `maptr_impl/dataset.select_frames()`
**唯一落点**(train / eval 共用),段名拼错**报错而不是静默给 0 帧**。

**实测基线(2026-09-24,RTX 4090 48G)**:`train_maptr.py` 自适应 batch **6**(4 路以上与 1242 的
batch 16 对照见下)、**0.343 s/样本** ⇒ 500 帧/epoch = 2.85 min。**128 epoch 是按梯度步数对齐**
旧基线选的:400 帧 × 128 = 51200 样本 = 旧 200 帧 × 256 ep(0.0510);旧 ep512(102400)到 0.0674 时
已裁定"下轮收益靠扩数据"。**长训一律 `--lr-halve 0`**(默认 12 会让 128 ep 后半程 lr 归零,
平台是 lr 死掉不是收敛)。

**两套留出口径**:路线级 = 训练 `--exclude-seg seg4`(400 帧)/ 留出 `--seg seg4`(100 帧,独立路线);
帧级 = `--keep-in-seg 0:80`(400 帧)/ `--keep-in-seg 80:100`(100 帧,**与旧口径同构但偏乐观**,
同街相邻帧仍在)。chamfer AP **一律固定 `--score-thr 0.2`**,另附 `--sweep` 曲线
(红线:单独报一个 mAP 数字而不写阈值 = 无效结论)。留出段仅 100 帧,报数时一并给帧数。

**★ 阶段 4.3 订正:窗口守卫的边界是「切分」不是「段」**(计划里写的是「窗口不跨段边界,跨界处该帧
作为窗口末帧时窗口自动截短」—— 两半都不对):

1. **边界不是段,是切分**。帧级切分画在段**内部**(`frame_in_seg == 80`),而 stride 5 ⇒ 相邻帧只走
   ~3 m、`warp_bev` 近乎恒等 ⇒ 若窗口能跨切分,留出帧 80/81 的历史 79/78 **就在训练集里**,模型可以
   直接复制它背下来的地图。**实测泄漏 8/80 帧**(5 段 × 每段的 80/81 两帧)。这是最贵的静默失效:
   症状只是按 `frame_in_seg` 分箱的 AP 在 80/81 翘起、~83 落回;训练日志(损失在降、留出确实是
   80 帧)和单测都看不出异常。
2. **「自动截短」不行**。截短 ⇒ 该帧用不到 2 帧历史却照样参与训练/评测,时序增益在段首被稀释且
   **样本数悄悄变多**。改成**整帧丢弃 + 报数**:`history_windows(infos, sel, window)` 只在本切分
   自己的池里取前驱,拿不到就丢该帧,`ds.dropped` 由 train/eval 两侧打日志。真实口径 = 留出
   `[80,100)` × 5 段 ⇒ 100 帧降到 **90 个窗口 / 丢 10 帧**(每段的 80/81)。
3. **键必须是 `(seg, frame_in_seg)`,不能用 `frame`**:`frame` 在段缝处**连续无缺口**(0..499,缝在
   99→100)⇒ `infos[idx−1]` 在 idx=100 处会静默解析到 **seg0 的最后一帧**。

**阶段 4 交付(✅ 代码 + 单测,GPU 部分待决策点)**:

| 落点 | 内容 |
|---|---|
| `maptr_impl/temporal.py` | `warp_bev`(`T_prev⁻¹·T_cur`,完整 3D 刚体,float64 链)+ `TemporalFusion`(1×1 conv 残差);两条口径与两条错法的误差律写在模块头注 |
| `maptr_impl/model.py` | `forward` 按 `isinstance(images, list)` 分派;历史帧走 `torch.no_grad()`(= 推理期 memory bank 语义,激活显存 ≈ 单帧 + K−1 个小 BEV);`fusion` 插在 **GKT 与 head 之间 ⇒ head 一行未动**;`proj` 零初始化 ⇒ 第 0 步与单帧模型**逐位相同**(`torch.equal` 钉) |
| `maptr_impl/model.py` | `load_map_weights` 单落点:`unexpected` 一律报错(时序权重跑单帧模型会把 AP 差异显示成"时序没用");`missing` 只有全 `fusion.*` 才放行(单帧 → 时序 = **有意**热启动) |
| `maptr_impl/dataset.py` | `history_windows` 唯一落点(窗口 = 池下标元组,旧 → 新,**末元素 = 目标帧**);`window=1` 返回结构**逐字节不变** |
| `autodrivedata/map/train_maptr.py` / `autodrivedata/map/eval_maptr.py` | `--temporal-window`(**两侧必须同值**)+ `ds.dropped` 报数 |

**两条错法的误差律**(2026-09-24,`outputs/surround_v2` 的 **200 对真实相邻帧**实测,格宽 0.3 m):
**漏转置**(用 `R_prev` 代 `R_prevᵀ`)∝ `2·|w|·sin(ψ_prev)`,中位 **138 格 = 41 m**、95% 分位 227 格
—— 盲点是 `sin(ψ_prev)=0`(**绝对**偏航,不是 Δψ)`:`pose_prev` 全零时 `R_prev = I`,转置与否同解
(**`tests/test_temporal.py` 第一版就是这么假绿的**);**错乘积**(`R_prev·R_curᵀ` 作用在 `p_cur`)∝
`2·|p_cur|·sin(Δψ)`,Δψ 中位 0.46° 时只 ≈ 2 格(直行近乎无害)、转弯时爆炸。⇒ 两条都必须用**非零
绝对偏航**的姿态去钉。

**另有三处测试假红(都是判据写错,不是代码错)** —— 记下来免得下次又重新踩:
① `poses` 传 `(K, 6)` ⇒ `poses[:, j]` **静默取第 j 列**(不是第 j 帧),一路传到 GKT 才报维度错,
堆栈指向 `cam_world_pose` 而不是根因 ⇒ 加 `poses.dim() != 3` 守卫 + 回归用例;② **判梯度不能看
`proj.weight.grad`** —— 零初始化 + head 采样的 BEV 单元恰落在 `relu(cat)==0` 处 ⇒ 权重梯度**恰好为 0**
而偏置梯度 834 ⇒ 判据改用偏置;③ **`num_vec=4` 下 head 对 BEV 免疫**(`rows=1` ⇒ 全部锚点落在 y=−24
画幅边缘)⇒ 「换历史帧输出不变」是结构导致的,不是融合失效(实测 delta **恰好 0.0** vs `num_vec=50`
的 28.3)⇒ 测试改 `num_vec=50` 并加前置断言。

**验证证据**:`ruff check` + `ruff format` 过;`python -m pytest tests/test_temporal.py
tests/test_maptr_select.py tests/test_gkt.py -q` → **72 passed**。CPU 冒烟(此时显存只剩 ~7.4 G):
`train_maptr.py --temporal-window 3` 打 `[data] 窗口 3 丢弃 2 帧(前驱不在本切分内):['seg0_000000','seg0_000001']`
并收敛一步存盘;`eval_maptr.py --temporal-window 3` 出逐类 AP 与 mAP。冒烟产物已删
(`outputs/_smoke_temporal.pt{,.opt}`)。**时序训练本身受阶段 3 决策点约束,尚未启动。**

**⚠️ 用户裁决(2026-09-24)**:**时序暂时不训练**。阶段 4 停在「代码 + 单测就绪」,先把单帧基线的两套
留出数出齐、把阶段 3 决策点走完,再谈时序。**不准把时序训练当"顺手跑的下一步"自动接上。**

**⚠️ 凌晨那次(400 帧)与本次(320 帧)的区别 —— 为什么必须重跑一次**:前者训练集 = seg0–3 的
**全部 100 帧**(400),即 **`[80,100)` 被训过** ⇒ 它只能给路线级留出(seg4)一个数,**帧级 AP 是脏的**;
而阶段 3 决策点恰恰挂在帧级 AP 上。本次 = `--exclude-seg seg4` ∩ `--keep-in-seg 0:80`(320),
**一个模型同时给两套数**:帧级留出 seg0–3 `[80,100)` = 80 帧(与旧口径同构,接续 0.0674)、路线级留出
seg4 = 100 帧(同样没训过,干净)。

**阶段 3 结果:单帧基线两套留出 ✅**(2026-09-24)—— 权重 `outputs/maptr_v2_singleF.pt`
(320 帧 = seg0–3 `[0,80)`,128 ep,`--lr-halve 0`,batch 6,wall **2 h 16 min**)。口径 = chamfer AP、
**`--score-thr 0.2`**、后端 GPU、3 阈值 `{0.5,1.0,1.5} m` 的 precision 均值(**无 recall 项**)。

| 评估集 | 帧数 | pred/gt(divider) | mAP @0.2 |
|---|---|---|---|
| 训练集 seg0–3 `[0,80)` | 320 | 4394/2518(1.74×) | 0.2727 |
| **帧级留出** seg0–3 `[80,100)` | 80 | 1343/907(1.48×) | **0.3043** |
| **路线级留出** seg4(整段未训) | 100 | 1053/871(1.21×) | **0.1114** |
| 对照:400 帧模型 @ seg4 | 100 | 1796/871 | 0.1039 |
| 对照:400 帧模型 @ 帧级留出(**它训过这些帧**) | 80 | 2030/907 | 0.3096 |

阈值扫描(同一口径,`--score-thr` 纯后处理):

| thr | 训练集 | 帧级留出 | 路线级留出 |
|---|---|---|---|
| 0.10 | 0.1786 | 0.2153 | 0.0874 |
| 0.20 | 0.2727 | 0.3043 | 0.1114 |
| 0.30 | 0.3765 | 0.3801 | 0.1062 |
| 0.40 | 0.4673 | 0.4795 | 0.0908 |

**决策点 → 通过**:帧级 **0.3043 ≥ 旧 0.0674**(旧值 = `ep512` 在 `surround_train` 200 帧上的帧级
留出)。⚠️ **这是闸门不是 A/B**:旧值采于**错误 rig** 的 1242×375 数据,新值采于冻结口径的 1600×900
数据 ⇒ 4.5× 的差**不可归因**(标定 / 分辨率 / 逐通道 FoV / 路线长 9.2× / 留出帧密度全变了)。

**★ 三条不许误读的实测(都是本轮直接量出来的)**:

1. **帧级留出 ≠ 泛化 —— 它就是"同街"这一天花板**。留出首帧 80 与训练末帧 79 **同街只隔
   2.65–3.01 m**(四段实测:`seg0 2.78 / seg1 2.99 / seg2 3.01 / seg3 2.65`),而一个**真的训过这些帧**
   的模型(400 帧版)在同一批帧上只有 **0.3096** vs 未训过的 **0.3043** —— **差 0.0053**。
   即「隔 3 m 没训过」与「直接背下来」在该口径下几乎无差别。真泛化看**路线级 0.1114**,独立模型
   佐证 0.1039(差 6.7%),两者差 **2.7×**。
2. **该 AP 口径不是「训练 / 留出差距」的度量**:训练集自身反而更低,且**每个阈值上都低于帧级留出**
   (0.2727 vs 0.3043 @0.2;0.4673 vs 0.4795 @0.4)。机制 = 口径是 **precision 均值、无 recall 项**,
   而模型在见过的输入上**预测更多**(训练集 pred/gt **1.74×** vs 路线级 1.21×)⇒ 分母惩罚更重。
   **故 §5.11 记的「泛化间隙 3.9×」(训练 0.2603 / 留出 0.0674)不能用这个口径复算** —— 它同时混了
   「过预测」与「GT 密度」两个效应,不是泛化能力的差。
3. **帧级留出还叠了一层「GT 密度」混淆**:留出帧本身更密(divider **11.34 实例/帧** vs 训练 7.87 /
   seg4 8.71;centerline 27.3 vs 20.4 / 24.3)。这是**各段尾段(80–100)恰是路口**的采样性质,不是随机
   抽样 ⇒ 同一预测预算下更容易命中。⇒ **0.3043 里含「同街 + 尾部更密」两个上行偏差,方向同向**。

**类别级证据**:`ped_crossing` 在**同街**帧级留出 0.2125(pred 91/gt 99),在**未训路线** seg4
**0.0000**(pred 8/gt 139),400 帧模型在 seg4 上也只有 0.0303(pred 22)⇒ 该类的得分基本来自
「这片区域的人行横道在哪」的**位置记忆**。其余三类在 seg4 上同量级(divider 0.1659 / boundary 0.1125 /
centerline 0.1674)⇒ 掉的主要是稀疏类。

**产物**:`outputs/eval_v2_singleF_{train,frame,route}.log`(各含阈值扫描)+ 权重
`outputs/maptr_v2_singleF.pt`(+ `.opt`)。**下一步(按用户 2026-09-24 裁决)= 不训时序**,先消化本表。

**★ 顺带查出的假警报(已修,2026-09-24)**:凌晨那次日志结尾的 `FAIL:损失卡在 2.314——匹配/loss/坐标
口径存在错误,禁止继续训练` **是误报**。过拟合闸门写的是 `if args.frames > 1: return`,而 `--frames 0`
的语义是「**不截断**」⇒ `0 > 1` 为假 ⇒ **400 帧的多帧训练被判成单帧过拟合锚点**,打印 FAIL 并
`raise SystemExit(1)`。权重在闸门**之前**已存盘 ⇒ 模型没坏,只有收尾信息与退出码是错的 ——
但**假警报会让人以为口径真坏了而白查一轮**,所以照修:判据改成**实际训练样本数**
(`is_single_frame_anchor(len(ds))`,纯谓词便于单测),并加**静态根因钉**(`TestOverfitGate` 用 AST 扫
源码禁止 `args.frames` 与整数字面量比较 —— 只禁这一形态,`args.frames > len(sel)` 的截断越界检查
合法)。回归钉 `tests/test_maptr_select.py`(35 用例过;相关三文件 **74 passed**)。**注意:正在跑的
320 帧用的是修改前的进程**(Python 已加载旧字节码)⇒ 它结束时**仍会打同样的假 FAIL**,读结果按此处理。

#### P-M.13 运行日志三件套:训练/推理脚本统一落 `logs/`(2026-09-27 ✅)

**动因**:包内 54 个可执行入口**全部只 `print()` 到 stdout,没有任何一处 `import logging`**。
留痕靠 shell 里临时 `| tee outputs/xxx.log` ⇒ ① 训练/推理**跑过就无痕**,权重与预测产物无法回溯到
"哪次跑、哪份参数、哪个代码版本";② tee 落点五花八门(`outputs/` 里日志与产物混堆);③ 用户**刚换到
RTX 4080 SUPER 服务器**,显存 12 G → 32 G 会让 `auto_tune_batch_size` 选到**不同的 batch**,新旧 AP
**不可比**,而旧日志里**没记 GPU**;④ 训练日志里**没有 mAP**(训练循环完全不算 AP,AP 只在 `eval_maptr` 里出)。

**用户裁决(AskUserQuestion,四项)**:

| 决策 | 取值 |
|---|---|
| 覆盖范围 | **训练 3 + 推理/评估 13 = 16 个脚本**(不含 `sim/collect_*` 采集器、`map/assemble_*` 数据组装、`calib/*` 探针) |
| 日志形态 | **三件套 `.log` + `.jsonl` + `.json`**(同 stem) |
| 训练期 mAP | **loss 逐迭代 + 训练结束自动评一次 mAP**(不逐迭代算 AP —— AP 口径只能有一份实现) |
| 环境指纹 | git rev + dirty / GPU 型号+显存+CUDA / conda env + python / 完整 argv + 起始 cwd(**四项全要**) |

> **2026-09-27 扩展:覆盖 16 → 18**,补入 `slam/slam_odometry` 与 `slam/slam_backend`(用户指示「要补」)。
> 理由 = 本节的原始动因(「shell 里临时 `| tee` 跑过即无痕」)恰恰在这两个入口上最尖锐:前端 12.6 min、
> 后端 11 min(802 帧闭环数据),而 `--out` 的产物**只在末尾落盘** —— 中断的跑法在盘上零痕迹,
> 连"跑到第几帧"都无从判。接线口径与其他入口一致,**只加登记、不改任何判据**:
> 前端逐帧 `metric()`(frame/rmse/overlap/iters/failed)、后端每 25 关键帧一条累计计数
> (SC 候选 / 先验拒 / 筛出局 / 短路 / 全量 ICP / 过门)+ 把 `LOOP_PRIOR_MAX_M`、`LOOP_SCREEN_OVERLAP`
> 两个**决定候选被砍多少**的常量提成 highlight(`n_loops` 跨版本比较前先看它们,与 AP 看 `batch` 同理)。
> 两个入口的 `main()` 拆成 `main()`(parse + `with runlog.run(...)`)+ `run(args, rl)`(原主体**逐字不动**),
> 与 `train_maptr` 同一手法。采集器 / 数据组装 / 标定探针**仍不在内**。

**设计**:新模块 [`autodrivedata/utils/runlog.py`](../autodrivedata/utils/runlog.py)(`utils/` 准入判据 =
无领域语义、无 carla/torch ⇒ 只用 stdlib 合规;落点 3 → 4)。
`with runlog.run("autodrivedata.<能力>.<模块>") as rl:` 包住 `main()` 主体,其余 `print()` **一字不改**
(靠 tee 进日志),三处登记:`rl.input()` / `rl.artifact()` / `rl.highlight()`。
`logs/<能力>_<模块>_<YYYYmmdd-HHMMSS>.{log,jsonl,json}` + `logs/latest/<能力>_<模块>.<ext>` **相对软链**。

**★ 显式告警(本次动因之一,写进代码注释与 `.json`)**:`batch` 已进 `train_maptr` 的 highlights ——
**换机器 ⇒ 显存变 ⇒ 自适应实测选到不同 batch ⇒ 新旧 AP 不可比**。旧日志里没这条,才让"换到 32 G 机器后
AP 变了"变成**不可归因**的问题。**跨机器比 AP 前先对 batch;batch 不同就别比。**

**本次实测的环境指纹(新机器)**:
`NVIDIA GeForce RTX 4080 SUPER | 32760 MiB | driver 595.91.07 | CUDA 13.2`;
`env=autodrivedata` / `python 3.11.16`;`git b0f39d2 dirty`。

**三条已踩的坑(各有单测钉)**:
1. **`_Tee` 必须 `__getattr__` 全量代理** —— tqdm/ultralytics 会直接问 `isatty()`/`fileno()`/`encoding`,
   只实现 `write`/`flush` 的包装器在**非 tty 跑法**里炸(而"恰好能跑"的假象只在管道里);
2. **退出时恢复构造时抓的那个流对象,不是 `sys.__stdout__`** —— pytest 的 `capsys` 会替换 `sys.stdout`,
   恢复错对象会把**外层捕获永久破坏**;
3. **环境指纹只在头块采一次并复用** —— 初版 `_summary()` 重采了一次,同一跑的 `.json["gpu"]` 报
   `used_mib 2504` 而 `.log` 头块报 `1`。另:`env` 名按 **`envs/<name>` 路径段**反解,
   **`sys.prefix == sys.base_prefix` 在本机是错的判据**(env 真身在数据盘、软链进 `envs/`,两者恒等)。

**口径不变量(硬门槛)**:`train_maptr` 的 `--eval-*` 复用 `eval_maptr.evaluate()`,**绝不把 chamfer AP
抄一份进训练脚本**。重构后重跑帧级留出,`mAP = 0.3043` **逐位未变**(divider 0.3152 / ped_crossing 0.2125 /
boundary 0.3075 / centerline 0.3821 @`--score-thr 0.2 --exclude-seg seg4 --keep-in-seg 80:100`)。

**逐脚本指标的落法按脚本实际口径,不套 mAP 字段凑格式**:`train_maptr` = 逐 epoch `loss`/`cls`/`pts`/`lr` +
尾部一次留出 mAP;`gs/train_3dgs_mini` = 每 200 步 `train_loss`/`val_psnr`(3DGS 是重建,没有 mAP 概念);
`perception/finetune_synth` = **从 AutoLabel 自己的 mmengine 日志解析** per-epoch loss(训练在子进程里跑,
本进程拿不到它的 stdout)。**实测本机 7 份 finetune 日志全部没有 `Epoch(val)` / `mAP` 行**(val 评测未接)⇒
`.json` 里 `eval_mAP = null` 并附 note —— **取不到就如实记 null,不许编成 0**。

**执行中发现并修的自身回归**:`traj/convert_hivt_pt.py` 以**文件路径**在 hivt env 里跑
(`<hivt>/bin/python autodrivedata/traj/convert_hivt_pt.py ...`),此时 `sys.path[0]` = **脚本所在目录**、
项目根不在路径上 ⇒ 新加的 `from autodrivedata.utils import runlog` 在 import 期就 `ModuleNotFoundError`。
已在该脚本内加**项目根 `sys.path` 引导**(与它既有的 HiVT 路径插入同一手法)。**教训**:当时那 16 个脚本里
**唯一一个不在本 env 里跑**的那个,机械改法必然漏 —— 改完必须**真跑**而不是只看 import。

**新机器上的三个环境阻塞(与本次改动无关,不是日志代码引起的;如实记录)**:
1. `gs/train_3dgs_mini`:环境变量 `TORCH_CUDA_ARCH_LIST` 含 `10.3`,而 torch 2.6.0+cu124 **不认识**该 arch
   ⇒ `ValueError: Unknown CUDA arch (10.3)`(`torch.cuda.get_device_capability(0)` 实测**正确**为 `(8, 9)`);
   改成 `TORCH_CUDA_ARCH_LIST=8.9` 后**更深一层**:`fatal error: crt/host_defines.h: No such file or directory`
   —— `/root/miniconda3/include/cuda_runtime_api.h` 是**指向 `/usr/local/cuda-11.8/` 的符号链接**,
   而 nvcc 是 **13.0**,且 `targets/x86_64-linux/include` 不在编译 include 路径上。**两者都是工具链配置问题。**
2. `perception/sem_bev`:`ModuleNotFoundError: No module named 'utils'`(YOLOPv2 的依赖)⇒ 成功路径**本机不可冒烟**。
3. `perception/finetune_synth --dry-run`:AutoLabel 的 `mmdet3d_configs/configs/pointpillars/...` 未落地
   ⇒ `FileNotFoundError`。

上述三者**只影响成功路径冒烟**,错误路径均已验证(`.json` 记 `status="error"` + `.log` 内完整 traceback)。

**待定(机制未证,不得当结论说)**:路线级留出 mAP **0.1114 → 0.1113**(1e-4,只影响 `centerline`)。
阈值扫描与 run-to-run 非确定性**均已排除**;本机 CPU ≡ GPU;git log 只有纯搬文件提交。
**归因为换机器带来的既存差异,但机制(kernel 舍入 vs 匹配并列打破)未证。**

#### P-M.14 MapQR 方法移植进自实现线(2026-09-27,阶段 0 ✅ / 消融未跑)

**用户裁决**:路线 = **移植进自实现线**(不重建已终止的官方栈);范围 = **query + HeightKernelAttention 两部分都要**。

**动因 —— MapQR 与我们同源**:实测 diff(`hdMapGitHub/MapQR` vs 同目录的 `MapTR_maptrv2` 参照物),
它**不是新模型**,是 MapTRv2 上的局部改动,且核心贡献**恰好打在我们自实现线的两处「工程简化」上**:

| 原 `maptr/head.py` | MapQR |
|---|---|
| `q_pt = q_pt + self.bev_proj(sampled)`(固定位置双线性) | deformable(多头 + **可学习采样偏移**) |
| `q_ins = q_ins + self.agg_proj(q_pt.mean(dim=2))` | `flatten(2)` **拼接** + MLP 聚合(不取均值) |
| 无 reference point 位置嵌入 | `pt_query_pos`(逐层 sine 位置嵌入) |

论文自称的 scatter-and-gather 即 **instance query 散成 P 个点查询 → 采样 → 拼接聚回 instance**;
**均值正是抹平同一要素内各点内容差异的那个操作**,所以这条改进对我们对症。

改动量实测(vs MapTRv2):`decoder.py` 409 行、`height_kernel_attention.py` **448 行新文件**、
`maptrv2_head.py` 68、`transformer.py` 45、`bevformer/encoder.py` 23;配置里 MapQR 把
`query_embed_type` 从 `instance_pts` 换成新增的 **`instance`**。

**已交付(阶段 0,全部增量、默认关 ⇒ 关时与原行为逐位一致)**:

| 落点 | 内容 |
|---|---|
| `deform_attn.py`(新) | 可变形注意力原语 + `normalize_ref` / `sine_pos_embed` / `build_sampling_locations`;**两支共用一份实现** |
| `head.py` | `MapTRHead(scatter_gather=)` —— 只做**装配**(按开关选解码器类)+ `match_assign` / `maptr_loss` |
| `decoder.py` / `decoder_mapqr.py` | **两套解码器各写各的文件**(2026-09-27 用户诉求):默认线 MapTRv2 口径 / MapQR scatter-and-gather 口径;依赖**单向**(MapQR 线 import 默认线的 `FFN`,反向禁止,`test_head.py` 有 AST 结构钉) |
| `bevenc.py`(新) | `BEVEncoder`(N 层)= 可变形自注意力 + `HeightKernelAttention`(带 `height_offsets`)+ FFN,插在 GKT 之后 |
| `model.py` | `scatter_gather=` / `bev_encoder=` 两个开关;**与 `temporal_window>1` 互斥** |
| `train_maptr.py` | CLI 五个新参 + 变体写进 `rl.highlight`(与 `batch`/`score_thr` 同处,结论可归属) |
| 单测 | `test_deform_attn.py`(9)+ `test_bevenc.py`(12)+ `test_head.py` 扩展(6) |

**z 跨度取 (−2, 2)**:MapQR nuScenes config 的 `point_cloud_range` 是 `[-15,-30,-2,15,30,2]` ——
其 BEV 网格(x∈[−15,15] × y∈[−30,30],200×100 @0.3 m/px)与我们 `BEV_DEFAULT` **逐位相同**,
只差 z。故引它,不自己发明。

**★ 从源码里查出的三处「不写下来就会静默错」**(全部已加断言或回归钉):

1. **`MSDeformableAttentionKernel` 里 `D == n_heads` 是隐式硬约束**。官方参考点是 **D 个高度锚点**、
   偏移是 **heads 个注意力头**,两者靠**广播对齐**(官方 config 恰是 8 = 8),语义即
   **第 h 个头专门采第 h 个高度锚点**。不成立时广播会拼出 7 维张量,`multi_scale_deform_attn`
   随后因维度不符报错,但**报错点离病根很远**。移植时用断言显式化,并让 `--bev-encoder-heads`
   直接派生 D,从配置上消掉这个组合。
2. **`build_lidar2img` 必须把 ego→world 折进 4×4**。BEV 锚点是 **ego 局部**坐标,而
   `gkt.project_pts` 是两步(`p_w = p_ego @ r_eᵀ + t_e` → `p_c = (p_w − t_cam) @ r_w2cᵀ`)。
   只做第二步、把 ego 局部点当世界点投,**只有 ego 位姿非恒等时才错** —— 玩具夹具位姿恒等,
   两版**数值上恰好同解**,看不出来。**首次实现正是踩了这个**,由「真实位姿 oracle」抓出。
   (推论:`p_c = C2K·R_sᵀ·(p_ego − t_s)`,**ego→图像矩阵本就与 ego 位姿无关** ——
   锚点与相机同挂 ego 上,位姿在复合里必然抵消;这条不变量本身成了回归钉。)
3. **`scale_k` 的 `(w,h)` 与 `feat_shape` 的 `(h,w)` 不同序**,传反不报错、只让投影错一位 ——
   与 GKT 头注记的坑 ②(内参不缩放)同类,已用非方形特征图钉死。

**★ 归一化是最大的静默失效风险**:官方整套数学吃 **归一化 `[0,1]`**,本项目坐标是**米**。
米直接进 `sine_pos_embed` **不报错**,只让正弦频率**别名**、位置嵌入退化成噪声 ——
**症状是「训不动」而不是异常**。故两处(sine 输入 / deformable 参考点)都必须先 `normalize_ref`:
`test_head.py` 用 monkeypatch 钉住**调用点**,`test_deform_attn.py` 钉住**语义**(米与归一化必须产生不同嵌入)。

**实测(阶段 0,RTX 3090 48 G,真实 `surround_v2` 帧)**:

| 配置 | 参数量 | 峰值显存 | 备注 |
|---|---|---|---|
| 基线 bs=4 / bs=2 | 33.2 M | 21.9 / 11.0 GiB | — |
| `scatter_gather` bs=4 | **119.6 M** | 23.2 GiB | 参数涨在 gather 的三段 MLP(`P·C → P·C/2 → C`,官方口径) |
| `bev_encoder` 1/2/3 层 bs=2 | 39.1 M(3 层) | 16.4 / 20.0 / 23.6 GiB | **每层 ≈ +3.6 GiB** ⇒ 这才是降档杠杆 |

**分块实测与假设相反**:`chunk=0` 与 `8192` 同为 23.6 GiB(该帧 `max_len = 5074 < 8192`,只有一块),
而 `4096 / 2048` **反升到** 26.3 / 29.5 GiB。故默认 0,并在帮助文本与模块头注写明
**「要压显存请调层数,别用分块」**。参数保留仅为在别的形状上可复测。

**几何自证**:投影修好后,六路相机各自可见 BEV 单元 7.4%–25.4%(合计 109%,含重叠),
**至少一路可见覆盖 19842/20000 = 99.2%**;`max_len = 5074`。(修前是「只有 CAM_FRONT 看见全部
20000、其余五路全零」—— 那个反常正是发现 bug 的线索。)

**有意分歧清单(与官方不同构,逐条记明,便于日后若重启官方栈时对账)**:
1. `init_reference` 仍是可学习 `nn.Parameter` 锚点(米),**不**用官方的 `Linear(query_pos).sigmoid()`
   —— 后者是归一化空间构造;代价是失去「由 query_pos 条件化生成初值」。
2. 回归输出保持**米**(不用官方 `inverse_sigmoid/sigmoid`)。
3. **BEV query 来自 GKT 输出**(+可学习位置编码),官方的 BEV 嵌入是**可学习参数**。
   GKT 在 MapQR 里对应 `LSSTransform`(图像→BEV 提升),故本模块是**细化**而非提升;
   代价是与官方不同源,好处是 GKT 与其回归测试一字未动。
4. 按 **per-batch** 算每相机可见 query 集合;官方用 `mask_per_img[0]`(bs>1 时官方那份是错的)。
5. 不做 `one2many` 辅助分支、不做 `aux_seg` 辅助分割(不在本次范围)。

**★ 换机器(本次实测发现,影响所有后续数字)**:本会话前段的 runlog 记录 GPU 为
**RTX 4080 SUPER / 32760 MiB / driver 595.91.07 / CUDA 13.2**,本轮实测已是
**RTX 3090 / 49152 MiB / driver 580.82.09** —— 显存 32 G → 48 G,**`auto_tune_batch_size`
选到的 batch 很可能不同**。按 CLAUDE.md 红线:**跨机器 AP 不许直接比,判据看 `logs/*.json`
的 `highlights.batch`,对不上就别比**。

**变体选择与权重归属(2026-09-27 补)**:
- **一键开关** `--variant {maptrv2,mapqr}`(= 同时打 `--scatter-gather` + `--bev-encoder height_kernel`);
  与细粒度开关**互斥**(同时给报错)—— 不定义"谁覆盖谁",消融走细粒度那条路。
- **变体逻辑只此一份**:`maptr/variants.py`(`VARIANTS` 表 + `resolve_variant` + 建模型 + 认权重)。
  `eval_maptr` / `viz_maptr_pred` / `live_common` 里与变体相关的代码**各只有一行**(构造 `MapTR`),
  靠 `MapTR.forward` 的输出契约与变体无关 ⇒ **不按变体复制文件**:`eval_maptr` 若复制一份给 MapQR,
  chamfer AP 就有两份实现,直接违反红线;studio 两份 800–970 行的复制差异只有一行。
- **checkpoint 自带结构说明**:落盘为 `{"state_dict": …, "model_kwargs": …}`。于是
  **`eval`/`viz`/`studio` 不用额外参数就能建对模型**(实测:`--variant mapqr` 训出的权重,
  `eval_maptr --ckpt …` 直接打出「变体 mapqr」并评出 AP);旧**裸 state_dict** 权重照旧可读
  (归为基线是**事实**——开关是今天才加的),只是不带架构参数,须由调用方用 `**overrides` 补
  (补不齐**当场报 length/size mismatch**,不静默)。实测 `outputs/maptr_v2_singleF.pt` 走新路径正常载入(33.2M)。

**下一步(阶段 1/2/3,未启动)**:①只开 query;②只开 BEV encoder;③两者同开。
三个阶段都:单帧 128 ep、与 `maptr_v2_singleF.pt` **同数据同 epoch 同 seed**、
`--score-thr 0.2` 固定、**帧级留出 + 路线级留出双报**、把 `batch` 一起记。
阶段 0 的端到端冒烟已过(三变体各 2 帧 2 ep,loss 正常下降;`logs/*.json` 的
`highlights.scatter_gather` / `bev_encoder` / `bev_encoder_layers` 已能正确归属)。
**注意 `maptr_v2_singleF.pt` 是在旧机器上训的**:阶段 1 之前应先在同一台机器上用**基线配置**
重训一次取得可比基线,否则「变体 vs 基线」的差里混着换机器的效应。

#### P-M.15 HD 地图三格式适配器:opendrive / lanelet2 / apollo(2026-09-27 ✅)

**诉求**:`map/apollo.py` 与 `lanelet2.py`(原为空文件)补成与 [`opendrive.py`](../autodrivedata/map/opendrive.py)
同角色的**地图格式适配器**,接进 mapvec 管线;**默认输出 opendrive,点名 `--lanelet2` / `--apollo` 才换**。

**用户裁决**(AskUserQuestion):
| 项 | 取值 |
|---|---|
| opendrive 那头 | **保持现状**(现有 `{map}_full.json`);新开关只**追加**写新格式 |
| Apollo 形态 | **text-format protobuf**(`base_map.txt` 风格)—— 本机无 `.proto`,免 protoc、零新依赖 |
| 对接点 | **mapvec 管线双向** + **`mapvec_pred/1` 两者都做**(来源字段 + 逐帧地图文件) |
| 几何粒度 | **只到标线级**;文档写明升级到 lane/lanelet 的路径 |

**★ 唯一的语义降级(必须知道)**:Apollo 的 `Map` **没有通用折线要素**(要素是 lane/crosswalk/stop_line/signal/
junction/road…),而我们的源是**标线中心**的六类折线。天然对应只有三类;`divider`/`boundary`/`centerline`
**借 `lane.central_curve` 承载**(即"一条只有中心曲线的 lane"),类名编进 `id.id` —— 保形可往返,但**把标线当车道**。
已写进 `apollo.py` 模块头注 + 本表,不静默。

**升级路径(选项二,已写进两个模块的头注)**:
- **Apollo 真 lane**:给 `centerline` 的 attrs 补 `road_id`/`lane_id`/`s`(`_extract_centerlines` 内部已知,只是没带出来)
  → 用 `opendrive.lane_boundary_t` 重算左右边界填进 `lane.left/right_boundary` → 此时 divider/boundary 不再单独出(否则重复);
- **Lanelet2 真 lanelet**:把 [`mapvec.py`](../autodrivedata/map/mapvec.py) 内部的 `_Edge`(边界线 + 两侧 lane 列表)
  提升为公开模型 → 按 section 为每个 driving lane 找左右边界 → 输出 `relation type=lanelet` + 左右 `member`。

**实现要点(踩过的坑都在这里)**:
- **lanelet2 的 `.osm` 存 lat/lon,我们是米** ⇒ 等距圆柱投影;**origin 写进文件内 XML 注释**
  (外部工具忽略、自己读得回)⇒ 往返**不依赖文件外的隐式约定**;缺 origin 又没显式给就**报错**,不默默按 (0,0)
  算(那会把整图挪到几内亚湾)。经纬度按 12 位小数落盘(≈1e-4 mm)。
- **保真通道**:lanelet2 走自定义 tag + `attrs:<key>`(**重复键**加 `:N` 后缀 —— OSM tag key 必须唯一);
  apollo 无 KV 槽 ⇒ `attrs`/`id`/`src`/**顺序**全编进 `id.id`(保留键 `@id`/`@src` 含 `@`,而 attr 键经 `quote(safe="")`
  会把 `@` 变成 `%40`,**不可能撞名**)。顺序必须编 —— 文件按种类分块写,读时按种类分组会**打乱顺序**。
- **逐类指定几何字段**:Apollo `lane` 只取 `central_curve`;整节点递归收点会把第三方 lane 的
  `left/right_boundary` 点**吸进中心线**(折线悄悄变长,比"丢要素"更难发现)。
- **丢弃必须计数**:单点折线在 OSM 里无合法表达、空几何在 Apollo 里无意义 ⇒ 丢弃 + 写进文件注释
  (`autodrivedata-dropped: N`),不静默。
- **写读两张表**:同一张表做 `--from` 会把写出器当读入器调,症状是"vecs 变成 str"而不是"函数不存在"。

**验证(真实图 Town10HD_Opt,816 实例)**:
- **默认路径逐位不变**:三格式各跑一遍,`{map}_full.json` 的 **sha256 完全相同**
  (`0f96d8f8…c192f`),因为 `vecs_dump` 一字未动;
- **往返**:lanelet2 / apollo 读回均 **816 实例、0 类不符**;apollo 源重跑管线产出的 JSON **逐位复现**原 JSON,
  lanelet2 源最大差 **1e-3 m**(= JSON 自身的 mm 量化步长;实测只有 **2/151311 个坐标分量**差 1 个量化步长 ——
  投影误差 1e-7 m 把恰好落在 `.0005` 边界的值推了过去,不是精度不足);
- **互斥**:`--lanelet2 --apollo` 同时给 ⇒ 报错;
- **逐帧契约**:`--out-frames --map-format apollo` ⇒ `{token}.json`(带 `map_format`) + `{token}.txt`(50 条 GT);
  旧文件(无 `map_format`)仍能 `frame_from_dict` ⇒ **不 bump 版本号**。

**⚠️ 没有官方校验工具**:本机**无 `lanelet2` 包、无 Apollo `.proto`/protoc** ⇒ 格式合法性只能做到
**结构自证 + 我们自己读得回**(`.osm` 能被 ElementTree 解析、元素计数与类计数一致、text-format 能被自写解析器读回)。
**不声称"通过官方工具校验"**。lat/lon 用的是等距圆柱(非 UTM),单张 CARLA 图尺度上可忽略,
**跨图拼接或与真实 GNSS 对齐不适用**。

#### P-M.16 跨图拼接 + 三处实测逼出来的修正(2026-09-27 ✅)

**诉求**:把几张图合并成一张整体道路地图。用户裁决:拼接对象 = **自己设计的分段图 + 官方 Town 互拼(两者都要)**;
顺序 = **先修两个拦路虎再拼**。

**★ 这件事的本质**:CARLA 各 Town 是**独立 UE 关卡、原点任意,彼此没有真值相对位姿** ⇒ 拼官方 Town
**不是配准问题**(没有共享内容可对齐)而是**人为摆位**,placement 表填的是人的决定。只有拼自己设计的分段图时,
接缝处才有真值。**官方 Town 之间摆多近,车道都接不上** —— 那是数据的性质,不是实现缺陷。

**先修的三处(前两处是 P-M.15 那轮带出来的)**:

| # | 问题 | 实测 | 修法 |
|---|---|---|---|
| A | `_extract_objects` 越界 raise,**8/20 张图整张提不出来** | object 轮廓跨路段端点,越界 ≤ **2.44 m**,双向;Town03 1.20 / Town04 1.88 / Town05 0.98 / Town06 2.44 m | `_geo_at` 加 `strict`(默认仍 raise);object 走**有界外推**(上限 10 m)+ 越界幅度写进 attrs |
| B1 | lanelet2 **丢 z** | `1.5/2.5 → 0/0`;而 12 张能提的图里 **10 张有非平凡高程**(Town07 9.4 m → **Town11 791 m**) | 高程走标准 `ele` 属性 |
| B2 | lanelet2 的 `<nd>` **内联坐标**(破坏性偏离) | 真实 reader 只认 `ref` ⇒ 每条 way 读成**空几何**;互操作是**单向**的 | 改标准 `<node>` + `<way><nd ref>` |
| B3 | 我的"往返精确"单测**名不副实** | 写的是 `for (ax, ay, _), (bx, by, _)` —— **故意跳过 z**,丢了也不红 | 改为**含 z 逐分量**比较 + 加标准结构钉 |

**修 A 为什么用外推不用 clamp**:越界最大 2.44 m,clamp 会把轮廓压回端点、斑马线多边形**肉眼可见变形** ——
那是"改数据"不是"修 bug"。外推用本路段几何延拓,几何量级同越界幅度,且**计数上报**(`attrs.s_extrapolated`)。

**修完的实测**:20/20 张图可提取(原 12/20),越界外推实例合计 112 条;lanelet2 **含 z 往返最大差 5e-7 m**;
**默认导出 sha256 逐位未变**(`0f96d8f8…`)** —— 未越界的实例不加 `s_extrapolated`,本来就好的图产出一个 bit 都不动。

**拼接实现**(`map/stitch.py`,纯值):`Placement(dx,dy,dz,yaw)` + `place()`(刚体,**恒等原样返回同一对象**)
+ `stitch()`(并集 / **多图**加 id 前缀 / **跨图**去重)+ `seams()`(接缝候选**报告**)。坐标走**纯平面米系**,
不引入 WGS84(拼接是摆位问题,lat/lon 只会再叠一层等距圆柱近似)。CLI:`export_mapvec --stitch "A=0,0;B=2000,0,0,90"`。

**★ 两条实测逼出来的去重判据(合成数据测不出来,真实图上一跑就露)**:
1. **同图内部一律不去重**。开着"同图也去重"实测会删掉 `Town10HD_Opt` 10 条、`Town01` 16 条 ——
   全是**不同 road 的中心线恰好重合**(实测 `road 90` vs `road 89`,互距 **0.0000 m**)。那是**真实的道路结构**
   (分隔带两侧、被拆成多个 road id 的同一条路)。单张图自身自洽,重复只可能来自**两图在交叠区各导一遍**。
2. **单点要素(信号灯)必须参与去重**。把它排除在外,同一张图复制两份同位会残留 **36/495** 条 —— 而那 36
   恰好就是信号灯的条数。修完正好 495。

另有一处是**既有单测当场抓住的**:重构把哈希键里的 `cls` 丢了 ⇒ 同位置的 divider 与 boundary 被当重复删掉。
`test_different_classes_are_not_deduped` 立刻红了。

**⚠️ 合并图的用途边界(必须写清)**:对 **MapTR 训练基本无用** —— 训练仍是逐帧 `BEV_RANGE`(±15/±30)裁剪,
合并图超窗口。价值在**全图质检 / 可视化 / 对外交付**。已写进模块头注,避免被当成训练数据源。

**没做的部分(如实记)**:plan 里的 `stitch_topology` **没有实现成"接 link"** —— 因为 [`MapVec`](../autodrivedata/map/mapvec.py)
**不携带道路图**(`predecessor`/`successor`/junction 在 `opendrive.Road` 上,提取时没带出来),没有 link 可接。
交付的是 `seams()` **接缝候选报告**(端点配对 + 朝向差),供人判断。要真接拓扑得先把道路图带进 MapVec,那是另一件事。

## 8 遗留缺口

**当前待办(§P-M.11 冻结口径下,按硬顺序)**:

1. **按冻结表大规模重采训练集** —— 现有 MapTR 训练数据(`surround_train` 200 帧 / Town10@spawn0)是
   **错误 rig** 下采的,全部不可复用;新集规模与图池待定(Town10HD_Opt 之外是否入 Town13/15)。
2. **重训 MapTR 自实现线**(`maptr_impl`:GKT + 分层 query)—— 旧 `ep256/ep512/600/1000` 全部废弃,
   基线口径仍用 chamfer AP 且**必须固定 `--score-thr`**(见红线)。**进行中**:单帧基线正在 128 ep 训练
   (`outputs/maptr_v2_singleF.pt`),两套留出(路线级 seg4 / 帧级 `[80,100)`)待评,见 §P-M.12。
3. **时序建图(online HD mapping)** —— 实现优先序 = **MapQR 或 MapTRv2 的时序版本**(StreamMapNet /
   MapTracker 已按用户裁决降级);待定项见 §P-M.6(时序窗口长度 / ego 运动补偿 / 评估口径)。
   **MapTRv2 时序版代码已就绪**(`maptr_impl/temporal.py` + `--temporal-window`,单测 72 项过),
   **训练受阶段 3 决策点约束尚未启动**,见 §P-M.12 阶段 4。
4. **AutoLabel 消费方仍未接**逐帧契约 `mapvec_pred/1`(`eval_maptr.py --out-frames` 已就绪)。

**教程能力侧缺口**:

- **教程 04(4 相机 IPM/单应拼接)**:仓库仍**无像素级 IPM 环视拼接**(`calib.py`/`mapviz.py` 只有标定与
  矢量投影;`sem_bev.py` 走的是 ground_intersection 射线投影,不是单应 warp)。判定:若需与教程 04 逐条对齐,
  这是**唯一剩余缺口**;当前语义 BEV(教程 06)已用射线投影达成同类目的,是否需要补 IPM 待用户定
- **教程 14 阶段 2**:✅ 已交付(2026-09-19)——C++ 位对齐对拍(`autodrivedata/slam/slam_cpp.cpp` +
  `autodrivedata/slam/slam_diff_test.py`,149/149 PASS);ROS 原生栈移植按用户裁决不做(见 P-H)
- **P2-A 后续子项**:像素级分割 GT 需 CARLA `sensor.camera.semantic_segmentation` 合成

## 9 迁移归档:HiVT 复现 + 教程能力执行记录(2026-09-19 自 Plan.md 迁入)

> **为什么在这里**:Plan.md 已冻结为「方案定案 + 历史执行记录」,§5.x 不再新增;本节的三个小节是
> 教程能力线的完整执行细节,**内容原样搬移**(日期/数字/踩坑一字未改),Plan.md 侧只留指针
> (§5.15 → §9.1,§5.16 → §9.2,§5.17 → §9.3)。

### 9.1 HiVT 复现:官方权重 Argoverse 评测(原 Plan.md §5.15,2026-09-16 ✅)

**缘起**:§5.14a 缺口表里"预测+规划"为零——MapTR 是感知(地图矢量),AutoLabel 是
检测(Box3D),缺**多智能体轨迹预测**这一能力面。选 HiVT(arxiv 2202.05882,CVPR2022)
作为对标基线:层次化 Vector Transformer,argoverse-api 数据表示(agent-centric 局部
坐标 / rotation 归一化 / HD map 车道向量化)与本项目 CARLA 环视链**同构可移植**。

**目标**:官方代码 + 官方预训练权重 + 官方验证集,零改动跑通 Argoverse 1.1 验证集
K=6 的 minADE / minFDE / MR,产出可审计日志。

**实测结果**(`/logs/eval_hivt64.log` 行 10-12 / `eval_hivt128.log` 行 7-9):

| 模型 | minADE | minFDE | MR | README 参考 | 偏差 |
|---|---|---|---|---|---|
| HiVT-64 | 0.6869 | 1.0301 | 0.1026 | 0.69/1.03/0.10 | ~0 |
| HiVT-128 | 0.6611 | 0.9692 | 0.0920 | 0.66/0.97/0.09 | ~0 |

**与论文报告在毫厘之间 → 复现成功**,环境/命令/依赖全部沉淀,可复跑。

**环境沉淀**(独立 conda env `/root/autodl-tmp/envs/hivt`,py3.8,CPU 推理不占 GPU):
torch1.8.0 / pl1.5.2 / pyg1.7.2(+scatter/sparse/cluster wheel)/ argoverse-api 1.1.0 /
omegaconf 2.0.6(手动 wheel)。`/logs/hivt_environment.yml` + `requirements.txt` 可复现。

**关键踩坑(全部已解,知识留存)**:
1. **sm_89 架构**:4080 SUPER(Ada)跑 torch1.8.0+cu111 会撞 "no kernel image" → 评测
   走 **CPU**(`CUDA_VISIBLE_DEVICES=""` + `--gpus 0`),HiVT 模型仅 66 万/253 万参数,
   CPU 推理可接受(预处理 35min + 推理 12min)
2. **argoverse-api 老依赖 2026 不可装**:`omegaconf==2.0.6` 被 PyPI yanked、`numpy==1.19`
   等钉死版本已下架 → 手动下载 wheel 解包到 site-packages + `--no-deps` 逐项装其余
3. **S3 数据下载两坑**:官方 bucket `argoai-argoverse` 404(正版在 `argoverse/datasets/av1.1/tars/`);
   aria2 多线程拼出损坏 gzip(S3 Range 分段错位)→ 用户 scp 上传 660MB 完美解决
4. **pl1.5.2 连带缺一堆**:fsspec/deprecate/utils(手写 void 桩)/tensorboard/protobuf/jinja2
   /joblib/networkx...逐个 --no-deps 补

**对项目的意义**:① 简历口径 = "复现官方权重在 Argoverse 验证集评测,minADE/minFDE/MR
与论文一致"而非"我训练的模型";② 获得一个**已验证的预测评测基座**——下一步可把
CARLA 采集的 ego/NPC 轨迹接进同类预测实验(agent-centric 局部坐标、rotation 归一化、
HD map 车道向量化正是 §5.11 A 阶段 xodr 已有数据的同构表示),补齐 §5.14 缺口表的
"预测"能力面;③ minADE/minFDE/MR/brier-minFDE、K=6 多模态口径的度量语义可面试讲清。

### 9.2 CARLA 轨迹 → HiVT 训练管线(原 Plan.md §5.16,2026-09-18 ✅,教程 02 升级)

**交付**:CARLA 采集 → HiVT(TemporalData)→ 训练 → 评估的闭环,补 §5.14a 缺口表"预测"能力面。

- `autodrivedata/sim/collect_traj.py`(每 tick 重发定速,修旧 Town10 后半程停车 bug)+ `autodrivedata/traj/assemble_traj_pt.py`
  (xodr centerline lane 切段,50 帧滑窗)+ `autodrivedata/traj/convert_hivt_pt.py`(dict → TemporalData,edge_index + agent 朝向)
- **数据**:train = Town10 250 场景(4 agents,ego 8m/s 直线);val = Town10 211 场景(同图时间外推)
  - **跨图泛化(Town13)因旧 val 数据静止分布弃用**(ego 3s 位移 2.6m vs train 20m,近零目标压制运动;
    best ckpt = epoch 0)→ 重新采集 Town13 运动段(v2,3 agents,160m)+ 组装 300 场景,但**更优做法是
    同图 val**(已改用 Town10 v3 滑动窗口 211 场景)——轨迹预测的跨图泛化要求场景分布对齐,控布局数据不满足
- **训练**:HiVT-64,CPU 100 epoch(约 25 min);`lightning_logs/version_0/epoch=99-step=3199.ckpt`
- **评估**(逐场景最优 mode,官方口径):minADE **6.08** / minFDE 15.85 / minMR 1.000
  - 诊断:输入尺度正常(0.8m/步),但输出各 mode 终点收敛在 6-8m(恒速 3s=24m 的 ~1/3);
    train reg_loss 每步下降(3.8→0.04)说明过拟合 train;差距主要在**数据规模与多样性**
    (250 场景 × 固定直线布局 vs Argoverse 20 万),非算法/管线问题
- **结论**:管线闭环成立;指标绝对值不刷 SOTA(控布局数据的预测任务无参考意义),
  后续收益靠扩数据(多图/多布局/长尾)而非继续长训

### 9.3 教程能力 P-D~P-G 落地(原 Plan.md §5.17,2026-09-18 ✅,教程 08~16)

**交付**:16 篇 Carla 教程(08 单目/09 双目/12 地面/13 聚类/15 多雷达/16 3DGS)能力在仓库栈上的
批量落地 + 采集器纯函数下沉。路线图 Plan2.md、里程碑 docs/milestone2.md,双文档同步勾选。
> **注**:本节 2026-09-19 自 Plan.md §5.17 迁入(Plan.md 侧只留指针)。教程能力线的后续
> 规划(教程 04 IPM 缺口、教程 14 阶段 2、P2-A 分割 GT 等)见本文件 §3/§8。

- **P-D 单目测距**(教程 08):`autodrivedata/perception/mono_distance.py` + `geometry.mono_depth_from_box`(z=H·fy/框高,H=1.6m)
  + `geometry.box_2d_from_3d`(GT 3D 框 8 角点 → p2 投影 → 前端 u/v min/max,与采集器同投影口径)
  - 修复:label 2D 列 59/97 是零宽退化框 → 检测框改走 GT 3D 投影框("已知位姿投影框"诚实基线)
  - **基线实测**(147 框):全距 mean 8.56%/median 7.8%;**10-20m 带 mean 8.47%、70% 框 <10%** 达标;
    7-10m 贴脸区系统低估(侧向角点拉大框高)如实排除;地平面投影法相机无俯仰不适用(None)
  - **生产口径**(`--detector yolo`):YOLO 检测框 → 配 GT 投影框 IoU 贪心(match_dets_to_gt,conf 降序)
    → **131 命中**/147 GT(漏检 39);mean err 10.53%(vs 基线 8.56,模型框抖动如实上升)、z10_20 36% <10%
  - 产物 `outputs/mono_distance/results.json`(project)/ `results_yolo.json`(yolo)
- **P-E 多雷达标定**(教程 15):`autodrivedata/perception/multilidar.py`(point-to-plane ICP,纯 numpy)+ `autodrivedata/calib/calib_multilidar.py`
  - 修复:收敛 = converged(增量阈值)∧ rmse_final<0.05m ∧ **overlap≥0.6** ∧ **plausible(t<5m、r<30°)**
  - 实测:small(0.1rad/0.1m)**converged**(overlap 0.991、iter 5、恢复 t 0.15m/r 5.7°);
    large(1.2rad/2m)**not_converged**(overlap 仍 0.991,recovered t 10.26m/r 68.8° 被 plausible 否决)
    → RMSE 对平面场景天然低,overlap+合理性双闸分开真伪标定;`outputs/multilidar/icp_result.json`
- **P-F 双目测距**(教程 09):`autodrivedata/sim/collect_stereo.py`(基线 0.4m,y 轴 ±0.2m)+ `autodrivedata/perception/stereo.py`
  (z=f·B/d 三角测量、SGBM 视差可选、NCC 纯 numpy、自监督 reprojection_loss)
  - 实测:定速 5.98 m/s(逐帧自证);SGM 近物点云 z≈5.5m ↔ GT 深度同值;`outputs/stereo/`(40 帧)
- **P-G 3DGS 重建**(教程 16,链路验证):`autodrivedata/sim/collect_3dgs.py`(360° 环绕采集,spectator 归位修复)+
  `autodrivedata/gs/train_3dgs_mini.py`(gsplat mini);位姿用 CARLA 真值(SfM 退化 5.7m/79.6°);psnr_all **17.9**/val 13.56
  - **调优(多俯仰,2026-09-18)**:采集 `--pitches "0,-15,-30"`(落盘分 pitch 目录
    images/p{p}/ + poses_{p}.json + pitches.json,位姿几何仍 collect_rig.ring_cam_pose);
    训练侧跨 pitch 平铺、`--scale` CLI 化、深度下限抽 `_DEPTH_LOWER`。120k 点 × 270 帧:
    **psnr_all 17.9 → 18.32**(3000 iters),**val 帧0 13.56 → 10.45** 不升反降——归因:
    val 帧 0 是无俯仰单视角,多俯仰把观测分给斜视角后该视角覆盖被稀释;单环对照
    在新采集上复现 val~10(采集条件年内变化,pitch 覆盖减半的消融 2×90→1500 iters 仍
    16.34/10.67)。多俯仰价值 = 补顶面/近地隐面(重建完整性,psnr_all ↑),代价 = 留出帧
    视角外推变难。产物 `gaussians_mp3_120k_v2.ply` / `train_result_mp3_120k_v2.json`。
- **累积建图 + 地面 + 聚类**(教程 11/12/13):`autodrivedata/slam/accum.py` / `autodrivedata/perception/ground.py` / `autodrivedata/perception/cluster.py`
  + 对应 bin;150 帧产物齐全,聚类均值 117.97 簇/帧
- **采集器纯函数下沉**(2026-09-18):`autodrivedata/collect_rig.py`(ring_cam_pose 环绕位姿 +
  stereo_rig_offsets 双目挂点;零 carla,AST 纪律守护)→ autodrivedata/sim/collect_3dgs.py / collect_stereo.py 改用,
  剩纯 carla 编排;回归测试先例 tests/test_collect_rig.py(手算锚点 8 passed)
- **结论**:教程能力批量落地完成,7/16 能力达到"链路通+数值如实"(缺 14 FAST-LIO2 外部 ROS 栈);
  P-G 3DGS 调优(多俯仰采集 + --scale/--iters,见上)与 P-D 生产口径评估已完成

## 10 outputs/ 磁盘清理(2026-09-20 ✅)

**背景**:`outputs/` 长到 15 G,其中大量是一次性探针产物与已被取代的数据集。清理脚本
`/root/autodl-tmp/outputs_cleanup.sh`(**项目外**,环境维护用,同 `disk_cleanup.sh` 的先例)
按风险分 A/B/C/D 四层,每层独立确认。**用户实跑 A+B 两层**(C/D 未跑)。

**回收**:数据盘 29 → 30 GiB(脚本口径 1327 MiB;`df` 取整掩盖了零头),`outputs/` **15 G → 13 G**。

### 10.1 已删(逐项 `test -e` 复核,非仅凭执行意图)

| 层 | 项 | 实测 |
|---|---|---|
| A | `dumps` `viz_check` `viz_maptr_e120` | 62 MiB |
| A | `kitti_slam_probe` `kitti_town13_probe` `kitti_town13_static_probe` | 161 MiB |
| A | `maptr_600_pred` + `mapvec_pred_{final,72e_held20}.{json,png}` | 26 MiB |
| A | 官方栈构建日志 / `zzz_probe.txt` / 空 `videos/` | ≈0 |
| B | `kitti_sunset_glare`(老 150 帧对,被 70 帧 A/B 新对取代) | 421 MiB |
| B | `kitti_sweep_day_clear_{4,12}`(**保留 `_8`**) | 534 MiB |
| B | `surround_micro_{legacy,official}`(§P-L.1 证据) | 115 MiB |

### 10.2 未删 —— 以及为什么

- **`kitti_day_clear` 保留、`kitti_sunset_glare` 删掉,这对不对称是刻意的**:
  两者是同一批 150 帧老数据,但 `kitti_day_clear` 仍被 §5 与 `autodrivedata/slam/slam_diff_test.py` 引用。
  代价:`autodrivedata/perception/eval_2d_ab.py` 的老口径配对**已不存在** → 已把该脚本默认值改为 A/B 新对并加注。
- **`kitti_sweep_day_clear_8` 保留**:CLAUDE.md / README / `eval_attr.py` 三处命令示例都用 `_8`,
  删了要同步改三处文档,不值。
- **12 帧天气探针只清一半**:`rain_night` / `dense_fog` 已有 70 帧 A/B 版 → 冗余(C 层,**未跑**);
  `wet_road` / `dense_rush` 是 P1-6 候选、`heavy_rain` / `night_clear` **无** A/B 版(12 帧是唯一数据)→ 全留。
- **`traj_*` 全部保留**:合计 < 1 M,回收量≈0,且 §5 与 `assemble_traj_pt.py` 按名字引用。
- **`surround_town13`(2.3 G)保留**:它是 `maptr_1000.pt` **当前唯一剩下的数据源**
  (`maptr_1000/images` 已于 2026-09-19 删)。删它 = 放弃"多图扩数据"方向,是**取舍**不是清理 ⇒ 单列 D 层 opt-in,默认不做。
- **`kitti_ft` 在脚本 `PROTECTED` 白名单首位**:被 AutoLabel `finetune_config.py:8` 硬编码为 `data_root`。

### 10.3 订正:`maptr_600/images` 也不存在

复核时发现 **`maptr_600/images`(3600 图)同样已缺失**,只剩 `map_infos.json`(33 M)——
2026-09-19 只发现并记录了 `maptr_1000` 被删,`docs/fileTree.md` 里 "infos + images" 的说法对 600 也是错的,已一并订正。

### 10.4 教训

**"删了"必须当场 `test -e` 复核。** 本次 A/B 两层共 16 项,逐项 `test -e` 确认全部消失、
且 11 项保留项(含 `kitti_ft` / `surround_town13` / `kitti_sweep_day_clear_8` / 四个天气探针)全部在位。
这与 §9 归档里 maptr_official "当日未实际执行却写成已删"是同一类错误的两面 ——
**写归档前先跑一遍验证,不是复述执行意图。**

#### P-M.17 MapTRv2 时序融合(K=3)实测:两套留出给出相反结论(2026-09-29 ✅)

**做了什么**:训一个 `--temporal-window 3` 的模型,与单帧基线**同数据同 epoch 同 seed 同 batch(6)**
对照。328 帧(seg0–3 `[0,80)`,窗口丢 8 帧 → 312)/128 ep;欠拟合后**续训 128 ep**(共 256 ep,
`--init-ckpt` 接 Adam 力矩)。权重 `outputs/maptr_v2_k3.pt`(128ep)/ `maptr_v2_k3b.pt`(256ep)。

**结果**(同帧集、固定 `--batch 1`、`--score-thr 0.2`、TF32 已关):

| 集合 | K=1 单帧基线 | K=3(128ep) | **K=3b(256ep)** |
|---|---|---|---|
| 训练集自身(312 帧) | 0.2731 | 0.1994 | **0.4024**(+47%) |
| **帧级留出**(72 帧) | 0.3043 | 0.2263 | **0.4021**(+32%) |
| **路线级留出**(98 帧) | **0.1128** | 0.0892 | 0.1107(−0.0021) |

**三条结论**:

1. **128 ep 时不是"时序无效"而是"没训完"**。证据:训练集自身也落后(0.1994 vs 0.2731)⇒ 欠拟合,
   不是过拟合;loss 末 30 epoch 斜率仍 **−0.0255/epoch** 未趋平;BEV 图上 query 成片重复
   (pred 88 vs GT 17,单帧基线是 28)。续训到 256 ep 后三者全部反转。
2. **两套留出给出相反结论 —— 这是 §P-M.12 那条红线的独立复现**。同一模型:帧级留出 **+32%**、
   路线级留出 **−0.0021**。机制与 §P-M.12 记的一致:帧级留出的首帧与训练末帧**同街只隔 2.65–3.01 m**,
   时序窗口吃的正是这批近重复帧;换到**整段没训过的路线**,收益消失。
3. **路线级那个 −0.0021 落在复现性下限量级**(CLAUDE.md 红线:跨设备/跨机型差 <2e-3 视为不显著;
   且 AP 随 `--batch` 单调漂 ≈5e-5/2batch)⇒ **判为"时序融合在真泛化口径上没有可测收益"**,
   而不是"时序更差"。

**顺带量出的两条口径参数**(已进 CLAUDE.md 红线):

- **TF32**:`cudnn.allow_tf32` 在 torch 2.x **默认 True**,而全仓原本无人设置 ⇒ 归档的 0.3043 是
  **TF32 关**的口径、今天默认开着跑出 **0.3048**(`pred/gt` 计数完全一致,只是匹配数值边界动了)。
  `eval_maptr` 现在入口即 `disable_tf32()`。
- **`--batch` 是第二个必须与 AP 一起引用的参数**(继 `--score-thr`):1/2/4/6 → 0.3043/0.3043/0.3044/0.3045,
  **单调漂** ≈5e-5 每 2 个 batch。故 **评估默认 `--batch 1`**(自适应用的是**空闲**显存,
  会随机器状态变 —— 训练用它是对的、评估用它会让同一权重报出不同的 AP)。

**产物**:`outputs/showcase/06_temporal/bev_k1_k3_k3b.png`(同帧 252 的三态 BEV 对比:
21 / 88 / 28 条预测与贴合度肉眼可辨)。

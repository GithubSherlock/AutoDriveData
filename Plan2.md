# Plan2:Carla 系列教程能力补全路线图(2026-09-19 更新)

> **文档分工(2026-09-19 用户拍板,Plan.md 头部同步声明)**:本文件 = **项目计划的制定地**
> (后续所有新计划、待办、排期都在这里);Plan.md = **方案定案 + 历史执行记录**(§1~§4 契约/架构、
> §5 里程碑归档、§6 骨架、§7 待办快照,均**冻结不再新增**)。
> 起点是「16 篇教程能力 → 本仓库栈补全」的路线图,现已扩展为**全项目计划载体**(§3 执行项、
> §7 执行进度、§8 遗留缺口);红线纪律/接口契约/环境表仍以 Plan.md 为准(§6 复述其要点)。

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
| 02 | YOLOv8 实时检测 | `bin/eval_2d_ab.py`(YOLO11s 2D AP)+ `attribution.py` | 已有(离线评估) |
| 03 | 4 相机安装 + 标定网格 | `collect_surround.py` + `SENSOR_MOUNTS`(**6 相机超集**);纯函数下沉 `autodrivedata/collect_rig.py`(`ring_cam_pose` / `stereo_rig_offsets`) | 已有 |
| 04 | 4 相机标定 + BEV 环视拼接(单应/IPM) | `calib.py` + `mapviz.py` 有标定/矢量投影;`sem_bev.py` 走 ground_intersection 射线投影;**无像素级 IPM 拼接** | 缺口(见 §8) |
| 05 | 实时图像语义分割(SegFormer / YOLOPv2) | `bin/sem_bev.py`(YOLOPv2 检测+车道线+可行驶;YOLO11s-seg 实例掩膜) | ✅ |
| 06 | BEV + 语义分割融合 | `bin/sem_bev.py` 像素级语义 BEV(ground_intersection 投影 + 世界→ego 旋转) | ✅ |
| 07 | 相机+LiDAR 融合,点云→图像 | `calib.world_to_img` / `tr_velo_to_cam` + `geometry.py` 完整投影链(含单测) | 已有 |
| 08 | 单目测距(4 法) | `bin/mono_distance.py` + `autodrivedata/mono_depth.py` / `geometry.mono_depth_from_box`(迭代深度法)+ `box_2d_from_3d`(GT 3D 投影框基线) | ✅ |
| 09 | 双目测距(视差) | `bin/collect_stereo.py` 双目 rig + `autodrivedata/stereo.py`(SGBM/NCC/三角测量) | ✅ |
| 10 | 上帝视角可视化(OpenDRIVE+NPC) | `view_stream.py --view top` + `opendrive.py` + `mapviz` | 已有 |
| 11 | LiDAR+语义建点云地图 | `autodrivedata/accum.py` + `bin/build_accum_map.py`(多帧累积,时序证据加权) | ✅ |
| 12 | 点云地面提取 | `autodrivedata/ground.py` + `bin/extract_ground.py`(RANSAC 平面拟合) | ✅ |
| 13 | 点云障碍物检测(聚类) | `autodrivedata/cluster.py` + `bin/cluster_obstacles.py`(欧氏聚类) | ✅ |
| 14 | FAST-LIO2 + SC-PGO SLAM | `autodrivedata/slam.py`(纯 numpy 两段式降档:帧间点面 ICP 前端 + ScanContext 回环/PGO 后端)+ `bin/slam_odometry.py` + `bin/slam_backend.py` + `bin/slam_cpp.cpp`(阶段 2 位对齐对拍) | ✅(阶段 1+2) |
| 15 | 多激光雷达标定 | `autodrivedata/multilidar.py`(point-to-plane ICP + overlap/plausible 判据)+ `bin/calib_multilidar.py` | ✅ |
| 16 | 3DGS 重建 | `bin/collect_3dgs.py`(环绕采集)+ `bin/train_3dgs_mini.py`(gsplat 训练) | ✅(链路) |

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
  `bin/slam_cpp.cpp`(C++17 单文件零依赖)独立实现前端,`bin/slam_diff_test.py` 在**同一
  `(prev, cur, init, seed)`** 下逐帧比对单次 ICP 的 T → 149/149 PASS(最差 1.9e-14 rad /
  7.0e-12 m),证明"纯 numpy 环 = C++ 环"在数值上等价,即阶段 1 的环可直接作为原生栈的 oracle

## 4 挂起项(衔接 Plan.md 冻结区)

- **HiVT-CARLA 轨迹预测主线** —— ✅ **已闭环**(见 §7 末条 P-K 与 §9.2)。原卡点(env 迁移未完成
  + 无 GPU)已随 2026-09-18 会话恢复解除,val minADE 6.08。
- **激光 SLAM 阶段 2(C++ 位对齐对拍)** —— ✅ **已交付**(2026-09-19)。未做 ROS 原生栈移植,
  改以 `bin/slam_cpp.cpp` 独立 C++ 实现 + `bin/slam_diff_test.py` 单次 ICP 位对齐对拍收口
  (149/149 PASS);阶段 1 的纯 numpy 环由此被确认为可移植的 oracle(见 §7 P-H)。

## 5 数据资产(可复用,无需重采)

- 图像:`surround_train`(300 帧 × 6)/ `surround_p3` / `surround_town13` / `kitti_*` image_2
- 点云:`kitti_*/training/velodyne/*.bin`(64 线语义强度)/ `nus_mini` LIDAR_TOP + RADAR
  - **SLAM 输入**:`kitti_drive/training/velodyne/`(150 帧,`bin/slam_odometry.py` 的默认输入)
- 标定:`calib.json`(surround 精确 sensor2ego + intrinsic)/ `training/calib`
- 轨迹:`traj_town10`(550 帧 4 agents)/ `traj_town13_clean`(175 帧)
- 3DGS:`outputs/3dgs/`(单俯仰 90 帧 + 多俯仰 270 帧 3×90,含真值深度与位姿)
- 双目:`outputs/stereo/`(基线 0.4m,left/right/depth 40 帧 + depth_pc 点云)

## 6 纪律

- 本仓栈(纯值 + carla 0.9.16)不引入 ROS/rosbridge(旧栈教程能力一律在本栈上重新实现或标注不适用)
- 产物一律落 `outputs/`(`paths.project_path`,不随 cwd 漂移)
- 语义分割 GT:需要像素级 GT 时用 CARLA `sensor.camera.semantic_segmentation` 合成(采集侧),不手工标注
- 与 Plan.md 同律:Conventional Commits、提交不附 AI 署名、`ruff check && ruff format`、相关单测

## 7 执行进度(2026-09-19 更新)

> 单测口径:`autodrivedata` env 全量 **353 passed / 3 skipped**(2026-09-19 torch 2.6.0+cu124 换装后复跑);
> 本节各条目的"N passed"是该模块自身测试文件的口径。

### P2-A 语义 BEV(教程 05+06)——✅ 链路已通
- `bin/sem_bev.py`:YOLOPv2(检测+车道线+可行驶)+ YOLO11s-seg(实例掩膜)→ 像素级 BEV 投影
- 修 3 个坑:jit.load→torch.load(weights_only=False)、ego_pose 列表访问、投影坐标帧(世界→ego 局部旋转)
- 输出 `outputs/sem_bev/bev_{000000..000020}.png` + `panel_*.png`(21 帧样例),三色均值 da≈6754 / ll≈2560 / obj≈8146 px
- **验收通过**:像素级语义 BEV 图可生成,与 MapTR 矢量 BEV 面板对照成立

### P-D 单目测距(教程 08)——✅ 链路已通 + 评估修复
- `bin/mono_distance.py` + 两个纯值模块:`geometry.mono_depth_from_box`(迭代深度法 z=H·fy/框高,H=1.6m)
  + `autodrivedata/mono_depth.py:box_to_ground_distance`(地平面投影法)+ `geometry.box_2d_from_3d`(GT 3D 框角点投影框基线)
- **修复**:label 2D 列 59/97 是零宽退化框 → 检测框改走 GT 3D 框投影(与采集器 `box_to_gt_line` 同投影口径)="2D 检测框 = GT 3D 投影框"诚实基线
- **实测**(147 框):全距 mean 8.56% / median 7.8%;**10-20m 带 mean 8.47%、70% 框 <10%** 达标;7-10m 贴脸区系统低估(侧向角点拉大框高)如实排除;地平面投影法相机无俯仰不适用(None)
- 输出 `outputs/mono_distance/results.json`;单测 tests/test_mono_depth.py 10 passed

### P-E 多雷达标定(教程 15)——✅ 链路已通 + 判据修复
- `autodrivedata/multilidar.py`(point-to-plane ICP,纯 numpy,零 carla/零 open3d)+ `bin/calib_multilidar.py`(注入已知误差 → 判据)+ tests/test_multilidar.py 4 passed
- **修复**:收敛 = converged(增量阈值)∧ rmse_final<0.05m ∧ **overlap≥0.6** ∧ **plausible(t<5m、r<30°)**
- **实测**:small(0.1rad/0.1m)**converged**(overlap 0.991、iter 5、恢复 t 0.15m/r 5.7°);large(1.2rad/2m)**not_converged**(overlap 仍 0.991,recovered t 10.26m/r 68.8° 被 plausible 否决)→ RMSE 对平面场景天然低,overlap+合理性双闸分开真伪标定
- 输出 `outputs/multilidar/icp_result.json`

### P-F 双目测距(教程 09)——✅ 链路已通
- `bin/collect_stereo.py` CARLA 双目 rig(基线 0.4m,同朝向 yaw=0、y 轴 ±0.2m)+ `autodrivedata/stereo.py` 纯值链路(z=f·B/d 三角测量、SGBM 视差可选、NCC 纯 numpy、自监督 `reprojection_loss`)+ tests/test_stereo.py(手算锚点 10 passed)
- **实测**:定速 5.98 m/s(逐帧自证);SGM 近物点云 z≈5.5m ↔ GT 深度同值(三角测量链与真值深度对得上);depth_check.png 目检图已生成
- 输出 `outputs/stereo/`(calib.json + left/right/depth 40 帧 + depth_pc 点云)

### P-G 3DGS 重建(教程 16,降档链路验证)——✅ 链路闭环
- `bin/collect_3dgs.py` 静态场景 360° 环绕采集(spectator 归位修复:attach 子 actor set_transform 是**相对父**位姿,不归位会绕空场地、PSNR~10)+ `bin/train_3dgs_mini.py` gsplat mini 训练
- 初始化真值深度网格反投影 ~40k 点;**位姿用 CARLA 真值**(定位降级:pycolmap SfM Sim3 对齐误差 ~5.7m/79.6° → outputs/3dgs/sfm_eval.json),留出帧 0 作 val
- **实测**(train_result_ep1500.json):**psnr_all_mean 17.9、val 帧 0 13.56**;GT|渲染|差值三栏目检图 render_compare_ep1500.png 已生成
- 输出 `outputs/3dgs/`(capture 90 帧 + gaussians_ep1500.ply + train_result_ep1500.json)
- **调优(2026-09-18,多俯仰)**:`collect_3dgs.py --pitches "0,-15,-30"`(分 pitch 目录 images/p{p}/ + poses_{p}.json + pitches.json)+ `train_3dgs_mini.py` 跨 pitch 平铺 / `--scale` / `_DEPTH_LOWER`。120k 点 × 270 帧:psnr_all **17.9→18.32**(3000 iters),val 帧0 **13.56→10.45** 不升反降(多俯仰分走观测,留出单水平视角被稀释);消融与单环对照复现 val~10。多俯仰价值 = 补顶面/近地隐面(psnr_all ↑),代价 = 留出帧视角外推变难。产物 `gaussians_mp3_120k_v2.ply` / `train_result_mp3_120k_v2.json`

### P-H 激光 SLAM(教程 14,纯 numpy 两段式降档)——✅ 阶段 1+2 链路闭环
- `autodrivedata/slam.py`:**纯 numpy 核心环**(零 carla/零 torch/零 ROS),并作为阶段 2 C++ 移植的 oracle
  - 前端 = **帧间点面 ICP**(替代 FAST-LIO2 的 ikd-tree scan-to-map;帧间重叠 ~90% 时等效)+ 恒速先验初始化 + λ 正则化法方程
  - 后端 = **ScanContext 回环**(点计数描述子,列滚动不变)+ **位姿图 G-N**(节点 ≤200,纯 numpy,无 g2o)
- `bin/slam_odometry.py`(S1.3 前端):逐帧 velodyne → 链式位姿 `T_0→k`,落 `outputs/slam/{traj_raw,icp_stats}.json`
- **位对齐纪律**(阶段 2 对拍前提):全 double、网格哈希 tie-break 钉字典序、体素重心按扫描序累加、
  λ 正则化解代替 lstsq/SVD —— numpy/C++ 对拍只允许 ~1e-12 求解舍入偏差
- 单测 `tests/test_slam.py` **27 passed**(手算锚点:exp/log 往返、多分辨率近邻、rad0 早停回归、批量 nearest vs 标量逐位一致、
  恒速先验链、SC 描述子旋转不变、SC 默认 min_gap 回归、PGO 纠偏、直线序列零漂移)
- **阶段 1 验收(2026-09-19 端到端实测,输入 `outputs/kitti_drive/training/velodyne/` 150 帧)**:
  `bin/slam_odometry.py --root outputs/kitti_drive --frames 0-149 --out outputs/slam` →
  **wall 242.42 s(< 5 min 达标)、NaN 0、failed 1 帧、平均 RMSE 0.19593 m、平均 overlap 0.779**;
  产物 `outputs/slam/traj_raw.json` + `icp_stats.json`
  - **轨迹形状**:起点 (0,0,0) → 终点 (−42.21, −50.56, −1.40),路径长 **94.73 m**,yaw 0° → 121.12°,
    逐帧步长 mean 0.636 m / median 0.524 m —— **不是直线**(绕行约 121°),故 closure 的
    `drift_m 65.88` 是**开放路径首末位姿距离,不是漂移率**(源数据是 `collect_drive.py` 的 ego autopilot,
    无 GT 位姿文件可比 → 真漂移率不可得,如实标注)
  - **failed 帧 = 帧 3**:逐帧 ICP 步长 9.517 m / rmse 2.153 / overlap 0.022。已核为**数据侧 ego 瞬移**
    (相机帧 2→3 `mean|d|` 60.16、平均亮度 159.0→128.7,远超其余相邻对 12–24;GT 标注数 2→1;
    纯 x 平移 −1..10 m 暴力扫描 overlap 上限仅 0.128)——**非 ICP 缺陷**,链从帧 4 正常续上(overlap 0.875)
- **阶段 2(2026-09-19,C++ 位对齐对拍)**:`bin/slam_cpp.cpp`(C++17 单文件,零外部依赖,含 `--selftest`
  语义自检)+ `bin/slam_diff_test.py`
  - **判据订正**:不比"150 帧链式位姿末端"——链式 `T_k = Δ_k·T_{k-1}` 对 Δ 的舍入差**指数放大**(最近邻赋值
    是离散的:1e-16 的 seed 差翻格 → Δ 跳 ~1e-5 → 进入下一帧 seed,实测 ~2.4×/帧;两条纯 Python 链只把求逆
    从 `np.linalg.inv`(LU)换成刚体 Rᵀ 就能在 40 帧内发散到米级)。正确判据 = **同一 `(prev, cur, init, seed)`
    下比对单次 ICP 的 T**
  - **实测**:`bin/slam_diff_test.py --root outputs/kitti_day_clear --start 0 --end 149` →
    **149/149 PASS**(tol 1e-3 rad / 1e-2 m),最差旋转 **1.9e-14 rad**、最差平移 **7.0e-12 m**;
    链式末端差(仅作放大率参考,不判 FAIL)rot 0 / trans 5.1e-14 m
- **后端实测**(`bin/slam_backend.py`):15 关键帧(每 10 帧)、**候选 0 / 过门 0** —— 该序列是**开放路径无重访**,
  `n_loops=0` 是合法结果(**不造回环**);闭合 pre 63.185 m → post 63.185 m(无回环边时 PGO 不动,
  即"零信息时保持原样"的预期行为)。产物 `outputs/slam/{traj_pgo,loops,slam_summary}.json`

#### P-H.1 精度口径修正 + 带 GT 的 400 帧基线(2026-09-19)

此前所有 P-H 数字都建在**没有 GT 位姿**的 `outputs/kitti_drive`(ego autopilot)上,只能看轨迹形状。
本次用 `bin/collect_slam.py` 重采 **400 帧带 GT 位姿**的序列(Town10HD_Opt,autopilot,路径 206.24 m,
`outputs/kitti_slam/`,位姿落 `training/pose/{fid}.txt`),才第一次能算真 ATE/RPE。

- **位姿约定 bug(94×)**——`icp_odometry` 原出口是 `T_delta @ init_T`,把**点映射当位姿左乘**:
  纯平移时看着像在累加,一转弯就发散。实测同一 400 帧序列 **ATE 17.53 m → 修正后 0.187 m**。
  正解 = `T = init_T @ inv(T_delta)`(推导与 src/ref 方向记忆法见 `autodrivedata/slam.py::icp_odometry`
  docstring + `tests/test_slam.py::TestIcpOdometry::test_chain_of_turning_motion_matches_ground_truth`)。
  **教训:ICP 解出的 `T_delta` 是点映射(src→ref),不是位姿;方向由哪个帧当 src 决定,别按形状记。**
- **坐标系换算**——LiDAR 系位姿 → ego 系:CARLA(ego,y 右)与 KITTI(LiDAR,y 左)手性差 = 共轭 `M·T·M`
  (`M = diag(1,−1,1,1)`);LiDAR 挂点 `(1.2, 0, 1.65)` ⇒ `ego_pose = M·T_lidar·M @ inv(L)`。
  **不带杆臂 ATE 0.4589 m vs 带杆臂 0.1877 m(2.44×)** —— 方向是 `inv(L)` 不是 `L`。
- **基线**(`bin/eval_slam.py --traj outputs/slam_gt/traj_raw.json --gt outputs/kitti_slam`,产物 `outputs/slam_gt/eval.json`):

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
- **阶段 2 对拍复核**:`bin/slam_diff_test.py --root outputs/kitti_slam --start 0 --end 99` →
  **99/99 PASS**(最差平移 2.2e-11 m,旋转 0),修正后的位姿约定在 numpy/C++ 两侧一致。

#### P-H.2 路线 B(B1 IMU / B2 scan-to-map)实测裁决 —— **两项均不投**(2026-09-19)

用户指令"做 B1 的实测,没问题的话直接上 B2"。两项都做了实测,结论都是**负面**,故 B3/B4 不启动。

- **B1(CARLA IMU 能否支撑 IESKF 预测)⇒ 不投**(`bin/probe_imu.py`,产物 `outputs/imu_probe/*.json`):
  - IMU 陀螺读数**就是物理引擎报的角速度**(三轴比值恒 ≈1.000),但 8 m/s 直行时
    `gyro.z = −1.29°/s` 而旋转矩阵差分算出的真实 yaw 速率只有 −2e-5 rad/s(**差 1000×**);
    该伪角速度完全可复现(三次运行一致到 6 位小数),只在部分速度档出现(6/7/8 m/s 明显,4/9/10/12 几乎为 0)。
    **车没转,IMU 说有转** ⇒ 纯积分把 4.5° 假偏航灌进姿态。
  - **单步预测误差(决策判据)**:直行 8 m/s 时 IMU 位置 0.326 mm vs 恒速 CV **0.120 mm**(差 2.7×)、
    姿态 0.0903° vs **0.0041°**(差 22×);只在满舵绕圈时 IMU 才赢(6.016 vs 15.458 mm)。
  - 叠加已测的"CARLA **不模拟帧内扫描延迟**"(运动畸变校正是空操作):FAST-LIO2 用 IMU 的两个卖点
    在本仿真里**一个为空、一个为负** ⇒ **IESKF 不投**。
- **B2(ikd-Tree + scan-to-map 前端)⇒ 不投**(`bin/probe_scan_to_map.py`,产物 `outputs/s2m_probe/s2m_probe.json`):
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
- **副产品**:`bin/eval_slam.py`(ATE/RPE,含双坐标换算)+ `autodrivedata/slam_eval.py`(纯值)
  + `bin/collect_slam.py`(带 GT 位姿的采集)+ `tests/test_slam_eval.py`。

### P-I 累积语义建图 / 地面提取 / 聚类(教程 11+12+13)——✅ 链路闭环
- 累积:`autodrivedata/accum.py` + `bin/build_accum_map.py`(ego 位姿变换累积到全局系 + 时序证据加权)→
  `outputs/accum_map/map.ply`(150 帧);`tests/test_accum.py` 11 passed。累积显著抑制单帧伪影
- 地面:`autodrivedata/ground.py` + `bin/extract_ground.py`(RANSAC 平面拟合 + 内点掩码)→
  `outputs/ground/`;`tests/test_ground.py` 8 passed
- 聚类:`autodrivedata/cluster.py` + `bin/cluster_obstacles.py`(DBSCAN 风格邻域密度连通)→
  `outputs/cluster/`(150 帧,均值 117.97 簇/帧);`tests/test_cluster.py` 4 passed
- 注:上一会话"聚类任务失败"是误报——产物齐全,仅末行 bash 因 /tmp 配额满报错

### P-J 采集器纯函数下沉(回归测试先例)——✅
- `autodrivedata/collect_rig.py`(零 carla,AST 纪律守护)+ `tests/test_collect_rig.py` 8 passed
  - `ring_cam_pose`(3DGS 环绕位姿)+ `stereo_rig_offsets`(双目挂点 ±baseline/2)
- `bin/collect_3dgs.py` / `bin/collect_stereo.py` 改为 import 纯函数,bin 只剩 carla 编排
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
`bin/live_common.py` + 新 `bin/live_studio.py`(**不动**已验证的 `--maptr` 路径)。

**A 期交付**:

| 文件 | 动作 | 要点 |
|---|---|---|
| `bin/live_common.py` | 新增 | 多槽 MJPEG(单端口 `/stream/<name>` + `/` 索引页)/ 拼图 / GT overlay / 环视 rig / 第三方视角 / `KeyboardState` / MapTR 懒加载 |
| `bin/live_studio.py` | 新增 | 9 槽 = 6 相机 + `BEV` + `THIRD_PERSON` + `grid`(4×2 拼图);`--keyboard` 折进 tick 循环 |
| `bin/view_stream.py` | 改 | 薄编排:共享件全部改走 `live_common`;新增 `--rig` |
| `bin/drive_ego.py` | 改 | 薄封装 `live_common.KeyboardState`(独立进程遥控用法保留,studio 内置键盘是首选) |

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
会静默反号(第三方视角俯仰 −12° → +12°)。已落 `autodrivedata/geometry.py` 纯值实现 +
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
  **该 commit 未改 `bin/view_stream.py`** ⇒ 旧 rig 一直是 ep512 的正确口径。

**A/B 探针**(`bin/probe_rig_mount.py`,同一 ego 位姿同一 tick 帧,只变 rig):

| 权重 @ rig | 预测实例 | 段 | 品红 px | 光轴以上 |
|---|---|---|---|---|
| `maptr_ep512.pt` @ legacy(训练口径 ✓) | 153 | 269 | 122711 | 0 |
| `maptr_ep512.pt` @ official(错配 ✗) | 148 | 290 | 135989 | 0 |
| `maptr_600.pt` @ official(训练口径 ✓) | 163 | 326 | 173711 | 0 |
| `maptr_600.pt` @ legacy(错配 ✗) | 158 | 282 | 140562 | 0 |

**处置**:`live_common` 同时保留两套口径(`rig_spec`),`view_stream` / `live_studio` 加
`--rig {auto,official,legacy}`;`auto` 按权重名查 `LEGACY_CKPTS` 选(**默认喂对**)。
**正确性判据 = 外参与该权重训练数据逐字段一致**,不是"数字变了"也不是"用最新布局"。
⇒ `CLAUDE.md` / Plan.md §5.11f 里 `--maptr-ckpt outputs/maptr_ep512.pt` 的命令现在经
`auto` 自动走 legacy,**行为与文档化验收一致**。

**B 期交付(在线 SLAM,2026-09-20 ✅)**:

| 文件 | 动作 | 要点 |
|---|---|---|
| `autodrivedata/live_slam.py` | 新增(纯值) | `LiveSlam.push/snapshot`(链式约定逐字复用 `slam_odometry`)+ `map_in_ego_frame`/`traj_in_ego_frame`(LiDAR-0 系 → 当前 ego 系)+ **`SlamWorker`**(有界丢旧队列 + 帧间隙止损) |
| `bin/live_studio.py` | 改 | `--slam` 挂语义 LiDAR → `SlamWorker`;BEV 槽画地图点(灰)+ 轨迹(青);HUD 显式报滞后;`--slam-report` 落验收 JSON;`finally` 先 join 再销毁 world |
| `autodrivedata/mapviz.py` | 改 | `bev_points`(散点,批量像素)/ `bev_trajectory`(只连窗内相邻点)/ `bev_window_mask`(窗口判据单一来源) |
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

## 8 遗留缺口

- **教程 04(4 相机 IPM/单应拼接)**:仓库仍**无像素级 IPM 环视拼接**(`calib.py`/`mapviz.py` 只有标定与
  矢量投影;`sem_bev.py` 走的是 ground_intersection 射线投影,不是单应 warp)。判定:若需与教程 04 逐条对齐,
  这是**唯一剩余缺口**;当前语义 BEV(教程 06)已用射线投影达成同类目的,是否需要补 IPM 待用户定
- **教程 14 阶段 2**:✅ 已交付(2026-09-19)——C++ 位对齐对拍(`bin/slam_cpp.cpp` +
  `bin/slam_diff_test.py`,149/149 PASS);ROS 原生栈移植按用户裁决不做(见 P-H)
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

- `bin/collect_traj.py`(每 tick 重发定速,修旧 Town10 后半程停车 bug)+ `bin/assemble_traj_pt.py`
  (xodr centerline lane 切段,50 帧滑窗)+ `bin/convert_hivt_pt.py`(dict → TemporalData,edge_index + agent 朝向)
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

- **P-D 单目测距**(教程 08):`bin/mono_distance.py` + `geometry.mono_depth_from_box`(z=H·fy/框高,H=1.6m)
  + `geometry.box_2d_from_3d`(GT 3D 框 8 角点 → p2 投影 → 前端 u/v min/max,与采集器同投影口径)
  - 修复:label 2D 列 59/97 是零宽退化框 → 检测框改走 GT 3D 投影框("已知位姿投影框"诚实基线)
  - **基线实测**(147 框):全距 mean 8.56%/median 7.8%;**10-20m 带 mean 8.47%、70% 框 <10%** 达标;
    7-10m 贴脸区系统低估(侧向角点拉大框高)如实排除;地平面投影法相机无俯仰不适用(None)
  - **生产口径**(`--detector yolo`):YOLO 检测框 → 配 GT 投影框 IoU 贪心(match_dets_to_gt,conf 降序)
    → **131 命中**/147 GT(漏检 39);mean err 10.53%(vs 基线 8.56,模型框抖动如实上升)、z10_20 36% <10%
  - 产物 `outputs/mono_distance/results.json`(project)/ `results_yolo.json`(yolo)
- **P-E 多雷达标定**(教程 15):`autodrivedata/multilidar.py`(point-to-plane ICP,纯 numpy)+ `bin/calib_multilidar.py`
  - 修复:收敛 = converged(增量阈值)∧ rmse_final<0.05m ∧ **overlap≥0.6** ∧ **plausible(t<5m、r<30°)**
  - 实测:small(0.1rad/0.1m)**converged**(overlap 0.991、iter 5、恢复 t 0.15m/r 5.7°);
    large(1.2rad/2m)**not_converged**(overlap 仍 0.991,recovered t 10.26m/r 68.8° 被 plausible 否决)
    → RMSE 对平面场景天然低,overlap+合理性双闸分开真伪标定;`outputs/multilidar/icp_result.json`
- **P-F 双目测距**(教程 09):`bin/collect_stereo.py`(基线 0.4m,y 轴 ±0.2m)+ `autodrivedata/stereo.py`
  (z=f·B/d 三角测量、SGBM 视差可选、NCC 纯 numpy、自监督 reprojection_loss)
  - 实测:定速 5.98 m/s(逐帧自证);SGM 近物点云 z≈5.5m ↔ GT 深度同值;`outputs/stereo/`(40 帧)
- **P-G 3DGS 重建**(教程 16,链路验证):`bin/collect_3dgs.py`(360° 环绕采集,spectator 归位修复)+
  `bin/train_3dgs_mini.py`(gsplat mini);位姿用 CARLA 真值(SfM 退化 5.7m/79.6°);psnr_all **17.9**/val 13.56
  - **调优(多俯仰,2026-09-18)**:采集 `--pitches "0,-15,-30"`(落盘分 pitch 目录
    images/p{p}/ + poses_{p}.json + pitches.json,位姿几何仍 collect_rig.ring_cam_pose);
    训练侧跨 pitch 平铺、`--scale` CLI 化、深度下限抽 `_DEPTH_LOWER`。120k 点 × 270 帧:
    **psnr_all 17.9 → 18.32**(3000 iters),**val 帧0 13.56 → 10.45** 不升反降——归因:
    val 帧 0 是无俯仰单视角,多俯仰把观测分给斜视角后该视角覆盖被稀释;单环对照
    在新采集上复现 val~10(采集条件年内变化,pitch 覆盖减半的消融 2×90→1500 iters 仍
    16.34/10.67)。多俯仰价值 = 补顶面/近地隐面(重建完整性,psnr_all ↑),代价 = 留出帧
    视角外推变难。产物 `gaussians_mp3_120k_v2.ply` / `train_result_mp3_120k_v2.json`。
- **累积建图 + 地面 + 聚类**(教程 11/12/13):`autodrivedata/accum.py` / `ground.py` / `cluster.py`
  + 对应 bin;150 帧产物齐全,聚类均值 117.97 簇/帧
- **采集器纯函数下沉**(2026-09-18):`autodrivedata/collect_rig.py`(ring_cam_pose 环绕位姿 +
  stereo_rig_offsets 双目挂点;零 carla,AST 纪律守护)→ bin/collect_3dgs.py / collect_stereo.py 改用,
  剩纯 carla 编排;回归测试先例 tests/test_collect_rig.py(手算锚点 8 passed)
- **结论**:教程能力批量落地完成,7/16 能力达到"链路通+数值如实"(缺 14 FAST-LIO2 外部 ROS 栈);
  P-G 3DGS 调优(多俯仰采集 + --scale/--iters,见上)与 P-D 生产口径评估已完成

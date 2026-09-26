# 教程能力补全里程碑(2026-09-18)

> 16 篇 Carla 仿真教程(docs/Carla_Sim_Tutorial_01..16.md)能力在本仓栈的落地记录。
> 路线图与待办见 [Plan2.md](../Plan2.md);**2026-09-19 起新计划一律在 Plan2.md 制定**,
> [Plan.md](../Plan.md) 转为方案定案 + 历史执行记录(冻结)。

## ✅ HiVT-CARLA 轨迹预测主线(教程 02 升级——多智能体轨迹预测)

**交付**:CARLA 轨迹 → HiVT(TemporalData)训练管线闭环,含 3D 赛道记录。

- 采集:`autodrivedata/sim/collect_traj.py`(定速重发修复后)Town10 与 Town13 运动轨迹
- 组装:`bin/assemble_traj_pt.py`(xodr centerline lane 切段,滑动窗口 50 帧)
- 转换:`bin/convert_hivt_pt.py`(plain dict → TemporalData,全排列 edge_index + agent 朝向)
- 训练:HiVT-64,CPU(100 epoch)
- 评估:minADE / minFDE / minMR K=6,与 Plan2.md §9.1 Argoverse 复现同工具链

**数据**:train = Town10 250 场景 × 4 agents;val = Town10 211 场景(同图时间外推)
跨图泛化(Town13)因旧 val 静止分布弃用——见 Plan2 记录。

**实测**:

| 模型 | minADE | minFDE | minMR | 说明 |
|---|---|---|---|---|
| HiVT-64 ep99 | **6.08** | 15.85 | 1.000 | val 211 场景,逐场景最优 mode |

**诊断**:输入尺度正常(每步位移 ~0.8m),但模式输出退化——各模式终点集中在
6-8m(恒速 3s=24m 的 ~1/3),模型学到的是"保守匀速子集",未外推到位移全量。
训练 loss 每步持续下降(reg_loss 3.8→0.04),说明过拟合 train 而非欠学习。
这与 Argoverse 复现(0.6869/1.0301)的差距主要在**数据规模**(250 vs 20 万)与
**场景多样性**(固定 4 agent 直线布局)。

**结论**:管线闭环成立(能力面"预测"已通);指标绝对值受数据规模限制,
后续收益靠扩数据(多图/多布局/长尾)而非继续长训。

## ✅ 语义 BEV(教程 05+06)

**交付**:`bin/sem_bev.py` 语义 BEV 管线(纯值,不 import carla)。

- 模型:YOLOPv2(TorchScript 三合一:检测 + 车道线 + 可行驶区)+ YOLO11s-seg(实例掩膜)
- 投影:掩膜像素沿相机射线与地平面求交 → ego 局部系 → BEV 面板
- 输出:`outputs/sem_bev/{bev,panel}_{frame}.png`(21 帧样例)

**实测**:BEV 三通道着色统计(21 帧均值)da(绿)≈6754px / ll(黄)≈2560px / obj(品红)≈8146px。

**关键修定**:
- `torch.jit.load` → `torch.load(weights_only=False)`(autodrivedata torch cu130 无 CUDA 驱动)
- 投影坐标帧:ground_intersection 返回世界系 → 须按 ego yaw 旋转到 ego 局部系
  (漏转时 18 万像素仅 90 个落进窗口)

## ✅ 单目测距(教程 08)

**交付**:`bin/mono_distance.py` 单目测距评估(检测框 → 距离,与 KITTI GT 真距对照)。

- 方法:迭代深度法(尺寸假设 z=H·fy/框高,H=1.6m)+ 地平面投影法(`ground_intersection`)
- 检测框口径:**GT 3D 框角点投影框**(`geometry.box_2d_from_3d`,与采集器 `box_to_gt_line`
  同投影口径)——已知位姿投影的诚实基线,无 2D 模型误差
- GT 真距 = label_2 相机系 z(第 13 列);近距剔除 z<7m(贴脸框被侧向角点拉爆,如实)

**实测**(147 框,迭代深度法):全距 mean 8.56% / median 7.8%;**10-20m 带 mean 8.47%、
70% 框 <10%**;7-10m 贴脸区系统性低估(框高偏大),如实排除。地平面投影法对该 rig
不适用(相机无俯仰,None)。

**生产口径(YOLO 检测框,`--detector yolo`)**:131 命中/147 GT(漏检 39);mean err
10.53%(vs 基线 8.56,模型框抖动如实上升)、z10_20 带 11.76%。产物
`outputs/mono_distance/results.json`(project)/ `results_yolo.json`(yolo)。

**结论**:10-20m 带达标(<10%),链路"检测框→距离"成立;单目尺度歧义为上限(尺寸假设)。

## ✅ 双目视差(教程 09)

**交付**:`autodrivedata/sim/collect_stereo.py` CARLA 双目 rig(基线 0.4m)采集 + `autodrivedata/stereo.py`
纯值双目链路 + `tests/test_stereo.py`(手算锚点 10 passed)。

- 三角测量 z=f·B/d;SGBM 视差(OpenCV 可选,CV_8U)+ 自研 NCC 纯 numpy 匹配
- 自监督损失:`reprojection_loss`(右图按视差右移重建左图)
- 落盘 `outputs/stereo/`(calib.json + left/right/depth 40 帧 + depth_pc 点云)

**实测**:定速 5.98 m/s(逐帧自证);SGM 近物点云 z≈5.5m ↔ GT 深度同值(三角测量链
与真值深度对得上);depth_check.png 目检图已生成。

**结论**:链路"视差→深度→点云"闭环成立;近物视差大/深度小、远物视差小/深度大。

## ✅ 累积语义建图(教程 11)

**交付**:`autodrivedata/accum.py` + `autodrivedata/slam/build_accum_map.py`(语义单帧 → 多帧累积 BEV)。

- 语义帧间对齐:ego 位姿变换累积到全局系;时序证据加权
- 输出 `outputs/accum_map/map.ply`(150 帧累积)

**实测**:单测 4 passed;150 帧累积产物齐全。累积显著抑制单帧伪影(建图长尾收敛)。

## ✅ 地面提取 / 聚类障碍物(教程 12/13)

**交付**:`autodrivedata/ground.py` + `bin/extract_ground.py`(RANSAC 地面拟合);
`autodrivedata/cluster.py` + `bin/cluster_obstacles.py`(欧氏聚类)。

- 地面:RANSAC 平面拟合 + 内点掩码;聚类:DBSCAN 风格邻域密度连通
- 输出 `outputs/ground/`、`outputs/cluster/`(150 帧)

**实测**:单测合计通过(ground/cluster 各含手算锚点);聚类均值 117.97 簇/帧。
(注:上次会话"聚类任务失败"是误报——产物齐全,仅最后一行 bash 因 /tmp NFS 配额满报错)

## ✅ 多雷达标定判据(教程 15)

**交付**:`autodrivedata/multilidar.py`(point-to-plane ICP,纯 numpy,零 carla/零 open3d)+
`autodrivedata/calib/calib_multilidar.py`(注入已知误差 → 判据)+ `tests/test_multilidar.py`(4 passed)。

- 收敛判据:**converged(增量阈值)且 rmse_final<0.05m 且 overlap≥0.6 且 plausible(t<5m、r<30°)**
- overlap = 变换后源点在参考云 0.3m 近邻内的比例——真伪标定的分水岭

**实测**(icp_result.json 两分支):
- small_error(0.1rad/0.1m):**converged**(overlap 0.991,iter 5,恢复 t 0.149m/r 5.7°)
- large_error(1.2rad/2m):**not_converged**(overlap 虽 0.991,但 plausible 否决:恢复 t 10.26m/r 68.8°)

**结论**:RMSE 对平面富场景天然低,单靠它判收敛会误报;overlap + 变换合理性双闸把真伪标定分开。

## ✅ 采集器纯函数下沉(回归测试先例)

**交付**:`autodrivedata/collect_rig.py`(零 carla,AST 纪律守护)+ `tests/test_collect_rig.py`
(手算锚点 8 passed)。

- `ring_cam_pose`(3DGS 环绕位姿)+ `stereo_rig_offsets`(双目挂点 ±baseline/2)
- `autodrivedata/sim/collect_3dgs.py` / `autodrivedata/sim/collect_stereo.py` 改为 import 纯函数,bin 只剩 carla 编排

**结论**:采集器行为被单测锁定(回归测试先例);匹配纯函数 `match_dets_to_gt` 并入
`attribution.py`(P-D 生产口径共用,IoU 与 AP 评估同口径)。

## ✅ 3DGS 重建(教程 16,降档链路验证)

**交付**:`autodrivedata/sim/collect_3dgs.py`(静态场景 360° 环绕采集,spectator 归位修复)+
`bin/train_3dgs_mini.py`(gsplat mini 训练)。

- 采集:90 相机环绕(半径 6m)spawn 120 十字路口,RGB + 真值深度
- 初始化:真值深度网格反投影 ~40k 点;位姿用 CARLA 真值(定位降级:pycolmap SfM
  Sim3 对齐误差 ~5.7m/79.6° → 真值位姿,见 outputs/3dgs/sfm_eval.json)
- 训练:gsplat 光栅化 + Adam 1500 iters;留出帧 0 作 val

**实测**(train_result_ep1500.json):**psnr_all_mean 17.9、val 帧 0 13.56**(旧空场地采集
PSNR ~10.6)。GT|渲染|差值三栏目检图 render_compare_ep1500.png 已生成。

**调优(多俯仰采集,2026-09-18)**:

- `collect_3dgs.py` 多俯仰:`--pitches "0,-15,-30"`,落盘按 pitch 分目录
  `images/p{p}/{i}.png` + `poses_{p}.json` + `pitches.json`(每俯仰一圈独立,
  不再撞文件名);位姿几何仍是 `collect_rig.ring_cam_pose`。
- `train_3dgs_mini.py` 适配:跨 pitch 平铺(全局序 = pitch_idx × frames + i),
  `--scale` CLI 化(初始高斯尺度),深度 init 下限抽常量 `_DEPTH_LOWER`。
- 采集 270 帧(3 俯仰 × 90 环绕);深度点预算 120k(3 倍视角 → 不稀释覆盖)。

**调优实测**(同中心点、同半径,与基线同 val 帧 0 口径):

| 配置 | psnr_all | val 帧0 | 备注 |
|---|---|---|---|
| 基线(单俯仰 p0,1500 iters,40k 点) | 17.90 | 13.56 | 旧口径 |
| 多俯仰 3×90,120k 点,1500 iters | 16.53 | 11.04 | 点预算翻倍后首跑 |
| 多俯仰 + 3000 iters | 18.32 | 10.45 | 120k 点 |
| 多俯仰 + 3000 iters + scale 0.1 | 18.28 | 9.92 | 更大高斯无益 |
| 消融(2 俯仰 {0,-15},1500 iters,120k 点) | 16.34 | 10.67 | — |
| 消融(单俯仰 p0,1500 iters,40k 点,新采集) | 16.32 | 10.10 | 新采集后单环对照 |

**调优结论**:psnr_all 从 17.9 → 18.32(3000 iters,多俯仰视野翻倍),但 val 帧 0 不升反降
(13.56 → 10.45)。归因:val 帧 0 是**无俯仰单视角**——多俯仰把观测多分给斜视角后,
单一水平视角的像素被多视角高斯均摊,留出帧覆盖反而稀释;单环对照在新采集上仍复现
val~10。链路结论不变(链路验证非重建质量);多俯仰的价值在**重建完整性**(斜视角补
顶面/近地隐面,psnr_all 升),代价是留出帧的视角外推变难。权重重训入
`gaussians_mp3_120k_v2.ply` / `train_result_mp3_120k_v2.json`。

**结论**:链路"环绕采集 → 位姿 → 3DGS 渲染"闭环成立;PSNR 受初始化/纹理上限限制,
如实报告,目标为链路验证非重建质量。

## ✅ 激光 SLAM(教程 14,纯 numpy 两段式降档)

**交付**:`autodrivedata/slam.py`(纯 numpy 核心环,零 carla/零 torch/零 ROS)+
`autodrivedata/slam/slam_odometry.py`(前端)+ `autodrivedata/slam/slam_backend.py`(后端)+ `autodrivedata/slam/slam_cpp.cpp`(C++17 单文件,
零外部依赖,阶段 2 位对齐对拍)+ `autodrivedata/slam/slam_diff_test.py`(对拍脚本)+ `tests/test_slam.py`(27 passed)。

- 前端 = **帧间点面 ICP**(替代 FAST-LIO2 的 ikd-tree scan-to-map;帧间重叠 ~90% 时等效)
  + 恒速先验初始化 + λ=1e-4 正则化法方程(`estimate_transform_gn`)
- 后端 = **ScanContext 回环**(点计数描述子、mean-per-ring 余弦距离、列滚动不变、180° 歧义用双 yaw 初值 ICP 消解)
  + **位姿图 G-N**(右扰动局部坐标 + LM 阻尼,纯 numpy 无 g2o)
- **性能改造**(S1.3 达标线 = 150 帧 <5min):`GridHash.nearest` 由粗到细双通道批量扫描
  (rank 折叠 = min(d²)+min(rank),rad0 早停修正后与暴力逐位一致)+ `estimate_normals` 批 PCA 向量化

**实测**(`outputs/kitti_drive/training/velodyne/` 150 帧,`outputs/slam/`):

| 指标 | 值 |
|---|---|
| 端到端耗时 | **242.42 s**(< 5 min 达标) |
| NaN / failed 帧 | **0 / 1**(帧 3) |
| 平均 RMSE / 平均 overlap | 0.19593 m / 0.779 |
| 路径长 / 终点 | 94.73 m / (−42.21, −50.56, −1.40),yaw 0°→121.12° |
| 回环(候选 / 过门) | 0 / 0(**开放路径无重访,如实为 0,不造回环**) |
| 阶段 2 对拍 | **149/149 PASS**(最差 1.9e-14 rad / 7.0e-12 m) |

- **failed 帧 = 帧 3**,已核为**数据侧 ego 瞬移**而非 ICP 缺陷:相机帧 2→3 `mean|d|` 60.16 /
  平均亮度 159.0→128.7(其余相邻对仅 12–24)、GT 标注数 2→1、恒等位姿下 overlap 0.096(其余对 0.57–0.91)、
  纯 x 平移 −1..10 m 暴力扫描 overlap 上限仅 0.128。链从帧 4 正常续上(overlap 0.875)。
- **closure 口径**:源数据是 `collect_drive.py` 的 ego autopilot,**无 GT 位姿可比** →
  `drift_m 65.88` 是**开放路径首末位姿距离,不是漂移率**,如实标注不冒充精度指标。

**阶段 2 判据订正(踩坑记录)**:最初想比"150 帧链式位姿末端",实测**不可行**——链式
`T_k = Δ_k·T_{k-1}` 对 Δ 的舍入差是**指数放大**的(最近邻赋值离散:1e-16 的 seed 差翻格 → Δ 跳 ~1e-5 →
进入下一帧 seed,实测 ~2.4×/帧;两条纯 Python 链只把求逆从 `np.linalg.inv`(LU)换成刚体 Rᵀ,
就能在 40 帧内发散到米级)。**正确判据 = 同一 `(prev, cur, init, seed)` 下比对单次 ICP 的 T**,
链式末端差只作放大率参考(不判 FAIL)。

**结论**:教程 14 两段式链路在本仓栈内闭环(未引 ROS);纯 numpy 环经 C++ 逐帧位对齐验证可作为
原生栈的 oracle。ROS 原生 FAST-LIO2 + SC-PGO 移植按用户裁决不做。

**精度口径修正 + 带 GT 的 400 帧基线(2026-09-19)**:此前数字建在**无 GT 位姿**的
`outputs/kitti_drive` 上,只能看轨迹形状。重采带 GT 的 `outputs/kitti_slam`(400 帧 / 206.24 m)
后第一次能算真 ATE:

| 指标 | 值 |
|---|---|
| **ATE(对齐后)** | **0.1877 m**(尺度 1.00829);不带杆臂 0.4589 m(**2.44×**) |
| RPE Δ=1 / 5 / 10 | 0.0216 / 0.0568 / 0.0948 m |
| 相对误差 | 路径长的 **0.091%** |
| 前端 wall / NaN / failed | 310.01 s / 0 / 0,平均 RMSE 0.1306 m、overlap 0.8177 |
| 后端 | 40 关键帧、**候选 0 / 接受 0**(最近关键帧对相隔 58.69 m,折返但从未空间重访 ⇒ `n_loops=0` 正确) |
| 阶段 2 对拍复核 | **99/99 PASS**(最差平移 2.2e-11 m) |

- **位姿约定 bug(94×)**:`icp_odometry` 原出口 `T_delta @ init_T` 把**点映射当位姿左乘**,
  纯平移看着像累加、一转弯就发散;正解 `T = init_T @ inv(T_delta)`(ATE 17.53 → 0.187 m)。
- **坐标系换算**:`ego_pose = M·T_lidar·M @ inv(L)`(手性共轭 + 杆臂 `inv(L)`,方向写反差 2.44×)。
- 评估工具:`autodrivedata/slam/eval_slam.py` + `autodrivedata/slam_eval.py`;采集 `autodrivedata/sim/collect_slam.py`。

**路线 B 实测裁决 —— B1/B2 均不投(2026-09-19)**:

- **B1 CARLA IMU 能否支撑 IESKF**(`autodrivedata/sim/probe_imu.py`):IMU 陀螺读数就是物理引擎角速度,但 8 m/s
  直行时 `gyro.z = −1.29°/s` 而旋转矩阵差分的真实 yaw 速率仅 −2e-5 rad/s(**差 1000×**,可复现,
  只在 6/7/8 m/s 档出现)。单步预测 IMU 位置 0.326 mm vs 恒速 **0.120 mm**、姿态 0.0903° vs
  **0.0041°**(只在绕圈时 IMU 才赢)。叠加"CARLA 不模拟帧内扫描延迟"⇒ FAST-LIO2 用 IMU 的
  两个卖点一空一负 ⇒ **IESKF 不投**。
- **B2 ikd-Tree + scan-to-map 前端**(`autodrivedata/slam/probe_scan_to_map.py`):**oracle GT 位姿**构造局部地图
  (= 收益上限),误差随地图深度 K **单调变差**——K=1 **1.05×** / K=3 **1.55×** / K=8 **2.75×**,
  重新体素化更差(3.26×)。机制:残差下降 ≠ 位姿正确(ICP 净赚代价降幅仅 0.0012→0.0050 m 而
  位姿误差 0.040→0.105 m)、约束方向塌陷(λ1 1.23e-1→9.80e-2,误差沿最软方向滑走)、
  内点被地面稀释(地面占比 0.555→0.605 而地面法向在 x/y/yaw 零信息)、法向稳定性假设**被否证**
  (一致性 0.849→0.954 反而上升)⇒ **帧间重叠 ~90% 时 scan-to-map 无收益,不建该前端**。
  **边界**:单序列 / Town10HD_Opt / 体素 0.5 / oracle GT;低重叠场景(高速、稀疏扫描、大转弯)
  可能翻转,脚本可直接复跑复核。

## ✅ 8 路实时可视化 studio + 键盘操控 + 在线 SLAM(2026-09-19 A 期 / 2026-09-20 B 期)

目标(用户原话):「输出 六相机视角 + BEV 视角 + 第三方视角 共 8 个终端可视化输出,
我操控汽车便可采集动静态目标和道路特征,输出感知结果的同时也做 slam 重建。」

**交付**:`autodrivedata/sim/live_common.py`(共享件:单端口多槽 MJPEG `/stream/<name>` + `/` 索引页、
拼图、GT overlay、环视 rig、第三方视角、`KeyboardState`、MapTR 懒加载)+
`autodrivedata/sim/live_studio.py`(9 槽 = 6 相机 + `BEV` + `THIRD_PERSON` + `grid` 拼图;
`--keyboard` 折进 tick 循环)。`view_stream.py` / `drive_ego.py` 改为薄编排。

**验收(数值)**:挂点自检 `平移 0.000 m / 偏航 0.000°`;第三方 ego 框 `中心偏移 0.001 画幅 /
宽 0.116 画幅 / 深度 9.71 m`(判据 偏移 ≤0.5、宽 ∈[0.05,0.6]);8 路同帧 raw/overlay 差集
6 相机路 2.4–7.9%、第三方 11.1%、BEV 10.0%;`--maptr` 回归品红 6002 px(6 相机路)vs raw 0、
**光轴以上 0**。

**两处踩坑(已落单测/文档)**:

1. 世界系相机位姿 → ego 系必须 `inv_ego @ cam`,写成 `cam @ inv_ego` 会把 ego 世界坐标混进
   平移块 —— 实测偏差 **138.243 m**,而**偏航仍恰好 0.000°**(只看偏航自检会漏掉)。
2. 旋转阵逆分解 `pitch = asin(R[2,0])`,写成 `asin(−R[2,0])` 静默反号(俯角 −12° → +12°)
   → 已下沉 `autodrivedata/geometry.rotation_matrix_to_carla` + 往返单测。

**环视 rig 两代并存(重要订正)**:rig **不是"越新越好",必须与权重训练数据一致**。
`legacy`(共用 `SENSOR_OFFSET` + BACK_LEFT/RIGHT 235/125)对应 `maptr_ep256/ep512`;
`official`(逐相机 `SENSOR_MOUNTS` + 108.6/−110.8)对应 `maptr_600/1000`。证据:
`surround_train/map_infos.json` 是 legacy、`surround_p3`/`surround_town13` 是 official,
且 `maptr_600/map_infos.json` 帧 0-199 为 legacy、帧 200-599 为 official。
`--rig {auto,official,legacy}` 中 `auto` 按权重名选(默认喂对);A/B 探针
`autodrivedata/calib/probe_rig_mount.py` 量化错配代价(ep512 legacy 122711 px vs official 135989 px)。
详见 Plan2.md §P-L.1。

> ⚠️ **2026-09-22 订正**:`official` 那一代 rig **本身是错的**(偏航漏了 `yaw_carla = −az_nus`
> ⇒ 四个侧/后相机左右镜像;pitch/roll 硬编码 0),已更名 **`nuscenes`** 并重标;
> **全部 MapTR 权重**(`ep256/ep512/600/1000`)据此标废弃、重采重训。见下节 §P-M 与
> Plan2.md §P-M。本节保留的价值 = "rig 必须匹配训练数据"这条方法论与错配代价的量级。

**B 期 ✅(在线 SLAM,2026-09-20)**:`autodrivedata/live_slam.py`(纯值 `LiveSlam.push/snapshot`
+ `map_in_ego_frame`/`traj_in_ego_frame` + **`SlamWorker`**)+ `live_studio --slam`
(挂语义 LiDAR → worker;BEV 槽画地图点灰 / 轨迹青;HUD 显式报滞后)+ `mapviz.bev_points`/
`bev_trajectory`/`bev_window_mask`。验收见 Plan2.md §P-L.2~P-L.4。

**B 期两条原前提都不成立(重要订正)**:

1. **「离线 ICP 0.78 s/帧 ⇒ 必须 worker 线程」** —— 0.78 s 是 400 帧**含转弯/重访的平均值**,
   在线逐帧(gap 1)只有 **0.15–0.35 s**。
2. **「worker 线程能救帧率」** —— 相反,worker 被 **GIL 饿死**:主线程每 tick 的 overlay/拼图/
   HUD 是纯 Python 字节码持 GIL 不放。同一对点云:worker 线程 eff **0.04–0.24** vs 主线程同步
   eff **0.90–1.00**;钉 `OPENBLAS_NUM_THREADS=1` 不改结论 ⇒ 不是 BLAS 线程池。
   ⇒ **默认同步执行**(`SlamWorker(sync=True)`,帧率 ~2.4 fps),`--slam-async` 留作对照。

**有界丢旧队列不足以保证滞后有界(关键机制)**:队列有界的是**深度**,不是 `prev_down` 与当前帧的
**间隙**,而 ICP 成本随间隙爆炸(0.8 m 0.2 s → 8 m 2.8 s → 32 m 39 s)→ 丢帧 ⇒ 间隙更大 ⇒ 更慢
⇒ 更多丢帧,**无界正反馈**。异步模式实测:70 s / 259 tick 只处理 **8 帧**、滞后单调涨到
**157 帧(45 s)**、`lag_bounded False`。修复 = **按帧号差止损**(`--slam-max-gap`,默认 3):
间隙超阈的帧不做 ICP、直接恒速外推,`prev_down` 照推进 ⇒ 下一帧回到 gap 1,如实计入 `n_dead`。

**验收(全部数值)**:

| 判据 | 结果 |
|---|---|
| 与离线基线对拍 | 400 帧链式位姿**逐元素差 0.0**、ATE **0.18768 m = 离线 = 参照**、尺度 1.00829 |
| 滞后有界 | 同步模式 136 tick / 处理 136 / **丢 0 / 止损 0** / `lag_max 0` / `lag_bounded True` |
| 停干净 | `nvidia-smi` 6394 MiB = 基线、残留进程 0、日志 `worker 已停(干净退出)` |
| BEV 数值自证 | 画出 90611 点 / 轨迹 20 段 / `bev_drawn_equals_in_window True` |

**判据订正**:原写"地图点窗内占比 ≈100%"是误读 —— 地图覆盖 ~300 m 行程、窗口只有 30×60 m,
故 `bev_map_in_window_ratio` 0.241 正常;真正的自证是「画出的点数 = 窗内点数」。

**八视角视频段 `--video`(2026-09-20)**:`live_studio --video <mp4>` 把拼图槽逐帧写成 mp4
(cv2/mp4v,惰性开编码器;`--video-fps` 是标称帧率,结束打印**实际采集 fps 与播放倍速**)。
实测 `outputs/videos/studio_8view.mp4`:43 帧 / 2484×374 / 86 s、实际 0.43 fps(播放 1.2×)、
8 格非黑占比 84–100%;HUD 含 `pred=156/seg=354`、`SLAM 滞后 0帧`、`丢 0 / 止损 0`。
帧率口径:MapTR + SLAM 同开 **0.3–0.5 fps**,仅 GT overlay ~2 fps(纯 Python overlay/拼图持 GIL)。
`--video-tile >1` 只是插值放大(文件翻倍、无新信息),默认 1。详见 Plan2.md §P-L.5。

**拼图改三层 + 不缩像素(2026-09-20,§P-L.6)**:用户看完视频报告「6 视角摄像头的 FoV 缩小得都
看不到地面了」。**根因是 `PIL.Image.paste` 在源图大于目标框时不报错、不缩放、只贴左上角** ——
旧 4×2 等尺寸拼图(`disp_w = 621`)把 1242×375 的相机图裁成 621×187,右半 + **下半(地面)** 无声丢弃。
数值判据:拼图格与「源图左上角裁剪」差 **0.128** vs 与「整幅缩放」差 **66.18** ⇒ 是裁剪不是缩放。

改动:`live_common.compose_rows`(按行拼,每格**原生像素**)+ `compose_grid` 加尺寸守卫
(不符即 `ValueError`,把这类坑钉死不复发)+ `live_studio.GRID_ROWS` 三层(①左前/前/右前
②右后/后/左后 ③第三方 + BEV,**不沿用 `SURROUND_CAMS` 字典序**)。画布 2484×374 → **3726×1170**。
验收:逐格与源图**最大差 0.0**、相机下半地面区域最大差 0.0、三层行序逐格 x 坐标吻合。
回归 `tests/test_live_common.py`(新增 12 用例)。

## ✅ 环视相机标定修正(nuScenes 口径)+ 七锚自证 + 实时监看槽(2026-09-22,教程 03)

**目标(用户原话)**:「环视相机的标定按照 nuScenes 来是不是更好一些」—— 时序建图的前置,用户设了
硬顺序 **"先修完标定,才讨论时序建图"**。详见 Plan2.md §P-M。

**根因**:旧 `official` rig 从官方 azimuth 抄值时**没做 CARLA/nuScenes 符号转换**
(`yaw_carla = −az_nus`)⇒ 四个侧/后相机**左右镜像**(FRONT_LEFT/RIGHT 差 110.3°、
BACK_LEFT/RIGHT 差 217.2°),pitch/roll 还硬编码 0。**前/后相机因光轴近自逆"看着对"**,
故长期没暴露 —— 与"能画出图不是投影正确的证据"是同一条教训。真值改由
`autodrivedata/camera_rig.py` 从官方四元数单点导出(导出前**归一化**:官方值模长
0.99994~1.00005,不归一化矩阵非正交)。

**七锚自证 `autodrivedata/calib/probe_calib.py`**(判据全数值,不目检)→ `outputs/calib_check/report.json`,
A0–A6 **全 true**:A0 光轴 vs 官方方位角 `7.1e-15°` / A1 侧别 4/4 同侧 / A2 四方位锥四对全 match /
A3 LiDAR-平面-深度图交叉验证 median |e| **0.0003–0.0009 m** / A4 轴目标物掩膜质心 `cx = 620.5`、
`fx_est 621.6 px`、残差 max **0.200 px** / A5 实挂 vs 规格 平移 `3.8e-06 m` 偏航 `4.5e-05°` /
A6 主点锁定 `(w−1)/2`。

**★ 像素约定裁决 = CORNER**(主干结论):CARLA 渲染栅格**索引 i 的连续坐标恰为 i** ⇒
`cx=(w−1)/2=620.5`、`cy=(h−1)/2=187.0`。A3 corner `0.0003 m` vs center `0.023 m`(**70×**,
六相机一致);A4 用掩膜**索引**中点独立测得 `fx=621.6 px`。`fx=(w/2)/tan(fov/2)=621.0` 与
`cx=620.5` **并存不矛盾**(前者"半 FOV↔半宽",后者索引约定中心)。**角色分类防再犯**:
采样 CARLA 栅格 ⇒ 必须 corner;采样 torch 栅格(FPN/`align_corners=False`/gsplat)⇒
`(u+0.5)/W*2−1` **正确**;PIL 纯绘制 ⇒ 无关;cx/cy **从 K 直读**不重算。

**实时监看槽 `live_studio --calib`**(`autodrivedata/calib_live.py`):另挂 6 深度相机
(同挂点/同内参/同分辨率)+ ray_cast ⇒ 世界系平面投影回相机按深度残差着色。验收(77 tick):
5/6 相机可用、pooled median **0.00032 m**、时序 0.00033/最大 **0.00044 m**。**成本**:平面拟合
~100–150 ms vs 六相机采样 ~9 ms ⇒ `--calib-refit` 默认 2;**平面是世界系的**故可跨 tick 复用。

**★ CAM_BACK 自遮挡 = 平台边界**(**⚠️ 已于 2026-09-23 §P-M.10 订正**:那是**挂点原点修正前**、
CAM_BACK 落在车身中部时的读数;修正后重测**六路 `near_fraction` 全 0.0**):官方挂点 z=1.5791 只比 CARLA ego 自身车顶(≈1.556)高
**0.023 m** ⇒ 近场(<0.5 m)占比 **1242×375 下 0.367 / 640×360 下 0.195**,其余五路 0.000。
判据必须是"**样本不足 ⇒ 报 `–` 而不是 0.000**"(它残差中位数 0.0003 与其它同级),**不是**
"median 大 = 坏标定"。**踩坑**:自遮挡判据曾写死 `> 0.2` —— 1242×375 成立、640×360 **静默失效**;
根因是近场占比**随画幅宽高比变**(水平 FOV 都 90°,640×360 竖直 FOV 大 ⇒ 车顶占比小)⇒
改**相对判据**(基准 = 同批可用相机近场占比中位数,阈值 `max(0.05, 10×基准)`),不按相机名硬编码。

**回归**:`tests/test_calib_live.py`(51)+ `tests/test_geometry_nus.py` 的 `camera_rig` 推导钉
(`yaw_carla = −az_nus` 到 1e-9、平移只翻 y、6DoF 不可降 yaw-only、历史字面表 110°/217° 偏差量级)
+ `tests/test_live_common.py` 的 `rig_spec`/`resolve_rig`/**`rig_mount_deviation` 规格对账**/`draw_hud(y=)`。

## ✅ 全传感器「声明 ≠ 渲染」修正(2026-09-23,§P-M.7)

**新失效模式**:`collect_nus.py` 此前**表对了、图错了** —— `calibrated_sensor` 写官方正确值,
传感器却 spawn 在另一套位姿(6 相机一律挂 **LiDAR 挂点** `(1.2,0,1.65)` + 旧镜像偏航表;5 雷达偏航
`{0,±45,±90}` 猜测值;LiDAR **无 rotation**;六路共用 `fov=90`/`fx=800`)。**这类缺陷只查表全绿、
只目检"图能出"也全绿** —— 与 §P-M.1 的「表错了」是同类病的不同面。

**修法 = 渲染与声明由同一份常量导出**:相机走 `NUS_CAMERA_RIG`(逐相机 6DoF)+ 逐通道蓝图 fov;
雷达 `−az_nus` 由 `NUS_RADAR_OFFSETS` 导出;LiDAR 走官方挂点 + 由四元数**导出**的 `LIDAR_ROT`
(不手抄);内参换逐通道官方 n015 K。LiDAR/雷达落盘表与官方**逐位相同**。

**六条验收判据全过**(`autodrivedata/calib/verify_nus_calib.py`,修前 → 修后):① 相机实挂 vs 声明 平移 `4.3e-06 m` /
偏航 `3.9e-05°`(修前 0.52–1.17 m、0.15–126.40°)② 雷达实挂 `2.9e-05°`(修前四路差 93.89–135.98°)
③ 雷达点进自身 FOV `0.86/0.81/0.85/0.91/0.90`(修前 `0.0000/0.0000/0.0011/0.0015`)④ LiDAR 复现
`num_lidar_pts` **1.0000**(修前 0.0854)⑤ 内参偏差 **0.0 px**(修前 32–473 px)⑥ 渲染 FOV dev
`0.0024–0.0672°`(修前六路共用 90° vs 声明 64.3°)。

**两处口径/踩坑(已落回归钉)**:① **判据 ③ 必须 `R_official^T @ R_declared @ p_sensor`** —— 点云存的是
传感器自身系,+x 即光轴,只问"点在自己系里落不落进 ±38.1° 锥"是**恒真**的;**明令禁止
`num_radar_pts` 作判据**(跨 5 通道求和,官方 R 也只复现 0.6279)。② **判据 ⑥ 每 tick 必须抽干全部
相机队列** —— 只取被测相机那一帧会让其余五路积压,轮到它们时读到锥体 spawn **之前**的陈旧帧,
症状是"六相机只有第一个测得出、其余全 0 且**与距离无关**"(看着像摆不进去,其实不是);
**Z 不能写死**(地图遮挡随出生点变)⇒ Z 阶梯 + 横移 `±0.45·Z`。

**产物处置**:`outputs/nus_mini`(33M)**已重采覆盖**;`nus_mini_l3` 已删;修前基线固化在
`report_prefix.json`。

## ✅ 覆盖层中文豆腐块:字体落点收敛(2026-09-23,§P-M.8)

`check_geometry.png` 的中文全成方框 ⇒ 裁定**是字体问题不是编码问题且本机可解**(不降级英文)。
新建 `autodrivedata/fonts.py` 收敛为**唯一字体落点**:渲染探针判字形(`U+10FFFF` 像素签名 = `.notdef`
签名,`文相机字` 四签名必须互不相同),`sanitize` 缺字换 ASCII、兜底 `?`,**绝不留豆腐块**;
六个绘制文件全部迁移,回归 `tests/test_fonts.py`(含 AST 根因钉)。

**两条成因各修一次才全绿**:① 本机字体族无 CJK 而代码硬写 DejaVu;② **PIL 没有回退链,不传
`font=` 就用内置位图字体**(同样无 CJK、仅 ~11 px)。本机唯一 CJK 字体 = **CARLA 随包的
`DroidSansFallback.ttf`**。**判据是渲染探针,不看文件名、不看 `fc-list`、不用 `getmask` 字节**
(实测 PUA 反例)。**长文本一律 `fonts.wrap(text, size, 最大宽)` 折行** —— 单行超出被 PIL
**静默裁掉**(不报错、不告警,图上看着像"写完了")。

## ✅ 宽视场 wide rig:后移挂点 + 新 FoV(2026-09-23,§P-M.9)

**用户口径**:前三个 55°、后侧 110°、后 180°;挂点后移到车尾;**画幅里不能有自身车体像素**;
另出 `outputs/calib_check` 配置图。**两处先被离线复算推翻**:① **180° 针孔退化**(`fx → 5e-14` ⇒
`det(K) ≈ 0`,投影恒等于主点;CARLA 渲染另有一道崖:**≤138° 忠实、139° 起中央带消失、180° 全黑**,
0.9.16 无鱼眼蓝图)⇒ 用户裁决封顶 **120°**;② **零车体像素的解析上限
`hfov ≤ 2·min(az−90°, 270°−az)`** —— 官方轴方位角下后侧上限只有 37.19°/41.58° ⇒ **必须同时改轴**,
用户裁决改 **145/180/215**。另订正外部事实:官网 55/110/180 是**方位角**不是 FoV(官方 FoV 实为
64.31–64.96°×5 + 89.34°)。

rig 落在 `autodrivedata/camera_rig.py`(`NUS_WIDE_*`;**前三个与官方逐位相同**、后三路 `x = −1.90`
在车身最后点之后 4.7 cm),官方口径仍是默认。**八条判据全过**,覆盖 **95.79%** / 盲区 3 段
**15.1561°** = 4.21%(正侧方两段是"前视 55° + 后视不越 90°"的**结构性代价**,如实报出)。

**★ 方位轴重叠 ≠ 有限距离下的共同可见**(本轮最重要的订正):方位轴是**无穷远**口径,挂点视差让同一
世界点两路方位角差最多 **1.7°** ⇒ 官方 `FL↔BL` 11.20° 在 12 m 处只剩 **4.838°**、官方 `B↔BL` 的
5.89° **整个消失**;判重叠要按有限距离算(`rig_check.common_band`),**锥心必须抬 0.45 m**(否则侧视
相机擦车顶被自车挡,读数变成"有没有被自车挡")。

**交付物**(`autodrivedata/calib/viz_rig_check.py` → `outputs/calib_check/`):`rig_layout_{nuscenes,wide}.png`(俯视挂点
+视锥 / 方位环重叠橙·盲区红带 / 数字表 `通道·挂点·方位角·FoV·az±fov/2`)、`views_{rig}.png`(六视角
**原生像素**拼图 + 线性方位尺 + 车体像素就地染品红)、`report_{rig}.json`。**图不能替代
`verify_nus_calib`** ——「声明 ≠ 渲染」在图上看不见(§P-M.7)。两处踩坑:跨 0° 扇区写 `az % 360`
会让 `rectangle` 抛 `ValueError`(必须钳不取模)、`np.asarray(PIL)` 是只读视图(染色须 `np.array` 拷贝)。

## ✅ 挂点原点修正:整套传感器后移 1.2563 m(后轴对齐)(2026-09-23,§P-M.10)

**用户发现**:目检配置图看出 **CAM_FRONT 落在引擎盖上方**而非车顶前部。**根因是两套系的原点定义不同**
—— CARLA 车辆 actor 的原点 = **车身长度中点**(a2 实测包围盒 x ∈ [−1.8527, +1.8527]);nuScenes 官方
标定表(及 `ego_pose`)以**后轴中心**为原点 ⇒ 把官方表值当 CARLA 局部坐标直接用 = 整组 **12 路偏前
1.2563 m**(官方 `CAM_FRONT` x=1.7008 落在 CARLA +1.7008,而应在 **+0.4445**)。

**用户裁决**:平移量 = **后轴对齐 Δx = −1.2563**(**不是**"中点规则"的 −0.8646 —— 那会让 ego 原点与
真后轴错位 0.39 m,`ego_pose` 与 devkit 语义不再一致)+ 范围 = **相机 + 5 雷达 + LiDAR 全套**刚体同移
(只挪相机会让跨模态融合再错 1.2563 m)。真值落 `geometry.NUS_EGO_ORIGIN_X` 单点。

**两条新判据**:**①(实挂 vs 声明)②(雷达实挂)相对同一个 ego 比,对原点误差在结构上盲** ⇒ 新增
**⑨ 世界系链**(`declared = 落盘表 ⊕ 实测后轴位姿` vs `rendered = 实挂经共轭`,12 路逐位比)与
**⑩ 独立复测后轴**(双偏航自解,**不读常量**)。⑩ 实测后轴 **−1.2562963447285285** / 前轴
+1.2502001429339191 / 轴距 2.506496487662447 / 四轮 C 一致 **1.73e-06 m**。

**★ ⑨ 首跑暴露两条真陷阱**:①**六路相机齐刷刷 119.93–120.07° 不是装反** —— CARLA 相机局部轴
(x 前,y 右,z 上) vs nuScenes (x 右,y 下,z 前),拿 LiDAR/雷达那条 `M·R·M` 比相机必得**轮换阵本征角
(恒 120°)**;修法 `CARLA_CAM_TO_NUS_CAM = [e1, −e2, e0]`(**正交但 det = −1**,局部基翻转与全局基翻转
相乘才抵消)。②雷达/LiDAR 齐刷刷 **0.0846°** 是**第二处「声明 ≠ 渲染」**:`ego_pose` 只写纯偏航,
**丢掉实测悬架俯仰 +0.0642°**(静置 8 tick 收敛)⇒ 新增 `nus_ego_rotation` 出全 6DoF;③同一处再暴露
**符号陷阱**:因子分解只对**右手**矩阵成立,而 `carla_rotation_matrix` 是 **UE 左手口径**(纯 pitch 时
`R[2,0] = +sin p`)⇒ 正确是 `Rz(−yaw)·Ry(−pitch)·Rx(+roll)`,**yaw 项对得上故极隐蔽**,
回归**按矩阵相等**判(比四元数会被 `±q` 骗过)。

**验收**:⑦ 由 `CAM_BACK 619189 px = 42.9992%` → **六路全 0 px**;⑨ 相机 `1.5e-05–4.0e-05°` /
`3.8e-06–9.5e-06 m`,雷达·LiDAR `4.2e-06–2.6e-05°` / `1.2e-06–1.1e-05 m`;**两代 rig 十条判据全过**
(wide 覆盖仍 95.79%/15.156° 逐位不变 ⇒ 无回归);`live_studio --calib` 重测六路 `near_fraction` **全 0.0**
⇒ **订正上面的「CAM_BACK 自遮挡 = 平台边界」**。`nus_mini` / `nus_mini_wide` 已重采,
`smoke_radar_collect.sh` 四判据全过。另记一条**与原点无关的平台边界**:`RADAR_FRONT` 落在 a2
前保险杠**外侧 0.30 m**,因为 a2(3.705 m)比官方用的 Zoe(4.084 m)**短 0.38 m** —— 任何刚体映射都修不掉。

## ★ 标定口径冻结(2026-09-23 用户裁决,Plan2.md §P-M.11)

用户验收两张交付图后拍板:**以上即标定最终口径,以后一律按此进行**,后续采集 / 导出 / 训练 / 评测
**不再逐轮重新推导**。冻结表的完整版在 Plan2.md §P-M.11,要点:ego 原点 = **后轴**
(`NUS_EGO_ORIGIN_X = −1.2563`)、ego 姿态 = **6DoF**、相机局部基 `CARLA_CAM_TO_NUS_CAM`(**det = −1**)、
像素约定 = **CORNER**、官方 rig 默认 / wide rig 并行、**12 路传感器刚体同移**。

**三条不变量**:① spawn 与落盘由**同一份常量**导出(「表对了、图错了」只查表全绿、只目检也全绿);
② 任何"实挂 vs 声明"判据**先 tick**;③ 写盘经 `paths.project_path()`。
**验收唯一口径** = `autodrivedata/calib/verify_nus_calib.py --offline --live` 的**十条判据、两代 rig 各跑一遍**
(⑨⑩ 是原点误差的唯一探针);**图不能替代它**。

**本口径下作废的旧结论**(防照旧文档"修正"回去):❌「CAM_BACK 自遮挡 = 平台边界」(§P-M.4);
❌ wide rig 用 ⑦ 对官方 rig 的区分度(§P-M.9);❌「官网 55/110/180 是 FoV」(那是方位角)。
✅ 保留「rig 必须与权重训练数据一致」(§P-L.1)—— **全部旧权重仍不可复用**。

## 待办(教程 7-15 中尚未落地的能力)

- [x] 单目测距(教程 08):GT 3D 投影框基线 ✅(见上)
- [x] 双目视差(教程 09):双目 rig + 三角测量 ✅(见上)
- [x] 累积语义建图(教程 11):`accum.py` + 多帧累积 ✅(见上)
- [x] 地面提取 / 聚类障碍物(教程 12/13)✅(见上)
- [x] 激光 SLAM(教程 14):纯 numpy 两段式降档 + C++ 位对齐对拍 ✅(见上;ROS 原生栈按裁决不做)
- [x] 多雷达标定(教程 15):point-to-plane ICP + overlap 判据 ✅(见上)
- [x] 3DGS(教程 16):gsplat mini 链路闭环,PSNR 如实 ✅(见上)
- [x] 环视相机标定(教程 03):三处缺陷全修 + 口径冻结 ✅(见上 §P-M.7~.11)

## 待办(§P-M.11 冻结口径下的下游任务)

> **本轮范围与采样口径见 Plan2.md §P-M.12**(2026-09-24):500 帧 / Town10HD_Opt 多 spawn point /
> **1600×900** / stride 5(= nuScenes 关键帧率)/ 3 帧时序窗口 / 两套留出口径都报。

- [x] **按冻结表大规模重采 MapTR 训练集** —— ✅ 2026-09-24:`outputs/surround_v2/seg{0..4}`,
      5 段 × 100 帧(stride 5),贪心最大最小距离选 spawn point(44/14/15/152/55,两两最近 114 m),
      合计路线 **1280.7 m**(旧集 300 帧只有 139.2 m)。**采集器第三处「声明 ≠ 渲染」同时修掉**
      (六路共用 fov=90 → 逐通道官方 fov;`live_common.rig_frame()` 为实时侧唯一落点)
- [x] **多段组装 + 留出划分基建** —— ✅ `assemble_maptr.py --segs-dir`(每帧带 `seg`/`frame_in_seg`);
      划分收敛到 `maptr_impl.dataset.select_frames()` 唯一落点;回归钉 `tests/test_maptr_select.py`
- [x] **重训 MapTR 自实现线(单帧基线)** —— ✅ 2026-09-24:128 ep(`--lr-halve 0`)、batch 6、
      wall 2 h 16 min,两个权重 —— `maptr_v2_single.pt`(400 帧 = seg0–3 全段)与
      **`maptr_v2_singleF.pt`(320 帧 = `--exclude-seg seg4` ∩ `--keep-in-seg 0:80`)**。
      **chamfer AP @`--score-thr 0.2`(另附 `--sweep` 曲线)**:帧级留出 `[80,100)` **0.3043** /
      路线级留出 seg4 **0.1114**(独立模型佐证 0.1039)/ 训练集自身 0.2727 ⇒ **决策点通过**
      (帧级 ≥ 旧 0.0674;旧值采于错误 rig 的旧数据,**是闸门不是 A/B**)。
      **★ 三条不许误读**:① 帧级留出 ≠ 泛化 —— 留出首帧与训练末帧**同街只隔 2.65–3.01 m**,而真训过
      这些帧的模型在同一批帧上只 0.3096 vs 未训过 0.3043(差 0.0053)⇒ 0.3043 是"同街"天花板,
      真泛化看路线级(差 2.7×);② 该 AP 是 **precision 均值、无 recall 项**,训练集**每个阈值上都低于**
      帧级留出(过预测更多:1.74× vs 1.21×)⇒ §5.11 的「泛化间隙 3.9×」不能用它复算;③ 帧级留出还叠了
      **GT 密度**混淆(11.34 实例/帧 vs 训练 7.87,各段尾段恰是路口)。类别级:`ped_crossing` 同街 0.2125
      → 未训路线 **0.0000**(pred 8/gt 139)⇒ 靠位置记忆。详见 Plan2.md §P-M.12
- [ ] **时序建图(online HD mapping)** —— **代码 + 单测已就绪**(`maptr_impl/temporal.py`、
      `--temporal-window`,相关 74 用例过;**运行时序暂不训练 = 用户 2026-09-24 裁决**,不准当顺手跑的
      下一步自动接上)。实现优先序 = **MapQR 或 MapTRv2 的时序版本**(用户已降级 StreamMapNet /
      MapTracker);窗口 = **3 帧 (t, t−1, t−2)**(用户裁决),历史帧走 `no_grad` 编码;窗口守卫的边界是
      **切分**不是段(帧级切分画在段内部,否则留出帧 80/81 的历史 79/78 在训练集里 = 泄漏 8/80 帧);
      与单帧基线**同段划分、同 score_thr** 出 AP
- [ ] **AutoLabel 接逐帧契约** `mapvec_pred/1`(`eval_maptr.py --out-frames` 已就绪)

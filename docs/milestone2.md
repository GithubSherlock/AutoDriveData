# 教程能力补全里程碑(2026-09-18)

> 16 篇 Carla 仿真教程(docs/Carla_Sim_Tutorial_01..16.md)能力在本仓栈的落地记录。
> 路线图与待办见 [Plan2.md](../Plan2.md);**2026-09-19 起新计划一律在 Plan2.md 制定**,
> [Plan.md](../Plan.md) 转为方案定案 + 历史执行记录(冻结)。

## ✅ HiVT-CARLA 轨迹预测主线(教程 02 升级——多智能体轨迹预测)

**交付**:CARLA 轨迹 → HiVT(TemporalData)训练管线闭环,含 3D 赛道记录。

- 采集:`bin/collect_traj.py`(定速重发修复后)Town10 与 Town13 运动轨迹
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

**交付**:`bin/collect_stereo.py` CARLA 双目 rig(基线 0.4m)采集 + `autodrivedata/stereo.py`
纯值双目链路 + `tests/test_stereo.py`(手算锚点 10 passed)。

- 三角测量 z=f·B/d;SGBM 视差(OpenCV 可选,CV_8U)+ 自研 NCC 纯 numpy 匹配
- 自监督损失:`reprojection_loss`(右图按视差右移重建左图)
- 落盘 `outputs/stereo/`(calib.json + left/right/depth 40 帧 + depth_pc 点云)

**实测**:定速 5.98 m/s(逐帧自证);SGM 近物点云 z≈5.5m ↔ GT 深度同值(三角测量链
与真值深度对得上);depth_check.png 目检图已生成。

**结论**:链路"视差→深度→点云"闭环成立;近物视差大/深度小、远物视差小/深度大。

## ✅ 累积语义建图(教程 11)

**交付**:`autodrivedata/accum.py` + `bin/build_accum_map.py`(语义单帧 → 多帧累积 BEV)。

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
`bin/calib_multilidar.py`(注入已知误差 → 判据)+ `tests/test_multilidar.py`(4 passed)。

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
- `bin/collect_3dgs.py` / `bin/collect_stereo.py` 改为 import 纯函数,bin 只剩 carla 编排

**结论**:采集器行为被单测锁定(回归测试先例);匹配纯函数 `match_dets_to_gt` 并入
`attribution.py`(P-D 生产口径共用,IoU 与 AP 评估同口径)。

## ✅ 3DGS 重建(教程 16,降档链路验证)

**交付**:`bin/collect_3dgs.py`(静态场景 360° 环绕采集,spectator 归位修复)+
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
`bin/slam_odometry.py`(前端)+ `bin/slam_backend.py`(后端)+ `bin/slam_cpp.cpp`(C++17 单文件,
零外部依赖,阶段 2 位对齐对拍)+ `bin/slam_diff_test.py`(对拍脚本)+ `tests/test_slam.py`(27 passed)。

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

## 待办(教程 7-15 中尚未落地的能力)

- [x] 单目测距(教程 08):GT 3D 投影框基线 ✅(见上)
- [x] 双目视差(教程 09):双目 rig + 三角测量 ✅(见上)
- [x] 累积语义建图(教程 11):`accum.py` + 多帧累积 ✅(见上)
- [x] 地面提取 / 聚类障碍物(教程 12/13)✅(见上)
- [x] 激光 SLAM(教程 14):纯 numpy 两段式降档 + C++ 位对齐对拍 ✅(见上;ROS 原生栈按裁决不做)
- [x] 多雷达标定(教程 15):point-to-plane ICP + overlap 判据 ✅(见上)
- [x] 3DGS(教程 16):gsplat mini 链路闭环,PSNR 如实 ✅(见上)

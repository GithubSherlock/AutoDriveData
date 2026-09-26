# 文件树:AutoDriveData 仓库结构与文件职责

> ## ⚠️ 目录重构进行中(2026-09-26 起)
>
> 本文件正在随一次**分阶段的目录重构**同步。**目标结构与权威计划见
> [Plan_fileTree.md](../Plan_fileTree.md)**;本文档是「现状索引」,两者冲突时以 Plan_fileTree 为准。
>
> 已完成:
> - **`autodrivedata/sim/`** —— 原 `bin/` 的 CARLA 层(22 模块 + `smoke_radar_collect.sh`),
>   测试在 `autodrivedata/tests/sim/`。命令从 `python bin/x.py` 改为 **`python -m autodrivedata.sim.x`**。
> - **`autodrivedata/calib/`** —— 标定层(14 模块),测试在 `autodrivedata/tests/calib/`。
>   **`calib.py` 已改名 `calib/core.py`**(`calib/calib.py` 会自反)。
>   调用点从 `from autodrivedata.calib import X` 改为 **`from autodrivedata.calib.core import X`**。
>
> **本文档的目录表尚未逐行重排**(排在重构的收尾阶段)——下面涉及 `bin/` 的各表可能仍列着已迁走的条目,
> **以磁盘为准**。全部阶段完成后会按新树重写 §1–§5。

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
├── autodrivedata/               # ★ 主包。**按目录分层**:包根=纯值(36 模块,不 import carla);
│                                #   `sim/`=CARLA 仿真交互层(22 模块,import carla)
├── maptr_impl/                  # ★ MapTR 参考自实现(torch2.x 现代栈)
├── maptr_official/              #   官方 MapTR/MapQR 项目侧胶水(仅适配器与配置)
├── bin/                         # ★ 可执行入口:采集 / 评估 / 可视化 / 探针(依赖 pycarla)
├── tools/                       # ★ 开放性工具(判据:不含本项目领域知识)
├── tests/                       # ★ 单测 + oracle 对比(autodrivedata env)
├── docs/                        #   文档:教程 / 里程碑 / 本文件
│
├── outputs/                     # 【未入库】全部产物的唯一落点(不展开,见 §7)
├── training/                    # 【未入库】地图矢量全量导出(数据侧,非训练代码)
├── lightning_logs/              # 【未入库】HiVT 训练日志与 ckpt(PyTorch Lightning 默认输出)
├── auto3dlabel/weights/         # 【未入库】3D 检测微调权重(AutoLabel 侧消费)
├── hdMapGitHub/                 # 【未入库】上游开源仓库克隆(HiVT/MapTR/MapQR,保持 pristine)
├── models/                      # 【未入库】模型权重落点(yolo11s-seg.pt;bin/sem_bev.py 消费)
│
├── build/                       # 【未入库】构建残留(setuptools 产物,可删)
└── .vscode/ .claude/ .pytest_cache/ .ruff_cache/ .ipynb_checkpoints/   # 【未入库】本地工具配置与缓存
```

**依赖方向(硬纪律)**:`bin/` → `autodrivedata/` / `maptr_impl/`;项目整体 → AutoLabel(3D 检测消费方)单向,**禁止反向**。

**「谁允许 import 什么」由 [tests/test_layer_guard.py](../tests/test_layer_guard.py) 的 `LAYER_RULES` 机械强制**
(**按目录**声明,不是一条全局禁令):包根 / `utils` / `gt` / `slam` / `export` 禁 carla+torch;
`calib` 许 carla 禁 torch;`sim` / `map` / `perception` 等按各自需要放开。
马甲库(`ultralytics` / `mmdet3d` / `mmcv` / `lightning` —— import 即拉起 torch)与字面量动态导入同样在守。

## 2 `autodrivedata/` — 按目录分层的主包

| 文件 | 职责 |
|---|---|
| `geometry.py` | 坐标转换唯一落点:CARLA 系 ↔ KITTI 相机系 ↔ nuScenes 系。`CARLA_TO_NUS`(对合)/ `nus_camera_rotation_to_carla`(相机自身系另需 `CARLA_TO_CAM`)/ **`nus_sensor_rotation_to_carla`**(LiDAR/雷达自身系与 nus 同轴序 ⇒ 两侧同阵,对纯 yaw 即 `yaw_carla = −az_nus`,6DoF 连 pitch/roll 一起正确翻过去)。`quat_normalize` 的头注归因订正:非单位来自**手抄 4 位小数**。**挂点原点单点真值**(§P-M.10):`NUS_EGO_ORIGIN_X = -1.2563`(CARLA actor 原点在**车身中点**、nus 官方表在**后轴中心** ⇒ 整套 12 路传感器偏前 1.2563 m)+ `nus_ego_translation` / `carla_actor_origin_to_nus_ego`(**只收 6DoF 三元组**,yaw-only 入口会静默丢掉 0.0642° 悬架俯仰)+ `nus_ego_rotation`(全 6DoF `ego_pose.rotation`;**UE 左手口径 ⇒ 正确分解是 `Rz(−yaw)·Ry(−pitch)·Rx(+roll)`,原样代入会静默反号**)+ `CARLA_CAM_TO_NUS_CAM`(相机**局部**基重排 `[e1,−e2,e0]`,正交但 **det = −1**;与全局基翻转相乘才抵消。拿 `M·R·M` 比相机会得到**恒 120° 的假误差**) |
| `calib.py` | KITTI 标定生成(内参/外参 → calib txt,含 `world_to_img` 投影共用件) |
| `camera_rig.py` | **环视相机 rig 唯一来源**:官方 nuScenes `calibrated_sensor`(6DoF 四元数)→ CARLA 采集口径 `NUS_CAMERA_RIG`(平移 y 翻号 + 姿态走 `nus_camera_rotation_to_carla`)。采集器 / 实时流 / 导出器同源;模块头注记录**镜像 bug**(`yaw_carla = −az_nus` 漏翻 ⇒ 四个侧/后相机左右互换)。平移一律 = 官方表 + `geometry.NUS_EGO_ORIGIN_X`(§P-M.10 后轴对齐);wide 的后三路常量是 **CARLA 口径**的 `NUS_WIDE_REAR_X_CARLA = -1.9000`(落盘 nus 侧 = −0.6437),命名即口径,别混。**头注的四元数模长归因已于 2026-09-23 订正**:非单位的来源是**本表手抄的 4 位小数**(如 `CAM_FRONT_LEFT` `|q|−1 = −5.04e-05`),官方 mini 120 条 `calibrated_sensor` 的 |q| 实测全为 1.000000000000 |
| `calib_probe.py` | **标定自证纯值件**:平面拟合(`fit_plane`/`fit_local_planes`)、相机射线/反投影(`cam_rays`/`project_world`/`backproject_depth`)、深度图采样(`collect_samples`/`DepthSamples`)、轴目标物质心法主点裁决(`estimate_axis_delta`/`estimate_delta_uv`)、镜像不对称度(`mirror_asymmetry`)、径向误差剖面 |
| `depth_codec.py` | CARLA 深度图编解码(`decode_depth`/`encode_depth`,BGRA→米)+ 采样口径(`CONVENTION_CENTER`/`CONVENTION_CORNER` + `sample_bilinear_many`)——**实测裁决 CARLA 光栅 = corner**(索引 i 即连续坐标 i;见 `probe_calib.py` A3/A4),`CENTER` 保留供对照。**坑:`decode` 与 `sample` 的像素索引约定必须一致**,差 0.5 px 在近处 = 米级深度误差 |
| `calib_live.py` | **实时标定槽纯值件**(`live_studio --calib`):`live_planes`(体素+抽样+逐点邻域平面,`LIVE_*` 廉价预算)/ `sample_camera`(复用 `calib_probe.collect_samples`,`CONVENTION_CORNER`,不开窗口极差)/ `residual_colors`+`paint_residuals`(着色,**与离线探针同源**)/ `summarize`+`CameraResidual`(样本 < `MIN_CAM_SAMPLES` 时 `median_abs is None`,**不许报假数字**)/ `self_occluded_cameras`+`near_fraction`(**相对**判据,见下)/ `hud_line`。**坑:自遮挡判据不能写死阈值**——近场占比随**画幅宽高比**变(CAM_BACK 1242×375 是 0.367、640×360 只有 0.195),故按同批可用相机的近场占比中位数定阈;平面是**世界系**故可跨 tick 复用(瓶颈全在拟合 ~100–150 ms vs 六相机采样 ~9 ms ⇒ 默认每 2 tick 重拟合) |
| `gt.py` | CARLA actor → KITTI label_2 GT 行 |
| `static_gt.py` | 静态目标/道路特征 GT(地图查询源:P2) |
| `traffic_light.py` | 交通信号灯状态 GT(动态时序层:状态归一/前向判据/相位查表) |
| `semantic.py` | CARLA 语义 LiDAR 标签 → KITTI 式强度合成(域差距修复) |
| `opendrive.py` | OpenDRIVE 1.4 解析(planView/lanes/objects/signals/junction) |
| `mapvec.py` | 地图矢量 GT 提取与采样(MapTR 口径:六类要素 + 裁剪 + 重采样) |
| `mapvec_schema.py` | 矢量预测对外契约 `mapvec_pred/1`(schema 校验/JSON 往返) |
| `mapviz.py` | 矢量投影与绘制:ego 系折线 → 相机像素 + BEV 面板。`intrinsics_from_k` **直读 K 的 cx/cy**(不重算);`calib_from_fov` 是**全仓唯一 fov→fx 落点**(主点 = 索引约定中心 `(w−1)/2`) |
| `rigviz.py` | **自车 + 传感器标定配置图**(纯值,PIL;见 Plan2.md §P-M.9):`draw_rig_layout` 一页三区 = 俯视(车体实测包围盒 + 逐相机挂点与视锥 + 短码)+ 方位环(重叠橙、盲区红带度数)+ 数字表(`通道 · 挂点 x,y,z · 方位角 · FoV · **az ± fov/2**`)。`azimuth_of` 是方位角算式的**第二份独立实现**(单测钉它与 `camera_rig.camera_azimuth_nus` 相等)。**分区是硬坐标**——图例压锥 / 方位环压表都踩过。接受**显式 calibs/fov 参数**,故"还没采过的候选 rig"只改常量就能出图。俯视图另有**后轴标记线**(洋红 = nus 原点)+ **空心灰圈 = 修正前挂点**(整体后移 1.2563 m 落在后轴线上,§P-M.10) |
| `fonts.py` | **覆盖层文本的唯一字体落点**(PIL 唯一依赖,不 import carla):`font_path()` 按「`AUTODRIVEDATA_FONT` → 系统 CJK → **CARLA 随包 `DroidSansFallback.ttf`** → DejaVu 兜底(并 warn)」解析;**能画中文吗 = 渲染探针**(`U+10FFFF` 的像素签名 = 该字体的 `.notdef` 签名,某字签名与它相同即豆腐块;`has_cjk` 要求 `文相机字` 四签名互不相同)——**不看文件名、不看 `fc-list`、不依赖 fontTools**。`sanitize` 把字体缺的码位(`REPLACE` 表,实测只 4 个)换成等价 ASCII、兜底 `?`,**绝不留豆腐块**;`get_font`/`draw_text`/`width`/`bbox`/`wrap`(按**实测像素宽**折行 —— 单行画超画布会被 PIL 静默裁掉,见 §P-M.9)是全部绘制的入口。**两条独立成因都在这解决**:① 本机字体族**一个 CJK 字形都没有**而代码硬写 DejaVu;② **PIL 没有字体回退链**,`ImageDraw.text()` 不传 `font=` 就用内置位图字体(同样无 CJK 且只有 ~11 px)。见 Plan2.md §P-M.8 |
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
| `export/nuscenes.py` | nuScenes 迷你集生成器(照 devkit 契约)。**官方标定表的家**:`NUS_CAMERA_CALIBS` 从 `camera_rig` **单点导入**(不再另抄一份)、`NUS_CAMERA_INTRINSICS`(官方 n015 逐通道 fx/cx/cy,1600×900)+ `NUS_CAMERA_FOV`(`2·atan((w/2)/fx)` 导出,不手抄)、`NUS_LIDAR_CALIB`(官方挂点 + 含 1.4289° up 轴倾角的四元数)、`NUS_RADAR_OFFSETS`(官方 n015 弧度)。`NusSample.ego_rotation_nus` 是**全 6DoF 四元数**(不是 `ego_yaw_nus`:实测悬架俯仰 +0.0642°,拍平会让 `ego_pose ⊕ calibrated_sensor` 与世界系差这个量级,§P-M.10)+ `_quat_of` 落盘口。`points_sensor_to_global_nus` 的 `calib_quat` 是**四元数不是 yaw**(LiDAR 倾角表达不了 yaw-only;雷达 pitch/roll 精确为 0 ⇒ yaw-only 无损,故两者共用一条链) |

## 3 `maptr_impl/` — MapTR 参考自实现(§5.11 C/D 阶段)

| 文件 | 职责 |
|---|---|
| `model.py` | 模型组装:ResNet50+FPN backbone + GKT BEV 变换 + 分层 query head。`temporal_window=K>1` 时在 GKT 与 head 之间插 `TemporalFusion`(**head 一行不改**);`forward` 按 `images` 是 dict / list 分派单帧/时序,历史帧在 `torch.no_grad()` 下编码(memory bank 语义)。另有 `load_map_weights()` = 权重口径的**唯一落点**(单帧权重 → 时序模型 = 缺失 `fusion.*`、放行热启动;时序权重 → 单帧模型 = 多余键、报错) |
| `temporal.py` | **MapTRv2 时序版(§P-M.12 阶段 4)**:`warp_bev`(历史 BEV 按两帧相对位姿扭到当前 ego 系,整链 float64 + 完整 3D,只在最后取 (x,y)) + `TemporalFusion`(拼接 → 1×1 conv → 残差加回,`proj` **零初始化** ⇒ 第 0 步与单帧模型逐位相同)+ `ego_rotations`(位姿六元组 → ego→world 旋转阵,与 `gkt.cam_world_pose` 同源同序)。两处口径:**变换顺序 `R_prevᵀ·(R_cur·p + t_cur − t_prev)`**(漏转置的盲点是 `ψ_prev=0` 而非 `Δψ=0`,姿态全零时测试会假绿)、**grid 末维必须是 (gx, gy)**(BEV 张量 H 索引吃 gy)。两种错法的误差律与实测格位移写在模块头注 |
| `gkt.py` | GKT(Geometry-aware Kernel Transform):环视相机特征 → BEV 特征。**两处已修 bug(2026-09-22)**:①infos 六元组 `[x,y,z,yaw,pitch,roll]` 必须换序成 `(pitch,yaw,roll)`(`_ROT_TO_CARLA`),漏了 = 5/6 相机指向错;②K 必须缩到**特征图**分辨率(`scale_k`,FPN P2 = 311×94),漏了 = 全分辨率像素与 `feat_w−1` 比。两坑叠加把 BEV 有效覆盖从 94.6% 打到 1.25% |
| `head.py` | 分层 query head:实例级 query + 点级 query + 置换等价匹配 |
| `dataset.py` | B2 infos json → 训练数据集(图像加载 + 位姿/标定透传 + GT 解析)。另有 **`select_frames()` = 留出划分的唯一落点**(train/eval 共用;`--seg` / `--exclude-seg` / `--keep-in-seg` 三选择器取交集,段名拼错**报错而非静默给 0 帧**,旧单段 infos 无 `seg` 键时不带选择器照旧可用)+ `parse_segs` / `parse_frame_range`(§P-M.12)。**`window=K` 时序窗口**:`history_windows()` 的守卫边界是**切分**不是段(帧级切分画在段内部 ⇒ 必须让窗口只从**本切分自己的帧列表**里取同段前驱,否则留出帧 80/81 的历史 79/78 在训练集里 = 泄漏);拿不到完整历史的帧**丢弃并计数上报**(`ds.dropped`),不静默截短;`window=1` 结构与单帧基线逐字节相同 |
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
| `collect_nus.py` | nuScenes 迷你集:6 相机 + LiDAR + 5 雷达。**全传感器标定「渲染位姿 = 声明位姿」同源**(2026-09-23,§P-M.7):相机走 `NUS_CAMERA_RIG`(逐相机 6DoF 挂点)+ 逐通道蓝图 `fov`;LiDAR 走官方挂点 + 由四元数**导出**的姿态(`LIDAR_ROT` 不手抄);雷达偏航 `−az_nus` **由 `NUS_RADAR_OFFSETS` 导出**。历史缺陷(已删 `CAM_YAW_OFFSET` 镜像表 / 雷达猜测表 / LiDAR 无 rotation)见模块头注的对照表。判据复现器 = `bin/verify_nus_calib.py`。**挂点原点**:位置/姿态全走 `geometry.nus_ego_translation` + `nus_ego_rotation`(后轴,§P-M.10);挂传感器前先让车静置收敛(`ego_settle`,实测 8 tick),`ego_pose` 写全 6DoF |
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
| `assemble_maptr.py` | 环视采集 + 地图矢量 → MapTRv2 infos 同构 json。`--surround`(单段,旧用法)/ `--segs-dir`(多段,收 `seg*`)**二选一**;多段按帧带 `seg` / `frame_in_seg`,`token` = `{seg}_{i:06d}`(**全局唯一**,`--out-frames` 拿它做文件名)。`roots_and_prefixes()` 决定 `data_path` 前缀:单段**空**(与旧产物逐帧等价,实测只有 3 个键变)、多段 `segK/`(§P-M.12) |
| `merge_train_infos.py` | 拼接多组环视训练数据 → 合并 infos(帧号连续重排) |
| `convert_mapvec.py` | 地图矢量 → MapTRv2 annotation 口径 |
| `export_mapvec.py` | 地图矢量导出全量/帧级裁剪 json + BEV overlay |
| `assemble_traj_pt.py` | CARLA 轨迹 → HiVT TemporalData 组装(纯值) |
| `convert_hivt_pt.py` | plain dict → HiVT TemporalData(在 hivt env 跑) |
| `prepare_official_dataset.py` | 环视数据 → 官方栈可直吃的 nuScenes 形状数据集(已终止线) |

### 4.3 训练

| 文件 | 职责 |
|---|---|
| `train_maptr.py` | MapTR 训练入口(单帧过拟合 = 正确性锚点;多帧 = 常规训练)。`--seg` / `--exclude-seg` / `--keep-in-seg` 走 `select_frames`(`--frames 0` = 筛后不截断);`--temporal-window K` = MapTRv2 时序版(与 `eval_maptr.py` **必须同值**),训练集丢弃数经 `ds.dropped` 上报;**长训一律 `--lr-halve 0`** —— 默认 12 会让 128 ep 后半程 lr 归零,平台是 lr 死掉不是收敛;过拟合闸门由 `is_single_frame_anchor(n_samples)` 判定,**按实际训练样本数而不是 `--frames` 标志**(`--frames 0` 是"不截断",按标志判会把 400 帧训练误判成单帧锚点并打假 FAIL + 退出码 1,见 §P-M.12) |
| `train_3dgs_mini.py` | 3DGS mini 训练(gsplat 光栅化) |
| `finetune_synth.py` | 合成 KITTI → pointpillars_kitti 微调(复用 AutoLabel train3d) |

### 4.4 评估

| 文件 | 职责 |
|---|---|
| `eval_2d_ab.py` | P1 逆光 A/B:冻结 YOLO11s 在两个 KITTI root 的 2D AP 对比 |
| `eval_attr.py` | 失效归因评估:多跑 × 距离/框高/TTC 网格 + 漏检画像 |
| `eval_kitti.py` | GT vs AutoLabel 伪标签比对报表(比对层 CLI) |
| `eval_maptr.py` | MapTR 评估:权重 → 逐帧推理 → 四类 chamfer AP(+ 逐帧契约落盘)。选择器与 `train_maptr` **同一份实现**(`select_frames`),`--start` / `--frames` 在**过滤后**的列表上再截(§P-M.12) |
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
| `live_common.py` | **实时可视化共享件**(从 view_stream 抽出):多槽 MJPEG 服务(单端口 `/stream/<name>` + `/` 索引页)/ **拼图(两套:`compose_grid` 等尺寸 + **尺寸守卫**,不符即 `ValueError` —— `paste` 源图大于目标框时只贴左上角、静默裁;`compose_rows` 按行拼、每格**原生像素**,studio 三层用它)** / GT overlay / **环视 rig(两代口径 `rig_spec`:`nuscenes` 逐相机 `SENSOR_MOUNTS` + 官方 6DoF 姿态,`legacy` 共用 `SENSOR_OFFSET` + 235/125;`resolve_rig` 按权重名选)** / 第三方视角 / 键盘 —— view_stream 与 live_studio 共用。**`build_surround_rig(..., kind=)`** 可挂 `rgb` 或 `depth`(深度槽必须与 RGB 槽**同挂点同内参同分辨率**,否则 overlay 无法逐像素对齐);`draw_hud(..., y=)` 支持第二行(条带 16 px) |
| `live_studio.py` | **8 路 studio**:6 相机 + BEV + 第三方 + `grid` 拼图槽,各占一路;`--keyboard` 折进 tick 循环(WASD 开采集);第三方非 attach 每 tick 摆位。**拼图 = `GRID_ROWS` 三层**(①左前/前/右前 ②右后/后/左后 ③第三方 + BEV),**每格原生像素不缩放**(相机 1242×375 / 第三方 640×360 / BEV `--bev-size` ⇒ 画布 3726×1170;旧 4×2 等尺寸布局把相机图裁到 621×187 丢掉地面,见 Plan2.md §P-L.6)。**`--slam` 接在线 SLAM**(挂语义 LiDAR → `SlamWorker`,BEV 槽画地图点/轨迹;`--slam-async` / `--slam-voxel` / `--slam-max-gap` / `--slam-report` 落验收 JSON)。**`--calib` 接实时标定监看**(另挂 6 深度相机同挂点同内参 + `sensor.lidar.ray_cast`,LiDAR→世界平面→投影回相机按深度残差着色画进各相机槽,`draw_hud` 第二行报 pooled |e| 与逐路样本数,`--calib-report` 落 JSON;`--calib-refit` 默认 2 tick 重拟合一次,预算见 `calib_live.py`)。**`--video` 落八视角视频段**(拼图槽逐帧写 mp4,cv2/mp4v 惰性开编码器;`--video-fps` 标称帧率 / `--video-tile` 放大倍数) |
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
| `probe_calib.py` | **标定自证探针(七锚 A0–A6)**:spawn 6 RGB + 6 depth + LiDAR + 施工锥 → `outputs/calib_check/{report.json,overlay.png}`。A0 光轴 vs 官方方位角 / A1 侧别一致性 / A2 实例分割解码(`id = G + 256·B`)/ A3 LiDAR-平面-深度图交叉验证(裁决**像素约定 = corner**)/ A4 轴目标物掩膜质心回归主点 / A5 实挂 vs 规格 / A6 主点锁定 `(w−1)/2`。**判据全数值,不目检** |
| `viz_calib_check.py` | **标定修正的人工复核图**(`probe_calib` 的数值结论 → 人能对着看的图,判据数字烧进画面):①`check_geometry.png`(纯值,不依赖 CARLA,先落盘)——官方方位角极坐标轮(实线=修正后 / 淡线=历史字面值)+ 镜像差表(**不 wrap**,217.2° 折成 142.8° 就看不出镜像)+ A4/A3 读数;②`check_raw.png`/`check_overlay.png`(A3 六相机**同一帧**的 raw 与残差 overlay,两张逐像素差 = 画上去的点数);③`check_rig_ab.png`(同一 ego、同一批施工锥,`nuscenes` vs `legacy` 各拍一遍 → **世界左方的锥出现在哪一路**即镜像的直接证据)。落 `outputs/calib_check/check_*.png` + `viz_summary.json` |
| `verify_nus_calib.py` | **`collect_nus` 验收判据的复现器**(全数值,不目检;落 `outputs/nus_calib_check/report.json`)。`--offline`(不需 CARLA):③ 雷达点落进**自身 FOV** 占比(以官方 R 为锚 —— 点云在传感器自身系,+x 即光轴,不转到 ego 系再比就是恒真)、④ LiDAR 复现 `num_lidar_pts`(官方集 1.0000 vs 单位阵消融 0.1684)、⑤ 相机内参 vs 该 rig 声明表;`--live`:① 相机实挂 vs 声明(`live_common.mount_deviation_of`,**必须先 tick**)、② 雷达实挂 vs 官方 az、⑥ 渲染 FOV vs 蓝图 fov(复用 `probe_calib` 的锥体/掩膜回归,**不另写实例分割解码**)、⑦⑧ 转 `rig_check.instance_probe`。**⑥ 两个实测坑**:每 tick 必须**抽干全部相机队列**(否则未测相机积压陈旧帧 ⇒ 掩膜恒 0,症状是"只有第一个相机测得出"且与距离无关)、Z 取**阶梯** `(20,14,10,8,6)`(地图遮挡随出生点变,写死会假失败)。**明令禁止 `num_radar_pts` 作判据**(跨 5 通道求和,官方 R 也只复现 0.6279)。**+ ⑨⑩(§P-M.10)**:⑨ 世界系链(`declared = 落盘表 ⊕ **实测**后轴位姿` vs `rendered = CARLA 实挂经共轭`,12 路逐位比 —— ①② 相对同一个 ego,**对原点误差在结构上盲**,必须有它);⑩ 独立复测后轴(双偏航自解,**不读常量**,防常量腐化静默错位)。`--rig {nuscenes,wide}`:③④ 与相机无关原样适用,其余换该 rig 的声明表;默认落点按 rig 分叉(`nus_mini[_wide]` / `report[_wide].json`,**不互相覆盖**) |
| `rig_check.py` | **判据 ⑦⑧ 的唯一执行点**(`RigCameras` 按任一 rig 的**声明位姿/FoV** spawn 6 路 instance_seg,1600×900):⑦ 逐像素数 **ego 自己的 actor id**(对照:**修正前**官方 rig `CAM_BACK 619189 px = 42.9992%`;§P-M.10 原点修正后**两代 rig 六路全 0 px** ⇒ 该判据不再是 wide 的区分度,wide 的取舍是 FoV spec 换覆盖率);⑧ 往重叠区正中摆施工锥,两路掩膜都命中才算共视。**两条边界写死在模块里**:锥心抬 `COVIS_Z_LIFT=0.45 m`(否则擦车顶被自车挡,读数变成"有没有被自车挡")、几何重叠 < `MIN_COVIS_OVERLAP_DEG=5°` 的对**如实 `skipped`**。★ `common_band()` 是核心订正:**方位轴重叠(无穷远)≠ 有限距离下的共同可见**(挂点视差最多 1.7°,官方 `FL↔BL` 11.20° → 12 m 处 4.838°) |
| `viz_rig_check.py` | **rig 的两件目检交付物**(`--rig`,`--live` 需 CARLA;落 `outputs/calib_check/`):`rig_layout_{rig}.png`(纯值配置图,`rigviz.draw_rig_layout`)、`views_{rig}.png`(六视角**原生像素**拼图 + 逐格 `az ± fov/2` 与 ego 像素读数 + 底部**线性方位尺**:逐相机一条泳道 ⇒ 竖线穿过的行数 = 该方位被几路覆盖;ego 像素**就地染品红**,0 px 时是空操作)、`report_{rig}.json`。**图不能替代 `verify_nus_calib`**:"声明 ≠ 渲染"在图上看不见(§P-M.7)。两个踩坑:`360.0 % 360 == 0` 会让 `rectangle` 抛 `ValueError`(跨 0° 扇区必须钳,不取模)、`np.asarray(PIL)` 是只读视图(染色必须 `np.array` 拷贝再 `Image.fromarray`) |
| `smoke.py` | M0 smoke:CARLA headless 连接 → 同步模式 → 各取一帧落盘 |

### 4.7 共用件与运维脚本

| 文件 | 职责 |
|---|---|
| `carla_common.py` | 采集公共件:位姿换算 / NPC 摆放 / 传感器参数 / 同步模式 / 灯态归一与绘制 |
| `assemble_and_merge.sh` | 组装 + 合并 infos 的批量编排 |
| `finalize_maptr_600.sh` | 600 帧扩数据轮的收尾编排 |
| `smoke_radar_collect.sh` | 雷达采集冒烟 |
| `setup_maptr_official.sh` / `run_official.sh` | 官方栈环境搭建与运行(**已终止线**,重建设路子在 Plan.md §5.12) |

**`tools/` — 开放性工具(顶层;判据:不含本项目领域知识)**

| 文件 | 职责 |
|---|---|
| `carla_server.sh` | CARLA 服务器启动/停止(GPU 修复栈 + Vulkan 兼容层自愈) |
| `gpu_fix/` | GPU 修复栈:`install.sh`(NVIDIA 用户态补齐 + shim 安装)+ `mhookshim.c`(LD_PRELOAD shim 源码) |
| `clear_cache.sh` / `gitpush.sh` | 【未入库】本机磁盘清理 / 推送辅助(环境维护,非项目代码) |

## 5 `tests/` — 单测与 oracle 对比(autodrivedata env)

| 类别 | 文件 | 说明 |
|---|---|---|
| 纯值库单测 | `test_geometry.py` `test_calib.py` `test_gt.py` `test_compare.py` `test_paths.py` `test_scenarios.py` `test_static_gt.py` `test_traffic_light.py` `test_semantic.py` `test_radar.py` | 手算断言,不依赖 carla / AutoLabel |
| **层守卫** | `test_layer_guard.py` | **包纪律的可执行版本**(Plan_fileTree.md §3):`LAYER_RULES` = 目录 → 禁止 import 的三方名,最长前缀匹配。旧版(`test_paths.py` 的整包禁令)的两个洞已堵:**马甲库**(`ultralytics`/`mmdet3d`/`mmcv`/`lightning` 会拉起 torch 但字面无 torch)、**字面量动态导入**(`importlib.import_module("x")`)。`TestPackageLayers` 扫真实包 + 强制新子目录必须显式声明;`TestLayerGuardSelfCheck` 用**合成源码注入**做立论自证(12 条:抓得住三类违规,且规则能区分、不是"见 carla 就红") |
| 地图矢量线 | `test_opendrive.py` `test_mapvec.py` `test_mapvec_schema.py` `test_mapviz.py` `test_chamfer_ap.py` `test_chamfer_gpu.py` | 含闭式解手算锚点与真实 xodr 计数锚点 |
| 教程能力线 | `test_mono_depth.py` `test_stereo.py` `test_multilidar.py` `test_slam.py` `test_accum.py` `test_ground.py` `test_cluster.py` `test_collect_rig.py` | 各含手算锚点;`collect_rig` 兼作采集器回归先例 |
| SLAM 精度/在线线 | `test_slam_eval.py` `test_live_slam.py` | `slam_eval`:ATE/RPE 手算锚点 + 杆臂方向(不补杆臂 ATE 2.44×);`live_slam`:与离线 `slam_odometry` **逐帧同输入同输出**(<1e-12)+ `SlamWorker` 滞后有界/止损/同步模式 |
| 实时可视化 | `test_live_common.py` | `compose_grid` **尺寸守卫**(不符必抛,防 `paste` 静默裁)+ `compose_rows` 每格**原生像素**逐像素等于源图 + studio `GRID_ROWS` 三层行序 + **`rig_spec`/`resolve_rig`**(两代 rig 口径与"按权重选":legacy 共用挂点 + pitch/roll=0,nuscenes 逐相机 6DoF;显式指定不被文件名覆盖)+ **`mount_deviation_of` 规格对账**(相机与雷达共用;矩阵顺序写反 ⇒ 平移爆掉而偏航仍 ~0、偏差随 ego 离原点变远而变大、legacy 实挂对 nuscenes 规格必报 110°、tick 前全 0 陈旧位姿**不许**判成"通过";`rig_mount_deviation` 是它的薄封装)+ `draw_hud(y=)` 第二行(第一行逐像素不变、`y=0` 与旧行为一致)。import `bin/` 模块需运行时加 `sys.path`(非包),静态分析跟不到 ⇒ 就地 `pyright: ignore[reportMissingImports]` |
| 配置图 / 覆盖表 | `test_rigviz.py` | **交付物图的数值侧回归**(纯值,PIL + numpy)。`TestCoverageTable`:wide 三个盲区**逐项等于设计预算**(7.3353/6.0984/1.7224 = 15.1561°、覆盖 0.9579)、官方 rig 零盲区;重叠对的 `span` 必须落在两个相机各自的扇区里(共视探针按它摆锥)。`TestAzimuthIndependentImplementation`:`rigviz.azimuth_of` == `camera_rig.camera_azimuth_nus`(两套独立实现)。`TestRigLayoutFigure`:**盲区红弧"有当且有、无当且无"**(官方 0 个盲区 ⇒ 0 红像素;wide 有 ⇒ >0)+ 六通道都有画色与短码。`TestRulerLanes`:底尺**跨 0° 不崩**(`360.0 % 360 == 0` 会让 `rectangle` 抛 `ValueError`)+ 盲区红**列数** ∝ Σ盲区度数 + 六条泳道都画出来 + 页脚折行后每行实测宽 ≤ 画布且不压数字表。**不钉排版/错别字**——那些写成断言只会得到"改个字就红"的脆测试 |
| 绘制字体 | `test_fonts.py` | **中文字形不许静默变豆腐块**的回归钉。`TestProbe`:探针立论自证(`U+10FFFF` 在任何字体下都落 `.notdef`)+ **DejaVu 被正确判否**(它有 `−`/`°`/`★` 却画不了中文 ⇒ 判据不是"文件在不在")+ 生效字体实测能画中文。`TestRendering`:两个不同汉字在**画布上必须像素不同**(最强钉——豆腐块下它们逐像素相同)、`sanitize` 后零缺字 / 替换目标自己画得出 / 不等长(排版不错位)、CJK 宽 ≈ 2× ASCII(HUD 底条据此定宽)。`TestDrawnStringsAreRenderable`:**AST 扫全仓绘制字符串**(8 个绘制模块 × 绘制调用实参 + `hud_line` 之类构造器**函数体**——漏后者 `calib_live` 整行中文 HUD 会逃检)⇒ 逐个 `sanitize` 后零缺字;另有**根因钉** `test_no_module_draws_with_a_bare_text_call`(不许出现不带 `font=` 的 `d.text(...)`)与每模块 `import fonts` 钉 |
| MapTR 数据划分 | `test_maptr_select.py` | **留出划分与多段组装的回归钉**(§P-M.12)。两条被测契约都是**静默失效型**:划分有交集只会让 AP 看起来更高、`data_path` 前缀写错只会让旧命令指错文件 —— 都不报错。`TestSelectFrames`:路线级(`--exclude-seg seg4`)与帧级(`--keep-in-seg 0:2`)两侧**互斥且并集为全集**、边界左闭右开、选择器取交集、**段名拼错必须 `ValueError` 而不是空列表**、旧单段 infos 不带选择器照旧可用。`TestArgParsing`:`parse_segs` 空串/纯空白 → `None`;`parse_frame_range` 拒绝 `80` / `80:100:2` / `100:80` / `5:5`。`TestRootsAndPrefixes`:单段前缀空 / 多段 `segK/`、glob 只收目录、必须且只能给一个来源。`TestHistoryWindows` / `TestWindowDataset`(时序窗口,§P-M.12 阶段 4):窗口**不跨段**、**不跨切分**(训练/留出各自只见自己的帧,窗口帧 ⊆ 本切分池)、段首帧丢弃且计数上报、`frame` 在段缝连续故**不许**当历史键、`window=1` 返回结构与单帧基线逐字节相同。`TestOverfitGate`(假警报型,§P-M.12):过拟合闸门按**实际训练样本数**判(`is_single_frame_anchor`),`320/400/500` 全不是锚点;外加**静态根因钉** —— AST 扫源码禁止 `args.frames` 与整数字面量比较(只禁这一形态,`args.frames > len(sel)` 的截断检查合法),因为"再写回 `args.frames > 1`"就是本条缺陷的复发式 |
| MapTR 自实现 | `test_gkt.py` `test_head.py` `test_device.py` | 单帧过拟合正确性锚定。`test_gkt` 三条**回归钉**:`test_pose_rotation_order_is_carla_convention`(换序)、`test_scale_k_to_feature_resolution`(K 缩放)、`test_gkt_valid_coverage_on_real_rig`(真实 rig BEV 可见率 ≈94%,修前 1.25%) |
| MapTR 时序 | `test_temporal.py` | **§P-M.12 阶段 4 的回归钉**,判据全是手算可验的闭式 oracle(线性场在均匀栅格上双线性插值恒等 ⇒ 采样值必须等于解析值,一条断言同时钉死归一化仿射 / `align_corners` / 两轴顺序)。`TestWarpBeV`:纯平移整格位移**逐元素相等**、float32 位姿量化上界、漏转置对照可被区分且**钉住它的盲点是 `ψ_prev=0`**、轴对调检查(第 49 列 / 第 99 行)、相对 pitch/roll 有位移而共同 pitch/roll 是 no-op。`TestTemporalFusion`:零初始化恒等、通道块序 `[当前, t−1, …]`、历史条数/`n_hist<1` 报错。`TestEgoRotations`:与 `gkt.cam_world_pose` 的 `r_e` 同源 + det=+1。`TestModelSeam`:**零初始化下时序版与单帧版输出逐位相同**(A/B 可比性的前提)+ 融合层在梯度路径上(判据用 **bias** 的梯度 —— `weight` 的梯度还要求 head 采样区落在 ReLU 激活区,实测可恰好为 0,拿它当判据会假红)+ **换历史帧必须改变输出**(证明历史 BEV 真的流到 head;前提 self-check:`num_vec` 太小会让锚点全落在 rig 覆盖外、head 对 BEV 免疫)。`TestCheckpoint口径`:`load_map_weights` 的两种错配分类(单帧→时序 = 放行热启动 / 时序→单帧 = 报错) |
| 标定自证 | `test_calib_probe.py` `test_depth_codec.py` `test_probe_calib.py` `test_calib_live.py` | `calib_probe`:轴目标物质心法在合成对称掩膜上精确复原注入的 cx;ray-plane 深度闭式解手算锚点;单侧可见性判据(渲染更近才判遮挡,对称窗口极差会误杀 95%)。`depth_codec`:深度编解码往返 + 像素约定。`probe_calib`:**实例分割解码公式**(`id = G + 256·B`,含两个旧错误候选作反例)+ A0/A1 锚 + "A1 能拦下历史镜像 bug" 的反向自证。`calib_live`:**判据不许报假数字**(样本不足 ⇒ `median_abs is None`,HUD 报"无数据"而非 0.000)+ 着色用 `index = u`(corner)+ `pooled_median` **等权** + 自遮挡**相对**判据(0.195 这个实时分辨率实测值也必须判得出来——绝对阈值 `> 0.2` 会漏) |
| 落盘契约 | `test_export_kitti.py` `test_export_nuscenes.py` | 路径/字段与消费方契约一致;`test_export_nuscenes` 另有**逐通道相机内参 == 官方 n015 K**(内参以前零覆盖)+ 蓝图 `fov` == `2·atan((w/2)/fx)` + LIDAR_TOP 落盘四元数 == 官方值(非单位) |
| devkit 表读取 | `test_nuscenes_cali_sensors.py` | 读 `sensor`/`calibrated_sensor`,打印每传感器相对 ego 的位姿 + 视线轴方位角/俯仰角。两处坑:①**视线轴因 modality 而异**(相机自身系 z 前 ⇒ 视线轴 +z;激光/雷达 x 前 ⇒ +x),对相机用 `Quaternion.yaw_pitch_roll` 读出的三个角无几何意义;②官方 mini 每通道 10 条记录但只 2 套位姿(n015 6 条 / n008 4 条),车辆/地点字段只能经 sample_data→sample→scene→log 反查。**可直接 `python` 跑**(打印),也可 pytest 收集;缺 devkit / 缺 dataroot 自动跳过。`test_repo_lidar_quat_is_identity` 于 2026-09-23 **改名为 `test_repo_lidar_quat_matches_official`** —— 旧断言测的正是缺陷(单位四元数) |
| oracle 对比 | `test_geometry_carla_oracle.py` `test_calib_oracle_autolabel.py` `test_gt_oracle_autolabel.py` `test_nuscenes_oracle_autolabel.py` | **需 autolabel env / CARLA 机器**,缺失时自动 skip |
| nuScenes 约定纯值 | `test_geometry_nus.py` `test_nuscenes_calib_consistency.py` | **不 skip、任何 env 可跑**。`test_geometry_nus`:四元数归一化(非单位的**来源是手抄 4 位小数舍入**,≥1e-5;官方原值 |q| = 1.000000000000)+ 相机 rig 推导钉(`yaw_carla = −az_nus` 到 1e-9、平移只翻 y、6DoF 不可降 yaw-only、历史字面表的 110°/217° 偏差量级)+ **LiDAR/雷达同一条对合规则**(`R_carla = CARLA_TO_NUS @ R_nus @ CARLA_TO_NUS`,雷达历史表差 94–136°、LiDAR 倾角 1.4289°)。`test_nuscenes_calib_consistency`:**「渲染 = 声明」同源钉**——① `collect_nus.CAM_YAW_OFFSET` 已删 / `CAM_ATTRS` 已拆、② spawn 位姿由 `NUS_CAMERA_RIG` / 官方雷达表导出、③ `LIDAR_MOUNT` + `LIDAR_ROT` 由四元数导出(≠ KITTI 线 `SENSOR_OFFSET`)、④ `CAM_FOV` == `2·atan(800/官方fx)`(fx 表在测试里独立硬编码,故非恒真)、⑤ `TestFovCriterionShape`:**AST 静态钉判据 ⑥ 的两个实测坑**(`world.tick()`/`q.get` 只能出现在 `drain` 内 = 每 tick 抽干全部相机队列;Z 必须是从远到近的**阶梯**、横移表关于 0 对称且 |frac| < 0.5 但 > 0.4)。**这是 §P-M.7「表对了图错了」的回归锁**。**wide rig 分支(§P-M.9)**:`TestWideRigDerivation`(前三个与官方**逐位相等**、后三路 `x == NUS_WIDE_REAR_X` 且在车身最后点之后、`y/z`+`pitch/roll` 保留官方、方位角由四元数**独立反解**得 145/180/215 且 `yaw_carla == −az_nus`)、`TestWideIntrinsics`(`fx` 闭式、主点 corner、fov↔K 往返、**`CAM_BACK` 必须窄于 180° 且 `det(K) > 0`** 的反回归闸门)、`TestCollectNusRigSelection` / `TestVerifyRigSelection`(拨模块级 `RIG` 后取表与内参必须跟着走 —— 防"wide 的验收静默拿官方表去比") |

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
| `surround_*` | 环视 6 相机数据集(`surround_train` / `surround_p3` / `surround_town13` / `surround_pred` 逐帧契约)。**`surround_v2` = §P-M.12 的 1600×900 官方口径训练集**:5 段 × 100 帧(stride 5,合计路线 1280.7 m),`seg0..seg4` 各为一段独立路线 + `map_infos.json`(500 帧,带 `seg`/`frame_in_seg`);`collect.log` 为采集计时 | `collect_surround.py` / `assemble_maptr.py` / `eval_maptr.py` |
| `traj_town*` | 多 agent 轨迹数据集(HiVT 输入) | `collect_traj.py` / `assemble_traj_pt.py` |
| `maptr_*.pt` | MapTR 权重(`maptr_600` / `maptr_1000` / `ep256` / `ep512` + `.opt` 优化器态)。**⚠️ 全部四个已标废弃**(2026-09-22,§P-M):采集时用的 rig 是错的(镜像 + pitch/roll 硬编码 0),数据本身错 ⇒ 重采重训。文件保留仅供历史对照与 legacy 路径回归,勿作新实验基线 | `train_maptr.py` |
| `maptr_v2_single{F,}.pt` | **§P-M.11 冻结口径下的第一批新权重**(1600×900 / `surround_v2`)。`maptr_v2_single` = 路线级训练集(seg0–3 × 100 = 400 帧,留出只能评 seg4);**`maptr_v2_singleF` = 帧级口径(320 帧 = `--exclude-seg seg4` ∩ `--keep-in-seg 0:80`),一个模型给两套留出**(帧级 seg0–3 `[80,100)` / 路线级 seg4)。日志同名 `.log`;**结尾的 `FAIL` 是过拟合闸门误报**(见 `train_maptr.py` 行,修于 2026-09-24) | `train_maptr.py` |
| `maptr_600` / `maptr_1000` | MapTR 扩数据轮的 infos 与图像。**两轮的 `images/` 均已清理**(`maptr_1000` 于 2026-09-19 删 5.5 G;`maptr_600/images` 3600 图同期发现已不存在),现仅存 `maptr_600/map_infos.json`(33 M);`maptr_1000/` 已整目录不存在。**权重 `.pt` 均保留**。复现训练需重跑组装链(`surround_train` + `surround_p3` + `training/map/Town10HD_Opt_full.json` → `bin/finalize_maptr_600.sh`) | `assemble_maptr.py` / `merge_train_infos.py` |
| `kitti_sweep_day_clear_8` | **仅存**的速度档(70 帧 @8 m/s)。`_4` / `_12` 于 2026-09-20 清理(§5.10 结论已归档);重采 `collect_ab_route.py --speed N` | `collect_ab_route.py` |
| `kitti_wet_road` / `kitti_dense_rush` | **P1-6 候选**场景探针(各 12 帧,尚无 A/B 版)。同批的 `rain_night` / `dense_fog` 因已有 70 帧 A/B 版而列入待清;`heavy_rain` / `night_clear` **无** A/B 版,12 帧版是唯一数据 | `collect_drive.py` |
| `kitti_slam` / `slam_gt/` | SLAM 数据集(velodyne + **pose 真值**)与轨迹/精度产物(`traj_raw.json` / `icp_stats.json` / `eval_*.json` / **B 期验收 `accept_sync|accept_async|accept_parity_full.json`**) | `collect_slam.py` / `slam_odometry.py` / `eval_slam.py` / `live_studio.py --slam-report` |
| `calib_check/` | 标定自证产物:`report.json`(七锚 A0–A6 全数值)+ `overlay.png`(6 相机三层 overlay)+ **`check_{geometry,raw,overlay,rig_ab}.png` + `viz_summary.json`(人工复核图,见 `viz_calib_check.py`)**;`live.json` = `live_studio --calib` 实时槽读数(逐相机 n/median/p90/近场占比 + 时序;§P-M.10 重测:六路 `near_fraction` **全 0.0**,CAM_BACK 样本 n=11<20 ⇒ 报"无数据");**`rig_layout_{nuscenes,wide}.png`**(配置图)+ **`views_{rig}.png`**(六视角原生像素拼图 + 线性方位尺)+ **`report_{rig}.json`**(ego 像素/实挂偏差/覆盖表/共视) | `probe_calib.py` / `viz_calib_check.py` / `live_studio.py --calib-report` / `viz_rig_check.py` |
| `sem_bev/` `mono_distance/` `stereo/` `multilidar/` `accum_map/` `ground/` `cluster/` | 教程能力线各产物的图/点云/结果 json | 各自 `bin/*.py` |
| `3dgs/` | 3DGS 环绕采集帧 + 真值深度位姿 + `.ply` 高斯 + 训练结果 json | `collect_3dgs.py` / `train_3dgs_mini.py` |
| `hivt_carla/` | HiVT 训练用 TemporalData 与场景划分 | `convert_hivt_pt.py` |
| `carla/` | **运行支撑物**:服务器日志 + LD_PRELOAD shim + Vulkan 兼容层 | `carla_server.sh` |
| `models/` | 推理权重(YOLOPv2 等) | 下载/转换 |
| `nus_mini*` | nuScenes 迷你集(含 L3 雷达口径变体)。**⚠️ 两个旧产物已标废弃**(2026-09-23,§P-M.7):`nus_mini`(已重采覆盖,现 36M)——旧版是「**表对了、图错了**」(相机挂 LiDAR 挂点 + 镜像偏航、雷达偏航差 94–136°、LiDAR 无 rotation)⇒ 不存在"改几行标定就能救"的路径,与 §P-M.5 对 MapTR 权重的处置同口径。**2026-09-23 又重采一次**(§P-M.10 挂点原点:旧版 12 路全偏前 1.2563 m、`ego_pose` 丢悬架俯仰) | `collect_nus.py` |
| `nus_calib_check/` | `collect_nus` 验收判据的复现报告:`report.json`(**修后**逐判据 pass + 原始数字,含 LiDAR 单位阵消融对照 1.0000→0.1684)+ `report_prefix.json`(**修前**基线,三条离线判据全 ✗ 的固化证据)+ **`report_wide.json`**(wide rig;**十条判据**,③④ 为与相机无关的不变项,⑨⑩ = §P-M.10 新增) | `verify_nus_calib.py` |

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

# 文件树:AutoDriveData 仓库结构与文件职责

> **本文档的用处**:仓库的**文件级索引**——"某个文件是干什么的、该改哪、产物落在哪"。
> 定位是**导航**,不是事实源:当前状态与红线看 [CLAUDE.md](../CLAUDE.md),目录结构的**决策依据**看
> [docs/refactor-2026-09.md](refactor-2026-09.md),方案定案看 [Plan.md](../Plan.md),新计划/待办看 [Plan2.md](../Plan2.md),
> 里程碑看 [milestone.md](milestone.md) / [milestone2.md](milestone2.md)。
>
> **用法**:
> 1. **找文件**——先定位能力目录(§2),再看该目录的表;
> 2. **加文件**——新增代码/脚本后**必须回来补一行**(见下方「维护约定」),否则本文件失效;
> 3. **找产物**——只看 §6;产物目录**不展开子文件**(数量大、随采集变动)。
>
> **维护约定**:
> - 只收录**顶层与二级**条目;`outputs/` 下的具体数据集目录不逐一登记,只列类别。
> - 每行格式:`路径 — 一句话职责`;新增/改名/删除时同步本表,与代码改动同一个 commit。
> - 标注 **【未入库】** 的条目被 `.gitignore` 排除(权重/产物/本地配置),克隆后不存在,需自行生成。

## 1 顶层结构

```txt
AutoDriveData/
├── CLAUDE.md / README.md            # AI 协作入口 / 对外简介
├── Plan.md(冻结)· Plan2.md(新计划)
├── pyproject.toml                   # 包定义 + [tool.ruff] + [tool.pytest.ini_options] testpaths
├── requirements.txt / .envrc / .gitignore
│
├── autodrivedata/                   # ★ 主包 —— 按能力面分层,包根只有 __init__.py(见 §2)
├── tools/                           # 开放性工具:不含本项目领域知识(见 §4)
├── docs/                            # 文档(见 §5)
│
└── outputs/ logs/ training/ lightning_logs/ hdMapGitHub/ auto3dlabel/  # 【未入库】产物/运行日志/上游克隆(§6/§7)
```

**依赖方向(硬纪律)**:`utils/` `gt/` `slam/` 不许依赖兄弟能力包;**依赖单向 AutoDriveData → AutoLabel,禁止反向**。
**「谁允许 import 什么」由 [autodrivedata/tests/test_layer_guard.py](../autodrivedata/tests/test_layer_guard.py) 的
`LAYER_RULES` 按目录机械强制**(四档 `_PURE`/`_CARLA_OK`/`_TORCH_OK`/`_ANY`,最长前缀匹配)。

## 2 `autodrivedata/` — 按能力面分层的主包

| 能力目录 | 规模 | 一句话 | 层约束 |
|---|---|---|---|
| `sim/` | 24 | CARLA 仿真交互层 + 全部采集器 + 平台探针 + 闭环路线纯值 | 许 carla |
| `calib/` | 14 | 标定原语 / rig 表 / 自证探针 / 实时监看 / 配置图 | 许 carla,禁 torch |
| `map/` | 35 | 地图矢量 + MapTR(含 `maptr/`、`maptr_official/`) | 不设限 |
| `slam/` | 10 | 两段式激光 SLAM + 精度评估 + C++ 对拍 | **纯值** |
| `perception/` | 17 | 检测 / 单双目 / 雷达 / 语义 / 点云 | **不 import carla** |
| `gt/` | 5 | 动态目标 + 静态目标 + 灯态 + 落盘导出 | **纯值** |
| `traj/` `gs/` | 3 | 轨迹组装转换 / 3DGS 训练 | 许 torch,禁 carla |
| `utils/` | 4 | `geometry` `paths` `fonts` `runlog` | **纯值** |
| `tests/` | 58 | 与能力目录镜像(见 §3) | 不设限 |

**规模列口径(两个,别混)**:§2 表 = **递归**(`find <目录> -name '*.py' ! -name '__init__.py'`);
各小节标题 = 该目录**本级**文件数(子包另立小节,如 `gt/` 记 `3 + export/ 子包`、`map/` 记 18
而 `maptr/` 12 + `maptr_official/` 4 另行)。两者都**不含 `__init__.py`、不含 gitignore 的
Jupyter 残留**(`.ipynb_checkpoints/`,它会污染计数);标的是量级,加/删文件后回来顺手改一行。

**命名约定**:模块与所在目录同名时改叫 **`core.py`**(`calib/core.py`、`slam/core.py`、`gt/core.py`)——
避免 `autodrivedata.calib.calib` 这类自反名。

**运行留痕(2026-09-27)**:每个**训练 / 推理 / 评估**入口跑一次就在 `logs/` 落**三件套**
(stem 相同:`<能力>_<模块>_<时间戳>.log` 全量文本 / `.jsonl` 逐迭代指标 / `.json` 汇总),
口径与关法见下方 [`utils/runlog.py`](#utils--通用件4) 行,落点见 §6。
下表标 **落 `logs/` 三件套** 的**恰好 18 个**:用户裁决的覆盖范围(16 = 训练 3 + 推理/评估 13)外加
**2026-09-27 补入的两个 SLAM 链路入口**(`slam/slam_odometry`、`slam/slam_backend` —— 前端 12.6 min、
后端 11 min 却都**只在末尾落盘**,中断即零痕迹,是这三类里留痕价值最高的一对);
**采集器(`sim/collect_*`)、数据组装(`map/assemble_*` / `merge_train_infos`)、标定探针(`calib/*`)仍不在其中** ——
它们的产物自带逐帧索引,留痕价值低于上述三类。

### `sim/` — CARLA 仿真交互层 + 全部采集器(24)

**唯一大面积 `import carla` 的目录**;采集器、实时流、平台探针都在这。

| 文件 | 职责 |
|---|---|
| `scenarios.py` | Corner case 场景目录 + 可模拟性矩阵(P1) |
| `collect_rig.py` | 采集器纯值位姿/挂点计算(环绕位姿 / 双目挂点;分层守卫见 `tests/test_layer_guard.py`) |
| `route.py` | **闭环路线纯值**:`find_cycle`(路网 BFS 最短有向环,吃 `successors` 回调,可在合成图上单测)/ `make_successors`(CARLA `wp.next()` → 节点键 `(road_id, lane_id, round(s/step))` 的适配层,只吃 duck-typed `WaypointLike` ⇒ **零 carla**)/ `pure_pursuit`(前视点 + 单调进度 + 打舵符号)/ `lap_budget` + `speed_ceiling`(**一圈帧数 ≥ 250 帧 ⇒ 速度上限**,采集前的第一道闸)。零 carla,供 `collect_slam --route loop` |
| `collect_kitti.py` | 静态采集:ego 静止 + 摆 NPC + 同步模式 → KITTI root(raw + GT) |
| `collect_drive.py` | 动态采集:ego autopilot + TM 车流 + 行人 → KITTI 序列 |
| `collect_ab_route.py` | P1 A/B 专用:ego 定速直行 + 路侧静置车(固定位置,帧级配对) |
| `collect_nus.py` | nuScenes 迷你集:6 相机 + LiDAR + 5 雷达。**全传感器标定「渲染位姿 = 声明位姿」同源**(2026-09-23,§P-M.7):相机走 `NUS_CAMERA_RIG`(逐相机 6DoF 挂点)+ 逐通道蓝图 `fov`;LiDAR 走官方挂点 + 由四元数**导出**的姿态(`LIDAR_ROT` 不手抄);雷达偏航 `−az_nus` **由 `NUS_RADAR_OFFSETS` 导出**。历史缺陷(已删 `CAM_YAW_OFFSET` 镜像表 / 雷达猜测表 / LiDAR 无 rotation)见模块头注的对照表。判据复现器 = `autodrivedata/calib/verify_nus_calib.py`。**挂点原点**:位置/姿态全走 `geometry.nus_ego_translation` + `nus_ego_rotation`(后轴,§P-M.10);挂传感器前先让车静置收敛(`ego_settle`,实测 8 tick),`ego_pose` 写全 6DoF |
| `collect_surround.py` | 环视 6 相机采集(nuScenes 布局)→ 图像 + 内外参 + 逐帧 ego 位姿 |
| `collect_surround_micro.py` | 环视微采样(10 帧)→ 相机布局对照微实验 |
| `collect_static_gt.py` | 静态目标/道路特征 GT(地图查询源,含 overlay 目检图) |
| `collect_tl_states.py` | 灯色动态 GT(记录模式 / `--cycle` 受控切灯) |
| `collect_traj.py` | 多 agent 轨迹采集(HiVT 训练数据源) |
| `collect_stereo.py` | 双目 rig 采集(基线 0.4m)+ 真值深度 |
| `collect_3dgs.py` | 静态场景 360° 环绕采集(RGB + 真值深度,支持多俯仰) |
| `collect_slam.py` | SLAM 数据集采集:ego 定速巡游 → `training/velodyne/` + **`training/pose/`(ego 真值位姿,ATE/RPE 评估的 GT)**。**`--route loop`** = 路网找环 + 纯追踪跑 `--laps` 圈(让同一处真被走两次,后端回环链路的端到端数据源);`--dry-run` 只找环报预算(闸门见 `route.py`),`--spawn-index` 换可复现起点 |
| `view_stream.py` | 场景实时流:真 UE 渲染 + GT/预测 overlay → 浏览器 MJPEG |
| `live_common.py` | **实时可视化共享件**(从 view_stream 抽出):多槽 MJPEG 服务(单端口 `/stream/<name>` + `/` 索引页)/ **拼图(两套:`compose_grid` 等尺寸 + **尺寸守卫**,不符即 `ValueError` —— `paste` 源图大于目标框时只贴左上角、静默裁;`compose_rows` 按行拼、每格**原生像素**,studio 三层用它)** / GT overlay / **环视 rig(两代口径 `rig_spec`:`nuscenes` 逐相机 `SENSOR_MOUNTS` + 官方 6DoF 姿态,`legacy` 共用 `SENSOR_OFFSET` + 235/125;`resolve_rig` 按权重名选)** / 第三方视角 / 键盘 —— view_stream 与 live_studio 共用。**`build_surround_rig(..., kind=)`** 可挂 `rgb` 或 `depth`(深度槽必须与 RGB 槽**同挂点同内参同分辨率**,否则 overlay 无法逐像素对齐);`draw_hud(..., y=)` 支持第二行(条带 16 px) |
| `live_studio.py` | **8 路 studio**:6 相机 + BEV + 第三方 + `grid` 拼图槽,各占一路;`--keyboard` 折进 tick 循环(WASD 开采集);第三方非 attach 每 tick 摆位。**拼图 = `GRID_ROWS` 三层**(①左前/前/右前 ②右后/后/左后 ③第三方 + BEV),**每格原生像素不缩放**(相机 1242×375 / 第三方 640×360 / BEV `--bev-size` ⇒ 画布 3726×1170;旧 4×2 等尺寸布局把相机图裁到 621×187 丢掉地面,见 Plan2.md §P-L.6)。**`--slam` 接在线 SLAM**(挂语义 LiDAR → `SlamWorker`,BEV 槽画地图点/轨迹;`--slam-async` / `--slam-voxel` / `--slam-max-gap` / `--slam-report` 落验收 JSON)。**`--calib` 接实时标定监看**(另挂 6 深度相机同挂点同内参 + `sensor.lidar.ray_cast`,LiDAR→世界平面→投影回相机按深度残差着色画进各相机槽,`draw_hud` 第二行报 pooled |e| 与逐路样本数,`--calib-report` 落 JSON;`--calib-refit` 默认 2 tick 重拟合一次,预算见 `calib_live.py`)。**`--video` 落八视角视频段**(拼图槽逐帧写 mp4,cv2/mp4v 惰性开编码器;`--video-fps` 标称帧率 / `--video-tile` 放大倍数) |
| `drive_ego.py` | live 手动驾驶(服务器终端 WASD 遥控);薄封装 `live_common.KeyboardState`(studio 内置键盘是首选) |
| `probe_radar_l3.py` | 雷达物理合理性探针(对照真实 ars408 规格) |
| `probe_vulkan.py` | Vulkan 设备枚举——CARLA 渲染停摆的一线判据 |
| `probe_imu.py` | **B1 实测**:CARLA IMU 能否支撑 FAST-LIO2 的 IESKF 预测(结论:直行段 IMU 预测比恒速先验更差 → B3 不投) |
| `smoke.py` | M0 smoke:CARLA headless 连接 → 同步模式 → 各取一帧落盘 |
| `carla_common.py` | 采集公共件:位姿换算 / NPC 摆放 / 传感器参数 / 同步模式 / 灯态归一与绘制 |
| `smoke_radar_collect.sh` | 雷达采集冒烟 |

### `calib/` — 标定(14)

许 carla、禁 torch。

| 文件 | 职责 |
|---|---|
| `core.py`(原 `calib.py`) | KITTI 标定生成(内参/外参 → calib txt,含 `world_to_img` 投影共用件) |
| `camera_rig.py` | **环视相机 rig 唯一来源**:官方 nuScenes `calibrated_sensor`(6DoF 四元数)→ CARLA 采集口径 `NUS_CAMERA_RIG`(平移 y 翻号 + 姿态走 `nus_camera_rotation_to_carla`)。采集器 / 实时流 / 导出器同源;模块头注记录**镜像 bug**(`yaw_carla = −az_nus` 漏翻 ⇒ 四个侧/后相机左右互换)。平移一律 = 官方表 + `geometry.NUS_EGO_ORIGIN_X`(§P-M.10 后轴对齐);wide 的后三路常量是 **CARLA 口径**的 `NUS_WIDE_REAR_X_CARLA = -1.9000`(落盘 nus 侧 = −0.6437),命名即口径,别混。**头注的四元数模长归因已于 2026-09-23 订正**:非单位的来源是**本表手抄的 4 位小数**(如 `CAM_FRONT_LEFT` `|q|−1 = −5.04e-05`),官方 mini 120 条 `calibrated_sensor` 的 |q| 实测全为 1.000000000000 |
| `selfcheck.py` | **标定自证纯值件**:平面拟合(`fit_plane`/`fit_local_planes`)、相机射线/反投影(`cam_rays`/`project_world`/`backproject_depth`)、深度图采样(`collect_samples`/`DepthSamples`)、轴目标物质心法主点裁决(`estimate_axis_delta`/`estimate_delta_uv`)、镜像不对称度(`mirror_asymmetry`)、径向误差剖面 |
| `depth_codec.py` | CARLA 深度图编解码(`decode_depth`/`encode_depth`,BGRA→米)+ 采样口径(`CONVENTION_CENTER`/`CONVENTION_CORNER` + `sample_bilinear_many`)——**实测裁决 CARLA 光栅 = corner**(索引 i 即连续坐标 i;见 `probe_calib.py` A3/A4),`CENTER` 保留供对照。**坑:`decode` 与 `sample` 的像素索引约定必须一致**,差 0.5 px 在近处 = 米级深度误差 |
| `calib_live.py` | **实时标定槽纯值件**(`live_studio --calib`):`live_planes`(体素+抽样+逐点邻域平面,`LIVE_*` 廉价预算)/ `sample_camera`(复用 `selfcheck.collect_samples`,`CONVENTION_CORNER`,不开窗口极差)/ `residual_colors`+`paint_residuals`(着色,**与离线探针同源**)/ `summarize`+`CameraResidual`(样本 < `MIN_CAM_SAMPLES` 时 `median_abs is None`,**不许报假数字**)/ `self_occluded_cameras`+`near_fraction`(**相对**判据,见下)/ `hud_line`。**坑:自遮挡判据不能写死阈值**——近场占比随**画幅宽高比**变(CAM_BACK 1242×375 是 0.367、640×360 只有 0.195),故按同批可用相机的近场占比中位数定阈;平面是**世界系**故可跨 tick 复用(瓶颈全在拟合 ~100–150 ms vs 六相机采样 ~9 ms ⇒ 默认每 2 tick 重拟合) |
| `rigviz.py` | **自车 + 传感器标定配置图**(纯值,PIL;见 Plan2.md §P-M.9):`draw_rig_layout` 一页三区 = 俯视(车体实测包围盒 + 逐相机挂点与视锥 + 短码)+ 方位环(重叠橙、盲区红带度数)+ 数字表(`通道 · 挂点 x,y,z · 方位角 · FoV · **az ± fov/2**`)。`azimuth_of` 是方位角算式的**第二份独立实现**(单测钉它与 `camera_rig.camera_azimuth_nus` 相等)。**分区是硬坐标**——图例压锥 / 方位环压表都踩过。接受**显式 calibs/fov 参数**,故"还没采过的候选 rig"只改常量就能出图。俯视图另有**后轴标记线**(洋红 = nus 原点)+ **空心灰圈 = 修正前挂点**(整体后移 1.2563 m 落在后轴线上,§P-M.10) |
| `calib_multilidar.py` | 多雷达标定判据评估(注入已知误差 → 判据数值) |
| `viz_layout_cmp.py` | 相机布局对照数值化(同镜头两布局的可见性对比) |
| `probe_rig_mount.py` | **环视挂点口径 A/B 实测**:同一权重喂「它训练时见过的 rig」vs「另一代 rig」→ 品红像素/段数差 = 错配代价(结论见 Plan2.md §P-L.1) |
| `probe_calib.py` | **标定自证探针(七锚 A0–A6)**:spawn 6 RGB + 6 depth + LiDAR + 施工锥 → `outputs/calib_check/{report.json,overlay.png}`。A0 光轴 vs 官方方位角 / A1 侧别一致性 / A2 实例分割解码(`id = G + 256·B`)/ A3 LiDAR-平面-深度图交叉验证(裁决**像素约定 = corner**)/ A4 轴目标物掩膜质心回归主点 / A5 实挂 vs 规格 / A6 主点锁定 `(w−1)/2`。**判据全数值,不目检** |
| `viz_calib_check.py` | **标定修正的人工复核图**(`probe_calib` 的数值结论 → 人能对着看的图,判据数字烧进画面):①`check_geometry.png`(纯值,不依赖 CARLA,先落盘)——官方方位角极坐标轮(实线=修正后 / 淡线=历史字面值)+ 镜像差表(**不 wrap**,217.2° 折成 142.8° 就看不出镜像)+ A4/A3 读数;②`check_raw.png`/`check_overlay.png`(A3 六相机**同一帧**的 raw 与残差 overlay,两张逐像素差 = 画上去的点数);③`check_rig_ab.png`(同一 ego、同一批施工锥,`nuscenes` vs `legacy` 各拍一遍 → **世界左方的锥出现在哪一路**即镜像的直接证据)。落 `outputs/calib_check/check_*.png` + `viz_summary.json` |
| `verify_nus_calib.py` | **`collect_nus` 验收判据的复现器**(全数值,不目检;落 `outputs/nus_calib_check/report.json`)。`--offline`(不需 CARLA):③ 雷达点落进**自身 FOV** 占比(以官方 R 为锚 —— 点云在传感器自身系,+x 即光轴,不转到 ego 系再比就是恒真)、④ LiDAR 复现 `num_lidar_pts`(官方集 1.0000 vs 单位阵消融 0.1684)、⑤ 相机内参 vs 该 rig 声明表;`--live`:① 相机实挂 vs 声明(`live_common.mount_deviation_of`,**必须先 tick**)、② 雷达实挂 vs 官方 az、⑥ 渲染 FOV vs 蓝图 fov(复用 `probe_calib` 的锥体/掩膜回归,**不另写实例分割解码**)、⑦⑧ 转 `rig_check.instance_probe`。**⑥ 两个实测坑**:每 tick 必须**抽干全部相机队列**(否则未测相机积压陈旧帧 ⇒ 掩膜恒 0,症状是"只有第一个相机测得出"且与距离无关)、Z 取**阶梯** `(20,14,10,8,6)`(地图遮挡随出生点变,写死会假失败)。**明令禁止 `num_radar_pts` 作判据**(跨 5 通道求和,官方 R 也只复现 0.6279)。**+ ⑨⑩(§P-M.10)**:⑨ 世界系链(`declared = 落盘表 ⊕ **实测**后轴位姿` vs `rendered = CARLA 实挂经共轭`,12 路逐位比 —— ①② 相对同一个 ego,**对原点误差在结构上盲**,必须有它);⑩ 独立复测后轴(双偏航自解,**不读常量**,防常量腐化静默错位)。`--rig {nuscenes,wide}`:③④ 与相机无关原样适用,其余换该 rig 的声明表;默认落点按 rig 分叉(`nus_mini[_wide]` / `report[_wide].json`,**不互相覆盖**) |
| `rig_check.py` | **判据 ⑦⑧ 的唯一执行点**(`RigCameras` 按任一 rig 的**声明位姿/FoV** spawn 6 路 instance_seg,1600×900):⑦ 逐像素数 **ego 自己的 actor id**(对照:**修正前**官方 rig `CAM_BACK 619189 px = 42.9992%`;§P-M.10 原点修正后**两代 rig 六路全 0 px** ⇒ 该判据不再是 wide 的区分度,wide 的取舍是 FoV spec 换覆盖率);⑧ 往重叠区正中摆施工锥,两路掩膜都命中才算共视。**两条边界写死在模块里**:锥心抬 `COVIS_Z_LIFT=0.45 m`(否则擦车顶被自车挡,读数变成"有没有被自车挡")、几何重叠 < `MIN_COVIS_OVERLAP_DEG=5°` 的对**如实 `skipped`**。★ `common_band()` 是核心订正:**方位轴重叠(无穷远)≠ 有限距离下的共同可见**(挂点视差最多 1.7°,官方 `FL↔BL` 11.20° → 12 m 处 4.838°) |
| `viz_rig_check.py` | **rig 的两件目检交付物**(`--rig`,`--live` 需 CARLA;落 `outputs/calib_check/`):`rig_layout_{rig}.png`(纯值配置图,`rigviz.draw_rig_layout`)、`views_{rig}.png`(六视角**原生像素**拼图 + 逐格 `az ± fov/2` 与 ego 像素读数 + 底部**线性方位尺**:逐相机一条泳道 ⇒ 竖线穿过的行数 = 该方位被几路覆盖;ego 像素**就地染品红**,0 px 时是空操作)、`report_{rig}.json`。**图不能替代 `verify_nus_calib`**:"声明 ≠ 渲染"在图上看不见(§P-M.7)。两个踩坑:`360.0 % 360 == 0` 会让 `rectangle` 抛 `ValueError`(跨 0° 扇区必须钳,不取模)、`np.asarray(PIL)` 是只读视图(染色必须 `np.array` 拷贝再 `Image.fromarray`) |

### `map/` — 地图矢量 + MapTR(19)

| 文件 | 职责 |
|---|---|
| `opendrive.py` | OpenDRIVE 1.4 解析(planView/lanes/objects/signals/junction)。`_geo_at` 有 `strict` 档:默认对 `s ∉ [0,length]` raise(**抓调用方 bug** 的守卫,不是物理约束);`strict=False` 才**有界外推**(上限 10 m) |
| `stitch.py` | **跨图拼接**(§P-M.16):`Placement` 表 + `place()`(刚体,**恒等原样返回**)+ `stitch()`(并集 / 多图加 id 前缀 / **跨图**去重)+ `seams()`(接缝候选**报告**)。⚠️ 官方 Town 无真值相对位姿,placement 是**人为摆位**;⚠️ `MapVec` 不带道路图 ⇒ 接不了 link;⚠️ 合并图**对 MapTR 训练无用**(训练仍逐帧 `BEV_RANGE` 裁剪)。两条实测逼出来的判据:**同图内部不去重**(不同 road 的中心线会恰好重合)、**单点要素(信号灯)必须参与去重** |
| `lanelet2.py` | **Lanelet2(.osm XML)适配器,双向**(§P-M.15):**标准 OSM 结构**(`<node id lat lon ele>` + `<way><nd ref>`;早先把坐标内联进 `<nd>`,真实 reader 会**读成空图**)+ **高程走 `ele`** + `autodrivedata:seq` 还原顺序;六类要素→`way`/`node`(标线级,`divider`→`line_thin`、`boundary`→`line_thick`、`ped_crossing`→`crosswalk`…);坐标系是**米**而 .osm 存 lat/lon ⇒ **等距圆柱投影**,`origin` 默认取质心并**写进文件注释**(外部工具忽略、自己读得回 ⇒ 往返不依赖文件外的隐式约定)。保真走自定义 tag `autodrivedata:cls`/`:id`/`:src` + `attrs:<key>`(**重复键**加 `:N` 后缀);读回**优先自定义 tag、回退语义标签** ⇒ 也能尽力读第三方。**只到标线级**;升级到真 lanelet 需要什么见模块头注 |
| `apollo.py` | **Apollo HD Map(text-format protobuf)适配器,双向**(§P-M.15):`ped_crossing`/`stop_line`/`traffic_light` → `crosswalk`/`stop_line`/`signal`(**天然对应**);⛔ **`divider`/`boundary`/`centerline` 在 Apollo `Map` 里没有落点** ⇒ 降级为借 `lane.central_curve` 承载(**把标线当车道**),类名编进 `id.id`。保真 ≈ 无 KV 槽 ⇒ `attrs`/`id`/`src`/顺序全编进 `id.id`(保留键 `@id`/`@src` 带 `@`,与 quote 后的 attr 键不可能撞名)。走 text-format 是因为**本机无 `.proto`**、免 protoc。升级到真 lane 需要什么见模块头注 |
| `mapvec.py` | 地图矢量 GT 提取与采样(MapTR 口径:六类要素 + 裁剪 + 重采样)。**object 轮廓走非严格外推**(斑马线/停车线摆在路段起止处会合法溢到相邻路段,实测最多 2.44 m)⇒ 修前 Town03/04/05/06 **整张图提不出来**(12/20),现 **20/20**;越界实例在 attrs 里带 `s_extrapolated`,未越界的**不加该键**(⇒ 本来就好的图产出逐位不变) |
| `mapvec_schema.py` | 矢量预测对外契约 `mapvec_pred/1`(schema 校验/JSON 往返)。`map_format` 是**可选加字段**(缺省 `opendrive`)⇒ 旧文件照收,**不 bump 版本号**(§P-M.15) |
| `mapviz.py` | 矢量投影与绘制:ego 系折线 → 相机像素 + BEV 面板。`intrinsics_from_k` **直读 K 的 cx/cy**(不重算);`calib_from_fov` 是**全仓唯一 fov→fx 落点**(主点 = 索引约定中心 `(w−1)/2`) |
| `chamfer_ap.py` | MapTR 评估纯值:Chamfer 距离匹配 + 多阈值 AP(官方口径) |
| `assemble_maptr.py` | 环视采集 + 地图矢量 → MapTRv2 infos 同构 json。`--surround`(单段,旧用法)/ `--segs-dir`(多段,收 `seg*`)**二选一**;多段按帧带 `seg` / `frame_in_seg`,`token` = `{seg}_{i:06d}`(**全局唯一**,`--out-frames` 拿它做文件名)。`roots_and_prefixes()` 决定 `data_path` 前缀:单段**空**(与旧产物逐帧等价,实测只有 3 个键变)、多段 `segK/`(§P-M.12) |
| `merge_train_infos.py` | 拼接多组环视训练数据 → 合并 infos(帧号连续重排) |
| `convert_mapvec.py` | 地图矢量 → MapTRv2 annotation 口径 |
| `export_mapvec.py` | 地图矢量导出全量/帧级裁剪 json + BEV overlay。`--lanelet2`/`--apollo` **追加**写该格式(**互斥**);`--from {opendrive,lanelet2,apollo}` 反向读回当 GT 源。**默认路径逐位不变**(`vecs_dump` 没动,实测 sha256 与改动前相同) |
| `prepare_official_dataset.py` | 环视数据 → 官方栈可直吃的 nuScenes 形状数据集(已终止线) |
| `train_maptr.py` | MapTR 训练入口(单帧过拟合 = 正确性锚点;多帧 = 常规训练)。`--seg` / `--exclude-seg` / `--keep-in-seg` 走 `select_frames`(`--frames 0` = 筛后不截断);`--temporal-window K` = MapTRv2 时序版(与 `eval_maptr.py` **必须同值**),训练集丢弃数经 `ds.dropped` 上报;**长训一律 `--lr-halve 0`** —— 默认 12 会让 128 ep 后半程 lr 归零,平台是 lr 死掉不是收敛;过拟合闸门由 `is_single_frame_anchor(n_samples)` 判定,**按实际训练样本数而不是 `--frames` 标志**(`--frames 0` 是"不截断",按标志判会把 400 帧训练误判成单帧锚点并打假 FAIL + 退出码 1,见 §P-M.12) · 落 `logs/` 三件套 |
| `eval_maptr.py` | MapTR 评估:权重 → 逐帧推理 → 四类 chamfer AP(+ 逐帧契约落盘)。选择器与 `train_maptr` **同一份实现**(`select_frames`),`--start` / `--frames` 在**过滤后**的列表上再截(§P-M.12) · 落 `logs/` 三件套 |
| `eval_official_metric.py` | A′ 口径复算:并排算"自实现 chamfer AP"与"官方 eval_map" · 落 `logs/` 三件套 |
| `viz_maptr_pred.py` | 预测回投目检:预测/GT 折线 → 6 相机 overlay + BEV 面板 · 落 `logs/` 三件套 |
| `probe_mapvec_oracle.py` | 离线 xodr 解析 vs CARLA 运行时几何对账(0.00cm 验收) · 落 `logs/` 三件套 |
| `probe_mapvec_proj.py` | 矢量投影回 6 视角图像的路面性验收(数值诊断) |
| `assemble_and_merge.sh` | 组装 + 合并 infos 的批量编排 |
| `finalize_maptr_600.sh` | 600 帧扩数据轮的收尾编排 |

### `map/maptr/` — MapTR 参考自实现(12)

原顶层 `maptr_impl/`。torch 栈,禁 carla。**MapQR 两个变体**(2026-09-27,§P-M.14)默认全关,
关时与本文其余描述逐位一致;开关见 `model.py` 行。

| 文件 | 职责 |
|---|---|
| `model.py` | 模型组装:ResNet50+FPN backbone + GKT BEV 变换 + 分层 query head。**两个 MapQR 变体开关(默认全关)**:`scatter_gather=True`(head 换 `head.py` 的解码器)、`bev_encoder="height_kernel"`(GKT 与 head 之间插 `bevenc.py`,与 `temporal_window>1` **互斥**——官方时序在 BEV encoder 内部,叠 `TemporalFusion` 会让归因无从下手)。两者输出契约不变,但**开了就必须从头训练**(参数名集合变 ⇒ `load_map_weights` 拦下并点明原因)。`temporal_window=K>1` 时在 GKT 与 head 之间插 `TemporalFusion`(**head 一行不改**);`forward` 按 `images` 是 dict / list 分派单帧/时序,历史帧在 `torch.no_grad()` 下编码(memory bank 语义)。另有 `load_map_weights()` = 权重口径的**唯一落点**(单帧权重 → 时序模型 = 缺失 `fusion.*`、放行热启动;时序权重 → 单帧模型 = 多余键、报错) |
| `deform_attn.py` | **可变形注意力原语**(§P-M.14):`multi_scale_deform_attn`(签名对齐官方 `multi_scale_deformable_attn_pytorch`,纯 torch)+ `normalize_ref` + `sine_pos_embed` + `build_sampling_locations`。MapQR 的**两支共用**一份实现(解码器可学习偏移 / BEV 编码器固定 kernel 网格),避免漂移。**归一化的唯一落点**:官方整套数学吃 `[0,1]`,本项目坐标是米 —— 米直接进 `sine_pos_embed` **不报错**,只让正弦频率别名、位置嵌入退化成噪声(症状是「训不动」),故有回归钉。`chunk` 参数保留但**实测不降显存**(见模块头注),别当省显存手段 |
| `bevenc.py` | **BEV 编码器**(§P-M.14,MapQR 的 `HeightKernelAttention` + `with_height_refine`):`BEVEncoder` = N 层(可变形自注意力 → 图像交叉注意力 → FFN),插在 GKT 之后做**细化**。`height_offsets` 初值常零 ⇒ 起点是 `linspace(zmin,zmax,D)` 均匀柱。**三处易错都写了断言/钉**:①官方靠广播把「D 个高度锚点」与「注意力头」对齐 ⇒ `D == n_heads` 必须成立(否则静默拼出 7 维张量,报错点离病根很远);②`build_lidar2img` 必须把 **ego→world** 折进 4×4,且行/列向两次转置一次都不能错(错法的盲点是**恒等 ego 位姿**,故 oracle 用真实位姿对拍);③`scale_k` 的 `(w,h)` 与 `feat_shape` 的 `(h,w)` **不同序**,传反不报错只投错位。**与官方一处刻意分歧**:官方用 `mask_per_img[0]` ⇒ bs>1 时所有样本套用第 0 个的可见集合,本实现按 per-batch 算;`bev` 来自 GKT 而非官方可学习嵌入(有意分歧,已记 Plan2) |
| `temporal.py` | **MapTRv2 时序版(§P-M.12 阶段 4)**:`warp_bev`(历史 BEV 按两帧相对位姿扭到当前 ego 系,整链 float64 + 完整 3D,只在最后取 (x,y)) + `TemporalFusion`(拼接 → 1×1 conv → 残差加回,`proj` **零初始化** ⇒ 第 0 步与单帧模型逐位相同)+ `ego_rotations`(位姿六元组 → ego→world 旋转阵,与 `gkt.cam_world_pose` 同源同序)。两处口径:**变换顺序 `R_prevᵀ·(R_cur·p + t_cur − t_prev)`**(漏转置的盲点是 `ψ_prev=0` 而非 `Δψ=0`,姿态全零时测试会假绿)、**grid 末维必须是 (gx, gy)**(BEV 张量 H 索引吃 gy)。两种错法的误差律与实测格位移写在模块头注 |
| `gkt.py` | GKT(Geometry-aware Kernel Transform):环视相机特征 → BEV 特征。**两处已修 bug(2026-09-22)**:①infos 六元组 `[x,y,z,yaw,pitch,roll]` 必须换序成 `(pitch,yaw,roll)`(`_ROT_TO_CARLA`),漏了 = 5/6 相机指向错;②K 必须缩到**特征图**分辨率(`scale_k`,FPN P2 = 311×94),漏了 = 全分辨率像素与 `feat_w−1` 比。两坑叠加把 BEV 有效覆盖从 94.6% 打到 1.25% |
| `variants.py` | **变体表的唯一落点**(§P-M.14):`VARIANTS` 表把 `maptrv2` / `mapqr` 展开成两个细粒度开关;`resolve_variant` 管**聚合开关与细粒度开关互斥**;`save/load_map_model` 让 checkpoint **自带结构说明**(`{"state_dict":…, "model_kwargs":…}`,旧裸 state_dict 仍可读)。**eval / viz / studio 都经它建模型** —— 变体逻辑只此一份,三个消费方各一行,chamfer AP 不会出现第二份实现 |
| `head.py` | **分层 query 外壳**(装配层,不实现解码器):每类 `num_vec` 个实例查询 + 可学习初始锚线 `anchor` + 实例级 `cls_branch`;按 `scatter_gather` 开关**选解码器类**;`match_assign`(按类匈牙利 + 双向 GT 增强,MapTRv2 §3.2)+ `maptr_loss`(focal 分类 + 点级 L1,权重 5) |
| `decoder.py` | **默认线解码器**(MapTRv2 口径):实例自注意力 → 点级 BEV 双线性采样(官方 deformable 的 topk=1 简化)→ 点回归 → **均值**回聚实例。构件 `MLP` / `FFN` 也在此(默认线为基)。`_sample_bev` **同时是 `test_deform_attn` 的 oracle** |
| `decoder_mapqr.py` | **MapQR 解码器**(§P-M.14,ECCV 2024 移植):三处替换 —— ① 点查询 = 实例查询 + **参考点正弦位置嵌入**;② **可学习采样偏移**的 deformable 注意力;③ P 个点特征 **`flatten` 拼接**后 MLP 压回实例(替代 `mean(dim=2)`,论文的着眼点就是**均值会抹平点间内容差异**)。**必须显式归一化**(米直接进 `sine_pos_embed` 不报错、只「训不动」);**有意分歧**:回归保持**米**空间,不搬官方 `inverse_sigmoid/sigmoid` |
| `dataset.py` | B2 infos json → 训练数据集(图像加载 + 位姿/标定透传 + GT 解析)。另有 **`select_frames()` = 留出划分的唯一落点**(train/eval 共用;`--seg` / `--exclude-seg` / `--keep-in-seg` 三选择器取交集,段名拼错**报错而非静默给 0 帧**,旧单段 infos 无 `seg` 键时不带选择器照旧可用)+ `parse_segs` / `parse_frame_range`(§P-M.12)。**`window=K` 时序窗口**:`history_windows()` 的守卫边界是**切分**不是段(帧级切分画在段内部 ⇒ 必须让窗口只从**本切分自己的帧列表**里取同段前驱,否则留出帧 80/81 的历史 79/78 在训练集里 = 泄漏);拿不到完整历史的帧**丢弃并计数上报**(`ds.dropped`),不静默截短;`window=1` 结构与单帧基线逐字节相同 |
| `chamfer_gpu.py` | Chamfer 代价矩阵 GPU 实现(与 `chamfer_ap` 同口径) |
| `device.py` | GPU 显存自适应实测工具 |

### `map/maptr_official/` — 官方栈项目侧胶水(4)

**已终止线**,勿主动重提(Plan.md §5.12)。

| 文件 | 职责 |
|---|---|

### `slam/` — SLAM(10)

> ⚠️ **按名字搜 `FAST-LIO2` / `SC-PGO` 会搜空 —— 它们是「对标名词」,不是文件名。**
> 本目录是[教程 14](milestone2.md) 那条线的**纯 numpy 降档**:对标 FAST-LIO2(ikd-Tree
> scan-to-map + IESKF 紧耦合)与 SC-PGO(ROS 节点 + g2o 位姿图),但**没有 ROS、
> 没有 C++ 后端、没有 IMU 融合**。降档的两条实证依据:
> ① `probe_scan_to_map.py` —— oracle GT 地图下 scan-to-map 反而**单调变差**(1.05×→2.75×),
> ⇒ 不建 ikd-Tree 前端,走帧间点面 ICP;
> ② `probe_imu.py` —— 直行段 IMU 预测**比恒速先验更差**,⇒ 不投 IESKF。
> 所以「有没有按 FAST-LIO2/SC-PGO 实现」的答案是:**能力面对齐,实现是降档版**;
> 回环链路(ScanContext + PGO)在 `slam_backend.py`;历史上因**数据无重访**而一次未触发,
> **2026-09-27 已在闭环序列上端到端跑通**(802 帧 / 114 条回环边 / `ate_aligned` 1.4127→**0.1352 m**;
> 见 Plan2.md §P-H.3 与 `sim/collect_slam.py --route loop`)。

`core.py` = 原 `slam.py`;纯值(禁 carla 与 torch)。`slam_cpp.cpp` 是 C++17 对拍实现。

| 文件 | 职责 |
|---|---|
| `core.py`(原 `slam.py`) | 激光 SLAM 纯值两段式(帧间点面 ICP 前端 + ScanContext 回环/PGO 后端);`icp_odometry` 双出口 = 位姿 `T` / 点映射 `T_delta` |
| `slam_eval.py` | 轨迹精度评估纯值:Umeyama 对齐 / ATE / RPE(evo·KITTI 口径);**纯直行序列的绕轴旋转不可辨识**见模块 docstring |
| `live_slam.py` | **在线 SLAM 会话**(纯值):`LiveSlam.push` 逐帧增量重建(链式约定逐字复用 `slam_odometry`)+ `map_in_ego_frame`/`traj_in_ego_frame` 换到当前 ego 系;**`SlamWorker` = 有界丢旧队列 + 帧间隙止损(`max_gap`,防"丢帧→间隙更大→ICP 更慢"正反馈),默认同步执行(worker 线程被 GIL 压到 eff 0.04–0.24)** —— 滞后有界的判据靠它单测 |
| `accum.py` | 累积语义点云建图:多帧 velodyne 全局累积 + 语义着色 |
| `slam_odometry.py` | SLAM 前端:逐帧 velodyne → 链式位姿 `T_k = P_{k-1}·inv(T_delta)`(双出口契约见 `slam.py`) · 落 `logs/` 三件套(**逐帧** `metric()` 进 `.jsonl` —— 产物只在末尾落盘,中断时它是唯一的进度证据) |
| `slam_backend.py` | SLAM 后端:关键帧 + ScanContext 回环候选 + **几何先验闸 `LOOP_PRIOR_MAX_M` + 抽稀云廉价筛 `LOOP_SCREEN_*`**(都只用来**拒**)+ 双 yaw ICP 验证(反极支在首支过门时**短路**)+ PGO(边存点映射 `Z_ij`)。成本与三条实测依据见 Plan2 §P-H.3.3 · 落 `logs/` 三件套(**每 25 关键帧**一条累计计数 `metric()`:先验拒/筛出局/短路/全量 ICP —— 小时级跑法靠它判"卡在哪一档") |
| `slam_diff_test.py` | 前端位对齐对拍:numpy vs `slam_cpp` 同一 `(prev,cur,init,seed)` 下比单次 ICP |
| `eval_slam.py` | SLAM 精度评估:LiDAR 系位姿 → ego 系(手性共轭 `M·T·M` + 杆臂 `inv(L)`)→ ATE/RPE · 落 `logs/` 三件套 |
| `build_accum_map.py` | 累积语义建图(多帧 velodyne → 全局语义地图) |
| `probe_scan_to_map.py` | **B2 实测**:oracle GT 局部地图下 scan-to-map vs scan-to-scan(结论:误差随地图深度 K 单调变差 1.05×→2.75×,重新体素化救不回 → 不建 ikd-Tree 前端;含代价/谱/地面占比三条机制证据) |

### `perception/` — 感知(17)

**不 import carla**(许 torch/ultralytics)。

| 文件 | 职责 |
|---|---|
| `semantic.py` | CARLA 语义 LiDAR 标签 → KITTI 式强度合成(域差距修复) |
| `compare.py` | 伪标签 vs GT 比对层:3D IoU 匹配 → 分歧帧 → AP/复核率报表 |
| `attribution.py` | 失效归因:逐帧 2D 匹配 → 漏检按距离/框高/TTC 分箱 |
| `radar.py` | CARLA radar 原始检测 → nuScenes 18 字段雷达点云 |
| `mono_depth.py` | 单目测距:检测框 → 地平面投影距离 + 迭代深度法 |
| `stereo.py` | 双目立体视觉:三角测量 / SGBM 视差 / NCC 匹配 / 自监督损失 |
| `multilidar.py` | 多雷达标定:point-to-plane ICP + overlap/plausible 判据 |
| `ground.py` | 点云地面提取:RANSAC 平面拟合 + 网格法双路线 |
| `cluster.py` | 点云聚类障碍物检测:欧氏聚类 + 3D 包围盒 |
| `finetune_synth.py` | 合成 KITTI → pointpillars_kitti 微调(复用 AutoLabel train3d) · 落 `logs/` 三件套 |
| `eval_2d_ab.py` | P1 逆光 A/B:冻结 YOLO11s 在两个 KITTI root 的 2D AP 对比 · 落 `logs/` 三件套 |
| `eval_attr.py` | 失效归因评估:多跑 × 距离/框高/TTC 网格 + 漏检画像 · 落 `logs/` 三件套 |
| `eval_kitti.py` | GT vs AutoLabel 伪标签比对报表(比对层 CLI) · 落 `logs/` 三件套 |
| `mono_distance.py` | 单目测距评估(检测框 → 距离,与 KITTI GT 真距对照) · 落 `logs/` 三件套 |
| `extract_ground.py` | 地面提取(逐帧点云 → 地面/非地面分离 + 统计) |
| `cluster_obstacles.py` | 聚类障碍物检测(地面分割 → 聚类 → 3D bbox) · 落 `logs/` 三件套 |
| `sem_bev.py` | 语义 BEV:图像 → YOLOPv2 + YOLO11s-seg → BEV 鸟瞰 · 落 `logs/` 三件套 |

### `gt/` — GT 生成(3 + `export/` 子包)

`core.py` = 原 `gt.py`;纯值。`export/` 从包根移入。

| 文件 | 职责 |
|---|---|
| `core.py`(原 `gt.py`) | CARLA actor → KITTI label_2 GT 行 |
| `static_gt.py` | 静态目标/道路特征 GT(地图查询源:P2) |
| `traffic_light.py` | 交通信号灯状态 GT(动态时序层:状态归一/前向判据/相位查表) |

### `traj/` — 轨迹(2)

| 文件 | 职责 |
|---|---|
| `assemble_traj_pt.py` | CARLA 轨迹 → HiVT TemporalData 组装(纯值) · 落 `logs/` 三件套 |
| `convert_hivt_pt.py` | plain dict → HiVT TemporalData(在 hivt env 跑) · 落 `logs/` 三件套 |

### `gs/` — 3DGS(1)

| 文件 | 职责 |
|---|---|
| `train_3dgs_mini.py` | 3DGS mini 训练(gsplat 光栅化) · 落 `logs/` 三件套 |

### `utils/` — 通用件(4)

**准入判据**:无项目领域语义、无 carla/torch 依赖;超过 6 个文件即视为 junk drawer。

| 文件 | 职责 |
|---|---|
| `geometry.py` | 坐标转换唯一落点:CARLA 系 ↔ KITTI 相机系 ↔ nuScenes 系。`CARLA_TO_NUS`(对合)/ `nus_camera_rotation_to_carla`(相机自身系另需 `CARLA_TO_CAM`)/ **`nus_sensor_rotation_to_carla`**(LiDAR/雷达自身系与 nus 同轴序 ⇒ 两侧同阵,对纯 yaw 即 `yaw_carla = −az_nus`,6DoF 连 pitch/roll 一起正确翻过去)。`quat_normalize` 的头注归因订正:非单位来自**手抄 4 位小数**。**挂点原点单点真值**(§P-M.10):`NUS_EGO_ORIGIN_X = -1.2563`(CARLA actor 原点在**车身中点**、nus 官方表在**后轴中心** ⇒ 整套 12 路传感器偏前 1.2563 m)+ `nus_ego_translation` / `carla_actor_origin_to_nus_ego`(**只收 6DoF 三元组**,yaw-only 入口会静默丢掉 0.0642° 悬架俯仰)+ `nus_ego_rotation`(全 6DoF `ego_pose.rotation`;**UE 左手口径 ⇒ 正确分解是 `Rz(−yaw)·Ry(−pitch)·Rx(+roll)`,原样代入会静默反号**)+ `CARLA_CAM_TO_NUS_CAM`(相机**局部**基重排 `[e1,−e2,e0]`,正交但 **det = −1**;与全局基翻转相乘才抵消。拿 `M·R·M` 比相机会得到**恒 120° 的假误差**) |
| `fonts.py` | **覆盖层文本的唯一字体落点**(PIL 唯一依赖,不 import carla):`font_path()` 按「`AUTODRIVEDATA_FONT` → 系统 CJK → **CARLA 随包 `DroidSansFallback.ttf`** → DejaVu 兜底(并 warn)」解析;**能画中文吗 = 渲染探针**(`U+10FFFF` 的像素签名 = 该字体的 `.notdef` 签名,某字签名与它相同即豆腐块;`has_cjk` 要求 `文相机字` 四签名互不相同)——**不看文件名、不看 `fc-list`、不依赖 fontTools**。`sanitize` 把字体缺的码位(`REPLACE` 表,实测只 4 个)换成等价 ASCII、兜底 `?`,**绝不留豆腐块**;`get_font`/`draw_text`/`width`/`bbox`/`wrap`(按**实测像素宽**折行 —— 单行画超画布会被 PIL 静默裁掉,见 §P-M.9)是全部绘制的入口。**两条独立成因都在这解决**:① 本机字体族**一个 CJK 字形都没有**而代码硬写 DejaVu;② **PIL 没有字体回退链**,`ImageDraw.text()` 不传 `font=` 就用内置位图字体(同样无 CJK 且只有 ~11 px)。见 Plan2.md §P-M.8 |
| `paths.py` | 项目路径锚定(`project_path()`:相对路径 = 相对项目根) |
| `runlog.py` | **每次训练/推理的运行留痕唯一落点**(只用 stdlib,不 import carla/torch)。`run(script)` 上下文管理器 → `logs/<能力>_<模块>_<YYYYmmdd-HHMMSS>.{log,jsonl,json}` **三件同 stem**:`.log` = **tee `sys.stdout`/`sys.stderr` 的全量文本**(头块 `script/started/argv/cwd/git(dirty 计数)/python+env/gpu(型号+显存+驱动+CUDA)/host`;尾块 产物表 + highlights;异常写完整 traceback 且**照常抛出**,`SystemExit.code != 0` 记 `status=\"failed\"`)+ `.jsonl` = `metric(step, **kv)` **逐行 flush** 的机读指标 + `.json` = 汇总(env 指纹 / `inputs` / `artifacts`(**≤512 MiB 全量 sha256**,超限只记 `bytes` 并 note)/ highlights / notes / `exit_code`)。另有 `logs/latest/<能力>_<模块>.<ext>` **相对软链**指向最新一次。**三条已踩的坑**:① `_Tee` 必须 `__getattr__` **全量代理**(`isatty`/`fileno`/`encoding` —— tqdm/ultralytics 会直接问,只实现 `write`/`flush` 在非 tty 跑法里炸);② 退出时恢复**构造时抓的那个流对象**而非 `sys.__stdout__`(pytest `capsys` 会替换 `sys.stdout`,写错就把外层捕获**永久**破坏);③ 环境指纹**只在头块采一次**并复用(`_summary()` 重采会让 `.json["gpu"]` 与同一跑的 `.log` 头块不一致)。`env` 名按 **`envs/<name>` 路径段**反解 —— `sys.prefix == sys.base_prefix` 在本机**是错的判据**(env 真身在数据盘、软链进 `envs/`)。关掉:`--no-runlog`(扫 `sys.argv` **字面量**,因为 `start()` 早于 argparse)或 `AUTODRIVEDATA_RUNLOG=0`。**注意 `convert_hivt_pt.py` 以文件路径在 hivt env 跑,脚本自带项目根 `sys.path` 引导**(否则 `import autodrivedata` 必炸) |

## 3 `tests/` — 单测与 oracle 对比(autodrivedata env)

> 与 §2 的能力目录**镜像**;`test_layer_guard.py` 与 `test_docs.py` 在 `tests/` 根(包级守卫)。
> 用例基线数**只在 CLAUDE.md 常用命令里写一处** —— 复述在多处必烂(这三个文档都写过 913,早已过期)。

| 类别 | 文件 | 说明 |
|---|---|---|
| 纯值库单测 | `test_geometry.py` `test_calib.py` `test_gt.py` `test_compare.py` `test_paths.py` `test_scenarios.py` `test_static_gt.py` `test_traffic_light.py` `test_semantic.py` `test_radar.py` | 手算断言,不依赖 carla / AutoLabel |
| 运行日志 | `test_runlog.py` | 三件套契约的**纯值**回归钉(不依赖 carla/torch/GPU)。`TestHeader` 钉头块 `script`/`started` 正则/`argv`/`cwd`,**且指纹只采一次**(monkeypatch 计数器 —— 重采会让同一跑的 `.json["gpu"]` 与 `.log` 头块不一致);`TestTee` 钉 `print()` 进 `.log` + **`__getattr__` 全量代理**(`isatty`/`encoding` 与内层流一致)+ **恢复的是构造时那个对象**(`sys.stdout` 身份相等,自证 `capsys` 不被破坏);`TestRegistries` 钉**同名重复登记去重**(同一文件登记两次只留一行 —— `inputs` 是"读了哪些"的**集合**,不是调用流水账)+ 不存在的路径记 `missing` **不虚报** + `artifact_dir` 的 `n_files`/`bytes_total`;`TestFailure` 钉 `ValueError` → traceback 进 `.log` / `status="error"` / **异常照常抛出**,`SystemExit(1)` → `exit_code=1` / `status="failed"`;`TestSwitches` 钉 `AUTODRIVEDATA_RUNLOG=0` 与 `sys.argv` 里的 `--no-runlog` 都**一个文件不建**;`TestFingerprint` 钉无 CUDA / 非 git 下取 `null` 而**不抛** |
| **层守卫** | `test_layer_guard.py` | **包纪律的可执行版本**(docs/refactor-2026-09.md §3):`LAYER_RULES` = 目录 → 禁止 import 的三方名,最长前缀匹配。旧版(`test_paths.py` 的整包禁令)的两个洞已堵:**马甲库**(`ultralytics`/`mmdet3d`/`mmcv`/`lightning` 会拉起 torch 但字面无 torch)、**字面量动态导入**(`importlib.import_module("x")`)。`TestPackageLayers` 扫真实包 + 强制新子目录必须显式声明;`TestLayerGuardSelfCheck` 用**合成源码注入**做立论自证(12 条:抓得住三类违规,且规则能区分、不是"见 carla 就红") |
| **文档守卫** | `test_docs.py` | **「文档/入口不腐」的可执行判据**(2026-09-26 重构后补 —— 那次 21 条引用失效**全是静默的**,只有照着做的人拿到 `FileNotFoundError`;失效模式与 `paths.py::parents[1]` 同族:写的时候对、挪了之后静默错)。三类:① `CLAUDE.md`/`README.md`/`docs/fileTree.md` 的 markdown 链接目标必须存在;② 同三份文档里的 `python -m autodrivedata.<...>` 必须 `find_spec` 可解析;③ **包内 `.py` docstring 的 markdown 链接**必须存在。**第三类是第一版的漏网** —— 只扫 `.md` 时,`utils/fonts.py` 等处的 **16 条**坏链在「全文档坏链 0」的结论下整体逃检 ⇒ **判据的覆盖范围本身也是判据的一部分**。每条各带一条**扫描器自证**(防正则腐化后空过)。只判机械可判者:docstring 里的裸文件名不算路径;形如**方括号后紧跟圆括号单位**的**单位注记**(如米/像素、度、yaw=0)由后缀白名单滤掉(不加会多 9 条假阳性)。**`.`md` 侧不加白名单** —— 那里 `]` + `(` 就是 Markdown 链接,写它就是真坏链(本行的初版用实例演示,当场把守卫测红) |
| 地图矢量线 | `test_opendrive.py` `test_mapvec.py` `test_mapvec_schema.py` `test_mapviz.py` `test_chamfer_ap.py` `test_chamfer_gpu.py` | 含闭式解手算锚点与真实 xodr 计数锚点 |
| 教程能力线 | `test_mono_depth.py` `test_stereo.py` `test_multilidar.py` `test_slam.py` `test_accum.py` `test_ground.py` `test_cluster.py` `test_collect_rig.py` | 各含手算锚点;`collect_rig` 兼作采集器回归先例 |
| SLAM 精度/在线线 | `test_slam_eval.py` `test_live_slam.py` | `slam_eval`:ATE/RPE 手算锚点 + 杆臂方向(不补杆臂 ATE 2.44×);`live_slam`:与离线 `slam_odometry` **逐帧同输入同输出**(<1e-12)+ `SlamWorker` 滞后有界/止损/同步模式 |
| 闭环路线(§P-H.3) | `tests/sim/test_route.py` | **回环数据源的可离线验证部分**(纯值,零 carla)。三条手算钉:① 方框图 → 4 节点最短环、`A→B→A` 掉头被 `min_len` 拒、`max_nodes` 兜住无限链;② ★ **进度单调**(前视点索引不许回退 —— 纯追踪最经典的失败是车在起点附近来回蹭,而它"看着像在开")+ ★ **打舵符号**(目标在右 ⇒ `steer` 为正;符号反了车朝反方向冲出去)+ 双踏板恒互斥;③ **路网键适配**(`TestMakeSuccessors`,**合成 Waypoint**):4 段×30 m ⇒ 40 节点环、单圈几何 **120.0 m**(=39×3 + 收尾那 3 m,同时钉住"键里必须有 `s`"与"闭环长度要算收尾段")、闭合容差必须按**采样步长**给、回调吐出的键当次就进缓存。外加 `lap_budget`/`speed_ceiling`:**一圈 ≥250 帧 ⇒ 速度上限**(200 m 环 8.0 m/s、120 m 环只有 4.8 —— 开快了两次到访帧差不够,**回环必然不触发**) |
| 实时可视化 | `test_live_common.py` | `compose_grid` **尺寸守卫**(不符必抛,防 `paste` 静默裁)+ `compose_rows` 每格**原生像素**逐像素等于源图 + studio `GRID_ROWS` 三层行序 + **`rig_spec`/`resolve_rig`**(两代 rig 口径与"按权重选":legacy 共用挂点 + pitch/roll=0,nuscenes 逐相机 6DoF;显式指定不被文件名覆盖)+ **`mount_deviation_of` 规格对账**(相机与雷达共用;矩阵顺序写反 ⇒ 平移爆掉而偏航仍 ~0、偏差随 ego 离原点变远而变大、legacy 实挂对 nuscenes 规格必报 110°、tick 前全 0 陈旧位姿**不许**判成"通过";`rig_mount_deviation` 是它的薄封装)+ `draw_hud(y=)` 第二行(第一行逐像素不变、`y=0` 与旧行为一致)。测试搬进包后**直连 `autodrivedata.sim.live_common`**,旧的 `sys.path` hack 与 `pyright: ignore` 已摘除 |
| 配置图 / 覆盖表 | `test_rigviz.py` | **交付物图的数值侧回归**(纯值,PIL + numpy)。`TestCoverageTable`:wide 三个盲区**逐项等于设计预算**(7.3353/6.0984/1.7224 = 15.1561°、覆盖 0.9579)、官方 rig 零盲区;重叠对的 `span` 必须落在两个相机各自的扇区里(共视探针按它摆锥)。`TestAzimuthIndependentImplementation`:`rigviz.azimuth_of` == `camera_rig.camera_azimuth_nus`(两套独立实现)。`TestRigLayoutFigure`:**盲区红弧"有当且有、无当且无"**(官方 0 个盲区 ⇒ 0 红像素;wide 有 ⇒ >0)+ 六通道都有画色与短码。`TestRulerLanes`:底尺**跨 0° 不崩**(`360.0 % 360 == 0` 会让 `rectangle` 抛 `ValueError`)+ 盲区红**列数** ∝ Σ盲区度数 + 六条泳道都画出来 + 页脚折行后每行实测宽 ≤ 画布且不压数字表。**不钉排版/错别字**——那些写成断言只会得到"改个字就红"的脆测试 |
| 绘制字体 | `test_fonts.py` | **中文字形不许静默变豆腐块**的回归钉。`TestProbe`:探针立论自证(`U+10FFFF` 在任何字体下都落 `.notdef`)+ **DejaVu 被正确判否**(它有 `−`/`°`/`★` 却画不了中文 ⇒ 判据不是"文件在不在")+ 生效字体实测能画中文。`TestRendering`:两个不同汉字在**画布上必须像素不同**(最强钉——豆腐块下它们逐像素相同)、`sanitize` 后零缺字 / 替换目标自己画得出 / 不等长(排版不错位)、CJK 宽 ≈ 2× ASCII(HUD 底条据此定宽)。`TestDrawnStringsAreRenderable`:**AST 扫全仓绘制字符串**(8 个绘制模块 × 绘制调用实参 + `hud_line` 之类构造器**函数体**——漏后者 `calib_live` 整行中文 HUD 会逃检)⇒ 逐个 `sanitize` 后零缺字;另有**根因钉** `test_no_module_draws_with_a_bare_text_call`(不许出现不带 `font=` 的 `d.text(...)`)与每模块 `import fonts` 钉 |
| **验收补钉(2026-09-28)** | `tests/perception/test_sem_bev.py` `tests/calib/test_viz_layout_cmp.py` | 两条都补的是**同一类失效:CLI 早就跑不起来,而 pytest 全绿**(见 [docs/acceptance-2026-09-28.md](acceptance-2026-09-28.md) §4.1/§4.2)。`test_sem_bev`:桩模型跑通 YOLOPv2 掩膜链(**该模块此前零覆盖**),顺带把一直没人测的 letterbox→裁 padding→缩回原图几何钉住(含反例对照:把带挪位置,输出必须跟着挪);并 AST 钉住"外部 `utils` 包不许回来"(注意与 `traj/convert_hivt_pt.py` 的 HiVT `utils` 是**同名多义**,别混)。`test_viz_layout_cmp`:`--a`/`--b` 必须真的决定**读图**路径 —— 两个 root 的图染成**纯红/纯蓝**,断言两张输出各自取自自己的 root(路径再写死必然同色);**改代码前先确认它对旧逻辑报红** |
| MapTR 数据划分 | `test_maptr_select.py` | **留出划分与多段组装的回归钉**(§P-M.12)。两条被测契约都是**静默失效型**:划分有交集只会让 AP 看起来更高、`data_path` 前缀写错只会让旧命令指错文件 —— 都不报错。`TestSelectFrames`:路线级(`--exclude-seg seg4`)与帧级(`--keep-in-seg 0:2`)两侧**互斥且并集为全集**、边界左闭右开、选择器取交集、**段名拼错必须 `ValueError` 而不是空列表**、旧单段 infos 不带选择器照旧可用。`TestArgParsing`:`parse_segs` 空串/纯空白 → `None`;`parse_frame_range` 拒绝 `80` / `80:100:2` / `100:80` / `5:5`。`TestRootsAndPrefixes`:单段前缀空 / 多段 `segK/`、glob 只收目录、必须且只能给一个来源。`TestHistoryWindows` / `TestWindowDataset`(时序窗口,§P-M.12 阶段 4):窗口**不跨段**、**不跨切分**(训练/留出各自只见自己的帧,窗口帧 ⊆ 本切分池)、段首帧丢弃且计数上报、`frame` 在段缝连续故**不许**当历史键、`window=1` 返回结构与单帧基线逐字节相同。`TestOverfitGate`(假警报型,§P-M.12):过拟合闸门按**实际训练样本数**判(`is_single_frame_anchor`),`320/400/500` 全不是锚点;外加**静态根因钉** —— AST 扫源码禁止 `args.frames` 与整数字面量比较(只禁这一形态,`args.frames > len(sel)` 的截断检查合法),因为"再写回 `args.frames > 1`"就是本条缺陷的复发式 |
| MapTR 自实现 | `test_gkt.py` `test_head.py` `test_device.py` | 单帧过拟合正确性锚定。`test_gkt` 三条**回归钉**:`test_pose_rotation_order_is_carla_convention`(换序)、`test_scale_k_to_feature_resolution`(K 缩放)、`test_gkt_valid_coverage_on_real_rig`(真实 rig BEV 可见率 ≈94%,修前 1.25%) |
| 地图格式(§P-M.15/.16) | `test_lanelet2.py` `test_apollo.py` `test_stitch.py` | **往返是主判据**:六类 + 重复键 attrs + id/src + **顺序**逐字段全等;投影往返 < 1e-4 m 且**换 origin 结果必须不同**(防投影没生效);origin 随文件走、缺 origin **报错不猜**;格式合法性与**元素计数一致**(防静默丢要素);**第三方文件按语义标签尽力读回**;Apollo 侧另钉「`lane` 不许把 `left/right_boundary` 的点吸进 `central_curve`」与「按种类分组读**不许打乱顺序**」;**含 z** 逐分量往返(Town11 的 791 m 是压力点)。`test_stitch`:恒等 placement **返回同一对象**、单图拼接**逐位不变**、计数守恒、**z 保真**、去重容差边界、**同图内部不去重**、**单点要素参与去重**、变换口径手算钉 |
| MapQR 变体(§P-M.14) | `test_deform_attn.py` `test_bevenc.py` `test_variants.py`(结构钉在 `test_head.py`) | `test_deform_attn`:★ **用既有 `_sample_bev` 当 oracle** —— 偏移全零 + 权重 one-hot 时可变形注意力必须**逐点等于**该参考点处的双线性采样(两套实现互为对照,归一化/网格口径写错当场被抓);分块 == 不分块;梯度可达 value/偏移/权重;★ **静默失效钉**——米直接进 `sine_pos_embed` 与归一化后进**必须不同**(那错不报错、只「训不动」)。`test_bevenc`:★ **投影 oracle** —— `project_bev_anchors` 与 `gkt.project_pts` 逐点一致,**恒等/真实两套 ego 位姿各跑一遍**(恒等位姿只覆盖相机外参,K 缩放与 ego→world 复合都是它的盲点);★ **不变量钉** —— ego→图像矩阵**必须与 ego 位姿无关**(锚点与相机同挂 ego 上,位姿在复合里抵消;少做一步抵消就不成立);高度锚点恰为 `linspace(zmin,zmax,D)` 且偏移以归一化单位换算;可见掩码语义(含与 GKT 的**近平面口径差异**:本模块照官方只判 `z>eps`,无 GKT 的 0.5 m 近平面);kernel 网格偏移**末维必须是 (dx,dy)**;`D == n_heads` 与 `pc_range` 同源两处断言 |
| MapTR 时序 | `test_temporal.py` | **§P-M.12 阶段 4 的回归钉**,判据全是手算可验的闭式 oracle(线性场在均匀栅格上双线性插值恒等 ⇒ 采样值必须等于解析值,一条断言同时钉死归一化仿射 / `align_corners` / 两轴顺序)。`TestWarpBeV`:纯平移整格位移**逐元素相等**、float32 位姿量化上界、漏转置对照可被区分且**钉住它的盲点是 `ψ_prev=0`**、轴对调检查(第 49 列 / 第 99 行)、相对 pitch/roll 有位移而共同 pitch/roll 是 no-op。`TestTemporalFusion`:零初始化恒等、通道块序 `[当前, t−1, …]`、历史条数/`n_hist<1` 报错。`TestEgoRotations`:与 `gkt.cam_world_pose` 的 `r_e` 同源 + det=+1。`TestModelSeam`:**零初始化下时序版与单帧版输出逐位相同**(A/B 可比性的前提)+ 融合层在梯度路径上(判据用 **bias** 的梯度 —— `weight` 的梯度还要求 head 采样区落在 ReLU 激活区,实测可恰好为 0,拿它当判据会假红)+ **换历史帧必须改变输出**(证明历史 BEV 真的流到 head;前提 self-check:`num_vec` 太小会让锚点全落在 rig 覆盖外、head 对 BEV 免疫)。`TestCheckpoint口径`:`load_map_weights` 的两种错配分类(单帧→时序 = 放行热启动 / 时序→单帧 = 报错) |
| 标定自证 | `test_selfcheck.py` `test_depth_codec.py` `test_probe_calib.py` `test_calib_live.py` | `selfcheck`:轴目标物质心法在合成对称掩膜上精确复原注入的 cx;ray-plane 深度闭式解手算锚点;单侧可见性判据(渲染更近才判遮挡,对称窗口极差会误杀 95%)。`depth_codec`:深度编解码往返 + 像素约定。`probe_calib`:**实例分割解码公式**(`id = G + 256·B`,含两个旧错误候选作反例)+ A0/A1 锚 + "A1 能拦下历史镜像 bug" 的反向自证。`calib_live`:**判据不许报假数字**(样本不足 ⇒ `median_abs is None`,HUD 报"无数据"而非 0.000)+ 着色用 `index = u`(corner)+ `pooled_median` **等权** + 自遮挡**相对**判据(0.195 这个实时分辨率实测值也必须判得出来——绝对阈值 `> 0.2` 会漏) |
| 落盘契约 | `test_export_kitti.py` `test_export_nuscenes.py` | 路径/字段与消费方契约一致;`test_export_nuscenes` 另有**逐通道相机内参 == 官方 n015 K**(内参以前零覆盖)+ 蓝图 `fov` == `2·atan((w/2)/fx)` + LIDAR_TOP 落盘四元数 == 官方值(非单位) |
| devkit 表读取 | `test_nuscenes_cali_sensors.py` | 读 `sensor`/`calibrated_sensor`,打印每传感器相对 ego 的位姿 + 视线轴方位角/俯仰角。两处坑:①**视线轴因 modality 而异**(相机自身系 z 前 ⇒ 视线轴 +z;激光/雷达 x 前 ⇒ +x),对相机用 `Quaternion.yaw_pitch_roll` 读出的三个角无几何意义;②官方 mini 每通道 10 条记录但只 2 套位姿(n015 6 条 / n008 4 条),车辆/地点字段只能经 sample_data→sample→scene→log 反查。**可直接 `python` 跑**(打印),也可 pytest 收集;缺 devkit / 缺 dataroot 自动跳过。`test_repo_lidar_quat_is_identity` 于 2026-09-23 **改名为 `test_repo_lidar_quat_matches_official`** —— 旧断言测的正是缺陷(单位四元数) |
| oracle 对比 | `test_geometry_carla_oracle.py` `test_calib_oracle_autolabel.py` `test_gt_oracle_autolabel.py` `test_nuscenes_oracle_autolabel.py` | **需 autolabel env / CARLA 机器**,缺失时自动 skip |
| nuScenes 约定纯值 | `test_geometry_nus.py` `test_nuscenes_calib_consistency.py` | **不 skip、任何 env 可跑**。`test_geometry_nus`:四元数归一化(非单位的**来源是手抄 4 位小数舍入**,≥1e-5;官方原值 |q| = 1.000000000000)+ 相机 rig 推导钉(`yaw_carla = −az_nus` 到 1e-9、平移只翻 y、6DoF 不可降 yaw-only、历史字面表的 110°/217° 偏差量级)+ **LiDAR/雷达同一条对合规则**(`R_carla = CARLA_TO_NUS @ R_nus @ CARLA_TO_NUS`,雷达历史表差 94–136°、LiDAR 倾角 1.4289°)。`test_nuscenes_calib_consistency`:**「渲染 = 声明」同源钉**——① `collect_nus.CAM_YAW_OFFSET` 已删 / `CAM_ATTRS` 已拆、② spawn 位姿由 `NUS_CAMERA_RIG` / 官方雷达表导出、③ `LIDAR_MOUNT` + `LIDAR_ROT` 由四元数导出(≠ KITTI 线 `SENSOR_OFFSET`)、④ `CAM_FOV` == `2·atan(800/官方fx)`(fx 表在测试里独立硬编码,故非恒真)、⑤ `TestFovCriterionShape`:**AST 静态钉判据 ⑥ 的两个实测坑**(`world.tick()`/`q.get` 只能出现在 `drain` 内 = 每 tick 抽干全部相机队列;Z 必须是从远到近的**阶梯**、横移表关于 0 对称且 |frac| < 0.5 但 > 0.4)。**这是 §P-M.7「表对了图错了」的回归锁**。**wide rig 分支(§P-M.9)**:`TestWideRigDerivation`(前三个与官方**逐位相等**、后三路 `x == NUS_WIDE_REAR_X` 且在车身最后点之后、`y/z`+`pitch/roll` 保留官方、方位角由四元数**独立反解**得 145/180/215 且 `yaw_carla == −az_nus`)、`TestWideIntrinsics`(`fx` 闭式、主点 corner、fov↔K 往返、**`CAM_BACK` 必须窄于 180° 且 `det(K) > 0`** 的反回归闸门)、`TestCollectNusRigSelection` / `TestVerifyRigSelection`(拨模块级 `RIG` 后取表与内参必须跟着走 —— 防"wide 的验收静默拿官方表去比") |

## 4 `tools/` — 开放性工具

> 判据:**不含本项目领域知识**。含领域知识的编排脚本跟着它的 Python 走(如 `map/assemble_and_merge.sh`)。
>
> ⚠️ **一条明写的例外(2026-09-28,用户裁决)**:`showcase.py` 破了上面的判据 —— 它内嵌全部项目入口命令,
> 但它**跨能力面**,放 `calib/` 或 `map/` 都是误导,而包根只允许有 `__init__.py`。
> 裁决 = **留在 `tools/`**,并给它加一条**比"无领域知识"更硬**的约束:
> **只许调 CLI(`subprocess`),不许 import 任何业务模块** —— 某条命令写法错了,它必须**像用户手敲一样**地失败,
> 而不是靠 import 到内部函数把错遮住。

| 文件 | 职责 |
|---|---|
| `carla_server.sh` | CARLA 服务器启动/停止(GPU 修复栈 + **Vulkan 兼容层自愈**;宿主驱动升版致 `libnvidia-gpucomp.so.<ver>` 缺失时自动顶名,见 Plan.md §5.11f) |
| `gpu_fix/` | GPU 修复栈:`install.sh`(NVIDIA 用户态补齐 + shim 安装)+ `mhookshim.c`(LD_PRELOAD shim 源码) |
| `showcase.py` | **全能力面验收编排器**(2026-09-28):四阶段 `baseline`/`offline`/`online`/`figures`,把 [docs/acceptance-2026-09-28.md](acceptance-2026-09-28.md) 里逐条跑过的命令固化成可复跑入口。**只调 CLI 不 import 业务模块**(见上方例外条款);`online` 会改写 CARLA 所在图,跑完须 stop/start 回默认图 |
| `clear_cache.sh` / `gitpush.sh` | 【未入库】本机磁盘清理 / 推送辅助(环境维护,非项目代码) |

## 5 `docs/` — 文档

| 文件 | 职责 |
|---|---|
| `fileTree.md` | **本文件**:文件级索引与维护约定 |
| `milestone.md` | 版本里程碑(P1 / MapTR 线) |
| `milestone2.md` | 教程能力线里程碑(HiVT-CARLA / 语义 BEV / 单双目 / 建图 / SLAM / 3DGS) |
| `testLog.md` | 测试与验证日志 |
| `PRD.md` / `TRD.md` | 需求/技术文档(**当前为空占位**) |
| `Carla_Sim_Tutorial_01..16.md` | 16 篇 Carla 仿真教程(ros-bridge 旧栈),Plan2.md 的能力对照来源 |
| `refactor-2026-09.md` | **目录结构的决策与依据**:能力面划分的理由、已实测否决的方案、搬迁清单 |
| `acceptance-2026-09-28.md` | **全能力面验收记录**(2026-09-28):一次"从零把整条流水线重跑一遍"的逐数字对账(§0 扫描表)+ 复跑中新发现的 11 条问题(3 条已修并补回归钉、4 条待用户裁决、4 条口径注记)+ 产物索引 + 复跑须知。复跑入口 = [tools/showcase.py](../tools/showcase.py) |
| `ros2-humble-build.md` | **ROS2 Humble 源码构建记录**(2026-09-28,**环境侧**):三档成本实测(136/158/349 源码包)、四步命令与每步的坑、踩坑清单、构建期 github 抓取的镜像注入。**与能力线无关**(主线有意纯 numpy 不吃 ROS);`github.com` SNI 级被封的判据也在这一份 |
| `ros2-feasibility.md` | **ROS2 引入可行性评估**(2026-09-28):结论 = **不引入**。三条判据(GIL 疼点已被"同步执行"绕过 / 教程能力线 16 次选择不用它 / 代价可量化而收益不可量化)+ **分层守卫会静默失效**(`rclpy` 不在任何禁用集里)+ 工业惯例三层对照 + 唯一真候选(FAST-LIO2 对标基线)为何区分度低 + **五条触发重评条件**。与 Plan.md §5.12(官方 MapTR/MapQR 复线终止)**无关** |

## 6 `outputs/` — 产物目录(唯一落点,**不展开子文件**)

> 全部写盘路径经 `autodrivedata/utils/paths.project_path()`(相对路径 = 相对项目根,不随 cwd 漂移)。
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
| `maptr_600` / `maptr_1000` | MapTR 扩数据轮的 infos 与图像。**两轮的 `images/` 均已清理**(`maptr_1000` 于 2026-09-19 删 5.5 G;`maptr_600/images` 3600 图同期发现已不存在),现仅存 `maptr_600/map_infos.json`(33 M);`maptr_1000/` 已整目录不存在。**权重 `.pt` 均保留**。复现训练需重跑组装链(`surround_train` + `surround_p3` + `training/map/Town10HD_Opt_full.json` → `autodrivedata/map/finalize_maptr_600.sh`) | `assemble_maptr.py` / `merge_train_infos.py` |
| `kitti_sweep_day_clear_8` | **仅存**的速度档(70 帧 @8 m/s)。`_4` / `_12` 于 2026-09-20 清理(§5.10 结论已归档);重采 `collect_ab_route.py --speed N` | `collect_ab_route.py` |
| `kitti_wet_road` / `kitti_dense_rush` | **P1-6 候选**场景探针(各 12 帧,尚无 A/B 版)。同批的 `rain_night` / `dense_fog` 因已有 70 帧 A/B 版而列入待清;`heavy_rain` / `night_clear` **无** A/B 版,12 帧版是唯一数据 | `collect_drive.py` |
| `kitti_slam` / `slam_gt/` | SLAM 数据集(velodyne + **pose 真值**)与轨迹/精度产物(`traj_raw.json` / `icp_stats.json` / `eval_*.json` / **B 期验收 `accept_sync|accept_async|accept_parity_full.json`**) | `collect_slam.py` / `slam_odometry.py` / `eval_slam.py` / `live_studio.py --slam-report` |
| `kitti_loop` / `slam_loop/` | **§P-H.3 回环验证序列**:路网找环 + 纯追踪跑 2 圈的**闭环**数据(802 帧 / 1.98 圈,`route_loop.json` = 环几何 + 速度上限 + 帧预算,`slam_seq_summary.json` = 跟线偏差/里程/圈数)及其 SLAM 产物(`traj_raw.json` / `loops.json` / `traj_pgo.json` / `slam_summary.json` / `eval_pre|post.json`)。**后端实测 669 s**(196 个全量 ICP 候选;成本主项 = 失配云首迭代,见 §P-H.3.3) | `collect_slam.py --route loop` / `slam_odometry.py` / `slam_backend.py` / `eval_slam.py` |
| `calib_check/` | 标定自证产物:`report.json`(七锚 A0–A6 全数值)+ `overlay.png`(6 相机三层 overlay)+ **`check_{geometry,raw,overlay,rig_ab}.png` + `viz_summary.json`(人工复核图,见 `viz_calib_check.py`)**;`live.json` = `live_studio --calib` 实时槽读数(逐相机 n/median/p90/近场占比 + 时序;§P-M.10 重测:六路 `near_fraction` **全 0.0**,CAM_BACK 样本 n=11<20 ⇒ 报"无数据");**`rig_layout_{nuscenes,wide}.png`**(配置图)+ **`views_{rig}.png`**(六视角原生像素拼图 + 线性方位尺)+ **`report_{rig}.json`**(ego 像素/实挂偏差/覆盖表/共视) | `probe_calib.py` / `viz_calib_check.py` / `live_studio.py --calib-report` / `viz_rig_check.py` |
| `sem_bev/` `mono_distance/` `stereo/` `multilidar/` `accum_map/` `ground/` `cluster/` | 教程能力线各产物的图/点云/结果 json | 各自 对应的 `autodrivedata/<能力>/` 脚本 |
| `3dgs/` | 3DGS 环绕采集帧 + 真值深度位姿 + `.ply` 高斯 + 训练结果 json | `collect_3dgs.py` / `train_3dgs_mini.py` |
| `hivt_carla/` | HiVT 训练用 TemporalData 与场景划分 | `convert_hivt_pt.py` |
| `carla/` | **运行支撑物**:服务器日志 + LD_PRELOAD shim + Vulkan 兼容层 | `carla_server.sh` |
| `models/` | 推理权重(YOLOPv2 等) | 下载/转换 |
| `nus_mini*` | nuScenes 迷你集(含 L3 雷达口径变体)。**⚠️ 两个旧产物已标废弃**(2026-09-23,§P-M.7):`nus_mini`(已重采覆盖,现 36M)——旧版是「**表对了、图错了**」(相机挂 LiDAR 挂点 + 镜像偏航、雷达偏航差 94–136°、LiDAR 无 rotation)⇒ 不存在"改几行标定就能救"的路径,与 §P-M.5 对 MapTR 权重的处置同口径。**2026-09-23 又重采一次**(§P-M.10 挂点原点:旧版 12 路全偏前 1.2563 m、`ego_pose` 丢悬架俯仰) | `collect_nus.py` |
| `nus_calib_check/` | `collect_nus` 验收判据的复现报告:`report.json`(**修后**逐判据 pass + 原始数字,含 LiDAR 单位阵消融对照 1.0000→0.1684)+ `report_prefix.json`(**修前**基线,三条离线判据全 ✗ 的固化证据)+ **`report_wide.json`**(wide rig;**十条判据**,③④ 为与相机无关的不变项,⑨⑩ = §P-M.10 新增) | `verify_nus_calib.py` |

### `logs/` — 运行日志(顶层独立目录,**不是产物**)

> 【未入库】,已被 `.gitignore` 忽略(回归钉 `test_paths.py::test_logs_dir_is_ignored`)。

18 个训练/推理/评估入口**每次跑都落三件套**(同 stem;语义见 `utils/runlog.py` 行):
`<能力>_<模块>_<YYYYmmdd-HHMMSS>.log`(全量 stdout 文本)/ `.jsonl`(逐迭代指标)/ `.json`(环境指纹 + 入参 + 产物表带 sha256 + 结论)。
`logs/latest/<能力>_<模块>.<ext>` 是**相对软链**,指向该脚本最近一次 —— `tail -f` 用它。
**增量式、不滚动**(每次一份):先看实际增长速度,嫌多再议清理,**本轮不做**。

**可否删**:数据集与权重删前先确认 Plan2.md §5「数据资产」是否仍被引用。

> ⚠️ **已删目录已从本表移除**(2026-09-20,详见 Plan2.md §10):`dumps/` `viz_check` `viz_maptr_e120`
> `kitti_{slam,town13,town13_static}_probe` `maptr_600_pred` `mapvec_pred_*` `kitti_sunset_glare`
> `kitti_sweep_day_clear_{4,12}` `surround_micro_{legacy,official}` `zzz_probe.txt` `videos/`(空目录)。
> 这些目录**克隆后本来就不存在**(【未入库】),由对应的 `autodrivedata/<能力>/` 脚本重新生成 —— 需要时重跑即可。
> 保留 `kitti_sweep_day_clear_8`(速度档只剩 `_8` 一档)。

## 7 非代码目录(环境 / 上游 / 缓存)

| 目录 | 职责 |
|---|---|
| `training/map/` | 【未入库】地图矢量**全量**导出(`{map}_full.json` + BEV overlay + 帧级裁剪版),供 MapTR 训练组装消费 |
| `lightning_logs/` | 【未入库】HiVT 训练日志与 ckpt(PyTorch Lightning 默认落点) |
| `auto3dlabel/weights/` | 【未入库】3D 检测微调权重(AutoLabel 消费方) |
| `hdMapGitHub/` | 【未入库】上游开源仓库克隆:`HiVT` / `MapTR` / `MapTR_maptrv2` / `MapQR` / `FAST_LIO` / `maptracker`,**保持 pristine**,项目侧改动一律放 `autodrivedata/map/maptr/` 与 `autodrivedata/map/maptr_official/` |
| `.vscode/` `.claude/` `build/` `*_cache/` | 【未入库】本地 IDE 配置、AI 会话配置、构建与测试缓存 |

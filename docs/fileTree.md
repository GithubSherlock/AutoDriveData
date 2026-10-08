# 文件树:AutoDriveData 仓库结构与文件职责

> **本文档的用处**:仓库的**文件级索引**——"某个文件是干什么的、该改哪、产物落在哪"。
> 定位是**导航**,不是事实源:当前状态与红线看 [CLAUDE.md](../CLAUDE.md),目录结构的**决策依据**看
> [docs/refactor-2026-09.md](refactor-2026-09.md),方案定案看 [Plan.md](../Plan.md),新计划/待办看 [Plan4.md](../Plan4.md),主线外的外部数据集对照看 [Plan3.md](../Plan3.md),
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
├── Plan.md(冻结)· Plan2.md(冻结 · §P-M.1–P-M.20)· Plan3.md(主线外:外部数据集对照)· Plan4.md(新计划制定地)
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

> ⚠️ **下面「规模」那一列是 2026-09 的快照,已经漂了**(2026-10-07 复核:实测
> `perception/` **30**、`edit/` **14**、`sim/` **32**、`tests/` **109** 个 `test_*.py`,
> 而表里写着 17 / 11 / 25 / 66)。**别照着表里的数做任何判断**,复核用:
> `ls autodrivedata/<目录>/*.py | grep -v __init__ | wc -l`。
> (未在本轮统一重算:那要先定一个口径 —— 现在这些数字的原始定义已经找不回来了。)

| 能力目录 | 规模 | 一句话 | 层约束 |
|---|---|---|---|
| `sim/` | 25 | CARLA 仿真交互层 + 全部采集器 + 平台探针 + 闭环路线纯值 | 许 carla |
| `calib/` | 13 | 标定原语 / rig 表 / 自证探针 / 实时监看 / 配置图 | 许 carla,禁 torch |
| `map/` | 36 | 地图矢量 + MapTR(含 `maptr/`、`maptr_official/`)+ 时序拼接 | 不设限 |
| `slam/` | 10 | 两段式激光 SLAM + 精度评估 + C++ 对拍 | **纯值** |
| `perception/` | 17 | 检测 / 单双目 / 雷达 / 语义 / 点云 | **不 import carla** |
| `gt/` | 5 | 动态目标 + 静态目标 + 灯态 + 落盘导出 | **纯值** |
| `traj/` `gs/` | 5 | 轨迹组装转换 / 3DGS 训练 **+ 归属 + 帧同步判据** | 许 torch,禁 carla |
| `runtime/` | 3 | 设备探测 / TF32 口径 / 显存体检 / 批量超参实测 + **训练早停判据** + **jsonl→TensorBoard 转换** | 许 torch,禁 carla |
| `edit/` | 11 | 图像/场景编辑:后端 + 条件源 + 保真度 / 下游 / 注入 / 时序 / 和谐化 判据 | 许 torch,禁 carla |
| `utils/` | 4 | `geometry` `paths` `fonts` `runlog` | **纯值** |
| `tests/` | 66 | 与能力目录镜像(见 §3) | 不设限 |

**规模列口径(两个,别混)**:§2 表 = **递归**(`find <目录> -name '*.py' ! -name '__init__.py'`);
各小节标题 = 该目录**本级**文件数(子包另立小节,如 `gt/` 记 `3 + export/ 子包`、`map/` 记 18
而 `maptr/` 12 + `maptr_official/` 4 另行)。两者都**不含 `__init__.py`、不含 gitignore 的
Jupyter 残留**(`.ipynb_checkpoints/`,它会污染计数);标的是量级,加/删文件后回来顺手改一行。

**命名约定**:模块与所在目录同名时改叫 **`core.py`**(`calib/core.py`、`slam/core.py`、`gt/core.py`)——
避免 `autodrivedata.calib.calib` 这类自反名。

**运行留痕(2026-09-27)**:每个**训练 / 推理 / 评估**入口跑一次就在 `logs/` 落**三件套**
(stem 相同:`<能力>_<模块>_<时间戳>.log` 全量文本 / `.jsonl` 逐迭代指标 / `.json` 汇总),
口径与关法见下方 [`utils/runlog.py`](#utils--通用件4) 行,落点见 §6。
下表标 **落 `logs/` 三件套** 的**到 2026-10-02 为止 20 个**(之后又补了 `gs/frame_sync` 与
`gs/attribute_instances` —— 两个**判据类**入口,跑出来的数要能事后归因 ⇒ 现在 **22 个**;
⚠️ **计数以 [CLAUDE.md](../CLAUDE.md) 常用命令那一行为准**,这里只是当时的快照):
用户裁决的覆盖范围(16 = 训练 3 + 推理/评估 13)外加
**2026-09-27 补入的两个 SLAM 链路入口**(`slam/slam_odometry`、`slam/slam_backend` —— 前端 12.6 min、
后端 11 min 却都**只在末尾落盘**,中断即零痕迹,是这三类里留痕价值最高的一对);
**采集器(`sim/collect_*`)、数据组装(`map/assemble_*` / `merge_train_infos`)、标定探针(`calib/*`)仍不在其中** ——
它们的产物自带逐帧索引,留痕价值低于上述三类。

### `sim/` — CARLA 仿真交互层 + 全部采集器(25)

**唯一大面积 `import carla` 的目录**;采集器、实时流、平台探针都在这。

| 文件 | 职责 |
|---|---|
| `scenarios.py` | Corner case 场景目录 + 可模拟性矩阵(P1) |
| `collect_rig.py` | 采集器纯值位姿/挂点计算(环绕位姿 / 双目挂点;分层守卫见 `tests/test_layer_guard.py`) |
| `probe_ego_teleport.py` | ★ **ego 逐帧瞬移的位姿可不可复现**(需 CARLA)。同一串绝对位姿跑两遍,用**已有的** `lidar_ab.pose_series_report` 比(跨 root 才是「逐位可复现」的定义)。实测:瞬移两遍 **0.39 mm**、物理对照 **11.43 mm**、归档物理 A/B **61.9 mm** ⇒ **不是「逐位」(0.39≠0)但改善 158×**。⚠️ 两处自己踩的坑:① 拿物理臂算「请求 vs 读回」**是范畴错误**;② 第一版传了**单位矢量**(1 m/s 而非 8 m/s) |
| `route.py` | **闭环路线纯值**:`find_cycle`(路网 BFS 最短有向环,吃 `successors` 回调,可在合成图上单测)/ `make_successors`(CARLA `wp.next()` → 节点键 `(road_id, lane_id, round(s/step))` 的适配层,只吃 duck-typed `WaypointLike` ⇒ **零 carla**)/ `pure_pursuit`(前视点 + 单调进度 + 打舵符号)/ `lap_budget` + `speed_ceiling`(**一圈帧数 ≥ 250 帧 ⇒ 速度上限**,采集前的第一道闸)。零 carla,供 `collect_slam --route loop`。**另含多段采集的段起点选点**:`farthest_from_centroid`(★ 种子规则,**反推出来的** —— 它在 Town10HD_Opt 上逐位复现了 `surround_v2_epic` 现算的 `44/14/15/152/55`,最近间距 113.994 m)/ `greedy_maxmin{,_order}`(贪心最大最小距离;**选择序** vs **升序集合**是两个口径,`milestone2` 记的是选择序)/ `min_pairwise`(取**最近**那一对;单点返回 `inf` 不是 0)/ `spread_curve`(k→最近间距,判"这张图撑不撑得起 k 段") |
| `collect_kitti.py` | 静态采集:ego 静止 + 摆 NPC + 同步模式 → KITTI root(raw + GT) |
| `collect_drive.py` | 动态采集:ego autopilot + TM 车流 + 行人 → KITTI 序列 |
| `occlusion.py` | **静态遮挡 A/B 的纯几何**(零 carla,§P-V12):`sight_line_occluders`(遮挡物排在**相机→车 的视线**上、长轴垂直于视线 —— 排 ego 正前方会因视差在车内侧漏一条**全高**的缝,实测占车宽 18.3%)/ `required_span`(车宽投到遮挡物所在**深度**上的跨度)/ `visible_fraction` + `project`(挡完还剩多少 px)/ `occluder_depth`(深度闭式解)。**两个约定混一个数就全错**:成像吃**深度**(沿光轴)不吃斜距 —— 20 m 处整框 49.7 vs 58.6 px。资产尺寸只吃 `OCCLUDER_DIMS` 一份声明,采集时另起 **yaw=0 探针**读回对表 |
| `collect_ab_route.py` | P1 A/B 专用:ego 定速直行 + 路侧静置车(固定位置,帧级配对)。**`--occluders {none,partial,full}`**(§P-V12)+ `--occluder-cars {all,nearest}` + `--occluder-gap`:只用 `static.prop.*` 道具 ⇒ **结构上进不了 `label_2`**(GT 过滤器只认 `vehicle*`/`walker*`)⇒ 同目标、逐帧 GT 相等、单一变量。⚠️ 它**不读 `scene.traffic`**,`--scene dense_rush` 会静默退化成 day_clear。逐帧落 `training/pose/`(A/B 轨迹配对的机械判据) |
| `collect_nus.py` | nuScenes 迷你集:6 相机 + LiDAR + 5 雷达。**全传感器标定「渲染位姿 = 声明位姿」同源**(2026-09-23,§P-M.7):相机走 `NUS_CAMERA_RIG`(逐相机 6DoF 挂点)+ 逐通道蓝图 `fov`;LiDAR 走官方挂点 + 由四元数**导出**的姿态(`LIDAR_ROT` 不手抄);雷达偏航 `−az_nus` **由 `NUS_RADAR_OFFSETS` 导出**。历史缺陷(已删 `CAM_YAW_OFFSET` 镜像表 / 雷达猜测表 / LiDAR 无 rotation)见模块头注的对照表。判据复现器 = `autodrivedata/calib/verify_nus_calib.py`。**挂点原点**:位置/姿态全走 `geometry.nus_ego_translation` + `nus_ego_rotation`(后轴,§P-M.10);挂传感器前先让车静置收敛(`ego_settle`,实测 8 tick),`ego_pose` 写全 6DoF |
| `collect_surround.py` | 环视 6 相机采集(nuScenes 布局)→ 图像 + 内外参 + 逐帧 ego 位姿。**`--sem`**(§P-V13)逐相机多挂一路 `sensor.camera.semantic_segmentation`(**同挂点同 fov**),落 `sem_<cam>/{fid}.png` = 8 位灰度、tag 即像素值的无损 PNG ⇒ 这是 **P2-A 的像素级分割 GT**;**`--inst`**(§P-V15)再挂 `instance_segmentation`,落 `inst_<cam>/{fid}.png` = 16 位灰度、**像素值 = actor id**,R 通道同时带语义类 ⇒ 实例分割 GT。两者默认关,对既有管线零改动。⚠️ 落盘**走 PIL 不走 `save_to_disk`** —— 后者写的是当前转换器下的图(默认是上色预览)。⚠️ **预热与 stride 两个循环都必须逐 tick 抽干全部队列**(含后来的 sem/inst):漏一路不是"少收几帧",而是**那一路整体滞后 N 帧**,而它与别的路"都有数据、帧号也对得上"(2026-10-01 实测:漏 `inst_qs` ⇒ 首帧自证 0.76 而非 1.000) |
| `collect_surround_lidar.py` | **多传感器采集:6 环视相机 + LiDAR + 5 雷达 + ego 真值位姿,一次过落进同一个 root**。存在的理由:两条下游链各要各的,而**磁盘上没有任何一份数据集同时有两者**(`collect_surround` 只挂相机 / `collect_slam` 只挂 LiDAR)⇒ 「逐帧环视预测 + SLAM 位姿」这种组合**无数据可跑**(`map/stitch_temporal --pose slam` 就卡在这)。落盘**三份布局**(**刻意不统一**:各走各自已有消费方的形状)—— ① `cam_*/` + `calib.json` + `ego_pose.json`(喂 `assemble_maptr`)② `training/velodyne/` + `training/pose/`(喂 `slam_odometry` / `eval_slam`)③ `samples/RADAR_*/*.pcd`(devkit 18 字段,喂 AutoLabel 雷达线)。雷达有**相位空 tick**(~7%)⇒ 非阻塞 drain + 空 pcd,`--no-radar` 可关。定速照 `collect_ab_route` 红线(清制动残留 + **逐帧反算实测速度自证**) |
| `collect_surround_micro.py` | 环视微采样(10 帧)→ **极小规模**的环视输入(投影链 / 组装器 / 可视化改动的冒烟用,比 400 帧的 `collect_surround` 便宜)。原先的「与 legacy 旧布局做对照」用途已随 legacy 移除 |
| `collect_static_gt.py` | 静态目标/道路特征 GT(地图查询源,含 overlay 目检图)。**★ 帧同步自证**(§P-V22):`settle()` 排空 + tick 到「每队列恰好剩当 tick 那一帧」才进主循环(原实现只排空一遍,**在途帧**会补进来 ⇒ 整段序列恒定滞后 1 帧),`assert_synced()` 每 tick 比三路 `Image.frame`、不等**当场停**。**`--sem`(§P-V21,默认关)**:同挂一台语义相机 → `static_sem/{fid}.png`(**判据的 oracle**),并在 `static_gt/{fid}.json` 里落**可选的**相机位姿(`StaticFrame.camera` —— 判据在 `perception/`,层规则禁 carla,拼不出投影链)。与 `--props` **共用同一台**语义相机(`--props` 写 `prop_sem/`、`--sem` 写 `static_sem/`,同开时两份逐字节相同);缺位姿**直接抛**,不回落默认内参;拒绝往已有 `static_sem/` 的 root 追加。**`--props`(§P-V14,默认关)**:摆 5 个受控道具(锥桶/路障/施工围挡)出 `static_prop_gt/` + 实例 id 图(uint16 PNG) + 语义 tag 图 + 3D 盒 overlay;横向由「ego 半宽 + 资产半宽 + 余量」**现算**(写死米数对宽 2.11 m 的围挡会把车身伸进 ego 路径,而**撞了照样采得完**);摆位自证比 x/y/z 三项;拒绝往已有道具 GT 的 root 里写(两轮产物混在一起,判据会误报"实例没露面") |
| `probe_radar_on_targets.py` | **雷达到底打不打得到目标**(§P-V17 六之二,需 CARLA):**一场戏三臂、不销毁任何东西**(上次崩在 `car.destroy()`)—— 同距离 20 m、横向 −3.5 / +3.5(5 m/s 直行)/ **0(空位对照)**,一次 tick 序列同时量。换算**不用我们的任何表**(CARLA 原始检测 → 传感器自身系 → `sensor.get_transform()` 送世界系),绕开 §P-V17 那场偏航符号之争。自证两条:LiDAR 两臂都看得见(2349/832 点)+ 运动臂逐帧确实在动(0.50 m/帧)。**现象刻画 + 六个排除项;机制未知不写成因**(§P-V17 六之二):可归因回波**只在视轴上**(横向 0 槽位换什么蓝图都出:车 32318 / 行人 21019 / 挡板 6347),±9.9°/±19.3° **一个点都不给**,而 P1 目标正在 ±3.5 m。已排除:离得近 / 射线预算 100× / FOV 实测 ±38.49° / 射线密度只低 35% / 假点云(65% 与 LiDAR 表面差 <0.3 m)/ 简单复制(88/88 唯一)。那批点:确定性(逐帧 2582–2722)、最密 0.05° 箱内 **88 个检测**深跨 **0.06 m**、速度与视线方向无关(R²≈0);`moving` 臂结构正常(唯一值 ≈ 点数)⇒ 不是传感器坏。★ 两条边界:是「这套配置下、这批目标」的结论;**目标摆在视轴上雷达就有回波**。开关:`--no-targets` / `--pps-mult` / `--dump-shape` / `--ego-speed` / `--project` / `--slot0-bp` / `--depth-check` |
| `probe_radar_camera_coaxial.py` | **共轴相机 + 雷达**(需 CARLA,Plan4 §D1/D2):两台传感器**逐字相同的 `Transform`**、都不挂父 actor,按方位扫一台车。实测:**相机五个方位都看得见**(2.3–3.0 万 px),雷达**净贡献只在视轴**(+39;±10° 仅 +1/+2、±20° **0**)⇒ §P-V17「回波只在视轴」在共轴下**复现**,排除掉「是生产挂点差 1.99 m 造成的」这个解释。⚠️ 视轴 8 m 处本底有 **29.5 点**杂波 ⇒ **必须减本底** |
| `probe_prop_renders.py` | **道具到底渲不渲染 —— 重做 §C.0.4 ①**(需 CARLA):原判据「车渲染、道具不渲染」的产出探针**没留下来**,而且是 79 m 位姿缺陷**之前**量的。重测(先量本底,再看倍数):pitch −15 道具 **7.2×** 本底、车 **20.2×**,**两者都渲染**;差分图里是**连贯的物体形状**(车那张能认出车牌与奥迪车标)。★ 探针第一版**自己踩了「抽干≠抽干净」**(换位姿后只 drain 一次 ⇒ 拿到旧视角那一帧),得到 437864 px 的整幅误报 —— **铁证是两个完全不同的物体给出几乎相同的差**(437864 vs 438030) |
| `probe_3dgs_cam_pose.py` | **3DGS 相机位姿口径的反向自证**(需 CARLA,§C.0.4 ③):相机 attach 在 spectator 上 ⇒ `set_transform` 按**父系**解释,而 `ring_cam_pose` 给的是**世界**坐标。探针**两种写法各测一次**:`raw`(修复前)差 **78.034 m**、读回 `(-150.068, 25.934, 4.200)` —— 与归档记的那组读数**逐分量吻合**;`converted` 差 **0.000 m**。`|spectator 世界位姿| = 79.132 m`。★ 反向对照写在判据里:若 raw 也 ≈0(父恰好在原点)则判**「无分辨力」并 exit 1**,不让'看着都对'混过去 |
| `probe_spawn_points.py` | **多段采集的段起点选点**(需 CARLA,只读:不 spawn/不 tick/不清场)。★ **自证**:`--expect 44,14,15,152,55` 在 Town10HD_Opt 上必须复现 `surround_v2_epic` 那五个索引(默认规则 = 离质心最远,实测种子 44、最近间距 **113.994 m**);对不上时**遍历所有种子**给最接近的一组,不做"差不多就算过"。`--out` 落 `outputs/<图>_spawns.json`。⚠️ 大图(Town13 12478 个点)跳过遍历诊断(`MAX_SEED_SCAN`),不然 O(N²) 跑到天荒地老。实测 Town05_Opt:38/92/249/255/267,最近 **213.4 m** |
| `probe_static_prop_gt.py` | **一次性探针**(§P-V14,需 CARLA):静态道具能不能出 GT 的四方实测。① 地图**烘死**的道具在 `get_actors()` 里 **0 个**,而清空所有车/行人后语义相机仍有 **6999 个 `Dynamic` 像素**(关卡网格,无 transform 可查);② `Actor.bounding_box` 对**转过**的 actor 给的是被**剪切**的错值(yaw=30° 读 `0.1246×0.4704` vs 真值 `0.3441×0.3441`),拿渲染轮廓量实测 **3.4% 像素落到框外、IoU 0.897→0.708**,而 **0°/90° 都读对**;③ 三个资产在实例相机与语义相机上**都是** 21 `Dynamic`(两代同源);④ 杂项 tag 的**连通域形态**。落 `outputs/probe_static_prop_gt/report.json` |
| `collect_tl_states.py` | 灯色动态 GT(记录模式 / `--cycle` 受控切灯) |
| `collect_traj.py` | 多 agent 轨迹采集(HiVT 训练数据源) |
| `collect_stereo.py` | 双目 rig 采集(基线 0.4m)+ 真值深度 |
| `collect_3dgs.py` | 静态场景 360° 环绕采集(RGB + 真值深度,支持多俯仰) |
| `collect_slam.py` | SLAM 数据集采集:ego 定速巡游 → `training/velodyne/` + **`training/pose/`(ego 真值位姿,ATE/RPE 评估的 GT)**。**`--route loop`** = 路网找环 + 纯追踪跑 `--laps` 圈(让同一处真被走两次,后端回环链路的端到端数据源);`--dry-run` 只找环报预算(闸门见 `route.py`),`--spawn-index` 换可复现起点 |
| `view_stream.py` | 场景实时流:真 UE 渲染 + GT/预测 overlay → 浏览器 MJPEG |
| `live_common.py` | **实时可视化共享件**(从 view_stream 抽出):多槽 MJPEG 服务(单端口 `/stream/<name>` + `/` 索引页)/ **拼图(两套:`compose_grid` 等尺寸 + **尺寸守卫**,不符即 `ValueError` —— `paste` 源图大于目标框时只贴左上角、静默裁;`compose_rows` 按行拼、每格**原生像素**,studio 三层用它)** / GT overlay / **环视 rig(`rig_spec`,可选口径 = `RIGS`:`nuscenes` 逐相机 `SENSOR_MOUNTS` + 官方 6DoF 姿态 / `wide` 后三路后移;`legacy` 与 `resolve_rig` 已于 2026-09-28 移除)** / 第三方视角 / 键盘 —— view_stream 与 live_studio 共用。**`build_surround_rig(..., kind=)`** 可挂 `rgb` 或 `depth`(深度槽必须与 RGB 槽**同挂点同内参同分辨率**,否则 overlay 无法逐像素对齐);`draw_hud(..., y=)` 支持第二行(条带 16 px) |
| `live_studio.py` | **8 路 studio**:6 相机 + BEV + 第三方 + `grid` 拼图槽,各占一路;`--keyboard` 折进 tick 循环(WASD 开采集);第三方非 attach 每 tick 摆位。**拼图 = `GRID_ROWS` 三层**(①左前/前/右前 ②右后/后/左后 ③第三方 + BEV),**每格原生像素不缩放**(相机 1242×375 / 第三方 640×360 / BEV `--bev-size` ⇒ 画布 3726×1170;旧 4×2 等尺寸布局把相机图裁到 621×187 丢掉地面,见 Plan2.md §P-L.6)。**`--slam` 接在线 SLAM**(挂语义 LiDAR → `SlamWorker`,BEV 槽画地图点/轨迹;`--slam-async` / `--slam-voxel` / `--slam-max-gap` / `--slam-report` 落验收 JSON)。**`--calib` 接实时标定监看**(另挂 6 深度相机同挂点同内参 + `sensor.lidar.ray_cast`,LiDAR→世界平面→投影回相机按深度残差着色画进各相机槽,`draw_hud` 第二行报 pooled |e| 与逐路样本数,`--calib-report` 落 JSON;`--calib-refit` 默认 2 tick 重拟合一次,预算见 `calib_live.py`)。**`--video` 落八视角视频段**(拼图槽逐帧写 mp4,cv2/mp4v 惰性开编码器;`--video-fps` 标称帧率 / `--video-tile` 放大倍数) |
| `drive_ego.py` | live 手动驾驶(服务器终端 WASD 遥控);薄封装 `live_common.KeyboardState`(studio 内置键盘是首选) |
| `probe_radar_l3.py` | 雷达物理合理性探针(对照真实 ars408 规格) |
| `probe_vulkan.py` | Vulkan 设备枚举——CARLA 渲染停摆的一线判据 |
| `probe_imu.py` | **B1 实测**:CARLA IMU 能否支撑 FAST-LIO2 的 IESKF 预测(结论:直行段 IMU 预测比恒速先验更差 → B3 不投) |
| `smoke.py` | M0 smoke:CARLA headless 连接 → 同步模式 → 各取一帧落盘 |
| `carla_common.py` | 采集公共件:位姿换算 / NPC 摆放 / 传感器参数 / 同步模式 / 灯态归一与绘制。**`load_world(client, name)`**(2026-10-03):切图期间把客户端超时抬到 `LOAD_WORLD_TIMEOUT_S=300`,切完恢复 —— 三个采集器原来各写一遍 `set_timeout(60)` → `load_world` → `set_timeout(60)`,而 `collect_surround` 那处的注释**自己写着**「~2min」,**注释与超时值互相矛盾**;Town13 一跑就现形:`RuntimeError: time-out of 60000ms`,而**世界其实加载成功了**(紧接着 `get_world()` 已是新图)⇒ 属于"报错但没坏",照着错误提示去查服务器会白查。`ClientLike` 是 duck-typed Protocol(同 `route.WaypointLike` 的理由:让超时语义能在假 client 上单测),⚠️ **形参名必须与 carla 桩逐字一致**(`second`/`map_name`) |
| `smoke_radar_collect.sh` | 雷达采集冒烟 |

### `runtime/` — 运行时设备工具(3)

**为什么单独立档**:这些是**跨能力面**的 torch 运行时问题(map 用它调训练 batch、eval 用它调推理
batch、sim 用它做设备探测)。放 `utils/` 不行 —— 那层是 `_PURE`、**禁 torch**,有守卫强制。

| 文件 | 职责 |
|---|---|
| `device.py` | 设备探测(`get_device`/`print_device`)、**TF32 口径**(`disable_tf32`)、显存体检(`check_gpu_headroom`)、批量超参实测(`tune_batch_size` 通用核心 + 训练/推理两个薄封装)。**最有价值的是 TF32**:`cudnn.allow_tf32` 在 torch 2.x 默认 True 而全仓原本无人设置 ⇒ 归档 AP 是 TF32 关的口径、今天默认开着跑出 0.3048 vs 归档 0.3043(`pred/gt` 计数完全一致)。合并自 AutoLabel `auto2dlabel/tools/device.py` + 原 `map/maptr/device.py`,三处改动见模块头注 |

| `early_stop.py` | **训练早停:两段式平台检测**(2026-09-29,**纯值**,不 import torch/carla ⇒ 判据可离线单测)。`PlateauDetector` 判 loss **相对**改善是否停滞(相对而非绝对:loss 尺度随配置变,实测时序 15→2.68 / 单帧要求 →0),且**触发理由分型** `converged` / `lr_exhausted`(后者也停,但该去调 lr 计划重跑 —— Plan2 明写「平台是 lr 归零不是收敛」);`EarlyStopper` 在平台候选上按 `confirm_every` 节流调**注入的** `confirm_fn` 复核 AP,`confirm_fn` 返回 None(复核做不了)⇒ **不敢停**。本模块**不知道 AP 是什么**,那是 `train_maptr` 注入的 |
| `tb_export.py` | **`logs/*.jsonl` → TensorBoard event(离线转换,2026-09-30)**。**不在训练里实时写** `SummaryWriter`:jsonl 已是逐迭代的唯一口径,再开一路 event 就是两份记录(本项目的既有过敏:`SAFETY_FACTOR` 两处不同值),且实时写对**已在跑的** run 无效。转换器**不认任何训练语义** —— `runlog.metric(step, **kv)` 是全仓契约,一份代码同时能画 `loss` / `ATE` / `closure`。两个必处理的坑:**标签稀疏**(早停复核行同 step 但没有 `loss/lr` ⇒ 逐行"有什么写什么",补 0 会拉出假台阶)、**一个名字配多个 jsonl 必须拆成多个 run**(合并 = 两段训练交替进同一条曲线且 x 轴互相覆盖,而曲线看着正常)。另记:TB scalar 存 **float32**(`18.04` 读回 `18.040000915527344`)⇒ 图与 jsonl 不逐位一致,要精读回到 jsonl;event 是**快照**,刷新要重跑 |

### `calib/` — 标定(13)

许 carla、禁 torch。

| 文件 | 职责 |
|---|---|
| `core.py`(原 `calib.py`) | KITTI 标定生成(内参/外参 → calib txt,含 `world_to_img` 投影共用件) |
| `camera_rig.py` | **环视相机 rig 唯一来源**:官方 nuScenes `calibrated_sensor`(6DoF 四元数)→ CARLA 采集口径 `NUS_CAMERA_RIG`(平移 y 翻号 + 姿态走 `nus_camera_rotation_to_carla`)。采集器 / 实时流 / 导出器同源;模块头注记录**镜像 bug**(`yaw_carla = −az_nus` 漏翻 ⇒ 四个侧/后相机左右互换)。平移一律 = 官方表 + `geometry.NUS_EGO_ORIGIN_X`(§P-M.10 后轴对齐);wide 的后三路常量是 **CARLA 口径**的 `NUS_WIDE_REAR_X_CARLA = -1.9000`(落盘 nus 侧 = −0.6437),命名即口径,别混。**头注的四元数模长归因已于 2026-09-23 订正**:非单位的来源是**本表手抄的 4 位小数**(如 `CAM_FRONT_LEFT` `|q|−1 = −5.04e-05`),官方 mini 120 条 `calibrated_sensor` 的 |q| 实测全为 1.000000000000。**另含六视角画布口径**:`CAMERA_GRID_ROWS`(2 行×3 列,按方位绕车:左前/前/右前 + 右后/后/左后)+ 取序入口 `camera_grid_rows` / `camera_grid_order`。**七处产出点同源** —— `live_studio`(实时拼图,定义源头)/ `viz_rig_check`(六视角实拍)/ `viz_layout_cmp`(布局对照)/ `viz_calib_check`(标定 A/B)/ `probe_calib`(残差 overlay)/ `viz_maptr_pred`(预测回投)/ `view_stream --view grid6`。2026-09-28 收敛:此前各处自己写,其中**四处是错的**(一处第二行左右反、一处 1×6 字母序、一处把后三路排到了第一行、一处 2 列×3 行形状不符)。回归钉 `tests/calib/test_rigviz.py::TestCameraGridRows`(方位扫描 + 产出者必须真 import + 禁 `sorted(相机集合)`) |
| `selfcheck.py` | **标定自证纯值件**:平面拟合(`fit_plane`/`fit_local_planes`)、相机射线/反投影(`cam_rays`/`project_world`/`backproject_depth`)、深度图采样(`collect_samples`/`DepthSamples`)、轴目标物质心法主点裁决(`estimate_axis_delta`/`estimate_delta_uv`)、镜像不对称度(`mirror_asymmetry`)、径向误差剖面 |
| `depth_codec.py` | CARLA 深度图编解码(`decode_depth`/`encode_depth`,BGRA→米)+ 采样口径(`CONVENTION_CENTER`/`CONVENTION_CORNER` + `sample_bilinear_many`)——**实测裁决 CARLA 光栅 = corner**(索引 i 即连续坐标 i;见 `probe_calib.py` A3/A4),`CENTER` 保留供对照。**坑:`decode` 与 `sample` 的像素索引约定必须一致**,差 0.5 px 在近处 = 米级深度误差 |
| `calib_live.py` | **实时标定槽纯值件**(`live_studio --calib`):`live_planes`(体素+抽样+逐点邻域平面,`LIVE_*` 廉价预算)/ `sample_camera`(复用 `selfcheck.collect_samples`,`CONVENTION_CORNER`,不开窗口极差)/ `residual_colors`+`paint_residuals`(着色,**与离线探针同源**)/ `summarize`+`CameraResidual`(样本 < `MIN_CAM_SAMPLES` 时 `median_abs is None`,**不许报假数字**)/ `self_occluded_cameras`+`near_fraction`(**相对**判据,见下)/ `hud_line`。**坑:自遮挡判据不能写死阈值**——近场占比随**画幅宽高比**变(CAM_BACK 1242×375 是 0.367、640×360 只有 0.195),故按同批可用相机的近场占比中位数定阈;平面是**世界系**故可跨 tick 复用(瓶颈全在拟合 ~100–150 ms vs 六相机采样 ~9 ms ⇒ 默认每 2 tick 重拟合) |
| `rigviz.py` | **自车 + 传感器标定配置图**(纯值,PIL;见 Plan2.md §P-M.9):`draw_rig_layout` 一页三区 = 俯视(车体实测包围盒 + 逐相机挂点与视锥 + 短码)+ 方位环(重叠橙、盲区红带度数)+ 数字表(`通道 · 挂点 x,y,z · 方位角 · FoV · **az ± fov/2**`)。`azimuth_of` 是方位角算式的**第二份独立实现**(单测钉它与 `camera_rig.camera_azimuth_nus` 相等)。**分区是硬坐标**——图例压锥 / 方位环压表都踩过。接受**显式 calibs/fov 参数**,故"还没采过的候选 rig"只改常量就能出图。俯视图另有**后轴标记线**(洋红 = nus 原点)+ **空心灰圈 = 修正前挂点**(整体后移 1.2563 m 落在后轴线上,§P-M.10) |
| `calib_multilidar.py` | 多雷达标定判据评估(注入已知误差 → 判据数值) |
| `viz_layout_cmp.py` | 相机布局对照数值化(同镜头两布局的可见性对比) |
| `probe_calib.py` | **标定自证探针(七锚 A0–A6)**:spawn 6 RGB + 6 depth + LiDAR + 施工锥 → `outputs/calib_check/{report.json,overlay.png}`。A0 光轴 vs 官方方位角 / A1 侧别一致性 / A2 实例分割解码(`id = G + 256·B`)/ A3 LiDAR-平面-深度图交叉验证(裁决**像素约定 = corner**)/ A4 轴目标物掩膜质心回归主点 / A5 实挂 vs 规格 / A6 主点锁定 `(w−1)/2`。**判据全数值,不目检** |
| `viz_calib_check.py` | **标定修正的人工复核图**(`probe_calib` 的数值结论 → 人能对着看的图,判据数字烧进画面):①`check_geometry.png`(纯值,不依赖 CARLA,先落盘)——官方方位角极坐标轮(实线=修正后 / 淡线=历史字面值)+ 镜像差表(**不 wrap**,217.2° 折成 142.8° 就看不出镜像)+ A4/A3 读数;②`check_raw.png`/`check_overlay.png`(A3 六相机**同一帧**的 raw 与残差 overlay,两张逐像素差 = 画上去的点数);③`check_rig_ab.png`(同一 ego、同一批施工锥,`nuscenes` vs `legacy` 各拍一遍 → **世界左方的锥出现在哪一路**即镜像的直接证据)。落 `outputs/calib_check/check_*.png` + `viz_summary.json` |
| `verify_nus_calib.py` | **`collect_nus` 验收判据的复现器**(全数值,不目检;落 `outputs/nus_calib_check/report.json`)。`--offline`(不需 CARLA):③ 雷达点落进**自身 FOV** 占比(以官方 R 为锚 —— 点云在传感器自身系,+x 即光轴,不转到 ego 系再比就是恒真)、④ LiDAR 复现 `num_lidar_pts`(官方集 1.0000 vs 单位阵消融 0.1684)、⑤ 相机内参 vs 该 rig 声明表;`--live`:① 相机实挂 vs 声明(`live_common.mount_deviation_of`,**必须先 tick**)、② 雷达实挂 vs 官方 az、⑥ 渲染 FOV vs 蓝图 fov(复用 `probe_calib` 的锥体/掩膜回归,**不另写实例分割解码**)、⑦⑧ 转 `rig_check.instance_probe`。**⑥ 两个实测坑**:每 tick 必须**抽干全部相机队列**(否则未测相机积压陈旧帧 ⇒ 掩膜恒 0,症状是"只有第一个相机测得出"且与距离无关)、Z 取**阶梯** `(20,14,10,8,6)`(地图遮挡随出生点变,写死会假失败)。**明令禁止 `num_radar_pts` 作判据**(跨 5 通道求和,官方 R 也只复现 0.6279)。**+ ⑨⑩(§P-M.10)**:⑨ 世界系链(`declared = 落盘表 ⊕ **实测**后轴位姿` vs `rendered = CARLA 实挂经共轭`,12 路逐位比 —— ①② 相对同一个 ego,**对原点误差在结构上盲**,必须有它);⑩ 独立复测后轴(双偏航自解,**不读常量**,防常量腐化静默错位)。`--rig {nuscenes,wide}`:③④ 与相机无关原样适用,其余换该 rig 的声明表;默认落点按 rig 分叉(`nus_mini[_wide]` / `report[_wide].json`,**不互相覆盖**) |
| `rig_check.py` | **判据 ⑦⑧ 的唯一执行点**(`RigCameras` 按任一 rig 的**声明位姿/FoV** spawn 6 路 instance_seg,1600×900):⑦ 逐像素数 **ego 自己的 actor id**(对照:**修正前**官方 rig `CAM_BACK 619189 px = 42.9992%`;§P-M.10 原点修正后**两代 rig 六路全 0 px** ⇒ 该判据不再是 wide 的区分度,wide 的取舍是 FoV spec 换覆盖率);⑧ 往重叠区正中摆施工锥,两路掩膜都命中才算共视。**两条边界写死在模块里**:锥心抬 `COVIS_Z_LIFT=0.45 m`(否则擦车顶被自车挡,读数变成"有没有被自车挡")、几何重叠 < `MIN_COVIS_OVERLAP_DEG=5°` 的对**如实 `skipped`**。★ `common_band()` 是核心订正:**方位轴重叠(无穷远)≠ 有限距离下的共同可见**(挂点视差最多 1.7°,官方 `FL↔BL` 11.20° → 12 m 处 4.838°) |
| `viz_rig_check.py` | **rig 的两件目检交付物**(`--rig`,`--live` 需 CARLA;落 `outputs/calib_check/`):`rig_layout_{rig}.png`(纯值配置图,`rigviz.draw_rig_layout`)、`views_{rig}.png`(六视角**原生像素**拼图 + 逐格 `az ± fov/2` 与 ego 像素读数 + 底部**线性方位尺**:逐相机一条泳道 ⇒ 竖线穿过的行数 = 该方位被几路覆盖;ego 像素**就地染品红**,0 px 时是空操作)、`report_{rig}.json`。**图不能替代 `verify_nus_calib`**:"声明 ≠ 渲染"在图上看不见(§P-M.7)。两个踩坑:`360.0 % 360 == 0` 会让 `rectangle` 抛 `ValueError`(跨 0° 扇区必须钳,不取模)、`np.asarray(PIL)` 是只读视图(染色必须 `np.array` 拷贝再 `Image.fromarray`) |
| `probe_rig_cmp.py` | **多 rig 数值对照 + 自证**(2026-10-01):同一个 ego 上三套相机口径(`nuscenes`/`wide`/`nucarla`)逐通道列 fov/fx/cx + 方位覆盖/盲区,并**跑该 rig 的独立复算**报 `max|Δ|`(nuCarla 走四元数链对矩阵链,实测 **0.00e+00**)。落 `.json` + `.md` |

### `map/` — 地图矢量 + MapTR(19)

| 文件 | 职责 |
|---|---|
| `opendrive.py` | OpenDRIVE 1.4 解析(planView/lanes/objects/signals/junction)。`_geo_at` 有 `strict` 档:默认对 `s ∉ [0,length]` raise(**抓调用方 bug** 的守卫,不是物理约束);`strict=False` 才**有界外推**(上限 10 m) |
| `stitch.py` | **跨图拼接**(§P-M.16):`Placement` 表 + `place()`(刚体,**恒等原样返回**)+ `stitch()`(并集 / 多图加 id 前缀 / **跨图**去重)+ `seams()`(接缝候选**报告**)。⚠️ 官方 Town 无真值相对位姿,placement 是**人为摆位**;⚠️ `MapVec` 不带道路图 ⇒ 接不了 link;⚠️ 合并图**对 MapTR 训练无用**(训练仍逐帧 `BEV_RANGE` 裁剪)。两条实测逼出来的判据:**同图内部不去重**(不同 road 的中心线会恰好重合)、**单点要素(信号灯)必须参与去重** |
| `stitch_temporal.py` | **时序拼接**:逐帧 BEV 矢量预测按位姿拼成全局矢量图。与 `stitch.py`(跨**图**、人为摆位)分工不同,它跨**帧**、位姿来自轨迹(`--pose gt` 上界 / `--pose slam` 车载口径)。几何层复用 `stitch.place`(正好 = ego→world)与 `_dedup`;**融合**另有 `--fusion cluster`(Chamfer + 朝向门 + 簇内平均)—— 实测 `_dedup` 的 Hausdorff 对跨帧是错配(tol 0.5 m 只合掉 0.7%)。**按 seg 分段是硬约束**(段缝位移 56.8–109.6 m)。`--which {pred,gt}` 决定拼预测还是同帧 GT(⚠️ GT 是逐帧窗口裁剪后的并集,不是完整地图矢量)。`--lidar-root` 给世界系点云底图(**必须与 `--infos` 同一段数据**,且用**同一个 `--pose` 源** —— 一边 GT 位姿一边 SLAM 位姿会得到两张各自都对、叠起来错位的图);点数上限 `--base-max-points` 超限按均匀步长抽样,**丢弃量写进 JSON 与图上标题**(不静默截断)。产物 **JSON + PNG 两份** |
| `lanelet2.py` | **Lanelet2(.osm XML)适配器,双向**(§P-M.15):**标准 OSM 结构**(`<node id lat lon ele>` + `<way><nd ref>`;早先把坐标内联进 `<nd>`,真实 reader 会**读成空图**)+ **高程走 `ele`** + `autodrivedata:seq` 还原顺序;六类要素→`way`/`node`(标线级,`divider`→`line_thin`、`boundary`→`line_thick`、`ped_crossing`→`crosswalk`…);坐标系是**米**而 .osm 存 lat/lon ⇒ **等距圆柱投影**,`origin` 默认取质心并**写进文件注释**(外部工具忽略、自己读得回 ⇒ 往返不依赖文件外的隐式约定)。保真走自定义 tag `autodrivedata:cls`/`:id`/`:src` + `attrs:<key>`(**重复键**加 `:N` 后缀);读回**优先自定义 tag、回退语义标签** ⇒ 也能尽力读第三方。**只到标线级**;升级到真 lanelet 需要什么见模块头注 |
| `apollo.py` | **Apollo HD Map(text-format protobuf)适配器,双向**(§P-M.15):`ped_crossing`/`stop_line`/`traffic_light` → `crosswalk`/`stop_line`/`signal`(**天然对应**);⛔ **`divider`/`boundary`/`centerline` 在 Apollo `Map` 里没有落点** ⇒ 降级为借 `lane.central_curve` 承载(**把标线当车道**),类名编进 `id.id`。保真 ≈ 无 KV 槽 ⇒ `attrs`/`id`/`src`/顺序全编进 `id.id`(保留键 `@id`/`@src` 带 `@`,与 quote 后的 attr 键不可能撞名)。走 text-format 是因为**本机无 `.proto`**、免 protoc。升级到真 lane 需要什么见模块头注 |
| `mapvec.py` | 地图矢量 GT 提取与采样(MapTR 口径:六类要素 + 裁剪 + 重采样)。**object 轮廓走非严格外推**(斑马线/停车线摆在路段起止处会合法溢到相邻路段,实测最多 2.44 m)⇒ 修前 Town03/04/05/06 **整张图提不出来**(12/20),现 **20/20**;越界实例在 attrs 里带 `s_extrapolated`,未越界的**不加该键**(⇒ 本来就好的图产出逐位不变) |
| `mapvec_schema.py` | 矢量预测对外契约 `mapvec_pred/1`(schema 校验/JSON 往返)。`map_format` 是**可选加字段**(缺省 `opendrive`)⇒ 旧文件照收,**不 bump 版本号**(§P-M.15) |
| `mapviz.py` | 矢量投影与绘制:ego 系折线 → 相机像素 + BEV 面板。`intrinsics_from_k` **直读 K 的 cx/cy**(不重算);`calib_from_fov` 是**全仓唯一 fov→fx 落点**(主点 = 索引约定中心 `(w−1)/2`) |
| `bev_base.py` | **BEV 底图**:同段序列的 LiDAR/Radar 点云按位姿投到**与矢量同一平面系**,叠在图下当上下文。⚠️ **不是多模态融合** —— 模型纯相机,点云不进网络(头注首段)。三层坐标系显式化(`B`=CARLA actor 原点 / velodyne / nus sensor),换算全走**已有常量**(`slam_eval.LIDAR_LEVER`、`NUS_RADAR_OFFSETS`),两条链各有**对表自证**(`tests/map/test_bev_base.py`);分层阈值 `OBJECT_Z_MIN = 0.4` 由实测 z 直方图的天然谷定(路面 38140 点 z∈[0,0.2),谷在 0.2–0.6)。`Px`(批量仿射)定义在 `mapviz`(依赖序最低) |
| `chamfer_ap.py` | MapTR 评估纯值:Chamfer 距离匹配 + 多阈值 AP(官方口径) |
| `assemble_maptr.py` | 环视采集 + 地图矢量 → MapTRv2 infos 同构 json。`--surround`(单段,旧用法)/ `--segs-dir`(多段,收 `seg*`)**二选一**;多段按帧带 `seg` / `frame_in_seg`,`token` = `{seg}_{i:06d}`(**全局唯一**,`--out-frames` 拿它做文件名)。`roots_and_prefixes()` 决定 `data_path` 前缀:单段**空**(与旧产物逐帧等价,实测只有 3 个键变)、多段 `segK/`(§P-M.12)。**`--map-json auto`(2026-10-03,双图池)**:矢量 json **逐段**由该段 `calib.json["map"]` 推 `{--map-dir}/{basename}_full.json` —— 原来只吃**一个**文件,两张图的段进不了同一个 infos。`map_label()` 认**三种形态**(`Town13` / `Carla/Maps/Town10HD_Opt` / 平铺大图的 `Carla/Maps/Town13/Town13`);矢量 json **按路径缓存**(412 MB 的 Town13 逐段重读会很慢)。⚠️ **显式给文件时逐字节不变**。★ **`assert_same_rig`**:`MapTRDataset` 是 `self.calibs = infos[0]["cams"]`(**从首帧取一份给全池共用**),混 rig **不抛异常**、只让训练学不动 ⇒ 组装收尾逐段比 `(sensor2ego, intrinsic)`、不等就停并点名。⚠️ **不能整份 `cams` 比** —— `data_path` 带段名前缀(`seg0/…` vs `seg1/…`)逐段必然不同,那是设计不是错(本守卫第一版就是这么错的,被单测当场抓到) |
| `merge_train_infos.py` | 拼接多组环视训练数据 → 合并 infos(帧号连续重排) |
| `convert_mapvec.py` | 地图矢量 → MapTRv2 annotation 口径 |
| `export_mapvec.py` | 地图矢量导出全量/帧级裁剪 json + BEV overlay。`--lanelet2`/`--apollo` **追加**写该格式(**互斥**);`--from {opendrive,lanelet2,apollo}` 反向读回当 GT 源。**默认路径逐位不变**(`vecs_dump` 没动,实测 sha256 与改动前相同) |
| `prepare_official_dataset.py` | 环视数据 → 官方栈可直吃的 nuScenes 形状数据集(已终止线) |
| `train_maptr.py` | MapTR 训练入口(单帧过拟合 = 正确性锚点;多帧 = 常规训练)。`--seg` / `--exclude-seg` / `--keep-in-seg` 走 `select_frames`(`--frames 0` = 筛后不截断);`--temporal-window K` = MapTRv2 时序版(与 `eval_maptr.py` **必须同值**),训练集丢弃数经 `ds.dropped` 上报;**长训一律 `--lr-halve 0`** —— 默认 12 会让 128 ep 后半程 lr 归零,平台是 lr 死掉不是收敛;过拟合闸门由 `is_single_frame_anchor(n_samples)` 判定,**按实际训练样本数而不是 `--frames` 标志**(`--frames 0` 是"不截断",按标志判会把 400 帧训练误判成单帧锚点并打假 FAIL + 退出码 1,见 §P-M.12) · 落 `logs/` 三件套 |。**逐类留出 AP 落时间序列**(2026-09-30):`holdout_class_tags` 把 `evaluate()` 已经在算的 `aps`/`n_pred`/`n_gt` 逐类写成 jsonl 标签 `confirm_AP/<类>` 等 —— `rl.highlight` 是**末次覆盖**(`_highlights[k]=v`),逐类 AP 每次复核都盖掉上一次、时间序列整个丢失,不分类就看不出来「一类到顶、另一类还在涨」互相抵消。**收尾顺序(2026-09-30 修)**:best-AP 恢复**必须是最后一次写 `--out`**。恢复块之后不许再有 `save_map_checkpoint`/`_save_opt_sidecar`,权重类 `rl.artifact` 也必须排在恢复之后(它**调用时立刻 sha256**)。修前的 bug:恢复后被无条件存盘盖回触发时刻的权重,日志却照写「已恢复」(Plan2 §P-M.20);判据见 `tests/map/test_maptr_select.py::TestCheckpointFinalizeOrder`。**回评对照(2026-09-30)**:收尾那次 `_auto_eval` 评的正是**恢复后的 `--out`**,于是拿它的 mAP 与日志声称的 `best_ap` 比对(`_restore_mismatch`,容差 2e-3)—— **零额外 GPU 成本**就把「产物与自述不一致」变成自动检查;不一致则响亮报错并 `exit 1`(权重已在盘上,raise 不丢东西)。三种「判不了」各自落 `restore_verified` 的三个取值,**「没验」与「验过且通过」必须可区分**
| `eval_maptr.py` | MapTR 评估:权重 → 逐帧推理 → 四类 chamfer AP(+ 逐帧契约落盘)。选择器与 `train_maptr` **同一份实现**(`select_frames`),`--start` / `--frames` 在**过滤后**的列表上再截(§P-M.12) · 落 `logs/` 三件套 |
| `eval_official_metric.py` | A′ 口径复算:并排算"自实现 chamfer AP"与"官方 eval_map" · 落 `logs/` 三件套 |
| `viz_maptr_pred.py` | 预测回投目检:预测/GT 折线 → 6 相机 overlay + BEV 面板。`--bev-pair` 额外落 `bev/frame_XXXX_{pred,gt}.png`(拼图里两者叠加,分开才看得出「谁多谁少」);`--lidar-root` 给每帧 BEV 加 LiDAR/Radar 底图(同帧 ego 系,零点是**判据**) · 落 `logs/` 三件套 |
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
| `fusion.py` | **后融合的纯值核心**(§P-V18):LiDAR 簇(**velodyne 系 AABB**)→ **相机系 `Box7`**、尺寸门、2D 关联(贪心一对一,与 `compare.match_boxes` 同口径)、两档 `fuse`。★ 三个口径错了都**不报错**:① **`ry` 恒 `−π/2`**(`label_2` 实测四台车全 −1.57;照搬 0 ⇒ 长轴横过来、IoU 掉到 0.2);② **`l` 取 velo 的 x 跨度**;③ **`Box7.y` 取底部**(velo 的 `cz − half_z`)。★ **尺寸门按"LiDAR 真能测到的量"给,不按 GT 的 `(h,w,l)`** —— 首版照 GT 给(`l ∈ (2.8,5.8)`),而 `l` 是**车长**、LiDAR 只看得见车尾 ⇒ 实测簇 `l=1.86–1.89` **全被挡**,召回 **10%**,形态像"点云不行"。改用高度 / 最大水平跨度 / 最小水平跨度,去掉长宽比(车尾迎面时 `l≈w` 是遮挡不是形状)。**雷达不在此表**:平台边界已实测(§P-V17 六) |
| `eval_fusion.py` | **融合判据**(§P-V18):LiDAR 簇(RANSAC 地面 + DBSCAN)+ YOLO26 相机检出 → 两档融合表 → 与 `label_2` 比 **3D AP@0.5 + 逐条 `TP/FP/FN`**。**必须报计数**:两档 `conf` 来源不同(簇点数 vs 相机置信度),只比 AP 会把"排序变了"读成"检测变好了"。⚠️ `cluster.dbscan(distance_scale≠0)` 会切到**纯 Python 网格实现** —— 实测同帧 RANSAC **0.08 s** 而整个聚类 **129.7 s**(20 帧 = 43 min,表观挂死);默认走 sklearn(KD-tree)。⚠️ 消融口径:GT **100% 是 `Car`** ⇒ 「相机出类」测不出东西,改量**精确率/召回各自的增减**。`--gates loose,strict` 出 **2×2**(门×模态)——「严门单独一档」是「相机有没有独立贡献」的对照。★ 结果:**P1 静置车数据上融合没赢**(`strict/lidar` 0.0462 > `loose/lidar+cam` 0.0202);补上缺的两点(多类 + 近距)后**翻过来**(0.0758 → **0.1742**) |
| `radar_eval.py` | **雷达速度对表**(§P-V19,纯值零 carla):**不需要目标** —— 静止场景里每个回波的径向速度必然等于自车速度在该视线上的投影,所以它能在「回波 0 命中」的情况下照样给结论。口径两块都踩坑才定:① **不假设自车速度沿传感器自己的 x 轴**(那只对前置雷达成立;±90° 侧雷达的视线与前进方向垂直)⇒ 同时拟合 `vel ≈ s·(k_x·u_x + k_y·u_y)`;② **R² 必须中心化**(雷达 FOV ±38° + 车速恒定 ⇒ `v`/`e` 都是近似常数,绕原点的 R² 被**均值**主导、恒 0.999,**连打乱对照都塌不下来**)。判据三条:`|k|≈1` ∧ `R²高且对照塌` ∧ 朝向对表(**允许一个全局符号**)。**实测 3/5 达标**(FRONT / BACK_LEFT / BACK_RIGHT);两个**前侧**雷达的速度读数是 0 |
| `compare.py` | 伪标签 vs GT 比对层:3D IoU 匹配 → 分歧帧 → AP/复核率报表 |
| `attribution.py` | 失效归因:逐帧 2D 匹配 → 漏检按距离/框高/TTC 分箱 |
| `radar.py` | CARLA radar 原始检测 → nuScenes 18 字段雷达点云 |
| `mono_depth.py` | 单目测距:检测框 → 地平面投影距离 + 迭代深度法 |
| `stereo.py` | 双目立体视觉:三角测量 / SGBM 视差 / NCC 匹配 / 自监督损失 |
| `multilidar.py` | 多雷达标定:point-to-plane ICP + overlap/plausible 判据 |
| `ground.py` | 点云地面提取:RANSAC 平面拟合 + 网格法双路线 |
| `cluster.py` | 点云聚类障碍物检测:欧氏聚类 + 3D 包围盒 |
| `finetune_synth.py` | 合成 KITTI → pointpillars_kitti 微调(复用 AutoLabel train3d) · 落 `logs/` 三件套 |
| `eval_2d_ab.py` | P1 逆光 A/B:冻结**后端**在两个 KITTI root 的 2D AP 对比 · 落 `logs/` 三件套。`--backend {sam3,yolo}` **默认 sam3**(见 `backends.py`)。⚠️ **归档矩阵(0.669 / −0.578 …)是 yolo 口径**;换后端后四个 Δ 全落进 ±0.02(§P-V23) |
| `eval_attr.py` | 失效归因评估:多跑 × 距离/框高/TTC 网格 + 漏检画像 · 落 `logs/` 三件套 |
| `eval_kitti.py` | GT vs AutoLabel 伪标签比对报表(比对层 CLI) · 落 `logs/` 三件套 |
| `mono_distance.py` | 单目测距评估(检测框 → 距离,与 KITTI GT 真距对照) · 落 `logs/` 三件套 |
| `extract_ground.py` | 地面提取(逐帧点云 → 地面/非地面分离 + 统计) |
| `cluster_obstacles.py` | 聚类障碍物检测(地面分割 → 聚类 → 3D bbox) · 落 `logs/` 三件套 |
| `sem_bev.py` | 语义 BEV:图像 → YOLOPv2 + **障碍物那一路的后端(默认 SAM3,`--backend`)** → BEV 鸟瞰 · 落 `logs/` 三件套。掩膜计算(`predict_masks`/`yolopv2_predict`/`yolo11_object_mask`)与投影(`mask_to_bev`)都抽成**可复用纯函数** —— 判据 `sem_eval` 吃**同一份预测、同一套投影**,不各跑一遍。`mask_to_bev` 已**向量化**(判据规模 7200 万像素,逐像素调 `ground_intersection` 跑不动),与标量版**逐位等价**有对拍钉 |
| `sem_tags.py` | **CARLA 语义相机的 tag 表 / 调色板 / 三类映射**(纯值,零 carla)。tag 编在**R 通道**(`B=G=0`)、编号 = `carla.CityObjectLabel`(⚠️ 不是旧版的 `6=RoadLine/7=Road`);`drivable←Roads` / `lane←RoadLines` / `obstacle←Pedestrians+Car+Truck+Bus`,**两侧都不算** `Rider/Motorcycle/Bicycle/Train`(预测器产不出)。灰度 PNG 编解码(**只认 `mode="L"`**,上色预览一律报错)。编号由 `tests/sim/test_sem_tags_oracle.py` 用 carla 当 oracle 每跑必验,含"防抄成旧版"的反向对照。**`PROP_TAGS`(§P-V14)**:地图自带静态道具的 tag 集,只含**实测过**的 `Dynamic`(探针把三个已知资产摆到镜头前,3/3 排他)—— `Static`/`Other` 各有几千像素但**没有已知资产能归因**,不许顺手并进来。`excluded_share` 因此有**三个**桶,第三桶 `map_prop` 此前**就在那儿但没人报**("没报"与"没有"在下游长得一样) |
| `backends.py` | **检测/分割的后端抽象**(2026-10-02):`Predictor`(`给一张图 → [(项目类名, conf, xyxy)]`)+ `YoloPredictor`(闭集,**回退**)+ `Sam3Predictor`(开放词表,**默认**)+ `make_predictor` / `describe` / `norm_cls`。★ **`DEFAULT_CONF` / `resolve_conf`:两个后端的默认阈值不是一回事**(sam3 0.5 / yolo 0.25),各入口 `--conf` 默认 `None` 再解析 —— **显式传 0.25 与不传必须分得开**。抽出来是因为**「类名从哪来」在两个后端下语义不同**(yolo = 模型判的;sam3 = **提示词**给的)—— 写在四个入口里必然漂,而漂了的表现是「AP 里那个分类正确率悄悄不见了」 |
| `sam3_backend.py` | **SAM3 推理后端**(2026-10-02 起是**默认**):`detect` / `segment` / `load`(模块级缓存,3.15 GiB 常驻、加载 4.6 s)。★ **每条提示一次前向是硬约束** —— SAM3 吃**单概念**提示,拼串实测返回 **0 个掩膜**(`"car person bicycle"` 与 `"car, person, bicycle"` 都试过)⇒ 多概念只能逐条前向再并集,代价线性(单条 0.5–1.0 s)。`DETECT_PROMPTS`(检测,`Car` 含 truck/bus)/ `THING_PROMPTS`(实例的 things 并集)是**类别从哪来的唯一出处**。★ **`dedup` / `dedup_dets` 默认开**:一条提示一次前向 ⇒ 同一物体被两条提示各切一次,判据一对一匹配就把多的记成 FP(实测 mask AP +0.063)。★ `DEFAULT_THRESHOLD` 是**按指标定标的操作点**(PQ 涨到 0.7 而 mask AP 在 0.5 见顶)⇒ **报数必须带阈值**。⚠️ 依赖 `transformers`,装法只走清华源(aliyun 403 / pypi.org 超时,实测) |
| `props.py` | **静态道具 GT**(纯值,零 carla,§P-V14):`static.prop.*` → 归一类别(`cone/barrier/barrel/sign/prop`,**认不出落 `prop`、不返回 None**)、nuScenes 官方类名(补上 `traffic_cone`/`barrier` —— 原先非 vehicle 一律 `None`,把锥桶记成"官方忽略类"是把 23 类表读错了)、`PropBox`/`PropFrame`(度口径 + **yaw=0 探针测的尺寸** + `instance_id` + 每帧相机位姿与内参)。**与 `label_2` 的关系是"不进"** —— 进去会破 P1 A/B 帧级配对硬门槛。⚠️ 尺寸存的是 **yaw=0 那一读**:转过之后读回的是被剪切的值(见 `probe_static_prop_gt.py` 行) |
| `static_eval.py` | **静态 GT(信号/标志 + 车道线)的判据**(离线纯值,零 carla,§P-V21):读 `static_gt/*.json`(带相机位姿)+ `static_sem/*.png`。★ **形状是量出来的** —— 点投影必错(信号锚点是**地面**点,40/40 落在 `Roads`;车道线只 ~2 px 宽,单像素 21.7%/27.8%),故用「锚定 + 邻域」:信号 = 锚点向上 **6 m** 竖带(±1 px),车道线 = ±**4 px** 方窗。报**真命中 / 横移对照 / 同 v 随机基准**三个数;裁决两条(命中率下限 + `real ≥ 5×max(对照)`,少了第二条「到处都是标线」会被读成「xodr 对得上」)。`n=0` 的组是**未判**(第三种裁决);`off_frame` 拆**车后 / 视场外**两栏。`--self-test` 五条自证(窗高按解析式核对、带外 1 px 不计、`real==control` 必须不过、无位姿必须抛、横向按该点自己的切向) |
| `prop_eval.py` | **道具 GT 的判据**(离线纯值,零 carla,§P-V14):读 `static_prop_gt/*.json` + 实例 id 图 + 语义 tag 图,**独立重算** `coverage`(轮廓像素落在投影框内的比例)/`tightness`/`IoU`,**不信采集器自报的数**。自证:平移投影框 40 px 后 coverage 必须下降(**且只在"可判"样本上做** —— 裁断样本的框巨到 ±1e17,平移仍全覆盖 ⇒ 假失败)。**被画幅裁断的样本单独计数、不并入裁决**;`never_seen` 捕获"GT 与 id 图配错帧"。离线可跑是刻意的:判据住在 `perception/`(层规则禁 carla),相机位姿与内参**落进 GT 文件**才让它不依赖 `carla_common` |
| `domain_gap.py` | ★★ **域差有多大**(§1.15):**2×2 迁移矩阵**(两个模型 × 两个域)在**检测 AP 的单位下**量域差。实测**不对称** —— 真实训练的模型往合成域走**不掉**(Car 0.774→0.873);★ **合成训练的往真实域走塌**(Car AP 0.647→**0.135**、召回 0.705→**0.140**)。★ 第一版的对照(打乱图与 GT 的配对)被**自己的读数**打红 —— `ap_for` 池化 ⇒ 重排是**恒等**,改成**平移 GT 框 1/3 画幅**才立住(0.020/0.001)。⚠️ CARLA 侧只有 `Car` 有 GT,**逐类读**;两个模型都是 yolo 家族才同 `conf` 可比 |
| `domain_adapt.py` | ★ **域自适应:先试最笨的那条**(§1.16)。四臂(原图 / 对齐到目标域 / 对齐到错目标 / 参考抖动)。实测 **align −0.0516 ≈ 抖动地板 −0.0502** ⇒ **training-free 的光度对齐不但没用还有害**。⚠️ 两处自己写反:① 裁决只比 `|gain|>floor` ⇒ 把 **−0.0516** 读成「有用」;② 守卫写成 `abs(ctrl−mAP)>0.5` ⇒ **尺子正常/坏掉都不触发,从来没生效过**。⚠️ 检测**落盘成文件**再喂 —— `ultralytics.predict(ndarray)` 走 **BGR** |
| `probe_bev_depth.py` | ★ **BEV 深度投影的三条对照**(§P-V26)。① 接地类两条路 IoU **0.574**;② 换错深度后 **0.000**(深度**真的在链路里**);③ 障碍物**离地 1.28 m**(★ 最硬 —— 地平面假设按定义给 0)。⚠️ 第一版对照②算出 **nan** 而 `nan > 0.3` 是 **False** ⇒ **静默判「全过」**,已改成先剔 nan + `None` 单独算**未判** |
| `size_prior.py` | **按类尺寸先验**给 LiDAR 簇补全长(§P-V27)。★ 两条纪律:① **不许与评测同源**(`label_2` 本来就有真值尺寸,同 root 量再评 = 抄答案)⇒ 先验记 `source_root`,`eval_fusion` 同源**当场 SystemExit**;② **要有错先验对照** —— ⚠️ 实测单类时「打乱类到尺寸的配对」是**恒等**,退化成**打乱分量**(`l↔w`)。⚠️ 结论:**先验没帮上忙**(AP 0.014→0.028 而 **TP 10→2**) |
| `inst_tags.py` | **实例分割的纯值核心**(零 carla,§P-V15):**16 位灰度** id 图编解码(只认 `mode="I;16"`;**越界抛异常不截断** —— 截断会把两个物体并成一个,症状是"实例数少了几个",与"本来就没那么多"同形)、`instances_from_maps`(按 id 分组,**不拆连通域** —— 被遮挡切断的一块仍属同一实例,拆了会把实例数**虚高**)、`THING_TAGS` / `is_thing`。★ 两条实测口径:① CARLA 给**每个关卡网格**都发 id(首采 20 帧 `Roads` 113 个 id vs `Car` 13 个)⇒ 不筛则 PQ 分母全是路面;② **实例相机的 R 通道 == 语义相机的 tag**,`class_channel_matches_semantic` 是这条的可回归落点(采集器首帧自证 1.000000 × 6 路)。`drivable`/`lane` 是 stuff,没有实例语义 —— **两套判据分母不同是角色不同,不是不一致** |
| `inst_eval.py` | **实例分割判据**(§P-V15):逐实例贪心匹配(与 `compare.match_boxes` 同口径)→ **mask AP** + **PQ/SQ/RQ**。`--backend {sam3,yolo}` **默认 sam3**(SAM3 要**逐概念前向再并集**,见 `sam3_backend`)。⚠️ **`yolo` 分支跑的是现成的 `weights/yolo11s-seg.pt`(通用 COCO 权重,不是本项目训练的)** —— 这条链通的是「实例 GT + 判据」这个**评测口径**,不是一个训好的分割模型;引用 PQ 必须连这句一起说。`PQ = SQ × RQ` 是**定义性质**,被当成自证(对不上就是实现错)。★ **PQ 与 mask AP 的差别不是"多一个语义项"**(两个都要求类匹配),而是 ① AP 对掩膜质量不敏感(一个 IoU=1.00 与一个 IoU=0.55 的预测在 `AP@0.5` 下都算命中)② AP 是排序型吃 conf,PQ 不是。`--self-test` 不出模型(GT 当预测 ⇒ 三个数必须恰好 1.0)。类级 `sem_eval` **保留不动**,两者并列 |
| `sem_eval.py` | **P2-A 判据**(§P-V13):`sem_*/` GT vs `sem_bev` 预测,**图像空间 mIoU + BEV 空间逐类 IoU** 双口径。自证:① GT 当预测 ⇒ mIoU 必须恰好 1.0(顺带钉键名对齐、**先挡 nan** —— `abs(nan-1)>ε` 恒 False,不挡就会静默跳过);② **几何** —— `Sky` 的像素射线打不到地面、`Roads` 几乎全打到(全局汇总,**不平均每路比值**)。空类被 mIoU 跳过时会**单独打印警告** |

### `gt/` — GT 生成(3 + `export/` 子包)

`core.py` = 原 `gt.py`;纯值。`export/` 从包根移入。

| 文件 | 职责 |
|---|---|
| `core.py`(原 `gt.py`) | CARLA actor → KITTI label_2 GT 行。**2D 框是两句配套的口径**(§P-V20):① 框 = 全 front 角点 min/max **再钳到画幅**(KITTI 口径,`box2d_from_projection` —— 出框侧与重算侧**唯一实现**);② **退化剔除**看**未钳**的 `inside` 角点跨度(**`MIN_BOX_SIDE_PX = 1.0`**,§P-V12):车**擦过镜头**时会投出**零面积**框,它与任何预测的 IoU 恒为 0(白送一次漏检,P1 每份数据 11/194,recall 天花板被压到 94.3%),且进出由**亚帧抖动**决定(A/B 的 GT 逐帧条数会不等)。**只做①不做②** ⇒ 钳完是接近满幅的框 ⇒ **凭空造 TP**,比"框太小"更毒 |
| `export/coco2kitti.py` | ★ **COCO val2017 → KITTI 布局**(2026-10-08,只落 `image_2`+`label_2`)。服务**跨仓交付**:兄弟仓 `auto2dlabel` 的 2D 微调/评测只读这两样。★ **两个静默陷阱必须报出来**:① 下游按**框高 ≥25 px** 过滤而**一个字都不打印**(实测 **18.3%** 的框会被丢);② 他们类表里的 `train`/`truck` 在本 root 里**恒为 0 实例**而**照训不报错**。⚠️ **3D 列是占位**(COCO 没有 3D 标注);截断/遮挡也是近似。类表**复用** `backends.COCO_FALLBACK`,不新造第二张。切分钉**两半零重叠**(`split_ids` + `assert_disjoint`) |
| `refilter.py` | **GT 口径迁移**:把已落盘 root 的 `label_2/` 按**当前**判据重筛,旁路(`image_2`/`velodyne`/`calib`/`pose`/**`samples/RADAR_*`**,按谓词 `is_rewritten()` 全量)走**硬链接**(逐字节不变、盘上零增量,每 root 只多 ~284 KB label 文本)。判据与出框侧**同源**(`is_degenerate_gt_line`),所以"筛出来的"就是"重采一份会得到的" —— 而重采是投骰子(退化框的进出由亚帧抖动决定,5 个 root 12 个阈值穿越点几乎必有翻转)。只写新 root,不就地改。**`--clip2d`**(§P-V20)= 一次做完**两个修正**(退化剔除 + 裁断重算,`rebox_line` 按 `P2` + PNG 头画幅离线重算 2D 列,**不重采**);**未裁断的行逐字节不变**(短路,不是"算出来恰好相等");缺 `calib`/缺图**直接抛**,不回落旧口径。默认关 ⇒ 不传时与旧版逐字节相同 |
| `static_gt.py` | 静态目标/道路特征 GT(地图查询源:P2)。`StaticFrame.camera` 是**可选**新增字段(§P-V21):判据离线复现投影链要用,**2026-10-02 之前的归档没有这个键**,`from_json` 读回 `None` ⇒ 判据必须报错而不是拿默认内参硬算 |
| `traffic_light.py` | 交通信号灯状态 GT(动态时序层:状态归一/前向判据/相位查表) |

### `traj/` — 轨迹(2)

| 文件 | 职责 |
|---|---|
| `assemble_traj_pt.py` | CARLA 轨迹 → HiVT TemporalData 组装(纯值) · 落 `logs/` 三件套 |
| `convert_hivt_pt.py` | plain dict → HiVT TemporalData(在 hivt env 跑) · 落 `logs/` 三件套 |

### `gs/` — 3DGS(3)

> 本目录**有重建、有归属,还没有编辑**:`train_3dgs_mini` 出高斯,`attribute_instances` 把每个高斯
> 认到一个 CARLA 实例上(`edit-3dgs-plan.md` 阶段 B)。**许 torch,禁 carla**(层守卫强制)。

| 文件 | 职责 |
|---|---|
| `train_3dgs_mini.py` | 3DGS mini 训练(gsplat 光栅化) · 落 `logs/` 三件套。`--val-frames` **默认自动取 5 帧**(旧默认 `0` 是"psnr_val 算在一张坏帧上"的一半原因);`--cam-convention {carla,legacy}` **2026-10-04 起默认 `carla`** —— 旧口径 `rw = Rz(yaw)@Rx(pitch)` **不是 CARLA 的相机系**(`means*.npy` 不在世界系,归属没法用),`legacy` 只为**复现归档**(§A.1–A.3 的 PSNR 表 / `means*.npy` / `gaussians*.ply`)而留;默认值抽成常量 `DEFAULT_CAM_CONVENTION`(**唯一落点**,判据 `TestDefaultConvention`)。实测判据与量级见 `edit-3dgs-plan.md` §B.4。⚠️ **跑之前先读模块头注的「用法」**:不带 `CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH` 会让 torch 的 JIT 缓存 hash 不命中、**用 nvcc 13.0 重编并静默覆盖规范 `.so`**(包括一次性探针 `python -c "import gsplat"`);`--densify` **默认关,且 2026-10-04 起确认应当关**(carla 口径下两个 seed 一致更差,见 §A.4.1 ②) |
| `cuda_env.py` | `import gsplat` **之前**必须做完的两件事的唯一落点:设 `TORCH_CUDA_ARCH_LIST`(预置值先过一遍 `_get_cuda_arch_flags()`,不可用就换实卡架构)、检查 `CUDA_HOME`(不设时 torch 的扩展缓存 hash 不命中 ⇒ **重编并静默覆盖规范 `.so`**,实测代价 ≈1 h)。★ 取 gsplat **只能走 `ensure_gsplat()`** —— 裸 `import gsplat` 会被 `ruff format --fix`(isort) 挪到本模块之前,顺序就没了(2026-10-04 实测被自动修坏过一次)。AST 守卫 `tests/gs/test_cuda_env.py` |
| `render_gs.py` | **离线渲染器**:加载盘上一份高斯(`means/rots/scales/col/opac` 五件套)+ capture → 渲染指定帧,落 PNG(目检)+ npy(判据用,PNG 是 8 bit 判据不能吃)。`rasterize()`/`mse_to_psnr()` 从训练脚本抽出**共用**(唯一口径);`--verify-json` 复算逐帧 PSNR 与 `train_result*.json` 对账(容差 ≤1e-3 dB)—— **专抓「两套渲染口径」那类不报错的错**。⚠️ 别写成两层:训练时的渲染与编辑后的渲染必须是同一套 |
| `edit_gs.py` | **按实例 id 编辑一份高斯**(目前只有「删」)。★ 两条纪律都对应**删错了但看着像成功**:① `-1`(未归属)一个不许动 —— 删它等于删整片背景,而那渲染出来只是「场景没了」;② **删掉 0 个必须抛** —— 静默成功会让下游把「编辑前后一模一样」读成「这个物体删不掉」。`--prop-json` 从 `capture/prop.json` 读 `instance_id`(编辑对象的**权威来源**),`--list` 看 id 直方图(含未归属占比) |
| `eval_edit.py` | **编辑判据**:**三档 + 分区**。三档 = 不编辑基线 / 编辑后 / **天花板**(B 自己训一份) —— 少一档,"20 dB"无从解读;分区 = 只在**真值掩膜**(A 的实例图 == `instance_id`)内算,全场均值会把窟窿稀释掉。★ **A/B 配对是硬门槛**(位姿最大差 > `1e-6` 直接拒,别放宽成 P1 的 0.07 —— 两者量的不是一回事);**掩膜为空 = 未判**(第三态) |
| `frame_sync.py` | **capture 的帧同步判据**(离线纯值):逐帧 `>50 m` 占比 / 中位深度 / 相邻帧平均绝对差,**首帧必须与其余帧同分布**。第三态:帧数 < 5 ⇒ **未判**。⚠️ "其余"取 **p5–p95 / 中位**而不是 min/max —— 首版用 max 时"2 帧瞬态"恰好顶住上界,**差点放过去** |
| `lidar_ab.py` | **LiDAR 的 A/B 判据**(纯值,禁 carla/torch):**确定性**(位姿差 / 点数)与**「靶有多大」**(世界系最近邻距离 + 点数差)。★ **自带对照**:遮挡物在车前 ⇒ `x<0` 的后方**必然**只有位姿抖动,是该对自身的 floor。★ **不是「逐点相同」** —— 实测 A/B 位姿本就差 **0.0619 m**(对相机可接受、对 LiDAR 是整片平移)。结论见 [edit-pointcloud-plan.md](edit-pointcloud-plan.md) §1/§2 |
| `attribute_instances.py` | **Gaussian → 实例归属**:逐帧投影读 `inst/*.png` 的 id 投票取众数(`min_samples` / `min_share` 不足 ⇒ **未归属 -1**),落 `outputs/3dgs/attr{_tag}.npy`。判据 = **跨视角一致率**(偶数帧/奇数帧各自独立投票)+ 两条对照(**平凡基线** `Σpᵢ²` / **随机置换**)⇒ **裁决用 κ**,不用倍率(基线 0.844 时倍率式子无解,见模块头注)。`--self-test` 用合成夹具(那里的一致率**同时是正确率**);`--shuffle-inst` 是反向自证 |
| `insert_gs.py` | ★ **3D 编辑的「插入」那一半**(2026-10-08):把一个物体的高斯**搬进**另一个场景。★ 本项目**免费的真值靶** —— A(有道具)/B(无道具)**共用同一个世界系与同一串位姿**,所以**搬运 = 直接拼接,不需要配准**。`merge_sets` **五字段一起拼**(只拼 means 会让新来的继承前一批的朝向/颜色,而渲染上只表现为「糊」)。`translate` 只给**控制臂**用 |
| `probe_edit_downstream.py` | ★★ **编辑 → 感知闭环**(JD「确保仿真数据可用于感知模型训练」的落点)。⚠️ **必须用开放词表**:道具是施工围挡,**不在** KITTI 三类、也**不在** COCO 80 ⇒ 闭集检测器**按定义检不出它**,而「编辑前后都没检出」会被读成「编辑没生效」。⇒ 走 `sam3_backend.segment`(直接吃 concepts),**不走 `detect`**(它的 `DETECT_PROMPTS` 只认项目三类、且**故意拒绝调用点现编** —— 那条守卫是对的)。⚠️ **控制臂是硬的**:`insert_gs --shift` 挪开之后检出必须掉。实测 `barrier` 检出 **0 → 4**(框高 42–101 px,**不是「太小没检出」**),控制臂掉到 1 |
| `probe_hole_visibility.py` | ★★ **删掉物体那块窟窿有没有别的视角看得见**(纯 CPU,§C1.2)。用 B 侧真值深度反投影出被遮的背景点(80.9 万),投回全部 270 个视角 ⇒ **中位 104 个视角看得到、0.0% 一个都看不到**;A 侧模型在该区域**高斯覆盖 93.5%**(删除前后**一模一样**)。⇒ **推翻了两条写进文档的说法**。⚠️ 自证必须用**收紧**的门槛(放松 `+2 m` 只让读数从 104 动到 105 —— **那不是对照**) |
| `probe_hole_render.py` | ★ **那块渲不好到底是画错还是画糊**(§C1.2,GPU 秒级)。掩膜内三方距离:编辑后 vs 真值 **14.20** / 天花板 vs 真值 **40.95** / 编辑后 vs 删除前 **12.49**。⇒ **问题在 A 侧的表示**(B 证明可达),而且删完那块**仍更像「道具还在」**。⚠️ 它**把当天刚修的帧索引 bug 又长了一遍**(绕开 `three_tier` 自己写判据);裁决方向也曾写反 |

### `edit/` — 图像/场景编辑(11)

> **新能力面(2026-10-05 用户裁决"成项目模块,放 `edit/`")**。生成模型(StyleGAN3 / ControlNet)
> 的后端与条件源。**许 torch,禁 carla** —— 条件源吃的是盘上**已采好**的数据(真值深度 /
> 语义 tag / RGB),采集是 `sim/` 的事(与 `perception/` 同形:评测层不连仿真器)。
> 两个官方 repo 克隆在 `hdMapGitHub/` 且**保持 pristine** —— 全部版本适配收在 `cldm_backend`。
> 计划与全部实测见 [edit-image-plan.md](edit-image-plan.md)。

| 文件 | 职责 |
|---|---|
| `cldm_backend.py` | ControlNet 官方 repo 的**唯一落点**。**6 处版本适配**(① 传递 import 要 `pytorch_lightning` ② PL 2.x 挪走 `rank_zero_only` ③ torch 2.6 的 `torch.load` 默认 `weights_only` ④ CLIP 文本塔 **196 键** `text_model.` 前缀 ⑤ 官方 demo 隐含要求**方图** ⑥ `MidasDetector` **自己不 resize**,喂 375² 会在 ViT 里报 `a (577) must match b (530)`)+ 权重 **sha256 校验**(边车按 `(size,mtime)` 缓存,边界"能改 mtime 的人能骗过它"**明写成测试**)+ 两个官方条件估计器。★ **缺键必须抛**:宽松加载会把缺的键当随机初始化,**模型照样出图**,CLIP 那一半是白噪声 |
| `depth_cond.py` | **纯值(禁 carla/torch)**:度量深度 → ControlNet 条件图。口径对齐 `annotator/midas/__init__.py` 的 `disparity → 逐图 min-max → ×255`;真值侧补 `1/d`。**三种保序归一化** `{minmax, histeq, match}` —— 它存在的理由是一个**ρ 看不见的**缺陷:`1/d` 把远场压成窄带 ⇒ min-max 出来是**发灰平场**,而 ControlNet 吃**像素值**不是排序。判据 `spearman`(对单调变换不变) |
| `conditioned_gen.py` | CLI:**条件源组装 → 生成 → 条件保真度(带对照臂)**。`--kind {gt-depth,midas-depth,canny}` × `--normalize` × `--fidelity`。★ 头注列了这把尺子的**三条已知边界**(整图 ρ 被共享先验撑满 / ρ 看不见强度分布 / MiDaS 本身在 512² 中心裁上不稳)⇒ "ρ 高既不充分也不必要",主判据用**余量**(自比 − 对照)。⚠️ hint 必须 **3 通道**(官方靠 `HWC3()` 复制),漏了会在 `openaimodel.py` 深处报 `expected input[2, 1, 512, 512] to have 3 channels` |

| `kitti_square.py` | **KITTI root → 方裁 root**(纯值,禁 carla/torch)。落 `image_2`+`label_2`+`pose`+**`depth`(有则按同一仿射裁)**。生成管线吃 512² 方图,而采集帧是 1242×375 ⇒ **图裁了而框没裁不报错,只让下游 AP 崩掉**。`label_2` 的 2D 列(4–7)做同一仿射、**3D 列原样不动**(动它们 = 伪造几何);`clipped` 与 `dropped_out` **分开计数**(裁断样本按 2D 口径红线必须单独分类)。★ **不落 `calib`/`velodyne`** —— 裁剪后主点要平移重缩放,落一份没改的比不落更危险。★ 中心裁起点与生成侧 `to_square_rgb` **同一套取整**(`(w-s)//2`),有机械判据 `test_crop_matches_generator`(差 1 px 不报错,只让 AP 莫名低) |
| `downstream_eval.py` | **下游闭环判据**:四臂(晴 / 真值雾 / 生成雾 / **对照**)配同一套 GT 跑 `perception.eval_2d_ab`,报 **ΔAP 三档**。★ **对照臂是分辨力来源** —— 两版都错过:① 转 GT ⇒ GT 条数变了;② 转图 ⇒ **与 C 臂逐位相同**,因为 `eval_2d_ab` 把框**池化**、配对顺序不起作用 ⇒ 现有版本取**后半程的图配前半程的 GT**(图集合真的不同) |

| `degrade.py` | **人工注入退化**(纯值,禁 carla/torch):运动模糊 / 高斯噪声 / 雨痕 / **全局遮蔽雾** / ★ **距离相关雾**。依据是本仓红线「**CARLA 无运动模糊,退化只能人工注入**」。★ 两条判据钉住:强度 0 **必须逐位恒等**、**seed 逐帧固定**。★★ **`fogdepth` 是本轮最强的一条**:全局遮蔽雾做到亮度 186 只掉 **−0.002**,而**距离相关雾**(用**真值深度** `t=e^(−βd)`)β=0.10 掉 **−0.262**(生成图上 **−0.348**)⇒ **能移动检测器的是"退化随距离增长",不是"雾有多白"** |
| `noise_curve.py` | **「强度 → ΔAP」标定曲线**:给另外两种退化当标尺,`fit_equivalent_level` 把某个 Δ 投影回曲线。★ 实测**在本后端上立不起来**(三种注入 9 档全落在 ±0.011)—— 那是**检测器的性质**,不是工具的错;`--backend yolo` 再跑才有意义 |
| `video_edit.py` | **视频编辑的时序一致性**(JD 明确列了)。判据 = 相邻帧**结构跳变**,两条自证:**与真值序列比而非与 0 比**、**打乱必须报更大**。⚠️ 描述子**刻意不用 MiDaS**(它自己就抖,拿抖尺量抖分不开);两侧**必须先中心方裁**再比(画幅不同比的是构图) |
| `harmonize.py` | **和谐化**:没有真值靶 ⇒ 立**代理**(生成图与真值帧的 LAB 一阶/二阶统计距离),方法只做基线(`Reinhard`),**两条判据**:下降**且**与「对齐到别帧」的对照分开。★ 踩到一个真缺陷:OpenCV 的 LAB **有两套量纲**(uint8 `L∈[0,255]` vs float32 `L∈[0,100]`),混用让 L 通道彻底错。★ 掩膜版(`dilate`/`context_ring`/`reinhard_transfer_masked`/`lab_stats(mask=)`)给 `harmonize_target` 用 —— **掩膜外必须逐位不变**(第一版整幅 `from_lab` 出去,RGB→LAB→RGB 有损,把外面也改了) |
| `harmonize_data.py` | **把 §1.12 的真值靶批量做成训练集**:逐帧逐框切 128² 的 `(composite, truth, mask)` npz。★ **掩膜铺满 patch 的样本丢掉**(`MAX_MASK_SHARE=0.5`)—— 那种样本里**没有"周围"**,连"按上下文对齐"这条基线的定义都没有(实测当场抛 `掩膜内只有 0 个像素`) |
| `train_harmonize.py` | **在小 U-Net 上训和谐化**(残差式;L1 训练 / §1.12 的统计距离评测)。★ **结论是负的且已定位**:① 只训单向光照 ⇒ 反向时**主动帮倒忙**(掩膜均值 0.82→0.99);补上反向后 +1.499 → +0.352 但仍未超基线;② 剩下那部分指向 **`stat_distance` 本身就是 Reinhard 的目标函数** ⇒ 用 L1 训的网络在它的主场比,**输是结构性的**。⇒ 卡点是**判据不是网络**。★ 逐对拆开后更正过:汇总中位**不可读**(四对的 composite 是双峰,极差 3.42×),模型实际**赢 reinhard 2/4、赢不处理 1/4**;而两个留出对各问了一件不同的事 —— 雾对是「抄周围统计」的主场(填周围均值实测离真值 1.085 vs 合成 4.073)、湿对**没有改进空间** |
| `calibrate.py` | **β 标定产线**(§5 #14)。一条命令 = **扫曲线 + 拟合 + 出配方**(`--apply` 直接落产物),并**自动报曲线是否单调**。★★ **默认 `--grid-points 101`**:11 点口径下曲线**不单调**(实测 `+0.0763 / −0.0026 / −0.0013`),拿它内插 = 把尺子的台阶当退化效应(见 `perception/eval_2d_ab` 头注)。★ 落不上曲线时返回 `β=None` —— **那是结论不是失败** |
| `harmonize_target.py` | ★★ **和谐化的真值靶**(§1.12):A/B **同帧跨天气**粘贴 —— 背景/真值 = A 天气第 f 帧,补丁 = B 天气**同一帧**的车辆框区域,真值 = A 帧原样,掩膜 = GT 2D 框。⇒ **精确真值,不是代理**。★ 几何由 A/B 硬门槛保证**且可验**:`box_shift` 实测 70 帧 max **0.31 px**(超 2 px 当场抛)。★ 两条自证:补丁取自真值本身 ⇒ 距离必须 0;**行序颠倒 ⇒ `box_shift` 必须 0**(两家 root 的 `label_2` 行序真的不同,逐行 zip 会读出 215 px 假错位) |
| `augment.py` | ★ **注入式退化 → 训练数据配方**(2026-10-08,纯值)。把「生成的退化不能替代真值退化」这条**结构性负结论**翻成正面交付。**一条命令 = 围绕 β* 铺 N 档 + manifest**。★ 自证两条:`β=0` 那一档必须与 src **逐位相同**(恒等塔基)、**至少一档非恒等**(否则是 no-op)。★ **一档一个 root** —— 塞进同一个 root 会让下游的随机切分产生**近重复泄漏**。⚠️ `fogdepth` 硬前置 `training/depth/` |

| `harmonize_axis.py` | ★★ **和谐化收口判据**(§1.14,纯值零 GPU):拿**不依赖任何假设**的臂问「这一格有没有"只有网络能解"的余地」—— 四臂 = 不处理 / ★**抄周围均值**(最笨的"用上下文") / reinhard / 对照(错光源统计)。实测**四个靶没有一个是只有网络能解的**(三个由统计对齐解完、湿路面无余地)。⚠️ 沿途**三条自造的判据被各自的对照打红**(分块结构比 / 全局仿射解释率 / 稳健版),教训与 §1.10 那版 WSD 同型 |

### `utils/` — 通用件(4)

**准入判据**:无项目领域语义、无 carla/torch 依赖;超过 6 个文件即视为 junk drawer。

| 文件 | 职责 |
|---|---|
| `geometry.py` | 坐标转换唯一落点:CARLA 系 ↔ KITTI 相机系 ↔ nuScenes 系。`CARLA_TO_NUS`(对合)/ `nus_camera_rotation_to_carla`(相机自身系另需 `CARLA_TO_CAM`)/ **`nus_sensor_rotation_to_carla`**(LiDAR/雷达自身系与 nus 同轴序 ⇒ 两侧同阵,对纯 yaw 即 `yaw_carla = −az_nus`,6DoF 连 pitch/roll 一起正确翻过去)。`quat_normalize` 的头注归因订正:非单位来自**手抄 4 位小数**。**挂点原点单点真值**(§P-M.10):`NUS_EGO_ORIGIN_X = -1.2563`(CARLA actor 原点在**车身中点**、nus 官方表在**后轴中心** ⇒ 整套 12 路传感器偏前 1.2563 m)+ `nus_ego_translation` / `carla_actor_origin_to_nus_ego`(**只收 6DoF 三元组**,yaw-only 入口会静默丢掉 0.0642° 悬架俯仰)+ `nus_ego_rotation`(全 6DoF `ego_pose.rotation`;**UE 左手口径 ⇒ 正确分解是 `Rz(−yaw)·Ry(−pitch)·Rx(+roll)`,原样代入会静默反号**)+ `CARLA_CAM_TO_NUS_CAM`(相机**局部**基重排 `[e1,−e2,e0]`,正交但 **det = −1**;与全局基翻转相乘才抵消。拿 `M·R·M` 比相机会得到**恒 120° 的假误差**) |
| `fonts.py` | **覆盖层文本的唯一字体落点**(PIL 唯一依赖,不 import carla):`font_path()` 按「`AUTODRIVEDATA_FONT` → 系统 CJK → **CARLA 随包 `DroidSansFallback.ttf`** → DejaVu 兜底(并 warn)」解析;**能画中文吗 = 渲染探针**(`U+10FFFF` 的像素签名 = 该字体的 `.notdef` 签名,某字签名与它相同即豆腐块;`has_cjk` 要求 `文相机字` 四签名互不相同)——**不看文件名、不看 `fc-list`、不依赖 fontTools**。`sanitize` 把字体缺的码位(`REPLACE` 表,实测只 4 个)换成等价 ASCII、兜底 `?`,**绝不留豆腐块**;`get_font`/`draw_text`/`width`/`bbox`/`wrap`(按**实测像素宽**折行 —— 单行画超画布会被 PIL 静默裁掉,见 §P-M.9)是全部绘制的入口。**两条独立成因都在这解决**:① 本机字体族**一个 CJK 字形都没有**而代码硬写 DejaVu;② **PIL 没有字体回退链**,`ImageDraw.text()` 不传 `font=` 就用内置位图字体(同样无 CJK 且只有 ~11 px)。见 Plan2.md §P-M.8 |
| `paths.py` | 项目路径锚定(`project_path()`:相对路径 = 相对项目根) |
| `runlog.py` | **每次训练/推理的运行留痕唯一落点**(只用 stdlib,不 import carla/torch)。`run(script)` 上下文管理器 → `logs/<能力>_<模块>_<YYYYmmdd-HHMMSS>.{log,jsonl,json}` **三件同 stem**:`.log` = **tee `sys.stdout`/`sys.stderr` 的全量文本**(头块 `script/started/argv/cwd/git(dirty 计数)/python+env/gpu(型号+显存+驱动+CUDA)/host`;尾块 产物表 + highlights;异常写完整 traceback 且**照常抛出**,`SystemExit.code != 0` 记 `status=\"failed\"`)+ `.jsonl` = `metric(step, **kv)` **逐行 flush** 的机读指标 + `.json` = 汇总(env 指纹 / `inputs` / `artifacts`(**≤512 MiB 全量 sha256**,超限只记 `bytes` 并 note)/ highlights / notes / `exit_code`)。另有 `logs/latest/<能力>_<模块>.<ext>` **相对软链**指向最新一次。**三条已踩的坑**:① `_Tee` 必须 `__getattr__` **全量代理**(`isatty`/`fileno`/`encoding` —— tqdm/ultralytics 会直接问,只实现 `write`/`flush` 在非 tty 跑法里炸);② 退出时恢复**构造时抓的那个流对象**而非 `sys.__stdout__`(pytest `capsys` 会替换 `sys.stdout`,写错就把外层捕获**永久**破坏);③ 环境指纹**只在头块采一次**并复用(`_summary()` 重采会让 `.json["gpu"]` 与同一跑的 `.log` 头块不一致)。`env` 名按 **`envs/<name>` 路径段**反解 —— `sys.prefix == sys.base_prefix` 在本机**是错的判据**(env 真身在数据盘、软链进 `envs/`)。关掉:`--no-runlog`(扫 `sys.argv` **字面量**,因为 `start()` 早于 argparse)或 `AUTODRIVEDATA_RUNLOG=0`。**注意 `convert_hivt_pt.py` 以文件路径在 hivt env 跑,脚本自带项目根 `sys.path` 引导**(否则 `import autodrivedata` 必炸) |

## 3 `tests/` — 单测与 oracle 对比(autodrivedata env)

> 与 §2 的能力目录**镜像**;`test_layer_guard.py`、`test_docs.py`、**`test_entrypoints.py`** 在 `tests/` 根(包级守卫)。
> ★ `test_entrypoints.py` **扫全包**所有带 `main()` 的模块,要求都有 `if __name__ == "__main__"` 守卫
> (缺了 ⇒ `python -m` 进去**什么都不做、退出码 0**,批处理里会被当成"跑过了"而产物目录是空的) ——
> 它**取代**了原先只扫 `edit/` 的那份,因为`gs/eval_edit.py` 从没被扫到过。
> 用例基线数**只在 CLAUDE.md 常用命令里写一处** —— 复述在多处必烂(这三个文档都写过 913,早已过期)。

| 类别 | 文件 | 说明 |
|---|---|---|
| 纯值库单测 | `test_geometry.py` `test_calib.py` `test_gt.py` `test_compare.py` `test_paths.py` `test_scenarios.py` `test_static_gt.py` `test_traffic_light.py` `test_semantic.py` `test_radar.py` | 手算断言,不依赖 carla / AutoLabel |
| 运行日志 | `test_runlog.py` | 三件套契约的**纯值**回归钉(不依赖 carla/torch/GPU)。`TestHeader` 钉头块 `script`/`started` 正则/`argv`/`cwd`,**且指纹只采一次**(monkeypatch 计数器 —— 重采会让同一跑的 `.json["gpu"]` 与 `.log` 头块不一致);`TestTee` 钉 `print()` 进 `.log` + **`__getattr__` 全量代理**(`isatty`/`encoding` 与内层流一致)+ **恢复的是构造时那个对象**(`sys.stdout` 身份相等,自证 `capsys` 不被破坏);`TestRegistries` 钉**同名重复登记去重**(同一文件登记两次只留一行 —— `inputs` 是"读了哪些"的**集合**,不是调用流水账)+ 不存在的路径记 `missing` **不虚报** + `artifact_dir` 的 `n_files`/`bytes_total`;`TestFailure` 钉 `ValueError` → traceback 进 `.log` / `status="error"` / **异常照常抛出**,`SystemExit(1)` → `exit_code=1` / `status="failed"`;`TestSwitches` 钉 `AUTODRIVEDATA_RUNLOG=0` 与 `sys.argv` 里的 `--no-runlog` 都**一个文件不建**;`TestFingerprint` 钉无 CUDA / 非 git 下取 `null` 而**不抛** |
| **图像编辑(2026-10-05)** | `edit/test_depth_cond.py`(36) `edit/test_cldm_backend.py`(24) `edit/test_conditioned_gen.py`(15) `edit/test_calibrate.py`(12) `edit/test_harmonize_target.py`(14) | 三个文件**全部不出模型、不读 5.71 GB 权重**。被测契约全是**静默型**:条件图方向反了(`1/d` 漏掉 ⇒ 白 = 远,**`minmax` 也是单调的所以"看着像样"**)只有拿错误写法与正确写法做 Spearman 才分得开(实测 **−0.988**);CLIP 196 键前缀重映射错了会走到 `load_state_dict` 报一堆不相干的键,而**宽松加载会把缺的键当随机初始化 —— 模型照样出图,CLIP 那一半是白噪声**;权重 sha256 缓存边车按 `(size,mtime)` 命中,**"能改 mtime 的人能骗过它"这条边界明写成测试**(同族的 `ptp` 断言也曾因为"min-max 按定义就铺满量程"判红过 —— 量的应是**中段占比**);hint **必须 3 通道**(官方靠 `HWC3()`,漏了在 `openaimodel.py` 深处报 `expected input[2, 1, 512, 512] to have 3 channels`);三个条件源**尺寸必须都是 `COND_OUT`**。★ 三条断言是**我自己写过头**后被实测打回的(`== -1.0` / `== 逐位` / `ptp`),都改成带容差或换统计量,并把"为什么不是严格相等"写在注释里。★★ 2026-10-06 补:`test_calibrate` 钉**口径默认必须是 101** + `--grid-points` **真的传下去了**(AST 判据 —— 同族 `noise_curve --levels` 曾被静默忽略);`test_harmonize_target` 钉**行序无关的 `box_shift`**、**掩膜外逐位不变**、**对齐超限当场抛**、以及"补丁取自真值 ⇒ 距离 0"的管道自证 |
| **层守卫** | `test_layer_guard.py` | **包纪律的可执行版本**(docs/refactor-2026-09.md §3):`LAYER_RULES` = 目录 → 禁止 import 的三方名,最长前缀匹配。旧版(`test_paths.py` 的整包禁令)的两个洞已堵:**马甲库**(`ultralytics`/`mmdet3d`/`mmcv`/`lightning` 会拉起 torch 但字面无 torch)、**字面量动态导入**(`importlib.import_module("x")`)。`TestPackageLayers` 扫真实包 + 强制新子目录必须显式声明;`TestLayerGuardSelfCheck` 用**合成源码注入**做立论自证(12 条:抓得住三类违规,且规则能区分、不是"见 carla 就红") |
| **文档守卫** | `test_docs.py` | **「文档/入口不腐」的可执行判据**(2026-09-26 重构后补 —— 那次 21 条引用失效**全是静默的**,只有照着做的人拿到 `FileNotFoundError`;失效模式与 `paths.py::parents[1]` 同族:写的时候对、挪了之后静默错)。三类:① `CLAUDE.md`/`README.md`/`docs/fileTree.md` 的 markdown 链接目标必须存在;② 同三份文档里的 `python -m autodrivedata.<...>` 必须 `find_spec` 可解析;③ **包内 `.py` docstring 的 markdown 链接**必须存在。**第三类是第一版的漏网** —— 只扫 `.md` 时,`utils/fonts.py` 等处的 **16 条**坏链在「全文档坏链 0」的结论下整体逃检 ⇒ **判据的覆盖范围本身也是判据的一部分**。每条各带一条**扫描器自证**(防正则腐化后空过)。只判机械可判者:docstring 里的裸文件名不算路径;形如**方括号后紧跟圆括号单位**的**单位注记**(如米/像素、度、yaw=0)由后缀白名单滤掉(不加会多 9 条假阳性)。**`.`md` 侧不加白名单** —— 那里 `]` + `(` 就是 Markdown 链接,写它就是真坏链(本行的初版用实例演示,当场把守卫测红) |
| 地图矢量线 | `test_opendrive.py` `test_mapvec.py` `test_mapvec_schema.py` `test_mapviz.py` `test_chamfer_ap.py` `test_chamfer_gpu.py` | 含闭式解手算锚点与真实 xodr 计数锚点 |
| **多传感器采集器 + BEV 底图(2026-09-30)** | `tests/sim/test_collect_surround_lidar.py` `tests/map/test_bev_base.py` | 两条都补的是**「不报错、只是数偏」**类失效。采集器那组:① `SENSOR_OFFSET` 必须 == `slam_eval.LIDAR_LEVER`(挂点错 ⇒ `eval_slam` 补错杆臂,ATE 静默偏);② LiDAR `spawn_actor` 的位姿实参**用 AST 判**必须是 `SENSOR_OFFSET`(纯文本扫会被注释里那句 `carla.Transform()` 骗过);③ `calib.json` 的**自述字段**(`provenance()`,纯函数):`--autopilot` 的 run 不许写成 `const_speed` + 具体速度 —— 那份假溯源会把人引去做红线禁止的 A/B;另有「spawn 侧与自述侧必须由**同一个** `AUTOPILOT_SPEED_PCT` 导出」与「`provenance()` 真的被 `main()` 调用」两条。底图那组:两条换算链各把**传感器原点**映回**已验证的表**(`SENSOR_OFFSET` / `NUS_RADAR_MOUNTS_CARLA` 翻号),外加**非原点处**一致(否则删掉 `R_yaw` 也照样绿)、五雷达原点必须互不相同、PCD 读侧与写侧逐位往返、空点云(NaN 哨兵)读成 0 点、`Px.arr` 与 `Px.__call__` 逐点一致 |
| **静态遮挡 A/B(§P-V12)** | `tests/sim/test_occlusion.py`(35)`tests/sim/test_collect_ab_route.py`(22) | 这一档的失败模式**全是不可见的**:墙摆偏/矮/离车道近一点,采集照样跑完、GT 照样相等、图看着也对,只有"效果比预期小"——而那与"模型对遮挡鲁棒"长得一模一样。纯值组:**几何锚**(净空 > 0、部分遮挡落 21–24 px 断崖、`full` 恒 0、`gap` 单调性方向)、**深度 ≠ 斜距**(两种口径相差 8%,`label_2` 实测 58.6 px 才对得上)、**视线摆放 vs ego 正前方**(正前方漏 18.3% 车宽的**全高**缝)、摆位代数(同一平面 / 对称 / 首尾相接铺满 / 长轴垂直)。结构组:**GT 过滤器与道具 id 无交集**(AST 读源码前缀集对表,不是"读代码确认过了")、`--occluders` 默认 `none`、`--occluder-gap` 默认**引用**模块常量、`is_cleanup_target` 清场与收尾同源、`dims_match` 反例(长厚互换必须红) |
| **实例分割 GT + 判据(§P-V15)** | `tests/perception/test_inst_tags.py`(18)`tests/perception/test_inst_eval.py`(20)+ `test_sem_bev.py` 的并集等价(3)+ `tests/gt/test_refilter.py` 的雷达与跨设备(2) | 纯值组钉**两条实测口径**:关卡网格也有 id(不筛则 PQ 分母全是路面)、实例相机 R 通道 == 语义 tag(塌了的症状是"PQ 语义项恒为 0",会被读成"模型的类报得差")。判据组用**合成掩膜**(答案是我摆的):配对不看类会虚高、**空并集是 `nan` 不是 0**、`PQ = SQ × RQ` 这条**定义性质**必须成立、漏检只掉 RQ 不掉 SQ、平移掩膜 SQ 必须降、**conf 顺序不许重排**(高 conf 命中 vs 没命中,AP 必须不同)。并集等价:类级掩膜必须是实例掩膜的并集,否则两条链**看的不是同一次推理**。refilter 组钉**一路都不能漏**(雷达在 `samples/` 下,是 `training/` 的兄弟目录)+ 跨设备退化成拷贝时 `linked`/`copied` 必须分开计数 |
| **P2-A 语义分割 GT + 判据(§P-V13)** | `tests/perception/test_sem_tags.py`(17)`tests/perception/test_sem_eval.py`(14)`tests/sim/test_sem_tags_oracle.py`(11)+ `test_sem_bev.py` 新增 7 | 纯值组钉**编解码与口径**:灰度往返无损(含 0 与 255 这两个最容易被当背景吞掉的值)、`decode` **只认 `mode="L"`**(上色预览读回来照样是数组,不拦就一路算完)、三类**两两不交**、排除集与三类**不相交**(既算 GT 又报"被排除",占比与 IoU 会自相矛盾)。判据组钉**尺子真的在量重叠**:手算 IoU、**没样本是 `nan` 不是 0**、把预测平移 IoU 必须**单调下降**(恒返回常数的假尺子当场红)、空类被 mIoU 跳过(有意为之,所以更要钉)。oracle 组钉**tag 编号 == `carla.CityObjectLabel`**,含"防抄成旧版编号"的反向对照。向量化对拍:同一个掩膜走两条实现必须**逐点相同**(拿仍在的标量版当 oracle) |
| **静态道具 GT + 判据(§P-V14)** | `tests/gt/test_props.py`(29)`tests/perception/test_prop_eval.py`(15)+ `test_sem_tags.py` 新增 7 | 纯值组钉**三处口径的汇合点**(`actor_box_from_prop`):**度/弧度**(把度直接当弧度传,六路里**只有 yaw≈0 那路看着正常**)、**全长/半长**(差一倍)、**盒偏移(局部系) vs actor 位姿(世界系)** —— 三处错了都**不抛异常**,只让投影框默默不对。类别组钉**tag 集不许被顺手扩大**:`PROP_TAGS` 只含实测过的 `Dynamic`,`Static`/`Other` 不许并进来、不许并进三个 GT 类、三桶互不相交。判据组用**合成 root**(答案是我摆出来的,不用猜):尺子平移单调性、空掩膜是 `nan` 不是 1.0、**裁断样本单独计数**、`instance_id` 对不上要报 `never_seen`、样本不足**作废而非通过**。⚠️ 合成夹具的**轴向**要按实测摆:世界 +x → 相机 +z,摆错轴的症状是"投影出画(None)"而不是报错 |
| **GT 口径迁移(§P-V12 / §P-V20)** | `tests/gt/test_refilter.py` + `tests/gt/test_gt.py::TestGtLine` 的 7 条 + `TestReboxLine`(5)+ `TestClip2d`(6)+ `TestPngSize`(2) | 在**合成 root** 上跑,不碰真数据。读侧判据:① **不得写成 `x1 == x2`** —— 实测 `wet_road` 帧 57 有一条 **0.01 px 高**的框,两个数打印出来不相等却早该剔除,写成相等判断会让那帧 GT 比别的 root 多 1、A/B 硬门槛当场破(边界 1.00 px 留、0.99 px 剔,都钉了);② 迁移工具**只许动 `label_2`** —— 旁路目录必须是**硬链接**,判据用 `st_ino` 相等(**"文件大小一样"对拷贝也成立,证明不了没重写**),外加反向对照"源 root 不许被改"与"跑第二次必须报错"。**重算侧(`--clip2d`)**:真归档行(逐字抄录,抄错 z 会差 13.6 px 而**看着像代码 bug**)→ 框贴到 `W-1/H-1`;**未裁断的行逐字节不变**(不是"差得少" —— 按舍入过的字段重打会注入 ~0.1 px 噪声,而这类行占 82%);两个修正**一次做完**(退化行在 `--clip2d` 下照样剔除);**逐帧各取各的 `P2`**;缺 `calib`/缺图**抛而不回落**(混口径的 root 在 A/B 硬门槛下看不出来);**默认关 = 旧口径逐字节相同**(新口径必须显式索取) |
| 教程能力线 | `test_mono_depth.py` `test_stereo.py` `test_multilidar.py` `test_slam.py` `test_accum.py` `test_ground.py` `test_cluster.py` `test_collect_rig.py` | 各含手算锚点;`collect_rig` 兼作采集器回归先例 |
| SLAM 精度/在线线 | `test_slam_eval.py` `test_live_slam.py` | `slam_eval`:ATE/RPE 手算锚点 + 杆臂方向(不补杆臂 ATE 2.44×);`live_slam`:与离线 `slam_odometry` **逐帧同输入同输出**(<1e-12)+ `SlamWorker` 滞后有界/止损/同步模式 |
| 闭环路线(§P-H.3) | `tests/sim/test_route.py` | **回环数据源的可离线验证部分**(纯值,零 carla)。三条手算钉:① 方框图 → 4 节点最短环、`A→B→A` 掉头被 `min_len` 拒、`max_nodes` 兜住无限链;② ★ **进度单调**(前视点索引不许回退 —— 纯追踪最经典的失败是车在起点附近来回蹭,而它"看着像在开")+ ★ **打舵符号**(目标在右 ⇒ `steer` 为正;符号反了车朝反方向冲出去)+ 双踏板恒互斥;③ **路网键适配**(`TestMakeSuccessors`,**合成 Waypoint**):4 段×30 m ⇒ 40 节点环、单圈几何 **120.0 m**(=39×3 + 收尾那 3 m,同时钉住"键里必须有 `s`"与"闭环长度要算收尾段")、闭合容差必须按**采样步长**给、回调吐出的键当次就进缓存。外加 `lap_budget`/`speed_ceiling`:**一圈 ≥250 帧 ⇒ 速度上限**(200 m 环 8.0 m/s、120 m 环只有 4.8 —— 开快了两次到访帧差不够,**回环必然不触发**)。**段起点选点组**(`TestSpawnPointSelection`/`TestFarthestFromCentroid`,+11):★ 返回必须**升序**(否则"能不能复现已知集合"退化成"得先猜对方什么顺序")、★ **贪心对种子敏感**(同 5 点同 k=3,种子 0/3/4 给 `{0,3,4}` 而 1/2 给 `{1,2,4}` —— 所以种子不许手挑)、★ `min_pairwise` 取**最近**那一对(**写成 max 会让"段间够不够远"永远通过**)、单点返回 `inf` 不是 0(写成 0 会让 `--min-gap` 在 k 给错时误报"图太挤")、选择序首元素恒为种子、嵌套性(k 是 k+1 的前缀 —— `spread_curve` 单调不增就靠它)、平局取下标最小 |
| **多图池 + 切图超时(2026-10-03)** | `tests/map/test_assemble_maptr.py`(18)`tests/sim/test_carla_common.py`(5) | 两组都补「**不报错、只是数偏**」类失效。**多图池**:`--map-json auto` 的三形态解析(`Town13`/`Carla/Maps/Town10HD_Opt`/平铺 `Carla/Maps/Town13/Town13`;写成"去掉 `Carla/Maps/` 前缀"会在平铺图上剩一层)、★ **显式给文件与 `auto` 逐字节相同**(旧命令零改动的唯一判据)、缺 `map` 字段**必须抛**(静默沿用上一段 ⇒ 整段用错 GT,症状只是"学不动")、矢量 json **只读一次**、坐标系必须 `carla_world`、★ **混 rig 反向自证**(两段 fx 621/1266 ⇒ 必须抛**且点名哪两段**),以及**同 rig 池必须放行** —— 后一条同时钉住本守卫第一版的 bug:那时整份 `cams` 比,而 `data_path` 带段名前缀逐段必然不同,**正确的池子被判成"混 rig"**。**切图超时**:加载那一刻超时必须 ≫60 s、加载后恢复、★ **失败也要恢复**(`finally`;不恢复会把"一次明确失败"放大成"每步卡 5 分钟像挂死") |
| 实时可视化 | `test_live_common.py` | `compose_grid` **尺寸守卫**(不符必抛,防 `paste` 静默裁)+ `compose_rows` 每格**原生像素**逐像素等于源图 + studio `GRID_ROWS` 三层行序 + **`rig_spec`/`resolve_rig`**(两代 rig 口径与"按权重选":legacy 共用挂点 + pitch/roll=0,nuscenes 逐相机 6DoF;显式指定不被文件名覆盖)+ **`mount_deviation_of` 规格对账**(相机与雷达共用;矩阵顺序写反 ⇒ 平移爆掉而偏航仍 ~0、偏差随 ego 离原点变远而变大、legacy 实挂对 nuscenes 规格必报 110°、tick 前全 0 陈旧位姿**不许**判成"通过";`rig_mount_deviation` 是它的薄封装)+ `draw_hud(y=)` 第二行(第一行逐像素不变、`y=0` 与旧行为一致)。测试搬进包后**直连 `autodrivedata.sim.live_common`**,旧的 `sys.path` hack 与 `pyright: ignore` 已摘除 |
| 配置图 / 覆盖表 | `test_rigviz.py` | **交付物图的数值侧回归**(纯值,PIL + numpy)。`TestCoverageTable`:wide 三个盲区**逐项等于设计预算**(7.3353/6.0984/1.7224 = 15.1561°、覆盖 0.9579)、官方 rig 零盲区;重叠对的 `span` 必须落在两个相机各自的扇区里(共视探针按它摆锥)。`TestAzimuthIndependentImplementation`:`rigviz.azimuth_of` == `camera_rig.camera_azimuth_nus`(两套独立实现)。`TestRigLayoutFigure`:**盲区红弧"有当且有、无当且无"**(官方 0 个盲区 ⇒ 0 红像素;wide 有 ⇒ >0)+ 六通道都有画色与短码。`TestRulerLanes`:底尺**跨 0° 不崩**(`360.0 % 360 == 0` 会让 `rectangle` 抛 `ValueError`)+ 盲区红**列数** ∝ Σ盲区度数 + 六条泳道都画出来 + 页脚折行后每行实测宽 ≤ 画布且不压数字表。**不钉排版/错别字**——那些写成断言只会得到"改个字就红"的脆测试 |
| 绘制字体 | `test_fonts.py` | **中文字形不许静默变豆腐块**的回归钉。`TestProbe`:探针立论自证(`U+10FFFF` 在任何字体下都落 `.notdef`)+ **DejaVu 被正确判否**(它有 `−`/`°`/`★` 却画不了中文 ⇒ 判据不是"文件在不在")+ 生效字体实测能画中文。`TestRendering`:两个不同汉字在**画布上必须像素不同**(最强钉——豆腐块下它们逐像素相同)、`sanitize` 后零缺字 / 替换目标自己画得出 / 不等长(排版不错位)、CJK 宽 ≈ 2× ASCII(HUD 底条据此定宽)。`TestDrawnStringsAreRenderable`:**AST 扫全仓绘制字符串**(8 个绘制模块 × 绘制调用实参 + `hud_line` 之类构造器**函数体**——漏后者 `calib_live` 整行中文 HUD 会逃检)⇒ 逐个 `sanitize` 后零缺字;另有**根因钉** `test_no_module_draws_with_a_bare_text_call`(不许出现不带 `font=` 的 `d.text(...)`)与每模块 `import fonts` 钉 |
| **验收补钉(2026-09-28)** | `tests/perception/test_sem_bev.py` `tests/calib/test_viz_layout_cmp.py` | 两条都补的是**同一类失效:CLI 早就跑不起来,而 pytest 全绿**(见 [docs/acceptance-2026-09-28.md](acceptance-2026-09-28.md) §4.1/§4.2)。`test_sem_bev`:桩模型跑通 YOLOPv2 掩膜链(**该模块此前零覆盖**),顺带把一直没人测的 letterbox→裁 padding→缩回原图几何钉住(含反例对照:把带挪位置,输出必须跟着挪);并 AST 钉住"外部 `utils` 包不许回来"(注意与 `traj/convert_hivt_pt.py` 的 HiVT `utils` 是**同名多义**,别混)。`test_viz_layout_cmp`:`--a`/`--b` 必须真的决定**读图**路径 —— 两个 root 的图染成**纯红/纯蓝**,断言两张输出各自取自自己的 root(路径再写死必然同色);**改代码前先确认它对旧逻辑报红** |
| **3DGS 阶段 B(2026-10-04)** | `tests/gs/test_frame_sync.py`(12)`tests/gs/test_attribute_instances.py`(20)`tests/sim/test_collect_3dgs.py`(17) | 三组都钉「**不报错、只是数偏**」类失效。**帧同步**:坏首帧必须红、干净首帧必须绿、**帧数不够必须 `None`(未判)**、尺寸不等的两帧比 mean **必须抛**(广播出来的 `mean` 是个看着正常的错数)、**两三帧坏掉不许把"其余"的统计量撑起来**(首版用 min/max 时正是这么差点放过去的)。**归属**:★ **向量化 `project_points` 与 `calib.core.world_to_img` 逐点对拍**(投影口径只许有一套实现)、已知布局逐点还原、视场外/身后必须判无效、`min_samples`/`min_share` 各自生效、**打乱实例图后一致率必须塌到 κ 基线**、两半无共同归属必须**未判**;★ `TestCamConvention`:同一位姿下 `carla` 口径的 viewmat 与 `world_to_img` 差 <1e-3 px 而 `legacy` 差 >20 px(**它必须能红 —— 否则它就不是判据**)。**采集器结构钉**:`make_kind_blueprints` 必须逐字设 `CAM_ATTRS`(用假 `bp_lib` 问,不读代码)、`apply_pose` 四路同一个 tf、主循环必须走 `shoot_synced` 且**不许再出现 `q.get(`**(旧写法 = 恒定滞后 2 帧) |
| **CUDA 环境顺序(2026-10-04)** | `tests/gs/test_cuda_env.py`(12) | ★ **AST 扫全包**:除 `cuda_env.py` 外,`gs/` 里任何模块的**顶层**不许出现 `import gsplat` —— 那让正确性依赖"两行 import 的先后",而 `ruff format --fix`(isort) 会把 gsplat 挪到 first-party 之前(实测坏过一次)。再加一条宽松档:碰了 gsplat 的模块必须 import 过 `cuda_env`(覆盖"删掉 `ensure_gsplat()`、改函数内懒 import"这条绕过路径)。另有 `_arch_list_usable` 的行为钉(拒 `10.3`、收 `8.6`)+ **探针不许留副作用**(只问一句,不是设置)+ 豁免名单不许是死链 |
| **离线渲染器(2026-10-04)** | `tests/gs/test_render_gs.py`(10) | 两条被测契约都是**不报错的错**:① `rots` 缺了 —— 渲染**照样跑得起来**(gsplat 不在乎你给的是不是训练出的四元数),只是结果不是那个模型 ⇒ `test_missing_rots_raises` 是它的直接反例;② `opac` 存的是**已 sigmoid** 的值而渲染只 clamp ⇒ `test_opacity_is_not_double_activated` 按**行为**钉(双重激活必须给出不同结果;若夹具恰在 0/1 附近使两者相同,判据自己会喊"失去分辨力")。另有 `parse_frames` 越界**必须抛不许静默截断**、五件套往返逐位相同、长度不齐当场抛。⚠️ 真调 gsplat 的那 2 条**没给 `CUDA_HOME` 就跳过**(见 `_NEEDS_CUDA_ENV`)—— 不是"通过了",是**第三态** |
| **编辑算子与判据(2026-10-04)** | `tests/gs/test_edit_eval.py`(23) | 三条被测契约都是**删错了但看着像成功**(`-1` 被顺手删 / 请求的 id 一个没命中却静默成功 / A/B 位姿其实对不上),加一条**第三态**(掩膜空 ⇒ 未判)。三档读数有**手算锚点**(误差 0.1/0.01/0.001 ⇒ PSNR 20/40/60,缺口 40、补上 20、闭合率 0.5);掩膜与全局**真的分区**(误差只在掩膜内时两个数必须分开)。另有**冒烟才抓到的两条接线问题**钉:`--list` 不该被 `--out-tag` 卡住、`check_ids` 必须排在 `load_set` **之前**。全部带反向自证(去掉 `-1` 守卫 / 去掉「删 0 个报错」/ 把容差放宽成 0.07 ⇒ 各自当场红) |
| **训练环境的三个静默失效(2026-10-04)** | `tests/gs/test_train_3dgs_mini_env.py`(12) | 三条底层机制**同一个**:设置与使用之间隔着别的东西。**`--seed`(4 条)**:覆盖**不报错、不改任何输出形状**,只有"换个 seed 数字、结果一模一样"才看得出来 —— `TestSeedEverything` 钉三处随机源真的被钉住、★ **换 seed 必须换序列**(原 bug 的直接反例)、`-1` 不动上游;`TestNoClobber` 是**源码顺序钉**(`_seed_everything` 之后到读数据之间不许再出现 `manual_seed` / `.seed(`,只看代码行)。**`CUDA_HOME`(4 条)**:`TestCudaNote` 钉 ★ **告警块必须排在 `import gsplat` 之前**(torch 的扩展缓存按 flags 的 hash 定位,环境不同 ⇒ **重编并覆盖规范 `.so`**,而不是报错)、`CUDA_NOTE` 在未设/已设两种环境下各说对、**`CUDA_HOME` 进了 runlog**(跨 build 不许混比的前提)。**相机系默认值(4 条)**:`TestDefaultConvention` 钉默认是 `carla`,且 ★ **签名与 argparse 都取同一个常量 `DEFAULT_CAM_CONVENTION`**、源码里**不许再出现任何写死的 `default="…"`** —— 那个默认值**翻过一次**(`legacy`→`carla`),两处各写一个字面量正是"改了一处、另一处还是旧值"的静默失效;另钉 `legacy` 仍**可选**(归档产物要靠它复现)。三条都有**反向自证**:seed 那行放回去 / 告警块挪到 import 之后 / 常量改回 `legacy` 或只在 CLI 处写死 ⇒ 当场变红 |
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
| `temporal-stitch-2026-09-29.md` | **时序拼接实测**(2026-09-29):逐帧 BEV 矢量预测 → 全局矢量图首次真跑通。含**两个位姿源对比**(CARLA 真值 vs SLAM,ATE 0.877 m **小于**模型逐帧抖动 2–4 m ⇒ **位姿不是瓶颈**)、**融合口径的 60× 差**(Hausdorff 去重对时序是错配,只合 0.7%)、以及为它新建的双传感器采集器的首跑记录 |
| `legacy-rig-archive.md` | **legacy rig 归档**(2026-09-28):该口径按用户裁决从代码中**完全移除**,本文件冻结镜像 bug 的**成因 + 逐相机方位数字 + 覆盖表 + 图**(`assets/legacy-rig/`)。移除的代价是「镜像 bug 已修」无法在仓内重跑复现 —— 要查当年长什么样看这一份,别在代码里找 |
| `acceptance-2026-09-28.md` | **全能力面验收记录**(2026-09-28):一次"从零把整条流水线重跑一遍"的逐数字对账(§0 扫描表)+ 复跑中新发现的 11 条问题(3 条已修并补回归钉、4 条待用户裁决、4 条口径注记)+ 产物索引 + 复跑须知。复跑入口 = [tools/showcase.py](../tools/showcase.py) |
| `ros2-humble-build.md` | **ROS2 Humble 源码构建记录**(2026-09-28,**环境侧**):三档成本实测(136/158/349 源码包)、四步命令与每步的坑、踩坑清单、构建期 github 抓取的镜像注入。**与能力线无关**(主线有意纯 numpy 不吃 ROS);`github.com` SNI 级被封的判据也在这一份 |
| `ros2-feasibility.md` | **ROS2 引入可行性评估**(2026-09-28):结论 = **不引入**。三条判据(GIL 疼点已被"同步执行"绕过 / 教程能力线 16 次选择不用它 / 代价可量化而收益不可量化)+ **分层守卫会静默失效**(`rclpy` 不在任何禁用集里)+ 工业惯例三层对照 + 唯一真候选(FAST-LIO2 对标基线)为何区分度低 + **五条触发重评条件**。与 Plan.md §5.12(官方 MapTR/MapQR 复线终止)**无关** |
| `edit-3dgs-plan.md` | **3D 场景编辑(3DGS)** 探索计划。§A 三条杠杆(分辨率/迭代/致密化)全部实测收口;§B 相机系与帧同步两个前置缺陷;§C.0.3 C1 搁置;★ **§C.0.4(2026-10-05)** = 重开时的 TAA 对照**做完了**,结论**不是 TAA** —— 三条独立缺陷(道具资产**不渲染** / 逐像素签名**尺子无分辨力** / 相机位姿与 `poses_*.json` **差 79.132 m**) |
| `edit-pointcloud-plan.md` | **点云编辑(LiDAR)** 计划,**已以证伪收口**:实测 A/B 位姿本就差 0.0619 m,而编辑该产生的信号(前方点数差 +2)**小于**对照 floor(−65)⇒ 靶落在噪声里。三个**重评条件**写在 §4 |
| `edit-image-plan.md` | **图像编辑 / 合成与和谐化 / 域自适应**计划 + **StyleGAN3 与 ControlNet 的真实接入实测**(2026-10-05):两条链都在**现成 env** 里跑通(只加 6 个包 + 4 处版本 shim);★ 关键读数:StyleGAN3 官方权重**全域外**(只有人脸/动物脸)且朴素投影**投不进去**;ControlNet **目检确认在按条件控布局**;★ 结构性发现 = **本项目的真值 depth/seg 正好是 ControlNet 的条件源** |

## 6 `outputs/` — 产物目录(唯一落点,**不展开子文件**)

> 全部写盘路径经 `autodrivedata/utils/paths.project_path()`(相对路径 = 相对项目根,不随 cwd 漂移)。
> 下列目录**均【未入库】**,克隆后不存在,由对应脚本重新生成。

| 目录 | 装什么 | 产出者 |
|---|---|---|
| `kitti_*` | KITTI root 结构数据集(`image_2` / `velodyne` / `calib` / `label_2`):`kitti_day_clear` `kitti_drive` `kitti_sweep_*` 等 | `collect_drive.py` / `collect_kitti.py` |
| `kitti_ab_epic_*` | P1 A/B 帧级配对序列,Epic 质量档重采(day_clear / sunset_glare / rain_night / dense_fog / wet_road,各 70 帧)。⚠️ **旧口径**:GT 含 11 条零面积退化框(GT=194),AP 因此偏低(§P-V12)。**引用前先确认用的是哪一版** |
| `kitti_ab_epic_gtfix_*` | 同上,经 `gt/refilter.py` **重筛退化框**(GT=**183**,4 对逐帧条数全等;`image_2`/`calib` 是硬链接,与上面逐字节相同)。⚠️ **现在是中间代** —— 裁断口径未修,已被下面的 `_clip2d_` 取代 | `gt/refilter.py` |
| `kitti_ab_epic_clip2d_*` | **★ P1 矩阵的当前口径**(§P-V20):`gt/refilter.py --clip2d`,GT=**183** + **20 条被画幅裁断的框按 KITTI 口径重画**(位移中位 153.5 px、max 256 px;其余 163 行**逐字节不变**)。数字与 `_gtfix_` **逐位相同** —— day_clear 0.669 / 浓雾 −0.578 / 雨夜 −0.487 / 湿路面 −0.215 / 逆光 −0.125(贪心匹配把 6 条 FN→TP 翻转抵消掉了,tp 向量逐位相同)。⚠️ 无 `training/pose/`(源 root 采于加逐帧位姿之前) | `gt/refilter.py --clip2d` |
| `kitti_ab_occl2_*` | **P1-6b 静态遮挡 A/B**(§P-V12):`none_all`(基线)/ `partial_all` / `full_all`(阳性对照)/ `partial_nearest`。⚠️ 跑在**修退化 GT 之后**,GT 总数 **183** 而非 194,须与 `kitti_ab_epic_*` 分开读;⚠️ 裁断口径未修(被下面 `_clip2d_` 取代) | `collect_ab_route.py --occluders ...` |
| `kitti_ab_occl2_clip2d_*` | 同上 + `--clip2d`(§P-V20)。**当前口径**;四项数字逐位不变(0.674 / 0.007 / 0.021 / 0.577,Δ −0.667 / −0.652 / −0.096) | `gt/refilter.py --clip2d` |
| `kitti3d_ab_*` | 上列 A/B 的 3D 伪标签输出(AutoLabel 消费) | `auto3dlabel run` + `eval_kitti.py` |
| `surround_*` | 环视 6 相机数据集(`surround_train` / `surround_p3` / `surround_town13` / `surround_pred` 逐帧契约)。**`surround_v2` = §P-M.12 的 1600×900 官方口径训练集**:5 段 × 100 帧(stride 5,合计路线 1280.7 m),`seg0..seg4` 各为一段独立路线 + `map_infos.json`(500 帧,带 `seg`/`frame_in_seg`);`collect.log` 为采集计时。**`surround_town05_epic`(§P-V25,第二张图)**:Town05_Opt 5 段 × 100 帧,spawn `38/92/249/255/267`,与 `surround_v2_epic` **完全同源**(同采集器 / 同 rig / 同 autopilot 70%);⚠️ 中位 **8.28 m/帧**(16.6 m/s)/ 路线 **737 m**,而 v2_epic 是 2.59 m/帧(5.2 m/s)/ 215 m —— 差的是**地图限速**,是"换图"自带的一部分 | `collect_surround_lidar.py` / `assemble_maptr.py` |
| `multimap` | **双图池(§P-V25)**:`seg0..seg4` → `surround_v2_epic/seg{0..4}`(Town10HD_Opt)、`seg5..seg9` → `surround_town05_epic/seg{0..4}`(Town05_Opt),**符号链接不复制 8.6 G**;`map_infos.json` 1000 帧 / 10 段,由 `assemble_maptr --segs-dir outputs/multimap --map-json auto` 组装。⚠️ 组装收尾有 **rig 自证**(`assert_same_rig`):`MapTRDataset` 只认首帧那一份 `cams`,混 rig **不报错**、只让训练学不动 —— 真实数据负向对照实测 **5 s 抛出并点名两段** | `assemble_maptr.py --map-json auto` |
| `surround_inst_demo` | **实例分割 GT 首批**(§P-V15):20 帧 × 6 路 RGB + `sem_<cam>/`(语义 tag)+ **`inst_<cam>/`**(16 位灰度实例 id 图)。`perception/inst_eval.py` 出第一条实例级读数(5 帧 × 5 路、GT 64 实例):**PQ 0.6257 / SQ 0.8447 / RQ 0.7407**,mask AP@0.5 = 0.6341 —— SQ 与 RQ 分得开"掩膜画得准"(0.845)与"物体找得全"(0.741)。⚠️ **预测侧是借来的通用 COCO 权重 `yolo11s-seg.pt`,不是本项目训练的** —— 数字读作"通用分割模型在这批数据上有多行",不是"本项目的实例分割能力" | `collect_surround.py --sem --inst` / `perception/inst_eval.py` |
| `surround_sem_demo` | **P2-A 像素级分割 GT 首批**(§P-V13):20 帧 × 6 路 RGB + **6 路 `sem_<cam>/` 语义 tag**(8 位灰度,tag 即像素值),`calib.json` 带 `semantic` 键可溯源。跑 `perception/sem_eval.py` 出第一条可打分的语义 BEV 数(图像 mIoU 0.557 / BEV 0.512);`outputs/sem_eval/gt_preview_*.png` 是 RGB 与 GT 上色的并排目检图 | `collect_surround.py --sem` / `perception/sem_eval.py` |
| `kitti_static_sem` | **静态 GT 的判据数据集**(§P-V21):`collect_static_gt --sem --props` 40 帧,`static_sem/{fid}.png`(语义 tag = 判据的 oracle)+ 带相机位姿的 `static_gt/*.json`。首跑:`signal:traffic_light` 真命中 **1.000**(横移 0.103 / 随机 0.017,n=58)、`lane:Solid` **0.997** / `lane:Broken` **0.930**(横移均为 **0.000**);`lane:SolidSolid` **未判**(25 条全在画外)。⚠️ **`image_2` 比同帧 GT 差 1 帧**(§P-V22,修帧偏移**之前**采的)| `collect_static_gt.py --sem --props` / `perception/static_eval.py` |
| `kitti_static_sem_v2` | **静态 GT 判据数据集(修帧偏移后)**(§P-V22):`collect_static_gt --sem --props` 20 帧。⚠️ **与上面那份的区别是它没有 1 帧偏移** —— 首跑自证 `frame=…(RGB/实例/语义)` 三路一致,离线跨帧 argmax 也给 offset 0。**新数据一律用这份**;`kitti_static_sem` / `kitti_static_props` 的 `image_2` 比同帧 GT 差 1 帧(**判据不受影响**,但**别拿它们的 `image_2` 配 GT**) | `collect_static_gt.py --sem --props` / `perception/static_eval.py` |
| `kitti_static_props` | **静态道具 GT**(§P-V14):`collect_static_gt --props` 20 帧 —— 每帧 `static_prop_gt/{fid}.json`(5 个道具的世界系记录 + 相机位姿与内参)+ `prop_inst/{fid}.png`(实例 id 图,uint16,**判据的几何 oracle**)+ `prop_sem/{fid}.png`(语义 tag,**判据的类别 oracle**)+ `overlay/`(信号锚点 + 车道线 + 道具 3D 盒)。判据 `perception/prop_eval.py`:可判 84 条 coverage **逐条 1.0000**、类别命中 **1.0000**,另有 4 条被画幅裁断(**单独计数,不并入裁决**)。⚠️ **`image_2` 比同帧 GT 差 1 帧**(§P-V22):判据与 `overlay` 之外一切拿 `image_2` 配这套 GT 的用法都不成立 | `collect_static_gt.py --props` / `perception/prop_eval.py` |
| `probe_static_prop_gt` | §P-V14 探针的一次性产物:`report.json` 记四问的全部实测数字(actor 普查 / yaw 扫描的两种读法对拍 / tag 归属 / 杂项 tag 形态) | `sim/probe_static_prop_gt.py` |
| `traj_town*` | 多 agent 轨迹数据集(HiVT 输入) | `collect_traj.py` / `assemble_traj_pt.py` |
| `maptr_*.pt` | MapTR 权重(`maptr_600` / `maptr_1000` / `ep256` / `ep512` + `.opt` 优化器态)。**⚠️ 全部四个已标废弃**(2026-09-22,§P-M):采集时用的 rig 是错的(镜像 + pitch/roll 硬编码 0),数据本身错 ⇒ 重采重训。文件保留仅供历史对照与 legacy 路径回归,勿作新实验基线 | `train_maptr.py` |
| `maptr_v2_single{F,}.pt` | **§P-M.11 冻结口径下的第一批新权重**(1600×900 / `surround_v2`)。`maptr_v2_single` = 路线级训练集(seg0–3 × 100 = 400 帧,留出只能评 seg4);**`maptr_v2_singleF` = 帧级口径(320 帧 = `--exclude-seg seg4` ∩ `--keep-in-seg 0:80`),一个模型给两套留出**(帧级 seg0–3 `[80,100)` / 路线级 seg4)。日志同名 `.log`;**结尾的 `FAIL` 是过拟合闸门误报**(见 `train_maptr.py` 行,修于 2026-09-24) | `train_maptr.py` |
| `training/map/Town05_Opt_full.json` | 第二张图的**全图矢量**(2116 实例 / 2062 折线,`export_mapvec --map Town05_Opt` 实测 **3.4 s**)。`--map-json auto` 靠它 + 各段 `calib.json["map"]` 逐段解析。⚠️ 这里也是 **Town13 不可用**的物证所在:同目录的 `Town13_full.json` 有 **412 MB**,是**平铺大图** `Carla/Maps/Town13/Town13` 的产物,而该图在本机的相机采集**必崩**(§P-V25 一) |
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
| `hdMapGitHub/stylegan3` · `hdMapGitHub/ControlNet` | 【未入库】**图像编辑线的两个参考仓库**(2026-10-05)。**同样保持 pristine** —— 版本适配全部走 `sys.modules` 替身与键重映射(见 [edit-image-plan.md](edit-image-plan.md) §1),**没有改动这两个 repo 里的任何文件**。权重在 `weights/stylegan3/` 与 `weights/controlnet/`(各带 `MANIFEST.txt` 记 sha256) |
| `.vscode/` `.claude/` `build/` `*_cache/` | 【未入库】本地 IDE 配置、AI 会话配置、构建与测试缓存 |

# 测试与踩坑日志

开发踩坑与测试记录(现象 → 根因 → 修复 → 回归保护)。里程碑叙事见 [milestone.md](milestone.md),方案细节见 Plan.md(单一事实源)。**代码里带回归注释的坑勿重犯。**

## 测试基线

- **base env** 单测(纯数学,不依赖 carla/auto3dlabel):**131 passed**(2026-09-08)
- **autolabel env** oracle 脚本:import auto3dlabel 当单一事实源做往返断言(KittiCalib 投影链、GT 往返、devkit 构造、NusBox 往返)
- 测试环境策略见 Plan.md §5.6:纯数学单测 base、oracle autolabel,两边互不污染

## 环境 / 平台

| # | 坑 | 修复 |
|---|---|---|
| E1 | UE4 拒绝 root 运行 | 专用用户 carla 跑服务器(`bin/carla_server.sh`) |
| E2 | /root 权限 700 致 carla 用户不可读 | 改 711 |
| E3 | headless GPU 栈崩溃 | EGL json 重建 + LD_PRELOAD 符号 shim |
| E4 | TaskStop 杀不死 UE4 子进程 → 端口占用崩新实例 | `pkill -f CarlaUE4` |
| E5 | teardown 阶段偶发 segfault(exit 139) | 数据从不丢(采集已完整落盘);纪律 = 每次采集前 `carla_server.sh start` 干净启动 |

## 采集 / 数据

| # | 坑 | 修复 / 回归保护 |
|---|---|---|
| C1 | 传感器属性 spawn 后才设不生效 | 必须先设 blueprint 再 spawn |
| C2 | 同步模式 spawn 后 get_transform 返回恒等 | spawn 后必须 tick |
| C3 | LiDAR 200k pps 车簇仅 13~117 点 | 1.3M pps(实测 63k 点/帧、360° 全覆盖) |
| C4 | 强度通道退化(std 0.055)致真车全漏 | 语义 LiDAR + 反照率×入射角强度合成,Car AP 0→0.529(semantic.py) |
| C5 | dropoff_general_rate>0 点数反常上升(105k vs 64k) | 机制未明,弃用 |
| C6 | autopilot/TM 路线不可复现(同 seed GT 分布差 7 倍) | A/B 采集改固定 ego 定速直行 + 静置车;红线:禁用于 A/B |
| C7 | az=300 误采顺光(太阳在车后,假"逆光更强" +0.054) | az=90=车头正前;判据 = 日盘入 FOV 时全图过曝最低(AE 压最狠) |
| C8 | 残留 actor 阻塞 spawn point 0 → spawn_ego fallback 反向出生点、轨迹失配 | 采集器清场 + 强制锚定 pts[0](yaw=0) + 起点校验 raise |
| C9 | 第 4 台静置车 65.0m 恰好卡 GT max_distance 阈值,起步抖动 GT 216 vs 219 不可比 | 改 62.0m 留 3m 裕量 → GT 218=218 |
| C10 | 语义 LiDAR 行人/骑行者 tag=0;walker 点稀疏 | 如实记录为平台边界 |
| C11 | mini_val 缺 scene-0916 → nuscenes-queue 遍历 KeyError | 生成器强制两场景(scene-0103 + scene-0916) |
| C12 | P2 信号虚高 21→5:同一物理信号杆挂多条 lane(landmark 按 lane 引用) | 按 (name, 位置) 去重 |
| C13 | P2 landmark yaw 怪值 -360/-540(OpenDRIVE 方位累积) | `% 360` |
| C14 | P2 车道线全断成 1 点段:left/right 交替采样打断顺序合并 | 按 side 分流各自合并(回归:`test_merge_lane_marks_alternating_sides_still_joins`) |
| C15 | P2 无实体标线(type/color NONE、w=0 xodr 占位)混入 | 过滤 |
| C16 | M3 行人 NAV 失败 → 全部导向地图原点(528 GT 聚集) | 只放有效 spawn 点 + 不导航 |
| C17 | M3 远距无点 GT(163m 框 > LiDAR 70m 量程)毒化校准 | box_to_gt_line max_distance=65 |
| C18 | M3 微调 3 败:驱动帧塌缩(官方 78.3 → 9.5/5.5/0.0),静态帧反而干净 | 疑似灾难性遗忘(训练域/推理域分布差),待查;生产定论 = 官方权重 |
| C19 | `pkill -9 -f CarlaUE4-Linux-Shipping` 在后台任务里把外层 bash 自己杀了(命令行含进程名字样,`-f` 全命令行匹配) | `pkill -9 -f 'CarlaUE4-Linux-[S]hipping'`([S] 技巧防自匹配) |
| C20 | 附加图 Town11/12 采集必崩:sync + ego + tick 均正常,**spawn camera(attach_to=ego)瞬间 segfault(Signal 11)**;Town10HD_Opt/Town13/Town15 同链全通 | 该二图渲染资源与 headless GPU shim 栈冲突(机制未明);已知边界:采集用 Town13/15,记入 Plan.md |
| C21 | 附加图 Town13 TM 车流:15 车 + TM 8000 首次 tick 把服务器打满(153% CPU,client 30s 无响应)——与 segfault 不同类,是算不过来 | 双层大图 TM 计算过载;降级 `--npc-vehicles 0 --npc-walkers 0`(仅 ego autopilot)可跑 |
| C22 | collect_static_gt 锚定 **yaw 硬编码 0**:Town10HD_Opt pts[0] 固有 yaw=0.16° 恰好成立;Town13 pts[0] yaw=125.9° → 车道线采样沿 lane 方向走到车后、overlay 全空(投影深度全负,数值诊断发现) | 锚定改用 **spawn point 固有 rotation**(地图作者设定的沿车道朝向);原图回归行为不变,Town13 overlay 恢复(988-1108 标注像素/帧) |
| C23 | **数 overlay 绝对颜色会误判**:场景自带绿(植被)/黄(标线)与类别色撞色——top 视角曾报"Car 仅 3 px" | 改 **同帧 raw vs overlay 差集**(`view_stream.py --dump`);另 35m 高度只覆盖 ±20m(22/30m 的车在画面外),提到 60m 后 Car 603 px 与框周长量级吻合 |
| C24 | **P2 "世界 0 信号 actor" 结论有误**(Plan.md §5.6a 原文):Town10HD_Opt 实测 **15 个 traffic_light actor**(Red/Green,opendrive id 943-962,与 Signal landmark 位置重合 0.1m) | xodr 实证:21 条 `<signal>` = 17 个 `dynamic="yes"`(15 命名 + 2 无名)+ 4 个 `dynamic="no"`(Stop/Yield);CARLA 只实例化动态信号。静态源仍用 landmark(更细),灯色属动态 GT 不入 P2 json;可视化侧 actor API 直读画灯色 |
| C25 | **灯态 GT 圆形 horizon 收进大量身后灯**:90 帧视距内 1046 灯次,79% 在 ego 车后(前方仅 221、画面内 131)——与本车决策/图像无关 | 加前向半平面过滤 `traffic_light.in_front`(默认开)→ 221 灯次;正侧方归"不在前方"(cos(radians(90))=6.1e-17 是浮点刀口,不为此加 epsilon) |
| C26 | **"渲染不随 set_state 变"是误判**:相机放在灯头盒 **yaw 方向**两侧,拍到的是黄色灯箱背面(三态图亮度均值 126.53 完全相同) | 镜片法向 = 盒**较薄那条局部轴**(extent 小的轴),不是盒 yaw;按薄轴放相机后拍到 **红上/黄中/绿下** 依次点亮(带网点发光纹理)→ 渲染确实跟随 set_state |
| C27 | **按像素颜色验证灯态在 ego 视角不成立**:12–30m 处"最饱和像素"恒为 (255,237,0) 纯黄,一致率 23% 假象——采到的是**黄色灯箱外壳**,与黄灯镜片同色相;镜片直径 0.2m 在 f=621 下 30m 处仅约 4px | 放弃视觉回归,灯态 GT 定位为**逻辑层**(真值取自 actor API);另 `elapsed_s` 实测红灯相位恒 0、绿/黄相位自相位起点计时 → 变灯时刻以状态序列变化点为准 |
| C28 | **同步模式首个 `get_actors()` 返回空**(连到已 sync 的服务器时,快照只在 tick 后刷新)→ 采集器"清场"循环静默漏清,残留 ego 阻塞 pts[0] 即复现 C8 的起点失配 | 统一修在 `carla_common.sync_mode()`:apply_settings 后补一次 tick(4 个采集器 + view_stream 同时受益) |

| C29 | **"定速"名不副实**:collect_ab_route 放静置车时设 `brake=1.0` 站定,进采集循环**未解除**——VehicleControl 每步生效,`set_target_velocity` 被抵消(probe:命令 4/8/12 → 实际 2.60/6.59/10.59 = 0.65/0.82/0.88×) | 循环前 `apply_control(carla.VehicleControl())` 清残留;P1 四个老数据集实为 **6.60 m/s**(标称 8.0)→ TTC 用标称值偏小 18%,已改用 `closing_speed_series` 逐帧自证 |
| C30 | **误判"2 秒加速段"**:据 ego-x 打印(误设起点 x=-60,实为 -64.7)推断 set_target_velocity 有长 ramp,写进了注释 | 逐帧速度序列实测 1–2 帧到稳态(无 ramp);标签改"慢速帧";教训:**时序结论必须逐帧测,不能从首末值反推** |
| C31 | **CARLA 无运动模糊**(平台边界):4/8/12 m/s 同距离箱梯度能量 35.6/35.2/34.8(10-20m)、61.6/61.0/60.8(30-40m),池化检出率 0.914/0.886/0.909 | 合成渲染逐帧静态曝光,速度**不改变图像质量**;速度退化只能人工注入(与"雨雾无 LiDAR 回波"同类),勿把速度当 corner case 采集维度 |
| C32 | 漏检归因**先入为主怀疑"暗"**:实测漏检框内亮度与命中几乎相同(day 118 vs 134、rain 33 vs 37) | 主因是**尺度**:<32px 一律 0.15-0.47、≥32px 一律 0.78-1.00;断崖 ≈21-24px(30-40m)。诊断口径:先看框高箱,再看亮度/对比度 |

| C33 | **端口守卫静默失效**:本机未装 `ss`(`command not found`),`carla_server.sh start` 里的 `ss -tlnp \| grep ":2000 "` 恒为空 → "端口仍被占用"检查从未生效(排查时也被误导成"无监听",实际端口可连) | 改用 `port_busy()` = `python -c` 的 `socket.connect_ex` 探测(0=被占用);`status` 同时显示端口占用;验证:临时监听端口 → rc=0"被占用"、退出后 → rc=1"空闲" |

## 评估口径

| # | 坑 | 修复 |
|---|---|---|
| V1 | **AP 尾部注水(重要)**:旧尾行 `ap += (1-prev_r)*prev_p` 把未达 recall 段按最后 precision 计入——雨夜检出 0.48 却报 AP 0.976 | 11 点插值(与 compare.ap11 同口径,尾部=0);P1-3 数字重算 Δ-0.020→-0.014 |
| V2 | 无 GT 的类稀释 mAP(行人 GT 稀疏) | eval_2d_ab 只对有 GT 的类求 mAP |
| V3 | **TTC 箱不可跨速度比较检出率**(同一 4.5-6s 箱在 4/8/12 m/s 对应 18-24/36-48/54-72m,实测 0.96/0.13/无样本——尺度就不同) | 跨速度/天气比检出率用**距离/框高箱**(图像几何量);TTC 箱只用于把距离断崖换算成安全余量(attribution docstring 已写明) |
| V4 | **纯 Python 主线程会把 worker 线程饿死(GIL)**:studio 主线程每 tick 的 overlay/拼图/HUD 是字节码,持 GIL 不放 ⇒ 同一对点云 ICP 在 worker 线程 eff **0.04–0.24** vs 主线程同步 **0.90–1.00**(studio 实测 `process_time` 12.9 s vs wall 2.4 s) | 判据用 **`time.thread_time()/wall`(eff)**,不是 wall 单值;`OPENBLAS_NUM_THREADS=1` **不改结论** ⇒ 与 BLAS 线程池无关;施负载必须用**纯 Python 自旋**(PIL/`world.tick()` 会释放 GIL,测不出)。处置:在线 SLAM 默认**同步执行**(`SlamWorker(sync=True)`) |
| V5 | **"有界丢旧队列 ⇒ 有界滞后"对 ICP 是假的**:队列有界的是**深度**不是 `prev_down` 与当前帧的**间隙**,ICP 成本随间隙爆炸(0.8 m 0.2 s → 8 m 2.8 s → 32 m 39 s)⇒ 丢帧→间隙更大→更慢→更多丢帧**无界正反馈**(异步实测:259 tick 只处理 8 帧、滞后单调涨到 157 帧/45 s) | 按**帧号差止损**(`max_gap`):超阈帧不做 ICP、直接恒速外推,`prev_down` 照推进 ⇒ 下一帧回到 gap 1。**滞后口径 = 已 tick 帧号 − 已处理帧号**,不是 `n_offered − n_processed`(后者随丢帧无界增长,是假故障) |
| V6 | **"地图点在 ego 系窗内占比 ≈100%"是误读**:地图覆盖整段行程(~300 m),BEV 窗口只有 30×60 m ⇒ 占比 0.24 属正常 | 真正的自证是「**画出的点数 = 窗内点数**」(两处独立算:`bev_points` 内部过滤 vs 调用方 `bev_window_mask` 重算),相等才说明窗外点一个都没画上 |
| V7 | **"离线 ICP 0.78 s/帧"是平均值不是逐帧成本**:`icp_stats.json` 的 0.78 = 400 帧含转弯/重访的均值,在线逐帧(gap 1)只有 **0.15–0.35 s** | 引用成本数字必须写清口径(总均值 vs 逐帧稳态);据此下的架构结论(「必须 worker 线程」)会整条走偏 |

## 实时可视化 / 拼图

| # | 坑 | 修复 |
|---|---|---|
| R1 | **`PIL.Image.paste` 在源图大于目标框时不报错、不缩放,只贴左上角**(超出部分静默丢弃):studio 旧 4×2 等尺寸拼图(`disp_w=621`)把 1242×375 相机图裁成 621×187,右半 + **下半(地面)** 无声消失;用户看到的现象是"6 视角 FoV 缩得看不到地面"(**不是 FoV 变了,是画面被切走一半**) | 判据不看图看数:拼图格与「源图左上角裁剪」平均绝对差 **0.128** vs 与「整幅缩放」差 **66.18** ⇒ 裁剪不是缩放。修复 = `live_common.compose_rows`(按行拼、每格**原生像素**)+ `compose_grid` **尺寸守卫**(不符即 `ValueError`,把这类坑钉死不复发)+ `live_studio.GRID_ROWS` 三层。回归 `tests/test_live_common.py`(12 用例,核心判据 = 每格逐像素等于源图) |
| R2 | **`bin/` 不是包,静态分析跟不到测试里的运行时 `sys.path.insert`** → pyright 报 4 处 `reportMissingImports`(而全仓 pyright 基线本来就有 7580 错,`tests/` 一条没有) | 就地 `# pyright: ignore[reportMissingImports]` 标注 import 行(不改全局 pyright 配置,不给仓库引入新文件);判据 = `pyright tests/test_live_common.py` → 0 errors |

## 标定自证 / 实时监看

| # | 坑 | 修复 |
|---|---|---|
| K1 | **rig 镜像:`yaw_carla = −az_nus` 漏翻** —— 旧 `official` rig 把官方方位角原样抄成正数,pitch/roll 还硬编码 0 ⇒ 四个侧/后相机左右互换(FRONT_LEFT/RIGHT 差 **110.3°**、BACK_LEFT/RIGHT 差 **217.2°**);**前/后相机因近自逆"看着对",长期没暴露** | 真值单点提供:`camera_rig.NUS_CAMERA_RIG`(官方 `calibrated_sensor` 四元数导出,**导出前必须归一化**——官方存储的四元数不是单位长度),采集器/实时流/导出器同源。**判据 = A1 侧别一致性(挂点 y 与光轴 y 同号)**:镜像 rig 会让 4/4 判 false,能当场拦下。回归 `tests/test_probe_calib.py`(含"A1 能拦下历史镜像 bug"的反向自证) |
| K2 | **像素约定 = corner 还是 center?** 差 0.5 px 在近处就是米级深度误差 | A3(LiDAR 平面 × 深度图交叉验证)裁决:CARLA 渲染栅格 **索引 i 的连续坐标恰为 i** ⇒ corner。median \|e\| **0.0003 m**(corner) vs **0.023 m**(center),**约 70×**,六相机一致;A4 用**实例掩膜索引**中点独立测得 `fx = 621.6 px`。`cx = (w−1)/2 = 620.5` 与 `fx = (w/2)/tan(fov/2) = 621.0` **并存不矛盾**(前者索引约定中心、后者"半 FOV↔半宽")。**角色分类防再犯**:采样 CARLA 栅格 ⇒ 必须 corner;采样 torch 栅格(FPN/`align_corners=False`/gsplat)⇒ `(u+0.5)/W*2−1` **正确**;PIL 纯绘制 ⇒ 无关;读内参 ⇒ cx/cy **从 K 直读**不重算。回归 `tests/test_depth_codec.py` + `tests/test_calib_live.py::test_index_equals_coordinate` |
| K3 | **自遮挡判据写死绝对阈值 ⇒ 换分辨率静默失效**:`self_occluded` 用 `near_fraction > 0.2`,该阈值在 1242×375 下成立(CAM_BACK 实测 **0.367**),在 640×360 下**判不出来**(实测 **0.195**) | 根因:**近场占比随画幅宽高比变**(水平 FOV 都是 90°,640×360 竖直 FOV 大得多 ⇒ 车顶占画面比例小)。修复 = **相对判据** `calib_live.self_occluded_cameras`:基准取**同批可用相机**近场占比中位数(典型 0.000),阈值 `max(NEAR_FRACTION_MIN=0.05, NEAR_FRACTION_RATIO=10×基准)`;全部不可用时基准退回 0.0(没有参照也要给结论,不沉默)。**不按相机名硬编码**。回归 `TestSelfOccluded::test_flags_at_the_live_resolution_fraction`(0.195 必须判得出) |
| K4 | **样本少 ≠ 标定坏**:CAM_BACK 挂点只比 ego 自身车顶高 **0.023 m**(官方 z=1.5791 vs 车顶 ≈1.556)⇒ 可用样本常年 0–30(其余 50–200),但**残差中位数与其它相机同级**(0.0003 m) | 判据做成"**样本 < `MIN_CAM_SAMPLES` ⇒ `median_abs = None`**",HUD 报"无数据"而**不是** 0.000(0.000 会被读成"标定完美")。回归 `TestSummarize::test_below_threshold_reports_none_not_a_number` + `TestHudLine::test_no_data_says_so_instead_of_zero` |
| K5 | 实时槽平面重拟合成本:逐点邻域平面是 **O(N²)**,实测 **~100–150 ms/tick**,而六相机采样合计仅 **~9 ms** | 瓶颈全在拟合 ⇒ `--calib-refit` 默认 2(每 2 tick 重拟合)。**平面是"世界系"的**(描述场景表面,不是"这一帧的点云")⇒ ego 移动几米后同一块路面仍是同一平面,可跨 tick 复用;被挡住的点由单侧可见性判据剔掉。**抽样必须在拟合之前**(逐点独立 ⇒ 抽样不改变任一保留点的判定;先拟合再抽会把"哪些点过闸"交给运气)。可靠最低配置 = voxel 1.0 / 半径 2.0 / dist<20 / 上限 1500(半径 1.5 或更低 ⇒ 邻域低于 `MIN_PLANE_PTS=12`,**0 个合格点**) |

## 类型系统(pyright 0 纪律)

| # | 坑 | 修复 |
|---|---|---|
| T1 | carla pyi 桩:`try_spawn_actor` 标返回 `Actor`(实为 `Actor\|None`) | 判 None 后 `cast(carla.Vehicle, v)` 再调 Vehicle 方法 |
| T2 | Vector3D 运算结果不能直接进 `carla.Transform` | 显式 `carla.Location(x=..., y=..., z=...)` |
| T3 | 变长 tuple 推导不能赋给 `tuple[float, float, float]` 形参 | 显式三元组 `(float(a[0]), float(a[1]), float(a[2]))` |
| T4 | ultralytics `model.predict` 返回 union(Iterator\|list) | `cast(Results, list(...)[0])` 物化取首帧 |
| T5 | `res.boxes` 可 None | `res.boxes or []` |
| T6 | oracle 测试命名空间包致 importorskip 失效 | 判定改用具体子模块 |

## 工具链集成

| # | 坑 | 修复 |
|---|---|---|
| G1 | `python -m auto3dlabel` 无 `__main__` | 入口是 console script `auto3dlabel run` |
| G2 | 检测指令 "car" 报 ValueError | CN_EN_MAP 只认中文,须 "检测汽车" |
| G3 | auto3dlabel config FileNotFoundError | config 相对路径,**cwd 必须在 AutoLabel 根目录** |
| G4 | oracle 判等"肉眼差不多"实则有差 | 全部改往返断言(双向转换一致性) |

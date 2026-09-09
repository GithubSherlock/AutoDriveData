# 测试与踩坑日志

开发踩坑与测试记录(现象 → 根因 → 修复 → 回归保护)。里程碑叙事见 [milestone.md](milestone.md),方案细节见 Plan.md(单一事实源)。**代码里带回归注释的坑勿重犯。**

## 测试基线

- **base env** 单测(纯数学,不依赖 carla/auto3dlabel):**131 passed**(2026-09-08)
- **autolabel env** oracle 脚本:import auto3dlabel 当单一事实源做往返断言(KittiCalib 投影链、GT 往返、devkit 构造、NusBox 往返)
- 测试环境策略见 Plan.md §5.6:纯数学单测 base、oracle autolabel,两边互不污染

## 环境 / 平台

| # | 坑 | 修复 |
|---|---|---|
| E1 | UE4 拒绝 root 运行 | 专用用户 carla 跑服务器(`scripts/carla_server.sh`) |
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

## 评估口径

| # | 坑 | 修复 |
|---|---|---|
| V1 | **AP 尾部注水(重要)**:旧尾行 `ap += (1-prev_r)*prev_p` 把未达 recall 段按最后 precision 计入——雨夜检出 0.48 却报 AP 0.976 | 11 点插值(与 compare.ap11 同口径,尾部=0);P1-3 数字重算 Δ-0.020→-0.014 |
| V2 | 无 GT 的类稀释 mAP(行人 GT 稀疏) | eval_2d_ab 只对有 GT 的类求 mAP |

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

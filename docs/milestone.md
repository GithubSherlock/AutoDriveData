# AutoDriveData 里程碑

CARLA 0.9.16 → AutoLabel 数据输出流水线的迭代记录。**单一事实源 = 根目录 Plan.md**(方案定案/执行细节/待办);本文档 = 里程碑时间线与验收结论速览;踩坑与测试记录 = [testLog.md](testLog.md)。

## 时间线概览(2026-09-06 → 2026-09-08)

| 里程碑 | 日期 | 验收 |
|---|---|---|
| M0 环境 | 09-06 | ✅ 0.9.16 prebuilt + headless + GPU 修复栈 |
| M1a KITTI 静态一帧闭环 | 09-07 | ✅ 数据层零改动入 AutoLabel,PointPillars 出伪标签 |
| M1b nuScenes 语义 | 09-07 | ✅ devkit 直读 + nuscenes-queue 分派通 |
| M2 动态采集 + 比对闭环 | 09-07 | ✅ 工具链(分歧率 <10% 未达,如实记录) |
| M3 微调排查 | 09-07 | ✅ 工具链(微调 3 败塌缩,生产配置定论) |
| P1 Corner Case 场景矩阵 | 09-08 | ✅ 三 corner case A/B 定量 |
| P2 静态 GT | 09-08 | ✅ 地图查询 API + 目检验收 |
| M4 定制街道 | 09-08 | ⏸ 降级挂起(源码构建成本过载) |
| 地图池扩展 AdditionalMaps | 09-08 | ✅ Town11/12/13/15 加载 + 新地图采集 |

## M0 环境(2026-09-06)✅

- 选型 **0.9.16** prebuilt(19G,`/root/autodl-tmp/CARLA_0.9.16/`),pycarla 装 base env(3.10)
- headless 适配:专用用户 carla(UE4 拒绝 root)+ 离屏渲染 + GPU 修复栈(EGL json 重建 + 符号 shim)
- 验收:smoke 单帧 raw 落盘 + 可视化

## M1a KITTI 静态一帧闭环(2026-09-07)✅

- 交付:autodrivedata 四纯值模块 `geometry`(坐标转换唯一落点)/ `calib` / `gt` / `export/kitti` + `collect_kitti` 静态采集(ego 静止 + 摆 NPC)
- 集成验收:`KITTI_OBJECT_ROOT=outputs/kitti_scene auto3dlabel run` 数据层**零改动**读入,PointPillars 伪标签落盘(采纳 0/复核 3);点云投影全部落入 GT 2D 框
- 域差距首测:合成帧真车全漏、建筑立面误报(强度通道退化 std 0.055)——修复归 M2
- 测试:72 base + 5 oracle(autolabel env 往返断言)

## M1b nuScenes 语义(2026-09-07)✅

- 交付:geometry 补 nuScenes 约定(CARLA_TO_NUS/yaw↔quat)+ `export/nuscenes` 14 表迷你集生成器 + `collect_nus`(6 相机 + LiDAR)
- 验收:devkit 直读、nuscenes-queue 分派通(复核 5,域差距同 KITTI 侧)

## M2 动态采集 + 比对闭环(2026-09-07)✅(带已知缺口)

- 交付:`collect_drive`(autopilot + TM 车流)+ `compare` 比对层(Sutherland-Hodgman IoU + 贪心匹配 + 11 点 AP)+ `eval_kitti` + `semantic.py`
- **语义强度合成修复域差距**(6 变体实验):密度/噪声/丢点均无效,反照率×入射角强度合成 → Car AP **0 → 0.529**(决定性修复)
- 150 帧动态实测:Car AP 0.294(TP 118/233)、复核率 100%;**分歧率 <10% 未达**——根因:伪标签 FP 高、动态更难、行人 GT 缺失(如实记录,归 M3)
- 已知限制:headless teardown 偶发 segfault(数据不丢);语义 LiDAR 行人 tag=0

## M3 微调排查(2026-09-07)✅(微调质量未达标,如实记录)

- 交付:`finetune_synth.py`(合成 KITTI → train3d 五步自动化;~5 分钟/轮收敛正常)
- 微调 3 次全部塌缩:驱动帧 40 点 easy AP 官方 78.3 → 9.5/5.5/0.0;2 个毒化源已修(行人 NAV 失败原点聚集 528 GT、远距无点 GT 163m>70m 量程),1 个疑似(灾难性遗忘,待查)
- **生产配置定论**:官方 pointpillars_kitti + 语义强度合成 + max_distance=65 GT 过滤

## P1 Corner Case 场景矩阵(2026-09-08)✅

- P1-1 `scenarios.py`:8 场景目录 + 可模拟性矩阵(如实标注:无 flare/镜头光学、无雪、雨/雾对 LiDAR 无物理回波)
- P1-2 `collect_drive --scene` 场景档采集
- P1-3/4/5 三 corner case A/B 定量(`collect_ab_route` + `eval_2d_ab`,帧级配对:锚定 spawn point 0 + 4 静置车 + 只变 weather):

| corner case | 相机 Δ AP | 掉点型 | LiDAR Δ AP |
|---|---|---|---|
| 逆光(sunset_glare, az=90) | -0.014 | 轻掉点(AE 补偿) | +0.003 噪声 |
| 雨夜(rain_night) | **-0.153** | 漏检型(检出 0.72→0.48) | +0.017 噪声 |
| 浓雾(dense_fog) | -0.013 | FP 型(检出 0.72→0.85) | 0.000 |

- 结论:相机在三型真实驾驶长尾上均有可量化掉点;LiDAR 兜底不受天气/光照(平台边界:雨/雾无物理回波,退化只能人工注入)

## P2 静态 GT(2026-09-08)✅

- 选型裁决(probe 实测):semantic LiDAR 后处理**出局**(信号灯非 actor、车道线非实体,LiDAR 打不到);RoadRunner 资产化挂起 → **地图查询 API 定案**(landmark 65 个 + waypoint lane_marking)
- 交付:`autodrivedata/static_gt.py` 纯值格式(StaticFrame:signals + lane_lines)+ `collect_static_gt.py`(锚定 pt0 定速直行,65m 视距),落盘 `training/static_gt/{fid}.json` + overlay 目检图
- 验收:人工目检通过——信号锚点在真实灯杆基座处、车道线双线透视收敛正确、跨帧一致性 ✓
- 边界:静态 GT 与天气/光照解耦(换地图即换真值,M4 复用本链路);灯色属**动态** GT 不入静态 json(2026-09-09 更正:Opt 图**有** 15 个灯 actor,原记"无 actor"有误,见下)

## M4 定制街道(2026-09-08)⏸ 降级挂起

- M4-0 调研:prebuilt 无 UnrealEditor/Import 链、PythonAPI 无运行时 xodr 入口 → 定制地图**必须 CARLA 源码构建**(~170G 磁盘 + Epic GitHub 授权 + 天级编译 + GPU 驱动栈回归)
- 用户裁决:对"一条演示街道"收益过载 → **降级为扩展 P1**;重开条件 = 换盘/有需求(按 Plan.md §5.7 沉淀路线)

## 地图池扩展 AdditionalMaps(2026-09-09)✅

- 官方附加包 14.8G(断点续传下载,零构建扩地图池)→ 13+4 = 17 图
- Town11/12/13/15 `load_world` 全部通过(15s/50s/123s/188s);默认图仍 Town10HD_Opt
- 采集验证(probe 分层定位):
  - Town11/12 **禁采集**:spawn camera 瞬间 segfault(渲染资源与 headless GPU 栈冲突,机制未明)
  - Town13 TM 车流过载(153% CPU 无响应)→ 降级 0 NPC 可跑;静态 GT 10 帧 + 动态 12 帧通过
  - **锚定 yaw bug 已修**:硬编码 yaw=0 在 Town13(pts[0] 固有 yaw 125.9°)致车道线采样到车后、overlay 全空 → 改用 spawn point 固有 rotation,原图回归不变、Town13 overlay 恢复
- 结论:新图采集用 Town13/15;静态 GT 地图查询链路新图自动生效

## 可视化实时流(2026-09-09)✅

- 决策:carlaviz(Three.js 线框)/ ROS2-Bridge+RViz2 出局——**都不是 UE 渲染**(P1 验证对象全是渲染效果)、carlaviz 官方最高 0.9.15(无 0.9.16)、容器无 docker/ROS;自建 MJPEG 更省且口径同源
- 交付:`scripts/view_stream.py`(follow/top/grid6 三视角 + `--scene` 天气 + `--npcs` 静置 NPC + GT 框/灯色 overlay + HUD,MJPEG 只绑 127.0.0.1 走 SSH 隧道);`world_to_img` 上移 `calib.py` 供采集器与实时流共用(+5 单测)
- 验收(数值诊断,同帧 raw/overlay 差集):follow 3023 px、top(60m)1995 px,类别色全部命中;流 30+ 段 JPEG 有效、约 5 fps、退出清理干净
- 顺带更正:Town10HD_Opt **有 15 个 traffic_light actor**(xodr 17 个 dynamic 信号,15 个被实例化)→ P2 "无灯色状态"边界撤回

## 灯色动态 GT(2026-09-09)✅

- 工业口径:灯态 = 独立时序语义层(BDD 挂框属性无时序 / WOMD 逐帧序列但 71.7% 缺失 / nuScenes 地图层只有静态几何)→ 本项目取 WOMD 形态,Off/Unknown **不猜**
- 交付:`autodrivedata/traffic_light.py`(纯值:normalize_state / in_front / phase_at / Frame JSON 往返)+ `scripts/collect_tl_states.py`(记录模式 / `--cycle 6,2,6` 受控切灯)+ `carla_common.traffic_light_frame`/`draw_traffic_lights`(采集器与实时流共用)
- 落盘:KITTI root 扩展 `training/traffic_light/{fid}.json` + `image_2/` + `overlay/`
- 验收(受控 90 帧):状态变化点 = 帧 0/60/80 与计划逐帧吻合;管制车道/停车线非空;overlay 差集与画面内灯数相关 0.99;view_stream 回归差集 3522 px
- 实测驱动的两处修正:①圆形 horizon 收进 79% 身后灯 → 加前向半平面过滤(1046→221 灯次);②同步模式首个 `get_actors()` 为空致清场漏清 → 修在 `sync_mode()`
- 平台事实:渲染确实跟随 set_state(探针拍到红上/黄中/绿下点亮),但镜片 0.2m 在 30m 处仅约 4px + 黄色灯箱外壳同色相 → **不做视觉回归**,灯态 GT 定位为逻辑层;`elapsed_s` 红灯相位恒 0,变灯时刻以状态序列变化点为准

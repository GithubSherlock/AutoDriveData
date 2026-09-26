# AutoDriveData

CARLA 0.9.16 → AutoLabel 自动驾驶数据输出流水线:自定义地图/场景采集车检测动态/静态目标与道路特征,构建 0→1 数据输出。当前主线 = **Corner Case 场景矩阵定量验证**(P1)+ 静态 GT(P2);raw 输出供 AutoLabel 验证场 + 长尾数据源。

## 当前进度(2026-09-23)

- **P1 ✅**(2388336 起):场景目录 [autodrivedata/scenarios.py](autodrivedata/scenarios.py)(8 场景,不 import carla)+ `collect_drive --scene` 采集链 + 三 corner case A/B(统一 11 点插值 AP 口径,见 [Plan.md](Plan.md) §5.7c 汇总表):
  | corner case | 相机 Δ | LiDAR Δ |
  |---|---|---|
  | 逆光(az=90 校准) | -0.014 轻掉点(AE 补偿) | +0.003 噪声 |
  | 雨夜 | **-0.153 漏检型**(检出 0.72→0.48) | +0.017 噪声 |
  | 浓雾 | -0.013 FP 型(检出 0.72→0.85) | 0.000 |
  - 结论:相机三型可量化掉点;LiDAR 兜底不受天气/光照(平台边界:雨/雾无物理回波,退化只能人工注入)
- **P2 ✅**(f66558c):静态 GT = **地图查询 API**(信号/标志是 landmark、车道线是 lane_marking 实体;semantic LiDAR 打不到)→ [autodrivedata/static_gt.py](autodrivedata/static_gt.py) + [autodrivedata/sim/collect_static_gt.py](autodrivedata/sim/collect_static_gt.py),落盘 `training/static_gt/{fid}.json` + overlay 目检图。**更正**:Town10HD_Opt 有 15 个 traffic_light actor(xodr 17 个 dynamic 信号),原记"无信号 actor"有误——灯色属动态 GT,仍不入静态 json
- **M4 挂起**(§5.7a 用户裁决):定制街道 = CARLA **源码构建**(prebuilt 无 UnrealEditor,~170G 磁盘/Epic 账号),成本过载降级为扩展 P1。开源 AdditionalMaps(Town11-15,14.8G)已探明可下载,零构建扩地图池
- **地图池扩展 ✅**(§5.7d):AdditionalMaps 14.8G 已装,Town11/12/13/15 入池(17 图)。**新图采集约束:Town11/12 禁采集**(spawn camera segfault)、Town13 TM 车流降级 0 NPC、可用 Town13/15;默认图仍 Town10HD_Opt(重启即恢复)
- **可视化实时流 ✅**(§5.8):[autodrivedata/sim/view_stream.py](autodrivedata/sim/view_stream.py) 自建 MJPEG(3 视角 + GT 框/灯色 overlay,只绑 127.0.0.1 走 SSH 隧道);carlaviz/RViz2 出局(非 UE 渲染 + 版本/依赖不成立);`world_to_img` 上移 [autodrivedata/calib.py](autodrivedata/calib.py) 供采集器与实时流共用
- **灯色动态 GT ✅**(§5.9):[autodrivedata/traffic_light.py](autodrivedata/traffic_light.py)(纯值:状态归一/前向判据/相位查表/JSON 往返)+ [autodrivedata/sim/collect_tl_states.py](autodrivedata/sim/collect_tl_states.py)(记录模式 / `--cycle 6,2,6` 受控切灯 → 确定性变灯序列),落盘 `training/traffic_light/{fid}.json`。工业口径:灯态 = 独立时序层,Off/Unknown **不猜**;受控 90 帧状态变化点 = 帧 0/60/80 与计划逐帧吻合。**边界:不做视觉回归**——镜片 0.2m 在 f=621 下 30m 处仅约 4px,且黄色灯箱外壳同色相
- **P1 参数扫描 + 失效归因 ✅**(§5.10):[autodrivedata/attribution.py](autodrivedata/attribution.py) 纯值(逐帧匹配/分箱/逐帧速度自证)+ [bin/eval_attr.py](bin/eval_attr.py)(多跑 × 距离/框高/TTC 网格 + 漏检画像),与 AP 共用同一 `box_iou2d`。三结论:**尺度主导**(<32px 0.15–0.47 / ≥32px 0.78–1.00,断崖 ≈21–24px)、**CARLA 无运动模糊**(4/8/12 m/s 梯度能量 35.6/35.2/34.8,检出率 0.914/0.886/0.909 → 速度不改图像,退化只能人工注入)、**天气只前移断崖**(雨夜 40-50m 零检出→30-40m 0.32,雾最晚 0.91)
- **MapTR 矢量管道 ✅**(§5.11):A 阶段矢量库(opendrive/mapvec + A6 oracle 0.00cm)→ B 阶段环视采集/组装/投影验收 → C 阶段**参考自实现**(`maptr_impl/`:GKT + 分层 query,单帧过拟合锚定正确性)→ D 阶段 chamfer AP。训练数据 200 帧(Town10HD_Opt@spawn0)
  - **第二轮 ep512 结果口径分化**:留出集 @0.2 **0.0674**(vs ep256 0.0510,+32%)、@0.3 0.0904、@0.4 0.1280(后两档持平略降)→ 保守操作点仍获益、高阈值已饱和;训练对照 0.2603 → **泛化间隙 3.9×**(ep256 时 2.1×)→ 下轮收益靠**扩数据**而非继续长训。权重 `outputs/maptr_ep512.pt`
  - **实时 overlay ✅**(§5.11f):`view_stream.py --maptr-ckpt outputs/maptr_ep512.pt --maptr-bev`(rig 走 `live_common.rig_spec` 两处定义;推理用实挂相机世界位姿);实况数值验证 overlay 品红 25553 px vs raw 0、地平线以上 0/18496、FPS 1.1–1.4
  - **rig 两代并存(2026-09-19,§P-L.1)**:`legacy`(共用 `SENSOR_OFFSET` + BACK_LEFT/RIGHT 235/125)对 `maptr_ep256/ep512`;`nuscenes`(逐相机 `SENSOR_MOUNTS` + 官方 6DoF 姿态)对 `maptr_600/1000`(**该两代权重均已于 2026-09-22 标废弃,见下条**)。**rig 必须与权重训练数据一致,不是"越新越好"**;`--rig auto`(默认)按权重名选。A/B 探针 `bin/probe_rig_mount.py`
  - **⚠️ 全部 MapTR 权重标废弃 + GKT 两处 bug(2026-09-22,§P-M)**:①**rig 镜像**——旧 `official` rig 的偏航是官方方位角原样抄的正数,漏了 `yaw_carla = −az_nus` ⇒ 四个侧/后相机左右互换(FRONT_LEFT 差 110.3°、BACK_LEFT 差 217.2°),pitch/roll 还硬编码 0;前/后相机因近自逆而"看着对",长期没暴露。真值改由 [autodrivedata/camera_rig.py](autodrivedata/camera_rig.py) 单点提供(官方四元数导出,`NUS_CAMERA_RIG`),采集器/实时流/导出器同源;`official` 更名 `nuscenes`。②**GKT 两处 bug**——infos 六元组是 `[x,y,z,yaw,pitch,roll]`,而 `_carla_rotation_torch` 要 `(pitch,yaw,roll)`,漏换序 ⇒ 5/6 相机指向错(yaw≈0 的 CAM_FRONT 看不出异常,单测 oracle 复刻了同一个错);K 未缩到特征图分辨率(FPN P2 = 311×94)⇒ 全分辨率像素与 `feat_w−1` 比。两坑叠加把 BEV 有效覆盖从 **94.6% 打到 1.25%**,head 采样 100% 零单元(实测 `head(真 BEV)` vs `head(零 BEV)` L1 仅 1.51 m)。修复后 `maptr_ep256/ep512/600/1000` **全部废弃**(采集数据本身就错),重采重训;`maptr_impl/gkt.py` 的 `_ROT_TO_CARLA` + `scale_k` + `tests/test_gkt.py` 三条回归钉死
- **8 路实时 studio ✅**(2026-09-19 A 期 / 2026-09-20 B 期,Plan2.md §P-L):`autodrivedata/sim/live_common.py`(共享件:单端口多槽 MJPEG `/stream/<name>` + `/` 索引页 / 拼图 / GT overlay / 环视 rig / 第三方视角 / `KeyboardState` / MapTR 懒加载)+ `autodrivedata/sim/live_studio.py`(9 槽 = 6 相机 + `BEV` + `THIRD_PERSON` + `grid` 拼图;`--keyboard` 折进 tick 循环)。`view_stream.py`/`drive_ego.py` 改薄编排
  - **拼图三层 + 不缩像素 ✅**(§P-L.6):用户报告"6 视角 FoV 缩得看不到地面"——根因是 **`PIL.Image.paste` 源图大于目标框时只贴左上角、不报错不缩放**,旧 4×2 等尺寸拼图把 1242×375 裁成 621×187,右半 + **下半(地面)** 无声丢弃(判据:与「源图左上角裁剪」差 **0.128** vs 与「整幅缩放」差 **66.18**)。改 `compose_rows`(按行拼、每格**原生像素**)+ `compose_grid` **尺寸守卫**(不符即 `ValueError`,钉死不复发)+ `GRID_ROWS` 三层(①左前/前/右前 ②右后/后/左后 ③第三方 + BEV,**不沿用 `SURROUND_CAMS` 字典序**)。画布 2484×374 → **3726×1170**,逐格与源图最大差 **0.0**;回归 `tests/test_live_common.py`(12 用例)
  - **B 期在线 SLAM ✅**(§P-L.2~P-L.4):`autodrivedata/live_slam.py`(`LiveSlam.push/snapshot` + 地图/轨迹换到当前 ego 系 + `SlamWorker`)+ `live_studio --slam`(语义 LiDAR → BEV 槽画地图点灰/轨迹青,HUD 显式报滞后)。**两条原前提都被实测推翻**:①离线 ICP 0.78 s/帧是 400 帧**含转弯的平均值**,在线逐帧只有 0.15–0.35 s;②**worker 线程被 GIL 饿死**(主线程 overlay/拼图/HUD 是纯 Python 字节码;同一对点云 worker eff 0.04–0.24 vs 主线程同步 0.90–1.00;钉 `OPENBLAS_NUM_THREADS=1` 不改结论 ⇒ 不是 BLAS 线程池)⇒ **默认同步执行**(~2.4 fps),`--slam-async` 留作对照
  - **有界丢旧队列 ≠ 滞后有界**(关键机制):队列有界的是**深度**不是 `prev_down` 与当前帧的**间隙**,而 ICP 成本随间隙爆炸(0.8 m 0.2 s → 8 m 2.8 s → 32 m 39 s)⇒ 丢帧→间隙更大→更慢→更多丢帧**无界正反馈**(异步实测:259 tick 只处理 8 帧、滞后涨到 157 帧/45 s)。修复 = **按帧号差止损**(`--slam-max-gap` 默认 3,超阈帧不做 ICP 直接恒速外推,`prev_down` 照推进)。**滞后口径 = 已 tick 帧号 − 已处理帧号**,不是 `n_offered − n_processed`(后者随丢帧无界增长,是假故障)
  - **验收数字**:400 帧在线 vs 离线链式位姿**逐元素差 0.0**、ATE **0.18768 m**(= 离线 = 参照);同步 136 tick **丢 0/止损 0/lag_max 0**;退出后 `nvidia-smi` 回基线、残留进程 0;BEV 画出 90611 点 / 轨迹 20 段 / **画出的点数 = 窗内点数**。产物 `outputs/slam_gt/{accept_sync,accept_async,accept_parity_full}.json`。**判据订正**:"地图点窗内占比 ≈100%"是误读 —— 地图覆盖 ~300 m 而窗口只有 30×60 m,占比 0.24 正常
  - **逐帧契约 ✅**(§5.11h):`eval_maptr.py --out-frames` → `outputs/surround_pred/{token}.json`(`mapvec_pred/1`:schema/帧归属/类序/坐标系/窗口/阈值/溯源,GT 同文件携带;旧 `--out-pred` 是跨帧汇聚、实例无帧归属,不可对外)。供 AutoLabel 消费——**消费方未接**
- **环视相机标定修正 + 七锚自证 + 实时监看槽 ✅**(2026-09-22,Plan2.md §P-M):按 nuScenes 官方硬件布局重标(时序建图的前置,用户设硬顺序"先修完标定才讨论时序建图")。**七锚探针** [bin/probe_calib.py](bin/probe_calib.py) → `outputs/calib_check/report.json`,判据全数值不目检,A0–A6 **全 true**:A0 光轴 vs 官方方位角 `7.1e-15°` / A1 侧别 4/4 同侧 / A2 四方位锥四对全 match / A3 LiDAR-平面-深度图交叉验证 median \|e\| **0.0003–0.0009 m** / A4 轴目标物掩膜质心 `cx = 620.5`、`fx_est 621.6 px`、残差 max **0.200 px** / A5 实挂 vs 规格 平移 `3.8e-06 m` 偏航 `4.5e-05°` / A6 主点锁定 `(w−1)/2`
  - **★ 像素约定裁决 = CORNER**(主干结论):CARLA 渲染栅格**索引 i 的连续坐标恰为 i** ⇒ `cx=(w−1)/2=620.5`、`cy=(h−1)/2=187.0`。A3 corner `0.0003 m` vs center `0.023 m`(**70×**,六相机一致);A4 掩膜索引中点独立测得 `fx=621.6 px`。`fx=(w/2)/tan(fov/2)=621.0` 与 `cx=620.5` **并存不矛盾**(前者"半 FOV↔半宽",后者索引约定中心)。**角色分类**:采样 CARLA 栅格 ⇒ 必须 corner;采样 torch 栅格(FPN/`grid_sample(align_corners=False)`/gsplat)⇒ `(u+0.5)/W*2−1` **正确**;PIL 纯绘制 ⇒ 无关;cx/cy **必须从 K 直读**不重算(`mapviz.intrinsics_from_k` 旧实现丢掉重算 = 纯缺陷,已修;fov→fx 收敛到 `mapviz.calib_from_fov` 唯一落点)
  - **实时监看槽**(`autodrivedata/calib_live.py` + `live_studio --calib`):另挂 6 深度相机(同挂点/同内参/同分辨率)+ ray_cast ⇒ 世界系平面投影回相机按深度残差着色,`draw_hud` 第二行报 pooled \|e\| 与逐路样本数。验收(77 tick):5/6 相机可用、pooled median **0.00032 m**、时序 0.00033/最大 **0.00044 m**。**成本**:平面拟合 ~100–150 ms vs 六相机采样 ~9 ms ⇒ `--calib-refit` 默认 2;**平面是世界系的**(描述场景表面)⇒ 可跨 tick 复用
  - **CAM_BACK 自遮挡 = 平台边界**(**⚠️ 已于 §P-M.10 订正**:那是**挂点原点修正前**、CAM_BACK 落在车身中部时的读数;修正后重测**六路 `near_fraction` 全 0.0**):官方挂点 z=1.5791 只比 CARLA ego 自身车顶(≈1.556)高 **0.023 m** ⇒ 近场(<0.5 m)占比 **1242×375 下 0.367 / 640×360 下 0.195**,其余五路 0.000。**判据必须是"样本不足 ⇒ `median_abs=None` 报'无数据'"**,不是"median 大 = 坏标定"(它残差中位数 0.0003 与其它同级)。**★ 踩坑**:`self_occluded` 曾写死 `> 0.2` —— 1242×375 成立、640×360 **静默失效**;根因是近场占比**随画幅宽高比变**(水平 FOV 都 90°,640×360 竖直 FOV 大 ⇒ 车顶占比小)⇒ 改**相对判据**(基准 = 同批可用相机近场占比中位数,阈值 `max(0.05, 10×基准)`),不按相机名硬编码;回归钉 `tests/test_calib_live.py`
- **`collect_nus.py` 全传感器标定修正「声明 ≠ 渲染」✅**(2026-09-23,Plan2.md §P-M.7):`collect_nus.py` 曾是「**表对了、图错了**」的新失效模式 —— `calibrated_sensor` 写官方正确值,传感器却 spawn 在另一套位姿(6 相机一律挂 **LiDAR 挂点** `(1.2,0,1.65)` + 旧镜像偏航表 `{0,±55,180,±125}`;5 雷达偏航 `{0,±45,±90}` 猜测值;LiDAR **无 rotation**;六路共用 `fov=90`/`fx=800`)。**这类缺陷只查表全绿、只目检"图能出"也全绿**。修法 = 渲染与声明**由同一份常量导出**:相机走 `NUS_CAMERA_RIG`(逐相机 6DoF)+ 逐通道蓝图 fov;雷达 `−az_nus` 由 `NUS_RADAR_OFFSETS` 导出;LiDAR 走官方挂点 + 由四元数**导出**的 `LIDAR_ROT`(不手抄);内参换逐通道官方 n015 K;`calibrated_sensor` 落盘表 LiDAR/雷达与官方**逐位相同**
  - **六条验收判据全过**(复现器 [bin/verify_nus_calib.py](bin/verify_nus_calib.py) → `outputs/nus_calib_check/report.json`;`--offline` ③④⑤ / `--live` ①②⑥):① 相机实挂 vs 声明 平移 `4.3e-06 m` / 偏航 `3.9e-05°`(修前 0.52–1.17 m、0.15–126.40°)② 雷达实挂 `2.9e-05°`(修前四路差 93.89–135.98°)③ 雷达点落进自身 FOV `0.8607/0.8135/0.8458/0.9076/0.8996`(修前 0.0000/0.0000/0.0011/0.0015)④ LiDAR 复现 `num_lidar_pts` **1.0000**(修前 0.0854;官方锚 1.0000 / 单位阵消融 0.1684)⑤ 内参偏差 **0.0 px**(修前 32–473 px)⑥ 渲染 FOV dev `0.0024–0.0672°`(修前六路共用 90° vs 声明 64.3°)。`smoke_radar_collect.sh` 四判据全过
  - **★ 判据 ③ 的口径**(为什么以官方 R 为锚):点云存的是**传感器自身系**,+x 即光轴 —— 只问"点在自己系里落不落进 ±38.1° 锥"是**恒真**的。必须 `R_official^T @ R_declared @ p_sensor`。**明令禁止 `num_radar_pts` 作判据**(跨 5 通道求和,官方 R 也只复现 0.6279)
  - **★ 判据 ⑥ 两个实测坑**(已写成 `TestFovCriterionShape` 的 AST 静态回归钉):①**每 tick 必须抽干全部相机队列** —— 只取被测相机那一帧会让其余五路积压,轮到它们时读到锥体 spawn **之前**的陈旧帧,症状是"六相机只有第一个测得出、其余全 0 且**与距离无关**"(看着像摆不进去,其实不是);②**Z 不能写死** —— 地图遮挡随 ego 出生点变(实测同一脚本两次跑,Z=20 可用/需退到 Z=10 都出现过)⇒ 改 Z 阶梯 `(20,14,10,8,6)` 取第一档可用样本 ≥6,全档不够如实报 `reason`;横移从固定 5 m 改 **±0.45·Z**(杠杆臂 ∝ `fx·x/Z`)把 CAM_FRONT_LEFT 的 fx 从 1292(dev 0.78°)拉到 1272.54(dev 0.0024°)
  - **产物处置**:`outputs/nus_mini` **已重采覆盖**(旧版 33M 表对图错);`nus_mini_l3` **已删除**;修前基线固化在 `report_prefix.json`。**改动未提交,待用户手动 `git commit`**
- **覆盖层中文豆腐块修复 ✅**(2026-09-23,Plan2.md §P-M.8):`check_geometry.png` 中文全成方框 ⇒ 裁定**是字体问题不是编码问题且本机可解**(不降级英文)。新建 [autodrivedata/fonts.py](autodrivedata/fonts.py) 收敛为**唯一字体落点**(渲染探针判字形:`U+10FFFF` 像素签名 = `.notdef` 签名,`文相机字` 四签名必须互不相同;本机唯一 CJK 字体 = **CARLA 随包的 `DroidSansFallback.ttf`**),`sanitize` 缺字换 ASCII、兜底 `?`,**绝不留豆腐块**;`viz_calib_check`/`live_common`/`mapviz`/`probe_calib`/`carla_common`/`collect_static_gt` 六个绘制文件全部迁移,回归钉 [tests/test_fonts.py](tests/test_fonts.py)(含 AST 根因钉:不许裸写不带 `font=` 的 `d.text(...)`)。**两条成因**:①本机字体族无 CJK 而代码硬写 DejaVu;②**PIL 没有回退链,不传 `font=` 就用内置位图字体**(同样无 CJK、仅 ~11 px)。验收:151 用例过 + 四张 `check_*.png` 用 CARLA 重生成逐张确认
- **宽视场 wide rig ✅**(2026-09-23,Plan2.md §P-M.9):用户口径「前三个 55°、后侧 110°、后 180°,挂点后移到车尾,画幅里不能有自身车体像素」。**两处先被离线复算推翻**:①**180° 针孔退化**(`fx=(w/2)/tan(90°)≈5e-14` ⇒ `det(K)≈0`,投影恒等于主点;CARLA 渲染另有崖:**≥139° 中央带消失、180° 全黑**,0.9.16 无鱼眼蓝图)⇒ 用户裁决封顶 **120°**;②**零车体像素的解析上限 `hfov ≤ 2·min(az−90°,270°−az)`** —— 官方轴方位角下后侧上限只有 ~37°/42° ⇒ **必须同时改轴**,用户裁决改 **145/180/215**;另订正外部事实:官网 55/110/180 是**方位角**不是 FoV(官方 FoV 实为 64.3–65.0°×5 + 89.3°)。rig 落在 [autodrivedata/camera_rig.py](autodrivedata/camera_rig.py)(`NUS_WIDE_*`,**前三个与官方逐位相同**、后三路 `x=−1.90` 在车身最后点之后 4.7 cm),官方口径仍是默认
  - **八条判据全过**(`bin/verify_nus_calib.py --rig wide` → `outputs/nus_calib_check/report_wide.json`):③④ 与相机无关原样达标、⑤ 逐通道 **0.0 px**、⑥ 渲染 FOV dev `0.0004–0.0588°`、**⑦ 六路车体像素全 0 px**(⚠️ 该判据的**对照已过时**:`CAM_BACK = 619189 px = 42.9992%` 是 §P-M.10 挂点原点修正**前**的读数,修正后官方 rig 同样六路全 0 ⇒ ⑦ 不再是 wide 的区分度,wide 买到的是 FoV spec 而**覆盖率更差** 95.79% vs 100%)、⑧ 后三路共同可见带 **66.5/66.5/29.0°**。覆盖 **95.79%**,盲区 3 段 **15.1561°**(7.3353+6.0984+1.7224)= 4.21% —— 正侧方两段是"前视 55° + 后视不越 90°"的**结构性代价**,如实报出
  - **★ 方位轴重叠 ≠ 有限距离下的共同可见**(本轮最重要的订正):方位轴是**无穷远**口径,而挂点视差让同一世界点两路方位角差最多 **1.7°** ⇒ 官方 `FL↔BL` 11.20° 在 12 m 处只剩 **4.838°**、官方 `B↔BL` 的 5.89° **整个消失**(见 [bin/rig_check.py](bin/rig_check.py) 的 `common_band`);**锥心必须抬 0.45 m** —— 否则侧视相机擦车顶被自车挡,读数变成"有没有被自车挡"
  - **交付物 ✅**([bin/viz_rig_check.py](bin/viz_rig_check.py) → `outputs/calib_check/`):`rig_layout_{nuscenes,wide}.png`(配置图:俯视挂点+视锥 / 方位环重叠橙·盲区红带度数 / 数字表 `通道·挂点·方位角·FoV·**az±fov/2**`;纯值落点 [autodrivedata/rigviz.py](autodrivedata/rigviz.py))、`views_{rig}.png`(六视角原生像素拼图 + 逐格读数 + **线性方位尺**:逐相机一条泳道,竖线穿过的行数 = 该方位被几路覆盖;车体像素**就地染品红**)、`report_{rig}.json`。**图不能替代 `verify_nus_calib`**——"声明 ≠ 渲染"在图上看不见(§P-M.7)。两处踩坑:跨 0° 扇区写 `az % 360` 会让 `rectangle` 抛 `ValueError`(必须钳不取模)、`np.asarray(PIL)` 是只读视图(染色须 `np.array` 拷贝再 `fromarray`)
- **挂点原点修正:整套传感器后移 1.2563 m(后轴对齐)✅**(2026-09-23,Plan2.md §P-M.10):用户目检配置图发现 **CAM_FRONT 落在引擎盖上方**。根因是**两套系原点定义不同** —— CARLA 车辆 actor 原点 = **车身长度中点**,nuScenes 官方标定表(及 `ego_pose`)原点 = **后轴中心** ⇒ 把官方表值当 CARLA 局部坐标用 = 整组 **12 路偏前 1.2563 m**(官方 `CAM_FRONT` x=1.7008 落在 CARLA +1.7008,而应在 **+0.4445**)。用户裁决 **= 后轴对齐 Δx=−1.2563**(不是"中点规则"−0.8646:那会让 ego 原点与真后轴错位 0.39 m,`ego_pose` 与 devkit 语义不再一致)+ **相机+5 雷达+LiDAR 全套**刚体平移。单点真值 [autodrivedata/geometry.py](autodrivedata/geometry.py) `NUS_EGO_ORIGIN_X`,采集/导出/实时流/验收/配置图全部经它
  - **两条新判据**:①(实挂 vs 声明)②(雷达实挂)**相对同一个 ego 比,对原点误差在结构上盲** ⇒ 新增 **⑨ 世界系链**(`declared = 落盘表 ⊕ 实测后轴位姿` vs `rendered = 实挂经共轭`,12 路逐位比)与 **⑩ 独立复测后轴**(双偏航自解,**不读常量**,防常量腐化静默错位)。⑩ 实测后轴 **−1.2562963447285285** / 前轴 +1.2502001429339191 / 轴距 2.506496487662447 / 四轮 C 一致 **1.73e-06 m**
  - **★ ⑨ 首跑暴露两条真陷阱**:①**相机齐刷刷 119.93–120.07° 不是装反** —— CARLA 相机局部轴 (x 前,y 右,z 上) vs nuScenes (x 右,y 下,z 前),拿 LiDAR/雷达那条 `M·R·M` 比相机必得**轮换阵本征角(恒 120°)**;修法 = `CARLA_CAM_TO_NUS_CAM = [e1,−e2,e0]`(**正交但 det = −1**,局部基翻转与全局基翻转相乘才抵消)。②雷达/LiDAR 齐刷刷 **0.0846°** 是**第二处「声明 ≠ 渲染」**:`ego_pose` 只写纯偏航,**丢掉实测悬架俯仰 +0.0642°**(静置 8 tick 收敛)⇒ 新增 `nus_ego_rotation` 出全 6DoF;③同一处再暴露**符号陷阱**:因子分解只对**右手**矩阵成立,而 `carla_rotation_matrix` 是 **UE 左手口径**(纯 pitch 时 `R[2,0]=+sin p`)⇒ 正确是 `Rz(−yaw)·Ry(−pitch)·Rx(+roll)`,yaw 项对得上故极隐蔽,回归**按矩阵相等**判(比四元数会被 ±q 骗过)
  - **验收**:⑦ 由 `CAM_BACK 619189 px = 42.9992%` → **六路全 0 px**;⑨ 相机 `1.5e-05–4.0e-05°` / `3.8e-06–9.5e-06 m`,雷达·LiDAR `4.2e-06–2.6e-05°` / `1.2e-06–1.1e-05 m`;**两代 rig 十条判据全过**(wide 覆盖仍 95.79%/15.156° 逐位不变 ⇒ 无回归);`live_studio --calib` 重测**六路 `near_fraction` 全 0.0**(修前 CAM_BACK 0.367)⇒ **订正 §P-M.4 的「CAM_BACK 自遮挡 = 平台边界」**:那是修正前挂点落在车身中部的读数。`nus_mini` / `nus_mini_wide` 已重采,`smoke_radar_collect.sh` 四判据全过。**改动未提交,待用户手动 `git commit`**
- **★ 标定口径冻结 ✅**(2026-09-23 用户裁决,**Plan2.md §P-M.11**):用户验收 `outputs/calib_check/{rig_layout,views}_nuscenes.png` 后拍板「**以后就按照这样进行**」。冻结要点:ego 原点 = **后轴**(`NUS_EGO_ORIGIN_X = −1.2563`)、ego 姿态 = **6DoF**(含悬架俯仰 +0.0642°)、相机局部基 `CARLA_CAM_TO_NUS_CAM`(**det = −1**,与全局翻转 `M` 相乘才抵消)、像素约定 = **CORNER**、官方 rig 默认 / wide rig 并行、**12 路传感器刚体同移**。**三条不变量**:① spawn 与落盘由同一份常量导出(「表对了、图错了」只查表全绿、只目检也全绿);② 任何"实挂 vs 声明"判据**先 tick**;③ 写盘经 `paths.project_path()`。**验收唯一口径** = `bin/verify_nus_calib.py --offline --live` 的**十条判据、两代 rig 各跑一遍**(⑨⑩ 是原点误差的唯一探针,①② 对它结构上盲);**图不能替代它**。**本口径下作废**:❌「CAM_BACK 自遮挡 = 平台边界」(§P-M.4)、❌ wide rig 用 ⑦ 对官方 rig 的区分度(§P-M.9)、❌「官网 55/110/180 是 FoV」(那是方位角);✅ 保留「rig 必须匹配训练数据」(§P-L.1,全部旧权重仍不可复用)。**解锁**:§P-M.5 / §P-M.6 的重采重训前置已全部满足
- **重采重训(§P-M.11 冻结口径下)🔄**(2026-09-24,Plan2.md §P-M.12):迁移到 **RTX 4090 48G** 后开工。**采集 ✅**:`outputs/surround_v2/seg{0..4}`,5 段 × 100 帧,**stride 5 = 0.5 s/帧 = nuScenes 关键帧率**;spawn point 44/14/15/152/55(贪心最大最小距离,两两最近 114 m),合计路线 **1280.7 m**(旧集 300 帧只有 139.2 m,**9.2×**);采集吞吐 0.93 s/tick,5 段 46 min。**组装/划分 ✅**:`assemble_maptr.py --segs-dir`(帧带 `seg`/`frame_in_seg`,`token` 全局唯一);划分收敛到 `maptr_impl.dataset.select_frames()` **唯一落点**,回归钉 `tests/test_maptr_select.py`(20 用例)。**训练 ✅ / 结果 ✅**:单帧基线 128 ep、`--lr-halve 0`、batch 6、wall 2 h 16 min,两个权重 —— `maptr_v2_single.pt`(400 帧 = seg0–3 全段,留出只能评 seg4)与 **`maptr_v2_singleF.pt`(320 帧 = `--exclude-seg seg4` ∩ `--keep-in-seg 0:80`,**一个模型给两套留出**)。**chamfer AP @`--score-thr 0.2`**:帧级留出 `[80,100)` **0.3043** / 路线级留出 seg4 **0.1114**(独立模型佐证 0.1039)/ 训练集自身 0.2727。**决策点通过**(帧级 0.3043 ≥ 旧 0.0674,但**旧值采于错误 rig 的旧数据 ⇒ 是闸门不是 A/B,不可归因**)。**★ 三条不许误读**:① **帧级留出 ≠ 泛化**——留出首帧与训练末帧**同街只隔 2.65–3.01 m**,而**真训过这些帧**的 400 帧模型在同一批帧上只有 0.3096 vs 未训过 0.3043(**差 0.0053**)⇒ 0.3043 就是"同街"这一天花板,真泛化看路线级(差 **2.7×**);② **该 AP 口径不是"训练/留出差距"的度量**——训练集在**每个阈值上都低于**帧级留出(0.2727 vs 0.3043 @0.2),因为它是 **precision 均值、无 recall 项**而模型在见过的输入上过预测更多(训练 pred/gt **1.74×** vs 路线级 1.21×)⇒ **§5.11 的「泛化间隙 3.9×」不能用它复算**;③ 帧级留出还叠了 **GT 密度**混淆(留出 11.34 实例/帧 vs 训练 7.87,各段尾段恰是路口)。**类别级**:`ped_crossing` 同街 0.2125 → 未训路线 **0.0000**(pred 8/gt 139)⇒ 该类靠位置记忆。**用户裁决(2026-09-24):时序暂不训练**(阶段 4 停在「代码+单测就绪」,`maptr_impl/temporal.py` + `--temporal-window` + **74 用例过**;**不准当顺手跑的下一步自动接上**)。**★ 顺带修掉一个假警报**:过拟合闸门原按 `--frames > 1` 判,而 `--frames 0` 是"不截断"⇒ 400 帧训练被误判成单帧锚点、打假 FAIL + 退出码 1(权重在闸门之前已存盘,模型没坏);改 `is_single_frame_anchor(len(ds))` + AST 静态根因钉
  - **★ 两处口径订正**:① **分辨率 1600×900 是几何决定,不是"追官方"** —— 1242×375 宽高比 3.31 配官方 64.3° 水平 FoV ⇒ 垂直 FoV 只剩 **33.6°**(官方 38.9°),**换分辨率会改掉投影几何本身**;K 仍是 `fx 取官方、主点取 corner`(官方 cx 792–829 是**真实相机装配公差**,消费方是 devkit;我们的图中心恒为 799.5,落官方 cx 会让 GKT 逐通道偏 7–27 px ⇒ 又一次「声明 ≠ 渲染」。两条是**角色不同,不许"统一"**);② **采样步长必须一起改** —— 旧集每帧位移中位仅 0.520 m,而留出集是**按帧切**的 ⇒ §5.11 记的「泛化间隙 3.9×」里有相当一部分是**泄漏**
  - **`collect_surround.py` = 第三处「声明 ≠ 渲染」(已修)**:六路共用 `cam_bp` 的 `fov=90` + 一表六用的 K,而官方逐通道 FoV 是 64.31–64.96°×5 + **89.34°**(差 25°)。改逐相机蓝图 + 逐通道 fov;`live_common.rig_frame()` 是**实时侧画幅/FoV 唯一落点**(`legacy` 口径一字未动)。**绝不改 `carla_common.CAM_ATTRS`**(16 处引用,动它就是动 P1 复现性红线)
- **官方栈对照已终止**(2026-09-14,§5.12):官方 MapTR/MapQR 复线(同数据重训对照)中止,官方栈产物 `outputs/maptr_official/` 已删(**env 当日漏删,2026-09-19 补删**,合计回收 15 G);**自实现线全部资产不受影响**(ep512 权重/逐帧契约/实时 overlay 照旧——**"不受影响"指"没被这次终止删掉";ep512 本身已于 2026-09-22 因 rig bug 标废弃,见上条 §P-M**)。A′ 的**口径**结论保留(两套 mAP 口径的关系),但"官方实现 vs 自实现"的对照表不存在
  - **八视角视频段 ✅**(§P-L.5):`live_studio --video <mp4>`(cv2/mp4v 惰性开编码器,拼图槽逐帧写盘;`--video-fps` 是**标称**帧率,结束打印实际采集 fps 与播放倍速——不调到实测值视频就是加速的;`--video-tile >1` 只是插值放大)。实测 MapTR+SLAM 同开 0.3–0.5 fps、仅 GT overlay ~2 fps
- **P1-6 候选**:wet_road 眩光 / dense_rush 遮挡(待用户定)

## 环境(勿新建;direnv 进入目录自动激活 autodrivedata,首次需 `direnv allow`)

| 环境 | Python | 用途 |
|---|---|---|
| **autodrivedata**(本项目) | 3.11.16 | pycarla + ultralytics;采集 `bin/collect_*.py`、2D 评估 eval_2d_ab.py、3D 比对 eval_kitti.py、全部单测 |
| **autolabel** `/root/miniconda3/envs/autolabel` | 3.11.15 | mmdet3d;3D 检测 `auto3dlabel run`、oracle 对比 |
| **hivt** `/root/autodl-tmp/envs/hivt` | 3.8.20 | HiVT 复现栈(torch1.8.0 / pl1.5.2 / pyg1.7.2 / argoverse-api),CPU 推理。**未注册进 conda envs_dirs** → `conda info --envs` 看不到、`activate hivt` 失败,**只能用绝对路径调** `envs/hivt/bin/python`(见 [bin/convert_hivt_pt.py](bin/convert_hivt_pt.py) 用法头)。数据盘 2.5 G,勿删 |
| **base** | 3.10.8 | conda 底座 + direnv;pycarla/ultralytics 已于 2026-09-10 迁出,不承担项目职责 |
| ~~**maptr_official**~~ **已删**(2026-09-13 建 → 2026-09-14 终止,env 于 **2026-09-19 补删**,§5.12) | ~~3.8~~ | 官方 MapTR/MapQR 老栈(torch1.9.1+cu111 / mmcv-full1.4.0 / mmdet2.14.0)。**用户裁决整条官方复线中止**(41 h 训练预算仍过长)→ env 与产物已删、回收 15 G。**重建设路子**:`bin/setup_maptr_official.sh all` + Plan.md §5.12(含四个钉子/两处 bug/独立评测四坑/A′ 口径结论,知识都留在文档里) |

**env 落点与可见性(2026-09-19 核实,勿再困惑)**:`conda config --show envs_dirs` = `/root/miniconda3/envs` + `/root/.conda/envs`,
即**只有这两个目录下的环境才被 conda 按名字发现**。而 `/root/autodl-tmp/envs/` 是数据盘上的**独立目录,不在 envs_dirs 里**:

| 环境 | 物理落点 | conda 可见? | 机制 |
|---|---|---|---|
| autodrivedata | `/root/autodl-tmp/envs/autodrivedata`(7.5 G,**数据盘**) | ✅ 可见 | `/root/miniconda3/envs/autodrivedata` 是**符号链接**指向它(2026-09-10 建,把大 env 挪出 30 G 系统盘) |
| hivt | `/root/autodl-tmp/envs/hivt`(2.5 G,数据盘) | ❌ 不可见 | 用 `conda create -p <路径>` 创建(见其 `conda-meta/history`),**从未建符号链接** → 只能绝对路径调用 |
| autolabel | `/root/miniconda3/envs/autolabel`(8.0 G,系统盘) | ✅ 可见 | 常规 `-n` 创建,无副本、无链接 |

⇒ **"为什么 autodrivedata 两处都有、autolabel 只有一处、hivt 一处却看不见"**:autodrivedata 是「真身 + 符号链接」两处(链接 0 字节,不占额外空间);
autolabel 从未迁移故只有真身一处;hivt 有真身但**没有链接**故 conda 看不见。三者都不是"副本",不存在重复占盘。

纪律:autodrivedata 包**不 import carla**(纯值,任何 env 可单测);依赖单向 AutoDriveData → AutoLabel(3D 检测消费方),禁止反向。
该纪律的执行者 = [tests/test_layer_guard.py](tests/test_layer_guard.py) 的 `LAYER_RULES`(**按目录**声明允许 import 什么,
不是一条全局禁令)。改目录结构前先读 [Plan_fileTree.md](Plan_fileTree.md) §3。

## 项目结构

> 📁 **文件级索引见 [docs/fileTree.md](docs/fileTree.md)**——每个文件/脚本/产物目录的职责、依赖方向、
> 哪些是【未入库】产物。**新增或改名文件后回来补一行**(维护约定在该文档头部);下面是速览版。

- `autodrivedata/` — 纯值库(geometry/calib/**camera_rig**/calib_probe/**calib_live**/**rigviz**/fonts/depth_codec/gt/static_gt/traffic_light/attribution/semantic/export/compare/scenarios/paths/mapvec_schema),不 import carla
- `bin/` — carla 采集器(collect_drive/collect_ab_route/collect_static_gt/collect_tl_states/collect_nus)+ 评估(eval_2d_ab/eval_attr/eval_kitti)+ 可视化(view_stream)+ 自证探针(probe_calib/probe_rig_mount/probe_vulkan/**verify_nus_calib**/**rig_check**)+ 配置图/实拍图(**viz_rig_check**)+ `carla_common.py`(位姿/NPC/传感器/灯态归一与绘制共用件)
- `tools/` — 开放性工具(判据:**不含本项目领域知识**):`carla_server.sh`(GPU 修复版启动 + **Vulkan 兼容层自愈**;宿主驱动升版致 `libnvidia-gpucomp.so.<ver>` 缺失时自动顶名,见 Plan.md §5.11f)+ `gpu_fix/`(LD_PRELOAD shim 源码与安装脚本)
- `tests/` — 单测(autodrivedata env;**802 用例**已收集,2026-09-23 `--collect-only`;改动后只跑相关单测,不跑全量)
- `outputs/` — **全部产物的唯一落点**(采集/权重/可视化/运行支撑物):kitti_* 为 KITTI root 结构;kitti_ab_* = P1 A/B 序列;kitti3d_ab_* = 3D 伪标签;`maptr_*.pt` = 权重;`carla/` = 服务器日志 + shim + Vulkan 兼容层
- `docs/` — `fileTree.md`(**文件级索引,加/改文件后必须回来补一行**);`milestone.md` / `milestone2.md`(里程碑);`testLog.md`(测试日志);`Carla_Sim_Tutorial_01..16.md`(教程,Plan2.md 能力对照源);`PRD.md` / `TRD.md`(空占位)
- 顶层文档 — `Plan.md` **方案定案 + 历史执行记录**(§1~§4 契约/架构、§5 里程碑归档、§6 骨架、§7 待办快照,**冻结不再新增**);`Plan2.md` **新计划的制定地**(2026-09-19 起,改决策先读 Plan2.md 再改)

## 常用命令

```bash
# CARLA 服务器(专用用户 carla + LD_PRELOAD shim,GPU 修复栈;headless)
bash tools/carla_server.sh        # 启动;停止用 stop(start/stop/status;勿手敲 pkill -f CarlaUE4,自匹配坑见 C19)

# 场景采集(KITTI root,含 label_2 GT/velodyne/calib)
python -m autodrivedata.sim.collect_drive --scene rain_night --frames 70
python -m autodrivedata.sim.collect_ab_route --scene sunset_glare --frames 70   # P1 A/B 专用:锚定 pt0 + 4 静置车

# 静态 GT(landmark + 车道线,含 overlay 目检图)
python -m autodrivedata.sim.collect_static_gt --frames 40

# 灯色动态 GT(记录模式默认不动灯;--cycle 绿,黄,红 秒数 = 受控切灯)
python -m autodrivedata.sim.collect_tl_states --frames 40 --speed 8
python -m autodrivedata.sim.collect_tl_states --frames 90 --speed 8 --cycle 6,2,6

# 实时可视化(本地 ssh -L 8080:127.0.0.1:8080 <autodl> → 浏览器 127.0.0.1:8080)
python -m autodrivedata.sim.view_stream --view follow --npcs        # 跟车视角
python -m autodrivedata.sim.view_stream --view top --map Town13     # 俯视看街区
python -m autodrivedata.sim.view_stream --scene rain_night --speed 8  # 天气 + 定速直行
python -m autodrivedata.sim.view_stream --view follow --dump outputs/dumps/f.png  # 落 raw+overlay 做差集诊断
# MapTR 实时预测 overlay(需 --view grid6;投影链与离线 viz 共用 autodrivedata/mapviz)
python -m autodrivedata.sim.view_stream --view grid6 --maptr-ckpt outputs/maptr_ep512.pt --maptr-bev --dump outputs/dumps/m.png
# 8 路 studio(6 相机 + BEV + 第三方 + 拼图;WASD 操控 + 在线 SLAM)
python -m autodrivedata.sim.live_studio                             # 8 路 + 键盘(stdin 是 tty 时默认开)
python -m autodrivedata.sim.live_studio --speed 8 --npcs            # 定速直行(键盘自动关;两者互斥会报错)
python -m autodrivedata.sim.live_studio --slam --speed 8 --duration 90 --slam-report outputs/slam_gt/accept.json
python -m autodrivedata.sim.live_studio --maptr-ckpt outputs/maptr_ep512.pt --slam --speed 8  # 感知 + SLAM 同屏
# 落一段八视角视频(--video-fps 调到接近实际采集 fps 才是实时播放;结束会打印实测 fps 与倍速)
python -m autodrivedata.sim.live_studio --npcs --speed 6 --duration 100 --fps 10 --no-keyboard \
  --maptr-ckpt outputs/maptr_ep512.pt --slam --video outputs/videos/studio_8view.mp4 --video-fps 0.5
PYTHONPATH=$PWD python bin/viz_maptr_pred.py --start 250 --frames 6   # 离线:预测回投 6 相机拼图 + BEV
# 逐帧契约落盘(供 AutoLabel 消费;GT 同文件携带,见 autodrivedata/mapvec_schema.py)
PYTHONPATH=$PWD python bin/eval_maptr.py --infos outputs/surround_train/map_infos.json \
  --root outputs/surround_train --ckpt outputs/maptr_ep512.pt --start 200 --out-frames outputs/surround_pred

# 2D A/B 评估(A=day_clear 基线与 B 帧级配对)
python bin/eval_2d_ab.py --root-a outputs/kitti_ab_day_clear --root-b outputs/kitti_ab_sunset_glare

# 失效归因(逐帧匹配 → 距离/框高/TTC 分箱 + 漏检画像)
python bin/eval_attr.py --run day8=outputs/kitti_sweep_day_clear_8:8.0 \
  --run rain=outputs/kitti_ab_rain_night:8.0 --json outputs/attr.json

# 3D LiDAR 检测(autolabel env;**cwd 必须在 AutoLabel 根**,config 相对路径)
cd /root/autodl-tmp/Documents/Projects/AutoLabel && KITTI_OBJECT_ROOT=<abs kitti root> \
  /root/miniconda3/envs/autolabel/bin/auto3dlabel run 000000-000069 "检测汽车" \
  --det-model pointpillars_kitti --batch --no-viz --out-dir <abs out>
python bin/eval_kitti.py --root outputs/kitti_ab_x --pred outputs/kitti3d_ab_x   # 3D 比对

# nuScenes 迷你集(全传感器「渲染 = 声明」同源;重采后跑验收)
python -m autodrivedata.sim.collect_nus --frames 2                     # 重采(需 CARLA 在跑)
PYTHONPATH=$PWD python bin/verify_nus_calib.py --offline --live   # 八条判据 → outputs/nus_calib_check/
bash autodrivedata/sim/smoke_radar_collect.sh                          # devkit 直读四判据

# wide rig(挂点后移 + 新 FoV,画幅内零车体像素;官方口径仍是默认)
python -m autodrivedata.sim.collect_nus --rig wide --out outputs/nus_mini_wide --frames 2
PYTHONPATH=$PWD python bin/verify_nus_calib.py --rig wide --offline --live   # → report_wide.json
PYTHONPATH=$PWD python bin/viz_rig_check.py --rig wide --live     # → outputs/calib_check/{rig_layout_*,views_*,report_*}

# 规范 + 测试(提交前两件套;规则集钉死在 pyproject [tool.ruff],110 列)
ruff check && ruff format        # format 无参数即就地格式化,全仓口径统一
python -m pytest tests/ -q
```

## 红线(A/B 实验纪律与已踩坑,勿再犯)

- **A/B 帧级配对是硬门槛**:同 ego 锚定 spawn point 0(yaw=0)、同静置车布局(20/35/50/62m——65m 会卡 GT max_distance 阈值抖动)、只变 weather;GT 数必须相等,否则样本不可比、结论作废。autopilot/TM 路线不可复现,禁用于 A/B
- **AP 尾部不注水**:未达 recall=1 段 precision=0(11 点插值,与 compare.ap11 同口径)。旧尾行 `ap += (1-prev_r)*prev_p` 曾把低 recall 吹高(雨夜 0.48 检出报 0.976),已修
- **MapTR chamfer AP 必须带 score_thr 引用**:该口径是 3 阈值 precision 均值、**无 recall 项** → 高阈值(预测少而准)天然占优;同一权重 0.2→0.4 能翻 2.6×(0.0510→0.1350)。跨权重比较**固定 --score-thr**,看曲线用 `--sweep`;单独报一个 mAP 数字而不写阈值 = 无效结论
- **carla pyi 桩坑**:`try_spawn_actor` 桩标返回 `Actor`(实为 `Actor|None`)→ 用 Vehicle 方法必须 `cast(carla.Vehicle, v)`;Vector3D 运算结果不能直接进 `carla.Transform`(显式 `carla.Location`);函数签名要 `tuple[float, float, float]` 定长时禁用 tuple 推导(变长 tuple)
- **sunset_glare 方位**:az=90=东=+x=车头正前(yaw=0 时);az=300 是顺光陷阱(太阳在车后)。判据 = 全图过曝最低(AE 压最狠)
- **采集器清场 + 起点校验**:残留 actor 阻塞 spawn point 会致 fallback 反向出生点、轨迹失配(collect_ab_route/collect_static_gt 已内置,勿删)
- **锚定 yaw 用 spawn point 固有 rotation**:collect_static_gt 曾硬编码 yaw=0,Town10HD_Opt pts[0] 固有 yaw=0.16° 恰好成立、Town13(125.9°)车道线采样走到车后 overlay 全空;已改用 pts[0].rotation(沿车道),collect_ab_route 的 yaw=0 是 P1 已验证基线勿动
- **"定速"必须清制动残留 + 逐帧自证**:`VehicleControl` 一旦设置就每步生效,`brake=1.0` 站定后不解除会让 `set_target_velocity` 打 0.82 折(P1 四个老数据集实为 6.60 m/s 而非 8.0,已修);速度/时序结论**用逐帧序列测**(`closing_speed_series`),不要从 ego-x 首末值反推(曾误判"2 秒加速段")
- **漏检归因先看框高箱再看亮度**:<32px 一律 0.15–0.47、≥32px 一律 0.78–1.00,漏检框内亮度与命中几乎相同——主因是尺度不是"暗";TTC 箱**不可跨速度比检出率**(同箱在不同速度对应不同距离)
- 模型无法读图时用**数值诊断**(亮度带/过曝占比/梯度),不要硬目检;验证 overlay 必须做**同帧 raw/overlay 差集**——场景自带绿(植被)/黄(标线)与类别色撞色,数绝对颜色会误判(曾报"Car 仅 3 px")
- **灯态 GT 不做视觉回归**:镜片 0.2m,在 KITTI 口径相机(f=621)下 30m 处仅约 4px;12–30m 处按颜色采样命中的是**黄色灯箱外壳**(≈(255,237,0),与黄灯镜片同色相)。真值取自 actor API(逻辑层)。要拍镜片必须按灯头盒**薄轴**放相机(盒 yaw 方向拍的是背面,曾据此误判"渲染不随 set_state 变")
- **覆盖层文字一律走 `autodrivedata.fonts`,不许裸写 `d.text(...)`**:PIL 遇缺字**静默**画 `.notdef` 方框(不报错不告警),且 `ImageDraw.text()` **不传 `font=` 就用内置位图字体(无 CJK、仅 ~11 px)** —— 本机踩到的两条成因各修一次才全绿(见 Plan2.md §P-M.8)。"这字体能画中文吗"的判据是**渲染探针**(`U+10FFFF` 的像素签名 = `.notdef` 签名),**不看文件名、不看 `fc-list`、不用 `getmask` 字节**(实测 PUA 反例);本机唯一 CJK 字体是 **CARLA 随包的 `DroidSansFallback.ttf`**。字体一换排版会连带暴露三处:**比例拉丁字体下 `:<16` 空位对齐不成立**(改显式列坐标)、**底条宽度不能按字符数估**(按实测宽度)、**测试里按像素裁的魔数会失效**(改从 live_common 导出常量推导)。**长文本一律 `fonts.wrap(text, size, 最大宽)` 折行再画**——单行 `draw_text` 超出画布被 PIL **静默裁掉**(不报错、不告警,图上看着像"写完了"),而"几个字一行"必然估错宽度(同行的比例拉丁与全宽 CJK 差 ~2×);页脚类文字提为模块常量以便单测对原文断言"折完每行都装得下"
- **同步模式首个 `get_actors()` 为空**(快照只在 tick 后刷新)→ 清场会静默漏清;已修在 `carla_common.sync_mode()`(apply_settings 后补 tick),勿绕过它自己 apply_settings
- **快照陈旧不止 `get_actors`,传感器 `get_transform()` 同样**:tick 前读到的相机位姿全是 0(不是真实挂点)。view_stream 的 rig 自检曾因此假报 179.841°(= CAM_BACK 规格 180 − ego 固有 yaw 0.159),加一次 `world.tick()` 后 0.000°。**任何"实挂位姿 vs 规格"的判据都必须先 tick**
- **投影链的出口口径是弧度**:`autodrivedata/mapviz.cam_pose` 位置米 / 姿态弧度(与 `calib.world_to_img` 一致);把度直接传进去时 6 相机里**只有 yaw≈0 的 CAM_FRONT 看着正常**,侧/后相机全错位。判据不看图,看"命中点落在该相机自身 FOV 内的比例"(弧度 91–100% vs 度数 0–6%)——**"能画出图"不是投影正确的证据**
- **灯态收集侧要前向过滤**:圆形 horizon 会把身后 120m 的灯全收进来(实测占 79%),`traffic_light_frame(forward_only=True)` 是默认口径
- **方位角扇区跨 0° 时不许取模、PIL 染色不许就地改**:① `camera_rig._sector_intervals` 把跨 0° 的扇区劈成 `(lo, 360.0)`+`(0.0, …)`,画的时候写 `az % 360` 会把右端折回**最左** ⇒ `ImageDraw.rectangle` 拿到 `x1 < x0` **直接抛 `ValueError`**(不是画错,是崩)⇒ 钳到 `[0,360]` 即可;② `np.asarray(PIL 图)` 是**只读**视图,`arr[mask] = …` 抛 `assignment destination is read-only`,必须 `np.array` 拷贝 + `Image.fromarray` 回包。**两条都只有"新 rig / 官方 rig"才走到**(wide 的零车体像素让染色分支永不触发)⇒ 出新图必须**两代 rig 各跑一遍**,否则测不到的那条分支会一直藏着
- **"方位轴重叠"不等于"重叠区真的存在"**:方位轴重叠是**无穷远**口径,挂点视差让同一世界点在两路里的方位角差最多 1.7°(实测官方 `FL↔BL` 11.20° → 12 m 处 4.838°,`B↔BL` 的 5.89° 整个消失)⇒ 判重叠要按**有限距离**算(`rig_check.common_band`),别拿两扇区求交的度数下结论
- **产出必须落在项目内**:写盘路径一律经 `autodrivedata/paths.project_path()`(**相对路径 = 相对项目根**,不随 cwd 漂移;绝对路径原样放行)。历史口径"相对 cwd 的 outputs/"在换 cwd/换会话时会把权重与可视化散到项目外(清盘时无从分辨)。运行支撑物同理:shim/兼容层/服务器日志在 `outputs/carla/`(原 /tmp 与 carla_home 副本已废弃)。**读路径不锚定**(输入沿用 cwd 口径,便于临时 `cd`)
- **官方栈复线(§5.12)已终止**(2026-09-14 用户裁决;**不要主动重提**)。重启前先读 Plan.md §5.12:四个钉子(`bs1×累积5` 口径 / config 必须钉 `color_type="color"`、否则在线评测一开就炸 / v1 同位姿帧放行判据 / 预算对齐基线**第一轮** 256 ep)、独立评测的四个坑(单卡 `assert False`、`init_dist` 强制 spawn + `dict_keys`、产物路径相对 cwd、dist_test.sh 硬编码 `--eval bbox`)、A′ 口径结论(官方 eval_map 100 点 GT 重采样 = 0.0699 才是参照值)。**成本在 BEV transformer 不在主干**——降分辨率换不到吞吐(0.5 反而更慢)
- **停训练必须连 DataLoader worker 一起收**:worker 是 fork 出来的,而 fork 发生在 CUDA 初始化**之后** ⇒ worker **继承 CUDA 上下文**,父进程被杀后变 PPID=1 的孤儿**继续占显存**(nvidia-smi 仍把额度挂在已死的父 PID 名下,实测 `kill` 父进程后仍占 5068 MiB)→ 收完所有相关 PID 后 GPU 才归零。**判据:`nvidia-smi` 归零才算停干净,不是"父进程没了"**
- **纯 Python 主循环里别指望 worker 线程**:studio 主线程每 tick 的 GT overlay / `compose_grid` / HUD / 灯态绘制全是**字节码**,持 GIL 不放 ⇒ 同一对点云 ICP 在 worker 线程 eff **0.04–0.24** vs 主线程同步 **0.90–1.00**(差 10–20×)。**判据看 `time.thread_time()/wall`(eff),不是 wall 单值**;`OPENBLAS_NUM_THREADS=1` 不改结论 ⇒ 与 BLAS 线程池无关。用 PIL/CARLA tick 施负载测不出(它们会释放 GIL)—— 必须用**纯 Python 小矩阵自旋**才能复现。同理:**有界队列有界的是深度不是"状态间隙"**,任何"上一帧 vs 当前帧"成本随间隙超线性的算法(ICP/配准/图优化),丢帧都会变成"丢帧→间隙更大→更慢→更多丢帧"的正反馈,**必须按帧号差止损**(`SlamWorker.max_gap`)
- 提交:Conventional Commits;**提交信息不附 AI 署名**(不加 `Co-Authored-By: Claude` 等 trailer);改动后 `ruff check && ruff format` + 相关单测;决策与执行记录同步进 **Plan2.md**(教程能力线另同步 docs/milestone2.md);Plan.md 已冻结,只保留定案与历史记录
- **格式口径已定死**:`[tool.ruff]` 在 pyproject(line-length 110 / select E,F,I,UP,B / ignore E501,E741),`ruff format` 是唯一 formatter;批量纯格式提交要追加到 `.git-blame-ignore-revs`

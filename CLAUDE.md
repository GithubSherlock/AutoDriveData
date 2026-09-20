# AutoDriveData

CARLA 0.9.16 → AutoLabel 自动驾驶数据输出流水线:自定义地图/场景采集车检测动态/静态目标与道路特征,构建 0→1 数据输出。当前主线 = **Corner Case 场景矩阵定量验证**(P1)+ 静态 GT(P2);raw 输出供 AutoLabel 验证场 + 长尾数据源。

## 当前进度(2026-09-09)

- **P1 ✅**(2388336 起):场景目录 [autodrivedata/scenarios.py](autodrivedata/scenarios.py)(8 场景,不 import carla)+ `collect_drive --scene` 采集链 + 三 corner case A/B(统一 11 点插值 AP 口径,见 [Plan.md](Plan.md) §5.7c 汇总表):
  | corner case | 相机 Δ | LiDAR Δ |
  |---|---|---|
  | 逆光(az=90 校准) | -0.014 轻掉点(AE 补偿) | +0.003 噪声 |
  | 雨夜 | **-0.153 漏检型**(检出 0.72→0.48) | +0.017 噪声 |
  | 浓雾 | -0.013 FP 型(检出 0.72→0.85) | 0.000 |
  - 结论:相机三型可量化掉点;LiDAR 兜底不受天气/光照(平台边界:雨/雾无物理回波,退化只能人工注入)
- **P2 ✅**(f66558c):静态 GT = **地图查询 API**(信号/标志是 landmark、车道线是 lane_marking 实体;semantic LiDAR 打不到)→ [autodrivedata/static_gt.py](autodrivedata/static_gt.py) + [bin/collect_static_gt.py](bin/collect_static_gt.py),落盘 `training/static_gt/{fid}.json` + overlay 目检图。**更正**:Town10HD_Opt 有 15 个 traffic_light actor(xodr 17 个 dynamic 信号),原记"无信号 actor"有误——灯色属动态 GT,仍不入静态 json
- **M4 挂起**(§5.7a 用户裁决):定制街道 = CARLA **源码构建**(prebuilt 无 UnrealEditor,~170G 磁盘/Epic 账号),成本过载降级为扩展 P1。开源 AdditionalMaps(Town11-15,14.8G)已探明可下载,零构建扩地图池
- **地图池扩展 ✅**(§5.7d):AdditionalMaps 14.8G 已装,Town11/12/13/15 入池(17 图)。**新图采集约束:Town11/12 禁采集**(spawn camera segfault)、Town13 TM 车流降级 0 NPC、可用 Town13/15;默认图仍 Town10HD_Opt(重启即恢复)
- **可视化实时流 ✅**(§5.8):[bin/view_stream.py](bin/view_stream.py) 自建 MJPEG(3 视角 + GT 框/灯色 overlay,只绑 127.0.0.1 走 SSH 隧道);carlaviz/RViz2 出局(非 UE 渲染 + 版本/依赖不成立);`world_to_img` 上移 [autodrivedata/calib.py](autodrivedata/calib.py) 供采集器与实时流共用
- **灯色动态 GT ✅**(§5.9):[autodrivedata/traffic_light.py](autodrivedata/traffic_light.py)(纯值:状态归一/前向判据/相位查表/JSON 往返)+ [bin/collect_tl_states.py](bin/collect_tl_states.py)(记录模式 / `--cycle 6,2,6` 受控切灯 → 确定性变灯序列),落盘 `training/traffic_light/{fid}.json`。工业口径:灯态 = 独立时序层,Off/Unknown **不猜**;受控 90 帧状态变化点 = 帧 0/60/80 与计划逐帧吻合。**边界:不做视觉回归**——镜片 0.2m 在 f=621 下 30m 处仅约 4px,且黄色灯箱外壳同色相
- **P1 参数扫描 + 失效归因 ✅**(§5.10):[autodrivedata/attribution.py](autodrivedata/attribution.py) 纯值(逐帧匹配/分箱/逐帧速度自证)+ [bin/eval_attr.py](bin/eval_attr.py)(多跑 × 距离/框高/TTC 网格 + 漏检画像),与 AP 共用同一 `box_iou2d`。三结论:**尺度主导**(<32px 0.15–0.47 / ≥32px 0.78–1.00,断崖 ≈21–24px)、**CARLA 无运动模糊**(4/8/12 m/s 梯度能量 35.6/35.2/34.8,检出率 0.914/0.886/0.909 → 速度不改图像,退化只能人工注入)、**天气只前移断崖**(雨夜 40-50m 零检出→30-40m 0.32,雾最晚 0.91)
- **MapTR 矢量管道 ✅**(§5.11):A 阶段矢量库(opendrive/mapvec + A6 oracle 0.00cm)→ B 阶段环视采集/组装/投影验收 → C 阶段**参考自实现**(`maptr_impl/`:GKT + 分层 query,单帧过拟合锚定正确性)→ D 阶段 chamfer AP。训练数据 200 帧(Town10HD_Opt@spawn0)
  - **第二轮 ep512 结果口径分化**:留出集 @0.2 **0.0674**(vs ep256 0.0510,+32%)、@0.3 0.0904、@0.4 0.1280(后两档持平略降)→ 保守操作点仍获益、高阈值已饱和;训练对照 0.2603 → **泛化间隙 3.9×**(ep256 时 2.1×)→ 下轮收益靠**扩数据**而非继续长训。权重 `outputs/maptr_ep512.pt`
  - **实时 overlay ✅**(§5.11f):`view_stream.py --maptr-ckpt outputs/maptr_ep512.pt --maptr-bev`(rig 走 `live_common.rig_spec` 两处定义;推理用实挂相机世界位姿);实况数值验证 overlay 品红 25553 px vs raw 0、地平线以上 0/18496、FPS 1.1–1.4
  - **rig 两代并存(2026-09-19,§P-L.1)**:`legacy`(共用 `SENSOR_OFFSET` + BACK_LEFT/RIGHT 235/125)对 `maptr_ep256/ep512`;`official`(逐相机 `SENSOR_MOUNTS` + 108.6/−110.8)对 `maptr_600/1000`。**rig 必须与权重训练数据一致,不是"越新越好"**;`--rig auto`(默认)按权重名选,拿 official 喂 ep512 是错配(品红 122711→135989 px)。A/B 探针 `bin/probe_rig_mount.py`
- **8 路实时 studio ✅**(2026-09-19 A 期 / 2026-09-20 B 期,Plan2.md §P-L):`bin/live_common.py`(共享件:单端口多槽 MJPEG `/stream/<name>` + `/` 索引页 / 拼图 / GT overlay / 环视 rig / 第三方视角 / `KeyboardState` / MapTR 懒加载)+ `bin/live_studio.py`(9 槽 = 6 相机 + `BEV` + `THIRD_PERSON` + `grid` 拼图;`--keyboard` 折进 tick 循环)。`view_stream.py`/`drive_ego.py` 改薄编排
  - **拼图三层 + 不缩像素 ✅**(§P-L.6):用户报告"6 视角 FoV 缩得看不到地面"——根因是 **`PIL.Image.paste` 源图大于目标框时只贴左上角、不报错不缩放**,旧 4×2 等尺寸拼图把 1242×375 裁成 621×187,右半 + **下半(地面)** 无声丢弃(判据:与「源图左上角裁剪」差 **0.128** vs 与「整幅缩放」差 **66.18**)。改 `compose_rows`(按行拼、每格**原生像素**)+ `compose_grid` **尺寸守卫**(不符即 `ValueError`,钉死不复发)+ `GRID_ROWS` 三层(①左前/前/右前 ②右后/后/左后 ③第三方 + BEV,**不沿用 `SURROUND_CAMS` 字典序**)。画布 2484×374 → **3726×1170**,逐格与源图最大差 **0.0**;回归 `tests/test_live_common.py`(12 用例)
  - **B 期在线 SLAM ✅**(§P-L.2~P-L.4):`autodrivedata/live_slam.py`(`LiveSlam.push/snapshot` + 地图/轨迹换到当前 ego 系 + `SlamWorker`)+ `live_studio --slam`(语义 LiDAR → BEV 槽画地图点灰/轨迹青,HUD 显式报滞后)。**两条原前提都被实测推翻**:①离线 ICP 0.78 s/帧是 400 帧**含转弯的平均值**,在线逐帧只有 0.15–0.35 s;②**worker 线程被 GIL 饿死**(主线程 overlay/拼图/HUD 是纯 Python 字节码;同一对点云 worker eff 0.04–0.24 vs 主线程同步 0.90–1.00;钉 `OPENBLAS_NUM_THREADS=1` 不改结论 ⇒ 不是 BLAS 线程池)⇒ **默认同步执行**(~2.4 fps),`--slam-async` 留作对照
  - **有界丢旧队列 ≠ 滞后有界**(关键机制):队列有界的是**深度**不是 `prev_down` 与当前帧的**间隙**,而 ICP 成本随间隙爆炸(0.8 m 0.2 s → 8 m 2.8 s → 32 m 39 s)⇒ 丢帧→间隙更大→更慢→更多丢帧**无界正反馈**(异步实测:259 tick 只处理 8 帧、滞后涨到 157 帧/45 s)。修复 = **按帧号差止损**(`--slam-max-gap` 默认 3,超阈帧不做 ICP 直接恒速外推,`prev_down` 照推进)。**滞后口径 = 已 tick 帧号 − 已处理帧号**,不是 `n_offered − n_processed`(后者随丢帧无界增长,是假故障)
  - **验收数字**:400 帧在线 vs 离线链式位姿**逐元素差 0.0**、ATE **0.18768 m**(= 离线 = 参照);同步 136 tick **丢 0/止损 0/lag_max 0**;退出后 `nvidia-smi` 回基线、残留进程 0;BEV 画出 90611 点 / 轨迹 20 段 / **画出的点数 = 窗内点数**。产物 `outputs/slam_gt/{accept_sync,accept_async,accept_parity_full}.json`。**判据订正**:"地图点窗内占比 ≈100%"是误读 —— 地图覆盖 ~300 m 而窗口只有 30×60 m,占比 0.24 正常
  - **逐帧契约 ✅**(§5.11h):`eval_maptr.py --out-frames` → `outputs/surround_pred/{token}.json`(`mapvec_pred/1`:schema/帧归属/类序/坐标系/窗口/阈值/溯源,GT 同文件携带;旧 `--out-pred` 是跨帧汇聚、实例无帧归属,不可对外)。供 AutoLabel 消费——**消费方未接**
- **官方栈对照已终止**(2026-09-14,§5.12):官方 MapTR/MapQR 复线(同数据重训对照)中止,官方栈产物 `outputs/maptr_official/` 已删(**env 当日漏删,2026-09-19 补删**,合计回收 15 G);**自实现线全部资产不受影响**(ep512 权重/逐帧契约/实时 overlay 照旧)。A′ 的**口径**结论保留(两套 mAP 口径的关系),但"官方实现 vs 自实现"的对照表不存在
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

## 项目结构

> 📁 **文件级索引见 [docs/fileTree.md](docs/fileTree.md)**——每个文件/脚本/产物目录的职责、依赖方向、
> 哪些是【未入库】产物。**新增或改名文件后回来补一行**(维护约定在该文档头部);下面是速览版。

- `autodrivedata/` — 纯值库(geometry/calib/gt/static_gt/traffic_light/attribution/semantic/export/compare/scenarios/paths/mapvec_schema),不 import carla
- `bin/` — carla 采集器(collect_drive/collect_ab_route/collect_static_gt/collect_tl_states/collect_nus)+ 评估(eval_2d_ab/eval_attr/eval_kitti)+ 可视化(view_stream)+ `carla_common.py`(位姿/NPC/传感器/灯态归一与绘制共用件)+ `probe_vulkan.py`(Vulkan 设备枚举,CARLA 渲染停摆的一线判据)+ `carla_server.sh`(GPU 修复版启动 + **Vulkan 兼容层自愈**;宿主驱动升版致 `libnvidia-gpucomp.so.<ver>` 缺失时自动顶名,见 Plan.md §5.11f)
- `tests/` — 单测(autodrivedata env,353 passed / 3 skipped)
- `outputs/` — **全部产物的唯一落点**(采集/权重/可视化/运行支撑物):kitti_* 为 KITTI root 结构;kitti_ab_* = P1 A/B 序列;kitti3d_ab_* = 3D 伪标签;`maptr_*.pt` = 权重;`carla/` = 服务器日志 + shim + Vulkan 兼容层
- `docs/` — `fileTree.md`(**文件级索引,加/改文件后必须回来补一行**);`milestone.md` / `milestone2.md`(里程碑);`testLog.md`(测试日志);`Carla_Sim_Tutorial_01..16.md`(教程,Plan2.md 能力对照源);`PRD.md` / `TRD.md`(空占位)
- 顶层文档 — `Plan.md` **方案定案 + 历史执行记录**(§1~§4 契约/架构、§5 里程碑归档、§6 骨架、§7 待办快照,**冻结不再新增**);`Plan2.md` **新计划的制定地**(2026-09-19 起,改决策先读 Plan2.md 再改)

## 常用命令

```bash
# CARLA 服务器(专用用户 carla + LD_PRELOAD shim,GPU 修复栈;headless)
bash bin/carla_server.sh        # 启动;停止用 stop(start/stop/status;勿手敲 pkill -f CarlaUE4,自匹配坑见 C19)

# 场景采集(KITTI root,含 label_2 GT/velodyne/calib)
python bin/collect_drive.py --scene rain_night --frames 70
python bin/collect_ab_route.py --scene sunset_glare --frames 70   # P1 A/B 专用:锚定 pt0 + 4 静置车

# 静态 GT(landmark + 车道线,含 overlay 目检图)
python bin/collect_static_gt.py --frames 40

# 灯色动态 GT(记录模式默认不动灯;--cycle 绿,黄,红 秒数 = 受控切灯)
python bin/collect_tl_states.py --frames 40 --speed 8
python bin/collect_tl_states.py --frames 90 --speed 8 --cycle 6,2,6

# 实时可视化(本地 ssh -L 8080:127.0.0.1:8080 <autodl> → 浏览器 127.0.0.1:8080)
python bin/view_stream.py --view follow --npcs        # 跟车视角
python bin/view_stream.py --view top --map Town13     # 俯视看街区
python bin/view_stream.py --scene rain_night --speed 8  # 天气 + 定速直行
python bin/view_stream.py --view follow --dump outputs/dumps/f.png  # 落 raw+overlay 做差集诊断
# MapTR 实时预测 overlay(需 --view grid6;投影链与离线 viz 共用 autodrivedata/mapviz)
PYTHONPATH=$PWD python bin/view_stream.py --view grid6 --maptr-ckpt outputs/maptr_ep512.pt --maptr-bev --dump outputs/dumps/m.png
# 8 路 studio(6 相机 + BEV + 第三方 + 拼图;WASD 操控 + 在线 SLAM)
python bin/live_studio.py                             # 8 路 + 键盘(stdin 是 tty 时默认开)
python bin/live_studio.py --speed 8 --npcs            # 定速直行(键盘自动关;两者互斥会报错)
python bin/live_studio.py --slam --speed 8 --duration 90 --slam-report outputs/slam_gt/accept.json
python bin/live_studio.py --maptr-ckpt outputs/maptr_ep512.pt --slam --speed 8  # 感知 + SLAM 同屏
# 落一段八视角视频(--video-fps 调到接近实际采集 fps 才是实时播放;结束会打印实测 fps 与倍速)
PYTHONPATH=$PWD python bin/live_studio.py --npcs --speed 6 --duration 100 --fps 10 --no-keyboard \
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
- **同步模式首个 `get_actors()` 为空**(快照只在 tick 后刷新)→ 清场会静默漏清;已修在 `carla_common.sync_mode()`(apply_settings 后补 tick),勿绕过它自己 apply_settings
- **快照陈旧不止 `get_actors`,传感器 `get_transform()` 同样**:tick 前读到的相机位姿全是 0(不是真实挂点)。view_stream 的 rig 自检曾因此假报 179.841°(= CAM_BACK 规格 180 − ego 固有 yaw 0.159),加一次 `world.tick()` 后 0.000°。**任何"实挂位姿 vs 规格"的判据都必须先 tick**
- **投影链的出口口径是弧度**:`autodrivedata/mapviz.cam_pose` 位置米 / 姿态弧度(与 `calib.world_to_img` 一致);把度直接传进去时 6 相机里**只有 yaw≈0 的 CAM_FRONT 看着正常**,侧/后相机全错位。判据不看图,看"命中点落在该相机自身 FOV 内的比例"(弧度 91–100% vs 度数 0–6%)——**"能画出图"不是投影正确的证据**
- **灯态收集侧要前向过滤**:圆形 horizon 会把身后 120m 的灯全收进来(实测占 79%),`traffic_light_frame(forward_only=True)` 是默认口径
- **产出必须落在项目内**:写盘路径一律经 `autodrivedata/paths.project_path()`(**相对路径 = 相对项目根**,不随 cwd 漂移;绝对路径原样放行)。历史口径"相对 cwd 的 outputs/"在换 cwd/换会话时会把权重与可视化散到项目外(清盘时无从分辨)。运行支撑物同理:shim/兼容层/服务器日志在 `outputs/carla/`(原 /tmp 与 carla_home 副本已废弃)。**读路径不锚定**(输入沿用 cwd 口径,便于临时 `cd`)
- **官方栈复线(§5.12)已终止**(2026-09-14 用户裁决;**不要主动重提**)。重启前先读 Plan.md §5.12:四个钉子(`bs1×累积5` 口径 / config 必须钉 `color_type="color"`、否则在线评测一开就炸 / v1 同位姿帧放行判据 / 预算对齐基线**第一轮** 256 ep)、独立评测的四个坑(单卡 `assert False`、`init_dist` 强制 spawn + `dict_keys`、产物路径相对 cwd、dist_test.sh 硬编码 `--eval bbox`)、A′ 口径结论(官方 eval_map 100 点 GT 重采样 = 0.0699 才是参照值)。**成本在 BEV transformer 不在主干**——降分辨率换不到吞吐(0.5 反而更慢)
- **停训练必须连 DataLoader worker 一起收**:worker 是 fork 出来的,而 fork 发生在 CUDA 初始化**之后** ⇒ worker **继承 CUDA 上下文**,父进程被杀后变 PPID=1 的孤儿**继续占显存**(nvidia-smi 仍把额度挂在已死的父 PID 名下,实测 `kill` 父进程后仍占 5068 MiB)→ 收完所有相关 PID 后 GPU 才归零。**判据:`nvidia-smi` 归零才算停干净,不是"父进程没了"**
- **纯 Python 主循环里别指望 worker 线程**:studio 主线程每 tick 的 GT overlay / `compose_grid` / HUD / 灯态绘制全是**字节码**,持 GIL 不放 ⇒ 同一对点云 ICP 在 worker 线程 eff **0.04–0.24** vs 主线程同步 **0.90–1.00**(差 10–20×)。**判据看 `time.thread_time()/wall`(eff),不是 wall 单值**;`OPENBLAS_NUM_THREADS=1` 不改结论 ⇒ 与 BLAS 线程池无关。用 PIL/CARLA tick 施负载测不出(它们会释放 GIL)—— 必须用**纯 Python 小矩阵自旋**才能复现。同理:**有界队列有界的是深度不是"状态间隙"**,任何"上一帧 vs 当前帧"成本随间隙超线性的算法(ICP/配准/图优化),丢帧都会变成"丢帧→间隙更大→更慢→更多丢帧"的正反馈,**必须按帧号差止损**(`SlamWorker.max_gap`)
- 提交:Conventional Commits;**提交信息不附 AI 署名**(不加 `Co-Authored-By: Claude` 等 trailer);改动后 `ruff check && ruff format` + 相关单测;决策与执行记录同步进 **Plan2.md**(教程能力线另同步 docs/milestone2.md);Plan.md 已冻结,只保留定案与历史记录
- **格式口径已定死**:`[tool.ruff]` 在 pyproject(line-length 110 / select E,F,I,UP,B / ignore E501,E741),`ruff format` 是唯一 formatter;批量纯格式提交要追加到 `.git-blame-ignore-revs`

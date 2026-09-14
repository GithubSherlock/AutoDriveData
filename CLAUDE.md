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
  - **实时 overlay ✅**(§5.11f):`view_stream.py --maptr-ckpt outputs/maptr_ep512.pt --maptr-bev`(rig 只认 `collect_surround.SURROUND_CAMS` 一处定义;推理用实挂相机世界位姿);实况数值验证 overlay 品红 25553 px vs raw 0、地平线以上 0/18496、FPS 1.1–1.4
  - **逐帧契约 ✅**(§5.11h):`eval_maptr.py --out-frames` → `outputs/surround_pred/{token}.json`(`mapvec_pred/1`:schema/帧归属/类序/坐标系/窗口/阈值/溯源,GT 同文件携带;旧 `--out-pred` 是跨帧汇聚、实例无帧归属,不可对外)。供 AutoLabel 消费——**消费方未接**
- **官方栈对照已终止**(2026-09-14,§5.12):官方 MapTR/MapQR 复线(同数据重训对照)中止,env + 产物已删(回收 15 G,磁盘 45 G 可用);**自实现线全部资产不受影响**(ep512 权重/逐帧契约/实时 overlay 照旧)。A′ 的**口径**结论保留(两套 mAP 口径的关系),但"官方实现 vs 自实现"的对照表不存在
- **P1-6 候选**:wet_road 眩光 / dense_rush 遮挡(待用户定)

## 环境(勿新建;direnv 进入目录自动激活 autodrivedata,首次需 `direnv allow`)

| 环境 | Python | 用途 |
|---|---|---|
| **autodrivedata**(本项目) | 3.11.16 | pycarla + ultralytics;采集 `bin/collect_*.py`、2D 评估 eval_2d_ab.py、3D 比对 eval_kitti.py、全部单测 |
| **autolabel** `/root/miniconda3/envs/autolabel` | 3.11.15 | mmdet3d;3D 检测 `auto3dlabel run`、oracle 对比 |
| **base** | 3.10.8 | conda 底座 + direnv;pycarla/ultralytics 已于 2026-09-10 迁出,不承担项目职责 |
| ~~**maptr_official**~~ **已删**(2026-09-13 建 → 2026-09-14 终止并删除,§5.12) | ~~3.8~~ | 官方 MapTR/MapQR 老栈(torch1.9.1+cu111 / mmcv-full1.4.0 / mmdet2.14.0)。**用户裁决整条官方复线中止**(41 h 训练预算仍过长)→ env 与产物已删、回收 15 G。**重建设路子**:`bin/setup_maptr_official.sh all` + Plan.md §5.12(含四个钉子/两处 bug/独立评测四坑/A′ 口径结论,知识都留在文档里) |

纪律:autodrivedata 包**不 import carla**(纯值,任何 env 可单测);依赖单向 AutoDriveData → AutoLabel(3D 检测消费方),禁止反向。

## 项目结构

- `autodrivedata/` — 纯值库(geometry/calib/gt/static_gt/traffic_light/attribution/semantic/export/compare/scenarios/paths/mapvec_schema),不 import carla
- `bin/` — carla 采集器(collect_drive/collect_ab_route/collect_static_gt/collect_tl_states/collect_nus)+ 评估(eval_2d_ab/eval_attr/eval_kitti)+ 可视化(view_stream)+ `carla_common.py`(位姿/NPC/传感器/灯态归一与绘制共用件)+ `probe_vulkan.py`(Vulkan 设备枚举,CARLA 渲染停摆的一线判据)+ `carla_server.sh`(GPU 修复版启动 + **Vulkan 兼容层自愈**;宿主驱动升版致 `libnvidia-gpucomp.so.<ver>` 缺失时自动顶名,见 Plan.md §5.11f)
- `tests/` — 单测(autodrivedata env,213 passed / 3 skipped)
- `outputs/` — **全部产物的唯一落点**(采集/权重/可视化/运行支撑物):kitti_* 为 KITTI root 结构;kitti_ab_* = P1 A/B 序列;kitti3d_ab_* = 3D 伪标签;`maptr_*.pt` = 权重;`carla/` = 服务器日志 + shim + Vulkan 兼容层
- `docs/milestone.md` — 版本里程碑;`Plan.md` — **单一事实源**(方案定案/执行记录/待办全在此,改决策先读再改)

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
- 提交:Conventional Commits;**提交信息不附 AI 署名**(不加 `Co-Authored-By: Claude` 等 trailer);改动后 `ruff check && ruff format` + 相关单测;决策与执行记录同步进 Plan.md
- **格式口径已定死**:`[tool.ruff]` 在 pyproject(line-length 110 / select E,F,I,UP,B / ignore E501,E741),`ruff format` 是唯一 formatter;批量纯格式提交要追加到 `.git-blame-ignore-revs`

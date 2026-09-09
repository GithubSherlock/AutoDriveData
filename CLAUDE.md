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
- **P2 ✅**(f66558c):静态 GT = **地图查询 API**(信号/标志是 landmark、车道线是 lane_marking 实体;semantic LiDAR 打不到)→ [autodrivedata/static_gt.py](autodrivedata/static_gt.py) + [scripts/collect_static_gt.py](scripts/collect_static_gt.py),落盘 `training/static_gt/{fid}.json` + overlay 目检图。**更正**:Town10HD_Opt 有 15 个 traffic_light actor(xodr 17 个 dynamic 信号),原记"无信号 actor"有误——灯色属动态 GT,仍不入静态 json
- **M4 挂起**(§5.7a 用户裁决):定制街道 = CARLA **源码构建**(prebuilt 无 UnrealEditor,~170G 磁盘/Epic 账号),成本过载降级为扩展 P1。开源 AdditionalMaps(Town11-15,14.8G)已探明可下载,零构建扩地图池
- **地图池扩展 ✅**(§5.7d):AdditionalMaps 14.8G 已装,Town11/12/13/15 入池(17 图)。**新图采集约束:Town11/12 禁采集**(spawn camera segfault)、Town13 TM 车流降级 0 NPC、可用 Town13/15;默认图仍 Town10HD_Opt(重启即恢复)
- **可视化实时流 ✅**(§5.8):[scripts/view_stream.py](scripts/view_stream.py) 自建 MJPEG(3 视角 + GT 框/灯色 overlay,只绑 127.0.0.1 走 SSH 隧道);carlaviz/RViz2 出局(非 UE 渲染 + 版本/依赖不成立);`world_to_img` 上移 [autodrivedata/calib.py](autodrivedata/calib.py) 供采集器与实时流共用
- **灯色动态 GT ✅**(§5.9):[autodrivedata/traffic_light.py](autodrivedata/traffic_light.py)(纯值:状态归一/前向判据/相位查表/JSON 往返)+ [scripts/collect_tl_states.py](scripts/collect_tl_states.py)(记录模式 / `--cycle 6,2,6` 受控切灯 → 确定性变灯序列),落盘 `training/traffic_light/{fid}.json`。工业口径:灯态 = 独立时序层,Off/Unknown **不猜**;受控 90 帧状态变化点 = 帧 0/60/80 与计划逐帧吻合。**边界:不做视觉回归**——镜片 0.2m 在 f=621 下 30m 处仅约 4px,且黄色灯箱外壳同色相
- **P1 参数扫描 + 失效归因 ✅**(§5.10):[autodrivedata/attribution.py](autodrivedata/attribution.py) 纯值(逐帧匹配/分箱/逐帧速度自证)+ [scripts/eval_attr.py](scripts/eval_attr.py)(多跑 × 距离/框高/TTC 网格 + 漏检画像),与 AP 共用同一 `box_iou2d`。三结论:**尺度主导**(<32px 0.15–0.47 / ≥32px 0.78–1.00,断崖 ≈21–24px)、**CARLA 无运动模糊**(4/8/12 m/s 梯度能量 35.6/35.2/34.8,检出率 0.914/0.886/0.909 → 速度不改图像,退化只能人工注入)、**天气只前移断崖**(雨夜 40-50m 零检出→30-40m 0.32,雾最晚 0.91)
- **P1-6 候选**:wet_road 眩光 / dense_rush 遮挡(待用户定)

## 环境(双环境,勿新建)

| 环境 | Python | 用途 |
|---|---|---|
| **base**(当前 shell) | 3.10.8 | pycarla + ultralytics;采集 `scripts/collect_*.py`、2D 评估 eval_2d_ab.py、3D 比对 eval_kitti.py、全部单测 |
| **autolabel** `/root/miniconda3/envs/autolabel` | 3.11.15 | mmdet3d;3D 检测 `auto3dlabel run`、oracle 对比 |

纪律:autodrivedata 包**不 import carla**(纯值,两 env 可单测);依赖单向 AutoDriveData → AutoLabel(3D 检测消费方),禁止反向。

## 项目结构

- `autodrivedata/` — 纯值库(geometry/calib/gt/static_gt/traffic_light/attribution/semantic/export/compare/scenarios),不 import carla
- `scripts/` — carla 采集器(collect_drive/collect_ab_route/collect_static_gt/collect_tl_states/collect_nus)+ 评估(eval_2d_ab/eval_attr/eval_kitti)+ 可视化(view_stream)+ `carla_common.py`(位姿/NPC/传感器/灯态归一与绘制共用件)+ `carla_server.sh`(GPU 修复版启动)
- `tests/` — 单测(base env,178 passed / 3 skipped)
- `outputs/` — 采集产物(kitti_* 为 KITTI root 结构;kitti_ab_* = P1 A/B 序列;kitti3d_ab_* = 3D 伪标签)
- `docs/milestone.md` — 版本里程碑;`Plan.md` — **单一事实源**(方案定案/执行记录/待办全在此,改决策先读再改)

## 常用命令

```bash
# CARLA 服务器(专用用户 carla + LD_PRELOAD shim,GPU 修复栈;headless)
bash scripts/carla_server.sh        # 启动;shutdown: pkill -f CarlaUE4

# 场景采集(KITTI root,含 label_2 GT/velodyne/calib)
python scripts/collect_drive.py --scene rain_night --frames 70
python scripts/collect_ab_route.py --scene sunset_glare --frames 70   # P1 A/B 专用:锚定 pt0 + 4 静置车

# 静态 GT(landmark + 车道线,含 overlay 目检图)
python scripts/collect_static_gt.py --frames 40

# 灯色动态 GT(记录模式默认不动灯;--cycle 绿,黄,红 秒数 = 受控切灯)
python scripts/collect_tl_states.py --frames 40 --speed 8
python scripts/collect_tl_states.py --frames 90 --speed 8 --cycle 6,2,6

# 实时可视化(本地 ssh -L 8080:127.0.0.1:8080 <autodl> → 浏览器 127.0.0.1:8080)
python scripts/view_stream.py --view follow --npcs        # 跟车视角
python scripts/view_stream.py --view top --map Town13     # 俯视看街区
python scripts/view_stream.py --scene rain_night --speed 8  # 天气 + 定速直行
python scripts/view_stream.py --view follow --dump /tmp/f.png  # 落 raw+overlay 做差集诊断

# 2D A/B 评估(base env;A=day_clear 基线与 B 帧级配对)
python scripts/eval_2d_ab.py --root-a outputs/kitti_ab_day_clear --root-b outputs/kitti_ab_sunset_glare

# 失效归因(base env;逐帧匹配 → 距离/框高/TTC 分箱 + 漏检画像)
python scripts/eval_attr.py --run day8=outputs/kitti_sweep_day_clear_8:8.0 \
  --run rain=outputs/kitti_ab_rain_night:8.0 --json outputs/attr.json

# 3D LiDAR 检测(autolabel env;**cwd 必须在 AutoLabel 根**,config 相对路径)
cd /root/autodl-tmp/Documents/Projects/AutoLabel && KITTI_OBJECT_ROOT=<abs kitti root> \
  /root/miniconda3/envs/autolabel/bin/auto3dlabel run 000000-000069 "检测汽车" \
  --det-model pointpillars_kitti --batch --no-viz --out-dir <abs out>
python scripts/eval_kitti.py --root outputs/kitti_ab_x --pred outputs/kitti3d_ab_x   # 3D 比对(base env)

# 规范 + 测试(提交前两件套;规则集钉死在 pyproject [tool.ruff],110 列)
ruff check && ruff format        # format 无参数即就地格式化,全仓口径统一
python -m pytest tests/ -q
```

## 红线(A/B 实验纪律与已踩坑,勿再犯)

- **A/B 帧级配对是硬门槛**:同 ego 锚定 spawn point 0(yaw=0)、同静置车布局(20/35/50/62m——65m 会卡 GT max_distance 阈值抖动)、只变 weather;GT 数必须相等,否则样本不可比、结论作废。autopilot/TM 路线不可复现,禁用于 A/B
- **AP 尾部不注水**:未达 recall=1 段 precision=0(11 点插值,与 compare.ap11 同口径)。旧尾行 `ap += (1-prev_r)*prev_p` 曾把低 recall 吹高(雨夜 0.48 检出报 0.976),已修
- **carla pyi 桩坑**:`try_spawn_actor` 桩标返回 `Actor`(实为 `Actor|None`)→ 用 Vehicle 方法必须 `cast(carla.Vehicle, v)`;Vector3D 运算结果不能直接进 `carla.Transform`(显式 `carla.Location`);函数签名要 `tuple[float, float, float]` 定长时禁用 tuple 推导(变长 tuple)
- **sunset_glare 方位**:az=90=东=+x=车头正前(yaw=0 时);az=300 是顺光陷阱(太阳在车后)。判据 = 全图过曝最低(AE 压最狠)
- **采集器清场 + 起点校验**:残留 actor 阻塞 spawn point 会致 fallback 反向出生点、轨迹失配(collect_ab_route/collect_static_gt 已内置,勿删)
- **锚定 yaw 用 spawn point 固有 rotation**:collect_static_gt 曾硬编码 yaw=0,Town10HD_Opt pts[0] 固有 yaw=0.16° 恰好成立、Town13(125.9°)车道线采样走到车后 overlay 全空;已改用 pts[0].rotation(沿车道),collect_ab_route 的 yaw=0 是 P1 已验证基线勿动
- **"定速"必须清制动残留 + 逐帧自证**:`VehicleControl` 一旦设置就每步生效,`brake=1.0` 站定后不解除会让 `set_target_velocity` 打 0.82 折(P1 四个老数据集实为 6.60 m/s 而非 8.0,已修);速度/时序结论**用逐帧序列测**(`closing_speed_series`),不要从 ego-x 首末值反推(曾误判"2 秒加速段")
- **漏检归因先看框高箱再看亮度**:<32px 一律 0.15–0.47、≥32px 一律 0.78–1.00,漏检框内亮度与命中几乎相同——主因是尺度不是"暗";TTC 箱**不可跨速度比检出率**(同箱在不同速度对应不同距离)
- 模型无法读图时用**数值诊断**(亮度带/过曝占比/梯度),不要硬目检;验证 overlay 必须做**同帧 raw/overlay 差集**——场景自带绿(植被)/黄(标线)与类别色撞色,数绝对颜色会误判(曾报"Car 仅 3 px")
- **灯态 GT 不做视觉回归**:镜片 0.2m,在 KITTI 口径相机(f=621)下 30m 处仅约 4px;12–30m 处按颜色采样命中的是**黄色灯箱外壳**(≈(255,237,0),与黄灯镜片同色相)。真值取自 actor API(逻辑层)。要拍镜片必须按灯头盒**薄轴**放相机(盒 yaw 方向拍的是背面,曾据此误判"渲染不随 set_state 变")
- **同步模式首个 `get_actors()` 为空**(快照只在 tick 后刷新)→ 清场会静默漏清;已修在 `carla_common.sync_mode()`(apply_settings 后补 tick),勿绕过它自己 apply_settings
- **灯态收集侧要前向过滤**:圆形 horizon 会把身后 120m 的灯全收进来(实测占 79%),`traffic_light_frame(forward_only=True)` 是默认口径
- 提交:Conventional Commits;**提交信息不附 AI 署名**(不加 `Co-Authored-By: Claude` 等 trailer);改动后 `ruff check && ruff format` + 相关单测;决策与执行记录同步进 Plan.md
- **格式口径已定死**:`[tool.ruff]` 在 pyproject(line-length 110 / select E,F,I,UP,B / ignore E501,E741),`ruff format` 是唯一 formatter;批量纯格式提交要追加到 `.git-blame-ignore-revs`

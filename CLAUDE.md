# AutoDriveData

CARLA 0.9.16 → AutoLabel 自动驾驶数据输出流水线:自定义地图/场景采集车检测动态/静态目标与道路特征,构建 0→1 数据输出。当前主线 = **Corner Case 场景矩阵定量验证**(P1)+ 静态 GT(P2);raw 输出供 AutoLabel 验证场 + 长尾数据源。

## 开发前提

- 绝对不要创建子代理。无明确用户指示或确认，严禁使用任何子代理或委派任务！

## 当前状态

**目录结构(2026-09-26 重构完成)**:`bin/` 与 `tests/` 顶层目录**已删除**,全部收进 [`autodrivedata/`](autodrivedata/) 主包,
按**能力面**分 10 个目录,包根只剩 `__init__.py`。**可执行入口一律 `python -m autodrivedata.<能力>.<模块>`**。

| 主题 | 结论 | 详述 |
|---|---|---|
| **P1** corner case 矩阵 | 相机三型可量化掉点(逆光 −0.014 轻 / 雨夜 **−0.153 漏检型** / 浓雾 −0.013 FP 型);**LiDAR 不受天气光照** | [Plan.md](Plan.md) §5.7c |
| **P2** 静态 GT | 信号/标志是 landmark、车道线是 lane_marking;**semantic LiDAR 打不到** ⇒ 只能走地图查询 API | Plan.md §5.7c |
| 灯色动态 GT | 灯态 = **独立时序层**,Off/Unknown **不猜**;**不做视觉回归**(镜片 30m 处仅 ~4px) | Plan.md §5.9 |
| 失效归因 | **尺度主导**(<32px 0.15–0.47 vs ≥32px 0.78–1.00,断崖 ≈21–24px);CARLA **无运动模糊**(退化只能人工注入);天气只**前移断崖** | Plan.md §5.10 |
| MapTR 矢量管道 | 参考自实现打通。chamfer AP @`--score-thr 0.2`:**0.3043** 帧级留出 / **0.1114** 路线级留出 | [Plan2.md](Plan2.md) §P-M.12 |
| 8 路实时 studio | 拼图**每格原生像素不缩放**;在线 SLAM **默认同步执行**(worker 被 GIL 饿死,eff 0.04–0.24 vs 同步 0.90–1.00) | Plan2.md §P-L |
| 环视标定 | **像素约定 = CORNER**;ego 原点 = **后轴**(`NUS_EGO_ORIGIN_X = −1.2563`);ego 姿态 = **全 6DoF** | Plan2.md §P-M.10/.11 |
| 全传感器「声明 ≠ 渲染」 | 渲染位姿与声明位姿**由同一份常量导出**;十条判据两代 rig 全过 | Plan2.md §P-M.7 |
| 中间件线 | SLAM / 单双目 / 多雷达 / 语义建图 / 3DGS / 轨迹 — 教程 11–16 全部打通 | [docs/milestone2.md](docs/milestone2.md) |
| 官方栈复线 | **已终止**(2026-09-14 用户裁决),**不要主动重提** | Plan.md §5.12 |
| P1-6 候选 | wet_road 眩光 / dense_rush 遮挡(**待用户定**) | — |

> ⚠️ **三条不许误读**(MapTR 结果口径,详见 Plan2.md §P-M.12):
> ① **帧级留出 ≠ 泛化** —— 留出首帧与训练末帧**同街只隔 2.65–3.01 m**,真泛化看路线级(差 **2.7×**);
> ② 该 AP 口径是 **precision 均值、无 recall 项** —— 不能用来算"训练/留出差距";
> ③ 帧级留出还叠了 **GT 密度**混淆(11.34 vs 7.87 实例/帧)。
> **改动未提交 ≠ 待办** —— 引用历史结论前先 `git status`,别照着旧记录找不存在的未提交改动。

## 环境(勿新建;direnv 进入目录自动激活 autodrivedata,首次需 `direnv allow`)

| 环境 | Python | 用途 |
|---|---|---|
| **autodrivedata**(本项目) | 3.11.16 | pycarla + ultralytics;采集 `python -m autodrivedata.sim.collect_*`、2D 评估 `perception.eval_2d_ab`、3D 比对 `perception.eval_kitti`、全部单测 |
| **autolabel** | 3.11.15 | mmdet3d;3D 检测 `auto3dlabel run`、oracle 对比 |
| **hivt** | 3.8.20 | HiVT 复现栈(torch1.8 / pl1.5 / pyg1.7 / argoverse-api),CPU 推理。**未注册进 conda envs_dirs** ⇒ `conda info --envs` 看不到、`activate hivt` 失败,**只能绝对路径调** `envs/hivt/bin/python`(见 [autodrivedata/traj/convert_hivt_pt.py](autodrivedata/traj/convert_hivt_pt.py) 用法头)。数据盘 2.5 G,勿删 |
| **base** | 3.10.8 | conda 底座 + direnv;pycarla/ultralytics 已于 2026-09-10 迁出,不承担项目职责 |

**env 落点**:`conda config --show envs_dirs` = `/root/miniconda3/envs` + `/root/.conda/envs` —— **只有这两个目录下的环境才被 conda 按名字发现**。
`autodrivedata` 与 `autolabel` 都是「数据盘真身 + `/root/miniconda3/envs/` 下的符号链接」(链接 0 字节,不占额外空间,conda 可见);
`hivt` 只有数据盘真身、**没有链接** ⇒ conda 看不见。**三者都不是副本,不存在重复占盘。**
(曾经的 `maptr_official` env 已随官方栈终止删除,见 Plan.md §5.12。)

**分层纪律**:包内「谁允许 import 什么」由 [autodrivedata/tests/test_layer_guard.py](autodrivedata/tests/test_layer_guard.py) 的
`LAYER_RULES` **机械强制**(按目录声明,四档 `_PURE` / `_CARLA_OK` / `_TORCH_OK` / `_ANY`,最长前缀匹配)。
依赖单向 AutoDriveData → AutoLabel(3D 检测消费方),**禁止反向**。

## 项目结构

> 📁 **文件级索引见 [docs/fileTree.md](docs/fileTree.md)**(每个文件的职责、产物落点、【未入库】标注)。
> **新增或改名文件后回来补一行**;下面是能力面速览。

```
autodrivedata/          ★ 主包 —— 按能力面分层,包根只有 __init__.py
├── sim/          22  CARLA 仿真交互层(唯一大面积 import carla 的地方)+ 全部采集器
├── calib/        14  标定:原语(core.py = 原 calib.py)/ rig 表 / 自证探针 / 实时监看 / 配置图
├── map/          29  地图矢量 + MapTR(含 maptr/ = 原 maptr_impl/、maptr_official/ = 已终止线)
├── slam/         11  两段式激光 SLAM(core.py = 原 slam.py)+ 精度评估 + slam_cpp.cpp
├── perception/   17  检测 / 单双目 / 雷达 / 语义 / 点云(**不 import carla**)
├── gt/            6  动态目标 + 静态目标 + 灯态时序层 + export/ 落盘
├── traj/  gs/     3  轨迹组装转换 / 3DGS 训练
├── utils/         3  通用件:geometry.py paths.py fonts.py(**无领域语义、无 carla/torch**)
└── tests/        48  与能力目录镜像(包级守卫 test_layer_guard.py 在根)

tools/                 开放性工具(判据:不含本项目领域知识):carla_server.sh + gpu_fix/
docs/(含 fileTree.md / refactor-2026-09.md)  README.md  Plan.md(冻结)  Plan2.md(新计划制定地)
outputs/  training/  lightning_logs/  hdMapGitHub/  auto3dlabel/   【未入库】产物与上游克隆
```

**命名约定**:目录名 = 能力面;模块名 = 能力内的构件。**模块与所在目录同名时改名 `core.py`**
(`calib/core.py`、`slam/core.py`、`gt/core.py`)—— 避免 `autodrivedata.calib.calib` 这类自反名。
**`utils/` 准入判据**:无项目领域语义、无 carla/torch 依赖;超过 6 个文件即视为 junk drawer,需重新裁决。

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
# MapTR 实时预测 overlay(需 --view grid6;投影链与离线 viz 共用 autodrivedata.map.mapviz)
python -m autodrivedata.sim.view_stream --view grid6 --maptr-ckpt outputs/maptr_v2_singleF.pt --maptr-bev --dump outputs/dumps/m.png
# 8 路 studio(6 相机 + BEV + 第三方 + 拼图;WASD 操控 + 在线 SLAM)
python -m autodrivedata.sim.live_studio                             # 8 路 + 键盘(stdin 是 tty 时默认开)
python -m autodrivedata.sim.live_studio --speed 8 --npcs            # 定速直行(键盘自动关;两者互斥会报错)
python -m autodrivedata.sim.live_studio --slam --speed 8 --duration 90 --slam-report outputs/slam_gt/accept.json
python -m autodrivedata.sim.live_studio --maptr-ckpt outputs/maptr_v2_singleF.pt --slam --speed 8  # 感知 + SLAM 同屏
# 落一段八视角视频(--video-fps 调到接近实际采集 fps 才是实时播放;结束会打印实测 fps 与倍速)
python -m autodrivedata.sim.live_studio --npcs --speed 6 --duration 100 --fps 10 --no-keyboard \
  --maptr-ckpt outputs/maptr_v2_singleF.pt --slam --video outputs/videos/studio_8view.mp4 --video-fps 0.5
python -m autodrivedata.map.viz_maptr_pred --start 250 --frames 6   # 离线:预测回投 6 相机拼图 + BEV
# 逐帧契约落盘(供 AutoLabel 消费;GT 同文件携带,见 autodrivedata/map/mapvec_schema.py)
python -m autodrivedata.map.eval_maptr --infos outputs/surround_v2/map_infos.json \
  --root outputs/surround_v2 --ckpt outputs/maptr_v2_singleF.pt --start 200 --out-frames outputs/surround_pred

# 2D A/B 评估(A=day_clear 基线与 B 帧级配对)
python -m autodrivedata.perception.eval_2d_ab --root-a outputs/kitti_ab_day_clear --root-b outputs/kitti_ab_sunset_glare

# 失效归因(逐帧匹配 → 距离/框高/TTC 分箱 + 漏检画像)
python -m autodrivedata.perception.eval_attr --run day8=outputs/kitti_sweep_day_clear_8:8.0 \
  --run rain=outputs/kitti_ab_rain_night:8.0 --json outputs/attr.json

# 3D LiDAR 检测(autolabel env;**cwd 必须在 AutoLabel 根**,config 相对路径)
cd /root/autodl-tmp/Documents/Projects/AutoLabel && KITTI_OBJECT_ROOT=<abs kitti root> \
  /root/miniconda3/envs/autolabel/bin/auto3dlabel run 000000-000069 "检测汽车" \
  --det-model pointpillars_kitti --batch --no-viz --out-dir <abs out>
python -m autodrivedata.perception.eval_kitti --root outputs/kitti_ab_x --pred outputs/kitti3d_ab_x   # 3D 比对

# nuScenes 迷你集(全传感器「渲染 = 声明」同源;重采后跑验收)
python -m autodrivedata.sim.collect_nus --frames 2                     # 重采(需 CARLA 在跑)
python -m autodrivedata.calib.verify_nus_calib --offline --live        # 十条判据(**--live 需 CARLA**)→ outputs/nus_calib_check/
bash autodrivedata/sim/smoke_radar_collect.sh                          # devkit 直读四判据

# wide rig(挂点后移 + 新 FoV,画幅内零车体像素;官方口径仍是默认)
python -m autodrivedata.sim.collect_nus --rig wide --out outputs/nus_mini_wide --frames 2
python -m autodrivedata.calib.verify_nus_calib --rig wide --offline --live   # → report_wide.json
python -m autodrivedata.calib.viz_rig_check --rig wide --live     # → outputs/calib_check/{rig_layout_*,views_*,report_*}

# 规范 + 测试(提交前两件套;规则集钉死在 pyproject [tool.ruff],110 列)
ruff check && ruff format        # format 无参数即就地格式化,全仓口径统一
python -m pytest -q              # testpaths 已钉在 pyproject;**别裸敲 pytest 之外的路径前缀**
                                 # 基线:925 收集项(919 passed + 6 跳过 + 0 失败)
                                 # 基线数**只写在这一处**;加/删用例后回来改这一行,别在多处复述
```

## 红线(A/B 实验纪律与已踩坑,勿再犯)

- **A/B 帧级配对是硬门槛**:同 ego 锚定 spawn point 0(yaw=0)、同静置车布局(20/35/50/62m——65m 会卡 GT max_distance 阈值抖动)、只变 weather;GT 数必须相等,否则样本不可比、结论作废。autopilot/TM 路线不可复现,禁用于 A/B
- **AP 尾部不注水**:未达 recall=1 段 precision=0(11 点插值,与 compare.ap11 同口径)。旧尾行 `ap += (1-prev_r)*prev_p` 曾把低 recall 吹高(雨夜 0.48 检出报 0.976),已修
- **MapTR chamfer AP 必须带 score_thr 引用**;跨权重比较**固定 `--score-thr`**,看曲线用 `--sweep`;**单独报一个 mAP 数字而不写阈值 = 无效结论**
- **carla pyi 桩坑**:`try_spawn_actor` 桩标返回 `Actor`(实为 `Actor|None`)→ 用 Vehicle 方法必须 `cast(carla.Vehicle, v)`;Vector3D 运算结果不能直接进 `carla.Transform`(显式 `carla.Location`);函数签名要 `tuple[float, float, float]` 定长时禁用 tuple 推导(变长 tuple)
- **sunset_glare 方位**:az=90=东=+x=车头正前(yaw=0 时);az=300 是顺光陷阱(太阳在车后)。判据 = 全图过曝最低(AE 压最狠)
- **采集器清场 + 起点校验**:残留 actor 阻塞 spawn point 会致 fallback 反向出生点、轨迹失配(collect_ab_route/collect_static_gt 已内置,勿删)
- **锚定 yaw 用 spawn point 固有 rotation**:collect_static_gt 曾硬编码 yaw=0,Town10HD_Opt pts[0] 固有 yaw=0.16° 恰好成立、Town13(125.9°)车道线采样走到车后 overlay 全空;已改用 pts[0].rotation(沿车道),**collect_ab_route 的 yaw=0 是 P1 已验证基线勿动**
- **"定速"必须清制动残留 + 逐帧自证**:`VehicleControl` 一旦设置就每步生效,`brake=1.0` 站定后不解除会让 `set_target_velocity` 打 0.82 折(P1 四个老数据集实为 6.60 m/s 而非 8.0,已修);速度/时序结论**用逐帧序列测**(`closing_speed_series`),不要从 ego-x 首末值反推
- **漏检归因先看框高箱再看亮度**:<32px 一律 0.15–0.47、≥32px 一律 0.78–1.00,漏检框内亮度与命中几乎相同——主因是尺度不是"暗";TTC 箱**不可跨速度比检出率**
- 模型无法读图时用**数值诊断**(亮度带/过曝占比/梯度),不要硬目检;验证 overlay 必须做**同帧 raw/overlay 差集**——场景自带绿(植被)/黄(标线)与类别色撞色,数绝对颜色会误判(曾报"Car 仅 3 px")
- **灯态 GT 不做视觉回归**:镜片 0.2m 在 KITTI 口径相机(f=621)下 30m 处仅约 4px;12–30m 处按颜色采样命中的是**黄色灯箱外壳**。真值取自 actor API(逻辑层)。要拍镜片必须按灯头盒**薄轴**放相机
- **覆盖层文字一律走 [`autodrivedata.utils.fonts`](autodrivedata/utils/fonts.py),不许裸写 `d.text(...)`**:PIL 遇缺字**静默**画 `.notdef` 方框,且不传 `font=` 就用内置位图字体(无 CJK、仅 ~11 px)。判据是**渲染探针**(`U+10FFFF` 的像素签名),**不看文件名、不看 `fc-list`**;本机唯一 CJK 字体是 CARLA 随包的 `DroidSansFallback.ttf`。**长文本一律 `fonts.wrap()` 折行再画** —— 单行超出画布被 PIL **静默裁掉**。三条连带陷阱(比例拉丁下 `:<16` 对不齐 / 底条宽度不能按字符数估 / 测试像素魔数失效)与完整成因见 Plan2.md §P-M.8
- **同步模式首个 `get_actors()` 为空**(快照只在 tick 后刷新)→ 清场会静默漏清;已修在 `carla_common.sync_mode()`,**勿绕过它自己 apply_settings**
- **快照陈旧不止 `get_actors`,传感器 `get_transform()` 同样**:tick 前读到的相机位姿全是 0。**任何"实挂位姿 vs 规格"的判据都必须先 tick**
- **投影链的出口口径是弧度**:`autodrivedata/map/mapviz.cam_pose` 位置米 / 姿态弧度(与 `calib.core.world_to_img` 一致);把度直接传进去时 6 相机里**只有 yaw≈0 的 CAM_FRONT 看着正常**。判据不看图,看"命中点落在该相机自身 FOV 内的比例"(弧度 91–100% vs 度数 0–6%)——**"能画出图"不是投影正确的证据**
- **灯态收集侧要前向过滤**:圆形 horizon 会把身后 120m 的灯全收进来(实测占 79%),`traffic_light_frame(forward_only=True)` 是默认口径
- **方位角扇区跨 0° 时不许取模、PIL 染色不许就地改**:`az % 360` 会把右端折回最左 ⇒ `rectangle` 拿到 `x1 < x0` **直接抛 `ValueError`**(钳到 `[0,360]`);`np.asarray(PIL 图)` 是**只读**视图,必须 `np.array` 拷贝再 `Image.fromarray`。**两条都只有"新 rig / 官方 rig"才走到** ⇒ 出新图必须**两代 rig 各跑一遍**
- **"方位轴重叠"不等于"重叠区真的存在"**:方位轴是**无穷远**口径,挂点视差让同一世界点在两路里的方位角差最多 1.7°(官方 `FL↔BL` 11.20° → 12 m 处 4.838°,`B↔BL` 的 5.89° 整个消失)⇒ 判重叠要按**有限距离**算(`rig_check.common_band`)
- **产出必须落在项目内**:写盘路径一律经 [`autodrivedata/utils/paths.project_path()`](autodrivedata/utils/paths.py)(**相对路径 = 相对项目根**)。运行支撑物在 `outputs/carla/`。**读路径不锚定**(输入沿用 cwd 口径,便于临时 `cd`)
- **停训练必须连 DataLoader worker 一起收**:worker 在 CUDA 初始化**之后** fork、**继承 CUDA 上下文**,父进程被杀后变 PPID=1 孤儿**继续占显存**。**判据:`nvidia-smi` 归零才算停干净,不是"父进程没了"**
- **纯 Python 主循环里别指望 worker 线程**:主线程每 tick 的 overlay / `compose_grid` / HUD 全是**字节码**,持 GIL 不放 ⇒ 同一对点云 ICP 在 worker 线程 eff **0.04–0.24** vs 主线程同步 **0.90–1.00**。**判据看 `time.thread_time()/wall`(eff),不是 wall 单值**;同理**有界队列有界的是深度不是"状态间隙"** —— ICP 成本随间隙超线性,丢帧会变成正反馈,**必须按帧号差止损**(`SlamWorker.max_gap`)
- 提交:Conventional Commits;**提交信息不附 AI 署名**(不加 `Co-Authored-By: Claude` 等 trailer);改动后 `ruff check && ruff format` + 相关单测;决策与执行记录同步进 **Plan2.md**(教程能力线另同步 docs/milestone2.md);**Plan.md 已冻结**
- **格式口径已定死**:`[tool.ruff]` 在 pyproject(line-length 110 / select E,F,I,UP,B / ignore E501,E741),`ruff format` 是唯一 formatter;批量纯格式提交要追加到 `.git-blame-ignore-revs`

> **搬目录/改结构时注意** —— 本轮重构实测出 **11 类引用形态**,照单扫一遍再动手(清单与各类实例见
> [docs/refactor-2026-09.md](docs/refactor-2026-09.md) 的阶段 4 记录):`import X` / `from X import` / `import X as Y` /
> `from <包> import X` / `from autodrivedata.<m> import` / 路径字符串 / 硬编码源码路径常量 /
> **算自己位置的表达式**(`Path(__file__).parents[N]`、`cd "$(dirname "$0")/.."`)/ **方法体内的惰性 import**(逃过 `--collect-only`)/
> **同名多义**(包名 vs 输出目录 vs env 名)/ **`包/模块.属性` 散文写法**。
> 改完**必须真跑全量**并对用例数 —— 计数对账抓不到惰性 import,只有真跑能抓到。

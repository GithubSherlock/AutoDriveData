# AutoDriveData

CARLA 0.9.16 → AutoLabel 自动驾驶数据输出流水线:自定义地图/场景采集车检测动态/静态目标与道路特征,构建 0→1 数据输出。当前主线 = **Corner Case 场景矩阵定量验证**(P1)+ 静态 GT(P2);raw 输出供 AutoLabel 验证场 + 长尾数据源。

## 当前进度(2026-09-09)

- **P1 ✅**(fdfe065 起):场景目录 [autodrivedata/scenarios.py](autodrivedata/scenarios.py)(8 场景,不 import carla)+ `collect_drive --scene` 采集链 + 三 corner case A/B(统一 11 点插值 AP 口径,见 [Plan.md](Plan.md) §5.7c 汇总表):
  | corner case | 相机 Δ | LiDAR Δ |
  |---|---|---|
  | 逆光(az=90 校准) | -0.014 轻掉点(AE 补偿) | +0.003 噪声 |
  | 雨夜 | **-0.153 漏检型**(检出 0.72→0.48) | +0.017 噪声 |
  | 浓雾 | -0.013 FP 型(检出 0.72→0.85) | 0.000 |
  - 结论:相机三型可量化掉点;LiDAR 兜底不受天气/光照(平台边界:雨/雾无物理回波,退化只能人工注入)
- **P2 ✅**(1a3d357):静态 GT = **地图查询 API**(Town10HD_Opt 信号是 landmark 非 actor、车道线是 lane_marking 实体;semantic LiDAR 打不到)→ [autodrivedata/static_gt.py](autodrivedata/static_gt.py) + [scripts/collect_static_gt.py](scripts/collect_static_gt.py),落盘 `training/static_gt/{fid}.json` + overlay 目检图
- **M4 挂起**(§5.7a 用户裁决):定制街道 = CARLA **源码构建**(prebuilt 无 UnrealEditor,~170G 磁盘/Epic 账号),成本过载降级为扩展 P1。开源 AdditionalMaps(Town11-15,14.8G)已探明可下载,零构建扩地图池
- **P1-6 候选**:wet_road 眩光 / dense_rush 遮挡(待用户定)

## 环境(双环境,勿新建)

| 环境 | Python | 用途 |
|---|---|---|
| **base**(当前 shell) | 3.10.8 | pycarla + ultralytics;采集 `scripts/collect_*.py`、2D 评估 eval_2d_ab.py、3D 比对 eval_kitti.py、全部单测 |
| **autolabel** `/root/miniconda3/envs/autolabel` | 3.11.15 | mmdet3d;3D 检测 `auto3dlabel run`、oracle 对比 |

纪律:autodrivedata 包**不 import carla**(纯值,两 env 可单测);依赖单向 AutoDriveData → AutoLabel(3D 检测消费方),禁止反向。

## 项目结构

- `autodrivedata/` — 纯值库(geometry/calib/gt/static_gt/semantic/export/compare/scenarios),不 import carla
- `scripts/` — carla 采集器(collect_drive/collect_ab_route/collect_static_gt/collect_nus)+ 评估(eval_2d_ab/eval_kitti)+ `carla_server.sh`(GPU 修复版启动)
- `tests/` — 单测(base env,131 passed)
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

# 2D A/B 评估(base env;A=day_clear 基线与 B 帧级配对)
python scripts/eval_2d_ab.py --root-a outputs/kitti_ab_day_clear --root-b outputs/kitti_ab_sunset_glare

# 3D LiDAR 检测(autolabel env;**cwd 必须在 AutoLabel 根**,config 相对路径)
cd /root/autodl-tmp/Documents/Projects/AutoLabel && KITTI_OBJECT_ROOT=<abs kitti root> \
  /root/miniconda3/envs/autolabel/bin/auto3dlabel run 000000-000069 "检测汽车" \
  --det-model pointpillars_kitti --batch --no-viz --out-dir <abs out>
python scripts/eval_kitti.py --root outputs/kitti_ab_x --pred outputs/kitti3d_ab_x   # 3D 比对(base env)

# 测试
python -m pytest tests/ -q
```

## 红线(A/B 实验纪律与已踩坑,勿再犯)

- **A/B 帧级配对是硬门槛**:同 ego 锚定 spawn point 0(yaw=0)、同静置车布局(20/35/50/62m——65m 会卡 GT max_distance 阈值抖动)、只变 weather;GT 数必须相等,否则样本不可比、结论作废。autopilot/TM 路线不可复现,禁用于 A/B
- **AP 尾部不注水**:未达 recall=1 段 precision=0(11 点插值,与 compare.ap11 同口径)。旧尾行 `ap += (1-prev_r)*prev_p` 曾把低 recall 吹高(雨夜 0.48 检出报 0.976),已修
- **carla pyi 桩坑**:`try_spawn_actor` 桩标返回 `Actor`(实为 `Actor|None`)→ 用 Vehicle 方法必须 `cast(carla.Vehicle, v)`;Vector3D 运算结果不能直接进 `carla.Transform`(显式 `carla.Location`);函数签名要 `tuple[float, float, float]` 定长时禁用 tuple 推导(变长 tuple)
- **sunset_glare 方位**:az=90=东=+x=车头正前(yaw=0 时);az=300 是顺光陷阱(太阳在车后)。判据 = 全图过曝最低(AE 压最狠)
- **采集器清场 + 起点校验**:残留 actor 阻塞 spawn point 会致 fallback 反向出生点、轨迹失配(collect_ab_route/collect_static_gt 已内置,勿删)
- 模型无法读图时用**数值诊断**(亮度带/过曝占比/梯度),不要硬目检
- 提交:Conventional Commits;改动后 ruff + 相关单测;决策与执行记录同步进 Plan.md

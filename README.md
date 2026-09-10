# AutoDriveData

CARLA 0.9.16 → [AutoLabel](https://github.com/GithubSherlock/AutoLabel) 自动驾驶数据输出流水线:自定义地图/场景采集车检测动态/静态目标与道路特征,构建 0→1 数据输出。raw 输出供 AutoLabel 验证场 + 长尾数据源。

## 核心成果

**P1 Corner Case 场景矩阵**(帧级配对 A/B,统一 11 点插值 AP 口径):

| corner case | 相机 2D Δ AP | 掉点型 | LiDAR 3D Δ AP |
|---|---|---|---|
| 逆光(sunset_glare) | -0.014 | 轻掉点(AE 补偿) | +0.003 噪声 |
| 雨夜(rain_night) | **-0.153** | 漏检型(检出 0.72→0.48) | +0.017 噪声 |
| 浓雾(dense_fog) | -0.013 | FP 型(检出 0.72→0.85) | 0.000 |

结论:相机在三型真实驾驶长尾上均有可量化掉点;**LiDAR 兜底不受天气/光照**(平台边界:雨/雾对合成 LiDAR 无物理回波,退化只能人工注入)。

**失效归因**(逐帧匹配 + 每 GT 上下文,与 AP 共用同一 IoU 口径):

- **尺度主导**:框高 <32px 检出率一律 0.15–0.47,≥32px 一律 0.78–1.00,断崖 ≈21–24px(30–40m);漏检框内亮度与命中几乎相同 → 漏的是"小",不是"暗"
- **CARLA 无运动模糊**(平台边界):4/8/12 m/s 同距离箱梯度能量 35.6/35.2/34.8、池化检出率 0.914/0.886/0.909 → 速度不改变图像质量
- **天气只是把断崖前移**:雨夜零检出从 40–50m 提前到 30–40m(0.32),雾反而最晚(0.91)

**P2 静态 GT**:地图查询 API(landmark 信号 + lane_marking 车道线)→ `training/static_gt/{fid}.json` + overlay 目检图,与天气/光照解耦。

**灯色动态 GT**:灯态 = 独立时序语义层(工业口径,Off/Unknown 不猜)→ `training/traffic_light/{fid}.json`(逐帧状态 + 管制车道/停车线 + 相位计划);受控切灯 `--cycle 6,2,6` 给出**确定性变灯序列**。

**地图池**:17 图(Town01-10 + AdditionalMaps Town11/12/13/15,零构建)。

**实时可视化**:自建 MJPEG 流(真 UE 渲染 + GT 框/灯色 overlay,3 视角),浏览器直接看采集链所见画面——不依赖 carlaviz/RViz2(非 UE 渲染)。

## 快速开始

> 环境:进入项目目录自动激活 **autodrivedata** conda env(direnv + .envrc;首次 `direnv allow`)。VSCode 解释器已指向该 env。

```bash
# 1. CARLA 服务器(headless,GPU 修复栈;专用用户 carla)
bash bin/carla_server.sh

# 2. 场景采集(KITTI root:image_2 + label_2 GT + velodyne + calib)
python bin/collect_drive.py --scene rain_night --frames 70
python bin/collect_ab_route.py --scene sunset_glare --frames 70   # P1 A/B 专用

# 3. 静态 GT(landmark + 车道线,含 overlay 目检图)
python bin/collect_static_gt.py --frames 40

# 4. 灯色动态 GT(记录模式;--cycle 绿,黄,红 秒数 = 受控切灯)
python bin/collect_tl_states.py --frames 40 --speed 8
python bin/collect_tl_states.py --frames 90 --speed 8 --cycle 6,2,6

# 5. 实时可视化(自建 MJPEG:真 UE 渲染 + GT 框/灯色 overlay)
python bin/view_stream.py --view follow --npcs     # 本地 ssh -L 8080:127.0.0.1:8080 → 浏览器打开
python bin/view_stream.py --view top --map Town13  # 俯视看街区/NPC

# 6. 2D A/B 评估
python bin/eval_2d_ab.py --root-a outputs/kitti_ab_day_clear --root-b outputs/kitti_ab_sunset_glare

# 7. 失效归因(逐帧匹配 → 距离/框高/TTC 分箱 + 漏检画像;速度用于 TTC 归一化)
python bin/eval_attr.py \
  --run day4=outputs/kitti_sweep_day_clear_4:4.0 \
  --run day8=outputs/kitti_sweep_day_clear_8:8.0 \
  --run rain=outputs/kitti_ab_rain_night:8.0 --json outputs/attr.json

# 8. 3D LiDAR 检测(AutoLabel autolabel env;cwd 必须在 AutoLabel 根)
cd /root/autodl-tmp/Documents/Projects/AutoLabel && KITTI_OBJECT_ROOT=<abs kitti root> \
  /root/miniconda3/envs/autolabel/bin/auto3dlabel run 000000-000069 "检测汽车" \
  --det-model pointpillars_kitti --batch --no-viz --out-dir <abs out>
python bin/eval_kitti.py --root outputs/kitti_ab_x --pred outputs/kitti3d_ab_x

# 9. 测试
python -m pytest tests/ -q
```

## 环境(勿新建;direnv 进入目录自动激活 autodrivedata)

| 环境 | Python | 用途 |
|---|---|---|
| **autodrivedata**(本项目) | 3.11.16 | pycarla + ultralytics;采集、2D 评估、3D 比对、全部单测 |
| **autolabel** `/root/miniconda3/envs/autolabel` | 3.11.15 | mmdet3d;3D 检测、oracle 对比 |
| **base** | 3.10.8 | conda 底座 + direnv;pycarla/ultralytics 已迁出(2026-09-10) |
| **maptr**(未建,§5.11 C 阶段预留) | 3.8 | MapTR/MapQR 老栈 |

纪律:autodrivedata 包**不 import carla**(纯值,任何 env 可单测);依赖单向 AutoDriveData → AutoLabel,禁止反向。

## 项目结构

- `autodrivedata/` — 纯值库(geometry/calib/gt/static_gt/traffic_light/attribution/semantic/export/compare/scenarios),不 import carla
- `bin/` — carla 采集器(collect_drive/collect_ab_route/collect_static_gt/collect_tl_states/collect_nus)+ 评估(eval_2d_ab/eval_attr/eval_kitti)+ 可视化(view_stream)+ `carla_common.py` 共用件 + `carla_server.sh`
- `tests/` — 单测(autodrivedata env,213 passed / 3 skipped)
- `outputs/` — 采集产物(不进 git)

## 文档

- [Plan.md](Plan.md) — **单一事实源**:方案定案/执行记录/待办,改决策先读再改
- [docs/milestone.md](docs/milestone.md) — 里程碑时间线与验收结论速览
- [docs/testLog.md](docs/testLog.md) — 测试与踩坑日志(现象 → 修复 → 回归保护)
- [CLAUDE.md](CLAUDE.md) — 项目会话说明(A/B 实验纪律与红线)

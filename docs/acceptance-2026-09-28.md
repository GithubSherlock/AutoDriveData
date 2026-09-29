# 全能力面验收(2026-09-28)

> **本文档的定位**:一次**从零把整条流水线重新跑一遍**的验收记录 —— 不是新结论,是"已有结论**现在还成不成立**"的复跑对账。
> 复跑入口 = [`tools/showcase.py`](../tools/showcase.py)(四阶段:`baseline` / `offline` / `online` / `figures`)。
> 产物索引 = `outputs/showcase/`。**改代码后要重跑,先读 §6 的复跑须知。**
>
> **与 Plan2.md 的关系**:本文**不改任何已有结论**,只做三件事 ——
> ① 复跑并**逐数字对账**;② 记录复跑中**新发现的问题**(§4);③ 把"结论 → 证据 → 复跑命令"三者接成一条链。
> 各能力面的**方法论**仍在 Plan2.md / README.md,本文不重复。

---

## 0 结论扫描

**底盘稳**:`ruff` 全绿;全量 pytest **1076 passed + 6 skipped + 0 failed**(172 s,基线 1071 —— 差额 = 本轮新增的 5 个回归钉,见 §4)。

| 能力面 | 关键数字 | 归档 | 今日实测 | 判定 |
|---|---|---|---|---|
| P1 逆光相机的 2D | ΔmAP | −0.014 | **−0.0139** | ✅ 逐位 |
| P1 雨夜相机的 2D | ΔmAP | −0.153 | **−0.1531** | ✅ 逐位 |
| P1 浓雾相机的 2D | ΔmAP | −0.013 | **−0.0132** | ✅ 逐位 |
| P1 检出率(雨夜) | 检出/GT | 0.72→0.48 | **0.72→0.48** | ✅ 逐位 |
| P1 失效归因 | 尺度断崖 | <32px 0.15–0.47 | **16–32px 0.27/0.15/0.47** vs 32–64px **0.93/0.78/1.00** | ✅ |
| P1 逐帧速度自证 | 定速 | 6.60 m/s(非 8.0) | **6.60** | ✅ |
| LiDAR 3D(天气不敏感) | ΔAP | +0.003 / +0.017 / 0.000 | **−0.0002 / +0.0241 / 0.000** | ⚠️ 方向一致,数字有差 → §4.6 |
| MapTR 帧级留出 | mAP @0.2 | 0.3043 | **0.3048** | ⚠️ ±5e-4 → §4.4 |
| MapTR 路线级留出 | mAP @0.2 | 0.1114 | **0.1112** | ⚠️ 同上 |
| MapTR 训练集自身 | mAP @0.2 | 0.2727 | **0.2726** | ✅ |
| 地图矢量 A6 对账 | 3D 位置误差 | <5 cm | **max 0.002 cm / 184 抽样 / 0 超差** | ✅ |
| 标定自证 | A0–A6 | 全 ✓ | **全 ✓**(corner 约定复核通过) | ✅ |
| nuScenes 十条判据(默认图) | 两代 rig 全过 | 全过 | **nuscenes 7/7 live + wide 7/7 live** | ✅ |
| 六路 ego 像素 | 0 px | 0 | **0 px × 6 路 × 两代 rig** | ✅ |
| SLAM 前端 ATE | m | 1.4127 | **1.4126** | ✅ 逐位 |
| SLAM PGO 后 ATE | m | 0.1352 | **0.1352** | ✅ 逐位 |
| SLAM 闭合误差 | pre→post | 12.386→10.808(对 GT 10.994) | **12.386→10.808** | ✅ 逐位 |
| SLAM 回环 | 条数 | 114 / 196 候选 | **114 / 196** | ✅ 逐位 |
| Python vs C++ ICP | 对拍 | 9/9 PASS | **9/9 PASS** | ✅ |
| 聚类 | 簇/帧 | 117.97 | **117.97** | ✅ 逐位 |
| 单目测距(project) | 147 框 / mean | 8.56% / median 7.8% / z10_20 8.47% | **8.56% / 7.8% / 8.47%** | ✅ 逐位(需 `--max-frames 70`) |
| 单目测距(yolo) | 命中 / mean | 131/147 / 10.53% | **131/147 / 10.54%** | ✅ |
| 3DGS | psnr_all | 17.9 | **17.07**(1500 iters,≠ 归档口径) | ⚠️ 见 §3.6 |
| 在线 SLAM(studio) | 滞后/丢帧 | eff 0.90–1.00 同步 | **滞后 0 帧 / 丢 0 / 止损 0** | ✅ |

---

## 1 环境与口径(先说清楚"这不是同一台机器")

| 项 | 今日 | 备注 |
|---|---|---|
| GPU | **RTX 3090 / 48 GB / sm_86** | 归档日志里是 **RTX 4080 SUPER / sm_89**(另一台机) |
| CUDA / driver | 13.0 / 580.82.09 | |
| python | 3.11.16 @ conda `autodrivedata` | |
| git | `8c0a1ff`(**dirty: 9 files**) | 本轮改动见 §4 |
| CARLA | 0.9.16,headless,默认图 **Town10HD_Opt** | 记住:任何"标定/渲染"判据都**先确认服务器在哪张图**(§4.5) |

> ⚠️ **跨机型口径**:README/Plan2 里的 AP 数字采于 RTX 4080 SUPER。本轮复跑在 3090 上,
> 且**没有重训任何模型**(用的还是归档权重)。所以本文的 AP 对账是"**同权重、跨机型**"的重算 ——
> 见 §4.4:这正是发现"AP 不是逐位可复现"的地方。

---

## 2 阶段 P0 · 基线

```bash
ruff check && ruff format --check     # All checks passed! / 212 files already formatted
python -m pytest -q                   # 1076 passed, 6 skipped in 172.07s
python -m pytest --collect-only -q    # 1079 tests collected(差额 3 = 模块级 importorskip)
```

pytest 基线已从 1071 更新到 **1076**(CLAUDE.md 那一行同步改了 —— 基线数只写一处)。

---

## 3 阶段 P1 离线 / P2 在线,逐面

### 3.1 calib —— 标定自证与 rig 图

| 复跑项 | 结果 |
|---|---|
| `viz_rig_check --offline` ×2 | 覆盖 **100.00%**(nuscenes,盲区 0°)/ **95.79%**(wide,盲区 15.156°) |
| `viz_rig_check --live` ×2 | 六路 **ego px = 0 / 0.0000%**;views_*.png 落盘 |
| `probe_calib` | **A0–A6 全 ✓**;corner 约定复核:`cx_corner=620.5`(标称 620.5),采样约定裁决 corner 0.0003 m vs center 0.023 m |
| `viz_calib_check` | check_{geometry,raw,overlay,rig_ab}.png;逐相机实挂 yaw 与看见锥数 |
| `calib_multilidar` | small(0.1rad/0.1m)**converged**(overlap 0.991/iter 5);large(1.2rad)**not_converged**(iter 9) |
| `viz_layout_cmp` | **本轮修复后**才跑通,见 §4.2;数值:official 后相机覆盖更多(BL 45→56 / FL 31→45 / FR 45→30) |
| `verify_nus_calib --offline` | 判据 ③④⑤ 全 ✓ |

### 3.2 P1 corner case —— A/B 矩阵

三组 A/B 全部**逐位复现**(见 §0 表)。归因链也复现:框高分箱 **16–32px 一组 0.27/0.15/0.47**,
**32–64px 一组 0.93/0.78/1.00** —— 尺度断崖在,漏检框内亮度与命中几乎相同
(day8: 33.4 vs 37.5;fog: 79.0 vs 94.5 —— **漏检的反而略亮**,再次否定"暗导致漏")。

逐帧速度自证仍是 **6.60 m/s**(标称 8.0)—— 这是**归档数据集采集时**的既有事实(修复在采集器侧,
不追溯旧数据),A/B 两侧同速,**配对有效性不受影响**。

### 3.3 map —— MapTR 与地图矢量

| 复跑项 | 结果 |
|---|---|
| `eval_maptr` 帧级留出 + 阈值扫描 | @0.2 = **0.3048**;扫描 0.1/0.2/0.3/0.4 = 0.2155/0.3048/0.3803/0.4796;`pred/gt` 计数与归档**完全相同**(divider 1343/907) |
| `eval_maptr` 路线级留出 | **0.1112**;divider **1052**/871(归档 1053 —— **差 1 条预测**,见 §4.4) |
| `eval_maptr` 训练集自身 | **0.2726** |
| 逐帧契约落盘 | 300 帧 `mapvec_pred/1` → `outputs/showcase/03_map/contract/` |
| `viz_maptr_pred` | 6 帧 6 相机 + BEV 拼图,pred 品红 / GT 青绿,折线贴合车道几何 |
| `export_mapvec` 三格式 + **读回** | opendrive / lanelet2 / apollo 各 816 实例(799 折线);`--from` 三种读回各 816(**往返无损**) |
| `probe_mapvec_oracle` | 184 抽样,3D 位置误差 **mean 0.00 cm / max 0.002 cm**,超 5 cm **0 组** → **PASS** |
| `probe_mapvec_proj` | 帧 0:六相机在画比例 2.4%–51.0%,路面性 56.5%–100% |
| `convert_mapvec` | divider 72 / ped 7 / boundary 82 / centerline 163 |

### 3.4 slam —— 两段式闭环

802 帧闭环(`outputs/kitti_loop`)全链重跑:**前端 977.9 s → 后端 1127.9 s**(≈35 min 合计)。

- 前端:mean_rmse **0.17108**、overlap 0.7563、**n_failed 0 / n_nan 0**
- 后端:81 关键帧,SC 候选 **351** → 几何先验拒 **155** → 全量 ICP **196** → 接受 **114**
- 闭合:pre **12.386** → post **10.808**(对 GT 10.994)
- ATE:前端 **1.4126** → PGO **0.1352**
- **本轮新增的分解**(§4.8):前端 3D ATE 1.4126 里 **xy 只 0.273、z 达 1.386** ⇒ **96% 是垂直漂移**
- `slam_diff_test`:Python vs C++ 单次 ICP **9/9 PASS**(tol 0.001 rad / 0.01 m)
- `probe_scan_to_map` / `build_accum_map`(205 万点 PLY)/ `collect_slam --dry-run`(环 86 节点 / 269.4 m / 337 帧一圈 ≥ 250 门槛)均通过

### 3.5 perception —— 检测 / 单目 / 语义 / 点云

| 复跑项 | 结果 |
|---|---|
| `eval_2d_ab` ×3 | 见 §3.2 |
| `eval_attr` ×3 跑 | 尺度断崖 + 漏检画像 + TTC 箱 |
| `eval_kitti`(3D) | day_clear 基线 0.4729;sunset 0.4727 / rain 0.4970 / fog 0.4729 |
| `extract_ground` | 150 帧,均值地面占比 **57.22%** |
| `cluster_obstacles` | **117.97 簇/帧**(最大簇 18272 点)—— 与归档逐位一致 |
| `mono_distance` project / yolo | 94 可评框 mean 10.17% / 81 命中 mean 10.27% → §4.7 |
| `sem_bev` | **本轮修复后**才跑通,见 §4.1;6 帧 BEV(绿=可行驶 / 黄=车道线 / 品红=障碍物) |

### 3.6 gs / traj

- `train_3dgs_mini`:120k 高斯 / 1500 iters → **psnr_all_mean 17.07 / min 10.78**,产出 `gaussians_showcase1500.ply` + `render_compare_showcase1500.png`。
  ⚠️ **与归档 17.9 不可直接比**:归档是 `--scale` 调优后的口径,本轮用默认;此处只证"链路通"。
  **跑之前必须先修架构常量**,见 §4.9。
- `assemble_traj_pt`:Town13 → **31 场景**(9278 条 centerline);`convert_hivt_pt`(hivt env)**本轮未跑**。
- `collect_traj`:60 帧 × 3 agent(Town13)。

### 3.7 P2 静态 GT / 灯态动态 GT(在线真采)

| 采集器 | 产物 |
|---|---|
| `collect_static_gt --frames 10` | 逐帧 **信号 6** + **车道线段 2(14 点)**;落 `static_gt/{fid}.json` + **overlay 目检图** |
| `collect_tl_states --frames 30 --cycle 6,2,6` | 受控切灯:全图 15 个信号灯 actor,视距内 4 个呈 **Green**,落 image_2 + overlay + traffic_light 三层 |

### 3.8 在线采集器(全部真跑,短帧)

`smoke` / `collect_kitti`(10 帧,5 GT,11.67 万点) / `collect_drive`(20 帧,9 GT,15 车 + 6 行人) /
`collect_ab_route`(20 帧,**4/4 路肩车**锚定 pts[0]) / `collect_nus`(nuscenes + wide 两代 rig,6 相机 + LiDAR + 5 雷达) /
`collect_stereo`(10 帧,定速 **7.99 m/s** 逐帧自证) / `collect_surround_micro`(两代后相机布局) / `collect_traj` /
`probe_vulkan`(RTX 3090 + llvmpipe 枚举正常)—— **全部 exit 0,产物结构正确**。

### 3.9 live_studio —— 8 路实时 + 在线 SLAM + 实时 MapTR

`--npcs --speed 6 --duration 60 --maptr-ckpt ... --slam`:

- HTTP 首页列出 **8 路 stream**;实测 `curl` 首页 + 抓 `multipart/x-mixed-replace` 流(每帧 `Content-Length` 2.5 MB)成功
- 在线 SLAM:**滞后 0 帧 / 已处理 19 / 丢 0 / 止损 0**;地图 273215 点,rmse 0.159
- 实时 MapTR:pred 65–89 实例 / seg 169–231 段
- 视频 `studio_8view.mp4`(**4800×2220 × 20 帧**,标称 0.5 fps / 实测采集 0.32 fps ⇒ 1.6× 播放)
- worker 干净退出,服务器恢复异步

---

## 4 复跑中发现的问题

> 分三类:**已修**(代码已改 + 补了回归钉)、**待裁决**(要你定口径)、**口径注记**(不改代码,但引用数字时要带上)。

### 4.1 【已修】`sem_bev` 被一段**从未执行过**的死代码卡死(能力面长期不可用)

**症状**:整个语义 BEV 出不了图,抛 `ModuleNotFoundError: No module named 'utils'`。

**根因**:`yolopv2_predict` 里有一句**函数体内**的
`from utils.utils import non_max_suppression, split_for_trace_model` —— 那是 **YOLOPv2 官方仓库自带的包**,本机不存在。
而**唯一调用点写的是 `_, da, llm = ...`,那个返回值从来没被消费过**。

**为什么一直没被发现**(三层叠加,每层单独都不致命):

1. `runlog` **两次都如实记了**(2026-09-27 与 09-28),只是没人回头看日志;
2. `tests/` 里**没有任何用例覆盖本模块** —— pytest 全绿;
3. 这正是重构文档列的「**方法体内的惰性 import**(逃过 `--collect-only`)」那一类。

**修**:整段删除检测/NMS 路径(检测框本就走 YOLO11s-seg),`model(img)` 只取 seg/ll 两路输出。
**补钉**:[`tests/perception/test_sem_bev.py`](../autodrivedata/tests/perception/test_sem_bev.py) —— 桩模型跑通整条掩膜链,
并顺带钉住此前**零覆盖**的 letterbox→裁 padding→缩回原图几何(含一个反例对照)。

### 4.2 【已修】`viz_layout_cmp` 读图路径硬编码,**无视 `--a`/`--b`**

**症状**:按它**自己 docstring 教的用法**(重采到别处 → `--a/--b` 指过去)必然崩:
数值表照常打印(那几行用的是 `--a/--b`),一到拼图就
`FileNotFoundError: .../outputs/surround_micro_legacy/cam_back/000000.png`。

**根因**:读 infos/calib 用 `--a/--b`,读图却拼死 `<项目根>/outputs/surround_micro_{tag}` ——
与重构文档列的「**硬编码源码路径常量**」同类。

**修**:`root` 一路带到读图处。
**补钉**:[`tests/calib/test_viz_layout_cmp.py`](../autodrivedata/tests/calib/test_viz_layout_cmp.py) ——
两个 root 的图染成**纯红 / 纯蓝**,断言两张输出各自取自自己的 root(路径再写死必然同色)。
**先确认它对旧逻辑报红**(复现出同一条 `FileNotFoundError`)再改的代码。

### 4.3 【已修】权重路径二次断链 —— 且**专门防它的守卫漏判了**

**症状**:`sem_bev` 拉 `models/yolo11s-seg.pt` ⇒ 触发 ultralytics **静默联网重下** ⇒
本机 github SNI 不可达 ⇒ **进程悬挂无超时**,并在工作区留下一个 **1.9 MB 的截断权重**(1,982,464 B / 19.7 MB)。

**根因(两层)**:

1. **同一个坑的第二次复现**:`docs/refactor-2026-09.md §5.1` 记过它,要求"代码/文档/文件位置三处同口径"。
   但那次只让**代码与文档**同口径指向 `models/`,而**文件真身在 `weights/`**(20,669,228 B)—— **三个位置又错成了新的一致**。
2. **守卫白名单过期**:`TestWeightPaths._WEIGHT_ROOTS = ("", "models", "outputs/models")`,
   **没有 `weights`** ⇒ 守卫只在白名单内找同名文件,找不到就判"合法的未入库"而放行。**守卫只是假装在守。**

**修**(按你的裁决,以 `weights/` 为准):① `sem_bev` 指向 `weights/`;② 白名单补 `weights`;
③ 订正 `refactor-2026-09.md §5.1` 并记下教训。**顺序是先补白名单 → 确认守卫报红且指认准确 → 再改代码 → 复绿。**

### 4.4 【待裁决】MapTR 留出 AP **不是逐位可复现的** —— 可复现性下限 ≈ ±5e-4

三组留出**归档 vs 今日**的差都在 ±5e-4 内,**方向不一致**(两降一升),看着像噪声;但做了判据后可以排除"随机抖动":

| 判据 | 结果 |
|---|---|
| 同样配置连跑 4 次(含 `--match cpu` ×2 / `--match gpu` ×2) | **全部 0.3048**,完全相同 |
| 帧级留出逐类 AP | 只有 divider 动了(**0.3152→0.3167**);ped 0.2125 不变、boundary +0.0002、centerline +0.0001 |
| **帧级 `pred/gt` 计数** | **与归档完全相同**(divider 1343/907) |
| **路线级 `pred/gt` 计数** | divider **1052** vs 归档 **1053** —— **少 1 条预测** |
| 换推理设备(同权重、同数据、同后处理) | GPU `boundary` pred **1308** vs CPU **1307** —— **一条实例跨过阈值翻面** |

**结论(可下的)**:该 AP 的复现性下限 ≈ **±5e-4(帧级)/ ±2e-3(路线级)**,来源是**模型前向的浮点规约顺序**
(设备 / 进程 / cuDNN 算法选择)让**边界实例的 sigmoid 得分跨过 `--score-thr`**。
**不是**代码回归(计数一致、方向不一致、且与设备相关),也**不影响任何已有结论**
(阶段 3 决策点是"≥ 旧 0.0674"的闸门,量级差 4.5×,5e-4 动不了它)。

**修(用户裁决:补进红线)**:CLAUDE.md 红线段已加一条 ——
「**AP 的复现性下限 ≈ 2e-3**:同权重、同数据、同后处理,换推理设备或换进程也会让 AP 动。
⇒ **跨设备/跨机型的 AP 差 < 2e-3 一律视为不显著**,不许当"涨了/掉了"报;
判据 = 固定 `--score-thr` + 记录是否 GPU 推理 + **`pred/gt` 计数**(计数不等 = 预测真变了,
计数相同而 AP 变 = 阈值边界抖动)」。

### 4.5 【已修】`verify_nus_calib --live` 的判据 6/8 **是场景相关的**,会给出"标定坏了"的假警报

**实测**:

| 服务器所在图 | 判据 6(渲染 FoV) | 判据 8(相邻共视) | 其余 |
|---|---|---|---|
| **Town13**(被 `collect_traj --map Town13` 留下的) | ✗ `CAM_FRONT_RIGHT` dev **+0.10024°**(阈值 0.1°,**超 0.00024°**) | ✗ `FR↔BR` 锥体 `occluded_in_fov`(两路 px=0) | 全 ✓ |
| **Town10HD_Opt**(默认图) | ✓ FR dev **+0.00431°** | ✓ | 全 ✓ |

- **确定性**:Town13 上连跑两次,`dev=0.10024 / z_used=10.0 / n_used=7` **逐位相同** —— 不是抖动。
- 报告里**有 `map` 字段**,但**判据本身不校验它** —— 运维换图后跑一次,会得到"标定坏了"的错误结论。

**修(用户裁决:断言当前图)**:`--live` 在**跑判据之前**先断言图 = `CALIB_MAP`(`Town10HD_Opt`),
不等就直接退出(exit 1)并给出 stop/start 的处置;逃生口 `--any-map` 放行但报告里带
`map_warning` + `map_is_calibration_map: false`(那份报告**不可用于验收**)。
匹配走纯函数 `on_calibration_map`,**按路径分量而不是 `==`** —— 同一个图 CARLA 给两种写法
(`Carla/Maps/Town10HD_Opt` 与 `Carla/Maps/Town13/Town13`,都实测到)。
**回归钉**:`tests/calib/test_nuscenes_calib_consistency.py::TestCalibMapGuard`(4 条,
含"不许退化成子串匹配"与"断言必须在跑判据之前 + 必须是 raise 不是 warn")。

实测三条路径:错期望图 → **exit 1** 且给处置;正确图 → **exit 0 / 7 条全 ✓**;`--any-map` → exit 0 + 报告留痕。

### 4.6 【口径注记】LiDAR 3D ΔAP 与归档差 ≤0.007 —— 且归档数**在 Plan2 里无命令可溯**

今日:day_clear 基线 3D AP **0.4729**;sunset **0.4727**(Δ −0.0002)/ rain **0.4970**(Δ +0.0241)/ fog **0.4729**(Δ 0.000)。
README 记的是 **+0.003 / +0.017 / 0.000**。
**方向与定性结论完全一致**(天气对合成 LiDAR 无系统性影响,|Δ| ≤ 0.024 = 噪声级),但数字有 ≤0.007 的差,
且 README 那三个数**在 Plan2.md 里搜不到对应命令/日志**(`grep` 无命中)。建议:要么补上可溯源的命令,要么以后引用今日这组。

### 4.7 【已结案 · **本报告初版判错了**】`mono_distance` 的"147 框"是对的,错的是我

**初版结论(错)**:"归档 147 框、实测 94 框,疑为归档把 GT 框总数当成了可评框数。"
**实情**:我上一轮**没传 `--max-frames`**,用了它的默认值 **40**;而归档用的是文档里写的 **70**。

| 命令 | 框数 | mean | median | z10_20 |
|---|---|---|---|---|
| `--detector project --max-frames 70` | **147** | **8.56%** | **7.8%** | **8.47%** |
| `--detector project`(默认 40) | 94 | 10.17% | 10.59% | 10.72% |
| `--detector yolo --max-frames 70` | **131/147** | **10.54%** | 10.1% | 11.78% |

70 帧那一行与 Plan2 §P-D 的归档(**147 框 / 8.56% / 7.8% / 8.47% / 70% 框 <10%**)与生产口径
(**131 命中/147 GT / 10.53% / z10_20 36%**)**逐位一致** —— 归档完全正确,是我的复跑口径错了。

**残留的真问题(已修)**:`--max-frames` **默认 40 与文档口径 70 不一致**,不传参就会跑出**另一批框**;
更要命的是 **runlog 里不记帧数** ⇒ 这种差异**事后无从追溯**(本轮就是被它绊了一跤)。
**修**:`rl.highlight("max_frames", ...)` 已加 —— "detector 必须与误差数字一起留痕"的同一条理由,
帧窗同样必须留痕。

> **教训**:报告一条"归档对不上"之前,先按**文档里写的完整命令**再跑一遍。
> 本轮我先跑了默认值、再去`git log -S` 翻常量历史、还试了另外两个 root —— 唯独没做最该做的那件事:
> 把 `[--max-frames 70]` 抄全。

### 4.8 【口径注记 · 新增信息】SLAM 前端 ATE 的 **96% 是垂直漂移**

对齐后分解(3D Umeyama,与 `eval_slam` 同口径):

| | 3D ATE | xy | z |
|---|---|---|---|
| 前端 | **1.4126 m** | 0.273 m | **1.386 m** |
| PGO 后 | **0.1352 m** | 0.115 m | 0.071 m |

即"前端 1.41 m"这个数字**几乎全是 z 方向的累积漂移**;水平精度其实只有 0.27 m。
PGO 把 z 压到 0.071。**这条此前只被记成一个 3D 合计值**,复跑时才拆开(见 `outputs/showcase/04_slam/traj_compare.png` 右panel)。
**画这条曲线时踩过一次坑**:`traj_*.json` 的 `T` 是 **LiDAR 系**位姿,不先走 `slam_eval.lidar_pose_to_ego`
(手性共轭 `L·M·T·M·inv(L)`)的话两组轨迹**手性相反**,Umeyama 永远对不上(裸对齐 rmse 50 m),
画出来像是"定位全错"—— **复用既有实现,别自己推换算**。

### 4.9 【已修 · 环境】gsplat 的编译架构常量过期,3DGS 在内核里崩

**症状**:`RuntimeError: Failed to set maximum shared memory size (requested 7168 bytes), try lowering tile_size.`

**根因**:项目文档里的调用写死 `TORCH_CUDA_ARCH_LIST=8.9`(上一台 **RTX 4080 SUPER / sm_89**),
本机是 **RTX 3090 / sm_86** ⇒ JIT 缓存的 CUDA 内核目标架构不符。
改成 **`TORCH_CUDA_ARCH_LIST=8.6`** 后正常(120k 高斯 / 1500 iters 跑通)。

**修(用户裁决:按当前硬件配置来)**:`train_3dgs_mini` 在 `import gsplat` **之前**用
`torch.cuda.get_device_capability()` 自设 `TORCH_CUDA_ARCH_LIST`(`setdefault` ⇒ 外部显式指定仍优先),
文档里的 `8.9` 已删。实测:不设任何环境变量时模块自取 **8.6**,150 iters 短训练跑通。
⇒ **这类依赖实卡的常量一律不许写死在文档里**;引用归档数字前先看 `logs/*.json` 的 `env.gpu`。

### 4.10 【已修】`assemble_traj_pt` 的 docstring 用法列了个**不存在的参数**

docstring 写 `--map-json training/map/Town13_full.json --map Town13`,
但 `argparse` 里**只有 `--map`**(xodr 由地图名解析)。照 docstring 抄必报
`unrecognized arguments: --map-json`。
**修**:docstring 删掉 `--map-json`,并加一句"**没有这个参数**"的显式说明(防止有人再照旧版加回来)。

### 4.11 【口径注记 · 回答一个会被反复问的问题】`legacy` rig **不是**被抛弃的旧设置

**问法**:"showcase 的 calib 里出现了 `legacy_*`,之后的项目不该用这个标定 —— 它被抛弃了吗?"

**答**:`legacy` **是刻意保留的、喂旧权重的兼容口径**,不是项目当前标定。三条事实:

| 事实 | 出处 |
|---|---|
| rig 表里写明 "`nuscenes`(**当前**)" vs "`legacy`(**早期**)" | `sim/live_common.py:10-15` |
| **"它**不是无条件 bug**:`maptr_ep512.pt` 就是在这套 rig 上训出来的,拿 nuscenes 喂它**反而是错配**。故本文件同时保留两套口径" | `sim/live_common.py:23-26` |
| 选法是 `resolve_rig(choice, ckpt)`:`auto` 时**按权重文件名**判 —— `maptr_ep256/512` → legacy,其余 → nuscenes | `sim/live_common.py:92,144-155` |

**现役权重 `maptr_v2_singleF.pt` 走 `auto` 得到的是 `nuscenes`** —— 管线没有在用 legacy。

**showcase 里那两张 `legacy_*` / `official_*`** 来自 `viz_layout_cmp`,是**两代布局对照**(同一段路、同一份地图 GT,
A = 旧布局 / B = 官方布局,比逐相机覆盖量),**不是**"用 legacy 标定出的管线产物"。这一项是本轮自选的展示图,
与 `live_studio` 的 `--rig` 选择无关。**若认为它不该出现在验收图集里,删掉 `showcase.py` 的 `viz_layout_cmp` 那一条即可**
(采集侧 `collect_surround_micro --cam-back legacy` 也只服务于这个对照)。

> **教训(展示层的)**:把"两代对照"混进"能力面代表图"里,会让人以为被对照的那一代还在生产链上。
> 展示项的**选取**也需要判据,不能只图信息量大。

**★ 2026-09-28 用户裁决:legacy 口径从代码中完全移除**(范围 = 运行时选项 + 诊断 + 证据归档)。
证据冻结在 [docs/legacy-rig-archive.md](legacy-rig-archive.md) + `assets/legacy-rig/`(4 张图,3.7 MB)。

| 动作 | 内容 |
|---|---|
| 证据归档 | 新建 `docs/legacy-rig-archive.md`(镜像成因 / 逐相机方位数字 / 覆盖表 / 4 张图)+ `assets/legacy-rig/` |
| 运行时选项删 | `RIG_LEGACY` / `LEGACY_CAM_YAW` / `LEGACY_CKPTS` / `LEGACY_FRAME` / `LEGACY_FOV` / `resolve_rig` / `--rig auto`;`rig_spec` 未知 rig 改为**直接抛**(不再静默回退) |
| 诊断删 | `calib/probe_rig_mount.py`(**整文件删**)、`viz_calib_check` 的 `check_rig_ab.png` 与 `sheet_geometry` 的历史对照、`collect_surround_micro --cam-back legacy` |
| 标签改名 | `viz_layout_cmp` 的产物标签 `legacy/official` → **`a`/`b`**(它就是用户最初看到并起疑的那张图) |
| 测试 | 删 5 条 pin legacy 的用例;`_PRODUCERS` 移除 `viz_calib_check`(它不再是六格产出者) |

⚠️ **移除的代价(写进归档,别忘)**:镜像 bug 的**直接对照**以后无法在仓内重跑 ——
同一失效模式改由 `verify_nus_calib` 十条 / `probe_calib` A0–A6 / `viz_rig_check` 的车体像素判据覆盖
(它们本来就不依赖 legacy 口径存在)。

### 4.12 【口径注记】`--bev-chunk` 之外:本轮无新增显存/性能结论

`--bev-chunk` 的老结论(分块不降反升)未复跑;在线 SLAM 的同步/异步口径本轮只验证了**同步路径正常**
(滞后 0 / 丢 0 / 止损 0),**未做 `--slam-async` 对照**。

---

## 5 产物索引(`outputs/showcase/`)

| 目录 | 内容 |
|---|---|
| `01_calib/` | `rig_layout_{nuscenes,wide}.png`、`views_*.png`、`report_live_*.json`、`legacy/official_frame000000.png`(布局对照) |
| `02_p1_corner/` | `ab_side_by_side_f000030.png`(四天气同帧对照 + GT 框)、`attr.json` |
| `03_map/` | `frame_holdout.png` / `route_holdout.png` / `train_self.png`(BEV 目检)、`viz_pred/frame_025*.png`(6 相机 + BEV)、`contract/`(300 帧逐帧契约) |
| `04_slam/` | `traj_compare.png`(轨迹 + 误差分解)、`eval_pre.json` / `eval_post.json`、`run/`(traj_raw / traj_pgo / loops / slam_summary) |
| `05_perception/` | `sem_bev/bev_*.png` + `panel_*.png` |
| `08_traj/` | `hivt_scene0.png`、`carla_traj/`(60 帧)、`processed/`(31 场景) |
| `09_online/` | 各采集器 root、`studio/studio_8view.mp4`(**视频**)、`studio/web_grid_frame{0,1,2}.jpg`(**网页流实抓帧**)、`studio/frame.png_*.png`(八路 overlay + raw 差集) |
| `logs/` | 各入口的 stdout/stderr 全文 |

---

## 6 复跑须知

```bash
python tools/showcase.py --phase baseline     # ruff + pytest(不需 CARLA)
python tools/showcase.py --phase offline      # 复用现有 outputs
bash tools/carla_server.sh start              # online / online 段必须先启(且**默认图**)
python tools/showcase.py --phase online
python tools/showcase.py --phase figures      # 只出图,不需要跑任何入口
python tools/showcase.py --list               # 只列命令
```

**四条坑(都踩过)**:

1. **`online` 会改写 CARLA 所在图**(`collect_traj --map Town13` 会把它留在 Town13)。
   跑完必须 `carla_server.sh stop && start` 回到默认图,否则 §4.5 的假警报会找上门。
2. **别用 `| tail` 包入口命令** —— 管道会把真实退出码吃成 `tail` 的 0。
   本轮我自己的临时命令就吃过一次(`train_3dgs_mini` 明明崩了,后台却报 exit 0)。`showcase.py` 内部走 `subprocess` 取真实码。
3. **3DGS 先确认 `TORCH_CUDA_ARCH_LIST` 与本机 arch 一致**(§4.9)。
4. **长任务中途看进度靠 `logs/*.jsonl`**,不要看 stdout(§P-L 的老结论)。

---

## 附:改动清单(**未提交**)

> 分两批:**验收轮**(发现问题 + 补回归钉)与**裁决轮**(用户就 §4.4–§4.7、§4.9、§4.10 拍板后的实施)。
> 两批在设计上**互相独立**,可以分别 review。

### 验收轮

| 文件 | 改动 |
|---|---|
| `autodrivedata/perception/sem_bev.py` | 删死代码惰性 import(§4.1);改成 `weights/` 权重路径(§4.3) |
| `autodrivedata/calib/viz_layout_cmp.py` | `--a/--b` 真正决定读图路径(§4.2) |
| `autodrivedata/tests/utils/test_paths.py` | `_WEIGHT_ROOTS` 补 `weights` + docstring 记第二次复现(§4.3) |
| `autodrivedata/tests/perception/test_sem_bev.py` | **新增**(§4.1) |
| `autodrivedata/tests/calib/test_viz_layout_cmp.py` | **新增**(§4.2) |
| `docs/refactor-2026-09.md` | §5.1 订正为文件真身位置 + 记第二次复现与教训 |
| `docs/acceptance-2026-09-28.md` | **新增**(本文) |
| `tools/showcase.py` | **新增**(验收编排器;fileTree §4 为它明写了一条例外条款) |
| `docs/fileTree.md` | §4 tools/ 例外条款 + `showcase.py` 行;§5 本文行;§3 两个新用例行 |
| `CLAUDE.md` | pytest 基线 1071 → **1076**(那一行是**唯一**记基线数的地方) |

### 裁决轮(2026-09-28 用户四条指示)

| 文件 | 改动 |
|---|---|
| `CLAUDE.md` | **红线新增一条**:AP 复现性下限 ≈ 2e-3,跨设备/跨机型差 <2e-3 视为不显著(§4.4) |
| `autodrivedata/calib/verify_nus_calib.py` | `--live` **断言当前图**(`CALIB_MAP` + 纯函数 `on_calibration_map`)+ `--expect-map` / `--any-map`;报告加 `calibration_map` / `map_is_calibration_map` / `map_warning`(§4.5) |
| `autodrivedata/tests/calib/test_nuscenes_calib_consistency.py` | 新增 `TestCalibMapGuard` 4 条(§4.5) |
| `autodrivedata/perception/mono_distance.py` | runlog 补 `max_frames`(§4.7 的残留真问题) |
| `autodrivedata/gs/train_3dgs_mini.py` | **按实卡自取** `TORCH_CUDA_ARCH_LIST`,文档里的 `8.9` 删除(§4.9) |
| `autodrivedata/traj/assemble_traj_pt.py` | docstring 删掉不存在的 `--map-json`(§4.10) |
| `tools/carla_server.sh` | `probe_vulkan` 的判据**不再写卡型号**(原写"应列出 RTX 3080 Ti" —— 那是活判据不是历史记录,换卡即失效)。判据改为"有 NVIDIA 设备、不是只剩 llvmpipe" |

### 追加轮(2026-09-28 用户两条追加指示)

| 文件 | 改动 |
|---|---|
| `autodrivedata/calib/camera_rig.py` | **新增 `CAMERA_GRID_ROWS`** = 六视角画布行序(2 行×3 列,按方位绕车)。放在这里是因为本模块已是"环视相机 rig 唯一来源",且**纯值**(离线脚本也能引用) |
| `autodrivedata/sim/live_studio.py` | `GRID_ROWS` 的前两行改为引自 `CAMERA_GRID_ROWS`(**值不变**,只是不再就地写字面量) |
| `autodrivedata/calib/camera_rig.py` | 另加**取序入口** `camera_grid_rows()` / `camera_grid_order()`(展平版),产出点不再各写一遍过滤/展平 |
| `autodrivedata/calib/viz_rig_check.py` | 同上;并**订正第二行**:原先 `BACK_LEFT, BACK, BACK_RIGHT` 左右反了 |
| `autodrivedata/calib/viz_layout_cmp.py` | 由 `sorted()` **字母序 + 一行六列**改为 2×3 + 同一行序;顺带给每格加相机名标签(2×3 之后靠"数第几格"读不出来) |
| `autodrivedata/map/viz_maptr_pred.py` | `sorted(ds.cam_names)` → `camera_grid_order(...)`。**字母序把后三路排到了第一行**、前一行挤到第二行,且每行内部左右也反 |
| `autodrivedata/sim/view_stream.py` | `--view grid6` 的列序原来自 `rig_spec` 字典序(`F, FL, FR, B, BL, BR`)→ 改用取序入口 |
| `autodrivedata/calib/viz_calib_check.py` | `check_rig_ab.png` 原按 `for name in NUS_CAMERA_RIG` 每 3 个硬切 → 改用取序入口 |
| `autodrivedata/calib/probe_calib.py` | `overlay.png` 列序改用取序入口;**形状由 2 列×3 行改为 3 列×2 行** |
| `tests/calib/test_rigviz.py` | 新增 `TestCameraGridRows` 3 条:①行优先必须是**绕车的方位扫描**(间隔 ≤120°);②七个产出者**必须真 import 单一定义**(AST 查 import,不是文本匹配);③呈现层禁 `sorted(相机集合)` |
| `tests/calib/test_viz_layout_cmp.py` | 新增 `TestTileGrid`:六路染**互不相同的纯色**,逐格采样断言 (r,c) 格 = 该行第 c 路 —— 顺带把画布形状钉成 2×3 |

**这条的性质**:不是"新约定",而是**把各处绘制对齐到已有的用户口径** —— `live_studio.GRID_ROWS` 本来就是
`左前/前/右前 + 右后/后/左后`,连注释都写着同一理由("不沿用字典序,那样第二行会变成左后/右后与地理直觉相反")。
真正的问题是它被**抄成了七份,其中四处是错的**。现在七处同源。

**七处产出点全部逐格复核**(不靠目检:裁格内标签到原生像素,或按图内容与源图做匹配):

| 产出点 | 形状 | 验证 |
|---|---|---|
| `sim/live_studio.py`(定义源头) | 3×2 + 第三层 | 既有测试 pin |
| `calib/viz_rig_check.py` `views_{rig}.png` | 3×2 + 方位尺 | 裁标签(两代 rig 各跑一遍) |
| `calib/viz_layout_cmp.py` | 3×2 | 逐格纯色采样 + 裁标签 |
| `map/viz_maptr_pred.py` | 3×2 + BEV 面板 | 逐格与源图 MAE 匹配:命中 1.2–4.3 vs 次近 46–59(≈15× 分离) |
| `calib/viz_calib_check.py` `check_rig_ab.png` | 3×2 | 裁标签 |
| `calib/probe_calib.py` `overlay.png` | 3×2 | 裁标签 |
| `sim/view_stream.py --view grid6` | 3×2 | 裁标签 |

> ⚠️ **两个判据都踩过,值得记** —— 都是"我以为它抓得住,实测抓不住":
>
> 1. **恒真的断言**:初版把"行优先必须是**严格递减**的方位序列"写成断言 —— 那是**恒真**的
>    ("减到 ≤ 前一个为止"这个解缠 `while` 对**任何**输入都产出严格递减序列),把第二行左右对调
>    它**照样绿**(实测确认)。真正有区分度的是**相邻间隔**:正确顺序实测最大 **71.3°**,对调后跳到
>    **195.0°/288.7°**、字母序 **248.9°** —— 取 120° 作界,三方都离得很远。
> 2. **文本匹配被注释骗过**:"产出者必须引用单一定义"初版是 `tok in 源码` 的裸字符串查,把 import
>    与调用一起删掉、只在注释里留一句 `camera_grid_order`,它**照样绿**。改成 **AST 查真实 import** 才抓住。
>
> **"我加的断言能抓住目标缺陷"必须实测,不能假定。** 两条钉现在都有实测红证(sort 版报
> `map/viz_maptr_pred.py:91 sorted(ds.cam_names)`;删 import 版报 `已 import 到的:['argparse','json','time','torch']`)。
>
> ⚠️ **一条不该改的地方**(查过才没动):`map/maptr/dataset.py` 的 `self.cam_names = sorted(...)` 看着像同一个
> bug,但它是**数据序不是画布序** —— 那个 dict 顺序喂给 GKT,只在"两台相机深度恰好相等"的并列 BEV 像素上
> 决定谁胜(GKT 用 `d < best_d` 严格取先到者);改它等于动那批像素的裁决。画布序该在**呈现点**换,
> 这条已写进回归钉的判据边界里。
| `docs/acceptance-2026-09-28.md` | §4.4/§4.5/§4.7/§4.9/§4.10 更新;§4.7 含**对本文初版的订正** |

> **未提交任何东西**(项目纪律:不自动提交)。
> **本文初版有一条结论是错的**(§4.7:"归档 147 框对不上")—— 已订正,并保留教训,不作无声修改。

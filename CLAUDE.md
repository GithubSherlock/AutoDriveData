# AutoDriveData

CARLA 0.9.16 → AutoLabel 自动驾驶数据输出流水线:自定义地图/场景采集车检测动态/静态目标与道路特征,构建 0→1 数据输出。当前主线 = **Corner Case 场景矩阵定量验证**(P1)+ 静态 GT(P2);raw 输出供 AutoLabel 验证场 + 长尾数据源。

## 开发前提

- 绝对不要创建子代理。无明确用户指示或确认，严禁使用任何子代理或委派任务！但可以启动 workflow

## 当前状态

**目录结构(2026-09-26 重构完成)**:`bin/` 与 `tests/` 顶层目录**已删除**,全部收进 [`autodrivedata/`](autodrivedata/) 主包,
按**能力面**分 10 个目录,包根只剩 `__init__.py`。**可执行入口一律 `python -m autodrivedata.<能力>.<模块>`**。

| 主题 | 结论 | 详述 |
|---|---|---|
| **P1** corner case 矩阵 | ⚠️ **下面这组数是 `yolo` 后端口径**(2026-10-02 换默认后端后重跑过,见下一行)。**Epic 口径 + 修退化 GT(2026-10-01)**:四型可量化掉点 —— 浓雾 **−0.578** > 雨夜 −0.487 > 湿路面 −0.215 > 逆光 −0.125(day_clear **0.669**)。⚠️ 上一版(Epic 但含退化 GT,GT=194)是 −0.500 / −0.409 / −0.227 / −0.047(0.591),**已作废**:那 11 条零面积 GT 永远配不上,去掉后**基线涨 0.078 而 B 侧(检出稀的)一条没动** ⇒ 所有 Δ 同步变大。重新看再上一版的 Low 档数字**也作废** —— Low 档**根本不渲染雾**(**非相机模态两侧对称不动**:LiDAR 点数差 <0.002%、雷达 <0.3%、分布逐项相同 —— **平台边界**:CARLA 不给雨雾建模消光,这条不受 GT 口径影响) | [Plan4.md](Plan4.md) §P-V4 / §P-V12 |
| **★ P1 的 SAM3 口径读数**(2026-10-03 定稿,§P-V24) | 读数已重设:**六列 = `AP | GT | 检出 | 命中 | 召回 | 检出/GT`**(`ap_for` 多返回一个 `n_tp`,就是那次贪心匹配的 `matched` 计数)。⚠️ 起因是**修去重/阈值之前**四个 Δ 全落进 ±0.02,一度读成「SAM3 抗退化」—— 实际是**过检 2.36×GT 把 recall 撑住了**,AP 的尾部纪律无从发力。修好后(conf 0.5 + 去重)矩阵恢复分辨力:**浓雾 ΔAP −0.075 / Δ召回 −0.131**(day_clear 召回 0.962 → 浓雾 0.831);雨夜 −0.015/−0.033;湿路面 −0.014/−0.011;**逆光 +0.008/+0.005(不退化)**。yolo 同口径:浓雾召回 **0.705→0.071**(掉 63 个百分点)。★ 三条:① **排序跨后端一致**(浓雾最重),**量级不可比**;② 「逆光是否退化」两个后端给出**相反**答案;③ 这批合成退化上**通用基础模型比 KITTI 微调的闭集检测器稳健得多** —— 但那只是**这两个模型**的比较,不是普适律(yolo 是在**无雾的 KITTI** 上微调的,域外本就是它的弱项)。⚠️ 报数必须带阈值 | Plan4 §P-V23 / **§P-V24** |
| **P2** 静态 GT | 信号/标志是 landmark、车道线是 lane_marking;**semantic LiDAR 打不到** ⇒ 只能走地图查询 API | Plan.md §5.7c |
| **P2 判据** | **已补**(§P-V21):`collect_static_gt --sem` 出语义 oracle,`perception/static_eval` 判据(离线)。★ **形状是量出来的**:想当然的"锚点像素是不是 TrafficLight" **40/40 全红** —— landmark 是**地面**锚点,灯头在它上方 5 m;车道线单像素只有 21.7%/27.8%(线只有 ~2 px 宽)。改成「锚定 + 邻域」后首跑:**信号 1.000**(横移 0.103 / 随机 0.017)、**车道线 0.930 / 0.997**(横移 **0.000**)。⚠️ 邻域是**定标**的(4 px / 6 m),放宽到 8 m **判别力反而塌**(横移对照爬到 0.30)。⚠️ 裁决必须有第二条 `real ≥ 5×对照`,否则"这场景到处都是标线"会被读成"xodr 对得上"。⚠️ `n=0` 的组是**未判**(第三种裁决),不是通过也不是不通过 | Plan4 §P-V21 |
| **P2-B** 静态道具 GT | **能出,而且不难**。道具走**独立通道**(`collect_static_gt --props`)—— **绝不许塞进 `label_2`**,进去会破 P1 A/B 帧级配对硬门槛。尺寸**只认 yaw=0 探针**(转过之后 `bounding_box` 给的是**被剪切**的值:锥桶 yaw=30° 读 `0.1246×0.4704` vs 真值 `0.3441×0.3441`,3.4% 的物体像素落到投影框外)。判据 `perception/prop_eval.py` **离线**跑(投影框 vs **渲染轮廓**):可判 84 条 coverage **逐条 1.0000**、类别命中 1.0000。**地图自带**的道具出不了实例(关卡网格,`get_actors()` 里 0 个),只能出类掩膜 `sem_tags.PROP_TAGS` | Plan4 §P-V14 |
| **P2-A** 像素级分割 GT | 已通:`collect_surround --sem` 落 CARLA 语义 tag(**8 位灰度,tag 即像素值**;tag 编在 **R 通道**、编号 = `CityObjectLabel`),`perception/sem_eval.py` 出**图像 mIoU 0.514 / BEV 0.572** 双口径。★ **BEV 障碍物通道的「结构性恒空」已修**(§P-V26):旧路径对**所有类**一律与**地平面**求交,对**离地 1–2 m** 的物体把射线送过头顶(落点中位 **120 m**、±30 m 内 **0.0%**)。修法 = `collect_surround --depth`(uint16 毫米)+ `sem_bev.mask_to_bev_depth` **只给 obstacle 走**;障碍物 BEV IoU **0.688**(三类里最高)。★ 三条对照全过:接地类两条路 IoU 0.574 / 换错深度后 0.000(**真的在用**)/ 障碍物离地 **1.28 m**(地平面假设会给 0) | Plan4 §P-V13 / **§P-V26** |
| **实例分割** | 已通:`collect_surround --inst` 落 **16 位灰度实例 id 图**(像素值 = CARLA actor id;R 通道同时带语义类 ⇒ 一台相机给「有几个」+「各是什么」),`perception/inst_eval.py` 出 **PQ 0.6257 / SQ 0.8447 / RQ 0.7407** + mask AP@0.5 0.6341。⚠️ 两条实测口径:CARLA 给**每个关卡网格**都发 id(`Roads` 113 个 vs `Car` 13 个)⇒ 实例评测**只取可数物体**;⚠️ **PQ 与 mask AP 的差别不是「多一个语义项」**(两个都要求类匹配)——AP 对掩膜质量不敏感,而 PQ 里 IoU 1.00 与 0.55 差得很远。⚠️ **那组读数的模型是「借来的」** —— `inst_eval.load_instance_predictor` 跑的是现成的 `weights/yolo11s-seg.pt`(**通用 COCO 权重,不是本项目训练的**)。通的是「实例 GT + 判据」这条**评测链**,不是一个训练好的分割模型;引用 PQ 时必须写明这一点 | Plan4 §P-V15 |
| **三传感器融合** | **后融合,结论是「有条件的」**(§P-V18/.19):**在 P1 静置车数据上它没赢** —— 严门单独 AP `0.0462` > 宽门+相机 `0.0202`(**一个更严的尺寸门就能替代相机**)。补上缺的两点(**多类 + 近距**,`collect_ab_route --layout close`,**默认关**)之后**翻过来**:`0.0758 → 0.1742`,FP `65→12`(宽门那档 `1476→39`);`Cyclist` 的类**完全是相机给的**(LiDAR 分不出自行车)。⇒ **价值有前提:目标多类、且在 LiDAR 打得动的距离(≲25 m)**。⚠️ **雷达不进消融表**(§P-V17 六之二):可复现实测 —— **可归因回波只出现在视轴上**(横向 0 槽位换什么蓝图都出:车 32318 / 行人 21019 / 挡板 6347),而 ±9.9°、±19.3° 的**一个点都不给**,而 P1 的目标正好在 ±3.5 m。**六个解释已排除**(离得近 / 射线预算 / FOV 实测 ±38.49° / 射线密度只低 35% / 假点云[65% 与 LiDAR 表面差 <0.3 m] / 简单复制[88/88 唯一]);**机制未知,不写成因**。⚠️ 两条边界:① 是「这套配置下、这批目标」的结论,不是「永远出不了目标」;② **例外存在** —— 目标摆在**视轴**上雷达就有回波。它另有一条**不需要目标**的判据(速度对表,**3/5 通道达标**,`radar_eval.py`)。⚠️ 未闭合:`Pedestrian` GT 框内 **0 个 LiDAR** 点(两个假设都证伪) | Plan4 §P-V18 / §P-V19。★ **簇框的尺寸先验**(§P-V27):泄漏已修(先验取自**别的 root** + 同源当场 SystemExit)、错先验对照是活的(TP→0),**但先验没帮上忙** —— AP 0.014→0.028 而 **TP 10→2**,可读的是后者(AP 在 712 条预测 vs 20 条 GT 的池子上算,是排序抖动) | Plan4 §P-V18 / **§P-V27** |
| 灯色动态 GT | 灯态 = **独立时序层**,Off/Unknown **不猜**;**不做视觉回归**(镜片 30m 处仅 ~4px) | Plan.md §5.9 |
| 失效归因 | **尺度主导**(<32px 0.15–0.47 vs ≥32px 0.78–1.00,断崖 ≈21–24px);CARLA **无运动模糊**(退化只能人工注入);天气只**前移断崖** | Plan.md §5.10 |
| MapTR 矢量管道 | 参考自实现打通。chamfer AP @`--score-thr 0.2`:**0.3043** 帧级留出 / **0.1114** 路线级留出 | [Plan2.md](Plan2.md) §P-M.12 |
| **训练早停** | 两段式(loss 平台作候选 → 留出 AP 复核),默认开;**理由必须分型**(converged / lr_exhausted);停时恢复 best-AP 权重(**且恢复必须是最后一次写 `--out`**,2026-09-30 修);三条不启用路径会打印原因 | Plan2.md §P-M.18/.20 |
| **时序融合 K=3** | 续训到 256 ep 后**帧级留出 +32%**(0.3043→0.4021)但**路线级留出 −0.0021** —— 两套留出相反,是 §P-M.12 红线的独立复现;**0.0021 落在复现性下限量级 ⇒ 真泛化上无可测收益** | Plan2.md §P-M.17 |
| **MapQR 移植** | 两个变体(散聚 query / 高度核 BEV 编码器)已实现、**默认关**;阶段 0 通过(单测 + 冒烟 + 显存实测);**消融已跑完四档**(§P-V11 第一批两档 + **§P-V25 八**补齐 ①②);官方权重 404 ⇒ 只能自训,绝对值不可比官方 | [Plan2.md](Plan2.md) §P-M.14 / **Plan4 §P-V25 八** |
| 8 路实时 studio | 拼图**每格原生像素不缩放**;在线 SLAM **默认同步执行**(worker 被 GIL 饿死,eff 0.04–0.24 vs 同步 0.90–1.00) | Plan2.md §P-L |
| 环视标定 | **像素约定 = CORNER**;ego 原点 = **后轴**(`NUS_EGO_ORIGIN_X = −1.2563`);ego 姿态 = **全 6DoF** | Plan2.md §P-M.10/.11 |
| 全传感器「声明 ≠ 渲染」 | 渲染位姿与声明位姿**由同一份常量导出**;十条判据两代 rig 全过 | Plan2.md §P-M.7 |
| **双传感器采集 + BEV 底图** | 一个 root 同落 6 相机 + LiDAR + 5 雷达 + GT 位姿(`--autopilot` 跑长序列,与 `--speed` 互斥);底图**只是视觉上下文**(模型纯相机,点云不进网络),`stitch_temporal --lidar-root` / `viz_maptr_pred --bev-pair --lidar-root` | Plan2.md §P-M.19 |
| **自适应 batch** | `--batch 0` 实测选批;**峰值不过原点** —— 必须扣掉 batch=1 的 ~13 GiB 常驻足迹,否则选出跑起来就 OOM 的档(MapQR 实测:旧公式 4/OOM,新公式 3/37.8 GiB) | Plan2.md §P-M.19 |
| **★ 四档消融·新版**(2026-10-05) | 四臂按**论文 LR 配方**重训(256 ep + cosine + warmup + AMP)后重评,**旧权重对照逐位复现**。★ 基线帧级 **0.2843 → 0.4248(+0.1405)**,这是这条线最大单杠杆 ⇒ 旧配方确实欠训。★ 但**① 的惩罚不降反升**(帧级 3.3σ → **5.7σ**)⇒ **结构性,不是收敛速度**。★ **② 首次落到基线之上**(路线级 +0.0104 / 跨图 **+0.0137 = 1.6σ**)。★★ **② 在三个测试集上全面压过 ③**(②=0.3939/0.1115/0.0968 vs ③=0.3726/0.1058/0.0760)⇒ **①(散聚 query)是净负面,③ 被它拖累**;旧读到的「③ 域内最高/跨图最低」**是欠训基线的产物,已作废**。⚠️ 新配方绝对数与旧表**不可混比**;报数必须带 `score_thr` | **Plan4 §P-V25 九** |
| **★ 图池 / 跨图零样本 / 四档消融**(2026-10-03,⚠️ 四档消融那半已被上一行取代) | 第二张图 = **Town05_Opt**(`surround_town05_epic`,与 v2_epic **同源**:逐帧 rig 逐字段相同、实例/帧 42.0 vs 42.4)。⚠️ **Town11/12/13/15 是平铺大图,本机相机采集必崩**(段错误,已排除 LiDAR/交通/质量档/点位)。**零样本跨图没崩塌**:Town05 400 帧 0.0775 vs 域内路线级 0.0814(seg9 的 0.1195 是**分段特殊**)。★★ **四档消融(2×2)**:①② 单独在三个测试集上 **6/6 全低于基线**(① 在域内帧级留出 −0.0647 = **3.3σ**);**③both 的排序在域内与跨图之间翻转**(域内最高但 +0.008 在带内、跨图最低且 −0.0197 = **2.2σ**)⇒ **MapQR 的收益不迁移,「域内不亏」≠「跨图不亏」**。⇒ **§P-V11 的「不可区分」是当时只看两臂 + 测试集小(100 帧 σ≈0.017)**,不是架构无差异。⚠️ **四档都欠训**(末 20 ep loss 都还在降)⇒ 差里混着收敛速度。机制:`ped_crossing` 占 ③ 差距 59%(跨图检出/GT **5.77×**)。**报数必须带 score_thr**(①②③ 的单调序只在 0.2/0.3 成立) | **Plan4 §P-V25** |
| 官方栈复线 | **已终止**(2026-09-14 用户裁决),**不要主动重提** | Plan.md §5.12 |
| **P1-6 静态遮挡** | 两条都做完:**wet_road 眩光 Δ−0.227**(§P-V4);`dense_rush` 改做成**静态遮挡** —— 只挡最近一台时 **Δ−0.096**(可见 21.6 px,正是断崖中心),四台全挡时 −0.667 但**与阳性对照 −0.652 不可分辨**。⚠️ `dense_rush` 的**车流** A/B 不可做(TM 违反红线,且 `collect_ab_route` 不读 `scene.traffic` ⇒ `--scene dense_rush` 会静默退化成 day_clear) | Plan4 §P-V12 |

> ⚠️ **三条不许误读**(MapTR 结果口径,详见 Plan2.md §P-M.12):
> ① **帧级留出 ≠ 泛化** —— 留出首帧与训练末帧**同街只隔 2.65–3.01 m**,真泛化看路线级(差 **2.7×**);
> ② 该 AP 口径是 **precision 均值、无 recall 项** —— 不能用来算"训练/留出差距";
> ③ 帧级留出还叠了 **GT 密度**混淆(11.34 vs 7.87 实例/帧)。
> **改动未提交 ≠ 待办** —— 引用历史结论前先 `git status`,别照着旧记录找不存在的未提交改动。

## 环境(勿新建;direnv 进入目录自动激活 autodrivedata,首次需 `direnv allow`)

| 环境 | Python | 用途 |
|---|---|---|
| **autodrivedata**(本项目) | 3.11.16 | pycarla + ultralytics + **transformers(SAM3,2026-10-02 装)** + **编辑线(2026-10-05 装:`omegaconf` `open_clip_torch` `einops` `pytorch_lightning` `hf_transfer`,装前 `--dry-run` 验过 torch/numpy 一个没动)**;采集 `python -m autodrivedata.sim.collect_*`、2D 评估 `perception.eval_2d_ab`、3D 比对 `perception.eval_kitti`、条件生成 `edit.conditioned_gen`、全部单测。⚠️ **装包只走清华源** —— aliyun 403、pypi.org 超时(实测);**HF 也下不动,走 `HF_ENDPOINT=https://hf-mirror.com`** |
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
├── sim/          24  CARLA 仿真交互层(唯一大面积 import carla 的地方)+ 全部采集器+ 闭环路线纯值
├── calib/        14  标定:原语(core.py = 原 calib.py)/ rig 表 / 自证探针 / 实时监看 / 配置图
├── map/          35  地图矢量 + MapTR(含 maptr/ = 原 maptr_impl/、maptr_official/ = 已终止线)
├── slam/         10  两段式激光 SLAM(core.py = 原 slam.py)+ 精度评估 + slam_cpp.cpp
├── perception/   19  检测 / 单双目 / 雷达 / 语义 / 点云(**不 import carla**);backends.py + sam3_backend.py = 默认后端
├── gt/            5  动态目标 + 静态目标 + 灯态时序层 + export/ 落盘
├── traj/  gs/     3  轨迹组装转换 / 3DGS 训练
├── runtime/       3  运行时设备/显存/批量超参 + **训练早停判据** + **jsonl→TensorBoard 转换**(跨能力面的 torch 工具;放 utils/ 不行,那层禁 torch)
├── edit/         11  图像/场景**编辑**:ControlNet 官方 repo 唯一落点(cldm_backend)+ 真值深度→条件图(depth_cond,纯值)+ 条件可控生成(conditioned_gen)+ KITTI 方裁(kitti_square)+ **下游 ΔAP 四臂判据**(downstream_eval)+ 人工注入(degrade/noise_curve)+ 视频时序(video_edit)+ 和谐化(harmonize)。**许 torch,禁 carla**
├── utils/         4  通用件:geometry.py paths.py fonts.py runlog.py(**无领域语义、无 carla/torch**)
└── tests/        54  与能力目录镜像(包级守卫 test_layer_guard.py 在根)

tools/                 开放性工具(判据:不含本项目领域知识):carla_server.sh + gpu_fix/
docs/(含 fileTree.md / refactor-2026-09.md)  README.md  Plan.md(冻结)  Plan2.md(冻结 · §P-M.1–P-M.20)
                        Plan3.md(主线外:外部数据集对照)  Plan4.md(新计划制定地)
outputs/  logs/(19 入口的运行三件套)  training/  lightning_logs/  hdMapGitHub/  auto3dlabel/  【未入库】
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
python -m autodrivedata.sim.collect_ab_route --scene day_clear --occluders partial \
    --occluder-cars nearest --out outputs/kitti_ab_occl2_partial_nearest            # P1-6b 静态遮挡(§P-V12)
#   `--occluders {none,partial,full}` 只添 `static.prop.*` ⇒ 进不了 label_2、逐帧 GT 仍相等;
#   ⚠️ `--occluder-cars all`(默认)会**饱和**(四堵墙同侧串联 ⇒ 与阳性对照不可分辨),要可分辨用 nearest
#   A/B 硬门槛现在多一条:两侧 `training/pose/` 逐帧位置差(max|Δx| ≤ ~0.07 m;大了先查启动瞬态)

# 3DGS 环绕采集(**天气无条件钉死**,默认 day_clear;A/B 只差"有没有道具";
#   落 capture/weather.json 记读回值 —— 采集器原先不 set_weather、也不记,
#   导致一对 A/B 实际差的是天气而全程静默,见 docs/edit-3dgs-plan.md §C1)
# python -m autodrivedata.sim.collect_3dgs --scene day_clear --inst --props --out outputs/3dgs_ab3/A
# python -m autodrivedata.sim.collect_3dgs --scene day_clear --inst          --out outputs/3dgs_ab3/B
# 逐类编辑 LiDAR 的前置(点云线重评条件 #3):collect_ab_route 另落一路
#   training/semantic_velodyne/{fid}.bin(float32 (N,4) = x,y,z,tag,逐行与 velodyne/ 对齐)

# 静态 GT(landmark + 车道线,含 overlay 目检图)
python -m autodrivedata.sim.collect_static_gt --frames 40

# 双传感器采集(6 相机 + LiDAR + 5 雷达 + GT 位姿,**一个 root 三种布局**;唯一能跑
#   「环视预测 + SLAM 位姿」组合的数据源,`stitch_temporal --pose slam` 靠它)
python -m autodrivedata.sim.collect_surround_lidar --frames 200 --out outputs/dual_town10
python -m autodrivedata.sim.collect_surround_lidar --autopilot --frames 240 --stride 5 \
    --map Town10HD_Opt --out outputs/dual_run/seq     # ★ 长序列的唯一办法(定速 ~133 m 被挡停)
#   ⚠️ autopilot **不可复现** ⇒ 只用于训练/演示,不得进 A/B;`calib.json` 会写 route=autopilot
#   + reproducible=false;**必须与 --speed 互斥**(两个都写 ego 控制,不静默取一)

# BEV 底图(⚠️ **只是视觉上下文**:模型纯相机,LiDAR/Radar 不进网络 —— 别读成多模态融合)
#   底图与矢量**必须同一个 --pose 源**,否则两张各自都对、叠起来错位
python -m autodrivedata.map.eval_maptr --infos <out>/map_infos.json --root <out> \
    --ckpt <ckpt> --start 0 --out-frames <preds>
python -m autodrivedata.map.stitch_temporal --frames <preds> --infos <out>/map_infos.json \
    --which pred --lidar-root <out> --out outputs/temporal/map_pred   # pred / gt 各拼一张
python -m autodrivedata.map.viz_maptr_pred --infos <out>/map_infos.json --root <out> \
    --ckpt <ckpt> --start 0 --frames 6 --bev-pair --lidar-root <out> --out-dir <viz>

# SLAM 序列采集(闭环模式:路网找环 + 纯追踪跑两圈 ⇒ 让同一处真被走两次,后端回环才有数据)
#   ★ 先 --dry-run:报环长/速度上限/帧预算(--speed 0 = 自动取 min(8, 环的上限))
python -m autodrivedata.sim.collect_slam --route loop --dry-run --map Town10HD_Opt
python -m autodrivedata.sim.collect_slam --route loop --laps 2 --map Town10HD_Opt --out outputs/kitti_loop
#   闸门:一圈帧数 = 环长/(速度×tick) 必须 ≥ 250(SC_MIN_GAP_NODES × KEYFRAME_EVERY),
#   否则两次到访帧差不够、回环候选必然为空 —— 环短就得开慢,或换 --spawn-index

# 回环链路端到端(前端 → 后端 → 评估;802 帧闭环数据实测 = 前端 12.6 min + 后端 11 min,
#   成本主项是**失配候选**的 ICP(未加闸前是**小时级**),见 Plan2 §P-H.3.3;判据与边界见 §P-H.3)
#   ★ 三步每次跑都落 logs/ 三件套;长跑中途被中断时,靠 .jsonl 判"跑到第几帧/卡在哪一档"
python -m autodrivedata.slam.slam_odometry --root outputs/kitti_loop --frames 0-801 --out outputs/slam_loop
python -m autodrivedata.slam.slam_backend  --traj outputs/slam_loop/traj_raw.json --root outputs/kitti_loop --out outputs/slam_loop
python -m autodrivedata.slam.eval_slam --traj outputs/slam_loop/traj_raw.json --gt outputs/kitti_loop --out outputs/slam_loop/eval_pre.json
python -m autodrivedata.slam.eval_slam --traj outputs/slam_loop/traj_pgo.json  --gt outputs/kitti_loop --out outputs/slam_loop/eval_post.json

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
# 地图矢量三格式(默认 opendrive = 现有 json;两个开关**互斥**,可 --from 读回当 GT 源)
python -m autodrivedata.map.export_mapvec --map Town10HD_Opt --lanelet2
python -m autodrivedata.map.export_mapvec --map Town10HD_Opt --apollo   # ⚠️ divider/boundary/centerline 借 lane 承载

# 逐帧契约落盘(供 AutoLabel 消费;GT 同文件携带,见 autodrivedata/map/mapvec_schema.py)
python -m autodrivedata.map.eval_maptr --infos outputs/surround_v2/map_infos.json \
  --root outputs/surround_v2 --ckpt outputs/maptr_v2_singleF.pt --start 200 --out-frames outputs/surround_pred

# 2D A/B 评估(A=day_clear 基线与 B 帧级配对)。**默认后端 = SAM3 开放词表**;
#   `--backend yolo` 回退到闭集 KITTI 微调模型 —— ⚠️ **两边 AP 不可比**(见红线)
python -m autodrivedata.perception.eval_2d_ab --root-a outputs/kitti_ab_epic_clip2d_day_clear --root-b outputs/kitti_ab_epic_clip2d_dense_fog
python -m autodrivedata.perception.eval_2d_ab --root-a ... --root-b ... --backend yolo   # P1 归档矩阵是这一档

# 失效归因(逐帧匹配 → 距离/框高/TTC 分箱 + 漏检画像)
python -m autodrivedata.perception.eval_attr --run day8=outputs/kitti_sweep_day_clear_8:8.0 \
  --run rain=outputs/kitti_ab_rain_night:8.0 --json outputs/attr.json

# P2-A 像素级分割 GT + 判据(§P-V13):采 GT 时加 `--sem`(逐相机多挂一路语义相机,
#   同挂点同 fov ⇒ tag 与 RGB 像素对齐;默认关),再拿它给语义 BEV 打分
python -m autodrivedata.sim.collect_surround --sem --frames 20 --out outputs/surround_sem_demo
python -m autodrivedata.perception.sem_eval --root outputs/surround_sem_demo --frames 0-19
python -m autodrivedata.perception.sem_eval --root <root> --frames 0-19 --self-test  # 不出模型,只验尺子
#   报**两个口径**:图像 mIoU(分割网络行不行)/ BEV 逐类 IoU(这张 BEV 图能不能用)
#   ⚠️ 某类 BEV 一个像素都没有时 mIoU 会**跳过它**并单独打印警告 —— 别把那条警告读成"这类做得好"

# P2-B 静态道具 GT + 判据(§P-V14):`--props` 摆 5 个受控道具(锥桶/路障/施工围挡),
#   默认关 ⇒ 关着时产物与既有口径逐字节一致。**道具不进 label_2**(会破 A/B 帧级配对)
python -m autodrivedata.sim.collect_static_gt --props --frames 40 --out outputs/kitti_static_props
python -m autodrivedata.perception.prop_eval --root outputs/kitti_static_props   # 离线,不需 CARLA

# P2 静态 GT(landmark + 车道线)的判据(§P-V21):`--sem` 同挂一台语义相机当 oracle,
#   并在 static_gt/{fid}.json 里落**实读**相机位姿(判据在 perception/,层规则禁 carla,
#   拿不到 CAM_ATTRS/SENSOR_OFFSET ⇒ 位姿必须随帧落盘,否则判不了 —— **缺了直接抛**)
python -m autodrivedata.sim.collect_static_gt --sem --frames 40 --out outputs/kitti_static_sem
python -m autodrivedata.perception.static_eval --root outputs/kitti_static_sem   # 离线
python -m autodrivedata.perception.static_eval --self-test    # 不出数据,只验尺子(5 条)
#   报 **真命中 / 横移对照 / 同 v 随机基准** 三个数,缺一个都读不出结论:
#   横移对照 = 位置信息有没有起作用;随机基准 = 这个命中率是不是随便一指就有
#   ⚠️ **裁决要有第二条** `real ≥ 5×max(对照)`:少了它,一个"到处都是标线"的场景
#     里 real=0.4 也会判通过 —— 那不是"xodr 对得上"
#   ⚠️ `n=0` 的组(全在画外)是**未判**(第三种裁决),单独喊一声、不计入裁决
#   ⚠️ off_frame 拆成 **车后 / 视场外** 两栏 —— 信号锚点实测 144/221 **在相机后**
#     (LANDMARK_HORIZON 是**圆形**过滤,与红线里灯态那条同源)
#   产物:static_prop_gt/{fid}.json(世界系记录 + 相机位姿/内参) + prop_inst/{fid}.png(uint16
#   实例 id 图 = **几何 oracle**) + prop_sem/{fid}.png(语义 tag = **类别 oracle**) + overlay
#   判据报 coverage(轮廓像素落在投影框内的比例,应当 ≈1.0)/ tightness / IoU
#   ⚠️ 物体**被画幅裁断**的样本会被**单独计数、不并入裁决** —— 那是共享投影函数的口径问题
#     (归档 label_2 里 690/3922 = 17.6% 同形),不是道具 GT 的错,见 Plan4 §P-V14 四 / §2 待决项
#   ⚠️ 往已有道具 GT 的 root 里再写会被**拒绝**(两轮产物混在一起会让判据误报"实例没露面")
#   探针(一次性,需 CARLA):python -m autodrivedata.sim.probe_static_prop_gt

# 实例分割 GT + 判据(§P-V15):`--inst` 逐相机多挂一路 instance_segmentation(同挂点同 fov)
python -m autodrivedata.sim.collect_surround --sem --inst --frames 20 --out outputs/surround_inst_demo
python -m autodrivedata.perception.inst_eval --root outputs/surround_inst_demo --frames 0-4
python -m autodrivedata.perception.inst_eval --root <root> --frames 0-4 --self-test  # 不出模型,只验尺子
#   报 mask AP + PQ/SQ/RQ:RQ = 物体找得全不全,SQ = 掩膜画得准不准,**两个数分得开**
#   ⚠️ 实例评测**只对可数物体**(Pedestrians/Car/Truck/Bus);`drivable`/`lane` 是 stuff,没有实例语义
#   ⚠️ 类级 `sem_eval` **保留不动**,两把尺子并列 —— 一张"把两辆车连成一片"的图,类级可能很高而实例级同时记 FP+FN

# 三传感器融合(§P-V18):后融合 = LiDAR 出几何 + 相机出类 → 3D 关联 → 判据
python -m autodrivedata.perception.eval_fusion --root outputs/kitti_ab_occl2_none_all --frames 0-19
python -m autodrivedata.perception.eval_fusion --root <root> --frames 0-19 --no-camera  # 只跑 lidar 档
#   报 **3D AP@0.5 + 逐条 TP/FP/FN**(两档 conf 来源不同,只比 AP 会把"排序变了"读成"检测变好了")
#   ⚠️ 雷达**不在表里** —— 平台边界:静止车上归因回波实测 0(§P-V17 六)
#   ⚠️ `cluster.dbscan(distance_scale≠0)` 会切到纯 Python 实现(同帧 RANSAC 0.08 s vs 聚类 129.7 s)
#   `--gates loose,strict` 出 **2×2**(门×模态):「严门单独」是「相机有没有独立贡献」的对照
#     ⇒ P1 静置车上**融合没赢**;`collect_ab_route --layout close`(多类+近距,默认关)上**翻过来**

# 雷达速度对表(§P-V19):**不需要目标**的雷达判据 —— 静止场景里径向速度必然 = 自车速度在
#   该视线上的投影。用来答"雷达的速度通道有没有物理意义"(目标那一路实测回波 0)
python -m autodrivedata.perception.radar_eval --root outputs/kitti_ab_occl2_none_all --frames 0-39
#   实测 3/5 达标(|k|≈1、R²≈0.70、打乱对照塌到负):FRONT / BACK_LEFT / BACK_RIGHT
#   ⚠️ 两个**前侧**雷达的速度读数是 0;判据**允许一个全局符号**(vel「朝传感器为正」)

# 变体选择:一键 --variant mapqr(= 散聚 query + 高度核 BEV 编码器),或细粒度开关做消融
python -m autodrivedata.map.train_maptr --infos outputs/surround_v2/map_infos.json \
  --root outputs/surround_v2 --variant mapqr --out outputs/maptr_mapqr.pt
#   两者互斥(同时给报错);eval/viz/studio **不必也不能**再指定变体 —— 权重自带结构说明
#   细粒度消融(§P-V11 第二批,四档参数量):基线 33.2M / ②`--bev-encoder height_kernel` 40.6M /
#     ①`--scatter-gather` 119.6M / ③`--variant mapqr` 127.0M

# 第二张图 + 双图池(§P-V25)。⚠️ **Town11/12/13/15 是平铺大图,本机相机采集必崩**(段错误),
#   第二张图只能用 Town01–07 / Town10HD。先用探针选段起点(带自证:必须复现 v2_epic 那五个索引)
python -m autodrivedata.sim.probe_spawn_points --map Town05_Opt --k 5 --out outputs/town05_spawns.json
python -m autodrivedata.map.export_mapvec --map Town05_Opt --out training/map     # 出 {图}_full.json
#   逐段采集(与 surround_v2_epic 同源:同采集器 / 同 rig / 同 autopilot 70%)
python -m autodrivedata.sim.collect_surround_lidar --map Town05_Opt --spawn-index <i> \
    --autopilot --frames 100 --stride 5 --out outputs/surround_town05_epic/seg<K>
#   双图池 = 两个数据集各段**符号链接**进同一父目录(不复制 8.6 G),再按段自动解析矢量 json
#   out/multimap: seg0-4 -> surround_v2_epic/seg{0..4}、seg5-9 -> surround_town05_epic/seg{0..4}
python -m autodrivedata.map.assemble_maptr --segs-dir outputs/multimap \
    --map-json auto --out outputs/multimap/map_infos.json
#   ★ `--map-json auto` 逐段读 calib.json["map"] 推 `{--map-dir}/{名字}_full.json`;
#     **显式给文件时逐字节不变**(在真实数据上验过:500 帧 sha256 与归档相同)
#   ★ 收尾有 **rig 自证**:`MapTRDataset` 只认首帧那一份 cams,混 rig 不报错、只让训练学不动
#     —— 真实数据负向对照实测 5 s 抛出并点名两段(改前排在末尾、跑 300 s 都报不出来)
# 早停(默认开,两段式):loss 平台作候选 → **留出 AP 复核** → 都没升才停;停时把 best-AP 权重
#   恢复成 --out(不是"停在触发时刻"),并存 <out>.early_stop.png(曲线 + lr 变化点 + 触发点)
#   ★ 三条不启用路径会在开跑时打印原因:单帧锚点豁免 / 没给 --eval-* 留出选择器 / --no-early-stop
python -m autodrivedata.map.train_maptr --infos outputs/surround_v2/map_infos.json \
  --root outputs/surround_v2 --frames 0 --exclude-seg seg4 --keep-in-seg 0:80 --epochs 400 \
  --eval-exclude-seg seg4 --eval-keep-in-seg 80:100 --out outputs/maptr_es.pt
#   调参:--patience(默认 20)/ --min-improve(1e-3,loss 相对门槛)/ --ap-min-improve(1e-2,须 ≫2e-3 噪声下限)
#   ⚠️ 至少两次成功复核才会停(第一次只建基线)⇒ 最短也要 patience + 2×confirm-every 个 epoch
#   ⚠️ 复核会写临时 ckpt 再评(每次 ~134 MB + 一次留出评估),评估**固定 batch=1**(自适应读空闲显存,会随机器状态变)

# 跨图拼接(⚠️ 官方 Town 无真值相对位姿,placement 是人为摆位;合并图对训练无用)
python -m autodrivedata.map.export_mapvec --out outputs/stitched \
  --stitch "Town10HD_Opt=0,0,0,0;Town01=2000,0,0,0"    # 或 placements.json

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

# 运行日志:**60 个**训练/推理/评估/判据入口**每次跑都落三件套**到 logs/(同 stem;
#   = 用户裁决的 16 + 2026-09-27 补入的 slam_odometry / slam_backend
#     + 2026-09-30 补入的 runtime/tb_export
#     + 2026-10-02 补入的 perception/static_eval
#     + 2026-10-04 补入的 gs/frame_sync / gs/attribute_instances
#     + 2026-10-07 补入的 sim/probe_3dgs_cam_pose / probe_prop_renders / probe_radar_camera_coaxial
#                       与 edit/harmonize_data / edit/train_harmonize / **edit/harmonize_axis**
#     + 2026-10-08 补入的 **perception/domain_gap / domain_adapt / probe_bev_depth / size_prior**
#                       与 **gs/probe_hole_visibility / probe_hole_render**、**sim/probe_ego_teleport**
#     + 2026-10-08 再补入的 **gt/export/coco2kitti**、**edit/augment**、
#       **gs/insert_gs**、**gs/probe_edit_downstream**
#     + 2026-10-06 补入的 edit/calibrate / edit/harmonize_target —— ⚠️ 计数用
#       `grep -rl 'runlog.run(' autodrivedata/ --include=*.py` 复核,别手抄)
#   <能力>_<模块>_<YYYYmmdd-HHMMSS>.log = tee 的全量 stdout(头块含 git/GPU/env/argv)
#   .jsonl = 逐迭代指标(逐行 flush);.json = 环境指纹 + 入参 + 产物表(带 sha256) + 结论
#   logs/latest/<能力>_<模块>.<ext> = 指向该脚本最近一次的**相对软链**(tail -f 用它)
#   关掉:--no-runlog,或 AUTODRIVEDATA_RUNLOG=0;口径与坑见 autodrivedata/utils/runlog.py

# TensorBoard(离线转换,不在训练里实时写 —— 理由见模块头注:两份记录迟早漂)
#   .jsonl 已是逐迭代的唯一口径 ⇒ event 文件是 **view 不是 source**;转换器不认训练语义,
#   对 train/eval/slam 一视同仁。⚠️ 标签稀疏(早停复核行没有 loss)⇒ 逐行按"有什么写什么"
pip install tensorboard
python -m autodrivedata.runtime.tb_export \
    logs/map_train_maptr_20260930-005454.jsonl=mapqr_new \
    'logs/map_train_maptr_20260929-*.jsonl=k3_baseline' --out outputs/tb
tensorboard --logdir outputs/tb --host 127.0.0.1 --port 6006
#   远程:ssh -L 6006:127.0.0.1:6006 <autodl> → 浏览器 127.0.0.1:6006
#   ⚠️ event 是**快照**,刷新要重跑转换器;*想*实时进度请 tail .jsonl
#   曲线内容:`loss`/`cls`/`pts`/`lr` 每 epoch;**留出 AP 按类** `confirm_AP/<类>` + 逐类
#   `confirm_n_pred|n_gt/<类>`(每次复核一个点 —— 分类别才看得出是「一类到顶、另一类还在涨」
#   互相抵消);`confirm_mAP`/`best_ap`;`wall_s`。SLAM/3DGS/eval 的 jsonl 走同一个转换器

# 图像编辑线(§edit-image-plan.md;两个官方 repo 在 hdMapGitHub/ 且保持 pristine,
#   版本适配全在 edit/cldm_backend;权重在 weights/controlnet/,带 MANIFEST.txt 记 sha256)
#   真值深度 → 条件图(纯值,不出模型)
python -m autodrivedata.edit.depth_cond --root outputs/3dgs_sync --frames 0,10,45 --out outputs/edit/cond
#   条件可控生成 + **条件保真度(带对照臂)**:--kind {gt-depth,midas-depth,canny}
python -m autodrivedata.edit.conditioned_gen --root outputs/3dgs_sync --frames 0,10,45 \
    --kind gt-depth --normalize minmax --fidelity --out outputs/edit/gt_minmax
#   ⚠️ 三个口径**必须随读数一起报**:--steps(20) / --scale(9.0) / --seed(42);512²/20 步 ≈ 8 s、峰值 10.7 GiB
#   ⚠️ 实测结论(3 帧):真值条件余量 **+0.219/+0.326/+0.215** vs MiDaS 估计 **+0.052/+0.157/+0.017**
#      ⇒ **真值胜 4.2×/2.1×/12.8×**。⚠️ n=3 是**机制验证**不是分布读数
#   ⚠️ 整图 Spearman **不是**好判据(被共享先验撑满、看不见强度分布、MiDaS 在 512² 中心裁上不稳)
#   换 prompt 不换条件(JD 要的"按需换天气、**布局不变**"):固定 --frames 45 扫 --prompt
#   —— 实测**结构余量 +0.4186**(同条件 0.986 vs 换条件 0.568)、外观亮度极差 **22.0 vs 1.6**
#   ⇒ **布局由条件管、外观由 prompt 管,两个方向都成立**(§1.5)
#   ★ **下游闭环**(JD 那句「确保仿真数据可用于感知模型训练」的落点):四臂配**同一套 GT** 比 ΔAP
python -m autodrivedata.edit.kitti_square --src <root> --dst <方裁root> --dry-run
#     1242×375 → 512²;label_2 的 2D 列做**同一仿射**(3D 列不动)。实测 183 条 → 115 kept / 55 clipped / 13 dropped
python -m autodrivedata.edit.downstream_eval --clear <方裁> --fog <真值退化方裁> \
    --gen <生成产物目录> --work <四臂落点> [--backend yolo]
#   β 标定产线(§1.11):一条命令 = 扫曲线 + 拟合 + 出配方。★ **默认 101 点口径** ——
#   11 点下曲线**不单调**(实测 +0.0763 / −0.0026 / −0.0013),拿它内插没有意义
python -m autodrivedata.edit.calibrate --base <底图root> --work <曲线落点> \
    --target-arms <参考root>:<退化root> [--grid-points 101] [--apply <产物root>]
#   和谐化的**真值靶**(§1.12):A/B 同帧跨天气粘贴,真值 = 原图;掩膜 = GT 车辆框。
#   ★ 几何由 A/B 硬门槛保证**且可验**:两 root 框坐标 max 差超 2 px **当场抛**
python -m autodrivedata.edit.harmonize_target --ctx <A天气root> --src <B天气root> \
    --frames 0-39 --out outputs/edit_p1/harmonize_target
#   ★ 实测(35 帧,SAM3 conf 0.5):对照地板 **−0.429** ≪ 真值浓雾 **−0.170** ≪ 生成浓雾 **+0.010**
#     ⇒ **生成的退化不能替代真值退化**,且原因是**结构性的**:深度条件要求把物体画清楚 ⇒ 它必然可检测
#     (生成雾的天空带亮度 156 vs 真值雾 196 —— **亮度≠可见度**;目检见 edit-image-plan §1.6)
#   ⚠️ `eval_2d_ab` 把**所有帧的框池化**后算 AP ⇒ "打乱配对"那类对照**不是对照**(重排=恒等);
#      对照臂必须让**图集合**不同(downstream_eval 取后半程的图配前半程的 GT)

# 规范 + 测试(提交前两件套;规则集钉死在 pyproject [tool.ruff],110 列)
ruff check && ruff format        # format 无参数即就地格式化,全仓口径统一
python -m pytest -q              # testpaths 已钉在 pyproject;**别裸敲 pytest 之外的路径前缀**
                                 # ⚠️ **本机 `.bashrc:131` 无条件 `source ros2_humble/install/setup.bash`**
                                 #   ⇒ 每个新 shell 的 PYTHONPATH 里塞着 ~90 条 ROS2 路径,
                                 #   其中 `launch_testing` 注册的 pytest 插件在**启动期** import
                                 #   失败(缺 `lark`)⇒ **上面这条命令直接崩**。实测绕法:
                                 #   `env -u PYTHONPATH python -m pytest -q`。
                                 #   这是"不引入 ROS2"那个结论的**执行面漏洞**(Plan4 §P-V? / 待裁决)
                                 # 基线:2153 passed + **9 跳过** + 0 失败(2026-10-08 实测,175 s;
                                 #   本轮 **+29**:COCO→KITTI 导出 20 + 注入配方 9
                                 #    (另一批见下)
                                 #   上一轮 **+73**:窟窿可见性 14 + 窟窿渲染 7 + ego 瞬移 8
                                 #   + runlog 环境链 5 + 域自适应 12 + BEV 深度 14
                                 #   + 尺寸先验 12 + (上一轮 +48 见下)
                                 #   上上版 2051:3DGS 帧对齐钉 5 + A/B 光度门槛 6 + 天气门槛 4
                                 #   + collect_3dgs 天气钉 4 + 下游 --grid-points 通路 3
                                 #   + 和谐化收口判据 8 + semantic 标签版 4
                                 #   + LiDAR 位姿诊断 5 + 域差矩阵 9
                                 #   上一版 2003 / 1990 / 1988 / 1980 / 1975 / 1924 / 1919 / 1909 / 1861 / 1812 / 1741 ——
                                 #   本轮共 **+262**(+64:AP 台阶判据 15 + 标定产线 12
                                 #   + 和谐化真值靶 14 + 掩膜版 harmonize 6 + 四臂 GT 冻结 2
                                 #   + 文档守卫扩到 3 份 edit 计划 5
                                 #   + 相机位姿口径 8(纯函数手算锚点 3 + 源码结构钉 5)
                                 #   + 入口守卫扩到全包 +3、负缺口守卫 +2、取帧序号 +1
                                 #     − 被取代的 edit 专属守卫 2 = +4
                                 #   + 和谐化数据/训练 13)
                                 #   + 上一段 +183)
                                 #   (degrade 20 + video_edit 14 + harmonize 12 + 入口守卫 2)。
                                 #   **8 条跳过里有 2 条是显式的**:没给 `CUDA_HOME` 就跳过真调 gsplat
                                 #   的用例,理由见 `tests/gs/test_render_gs.py` 的 `_NEEDS_CUDA_ENV`
                                 #   —— 不给环境时 torch 会**重编并可能覆盖规范 `.so`**。
                                 #   上一版 1519(再上一版 1498 / 1482 / 1327 / 1174 / 1110),
                                 #   +35 静态遮挡纯几何,+22 采集器结构钉,+2 退化 GT 剔除,
                                 #   +7 读侧判据(含 0.01 px 那条),+5 GT 口径迁移工具 = +71;
                                 #   +17 语义 tag 表纯值,+11 carla oracle,+14 分割判据,
                                 #   +7 投影向量化对拍 = +49;
                                 #   再 +29 道具类别表与三处口径汇合点,+15 道具判据(合成 root),
                                 #   +7 PROP_TAGS 与第三桶 = +51;
                                 #   再 +18 实例 id/类纯值,+20 实例判据(合成掩膜),
                                 #   +3 类级掩膜并集等价,+2 refilter 雷达与跨设备 = +43;
                                 #   再 +22 后融合几何/尺寸门/关联/消融,
                                 #   +13 雷达速度对表判据,+13 融合判据键名对表 = +48;
                                 #   再 +5 裁断口径重算(rebox_line)+6 --clip2d 迁移
                                 #   +2 PNG 画幅头 = +13)
                                 #   再 +13 裁断口径重算(rebox_line)+6 --clip2d 迁移 +2 PNG 画幅头
                                 #   = +21;
                                 #   再 +21 静态 GT 判据(static_eval 20 + static_gt 的相机位姿往返 1);
                                 #   再 +9 采集器帧同步自证(settle/assert_synced);
                                 #   再 +13 后端抽象与提示词表(SAM3 换默认);
                                 #   再 +10 去重与按后端定阈值;再 +4 操作点读数(命中/召回);
                                 #   再 +16 段起点选点(种子规则/单调性/平局/嵌套),
                                 #   +5 切图超时, +15 多图池(三形态解析/逐字节等价/rig 反向自证)
                                 #   = +36;
                                 #   再 +43 3DGS 阶段 B(帧同步判据 12 + Gaussian→实例归属 20
                                 #   + 采集器挂载/帧同步结构钉 11)
                                 #   + 12 训练环境的三个静默失效(seed 被覆盖回 0 的 4 条
                                 #   + CUDA 构建告警排在 import 之前的 4 条
                                 #   + 相机系默认值的**唯一落点** 4 条
                                 #   + 产物归属字段不许掉 1 条)
                                 #   + 28 3DGS 编辑开局(cuda_env 顺序守卫 12 + render_gs 10
                                 #   + 采集器清场/道具 6
                                 #   + 编辑算子与三档判据 23
                                 #   + 3D 框删除/掩膜源/容差耦合 10)
                                 #   + 22 LiDAR A/B 判据(含 2 条反向自证)= **1741**
                                 #   ⚠️ 上面 1741 是**实测值**。**加删用例后改这一行**
                                 # ⚠️ 其中 6 条在 `importorskip("carla")` 的模块里 —— 收集期不计
                                 # ⚠️ 下游 AutoLabel 的 mapvec 用例另算:`autolabel` env 下
                                 #   `pytest auto3dlabel/tests/functional/` = 342 passed + 4 skipped
                                 #   (2026-09-30 实测)
                                 # 基线数**只写在这一处**;加/删用例后回来改这一行,别在多处复述
```

## 红线(A/B 实验纪律与已踩坑,勿再犯)

- **A/B 帧级配对是硬门槛**:同 ego 锚定 spawn point 0(yaw=0)、同静置车布局(20/35/50/62m——65m 会卡 GT max_distance 阈值抖动)、只变 weather;GT 数必须相等,否则样本不可比、结论作废。autopilot/TM 路线不可复现,禁用于 A/B
- **跨机器/跨机型的 AP 不许直接比**:显存变 ⇒ `auto_tune_batch_size` 实测选到**不同的 batch** ⇒ 新旧 AP 不可比。判据看 `logs/*.json` 的 `highlights.batch`,**对不上就别比**;`.log` 头块的 GPU 型号/显存/CUDA/driver + git rev 是归属依据
- **AP 尾部不注水**:未达 recall=1 段 precision=0(11 点插值,与 compare.ap11 同口径)。旧尾行 `ap += (1-prev_r)*prev_p` 曾把低 recall 吹高(雨夜 0.48 检出报 0.976),已修
- **MapTR AP 的口径参数现在有三个:阈值 / batch / TF32**。`--batch` 会让 AP **单调漂**(实测 1/2/4/6 → 0.3043/0.3043/0.3044/0.3045,约 +5e-5 每 2 个 batch)—— 跨权重比较两边必须同 batch。**TF32 更隐蔽**:`cudnn.allow_tf32` 在 torch 2.x **默认 True**,而全仓原本无人设置它 ⇒ 归档的 0.3043 是 TF32 关的口径、今天默认开着跑出 0.3048(差 5e-4,`pred/gt` 计数完全一致)。`eval_maptr` 现在**入口即 `disable_tf32()`**(见 `autodrivedata/runtime/device.py` 头注的实测表);**新增 torch 推理入口请照做**
- **MapTR chamfer AP 必须带 score_thr 引用**;跨权重比较**固定 `--score-thr`**,看曲线用 `--sweep`;**单独报一个 mAP 数字而不写阈值 = 无效结论**
- **早停的 best-AP 恢复必须是最后一次写 `--out`**:恢复块之后**不许再有** `save_map_checkpoint(model, args.out)` / `_save_opt_sidecar`。2026-09-30 实测的真 bug:恢复之后紧跟一次无条件存盘,把内存里**触发时刻**的权重盖回去 —— 日志照写「已恢复 epoch 296」,盘上却是 ep329(留出 mAP **0.1557 → 0.1243**),而 `.best` 已被 unlink、`--save-every` 又覆盖同一个 `--out` 不留历史 ⇒ **那份权重不可恢复**。同族:`rl.artifact` 是**调用时立刻 sha256**,权重类产物的登记也必须排在恢复之后。判据是**源码顺序**(`tests/map/test_maptr_select.py::TestCheckpointFinalizeOrder`),行为测试要跑 12 h 真训练
- **质量档是渲染口径,不是性能旋钮**:`-quality-level=Low` 下 CARLA **不渲染雾**(旧「浓雾 Δ−0.013」量的是晴天),**湿路面材质也坏**(渲染成饱和异常色 —— 同键在静止探针出蓝、在 A/B 采集出品红 ⇒ 着色器问题,不是「路面湿了长这样」)。默认已改 `Epic`(实测 6.0 GB/48 GB、4.94× 实时;**当年选 Low 的 12 GB 显存约束已随机器更替消失**)。**跨档数据集不可混比** —— 旧 P1 矩阵 / `surround_v2` / `nus_mini` 全部是 Low 档,引用时必须写档位
- **跨权重的比较有「选择下限」σ ≈ 0.025 —— 不是一个数,是分辨率**:同一份 48 帧留出、同一口径下,四个不同 checkpoint 的 mAP 实测 `0.0945 / 0.1173 / 0.1243 / 0.1557`(样本 σ = **0.0253**,极差 0.061)。**这与已记录的复现性下限 2e-3 差 12.6 倍,是两个量** —— 2e-3 说的是「同一份权重换进程能差多少」,0.025 说的是「**不同权重之间**数字本来就散多少」。⇒ **效应 < 0.03 一律不许报涨/跌**;报差必须写明评测集是什么(帧级/路线级、多少帧)并带出该比较的分辨率。实测代价:2026-09-30 一轮 15 h,交付权重最后是**按溯源一致选的、不是因为它更好**(两者差 0.007 = 0.28σ);早停判 `converged` 的依据也是 0.1557 vs 0.1243 的**单次比较**。⚠️ 全 240 帧与 48 帧留出曾给出**相反**方向 —— 因为全量里 80% 是训练帧,**奖励过拟合**
- **自适应 batch 的峰值不过原点**:`_bs_from_budget` 必须减掉 **batch=1 的常驻足迹**(参数 + 优化器状态 + cudnn workspace + 一份激活)。实测(MapQR @3090-48G)12.98 / 25.39 / 37.77 / **OOM** GiB,旧公式 `1 + floor(空闲×0.95/增量)` 算出 4(需要 ≈50 GiB)—— **一跑就炸**。教训同族:探针测的是**差值**,预算里要填的是**峰值**。改这里跑 `tests/runtime/test_device.py::TestBatch1Footprint`(纯算术,不需要 GPU)
- **★ 3DGS 的 `--seed` 不给逐位复现**(2026-10-04 实测,同 seed 重跑):`psnr_all` 分辨力 = **σ(n=5) = 0.0335 dB**(2026-10-07 补齐:旧记的「极差 0.08」是 **n=2** 的,而本仓自己有条「n=2 的极差不是分辨力」—— 补跑 3 次后 mean 30.158、极差仍是 0.080(**巧合**)但 sd 只有 0.0335;`val45` 的 0.18 未补) —— gsplat 的光栅化反向用 **`atomicAdd` 累加梯度**,浮点加法顺序随线程调度变,**seed 管不到 CUDA 原子序**。⇒ 报 3DGS 差异必须写明分辨力;`< 0.1 dB` 一律不许当「涨了/掉了」。⚠️ 与「AP 复现性下限 2e-3」**是两回事、量级也不同**(那个是阈值边界抖动),**两个数不许互相引用**。⚠️ 别拿「两个 seed 的极差」当分辨力:实测它比同 seed 重跑**还小**,会把任何 Δ 系统性放大。★ **「<0.1 dB 不许当涨跌」= 3.0σ** —— 门槛数量级对得上,现在有实测 σ 撑着了(见 [docs/edit-3dgs-plan.md](docs/edit-3dgs-plan.md) §A.3.6)
- **BEV 底图不是多模态融合**:MapTR/MapQR 是**纯相机**模型,LiDAR/Radar 一个点都没进网络;底图只把点云按位姿投进**矢量同款平面系**当上下文。带底图的产读成「融合结果」,就是把不存在的因果讲了出来。底图与矢量**必须同一个 `--pose` 源**(一边 GT 一边 SLAM = 两张各自都对、叠起来错位)
- **★ AP 是台阶函数:11 点插值下,单条边界检测最多可换 `1/11 ≈ 0.091`**(2026-10-06 实测)。11 点的 recall 格点是 `j/10`,格点 `j` 可达 ⟺ `n_tp ≥ ceil(j/10·n_gt)` ⇒ **`n_tp` 每跨过一个门槛,那一格的 precision 从 0 跳成正值**。实测(`blur`、单类 Car、`n_gt=119`):`tp 107→108` 跨过 `108/119 = 0.908 ≥ 0.90`,**一条压在 `IoU = 0.540` 上的检测就让 ΔAP 从 −0.026 翻成 +0.055**;同一批检测换 **101 点**口径凸起整体消失。**放大倍数 = `n_gt/11`**(119 时 10.8×;有合成对照钉住)。★ 它**把结论改判过一次**:生成浓雾的 Δ 从 **+0.003(11 点)** 变 **−0.062(101 点)** ⇒「假退化」→「退化偏弱(~1/3)」。⇒ **`ΔAP` 必须与 `n_tp/n_gt` 一起读**(`eval_2d_ab` 会自己喊"距格点还剩几个 TP");**要插值/拟合/投影一律走 `--grid-points 101`**(`edit/calibrate` 的默认);归档口径仍是 11 点,**两者不可混比**。⚠️ 与下面那条"复现性下限 2e-3"是**两个量**(那个是阈值边界抖动,量级差 ~45 倍),**不许互相引用**。见 [docs/edit-image-plan.md](docs/edit-image-plan.md) §1.10
- **★ 跨 root 比 GT / 比框坐标必须按「多重集」比,不能逐行 zip**(2026-10-06 实测)。两家 `label_2` 的**条数与帧号完全相等**、排序后 2D 差 **≤0.78 px**、3D 差 **≤0.06 m**(正是 A/B 硬门槛那一档),但**行序真的不同**(CARLA `get_actors()` 跨采集不稳定)⇒ 逐行 zip 读出 **215 px** 的假错位,差一点被读成"两个采集位置不一样"。★ 同一条也适用于"四臂 GT 必须冻结成一套":那 0.78 px 足以让一条压在 `IoU=0.5` 上的检测翻面,同一份预测换一套 GT 在 101 点口径下差 **0.009**(11 点口径下看不见)。判据 `harmonize_target.box_shift` / `downstream_eval.run_arms`
- **AP 的复现性下限 ≈ 2e-3**:同权重、同数据、同后处理,**换推理设备或换进程**也会让 AP 动 —— 边界实例的 sigmoid 得分跨过 `--score-thr` 就翻面(2026-09-28 实测:路线级 `boundary` **pred 1308@GPU vs 1307@CPU**,帧级留出 mAP 归档 0.3043 / 实测 0.3048,同一配置连跑 4 次则**逐位相同**)。⇒ **跨设备/跨机型的 AP 差 < 2e-3 一律视为不显著**,不许当"涨了/掉了"报;判据是**固定 `--score-thr` + 记录是否 GPU 推理 + `pred/gt` 计数**(计数不等 = 预测真变了,计数相同而 AP 变 = 阈值边界抖动)
- **每一路传感器队列每个 tick 都要抽干**(第三次现形,2026-10-01):队列是 **FIFO**,`get()` 取的是**最旧**那帧。漏掉一路的代价**不是"少收几帧"**,而是**那一路整体滞后 N 帧** —— 而它与别的路"都有数据、帧号也对得上",看不出来。三次实测:`probe_static_prop_gt`(裸 tick 绕过 drain ⇒ 掩膜恒 0)、`verify_nus_calib` 判据⑥(未测相机积压 ⇒ 应力掩膜恒 0)、`collect_surround` 预热循环漏 `inst_qs`(实例相机 R 通道 vs 语义 tag 从 1.000 掉到 0.76)。**判据 = 两路本应逐像素相同的东西比一比**(实例相机 R 通道 vs 语义 tag 有现成的)
- **★ 抽干≠抽干净:`drain` 一遍会留下"在途帧"**(第四次现形,**另一个变体**,2026-10-02):客户端投递是**异步**的,判"队列空"只代表*已经到的*取完了,**还在途的**会在之后补进来 ⇒ 每 tick 补一帧、取一帧,**陈旧量恒定**。实测 `collect_static_gt` 的 `image_2/{f}.png` 与**同一采集器**写出的 `prop_inst`/`prop_sem`/`static_prop_gt` **差 1 帧**,而帧号一张张对得上、图一张张出得来。**修法两条**:`settle()`(排空+tick 到"每队列恰好剩当 tick 那一帧")+ `assert_synced()`(每 tick 比三路 `Image.frame`,不等**当场停**)。⚠️ **受害者只有一半**:`prop_eval`/`static_eval` 只吃 `prop_inst`/`static_sem`,**不受影响**;错的是 `overlay/` 目检图与一切拿 `image_2` 配这套 GT 的下游。见 Plan4 §P-V22
- **★ 查"两帧对没对上"时,仪器本身要先验**:同一次排查里我换了四种读法,**两种给出过干净但错误的答案** ——「橙色像素占比」同时命中黄色标线(锥桶在该光照下渲染偏灰,掩膜内真实橙占比只有 0.03);「同帧比车道线亮度」没有判别力(同帧恰好也高于基线,被我读成"对齐")。**只有「固定掩膜 + 跨帧 argmax + 与基线分得开」立得住**,它给出 +1 帧;修完给 0。⇒ 判据**必须先证明"错了会怎样"**(把已知对齐的一对比一比、或看基线与峰值的间距),否则读数再干净也不作数
- **`refilter` 只动 `label_2`,旁路产物按谓词全链**:原实现写死 `("image_2","velodyne","calib","pose")`,**漏了 `samples/RADAR_*`**(它是 `training/` 的**兄弟**目录)⇒ 五份 `kitti_ab_epic_gtfix_*` 的雷达**静默消失**。名单是会过期的产物,"只动 label_2"是这条工具的定义 —— 定义写成 `is_rewritten()`。跨设备(`--out` 在别的盘)时 `os.link` 抛 `EXDEV`,`linked`/`copied` **分开计数**
- **★ CARLA 里 attach 到父 actor 的子 actor,`set_transform` 按【父系】解释**(2026-10-07 实测)。`collect_3dgs` 把相机 attach 在 spectator 上,却把 `ring_cam_pose` 给的**世界**坐标直接喂进去 ⇒ 相机落在 `请求 + spectator 的世界位置`,整条环绕链被**平移 79.132 m**。症状**全静默**:重建仍自洽(相对几何没变,与 §A/§B 的 27 dB 不矛盾)、不报错,只有**按世界坐标摆的东西进不了画面** —— 长得像「道具资产不渲染」。⇒ 换算走 `collect_rig.to_parent_frame`(纯平移,**只在父旋转为单位时成立**,故采集器先读回自证 spectator 的 rotation),每帧再读回自证相机落在哪;反向自证 `sim/probe_3dgs_cam_pose`:旧写法差 **78.034 m** / 新写法差 **0.000 m**。旧 capture 各带一份 `POSE_CONVENTION.txt`,**两代口径不可混比**
- **carla pyi 桩坑**:`try_spawn_actor` 桩标返回 `Actor`(实为 `Actor|None`)→ 用 Vehicle 方法必须 `cast(carla.Vehicle, v)`;Vector3D 运算结果不能直接进 `carla.Transform`(显式 `carla.Location`);函数签名要 `tuple[float, float, float]` 定长时禁用 tuple 推导(变长 tuple)
- **`Actor.bounding_box` 对转过 actor 给的是被「剪切」的值,不是两轴对调**:锥桶真值 `0.3441×0.3441`,yaw=30° 读回 **`0.1246×0.4704`**(≈ `s·|cosθ−sinθ|` 与 `s·(cosθ+sinθ)`)。拿渲染轮廓量:3.4% 的物体像素落到投影框**外**、IoU 0.897→0.708。**0° 与 90° 都读对** ⇒ 拿一个 yaw≈0 的样本验一次会得"没问题"。判据 = `carla_common.measure_actor_size_yaw0`(**四处共用:探针 / 采集器 / 遮挡摆位 / 尺寸对表**,别各写一份)
- **2D 框口径是两句配套的,拆开任一句都会坏**(2026-10-02 定案,Plan4 §P-V20):① **框** = 全 front 角点 min/max **再钳到画幅**(KITTI 口径);② **退化剔除**看**未钳**的 `inside` 角点跨度(`MIN_BOX_SIDE_PX`)。**只①不做②** ⇒ 「擦过镜头」的角点投影发散,钳完是**接近满幅**的框 ⇒ 与任何预测都能配上 ⇒ **凭空造 TP**,比"框太小"更毒;只②不做① ⇒ 被边缘裁掉的框偏小(归档 **690/3922 = 17.6%** 有角点出画,其中 370 个差 >150 px、80% 在 0–8 m)。两条共用 `gt.core.box2d_from_projection`(出框侧与 `rebox_line` 重算侧**唯一实现**)。
  - **判据侧读法**:`CornerProjection.all_inside` —— 拿 coverage/IoU 这类"框有没有包住物体"的判据打分时,裁断样本必须**单独分类**,既不能并入裁决(会得出永远红的判据,然后被调容差调绿),也不能装作没看见。
  - **归档重算走 `gt/refilter.py --clip2d`,不重采**(`label_2` 自带 `h w l x y z ry` + calib 的 `P2` 就够,画幅读 PNG 头)。**未裁断的行走短路、逐字节不变** —— 那 82% 的行本来没问题,按舍入过的字段重打只会注入 ~0.1 px 噪声。"只动该动的行"因此可检验。
  - ⚠️ **但实测它对 P1 的 2D 数字是零影响**(五个场景 + 遮挡四项**逐位不变**):每帧只有 2–3 条 GT、1–3 个预测,全局贪心匹配**饱和** —— 那 20 条里 6 条单个看是 FN→TP,却被"预测早已被邻车占用"抵消,**tp 向量逐位相同**。**别把它读成"这个修正没意义"**:口径本身是错的,且**判据侧**(如 `prop_eval` 的 coverage)吃到的就是这批框;换一个匹配不饱和的场景差就会露出来。旧口径物证 `kitti_ab_epic_gtfix_*` / `kitti_ab_occl2_*` 保留,**三代在 A/B 硬门槛下长得一样**(逐帧条数全等),引用必须写明是哪一代
- **★ 默认检测/分割后端已换成 SAM3(开放词表);它与 YOLO 的读数**不可比****(2026-10-02 用户裁决):`--backend {sam3,yolo}` 默认 `sam3`,`yolo` 是回退。**差别不是精度,是「类名从哪来」** —— yolo 的类名是**模型判的**,SAM3 的是**提示词给的**(`perception/backends.sam3_backend.DETECT_PROMPTS`;`Car` 必须含 `truck`/`bus`,否则**静默少检一类**)。⇒ SAM3 那侧的 AP 里「分类正确率」这一项退化成**「我的提示词写对没有」**,**跨后端比 AP 是把两件不同的事比大小**;引用任何数字(AP / PQ / mIoU)**必须写明后端**,`logs/*.json` 的 `highlights.backend` 是归属依据。⚠️ **SAM3 的 schema 与 YOLO 不同,去重和阈值都得重定**(2026-10-03):① **掩膜/框去重默认开**(一条提示一次前向 ⇒ 同一物体被 `car`/`truck` 各切一次,判据一对一匹配就把多的记 FP;实测 mask AP +0.063);② **`--conf` 默认 `None`,按后端解析**(`backends.DEFAULT_CONF`:sam3 **0.5** / yolo 0.25)——把 YOLO 的 0.25 套给 SAM3 会放进大量幻觉(`person` 单条提示在 5 帧里产出 53 个落在建筑/杆/Static 上的假人,掩膜下**没有一个** `Pedestrians` tag);③ **阈值是「按指标选」的操作点,不是模型的属性** —— PQ 涨到 0.7 而 mask AP 在 0.5 见顶,**报数必须带阈值**。修后 `inst_eval`:PQ 0.7198 / SQ 0.8998 / RQ 0.8000 / mask AP 0.8061(yolo 归档 0.6257 / 0.8447 / 0.7407 / 0.6341)。★ ★ **每条提示一次前向是硬约束**:SAM3 吃**单概念**提示,拼串实测返回 **0 个掩膜**(`"car person bicycle"`、`"car, person, bicycle"` 都试过)⇒ 多概念只能逐条前向再并集,代价线性(0.5–1.0 s/条)。P1 归档矩阵(0.669 / −0.578 …)是 **yolo 基线**
- **`weights/sam3/` 的权重在盘上,但本项目 env 跑不动**(2026-10-02):3.45 GB(`sam3.pt` + HF `model.safetensors`),**全仓 0 引用**;`autodrivedata` 里 `transformers`/`sam3` **都没装**,而 PyPI 不通(aliyun 403、pypi.org 超时;**只有 GitHub 通**)⇒ **2026-10-02 已装进本项目 env**(`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers`,只加 13 个包、**不动 torch/numpy**)。实测成本:加载 4.6 s / 3.15 GiB,单帧单提示 **0.5–1.0 s**。⚠️ **别把 SAM3 当「换个更强的检测器」** —— 它的类别是**提示词给的、不是模型判的**,接进 `eval_2d_ab` 会让 AP 里的分类正确率退化成「我提示词写的是不是 car」(循环论证,且系统性偏乐观)。它真正强的是**开放词表**:实测 `traffic cone` 17/17、`barrier` 30/30(中位 IoU 0.65/0.78)命中我们摆的道具 —— 而这两类**不在** KITTI 3 类 / COCO 类表里。见 Plan4 §P-V22
- **`Dynamic`(21) 是地图自带静态道具的 tag**(实测 3/3 资产排他),它**不在** P2-A 三个 GT 类里、此前也**不在** `EXCLUDED_TAGS` 里 ⇒ 落在一个**不被报出来的第三桶**("没报"与"没有"在下游长得一样)。现由 `sem_tags.PROP_TAGS` + `excluded_share` 的 `map_prop` 桶报出。**`Static`(20)/`Other`(22) 不许顺手并进来** —— 各有几千像素但**没有已知资产能归因**,并进去是把没验过的假设写死
- **sunset_glare 方位**:az=90=东=+x=车头正前(yaw=0 时);az=300 是顺光陷阱(太阳在车后)。判据 = 全图过曝最低(AE 压最狠)
- **GT 不许出零面积框**:车**擦过镜头**时(center depth 0.58–3.5 m)投影会退化成一个点或一条线(`x1==x2` / `y1==y2`)。它与任何预测的 IoU **恒为 0** ⇒ 白送一次漏检;实测 **P1 每份数据集 11/194(5.7%)**,⇒ **recall 天花板被压到 94.3%**、每份 AP 都带 ≈ −0.08 的折扣(修掉后 day_clear 从 0.594 回到 0.674)。**更阴的是它进出由亚帧抖动决定**:车的远底角恰落在像面下边缘(v≈375)时,0.24 m 的 ego 偏移就让整框在「0 px 高」与「164 px 高」之间翻面 ⇒ 两次采集的 GT **逐帧条数**不再相等,A/B 硬门槛当场破。判据 = [`gt/core.py`](autodrivedata/gt/core.py) 的 `MIN_BOX_SIDE_PX = 1.0`(两维都要 ≥ 1 px)。⚠️ **归档的 P1 矩阵整条带这个折扣**(Δ 不受影响,绝对值要按新口径读)
- **A/B 采集有启动瞬态,不止看"帧级配对"**:松制动后**首帧**可能只走 0.537 m(正常 0.801),之后正常 ⇒ 整段序列**恒定滞后 0.24 m**(≈0.3 帧),而它只在"车擦过镜头"那几帧上翻面。判据 = 两次采集的 `training/pose/` 逐帧位置差(`max|Δx|`;四次实测 0.017–0.07 m 算合格,0.24 m 就是它)
- **采集器清场 + 起点校验**:残留 actor 阻塞 spawn point 会致 fallback 反向出生点、轨迹失配(collect_ab_route/collect_static_gt 已内置,勿删)。**遮挡道具也要清**(`static.prop.*` 不在 vehicle/walker/controller 三类里)——残留的墙会让下一轮 A 侧带着上一轮 B 侧的墙采完,**数据照出,只是 A/B 的差凭空小一截**;清场与收尾共用 `collect_ab_route.is_cleanup_target` 一个谓词
- **锚定 yaw 用 spawn point 固有 rotation**:collect_static_gt 曾硬编码 yaw=0,Town10HD_Opt pts[0] 固有 yaw=0.16° 恰好成立、Town13(125.9°)车道线采样走到车后 overlay 全空;已改用 pts[0].rotation(沿车道),**collect_ab_route 的 yaw=0 是 P1 已验证基线勿动**。⚠️ **"沿本车道采样"只在 ego 还在车道上时成立** —— 采集器是**定速直行不跟车道**,而实测车道在前 35 m 内就拐了 90°(yaw −0.57°→−90.16°)⇒ 采样落到侧向 27+ m。GT 值**是对的**(逐点与地图查询吻合),但那是**另一条车道**;见 Plan4 §P-V21 五 / §2 待决项 #7(改行驶方式会动全部已归档 `kitti_static_*`,故未做)
- **静态 GT 的判据形状必须是「锚定 + 邻域」**(§P-V21):`static_gt` 的两个量都有"点投影必错"的几何原因 —— **信号锚点是地面点**(要沿锚点向上搜 6 m,点投影 40/40 落在 `Roads`),**车道线只有 ~2 px 宽**(单像素命中 21.7%/27.8%,±4 px 窗 → 99.6%)。⚠️ 邻域大小是**定标的**:信号窗**放宽到 8 m 反而更差**(横移对照 0.000 → 0.300),别"顺手放宽";⚠️ 判据的判别力靠**对照**(横移 + 同 v 随机),`real` 单看会被"这场景本就到处都是标线"骗过
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
- **★ `.gitignore` 的否定规则**含斜杠 ⇒ 被**锚定在仓库根**(2026-10-06 实测):`!assets/**` 只放行**顶层** `assets/`,放行不到 `docs/assets/` ⇒ **20 张证据图从来没进过版本库**(`git add` 不报错、直接跳过;判据 = `git log -- docs/assets` 为空、`git check-ignore -q` 退出码 0)。已补 `!docs/assets/` + `!docs/assets/**`,并在其后**再挡一次** `docs/assets/**/.ipynb_checkpoints/`(**顺序要紧** —— 那个 `**` 会把检查点一起放行)。⇒ **判据是 `git add -n <路径>` 真的打印 `add '...'`,不是「我加了规则」**。同族:上面那条 `ruff format` 会走进未跟踪目录
- **★ 子进程/CLI 入口缺 `__main__` 守卫 ⇒ `python -m` 进去什么都不做、退出码 0**(2026-10-06 首现、**2026-10-07 第二次现形**)。比崩溃恶劣:批处理里会被当成"跑过了",而产物目录是空的。★ 第二次的教训是**判据的覆盖范围**:当时已经有一条 `test_edit_entrypoints.py`(理由写的正是这条),但它**只扫 `autodrivedata/edit/`** ⇒ `gs/eval_edit.py` 从来没被扫过。现已扩成全包(`tests/test_entrypoints.py`,88 个带 `main()` 的模块,带"扫到 0 个就恒绿"的自证)。★ 这是「**判据的覆盖范围本身也是判据的一部分**」的**第三次**现形(前两次:只扫 3 个 `.md` 漏了包内 docstring 链接;只扫 3 份文档漏了 edit 计划)
- 提交:Conventional Commits;**提交信息不附 AI 署名**(不加 `Co-Authored-By: Claude` 等 trailer);改动后 `ruff check && ruff format` + 相关单测;决策与执行记录同步进 **Plan4.md**(教程能力线另同步 docs/milestone2.md);**Plan.md 与 Plan2.md 均已冻结**
- **MapQR 变体有两处「不报错的错」,改这块先跑 `tests/map/test_{deform_attn,bevenc}.py`**:
  ① 官方整套数学吃**归一化 `[0,1]`**,本项目坐标是**米** —— 米直接进 `sine_pos_embed` **不抛异常**,
  只让正弦频率**别名**、位置嵌入退化成噪声,**症状是「训不动」而不是报错**;
  ② BEV 锚点投影必须把 **ego→world** 折进 4×4(`p_c = C2K·R_sᵀ·(p_ego − t_s)`),
  漏掉它**只有在 ego 位姿非恒等时才错** —— 恒等位姿下两版**数值上恰好同解**,
  玩具夹具因此看不出来(本项目同类坑:"yaw≈0 的相机看着正常")。两条都有回归钉。
- **`--bev-chunk` 不是省显存的手段**:实测分块**不降反升**(23.6 → 29.5 GiB),要压显存调**层数**(每层 ≈ +3.6 GiB @bs2)。见 Plan2 §P-M.14
- **格式口径已定死**:`[tool.ruff]` 在 pyproject(line-length 110 / select E,F,I,UP,B / ignore E501,E741),`ruff format` 是唯一 formatter;批量纯格式提交要追加到 `.git-blame-ignore-revs`
- **裸敲 `ruff format` 会走进未跟踪且未被 `.gitignore` 覆盖的目录**:ruff 0.16 **会格式化 Markdown 里的 python 代码块** ⇒ 它对 `weights/`(只有 `*.pt` 被忽略、目录本身没忽略)下手,把**下载来的第三方 README 改了**(2026-09-28 实测:`weights/sam3/README.md`)。已在 pyproject 用 `extend-exclude = ["weights/"]` 堵住。**新开未跟踪目录放外部内容时,同步加进这个 exclude** —— 其余产物目录靠 `.gitignore` 兜住(ruff 默认 `respect-gitignore`)

> **搬目录/改结构时注意** —— 本轮重构实测出 **12 类引用形态**,照单扫一遍再动手(清单与各类实例见
> [docs/refactor-2026-09.md](docs/refactor-2026-09.md) 的阶段 4 记录):`import X` / `from X import` / `import X as Y` /
> `from <包> import X` / `from autodrivedata.<m> import` / 路径字符串 / 硬编码源码路径常量 /
> **算自己位置的表达式**(`Path(__file__).parents[N]`、`cd "$(dirname "$0")/.."`)/ **方法体内的惰性 import**(逃过 `--collect-only`)/
> **同名多义**(包名 vs 输出目录 vs env 名)/ **`包/模块.属性` 散文写法** /
> **★ 下游仓库的引用**(2026-09-30 补 —— 前 11 类全是**本仓内**的,这一类漏了整整 4 天:
> AutoLabel 的 `test_mapvec_crosscheck.py` 指着重构前的 `autodrivedata.chamfer_ap` 等旧路径,
> 自 09-26 起 **collect 期就 ERROR**,那是 `mapvec_pred/1` 契约**唯一**的漂移保护网)。
> 机械判据:[`autodrivedata/tests/test_downstream_refs.py`](autodrivedata/tests/test_downstream_refs.py)
> —— AST 扫下游 `.py` 的 `autodrivedata.*` 引用并逐个 `find_spec`(下游不在本机则 skip,带自证)。
> ⚠️ 该判据**只查 import,不查路径字符串**(如下游 config 里写死的 `.../AutoDriveData/outputs/kitti_ft/`)——
> 那属数据可用性,写进单测会让测试依赖数据集在不在盘上。**别以为那里绿了就代表全部。**
> 改完**必须真跑全量**并对用例数 —— 计数对账抓不到惰性 import,只有真跑能抓到。

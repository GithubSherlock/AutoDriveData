# Plan3 —— 主线之外:外部数据集对照与方向评估

> **本文档定位**:[Plan.md](Plan.md) 已**冻结**(定案与终止记录),[Plan2.md](Plan2.md) 是**当前主线**的计划制定地。
> 本文档收纳**主线之外**的评估:外部公开数据集的可用性判定、新方向可行性、以及"为什么不直接用现成数据"的外部依据。
>
> **写法**(与 [docs/ros2-feasibility.md](docs/ros2-feasibility.md) 同一范式):每条结论必须附
> **可复核来源**(arXiv 号 / 原文引用 / 具体数字),并给出 **触发重评条件** —— 没有触发条件的"不采用"等于没结论。
>
> **免责**:本文档结论**不含盘容量约束**(用户 2026-09-30 明确:盘可扩容,只论实用性)。

---

## 1 nuCarla(arXiv:2511.13744)

**结论:不下载。与本项目主线结构性不匹配 —— 不是成本问题,是"它没有我们要的东西"。**
但**值得作为工业界对照保留** —— 它把本项目的 novelty 边界划得很清楚(§1.3)。

### 1.1 事实(全部来自官方 README / 论文,可复核)

| 项 | 事实 |
|---|---|
| 出处 | Qiao / Cao / Liu(University of Michigan,Michigan Traffic Lab);arXiv **2511.13744**,2025-11-12 |
| 仓库 | [michigan-traffic-lab/nuCarla](https://github.com/michigan-traffic-lab/nuCarla);数据在 [HF `zhijieq/nuCarla`](https://huggingface.co/datasets/zhijieq/nuCarla) |
| 自我定位 | *"the first large-scale CARLA-based perception dataset with full compatibility to the nuScenes format"* |
| 规模 | **1000 场景**(700 train / 150 val / 150 test,照 nuScenes 划分)× **40 帧 @0.5 s** = 40,000 帧 |
| 地理 | 9 张图:Town01–Town07、Town10、Mcity 数字孪生;**Town10 与 Mcity 留作 unseen 测试** |
| 天气 | 14 种(sunny / cloudy / rainy × noon / sunset),**逐场景随机施加**,分布大致均匀 |
| 类别 | 6 类:car / truck / bus / pedestrian / motorcycle / bicycle(放弃 construction vehicle / trailer / barrier / traffic cone 四类) |
| 标注量 | 459,632 个实例标注(对照 nuScenes 的 417,609) |
| 传感器 | **纯相机**。原文:*"This is a camera-based perception dataset. The LiDAR files in the sample folder are **dummy placeholders** provided solely for compatibility with the MMDetection3D framework conventions."* |
| 目录 | `maps/ · samples/ · v1.0-mini/ · v1.0-test/ · v1.0-trainval/`(**`maps/` 内是 png 还是 expansion json 未核实**) |
| 验证 | BEVFormer 0.813 / 0.778,BEVDet 0.811 / 0.753,FastBEV 0.777 / 0.728,PETR 0.745 / 0.710(nuScenes 官方检测指标 mAP / NDS) |
| 算力成本 | BEVFormer Base 24 ep:**30500 M 显存 / 300 GPU-hours**;PETR 9500 M / 150 h;BEVDet 8500 M / 300 h;FastBEV 14000 M / 50 h |
| 附带 | 预训练权重 + 训练日志(releases/v1.0);MMDetection3D-1.0 升级到 PyTorch 2.7+ / CUDA 12.8;2026-01-02 补发 `data` 分支的生成 pipeline |

> ⚠️ **口径自查**:其 Model Zoo 表头自述 *"All metrics are post-computed based on the six available classes and **are not the direct output from the nuScenes console**"*。
> 与本项目「单独报一个 mAP 而不写阈值 = 无效结论」(CLAUDE.md 红线)是同一问题的两种处理 —— **它宽口径免责,本项目窄口径自证**。引用其数字时必须带上这条限定。

### 1.2 结构性不匹配(逐条对主线)

| nuCarla | 本项目主线 | 判定 |
|---|---|---|
| LiDAR 是 dummy 占位符 | P1 结论「**LiDAR 不受天气光照**」;`gt/export/nuscenes.py` 的雷达 offset 逐分量对齐 | ❌ **多传感器方向是空的** |
| 只有 6 类 3D 检测框 | MapTR map 矢量(§P-M.12)、P2 静态 GT(landmark / lane_marking)、灯态时序层(§5.9) | ❌ 标注体系**无交集** |
| 14 天气**随机均匀分布** | A/B 帧级配对 = 同锚点 + 同静置车布局 + **只变 weather** | ❌ 随机分布**无法配对** ⇒ 做不了定量归因 |
| 官方 Town01–07/10 + Mcity | Town10HD_Opt / Town13 + 自定义地图与场景 | ⚠️ 地图集不重合 |
| 40,000 帧 | 自采 200–800 帧级 | ✅ **其唯一硬优势** |
| 300 GPU-h / 模型 | 单卡(机型会漂) | ⚠️ ≈12.5 天连续满载;且跨机型 AP 不可比(CLAUDE.md 红线) |

### 1.3 战略含义(真正的价值)

它占了「**大规模 + CARLA + nuScenes 格式**」这个位置(2025-11)。本项目在**规模**这条轴上拼不过它
(40,000 帧 + 4 个 SOTA 模型 + 预训练权重),所以**必须靠它没有的东西立足** —— 而这恰好就是 P1/P2 的全部内容:

1. **多传感器真值**(LiDAR / 雷达,而非占位符)
2. **受控 A/B**(可定量归因,而非随机天气)
3. **地图矢量 + 静态 GT + 灯态时序层**
4. **自定义地图 / 场景的采集能力**

⇒ 它是「**为什么不直接下载现成数据**」的外部依据,也是对外汇报/投稿时的 related work 锚点。

### 1.4 触发重评条件

任一成立才重新评估(否则不必再查):

- 本项目要做 **相机 BEV 3D 检测** —— 当前不做,主线是 map 矢量 + 多传感器;
- 要做 **sim2real / 域适应**,需要合成侧的大规模配对数据
  (`nuCarla × nuScenes` 两边同为 nuScenes 格式,转换成本近零,是天然实验床);
- nuCarla 后续版本**补上真实 LiDAR 或地图矢量标注**。

# 时序拼接实测(2026-09-29)

> **本文档的定位**:把「逐帧 BEV 矢量预测 → 全局矢量图」这条链**第一次真跑通**的记录 ——
> 含**两个位姿源**(CARLA 真值 / SLAM 估计)的对比、失败的**融合口径**分析、以及为此新建的
> 双传感器采集器。工具本体见 [`map/stitch_temporal.py`](../autodrivedata/map/stitch_temporal.py)。

---

## 0 结论先行

| 问题 | 答案 |
|---|---|
| 逐帧预测能拼成全局矢量图吗 | **能**。JSON + PNG 双产物,按 `seg` 独立拼(段缝位移 56.8–109.6 m,跨段直拼是废的) |
| 位姿源换真值 vs SLAM,图有差别吗 | **这张图上看不出**。SLAM ATE **0.877 m**(相对 0.671%)**小于**模型自身逐帧抖动(2–4 m)⇒ **位姿不是瓶颈,感知才是** |
| 直接叠图够用吗 | **不够**。融合口径选错时合掉的比例差 **60×**(见 §3) |
| 数据从哪来 | 原有数据集**没有任何一份同时有环视相机与 LiDAR** ⇒ 新建 `collect_surround_lidar.py`(6 相机 + LiDAR + 5 雷达,一次采集喂三条链) |

## 1 双传感器采集(`collect_surround_lidar.py`,首次真跑)

`outputs/showcase/07_dual/seq` — Town10HD_Opt,34 帧,定速 8 m/s,**逐帧自证中位 = 均值 = 7.92 m/s**。

| 布局 | 内容 | 消费方 |
|---|---|---|
| `cam_*/` + `calib.json` + `ego_pose.json` | 6 视角 1600×900(逐通道 fov) | `assemble_maptr` |
| `training/velodyne/` + `training/pose/` | KITTI 口径点云 + 12 数位姿 | `slam_odometry` / `eval_slam` |
| `samples/RADAR_*/` | 5 通道 devkit 18 字段 pcd(**282–319 点/帧,非空**) | AutoLabel 雷达线 |

**首跑抓到的两个问题**(都已修):

1. **`finally` 漏销毁雷达** ⇒ 5 个 radar actor 留在世界里,收尾 core dump。数据当时已完整落盘,
   但"崩了"会让人误读成采集失败 —— 与 `tools/carla_server.sh` 头注记的 teardown segfault 同款现象。
2. **定速自证抓到中途停摆**:第一次采 50 帧,自证报"中位 7.92 / **均值 5.49**",逐帧步长显示
   **帧 34 起被挡停**(≤0.4 m/帧,后 16 帧全静止)。改采 34 帧后中位 = 均值 = 7.92。
   —— 这正是红线「定速必须逐帧自证、不能只看命令值」的价值:**只看命令值这 16 帧是隐形的**。

## 2 两个位姿源的拼接对比

同一段数据、同一份逐帧预测(`maptr_v2_singleF.pt` @`--score-thr 0.2`),只换位姿源:

| 位姿源 | 实例 | 去重后 | 包络(米) |
|---|---|---|---|
| CARLA 真值 `ego2global`(**上界**) | 1521 | 1455 | **162.4 × 43.4** |
| SLAM `traj_pgo`(**车载可做**) | 1521 | 1457 | **161.7 × 43.8** |

`eval_slam` 在同一序列上:**ATE(对齐)0.877 m / 相对 0.671%**(GT 路径长 130.65 m)。

⇒ **两者拼出的图肉眼无差别**。原因是尺度对比:**位姿误差 0.88 m ≪ 模型逐帧抖动 2–4 m**
(后者是 §3 量出来的)—— 位姿源在当前感知精度下**不是瓶颈**。这条把"要不要先做更好的定位"
这个问题**定量地否掉**了:先提感知。

> ⚠️ **本次 SLAM 轨迹无回环**(直线段,`n_loops=0` 是合法结果,closure pre=post=118.5 m 未变)。
> 所以这里比的是**纯里程计精度**下的拼接,不含回环增益。真回环场景见 `kitti_loop` 链。

## 3 融合口径:直接叠图不够用,且**判据选错会差 60×**

`stitch_temporal` 支持三种融合:

| 口径 | 18145 实例 → | 说明 |
|---|---|---|
| `overlay` 逐帧叠画 | 18145 | 只是把 300 帧画在一起,**不是地图** |
| `dedup`(复用 `stitch._dedup`,Hausdorff) | 18021(**只合掉 0.7%**) | **对时序是错配** |
| `cluster`(Chamfer + 朝向门 + 簇内平均) | 14176(合掉 22%) | 时序融合 |

**为什么 `dedup` 不行**:它的 `_polyline_gap` 是 **Hausdorff(最坏点主导)**—— 对"同一段线被导了两遍"
(几何近乎逐位相同)合适;而跨帧比的是"同一段路被不同视角各测一遍",折线两端与采样相位本就有差,
**一对最坏点就把整条判成不同**。容差扫描证实:

| tol | 簇数 | 平均支持帧数 |
|---|---|---|
| 0.5 m | 14176 | 1.4 |
| 1.0 m | 7364 | 2.5 |
| 2.0 m | 3807 | 4.8 |
| 4.0 m | 1770 | 10.3 |

期望支持 ≈ **18 帧**(60 m 窗口 ÷ 3.33 m 步长),而 **4.0 m 已超过车道宽 3.5 m**(开始误合并相邻车道)。
⇒ **逐帧预测的几何抖动 ≈2–4 m,比车道还宽**;在物理合理的容差下聚类无法收敛。
**这不是工具参数问题,是模型单帧几何精度的直接体现** —— 也正是时序融合(阶段 A)要改善的东西。

## 4 复现命令

```bash
bash tools/carla_server.sh start
python -m autodrivedata.sim.collect_surround_lidar --out outputs/showcase/07_dual/seq --frames 34 --stride 5 --speed 8
python -m autodrivedata.map.assemble_maptr --surround outputs/showcase/07_dual/seq \
    --map-json training/map/Town10HD_Opt_full.json --out outputs/showcase/07_dual/seq/map_infos.json
python -m autodrivedata.map.eval_maptr --infos outputs/showcase/07_dual/seq/map_infos.json \
    --root outputs/showcase/07_dual/seq --ckpt outputs/maptr_v2_singleF.pt --score-thr 0.2 \
    --out-frames outputs/showcase/07_dual/preds
python -m autodrivedata.slam.slam_odometry --root outputs/showcase/07_dual/seq --frames 0-33 --out outputs/showcase/07_dual/slam
python -m autodrivedata.slam.slam_backend  --traj outputs/showcase/07_dual/slam/traj_raw.json \
    --root outputs/showcase/07_dual/seq --out outputs/showcase/07_dual/slam
python -m autodrivedata.map.stitch_temporal --frames outputs/showcase/07_dual/preds \
    --infos outputs/showcase/07_dual/seq/map_infos.json --pose gt   --out outputs/showcase/07_dual/map_gt
python -m autodrivedata.map.stitch_temporal --frames outputs/showcase/07_dual/preds \
    --infos outputs/showcase/07_dual/seq/map_infos.json --pose slam \
    --slam-traj outputs/showcase/07_dual/slam/traj_pgo.json --out outputs/showcase/07_dual/map_slam
```

产物:`map_gt.{json,png}` / `map_slam.{json,png}` / `pose_gt_vs_slam.png`(并排)。

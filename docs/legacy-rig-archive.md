# legacy rig 归档:镜像 bug 的起因、证据与处置

> **本文档的定位**:`legacy` rig **已于 2026-09-28 由用户裁决从代码中完全移除**(见 [docs/acceptance-2026-09-28.md](acceptance-2026-09-28.md) §4.11)。
> 移除的代价是:**「镜像 bug 已修」这件事以后无法在仓内重跑复现** —— 那份证据依赖 `legacy` 口径本身。
> 故把**成因 + 数字 + 图**冻结在这里。图在 [`assets/legacy-rig/`](../assets/legacy-rig/)。

---

## 1 它是什么

CARLA 环视采集最早期的一套相机挂点口径(6 路**共用** `SENSOR_OFFSET`(1.2, 0, 1.65)、
仅偏航、BACK_LEFT/BACK_RIGHT 用 235/125)。它**不是"旧版本所以不要用"**,而是:

> **`maptr_ep256.pt` / `maptr_ep512.pt` 这两个权重就是在这套 rig 上训出来的,
> 拿后来的 nuscenes 口径喂它们反而是错配。**

所以它长期被保留(不是无条件 bug)。**移除是因为它服务的那两个权重已废弃**(2026-09-22 起
全部 MapTR 权重标注废弃),加上运行时按权重文件名自动选 rig 的机制(`resolve_rig` +
`LEGACY_CKPTS`)会让"当前到底用哪套口径"变得不可一眼读出 —— 收益归零,成本仍在。

**当前唯一在用的环视口径 = `nuscenes`**(官方 nuScenes `calibrated_sensor` → CARLA 口径),
另有 `wide`(挂点后移 + 新 FoV)。两者都在 [`autodrivedata/calib/camera_rig.py`](../autodrivedata/calib/camera_rig.py)。

## 2 镜像 bug(移除的代码里最值得留下的一条)

早期实现把官方**方位角**原样抄成正数,漏了 CARLA ↔ nuScenes 的 y 符号翻转
(`yaw_carla = −az_nus`)⇒ **四个侧/后相机左右镜像**。前/后相机因近自逆而"看起来对",
所以长期没暴露;同时 pitch/roll 被硬编码 0(官方实测 |pitch| 最大 0.96°)。

**逐相机读数**(修正值由官方四元数现算,历史值 = 移除前的 `collect_surround.SURROUND_CAMS` 字面量):

| 相机 | 官方 az_nus | 修正 yaw_carla = −az_nus | 历史字面值 | 原始差(不 wrap) |
|---|---|---|---|---|
| CAM_FRONT | +0.321° | −0.321° | 0.0 | 0.3° |
| CAM_FRONT_LEFT | +55.165° | **−55.165°** | **+55.0** | **110.2°** |
| CAM_FRONT_RIGHT | −56.402° | **+56.402°** | **−55.0** | **111.4°** |
| CAM_BACK | +179.855° | −179.855° | 180.0 | 359.9° |
| CAM_BACK_LEFT | +108.595° | **−108.595°** | **+108.6** | **217.2°** |
| CAM_BACK_RIGHT | −110.789° | **+110.789°** | **−110.8** | **221.6°** |

⚠️ **"不 wrap"是刻意的**:217.2° 折成 142.8° 就看不出"镜像"了(Plan2 §P-M 记过)。

**为什么这个 bug 难查**:① 图像照常渲染、GT 框照常画出(两处都在同一套错口径下自洽);
② 前后相机看起来完全正常;③ 它毁掉的是"第 i 路图 ↔ 它学过的语义"的对应关系,
**只有在下游模型上才表现为精度下降**,而不是任何形式的报错。

## 3 证据(图)

| 图 | 内容 | 现在还能说明什么 |
|---|---|---|
| [`mirror-polar.png`](../assets/legacy-rig/mirror-polar.png) | **几何裁决**:官方方位角 → 修正 yaw 的极坐标图 + 历史字面值对比。实线=修正,淡线=历史,**四个侧/后相机上两线张开 = 镜像**;前/后两线重合 = 它长期没暴露的原因 | 一图看懂镜像的**方位** |
| [`mirror-ab-views.png`](../assets/legacy-rig/mirror-ab-views.png) | **同一 ego、同一批锥体**,只变 rig 口径(nuscenes 在上 / legacy 在下) | 世界的**左**方锥只该出现在 `*_LEFT`;历史版把它放进 `*_RIGHT` |
| [`layout-legacy.png`](../assets/legacy-rig/layout-legacy.png) / [`layout-official.png`](../assets/legacy-rig/layout-official.png) | 同镜头、同起点、同一份地图 GT 在两代布局上的**逐相机投影覆盖** | 后相机布局差异(实测 legacy BL 45 段 / official BL 56 段) |

## 4 覆盖表(两代口径当时的读数,已冻结)

| rig | 覆盖 | 盲区合计 | 逐段盲区 |
|---|---|---|---|
| `nuscenes`(官方) | **100.00%**(360.0°) | **0.000°** | 无 |
| `wide` | **95.79%**(344.844°) | **15.156°** | 7.3353° + 6.0984° + 1.7224° |

`wide` 的三个盲区**逐项等于设计预算** —— 这是当时的判据(`tests/calib/test_rigviz.py`
至今仍在钉这两个数,与 legacy 无关,故未随移除而失效)。

## 5 现在怎么再验"环视标定是对的"

镜像 bug 的**回归锁没有随 legacy 一起消失**,它移到了更硬的判据上:

| 判据 | 位置 | 钉什么 |
|---|---|---|
| `viz_rig_check --live` 六视角实拍 | `calib/viz_rig_check.py` | 画幅内**自身车体像素 = 0**(镜像会让后路拍到自己) |
| `verify_nus_calib` 十条 | `calib/verify_nus_calib.py` | ①实挂 vs 声明 ②雷达 ③雷达 FOV 占比 ④LiDAR 点数 ⑤内参 ⑥渲染 FoV ⑦车体像素 ⑧相邻共视 ⑨世界位姿链 ⑩ego 原点复测 |
| `probe_calib` A0–A6 | `calib/probe_calib.py` | 把 CARLA 渲染器当第二把尺子,数值裁决 (K, 外参) |
| `test_layer_guard` / `test_rigviz` | `tests/` | 覆盖表逐项等于设计预算;方位角两套独立实现必须同值 |

**结论**:镜像 bug 当年的**直接对照**没法再跑,但**同一失效模式**被上述判据覆盖
(它们本来就是为"声明 ≠ 渲染"这类静默错位设计的,而且不依赖 legacy 口径存在)。

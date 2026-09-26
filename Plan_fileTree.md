# 目录结构:定案与执行计划

> **本文档的定位**:项目**目录结构的制定地 + 执行契约** —— 怎么切、为什么这么切、按什么顺序做、做完怎么验收。
> 与 [docs/fileTree.md](docs/fileTree.md) 的分工:**本文件是决策与计划**,fileTree.md 是**现状索引**(某文件现在在哪)。
> **改结构先读本文件**;每个阶段完成后回来勾掉,并同步 fileTree.md。
>
> **依据来源**:2026-09-26 两轮全仓审计(49 + 32 个 agent,含逐提案对抗对账)。**本文所有数字均为实测**,
> 引用点计数口径 = git 跟踪文件里含该路径 token 的行数(代码 import + 命令行串 + 文档 + shell)。
>
> **维护约定**:§6「已实测否决」**只增不删** —— 那是最省时间的部分,它记的是「这个问题已经量过了,别再提一遍」。

---

## 1 已裁决的方向(2026-09-26)

| 决策 | 内容 | 依据 |
|---|---|---|
| **划分主轴** | 一级目录 = **能力面**;生命周期由**文件名前缀**承载(`collect_` / `eval_` / `train_` / `probe_` / `viz_` / `assemble_`) | 两个维度只能有一个做目录。**能力做目录零信息损失**(生命周期已在文件名里);反过来会丢(`eval_kitti.py` / `eval_attr.py` 从名字看不出能力面) |
| **`sim/` 取代 `CARLA/`** | 小写;不叫 `carla/` | 实测 `import carla` 在子包内**不会**被遮蔽(Python 3 绝对导入,已实证)。但 `import carla` 与 `from ..carla.carla_common import` 同文件共存读法灾难。这层装的是**我们的仿真交互层**(`carla_common.py` 815 行、`live_common.py` 815 行 + 12 单测),不是 CARLA 平台 |
| **`map/` 取代 `HDMap/`** | | `HDMap` 在本仓已被三样别的东西占用(`hdMapGitHub/` 上游克隆、CARLA 装目录的 `HDMaps/` 见 `Plan.md:330`、`Plan2.md` 的 online HD mapping);地图域已有三套在用名字(`opendrive` / `mapvec` / `maptr`),再建是第四个 |
| **不建 `configs/`** | 等真有了 yaml 再建 | 全仓手写 yaml = **0 个**(`pyproject.toml` 必须在根,`lightning_logs/*/hparams.yaml` 是 Lightning 自动生成)。最像配置的 `scenarios.py` / `camera_rig.py` 是**冻结口径**不是可调参数,放进去会误导 |
| **`utils/` 建** | 通用数学/基础设施 | 用户指定。**准入判据**:无项目领域语义、无 carla/torch 依赖。首批 = `geometry.py` `paths.py` `fonts.py`(3 个)。**若超过 6 个文件说明它正在变成 junk drawer,需重新裁决** |
| **`bin/` + `tests/` 全收进 `autodrivedata/`** | | 两者**必须一起搬**:只搬一半会得到最坏结果(测试进包但被测脚本留在 `bin/` ⇒ 两边都要 `sys.path` hack,比现状更差) |
| **`maptr_impl/` / `maptr_official/` 收进 `map/`** | → `map/maptr/` `map/maptr_official/` | 用户要求 |
| **顶层 `tools/`** | 开放性 bash/python,判据 = **不含本项目领域知识** | `carla_server.sh` / `clear_cache.sh` / `gitpush.sh` / `gpu_fix/`。含领域知识的编排脚本(`assemble_and_merge.sh` 等)跟着它的 Python 走 |
| **入口调用方式改为 `python -m`** | `python -m autodrivedata.sim.collect_drive` | 搬进包后 `sys.path[0] = 脚本目录` 的旧机制失效。这也是我们要的 —— 那正是 `bin/` 里 29 处同级裸名 import 的成因 |

---

## 2 目标树(完整落位,共 106 个模块 + 48 个测试)

```
autodrivedata/
├── sim/                  22  CARLA 仿真交互层 + 全部采集器
│   ├── carla_common.py  live_common.py  live_studio.py  view_stream.py
│   ├── drive_ego.py     smoke.py        probe_vulkan.py collect_rig.py
│   ├── probe_imu.py     scenarios.py    collect_drive.py  collect_kitti.py
│   ├── collect_ab_route.py  collect_nus.py  collect_surround.py
│   ├── collect_surround_micro.py  collect_static_gt.py  collect_tl_states.py
│   ├── collect_slam.py  collect_traj.py  collect_stereo.py  collect_3dgs.py
│   └── smoke_radar_collect.sh
├── calib/                14  标定(自证 / 实时监看 / 配置图)
│   ├── core.py          ← **原 calib.py 改名**(`calib/calib.py` 会自反,见阶段 3 记录)
│   ├── camera_rig.py calib_probe.py calib_live.py depth_codec.py rigviz.py
│   ├── probe_calib.py verify_nus_calib.py rig_check.py probe_rig_mount.py
│   └── viz_calib_check.py viz_rig_check.py calib_multilidar.py viz_layout_cmp.py
├── map/                  29  地图矢量 + MapTR
│   ├── opendrive.py mapvec.py mapvec_schema.py mapviz.py chamfer_ap.py
│   ├── assemble_maptr.py convert_mapvec.py export_mapvec.py merge_train_infos.py
│   ├── train_maptr.py eval_maptr.py eval_official_metric.py
│   ├── prepare_official_dataset.py probe_mapvec_oracle.py probe_mapvec_proj.py
│   ├── viz_maptr_pred.py
│   ├── assemble_and_merge.sh  finalize_maptr_600.sh
│   ├── maptr/             8   ← maptr_impl/ 整体(7 torch + __init__)
│   └── maptr_official/    5   ← maptr_official/ 整体(__init__ + bridge + configs/3,已终止线)
├── slam/                 11
│   ├── core.py          ← **原 slam.py 改名**(与 `calib/core.py` 同款,避免 `slam.slam`)
│   ├── slam_eval.py live_slam.py accum.py
│   ├── slam_odometry.py slam_backend.py slam_diff_test.py eval_slam.py
│   └── build_accum_map.py probe_scan_to_map.py slam_cpp.cpp
├── perception/           17  检测 / 单双目 / 雷达 / 语义 / 点云(不 import carla)
│   ├── mono_depth.py stereo.py multilidar.py radar.py semantic.py
│   ├── compare.py attribution.py ground.py cluster.py
│   ├── mono_distance.py sem_bev.py eval_2d_ab.py eval_attr.py eval_kitti.py
│   └── finetune_synth.py extract_ground.py cluster_obstacles.py
│   ↑ **`probe_radar_l3.py` 已改归 `sim/`**(它 import carla,与 `probe_imu` 同类;见阶段 6 记录)
├── gt/                    6   GT 生成
│   ├── gt.py static_gt.py traffic_light.py
│   └── export/{__init__.py, kitti.py, nuscenes.py}
├── traj/                  2   assemble_traj_pt.py  convert_hivt_pt.py
├── gs/                    1   train_3dgs_mini.py
├── utils/                 3   geometry.py  paths.py  fonts.py
└── tests/                48   按能力镜像(与各子树同阶段搬迁)
    ├── sim/          test_live_common.py  test_scenarios.py
    ├── calib/        test_calib.py test_calib_probe.py test_calib_live.py
    │                 test_rigviz.py test_probe_calib.py test_depth_codec.py
    │                 test_collect_rig.py test_nuscenes_calib_consistency.py
    │                 test_nuscenes_cali_sensors.py test_calib_oracle_autolabel.py
    ├── map/          test_opendrive.py test_mapvec.py test_mapvec_schema.py
    │                 test_mapviz.py test_chamfer_ap.py test_chamfer_gpu.py
    │                 test_gkt.py test_head.py test_temporal.py test_device.py
    │                 test_maptr_select.py
    ├── slam/         test_slam.py test_slam_eval.py test_live_slam.py test_accum.py
    ├── perception/   test_mono_depth.py test_stereo.py test_multilidar.py
    │                 test_radar.py test_semantic.py test_compare.py
    │                 test_attribution.py test_ground.py test_cluster.py
    ├── gt/           test_gt.py test_static_gt.py test_traffic_light.py
    │                 test_export_kitti.py test_export_nuscenes.py
    │                 test_gt_oracle_autolabel.py test_nuscenes_oracle_autolabel.py
    └── utils/        test_geometry.py test_geometry_nus.py test_paths.py
                      test_fonts.py test_geometry_carla_oracle.py

AutoDriveData/              顶层保留
├── tools/                开放性工具(不含本项目领域知识)
│   ├── carla_server.sh  clear_cache.sh  gitpush.sh
│   └── gpu_fix/{install.sh, mhookshim.c}
├── models/  assets/  logs/    资源类(用户方针,见 §5.4/§5.5 的 gitignore 前置)
├── docs/  README.md  CLAUDE.md  Plan.md  Plan2.md  Plan_fileTree.md
├── pyproject.toml  requirements.txt  .envrc  .gitignore
└── outputs/ training/ lightning_logs/ hdMapGitHub/ auto3dlabel/   (产物/上游,【未入库】)
```

**依赖方向(升级后的硬纪律)**:`tests/` → 一切;`sim/` `map/` `perception/` → `utils/` `gt/` `calib/` `slam/`;
**`utils/` `gt/` `slam/` 不许依赖任何兄弟能力包,也不许 import carla/torch**。

---

## 3 守卫的新形状(阶段 1 的产物)

### 3.1 现状与它为什么必须改

`tests/test_paths.py:42-61` 用 `pkg.rglob("*.py")` **递归**强制「整包不 import carla/torch」。
`sim/` 一建,27 个 `import carla` 的文件立刻让它变红 —— **代码一个字没坏,守卫红了**。

两个已实测的洞:

- **马甲库**:`hit = mods & {"carla", "torch"}` 按模块名字面匹配。`from ultralytics import YOLO` 实测会拉起 torch
  (`import ultralytics` 后 `'torch' in sys.modules == True`),但字面上不含 `torch` ⇒ **静默溜过**。
  受影响:`eval_2d_ab.py` `eval_attr.py`(ultralytics)、`finetune_synth.py`(auto3dlabel→函数体内延迟 `mmdet3d`)。
- **相对导入**:`test_paths.py:54` 只统计 `node.level == 0` 的 `ImportFrom` ⇒ `from . import x` 完全绕过。

### 3.2 新守卫(替换 `test_package_stays_pure_value`,约 45 行)

```python
# 目录(相对 autodrivedata/;注解用) → 【禁止】import 的三方模块。最深匹配优先。
LAYER_RULES: dict[str, set[str]] = {
    "": {"carla", "torch", "ultralytics", "mmdet3d", "mmcv", "lightning"},  # 包根:迁移期残留模块
    "utils": {"carla", "torch", "ultralytics", "mmdet3d", "mmcv", "lightning"},
    "gt": {"carla", "torch", "ultralytics", "mmdet3d", "mmcv", "lightning"},
    "slam": {"carla", "torch", "ultralytics", "mmdet3d", "mmcv", "lightning"},
    "calib": {"torch", "ultralytics", "mmdet3d", "mmcv", "lightning"},  # 许 carla
    "perception": {"carla"},  # 许 torch
    "traj": {"carla"},
    "gs": {"carla"},
    "map": set(),  # 含 maptr/(torch) 与 probe_mapvec_oracle(carla)
    "map/maptr": {"carla"},
    "sim": set(),  # 含 live_common(carla+torch)
    "tests": set(),  # 测试跟着被测对象走
}
MASQUERADE_HINT = "ultralytics/mmdet3d/mmcv/lightning 会拉起 torch,已显式列入禁用集合"
```

判据:① 遍历 `autodrivedata/**/*.py`(**含相对导入**,`node.level != 0` 时按所属包解析成绝对名);
② 对每个文件取其**最长前缀匹配**的规则集;③ `hit = mods & forbidden`,非空即记 offender;④ `assert not offenders`。

**诚实说明**:新守卫**不是全面更强** —— 包根/`utils`/`gt`/`slam` 的保护力度与现状等价。它真正新增的是
**① 堵上马甲库与相对导入两个洞;② 获得「按目录表达规则」的能力**(现状根本写不出「`gt/` 不许依赖 `sim/`」);
**③ 让这次重构在结构上成为可能**。不要把它宣传成「安全升级」。

### 3.3 阶段 1 的验收判据(**已完成 2026-09-26**)

**落点**:守卫从 `test_paths.py::test_package_stays_pure_value`(已摘除)**迁到独立文件
`tests/test_layer_guard.py`**(约 200 行,含规则表 + 扫描器 + 自证),因为它从 20 行长到了需要自己的
`LAYER_RULES` / `rule_for` / `imported_toplevels` / `scan_package_layers` 这一整套。

| 判据 | 结果 |
|---|---|
| 目录未动时全绿 | ✅ 收集项 **910** = 899 − 1(摘旧守卫)+ 12(新守卫),0 失败 |
| `test_every_subpackage_is_declared` | ✅ 强制新能力子目录**必须显式声明** —— 防「静默回落到包根规则」(回落到 `_ANY` = 假绿,比红更坏) |

**★ 自证方式相对原计划升级了**:原计划是「临时把 `import carla` 写进 `mapvec.py` → 看它红」,
那是一次性手工动作、且要往仓库里写坏代码。实际做成 **`TestLayerGuardSelfCheck` 的 10 条常驻测试**,
用 `scan_package_layers(injected={...})` **注入合成源码**——不碰真实文件,且**每次都跑**:

| 自证 | 证明什么 |
|---|---|
| `import carla` in `utils/` → 红 | 基本判据有效(非"永远绿") |
| `from ultralytics import YOLO` → 红 | **马甲库洞已堵**(旧守卫放行) |
| `importlib.import_module("torch")` / `__import__("mmdet3d")` → 红 | 动态导入盲区已堵 |
| `import carla` in `sim/` / `perception/` → **不红** | 规则**能区分**,不是"见 carla 就红"的蠢规则 |
| `import torch` in `calib/`(许 carla 不许 torch)→ 红 | 单向规则真的单向 |
| `map/maptr/x.py` 红但 `map/x.py` 不红 | 最长前缀匹配生效 |

**⚠️ 一处与原计划的偏差必须记下**:原计划写「`from . import x` 形态 → **必须红**(验证相对导入)」。
这条判据**是错的**,实际实现也**没有**这么做 —— 相对导入按定义是**包内引用**,解析后顶层恒为
`autodrivedata`,永远不会命中禁用集合。真正的绕过面是**字面量动态导入**(已堵)。
现行处理:相对导入**显式解析**(不静默跳过节点,见 `imported_toplevels` 的注释),但**不假装它能变红**。

---

## 4 执行计划

### 4.0 开发流程(标准 SDLC,每阶段必走)

**分工**:AI 负责改文件 / 跑验证 / 汇报结果;**所有 `git commit` 由用户手动执行**(项目纪律:不自动提交)。
提交信息用 Conventional Commits,**不附 AI 署名 trailer**。

**分支**:**直接在 `main` 上做**,每个阶段一个 commit(2026-09-26 用户裁决,否决了"切 `refactor/filetree`
分支承载全部阶段"的方案)。

> ⚠️ **这个选择的直接代价,做之前必须知道**:阶段 0–9 会在 `main` 上留下 10 个中间状态,
> 阶段 2 试点若失败,**不能一次 `git branch -D` 废弃**,只能逐个 `git revert`(§8)。
> 因此「每阶段一个 commit」从"整洁惯例"升级成**硬约束** —— 一个阶段混进两个 commit,
> 或一个 commit 混进两个阶段,回滚粒度就失效。
>
> 配套:`tools/gitpush.sh` 的 main-only 闸门在 `main` 上正常工作,无需改动。

**每个 commit 的准入(Definition of Done)** —— 五条全过才允许提交:

| # | 检查 | 命令 / 判据 |
|---|---|---|
| 1 | 格式与静态检查 | `ruff check && ruff format --check` 干净 |
| 2 | 相关单测(项目常规口径) | `python -m pytest <被测子树> -q` 全绿 |
| 3 | **全量单测 + 用例数对账** | `python -m pytest <测试根> -q`;用例数**必须等于基线** |
| 4 | 残留引用 grep | 无指向旧路径的**活**引用(文档引用不会报错,见 阶段 9) |
| 5 | 文档与代码同 commit | 项目纪律:改结构必须同步 `docs/fileTree.md` |

**第 3 条是本重构的核心失效模式,不是走过场**:搬测试文件时 pytest 会**静默少收**——
文件搬过去但 `__init__.py` / `conftest.py` / `testpaths` 配错,结果是「跑出来全绿,但只跑了 41 个用例」。
所以必须**逐阶段对比用例数**,而不是只看绿不绿。

> **★ 迁移期有两个测试根,必须同时收集**(这正是上面那个静默失效模式最容易发生的地方):
>
> | 阶段 | 测试根 | 全量命令 |
> |---|---|---|
> | 0–1 | `tests/` | `python -m pytest tests -q` |
> | **2–8** | **`tests/` + `autodrivedata/tests/`(并存)** | **`python -m pytest tests autodrivedata/tests -q`** |
> | 9 | `autodrivedata/tests/` | `python -m pytest -q`(靠 `testpaths`) |
>
> **基线(2026-09-26 实测):895 个收集项 / 48 文件**(另有 3 个**模块级** `importorskip` 跳过 ——
> `auto3dlabel.schema` 缺失,与本次重构无关;任何阶段都应恰好还是这 3 个)。
>
> **★ 对账判据分两种口径,别混**:
>
> | 阶段 | 判据 | 理由 |
> |---|---|---|
> | **0–1**(修陷阱 + 升级守卫) | **收集项 ≥ 895**,0 失败 | 这两阶段**有意新增回归钉子**(§4.0 要求),数字只增不减 |
> | **2–8**(搬迁) | **收集项恒等于 911** | 搬迁**不得**增删用例;数字一变就是丢了或重复收集 |
> | 9(收口) | 同上,不再变化 | |
>
> **少一个就停手查;多一个也要查** —— 多通常意味着 `__init__.py` 缺失导致同名模块被两个测试根重复收集。
> **★ 计数台账(每次变动都要有归因,否则视同丢/重收)**:
>
> | 时点 | 收集项 | 增量归因 |
> |---|---|---|
> | 重构前基线 | 895 | — |
> | 阶段 0 末 | 899 | +4:权重路径 / gitignore 三条 / PROJECT_ROOT / 相对 sys.path 的回归钉 |
> | 阶段 1 末 | 910 | −1 摘旧守卫;+12 新层守卫(10 条自证 + 2 条包级) |
> | **阶段 2 末** | **911** | +1:守卫「子目录须显式声明」判据缺陷的回归钉(见下) |
> | **`send_error` 修复** | **913** | +2:未知流回 404 的回归钉(独立 commit,见阶段 2 记录末尾) |
> | **阶段 3 末** | **913** | 不变 ✓(911 passed + 2 条件跳过;2 条从 passed 变 skipped 是 test_fonts 判据收紧的**有意**结果) |
> | 阶段 4–8 | 应恒为 **913** | 搬迁**不得**增删用例;变了就是丢了或重复收集 |
>
> 3 个模块级跳过(`auto3dlabel.schema` 缺失)全程不变。
>
> 阶段 2 的 `pyproject.toml` 只加 `exclude`(防打包),**`testpaths` 等到阶段 9 再加** ——
> 提前加会让 `pytest` 裸跑只收新根,静默漏掉还在 `tests/` 的那一半。

**每个 bug 修复必须配回归测试**(项目纪律)。阶段 0 的**删除类项**(0.1/0.2)无代码可钉,其余四项各要有钉子:

| 阶段 0 项 | 钉子 |
|---|---|
| 0.3 `sem_bev` 权重路径 | 断言 `paths.project_path(<权重>)` 指向的**文件真实存在** |
| 0.4 `.gitignore` 三条 | 断言 `git check-ignore` 对 `models/*.pt.opt` / `logs/*.log` 返回**已忽略**、对 `assets/logo.png` 返回**未忽略** |
| 0.6 `paths.py` PROJECT_ROOT | 断言 `PROJECT_ROOT` **含 `pyproject.toml`**(而非只断言等于某个深度),搬迁后仍要绿 |
| 0.5 相对 `sys.path` | 该测试在 `cwd=/tmp` 下可单独跑通 |

> 说明:项目 CLAUDE.md 的常规口径是「改动后只跑相关单测,不跑全量」。**本重构是例外** ——
> 搬迁的失效模式恰恰是「没跑到的那些测试静默消失」,只跑相关单测正好避开这个失效模式。

**评审点**:阶段 2(试点)结束后**停下来**,把 diff + 阶段 2 的六条验收判据结果交用户拍板。
满意 → 继续阶段 3–9;不满意 → 按 §8 逐 commit revert(不再是"废弃分支")。

### 阶段 0 — 前置清理与陷阱修复(零引用点,独立可提交)

| # | 动作 | 验收 |
|---|---|---|
| 0.1 | `rmdir logs utils bin/utils autodrivedata/{configs,SLAM,HDMap,CARLA}`(先确认全空) | `git status` 不变 |
| 0.2 | 删 `.ipynb_checkpoints`:根 / `autodrivedata/` / `bin/` / `outputs/smoke*` | §5.5 |
| 0.3 | 修 `bin/sem_bev.py:213` 权重路径 + 同步 `docs/fileTree.md:44` | §5.1 **三处口径同改** |
| 0.4 | `.gitignore` 补:`logs/`、`*.opt`、`models/`、`assets/**` 白名单 | §5.2/§5.3/§5.4 |
| 0.5 | `tests/test_maptr_select.py:18` 相对 `sys.path` 改绝对 | 换 cwd 跑该测试仍绿 |
| 0.6 | `autodrivedata/paths.py:16` 改为**向上搜索 `pyproject.toml`** | §5.6 —— **搬迁的前置** |
| 0.7 | `tools/` 落地:迁 `carla_server.sh` `clear_cache.sh` `gitpush.sh` `gpu_fix/` | 逐条改引用 |

### 阶段 1 — 守卫升级(§3)

零引用点。改完先按 §3.3 做三条反向自证。

### 阶段 2 — `sim/` 试点(**最大最难的一棵,做完停下来验收**)

选 `sim/` 当试点的三个理由:① 最大(21 个模块 + 12 个采集器,占搬迁量 1/5);
② **对守卫冲击最剧烈**(27 个 `import carla` 全在这一棵);③ **唯一碰到非 `.py` 可执行文件**的
(`smoke_radar_collect.sh`)。它跑通了,`map/` `slam/` `calib/` 都是它的简化版。

| # | 动作 |
|---|---|
| 2.1 | `git mv` 21 个文件到 `autodrivedata/sim/`;`git mv tests/test_live_common.py tests/test_scenarios.py autodrivedata/tests/sim/` |
| 2.2 | 改 import:`from carla_common import X` → `from autodrivedata.sim.carla_common import X`(29 处同级裸名 import 全在此) |
| 2.3 | `pyproject.toml`:`packages.find` 加 `exclude = ["autodrivedata.tests*"]`(**只加这个**;`testpaths` 留到阶段 9,见上文) |
| 2.4 | 守卫 `LAYER_RULES` 加 `sim`(已含在 §3.2) |
| 2.5 | 入口文档改 `python -m autodrivedata.sim.<x>` |

**阶段 2 验收判据(必须全过才继续)**

1. `python -m pytest tests autodrivedata/tests -q` 全绿,**且收集项恰为 910**(两个根同时收集;少一个即停手)
2. `ruff check && ruff format --check` 干净
3. **跑一次真实采集**:`python -m autodrivedata.sim.collect_drive --scene rain_night --frames 5`
   → 产物落在**项目根** `outputs/`(验证 §5.6 的 `paths.py` 修复真的生效,没有落到 `autodrivedata/outputs/`)
4. `python -m autodrivedata.sim.view_stream --view follow` 起得来(验证 `--maptr-ckpt` 之外的实时流没断)
5. `python -c "import autodrivedata.sim.carla_common"` 在**干净 env** 能解析(editable finder 不须重装)
6. grep 残留:`grep -rn 'bin/' --include='*.md' .` 只剩 §6 已知的历史条目

**→ 停下来给用户看结果。满意再进阶段 3;不满意则回滚(§8),代价 = 1/5。**

#### ★ 阶段 2 执行记录(已完成 2026-09-26)

**六条验收判据全过**(逐条实测):

| # | 判据 | 实测 |
|---|---|---|
| 1 | 两测试根全量绿 + 计数对账 | **911 passed / 3 skipped / 0 failed**;911 = 910 + 1(见下) |
| 2 | `ruff check && ruff format --check` | 干净(183 文件) |
| 3 | **真实采集** | `python -m autodrivedata.sim.collect_drive --scene rain_night --frames 5` 跑通,产物落**项目根** `outputs/kitti_rain_night` ⇒ **§5.6 的 `paths.py` 修复验证有效** |
| 4 | 实时流起得来 | `/stream/main` 出帧(`Content-Length: 54885`),`--dump` 落 overlay+raw 两张,退出干净 |
| 5 | 干净 env 可 import | `from autodrivedata.sim import carla_common` ✓(**editable finder 不须重装**,与预判一致) |
| 6 | grep 残留 | 仅剩 `Plan.md`(冻结区) |

**★ 试点抓到的四件事**(这就是先做试点的价值——它们都只在真搬之后才暴露):

1. **两处漏网 import 形态**。批量替换只覆盖了 `from X import`,漏掉 `import X`(3 处 `import live_studio`)
   与 `import collect_nus`(3 处)。**教训**:机械替换必须覆盖 `import X` / `from X import` / `import X as Y` 三种形态。
2. **测试按路径读源码**。`test_nuscenes_calib_consistency.py` 用 `BIN / "collect_nus.py"` 读源码做 AST 检查,
   搬迁后 `FileNotFoundError`。**教训**:「搬迁清单」不能只数 import,还要数 `路径常量 + 读文件`。
   已加 `SIM = ROOT / "autodrivedata" / "sim"` 常量分流。
3. **守卫的判据缺陷(守卫自己抓出来的)**。`test_every_subdirectory_has_an_explicit_rule` 原判据是
   「目录名在 `LAYER_RULES` 表里」,建出 `autodrivedata/tests/sim/` 后误报 —— 而子目录由父规则覆盖**本就正确**。
   已改为「命中的最长前缀键 != `""`」,并加回归钉(判据缺陷同 commit 修,§4.0 要求)。
4. **pyright +7 是「解除遮蔽」不是「新增错误」**。`test_live_common.py` 在 HEAD 上零报错,搬进包后 7 条
   `reportArgumentType`(`_FakeActor` 测试替身 vs `rig_mount_deviation(ego: Vehicle)`)。**根因**:搬迁前
   `import live_common` 对 pyright **不可解析** ⇒ `lc.rig_mount_deviation` 是 `Unknown` ⇒ 实参根本不检查。
   搬进包后导入可解析,签名才可见。**同范围对比:42 → 49,增量全在 `reportArgumentType`(7→14),其余类型数量不变**
   ⇒ **净增 0 条真实错误**。

#### 附带修掉一条预存缺陷(独立 commit,非本阶段)

**症状**:`autodrivedata/sim/live_common.py` 的未知流分支
`self.send_error(404, f"未知流;可选:{', '.join(slots)}")` 把中文塞进 HTTP **状态行**,
而 `http.server` 用 **latin-1** 编码状态行 ⇒ `UnicodeEncodeError` 抛在请求线程里,
客户端拿到的是**连接被重置**(实测 curl 0 字节)而**不是 404**。
`git show HEAD:bin/live_common.py` 确认**搬迁前就是这样**,非本次引入。

**修法**:中文改走 `send_error` 的 `explain` 参数(进 **body**,UTF-8 编码);
URL 带来的流名经 `html.escape` 再进 HTML。

**回归钉**:`TestUnknownStreamIs404NotACrash`(2 条)——
① 请求不存在的槽名必须得 **HTTP 404 且 body 含中文与可选列表**,不是 `RemoteDisconnected`;
② 反向:正常路径(索引页)不受影响。
**已做反向自证**:临时还原修复前写法 ⇒ 测试红,报错正是
`http.client.RemoteDisconnected` + `UnicodeEncodeError: 'latin-1' ... position 13-15`。

> **注**:本条不属于「搬迁」,按 §4.0「一个阶段一个关注点」独立成 commit(阶段 2 已先提交,
> 否则两者混在一起会让 `git revert` 阶段 2 时连带打掉这个修复)。

### 阶段 3..9 — 其余子树,每个一棵一 commit

| 阶段 | 子树 | 文件数 | 难度 |
|---|---|---|---|
| 3 | `calib/` | 14 + 10 测试 | 中(许 carla,涉及 `-m` 改命令) |
| 4 | `map/` | 29 + 11 测试 | **最高**(含 `maptr_impl/`→`map/maptr/`、`test_fonts.py:34` 硬编码、`probe_mapvec_oracle` 许 carla) |
| 5 | `slam/` | 11 + 4 测试 | 中(`slam_cpp.cpp` + `slam_diff_test.py` 的 g++ 调用需重跑对拍) |
| 6 | `perception/` | 18 + 9 测试 | 低 |
| 7 | `gt/` + `utils/` | 9 + 12 测试 | **低但要小心**(§5.6 的 `paths.py` 在这批) |
| 8 | `traj/` + `gs/` | 3 | 低 |
| 9 | 文档同步 | — | 见 §4.1 |

#### ★ 阶段 3 执行记录(已完成 2026-09-26)

**搬迁**:14 模块 → `autodrivedata/calib/`(**`calib.py` 改名 `core.py`**,见下)+ 9 测试 → `autodrivedata/tests/calib/`。
引用重写 **20 文件 / 34 处**(含阶段 2 遗留的 `from autodrivedata import X` 形态)+ 摘掉 3 处失效的 `sys.path` 引导。

**结果**:`913 收集项 = 911 passed + 2 条件跳过`(另 3 条模块级跳过),**0 失败**;ruff 干净。

**★ 五个发现(四个都是计划没料到的)**:

1. **计划缺陷:`calib/calib.py` 会产生 `autodrivedata.calib.calib`**。计划 §2 把 `calib.py` 列进 `calib/`,没注意到自反名。
   **用户裁决改 `calib/core.py`**(30 处调用点改写),理由是「只有一条路径到 `CameraIntrinsics`,
   不留『同一名字两个入口』的长期歧义」。已回写 §2 目标树。
2. **第二类漏网 import 形态:`from autodrivedata import <模块>`**(11 处)。阶段 2 的教训只覆盖了
   `import X` / `from X import` / `import X as Y`,**没覆盖 `from <包> import <模块>`**。
   ⇒ **批量替换必须覆盖的完整形态清单**(以后照这个查):`import X`、`from X import`、`import X as Y`、
   **`from <pkg> import X`**、**`from <pkg> import X as Y`**、路径字符串。
3. **`test_fonts.py` 的 `DRAWING_MODULES` 硬编码路径** —— 计划 §5.7 预测的坑,真踩到了
   (3 条:`bin/viz_calib_check.py` / `bin/probe_calib.py` / `autodrivedata/calib_live.py`)。
4. **`test_drawing_module_imports_the_font_module` 判据过松,长期「因错误的原因」全绿**:
   旧判据 `n.module in ("autodrivedata", "autodrivedata.fonts")` —— **任何** `from autodrivedata import X`
   都算通过,`live_studio.py` 里那句 `from autodrivedata import calib_live as cl` 恰好把它喂饱了。
   阶段 3 把该行改精确后,假通过暴露:那两个模块**本来就不碰文字渲染**(只委托 `draw_hud` / 构造 `hud_line` 字符串)。
   **已改**为「源码出现 `fonts` ⇒ 必须真的 import 字体落点(两种惯用形式都认,但后者只认别名恰为 `fonts` 的)」,
   并做反向自证(删 import → 红)。**教训:判据里的每个「或」都要问「会不会被无关代码满足」。**
5. **`parents[N]` 陷阱再现**:`test_nuscenes_calib_consistency.py` 的 `ROOT = Path(__file__).parents[1]`
   随文件从 `tests/` 挪到 `autodrivedata/tests/calib/` ⇒ 从仓库根变成 `tests/`。
   已改走 **`paths.PROJECT_ROOT`**(§5.6 的同款修法),从此免疫。

**我自己的一个操作失误(记下以防再犯)**:清理时写了 `s.replace("import viz_rig_check", ...)`,
把**已经改好的** `from autodrivedata.calib import viz_rig_check` 二次替换成
`from autodrivedata.calib from autodrivedata.calib import ...`(语法错)。**替换必须带幂等护栏**。

#### ★ 阶段 4 执行记录(已完成 2026-09-26)——**全重构最大的一块**

**搬迁**:29 模块 → `autodrivedata/map/`(`maptr_impl/`→`map/maptr/`、`maptr_official/`→`map/maptr_official/`
**整包搬**,含 2 个官方栈编排脚本)+ 11 测试 → `autodrivedata/tests/map/`。
引用重写 **约 200 行 / 40+ 文件**(含补扫阶段 2/3 的遗留)。

**结果**:`913 收集项 = 911 passed + 2 条件跳过`(另 3 条模块级跳过),**0 失败**;ruff 干净。

**★ 六个发现,其中三个是「必须先把含义数清楚」类的**:

1. **`maptr_official` 在本仓有【三种含义】,盲替换会毁掉两种。** 实测:包引用 6 处、
   **输出目录 `outputs/maptr_official/` 26 处**、**conda env 名 `/root/.../envs/maptr_official` 3 处**。
   只有前者该改。**做法**:先分类计数,再用**逐条规则**(而非全局替换),改完用 `git diff | grep`
   确认后两类**零改动**(实测 26/3 与 HEAD 逐字一致)。
2. **幂等护栏失效一次**(`maptr_v1_carla.py:10` 出双前缀)。根因:护栏只检查**原始行**,
   而同一行内规则 A 的输出恰含规则 B 的输入 ⇒ 二次命中。
   **修法**:改为**前进式护栏**(每次替换前查匹配处的前缀是否已是新口径)。
3. **shell 的深度陷阱(与 `parents[N]` 同族)**:`assemble_and_merge.sh` / `finalize_maptr_600.sh` 的
   `cd "$(dirname "$0")/.."` 在 `bin/` 下是一级,搬到 `autodrivedata/map/` 后变两级 ⇒
   **会 cd 到 `autodrivedata/` 而不是仓库根**。已改 `../..`。
   ⇒ **搬迁清单必须包含「算自己位置的 shell 表达式」**,不只是 import 与路径字符串。
4. **阶段 2/3 的 `.md` 引用没扫**(当时只扫 `.py`/`.sh`):CLAUDE.md / README / Plan2 / milestone2 里
   一批命令与反引号路径仍是旧的。本次做了**全仓一次到位**的扫描(83 行 / 30 文件),
   排除 `Plan.md`(冻结)与 `docs/testLog.md`(历史日志)。
5. **测试的惰性 import 逃过 collection 检查**:`test_maptr_select.py` 在**方法体内** `import train_maptr`,
   所以 `--collect-only` 计数正常、**跑起来才 `ModuleNotFoundError`**。
   ⇒ **计数对账抓不到这类**,必须真跑全量(本次正是全量跑出来的)。
6. **`ruff check` 抓到了我漏的一整类**:`from autodrivedata.opendrive import` 等 5 个
   「从 `autodrivedata/` 根迁入 `map/`」的模块引用(35 处)。**ruff 是这次的重要安全网。**

> **★ 给后续阶段的搬迁清单(把三次教训合并成一张表,照它扫一遍再动手)**:
>
> | # | 形态 | 三次各自的实例 |
> |---|---|---|
> | 1 | `import X` / `from X import` / `import X as Y` | 阶段 2 漏 `import live_studio` |
> | 2 | `from <包> import X` / `from <包> import X as Y` | 阶段 3 漏 11 处 |
> | 3 | `from autodrivedata.<m> import`(模块换家) | 阶段 4 漏 35 处(ruff 抓到的) |
> | 4 | **路径字符串**(反引号散文 / docstring 用法示例 / shell) | 阶段 4 补扫 |
> | 5 | **硬编码源码路径常量**(测试按路径读源码) | 阶段 3 的 `DRAWING_MODULES` |
> | 6 | **算自己位置的表达式**(`Path(__file__).parents[N]` / `cd "$(dirname "$0")/.."`) | 阶段 3 + 阶段 4 |
> | 7 | **方法体内的惰性 import**(逃过 `--collect-only`) | 阶段 4 |
> | 8 | **同名多义**(包名 / 输出目录 / env 名) | 阶段 4 的 `maptr_official` |
> | 9 | **`import <包>.<模块> as 别名`** | 阶段 5 漏 1 处(`import autodrivedata.live_slam as ls`) |
> | 10 | **同一条目里同一路径出现两次**(`[X](X)` 型 markdown 链接) | 阶段 6:替换脚本每条规则只换**第一处** |
> | 11 | **散文里的 `包/模块.属性` 混合写法**(无 `.py` 后缀) | 阶段 7:`autodrivedata/paths.project_path` |

#### ★ 阶段 7 执行记录(已完成 2026-09-26)

**搬迁**:`gt.py`/`static_gt.py`/`traffic_light.py` + `export/` 子包(3)→ `autodrivedata/gt/`
(**`gt.py` 改名 `gt/core.py`**,与 `calib/core.py`、`slam/core.py` 同款);
`geometry.py`/`paths.py`/`fonts.py` → `autodrivedata/utils/`。12 测试随迁,
`test_layer_guard.py` 归 `autodrivedata/tests/` 根(它是包级守卫,不属于任何能力面)。
引用重写 **147 行 / 83 文件**——本次量最大,因为 `paths.py` 被 42 处引用(全仓最多的模块)。

**结果**:`913 收集项 = 911 passed + 2 条件跳过`,**0 失败**;ruff 干净。

**★ 方法升级(阶段 6 那个 bug 的正解)**:替换脚本从「逐条规则 `find`+一次赋值」改为
**单一正则 `PAT.sub(lambda m: MAP[m.group(0)], ln)`**。`re.sub` **不重扫替换结果**,
从机制上杜绝二次命中,不再依赖手写的幂等护栏。**后续阶段一律用这个写法。**

**★ 四个发现**:

1. **`parents[N]` 陷阱第三次出现**(`test_fonts.py` 从 `tests/` 挪进 `autodrivedata/tests/utils/`)。
   这一次**不只是修,而是把判据机械化**:写了个脚本遍历包内所有 `.py`,
   用 `p.parents[N] == 仓库根` 逐一判定,一次列出全部命中(含 docstring 里的假阳性)。
   ⇒ **沉淀成阶段 9 的一个候选**:把这条做成常驻守卫,格式如
   「若某文件的 `parents[N]` 解析结果既不是仓库根也不是 `autodrivedata/` 等已知锚点,则报错」。
2. **`.ipynb_checkpoints` 又冒出 5 处**(`autodrivedata/`、`tools/`、`autodrivedata/utils/`、`autodrivedata/perception/`)。
   VS Code 在编辑文件时会重建它们 ⇒ **阶段 0 的清理是一次性的,不是一劳永逸**。
   (已被 `.gitignore` 覆盖,不影响 git;只是视觉噪音 + 那个"纯值包里的过期副本"隐患。)
3. **两个预判断点都命中,预判有效**:① `test_paths.py:25` 断言 `autodrivedata/paths.py` 存在;
   ② `test_fonts` 的判据 `n.module in ("autodrivedata", ...)` 在 `fonts` 搬进 `utils/` 后失效。
   两条都在动手前就写下来了,修起来零排查成本 —— **这就是"搬迁清单先扫一遍"的回报**。
4. **我自己的编辑引入了回退**:`re.sub` 规则跑完后,我手写 `from autodrivedata import paths`
   去改 `test_fonts.py`,又把已经改好的形式写回了旧的(规则不会再跑第二遍)。
   ⇒ **教训:替换脚本跑完之后的手工编辑,必须按新口径写;改完要再扫一次残留。**

#### ★ 阶段 8 执行记录(已完成 2026-09-26)——**搬迁期结束**

**搬迁**:`assemble_traj_pt.py` / `convert_hivt_pt.py` → `autodrivedata/traj/`;
`train_3dgs_mini.py` → `autodrivedata/gs/`。

**★ 结构性里程碑:`bin/` 与 `tests/` 两个顶层目录消失。**
`bin/` 的最后 3 个文件迁走后已删;`tests/` 自阶段 7 起就空了,同期删除。
顶层从「6 个源码目录 + 2 个散落目录」收敛为:**`autodrivedata/` 一个主包**
(+ `tools/` 开放性工具、`docs/` 文档、`outputs/` 产物、`hdMapGitHub/` 上游)。

**结果**:`913 收集项 = 911 passed + 2 条件跳过`,**0 失败**;ruff 干净。

**发现**:

1. **形态 11(`包/模块.属性` 散文写法)在源码里还有 4 处**,阶段 7 只清了文档里的:
   `drive_ego.py`(`bin/live_common.KeyboardState`)、`rigviz.py`、`live_slam.py`、`traffic_light.py`。
   它们的共性是**没有 `.py` 后缀**,所以正则 `bin/<m>.py` 抓不到。
   ⇒ 已补扫 `bin/<模块>.<属性>` 这一形态。**后续任何搬迁都要连同这一形态一起扫。**
2. `convert_hivt_pt.py` 的 `sys.path.insert(0, ".../hdMapGitHub/HiVT")` 是**绝对路径**,
   移位后不受影响(实测复核)——**位置表达式要逐个判断"它算的是谁的位置"**,不能一律当成会断。

**阶段 0–8 合计**:搬迁 **~150 个文件**、重写 **~600 行引用**、涉及 **11 类引用形态**;
全程 `ruff` 干净、**收集项从 895 稳定推到 913 后一条未丢**、每阶段全量跑通。

#### ★ 阶段 6 执行记录(已完成 2026-09-26)

**搬迁**:**17** 模块 → `autodrivedata/perception/` + 9 测试 → `autodrivedata/tests/perception/`;
**`probe_radar_l3.py` 改去 `sim/`**(见下)。引用重写 **68 行 / 35 文件**。

**结果**:`913 收集项 = 911 passed + 2 条件跳过`,**0 失败**;ruff 干净。
产物路径零误伤(`kitti_ab_` 20、`kitti_sweep` 13、`sem_bev` 23、`mono_distance` 16、`accum_map` 13
均与 HEAD 逐字一致)。

**★ 两个发现**:

1. **计划内部矛盾:`probe_radar_l3` 是 CARLA 探针,却被计划放进 `perception/`(该规则禁 carla)。**
   实测该模块 `import carla` 且需要 CARLA 服务器(启动 client、spawn 雷达 actor)。
   **处置**:按既有先例改放 `sim/` —— `probe_imu.py`(CARLA IMU 能否支撑 FAST-LIO2)阶段 2 已进 `sim/`,
   二者同类(都是**验证 CARLA 平台某种传感器/能力是否可用**),而 `probe_calib` / `probe_rig_mount`
   进 `calib/` 是因为它们是**标定**域。
   **保住的收益**:`perception/` 维持「不 import carla」这一真实信息(那 17 个模块确实不需要 CARLA,
   可在无服务器的机器上跑),不必为了让一个文件住进去而把整条规则放宽成 `_ANY`。
2. **替换脚本的真 bug:每条规则只替换了第一处。** 症状:`CLAUDE.md` 里
   `[autodrivedata/perception/eval_attr.py](bin/eval_attr.py)` —— **显示文本改了、链接目标没改**。
   根因是 `out.find(old)` 取一次就赋值,没有遍历全部出现位置。
   ⇒ **清单加第 10 行**:`[X](X)` 这类同条目内重复出现,必须逐次替换。
   (本次全仓只有 CLAUDE.md 一处,已修并复核清零。)

#### ★ 阶段 5 执行记录(已完成 2026-09-26)

**搬迁**:10 模块 + `slam_cpp.cpp` → `autodrivedata/slam/` + 4 测试 → `autodrivedata/tests/slam/`。
引用重写 **74 行 / 19 文件**。

**结果**:`913 收集项 = 911 passed + 2 条件跳过`,**0 失败**;ruff 干净。

**发现**:

1. **新增第九种形态:`import <包>.<模块> as 别名`**(`import autodrivedata.live_slam as ls`)。
   前四种规则全是 `from ... import ...`,**`import X.Y as Z` 不在其中**。已补进上面的清单。
   注意它**只在全量跑时才暴露**(语法上合法、`--collect-only` 正常,调用时才 `ModuleNotFoundError`)。
2. **`slam.py` → `slam/core.py`**,与阶段 3 用户裁决的 `calib/core.py` 同款(避免 `autodrivedata.slam.slam`)。
   **同一模式在后续阶段要一致处理**:`perception`/`gt`/`traj`/`gs` 若出现「模块名 = 目录名」照此办。
3. **`CPP_SRC = Path(__file__).parent / "slam_cpp.cpp"` 恰好仍然正确** —— 因为 `.py` 与 `.cpp` **一起搬**。
   **但这必须验证而非假设**:实测 `g++ -std=c++17 -O2` 在新位置编译通过(产物 103848 bytes),
   `slam_diff_test.py` 的解析路径也指对了。若哪天只搬 `.py` 不搬 `.cpp`,这条就会静默断。
4. **同名多义**这次没有踩坑(规则只匹配 `bin/<m>.py` 与 `autodrivedata.<m>`,不碰 `outputs/slam*/`)——
   实测 `outputs/slam/` 8 处、`outputs/slam_gt` 8 处、`kitti_slam` 14 处**与 HEAD 逐字一致**。

约 1000 个引用点中,**绝大多数是文档**。它们**不会报错**。

1. `CLAUDE.md` 常用命令段:全部 `python bin/x.py` → `python -m autodrivedata.<cap>.x`(约 40 条)
2. `README.md`/`Plan2.md` 同改
3. `docs/fileTree.md`:按新树重写 §1–§5 + 更新维护约定
4. **`Plan.md` 冻结区处置** —— 见 §7
5. 机械校验:`grep -rn 'bin/\|tests/test_' --include='*.md' .` 逐条确认
6. `pyproject.toml` 收口:此时 `tests/` 已空,加 `[tool.pytest.ini_options] testpaths = ["autodrivedata/tests"]`;
   删空的 `tests/` 目录;`git mv` 用 `git log --follow` 确认历史可追

---

## 5 已实测的陷阱与修复(全部验证过,每条都会出事)

### 5.1 `bin/sem_bev.py` 权重路径已断(用户本次搬迁直接触发)

```
bin/sem_bev.py:213   project_path("yolo11s-seg.pt")   # 注释:「权重放项目根」
实际位置             models/yolo11s-seg.pt            (20,669,228 B)
docs/fileTree.md:44  仍写「放项目根」                   ← 第三处同口径的错
```
`project_path("yolo11s-seg.pt").exists() == False`。同一函数 `:216` 走的是 `outputs/models/yolopv2.pt`
⇒ 仓库有**四个**权重落点。ultralytics 8.4.115 的 `GITHUB_ASSETS_NAMES` 含该名,缺文件时**静默联网重下**。
**修**:三处同口径(代码 / 文档 / 实际文件位置)。

### 5.2 `.gitignore` 的 `*.pt` **不匹配 `.opt`** —— `models/` 会吞 1.2 G 侧车

```
models/maptr_ep512.pt       → IGNORED      (*.pt 命中)
models/maptr_ep512.pt.opt   → NOT-IGNORED  ← 会入库
outputs/maptr_ep512.pt.opt  → IGNORED      (靠 outputs/ 整体忽略兜住)
```
`.opt` 优化器侧车每个约 250 M、合计约 1.2 G,**比权重本身还大**。

### 5.3 `logs/` 不在 `.gitignore`
`git check-ignore logs/train.log → NOT-IGNORED`。`.gitignore` 只有 `lightning_logs/`(带下划线)。

### 5.4 `assets/` 里的图片会被**静默忽略**
`git check-ignore assets/logo.png → IGNORED`(全局 `*.png/*.jpg/*.bmp/*.tif/*.mp4/*.onnx/*.npy`)。
`git add` 不报错、直接跳过。**修**:在图片规则**之后**追加 `!assets/**`,再按体积关回大文件。

### 5.5 两个空 `utils/` + 四处 `.ipynb_checkpoints`
无 `__init__.py` 时按 **PEP 420 命名空间包**解析(`find_spec` 返回 `origin=None` 的 NamespacePath)——
可 import、可被 `rglob` 扫到、但 `find_packages` 不收录,**三种机制看法不一致**。
`autodrivedata/.ipynb_checkpoints/geometry-checkpoint.py` 与 `git show e7d45bc:autodrivedata/geometry.py`
**逐字节相同**(142 行 vs 现役 500 行,缺 `NUS_EGO_ORIGIN_X` / `CARLA_CAM_TO_NUS_CAM`)。

### 5.6 `paths.py:16` 的 `parents[1]` —— **搬迁会静默打断**

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # autodrivedata/paths.py → 项目根 ✓
```
`paths.py` 一旦挪进 `autodrivedata/utils/`,`parents[1]` 变成 **`autodrivedata/`** 而不是项目根 ⇒
`project_path("outputs/x")` 解析到 `autodrivedata/outputs/x`,**所有产物落错地方**。
被 `test_paths.py:17-19` 抓到(`(PROJECT_ROOT/"pyproject.toml").is_file()` 为假),但这是运气。
**修**:改成**向上搜索 `pyproject.toml`**,从此免疫任何搬迁。

### 5.7 硬编码路径字符串(不出现在任何「引用点计数」里)

| 位置 | 内容 | 后果 |
|---|---|---|
| `tests/test_fonts.py:34` | `DRAWING_MODULES` 含字符串 `"autodrivedata/mapviz.py"`,三处 `(ROOT/rel).read_text()`(58/238/253 行) | 搬 `mapviz.py` → `FileNotFoundError`,**一个字符串干掉三个 test function 加一整组 parametrize** |
| `tests/test_probe_calib.py:28` | `import bin.probe_calib as pc` | 单独搬 `probe_calib.py` 断这条测试 |

### 5.8 `bin/slam_cpp.cpp` + `slam_diff_test.py` 的 g++ 调用
路径敏感。搬完必须重跑对拍(验收:numpy 与 `slam_cpp` 同一 `(prev,cur,init,seed)` 下单次 ICP 一致)。

---

## 6 已实测否决(**别再提**)

| 提案 | 实测引用点 | 否决理由 |
|---|---|---|
| `bin/` 分子目录 | **422** | 29 处同级裸名 `import` + 5 处 `sys.path` ⇒ `bin/` 事实上一必须是单层非包目录。**注**:本计划通过「搬进包 + 改 `python -m`」消除了这个约束 —— 那 29 处 import 在阶段 2 被改成绝对导入,约束随之消失 |
| `bin/` 库代码迁出建**独立顶层包** | 77 | ① 5 个测试的 `sys.path` hack 删不掉(还要 import 留在 `bin/` 的脚本);② 新包不自足(`rig_check` 反向依赖 `bin/probe_calib`);③ pyright 的真正解是 1 个 `pyrightconfig.json`。**注**:本计划把 `bin/` 整体收进包内,该结论的前提已变 |
| `tests/` 目录镜像 | 91 | 反向映射实测 41 个源模块全部有测试、0 孤儿、0 失效 import —— 平铺没有漏。**注**:本次仍镜像,但动机**不是补漏**而是「让测试能用真 import 而非 `sys.path` hack」 |
| `outputs/` 命名规范大搬迁 | 37 | 要篡改 `Plan.md` **冻结账本** 9 处(含 `.opt` 的 `step=10,242` 对账量) |
| 废弃权重加文件名前缀 | **191** + 100 份契约 json | 从**有版本控制**的文档标记换成**无 diff / 无 blame** 的文件名;`live_common.py:91` 已按 stem 正确派发 rig |
| `.gitignore` 补 `models/ logs/ utils/` 目录级忽略 | 0 | **0 代价 ≠ 值得**:`rmdir` 严格占优。**注**:阶段 0.4 仍要补 `models/` 与 `logs/`,但理由是 §5.2/§5.3 的**真陷阱**(`.opt` 与训练日志会入库),不是「整洁」 |
| 权重搬迁到 `models/` | 48 | git 层面收益为零(468 个 `.pt` 一个都没入库)。**仍要做**,但理由是把 §5.1 的四个落点收敛成一个,不是 git 清洁度 |
| `pyrightconfig.json` 加 `extraPaths: ["bin"]` | 4 | 本计划把 `bin/` 收进包后 import 天然可解析,该方案作废 |

---

## 7 `Plan.md` 冻结区的处置(唯一没有好答案的地方)

`Plan.md` 已冻结(§5.7a「不再新增」),其内部约 45 条 md 引用中 **7 条指向 `bin/` / `tests/` 的路径**。
**改了违反冻结纪律,不改则永久指向不存在的路径** —— 这是个死结,不是能靠仔细解决的技术问题。

**处置**:不修改内容,在**文档头部**加一行声明,把它固化成**已知状态**:

> ⚠️ 本文档 §5.7c 之后的路径为 **2026-09-26 目录重构前口径**。
> `bin/x.py` → `autodrivedata/<能力>/x.py`、`tests/test_x.py` → `autodrivedata/tests/<能力>/test_x.py`,
> 完整映射见 [Plan_fileTree.md](Plan_fileTree.md) §2。

---

## 8 回滚方案

每个阶段一个 commit,**回滚粒度 = 一个子树**。任一步骤失败:

1. `git revert <阶段 commit>`(**不用 `reset --hard`**,保留历史;`main` 上尤其不能改写已推送历史)
2. 若已改了 `pyproject.toml` / 守卫,一并 revert
3. **不需要重装 editable**(实测 finder 对子模块委托 `PathFinder`,删除子目录后自然不再解析)

阶段 2 试点失败 ⇒ **逐个 revert 阶段 2、1、0 的 commit**,回到「现状 + 守卫升级 + 陷阱已修」——
注意顺序:**倒序** revert(2 → 1 → 0),否则守卫升级被先撤会让阶段 2 的 `sim/` 立刻违反旧红线。

> ⚠️ 这是「直接在 `main` 上做」的固有代价:没有一次性的 `git branch -D` 逃生口。
> 若嫌麻烦,可在阶段 2 **之前**补切分支(此时只有阶段 0–1 两个 commit,迁移成本最低)——
> 阶段 3 之后就别切了,历史已经纠缠。

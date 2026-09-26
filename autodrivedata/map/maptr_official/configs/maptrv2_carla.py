"""MapTRv2(4 类,w/ centerline)在 CARLA 环视数据上的官方复线配置(项目侧)。

`hdMapGitHub/` 保持 pristine:本文件不往仓库塞任何东西,靠 `_base_` **继承**官方 config。

口径账本(Plan.md §5.12;读结论前必须按这几条核对):

- **数据** = `outputs/maptr_official/data/`(200 训 / 100 留出,6 相机 1242×375,
  BEV 窗口 x±15 / y±30 = 官方 nuScenes 的 `point_cloud_range`,GT = ego 系 20 点折线)。
  官方**离线**数据集直吃 infos 里的 `annotation`,不做任何坐标变换 → 数据源零适配。
- **等预算口径**(2026-09-13 建账,2026-09-14 由 512 ep 裁到 **256 ep**;裁剪理由与记录见
  Plan.md §5.12):自实现基线是**两轮各 256 ep** —— 第一轮完整 schedule →
  `maptr_ep256.pt`(留出集 @0.2 **0.0510** / @0.3 0.0952 / @0.4 0.1350),第二轮 lr 2e-5
  每 128 ep 减半的低 lr 续训 +256 ep → `maptr_ep512.pt`(@0.2 **0.0674**,仅保守操作点获益)。
  `outputs/maptr_ep512.pt.opt` 里 AdamW 的 `step` 计数 = **10,242** ⇒ 该轮是 **batch 5**
  (200/5 = 40 步/epoch × 256 = 10,240,余 2 为自适应探针步),与 Plan.md §5.11 的
  "batch 5,~34s/ep"互证 ⇒ **两轮各 51,200 样本 / 10,240 步**。
  故对齐点取**第一轮**:本文件 `total_epochs=256` + **bs1 × 梯度累积 5**(数值上等价 bs5,
  理由见下方 ACCUM 块)→ 256 × 200 ÷ 5 = **10,240 步 / 51,200 样本**,与基线第一轮在
  **样本数与优化步数上双双相等**。末轮 512 ep 是**低 lr 续训**而非更长的一次 schedule
  (lr 2e-5 vs base 6e-4),拿它当对齐点反而多引入一个 schedule 形状的混淆项。
- **分辨率**:`IMG_SCALE=1.0` → 原生 1242×375,**与自实现基线同分辨率**(对照里少一个混淆项)。
  官方那份 config 写 0.5,是**针对 1600×900 的 nuScenes**(→800×450);我们的数据原生
  1242×375,再打 0.5 只剩 621×187 —— 比官方训练分辨率低一档,自带系统性劣势。故默认取
  1.0,官方 0.5 档留作消融(`MAPTR_IMG_SCALE=0.5`)。12GB 卡上 1.0 显存吃紧时先测
  (bs2@1242×375 的激活 ≈ 官方 800×450 的 1.3×)
- **继承而非重写**:模型/损失/优化器(lr 6e-4、backbone ×0.1、CosineAnnealing、
  `warmup_iters=500` **不**随总长缩放——官方 110e 长表与 24e 长表同款)/主干预训练,
  全部来自官方 config。本文件只覆盖:数据源、类表、批大小、总步数、评测/存盘间隔、work_dir
- **唯一的功能性偏差**:官方训练带 LiDAR 深度监督(pv_seg 的 depth loss)。我们的环视数据集
  没有 LiDAR → train_pipeline 去掉 `LoadPointsFromFile` + `CustomPointToMultiViewDepth`,
  `CustomCollect3D` 不再收 `gt_depth`;官方 detector 在 `gt_depth=None` 时**自动跳过**该项
  (`maptrv2.py`: `if gt_depth is not None and depth is not None`)。
  bev_seg/pv_seg 的 GT mask 仍由官方 `gen_vectorized_samples` 用 lidar2img 栅格化生成,
  **不依赖 LiDAR** → 除 depth loss 外训练与官方逐项一致
- `seed = 0`:官方 config 不带随机种子(不可复现);我们钉死种子,便于三方横向对账

启动(**必须在仓库根、且 PYTHONPATH 同时含仓库根与项目根**;官方 train.py 会把 `plugin_dir`
按字符串切分后 `import_module('projects')` → `plugin_dir` 只能是相对路径,绝对路径会拼成
`.root.autodl-tmp...` 直接炸):

    cd /root/autodl-tmp/Documents/Projects/AutoDriveData/hdMapGitHub/MapTR_maptrv2 && \
      PYTHONPATH=$PWD:/root/autodl-tmp/Documents/Projects/AutoDriveData \
      python tools/train.py \
        /root/autodl-tmp/Documents/Projects/AutoDriveData/autodrivedata/map/maptr_official/configs/maptrv2_carla.py
"""

_base_ = [
    "/root/autodl-tmp/Documents/Projects/AutoDriveData/hdMapGitHub/MapTR_maptrv2/projects/configs/"
    "maptrv2/maptrv2_nusc_r50_24ep_w_centerline.py",
]

_PROJ = "/root/autodl-tmp/Documents/Projects/AutoDriveData"
data_root = _PROJ + "/outputs/maptr_official/data/"
work_dir = _PROJ + "/outputs/maptr_official/maptrv2"  # 权重/日志落项目内(用户要求)

# 主干预训练权重:官方 base 指的是**相对 cwd** 的 'ckpts/resnet50-19c8e357.pth',而三个官方
# 仓库都没有 ckpts/ 目录(README 让用户自己放)→ 不覆盖就是构建即失败(该值会被
# MVXTwoStageDetector 转成 `img_backbone.init_cfg`,路径不存在直接 IOError)。
# 口径:不往 hdMapGitHub 里写任何文件 → 改用项目内绝对路径。
# 权重本体 = torchvision 的 ImageNet ResNet-50;本机 torch hub 缓存里是**新版文件命名**
# `resnet50-0676ba61.pth`(torchvision 换过下载 URL,哈希后缀随之变,内容同为
# IMAGENET1K_V1),由 `autodrivedata/map/maptr_official/setup_maptr_official.sh ckpt` 搬进项目。三条线用同一份。
model = dict(pretrained=dict(img=_PROJ + "/outputs/maptr_official/ckpts/resnet50-0676ba61.pth"))
# 读环境变量**不能**写 `import os`:mmcv 的 `Config._substitute_base_vars` 会把配置文件里
# 所有模块级变量 deepcopy 一遍,模块对象不可 pickle → `TypeError: cannot pickle 'module'`。
# 用 `__import__` 现取现用,不往 globals 里留模块对象。

# 分辨率口径(可用 MAPTR_IMG_SCALE 覆盖,便于逐实现钉死同一值):
# 官方 config 写的是 0.5,但那是**针对 1600×900 的 nuScenes**;我们的数据原生 1242×375,
# 再打 0.5 就成了 621×187(比官方训练分辨率低一档,自带系统性劣势)。默认取 **1.0** =
# 原生 1242×375,与自实现基线同分辨率 → 对照里少一个混淆项;官方 0.5 那档只在
# `MAPTR_IMG_SCALE=0.5` 显式指定时使用(留作消融)。
IMG_SCALE = float(__import__("os").environ.get("MAPTR_IMG_SCALE", "1.0"))
TOTAL_EPOCHS = int(__import__("os").environ.get("MAPTR_EPOCHS", "256"))
print(f"[carla-config] 4 类 w/ centerline | IMG_SCALE={IMG_SCALE} total_epochs={TOTAL_EPOCHS}")

# ---- 有效 batch:bs1 × 梯度累积 ACCUM(= 自实现基线末轮的 batch 5)-------------------
# 为什么不用真 bs5:`samples_per_gpu=5` 在 1242×375 上必然 OOM —— 实测 bs2 就要 ~13 GiB
# (本卡可用 11.4 GiB),bs1 峰值 8.1 GiB 是唯一放得下的档。
# 为什么累积 5 次等价 bs5:mmcv 的累积 hook 每步先 `loss = loss / loss_factor` 再 backward
# → 梯度 = ACCUM 个微批的**均值**,与 bs5 单步同式(不是求和)。
# 为什么写 `fp16 = None`:官方 train 脚本在 `fp16_cfg is not None` 时把 hook 类型**硬编码**
# 成 Fp16OptimizerHook(`Fp16OptimizerHook(**cfg.optimizer_config, **fp16_cfg, ...)`),
# 没法从 config 换型;置 None 后走 `else: optimizer_config = cfg.optimizer_config` 分支,
# 类型由本文件决定 → **纯配置**实现累积,官方代码一行未改。fp16 包装不受影响:
# wrap_fp16_model 是 Fp16OptimizerHook.before_run 做的(mmcv/runner/hooks/optimizer.py:187),
# 累积版继承同一个 before_run。
# 已知偏差(账本要写):①fp16 后端由 torch GradScaler 换成 mmcv LossScaler(同为 static
# loss_scale=512 + 溢出跳过);②BN 统计按微批(bs1)而非有效批算 —— mmcv 自己会 warning,
# 但每通道空间样本 2.4 万+,统计上可忽略(本项目基线是真 bs5 的 BN)。
# 口径自证:优化步数 = 数据迭代数 / ACCUM → 256 ep × 200 帧 / 5 = 10,240 步
# = 基线第一轮的 10,240 步(样本 51,200 亦相等);每个 checkpoint 里 AdamW 的 step
# 计数可直接核对(1 epoch 必须恰为 40,跑满 256 ep 收尾时必须恰为 10,240)。
ACCUM = int(__import__("os").environ.get("MAPTR_GRAD_ACCUM", "5"))
fp16 = None
optimizer_config = dict(
    type="GradientCumulativeFp16OptimizerHook",
    grad_clip=dict(max_norm=35, norm_type=2),
    loss_scale=512.0,
    cumulative_iters=ACCUM,
)
print(f"[carla-config] 有效 batch = bs1 × 累积 {ACCUM} 次")

dataset_type = "CustomNuScenesOfflineLocalMapDataset"

# 下面两项官方 config 里同名同值;此处重申只为在本文件的 pipeline 里可引用
# (mmcv 的 _base_ 合并发生在文件 exec **之后**,基类文件里的模块级变量在本文件不可见)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
map_classes = ["divider", "ped_crossing", "boundary", "centerline"]

# 逐行抄官方 train_pipeline,仅三处改动:scales → IMG_SCALE、去掉 LiDAR 深度两步、
# CustomCollect3D 去掉 'gt_depth'(见文件头"唯一的功能性偏差")
train_pipeline = [
    # color_type 显式钉 `color`(理由见 maptr_v1_carla.py 同一处注释):本 env 的 mmdet3d 是
    # MapQR 自带 vendored 副本,loader 默认 'unchanged' → 我们的 RGBA PNG 读成 4 通道,
    # 而 NormalizeMultiviewImage 的 mean 只有 3 通道 → 在线评测的 test_pipeline 一跑就
    # cv2 尺寸不匹配(train 靠 PhotoMetricDistortion 静默吞 alpha 才侥幸通过)。
    # 官方 nuScenes 是 3 通道 JPEG → 钉 color 才回到官方契约,且与自实现基线逐像素同源。
    dict(type="LoadMultiViewImageFromFiles", to_float32=True, color_type="color"),
    dict(type="RandomScaleImageMultiViewImage", scales=[IMG_SCALE]),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", with_gt=False, with_label=False, class_names=map_classes),
    dict(type="CustomCollect3D", keys=["img"]),
]

test_pipeline = [
    # color_type 显式钉 `color`(理由见 maptr_v1_carla.py 同一处注释):本 env 的 mmdet3d 是
    # MapQR 自带 vendored 副本,loader 默认 'unchanged' → 我们的 RGBA PNG 读成 4 通道,
    # 而 NormalizeMultiviewImage 的 mean 只有 3 通道 → 在线评测的 test_pipeline 一跑就
    # cv2 尺寸不匹配(train 靠 PhotoMetricDistortion 静默吞 alpha 才侥幸通过)。
    # 官方 nuScenes 是 3 通道 JPEG → 钉 color 才回到官方契约,且与自实现基线逐像素同源。
    dict(type="LoadMultiViewImageFromFiles", to_float32=True, color_type="color"),
    dict(type="RandomScaleImageMultiViewImage", scales=[IMG_SCALE]),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1242, 375),  # 官方写 (1600, 900);本链里没有 Resize 变换 → 该值不参与计算
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="PadMultiViewImage", size_divisor=32),
            dict(
                type="DefaultFormatBundle3D",
                with_gt=False,
                with_label=False,
                class_names=map_classes,
            ),
            dict(type="CustomCollect3D", keys=["img"]),
        ],
    ),
]

# 只列**要改**的键:其余(dict、pc_range、bev_size、fixed_ptsnum_per_line、
# eval_use_same_gt_sample_num_flag、aux_seg…)由 _base_ 深合并保留 = 官方原值
data = dict(
    samples_per_gpu=1,  # 有效 batch 靠梯度累积(见文件上方 ACCUM 块),显存按 bs1
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "carla_infos_train.pkl",
        pipeline=train_pipeline,
        map_classes=map_classes,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "carla_infos_val.pkl",
        map_ann_file=data_root + "carla_map_anns_val.json",
        pipeline=test_pipeline,
        map_classes=map_classes,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "carla_infos_val.pkl",
        map_ann_file=data_root + "carla_map_anns_val.json",
        pipeline=test_pipeline,
        map_classes=map_classes,
    ),
)

# 官方 24ep → 我们 256ep:CosineAnnealing 随 total_epochs 自适应,warmup_iters 保持官方 500
total_epochs = TOTAL_EPOCHS
runner = dict(type="EpochBasedRunner", max_epochs=TOTAL_EPOCHS)
# 每 64 ep 一次在线评测 → 256 ep 共 4 次(官方是每 2 ep;照官方跑是 128 次,纯浪费),
# best 权重按官方同一指标存
evaluation = dict(
    interval=64,
    pipeline=test_pipeline,
    metric="chamfer",
    save_best="NuscMap_chamfer/mAP",
    rule="greater",
)
checkpoint_config = dict(interval=64, max_keep_ckpts=3)
# 日志:官方 base 带 TextLoggerHook + TensorboardLoggerHook,这里**去掉 tensorboard**。
# 纯环境问题(与训练口径无关):torch 1.9 的 `torch/utils/tensorboard/__init__.py:4` 是
# `import distutils` 之后直接取 `distutils.version.LooseVersion`,而本 env 里 `distutils`
# 被 setuptools 75 的 `_distutils` 顶替、该 submodule 不会被隐式导入 →
# `AttributeError: module 'distutils' has no attribute 'version'`,hook 的 before_run 一抛,
# **训练根本起不来**(实测 `tools/train.py` 卡死在这一步)。TextLoggerHook 每 50 iter 落
# loss,足够对账;要曲线就从日志解析(本项目口径)。改口径不许顺手把它加回来。
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
seed = 0

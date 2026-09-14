"""MapQR(SGQ scatter-and-gather query + height kernel attention)在 CARLA 数据上的官方复线配置。

口径与 `maptrv2_carla.py` 完全一致(数据/步数/分辨率/继承策略/唯一偏差/启动方式都相同),
差异只有两点:

1. `_base_` 换成 MapQR 自己的 config —— 它相对 MapTRv2 的贡献全在仓库内:
   新增 `maptr/modules/height_kernel_attention.py`(BEVFormer 编码层用 `HeightKernelAttention`
   替 `SpatialCrossAttention` = "GKT-h"),加 SGQ 的 query 设计(`query_embed_type='instance'`
   + `InstancePointAttention`)。这些全部由 base 继承,**本文件一行模型代码都不碰**。
2. **类表保持官方 3 类**(divider/ped_crossing/boundary):MapQR 仓库只有 3 类 config,
   而 base 里 `num_map_classes = len(map_classes)` 是**在 base 文件里算好的** —— 若在此把
   `map_classes` 改成 4,head 的 `num_classes` 仍是 3,会静默错配。故不动(4 类对照看
   MapTRv2 那一行;三方共享 3 类对照看汇总表)。

启动(同 v2,换仓库与配置名):

    cd /root/autodl-tmp/Documents/Projects/AutoDriveData/hdMapGitHub/MapQR && \
      PYTHONPATH=$PWD:/root/autodl-tmp/Documents/Projects/AutoDriveData \
      python tools/train.py \
        /root/autodl-tmp/Documents/Projects/AutoDriveData/maptr_official/configs/mapqr_carla.py
"""

_base_ = [
    "/root/autodl-tmp/Documents/Projects/AutoDriveData/hdMapGitHub/MapQR/projects/configs/"
    "mapqr/mapqr_nusc_r50_24ep.py",
]

_PROJ = "/root/autodl-tmp/Documents/Projects/AutoDriveData"
data_root = _PROJ + "/outputs/maptr_official/data/"
work_dir = _PROJ + "/outputs/maptr_official/mapqr"  # 权重/日志落项目内(用户要求)

# 主干预训练权重:官方 base 指的是**相对 cwd** 的 'ckpts/resnet50-19c8e357.pth',而三个官方
# 仓库都没有 ckpts/ 目录(README 让用户自己放)→ 不覆盖就是构建即失败(该值会被
# MVXTwoStageDetector 转成 `img_backbone.init_cfg`,路径不存在直接 IOError)。
# 口径:不往 hdMapGitHub 里写任何文件 → 改用项目内绝对路径。
# 权重本体 = torchvision 的 ImageNet ResNet-50;本机 torch hub 缓存里是**新版文件命名**
# `resnet50-0676ba61.pth`(torchvision 换过下载 URL,哈希后缀随之变,内容同为
# IMAGENET1K_V1),由 `bin/setup_maptr_official.sh ckpt` 搬进项目。三条线用同一份。
model = dict(pretrained=dict(img=_PROJ + "/outputs/maptr_official/ckpts/resnet50-0676ba61.pth"))
# 读环境变量**不能**写 `import os`:mmcv 的 `Config._substitute_base_vars` 会把配置文件里
# 所有模块级变量 deepcopy 一遍,模块对象不可 pickle → `TypeError: cannot pickle 'module'`。
# 用 `__import__` 现取现用,不往 globals 里留模块对象。

# 分辨率口径同 maptrv2_carla.py(默认原生 1.0;官方 0.5 是给 1600×900 的 nuScenes 用的)
IMG_SCALE = float(__import__("os").environ.get("MAPTR_IMG_SCALE", "1.0"))
TOTAL_EPOCHS = int(__import__("os").environ.get("MAPTR_EPOCHS", "256"))
print(f"[carla-config] MapQR 3 类 | IMG_SCALE={IMG_SCALE} total_epochs={TOTAL_EPOCHS}")

# ---- 有效 batch:bs1 × 梯度累积 ACCUM(= 自实现基线末轮的 batch 5)-------------------
# 与 maptrv2_carla.py 同一套口径与理由,要点重复一遍便于单文件阅读:
# ①`bs5` 在 12GB 卡上放不下(bs2@1242×375 实测就要 ~13 GiB,可用 11.4 GiB;bs1 峰值 8.1 GiB);
# ②累积 5 次在数值上等价 bs5 —— mmcv 的累积 hook 每步 `loss = loss / loss_factor` 后
#   backward,梯度是 ACCUM 个微批的**均值**;
# ③`fp16 = None` 是为了绕开官方 train 脚本把 hook 类型硬编码成 Fp16OptimizerHook 的那条
#   分支(`fp16_cfg is not None`),置 None 后走 `else: optimizer_config = cfg.optimizer_config`,
#   类型由本文件决定 → 纯配置实现累积,官方代码一行未改;fp16 包装仍由
#   Fp16OptimizerHook.before_run 里的 wrap_fp16_model 完成,累积版继承同一 before_run;
# ④已知偏差:fp16 后端从 torch GradScaler 换成 mmcv LossScaler(同为 static 512 + 溢出跳过)、
#   BN 统计按微批算而非有效批(每通道空间样本 2.4 万+,可忽略)。
# 口径自证:256 ep × 200 帧 / 5 = 10,240 优化步 = 基线第一轮的 10,240 步
# (样本 51,200 亦相等);1 epoch 的 checkpoint 里 AdamW step 计数必须恰为 40,
# 跑满 256 ep 收尾时必须恰为 10,240。
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

# 官方 config 里同名同值;此处重申只为在本文件的 pipeline 里可引用(见 v2 文件的说明)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
map_classes = ["divider", "ped_crossing", "boundary"]

# 逐行抄官方 train_pipeline,仅三处改动:scales → IMG_SCALE、去掉 LiDAR 深度两步、
# CustomCollect3D 去掉 'gt_depth'(我们无 LiDAR;官方 detector 在 gt_depth=None 时跳过 depth loss)
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

total_epochs = TOTAL_EPOCHS
runner = dict(type="EpochBasedRunner", max_epochs=TOTAL_EPOCHS)
evaluation = dict(
    interval=64,
    pipeline=test_pipeline,
    metric="chamfer",
    save_best="NuscMap_chamfer/mAP",
    rule="greater",
)
checkpoint_config = dict(interval=64, max_keep_ckpts=3)
# 日志:同 maptrv2_carla.py —— 去掉 TensorboardLoggerHook(torch 1.9 的 shim 在本 env 下
# `AttributeError: module 'distutils' has no attribute 'version'`,训练起不来);理由见那份文件。
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
seed = 0

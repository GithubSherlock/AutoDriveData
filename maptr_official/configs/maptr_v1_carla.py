"""MapTR **v1**(3 类:divider / ped_crossing / boundary)在 CARLA 数据上的官方复线配置。

与前两份(v2 / MapQR)同数据、同步数、同分辨率口径;唯一不同的是**数据源接法**:

- v2/MapQR 有官方**离线**数据集(`nuscenes_offlinemap_dataset.py`),直吃 infos 里的
  `annotation`,零适配 → 配置里只是换 `data_root`/`ann_file`。
- v1 主分支**只有上线数据集** `CustomNuScenesLocalMapDataset`:它的 `VectorizedLocalMap`
  在 `__init__` 构造 4 张 `NuScenesMap`(需 `maps/expansion/*.json`,`bin/prepare_official_dataset.py`
  已生成空桩),训练时按位姿查地图库并做全局→ego 变换。故这里用项目侧子类
  `CarlaNuScenesLocalMapDataset`(`maptr_official/bridge.py`)把 `self.vector_map` 换成
  "按位姿查我们 infos 的 annotation"的替身 —— 官方代码一行未改,只换数据源。

继承策略同 v2:模型/损失/优化器/主干预训练全部继承官方 config。v1 是检测式 pipeline,
本来就没有 LiDAR 深度两步(v1 无 `gt_depth`);**pipeline 要显式覆盖**,因为 v1 官方那份
写死 `scales=[0.5]`,不覆盖就会比另两行低一档分辨率(见文件内 pipeline 处的说明)。

类表保持官方 3 类;v1 只比共享 3 类(4 类对照由 MapTRv2 那条线覆盖)。
预算口径(2026-09-14 探针,用户拍板):**128 ep**,仅 MapTR v1 一条线;评测走**在线评测**
(EvalHook,与独立 test 已验逐位一致,不需单独 test);GPU 利用 = **bs2 × 梯度累积 2**
= 有效 bs4 —— 这是对原协议 `bs1 × 累积 5`(有效 bs5,为了对齐基线末轮)的**主动偏离**:
v1@bs1 峰值仅 5077 MiB,12 GB 卡(可用 11.63 GiB)升到 bs2 把显存用足(期望 ~9-10 GiB,
OOM 则回退原协议,其余不变)。代价 = 有效 batch 5→4、步数 40→50 步/epoch,但探针判据是
**与官方 v1 是否同量级**(见 Plan.md §5.12 探针记录),batch 差一档可接受,已写死进本文件。
自证:1 epoch(100 iters)后 checkpoint 里 AdamW step 必须恰为 **100 ÷ 2 = 50**。
对账锚 = A′ 的官方 `eval_map` **3 类** 0.0557(自实现 ep512 preds;4 类的 0.0699 不适用,
因为本线只训 3 类)。"bs2 = 51,200 步"的旧记法已被 `outputs/maptr_ep512.pt.opt` 的
AdamW step=10,242 证伪,详见 maptrv2_carla.py 文件头。

启动(**必须仓库根 + PYTHONPATH 含仓库根与项目根**;`custom_imports` 会先导入
`maptr_official.bridge`,它内部按 `projects.mmdet3d_plugin...` 绝对包名导入官方模块,
所以仓库根必须在 PYTHONPATH 上):

    cd /root/autodl-tmp/Documents/Projects/AutoDriveData/hdMapGitHub/MapTR && \
      PYTHONPATH=$PWD:/root/autodl-tmp/Documents/Projects/AutoDriveData \
      python tools/train.py \
        /root/autodl-tmp/Documents/Projects/AutoDriveData/maptr_official/configs/maptr_v1_carla.py
"""

_base_ = [
    "/root/autodl-tmp/Documents/Projects/AutoDriveData/hdMapGitHub/MapTR/projects/configs/"
    "maptr/maptr_tiny_r50_110e.py",
]

_PROJ = "/root/autodl-tmp/Documents/Projects/AutoDriveData"
data_root = _PROJ + "/outputs/maptr_official/data/"
work_dir = _PROJ + "/outputs/maptr_official/maptr_v1"  # 权重/日志落项目内(用户要求)

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

# 分辨率/步数口径同另两份(v1 官方 pipeline 本来就写死 scales=[0.5] → 本文件**显式覆盖**
# train/val/test 的 pipeline 尺度,否则 v1 那一行会比另两行低一档分辨率)
IMG_SCALE = float(__import__("os").environ.get("MAPTR_IMG_SCALE", "1.0"))
TOTAL_EPOCHS = int(__import__("os").environ.get("MAPTR_EPOCHS", "128"))
print(f"[carla-config] MapTR v1 3 类 | IMG_SCALE={IMG_SCALE} total_epochs={TOTAL_EPOCHS}")

# ---- 有效 batch:bs2 × 梯度累积 ACCUM = 有效 4(2026-09-14 探针口径,见文件头)-------
# 原协议 `bs1 × 累积 5`(有效 5,为对齐自实现基线末轮)被探针主动偏离:
# ① 显存利用 —— v1@bs1 峰值仅 5077 MiB,12 GB 卡升到 bs2 把 ~11 GiB 用足,期望 ~9-10 GiB,
#    OOM 则回退 bs1 × 累积 5(此时步数回 40/epoch),其余不变;
# ② mmcv 累积 hook 每步 `loss = loss / loss_factor` 再 backward → 梯度 = ACCUM 个微批均值,
#    与真 bs4 单步同式(不是求和);
# ③ `fp16 = None` 仅为绕开官方 train 脚本里 hook 类型被硬编码的那条分支(见 maptrv2_carla.py
#    同一段),纯配置实现,官方代码一行未改;fp16 包装仍走 Fp16OptimizerHook.before_run;
# ④ 已知偏差:fp16 后端 GradScaler → mmcv LossScaler(同为 static 512 + 溢出跳过);
#    BN 统计按微批而非有效批。
# 自证:1 epoch(100 iters)后 checkpoint 里 AdamW step 必须恰为 100 ÷ 2 = **50**。
ACCUM = int(__import__("os").environ.get("MAPTR_GRAD_ACCUM", "2"))
fp16 = None
optimizer_config = dict(
    type="GradientCumulativeFp16OptimizerHook",
    grad_clip=dict(max_norm=35, norm_type=2),
    loss_scale=512.0,
    cumulative_iters=ACCUM,
)
print(f"[carla-config] 有效 batch = bs2 × 累积 {ACCUM} 次 = 有效 {2 * ACCUM}")

# 项目侧数据源桥(v1 专用;见 maptr_official/bridge.py)。allow_failed_imports=False:
# 桥没接上就必须**立刻**炸,不能退化成"官方数据集去查空地图"这种静默错误。
custom_imports = dict(imports=["maptr_official.bridge"], allow_failed_imports=False)
dataset_type = "CarlaNuScenesLocalMapDataset"

# 只列要改的键:data.train/val/test 的 map_classes、fixed_ptsnum_per_line、pc_range、
# bev_size… 由 _base_ 深合并保留(官方原值)
#
# pipeline **必须显式覆盖**:v1 官方那份写死 `RandomScaleImageMultiViewImage(scales=[0.5])`,
# 不覆盖就会比另两行低一档分辨率(见文件头"分辨率/步数口径")。逐行抄官方,只改 scales;
# 检测那几步(LoadAnnotations3D/ObjectRangeFilter/ObjectNameFilter/DefaultFormatBundle3D)
# **原样保留** —— 空检测 GT 也走得通(ann_info 里 gt_boxes/gt_names/valid_flag 都是等长空
# ndarray),且真正的 GT 在 `vectormap_pipeline` 里被整体覆写成地图矢量。
# 下面三项 base 里同名同值;此处重申只为在本文件的 pipeline 里可引用
# (mmcv 的 _base_ 合并发生在文件 exec **之后**,基类文件里的模块级变量在本文件不可见)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

train_pipeline = [
    # color_type 必须显式钉 `color`(2026-09-13 在线评测探针实测):本 env 的 mmdet3d 是
    # MapQR 自带的 vendored 副本(`hdMapGitHub/MapQR/mmdetection3d`),其 loader 默认
    # `color_type='unchanged'`(IMREAD_UNCHANGED),而我们的导出图是 **RGBA PNG** → 读成
    # 4 通道;`NormalizeMultiviewImage` 的 mean 只有 3 通道 → `cv2.subtract` 抛尺寸不匹配。
    # train 侥幸能跑只因 `PhotoMetricDistortion` 在前(其 bgr2hsv 静默吞掉 alpha),
    # test_pipeline 没有它 → **在线评测一开就炸**(epoch 64 才发现的话白等 63 轮)。
    # 官方 nuScenes 图是 3 通道 JPEG → 3 通道才是官方契约;钉 color = IMREAD_COLOR,
    # 与自实现基线(cv2.imread 默认口径)逐像素同源,alpha 本来就无信息(仅丢弃)。
    dict(type="LoadMultiViewImageFromFiles", to_float32=True, color_type="color"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="RandomScaleImageMultiViewImage", scales=[IMG_SCALE]),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="CustomCollect3D", keys=["gt_bboxes_3d", "gt_labels_3d", "img"]),
]

test_pipeline = [
    # color_type 必须显式钉 `color`(2026-09-13 在线评测探针实测):本 env 的 mmdet3d 是
    # MapQR 自带的 vendored 副本(`hdMapGitHub/MapQR/mmdetection3d`),其 loader 默认
    # `color_type='unchanged'`(IMREAD_UNCHANGED),而我们的导出图是 **RGBA PNG** → 读成
    # 4 通道;`NormalizeMultiviewImage` 的 mean 只有 3 通道 → `cv2.subtract` 抛尺寸不匹配。
    # train 侥幸能跑只因 `PhotoMetricDistortion` 在前(其 bgr2hsv 静默吞掉 alpha),
    # test_pipeline 没有它 → **在线评测一开就炸**(epoch 64 才发现的话白等 63 轮)。
    # 官方 nuScenes 图是 3 通道 JPEG → 3 通道才是官方契约;钉 color = IMREAD_COLOR,
    # 与自实现基线(cv2.imread 默认口径)逐像素同源,alpha 本来就无信息(仅丢弃)。
    dict(type="LoadMultiViewImageFromFiles", to_float32=True, color_type="color"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1242, 375),  # 官方写 (1600, 900);本链无 Resize 变换 → 不参与计算
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="RandomScaleImageMultiViewImage", scales=[IMG_SCALE]),
            dict(type="PadMultiViewImage", size_divisor=32),
            dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(type="CustomCollect3D", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2,  # 探针口径:bs2 × 累积 2 = 有效 bs4(显存用足;OOM 回退 1×5,见文件头)
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "carla_infos_train.pkl",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "carla_infos_val.pkl",
        map_ann_file=data_root + "carla_map_anns_val.json",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "carla_infos_val.pkl",
        map_ann_file=data_root + "carla_map_anns_val.json",
        pipeline=test_pipeline,
    ),
)

total_epochs = TOTAL_EPOCHS
runner = dict(type="EpochBasedRunner", max_epochs=TOTAL_EPOCHS)
# 在线评测:间隔/存盘外**必须显式给 pipeline** —— 官方 base 的 evaluation.pipeline 里写着
# `RandomScaleImageMultiViewImage(scales=[0.5])`,不覆盖就会"训练在 1.0、在线评测在 0.5"
# (best 权重按失真的指标选)。v2/MapQR 同样显式传了 test_pipeline。
# 探针口径:128 ep → 每 32 ep 一次在线评测,共 4 次(32/64/96/128),顺带看曲线是否仍陡升
evaluation = dict(
    interval=32,
    pipeline=test_pipeline,
    metric="chamfer",
    save_best="NuscMap_chamfer/mAP",
    rule="greater",
)
checkpoint_config = dict(interval=32, max_keep_ckpts=3)
# 日志:同 maptrv2_carla.py —— 去掉 TensorboardLoggerHook(torch 1.9 的 shim 在本 env 下
# `AttributeError: module 'distutils' has no attribute 'version'`,训练起不来);理由见那份文件。
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
find_unused_parameters = True  # 空检测 GT 下可能有未参与 loss 的分支(非 DDP 时无副作用)
seed = 0

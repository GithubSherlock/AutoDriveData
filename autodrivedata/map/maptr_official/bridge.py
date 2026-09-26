"""MapTR **v1** 的 CARLA 数据源桥(项目侧;官方仓库保持 pristine)。

**为什么只有 v1 需要它**:v2/MapQR 有官方**离线**数据集(`nuscenes_offlinemap_dataset.py`),
`get_data_info` 把 `info['annotation']` 原样交给 `VectorizedLocalMap.gen_vectorized_samples`,
零适配。v1 主分支只有**上线**版 `CustomNuScenesLocalMapDataset`,其 `VectorizedLocalMap`
在 `__init__` 就构造 4 张 `NuScenesMap`(需 `maps/expansion/*.json`),训练时按位姿查地图库
并做全局→ego 变换。我们的 GT 已经是 ego 系、已裁窗、每线 20 点,所以只需要替换"地图库"这一层。

两个类:

- `CarlaVectorMap`:`gen_vectorized_samples(location, lidar2global_translation,
  lidar2global_rotation)` 同签名同返回 `dict(gt_vecs_pts_loc=LiDARInstanceLines,
  gt_vecs_label=[...])`),但按**位姿**查 infos 的 `annotation`,不查地图 expansion。
  采样/定长/填充仍交给官方 `LiDARInstanceLines` → 与官方同口径。
  类标号 = `map_classes` 里的位置。这与官方一致:v1 `CLASS2LABEL` 是
  `{road_divider:0, lane_divider:0, ped_crossing:1, contours:2}`(注意 boundary 那类在
  码里叫 contours),落到 config 类序 `['divider','ped_crossing','boundary']` 恰是 0/1/2;
  v2 是 `{divider:0, ped_crossing:1, boundary:2, centerline:3}` —— 两边都等于类序。
- `CarlaNuScenesLocalMapDataset`:`CustomNuScenesLocalMapDataset` 子类,`super().__init__`
  照常跑(构造期的**空地图桩**由 `autodrivedata/map/prepare_official_dataset.py` 生成,空地图从不被查询),
  随后把 `self.vector_map` 换成 `CarlaVectorMap`。

**位姿查表为什么要"复刻官方那两步"**:官方 `vectormap_pipeline` 只把位姿传下来,且
`lidar2global = ego2global @ lidar2ego` —— 我们 infos 里 `lidar2ego` 恒等,故平移**逐位**
等于 `ego2global_translation`;旋转则是 `Quaternion(matrix=ego2global).q` → 再被
`gen_vectorized_samples` 转成 yaw。建索引时复用同一条链(同一个 pyquaternion),
两侧对同一帧必然算出同一个键;键按 6 位小数取整以吸收最后几位的重建误差。查不到时退回
最近邻并要求 max|Δ| < 1e-3(米/弧度),否则**直接报错** —— 静默错帧会让 GT 与图像错配,
这种错误在训练曲线上看不出来,不能容忍。
"""

from __future__ import annotations

import numpy as np
from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from projects.mmdet3d_plugin.datasets.nuscenes_map_dataset import (
    CustomNuScenesLocalMapDataset,
    LiDARInstanceLines,
)
from shapely.geometry import LineString

# 官方 `VectorizedLocalMap.__init__` 的默认采样口径:数据集只传 patch_size / map_classes /
# fixed_ptsnum_per_line / padding_value,其余走默认值 —— 这里显式写出来,不靠"官方默认没变"。
SAMPLE_DIST = 1
NUM_SAMPLES = 250
PADDING = False
KEY_DECIMALS = 6
POSE_TOL = 1e-3


def _pose_matrix(translation, quaternion) -> np.ndarray:
    """(平移, 四元数 [w,x,y,z]) → 4×4 位姿(官方 `vectormap_pipeline` 的同一构造)。"""
    m = np.eye(4)
    m[:3, :3] = Quaternion(quaternion).rotation_matrix
    m[:3, 3] = translation
    return m


def _yaw_official(matrix: np.ndarray) -> float:
    """位姿矩阵 → yaw,复刻官方**两步**:`Quaternion(matrix=…).q` → `Quaternion(list).yaw`。

    官方的 q 先被 `vectormap_pipeline` 转成 list(`list(Quaternion(matrix=lidar2global).q)`),
    再在 `gen_vectorized_samples` 里重新包成 `Quaternion` —— 这里少走一步会引入 ~1e-16 的
    差异(不影响 6 位取整,但"同一条链"要能逐字对上账)。
    """
    return quaternion_yaw(Quaternion(Quaternion(matrix=matrix).q))


def _key(translation, yaw: float) -> np.ndarray:
    x, y, z = (float(v) for v in translation[:3])
    return np.array([x, y, z, float(yaw)], dtype=np.float64)


def _same_annotation(a: dict, b: dict) -> bool:
    """两份 `annotation` 是否逐类、逐线、逐点全等。

    不能直接用 `==`:里面是 ndarray,`bool(array)` 会抛 ValueError(歧义真值)。
    """
    if set(a) != set(b):
        return False
    for cls, lines in a.items():
        other = b[cls]
        if len(lines) != len(other):
            return False
        for l1, l2 in zip(lines, other):  # noqa: B905 —— py3.8 无 strict=(本文件要在官方 env 跑)
            if np.shape(l1) != np.shape(l2) or not np.array_equal(l1, l2):
                return False
    return True


class CarlaVectorMap:
    """`VectorizedLocalMap` 的 CARLA 替身:位姿 → 本帧 `annotation`(ego 系,20 点/线)。"""

    def __init__(
        self,
        data_infos: list[dict],
        map_classes: list[str],
        patch_size: tuple[float, float],
        fixed_num: int,
        padding_value: float,
    ) -> None:
        if not data_infos:
            raise ValueError("data_infos 为空 —— infos pkl 没加载上,不可能是空数据集")
        self.vec_classes = list(map_classes)
        self.patch_size = patch_size
        self.fixed_num = fixed_num
        self.padding_value = padding_value
        missing = [c for c in self.vec_classes if c not in data_infos[0]["annotation"]]
        if missing:
            raise ValueError(
                f"infos 的 annotation 缺类 {missing}(现有关键字 {sorted(data_infos[0]['annotation'])}"
            )

        self._annotations = [info["annotation"] for info in data_infos]
        self._locations = [info["map_location"] for info in data_infos]
        self._keys = np.stack(
            [
                _key(
                    info["ego2global_translation"],
                    _yaw_official(_pose_matrix(info["ego2global_translation"], info["ego2global_rotation"])),
                )
                for info in data_infos
            ]
        )
        rounded = self._keys.round(KEY_DECIMALS)
        # 同位姿帧的处理(2026-09-13):留出集末尾有 16 帧停在同一位置(采集车停下不动),
        # 位姿键完全相同。原先一见重复就炸,结果**开验证连数据集都构造不出来**。
        # 判定准则:位姿查表"无歧义"的充要条件是**查出来的 GT 一样** —— 静态地图 + 同一位姿
        # ⇒ ego 系 GT 必然逐点相同(实测这 16 帧 1e-9 全等)。故同键帧只要 annotation 与
        # map_location 全等就放行(取首个为代表),有一帧不同才真歧义、照旧报错。
        # 注:同位姿帧的**图像**各不相同(不同时刻),但 GT 只由位姿决定 → 取谁都一样。
        groups: dict[tuple, list[int]] = {}
        for i, r in enumerate(rounded):
            groups.setdefault(tuple(r), []).append(i)
        for key, idxs in groups.items():
            if len(idxs) == 1:
                continue
            head = idxs[0]
            for j in idxs[1:]:
                if self._locations[j] != self._locations[head] or not _same_annotation(
                    self._annotations[j], self._annotations[head]
                ):
                    raise ValueError(
                        f"位姿相同的帧 {idxs} 的 GT 不一致 → 位姿查表必然歧义,请改用帧序接口"
                        f"(位姿键 {list(key)})"
                    )
        self._exact = {tuple(r): i for i, r in enumerate(rounded)}
        self.n_lookup = 0
        self.n_fallback = 0

    def _frame_index(self, translation, rotation) -> int:
        """查帧号:先精确键,再最近邻(要求足够近,否则报错而不是猜)。"""
        self.n_lookup += 1
        key = _key(translation, quaternion_yaw(Quaternion(rotation)))
        hit = self._exact.get(tuple(key.round(KEY_DECIMALS)))
        if hit is not None:
            return hit
        d = np.abs(self._keys - key).max(axis=1)
        j = int(np.argmin(d))
        if d[j] > POSE_TOL:
            raise ValueError(f"位姿查不到帧:{key.tolist()},最近帧偏差 {d[j]:.3e} > {POSE_TOL:.0e}")
        self.n_fallback += 1
        if self.n_fallback == 1:
            print(f"[carla-bridge] 首次走最近邻(浮点边界):帧 {j},偏差 {d[j]:.2e}")
        return j

    def gen_vectorized_samples(self, location, lidar2global_translation, lidar2global_rotation):
        """官方同名方法的数据源替换(返回结构与官方逐字段一致)。"""
        idx = self._frame_index(lidar2global_translation, lidar2global_rotation)
        if location != self._locations[idx]:
            raise ValueError(f"map_location 不一致:{location!r} != 帧 {idx} 的 {self._locations[idx]!r}")
        ann = self._annotations[idx]
        geoms, labels = [], []
        for label, cls_name in enumerate(self.vec_classes):
            for line in ann[cls_name]:
                geoms.append(LineString(np.asarray(line, dtype=np.float64)))
                labels.append(label)
        gt_instance = LiDARInstanceLines(
            geoms,
            SAMPLE_DIST,
            NUM_SAMPLES,
            PADDING,
            self.fixed_num,
            self.padding_value,
            patch_size=self.patch_size,
        )
        return dict(gt_vecs_pts_loc=gt_instance, gt_vecs_label=labels)


@DATASETS.register_module()
class CarlaNuScenesLocalMapDataset(CustomNuScenesLocalMapDataset):
    """v1 上线数据集 + CARLA 数据源(见模块 docstring;官方代码一行未改)。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vector_map = CarlaVectorMap(
            data_infos=self.data_infos,
            map_classes=self.MAPCLASSES,
            patch_size=self.patch_size,
            fixed_num=self.fixed_num,
            padding_value=self.padding_value,
        )
        print(
            f"[carla-bridge] {len(self.data_infos)} 帧 | 类序 {self.MAPCLASSES} | "
            f"patch_size={self.patch_size} | 定长 {self.fixed_num} 点"
        )

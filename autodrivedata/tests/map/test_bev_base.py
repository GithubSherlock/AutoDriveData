"""BEV 底图的**外参自证**:两条换算链必须把"传感器原点"映回**已验证的表**。

## 为什么值得单独钉

底图是"给人看的上下文层",**错了不会报错、不会崩,只是整张图悄悄偏 1–3 m** ——
而偏移在散点图上看不出来(此路无 GT 可对、无判据可跑)。这与本项目已记录的
「yaw≈0 的相机看着正常」「漏掉 ego→world 只有在非恒等位姿时才错」是同一族:
**判据必须是数值对表,不能是目检**。

对表的两侧**都是已经过更高阶验收的常量**:
- LiDAR → `carla_common.SENSOR_OFFSET`(采集器实挂位姿;上一轮刚因它写错而重采)
- Radar → `NUS_RADAR_MOUNTS_CARLA`(由官方 `NUS_RADAR_OFFSETS` 经
  `nus_mount_to_carla` 导出,过 `verify_nus_calib` 十条判据)

只要本模块换算出这两个原点,**两条链的平移部分就被钉死**;再叠一个"非原点处也一致"
的用例,把旋转部分也钉住。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autodrivedata.gt.export.nuscenes import NUS_RADAR_CHANNELS, NUS_RADAR_MOUNTS_CARLA, NUS_RADAR_OFFSETS
from autodrivedata.map import bev_base
from autodrivedata.perception.radar import RADAR_NUS_FIELDS


def _origin_of_sensor(chain, channel=None) -> np.ndarray:
    """把"传感器自身原点"喂进换算链,得到它在 B 系里的位置。

    传感器原点 = 该传感器自身系里的 `(0,0,0)` —— 喂零向量即可。
    """
    if channel is None:
        return np.asarray(chain(np.zeros((1, 4), dtype=np.float32))[0][:, :3])[0]
    zeros = np.zeros((1, len(RADAR_NUS_FIELDS)), dtype=np.float32)
    return np.asarray(chain(zeros, channel))[0]


class TestLidarChain:
    def test_lidar_origin_matches_sensor_offset(self, tmp_path):
        """★ LiDAR 原点换算结果必须 == **B 系的 `SENSOR_OFFSET`**(= y 翻号后的 CARLA 挂点)。

        B 系是 y 左,`SENSOR_OFFSET` 是 CARLA actor 系(y 右)⇒ 只 y 分量翻号。
        """
        pytest.importorskip("carla")
        from autodrivedata.sim.carla_common import SENSOR_OFFSET

        # 造一个"点就在 LiDAR 原点"的 bin:`load_lidar_velo` 读不了空文件,故直接调链尾
        f = tmp_path / "training" / "velodyne"
        f.mkdir(parents=True)
        np.zeros((1, 4), dtype=np.float32).tofile(f / "000000.bin")

        got = bev_base.lidar_ego(tmp_path, 0)[0]
        want = np.array(
            [SENSOR_OFFSET.location.x, -SENSOR_OFFSET.location.y, SENSOR_OFFSET.location.z],
            dtype=np.float64,
        )
        np.testing.assert_allclose(got, want, atol=1e-6)

    def test_lidar_lever_is_the_slam_constant(self):
        """杆臂**必须**是 `slam_eval.LIDAR_LEVER` 本体 —— 不是另抄的一份。

        抄一份的症状:改了 SLAM 侧、底图不跟,底图与 SLAM 轨迹差一个杆臂,
        而两边各自看都对。
        """
        from autodrivedata.slam.slam_eval import LIDAR_LEVER

        assert np.array_equal(np.asarray(bev_base.LIDAR_LEVER), np.asarray(LIDAR_LEVER))


class TestRadarChain:
    def test_radar_origin_matches_verified_mount_table(self):
        """★ 每个雷达的原点换算结果必须 == `NUS_RADAR_MOUNTS_CARLA[ch]`(y 翻号)。

        这一步同时钉住三件事:① 传感器 yaw 的符号;② 后轴→车身中点的 `+NUS_EGO_ORIGIN_X`;
        ③ nus 系 y 左 → B 系 y 左**不翻号**。
        """
        for ch in NUS_RADAR_CHANNELS:
            got = _origin_of_sensor(bev_base.radar_ego, ch)
            t = NUS_RADAR_MOUNTS_CARLA[ch]
            want = np.array([t[0], -t[1], t[2]], dtype=np.float64)
            np.testing.assert_allclose(got, want, atol=1e-6, err_msg=f"{ch} 原点对不上")

    def test_radar_orientation_matches_the_spawn_yaw(self):
        """★★ **判据是"朝向 == 采集侧 spawn 用的 yaw"**,不是"看着对不对"。

        ## 这条为什么必须这么写(2026-10-01 真实代价)

        原先只有一条"非原点处也要对",它拿**同一个公式**算期望值 —— 等于把实现的错误
        抄进断言;用的还是 `RADAR_FRONT`(yaw 仅 **0.2°**),而这个坑是**镜像**:
        `R(+yaw_nus)` 与 `R(−yaw_nus)` 在 yaw≈0 时数值上几乎相同 ⇒ 它对符号**完全不敏感**,
        **跑了一版都没红**。

        实证代价(同一帧同时喂两条路,与 CARLA 自己的 `sensor.get_transform()` 比位移):

        | 通道 | `+yaw_nus` | `−yaw_nus` |
        |---|---|---|
        | RADAR_FRONT | 1.18 m | 1.14 m |
        | RADAR_FRONT_LEFT | **44.13 m** | 2.33 m |
        | RADAR_FRONT_RIGHT | **24.36 m** | 2.30 m |
        | RADAR_BACK_LEFT | 2.10 m | 0.62 m |
        | RADAR_BACK_RIGHT | 3.16 m | 2.34 m |

        ⚠️ 而**不能拿"雷达点落在 LiDAR 表面上"当判据**:LiDAR 是 11.7 万点的密云,
        44 m 的位移照样落在"某个"表面附近(旧口径下 FRONT_LEFT 的 ≤1 m 占比 52%,
        **看着完全正常**)。**参考越密,判别力越差。**

        ## 真值

        `radar_yaw_offset_carla(ch)` = 采集侧 spawn 用的 CARLA yaw(纯值,不 import carla)。
        把传感器自身系的 **+x** 喂进 `radar_ego`,在 B 系量出的方向角必须等于它。
        """
        from autodrivedata.gt.export.nuscenes import radar_yaw_offset_carla

        arr = np.zeros((2, len(RADAR_NUS_FIELDS)), dtype=np.float32)
        arr[0, 0] = 10.0
        arr[:, 3], arr[:, 10], arr[:, 11], arr[:, 14] = 0, 1, 3, 0  # devkit 合法值
        for ch in NUS_RADAR_CHANNELS:
            pts = bev_base.radar_ego(arr, ch)
            d = pts[0] - pts[1]
            got = math.degrees(math.atan2(d[1], d[0]))
            assert got == pytest.approx(radar_yaw_offset_carla(ch), abs=1e-4), (
                f"{ch} 朝向与采集侧 spawn 的 yaw 差 {got - radar_yaw_offset_carla(ch):.1f}° —— 左右镜像"
            )

    def test_radar_yaw_rotates_points_not_just_translates(self):
        """**非原点处**也要对 —— 只对原点的话,把 `R_yaw` 整个删掉也照样绿。

        判据:在 RADAR_FRONT 前方 10 m 放一个点,换到 B 系后应当落在 ego 前方约 10 m、
        且横向位移≈0。同时它的**高度**必须原样带过来(z 不参与旋转)。

        ⚠️ 注意这条**按定义**对 yaw 符号不敏感(那正是它抓不到镜像的原因,见上一条)——
        它守的是"有没有旋转、z 有没有被动过",两者分工不同,都要留。
        """
        ch = "RADAR_FRONT"
        arr = np.zeros((1, len(RADAR_NUS_FIELDS)), dtype=np.float32)
        arr[0, 0], arr[0, 1], arr[0, 2] = 10.0, 0.0, 1.0
        t, yaw = NUS_RADAR_OFFSETS[ch]
        got = bev_base.radar_ego(arr, ch)[0]
        want_xy = (t[0] + 10.0 * np.cos(-yaw), t[1] + 10.0 * np.sin(-yaw))
        want_x = want_xy[0] + bev_base.NUS_EGO_ORIGIN_X
        assert got[0] == pytest.approx(want_x, abs=1e-6)
        assert got[1] == pytest.approx(want_xy[1], abs=1e-6)
        assert got[2] == pytest.approx(t[2] + 1.0, abs=1e-6), "z 不该被旋转/平移错"

    def test_side_radar_origin_differs_from_front(self):
        """反向对照:不同通道的结果**必须不同** —— 防"常量表被写成一维、所有雷达同解"。"""
        vals = {
            ch: tuple(np.round(_origin_of_sensor(bev_base.radar_ego, ch), 6)) for ch in NUS_RADAR_CHANNELS
        }
        assert len(set(vals.values())) == len(NUS_RADAR_CHANNELS), f"有雷达原点撞车:{vals}"


class TestPcdReader:
    def test_roundtrip_with_writer(self, tmp_path):
        """读侧必须与**写侧同一个 struct** —— 往返逐位相等(字段表漂了就在这里红)。"""
        from autodrivedata.perception.radar import nus18_to_pcd

        arr = np.zeros((3, len(RADAR_NUS_FIELDS)), dtype=np.float32)
        arr[:, 0] = [1.0, 2.0, 3.0]
        arr[:, 1] = [-0.5, 0.0, 0.5]
        arr[:, 3] = 3.0  # dyn_prop(整数字段)
        p = tmp_path / "samples" / "RADAR_FRONT"
        p.mkdir(parents=True)
        (p / "000007.pcd").write_bytes(nus18_to_pcd(arr))

        got = bev_base.load_radar_nus(tmp_path, 7, "RADAR_FRONT")
        np.testing.assert_array_equal(got, arr)

    def test_empty_cloud_decodes_to_zero_points(self, tmp_path):
        """空点云的 devkit 编码是**单点全 NaN** —— 读出来必须是 0 点,不是 1 个 NaN 点。"""
        from autodrivedata.perception.radar import nus18_to_pcd

        p = tmp_path / "samples" / "RADAR_BACK"
        p.mkdir(parents=True)
        (p / "000000.pcd").write_bytes(nus18_to_pcd(np.zeros((0, len(RADAR_NUS_FIELDS)), dtype=np.float32)))
        assert len(bev_base.load_radar_nus(tmp_path, 0, "RADAR_BACK")) == 0

    def test_missing_file_is_empty_not_crash(self, tmp_path):
        assert len(bev_base.load_radar_nus(tmp_path, 99, "RADAR_FRONT")) == 0
        assert len(bev_base.load_lidar_velo(tmp_path, 99)) == 0


class TestGeometry:
    def test_to_world_matches_manual_formula(self):
        """`to_world` 必须与 `stitch_temporal.frame_to_world` 内联的公式**同式**。

        两者若分叉,矢量与底图会落在两张错开的图上 —— 而"两张图各自看都对"。
        """
        pts = np.array([[1.0, 2.0, 0.5], [-3.0, 0.0, -1.0]])
        ego = [10.0, -20.0, 0.0, 37.0, 0.0, 0.0]
        got = bev_base.to_world(pts, ego)
        import math

        c, s = math.cos(math.radians(37.0)), math.sin(math.radians(37.0))
        for (x, y, z), g in zip(pts, got, strict=True):
            assert g[0] == pytest.approx(ego[0] + x * c - y * s, abs=1e-9)
            assert g[1] == pytest.approx(ego[1] + x * s + y * c, abs=1e-9)
            assert g[2] == pytest.approx(z, abs=1e-9)

    def test_px_arr_matches_scalar(self):
        """`Px.arr`(批量)与 `Px.__call__`(标量)必须逐点一致 —— 否则"图对了数不对"。"""
        px = bev_base.Px(sx=2.0, sy=-3.0, ox=5.0, oy=7.0)
        pts = np.array([[0.0, 0.0], [1.5, -2.5], [100.0, 40.0]])
        batch = px.arr(pts)
        for (x, y), b in zip(pts, batch, strict=True):
            assert b[0] == pytest.approx(px(x, y)[0], abs=1e-9)
            assert b[1] == pytest.approx(px(x, y)[1], abs=1e-9)

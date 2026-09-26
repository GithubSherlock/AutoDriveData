"""nuScenes `sensor` / `calibrated_sensor` 表读取 + 每传感器相对 ego 的位姿/方位打印。

**回答什么**:每张表里每个传感器相对 **ego(自车)系** 的安装位姿是什么、朝哪个方向
(方位角 az / 俯仰角 el)。可直接跑(打印),也可被 pytest 收集(无 devkit/dataroot 自动跳过)。

用法:
  python tests/test_nuscenes_cali_sensors.py                    # 自动挑 dataroot
  python tests/test_nuscenes_cali_sensors.py --dataroot <路径>
  python tests/test_nuscenes_cali_sensors.py --version v1.0-trainval
  pytest tests/test_nuscenes_cali_sensors.py -q

--- 坑 1:sensor 自身坐标系**不是统一约定**,视线轴因 modality 而异 ---

| modality | 自身系约定 | 视线轴(boresight) |
|---|---|---|
| camera | x 右 / y 下 / z 前(OpenCV 系) | **+z** |
| lidar / radar | x 前 / y 左 / z 上(nuScenes 系) | **+x** |

`calibrated_sensor.rotation` 是「自身系 → ego 系」的四元数 (w,x,y,z)。**把它直接喂
devkit 的 `Quaternion.yaw_pitch_roll` 对相机是没有意义的** —— 那三个角在解"自身系 z 轴
被转到了哪里",而相机自身系的 z 轴正是视线轴、x/y 是像素轴,解出来的既不是航向也不是
俯仰:实测 CAM_FRONT 得 44.757/89.541/−134.804,而它的真实方位角是 0.325°。
lidar/radar 恰好可以这样读,因为它们的自身系 x 轴 = 视线轴,与 yaw 的转轴定义重合。
⇒ 本文件一律用 **R @ 自身系轴向量** 求视线轴,再取 az/el,不对相机用 yaw_pitch_roll。

--- 坑 2:同一 channel 可以有多套标定 ---

官方 mini 每通道有 **10** 条 `calibrated_sensor` 记录,但只对应 **2 套不同位姿**:
n015 车(新加坡 3 个 log,每 log 一条 = 6 条)与 n008 车(boston-seaport,4 条)。
"每通道取第一条"会静默拿到其中一套(且是哪套取决于表内顺序);必须按 log 分组才能看出一条
标定属于哪台车 —— `calibrated_sensor` 表**不带**车辆/地点字段,要经
sample_data → sample → scene → log 反查。本仓自产的 `outputs/nus_mini` 每通道只有 1 套。
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autodrivedata.calib.camera_rig import NUS_CAMERA_CALIBS
from autodrivedata.utils.paths import project_path

# 官方 mini(标定权威,优先)→ 本仓自产(只有一套标定,做对照)
OFFICIAL_MINI = Path("/root/autodl-tmp/Documents/datasets/nuscenes_mini")
REPO_MINI = project_path("outputs/nus_mini")
VERSION = "v1.0-mini"

# 自身系 → ego 系四元数 (w,x,y,z)。相机自身系 x右/y下/z前,激光雷达系 x前/y左/z上。
NUS_SENSOR_CONVENTIONS = {
    "camera": "x右 / y下 / z前(视线轴 +z)",
    "lidar": "x前 / y左 / z上(视线轴 +x)",
    "radar": "x前 / y左 / z上(视线轴 +x)",
}


# ---------------------------------------------------------------- 纯值(numpy,无需 devkit)


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    """四元数 (w,x,y,z) → 3×3 旋转矩阵(自身系 → ego 系)。非单位四元数先归一化。"""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0.0:
        raise ValueError(f"零四元数无法归一化: {q!r}")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def sensor_basis(rotation: np.ndarray, modality: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """自身系三轴在 ego 系下的表示 → `(forward, left, up)`(见模块 docstring 坑 1)。

    相机自身系 x右/y下/z前 ⇒ left = −x 列、up = −y 列、forward = +z 列;
    激光/雷达自身系 x前/y左/z上 ⇒ forward/left/up = x/y/z 列。
    """
    r = np.asarray(rotation, dtype=np.float64)
    if modality == "camera":
        return r[:, 2], -r[:, 0], -r[:, 1]
    if modality in ("lidar", "radar"):
        return r[:, 0], r[:, 1], r[:, 2]
    raise ValueError(f"未知 modality: {modality!r}")


def az_el_deg(axis: np.ndarray | Sequence[float]) -> tuple[float, float]:
    """ego 系向量 → (方位角 az, 俯仰角 el),单位度。az 绕 +z(逆时针为正,x 前=0,y 左=+90)。"""
    v = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        raise ValueError("零向量无方位")
    v = v / n
    return float(np.degrees(np.arctan2(v[1], v[0]))), float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))


# ---------------------------------------------------------------- devkit 侧


@dataclass(frozen=True)
class SensorPose:
    """一条 calibrated_sensor 记录 + 其 sensor 表信息 + 反查到的车辆/地点。"""

    channel: str
    modality: str
    sensor_token: str
    calib_token: str
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    camera_intrinsic: tuple[tuple[float, float, float], ...]
    vehicles: tuple[str, ...]  # 反查 sample_data→sample→scene→log 得到的 "车/地点"
    n_calib_for_channel: int  # 该 channel 一共几条 calibrated_sensor 记录

    @property
    def rotation_matrix(self) -> np.ndarray:
        return quat_to_matrix(self.rotation)

    @property
    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return sensor_basis(self.rotation_matrix, self.modality)

    @property
    def az_el(self) -> tuple[float, float]:
        """视线轴在 ego 系下的 (方位角, 俯仰角),度。"""
        return az_el_deg(self.basis[0])


def calib_vehicles(nusc: Any) -> dict[str, set[str]]:
    """calibrated_sensor token → {车/地点}。表里没这字段,只能经 sample_data 链反查。

    一条标定可能被**多个 log** 引用(n015 一套标定服务新加坡 3 个 log),故收全集合。
    """
    out: dict[str, set[str]] = {}
    log_veh: dict[str, str] = {}
    for sd in nusc.sample_data:
        sample = nusc.get("sample", sd["sample_token"])
        scene = nusc.get("scene", sample["scene_token"])
        lt = scene["log_token"]
        if lt not in log_veh:
            log = nusc.get("log", lt)
            log_veh[lt] = f"{log['vehicle']}@{log['location']}"
        out.setdefault(sd["calibrated_sensor_token"], set()).add(log_veh[lt])
    return out


def sensor_poses(nusc: Any) -> list[SensorPose]:
    """全表 → SensorPose 列表,按 (modality, channel, translation) 排序。"""
    sensors = {s["token"]: s for s in nusc.sensor}
    veh = calib_vehicles(nusc)
    per_channel: dict[str, int] = {}
    for c in nusc.calibrated_sensor:
        ch = sensors[c["sensor_token"]]["channel"]
        per_channel[ch] = per_channel.get(ch, 0) + 1

    rows: list[SensorPose] = []
    for c in nusc.calibrated_sensor:
        s = sensors[c["sensor_token"]]
        tx, ty, tz = (float(v) for v in c["translation"])
        qw, qx, qy, qz = (float(v) for v in c["rotation"])
        rows.append(
            SensorPose(
                channel=s["channel"],
                modality=s["modality"],
                sensor_token=s["token"],
                calib_token=c["token"],
                translation=(tx, ty, tz),
                rotation=(qw, qx, qy, qz),
                camera_intrinsic=tuple(
                    (float(r[0]), float(r[1]), float(r[2])) for r in c["camera_intrinsic"]
                ),
                vehicles=tuple(sorted(veh.get(c["token"], ()))),
                n_calib_for_channel=per_channel[s["channel"]],
            )
        )
    rows.sort(key=lambda r: (r.modality, r.channel, r.translation))
    return rows


def load_nusc(dataroot: str | Path, version: str = VERSION) -> Any:
    """构造 devkit NuScenes(惰性 import,缺包时抛 ImportError 由调用方跳过)。"""
    from nuscenes import NuScenes

    return NuScenes(version=version, dataroot=str(dataroot), verbose=False)


def candidate_dataroots() -> list[Path]:
    """候选 dataroot,按优先级:显式 env → 官方 mini → 本仓 outputs/nus_mini。"""
    out: list[Path] = []
    env = os.environ.get("NUSCENES_DATAROOT")
    if env:
        out.append(Path(env))
    out.extend([OFFICIAL_MINI, REPO_MINI])
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve() if p.exists() else p
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def pick_dataroot(explicit: str | None = None) -> Path | None:
    """挑第一个「存在且含 <version>/sensor.json」的候选;全无则 None。"""
    cands = [Path(explicit)] if explicit else candidate_dataroots()
    for p in cands:
        if (p / VERSION / "sensor.json").is_file():
            return p
    return None


# ---------------------------------------------------------------- 打印


def _fmt_vec(v: Sequence[float], prec: int = 4) -> str:
    return "(" + ", ".join(f"{x:.{prec}f}" for x in v) + ")"


def _fmt_veh(vehicles: Sequence[str], width: int = 30) -> str:
    """把 {车@地点} 集合压成一列:同一台车多个地点时只留车名 + log 数。"""
    if not vehicles:
        return "(无引用)".ljust(width)
    names = sorted({v.split("@")[0] for v in vehicles})
    if len(names) == 1:
        s = f"{names[0]}({len(vehicles)} log)" if len(vehicles) > 1 else vehicles[0]
    else:
        s = ",".join(vehicles)
    return s[: width - 1].ljust(width)


def dedup_poses(rows: Sequence[SensorPose]) -> list[SensorPose]:
    """同 (channel, translation, rotation, intrinsic) 的重复记录合并成一条。

    表里同一套标定会重复 N 次(每个 log 一条),合并时**并集** `vehicles`
    —— "这套标定服务哪些 log" 正是要看的字段,不能只留第一条。
    """
    merged: dict[tuple, SensorPose] = {}
    for r in rows:
        key = (r.channel, r.translation, r.rotation, r.camera_intrinsic)
        old = merged.get(key)
        if old is None:
            merged[key] = r
        else:
            merged[key] = replace(old, vehicles=tuple(sorted(set(old.vehicles) | set(r.vehicles))))
    return list(merged.values())


def find_pose(rows: Sequence[SensorPose], channel: str, vehicle: str = "") -> SensorPose:
    """按 channel(+ 车名前缀)取唯一一条标定;无匹配 / 多匹配即抛。"""
    hit = [r for r in rows if r.channel == channel and any(v.startswith(vehicle) for v in r.vehicles)]
    if len(hit) != 1:
        raise AssertionError(f"{channel}@{vehicle!r} 命中 {len(hit)} 条(应恰好 1 条)")
    return hit[0]


def print_report(nusc: Any, dataroot: Path, version: str = VERSION) -> list[SensorPose]:
    rows = sensor_poses(nusc)
    uniq = dedup_poses(rows)
    n_ch = len({r.channel for r in rows})
    print(f"\ndataroot = {dataroot}   version = {version}")
    print(
        f"表规模: sensor {len(nusc.sensor)} 条 | calibrated_sensor {len(nusc.calibrated_sensor)} 条 "
        f"| sample_data {len(nusc.sample_data)} 条 | sample {len(nusc.sample)} 条 "
        f"| log {len(nusc.log)} 条"
    )
    print(
        f"去重后: {n_ch} 个通道 / {len(uniq)} 套不同标定(每条记录平均被引用 "
        f"{len(rows) / max(len(uniq), 1):.1f} 次)"
    )

    print("\n=== [1] 传感器自身系约定(视线轴因 modality 而异,见模块 docstring 坑 1)===")
    for mod, desc in NUS_SENSOR_CONVENTIONS.items():
        chans = sorted({r.channel for r in uniq if r.modality == mod})
        if chans:
            print(f"  {mod:<7} {desc:<28} 通道: {', '.join(chans)}")

    print("\n=== [2] 逐传感器位姿(translation / rotation 均为「自身系 → ego 系」)===")
    print(
        f"  {'channel':<18}{'modality':<9}{'车@地点':<30}{'translation 米':<26}"
        f"{'rotation (w,x,y,z)':<36}{'az°':>9}{'el°':>9}"
    )
    for r in uniq:
        az, el = r.az_el
        print(
            f"  {r.channel:<18}{r.modality:<9}{_fmt_veh(r.vehicles):<30}{_fmt_vec(r.translation):<26}"
            f"{_fmt_vec(r.rotation):<36}{az:>9.3f}{el:>9.3f}"
        )

    print("\n=== [3] 每通道标定套数(>1 = 多台车/多次标定,勿只取第一条)===")
    for ch in sorted({r.channel for r in uniq}):
        grp = [r for r in uniq if r.channel == ch]
        n_rec = grp[0].n_calib_for_channel
        if len(grp) == 1:
            print(f"  {ch:<18} 1 套(表内 {n_rec} 条)  {', '.join(grp[0].vehicles) or '(无引用)'}")
        else:
            print(f"  {ch:<18} {len(grp)} 套(表内 {n_rec} 条)")
            for r in grp:
                az, _ = r.az_el
                print(
                    f"      t={_fmt_vec(r.translation):<26} az={az:>9.3f}°   "
                    f"{', '.join(r.vehicles) or '(无引用)'}"
                )

    cams = [r for r in uniq if r.modality == "camera" and r.camera_intrinsic]
    if cams:
        print("\n=== [4] 相机内参(calibrated_sensor.camera_intrinsic,标称分辨率下的像素口径)===")
        for r in cams:
            k = np.array(r.camera_intrinsic)
            print(
                f"  {r.channel:<18} fx={k[0, 0]:9.3f} fy={k[1, 1]:9.3f} "
                f"cx={k[0, 2]:9.3f} cy={k[1, 2]:9.3f}  skew={k[0, 1]:.3f}   "
                f"{', '.join(r.vehicles) or '(无引用)'}"
            )

    print("\n注:az/el 由视线轴 R @ 自身系轴 求得,**未**用 Quaternion.yaw_pitch_roll")
    print("    (对相机那三个角无几何意义 —— 相机视线轴是自身系 +z,与 yaw 转轴重合)")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="打印 nuScenes 每传感器相对 ego 的位姿/方位")
    ap.add_argument("--dataroot", default=None, help="nuScenes dataroot(缺省自动挑)")
    ap.add_argument("--version", default=VERSION, help=f"表版本目录(默认 {VERSION})")
    args = ap.parse_args(argv)

    root = pick_dataroot(args.dataroot)
    if root is None:
        print(
            "[skip] 未找到可用 dataroot(候选:--dataroot / $NUSCENES_DATAROOT / 官方 mini / 本仓 outputs/nus_mini)"
        )
        return 1
    try:
        nusc = load_nusc(root, args.version)
    except ImportError as e:
        print(f"[skip] 缺 nuscenes-devkit: {e}")
        return 1
    print_report(nusc, root, args.version)
    return 0


# ---------------------------------------------------------------- 单测


class TestPureAnchors:
    """纯 numpy 手算锚点(不需要 devkit)。"""

    def test_quat_identity(self):
        np.testing.assert_allclose(quat_to_matrix((1.0, 0.0, 0.0, 0.0)), np.eye(3), atol=1e-15)

    def test_quat_yaw90_hand_computed(self):
        # yaw +90°:q = (cos45°, 0, 0, sin45°) → 把 ego +x 转到 +y
        s = float(np.sqrt(0.5))
        np.testing.assert_allclose(
            quat_to_matrix((s, 0.0, 0.0, s)), [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], atol=1e-12
        )

    def test_quat_non_unit_normalised(self):
        np.testing.assert_allclose(quat_to_matrix((2.0, 0.0, 0.0, 0.0)), np.eye(3), atol=1e-15)

    def test_quat_zero_raises(self):
        with pytest.raises(ValueError):
            quat_to_matrix((0.0, 0.0, 0.0, 0.0))

    def test_basis_identity_camera(self):
        # 相机自身系 x右/y下/z前 → 单位旋转下:forward=+z, left=−x, up=−y
        f, left, up = sensor_basis(np.eye(3), "camera")
        np.testing.assert_allclose(f, [0.0, 0.0, 1.0], atol=1e-15)
        np.testing.assert_allclose(left, [-1.0, 0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(up, [0.0, -1.0, 0.0], atol=1e-15)

    def test_basis_identity_lidar(self):
        f, left, up = sensor_basis(np.eye(3), "lidar")
        np.testing.assert_allclose(f, [1.0, 0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(left, [0.0, 1.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(up, [0.0, 0.0, 1.0], atol=1e-15)

    def test_camera_boresight_invariant_under_yaw(self):
        """坑 1 的可判据形式:相机视线轴 = 自身系 +z ⇒ 绕 z 的纯 yaw 四元数**不改变**方位角。

        所以 `Quaternion.yaw_pitch_roll` 读相机得到的是"自身系 z 轴被转到了哪",与朝向无关。
        """
        s = float(np.sqrt(0.5))
        f, _, _ = sensor_basis(quat_to_matrix((s, 0.0, 0.0, s)), "camera")
        np.testing.assert_allclose(f, [0.0, 0.0, 1.0], atol=1e-12)
        assert az_el_deg(f) == pytest.approx((0.0, 90.0))

    def test_lidar_boresight_rotates_with_yaw(self):
        s = float(np.sqrt(0.5))
        f, _, _ = sensor_basis(quat_to_matrix((s, 0.0, 0.0, s)), "lidar")
        assert az_el_deg(f) == pytest.approx((90.0, 0.0))

    def test_unknown_modality_raises(self):
        with pytest.raises(ValueError):
            sensor_basis(np.eye(3), "imu")

    @pytest.mark.parametrize(
        ("axis", "expected"),
        [
            ((1.0, 0.0, 0.0), (0.0, 0.0)),
            ((0.0, 1.0, 0.0), (90.0, 0.0)),
            ((-1.0, 0.0, 0.0), (180.0, 0.0)),
            ((0.0, -1.0, 0.0), (-90.0, 0.0)),
            ((0.0, 0.0, 1.0), (0.0, 90.0)),
            ((0.0, 0.0, -1.0), (0.0, -90.0)),
        ],
    )
    def test_az_el_anchors(self, axis, expected):
        assert az_el_deg(axis) == pytest.approx(expected, abs=1e-12)

    def test_az_el_normalises(self):
        assert az_el_deg((3.0, 4.0, 0.0))[0] == pytest.approx(np.degrees(np.arctan2(4.0, 3.0)))

    def test_az_el_zero_raises(self):
        with pytest.raises(ValueError):
            az_el_deg((0.0, 0.0, 0.0))


class TestQuatOracle:
    """`quat_to_matrix` vs devkit `Quaternion.rotation_matrix`(缺 devkit 自动跳过)。"""

    def test_matches_devkit(self):
        Quaternion = pytest.importorskip("nuscenes.utils.data_classes").Quaternion
        rng = np.random.default_rng(20260922)
        for q in rng.normal(size=(20, 4)):
            q = q / np.linalg.norm(q)
            np.testing.assert_allclose(quat_to_matrix(q), Quaternion(q.tolist()).rotation_matrix, atol=1e-12)


# ---------------------------------------------------------------- devkit 集成


HAS_MINI = (OFFICIAL_MINI / VERSION / "sensor.json").is_file()
needs_mini = pytest.mark.skipif(not HAS_MINI, reason=f"本机无官方 nuScenes mini: {OFFICIAL_MINI}")


@pytest.fixture(scope="module")
def nusc():
    """官方 mini 的 devkit 句柄(缺包/缺数据则整模块跳过)。"""
    if not HAS_MINI:
        pytest.skip(f"本机无官方 nuScenes mini: {OFFICIAL_MINI}")
    pytest.importorskip("nuscenes")
    return load_nusc(OFFICIAL_MINI)


@pytest.fixture(scope="module")
def poses(nusc) -> list[SensorPose]:
    """官方 mini 全部标定记录(未去重,120 条)。"""
    return sensor_poses(nusc)


@pytest.fixture(scope="module")
def uniq(poses) -> list[SensorPose]:
    """去重后的标定套数(24 套 = 12 通道 × 2 台车)。"""
    return dedup_poses(poses)


@needs_mini
class TestOfficialMini:
    """官方 mini 实测锚点(数字取自 `calibrated_sensor.json`,2026-09-22 核)。"""

    def test_table_sizes(self, nusc):
        assert len(nusc.sensor) == 12  # 6 相机 + 1 激光 + 5 雷达
        assert len(nusc.calibrated_sensor) == 120  # 12 通道 × 10 log
        assert len(nusc.log) == 8

    def test_channels_and_modalities(self, uniq):
        assert {r.channel for r in uniq} == {
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
            "LIDAR_TOP",
            "RADAR_FRONT",
            "RADAR_FRONT_LEFT",
            "RADAR_FRONT_RIGHT",
            "RADAR_BACK_LEFT",
            "RADAR_BACK_RIGHT",
        }
        assert sum(r.modality == "camera" for r in uniq) == 12  # 每通道 2 套标定
        assert sum(r.modality == "lidar" for r in uniq) == 2
        assert sum(r.modality == "radar" for r in uniq) == 10

    def test_every_channel_has_two_calibs(self, uniq):
        """坑 2:同 channel 多套标定 —— 每条记录只被部分 log 引用,故都要保留。

        每通道 10 条记录(n015 6 条 + n008 4 条),但只有 2 套**不同**的位姿。
        """
        for ch in {r.channel for r in uniq}:
            grp = [r for r in uniq if r.channel == ch]
            assert len(grp) == 2, ch
            assert all(r.n_calib_for_channel == 10 for r in grp)

    def test_vehicles_recovered_from_log(self, uniq):
        assert all(r.vehicles for r in uniq), "calibrated_sensor 表无车辆字段,必须经 log 反查"
        assert {v for r in uniq for v in r.vehicles} == {
            "n015@singapore-hollandvillage",
            "n015@singapore-onenorth",
            "n015@singapore-queenstown",
            "n008@boston-seaport",
        }

    def test_n015_calib_serves_three_logs(self, uniq):
        """一套标定被多个 log 引用(新加坡 3 个 log 共用 n015 的同一套标定)。"""
        assert len(find_pose(uniq, "CAM_FRONT", "n015").vehicles) == 3
        assert len(find_pose(uniq, "CAM_FRONT", "n008").vehicles) == 1

    def test_cam_front_pose_anchor(self, uniq):
        r = find_pose(uniq, "CAM_FRONT", "n015")
        assert r.translation == pytest.approx((1.70079, 0.01595, 1.51096), abs=1e-4)
        assert r.rotation == pytest.approx((0.4998, -0.50303, 0.49978, -0.49737), abs=1e-4)
        assert r.az_el == pytest.approx((0.325, -0.323), abs=0.01)

    def test_cam_back_boresight_points_backwards(self, uniq):
        az, el = find_pose(uniq, "CAM_BACK", "n015").az_el
        assert az == pytest.approx(179.857, abs=0.01)
        assert abs(el) < 1.0

    def test_lidar_boresight_is_plus_x_of_its_own_frame(self, uniq):
        """LIDAR_TOP 的四元数近似绕 z −90°,故自身系 +x 指向 ego −y(az≈−90°)。

        若误用「视线轴 = +z」会读到 el=90°(正上方),那是**上方向**不是朝向。
        """
        az, el = find_pose(uniq, "LIDAR_TOP", "n015").az_el
        assert az == pytest.approx(-89.883, abs=0.01)
        assert abs(el) < 2.0

    def test_radar_front_boresight_is_plus_x(self, uniq):
        r = find_pose(uniq, "RADAR_FRONT", "n015")
        assert r.az_el == pytest.approx((0.2, 0.0), abs=0.05)
        assert r.translation == pytest.approx((3.412, 0.0, 0.5), abs=1e-6)

    def test_camera_intrinsics_present_and_off_axis(self, uniq):
        """相机 K 的 cx/cy 不在 w/2 上(1600×900 ⇒ cx=816.27)⇒ 与 1242×375 一样是偶数宽,
        不存在"像素中心落在光轴上"的约定,主点只能实测(与本仓标定自证任务同源)。"""
        r = find_pose(uniq, "CAM_FRONT", "n015")
        k = np.array(r.camera_intrinsic)
        assert k.shape == (3, 3)
        assert k[0, 0] == pytest.approx(1266.417, abs=1e-3)
        assert k[0, 2] == pytest.approx(816.267, abs=1e-3)
        assert k[0, 2] != pytest.approx(800.0)  # ≠ w/2

    def test_non_camera_intrinsic_is_empty(self, uniq):
        assert all(r.camera_intrinsic == () for r in uniq if r.modality != "camera")


@needs_mini
class TestOfficialVsRepo:
    """官方 mini 对照本仓自产 `outputs/nus_mini`(缺则跳过)。"""

    @staticmethod
    def _repo_rows() -> list[SensorPose]:
        if not (REPO_MINI / VERSION / "sensor.json").is_file():
            pytest.skip(f"本仓未生成 {REPO_MINI}")
        pytest.importorskip("nuscenes")
        return sensor_poses(load_nusc(REPO_MINI))

    def test_repo_mini_single_calib_set(self):
        rows = self._repo_rows()
        assert all(r.n_calib_for_channel == 1 for r in rows), "本仓只有一套标定"
        assert {v for r in rows for v in r.vehicles} == {"ad_ego@carla_town10"}

    def test_repo_lidar_quat_matches_official(self):
        """本仓写 LIDAR_TOP 四元数 == 官方 n015 原值(含 1.4289° up 轴倾角)。

        **本测试 2026-09-23 前测的是缺陷**(旧断言 `rotation == (1,0,0,0)`、`az_el == (0,0)`,
        即"单位四元数 = 视线轴在 ego +x")。那不是本仓的选择而是 bug:`collect_nus.py` spawn
        LiDAR 时**完全没设 rotation**,导出侧又写 `_quat(0.0)`。后果是 devkit 按"传感器系 =
        ego 系"解释点云 ⇒ 整片点云绕 z 转 90°(`num_lidar_pts` 复现比值 1.0000 → 0.0854)。
        现在两侧同源 `export.nuscenes.NUS_LIDAR_CALIB`,官方值 = 绕 z −89.879° + 1.4289° 倾角。
        """
        from autodrivedata.gt.export.nuscenes import NUS_LIDAR_CALIB

        r = find_pose(self._repo_rows(), "LIDAR_TOP")
        assert r.translation == pytest.approx(NUS_LIDAR_CALIB[0], abs=1e-9)
        assert r.rotation == pytest.approx(NUS_LIDAR_CALIB[1], abs=1e-9)
        assert r.rotation != pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-3)  # 不是单位四元数
        az, _ = r.az_el
        assert az == pytest.approx(-89.879, abs=0.01)  # 与官方 n015 同(见 TestOfficialMini)

    def test_repo_camera_translations_match_official(self):
        """本仓相机平移抄自官方(逐字段一致);旋转亦然(见 export/nuscenes.NUS_CAMERA_CALIBS)。"""
        repo = {r.channel: r for r in self._repo_rows()}
        official = {r.channel: r for r in sensor_poses(load_nusc(OFFICIAL_MINI)) if "n015" in r.vehicles[0]}
        for ch in ("CAM_FRONT", "CAM_BACK_LEFT", "CAM_FRONT_RIGHT"):
            assert repo[ch].translation == pytest.approx(official[ch].translation, abs=1e-4), ch

    def test_repo_camera_boresights_match_spec(self):
        """本仓 6 相机方位角 = `camera_rig.NUS_CAMERA_CALIBS` 的规格(±0.01° 内)。

        这条是"规格 → 落盘"的对账:规格是 nuScenes 全局系(y 左)的官方标定,
        落盘(export)也走同一系 ⇒ az_nus 必须逐字段一致。**注意 CARLA 侧 spawn 用的是
        `NUS_CAMERA_RIG`(y 右,`yaw_carla = −az_nus`)**——2026-09-22 前采集器把 az 原样
        抄成正数,四个侧/后相机镜像;本测试当时测的是 export 侧(本来就对),故没拦住。
        """
        for r in self._repo_rows():
            if r.channel not in NUS_CAMERA_CALIBS:
                continue
            t_nus, q_nus = NUS_CAMERA_CALIBS[r.channel]
            az_spec, _ = az_el_deg(sensor_basis(quat_to_matrix(q_nus), "camera")[0])
            az, _ = r.az_el
            # 180° 附近的差要绕回(−179.9 vs 179.9 不是 360° 的差)
            d = (az - az_spec + 180.0) % 360.0 - 180.0
            assert abs(d) < 0.01, f"{r.channel}: 落盘 az={az:.3f}° vs 规格 {az_spec:.3f}°"
            assert r.translation == pytest.approx(t_nus, abs=1e-4), r.channel


if __name__ == "__main__":
    raise SystemExit(main())

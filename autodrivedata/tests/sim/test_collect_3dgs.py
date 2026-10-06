"""`collect_3dgs` 的**结构钉**(纯值,不连 CARLA)。

采集器本身进不了单测(要 CARLA 跑很久),但**两条纪律**必须能被机械钉住 ——
它们坏掉时的症状都不是"报错",而是"数据看着正常、其实是错的":

1. **帧同步**:主循环必须走 `shoot_synced`(等**当次位姿**那一帧落地),不能退回
   "每 tick `q.get()` 一次"。CARLA 的传感器投递比 `world.tick()` 晚定额 2 帧 ⇒
   FIFO 取最旧 = **整段序列**拍的都不是它自己那个位姿(帧号却一张张对得上)。
2. **同挂点同 fov**:`--sem` / `--inst` 那两路必须与 RGB 走**同一份** `CAM_ATTRS` 与
   **同一个** transform。不齐时 tag 图与 RGB 图不再逐像素对应,而"不齐"在下游
   只表现为「归属错」。
"""

from __future__ import annotations

import inspect

from autodrivedata.sim import collect_3dgs
from autodrivedata.sim.carla_common import CAM_ATTRS


class _BP:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attrs: dict = {}

    def set_attribute(self, k: str, v: object) -> None:
        self.attrs[k] = v


class _Lib:
    """假的 `carla.BlueprintLibrary`:只记"谁被建了、设了哪些属性"。"""

    def __init__(self) -> None:
        self.made: list[_BP] = []

    def find(self, name: str) -> _BP:
        b = _BP(name)
        self.made.append(b)
        return b


class TestKindNames:
    def test_default_is_no_extra_camera(self):
        assert collect_3dgs.kind_names(False, False) == []

    def test_flags_select_blueprints(self):
        assert collect_3dgs.kind_names(True, False) == ["semantic_segmentation"]
        assert collect_3dgs.kind_names(False, True) == ["instance_segmentation"]
        assert collect_3dgs.kind_names(True, True) == ["semantic_segmentation", "instance_segmentation"]

    def test_every_kind_has_a_report_label(self):
        """`assert_synced` 的报错靠它指出是哪一路掉队 —— 漏一个就少一条可定位的信息。"""
        for k in collect_3dgs.kind_names(True, True):
            assert k in collect_3dgs.KIND_LABEL


class TestMountDiscipline:
    def test_blueprints_get_exactly_cam_attrs(self):
        """★ 三项(`image_size_x/y` / `fov`)**逐字**取 `CAM_ATTRS` —— RGB 那路是这么设的。"""
        lib = _Lib()
        got = collect_3dgs.make_kind_blueprints(lib, collect_3dgs.kind_names(True, True))
        assert [b.name for b in lib.made] == [
            "sensor.camera.semantic_segmentation",
            "sensor.camera.instance_segmentation",
        ]
        for b in lib.made:
            assert b.attrs == CAM_ATTRS
        assert set(got) == {"semantic_segmentation", "instance_segmentation"}

    def test_no_extra_camera_when_flags_off(self):
        lib = _Lib()
        assert collect_3dgs.make_kind_blueprints(lib, []) == {}
        assert lib.made == []

    def test_apply_pose_gives_every_sensor_the_same_transform(self):
        """同一个 tf 逐字施加到四路 —— 差一点就是两幅画不重合。"""
        seen = []

        class _S:
            def __init__(self, n):
                self.n = n

            def set_transform(self, tf):
                seen.append((self.n, tf))

        tf = object()
        collect_3dgs.apply_pose([_S("rgb"), _S("depth"), _S("sem"), _S("inst")], tf)
        assert [n for n, _ in seen] == ["rgb", "depth", "sem", "inst"]
        assert all(t is tf for _, t in seen)


class TestFrameSyncPin:
    """★ 主循环的**帧同步**走哪条路 —— 这条退回旧写法时会静默坏掉整段序列。"""

    def setup_method(self):
        self.src = inspect.getsource(collect_3dgs.main)

    def test_main_loop_uses_shoot_synced(self):
        assert "shoot_synced(" in self.src

    def test_main_loop_does_not_take_raw_queue(self):
        """`q.get()` / `dq.get()` 是**旧写法**:FIFO 取最旧 ⇒ 恒定滞后 2 帧。"""
        assert "q.get(" not in self.src
        assert "dq.get(" not in self.src

    def test_assert_synced_is_still_run_each_frame(self):
        """`shoot_synced` 管"是不是当次位姿",`assert_synced` 管"几路彼此同不同步" —— 两件事。"""
        assert "assert_synced(" in self.src

    def test_output_layout_matches_the_readers(self):
        """落盘路径是**跨模块契约**:`gs/attribute_instances` 与 `gs/frame_sync` 按这个读。"""
        assert 'out / "inst" / f"p{int(p)}"' in self.src
        assert 'out / "sem" / f"p{int(p)}"' in self.src
        assert "{i:05d}.png" in self.src


class TestFlagsDefaultOff:
    def test_sem_and_inst_default_off(self):
        """默认关 ⇒ 关着时产物与 2026-09-18 那版逐字节一致(新增通道不改既有口径)。"""
        src = inspect.getsource(collect_3dgs.main)
        for flag in ('"--sem"', '"--inst"'):
            assert flag in src
        assert "default=True" not in src
        assert 'action="store_true"' in src


class TestPropsAndCleanup:
    """★ 2026-10-04 补的两件事:**清场**(原先完全没有)与 **`--props`**(编辑实验的 A 侧)。

    两条坏掉时的症状都不是报错:

    - 不清场 ⇒ A 摆的道具**跟着 B 侧采完** —— 数据照出,只是 A/B 的差凭空小一截;
    - `--props` 关着时多写了 `prop.json` ⇒ 下游把"没有道具"的那一轮也当成有道具(或反之)。
    """

    def setup_method(self):
        self.src = inspect.getsource(collect_3dgs.main)

    def test_cleanup_runs_before_anything_is_spawned(self):
        """清场必须在摆道具/挂相机**之前** —— 否则清掉的就是刚摆的那个。"""
        assert "clear_generated_actors(world)" in self.src
        i_clear = self.src.index("clear_generated_actors(world)")
        i_prop = self.src.index("spawn_prop_at(")
        assert i_clear < i_prop, "清场排在摆道具之后 —— 会把刚摆的道具清掉"

    def test_cleanup_predicate_is_the_shared_one(self):
        """**不许在这里另写一份** —— 清场与收尾不同源就是"清一半"这类不可见失败。"""
        assert "def clear_generated_actors" not in inspect.getsource(collect_3dgs)
        from autodrivedata.sim.carla_common import clear_generated_actors

        assert callable(clear_generated_actors)

    def test_prop_json_only_written_when_props_on(self):
        """`prop.json` 的写入必须**被 `prop_box is not None` 守住**。

        ⚠️ 认**写入点**不是认词 —— `prop.json` 最早出现在 `--props` 的 `--help` 文案里,
        拿第一次出现当判据会去检查一段注释。
        """
        i = self.src.index('(out / "prop.json").write_text(')
        # 往上找最近的 if 守卫
        head = self.src[:i]
        assert head.rindex("if prop_box is not None:") > head.rindex("for p in pitches:"), (
            "prop.json 的写入没有开关守卫(或守卫的位置不对)"
        )

    def test_spawn_prop_uses_yaw0_and_records_instance_id(self):
        """尺寸走 **yaw=0 蓝图探针**(转过之后 `bounding_box` 给的是被剪切的值),"""
        src = inspect.getsource(collect_3dgs.spawn_prop_at)
        assert "measure_actor_size_yaw0(" in src
        assert "carla.Rotation(yaw=0.0)" in src, "摆位给了 yaw ⇒ 记录尺寸与实摆口径不再一致"
        assert "instance_id=a.id" in src, "没记 actor id ⇒ 归属那一步只能靠猜哪个 id 是道具"

    def test_prop_position_is_ring_relative_not_ego_relative(self):
        """本采集器**没有 ego**(绕静止中心转) —— 照抄 `spawn_props` 会摆错地方。"""
        src = inspect.getsource(collect_3dgs.spawn_prop_at)
        assert "center.x + offset[0]" in src
        # 不给 ego 相对的那套件任何立足点(照抄 `spawn_props` 就会带进来)。
        # ⚠️ 只认**调用**:名字出现在本函数的 docstring 里(那是"别照抄"的解释),
        #    按词判会把解释当成犯错。
        assert "spawn_props(" not in src

    def test_placement_is_read_back_not_assumed(self):
        """摆位自证:摆偏了**不抛异常**,只会让后面每条判据都偏低。"""
        src = inspect.getsource(collect_3dgs.spawn_prop_at)
        assert "PLACE_TOL_M" in src
        assert "get_transform()" in src

"""CARLA live 手动驾驶:在服务器终端用 WASD 遥控已运行场景里的 ego 车。

**首选路径是 `bin/live_studio.py --keyboard`(键盘折进 tick 循环,单终端即可)**;
本脚本保留"独立进程遥控"用法:当另一个进程(如 `bin/view_stream.py`)持有 tick 时,
本脚本只持续 `apply_control`(控制命令在下次 tick 生效),两者无冲突。
ego 识别 = 挂着相机的车(cams 全部 attach_to=ego)。

键盘映射与控制状态机在 `bin/live_common.KeyboardState`(两处共用一个实现,勿另立):
  w/s 油门 0.8 / 刹车 0.8 · a/d 转向 ∓0.45 · x 滑行 · 空格 手刹急停 · q 退出

用法(与 view_stream 并行,另开一个服务器终端):
  python bin/drive_ego.py
"""

from __future__ import annotations

import time

import carla
from live_common import KeyboardState


def _find_ego(world: carla.World) -> carla.Vehicle | None:
    """挂相机的车 = view_stream 的 ego(spawn_ego 不设 role_name,相机挂点唯一)。

    重试等快照:同步模式新客户端首个 get_actors() 可能为空(快照 tick 后刷新)。
    """
    for _ in range(10):
        for a in world.get_actors():
            if a.type_id.startswith("sensor.camera."):
                parent = a.parent
                if parent is not None and parent.type_id.startswith("vehicle."):
                    return parent  # type: ignore[return-value]
        time.sleep(0.3)
    return None


def main() -> None:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    ego = _find_ego(world)
    if ego is None:
        print("[drive] 未找到 ego 车(请先启动 view_stream / live_studio)")
        return
    print(f"[drive] ego = {ego.type_id} @ {world.get_map().name},{KeyboardState.KEYS}")

    kb = KeyboardState()
    try:
        while not kb.quit:
            kb.poll()
            if kb.quit:
                break
            kb.apply(ego)  # 每轮重发:sync 模式命令在下次 tick 生效,必须持续喂
            print(f"\r{kb.hud(ego)}  ", end="")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        kb.close()  # 还原终端属性(否则退出后终端不回显)
        ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
    print()


if __name__ == "__main__":
    main()

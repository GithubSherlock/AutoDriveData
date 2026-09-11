"""CARLA live 手动驾驶:在服务器终端用 WASD 遥控 view_stream 场景里的 ego 车。

view_stream 运行在同步模式并负责 tick,本脚本只持续 apply_control(控制命令
在下次 tick 生效,view_stream ≈ 4.8Hz tick 驱动物理,车即响应)。view_stream
默认 --speed 0 不碰控制,两者无冲突;ego 识别 = 挂着相机的车(cams 全部
attach_to=ego)。

键盘(tty 原始模式按下即响应;stdin 为管道/重定向时按行读,便于测试):
  w/s   油门 0.8 / 刹车 0.8
  a/d   转向 -0.45 / +0.45
  x     滑行(油门刹车归零)
  空格  手刹急停
  q     退出

用法(与 view_stream 并行,另开一个服务器终端):
  python bin/drive_ego.py
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty

import carla


def _read_key(is_tty: bool) -> str | None:
    """tty 下原始模式读单键(按下即响应);管道下按行读首字符(测试路径)。"""
    if is_tty:
        if select.select([sys.stdin], [], [], 0.05)[0]:
            ch = sys.stdin.read(1)
            return ch.lower() if ch else None
        return None
    line = sys.stdin.readline()
    return line.strip().lower()[:1] if line else None


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
        print("[drive] 未找到 ego 车(请先启动 view_stream)")
        return
    print(f"[drive] ego = {ego.type_id} @ {world.get_map().name},WASD 驾驶,q 退出")

    throttle = brake = steer = 0.0
    handbrake = False
    is_tty = sys.stdin.isatty()
    old = termios.tcgetattr(sys.stdin.fileno()) if is_tty else None
    if is_tty:
        tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            ch = _read_key(is_tty)
            if ch is None:  # 管道 EOF:退出
                if not is_tty:
                    break
            elif ch == "w":
                throttle, brake = 0.8, 0.0
            elif ch == "s":
                throttle, brake = 0.0, 0.8
            elif ch == "a":
                steer = -0.45
            elif ch == "d":
                steer = 0.45
            elif ch == "x":
                throttle, brake = 0.0, 0.0
            elif ch == " ":
                brake, handbrake = 1.0, True
            elif ch == "q":
                ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                break
            # 每轮都重发控制:sync 模式命令在下次 tick 生效,必须持续喂
            ego.apply_control(
                carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, hand_brake=handbrake)
            )
            v = ego.get_velocity()
            kmh = (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5 * 3.6
            print(
                f"\r速度 {kmh:5.1f} km/h | 油门 {throttle} 刹车 {brake} 转向 {steer:+.2f} 手刹 {int(handbrake)}  ",
                end="",
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if is_tty and old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
    print()


if __name__ == "__main__":
    main()

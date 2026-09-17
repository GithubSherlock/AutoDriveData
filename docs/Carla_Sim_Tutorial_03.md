# Carla仿真系列：4_在 Carla 的 Tesla 上装 4 路摄像头，并创建网格为环视拼接做准备

前几期把环境搭好、加了 NPC、跑通了 YOLO 检测。这期开始为环视拼接做准备——在 Tesla 车上安装前后左右 4 路摄像头，并创建标定用的棋盘格。

## 1 为什么是 4 路摄像头

环视拼接需要覆盖车辆四周 360 度，用 4 路 120 度 FOV 的相机，每路间隔 90 度，留有 30 度重叠区域用于拼接。

       前 (yaw=0°)
       
        ↑
        
左 ←── 车 ──→ 右(yaw=90°)

(yaw=-90°)        ↓

       后 (yaw=180°)

## 2 摄像头安装配置

在 `carla_spawn_objects/config/objects.json` 中给 Tesla 配置 4 个传感器。

Tesla 尺寸：4.6 × 1.84 × 1.44 m

```json
{
  "type": "sensor.camera.rgb",
  "id": "rgb_front",
  "spawn_point": {"x": 2.3, "y": 0.0, "z": 0.8, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
  "image_size_x": 640,
  "image_size_y": 480,
  "fov": 120.0
},
{
  "type": "sensor.camera.rgb",
  "id": "rgb_back",
  "spawn_point": {"x": -2.3, "y": 0.0, "z": 0.8, "roll": 0.0, "pitch": 0.0, "yaw": 180.0},
  "image_size_x": 640,
  "image_size_y": 480,
  "fov": 120.0
},
{
  "type": "sensor.camera.rgb",
  "id": "rgb_left",
  "spawn_point": {"x": 0.0, "y": 0.92, "z": 0.8, "roll": 0.0, "pitch": 0.0, "yaw": 90.0},
  "image_size_x": 640,
  "image_size_y": 480,
  "fov": 120.0
},
{
  "type": "sensor.camera.rgb",
  "id": "rgb_right",
  "spawn_point": {"x": 0.0, "y": -0.92, "z": 0.8, "roll": 0.0, "pitch": 0.0, "yaw": -90.0},
  "image_size_x": 640,
  "image_size_y": 480,
  "fov": 120.0
}
```

下图是通过 rviz 查看 4 路摄像头的数据。

```bash
rosrun rviz rviz -d carla_sim_config.rviz
```

关键点解释

| 配置 | 说明 |
| --- | --- |
| x=2.3 / -2.3 | 前后相机装在车头 / 车尾前方 |
| y=0.92 / -0.92 | 左右相机装在车两侧（车宽 1.84m 的一半） |
| z=0.8 | 相机离地高度 0.8 米 |
| yaw 旋转 | 后 180°、左 90°、右 - 90°，让相机朝向车外 |
| fov=120 | 广角，保证相邻相机有重叠区域 |

坐标系用的是右手坐标系，跟 ROS 一致。

## 3 Python3 获取 4 路图像

跟之前一样的套路，终端启动基础环境，然后：

终端 3: TCP Bridge Server（多相机版）

终端 4: Python 3.10 接收图像

```bash
# 终端3：多相机 TCP 服务端（Python 2.7）
cd /opt/carla_ws
python tcp_bridge/tcp_bridge_server_multi_camera_py2.py

# 终端4：接收图像（Python 3.10）
source activate py310
cd /opt/carla_ws/sim_workspace/image_detect
python tcp_bridge_client_py3.py
```

## 4 创建标定网格

Carla 里面是没有标准的黑白棋盘格，所以我这边是创建网格来代替。

```python
# 终端5：创建标定棋盘格
python scripts/create_board_py2.py --vehicle-id 513
# --vehicle-id 是车辆的 ID，在创建 ego_vehicle 时会有打印输出，换成你自己的车辆 ID 即可。
```

## 5 下一步

4 路相机 + 棋盘格都准备好了，下一期就是真正的环视拼接 —— 把 4 路图像投影到统一的 BEV 鸟瞰视角，拼接成完整的 360 度环视图。

# Carla仿真系列：3_Carla 仿真中跑通 YOLOv8 实时目标检测

前两期搭建了 Carla + ROS 环境，添加了 NPC 交通流。这期是最关键的一步——用 YOLOv8 做实时目标检测。
但遇到一个麻烦：Carla ROS Bridge 跑在 Python 2.7 上，而 YOLOv8 需要 Python 3.10。

## 1 两种方案对比

为了解决"数据在 Python 2.7，模型在 Python 3.10"这个问题，我试了两种方案。

### 方案一：rosbridge + roslibpy

思路是通过 rosbridge 把 ROS 话题通过 WebSocket 转发出去，Python 3.10 这边用 roslibpy 接收。

```bash
# 启动 rosbridge
roslaunch rosbridge_server rosbridge_websocket.launch queue_size:=1 port:=9099
# Python 3.10 环境接收
python rosbridge_roslib.py
```

启动简单，但跑起来发现两个问题：
延迟高，画面卡顿
图像丢帧严重，实时性不行

### 方案二：TCP 直接传输图像数据

放弃 rosbridge，直接用 TCP 把图像数组从 Python 2.7 发给 Python 3.10。

服务端（Python 2.7）：

```python
# 订阅 ROS 图像话题 → 转为 numpy → TCP 发送
def image_callback(msg):
    img = np.frombuffer(msg.data, dtype=np.uint8)
    img = img.reshape(msg.height, msg.width, 4)[:, :, :3]
    header = str(len(img.tobytes())).encode().ljust(16)
    conn.sendall(header + img.tobytes())
```

客户端（Python 3.10）：

```python
# TCP 接收 → numpy → YOLOv8 检测 → 可视化
def recv_all(sock, length):
    buf = b""
    while len(buf) < length:
        chunk = sock.recv(length - len(buf))
        if not chunk: raise ConnectionError()
        buf += chunk
    return buf

# 主循环
while True:
    header = recv_all(sock, 16)
    img_bin = recv_all(sock, int(header.strip()))
    frame = pickle.loads(img_bin)
    results = model(frame)
    # 绘制检测框并显示
```

整个跑起来就很流畅了，基本不影响使用。

## 2 完整启动流程

终端 1: Carla UE4
终端 2: ROS Bridge
终端 3: NPC 生成（可选）
终端 4: TCP Bridge Server（Python 2.7 环境）
终端 5: YOLOv8 检测（Python 3.10 环境）

按顺序启动：

```python
# 终端1-3：跟第二期一样，先启动Carla、ROS Bridge、NPC
# 终端4：启动 TCP 服务端（Python 2.7）
cd /opt/carla_ws
python tcp_bridge/tcp_bridge_server.py

# 终端5：启动 YOLOv8 检测（Python 3.10）
cd /opt/carla_ws/sim_workspace/image_detect
python image_detect_nms.py
```

启动后，终端 5 会弹出检测窗口，实时显示 YOLOv8 的检测结果 —— 车辆、行人、骑行者都被框出来了。

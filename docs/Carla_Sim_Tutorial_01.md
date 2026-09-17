# Carla仿真：Python2.7 ROS与Python3.10图像跨环境通信方案

最经在做Carla仿真时遇到一个很尴尬的问题：Carla ROS Bridge 跑在 Python 2.7 上，但我想用 Python3.10 先获取到图像后面再做目标检测，两个环境不能直接通信，图像数据传不过去。目前我尝试了两种方案：

- 方案一：rosbridge + roslibpy，延迟比较高。
- 方案二：采用TCP进行数据传输，延迟低。

## 方案一：rosbridge + roslibpy
rosbridge 的原理是 WebSocket + Base64 编码，图像数据经过 Base64 编码后体积翻倍，网络开销大，延迟高。多摄像头场景更严重。而且 rosbridge 依赖 apt 源安装，在国内网络环境下，404 和密钥过期的问题很常见。

### 1 ROS Python2.7端

#### 第1步：环境安装

```bash
sudo apt install ros-melodic-rosbridge-server
```

#### 第 2 步：启动 rosbridge

### 2 Python3.10端

```bash
source /opt/ros/melodic/setup.bash # queue_size=1 丢掉旧帧缓解延迟堆积
roslaunch rosbridge_server rosbridge_websocket.launch queue_size:=1 port:=9099
```

默认地址 ws://127.0.0.1:9099

#### 第 1 步：conda py310 安装依赖

```
conda activate py310
pip install roslibpy opencv-python numpy
```

#### 第 2 步：代码 rosbridge_roslib.py

```python
# -*- coding: utf-8 -*-
import base64
import numpy as np
import cv2
import roslibpy

# ==================== 配置区 ====================
ROS_BRIDGE_HOST = "127.0.0.1"
ROS_BRIDGE_PORT = 9099
IMAGE_TOPIC = "/carla/ego_vehicle/rgb_front/image"
MSG_TYPE = "sensor_msgs/Image"
# ================================================

client = roslibpy.Ros(host=ROS_BRIDGE_HOST, port=ROS_BRIDGE_PORT)

def image_callback(msg):
    h = msg["height"]
    w = msg["width"]
    encoding = msg["encoding"]

    # rosbridge二进制数据经过base64编码，必须解码
    raw_bytes = base64.b64decode(msg["data"])
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = None

    if encoding == "bgra8":
        img = arr.reshape(h, w, 4)[:, :, :3]  # 丢弃Alpha透明通道
    elif encoding == "rgb8":
        img = arr.reshape(h, w, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding == "bgr8":
        img = arr.reshape(h, w, 3)
    else:
        print(f"不支持图像编码: {encoding}")
        return

    cv2.imshow("Rosbridge Camera View", img)
    # 按下 q 退出程序
    if cv2.waitKey(1) & 0xFF == ord('q'):
        client.terminate()

def on_connected():
    print("✅ Python3 成功连接 rosbridge server")
    topic = roslibpy.Topic(client, IMAGE_TOPIC, MSG_TYPE)
    topic.subscribe(image_callback)

client.on_ready(on_connected)
client.run()

try:
    while client.is_connected:
        pass
except KeyboardInterrupt:
    pass
finally:
    client.terminate()
    cv2.destroyAllWindows()
```

## 方案二：TCP 传输

主要思路：既然 rosbridge 的瓶颈在编码和协议，为什么不绕开 ROS，直接用 TCP 传 numpy 数组？
Python 2.7 端：订阅 ROS 图像话题 → 转成 numpy 数组 → pickle 序列化 → TCP 发送
Python 3.10 端：TCP 接收 → pickle 反序列化 → 得到 numpy 数组 → YOLO 推理

核心代码就两段。

### 1 服务端（Python 2.7）

订阅 ROS 图像话题，收到数据后转成 numpy，用 pickle 序列化后通过 TCP 发出去：

```python
#!/usr/bin/env python2
import rospy
import numpy as np
import socket
import threading
from sensor_msgs.msg import Image

TCP_PORT = 9999
conn = None
conn_lock = threading.Lock()

def send_frame(img_np):
    global conn
    with conn_lock:
        if conn is not None:
            try:
                img_bytes = img_np.tobytes()
                header = str(len(img_bytes)).encode("ascii").ljust(16)
                conn.sendall(header)
                conn.sendall(img_bytes)
            except Exception as e:
                print("client disconnected:", e)
                conn = None

def image_callback(msg):
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    h = msg.height
    w = msg.width
    img = raw.reshape(h, w, 4)[:, :, :3]
    send_frame(img)

def tcp_listener():
    global conn
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", TCP_PORT))
    sock.listen(1)
    print("TCP bridge server started 127.0.0.1:{}".format(TCP_PORT))
    while True:
        new_conn, addr = sock.accept()
        print("Py3 client connected from", addr)
        with conn_lock:
            conn = new_conn

if __name__ == "__main__":
    threading.Thread(target=tcp_listener).start()
    rospy.init_node("py2_tcp_bridge")
    rospy.Subscriber("/carla/ego_vehicle/rgb_front/image", Image, image_callback, queue_size=1)
    rospy.spin()
```

### 2 客户端（Python 3.10）

接收数据，反序列化，显示：

```python
import socket
import numpy as np
import cv2

TCP_PORT = 9999
HOST = "127.0.0.1"
IMG_HEIGHT = 600
IMG_WIDTH = 800

def recv_all(sock, length):
    buffer = b""
    while len(buffer) < length:
        chunk = sock.recv(length - len(buffer))
        if not chunk:
            raise ConnectionError("Server disconnected")
        buffer += chunk
    return buffer

def connect_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, TCP_PORT))
    print("Connected to server at {}:{}".format(HOST, TCP_PORT))
    return sock

def main():
    while True:
        try:
            sock = connect_socket()
            while True:
                header_buf = recv_all(sock, 16)
                data_len = int(header_buf.strip())
                buffer = recv_all(sock, data_len)
                img = np.frombuffer(buffer, dtype=np.uint8).reshape(IMG_HEIGHT, IMG_WIDTH, 3)
                cv2.imshow("Carla RGB", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    sock.close()
                    return
        except ConnectionError:
            print("Connection lost, reconnecting...")
            cv2.destroyAllWindows()
            import time
            time.sleep(1)
        except Exception as e:
            print("Error:", e)
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

这里有个关键细节：TCP 是流式协议，数据可能分成多个包到达，所以必须写一个 recv_all 函数确保收满指定字节数，否则 pickle 反序列化会报错。

## 延迟对比

方案一：rosbridge + roslibpy

方案二：TCP 传输

结论：tcp 基本延迟很低，rosbridge + roslibpy 的方案延迟太高。

说两句，做仿真经常遇到这种 "环境不兼容" 的问题，以前我第一反应是装包、配源、编译，折腾大半天。现在更倾向于用最薄的协议做数据传输，能省很多时间。

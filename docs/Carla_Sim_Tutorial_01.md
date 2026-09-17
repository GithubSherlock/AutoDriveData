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

# Carla仿真系列：8_相机与激光雷达融合，点云投影到图像

前面几期做的都是纯视觉（相机图像），这期引入激光雷达——把 3D 点云投影到相机图像上，实现传感器融合。这是自动驾驶感知的经典操作，也是"相机+雷达"融合方案的核心。

## 1 为什么做点云和图像融合

| 传感器 | 优势 | 劣势 |
| ---- | ---- | ---- |
|相机|颜色、纹理、语义丰富|没有深度，测距靠估计|
|激光雷达|精确的3D距离|没有颜色、稀疏|

融合 = 相机的"看得懂" + 雷达的"测得准"

点云投影到图像 → 图像上的每个目标都有了精确距离
 ↓
检测到"车" + 知道"车离我3米"

## 2 坐标系与传感器布局

自车坐标系（黑色框）= 车辆中心

激光雷达坐标系（红色框）= 与自车原点重合（z=2.4m 车顶）

相机坐标系（绿色框）= 前挡风玻璃处（x=2.4, y=0, z=1.0）

注意：ego和lidar原点重合，图上错开只是为了可视化

相机相对于自车的外参：

```yaml
transforms:
  -
    header:
      seq: 0
      stamp:
        secs: 11
        nsecs:  41401547
      frame_id: "ego_vehicle"
    child_frame_id: "ego_vehicle/rgb_main_front"
    transform:
      translation:
        x: 2.4
        y: 0.0
        z: 1.0
      rotation:
        x: -0.5
        y: 0.5
        z: -0.5
        w: 0.5
---
```

激光雷达相对于自车的外参：

```yaml
transforms:
  -
    header:
      seq: 0
      stamp:
        secs: 11
        nsecs:  41401547
      frame_id: "ego_vehicle"
    child_frame_id: "ego_vehicle/lidar"
    transform:
      translation:
        x: 0.0
        y: 0.0
        z: 2.4
      rotation:
        x: 0.0
        y: 0.0
        z: 0.0
        w: 1.0
```

## 3 融合原理：点云投影到图像

核心公式（3 步）：

第 1 步 点云转到相机坐标系

```python
# 相机相对自车的逆变换
T_ego_camera = inv(T_camera_ego)
# 点云（雷达系）→ 相机系
T_lidar_camera = T_ego_camera · T_lidar_ego
```

第 2 步 通过相机内参 K 投影到图像平面

```python
# p_image = K · P_camera（针孔模型）
points_proj = K · points_valid.T
```

第 3 步 除以 Z 归一化，得到像素坐标

```python
u = points_proj[:, 0] / points_proj[:, 2]
v = points_proj[:, 1] / points_proj[:, 2]
```

第 4 步 在图像上画圆可视化

每个点云投影到图像的对应位置，画个圆标记距离。

## 4 启动流程

```bash
# 终端1-3：启动环境 + Carla + ROS Bridge（同上）
# 终端4：图像+点云融合
cd /opt/carla_ws/sim_workspace/image_pointcloud
python image_pointcloud_fusion_py2.py
# 终端5：RVIZ 查看投影结果
rosrun rviz rviz -d carla_sim_config.rviz
```

## 5 融合效果

图像上每个物体都叠上了点云（带距离信息）

车辆、行人、建筑轮廓清晰可见

相机提供语义，雷达提供精确距离 

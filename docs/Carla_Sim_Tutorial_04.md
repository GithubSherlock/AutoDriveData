# Carla仿真系列：5_Carla 四路相机标定与BEV环视拼接

前面几期把环境搭好、加了NPC、跑通YOLO、装了四路相机，这期是系列的核心——把四路相机图像拼成一张完整的鸟瞰环视图。

## 1 为什么 BEV 环视重要

自动驾驶要看到车四周360度的情况，靠单路相机做不到。BEV（Bird’s Eye View）就是把前后左右四路相机的图像，投影到车顶俯视视角，拼成一张完整的环视图。

       前
       ↑
左 ←── 车 ──→ 右
       ↓ 
       后
       
4路相机 → 各自标定 → 投影到BEV → 拼接成360度环视

## 2 整体流程

启动环境（前几期一样）

    ↓
    
获取四路相机图像

    ↓
    
创建标定棋盘格

    ↓
    
每路相机计算 H 矩阵（单应矩阵）

    ↓
    
四路图像投影到 BEV 并拼接

第一步到第五步：准备阶段

环境启动、ROS Bridge、TCP图像传输、棋盘格创建，都跟前几期一样，按顺序执行：

```bash
# 终端1-2：启动容器 + Carla UE4
# 终端3：启动 ROS Bridge
# 终端4：多相机 TCP 服务端
python tcp_bridge/tcp_bridge_server_multi_camera_py2.py
# 终端5：接收图像
python tcp_bridge_client_py3.py
# 终端6：创建网格
python scripts/create_board_py2.py --vehicle-id 513
#这个ID是在启动 ROS Bridge时可以查看ego_vehicle参数
```

第六步：相机标定计算 H 矩阵

这是 BEV 拼接的核心。每路相机都需要计算一个单应矩阵 H，把相机视角的图像投影到 BEV 俯视视角。

```bash
cd /opt/carla_ws/sim_workspace/bev_surround
source activate py310
# 分别标定四个相机
python camera_bev_calib_py3.py --name front
python camera_bev_calib_py3.py --name back
python camera_bev_calib_py3.py --name left
python camera_bev_calib_py3.py --name right
```

标定原理：

相机图像中的棋盘格角点
    ↓ 检测角点建立 图像坐标 ↔ 世界坐标 对应关系
    ↓ 求解单应矩阵 H（3x3）
    ↓ 应用相机图像 → BEV 俯视图像

关键点： 四路相机必须用同一个世界坐标系（统一的地面平面），否则拼接时四个角会对不齐。

第七步：BEV 拼接

四路相机都标定好后，执行拼接：

```bash
python bev_surround_py3.py
```

运行后可以看到：
四路相机图像被投影到 BEV 视角
前后左右拼成完整的 360 度环视图
车辆周围的地面、车道线、障碍物都显示在鸟瞰图上
需要说明一下：目前只实现了 demo 的效果，没有对 4 个角进行优化对齐，所以效果不是很好。

## 3 几个关键点

| 要点 | 说明 |
| --- | --- |
| 统一坐标系 | 四路相机共用同一地面平面，拼接才对齐 |
| 棋盘格标定 | 标定板 / 标定布，角点检测要准确 |
| 重叠区域 | 相邻相机留 30 度重叠，拼接处做融合 |
| H 矩阵精度 | 标定不准，拼接就歪，这是最花时间的地方 |

## 4 下一期

做图像的语义分割，再将语义分割结果投影到 BEV 空间，得到语义 BEV。

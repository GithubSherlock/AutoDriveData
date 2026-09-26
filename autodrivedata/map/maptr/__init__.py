"""MapTR 参考自实现(§5.11 C/D 阶段,torch2.x 现代栈)。

忠实移植 MapTR 核心数学(GKT BEV 变换 + 分层 query head + 置换等价匹配 +
focal/L1),基础设施用现代栈(torch 2.13 + torchvision)。本包依赖 torch,
不进 autodrivedata 纯值包;依赖方向 autodrivedata(纯值)→ maptr_impl(训练)。

正确性锚点:单帧过拟合测试(小数据 loss → ~0)+ 投影链单测以 B3 已验证的
autodrivedata.calib.world_to_img 为 oracle。
"""

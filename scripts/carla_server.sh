#!/bin/bash
# CARLA 服务器生命周期辅助(项目纪律,源自 M3-5 排查结论)。
# 用法:
#   scripts/carla_server.sh start   # 干净启动(先杀残留 + 验证端口空闲)
#   scripts/carla_server.sh stop    # 彻底停止(UE4 会逃逸 su 包装,必须 pkill -9)
#   scripts/carla_server.sh status  # 状态(进程/端口/显存)
#
# M3-5 结论:headless 下客户端会话收尾(销毁传感器+断连)偶发 UE4 segfault(139),
# 但每次采集数据已完整落盘——崩溃只发生在 teardown 阶段。纪律:每次采集前 start。

CARLA_DIR=/root/autodl-tmp/CARLA_0.9.16
LOG=/tmp/carla_server.log
BIN_NAME=CarlaUE4-Linux-Shipping

kill_all() {
  pkill -9 -f "$BIN_NAME" 2>/dev/null || true
  for _ in $(seq 1 10); do
    pgrep -f "$BIN_NAME" >/dev/null 2>&1 || break
    sleep 1
  done
}

wait_ready() {
  for _ in $(seq 1 36); do
    python -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(3); print(c.get_server_version())" 2>/dev/null && return 0
    sleep 10
  done
  echo "❌ 服务器 6 分钟内未就绪" >&2
  return 1
}

case "$1" in
  start)
    kill_all
    if ss -tlnp 2>/dev/null | grep -q ":2000 "; then
      echo "❌ 端口 2000 仍被占用,请手动排查" >&2
      exit 1
    fi
    nohup su - carla -c "cd $CARLA_DIR && LD_PRELOAD=/tmp/libmhookshim.so ./CarlaUE4.sh -RenderOffScreen -quality-level=Low" > "$LOG" 2>&1 &
    wait_ready
    ;;
  stop)
    kill_all
    echo "✅ 已停止"
    ;;
  status)
    alive=$(pgrep -c -f "$BIN_NAME" 2>/dev/null || echo 0)
    echo "进程数(含本 shell 匹配,真实判定看显存): $alive"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
    python -c "import carla; print('server:', carla.Client('127.0.0.1',2000).get_server_version())" 2>/dev/null || echo "server: 未连接"
    ;;
  *)
    echo "用法: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac

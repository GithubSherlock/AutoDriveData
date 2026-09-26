#!/bin/bash
# CARLA 服务器生命周期辅助(项目纪律,源自 M3-5 排查结论)。
# 用法:
#   tools/carla_server.sh start   # 干净启动(先杀残留 + 验证端口空闲)
#   tools/carla_server.sh stop    # 彻底停止(UE4 会逃逸 su 包装,必须 pkill -9)
#   tools/carla_server.sh status  # 状态(进程/端口/显存/状态目录 shim 兼容层)
#
# M3-5 结论:headless 下客户端会话收尾(销毁传感器+断连)偶发 UE4 segfault(139),
# 但每次采集数据已完整落盘——崩溃只发生在 teardown 阶段。纪律:每次采集前 start。

CARLA_DIR=/root/autodl-tmp/CARLA_0.9.16
BIN_NAME=CarlaUE4-Linux-Shipping
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# 运行支撑物一律落在**项目内**(产出纪律:清系统盘/数据盘都不会带走它们;
# 目录被删也能自愈重建)。STATE 在 outputs/ 下(gitignore)。
STATE=$PROJECT_ROOT/outputs/carla
LOG=$STATE/carla_server.log
SHIM=$STATE/libmhookshim.so
SHIM_SRC=$PROJECT_ROOT/tools/gpu_fix/mhookshim.c

# M3-5b Vulkan 兼容层:宿主驱动升到 580.105.08 后,镜像里**该版本**的
# libnvidia-gpucomp 缺失(只有 580.76.05/580.82.07 是真实文件)→ Vulkan ICD
# (libGLX_nvidia.so.0)dlopen 失败 → 枚举 0 个 Vulkan 设备 → UE4 渲染线程起不来
# (GameThread timed out waiting for RenderThread + Signal 11),显存恒 0 MiB。
# 处置:用镜像自带副本按缺失的 SONAME 顶名,经本项目私有目录注入 LD_LIBRARY_PATH
# (不动 /usr/lib)。判据:`python bin/probe_vulkan.py` 应列出 RTX 3080 Ti(缺兼容层时只剩 llvmpipe)。
COMPAT_DIR=$STATE/nvidia-compat
GPUCOMP_FALLBACK=/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.580.76.05
COMPAT_LDPATH=

# LD_PRELOAD shim(malloc 钩子 + Xlib ErrorF 桩)缺了就现编——原来放 /tmp,清盘即失效
ensure_shim() {
  if [ -s "$SHIM" ]; then
    return 0
  fi
  echo "⚠️  $SHIM 缺失 → 现编(tools/gpu_fix/mhookshim.c)"
  gcc -shared -fPIC -O2 -o "$SHIM" "$SHIM_SRC" -ldl || {
    echo "❌ shim 编译失败(gcc 缺失?)" >&2
    exit 1
  }
}

setup_state() {
  mkdir -p "$STATE"
  ensure_shim
}

setup_gpucompat() {
  local ver real
  ver=$(basename "$(readlink -f /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0)" | sed 's/^libGLX_nvidia\.so\.//')
  real=/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so."$ver"
  if [ -s "$real" ]; then
    echo "✅ 驱动 $ver 的 libnvidia-gpucomp 完整,无需兼容层"
    return 0
  fi
  mkdir -p "$COMPAT_DIR"
  ln -sf "$GPUCOMP_FALLBACK" "$COMPAT_DIR/libnvidia-gpucomp.so.$ver"
  COMPAT_LDPATH="$COMPAT_DIR"
  echo "⚠️  驱动 $ver 缺 libnvidia-gpucomp → 兼容层 $(basename "$GPUCOMP_FALLBACK") 顶名"
}

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

# 端口占用探测:connect_ex 返回 0 = 连得上 = 有人监听。不用 ss——本机未安装
# (command not found),原 `ss -tlnp | grep ":2000 "` 恒为空 → start 守卫静默失效。
port_busy() {
  python -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(0 if not s.connect_ex(('127.0.0.1',2000)) else 1)" 2>/dev/null
}

case "$1" in
  start)
    kill_all
    if port_busy; then
      echo "❌ 端口 2000 仍被占用,请手动排查" >&2
      exit 1
    fi
    setup_state
    setup_gpucompat
    nohup su - carla -c "cd $CARLA_DIR && LD_LIBRARY_PATH=$COMPAT_LDPATH LD_PRELOAD=$SHIM ./CarlaUE4.sh -RenderOffScreen -quality-level=Low" > "$LOG" 2>&1 &
    wait_ready
    ;;
  stop)
    kill_all
    echo "✅ 已停止"
    ;;
  status)
    alive=$(pgrep -c -f "$BIN_NAME" 2>/dev/null || echo 0)
    echo "进程数(含本 shell 匹配,真实判定看显存): $alive"
    port_busy && echo "端口 2000: 被占用" || echo "端口 2000: 空闲"
    echo "状态目录: $STATE"
    [ -s "$SHIM" ] && echo "  shim: 已就绪" || echo "  shim: 缺失(start 时现编)"
    [ -d "$COMPAT_DIR" ] && echo "  兼容层: $(readlink -f "$COMPAT_DIR"/* 2>/dev/null | xargs -r basename)" || echo "  兼容层: 未启用(驱动完整时属正常)"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
    python -c "import carla; print('server:', carla.Client('127.0.0.1',2000).get_server_version())" 2>/dev/null || echo "server: 未连接"
    ;;
  *)
    echo "用法: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac

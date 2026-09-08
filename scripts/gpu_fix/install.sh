#!/bin/bash
# NVIDIA 用户态补齐 + shim 安装(平台驱动重装不完整的恢复脚本,2026-09-08)。
#
# 背景:平台重装驱动(580.76.05 Open Kernel Module)时用户态损坏:
#   1. /etc/vulkan/icd.d/nvidia_icd.json 与 /usr/share/glvnd/egl_vendor.d/10_nvidia.json 为 0 字节
#      → Vulkan/EGL 静默回退 mesa(llvmpipe),NVIDIA 不可见
#   2. 用户态缺 libnvidia-gpucomp(libGLX_nvidia 强依赖,缺失导致 dlopen 链失败)
#   3. NVIDIA glcore 强引用 glibc<2.34 的 __malloc_hook 族与 Xlib 内部 ErrorF
#      (裁剪容器 glibc 2.34+/libX11 不导出)→ dlopen undefined symbol
#
# 修复(本脚本):
#   1. 重建两个 0 字节 json
#   2. shim 源码编译为 .so,通过 LD_PRELOAD 注入(NVIDIA 库加载时补符号)
#
# 用法: bash install.sh   (编译 shim → /tmp/libmhookshim.so)
# 注意:json 修复写入系统目录,一次执行永久生效;shim 需在每次启动 CARLA 前存在。

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== 1/2 重建 ICD json(0 字节损坏修复)=="
cat > /etc/vulkan/icd.d/nvidia_icd.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0",
        "api_version" : "1.3.242"
    }
}
EOF
cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
echo "✅ nvidia_icd.json + 10_nvidia.json"

echo "== 2/2 编译 shim =="
gcc -shared -fPIC -O2 -o /tmp/libmhookshim.so "$HERE/mhookshim.c" -ldl
ls -l /tmp/libmhookshim.so
echo "✅ 完成。CARLA 启动需注入: LD_PRELOAD=/tmp/libmhookshim.so"

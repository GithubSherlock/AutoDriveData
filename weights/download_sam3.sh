#!/bin/bash
# SAM3 权重下载脚本（通过 ModelScope）
# 用法: bash download_sam3.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAM3_DIR="$SCRIPT_DIR/sam3"

# 检测 modelscope 是否已安装
if ! python -c "import modelscope" 2>/dev/null; then
    echo "📦 正在安装 modelscope..."
    pip install modelscope
    echo "✅ modelscope 安装完成"
else
    echo "✅ modelscope 已安装"
fi

# 创建 sam3 目录
mkdir -p "$SAM3_DIR"

# 在 sam3 目录中下载模型
cd "$SAM3_DIR"
echo "⬇️  正在从 ModelScope 下载 facebook/sam3 到 $SAM3_DIR ..."
modelscope download facebook/sam3 --local-dir "$SAM3_DIR"
echo "✅ SAM3 下载完成: $SAM3_DIR"

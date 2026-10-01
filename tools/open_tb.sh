#!/bin/bash
pkill -9 tensorboard
tensorboard --logdir outputs/tb --host 0.0.0.0 --port 6007

# 查看占用端口的进程
# ps aux | grep tensorboard
# 杀掉卡住的 tensorboard 进程, PID（第二列数字）
# kill -9 <PID>
# 批量一次性清理所有 tensorboard 进程
# pkill -9 tensorboard
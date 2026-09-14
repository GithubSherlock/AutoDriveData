#!/usr/bin/env bash
# 官方 MapTR/MapQR 复线的启动器(Plan.md §5.12 阶段 B/C)。
#
# 为什么要有这层壳 —— 官方 `tools/train.py` 的插件加载是**字符串拼接**:
#     plugin_dir = 'projects/mmdet3d_plugin/'
#     sys.path.insert(0, os.path.dirname(plugin_dir)) → import_module('projects')
# 于是 plugin_dir **只能是相对路径**(绝对路径会拼成 `.root.autodl-tmp...` 直接炸),
# 且 cwd 必须是仓库根;而项目侧 config 又必须能被 `custom_imports` 找到 → PYTHONPATH
# 同时含**仓库根**与**项目根**。三处约束靠人记必出错,故固化成脚本。
#
# 用法:
#   bash bin/run_official.sh train maptrv2 [--resume] [--bg]  # --resume=有 latest 就接着训
#   bash bin/run_official.sh chain [--bg]   # 三线串行(v1 → mapqr → maptrv2),失败即停
#   bash bin/run_official.sh test  maptrv2 [ckpt|best]        # 缺省=work_dir/latest.pth
#   bash bin/run_official.sh list
# 日志:outputs/maptr_official/logs/ 下 `<impl>_<cmd>[_tag]_<时间戳>.log`(项目内)。
# chain 用 `<impl>.done` 标记跳过已完成的线(中断后重跑同一条命令即续),每条线训完自动
# 补 best/final 两次官方评测 → 链跑完时所有记账数字都已在盘上,不必回头手补。
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 自身绝对路径:后台化与 chain 里要重入自己,用绝对路径 + 显式解释器 → 不依赖执行位、不依赖 cwd
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ENV="${MAPTR_ENV:-/root/autodl-tmp/envs/maptr_official}"
PY="$ENV/bin/python"
LOGDIR="$PROJ/outputs/maptr_official/logs"
# 后台化时会重新 exec 自己一次:时间戳经环境变量传下去,两次算出的 LOG 才是同一个文件
TS="${RUN_OFFICIAL_TS:-$(date +%Y%m%d_%H%M%S)}"
export RUN_OFFICIAL_TS="$TS"

usage() {
  sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-2}"
}

impl_repo() {  # $1=impl → 仓库根
  case "$1" in
    maptrv2) echo "$PROJ/hdMapGitHub/MapTR_maptrv2" ;;   # origin/maptrv2 e03f097
    mapqr) echo "$PROJ/hdMapGitHub/MapQR" ;;             # d1d9f38
    maptr_v1) echo "$PROJ/hdMapGitHub/MapTR" ;;          # main a6872d8
    *) echo "未知实现:$1(可选 maptrv2|mapqr|maptr_v1)" >&2; exit 2 ;;
  esac
}

cmd="${1:-}"
case "$cmd" in
  list)
    for i in maptrv2 mapqr maptr_v1; do printf '%-10s %s\n' "$i" "$(impl_repo "$i")"; done
    exit 0
    ;;
  train | test | chain) ;;
  *) usage 2 ;;
esac

# ---- 参数解析:位置参数(impl)+ 标志位(--resume/--bg)------------------------------
RESUME=0
BG=0
POS=()
for a in "${@:2}"; do
  case "$a" in
    --resume) RESUME=1 ;;
    --bg) BG=1 ;;
    -*) echo "未知标志:$a" >&2; usage 2 ;;
    *) POS+=("$a") ;;
  esac
done
IMPL="${POS[0]:-}"     # 第 1 位置参数:实现名
CKPT="${POS[1]:-}"     # 第 2 位置参数(仅 test 用):权重路径或 `best`

mkdir -p "$LOGDIR"

# ---- 后台化:重新 exec 自己一次(setsid 断开终端,SSH 断了也不掉)--------------------
# 用环境变量做哨兵,避免无限自我重启;日志仍由内层 tee 写,故最外层只打印落点。
if [ "$BG" = 1 ] && [ -z "${RUN_OFFICIAL_BG:-}" ]; then
  export RUN_OFFICIAL_BG=1
  # 后台进程的 stdout **不能**丢到 /dev/null:chain 的"哪条线开始/结束/失败"只在它自己的
  # stdout 上(各线训练日志另由 tee 落盘)→ 落 `*_boot_<TS>.log`,90 小时的无人值守才有据可查
  BOOT_LOG="$LOGDIR/${IMPL:-chain}_${cmd}_boot_${TS}.log"
  setsid nohup "$BASH" "$SELF" "$@" >"$BOOT_LOG" 2>&1 &
  echo "[run_official] 后台启动 pid=$! → 总进度 $BOOT_LOG"
  echo "[run_official] 各线训练/评测日志:$LOGDIR/<impl>_{train,test}_*_${TS}.log"
  exit 0
fi

work_dir_of() { echo "$PROJ/outputs/maptr_official/$1"; }

# ---- chain:三条线串行,任一失败立即中止(不带着坏结果往下跑)------------------------
if [ "$cmd" = chain ]; then
  for i in maptr_v1 mapqr maptrv2; do
    if [ -f "$LOGDIR/$i.done" ]; then
      echo "[chain] 跳过 $i(已有 $LOGDIR/$i.done)"
      continue
    fi
    echo "[chain] ===== $i 训练开始 $(date '+%F %T') ====="
    if ! "$SELF" train "$i" --resume; then
      echo "[chain] $i **失败** → 中止后续线(修好后重跑 chain 会从 $i 续)" >&2
      exit 1
    fi
    touch "$LOGDIR/$i.done"
    echo "[chain] ===== $i 训练结束 $(date '+%F %T'),补评测 ====="
    # best 与 final 各评一次:前者=记账口径,后者=与自实现同轮次口径
    "$SELF" test "$i" best || echo "[chain] 警告:$i best 评测失败(训练已成功,不中止)"
    "$SELF" test "$i" latest || echo "[chain] 警告:$i final 评测失败(训练已成功,不中止)"
    echo "[chain] ===== $i 全部完成 $(date '+%F %T') ====="
  done
  exit 0
fi

[ -n "$IMPL" ] || usage 2
REPO="$(impl_repo "$IMPL")"
CFG="$PROJ/maptr_official/configs/${IMPL}_carla.py"
WORK_DIR="$(work_dir_of "$IMPL")"
[ -f "$CFG" ] || { echo "缺配置:$CFG" >&2; exit 2; }
[ -x "$PY" ] || { echo "缺环境:$ENV(先跑 bin/setup_maptr_official.sh)" >&2; exit 2; }

case "$cmd" in
  train)
    LOG="$LOGDIR/${IMPL}_train_${TS}.log"
    echo "[run_official] $IMPL 训练 → $LOG"
    echo "[run_official] cwd=$REPO cfg=$CFG"
    cd "$REPO"
    export PYTHONPATH="$REPO:$PROJ${PYTHONPATH:+:$PYTHONPATH}"
    # 续训:--resume 只在真有 latest.pth 时才传(首跑即普通冷启动,同一个命令两种场合都对)
    resume_arg=()
    if [ "$RESUME" = 1 ] && [ -f "$WORK_DIR/latest.pth" ]; then
      resume_arg=(--resume-from "$WORK_DIR/latest.pth")
      echo "[run_official] 续训自 $WORK_DIR/latest.pth"
    fi
    # 官方 train.py 各版本都吃 `--no-validate` / `--seed`;此处不传,用 config 里的值
    "$PY" tools/train.py "$CFG" "${resume_arg[@]}" 2>&1 | tee "$LOG"
    ;;
  test)
    ckpt="$CKPT"
    # `best` 是 mmcv EvalHook 按 save_best='NuscMap_chamfer/mAP' 落的那一份(指标名带 `/`
    # → 它自己建了子目录)。记账口径 = best;与自实现同轮次口径 = latest。
    if [ "$ckpt" = best ]; then
      ckpt="$(ls -1 "$WORK_DIR"/best_NuscMap_chamfer/mAP_*.pth 2>/dev/null | tail -1)"
      [ -n "$ckpt" ] || { echo "没有 best 权重($WORK_DIR/best_NuscMap_chamfer/)" >&2; exit 2; }
    fi
    [ -n "$ckpt" ] || ckpt="$WORK_DIR/latest.pth"
    [ -f "$ckpt" ] || { echo "缺权重:$ckpt" >&2; exit 2; }
    tag="$(basename "$ckpt" .pth)"
    LOG="$LOGDIR/${IMPL}_test_${tag}_${TS}.log"
    echo "[run_official] $IMPL 评测 $ckpt → $LOG"
    # 单卡**不能**直接跑 `tools/test.py`:三个仓库的单卡分支都被官方写成 `assert False`
    # (test.py:225,只留分布式路径)。三个坑叠在一起,故这里手工起**单进程分布式**:
    # ① `--launcher none` 会被上面那个 assert 挡掉;而任何分布式初始化都会踩
    #    mmcv/runner/dist_utils.py:16 —— `init_dist` 在 start method 未设时强制 `spawn`,
    #    spawn 下 DataLoader 要 pickle 整个 dataset,官方 dataset 挂着
    #    `eval_detection_configs`(nuScenes `DetectionConfig`,内含 dict_keys)→
    #    `TypeError: cannot pickle 'dict_keys' object`(训练期 EvalHook 走 fork,从不暴露)。
    #    绕法 = `data.workers_per_gpu=0`(不开 worker 进程 → 根本不 pickle dataset;
    #    评测 100 帧,顺序读图代价可忽略)
    # ② 官方 `tools/dist_test.sh` 尾巴硬编码 `--eval bbox`,argparse 取最后一次 → 覆盖 chamfer
    # ③ `jsonfile_prefix='test/...'` 与 `args.tmpdir='tmp'` 都是**相对 cwd** → 从仓库根跑会往
    #    pristine 的官方仓库写文件
    # 于是:env 自举单进程分布式 + cwd 设 work_dir(相对路径产物全落项目内)+ 不开 DataLoader
    # worker。插件导入靠 PYTHONPATH,与 cwd 无关。
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    export PYTHONPATH="$REPO:$PROJ${PYTHONPATH:+:$PYTHONPATH}"
    # 官方 README 口径就是 `--eval chamfer`;不能加 --format-only(test.py:120 明确互斥)
    RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 \
      MASTER_PORT="${MAPTR_TEST_PORT:-29517}" \
      "$PY" "$REPO/tools/test.py" "$CFG" "$ckpt" --launcher pytorch \
      --cfg-options data.workers_per_gpu=0 --eval chamfer 2>&1 | tee "$LOG"
    ;;
esac

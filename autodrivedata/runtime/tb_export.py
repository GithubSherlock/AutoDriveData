"""`logs/*.jsonl` → TensorBoard event 文件(**离线转换**,不碰训练循环)。

## 为什么是离线转换,而不是在训练里实时写 SummaryWriter

两个理由,第二个是决定性的:

1. **两份记录迟早漂。** `logs/*.jsonl` 已经是**逐迭代的唯一口径**(`runlog.metric` 逐行
   flush ⇒ 训练被中断也能查到跑到第几帧)。再在训练循环里并排开一路 event 文件,就是
   "同一个事实两处维护" —— 本项目对它的过敏有前科(`SAFETY_FACTOR` 两处不同值被专门写进
   `runtime/device.py` 头注)。**event 文件在本设计里是 view,不是 source。**
2. **实时写对已经在跑的 run 无效** —— 得停训练重启。而转换器能把**已产出**的 epoch 全部补上,
   还能把归档的旧 run 一起拉进来叠图对比(K=3 / 单帧基线 / 这次 MapQR)。

## 本模块**不认任何训练语义**

`runlog.metric(step, **kv)` 是全仓统一的契约 ⇒ 这里只做"jsonl 行 → scalar tag",
对 `train_maptr` / `eval_slam` / `eval_maptr` 一视同仁。一份代码能同时画 `loss`、`ATE`、`closure`。

| 概念 | 取法 |
|---|---|
| x 轴(step) | 行里的 `step`(runlog 契约:它就是**迭代标识**) |
| 标签(tag) | 除 `step/epoch/frame/t` 外的**每个键各一条 scalar**;另加 `wall_s` = `t` |
| run 名 | `路径=名字` 里的名字;不写则取文件名 |

## ⚠️ 标签是**稀疏**的(这个坑不处理就会画出假曲线)

早停复核写的是**另一行、同一个 step**:

```json
{"t": 135.1, "step": 1,  "epoch": 1,  "loss": 18.04, "cls": 0.039, "pts": 18.0, "lr": 1e-4}
{"t": 900.2, "step": 20,              "confirm_mAP": 0.31, "best_ap": 0.31}   ← 没有 loss/lr
```

⇒ **必须按"这行有哪些键就写哪些 tag"**处理。写成"固定取 loss/cls/pts/lr"就会 KeyError;
写成"缺的补 0"会在曲线上拉出**假台阶**(loss 突然掉到 0 再弹回来)。

另:`confirm_mAP` / `best_ap` 每 `confirm-every` 个 epoch 才有一个点 ⇒ 在 TB 里是**散点**,
不要当折线读。

**TB 的 scalar 存 float32**(实测 jsonl 里的 `18.04` 读回 `18.040000915527344`)⇒ 图与 jsonl
**不是逐位一致**的。对 AP 量级够用(0.3 处仍有 ~1e-7 分辨率,远细于 2e-3 的复现性下限),
但**别拿 TB 读数去比 1e-6** —— 要精读回到 `.jsonl` 本身。

## ⚠️ event 文件是**快照**

训练在跑时,每导出一次就是一张当时的快照。**刷新要重跑本命令**,且 TB 进程要么重启、
要么等它自己重载 —— 所以**别把 TB 当成"实时看板"**:想看最新进度先 `tail` 一下 `.jsonl`。
(另一种做法是在训练循环里实时写 `SummaryWriter`,但那条路被上面第 1 条否掉了。)

用法:
  pip install tensorboard
  python -m autodrivedata.runtime.tb_export \\
      logs/map_train_maptr_20260930-005454.jsonl=mapqr_new \\
      'logs/map_train_maptr_20260929-*.jsonl=k3_baseline' \\
      --out outputs/tb
  tensorboard --logdir outputs/tb --host 127.0.0.1 --port 6006
  # 远程:ssh -L 6006:127.0.0.1:6006 <autodl> → 浏览器 127.0.0.1:6006
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from autodrivedata.utils import runlog
from autodrivedata.utils.paths import project_path

#: 这些键**不做 scalar**:前四个是"步"的不同写法(步轴自己会用),`t` 另以 `wall_s` 出。
_STEP_LIKE = ("step", "epoch", "frame", "t")


def parse_spec(spec: str) -> tuple[str, str | None]:
    """`路径[=名字]` → `(路径模式, 名字)`。

    `=` 取**最后一个**且要求左边确实是个存在的路径模式 —— 否则路径里带 `=` 就会被切错
    (本项目已有一次"`with_suffix` 把 `k3_v1.0` 的 `.0` 当扩展名吃掉"的同族教训)。
    """
    if "=" in spec:
        left, _, right = spec.rpartition("=")
        if left and right and glob.glob(left):
            return left, right
    return spec, None


def expand(spec: str) -> list[tuple[Path, str]]:
    """展开一个 spec → `[(jsonl 路径, run 名)]`(**一个 jsonl = 一个 run**)。

    glob 命中多个时**不合并**:合并会把两段不同的训练交替写进同一条曲线,而曲线看着正常。
    名字按文件名区分,并打印实际决定 —— 静默的命名规则等于没有规则。
    """
    pattern, name = parse_spec(spec)
    hits = sorted(Path(p) for p in glob.glob(pattern))
    if not hits:
        raise SystemExit(f"{spec}: 没有匹配到任何文件(模式 {pattern!r})")
    globbed = any(c in pattern for c in "*?[")
    out: list[tuple[Path, str]] = []
    for i, p in enumerate(hits):
        if name and len(hits) == 1:
            run = name
        elif name:  # 一个名字配多个文件:加**文件名词干**后缀,免得互相覆盖
            run = f"{name}__{p.stem}"
        else:
            run = p.stem
        if globbed and name and len(hits) > 1:
            # 只提示一次,别在循环里刷屏
            if i == 0:
                print(f"[warn] {pattern!r} 命中 {len(hits)} 个文件 → 拆成 {len(hits)} 个 run(不合并)")
        out.append((p, run))
    return out


def read_rows(p: Path) -> list[dict]:
    """jsonl → 行列表。**空文件/坏行不抛** —— 空 jsonl 是合法产物(那次跑没有逐迭代指标)。"""
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # 中断写了一半的行
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def step_of(row: dict) -> float | None:
    """x 轴取值:`step`(runlog 契约的迭代标识)> `epoch` > `frame` > `t`。"""
    for k in _STEP_LIKE:
        v = row.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def tags_of(row: dict) -> dict[str, float]:
    """该行**实际带**的 scalar(稀疏:只写有的) + `wall_s`。"""
    out: dict[str, float] = {}
    for k, v in row.items():
        if k in _STEP_LIKE:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    if isinstance(row.get("t"), (int, float)):
        out["wall_s"] = float(row["t"])
    return out


def sibling_meta(p: Path) -> dict:
    """同 stem 的 `logs/*.json`(runlog 三件套之一)里的环境指纹与 highlights。

    放进 TB 的 `meta/` 文本里 —— **同一个 run 的结论数字必须跟曲线同屏**,
    否则看图的人无从判断这条曲线属于哪个 batch / 哪个变体(那是本项目 AP 口径纪律的前提)。
    """
    j = p.with_suffix(".json")
    if not j.exists():
        return {}
    try:
        doc = json.loads(j.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    keep = {k: doc[k] for k in ("script", "argv", "git", "gpu", "env") if k in doc}
    if doc.get("highlights"):
        keep["highlights"] = doc["highlights"]
    return keep


def export_one(p: Path, run: str, out_root: Path, rl) -> dict:
    """一个 jsonl → `out_root/<run>/`。返回统计(供总表与数值自证)。"""
    # 走 `writer` 子模块而不是 `torch.utils.tensorboard` 包根:后者是**再导出**,
    # 静态检查会报 reportPrivateImportUsage(运行时两者都行)。
    from torch.utils.tensorboard.writer import SummaryWriter

    rows = read_rows(p)
    d = out_root / run
    d.mkdir(parents=True, exist_ok=True)
    n_scalar, n_row, tags = 0, 0, set()
    skipped_no_step = 0
    with SummaryWriter(log_dir=str(d)) as w:
        for row in rows:
            s = step_of(row)
            if s is None:
                skipped_no_step += 1  # 没有步标识的行无处安放 —— **数出来**,不静默丢
                continue
            n_row += 1
            for k, v in tags_of(row).items():
                w.add_scalar(k, v, int(s))
                n_scalar += 1
                tags.add(k)
        meta = sibling_meta(p)
        if meta:
            w.add_text("meta/runlog", json.dumps(meta, ensure_ascii=False, indent=1))
    stat = {
        "run": run,
        "src": str(p),
        "n_rows": n_row,
        "n_scalars": n_scalar,
        "n_skipped_no_step": skipped_no_step,
        "tags": sorted(tags),
        "out": str(d),
    }
    print(f"  [{run}] {n_row} 行 → {n_scalar} 点  tags={sorted(tags)}  跳过(无步标识)={skipped_no_step}")
    rl.metric(n_scalar, run=run, n_rows=n_row, n_scalars=n_scalar, n_tags=len(tags))
    return stat


def main() -> None:
    ap = argparse.ArgumentParser(description="logs/*.jsonl → TensorBoard event(离线;详见模块头注)")
    ap.add_argument("specs", nargs="+", help="`路径[=run名]`;路径可为 glob,建议加引号自己展开")
    ap.add_argument("--out", default="outputs/tb", help="event 输出根(每个 run 一个子目录)")
    ap.add_argument("--no-runlog", action="store_true")
    args = ap.parse_args()

    with runlog.run("autodrivedata.runtime.tb_export") as rl:
        out_root = project_path(args.out)
        out_root.mkdir(parents=True, exist_ok=True)
        pairs: list[tuple[Path, str]] = []
        for spec in args.specs:
            pairs.extend(expand(spec))
        # run 名撞车 = 后写的**静默覆盖**前一个,而 TB 里只看得到一条曲线
        seen: dict[str, Path] = {}
        for p, run in pairs:
            if run in seen:
                raise SystemExit(f"run 名撞车:{run!r} 同时来自 {seen[run]} 与 {p};给其中一个换名字")
            seen[run] = p

        print(f"[tb] {len(pairs)} 个 run → {out_root}")
        stats = []
        for p, run in pairs:
            rl.input(str(p), "runlog-jsonl")
            stats.append(export_one(p, run, out_root, rl))
        empty = [s["run"] for s in stats if s["n_rows"] == 0]
        if empty:
            rl.note(f"{len(empty)} 个 run 一个点都没写出(空 jsonl 或整文件无步标识):{empty}")
        rl.highlight("n_runs", len(stats))
        rl.highlight("n_scalars_total", sum(s["n_scalars"] for s in stats))
        rl.artifact_dir(out_root, "tb-events")
        print(f"\n下一步:\n  tensorboard --logdir {out_root} --host 127.0.0.1 --port 6006")
        print("  远程:ssh -L 6006:127.0.0.1:6006 <autodl> → 浏览器 127.0.0.1:6006")


if __name__ == "__main__":
    main()

"""项目路径锚定(纯值:只用 pathlib,不 import carla/torch)。

**产出逻辑纪律:所有产物落在项目内**。历史口径是"相对 cwd 的 `outputs/...`"——
从别处 cwd 调用(或换会话)就会把模型/可视化/采集结果散到项目外,清盘时无从分辨。
现口径:写盘路径一律经 `project_path()`,**相对路径解释为相对项目根**,绝对路径原样放行
(显式给绝对路径 = 用户明确要求写到别处)。

读路径(输入)不走这里——输入沿用 cwd 口径,便于 `cd` 到数据集目录临时跑。
"""

from __future__ import annotations

from pathlib import Path

# 本文件在 <项目根>/autodrivedata/paths.py → 上两级即项目根
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"  # 采集/训练/可视化产物的单一落点(gitignore)
STATE = OUTPUTS / "carla"  # CARLA 运行支撑物(shim/compat/服务器日志)


def project_path(p: str | Path) -> Path:
    """输出路径锚定项目根:相对 → PROJECT_ROOT/p;绝对 → 原样。"""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_parent(p: str | Path) -> Path:
    """锚定 + 建父目录(写盘前一步到位)。"""
    path = project_path(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

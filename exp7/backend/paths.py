"""路径锚定与产物归档。

两个必须绕开的坑：
- ONNX 路径含非 ASCII 字符时，cv2.FaceDetectorYN / FaceRecognizerSF 直接报
  "Can't read ONNX file"；imread 那套 unicode 封装对 dnn 加载无效。这里把模型
  复制到系统临时目录（ASCII）再交给 OpenCV。
- 相对路径依赖 cwd，换目录运行就写出散落的 outputs/。这里一律以 exp7/ 为锚点解析。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Union

BACKEND_ROOT = Path(__file__).resolve().parent
EXP7_ROOT = BACKEND_ROOT.parent
FRONTEND_ROOT = EXP7_ROOT / "frontend"
OUTPUTS_ROOT = EXP7_ROOT / "outputs"

ModelPath = Union[str, Path]


def resolve(path: ModelPath) -> Path:
    """相对路径按 exp7/ 解析，绝对路径原样返回。"""
    p = Path(path)
    return p if p.is_absolute() else EXP7_ROOT / p


def model_path(path: ModelPath) -> str:
    """返回 OpenCV 能加载的模型路径。

    项目路径含中文时复制一份到临时 ASCII 目录，之后复用同一份缓存。
    """
    src = resolve(path)
    if not src.is_file():
        raise FileNotFoundError(f"模型不存在: {src}")
    if str(src).isascii():
        return str(src)

    cache_dir = Path(tempfile.gettempdir()) / "exp7_ascii_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / src.name
    if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return str(dst)


def new_run_dir(name: str) -> Path:
    """创建 outputs/<时间戳>_<name>/ 并写入运行元数据，保证产物可追溯。"""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run = OUTPUTS_ROOT / f"{stamp}_{name}"
    run.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
    }
    (run / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run

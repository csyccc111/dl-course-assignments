# -*- coding: utf-8 -*-
"""
io_utils.py —— 中文路径安全的图片读写 + 目录/文件枚举
=====================================================

Windows 下 `cv2.imread` / `cv2.imwrite` 走的是单字节 ANSI 路径，
遇到中文目录（比如本项目的 `深度学习课程实验`）会直接返回 None / 静默失败。
这里统一用 `np.fromfile + cv2.imdecode` 绕开，工程里所有读写都走这两个函数。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """读取图片，支持中文路径。失败返回 None。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite_unicode(path, img, quality: int = 95) -> bool:
    """写出图片，支持中文路径。返回是否成功。"""
    if img is None:
        return False
    path = Path(path)
    ensure_dir(path.parent)
    ext = path.suffix.lower() or ".jpg"
    if ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    elif ext == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    else:
        params = []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except OSError:
        return False


def ensure_dir(path) -> Path:
    path = Path(path)
    if str(path) not in ("", "."):
        path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(root, recursive: bool = True) -> List[Path]:
    """列出目录下所有图片，按路径排序（保证多次运行顺序一致）。"""
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTS else []

    pattern = "**/*" if recursive else "*"
    files = [p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: str(p).lower())


def rel_name(path: Path, root) -> str:
    """相对路径（用于输出文件名，避免不同子目录同名覆盖）。"""
    try:
        rel = Path(path).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        rel = Path(path).name
    return str(rel).replace(os.sep, "__").replace("/", "__")

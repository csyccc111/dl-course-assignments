# -*- coding: utf-8 -*-
"""
download_models.py —— 拉取 YuNet / CenterFace 的 onnx 权重
==========================================================

    python download_models.py

模型会放到 ./models/ 下：
    models/face_detection_yunet_2023mar.onnx   ~230 KB   (opencv_zoo)
    models/centerface.onnx                      ~7.3 MB   (Star-Clouds/CenterFace)

两个仓库都用 Git LFS 存权重，直接下 GitHub 的 raw 链接时，如果被重定向到
LFS 指针文件而不是真实二进制，文件只有 130 字节左右且以
"version https://git-lfs.github.com/spec/v1" 开头。脚本会自动识别并换备用地址。
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

MODELS = [
    (
        "face_detection_yunet_2023mar.onnx",
        [
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
        100_000,  # 期望的最小字节数
    ),
    (
        "centerface.onnx",
        [
            "https://github.com/Star-Clouds/CenterFace/raw/master/models/onnx/centerface.onnx",
            "https://media.githubusercontent.com/media/Star-Clouds/CenterFace/master/models/onnx/centerface.onnx",
            "https://raw.githubusercontent.com/Star-Clouds/CenterFace/master/models/onnx/centerface.onnx",
        ],
        1_000_000,
    ),
]

LFS_MAGIC = b"version https://git-lfs"


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = 100.0 * done / total_size
    bar = "#" * int(pct / 3)
    sys.stdout.write("\r    [{:<33}] {:5.1f}%  {:.1f}/{:.1f} MB".format(
        bar, pct, done / 1e6, total_size / 1e6))
    sys.stdout.flush()


def fetch(urls, dest: Path, min_size: int) -> bool:
    for i, url in enumerate(urls, 1):
        print("  尝试 {}/{}: {}".format(i, len(urls), url))
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp, _progress)
            print()
        except Exception as e:  # noqa: BLE001
            print("\n    下载失败: {}".format(e))
            if tmp.exists():
                tmp.unlink()
            continue

        if not tmp.exists():
            continue

        size = tmp.stat().st_size
        with open(tmp, "rb") as f:
            head = f.read(32)

        if head.startswith(LFS_MAGIC):
            print("    拿到的是 Git LFS 指针文件（{} 字节），换下一个地址".format(size))
            tmp.unlink()
            continue

        if size < min_size:
            print("    文件过小（{} 字节 < {}），可能被限流或截断，换下一个地址".format(size, min_size))
            tmp.unlink()
            continue

        os.replace(str(tmp), str(dest))
        print("    OK  {:.2f} MB -> {}".format(size / 1e6, dest))
        return True

    return False


def main() -> int:
    root = Path(__file__).resolve().parent
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("模型目录: {}\n".format(model_dir))
    ok_all = True
    for name, urls, min_size in MODELS:
        dest = model_dir / name
        if dest.exists() and dest.stat().st_size >= min_size:
            print("[skip] {} 已存在（{:.2f} MB）".format(name, dest.stat().st_size / 1e6))
            continue

        print("[get ] {}".format(name))
        if not fetch(urls, dest, min_size):
            print("  !! {} 下载失败，请手动下载后放到 {}\n".format(name, dest))
            ok_all = False
            continue
        print()

    if ok_all:
        # 顺手验证 onnx 能被 OpenCV 读进来
        try:
            import cv2
            for name, _, _ in MODELS:
                net = cv2.dnn.readNetFromONNX(str(model_dir / name))
                print("[check] {} 可被 cv2.dnn 读取".format(name))
        except Exception as e:  # noqa: BLE001
            print("[check] 校验时出现问题：{}".format(e))

        print("\n全部就绪。下一步：")
        print("    python detect_align.py image --input demo.jpg")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

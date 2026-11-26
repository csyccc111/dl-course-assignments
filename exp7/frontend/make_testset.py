# -*- coding: utf-8 -*-
"""
make_testset.py —— 构建实验用测试集（原始 + 合成退化）
======================================================

真实场景里最常让检测/对齐掉链子的是这五类：小脸、遮挡、模糊、大滚转、暗光。
公开数据集下载动辄几个 G，课程实验没必要；这里用「真实人脸图 + 可控退化」的方式
构造一个小而可解释的压力测试集，每一类退化都对应一条明确的工程问题：

    small   整图缩到 1/4 再贴回原画布   -> 小脸 / 远距离监控
    occl    在脸部区域盖黑块            -> 口罩、手遮挡、头发遮挡
    blur    高斯模糊 σ=9                -> 运动模糊 / 失焦
    rot20   旋转 +20 度                 -> 歪头 / 相机倾斜
    dark    gamma=0.35                  -> 逆光、夜间
    jpeg10  JPEG 质量 10                -> 网络传输压缩

用法：
    python make_testset.py --src data/base --out data/test --hard data/test_hard
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from face_frontend.io_utils import ensure_dir, imread_unicode, imwrite_unicode, list_images


def _paste_small(img: np.ndarray, scale: float = 0.25) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), np.uint8)
    y0 = (h - small.shape[0]) // 2
    x0 = (w - small.shape[1]) // 2
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return canvas


def _occlude(img: np.ndarray, faces) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    if not faces:
        # 检测不到就退化成「中央偏下盖一块」，至少保证图上有遮挡物
        bw, bh = int(w * 0.28), int(h * 0.18)
        x0, y0 = (w - bw) // 2, int(h * 0.55)
        cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + bh), (0, 0, 0), -1)
        return out

    for f in faces:
        fw, fh = f.width, f.height
        # 盖住下半脸（嘴鼻区域），模拟口罩
        x0 = int(f.x1 + fw * 0.12)
        x1 = int(f.x2 - fw * 0.12)
        y0 = int(f.y1 + fh * 0.52)
        y1 = int(f.y2 - fh * 0.04)
        cv2.rectangle(out, (max(0, x0), max(0, y0)), (min(w - 1, x1), min(h - 1, y1)), (25, 25, 25), -1)
    return out


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


def _gamma(img: np.ndarray, g: float) -> np.ndarray:
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def _jpeg(img: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/base")
    ap.add_argument("--out", default="data/test")
    ap.add_argument("--hard", default="data/test_hard")
    ap.add_argument("--yunet-model", default="models/face_detection_yunet_2023mar.onnx")
    ap.add_argument("--tile", type=int, default=2, help="把每张图拼成 tile x tile 的多脸图")
    args = ap.parse_args()

    src_files = list_images(args.src)
    if not src_files:
        raise SystemExit("{} 里没有图片".format(args.src))

    out_dir = ensure_dir(args.out)
    hard_dir = ensure_dir(args.hard)

    # 用 YuNet 找脸，做「有依据」的遮挡
    det = None
    try:
        from face_frontend.detectors import YuNetDetector
        det = YuNetDetector(model_path=args.yunet_model, conf_thresh=0.5)
    except Exception as e:  # noqa: BLE001
        print("提示：YuNet 不可用（{}），遮挡块将退化为固定区域".format(e))

    random.seed(0)
    n_orig = n_hard = 0

    for k, p in enumerate(src_files):
        img = imread_unicode(p)
        if img is None:
            print("跳过（读不出）：{}".format(p))
            continue

        stem = "t{:02d}_{}".format(k, p.stem)

        # 1) 原始图
        imwrite_unicode(out_dir / "{}.jpg".format(stem), img)
        n_orig += 1

        # 2) 多脸拼图（tile x tile，模拟多人场景）
        if args.tile > 1:
            th, tw = img.shape[0] // args.tile, img.shape[1] // args.tile
            tiles = [cv2.resize(img, (tw, th)) for _ in range(args.tile * args.tile)]
            rows = [np.hstack(tiles[r * args.tile:(r + 1) * args.tile]) for r in range(args.tile)]
            grid = np.vstack(rows)
            imwrite_unicode(out_dir / "{}_multi.jpg".format(stem), grid)
            n_orig += 1

        # 3) 合成退化
        faces = []
        if det is not None:
            try:
                faces = det.detect(img)
            except Exception:  # noqa: BLE001
                faces = []

        variants = {
            "small": _paste_small(img, 0.25),
            "occl": _occlude(img, faces),
            "blur": cv2.GaussianBlur(img, (0, 0), 9.0),
            "rot20": _rotate(img, 20.0),
            "dark": _gamma(img, 0.35),
            "jpeg10": _jpeg(img, 10),
        }
        for tag, im in variants.items():
            imwrite_unicode(hard_dir / "{}_{}.jpg".format(stem, tag), im)
            n_hard += 1

        print("{}: 原图 {} 张，退化 {} 张".format(p.name, 2 if args.tile > 1 else 1, len(variants)))

    print("\n完成：{} 张原图 -> {}，{} 张退化图 -> {}".format(n_orig, out_dir, n_hard, hard_dir))


if __name__ == "__main__":
    main()

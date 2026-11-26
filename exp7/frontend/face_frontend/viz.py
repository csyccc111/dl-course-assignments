# -*- coding: utf-8 -*-
"""
viz.py —— 可视化
================

全部用 OpenCV 绘制，不依赖 matplotlib，方便直接跑在实验机上。

注意：`cv2.putText` 不支持中文（会渲染成 ???），所以图上的文字一律用英文/数字，
中文只出现在报表（CSV / Markdown / 报告）里。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np

# BGR
COLOR_OK = (60, 200, 60)
COLOR_WARN = (40, 200, 240)
COLOR_BAD = (60, 120, 255)
COLOR_LM = (255, 200, 0)
COLOR_TEXT = (255, 255, 255)


def _score_color(score: float):
    if score >= 0.85:
        return COLOR_OK
    if score >= 0.65:
        return COLOR_WARN
    return COLOR_BAD


def draw_faces(
    img: np.ndarray,
    faces: Sequence,
    title: Optional[str] = None,
    show_index: bool = True,
    show_landmarks: bool = True,
) -> np.ndarray:
    """在图上画检测框 + 5 点 + 得分。返回新图（不修改入参）。"""
    canvas = img.copy()
    thick = max(1, int(round(min(canvas.shape[:2]) / 400.0)))

    for i, f in enumerate(faces):
        color = _score_color(float(f.score))
        p1 = (int(round(f.x1)), int(round(f.y1)))
        p2 = (int(round(f.x2)), int(round(f.y2)))
        cv2.rectangle(canvas, p1, p2, color, thick)

        if show_landmarks:
            for (x, y) in np.asarray(f.landmarks):
                cv2.circle(canvas, (int(round(x)), int(round(y))), max(2, thick + 1), COLOR_LM, -1)

        label = "{} {:.2f}".format(f.detector or "face", float(f.score))
        if show_index:
            label = "#{} ".format(i) + label

        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(p1[1] - 4, th + 4)
        cv2.rectangle(
            canvas, (p1[0], ty - th - 2), (p1[0] + tw + 4, ty + base - 1), color, -1
        )
        cv2.putText(
            canvas, label, (p1[0] + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1,
            cv2.LINE_AA,
        )

    if title:
        _draw_banner(canvas, title)
    return canvas


def _draw_banner(img: np.ndarray, text: str, color=(0, 0, 0), alpha: float = 0.55):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    w = img.shape[1]
    h = th + base + 10
    roi = img[0:h, 0:min(w, tw + 16)]
    overlay = roi.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
    cv2.putText(
        img, text, (6, th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )


def _cell(img: np.ndarray, size: int) -> np.ndarray:
    """等比缩放到 size x size 的方格里，多余部分用深灰填充。"""
    if img is None or img.size == 0:
        return np.full((size, size, 3), 40, np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    s = float(size) / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)

    out = np.full((size, size, 3), 40, np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def make_grid(
    images: Sequence[np.ndarray],
    cols: int = 8,
    cell: int = 96,
    pad: int = 4,
    bg: int = 32,
) -> np.ndarray:
    """把一组图拼成网格（对齐结果拼图，报告里很好用）。"""
    if not images:
        return np.full((cell, cell, 3), bg, np.uint8)

    cols = max(1, int(cols))
    rows = int(np.ceil(len(images) / float(cols)))
    W = cols * cell + (cols + 1) * pad
    H = rows * cell + (rows + 1) * pad
    canvas = np.full((H, W, 3), bg, np.uint8)

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        y = pad + r * (cell + pad)
        x = pad + c * (cell + pad)
        canvas[y:y + cell, x:x + cell] = _cell(img, cell)
    return canvas


def stack_h(images: Sequence[np.ndarray], pad: int = 6, bg: int = 32, height: Optional[int] = None):
    """水平拼接（原图 vs 对齐图 这种对照）。"""
    imgs = [im for im in images if im is not None and im.size > 0]
    if not imgs:
        return np.full((10, 10, 3), bg, np.uint8)

    if height is None:
        height = max(im.shape[0] for im in imgs)

    scaled = []
    for im in imgs:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        s = height / float(im.shape[0])
        scaled.append(cv2.resize(im, (max(1, int(round(im.shape[1] * s))), height)))

    W = sum(im.shape[1] for im in scaled) + pad * (len(scaled) + 1)
    canvas = np.full((height + pad * 2, W, 3), bg, np.uint8)
    x = pad
    for im in scaled:
        canvas[pad:pad + height, x:x + im.shape[1]] = im
        x += im.shape[1] + pad
    return canvas


def draw_align_demo(orig: np.ndarray, face, aligned: np.ndarray, crop_size: int = 112):
    """单张人脸的「原图标注 + 对齐结果」对照图，用于报告插图。"""
    vis = draw_faces(orig, [face], title="detect")
    x1 = int(max(0, face.x1)); y1 = int(max(0, face.y1))
    x2 = int(min(orig.shape[1], face.x2)); y2 = int(min(orig.shape[0], face.y2))
    crop = orig[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else np.zeros((crop_size, crop_size, 3), np.uint8)

    h = 240
    left = cv2.resize(vis, (int(vis.shape[1] * h / vis.shape[0]), h))
    gap = np.full((h, 8, 3), 32, np.uint8)
    patch = cv2.resize(crop, (int(crop.shape[1] * h / crop.shape[0]), h))
    face_big = cv2.resize(aligned, (h, h), interpolation=cv2.INTER_NEAREST)
    return np.hstack([left, gap, patch, gap, face_big])

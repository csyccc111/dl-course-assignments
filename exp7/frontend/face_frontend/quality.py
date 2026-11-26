# -*- coding: utf-8 -*-
"""
quality.py —— 对齐实现 + 关键点顺序自检 + 质量指标 + 失败归因
==============================================================

为什么需要「关键点顺序自检」
----------------------------
不同检测器给出的 5 个关键点顺序并不统一：

    ArcFace 112 模板 : [图像左眼, 图像右眼, 鼻尖, 图像左嘴角, 图像右嘴角]
    MTCNN 系         : [被摄者左眼, 被摄者右眼, 鼻尖, 被摄者左嘴角, 被摄者右嘴角]
                       由于「被摄者左眼」在图像右侧，等价于模板顺序整体左右互换

一旦顺序对不上，对齐出来的脸会左右翻转。更隐蔽的是：如果允许镜像相似变换
（reflective=True），「顺序正确」和「顺序颠倒」两种解的残差都是 0，
靠残差**根本发现不了**。所以这里做了两件事：
    1. 相似变换默认关闭镜像解（reflective=False）；
    2. 对两种候选顺序各求一次变换，取残差小的那个 —— 顺序错误时残差会明显偏大。

残差本身还是个很好的对齐质量指标：单位是 112x112 裁剪图里的像素，
数值越大说明 5 个关键点越不符合刚性人脸结构（通常是遮挡、大侧脸或检测抖动）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .align_trans import (
    build_reference,
    cvt_tform_mat_for_cv2,
    get_similarity_transform,
    similarity_residual,
)

# 模板索引 -> 检测器关键点索引
CANDIDATE_ORDERS: Dict[str, Tuple[int, int, int, int, int]] = {
    # 检测器顺序与模板一致（YuNet 常见情况）
    "direct": (0, 1, 2, 3, 4),
    # 左右互换（MTCNN 系、以及部分镜像标注的数据集）
    "swapped": (1, 0, 2, 4, 3),
}


@dataclass
class AlignResult:
    """一次「检测框 + 5 点 -> 112x112 对齐图」的结果。"""

    image: np.ndarray          # 对齐后的人脸（crop_size x crop_size）
    tfm: np.ndarray            # 2x3，cv2.warpAffine 用的矩阵
    rms: float                 # 关键点拟合残差（像素，裁剪图坐标系）
    order: str                 # 实际采用的关键点顺序
    reflective: bool           # 是否用了镜像解
    scale: float               # 原图 -> 裁剪图 的缩放系数
    roll_deg: float            # 原图坐标系下的人脸滚转角（度）
    crop_size: Tuple[int, int]

    @property
    def ok(self) -> bool:
        return self.image is not None and self.image.size > 0


def _order_landmarks(landmarks: np.ndarray, order: Sequence[int]) -> np.ndarray:
    return np.asarray(landmarks, dtype=np.float32)[list(order)]


def _roll_angle(pts: np.ndarray) -> float:
    """两眼连线相对水平方向的夹角（度）。正值表示人脸向图像右侧倾斜。"""
    dx = float(pts[1, 0] - pts[0, 0])
    dy = float(pts[1, 1] - pts[0, 1])
    return float(np.degrees(np.arctan2(dy, dx)))


def align_face(
    img: np.ndarray,
    landmarks: np.ndarray,
    crop_size: int = 112,
    order: str = "auto",
    reflective: bool = False,
) -> AlignResult:
    """按 5 点相似变换把脸对齐到 crop_size x crop_size。

    参数
    ----
    img        : HxWx3 BGR 原图
    landmarks  : (5,2) 检测器给出的关键点
    crop_size  : 输出边长，常用 112（ArcFace）或 128/160
    order      : 'auto' | 'direct' | 'swapped'
                 auto 时对两种顺序各求一次变换，取残差小者（推荐）
    reflective : 是否允许镜像解，人脸对齐建议 False

    返回
    ----
    AlignResult
    """
    crop = (int(crop_size), int(crop_size))
    ref = build_reference(crop)

    lm = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)

    if order == "auto":
        candidates = list(CANDIDATE_ORDERS.items())
    elif order in CANDIDATE_ORDERS:
        candidates = [(order, CANDIDATE_ORDERS[order])]
    else:
        raise ValueError("未知关键点顺序：{}".format(order))

    best = None  # (rms, name, trans)
    for name, idx in candidates:
        src = _order_landmarks(lm, idx)
        try:
            trans, _ = get_similarity_transform(src, ref, reflective=reflective)
        except Exception:
            continue
        rms = similarity_residual(src, ref, trans)
        if best is None or rms < best[0]:
            best = (rms, name, trans, src)

    if best is None:
        # 极端退化情况：直接给一个空结果，交给上层记为失败样本
        return AlignResult(
            image=np.zeros((crop[1], crop[0], 3), np.uint8),
            tfm=np.array([[1, 0, 0], [0, 1, 0]], np.float32),
            rms=float("inf"),
            order=order,
            reflective=reflective,
            scale=1.0,
            roll_deg=float("nan"),
            crop_size=crop,
        )

    rms, used, trans, src = best
    tfm = cvt_tform_mat_for_cv2(trans)

    warped = cv2.warpAffine(
        img,
        tfm,
        crop,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    lin = np.asarray(trans, dtype=np.float64)[:2, :2]
    scale = float(np.sqrt(abs(np.linalg.det(lin))))

    return AlignResult(
        image=warped,
        tfm=tfm,
        rms=float(rms),
        order=used,
        reflective=reflective,
        scale=scale,
        roll_deg=_roll_angle(src),
        crop_size=crop,
    )


# ---------------------------------------------------------------------------
# 质量指标
# ---------------------------------------------------------------------------
def sharpness(img: np.ndarray) -> float:
    """清晰度（Laplacian 方差）。越小越糊，< 60 基本可以判定为模糊。"""
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_metrics(face, aligned: AlignResult, img: Optional[np.ndarray] = None) -> Dict[str, float]:
    """汇总单张人脸的量化指标，直接写进 CSV。"""
    lm = np.asarray(face.landmarks, dtype=np.float32)
    eye_dist = float(np.linalg.norm(lm[0] - lm[1]))
    width = max(face.width, 1e-6)

    crop_patch = None
    if img is not None:
        x1 = int(max(0, np.floor(face.x1)))
        y1 = int(max(0, np.floor(face.y1)))
        x2 = int(min(img.shape[1], np.ceil(face.x2)))
        y2 = int(min(img.shape[0], np.ceil(face.y2)))
        if x2 > x1 and y2 > y1:
            crop_patch = img[y1:y2, x1:x2]

    return {
        "score": float(face.score),
        "bbox_w": float(face.width),
        "bbox_h": float(face.height),
        "min_side": float(face.min_side),
        "aspect": float(face.aspect),
        # 眼距 / 框宽：正脸约 0.30~0.42，明显偏小通常是偏航角大（侧脸）
        "frontal": float(eye_dist / width),
        "align_rms": float(aligned.rms),
        "align_scale": float(aligned.scale),
        "roll_deg": float(aligned.roll_deg),
        "sharp": sharpness(crop_patch) if crop_patch is not None else float("nan"),
    }


# 失败类型阈值（可按数据集微调）
DEFAULT_THRESHOLDS = {
    "min_side": 24.0,      # 小于这个像素数算「小脸」
    "min_score": 0.5,      # 置信度下限
    "min_frontal": 0.22,   # 眼距/框宽 下限，低于此判为侧脸
    "max_rms": 8.0,        # 对齐残差上限（112 坐标系像素）
    "min_sharp": 40.0,     # 清晰度下限
}


def classify_failure(metrics: Dict[str, float], thresholds: Optional[Dict[str, float]] = None) -> List[str]:
    """给人脸打失败标签，返回标签列表（空列表表示质量合格）。

    多个标签可以并存，便于统计「小脸且模糊」这类复合失败样本。
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    reasons = []
    if metrics.get("min_side", 1e9) < th["min_side"]:
        reasons.append("small_face")
    if metrics.get("score", 1.0) < th["min_score"]:
        reasons.append("low_score")
    if metrics.get("frontal", 1.0) < th["min_frontal"]:
        reasons.append("profile")
    if metrics.get("align_rms", 0.0) > th["max_rms"]:
        reasons.append("bad_align")
    if metrics.get("sharp", 1e9) < th["min_sharp"]:
        reasons.append("blur")
    return reasons

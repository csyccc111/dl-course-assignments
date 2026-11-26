# -*- coding: utf-8 -*-
"""
face_frontend
=============

人脸「视觉前端」：检测 + 5 点对齐 + 112x112 裁剪。
对应方案 A（成员 A：检测与对齐前端）。

模块划分
--------
align_trans   : 改编自 face.evoLVe 的对齐/裁剪实现（相似变换 + warp）
detectors     : YuNet / CenterFace 两个检测器的统一封装
quality       : 关键点顺序自检、对齐质量指标、失败归因
viz           : 可视化（检测框、关键点、对齐拼图、对比图）
io_utils      : 中文路径安全的图片读写
pipeline      : 批量处理、检测器对比、阈值扫描、报表输出
"""

from .align_trans import (
    REFERENCE_FACIAL_POINTS,
    build_reference,
    get_reference_facial_points,
    get_similarity_transform,
    get_similarity_transform_for_cv2,
    similarity_residual,
    warp_and_crop_face,
)
from .detectors import CenterFaceDetector, Face, YuNetDetector, build_detector
from .quality import AlignResult, align_face, classify_failure

__all__ = [
    "REFERENCE_FACIAL_POINTS",
    "build_reference",
    "get_reference_facial_points",
    "get_similarity_transform",
    "get_similarity_transform_for_cv2",
    "similarity_residual",
    "warp_and_crop_face",
    "Face",
    "YuNetDetector",
    "CenterFaceDetector",
    "build_detector",
    "AlignResult",
    "align_face",
    "classify_failure",
]

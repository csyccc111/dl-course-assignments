# -*- coding: utf-8 -*-
"""
detectors.py —— YuNet / CenterFace 统一封装
============================================

两者都输出「人脸框 + 5 个关键点」，因此对外统一成 `Face` 对象，
上层（对齐、可视化、统计）完全不用关心具体是哪个检测器。

YuNet
-----
OpenCV 自带的轻量检测器（libfacedetection 作者开发，opencv_zoo 提供 onnx）。
    cv2.FaceDetectorYN.create(model, "", input_size, score_thresh, nms_thresh, top_k)
    det.setInputSize((w, h)); _, faces = det.detect(img)
输出 faces 形状 (N, 15)：
    [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
    其中框是「左上角 + 宽高」，关键点是绝对坐标。

CenterFace
----------
anchor-free、face-as-point 的检测器（7.3MB），OpenCV DNN 可直接读 onnx。
    net.forward(["537", "538", "539", "540"]) -> heatmap, scale, offset, landmarks
    heatmap (1,1,H/4,W/4)，scale (1,2,H/4,W/4)，offset (1,2,H/4,W/4)，landmark (1,10,H/4,W/4)
解码时注意两点（原仓库解码代码里就藏着这两个坑）：
    * 宽高是 exp(scale)*4，中心是 (j+offset+0.5)*4；
    * landmark 的通道是**交叉存放**的：通道 2j 存 y，通道 2j+1 存 x，
      写反了关键点会整体错位成镜像状。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np


DEFAULT_YUNET_MODEL = os.path.join("models", "face_detection_yunet_2023mar.onnx")
DEFAULT_CENTERFACE_MODEL = os.path.join("models", "centerface.onnx")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Face:
    """一张人脸的检测结果。坐标全部在**原图**坐标系（未归一化）。"""

    bbox: np.ndarray          # (4,) float32, x1, y1, x2, y2
    score: float
    landmarks: np.ndarray     # (5, 2) float32, (x, y)
    detector: str = ""

    # -- 便捷属性 ----------------------------------------------------------
    @property
    def x1(self):
        return float(self.bbox[0])

    @property
    def y1(self):
        return float(self.bbox[1])

    @property
    def x2(self):
        return float(self.bbox[2])

    @property
    def y2(self):
        return float(self.bbox[3])

    @property
    def width(self):
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self):
        return float(self.bbox[3] - self.bbox[1])

    @property
    def min_side(self):
        return min(self.width, self.height)

    @property
    def area(self):
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def aspect(self):
        """高宽比，正常正脸大约 1.2 ~ 1.5。"""
        return self.height / max(self.width, 1e-6)

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def _clip_bbox(box, w, h):
    box = np.asarray(box, dtype=np.float32).copy()
    box[0] = max(0.0, min(box[0], w - 1.0))
    box[1] = max(0.0, min(box[1], h - 1.0))
    box[2] = max(0.0, min(box[2], w - 1.0))
    box[3] = max(0.0, min(box[3], h - 1.0))
    return box


def _clip_landmarks(lms, w, h):
    lms = np.asarray(lms, dtype=np.float32).copy()
    lms[:, 0] = np.clip(lms[:, 0], 0.0, w - 1.0)
    lms[:, 1] = np.clip(lms[:, 1], 0.0, h - 1.0)
    return lms


# ---------------------------------------------------------------------------
# NMS（CenterFace 需要自己实现；YuNet 内部已做）
# ---------------------------------------------------------------------------
def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, thresh: float = 0.3) -> List[int]:
    """标准 IoU-NMS，boxes 为 (N,4) 的 x1,y1,x2,y2。返回保留下标。"""
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1 + 1.0) * np.maximum(0.0, y2 - y1 + 1.0)
    order = np.argsort(scores)[::-1]

    # 注意：原仓库用 np.bool，numpy>=1.24 已删除该别名，这里用内置 bool
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep = []

    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))

        ix1 = np.maximum(x1[i], x1[order])
        iy1 = np.maximum(y1[i], y1[order])
        ix2 = np.minimum(x2[i], x2[order])
        iy2 = np.minimum(y2[i], y2[order])
        inter = np.maximum(0.0, ix2 - ix1 + 1.0) * np.maximum(0.0, iy2 - iy1 + 1.0)
        ovr = inter / (areas[i] + areas[order] - inter + 1e-12)

        suppressed[order[np.where(ovr >= thresh)[0]]] = True
        suppressed[i] = False  # 自己始终保留
    return keep


# ---------------------------------------------------------------------------
# YuNet
# ---------------------------------------------------------------------------
class YuNetDetector:
    name = "yunet"

    def __init__(
        self,
        model_path: str = DEFAULT_YUNET_MODEL,
        conf_thresh: float = 0.6,
        nms_thresh: float = 0.3,
        top_k: int = 5000,
        max_input_side: Optional[int] = None,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                "找不到 YuNet 模型：{}\n请先运行 `python download_models.py`".format(model_path)
            )

        # OpenCV 4.5.4 ~ 4.7 是 cv2.FaceDetectorYN.create，
        # 4.8+ 同时提供 cv2.FaceDetectorYN_create，两个都兼容一下
        factory = getattr(cv2, "FaceDetectorYN_create", None)
        if factory is None:
            factory = cv2.FaceDetectorYN.create
        self.det = factory(
            str(model_path), "", (320, 320), float(conf_thresh), float(nms_thresh), int(top_k)
        )

        self.model_path = model_path
        self.conf_thresh = float(conf_thresh)
        self.nms_thresh = float(nms_thresh)
        self.top_k = int(top_k)
        self.max_input_side = max_input_side

    def detect(self, img: np.ndarray) -> List[Face]:
        h, w = img.shape[:2]
        scale = 1.0
        if self.max_input_side and max(h, w) > self.max_input_side:
            scale = float(self.max_input_side) / float(max(h, w))
            inp = cv2.resize(
                img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
            )
        else:
            inp = img

        ih, iw = inp.shape[:2]
        self.det.setInputSize((int(iw), int(ih)))
        _, faces = self.det.detect(inp)

        if faces is None or len(faces) == 0:
            return []

        out = []
        for f in np.asarray(faces, dtype=np.float32):
            box = np.array([f[0], f[1], f[0] + f[2], f[1] + f[3]], dtype=np.float32)
            lms = f[4:14].reshape(5, 2).astype(np.float32)
            if scale != 1.0:
                box = box / scale
                lms = lms / scale
            out.append(
                Face(
                    bbox=_clip_bbox(box, w, h),
                    score=float(f[14]),
                    landmarks=_clip_landmarks(lms, w, h),
                    detector=self.name,
                )
            )
        return out

    def describe(self) -> str:
        return "YuNet(conf={:.2f}, nms={:.2f})".format(self.conf_thresh, self.nms_thresh)


# ---------------------------------------------------------------------------
# CenterFace
# ---------------------------------------------------------------------------
class CenterFaceDetector:
    name = "centerface"

    def __init__(
        self,
        model_path: str = DEFAULT_CENTERFACE_MODEL,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.3,
        max_input_side: Optional[int] = None,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                "找不到 CenterFace 模型：{}\n请先运行 `python download_models.py`".format(model_path)
            )
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.model_path = model_path
        self.conf_thresh = float(conf_thresh)
        self.nms_thresh = float(nms_thresh)
        self.max_input_side = max_input_side

    # -- 内部 ------------------------------------------------------------
    def _forward(self, blob):
        """优先按名字取输出；某些导出模型输出名不是 537~540，则退化到按序取。"""
        self.net.setInput(blob)
        try:
            return self.net.forward(["537", "538", "539", "540"])
        except cv2.error:
            outs = self.net.forward()
            if len(outs) < 4:
                raise RuntimeError(
                    "CenterFace onnx 输出节点数量异常（{}），该模型可能不是带关键点的版本".format(
                        len(outs)
                    )
                )
            return outs[0], outs[1], outs[2], outs[3]

    def detect(self, img: np.ndarray) -> List[Face]:
        h, w = img.shape[:2]
        scale = 1.0
        if self.max_input_side and max(h, w) > self.max_input_side:
            scale = float(self.max_input_side) / float(max(h, w))
            src = cv2.resize(
                img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
            )
        else:
            src = img

        sh, sw = src.shape[:2]
        nh = int(np.ceil(sh / 32.0) * 32)
        nw = int(np.ceil(sw / 32.0) * 32)
        scale_h = nh / float(sh)
        scale_w = nw / float(sw)

        blob = cv2.dnn.blobFromImage(
            src, scalefactor=1.0, size=(nw, nh), mean=(0, 0, 0), swapRB=True, crop=False
        )
        heatmap, scale_map, offset, landmark = self._forward(blob)

        boxes, lms = self._decode(
            heatmap, scale_map, offset, landmark, (nh, nw), self.conf_thresh, self.nms_thresh
        )
        if len(boxes) == 0:
            return []

        # 网络输出坐标 -> 送网图坐标
        boxes[:, 0:4:2] /= scale_w
        boxes[:, 1:4:2] /= scale_h
        if lms is not None and len(lms) > 0:
            lms[:, 0:10:2] /= scale_w
            lms[:, 1:10:2] /= scale_h
        # 送网图坐标 -> 原图坐标
        if scale != 1.0:
            boxes[:, 0:4:2] /= scale
            boxes[:, 1:4:2] /= scale
            if lms is not None and len(lms) > 0:
                lms[:, 0:10:2] /= scale
                lms[:, 1:10:2] /= scale

        out = []
        for i in range(len(boxes)):
            box = _clip_bbox(boxes[i, :4], w, h)
            lm = lms[i].reshape(5, 2) if lms is not None and len(lms) > i else None
            if lm is None:
                # 极端情况：没有关键点就跳过（对齐需要 5 点）
                continue
            out.append(
                Face(
                    bbox=box,
                    score=float(boxes[i, 4]),
                    landmarks=_clip_landmarks(lm, w, h),
                    detector=self.name,
                )
            )
        return out

    @staticmethod
    def _decode(heatmap, scale, offset, landmark, size, threshold, nms_thresh):
        """把网络输出解码成 boxes(N,5)[x1,y1,x2,y2,score] 与 lms(N,10)。"""
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]

        c0, c1 = np.where(heatmap > threshold)
        boxes, lms = [], []

        for i in range(len(c0)):
            s0 = np.exp(scale0[c0[i], c1[i]]) * 4.0
            s1 = np.exp(scale1[c0[i], c1[i]]) * 4.0
            o0 = offset0[c0[i], c1[i]]
            o1 = offset1[c0[i], c1[i]]
            s = heatmap[c0[i], c1[i]]

            x1 = max(0.0, (c1[i] + o1 + 0.5) * 4.0 - s1 / 2.0)
            y1 = max(0.0, (c0[i] + o0 + 0.5) * 4.0 - s0 / 2.0)
            x1 = min(x1, size[1])
            y1 = min(y1, size[0])

            boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])

            lm = []
            for j in range(5):
                # 关键点通道交叉存放：2j+1 是 x，2j 是 y（写反就会整体错位）
                lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
            lms.append(lm)

        if not boxes:
            return np.empty((0, 5), np.float32), np.empty((0, 10), np.float32)

        boxes = np.asarray(boxes, dtype=np.float32)
        lms = np.asarray(lms, dtype=np.float32)

        keep = nms_xyxy(boxes[:, :4], boxes[:, 4], nms_thresh)
        if len(keep) == 0:
            return np.empty((0, 5), np.float32), np.empty((0, 10), np.float32)

        return boxes[keep, :], lms[keep, :]

    def describe(self) -> str:
        return "CenterFace(conf={:.2f}, nms={:.2f})".format(self.conf_thresh, self.nms_thresh)


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
def build_detector(name: str, **kwargs):
    """按名字构造检测器。kwargs 里不认识的关键字会被忽略，方便 CLI 直接透传。"""
    name = name.lower()
    if name == "yunet":
        keys = ("model_path", "conf_thresh", "nms_thresh", "top_k", "max_input_side")
        return YuNetDetector(**{k: v for k, v in kwargs.items() if k in keys})
    if name == "centerface":
        keys = ("model_path", "conf_thresh", "nms_thresh", "max_input_side")
        return CenterFaceDetector(**{k: v for k, v in kwargs.items() if k in keys})
    raise ValueError("未知检测器：{}（可选 yunet / centerface）".format(name))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) -> (N,M) IoU，用于跨检测器匹配。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    aa = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    bb = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)

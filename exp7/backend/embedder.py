"""YuNet 检测 + SFace 对齐/特征提取。

模型（OpenCV Zoo）：
- face_detection_yunet_2023mar.onnx
- face_recognition_sface_2021dec.onnx

说明：
- SFace 自带的 alignCrop 只是临时对齐，方便 B 先跑通识别闭环。
  最终系统请用 A 的 evoLVe 112x112 对齐结果，直接调用 embed_aligned()。
- 所有特征都做 L2 归一化，余弦相似度等价于点积。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from .paths import resolve
    from .paths import model_path as safe_model_path
except ImportError:  # 允许 python backend/embedder.py 直接运行
    from paths import resolve
    from paths import model_path as safe_model_path


@dataclass
class Face:
    """一次检测结果。"""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    score: float
    landmarks: np.ndarray            # (5, 2)
    row: np.ndarray                  # YuNet 原始 15 维输出
    aligned: Optional[np.ndarray] = None


def imread_unicode(path: str | Path) -> Optional[np.ndarray]:
    """读取图片，兼容 Windows 中文路径。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    """保存图片，兼容 Windows 中文路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        return False
    buffer.tofile(str(path))
    return True


class FaceEmbedder:
    """YuNet + SFace 的最小封装。"""

    def __init__(
        self,
        yunet_path: str | Path,
        sface_path: str | Path,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        input_size: tuple[int, int] = (320, 320),
    ) -> None:
        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError(
                "当前 OpenCV 不支持 FaceDetectorYN/FaceRecognizerSF，"
                "请安装 opencv-python>=4.5.4"
            )
        self.yunet_path = safe_model_path(yunet_path)
        self.sface_path = safe_model_path(sface_path)
        self.detector = cv2.FaceDetectorYN.create(
            self.yunet_path,
            "",
            input_size,
            score_threshold,
            nms_threshold,
            top_k,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(self.sface_path, "")

    def detect(self, image: np.ndarray) -> list[Face]:
        """返回图中所有人脸，按检测顺序。"""
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        if faces is None:
            return []

        result: list[Face] = []
        for row in np.asarray(faces, dtype=np.float32):
            x, y, bw, bh = row[:4]
            result.append(
                Face(
                    bbox=(
                        int(round(float(x))),
                        int(round(float(y))),
                        int(round(float(bw))),
                        int(round(float(bh))),
                    ),
                    score=float(row[14]),
                    landmarks=row[4:14].reshape(5, 2).astype(np.float32),
                    row=row.astype(np.float32).copy(),
                )
            )
        return result

    def align(self, image: np.ndarray, face: Face) -> np.ndarray:
        """用 SFace 的 alignCrop 对齐到 112x112。"""
        # alignCrop 需要 YuNet 的 1x15 原始输出
        return self.recognizer.alignCrop(image, face.row.reshape(1, -1))

    def embed_aligned(self, aligned: np.ndarray) -> np.ndarray:
        """输入 112x112 对齐图（BGR），返回 L2 归一化的特征。"""
        if aligned is None or aligned.size == 0:
            raise ValueError("对齐图为空")
        feat = self.recognizer.feature(aligned).ravel().astype(np.float32)
        norm = float(np.linalg.norm(feat))
        if norm <= 0:
            raise ValueError("特征范数为 0")
        return feat / norm

    def embed(self, image: np.ndarray, largest: bool = True) -> Optional[np.ndarray]:
        """检测 + 对齐 + 特征；没有人脸返回 None。"""
        faces = self.detect(image)
        if not faces:
            return None
        if largest:
            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        else:
            face = faces[0]
        return self.embed_aligned(self.align(image, face))

    def embed_all(self, image: np.ndarray) -> list[tuple[Face, np.ndarray]]:
        """返回图中所有人脸的特征，按人脸面积从大到小排序。"""
        faces = sorted(
            self.detect(image),
            key=lambda f: f.bbox[2] * f.bbox[3],
            reverse=True,
        )
        result: list[tuple[Face, np.ndarray]] = []
        for face in faces:
            aligned = self.align(image, face)
            face.aligned = aligned
            result.append((face, self.embed_aligned(aligned)))
        return result


def draw_faces(
    image: np.ndarray,
    faces: list[Face],
    labels: Optional[list[str]] = None,
) -> np.ndarray:
    """画框、5 点和标签，返回新图。"""
    vis = image.copy()
    for i, face in enumerate(faces):
        x, y, w, h = face.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for px, py in face.landmarks.astype(int):
            cv2.circle(vis, (int(px), int(py)), 2, (0, 0, 255), -1)
        text = f"{face.score:.2f}"
        if labels is not None and i < len(labels):
            text = f"{labels[i]} {text}"
        cv2.putText(
            vis,
            text,
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="单图检测 + 特征提取")
    parser.add_argument("--image", required=True)
    parser.add_argument("--yunet", default="models/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--sface", default="models/face_recognition_sface_2021dec.onnx")
    parser.add_argument("--score", type=float, default=0.6)
    parser.add_argument("--save", default="outputs/embedder_demo.jpg")
    args = parser.parse_args()

    image = imread_unicode(resolve(args.image))
    if image is None:
        raise SystemExit(f"读不到图片: {args.image}")

    embedder = FaceEmbedder(args.yunet, args.sface, score_threshold=args.score)
    pairs = embedder.embed_all(image)
    print(f"检测到 {len(pairs)} 张脸")
    for i, (face, feat) in enumerate(pairs):
        print(
            f"[{i}] bbox={face.bbox} score={face.score:.3f} "
            f"embedding={feat.shape} norm={np.linalg.norm(feat):.4f}"
        )

    out_path = resolve(args.save)
    if imwrite_unicode(out_path, draw_faces(image, [f for f, _ in pairs])):
        print(f"可视化已保存: {out_path}")
    else:
        print("可视化保存失败")


if __name__ == "__main__":
    main()

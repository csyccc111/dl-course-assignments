"""A 前端（YuNet 检测 + evoLVe 112×112 对齐）到 B 后端（SFace 特征）的桥。

只交换 112×112 BGR 对齐图，不共享两边的 Face 对象：A 的 bbox 是 (x1,y1,x2,y2)，
B 的是 (x,y,w,h)，且 SFace 的 alignCrop 需要 YuNet 的 1x15 原始行，A 不提供。

用法：
  python -m backend.bridge --image frontend/data/test/t02_obama.jpg
  python -m backend.bridge --dir  frontend/data/test_hard --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from .embedder import FaceEmbedder, imread_unicode, imwrite_unicode
    from .paths import FRONTEND_ROOT, new_run_dir, resolve
    from .paths import model_path as safe_model_path
except ImportError:  # 允许 python backend/bridge.py 直接运行
    from embedder import FaceEmbedder, imread_unicode, imwrite_unicode
    from paths import FRONTEND_ROOT, new_run_dir, resolve
    from paths import model_path as safe_model_path

if str(FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(FRONTEND_ROOT))

from face_frontend.detectors import YuNetDetector  # noqa: E402
from face_frontend.quality import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    align_face,
    classify_failure,
    face_metrics,
)

DEFAULT_YUNET = "frontend/models/face_detection_yunet_2023mar.onnx"
DEFAULT_SFACE = "models/face_recognition_sface_2021dec.onnx"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class AlignedFace:
    """一张脸的对齐结果 + 质检标签（来自 A 的指标口径）。"""

    index: int
    score: float
    rms: float
    order: str
    bbox: tuple  # x1, y1, x2, y2（原图坐标）
    flags: list
    aligned: np.ndarray

    @property
    def ok(self) -> bool:
        return not self.flags

    def to_row(self) -> dict:
        return {
            "index": self.index,
            "score": round(self.score, 4),
            "align_rms": round(self.rms, 3),
            "landmark_order": self.order,
            "bbox_xyxy": [round(v, 1) for v in self.bbox],
            "flags": self.flags,
            "ok": self.ok,
        }


class FrontendBridge:
    """检测 → 5 点对齐 → 质检，可选交给 SFace 提特征。"""

    def __init__(
        self,
        yunet: str = DEFAULT_YUNET,
        conf: float = 0.6,
        nms: float = 0.3,
        crop: int = 112,
        max_rms: float = 8.0,
        min_side: float = DEFAULT_THRESHOLDS["min_side"],
        min_frontal: float = DEFAULT_THRESHOLDS["min_frontal"],
        max_input_side: Optional[int] = None,
    ) -> None:
        self.detector = YuNetDetector(
            model_path=safe_model_path(yunet),
            conf_thresh=conf,
            nms_thresh=nms,
            max_input_side=max_input_side,
        )
        self.crop = int(crop)
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        self.thresholds.update(
            {"max_rms": float(max_rms), "min_side": float(min_side),
             "min_frontal": float(min_frontal)}
        )

    def detect(self, image: np.ndarray) -> list:
        """返回图中所有人脸的对齐结果，按置信度从高到低。"""
        faces = self.detector.detect(image)
        result: list[AlignedFace] = []
        for i, face in enumerate(sorted(faces, key=lambda f: -f.score)):
            aligned = align_face(image, face.landmarks, self.crop, "auto", False)
            metrics = face_metrics(face, aligned, image)
            result.append(
                AlignedFace(
                    index=i,
                    score=float(metrics["score"]),
                    rms=float(aligned.rms),
                    order=aligned.order,
                    bbox=(float(face.x1), float(face.y1),
                          float(face.x2), float(face.y2)),
                    flags=classify_failure(metrics, self.thresholds),
                    aligned=aligned.image,
                )
            )
        return result

    def embed(self, image: np.ndarray, embedder: FaceEmbedder,
              only_ok: bool = True) -> list:
        """返回 [(AlignedFace, 128 维特征)]；质检不合格的脸默认丢弃。"""
        pairs = []
        for face in self.detect(image):
            if only_ok and not face.ok:
                continue
            pairs.append((face, embedder.embed_aligned(face.aligned)))
        return pairs


def run(images: list, bridge: FrontendBridge, embedder: Optional[FaceEmbedder],
        out_dir: Path, save_crops: bool = True) -> dict:
    rows = []
    for path in images:
        image = imread_unicode(path)
        if image is None:
            rows.append({"image": path.name, "error": "读图失败"})
            continue
        faces = bridge.detect(image)
        feats = {}
        if embedder is not None:
            for face, feat in bridge.embed(image, embedder):
                feats[face.index] = feat
        for face in faces:
            row = {"image": path.name, **face.to_row()}
            if face.index in feats:
                row["embedding_norm"] = round(float(np.linalg.norm(feats[face.index])), 4)
                if save_crops:
                    crop_dir = out_dir / "aligned"
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    imwrite_unicode(
                        crop_dir / f"{path.stem}_f{face.index}.jpg", face.aligned
                    )
            else:
                row["embedded"] = False
            rows.append(row)

        tag = f"{len(faces)} 脸" if faces else "无脸"
        print(f"{path.name:<28} {tag:<8} " + " ".join(
            f"[{f.index}] {f.score:.3f}/rms{f.rms:.2f}{'' if f.ok else '/' + '|'.join(f.flags)}"
            for f in faces))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="A 前端 -> B 特征 联调")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="单张图片")
    src.add_argument("--dir", help="图片目录（递归）")
    parser.add_argument("--out", default=None, help="默认 outputs/<时间戳>_bridge/")
    parser.add_argument("--yunet", default=DEFAULT_YUNET)
    parser.add_argument("--sface", default=DEFAULT_SFACE)
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--crop", type=int, default=112)
    parser.add_argument("--max-rms", type=float, default=8.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-embed", action="store_true", help="只跑检测对齐")
    args = parser.parse_args()

    if args.dir:
        root = resolve(args.dir)
        if not root.is_dir():
            raise SystemExit(f"目录不存在: {root}")
        images = sorted(p for p in root.rglob("*")
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    else:
        images = [resolve(args.image)]
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit("没有图片")

    out_dir = resolve(args.out) if args.out else new_run_dir("bridge")
    out_dir.mkdir(parents=True, exist_ok=True)

    bridge = FrontendBridge(yunet=args.yunet, conf=args.conf, crop=args.crop,
                            max_rms=args.max_rms)
    embedder = None
    if not args.no_embed:
        embedder = FaceEmbedder(safe_model_path(args.yunet),
                                safe_model_path(args.sface))

    print(f"输出目录: {out_dir}\n")
    rows = run(images, bridge, embedder, out_dir)

    report = {
        "config": {
            "frontend": str(FRONTEND_ROOT),
            "yunet": args.yunet,
            "sface": args.sface,
            "conf": args.conf,
            "crop": args.crop,
            "max_rms": args.max_rms,
            "embed": not args.no_embed,
        },
        "n_images": len(images),
        "n_faces": len(rows),
        "n_rejected": sum(1 for r in rows if r.get("ok") is False),
        "rows": rows,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n图片 {len(images)}  人脸 {report['n_faces']}  "
          f"质检不合格 {report['n_rejected']}")
    print(f"明细: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()

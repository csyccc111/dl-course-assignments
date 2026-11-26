"""批量提取 embedding 并缓存为 npz。

数据目录约定（按身份分文件夹）：
  data/selfmade/
    alice/
      enroll/001.jpg
      test/002.jpg
    bob/
      enroll/001.jpg

输出 npz：
  embeddings (N, D) float32，已 L2 归一化
  labels     (N,)   身份名
  paths      (N,)   相对路径
  splits     (N,)   enroll/test/train/val/all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .embedder import FaceEmbedder, imread_unicode
    from .paths import resolve
except ImportError:  # 允许 python backend/build_embeddings.py 直接运行
    from embedder import FaceEmbedder, imread_unicode
    from paths import resolve

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KNOWN_SPLITS = {"enroll", "test", "train", "val"}


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="批量提取人脸 embedding")
    parser.add_argument("--root", required=True, help="数据集根目录，按身份分文件夹")
    parser.add_argument("--output", required=True, help="输出 npz 路径")
    parser.add_argument("--yunet", default="models/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--sface", default="models/face_recognition_sface_2021dec.onnx")
    parser.add_argument("--score", type=float, default=0.6)
    parser.add_argument("--all-faces", action="store_true", help="每张图提取所有人脸")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张，调试用")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = resolve(args.root)
    if not root.is_dir():
        raise SystemExit(f"数据目录不存在: {root}")

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    embedder = FaceEmbedder(args.yunet, args.sface, score_threshold=args.score)
    images = list(iter_images(root))
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"{root} 下没有图片")

    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    splits: list[str] = []
    failures: list[dict] = []

    for i, path in enumerate(images, start=1):
        rel = path.relative_to(root)
        if len(rel.parts) < 2:
            failures.append({"path": str(rel), "reason": "不在身份文件夹内"})
            continue

        label = rel.parts[0]
        split = (
            rel.parts[1]
            if len(rel.parts) >= 3 and rel.parts[1] in KNOWN_SPLITS
            else "all"
        )

        image = imread_unicode(path)
        if image is None:
            failures.append({"path": str(rel), "reason": "读图失败"})
            continue

        faces = embedder.detect(image)
        if not faces:
            failures.append({"path": str(rel), "reason": "未检测到人脸"})
            continue

        selected = (
            faces
            if args.all_faces
            else [max(faces, key=lambda f: f.bbox[2] * f.bbox[3])]
        )
        for j, face in enumerate(selected):
            try:
                aligned = embedder.align(image, face)
                feat = embedder.embed_aligned(aligned)
            except Exception as exc:
                failures.append({"path": str(rel), "reason": f"特征提取失败: {exc}"})
                continue
            embeddings.append(feat)
            labels.append(label)
            splits.append(split)
            paths.append(f"{rel}#{j}" if args.all_faces else str(rel))

        if not args.quiet and (i % 50 == 0 or i == len(images)):
            print(
                f"[{i}/{len(images)}] embeddings={len(embeddings)} "
                f"failures={len(failures)}"
            )

    if not embeddings:
        raise SystemExit("没有提取到任何 embedding，检查数据和模型")

    np.savez_compressed(
        output,
        embeddings=np.stack(embeddings).astype(np.float32),
        labels=np.asarray(labels, dtype=str),
        paths=np.asarray(paths, dtype=str),
        splits=np.asarray(splits, dtype=str),
    )

    meta = {
        "root": str(root),
        "n_images": len(images),
        "n_embeddings": len(embeddings),
        "n_identities": len(set(labels)),
        "split_counts": {s: splits.count(s) for s in sorted(set(splits))},
        "failures": failures,
        "yunet": str(args.yunet),
        "sface": str(args.sface),
        "score_threshold": args.score,
    }
    meta_path = output.with_suffix(".json")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"已保存: {output}")
    print(
        f"人数: {meta['n_identities']}  embedding: {meta['n_embeddings']}  "
        f"失败: {len(failures)}"
    )
    print(f"元数据: {meta_path}")


if __name__ == "__main__":
    main()

"""人脸库：注册、保存、余弦匹配。

用法：
  python -m backend.gallery \
      --embeddings outputs/self.npz \
      --output outputs/gallery.npz \
      --split enroll \
      --threshold 0.363
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from .paths import resolve
except ImportError:  # 允许 python backend/gallery.py 直接运行
    from paths import resolve


@dataclass
class Match:
    name: str
    score: float
    known: bool


class FaceGallery:
    """每人一条平均 embedding，查询时取余弦相似度最大者。"""

    def __init__(self, threshold: float = 0.363) -> None:
        self.threshold = float(threshold)
        self._embeddings: dict[str, list[np.ndarray]] = {}
        self._names: list[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._dirty = True

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(emb))
        if norm <= 0:
            raise ValueError(f"{name} 的 embedding 范数为 0")
        self._embeddings.setdefault(str(name), []).append(emb / norm)
        self._dirty = True

    def _build(self) -> None:
        names: list[str] = []
        matrix: list[np.ndarray] = []
        for name in sorted(self._embeddings):
            embs = np.stack(self._embeddings[name])
            mean = embs.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if norm <= 0:
                continue
            names.append(name)
            matrix.append(mean / norm)
        self._names = names
        self._matrix = (
            np.stack(matrix) if matrix else np.empty((0, 0), dtype=np.float32)
        )
        self._dirty = False

    @property
    def names(self) -> list[str]:
        if self._dirty:
            self._build()
        return list(self._names)

    def __len__(self) -> int:
        return len(self.names)

    def match(self, embedding: np.ndarray) -> Match:
        if self._dirty:
            self._build()
        assert self._matrix is not None
        if self._matrix.size == 0:
            return Match("unknown", -1.0, False)

        query = np.asarray(embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query))
        if norm <= 0:
            return Match("unknown", -1.0, False)
        query = query / norm

        scores = self._matrix @ query
        idx = int(np.argmax(scores))
        score = float(scores[idx])
        return Match(self._names[idx], score, score >= self.threshold)

    def save(self, path: str | Path) -> None:
        if self._dirty:
            self._build()
        assert self._matrix is not None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            names=np.asarray(self._names, dtype=str),
            embeddings=self._matrix.astype(np.float32),
            threshold=np.asarray([self.threshold], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FaceGallery":
        data = np.load(path, allow_pickle=False)
        gallery = cls(threshold=float(data["threshold"][0]))
        for name, emb in zip(data["names"], data["embeddings"]):
            gallery._embeddings[str(name)] = [emb.astype(np.float32)]
        gallery._build()
        return gallery


def build_from_npz(
    embeddings_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.363,
    split: str = "enroll",
) -> tuple[FaceGallery, int]:
    data = np.load(embeddings_path, allow_pickle=False)
    embs = data["embeddings"]
    labels = data["labels"]
    splits = data["splits"] if "splits" in data else None

    if split != "all" and splits is not None:
        available = sorted(set(splits.tolist()))
        if split not in available:
            raise SystemExit(f"没有 split={split} 的样本；可选: {available}")

    gallery = FaceGallery(threshold=threshold)
    used = 0
    for i, (emb, label) in enumerate(zip(embs, labels)):
        if split != "all" and splits is not None and str(splits[i]) != split:
            continue
        gallery.enroll(str(label), emb)
        used += 1

    gallery.save(output_path)
    return gallery, used


def main() -> None:
    parser = argparse.ArgumentParser(description="从 embedding npz 构建人脸库")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", default="outputs/gallery.npz")
    parser.add_argument("--threshold", type=float, default=0.363)
    parser.add_argument(
        "--split",
        default="enroll",
        help="只注册该 split（enroll/test/train/val）；all 表示全部",
    )
    args = parser.parse_args()

    embeddings = resolve(args.embeddings)
    output = resolve(args.output)
    gallery, used = build_from_npz(
        embeddings, output, args.threshold, args.split
    )
    print(f"注册 {len(gallery)} 人，使用 {used} 条 embedding")
    print(f"阈值: {gallery.threshold}")
    print(f"已保存: {output}")


if __name__ == "__main__":
    main()

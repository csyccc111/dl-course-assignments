"""LFW 1:1 人脸验证评测（官方 pairs.txt，10 折）。

流程：
  YuNet 检测 -> SFace alignCrop -> SFace feature -> L2 norm -> cosine
  - 每折用其余 9 折选阈值，在留出折上评估
  - 同时报告全量 EER / AUC

注意：
- 用 LFW 原图重新对齐，不要用 deep-funneled 版本（对齐模板不同）。
- 检测失败的对子会跳过，结果里会报告 skipped 数量。
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import cv2

try:
    from .embedder import FaceEmbedder, imread_unicode
    from .paths import resolve
except ImportError:
    from embedder import FaceEmbedder, imread_unicode
    from paths import resolve


def parse_pairs(pairs_path: Path) -> list[list[tuple[str, int, str, int, int]]]:
    """解析 LFW pairs.txt，返回每折的 (name1, idx1, name2, idx2, label)。"""
    lines = [
        line.strip()
        for line in pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise SystemExit(f"pairs 文件为空: {pairs_path}")

    n_folds = int(lines[0].split()[0])
    pairs: list[tuple[str, int, str, int, int]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 3:
            name, i, j = parts
            pairs.append((name, int(i), name, int(j), 1))
        elif len(parts) == 4:
            name1, i, name2, j = parts
            pairs.append((name1, int(i), name2, int(j), 0))
        else:
            raise ValueError(f"无法解析 pairs 行: {line!r}")

    if len(pairs) % n_folds != 0:
        raise ValueError(f"pairs 数量 {len(pairs)} 不能被折数 {n_folds} 整除")
    fold_size = len(pairs) // n_folds
    return [pairs[i * fold_size : (i + 1) * fold_size] for i in range(n_folds)]


def image_path(root: Path, name: str, index: int) -> Path:
    return root / name / f"{name}_{index:04d}.jpg"


def cache_meta(args: argparse.Namespace) -> dict:
    yunet = resolve(args.yunet)
    sface = resolve(args.sface)
    return {
        "yunet": str(yunet),
        "sface": str(sface),
        "align": args.align,
        "yunet_mtime": yunet.stat().st_mtime if yunet.exists() else None,
        "sface_mtime": sface.stat().st_mtime if sface.exists() else None,
        "score": args.score,
        "opencv": cv2.__version__,
    }


def load_cache(path: Path, meta: dict) -> dict[str, np.ndarray | None]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            cache = pickle.load(handle)
    except Exception as exc:
        print(f"缓存读取失败（{exc}），重新计算")
        return {}
    if cache.get("meta") != meta:
        print("缓存与当前模型不匹配，重新计算 embedding")
        return {}
    print(f"加载缓存: {path} ({len(cache['embeddings'])} 张图)")
    return cache["embeddings"]


def save_cache(
    path: Path, meta: dict, embeddings: dict[str, np.ndarray | None]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        pickle.dump({"meta": meta, "embeddings": embeddings}, handle)
    tmp.replace(path)


def get_embedding(
    embedder: FaceEmbedder,
    cache: dict[str, np.ndarray | None],
    path: Path,
    bridge=None,
) -> np.ndarray | None:
    key = str(path)
    if key in cache:
        return cache[key]

    image = imread_unicode(path)
    if image is None:
        cache[key] = None
        return None

    try:
        if bridge is not None:
            faces = [f for f in bridge.detect(image) if f.ok]
            feat = None
            if faces:
                face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                feat = embedder.embed_aligned(face.aligned)
        else:
            feat = embedder.embed(image)
    except Exception:
        feat = None
    cache[key] = feat
    return feat


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """在给定数据上找使准确率最高的余弦阈值（score >= t 判为同一人）。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if scores.size == 0:
        return 0.0, float("nan")

    if np.unique(labels).size < 2:
        # 退化情况：全正类用 -inf，全负类用 +inf，保证预测全对
        return (-np.inf if labels[0] == 1 else np.inf), 1.0

    candidates = np.concatenate(([-np.inf], np.unique(scores), [np.inf]))
    predictions = scores[None, :] >= candidates[:, None]
    accuracies = (predictions == labels[None, :]).mean(axis=1)
    index = int(np.argmax(accuracies))
    return float(candidates[index]), float(accuracies[index])


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = labels[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    if tps[-1] == 0 or fps[-1] == 0:
        return float("nan")

    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = labels[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    if tps[-1] == 0 or fps[-1] == 0:
        return float("nan")

    tpr = np.concatenate(([0.0], tps / tps[-1]))
    fpr = np.concatenate(([0.0], fps / fps[-1]))
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="LFW 1:1 验证评测")
    parser.add_argument("--lfw", required=True, help="LFW 根目录（含 pairs.txt）")
    parser.add_argument("--pairs", default=None, help="默认 <lfw>/pairs.txt")
    parser.add_argument("--yunet", default="models/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--sface", default="models/face_recognition_sface_2021dec.onnx")
    parser.add_argument("--score", type=float, default=0.6)
    parser.add_argument(
        "--align",
        choices=["sface", "evolve"],
        default="sface",
        help="sface=SFace alignCrop（baseline）/ evolve=A 的 evoLVe 112 模板（主口径）",
    )
    parser.add_argument("--cache", default=None, help="默认 outputs/lfw_embeddings_<align>.pkl")
    parser.add_argument("--output", default=None, help="默认 outputs/lfw_<align>.json")
    parser.add_argument("--force", action="store_true", help="忽略缓存")
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=0,
        help="每折只取前 N 对，冒烟测试用",
    )
    args = parser.parse_args()

    lfw = resolve(args.lfw)
    pairs_path = resolve(args.pairs) if args.pairs else lfw / "pairs.txt"
    cache_path = (
        resolve(args.cache) if args.cache
        else resolve(f"outputs/lfw_embeddings_{args.align}.pkl")
    )
    for path in (lfw, pairs_path, resolve(args.yunet), resolve(args.sface)):
        if not path.exists():
            raise SystemExit(f"路径不存在: {path}")

    folds = parse_pairs(pairs_path)
    if args.limit_pairs > 0:
        per_fold = max(2, args.limit_pairs // len(folds))
        # 每折前一半是同人对、后一半是异人对，等间距抽样才不会全取成正例
        folds = [fold[:: max(1, len(fold) // per_fold)][:per_fold] for fold in folds]
        print(f"警告：每折只使用 {per_fold} 对（等间距抽样），结果仅用于冒烟测试")

    embedder = FaceEmbedder(args.yunet, args.sface, score_threshold=args.score)
    bridge = None
    if args.align == "evolve":
        from .bridge import FrontendBridge

        bridge = FrontendBridge(conf=args.score)
    meta = cache_meta(args)
    cache: dict[str, np.ndarray | None] = (
        {} if args.force else load_cache(cache_path, meta)
    )

    scores: list[float] = []
    labels: list[int] = []
    fold_ids: list[int] = []
    skipped = 0

    for fold_index, fold in enumerate(folds):
        for name1, index1, name2, index2, label in fold:
            feat1 = get_embedding(embedder, cache, image_path(lfw, name1, index1), bridge)
            feat2 = get_embedding(embedder, cache, image_path(lfw, name2, index2), bridge)
            if feat1 is None or feat2 is None:
                skipped += 1
                continue
            scores.append(float(np.dot(feat1, feat2)))
            labels.append(label)
            fold_ids.append(fold_index)
        save_cache(cache_path, meta, cache)
        print(
            f"fold {fold_index + 1}/{len(folds)}: pairs={len(scores)} "
            f"skipped={skipped} cache={len(cache)}"
        )

    if not scores:
        raise SystemExit("没有任何有效对子，检查数据和模型")

    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int32)
    fold_ids_array = np.asarray(fold_ids, dtype=np.int32)

    fold_accuracies: list[float] = []
    for fold_index in range(len(folds)):
        train_mask = fold_ids_array != fold_index
        test_mask = fold_ids_array == fold_index
        if not test_mask.any():
            continue
        if np.unique(labels_array[train_mask]).size < 2:
            continue
        threshold, _ = best_threshold(
            scores_array[train_mask], labels_array[train_mask]
        )
        accuracy = float(
            ((scores_array[test_mask] >= threshold) == labels_array[test_mask]).mean()
        )
        fold_accuracies.append(accuracy)

    global_threshold, global_accuracy = best_threshold(scores_array, labels_array)
    eer = compute_eer(scores_array, labels_array)
    auc = compute_auc(scores_array, labels_array)
    detection_failures = sum(1 for value in cache.values() if value is None)

    result = {
        "n_pairs": int(scores_array.size),
        "n_skipped": skipped,
        "n_images_cached": len(cache),
        "n_detection_failures": detection_failures,
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(np.mean(fold_accuracies)) if fold_accuracies else None,
        "std_accuracy": float(np.std(fold_accuracies)) if fold_accuracies else None,
        "global_threshold": global_threshold,
        "global_accuracy": global_accuracy,
        "eer": eer,
        "auc": auc,
        "model": {
            "yunet": str(resolve(args.yunet)),
            "sface": str(resolve(args.sface)),
            "align": args.align,
            "score_threshold": args.score,
        },
    }

    output = (
        resolve(args.output) if args.output
        else resolve(f"outputs/lfw_{args.align}.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if fold_accuracies:
        print(
            f"LFW {len(folds)}-fold accuracy: {result['mean_accuracy']:.4f} "
            f"± {result['std_accuracy']:.4f}"
        )
    print(f"EER: {eer:.4f}  AUC: {auc:.4f}")
    print(f"全量最佳阈值: {global_threshold:.4f}  accuracy: {global_accuracy:.4f}")
    print(f"skipped pairs: {skipped}  检测失败图片: {detection_failures}")
    print(f"结果已保存: {output}")


if __name__ == "__main__":
    main()

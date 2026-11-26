"""开集 / 闭集 1:N 人脸识别评测协议。

用于替代自采数据的指标来源：注册库来自公开数据集，探针按身份不重叠划分。

布局支持
--------
dirs  : root/<identity>/<img>.jpg              LFW、A 的测试集、任何按人分目录的数据
csv   : --csv annotations.csv                  UMDFaces 的 umdfaces_batchN_ultraface.csv
        第 1 列是图片相对路径，subject id 自动识别；可用 --file-col / --id-col 指定

协议
----
1. 只保留图片数 >= --min-images 的身份，随机采样 --n-ids 个（0 = 全部）
2. 按身份切分：gallery 身份（每人 --shots 张注册，其余作 genuine probe）
                imposter 身份（全部图都不注册，作 stranger probe）
3. 探针再按 --dev-split 分成 dev / test：阈值只在 dev 上选，在 test 上报告
4. 特征：A 前端检测+对齐 -> rms 质检门 -> SFace 128 维；带 npz 缓存

指标
----
- closed-set Rank-1 / Rank-5（不问阈值）
- open-set：DIR@FPIR=1e-2 / 1e-3，以及 FPIR（陌生人被误认的比例）
- 验证视角：genuine vs max-imposter 分数 -> FAR/FRR/EER

用法
----
python -m backend.protocol_open --mode dirs --root data/selfmade --n-ids 0
python -m backend.protocol_open --mode csv \
    --csv "D:/umdfaces_batch1/umdfaces_batch1_ultraface.csv" \
    --images-root "D:/umdfaces_batch1" --n-ids 400 --shots 3 --min-images 8
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

try:
    from .bridge import FrontendBridge, imread_unicode
    from .embedder import FaceEmbedder
    from .paths import model_path, new_run_dir, resolve
except ImportError:  # 允许 python backend/protocol_open.py 直接运行
    from bridge import FrontendBridge, imread_unicode
    from embedder import FaceEmbedder
    from paths import model_path, new_run_dir, resolve

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# 数据集索引
# ---------------------------------------------------------------------------
def index_dirs(root: Path) -> dict:
    """root/<identity>/<img> 布局。"""
    index: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rel = path.relative_to(root)
            if len(rel.parts) < 2:
                continue
            index.setdefault(rel.parts[0], []).append(path)
    return index


def _pick_columns(header: list, body: list, file_col, id_col) -> tuple:
    if file_col is None:
        for i in range(len(header)):
            hits = sum(
                1 for row in body[:50]
                if i < len(row) and row[i].lower().endswith(tuple(IMAGE_EXTS))
            )
            if hits >= min(10, len(body[:50])):
                file_col = i
                break
        else:
            file_col = 0
    if id_col is None:
        for i, name in enumerate(header):
            if i == file_col:
                continue
            low = str(name).lower()
            if "subject" in low or low in ("id", "person", "name", "class"):
                id_col = i
                break
    if id_col is None:
        for i in range(len(header)):
            if i == file_col:
                continue
            numeric = sum(
                1 for row in body[:50] if i < len(row) and row[i].strip().isdigit()
            )
            if numeric >= min(10, len(body[:50])):
                id_col = i
                break
    if id_col is None:
        raise SystemExit("识别不出 subject id 列，请用 --id-col 指定")
    return file_col, id_col


def _locate(rel: str, images_root: Path) -> Path:
    rel_path = Path(rel.replace("\\", "/"))
    candidates = [images_root / rel_path, images_root.parent / rel_path]
    # 去掉 csv 里可能多出的第一层目录名
    if len(rel_path.parts) > 1:
        candidates.append(images_root / Path(*rel_path.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def index_csv(csv_path: Path, images_root: Path, file_col=None, id_col=None) -> dict:
    rows = list(csv.reader(csv_path.open(encoding="utf-8-sig", errors="ignore")))
    if not rows:
        raise SystemExit(f"CSV 为空: {csv_path}")
    header, body = rows[0], rows[1:]
    if any(str(v).lower().endswith(tuple(IMAGE_EXTS)) for v in rows[0]):
        header, body = [f"c{i}" for i in range(len(rows[0]))], rows  # 无表头
    file_col, id_col = _pick_columns(header, body, file_col, id_col)

    index: dict[str, list[Path]] = {}
    for row in body:
        if file_col >= len(row) or id_col >= len(row):
            continue
        subject = str(row[id_col]).strip()
        rel = row[file_col].strip()
        if not subject or not rel:
            continue
        index.setdefault(subject, []).append(_locate(rel, images_root))
    return index


# ---------------------------------------------------------------------------
# 特征提取（带缓存）
# ---------------------------------------------------------------------------
def load_cache(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = np.load(path, allow_pickle=False)
    return {str(k): v for k, v in zip(data["keys"], data["feats"])}


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.asarray(list(cache.keys()), dtype=str)
    feats = (
        np.stack(list(cache.values())).astype(np.float32)
        if cache else np.empty((0, 128), np.float32)
    )
    np.savez_compressed(path, keys=keys, feats=feats)


def embed_paths(
    paths: list,
    bridge,
    embedder: FaceEmbedder,
    cache: dict,
    cache_path: Path,
    save_every: int = 200,
) -> tuple:
    """返回 (features, skipped)；features 与 paths 一一对应，失败的为 None。

    bridge 为 None 表示输入已是 112×112 对齐人脸（prealigned），直接送特征。
    """
    feats, skipped = [], []
    for i, path in enumerate(paths, start=1):
        key = str(path)
        if key in cache:
            feats.append(cache[key])
            continue
        image = imread_unicode(path)
        if image is None:
            skipped.append({"path": key, "reason": "读图失败"})
            feats.append(None)
            continue
        try:
            if bridge is None:
                feat = embedder.embed_aligned(image)
            else:
                faces = [f for f in bridge.detect(image) if f.ok]
                if not faces:
                    skipped.append({"path": key, "reason": "未检出合格人脸"})
                    feats.append(None)
                    continue
                face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                feat = embedder.embed_aligned(face.aligned)
        except Exception as exc:
            skipped.append({"path": key, "reason": f"特征失败: {exc}"})
            feats.append(None)
            continue
        cache[key] = feat
        feats.append(feat)
        if i % save_every == 0:
            save_cache(cache_path, cache)
            print(f"  [{i}/{len(paths)}] 已缓存 {len(cache)} 条")
    save_cache(cache_path, cache)
    return feats, skipped


# ---------------------------------------------------------------------------
# 协议切分与指标
# ---------------------------------------------------------------------------
def _mean_norm(embs: list) -> np.ndarray:
    mean = np.mean(np.stack(embs), axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 0 else mean


def split_identities(ids: list, rng: random.Random, imposter_frac: float) -> tuple:
    ids = list(ids)
    rng.shuffle(ids)
    n_imp = max(1, int(round(len(ids) * imposter_frac))) if len(ids) > 2 else 0
    return ids[n_imp:], ids[:n_imp]


def build_gallery(
    index: dict, embs: dict, gallery_ids: list, shots: int, rng: random.Random
) -> tuple:
    """返回 (gallery_names, gallery_matrix, genuine_probes[(true_idx, feat, path)])。"""
    names, matrix, probes = [], [], []
    for pid in gallery_ids:
        items = [(p, embs[str(p)]) for p in index[pid] if str(p) in embs]
        if len(items) <= shots:
            continue
        rng.shuffle(items)
        enroll = items[:shots]
        names.append(pid)
        matrix.append(_mean_norm([e for _, e in enroll]))
        true_idx = len(names) - 1
        for path, feat in items[shots:]:
            probes.append((true_idx, feat, str(path)))
    return names, np.stack(matrix) if matrix else np.empty((0, 128), np.float32), probes


def evaluate_curve(gen_scores, imp_scores, thresholds) -> list:
    rows = []
    gen = np.asarray(gen_scores, np.float64)
    imp = np.asarray(imp_scores, np.float64)
    for t in thresholds:
        far = float((imp >= t).mean()) if imp.size else float("nan")
        tpir = float((gen >= t).mean()) if gen.size else float("nan")
        rows.append({"threshold": float(t), "fpir": far, "tvir": tpir, "frr": 1.0 - tpir})
    return rows


def eer_from(gen_scores, imp_scores) -> tuple:
    gen = np.asarray(gen_scores, np.float64)
    imp = np.asarray(imp_scores, np.float64)
    if not gen.size or not imp.size:
        return float("nan"), float("nan")
    grid = np.unique(np.concatenate([gen, imp]))
    best = (float("nan"), float("nan"))
    best_gap = np.inf
    for t in grid:
        frr = float((gen < t).mean())
        far = float((imp >= t).mean())
        if abs(far - frr) < best_gap:
            best_gap, best = abs(far - frr), (0.5 * (far + frr), float(t))
    return best


def threshold_at_fpir(imp_scores, target: float) -> float:
    imp = np.asarray(imp_scores, np.float64)
    if not imp.size:
        return float("nan")
    return float(np.quantile(imp, 1.0 - target))


def run_protocol(args) -> dict:
    rng = random.Random(args.seed)

    # 1) 索引
    root = resolve(args.root) if args.root else None
    if args.mode == "dirs":
        if not root or not root.is_dir():
            raise SystemExit(f"目录不存在: {root}")
        index = index_dirs(root)
    else:
        if not args.csv:
            raise SystemExit("--mode csv 需要 --csv")
        csv_path = resolve(args.csv)
        images_root = resolve(args.images_root) if args.images_root else csv_path.parent
        index = index_csv(csv_path, images_root, args.file_col, args.id_col)

    usable = {k: v for k, v in index.items() if len(v) >= max(args.min_images, args.shots + 1)}
    if len(usable) < 3:
        raise SystemExit(
            f"可用身份只有 {len(usable)} 个（每人 >= {args.min_images} 张）；"
            "协议至少需要 3 个身份（含 imposter）"
        )
    ids = sorted(usable)
    if args.n_ids > 0:
        rng.shuffle(ids)
        ids = ids[: args.n_ids]
        rng = random.Random(args.seed)
    gallery_ids, imposter_ids = split_identities(ids, rng, args.imposter_frac)
    print(
        f"身份: 可用 {len(usable)} / 采样 {len(ids)} -> "
        f"gallery {len(gallery_ids)} + imposter {len(imposter_ids)}"
    )

    # 2) 选图（只嵌 gallery 身份 + imposter 身份的图）
    wanted = []
    for pid in gallery_ids + imposter_ids:
        wanted.extend(usable[pid])
    wanted = sorted(set(wanted))
    print(f"待提特征: {len(wanted)} 张（缓存命中会跳过）")

    out_dir = resolve(args.out) if args.out else new_run_dir("protocol_open")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = resolve(args.cache) if args.cache else out_dir / "embeddings_cache.npz"

    bridge = None if args.prealigned else FrontendBridge(conf=args.conf, max_rms=args.max_rms)
    embedder = FaceEmbedder(model_path(args.yunet), model_path(args.sface))
    cache = load_cache(cache_path)
    feats, skipped = embed_paths(wanted, bridge, embedder, cache, cache_path)
    embs = {str(p): f for p, f in zip(wanted, feats) if f is not None}
    print(f"有效特征: {len(embs)}  跳过: {len(skipped)}")

    # 3) 建库 + 探针
    names, gallery, genuine = build_gallery(usable, embs, gallery_ids, args.shots, rng)
    stranger = []
    for pid in imposter_ids:
        items = [(p, embs[str(p)]) for p in usable[pid] if str(p) in embs]
        rng.shuffle(items)
        for path, feat in items[: args.max_stranger_probes]:
            stranger.append((-1, feat, str(path)))
    if len(names) < 2 or not genuine or not stranger:
        counts = {
            pid: sum(1 for p in usable[pid] if str(p) in embs) for pid in gallery_ids
        }
        raise SystemExit(
            "gallery/探针不足。质量门（rms/小脸/模糊）会丢掉一部分图，"
            f"gallery 身份有效图数需 > --shots({args.shots})：{counts}；"
            f"stranger 探针 {len(stranger)} 个。可调小 --shots/--min-images 或增加数据"
        )

    gallery_norm = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)

    def score_matrix(probe_list):
        if not probe_list:
            return np.empty((0, len(names)), np.float32)
        P = np.stack([f for _, f, _ in probe_list])
        return P @ gallery_norm.T

    # 4) 探针 dev/test 切分（按探针，不按身份）
    all_probes = genuine + stranger
    order = list(range(len(all_probes)))
    rng.shuffle(order)
    n_dev = max(1, int(len(order) * args.dev_split))
    dev_idx, test_idx = set(order[:n_dev]), set(order[n_dev:])

    S_all = score_matrix(all_probes)

    def compute(indices):
        gen, imp, stranger, rows = [], [], [], []
        for i, (true_idx, _, path) in enumerate(all_probes):
            if i not in indices:
                continue
            scores = S_all[i]
            if true_idx >= 0:
                gen.append(float(scores[true_idx]))
                mask = np.ones(len(names), bool)
                mask[true_idx] = False
                imp.append(float(scores[mask].max()) if mask.any() else -1.0)
                rows.append((true_idx, scores, path))
            else:
                top = float(scores.max()) if scores.size else -1.0
                imp.append(top)
                stranger.append(top)
                rows.append((-1, scores, path))
        return gen, imp, stranger, rows

    dev_gen, dev_imp, dev_stranger, _ = compute(dev_idx)
    test_gen, test_imp, test_stranger, test_rows = compute(test_idx)
    if not dev_stranger or not test_stranger:
        raise SystemExit("dev/test 陌生人探针不足，调大 --imposter-frac 或 --max-stranger-probes")
    warnings = []
    fpir_resolution = 1.0 / len(dev_stranger)
    if len(dev_stranger) < 1000:
        warnings.append(
            f"dev 陌生人探针只有 {len(dev_stranger)} 个，FPIR 分辨率 {fpir_resolution:.4f}；"
            "低于该值的 FPIR 目标（如 1e-3）无法可靠校准，需要更多 imposter 身份"
        )

    # 5) 指标
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 4)
    curve = evaluate_curve(dev_gen, dev_imp, thresholds)
    eer, eer_threshold = eer_from(test_gen, test_imp)

    # closed-set Rank-1/5（只看 gallery 身份的探针，不看阈值）
    rank1 = rank5 = float("nan")
    if test_rows:
        gen_rows = [r for r in test_rows if r[0] >= 0]
        if gen_rows:
            hits1 = [int(np.argmax(s)) == t for t, s, _ in gen_rows if s.size]
            top5 = [t in np.argsort(s)[-5:] for t, s, _ in gen_rows if s.size]
            rank1 = float(np.mean(hits1))
            rank5 = float(np.mean(top5))

    open_set = {}
    for target in (1e-2, 1e-3):
        t = threshold_at_fpir(dev_stranger, target)
        dir_rate = float(np.mean([s[t_idx] >= t for t_idx, s, _ in test_rows if t_idx >= 0])) if any(
            t_idx >= 0 for t_idx, _, _ in test_rows) else float("nan")
        fpir = float(np.mean([s.max() >= t for t_idx, s, _ in test_rows if t_idx < 0])) if any(
            t_idx < 0 for t_idx, _, _ in test_rows) else float("nan")
        open_set[f"fpir_{target:g}"] = {
            "threshold": t,
            "DIR": dir_rate,
            "FPIR_measured": fpir,
            "reliable": bool(target >= fpir_resolution),
        }

    result = {
        "warnings": warnings,
        "config": {
            "mode": args.mode,
            "root": str(root) if root else None,
            "csv": args.csv,
            "shots": args.shots,
            "imposter_frac": args.imposter_frac,
            "dev_split": args.dev_split,
            "min_images": args.min_images,
            "max_stranger_probes": args.max_stranger_probes,
            "align": "dataset_112_prealigned" if args.prealigned else "evolve_112 (frontend)",
            "embedding": "sface_2021dec_128d",
            "conf": args.conf,
            "max_rms": args.max_rms,
            "seed": args.seed,
        },
        "data": {
            "n_identities_sampled": len(ids),
            "n_gallery_identities": len(names),
            "n_imposter_identities": len(imposter_ids),
            "n_images_attempted": len(wanted),
            "n_embeddings": len(embs),
            "n_skipped": len(skipped),
            "n_genuine_probes": len(genuine),
            "n_stranger_probes": len(stranger),
            "n_stranger_probes_dev": len(dev_stranger),
            "fpir_resolution": fpir_resolution,
        },
        "closed_set": {"rank1": rank1, "rank5": rank5},
        "verification": {"eer": eer, "eer_threshold": eer_threshold},
        "open_set": open_set,
        "gallery_names": names,
        "skipped": skipped[:50],
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "thresholds_dev.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "fpir", "tvir", "frr"])
        writer.writeheader()
        writer.writerows(curve)
    np.savez_compressed(
        out_dir / "scores.npz",
        dev_gen=np.asarray(dev_gen, np.float32),
        dev_imp=np.asarray(dev_imp, np.float32),
        test_gen=np.asarray(test_gen, np.float32),
        test_imp=np.asarray(test_imp, np.float32),
    )

    print("\n=== 结果 ===")
    print(f"Rank-1: {rank1:.4f}   Rank-5: {rank5:.4f}   (test, {len(genuine)} genuine probes)")
    print(f"EER: {eer:.4f} @ thr {eer_threshold:.3f}   (test, {len(test_gen)}G/{len(test_imp)}I)")
    for key, value in open_set.items():
        print(
            f"{key}: thr={value['threshold']:.3f}  DIR={value['DIR']:.4f}  "
            f"FPIR={value['FPIR_measured']:.4f}"
        )
    print(f"\n输出: {out_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="开集/闭集 1:N 识别评测协议")
    parser.add_argument("--mode", choices=["dirs", "csv"], default="dirs")
    parser.add_argument("--root", default=None, help="dirs 模式的数据根目录")
    parser.add_argument("--csv", default=None, help="csv 模式的标注文件")
    parser.add_argument("--images-root", default=None, help="csv 里图片路径的相对根")
    parser.add_argument("--file-col", type=int, default=None)
    parser.add_argument("--id-col", type=int, default=None)
    parser.add_argument("--n-ids", type=int, default=0, help="采样身份数，0=全部")
    parser.add_argument("--shots", type=int, default=3, help="每人注册张数")
    parser.add_argument("--min-images", type=int, default=8, help="身份至少几张图才入选")
    parser.add_argument("--imposter-frac", type=float, default=0.25)
    parser.add_argument("--max-stranger-probes", type=int, default=2)
    parser.add_argument("--dev-split", type=float, default=0.4, help="探针中用于选阈值的比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--max-rms", type=float, default=8.0)
    parser.add_argument(
        "--prealigned",
        action="store_true",
        help="输入已是 112×112 对齐人脸（如 CASIA parquet 导出），跳过检测/对齐/质检",
    )
    parser.add_argument("--yunet", default="frontend/models/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--sface", default="models/face_recognition_sface_2021dec.onnx")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_protocol(args)


if __name__ == "__main__":
    main()

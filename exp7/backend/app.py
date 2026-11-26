"""动态人脸识别 demo：视频文件输入，逐帧独立检测，单目标（最大脸）策略。

不做跟踪、不接摄像头、不需要真人：注册库由静态照片构建，探针是视频帧。

用法
----
# 1) 用 make_demo_video.py 生成的素材（带逐帧真值，可算帧级准确率）
python -m backend.app --source data/demo/demo.mp4 --gt data/demo/demo_gt.json

# 2) 自带注册库
python -m backend.app --source clip.mp4 --gallery outputs/demo_gallery.npz

# 3) 从视频开头若干帧自动注册一个人（适合任意素材）
python -m backend.app --source frontend/data/demo.mp4 --auto-enroll 20 --name target

产物（默认 outputs/<时间戳>_app/）
- demo_annotated.mp4  画框/姓名/相似度
- frames.csv          逐帧：人数、是否检出、识别结果、分数、耗时
- summary.json        帧级指标、FPS、延迟分位、配置
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    from .bridge import FrontendBridge, imread_unicode
    from .embedder import FaceEmbedder
    from .gallery import FaceGallery
    from .paths import model_path, new_run_dir, resolve
except ImportError:  # 允许 python backend/app.py 直接运行
    from bridge import FrontendBridge, imread_unicode
    from embedder import FaceEmbedder
    from gallery import FaceGallery
    from paths import model_path, new_run_dir, resolve


def _open_video(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    return cap, info


def _open_writer(dest: Path, fps: float, size: tuple):
    """cv2.VideoWriter 对中文路径会静默失败，先写临时 ASCII 路径再移动。"""
    import cv2

    tmp = Path(tempfile.gettempdir()) / f"exp7_demo_{int(time.time() * 1000)}.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise SystemExit("VideoWriter 打开失败（缺少 mp4v 编码器？）")
    return writer, tmp


def build_gallery(
    mapping: dict, bridge, embedder, threshold: float
) -> FaceGallery:
    """mapping: identity -> [图片路径]；每张图走 A 前端对齐后取特征。"""
    gallery = FaceGallery(threshold=threshold)
    used = {}
    for name, paths in mapping.items():
        count = 0
        for path in paths:
            image = imread_unicode(path)
            if image is None:
                continue
            faces = [f for f in bridge.detect(image) if f.ok]
            if not faces:
                continue
            face = max(
                faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
            gallery.enroll(name, embedder.embed_aligned(face.aligned))
            count += 1
        used[name] = count
    empty = [n for n, c in used.items() if c == 0]
    if empty:
        raise SystemExit(f"这些身份没有可用照片: {empty}")
    return gallery


def auto_enroll(
    source: Path, bridge, embedder, frames: int, name: str, threshold: float
) -> FaceGallery:
    """从视频开头若干帧里出现最多的最大脸，注册成一个人。"""
    import cv2

    cap, _ = _open_video(source)
    gallery = FaceGallery(threshold=threshold)
    taken = 0
    while taken < frames:
        ok, frame = cap.read()
        if not ok:
            break
        taken += 1
        faces = [f for f in bridge.detect(frame) if f.ok]
        if not faces:
            continue
        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        gallery.enroll(name, embedder.embed_aligned(face.aligned))
    cap.release()
    if len(gallery) == 0:
        raise SystemExit("自动注册失败：开头这些帧里没有合格人脸")
    return gallery


def load_gt(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    frames = {}
    for seg in segments:
        for index in range(int(seg["start"]), int(seg["end"])):
            frames[index] = seg["identity"]
    data["frame_identity"] = frames
    return data


def run(args) -> dict:
    import cv2

    source = resolve(args.source)
    cap, info = _open_video(source)
    out_dir = resolve(args.out) if args.out else new_run_dir("app")
    out_dir.mkdir(parents=True, exist_ok=True)

    bridge = FrontendBridge(conf=args.conf, max_rms=args.max_rms)
    embedder = FaceEmbedder(model_path(args.yunet), model_path(args.sface))

    gt = load_gt(resolve(args.gt)) if args.gt else None
    if args.gallery:
        gallery = FaceGallery.load(resolve(args.gallery))
    elif gt:
        gallery = build_gallery(gt["gallery"], bridge, embedder, args.threshold)
    elif args.auto_enroll > 0:
        gallery = auto_enroll(
            source, bridge, embedder, args.auto_enroll, args.name, args.threshold
        )
    else:
        cap.release()
        raise SystemExit("需要 --gallery / --gt / --auto-enroll 三者之一来建立注册库")

    gallery.save(out_dir / "gallery.npz")
    print(f"注册库: {len(gallery)} 人  {gallery.names}  阈值 {gallery.threshold}")
    print(f"视频: {info['width']}x{info['height']}  {info['frames']} 帧  {info['fps']:.1f}fps")

    writer = tmp_path = None
    if not args.no_video:
        writer, tmp_path = _open_writer(out_dir / "demo_annotated.mp4", info["fps"],
                                        (info["width"], info["height"]))

    rows, latencies = [], []
    records = []  # (gt_truth, best_name|None, score|None) 供阈值扫描复用
    stats = {"frames": 0, "no_face": 0, "rejected": 0, "accepted": 0}
    by_class = {"genuine_frames": 0, "genuine_no_face": 0, "genuine_detected": 0,
                "stranger_frames": 0, "stranger_no_face": 0, "stranger_detected": 0}
    correct_detected = 0
    confusion = {"correct_accept": 0, "false_reject": 0, "misident": 0,
                 "correct_reject": 0, "false_accept": 0}

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % max(1, args.stride) != 0:
            frame_index += 1
            continue
        if args.limit and stats["frames"] >= args.limit:
            break

        started = time.perf_counter()
        faces = bridge.detect(frame)
        good = [f for f in faces if f.ok]
        truth = gt["frame_identity"].get(frame_index, "") if gt is not None else ""
        stats["frames"] += 1
        row = {"frame": frame_index, "n_faces": len(faces), "n_ok": len(good)}
        pred_raw, score_val = None, None

        if not good:
            stats["no_face"] += 1
            row.update({"face": 0, "pred": "", "score": "", "known": ""})
            if writer is not None:
                cv2.putText(frame, "no face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            face = max(
                good, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
            feat = embedder.embed_aligned(face.aligned)
            match = gallery.match(feat)
            pred_raw, score_val = match.name, float(match.score)
            label = match.name if match.known else "UNKNOWN"
            stats["accepted" if match.known else "rejected"] += 1
            row.update({"face": 1, "pred": label, "score": round(match.score, 4),
                        "known": match.known, "rms": round(face.rms, 2),
                        "bbox": [round(v, 1) for v in face.bbox]})

            if writer is not None:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                color = (0, 200, 0) if match.known else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {match.score:.2f}"
                cv2.putText(frame, text, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed_ms)
        row["ms"] = round(elapsed_ms, 1)

        if gt is not None:
            row["gt"] = truth
            records.append((truth, pred_raw, score_val))
            if truth:
                enrolled = gt.get("enrolled", [])
                is_genuine = truth in enrolled
                kind = "genuine" if is_genuine else "stranger"
                by_class[f"{kind}_frames"] += 1
                if score_val is None:
                    by_class[f"{kind}_no_face"] += 1
                else:
                    by_class[f"{kind}_detected"] += 1
                predicted = pred_raw if (score_val is not None and score_val >= args.threshold) else "UNKNOWN"
                if is_genuine:
                    if predicted == truth:
                        confusion["correct_accept"] += 1
                        correct_detected += 1 if score_val is not None else 0
                    elif predicted == "UNKNOWN":
                        confusion["false_reject"] += 1
                    else:
                        confusion["misident"] += 1
                else:
                    if predicted == "UNKNOWN":
                        confusion["correct_reject"] += 1
                        correct_detected += 1 if score_val is not None else 0
                    else:
                        confusion["false_accept"] += 1

        rows.append(row)
        if writer is not None:
            writer.write(frame)
        if args.show:
            cv2.imshow("demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
        shutil.move(str(tmp_path), str(out_dir / "demo_annotated.mp4"))
    if args.show:
        cv2.destroyAllWindows()

    # 阈值扫描：检测/特征只算一次，用同一批分数离线重算不同阈值下的帧级混淆
    sweep = []
    if gt is not None and records:
        enrolled = set(gt.get("enrolled", []))
        for text in str(args.sweep).split(","):
            if not text.strip():
                continue
            t = float(text)
            conf = {"correct_accept": 0, "false_reject": 0, "misident": 0,
                    "correct_reject": 0, "false_accept": 0}
            for truth, name, score in records:
                predicted = name if (name is not None and score is not None and score >= t) else "UNKNOWN"
                if truth in enrolled:
                    if predicted == truth:
                        conf["correct_accept"] += 1
                    elif predicted == "UNKNOWN":
                        conf["false_reject"] += 1
                    else:
                        conf["misident"] += 1
                elif predicted == "UNKNOWN":
                    conf["correct_reject"] += 1
                else:
                    conf["false_accept"] += 1
            total = sum(conf.values())
            sweep.append({
                "threshold": t,
                "accuracy": round((conf["correct_accept"] + conf["correct_reject"]) / total, 4)
                if total else None,
                "confusion": conf,
            })

    with (out_dir / "frames.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = sorted({k for row in rows for k in row})
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    summary = {
        "source": str(source),
        "video": info,
        "config": {
            "threshold": args.threshold,
            "conf": args.conf,
            "max_rms": args.max_rms,
            "stride": args.stride,
            "strategy": "single_target_largest_face",
            "align": "evolve_112 (frontend)",
            "embedding": "sface_2021dec_128d",
            "gallery": gallery.names,
        },
        "stats": stats,
        "by_class": by_class,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2) if latencies else None,
            "p50": round(statistics.median(latencies), 2) if latencies else None,
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2) if latencies else None,
            "fps_mean": round(1000.0 / statistics.fmean(latencies), 2) if latencies else None,
        },
        "confusion": confusion if gt is not None else None,
        "sweep": sweep or None,
    }
    if gt is not None:
        total = sum(confusion.values())
        summary["frame_accuracy"] = (
            round((confusion["correct_accept"] + confusion["correct_reject"]) / total, 4)
            if total else None
        )
        if sweep:
            best = max(sweep, key=lambda item: item["accuracy"] or 0.0)
            summary["best_threshold"] = best
        detected_total = by_class["genuine_detected"] + by_class["stranger_detected"]
        summary["accuracy_detected"] = (
            round(correct_detected / detected_total, 4) if detected_total else None
        )
        summary["detection_rate"] = (
            round((stats["frames"] - stats["no_face"]) / stats["frames"], 4)
            if stats["frames"] else None
        )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n处理帧数 {stats['frames']}  无人脸 {stats['no_face']}  "
          f"接受 {stats['accepted']}  拒识 {stats['rejected']}")
    if latencies:
        print(f"延迟 均值 {summary['latency_ms']['mean']}ms  p95 {summary['latency_ms']['p95']}ms  "
              f"≈ {summary['latency_ms']['fps_mean']} FPS")
    if gt is not None:
        print(f"帧级: {summary['confusion']}  accuracy {summary['frame_accuracy']}")
        print(f"检出率 {summary['detection_rate']:.3f}  检出帧准确率 {summary['accuracy_detected']:.4f}"
              f"  分类: {by_class}")
        for item in sweep:
            conf = item["confusion"]
            print(f"  阈值 {item['threshold']:.2f}: 准确率 {item['accuracy']:.4f}  "
                  f"误接受 {conf['false_accept']}  误拒 {conf['false_reject']}  "
                  f"误认 {conf['misident']}")
        if summary.get("best_threshold"):
            print(f"最佳阈值: {summary['best_threshold']['threshold']:.2f} "
                  f"(accuracy {summary['best_threshold']['accuracy']:.4f})")
    print(f"输出: {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="视频文件的人脸识别 demo")
    parser.add_argument("--source", required=True, help="视频文件路径")
    parser.add_argument("--gt", default=None, help="make_demo_video.py 生成的逐帧真值")
    parser.add_argument("--gallery", default=None, help="已有的 gallery.npz")
    parser.add_argument("--auto-enroll", type=int, default=0, help="用开头 N 帧自动注册")
    parser.add_argument("--name", default="target", help="自动注册时的身份名")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--sweep", default="0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--max-rms", type=float, default=8.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--yunet", default="frontend/models/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--sface", default="models/face_recognition_sface_2021dec.onnx")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

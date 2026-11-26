# -*- coding: utf-8 -*-
"""
pipeline.py —— 批量处理 / 检测器对比 / 阈值扫描 / 视频流
========================================================

对外主要入口：

    process_folder(cfg, root, out_dir)        单检测器批量跑一个目录
    compare_detectors(cfg, root, out_dir)     YuNet vs CenterFace 对比表
    sweep_threshold(cfg, root, out_dir)       置信度阈值扫描曲线数据
    process_video(cfg, path, out_dir)         视频流检测 + 对齐（含 FPS 统计）

所有结果都会落盘成 CSV / JSON / Markdown，方便直接贴进实验报告。
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detectors import (
    DEFAULT_CENTERFACE_MODEL,
    DEFAULT_YUNET_MODEL,
    Face,
    build_detector,
    iou_matrix,
)
from .io_utils import IMAGE_EXTS, ensure_dir, imread_unicode, imwrite_unicode, list_images, rel_name
from .quality import DEFAULT_THRESHOLDS, align_face, classify_failure, face_metrics
from .viz import draw_faces, make_grid, stack_h

CSV_HEADER = [
    "image", "detector", "face_idx", "score",
    "x1", "y1", "x2", "y2", "bbox_w", "bbox_h", "min_side", "aspect",
    "frontal", "align_rms", "align_scale", "roll_deg", "sharp",
    "landmark_order", "reasons", "aligned_path",
]


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class Config:
    detector: str = "yunet"
    yunet_model: str = DEFAULT_YUNET_MODEL
    centerface_model: str = DEFAULT_CENTERFACE_MODEL
    conf: Optional[float] = None        # None -> 用各检测器默认值
    nms: float = 0.3
    crop: int = 112
    order: str = "auto"
    reflective: bool = False
    max_input_side: Optional[int] = None
    save_vis: bool = True
    save_crops: bool = True
    thresholds: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))


def build_detectors(cfg: Config, names: Sequence[str] = None) -> "OrderedDict[str, object]":
    """按名字批量构造检测器，名字顺序即输出顺序。"""
    names = list(names) if names else [cfg.detector]
    dets = OrderedDict()
    for name in names:
        if name == "yunet":
            dets[name] = build_detector(
                "yunet",
                model_path=cfg.yunet_model,
                conf_thresh=cfg.conf if cfg.conf is not None else 0.6,
                nms_thresh=cfg.nms,
                max_input_side=cfg.max_input_side,
            )
        elif name == "centerface":
            dets[name] = build_detector(
                "centerface",
                model_path=cfg.centerface_model,
                conf_thresh=cfg.conf if cfg.conf is not None else 0.5,
                nms_thresh=cfg.nms,
                max_input_side=cfg.max_input_side,
            )
        else:
            raise ValueError("未知检测器：{}".format(name))
    return dets


# ---------------------------------------------------------------------------
# 单图处理
# ---------------------------------------------------------------------------
def run_one(cfg: Config, detector, path: Path, out_dir: Path, root: Path) -> Tuple[List[dict], dict]:
    """处理一张图：检测 -> 逐脸对齐 -> 写图 -> 返回 (行记录, 本图统计)。"""
    img = imread_unicode(path)
    if img is None:
        return [], {"image": str(path), "error": "read_failed", "n_faces": 0, "detect_ms": 0.0}

    t0 = time.perf_counter()
    faces = detector.detect(img)
    detect_ms = (time.perf_counter() - t0) * 1000.0

    faces = sorted(faces, key=lambda f: (-f.score, f.x1))

    stem = Path(rel_name(path, root)).with_suffix("")
    rows: List[dict] = []
    crops: List[np.ndarray] = []

    for idx, face in enumerate(faces):
        aligned = align_face(
            img, face.landmarks, crop_size=cfg.crop, order=cfg.order, reflective=cfg.reflective
        )
        m = face_metrics(face, aligned, img)
        reasons = classify_failure(m, cfg.thresholds)

        aligned_path = ""
        if cfg.save_crops and aligned.ok:
            aligned_path = str(
                Path("aligned") / "{}_{}_f{}.jpg".format(stem, detector.name, idx)
            ).replace(os.sep, "/")
            imwrite_unicode(out_dir / aligned_path, aligned.image)
            crops.append(aligned.image)

        rows.append(
            {
                "image": str(rel_name(path, root)),
                "detector": detector.name,
                "face_idx": idx,
                "score": m["score"],
                "x1": face.x1, "y1": face.y1, "x2": face.x2, "y2": face.y2,
                "bbox_w": m["bbox_w"], "bbox_h": m["bbox_h"],
                "min_side": m["min_side"], "aspect": m["aspect"],
                "frontal": m["frontal"],
                "align_rms": m["align_rms"],
                "align_scale": m["align_scale"],
                "roll_deg": m["roll_deg"],
                "sharp": m["sharp"],
                "landmark_order": aligned.order,
                "reasons": "|".join(reasons),
                "aligned_path": aligned_path,
            }
        )

    if cfg.save_vis:
        title = "{}: {} face(s)  {:.1f} ms".format(detector.name, len(faces), detect_ms)
        vis = draw_faces(img, faces, title=title)
        imwrite_unicode(out_dir / "vis" / "{}_{}.jpg".format(stem, detector.name), vis)

    if cfg.save_crops and crops:
        grid = make_grid(crops, cols=min(8, len(crops)), cell=max(64, cfg.crop))
        imwrite_unicode(out_dir / "grid" / "{}_{}_grid.jpg".format(stem, detector.name), grid)

    return rows, {
        "image": str(rel_name(path, root)),
        "error": "",
        "n_faces": len(faces),
        "detect_ms": detect_ms,
        "scores": [float(f.score) for f in faces],
    }


def _write_csv(path: Path, rows: List[dict], header: Sequence[str]) -> None:
    ensure_dir(path.parent)
    # utf-8-sig：Excel 直接双击打开中文不乱码
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _fmt_cell(r.get(k, "")) for k in header})


def _fmt_cell(v):
    if isinstance(v, float):
        return "{:.4f}".format(v) if np.isfinite(v) else ""
    return v


def summarize(per_image: List[dict], rows: List[dict], detector_name: str, elapsed_ms: float) -> dict:
    n_img = len(per_image)
    n_ok = sum(1 for p in per_image if p.get("error") == "")
    imgs_with_face = sum(1 for p in per_image if p.get("error") == "" and p.get("n_faces", 0) > 0)
    total_detect_ms = sum(p.get("detect_ms", 0.0) for p in per_image)
    total_faces = sum(p.get("n_faces", 0) for p in per_image)

    confs = [r["score"] for r in rows]
    rmss = [r["align_rms"] for r in rows if np.isfinite(r["align_rms"])]
    reason_counter = Counter()
    for r in rows:
        if r["reasons"]:
            for tag in r["reasons"].split("|"):
                reason_counter[tag] += 1

    return {
        "detector": detector_name,
        "images": n_img,
        "images_readable": n_ok,
        "images_with_face": imgs_with_face,
        "image_hit_rate": round(imgs_with_face / n_ok, 4) if n_ok else 0.0,
        "faces": total_faces,
        "faces_per_image": round(total_faces / n_ok, 3) if n_ok else 0.0,
        "mean_score": round(float(np.mean(confs)), 4) if confs else 0.0,
        "mean_align_rms": round(float(np.mean(rmss)), 4) if rmss else 0.0,
        "detect_ms_per_image": round(total_detect_ms / n_ok, 3) if n_ok else 0.0,
        "fps": round(1000.0 * n_ok / total_detect_ms, 2) if total_detect_ms > 0 else 0.0,
        "wall_ms_total": round(elapsed_ms, 1),
        "fail_reasons": dict(reason_counter),
        "failed_faces": int(sum(reason_counter.values())),
    }


# ---------------------------------------------------------------------------
# 批量
# ---------------------------------------------------------------------------
def process_folder(
    cfg: Config,
    root,
    out_dir,
    limit: Optional[int] = None,
    detector_names: Sequence[str] = None,
) -> dict:
    """跑一个目录（单检测器），输出 CSV / JSON / 可视化 / 对齐图 / 拼图。"""
    root = Path(root)
    out_dir = ensure_dir(out_dir)
    for sub in ("vis", "aligned", "grid"):
        ensure_dir(out_dir / sub)

    images = list_images(root)
    if limit:
        images = images[:limit]
    if not images:
        raise SystemExit("目录里没有找到图片：{}".format(root))

    dets = build_detectors(cfg, detector_names)
    results = {}

    for name, det in dets.items():
        t0 = time.perf_counter()
        all_rows, per_image = [], []
        for i, p in enumerate(images, 1):
            rows, stat = run_one(cfg, det, p, out_dir, root)
            all_rows.extend(rows)
            per_image.append(stat)
            if i % 20 == 0 or i == len(images):
                print("  [{}] {}/{}".format(name, i, len(images)), end="\r", flush=True)
        elapsed = (time.perf_counter() - t0) * 1000.0
        print(" " * 60, end="\r")

        _write_csv(out_dir / "report_{}.csv".format(name), all_rows, CSV_HEADER)
        summary = summarize(per_image, all_rows, name, elapsed)
        results[name] = summary

        print("\n=== {} ===".format(name))
        print_summary(summary)

    _write_json(out_dir / "summary.json", results)
    return results


def print_summary(s: dict):
    print("  图片: {}（可读 {}，含脸 {}，图片级检出率 {:.1%}）".format(
        s["images"], s["images_readable"], s["images_with_face"], s["image_hit_rate"]))
    print("  人脸: {} 张（平均 {:.2f}/图），平均置信度 {:.3f}".format(
        s["faces"], s["faces_per_image"], s["mean_score"]))
    print("  速度: {:.2f} ms/图，约 {:.1f} FPS（仅检测）".format(
        s["detect_ms_per_image"], s["fps"]))
    print("  对齐: 平均残差 {:.2f} px，质检不合格 {} 张 {}".format(
        s["mean_align_rms"], s["failed_faces"],
        s["fail_reasons"] if s["fail_reasons"] else ""))


def _write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 检测器对比
# ---------------------------------------------------------------------------
COMPARE_HEADER = [
    ("检测器", "detector"),
    ("图片数", "images"),
    ("含脸图片", "images_with_face"),
    ("图片级检出率", "image_hit_rate"),
    ("人脸总数", "faces"),
    ("平均每图", "faces_per_image"),
    ("平均置信度", "mean_score"),
    ("ms/图", "detect_ms_per_image"),
    ("FPS", "fps"),
    ("平均对齐残差(px)", "mean_align_rms"),
    ("质检不合格", "failed_faces"),
]


def compare_detectors(cfg: Config, root, out_dir, limit: Optional[int] = None) -> dict:
    """YuNet vs CenterFace：同一批图、同一套阈值策略，各自跑一遍并出对比表。"""
    out_dir = ensure_dir(out_dir)
    names = ["yunet", "centerface"]
    results = process_folder(cfg, root, out_dir / "run", limit=limit, detector_names=names)

    # --- 两个检测器的一致性（IoU>=0.5 视为同一张脸）---
    agree, detail = _agreement(cfg, root, limit)

    lines = ["| " + " | ".join(h for h, _ in COMPARE_HEADER) + " |",
             "|" + "|".join(["---"] * len(COMPARE_HEADER)) + "|"]
    for name in names:
        s = results.get(name)
        if not s:
            continue
        cells = []
        for _, key in COMPARE_HEADER:
            v = s.get(key, "")
            if isinstance(v, float):
                cells.append("{:.3f}".format(v) if key in ("mean_score", "mean_align_rms", "image_hit_rate")
                             else "{:.2f}".format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    md = ["# YuNet vs CenterFace 检测对比", "",
          "同一批 {} 张图片，conf/nms 取各自默认（YuNet 0.6/0.3，CenterFace 0.5/0.3），"
          "对齐统一使用 evoLVe align_trans + ArcFace 112 模板。".format(
              results.get("yunet", {}).get("images", 0)),
          ""] + lines

    md += ["", "## 两检测器一致性", "",
           "- 可匹配图片数（两边都检出且存在 IoU>=0.5 的配对）：{}".format(agree["images_matched"]),
           "- 平均 IoU：{:.3f}".format(agree["mean_iou"]),
           "- 仅 YuNet 检出的人脸数：{}".format(agree["only_yunet"]),
           "- 仅 CenterFace 检出的人脸数：{}".format(agree["only_centerface"]),
           "- 共同检出的人脸数：{}".format(agree["both"]),
           "", "逐图明细见 `agreement.csv`。"]

    (out_dir / "compare.md").write_text("\n".join(md), encoding="utf-8")
    _write_csv(out_dir / "agreement.csv", detail,
               ["image", "n_yunet", "n_centerface", "matched", "mean_iou"])
    _write_json(out_dir / "compare.json", {"summary": results, "agreement": agree})

    print("\n" + "\n".join(lines))
    return {"summary": results, "agreement": agree}


def _agreement(cfg: Config, root, limit: Optional[int]) -> Tuple[dict, List[dict]]:
    dets = build_detectors(cfg, ["yunet", "centerface"])
    images = list_images(root)
    if limit:
        images = images[:limit]

    detail, ious = [], []
    both = only_y = only_c = 0

    for p in images:
        img = imread_unicode(p)
        if img is None:
            continue
        a = dets["yunet"].detect(img)
        b = dets["centerface"].detect(img)
        if len(a) and len(b):
            m = iou_matrix(np.array([f.bbox for f in a]), np.array([f.bbox for f in b]))
            best = m.max(axis=1)
            matched = int((best >= 0.5).sum())
            ious.extend(best[best >= 0.5].tolist())
            both += matched
            only_y += int(len(a) - matched)
            only_c += int((m.max(axis=0) < 0.5).sum())
            detail.append({
                "image": rel_name(p, root),
                "n_yunet": len(a), "n_centerface": len(b),
                "matched": matched,
                "mean_iou": float(np.mean(best)) if len(best) else 0.0,
            })
        else:
            only_y += len(a)
            only_c += len(b)
            detail.append({
                "image": rel_name(p, root),
                "n_yunet": len(a), "n_centerface": len(b),
                "matched": 0, "mean_iou": 0.0,
            })

    return {
        "images_matched": sum(1 for d in detail if d["matched"] > 0),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "both": both,
        "only_yunet": only_y,
        "only_centerface": only_c,
    }, detail


# ---------------------------------------------------------------------------
# 阈值扫描
# ---------------------------------------------------------------------------
def sweep_threshold(cfg: Config, root, out_dir, confs: Sequence[float]) -> List[dict]:
    """扫置信度阈值，输出「检出率 / 人脸数 / 平均置信度」曲线数据（画 PR 图用）。"""
    out_dir = ensure_dir(out_dir)
    images = list_images(root)
    if not images:
        raise SystemExit("目录里没有找到图片：{}".format(root))

    rows = []
    for name in ("yunet", "centerface"):
        for conf in confs:
            sub = Config(**{**cfg.__dict__, "conf": float(conf), "save_vis": False, "save_crops": False})
            det = build_detectors(sub, [name])[name]
            n_face, imgs_with, scores, t_ms = 0, 0, [], 0.0
            for p in images:
                img = imread_unicode(p)
                if img is None:
                    continue
                t0 = time.perf_counter()
                faces = det.detect(img)
                t_ms += (time.perf_counter() - t0) * 1000.0
                n_face += len(faces)
                imgs_with += 1 if faces else 0
                scores.extend([f.score for f in faces])
            n = max(1, len(images))
            rows.append({
                "detector": name,
                "conf": conf,
                "faces": n_face,
                "images_with_face": imgs_with,
                "hit_rate": round(imgs_with / n, 4),
                "mean_score": round(float(np.mean(scores)), 4) if scores else 0.0,
                "ms_per_image": round(t_ms / n, 3),
            })

    _write_csv(out_dir / "sweep.csv", rows,
               ["detector", "conf", "faces", "images_with_face", "hit_rate", "mean_score", "ms_per_image"])
    _plot_sweep(rows, out_dir / "sweep.png")
    print("阈值扫描完成 -> {}".format(out_dir / "sweep.csv"))
    return rows


def _plot_sweep(rows: List[dict], out_path: Path) -> None:
    """画阈值-检出曲线（matplotlib 缺失时静默跳过，不影响主流程）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("（未安装 matplotlib，跳过曲线绘制）")
        return
    if not rows:
        return

    colors = {"yunet": "#2a9d8f", "centerface": "#e76f51"}
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    metrics = [("faces", "Detected faces"), ("hit_rate", "Image-level hit rate"), ("mean_score", "Mean confidence")]

    for name in sorted({r["detector"] for r in rows}):
        pts = sorted([r for r in rows if r["detector"] == name], key=lambda r: r["conf"])
        xs = [p["conf"] for p in pts]
        for ax, (key, _) in zip(axes, metrics):
            ax.plot(xs, [p[key] for p in pts], "o-", label=name, color=colors.get(name), lw=1.6, ms=4)

    for ax, (key, title) in zip(axes, metrics):
        ax.set_xlabel("confidence threshold")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 视频
# ---------------------------------------------------------------------------
def process_video(cfg: Config, path, out_dir, stride: int = 1, max_frames: Optional[int] = None) -> dict:
    """视频流：逐帧检测 + 对齐，输出标注视频与 FPS 统计。"""
    out_dir = ensure_dir(out_dir)
    name = cfg.detector
    det = build_detectors(cfg, [name])[name]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit("打不开视频：{}".format(path))

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter 在 Windows 上同样不支持中文路径，先写 ASCII 临时名再改名
    tmp = out_dir / "video_tmp.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

    n_read, n_face, t_det, t_align, frames_used = 0, 0, 0.0, 0.0, 0
    crops_for_grid: List[np.ndarray] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_read += 1
        if stride > 1 and (n_read - 1) % stride != 0:
            continue
        if max_frames and frames_used >= max_frames:
            break
        frames_used += 1

        t0 = time.perf_counter()
        faces = det.detect(frame)
        t_det += (time.perf_counter() - t0) * 1000.0
        n_face += len(faces)

        vis = draw_faces(frame, faces)
        for f in faces:
            t1 = time.perf_counter()
            aligned = align_face(frame, f.landmarks, cfg.crop, cfg.order, cfg.reflective)
            t_align += (time.perf_counter() - t1) * 1000.0
            if len(crops_for_grid) < 64 and aligned.ok:
                crops_for_grid.append(aligned.image)

        inst_fps = 1000.0 / max(t_det / max(frames_used, 1), 1e-6)
        cv2.putText(vis, "{}  {:.1f} FPS  faces={}".format(name, inst_fps, len(faces)),
                    (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(vis)

    cap.release()
    writer.release()

    final = out_dir / "video_{}.mp4".format(name)
    try:
        if final.exists():
            final.unlink()
        os.replace(str(tmp), str(final))
    except OSError:
        final = tmp

    if crops_for_grid:
        imwrite_unicode(out_dir / "video_grid.jpg", make_grid(crops_for_grid, cols=8, cell=96))

    stats = {
        "video": str(Path(path).name),
        "frames_read": n_read,
        "frames_used": frames_used,
        "faces": n_face,
        "faces_per_frame": round(n_face / max(frames_used, 1), 3),
        "detect_ms_per_frame": round(t_det / max(frames_used, 1), 3),
        "align_ms_per_frame": round(t_align / max(frames_used, 1), 3),
        "detect_fps": round(1000.0 * frames_used / max(t_det, 1e-6), 2),
        "output": str(final),
    }
    _write_json(out_dir / "video_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats

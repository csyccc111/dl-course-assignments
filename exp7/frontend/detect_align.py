# -*- coding: utf-8 -*-
"""
detect_align.py —— 人脸检测与对齐（方案 A / 成员 A 交付物）
==========================================================

功能
----
1. 检测：YuNet（主方案）/ CenterFace（对比方案），统一输出人脸框 + 5 关键点
2. 对齐：evoLVe `align_trans` 相似变换，裁剪到 112x112（可改 128/160）
3. 批量：目录/视频流跑通，导出可视化、对齐人脸、CSV 报表
4. 对比：两检测器的检出率、耗时、置信度、对齐残差对比表
5. 质检：小脸/侧脸/模糊/对齐差 等失败标签，便于失败样本分析

用法
----
    python detect_align.py image   --input demo.jpg            # 单图，看效果
    python detect_align.py batch   --input data/test --out out  # 批量 + 报表
    python detect_align.py compare --input data/test            # 检测器对比表
    python detect_align.py sweep   --input data/test            # 置信度阈值扫描
    python detect_align.py video   --input demo.mp4             # 视频流 + FPS

常用参数
--------
    --detector {yunet,centerface,both}   默认 yunet
    --conf 0.6        置信度阈值（不填用各检测器默认：YuNet 0.6 / CenterFace 0.5）
    --nms  0.3        NMS 阈值
    --crop 112        对齐输出边长
    --order {auto,direct,swapped}  关键点顺序，auto 会自动判别（默认）
    --reflective      允许镜像相似变换（默认关闭，见 align_trans.py 说明）
    --max-side 1024   先降采样再检测，可显著提速（默认不缩放）
    --min-size 24     小于该像素数的人脸记为 small_face
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from face_frontend.detectors import DEFAULT_CENTERFACE_MODEL, DEFAULT_YUNET_MODEL
from face_frontend.io_utils import ensure_dir, imread_unicode, imwrite_unicode, list_images
from face_frontend.pipeline import (
    Config,
    compare_detectors,
    process_folder,
    process_video,
    run_one,
    sweep_threshold,
)
from face_frontend.quality import DEFAULT_THRESHOLDS, align_face, classify_failure, face_metrics
from face_frontend.viz import draw_align_demo, draw_faces


# ---------------------------------------------------------------------------
def add_common(p: argparse.ArgumentParser):
    p.add_argument("--detector", default="yunet", choices=["yunet", "centerface", "both"])
    p.add_argument("--yunet-model", default=DEFAULT_YUNET_MODEL)
    p.add_argument("--centerface-model", default=DEFAULT_CENTERFACE_MODEL)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--nms", type=float, default=0.3)
    p.add_argument("--crop", type=int, default=112)
    p.add_argument("--order", default="auto", choices=["auto", "direct", "swapped"])
    p.add_argument("--reflective", action="store_true")
    p.add_argument("--max-side", type=int, default=None)
    p.add_argument("--min-size", type=float, default=DEFAULT_THRESHOLDS["min_side"])
    p.add_argument("--max-rms", type=float, default=DEFAULT_THRESHOLDS["max_rms"])
    p.add_argument("--min-frontal", type=float, default=DEFAULT_THRESHOLDS["min_frontal"])
    p.add_argument("--out", default="outputs")
    p.add_argument("--no-vis", action="store_true", help="不保存可视化图")
    p.add_argument("--no-crops", action="store_true", help="不保存对齐人脸")


def cfg_from_args(a) -> Config:
    th = dict(DEFAULT_THRESHOLDS)
    th.update({"min_side": a.min_size, "max_rms": a.max_rms, "min_frontal": a.min_frontal})
    return Config(
        detector=a.detector if a.detector != "both" else "yunet",
        yunet_model=a.yunet_model,
        centerface_model=a.centerface_model,
        conf=a.conf,
        nms=a.nms,
        crop=a.crop,
        order=a.order,
        reflective=bool(a.reflective),
        max_input_side=a.max_side,
        save_vis=not a.no_vis,
        save_crops=not a.no_crops,
        thresholds=th,
    )


# ---------------------------------------------------------------------------
def cmd_image(a):
    cfg = cfg_from_args(a)
    out_dir = ensure_dir(a.out)
    path = Path(a.input)

    names = ["yunet", "centerface"] if a.detector == "both" else [cfg.detector]
    from face_frontend.pipeline import build_detectors

    dets = build_detectors(cfg, names)
    img = imread_unicode(path)
    if img is None:
        raise SystemExit("读不到图片：{}".format(path))

    print("图片: {}  ({}x{})\n".format(path.name, img.shape[1], img.shape[0]))
    header = "{:<12}{:<6}{:<8}{:<10}{:<10}{:<10}{:<8}{:<10}".format(
        "detector", "#", "score", "size", "frontal", "rms(px)", "order", "flags")
    print(header)
    print("-" * len(header))

    for name, det in dets.items():
        faces = det.detect(img)
        faces = sorted(faces, key=lambda f: -f.score)
        if not faces:
            print("{:<12}(no face detected)".format(name))
            continue

        for i, f in enumerate(faces):
            aligned = align_face(img, f.landmarks, cfg.crop, cfg.order, cfg.reflective)
            m = face_metrics(f, aligned, img)
            flags = classify_failure(m, cfg.thresholds)
            print("{:<12}{:<6}{:<8.3f}{:<10}{:<10.3f}{:<10.2f}{:<8}{:<10}".format(
                name, "#{}".format(i), m["score"],
                "{}x{}".format(int(m["bbox_w"]), int(m["bbox_h"])),
                m["frontal"], m["align_rms"], aligned.order, "|".join(flags) or "-"))

            stem = path.stem
            imwrite_unicode(out_dir / "aligned" / "{}_{}_f{}.jpg".format(stem, name, i), aligned.image)
            demo = draw_align_demo(img, f, aligned.image, cfg.crop)
            imwrite_unicode(out_dir / "demo" / "{}_{}_f{}.jpg".format(stem, name, i), demo)

        vis = draw_faces(img, faces, title="{}: {} face(s)".format(name, len(faces)))
        imwrite_unicode(out_dir / "vis" / "{}_{}.jpg".format(path.stem, name), vis)

    print("\n结果已保存到: {}".format(out_dir.resolve()))


def cmd_batch(a):
    cfg = cfg_from_args(a)
    names = ["yunet", "centerface"] if a.detector == "both" else [cfg.detector]
    print("批量处理: {}  detector={}  crop={}  conf={}".format(
        a.input, a.detector, cfg.crop, cfg.conf if cfg.conf is not None else "default"))
    process_folder(cfg, a.input, a.out, limit=a.limit, detector_names=names)
    print("\n输出目录: {}".format(Path(a.out).resolve()))


def cmd_compare(a):
    cfg = cfg_from_args(a)
    res = compare_detectors(cfg, a.input, a.out, limit=a.limit)
    print("\n对比报告: {}".format((Path(a.out) / "compare.md").resolve()))
    return res


def cmd_sweep(a):
    cfg = cfg_from_args(a)
    confs = [float(x) for x in str(a.confs).split(",")]
    sweep_threshold(cfg, a.input, a.out, confs)


def cmd_video(a):
    cfg = cfg_from_args(a)
    cfg.detector = "yunet" if a.detector == "both" else cfg.detector
    process_video(cfg, a.input, a.out, stride=a.stride, max_frames=a.max_frames)


# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="人脸检测 + 5 点对齐（YuNet/CenterFace + evoLVe align_trans）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("image", help="单张图片")
    p.add_argument("--input", required=True)
    add_common(p)
    p.set_defaults(func=cmd_image)

    p = sub.add_parser("batch", help="批量处理目录")
    p.add_argument("--input", required=True)
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 张，调试用")
    add_common(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("compare", help="YuNet vs CenterFace 对比")
    p.add_argument("--input", required=True)
    p.add_argument("--limit", type=int, default=None)
    add_common(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("sweep", help="置信度阈值扫描")
    p.add_argument("--input", required=True)
    p.add_argument("--confs", default="0.3,0.45,0.6,0.7,0.8,0.9")
    add_common(p)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("video", help="视频流")
    p.add_argument("--input", required=True)
    p.add_argument("--stride", type=int, default=1, help="每 N 帧处理一次")
    p.add_argument("--max-frames", type=int, default=None)
    add_common(p)
    p.set_defaults(func=cmd_video)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)

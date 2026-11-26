# -*- coding: utf-8 -*-
"""
analyze_failures.py —— 失败样本分析 / 分场景统计
================================================

把 detect_align.py 跑出来的 report_*.csv 按「场景标签」聚合，输出可以直接贴进
报告的 Markdown 表。典型用法（对应本项目 data/test_hard 里 6 类合成退化）：

    python analyze_failures.py --report outputs/compare_hard/run/report_yunet.csv \
                              --images data/test_hard --mode suffix

    # 一般数据集：每个文件单独一行
    python analyze_failures.py --report outputs/compare/run/report_yunet.csv \
                              --images data/test --mode file

分组方式（--mode）
------------------
    suffix : 取文件名最后一个下划线之后的部分当标签
             例：t00_biden_blur.jpg -> blur
    dir    : 取图片所在的最近一级子目录名
    file   : 一个文件一组（默认）

同时会打印「对齐残差分位数」和「各类失败标签计数」，方便定位是哪一环拖后腿。
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from face_frontend.io_utils import list_images


def group_key(rel: str, mode: str) -> str:
    stem = Path(rel).stem
    if mode == "suffix":
        return stem.rsplit("_", 1)[-1] if "_" in stem else stem
    if mode == "dir":
        parent = str(Path(rel).parent)
        return parent if parent not in ("", ".") else "(root)"
    return stem


def percentile(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="report_xxx.csv 路径")
    ap.add_argument("--images", required=True, help="原图目录（用来统计漏检）")
    ap.add_argument("--mode", default="suffix", choices=["suffix", "dir", "file"])
    ap.add_argument("--out", default=None, help="输出 markdown 路径")
    args = ap.parse_args()

    report = Path(args.report)
    if not report.is_file():
        raise SystemExit("找不到报表：{}".format(report))

    with open(report, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    imgs = list_images(args.images)
    name_of = {}
    for p in imgs:
        try:
            rel = str(Path(p).resolve().relative_to(Path(args.images).resolve()))
        except (ValueError, OSError):
            rel = p.name
        name_of[rel.replace(os.sep, "/")] = p

    per_group = defaultdict(lambda: {
        "images": 0, "hit": 0, "faces": 0, "scores": [], "rms": [], "reasons": Counter(), "roll": []
    })

    seen = set()
    for r in rows:
        rel = r["image"].replace("\\", "/")
        g = group_key(rel, args.mode)
        d = per_group[g]
        d["faces"] += 1
        d["scores"].append(float(r["score"]) if r["score"] else 0.0)
        if r["align_rms"]:
            d["rms"].append(float(r["align_rms"]))
        if r["roll_deg"]:
            d["roll"].append(abs(float(r["roll_deg"])))
        if r["reasons"]:
            for t in r["reasons"].split("|"):
                d["reasons"][t] += 1
        seen.add(rel)

    # 图片级统计（含漏检）
    for rel in name_of:
        g = group_key(rel, args.mode)
        per_group[g]["images"] += 1
        if rel in seen:
            per_group[g]["hit"] += 1

    order = sorted(per_group.keys())
    lines = [
        "| 分组 | 图片数 | 检出图片 | 图片级检出率 | 人脸数 | 平均置信度 | 对齐残差均值(px) | P90 残差 | 主要失败标签 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for g in order:
        d = per_group[g]
        rate = d["hit"] / d["images"] if d["images"] else 0.0
        mean_s = float(np.mean(d["scores"])) if d["scores"] else float("nan")
        mean_r = float(np.mean(d["rms"])) if d["rms"] else float("nan")
        p90 = percentile(d["rms"], 90)
        tags = ", ".join("{}×{}".format(k, v) for k, v in d["reasons"].most_common(3)) or "-"
        lines.append("| {} | {} | {} | {:.1%} | {} | {:.3f} | {:.2f} | {:.2f} | {} |".format(
            g, d["images"], d["hit"], rate, d["faces"], mean_s, mean_r, p90, tags))

    all_rms = [float(r["align_rms"]) for r in rows if r["align_rms"]]
    all_scores = [float(r["score"]) for r in rows if r["score"]]
    summary = [
        "",
        "**整体**：{} 张图片 -> 检出 {} 张（{:.1%}），共 {} 张人脸。".format(
            len(name_of), len(seen), len(seen) / max(len(name_of), 1), len(rows)),
        "对齐残差 P50 = {:.2f} px，P90 = {:.2f} px，最大 {:.2f} px；".format(
            percentile(all_rms, 50), percentile(all_rms, 90), max(all_rms) if all_rms else float("nan")),
        "置信度均值 {:.3f}，最低 {:.3f}。".format(
            float(np.mean(all_scores)) if all_scores else float("nan"),
            min(all_scores) if all_scores else float("nan")),
    ]

    md = ["# 分场景检测 / 对齐统计", "", "报表：`{}`".format(report.name), ""] + lines + summary
    text = "\n".join(md)
    print(text)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print("\n已写入：{}".format(args.out))


if __name__ == "__main__":
    main()

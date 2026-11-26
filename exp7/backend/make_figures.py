"""把各次实验的产物画成报告用图（全部数据驱动，不手改）。

数据来源（缺哪个就跳过哪张图）：
- outputs/lfw_open_1n/scores.npz, outputs/lfw_open_1n/thresholds_dev.csv
- outputs/casia_open_872/scores.npz, outputs/casia_open_872/thresholds_dev.csv
- outputs/casia_open_872/metrics.json, outputs/casia_open_872_shots1/metrics.json
- outputs/cnn_train62/metrics.json, outputs/cmp_head_metrics.json

用法：python -m backend.make_figures
"""

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    from .paths import OUTPUTS_ROOT, resolve
except ImportError:  # 允许 python backend/make_figures.py 直接运行
    from paths import OUTPUTS_ROOT, resolve

FIG_DIR = OUTPUTS_ROOT / "figures"


def _load_npz(path):
    path = resolve(path)
    if not path.is_file():
        print(f"跳过（缺文件）: {path}")
        return None
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _load_json(path):
    path = resolve(path)
    if not path.is_file():
        print(f"跳过（缺文件）: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_thresholds(path):
    path = resolve(path)
    if not path.is_file():
        return None
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def fig_similarity():
    panels = []
    lfw = _load_npz("outputs/lfw_open_1n/scores.npz")
    if lfw:
        panels.append(("LFW 1:N (420 gallery)", lfw))
    casia = _load_npz("outputs/casia_open_872/scores.npz")
    if casia:
        panels.append(("CASIA 1:N (610 gallery)", casia))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4), squeeze=False)
    for ax, (title, data) in zip(axes[0], panels):
        gen = data["test_gen"]
        imp = data["test_imp"]
        bins = np.linspace(-0.2, 1.0, 60)
        ax.hist(imp, bins=bins, alpha=0.65, label=f"imposter (n={len(imp)})", color="#d9534f")
        ax.hist(gen, bins=bins, alpha=0.65, label=f"genuine (n={len(gen)})", color="#2e7d32")
        ax.set_title(title)
        ax.set_xlabel("cosine similarity")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Genuine vs imposter similarity", fontsize=11)
    fig.tight_layout()
    path = FIG_DIR / "fig1_similarity_hist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_threshold():
    panels = []
    lfw = _read_thresholds("outputs/lfw_open_1n/thresholds_dev.csv")
    if lfw:
        panels.append(("LFW 1:N (dev)", lfw))
    casia = _read_thresholds("outputs/casia_open_872/thresholds_dev.csv")
    if casia:
        panels.append(("CASIA 1:N (dev)", casia))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4), squeeze=False)
    for ax, (title, rows) in zip(axes[0], panels):
        thresholds = [r["threshold"] for r in rows]
        ax.plot(thresholds, [r["frr"] for r in rows], label="FRR", color="#2e7d32")
        ax.plot(thresholds, [r["fpir"] for r in rows], label="FPIR", color="#d9534f")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("threshold")
        ax.set_ylabel("rate (log)")
        ax.set_ylim(1e-4, 1.0)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, which="both")
    fig.suptitle("Error rates vs cosine threshold", fontsize=11)
    fig.tight_layout()
    path = FIG_DIR / "fig2_threshold_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_kshot():
    shots3 = _load_json("outputs/casia_open_872/metrics.json")
    shots1 = _load_json("outputs/casia_open_872_shots1/metrics.json")
    if not shots3 or not shots1:
        return None
    labels = ["1 shot", "3 shots"]
    rank1 = [shots1["closed_set"]["rank1"], shots3["closed_set"]["rank1"]]
    dirs = [
        shots1["open_set"]["fpir_0.001"]["DIR"],
        shots3["open_set"]["fpir_0.001"]["DIR"],
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(5.6, 4))
    bars1 = ax.bar(x - width / 2, rank1, width, label="Rank-1 (closed-set)", color="#2e7d32")
    bars2 = ax.bar(x + width / 2, dirs, width, label="DIR @ FPIR=1e-3", color="#1565c0")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("k-shot effect (CASIA, 610 gallery)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "fig3_kshot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_training():
    cnn = _load_json("outputs/cnn_train62/metrics.json")
    if not cnn:
        return None
    history = cnn["history"]
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [h["train_acc"] for h in history], label="train acc", color="#2e7d32")
    axes[0].plot(epochs, [h["test_acc"] for h in history], label="test acc", color="#d9534f")
    head = _load_json("outputs/cmp_head_metrics.json")
    if head:
        axes[0].axhline(head["accuracy"], linestyle="--", color="#1565c0",
                        label=f"SFace+LogReg {head['accuracy']:.4f}")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[0].set_title("From-scratch CNN vs frozen features")

    axes[1].plot(epochs, [h["train_loss"] for h in history], label="train loss", color="#2e7d32")
    axes[1].plot(epochs, [h["test_loss"] for h in history], label="test loss", color="#d9534f")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cross entropy")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    axes[1].set_title("Loss curves")
    fig.tight_layout()
    path = FIG_DIR / "fig4_training.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    produced = []
    for maker in (fig_similarity, fig_threshold, fig_kshot, fig_training):
        path = maker()
        if path:
            produced.append(path)
            print("生成:", path)
    print(f"\n共 {len(produced)} 张，目录: {FIG_DIR}")


if __name__ == "__main__":
    main()

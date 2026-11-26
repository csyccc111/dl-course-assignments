from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import DEFAULT_OUTPUT_DIR
from .summarize import mean_std

DISPLAY = {
    "rnn": "RNN",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "bert-base-chinese": "BERT-base",
    "hfl/rbt3": "rbt3",
}
COLORS = {
    "rnn": "#1a73e8",
    "lstm": "#188038",
    "transformer": "#f9ab00",
    "bert-base-chinese": "#d93025",
    "hfl/rbt3": "#9334e6",
}
MAIN_MODELS = ["rnn", "lstm", "transformer", "bert-base-chinese", "hfl/rbt3"]


def load_summary(output_dir: Path) -> dict:
    path = output_dir / "summary.json"
    if not path.exists():
        raise SystemExit("缺少 outputs/summary.json，请先运行 python -m src.summarize")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def best_run(runs: list[dict], model: str, tag: str = "") -> dict | None:
    candidates = [r for r in runs if r["model"] == model and r.get("tag", "") == tag]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["dev_accuracy"])


def plot_training(runs: list[dict], figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plotted = 0
    for model in MAIN_MODELS:
        run = best_run(runs, model)
        if run is None or not run["history"]:
            continue
        epochs = [row["epoch"] for row in run["history"]]
        axes[0].plot(epochs, [row["dev_accuracy"] for row in run["history"]],
                     marker="o", markersize=3, color=COLORS.get(model), label=f"{DISPLAY.get(model, model)} (s{run['seed']})")
        axes[1].plot(epochs, [row["train_loss"] for row in run["history"]],
                     marker="o", markersize=3, color=COLORS.get(model), label=DISPLAY.get(model, model))
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    axes[0].set_title("Dev accuracy per epoch (best seed per model)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Train loss per epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cross entropy")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_training.png", dpi=150)
    plt.close(fig)


def plot_main_results(runs: list[dict], figures_dir: Path) -> None:
    models = []
    all_mean, all_std, clean_mean, clean_std = [], [], [], []
    for model in MAIN_MODELS:
        rows = [r for r in runs if r["model"] == model and r.get("tag", "") == ""]
        if not rows:
            continue
        models.append(DISPLAY.get(model, model))
        mean, std = mean_std([r["test_all_accuracy"] for r in rows])
        all_mean.append(mean)
        all_std.append(std)
        mean, std = mean_std([r["test_clean_accuracy"] for r in rows])
        clean_mean.append(mean)
        clean_std.append(std)
    if not models:
        return
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, all_mean, width, yerr=all_std, capsize=3, color="#1a73e8", label="test_all")
    ax.bar(x + width / 2, clean_mean, width, yerr=clean_std, capsize=3, color="#188038", label="test_clean")
    for index, (value, err) in enumerate(zip(all_mean, all_std)):
        ax.text(index - width / 2, value + err + 0.005, f"{value:.3f}", ha="center", fontsize=8)
    for index, (value, err) in enumerate(zip(clean_mean, clean_std)):
        ax.text(index + width / 2, value + err + 0.005, f"{value:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_title("Test accuracy (mean ± std over seeds)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig4_main_results.png", dpi=150)
    plt.close(fig)


def plot_confusions(runs: list[dict], figures_dir: Path) -> None:
    entries = []
    for model in MAIN_MODELS:
        run = best_run(runs, model)
        if run is not None:
            entries.append((model, run))
    if not entries:
        return
    cols = 2
    rows = int(np.ceil(len(entries) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (model, run) in zip(axes, entries):
        ax.axis("on")
        matrix = np.asarray(run.get("test_all_confusion", [[0, 0], [0, 0]]))
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred neg", "pred pos"])
        ax.set_yticklabels(["true neg", "true pos"])
        ax.set_title(f"{DISPLAY.get(model, model)} (s{run['seed']}, acc={run['test_all_accuracy']:.3f})")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig5_confusion.png", dpi=150)
    plt.close(fig)


def plot_length_ablation(runs: list[dict], figures_dir: Path) -> None:
    rows = [r for r in runs if r["model"] == "lstm" and r.get("tag", "").startswith("len")]
    if not rows:
        return
    rows.sort(key=lambda r: r["max_len"])
    max_lens = [r["max_len"] for r in rows]
    accuracy = [r["test_clean_accuracy"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(max_lens, accuracy, marker="o", color="#188038")
    for x, y in zip(max_lens, accuracy):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xticks(max_lens)
    ax.set_xlabel("max_len")
    ax.set_ylabel("test_clean accuracy")
    ax.set_title("LSTM sequence-length ablation (seed 42)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig6_len_ablation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成实验报告图表")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(output_dir)
    runs = summary["runs"]
    for run in runs:
        metrics_path = output_dir / "runs" / run["name"] / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                payload = json.load(f)
            run["test_all_confusion"] = payload.get("test_all", {}).get("confusion", [[0, 0], [0, 0]])
    plot_training(runs, figures_dir)
    plot_main_results(runs, figures_dir)
    plot_confusions(runs, figures_dir)
    plot_length_ablation(runs, figures_dir)
    toys_figure = output_dir / "toys" / "fig_sine.png"
    if toys_figure.exists():
        shutil.copyfile(toys_figure, figures_dir / "fig2_sine.png")
    print(f"[figures] -> {figures_dir}", flush=True)


if __name__ == "__main__":
    main()
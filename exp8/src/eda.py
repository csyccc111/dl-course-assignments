from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, save_json
from .data import dedup_train, read_tsv


def main() -> None:
    parser = argparse.ArgumentParser(description="数据集统计与 EDA 图")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_train = read_tsv(data_dir / "train.tsv")
    raw_dev = read_tsv(data_dir / "dev.tsv")
    raw_test = read_tsv(data_dir / "test.tsv")
    train, conflicting = dedup_train(raw_train)
    train_texts = {text for text, _ in train}
    dev_texts = {text for text, _ in raw_dev}
    dev_clean = [(text, label) for text, label in raw_dev if text not in train_texts]
    test_clean = [(text, label) for text, label in raw_test if text not in train_texts and text not in dev_texts]

    def counts(rows):
        values = np.bincount([label for _, label in rows], minlength=2)
        return [int(values[0]), int(values[1])]

    stats = {
        "raw": {"train": len(raw_train), "dev": len(raw_dev), "test": len(raw_test)},
        "unique": {
            "train": len({text for text, _ in raw_train}),
            "dev": len(dev_texts),
            "test": len({text for text, _ in raw_test}),
        },
        "train_dedup": {"rows": len(train), "labels": counts(train), "conflicting_removed": conflicting},
        "dev_clean": {"rows": len(dev_clean), "labels": counts(dev_clean)},
        "test_all": {"rows": len(raw_test), "labels": counts(raw_test)},
        "test_clean": {"rows": len(test_clean), "labels": counts(test_clean)},
        "overlap": {
            "train_test_texts": len(train_texts & {text for text, _ in raw_test}),
            "train_dev_texts": len(train_texts & dev_texts),
        },
    }
    save_json(output_dir / "eda_stats.json", stats)

    train_lengths = np.asarray([len(text) for text, _ in train])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(["negative", "positive"], stats["train_dedup"]["labels"], color=["#d93025", "#188038"])
    axes[0].set_title(f"Train labels (dedup n={len(train)})")
    axes[0].set_ylabel("rows")
    axes[1].hist(train_lengths, bins=60, color="#1a73e8")
    axes[1].axvline(128, color="#f9ab00", linestyle="--", label="128 chars")
    axes[1].axvline(192, color="#d93025", linestyle="--", label="192 chars")
    axes[1].set_title("Train text length (characters)")
    axes[1].set_xlabel("characters")
    axes[1].legend(fontsize=8)
    groups = ["train/test", "train/dev"]
    values = [stats["overlap"]["train_test_texts"], stats["overlap"]["train_dev_texts"]]
    axes[2].bar(groups, values, color="#9334e6")
    axes[2].set_title("Exact duplicate texts across splits")
    axes[2].set_ylabel("texts")
    for index, value in enumerate(values):
        axes[2].text(index, value + 3, str(value), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig1_data.png", dpi=150)
    plt.close(fig)

    print(
        f"[eda] train={len(train)} dev_clean={len(dev_clean)} test_all={len(raw_test)} "
        f"test_clean={len(test_clean)} overlap={stats['overlap']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
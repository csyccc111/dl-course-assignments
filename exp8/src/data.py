from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def clean_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_tsv(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                label = int(row[0])
            except ValueError:
                continue
            if label not in (0, 1):
                continue
            text = clean_text(row[1])
            if text:
                rows.append((text, label))
    return rows


def dedup_train(rows: Sequence[tuple[str, int]]) -> tuple[list[tuple[str, int]], int]:
    labels_by_text: dict[str, set[int]] = defaultdict(set)
    order: list[str] = []
    for text, label in rows:
        if text not in labels_by_text:
            order.append(text)
        labels_by_text[text].add(label)
    kept = [(text, next(iter(labels_by_text[text]))) for text in order if len(labels_by_text[text]) == 1]
    conflicting = sum(1 for labels in labels_by_text.values() if len(labels) > 1)
    return kept, conflicting


def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def _texts_hash(texts: Iterable[str]) -> str:
    h = hashlib.md5()
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


@dataclass
class TokenizedSplit:
    name: str
    ids: np.ndarray
    masks: np.ndarray
    labels: np.ndarray
    texts: list[str]

    def __len__(self) -> int:
        return int(self.labels.shape[0])


class TokenDataset(Dataset):
    def __init__(self, split: TokenizedSplit) -> None:
        self.ids = torch.from_numpy(split.ids)
        self.masks = torch.from_numpy(split.masks)
        self.labels = torch.from_numpy(split.labels)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.ids[index], self.masks[index], self.labels[index]


def subset_split(split: TokenizedSplit, limit: int | None) -> TokenizedSplit:
    if limit is None or limit >= len(split):
        return split
    return TokenizedSplit(
        split.name,
        split.ids[:limit],
        split.masks[:limit],
        split.labels[:limit],
        split.texts[:limit],
    )


def build_tokenizer(name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        fallback = tokenizer.eos_token or tokenizer.sep_token or tokenizer.unk_token
        if fallback is None:
            raise SystemExit(f"{name} 没有可用的 pad/eos/sep token")
        tokenizer.pad_token = fallback
    return tokenizer


def tokenize_texts(tokenizer, texts: Sequence[str], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    return encoded["input_ids"].astype(np.int64), encoded["attention_mask"].astype(np.int64)


def _load_or_tokenize(
    tokenizer,
    texts: Sequence[str],
    labels: Sequence[int],
    max_len: int,
    cache_dir: Path | None,
    split_name: str,
) -> TokenizedSplit:
    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{split_name}_{_safe_name(str(tokenizer.name_or_path))}_{max_len}_{_texts_hash(texts)}.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            return TokenizedSplit(split_name, data["ids"], data["masks"], data["labels"], list(texts))
    ids, masks = tokenize_texts(tokenizer, texts, max_len)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, ids=ids, masks=masks, labels=labels_arr)
    return TokenizedSplit(split_name, ids, masks, labels_arr, list(texts))


def _char_stats(texts: Sequence[str]) -> dict[str, float | int]:
    lengths = np.asarray([len(text) for text in texts], dtype=np.float64)
    if lengths.size == 0:
        return {"count": 0, "mean": 0, "median": 0, "p95": 0, "max": 0}
    return {
        "count": int(lengths.size),
        "mean": round(float(lengths.mean()), 1),
        "median": int(np.median(lengths)),
        "p95": int(np.percentile(lengths, 95)),
        "max": int(lengths.max()),
    }


def _token_stats(split: TokenizedSplit, max_len: int) -> dict[str, float | int]:
    token_counts = split.masks.sum(axis=1)
    truncated = int(np.sum(token_counts >= max_len))
    return {
        "rows": len(split),
        "max_len": max_len,
        "tokens_mean": round(float(token_counts.mean()), 1),
        "tokens_p95": int(np.percentile(token_counts, 95)),
        "truncated_rows": truncated,
        "truncated_pct": round(100.0 * truncated / max(1, len(split)), 1),
    }


def _label_counts(rows: Sequence[tuple[str, int]]) -> dict[str, int]:
    counts = Counter(label for _, label in rows)
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def load_splits(
    data_dir: Path,
    tokenizer_name: str,
    max_len: int,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> tuple[dict[str, TokenizedSplit], dict, object]:
    data_dir = Path(data_dir)
    raw_train = read_tsv(data_dir / "train.tsv")
    raw_dev = read_tsv(data_dir / "dev.tsv")
    raw_test = read_tsv(data_dir / "test.tsv")

    train_rows, conflicting = dedup_train(raw_train)
    train_texts = {text for text, _ in train_rows}
    dev_texts = {text for text, _ in raw_dev}
    dev_rows = [(text, label) for text, label in raw_dev if text not in train_texts]
    test_clean_rows = [
        (text, label) for text, label in raw_test if text not in train_texts and text not in dev_texts
    ]

    tokenizer = build_tokenizer(tokenizer_name)
    cache_path = Path(cache_dir) if (cache_dir is not None and use_cache) else None

    splits: dict[str, TokenizedSplit] = {}
    for name, rows in (
        ("train", train_rows),
        ("dev", dev_rows),
        ("test_all", raw_test),
        ("test_clean", test_clean_rows),
    ):
        texts = [text for text, _ in rows]
        labels = [label for _, label in rows]
        splits[name] = _load_or_tokenize(tokenizer, texts, labels, max_len, cache_path, name)

    train_test_overlap = train_texts & {text for text, _ in raw_test}
    stats = {
        "raw": {
            "train_rows": len(raw_train),
            "dev_rows": len(raw_dev),
            "test_rows": len(raw_test),
            "train_unique": len({text for text, _ in raw_train}),
            "dev_unique": len(dev_texts),
            "test_unique": len({text for text, _ in raw_test}),
        },
        "train_dedup": {
            "rows": len(train_rows),
            "label_counts": _label_counts(train_rows),
            "duplicate_rows_removed": len(raw_train) - len({text for text, _ in raw_train}),
            "conflicting_label_texts_removed": conflicting,
        },
        "dev_clean": {"rows": len(dev_rows), "label_counts": _label_counts(dev_rows)},
        "test_all": {"rows": len(raw_test), "label_counts": _label_counts(raw_test)},
        "test_clean": {"rows": len(test_clean_rows), "label_counts": _label_counts(test_clean_rows)},
        "leakage": {
            "train_test_overlap_texts": len(train_test_overlap),
            "train_test_overlap_pct": round(100.0 * len(train_test_overlap) / max(1, len({t for t, _ in raw_test})), 1),
            "train_dev_overlap_texts": len(train_texts & dev_texts),
        },
        "char_len": {
            "train": _char_stats([t for t, _ in train_rows]),
            "dev": _char_stats([t for t, _ in dev_rows]),
            "test_all": _char_stats([t for t, _ in raw_test]),
            "test_clean": _char_stats([t for t, _ in test_clean_rows]),
        },
        "token": {name: _token_stats(split, max_len) for name, split in splits.items()},
        "tokenizer": tokenizer_name,
    }
    return splits, stats, tokenizer


def make_loader(
    split: TokenizedSplit,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        TokenDataset(split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
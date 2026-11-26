from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import amp_dtype
from .data import TokenizedSplit


def compute_metrics(y_true: list[int], y_pred: list[int], y_prob: list[float] | None = None) -> dict:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    n = int(true.size)
    confusion = np.zeros((2, 2), dtype=np.int64)
    valid = (true >= 0) & (true < 2) & (pred >= 0) & (pred < 2)
    np.add.at(confusion, (true[valid], pred[valid]), 1)

    accuracy = float((true == pred).mean()) if n else 0.0
    per_class = {}
    precisions, recalls, f1s, supports = [], [], [], []
    for label in (0, 1):
        tp = int(confusion[label, label])
        fp = int(confusion[:, label].sum()) - tp
        fn = int(confusion[label, :].sum()) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = int(confusion[label, :].sum())
        per_class[str(label)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    metrics = {
        "n": n,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(float(np.mean(precisions)), 4),
        "macro_recall": round(float(np.mean(recalls)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "weighted_f1": round(float(np.average(f1s, weights=supports)), 4) if sum(supports) else 0.0,
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }
    if y_prob is not None and len(y_prob):
        probs = np.asarray(y_prob, dtype=float)
        metrics["prob_mean"] = round(float(probs.mean()), 4)
        metrics["prob_min"] = round(float(probs.min()), 4)
        metrics["prob_max"] = round(float(probs.max()), 4)
    return metrics


def _extract_logits(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"无法从模型输出中取 logits: {type(output)}")


def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: str = "off",
) -> tuple[list[int], list[int], list[float]]:
    model.eval()
    dtype = amp_dtype(amp)
    enabled = device.type == "cuda" and dtype is not None
    labels: list[int] = []
    preds: list[int] = []
    probs: list[float] = []
    with torch.inference_mode():
        for input_ids, attention_mask, target in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                logits = _extract_logits(model(input_ids, attention_mask))
            prob = torch.softmax(logits.float(), dim=1)[:, 1]
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(target.tolist())
            probs.extend(prob.cpu().tolist())
    return labels, preds, probs


def measure_latency(
    model: torch.nn.Module,
    split: TokenizedSplit,
    device: torch.device,
    amp: str = "off",
    max_samples: int = 100,
) -> dict:
    model.eval()
    dtype = amp_dtype(amp)
    enabled = device.type == "cuda" and dtype is not None
    n = min(max_samples, len(split))
    times: list[float] = []
    with torch.inference_mode():
        for index in range(min(10, n)):
            input_ids = torch.from_numpy(split.ids[index:index + 1]).to(device)
            attention_mask = torch.from_numpy(split.masks[index:index + 1]).to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                model(input_ids, attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for index in range(n):
            input_ids = torch.from_numpy(split.ids[index:index + 1]).to(device)
            attention_mask = torch.from_numpy(split.masks[index:index + 1]).to(device)
            start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                model(input_ids, attention_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(times)
    return {
        "n": n,
        "mean_ms": round(float(values.mean()), 3),
        "p50_ms": round(float(np.percentile(values, 50)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
    }


def write_predictions(
    path: Path,
    split: TokenizedSplit,
    preds: list[int],
    probs: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "pred", "prob_positive", "correct"])
        for text, label, pred, prob in zip(split.texts, split.labels.tolist(), preds, probs):
            writer.writerow([text, label, pred, round(prob, 6), int(pred == label)])
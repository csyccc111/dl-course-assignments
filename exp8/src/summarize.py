from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import DEFAULT_OUTPUT_DIR, save_json


def load_runs(output_dir: Path) -> list[dict]:
    runs = []
    for metrics_path in sorted((output_dir / "runs").glob("*/metrics.json")):
        with open(metrics_path, encoding="utf-8") as f:
            payload = json.load(f)
        run = payload["run"]
        if run.get("tag") == "smoke" or "_smoke_" in run.get("name", ""):
            continue
        train = payload.get("train", {})
        args = run.get("args", {})
        runs.append({
            "name": run["name"],
            "model": run["model"],
            "seed": int(run["seed"]),
            "tag": run.get("tag", ""),
            "max_len": int(args.get("max_len", 0)),
            "params_total": int(payload.get("model", {}).get("params_total", 0)),
            "epochs_run": int(train.get("epochs_run", 0)),
            "best_epoch": int(train.get("best_epoch", 0)),
            "train_seconds": float(train.get("seconds", 0.0)),
            "dev_accuracy": float(payload.get("dev", {}).get("accuracy", 0.0)),
            "dev_macro_f1": float(payload.get("dev", {}).get("macro_f1", 0.0)),
            "test_all_accuracy": float(payload.get("test_all", {}).get("accuracy", 0.0)),
            "test_all_macro_f1": float(payload.get("test_all", {}).get("macro_f1", 0.0)),
            "test_clean_accuracy": float(payload.get("test_clean", {}).get("accuracy", 0.0)),
            "test_clean_macro_f1": float(payload.get("test_clean", {}).get("macro_f1", 0.0)),
            "latency_mean_ms": float(payload.get("latency_ms", {}).get("mean_ms", 0.0)),
            "history": train.get("history", []),
        })
    return runs


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 outputs/runs 下的实验指标")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    runs = load_runs(output_dir)
    runs.sort(key=lambda row: (row["model"], row["tag"], row["seed"]))
    if not runs:
        raise SystemExit("没有找到可汇总的运行结果（outputs/runs/*/metrics.json）")

    fields = [
        "name", "model", "seed", "tag", "max_len", "params_total", "epochs_run", "best_epoch",
        "train_seconds", "dev_accuracy", "dev_macro_f1", "test_all_accuracy",
        "test_all_macro_f1", "test_clean_accuracy", "test_clean_macro_f1", "latency_mean_ms",
    ]
    with open(output_dir / "summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)

    grouped_map: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in runs:
        grouped_map[(row["model"], row["tag"])].append(row)
    grouped = []
    for (model, tag), rows in sorted(grouped_map.items()):
        record = {"model": model, "tag": tag, "n": len(rows), "seeds": sorted(r["seed"] for r in rows)}
        for key in (
            "dev_accuracy", "test_all_accuracy", "test_all_macro_f1",
            "test_clean_accuracy", "test_clean_macro_f1", "latency_mean_ms", "train_seconds",
        ):
            mean, std = mean_std([r[key] for r in rows])
            record[f"{key}_mean"] = round(mean, 4)
            record[f"{key}_std"] = round(std, 4)
        grouped.append(record)

    with open(output_dir / "summary_grouped.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(grouped[0].keys()))
        writer.writeheader()
        writer.writerows(grouped)

    save_json(output_dir / "summary.json", {"runs": runs, "grouped": grouped})
    print(f"[summary] runs={len(runs)} groups={len(grouped)} -> {output_dir / 'summary.csv'}", flush=True)
    for record in grouped:
        if record["tag"]:
            continue
        print(
            f"  {record['model']:<20} n={record['n']} "
            f"test_all={record['test_all_accuracy_mean']:.4f}±{record['test_all_accuracy_std']:.4f} "
            f"test_clean={record['test_clean_accuracy_mean']:.4f}±{record['test_clean_accuracy_std']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
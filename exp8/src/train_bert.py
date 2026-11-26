from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from .common import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    amp_dtype,
    configure_torch,
    count_parameters,
    env_info,
    make_run_dir,
    resolve_device,
    save_json,
    set_seed,
)
from .data import load_splits, make_loader, subset_split
from .metrics import compute_metrics, measure_latency, predict_loader, write_predictions
from .train_scratch import serializable_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微调预训练中文 BERT 做情感分类")
    parser.add_argument("--model", default="bert-base-chinese")
    parser.add_argument("--tokenizer", default=None, help="默认与 --model 相同")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", default="bf16", choices=["off", "bf16", "fp16"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--tag", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--save-ckpt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_len = min(args.max_len, 64)
        args.batch_size = 16
        args.epochs = 1
        args.tag = args.tag or "smoke"
    tokenizer_name = args.tokenizer or args.model

    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_torch(device)
    dtype = amp_dtype(args.amp) if device.type == "cuda" else None
    amp_enabled = dtype is not None

    cache_dir = Path(args.output_dir) / "cache"
    splits, stats, tokenizer = load_splits(
        Path(args.data_dir), tokenizer_name, args.max_len, cache_dir, use_cache=not args.no_cache
    )
    if args.smoke:
        splits = {
            "train": subset_split(splits["train"], 200),
            "dev": subset_split(splits["dev"], 100),
            "test_all": subset_split(splits["test_all"], 100),
            "test_clean": subset_split(splits["test_clean"], 100),
        }
    if not args.smoke:
        save_json(Path(args.output_dir) / "data_stats.json", stats)

    train_loader = make_loader(
        splits["train"], args.batch_size, True, args.num_workers, device.type == "cuda"
    )
    dev_loader = make_loader(
        splits["dev"], args.batch_size, False, args.num_workers, device.type == "cuda"
    )
    test_all_loader = make_loader(
        splits["test_all"], args.batch_size, False, args.num_workers, device.type == "cuda"
    )
    test_clean_loader = make_loader(
        splits["test_clean"], args.batch_size, False, args.num_workers, device.type == "cuda"
    )

    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    params_total, params_trainable = count_parameters(model)
    run_dir = make_run_dir(Path(args.output_dir), args.model, args.seed, args.tag)
    print(f"[run] {run_dir}", flush=True)
    print(
        f"[env] model={args.model} device={device} "
        f"gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} "
        f"params={params_total:,} trainable={params_trainable:,}",
        flush=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    history = []
    best_dev_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        correct = 0
        for input_ids, attention_mask, target in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = criterion(logits, target)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            running_loss += loss.item() * target.size(0)
            seen += target.size(0)
            correct += (logits.argmax(dim=1) == target).sum().item()

        dev_labels, dev_preds, _ = predict_loader(model, dev_loader, device, args.amp)
        dev_metrics = compute_metrics(dev_labels, dev_preds)
        train_loss = running_loss / max(1, seen)
        train_accuracy = correct / max(1, seen)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 4),
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "seconds": round(time.perf_counter() - start, 1),
        }
        history.append(row)
        print(
            f"[epoch {epoch:02d}] loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"dev_acc={dev_metrics['accuracy']:.4f} dev_f1={dev_metrics['macro_f1']:.4f}",
            flush=True,
        )

        if dev_metrics["accuracy"] > best_dev_accuracy:
            best_dev_accuracy = dev_metrics["accuracy"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            print(f"[early-stop] no improvement for {args.patience} epochs", flush=True)
            break

    train_seconds = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
        if args.save_ckpt:
            torch.save(best_state, run_dir / "best.pt")

    dev_labels, dev_preds, dev_probs = predict_loader(model, dev_loader, device, args.amp)
    dev_metrics = compute_metrics(dev_labels, dev_preds, dev_probs)

    test_all_labels, test_all_preds, test_all_probs = predict_loader(
        model, test_all_loader, device, args.amp
    )
    test_all_metrics = compute_metrics(test_all_labels, test_all_preds, test_all_probs)
    write_predictions(run_dir / "predictions_test_all.csv", splits["test_all"], test_all_preds, test_all_probs)

    test_clean_labels, test_clean_preds, test_clean_probs = predict_loader(
        model, test_clean_loader, device, args.amp
    )
    test_clean_metrics = compute_metrics(test_clean_labels, test_clean_preds, test_clean_probs)
    write_predictions(run_dir / "predictions_test_clean.csv", splits["test_clean"], test_clean_preds, test_clean_probs)

    latency = measure_latency(model, splits["test_clean"], device, args.amp, max_samples=100)

    payload = {
        "run": {
            "name": run_dir.name,
            "model": args.model,
            "seed": args.seed,
            "tag": args.tag,
            "args": serializable_args(args),
        },
        "model": {
            "params_total": params_total,
            "params_trainable": params_trainable,
        },
        "train": {
            "seconds": round(train_seconds, 2),
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_dev_accuracy": round(best_dev_accuracy, 4),
            "history": history,
        },
        "dev": dev_metrics,
        "test_all": test_all_metrics,
        "test_clean": test_clean_metrics,
        "latency_ms": latency,
        "data": stats,
    }
    save_json(run_dir / "metrics.json", payload)
    save_json(run_dir / "config.json", serializable_args(args))
    save_json(run_dir / "env.json", env_info(device))

    print(
        f"[done] dev={dev_metrics['accuracy']:.4f} test_all={test_all_metrics['accuracy']:.4f} "
        f"test_clean={test_clean_metrics['accuracy']:.4f} "
        f"latency={latency['mean_ms']:.2f}ms train={train_seconds:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
"""从零训练一个小 CNN（端到端），用于演示「训练人脸识别模型的本质」。

与 `train_head.py` 的区别：
- `train_head.py`：冻结 SFace 特征，只训练一个浅层分类器（不反向传播到图像）
- 本脚本：从像素开始训练卷积网络，完整走 前向 -> 损失 -> 反向传播 -> 更新

本机无 CUDA，只跑 CPU。数据用按身份分目录的图片（LFW / CASIA 均可）。

用法：
    python -m backend.train_small_cnn --root data/lfw/images --min-images 20 --epochs 40
    python -m backend.train_small_cnn --root data/casia/images --min-images 8 \
        --max-classes 40 --per-class 20 --epochs 30 --out outputs/cnn_casia
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

try:
    from .embedder import imread_unicode
    from .paths import new_run_dir, resolve
except ImportError:  # 允许 python backend/train_small_cnn.py 直接运行
    from embedder import imread_unicode
    from paths import new_run_dir, resolve

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _need_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit(
            "需要 PyTorch（CPU 版即可）：\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc
    return torch, nn


def load_dataset(root: Path, min_images: int, max_classes: int, per_class: int,
                 size: int, seed: int) -> tuple:
    import cv2

    index = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rel = path.relative_to(root)
            if len(rel.parts) >= 2:
                index.setdefault(rel.parts[0], []).append(path)
    # 保持身份按字母序（与 build_embeddings 的 rglob 顺序一致），便于和 SFace 对照实验对齐划分
    ids = [name for name, files in index.items() if len(files) >= min_images]
    if max_classes > 0:
        keep = sorted(ids, key=lambda name: (-len(index[name]), name))[:max_classes]
        ids = [name for name in ids if name in set(keep)]
    if len(ids) < 2:
        raise SystemExit(f"满足 >= {min_images} 张的身份只有 {len(ids)} 个")

    rng = random.Random(seed)
    images, labels, paths = [], [], []
    for name in ids:
        files = index[name]
        if per_class > 0 and len(files) > per_class:
            files = sorted(rng.sample(files, per_class))
        for path in files:
            image = imread_unicode(path)
            if image is None:
                continue
            images.append(cv2.cvtColor(
                cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            ))
            labels.append(name)
            paths.append(str(path))
    if not images:
        raise SystemExit("没有读到任何图片")
    return np.stack(images), np.asarray(labels), paths


def build_model(num_classes: int):
    torch, nn = _need_torch()

    class SmallCNN(nn.Module):
        """4 个卷积块 + 全局池化，约 0.4M 参数，CPU 可训。"""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.pool(self.features(x)).flatten(1)
            return self.fc(self.dropout(x))

    return SmallCNN()


def _batches(n: int, batch: int, shuffle: bool, rng: np.random.Generator):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch):
        yield order[start:start + batch]


def evaluate(model, x, y, batch: int, torch) -> tuple:
    model.eval()
    loss_sum, correct = 0.0, 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for index in _batches(len(x), batch, False, np.random.default_rng(0)):
            bx = torch.from_numpy(x[index]).float().div_(255.0).sub_(0.5).div_(0.5)
            by = torch.from_numpy(y[index]).long()
            logits = model(bx)
            loss_sum += float(criterion(logits, by))
            correct += int((logits.argmax(1) == by).sum())
    return loss_sum / max(len(x), 1), correct / max(len(x), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="从零训练小 CNN（CPU）")
    parser.add_argument("--root", required=True, help="按身份分目录的图片根")
    parser.add_argument("--min-images", type=int, default=20)
    parser.add_argument("--max-classes", type=int, default=0)
    parser.add_argument("--per-class", type=int, default=0, help="每类最多取多少张")
    parser.add_argument("--size", type=int, default=112)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch, nn = _need_torch()
    torch.set_num_threads(max(1, args.threads))
    root = resolve(args.root)
    out_dir = resolve(args.out) if args.out else new_run_dir("cnn")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    images, names, paths = load_dataset(
        root, args.min_images, args.max_classes, args.per_class, args.size, args.seed
    )
    images = np.ascontiguousarray(images.transpose(0, 3, 1, 2))  # NHWC -> NCHW
    classes = sorted(set(names.tolist()))
    label_of = {name: i for i, name in enumerate(classes)}
    y_all = np.asarray([label_of[n] for n in names], dtype=np.int64)
    print(f"数据: {len(images)} 张 / {len(classes)} 类  "
          f"({time.time() - started:.1f}s 读入)")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    x_train, x_test, y_train, y_test = train_test_split(
        images, y_all, test_size=args.test_size, random_state=args.seed,
        stratify=y_all,
    )
    print(f"划分: 训练 {len(x_train)}  测试 {len(x_test)}")

    model = build_model(len(classes))
    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {params/1e6:.2f}M  线程: {args.threads}  设备: cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tick = time.time()
        loss_sum, correct, seen = 0.0, 0, 0
        for index in _batches(len(x_train), args.batch, True, rng):
            bx = torch.from_numpy(x_train[index]).float().div_(255.0).sub_(0.5).div_(0.5)
            by = torch.from_numpy(y_train[index]).long()
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            correct += int((logits.argmax(1) == by).sum())
            seen += len(index)
        train_loss, train_acc = loss_sum / seen, correct / seen
        test_loss, test_acc = evaluate(model, x_test, y_test, args.batch, torch)
        seconds = time.time() - tick
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
            "test_loss": round(test_loss, 4), "test_acc": round(test_acc, 4),
            "seconds": round(seconds, 2),
        })
        print(f"ep {epoch:02d}/{args.epochs}  train {train_loss:.3f}/{train_acc:.3f}  "
              f"test {test_loss:.3f}/{test_acc:.3f}  {seconds:.1f}s")

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test).float().div_(255.0).sub_(0.5).div_(0.5))
        predictions = logits.argmax(1).numpy()
    report = classification_report(
        y_test, predictions, target_names=classes, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=np.arange(len(classes)))

    torch.save({"state_dict": model.state_dict(), "classes": classes,
                "input_size": args.size}, out_dir / "weights.pt")
    with (out_dir / "history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (out_dir / "confusion.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + classes)
        for name, row in zip(classes, matrix):
            writer.writerow([name] + row.tolist())

    metrics = {
        "config": {
            "root": str(root), "min_images": args.min_images,
            "max_classes": args.max_classes, "per_class": args.per_class,
            "size": args.size, "epochs": args.epochs, "batch": args.batch,
            "lr": args.lr, "test_size": args.test_size, "seed": args.seed,
            "threads": args.threads, "device": "cpu",
            "model": "SmallCNN 4-block (16/32/64/128)", "params": params,
        },
        "data": {"images": int(len(images)), "classes": len(classes),
                 "train": int(len(x_train)), "test": int(len(x_test))},
        "history": history,
        "final": {"test_loss": history[-1]["test_loss"],
                  "test_acc": history[-1]["test_acc"],
                  "train_acc": history[-1]["train_acc"]},
        "evaluation": {"report": report, "confusion": matrix.tolist()},
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n最终 test acc {history[-1]['test_acc']:.4f}  "
          f"train acc {history[-1]['train_acc']:.4f}  用时 {metrics['elapsed_seconds']}s")
    print(f"输出: {out_dir}")


if __name__ == "__main__":
    main()

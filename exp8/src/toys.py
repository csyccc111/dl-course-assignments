from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import DEFAULT_OUTPUT_DIR, configure_torch, save_json, set_seed


def rnn_equivalence() -> dict:
    torch.manual_seed(0)
    seq_len, in_dim, hidden_dim = 16, 3, 10
    x = torch.randn(seq_len, in_dim)
    wxh = torch.randn(in_dim, hidden_dim) * 0.5
    whh = torch.randn(hidden_dim, hidden_dim) * 0.5
    h0 = torch.randn(hidden_dim)

    manual = []
    h = h0.clone()
    for t in range(seq_len):
        h = F.relu(x[t] @ wxh + h @ whh)
        manual.append(h.clone())
    manual = torch.stack(manual)

    cell = nn.RNNCell(in_dim, hidden_dim, nonlinearity="relu")
    with torch.no_grad():
        cell.weight_ih.copy_(wxh.T)
        cell.weight_hh.copy_(whh.T)
        cell.bias_ih.zero_()
        cell.bias_hh.zero_()
        outputs = []
        h = h0.clone()
        for t in range(seq_len):
            h = cell(x[t], h)
            outputs.append(h.clone())
    outputs = torch.stack(outputs)
    return {
        "max_abs_diff": float((manual - outputs).abs().max()),
        "mean_abs_diff": float((manual - outputs).abs().mean()),
        "seq_len": seq_len,
    }


class SineRNN(nn.Module):
    def __init__(self, hidden_dim: int = 10) -> None:
        super().__init__()
        self.rnn = nn.RNNCell(1, hidden_dim, nonlinearity="relu")
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        for t in range(x.size(1)):
            hidden = self.rnn(x[:, t, :], hidden)
        return self.fc(hidden)


def sine_experiment(output_dir: Path) -> dict:
    set_seed(42)
    torch.set_num_threads(8)
    total_steps = 1000
    time = torch.arange(1, total_steps + 1, dtype=torch.float32)
    signal = torch.sin(0.01 * time) + torch.randn(total_steps) * 0.2
    k = 4
    features = torch.zeros(total_steps - k, k)
    for i in range(k):
        features[:, i] = signal[i:total_steps - k + i]
    labels = signal[k:]

    model = SineRNN(hidden_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    train_n = 600
    x_train = features[:train_n].unsqueeze(-1)
    y_train = labels[:train_n]
    losses = []
    for epoch in range(1, 1001):
        hidden = torch.zeros(x_train.size(0), 10)
        optimizer.zero_grad()
        pred = model(x_train, hidden)
        loss = criterion(pred.squeeze(-1), y_train)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    with torch.inference_mode():
        hidden = torch.zeros(features.size(0), 10)
        pred_all = model(features.unsqueeze(-1), hidden).squeeze(-1)
    train_mse = float(F.mse_loss(pred_all[:train_n], labels[:train_n]).item())
    eval_mse = float(F.mse_loss(pred_all[train_n:], labels[train_n:]).item())

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "sine_data.npz",
        time=time.numpy(),
        signal=signal.numpy(),
        labels=labels.numpy(),
        pred=pred_all.numpy(),
        losses=np.asarray(losses),
        train_n=train_n,
    )
    with open(output_dir / "sine_predictions.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "signal", "label", "pred"])
        for i in range(len(labels)):
            writer.writerow([i + k + 1, round(float(signal[i + k]), 6), round(float(labels[i]), 6), round(float(pred_all[i]), 6)])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(time.numpy(), signal.numpy(), color="#9aa0a6", linewidth=1, label="signal")
    axes[0].plot(np.arange(k + 1, total_steps + 1), pred_all.numpy(), color="#1a73e8", linewidth=1.2, label="RNN prediction")
    axes[0].axvline(k + train_n, color="#d93025", linestyle="--", linewidth=1, label="train/eval split")
    axes[0].set_title(f"Sine autoregression (eval MSE={eval_mse:.4f})")
    axes[0].set_xlabel("t")
    axes[0].legend(fontsize=8)
    axes[1].plot(losses, color="#188038", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_title("Training loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_sine.png", dpi=150)
    plt.close(fig)

    return {
        "train_mse": round(train_mse, 6),
        "eval_mse": round(eval_mse, 6),
        "final_loss": round(losses[-1], 6),
        "epochs": 1000,
        "train_windows": train_n,
        "eval_windows": int(features.size(0) - train_n),
    }


def main() -> None:
    configure_torch(torch.device("cpu"))
    output_dir = Path(DEFAULT_OUTPUT_DIR) / "toys"
    equivalence = rnn_equivalence()
    sine = sine_experiment(output_dir)
    save_json(output_dir / "toys_metrics.json", {"rnn_equivalence": equivalence, "sine": sine})
    print(f"[toys] equivalence max_abs_diff={equivalence['max_abs_diff']:.3e}", flush=True)
    print(f"[toys] sine train_mse={sine['train_mse']:.6f} eval_mse={sine['eval_mse']:.6f}", flush=True)


if __name__ == "__main__":
    main()
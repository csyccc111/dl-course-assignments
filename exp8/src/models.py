from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class _TextRNNBase(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        cell: str,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if cell == "rnn":
            self.rnn = nn.RNN(
                emb_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                dropout=rnn_dropout, nonlinearity="tanh",
            )
        else:
            self.rnn = nn.LSTM(
                emb_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                dropout=rnn_dropout,
            )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        output, _ = self.rnn(embedded)
        pooled = masked_mean(output, attention_mask)
        return self.fc(self.dropout(pooled))


class TextRNN(_TextRNNBase):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 128,
                 num_layers: int = 1, dropout: float = 0.2) -> None:
        super().__init__(vocab_size, emb_dim, hidden_dim, num_layers, dropout, "rnn")


class TextLSTM(_TextRNNBase):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 128,
                 num_layers: int = 1, dropout: float = 0.2) -> None:
        super().__init__(vocab_size, emb_dim, hidden_dim, num_layers, dropout, "lstm")


class TextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)
        padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        pooled = masked_mean(x, attention_mask)
        return self.fc(self.dropout(pooled))


def build_model(
    name: str,
    vocab_size: int,
    max_len: int,
    emb_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 2,
    nhead: int = 4,
    ff_dim: int = 256,
    dropout: float = 0.2,
) -> nn.Module:
    if name == "rnn":
        return TextRNN(vocab_size, emb_dim, hidden_dim, num_layers=1, dropout=dropout)
    if name == "lstm":
        return TextLSTM(vocab_size, emb_dim, hidden_dim, num_layers=1, dropout=dropout)
    if name == "transformer":
        return TextTransformer(vocab_size, max_len, hidden_dim, nhead, num_layers, ff_dim, dropout)
    raise SystemExit(f"未知模型: {name}")
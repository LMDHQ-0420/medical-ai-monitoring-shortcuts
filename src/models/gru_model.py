"""Dual-layer GRU model for EHR time-series classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """Two-layer GRU for binary outcome prediction from EHR time series.

    Input:  [B, T, F]  — batch × time steps × features
    Output: [B, 1]     — predicted probability after Sigmoid
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)          # h_n: [num_layers, B, hidden_size]
        last_hidden = h_n[-1]         # [B, hidden_size]
        out = self.dropout(last_hidden)
        out = self.fc(out)            # [B, 1]
        return self.sigmoid(out)

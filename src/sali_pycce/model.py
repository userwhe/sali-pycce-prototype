"""PyTorch SALI-style 1D-to-2D CNN."""

from __future__ import annotations

import torch
from torch import nn


class SignalBranch(nn.Module):
    """A compact 1D CNN branch for one CPMG trace."""

    def __init__(self, pooled_len: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            nn.AdaptiveAvgPool1d(pooled_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, points]
        x = x.unsqueeze(1)
        x = self.net(x)
        return torch.flatten(x, start_dim=1)


class SALINet(nn.Module):
    """Two-input signal-to-image network.

    Output shape is ``[batch, 1, 32, 64]`` by default.
    """

    def __init__(self, n_inputs: int = 2, output_shape: tuple[int, int] = (32, 64)) -> None:
        super().__init__()
        if output_shape != (32, 64):
            raise ValueError("This compact prototype currently expects output_shape=(32, 64).")
        self.n_inputs = n_inputs
        self.output_shape = output_shape
        self.branches = nn.ModuleList([SignalBranch() for _ in range(n_inputs)])
        branch_dim = 16 * 32
        self.fc = nn.Sequential(
            nn.Linear(n_inputs * branch_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 16 * 4 * 8),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # 8 x 16
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),  # 16 x 32
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1),  # 32 x 64
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        signals:
            Tensor of shape ``[batch, n_inputs, n_points]``.
        """
        if signals.ndim != 3:
            raise ValueError(f"Expected [batch, n_inputs, points], got {tuple(signals.shape)}")
        if signals.shape[1] != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} inputs, got {signals.shape[1]}")
        features = [branch(signals[:, i, :]) for i, branch in enumerate(self.branches)]
        z = torch.cat(features, dim=1)
        z = self.fc(z)
        z = z.reshape(signals.shape[0], 16, 4, 8)
        return self.decoder(z)

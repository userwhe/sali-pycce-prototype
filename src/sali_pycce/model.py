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

    Output shape is ``[batch, 1, 128, 256]`` by default.
    """

    def __init__(
        self,
        n_inputs: int = 2,
        output_shape: tuple[int, int] = (128, 256),
        pooled_len: int = 64,
        decoder_channels: int = 32,
    ) -> None:
        super().__init__()
        height, width = output_shape
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("output_shape height and width must be divisible by 8.")
        self.n_inputs = int(n_inputs)
        self.output_shape = (int(height), int(width))
        self.base_shape = (self.output_shape[0] // 8, self.output_shape[1] // 8)
        self.decoder_channels = int(decoder_channels)
        self.branches = nn.ModuleList([SignalBranch(pooled_len=pooled_len) for _ in range(self.n_inputs)])
        branch_dim = 16 * int(pooled_len)
        self.fc = nn.Sequential(
            nn.Linear(self.n_inputs * branch_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, self.decoder_channels * self.base_shape[0] * self.base_shape[1]),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.decoder_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=3, padding=1),
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
        z = z.reshape(signals.shape[0], self.decoder_channels, self.base_shape[0], self.base_shape[1])
        return self.decoder(z)

from __future__ import annotations

import torch
from torch import nn

from sali.config import ModelConfig


class SignalBranch(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, cfg.conv1_filters, kernel_size=3, padding=1),
            nn.Conv1d(cfg.conv1_filters, cfg.conv2_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(cfg.conv2_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return torch.flatten(x, start_dim=1)


class SaliNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self._validate_config(cfg)
        self.cfg = cfg
        self.branch32 = SignalBranch(cfg)
        self.branch256 = SignalBranch(cfg)
        branch_features = cfg.conv2_filters * (cfg.signal_length // 2)
        merged_features = branch_features * 2
        bottleneck = (cfg.dense_height * cfg.dense_width) // 2
        dense_size = cfg.dense_height * cfg.dense_width
        self.fc = nn.Sequential(
            nn.Linear(merged_features, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(bottleneck, dense_size),
            nn.BatchNorm1d(dense_size),
            nn.ReLU(),
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(cfg.dropout),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _validate_config(cfg: ModelConfig) -> None:
        expected_height = cfg.dense_height * 2
        expected_width = cfg.dense_width * 2
        if cfg.output_height != expected_height:
            raise ValueError(
                f"output_height must equal dense_height * 2; "
                f"got {cfg.output_height}, expected {expected_height}"
            )
        if cfg.output_width != expected_width:
            raise ValueError(
                f"output_width must equal dense_width * 2; "
                f"got {cfg.output_width}, expected {expected_width}"
            )

    def _validate_signal(self, name: str, signal: torch.Tensor) -> None:
        if signal.ndim != 3:
            raise ValueError(
                f"{name} must be rank 3 with shape "
                f"(batch, 1, {self.cfg.signal_length}); got {tuple(signal.shape)}"
            )
        if signal.shape[1] != 1:
            raise ValueError(f"{name} must have channel count 1; got {signal.shape[1]}")
        if signal.shape[2] != self.cfg.signal_length:
            raise ValueError(
                f"{name} must have length {self.cfg.signal_length}; got {signal.shape[2]}"
            )

    def forward(self, signal32: torch.Tensor, signal256: torch.Tensor) -> torch.Tensor:
        self._validate_signal("signal32", signal32)
        self._validate_signal("signal256", signal256)
        if signal32.shape[0] != signal256.shape[0]:
            raise ValueError(
                f"signal32 and signal256 must have matching batch size; "
                f"got {signal32.shape[0]} and {signal256.shape[0]}"
            )
        x32 = self.branch32(signal32)
        x256 = self.branch256(signal256)
        merged = torch.cat([x32, x256], dim=1)
        x = self.fc(merged)
        x = x.view(-1, 1, self.cfg.dense_height, self.cfg.dense_width)
        out = self.conv2d(x)
        expected = (self.cfg.output_height, self.cfg.output_width)
        if out.shape[-2:] != expected:
            raise RuntimeError(f"model produced {out.shape[-2:]}, expected {expected}")
        return out

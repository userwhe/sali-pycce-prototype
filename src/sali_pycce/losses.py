"""Loss functions for sparse SALI heatmap training."""

from __future__ import annotations

import torch
from torch import nn


class WeightedBCEDiceLoss(nn.Module):
    """Detection-oriented loss for sparse spin heatmaps.

    Plain MSE makes all-background predictions attractive when only a few pixels
    contain spin signal. This loss upweights the heatmap target and adds a soft
    Dice term so training is rewarded for putting mass on spin blobs.
    """

    def __init__(self, pos_weight: float = 5.0, dice_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        if pos_weight <= 0:
            raise ValueError("pos_weight must be positive")
        if dice_weight < 0:
            raise ValueError("dice_weight must be non-negative")
        self.pos_weight = float(pos_weight)
        self.dice_weight = float(dice_weight)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(self.eps, 1.0 - self.eps)
        target = target.to(dtype=pred.dtype)

        reduce_dims = tuple(range(1, pred.ndim))
        positive_mass = target.sum(dim=reduce_dims)
        negative_weight = 1.0 - target
        positive_loss = -(target * torch.log(pred)).sum(dim=reduce_dims) / positive_mass.clamp_min(
            self.eps
        )
        negative_loss = -(
            negative_weight * torch.log(1.0 - pred)
        ).sum(dim=reduce_dims) / negative_weight.sum(dim=reduce_dims).clamp_min(self.eps)
        has_positive = (positive_mass > self.eps).to(dtype=pred.dtype)
        bce = (self.pos_weight * positive_loss * has_positive + negative_loss).mean()
        if self.dice_weight == 0.0:
            return bce

        intersection = (pred * target).sum(dim=reduce_dims)
        denominator = pred.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
        dice = 1.0 - (2.0 * intersection + self.eps) / (denominator + self.eps)
        return bce + self.dice_weight * dice.mean()


class WeightedMSELoss(nn.Module):
    """MSE with extra weight on nonzero heatmap target pixels."""

    def __init__(self, pos_weight: float = 200.0) -> None:
        super().__init__()
        if pos_weight < 0:
            raise ValueError("pos_weight must be non-negative")
        self.pos_weight = float(pos_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=pred.dtype)
        weight = 1.0 + self.pos_weight * target
        return (weight * (pred - target).square()).mean()


def make_heatmap_loss(name: str, pos_weight: float = 200.0, dice_weight: float = 1.0) -> nn.Module:
    """Build a heatmap loss by CLI name."""

    normalized = name.strip().lower()
    if normalized == "weighted-mse":
        return WeightedMSELoss(pos_weight=pos_weight)
    if normalized == "weighted-bce-dice":
        return WeightedBCEDiceLoss(pos_weight=pos_weight, dice_weight=dice_weight)
    if normalized == "weighted-bce":
        return WeightedBCEDiceLoss(pos_weight=pos_weight, dice_weight=0.0)
    if normalized == "mse":
        return nn.MSELoss()
    raise ValueError(f"unknown heatmap loss: {name}")

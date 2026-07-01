from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from sali.config import RunConfig
from sali.data import DataSplits, SaliDataset, Sample
from sali.metrics import SampleMetrics, evaluate_sample
from sali.model import SaliNet
from sali.postprocess import postprocess_heatmap


@dataclass(slots=True)
class TrainResult:
    model: SaliNet
    history: dict[str, list[float]]
    best_checkpoint: Path


class HeatmapLoss(nn.Module):
    def __init__(
        self,
        loss_type: str,
        positive_weight: float,
        border_penalty_weight: float,
        border_width: int,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.positive_weight = positive_weight
        self.border_penalty_weight = border_penalty_weight
        self.border_width = border_width

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            loss = torch.mean((prediction - target) ** 2)
        elif self.loss_type == "weighted_mse":
            weights = 1.0 + (self.positive_weight * target)
            loss = torch.mean(weights * ((prediction - target) ** 2))
        elif self.loss_type == "weighted_bce":
            weights = 1.0 + (self.positive_weight * target)
            clipped = torch.clamp(prediction, 1e-6, 1.0 - 1e-6)
            loss = torch.mean(weights * F.binary_cross_entropy(clipped, target, reduction="none"))
        else:
            raise ValueError("loss_type must be one of: mse, weighted_mse, weighted_bce")
        if self.border_penalty_weight > 0.0:
            loss = loss + (self.border_penalty_weight * _border_penalty(prediction, self.border_width))
        return loss


def _border_penalty(prediction: torch.Tensor, border_width: int) -> torch.Tensor:
    if border_width < 1:
        raise ValueError("border_width must be at least 1 when border penalty is enabled")
    height, width = prediction.shape[-2:]
    if border_width * 2 >= min(height, width):
        raise ValueError("border_width is too large for the heatmap dimensions")
    border = torch.zeros_like(prediction, dtype=torch.bool)
    border[..., :border_width, :] = True
    border[..., -border_width:, :] = True
    border[..., :, :border_width] = True
    border[..., :, -border_width:] = True
    return torch.mean(prediction[border] ** 2)


def make_heatmap_loss(cfg) -> nn.Module:
    if cfg.loss_type not in {"mse", "weighted_mse", "weighted_bce"}:
        raise ValueError("loss_type must be one of: mse, weighted_mse, weighted_bce")
    if not np.isfinite(cfg.positive_weight) or cfg.positive_weight < 0.0:
        raise ValueError("positive_weight must be finite and non-negative")
    if not np.isfinite(cfg.border_penalty_weight) or cfg.border_penalty_weight < 0.0:
        raise ValueError("border_penalty_weight must be finite and non-negative")
    if cfg.border_width < 1:
        raise ValueError("border_width must be at least 1")
    return HeatmapLoss(
        loss_type=cfg.loss_type,
        positive_weight=float(cfg.positive_weight),
        border_penalty_weight=float(cfg.border_penalty_weight),
        border_width=int(cfg.border_width),
    )


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _run_epoch(
    model: SaliNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    for signal32, signal256, target in loader:
        signal32 = signal32.to(device)
        signal256 = signal256.to(device)
        target = target.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            pred = model(signal32, signal256)
            loss = criterion(pred, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("loss became non-finite")
            if is_train:
                loss.backward()
                optimizer.step()
        batch_size = signal32.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def _save_history(path: Path, history: dict[str, list[float]]) -> None:
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _load_state_dict_to_device(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    model.load_state_dict({name: tensor.to(device) for name, tensor in state.items()})


def _drops_singleton_final_batch(sample_count: int, batch_size: int) -> bool:
    return sample_count > batch_size and sample_count % batch_size == 1


def _effective_train_batch_count(sample_count: int, batch_size: int, drop_last: bool) -> int:
    if drop_last:
        return sample_count // batch_size
    return (sample_count + batch_size - 1) // batch_size


def _validate_training_batches(sample_count: int, batch_size: int) -> bool:
    if batch_size < 2:
        raise ValueError("training batch_size must be at least 2 for BatchNorm")
    if sample_count < 2:
        raise ValueError("training split must contain at least 2 samples for BatchNorm")
    drop_last = _drops_singleton_final_batch(sample_count, batch_size)
    if _effective_train_batch_count(sample_count, batch_size, drop_last) <= 0:
        raise ValueError("effective training batch count must be greater than 0")
    return drop_last


def train_model(cfg: RunConfig, splits: DataSplits) -> TrainResult:
    drop_last = _validate_training_batches(len(splits.train), cfg.training.batch_size)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg.training.device)
    _set_torch_seed(cfg.data.seed)
    model = SaliNet(cfg.model).to(device)
    train_loader = DataLoader(
        SaliDataset(splits.train),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=_seeded_generator(cfg.data.seed),
    )
    val_loader = DataLoader(
        SaliDataset(splits.val),
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )
    criterion = make_heatmap_loss(cfg.training)
    optimizer = Adam(model.parameters(), lr=cfg.training.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.lr_reduction_factor,
        patience=cfg.training.lr_plateau_patience,
        min_lr=1e-8,
    )
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_loss = float("inf")
    early_stopping_loss = float("inf")
    best_state = _cpu_state_dict(model)
    best_checkpoint = cfg.output_dir / "best_model.pt"
    stale_epochs = 0
    for _epoch in range(cfg.training.max_epochs):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device, None)
        scheduler.step(val_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["lr"].append(lr)
        if val_loss < best_loss:
            best_loss = float(val_loss)
            best_state = _cpu_state_dict(model)
            torch.save(best_state, best_checkpoint)
        if val_loss < early_stopping_loss - cfg.training.min_delta:
            early_stopping_loss = float(val_loss)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= cfg.training.early_stopping_patience:
            break
    _load_state_dict_to_device(model, best_state, device)
    if not best_checkpoint.exists():
        torch.save(best_state, best_checkpoint)
    _save_history(cfg.output_dir / "history.json", history)
    return TrainResult(model=model, history=history, best_checkpoint=best_checkpoint)


def evaluate_model(
    model: SaliNet,
    cfg: RunConfig,
    samples: list[Sample],
    max_samples: int | None = None,
) -> list[SampleMetrics]:
    device = choose_device(cfg.training.device)
    model.to(device)
    model.eval()
    rng = np.random.default_rng(cfg.data.seed + 999)
    selected = samples if max_samples is None else samples[:max_samples]
    results: list[SampleMetrics] = []
    for sample in selected:
        signal32 = torch.from_numpy(sample.signals[0:1]).unsqueeze(0).to(device)
        signal256 = torch.from_numpy(sample.signals[1:2]).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(signal32, signal256).cpu().numpy()[0]
        predicted_nuclei = postprocess_heatmap(prediction, cfg.data, cfg.model, cfg.postprocess)
        results.append(
            evaluate_sample(
                predicted_nuclei,
                sample.nuclei,
                sample.raw_signals,
                cfg.data,
                cfg.model,
                cfg.physics,
                rng,
            )
        )
    return results

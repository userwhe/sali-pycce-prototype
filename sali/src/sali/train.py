from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from sali.config import RunConfig
from sali.data import DataSplits, SaliDataset
from sali.model import SaliNet


@dataclass(slots=True)
class TrainResult:
    model: SaliNet
    history: dict[str, list[float]]
    best_checkpoint: Path


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
    criterion = nn.MSELoss()
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

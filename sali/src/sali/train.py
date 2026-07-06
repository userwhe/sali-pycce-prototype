from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from sali.config import RunConfig
from sali.data import (
    DataSplits,
    NormalizationStats,
    SaliDataset,
    Sample,
    StreamedSaliDataset,
)
from sali.metrics import SampleMetrics, evaluate_sample
from sali.model import SaliNet
from sali.postprocess import postprocess_heatmap
from sali.shards import ShardedSaliDataset, load_shard_manifest


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


def _save_history_csv(path: Path, history: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "val_loss", "lr"]
    row_count = max((len(values) for values in history.values()), default=0)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(row_count):
            writer.writerow(
                {
                    "epoch": index + 1,
                    "train_loss": history.get("train_loss", [])[index],
                    "val_loss": history.get("val_loss", [])[index],
                    "lr": history.get("lr", [])[index],
                }
            )


def _config_snapshot(cfg: RunConfig) -> dict[str, object]:
    snapshot = asdict(cfg)
    snapshot["output_dir"] = str(cfg.output_dir)
    return snapshot


def _stats_snapshot(stats: NormalizationStats) -> dict[str, float]:
    return {"mean": float(stats.mean), "var": float(stats.var), "epsilon": float(stats.epsilon)}


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


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


def _training_samples_per_epoch(cfg: RunConfig, total_samples: int) -> int:
    samples_per_epoch = cfg.training.samples_per_epoch
    if samples_per_epoch is None:
        return total_samples
    if samples_per_epoch < 2:
        raise ValueError("samples_per_epoch must be at least 2 for BatchNorm")
    return min(total_samples, int(samples_per_epoch))


def _materialized_train_dataset_for_epoch(
    samples: list[Sample],
    *,
    sample_count: int,
    seed: int,
    epoch: int,
) -> SaliDataset | Subset:
    dataset = SaliDataset(samples)
    if sample_count >= len(samples):
        return dataset
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    indices = rng.permutation(len(samples))[:sample_count].astype(int).tolist()
    return Subset(dataset, indices)


def _save_full_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: Adam,
    scheduler: ReduceLROnPlateau,
    best_loss: float,
    early_stopping_loss: float,
    stale_epochs: int,
    history: dict[str, list[float]],
    cfg: RunConfig,
    stats: NormalizationStats,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": _cpu_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_loss": float(best_loss),
            "early_stopping_loss": float(early_stopping_loss),
            "stale_epochs": int(stale_epochs),
            "history": history,
            "config": _config_snapshot(cfg),
            "normalization_stats": _stats_snapshot(stats),
        },
        path,
    )


def _load_full_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Adam,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint


def _resolve_resume_checkpoint(output_dir: Path, resume_from: Path | str) -> Path:
    if str(resume_from) == "latest":
        return output_dir / "checkpoints" / "latest.pt"
    return Path(resume_from)


def _write_run_state(
    output_dir: Path,
    *,
    epoch: int,
    best_loss: float,
    stale_epochs: int,
    latest_checkpoint: Path,
    best_checkpoint: Path,
) -> None:
    _save_json(
        output_dir / "run_state.json",
        {
            "epoch": int(epoch),
            "best_loss": float(best_loss),
            "stale_epochs": int(stale_epochs),
            "latest_checkpoint": str(latest_checkpoint),
            "best_checkpoint": str(best_checkpoint),
        },
    )


def train_model(cfg: RunConfig, splits: DataSplits) -> TrainResult:
    train_sample_count = _training_samples_per_epoch(cfg, len(splits.train))
    drop_last = _validate_training_batches(train_sample_count, cfg.training.batch_size)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg.training.device)
    _set_torch_seed(cfg.data.seed)
    model = SaliNet(cfg.model).to(device)
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
    for epoch in range(1, cfg.training.max_epochs + 1):
        train_dataset = _materialized_train_dataset_for_epoch(
            splits.train,
            sample_count=train_sample_count,
            seed=cfg.data.seed,
            epoch=epoch,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=train_sample_count >= len(splits.train),
            drop_last=drop_last,
            generator=_seeded_generator(cfg.data.seed + epoch),
        )
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
    _save_history_csv(cfg.output_dir / "history.csv", history)
    return TrainResult(model=model, history=history, best_checkpoint=best_checkpoint)


def train_streamed_model(
    cfg: RunConfig,
    stats: NormalizationStats,
    *,
    checkpoint_every_epochs: int = 1,
    resume_from: Path | str | None = None,
    num_workers: int = 0,
    epoch_callback: Callable[[int, SaliNet, dict[str, list[float]]], None] | None = None,
) -> TrainResult:
    if checkpoint_every_epochs < 1:
        raise ValueError("checkpoint_every_epochs must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    train_sample_count = _training_samples_per_epoch(cfg, cfg.data.train_samples)
    drop_last = _validate_training_batches(train_sample_count, cfg.training.batch_size)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = cfg.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg.training.device)
    _set_torch_seed(cfg.data.seed)
    model = SaliNet(cfg.model).to(device)
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
    best_checkpoint = cfg.output_dir / "best_model.pt"
    best_state = _cpu_state_dict(model)
    stale_epochs = 0
    start_epoch = 1
    if resume_from is not None:
        resume_path = _resolve_resume_checkpoint(cfg.output_dir, resume_from)
        checkpoint = _load_full_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        history = {
            key: [float(value) for value in values]
            for key, values in dict(checkpoint["history"]).items()
        }
        best_loss = float(checkpoint["best_loss"])
        early_stopping_loss = float(checkpoint["early_stopping_loss"])
        stale_epochs = int(checkpoint["stale_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if best_checkpoint.exists():
            best_state = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
        else:
            best_state = _cpu_state_dict(model)

    val_loader = DataLoader(
        StreamedSaliDataset(cfg, "val", stats),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    latest_checkpoint = checkpoint_dir / "latest.pt"
    for epoch in range(start_epoch, cfg.training.max_epochs + 1):
        train_loader = DataLoader(
            StreamedSaliDataset(
                cfg,
                "train",
                stats,
                max_samples=train_sample_count,
                epoch=epoch,
                shuffle=True,
            ),
            batch_size=cfg.training.batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=num_workers,
        )
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

        _save_history(cfg.output_dir / "history.json", history)
        _save_history_csv(cfg.output_dir / "history.csv", history)
        _save_full_checkpoint(
            latest_checkpoint,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_loss=best_loss,
            early_stopping_loss=early_stopping_loss,
            stale_epochs=stale_epochs,
            history=history,
            cfg=cfg,
            stats=stats,
        )
        if epoch % checkpoint_every_epochs == 0:
            _save_full_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_loss=best_loss,
                early_stopping_loss=early_stopping_loss,
                stale_epochs=stale_epochs,
                history=history,
                cfg=cfg,
                stats=stats,
            )
        _write_run_state(
            cfg.output_dir,
            epoch=epoch,
            best_loss=best_loss,
            stale_epochs=stale_epochs,
            latest_checkpoint=latest_checkpoint,
            best_checkpoint=best_checkpoint,
        )
        if epoch_callback is not None:
            epoch_callback(epoch, model, history)
        if stale_epochs >= cfg.training.early_stopping_patience:
            break

    _load_state_dict_to_device(model, best_state, device)
    if not best_checkpoint.exists():
        torch.save(best_state, best_checkpoint)
    return TrainResult(model=model, history=history, best_checkpoint=best_checkpoint)


def train_sharded_model(
    cfg: RunConfig,
    dataset_dir: Path,
    *,
    checkpoint_every_epochs: int = 1,
    resume_from: Path | str | None = None,
    num_workers: int = 0,
    epoch_callback: Callable[[int, SaliNet, dict[str, list[float]]], None] | None = None,
) -> TrainResult:
    if checkpoint_every_epochs < 1:
        raise ValueError("checkpoint_every_epochs must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    manifest = load_shard_manifest(dataset_dir, cfg)
    stats = manifest.normalization_stats
    train_sample_count = _training_samples_per_epoch(cfg, cfg.data.train_samples)
    drop_last = _validate_training_batches(train_sample_count, cfg.training.batch_size)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = cfg.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg.training.device)
    _set_torch_seed(cfg.data.seed)
    model = SaliNet(cfg.model).to(device)
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
    best_checkpoint = cfg.output_dir / "best_model.pt"
    best_state = _cpu_state_dict(model)
    stale_epochs = 0
    start_epoch = 1
    if resume_from is not None:
        resume_path = _resolve_resume_checkpoint(cfg.output_dir, resume_from)
        checkpoint = _load_full_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        history = {
            key: [float(value) for value in values]
            for key, values in dict(checkpoint["history"]).items()
        }
        best_loss = float(checkpoint["best_loss"])
        early_stopping_loss = float(checkpoint["early_stopping_loss"])
        stale_epochs = int(checkpoint["stale_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if best_checkpoint.exists():
            best_state = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
        else:
            best_state = _cpu_state_dict(model)

    val_loader = DataLoader(
        ShardedSaliDataset(cfg, dataset_dir, "val"),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    latest_checkpoint = checkpoint_dir / "latest.pt"
    for epoch in range(start_epoch, cfg.training.max_epochs + 1):
        train_loader = DataLoader(
            ShardedSaliDataset(
                cfg,
                dataset_dir,
                "train",
                max_samples=train_sample_count,
                epoch=epoch,
                shuffle=True,
            ),
            batch_size=cfg.training.batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=num_workers,
        )
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

        _save_history(cfg.output_dir / "history.json", history)
        _save_history_csv(cfg.output_dir / "history.csv", history)
        _save_full_checkpoint(
            latest_checkpoint,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_loss=best_loss,
            early_stopping_loss=early_stopping_loss,
            stale_epochs=stale_epochs,
            history=history,
            cfg=cfg,
            stats=stats,
        )
        if epoch % checkpoint_every_epochs == 0:
            _save_full_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_loss=best_loss,
                early_stopping_loss=early_stopping_loss,
                stale_epochs=stale_epochs,
                history=history,
                cfg=cfg,
                stats=stats,
            )
        _write_run_state(
            cfg.output_dir,
            epoch=epoch,
            best_loss=best_loss,
            stale_epochs=stale_epochs,
            latest_checkpoint=latest_checkpoint,
            best_checkpoint=best_checkpoint,
        )
        if epoch_callback is not None:
            epoch_callback(epoch, model, history)
        if stale_epochs >= cfg.training.early_stopping_patience:
            break

    _load_state_dict_to_device(model, best_state, device)
    if not best_checkpoint.exists():
        torch.save(best_state, best_checkpoint)
    return TrainResult(model=model, history=history, best_checkpoint=best_checkpoint)


def evaluate_sample_iterable(
    model: SaliNet,
    cfg: RunConfig,
    samples: Iterable[Sample],
    max_samples: int | None = None,
) -> list[SampleMetrics]:
    device = choose_device(cfg.training.device)
    model.to(device)
    model.eval()
    rng = np.random.default_rng(cfg.data.seed + 999)
    selected = samples if max_samples is None else islice(samples, max_samples)
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


def evaluate_model(
    model: SaliNet,
    cfg: RunConfig,
    samples: list[Sample],
    max_samples: int | None = None,
) -> list[SampleMetrics]:
    selected = samples if max_samples is None else samples[:max_samples]
    return evaluate_sample_iterable(model, cfg, selected)

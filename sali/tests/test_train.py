from __future__ import annotations

import copy
import json

import pytest
import torch
from torch import nn

import sali.train as train_module
from sali.config import TrainingConfig
from sali.data import DataSplits, generate_splits
from sali.train import choose_device, make_heatmap_loss, train_model


def test_choose_device_accepts_cpu() -> None:
    assert str(choose_device("cpu")) == "cpu"


def test_weighted_mse_loss_upweights_positive_target_pixels() -> None:
    cfg = copy.deepcopy(default_training_config := TrainingConfig())
    cfg.loss_type = "weighted_mse"
    cfg.positive_weight = 9.0
    criterion = make_heatmap_loss(cfg)

    positive_target = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    positive_target[0, 0, 0, 0] = 1.0
    positive_prediction = torch.zeros_like(positive_target)
    background_target = torch.zeros_like(positive_target)
    background_prediction = torch.zeros_like(background_target)
    background_prediction[0, 0, 1, 1] = 1.0

    positive_loss = criterion(positive_prediction, positive_target)
    background_loss = criterion(background_prediction, background_target)

    assert positive_loss > background_loss * 5.0
    assert default_training_config.loss_type == "mse"


def test_border_penalty_adds_cost_for_border_predictions() -> None:
    cfg = TrainingConfig(loss_type="mse", border_penalty_weight=2.0, border_width=1)
    criterion = make_heatmap_loss(cfg)
    target = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    border_prediction = torch.zeros_like(target)
    interior_prediction = torch.zeros_like(target)
    border_prediction[0, 0, 0, 1] = 1.0
    interior_prediction[0, 0, 2, 2] = 1.0

    assert criterion(border_prediction, target) > criterion(interior_prediction, target)


def test_make_heatmap_loss_rejects_unknown_loss_type() -> None:
    cfg = TrainingConfig(loss_type="mystery")

    with pytest.raises(ValueError, match="loss_type"):
        make_heatmap_loss(cfg)


def test_train_model_smoke(tiny_config, tmp_path) -> None:
    tiny_config.output_dir = tmp_path / "run"
    splits, _stats = generate_splits(tiny_config)
    result = train_model(tiny_config, splits)
    history_path = tiny_config.output_dir / "history.json"
    checkpoint = torch.load(result.best_checkpoint, map_location="cpu", weights_only=True)

    assert result.best_checkpoint.exists()
    assert history_path.exists()
    assert json.loads(history_path.read_text(encoding="utf-8")) == result.history
    assert checkpoint
    assert all(tensor.device.type == "cpu" for tensor in checkpoint.values())
    assert len(result.history["train_loss"]) == 1
    assert len(result.history["val_loss"]) == 1


def test_train_model_rejects_training_batch_size_one(tiny_config, tmp_path) -> None:
    tiny_config.output_dir = tmp_path / "batch-size-one"
    tiny_config.training.batch_size = 1
    splits, _stats = generate_splits(tiny_config)

    with pytest.raises(ValueError, match="batch_size.*at least 2"):
        train_model(tiny_config, splits)


def test_train_model_rejects_single_item_training_split(tiny_config, tmp_path) -> None:
    tiny_config.output_dir = tmp_path / "single-train-sample"
    tiny_config.data.train_samples = 1
    splits, _stats = generate_splits(tiny_config)

    with pytest.raises(ValueError, match="training split.*at least 2"):
        train_model(tiny_config, splits)


def test_train_model_drops_singleton_final_training_batch(tiny_config, tmp_path) -> None:
    tiny_config.output_dir = tmp_path / "singleton-run"
    tiny_config.data.train_samples = 5
    tiny_config.training.batch_size = 4
    splits, _stats = generate_splits(tiny_config)

    result = train_model(tiny_config, splits)

    assert result.best_checkpoint.exists()
    assert len(result.history["train_loss"]) == 1


def test_best_checkpoint_tracks_lowest_validation_loss_ignoring_min_delta(
    tiny_config, tmp_path, monkeypatch
) -> None:
    class TinyNet(nn.Module):
        def __init__(self, _cfg) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.0]))

    current_epoch = 0
    val_losses = {1: 1.0, 2: 0.95}

    def fake_run_epoch(model, _loader, _criterion, _device, optimizer) -> float:
        nonlocal current_epoch
        if optimizer is not None:
            current_epoch += 1
            with torch.no_grad():
                model.weight.fill_(float(current_epoch))
            return 0.0
        return val_losses[current_epoch]

    tiny_config.output_dir = tmp_path / "best-checkpoint"
    tiny_config.training.batch_size = 2
    tiny_config.training.max_epochs = 2
    tiny_config.training.min_delta = 0.1
    tiny_config.training.early_stopping_patience = 10
    splits = DataSplits(train=[object(), object()], val=[object(), object()], test=[])
    monkeypatch.setattr(train_module, "SaliNet", TinyNet)
    monkeypatch.setattr(train_module, "_run_epoch", fake_run_epoch)

    result = train_model(tiny_config, splits)
    checkpoint = torch.load(result.best_checkpoint, map_location="cpu", weights_only=True)

    assert result.model.weight.item() == 2.0
    assert checkpoint["weight"].item() == 2.0


def test_cpu_state_dict_returns_detached_cpu_copies() -> None:
    model = nn.Linear(2, 1)

    state = train_module._cpu_state_dict(model)
    with torch.no_grad():
        model.weight.add_(1.0)

    assert state
    assert all(tensor.device.type == "cpu" for tensor in state.values())
    assert all(not tensor.requires_grad for tensor in state.values())
    assert not torch.equal(state["weight"], model.state_dict()["weight"])


def test_train_model_is_reproducible_for_same_seed(tiny_config, tmp_path) -> None:
    first_cfg = copy.deepcopy(tiny_config)
    second_cfg = copy.deepcopy(tiny_config)
    first_cfg.output_dir = tmp_path / "first"
    second_cfg.output_dir = tmp_path / "second"
    first_splits, _first_stats = generate_splits(first_cfg)
    second_splits, _second_stats = generate_splits(second_cfg)

    first = train_model(first_cfg, first_splits)
    second = train_model(second_cfg, second_splits)

    assert first.history == second.history

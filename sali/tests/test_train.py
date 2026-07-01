from __future__ import annotations

from sali.data import generate_splits
from sali.train import choose_device, train_model


def test_choose_device_accepts_cpu() -> None:
    assert str(choose_device("cpu")) == "cpu"


def test_train_model_smoke(tiny_config, tmp_path) -> None:
    tiny_config.output_dir = tmp_path / "run"
    splits, _stats = generate_splits(tiny_config)
    result = train_model(tiny_config, splits)
    assert result.best_checkpoint.exists()
    assert len(result.history["train_loss"]) == 1
    assert len(result.history["val_loss"]) == 1


def test_train_model_drops_singleton_final_training_batch(tiny_config, tmp_path) -> None:
    tiny_config.output_dir = tmp_path / "singleton-run"
    tiny_config.data.train_samples = 5
    tiny_config.training.batch_size = 4
    splits, _stats = generate_splits(tiny_config)

    result = train_model(tiny_config, splits)

    assert result.best_checkpoint.exists()
    assert len(result.history["train_loss"]) == 1

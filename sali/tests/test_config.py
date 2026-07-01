from __future__ import annotations

import pytest

from sali.config import FieldConfig, paper_config, practical_config


def test_practical_config_uses_low_field_value() -> None:
    cfg = practical_config(field="low")
    assert cfg.physics.bz_tesla == pytest.approx(0.0056)
    assert cfg.data.train_samples == 2048
    assert cfg.data.val_samples == 512
    assert cfg.data.test_samples == 512
    assert cfg.model.output_height == 204
    assert cfg.model.output_width == 104


def test_paper_config_matches_reported_training_recipe() -> None:
    cfg = paper_config(field="high")
    assert cfg.physics.bz_tesla == pytest.approx(0.056)
    assert cfg.data.total_samples == 3_600_000
    assert cfg.data.train_samples == 2_520_000
    assert cfg.data.val_samples == 540_000
    assert cfg.data.test_samples == 540_000
    assert cfg.training.batch_size == 64
    assert cfg.training.max_epochs == 250
    assert cfg.training.learning_rate == pytest.approx(0.001)
    assert cfg.training.lr_reduction_factor == pytest.approx(0.7)
    assert cfg.training.lr_plateau_patience == 5
    assert cfg.training.early_stopping_patience == 20


def test_invalid_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="field must be"):
        FieldConfig.from_name("medium")

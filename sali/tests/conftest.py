from __future__ import annotations

import numpy as np
import pytest

from sali.config import RunConfig, practical_config


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture()
def tiny_config() -> RunConfig:
    cfg = practical_config(field="low")
    cfg.data.train_samples = 8
    cfg.data.val_samples = 4
    cfg.data.test_samples = 4
    cfg.data.max_nuclei = 3
    cfg.training.batch_size = 4
    cfg.training.max_epochs = 1
    cfg.training.early_stopping_patience = 2
    cfg.training.lr_plateau_patience = 1
    return cfg

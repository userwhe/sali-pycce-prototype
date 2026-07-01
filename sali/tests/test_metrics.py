from __future__ import annotations

import numpy as np

from sali.config import DataConfig, ModelConfig, PhysicsConfig
from sali.metrics import box_iou, evaluate_sample
from sali.physics import Couplings, generate_sample_signals
from sali.postprocess import Prediction
from sali.targets import Box, true_boxes


def test_box_iou_detects_overlap() -> None:
    a = Box(0, 0, 4, 4)
    b = Box(2, 2, 6, 6)
    assert box_iou(a, b) > 0.0
    c = Box(10, 10, 12, 12)
    assert box_iou(a, c) == 0.0


def test_evaluate_sample_counts_tp_fp_fn(rng: np.random.Generator) -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    physics = PhysicsConfig(bz_tesla=0.0056, add_shot_noise=False)
    nuclei = Couplings(
        az_khz=np.array([0.0], dtype=np.float32),
        aperp_khz=np.array([52.0], dtype=np.float32),
    )
    box = true_boxes(nuclei, data, model)[0]
    predictions = [
        Prediction(az_khz=0.5, aperp_khz=52.5, row=102.0, col=52.0, box=box, confidence=0.9)
    ]
    signals = generate_sample_signals(nuclei, physics, rng)
    result = evaluate_sample(predictions, nuclei, signals, data, model, physics, rng)
    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.mae_az_khz >= 0.0
    assert result.mae_aperp_khz >= 0.0

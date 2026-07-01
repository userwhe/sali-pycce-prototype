from __future__ import annotations

import warnings

import numpy as np
import pytest

from sali.config import DataConfig, ModelConfig, PhysicsConfig
from sali.metrics import SampleMetrics, aggregate_by_true_count, box_iou, evaluate_sample
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


def test_evaluate_sample_uses_global_one_to_one_matching(rng: np.random.Generator) -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    physics = PhysicsConfig(bz_tesla=0.0056, add_shot_noise=False)
    nuclei = Couplings(
        az_khz=np.array([0.0, 0.0], dtype=np.float32),
        aperp_khz=np.array([20.0, 80.0], dtype=np.float32),
    )
    true_box_0, true_box_1 = true_boxes(nuclei, data, model)
    ambiguous_box = Box(
        row_min=min(true_box_0.row_min, true_box_1.row_min),
        col_min=min(true_box_0.col_min, true_box_1.col_min),
        row_max=max(true_box_0.row_max, true_box_1.row_max),
        col_max=max(true_box_0.col_max, true_box_1.col_max),
    )
    predictions = [
        Prediction(az_khz=0.0, aperp_khz=80.0, row=0.0, col=0.0, box=ambiguous_box, confidence=0.1),
        Prediction(az_khz=0.0, aperp_khz=20.0, row=0.0, col=0.0, box=true_box_0, confidence=0.9),
    ]
    signals = generate_sample_signals(nuclei, physics, rng)

    result = evaluate_sample(predictions, nuclei, signals, data, model, physics, rng)

    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0


def test_aggregate_by_true_count_ignores_all_nan_mae_without_warnings() -> None:
    results = [
        SampleMetrics(
            true_nuclei=2,
            predicted_nuclei=1,
            true_positives=0,
            false_positives=1,
            false_negatives=2,
            mae_az_khz=float("nan"),
            mae_aperp_khz=float("nan"),
            signal_mae_32=0.1,
            signal_mae_256=0.2,
        ),
        SampleMetrics(
            true_nuclei=2,
            predicted_nuclei=0,
            true_positives=0,
            false_positives=0,
            false_negatives=2,
            mae_az_khz=float("nan"),
            mae_aperp_khz=float("nan"),
            signal_mae_32=0.3,
            signal_mae_256=0.4,
        ),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        summary = aggregate_by_true_count(results)

    assert np.isnan(summary[2]["mae_az_khz"])
    assert np.isnan(summary[2]["mae_aperp_khz"])


def test_evaluate_sample_rejects_normalized_like_signals(rng: np.random.Generator) -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    physics = PhysicsConfig(bz_tesla=0.0056, add_shot_noise=False)
    nuclei = Couplings(
        az_khz=np.array([0.0], dtype=np.float32),
        aperp_khz=np.array([52.0], dtype=np.float32),
    )
    box = true_boxes(nuclei, data, model)[0]
    predictions = [
        Prediction(az_khz=0.0, aperp_khz=52.0, row=102.0, col=52.0, box=box, confidence=0.9)
    ]
    raw_signals = generate_sample_signals(nuclei, physics, rng)
    raw_signals[0, 0] = -0.01
    raw_signals[1, 0] = 1.01

    with pytest.raises(FloatingPointError, match="raw_signals"):
        evaluate_sample(predictions, nuclei, raw_signals=raw_signals, data=data, model=model, physics=physics, rng=rng)


def test_evaluate_sample_rejects_malformed_raw_signals(rng: np.random.Generator) -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    physics = PhysicsConfig(bz_tesla=0.0056, add_shot_noise=False)
    nuclei = Couplings(
        az_khz=np.array([0.0], dtype=np.float32),
        aperp_khz=np.array([52.0], dtype=np.float32),
    )
    box = true_boxes(nuclei, data, model)[0]
    predictions = [
        Prediction(az_khz=0.0, aperp_khz=52.0, row=102.0, col=52.0, box=box, confidence=0.9)
    ]

    with pytest.raises(ValueError, match="raw_signals must have shape"):
        evaluate_sample(
            predictions,
            nuclei,
            raw_signals=np.ones((physics.signal_points,), dtype=np.float32),
            data=data,
            model=model,
            physics=physics,
            rng=rng,
        )

    raw_signals = generate_sample_signals(nuclei, physics, rng)
    raw_signals[0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="raw_signals contains non-finite values"):
        evaluate_sample(predictions, nuclei, raw_signals=raw_signals, data=data, model=model, physics=physics, rng=rng)


def test_perfect_predictions_have_near_zero_raw_signal_mae(rng: np.random.Generator) -> None:
    data = DataConfig(train_samples=1, val_samples=1, test_samples=1)
    model = ModelConfig()
    physics = PhysicsConfig(bz_tesla=0.0056, add_shot_noise=False)
    nuclei = Couplings(
        az_khz=np.array([0.0], dtype=np.float32),
        aperp_khz=np.array([52.0], dtype=np.float32),
    )
    box = true_boxes(nuclei, data, model)[0]
    predictions = [
        Prediction(az_khz=0.0, aperp_khz=52.0, row=102.0, col=52.0, box=box, confidence=0.9)
    ]
    raw_signals = generate_sample_signals(nuclei, physics, rng)

    result = evaluate_sample(predictions, nuclei, raw_signals=raw_signals, data=data, model=model, physics=physics, rng=rng)

    assert result.signal_mae_32 == pytest.approx(0.0, abs=1e-7)
    assert result.signal_mae_256 == pytest.approx(0.0, abs=1e-7)

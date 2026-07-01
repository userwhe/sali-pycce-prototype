from __future__ import annotations

import numpy as np
import pytest

from sali.config import DataConfig, ModelConfig, PhysicsConfig, RunConfig, TrainingConfig
from sali.data import Sample
from sali.diagnostics import evaluate_predictions, select_threshold
from sali.physics import Couplings, generate_sample_signals
from sali.targets import render_heatmap


def _one_spin_config() -> RunConfig:
    return RunConfig(
        physics=PhysicsConfig(bz_tesla=0.0056, signal_points=16, add_shot_noise=False),
        data=DataConfig(train_samples=1, val_samples=1, test_samples=1),
        model=ModelConfig(signal_length=16),
        training=TrainingConfig(batch_size=2, max_epochs=1),
    )


def _one_spin_sample(cfg: RunConfig) -> Sample:
    nuclei = Couplings(
        az_khz=np.array([0.0], dtype=np.float32),
        aperp_khz=np.array([52.0], dtype=np.float32),
    )
    raw_signals = generate_sample_signals(nuclei, cfg.physics, np.random.default_rng(123))
    heatmap = render_heatmap(nuclei, cfg.data, cfg.model)
    return Sample(
        signals=raw_signals.copy(),
        raw_signals=raw_signals,
        heatmap=heatmap,
        nuclei=nuclei,
        split="test",
    )


def test_evaluate_predictions_reports_threshold_metrics_and_baselines() -> None:
    cfg = _one_spin_config()
    sample = _one_spin_sample(cfg)
    prediction = sample.heatmap.copy()

    report = evaluate_predictions([sample], [prediction], cfg, thresholds=[0.25])

    assert report["samples"] == 1
    assert report["model_mse"] == pytest.approx(0.0)
    assert report["zero_baseline_mse"] > 0.0
    assert report["pred_max_max"] == pytest.approx(1.0)
    assert report["true_pixel_pred_max"] == pytest.approx(1.0)
    assert report["threshold_sweep"][0]["threshold"] == pytest.approx(0.25)
    assert report["threshold_sweep"][0]["precision"] == pytest.approx(1.0)
    assert report["threshold_sweep"][0]["recall"] == pytest.approx(1.0)


def test_select_threshold_prefers_highest_f1_then_precision() -> None:
    sweep = [
        {"threshold": 0.25, "precision": 1.0, "recall": 0.1, "f1": 0.18},
        {"threshold": 0.10, "precision": 0.5, "recall": 0.5, "f1": 0.5},
        {"threshold": 0.05, "precision": 0.4, "recall": 0.8, "f1": 0.5},
    ]

    selected = select_threshold(sweep)

    assert selected["threshold"] == pytest.approx(0.10)
    assert selected["precision"] == pytest.approx(0.5)


def test_evaluate_predictions_rejects_mismatched_sample_and_prediction_counts() -> None:
    cfg = _one_spin_config()
    sample = _one_spin_sample(cfg)

    with pytest.raises(ValueError, match="same length"):
        evaluate_predictions([sample], [], cfg, thresholds=[0.25])

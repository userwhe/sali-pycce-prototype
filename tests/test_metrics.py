import numpy as np

from sali_pycce.heatmap import HeatmapSpec, make_heatmap
from sali_pycce.metrics import (
    aggregate_detection_metrics,
    compute_detection_metrics,
    threshold_sweep,
)
from sali_pycce.physics import SpinParams


def test_compute_detection_metrics_counts_precision_recall_and_mae():
    truth = np.asarray([[10.0, 20.0], [80.0, 90.0]], dtype=float)
    pred = [
        {"az_khz": 11.0, "aperp_khz": 19.0},
        {"az_khz": 200.0, "aperp_khz": 200.0},
    ]

    metrics = compute_detection_metrics(truth, pred, max_dist_khz=5.0)

    assert metrics["tp"] == 1.0
    assert metrics["fp"] == 1.0
    assert metrics["fn"] == 1.0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["mae_khz"] == 1.0


def test_aggregate_detection_metrics_weights_mae_by_true_positives():
    rows = [
        {"tp": 1.0, "fp": 0.0, "fn": 0.0, "mae_khz": 10.0},
        {"tp": 3.0, "fp": 0.0, "fn": 0.0, "mae_khz": 2.0},
    ]

    metrics = aggregate_detection_metrics(rows)

    assert metrics["mae_khz"] == 4.0


def test_compute_detection_metrics_ignores_nan_padded_truth_rows():
    truth = np.asarray([[10.0, 20.0], [np.nan, np.nan]], dtype=float)
    pred = [{"az_khz": 10.0, "aperp_khz": 20.0}]

    metrics = compute_detection_metrics(truth, pred, max_dist_khz=5.0)

    assert metrics["tp"] == 1.0
    assert metrics["fn"] == 0.0
    assert metrics["recall"] == 1.0


def test_compute_detection_metrics_uses_optimal_assignment():
    truth = np.asarray([[0.0, 0.0], [3.0, 0.0]], dtype=float)
    pred = [
        {"az_khz": 1.0, "aperp_khz": 0.0},
        {"az_khz": -2.0, "aperp_khz": 0.0},
    ]

    metrics = compute_detection_metrics(truth, pred, max_dist_khz=3.0)

    assert metrics["tp"] == 2.0
    assert metrics["fn"] == 0.0
    assert metrics["fp"] == 0.0
    assert metrics["recall"] == 1.0


def test_threshold_sweep_returns_one_row_per_threshold():
    spec = HeatmapSpec(height=32, width=64)
    spin = SpinParams(az_khz=10.0, aperp_khz=40.0)
    heatmap = make_heatmap([spin], spec)
    truth = np.asarray([[spin.az_khz, spin.aperp_khz]], dtype=float)

    rows = threshold_sweep(
        pred_heatmaps=np.asarray([heatmap]),
        truth_spins=[truth],
        spec=spec,
        thresholds=[0.2, 0.5],
        max_dist_khz=5.0,
    )

    assert [row["threshold"] for row in rows] == [0.2, 0.5]
    assert all(row["precision"] == 1.0 for row in rows)
    assert all(row["recall"] == 1.0 for row in rows)

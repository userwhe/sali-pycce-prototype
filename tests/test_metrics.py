import numpy as np

from sali_pycce.heatmap import HeatmapSpec, make_heatmap
from sali_pycce.metrics import compute_detection_metrics, threshold_sweep
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

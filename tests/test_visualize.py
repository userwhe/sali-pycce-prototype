import csv

import numpy as np

from sali_pycce.heatmap import HeatmapSpec
from sali_pycce.visualize import (
    plot_heatmap_comparison,
    plot_loss_history,
    plot_metric_history,
    plot_raw_traces_and_spins,
)


def test_visualization_helpers_write_pngs(tmp_path):
    taus = np.stack([np.linspace(0.0, 40.0, 32), np.linspace(0.0, 40.0, 32)])
    signals = np.stack([np.linspace(1.0, 0.8, 32), np.linspace(1.0, 0.7, 32)])
    spins = np.asarray([[10.0, 20.0], [-30.0, 90.0]])
    spec = HeatmapSpec(height=32, width=64)
    target = np.zeros((32, 64), dtype=np.float32)
    pred = np.zeros((32, 64), dtype=np.float32)
    history = tmp_path / "history.csv"
    with history.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "precision", "recall", "mae_khz"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 1,
                "train_loss": 0.2,
                "val_loss": 0.3,
                "precision": 0.4,
                "recall": 0.5,
                "mae_khz": 6.0,
            }
        )

    paths = [
        plot_raw_traces_and_spins(taus, signals, spins, spec, tmp_path / "raw.png"),
        plot_heatmap_comparison(target, pred, tmp_path / "heatmaps.png"),
        plot_loss_history(history, tmp_path / "loss.png"),
        plot_metric_history(history, tmp_path / "metrics.png"),
    ]

    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0

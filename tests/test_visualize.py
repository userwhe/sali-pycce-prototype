import csv

import numpy as np
import matplotlib.pyplot as plt

from sali_pycce.heatmap import HeatmapSpec
import sali_pycce.visualize as visualize
from sali_pycce.visualize import (
    _has_finite_metric,
    plot_heatmap_comparison,
    plot_loss_history,
    plot_metric_history,
    plot_raw_traces_and_spins,
    plot_signal_reconstruction_overlay,
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


def test_raw_traces_are_plotted_in_separate_panels(monkeypatch, tmp_path):
    taus = np.stack([np.linspace(0.0, 40.0, 32), np.linspace(0.0, 40.0, 32)])
    signals = np.stack([np.linspace(1.0, 0.8, 32), np.linspace(1.0, 0.7, 32)])
    spins = np.asarray([[10.0, 20.0], [-30.0, 90.0]])
    spec = HeatmapSpec(height=32, width=64)
    captured = {}

    def keep_fig_open(fig, out):
        captured["fig"] = fig
        captured["out"] = out
        return tmp_path / "raw.png"

    monkeypatch.setattr(visualize, "_finish", keep_fig_open)

    path = plot_raw_traces_and_spins(taus, signals, spins, spec, tmp_path / "raw.png")

    assert path == tmp_path / "raw.png"
    axes = captured["fig"].axes
    try:
        assert len(axes) == 3
        assert len(axes[0].lines) == 1
        assert len(axes[1].lines) == 1
        assert axes[0].get_title() == "Raw CPMG trace (N=32)"
        assert axes[1].get_title() == "Raw CPMG trace (N=256)"
    finally:
        plt.close(captured["fig"])


def test_signal_reconstruction_overlay_uses_one_panel_per_pulse_count(monkeypatch, tmp_path):
    taus = np.stack([np.linspace(0.0, 40.0, 32), np.linspace(0.0, 40.0, 32)])
    original = np.stack([np.linspace(1.0, 0.8, 32), np.linspace(1.0, 0.7, 32)])
    reconstructed = np.stack([np.linspace(0.98, 0.82, 32), np.linspace(0.96, 0.72, 32)])
    captured = {}

    def keep_fig_open(fig, out):
        captured["fig"] = fig
        captured["out"] = out
        return tmp_path / "overlay.png"

    monkeypatch.setattr(visualize, "_finish", keep_fig_open)

    path = plot_signal_reconstruction_overlay(
        taus,
        original,
        reconstructed,
        tmp_path / "overlay.png",
    )

    assert path == tmp_path / "overlay.png"
    axes = captured["fig"].axes
    try:
        assert len(axes) == 2
        assert len(axes[0].lines) == 2
        assert len(axes[1].lines) == 2
        assert axes[0].get_title() == "Signal overlay (N=32)"
        assert axes[1].get_title() == "Signal overlay (N=256)"
    finally:
        plt.close(captured["fig"])


def test_metric_history_detects_all_nan_mae(tmp_path):
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
                "precision": 0.0,
                "recall": 0.0,
                "mae_khz": "nan",
            }
        )
        writer.writerow(
            {
                "epoch": 2,
                "train_loss": 0.1,
                "val_loss": 0.2,
                "precision": 0.0,
                "recall": 0.0,
                "mae_khz": "nan",
            }
        )

    assert not _has_finite_metric(history, "mae_khz")
    path = plot_metric_history(history, tmp_path / "nan_metrics.png")
    assert path.exists()
    assert path.stat().st_size > 0

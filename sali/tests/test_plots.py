from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from sali.metrics import SampleMetrics


def _plots():
    return importlib.import_module("sali.plots")


def _signals(points: int = 8) -> np.ndarray:
    return np.vstack(
        [
            np.linspace(0.1, 0.9, points, dtype=np.float32),
            np.linspace(0.2, 0.8, points, dtype=np.float32),
        ]
    )


def _metric(
    *,
    true_nuclei: int = 2,
    mae_az_khz: float = 0.1,
    mae_aperp_khz: float = 0.2,
) -> SampleMetrics:
    return SampleMetrics(
        true_nuclei=true_nuclei,
        predicted_nuclei=2,
        true_positives=1,
        false_positives=1,
        false_negatives=1,
        mae_az_khz=mae_az_khz,
        mae_aperp_khz=mae_aperp_khz,
        signal_mae_32=0.3,
        signal_mae_256=0.4,
    )


def test_importing_plots_does_not_change_matplotlib_backend() -> None:
    original_backend = matplotlib.get_backend()
    sys.modules.pop("sali.plots", None)
    try:
        matplotlib.use("svg", force=True)
        expected_backend = matplotlib.get_backend()

        module = importlib.import_module("sali.plots")
        assert matplotlib.get_backend() == expected_backend

        importlib.reload(module)
        assert matplotlib.get_backend() == expected_backend
    finally:
        sys.modules.pop("sali.plots", None)
        matplotlib.use(original_backend, force=True)


def test_plot_loss_writes_file(tmp_path: Path) -> None:
    path = tmp_path / "loss.png"
    _plots().plot_loss({"train_loss": [1.0, 0.5], "val_loss": [1.2, 0.6]}, path)
    assert path.exists()


def test_plot_loss_accepts_string_path(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "loss.png"
    _plots().plot_loss({"train_loss": [1.0, 0.5], "val_loss": [1.2, 0.6]}, str(path))
    assert path.exists()


def test_plot_loss_rejects_empty_history(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="history"):
        _plots().plot_loss({"train_loss": [], "val_loss": []}, tmp_path / "loss.png")


def test_plot_loss_closes_figures_after_success(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    before = tuple(plt.get_fignums())
    _plots().plot_loss({"train_loss": [1.0, 0.5], "val_loss": [1.2, 0.6]}, tmp_path / "loss.png")
    assert tuple(plt.get_fignums()) == before


def test_plot_loss_closes_figures_after_save_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    def fail_savefig(self: Figure, *args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    before = tuple(plt.get_fignums())
    monkeypatch.setattr(Figure, "savefig", fail_savefig)
    with pytest.raises(OSError, match="disk full"):
        _plots().plot_loss({"train_loss": [1.0, 0.5], "val_loss": [1.2, 0.6]}, tmp_path / "loss.png")
    assert tuple(plt.get_fignums()) == before


def test_plot_heatmap_writes_file(tmp_path: Path) -> None:
    path = tmp_path / "heatmap.png"
    heatmap = np.zeros((1, 204, 104), dtype=np.float32)
    heatmap[0, 100, 50] = 1.0
    _plots().plot_heatmap(heatmap, "True Heatmap", path)
    assert path.exists()


def test_plot_heatmap_rejects_malformed_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="heatmap.*shape"):
        _plots().plot_heatmap(np.zeros((204, 104), dtype=np.float32), "Bad", tmp_path / "heatmap.png")


def test_plot_heatmap_rejects_nonfinite_values(tmp_path: Path) -> None:
    heatmap = np.zeros((1, 2, 2), dtype=np.float32)
    heatmap[0, 0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="heatmap.*finite"):
        _plots().plot_heatmap(heatmap, "Bad", tmp_path / "heatmap.png")


def test_plot_heatmap_validation_failure_does_not_open_figure(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    before = tuple(plt.get_fignums())
    with pytest.raises(ValueError, match="heatmap.*shape"):
        _plots().plot_heatmap(np.zeros((204, 104), dtype=np.float32), "Bad", tmp_path / "heatmap.png")
    assert tuple(plt.get_fignums()) == before


def test_plot_spectra_writes_file(tmp_path: Path) -> None:
    path = tmp_path / "spectra.png"
    _plots().plot_spectra(_signals(), path)
    assert path.exists()


@pytest.mark.parametrize(
    ("signals", "expected_error", "message"),
    [
        (np.zeros((1, 4), dtype=np.float32), ValueError, "signals.*shape"),
        (np.zeros((2, 1), dtype=np.float32), ValueError, "signals.*N > 1"),
        (
            np.array([[0.1, np.nan, 0.3], [0.2, 0.3, 0.4]], dtype=np.float32),
            FloatingPointError,
            "signals.*finite",
        ),
        (
            np.array([[0.1, 1.1, 0.3], [0.2, 0.3, 0.4]], dtype=np.float32),
            ValueError,
            "signals.*range",
        ),
    ],
)
def test_plot_spectra_validates_signals(
    signals: np.ndarray, expected_error: type[Exception], message: str, tmp_path: Path
) -> None:
    with pytest.raises(expected_error, match=message):
        _plots().plot_spectra(signals, tmp_path / "spectra.png")


def test_plot_signal_overlay_writes_file(tmp_path: Path) -> None:
    path = tmp_path / "overlay.png"
    original = _signals()
    reconstructed = np.clip(original * 0.95, 0.0, 1.0)
    _plots().plot_signal_overlay(original, reconstructed, path)
    assert path.exists()


def test_plot_signal_overlay_rejects_mismatched_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same shape"):
        _plots().plot_signal_overlay(_signals(8), _signals(7), tmp_path / "overlay.png")


def test_plot_signal_overlay_rejects_out_of_range_values(tmp_path: Path) -> None:
    reconstructed = _signals()
    reconstructed[0, 0] = -0.1
    with pytest.raises(ValueError, match="range"):
        _plots().plot_signal_overlay(_signals(), reconstructed, tmp_path / "overlay.png")


def test_plot_precision_recall_writes_file(tmp_path: Path) -> None:
    path = tmp_path / "precision-recall.png"
    _plots().plot_precision_recall([_metric(true_nuclei=1), _metric(true_nuclei=2)], path)
    assert path.exists()


def test_plot_precision_recall_rejects_empty_results(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="results"):
        _plots().plot_precision_recall([], tmp_path / "precision-recall.png")


def test_plot_mae_writes_file(tmp_path: Path) -> None:
    path = tmp_path / "mae.png"
    _plots().plot_mae([_metric(true_nuclei=1), _metric(true_nuclei=2)], path)
    assert path.exists()


def test_plot_mae_rejects_empty_results(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="results"):
        _plots().plot_mae([], tmp_path / "mae.png")


def test_plot_mae_handles_all_nan_coupling_mae_without_warnings(tmp_path: Path) -> None:
    path = tmp_path / "mae.png"
    results = [
        _metric(mae_az_khz=float("nan"), mae_aperp_khz=float("nan")),
        _metric(mae_az_khz=float("nan"), mae_aperp_khz=float("nan")),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _plots().plot_mae(results, path)

    assert path.exists()

"""Plot helpers for SALI-PyCCE notebooks and reports."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .heatmap import HeatmapSpec


def _finish(fig, out: str | Path) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _finite_spins(spins: np.ndarray) -> np.ndarray:
    arr = np.asarray(spins, dtype=float).reshape(-1, 2)
    return arr[np.isfinite(arr).all(axis=1)]


def plot_raw_traces_and_spins(
    taus_us: np.ndarray,
    signals: np.ndarray,
    spins: np.ndarray,
    spec: HeatmapSpec,
    out: str | Path,
) -> Path:
    taus = np.asarray(taus_us)
    traces = np.asarray(signals)
    spin_arr = _finite_spins(spins)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].plot(taus[0], traces[0], label="N=32")
    axes[0].set_xlabel("tau (us)")
    axes[0].set_ylabel("P_x")
    axes[0].set_title("Raw CPMG trace (N=32)")

    axes[1].plot(taus[1], traces[1], label="N=256")
    axes[1].set_xlabel("tau (us)")
    axes[1].set_ylabel("P_x")
    axes[1].set_title("Raw CPMG trace (N=256)")

    if len(spin_arr):
        axes[2].scatter(spin_arr[:, 0], spin_arr[:, 1], s=28)
    axes[2].set_xlim(*spec.az_range)
    axes[2].set_ylim(*spec.aperp_range)
    axes[2].set_xlabel("A_z (kHz)")
    axes[2].set_ylabel("A_perp (kHz)")
    axes[2].set_title("True spins")
    return _finish(fig, out)


def plot_heatmap_comparison(target: np.ndarray, pred: np.ndarray, out: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].imshow(np.asarray(target).squeeze(), origin="lower", aspect="auto")
    axes[0].set_title("Ground-truth heatmap")
    axes[1].imshow(np.asarray(pred).squeeze(), origin="lower", aspect="auto")
    axes[1].set_title("Predicted heatmap")
    return _finish(fig, out)


def plot_signal_reconstruction_overlay(
    taus_us: np.ndarray,
    original_signals: np.ndarray,
    reconstructed_signals: np.ndarray,
    out: str | Path,
    pulse_labels: tuple[str, str] = ("N=32", "N=256"),
) -> Path:
    taus = np.asarray(taus_us)
    original = np.asarray(original_signals)
    reconstructed = np.asarray(reconstructed_signals)
    if taus.shape != original.shape or original.shape != reconstructed.shape:
        raise ValueError(
            "taus_us, original_signals, and reconstructed_signals must have the same shape"
        )
    if original.shape[0] != 2:
        raise ValueError("expected two CPMG traces, one for N=32 and one for N=256")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for i, label in enumerate(pulse_labels):
        axes[i].plot(taus[i], original[i], label="original", linewidth=1.4)
        axes[i].plot(
            taus[i],
            reconstructed[i],
            label="identified C13s",
            linewidth=1.2,
            alpha=0.9,
        )
        axes[i].set_xlabel("tau (us)")
        axes[i].set_ylabel("P_x")
        axes[i].set_title(f"Signal overlay ({label})")
        axes[i].legend()
    return _finish(fig, out)


def _read_history(path: str | Path) -> list[dict[str, float]]:
    with Path(path).open() as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _has_finite_metric(history_csv: str | Path, key: str) -> bool:
    rows = _read_history(history_csv)
    return any(np.isfinite(row[key]) for row in rows)


def plot_loss_history(history_csv: str | Path, out: str | Path) -> Path:
    rows = _read_history(history_csv)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    epochs = [row["epoch"] for row in rows]
    ax.plot(epochs, [row["train_loss"] for row in rows], label="train")
    ax.plot(epochs, [row["val_loss"] for row in rows], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Train/validation loss")
    ax.legend()
    return _finish(fig, out)


def plot_metric_history(history_csv: str | Path, out: str | Path) -> Path:
    rows = _read_history(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    epochs = np.asarray([row["epoch"] for row in rows], dtype=float)
    axes[0].plot(epochs, [row["precision"] for row in rows], label="precision")
    axes[0].plot(epochs, [row["recall"] for row in rows], label="recall")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Detection quality")
    axes[0].legend()

    mae = np.asarray([row["mae_khz"] for row in rows], dtype=float)
    finite = np.isfinite(mae)
    if finite.any():
        axes[1].plot(epochs[finite], mae[finite], marker="o", label="MAE")
        axes[1].legend()
    else:
        axes[1].text(
            0.5,
            0.5,
            "No matched detections",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("kHz")
    axes[1].set_title("Matched-spin MAE")
    return _finish(fig, out)

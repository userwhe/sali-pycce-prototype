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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(taus[0], traces[0], label="N=32")
    axes[0].plot(taus[1], traces[1], label="N=256")
    axes[0].set_xlabel("tau (us)")
    axes[0].set_ylabel("P_x")
    axes[0].set_title("Raw CPMG traces")
    axes[0].legend()

    if len(spin_arr):
        axes[1].scatter(spin_arr[:, 0], spin_arr[:, 1], s=28)
    axes[1].set_xlim(*spec.az_range)
    axes[1].set_ylim(*spec.aperp_range)
    axes[1].set_xlabel("A_z (kHz)")
    axes[1].set_ylabel("A_perp (kHz)")
    axes[1].set_title("True spins")
    return _finish(fig, out)


def plot_heatmap_comparison(target: np.ndarray, pred: np.ndarray, out: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].imshow(np.asarray(target).squeeze(), origin="lower", aspect="auto")
    axes[0].set_title("Ground-truth heatmap")
    axes[1].imshow(np.asarray(pred).squeeze(), origin="lower", aspect="auto")
    axes[1].set_title("Predicted heatmap")
    return _finish(fig, out)


def _read_history(path: str | Path) -> list[dict[str, float]]:
    with Path(path).open() as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


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
    epochs = [row["epoch"] for row in rows]
    axes[0].plot(epochs, [row["precision"] for row in rows], label="precision")
    axes[0].plot(epochs, [row["recall"] for row in rows], label="recall")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()

    axes[1].plot(epochs, [row["mae_khz"] for row in rows], label="MAE")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("kHz")
    axes[1].legend()
    return _finish(fig, out)

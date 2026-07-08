from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sali.metrics import SampleMetrics
from sali.physics import tau_grid_us


def _prepare_path(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _validate_history(history: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if not history:
        raise ValueError("history must include non-empty train_loss and val_loss")
    try:
        train_loss = np.asarray(history["train_loss"], dtype=np.float64)
        val_loss = np.asarray(history["val_loss"], dtype=np.float64)
    except KeyError as exc:
        raise ValueError("history must include train_loss and val_loss") from exc
    if train_loss.ndim != 1 or val_loss.ndim != 1 or train_loss.size == 0 or val_loss.size == 0:
        raise ValueError("history must include non-empty train_loss and val_loss")
    if train_loss.size != val_loss.size:
        raise ValueError("history train_loss and val_loss must have the same length")
    if not np.all(np.isfinite(train_loss)) or not np.all(np.isfinite(val_loss)):
        raise FloatingPointError("history losses must be finite")
    if "step" in history:
        x_values = np.asarray(history["step"], dtype=np.float64)
        if x_values.ndim != 1 or x_values.size != train_loss.size:
            raise ValueError("history step values must match loss history length")
        if not np.all(np.isfinite(x_values)):
            raise FloatingPointError("history step values must be finite")
        return train_loss, val_loss, x_values, "Step"
    return train_loss, val_loss, np.arange(1, train_loss.size + 1, dtype=np.float64), "Epoch"


def _validate_heatmap(heatmap: np.ndarray) -> np.ndarray:
    values = np.asarray(heatmap)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError("heatmap must have shape (1, H, W) with H and W greater than 0")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("heatmap contains non-finite values")
    return values


def _validate_signals(signals: np.ndarray, name: str = "signals") -> np.ndarray:
    values = np.asarray(signals, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError(f"{name} must have shape (2, N)")
    if values.shape[1] <= 1:
        raise ValueError(f"{name} must have shape (2, N) with N > 1")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(f"{name} contains non-finite values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} values must be in probability range [0, 1]")
    return values


def _validate_results(results: list[SampleMetrics]) -> None:
    if not results:
        raise ValueError("results must not be empty")


def _finite_mean_or_nan(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def plot_loss(history: dict[str, list[float]], path: str | Path) -> None:
    train_loss, val_loss, x_values, x_label = _validate_history(history)
    output_path = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        ax.plot(x_values, train_loss, label="train")
        ax.plot(x_values, val_loss, label="validation")
        ax.set_xlabel(x_label)
        ax.set_ylabel("MSE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
    finally:
        plt.close(fig)


def plot_heatmap(heatmap: np.ndarray, title: str, path: str | Path) -> None:
    values = _validate_heatmap(heatmap)
    output_path = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(6, 8))
    try:
        image = ax.imshow(values[0], origin="lower", aspect="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Aperp pixel")
        ax.set_ylabel("Az pixel")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
    finally:
        plt.close(fig)


def plot_spectra(signals: np.ndarray, path: str | Path) -> None:
    values = _validate_signals(signals)
    output_path = _prepare_path(path)
    tau32 = tau_grid_us(6.0, 50.0, values.shape[1])
    tau256 = tau_grid_us(10.0, 40.0, values.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharey=True)
    try:
        axes[0].plot(tau32, values[0])
        axes[0].set_title("Generated spectrum N=32")
        axes[0].set_xlabel("tau (us)")
        axes[0].set_ylabel("P_x")
        axes[1].plot(tau256, values[1])
        axes[1].set_title("Generated spectrum N=256")
        axes[1].set_xlabel("tau (us)")
        axes[1].set_ylabel("P_x")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
    finally:
        plt.close(fig)


def plot_signal_overlay(original: np.ndarray, reconstructed: np.ndarray, path: str | Path) -> None:
    original_values = _validate_signals(original, "original")
    reconstructed_values = _validate_signals(reconstructed, "reconstructed")
    if original_values.shape != reconstructed_values.shape:
        raise ValueError("original and reconstructed must have the same shape")
    output_path = _prepare_path(path)
    tau32 = tau_grid_us(6.0, 50.0, original_values.shape[1])
    tau256 = tau_grid_us(10.0, 40.0, original_values.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharey=True)
    try:
        axes[0].plot(tau32, original_values[0], label="original", color="black")
        axes[0].plot(tau32, reconstructed_values[0], label="identified C13", color="tab:orange")
        axes[0].set_title("Original vs identified signal N=32")
        axes[0].legend()
        axes[1].plot(tau256, original_values[1], label="original", color="black")
        axes[1].plot(tau256, reconstructed_values[1], label="identified C13", color="tab:blue")
        axes[1].set_title("Original vs identified signal N=256")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
    finally:
        plt.close(fig)


def plot_precision_recall(results: list[SampleMetrics], path: str | Path) -> None:
    _validate_results(results)
    output_path = _prepare_path(path)
    counts = sorted({result.true_nuclei for result in results})
    precision = []
    recall = []
    for count in counts:
        selected = [result for result in results if result.true_nuclei == count]
        precision.append(float(np.mean([item.precision for item in selected])))
        recall.append(float(np.mean([item.recall for item in selected])))
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        ax.plot(counts, precision, marker="o", label="precision")
        ax.plot(counts, recall, marker="o", label="recall")
        ax.set_xlabel("True nuclei")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
    finally:
        plt.close(fig)


def plot_mae(results: list[SampleMetrics], path: str | Path) -> None:
    _validate_results(results)
    output_path = _prepare_path(path)
    counts = sorted({result.true_nuclei for result in results})
    mae_az = []
    mae_aperp = []
    signal32 = []
    signal256 = []
    for count in counts:
        selected = [result for result in results if result.true_nuclei == count]
        mae_az.append(_finite_mean_or_nan([item.mae_az_khz for item in selected]))
        mae_aperp.append(_finite_mean_or_nan([item.mae_aperp_khz for item in selected]))
        signal32.append(float(np.mean([item.signal_mae_32 for item in selected])))
        signal256.append(float(np.mean([item.signal_mae_256 for item in selected])))
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    try:
        axes[0].plot(counts, mae_az, marker="o", label="Az")
        axes[0].plot(counts, mae_aperp, marker="o", label="Aperp")
        axes[0].set_ylabel("Coupling MAE (kHz)")
        axes[0].legend()
        axes[1].plot(counts, signal32, marker="o", label="N=32")
        axes[1].plot(counts, signal256, marker="o", label="N=256")
        axes[1].set_xlabel("True nuclei")
        axes[1].set_ylabel("Signal MAE")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
    finally:
        plt.close(fig)


def plot_mae_by_nuclei(results: list[SampleMetrics], path: str | Path) -> None:
    _validate_results(results)
    output_path = _prepare_path(path)
    counts = sorted({result.true_nuclei for result in results})
    mae_az = []
    mae_aperp = []
    mean_coupling = []
    for count in counts:
        selected = [result for result in results if result.true_nuclei == count]
        az = _finite_mean_or_nan([item.mae_az_khz for item in selected])
        aperp = _finite_mean_or_nan([item.mae_aperp_khz for item in selected])
        mae_az.append(az)
        mae_aperp.append(aperp)
        mean_coupling.append(_finite_mean_or_nan([az, aperp]))
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        ax.plot(counts, mean_coupling, marker="o", label="mean")
        ax.plot(counts, mae_az, marker="o", linestyle="--", label="Az")
        ax.plot(counts, mae_aperp, marker="o", linestyle="--", label="Aperp")
        ax.set_xlabel("True nuclei")
        ax.set_ylabel("Coupling MAE (kHz)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
    finally:
        plt.close(fig)

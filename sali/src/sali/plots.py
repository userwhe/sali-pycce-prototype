from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sali.metrics import SampleMetrics
from sali.physics import tau_grid_us


def _prepare_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_loss(history: dict[str, list[float]], path: Path) -> None:
    _prepare_path(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_heatmap(heatmap: np.ndarray, title: str, path: Path) -> None:
    _prepare_path(path)
    fig, ax = plt.subplots(figsize=(6, 8))
    image = ax.imshow(heatmap[0], origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Aperp pixel")
    ax.set_ylabel("Az pixel")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_spectra(signals: np.ndarray, path: Path) -> None:
    _prepare_path(path)
    tau32 = tau_grid_us(6.0, 50.0, signals.shape[1])
    tau256 = tau_grid_us(10.0, 40.0, signals.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharey=True)
    axes[0].plot(tau32, signals[0])
    axes[0].set_title("Generated spectrum N=32")
    axes[0].set_xlabel("tau (us)")
    axes[0].set_ylabel("P_x")
    axes[1].plot(tau256, signals[1])
    axes[1].set_title("Generated spectrum N=256")
    axes[1].set_xlabel("tau (us)")
    axes[1].set_ylabel("P_x")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_signal_overlay(original: np.ndarray, reconstructed: np.ndarray, path: Path) -> None:
    _prepare_path(path)
    tau32 = tau_grid_us(6.0, 50.0, original.shape[1])
    tau256 = tau_grid_us(10.0, 40.0, original.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharey=True)
    axes[0].plot(tau32, original[0], label="original", color="black")
    axes[0].plot(tau32, reconstructed[0], label="identified C13", color="tab:orange")
    axes[0].set_title("Original vs identified signal N=32")
    axes[0].legend()
    axes[1].plot(tau256, original[1], label="original", color="black")
    axes[1].plot(tau256, reconstructed[1], label="identified C13", color="tab:blue")
    axes[1].set_title("Original vs identified signal N=256")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_precision_recall(results: list[SampleMetrics], path: Path) -> None:
    _prepare_path(path)
    counts = sorted({result.true_nuclei for result in results})
    precision = []
    recall = []
    for count in counts:
        selected = [result for result in results if result.true_nuclei == count]
        precision.append(float(np.mean([item.precision for item in selected])))
        recall.append(float(np.mean([item.recall for item in selected])))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(counts, precision, marker="o", label="precision")
    ax.plot(counts, recall, marker="o", label="recall")
    ax.set_xlabel("True nuclei")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_mae(results: list[SampleMetrics], path: Path) -> None:
    _prepare_path(path)
    counts = sorted({result.true_nuclei for result in results})
    mae_az = []
    mae_aperp = []
    signal32 = []
    signal256 = []
    for count in counts:
        selected = [result for result in results if result.true_nuclei == count]
        mae_az.append(float(np.nanmean([item.mae_az_khz for item in selected])))
        mae_aperp.append(float(np.nanmean([item.mae_aperp_khz for item in selected])))
        signal32.append(float(np.mean([item.signal_mae_32 for item in selected])))
        signal256.append(float(np.mean([item.signal_mae_256 for item in selected])))
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
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
    fig.savefig(path, dpi=160)
    plt.close(fig)

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SALI in Colab and generate report plots.")
    parser.add_argument("--archive", default="/content/sali_pycce_repo.tar.gz")
    parser.add_argument("--project-root", default="/content/sali-pycce-prototype")
    parser.add_argument("--out-dir", default="/content/sali_colab_report")
    parser.add_argument("--preset", default="colab-medium")
    parser.add_argument("--train-samples", type=int, default=2_048)
    parser.add_argument("--val-samples", type=int, default=384)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metric-threshold", type=float, default=0.15)
    parser.add_argument("--match-distance-khz", type=float, default=5.0)
    parser.add_argument("--loss", default="weighted-mse")
    parser.add_argument("--pos-weight", type=float, default=200.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-seed", type=int, default=5678)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-tests", action="store_true")
    return parser


def extract_project(archive: Path, project_root: Path) -> None:
    if project_root.exists():
        shutil.rmtree(project_root)
    project_root.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(project_root)


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def strict_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, allow_nan=False) + "\n"


def f1_score(row: dict[str, float]) -> float:
    precision = float(row["precision"])
    recall = float(row["recall"])
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def save_fig(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    return out


def save_raw_cpmg_traces(taus_us, signals, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(taus_us[0], signals[0])
    axes[0].set_title("Raw CPMG trace (N=32)")
    axes[0].set_xlabel("tau (us)")
    axes[0].set_ylabel("P_x")
    axes[1].plot(taus_us[1], signals[1])
    axes[1].set_title("Raw CPMG trace (N=256)")
    axes[1].set_xlabel("tau (us)")
    axes[1].set_ylabel("P_x")
    save_fig(fig, out)
    plt.close(fig)


def save_spin_scatter(spins, spec, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.asarray(spins, dtype=float).reshape(-1, 2)
    arr = arr[np.isfinite(arr).all(axis=1)]
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    if len(arr):
        ax.scatter(arr[:, 0], arr[:, 1], s=32)
    ax.set_xlim(*spec.az_range)
    ax.set_ylim(*spec.aperp_range)
    ax.set_xlabel("A_z (kHz)")
    ax.set_ylabel("A_perp (kHz)")
    ax.set_title("True spins")
    save_fig(fig, out)
    plt.close(fig)


def save_ground_truth_heatmap(heatmap, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    image = ax.imshow(heatmap.squeeze(), origin="lower", aspect="auto")
    ax.set_title("Ground-truth heatmap label")
    fig.colorbar(image, ax=ax)
    save_fig(fig, out)
    plt.close(fig)


def save_threshold_sweep(rows: list[dict[str, float]], out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    thresholds = np.asarray([row["threshold"] for row in rows], dtype=float)
    precision = np.asarray([row["precision"] for row in rows], dtype=float)
    recall = np.asarray([row["recall"] for row in rows], dtype=float)
    mae = np.asarray([row["mae_khz"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(thresholds, precision, marker="o", label="precision")
    axes[0].plot(thresholds, recall, marker="o", label="recall")
    axes[0].set_xlabel("threshold")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Precision/recall vs threshold")
    axes[0].legend()

    finite = np.isfinite(mae)
    if finite.any():
        axes[1].plot(thresholds[finite], mae[finite], marker="o")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No matched detections",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set_xlabel("threshold")
    axes[1].set_ylabel("kHz")
    axes[1].set_title("MAE vs threshold")
    save_fig(fig, out)
    plt.close(fig)


def reconstruct_signals_from_detections(simulator, detections: list[dict[str, float]]):
    from sali_pycce.physics import SpinParams

    spins = [
        SpinParams(float(detection["az_khz"]), float(detection["aperp_khz"]))
        for detection in detections
    ]
    return simulator.sample(spins, noisy=False)["signals"]


def build_dataset_from_checkpoint(ckpt: dict, spec, samples: int, seed: int):
    from sali_pycce.config import PRESETS
    from sali_pycce.data import SyntheticSALIDataset
    from sali_pycce.physics import AnalyticCPMGSimulator

    ckpt_args = ckpt.get("args", {})
    preset = PRESETS[str(ckpt_args.get("preset", "colab-medium"))]
    tau_start = float(ckpt_args.get("tau_start_us", preset.tau_ranges_us[0][0]))
    tau_stop = float(ckpt_args.get("tau_stop_us", preset.tau_ranges_us[0][1]))
    simulator = AnalyticCPMGSimulator(
        b_gauss=float(ckpt_args.get("b_gauss", preset.b_gauss)),
        pulses=(32, 256),
        tau_ranges_us=((tau_start, tau_stop), (tau_start, tau_stop)),
        signal_points=int(ckpt_args.get("signal_points", preset.signal_points)),
        shots=ckpt_args.get("shots", preset.shots),
        t2_us=ckpt_args.get("t2_us", preset.t2_us),
        t2_stretch=float(ckpt_args.get("t2_stretch", preset.t2_stretch)),
    )
    return SyntheticSALIDataset(
        samples,
        simulator=simulator,
        heatmap_spec=spec,
        min_spins=int(ckpt_args.get("min_spins", preset.min_spins)),
        max_spins=int(ckpt_args.get("max_spins", preset.max_spins)),
        az_range=spec.az_range,
        aperp_range=spec.aperp_range,
        seed=seed,
    )


def collect_predictions(model, dataset, batch_size: int, device: str):
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pred_batches = []
    truth_spins = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["signals"].to(device)).cpu().numpy()
            pred_batches.append(pred)
            truth_spins.extend(batch["spins"].numpy())
    return pred_batches, truth_spins


def compute_metrics_for_threshold(predictions, truth_spins, spec, threshold: float, max_dist_khz: float):
    from sali_pycce.metrics import aggregate_detection_metrics, compute_detection_metrics
    from sali_pycce.heatmap import decode_heatmap
    import numpy as np

    rows = []
    heatmaps = np.concatenate(predictions, axis=0)
    for heatmap, truth in zip(heatmaps[:, 0], truth_spins, strict=True):
        rows.append(
            compute_detection_metrics(
                truth,
                decode_heatmap(heatmap, spec, threshold=threshold),
                max_dist_khz=max_dist_khz,
            )
        )
    return aggregate_detection_metrics(rows)


def generate_report(args: argparse.Namespace, project_root: Path, out_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    import torch

    sys.path.insert(0, str(project_root / "src"))
    from sali_pycce.heatmap import HeatmapSpec
    from sali_pycce.heatmap import decode_heatmap
    from sali_pycce.metrics import threshold_sweep
    from sali_pycce.model import SALINet
    from sali_pycce.visualize import (
        plot_heatmap_comparison,
        plot_loss_history,
        plot_metric_history,
        plot_raw_traces_and_spins,
        plot_signal_reconstruction_overlay,
    )

    checkpoint = out_dir / "checkpoint.pt"
    history = out_dir / "history.csv"
    ckpt = torch.load(checkpoint, map_location=args.device)
    spec = HeatmapSpec(**ckpt.get("heatmap_spec", {}))
    model = SALINet(n_inputs=2, output_shape=(spec.height, spec.width)).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_dataset = build_dataset_from_checkpoint(ckpt, spec, args.val_samples, args.eval_seed + 1)
    test_dataset = build_dataset_from_checkpoint(ckpt, spec, args.test_samples, args.eval_seed)
    val_predictions, val_truth = collect_predictions(model, val_dataset, args.batch_size, args.device)
    test_predictions, test_truth = collect_predictions(model, test_dataset, args.batch_size, args.device)
    val_heatmaps = np.concatenate(val_predictions, axis=0)
    test_heatmaps = np.concatenate(test_predictions, axis=0)

    thresholds = [
        0.005,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.5,
        0.6,
        0.7,
    ]
    sweep_rows = threshold_sweep(
        val_heatmaps,
        val_truth,
        spec,
        thresholds=thresholds,
        max_dist_khz=args.match_distance_khz,
    )
    best_row = max(sweep_rows, key=lambda row: (f1_score(row), row["recall"], row["precision"]))
    best_threshold = float(best_row["threshold"])
    test_metrics = compute_metrics_for_threshold(
        test_predictions,
        test_truth,
        spec,
        threshold=best_threshold,
        max_dist_khz=args.match_distance_khz,
    )

    sample = test_dataset[0]
    with torch.no_grad():
        sample_pred = model(sample["signals"].unsqueeze(0).to(args.device)).cpu().numpy()[0, 0]
    sample_detections = decode_heatmap(sample_pred, spec, threshold=best_threshold)
    reconstructed_signals = reconstruct_signals_from_detections(
        test_dataset.simulator,
        sample_detections,
    )
    save_raw_cpmg_traces(sample["taus_us"].numpy(), sample["signals"].numpy(), out_dir / "raw_cpmg_traces.png")
    save_spin_scatter(sample["spins"].numpy(), spec, out_dir / "true_spin_scatter.png")
    save_ground_truth_heatmap(sample["heatmap"].numpy()[0], out_dir / "ground_truth_heatmap.png")
    plot_raw_traces_and_spins(
        sample["taus_us"].numpy(),
        sample["signals"].numpy(),
        sample["spins"].numpy(),
        spec,
        out_dir / "raw_traces_and_spins.png",
    )
    plot_heatmap_comparison(sample["heatmap"].numpy()[0], sample_pred, out_dir / "heatmap_comparison.png")
    plot_signal_reconstruction_overlay(
        sample["taus_us"].numpy(),
        sample["signals"].numpy(),
        reconstructed_signals,
        out_dir / "identified_signal_overlay.png",
    )
    plot_loss_history(history, out_dir / "loss_curve.png")
    plot_metric_history(history, out_dir / "metric_curve.png")
    write_csv(out_dir / "threshold_sweep.csv", sweep_rows)
    save_threshold_sweep(sweep_rows, out_dir / "threshold_sweep.png")

    metrics_payload = {
        "samples": int(args.test_samples),
        "threshold": best_threshold,
        "tp": float(test_metrics["tp"]),
        "fp": float(test_metrics["fp"]),
        "fn": float(test_metrics["fn"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "matched_mae_khz": finite_or_none(test_metrics["mae_khz"]),
    }
    (out_dir / "metrics.json").write_text(strict_json(metrics_payload))
    summary = {
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "pos_weight": args.pos_weight,
        "dice_weight": args.dice_weight,
        "metric_threshold_during_training": args.metric_threshold,
        "best_validation_threshold": best_threshold,
        "best_validation_f1": f1_score(best_row),
        "test_metrics": metrics_payload,
        "prediction_min": float(test_heatmaps.min()),
        "prediction_max": float(test_heatmaps.max()),
        "prediction_mean": float(test_heatmaps.mean()),
        "sample_detected_c13_count": len(sample_detections),
    }
    (out_dir / "summary.json").write_text(strict_json(summary))
    return summary


def main() -> None:
    args, _unknown = build_parser().parse_known_args()
    archive = Path(args.archive)
    project_root = Path(args.project_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir("/content")
    extract_project(archive, project_root)
    os.chdir(project_root)
    run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev]"], cwd=project_root)
    args.device = resolve_device(args.device)
    if not args.skip_tests:
        run([sys.executable, "-m", "pytest", "-q"], cwd=project_root)

    checkpoint = out_dir / "checkpoint.pt"
    history = out_dir / "history.csv"
    run(
        [
            sys.executable,
            "-m",
            "sali_pycce.train",
            "--preset",
            args.preset,
            "--train-samples",
            str(args.train_samples),
            "--val-samples",
            str(args.val_samples),
            "--test-samples",
            str(args.test_samples),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--signal-points",
            "4000",
            "--heatmap-height",
            "128",
            "--heatmap-width",
            "256",
            "--b-gauss",
            "525",
            "--tau-start-us",
            "0",
            "--tau-stop-us",
            "40",
            "--t2-us",
            "800",
            "--shots",
            "1000",
            "--loss",
            args.loss,
            "--pos-weight",
            str(args.pos_weight),
            "--dice-weight",
            str(args.dice_weight),
            "--metric-threshold",
            str(args.metric_threshold),
            "--match-distance-khz",
            str(args.match_distance_khz),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--out",
            str(checkpoint),
            "--history-out",
            str(history),
        ],
        cwd=project_root,
    )
    summary = generate_report(args, project_root, out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

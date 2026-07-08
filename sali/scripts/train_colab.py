#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


def _ensure_writable_plot_cache() -> None:
    cache_root = Path(tempfile.gettempdir())
    mpl_config = cache_root / "sali-matplotlib"
    xdg_cache = cache_root / "sali-cache"
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


_ensure_writable_plot_cache()

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", Path("/content/sali/src")):
    if (candidate / "sali").exists():
        sys.path.insert(0, str(candidate))
        break

from sali.config import RunConfig, paper_config, practical_config
from sali.data import (
    DataSplits,
    NormalizationStats,
    estimate_normalization_stats,
    generate_splits,
    iter_streamed_samples,
    materialize_streamed_samples,
)
from sali.diagnostics import DEFAULT_THRESHOLDS, diagnose_split, diagnose_splits, select_threshold
from sali.metrics import aggregate_by_true_count
from sali.physics import Couplings, generate_sample_signals
from sali.plots import (
    plot_heatmap,
    plot_loss,
    plot_mae,
    plot_mae_by_nuclei,
    plot_precision_recall,
    plot_signal_overlay,
    plot_spectra,
)
from sali.postprocess import postprocess_heatmap
from sali.shards import generate_shards, iter_sharded_samples, materialize_sharded_samples
from sali.train import (
    TrainResult,
    evaluate_model,
    evaluate_sample_iterable,
    train_model,
    train_sharded_model,
    train_streamed_model,
)


DRIVE_MYDRIVE = Path("/content/drive/MyDrive")


def default_data_mode(preset: str) -> str:
    if preset == "paper":
        return "sharded"
    if preset == "practical":
        return "materialized"
    raise ValueError("preset must be either 'practical' or 'paper'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the SALI PyTorch reproduction.")
    parser.add_argument("--preset", choices=["practical", "paper"], default="practical")
    parser.add_argument("--field", choices=["low", "high"], default="low")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--data-mode", choices=["materialized", "stream", "sharded"], default=None)
    parser.add_argument("--resume", default=None, help="Path to a streamed training checkpoint, or 'latest'.")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--calibrate-every-epochs", type=int, default=0)
    parser.add_argument("--sample-plots-every-epochs", type=int, default=0)
    parser.add_argument("--normalization-samples", type=int, default=10_000)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--generate-shards", action="store_true")
    parser.add_argument("--shard-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--raw-signal-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--cache-dataset-dir", type=Path, default=None)
    parser.add_argument("--skip-existing-shards", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-test-eval", action="store_true")
    parser.add_argument("--no-final-eval", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="Limit train examples consumed per epoch; default uses the full train split.",
    )
    parser.add_argument("--loss-type", choices=["mse", "weighted_mse", "weighted_bce"], default=None)
    parser.add_argument("--positive-weight", type=float, default=None)
    parser.add_argument("--border-penalty-weight", type=float, default=None)
    parser.add_argument("--border-width", type=int, default=None)
    parser.add_argument("--threshold-mode", choices=["fixed", "calibrate"], default="fixed")
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="Comma-separated post-processing thresholds to sweep for diagnostics/calibration.",
    )
    parser.add_argument("--diagnostic-samples", type=int, default=64)
    parser.add_argument("--disable-diagnostic-morphology", action="store_true")
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[float]:
    thresholds = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("at least one threshold is required")
    if any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("thresholds must be between 0 and 1")
    return thresholds


def config_from_args(args: argparse.Namespace) -> RunConfig:
    cfg = practical_config(args.field) if args.preset == "practical" else paper_config(args.field)
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.train_samples is not None:
        cfg.data.train_samples = args.train_samples
    if args.val_samples is not None:
        cfg.data.val_samples = args.val_samples
    if args.test_samples is not None:
        cfg.data.test_samples = args.test_samples
    if args.epochs is not None:
        cfg.training.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.samples_per_epoch is not None:
        cfg.training.samples_per_epoch = args.samples_per_epoch
    if args.loss_type is not None:
        cfg.training.loss_type = args.loss_type
    if args.positive_weight is not None:
        cfg.training.positive_weight = args.positive_weight
    if args.border_penalty_weight is not None:
        cfg.training.border_penalty_weight = args.border_penalty_weight
    if args.border_width is not None:
        cfg.training.border_width = args.border_width
    cfg.training.device = args.device
    return cfg


def apply_run_root(cfg: RunConfig, run_root: Path | None) -> None:
    if run_root is None:
        return
    if str(run_root).startswith("/content/drive") and not DRIVE_MYDRIVE.exists():
        raise RuntimeError(
            "Google Drive is not mounted in this Colab runtime. "
            "Expected /content/drive/MyDrive before writing durable SALI artifacts."
        )
    run_root.mkdir(parents=True, exist_ok=True)
    if not cfg.output_dir.is_absolute():
        cfg.output_dir = run_root / cfg.output_dir


def resolve_dataset_dir(cfg: RunConfig, args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir
    else:
        dataset_dir = Path(
            f"datasets/{args.preset}-{args.field}-"
            f"{cfg.data.train_samples}-{cfg.data.val_samples}-{cfg.data.test_samples}"
        )
    if args.cache_dataset_dir is not None:
        dataset_dir = args.cache_dataset_dir
    elif args.run_root is not None and not dataset_dir.is_absolute():
        dataset_dir = args.run_root / dataset_dir
    if str(dataset_dir).startswith("/content/drive") and not DRIVE_MYDRIVE.exists():
        raise RuntimeError(
            "Google Drive is not mounted in this Colab runtime. "
            "Expected /content/drive/MyDrive before writing durable SALI dataset shards."
        )
    return dataset_dir


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def make_example_plots_for_sample(cfg: RunConfig, sample, model: torch.nn.Module, figures: Path) -> None:
    plot_spectra(sample.raw_signals, figures / "generated_spectra.png")
    plot_heatmap(sample.heatmap, "True heatmap", figures / "true_heatmap.png")

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.from_numpy(sample.signals[0:1]).unsqueeze(0).to(device),
            torch.from_numpy(sample.signals[1:2]).unsqueeze(0).to(device),
        ).cpu().numpy()[0]
    plot_heatmap(pred, "Predicted heatmap", figures / "predicted_heatmap.png")

    predictions = postprocess_heatmap(pred, cfg.data, cfg.model, cfg.postprocess)
    post = np.zeros_like(pred)
    for item in predictions:
        row = int(round(item.row))
        col = int(round(item.col))
        post[0, max(0, row - 2) : row + 3, max(0, col - 2) : col + 3] = 1.0
    plot_heatmap(post, "Post-processed heatmap", figures / "postprocessed_heatmap.png")

    pred_couplings = Couplings(
        az_khz=np.array([item.az_khz for item in predictions], dtype=np.float32),
        aperp_khz=np.array([item.aperp_khz for item in predictions], dtype=np.float32),
    )
    clean_physics = replace(cfg.physics, add_shot_noise=False)
    reconstructed = generate_sample_signals(
        pred_couplings,
        clean_physics,
        np.random.default_rng(cfg.data.seed + 202),
    )
    plot_signal_overlay(sample.raw_signals, reconstructed, figures / "signal_overlay.png")


def make_example_plots(cfg: RunConfig, splits: DataSplits, result: TrainResult) -> None:
    make_example_plots_for_sample(cfg, splits.test[0], result.model, cfg.output_dir / "figures")


def streamed_sample_views(
    cfg: RunConfig,
    stats: NormalizationStats,
    max_samples: int,
) -> DataSplits:
    return DataSplits(
        train=materialize_streamed_samples(cfg, "train", stats, max_samples=max_samples),
        val=materialize_streamed_samples(cfg, "val", stats, max_samples=max_samples),
        test=materialize_streamed_samples(cfg, "test", stats, max_samples=max_samples),
    )


def sharded_sample_views(
    cfg: RunConfig,
    dataset_dir: Path,
    max_samples: int,
) -> DataSplits:
    return DataSplits(
        train=materialize_sharded_samples(cfg, dataset_dir, "train", max_samples=max_samples),
        val=materialize_sharded_samples(cfg, dataset_dir, "val", max_samples=max_samples),
        test=materialize_sharded_samples(cfg, dataset_dir, "test", max_samples=max_samples),
    )


def maybe_run_periodic_stream_artifacts(
    *,
    cfg: RunConfig,
    stats: NormalizationStats,
    thresholds: list[float],
    epoch: int,
    model: torch.nn.Module,
    history: dict[str, list[float]],
    calibrate_every_epochs: int,
    sample_plots_every_epochs: int,
    diagnostic_samples: int,
    threshold_mode: str,
    disable_morphology: bool,
) -> None:
    if calibrate_every_epochs > 0 and epoch % calibrate_every_epochs == 0:
        val_samples = materialize_streamed_samples(cfg, "val", stats, max_samples=diagnostic_samples)
        diagnostics = diagnose_split(
            model,
            val_samples,
            cfg,
            thresholds=thresholds,
            disable_morphology=disable_morphology,
        )
        save_json(cfg.output_dir / f"diagnostics_epoch_{epoch:04d}.json", {"val": diagnostics})
        if threshold_mode == "calibrate":
            selected = select_threshold(diagnostics["threshold_sweep"])
            cfg.postprocess.threshold = float(selected["threshold"])
            save_json(cfg.output_dir / f"selected_threshold_epoch_{epoch:04d}.json", {"mode": "calibrate", **selected})
    if sample_plots_every_epochs > 0 and epoch % sample_plots_every_epochs == 0:
        sample = materialize_streamed_samples(cfg, "test", stats, max_samples=1)[0]
        make_example_plots_for_sample(cfg, sample, model, cfg.output_dir / "figures" / f"epoch_{epoch:04d}")
        plot_loss(history, cfg.output_dir / "figures" / f"loss_epoch_{epoch:04d}.png")


def maybe_run_periodic_sharded_artifacts(
    *,
    cfg: RunConfig,
    dataset_dir: Path,
    thresholds: list[float],
    epoch: int,
    model: torch.nn.Module,
    history: dict[str, list[float]],
    calibrate_every_epochs: int,
    sample_plots_every_epochs: int,
    diagnostic_samples: int,
    threshold_mode: str,
    disable_morphology: bool,
) -> None:
    if calibrate_every_epochs > 0 and epoch % calibrate_every_epochs == 0:
        val_samples = materialize_sharded_samples(cfg, dataset_dir, "val", max_samples=diagnostic_samples)
        diagnostics = diagnose_split(
            model,
            val_samples,
            cfg,
            thresholds=thresholds,
            disable_morphology=disable_morphology,
        )
        save_json(cfg.output_dir / f"diagnostics_epoch_{epoch:04d}.json", {"val": diagnostics})
        if threshold_mode == "calibrate":
            selected = select_threshold(diagnostics["threshold_sweep"])
            cfg.postprocess.threshold = float(selected["threshold"])
            save_json(cfg.output_dir / f"selected_threshold_epoch_{epoch:04d}.json", {"mode": "calibrate", **selected})
    if sample_plots_every_epochs > 0 and epoch % sample_plots_every_epochs == 0:
        sample = materialize_sharded_samples(cfg, dataset_dir, "test", max_samples=1)[0]
        make_example_plots_for_sample(cfg, sample, model, cfg.output_dir / "figures" / f"epoch_{epoch:04d}")
        plot_loss(history, cfg.output_dir / "figures" / f"loss_epoch_{epoch:04d}.png")


def run_materialized(cfg: RunConfig, args: argparse.Namespace, thresholds: list[float]) -> None:
    splits, stats = generate_splits(cfg)
    save_json(cfg.output_dir / "normalization.json", asdict(stats))

    result = train_model(cfg, splits)
    plot_loss(result.history, cfg.output_dir / "figures" / "loss.png")

    diagnostics = diagnose_splits(
        result.model,
        splits,
        cfg,
        thresholds=thresholds,
        max_samples=args.diagnostic_samples,
        disable_morphology=args.disable_diagnostic_morphology,
    )
    save_json(cfg.output_dir / "diagnostics.json", diagnostics)
    selected_threshold = {"mode": args.threshold_mode, "threshold": cfg.postprocess.threshold}
    if args.threshold_mode == "calibrate":
        selected = select_threshold(diagnostics["val"]["threshold_sweep"])
        cfg.postprocess.threshold = float(selected["threshold"])
        selected_threshold = {"mode": "calibrate", **selected}
    save_json(cfg.output_dir / "selected_threshold.json", selected_threshold)

    if not args.no_final_eval:
        metrics = evaluate_model(result.model, cfg, splits.test, max_samples=args.max_eval_samples)
        summary = aggregate_by_true_count(metrics)
        save_json(cfg.output_dir / "metrics_by_true_count.json", summary)
        plot_precision_recall(metrics, cfg.output_dir / "figures" / "precision_recall.png")
        plot_mae(metrics, cfg.output_dir / "figures" / "mae.png")
        plot_mae_by_nuclei(metrics, cfg.output_dir / "figures" / "mae_by_nuclei.png")
    make_example_plots(cfg, splits, result)


def run_streamed(cfg: RunConfig, args: argparse.Namespace, thresholds: list[float]) -> None:
    stats = estimate_normalization_stats(cfg, sample_limit=args.normalization_samples)
    save_json(cfg.output_dir / "normalization.json", asdict(stats))

    def epoch_callback(epoch: int, model: torch.nn.Module, history: dict[str, list[float]]) -> None:
        maybe_run_periodic_stream_artifacts(
            cfg=cfg,
            stats=stats,
            thresholds=thresholds,
            epoch=epoch,
            model=model,
            history=history,
            calibrate_every_epochs=args.calibrate_every_epochs,
            sample_plots_every_epochs=args.sample_plots_every_epochs,
            diagnostic_samples=args.diagnostic_samples,
            threshold_mode=args.threshold_mode,
            disable_morphology=args.disable_diagnostic_morphology,
        )

    result = train_streamed_model(
        cfg,
        stats,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        resume_from=args.resume,
        num_workers=args.num_workers,
        epoch_callback=epoch_callback,
    )
    plot_loss(result.history, cfg.output_dir / "figures" / "loss.png")

    diagnostic_views = streamed_sample_views(cfg, stats, args.diagnostic_samples)
    diagnostics = diagnose_splits(
        result.model,
        diagnostic_views,
        cfg,
        thresholds=thresholds,
        max_samples=None,
        disable_morphology=args.disable_diagnostic_morphology,
    )
    save_json(cfg.output_dir / "diagnostics.json", diagnostics)
    selected_threshold = {"mode": args.threshold_mode, "threshold": cfg.postprocess.threshold}
    if args.threshold_mode == "calibrate":
        selected = select_threshold(diagnostics["val"]["threshold_sweep"])
        cfg.postprocess.threshold = float(selected["threshold"])
        selected_threshold = {"mode": "calibrate", **selected}
    save_json(cfg.output_dir / "selected_threshold.json", selected_threshold)

    if not args.no_final_eval:
        if args.full_test_eval:
            metrics = evaluate_sample_iterable(result.model, cfg, iter_streamed_samples(cfg, "test", stats))
        else:
            test_samples = materialize_streamed_samples(cfg, "test", stats, max_samples=args.max_eval_samples)
            metrics = evaluate_model(result.model, cfg, test_samples)
        summary = aggregate_by_true_count(metrics)
        save_json(cfg.output_dir / "metrics_by_true_count.json", summary)
        plot_precision_recall(metrics, cfg.output_dir / "figures" / "precision_recall.png")
        plot_mae(metrics, cfg.output_dir / "figures" / "mae.png")
        plot_mae_by_nuclei(metrics, cfg.output_dir / "figures" / "mae_by_nuclei.png")
    sample = materialize_streamed_samples(cfg, "test", stats, max_samples=1)[0]
    make_example_plots_for_sample(cfg, sample, result.model, cfg.output_dir / "figures")


def run_sharded(cfg: RunConfig, args: argparse.Namespace, thresholds: list[float]) -> None:
    dataset_dir = resolve_dataset_dir(cfg, args)
    if args.generate_shards:
        manifest = generate_shards(
            cfg,
            dataset_dir,
            shard_size=args.shard_size,
            normalization_samples=args.normalization_samples,
            shard_dtype=args.shard_dtype,
            raw_signal_dtype=args.raw_signal_dtype,
            skip_existing=args.skip_existing_shards,
        )
        save_json(
            cfg.output_dir / "dataset_manifest.json",
            {
                "dataset_dir": str(dataset_dir),
                "config_hash": manifest.config_hash,
                "shard_count": len(manifest.shards),
            },
        )

    def epoch_callback(epoch: int, model: torch.nn.Module, history: dict[str, list[float]]) -> None:
        maybe_run_periodic_sharded_artifacts(
            cfg=cfg,
            dataset_dir=dataset_dir,
            thresholds=thresholds,
            epoch=epoch,
            model=model,
            history=history,
            calibrate_every_epochs=args.calibrate_every_epochs,
            sample_plots_every_epochs=args.sample_plots_every_epochs,
            diagnostic_samples=args.diagnostic_samples,
            threshold_mode=args.threshold_mode,
            disable_morphology=args.disable_diagnostic_morphology,
        )

    result = train_sharded_model(
        cfg,
        dataset_dir,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        resume_from=args.resume,
        num_workers=args.num_workers,
        epoch_callback=epoch_callback,
    )
    plot_loss(result.history, cfg.output_dir / "figures" / "loss.png")

    diagnostic_views = sharded_sample_views(cfg, dataset_dir, args.diagnostic_samples)
    diagnostics = diagnose_splits(
        result.model,
        diagnostic_views,
        cfg,
        thresholds=thresholds,
        max_samples=None,
        disable_morphology=args.disable_diagnostic_morphology,
    )
    save_json(cfg.output_dir / "diagnostics.json", diagnostics)
    selected_threshold = {"mode": args.threshold_mode, "threshold": cfg.postprocess.threshold}
    if args.threshold_mode == "calibrate":
        selected = select_threshold(diagnostics["val"]["threshold_sweep"])
        cfg.postprocess.threshold = float(selected["threshold"])
        selected_threshold = {"mode": "calibrate", **selected}
    save_json(cfg.output_dir / "selected_threshold.json", selected_threshold)

    if not args.no_final_eval:
        if args.full_test_eval:
            metrics = evaluate_sample_iterable(result.model, cfg, iter_sharded_samples(cfg, dataset_dir, "test"))
        else:
            test_samples = materialize_sharded_samples(cfg, dataset_dir, "test", max_samples=args.max_eval_samples)
            metrics = evaluate_model(result.model, cfg, test_samples)
        summary = aggregate_by_true_count(metrics)
        save_json(cfg.output_dir / "metrics_by_true_count.json", summary)
        plot_precision_recall(metrics, cfg.output_dir / "figures" / "precision_recall.png")
        plot_mae(metrics, cfg.output_dir / "figures" / "mae.png")
        plot_mae_by_nuclei(metrics, cfg.output_dir / "figures" / "mae_by_nuclei.png")
    sample = materialize_sharded_samples(cfg, dataset_dir, "test", max_samples=1)[0]
    make_example_plots_for_sample(cfg, sample, result.model, cfg.output_dir / "figures")


def main() -> None:
    args = parse_args()
    data_mode = args.data_mode or default_data_mode(args.preset)
    cfg = config_from_args(args)
    apply_run_root(cfg, args.run_root)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)
    save_json(cfg.output_dir / "config.json", {"data_mode": data_mode, **asdict(cfg)})
    if data_mode == "materialized":
        run_materialized(cfg, args, thresholds)
    elif data_mode == "stream":
        run_streamed(cfg, args, thresholds)
    else:
        run_sharded(cfg, args, thresholds)
    print(f"Run complete: {cfg.output_dir}")


if __name__ == "__main__":
    main()

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
from sali.data import DataSplits, generate_splits
from sali.metrics import aggregate_by_true_count
from sali.physics import Couplings, generate_sample_signals
from sali.plots import (
    plot_heatmap,
    plot_loss,
    plot_mae,
    plot_precision_recall,
    plot_signal_overlay,
    plot_spectra,
)
from sali.postprocess import postprocess_heatmap
from sali.train import TrainResult, evaluate_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the SALI PyTorch reproduction.")
    parser.add_argument("--preset", choices=["practical", "paper"], default="practical")
    parser.add_argument("--field", choices=["low", "high"], default="low")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


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
    cfg.training.device = args.device
    return cfg


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def make_example_plots(cfg: RunConfig, splits: DataSplits, result: TrainResult) -> None:
    figures = cfg.output_dir / "figures"
    sample = splits.test[0]
    plot_spectra(sample.raw_signals, figures / "generated_spectra.png")
    plot_heatmap(sample.heatmap, "True heatmap", figures / "true_heatmap.png")

    device = next(result.model.parameters()).device
    result.model.eval()
    with torch.no_grad():
        pred = result.model(
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


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg.output_dir / "config.json", asdict(cfg))

    splits, stats = generate_splits(cfg)
    save_json(cfg.output_dir / "normalization.json", asdict(stats))

    result = train_model(cfg, splits)
    plot_loss(result.history, cfg.output_dir / "figures" / "loss.png")

    metrics = evaluate_model(result.model, cfg, splits.test, max_samples=args.max_eval_samples)
    summary = aggregate_by_true_count(metrics)
    save_json(cfg.output_dir / "metrics_by_true_count.json", summary)
    plot_precision_recall(metrics, cfg.output_dir / "figures" / "precision_recall.png")
    plot_mae(metrics, cfg.output_dir / "figures" / "mae.png")
    make_example_plots(cfg, splits, result)
    print(f"Run complete: {cfg.output_dir}")


if __name__ == "__main__":
    main()

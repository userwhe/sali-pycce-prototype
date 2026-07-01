from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from sali.config import RunConfig
from sali.data import DataSplits, SaliDataset, Sample
from sali.metrics import _match_predictions
from sali.model import SaliNet
from sali.postprocess import postprocess_heatmap
from sali.targets import coupling_to_pixel
from sali.train import choose_device


DEFAULT_THRESHOLDS: tuple[float, ...] = (0.25, 0.10, 0.075, 0.05, 0.035, 0.025, 0.015)


def _safe_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _threshold_metrics(
    samples: list[Sample],
    predictions: list[np.ndarray],
    cfg: RunConfig,
    threshold: float,
    *,
    disable_morphology: bool = False,
) -> dict[str, float]:
    pp = replace(cfg.postprocess, threshold=float(threshold))
    if disable_morphology:
        pp = replace(pp, min_area=1, erosion_size=0, dilation_size=0)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    predicted_nuclei = 0
    true_nuclei = 0
    for sample, prediction in zip(samples, predictions, strict=True):
        detected = postprocess_heatmap(prediction, cfg.data, cfg.model, pp)
        matches = _match_predictions(detected, sample.nuclei, cfg.data, cfg.model)
        tp = len(matches)
        fp = len(detected) - tp
        fn = sample.nuclei.count - tp
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        predicted_nuclei += len(detected)
        true_nuclei += sample.nuclei.count
    precision = 0.0 if true_positives + false_positives == 0 else true_positives / (true_positives + false_positives)
    recall = 0.0 if true_positives + false_negatives == 0 else true_positives / (true_positives + false_negatives)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": _f1(float(precision), float(recall)),
        "true_positives": float(true_positives),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
        "predicted_nuclei": float(predicted_nuclei),
        "true_nuclei": float(true_nuclei),
        "predictions_per_sample": float(predicted_nuclei / max(len(samples), 1)),
    }


def evaluate_predictions(
    samples: list[Sample],
    predictions: list[np.ndarray],
    cfg: RunConfig,
    *,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    disable_morphology: bool = False,
) -> dict[str, object]:
    if len(samples) != len(predictions):
        raise ValueError("samples and predictions must have the same length")
    if not samples:
        raise ValueError("samples must not be empty")

    model_mse_values: list[float] = []
    zero_mse_values: list[float] = []
    prediction_max_values: list[float] = []
    true_pixel_values: list[float] = []
    for sample, prediction in zip(samples, predictions, strict=True):
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape != sample.heatmap.shape:
            raise ValueError(f"prediction must have shape {sample.heatmap.shape}")
        if not np.isfinite(prediction).all():
            raise FloatingPointError("prediction contains non-finite values")
        model_mse_values.append(float(np.mean((prediction - sample.heatmap) ** 2)))
        zero_mse_values.append(float(np.mean(sample.heatmap**2)))
        prediction_max_values.append(float(np.max(prediction)))
        for az, aperp in zip(sample.nuclei.az_khz, sample.nuclei.aperp_khz, strict=True):
            row, col = coupling_to_pixel(float(az), float(aperp), cfg.data, cfg.model)
            true_pixel_values.append(float(prediction[0, row, col]))

    prediction_max_stats = _safe_stats(prediction_max_values)
    true_pixel_stats = _safe_stats(true_pixel_values)
    threshold_sweep = [
        _threshold_metrics(samples, predictions, cfg, threshold, disable_morphology=disable_morphology)
        for threshold in thresholds
    ]
    return {
        "samples": len(samples),
        "model_mse": float(np.mean(model_mse_values)),
        "zero_baseline_mse": float(np.mean(zero_mse_values)),
        "pred_max_min": prediction_max_stats["min"],
        "pred_max_median": prediction_max_stats["median"],
        "pred_max_p95": prediction_max_stats["p95"],
        "pred_max_max": prediction_max_stats["max"],
        "true_pixel_pred_min": true_pixel_stats["min"],
        "true_pixel_pred_median": true_pixel_stats["median"],
        "true_pixel_pred_p95": true_pixel_stats["p95"],
        "true_pixel_pred_max": true_pixel_stats["max"],
        "threshold_sweep": threshold_sweep,
    }


def select_threshold(threshold_sweep: list[dict[str, float]]) -> dict[str, float]:
    if not threshold_sweep:
        raise ValueError("threshold_sweep must not be empty")
    return max(
        threshold_sweep,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            float(row["recall"]),
            float(row["threshold"]),
        ),
    )


def predict_heatmaps(
    model: SaliNet,
    samples: list[Sample],
    cfg: RunConfig,
    *,
    batch_size: int | None = None,
) -> list[np.ndarray]:
    if not samples:
        return []
    device = choose_device(cfg.training.device)
    model.to(device)
    model.eval()
    loader = DataLoader(
        SaliDataset(samples),
        batch_size=batch_size or cfg.training.batch_size,
        shuffle=False,
    )
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for signal32, signal256, _target in loader:
            prediction = model(signal32.to(device), signal256.to(device)).cpu().numpy()
            predictions.extend(prediction)
    return predictions


def diagnose_split(
    model: SaliNet,
    samples: list[Sample],
    cfg: RunConfig,
    *,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    max_samples: int | None = None,
    disable_morphology: bool = False,
) -> dict[str, object]:
    selected = samples if max_samples is None else samples[:max_samples]
    predictions = predict_heatmaps(model, selected, cfg)
    return evaluate_predictions(
        selected,
        predictions,
        cfg,
        thresholds=thresholds,
        disable_morphology=disable_morphology,
    )


def diagnose_splits(
    model: SaliNet,
    splits: DataSplits,
    cfg: RunConfig,
    *,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    max_samples: int | None = None,
    disable_morphology: bool = False,
) -> dict[str, dict[str, object]]:
    return {
        "train": diagnose_split(
            model,
            splits.train,
            cfg,
            thresholds=thresholds,
            max_samples=max_samples,
            disable_morphology=disable_morphology,
        ),
        "val": diagnose_split(
            model,
            splits.val,
            cfg,
            thresholds=thresholds,
            max_samples=max_samples,
            disable_morphology=disable_morphology,
        ),
        "test": diagnose_split(
            model,
            splits.test,
            cfg,
            thresholds=thresholds,
            max_samples=max_samples,
            disable_morphology=disable_morphology,
        ),
    }

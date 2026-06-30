"""Shared spin-detection metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .heatmap import HeatmapSpec, decode_heatmap, nearest_match_errors


def compute_detection_metrics(
    truth: np.ndarray,
    pred: list[dict[str, float]],
    max_dist_khz: float = 5.0,
) -> dict[str, float]:
    truth_arr = np.asarray(truth, dtype=float).reshape(-1, 2)
    truth_arr = truth_arr[np.isfinite(truth_arr).all(axis=1)]
    counts = nearest_match_errors(truth_arr, pred, max_dist_khz=max_dist_khz)
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {
        **counts,
        "precision": float(precision),
        "recall": float(recall),
        "mae_khz": float(counts["mae_khz"]),
    }


def aggregate_detection_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    tp = float(sum(row["tp"] for row in rows))
    fp = float(sum(row["fp"] for row in rows))
    fn = float(sum(row["fn"] for row in rows))
    mae_weight = float(
        sum(
            row["mae_khz"] * row["tp"]
            for row in rows
            if row["tp"] > 0 and np.isfinite(row["mae_khz"])
        )
    )
    mae_tp = float(
        sum(
            row["tp"]
            for row in rows
            if row["tp"] > 0 and np.isfinite(row["mae_khz"])
        )
    )
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "mae_khz": mae_weight / mae_tp if mae_tp > 0 else float("nan"),
    }


def threshold_sweep(
    pred_heatmaps: np.ndarray,
    truth_spins: Sequence[np.ndarray],
    spec: HeatmapSpec,
    thresholds: Iterable[float],
    max_dist_khz: float = 5.0,
    min_area: int = 3,
    morph: bool = True,
) -> list[dict[str, float]]:
    heatmaps = np.asarray(pred_heatmaps)
    if heatmaps.ndim == 4:
        heatmaps = heatmaps[:, 0]
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        per_sample: list[dict[str, float]] = []
        for heatmap, truth in zip(heatmaps, truth_spins, strict=True):
            detections = decode_heatmap(
                heatmap,
                spec,
                threshold=float(threshold),
                min_area=min_area,
                morph=morph,
            )
            per_sample.append(
                compute_detection_metrics(truth, detections, max_dist_khz=max_dist_khz)
            )
        rows.append({"threshold": float(threshold), **aggregate_detection_metrics(per_sample)})
    return rows

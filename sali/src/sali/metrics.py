from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from sali.config import DataConfig, ModelConfig, PhysicsConfig
from sali.physics import Couplings, generate_sample_signals
from sali.postprocess import Prediction
from sali.targets import Box, true_boxes


_PROBABILITY_TOLERANCE = 1e-6
_MATCH_CARDINALITY_WEIGHT = 1_000_000.0
_CONFIDENCE_TIE_BREAK_WEIGHT = 1e-9


@dataclass(slots=True)
class SampleMetrics:
    true_nuclei: int
    predicted_nuclei: int
    true_positives: int
    false_positives: int
    false_negatives: int
    mae_az_khz: float
    mae_aperp_khz: float
    signal_mae_32: float
    signal_mae_256: float

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return 0.0 if denom == 0 else self.true_positives / denom

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return 0.0 if denom == 0 else self.true_positives / denom


def box_iou(a: Box, b: Box) -> float:
    row_min = max(a.row_min, b.row_min)
    col_min = max(a.col_min, b.col_min)
    row_max = min(a.row_max, b.row_max)
    col_max = min(a.col_max, b.col_max)
    if row_max < row_min or col_max < col_min:
        return 0.0
    intersection = (row_max - row_min + 1) * (col_max - col_min + 1)
    area_a = a.height * a.width
    area_b = b.height * b.width
    union = area_a + area_b - intersection
    return float(intersection / union)


def _match_predictions(predictions: list[Prediction], true: Couplings, data: DataConfig, model: ModelConfig) -> list[tuple[int, int]]:
    boxes = true_boxes(true, data, model)
    if not predictions or not boxes:
        return []
    scores = np.zeros((len(predictions), len(boxes)), dtype=np.float64)
    for pred_idx, pred in enumerate(predictions):
        for true_idx, box in enumerate(boxes):
            iou = box_iou(pred.box, box)
            if iou > 0.0:
                scores[pred_idx, true_idx] = (
                    _MATCH_CARDINALITY_WEIGHT
                    + iou
                    + (_CONFIDENCE_TIE_BREAK_WEIGHT * pred.confidence)
                )
    pred_indices, true_indices = linear_sum_assignment(scores, maximize=True)
    return [
        (int(pred_idx), int(true_idx))
        for pred_idx, true_idx in zip(pred_indices, true_indices, strict=True)
        if scores[pred_idx, true_idx] > 0.0
    ]


def _validate_raw_signals(raw_signals: np.ndarray, physics: PhysicsConfig) -> np.ndarray:
    signals = np.asarray(raw_signals, dtype=np.float32)
    expected_shape = (2, physics.signal_points)
    if signals.shape != expected_shape:
        raise ValueError(f"raw_signals must have shape {expected_shape}")
    if not np.isfinite(signals).all():
        raise FloatingPointError("raw_signals contains non-finite values")
    if signals.size > 0:
        min_value = float(np.min(signals))
        max_value = float(np.max(signals))
        if min_value < -_PROBABILITY_TOLERANCE or max_value > 1.0 + _PROBABILITY_TOLERANCE:
            raise FloatingPointError("raw_signals contains probability values outside [0, 1]")
    return np.clip(signals, 0.0, 1.0).astype(np.float32)


def evaluate_sample(
    predictions: list[Prediction],
    true: Couplings,
    raw_signals: np.ndarray,
    data: DataConfig,
    model: ModelConfig,
    physics: PhysicsConfig,
    rng: np.random.Generator,
) -> SampleMetrics:
    raw_signals = _validate_raw_signals(raw_signals, physics)
    matches = _match_predictions(predictions, true, data, model)
    tp = len(matches)
    fp = len(predictions) - tp
    fn = true.count - tp
    if matches:
        az_errors = [
            abs(predictions[pred_idx].az_khz - float(true.az_khz[true_idx]))
            for pred_idx, true_idx in matches
        ]
        aperp_errors = [
            abs(predictions[pred_idx].aperp_khz - float(true.aperp_khz[true_idx]))
            for pred_idx, true_idx in matches
        ]
        mae_az = float(np.mean(az_errors))
        mae_aperp = float(np.mean(aperp_errors))
    else:
        mae_az = float("nan")
        mae_aperp = float("nan")
    pred_couplings = Couplings(
        az_khz=np.array([pred.az_khz for pred in predictions], dtype=np.float32),
        aperp_khz=np.array([pred.aperp_khz for pred in predictions], dtype=np.float32),
    )
    clean_physics = PhysicsConfig(
        bz_tesla=physics.bz_tesla,
        gamma_mhz_per_t=physics.gamma_mhz_per_t,
        t2_us=physics.t2_us,
        measurements=physics.measurements,
        signal_points=physics.signal_points,
        tau_32_min_us=physics.tau_32_min_us,
        tau_32_max_us=physics.tau_32_max_us,
        tau_256_min_us=physics.tau_256_min_us,
        tau_256_max_us=physics.tau_256_max_us,
        add_decoherence=physics.add_decoherence,
        add_shot_noise=False,
    )
    reconstructed = generate_sample_signals(pred_couplings, clean_physics, rng)
    return SampleMetrics(
        true_nuclei=true.count,
        predicted_nuclei=len(predictions),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        mae_az_khz=mae_az,
        mae_aperp_khz=mae_aperp,
        signal_mae_32=float(np.mean(np.abs(reconstructed[0] - raw_signals[0]))),
        signal_mae_256=float(np.mean(np.abs(reconstructed[1] - raw_signals[1]))),
    )


def _finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def aggregate_by_true_count(results: list[SampleMetrics]) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[SampleMetrics]] = {}
    for result in results:
        grouped.setdefault(result.true_nuclei, []).append(result)
    summary: dict[int, dict[str, float]] = {}
    for count, items in grouped.items():
        summary[count] = {
            "precision": float(np.mean([item.precision for item in items])),
            "recall": float(np.mean([item.recall for item in items])),
            "mae_az_khz": _finite_mean([item.mae_az_khz for item in items]),
            "mae_aperp_khz": _finite_mean([item.mae_aperp_khz for item in items]),
            "signal_mae_32": float(np.mean([item.signal_mae_32 for item in items])),
            "signal_mae_256": float(np.mean([item.signal_mae_256 for item in items])),
        }
    return summary

"""Hyperfine tensor projection utilities."""

from __future__ import annotations

import numpy as np


def _to_khz(tensor: np.ndarray, units: str) -> np.ndarray:
    normalized = units.strip().lower()
    if normalized == "khz":
        return tensor.astype(float, copy=False)
    if normalized == "hz":
        return tensor.astype(float, copy=False) / 1_000.0
    if normalized == "mhz":
        return tensor.astype(float, copy=False) * 1_000.0
    if normalized in {"rad/us", "rad_per_us"}:
        return tensor.astype(float, copy=False) / (2.0 * np.pi * 1e-3)
    raise ValueError(f"Unsupported hyperfine tensor units: {units!r}")


def _unit_axis(axis: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    vec = np.asarray(axis, dtype=float)
    if vec.shape != (3,):
        raise ValueError(f"axis must have shape (3,), got {vec.shape}")
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("axis must be nonzero")
    return vec / norm


def project_hyperfine_tensor(
    tensor: np.ndarray,
    axis: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
    units: str = "kHz",
) -> tuple[float, float]:
    arr = _to_khz(np.asarray(tensor, dtype=float), units)
    if arr.shape != (3, 3):
        raise ValueError(f"tensor must have shape (3, 3), got {arr.shape}")
    b_hat = _unit_axis(axis)
    column = arr @ b_hat
    az = float(b_hat @ column)
    aperp_sq = float(column @ column - az * az)
    aperp = float(np.sqrt(max(aperp_sq, 0.0)))
    return az, aperp


def project_hyperfine_tensors(
    tensors: np.ndarray,
    axis: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
    units: str = "kHz",
) -> np.ndarray:
    arr = np.asarray(tensors, dtype=float)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(f"tensors must have shape (n, 3, 3), got {arr.shape}")
    return np.asarray(
        [project_hyperfine_tensor(tensor, axis=axis, units=units) for tensor in arr],
        dtype=np.float32,
    )

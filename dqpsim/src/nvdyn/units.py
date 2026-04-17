"""Unit helpers used across the package."""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ArrayLikeFloat: TypeAlias = float | NDArray[np.float64]

TAU = 2.0 * math.pi


def hz(value: float) -> float:
    """Return a frequency already expressed in Hz."""

    return float(value)


def khz(value: float) -> float:
    """Convert kHz to Hz."""

    return float(value) * 1e3


def mhz(value: float) -> float:
    """Convert MHz to Hz."""

    return float(value) * 1e6


def ghz(value: float) -> float:
    """Convert GHz to Hz."""

    return float(value) * 1e9


def ns(value: float) -> float:
    """Convert ns to seconds."""

    return float(value) * 1e-9


def us(value: float) -> float:
    """Convert us to seconds."""

    return float(value) * 1e-6


def ms(value: float) -> float:
    """Convert ms to seconds."""

    return float(value) * 1e-3


def angular_frequency(value_hz: ArrayLikeFloat) -> ArrayLikeFloat:
    """Convert cycles-per-second to angular frequency."""

    return TAU * value_hz


def as_float_array(values: ArrayLikeFloat) -> NDArray[np.float64]:
    """Coerce values into a one-dimensional float array."""

    return np.asarray(values, dtype=float)

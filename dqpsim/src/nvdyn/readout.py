"""Readout helpers for ideal and synthetic-count modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray


@dataclass(slots=True, frozen=True)
class CountsConfig:
    """Simple synthetic shot-noise configuration."""

    shots: int
    bright_probability: float = 1.0
    dark_probability: float = 0.0
    seed: int | None = None


def ideal_readout(expectation_trace: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the ideal expectation trace unchanged."""

    return np.asarray(expectation_trace, dtype=float)


def synthetic_counts(
    expectation_trace: NDArray[np.float64],
    config: CountsConfig,
    *,
    rng: Generator | None = None,
) -> NDArray[np.int64]:
    """Sample a simple count trace from a probability-like expectation trace."""

    generator = rng if rng is not None else default_rng(config.seed)
    probability = np.clip(np.asarray(expectation_trace, dtype=float), 0.0, 1.0)
    effective_probability = config.dark_probability + probability * (
        config.bright_probability - config.dark_probability
    )
    return generator.binomial(config.shots, np.clip(effective_probability, 0.0, 1.0))

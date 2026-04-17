"""Waveform models used by drive pulses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class Waveform(ABC):
    """Abstract base class for normalized envelopes."""

    duration: float

    @abstractmethod
    def value(self, time_in_pulse: float) -> float:
        """Return the normalized envelope value at ``time_in_pulse``."""

    def sample(self, dt: float) -> NDArray[np.float64]:
        """Sample the waveform on a left-endpoint grid."""

        if dt <= 0.0:
            raise ValueError("Sampling dt must be positive.")
        steps = max(1, int(np.ceil(self.duration / dt)))
        times = np.linspace(0.0, self.duration, steps, endpoint=False)
        return np.asarray([self.value(float(time)) for time in times], dtype=float)


AnalyticShape = Literal["square", "gaussian", "sine", "blackman"]


@dataclass(slots=True, frozen=True)
class AnalyticWaveform(Waveform):
    """Analytic waveform with a normalized envelope."""

    kind: AnalyticShape
    duration: float
    sigma: float | None = None
    cycles: float = 1.0

    def value(self, time_in_pulse: float) -> float:
        if time_in_pulse < 0.0 or time_in_pulse >= self.duration:
            return 0.0
        x = time_in_pulse / self.duration
        if self.kind == "square":
            return 1.0
        if self.kind == "gaussian":
            sigma = self.sigma if self.sigma is not None else 0.18
            return float(np.exp(-0.5 * ((x - 0.5) / sigma) ** 2))
        if self.kind == "sine":
            return float(np.sin(np.pi * self.cycles * x) ** 2)
        if self.kind == "blackman":
            return float(0.42 - 0.5 * np.cos(2.0 * np.pi * x) + 0.08 * np.cos(4.0 * np.pi * x))
        raise ValueError(f"Unsupported analytic waveform kind {self.kind!r}.")


@dataclass(slots=True, frozen=True)
class SampledWaveform(Waveform):
    """Sampled waveform with an explicit sample spacing."""

    samples: NDArray[np.float64]
    dt: float

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("Sampled waveforms need a positive dt.")
        if len(self.samples) == 0:
            raise ValueError("Sampled waveforms need at least one sample.")
        object.__setattr__(self, "samples", np.asarray(self.samples, dtype=float))

    @property
    def duration(self) -> float:
        return float(len(self.samples) * self.dt)

    def value(self, time_in_pulse: float) -> float:
        if time_in_pulse < 0.0 or time_in_pulse >= self.duration:
            return 0.0
        index = min(int(time_in_pulse / self.dt), len(self.samples) - 1)
        return float(self.samples[index])

    def sample(self, dt: float) -> NDArray[np.float64]:
        if abs(dt - self.dt) < 1e-18:
            return self.samples.copy()
        return super().sample(dt)

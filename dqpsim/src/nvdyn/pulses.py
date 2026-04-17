"""Pulse primitives."""

from __future__ import annotations

from dataclasses import dataclass

from .channels import DriveChannel
from .waveforms import AnalyticWaveform, Waveform


@dataclass(slots=True, frozen=True)
class Pulse:
    """Physical MW or RF pulse."""

    channel: DriveChannel
    amplitude: float
    carrier: float
    phase: float
    duration: float
    waveform: Waveform | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.duration <= 0.0:
            raise ValueError("pulse duration must be positive")
        waveform = self.waveform or AnalyticWaveform("square", self.duration)
        if abs(waveform.duration - self.duration) > 1e-15:
            raise ValueError("pulse duration must match waveform duration")
        object.__setattr__(self, "waveform", waveform)


def square_pulse(
    *,
    channel: DriveChannel,
    amplitude: float,
    carrier: float,
    phase: float = 0.0,
    duration: float,
    label: str = "",
) -> Pulse:
    return Pulse(
        channel=channel,
        amplitude=amplitude,
        carrier=carrier,
        phase=phase,
        duration=duration,
        waveform=AnalyticWaveform("square", duration),
        label=label,
    )

"""Canonical pulse schedule representation used across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .pulses import Pulse


@dataclass(slots=True, frozen=True)
class Delay:
    """Free evolution interval."""

    duration: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.duration < 0.0:
            raise ValueError("delay duration cannot be negative")


@dataclass(slots=True, frozen=True)
class FrameChange:
    """Virtual phase update applied to subsequent pulses on one target."""

    target: str
    phase: float
    label: str = ""


@dataclass(slots=True, frozen=True)
class Measure:
    """Measurement marker."""

    observable: str = "P_ms0"
    label: str = ""


ScheduleItem = Pulse | Delay | FrameChange | Measure


@dataclass(slots=True)
class PulseSchedule:
    """Sequential pulse schedule shared by sequences and circuits."""

    items: list[ScheduleItem] = field(default_factory=list)
    label: str = ""

    def append(self, item: ScheduleItem) -> "PulseSchedule":
        self.items.append(item)
        return self

    def extend(self, items: Iterable[ScheduleItem]) -> "PulseSchedule":
        self.items.extend(items)
        return self

    def pulse(self, item: Pulse) -> "PulseSchedule":
        return self.append(item)

    def delay(self, duration: float, *, label: str = "") -> "PulseSchedule":
        return self.append(Delay(duration=duration, label=label))

    def frame_change(self, *, target: str, phase: float, label: str = "") -> "PulseSchedule":
        return self.append(FrameChange(target=target, phase=phase, label=label))

    def measure(self, observable: str = "P_ms0", *, label: str = "") -> "PulseSchedule":
        return self.append(Measure(observable=observable, label=label))

    def copy(self, *, label: str | None = None) -> "PulseSchedule":
        return PulseSchedule(items=list(self.items), label=self.label if label is None else label)

    @property
    def total_duration(self) -> float:
        total = 0.0
        for item in self.items:
            if isinstance(item, (Pulse, Delay)):
                total += item.duration
        return total

    def timeline(self) -> list[tuple[float, ScheduleItem]]:
        current_time = 0.0
        entries: list[tuple[float, ScheduleItem]] = []
        for item in self.items:
            entries.append((current_time, item))
            if isinstance(item, (Pulse, Delay)):
                current_time += item.duration
        return entries

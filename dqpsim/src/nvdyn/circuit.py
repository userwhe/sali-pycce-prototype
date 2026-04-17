"""Lightweight circuit abstraction compiled into pulse schedules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

CircuitOpKind = Literal["rx", "ry", "rz", "delay", "swap", "native_entangle"]


@dataclass(slots=True, frozen=True)
class CircuitOperation:
    """One logical circuit operation."""

    kind: CircuitOpKind
    target: str | tuple[str, str] | None = None
    angle: float | None = None
    duration: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Circuit:
    """Logical circuit that lowers to the same pulse schedule as sequences."""

    operations: list[CircuitOperation] = field(default_factory=list)
    label: str = ""

    @property
    def name(self) -> str:
        return self.label

    def append(self, operation: CircuitOperation) -> "Circuit":
        self.operations.append(operation)
        return self

    def rx(self, target: str, angle: float, *, duration: float | None = None) -> "Circuit":
        return self.append(CircuitOperation("rx", target=target, angle=angle, duration=duration))

    def ry(self, target: str, angle: float, *, duration: float | None = None) -> "Circuit":
        return self.append(CircuitOperation("ry", target=target, angle=angle, duration=duration))

    def rz(self, target: str, angle: float) -> "Circuit":
        return self.append(CircuitOperation("rz", target=target, angle=angle))

    def delay(self, duration: float) -> "Circuit":
        return self.append(CircuitOperation("delay", duration=duration))

    def swap(self, *targets: str) -> "Circuit":
        if len(targets) == 1:
            pair: tuple[str, str] = ("electron", targets[0])
        elif len(targets) == 2:
            pair = (targets[0], targets[1])
        else:
            raise ValueError("swap expects one carbon target or an explicit two-qubit pair")
        return self.append(CircuitOperation("swap", target=pair))

    def native_entangle(
        self,
        left: str,
        right: str,
        *,
        duration: float | None = None,
        angle: float | None = None,
        **metadata: Any,
    ) -> "Circuit":
        payload = dict(metadata)
        return self.append(
            CircuitOperation(
                "native_entangle",
                target=(left, right),
                duration=duration,
                angle=angle,
                metadata=payload,
            )
        )

    def native_entangling(self, target: str, angle: float) -> "Circuit":
        return self.native_entangle("electron", target, angle=angle)

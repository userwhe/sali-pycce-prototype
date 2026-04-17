"""Simulation result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qutip import Qobj

from .readout import CountsConfig, synthetic_counts
from .schedule import PulseSchedule


@dataclass(slots=True)
class Result:
    """One simulator result."""

    times: NDArray[np.float64]
    schedule: PulseSchedule
    expectations: dict[str, NDArray[np.float64]]
    final_state: Qobj
    states: list[Qobj] = field(default_factory=list)
    observables: dict[str, Qobj] = field(default_factory=dict)
    measurements: dict[str, float] = field(default_factory=dict)
    readout: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def expectation(self, name: str) -> NDArray[np.float64]:
        return self.expectations[name]


@dataclass(slots=True)
class BatchResult:
    """Ordered list of simulator results."""

    results: list[Result]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SweepResult:
    """Ordered parameter scan result."""

    parameter_name: str
    values: NDArray[np.float64]
    results: list[Result]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def parameter_values(self) -> NDArray[np.float64]:
        return self.values

    def expectation_matrix(self, name: str) -> NDArray[np.float64]:
        return np.vstack([result.expectation(name) for result in self.results])


def result_from_qutip(
    *,
    qutip_result: Any,
    times: NDArray[np.float64],
    schedule: PulseSchedule,
    observables: dict[str, Qobj],
    measurements: tuple[tuple[float, str], ...],
    shots: int | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Result:
    if isinstance(qutip_result.expect, dict):
        expectations = {
            name: np.asarray(values, dtype=float)
            for name, values in qutip_result.expect.items()
        }
    else:
        names = list(observables.keys())
        expectations = {
            name: np.asarray(values, dtype=float)
            for name, values in zip(names, qutip_result.expect, strict=True)
        }

    final_state = getattr(qutip_result, "final_state", None)
    if final_state is None:
        final_state = qutip_result.states[-1]

    states = list(qutip_result.states) if getattr(qutip_result, "states", None) is not None else []
    result = Result(
        times=np.asarray(times, dtype=float),
        schedule=schedule,
        expectations=expectations,
        final_state=final_state,
        states=states,
        observables=observables,
        metadata=dict(metadata or {}),
    )

    for time_point, observable in measurements:
        if observable not in expectations:
            continue
        index = int(np.argmin(np.abs(result.times - time_point)))
        result.measurements[f"{observable}@{time_point:.9g}s"] = float(expectations[observable][index])

    if shots is not None:
        counts_cfg = CountsConfig(shots=shots, seed=seed)
        for name, trace in expectations.items():
            if name.startswith("P_"):
                result.readout[f"counts:{name}"] = synthetic_counts(trace, counts_cfg)

    return result

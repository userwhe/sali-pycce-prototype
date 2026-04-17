"""Common simulation utilities shared by the solver backends."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qutip import Qobj, QobjEvo

from ..hamiltonian import HamiltonianModel, build_hamiltonian_model
from ..schedule import Delay, FrameChange, Measure, PulseSchedule
from ..units import angular_frequency


@dataclass(slots=True)
class CompiledScheduleModel:
    """Pulse schedule compiled onto a fixed simulation grid."""

    schedule: PulseSchedule
    model: HamiltonianModel
    times: NDArray[np.float64]
    step: float
    hamiltonian: QobjEvo
    coefficients: dict[tuple[str, str], NDArray[np.float64]]


@dataclass(slots=True)
class SolverOutcome:
    """Minimal backend-independent solver output."""

    times: NDArray[np.float64]
    final_state: Qobj
    states: list[Qobj]
    expectations: dict[str, NDArray[np.float64]]


def _piecewise_constant(values: NDArray[np.float64], step: float) -> callable:
    def coefficient(time: float, args: dict[str, object] | None = None) -> complex:
        if len(values) == 1:
            return complex(values[0])
        clipped = min(max(time, 0.0), step * len(values) - 1e-15)
        index = min(int(math.floor(clipped / step)), len(values) - 1)
        return complex(values[index])

    return coefficient


def _time_grid(total_duration: float, dt: float) -> tuple[NDArray[np.float64], float]:
    if total_duration <= 0.0:
        return np.asarray([0.0], dtype=float), dt
    steps = max(1, int(np.ceil(total_duration / dt)))
    times = np.linspace(0.0, total_duration, steps + 1)
    return times, float(times[1] - times[0])


def compile_schedule_model(register, schedule: PulseSchedule, dt: float) -> CompiledScheduleModel:
    """Compile a pulse schedule into a piecewise-constant :class:`QobjEvo`."""

    model = build_hamiltonian_model(register)
    times, step = _time_grid(schedule.total_duration, dt)
    sample_times = times[:-1] if len(times) > 1 else np.asarray([0.0], dtype=float)
    coefficients = {
        key: np.zeros(len(sample_times), dtype=float)
        for key in model.control_ops
    }

    cursor = 0.0
    frame_phases: dict[str, float] = {"electron": 0.0}
    for carbon in register.carbons:
        frame_phases[carbon.name] = 0.0

    for item in schedule.items:
        if isinstance(item, Delay):
            cursor += item.duration
            continue
        if isinstance(item, FrameChange):
            frame_phases[item.target] = frame_phases.get(item.target, 0.0) + item.phase
            continue
        if isinstance(item, Measure):
            continue

        start = cursor
        stop = cursor + item.duration
        mask = (sample_times >= start - 1e-18) & (sample_times < stop - 1e-18)
        local_times = sample_times[mask] - start
        envelope = np.asarray([item.waveform.value(float(local_time)) for local_time in local_times], dtype=float)
        carrier_phase = angular_frequency(item.carrier) * local_times + item.phase
        for target in item.channel.targets:
            total_phase = carrier_phase + frame_phases.get(target, 0.0)
            drive_scale = angular_frequency(item.amplitude) * envelope
            coefficients[(target, "x")][mask] += drive_scale * np.cos(total_phase)
            coefficients[(target, "y")][mask] += drive_scale * np.sin(total_phase)
        cursor = stop

    terms: list[Qobj | list[object]] = [model.static]
    for key, values in coefficients.items():
        if np.any(np.abs(values) > 0.0):
            terms.append([model.control_ops[key], _piecewise_constant(values, step)])
    return CompiledScheduleModel(
        schedule=schedule,
        model=model,
        times=times,
        step=step,
        hamiltonian=QobjEvo(terms),
        coefficients=coefficients,
    )


def solver_outcome_from_qutip(result, observable_names: list[str]) -> SolverOutcome:
    """Convert a QuTiP solver result into the common backend format."""

    expectations = {
        name: np.asarray(values, dtype=float)
        for name, values in zip(observable_names, result.expect, strict=True)
    }
    states = list(result.states)
    return SolverOutcome(
        times=np.asarray(result.times, dtype=float),
        final_state=states[-1],
        states=states,
        expectations=expectations,
    )

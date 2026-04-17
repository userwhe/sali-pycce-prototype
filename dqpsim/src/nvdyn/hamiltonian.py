"""Hamiltonian assembly and pulse-schedule compilation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qutip import Qobj, QobjEvo

from .device import Register
from .operators import carbon_operator, control_qubit_operators, electron_operator
from .pulses import Pulse
from .schedule import Delay, FrameChange, Measure, PulseSchedule
from .units import angular_frequency


@dataclass(slots=True, frozen=True)
class HamiltonianModel:
    """Static Hamiltonian plus selective control operators."""

    static: Qobj
    control_ops: dict[tuple[str, str], Qobj]


@dataclass(slots=True, frozen=True)
class CompiledSchedule:
    tlist: NDArray[np.float64]
    hamiltonian: QobjEvo
    measurements: tuple[tuple[float, str], ...]


def hyperfine_term(register: Register, carbon_index: int | str) -> Qobj:
    """Return the hyperfine term for one carbon.

    The default path is the required secular approximation
    ``A_par * Sz * Iz + A_perp * Sx * Ix``.
    A supplied full tensor uses the advanced alternate bilinear path.
    """

    index = register.carbon_index(carbon_index)
    carbon = register.carbons[index]

    if carbon.uses_tensor_override:
        tensor_override = carbon.tensor_array
        assert tensor_override is not None
        term = 0.0 * electron_operator(register, "z")
        for row_axis, row in zip(("x", "y", "z"), tensor_override, strict=True):
            for col_axis, coefficient in zip(("x", "y", "z"), row, strict=True):
                term = term + angular_frequency(float(coefficient)) * (
                    electron_operator(register, row_axis)
                    * carbon_operator(register, index, col_axis)
                )
        return term

    return (
        angular_frequency(carbon.A_par)
        * electron_operator(register, "z")
        * carbon_operator(register, index, "z")
        + angular_frequency(carbon.A_perp)
        * electron_operator(register, "x")
        * carbon_operator(register, index, "x")
    )


def static_hamiltonian(register: Register) -> Qobj:
    """Construct the time-independent Hamiltonian in angular-frequency units."""

    electron = register.electron
    sx = electron_operator(register, "x")
    sy = electron_operator(register, "y")
    sz = electron_operator(register, "z")
    h0 = angular_frequency(electron.D) * (sz * sz)
    h0 = h0 + angular_frequency(electron.gamma * electron.B[0]) * sx
    h0 = h0 + angular_frequency(electron.gamma * electron.B[1]) * sy
    h0 = h0 + angular_frequency(electron.gamma * electron.B[2]) * sz

    for index, carbon in enumerate(register.carbons):
        h0 = h0 - angular_frequency(carbon.gamma * electron.B[0]) * carbon_operator(register, index, "x")
        h0 = h0 - angular_frequency(carbon.gamma * electron.B[1]) * carbon_operator(register, index, "y")
        h0 = h0 - angular_frequency(carbon.gamma * electron.B[2]) * carbon_operator(register, index, "z")
        h0 = h0 + hyperfine_term(register, index)
    return h0


def build_hamiltonian_model(register: Register) -> HamiltonianModel:
    controls: dict[tuple[str, str], Qobj] = {}
    electron_controls = control_qubit_operators(register, "electron")
    controls[("electron", "x")] = electron_controls["x"]
    controls[("electron", "y")] = electron_controls["y"]
    for carbon in register.carbons:
        ops = control_qubit_operators(register, carbon.name)
        controls[(carbon.name, "x")] = ops["x"]
        controls[(carbon.name, "y")] = ops["y"]
    return HamiltonianModel(static=static_hamiltonian(register), control_ops=controls)


def compile_schedule(register: Register, schedule: PulseSchedule, *, dt: float) -> CompiledSchedule:
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    model = build_hamiltonian_model(register)
    total_duration = schedule.total_duration
    if total_duration == 0.0:
        tlist = np.asarray([0.0, dt], dtype=float)
    else:
        tlist = np.arange(0.0, total_duration + dt, dt, dtype=float)
        if tlist[-1] < total_duration:
            tlist = np.append(tlist, total_duration)

    coeffs: dict[tuple[str, str], NDArray[np.float64]] = {
        key: np.zeros_like(tlist) for key in model.control_ops
    }
    measurements: list[tuple[float, str]] = []
    frames: dict[str, float] = {}

    for start, item in schedule.timeline():
        if isinstance(item, FrameChange):
            frames[item.target] = frames.get(item.target, 0.0) + item.phase
            continue
        if isinstance(item, Measure):
            measurements.append((start, item.observable))
            continue
        if isinstance(item, Delay):
            continue
        if isinstance(item, Pulse):
            stop = start + item.duration
            mask = (tlist >= start) & (tlist < stop)
            if not np.any(mask):
                mask = np.isclose(tlist, start)
            local_times = tlist[mask] - start
            envelope = np.asarray([item.waveform.value(float(time)) for time in local_times], dtype=float)
            for target in item.channel.targets:
                total_phase = item.phase + frames.get(target, 0.0)
                omega_t = angular_frequency(item.carrier) * tlist[mask] + total_phase
                coeffs[(target, "x")][mask] += angular_frequency(item.amplitude) * envelope * np.cos(omega_t)
                coeffs[(target, "y")][mask] += angular_frequency(item.amplitude) * envelope * np.sin(omega_t)
            continue
        raise TypeError(f"unsupported schedule item {type(item)!r}")

    terms: list[Qobj | list[object]] = [model.static]
    for key, operator in model.control_ops.items():
        if np.any(np.abs(coeffs[key]) > 0.0):
            terms.append([operator, coeffs[key]])
    return CompiledSchedule(
        tlist=tlist,
        hamiltonian=QobjEvo(terms, tlist=tlist, order=0),
        measurements=tuple(measurements),
    )

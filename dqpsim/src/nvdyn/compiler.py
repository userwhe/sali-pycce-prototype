"""Circuit to :class:`~nvdyn.schedule.PulseSchedule` lowering."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

from .circuit import Circuit, CircuitOperation
from .constants import DEFAULT_ELECTRON_RABI_HZ, DEFAULT_NUCLEAR_RABI_HZ
from .device import Register
from .schedule import Delay, FrameChange, PulseSchedule
from .sequences import _rotation_pulse


@dataclass(slots=True)
class CircuitCompiler:
    electron_amplitude: float = DEFAULT_ELECTRON_RABI_HZ
    nuclear_amplitude: float = DEFAULT_NUCLEAR_RABI_HZ

    def compile(self, register: Register, circuit: Circuit) -> PulseSchedule:
        schedule = PulseSchedule(label=circuit.label or "circuit")
        for operation in circuit.operations:
            if operation.kind in {"rx", "ry", "rz"}:
                self._append_rotation(register, schedule, operation)
            elif operation.kind == "delay":
                assert operation.duration is not None
                schedule.append(Delay(operation.duration, label="circuit-delay"))
            elif operation.kind == "native_entangle":
                left, right = self._pair(operation.target)
                schedule.extend(
                    self._native_entangling_block(
                        register,
                        left,
                        right,
                        angle=operation.angle,
                        duration=operation.duration,
                    ).items
                )
            elif operation.kind == "swap":
                left, right = self._pair(operation.target)
                schedule.extend(self._swap_block(register, left, right).items)
            else:
                raise TypeError(f"unsupported circuit operation kind {operation.kind!r}")
        return schedule

    def _append_rotation(self, register: Register, schedule: PulseSchedule, operation: CircuitOperation) -> None:
        assert isinstance(operation.target, str)
        if operation.kind == "rz":
            assert operation.angle is not None
            schedule.append(FrameChange(target=operation.target, phase=operation.angle, label="virtual-z"))
            return

        amplitude = self.electron_amplitude if operation.target == "electron" else self.nuclear_amplitude
        carrier = (
            register.default_electron_transition_hz()
            if operation.target == "electron"
            else register.default_nuclear_transition_hz(operation.target)
        )
        phase = 0.0 if operation.kind == "rx" else pi / 2.0
        if operation.duration is not None and operation.angle is None:
            raise ValueError("duration-only circuit rotations are not supported in the MVP")
        assert operation.angle is not None
        schedule.append(
            _rotation_pulse(
                target=operation.target,
                angle=operation.angle,
                phase=phase,
                amplitude=amplitude,
                carrier=carrier,
                label=operation.kind,
            )
        )

    def _pair(self, target: str | tuple[str, str] | None) -> tuple[str, str]:
        if not isinstance(target, tuple) or len(target) != 2:
            raise ValueError("two-qubit circuit operations require a pair target")
        return target

    def _native_entangling_block(
        self,
        register: Register,
        left: str,
        right: str,
        *,
        angle: float | None,
        duration: float | None,
    ) -> PulseSchedule:
        carbon_target = right if left == "electron" else left
        carbon = register.carbons[register.carbon_index(carbon_target)]
        coupling = max(abs(carbon.A_par), abs(carbon.A_perp), 1.0)
        resolved_duration = duration if duration is not None else abs(angle if angle is not None else (pi / 2.0)) / (2.0 * pi * coupling)
        return PulseSchedule(label="native-entangling").append(Delay(resolved_duration, label=f"native-{carbon_target}"))

    def _swap_block(self, register: Register, left: str, right: str) -> PulseSchedule:
        carbon_target = right if left == "electron" else left
        schedule = PulseSchedule(label="swap")
        schedule.extend(self._native_entangling_block(register, left, right, angle=pi / 2.0, duration=None).items)
        schedule.append(FrameChange(target="electron", phase=pi / 2.0, label="swap-frame"))
        schedule.extend(self._native_entangling_block(register, left, right, angle=pi / 2.0, duration=None).items)
        schedule.append(FrameChange(target=carbon_target, phase=pi / 2.0, label="swap-frame"))
        schedule.extend(self._native_entangling_block(register, left, right, angle=pi / 2.0, duration=None).items)
        return schedule


def compile_circuit(register: Register, circuit: Circuit, compiler: CircuitCompiler | None = None) -> PulseSchedule:
    return (compiler or CircuitCompiler()).compile(register, circuit)


CompilerConfig = CircuitCompiler

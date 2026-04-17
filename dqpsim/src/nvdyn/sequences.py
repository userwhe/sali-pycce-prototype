"""Experiment-style pulse-sequence builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Any, Sequence

from .channels import DriveChannel, mw_channel, rf_channel
from .constants import DEFAULT_ELECTRON_RABI_HZ, DEFAULT_NUCLEAR_RABI_HZ
from .pulses import Pulse, square_pulse
from .schedule import PulseSchedule
from .waveforms import AnalyticWaveform, Waveform


def _channel_for_target(target: str) -> DriveChannel:
    return mw_channel() if target == "electron" else rf_channel(target=target)


def _resolve_frequency(*, carrier: float | None = None, frequency: float | None = None) -> float:
    resolved = carrier if carrier is not None else frequency
    return 0.0 if resolved is None else float(resolved)


def _duration_for_rotation(angle: float, amplitude: float) -> float:
    if amplitude <= 0.0:
        raise ValueError("drive amplitude must be positive")
    return abs(angle) / (2.0 * pi * amplitude)


def _rotation_pulse(
    *,
    target: str,
    angle: float,
    amplitude: float,
    carrier: float,
    phase: float,
    duration: float | None = None,
    waveform: Waveform | None = None,
    label: str,
) -> Pulse:
    resolved_duration = duration if duration is not None else _duration_for_rotation(angle, amplitude)
    resolved_waveform = waveform or AnalyticWaveform("square", resolved_duration)
    if abs(resolved_waveform.duration - resolved_duration) > 1e-15:
        raise ValueError("waveform duration must match the pulse duration")
    resolved_phase = phase if angle >= 0.0 else phase + pi
    return Pulse(
        channel=_channel_for_target(target),
        amplitude=amplitude,
        carrier=carrier,
        phase=resolved_phase,
        duration=resolved_duration,
        waveform=resolved_waveform,
        label=label,
    )


@dataclass(slots=True)
class Sequence:
    """Named wrapper around the canonical :class:`PulseSchedule`."""

    name: str
    schedule: PulseSchedule
    metadata: dict[str, Any] = field(default_factory=dict)

    def compile(self) -> PulseSchedule:
        return self.schedule.copy(label=self.name)

    def to_schedule(self) -> PulseSchedule:
        return self.compile()


def ODMR(
    *,
    target: str = "electron",
    carrier: float | None = None,
    frequency: float | None = None,
    amplitude: float = DEFAULT_ELECTRON_RABI_HZ,
    duration: float = 1.0e-6,
    phase: float = 0.0,
    waveform: Waveform | None = None,
) -> Sequence:
    resolved_frequency = _resolve_frequency(carrier=carrier, frequency=frequency)
    resolved_waveform = waveform or AnalyticWaveform("square", duration)
    schedule = PulseSchedule(label="ODMR")
    schedule.pulse(
        Pulse(
            channel=_channel_for_target(target),
            amplitude=amplitude,
            carrier=resolved_frequency,
            phase=phase,
            duration=duration,
            waveform=resolved_waveform,
            label="odmr-drive",
        )
    )
    schedule.measure()
    return Sequence("ODMR", schedule, {"target": target})


def ElectronRabi(
    *,
    duration: float,
    carrier: float | None = None,
    frequency: float | None = None,
    amplitude: float = DEFAULT_ELECTRON_RABI_HZ,
    phase: float = 0.0,
    waveform: Waveform | None = None,
) -> Sequence:
    return ODMR(
        target="electron",
        carrier=carrier,
        frequency=frequency,
        amplitude=amplitude,
        duration=duration,
        phase=phase,
        waveform=waveform,
    )


def NuclearRabi(
    *,
    target: str,
    duration: float,
    carrier: float | None = None,
    frequency: float | None = None,
    amplitude: float = DEFAULT_NUCLEAR_RABI_HZ,
    phase: float = 0.0,
    waveform: Waveform | None = None,
) -> Sequence:
    return ODMR(
        target=target,
        carrier=carrier,
        frequency=frequency,
        amplitude=amplitude,
        duration=duration,
        phase=phase,
        waveform=waveform,
    )


def Ramsey(
    *,
    target: str = "electron",
    tau: float,
    carrier: float | None = None,
    frequency: float | None = None,
    amplitude: float | None = None,
    final_phase: float = 0.0,
    pi_over_2_duration: float | None = None,
) -> Sequence:
    amp = amplitude if amplitude is not None else (
        DEFAULT_ELECTRON_RABI_HZ if target == "electron" else DEFAULT_NUCLEAR_RABI_HZ
    )
    resolved_frequency = _resolve_frequency(carrier=carrier, frequency=frequency)
    schedule = PulseSchedule(label="Ramsey")
    schedule.pulse(
        _rotation_pulse(
            target=target,
            angle=pi / 2.0,
            amplitude=amp,
            carrier=resolved_frequency,
            phase=0.0,
            duration=pi_over_2_duration,
            label="ramsey-pi2-a",
        )
    )
    schedule.delay(tau, label="ramsey-free")
    schedule.pulse(
        _rotation_pulse(
            target=target,
            angle=pi / 2.0,
            amplitude=amp,
            carrier=resolved_frequency,
            phase=final_phase,
            duration=pi_over_2_duration,
            label="ramsey-pi2-b",
        )
    )
    schedule.measure()
    return Sequence("Ramsey", schedule, {"target": target, "tau": tau})


def HahnEcho(
    *,
    target: str = "electron",
    tau: float,
    carrier: float | None = None,
    frequency: float | None = None,
    amplitude: float | None = None,
    refocus_phase: float = pi / 2.0,
    pi_over_2_duration: float | None = None,
    pi_duration: float | None = None,
) -> Sequence:
    amp = amplitude if amplitude is not None else (
        DEFAULT_ELECTRON_RABI_HZ if target == "electron" else DEFAULT_NUCLEAR_RABI_HZ
    )
    resolved_frequency = _resolve_frequency(carrier=carrier, frequency=frequency)
    schedule = PulseSchedule(label="HahnEcho")
    schedule.pulse(
        _rotation_pulse(
            target=target,
            angle=pi / 2.0,
            amplitude=amp,
            carrier=resolved_frequency,
            phase=0.0,
            duration=pi_over_2_duration,
            label="echo-pi2-a",
        )
    )
    schedule.delay(tau / 2.0, label="echo-half-a")
    schedule.pulse(
        _rotation_pulse(
            target=target,
            angle=pi,
            amplitude=amp,
            carrier=resolved_frequency,
            phase=refocus_phase,
            duration=pi_duration,
            label="echo-pi",
        )
    )
    schedule.delay(tau / 2.0, label="echo-half-b")
    schedule.pulse(
        _rotation_pulse(
            target=target,
            angle=pi / 2.0,
            amplitude=amp,
            carrier=resolved_frequency,
            phase=0.0,
            duration=pi_over_2_duration,
            label="echo-pi2-b",
        )
    )
    schedule.measure()
    return Sequence("HahnEcho", schedule, {"target": target, "tau": tau})


def CPMG(
    *,
    target: str = "electron",
    n: int,
    tau: float,
    carrier: float | None = None,
    frequency: float | None = None,
    amplitude: float | None = None,
    refocus_phase: float = pi / 2.0,
    pi_over_2_duration: float | None = None,
    pi_duration: float | None = None,
) -> Sequence:
    if n < 1:
        raise ValueError("CPMG requires n >= 1")
    amp = amplitude if amplitude is not None else (
        DEFAULT_ELECTRON_RABI_HZ if target == "electron" else DEFAULT_NUCLEAR_RABI_HZ
    )
    resolved_frequency = _resolve_frequency(carrier=carrier, frequency=frequency)
    schedule = PulseSchedule(label="CPMG")
    schedule.pulse(
        _rotation_pulse(
            target=target,
            angle=pi / 2.0,
            amplitude=amp,
            carrier=resolved_frequency,
            phase=0.0,
            duration=pi_over_2_duration,
            label="cpmg-pi2-a",
        )
    )
    for pulse_index in range(n):
        schedule.delay(tau / 2.0, label=f"cpmg-delay-a-{pulse_index}")
        schedule.pulse(
            _rotation_pulse(
                target=target,
                angle=pi,
                amplitude=amp,
                carrier=resolved_frequency,
                phase=refocus_phase,
                duration=pi_duration,
                label=f"cpmg-pi-{pulse_index}",
            )
        )
        schedule.delay(tau / 2.0, label=f"cpmg-delay-b-{pulse_index}")
    schedule.pulse(
        _rotation_pulse(
            target=target,
            angle=pi / 2.0,
            amplitude=amp,
            carrier=resolved_frequency,
            phase=0.0,
            duration=pi_over_2_duration,
            label="cpmg-pi2-b",
        )
    )
    schedule.measure()
    return Sequence("CPMG", schedule, {"target": target, "n": n, "tau": tau})


def DDRF(
    *,
    target: str,
    tau: float,
    blocks: int = 2,
    phase_table: Sequence[float] = (0.0, pi / 2.0),
    amplitude: float = DEFAULT_NUCLEAR_RABI_HZ,
    carrier: float | None = None,
    frequency: float | None = None,
    pi_duration: float | None = None,
) -> Sequence:
    resolved_frequency = _resolve_frequency(carrier=carrier, frequency=frequency)
    schedule = PulseSchedule(label="DDRF")
    for block_index in range(blocks):
        for step_index, phase in enumerate(phase_table):
            schedule.pulse(
                _rotation_pulse(
                    target=target,
                    angle=pi,
                    amplitude=amplitude,
                    carrier=resolved_frequency,
                    phase=phase,
                    duration=pi_duration,
                    label=f"ddrf-pi-{block_index}-{step_index}",
                )
            )
            schedule.delay(tau, label=f"ddrf-delay-{block_index}-{step_index}")
    schedule.measure()
    return Sequence("DDRF", schedule, {"target": target, "blocks": blocks, "phase_table": tuple(phase_table)})


def PulsePol(
    *,
    target: str,
    tau: float,
    cycles: int | None = None,
    blocks: int | None = None,
    phase_table: Sequence[float] | None = None,
    block_phases: Sequence[float] | None = None,
    amplitude: float | None = None,
    nuclear_amplitude: float | None = None,
    carrier: float | None = None,
    frequency: float | None = None,
    nuclear_frequency: float | None = None,
    electron_amplitude: float = DEFAULT_ELECTRON_RABI_HZ,
    electron_frequency: float | None = None,
    electron_carrier: float = 0.0,
) -> Sequence:
    resolved_blocks = cycles if cycles is not None else (blocks if blocks is not None else 1)
    phases = tuple(block_phases if block_phases is not None else (phase_table if phase_table is not None else (0.0, pi / 2.0, 0.0, pi / 2.0)))
    resolved_amplitude = nuclear_amplitude if nuclear_amplitude is not None else (amplitude if amplitude is not None else DEFAULT_NUCLEAR_RABI_HZ)
    resolved_frequency = (
        float(nuclear_frequency)
        if nuclear_frequency is not None
        else _resolve_frequency(carrier=carrier, frequency=frequency)
    )
    resolved_electron_frequency = (
        float(electron_frequency)
        if electron_frequency is not None
        else float(electron_carrier)
    )
    schedule = PulseSchedule(label="PulsePol")
    for block_index in range(resolved_blocks):
        for step_index, phase in enumerate(phases):
            schedule.pulse(
                _rotation_pulse(
                    target="electron",
                    angle=pi / 2.0,
                    amplitude=electron_amplitude,
                    carrier=resolved_electron_frequency,
                    phase=phase,
                    label=f"pulsepol-mw-{block_index}-{step_index}",
                )
            )
            schedule.delay(tau, label=f"pulsepol-gap-a-{block_index}-{step_index}")
            schedule.pulse(
                _rotation_pulse(
                    target=target,
                    angle=pi,
                    amplitude=resolved_amplitude,
                    carrier=resolved_frequency,
                    phase=phase,
                    label=f"pulsepol-rf-{block_index}-{step_index}",
                )
            )
            schedule.delay(tau, label=f"pulsepol-gap-b-{block_index}-{step_index}")
    schedule.measure()
    return Sequence("PulsePol", schedule, {"target": target, "cycles": resolved_blocks, "block_phases": phases})


def SWAP(
    *,
    carbon: str | None = None,
    target: str | None = None,
    electron_carrier: float | None = None,
    electron_frequency: float | None = None,
    nuclear_carrier: float | None = None,
    nuclear_frequency: float | None = None,
    electron_amplitude: float = DEFAULT_ELECTRON_RABI_HZ,
    nuclear_amplitude: float = DEFAULT_NUCLEAR_RABI_HZ,
    interaction_delay: float | None = None,
    interaction_time: float | None = None,
) -> Sequence:
    resolved_carbon = carbon if carbon is not None else target
    if resolved_carbon is None:
        raise ValueError("SWAP requires a carbon target")
    e_freq = _resolve_frequency(carrier=electron_carrier, frequency=electron_frequency)
    n_freq = _resolve_frequency(carrier=nuclear_carrier, frequency=nuclear_frequency)
    wait = interaction_delay if interaction_delay is not None else (interaction_time if interaction_time is not None else 1.0e-6)

    schedule = PulseSchedule(label="SWAP")
    schedule.pulse(
        _rotation_pulse(
            target="electron",
            angle=pi / 2.0,
            amplitude=electron_amplitude,
            carrier=e_freq,
            phase=0.0,
            label="swap-e-pi2-a",
        )
    )
    schedule.pulse(
        _rotation_pulse(
            target=resolved_carbon,
            angle=pi / 2.0,
            amplitude=nuclear_amplitude,
            carrier=n_freq,
            phase=pi / 2.0,
            label="swap-c-pi2-a",
        )
    )
    schedule.delay(wait, label="swap-wait-a")
    schedule.pulse(
        _rotation_pulse(
            target="electron",
            angle=pi,
            amplitude=electron_amplitude,
            carrier=e_freq,
            phase=pi / 2.0,
            label="swap-e-pi",
        )
    )
    schedule.delay(wait, label="swap-wait-b")
    schedule.pulse(
        _rotation_pulse(
            target=resolved_carbon,
            angle=pi / 2.0,
            amplitude=nuclear_amplitude,
            carrier=n_freq,
            phase=0.0,
            label="swap-c-pi2-b",
        )
    )
    schedule.pulse(
        _rotation_pulse(
            target="electron",
            angle=pi / 2.0,
            amplitude=electron_amplitude,
            carrier=e_freq,
            phase=pi / 2.0,
            label="swap-e-pi2-b",
        )
    )
    schedule.measure()
    return Sequence("SWAP", schedule, {"carbon": resolved_carbon})

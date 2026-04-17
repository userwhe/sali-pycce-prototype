from __future__ import annotations

import numpy as np

from nvdyn import NVElectronSpec, Register, Simulator
from nvdyn.channels import mw_channel
from nvdyn.pulses import Pulse
from nvdyn.schedule import Delay, FrameChange, PulseSchedule


def _constant_x_pulse(duration: float, amplitude: float) -> Pulse:
    return Pulse(
        channel=mw_channel(),
        amplitude=amplitude,
        carrier=0.0,
        phase=0.0,
        duration=duration,
    )


def _ramsey_schedule(tau: float, amplitude: float, detuning_hz: float) -> PulseSchedule:
    quarter_turn = 1.0 / (8.0 * amplitude)
    schedule = PulseSchedule(label="manual-ramsey")
    schedule.append(_constant_x_pulse(quarter_turn, amplitude))
    schedule.append(Delay(tau))
    schedule.append(FrameChange(target="electron", phase=2.0 * np.pi * detuning_hz * tau))
    schedule.append(_constant_x_pulse(quarter_turn, amplitude))
    schedule.measure("P_ms0")
    return schedule


def test_resonant_electron_rabi_oscillation() -> None:
    register = Register(electron=NVElectronSpec(D=0.0, B=(0.0, 0.0, 0.0)))
    amplitude = 0.25e6
    flip_duration = 1.0 / (4.0 * amplitude)
    return_duration = 1.0 / (2.0 * amplitude)
    simulator = Simulator(method="exact", dt=10e-9)

    flip = simulator.run(
        register,
        PulseSchedule().append(_constant_x_pulse(flip_duration, amplitude)).measure("P_ms0"),
        observables=["P_ms0"],
    )
    returned = simulator.run(
        register,
        PulseSchedule().append(_constant_x_pulse(return_duration, amplitude)).measure("P_ms0"),
        observables=["P_ms0"],
    )

    assert flip.expectation("P_ms0")[-1] < 0.2
    assert returned.expectation("P_ms0")[-1] > 0.8


def test_ramsey_phase_contrast_and_lindblad_dephasing() -> None:
    register = Register(electron=NVElectronSpec(D=0.0, B=(0.0, 0.0, 0.0)))
    amplitude = 0.25e6
    detuning = 0.25e6
    tau_zero = 0.0
    tau_pi = 1.0 / (2.0 * detuning)

    exact = Simulator(method="exact", dt=10e-9)
    no_phase = exact.run(register, _ramsey_schedule(tau_zero, amplitude, detuning), observables=["P_ms0"])
    pi_phase = exact.run(register, _ramsey_schedule(tau_pi, amplitude, detuning), observables=["P_ms0"])

    lindblad = Simulator(
        method="lindblad",
        dt=10e-9,
        lindblad={"T2": 0.3e-6},
    )
    damped = lindblad.run(register, _ramsey_schedule(tau_pi, amplitude, detuning), observables=["P_ms0"])

    no_phase_final = no_phase.expectation("P_ms0")[-1]
    pi_phase_final = pi_phase.expectation("P_ms0")[-1]
    damped_final = damped.expectation("P_ms0")[-1]

    assert no_phase_final < 0.25
    assert pi_phase_final > 0.75
    assert abs(damped_final - 0.5) < abs(pi_phase_final - 0.5)

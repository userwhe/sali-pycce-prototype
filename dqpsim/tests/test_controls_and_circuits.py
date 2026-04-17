from __future__ import annotations

import numpy as np
import qutip

from nvdyn import C13Spec, Circuit, NVElectronSpec, Register, Simulator
from nvdyn.channels import rf_channel
from nvdyn.compiler import compile_circuit
from nvdyn.hamiltonian import compile_schedule
from nvdyn.observables import observable_map
from nvdyn.pulses import Pulse
from nvdyn.schedule import FrameChange, PulseSchedule
from nvdyn.sequences import CPMG
from nvdyn.waveforms import SampledWaveform


def test_sampled_waveform_support() -> None:
    register = Register(
        electron=NVElectronSpec(D=0.0, B=(0.0, 0.0, 0.0)),
        carbons=[C13Spec(name="c1")],
    )
    waveform = SampledWaveform(np.asarray([0.0, 0.5, 1.0, 0.5]), dt=10e-9)
    pulse = Pulse(
        channel=rf_channel(target="c1"),
        amplitude=50e3,
        carrier=0.0,
        phase=0.0,
        duration=waveform.duration,
        waveform=waveform,
    )
    schedule = PulseSchedule().append(pulse)
    compiled = compile_schedule(register, schedule, dt=waveform.dt)

    assert compiled.tlist[-1] >= waveform.duration
    assert compiled.hamiltonian is not None


def test_cpmg_schedule_structure() -> None:
    sequence = CPMG(target="electron", n=4, tau=0.6e-6, frequency=2.0e6, amplitude=0.5e6)
    schedule = sequence.compile()

    pulse_count = sum(1 for item in schedule.items if isinstance(item, Pulse))
    delay_count = sum(1 for item in schedule.items if item.__class__.__name__ == "Delay")
    measure_count = sum(1 for item in schedule.items if item.__class__.__name__ == "Measure")

    assert pulse_count == 6
    assert delay_count == 8
    assert measure_count == 1


def test_perfect_initialization_default_is_ms0() -> None:
    register = Register(electron=NVElectronSpec(D=0.0, B=(0.0, 0.0, 0.0)))
    state = register.default_state()
    ms0 = observable_map(register)["P_ms0"]

    assert abs(qutip.expect(ms0, state) - 1.0) < 1e-12


def test_circuit_compiles_to_shared_pulse_schedule_layer() -> None:
    register = Register(
        electron=NVElectronSpec(D=3.0e6, B=(0.0, 0.0, 0.0)),
        carbons=[C13Spec(name="c1", A_par=120e3, A_perp=40e3)],
    )
    circuit = Circuit().rx("electron", np.pi / 2.0).delay(0.4e-6).swap("c1")
    schedule = compile_circuit(register, circuit)

    assert isinstance(schedule, PulseSchedule)
    assert any(isinstance(item, FrameChange) for item in schedule.items)

    simulator = Simulator(method="exact", dt=10e-9)
    prepared = simulator.prepare(register, circuit, observables=["P_ms0"])
    assert isinstance(prepared.schedule, PulseSchedule)

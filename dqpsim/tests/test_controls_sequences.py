from __future__ import annotations

import numpy as np

from nvdyn import Pulse, PulseSchedule
from nvdyn.channels import mw_channel
from nvdyn.sequences import CPMG, DDRF, PulsePol, SWAP
from nvdyn.sim.common import compile_schedule_model
from nvdyn.waveforms import SampledWaveform


def test_sampled_waveform_support(bare_register) -> None:
    waveform = SampledWaveform(samples=np.asarray([0.0, 0.5, 1.0, 0.5]), dt=2.0e-9)
    pulse = Pulse(
        channel=mw_channel(),
        amplitude=1.0e6,
        carrier=0.0,
        phase=0.0,
        duration=waveform.duration,
        waveform=waveform,
        label="sampled",
    )
    schedule = PulseSchedule().pulse(pulse).measure()
    compiled = compile_schedule_model(bare_register, schedule, dt=2.0e-9)

    expected = 2.0 * np.pi * 1.0e6 * waveform.samples
    assert np.allclose(compiled.coefficients[("electron", "x")][: len(expected)], expected)
    assert np.allclose(compiled.coefficients[("electron", "y")], 0.0)


def test_cpmg_schedule_structure() -> None:
    sequence = CPMG(target="electron", n=3, tau=1.0e-6)
    pulses = [item for item in sequence.schedule.items if item.__class__.__name__ == "Pulse"]
    delays = [item for item in sequence.schedule.items if item.__class__.__name__ == "Delay"]
    measures = [item for item in sequence.schedule.items if item.__class__.__name__ == "Measure"]

    assert len(pulses) == 5
    assert len(delays) == 6
    assert len(measures) == 1


def test_ddrf_pulsepol_and_swap_macros_are_explicit() -> None:
    ddrf = DDRF(target="c1", blocks=2, tau=0.5e-6, phase_table=(0.0, np.pi / 2.0, np.pi))
    pulsepol = PulsePol(target="c1", cycles=2, tau=0.5e-6, block_phases=(0.0, np.pi / 2.0))
    swap = SWAP(carbon="c1")

    assert sum(item.__class__.__name__ == "Pulse" for item in ddrf.schedule.items) == 6
    assert sum(item.__class__.__name__ == "Pulse" for item in pulsepol.schedule.items) == 8
    assert sum(item.__class__.__name__ == "Pulse" for item in swap.schedule.items) >= 1

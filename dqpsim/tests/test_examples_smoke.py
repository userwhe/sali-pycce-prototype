from __future__ import annotations

from nvdyn.examples.circuit_swap import run_demo as run_circuit_swap
from nvdyn.examples.cpmg_with_c13 import run_demo as run_cpmg
from nvdyn.examples.electron_rabi import run_demo as run_electron_rabi
from nvdyn.examples.odmr_scan import run_demo as run_odmr
from nvdyn.examples.pulsepol_demo import run_demo as run_pulsepol
from nvdyn.examples.ramsey import run_demo as run_ramsey


def test_examples_smoke() -> None:
    assert len(run_odmr(points=3).results) == 3
    assert len(run_electron_rabi(points=3).results) == 3
    assert len(run_ramsey(points=3).results) == 3
    assert run_cpmg().expectation("P_ms0").size >= 1
    assert run_pulsepol().expectation("P_ms0").size >= 1
    assert run_circuit_swap().expectation("P_ms0").size >= 1

from __future__ import annotations

import importlib.util
import numpy as np
import pytest

from nvdyn import NVElectronSpec, ODMR, Register, Simulator
from nvdyn.parallel import ParallelConfig, effective_parallel_config


def _scan_register() -> Register:
    return Register(electron=NVElectronSpec(D=2.0e6, B=(0.0, 0.0, 0.0)))


def _odmr_builder(frequency: float) -> ODMR:
    return ODMR(frequency=frequency, amplitude=0.5e6, duration=0.8e-6)


def test_serial_vs_parallel_equivalence_for_small_sweep() -> None:
    register = _scan_register()
    values = [1.6e6, 1.9e6, 2.1e6, 2.4e6]

    serial = Simulator(method="exact", dt=20e-9, parallel={"map": "serial"}).run_parameter_scan(
        register,
        values,
        _odmr_builder,
        parameter_name="frequency_hz",
        observables=["P_ms0"],
    )
    parallel = Simulator(method="exact", dt=20e-9, parallel={"map": "parallel", "num_cpus": 2}).run_parameter_scan(
        register,
        values,
        _odmr_builder,
        parameter_name="frequency_hz",
        observables=["P_ms0"],
    )

    assert np.allclose(serial.expectation_matrix("P_ms0"), parallel.expectation_matrix("P_ms0"), atol=1e-6)


def test_sweep_result_order_is_preserved() -> None:
    register = _scan_register()
    values = [2.4e6, 1.6e6, 2.1e6, 1.9e6]
    sweep = Simulator(method="exact", dt=20e-9, parallel={"map": "parallel", "num_cpus": 2}).run_parameter_scan(
        register,
        values,
        _odmr_builder,
        parameter_name="frequency_hz",
        observables=["P_ms0"],
    )

    assert np.allclose(sweep.values, np.asarray(values))


def test_no_nested_parallelism_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVDYN_IN_PARALLEL", "1")
    resolved = effective_parallel_config(ParallelConfig(map="parallel", num_cpus=2))
    assert resolved.map == "serial"


def test_repeated_run_solver_reuse_path_works() -> None:
    register = _scan_register()
    simulator = Simulator(method="exact", dt=20e-9)
    prepared = simulator.prepare(register, _odmr_builder(2.0e6), observables=["P_ms0"])

    first = prepared.run()
    second = prepared.run()

    assert np.allclose(first.expectation("P_ms0"), second.expectation("P_ms0"), atol=1e-8)


def test_parallel_falls_back_cleanly_to_serial_when_disabled() -> None:
    resolved = effective_parallel_config(ParallelConfig(map="parallel", num_cpus=1))
    assert resolved.map == "serial"


def test_serial_vs_loky_equivalence_when_installed() -> None:
    if importlib.util.find_spec("loky") is None:
        pytest.skip("loky is not installed")

    register = _scan_register()
    values = [1.8e6, 2.0e6, 2.2e6]
    serial = Simulator(method="exact", dt=20e-9, parallel={"map": "serial"}).run_parameter_scan(
        register,
        values,
        _odmr_builder,
        observables=["P_ms0"],
    )
    loky = Simulator(method="exact", dt=20e-9, parallel={"map": "loky", "num_cpus": 2}).run_parameter_scan(
        register,
        values,
        _odmr_builder,
        observables=["P_ms0"],
    )

    assert np.allclose(serial.expectation_matrix("P_ms0"), loky.expectation_matrix("P_ms0"), atol=1e-6)

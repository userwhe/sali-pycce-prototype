from __future__ import annotations

import numpy as np
import pytest

from nvdyn import Circuit, NVElectronSpec, ODMR, ParallelConfig, PulseSchedule, Register, Simulator, compile_circuit
from nvdyn.parallel import effective_parallel_config, normalize_parallel_config
from nvdyn.schedule import Delay, FrameChange
from nvdyn.channels import mw_channel
from nvdyn.pulses import Pulse


def _finals(sweep) -> np.ndarray:
    return np.asarray([result.expectation("P_ms0")[-1] for result in sweep.results], dtype=float)


def _constant_x_pulse(duration: float, amplitude: float) -> Pulse:
    return Pulse(
        channel=mw_channel(),
        amplitude=amplitude,
        carrier=0.0,
        phase=0.0,
        duration=duration,
    )


def _manual_ramsey_schedule(tau: float, amplitude: float, detuning_hz: float) -> PulseSchedule:
    quarter_turn = 1.0 / (8.0 * amplitude)
    return (
        PulseSchedule()
        .pulse(_constant_x_pulse(quarter_turn, amplitude))
        .delay(tau)
        .append(FrameChange(target="electron", phase=2.0 * np.pi * detuning_hz * tau))
        .pulse(_constant_x_pulse(quarter_turn, amplitude))
        .measure()
    )


def test_perfect_initialization_default_behavior(bare_register) -> None:
    simulator = Simulator(method="exact", dt=5e-9, perfect_init=True)
    schedule = PulseSchedule().delay(50e-9).measure()
    result = simulator.run(bare_register, schedule, observables=["P_ms0"])
    assert result.expectation("P_ms0")[0] == pytest.approx(1.0, abs=1e-9)
    assert result.expectation("P_ms0")[-1] == pytest.approx(1.0, abs=1e-6)


def test_resonant_electron_rabi_oscillation(bare_register) -> None:
    simulator = Simulator(method="exact", dt=2e-9)
    carrier = bare_register.default_electron_transition_hz()
    amplitude = 2.0e6
    pi_duration = 1.0 / (2.0 * amplitude)
    durations = [0.5 * pi_duration, pi_duration, 2.0 * pi_duration]
    sweep = simulator.run_parameter_scan(
        bare_register,
        durations,
        lambda duration: ODMR(carrier=carrier, amplitude=amplitude, duration=duration),
        parameter_name="duration_s",
        observables=["P_ms0"],
    )
    finals = _finals(sweep)
    assert finals[1] < finals[0]
    assert finals[2] > finals[1]


def test_ramsey_fringe_and_dephasing_behavior(bare_register) -> None:
    register = Register(electron=NVElectronSpec(D=0.0, B=(0.0, 0.0, 0.0)))
    amplitude = 0.25e6
    detuning = 0.25e6
    taus = [0.0, 1.0 / (2.0 * detuning)]

    exact = Simulator(method="exact", dt=2e-9)
    exact_sweep = exact.run_parameter_scan(
        register,
        taus,
        lambda tau: _manual_ramsey_schedule(tau, amplitude, detuning),
        observables=["P_ms0"],
    )
    exact_finals = _finals(exact_sweep)
    assert exact_finals[0] < 0.25
    assert exact_finals[1] > 0.75

    lindblad = Simulator(
        method="lindblad",
        dt=2e-9,
        lindblad={"T2": 0.3e-6},
    )
    lindblad_sweep = lindblad.run_parameter_scan(
        register,
        [taus[-1]],
        lambda tau: _manual_ramsey_schedule(tau, amplitude, detuning),
        observables=["P_ms0"],
    )
    lindblad_long = _finals(lindblad_sweep)[0]
    assert abs(lindblad_long - 0.5) < abs(exact_finals[-1] - 0.5)


def test_circuit_compilation_uses_same_schedule_layer(one_carbon_register) -> None:
    circuit = Circuit(label="demo").rx("electron", np.pi / 2.0).delay(0.2e-6).swap("electron", "c1")
    simulator = Simulator(method="exact", dt=5e-9)

    compiled = compile_circuit(one_carbon_register, circuit, simulator.compiler_config)
    assert isinstance(compiled, PulseSchedule)

    from_circuit = simulator.run(one_carbon_register, circuit, observables=["P_ms0"])
    from_schedule = simulator.run(one_carbon_register, compiled, observables=["P_ms0"])
    assert np.allclose(from_circuit.expectation("P_ms0"), from_schedule.expectation("P_ms0"))


def test_serial_vs_parallel_equivalence_for_small_sweep(bare_register) -> None:
    values = np.linspace(
        bare_register.default_electron_transition_hz() - 1.0e6,
        bare_register.default_electron_transition_hz() + 1.0e6,
        3,
    )
    serial = Simulator(method="exact", dt=5e-9, parallel={"map": "serial"})
    parallel = Simulator(method="exact", dt=5e-9, parallel={"map": "parallel", "num_cpus": 2})

    serial_sweep = serial.run_parameter_scan(
        bare_register,
        values.tolist(),
        lambda carrier: ODMR(carrier=carrier, amplitude=1.0e6, duration=0.8e-6),
        observables=["P_ms0"],
    )
    parallel_sweep = parallel.run_parameter_scan(
        bare_register,
        values.tolist(),
        lambda carrier: ODMR(carrier=carrier, amplitude=1.0e6, duration=0.8e-6),
        observables=["P_ms0"],
    )
    assert np.allclose(_finals(serial_sweep), _finals(parallel_sweep), atol=1e-6)


def test_serial_vs_loky_equivalence_when_installed(bare_register) -> None:
    pytest.importorskip("loky")
    values = [
        bare_register.default_electron_transition_hz() - 0.8e6,
        bare_register.default_electron_transition_hz(),
        bare_register.default_electron_transition_hz() + 0.8e6,
    ]
    serial = Simulator(method="exact", dt=5e-9, parallel={"map": "serial"})
    loky = Simulator(method="exact", dt=5e-9, parallel={"map": "loky", "num_cpus": 2})

    serial_sweep = serial.run_parameter_scan(
        bare_register,
        values,
        lambda carrier: ODMR(carrier=carrier, amplitude=1.0e6, duration=0.8e-6),
        observables=["P_ms0"],
    )
    loky_sweep = loky.run_parameter_scan(
        bare_register,
        values,
        lambda carrier: ODMR(carrier=carrier, amplitude=1.0e6, duration=0.8e-6),
        observables=["P_ms0"],
    )
    assert np.allclose(_finals(serial_sweep), _finals(loky_sweep), atol=1e-6)


def test_sweep_result_order_is_preserved(bare_register) -> None:
    values = [
        bare_register.default_electron_transition_hz() + 0.4e6,
        bare_register.default_electron_transition_hz() - 0.4e6,
        bare_register.default_electron_transition_hz(),
    ]
    simulator = Simulator(method="exact", dt=5e-9, parallel={"map": "parallel", "num_cpus": 2})
    sweep = simulator.run_parameter_scan(
        bare_register,
        values,
        lambda carrier: ODMR(carrier=carrier, amplitude=1.0e6, duration=0.8e-6),
        parameter_name="carrier_hz",
        observables=["P_ms0"],
    )
    assert sweep.values.tolist() == values


def test_no_nested_parallelism_by_default(monkeypatch) -> None:
    monkeypatch.setenv("NVDYN_IN_PARALLEL", "1")
    config = effective_parallel_config(ParallelConfig(map="parallel", num_cpus=2))
    assert config.map == "serial"


def test_repeated_run_solver_reuse_path_works(bare_register) -> None:
    simulator = Simulator(method="exact", dt=5e-9)
    prepared = simulator.prepare(
        bare_register,
        ODMR(
            carrier=bare_register.default_electron_transition_hz(),
            amplitude=1.0e6,
            duration=0.8e-6,
        ),
        observables=["P_ms0"],
    )
    first = prepared.run()
    solver_id = id(prepared.backend._density_solver)
    second = prepared.run()
    assert id(prepared.backend._density_solver) == solver_id
    assert np.allclose(first.expectation("P_ms0"), second.expectation("P_ms0"))


def test_parallel_config_falls_back_to_serial_by_default() -> None:
    config = normalize_parallel_config(None)
    assert config.map == "serial"

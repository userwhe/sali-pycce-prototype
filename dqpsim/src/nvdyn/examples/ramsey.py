"""Ramsey example."""

from __future__ import annotations

import numpy as np

from ..sequences import Ramsey
from ..simulator import Simulator
from ._shared import demo_register


def run_demo(points: int = 5):
    register = demo_register(with_carbon=False)
    simulator = Simulator(method="exact", dt=2e-9)
    detuning = 0.25e6
    frequency = register.default_electron_transition_hz() + detuning
    taus = np.linspace(0.2e-6, 2.0e-6, points)
    return simulator.run_parameter_scan(
        register,
        taus.tolist(),
        lambda tau: Ramsey(tau=tau, frequency=frequency, amplitude=1.2e6),
        parameter_name="tau_s",
        observables=["P_ms0"],
    )


def main() -> None:
    sweep = run_demo()
    print("tau_s", sweep.values)
    print("final P_ms0", [result.expectation("P_ms0")[-1] for result in sweep.results])


if __name__ == "__main__":
    main()

"""Electron Rabi example."""

from __future__ import annotations

import numpy as np

from ..sequences import ElectronRabi
from ..simulator import Simulator
from ._shared import demo_register


def run_demo(points: int = 5):
    register = demo_register(with_carbon=False)
    simulator = Simulator(method="exact", dt=2e-9)
    carrier = register.default_electron_transition_hz()
    durations = np.linspace(50e-9, 800e-9, points)
    return simulator.run_parameter_scan(
        register,
        durations.tolist(),
        lambda duration: ElectronRabi(duration=duration, frequency=carrier, amplitude=1.2e6),
        parameter_name="duration_s",
        observables=["P_ms0"],
    )


def main() -> None:
    sweep = run_demo()
    print("duration_s", sweep.values)
    print("final P_ms0", [result.expectation("P_ms0")[-1] for result in sweep.results])


if __name__ == "__main__":
    main()

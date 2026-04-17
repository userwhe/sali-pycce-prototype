"""ODMR scan example."""

from __future__ import annotations

import numpy as np

from ..sequences import ODMR
from ..simulator import Simulator
from ._shared import demo_register


def run_demo(points: int = 5):
    register = demo_register(with_carbon=False)
    simulator = Simulator(method="exact", dt=2e-9)
    center = register.default_electron_transition_hz()
    carriers = np.linspace(center - 2.0e6, center + 2.0e6, points)
    return simulator.run_parameter_scan(
        register,
        carriers.tolist(),
        lambda carrier: ODMR(frequency=carrier, amplitude=0.9e6, duration=0.8e-6),
        parameter_name="carrier_hz",
        observables=["P_ms0"],
    )


def main() -> None:
    sweep = run_demo()
    print("carrier_hz", sweep.values)
    print("final P_ms0", [result.expectation("P_ms0")[-1] for result in sweep.results])


if __name__ == "__main__":
    main()

"""CPMG example with one coupled :sup:`13`C."""

from __future__ import annotations

from ..sequences import CPMG
from ..simulator import Simulator
from ._shared import demo_register


def run_demo():
    register = demo_register(with_carbon=True)
    simulator = Simulator(method="exact", dt=2e-9)
    sequence = CPMG(
        target="electron",
        n=4,
        tau=0.8e-6,
        frequency=register.default_electron_transition_hz(),
        amplitude=1.2e6,
    )
    return simulator.run(register, sequence, observables=["P_ms0", "Iz:c1"])


def main() -> None:
    result = run_demo()
    print("final P_ms0", result.expectation("P_ms0")[-1])
    print("final Iz:c1", result.expectation("Iz:c1")[-1])


if __name__ == "__main__":
    main()

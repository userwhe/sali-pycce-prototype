"""PulsePol example."""

from __future__ import annotations

from ..sequences import PulsePol
from ..simulator import Simulator
from ._shared import demo_register


def run_demo():
    register = demo_register(with_carbon=True)
    simulator = Simulator(method="exact", dt=5e-9)
    sequence = PulsePol(
        target="c1",
        blocks=2,
        tau=0.4e-6,
        electron_frequency=register.default_electron_transition_hz(),
        nuclear_frequency=register.default_nuclear_transition_hz("c1"),
        electron_amplitude=1.0e6,
        nuclear_amplitude=80e3,
    )
    return simulator.run(register, sequence, observables=["P_ms0", "P_up:c1"])


def main() -> None:
    result = run_demo()
    print("final P_ms0", result.expectation("P_ms0")[-1])
    print("final P_up:c1", result.expectation("P_up:c1")[-1])


if __name__ == "__main__":
    main()

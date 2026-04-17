"""Pulse-accurate circuit example."""

from __future__ import annotations

from ..circuit import Circuit
from ..simulator import Simulator
from ._shared import demo_register


def run_demo():
    register = demo_register(with_carbon=True)
    circuit = Circuit(label="swap-demo").rx("electron", 0.5).rz("electron", 0.3).swap("c1")
    simulator = Simulator(method="exact", dt=5e-9)
    return simulator.run(register, circuit, observables=["P_ms0", "P_up:c1"])


def main() -> None:
    result = run_demo()
    print("final P_ms0", result.expectation("P_ms0")[-1])
    print("final P_up:c1", result.expectation("P_up:c1")[-1])


if __name__ == "__main__":
    main()

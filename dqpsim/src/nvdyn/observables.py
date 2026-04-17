"""Observable helpers."""

from __future__ import annotations

from typing import Iterable

import qutip

from .device import Register
from .operators import carbon_operator, carbon_qubit_operators, electron_ms0_projector, electron_operator, electron_qubit_operators


def observable_map(register: Register) -> dict[str, qutip.Qobj]:
    electron_ops = electron_qubit_operators(register)
    observables: dict[str, qutip.Qobj] = {
        "P_ms0": electron_ms0_projector(register),
        "P_msm1": electron_ops["p1"],
        "Sx": electron_operator(register, "x"),
        "Sy": electron_operator(register, "y"),
        "Sz": electron_operator(register, "z"),
    }
    for carbon in register.carbons:
        carbon_ops = carbon_qubit_operators(register, carbon.name)
        observables[f"{carbon.name}:Ix"] = carbon_operator(register, carbon.name, "x")
        observables[f"{carbon.name}:Iy"] = carbon_operator(register, carbon.name, "y")
        observables[f"{carbon.name}:Iz"] = carbon_operator(register, carbon.name, "z")
        observables[f"Ix:{carbon.name}"] = observables[f"{carbon.name}:Ix"]
        observables[f"Iy:{carbon.name}"] = observables[f"{carbon.name}:Iy"]
        observables[f"Iz:{carbon.name}"] = observables[f"{carbon.name}:Iz"]
        observables[f"P_up:{carbon.name}"] = carbon_ops["p0"]
    return observables


def resolve_observables(register: Register, observables: Iterable[str] | dict[str, qutip.Qobj] | None) -> dict[str, qutip.Qobj]:
    available = observable_map(register)
    if observables is None:
        return {"P_ms0": available["P_ms0"]}
    if isinstance(observables, dict):
        return observables
    resolved: dict[str, qutip.Qobj] = {}
    for name in observables:
        if name not in available:
            raise KeyError(f"unknown observable {name!r}")
        resolved[name] = available[name]
    return resolved

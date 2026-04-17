"""Shared helpers for examples."""

from __future__ import annotations

from ..device import C13Spec, NVElectronSpec, Register


def demo_register(*, with_carbon: bool = True) -> Register:
    """Return a numerically light register for runnable examples.

    The examples use reduced frequencies compared to a room-temperature NV to
    keep pulse-accurate laboratory-frame simulations interactive on a laptop.
    """

    carbons = [C13Spec(name="c1", A_par=414e3, A_perp=95e3)] if with_carbon else []
    return Register(
        electron=NVElectronSpec(D=100e6, B=(0.0, 0.0, 0.003)),
        carbons=carbons,
    )

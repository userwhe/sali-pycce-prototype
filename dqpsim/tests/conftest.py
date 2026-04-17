from __future__ import annotations

import pytest

from nvdyn import C13Spec, NVElectronSpec, Register


@pytest.fixture()
def bare_register() -> Register:
    return Register(
        electron=NVElectronSpec(D=2.87e9, B=(0.0, 0.0, 0.05)),
        carbons=[],
    )


@pytest.fixture()
def one_carbon_register() -> Register:
    return Register(
        electron=NVElectronSpec(D=2.87e9, B=(0.0, 0.0, 0.04)),
        carbons=[C13Spec(name="c1", A_par=414e3, A_perp=95e3)],
    )


@pytest.fixture()
def two_carbon_register() -> Register:
    return Register(
        electron=NVElectronSpec(D=2.87e9, B=(0.0, 0.0, 0.04)),
        carbons=[
            C13Spec(name="c1", A_par=414e3, A_perp=95e3),
            C13Spec(name="c2", A_par=120e3, A_perp=40e3),
        ],
    )

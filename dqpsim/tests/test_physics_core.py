from __future__ import annotations

import numpy as np

from nvdyn import C13Spec, NVElectronSpec, Register
from nvdyn.hamiltonian import hyperfine_term, static_hamiltonian
from nvdyn.operators import carbon_operator, electron_operator
from nvdyn.units import angular_frequency


def test_operator_dimensions_and_tensor_product_assembly() -> None:
    register = Register(
        electron=NVElectronSpec(D=1.0e6, B=(0.0, 0.0, 0.0)),
        carbons=[C13Spec(name="c1"), C13Spec(name="c2")],
    )
    sx = electron_operator(register, "x")
    iz = carbon_operator(register, "c2", "z")

    assert sx.shape == (12, 12)
    assert iz.shape == (12, 12)
    assert sx.dims == [[3, 2, 2], [3, 2, 2]]
    assert iz.dims == [[3, 2, 2], [3, 2, 2]]


def test_default_secular_hyperfine_assembly() -> None:
    register = Register(
        electron=NVElectronSpec(D=0.0, B=(0.0, 0.0, 0.0)),
        carbons=[C13Spec(name="c1", A_par=414e3, A_perp=95e3)],
    )
    expected = (
        angular_frequency(414e3) * electron_operator(register, "z") * carbon_operator(register, "c1", "z")
        + angular_frequency(95e3) * electron_operator(register, "x") * carbon_operator(register, "c1", "x")
    )
    assert (hyperfine_term(register, "c1") - expected).norm() < 1e-12


def test_full_tensor_override_path() -> None:
    tensor = np.asarray(
        [
            [1.0e3, 2.0e3, 3.0e3],
            [4.0e3, 5.0e3, 6.0e3],
            [7.0e3, 8.0e3, 9.0e3],
        ]
    )
    register = Register(
        electron=NVElectronSpec(D=0.0, B=(0.0, 0.0, 0.0)),
        carbons=[C13Spec(name="c1", tensor=tensor)],
    )

    manual = 0.0 * electron_operator(register, "z")
    for row_axis, row in zip(("x", "y", "z"), tensor, strict=True):
        for col_axis, coefficient in zip(("x", "y", "z"), row, strict=True):
            manual = manual + angular_frequency(float(coefficient)) * (
                electron_operator(register, row_axis) * carbon_operator(register, "c1", col_axis)
            )

    assert (hyperfine_term(register, "c1") - manual).norm() < 1e-12


def test_static_hamiltonian_has_no_c13_c13_terms() -> None:
    register = Register(
        electron=NVElectronSpec(D=2.0e6, B=(0.0, 0.0, 2.0e-4)),
        carbons=[
            C13Spec(name="c1", A_par=110e3, A_perp=20e3),
            C13Spec(name="c2", A_par=70e3, A_perp=15e3),
        ],
    )

    sx = electron_operator(register, "x")
    sz = electron_operator(register, "z")
    manual = angular_frequency(register.electron.D) * (sz * sz)
    manual = manual + angular_frequency(register.electron.gamma * register.electron.B[2]) * sz

    for carbon in register.carbons:
        iz = carbon_operator(register, carbon.name, "z")
        ix = carbon_operator(register, carbon.name, "x")
        manual = manual - angular_frequency(carbon.gamma * register.electron.B[2]) * iz
        manual = manual + angular_frequency(carbon.A_par) * sz * iz
        manual = manual + angular_frequency(carbon.A_perp) * sx * ix

    assert (static_hamiltonian(register) - manual).norm() < 1e-12

"""Operator construction utilities for the NV register."""

from __future__ import annotations

from functools import lru_cache

from qutip import Qobj, basis, jmat, qeye, tensor

from .device import Register


@lru_cache(maxsize=None)
def local_spin_operators(spin: float) -> tuple[Qobj, Qobj, Qobj]:
    """Return local spin operators for a given spin quantum number."""

    return (jmat(spin, "x"), jmat(spin, "y"), jmat(spin, "z"))


@lru_cache(maxsize=None)
def local_identity(dim: int) -> Qobj:
    return qeye(dim)


def embed_single_operator(register: Register, site: int, operator: Qobj) -> Qobj:
    factors = [local_identity(dim) for dim in register.hilbert_dims]
    factors[site] = operator
    return tensor(factors)


def electron_operator(register: Register, axis: str) -> Qobj:
    sx, sy, sz = local_spin_operators(1.0)
    local = {"x": sx, "y": sy, "z": sz}[axis]
    return embed_single_operator(register, 0, local)


def carbon_operator(register: Register, carbon: int | str, axis: str) -> Qobj:
    index = register.carbon_index(carbon)
    ix, iy, iz = local_spin_operators(0.5)
    local = {"x": ix, "y": iy, "z": iz}[axis]
    return embed_single_operator(register, index + 1, local)


def electron_ms0_projector(register: Register) -> Qobj:
    return embed_single_operator(register, 0, basis(3, 1).proj())


def electron_qubit_operators(register: Register) -> dict[str, Qobj]:
    ket_zero = basis(3, 1)
    ket_one = basis(3, 2)
    x_local = ket_zero * ket_one.dag() + ket_one * ket_zero.dag()
    y_local = -1j * ket_zero * ket_one.dag() + 1j * ket_one * ket_zero.dag()
    z_local = ket_zero.proj() - ket_one.proj()
    p0_local = ket_zero.proj()
    p1_local = ket_one.proj()
    return {
        "x": embed_single_operator(register, 0, x_local),
        "y": embed_single_operator(register, 0, y_local),
        "z": embed_single_operator(register, 0, z_local),
        "p0": embed_single_operator(register, 0, p0_local),
        "p1": embed_single_operator(register, 0, p1_local),
    }


def carbon_qubit_operators(register: Register, carbon: int | str) -> dict[str, Qobj]:
    index = register.carbon_index(carbon) + 1
    ket_up = basis(2, 0)
    ket_down = basis(2, 1)
    x_local = ket_up * ket_down.dag() + ket_down * ket_up.dag()
    y_local = -1j * ket_up * ket_down.dag() + 1j * ket_down * ket_up.dag()
    z_local = ket_up.proj() - ket_down.proj()
    p0_local = ket_up.proj()
    p1_local = ket_down.proj()
    return {
        "x": embed_single_operator(register, index, x_local),
        "y": embed_single_operator(register, index, y_local),
        "z": embed_single_operator(register, index, z_local),
        "p0": embed_single_operator(register, index, p0_local),
        "p1": embed_single_operator(register, index, p1_local),
    }


def control_qubit_operators(register: Register, target: int | str) -> dict[str, Qobj]:
    if target == "electron":
        return electron_qubit_operators(register)
    return carbon_qubit_operators(register, target)

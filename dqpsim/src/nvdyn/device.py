"""Typed device specifications for the NV register."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from qutip import Qobj, basis, ket2dm, qeye, tensor

from .constants import GAMMA_C13_HZ_PER_T, GAMMA_E_HZ_PER_T, ZERO_FIELD_SPLITTING_HZ

Vector3 = tuple[float, float, float]


def _as_vector3(value: Sequence[float]) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"expected a length-3 vector, received {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


@dataclass(slots=True, frozen=True)
class NVElectronSpec:
    """Ground-state NV electron specification.

    The electron Hilbert space always uses QuTiP's spin-1 ordering
    ``|ms=+1>``, ``|ms=0>``, ``|ms=-1>``.
    """

    D: float = ZERO_FIELD_SPLITTING_HZ
    B: Vector3 = (0.0, 0.0, 0.0)
    gamma: float = GAMMA_E_HZ_PER_T

    def __post_init__(self) -> None:
        object.__setattr__(self, "B", _as_vector3(self.B))
        if self.D < 0.0:
            raise ValueError("zero-field splitting D must be non-negative")


@dataclass(slots=True, frozen=True)
class C13Spec:
    """Nearby :sup:`13`C specification.

    The public MVP API exposes the secular parameters ``A_par`` and ``A_perp``.
    A full 3x3 tensor can optionally override the default secular model as an
    advanced path for a specific carbon.
    """

    name: str
    A_par: float = 0.0
    A_perp: float = 0.0
    tensor: Sequence[Sequence[float]] | None = None
    gamma: float = GAMMA_C13_HZ_PER_T

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("each carbon needs a non-empty name")
        if self.tensor is not None:
            tensor_array = np.asarray(self.tensor, dtype=float)
            if tensor_array.shape != (3, 3):
                raise ValueError("tensor override must be a 3x3 matrix")
            object.__setattr__(self, "tensor", tensor_array)

    @property
    def uses_tensor_override(self) -> bool:
        return self.tensor is not None

    @property
    def tensor_array(self) -> NDArray[np.float64] | None:
        if self.tensor is None:
            return None
        return np.asarray(self.tensor, dtype=float)


@dataclass(slots=True)
class Register:
    """Complete simulated spin register.

    Tensor-product ordering is fixed to
    ``electron ⊗ carbon_0 ⊗ carbon_1 ⊗ ...``.
    """

    electron: NVElectronSpec = field(default_factory=NVElectronSpec)
    carbons: list[C13Spec] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = [carbon.name for carbon in self.carbons]
        if len(set(names)) != len(names):
            raise ValueError("carbon names must be unique")

    @property
    def hilbert_dims(self) -> tuple[int, ...]:
        return (3,) + tuple(2 for _ in self.carbons)

    @property
    def hilbert_dim(self) -> int:
        dim = 1
        for local_dim in self.hilbert_dims:
            dim *= local_dim
        return dim

    @property
    def num_carbons(self) -> int:
        return len(self.carbons)

    def carbon_index(self, target: int | str) -> int:
        if isinstance(target, int):
            if target < 0 or target >= len(self.carbons):
                raise IndexError(f"carbon index {target} is out of range")
            return target
        for index, carbon in enumerate(self.carbons):
            if carbon.name == target:
                return index
        raise KeyError(f"unknown carbon target {target!r}")

    def target_label(self, target: int | str) -> str:
        if isinstance(target, str) and target == "electron":
            return target
        return self.carbons[self.carbon_index(target)].name

    def electron_ms0_ket(self) -> Qobj:
        return basis(3, 1)

    def electron_msm1_ket(self) -> Qobj:
        return basis(3, 2)

    def carbon_up_ket(self) -> Qobj:
        return basis(2, 0)

    def default_state(self, *, perfect_init: bool = True, nuclear_state: str = "mixed") -> Qobj:
        """Return the default initial state.

        By default the electron is perfectly initialized into ``ms = 0`` and
        each carbon starts maximally mixed. ``nuclear_state='up'`` is a
        convenient pure-state option for demos and unit tests.
        """

        electron_factor: Qobj = self.electron_ms0_ket() if perfect_init else qeye(3) / 3.0
        factors: list[Qobj] = [electron_factor]
        for _ in self.carbons:
            if nuclear_state == "up":
                factors.append(self.carbon_up_ket())
            elif nuclear_state == "mixed":
                factors.append(0.5 * qeye(2))
            else:
                raise ValueError("nuclear_state must be 'mixed' or 'up'")

        if all(factor.isket for factor in factors):
            return tensor(factors)
        return tensor([factor if factor.isoper else ket2dm(factor) for factor in factors])

    def default_initial_state(self, *, polarized_nuclei: bool = False) -> Qobj:
        return self.default_state(
            perfect_init=True,
            nuclear_state="up" if polarized_nuclei else "mixed",
        )

    def default_electron_transition_hz(self) -> float:
        """Approximate ``ms=0 <-> ms=-1`` transition frequency in Hz."""

        bz = self.electron.B[2]
        return self.electron.D - self.electron.gamma * bz

    def default_nuclear_transition_hz(self, target: int | str) -> float:
        """Approximate nuclear Larmor frequency in Hz."""

        carbon = self.carbons[self.carbon_index(target)]
        bz = self.electron.B[2]
        return abs(carbon.gamma * bz)

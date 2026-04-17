"""Physical constants and shared conventions for :mod:`nvdyn`."""

from __future__ import annotations

ZERO_FIELD_SPLITTING_HZ = 2.87e9
GAMMA_E_HZ_PER_T = 28.024_951_64e9
GAMMA_C13_HZ_PER_T = 10.708_4e6

ELECTRON_BASIS_LABELS = ("ms=+1", "ms=0", "ms=-1")
C13_BASIS_LABELS = ("mI=+1/2", "mI=-1/2")

DEFAULT_ELECTRON_RABI_HZ = 5.0e6
DEFAULT_NUCLEAR_RABI_HZ = 50.0e3
DEFAULT_OBSERVABLE = "P_ms0"

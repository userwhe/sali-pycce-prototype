"""Runnable package examples."""

from .circuit_swap import run_demo as run_circuit_swap_demo
from .cpmg_with_c13 import run_demo as run_cpmg_demo
from .electron_rabi import run_demo as run_electron_rabi_demo
from .odmr_scan import run_demo as run_odmr_demo
from .pulsepol_demo import run_demo as run_pulsepol_demo
from .ramsey import run_demo as run_ramsey_demo

__all__ = [
    "run_circuit_swap_demo",
    "run_cpmg_demo",
    "run_electron_rabi_demo",
    "run_odmr_demo",
    "run_pulsepol_demo",
    "run_ramsey_demo",
]

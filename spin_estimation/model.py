from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GAMMA_C13_KHZ_PER_G = 1.0705
TWO_PI = 2.0 * np.pi
KHZ_TO_RAD_PER_US = TWO_PI * 1e-3


def carbon13_larmor_khz(b0_gauss: float) -> float:
    return GAMMA_C13_KHZ_PER_G * float(b0_gauss)


def khz_to_rad_per_us(frequency_khz: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(frequency_khz) * KHZ_TO_RAD_PER_US


def rad_per_us_to_khz(angular_frequency: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(angular_frequency) / KHZ_TO_RAD_PER_US


@dataclass(frozen=True)
class HyperfineParameters:
    a_khz: float
    b_khz: float


def single_spin_modulation(
    tau_us: np.ndarray,
    n_pulses: int,
    a_khz: float,
    b_khz: float,
    larmor_khz: float,
) -> np.ndarray:
    tau_us = np.asarray(tau_us, dtype=float)
    omega_l = khz_to_rad_per_us(larmor_khz)
    a = khz_to_rad_per_us(a_khz)
    b = khz_to_rad_per_us(b_khz)

    omega_tilde = np.sqrt((a + omega_l) ** 2 + b**2)
    alpha = omega_tilde * tau_us
    beta = omega_l * tau_us
    mz = (a + omega_l) / omega_tilde
    mx = b / omega_tilde

    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    sin_alpha = np.sin(alpha)
    sin_beta = np.sin(beta)

    cos_phi = cos_alpha * cos_beta - mz * sin_alpha * sin_beta
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.arccos(cos_phi)

    denominator = 1.0 + cos_alpha * cos_beta - mz * sin_alpha * sin_beta
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    one_minus_dot = mx**2 * (1.0 - cos_alpha) * (1.0 - cos_beta) / denominator

    modulation = 1.0 - one_minus_dot * np.sin(n_pulses * phi / 2.0) ** 2
    return np.clip(modulation, -1.0, 1.0)


def simulate_cpmg_signal(
    tau_us: np.ndarray,
    n_pulses: int,
    spins: list[HyperfineParameters] | list[tuple[float, float]],
    larmor_khz: float,
) -> np.ndarray:
    tau_us = np.asarray(tau_us, dtype=float)

    modulation = np.ones_like(tau_us)
    for spin in spins:
        if isinstance(spin, HyperfineParameters):
            a_khz = spin.a_khz
            b_khz = spin.b_khz
        else:
            a_khz, b_khz = spin
        modulation *= single_spin_modulation(
            tau_us=tau_us,
            n_pulses=n_pulses,
            a_khz=a_khz,
            b_khz=b_khz,
            larmor_khz=larmor_khz,
        )
    return (modulation + 1.0) / 2.0 

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sali.config import PhysicsConfig


_PROBABILITY_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class Couplings:
    az_khz: np.ndarray
    aperp_khz: np.ndarray

    def __post_init__(self) -> None:
        az_khz = np.array(self.az_khz, dtype=np.float32, copy=True)
        aperp_khz = np.array(self.aperp_khz, dtype=np.float32, copy=True)
        if az_khz.shape != aperp_khz.shape:
            raise ValueError("az_khz and aperp_khz must have matching shapes")
        if az_khz.ndim != 1:
            raise ValueError("coupling arrays must be one-dimensional")
        if not np.isfinite(az_khz).all() or not np.isfinite(aperp_khz).all():
            raise ValueError("coupling arrays must contain finite values")
        az_khz.setflags(write=False)
        aperp_khz.setflags(write=False)
        object.__setattr__(self, "az_khz", az_khz)
        object.__setattr__(self, "aperp_khz", aperp_khz)

    @property
    def count(self) -> int:
        return int(self.az_khz.size)


def tau_grid_us(start_us: float, stop_us: float, points: int) -> np.ndarray:
    if points <= 1:
        raise ValueError("points must be greater than 1")
    if stop_us <= start_us:
        raise ValueError("tau stop must be greater than tau start")
    return np.linspace(start_us, stop_us, points, dtype=np.float32)


def _validate_probability_signal(signal: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    if not np.isfinite(values).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    if values.size > 0:
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if min_value < -_PROBABILITY_TOLERANCE or max_value > 1.0 + _PROBABILITY_TOLERANCE:
            raise FloatingPointError(f"{name} contains probability values outside [0, 1]")
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def _ideal_signal(nuclei: Couplings, tau_us: np.ndarray, n_pulses: int, cfg: PhysicsConfig) -> np.ndarray:
    if nuclei.count == 0:
        return np.ones_like(tau_us, dtype=np.float32)
    tau_s = tau_us.astype(np.float64) * 1e-6
    gamma_rad_s_t = 2.0 * np.pi * cfg.gamma_mhz_per_t * 1e6
    omega_l = gamma_rad_s_t * cfg.bz_tesla
    az = nuclei.az_khz.astype(np.float64) * 2.0 * np.pi * 1e3
    aperp = nuclei.aperp_khz.astype(np.float64) * 2.0 * np.pi * 1e3
    shifted = az + omega_l
    omega_tilde = np.sqrt(shifted[:, None] ** 2 + aperp[:, None] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        mz = shifted[:, None] / omega_tilde
        mx = aperp[:, None] / omega_tilde
    alpha = omega_tilde * tau_s[None, :]
    beta = omega_l * tau_s[None, :]
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    sin_alpha = np.sin(alpha)
    sin_beta = np.sin(beta)
    cos_phi = cos_alpha * cos_beta - mz * sin_alpha * sin_beta
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.arccos(cos_phi)
    denom = np.maximum(1.0 + cos_phi, 1e-12)
    modulation = ((1.0 - cos_alpha) * (1.0 - cos_beta)) / denom
    sin_term = np.sin(n_pulses * phi / 2.0) ** 2
    mj = 1.0 - (mx**2) * modulation * sin_term
    product = np.prod(mj, axis=0)
    px = 0.5 * (1.0 + product)
    return _validate_probability_signal(px, "ideal signal")


def _apply_decoherence(signal: np.ndarray, tau_us: np.ndarray, cfg: PhysicsConfig) -> np.ndarray:
    if not cfg.add_decoherence:
        return signal
    decay = np.exp(-tau_us.astype(np.float32) / np.float32(cfg.t2_us))
    return _validate_probability_signal(signal * decay, "decohered signal")


def _apply_shot_noise(signal: np.ndarray, cfg: PhysicsConfig, rng: np.random.Generator) -> np.ndarray:
    signal = _validate_probability_signal(signal, "shot-noise input signal")
    if not cfg.add_shot_noise:
        return signal.astype(np.float32)
    counts = rng.binomial(cfg.measurements, signal)
    noisy = counts.astype(np.float32) / np.float32(cfg.measurements)
    return _validate_probability_signal(noisy, "shot-noise signal")


def simulate_cpmg_signal(
    nuclei: Couplings,
    tau_us: np.ndarray,
    n_pulses: int,
    cfg: PhysicsConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_pulses <= 0:
        raise ValueError("n_pulses must be positive")
    if tau_us.shape != (cfg.signal_points,):
        raise ValueError(f"tau_us must have shape ({cfg.signal_points},)")
    signal = _ideal_signal(nuclei, tau_us, n_pulses, cfg)
    signal = _apply_decoherence(signal, tau_us, cfg)
    signal = _apply_shot_noise(signal, cfg, rng)
    if not np.isfinite(signal).all():
        raise FloatingPointError("generated signal contains non-finite values")
    return signal.astype(np.float32)


def generate_sample_signals(
    nuclei: Couplings,
    cfg: PhysicsConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    tau32 = tau_grid_us(cfg.tau_32_min_us, cfg.tau_32_max_us, cfg.signal_points)
    tau256 = tau_grid_us(cfg.tau_256_min_us, cfg.tau_256_max_us, cfg.signal_points)
    signal32 = simulate_cpmg_signal(nuclei, tau32, 32, cfg, rng)
    signal256 = simulate_cpmg_signal(nuclei, tau256, 256, cfg, rng)
    signals = np.stack([signal32, signal256], axis=0).astype(np.float32)
    if signals.shape != (2, cfg.signal_points):
        raise AssertionError("signals must have shape (2, signal_points)")
    return signals

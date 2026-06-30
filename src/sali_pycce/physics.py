"""Fast analytic CPMG signal simulator.

This module is designed for ML data generation.  It implements a compact
closed-form approximation similar to the expression used in SALI-like synthetic
training pipelines.  Frequencies are specified in kHz and times in microseconds.

For high-fidelity spin-bath dynamics, use/replace the optional PyCCE backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SpinParams:
    """Hyperfine parameters for one nucleus.

    Parameters
    ----------
    az_khz:
        Longitudinal hyperfine parameter A_z in kHz.
    aperp_khz:
        Transverse hyperfine parameter A_perp in kHz.  The CPMG response is
        symmetric under A_perp -> -A_perp for this model, so positive values are
        usually sufficient.
    """

    az_khz: float
    aperp_khz: float


@dataclass
class AnalyticCPMGSimulator:
    """Generate SALI-style CPMG survival-probability traces.

    The simulator returns two input traces by default, matching the SALI idea of
    using short and long CPMG sequences to reveal strongly and weakly coupled
    nuclei.
    """

    b_gauss: float = 525.0
    pulses: tuple[int, int] = (32, 256)
    tau_ranges_us: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 40.0), (0.0, 40.0))
    signal_points: int = 4000
    # gamma_13C / 2pi = 10.705 MHz/T = 1.0705 kHz/G
    gamma_c13_khz_per_g: float = 1.0705
    # Factor multiplying A in the conditional nuclear frequency. The Hamiltonian
    # convention in the paper uses an f(t) A/2 term; keep this configurable.
    hyperfine_factor: float = 0.5
    t2_us: float | None = 800.0
    t2_stretch: float = 1.0
    readout_offset: float = 0.0
    readout_visibility: float = 1.0
    shots: int | None = 1000

    @property
    def omega_l_khz(self) -> float:
        """13C Larmor frequency in kHz cycles, not angular rad/us."""
        return self.gamma_c13_khz_per_g * self.b_gauss

    def tau_grid(self, pulse_index: int) -> np.ndarray:
        start, stop = self.tau_ranges_us[pulse_index]
        return np.linspace(start, stop, self.signal_points, dtype=np.float64)

    @staticmethod
    def _phase_from_khz(freq_khz: np.ndarray | float, tau_us: np.ndarray) -> np.ndarray:
        """Convert kHz * us to radians.

        1 kHz = 1 cycle/ms = 1e-3 cycles/us.
        """
        return 2.0 * np.pi * np.asarray(freq_khz) * tau_us * 1e-3

    def single_spin_modulation(self, spin: SpinParams, n_pulses: int, tau_us: np.ndarray) -> np.ndarray:
        """Return the single-spin modulation factor M_j(tau)."""
        f = self.hyperfine_factor
        omega_l = self.omega_l_khz
        wx = f * float(spin.aperp_khz)
        wz = omega_l + f * float(spin.az_khz)
        omega_j = np.sqrt(wz * wz + wx * wx)

        if omega_j == 0:
            return np.ones_like(tau_us)

        mx = wx / omega_j
        mz = wz / omega_j

        alpha = self._phase_from_khz(omega_j, tau_us)
        beta = self._phase_from_khz(omega_l, tau_us)

        cos_a = np.cos(alpha)
        cos_b = np.cos(beta)
        sin_a = np.sin(alpha)
        sin_b = np.sin(beta)
        denom = 1.0 + cos_a * cos_b - mz * sin_a * sin_b
        denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12 + 1e-12, denom)

        cos_phi = cos_a * cos_b - mz * sin_a * sin_b
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        phi = np.arccos(cos_phi)

        amp = (mx * mx) * ((1.0 - cos_a) * (1.0 - cos_b)) / denom
        modulation = 1.0 - amp * np.sin(0.5 * n_pulses * phi) ** 2
        return np.clip(modulation, -1.0, 1.0)

    def signal(self, spins: Sequence[SpinParams], n_pulses: int, tau_us: np.ndarray) -> np.ndarray:
        """Compute P_x(tau) for a list of spin parameters."""
        product = np.ones_like(tau_us, dtype=np.float64)
        for spin in spins:
            product *= self.single_spin_modulation(spin, n_pulses, tau_us)
        px = 0.5 * (1.0 + product)

        if self.t2_us is not None and self.t2_us > 0:
            # Simple contrast decay model. This is intentionally separated from
            # the coherent product, so the long-time trace relaxes toward 1/2.
            stretch = max(float(self.t2_stretch), 1e-12)
            contrast = np.exp(-np.power(tau_us / float(self.t2_us), stretch))
            px = 0.5 + (px - 0.5) * contrast

        px = self.readout_offset + self.readout_visibility * px
        return np.clip(px, 0.0, 1.0)

    def noisy_signal(
        self,
        spins: Sequence[SpinParams],
        n_pulses: int,
        tau_us: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Compute signal and add binomial shot noise if ``shots`` is set."""
        px = self.signal(spins, n_pulses, tau_us)
        if self.shots is None or self.shots <= 0:
            return px.astype(np.float32)
        counts = rng.binomial(int(self.shots), px)
        return (counts / float(self.shots)).astype(np.float32)

    def sample(
        self,
        spins: Sequence[SpinParams],
        rng: np.random.Generator | None = None,
        noisy: bool = True,
    ) -> dict[str, np.ndarray]:
        """Return all CPMG traces for the configured pulse counts.

        Returns a dictionary with keys ``signals`` of shape ``(n_inputs, points)``
        and ``taus_us`` of the same shape.
        """
        rng = np.random.default_rng() if rng is None else rng
        signals: list[np.ndarray] = []
        taus: list[np.ndarray] = []
        for i, n in enumerate(self.pulses):
            tau = self.tau_grid(i)
            sig = self.noisy_signal(spins, n, tau, rng) if noisy else self.signal(spins, n, tau).astype(np.float32)
            taus.append(tau.astype(np.float32))
            signals.append(sig.astype(np.float32))
        return {"signals": np.stack(signals, axis=0), "taus_us": np.stack(taus, axis=0)}


def random_spins(
    rng: np.random.Generator,
    n_spins: int,
    az_range: tuple[float, float] = (-100.0, 100.0),
    aperp_range: tuple[float, float] = (2.0, 102.0),
) -> list[SpinParams]:
    """Sample a random spin configuration."""
    az = rng.uniform(*az_range, size=n_spins)
    ap = rng.uniform(*aperp_range, size=n_spins)
    return [SpinParams(float(a), float(p)) for a, p in zip(az, ap, strict=True)]


def spins_to_array(spins: Iterable[SpinParams]) -> np.ndarray:
    """Convert spin list to an ``(n, 2)`` float32 array."""
    return np.asarray([[s.az_khz, s.aperp_khz] for s in spins], dtype=np.float32)

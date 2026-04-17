from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_prominences, peak_widths

from model import (
    HyperfineParameters,
    carbon13_larmor_khz,
    khz_to_rad_per_us,
    rad_per_us_to_khz,
    simulate_cpmg_signal,
)

SQRT_2_LN_2 = float(np.sqrt(2.0 * np.log(2.0)))
EPSILON = 1e-12


@dataclass(frozen=True)
class GaussianComponent:
    amplitude: float
    center_us: float
    sigma_us: float
    fwhm_us: float
    prominence: float


@dataclass(frozen=True)
class SpinEstimate:
    peak_tau_us: float
    sigma_us: float
    amplitude: float
    prominence: float
    a_khz: float
    b_khz: float
    alternate_a_khz: float
    omega_tilde_khz: float
    used_physical_clip: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "peak_tau_us": self.peak_tau_us,
            "sigma_us": self.sigma_us,
            "amplitude": self.amplitude,
            "prominence": self.prominence,
            "a_khz": self.a_khz,
            "b_khz": self.b_khz,
            "alternate_a_khz": self.alternate_a_khz,
            "omega_tilde_khz": self.omega_tilde_khz,
            "used_physical_clip": self.used_physical_clip,
        }


@dataclass(frozen=True)
class SegmentFitResult:
    revival_index: int
    baseline: float
    slope: float
    gaussians: list[GaussianComponent]
    initial_spins: list[SpinEstimate]
    gaussian_reconstruction: np.ndarray
    gaussian_rmse: float
    refined_spins: list[HyperfineParameters] | None = None
    physical_reconstruction: np.ndarray | None = None
    physical_rmse: float | None = None
    readout_offset: float | None = None
    readout_contrast: float | None = None

    def spin_rows(self) -> list[dict[str, float | bool | None]]:
        rows: list[dict[str, float | bool | None]] = []
        for index, spin in enumerate(self.initial_spins):
            row: dict[str, float | bool | None] = {
                "spin": index + 1,
                **spin.as_dict(),
                "refined_a_khz": None,
                "refined_b_khz": None,
            }
            if self.refined_spins is not None and index < len(self.refined_spins):
                row["refined_a_khz"] = self.refined_spins[index].a_khz
                row["refined_b_khz"] = self.refined_spins[index].b_khz
            rows.append(row)
        return rows


def load_cpmg_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(Path(path), delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def cpmg_revival_period_us(b0_gauss: float) -> float:
    omega_l = khz_to_rad_per_us(carbon13_larmor_khz(b0_gauss))
    return float(np.pi / omega_l)


def extract_revival_segment(
    tau_us: Sequence[float],
    signal: Sequence[float],
    revival_index: int,
    b0_gauss: float,
    *,
    padding_us: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    tau_us = np.asarray(tau_us, dtype=float)
    signal = np.asarray(signal, dtype=float)

    period_us = cpmg_revival_period_us(b0_gauss)
    start_us = (revival_index - 1) * period_us - padding_us
    stop_us = revival_index * period_us + padding_us
    mask = (tau_us >= start_us) & (tau_us <= stop_us)
    if mask.sum() < 5:
        raise ValueError(
            f"Revival window k={revival_index} produced only {mask.sum()} points."
        )
    return tau_us[mask], signal[mask]


def _sort_by_tau(
    tau_us: Sequence[float], signal: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    tau_us = np.asarray(tau_us, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if tau_us.shape != signal.shape:
        raise ValueError("tau_us and signal must have the same shape.")
    if tau_us.ndim != 1:
        raise ValueError("tau_us and signal must be one-dimensional.")
    order = np.argsort(tau_us)
    return tau_us[order], signal[order]


def _default_prominence(dip_profile: np.ndarray) -> float:
    noise = 1.4826 * np.median(np.abs(dip_profile - np.median(dip_profile)))
    peak_scale = 0.04 * float(np.max(dip_profile))
    return float(max(3.0 * noise, peak_scale, 1e-4))


def _gaussian_signal(
    tau_us: np.ndarray,
    baseline: float,
    slope: float,
    components: Sequence[tuple[float, float, float]],
) -> np.ndarray:
    signal = gaussian_background(tau_us, baseline=baseline, slope=slope)
    for amplitude, center_us, sigma_us in components:
        exponent = -0.5 * ((tau_us - center_us) / max(sigma_us, EPSILON)) ** 2
        signal -= amplitude * np.exp(exponent)
    return signal


def gaussian_background(
    tau_us: Sequence[float],
    *,
    baseline: float,
    slope: float,
) -> np.ndarray:
    tau_us = np.asarray(tau_us, dtype=float)
    centered_tau = tau_us - float(np.mean(tau_us))
    return baseline + slope * centered_tau


def gaussian_component_curves(
    tau_us: Sequence[float],
    *,
    baseline: float,
    slope: float,
    gaussians: Sequence[GaussianComponent],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    tau_us = np.asarray(tau_us, dtype=float)
    background = gaussian_background(tau_us, baseline=baseline, slope=slope)
    dip_curves: list[np.ndarray] = []
    signal_curves: list[np.ndarray] = []

    for gaussian in gaussians:
        exponent = -0.5 * ((tau_us - gaussian.center_us) / max(gaussian.sigma_us, EPSILON)) ** 2
        dip_curve = gaussian.amplitude * np.exp(exponent)
        dip_curves.append(dip_curve)
        signal_curves.append(background - dip_curve)

    return background, dip_curves, signal_curves


def _gaussian_signal_from_params(
    params: np.ndarray, tau_us: np.ndarray, n_components: int
) -> np.ndarray:
    components: list[tuple[float, float, float]] = []
    for index in range(n_components):
        offset = 2 + 3 * index
        components.append(
            (params[offset], params[offset + 1], params[offset + 2])
        )
    return _gaussian_signal(tau_us, params[0], params[1], components)


def decompose_segment_into_gaussians(
    tau_us: Sequence[float],
    signal: Sequence[float],
    *,
    min_prominence: float | None = None,
    min_distance_us: float = 0.01,
    n_components: int | None = None,
    max_components: int | None = None,
) -> tuple[float, float, list[GaussianComponent], np.ndarray, float]:
    tau_us, signal = _sort_by_tau(tau_us, signal)
    if tau_us.size < 5:
        raise ValueError("At least five points are required to fit a segment.")
    if n_components is not None and n_components < 1:
        raise ValueError("n_components must be a positive integer.")
    if (
        n_components is not None
        and max_components is not None
        and n_components > max_components
    ):
        raise ValueError("n_components cannot be larger than max_components.")

    step_us = float(np.median(np.diff(tau_us)))
    baseline_guess = float(np.quantile(signal, 0.95))
    dip_profile = np.clip(baseline_guess - signal, 0.0, None)
    prominence = (
        _default_prominence(dip_profile)
        if min_prominence is None
        else float(min_prominence)
    )
    distance_samples = max(1, int(round(min_distance_us / max(step_us, EPSILON))))

    peaks, properties = find_peaks(
        dip_profile,
        prominence=prominence,
        distance=distance_samples,
    )
    prominences = np.asarray(properties.get("prominences", np.array([])), dtype=float)

    if n_components is not None:
        candidate_peaks, _ = find_peaks(dip_profile)
        if candidate_peaks.size == 0:
            candidate_peaks = np.array([int(np.argmax(dip_profile))], dtype=int)
        candidate_prominences = peak_prominences(dip_profile, candidate_peaks)[0]
        scores = np.maximum(candidate_prominences, dip_profile[candidate_peaks])
        order = np.argsort(scores)[::-1]
        selected = list(candidate_peaks[order[:n_components]])
        if len(selected) < n_components:
            for index in np.argsort(dip_profile)[::-1]:
                if index == 0 or index == dip_profile.size - 1:
                    continue
                if int(index) not in selected:
                    selected.append(int(index))
                if len(selected) == n_components:
                    break
        peaks = np.asarray(selected[:n_components], dtype=int)
        prominences = peak_prominences(dip_profile, peaks)[0]
    else:
        if peaks.size == 0:
            peaks = np.array([int(np.argmax(dip_profile))], dtype=int)
            prominences = np.array([float(dip_profile[peaks[0]])])

        if max_components is not None and peaks.size > max_components:
            keep = np.argsort(prominences)[-max_components:]
            peaks = peaks[keep]
            prominences = prominences[keep]

    widths = peak_widths(dip_profile, peaks, rel_height=0.5)[0]
    sigma_guess = np.maximum(widths * step_us / (2.0 * SQRT_2_LN_2), 2.0 * step_us)
    amplitude_guess = np.maximum(dip_profile[peaks], prominence)

    order = np.argsort(tau_us[peaks])
    peaks = peaks[order]
    sigma_guess = sigma_guess[order]
    amplitude_guess = amplitude_guess[order]
    prominences = prominences[order]

    initial_params = [baseline_guess, 0.0]
    lower_bounds = [float(np.min(signal)) - 0.25, -np.inf]
    upper_bounds = [float(np.max(signal)) + 0.25, np.inf]
    sigma_max = max(0.25 * float(tau_us[-1] - tau_us[0]), 5.0 * step_us)
    amplitude_max = max(1.25, 1.5 * float(np.max(dip_profile)) + 0.1)

    for amplitude, peak_index, sigma in zip(amplitude_guess, peaks, sigma_guess):
        initial_params.extend([float(amplitude), float(tau_us[peak_index]), float(sigma)])
        lower_bounds.extend([0.0, float(tau_us[0]), max(step_us, 1e-6)])
        upper_bounds.extend([amplitude_max, float(tau_us[-1]), sigma_max])

    initial_params = np.asarray(initial_params, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    n_components = peaks.size

    result = least_squares(
        lambda params: _gaussian_signal_from_params(params, tau_us, n_components) - signal,
        x0=initial_params,
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1",
        f_scale=max(prominence, 1e-3),
    )
    fitted_signal = _gaussian_signal_from_params(result.x, tau_us, n_components)
    rmse = float(np.sqrt(np.mean((fitted_signal - signal) ** 2)))

    fitted_components: list[GaussianComponent] = []
    for index in range(n_components):
        offset = 2 + 3 * index
        sigma_us = float(result.x[offset + 2])
        fitted_components.append(
            GaussianComponent(
                amplitude=float(result.x[offset]),
                center_us=float(result.x[offset + 1]),
                sigma_us=sigma_us,
                fwhm_us=2.0 * SQRT_2_LN_2 * sigma_us,
                prominence=float(prominences[index]),
            )
        )

    fitted_components.sort(key=lambda component: component.center_us)
    return (
        float(result.x[0]),
        float(result.x[1]),
        fitted_components,
        fitted_signal,
        rmse,
    )


def gaussian_to_spin_estimate(
    gaussian: GaussianComponent,
    revival_index: int,
    b0_gauss: float,
    *,
    branch: str = "weak_coupling",
) -> SpinEstimate:
    omega_l = float(khz_to_rad_per_us(carbon13_larmor_khz(b0_gauss)))
    omega_sum = float(((2 * revival_index - 1) * np.pi) / gaussian.center_us)
    omega_tilde = omega_sum - omega_l
    if omega_tilde <= 0.0:
        raise ValueError(
            "The fitted peak position is inconsistent with the supplied k and B0."
        )

    b_rad_per_us = float(
        gaussian.sigma_us * SQRT_2_LN_2 * omega_sum * omega_tilde
    )
    parallel_sq = omega_tilde**2 - b_rad_per_us**2
    used_physical_clip = parallel_sq < 0.0
    parallel_abs = float(np.sqrt(max(parallel_sq, 0.0)))

    a_plus = parallel_abs - omega_l
    a_minus = -parallel_abs - omega_l
    if branch == "weak_coupling":
        chosen_a = a_plus if abs(a_plus) <= abs(a_minus) else a_minus
    elif branch == "positive":
        chosen_a = a_plus
    elif branch == "negative":
        chosen_a = a_minus
    else:
        raise ValueError("branch must be 'weak_coupling', 'positive', or 'negative'.")

    alternate_a = a_minus if chosen_a == a_plus else a_plus
    return SpinEstimate(
        peak_tau_us=gaussian.center_us,
        sigma_us=gaussian.sigma_us,
        amplitude=gaussian.amplitude,
        prominence=gaussian.prominence,
        a_khz=float(rad_per_us_to_khz(chosen_a)),
        b_khz=float(rad_per_us_to_khz(b_rad_per_us)),
        alternate_a_khz=float(rad_per_us_to_khz(alternate_a)),
        omega_tilde_khz=float(rad_per_us_to_khz(omega_tilde)),
        used_physical_clip=used_physical_clip,
    )


def estimate_spins_from_gaussians(
    gaussians: Sequence[GaussianComponent],
    revival_index: int,
    b0_gauss: float,
    *,
    branch: str = "weak_coupling",
) -> list[SpinEstimate]:
    return [
        gaussian_to_spin_estimate(
            gaussian,
            revival_index=revival_index,
            b0_gauss=b0_gauss,
            branch=branch,
        )
        for gaussian in gaussians
    ]


def _extract_spin_parameters(
    spins: Sequence[SpinEstimate | HyperfineParameters | tuple[float, float]]
) -> list[tuple[float, float]]:
    extracted: list[tuple[float, float]] = []
    for spin in spins:
        if isinstance(spin, SpinEstimate):
            extracted.append((spin.a_khz, spin.b_khz))
        elif isinstance(spin, HyperfineParameters):
            extracted.append((spin.a_khz, spin.b_khz))
        else:
            extracted.append((float(spin[0]), float(spin[1])))
    return extracted


def refine_spin_parameters(
    tau_us: Sequence[float],
    signal: Sequence[float],
    *,
    n_pulses: int,
    b0_gauss: float,
    initial_spins: Sequence[SpinEstimate | HyperfineParameters | tuple[float, float]],
    a_margin_khz: float = 30.0,
    b_margin_khz: float = 30.0,
    fit_readout: bool = True,
) -> tuple[list[HyperfineParameters], np.ndarray, float, float, float]:
    tau_us, signal = _sort_by_tau(tau_us, signal)
    start_spins = _extract_spin_parameters(initial_spins)
    if not start_spins:
        raise ValueError("At least one initial spin estimate is required for refinement.")

    x0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    if fit_readout:
        x0.extend([0.0, 1.0])
        lower.extend([-1.0, 0.0])
        upper.extend([1.0, 2.0])

    for a_khz, b_khz in start_spins:
        a_span = max(a_margin_khz, 0.5 * abs(a_khz))
        b_upper = max(b_khz + b_margin_khz, 1.8 * max(b_khz, 1.0))
        x0.extend([a_khz, b_khz])
        lower.extend([a_khz - a_span, 0.0])
        upper.extend([a_khz + a_span, b_upper])

    x0_array = np.asarray(x0, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    larmor_khz = carbon13_larmor_khz(b0_gauss)
    first_spin_index = 2 if fit_readout else 0

    def residuals(params: np.ndarray) -> np.ndarray:
        offset = float(params[0]) if fit_readout else 0.0
        contrast = float(params[1]) if fit_readout else 1.0

        spins: list[tuple[float, float]] = []
        for index in range(first_spin_index, params.size, 2):
            spins.append((float(params[index]), float(params[index + 1])))
        model_signal = offset + contrast * simulate_cpmg_signal(
            tau_us=tau_us,
            n_pulses=n_pulses,
            spins=spins,
            larmor_khz=larmor_khz,
        )
        return model_signal - signal

    result = least_squares(
        residuals,
        x0=x0_array,
        bounds=(lower_array, upper_array),
        loss="soft_l1",
        f_scale=max(0.01, 0.1 * float(np.std(signal))),
    )
    optimized_params = result.x
    offset = float(optimized_params[0]) if fit_readout else 0.0
    contrast = float(optimized_params[1]) if fit_readout else 1.0

    refined_spins: list[HyperfineParameters] = []
    for index in range(first_spin_index, optimized_params.size, 2):
        refined_spins.append(
            HyperfineParameters(
                a_khz=float(optimized_params[index]),
                b_khz=float(optimized_params[index + 1]),
            )
        )

    fitted_signal = offset + contrast * simulate_cpmg_signal(
        tau_us=tau_us,
        n_pulses=n_pulses,
        spins=refined_spins,
        larmor_khz=larmor_khz,
    )
    rmse = float(np.sqrt(np.mean((fitted_signal - signal) ** 2)))
    return refined_spins, fitted_signal, rmse, offset, contrast


def fit_spins_from_segment(
    tau_us: Sequence[float],
    signal: Sequence[float],
    *,
    revival_index: int,
    b0_gauss: float,
    n_pulses: int | None = None,
    min_prominence: float | None = None,
    min_distance_us: float = 0.01,
    n_components: int | None = None,
    max_components: int | None = None,
    branch: str = "weak_coupling",
    refine: bool = False,
    fit_readout: bool = True,
) -> SegmentFitResult:
    tau_us, signal = _sort_by_tau(tau_us, signal)
    baseline, slope, gaussians, gaussian_signal, gaussian_rmse = (
        decompose_segment_into_gaussians(
            tau_us=tau_us,
            signal=signal,
            min_prominence=min_prominence,
            min_distance_us=min_distance_us,
            n_components=n_components,
            max_components=max_components,
        )
    )
    initial_spins = estimate_spins_from_gaussians(
        gaussians,
        revival_index=revival_index,
        b0_gauss=b0_gauss,
        branch=branch,
    )

    refined_spins: list[HyperfineParameters] | None = None
    physical_signal: np.ndarray | None = None
    physical_rmse: float | None = None
    readout_offset: float | None = None
    readout_contrast: float | None = None

    if refine:
        if n_pulses is None:
            raise ValueError("n_pulses is required when refine=True.")
        (
            refined_spins,
            physical_signal,
            physical_rmse,
            readout_offset,
            readout_contrast,
        ) = refine_spin_parameters(
            tau_us=tau_us,
            signal=signal,
            n_pulses=n_pulses,
            b0_gauss=b0_gauss,
            initial_spins=initial_spins,
            fit_readout=fit_readout,
        )

    return SegmentFitResult(
        revival_index=revival_index,
        baseline=baseline,
        slope=slope,
        gaussians=gaussians,
        initial_spins=initial_spins,
        gaussian_reconstruction=gaussian_signal,
        gaussian_rmse=gaussian_rmse,
        refined_spins=refined_spins,
        physical_reconstruction=physical_signal,
        physical_rmse=physical_rmse,
        readout_offset=readout_offset,
        readout_contrast=readout_contrast,
    )

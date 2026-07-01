from __future__ import annotations

import numpy as np
import pytest

from sali.config import PhysicsConfig
from sali.physics import (
    Couplings,
    _apply_shot_noise,
    _ideal_signal,
    generate_sample_signals,
    simulate_cpmg_signal,
    tau_grid_us,
)


def test_tau_grid_uses_requested_bounds() -> None:
    tau = tau_grid_us(6.0, 50.0, 1000)
    assert tau.shape == (1000,)
    assert tau[0] == np.float32(6.0)
    assert tau[-1] == np.float32(50.0)


def test_simulate_cpmg_signal_is_finite_and_probability_like(rng: np.random.Generator) -> None:
    cfg = PhysicsConfig(bz_tesla=0.0056, add_shot_noise=False)
    nuclei = Couplings(
        az_khz=np.array([-20.0, 35.0], dtype=np.float32),
        aperp_khz=np.array([12.0, 50.0], dtype=np.float32),
    )
    tau = tau_grid_us(6.0, 50.0, 1000)
    signal = simulate_cpmg_signal(nuclei, tau, n_pulses=32, cfg=cfg, rng=rng)
    assert signal.shape == (1000,)
    assert np.isfinite(signal).all()
    assert signal.min() >= 0.0
    assert signal.max() <= 1.0


def test_generate_sample_signals_returns_two_inputs(rng: np.random.Generator) -> None:
    cfg = PhysicsConfig(bz_tesla=0.056, add_shot_noise=False)
    nuclei = Couplings(
        az_khz=np.array([10.0], dtype=np.float32),
        aperp_khz=np.array([25.0], dtype=np.float32),
    )
    signals = generate_sample_signals(nuclei, cfg, rng)
    assert signals.shape == (2, 1000)
    assert np.isfinite(signals).all()


def test_zero_nuclei_without_decay_or_noise_returns_ones(rng: np.random.Generator) -> None:
    cfg = PhysicsConfig(bz_tesla=0.0056, add_decoherence=False, add_shot_noise=False)
    nuclei = Couplings(
        az_khz=np.array([], dtype=np.float32),
        aperp_khz=np.array([], dtype=np.float32),
    )
    tau = tau_grid_us(6.0, 50.0, 1000)
    signal = simulate_cpmg_signal(nuclei, tau, n_pulses=32, cfg=cfg, rng=rng)
    np.testing.assert_array_equal(signal, np.ones(1000, dtype=np.float32))


def test_shot_noise_is_reproducible_with_same_seed() -> None:
    cfg = PhysicsConfig(
        bz_tesla=0.0056,
        add_decoherence=False,
        add_shot_noise=True,
        measurements=250,
    )
    nuclei = Couplings(
        az_khz=np.array([-20.0, 35.0], dtype=np.float32),
        aperp_khz=np.array([12.0, 50.0], dtype=np.float32),
    )
    tau = tau_grid_us(6.0, 50.0, 1000)
    signal_a = simulate_cpmg_signal(nuclei, tau, 32, cfg, np.random.default_rng(2026))
    signal_b = simulate_cpmg_signal(nuclei, tau, 32, cfg, np.random.default_rng(2026))
    np.testing.assert_array_equal(signal_a, signal_b)


def test_couplings_copy_coerce_and_freeze_input_arrays() -> None:
    az = np.array([1.5, -2.25], dtype=np.float64)
    aperp = np.array([3, 4], dtype=np.int64)

    nuclei = Couplings(az_khz=az, aperp_khz=aperp)
    az[0] = 99.0
    aperp[1] = 88

    assert nuclei.az_khz.dtype == np.float32
    assert nuclei.aperp_khz.dtype == np.float32
    assert not nuclei.az_khz.flags.writeable
    assert not nuclei.aperp_khz.flags.writeable
    np.testing.assert_array_equal(nuclei.az_khz, np.array([1.5, -2.25], dtype=np.float32))
    np.testing.assert_array_equal(nuclei.aperp_khz, np.array([3.0, 4.0], dtype=np.float32))
    with pytest.raises(ValueError):
        nuclei.az_khz[0] = np.float32(7.0)


def test_couplings_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        Couplings(
            az_khz=np.array([np.nan], dtype=np.float32),
            aperp_khz=np.array([1.0], dtype=np.float32),
        )
    with pytest.raises(ValueError, match="finite"):
        Couplings(
            az_khz=np.array([1.0], dtype=np.float32),
            aperp_khz=np.array([np.inf], dtype=np.float32),
        )


def test_known_coupling_matches_reference_value(rng: np.random.Generator) -> None:
    cfg = PhysicsConfig(bz_tesla=0.0056, add_decoherence=False, add_shot_noise=False)
    nuclei = Couplings(
        az_khz=np.array([10.0], dtype=np.float32),
        aperp_khz=np.array([25.0], dtype=np.float32),
    )
    tau = tau_grid_us(6.0, 50.0, 1000)
    signal = simulate_cpmg_signal(nuclei, tau, n_pulses=32, cfg=cfg, rng=rng)
    assert signal[123] == pytest.approx(0.8777403831481934, abs=1e-7)


def test_ideal_signal_rejects_nonfinite_values_before_clipping() -> None:
    cfg = PhysicsConfig(bz_tesla=0.0, add_decoherence=False, add_shot_noise=False)
    nuclei = Couplings(
        az_khz=np.array([0.0], dtype=np.float32),
        aperp_khz=np.array([0.0], dtype=np.float32),
    )
    tau = tau_grid_us(6.0, 50.0, 1000)
    with pytest.raises(FloatingPointError, match="ideal signal"):
        _ideal_signal(nuclei, tau, n_pulses=32, cfg=cfg)


def test_shot_noise_rejects_materially_invalid_probabilities(rng: np.random.Generator) -> None:
    cfg = PhysicsConfig(bz_tesla=0.0056, add_shot_noise=True)
    with pytest.raises(FloatingPointError, match="probability"):
        _apply_shot_noise(np.array([0.25, 1.00001], dtype=np.float32), cfg, rng)
    with pytest.raises(FloatingPointError, match="finite"):
        _apply_shot_noise(np.array([0.25, np.nan], dtype=np.float32), cfg, rng)


def test_shot_noise_clips_tiny_probability_roundoff(rng: np.random.Generator) -> None:
    cfg = PhysicsConfig(bz_tesla=0.0056, add_shot_noise=False)
    signal = np.array([-5e-7, 0.5, 1.0000005], dtype=np.float32)
    clipped = _apply_shot_noise(signal, cfg, rng)
    np.testing.assert_array_equal(clipped, np.array([0.0, 0.5, 1.0], dtype=np.float32))

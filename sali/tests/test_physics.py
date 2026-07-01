from __future__ import annotations

import numpy as np

from sali.config import PhysicsConfig
from sali.physics import Couplings, generate_sample_signals, simulate_cpmg_signal, tau_grid_us


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

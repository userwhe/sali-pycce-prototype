import numpy as np

from sali_pycce.physics import AnalyticCPMGSimulator, SpinParams


def test_signal_shape_and_bounds():
    sim = AnalyticCPMGSimulator(signal_points=64, shots=None)
    out = sim.sample([SpinParams(az_khz=20.0, aperp_khz=30.0)], rng=np.random.default_rng(0), noisy=False)
    assert out["signals"].shape == (2, 64)
    assert out["taus_us"].shape == (2, 64)
    assert np.all(out["signals"] >= 0.0)
    assert np.all(out["signals"] <= 1.0)


def test_empty_spin_signal_is_flat_with_decay():
    sim = AnalyticCPMGSimulator(signal_points=64, shots=None, t2_us=None)
    tau = sim.tau_grid(0)
    px = sim.signal([], 32, tau)
    assert np.allclose(px, 1.0)

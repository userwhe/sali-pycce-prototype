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


def test_t2_decoherence_uses_configurable_default_scale():
    sim = AnalyticCPMGSimulator(
        b_gauss=525.0,
        signal_points=5,
        tau_ranges_us=((0.0, 40.0), (0.0, 40.0)),
        t2_us=800.0,
        t2_stretch=1.0,
        shots=None,
    )
    tau = sim.tau_grid(0)
    px = sim.signal([], 32, tau)

    expected = 0.5 + 0.5 * np.exp(-tau / 800.0)
    np.testing.assert_allclose(px, expected, rtol=1e-7, atol=1e-7)

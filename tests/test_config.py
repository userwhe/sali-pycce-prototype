import numpy as np

from sali_pycce.config import COLAB_MEDIUM_CONFIG, RESEARCH_CONFIG, SMOKE_CONFIG


def test_research_config_matches_approved_defaults():
    assert RESEARCH_CONFIG.b_gauss == 525.0
    assert RESEARCH_CONFIG.pulses == (32, 256)
    assert RESEARCH_CONFIG.tau_ranges_us == ((0.0, 40.0), (0.0, 40.0))
    assert RESEARCH_CONFIG.signal_points == 4000
    assert RESEARCH_CONFIG.az_range == (-250.0, 250.0)
    assert RESEARCH_CONFIG.aperp_range == (2.0, 250.0)
    assert RESEARCH_CONFIG.heatmap_shape == (128, 256)
    assert RESEARCH_CONFIG.t2_us == 800.0
    assert RESEARCH_CONFIG.train_samples == 100_000
    assert RESEARCH_CONFIG.val_samples == 10_000
    assert RESEARCH_CONFIG.test_samples == 10_000


def test_presets_build_consistent_simulator_and_heatmap_spec():
    sim = RESEARCH_CONFIG.build_simulator(shots=123)
    spec = RESEARCH_CONFIG.build_heatmap_spec()

    assert sim.b_gauss == 525.0
    assert sim.pulses == (32, 256)
    assert sim.signal_points == 4000
    assert sim.shots == 123
    assert sim.t2_us == 800.0
    assert spec.height == 128
    assert spec.width == 256
    assert spec.az_range == (-250.0, 250.0)
    assert spec.aperp_range == (2.0, 250.0)


def test_smoke_and_colab_medium_have_smaller_counts_than_research():
    assert SMOKE_CONFIG.train_samples < COLAB_MEDIUM_CONFIG.train_samples
    assert COLAB_MEDIUM_CONFIG.train_samples < RESEARCH_CONFIG.train_samples
    assert SMOKE_CONFIG.val_samples < COLAB_MEDIUM_CONFIG.val_samples
    assert COLAB_MEDIUM_CONFIG.val_samples < RESEARCH_CONFIG.val_samples
    assert SMOKE_CONFIG.test_samples < COLAB_MEDIUM_CONFIG.test_samples
    assert COLAB_MEDIUM_CONFIG.test_samples < RESEARCH_CONFIG.test_samples


def test_research_tau_grid_is_zero_to_forty_us():
    sim = RESEARCH_CONFIG.build_simulator(shots=None)
    tau = sim.tau_grid(0)

    assert tau.shape == (4000,)
    np.testing.assert_allclose(tau[0], 0.0)
    np.testing.assert_allclose(tau[-1], 40.0)

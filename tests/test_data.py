import numpy as np

from sali_pycce.config import SMOKE_CONFIG
from sali_pycce.data import SyntheticSALIDataset


def test_dataset_returns_taus_and_configured_shapes():
    sim = SMOKE_CONFIG.build_simulator(shots=None)
    spec = SMOKE_CONFIG.build_heatmap_spec()
    ds = SyntheticSALIDataset(
        2,
        simulator=sim,
        heatmap_spec=spec,
        min_spins=SMOKE_CONFIG.min_spins,
        max_spins=SMOKE_CONFIG.max_spins,
        az_range=SMOKE_CONFIG.az_range,
        aperp_range=SMOKE_CONFIG.aperp_range,
        seed=7,
        noisy=False,
    )

    item = ds[0]

    assert item["signals"].shape == (2, SMOKE_CONFIG.signal_points)
    assert item["taus_us"].shape == (2, SMOKE_CONFIG.signal_points)
    assert item["heatmap"].shape == (1, *SMOKE_CONFIG.heatmap_shape)
    assert item["spins"].shape == (SMOKE_CONFIG.max_spins, 2)
    assert int(item["n_spins"]) >= SMOKE_CONFIG.min_spins
    assert np.isfinite(item["signals"].numpy()).all()

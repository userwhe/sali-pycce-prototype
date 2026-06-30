import numpy as np

from sali_pycce.pycce_benchmark import (
    PyCCEBenchmarkConfig,
    bath_arrays_to_spin_table,
    detectable_spin_mask,
)


def test_detectable_spin_mask_uses_range_and_depth():
    spin_table = np.asarray(
        [
            [0.0, 50.0, 0.10],
            [300.0, 50.0, 0.20],
            [0.0, 1.0, 0.20],
            [0.0, 50.0, 0.001],
        ],
        dtype=float,
    )
    config = PyCCEBenchmarkConfig(
        az_range=(-250.0, 250.0),
        aperp_range=(2.0, 250.0),
        min_signal_depth=0.01,
    )

    mask = detectable_spin_mask(spin_table, config)

    assert mask.tolist() == [True, False, False, False]


def test_bath_arrays_to_spin_table_uses_projected_parameters():
    xyz = np.asarray([[1.0, 2.0, 3.0]])
    tensors = np.asarray([[[1.0, 0.0, 3.0], [0.0, 2.0, 4.0], [3.0, 4.0, 5.0]]])
    names = np.asarray(["13C"])

    table = bath_arrays_to_spin_table(names, xyz, tensors, tensor_units="kHz")

    assert table.shape == (1, 9)
    assert table[0, 0] == 0.0
    assert table[0, 1] == 1.0
    assert table[0, 2] == 2.0
    assert table[0, 3] == 3.0
    assert table[0, 4] == 5.0
    assert table[0, 5] == 5.0

import numpy as np
import pytest

from sali_pycce.tensor import project_hyperfine_tensor, project_hyperfine_tensors


def test_project_hyperfine_tensor_khz_with_z_axis():
    tensor = np.asarray(
        [
            [1.0, 0.0, 3.0],
            [0.0, 2.0, 4.0],
            [3.0, 4.0, 5.0],
        ]
    )

    az, aperp = project_hyperfine_tensor(tensor, axis=(0.0, 0.0, 1.0), units="kHz")

    assert az == 5.0
    np.testing.assert_allclose(aperp, 5.0)


def test_project_hyperfine_tensor_hz_to_khz():
    tensor_hz = np.diag([0.0, 0.0, 12_000.0])

    az, aperp = project_hyperfine_tensor(tensor_hz, units="Hz")

    assert az == 12.0
    assert aperp == 0.0


def test_project_hyperfine_tensors_batch_shape():
    tensors = np.asarray([np.diag([0.0, 0.0, 3.0]), np.diag([0.0, 0.0, -7.0])])

    out = project_hyperfine_tensors(tensors, units="kHz")

    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[:, 0], [3.0, -7.0])
    np.testing.assert_allclose(out[:, 1], [0.0, 0.0])


def test_project_hyperfine_tensor_rejects_unknown_units():
    with pytest.raises(ValueError, match="Unsupported hyperfine tensor units"):
        project_hyperfine_tensor(np.eye(3), units="cycles")

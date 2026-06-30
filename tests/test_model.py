import pytest
import torch

from sali_pycce.model import SALINet


def test_model_forward_shape_smoke_grid():
    model = SALINet(output_shape=(32, 64))
    x = torch.rand(3, 2, 128)
    y = model(x)
    assert y.shape == (3, 1, 32, 64)
    assert torch.isfinite(y).all()


def test_model_forward_shape_research_grid():
    model = SALINet(output_shape=(128, 256))
    x = torch.rand(2, 2, 4000)
    y = model(x)
    assert y.shape == (2, 1, 128, 256)
    assert torch.isfinite(y).all()


def test_model_rejects_output_shape_not_divisible_by_eight():
    with pytest.raises(ValueError, match="divisible by 8"):
        SALINet(output_shape=(130, 256))

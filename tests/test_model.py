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


def test_model_rejects_zero_output_shape_height():
    with pytest.raises(ValueError, match="positive"):
        SALINet(output_shape=(0, 64))


def test_model_rejects_negative_output_shape_height():
    with pytest.raises(ValueError, match="positive"):
        SALINet(output_shape=(-8, 64))


def test_model_rejects_zero_n_inputs():
    with pytest.raises(ValueError, match="positive"):
        SALINet(n_inputs=0)


def test_model_rejects_zero_pooled_len():
    with pytest.raises(ValueError, match="positive"):
        SALINet(pooled_len=0)


def test_model_rejects_zero_decoder_channels():
    with pytest.raises(ValueError, match="positive"):
        SALINet(decoder_channels=0)

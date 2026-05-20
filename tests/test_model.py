import torch

from sali_pycce.model import SALINet


def test_model_forward_shape():
    model = SALINet()
    x = torch.rand(3, 2, 128)
    y = model(x)
    assert y.shape == (3, 1, 32, 64)
    assert torch.isfinite(y).all()

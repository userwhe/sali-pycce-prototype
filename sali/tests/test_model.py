from __future__ import annotations

import torch

from sali.config import ModelConfig
from sali.model import SaliNet


def test_sali_net_forward_shape() -> None:
    model = SaliNet(ModelConfig())
    signal32 = torch.zeros((2, 1, 1000), dtype=torch.float32)
    signal256 = torch.zeros((2, 1, 1000), dtype=torch.float32)
    output = model(signal32, signal256)
    assert tuple(output.shape) == (2, 1, 204, 104)
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0

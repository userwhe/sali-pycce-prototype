from __future__ import annotations

import pytest
import torch

from sali.config import ModelConfig
from sali.model import SaliNet


def _signals(batch_size: int = 2, length: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    signal32 = torch.zeros((batch_size, 1, length), dtype=torch.float32)
    signal256 = torch.zeros((batch_size, 1, length), dtype=torch.float32)
    return signal32, signal256


def test_sali_net_forward_shape() -> None:
    model = SaliNet(ModelConfig())
    model.eval()
    signal32, signal256 = _signals()

    with torch.inference_mode():
        output = model(signal32, signal256)

    assert tuple(output.shape) == (2, 1, 204, 104)
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("output_height", 205, "output_height.*dense_height"),
        ("output_width", 105, "output_width.*dense_width"),
    ],
)
def test_sali_net_rejects_output_dimensions_that_do_not_double_dense(
    field: str, value: int, message: str
) -> None:
    cfg = ModelConfig()
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match=message):
        SaliNet(cfg)


@pytest.mark.parametrize(
    ("name", "bad_signal", "message"),
    [
        ("signal32", torch.zeros((2, 1000), dtype=torch.float32), "signal32.*rank 3"),
        ("signal256", torch.zeros((2, 1000), dtype=torch.float32), "signal256.*rank 3"),
        ("signal32", torch.zeros((2, 2, 1000), dtype=torch.float32), "signal32.*channel count 1"),
        ("signal256", torch.zeros((2, 2, 1000), dtype=torch.float32), "signal256.*channel count 1"),
        ("signal32", torch.zeros((2, 1, 999), dtype=torch.float32), "signal32.*length 1000"),
        ("signal256", torch.zeros((2, 1, 999), dtype=torch.float32), "signal256.*length 1000"),
    ],
)
def test_sali_net_rejects_invalid_signal_shapes(
    name: str, bad_signal: torch.Tensor, message: str
) -> None:
    model = SaliNet(ModelConfig())
    signal32, signal256 = _signals()
    if name == "signal32":
        signal32 = bad_signal
    else:
        signal256 = bad_signal

    with pytest.raises(ValueError, match=message):
        model(signal32, signal256)


def test_sali_net_rejects_mismatched_batch_sizes() -> None:
    model = SaliNet(ModelConfig())
    signal32, _ = _signals(batch_size=2)
    _, signal256 = _signals(batch_size=3)

    with pytest.raises(ValueError, match="batch size"):
        model(signal32, signal256)


def test_sali_net_branches_have_independent_influence() -> None:
    torch.manual_seed(0)
    model = SaliNet(ModelConfig())
    model.eval()
    signal32 = torch.full((2, 1, 1000), 0.25, dtype=torch.float32)
    signal256 = torch.full((2, 1, 1000), -0.25, dtype=torch.float32)

    with torch.inference_mode():
        baseline = model(signal32, signal256)
        changed32 = model(signal32 + 0.5, signal256)
        changed256 = model(signal32, signal256 + 0.5)

    assert model.branch32 is not model.branch256
    assert model.branch32.layers[0].weight is not model.branch256.layers[0].weight
    assert not torch.allclose(changed32, baseline)
    assert not torch.allclose(changed256, baseline)


def test_sali_net_default_parameter_count_regression() -> None:
    model = SaliNet(ModelConfig())

    assert sum(p.numel() for p in model.parameters()) == 98_966_237


def test_sali_net_eval_mode_accepts_singleton_batch() -> None:
    # Task 9 training should avoid singleton training batches for BatchNorm.
    model = SaliNet(ModelConfig())
    model.eval()
    signal32, signal256 = _signals(batch_size=1)

    with torch.inference_mode():
        output = model(signal32, signal256)

    assert tuple(output.shape) == (1, 1, 204, 104)
